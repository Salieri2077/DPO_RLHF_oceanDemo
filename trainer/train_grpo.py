import argparse
import math
import os
import re
import shutil
import sys
import time
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import RLAIFDataset
from model.model_minimind import MiniMindConfig
from trainer.rollout_engine import completion_log_probs, rollout
from trainer.trainer_utils import (
    Logger, MetricWindow, SkipBatchSampler, SiliconFlowRewardModel, experiment_config,
    get_lr, init_distributed_mode, init_model, init_swanlab, is_main_process, lm_checkpoint,
    reduce_sums, safe_swanlab_log, setup_seed,
)


def repetition_penalty(text, n=3, cap=0.5):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) / max(len(grams), 1))


def group_advantages(rewards, group_size):
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(1, keepdim=True)
    stds = grouped.std(1, unbiased=False, keepdim=True)
    return ((grouped - means) / (stds + 1e-4)).flatten(), stds.flatten()


def grpo_objective(new_logps, old_logps, ref_logps, mask, advantages, beta=0.1, epsilon=0.2):
    ratio = torch.exp(new_logps - old_logps)
    advantage = advantages.unsqueeze(1)
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    log_ratio = ref_logps - new_logps
    kl = torch.exp(log_ratio) - log_ratio - 1
    denominator = mask.sum().clamp_min(1)
    policy_loss = (policy * mask).sum() / denominator
    mean_kl = (kl * mask).sum() / denominator
    return {
        "policy_loss": policy_loss,
        "kl": mean_kl,
        "grpo_loss": policy_loss + beta * mean_kl,
        "clip_fraction": (((ratio - 1).abs() > epsilon) * mask).sum() / denominator,
    }


def score_responses(judge, questions, references, responses, group_size, device):
    scores, latencies = [], []
    for index, (question, reference) in enumerate(zip(questions, references)):
        group = responses[index * group_size:(index + 1) * group_size]
        started = time.time()
        scores.extend(judge.score_group(question, reference, group))
        latencies.append(time.time() - started)
    rewards = torch.tensor(scores, dtype=torch.float32, device=device)
    for index, response in enumerate(responses):
        rewards[index] += (0.1 if 20 <= len(response.strip()) <= 800 else -0.1) - repetition_penalty(response)
    return rewards, sum(latencies), len(latencies)


def parse_args():
    parser = argparse.ArgumentParser(description="OceanHeart GRPO with an online ocean-domain judge")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--save_weight", default="ocean_grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--thinking_ratio", type=float, default=0.0)
    parser.add_argument("--data_path", default="../data/processed/ocean_grpo_train.jsonl")
    parser.add_argument("--val_data_path", default="../data/processed/ocean_grpo_val.jsonl")
    parser.add_argument("--from_weight", default="ocean_sft_replay")
    parser.add_argument("--from_resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--reward_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true")
    parser.add_argument("--swanlab_mode", "--wandb_mode", dest="swanlab_mode", choices=["cloud", "local", "offline", "disabled"], default="cloud")
    parser.add_argument("--swanlab_logdir", "--wandb_logdir", dest="swanlab_logdir", default=None)
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", default="OceanHeart-GRPO")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_generations < 2:
        raise SystemExit("--num_generations must be at least 2 for group-relative advantages")
    if args.max_seq_len < 1 or args.max_gen_len < 1 or args.accumulation_steps < 1:
        raise SystemExit("sequence lengths and accumulation steps must be positive")
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is required; set it in the shell before starting GRPO")
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(42 + rank)
    os.makedirs(args.save_dir, exist_ok=True)
    config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe),
    )
    checkpoint = lm_checkpoint(config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume else None
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)
    tracker = None
    if args.use_swanlab and is_main_process():
        import swanlab
        tracker = swanlab
        init_swanlab(
            tracker, checkpoint, args.swanlab_mode,
            project=args.swanlab_project, name=args.run_name or "ocean-grpo",
            logdir=args.swanlab_logdir, config=experiment_config(args),
        )
    model, tokenizer = init_model(config, args.from_weight, save_dir=args.save_dir, device=args.device)
    reference, _ = init_model(config, args.from_weight, save_dir=args.save_dir, device=args.device)
    reference.eval().requires_grad_(False)
    judge = SiliconFlowRewardModel(api_key, args.reward_model)
    train_ds = RLAIFDataset(args.data_path, tokenizer, args.max_seq_len, args.thinking_ratio)
    val_ds = RLAIFDataset(args.val_data_path, tokenizer, args.max_seq_len, 0.0) if args.val_data_path else None
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device_type == "cuda" and args.dtype == "float16")
    start_epoch = start_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_step = checkpoint["epoch"], checkpoint.get("step", 0)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    def save(epoch, step):
        if is_main_process():
            lm_checkpoint(config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=tracker, save_dir="../checkpoints")
            source = f"../checkpoints/{args.save_weight}_{config.hidden_size}{'_moe' if config.use_moe else ''}.pth"
            target = f"{args.save_dir}/{args.save_weight}_{config.hidden_size}{'_moe' if config.use_moe else ''}.pth"
            temp = target + ".tmp"
            shutil.copyfile(source, temp)
            os.replace(temp, target)

    api_calls_total = 0

    def validate(update):
        global tracker, api_calls_total
        if dist.is_initialized():
            dist.barrier()
        if val_ds and is_main_process():
            raw = model.module if isinstance(model, DistributedDataParallel) else model
            loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            rewards = []
            for index, batch in enumerate(loader):
                if args.eval_batches and index >= args.eval_batches:
                    break
                inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(args.device)
                result = rollout(raw, tokenizer, inputs["input_ids"][:, -args.max_seq_len:], inputs["attention_mask"][:, -args.max_seq_len:], args.num_generations, args.max_gen_len, args.temperature)
                score, _, calls = score_responses(judge, batch["question"], batch["answer"], result.completions, args.num_generations, args.device)
                api_calls_total += calls
                rewards.append(score.mean().item())
            if rewards:
                tracker = safe_swanlab_log(tracker, {"val/reward_mean": sum(rewards) / len(rewards)}, step=update)
        if dist.is_initialized():
            dist.barrier()

    if val_ds and not checkpoint:
        validate(0)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        sampler and sampler.set_epoch(epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=device_type == "cuda")
        iterations = len(loader) + skip
        window = MetricWindow()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=skip + 1):
            inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(args.device)
            prompt_ids = inputs["input_ids"][:, -args.max_seq_len:]
            prompt_mask = inputs["attention_mask"][:, -args.max_seq_len:]
            with autocast:
                result = rollout(model, tokenizer, prompt_ids, prompt_mask, args.num_generations, args.max_gen_len, args.temperature)
            rewards, api_latency, api_calls = score_responses(judge, batch["question"], batch["answer"], result.completions, args.num_generations, args.device)
            api_calls_total += api_calls
            advantages, group_stds = group_advantages(rewards, args.num_generations)
            full_mask = result.output_ids.ne(tokenizer.pad_token_id).long()
            # Do not reuse generate's inference-mode autocast cache for training.
            with autocast:
                output = model(
                    result.output_ids, attention_mask=full_mask,
                    logits_to_keep=result.completion_ids.size(1) + 1,
                )
                completion_logits = output.logits[:, :-1]
                new_logps = F.log_softmax(completion_logits, dim=-1).gather(2, result.completion_ids.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    ref_logps = completion_log_probs(reference, result.output_ids, result.completion_ids, full_mask)
                metrics = grpo_objective(new_logps, result.old_log_probs, ref_logps, result.completion_mask, advantages, args.beta, args.epsilon)
                loss = (metrics["grpo_loss"] + output.aux_loss) / args.accumulation_steps
            scaler.scale(loss).backward()
            for name, value in metrics.items():
                window.add(name, value)
            window.add("aux_loss", output.aux_loss)
            window.add("reward_mean", rewards.mean())
            window.add("reward_std", rewards.std(unbiased=False))
            window.add("group_std", group_stds.mean())
            window.add("degenerate_group_ratio", (group_stds < 1e-6).float().mean())
            window.add("response_length", result.completion_mask.sum(1).float().mean())
            window.add("api_latency", api_latency / max(api_calls, 1))
            final_micro = step == iterations
            if step % args.accumulation_steps and not final_micro:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update = math.ceil((epoch * iterations + step) / args.accumulation_steps)
            lr = get_lr(update, max(math.ceil(args.epochs * iterations / args.accumulation_steps), 1), args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window.add("grad_norm", grad_norm)
            if update % args.log_interval == 0 or final_micro:
                summary = window.means(args.device)
                summary["loss"] = summary["grpo_loss"] + summary["aux_loss"]
                summary["learning_rate"] = lr
                summary["api_call_count"] = reduce_sums([api_calls_total], args.device)[0]
                tracker = safe_swanlab_log(tracker, {f"train/{key}": value for key, value in summary.items()}, step=update)
                Logger(f"Epoch [{epoch + 1}/{args.epochs}] step={step}/{iterations} update={update} loss={summary['loss']:.4f} reward={summary['reward_mean']:.3f}")
            if val_ds and update % args.eval_interval == 0:
                validate(update)
            if update % args.save_interval == 0:
                save(epoch, step)
            if args.max_steps and update >= args.max_steps:
                save(epoch, step)
                stop = True
                break
        start_step = 0
        if stop:
            break
        save(epoch, iterations)
    if tracker and hasattr(tracker, "finish"):
        tracker.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
