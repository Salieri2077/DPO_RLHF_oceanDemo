"""Kimi-style on-policy distillation for OceanHeart."""

import argparse
import hashlib
import math
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import RLAIFDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.rollout_engine import completion_log_probs, rollout
from trainer.trainer_utils import (
    Logger, MetricWindow, SkipBatchSampler, experiment_config, get_lr,
    init_distributed_mode, init_model, init_swanlab, is_main_process, lm_checkpoint,
    reduce_sums, safe_swanlab_log, setup_seed,
)


OPD_METRIC_NAMES = (
    "policy_loss", "opd_reward", "reverse_kl", "student_logp", "teacher_logp",
    "reward_clip_fraction", "ratio_clip_fraction",
)


def opd_objective(new_logps, old_logps, teacher_logps, mask, reward_clip=5.0, epsilon=0.2):
    raw_reward = (teacher_logps - old_logps).detach()
    advantage = raw_reward.clamp(-reward_clip, reward_clip)
    ratio = torch.exp(new_logps - old_logps)
    policy = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - epsilon, 1 + epsilon) * advantage,
    )
    denominator = mask.sum().clamp_min(1)

    def mean(value):
        return (value * mask).sum() / denominator

    return {
        "policy_loss": mean(policy),
        "opd_reward": mean(advantage),
        "reverse_kl": mean(old_logps - teacher_logps),
        "student_logp": mean(old_logps),
        "teacher_logp": mean(teacher_logps),
        "reward_clip_fraction": mean(raw_reward.abs().gt(reward_clip)),
        "ratio_clip_fraction": mean((ratio - 1).abs().gt(epsilon)),
    }


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher(config, path, device, dtype):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "model" in state:
        raise ValueError(f"teacher must be a final model state_dict, not a resume checkpoint: {path}")
    teacher = MiniMindForCausalLM(config)
    try:
        teacher.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"teacher architecture does not match the student: {path}\n{error}") from error
    del state
    teacher = teacher.to(device=device, dtype=dtype if str(device).startswith("cuda") else torch.float32)
    return teacher.eval().requires_grad_(False)


@torch.no_grad()
def routed_teacher_log_probs(teachers, domains, output_ids, completion_ids, attention_mask):
    result = torch.empty(
        completion_ids.shape, dtype=torch.float32, device=completion_ids.device
    )
    for domain in sorted(set(domains)):
        if domain not in teachers:
            raise ValueError(f"no teacher configured for domain {domain!r}")
        indices = torch.tensor(
            [index for index, value in enumerate(domains) if value == domain],
            device=output_ids.device,
        )
        logps = completion_log_probs(
            teachers[domain], output_ids.index_select(0, indices),
            completion_ids.index_select(0, indices), attention_mask.index_select(0, indices),
        )
        result.index_copy_(0, indices, logps.float())
    return result


def build_parser(multi_domain=False):
    name = "MOPD" if multi_domain else "OPD"
    parser = argparse.ArgumentParser(description=f"OceanHeart {name} training")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--save_weight", default="ocean_mopd" if multi_domain else "ocean_opd")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1, help="optimizer steps")
    parser.add_argument("--save_interval", type=int, default=50, help="optimizer steps")
    parser.add_argument("--eval_interval", type=int, default=10, help="optimizer steps")
    parser.add_argument("--eval_batches", type=int, default=50, help="0 evaluates the whole validation set")
    parser.add_argument("--max_steps", type=int, default=0, help="maximum optimizer steps; 0 is unlimited")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--thinking_ratio", type=float, default=0.0)
    default_data = "../data/processed/ocean_mopd_train.jsonl" if multi_domain else "../data/processed/ocean_opd_train.jsonl"
    default_val = "../data/processed/ocean_mopd_val.jsonl" if multi_domain else "../data/processed/ocean_opd_val.jsonl"
    parser.add_argument("--data_path", default=default_data)
    parser.add_argument("--val_data_path", default=default_val)
    parser.add_argument("--from_weight", default="ocean_sft_replay")
    parser.add_argument("--from_resume", type=int, choices=[0, 1], default=0)
    if multi_domain:
        parser.add_argument("--ocean_teacher_path", type=Path, default=Path("../out/ocean_dpo_768.pth"))
        parser.add_argument("--general_teacher_path", type=Path, default=Path("/home/anhuang/minimind/out/full_sft_768.pth"))
    else:
        parser.add_argument("--domain", choices=["ocean", "general"], default="ocean")
        parser.add_argument("--teacher_path", type=Path, default=Path("../out/ocean_dpo_768.pth"))
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true")
    parser.add_argument("--swanlab_mode", "--wandb_mode", dest="swanlab_mode", choices=["cloud", "local", "offline", "disabled"], default="cloud")
    parser.add_argument("--swanlab_logdir", "--wandb_logdir", dest="swanlab_logdir", default=None)
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", default=f"OceanHeart-{name}")
    parser.add_argument("--run_name", default=None)
    return parser


def teacher_paths(args):
    if hasattr(args, "ocean_teacher_path"):
        return {"ocean": args.ocean_teacher_path, "general": args.general_teacher_path}
    return {args.domain: args.teacher_path}


def run_training(args):
    if min(args.max_seq_len, args.max_gen_len, args.accumulation_steps, args.eval_interval) < 1:
        raise SystemExit("sequence lengths, accumulation steps, and eval interval must be positive")
    if args.max_steps < 0 or args.reward_clip <= 0 or not 0 < args.epsilon < 1 or args.temperature <= 0:
        raise SystemExit("invalid max_steps, reward_clip, epsilon, or temperature")

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

    paths = {domain: Path(path).expanduser().resolve() for domain, path in teacher_paths(args).items()}
    hashes = {domain: checkpoint_sha256(path) for domain, path in paths.items()}
    if checkpoint and checkpoint.get("teacher_hashes") not in (None, hashes):
        raise RuntimeError("teacher checkpoint changed since this run was saved")

    tracker = None
    if args.use_swanlab and is_main_process():
        import swanlab

        config_log = experiment_config(args)
        config_log["teacher_sha256"] = hashes
        tracker = swanlab
        init_swanlab(
            tracker, checkpoint, args.swanlab_mode,
            project=args.swanlab_project,
            name=args.run_name or args.save_weight.replace("_", "-"),
            logdir=args.swanlab_logdir, config=config_log,
        )

    model, tokenizer = init_model(config, args.from_weight, save_dir=args.save_dir, device=args.device)
    teachers = {domain: load_teacher(config, path, args.device, dtype) for domain, path in paths.items()}
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
        if not is_main_process():
            return
        lm_checkpoint(
            config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
            epoch=epoch, step=step, wandb=tracker, save_dir="../checkpoints",
            teacher_hashes=hashes,
        )
        suffix = "_moe" if config.use_moe else ""
        source = f"../checkpoints/{args.save_weight}_{config.hidden_size}{suffix}.pth"
        target = f"{args.save_dir}/{args.save_weight}_{config.hidden_size}{suffix}.pth"
        temp = target + ".tmp"
        shutil.copyfile(source, temp)
        os.replace(temp, target)

    def add_metrics(window, metrics, weight, prefix=""):
        for name, value in metrics.items():
            window.add(f"{prefix}{name}", value, weight)

    def validate(update, full=False):
        nonlocal tracker
        if not val_ds:
            return
        if dist.is_initialized():
            dist.barrier()
        if is_main_process():
            raw = model.module if isinstance(model, DistributedDataParallel) else model
            loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            totals = {}

            def add(name, value, weight):
                total, count = totals.get(name, (0.0, 0.0))
                scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                totals[name] = (total + scalar * weight, count + weight)

            devices = [torch.device(args.device).index or 0] if device_type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                setup_seed(42 + update)
                for index, batch in enumerate(loader):
                    if not full and args.eval_batches and index >= args.eval_batches:
                        break
                    inputs = tokenizer(
                        batch["prompt"], return_tensors="pt", padding=True,
                        padding_side="left", add_special_tokens=False,
                    ).to(args.device)
                    with autocast:
                        result = rollout(
                            raw, tokenizer, inputs["input_ids"][:, -args.max_seq_len:],
                            inputs["attention_mask"][:, -args.max_seq_len:], 1,
                            args.max_gen_len, args.temperature,
                        )
                        full_mask = result.output_ids.ne(tokenizer.pad_token_id).long()
                        teacher_logps = routed_teacher_log_probs(
                            teachers, batch["domain"], result.output_ids,
                            result.completion_ids, full_mask,
                        )
                        metrics = opd_objective(
                            result.old_log_probs.float(), result.old_log_probs.float(),
                            teacher_logps, result.completion_mask, args.reward_clip, args.epsilon,
                        )
                    tokens = result.completion_mask.sum().item()
                    for name, value in metrics.items():
                        add(name, value, tokens)
                    add("response_length", result.completion_mask.sum(1).float().mean(), len(batch["domain"]))
                    for domain in sorted(set(batch["domain"])):
                        indices = torch.tensor(
                            [i for i, value in enumerate(batch["domain"]) if value == domain],
                            device=args.device,
                        )
                        mask = result.completion_mask.index_select(0, indices)
                        domain_metrics = opd_objective(
                            result.old_log_probs.index_select(0, indices).float(),
                            result.old_log_probs.index_select(0, indices).float(),
                            teacher_logps.index_select(0, indices), mask,
                            args.reward_clip, args.epsilon,
                        )
                        domain_tokens = mask.sum().item()
                        for name, value in domain_metrics.items():
                            add(f"{domain}/{name}", value, domain_tokens)
            summary = {name: total / max(count, 1) for name, (total, count) in totals.items()}
            tracker = safe_swanlab_log(
                tracker, {f"val/{name}": value for name, value in summary.items()}, step=update
            )
            Logger(f"Validation step {update}: reward={summary.get('opd_reward', 0):.4f}, reverse_kl={summary.get('reverse_kl', 0):.4f}")
        if dist.is_initialized():
            dist.barrier()

    if val_ds and not checkpoint:
        validate(0)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        sampler and sampler.set_epoch(epoch)
        setup_seed(42 + epoch + rank)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
            pin_memory=device_type == "cuda",
        )
        iterations = len(loader) + skip
        total_updates = max(math.ceil(args.epochs * iterations / args.accumulation_steps), 1)
        window, domain_prompts, domain_tokens = MetricWindow(), Counter(), Counter()
        window_tokens, window_start = 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=skip + 1):
            inputs = tokenizer(
                batch["prompt"], return_tensors="pt", padding=True,
                padding_side="left", add_special_tokens=False,
            ).to(args.device)
            prompt_ids = inputs["input_ids"][:, -args.max_seq_len:]
            prompt_mask = inputs["attention_mask"][:, -args.max_seq_len:]
            with autocast:
                result = rollout(
                    model, tokenizer, prompt_ids, prompt_mask, 1,
                    args.max_gen_len, args.temperature,
                )
            full_mask = result.output_ids.ne(tokenizer.pad_token_id).long()
            # Exit rollout autocast first: generate runs in inference_mode and its
            # cached FP16 weights cannot be reused by an autograd-enabled pass.
            with autocast:
                output = model(
                    result.output_ids, attention_mask=full_mask,
                    logits_to_keep=result.completion_ids.size(1) + 1,
                )
                new_logps = F.log_softmax(output.logits[:, :-1], dim=-1).gather(
                    2, result.completion_ids.unsqueeze(-1)
                ).squeeze(-1).float()
                teacher_logps = routed_teacher_log_probs(
                    teachers, batch["domain"], result.output_ids,
                    result.completion_ids, full_mask,
                )
                metrics = opd_objective(
                    new_logps, result.old_log_probs.float(), teacher_logps,
                    result.completion_mask, args.reward_clip, args.epsilon,
                )
                loss = (metrics["policy_loss"] + output.aux_loss) / args.accumulation_steps
            scaler.scale(loss).backward()
            tokens = result.completion_mask.sum().item()
            add_metrics(window, metrics, tokens)
            window.add("aux_loss", output.aux_loss, len(batch["domain"]))
            window.add("response_length", result.completion_mask.sum(1).float().mean(), len(batch["domain"]))
            window_tokens += tokens
            for domain in sorted(set(batch["domain"])):
                indices = torch.tensor(
                    [i for i, value in enumerate(batch["domain"]) if value == domain],
                    device=args.device,
                )
                mask = result.completion_mask.index_select(0, indices)
                domain_metrics = opd_objective(
                    new_logps.index_select(0, indices),
                    result.old_log_probs.index_select(0, indices).float(),
                    teacher_logps.index_select(0, indices), mask,
                    args.reward_clip, args.epsilon,
                )
                domain_count = len(indices)
                domain_token_count = mask.sum().item()
                add_metrics(window, domain_metrics, domain_token_count, f"{domain}/")
                domain_prompts[domain] += domain_count
                domain_tokens[domain] += domain_token_count

            final_micro = step == iterations
            if step % args.accumulation_steps and not final_micro:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update = math.ceil((epoch * iterations + step) / args.accumulation_steps)
            lr = get_lr(update, total_updates, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window.add("grad_norm", grad_norm)
            window.add("grad_clipped", float(grad_norm > args.grad_clip))
            if update % args.log_interval == 0 or final_micro:
                # Every rank must reduce identical metric keys even when its local
                # batch contains only one MOPD domain.
                for domain in sorted(teachers):
                    for name in OPD_METRIC_NAMES:
                        window.add(f"{domain}/{name}", 0, 0)
                summary = window.means(args.device)
                ordered = sorted(teachers)
                packed = [value for domain in ordered for value in (domain_prompts[domain], domain_tokens[domain])]
                counts = reduce_sums(packed, args.device)
                prompt_total = sum(counts[::2])
                token_total = sum(counts[1::2])
                for index, domain in enumerate(ordered):
                    summary[f"{domain}/prompt_share"] = counts[index * 2] / max(prompt_total, 1)
                    summary[f"{domain}/token_share"] = counts[index * 2 + 1] / max(token_total, 1)
                elapsed = max(time.time() - window_start, 1e-6)
                summary["loss"] = summary["policy_loss"] + summary["aux_loss"]
                summary["learning_rate"] = lr
                summary["tokens_per_second"] = reduce_sums([window_tokens], args.device)[0] / elapsed
                tracker = safe_swanlab_log(
                    tracker, {f"train/{name}": value for name, value in summary.items()}, step=update
                )
                Logger(
                    f"Epoch [{epoch + 1}/{args.epochs}] step={step}/{iterations} update={update} "
                    f"loss={summary['loss']:.4f} reward={summary['opd_reward']:.4f}"
                )
                domain_prompts.clear()
                domain_tokens.clear()
                window_tokens, window_start = 0, time.time()
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
        validate(math.ceil((epoch + 1) * iterations / args.accumulation_steps), full=True)

    if tracker and hasattr(tracker, "finish"):
        tracker.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_training(build_parser().parse_args())
