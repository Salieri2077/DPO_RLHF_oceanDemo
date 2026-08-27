import argparse
import math
import os
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

from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    MetricWindow,
    SkipBatchSampler,
    experiment_config,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    reduce_sums,
    safe_swanlab_log,
    setup_seed,
)


def sequence_log_probs(logits, labels, mask):
    token_log_probs = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(2)).squeeze(2)
    return (token_log_probs * mask).sum(dim=1)


def dpo_metrics(ref_log_probs, policy_log_probs, beta):
    half = ref_log_probs.shape[0] // 2
    chosen_reward = beta * (policy_log_probs[:half] - ref_log_probs[:half])
    rejected_reward = beta * (policy_log_probs[half:] - ref_log_probs[half:])
    margin = chosen_reward - rejected_reward
    return {
        "dpo_loss": -F.logsigmoid(margin).mean(),
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "reward_margin": margin.mean(),
        "preference_accuracy": (margin > 0).float().mean(),
    }


def unpack_batch(batch, device):
    values = {key: value.to(device) for key, value in batch.items()}
    x = torch.cat([values["x_chosen"], values["x_rejected"]])
    y = torch.cat([values["y_chosen"], values["y_rejected"]])
    mask = torch.cat([values["mask_chosen"], values["mask_rejected"]])
    return x, y, mask


@torch.no_grad()
def evaluate(model, ref_model, loader, device, autocast_ctx, max_batches=0):
    if dist.is_initialized():
        dist.barrier()
    result = None
    if is_main_process():
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        was_training = raw_model.training
        raw_model.eval()
        totals, pair_count = {}, 0
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            x, y, mask = unpack_batch(batch, device)
            with autocast_ctx:
                ref = sequence_log_probs(ref_model(x).logits, y, mask)
                output = raw_model(x)
                policy = sequence_log_probs(output.logits, y, mask)
                metrics = dpo_metrics(ref, policy, args.beta)
            pairs = x.size(0) // 2
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value.item() * pairs
            totals["aux_loss"] = totals.get("aux_loss", 0.0) + output.aux_loss.item() * pairs
            pair_count += pairs
        result = {name: total / max(pair_count, 1) for name, total in totals.items()}
        result["loss"] = result["dpo_loss"] + result["aux_loss"]
        if was_training:
            raw_model.train()
    if dist.is_initialized():
        dist.barrier()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="OceanHeart Direct Preference Optimization")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--save_weight", default="ocean_dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5, help="optimizer steps")
    parser.add_argument("--save_interval", type=int, default=50, help="optimizer steps")
    parser.add_argument("--eval_interval", type=int, default=10, help="optimizer steps")
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--data_path", default="../data/processed/ocean_dpo_train.jsonl")
    parser.add_argument("--val_data_path", default="../data/processed/ocean_dpo_val.jsonl")
    parser.add_argument("--from_weight", default="ocean_sft_replay")
    parser.add_argument("--from_resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true")
    parser.add_argument("--swanlab_mode", "--wandb_mode", dest="swanlab_mode", choices=["cloud", "local", "offline", "disabled"], default="cloud")
    parser.add_argument("--swanlab_logdir", "--wandb_logdir", dest="swanlab_logdir", default=None)
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", default="OceanHeart-DPO")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_compile", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(42 + rank)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume else None
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)

    tracker = None
    if args.use_swanlab and is_main_process():
        import swanlab

        run_id = (ckp_data or {}).get("swanlab_id") or (ckp_data or {}).get("wandb_id")
        tracker = swanlab
        tracker.init(
            project=args.swanlab_project,
            name=args.run_name or f"OceanHeart-DPO-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
            id=run_id,
            resume="must" if run_id else None,
            mode=args.swanlab_mode,
            logdir=args.swanlab_logdir,
            config=experiment_config(args),
        )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval().requires_grad_(False)
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_ds = DPODataset(args.val_data_path, tokenizer, max_length=args.max_seq_len, deterministic=True) if args.val_data_path else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler("cuda", enabled=device_type == "cuda" and args.dtype == "float16")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch = start_step = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)
    if args.use_compile:
        model = torch.compile(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    def save_checkpoint(epoch, step):
        if not is_main_process():
            return
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        suffix = "_moe" if lm_config.use_moe else ""
        path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{suffix}.pth"
        state = {key: value.half().cpu() for key, value in raw_model.state_dict().items()}
        temp = path + ".tmp"
        torch.save(state, temp)
        os.replace(temp, path)
        lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=tracker, save_dir="../checkpoints")

    def validate(global_update, full=False):
        global tracker
        if not val_loader:
            return
        metrics = evaluate(model, ref_model, val_loader, args.device, autocast_ctx, 0 if full else args.eval_batches)
        if metrics:
            tracker = safe_swanlab_log(tracker, {f"val/{key}": value for key, value in metrics.items()}, step=global_update)
            Logger(f"Validation step {global_update}: loss={metrics['loss']:.4f}, accuracy={metrics['preference_accuracy']:.3f}")

    if val_loader and not ckp_data:
        validate(0)

    stop = False
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=device_type == "cuda")
        iters = len(loader) + skip
        window, window_tokens, window_start = MetricWindow(), 0, time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=skip + 1):
            x, y, mask = unpack_batch(batch, args.device)
            lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast_ctx:
                with torch.no_grad():
                    ref = sequence_log_probs(ref_model(x).logits, y, mask)
                output = model(x)
                policy = sequence_log_probs(output.logits, y, mask)
                metrics = dpo_metrics(ref, policy, args.beta)
                loss = (metrics["dpo_loss"] + output.aux_loss) / args.accumulation_steps
            scaler.scale(loss).backward()
            pairs = x.size(0) // 2
            for name, value in metrics.items():
                window.add(name, value, pairs)
            window.add("aux_loss", output.aux_loss, pairs)
            window_tokens += mask.sum().item()

            final_micro_step = step == iters
            if step % args.accumulation_steps and not final_micro_step:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_update = math.ceil((epoch * iters + step) / args.accumulation_steps)
            window.add("grad_norm", grad_norm)
            window.add("grad_clipped", float(grad_norm > args.grad_clip))

            if global_update % args.log_interval == 0 or final_micro_step:
                summary = window.means(args.device)
                summary["loss"] = summary["dpo_loss"] + summary["aux_loss"]
                summary["learning_rate"] = lr
                summary["tokens_per_second"] = reduce_sums([window_tokens], args.device)[0] / max(time.time() - window_start, 1e-6)
                tracker = safe_swanlab_log(tracker, {f"train/{key}": value for key, value in summary.items()}, step=global_update)
                Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) update={global_update}, loss={summary['loss']:.4f}, accuracy={summary['preference_accuracy']:.3f}")
                window_tokens, window_start = 0, time.time()
            if val_loader and global_update % args.eval_interval == 0:
                validate(global_update)
            if global_update % args.save_interval == 0:
                save_checkpoint(epoch, step)
            if args.max_steps and global_update >= args.max_steps:
                save_checkpoint(epoch, step)
                stop = True
                break

        if not stop:
            save_checkpoint(epoch, iters)
            validate(math.ceil(((epoch + 1) * iters) / args.accumulation_steps), full=True)
        start_step = 0
        if stop:
            break

    if tracker and hasattr(tracker, "finish"):
        tracker.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
