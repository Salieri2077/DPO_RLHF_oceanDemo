#!/usr/bin/env python3
"""Ocean SFT of a Hugging Face base model (Qwen3-1.7B-Base) with LoRA on the same data as the MiniMind SFT.

Same data files and 10% general replay as train_full_sft.py; ChatML from trainer/hf_chat.py. The frozen
base stays fp16 (2080 Ti has no bf16) and LoRA trains in fp32 under autocast with a GradScaler. Loss is
computed only at assistant positions, in checkpointed vocabulary chunks, so 152K-way logits never
materialise for a whole micro-batch. Validation reports token loss and nats per character; the latter
is comparable across tokenizers (MiniMind vs Qwen).
"""
import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from trainer.hf_chat import encode_conversation, load_tokenizer
from trainer.train_agent import atomic_json
from trainer.trainer_utils import MetricWindow, experiment_config, init_distributed_mode, init_swanlab, reduce_sums, setup_seed

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--save_adapter", required=True, help="adapter name under out/hf/ (writes _best, _last, _final)")
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_path", type=Path, default=ROOT / "data/processed/ocean_sft_replay_train.jsonl")
    p.add_argument("--val_data_path", type=Path, default=ROOT / "data/processed/ocean_sft_val.jsonl")
    p.add_argument("--val_samples", type=int, default=512)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_train_seconds", type=float, default=86400)
    p.add_argument("--samples_per_update", type=int, default=16, help="per rank")
    p.add_argument("--micro_tokens", type=int, default=4096, help="padded tokens per micro-batch")
    p.add_argument("--loss_chunk", type=int, default=1024, help="assistant positions per vocabulary chunk")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=.05)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--use_swanlab", action="store_true")
    p.add_argument("--swanlab_mode", choices=["cloud", "offline", "disabled"], default="cloud")
    p.add_argument("--swanlab_project", default="OceanHeart-SFT")
    return p


def read_rows(path, tokenizer, max_len, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids, labels, chars = encode_conversation(tokenizer, json.loads(line)["conversations"], max_len)
            rows.append((ids, labels, chars))
            if limit and len(rows) >= limit:
                break
    return rows


def update_batches(lengths, rank, world, per_update, seed):
    """This rank's updates for one epoch: shuffled, grouped by similar length to cut padding."""
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    order = order[:len(order) // world * world][rank::world]
    group = per_update * 32
    updates = []
    for start in range(0, len(order), group):
        chunk = sorted(order[start:start + group], key=lambda i: lengths[i])
        updates += [chunk[i:i + per_update] for i in range(0, len(chunk), per_update)]
    random.Random(seed + 1).shuffle(updates)  # same seed on every rank keeps update counts aligned
    return updates


def micro_batches(indices, lengths, budget):
    batches, current = [], []
    for index in sorted(indices, key=lambda i: lengths[i]):
        if current and (len(current) + 1) * max(lengths[i] for i in current + [index]) > budget:
            batches.append(current)
            current = []
        current.append(index)
    return batches + [current] if current else batches


def collate(rows, indices, pad_id, device):
    width = max(len(rows[i][0]) for i in indices)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    labels = torch.full((len(indices), width), -100, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, i in enumerate(indices):
        n = len(rows[i][0])
        ids[row, :n], labels[row, :n], mask[row, :n] = torch.tensor(rows[i][0]), torch.tensor(rows[i][1]), 1
    return ids.to(device), labels.to(device), mask.to(device)


def summed_loss(model, ids, labels, mask, chunk):
    """Sum of next-token NLL over labeled positions and the number of those positions."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = core.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    keep = targets.ne(-100)
    hidden, targets = hidden[keep], targets[keep]

    def piece(h, t):
        return F.cross_entropy(core.lm_head(h).float(), t, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        h, t = hidden[start:start + chunk], targets[start:start + chunk]
        total = total + (checkpoint(piece, h, t, use_reentrant=False) if torch.is_grad_enabled() else piece(h, t))
    return total, targets.numel()


class AssistantLoss(torch.nn.Module):
    """Routes the loss through one module forward so DDP prepares its gradient reducer."""

    def __init__(self, model, chunk):
        super().__init__()
        self.model, self.chunk = model, chunk

    def forward(self, ids, labels, mask):
        return summed_loss(self.model, ids, labels, mask, self.chunk)


def main():
    args = parser().parse_args()
    if Path(args.run_name).name != args.run_name or Path(args.save_adapter).name != args.save_adapter:
        raise ValueError("run_name and save_adapter must be simple names")
    if any((ROOT / "out/hf" / f"{args.save_adapter}{s}").exists() for s in ("_best", "_last", "_final")):
        raise FileExistsError("Refusing to overwrite an existing adapter; use a new --save_adapter")
    local_rank = init_distributed_mode()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    device = torch.device(f"cuda:{local_rank}")
    setup_seed(42 + rank)
    directory = ROOT / "artifacts/sft_hf" / args.run_name
    directory.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path)
    train_rows = read_rows(args.data_path, tokenizer, args.max_len)
    val_rows = read_rows(args.val_data_path, tokenizer, args.max_len, args.val_samples)
    lengths = [len(r[0]) for r in train_rows]

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.train()  # from_pretrained returns eval mode, where checkpointing and LoRA dropout are off
    trainable = [p for p in model.parameters() if p.requires_grad]

    updates_per_epoch = len(update_batches(lengths, rank, world, args.samples_per_update, 0))
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    metadata = experiment_config(args)
    metadata.update(world_size=world, dtype="float16 base, float32 LoRA", chat_format="ChatML, no think",
                    train_samples=len(train_rows), train_tokens=sum(lengths), updates_per_epoch=updates_per_epoch,
                    total_steps=total_steps, global_samples_per_update=args.samples_per_update * world,
                    truncated_samples=sum(n >= args.max_len for n in lengths),
                    trainable_parameters=sum(p.numel() for p in trainable),
                    total_parameters=sum(p.numel() for p in model.parameters()))
    tracker = None
    if rank == 0:
        atomic_json(directory / "config.json", metadata)
        print(json.dumps({k: metadata[k] for k in ("train_samples", "train_tokens", "updates_per_epoch", "total_steps",
                                                   "trainable_parameters", "truncated_samples")}), flush=True)
        if args.use_swanlab:
            import swanlab
            tracker = swanlab
            init_swanlab(tracker, None, args.swanlab_mode, project=args.swanlab_project, name=args.run_name,
                         logdir=str(directory / "swanlog"), config=metadata)

    def log(values, step):
        if rank == 0:
            if tracker:
                tracker.log(values, step=step)
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **values}) + "\n")

    def save(suffix):
        if rank == 0:
            target = ROOT / "out/hf" / f"{args.save_adapter}{suffix}"
            model.save_pretrained(target)  # LoRA weights only
            tokenizer.save_pretrained(target)  # carries the ChatML template used in training
        if world > 1:
            dist.barrier()

    @torch.no_grad()
    def validation():
        model.eval()
        mine = list(range(rank, len(val_rows), world))
        lengths_val = [len(r[0]) for r in val_rows]
        nll = tokens = chars = 0.
        for batch in micro_batches(mine, lengths_val, args.micro_tokens):
            ids, labels, mask = collate(val_rows, batch, tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                total, count = summed_loss(model, ids, labels, mask, args.loss_chunk)
            nll, tokens, chars = nll + total.item(), tokens + count, chars + sum(val_rows[i][2] for i in batch)
        nll, tokens, chars = reduce_sums([nll, tokens, chars], device)
        model.train()
        return {"val/loss": nll / tokens, "val/ppl": math.exp(nll / tokens), "val/nats_per_char": nll / chars}

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
    loss_module = AssistantLoss(model, args.loss_chunk)
    wrapped = DDP(loss_module, device_ids=[local_rank]) if world > 1 else loss_module

    def learning_rate(step):  # linear warmup, cosine decay to 10%
        return args.learning_rate * min(1., step / max(args.warmup_steps, 1)) * (
            .1 + .9 * (1 + math.cos(math.pi * min(step / total_steps, 1))) / 2)

    deadline, started = time.time() + args.max_train_seconds, time.monotonic()
    best, global_step, stop_reason = float("inf"), 0, "epochs_completed"

    def evaluate_and_save():
        nonlocal best
        torch.cuda.empty_cache()
        values = validation()
        log(values, global_step)
        if rank == 0:
            print(f"validation step={global_step} {json.dumps(values)}", flush=True)
        save("_last")
        if values["val/loss"] < best:
            best = values["val/loss"]
            save("_best")

    log(validation(), 0)
    window, last_time = MetricWindow(), time.monotonic()
    for epoch in range(args.epochs):
        for update in update_batches(lengths, rank, world, args.samples_per_update, epoch):
            flag = torch.tensor(int(time.time() >= deadline), device=device)
            if world > 1:
                dist.broadcast(flag, 0)
            if flag.item():
                stop_reason = "time_budget"
                break
            batches = micro_batches(update, lengths, args.micro_tokens)
            label_tokens = sum(sum(t != -100 for t in train_rows[i][1][1:]) for i in update)
            for number, batch in enumerate(batches, 1):
                ids, labels, mask = collate(train_rows, batch, tokenizer.pad_token_id, device)
                sync = number == len(batches)
                with (wrapped.no_sync() if world > 1 and not sync else nullcontext()):
                    with torch.autocast("cuda", dtype=torch.float16):
                        total, count = wrapped(ids, labels, mask)
                    scaler.scale(total / label_tokens).backward()
                window.add("loss", total / max(count, 1), count)
            global_step += 1
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(global_step)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if reduce_sums([float(torch.isfinite(norm))], device)[0] != world:
                if rank == 0:
                    print(f"update={global_step} non-finite gradient; GradScaler skips it", flush=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            window.add("grad_norm", norm if torch.isfinite(norm) else torch.zeros(()))
            values = window.means(device)
            values.update(learning_rate=optimizer.param_groups[0]["lr"], loss_scale=scaler.get_scale(), epoch=epoch + 1,
                          seconds_per_update=time.monotonic() - last_time,
                          peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30)
            last_time = time.monotonic()
            log({f"train/{k}": v for k, v in values.items()}, global_step)
            if rank == 0 and (global_step <= 5 or global_step % 10 == 0):
                print(f"update={global_step}/{total_steps} {json.dumps(values)}", flush=True)
            if global_step % args.eval_interval == 0:
                evaluate_and_save()
            if args.max_steps and global_step >= args.max_steps:
                stop_reason = "max_steps"
                break
        if stop_reason != "epochs_completed":
            break
    if global_step % args.eval_interval:
        evaluate_and_save()
    save("_final")
    if rank == 0:
        atomic_json(directory / "completion.json", {"global_step": global_step, "stop_reason": stop_reason,
                                                    "best_val_loss": best, "elapsed_seconds": time.monotonic() - started})
    if tracker:
        tracker.finish()
    if world > 1:
        torch.cuda.empty_cache()  # NCCL shutdown allocates outside PyTorch's cache on near-full cards
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
