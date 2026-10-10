#!/usr/bin/env python3
"""Agent SFT (C0 common tool adaptation) on verified OceanDepths demonstrations, LoRA on a Hugging Face model.

Each assistant turn of a demonstration is one training sample: its exact inference prompt (OceanLabHarness.prompt_ids
over the messages before it, i.e. what the SDK harness feeds the model at that turn) followed by the assistant text
and the end-of-turn token; only those target tokens carry loss. One sample per turn is required because the Instruct
template renders earlier assistant turns without the empty think block that the generation prompt contains, so no
single sequence matches every turn's inference context. Logits are computed only at target positions.

Subset: demonstrations with rank < --subset and not in the hold-out (nested learning-curve subsets 250/500/950); the
hold-out demonstrations give a validation loss. The frozen base stays fp16; LoRA trains in fp32 under autocast (DDP).
Launch with torchrun; the adapter is written to out/hf/<save_adapter>_final.
"""
import argparse
import hashlib
import json
import math
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
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM

from agent.ocean import file_hash, read_jsonl
from agent.oceanlab.env import OceanLabHarness
from trainer.qwen_sdpa import memory_efficient_attention
from trainer.train_agent import atomic_json
from trainer.train_agent_hf import LORA_TARGETS, agent_tokenizer, native_tokenizer
from trainer.trainer_utils import MetricWindow, init_distributed_mode, init_swanlab, reduce_sums, setup_seed

DEMOS = ROOT / "data/processed/oceanlab/c0/oceanlab_c0_demos_v1.jsonl"


class TurnSamples(torch.utils.data.Dataset):
    """(prompt + target ids, index of the first target token) for every assistant turn of the selected demos."""

    def __init__(self, rows, tokenizer, max_len):
        harness = OceanLabHarness()
        self.samples = []
        for row in rows:
            messages = row["messages"]
            if messages[0] != {"role": "system", "content": harness.system}:
                raise ValueError(f"{row['id']}: demonstration was made with a different system prompt")
            for i, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                prompt = harness.prompt_ids(tokenizer, messages[:i])
                target = tokenizer.encode(message["content"], add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(prompt) + len(target) > max_len:
                    raise ValueError(f"{row['id']} turn {i}: {len(prompt) + len(target)} tokens exceed {max_len}")
                self.samples.append((prompt + target, len(prompt)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TurnLoss(torch.nn.Module):
    """Mean cross-entropy over a sample's target tokens; logits only where there is a target (DDP wraps this)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, start):
        base = self.model.get_base_model()
        hidden = base.model(input_ids=ids).last_hidden_state[0, start - 1:-1]
        logits = base.lm_head(hidden).float()
        return F.cross_entropy(logits, ids[0, start:])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=Path, default=Path("/data/anhuang/oceanheart_models/qwen3-1.7b"))
    p.add_argument("--chat_template", choices=["minimind", "native"], default="native")
    p.add_argument("--demos", type=Path, default=DEMOS)
    p.add_argument("--subset", type=int, required=True, help="train on demonstrations with rank < subset")
    p.add_argument("--save_adapter", required=True, help="adapter name under out/hf/ (writes <name>_final)")
    p.add_argument("--run_name", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=.05)
    p.add_argument("--max_len", type=int, default=6144)
    p.add_argument("--val_interval", type=int, default=50)
    p.add_argument("--verify_sync", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_swanlab", action="store_true")
    p.add_argument("--swanlab_mode", choices=["cloud", "offline", "disabled"], default="cloud")
    p.add_argument("--swanlab_project", default="OceanHeart-Agent")
    return p


def main():
    args = parser().parse_args()
    if Path(args.run_name).name != args.run_name or Path(args.save_adapter).name != args.save_adapter:
        raise ValueError("run_name and save_adapter must be simple names")
    if (ROOT / "out/hf" / f"{args.save_adapter}_final").exists():
        raise FileExistsError("Refusing to overwrite an existing adapter; use a new --save_adapter")
    memory_efficient_attention()
    torch.cos(torch.zeros(4096))  # keep the first-call cos glitch off the RoPE table
    local_rank = init_distributed_mode()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed + rank)
    directory = ROOT / "artifacts/oceanlab/c0" / args.run_name
    directory.mkdir(parents=True, exist_ok=True)

    tokenizer = agent_tokenizer(args.model_path) if args.chat_template == "minimind" else native_tokenizer(args.model_path, False)
    rows = read_jsonl(args.demos)
    train_rows = [r for r in rows if not r["holdout"] and r["rank"] < args.subset]
    holdout_rows = [r for r in rows if r["holdout"]]
    if args.subset > sum(not r["holdout"] for r in rows):
        raise ValueError("subset larger than the training demonstrations")
    train, holdout = TurnSamples(train_rows, tokenizer, args.max_len), TurnSamples(holdout_rows, tokenizer, args.max_len)

    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.config.use_cache = False
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    loss_module = TurnLoss(model)
    # No buffer broadcast: validation runs the unwrapped module a different number of times on each rank.
    wrapped = DDP(loss_module, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else loss_module

    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    updates_per_epoch = math.ceil(len(sampler) / args.accumulation_steps)
    total_steps = max(args.max_steps or updates_per_epoch * args.epochs, 1)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(world_size=world, train_demos=len(train_rows), train_samples=len(train),
                  holdout_demos=len(holdout_rows), holdout_samples=len(holdout), updates_per_epoch=updates_per_epoch,
                  total_steps=total_steps, effective_batch=args.accumulation_steps * world,
                  demos_sha256=file_hash(args.demos), env_sha256=file_hash(ROOT / "agent/oceanlab/env.py"),
                  chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                  model_config_sha256=file_hash(args.model_path / "config.json"),
                  trainable_parameters=sum(p.numel() for p in trainable), dtype="float16 base, float32 LoRA",
                  attention="sdpa, KV heads expanded (memory-efficient kernel)",
                  train_target_tokens=sum(len(ids) - start for ids, start in train.samples),
                  train_total_tokens=sum(len(ids) for ids, _ in train.samples))
    tracker = None
    if rank == 0:
        atomic_json(directory / "config.json", config)
        print(json.dumps(config, ensure_ascii=False), flush=True)
        if args.use_swanlab:
            try:  # tracking must never stop training
                import swanlab
                init_swanlab(swanlab, None, args.swanlab_mode, project=args.swanlab_project, name=args.run_name,
                             logdir=str(directory / "swanlog"), config=config)
                tracker = swanlab
            except Exception as exc:
                print(f"swanlab disabled: {exc}", flush=True)

    def log(values, step):
        if rank == 0:
            if tracker:
                try:
                    tracker.log(values, step=step)
                except Exception as exc:  # tracking must never stop training
                    print(f"swanlab log failed: {exc}", flush=True)
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **values}) + "\n")

    def run(ids_start, module=wrapped):
        ids, start = ids_start
        ids = torch.tensor([ids], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return module(ids, start)

    def validation(step):
        model.eval()
        total = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for index in range(rank, len(holdout), world):
                ids, start = holdout[index]
                loss = run((ids, start), loss_module)
                total += torch.tensor([float(loss) * (len(ids) - start), len(ids) - start], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(total)
        model.train()
        value = float(total[0] / total[1])
        log({"val/loss": value}, step)
        if rank == 0:
            print(f"validation step={step} holdout_loss={value:.4f}", flush=True)
        return value

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=device.type == "cuda")

    def learning_rate(step):  # short linear warmup, then cosine decay to 10%
        return args.learning_rate * min(1., step / max(args.warmup_steps, 1)) * (
            .1 + .9 * (1 + math.cos(math.pi * min(step / total_steps, 1))) / 2)

    started = time.monotonic()
    losses = {"start": validation(0)}
    global_step, stop_reason = 0, "epochs_completed"
    window, last_time = MetricWindow(), time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        for micro, index in enumerate(indices, 1):
            flush = micro % args.accumulation_steps == 0 or micro == len(indices)
            group = min(args.accumulation_steps, len(indices) - ((micro - 1) // args.accumulation_steps) * args.accumulation_steps)
            ids, start = train[index]
            with (wrapped.no_sync() if world > 1 and not flush else nullcontext()):
                loss = run((ids, start))
                scaler.scale(loss / group).backward()
            window.add("loss", loss, len(ids) - start)
            window.add("tokens", len(ids))
            if not flush:
                continue
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate(global_step + 1)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if reduce_sums([float(torch.isfinite(norm))], device)[0] != world:
                raise FloatingPointError("non-finite gradient")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            window.add("grad_norm", norm)
            window.add("learning_rate", optimizer.param_groups[0]["lr"])
            summary = window.means(device)
            summary.update(seconds_per_update=time.monotonic() - last_time, epoch=epoch + 1,
                           peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0)
            window, last_time = MetricWindow(), time.monotonic()
            log({f"train/{k}": v for k, v in summary.items()}, global_step)
            if rank == 0:
                print(f"update={global_step}/{total_steps} {json.dumps(summary)}", flush=True)
            if args.verify_sync:
                digest = hashlib.sha256()
                for parameter in trainable:
                    digest.update(parameter.detach().cpu().numpy().tobytes())
                hashes = [None] * world
                if world > 1:
                    dist.all_gather_object(hashes, digest.hexdigest())
                    assert len(set(hashes)) == 1, "DDP LoRA parameters diverged"
            if global_step % args.val_interval == 0:
                losses[f"step{global_step}"] = validation(global_step)
            if args.max_steps and global_step >= args.max_steps:
                stop_reason = "max_steps"
                break
        if stop_reason != "epochs_completed":
            break
    if global_step % args.val_interval:
        losses[f"step{global_step}"] = validation(global_step)
    if rank == 0:
        model.save_pretrained(ROOT / "out/hf" / f"{args.save_adapter}_final")  # LoRA weights only
        atomic_json(directory / "completion.json", {
            "global_step": global_step, "stop_reason": stop_reason, "elapsed_seconds": time.monotonic() - started,
            "holdout_loss": losses, "adapter": f"out/hf/{args.save_adapter}_final", "sync_verified": args.verify_sync})
        if tracker:
            tracker.finish()
    if world > 1:
        dist.barrier()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
