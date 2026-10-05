#!/usr/bin/env python3
"""OceanHeart multi-turn tool SFT / GRPO. MiniMind architecture, Apache-2.0.

Run from any working directory. No online judge in the optimization loop.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from agent.ocean import OceanTools, VERSION, file_hash, read_jsonl, run_trajectory
from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.prepare_ocean_data import atomic_jsonl
from trainer.train_grpo import group_advantages, grpo_objective
from trainer.trainer_utils import (MetricWindow, experiment_config, init_distributed_mode, init_swanlab,
                                   lm_checkpoint, reduce_sums, setup_seed)


class AgentSFTDataset(SFTDataset):
    """Preserve the exact inference template, including its empty thinking prefix.

    The QA SFTDataset intentionally removes that prefix in deterministic mode;
    that normalization must not be used for tool protocol supervision.
    """
    def __getitem__(self, index):
        prompt = self.create_chat_prompt(self.samples[index]["conversations"])
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError("Agent demonstration exceeds context; regenerate data rather than truncate")
        labels = self.generate_labels(ids)
        padding = self.max_length - len(ids)
        return (torch.tensor(ids + [self.tokenizer.pad_token_id] * padding, dtype=torch.long),
                torch.tensor(labels + [-100] * padding, dtype=torch.long))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def public_trace(trace):
    return {**trace, "rounds": [{**r, "input_tokens": len(r["input_ids"]),
                                 "output_tokens": len(r["completion_ids"])} for r in trace["rounds"]]}


def aggregate(traces):
    metrics = {}
    for key in ("reward", "reward_tool", "reward_progress", "reward_answer", "success", "turns", "context_limit"):
        metrics[key] = sum(t["metrics"][key] for t in traces) / max(len(traces), 1)
    attempts = sum(t["metrics"]["attempted_calls"] for t in traces)
    metrics["tool_valid_rate"] = sum(t["metrics"]["valid_calls"] for t in traces) / max(attempts, 1)
    metrics["attempted_calls"] = attempts
    for kind in ("calculate", "retrieve", "chain"):
        subset = [t for t in traces if t["kind"] == kind]
        metrics[f"{kind}_success"] = sum(t["metrics"]["success"] for t in subset) / max(len(subset), 1)
    groups = {}
    for trace in traces:
        groups.setdefault(trace["id"], []).append(trace["metrics"]["reward"])
    metrics["nonzero_group_ratio"] = sum(max(r) - min(r) > 1e-6 for r in groups.values()) / max(len(groups), 1)
    metrics["questions"], metrics["trajectories"] = len(groups), len(traces)
    metrics["gate_pass"] = (metrics["tool_valid_rate"] >= .8 and metrics["success"] >= .2 and
                            metrics["nonzero_group_ratio"] >= .2 and
                            all(metrics[f"{kind}_success"] > 0 for kind in ("calculate", "retrieve", "chain")))
    return metrics


def round_logps(model, round_, device):
    inputs = torch.tensor([round_["input_ids"] + round_["completion_ids"]], device=device)
    target = torch.tensor([round_["completion_ids"]], device=device)
    output = model(inputs, logits_to_keep=target.size(1) + 1)
    return output.logits[:, :-1].float().log_softmax(-1).gather(2, target.unsqueeze(-1)).squeeze(-1)


def model_generator(model, tokenizer, device, sample=True, save_logps=True):
    @torch.no_grad()
    def generate(ids, limit):
        inputs = torch.tensor([ids], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=limit,
                                    do_sample=sample, temperature=1., top_p=1., top_k=0,
                                    eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.)
        tokens = output[0, len(ids):].tolist()
        if save_logps:
            # Fresh autocast scope: do not reuse inference-mode cached weights.
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logps = round_logps(model, {"input_ids": ids, "completion_ids": tokens}, device)[0].tolist()
        else:
            logps = []
        return tokens, logps
    return generate


def evaluate(model, tokenizer, tasks, env, args, rank, world, directory, label, group_size=1):
    rollout, summarize = run_trajectory, aggregate
    if args.agent_version == "v2":
        from agent.react import run_trajectory as rollout, aggregate as summarize
    if args.runtime == "sdk":
        from agent.sdk import run_trajectory as rollout
    was_training = model.training
    model.eval()
    with torch.random.fork_rng(devices=[args.device.index] if args.device.type == "cuda" else []):
        torch.manual_seed(42 + rank)
        generator = model_generator(model, tokenizer, args.device, sample=group_size > 1 or args.sample_eval, save_logps=False)
        traces = []
        for task in tasks[rank::world]:
            for sample_index in range(group_size):
                if args.agent_version == "v2":
                    seed = int(hashlib.sha256(f"{args.eval_seed}:{task['id']}:{sample_index}".encode()).hexdigest()[:8], 16)
                    torch.manual_seed(seed)
                traces.append(public_trace(rollout(task, tokenizer, env, generator,
                                                         max_total_len=args.max_total_len, max_new_tokens=args.max_gen_len)))
    model.train(was_training)
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, traces)
        traces = [t for shard in gathered for t in shard]
    traces.sort(key=lambda t: t["id"])
    summary = summarize(traces)
    if rank == 0:
        atomic_jsonl(directory / f"{label}.jsonl", traces)
        atomic_json(directory / f"{label}.json", summary)
        with (directory / f"{label}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "kind", "stop", "reward", "success", "final"])
            writer.writeheader()
            for t in traces:
                writer.writerow({**{k: t[k] for k in ("id", "kind", "stop", "final")}, "reward": t["metrics"]["reward"], "success": t["metrics"]["success"]})
        print(f"{label}: {json.dumps(summary)}", flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["grpo", "sft", "eval", "diagnose"], default="grpo")
    p.add_argument("--agent_version", choices=["v1", "v2"], default="v1")
    p.add_argument("--runtime", choices=["local", "sdk"],
                   help="Default: sdk for v2 SFT/eval, local for legacy v1")
    p.add_argument("--sample_eval", action="store_true")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--from_weight", default="ocean_grpo_deepseekv3_eval50")
    p.add_argument("--save_weight", default="ocean_agent_grpo")
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--max_gen_len", type=int, default=192)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-7)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_train_seconds", type=float, default=32400)
    p.add_argument("--deadline", type=float, default=0, help="absolute UNIX deadline, shared by pipeline stages")
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--epsilon", type=float, default=.2)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--from_resume", action="store_true")
    p.add_argument("--verify_sync", action="store_true")
    p.add_argument("--skip_eval", action="store_true", help="engineering smoke only")
    p.add_argument("--use_swanlab", action="store_true")
    p.add_argument("--swanlab_mode", choices=["cloud", "offline", "disabled"], default="cloud")
    p.add_argument("--swanlab_project", default="OceanHeart-Agent")
    return p


def runtime_for(args):
    runtime = args.runtime or ("sdk" if args.agent_version == "v2" else "local")
    if runtime == "sdk" and (args.agent_version != "v2" or args.mode not in {"sft", "eval"}):
        raise ValueError("SDK runtime supports v2 SFT validation/evaluation, not Agentic RL")
    return runtime


def main():
    args = parser().parse_args()
    args.runtime = runtime_for(args)
    if args.agent_version == "v2" and args.mode in {"grpo", "diagnose"}:
        raise ValueError("ReAct v2 currently supports SFT/evaluation only; Agentic RL is out of scope")
    if args.runtime == "sdk":
        import agent.sdk  # Fail before GPU loading if the SDK environment is missing.
    if min(args.accumulation_steps, args.max_gen_len, args.epochs, args.save_interval, args.eval_interval) < 1 or args.num_generations < 2:
        raise ValueError("invalid training parameters")
    local_rank = init_distributed_mode()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    args.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(42 + rank)
    directory = ROOT / "artifacts/agent" / args.run_name
    if Path(args.run_name).name != args.run_name or Path(args.save_weight).name != args.save_weight:
        raise ValueError("run_name and save_weight must be simple names")
    if args.mode in {"sft", "grpo"} and not args.from_resume and (ROOT / "out" / f"{args.save_weight}_{args.hidden_size}.pth").exists():
        raise FileExistsError("Refusing to overwrite an existing training weight; use a new name or --from_resume")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = min(time.time() + args.max_train_seconds, args.deadline or float("inf"))
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=False)
    source = ROOT / "out" / f"{args.from_weight}_{args.hidden_size}.pth"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    raw = MiniMindForCausalLM(config)
    raw.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=True)
    raw.to(args.device)
    ckpt_dir = ROOT / "checkpoints"
    checkpoint = lm_checkpoint(config, weight=args.save_weight, save_dir=str(ckpt_dir)) if args.from_resume else None
    if args.from_resume and checkpoint is None:
        raise FileNotFoundError("requested resume checkpoint is absent")
    args.data_path = str(args.data_dir / "ocean_agent_train.jsonl")
    manifest = json.loads((args.data_dir / "agent_manifest.json").read_text())
    if manifest["tool_sha256"] != file_hash(ROOT / "agent/ocean.py"):
        raise ValueError("Tool version changed; regenerate Agent data")
    if manifest.get("version") == "ocean-react-v2":
        if args.agent_version != "v2" or manifest["harness_sha256"] != file_hash(ROOT / "agent/react.py"):
            raise ValueError("ReAct data/harness mismatch; regenerate data")
    elif args.agent_version == "v2" and args.mode != "eval":
        raise ValueError("v2 training requires a v2 data manifest")
    for split in manifest["splits"].values():
        for filename, digest in split["artifacts"].items():
            if file_hash(args.data_dir / filename) != digest:
                raise ValueError(f"Data does not match manifest: {filename}")
    metadata = experiment_config(args)
    if args.runtime == "sdk":
        from importlib.metadata import version
        metadata.update(sdk_version=version("openai-agents"), adapter_sha256=file_hash(ROOT / "agent/sdk.py"))
    metadata["device"] = str(args.device)
    metadata.update(input_weight_sha256=file_hash(source), data_manifest_sha256=file_hash(args.data_dir / "agent_manifest.json"),
                    tool_sha256=file_hash(ROOT / "agent/ocean.py"), version=VERSION, dtype="float16", use_moe=False,
                    batch_size_per_rank=1, world_size=world, max_turns=3, temperature=1., top_p=1., top_k=0)
    if args.agent_version == "v2":
        metadata.update(version="ocean-react-v2", max_turns=6, max_calls=4,
                        harness_sha256=file_hash(ROOT / "agent/react.py"))
    if checkpoint:
        if checkpoint["metadata"].get("runtime", "local") != args.runtime:
            raise ValueError("resume runtime mismatch; use --runtime local for legacy checkpoints, or start a new run from weights")
        for key in ("sdk_version", "adapter_sha256"):
            if checkpoint["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"resume SDK mismatch: {key}")
        for key in ("input_weight_sha256", "data_manifest_sha256", "tool_sha256", "world_size", "mode",
                    "accumulation_steps", "max_total_len", "max_gen_len", "num_generations", "beta", "epsilon", "learning_rate"):
            if checkpoint["metadata"][key] != metadata[key]:
                raise ValueError(f"resume configuration mismatch: {key}")
        if checkpoint["metadata"].get("version") != metadata["version"] or checkpoint["metadata"].get("harness_sha256") != metadata.get("harness_sha256"):
            raise ValueError("resume harness mismatch")
        raw.load_state_dict(checkpoint["full_precision_model"], strict=True)
    tracker = None
    if rank == 0:
        atomic_json(directory / "config.json", metadata)
        if args.use_swanlab:
            import swanlab
            tracker = swanlab
            init_swanlab(tracker, checkpoint, args.swanlab_mode, project=args.swanlab_project,
                         name=args.run_name, logdir=str(directory / "swanlog"), config=metadata)

    def log(values, step):
        if rank == 0:
            # Cloud logging is an acceptance requirement; don't silently disable it.
            if tracker:
                tracker.log(values, step=step)
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **values}) + "\n")

    def environment(split):
        if args.agent_version == "v2":
            from agent.react import ReactTools
            return ReactTools(read_jsonl(args.data_dir / f"ocean_agent_corpus_{split}.jsonl"))
        return OceanTools(read_jsonl(args.data_dir / f"ocean_agent_corpus_{split}.jsonl"))

    if args.mode in {"eval", "diagnose"}:
        split = "train" if args.mode == "diagnose" else args.split
        tasks = read_jsonl(args.data_dir / f"ocean_agent_{split}.jsonl")
        limit = args.limit or (32 if args.mode == "diagnose" else len(tasks))
        summary = evaluate(raw, tokenizer, tasks[:limit], environment(split), args, rank, world, directory, "result",
                           args.num_generations if args.mode == "diagnose" else 1)
        log({f"{args.mode}/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}, 0)
        if tracker:
            tracker.finish()
        if world > 1:
            dist.destroy_process_group()
        return

    train_tasks = read_jsonl(args.data_dir / "ocean_agent_train.jsonl")
    val_tasks = read_jsonl(args.data_dir / "ocean_agent_val.jsonl")
    train_env, val_env = environment("train"), environment("val")
    reference = None
    if args.mode == "grpo":
        reference = MiniMindForCausalLM(config)
        reference.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=True)
        reference.to(args.device).eval().requires_grad_(False)
        dataset = train_tasks
    else:
        dataset_class = AgentSFTDataset
        if args.agent_version == "v2":
            from agent.react import ReactSFTDataset
            dataset_class = ReactSFTDataset
        dataset = dataset_class(args.data_dir / "ocean_agent_sft_train.jsonl", tokenizer, args.max_total_len, deterministic=True)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=42)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=args.device.type == "cuda")
    start_epoch = start_micro = global_step = nonzero_updates = 0
    best = -1.
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_micro = checkpoint["epoch"], checkpoint["step"]
        global_step, best = checkpoint["global_step"], checkpoint["best"]
        nonzero_updates = checkpoint["nonzero_updates"]
    model = DDP(raw, device_ids=[local_rank]) if world > 1 else raw
    if checkpoint:
        rng = checkpoint["rng_states"][rank]
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        torch.set_rng_state(rng["torch"])
        if args.device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], args.device)
    total_steps = max(args.max_steps or math.ceil(len(sampler) / args.accumulation_steps) * args.epochs, 1)
    elapsed_before = checkpoint.get("elapsed_seconds", 0) if checkpoint else 0
    if checkpoint:
        deadline = min(deadline, time.time() + max(0, args.max_train_seconds - elapsed_before))

    def expired():
        flag = torch.tensor(int(rank == 0 and time.time() >= deadline), device=args.device)
        if world > 1:
            dist.broadcast(flag, 0)
        return bool(flag.item())

    def save(epoch, micro, suffix="", reason="periodic"):
        numpy_state = np.random.get_state()
        rng = {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state(args.device) if args.device.type == "cuda" else None}
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states = [rng]
        if rank == 0:
            lm_checkpoint(config, weight=args.save_weight + suffix, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=micro, wandb=tracker, save_dir=str(ckpt_dir),
                          global_step=global_step, best=best, rng_states=states, metadata=metadata,
                          nonzero_updates=nonzero_updates, elapsed_seconds=elapsed_before + time.monotonic() - started,
                          full_precision_model={k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}, stop_reason=reason)
            source_path = ckpt_dir / f"{args.save_weight + suffix}_{args.hidden_size}.pth"
            target = ROOT / "out" / source_path.name
            import shutil
            shutil.copyfile(source_path, str(target) + ".tmp")
            os.replace(str(target) + ".tmp", target)
        if world > 1:
            dist.barrier()

    def validation(label):
        nonlocal best
        summary = evaluate(raw, tokenizer, val_tasks, val_env, args, rank, world, directory, label)
        log({f"val/{k}": v for k, v in summary.items() if k != "gate_pass" and isinstance(v, (int, float))}, global_step)
        improved = summary["success"] > best
        best = max(best, summary["success"])
        return improved

    if not args.skip_eval and not checkpoint:
        validation("val_start")
        save(0, 0, "_best", "initial_best")
    optimizer.zero_grad(set_to_none=True)
    stop_reason, last_epoch, last_micro = "epochs_completed", start_epoch, start_micro
    window = MetricWindow()
    last_time = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        skip = start_micro if epoch == start_epoch else 0
        for micro, index in enumerate(indices, 1):
            if micro <= skip:
                continue
            if (micro - 1) % args.accumulation_steps == 0 and expired():
                stop_reason = "time_budget"
                break
            last_epoch, last_micro = epoch, micro
            flush = micro % args.accumulation_steps == 0 or micro == len(indices)
            group_start = ((micro - 1) // args.accumulation_steps) * args.accumulation_steps
            accumulation = min(args.accumulation_steps, len(indices) - group_start)
            if args.mode == "sft":
                ids, labels = [t.unsqueeze(0).to(args.device) for t in dataset[index]]
                end = int(ids[0].ne(tokenizer.pad_token_id).nonzero()[-1]) + 1
                ids, labels = ids[:, :end], labels[:, :end]
                with (model.no_sync() if world > 1 and not flush else nullcontext()):
                    with torch.autocast(args.device.type, dtype=torch.float16, enabled=args.device.type == "cuda"):
                        loss = model(ids, labels=labels).loss
                    scaler.scale(loss / accumulation).backward()
                window.add("loss", loss, int((labels[:, 1:] != -100).sum()))
            else:
                # Even a rank with only context-limited (zero-action) trajectories
                # must reduce exactly the same metric keys as the other ranks.
                for key in ("policy_loss", "kl", "grpo_loss", "clip_fraction"):
                    window.add(key, 0., 0.)
                raw.eval()
                generator = model_generator(raw, tokenizer, args.device)
                task = train_tasks[index]
                traces = [run_trajectory(task, tokenizer, train_env, generator, max_total_len=args.max_total_len,
                                         max_new_tokens=args.max_gen_len) for _ in range(args.num_generations)]
                rewards = torch.tensor([t["metrics"]["reward"] for t in traces], device=args.device)
                advantages, stds = group_advantages(rewards, args.num_generations)
                raw.train()
                window.add("group_std", stds.mean())
                window.add("degenerate_group_ratio", (stds < 1e-6).float().mean())
                for i, trace in enumerate(traces):
                    for key, value in trace["metrics"].items():
                        window.add(key, value)
                    token_count = sum(len(r["completion_ids"]) for r in trace["rounds"])
                    window.add("generated_tokens", token_count)
                    # Fixed three forwards per trajectory on every rank. Dummy rounds
                    # carry zero loss but keep DDP's final synchronization aligned.
                    for turn in range(3):
                        actual = turn < len(trace["rounds"])
                        round_ = trace["rounds"][turn] if actual else {"input_ids": [tokenizer.bos_token_id], "completion_ids": [tokenizer.eos_token_id]}
                        synchronize = flush and i == args.num_generations - 1 and turn == 2
                        with (model.no_sync() if world > 1 and not synchronize else nullcontext()):
                            with torch.autocast(args.device.type, dtype=torch.float16, enabled=args.device.type == "cuda"):
                                new = round_logps(model, round_, args.device)
                                with torch.no_grad():
                                    ref = round_logps(reference, round_, args.device) if actual else new.detach()
                                old = torch.tensor([round_["old_logps"]], device=args.device) if actual else new.detach()
                                assert new.shape == old.shape
                                m = grpo_objective(new, old, ref, torch.ones_like(new), advantages[i:i + 1], args.beta, args.epsilon)
                                weight = len(round_["completion_ids"]) / max(token_count, 1) if actual else 0.
                                loss = m["grpo_loss"] * weight / args.num_generations / accumulation
                            scaler.scale(loss).backward()
                        if actual:
                            for key, value in m.items():
                                window.add(key, value, weight)
                with (directory / f"train_rank{rank}.jsonl").open("a", encoding="utf-8") as f:
                    for trace in traces:
                        f.write(json.dumps({"step": global_step, **public_trace(trace)}, ensure_ascii=False) + "\n")
            if not flush:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.)
            finite = reduce_sums([float(torch.isfinite(norm))], args.device)[0] == world
            if not finite:
                raise FloatingPointError("non-finite gradient; long-run gate failed")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            nonzero_updates += int(norm > 0)
            lr = args.learning_rate * (.1 + .9 * (1 + math.cos(math.pi * min(global_step / total_steps, 1))) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window.add("grad_norm", norm)
            window.add("learning_rate", lr)
            summary = window.means(args.device)
            summary["seconds_per_update"] = time.monotonic() - last_time
            last_time = time.monotonic()
            if args.mode == "grpo":
                summary["tool_valid_rate"] = summary["valid_calls"] / max(summary["attempted_calls"], 1e-9)
                summary["tokens_per_second"] = summary["generated_tokens"] * args.num_generations * accumulation * world / summary["seconds_per_update"]
            summary["processed_questions"] = (epoch * len(indices) + micro) * world
            log({f"train/{k}": v for k, v in summary.items()}, global_step)
            if rank == 0:
                print(f"update={global_step} epoch={epoch + 1} micro={micro} {json.dumps(summary)}", flush=True)
            if args.verify_sync:
                digest = hashlib.sha256()
                for parameter in raw.parameters():
                    digest.update(parameter.detach().cpu().numpy().tobytes())
                hashes = [None] * world
                if world > 1:
                    dist.all_gather_object(hashes, digest.hexdigest())
                    assert len(set(hashes)) == 1, "DDP model parameters diverged"
            if args.max_steps and global_step >= args.max_steps:
                stop_reason = "max_steps"
                break
            if expired():
                stop_reason = "time_budget"
                break
            if not args.skip_eval and global_step % args.eval_interval == 0:
                if validation(f"val_step{global_step}"):
                    save(epoch, micro, "_best", "best_validation")
            if global_step % args.save_interval == 0:
                save(epoch, micro)
        start_micro = 0
        if stop_reason != "epochs_completed":
            break
        save(epoch, len(indices))
    save(last_epoch, last_micro, reason=stop_reason)
    if not args.skip_eval:
        if validation("val_final"):
            save(last_epoch, last_micro, "_best", "best_validation")
        if args.mode == "grpo":
            test = evaluate(raw, tokenizer, read_jsonl(args.data_dir / "ocean_agent_test.jsonl"), environment("test"),
                            args, rank, world, directory, "test_final")
            log({f"test/{k}": v for k, v in test.items() if k != "gate_pass"}, global_step)
    if rank == 0:
        atomic_json(directory / "completion.json", {"global_step": global_step, "stop_reason": stop_reason,
                    "nonzero_updates": nonzero_updates, "epoch": last_epoch, "micro_step": last_micro,
                    "elapsed_seconds": elapsed_before + time.monotonic() - started, "world_size": world,
                    "sync_verified": args.verify_sync, "input_weight_sha256": metadata["input_weight_sha256"]})
    if tracker:
        tracker.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
