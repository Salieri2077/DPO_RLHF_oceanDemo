#!/usr/bin/env python3
"""Tool SFT / evaluation of a Hugging Face backbone with LoRA on the ReAct data.

Same data, prompt text (chat template, system prompt, tool schema), harness, scorer and OpenAI Agents SDK
runtime as train_agent.py; only the backbone and its tokenizer change, so results compare question by
question with the MiniMind agent. The frozen base stays fp16; LoRA weights train in fp32 under autocast.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent.ocean import file_hash, read_jsonl
from agent.react import ReactSFTDataset, ReactTools
from trainer.train_agent import atomic_json, evaluate
from trainer.trainer_utils import MetricWindow, experiment_config, init_distributed_mode, init_swanlab, reduce_sums, setup_seed

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PLAIN_TEXT_TAGS = {"<tool_call>", "</tool_call>"}  # tags the model must write; see agent_tokenizer
ASSISTANT_END = "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"


def agent_tokenizer(model_path):
    """Qwen tokenizer with the MiniMind agent chat template, so the prompt text matches the MiniMind agent.

    Instruct tokenizers end turns with their eos, <|im_end|>. A Base model's eos is <|endoftext|>, and its <|im_end|>,
    <tool_call> and </tool_call> embeddings were never trained: they equal ~4.8K unused rows, and with tied embeddings
    the model cannot emit one of them rather than its twins (see trainer/hf_chat.py). For a Base tokenizer, assistant
    turns therefore end with its eos, and the tool-call tags are tokenized as ordinary text pieces. The text the model
    reads and writes is unchanged.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template = AutoTokenizer.from_pretrained(ROOT / "model").chat_template
    if tokenizer.eos_token != "<|im_end|>":
        if template.count(ASSISTANT_END) != 1:
            raise ValueError("MiniMind chat template changed; update ASSISTANT_END")
        template = template.replace(ASSISTANT_END, ASSISTANT_END.replace("<|im_end|>", tokenizer.eos_token))
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        spec = json.loads(tokenizer.backend_tokenizer.to_str())
        spec["added_tokens"] = [token for token in spec["added_tokens"] if token["content"] not in PLAIN_TEXT_TAGS]
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(spec)),
                                            eos_token=tokenizer.eos_token, pad_token=tokenizer.pad_token)
    tokenizer.chat_template = template
    return tokenizer


class ThinkingTokenizer:
    """Qwen3 Instruct in thinking mode: generations start with <think>...</think>. The harness parses decoded text,
    so the reasoning block is removed after decoding (Qwen's own template also drops it from earlier turns)."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return re.sub(r"^\s*<think>.*?</think>\s*", "", self._tokenizer.decode(*args, **kwargs), count=1, flags=re.S)


def native_tokenizer(model_path, thinking):
    """Zero-shot reference for Instruct models: their own chat template (tools listed in the model's native format),
    non-thinking unless `thinking`; the system prompt, protocol, tools and scorer are those of the agent."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError("the native chat template is only for Instruct tokenizers")
    if thinking:
        return ThinkingTokenizer(tokenizer)
    tokenizer.chat_template = "{%- set enable_thinking = false %}\n" + tokenizer.chat_template
    return tokenizer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sft", "eval"], required=True)
    p.add_argument("--model_path", type=Path, required=True, help="Hugging Face causal LM directory")
    p.add_argument("--adapter", type=Path, help="eval: LoRA adapter directory (omit to evaluate the base)")
    p.add_argument("--chat_template", choices=["minimind", "native", "native_think"], default="minimind",
                   help="eval: the MiniMind agent template (training format) or the Instruct model's own template")
    p.add_argument("--save_adapter", help="sft: adapter name under out/hf/ (writes <name>_best and <name>_final)")
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--max_gen_len", type=int, default=192)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_train_seconds", type=float, default=36000)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=.05)
    p.add_argument("--eval_interval", type=int, default=150)
    p.add_argument("--skip_eval", action="store_true", help="engineering smoke only")
    p.add_argument("--verify_sync", action="store_true")
    p.add_argument("--sample_eval", action="store_true")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--use_swanlab", action="store_true")
    p.add_argument("--swanlab_mode", choices=["cloud", "offline", "disabled"], default="cloud")
    p.add_argument("--swanlab_project", default="OceanHeart-Agent")
    return p


def check_data(data_dir, training):
    """The data contract of train_agent.py: ReAct v2 protocol with unchanged tools and harness."""
    manifest = json.loads((data_dir / "agent_manifest.json").read_text())
    if manifest["tool_sha256"] != file_hash(ROOT / "agent/ocean.py"):
        raise ValueError("Tool version changed; regenerate Agent data")
    if manifest.get("version") == "ocean-react-v2":
        if manifest["harness_sha256"] != file_hash(ROOT / "agent/react.py"):
            raise ValueError("ReAct data/harness mismatch; regenerate data")
    elif training:
        raise ValueError("training requires a ReAct v2 data manifest")
    for split in manifest["splits"].values():
        for filename, digest in split["artifacts"].items():
            if file_hash(data_dir / filename) != digest:
                raise ValueError(f"Data does not match manifest: {filename}")


def main():
    args = parser().parse_args()
    args.agent_version, args.runtime = "v2", "sdk"  # evaluate() reads these
    import agent.sdk  # Use the SDK venv; fail before loading weights.
    if Path(args.run_name).name != args.run_name or (args.save_adapter and Path(args.save_adapter).name != args.save_adapter):
        raise ValueError("run_name and save_adapter must be simple names")
    if args.mode == "sft":
        if not args.save_adapter:
            raise ValueError("--save_adapter is required for sft")
        if any((ROOT / "out/hf" / f"{args.save_adapter}{suffix}").exists() for suffix in ("_best", "_final")):
            raise FileExistsError("Refusing to overwrite an existing adapter; use a new --save_adapter")
    local_rank = init_distributed_mode()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    args.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(42 + rank)
    directory = ROOT / "artifacts/agent" / args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    args.data_path = str(args.data_dir / "ocean_agent_train.jsonl")
    check_data(args.data_dir, args.mode == "sft")

    if args.chat_template != "minimind" and args.mode != "eval":
        raise ValueError("--chat_template native/native_think is an evaluation-only reference")
    tokenizer = (agent_tokenizer(args.model_path) if args.chat_template == "minimind"  # same prompt text as MiniMind
                 else native_tokenizer(args.model_path, args.chat_template == "native_think"))
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(args.device)
    from peft import LoraConfig, PeftModel, get_peft_model
    if args.mode == "eval":
        model = PeftModel.from_pretrained(model, args.adapter) if args.adapter else model
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.config.use_cache = False
        model.train()  # from_pretrained returns eval mode, where checkpointing and LoRA dropout are off
    trainable = [p for p in model.parameters() if p.requires_grad]

    metadata = experiment_config(args)
    metadata.update(device=str(args.device), world_size=world, dtype="float16 base, float32 LoRA", runtime="sdk",
                    version="ocean-react-v2", max_turns=6, max_calls=4, harness_sha256=file_hash(ROOT / "agent/react.py"),
                    tool_sha256=file_hash(ROOT / "agent/ocean.py"), data_manifest_sha256=file_hash(args.data_dir / "agent_manifest.json"),
                    chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    model_config_sha256=file_hash(args.model_path / "config.json"), batch_size_per_rank=1,
                    trainable_parameters=sum(p.numel() for p in trainable),
                    adapter_sha256=file_hash(args.adapter / "adapter_model.safetensors") if args.adapter else None)
    tracker = None
    if rank == 0:
        atomic_json(directory / "config.json", metadata)
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

    def environment(split):
        return ReactTools(read_jsonl(args.data_dir / f"ocean_agent_corpus_{split}.jsonl"))

    def finish():
        if tracker:
            tracker.finish()
        if world > 1:
            torch.cuda.empty_cache()  # NCCL shutdown allocates outside PyTorch's cache on near-full cards
            dist.destroy_process_group()

    if args.mode == "eval":
        tasks = read_jsonl(args.data_dir / f"ocean_agent_{args.split}.jsonl")
        summary = evaluate(model, tokenizer, tasks[:args.limit or len(tasks)], environment(args.split), args, rank, world,
                           directory, "result")
        log({f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}, 0)
        return finish()

    dataset = ReactSFTDataset(args.data_dir / "ocean_agent_sft_train.jsonl", tokenizer, args.max_total_len)
    val_tasks, val_env = read_jsonl(args.data_dir / "ocean_agent_val.jsonl"), environment("val")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=42)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=args.device.type == "cuda")
    wrapped = DDP(model, device_ids=[local_rank]) if world > 1 else model
    total_steps = max(args.max_steps or math.ceil(len(sampler) / args.accumulation_steps) * args.epochs, 1)
    deadline, started = time.time() + args.max_train_seconds, time.monotonic()
    global_step = nonzero_updates = 0
    best = -1.

    def learning_rate(step):  # short linear warmup, then train_agent.py's cosine decay to 10%
        return args.learning_rate * min(1., step / max(args.warmup_steps, 1)) * (
            .1 + .9 * (1 + math.cos(math.pi * min(step / total_steps, 1))) / 2)

    def expired():
        flag = torch.tensor(int(rank == 0 and time.time() >= deadline), device=args.device)
        if world > 1:
            dist.broadcast(flag, 0)
        return bool(flag.item())

    def save(suffix):
        if rank == 0:
            model.save_pretrained(ROOT / "out/hf" / f"{args.save_adapter}{suffix}")  # LoRA weights only
        if world > 1:
            dist.barrier()

    def validation(label):
        nonlocal best
        torch.cuda.empty_cache()
        summary = evaluate(model, tokenizer, val_tasks, val_env, args, rank, world, directory, label)
        log({f"val/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}, global_step)
        improved = summary["success"] > best
        best = max(best, summary["success"])
        if improved:
            save("_best")

    if not args.skip_eval:
        validation("val_start")
    optimizer.zero_grad(set_to_none=True)
    stop_reason, last_epoch, last_micro = "epochs_completed", 0, 0
    window, last_time = MetricWindow(), time.monotonic()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        for micro, index in enumerate(indices, 1):
            if (micro - 1) % args.accumulation_steps == 0 and expired():
                stop_reason = "time_budget"
                break
            last_epoch, last_micro = epoch, micro
            flush = micro % args.accumulation_steps == 0 or micro == len(indices)
            accumulation = min(args.accumulation_steps, len(indices) - ((micro - 1) // args.accumulation_steps) * args.accumulation_steps)
            ids, labels = [t.unsqueeze(0).to(args.device) for t in dataset[index]]
            end = int(ids[0].ne(tokenizer.pad_token_id).nonzero()[-1]) + 1
            ids, labels = ids[:, :end], labels[:, :end]
            with (wrapped.no_sync() if world > 1 and not flush else nullcontext()):
                with torch.autocast(args.device.type, dtype=torch.float16, enabled=args.device.type == "cuda"):
                    loss = wrapped(input_ids=ids, labels=labels).loss
                scaler.scale(loss / accumulation).backward()
            window.add("loss", loss, int((labels[:, 1:] != -100).sum()))
            if not flush:
                continue
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(global_step + 1)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if reduce_sums([float(torch.isfinite(norm))], args.device)[0] != world:
                raise FloatingPointError("non-finite gradient")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            nonzero_updates += int(norm > 0)
            window.add("grad_norm", norm)
            window.add("learning_rate", optimizer.param_groups[0]["lr"])
            summary = window.means(args.device)
            summary.update(seconds_per_update=time.monotonic() - last_time,
                           processed_questions=(epoch * len(indices) + micro) * world,
                           peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2 ** 30 if args.device.type == "cuda" else 0)
            last_time = time.monotonic()
            log({f"train/{k}": v for k, v in summary.items()}, global_step)
            if rank == 0:
                print(f"update={global_step} epoch={epoch + 1} micro={micro} {json.dumps(summary)}", flush=True)
            if args.verify_sync:
                digest = hashlib.sha256()
                for parameter in trainable:
                    digest.update(parameter.detach().cpu().numpy().tobytes())
                hashes = [None] * world
                if world > 1:
                    dist.all_gather_object(hashes, digest.hexdigest())
                    assert len(set(hashes)) == 1, "DDP LoRA parameters diverged"
            if args.max_steps and global_step >= args.max_steps:
                stop_reason = "max_steps"
                break
            if expired():
                stop_reason = "time_budget"
                break
            if not args.skip_eval and global_step % args.eval_interval == 0:
                validation(f"val_step{global_step}")
        if stop_reason != "epochs_completed":
            break
    save("_final")
    if not args.skip_eval and global_step % args.eval_interval:
        validation("val_final")
    if rank == 0:
        atomic_json(directory / "completion.json", {
            "global_step": global_step, "stop_reason": stop_reason, "nonzero_updates": nonzero_updates,
            "epoch": last_epoch, "micro_step": last_micro, "elapsed_seconds": time.monotonic() - started,
            "world_size": world, "sync_verified": args.verify_sync, "best_val_success": best})
    finish()


if __name__ == "__main__":
    main()
