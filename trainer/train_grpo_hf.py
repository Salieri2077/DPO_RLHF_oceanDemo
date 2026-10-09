#!/usr/bin/env python3
"""GRPO for the Qwen3-1.7B Ocean SFT model with an LLM-judge reward.

Policy = merged SFT model + a fresh LoRA; reference = the same model with the LoRA disabled, so the KL term keeps
the policy near the SFT model without a second copy in memory. Each rank takes one prompt per step and samples
`num_generations` answers (no truncation at the default 1024 new tokens). The judge scores the whole group in one
request with the candidates shuffled; group-normalised scores are the advantages. Groups whose judge scores are
flat (std below --flat_std) get no update, so small shaping noise is never amplified. One update per rollout
(on-policy): the ratio is exp(logp - logp.detach()), so PPO clipping is moot and not used. LoRA gradients are
summed across ranks and divided by the number of ranks with an active group.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from trainer.hf_chat import generation_prompt, load_tokenizer
from trainer.train_agent import atomic_json
from trainer.trainer_utils import SiliconFlowRewardModel, experiment_config, init_distributed_mode, init_swanlab, reduce_sums, setup_seed

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DATA = ROOT / "data/processed"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_sft_r64_eot_merged")
    p.add_argument("--save_adapter", required=True, help="adapter name under out/hf/ (writes _last and _final)")
    p.add_argument("--run_name", required=True)
    p.add_argument("--resume", action="store_true", help="continue from artifacts/grpo_hf/<run_name>/state.pt")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--micro_seqs", type=int, default=4, help="sequences per forward/backward chunk")
    p.add_argument("--loss_chunk", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine",
                   help="cosine: warmup then cosine to 10%% at max_steps; wsd: warmup, constant, cosine to 10%% over the "
                        "last --decay_steps (lets a run be extended without re-planning its length)")
    p.add_argument("--schedule_start", type=int, default=0, help="step where (re)warmup starts, e.g. the resumed step")
    p.add_argument("--decay_steps", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.04, help="KL(policy || SFT reference) coefficient")
    p.add_argument("--flat_std", type=float, default=0.25, help="judge-score std below which a group is skipped")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--reward_model", default="Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--max_judge_failures", type=int, default=40, help="cumulative failed groups before stopping")
    p.add_argument("--swanlab_id", help="resume this SwanLab run (default: the id stored in state.pt)")
    p.add_argument("--val_questions", type=int, default=16)
    p.add_argument("--val_generations", type=int, default=4)
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_swanlab", action="store_true")
    p.add_argument("--swanlab_mode", choices=["cloud", "offline", "disabled"], default="cloud")
    p.add_argument("--swanlab_project", default="OceanHeart-GRPO")
    return p


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["conversations"] for line in f]


def prompt_pool():
    """Ocean SFT-val questions the SFT model never trained on: drop the first 512 (SFT validation loss) and the
    200 GRPO-val questions, deduplicate by question."""
    held = {c[-2]["content"] for c in read_jsonl(DATA / "ocean_grpo_val.jsonl")}
    pool, seen = [], set()
    for conversations in read_jsonl(DATA / "ocean_sft_val.jsonl")[512:]:
        question = conversations[-2]["content"]
        if question not in held and question not in seen and len(conversations) == 3:
            seen.add(question)
            pool.append(conversations)
    return pool


def completion_logprobs(model, ids, mask, start, chunk):
    """Log-probabilities of ids[:, start:], projecting to the vocabulary in checkpointed chunks."""
    core = model.get_base_model()
    hidden = core.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, start - 1:-1]
    batch, length, width = hidden.shape
    flat_hidden, flat_targets = hidden.reshape(-1, width), ids[:, start:].reshape(-1)

    def piece(h, t):
        logits = core.lm_head(h).float()
        return logits.gather(1, t[:, None]).squeeze(1) - logits.logsumexp(-1)

    parts = []
    for s in range(0, flat_targets.numel(), chunk):
        h, t = flat_hidden[s:s + chunk], flat_targets[s:s + chunk]
        parts.append(checkpoint(piece, h, t, use_reentrant=False) if torch.is_grad_enabled() else piece(h, t))
    return torch.cat(parts).view(batch, length)


@torch.no_grad()
def sample(model, tokenizer, conversations, n, args, stop, device):
    model.eval()
    inputs = tokenizer(generation_prompt(conversations), return_tensors="pt", add_special_tokens=False).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model.generate(input_ids=inputs.input_ids.repeat(n, 1), attention_mask=inputs.attention_mask.repeat(n, 1),
                                do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, use_cache=True,
                                max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop)
    model.train()
    start = inputs.input_ids.shape[1]
    completion = output[:, start:]
    is_stop = torch.zeros_like(completion, dtype=torch.bool)
    for token in stop:
        is_stop |= completion.eq(token)
    ended = is_stop.any(1)
    first = torch.where(ended, is_stop.int().argmax(1), torch.full_like(ended, completion.shape[1] - 1, dtype=torch.long))
    completion_mask = torch.arange(completion.shape[1], device=device)[None] <= first[:, None]  # includes the stop token
    texts = [tokenizer.decode(row[:int(end) + (0 if done else 1)], skip_special_tokens=True)
             for row, end, done in zip(completion, first, ended)]
    full_mask = torch.cat([inputs.attention_mask.repeat(n, 1), completion_mask.long()], 1)
    return output, full_mask, completion_mask, start, texts, ended


def judge_group(judge, conversations, texts, rng):
    order = list(range(len(texts)))
    rng.shuffle(order)  # spreads the judge's position bias over candidates instead of fixing it to sample index
    scores = judge.score_group(conversations[-2]["content"], conversations[-1]["content"], [texts[i] for i in order])
    restored = [0.] * len(texts)
    for position, index in enumerate(order):
        restored[index] = scores[position]
    return restored


def main():
    args = parser().parse_args()
    import os
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is required")
    if Path(args.run_name).name != args.run_name or Path(args.save_adapter).name != args.save_adapter:
        raise ValueError("run_name and save_adapter must be simple names")
    directory = ROOT / "artifacts/grpo_hf" / args.run_name
    if not args.resume and (directory / "state.pt").exists():
        raise FileExistsError("run exists; pass --resume or use a new --run_name")
    local_rank = init_distributed_mode()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    device = torch.device(f"cuda:{local_rank}")
    setup_seed(args.seed + rank)
    directory.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path)
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    pool = prompt_pool()
    random.Random(args.seed).shuffle(pool)
    val = read_jsonl(DATA / "ocean_grpo_val.jsonl")[:args.val_questions]

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
    scaler.scale(torch.zeros((), device=device))  # initialise the scale on every rank, even one whose first group is flat
    judge = SiliconFlowRewardModel(api_key, args.reward_model)
    step, judge_failures, swanlab_id = 0, 0, args.swanlab_id
    if args.resume:
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, judge_failures = state["step"], state["judge_failures"]
        swanlab_id = swanlab_id or state.get("swanlab_id")

    def learning_rate(s):
        warmup = min(1., max(s - args.schedule_start, 0) / max(args.warmup_steps, 1))
        if args.schedule == "cosine":
            progress = min(s / args.max_steps, 1)
        else:  # warmup-stable-decay
            progress = min(max(s - (args.max_steps - args.decay_steps), 0) / max(args.decay_steps, 1), 1)
        return args.learning_rate * warmup * (.1 + .9 * (1 + math.cos(math.pi * progress)) / 2)

    metadata = experiment_config(argparse.Namespace(**vars(args), data_path=str(DATA / "ocean_sft_val.jsonl")))
    metadata.update(world_size=world, prompt_pool=len(pool), prompts_per_step=world, dtype="float16 base, float32 LoRA",
                    trainable_parameters=sum(p.numel() for p in trainable), reference="merged SFT model, LoRA disabled")
    tracker = None
    if rank == 0:
        atomic_json(directory / "config.json", metadata)
        print(json.dumps({k: metadata[k] for k in ("prompt_pool", "trainable_parameters", "world_size")}), flush=True)
        if args.use_swanlab:
            import swanlab
            tracker = swanlab
            run = init_swanlab(tracker, {"swanlab_id": swanlab_id} if swanlab_id else None, args.swanlab_mode,
                               project=args.swanlab_project, name=args.run_name, logdir=str(directory / "swanlog"), config=metadata)
            swanlab_id = getattr(getattr(run, "public", None), "cuid", None) or swanlab_id

    def log(values, at):
        if rank == 0:
            if tracker:
                tracker.log(values, step=at)
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": at, **values}) + "\n")

    def save(suffix):
        if rank == 0:
            target = ROOT / "out/hf" / f"{args.save_adapter}{suffix}"
            model.save_pretrained(target)
            tokenizer.save_pretrained(target)
            from peft import get_peft_model_state_dict
            torch.save({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "judge_failures": judge_failures,
                        "swanlab_id": swanlab_id},
                       directory / "state.pt.tmp")
            (directory / "state.pt.tmp").replace(directory / "state.pt")
        if world > 1:
            dist.barrier()

    def usage_totals():
        return reduce_sums([judge.usage["requests"], judge.usage["prompt_tokens"], judge.usage["completion_tokens"],
                            judge.usage["rate_limited"]], device)

    def validate():
        torch.cuda.empty_cache()
        rng = random.Random(f"val:{step}:{rank}")
        scores, tokens, ends, failures = [], 0, 0, 0
        for conversations in val[rank::world]:
            _, _, completion_mask, _, texts, ended = sample(model, tokenizer, conversations, args.val_generations, args, stop, device)
            tokens += int(completion_mask.sum())
            ends += int(ended.sum())
            try:
                group = judge_group(judge, conversations, texts, rng)
                scores += group
            except RuntimeError:
                group, failures = None, failures + 1
            with (directory / f"val_rank{rank}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "question": conversations[-2]["content"][:80], "scores": group,
                                    "tokens": completion_mask.sum(1).tolist()}, ensure_ascii=False) + "\n")
        total, count, tokens, ends, failures = reduce_sums([sum(scores), len(scores), tokens, ends, failures], device)
        samples = len(val) * args.val_generations
        values = {"val/reward_mean": total / max(count, 1), "val/response_tokens": tokens / samples,
                  "val/eos_rate": ends / samples, "val/judge_failures": failures}
        log(values, step)
        if rank == 0:
            print(f"validation step={step} {json.dumps(values)}", flush=True)

    if step == 0:
        validate()
    started = time.monotonic()
    while step < args.max_steps:
        conversations = pool[(step * world + rank) % len(pool)]
        t0 = time.monotonic()
        output, full_mask, completion_mask, start, texts, ended = sample(model, tokenizer, conversations, args.num_generations, args, stop, device)
        t1 = time.monotonic()
        try:
            rewards = torch.tensor(judge_group(judge, conversations, texts, random.Random(f"{step}:{rank}")), device=device)
            failed = 0
        except RuntimeError as error:
            print(f"rank={rank} step={step} judge failed: {error}", flush=True)
            rewards, failed = torch.zeros(args.num_generations, device=device), 1
        t2 = time.monotonic()
        std = rewards.std(unbiased=False)
        active = int(not failed and std.item() >= args.flat_std)
        advantages = (rewards - rewards.mean()) / (std + 1e-4) if active else torch.zeros_like(rewards)
        kl_sum = policy_sum = 0.
        mask = completion_mask.float()
        denominator = mask.sum().clamp_min(1)
        if active:
            for s in range(0, args.num_generations, args.micro_seqs):
                part = slice(s, s + args.micro_seqs)
                ids, attention, cmask = output[part], full_mask[part], mask[part]
                with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.float16):
                    reference = completion_logprobs(model, ids, attention, start, args.loss_chunk)
                with torch.autocast("cuda", dtype=torch.float16):
                    logp = completion_logprobs(model, ids, attention, start, args.loss_chunk)
                ratio = torch.exp(logp - logp.detach())
                policy = -ratio * advantages[part, None]
                log_ratio = reference - logp
                kl = torch.exp(log_ratio) - log_ratio - 1
                loss = ((policy + args.beta * kl) * cmask).sum() / denominator
                scaler.scale(loss).backward()
                kl_sum += float((kl.detach() * cmask).sum())
                policy_sum += float((policy.detach() * cmask).sum())
        t3 = time.monotonic()
        # Average LoRA gradients over ranks that trained this step (flat or failed groups contribute nothing).
        active_ranks = reduce_sums([active], device)[0]
        for parameter in trainable:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        if world > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in trainable])
            dist.all_reduce(flat)
            offset = 0
            for p in trainable:
                p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
                offset += p.numel()
        step += 1
        norm = torch.zeros((), device=device)
        if active_ranks:
            for p in trainable:
                p.grad.div_(active_ranks)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        judge_failures += int(reduce_sums([failed], device)[0])
        sums = reduce_sums([float(rewards.mean()), float(std), active, failed, float(mask.sum()), float(ended.sum()),
                            kl_sum, policy_sum, float(denominator) if active else 0., t1 - t0, t2 - t1, t3 - t2], device)
        (reward, group_std, actives, failures, tokens, ends, kl_total, policy_total, active_tokens,
         rollout_s, judge_s, train_s) = sums
        requests, prompt_tokens, completion_tokens, rate_limited = usage_totals()
        values = {"train/reward_mean": reward / world, "train/group_std": group_std / world,
                  "train/active_group_ratio": actives / world, "train/judge_failures_step": failures,
                  "train/response_tokens": tokens / (world * args.num_generations),
                  "train/eos_rate": ends / (world * args.num_generations),
                  "train/kl": kl_total / max(active_tokens, 1), "train/policy_term": policy_total / max(active_tokens, 1),
                  "train/grad_norm": float(norm), "train/learning_rate": learning_rate(step), "train/loss_scale": scaler.get_scale(),
                  "train/rollout_seconds": rollout_s / world, "train/judge_seconds": judge_s / world, "train/train_seconds": train_s / world,
                  "train/step_seconds": time.monotonic() - t0, "judge/requests": requests,
                  "judge/prompt_tokens": prompt_tokens, "judge/completion_tokens": completion_tokens,
                  "judge/rate_limited": rate_limited}
        log(values, step)
        if rank == 0:
            print(f"step={step}/{args.max_steps} " + json.dumps({k.split('/')[-1]: float(f"{v:.4g}") for k, v in values.items()}), flush=True)
        if judge_failures > args.max_judge_failures:
            save("_last")
            raise RuntimeError(f"judge failed {judge_failures} times; state saved, fix the API and --resume")
        if step % args.eval_interval == 0:
            validate()
        if step % args.save_interval == 0:
            save("_last")
    if step % args.eval_interval:
        validate()
    save("_final")
    requests, prompt_tokens, completion_tokens, _ = usage_totals()  # collective: every rank must call it
    if rank == 0:
        atomic_json(directory / "completion.json", {"step": step, "elapsed_seconds": time.monotonic() - started,
                                                    "judge_requests": requests, "judge_prompt_tokens": prompt_tokens,
                                                    "judge_completion_tokens": completion_tokens, "judge_failures": judge_failures})
    if tracker:
        tracker.finish()
    if world > 1:
        torch.cuda.empty_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
