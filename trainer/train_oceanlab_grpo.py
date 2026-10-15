#!/usr/bin/env python3
"""Multi-turn agent GRPO / RLOO on OceanDepths requests (M4): GRPO from C0, Plain-SFT + GRPO from C1_plain, SkillRL
from C1 (design sections 6-8).

Policy and reference: the Instruct model with the start checkpoint's LoRA adapter twice, "default" trained further and
"reference" frozen, so the KL term keeps each method near its own start. LoRA dropout is removed so that the training
forward is the sampling distribution.

One step: every rank takes --groups_per_rank training requests and samples --group_size complete episodes of each
through the SDK harness (frozen budgets, temperature 1); all of a rank's episodes run at once with batched generation
(trainer/oceanlab_rollout.py). Reward is the scorer's r = S - 0.1 x protocol violation. Advantages per group: GRPO
(r - mean) / (std + 1e-4), or RLOO r_i - mean of the other rewards. Groups with a constant reward have no advantage and
are not trained. Every generated round of an episode is one training sample carrying the episode's advantage: the
exact prompt it was sampled from (input_ids) and its completion. Loss per completion token: -A x ratio + beta x k3
KL(policy || reference), averaged over all completion tokens of the step on all ranks (gradients summed across ranks).
One update per rollout batch (on-policy: ratio = exp(logp - logp.detach()), so clipping would be a no-op).

--skills retrieves a SkillBank into every episode's prompt (SkillRL); one bank snapshot serves a whole step. Every
--eval_interval steps the policy runs greedy episodes on the dev requests (batched); the frozen sequential evaluation
remains scripts/run_oceanlab_eval.py. With --evolve, the validation also drives SkillBank evolution
(agent/oceanlab/evolve.py): weak categories get at most --evolve_max_new teacher-written skills from the policy's
failed validation episodes, each kept only if a greedy rerun with it fixes more of the category's verification
requests than it breaks (--no_evolve_verify skips that check); the new bank version takes effect from the next step on
every rank and is saved with each adapter (skillbank.json) and as artifacts/oceanlab/m4/<run_name>/skills_step<N>.json. Adapters: out/hf/<save_adapter>_{last,stepN,final}; logs and rollouts:
artifacts/oceanlab/m4/<run_name>/. Launch with torchrun, one process per GPU.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from agent.ocean import file_hash, read_jsonl
from agent.oceanlab.data import OceanDepths
from agent.oceanlab.env import OceanLab, OceanLabHarness
from agent.oceanlab.scoring import Scorer
from agent.sdk import run_trajectory
from scripts.run_oceanlab_eval import compact, summarize
from trainer.oceanlab_rollout import BatchedEpisodes, hf_batch_generator
from trainer.qwen_sdpa import memory_efficient_attention
from trainer.rl_schedule import scheduled_lr
from trainer.train_agent import atomic_json
from trainer.train_agent_hf import native_tokenizer
from trainer.trainer_utils import init_distributed_mode, init_swanlab, reduce_sums, setup_seed

DATA = ROOT / "data/processed/oceanlab/v2"
BUDGET = dict(max_turns=12, max_calls=10, max_total_len=12288, max_new_tokens=512)  # frozen after M2
STOPS = ("final", "call_limit", "turn_limit", "context_limit", "generation_limit", "repeated_action", "empty_generation")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=Path, default=Path("/data/anhuang/oceanheart_models/qwen3-1.7b"))
    p.add_argument("--start_adapter", type=Path, required=True, help="start checkpoint (C0, C1 or C1_plain LoRA); also the KL reference")
    p.add_argument("--skills", type=Path, help="SkillBank JSON retrieved into every prompt (SkillRL)")
    p.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    p.add_argument("--requests", type=Path, default=DATA / "oceanlab_train_v2.jsonl")
    p.add_argument("--val_requests", type=Path, default=DATA / "oceanlab_dev_v2.jsonl")
    p.add_argument("--val_limit", type=int, default=0, help="first N dev requests for validation (0 = all)")
    p.add_argument("--save_adapter", required=True, help="adapter name under out/hf/")
    p.add_argument("--run_name", required=True)
    p.add_argument("--resume", action="store_true", help="continue from artifacts/oceanlab/m4/<run_name>/state.pt")
    p.add_argument("--estimator", choices=["grpo", "rloo"], default="grpo")
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--groups_per_rank", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant",
                   help="constant: warmup then constant (verl / SkillRL default); cosine: warmup then cosine to 10%%")
    p.add_argument("--beta", type=float, default=0.01, help="KL(policy || start adapter) coefficient")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--token_budget", type=int, default=32768,
                   help="rows x (longest prompt + new-token limit) per generation sub-batch (KV cache bound)")
    for key, value in BUDGET.items():
        p.add_argument(f"--{key}", type=int, default=value)
    p.add_argument("--evolve", action="store_true", help="SkillBank evolution at every validation (needs --skills)")
    p.add_argument("--evolve_threshold", type=float, default=0.4, help="categories below this validation success evolve")
    p.add_argument("--evolve_relative", action="store_true",
                   help="evolve categories below the overall validation success instead of below --evolve_threshold")
    p.add_argument("--evolve_min_episodes", type=int, default=10)
    p.add_argument("--evolve_max_categories", type=int, default=2)
    p.add_argument("--evolve_failures", type=int, default=8, help="failed validation episodes shown per category")
    p.add_argument("--evolve_max_new", type=int, default=3)
    p.add_argument("--no_evolve_verify", dest="evolve_verify", action="store_false",
                   help="add the reviewed candidates without the greedy verification (SkillRL as published)")
    p.add_argument("--evolve_min_gain", type=int, default=1, help="verification: failures a candidate must fix")
    p.add_argument("--teacher_model", default="deepseek-ai/DeepSeek-V3")
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--skip_initial_eval", action="store_true")
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_swanlab", action="store_true")
    p.add_argument("--swanlab_mode", choices=["cloud", "offline", "disabled"], default="cloud")
    p.add_argument("--swanlab_project", default="OceanHeart-Agent")
    return p


def group_advantages(rewards, estimator):
    """Advantages of one group of episode rewards; None when the rewards are all equal (nothing to learn)."""
    n = len(rewards)
    mean = sum(rewards) / n
    if max(rewards) - min(rewards) < 1e-9:
        return None
    if estimator == "rloo":
        return [r - (n * mean - r) / (n - 1) for r in rewards]
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** .5
    return [(r - mean) / (std + 1e-4) for r in rewards]


def training_samples(trace, advantage):
    """(prompt ids + completion ids, completion start, advantage) for every generated round of an episode."""
    return [(r["input_ids"] + r["completion_ids"], len(r["input_ids"]), advantage)
            for r in trace["rounds"] if r["completion_ids"]]


def main():
    args = parser().parse_args()
    if Path(args.run_name).name != args.run_name or Path(args.save_adapter).name != args.save_adapter:
        raise ValueError("run_name and save_adapter must be simple names")
    if args.group_size < 2:
        raise ValueError("a group needs at least two episodes")
    if args.evolve and not args.skills:
        raise ValueError("--evolve needs an initial --skills bank")
    if args.evolve and not os.environ.get("SILICONFLOW_API_KEY"):
        raise ValueError("--evolve needs SILICONFLOW_API_KEY for the teacher")
    directory = ROOT / "artifacts/oceanlab/m4" / args.run_name
    if not args.resume and (directory / "state.pt").exists():
        raise FileExistsError("run exists; pass --resume or use a new --run_name")
    if not args.resume and (ROOT / "out/hf" / f"{args.save_adapter}_final").exists():
        raise FileExistsError("Refusing to overwrite an existing adapter; use a new --save_adapter")
    memory_efficient_attention()
    torch.cos(torch.zeros(4096))  # keep the first-call cos glitch off the RoPE table
    local_rank = init_distributed_mode()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed + rank)
    directory.mkdir(parents=True, exist_ok=True)

    tokenizer = native_tokenizer(args.model_path, False)
    requests = read_jsonl(args.requests)
    val_requests = read_jsonl(args.val_requests)
    val_requests = val_requests[:args.val_limit] if args.val_limit else val_requests
    if any(r["split"] != "train" for r in requests) or any(r["split"] != "dev" for r in val_requests):
        raise ValueError("training requests must come from the training partition and validation from dev")
    order = list(range(len(requests)))
    random.Random(args.seed).shuffle(order)  # the same request order for every method and rank
    data = OceanDepths()
    labs = {"train": OceanLab("train", data), "dev": OceanLab("dev", data)}
    memory = None
    if args.skills:
        from agent.oceanlab.skills import SkillMemory, load
        memory = SkillMemory(load(args.skills))
    harness = OceanLabHarness(Scorer(data), prompt=args.prompt, memory=memory)
    teacher = None
    if args.evolve and rank == 0:
        from agent.oceanlab.teacher import Teacher
        teacher = Teacher(directory / "teacher_cache.jsonl", model=args.teacher_model)
    budget = {k: getattr(args, k) for k in BUDGET}

    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.start_adapter, is_trainable=True)
    model.load_adapter(str(args.start_adapter), adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for key in list(module.lora_dropout.keys()):
                module.lora_dropout[key] = torch.nn.Identity()
    trainable = []
    for name, parameter in model.named_parameters():
        if "lora_" in name and ".default." in name:  # after loading both adapters: loading re-casts a layer's adapters
            parameter.data = parameter.data.float()
            trainable.append(parameter)
        elif parameter.requires_grad:
            raise RuntimeError(f"unexpected trainable parameter {name}")
    if not trainable or not all(p.requires_grad for p in trainable):
        raise RuntimeError("the default adapter is not trainable")
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=device.type == "cuda")
    step, swanlab_id = 0, None
    if args.resume:
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, state["adapter"], adapter_name="default")
        for parameter in trainable:
            parameter.data = parameter.data.float()
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, swanlab_id = state["step"], state.get("swanlab_id")
        if state.get("skills"):
            harness.memory = SkillMemory(state["skills"])
    # The scale is created lazily by scale(); a rank whose groups are all flat calls only unscale_, so create it now.
    scaler.scale(torch.zeros((), device=device))

    def learning_rate(at):
        decay_start, decay_steps = (0, args.max_steps) if args.schedule == "cosine" else (10 ** 9, 1)
        return scheduled_lr(at, args.learning_rate, args.warmup_steps, 0, decay_start, decay_steps)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(world_size=world, train_requests=len(requests), val_requests=len(val_requests),
                  episodes_per_step=world * args.groups_per_rank * args.group_size, budget=budget,
                  trainable_parameters=sum(p.numel() for p in trainable), dtype="float16 base, float32 LoRA",
                  reference="start adapter, frozen", lora_dropout="removed",
                  start_adapter_sha256=file_hash(args.start_adapter / "adapter_model.safetensors"),
                  skills_sha256=file_hash(args.skills) if args.skills else None,
                  requests_sha256=file_hash(args.requests), val_requests_sha256=file_hash(args.val_requests),
                  env_sha256=file_hash(ROOT / "agent/oceanlab/env.py"), scoring_sha256=file_hash(ROOT / "agent/oceanlab/scoring.py"),
                  sdk_sha256=file_hash(ROOT / "agent/sdk.py"),
                  chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                  git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    tracker = None
    if rank == 0:
        atomic_json(directory / ("config.resume.json" if args.resume else "config.json"), config)
        print(json.dumps({k: config[k] for k in ("trainable_parameters", "episodes_per_step", "world_size")}), flush=True)
        if args.use_swanlab:
            try:  # tracking must never stop training
                import swanlab
                run = init_swanlab(swanlab, {"swanlab_id": swanlab_id} if swanlab_id else None, args.swanlab_mode,
                                   project=args.swanlab_project, name=args.run_name, logdir=str(directory / "swanlog"),
                                   config=config)
                swanlab_id = getattr(getattr(run, "public", None), "cuid", None) or swanlab_id
                tracker = swanlab
            except Exception as exc:
                print(f"swanlab disabled: {exc}", flush=True)

    def log(values, at):
        if rank == 0:
            if tracker:
                try:
                    tracker.log(values, step=at)
                except Exception as exc:
                    print(f"swanlab log failed: {exc}", flush=True)
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": at, **values}) + "\n")

    def save(suffix):
        if rank == 0:
            target = ROOT / "out/hf" / f"{args.save_adapter}{suffix}"
            model.save_pretrained(target, selected_adapters=["default"])
            bank = harness.memory.bank if harness.memory else None
            if bank:  # the SkillBank version this adapter was trained with (evaluate with --skills <this file>)
                atomic_json(target / "skillbank.json", bank)
            torch.save({"adapter": get_peft_model_state_dict(model, adapter_name="default"), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "swanlab_id": swanlab_id, "skills": bank},
                       directory / "state.pt.tmp")
            (directory / "state.pt.tmp").replace(directory / "state.pt")
        if world > 1:
            dist.barrier()

    def episode(spec, lab, generate, memory_harness=None):
        started = time.monotonic()
        trace = run_trajectory(spec, tokenizer, lab, generate, harness=memory_harness or harness, **budget)
        trace["seconds"] = round(time.monotonic() - started, 1)
        return trace

    def rollouts(specs, lab, sample, copies):
        model.eval()
        runner = BatchedEpisodes(hf_batch_generator(model, tokenizer, device, sample, args.token_budget, args.temperature))
        traces = runner.run([partial(episode, spec, lab) for spec in specs for _ in range(copies)])
        model.train()
        torch.cuda.empty_cache()
        return traces, runner.batch_sizes

    def completion_logps(ids, start):
        """Log-probabilities of ids[start:] given the tokens before them (logits only at those positions)."""
        base = model.get_base_model()
        tensor = torch.tensor([ids], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = base.model(input_ids=tensor, use_cache=False).last_hidden_state[0, start - 1:-1]
            logits = base.lm_head(hidden).float()
        return logits.gather(1, tensor[0, start:, None]).squeeze(1) - logits.logsumexp(-1)

    def check_environment(traces, where):
        """Collective: every rank stops on an environment error found by any rank (never train on a broken reward)."""
        bad = sorted({t["id"] for t in traces if t["metrics"]["env_error"]})
        if reduce_sums([len(bad)], device)[0]:
            raise RuntimeError(f"environment errors in {where} episodes (this rank: {bad}); fix the environment and --resume")

    def validate():
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        traces, _ = rollouts(val_requests[rank::world], labs["dev"], sample=False, copies=1)
        check_environment(traces, "validation")
        mine = []
        for trace in traces:
            mine.append(compact(trace))
            mine[-1].update(sample=-1)
        gathered = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
        if rank == 0:
            traces = sorted((t for part in gathered for t in part), key=lambda t: t["id"])
            summary = summarize(traces, val_requests)
            with (directory / f"val_step{step}.jsonl").open("w", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            atomic_json(directory / f"val_step{step}.json", summary)
            g = summary["greedy"]
            values = {"val/success": g["success"], "val/reward": g["reward"], "val/computable_success": g["computable_success"],
                      "val/refusal_correct": g["refusal_correct"], "val/false_refusal_rate": g["false_refusal_rate"],
                      "val/protocol_violation_rate": g["protocol_violation_rate"], "val/calls": g["calls"],
                      "val/seconds": time.monotonic() - started,
                      **{f"val/success_{k}": v for k, v in summary["success_by_request_type"]["greedy"].items()},
                      **{f"val/success_{k}": v for k, v in g["success_by_family"].items()}}
            log(values, step)
            print(f"validation step={step} " + json.dumps({k.split('/', 1)[1]: round(v, 4) for k, v in values.items()}), flush=True)
        if args.evolve and 0 < step < args.max_steps:  # a bank made after the last update would never be trained with
            evolve_skills(traces if rank == 0 else None, summary["greedy"]["success"] if rank == 0 else None)
        if args.evolve and world > 1:  # every rank continues with rank 0's bank version
            shared = [harness.memory.bank]
            dist.broadcast_object_list(shared, src=0)
            harness.memory = SkillMemory(shared[0])
        if world > 1:
            dist.barrier()

    def evolve_skills(traces, overall):
        """Collective SkillBank evolution after a validation (traces and overall success on rank 0 only).

        Rank 0 picks the weak categories and gets the teacher's reviewed candidates with their verification requests
        (the evidence failures and as many successes); with --evolve_verify every rank then runs its share of
        (candidate, request) greedy episodes with the current bank plus that one candidate, and rank 0 keeps the
        candidates that fix more than they break. harness.memory gets the new version on rank 0 (validate() shares it)."""
        from agent.oceanlab import evolve as ev
        from scripts.oceanlab_distill_skills import RULES, condense
        specs = {r["id"]: r for r in val_requests}
        bank = harness.memory.bank
        plan = [None]
        if rank == 0:
            stats = ev.category_success(traces, specs)
            weak = ev.weak_categories(stats, overall, args.evolve_threshold, args.evolve_min_episodes,
                                      args.evolve_max_categories, args.evolve_relative)
            logs, candidates, verify = {}, [], {}
            for category in weak:
                inside = [t for t in traces if category in ev.categories_of(
                    [specs[t["id"]]["text"]] + ([specs[t["id"]]["followup"]["text"]] if specs[t["id"]].get("followup") else []),
                    bool(specs[t["id"]].get("followup")))]
                failures = ev.diverse_failures(inside, args.evolve_failures, f"{args.seed}:{step}")
                if not failures:
                    continue
                found, logs[category] = ev.propose(bank, category, [(t, specs[t["id"]]) for t in failures],
                                                   lambda t, spec: condense(t, spec, data.depth), teacher, RULES,
                                                   args.evolve_max_new)
                candidates += [(category, skill) for skill in found]
                controls = ev.sample_successes(inside, len(failures), f"{args.seed}:{step}")
                verify[category] = {**{t["id"]: 0.0 for t in failures}, **{t["id"]: 1.0 for t in controls}}
            plan = [{"stats": stats, "weak": weak, "logs": logs, "candidates": candidates, "verify": verify}]
        if world > 1:
            dist.broadcast_object_list(plan, src=0)
        plan = plan[0]
        trials = {}  # candidate index -> {request id: S}
        if args.evolve_verify and plan["candidates"]:
            jobs = [(i, request_id) for i, (category, _) in enumerate(plan["candidates"]) for request_id in plan["verify"][category]]
            mine = jobs[rank::world]
            trial_harness = {i: OceanLabHarness(harness.scorer, prompt=args.prompt, memory=SkillMemory(ev.with_skill(bank, *c)))
                             for i, c in enumerate(plan["candidates"])}
            model.eval()
            runner = BatchedEpisodes(hf_batch_generator(model, tokenizer, device, False, args.token_budget))
            results = runner.run([partial(episode, specs[request_id], labs["dev"], memory_harness=trial_harness[i])
                                  for i, request_id in mine])
            model.train()
            torch.cuda.empty_cache()
            check_environment(results, "verification")
            local = [(i, request_id, t["metrics"]["success"]) for (i, request_id), t in zip(mine, results)]
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, local)
            else:
                gathered = [local]
            for i, request_id, success in (x for part in gathered for x in part):
                trials.setdefault(i, {})[request_id] = success
        if rank != 0:
            return
        accepted, checks = {}, []
        for i, (category, skill) in enumerate(plan["candidates"]):
            if args.evolve_verify:
                fixed, broken, ok = ev.verdict(plan["verify"][category], trials[i], args.evolve_min_gain)
                checks.append({"category": category, "title": skill.get("title"), "principle": skill["principle"],
                               "fixed": fixed, "broken": broken, "accepted": ok})
            else:
                ok = True
            if ok:
                accepted.setdefault(category, []).append(skill)
        added = []
        for category, skills in accepted.items():
            bank, new = ev.add_skills(bank, category, skills, f"e{step}", args.evolve_max_new)
            added += new
        bank = {**bank, "version": f"{args.run_name}-step{step}" if added else bank.get("version")}
        harness.memory = SkillMemory(bank)
        atomic_json(directory / f"skills_step{step}.json", bank)
        with (directory / "evolution.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "category_success": plan["stats"], "weak": plan["weak"], "logs": plan["logs"],
                                "verification": checks, "added": added, "teacher_usage": teacher.usage,
                                "teacher_cost_cny": teacher.cost()}, ensure_ascii=False) + "\n")
        log({"evolve/weak_categories": len(plan["weak"]), "evolve/candidates": len(plan["candidates"]),
             "evolve/added_skills": len(added), "evolve/teacher_cost_cny": teacher.cost(),
             "evolve/bank_size": len(bank["general"]) + sum(map(len, bank["task_specific"].values())) + len(bank["common_mistakes"]),
             **{f"evolve/success_{c}": rate for c, (_, rate) in plan["stats"].items()}}, step)
        print(f"evolution step={step} weak={plan['weak']} candidates={len(plan['candidates'])} added={len(added)} "
              f"checks={[(c['fixed'], c['broken']) for c in checks]} cost={teacher.cost()}", flush=True)

    if step == 0 and not args.skip_initial_eval:
        validate()
    started = time.monotonic()
    while step < args.max_steps:
        t0 = time.monotonic()
        torch.manual_seed(int(hashlib.sha256(f"{args.seed}:{step}:{rank}".encode()).hexdigest()[:8], 16))
        torch.cuda.reset_peak_memory_stats(device)
        first = (step * world + rank) * args.groups_per_rank
        specs = [requests[order[(first + g) % len(requests)]] for g in range(args.groups_per_rank)]
        traces, batch_sizes = rollouts(specs, labs["train"], sample=True, copies=args.group_size)
        rollout_memory = torch.cuda.max_memory_allocated(device) / 2 ** 30
        t1 = time.monotonic()
        check_environment(traces, "training")
        samples, groups = [], Counter()
        for g, spec in enumerate(specs):
            group = traces[g * args.group_size:(g + 1) * args.group_size]
            rewards = [t["metrics"]["reward"] for t in group]
            successes = [t["metrics"]["success"] for t in group]
            advantages = group_advantages(rewards, args.estimator)
            groups["active"] += advantages is not None
            groups["all_success"] += all(successes)
            groups["all_fail"] += not any(successes)
            groups["S_varies"] += len(set(successes)) > 1
            for i, trace in enumerate(group):
                trace["advantage"] = advantages[i] if advantages else 0.
                if advantages:
                    samples += training_samples(trace, advantages[i])
        local_tokens = sum(len(ids) - start for ids, start, _ in samples)
        global_tokens = reduce_sums([local_tokens], device)[0]

        # reference log-probabilities (frozen start adapter), then the policy pass with gradients
        torch.cuda.reset_peak_memory_stats(device)
        references = []
        if samples:
            model.eval()
            model.set_adapter("reference")
            with torch.no_grad():
                references = [completion_logps(ids, start) for ids, start, _ in samples]
            model.set_adapter("default")
            model.train()
        t2 = time.monotonic()
        kl_sum = policy_sum = 0.
        for (ids, start, advantage), reference in zip(samples, references):
            logp = completion_logps(ids, start)
            ratio = torch.exp(logp - logp.detach())
            log_ratio = reference - logp
            kl = torch.exp(log_ratio) - log_ratio - 1
            policy = -ratio * advantage
            loss = (policy + args.beta * kl).sum() / global_tokens
            scaler.scale(loss).backward()
            kl_sum += float(kl.detach().sum())
            policy_sum += float(policy.detach().sum())
        train_memory = torch.cuda.max_memory_allocated(device) / 2 ** 30
        for parameter in trainable:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        if world > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in trainable])
            dist.all_reduce(flat)  # the loss is already divided by the global token count
            offset = 0
            for parameter in trainable:
                parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()
        step += 1
        norm = torch.zeros((), device=device)
        if global_tokens:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate(step)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        t3 = time.monotonic()

        m = [t["metrics"] for t in traces]
        stops = Counter(x["stop"] for x in m)
        sums = reduce_sums([len(m), sum(x["reward"] for x in m), sum(x["success"] for x in m),
                            sum(x["protocol_violation"] for x in m), sum(x["false_refusal"] for x in m),
                            sum(x["turns"] for x in m), sum(x["calls"] for x in m), sum(x["generated_tokens"] for x in m),
                            sum(x["input_tokens"] for x in m), len(specs), groups["active"], groups["all_success"],
                            groups["all_fail"], groups["S_varies"], len(samples), kl_sum, policy_sum,
                            sum(batch_sizes) / max(len(batch_sizes), 1), t1 - t0, t2 - t1, t3 - t2,
                            rollout_memory, train_memory] + [stops[s] for s in STOPS], device)
        (n, reward, success, protocol, false_refusal, turns, calls, generated, inputs, n_groups, active, all_success,
         all_fail, s_varies, n_samples, kl_total, policy_total, batch, rollout_s, reference_s, train_s,
         rollout_gib, train_gib), stop_counts = sums[:23], sums[23:]
        values = {"train/reward": reward / n, "train/success": success / n, "train/protocol_violation": protocol / n,
                  "train/false_refusal": false_refusal / n, "train/turns": turns / n, "train/calls": calls / n,
                  "train/generated_tokens": generated / n, "train/input_tokens": inputs / n,
                  "train/active_groups": active / n_groups, "train/all_success_groups": all_success / n_groups,
                  "train/all_fail_groups": all_fail / n_groups, "train/S_varies_groups": s_varies / n_groups,
                  "train/samples": n_samples, "train/trained_tokens": global_tokens,
                  "train/kl": kl_total / max(global_tokens, 1), "train/policy_term": policy_total / max(global_tokens, 1),
                  "train/grad_norm": float(norm), "train/learning_rate": learning_rate(step), "train/loss_scale": scaler.get_scale(),
                  "time/rollout_seconds": rollout_s / world, "time/reference_seconds": reference_s / world,
                  "time/train_seconds": train_s / world, "time/step_seconds": time.monotonic() - t0,
                  "time/generation_batch": batch / world, "memory/rollout_gib": rollout_gib / world,
                  "memory/train_gib": train_gib / world,
                  **{f"stop/{s}": c / n for s, c in zip(STOPS, stop_counts)}}
        log(values, step)
        with (directory / f"rollouts_rank{rank}.jsonl").open("a", encoding="utf-8") as f:
            for trace in traces:
                row = compact(trace)
                for r in row["rounds"]:
                    r.pop("completion_ids")
                f.write(json.dumps({"step": step, **row}, ensure_ascii=False) + "\n")
        if rank == 0:
            print(f"step={step}/{args.max_steps} " + json.dumps({k.split('/', 1)[1]: float(f"{v:.4g}") for k, v in values.items()}), flush=True)
        if step % args.eval_interval == 0:
            validate()
            save(f"_step{step}")
        if step % args.save_interval == 0:
            save("_last")
    if step % args.eval_interval:
        validate()
    save("_final")
    if rank == 0:
        atomic_json(directory / "completion.json", {"step": step, "elapsed_seconds": time.monotonic() - started,
                                                    "adapter": f"out/hf/{args.save_adapter}_final"})
        if tracker:
            tracker.finish()
    if world > 1:
        dist.barrier()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
