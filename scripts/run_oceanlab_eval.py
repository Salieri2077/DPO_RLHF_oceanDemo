#!/usr/bin/env python3
"""OceanDepths analysis episodes for a Hugging Face agent checkpoint (M2 zero-shot / baseline evaluation).

Each request runs one greedy episode and `--samples` sampled episodes (temperature 1, the RL rollout setting; seeds per
request and sample) through the SDK harness and the rule scorer. Budgets default to a loose measurement setting, so
the run shows how much context, generation and how many calls the model actually uses; they are frozen after M2.

Sharding: `--shard i/n` processes requests[i::n] into <out>/shards/i.jsonl (one GPU per process); `--merge` combines
the shards into <out>/episodes.jsonl and <out>/summary.json. Saved traces keep each round's token counts, completion
ids and text, but not the prompt ids (rebuildable from the messages).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash, read_jsonl
from trainer.qwen_sdpa import memory_efficient_attention

REQUESTS = ROOT / "data/processed/oceanlab/oceanlab_dev_v1.jsonl"
BUDGET = dict(max_turns=24, max_calls=20, max_total_len=16384, max_new_tokens=1024)


def seed_of(request_id, sample):
    return int(hashlib.sha256(f"oceanlab:{request_id}:{sample}".encode()).hexdigest()[:8], 16)


def compact(trace):
    out = {k: v for k, v in trace.items() if k != "rounds"}
    out["rounds"] = [{"input_tokens": len(r["input_ids"]), "output_tokens": len(r["completion_ids"]),
                      "completion_ids": r["completion_ids"], "text": r["text"], "action": r.get("action"),
                      "seconds": round(r["seconds"], 2)} for r in trace["rounds"]]
    return out


def run_shard(args):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from agent.oceanlab.env import OceanLab, OceanLabHarness
    from agent.oceanlab.scoring import Scorer
    from agent.sdk import run_trajectory
    from trainer.train_agent import model_generator
    from trainer.train_agent_hf import agent_tokenizer, native_tokenizer

    index, count = map(int, args.shard.split("/"))
    requests = read_jsonl(args.requests)
    requests = (requests[:args.limit] if args.limit else requests)[index::count]
    torch.cos(torch.zeros(4096))  # keep the first-call cos glitch off the RoPE table
    memory_efficient_attention()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # minimind: the agent template all our tool SFT used; native: the Instruct model's own template (non-thinking).
    tokenizer = agent_tokenizer(args.model_path) if args.chat_template == "minimind" else native_tokenizer(args.model_path, False)
    lab = OceanLab(args.scope)
    memory = None
    if args.skills:  # SkillBank retrieved into a preface of the first user message
        from agent.oceanlab.skills import SkillMemory, load
        memory = SkillMemory(load(args.skills))
    harness = OceanLabHarness(Scorer(lab.data), prompt=args.prompt, memory=memory)
    if args.teacher:  # an API model as the agent; the local tokenizer only frames its replies as token ids
        from agent.oceanlab.teacher import Teacher, TeacherPolicy
        teacher = Teacher(args.out / "teacher_cache" / f"{index}.jsonl", model=args.teacher)
        greedy = TeacherPolicy(teacher, tokenizer, harness, temperature=0.0)
        sampled = TeacherPolicy(teacher, tokenizer, harness, temperature=1.0)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(device)
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        greedy = model_generator(model, tokenizer, device, sample=False, save_logps=False)
        sampled = model_generator(model, tokenizer, device, sample=True, save_logps=False)
    budget = {k: getattr(args, k) for k in BUDGET}
    path = args.out / "shards" / f"{index}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(t["id"], t["sample"]) for t in read_jsonl(path)} if path.exists() else set()
    with path.open("a", encoding="utf-8") as f:
        for spec in requests:
            for sample in range(-1, args.samples):  # -1 is the greedy episode
                if (spec["id"], sample) in done:
                    continue
                torch.manual_seed(seed_of(spec["id"], sample))
                started = time.monotonic()
                trace = run_trajectory(spec, tokenizer, lab, greedy if sample < 0 else sampled, harness=harness, **budget)
                trace = compact(trace)
                trace.update(sample=sample, seconds=round(time.monotonic() - started, 1))
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                f.flush()
                m = trace["metrics"]
                print(f"{spec['id']} s{sample} S={m['success']:.0f} r={m['reward']:.1f} {m['failure']} stop={m['stop']} "
                      f"turns={m['turns']} calls={m['calls']} gen={m['generated_tokens']} {trace['seconds']}s", flush=True)
                if args.teacher:
                    (args.out / f"teacher_usage_{index}.json").write_text(json.dumps(
                        {**teacher.usage, "cost_cny": teacher.cost()}, indent=1))


def summarize(traces, requests=None):
    from agent.oceanlab.scoring import aggregate
    env_bad = sorted({t["id"] for t in traces if t["metrics"]["env_error"]})
    greedy = [t for t in traces if t["sample"] < 0]
    sampled = [t for t in traces if t["sample"] >= 0]
    out = {"env_error_requests": env_bad}
    if env_bad:
        return out
    out["greedy"] = aggregate(greedy)
    out["sampled"] = aggregate(sampled) if sampled else None
    groups = {}
    for t in sampled:
        groups.setdefault(t["id"], []).append(t["metrics"])
    if groups:
        s = {i: [m["success"] for m in g] for i, g in groups.items()}
        r = {i: [m["reward"] for m in g] for i, g in groups.items()}
        out["groups"] = {"requests": len(groups), "pass_at_k": sum(any(v) for v in s.values()) / len(s),
                         "all_success": sum(all(v) for v in s.values()), "all_fail": sum(not any(v) for v in s.values()),
                         "S_varies": sum(len(set(v)) > 1 for v in s.values()),
                         "r_varies": sum(len(set(v)) > 1 for v in r.values()),
                         "r_varies_without_S": sum(len(set(r[i])) > 1 and len(set(s[i])) == 1 for i in s)}
    calls = [c for t in traces for c in t["calls"]]
    out["tool_use"] = dict(Counter(c["call"].get("name", "<invalid action>") for c in calls))
    out["tool_errors"] = dict(Counter(c["result"].get("error_kind", "other") for c in calls if "error" in c["result"]))
    out["actions"] = dict(Counter((r.get("action") or {}).get("type", "none") for t in traces for r in t["rounds"]))
    lengths = {"round_output_tokens": [r["output_tokens"] for t in traces for r in t["rounds"]],
               "round_input_tokens": [r["input_tokens"] for t in traces for r in t["rounds"]],
               "episode_input_tokens": [t["metrics"]["input_tokens"] for t in traces],
               "episode_generated_tokens": [t["metrics"]["generated_tokens"] for t in traces],
               "calls": [t["metrics"]["calls"] for t in traces], "turns": [t["metrics"]["turns"] for t in traces],
               "episode_seconds": [t["seconds"] for t in traces]}

    def pct(values):
        values = sorted(values)
        at = lambda q: values[min(len(values) - 1, int(q * len(values)))]
        return {"p50": at(.5), "p90": at(.9), "p99": at(.99), "max": values[-1]}
    out["lengths"] = {k: pct(v) for k, v in lengths.items() if v}
    if requests:  # request set v2: success by request type, period kind and follow-up
        info = {r["id"]: r for r in requests}
        for name, key in (("request_type", lambda r: r.get("request_type", "v1")),
                          ("kind", lambda r: f"{r.get('request_type', 'v1')}/{r.get('kind', '-')}"),
                          ("followup", lambda r: r.get("followup", {}).get("type", "-"))):
            for label, subset in (("greedy", greedy), ("sampled", sampled)):
                cells = {}
                for t in subset:
                    cells.setdefault(key(info[t["id"]]), []).append(t["metrics"]["success"])
                out.setdefault(f"success_by_{name}", {})[label] = {k: round(sum(v) / len(v), 4) for k, v in sorted(cells.items())}
    return out


def merge(args):
    config = json.loads((args.out / "config.json").read_text())
    traces = [t for f in sorted((args.out / "shards").glob("*.jsonl")) for t in read_jsonl(f)]
    requests = read_jsonl(Path(config["requests"]))
    expected = len(requests[:config["limit"]] if config["limit"] else requests) * (config["samples"] + 1)
    if len(traces) != expected:
        sys.exit(f"incomplete run: {len(traces)} of {expected} episodes; rerun the missing shards (they resume)")
    traces.sort(key=lambda t: (t["id"], t["sample"]))
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    summary = {"config": config, "episodes": len(traces),
               **summarize(traces, requests)}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False, indent=1))
    if summary["env_error_requests"]:
        sys.exit(f"environment errors in {summary['env_error_requests']}: fix the environment and rerun them")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Agent starting point (design section 7): the official Qwen3-1.7B Instruct, no adapter. The earlier Base-chain
    # start was out/hf/qwen3_1.7b_ocean_grpo400_merged + out/hf/react-qwen1.7b-grpo400-v3.1-20261009-1806-sft_best.
    p.add_argument("--model_path", type=Path, default=Path("/data/anhuang/oceanheart_models/qwen3-1.7b"))
    p.add_argument("--adapter", type=Path, help="LoRA adapter directory (omit for the bare model)")
    p.add_argument("--chat_template", choices=["minimind", "native"], default="minimind")
    p.add_argument("--prompt", choices=["v1", "v2"], default="v1", help="environment system prompt (v2 for request set v2)")
    p.add_argument("--scope", choices=["dev", "train"], default="dev", help="environment partition of the requests")
    p.add_argument("--teacher", help="SiliconFlow model id to act as the agent instead of a local model")
    p.add_argument("--skills", type=Path, help="SkillBank JSON to retrieve into the prompt (memory policy 'skills')")
    p.add_argument("--requests", type=Path, default=REQUESTS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--merge", action="store_true")
    for key, value in BUDGET.items():
        p.add_argument(f"--{key}", type=int, default=value)
    args = p.parse_args()
    if args.merge:
        return merge(args)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"model_path": str(args.model_path), "chat_template": args.chat_template, "prompt": args.prompt,
              "scope": args.scope, "teacher": args.teacher, "skills": str(args.skills) if args.skills else None,
              "skills_sha256": file_hash(args.skills) if args.skills else None,
              "model_config_sha256": file_hash(args.model_path / "config.json"),
              "adapter": str(args.adapter) if args.adapter else None,
              "adapter_sha256": file_hash(args.adapter / "adapter_model.safetensors") if args.adapter else None,
              "requests": str(args.requests), "requests_sha256": file_hash(args.requests), "limit": args.limit,
              "samples": args.samples, "temperature": 1.0, "budget": {k: getattr(args, k) for k in BUDGET},
              "attention": "sdpa, KV heads expanded (memory-efficient kernel)",
              "sdk_sha256": file_hash(ROOT / "agent/sdk.py"), "env_sha256": file_hash(ROOT / "agent/oceanlab/env.py"),
              "scoring_sha256": file_hash(ROOT / "agent/oceanlab/scoring.py"),
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
    path = args.out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"{path} differs from this run's configuration; use a new --out")
    path.write_text(json.dumps(config, indent=1))
    run_shard(args)


if __name__ == "__main__":
    main()
