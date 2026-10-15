#!/usr/bin/env python3
"""v3.1 regression for the pluggable SDK adapter: same weights, greedy, old vs new adapter, token for token.

The pre-refactor adapter is loaded from a git revision next to the working-tree adapter, and every task of the v3.1
evaluation sets runs through both with the same model in one process (evaluate()'s budgets: 2048 context, 192 new
tokens, adapter defaults for turns and calls). Per task it compares every round's input_ids and completion_ids, the
messages, tool calls and results, final text, stop reason and the scorer's metrics. It also compares with the traces
stored by the original 2026-10-06 evaluation of the same weights (different process and GPU layout, reported apart).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer

import agent.sdk as new_sdk
from agent.ocean import file_hash, read_jsonl
from agent.react import ReactTools
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_agent import atomic_json, model_generator

EVALS = {"v2-val": ("data/processed/react-v2", "val"), "v2-test": ("data/processed/react-v2", "test"),
         "v1-regression": ("data/processed", "val"), "v3-test": ("data/processed/react-v3", "test"),
         "v3.1-test": ("data/processed/react-v3.1", "test")}
RUN = "react-v3.1-20261006-1539-contrast"
WEIGHT = ROOT / f"out/ocean_{RUN}-sft_best_768.pth"
KEYS = ("messages", "calls", "final", "stop", "unrecovered_error", "duplicates", "metrics")


def load_adapter(rev):
    source = subprocess.run(["git", "show", f"{rev}:agent/sdk.py"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout
    path = Path(tempfile.mkdtemp()) / "sdk_before.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("sdk_before", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source.encode()).hexdigest()


def difference(a, b):
    """First differing field of two traces, or None."""
    if len(a["rounds"]) != len(b["rounds"]):
        return "rounds"
    for x, y in zip(a["rounds"], b["rounds"]):
        for key in ("input_ids", "completion_ids", "text"):
            if x[key] != y[key]:
                return f"rounds.{key}"
    for key in KEYS:
        if a.get(key) != b.get(key):
            return key
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before_rev", default="73bfa06", help="git revision of the pre-refactor agent/sdk.py")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sets", nargs="+", choices=list(EVALS), default=list(EVALS), help="run a subset (one per GPU)")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/oceanlab/sdk_regression_v3.1.json")
    args = p.parse_args()
    old_sdk, old_sha = load_adapter(args.before_rev)
    torch.cos(torch.zeros(4096))  # first multithreaded cos per process can be inaccurate; keep it off the RoPE table
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    generate = model_generator(model, tokenizer, device, sample=False, save_logps=False)
    report = {"weight": str(WEIGHT.relative_to(ROOT)), "weight_sha256": file_hash(WEIGHT),
              "before_rev": args.before_rev, "before_sha256": old_sha,
              "after_sha256": file_hash(ROOT / "agent/sdk.py"), "device": str(device),
              "budgets": {"max_total_len": 2048, "max_new_tokens": 192, "max_turns": "adapter default",
                          "max_calls": "adapter default"}, "sets": {}}
    started = time.monotonic()
    for name in args.sets:
        data, split = EVALS[name]
        tasks = read_jsonl(ROOT / data / f"ocean_agent_{split}.jsonl")
        tasks = tasks[:args.limit] if args.limit else tasks
        env = ReactTools(read_jsonl(ROOT / data / f"ocean_agent_corpus_{split}.jsonl"))
        original_path = ROOT / f"artifacts/agent/{RUN}-{name}/result.jsonl"
        original = {t["id"]: t for t in read_jsonl(original_path)} if original_path.exists() else {}
        stats = {"tasks": len(tasks), "identical": 0, "mismatches": [], "success_before": 0., "success_after": 0.,
                 "original_compared": 0, "original_identical": 0, "original_mismatches": []}
        for task in tasks:
            kwargs = dict(max_total_len=2048, max_new_tokens=192)
            before = old_sdk.run_trajectory(task, tokenizer, env, generate, **kwargs)
            after = new_sdk.run_trajectory(task, tokenizer, env, generate, **kwargs)
            diff = difference(before, after)
            stats["identical"] += diff is None
            if diff:
                stats["mismatches"].append({"id": task["id"], "field": diff})
            stats["success_before"] += before["metrics"]["success"]
            stats["success_after"] += after["metrics"]["success"]
            if task["id"] in original:
                ref = original[task["id"]]
                stats["original_compared"] += 1
                same = ([r["completion_ids"] for r in ref["rounds"]] == [r["completion_ids"] for r in after["rounds"]]
                        and ref["metrics"] == after["metrics"] and ref["final"] == after["final"])
                stats["original_identical"] += same
                if not same:
                    stats["original_mismatches"].append(task["id"])
        stats["success_before"] /= max(len(tasks), 1)
        stats["success_after"] /= max(len(tasks), 1)
        report["sets"][name] = stats
        print(name, json.dumps({k: v for k, v in stats.items() if "mismatches" not in k}), flush=True)
    report["seconds"] = round(time.monotonic() - started, 1)
    report["all_identical"] = all(s["identical"] == s["tasks"] for s in report["sets"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)
    print("all_identical", report["all_identical"], args.out)
    if not report["all_identical"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
