#!/usr/bin/env python3
"""C0 demonstrations: verified oracle trajectories on the training requests (train partition only).

Every training request's oracle trajectory (agent/oceanlab/oracle.py) is replayed through the SDK harness with the
agent's tokenizer and chat template, so the saved messages, observations included, are exactly the model's inputs at
inference time. A demonstration is kept only if the scorer gives S = 1 with no environment error.

A seeded permutation ranks the demonstrations: the last `--holdout` ranks are a validation-loss hold-out; nested
training subsets are the first 250 / 500 / all remaining ranks (learning curve on dev decides how many C0 needs).
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash, read_jsonl
from agent.oceanlab.env import OceanLab, OceanLabHarness
from agent.oceanlab.oracle import oracle_texts
from agent.oceanlab.scoring import Scorer
from agent.sdk import run_trajectory
from trainer.train_agent_hf import agent_tokenizer, native_tokenizer

VERSION = "oceanlab_c0_demos_v1"


def form_of(spec):
    groups = spec["groups"]
    if len(groups) == 1:
        return "season" if len(groups[0]["months"]) > 1 else "single"
    return "monthly_season" if all(len(g["months"]) == 1 for g in groups) and len(groups) == 3 else "two_months"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, default=ROOT / "data/processed/oceanlab/oceanlab_train_v1.jsonl")
    p.add_argument("--model_path", type=Path, default=Path("/data/anhuang/oceanheart_models/qwen3-1.7b"))
    p.add_argument("--chat_template", choices=["minimind", "native"], default="native")
    p.add_argument("--prompt", choices=["v1", "v2"], default="v1",
                   help="environment system prompt; v2 renders the same demonstrations under the v2 prompt")
    p.add_argument("--holdout", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/oceanlab/c0")
    args = p.parse_args()
    version = VERSION if args.prompt == "v1" else f"{VERSION}_prompt{args.prompt}"
    path = args.output / f"{version}.jsonl"
    if path.exists():
        raise FileExistsError(f"{path} exists; demonstration sets are frozen once written")
    tokenizer = agent_tokenizer(args.model_path) if args.chat_template == "minimind" else native_tokenizer(args.model_path, False)
    lab = OceanLab("train")
    harness = OceanLabHarness(Scorer(lab.data), prompt=args.prompt)
    requests = read_jsonl(args.requests)
    rows, rejected = [], []
    for spec in requests:
        texts = iter(oracle_texts(spec, lab))

        def generate(ids, limit):
            return tokenizer.encode(next(texts), add_special_tokens=False) + [tokenizer.eos_token_id], []
        trace = run_trajectory(spec, tokenizer, lab, generate, harness=harness)
        m = trace["metrics"]
        if m["success"] != 1 or m["env_error"] or trace["metrics"]["protocol_violation"]:
            rejected.append({"id": spec["id"], "metrics": {k: m[k] for k in ("success", "failure", "env_error")}})
            continue
        ids = harness.prompt_ids(tokenizer, trace["messages"][:-1])
        rows.append({"id": spec["id"], "family": spec["family"], "status": spec["reference"]["status"],
                     "form": form_of(spec), "shallow": spec.get("consistency", {}).get("shallow_requested", False),
                     "hemisphere": ("N" if spec["region"][0][0] >= 0 else "S") + ("E" if spec["region"][1][0] >= 0 else "W"),
                     "messages": trace["messages"], "calls": m["calls"],
                     "last_prompt_tokens": len(ids),
                     "assistant_tokens": sum(len(r["completion_ids"]) for r in trace["rounds"])})
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    for rank, i in enumerate(order):
        rows[i]["rank"] = rank
        rows[i]["holdout"] = rank >= len(rows) - args.holdout
    args.output.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    train = [r for r in rows if not r["holdout"]]
    count = lambda key, subset: dict(collections.Counter(r[key] for r in subset))
    manifest = {
        "version": version, "prompt": args.prompt, "requests": str(args.requests.relative_to(ROOT)), "requests_sha256": file_hash(args.requests),
        "model_path": str(args.model_path), "chat_template": args.chat_template,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "env_sha256": file_hash(ROOT / "agent/oceanlab/env.py"), "oracle_sha256": file_hash(ROOT / "agent/oceanlab/oracle.py"),
        "sdk_sha256": file_hash(ROOT / "agent/sdk.py"), "scoring_sha256": file_hash(ROOT / "agent/oceanlab/scoring.py"),
        "seed": args.seed, "kept": len(rows), "rejected": rejected, "holdout": args.holdout,
        "subsets": {str(n): f"rank < {n}" for n in (250, 500, len(train))},
        "train": {k: count(k, train) for k in ("family", "status", "form", "hemisphere", "shallow")},
        "first_250": {k: count(k, [r for r in train if r["rank"] < 250]) for k in ("family", "status", "form")},
        "last_prompt_tokens": {"max": max(r["last_prompt_tokens"] for r in rows),
                               "mean": round(sum(r["last_prompt_tokens"] for r in rows) / len(rows), 1)},
        "assistant_tokens_per_demo": round(sum(r["assistant_tokens"] for r in rows) / len(rows), 1),
        "sha256": file_hash(path),
    }
    (args.output / f"{version}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rejected"} | {"rejected": len(rejected)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
