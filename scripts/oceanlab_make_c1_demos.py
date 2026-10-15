#!/usr/bin/env python3
"""C1 / C1_plain demonstrations from the teacher's verified trajectories on training requests (design section 7).

Input: an evaluation run of the teacher with the SkillBank in its prompt (run_oceanlab_eval.py --teacher --skills
--scope train). Kept: episodes with S = 1, no protocol violation and no environment error. Both sets contain the same
trajectories and supervise the same generated turns (so supervised tokens and updates match):
- C1 keeps the retrieved-skill preface of the first user message;
- C1_plain removes it, leaving exactly the plain request.
For multiturn tasks the answered first turn is context only (supervision_start). A seeded permutation ranks the
demonstrations; the last `--holdout` ranks are the validation-loss hold-out.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="teacher run directory (episodes.jsonl, config.json)")
    p.add_argument("--requests", type=Path, default=ROOT / "data/processed/oceanlab/v2/oceanlab_train_v2.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/oceanlab/c1")
    p.add_argument("--version", default="v1")
    p.add_argument("--holdout", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    if not (config.get("teacher") and config.get("skills") and config.get("scope") == "train"):
        raise ValueError("C1 demonstrations come from a teacher run with skills on training requests")
    specs = {r["id"]: r for r in read_jsonl(args.requests)}
    rows, dropped = [], collections.Counter()
    for t in read_jsonl(args.run / "episodes.jsonl"):
        m = t["metrics"]
        if m["env_error"]:
            raise RuntimeError(f"{t['id']}: environment error in the teacher run; fix and rerun first")
        if not m["success"] or m["protocol_violation"]:
            dropped["failed" if not m["success"] else "protocol_violation"] += 1
            continue
        spec, messages = specs[t["id"]], t["messages"]
        preface_chars = t["memory"]["preface_chars"]
        if messages[1]["role"] != "user" or messages[1]["content"][preface_chars:] != spec["text"]:
            raise ValueError(f"{t['id']}: first user message is not preface + request text")
        start = 2 + len(spec.get("history", []))
        rows.append({"id": t["id"], "sample": t["sample"], "family": spec["family"], "request_type": spec["request_type"],
                     "kind": spec["kind"], "status": spec["reference"]["status"], "supervision_start": start,
                     "skills": t["memory"]["items"], "messages": messages, "preface_chars": preface_chars})
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    for rank, i in enumerate(order):
        rows[i]["rank"] = rank
        rows[i]["holdout"] = rank >= len(rows) - args.holdout
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output / f"{name}_demos_{args.version}.jsonl" for name in ("c1", "c1plain")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("demonstration sets are frozen once written")
    with paths["c1"].open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with paths["c1plain"].open("w", encoding="utf-8") as f:
        for row in rows:
            plain = json.loads(json.dumps(row))
            plain["messages"][1]["content"] = row["messages"][1]["content"][row["preface_chars"]:]
            plain["skills"], plain["preface_chars"] = [], 0
            f.write(json.dumps(plain, ensure_ascii=False) + "\n")
    train = [r for r in rows if not r["holdout"]]
    count = lambda key: dict(collections.Counter(r[key] for r in train))
    manifest = {"version": args.version, "teacher_run": str(args.run), "teacher": config["teacher"],
                "skills": config["skills"], "skills_sha256": config["skills_sha256"],
                "requests_sha256": file_hash(args.requests), "kept": len(rows), "dropped": dict(dropped),
                "holdout": args.holdout, "train": {k: count(k) for k in ("request_type", "kind", "family", "status")},
                "supervised_turns": sum(sum(1 for i, m in enumerate(r["messages"]) if m["role"] == "assistant"
                                            and i >= r["supervision_start"]) for r in train),
                "sha256": {name: file_hash(path) for name, path in paths.items()}}
    (args.output / f"c1_demos_{args.version}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
