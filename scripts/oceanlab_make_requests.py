#!/usr/bin/env python3
"""Generate OceanDepths analysis requests of one split (agent/oceanlab/tasks.py) with scorer-side reference results.

dev: the 48 M1 development requests (oceanlab_dev_v1, frozen; regenerating it today also adds the informational
`profiles_rejected_qc` field to consistency references, everything else is identical). train: ~1000 requests for C0
demonstrations (oceanlab_train_v1). Existing sets are never overwritten.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.oceanlab import tasks
from agent.oceanlab.data import PARTITION, QC_POLICY, RESERVED_REGION, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/oceanlab")
    args = parser.parse_args()
    version, plan = {"dev": (tasks.TASK_VERSION, tasks.FAMILY_PLAN), "train": (tasks.TRAIN_VERSION, tasks.TRAIN_PLAN)}[args.split]
    path = args.output / f"{version}.jsonl"
    if path.exists():
        raise FileExistsError(f"{path} exists; request sets are frozen once written")
    requests, rejected = tasks.build_requests(args.split, plan, version, args.seed)
    tasks.write_jsonl(path, requests)
    summary = {
        "task_version": version, "split": args.split, "seed": args.seed, "dataset_revision": REVISION,
        "qc_policy": QC_POLICY, "partition": PARTITION, "reserved_region": RESERVED_REGION, "family_plan": plan,
        "months": [str(m) for m in tasks.split_months(args.split)[[0, -1]]],
        "kept": len(requests), "rejected_disagreement": [r["id"] for r in rejected],
        "status": dict(collections.Counter(r["reference"]["status"] for r in requests)),
        "status_by_family": {f: dict(collections.Counter(r["reference"]["status"] for r in requests if r["family"] == f))
                             for f in plan},
        "density_tertile": dict(collections.Counter(r["density_tertile"] for r in requests)),
        "hemisphere": dict(collections.Counter("N" if r["region"][0][0] >= 0 else "S" for r in requests)),
        "profiles_per_request": sorted(r["reference"]["profiles"] for r in requests) if args.split == "dev" else
        {"min": min(r["reference"]["profiles"] for r in requests), "max": max(r["reference"]["profiles"] for r in requests)},
        "years": dict(sorted(collections.Counter(r["groups"][0]["months"][0][:4] for r in requests).items())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.output / f"{version}_manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
