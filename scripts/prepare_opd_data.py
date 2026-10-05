#!/usr/bin/env python3
"""Create stable two-domain OPD/MOPD datasets from processed OceanHeart data."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_ocean_data import digest, normalized


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_key(item):
    messages = item["conversations"][:-1]
    return normalized(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def select(items, count, seed, excluded=()):
    excluded = set(excluded)
    ranked, seen = [], set()
    for item in items:
        key = prompt_key(item)
        if key in excluded or key in seen:
            continue
        seen.add(key)
        score = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        ranked.append((score, digest(item), item))
    if len(ranked) < count:
        raise ValueError(f"requested {count} samples, only {len(ranked)} unique prompts are available")
    return [item for _, _, item in sorted(ranked)[:count]]


def tagged(items, domain):
    return [{**item, "domain": domain} for item in items]


def stable_mix(items, seed):
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{digest(item)}".encode()).digest())


def atomic_jsonl(path, items):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    return {"path": str(path), "count": len(items), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(input_dir, seed=42, ocean_train_count=1600, general_train_count=400, val_count=100):
    ocean_train_source = load_jsonl(input_dir / "ocean_grpo_train.jsonl")
    ocean_val_source = load_jsonl(input_dir / "ocean_grpo_val.jsonl")
    ocean_sft = load_jsonl(input_dir / "ocean_sft_train.jsonl")
    replay = load_jsonl(input_dir / "ocean_sft_replay_train.jsonl")
    ocean_digests = {digest(item) for item in ocean_sft}
    general_train_source = [item for item in replay if digest(item) not in ocean_digests]
    general_val_source = load_jsonl(input_dir / "generic_sft_eval.jsonl")

    ocean_train = select(ocean_train_source, ocean_train_count, seed)
    ocean_val = select(ocean_val_source, val_count, seed)
    ocean_prompts = {prompt_key(item) for item in [*ocean_train, *ocean_val]}
    general_train = select(general_train_source, general_train_count, seed, ocean_prompts)
    train_prompts = ocean_prompts | {prompt_key(item) for item in general_train}
    general_val = select(general_val_source, val_count, seed, train_prompts)

    datasets = {
        "ocean_opd_train": tagged(ocean_train, "ocean"),
        "ocean_opd_val": tagged(ocean_val, "ocean"),
        "general_opd_train": tagged(general_train, "general"),
        "general_opd_val": tagged(general_val, "general"),
    }
    datasets["ocean_mopd_train"] = stable_mix(
        datasets["ocean_opd_train"] + datasets["general_opd_train"], seed
    )
    datasets["ocean_mopd_val"] = stable_mix(
        datasets["ocean_opd_val"] + datasets["general_opd_val"], seed
    )

    train_keys = {prompt_key(item) for item in datasets["ocean_mopd_train"]}
    val_keys = {prompt_key(item) for item in datasets["ocean_mopd_val"]}
    if train_keys & val_keys:
        raise ValueError("OPD train/validation prompt leakage detected")

    outputs = {
        name: atomic_jsonl(input_dir / f"{name}.jsonl", items)
        for name, items in datasets.items()
    }
    manifest = {
        "seed": seed,
        "selection": "smallest sha256(seed:normalized conversation prompt)",
        "domains": {"ocean": ocean_train_count, "general": general_train_count},
        "train_ratio": {
            "ocean": ocean_train_count / (ocean_train_count + general_train_count),
            "general": general_train_count / (ocean_train_count + general_train_count),
        },
        "prompt_leakage": 0,
        "sources": {
            "ocean": "ocean_grpo_{train,val}.jsonl",
            "general_train": "generic rows from ocean_sft_replay_train.jsonl",
            "general_val": "generic_sft_eval.jsonl",
        },
        "outputs": outputs,
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    manifest["content_sha256"] = hashlib.sha256(payload).hexdigest()
    path = input_dir / "opd_manifest.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ocean-train-samples", type=int, default=1600)
    parser.add_argument("--general-train-samples", type=int, default=400)
    parser.add_argument("--val-samples", type=int, default=100, help="samples per domain")
    args = parser.parse_args()
    manifest = prepare(
        args.input_dir, args.seed, args.ocean_train_samples,
        args.general_train_samples, args.val_samples,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
