#!/usr/bin/env python3
"""Create stable Ocean GRPO prompt subsets from processed SFT data."""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def select(path, count, seed=42):
    ranked = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            question = next(message["content"] for message in item["conversations"] if message["role"] == "user")
            score = hashlib.sha256(f"{seed}:{question}".encode()).digest()
            ranked.append((score, item))
    return [item for _, item in sorted(ranked, key=lambda pair: pair[0])[:count]]


def atomic_jsonl(path, items):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outputs = {}
    for split, count in (("train", args.train_samples), ("val", args.val_samples)):
        source = args.input_dir / f"ocean_sft_{split}.jsonl"
        output = args.input_dir / f"ocean_grpo_{split}.jsonl"
        items = select(source, count, args.seed)
        atomic_jsonl(output, items)
        outputs[split] = {"source": str(source), "path": str(output), "count": len(items)}
    manifest = {"seed": args.seed, "selection": "smallest sha256(seed:question)", "splits": outputs}
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    temp = args.input_dir / "grpo_manifest.json.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.input_dir / "grpo_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
