#!/usr/bin/env python3
"""Download and normalize the small OceanHeart pretrain benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset


DEFAULT_ROOT = Path("/data/anhuang/oceanheart_pretrain_eval")
SEED = 42
SOURCES = {
    "hellaswag": {
        "repo": "Rowan/hellaswag", "config": None, "split": "validation",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "license": "MIT",
        "domain": "General English",
    },
    "xcopa_zh": {
        "repo": "cambridgeltl/xcopa", "config": "zh", "split": "test",
        "revision": "042f78955ba48e6404616762fa6e05e839c3907a", "license": "CC-BY-4.0",
        "domain": "General Chinese",
    },
    "arc_easy": {
        "repo": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453", "license": "CC-BY-SA-4.0",
        "domain": "Science / Reasoning",
    },
    "ocean_science": {
        "repo": "zjunlp/OceanBenchmark", "config": "Science_Text", "split": "train",
        "revision": "640b00c5f84c4bb42425ce8214762fd2e5745216", "license": "MIT",
        "domain": "Ocean",
    },
    "maritime": {
        "repo": "Hi-Dolphin/MaritimeBench", "config": None, "split": "test",
        "revision": "08839d787bb0a655806b4edb20f3f0e3ee52ebd5", "license": "Apache-2.0",
        "domain": "Ocean",
    },
}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def stable_rows(rows, limit: int, key):
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        f"{SEED}:{key(row)}".encode("utf-8")
    ).digest())
    return ranked[: min(limit, len(ranked))]


def record(benchmark, domain, item_id, prompt, choices, answer, choice_prefix=""):
    choices = [str(choice).strip() for choice in choices]
    answer = int(answer)
    if not prompt.strip() or len(choices) < 2 or not 0 <= answer < len(choices) or any(not choice for choice in choices):
        raise ValueError(f"invalid {benchmark} record: {item_id}")
    return {
        "id": str(item_id), "benchmark": benchmark, "domain": domain,
        "prompt": prompt.strip(), "choices": choices, "answer": answer,
        "choice_prefix": choice_prefix,
    }


def adapt_hellaswag(row):
    item_id = hashlib.sha256(json.dumps([row["ctx"], row["endings"]], ensure_ascii=False).encode()).hexdigest()[:16]
    return record("hellaswag", "General English", item_id, row["ctx"], row["endings"], row["label"], " ")


def adapt_xcopa(row):
    bridge = "这是因为" if row["question"] == "cause" else "所以"
    return record("xcopa_zh", "General Chinese", row["idx"], f"{row['premise']}{bridge}",
                  [row["choice1"], row["choice2"]], row["label"])


def adapt_arc(row):
    labels = row["choices"]["label"]
    return record("arc_easy", "Science / Reasoning", row["id"],
                  f"Question: {row['question']}\nAnswer:", row["choices"]["text"], labels.index(row["answerKey"]), " ")


def adapt_ocean(row):
    choices = json.loads(row["choices"])
    labels = list(choices)
    return record("ocean_science", "Ocean", row["id"],
                  f"Question: {row['question']}\nAnswer:", list(choices.values()), labels.index(row["answer"]), " ")


def adapt_maritime(row):
    labels = ["A", "B", "C", "D"]
    return record("maritime", "Ocean", hashlib.sha256(row["question"].encode()).hexdigest()[:16],
                  f"问题：{row['question']}\n答案：", [row[label] for label in labels], labels.index(row["answer"]))


ADAPTERS = {
    "hellaswag": adapt_hellaswag,
    "xcopa_zh": adapt_xcopa,
    "arc_easy": adapt_arc,
    "ocean_science": adapt_ocean,
    "maritime": adapt_maritime,
}


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows):
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def exact_leakage(paths, pretrain_source: Path):
    prompt_hashes = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = hashlib.sha256(normalized(row["prompt"]).encode()).digest()
                prompt_hashes.setdefault(key, []).append(f"{row['benchmark']}:{row['id']}")
    matches = []
    if not pretrain_source.exists():
        return {"status": "source_missing", "source": str(pretrain_source), "matches": matches}
    with pretrain_source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            key = hashlib.sha256(normalized(str(text)).encode()).digest()
            if key in prompt_hashes:
                matches.extend({"benchmark_id": item_id, "pretrain_line": line_number} for item_id in prompt_hashes[key])
    return {
        "status": "checked_exact_normalized_prompt",
        "source": str(pretrain_source),
        "checked_prompts": sum(len(value) for value in prompt_hashes.values()),
        "matches": matches,
        "limitation": "Exact normalized matches only; substring or semantic contamination is not ruled out.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples", type=int, default=500, help="maximum samples per benchmark")
    parser.add_argument("--pretrain-source", type=Path, default=Path("/home/anhuang/minimind/dataset/pretrain_t2t_mini.jsonl"))
    args = parser.parse_args()
    benchmark_dir = args.data_root / "benchmarks"
    cache_dir = args.data_root / "hf" / "datasets"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"schema_version": 1, "seed": SEED, "data_root": str(args.data_root), "benchmarks": {}}
    paths = []
    for name, source in SOURCES.items():
        dataset = load_dataset(source["repo"], source["config"], split=source["split"],
                               revision=source["revision"], cache_dir=str(cache_dir))
        selected = stable_rows(dataset, args.samples, lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
        rows = [ADAPTERS[name](row) for row in selected]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate IDs after adapting {name}")
        path = benchmark_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
        manifest["benchmarks"][name] = {
            **source, "path": str(path), "source_count": len(dataset), "samples": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest["leakage_audit"] = exact_leakage(paths, args.pretrain_source)
    atomic_json(args.data_root / "manifest.json", manifest)
    print(json.dumps({name: item["samples"] for name, item in manifest["benchmarks"].items()}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.data_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
