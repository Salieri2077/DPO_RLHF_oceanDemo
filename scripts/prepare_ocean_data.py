#!/usr/bin/env python3
"""Convert the legacy OceanHeart data into MiniMind JSONL datasets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "你是 OceanHeart，一个面向海洋科学任务的双语语言模型助手。"
    "请基于可靠知识准确回答；不确定时明确说明。"
)
ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def stable_score(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{normalized(text)}".encode()).hexdigest(), 16)


def ocean_split(prompt: str, seed: int) -> str:
    bucket = stable_score(prompt, seed) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def clean_message(message: dict) -> dict | None:
    role = ROLE_MAP.get(message.get("role", message.get("from")))
    content = message.get("content", message.get("value"))
    if role is None or not isinstance(content, str):
        return None
    content = content.strip()
    if not content or "\ufffd" in content:
        return None
    return {"role": role, "content": content}


def with_system(messages: list[dict]) -> list[dict]:
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def valid_dialogue(messages: list[dict]) -> bool:
    if any(message["role"] == "system" for message in messages[1:]):
        return False
    roles = [message["role"] for message in messages if message["role"] != "system"]
    return bool(roles) and roles[-1] == "assistant" and all(
        role == ("user" if index % 2 == 0 else "assistant")
        for index, role in enumerate(roles)
    )


def convert_sft(item: dict) -> dict | None:
    raw = item.get("conversations")
    if not isinstance(raw, list):
        return None
    messages = []
    for message in raw:
        cleaned = clean_message(message)
        if cleaned is None:
            return None
        messages.append(cleaned)
    if not valid_dialogue(messages):
        return None
    return {"conversations": with_system(messages)}


def convert_dpo(item: dict) -> dict | None:
    context = item.get("conversations", item.get("context"))
    if not isinstance(context, list):
        return None
    messages = []
    for message in context:
        cleaned = clean_message(message)
        if cleaned is None:
            return None
        messages.append(cleaned)
    chosen, rejected = clean_message(item.get("chosen", {})), clean_message(item.get("rejected", {}))
    roles = [message["role"] for message in messages if message["role"] != "system"]
    if (
        not roles
        or any(role != ("user" if index % 2 == 0 else "assistant") for index, role in enumerate(roles))
        or roles[-1] != "user"
        or not chosen
        or not rejected
    ):
        return None
    if chosen["role"] != "assistant" or rejected["role"] != "assistant":
        return None
    if normalized(chosen["content"]) == normalized(rejected["content"]):
        return None
    context = with_system(messages)
    return {"chosen": [*context, chosen], "rejected": [*context, rejected]}


def prompt_of(item: dict) -> str:
    conversations = item.get("conversations") or item.get("chosen") or []
    return next(message["content"] for message in conversations if message["role"] == "user")


def digest(item: dict) -> str:
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_jsonl(path: Path, items) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temp, path)
    return count


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1]


def language(text: str) -> str:
    # ponytail: this cheap script heuristic is enough for corpus reporting; use a detector if routing depends on it.
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if han and latin:
        return "mixed"
    return "zh" if han else "en" if latin else "other"


def describe_chat(items: list[dict], tokenizer, max_length: int, limit: int) -> dict:
    lengths, languages = [], Counter()
    for item in items[:limit]:
        messages = item.get("conversations") or item["chosen"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False).input_ids))
        languages[language(" ".join(
            message["content"] for message in messages if message["role"] != "system"
        ))] += 1
    sampled = len(lengths)
    return {
        "sampled": sampled,
        "language": {
            name: {"count": count, "ratio": count / sampled}
            for name, count in sorted(languages.items())
        },
        "tokens": {
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths, default=None),
            "truncation_rate": sum(value > max_length for value in lengths) / len(lengths) if lengths else 0,
        },
    }


def prepare_ocean_sft(source: Path, seed: int) -> tuple[dict[str, list[dict]], Counter]:
    with source.open(encoding="utf-8") as handle:
        raw_items = json.load(handle)
    splits = {name: [] for name in ("train", "val", "test")}
    seen, stats = set(), Counter(input=len(raw_items))
    for raw in raw_items:
        item = convert_sft(raw)
        if item is None:
            stats["invalid"] += 1
            continue
        key = digest(item)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        splits[ocean_split(prompt_of(item), seed)].append(item)
    return splits, stats


def prepare_dpo(source: Path, seed: int) -> tuple[dict[str, list[dict]], Counter]:
    with source.open(encoding="utf-8") as handle:
        raw_items = json.load(handle)
    splits = {name: [] for name in ("train", "val", "test")}
    seen, stats = set(), Counter(input=len(raw_items))
    for raw in raw_items:
        item = convert_dpo(raw)
        if item is None:
            stats["invalid"] += 1
            continue
        key = digest(item)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        splits[ocean_split(prompt_of(item), seed)].append(item)
    return splits, stats


def prepare_pretrain(source: Path, output: Path, seed: int) -> tuple[dict, list[str]]:
    paths = {name: output / f"pretrain_{name}.jsonl" for name in ("train", "val")}
    temps = {name: path.with_suffix(".jsonl.tmp") for name, path in paths.items()}
    handles = {name: path.open("w", encoding="utf-8") for name, path in temps.items()}
    counts, sample, seen = Counter(), [], set()
    try:
        with source.open(encoding="utf-8") as src:
            for line in src:
                try:
                    item = json.loads(line)
                    text = item["text"].strip()
                except (json.JSONDecodeError, KeyError, AttributeError):
                    counts["invalid"] += 1
                    continue
                if not text or "\ufffd" in text:
                    counts["invalid"] += 1
                    continue
                key = normalized(text)
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(key)
                split = "val" if stable_score(text, seed) % 100 == 99 else "train"
                handles[split].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                counts[split] += 1
                if len(sample) < 10_000:
                    sample.append(text)
    finally:
        for handle in handles.values():
            handle.close()
    for name, path in paths.items():
        os.replace(temps[name], path)
    return dict(counts), sample


def generic_replay(source: Path, count: int, seed: int) -> tuple[list[dict], Counter]:
    heap: list[tuple[int, int, dict]] = []
    seen, stats = set(), Counter()
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stats["input"] += 1
            try:
                item = convert_sft(json.loads(line))
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue
            if item is None:
                stats["invalid"] += 1
                continue
            key = digest(item)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            score = stable_score(key, seed)
            candidate = (-score, index, item)
            if len(heap) < count:
                heapq.heappush(heap, candidate)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
    return [item for _, _, item in sorted(heap, reverse=True)], stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocean-sft-source", type=Path, default=ROOT / "data/raw/ocean_instruct/OceanInstruct-v0.2.json")
    parser.add_argument("--ocean-dpo-source", type=Path, default=ROOT / "data/raw/ocean_dpo_data.json")
    parser.add_argument("--generic-pretrain-source", type=Path, default=ROOT.parent / "minimind/dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--generic-sft-source", type=Path, default=ROOT.parent / "minimind/dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-max-seq-len", type=int, default=768)
    parser.add_argument("--pretrain-max-seq-len", type=int, default=340)
    parser.add_argument("--report-sample-limit", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    ocean, ocean_stats = prepare_ocean_sft(args.ocean_sft_source, args.seed)
    dpo, dpo_stats = prepare_dpo(args.ocean_dpo_source, args.seed)
    pretrain_counts, pretrain_sample = prepare_pretrain(args.generic_pretrain_source, args.output_dir, args.seed)

    replay_count = math.ceil(len(ocean["train"]) / 9)
    generic, generic_stats = generic_replay(args.generic_sft_source, replay_count + 1000, args.seed)
    replay, generic_eval = generic[:replay_count], generic[replay_count:]
    outputs = {}
    for split, items in ocean.items():
        outputs[f"ocean_sft_{split}"] = atomic_jsonl(args.output_dir / f"ocean_sft_{split}.jsonl", items)
    outputs["ocean_sft_replay_train"] = atomic_jsonl(
        args.output_dir / "ocean_sft_replay_train.jsonl", [*ocean["train"], *replay]
    )
    outputs["generic_sft_eval"] = atomic_jsonl(args.output_dir / "generic_sft_eval.jsonl", generic_eval)
    for split, items in dpo.items():
        outputs[f"ocean_dpo_{split}"] = atomic_jsonl(args.output_dir / f"ocean_dpo_{split}.jsonl", items)

    pretrain_lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in pretrain_sample]
    prompt_sets = {name: {normalized(prompt_of(item)) for item in items} for name, items in ocean.items()}
    assert not (prompt_sets["train"] & prompt_sets["val"] | prompt_sets["train"] & prompt_sets["test"] | prompt_sets["val"] & prompt_sets["test"])

    manifest = {
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "sources": {
            "ocean_instruct": {"path": str(args.ocean_sft_source), "license": "MIT", "citation": "OceanGPT, arXiv:2310.02031"},
            "legacy_dpo": {"path": str(args.ocean_dpo_source), "provenance": "OceanHeart legacy Qwen2.5-7B listwise judge"},
            "minimind_pretrain": {
                "path": str(args.generic_pretrain_source),
                "license": "see MiniMind upstream dataset notices",
            },
            "minimind_sft": {
                "path": str(args.generic_sft_source),
                "license": "see MiniMind upstream dataset notices",
                "replay_ratio": 0.1,
            },
        },
        "counts": {
            **outputs,
            "pretrain": pretrain_counts,
            "ocean_rejected": dict(ocean_stats),
            "dpo_rejected": dict(dpo_stats),
            "generic_sft_rejected": dict(generic_stats),
        },
        "quality": {
            "ocean_sft_train": describe_chat(ocean["train"], tokenizer, args.sft_max_seq_len, args.report_sample_limit),
            "ocean_dpo_train": describe_chat(dpo["train"], tokenizer, 1024, args.report_sample_limit),
            "pretrain": {
                "sampled": len(pretrain_lengths),
                "language": {
                    name: {"count": count, "ratio": count / len(pretrain_sample)}
                    for name, count in sorted(Counter(language(text) for text in pretrain_sample).items())
                } if pretrain_sample else {},
                "tokens": {
                    "p50": percentile(pretrain_lengths, 0.50),
                    "p95": percentile(pretrain_lengths, 0.95),
                    "p99": percentile(pretrain_lengths, 0.99),
                    "max": max(pretrain_lengths, default=None),
                    "truncation_rate": sum(value > args.pretrain_max_seq_len for value in pretrain_lengths) / len(pretrain_lengths) if pretrain_lengths else 0,
                },
            },
            "prompt_leakage": 0,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, manifest_path)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
