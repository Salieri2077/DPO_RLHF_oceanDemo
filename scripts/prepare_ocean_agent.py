#!/usr/bin/env python3
"""Build deterministic, split-isolated OceanHeart tool tasks and demonstrations."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from agent.ocean import VERSION, OceanTools, calculate, demonstration, file_hash, read_jsonl
from scripts.prepare_ocean_data import atomic_jsonl, normalized, percentile, stable_score


def prepare(directory):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    rng, used_parameters, used_sources = random.Random(42), set(), set()
    manifest = {"version": VERSION, "seed": 42, "source_license": "MIT", "source": "zjunlp/OceanInstruct-v0.2",
                "warning": "OceanInstruct includes synthetic/errors; cards are simulated, not observations. Open-book tool benchmark, not closed-book knowledge.",
                "tool_sha256": file_hash(ROOT / "agent/ocean.py"), "splits": {}}
    for split, size in (("train", 2000), ("val", 50), ("test", 50)):
        source = directory / f"ocean_sft_{split}.jsonl"
        records = sorted(read_jsonl(source), key=lambda r: stable_score(r["conversations"][1]["content"], 42))
        documents, tasks = [], []
        for row in records:
            question, answer = [m["content"] for m in row["conversations"] if m["role"] != "system"]
            sentences = re.split(r"(?<=[。！？.!?])\s*|\n", answer.strip())
            excerpt = next((s.strip() for s in sentences if 12 <= len(s.strip()) <= 80 and len(tokenizer.encode(s.strip())) <= 100), "")
            source_id = hashlib.sha256(normalized(question).encode()).hexdigest()
            if not (3 <= len(question) <= 120 and 12 <= len(excerpt) <= 80):
                continue
            if any(x in question + excerpt for x in ("<|", "<tool", "\ufffd")) or len(tokenizer.encode(excerpt)) > 100:
                continue
            if source_id in used_sources:
                continue
            used_sources.add(source_id)
            doc_id = "O" + source_id[:10]
            documents.append({"id": doc_id, "title": question, "excerpt": excerpt, "source": "OceanInstruct-v0.2", "source_id": source_id})
            tasks.append({"id": f"{split}-{doc_id}", "kind": "retrieve", "query": question, "doc_id": doc_id,
                          "question": f"检索资料：{question}\n请原样摘录返回的目标资料原文，并在末尾标注[文档编号]。", "answer": excerpt})
            if len(tasks) == size * 2 // 5:
                break
        if len(tasks) != size * 2 // 5:
            raise ValueError(f"Not enough short source records for {split}")
        for kind, count in (("calculate", size * 2 // 5), ("chain", size // 5)):
            for i in range(count):
                while True:
                    speed, hours, distance = rng.randint(40, 400) / 10, rng.randint(1, 240) / 10, rng.randint(10, 3000) / 10
                    operation = "distance" if kind == "chain" else ("convert", "distance", "time", "speed")[i % 4]
                    if operation == "convert":
                        pairs = [("kn", "km/h"), ("nmi", "km"), ("m/s", "km/h"), ("km", "nmi")]
                        fr, to = pairs[(i // 4) % len(pairs)]
                        args = {"operation": operation, "value": distance, "from_unit": fr, "to_unit": to}
                        question = f"请用工具将 {distance:g} {fr} 换算成 {to}。只回答数值和单位。"
                    elif operation == "distance":
                        args = {"operation": operation, "speed": speed, "hours": hours}
                        question = f"调查船以 {speed:g} km/h 航行 {hours:g} h，用工具计算航程。只回答数值和单位。"
                    elif operation == "time":
                        args = {"operation": operation, "distance": distance, "speed": speed}
                        question = f"调查船航程 {distance:g} km，速度 {speed:g} km/h，用工具计算时间。只回答数值和单位。"
                    else:
                        args = {"operation": operation, "distance": distance, "hours": hours}
                        question = f"调查船用 {hours:g} h 航行 {distance:g} km，用工具计算速度。只回答数值和单位。"
                    fingerprint = json.dumps(args, sort_keys=True)
                    if fingerprint not in used_parameters:
                        used_parameters.add(fingerprint)
                        break
                task_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:10]
                task = {"id": f"{split}-N{task_id}", "kind": kind, "question": question,
                        "calc": args, "expected": calculate(args), "parameter_key": fingerprint, "synthetic": True}
                if kind == "chain":
                    doc_id = "C" + task_id
                    title = f"模拟航次卡 {doc_id}"
                    documents.append({"id": doc_id, "title": title, "excerpt": f"模拟练习，非实测：speed={speed:g} km/h，hours={hours:g} h。", "source": "synthetic-v1"})
                    task.update(doc_id=doc_id, query=title, question=f"先检索{title}，再用计算工具计算该船航程。只回答数值和单位。")
                tasks.append(task)
        environment = OceanTools(documents)
        tasks.sort(key=lambda t: stable_score(t["id"], 42))
        demos, lengths = [], []
        for task in tasks:
            demo = demonstration(task, environment, tokenizer)
            ids = tokenizer.apply_chat_template(demo["conversations"], tools=json.loads(demo["conversations"][0]["tools"]), tokenize=True)
            if len(ids) > 2048:
                raise ValueError(f"Demonstration exceeds context budget: {task['id']}")
            demos.append(demo)
            lengths.append(len(ids))
        outputs = {f"ocean_agent_{split}.jsonl": tasks, f"ocean_agent_sft_{split}.jsonl": demos,
                   f"ocean_agent_corpus_{split}.jsonl": documents}
        for name, rows in outputs.items():
            atomic_jsonl(directory / name, rows)
        manifest["splits"][split] = {"counts": dict(Counter(t["kind"] for t in tasks)), "source_sha256": file_hash(source),
                                    "artifacts": {name: file_hash(directory / name) for name in outputs},
                                    "tokens": {"p50": percentile(lengths, .5), "p95": percentile(lengths, .95), "max": max(lengths), "truncation_rate": 0}}
    path = directory / "agent_manifest.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    prepare(parser.parse_args().input_dir)
