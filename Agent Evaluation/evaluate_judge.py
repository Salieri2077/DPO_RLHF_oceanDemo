#!/usr/bin/env python3
"""Optional DeepSeek V3 audit of the fixed twenty retrieval validation tasks."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash, read_jsonl
from scripts.prepare_ocean_data import atomic_jsonl
from trainer.train_agent import atomic_json
from trainer.trainer_utils import SiliconFlowRewardModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=0)
    args = parser.parse_args()
    tasks = {t["id"]: t for t in read_jsonl(ROOT / "data/processed/ocean_agent_val.jsonl") if t["kind"] == "retrieve"}
    traces = {t["id"]: t for t in read_jsonl(args.traces)}
    key = os.environ.get("SILICONFLOW_API_KEY")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if key:
        judge = SiliconFlowRewardModel(key, "deepseek-ai/DeepSeek-V3")
        for task_id, task in tasks.items():
            if args.deadline and time.time() >= args.deadline:
                break
            trace = traces[task_id]
            row = {"id": task_id, "question": task["question"], "answer": trace["final"], "reference": task["answer"]}
            try:
                row["score"] = judge.score_group(task["question"], task["answer"], [trace["final"]])[0]
            except Exception as exc:
                row["error"] = type(exc).__name__
            rows.append(row)
            atomic_jsonl(args.output.with_suffix(".jsonl"), rows)
    scores = [r["score"] for r in rows if "score" in r]
    atomic_json(args.output, {"status": "completed" if len(scores) == 20 else "partial" if key else "skipped_missing_key",
                             "model": "deepseek-ai/DeepSeek-V3", "trace_sha256": file_hash(args.traces),
                             "requested_questions": 20, "scored": len(scores), "errors": sum("error" in r for r in rows),
                             "score_mean": sum(scores) / len(scores) if scores else None,
                             "warning": "Auxiliary semantic audit only; not the rule reward or proof of scientific correctness."})


if __name__ == "__main__":
    main()
