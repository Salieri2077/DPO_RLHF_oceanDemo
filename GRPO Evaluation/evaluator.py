#!/usr/bin/env python3
"""Paired MoE SFT versus GRPO evaluation on fixed ocean questions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import SiliconFlowRewardModel, setup_seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def load_questions(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            conversations = json.loads(line)["conversations"]
            rows.append({
                "id": index,
                "messages": conversations[:-1],
                "question": next(item["content"] for item in reversed(conversations) if item["role"] == "user"),
                "reference": conversations[-1]["content"],
            })
            if len(rows) >= limit:
                break
    if len(rows) != limit:
        raise ValueError(f"requested {limit} questions, found {len(rows)} in {path}")
    return rows


def load_model(checkpoint: Path, args) -> MiniMindForCausalLM:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        max_seq_len=args.max_seq_len + args.max_new_tokens,
        use_moe=True,
    )
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(args.device)
    return model.half() if args.device.startswith("cuda") else model


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], args) -> tuple[str, float]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text.replace("<think>\n\n</think>\n\n", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
    started = time.time()
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    elapsed = max(time.time() - started, 1e-6)
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens) / elapsed


def bootstrap_mean_delta(deltas: list[float], seed: int, repeats: int) -> list[float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(repeats))
    return [means[int(0.025 * (repeats - 1))], means[int(0.975 * (repeats - 1))]]


def summarize(rows: list[dict], seed: int, repeats: int) -> dict:
    sft = [row["judge_scores"]["sft"] for row in rows]
    grpo = [row["judge_scores"]["grpo"] for row in rows]
    deltas = [right - left for left, right in zip(sft, grpo)]
    wins = {
        "sft": sum(delta < 0 for delta in deltas),
        "grpo": sum(delta > 0 for delta in deltas),
        "tie": sum(delta == 0 for delta in deltas),
    }
    return {
        "samples": len(rows),
        "position_balanced": True,
        "score_definition": "GRPO minus SFT; positive favors GRPO",
        "mean_score": {"sft": sum(sft) / len(sft), "grpo": sum(grpo) / len(grpo)},
        "mean_score_delta": sum(deltas) / len(deltas),
        "mean_score_delta_ci95": bootstrap_mean_delta(deltas, seed, repeats),
        "wins": wins,
        "grpo_win_rate": wins["grpo"] / len(rows),
        "grpo_non_tie_win_rate": wins["grpo"] / max(wins["grpo"] + wins["sft"], 1),
        "paired_bootstrap_repeats": repeats,
    }


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    write_jsonl(output_dir / "generations_and_scores.jsonl", rows)
    columns = [
        "id", "question", "reference", "sft_answer", "grpo_answer",
        "sft_score", "grpo_score", "score_delta_grpo_minus_sft", "winner", "candidate_order",
    ]
    csv_rows = []
    for row in rows:
        csv_rows.append({
            **{key: row[key] for key in ("id", "question", "reference", "sft_answer", "grpo_answer")},
            "sft_score": row["judge_scores"]["sft"],
            "grpo_score": row["judge_scores"]["grpo"],
            "score_delta_grpo_minus_sft": row["score_delta_grpo_minus_sft"],
            "winner": row["winner"],
            "candidate_order": ",".join(row["candidate_order"]),
        })
    temp = output_dir / "comparison.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(temp, output_dir / "comparison.csv")
    atomic_json(output_dir / "summary.json", summary)
    ci = summary["judge"]["mean_score_delta_ci95"]
    lines = [
        "# MoE SFT vs MoE GRPO", "",
        "Judge 分差定义为 GRPO - SFT，正值支持 GRPO。", "",
        f"- 样本数：{summary['judge']['samples']}",
        f"- SFT / GRPO 平均分：{summary['judge']['mean_score']['sft']:.4f} / {summary['judge']['mean_score']['grpo']:.4f}",
        f"- 平均分差：{summary['judge']['mean_score_delta']:+.4f}（95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]）",
        f"- 胜负：GRPO {summary['judge']['wins']['grpo']} / SFT {summary['judge']['wins']['sft']} / 平局 {summary['judge']['wins']['tie']}",
        f"- GRPO 非平局胜率：{summary['judge']['grpo_non_tie_win_rate']:.2%}", "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/processed/ocean_grpo_val.jsonl")
    parser.add_argument("--sft-checkpoint", type=Path, default=ROOT / "out/ocean_sft_replay_768_moe.pth")
    parser.add_argument("--grpo-checkpoint", type=Path, default=ROOT / "out/ocean_grpo_eval50_768_moe.pth")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/moe")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge", choices=["none", "siliconflow"], default="siliconflow")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.bootstrap_repeats < 1:
        raise SystemExit("--samples and --bootstrap-repeats must be positive")
    setup_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    rows = load_questions(args.data_path, args.samples)
    speeds = {}
    for name, checkpoint in (("sft", args.sft_checkpoint), ("grpo", args.grpo_checkpoint)):
        model = load_model(checkpoint, args)
        model_speeds = []
        for index, row in enumerate(rows, 1):
            row[f"{name}_answer"], speed = generate(model, tokenizer, row["messages"], args)
            model_speeds.append(speed)
            print(f"{name}: generated {index}/{len(rows)}", flush=True)
        speeds[name] = sum(model_speeds) / len(model_speeds)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_jsonl(args.output_dir / "generations.partial.jsonl", rows)

    if args.judge == "none":
        print(f"generations: {args.output_dir / 'generations.partial.jsonl'}")
        return
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is required for --judge siliconflow")
    judge = SiliconFlowRewardModel(api_key, args.judge_model)
    for index, row in enumerate(rows):
        order = ("sft", "grpo") if index % 2 == 0 else ("grpo", "sft")
        scores = judge.score_group(row["question"], row["reference"], [row[f"{name}_answer"] for name in order])
        row["candidate_order"] = list(order)
        row["judge_scores"] = dict(zip(order, scores))
        row["score_delta_grpo_minus_sft"] = row["judge_scores"]["grpo"] - row["judge_scores"]["sft"]
        row["winner"] = "grpo" if row["score_delta_grpo_minus_sft"] > 0 else "sft" if row["score_delta_grpo_minus_sft"] < 0 else "tie"
        write_jsonl(args.output_dir / "judged.partial.jsonl", rows[:index + 1])
        print(f"judge: scored {index + 1}/{len(rows)}", flush=True)

    judge_summary = summarize(rows, args.seed, args.bootstrap_repeats)
    summary = {
        "protocol": {
            "questions": "first N rows from the fixed GRPO validation split",
            "decoding": "greedy",
            "position_balanced": True,
            "judge_model": args.judge_model,
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
        },
        "data": {"path": str(args.data_path.resolve()), "sha256": sha256(args.data_path)},
        "checkpoints": {
            "sft": {"path": str(args.sft_checkpoint.resolve()), "sha256": sha256(args.sft_checkpoint)},
            "grpo": {"path": str(args.grpo_checkpoint.resolve()), "sha256": sha256(args.grpo_checkpoint)},
        },
        "generation_tokens_per_second": speeds,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "judge": judge_summary,
    }
    write_outputs(rows, summary, args.output_dir)
    print(f"comparison: {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
