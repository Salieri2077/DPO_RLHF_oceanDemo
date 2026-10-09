#!/usr/bin/env python3
"""Best-of-N headroom check for GRPO on the Qwen3-1.7B Ocean SFT model.

GRPO can only reinforce what the policy already samples. Per question: one greedy answer plus N samples at the
GRPO rollout temperature. A judge scores all N+1 answers of a question in one request, twice (seeded random
order, then reversed) and the two scores are averaged. Reports mean sample, best-of-N, greedy, within-group
spread and how often a group is flat (no signal for GRPO). Held-out val prompts and SFT-train prompts are
reported separately: GRPO prompts come from the SFT training split, which the policy has already imitated.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch

QWEN = Path("/data/anhuang/oceanheart_models/qwen3-1.7b-base")
SPLITS = {"val": ROOT / "data/processed/ocean_grpo_val.jsonl", "train": ROOT / "data/processed/ocean_grpo_train.jsonl"}


def repetition(text: str, n: int = 6) -> float:
    s = "".join(text.split())
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1 - len(set(grams)) / max(len(grams), 1)


def questions(args):
    rows = []
    for split, count in (("val", args.val_questions), ("train", args.train_questions)):
        with SPLITS[split].open(encoding="utf-8") as f:
            for index, line in zip(range(count), f):
                conversations = json.loads(line)["conversations"]
                rows.append({"key": f"{split}-{index}", "split": split, "conversations": conversations,
                             "question": conversations[-2]["content"], "reference": conversations[-1]["content"]})
    return rows


@torch.no_grad()
def stage_generate(args):
    from trainer.hf_chat import generation_prompt, load_model, load_tokenizer
    tokenizer = load_tokenizer(QWEN)
    model = load_model(QWEN, args.adapter, "cuda")
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    rows = questions(args)[args.shard::args.num_shards]
    output = args.output / f"generations_shard{args.shard}.jsonl"
    done = {json.loads(line)["key"] for line in output.open(encoding="utf-8")} if output.exists() else set()
    args.output.mkdir(parents=True, exist_ok=True)

    def decode(sequence):
        sequence = sequence.tolist()
        ended = any(t in stop for t in sequence)
        if ended:
            sequence = sequence[:min(sequence.index(t) for t in stop if t in sequence)]
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        return {"text": text, "eos": ended, "tokens": len(sequence), "rep6": repetition(text)}

    with output.open("a", encoding="utf-8") as f:
        for row in rows:
            if row["key"] in done:
                continue
            inputs = tokenizer(generation_prompt(row["conversations"]), return_tensors="pt", add_special_tokens=False).to("cuda")
            common = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop)
            greedy = model.generate(**inputs, do_sample=False, **common)[0, inputs.input_ids.shape[1]:]
            torch.manual_seed(args.seed + zlib.crc32(row["key"].encode()))  # reproducible per question
            sampled = model.generate(input_ids=inputs.input_ids.repeat(args.n, 1), attention_mask=inputs.attention_mask.repeat(args.n, 1),
                                     do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, **common)
            record = {k: row[k] for k in ("key", "split", "question", "reference")}
            record["greedy"] = decode(greedy)
            record["samples"] = [decode(s[inputs.input_ids.shape[1]:]) for s in sampled]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(record["key"], [(s["tokens"], s["eos"]) for s in record["samples"]], flush=True)


def stage_judge(args):
    from trainer.trainer_utils import SiliconFlowRewardModel
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise SystemExit("SILICONFLOW_API_KEY is required for the judge stage")
    judge = SiliconFlowRewardModel(key, args.judge_model)
    rows = [json.loads(line) for path in sorted(args.output.glob("generations_shard*.jsonl")) for line in path.open(encoding="utf-8")]
    rows.sort(key=lambda r: (r["split"], int(r["key"].split("-")[1])))
    judged_path = args.output / "judged.jsonl"
    done = {json.loads(line)["key"] for line in judged_path.open(encoding="utf-8")} if judged_path.exists() else set()
    with judged_path.open("a", encoding="utf-8") as f:
        for row in rows:
            if row["key"] in done:
                continue
            candidates = [("greedy", row["greedy"]["text"])] + [(f"s{i}", s["text"]) for i, s in enumerate(row["samples"])]
            order = candidates[:]
            random.Random(f"{args.seed}:{row['key']}").shuffle(order)
            passes = []
            for sequence in (order, order[::-1]):
                scores = judge.score_group(row["question"], row["reference"], [text for _, text in sequence])
                passes.append({name: score for (name, _), score in zip(sequence, scores)})
            f.write(json.dumps({"key": row["key"], "passes": passes,
                                "scores": {name: (passes[0][name] + passes[1][name]) / 2 for name, _ in candidates}}) + "\n")
            f.flush()
            print(row["key"], "done", flush=True)
    summarize(args, rows)


def summarize(args, rows):
    judged = {json.loads(line)["key"]: json.loads(line) for line in (args.output / "judged.jsonl").open(encoding="utf-8")}
    summary = {"judge_model": args.judge_model, "n": args.n, "temperature": args.temperature,
               "max_new_tokens": args.max_new_tokens, "adapter": str(args.adapter), "splits": {}}
    pairs = []  # (score, rep6, tokens) for samples, to inform reward shaping
    for split in ("val", "train"):
        group = [r for r in rows if r["split"] == split and r["key"] in judged]
        if not group:
            continue
        means, bests, worsts, greedys, stds, spreads = [], [], [], [], [], []
        for row in group:
            scores = judged[row["key"]]["scores"]
            samples = [scores[f"s{i}"] for i in range(len(row["samples"]))]
            means.append(st.mean(samples))
            bests.append(max(samples))
            worsts.append(min(samples))
            greedys.append(scores["greedy"])
            stds.append(st.pstdev(samples))
            spreads.append(max(samples) - min(samples))
            pairs += [(scores[f"s{i}"], s["rep6"], s["tokens"]) for i, s in enumerate(row["samples"])]
        sample_meta = [s for row in group for s in row["samples"]]
        summary["splits"][split] = {
            "questions": len(group),
            "mean_sample_score": st.mean(means), "mean_best_of_n": st.mean(bests), "mean_worst_of_n": st.mean(worsts),
            "mean_greedy": st.mean(greedys), "mean_within_group_std": st.mean(stds),
            "flat_groups_std_lt_0.25": sum(s < .25 for s in stds) / len(stds),
            "groups_spread_ge_2": sum(s >= 2 for s in spreads) / len(spreads),
            "groups_best_ge_2.5": sum(b >= 2.5 for b in bests) / len(bests),
            "groups_best_le_0": sum(b <= 0 for b in bests) / len(bests),
            "sample_eos_rate": sum(s["eos"] for s in sample_meta) / len(sample_meta),
            "sample_mean_tokens": st.mean(s["tokens"] for s in sample_meta),
        }
    if len(pairs) > 2:
        scores, reps, tokens = zip(*pairs)
        def corr(a, b):
            return st.correlation(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else None
        summary["sample_correlations"] = {"score_vs_rep6": corr(scores, reps), "score_vs_tokens": corr(scores, tokens)}
    summary["order_disagreement_gt1"] = sum(abs(j["passes"][0][n] - j["passes"][1][n]) > 1 for j in judged.values() for n in j["scores"])
    summary["scored_candidates"] = sum(len(j["scores"]) for j in judged.values())
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["generate", "judge", "summary"], required=True)
    parser.add_argument("--adapter", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_sft_r64_eot_best")
    parser.add_argument("--output", type=Path, default=HERE / "results/qwen_best_of_8")
    parser.add_argument("--val_questions", type=int, default=20)
    parser.add_argument("--train_questions", type=int, default=10)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--judge_model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stage == "generate":
        stage_generate(args)
    elif args.stage == "judge":
        stage_judge(args)
    else:
        rows = [json.loads(line) for path in sorted(args.output.glob("generations_shard*.jsonl")) for line in path.open(encoding="utf-8")]
        summarize(args, rows)


if __name__ == "__main__":
    main()
