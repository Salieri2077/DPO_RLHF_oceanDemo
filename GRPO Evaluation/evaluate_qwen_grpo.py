#!/usr/bin/env python3
"""Paired evaluation of the Qwen3-1.7B Ocean SFT model and its GRPO LoRA(s) on fixed Ocean test questions.

All models share the merged SFT weights; the SFT side runs with adapters disabled. Several GRPO adapters
(e.g. two checkpoints of one run) can be compared in the same judge requests with --adapter name=path. Per question and model:
one greedy answer and `samples` answers at the GRPO rollout temperature (GRPO is expected to help sampled answers
most). All candidates of a question are judged in one request, twice (seeded random order, then reversed), and
averaged. The evaluation judge differs from the training judge to limit reward hacking of one judge.
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


def adapters(args):
    """[(name, path)] from repeated --adapter name=path (a bare path is named "grpo")."""
    parsed = []
    for item in args.adapter:
        name, _, path = item.rpartition("=") if "=" in item else ("grpo", "", item)
        parsed.append((name, Path(path)))
    return parsed


def model_names(args):
    return ["sft"] + [name for name, _ in adapters(args)]


def test_questions(limit):
    with (ROOT / "data/processed/ocean_sft_test.jsonl").open(encoding="utf-8") as f:
        return [{"id": i, "conversations": json.loads(line)["conversations"]} for i, line in zip(range(limit), f)]


@torch.no_grad()
def stage_generate(args):
    from peft import PeftModel
    from trainer.hf_chat import generation_prompt, load_model, load_tokenizer
    tokenizer = load_tokenizer(args.merged)
    (first, first_path), *rest = adapters(args)
    model = PeftModel.from_pretrained(load_model(args.merged, None, "cuda"), first_path, adapter_name=first).eval()
    for name, path in rest:
        model.load_adapter(path, adapter_name=name)
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    output = args.output / f"generations_shard{args.shard}.jsonl"
    done = {json.loads(line)["id"] for line in output.open(encoding="utf-8")} if output.exists() else set()
    args.output.mkdir(parents=True, exist_ok=True)

    def decode(sequence):
        sequence = sequence.tolist()
        ended = any(t in stop for t in sequence)
        if ended:
            sequence = sequence[:min(sequence.index(t) for t in stop if t in sequence)]
        return {"text": tokenizer.decode(sequence, skip_special_tokens=True), "eos": ended, "tokens": len(sequence)}

    def answers(inputs, seed):
        common = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop)
        with torch.autocast("cuda", dtype=torch.float16):
            greedy = model.generate(**inputs, do_sample=False, **common)
            torch.manual_seed(seed)
            sampled = model.generate(input_ids=inputs.input_ids.repeat(args.samples, 1),
                                     attention_mask=inputs.attention_mask.repeat(args.samples, 1), do_sample=True,
                                     temperature=args.temperature, top_p=1.0, top_k=0, **common)
        start = inputs.input_ids.shape[1]
        return {"greedy": decode(greedy[0, start:]), "samples": [decode(s[start:]) for s in sampled]}

    with output.open("a", encoding="utf-8") as f:
        for row in test_questions(args.questions)[args.shard::args.num_shards]:
            if row["id"] in done:
                continue
            conversations = row["conversations"]
            inputs = tokenizer(generation_prompt(conversations), return_tensors="pt", add_special_tokens=False).to("cuda")
            seed = args.seed + zlib.crc32(str(row["id"]).encode())
            with model.disable_adapter():
                result = {"sft": answers(inputs, seed)}
            for name, _ in adapters(args):
                model.set_adapter(name)
                result[name] = answers(inputs, seed)
            f.write(json.dumps({"id": row["id"], "question": conversations[-2]["content"], "reference": conversations[-1]["content"],
                                "answers": result}, ensure_ascii=False) + "\n")
            f.flush()
            print(row["id"], flush=True)


def bootstrap(values, seed, repeats=2000):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats))
    return [means[int(.025 * (repeats - 1))], means[int(.975 * (repeats - 1))]]


def candidates(row, names):
    out = []
    for model in names:
        out.append((f"{model}:greedy", row["answers"][model]["greedy"]["text"]))
        out += [(f"{model}:s{i}", s["text"]) for i, s in enumerate(row["answers"][model]["samples"])]
    return out


def stage_judge(args):
    from trainer.trainer_utils import SiliconFlowRewardModel
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise SystemExit("SILICONFLOW_API_KEY is required for the judge stage")
    judge = SiliconFlowRewardModel(key, args.judge_model)
    rows = sorted((json.loads(line) for path in args.output.glob("generations_shard*.jsonl") for line in path.open(encoding="utf-8")),
                  key=lambda r: r["id"])
    path = args.output / f"judged_{args.judge_model.replace('/', '_')}.jsonl"
    done = {json.loads(line)["id"] for line in path.open(encoding="utf-8")} if path.exists() else set()
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            if row["id"] in done:
                continue
            items = candidates(row, model_names(args))
            order = items[:]
            random.Random(f"{args.seed}:{row['id']}").shuffle(order)
            passes = []
            for sequence in (order, order[::-1]):
                scores = judge.score_group(row["question"], row["reference"], [text for _, text in sequence])
                passes.append({name: score for (name, _), score in zip(sequence, scores)})
            f.write(json.dumps({"id": row["id"], "passes": passes,
                                "scores": {n: (passes[0][n] + passes[1][n]) / 2 for n, _ in items}}) + "\n")
            f.flush()
    summarize(args, rows, judge.usage)


def summarize(args, rows, usage=None):
    path = args.output / f"judged_{args.judge_model.replace('/', '_')}.jsonl"
    judged = {json.loads(line)["id"]: json.loads(line) for line in path.open(encoding="utf-8")}
    rows = [r for r in rows if r["id"] in judged]
    names = model_names(args)
    summary = {"judge_model": args.judge_model, "questions": len(rows), "adapters": {n: str(p) for n, p in adapters(args)},
               "samples": args.samples, "temperature": args.temperature, "models": {}, "deltas": {}}
    per = {}
    for model in names:
        greedy = [judged[r["id"]]["scores"][f"{model}:greedy"] for r in rows]
        sampled = [st.mean(judged[r["id"]]["scores"][f"{model}:s{i}"] for i in range(args.samples)) for r in rows]
        worst = [min(judged[r["id"]]["scores"][f"{model}:s{i}"] for i in range(args.samples)) for r in rows]
        meta = [a for r in rows for a in [r["answers"][model]["greedy"]] + r["answers"][model]["samples"]]
        per[model] = {"greedy": greedy, "sampled": sampled, "worst_sample": worst}
        summary["models"][model] = {
            "greedy_mean": st.mean(greedy), "sampled_mean": st.mean(sampled), "worst_sample_mean": st.mean(worst),
            "share_scores_le_minus1": sum(judged[r["id"]]["scores"][f"{model}:{k}"] <= -1 for r in rows
                                          for k in ["greedy"] + [f"s{i}" for i in range(args.samples)]) / (len(rows) * (args.samples + 1)),
            "eos_rate": sum(a["eos"] for a in meta) / len(meta), "mean_tokens": st.mean(a["tokens"] for a in meta)}
    pairs = [(name, "sft") for name in names[1:]] + ([(names[-1], names[1])] if len(names) > 2 else [])
    for new, old in pairs:
        summary["deltas"][f"{new} - {old}"] = {}
        for kind in ("greedy", "sampled", "worst_sample"):
            deltas = [a - b for a, b in zip(per[new][kind], per[old][kind])]
            summary["deltas"][f"{new} - {old}"][kind] = {
                "mean": st.mean(deltas), "ci95": bootstrap(deltas, args.seed), "wins": sum(d > 0 for d in deltas),
                "losses": sum(d < 0 for d in deltas), "ties": sum(d == 0 for d in deltas)}
    summary["order_disagreement_gt1"] = sum(abs(j["passes"][0][n] - j["passes"][1][n]) > 1 for j in judged.values() for n in j["scores"])
    if usage:
        summary["judge_usage_this_run"] = usage
    (args.output / f"summary_{args.judge_model.replace('/', '_')}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["generate", "judge", "summary"], required=True)
    parser.add_argument("--merged", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_sft_r64_eot_merged")
    parser.add_argument("--adapter", action="append", required=True,
                        help="GRPO LoRA adapter as name=path (repeatable); a bare path is named grpo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
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
        rows = [json.loads(line) for path in args.output.glob("generations_shard*.jsonl") for line in path.open(encoding="utf-8")]
        summarize(args, rows)


if __name__ == "__main__":
    main()
