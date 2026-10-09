#!/usr/bin/env python3
"""Compare Ocean SFT across backbones: MiniMind 64M SFT, Qwen3-1.7B-Base (zero-shot) and Qwen3-1.7B SFT.

Stage `generate` (GPU, no API): on the same fixed ocean test questions it measures
  * reference-answer NLL in nats per character (comparable across tokenizers; per-token PPL is not);
  * greedy generations: EOS rate, length and 6-gram repetition.
Stage `judge` (SiliconFlow): every question's candidates are scored together twice, in a seeded random
order and reversed, and the two scores are averaged to cancel position bias. Paired bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

QWEN = Path("/data/anhuang/oceanheart_models/qwen3-1.7b-base")


def read_conversations(path: Path, limit: int):
    rows = []
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= limit:
                break
            rows.append({"id": index, "conversations": json.loads(line)["conversations"]})
    return rows


def repetition(text: str, n: int = 6) -> float:
    s = "".join(text.split())
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1 - len(set(grams)) / max(len(grams), 1)


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


class Backbone:
    """Uniform prompt/score/generate over MiniMind and Hugging Face models."""

    def __init__(self, kind, adapter=None, device="cuda"):
        self.kind, self.device = kind, device
        if kind == "minimind":
            from transformers import AutoTokenizer
            from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
            self.context = 768  # MiniMind SFT sequence length
            self.model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=1536))
            state = torch.load(ROOT / "out/ocean_sft_replay_768.pth", map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model = self.model.eval().to(device).half()
            self.stop = [self.tokenizer.eos_token_id]
        else:
            from trainer.hf_chat import load_model, load_tokenizer
            self.tokenizer = load_tokenizer(QWEN)
            self.context = 2048
            self.model = load_model(QWEN, adapter, device)
            self.stop = [self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id]

    def prompt(self, conversations):
        if self.kind == "minimind":
            text = self.tokenizer.apply_chat_template(conversations[:-1], tokenize=False, add_generation_prompt=True)
            return text.replace("<think>\n\n</think>\n\n", "")
        from trainer.hf_chat import generation_prompt
        return generation_prompt(conversations)

    @torch.no_grad()
    def answer_nll(self, conversations):
        """Sum NLL of the reference answer's tokens inside the context, and the characters they cover."""
        prompt = self.tokenizer(self.prompt(conversations), add_special_tokens=False).input_ids
        answer = self.tokenizer(conversations[-1]["content"], add_special_tokens=False).input_ids
        answer = answer[:max(self.context - len(prompt), 0)]
        ids = torch.tensor([prompt + answer], device=self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = self.model(ids).logits[0, len(prompt) - 1:-1].float()
        nll = F.cross_entropy(logits, ids[0, len(prompt):], reduction="sum").item()
        return nll, len(self.tokenizer.decode(answer)), len(answer)

    @torch.no_grad()
    def generate(self, conversations, max_new_tokens):
        inputs = self.tokenizer(self.prompt(conversations), return_tensors="pt", add_special_tokens=False).to(self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        if self.kind == "minimind":
            kwargs["eos_token_id"] = self.stop[0]
        else:
            kwargs.update(eos_token_id=self.stop, repetition_penalty=1.0)
        output = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **kwargs)
        new = output[0, inputs.input_ids.shape[1]:].tolist()
        ended = any(token in self.stop for token in new)
        if ended:
            new = new[:min(new.index(t) for t in self.stop if t in new)]
        return self.tokenizer.decode(new, skip_special_tokens=True), ended, len(new)


def stage_generate(args):
    rows = read_conversations(ROOT / "data/processed/ocean_sft_test.jsonl", max(args.nll_samples, args.questions))
    generic = read_conversations(ROOT / "data/processed/generic_sft_eval.jsonl", args.nll_samples)
    models = {"minimind_64m_sft": ("minimind", None), "qwen3_1.7b_base": ("hf", None),
              "qwen3_1.7b_sft": ("hf", args.adapter)}
    generations = {row["id"]: {"id": row["id"], "question": row["conversations"][-2]["content"],
                               "reference": row["conversations"][-1]["content"],
                               "messages": row["conversations"][:-1], "answers": {}} for row in rows[:args.questions]}
    summary = {"adapter": str(args.adapter), "questions": args.questions, "nll_samples": args.nll_samples, "models": {}}
    for name, (kind, adapter) in models.items():
        if args.models and name not in args.models:
            continue
        started = time.time()
        backbone = Backbone(kind, adapter)
        result = {}
        for split, data in (("ocean_test", rows[:args.nll_samples]), ("generic", generic)):
            buckets = {"all": [0., 0, 0], "zh": [0., 0, 0], "en": [0., 0, 0]}
            for row in data:
                nll, chars, tokens = backbone.answer_nll(row["conversations"])
                for key in ("all", "zh" if is_chinese(row["conversations"][-2]["content"]) else "en"):
                    buckets[key][0] += nll
                    buckets[key][1] += chars
                    buckets[key][2] += tokens
            result[split] = {key: {"nats_per_char": n / max(c, 1), "token_ppl": float(torch.tensor(n / max(t, 1)).exp()),
                                   "chars": c} for key, (n, c, t) in buckets.items() if c}
        lengths, reps, ends = [], [], []
        max_new = args.max_new_tokens if kind == "hf" else min(args.max_new_tokens, 768)
        for row in rows[:args.questions]:
            text, ended, count = backbone.generate(row["conversations"], max_new)
            generations[row["id"]]["answers"][name] = {"text": text, "eos": ended, "tokens": count, "rep6": repetition(text)}
            lengths.append(len(text))
            reps.append(repetition(text))
            ends.append(ended)
        reps_sorted = sorted(reps)
        result["generation"] = {"eos_rate": sum(ends) / len(ends), "mean_chars": sum(lengths) / len(lengths),
                                "median_rep6": reps_sorted[len(reps) // 2],
                                "degenerate_rate_rep6_gt_0.3": sum(r > .3 for r in reps) / len(reps),
                                "max_new_tokens": max_new}
        result["seconds"] = time.time() - started
        summary["models"][name] = result
        print(name, json.dumps(result, ensure_ascii=False), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)  # one file per model, so models can run on separate GPUs
        with (args.output / f"generations_{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in generations.values():
                f.write(json.dumps({**row, "answers": {name: row["answers"][name]}}, ensure_ascii=False) + "\n")
        (args.output / f"offline_{name}.json").write_text(json.dumps(
            {**{k: v for k, v in summary.items() if k != "models"}, name: result}, ensure_ascii=False, indent=2) + "\n")
        del backbone
        torch.cuda.empty_cache()


def bootstrap(values, seed, repeats=2000):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats))
    return [means[int(.025 * (repeats - 1))], means[int(.975 * (repeats - 1))]]


def stage_judge(args):
    from trainer.trainer_utils import SiliconFlowRewardModel
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is required for the judge stage")
    judge = SiliconFlowRewardModel(api_key, args.judge_model)
    merged = {}
    for path in sorted(args.output.glob("generations_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            merged.setdefault(row["id"], {**row, "answers": {}})["answers"].update(row["answers"])
    rows = [merged[key] for key in sorted(merged)]
    if any(len(row["answers"]) != 3 for row in rows):
        raise ValueError("expected generations from all three models")
    partial = args.output / f"judged_{args.judge_model.replace('/', '_')}.jsonl"
    done = {json.loads(line)["id"]: json.loads(line) for line in partial.open(encoding="utf-8")} if partial.exists() else {}
    rng = random.Random(args.seed)
    with partial.open("a", encoding="utf-8") as f:
        for row in rows:
            names = sorted(row["answers"])
            order = names[:]
            rng.shuffle(order)
            if row["id"] in done:
                continue
            passes = []
            for sequence in (order, order[::-1]):
                scores = judge.score_group(row["question"], row["reference"], [row["answers"][n]["text"] for n in sequence])
                passes.append(dict(zip(sequence, scores)))
            judged = {"id": row["id"], "order": order, "passes": passes,
                      "scores": {n: (passes[0][n] + passes[1][n]) / 2 for n in names}}
            f.write(json.dumps(judged, ensure_ascii=False) + "\n")
            f.flush()
            done[row["id"]] = judged
    judged = [done[row["id"]] for row in rows]
    names = sorted(judged[0]["scores"])
    summary = {"judge_model": args.judge_model, "questions": len(judged), "protocol": "all candidates per request, "
               "seeded random order + reversed order, averaged", "mean_score": {}, "pairs": {}}
    for name in names:
        values = [j["scores"][name] for j in judged]
        summary["mean_score"][name] = {"mean": sum(values) / len(values), "ci95": bootstrap(values, args.seed)}
    summary["order_disagreement_gt1"] = sum(abs(j["passes"][0][n] - j["passes"][1][n]) > 1 for j in judged for n in names)
    for left, right in (("qwen3_1.7b_base", "qwen3_1.7b_sft"), ("minimind_64m_sft", "qwen3_1.7b_sft")):
        deltas = [j["scores"][right] - j["scores"][left] for j in judged]
        summary["pairs"][f"{right} - {left}"] = {
            "mean_delta": sum(deltas) / len(deltas), "ci95": bootstrap(deltas, args.seed),
            "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas), "ties": sum(d == 0 for d in deltas)}
    (args.output / f"judge_summary_{args.judge_model.replace('/', '_')}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["generate", "judge"], required=True)
    parser.add_argument("--adapter", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_sft_r64_best")
    parser.add_argument("--output", type=Path, default=HERE / "results/backbones")
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--nll_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--judge_model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--models", nargs="*", help="generate only these (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    stage_generate(args) if args.stage == "generate" else stage_judge(args)


if __name__ == "__main__":
    main()
