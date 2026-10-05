#!/usr/bin/env python3
"""Evaluate OceanHeart Dense/MoE base checkpoints with candidate likelihood."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


DEFAULT_DATA = Path("/data/anhuang/oceanheart_pretrain_eval")
CHECKPOINTS = {"dense": ROOT / "out/pretrain_768.pth", "moe": ROOT / "out/pretrain_768_moe.pth"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def load_rows(data_root: Path, max_samples: int):
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for name, info in manifest["benchmarks"].items():
        path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise RuntimeError(f"benchmark changed after preparation: {path}")
        benchmark_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.extend(benchmark_rows[:max_samples or None])
    return rows, manifest_path, sha256(manifest_path)


def load_model(kind: str, checkpoint: Path, device: torch.device):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=kind == "moe")
    model = MiniMindForCausalLM(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    total = sum(parameter.numel() for parameter in model.parameters())
    expert = sum(parameter.numel() for name, parameter in model.named_parameters() if ".experts." in name)
    active = total if kind == "dense" else total - expert + expert * config.num_experts_per_tok / config.num_experts
    return model, total, int(active)


def encoded_candidates(rows, tokenizer, max_seq_len: int):
    encoded = []
    bos = tokenizer.bos_token_id
    if bos is None:
        raise RuntimeError("tokenizer must define bos_token_id")
    for row in rows:
        context = tokenizer.encode(row["prompt"], add_special_tokens=False)
        for choice_index, choice in enumerate(row["choices"]):
            continuation = tokenizer.encode(row.get("choice_prefix", "") + choice, add_special_tokens=False)
            if not continuation or len(continuation) >= max_seq_len:
                raise ValueError(f"invalid/too-long continuation: {row['benchmark']}:{row['id']}:{choice_index}")
            kept_context = context[-(max_seq_len - len(continuation) - 1):]
            ids = [bos] + kept_context + continuation
            encoded.append({
                "key": f"{row['benchmark']}:{row['id']}", "choice": choice_index,
                "ids": ids, "context_len": 1 + len(kept_context),
                "continuation_tokens": len(continuation), "truncated": len(kept_context) < len(context),
            })
    return encoded


@torch.inference_mode()
def score_candidates(model, candidates, pad_id: int, batch_size: int, device: torch.device):
    scores = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        width = max(len(item["ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        for index, item in enumerate(batch):
            input_ids[index, :len(item["ids"])] = torch.tensor(item["ids"], device=device)
        logits = model(input_ids=input_ids).logits[:, :-1].float().log_softmax(-1)
        targets = input_ids[:, 1:]
        token_logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for index, item in enumerate(batch):
            first = item["context_len"] - 1
            last = len(item["ids"]) - 1
            values = token_logp[index, first:last]
            scores[(item["key"], item["choice"])] = (values.sum().item(), values.mean().item(), item)
    return scores


def predictions(rows, scores):
    output = []
    for row in rows:
        key = f"{row['benchmark']}:{row['id']}"
        choice_scores = [scores[(key, index)] for index in range(len(row["choices"]))]
        sum_scores = [value[0] for value in choice_scores]
        mean_scores = [value[1] for value in choice_scores]
        answer = row["answer"]
        output.append({
            "id": key, "benchmark": row["benchmark"], "domain": row["domain"], "answer": answer,
            "prediction": int(np.argmax(sum_scores)), "prediction_norm": int(np.argmax(mean_scores)),
            "correct": int(np.argmax(sum_scores) == answer), "correct_norm": int(np.argmax(mean_scores) == answer),
            "choice_loglikelihood": sum_scores, "choice_mean_loglikelihood": mean_scores,
            "correct_nll": -sum_scores[answer], "correct_tokens": choice_scores[answer][2]["continuation_tokens"],
            "context_truncated": any(value[2]["truncated"] for value in choice_scores),
        })
    return output


def metric(rows):
    count = len(rows)
    tokens = sum(row["correct_tokens"] for row in rows)
    nll = sum(row["correct_nll"] for row in rows)
    return {
        "samples": count, "accuracy": sum(row["correct"] for row in rows) / count,
        "accuracy_norm": sum(row["correct_norm"] for row in rows) / count,
        "correct_choice_nll": nll / tokens, "correct_choice_ppl": math.exp(min(nll / tokens, 50)),
        "truncation_rate": sum(row["context_truncated"] for row in rows) / count,
    }


def aggregate(rows):
    groups = {"benchmark": defaultdict(list), "domain": defaultdict(list)}
    for row in rows:
        groups["benchmark"][row["benchmark"]].append(row)
        groups["domain"][row["domain"]].append(row)
    result = {kind: {name: metric(items) for name, items in values.items()} for kind, values in groups.items()}
    result["overall"] = {
        "samples": len(rows),
        "macro_domain_accuracy_norm": float(np.mean([item["accuracy_norm"] for item in result["domain"].values()])),
        "micro_accuracy_norm": sum(row["correct_norm"] for row in rows) / len(rows),
        "correct_choice_nll": sum(row["correct_nll"] for row in rows) / sum(row["correct_tokens"] for row in rows),
    }
    result["overall"]["correct_choice_ppl"] = math.exp(min(result["overall"]["correct_choice_nll"], 50))
    return result


def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def evaluate(kind: str, checkpoint: Path, args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows, manifest_path, manifest_hash = load_rows(args.data_root, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    device = torch.device(args.device)
    model, total, active = load_model(kind, checkpoint, device)
    started = time.time()
    candidates = encoded_candidates(rows, tokenizer, args.max_seq_len)
    scored = score_candidates(model, candidates, tokenizer.pad_token_id or tokenizer.eos_token_id, args.batch_size, device)
    raw = predictions(rows, scored)
    output_dir = args.results_dir / kind
    write_jsonl(output_dir / "predictions.jsonl", raw)
    tokenizer_files = [ROOT / "model/tokenizer.json", ROOT / "model/tokenizer_config.json"]
    summary = {
        "model": kind, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "git_commit": git_commit(), "data_manifest": str(manifest_path), "data_manifest_sha256": manifest_hash,
        "tokenizer_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in tokenizer_files)).hexdigest(),
        "protocol": {"method": "candidate_loglikelihood", "primary_metric": "length_normalized_accuracy",
                     "seed": args.seed, "max_seq_len": args.max_seq_len, "max_samples_per_benchmark": args.max_samples,
                     "batch_size": args.batch_size, "dtype": "float16" if device.type == "cuda" else "float32"},
        "parameters": {"total": total, "activated_per_token": active}, "elapsed_seconds": time.time() - started,
        "metrics": aggregate(raw),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(f"{kind}: {summary['metrics']['overall']['macro_domain_accuracy_norm']:.4f} macro-domain accuracy")
    return summary


def bootstrap_delta(dense, moe, seed: int, repeats: int = 2000):
    delta = np.array([m - d for d, m in zip(dense, moe)], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(repeats, len(delta)), replace=True).mean(axis=1)
    return float(delta.mean()), [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def compare(results_dir: Path, seed: int):
    summaries = {kind: json.loads((results_dir / kind / "summary.json").read_text()) for kind in ("dense", "moe")}
    if (summaries["dense"]["data_manifest_sha256"] != summaries["moe"]["data_manifest_sha256"]
            or summaries["dense"]["tokenizer_sha256"] != summaries["moe"]["tokenizer_sha256"]
            or summaries["dense"]["protocol"] != summaries["moe"]["protocol"]):
        raise RuntimeError("Dense and MoE were not evaluated with the same data/tokenizer/protocol")
    raw = {}
    for kind in ("dense", "moe"):
        raw[kind] = {row["id"]: row for row in map(json.loads, (results_dir / kind / "predictions.jsonl").read_text().splitlines())}
    if raw["dense"].keys() != raw["moe"].keys():
        raise RuntimeError("Dense and MoE prediction IDs differ")
    all_ids = list(raw["dense"])
    delta, interval = bootstrap_delta([raw["dense"][key]["correct_norm"] for key in all_ids],
                                      [raw["moe"][key]["correct_norm"] for key in all_ids], seed)
    comparisons = [{
        "group": "overall", "name": "All samples (micro)", "samples": len(all_ids),
        "dense_accuracy_norm": summaries["dense"]["metrics"]["overall"]["micro_accuracy_norm"],
        "moe_accuracy_norm": summaries["moe"]["metrics"]["overall"]["micro_accuracy_norm"],
        "delta_moe_minus_dense": delta, "ci95": interval,
        "winner": "MoE" if interval[0] > 0 else "Dense" if interval[1] < 0 else "Inconclusive",
        "dense_ppl": summaries["dense"]["metrics"]["overall"]["correct_choice_ppl"],
        "moe_ppl": summaries["moe"]["metrics"]["overall"]["correct_choice_ppl"],
    }]
    for group in ("domain", "benchmark"):
        names = summaries["dense"]["metrics"][group]
        for name in names:
            ids = [key for key, row in raw["dense"].items() if row[group] == name]
            delta, interval = bootstrap_delta([raw["dense"][key]["correct_norm"] for key in ids],
                                              [raw["moe"][key]["correct_norm"] for key in ids], seed)
            winner = "MoE" if interval[0] > 0 else "Dense" if interval[1] < 0 else "Inconclusive"
            comparisons.append({
                "group": group, "name": name, "samples": len(ids),
                "dense_accuracy_norm": summaries["dense"]["metrics"][group][name]["accuracy_norm"],
                "moe_accuracy_norm": summaries["moe"]["metrics"][group][name]["accuracy_norm"],
                "delta_moe_minus_dense": delta, "ci95": interval, "winner": winner,
                "dense_ppl": summaries["dense"]["metrics"][group][name]["correct_choice_ppl"],
                "moe_ppl": summaries["moe"]["metrics"][group][name]["correct_choice_ppl"],
            })
    comparison = {"primary_metric": "length_normalized_accuracy", "paired_bootstrap_repeats": 2000,
                  "overall": {kind: summaries[kind]["metrics"]["overall"] for kind in summaries}, "comparisons": comparisons}
    atomic_json(results_dir / "comparison.json", comparison)
    with (results_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "name", "samples", "dense_accuracy_norm", "moe_accuracy_norm",
                                                        "delta_moe_minus_dense", "ci95_low", "ci95_high", "winner", "dense_ppl", "moe_ppl"])
        writer.writeheader()
        for row in comparisons:
            writer.writerow({**{key: value for key, value in row.items() if key != "ci95"}, "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1]})
    lines = ["# Dense vs MoE", "", "Primary metric: length-normalized candidate accuracy. Delta is MoE minus Dense.", "",
             "| Group | Domain / benchmark | N | Dense | MoE | Delta (pp) | 95% CI (pp) | Conclusion |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in comparisons:
        lines.append(f"| {row['group']} | {row['name']} | {row['samples']} | {row['dense_accuracy_norm']:.3f} | "
                     f"{row['moe_accuracy_norm']:.3f} | {100 * row['delta_moe_minus_dense']:+.1f} | "
                     f"[{100 * row['ci95'][0]:+.1f}, {100 * row['ci95'][1]:+.1f}] | {row['winner']} |")
    (results_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"comparison: {results_dir / 'comparison.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dense", "moe", "all"], required=True)
    parser.add_argument("--dense-checkpoint", type=Path, default=CHECKPOINTS["dense"])
    parser.add_argument("--moe-checkpoint", type=Path, default=CHECKPOINTS["moe"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=340)
    parser.add_argument("--max-samples", type=int, default=0, help="per benchmark; 0 evaluates all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    kinds = ("dense", "moe") if args.model == "all" else (args.model,)
    for kind in kinds:
        checkpoint = getattr(args, f"{kind}_checkpoint")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        evaluate(kind, checkpoint, args)
    if args.model == "all":
        compare(args.results_dir, args.seed)
    elif all((args.results_dir / kind / "summary.json").is_file() for kind in ("dense", "moe")):
        try:
            compare(args.results_dir, args.seed)
        except RuntimeError as error:
            print(f"comparison skipped until the other model uses the same data/protocol: {error}")


if __name__ == "__main__":
    main()
