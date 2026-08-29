#!/usr/bin/env python3
"""Compare OceanHeart Pretrain, Pure SFT and Replay SFT checkpoints."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import SiliconFlowRewardModel


SYSTEM_PROMPT = (
    "你是 OceanHeart，一个面向海洋科学任务的双语语言模型助手。"
    "请基于可靠知识准确回答；不确定时明确说明。"
)
MODEL_NAMES = ("pretrain", "ocean_sft_pure", "ocean_sft_replay")
DISPLAY_NAMES = {"pretrain": "Pretrain", "ocean_sft_pure": "Pure", "ocean_sft_replay": "Replay"}


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


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def load_benchmarks(data_root: Path, max_samples: int):
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for info in manifest["benchmarks"].values():
        path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise RuntimeError(f"benchmark changed after preparation: {path}")
        benchmark_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.extend(benchmark_rows[:max_samples or None])
    return rows, manifest_path, sha256(manifest_path)


def checkpoint_path(name: str, architecture: str, save_dir: Path, hidden_size: int) -> Path:
    suffix = "_moe" if architecture == "moe" else ""
    return save_dir / f"{name}_{hidden_size}{suffix}.pth"


def load_model(checkpoint: Path, architecture: str, hidden_size: int, layers: int, device: torch.device):
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=layers, use_moe=architecture == "moe")
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, total


@torch.inference_mode()
def assistant_nll(model, dataset, batch_size: int, device: torch.device, max_samples: int):
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows, total_nll, total_tokens = [], 0.0, 0
    sample_index = 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids=input_ids).logits[:, :-1].float()
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        safe_targets = targets.masked_fill(~mask, 0)
        token_nll = -F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        for index in range(input_ids.size(0)):
            tokens = int(mask[index].sum().item())
            nll_sum = float((token_nll[index] * mask[index]).sum().item())
            rows.append({"id": sample_index, "tokens": tokens, "nll": nll_sum / max(tokens, 1)})
            total_nll += nll_sum
            total_tokens += tokens
            sample_index += 1
    nll = total_nll / max(total_tokens, 1)
    return rows, {"samples": len(rows), "tokens": total_tokens, "nll": nll, "perplexity": math.exp(min(nll, 50))}


def chat_candidates(rows, tokenizer, max_seq_len: int):
    encoded = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        ).replace("<think>\n\n</think>\n\n", "")
        context = tokenizer.encode(prompt, add_special_tokens=False)
        for choice_index, choice in enumerate(row["choices"]):
            continuation = tokenizer.encode(
                row.get("choice_prefix", "") + choice + tokenizer.eos_token, add_special_tokens=False
            )
            if not continuation or len(continuation) >= max_seq_len:
                raise ValueError(f"invalid/too-long continuation: {row['benchmark']}:{row['id']}:{choice_index}")
            keep = max_seq_len - len(continuation)
            kept_context = context[-keep:]
            encoded.append({
                "key": f"{row['benchmark']}:{row['id']}",
                "choice": choice_index,
                "ids": kept_context + continuation,
                "context_len": len(kept_context),
                "continuation_tokens": len(continuation),
                "truncated": len(kept_context) < len(context),
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
            length = len(item["ids"])
            input_ids[index, :length] = torch.tensor(item["ids"], device=device)
        # Padding is only appended after each candidate, so it cannot affect earlier causal logits.
        logits = model(input_ids=input_ids).logits[:, :-1].float().log_softmax(-1)
        targets = input_ids[:, 1:]
        token_logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for index, item in enumerate(batch):
            first, last = item["context_len"] - 1, len(item["ids"]) - 1
            values = token_logp[index, first:last]
            scores[(item["key"], item["choice"])] = (values.sum().item(), values.mean().item(), item)
    return scores


def benchmark_predictions(rows, scores):
    output = []
    for row in rows:
        key = f"{row['benchmark']}:{row['id']}"
        values = [scores[(key, index)] for index in range(len(row["choices"]))]
        sums, means = [item[0] for item in values], [item[1] for item in values]
        answer = row["answer"]
        output.append({
            "id": key,
            "benchmark": row["benchmark"],
            "domain": row["domain"],
            "answer": answer,
            "prediction": int(np.argmax(sums)),
            "prediction_norm": int(np.argmax(means)),
            "correct": int(np.argmax(sums) == answer),
            "correct_norm": int(np.argmax(means) == answer),
            "choice_loglikelihood": sums,
            "choice_mean_loglikelihood": means,
            "correct_nll": -sums[answer],
            "correct_tokens": values[answer][2]["continuation_tokens"],
            "context_truncated": any(item[2]["truncated"] for item in values),
        })
    return output


def benchmark_metric(rows):
    tokens = sum(row["correct_tokens"] for row in rows)
    nll = sum(row["correct_nll"] for row in rows) / max(tokens, 1)
    return {
        "samples": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "accuracy_norm": sum(row["correct_norm"] for row in rows) / len(rows),
        "correct_choice_nll": nll,
        "correct_choice_ppl": math.exp(min(nll, 50)),
        "truncation_rate": sum(row["context_truncated"] for row in rows) / len(rows),
    }


def aggregate_benchmarks(rows):
    groups = {"benchmark": defaultdict(list), "domain": defaultdict(list)}
    for row in rows:
        groups["benchmark"][row["benchmark"]].append(row)
        groups["domain"][row["domain"]].append(row)
    result = {kind: {name: benchmark_metric(items) for name, items in values.items()} for kind, values in groups.items()}
    result["overall"] = benchmark_metric(rows)
    result["overall"]["macro_domain_accuracy_norm"] = float(
        np.mean([item["accuracy_norm"] for item in result["domain"].values()])
    )
    return result


def prompts_with_references(path: Path, limit: int):
    if limit <= 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            conversations = json.loads(line)["conversations"]
            rows.append({
                "id": index,
                "prompt": next(item["content"] for item in conversations if item["role"] == "user"),
                "reference": next(item["content"] for item in reversed(conversations) if item["role"] == "assistant"),
            })
            if len(rows) >= limit:
                break
    return rows


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_seq_len: int, max_new_tokens: int, device: torch.device):
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    ).replace("<think>\n\n</think>\n\n", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
    started = time.time()
    output = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens) / max(time.time() - started, 1e-6)


def bootstrap_delta(left, right, seed: int, repeats: int = 2000):
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    if not len(delta):
        raise ValueError("paired bootstrap requires at least one sample")
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(repeats, len(delta)), replace=True).mean(axis=1)
    return float(delta.mean()), [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def compare(raw, summaries, seed: int, output_dir: Path):
    pure, replay = "ocean_sft_pure", "ocean_sft_replay"
    comparisons = []
    for dataset in ("ocean", "generic"):
        left = raw[pure][dataset]
        right = raw[replay][dataset]
        if [row["id"] for row in left] != [row["id"] for row in right]:
            raise RuntimeError(f"{dataset} evaluation sample IDs differ")
        delta, interval = bootstrap_delta([row["nll"] for row in left], [row["nll"] for row in right], seed)
        comparisons.append({
            "group": "assistant_nll", "name": dataset, "samples": len(left),
            "pure": summaries[pure][dataset]["nll"], "replay": summaries[replay][dataset]["nll"],
            "delta_replay_minus_pure": delta, "ci95": interval,
            "winner": "Replay" if interval[1] < 0 else "Pure" if interval[0] > 0 else "Inconclusive",
        })
    for group, name in [("overall", "All samples"), *[("domain", key) for key in summaries[pure]["benchmarks"]["domain"]],
                        *[("benchmark", key) for key in summaries[pure]["benchmarks"]["benchmark"]]]:
        left = raw[pure]["benchmarks"]
        ids = [row["id"] for row in left if group == "overall" or row[group] == name]
        pure_by_id = {row["id"]: row for row in left}
        replay_by_id = {row["id"]: row for row in raw[replay]["benchmarks"]}
        delta, interval = bootstrap_delta(
            [pure_by_id[key]["correct_norm"] for key in ids],
            [replay_by_id[key]["correct_norm"] for key in ids], seed,
        )
        pure_metric = summaries[pure]["benchmarks"][group][name]["accuracy_norm"] if group != "overall" else summaries[pure]["benchmarks"]["overall"]["accuracy_norm"]
        replay_metric = summaries[replay]["benchmarks"][group][name]["accuracy_norm"] if group != "overall" else summaries[replay]["benchmarks"]["overall"]["accuracy_norm"]
        comparisons.append({
            "group": f"benchmark_{group}", "name": name, "samples": len(ids),
            "pure": pure_metric, "replay": replay_metric, "delta_replay_minus_pure": delta, "ci95": interval,
            "winner": "Replay" if interval[0] > 0 else "Pure" if interval[1] < 0 else "Inconclusive",
        })
    result = {"paired_bootstrap_repeats": 2000, "delta_definition": "Replay minus Pure", "comparisons": comparisons}
    atomic_json(output_dir / "comparison.json", result)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["group", "name", "samples", "pure", "replay", "delta_replay_minus_pure", "ci95_low", "ci95_high", "winner"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({**{key: value for key, value in row.items() if key != "ci95"}, "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1]})
    lines = [
        "# Pure vs Replay SFT", "", "Delta 均为 Replay - Pure；NLL 越低越好，accuracy_norm 越高越好。", "",
        "| Metric | Dataset / benchmark | N | Pure | Replay | Delta | 95% CI | Conclusion |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['group']} | {row['name']} | {row['samples']} | {row['pure']:.4f} | {row['replay']:.4f} | "
            f"{row['delta_replay_minus_pure']:+.4f} | [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | {row['winner']} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_judge(rows, output_dir: Path):
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("SILICONFLOW_API_KEY is absent; skip optional judge")
        return None
    judge = SiliconFlowRewardModel(api_key)
    wins = {"Pure": 0, "Replay": 0, "Tie": 0}
    pure_scores, replay_scores = [], []
    for index, row in enumerate(rows):
        order = ("ocean_sft_pure", "ocean_sft_replay") if index % 2 == 0 else ("ocean_sft_replay", "ocean_sft_pure")
        scores = judge.score_group(row["prompt"], row["reference"], [row[name] for name in order])
        mapped = dict(zip(order, scores))
        row["judge_scores"] = mapped
        pure_scores.append(mapped["ocean_sft_pure"])
        replay_scores.append(mapped["ocean_sft_replay"])
        winner = "Pure" if pure_scores[-1] > replay_scores[-1] else "Replay" if replay_scores[-1] > pure_scores[-1] else "Tie"
        wins[winner] += 1
    result = {
        "position_balanced": True, "samples": len(rows), "wins": wins,
        "mean_score": {"Pure": float(np.mean(pure_scores)), "Replay": float(np.mean(replay_scores))},
    }
    atomic_json(output_dir / "judge_summary.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["dense", "moe"], default="dense")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "out")
    parser.add_argument("--ocean-test", type=Path, default=ROOT / "data/processed/ocean_sft_test.jsonl")
    parser.add_argument("--generic-test", type=Path, default=ROOT / "data/processed/generic_sft_eval.jsonl")
    parser.add_argument("--benchmark-root", type=Path, default=Path("/data/anhuang/oceanheart_pretrain_eval"))
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="limit each dataset/benchmark; 0 evaluates all")
    parser.add_argument("--generation-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge", choices=["none", "siliconflow"], default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.results_dir / args.architecture
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    benchmarks, benchmark_manifest, benchmark_manifest_hash = load_benchmarks(args.benchmark_root, args.max_samples)
    candidates = chat_candidates(benchmarks, tokenizer, args.max_seq_len)
    generation_rows = prompts_with_references(args.ocean_test, args.generation_samples)
    datasets = {
        "ocean": SFTDataset(args.ocean_test, tokenizer, max_length=args.max_seq_len, deterministic=True),
        "generic": SFTDataset(args.generic_test, tokenizer, max_length=args.max_seq_len, deterministic=True),
    }
    data_manifest = ROOT / "data/processed/manifest.json"
    tokenizer_files = [ROOT / "model/tokenizer.json", ROOT / "model/tokenizer_config.json"]
    raw, summaries = {}, {}
    for name in MODEL_NAMES:
        checkpoint = checkpoint_path(name, args.architecture, args.save_dir, args.hidden_size)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, parameters = load_model(checkpoint, args.architecture, args.hidden_size, args.layers, device)
        started = time.time()
        model_raw, model_summary = {}, {}
        for dataset_name, dataset in datasets.items():
            rows, metrics = assistant_nll(model, dataset, args.batch_size, device, args.max_samples)
            model_raw[dataset_name], model_summary[dataset_name] = rows, metrics
            write_jsonl(output_dir / name / f"{dataset_name}_nll.jsonl", rows)
        scores = score_candidates(model, candidates, tokenizer.pad_token_id or tokenizer.eos_token_id, args.batch_size, device)
        predictions = benchmark_predictions(benchmarks, scores)
        model_raw["benchmarks"] = predictions
        model_summary["benchmarks"] = aggregate_benchmarks(predictions)
        write_jsonl(output_dir / name / "benchmark_predictions.jsonl", predictions)
        speeds = []
        for row in generation_rows:
            response, speed = generate(model, tokenizer, row["prompt"], args.max_seq_len, args.max_new_tokens, device)
            row[name] = response
            speeds.append(speed)
        model_summary["generation_tokens_per_second"] = float(np.mean(speeds)) if speeds else 0.0
        model_summary["metadata"] = {
            "display_name": DISPLAY_NAMES[name], "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint), "parameters": parameters,
            "elapsed_seconds": time.time() - started,
        }
        summary = {
            **model_summary,
            "architecture": args.architecture,
            "git_commit": git_commit(),
            "data_manifest_sha256": sha256(data_manifest),
            "benchmark_manifest": str(benchmark_manifest),
            "benchmark_manifest_sha256": benchmark_manifest_hash,
            "tokenizer_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in tokenizer_files)).hexdigest(),
            "protocol": {
                "assistant_metric": "teacher_forced_assistant_token_nll",
                "benchmark_metric": "chat_template_candidate_loglikelihood",
                "primary_benchmark_metric": "length_normalized_accuracy",
                "seed": args.seed, "max_seq_len": args.max_seq_len,
                "max_samples_per_dataset": args.max_samples, "generation_samples": args.generation_samples,
            },
        }
        atomic_json(output_dir / name / "summary.json", summary)
        raw[name], summaries[name] = model_raw, model_summary
        print(f"{DISPLAY_NAMES[name]}: ocean PPL={model_summary['ocean']['perplexity']:.3f}, generic PPL={model_summary['generic']['perplexity']:.3f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    judge_result = run_judge(generation_rows, output_dir) if args.judge == "siliconflow" else None
    write_jsonl(output_dir / "generations.jsonl", generation_rows)
    compare(raw, summaries, args.seed, output_dir)
    atomic_json(output_dir / "run_summary.json", {
        "architecture": args.architecture,
        "models": {name: summaries[name] for name in MODEL_NAMES},
        "judge": judge_result,
    })
    print(f"comparison: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
