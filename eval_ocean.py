#!/usr/bin/env python3
"""Offline checkpoint comparison for OceanHeart."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset, SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import SiliconFlowRewardModel, setup_seed


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "你是 OceanHeart，一个面向海洋科学任务的双语语言模型助手。"
    "请基于可靠知识准确回答；不确定时明确说明。"
)


def limited_loader(dataset, batch_size, limit):
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def sft_nll(model, loader, device, autocast):
    loss_sum = tokens = 0.0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        with autocast:
            output = model(input_ids, labels=labels)
        count = labels[..., 1:].ne(-100).sum().item()
        loss_sum += output.loss.item() * count
        tokens += count
    loss = loss_sum / max(tokens, 1)
    return {"nll": loss, "perplexity": math.exp(min(loss, 20)), "tokens": int(tokens)}


@torch.no_grad()
def preference_metrics(model, loader, device, autocast):
    correct = pairs = 0
    margin_sum = 0.0
    for batch in loader:
        chosen_x, chosen_y, chosen_mask = (batch[key].to(device) for key in ("x_chosen", "y_chosen", "mask_chosen"))
        rejected_x, rejected_y, rejected_mask = (batch[key].to(device) for key in ("x_rejected", "y_rejected", "mask_rejected"))
        x, y, mask = torch.cat([chosen_x, rejected_x]), torch.cat([chosen_y, rejected_y]), torch.cat([chosen_mask, rejected_mask])
        with autocast:
            logits = model(x).logits
            token_log_probs = torch.gather(F.log_softmax(logits, dim=-1), 2, y.unsqueeze(2)).squeeze(2)
            scores = (token_log_probs * mask).sum(dim=1)
        half = scores.size(0) // 2
        margins = scores[:half] - scores[half:]
        correct += (margins > 0).sum().item()
        margin_sum += margins.sum().item()
        pairs += half
    return {"preference_accuracy": correct / max(pairs, 1), "preference_margin": margin_sum / max(pairs, 1), "pairs": pairs}


def load_model(weight, args):
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(config)
    checkpoint = args.save_dir / f"{weight}_{args.hidden_size}.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.eval().to(args.device)
    return model.half() if args.device.startswith("cuda") else model


@torch.inference_mode()
def generate(model, tokenizer, prompt, weight, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.bos_token + prompt if weight == "pretrain" else tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
    start = time.time()
    output = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    elapsed = max(time.time() - start, 1e-6)
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens) / elapsed


def prompts_from_jsonl(path, limit):
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            prompt = next(message["content"] for message in item["conversations"] if message["role"] == "user")
            prompts.append(prompt)
            if len(prompts) >= limit:
                break
    return prompts


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", default=["pretrain", "ocean_sft_pure", "ocean_sft_replay", "ocean_dpo"])
    parser.add_argument("--save-dir", type=Path, default=ROOT / "out")
    parser.add_argument("--ocean-test", type=Path, default=ROOT / "data/processed/ocean_sft_test.jsonl")
    parser.add_argument("--generic-test", type=Path, default=ROOT / "data/processed/generic_sft_eval.jsonl")
    parser.add_argument("--dpo-test", type=Path, default=ROOT / "data/processed/ocean_dpo_test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/eval")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--generation-samples", type=int, default=100)
    parser.add_argument("--judge", choices=["none", "siliconflow"], default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    ocean_ds = SFTDataset(args.ocean_test, tokenizer, max_length=args.max_seq_len, deterministic=True)
    generic_ds = SFTDataset(args.generic_test, tokenizer, max_length=args.max_seq_len, deterministic=True)
    dpo_ds = DPODataset(args.dpo_test, tokenizer, max_length=args.max_seq_len, deterministic=True)
    ocean_loader = limited_loader(ocean_ds, args.batch_size, args.max_eval_samples)
    generic_loader = limited_loader(generic_ds, args.batch_size, args.max_eval_samples)
    dpo_loader = limited_loader(dpo_ds, args.batch_size, args.max_eval_samples)
    autocast = torch.amp.autocast("cuda", dtype=torch.float16) if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    prompts = prompts_from_jsonl(args.ocean_test, args.generation_samples)

    summary, rows = {}, [{"prompt": prompt} for prompt in prompts]
    for weight in args.weights:
        checkpoint = args.save_dir / f"{weight}_{args.hidden_size}.pth"
        if not checkpoint.exists():
            print(f"skip missing checkpoint: {checkpoint}")
            continue
        model = load_model(weight, args)
        setup_seed(42)
        summary[weight] = {
            "ocean": sft_nll(model, ocean_loader, args.device, autocast),
            "generic": sft_nll(model, generic_loader, args.device, autocast),
            "dpo": preference_metrics(model, dpo_loader, args.device, autocast),
        }
        speeds = []
        for row in rows:
            response, speed = generate(model, tokenizer, row["prompt"], weight, args)
            row[weight] = response
            speeds.append(speed)
        summary[weight]["generation_tokens_per_second"] = sum(speeds) / max(len(speeds), 1)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.judge == "siliconflow":
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            print("SILICONFLOW_API_KEY is absent; skip optional judge")
        elif "ocean_sft_replay" in summary and "ocean_dpo" in summary:
            judge = SiliconFlowRewardModel(api_key)
            wins = {"ocean_sft_replay": 0, "ocean_dpo": 0, "tie": 0}
            for row in rows:
                messages = [{"role": "user", "content": row["prompt"]}]
                left = judge.get_score(messages, row["ocean_sft_replay"])
                right = judge.get_score(messages, row["ocean_dpo"])
                wins["ocean_sft_replay" if left > right else "ocean_dpo" if right > left else "tie"] += 1
            summary["siliconflow_judge"] = wins

    atomic_json(args.output_dir / "summary.json", summary)
    jsonl = args.output_dir / "generations.jsonl"
    temp = jsonl.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, jsonl)
    columns = ["prompt", *[weight for weight in args.weights if weight in summary]]
    csv_path = args.output_dir / "generations.csv"
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, csv_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
