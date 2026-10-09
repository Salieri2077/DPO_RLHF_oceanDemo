#!/usr/bin/env python3
"""Merge a Qwen LoRA adapter into its base model (fp32 merge, fp16 save) for use as a GRPO start/reference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from trainer.hf_chat import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/data/anhuang/oceanheart_models/qwen3-1.7b-base"))
    parser.add_argument("--adapter", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_sft_r64_eot_best")
    parser.add_argument("--output", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_sft_r64_eot_merged")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    largest = max(p.abs().max().item() for p in merged.parameters())
    if largest > 6e4:
        raise ValueError(f"weight magnitude {largest} overflows fp16")
    merged.half().save_pretrained(args.output, safe_serialization=True)
    load_tokenizer(args.base).save_pretrained(args.output)
    (args.output / "MERGE.json").write_text(json.dumps({"base": str(args.base), "adapter": str(args.adapter),
                                                        "max_abs_weight": largest}, indent=2) + "\n")
    print(f"merged -> {args.output} (max |w| = {largest:.3f})")


if __name__ == "__main__":
    main()
