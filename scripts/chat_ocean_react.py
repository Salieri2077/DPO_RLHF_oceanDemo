#!/usr/bin/env python3
"""Local OceanHeart ReAct CLI. No network, shell tools, or hidden answer repair."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from agent.ocean import file_hash, read_jsonl
from agent.react import ReactTools, run, VERSION
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_agent import atomic_json, model_generator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=ROOT / "data/processed/react-v2/ocean_agent_corpus_train.jsonl")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--question")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--max_calls", type=int, default=4)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=192)
    args = p.parse_args()
    if not args.resume and not args.question and not args.interactive:
        p.error("provide --question or --interactive")
    if args.resume and args.question:
        p.error("resume completes the stored question; use --interactive for follow-up")
    if args.session.exists() != args.resume:
        p.error("existing sessions require --resume; a new session must not already exist")
    budget = {k: getattr(args, k) for k in ("max_turns", "max_calls", "max_total_len", "max_new_tokens")}
    metadata = {"version": VERSION, "weight_sha256": file_hash(args.weight), "corpus_sha256": file_hash(args.corpus),
                "harness_sha256": file_hash(ROOT / "agent/react.py"), "tool_sha256": file_hash(ROOT / "agent/ocean.py"),
                "tokenizer_sha256": file_hash(ROOT / "model/tokenizer.json"),
                "template_sha256": file_hash(ROOT / "model/tokenizer_config.json"), "budget": budget}
    stored = json.loads(args.session.read_text()) if args.resume else None
    if stored and stored["metadata"] != metadata:
        raise ValueError("Session model/tool/corpus/template/budget changed; use a new session")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    model.load_state_dict(torch.load(args.weight, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    env = ReactTools(read_jsonl(args.corpus))
    generate = model_generator(model, tokenizer, device, sample=False, save_logps=False)
    trace = stored["trace"] if stored else None
    archive = stored.get("previous_episodes", []) if stored else []
    question = stored["question"] if stored else args.question
    def save(state):
        atomic_json(args.session, {"metadata": metadata, "question": question, "trace": state, "previous_episodes": archive})
    while True:
        if not question:
            try:
                question = input("你：").strip()
            except EOFError:
                break
            if question in ("/exit", "/quit"):
                break
            if not question:
                continue
        trace = run(question, tokenizer, env, generate, state=trace, save=save, **budget)
        for r in trace["rounds"]:
            print("助手：", r["text"], flush=True)
        print(f"[{trace['stop']}] {trace['final']}\n轨迹：{args.session}", flush=True)
        if not args.interactive or trace["stop"] not in {"final", "clarify"}:
            break
        try:
            question = input("你（/exit退出）：").strip()
        except EOFError:
            break
        if not question or question in ("/exit", "/quit"):
            break
        archive.append(trace)
        trace = {"messages": trace["messages"] + [{"role": "user", "content": question}], "rounds": [], "calls": [],
                 "final": "", "stop": "running", "unrecovered_error": False, "duplicates": {}, "transient_used": False}
        save(trace)


if __name__ == "__main__":
    main()
