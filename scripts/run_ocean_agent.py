#!/usr/bin/env python3
"""Gated Dense Agent experiment. Run inside tmux, not a detached polling service."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.train_agent import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hours", type=float, default=10.)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.hours <= 0:
        raise ValueError("invalid tag / budget")
    base = ROOT / "artifacts/agent" / args.tag
    if base.exists():
        raise FileExistsError(f"Do not overwrite an existing experiment: {base}")
    base.mkdir(parents=True)
    deadline = time.time() + args.hours * 3600
    train_deadline = deadline - 1800
    state = {"tag": args.tag, "started_at": time.time(), "deadline": deadline, "stages": [], "phase": "starting"}
    environment = dict(os.environ, OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    current_weight = "ocean_grpo_deepseekv3_eval50"

    def run(stage, mode, weight, extras=()):
        if time.time() >= train_deadline:
            raise RuntimeError("Preflight consumed training budget; refusing a new training stage")
        name = args.tag + "-" + stage
        state["phase"] = stage
        atomic_json(base / "status.json", state)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   str(ROOT / "trainer/train_agent.py"), "--mode", mode, "--from_weight", weight,
                   "--save_weight", "ocean_agent_" + name, "--run_name", name,
                   "--deadline", str(train_deadline), "--use_swanlab", "--swanlab_mode", "cloud", *map(str, extras)]
        print("START", stage, " ".join(command), flush=True)
        started = time.time()
        log_path = base / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state["stages"].append({"stage": stage, "returncode": result.returncode, "seconds": time.time() - started,
                                "run_name": name, "log": str(log_path)})
        atomic_json(base / "status.json", state)
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see {log_path}")
        return ROOT / "artifacts/agent" / name

    def judge(traces, stage):
        # An optional audit cannot prevent core training or become negative reward.
        result = subprocess.run([sys.executable, str(ROOT / "Agent Evaluation/evaluate_judge.py"),
                                 "--traces", str(traces), "--output", str(base / f"judge_{stage}.json"),
                                 "--deadline", str(min(time.time() + 900, deadline))], cwd=ROOT, env=environment)
        if result.returncode:
            atomic_json(base / f"judge_{stage}.json", {"status": "failed", "returncode": result.returncode})

    try:
        baseline = run("baseline", "eval", current_weight)
        judge(baseline / "result.jsonl", "baseline")
        diagnostic = run("diagnose-initial", "diagnose", current_weight)
        gate = json.loads((diagnostic / "result.json").read_text())
        if not gate["gate_pass"]:
            warmup = run("sft", "sft", current_weight, ["--epochs", 3, "--max_steps", 200,
                         "--max_train_seconds", 3600, "--accumulation_steps", 8, "--learning_rate", "1e-5", "--eval_interval", 200])
            current_weight = "ocean_agent_" + args.tag + "-sft"
            diagnostic = run("diagnose-sft", "diagnose", current_weight)
            gate = json.loads((diagnostic / "result.json").read_text())
        state["gate"] = gate
        state["rl_start_weight"] = current_weight
        if not gate["gate_pass"]:
            state["phase"] = "stopped_gate_failed"
            atomic_json(base / "status.json", state)
            print("STOP: tool SFT did not meet the agreed capability gate; no long GRPO launched.", flush=True)
            return
        smoke = run("grpo-smoke", "grpo", current_weight, ["--max_steps", 5, "--skip_eval", "--verify_sync", "--save_interval", 1])
        completion = json.loads((smoke / "completion.json").read_text())
        if completion["global_step"] != 5 or completion["nonzero_updates"] != 5 or not completion["sync_verified"]:
            raise RuntimeError("Four-GPU smoke did not pass all update/gradient/synchronization checks")
        run("grpo-smoke-resume", "grpo", current_weight, ["--max_steps", 6, "--skip_eval", "--verify_sync", "--from_resume",
              "--save_weight", "ocean_agent_" + args.tag + "-grpo-smoke"])
        seconds = completion["elapsed_seconds"] / 5
        state["smoke_seconds_per_update"] = seconds
        state["estimated_remaining_updates_upper_bound"] = int(max(0, train_deadline - time.time()) / max(seconds, 1))
        atomic_json(base / "status.json", state)
        trained = run("grpo", "grpo", current_weight, ["--max_train_seconds", max(1, train_deadline - time.time())])
        judge(trained / "val_final.jsonl", "final")
        state["phase"] = "completed"
        state["completion"] = json.loads((trained / "completion.json").read_text())
        state["elapsed_seconds"] = time.time() - state["started_at"]
        atomic_json(base / "status.json", state)
        print("COMPLETE", json.dumps(state), flush=True)
    except Exception as exc:
        state["phase"] = "failed"
        state["error"] = str(exc)
        atomic_json(base / "status.json", state)
        raise


if __name__ == "__main__":
    main()
