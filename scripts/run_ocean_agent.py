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
from agent.ocean import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hours", type=float, default=10.)
    parser.add_argument("--reuse_preflight", help="Reuse unchanged baseline/initial diagnosis after an implementation fix; preserves the original deadline")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.hours <= 0:
        raise ValueError("invalid tag / budget")
    base = ROOT / "artifacts/agent" / args.tag
    if base.exists():
        raise FileExistsError(f"Do not overwrite an existing experiment: {base}")
    base.mkdir(parents=True)
    deadline = time.time() + args.hours * 3600
    prior = None
    if args.reuse_preflight:
        if Path(args.reuse_preflight).name != args.reuse_preflight:
            raise ValueError("invalid preflight tag")
        prior = json.loads((ROOT / "artifacts/agent" / args.reuse_preflight / "status.json").read_text())
        if prior["phase"] not in {"stopped_gate_failed", "failed"}:
            raise ValueError("Can only reuse preflight from a stopped experiment")
        deadline = min(deadline, prior["deadline"])
    train_deadline = deadline - 1800
    state = {"tag": args.tag, "started_at": time.time(), "deadline": deadline, "stages": [], "phase": "starting"}
    if prior:
        state.update(started_at=prior["started_at"], reused_preflight=args.reuse_preflight)
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
        if prior:
            baseline = ROOT / "artifacts/agent" / (args.reuse_preflight + "-baseline")
            diagnostic = ROOT / "artifacts/agent" / (args.reuse_preflight + "-diagnose-initial")
            for directory in (baseline, diagnostic):
                config = json.loads((directory / "config.json").read_text())
                for key, path in (("input_weight_sha256", ROOT / "out" / (current_weight + "_768.pth")),
                                  ("tool_sha256", ROOT / "agent/ocean.py"), ("data_manifest_sha256", ROOT / "data/processed/agent_manifest.json")):
                    if config[key] != file_hash(path):
                        raise ValueError(f"Cannot reuse changed preflight: {key}")
            state["baseline_traces"] = str(baseline / "result.jsonl")
            audit = ROOT / "artifacts/agent" / args.reuse_preflight / "judge_baseline_actual.json"
            if audit.exists() and json.loads(audit.read_text()).get("answer_selection") == "final_or_last_partial":
                state["baseline_judge"] = str(audit)
            else:
                judge(baseline / "result.jsonl", "baseline")
                state["baseline_judge"] = str(base / "judge_baseline.json")
        else:
            baseline = run("baseline", "eval", current_weight)
            judge(baseline / "result.jsonl", "baseline")
            diagnostic = run("diagnose-initial", "diagnose", current_weight)
        gate = json.loads((diagnostic / "result.json").read_text())
        if not gate["gate_pass"]:
            sft_budget = 3600 - sum(s["seconds"] for s in (prior or {}).get("stages", []) if s["stage"] == "sft")
            if sft_budget <= 0:
                raise RuntimeError("Tool SFT budget already exhausted")
            warmup = run("sft", "sft", current_weight, ["--epochs", 3, "--max_steps", 200,
                         "--max_train_seconds", sft_budget, "--accumulation_steps", 8, "--learning_rate", "1e-5", "--eval_interval", 200])
            current_weight = "ocean_agent_" + args.tag + "-sft"
            diagnostic = run("diagnose-sft", "diagnose", current_weight)
            gate = json.loads((diagnostic / "result.json").read_text())
        state["gate"] = gate
        state["rl_start_weight"] = current_weight
        if not gate["gate_pass"]:
            state["phase"] = "stopped_gate_failed"
            atomic_json(base / "status.json", state)
            print("STOP: tool SFT did not meet the agreed capability gate; no long GRPO launched.", flush=True)
            judge(warmup / "val_final.jsonl", "final_sft_gate_failed")
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
