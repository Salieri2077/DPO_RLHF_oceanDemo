#!/usr/bin/env python3
"""Run tool SFT with the SDK runtime and paired evaluation (never RL)."""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import read_jsonl, file_hash
from trainer.train_agent import atomic_json

BASE_WEIGHT = "ocean_agent_dense-agent-20260927-0232-aligned-sft"


def paired(before, after):
    a, b = {t["id"]: t for t in before}, {t["id"]: t for t in after}
    if a.keys() != b.keys() or len(a) != len(before) or len(b) != len(after):
        raise ValueError("paired evaluation requires the same unique question ids")
    diffs = [b[k]["metrics"]["success"] - a[k]["metrics"]["success"] for k in sorted(a)]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000))
    return {"questions": len(diffs), "delta_success": sum(diffs) / len(diffs),
            "paired_bootstrap_95ci": [boot[50], boot[1949]], "improved": sum(d > 0 for d in diffs),
            "regressed": sum(d < 0 for d in diffs), "unchanged": sum(d == 0 for d in diffs)}


def report(base, records):
    summary = {"stages": {}, "paired": {}, "note": "Tool-SFT and harness effects are separate; no Agentic RL in this experiment."}
    rows = []
    for stage, directory in records.items():
        path = directory / "result.json"
        if not path.exists():
            continue
        summary["stages"][stage] = json.loads(path.read_text())
        for trace in read_jsonl(directory / "result.jsonl"):
            m = trace["metrics"]
            failure = "success" if m["success"] else trace["stop"] if trace["stop"] not in {"final", "clarify"} else (
                "parameters" if m.get("parameter_cases") and not m.get("parameter_correct") else
                "citation" if m.get("citation_cases") and not m.get("citation_correct") else
                "evidence_copy" if m.get("faithful_cases") and not m.get("faithful") else "answer_or_required_actions")
            rows.append({"stage": stage, "id": trace["id"], "kind": trace["kind"], "success": m["success"],
                         "calls": m["attempted_calls"], "failure": failure, "final": trace["final"]})
    for suffix in ("regression", "val", "test", "test-seed42", "test-seed43", "test-seed44"):
        if "b-" + suffix in records and "c-" + suffix in records:
            summary["paired"][suffix] = paired(read_jsonl(records["b-" + suffix] / "result.jsonl"),
                                                read_jsonl(records["c-" + suffix] / "result.jsonl"))
    if "a-regression" in records and "b-regression" in records:
        summary["paired"]["harness_only_regression"] = paired(read_jsonl(records["a-regression"] / "result.jsonl"),
                                                                read_jsonl(records["b-regression"] / "result.jsonl"))
    if all(f"{arm}-test-seed{seed}" in records for arm in ("b", "c") for seed in (42, 43, 44)):
        sampled = {}
        for arm in ("b", "c"):
            totals = {}
            for seed in (42, 43, 44):
                for t in read_jsonl(records[f"{arm}-test-seed{seed}"] / "result.jsonl"):
                    totals[t["id"]] = totals.get(t["id"], 0) + t["metrics"]["success"] / 3
            sampled[arm] = [{"id": k, "metrics": {"success": v}} for k, v in totals.items()]
            summary["stages"][arm + "-sampled-mean"] = {"success": sum(totals.values()) / len(totals), "questions": len(totals), "seeds": [42, 43, 44]}
        summary["paired"]["sampled_question_cluster"] = paired(sampled["b"], sampled["c"])
    atomic_json(base / "comparison.json", summary)
    with (base / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "id", "kind", "success", "calls", "failure", "final"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data/processed/react-v2")
    args = p.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("tag must be a simple name")
    import agent.sdk  # Use the SDK venv; fail before creating artifacts or starting GPUs.
    base = ROOT / "artifacts/agent" / args.tag
    base.mkdir(parents=True, exist_ok=False)
    state = {"phase": "starting", "stages": [], "started_at": time.time(), "rl_enabled": False,
             "base_weight": BASE_WEIGHT, "runtime": "sdk",
             "data_manifest_sha256": file_hash(args.data_dir / "agent_manifest.json")}
    records = {}
    environment = dict(os.environ, OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")

    def run(stage, mode="eval", weight=BASE_WEIGHT, data=None, version="v2", extra=()):
        name = args.tag + "-" + stage
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   str(ROOT / "trainer/train_agent.py"), "--agent_version", version, "--mode", mode,
                   "--runtime", "sdk" if version == "v2" else "local",
                   "--from_weight", weight, "--save_weight", "ocean_" + name, "--run_name", name,
                   "--data_dir", str(data or args.data_dir), "--use_swanlab", "--swanlab_mode", "cloud", *map(str, extra)]
        state["phase"] = stage
        state["active_command"] = command
        atomic_json(base / "status.json", state)
        started = time.time()
        path = base / (stage + ".log")
        print("START", stage, flush=True)
        with path.open("w") as f:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=f, stderr=subprocess.STDOUT)
        urls = re.findall(r"https://swanlab.cn/[^\s\x1b]+/runs/[A-Za-z0-9]+", path.read_text())
        state["stages"].append({"stage": stage, "returncode": result.returncode, "seconds": time.time() - started,
                                "log": str(path), "swanlab": urls[-1] if urls else None})
        atomic_json(base / "status.json", state)
        if result.returncode:
            raise RuntimeError(f"Stage {stage} failed: {path}")
        records[stage] = ROOT / "artifacts/agent" / name
        return records[stage]

    try:
        smoke = run("smoke", "sft", extra=["--epochs", 3, "--max_steps", 5, "--accumulation_steps", 8,
                    "--learning_rate", "1e-5", "--skip_eval", "--verify_sync", "--save_interval", 1])
        completed = json.loads((smoke / "completion.json").read_text())
        if completed["global_step"] != 5 or completed["nonzero_updates"] != 5 or not completed["sync_verified"]:
            raise RuntimeError("Four-card smoke failed update/gradient/synchronization checks")
        run("smoke-resume", "sft", extra=["--epochs", 3, "--max_steps", 6, "--accumulation_steps", 8,
             "--learning_rate", "1e-5", "--skip_eval", "--verify_sync", "--from_resume", "--save_interval", 1,
             "--save_weight", "ocean_" + args.tag + "-smoke"])
        resumed = json.loads((records["smoke-resume"] / "completion.json").read_text())
        if resumed["global_step"] != 6 or resumed["nonzero_updates"] != 6:
            raise RuntimeError("Smoke resume step did not advance")
        if state["stages"][0]["swanlab"] != state["stages"][1]["swanlab"]:
            raise RuntimeError("Cloud resume did not reuse the smoke run")
        # Verify actual cloud metrics before allowing the formal training stage.
        from swanlab.api import OpenApi
        url = next(s["swanlab"] for s in state["stages"] if s["stage"] == "smoke")
        if not url:
            raise RuntimeError("Smoke has no SwanLab URL")
        api = OpenApi(log_level="error")
        cloud = None
        for _ in range(6):
            cloud = api.experiment.get_metrics(exp_id=url.rsplit("/", 1)[-1], keys=["train/grad_norm"])
            if cloud is not None and cloud.data is not None and not cloud.data.empty:
                break
            time.sleep(5)
        if cloud is None or cloud.data is None or cloud.data.empty:
            raise RuntimeError("SwanLab smoke metrics not readable; no formal training launched")
        state["cloud_smoke_verified"] = True
        state["smoke_seconds_per_update"] = completed["elapsed_seconds"] / 5
        old_data = ROOT / "data/processed"
        run("a-regression", data=old_data, version="v1")
        run("b-regression", data=old_data)
        run("b-val")
        report(base, records)
        # Formal SFT starts from BASE_WEIGHT, NOT smoke updates.
        trained = run("sft", "sft", extra=["--epochs", 3, "--max_steps", 300, "--max_train_seconds", 3600,
                      "--accumulation_steps", 8, "--learning_rate", "1e-5", "--eval_interval", 50, "--save_interval", 25])
        best = "ocean_" + args.tag + "-sft_best"
        state["selected_weight"] = best
        state["sft_completion"] = json.loads((trained / "completion.json").read_text())
        run("c-regression", weight=best, data=old_data)
        run("c-val", weight=best)
        state["test_unlocked_at"] = time.time()
        atomic_json(base / "status.json", state)
        for arm, weight in (("b", BASE_WEIGHT), ("c", best)):
            run(arm + "-test", weight=weight, extra=["--split", "test"])
            for seed in (42, 43, 44):
                run(f"{arm}-test-seed{seed}", weight=weight, extra=["--split", "test", "--sample_eval", "--eval_seed", seed])
        report(base, records)
        state.update(phase="completed", elapsed_seconds=time.time() - state["started_at"])
        atomic_json(base / "status.json", state)
        print("COMPLETE", str(base / "comparison.json"), flush=True)
    except Exception as exc:
        state.update(phase="failed", error=str(exc))
        atomic_json(base / "status.json", state)
        raise


if __name__ == "__main__":
    main()
