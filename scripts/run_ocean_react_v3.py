#!/usr/bin/env python3
"""ReAct v3 data experiments: tool SFT with the start weight, harness, scorer and settings of the react-v2
SFT; only the training data changes. Then paired evaluation on the same questions (never RL)."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash, read_jsonl
from scripts.run_ocean_react import BASE_WEIGHT, paired
from trainer.train_agent import atomic_json

ARTIFACTS = ROOT / "artifacts/agent"
OLD_RUN = "react-v2-20260927-1046"  # react-v2-data SFT, already evaluated with this harness
OLD_WEIGHT = f"ocean_{OLD_RUN}-sft_best"
VARIANTS = {"fixed": ROOT / "data/processed/react-v3", "slots": ROOT / "data/processed/react-v3-slots",
            "contrast": ROOT / "data/processed/react-v3.1"}
EVALS = {  # name: (data, split, OLD_RUN stage that answered the same questions)
    "v2-val": (ROOT / "data/processed/react-v2", "val", "c-val"),
    "v2-test": (ROOT / "data/processed/react-v2", "test", "c-test"),
    "v1-regression": (ROOT / "data/processed", "val", "c-regression"),
    "v3-test": (VARIANTS["fixed"], "test", None),
    "v3.1-test": (VARIANTS["contrast"], "test", None),
}
# react-v2 SFT settings; its 300-step cap never bound (282 updates = 3 epochs), so v3 also trains 3 epochs.
SFT = ["--epochs", 3, "--max_train_seconds", 5400, "--accumulation_steps", 8, "--learning_rate", "1e-5",
       "--eval_interval", 100, "--save_interval", 50]
SMOKE = ["--epochs", 1, "--max_steps", 2, "--accumulation_steps", 8, "--learning_rate", "1e-5",
         "--skip_eval", "--verify_sync", "--save_interval", 1]


def results(directory):
    return read_jsonl(directory / "result.jsonl") if directory and (directory / "result.jsonl").exists() else None


def report(base, records, variants, previous):
    summary = {"stages": {}, "paired_vs_react_v2_sft": {}, "paired_vs_previous": {}, "paired_slots_vs_fixed": {},
               "note": f"Same start weight, harness, scorer and SFT settings as {OLD_RUN}; only data differs. No RL."}
    for stage, directory in records.items():
        for name in ("result.json", "completion.json"):
            if (directory / name).exists():
                summary["stages"].setdefault(stage, {}).update(json.loads((directory / name).read_text()))
    for variant in variants:
        for name, (_, _, old_stage) in EVALS.items():
            new = results(records.get(f"{variant}-{name}"))
            old = results(ARTIFACTS / f"{OLD_RUN}-{old_stage}" if old_stage else records.get("old-" + name))
            if new and old:
                summary["paired_vs_react_v2_sft"][f"{variant}/{name}"] = paired(old, new)
            if previous:
                prior = results(ARTIFACTS / f"{previous}-{name}") or results(records.get("prev-" + name))
                if new and prior:
                    summary["paired_vs_previous"][f"{variant}/{name}"] = paired(prior, new)
    for name in EVALS:
        fixed, slots = results(records.get(f"fixed-{name}")), results(records.get(f"slots-{name}"))
        if fixed and slots:
            summary["paired_slots_vs_fixed"][name] = paired(fixed, slots)
    atomic_json(base / "comparison.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["fixed", "slots"])
    p.add_argument("--compare_to", help="earlier variant run to pair against, e.g. react-v3-20261006-1314-fixed")
    args = p.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("tag must be a simple name")
    import agent.sdk  # Use the SDK venv; fail before creating artifacts or starting GPUs.
    base = ARTIFACTS / args.tag
    base.mkdir(parents=True, exist_ok=False)
    state = {"phase": "starting", "stages": [], "started_at": time.time(), "rl_enabled": False, "runtime": "sdk",
             "base_weight": BASE_WEIGHT, "baseline_run": OLD_RUN, "compare_to": args.compare_to,
             "data_manifest_sha256": {v: file_hash(VARIANTS[v] / "agent_manifest.json") for v in args.variants}}
    records = {}
    environment = dict(os.environ, OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    evals = {name: spec for name, spec in EVALS.items() if (spec[0] / "agent_manifest.json").exists()}

    def run(stage, mode, weight, data, extra=()):
        name = args.tag + "-" + stage
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   str(ROOT / "trainer/train_agent.py"), "--agent_version", "v2", "--runtime", "sdk", "--mode", mode,
                   "--from_weight", weight, "--save_weight", "ocean_" + name, "--run_name", name,
                   "--data_dir", str(data), "--use_swanlab", "--swanlab_mode", "cloud", *map(str, extra)]
        state.update(phase=stage, active_command=command)
        atomic_json(base / "status.json", state)
        started, path = time.time(), base / (stage + ".log")
        print("START", stage, flush=True)
        with path.open("w") as f:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=f, stderr=subprocess.STDOUT)
        urls = re.findall(r"https://swanlab.cn/[^\s\x1b]+/runs/[A-Za-z0-9]+", path.read_text())
        state["stages"].append({"stage": stage, "returncode": result.returncode, "seconds": time.time() - started,
                                "log": str(path), "swanlab": urls[-1] if urls else None})
        atomic_json(base / "status.json", state)
        if result.returncode:
            raise RuntimeError(f"Stage {stage} failed: {path}")
        records[stage] = ARTIFACTS / name
        return records[stage]

    try:
        for variant in args.variants:
            data = VARIANTS[variant]
            smoke = json.loads((run(variant + "-smoke", "sft", BASE_WEIGHT, data, SMOKE) / "completion.json").read_text())
            if smoke["global_step"] != 2 or smoke["nonzero_updates"] != 2 or not smoke["sync_verified"]:
                raise RuntimeError(f"Four-card smoke failed update/gradient/synchronization checks: {smoke}")
            for name, (eval_data, split, old_stage) in evals.items():  # references for questions they never saw
                if old_stage is None and "old-" + name not in records:
                    run("old-" + name, "eval", OLD_WEIGHT, eval_data, ["--split", split])
                if args.compare_to and not results(ARTIFACTS / f"{args.compare_to}-{name}") and "prev-" + name not in records:
                    previous_run, previous_variant = args.compare_to.rsplit("-", 1)
                    run("prev-" + name, "eval", f"ocean_{previous_run}-{previous_variant}-sft_best", eval_data, ["--split", split])
            # Formal SFT starts from BASE_WEIGHT, NOT smoke updates; best = highest success on the variant's val.
            trained = run(variant + "-sft", "sft", BASE_WEIGHT, data, SFT)
            state.setdefault("sft_completion", {})[variant] = json.loads((trained / "completion.json").read_text())
            for name, (eval_data, split, _) in evals.items():
                run(f"{variant}-{name}", "eval", f"ocean_{args.tag}-{variant}-sft_best", eval_data, ["--split", split])
            report(base, records, args.variants, args.compare_to)
        state.update(phase="completed", elapsed_seconds=time.time() - state["started_at"])
        atomic_json(base / "status.json", state)
        print("COMPLETE", str(base / "comparison.json"), flush=True)
    except Exception as exc:
        state.update(phase="failed", error=str(exc))
        atomic_json(base / "status.json", state)
        raise


if __name__ == "__main__":
    main()
