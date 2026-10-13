#!/usr/bin/env python3
"""Rescore saved evaluation runs with the current scorer, keeping the earlier files as *.scorer-v1.*.

Used after the delivery rule accepted several handles per analysis (an analysis computed per period from separate
queries). The environment self-check result of each trace is reused: rescoring changes only how the delivery is
judged, never the tools' results.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import read_jsonl
from agent.oceanlab.data import OceanDepths
from agent.oceanlab.scoring import Scorer
from scripts.run_oceanlab_eval import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--tag", default="scorer-v1", help="suffix of the kept earlier files")
    args = p.parse_args()
    scorer = Scorer(OceanDepths())
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        requests = read_jsonl(Path(config["requests"]))
        specs = {r["id"]: r for r in requests}
        episodes, summary = run / "episodes.jsonl", run / "summary.json"
        backup = run / f"episodes.{args.tag}.jsonl"
        if backup.exists():
            raise FileExistsError(f"{backup} exists; this run was already rescored")
        traces = read_jsonl(episodes)
        changed = 0
        for t in traces:
            old = t["metrics"]
            t["metrics"] = scorer.score(specs[t["id"]], t, env_problems=old["env_problems"] if old["env_error"] else [])
            changed += t["metrics"]["success"] != old["success"]
        episodes.rename(backup)
        summary.rename(run / f"summary.{args.tag}.json")
        with episodes.open("w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        new = {"config": config, "episodes": len(traces), "rescored": {"from": args.tag, "success_changed": changed},
               **summarize(traces, requests)}
        summary.write_text(json.dumps(new, ensure_ascii=False, indent=1))
        g = new.get("greedy", {})
        print(run.name, "episodes", len(traces), "success changed", changed, "greedy", round(g.get("success", 0), 3),
              "sampled", round(new["sampled"]["success"], 3) if new.get("sampled") else "-", flush=True)


if __name__ == "__main__":
    main()
