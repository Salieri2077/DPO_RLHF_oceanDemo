#!/usr/bin/env python3
"""M1b environment check on the 48 dev requests (no model involved).

1. Oracle episodes: the scripted correct trajectory of every request runs through the SDK harness with the Qwen agent
   tokenizer (the C0 starting point's prompt format) and is scored; every episode must succeed with no environment
   error. Records prompt and episode token counts as a lower bound for the M2 budgets.
2. Diagnostic modes: every request is also computed with finite_only and/or temporal_filter=any, and each of those
   handles is rechecked against the independent reference under its own parameters (the dual-implementation check
   of M1a covered the formal rules only). Also counts how often a diagnostic changes the result.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.oceanlab.env import OceanLab, OceanLabHarness
from agent.oceanlab.oracle import oracle_texts
from agent.oceanlab.scoring import Scorer, aggregate
from agent.oceanlab.tasks import group_dates
from agent.ocean import file_hash, read_jsonl
from agent.sdk import run_trajectory
from trainer.train_agent import atomic_json
from trainer.train_agent_hf import agent_tokenizer

REQUESTS = ROOT / "data/processed/oceanlab/oceanlab_dev_v1.jsonl"
DIAGNOSTICS = {"compute_mld": [("finite_only", None)],
               "compute_consistency": [("qc_policy_v1", "any"), ("finite_only", "nearest_or_exact"), ("finite_only", "any")]}


def diagnostic_handles(spec, lab, scorer):
    """Diagnostic computations for one request; returns (handles checked, problems, changed-vs-formal flags)."""
    episode = lab.episode()
    (lat0, lat1), (lon0, lon1) = spec["region"]
    groups = [{"label": g["label"], "start_date": s, "end_date": e} for g in spec["groups"] for s, e in [group_dates(g)]]
    ds = episode.execute({"name": "query_profiles", "arguments": {
        "lat_min": lat0, "lat_max": lat1, "lon_min": lon0, "lon_max": lon1,
        "start_date": min(g["start_date"] for g in groups), "end_date": max(g["end_date"] for g in groups)}})["dataset"]
    changed = []
    if "mld" in spec["analyses"]:
        base = {"dataset": ds, "variable": "potential_temperature", "groups": groups}
        formal = episode.execute({"name": "compute_mld", "arguments": {**base, "qc_policy": "qc_policy_v1"}})
        diag = episode.execute({"name": "compute_mld", "arguments": {**base, "qc_policy": "finite_only"}})
        changed.append(("mld/finite_only", [g["counts"] for g in formal["groups"]] != [g["counts"] for g in diag["groups"]]))
    if "consistency" in spec["analyses"]:
        c = spec["consistency"]
        base = {"dataset": ds, "variable": c["variable"], "depth_min": c["depth_range_m"][0],
                "depth_max": c["depth_range_m"][1], "groups": groups}
        formal = episode.execute({"name": "compute_consistency", "arguments": {
            **base, "qc_policy": "qc_policy_v1", "temporal_filter": "nearest_or_exact"}})
        for qc, temporal in DIAGNOSTICS["compute_consistency"]:
            episode.execute({"name": "compute_consistency", "arguments": {**base, "qc_policy": qc, "temporal_filter": temporal}})
        handles = list(episode.results)
        for handle, (qc, temporal) in zip(handles[-3:], DIAGNOSTICS["compute_consistency"]):
            a = [r["n"] for g in episode.results[handles[-4]]["groups"] for r in g["depths"]]
            b = [r["n"] for g in episode.results[handle]["groups"] for r in g["depths"]]
            changed.append((f"consistency/{qc}/{temporal}", a != b))
    problems = scorer.self_check({"calls": [], "registry": episode.public_registry()})
    diagnostic = sum(1 for r in episode.results.values() if not r["method_card"])
    return diagnostic, len(episode.results), problems, changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, default=ROOT / "out/hf/qwen3_1.7b_ocean_grpo400_merged")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/oceanlab/env_check_dev_v1.json")
    args = p.parse_args()
    tokenizer = agent_tokenizer(args.tokenizer)
    lab = OceanLab("dev")
    scorer = Scorer(lab.data)
    harness = OceanLabHarness(scorer)
    requests = read_jsonl(REQUESTS)
    started = time.monotonic()
    traces, rows = [], []
    diag_total, handle_total, diag_problems, changed = 0, 0, [], Counter()
    for spec in requests:
        texts = iter(oracle_texts(spec, lab))

        def generate(ids, limit):
            return tokenizer.encode(next(texts), add_special_tokens=False) + [tokenizer.eos_token_id], []
        trace = run_trajectory(spec, tokenizer, lab, generate, harness=harness)
        traces.append(trace)
        observation_tokens = [len(tokenizer.encode(m["content"], add_special_tokens=False))
                              for m in trace["messages"] if m["role"] == "tool"]
        rows.append({"id": spec["id"], "family": spec["family"], "status": spec["reference"]["status"],
                     "success": trace["metrics"]["success"], "env_error": trace["metrics"]["env_error"],
                     "first_prompt_tokens": len(trace["rounds"][0]["input_ids"]),
                     "last_prompt_tokens": len(trace["rounds"][-1]["input_ids"]),
                     "episode_input_tokens": trace["metrics"]["input_tokens"],
                     "generated_tokens": trace["metrics"]["generated_tokens"],
                     "max_round_tokens": max(len(r["completion_ids"]) for r in trace["rounds"]),
                     "max_observation_tokens": max(observation_tokens), "calls": trace["metrics"]["calls"]})
        n_diag, n_handles, problems, flags = diagnostic_handles(spec, lab, scorer)
        diag_total += n_diag
        handle_total += n_handles
        diag_problems += [f"{spec['id']}: {x}" for x in problems]
        for name, flag in flags:
            changed[name + (" changed" if flag else " same")] += 1
    summary = aggregate(traces)
    key = lambda k: [r[k] for r in rows]
    report = {"requests": str(REQUESTS.relative_to(ROOT)), "requests_sha256": file_hash(REQUESTS),
              "tokenizer": str(args.tokenizer.relative_to(ROOT)), "oracle": summary,
              "oracle_success_by_status": {s: sum(r["success"] for r in rows if r["status"] == s) /
                                           sum(1 for r in rows if r["status"] == s) for s in ("complete", "partial", "insufficient")},
              "tokens": {k: {"min": min(key(k)), "max": max(key(k)), "mean": round(sum(key(k)) / len(rows), 1)}
                         for k in ("first_prompt_tokens", "last_prompt_tokens", "episode_input_tokens",
                                   "generated_tokens", "max_round_tokens", "max_observation_tokens", "calls")},
              "diagnostics": {"diagnostic_handles": diag_total, "handles_checked": handle_total,
                              "reference_disagreements": diag_problems, "effect": dict(sorted(changed.items()))},
              "per_request": rows, "seconds": round(time.monotonic() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_request"}, ensure_ascii=False, indent=1))
    if summary["success"] != 1 or diag_problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
