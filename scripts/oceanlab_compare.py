#!/usr/bin/env python3
"""Compare evaluation runs of scripts/run_oceanlab_eval.py on the same requests (M4 dev evaluation, M5 test report).

Each run is `label=directory` (with episodes.jsonl and config.json). Per method: greedy success, mean success of the
samples, pass@k, computable success, correct refusals, false refusals, protocol violations, calls and input tokens per
episode, success by request type / status / any extra request field (--by), each with a 95% bootstrap interval over
requests. Pairs (--pair A,B): greedy wins / losses with an exact McNemar p, and the paired difference of the per-request
mean sampled success with a 95% paired-bootstrap interval. Writes <out>.json and <out>.md.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import read_jsonl


def load(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    traces = read_jsonl(directory / "episodes.jsonl")
    if any(t["metrics"]["env_error"] for t in traces):
        raise ValueError(f"{directory}: environment errors; fix and rerun before comparing")
    per = defaultdict(lambda: {"greedy": None, "samples": []})
    for t in traces:
        cell = per[t["id"]]
        if t["sample"] < 0:
            cell["greedy"] = t["metrics"]
        else:
            cell["samples"].append(t["metrics"])
    return config, dict(per)


def bootstrap(values, rounds=10000, seed=0):
    """Mean and 95% percentile interval of the mean over resampled requests."""
    if not values:
        return None
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return [sum(values) / n, means[int(.025 * rounds)], means[int(.975 * rounds) - 1]]


def mcnemar(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def method_table(per, requests, by):
    ids = sorted(per)
    greedy = {i: per[i]["greedy"] for i in ids if per[i]["greedy"]}
    sampled = {i: per[i]["samples"] for i in ids if per[i]["samples"]}
    mean = lambda ms, key: sum(m[key] for m in ms) / len(ms)
    out = {"requests": len(ids)}
    if greedy:
        g = list(greedy.values())
        out["greedy_success"] = bootstrap([m["success"] for m in g])
        out["greedy_computable_success"] = bootstrap([m["success"] for m in g if m["computable"]])
        out["greedy_refusal_correct"] = bootstrap([m["refusal_correct"] for m in g if not m["computable"]])
        out["greedy_false_refusal"] = bootstrap([m["false_refusal"] for m in g if m["computable"]])
        out["greedy_protocol_violation"] = bootstrap([m["protocol_violation"] for m in g])
    if sampled:
        s = list(sampled.values())
        out["samples_per_request"] = min(len(v) for v in s)
        out["sampled_success"] = bootstrap([mean(v, "success") for v in s])
        out["pass_at_k"] = bootstrap([float(any(m["success"] for m in v)) for v in s])
        out["sampled_computable_success"] = bootstrap([mean(v, "success") for v in s if v[0]["computable"]])
        out["sampled_refusal_correct"] = bootstrap([mean(v, "refusal_correct") for v in s if not v[0]["computable"]])
        out["sampled_protocol_violation"] = bootstrap([mean(v, "protocol_violation") for v in s])
        out["S_varies"] = sum(len({m["success"] for m in v}) > 1 for v in s)
    every = [m for i in ids for m in ([per[i]["greedy"]] if per[i]["greedy"] else []) + per[i]["samples"]]
    out["calls_per_episode"] = sum(m["calls"] for m in every) / len(every)
    out["input_tokens_per_episode"] = sum(m["input_tokens"] for m in every) / len(every)
    out["generated_tokens_per_episode"] = sum(m["generated_tokens"] for m in every) / len(every)
    for field in ["request_type", "status"] + by:
        cells = defaultdict(lambda: {"greedy": [], "sampled": []})
        for i in ids:
            spec = requests[i]
            key = spec["reference"]["status"] if field == "status" else str(spec.get(field, "-"))
            if per[i]["greedy"]:
                cells[key]["greedy"].append(per[i]["greedy"]["success"])
            if per[i]["samples"]:
                cells[key]["sampled"].append(mean(per[i]["samples"], "success"))
        out[f"by_{field}"] = {k: {"n": max(len(v["greedy"]), len(v["sampled"])), "greedy": bootstrap(v["greedy"]),
                                  "sampled": bootstrap(v["sampled"])} for k, v in sorted(cells.items())}
    return out


def pair_table(a, b, rounds=10000, seed=0):
    ids = sorted(set(a) & set(b))
    out = {"requests": len(ids)}
    both = [i for i in ids if a[i]["greedy"] and b[i]["greedy"]]
    if both:
        wins = sum(1 for i in both if a[i]["greedy"]["success"] and not b[i]["greedy"]["success"])
        losses = sum(1 for i in both if b[i]["greedy"]["success"] and not a[i]["greedy"]["success"])
        out["greedy"] = {"wins": wins, "losses": losses, "difference": (wins - losses) / len(both), "p": mcnemar(wins, losses)}
    both = [i for i in ids if a[i]["samples"] and b[i]["samples"]]
    if both:
        mean = lambda v: sum(m["success"] for m in v) / len(v)
        out["sampled_difference"] = bootstrap([mean(a[i]["samples"]) - mean(b[i]["samples"]) for i in both], rounds, seed)
    return out


def pct(cell):
    return "—" if not cell else f"{100 * cell[0]:.1f} [{100 * cell[1]:.1f}, {100 * cell[2]:.1f}]"


def markdown(methods, pairs, by):
    lines = ["| 方法 | greedy 成功 | 采样平均成功 | pass@k | 可算（greedy） | 正确拒答（greedy） | 误拒答 | 协议违规 | 调用/episode | 输入 token/episode |",
             "|---|---|---|---|---|---|---|---|---:|---:|"]
    for label, m in methods.items():
        lines.append(f"| {label} | {pct(m.get('greedy_success'))} | {pct(m.get('sampled_success'))} | {pct(m.get('pass_at_k'))} | "
                     f"{pct(m.get('greedy_computable_success'))} | {pct(m.get('greedy_refusal_correct'))} | "
                     f"{pct(m.get('greedy_false_refusal'))} | {pct(m.get('greedy_protocol_violation'))} | "
                     f"{m['calls_per_episode']:.2f} | {m['input_tokens_per_episode']:,.0f} |")
    for field in ["request_type", "status"] + by:
        keys = sorted({k for m in methods.values() for k in m[f"by_{field}"]})
        lines += ["", f"按 {field}（greedy / 采样平均，%）", "", "| 方法 | " + " | ".join(
            f"{k}（{max(m[f'by_{field}'].get(k, {}).get('n', 0) for m in methods.values())}）" for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for label, m in methods.items():
            cells = []
            for k in keys:
                c = m[f"by_{field}"].get(k)
                cells.append("—" if not c else f"{100 * c['greedy'][0]:.1f} / {100 * c['sampled'][0]:.1f}"
                             if c["greedy"] and c["sampled"] else pct(c["greedy"] or c["sampled"]))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if pairs:
        lines += ["", "| 对比 | greedy 胜 / 负（p） | 采样平均差 [95% 区间]（百分点） |", "|---|---|---|"]
        for name, p in pairs.items():
            g = p.get("greedy")
            s = p.get("sampled_difference")
            lines.append(f"| {name} | {g['wins']} / {g['losses']}（{g['p']:.3g}） | "
                         + ("—" if not s else f"{100 * s[0]:+.1f} [{100 * s[1]:+.1f}, {100 * s[2]:+.1f}]") + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="label=evaluation directory")
    p.add_argument("--pair", action="append", default=[], help="A,B: compare run A against run B (repeatable)")
    p.add_argument("--by", action="append", default=[], help="extra request field to break success down by")
    p.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = p.parse_args()
    runs, configs = {}, {}
    for item in args.runs:
        label, directory = item.split("=", 1)
        configs[label], runs[label] = load(directory)
    shas = {c["requests_sha256"] for c in configs.values()}
    if len(shas) != 1:
        raise ValueError("runs were made on different request sets")
    requests = {r["id"]: r for r in read_jsonl(Path(next(iter(configs.values()))["requests"]))}
    methods = {label: method_table(per, requests, args.by) for label, per in runs.items()}
    pairs = {}
    for item in args.pair:
        a, b = item.split(",")
        pairs[f"{a} − {b}"] = pair_table(runs[a], runs[b])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"runs": {label: {k: configs[label].get(k) for k in ("adapter", "skills", "teacher", "scope", "generation", "budget")}
                       for label in runs}, "requests_sha256": shas.pop(), "methods": methods, "pairs": pairs}
    args.out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    text = markdown(methods, pairs, args.by)
    args.out.with_suffix(".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
