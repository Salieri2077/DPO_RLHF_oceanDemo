#!/usr/bin/env python3
"""Request set v2 (agent/oceanlab/requests_v2.py): sampled specs, teacher-written user text with independent
re-extraction, references, multiturn histories from the oracle, and an end-to-end oracle check of every task.

Specs are oversampled by 10% per (request type, family) stratum; within each stratum the first specs whose text
passes verification fill the planned quota, in sampling order (selection depends only on text fidelity, never on any
agent's behaviour). Teacher calls are cached (`--cache`), so reruns are free and reproducible.
"""
import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash
from agent.oceanlab import requests_v2 as r2
from agent.oceanlab.data import OceanDepths, PARTITION, QC_POLICY, REVISION
from agent.oceanlab.env import OceanLab, OceanLabHarness
from agent.oceanlab.oracle import oracle_texts
from agent.oceanlab.scoring import Scorer
from agent.oceanlab.tasks import reference_result, write_jsonl
from agent.oceanlab.teacher import Teacher
from agent.sdk import run_trajectory
from trainer.train_agent_hf import native_tokenizer

ATTEMPTS = 4


def clean(text):
    return text.strip().strip("“”\"'「」").strip()


def write_text(teacher, spec):
    """Writer + independent extraction, retried with other personas. Returns the generation record."""
    log = []
    for attempt in range(ATTEMPTS):
        messages, style = r2.writer_messages(spec, attempt)
        text = clean(teacher.chat(messages, temperature=1.0, max_tokens=400))
        base = spec.get("first", spec)
        bad = r2.mismatches(base, teacher.chat(r2.extractor_messages([text]), max_tokens=300, json_mode=True))
        record = {"attempt": attempt, "style": style, "text": text, "mismatch": bad}
        if not bad and spec["request_type"] == "multiturn":
            follow_messages, follow_style = r2.writer_messages({**spec, "text": text}, attempt, followup=True)
            followup = clean(teacher.chat(follow_messages, temperature=1.0, max_tokens=200))
            bad = r2.mismatches(spec, teacher.chat(r2.extractor_messages([text, followup]), max_tokens=300, json_mode=True))
            record.update(followup=followup, followup_style=follow_style, followup_mismatch=bad)
        log.append(record)
        if not bad:
            return {"ok": True, "text": text, "followup": record.get("followup"), "attempts": attempt + 1,
                    "style": style, "log": log}
    return {"ok": False, "attempts": ATTEMPTS, "log": log}


def scripted(tokenizer, texts):
    texts = iter(texts)

    def generate(ids, limit):
        return tokenizer.encode(next(texts), add_special_tokens=False) + [tokenizer.eos_token_id], []
    return generate


def build_task(spec, gen, lab, tokenizer):
    """The final task: text, optional multiturn history (first turn answered by the oracle), reference."""
    task = {k: v for k, v in spec.items() if k not in ("first", "period_fact")}
    task.update(text=gen["text"], task_version=f"oceanlab_{spec['split']}_v2",
                generation={"attempts": gen["attempts"], "style": gen["style"]})
    if spec["request_type"] == "multiturn":
        first = {**spec["first"], "id": spec["id"] + "-turn1", "text": gen["text"]}
        trace = run_trajectory(first, tokenizer, lab, scripted(tokenizer, oracle_texts(first, lab)),
                               harness=OceanLabHarness(None, prompt="v2"))
        task["history"] = trace["messages"][2:] + [{"role": "user", "content": gen["followup"]}]
        task["history_calls"] = [c["call"] for c in trace["calls"]]
        task["history_handles"] = [c["result"].get("dataset", c["result"].get("result")) for c in trace["calls"]]
        task["first"] = {k: spec["first"][k] for k in ("family", "region", "groups", "analyses") if k in spec["first"]}
        if "consistency" in spec["first"]:
            task["first"]["consistency"] = spec["first"]["consistency"]
        task["followup"] = {**spec["followup"], "text": gen["followup"]}
    return task


def stratum(spec):
    return f"{spec['request_type']}/{(spec.get('first') or spec)['family']}"


def spec_key(spec):
    first = spec.get("first") or spec
    return (spec["request_type"], spec["kind"], tuple(map(tuple, first["region"])), tuple(tuple(g["months"]) for g in first["groups"]),
            spec.get("followup", {}).get("type"))


def process(specs, quota, args, data, teacher):
    """Text generation and verification, quota filling in sampling order, references and oracle checks."""
    with ThreadPoolExecutor(args.workers) as pool:
        generated = list(pool.map(lambda s: write_text(teacher, s), specs))
    print(f"text: {sum(g['ok'] for g in generated)}/{len(specs)} verified, teacher {teacher.usage}, cost ≈ ¥{teacher.cost()}",
          flush=True)
    chosen, used = [], collections.Counter()
    for spec, gen in zip(specs, generated):
        if gen["ok"] and (quota is None or used[stratum(spec)] < quota[stratum(spec)]):
            used[stratum(spec)] += 1
            chosen.append((spec, gen))
    lab = OceanLab(args.split, data)
    tokenizer = native_tokenizer(args.model_path, False)
    harness = OceanLabHarness(Scorer(data), prompt="v2")
    tasks, rejected = [], []
    for spec, gen in chosen:
        task = build_task(spec, gen, lab, tokenizer)
        task["reference"], agree = reference_result(data, task)
        if not agree:
            rejected.append({"id": task["id"], "reason": "tool/reference disagreement"})
            continue
        episode = lab.episode()
        for call in task.get("history_calls", []):
            episode.execute(call)
        trace = run_trajectory(task, tokenizer, lab, scripted(tokenizer, oracle_texts(task, lab, episode)), harness=harness)
        m = trace["metrics"]
        if m["success"] != 1 or m["env_error"]:
            rejected.append({"id": task["id"], "reason": f"oracle {m['failure']} env={m['env_error']}"})
            continue
        tasks.append(task)
    return generated, used, tasks, rejected


def describe(tasks):
    count = lambda key: dict(collections.Counter(key(t) for t in tasks))
    return {"request_type": count(lambda t: t["request_type"]), "kind": count(lambda t: f"{t['request_type']}/{t['kind']}"),
            "followup": count(lambda t: t.get("followup", {}).get("type", "-")), "family": count(lambda t: t["family"]),
            "status": count(lambda t: t["reference"]["status"]),
            "hemisphere": count(lambda t: "N" if t["region"][0][0] >= 0 else "S")}


def text_stats(specs, generated):
    attempts = collections.Counter(g["attempts"] for g in generated if g["ok"])
    failures = collections.Counter(tuple(r["mismatch"] or r.get("followup_mismatch") or []) for g in generated
                                   for r in g["log"] if r["mismatch"] or r.get("followup_mismatch"))
    return {"attempts_when_verified": dict(sorted(attempts.items())),
            "text_failures_by_field": {"+".join(k): v for k, v in failures.most_common()},
            "unverified_by_kind": dict(collections.Counter(f"{s['request_type']}/{s['kind']}"
                                                           for s, g in zip(specs, generated) if not g["ok"]))}


def supplement(args, data, teacher, started):
    """Fill the strata a frozen-in-progress set left short, from a fresh sample (another seed), same pipeline."""
    path = args.output / f"oceanlab_{args.split}_v2.jsonl"
    manifest_path = args.output / f"oceanlab_{args.split}_v2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    need = collections.Counter(manifest["unfilled_strata"])
    if not manifest.get("supplements"):  # rounded quotas sum below the plan; sample_specs pads with paraphrase/mld
        quota_total = sum(round(manifest["planned"] * share * fshare)
                          for share in r2.TYPE_SHARE.values() for fshare in r2.FAMILY_SHARE.values())
        need["paraphrase/mld"] += manifest["planned"] - quota_total
    need = +need
    if not need:
        print("nothing to supplement")
        return
    existing = [json.loads(l) for l in path.open(encoding="utf-8")]
    keys = {spec_key(t) for t in existing}
    seed = args.seed + 1000 * (len(manifest.get("supplements", [])) + 1)
    candidates, taken = [], collections.Counter()
    for s in r2.sample_specs(args.split, max(400, 4 * sum(need.values())), data, seed=seed):
        if stratum(s) in need and spec_key(s) not in keys and taken[stratum(s)] < 2 * need[stratum(s)] + 2:
            taken[stratum(s)] += 1
            candidates.append(s)
    specs = candidates  # at most twice the shortfall (+2) per stratum, in sampling order
    for s in specs:
        s["id"] = s["id"].replace(f"{args.split}2-", f"{args.split}2s{seed // 1000}-")
    generated, used, tasks, rejected = process(specs, need, args, data, teacher)
    with path.open("a", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    everything = existing + tasks
    manifest.setdefault("supplements", []).append({
        "seed": seed, "sampled": len(specs), "needed": dict(need), "kept": len(tasks), "rejected": rejected,
        **text_stats(specs, generated), "teacher_usage": teacher.usage, "teacher_cost_cny": teacher.cost(),
        "seconds": round(time.monotonic() - started)})
    still = collections.Counter(need)
    still.subtract(collections.Counter(stratum(t) for t in tasks))
    manifest.update(kept=len(everything), unfilled_strata={k: v for k, v in still.items() if v > 0}, **describe(everything),
                    sha256=file_hash(path))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    with (args.output / f"oceanlab_{args.split}_v2_generation_log.jsonl").open("a", encoding="utf-8") as f:
        for s, g in zip(specs, generated):
            f.write(json.dumps({"id": s["id"], **g}, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("kept", "unfilled_strata", "status")} | {"supplement": manifest["supplements"][-1]["kept"]},
                     ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "dev"], required=True)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="pilot: N evenly spaced sampled specs")
    p.add_argument("--supplement", action="store_true", help="fill the strata the written set left short")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", type=Path, default=ROOT / "data/processed/oceanlab/v2/teacher_cache.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/oceanlab/v2")
    p.add_argument("--model_path", type=Path, default=Path("/data/anhuang/oceanheart_models/qwen3-1.7b"))
    args = p.parse_args()
    started = time.monotonic()
    data = OceanDepths()
    teacher = Teacher(args.cache)
    if args.supplement:
        return supplement(args, data, teacher, started)
    version = f"oceanlab_{args.split}_v2" + (f"_pilot{args.limit}" if args.limit else "")
    path = args.output / f"{version}.jsonl"
    if path.exists() and not args.limit:
        raise FileExistsError(f"{path} exists; request sets are frozen once written")
    specs = r2.sample_specs(args.split, round(args.count * 1.1), data, seed=args.seed)
    quota = collections.Counter({f"{rtype}/{family}": round(args.count * share * fshare)
                                 for rtype, share in r2.TYPE_SHARE.items() for family, fshare in r2.FAMILY_SHARE.items()})
    if args.limit:  # pilot: an evenly spaced sample across strata
        specs = specs[::max(len(specs) // args.limit, 1)][:args.limit]
    generated, used, tasks, rejected = process(specs, None if args.limit else quota, args, data, teacher)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, tasks)
    manifest = {
        "version": version, "split": args.split, "seed": args.seed, "dataset_revision": REVISION, "qc_policy": QC_POLICY,
        "partition": PARTITION, "teacher": teacher.model, "sampled": len(specs), "planned": args.count,
        "text_verified": sum(g["ok"] for g in generated), "kept": len(tasks), "rejected": rejected,
        "unfilled_strata": {k: quota[k] - used[k] for k in quota if used[k] < quota[k]} if not args.limit else {},
        **text_stats(specs, generated), **describe(tasks),
        "teacher_usage": teacher.usage, "teacher_cost_cny": teacher.cost(),
        "env_sha256": file_hash(ROOT / "agent/oceanlab/env.py"), "requests_v2_sha256": file_hash(ROOT / "agent/oceanlab/requests_v2.py"),
        "seconds": round(time.monotonic() - started), "sha256": file_hash(path)}
    (args.output / f"{version}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    (args.output / f"{version}_generation_log.jsonl").write_text(
        "".join(json.dumps({"id": s["id"], **g}, ensure_ascii=False) + "\n" for s, g in zip(specs, generated)))
    print(json.dumps({k: v for k, v in manifest.items() if k != "rejected"} | {"rejected": len(rejected)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
