#!/usr/bin/env python3
"""Initial SkillBank (SkillRL-style) distilled by the teacher from the common start's trajectories on training requests.

Only training-partition data is used (design section 5: initial skills come from the training region). For each
category (agent/oceanlab/skills.py) a seeded sample of failed and successful C0 trajectories is condensed: request,
actions with key results, and for failures the scorer's reason with the expected parameters (training-time feedback).
The teacher writes category skills and common mistakes from each sample, then general skills from a cross-category
sample, then consolidates everything into one bank with size caps. Skills are rules and procedures; no request-specific
numbers or answers.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.ocean import file_hash, read_jsonl
from agent.oceanlab.data import OceanDepths
from agent.oceanlab.env import SYSTEM_V2
from agent.oceanlab.scoring import TOOL_OF, expected_params, parameter_errors
from agent.oceanlab.skills import CATEGORIES, categories_of
from agent.oceanlab.teacher import Teacher

CAPS = {"general": 5, "task_specific": 3, "common_mistakes": 6}
RULES = ("技能必须是可复用的规则或做法，用一句中文写清“遇到什么情况→怎么做”，每条 principle 不超过 80 个字；"
         "可以引用公开的方法卡和约定（如南纬西经为负、季节按半球、上层X米从5.078 m起、每项分析交一个句柄）；"
         "不要使用尖括号字符（写“小于”“大于”）；不要出现任何具体请求的区域、日期、剖面数、统计数值或答案；不要重复系统提示里逐字已有的句子，要写模型实际容易做错的点。"
         "技能不得与下面的公开规则矛盾；不要从少数例子过度概括：凡是取决于请求原意的做法（如分组、深度起点、是否跨年），"
         "要写成“如何根据请求文字判断”，而不是“一律怎么做”。\n\nAgent 已知的公开规则（系统提示）：\n" + SYSTEM_V2)


def condense(trace, spec, depth):
    lines = [f"请求：{spec['text']}"]
    if spec.get("followup"):
        lines.append(f"追问：{spec['followup']['text']}")
    for r, c in zip(trace["rounds"], trace["calls"] + [None] * len(trace["rounds"])):
        text = r["text"].replace("\n", " ")
        lines.append(f"动作：{text[:260]}")
        if c:
            res = c["result"]
            brief = (f"错误：{res['error'][:80]}" if "error" in res else
                     "得到 " + ", ".join(f"{k}={res[k]}" for k in ("dataset", "result") if k in res))
            lines.append(f"  结果：{brief}")
    m = trace["metrics"]
    if m["success"]:
        lines.append("评分：成功")
        return "\n".join(lines)
    lines.append(f"评分：失败（{m['failure']}；参数错误：{m['parameter_errors'] or '无'}；停止原因：{m['stop']}）")
    want = expected_params(spec, depth)
    for analysis, target in want.items():
        (la, lo) = spec["region"]
        detail = (f"正确参数（{analysis}）：区域 lat {la[0]:g}～{la[1]:g}、lon {lo[0]:g}～{lo[1]:g}；分组 "
                  + "；".join(f"{a}～{b}" for a, b in target["coverage"]))
        if "levels" in target:
            c = spec["consistency"]
            detail += f"；变量 {c['variable']}；深度 {c['depth_range_m'][0]:g}～{c['depth_range_m'][1]:g} m"
        lines.append(detail)
        made = [r for r in trace["registry"]["results"].values() if r["tool"] == TOOL_OF[analysis]]
        if made:
            p = made[-1]["params"]
            g = p["region"]
            lines.append(f"  模型最后一次的参数：区域 lat {g['lat_min']:g}～{g['lat_max']:g}、lon {g['lon_min']:g}～{g['lon_max']:g}；分组 "
                         + "；".join(f"{x['start_date']}～{x['end_date']}" for x in p["groups"])
                         + (f"；深度 {p['depth_min']:g}～{p['depth_max']:g}" if "depth_min" in p else ""))
    return "\n".join(lines)


def ask(teacher, task, cases, schema):
    system = "你是海洋数据分析 Agent 的教练，负责从 Agent 的成功和失败轨迹中提炼可复用的技能。" + RULES + "只输出 JSON。"
    user = f"{task}\n\n输出格式：{schema}\n\n轨迹：\n" + "\n\n".join(f"[{i}] (id={tid})\n{text}" for i, (tid, text) in enumerate(cases, 1))
    return json.loads(teacher.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   temperature=0.0, max_tokens=1500, json_mode=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True, help="episodes.jsonl of the common start on train v2")
    p.add_argument("--requests", type=Path, default=ROOT / "data/processed/oceanlab/v2/oceanlab_train_v2.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/oceanlab/skills/skillbank_v0.json")
    p.add_argument("--cache", type=Path, default=ROOT / "data/processed/oceanlab/skills/teacher_cache.jsonl")
    p.add_argument("--failures", type=int, default=10)
    p.add_argument("--successes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; skill banks are versioned, write a new one")
    specs = {r["id"]: r for r in read_jsonl(args.requests)}
    traces = read_jsonl(args.episodes)
    if any(t["id"] not in specs or specs[t["id"]]["split"] != "train" for t in traces):
        raise ValueError("skills may only be distilled from training-partition episodes")
    depth = OceanDepths().depth
    teacher = Teacher(args.cache)
    rng = random.Random(args.seed)
    by_category = collections.defaultdict(lambda: {"fail": [], "ok": []})
    for t in traces:
        spec = specs[t["id"]]
        texts = [spec["text"]] + ([spec["followup"]["text"]] if spec.get("followup") else [])
        for c in categories_of(texts, bool(spec.get("followup"))):
            by_category[c]["ok" if t["metrics"]["success"] else "fail"].append(t)

    def sample(pool, n):
        pool = sorted(pool, key=lambda t: (t["id"], t["sample"]))
        rng.shuffle(pool)
        seen, out = set(), []
        for t in pool:  # diverse failure kinds first, at most one episode per request
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            out.append(t)
        out.sort(key=lambda t: t["metrics"]["failure"] or "")
        kinds = collections.defaultdict(list)
        for t in out:
            kinds[t["metrics"]["failure"]].append(t)
        picked = []
        while len(picked) < n and any(kinds.values()):
            for k in list(kinds):
                if kinds[k] and len(picked) < n:
                    picked.append(kinds[k].pop(0))
        return picked

    drafts = {}
    schema_cat = ('{"task_specific": [{"id": "类别缩写-序号", "title": "短标题", "principle": "规则", "when": "适用情况", '
                  '"evidence": ["轨迹id"]}], "common_mistakes": [同样格式]}')
    for category in CATEGORIES:
        pool = by_category.get(category)
        if not pool:
            continue
        cases = [(t["id"], condense(t, specs[t["id"]], depth))
                 for t in sample(pool["fail"], args.failures) + sample(pool["ok"], args.successes)]
        drafts[category] = ask(teacher, f"类别：{category}。请从下面的轨迹中，为这一类请求提炼最多 {CAPS['task_specific'] + 1} 条"
                               f"类别技能，以及最多 4 条常见错误（写成“不要…，而要…”）。", cases, schema_cat)
    everything = [t for pool in by_category.values() for t in pool["fail"] + pool["ok"]]
    mixed = sample([t for t in everything if not t["metrics"]["success"]], 12) + sample(
        [t for t in everything if t["metrics"]["success"]], 6)
    general = ask(teacher, f"请从下面跨类别的轨迹中提炼最多 {CAPS['general'] + 1} 条通用技能（适用于所有请求的流程与交付规则）。",
                  [(t["id"], condense(t, specs[t["id"]], depth)) for t in mixed],
                  '{"general": [{"id": "g-序号", "title": "短标题", "principle": "规则", "when": "适用情况", "evidence": ["轨迹id"]}]}')
    consolidate_system = "你负责整理技能库：合并重复、删除与系统规则逐字重复或含具体数值的条目、保留最有用的条目。" + RULES + "只输出 JSON。"
    draft = {"general": general.get("general", []),
             "task_specific": {c: d.get("task_specific", []) for c, d in drafts.items()},
             "common_mistakes": [m for d in drafts.values() for m in d.get("common_mistakes", [])]}
    final = json.loads(teacher.chat([
        {"role": "system", "content": consolidate_system},
        {"role": "user", "content": f"上限：general 最多 {CAPS['general']} 条，每个类别的 task_specific 最多 {CAPS['task_specific']} 条，"
                                    f"common_mistakes 最多 {CAPS['common_mistakes']} 条。类别键保持不变：{list(CATEGORIES)}。"
                                    f"保留各条的 evidence。输出与草稿相同结构的 JSON。\n\n草稿：\n{json.dumps(draft, ensure_ascii=False)}"}],
        temperature=0.0, max_tokens=3000, json_mode=True))
    review = json.loads(teacher.chat([
        {"role": "system", "content": "你负责审核技能库。" + RULES + "只输出 JSON。"},
        {"role": "user", "content": "逐条对照公开规则审核下面的技能库：删除与公开规则矛盾、过度概括（“一律…”而实际取决于请求原意）、"
                                    "与其他条目重复（包括跨 general / task_specific / common_mistakes 的重复）或含具体数值的条目；"
                                    "检查每条的适用条件是否写全（例如只适用于温度的规则，不能写成适用于所有一致性请求）；"
                                    "可以改写措辞使其准确。保持相同结构和上限，保留 evidence。另外输出 removed 列表说明删了哪些、为什么。"
                                    f"\n\n技能库：\n{json.dumps(final, ensure_ascii=False)}"}],
        temperature=0.0, max_tokens=3000, json_mode=True))
    removed = review.pop("removed", [])
    final = review
    final["general"] = final.get("general", [])[:CAPS["general"]]
    final["task_specific"] = {c: v[:CAPS["task_specific"]] for c, v in final.get("task_specific", {}).items() if c in CATEGORIES}
    final["common_mistakes"] = final.get("common_mistakes", [])[:CAPS["common_mistakes"]]
    bank = {"version": args.output.stem, "source_episodes": str(args.episodes), "episodes_sha256": file_hash(args.episodes),
            "requests_sha256": file_hash(args.requests), "teacher": teacher.model, "caps": CAPS, "seed": args.seed,
            "pool_sizes": {c: {k: len(v) for k, v in pool.items()} for c, pool in by_category.items()},
            "review_removed": removed, "teacher_usage": teacher.usage, "teacher_cost_cny": teacher.cost(), **final}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=1) + "\n")
    (args.output.parent / f"{args.output.stem}_draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: bank[k] for k in ("general", "task_specific", "common_mistakes")}, ensure_ascii=False, indent=1))
    print("cost", bank["teacher_cost_cny"], "pools", bank["pool_sizes"])


if __name__ == "__main__":
    main()
