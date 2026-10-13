"""SkillBank evolution during RL (SkillRL's recursive skill evolution, design section 8).

At a validation step the categories that do badly are chosen (`weak_categories`). For each, the teacher reads the
current skills of that category and a diverse sample of the policy's failed validation episodes, condensed with the
scorer's reason and the correct parameters (validation feedback: request, actions, key results, why it failed; the
whole episode, not a fixed number of last steps), and proposes at most `max_new` new skills that the current bank
does not cover; a review pass against the public rules and the evidence removes contradictions, over-generalisations,
duplicates and request-specific values (`propose`).

Verification (our addition; SkillRL's teacher is o3): the teacher can still write a plausible but wrong rule, e.g.
"MAM means September-November in the southern hemisphere" from a failure whose months were right. So each candidate
is tried alone (current bank + that skill) on the category's evidence failures and as many of its successes, greedy;
it is accepted only if it fixes at least `min_gain` failures and fixes more than it breaks (`verdict`). Accepted
skills are added under new ids up to per-level caps (`add_skills`; existing skills are never removed).
"""
import collections
import copy
import json
import random

from agent.oceanlab.skills import CATEGORIES, categories_of, render

CAPS = {"task_specific": 6, "common_mistakes": 10}  # totals per category / for common mistakes after evolution
SCHEMA = ('{"task_specific": [{"title": "短标题", "principle": "规则", "when": "适用情况", "evidence": ["轨迹id"]}], '
          '"common_mistakes": [同样格式]}')


def category_success(traces, specs):
    """{category: (episodes, success rate)} over validation episodes."""
    cells = collections.defaultdict(list)
    for trace in traces:
        spec = specs[trace["id"]]
        texts = [spec["text"]] + ([spec["followup"]["text"]] if spec.get("followup") else [])
        for category in categories_of(texts, bool(spec.get("followup"))):
            cells[category].append(trace["metrics"]["success"])
    return {c: (len(v), sum(v) / len(v)) for c, v in cells.items()}


def weak_categories(stats, overall, threshold, min_episodes, max_categories, relative=False):
    """Categories with at least `min_episodes` episodes and success below `threshold` (SkillRL: 0.4), or with
    `relative`, below the overall validation success; the weakest `max_categories` of them."""
    bar = overall if relative else threshold
    chosen = [(rate, c) for c, (n, rate) in stats.items() if n >= min_episodes and rate < bar]
    return [c for _, c in sorted(chosen)[:max_categories]]


def diverse_failures(traces, n, seed):
    """Up to n failed episodes, one per request, cycling over failure kinds so one error type does not fill them."""
    kinds = collections.defaultdict(list)
    pool = sorted((t for t in traces if not t["metrics"]["success"]), key=lambda t: t["id"])
    random.Random(seed).shuffle(pool)
    seen = set()
    for t in pool:
        if t["id"] not in seen:
            seen.add(t["id"])
            kinds[t["metrics"]["failure"]].append(t)
    picked = []
    while len(picked) < n and any(kinds.values()):
        for kind in sorted(kinds):
            if kinds[kind] and len(picked) < n:
                picked.append(kinds[kind].pop(0))
    return picked


def propose(bank, category, failures, condense, teacher, rules, max_new=3):
    """Teacher proposal plus review for one category. failures: [(trace, spec)]; condense(trace, spec) -> text.
    Returns (candidate skills with their level, log)."""
    current = bank["general"] + bank["task_specific"].get(category, []) + bank["common_mistakes"]
    system = "你是海洋数据分析 Agent 的教练，负责根据 Agent 的失败轨迹补充技能库。" + rules + "只输出 JSON。"
    evidence = "\n\n".join(f"[{i}] (id={t['id']})\n{condense(t, spec)}" for i, (t, spec) in enumerate(failures, 1))
    user = (f"类别：{category}。下面是 Agent 当前在这一类请求上使用的技能，以及它在验证请求上的失败轨迹（附评分原因和正确参数）。"
            "每条轨迹里“正确参数”是标准答案，“模型最后一次的参数”是做错的地方；只根据二者实际不同的字段和评分原因总结，"
            f"不要推测轨迹里没有出现的错误。请找出现有技能没有覆盖、但导致这些失败的做法问题，提出最多 {max_new} 条新技能"
            f"（类别技能或常见错误，常见错误写成“不要…，而要…”）。不要重复或改写现有技能；如果现有技能已经覆盖，就少提或不提。"
            f"\n\n现有技能：\n{render(current)}\n\n输出格式：{SCHEMA}\n\n失败轨迹：\n{evidence}")
    proposal = json.loads(teacher.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       temperature=0.0, max_tokens=1500, json_mode=True))
    proposed = [dict(skill, level=level) for level in ("task_specific", "common_mistakes")
                for skill in proposal.get(level, []) if isinstance(skill, dict) and skill.get("principle")]
    # The review decides by number, so each skill keeps its level whatever structure the reply uses.
    listing = "\n".join(f"{i}. [{'类别技能' if s['level'] == 'task_specific' else '常见错误'}] {s.get('title', '')}："
                        f"{s['principle']}（适用：{s.get('when', '')}）" for i, s in enumerate(proposed, 1))
    review = json.loads(teacher.chat([
        {"role": "system", "content": "你负责审核新增技能。" + rules + "只输出 JSON。"},
        {"role": "user", "content": "逐条对照公开规则、现有技能和失败轨迹审核下面编号的新增技能。删除：与公开规则矛盾的；"
                                    "与轨迹中任何“正确参数”不一致的（照这条技能做，会得到与正确参数不同的区域、分组、深度或状态）；"
                                    "针对的错误在轨迹里并没有出现的；过度概括（“一律…”而实际取决于请求原意）的；与现有技能或彼此重复的；"
                                    "含具体数值或具体区域日期的。检查适用条件是否写全。保留的条目可以改写措辞使其准确。"
                                    '输出格式：{"keep": [{"index": 序号, "title": "标题", "principle": "规则", "when": "适用情况"}], '
                                    '"removed": [{"index": 序号, "reason": "原因"}]}'
                                    f"\n\n现有技能：\n{render(current)}\n\n新增技能：\n{listing}\n\n失败轨迹：\n{evidence}"}],
        temperature=0.0, max_tokens=1500, json_mode=True)) if proposed else {"keep": [], "removed": []}
    candidates, used = [], set()
    for keep in review.get("keep", []):
        index = keep.get("index") if isinstance(keep, dict) else None
        if not isinstance(index, int) or not 1 <= index <= len(proposed) or index in used:
            continue
        used.add(index)
        candidates.append({**proposed[index - 1], **{k: keep[k] for k in ("title", "principle", "when") if keep.get(k)}})
    log = {"category": category, "failures": [t["id"] for t, _ in failures], "proposed": proposed,
           "removed": review.get("removed", []), "candidates": candidates}
    return candidates, log


def with_skill(bank, category, skill):
    """The bank plus one candidate (caps ignored), for verifying that candidate alone."""
    trial = copy.deepcopy(bank)
    item = {"id": "candidate", **{k: skill.get(k, "") for k in ("title", "principle", "when")}, "evidence": []}
    (trial["task_specific"].setdefault(category, []) if skill["level"] == "task_specific" else trial["common_mistakes"]).append(item)
    return trial


def verdict(baseline, trial, min_gain=1):
    """baseline / trial: {request id: S} on the same verification requests -> (fixed, broken, accepted)."""
    fixed = sum(1 for i, s in trial.items() if s and not baseline[i])
    broken = sum(1 for i, s in trial.items() if baseline[i] and not s)
    return fixed, broken, fixed >= min_gain and fixed > broken


def add_skills(bank, category, skills, tag, max_new=3):
    """A new bank version with at most `max_new` of these skills added under new ids, within the level caps."""
    new = copy.deepcopy(bank)
    added = []
    for skill in skills:
        level = skill["level"]
        target = new["task_specific"].setdefault(category, []) if level == "task_specific" else new["common_mistakes"]
        if len(added) >= max_new or len(target) >= CAPS[level]:
            continue
        item = {"id": f"{tag}-{category}-{len(added) + 1}", "title": skill.get("title", ""), "principle": skill["principle"],
                "when": skill.get("when", ""), "evidence": skill.get("evidence", [])}
        target.append(item)
        added.append({**item, "level": level})
    return new, added


def evolve(bank, category, failures, condense, teacher, rules, tag, max_new=3):
    """Proposal, review and addition without verification. Returns (new bank, log)."""
    candidates, log = propose(bank, category, failures, condense, teacher, rules, max_new)
    new, log["added"] = add_skills(bank, category, candidates, tag, max_new)
    return new, log


def sample_successes(traces, n, seed):
    """Up to n successful episodes, one per request (verification controls)."""
    pool = sorted({t["id"]: t for t in traces if t["metrics"]["success"]}.values(), key=lambda t: t["id"])
    random.Random(seed).shuffle(pool)
    return pool[:n]
