"""SkillBank for the OceanDepths agent (SkillRL-style hierarchical experience) and its retrieval.

A bank is JSON with three levels:
    {"version": ..., "general": [skill...], "task_specific": {category: [skill...]}, "common_mistakes": [skill...]}
    skill = {"id", "title", "principle", "when", "evidence": [trajectory ids]}
Skills hold reusable rules and procedures, never per-request answers or numbers from the data.

Retrieval is deterministic and transparent: general skills and common mistakes always, plus the task-specific skills
of every category whose keywords occur in the conversation's user messages (multiturn when there is a follow-up).
The retrieved skills are rendered as a preface of the first user message; the system prompt stays the environment's,
and removing the preface gives exactly the plain prompt (C1_plain uses that).
"""
import json
from pathlib import Path

CATEGORIES = {
    "mld": ("混合层", "MLD", "mixed layer", "Mixed Layer"),
    "consistency": ("GLORYS", "EN4", "一致性", "bias", "RMSE", "MAE", "再分析", "偏差"),
    "season": ("冬季", "夏季", "春季", "秋季", "冬天", "夏天", "春天", "秋天", "季节"),
    "relative_time": ("去年", "前一年", "上一年", "同期", "同月", "前年"),
    "multiturn": (),
}
PREFACE = "以下是从以往分析中总结的经验，供参考：\n{skills}\n\n用户请求：\n"


def load(path):
    bank = json.loads(Path(path).read_text())
    for key in ("general", "task_specific", "common_mistakes"):
        if key not in bank:
            raise ValueError(f"skill bank lacks {key}")
    return bank


def categories_of(texts, multiturn):
    joined = "\n".join(texts)
    found = [c for c, words in CATEGORIES.items() if any(w in joined for w in words)]
    return found + (["multiturn"] if multiturn and "multiturn" not in found else [])


def retrieve(bank, texts, multiturn=False):
    """Skills for a conversation: general, common mistakes and the matching categories' task-specific skills."""
    chosen = list(bank["general"])
    for category in categories_of(texts, multiturn):
        chosen += bank["task_specific"].get(category, [])
    chosen += bank["common_mistakes"]
    seen, unique = set(), []
    for skill in chosen:
        if skill["id"] not in seen:
            seen.add(skill["id"])
            unique.append(skill)
    return unique


def render(skills):
    # No angle brackets reach the model: it copies text into its plans, where they are a protocol violation.
    clean = lambda text: text.replace("<=", "不大于").replace(">=", "不小于").replace("<", "小于").replace(">", "大于")
    lines = [f"{i}. {clean(s['title'])}：{clean(s['principle'])}" for i, s in enumerate(skills, 1)]
    return PREFACE.format(skills="\n".join(lines))


class SkillMemory:
    """Harness memory policy: prefaces the first user message with the retrieved skills."""
    name = "skills"

    def __init__(self, bank):
        self.bank = bank

    def preface(self, task):
        texts = [task["text"]] + [m["content"] for m in task.get("history", []) if m["role"] == "user"]
        skills = retrieve(self.bank, texts, multiturn=bool(task.get("history")))
        return render(skills), [s["id"] for s in skills]
