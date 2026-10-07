#!/usr/bin/env python3
"""ReAct v3 tool-SFT data: varied phrasing, balanced operations/units, held-out phrasing families.

Only the data recipe changes. Runtime protocol, tools and scorer stay those of ReAct v2, so the
manifest keeps version ocean-react-v2 and old/new weights remain comparable on react-v2 val/test.
Every demonstration is executed and scored through the real harness; no API, judge or download.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from agent.ocean import TOOLS, calculate, file_hash, read_jsonl
from agent.react import KINDS, VERSION, ReactTools, run_trajectory
from scripts import ocean_react_templates as bank
from scripts.prepare_ocean_data import atomic_jsonl, normalized, percentile, stable_score
from scripts.prepare_ocean_react import digest, tool

SEED = 42
# Jobs per split; card_quote is a retrieve task and direct_delimited a direct task.
CONTRAST_HELDOUT = {"calculate": 50, "recovery": 50, "chain": 50, "clarify": 50, "retrieve": 40, "card_quote": 10,
                    "direct": 35, "direct_delimited": 15}
RECIPES = {
    "react-v3-diverse": {"contrast": False, "sizes": {
        "train": {"calculate": 2400, "recovery": 1600, "chain": 2000, "retrieve": 2000, "direct": 1600, "clarify": 2000},
        "val": dict.fromkeys(KINDS, 50), "test": dict.fromkeys(KINDS, 50)}},
    # v3 plus contrast samples for the v3 failures: record numbers that are not quantities, record number
    # plus a stated missing field, varied card ids, form-style card questions, quoting a card instead of
    # computing, delimited records, and conflicts compared field by field before clarifying.
    "react-v3.1-contrast": {"contrast": True, "sizes": {
        "train": {"calculate": 2400, "recovery": 1600, "chain": 2600, "clarify": 2600, "retrieve": 2000,
                  "card_quote": 400, "direct": 1600, "direct_delimited": 600},
        "val": CONTRAST_HELDOUT, "test": CONTRAST_HELDOUT}},
}
JOB_KIND = {"card_quote": "retrieve", "direct_delimited": "direct"}
V1_REPLAY = {"calculate": 240, "retrieve": 240, "chain": 120}  # original v1 phrasing, train only
MAX_JACCARD, MAX_COMMON_RUN = .3, 8  # training phrasing vs ReAct v2 held-out phrasing

FIELDS = {"distance": ("speed", "hours"), "speed": ("distance", "hours"), "time": ("distance", "speed")}
TOOL_UNIT = {"speed": "km/h", "hours": "h", "distance": "km"}
RESULT_UNIT = {"distance": "km", "speed": "km/h", "time": "h"}
ALT_UNITS = {"speed": ("kn", "m/s"), "distance": ("nmi", "m")}  # converted before the formula
ANSWER_ALT = {"distance": "nmi", "speed": "kn"}  # converted after the formula
UNITS = {"distance": ("km", "m", "nmi"), "speed": ("km/h", "kn", "m/s")}
CONVERT_PAIRS = [(a, b) for dim in UNITS.values() for a in dim for b in dim if a != b]
RANGES = {"km/h": (4, 60), "kn": (3, 35), "m/s": (1, 20), "km": (5, 600), "nmi": (3, 320), "m": (200, 60000), "h": (.5, 24)}
DELTAS = {"speed": (2, 3, 4, 5, 6), "hours": (.5, 1, 1.5, 2, 3), "distance": (5, 10, 20, 30, 50)}
CLARIFY_PAIRS = [(op, missing) for op, fields in FIELDS.items() for missing in fields]
FIELD_CN = {"speed": "航速", "hours": "航行时长", "distance": "航程"}
ASKED_CN = {"distance": "航程", "speed": "平均航速", "time": "航行时间"}
MISSING = {"speed": ("航速", "缺少航速，请补充航速的数值和单位。"),
           "hours": ("时长", "缺少航行时长，请补充时长的数值和单位。"),
           "distance": ("航程", "缺少航程，请补充航程的数值和单位。")}
MISSING_CARD = ("资料库中未找到该航次资料，请补充正确的编号或原始资料。", ["资料", "补充"])
FIXED_PLAN = {"search": "先查找相关海洋资料。", "calc": "按已知数值与单位计算。"}
RETRY_PLAN = "工具暂时不可用，重试一次。"
CARD_ID = re.compile(r"(?:[A-Z]|rec|trip|log)?[0-9a-f]{10}")


def skeleton(template, mark="□"):
    text = re.sub(r"\{[a-z_]+\}", mark, template).replace("[文档编号]", mark).replace("只回答数值和单位。", "")
    return re.sub(r"\s+", "", text)


def jaccard(a, b, n=3):
    grams = lambda text: {text[i:i + n] for i in range(len(text) - n + 1)}
    x, y = grams(skeleton(a)), grams(skeleton(b))
    return len(x & y) / max(1, len(x | y))


def common_run(a, b):
    """Longest shared substring, placeholders excluded."""
    a, b = skeleton(a, "\x01"), skeleton(b, "\x02")
    best, previous = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def split_bank():
    """Stable-hash phrasing families per category; legacy v1/v2 phrasing stays train-only."""
    families = {split: {} for split in ("train", "val", "test")}
    for category, templates in {**bank.CATEGORIES, **bank.CONTRAST_CATEGORIES}.items():
        ordered = sorted(enumerate(templates), key=lambda p: stable_score(category + "\x00" + p[1], SEED))
        k = max(1 if category.startswith("card.") else 2, round(len(templates) / 10))
        items = [(f"{category}#{i}", t) for i, t in ordered]
        legacy = [(f"{category}#L{i}", t) for i, t in enumerate(bank.LEGACY_TRAIN.get(category, []))]
        families["val"][category], families["test"][category] = items[:k], items[k:2 * k]
        families["train"][category] = items[2 * k:] + legacy
    seen = {}
    for split, categories in families.items():
        for items in categories.values():
            for _, template in items:
                if seen.setdefault(template, split) != split:
                    raise ValueError(f"Phrasing shared across splits: {template}")
    return families


def leakage_report(families):
    """Training phrasing must stay away from ReAct v2 held-out phrasing; v3 val/test are reported too."""
    worst = []
    for category, items in families["train"].items():
        if category.startswith("card.") or category in ("suffix", "card_ref"):
            continue
        pool = bank.V2_HELDOUT_PREFIXES if category == "prefix" else bank.V2_HELDOUT
        for fid, template in items:
            worst.append((max(jaccard(template, h) for h in pool), max(common_run(template, h) for h in pool), fid))
    held = []
    for split in ("val", "test"):
        for category, items in families[split].items():
            if category in ("prefix", "suffix", "card_ref") or category.startswith("card."):
                continue
            for fid, template in items:
                held.append((max(jaccard(template, t) for _, t in families["train"][category]), fid))
    report = {"vs_react_v2_heldout": {"max_jaccard": round(max(w[0] for w in worst), 3),
                                      "max_common_run": max(w[1] for w in worst),
                                      "thresholds": {"jaccard": MAX_JACCARD, "common_run": MAX_COMMON_RUN},
                                      "closest": [w[2] for w in sorted(worst, reverse=True)[:5]]},
              "v3_heldout_vs_train": {"max_jaccard": round(max(h[0] for h in held), 3),
                                      "closest": [h[1] for h in sorted(held, reverse=True)[:5]]}}
    if report["vs_react_v2_heldout"]["max_jaccard"] >= MAX_JACCARD or report["vs_react_v2_heldout"]["max_common_run"] >= MAX_COMMON_RUN:
        raise ValueError(f"Training phrasing too close to ReAct v2 held-out phrasing: {report}")
    return report


def is_english(template):
    return not re.search(r"[\u4e00-\u9fff]", re.sub(r"\{[a-z_]+\}|\[文档编号\]", "", template))


def unit_word(rng, unit, english=False):
    return rng.choice((bank.ENGLISH_UNIT_SURFACES if english else bank.UNIT_SURFACES)[unit])


def quantity(rng, value, unit, english=False):
    word = unit_word(rng, unit, english)
    space = " " if english or (word.isascii() and rng.random() < .5) else ""
    return f"{value:g}{space}{word}"


def sample(rng, unit):
    low, high = RANGES[unit]
    if unit == "m":
        return float(rng.randrange(int(low), int(high), 10))
    value = rng.uniform(low, high)
    return float(max(1, round(value))) if rng.random() < .3 else round(value, 1)


def number(value):
    return json.dumps(value)


def record_id(rng):
    """A record number containing digits; a label, never a quantity."""
    hexes = "".join(rng.choice("0123456789abcdef") for _ in range(10))
    return rng.choice([hexes, "N" + hexes[:9], f"HY{rng.randint(2015, 2026)}-{rng.randint(1, 999):03d}",
                       f"{rng.randint(2015, 2026)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}-{rng.randint(1, 20):02d}",
                       f"No.{rng.randint(1000, 9999)}", f"SR-{rng.randint(10, 99)}{rng.choice('ABCDEFGH')}"])


def solve(op, values, units, target):
    """Real tool steps: convert any non-tool input unit, apply the formula, convert the answer if asked."""
    steps, args = [], {"operation": op}
    for field in FIELDS[op]:
        value = values[field]
        if units[field] != TOOL_UNIT[field]:
            step = dict(operation="convert", value=value, from_unit=units[field], to_unit=TOOL_UNIT[field])
            steps.append(step)
            value = calculate(step)["value"]
        args[field] = value
    steps.append(args)
    result = calculate(args)
    if target != result["unit"]:
        step = dict(operation="convert", value=result["value"], from_unit=result["unit"], to_unit=target)
        steps.append(step)
        result = calculate(step)
    return steps, result


class Builder:
    def __init__(self, split, families, rng, keys, forbidden, known_ids, recipe):
        self.split, self.families, self.rng = split, families[split], rng
        self.keys, self.forbidden, self.known_ids = keys, forbidden, known_ids
        self.recipe, self.contrast = recipe, RECIPES[recipe]["contrast"]
        self.checker = ReactTools([])
        self.tasks, self.docs = [], []

    def pick(self, category, need=None):
        return self.rng.choice([f for f in self.families[category] if not need or need in f[1]])

    def claim(self, steps):
        """Unique numeric plan within v3; no step may repeat a v1/v2 held-out calculation."""
        key = json.dumps(steps, sort_keys=True, ensure_ascii=False)
        if key in self.keys or any(json.dumps(s, sort_keys=True) in self.forbidden for s in steps if isinstance(s, dict)):
            return False
        self.keys.add(key)
        return True

    def card_id(self, seed):
        for salt in range(1000):
            h = digest(f"{self.recipe}:{self.split}:{seed}:{salt}")
            cid = "C" + h[:10]
            if self.contrast:  # ids led by other letters, digits or words, so an id is never read as a number
                leads = ["C"] * 5 + ["K", "R", "X", "", ("rec", "trip", "log")[int(h[12], 16) % 3]]
                cid = leads[int(h[10:12], 16) % 10] + h[:10]
            if not {cid, cid + "a", cid + "b"} & self.known_ids:
                self.known_ids.update({cid, cid + "a", cid + "b"})
                return cid
        raise ValueError("card id space exhausted")

    def add(self, kind, subtype, body, families, numeric, **fields):
        used = list(families)
        prefix = suffix = ""
        if self.rng.random() >= .3:
            fid, prefix = self.pick("prefix")
            used.append(fid)
        if numeric and self.rng.random() >= .25:
            fid, suffix = self.pick("suffix")
            used.append(fid)
            suffix = (" " if body[-1].isascii() else "") + suffix
        self.tasks.append({"kind": kind, "subtype": subtype, "question": prefix + body + suffix, "families": used, **fields})

    def calculation(self, i, kind):
        rng = self.rng
        op = ("distance", "speed", "time", "convert")[i % 4]
        while True:
            if op == "convert":
                source, target = CONVERT_PAIRS[(i // 4) % len(CONVERT_PAIRS)]
                dim = "distance" if source in UNITS["distance"] else "speed"
                fid, template = self.pick(rng.choice(("convert.any", "convert.any", "convert." + dim)))
                english = is_english(template)
                value = sample(rng, source)
                steps = [dict(operation="convert", value=value, from_unit=source, to_unit=target)]
                result = calculate(steps[0])
                fields = dict(value_q=quantity(rng, value, source, english), to_u=unit_word(rng, target, english))
                subtype = f"convert:{source}->{target}"
            else:
                output = op != "time" and any("{au}" in t for _, t in self.families["calc." + op])
                variant = rng.choices(("plain", "input", "output"), (.6, .25, .15) if output else (.7, .3, 0))[0]
                units = {f: TOOL_UNIT[f] for f in FIELDS[op]}
                if variant == "input":
                    field = rng.choice([f for f in FIELDS[op] if f != "hours"])
                    units[field] = rng.choice(ALT_UNITS[field])
                answer = ANSWER_ALT[op] if variant == "output" else RESULT_UNIT[op]
                fid, template = self.pick("calc." + op, "{au}" if variant == "output" else None)
                english = is_english(template)
                values = {f: sample(rng, units[f]) for f in FIELDS[op]}
                steps, result = solve(op, values, units, answer)
                fields = {f + "_q": quantity(rng, values[f], units[f], english) for f in FIELDS[op]}
                fields["au"] = unit_word(rng, answer, english)
                subtype = f"{op}:{variant}" + "".join(f":{u}" for f, u in units.items() if u != TOOL_UNIT[f])
            if .01 <= result["value"] <= 1e6 and self.claim(steps):
                break
        task = dict(parameter_key=json.dumps(steps, sort_keys=True), synthetic=True, required_calcs=steps, expected=result)
        if kind == "recovery":
            if rng.random() < .5:
                task["public"] = {"transient": True}
                subtype = "transient:" + subtype
            else:
                wrong, error, result = self.broken(steps[0])
                task["public"] = {"history": [
                    {"role": "assistant", "content": tool("marine_calculate", wrong, rng.choice(bank.HISTORY_PLANS))},
                    {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}]}
                subtype = f"history:{error}:" + subtype
        body, families = template.format(**fields), [fid]
        if self.contrast and rng.random() < .15:  # a record number nearby must not become a tool argument
            place = rng.choice(("id_lead", "id_tail"))
            cid_family, clause = self.pick(place)
            body = clause.format(rid=record_id(rng)) + body if place == "id_lead" else body + clause.format(rid=record_id(rng))
            families.append(cid_family)
            subtype += ":id"
        self.add(kind, subtype, body, families, True, **task)

    def broken(self, step):
        """A realistic wrong first call whose real tool result is an error."""
        rng, args = self.rng, dict(step)
        errors = ["wrong_operation", "missing_field", "extra_field"]
        if step["operation"] == "convert":
            errors += ["unit_word", "cross_dimension"]
        error = rng.choice(errors)
        if error == "wrong_operation":
            args["operation"] = rng.choice([op for op in FIELDS if op != step["operation"]])
        elif error == "missing_field":
            args.pop(rng.choice([k for k in args if k != "operation"]))
        elif error == "extra_field":
            args["unit"] = rng.choice(("km", "km/h", "h"))
        elif error == "unit_word":
            args["from_unit"] = rng.choice([w for w in bank.UNIT_SURFACES[args["from_unit"]] if not w.isascii()])
        else:
            args["to_unit"] = "km/h" if args["from_unit"] in UNITS["distance"] else "km"
        result = self.checker.execute({"name": "marine_calculate", "arguments": args})
        if "error" not in result:
            raise ValueError(f"History call unexpectedly succeeded: {args}")
        return args, error, result

    def card(self, cid, op, values, units, family=None, words=None, suffix=""):
        """Unit wording is fixed per card so conflicting records differ only in their numbers."""
        fid, fmt = family or self.pick("card." + "_".join(FIELDS[op]))
        if words is None:
            canonical, english = "=" in fmt, bool(re.search(r"[A-Za-z]{4,}", fmt))
            words = {}
            for f in FIELDS[op]:
                word = units[f] if canonical else unit_word(self.rng, units[f], english)
                words[f] = (" " if canonical or english or (word.isascii() and self.rng.random() < .5) else "") + word
        text = fmt.format(**{f + "_q": f"{values[f]:g}{words[f]}" for f in FIELDS[op]})
        self.docs.append({"id": cid + suffix, "title": f"模拟航次卡 {cid}", "excerpt": text, "source": "synthetic-v3"})
        return (fid, fmt), words

    def card_question(self, cid, op, category):
        rng = self.rng
        fid, template = self.pick(category)
        english = is_english(template)
        rid, ref = ("card_ref.en", rng.choice(bank.ENGLISH_CARD_REFS)) if english else self.pick("card_ref")
        body = template.format(card=ref.format(cid=cid), au=unit_word(rng, RESULT_UNIT[op], english))
        return body, [fid, rid]

    def chain(self, i):
        rng = self.rng
        op = ("distance", "speed", "time")[i % 3]
        options = {"distance": [("speed", u) for u in UNITS["speed"]],
                   "speed": [("distance", u) for u in UNITS["distance"]],
                   "time": [("distance", "km")] + [(f, u) for f in ("distance", "speed") for u in ALT_UNITS[f]]}[op]
        field, unit = options[(i // 3) % len(options)]
        units = dict({f: TOOL_UNIT[f] for f in FIELDS[op]}, **{field: unit})
        while True:
            values = {f: sample(rng, units[f]) for f in FIELDS[op]}
            steps, result = solve(op, values, units, RESULT_UNIT[op])
            if .01 <= result["value"] <= 1e6 and self.claim(steps):
                break
        cid = self.card_id(json.dumps(steps))
        (card_family, _), _ = self.card(cid, op, values, units)
        form = self.contrast and i % 4 == 3  # report/log wording: still compute after reading the card
        body, families = self.card_question(cid, op, ("chain_form." if form else "chain.") + op)
        self.add("chain", ("form:" if form else "") + f"{op}:{field}:{unit}", body, families + [card_family], True,
                 parameter_key=json.dumps(steps, sort_keys=True), synthetic=True, query=cid, required_docs=[cid],
                 required_calcs=steps, expected=result)

    def clarify(self, i):
        rng = self.rng
        cycle = ("missing_param", "missing_param") + (("missing_id",) if self.contrast else ()) + ("missing_card", "conflict")
        subtype, k = cycle[i % len(cycle)], i // len(cycle)
        op = ("distance", "speed", "time")[k % 3]
        if subtype == "missing_id":
            # A record number plus a stated missing field: ask for the field, do not search the number.
            op, missing = CLARIFY_PAIRS[k % len(CLARIFY_PAIRS)]
            given = next(f for f in FIELDS[op] if f != missing)
            while True:
                fid, template = self.pick("clarify_id")
                unit = TOOL_UNIT[given] if given == "hours" or rng.random() < .7 else rng.choice(ALT_UNITS[given])
                value, rid = sample(rng, unit), record_id(rng)
                if self.claim([["clarify_id", rid, op, missing, value, unit]]):
                    break
            word, answer = MISSING[missing]
            body = template.format(rid=rid, given_cn=FIELD_CN[given], missing_cn=FIELD_CN[missing], asked_cn=ASKED_CN[op],
                                   given_q=quantity(rng, value, unit))
            self.add("clarify", f"missing_id:{op}:{missing}", body, [fid], rng.random() < .5, answer=answer,
                     clarify_keywords=[word, "单位"], no_tools=True, required_calcs=[], synthetic=True)
        elif subtype == "missing_param":
            op, missing = CLARIFY_PAIRS[k % len(CLARIFY_PAIRS)]
            given = next(f for f in FIELDS[op] if f != missing)
            while True:
                fid, template = self.pick(f"clarify.{op}.{missing}")
                unit = TOOL_UNIT[given] if given == "hours" or rng.random() < .7 else rng.choice(ALT_UNITS[given])
                value = sample(rng, unit)
                if self.claim([["clarify", op, missing, value, unit]]):
                    break
            word, answer = MISSING[missing]
            body = template.format(**{given + "_q": quantity(rng, value, unit, is_english(template))})
            self.add("clarify", f"missing:{op}:{missing}", body, [fid], rng.random() < .5, answer=answer,
                     clarify_keywords=[word, "单位"], no_tools=True, required_calcs=[], synthetic=True)
        elif subtype == "missing_card":
            cid = self.card_id(f"missing:{i}")  # reserved but never written to the corpus
            body, families = self.card_question(cid, op, "lookup_missing" if rng.random() < .3 else "chain." + op)
            answer, keywords = MISSING_CARD
            self.add("clarify", "missing_card", body, families, rng.random() < .5, query=cid, require_empty_search=True,
                     answer=answer, clarify_keywords=keywords, required_calcs=[], synthetic=True)
        else:
            field = FIELDS[op][(k // 3) % 2]
            units = {f: TOOL_UNIT[f] for f in FIELDS[op]}
            values = {f: sample(rng, units[f]) for f in FIELDS[op]}
            other = dict(values, **{field: round(values[field] + rng.choice(DELTAS[field]), 1)})
            cid = self.card_id(f"conflict:{i}")
            family, words = self.card(cid, op, values, units, suffix="a")
            self.card(cid, op, other, units, family=family, words=words, suffix="b")
            body, families = self.card_question(cid, op, "conflict_hint" if rng.random() < .4 else "chain." + op)
            name = FIELD_CN[field]
            extra = {}
            if self.contrast:  # copy both records side by side per field, then name the field that differs
                extra["compare_plan"] = "；".join(f"{FIELD_CN[f]}：{values[f]:g}{words[f]}，{other[f]:g}{words[f]}"
                                                 for f in FIELDS[op]) + "。"
            self.add("clarify", f"conflict:{op}:{field}", body, families + [family[0]], rng.random() < .5, query=cid,
                     required_docs=[cid + "a", cid + "b"], answer=f"资料中的{name}存在冲突，请确认应采用哪条记录。",
                     clarify_keywords=[name, "冲突", "确认"], required_calcs=[], synthetic=True, **extra)

    def card_quote(self, i):
        """Same kind of card, but the request is its original text: search and quote, no calculation."""
        rng = self.rng
        op = ("distance", "speed", "time")[i % 3]
        units = {f: TOOL_UNIT[f] for f in FIELDS[op]}
        alternative = next(f for f in FIELDS[op] if f != "hours")
        if rng.random() < .5:
            units[alternative] = rng.choice(ALT_UNITS[alternative])
        values = {f: sample(rng, units[f]) for f in FIELDS[op]}
        cid = self.card_id(f"quote:{i}")
        (card_family, _), _ = self.card(cid, op, values, units)
        body, families = self.card_question(cid, op, "card_quote")
        self.add("retrieve", "card_quote", body, families + [card_family], False, answer=self.docs[-1]["excerpt"],
                 required_calcs=[], query=cid, doc_id=cid, required_docs=[cid], synthetic=True)

    def direct_delimited(self, doc):
        fid, template = self.pick("direct.delimited")
        self.add("direct", "delimited", template.format(e=doc["excerpt"]), [fid], False, source_id=doc["source_id"],
                 answer=doc["excerpt"], required_calcs=[])

    def retrieve(self, doc):
        fid, template = self.pick("retrieve")
        self.docs.append(doc)
        self.add("retrieve", "retrieve", template.format(q=doc["title"]), [fid], False, source_id=doc["source_id"],
                 answer=doc["excerpt"], required_calcs=[], query=doc["title"], doc_id=doc["id"], required_docs=[doc["id"]])

    def direct(self, doc):
        fid, template = self.pick("direct")
        self.add("direct", "direct", template.format(e=doc["excerpt"]), [fid], False, source_id=doc["source_id"],
                 answer=doc["excerpt"], required_calcs=[])


def select_documents(source, split, count, excluded):
    """Same eligibility rule as ReAct v2; sources used by earlier benchmarks are excluded."""
    rows = sorted(read_jsonl(source / f"ocean_sft_{split}.jsonl"), key=lambda r: stable_score(r["conversations"][1]["content"], SEED))
    selected, filtered = [], Counter()
    for row in rows:
        q, a = [m["content"] for m in row["conversations"] if m["role"] != "system"]
        sid = digest(q)
        excerpt = next((s.strip() for s in re.split(r"(?<=[。！？])\s*|\n", a)
                        if 20 <= len(s.strip()) <= 65 and len(re.findall(r"[\u4e00-\u9fff]", s)) >= 12
                        and s.strip().endswith(("。", "！", "？"))), "")
        if sid in excluded:
            filtered["excluded_or_duplicate_source"] += 1
            continue
        if not excerpt or not 3 <= len(q) <= 90 or any(c in q + excerpt for c in ("<", ">", "\ufffd")):
            filtered["ineligible_excerpt"] += 1
            continue
        excluded.add(sid)
        selected.append({"id": "O" + sid[:10], "source_id": sid, "title": q, "excerpt": excerpt, "source": "OceanInstruct-v0.2"})
        if len(selected) == count:
            return selected, filtered
    raise ValueError(f"Insufficient eligible source documents in {split}")


def search_plan(query, style):
    if style == "fixed":
        return FIXED_PLAN["search"]
    return f"查找航次卡{query}。" if CARD_ID.fullmatch(query) else "检索该问题的资料原文。"


def calc_plan(step, style):
    if style == "fixed":
        return FIXED_PLAN["calc"]
    n = {k: number(v) for k, v in step.items() if k in ("value", "speed", "hours", "distance")}
    return {"convert": lambda: f"将{n.get('value')} {step.get('from_unit')}换算为{step.get('to_unit')}。",
            "distance": lambda: f"用航速{n['speed']} km/h和时长{n['hours']} h计算航程。",
            "speed": lambda: f"用航程{n['distance']} km和时长{n['hours']} h计算航速。",
            "time": lambda: f"用航程{n['distance']} km和航速{n['speed']} km/h计算时长。"}[step["operation"]]()


def oracle(task, style):
    """Offline teacher only; the inference harness never sees it."""
    texts, legacy = [], task.get("version") != VERSION
    query = task["doc_id"] if legacy and task["kind"] == "chain" else task.get("query")
    if query:
        texts.append(tool("search_ocean", {"query": query}, search_plan(query, style)))
    steps = [task["calc"]] if legacy and task.get("calc") else task.get("required_calcs", [])
    for n, step in enumerate(steps):
        texts.append(tool("marine_calculate", step, calc_plan(step, style)))
        if n == 0 and task.get("public", {}).get("transient"):
            texts.append(tool("marine_calculate", step, RETRY_PLAN))
    if task["kind"] == "clarify":
        plan = f"<plan>{task['compare_plan']}</plan>" if task.get("compare_plan") else ""
        texts.append(f"{plan}<clarify>{task['answer']}</clarify>")
    elif "expected" in task:
        texts.append(f"<final>{number(task['expected']['value'])} {task['expected']['unit']}</final>")
    else:
        texts.append(f"<final>{task['answer']}" + (f"[{task['doc_id']}]" if task.get("doc_id") else "") + "</final>")
    return texts


def demonstrate(tasks, environment, tokenizer, style):
    demos, lengths = [], []
    for task in tasks:
        texts = iter(oracle(task, style))

        def generate(ids, limit):
            text = next(texts, None)
            if text is None:
                raise ValueError(f"Harness asked for more rounds than the teacher has: {task['id']}")
            output = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(output) > limit:
                raise ValueError(f"Overlong target: {task['id']}")
            return output, []
        trace = run_trajectory(task, tokenizer, environment, generate)
        if trace["metrics"]["success"] != 1 or next(texts, None) is not None:
            raise ValueError(f"Teacher failed real execution: {task['id']} {json.dumps(trace['calls'], ensure_ascii=False)}")
        messages = trace["messages"]
        if task["kind"] == "retrieve" and not any(
                m["role"] == "tool" and any(task["answer"] in d["excerpt"] for d in json.loads(m["content"]).get("documents", []))
                for m in messages):
            raise ValueError(f"Target evidence not visible to the model: {task['id']}")
        messages[0]["tools"] = json.dumps(TOOLS, ensure_ascii=False)
        demos.append({"conversations": messages, "supervision_start": 2 + len(task.get("public", {}).get("history", []))})
        length = len(tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True))
        if length > 2048:
            raise ValueError("Overlong demonstration; no silent truncation")
        lengths.append(length)
    return demos, lengths


def statistics(tasks, demos):
    first, operations, conversions = Counter(), Counter(), Counter()
    for demo in demos:
        supervised = [m["content"] for i, m in enumerate(demo["conversations"])
                      if m["role"] == "assistant" and i >= demo["supervision_start"]]
        head = supervised[0]
        first["search_ocean" if "search_ocean" in head else "marine_calculate" if "marine_calculate" in head else
              "clarify" if head.startswith("<clarify>") else "final"] += 1
        for text in supervised:
            match = re.search(r"<tool_call>(.*)</tool_call>", text, re.S)
            if match and "marine_calculate" in text:
                args = json.loads(match[1])["arguments"]
                operations[args["operation"]] += 1
                if args["operation"] == "convert":
                    conversions[f"{args['from_unit']}->{args['to_unit']}"] += 1
    return {"counts": dict(Counter(t["kind"] for t in tasks)),
            "subtypes": dict(sorted(Counter(t["kind"] + "/" + t.get("subtype", t.get("replay", "")).split(":")[0]
                                            for t in tasks).items())),
            "replayed_v1": sum(t.get("replay") == "ocean-agent-v1" for t in tasks),
            "first_supervised_action": dict(first), "calculator_operations": dict(operations),
            "conversions": dict(sorted(conversions.items())),
            "phrasing_families_used": len({f for t in tasks for f in t.get("families", [])})}


def held_out(legacy, previous):
    """Source documents, ids and calculations used by earlier benchmarks never enter v3."""
    sources, ids, calcs = set(), set(), set()
    benchmarks = [(legacy, ("train", "val", "test"))] + [(directory, ("val", "test")) for directory in previous]
    for directory, splits in benchmarks:
        for split in splits:
            for doc in read_jsonl(directory / f"ocean_agent_corpus_{split}.jsonl"):
                ids.add(doc["id"])
                sources.add(doc.get("source_id"))
    for directory, splits in benchmarks:
        for split in splits:
            for task in read_jsonl(directory / f"ocean_agent_{split}.jsonl"):
                sources.add(task.get("source_id"))  # direct-answer sources live only in task files
                if split != "train":
                    for calc in task.get("required_calcs", []) + ([task["calc"]] if task.get("calc") else []):
                        calcs.add(json.dumps(calc, sort_keys=True))
    return sources, ids, calcs


def replay(legacy):
    tasks = read_jsonl(legacy / "ocean_agent_train.jsonl")
    picked = []
    for kind, count in V1_REPLAY.items():
        # The ReAct v2 <final> tag cannot carry '<' or '>', so such v1 answers are not expressible.
        pool = sorted((t for t in tasks if t["kind"] == kind and not {"<", ">"} & set(t.get("answer", ""))),
                      key=lambda t: stable_score(t["id"], SEED))
        picked += [dict(t, replay="ocean-agent-v1") for t in pool[:count]]
    needed = {t["doc_id"] for t in picked if t.get("doc_id")}
    return picked, [d for d in read_jsonl(legacy / "ocean_agent_corpus_train.jsonl") if d["id"] in needed]


def prepare(source, previous, output, style, recipe="react-v3-diverse"):
    if (output / "agent_manifest.json").exists():
        raise FileExistsError("Use a new output directory; generated data is immutable")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    rng = random.Random(SEED)
    families = split_bank()
    excluded, known_ids, forbidden = held_out(source, previous)
    replay_tasks, replay_docs = replay(source)
    keys, used_questions = set(), set()
    manifest = {"version": VERSION, "recipe": recipe, "plan_style": style, "seed": SEED,
                "source": "zjunlp/OceanInstruct-v0.2", "source_license": "MIT",
                "warning": "OceanInstruct may contain synthetic/errors. Voyage cards, outages and tasks are synthetic; "
                           "not real navigation advice.",
                "tool_sha256": file_hash(ROOT / "agent/ocean.py"), "harness_sha256": file_hash(ROOT / "agent/react.py"),
                "generator_sha256": file_hash(Path(__file__)),
                "template_bank_sha256": file_hash(ROOT / "scripts/ocean_react_templates.py"),
                "held_out_benchmarks": [str(d.relative_to(ROOT)) for d in previous] + [str(source.relative_to(ROOT))],
                "leakage": leakage_report(families), "splits": {}}
    for split, sizes in RECIPES[recipe]["sizes"].items():
        builder = Builder(split, families, rng, keys, forbidden, known_ids, recipe)
        for kind in ("calculate", "recovery"):
            for i in range(sizes[kind]):
                builder.calculation(i, kind)
        for i in range(sizes["chain"]):
            builder.chain(i)
        for i in range(sizes["clarify"]):
            builder.clarify(i)
        retrieve, direct = sizes["retrieve"], sizes["direct"]
        documents, filtered = select_documents(source, split, retrieve + direct + sizes.get("direct_delimited", 0), excluded)
        for doc in documents[:retrieve]:
            builder.retrieve(doc)
        for doc in documents[retrieve:retrieve + direct]:
            builder.direct(doc)
        for doc in documents[retrieve + direct:]:
            builder.direct_delimited(doc)
        for i in range(sizes.get("card_quote", 0)):
            builder.card_quote(i)
        tasks, docs = builder.tasks, builder.docs
        for task in tasks:
            fingerprint = normalized(task["question"])
            if fingerprint in used_questions:
                raise ValueError(f"duplicate prompt: {task['question']}")
            used_questions.add(fingerprint)
            task.update(version=VERSION, recipe=recipe, template_family=split + ":" + "|".join(task["families"]),
                        id=split + "-" + digest(task["question"])[:16])
        if split == "train":
            tasks, docs = tasks + replay_tasks, docs + replay_docs
        if len({d["id"] for d in docs}) != len(docs):
            raise ValueError("duplicate document id")
        tasks.sort(key=lambda t: stable_score(t["question"] + t["id"], SEED))
        demos, lengths = demonstrate(tasks, ReactTools(docs), tokenizer, style)
        outputs = {f"ocean_agent_{split}.jsonl": tasks, f"ocean_agent_sft_{split}.jsonl": demos,
                   f"ocean_agent_corpus_{split}.jsonl": docs}
        for name, rows in outputs.items():
            atomic_jsonl(output / name, rows)
        manifest["splits"][split] = {**statistics(tasks, demos), "filtered_sources": dict(filtered),
                                    "source_sha256": file_hash(source / f"ocean_sft_{split}.jsonl"),
                                    "artifacts": {name: file_hash(output / name) for name in outputs},
                                    "tokens": {"p50": percentile(lengths, .5), "p95": percentile(lengths, .95),
                                               "max": max(lengths), "truncation_rate": 0}}
        print(split, json.dumps(manifest["splits"][split]["counts"], ensure_ascii=False), flush=True)
    from trainer.train_agent import atomic_json
    atomic_json(output / "agent_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed",
                        help="OceanInstruct splits and the original v1 agent benchmark")
    parser.add_argument("--previous", type=Path, nargs="+", default=[ROOT / "data/processed/react-v2"],
                        help="earlier ReAct data whose val/test stay held out (react-v3.1 also lists react-v3)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/react-v3")
    parser.add_argument("--plan_style", choices=["fixed", "slots"], default="fixed",
                        help="fixed: ReAct v2 plan sentences; slots: plans restate the numbers being used")
    parser.add_argument("--recipe", choices=list(RECIPES), default="react-v3-diverse")
    args = parser.parse_args()
    prepare(args.source.resolve(), [p.resolve() for p in args.previous], args.output.resolve(), args.plan_style, args.recipe)
