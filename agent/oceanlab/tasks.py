"""Development requests for the OceanDepths analysis agent (M1), with scorer-side reference results.

Sampling is declared before any model sees the data and never uses model outcomes:
- candidates: 10x10 degree blocks wholly inside the split's partition (outside the reserved region and its 1-degree
  buffer) and calendar months wholly inside the split's period after the 30-day year-boundary buffers (dev
  2017-02..2019-11, train 2000-01..2016-11);
- each (family, density tertile) cell draws blocks uniformly from blocks of that tertile, where density is the
  block's dev-period profile count among blocks with at least one profile, plus a share of blocks without profiles in
  the requested months arising naturally from the draw; periods and phrasings are drawn uniformly per family;
- every reference result is recomputed with the tool-side implementation and must agree before it is kept.
"""
import json
import random

import numpy as np
import pandas as pd

from agent.oceanlab import methods, reference
from agent.oceanlab.data import (BLOCK_DEG, QC_POLICY, RESERVED_REGION, REVISION, SPATIAL_BUFFER_DEG, OceanDepths,
                                 partition_of, profile_set_digest, split_dates)

DEV_MONTHS = pd.period_range("2017-02", "2019-11", freq="M")
TASK_VERSION = "oceanlab_dev_v1"
FAMILY_PLAN = {"mld": 18, "consistency_temperature": 10, "consistency_salinity": 8, "report": 12}
TRAIN_VERSION = "oceanlab_train_v1"
TRAIN_PLAN = {"mld": 375, "consistency_temperature": 210, "consistency_salinity": 165, "report": 250}  # x20.8 of dev
REANALYSIS = {"potential_temperature": "glorys_thetao", "salinity": "glorys_so"}
VARIABLE_ZH = {"potential_temperature": "势温", "salinity": "盐度"}


def block_box(block):
    lat0, lon0 = block
    return (lat0, lat0 + BLOCK_DEG), (lon0, lon0 + BLOCK_DEG)


def split_months(split):
    """Calendar months wholly inside a split's usable dates (after the year-boundary buffers)."""
    first, last = split_dates(split)
    start = first.to_period("M") if first.day == 1 else first.to_period("M") + 1
    end = last.to_period("M") if last == last.to_period("M").end_time.normalize() else last.to_period("M") - 1
    return pd.period_range(start, end, freq="M")


def dev_blocks(index, split="dev"):
    """Profile counts in a split of 10-degree blocks that do not touch the reserved region or its buffer."""
    (la0, la1), (lo0, lo1) = RESERVED_REGION["lat"], RESERVED_REGION["lon"]
    b = SPATIAL_BUFFER_DEG
    dev = index[partition_of(index["time"], index["latitude"], index["longitude"]) == split]
    # Fill values (latitude -999, etc.) carry a split label too; only valid coordinates define candidate blocks.
    dev = dev[dev["latitude"].between(-90, 90, inclusive="left") & dev["longitude"].between(-180, 180, inclusive="left")]
    lat0 = (np.floor(dev["latitude"] / BLOCK_DEG) * BLOCK_DEG).astype(int)
    lon0 = (np.floor(dev["longitude"] / BLOCK_DEG) * BLOCK_DEG).astype(int)
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([lat0, lon0])).groupby(level=[0, 1]).sum()
    keep = {}
    for (a, o), n in counts.items():
        touches = a < la1 + b and a + BLOCK_DEG > la0 - b and o < lo1 + b and o + BLOCK_DEG > lo0 - b
        if not touches:
            keep[(int(a), int(o))] = int(n)
    return keep


def depth_levels(depth, top, bottom):
    return [int(k) for k in np.flatnonzero((depth >= top - 1e-9) & (depth <= bottom + 1e-9))]


def place(box):
    (la0, la1), (lo0, lo1) = box
    ns = lambda v: f"{'北纬' if v >= 0 else '南纬'}{abs(v):g}°"
    ew = lambda v: f"{'东经' if v >= 0 else '西经'}{abs(v):g}°"
    return f"{ns(la0)}—{ns(la1)}、{ew(lo0)}—{ew(lo1)}"


def month_zh(period):
    return f"{period.year}年{period.month}月"


def periods_zh(groups):
    return "、".join(g["label_zh"] for g in groups)


def make_groups(rng, kind, DEV_MONTHS=DEV_MONTHS):
    """Calendar groups inside a split's months: single month, a three-month season, or two months to compare."""
    if kind == "season":
        start = rng.randrange(len(DEV_MONTHS) - 2)
        months = DEV_MONTHS[start:start + 3]
        return [{"label": f"{months[0]}..{months[-1]}", "label_zh": f"{month_zh(months[0])}至{month_zh(months[-1])}",
                 "months": [str(m) for m in months]}]
    if kind == "two_months":
        a, b = sorted(rng.sample(range(len(DEV_MONTHS)), 2))
        return [{"label": str(DEV_MONTHS[i]), "label_zh": month_zh(DEV_MONTHS[i]), "months": [str(DEV_MONTHS[i])]}
                for i in (a, b)]
    if kind == "monthly_season":
        start = rng.randrange(len(DEV_MONTHS) - 2)
        return [{"label": str(m), "label_zh": month_zh(m), "months": [str(m)]} for m in DEV_MONTHS[start:start + 3]]
    i = rng.randrange(len(DEV_MONTHS))
    return [{"label": str(DEV_MONTHS[i]), "label_zh": month_zh(DEV_MONTHS[i]), "months": [str(DEV_MONTHS[i])]}]


def render(spec, rng):
    """Chinese request text. Every field the scorer checks is stated explicitly; wording varies only in form."""
    where, when = place(spec["region"]), periods_zh(spec["groups"])
    lines = []
    if "mld" in spec["analyses"]:
        lines.append(rng.choice([
            f"请统计{where}海域{when}的温度阈值混合层深度，按{'每个时段' if len(spec['groups']) > 1 else '该时段'}给出可估计剖面数、中位数和四分位数，并说明因缺测、深度不足或质量不合格而无法给出精确值的剖面数量。",
            f"用温度阈值法（10m参考、0.2℃）诊断{where}在{when}的混合层深度；分别报告各时段的中位数、四分位数和样本数，以及截尾、支持中断、无参考和被排除的剖面数。",
            f"我需要{where}海域{when}混合层深度的统计（温度阈值法）。请先逐剖面计算再汇总，并交代哪些剖面不能给出精确深度及原因。"]))
    if "consistency" in spec["analyses"]:
        c = spec["consistency"]
        top, bottom = c["depth_range_m"]
        zh = VARIABLE_ZH[c["variable"]]
        depth_text = f"{top:g}—{bottom:g}米" if c["shallow_requested"] else f"上层{bottom:g}米"
        lines.append(rng.choice([
            f"在{where}海域{when}，比较EN4与GLORYS的{zh}（再分析减观测），对{depth_text}的各标准层报告bias、MAE、RMSE和配对数n。",
            f"请核验GLORYS再分析与EN4观测在{where}、{when}的{zh}一致性：{depth_text}内逐层给出偏差、平均绝对误差、均方根误差和有效配对数，并说明样本覆盖是否支持{'不同时段之间的比较' if len(spec['groups']) > 1 else '结论'}。",
            f"对{where}海域{when}的{zh}做观测—再分析对比，深度范围为{depth_text}，按层输出 bias / MAE / RMSE / n。"]))
    return "".join(lines)


def group_dates(group):
    """Calendar coverage of a request group: first day of its first month to last day of its last month."""
    return (pd.Period(group["months"][0], "M").start_time.strftime("%Y-%m-%d"),
            pd.Period(group["months"][-1], "M").end_time.strftime("%Y-%m-%d"))


def reference_result(data, spec):
    """Scorer-side result for one request, cross-checked against the tool-side methods."""
    lat, lon = spec["region"]
    months = [m for g in spec["groups"] for m in g["months"]]
    start = pd.Period(months[0], "M").start_time.strftime("%Y-%m-%d")
    end = pd.Period(months[-1], "M").end_time.strftime("%Y-%m-%d")
    profiles = data.select(lat, lon, start, end)
    month_of = profiles.index["time"].dt.strftime("%Y-%m").to_numpy()
    to_group = {m: g["label"] for g in spec["groups"] for m in g["months"]}
    keep = np.array([m in to_group for m in month_of], bool)
    profiles = type(profiles)(profiles.rows[keep], profiles.index[keep])
    groups = np.array([to_group[m] for m in month_of[keep]], dtype=object)
    result = {"profiles": int(len(profiles.rows)), "profile_set_sha256": profile_set_digest(profiles)}
    agree = True
    if "mld" in spec["analyses"]:
        ref = reference.mld_reference(data, profiles, groups)
        theta, accepted, ok = data.levels(profiles, "potential_temperature")
        tool = methods.summarize_mld(methods.mld_temperature_v1(theta, accepted, data.depth, ok), groups)
        for g in tool:
            agree &= {k: v for k, v in tool[g]["counts"].items() if v} == ref[g]["counts"]
            for key in ("mld_m", "interval_width_m"):
                if (tool[g][key] is None) != (ref[g][key] is None):
                    agree = False
                elif tool[g][key]:
                    agree &= all(abs(tool[g][key][q] - ref[g][key][q]) < 1e-9 for q in ("q25", "median", "q75"))
        result["mld"] = {g["label"]: ref.get(g["label"], {"profiles": 0, "counts": {}, "mld_m": None,
                                                           "interval_width_m": None}) for g in spec["groups"]}
    if "consistency" in spec["analyses"]:
        c = spec["consistency"]
        levels = depth_levels(data.depth, *c["depth_range_m"])
        ref = reference.consistency_reference(data, profiles, c["variable"], groups, levels)
        value, accepted, _ = data.levels(profiles, c["variable"])
        reanalysis = data.read(REANALYSIS[c["variable"]], profiles.rows).astype(np.float64)
        tool = methods.consistency(value, accepted, reanalysis, profiles.index["glorys_temporal_status"].to_numpy(),
                                   groups, levels)
        for g in tool:
            for a, b in zip(tool[g]["depths"], ref[g]["depths"]):
                agree &= a["n"] == b["n"] and all(a[k] is None or abs(a[k] - b[k]) < 1e-9 for k in ("bias", "mae", "rmse"))
        empty = lambda: {"profiles": 0, "excluded_temporal_status": 0,
                         "depths": [{"level": k, "n": 0, "bias": None, "mae": None, "rmse": None} for k in levels]}
        result["consistency"] = {g["label"]: ref.get(g["label"], empty()) for g in spec["groups"]}
        for g in result["consistency"].values():
            for row in g["depths"]:
                row["depth_m"] = float(data.depth[row["level"]])
    result["status"] = request_status(result)
    return result, agree


def request_status(result):
    """complete: every requested group/cell has support; insufficient: none has; partial otherwise.

    Support is valid support (estimable profiles; pairs per group x level), not raw profiles. A result without any
    group or depth cell is malformed (the caller lost a requested analysis or group) and raises instead of passing
    as complete through all([]).
    """
    cells = []
    for g in result.get("mld", {}).values():
        cells.append(g["counts"].get("estimable", 0) > 0)
    for g in result.get("consistency", {}).values():
        if not g["depths"]:
            raise ValueError("consistency group without depth cells")
        cells += [row["n"] > 0 for row in g["depths"]]
    if not cells:
        raise ValueError("no requested group or depth cell to judge support on")
    if all(cells):
        return "complete"
    return "insufficient" if not any(cells) else "partial"


def build_dev_requests(seed=0, data=None):
    return build_requests("dev", FAMILY_PLAN, TASK_VERSION, seed, data)


def build_requests(split, plan, version, seed=0, data=None):
    """Requests of one split by the predeclared sampling; exact repeats (same block, groups and analyses) are redrawn."""
    data = data or OceanDepths()
    rng = random.Random(seed)
    months = split_months(split)
    blocks = dev_blocks(data.index, split)
    ordered = sorted(blocks, key=lambda b: blocks[b])
    tertiles = np.array_split(np.array(ordered, dtype=object), 3)
    requests, rejected, seen = [], [], set()
    width = 3 if split == "dev" else 4
    for family, count in plan.items():
        for i in range(count):
            tier = i % 3
            while True:
                block = tuple(rng.choice(list(tertiles[tier])))
                kind = {"mld": ["monthly_season", "two_months", "season"],
                        "consistency_temperature": ["single", "two_months", "season"],
                        "consistency_salinity": ["single", "two_months", "season"],
                        "report": ["single", "season"]}[family]
                groups = make_groups(rng, rng.choice(kind), months)
                key = (family, block, tuple(tuple(g["months"]) for g in groups))
                if key not in seen:
                    seen.add(key)
                    break
            box = block_box(block)
            spec = {"id": f"{split}-{len(requests) + len(rejected):0{width}d}", "task_version": version, "split": split,
                    "family": family, "density_tertile": ["low", "mid", "high"][tier], "block": list(block),
                    "region": [list(box[0]), list(box[1])], "groups": groups, "analyses": [],
                    "method_cards": [], "qc_policy": QC_POLICY, "dataset_revision": REVISION}
            if family in ("mld", "report"):
                spec["analyses"].append("mld")
                spec["method_cards"].append("mld_temperature_v1")
            if family != "mld":
                variable = "salinity" if family == "consistency_salinity" else "potential_temperature"
                shallow = rng.random() < 0.25
                spec["analyses"].append("consistency")
                spec["method_cards"].append("profile_consistency_v1")
                spec["consistency"] = {"variable": variable, "shallow_requested": shallow,
                                       "depth_range_m": [0.49 if shallow else 5.0,
                                                         float(rng.choice([50, 100, 200, 500]))]}
            spec["text"] = render(spec, rng)
            result, agree = reference_result(data, spec)
            spec["reference"] = result
            (requests if agree else rejected).append(spec)
    return requests, rejected


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
