"""Scripted correct trajectories from a request's structured spec.

Used to test the environment end to end through the SDK, and later as verified demonstrations (C0). It is not a
baseline: it reads the structured spec, not the request text. Its support status comes only from what the public
tools returned, never from the hidden reference.
"""
import json

from agent.oceanlab.tasks import VARIABLE_ZH, group_dates, place

PLAN_LIMIT = 160  # agent.react.parse_action: one plan sentence, at most 160 characters, no angle brackets


def checked_plan(plan):
    if len(plan) > PLAN_LIMIT or "<" in plan or ">" in plan:
        raise ValueError(f"plan breaks the protocol: {plan}")
    return plan


def tool_text(plan, name, arguments):
    call = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<plan>{checked_plan(plan)}</plan><tool_call>{call}</tool_call>"


def number(value):
    """Whole numbers as integers, so demonstrations write 40 and 50 rather than 40 and 50.0."""
    return int(value) if float(value).is_integer() else float(value)


def status_of(cells):
    if all(cells):
        return "complete"
    return "insufficient" if not any(cells) else "partial"


def oracle_texts(spec, lab):
    """-> list of assistant texts: query, the requested computations, then the structured final.

    Each plan states, in one sentence, how the request text maps to the call (hemisphere signs, groups, depth range)
    or how the support status follows from the tool outputs."""
    episode = lab.episode()
    texts = []

    def act(plan, name, arguments):
        texts.append(tool_text(plan, name, arguments))
        result = episode.execute({"name": name, "arguments": arguments})
        if "error" in result:
            raise ValueError(f"{spec['id']}: {name} failed: {result}")
        return result

    (lat0, lat1), (lon0, lon1) = [[number(v) for v in pair] for pair in spec["region"]]
    groups = [{"label": g["label"], "start_date": s, "end_date": e} for g in spec["groups"] for s, e in [group_dates(g)]]
    start, end = min(g["start_date"] for g in groups), max(g["end_date"] for g in groups)
    box = ((lat0, lat1), (lon0, lon1))
    dataset = act(f"区域{place(box)}→纬度{lat0:g}～{lat1:g}、经度{lon0:g}～{lon1:g}，时段{start}～{end}。", "query_profiles",
                  {"lat_min": lat0, "lat_max": lat1, "lon_min": lon0, "lon_max": lon1,
                   "start_date": start, "end_date": end})["dataset"]
    labels = [g.get("label_zh", g["label"]) for g in spec["groups"]]
    grouping = (f"{len(groups)}组：{'、'.join(labels)}" if len(groups) > 1 else f"一组：{labels[0]}")
    results, cells, evidence = {}, [], []
    if "mld" in spec["analyses"]:
        out = act(f"混合层用势温和qc_policy_v1，{grouping}。", "compute_mld",
                  {"dataset": dataset, "variable": "potential_temperature", "qc_policy": "qc_policy_v1",
                   "groups": groups})
        results["mld"] = out["result"]
        cells += [g["counts"]["estimable"] > 0 for g in out["groups"]]
        evidence.append("混合层estimable " + "、".join(str(g["counts"]["estimable"]) for g in out["groups"]))
    if "consistency" in spec["analyses"]:
        c = spec["consistency"]
        top, bottom = map(number, c["depth_range_m"])
        depth = (f"请求明确从{top:g} m起，深度{top:g}～{bottom:g} m" if c["shallow_requested"]
                 else f"上层{bottom:g}米即5.078～{bottom:g} m")
        out = act(f"{VARIABLE_ZH[c['variable']]}一致性：{depth}，nearest_or_exact，{grouping}。", "compute_consistency",
                  {"dataset": dataset, "variable": c["variable"], "qc_policy": "qc_policy_v1",
                   "temporal_filter": "nearest_or_exact", "depth_min": top, "depth_max": bottom, "groups": groups})
        results["consistency"] = out["result"]
        for g in out["groups"]:
            cells += [True] * g["levels_with_pairs"] + [False] * len(g["empty_levels_m"])
        evidence.append("一致性有配对层 " + "、".join(f"{g['levels_with_pairs']}/{out['levels']}" for g in out["groups"]))
    status = status_of(cells)
    plan = checked_plan(f"{'；'.join(evidence)}→{status}。")
    notes = {"complete": "所有要求的组和深度层都有有效支持。", "partial": "部分组或深度层没有有效支持，已交付可计算部分。",
             "insufficient": "所选资料中没有有效支持，结果句柄显示零支持。"}[status]
    final = {"status": status, "results": results, "notes": notes}
    texts.append(f"<plan>{plan}</plan><final>" + json.dumps(final, ensure_ascii=False) + "</final>")
    return texts
