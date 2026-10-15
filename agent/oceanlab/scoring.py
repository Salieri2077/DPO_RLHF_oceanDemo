"""Rule scorer for OceanDepths analysis episodes (M1b). Hidden from the agent: nothing here, including the metrics it
returns, may enter an episode's prompt, a memory, or Reflexion feedback at test time.

Scientific success S = 1 requires all of:
1. a `<final>` JSON object with a valid status and a result handle for every requested analysis;
2. each delivered handle exists in this episode's registry, was made by the right tool, and its normalised computation
   matches the request: region, variable, method card (QC policy qc_policy_v1; nearest_or_exact for consistency) and
   the actual depth-level set; the groups of the handles delivered for an analysis (one handle, a list, or a
   {label: handle} map, e.g. one per period from separate queries) together equal the requested group date coverages
   exactly (labels and order are ignored);
3. the declared status equals the reference status. Insufficient support is judged on valid support (estimable
   profiles; pairs per group x level), and a refusal counts only with the formal handles that show zero support, so
   refusing without querying fails, and query errors, partition refusals or budget exhaustion are never evidence.

Diagnostic handles (finite_only, temporal_filter=any) may be computed along the way; delivering one fails item 2.

Environment self-check (kept apart from S): every result handle of the episode is recomputed with the independent
`reference` implementation under that handle's own parameters. A disagreement, or a tool's environment_error, is an
environment error: it must be fixed and the affected tasks rerun. It never turns a model failure into a pass or an
exclusion, and a model's wrong choice (QC, variable, filter, groups) is a model failure, not an environment error.

Optimisation reward r = S - 0.1 * I(any protocol violation): an unparsable action or unknown tool, a tool call that
violates the argument schema, or a `<final>` that is not a valid JSON object of the contract.
"""
import json
import math
from collections import Counter

from agent.oceanlab import reference
from agent.oceanlab.data import QC_POLICY
from agent.oceanlab.tasks import depth_levels, group_dates

STATUSES = ("complete", "partial", "insufficient")
TOOL_OF = {"mld": "compute_mld", "consistency": "compute_consistency"}
BUDGET_STOPS = {"context_limit", "call_limit", "turn_limit", "generation_limit", "empty_generation", "repeated_action"}
TOL = 1e-9


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def same_quartiles(a, b):
    if a is None or b is None:
        return a is None and b is None
    return all(close(a[q], b[q]) for q in ("q25", "median", "q75"))


def expected_params(spec, depth):
    (lat0, lat1), (lon0, lon1) = spec["region"]
    common = {"region": dict(lat_min=lat0, lat_max=lat1, lon_min=lon0, lon_max=lon1),
              "coverage": sorted(group_dates(g) for g in spec["groups"])}
    out = {}
    if "mld" in spec["analyses"]:
        out["mld"] = {**common, "variable": "potential_temperature", "qc_policy": QC_POLICY}
    if "consistency" in spec["analyses"]:
        c = spec["consistency"]
        out["consistency"] = {**common, "variable": c["variable"], "qc_policy": QC_POLICY,
                              "temporal_filter": "nearest_or_exact", "levels": depth_levels(depth, *c["depth_range_m"])}
    return out


def parameter_errors(record, want, records=None):
    """Names of the computation parameters that differ from the request (normalised semantics).

    `records` (all handles delivered for one analysis, default just `record`): an analysis may be delivered as several
    handles, e.g. one per period computed from separate queries; every handle must match the request on everything
    but groups, and their groups together must cover the requested groups exactly (no gap, no repeat)."""
    records = records or [record]
    errors = []
    if any(not close(r["params"]["region"][k], want["region"][k]) for r in records for k in want["region"]):
        errors.append("region")
    if sorted((g["start_date"], g["end_date"]) for r in records for g in r["params"]["groups"]) != want["coverage"]:
        errors.append("groups")
    for record in records:
        for error in _method_errors(record["params"], want):
            if error not in errors:
                errors.append(error)
    return errors


def _method_errors(p, want):
    errors = []
    for key in ("variable", "qc_policy", "temporal_filter"):
        if key in want and p.get(key) != want[key]:
            errors.append(key)
    if "levels" in want and sorted(p["levels"]) != sorted(want["levels"]):
        errors.append("levels")
    return errors


def delivered_handles(value):
    """A result entry of <final>: one handle, a list of handles, or {label: handle}. -> list of handles or None."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value
    if isinstance(value, dict) and value and all(isinstance(v, str) for v in value.values()):
        return list(value.values())
    return None


def compare_groups(tool, stored, ref):
    """Mismatches between a handle's stored group results and the reference under the same parameters."""
    problems = []
    for got, want in zip(stored, ref):
        where = f"{got['start_date']}..{got['end_date']}"
        for key in ("profiles", "profile_set_sha256"):
            if got[key] != want[key]:
                problems.append(f"{where}: {key}")
        if tool == "compute_mld":
            if got["counts"] != want["counts"]:
                problems.append(f"{where}: counts")
            for key in ("mld_m", "interval_width_m"):
                if not same_quartiles(got[key], want[key]):
                    problems.append(f"{where}: {key}")
        else:
            for key in ("excluded_temporal_status", "profiles_rejected_qc"):
                if got[key] != want[key]:
                    problems.append(f"{where}: {key}")
            for a, b in zip(got["depths"], want["depths"]):
                if a["level"] != b["level"] or a["n"] != b["n"] or not all(close(a[k], b[k]) for k in ("bias", "mae", "rmse")):
                    problems.append(f"{where}: level {a['level']}")
            if len(got["depths"]) != len(want["depths"]):
                problems.append(f"{where}: level count")
    if len(stored) != len(ref):
        problems.append("group count")
    return problems


def parse_final(text):
    """-> (deliverable dict or None, format error or None)."""
    try:
        final = json.loads(text)
    except (ValueError, TypeError):
        return None, "final is not JSON"
    if not isinstance(final, dict) or final.get("status") not in STATUSES:
        return None, "final needs an object with status complete/partial/insufficient"
    results = final.get("results")
    if not isinstance(results, dict) or any(delivered_handles(v) is None for v in results.values()):
        return None, "final needs results: {analysis: handle or handles}"
    return final, None


class Scorer:
    def __init__(self, data):
        self.data = data
        self.cache = {}

    def reference_for(self, tool, params):
        key = json.dumps([tool, {k: v for k, v in params.items() if k not in ("dataset", "depth_min", "depth_max")}],
                         sort_keys=True)
        if key not in self.cache:
            self.cache[key] = reference.handle_reference(self.data, tool, params)
        return self.cache[key]

    def self_check(self, trace):
        """Environment errors of an episode: tool crashes and handles that disagree with the reference."""
        problems = [c["result"]["error"] for c in trace["calls"] if c["result"].get("error_kind") == "environment"]
        for handle, record in trace["registry"]["results"].items():
            ref = self.reference_for(record["tool"], record["params"])
            problems += [f"{handle} {p}" for p in compare_groups(record["tool"], record["groups"], ref)]
        return problems

    def score(self, task, trace, env_problems=None):
        """env_problems: reuse an earlier self-check of the same trace (rescoring after a delivery-rule change)."""
        reference_status = task["reference"]["status"]
        protocol = [r for r in trace["rounds"] if (r.get("action") or {}).get("type") == "invalid"]
        protocol += [c for c in trace["calls"] if c["result"].get("error_kind") == "schema"]
        env_problems = self.self_check(trace) if env_problems is None else env_problems
        registry = trace["registry"]["results"]
        failure, params, declared, final_error = None, {}, None, None
        if trace["stop"] != "final":
            failure = "budget" if trace["stop"] in BUDGET_STOPS else "no_final"
        else:
            final, final_error = parse_final(trace["final"])
            if final_error:
                failure = "final_format"
            else:
                declared = final["status"]
                want = expected_params(task, self.data.depth)
                for analysis, target in want.items():
                    handles = delivered_handles(final["results"].get(analysis))
                    if handles is None:
                        failure = failure or "missing_analysis"
                    elif any(h not in registry for h in handles):
                        failure = failure or "unknown_handle"
                    elif any(registry[h]["tool"] != TOOL_OF[analysis] for h in handles):
                        failure = failure or "wrong_tool"
                    else:
                        errors = parameter_errors(None, target, [registry[h] for h in handles])
                        if errors:
                            params[analysis] = errors
                            failure = failure or "parameters"
                if not failure and declared != reference_status:
                    failure = "status"
        success = float(failure is None)
        violation = bool(protocol) or final_error is not None
        computable = reference_status != "insufficient"
        rounds = trace["rounds"]
        return {
            "success": success, "reward": success - 0.1 * violation, "failure": failure,
            "parameter_errors": params, "declared_status": declared, "reference_status": reference_status,
            "computable": float(computable),
            "false_refusal": float(computable and declared == "insufficient"),
            "refusal_correct": float(not computable and success == 1.0),
            "protocol_violation": float(violation), "protocol_errors": len(protocol) + (final_error is not None),
            "env_error": float(bool(env_problems)), "env_problems": env_problems[:20],
            "stop": trace["stop"], "turns": len(rounds), "calls": len(trace["calls"]),
            "tool_errors": sum("error" in c["result"] for c in trace["calls"]),
            "diagnostic_handles": sum(1 for r in registry.values() if not r["method_card"]),
            "input_tokens": sum(len(r["input_ids"]) if "input_ids" in r else r["input_tokens"] for r in rounds),
            "generated_tokens": sum(len(r["completion_ids"]) for r in rounds),
        }


def aggregate(traces):
    """Summary over episodes. Raises if any episode had an environment error: fix the environment and rerun first."""
    bad = [t["id"] for t in traces if t["metrics"]["env_error"]]
    if bad:
        raise RuntimeError(f"environment errors in {len(bad)} episodes (e.g. {bad[:5]}); fix and rerun them")
    n = max(len(traces), 1)
    m = [t["metrics"] for t in traces]
    computable = [x for x in m if x["computable"]]
    refusals = [x for x in m if not x["computable"]]
    out = {"episodes": len(traces), "success": sum(x["success"] for x in m) / n,
           "reward": sum(x["reward"] for x in m) / n,
           "computable_success": sum(x["success"] for x in computable) / max(len(computable), 1),
           "computable_episodes": len(computable),
           "refusal_correct": sum(x["refusal_correct"] for x in refusals) / max(len(refusals), 1),
           "insufficient_episodes": len(refusals),
           "false_refusal_rate": sum(x["false_refusal"] for x in computable) / max(len(computable), 1),
           "protocol_violation_rate": sum(x["protocol_violation"] for x in m) / n,
           "failures": dict(Counter(x["failure"] or "success" for x in m)),
           "parameter_errors": dict(Counter(e for x in m for errs in x["parameter_errors"].values() for e in errs)),
           "stops": dict(Counter(x["stop"] for x in m))}
    for key in ("turns", "calls", "tool_errors", "input_tokens", "generated_tokens", "diagnostic_handles"):
        out[key] = sum(x[key] for x in m) / n
    families = sorted({t["kind"] for t in traces})
    out["success_by_family"] = {f: sum(t["metrics"]["success"] for t in traces if t["kind"] == f)
                                / sum(1 for t in traces if t["kind"] == f) for f in families}
    return out
