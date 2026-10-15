"""Scorer-side reference for mld_temperature_v1 and profile_consistency_v1.

Written separately from `methods` on purpose: plain per-profile loops, its own qc_policy_v1 application from the raw
arrays, its own interpolation and quantile code. The two implementations must agree on real data before either is
trusted (tests/test_oceanlab.py); it shares only raw-array reading (OceanDepths.read) with the tools.

The diagnostic modes (qc_policy finite_only, temporal_filter any) are implemented here too, so the scorer can recheck
any result handle under that handle's own parameters (`handle_reference`); profile selection is also done separately,
on the raw JULD day numbers.
"""
import math

import numpy as np
import pandas as pd

from agent.oceanlab.data import JULD_EPOCH, VARIABLES, Profiles, profile_set_digest

MLD_STATUSES = ("estimable", "support_interrupted", "right_censored", "no_reference", "profile_excluded")


def accepted_levels(raw, variable_profile_qc, position_qc, juld_qc, profile_depth_qc, coords_ok, time_ok,
                    qc_policy="qc_policy_v1"):
    """QC for one profile: (profile accepted, list of booleans per level). raw: per-level lists for this profile."""
    if qc_policy == "finite_only":  # diagnostic: presence of a finite value only
        return True, [mask == 1 and math.isfinite(value) for value, mask in zip(raw["value"], raw["mask"])]
    if qc_policy != "qc_policy_v1":
        raise ValueError(qc_policy)
    optional_ok = juld_qc in (1, -1) and profile_depth_qc in (1, -1)
    profile_ok = bool(coords_ok and time_ok and position_qc == 1 and variable_profile_qc == 1 and optional_ok)
    levels = []
    for value, mask, qc, depth_qc in zip(raw["value"], raw["mask"], raw["level_qc"], raw["depth_qc"]):
        levels.append(profile_ok and mask == 1 and math.isfinite(value) and qc == 1 and depth_qc in (1, -1))
    return profile_ok, levels


def mld_one(theta, ok, depth, profile_ok, reference_depth=10.0, threshold=0.2):
    """One profile -> (status, mld, interval_top, interval_bottom, last_supported_depth)."""
    if not profile_ok:
        return "profile_excluded", None, None, None, None
    upper = next(i for i, d in enumerate(depth) if d > reference_depth)
    lower = upper - 1
    if not (ok[lower] and ok[upper]):
        return "no_reference", None, None, None, None
    theta10 = float(np.interp(reference_depth, [depth[lower], depth[upper]], [theta[lower], theta[upper]]))
    last_d, last_t = reference_depth, theta10
    for i in range(upper, len(depth)):
        if not ok[i]:
            deeper = any(ok[j] for j in range(i + 1, len(depth)))
            return ("support_interrupted" if deeper else "right_censored"), None, None, None, last_d
        if abs(theta[i] - theta10) >= threshold:
            goal = theta10 + (threshold if theta[i] > theta10 else -threshold)
            # Solve last_t + s * (theta_i - last_t) = goal for s in [0, 1]; the depth is linear in s.
            s = (goal - last_t) / (theta[i] - last_t)
            return "estimable", last_d + s * (depth[i] - last_d), last_d, depth[i], None
        last_d, last_t = depth[i], theta[i]
    return "right_censored", None, None, None, last_d


def quantile7(values, p):
    """Hyndman-Fan type 7 quantile (numpy 'linear'), written out for independence."""
    xs = sorted(values)
    h = (len(xs) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def load_profiles(data, profiles, variable, with_reanalysis=False):
    """Raw per-profile fields for the reference loops."""
    spec, rows = VARIABLES[variable], profiles.rows
    fields = {name: data.read(array, rows) for name, array in
              (("value", spec["value"]), ("mask", spec["mask"]), ("level_qc", spec["level_qc"]),
               ("depth_qc", "argo_depth_qc_on_glorys_depth"), ("profile_qc", spec["profile_qc"]),
               ("position_qc", "argo_position_qc"), ("juld_qc", "argo_juld_qc"),
               ("profile_depth_qc", "argo_profile_depth_qc"))}
    if with_reanalysis:
        fields["reanalysis"] = data.read(spec["reanalysis"], rows)
    lat, lon = profiles.index["latitude"].to_numpy(), profiles.index["longitude"].to_numpy()
    fields["coords_ok"] = [math.isfinite(a) and math.isfinite(b) and abs(a) <= 90 and -180 <= b < 180
                           for a, b in zip(lat, lon)]
    fields["time_ok"] = [math.isfinite(t) for t in profiles.index["profile_juld"].to_numpy()]
    fields["temporal_status"] = profiles.index["glorys_temporal_status"].to_numpy()
    return fields


def _accepted(fields, i, qc_policy="qc_policy_v1"):
    raw = {k: fields[k][i].tolist() for k in ("value", "mask", "level_qc", "depth_qc")}
    return accepted_levels(raw, int(fields["profile_qc"][i]), int(fields["position_qc"][i]), int(fields["juld_qc"][i]),
                           int(fields["profile_depth_qc"][i]), fields["coords_ok"][i], fields["time_ok"][i], qc_policy)


def mld_reference(data, profiles, groups, qc_policy="qc_policy_v1"):
    fields = load_profiles(data, profiles, "potential_temperature")
    depth = data.depth.tolist()
    per_group = {}
    for i, group in enumerate(groups):
        profile_ok, ok = _accepted(fields, i, qc_policy)
        status, mld, top, bottom, _ = mld_one(fields["value"][i].tolist(), ok, depth, profile_ok)
        g = per_group.setdefault(group, {"profiles": 0, "counts": {}, "mld": [], "width": []})
        g["profiles"] += 1
        g["counts"][status] = g["counts"].get(status, 0) + 1
        if status == "estimable":
            g["mld"].append(mld)
            g["width"].append(bottom - top)
    out = {}
    for group, g in per_group.items():
        summary = lambda xs: ({"q25": quantile7(xs, .25), "median": quantile7(xs, .5), "q75": quantile7(xs, .75)}
                              if xs else None)
        out[group] = {"profiles": g["profiles"], "counts": g["counts"], "mld_m": summary(g["mld"]),
                      "interval_width_m": summary(g["width"])}
    return out


def consistency_reference(data, profiles, variable, groups, depth_index, qc_policy="qc_policy_v1",
                          temporal_filter="nearest_or_exact"):
    fields = load_profiles(data, profiles, variable, with_reanalysis=True)
    sums = {}
    excluded, rejected = {}, {}
    for i, group in enumerate(groups):
        profile_ok, ok = _accepted(fields, i, qc_policy)
        rejected[group] = rejected.get(group, 0) + (not profile_ok)
        status = int(fields["temporal_status"][i])
        usable = status == 0 if temporal_filter == "nearest_or_exact" else True
        excluded[group] = excluded.get(group, 0) + (not usable)
        for k in depth_index:
            cell = sums.setdefault((group, k), [])
            if usable and ok[k] and math.isfinite(float(fields["reanalysis"][i][k])):
                cell.append(float(fields["reanalysis"][i][k]) - float(fields["value"][i][k]))
    out = {}
    for group in sorted(set(groups)):
        rows = []
        for k in depth_index:
            d = sums[(group, k)]
            if d:
                rows.append({"level": k, "n": len(d), "bias": math.fsum(d) / len(d),
                             "mae": math.fsum(abs(x) for x in d) / len(d),
                             "rmse": math.sqrt(math.fsum(x * x for x in d) / len(d))})
            else:
                rows.append({"level": k, "n": 0, "bias": None, "mae": None, "rmse": None})
        out[group] = {"profiles": int(sum(1 for g in groups if g == group)),
                      "excluded_temporal_status": int(excluded[group]),
                      "profiles_rejected_qc": int(rejected[group]), "depths": rows}
    return out


def select(index, region, start_date, end_date):
    """Profiles in [lat_min, lat_max) x [lon_min, lon_max) observed on start..end inclusive, from raw JULD days."""
    day0 = (pd.Timestamp(start_date) - JULD_EPOCH) / pd.Timedelta(days=1)
    day1 = (pd.Timestamp(end_date) - JULD_EPOCH) / pd.Timedelta(days=1) + 1
    lat, lon = index["latitude"].to_numpy(), index["longitude"].to_numpy()
    juld = index["profile_juld"].to_numpy()
    keep = ((lat >= region["lat_min"]) & (lat < region["lat_max"]) & (lon >= region["lon_min"])
            & (lon < region["lon_max"]) & (juld >= day0) & (juld < day1))
    part = index[keep]
    return Profiles(part["profile"].to_numpy(), part)


def handle_reference(data, tool, params):
    """Reference results, group by group, for a compute handle's own (possibly diagnostic) parameters."""
    out = []
    for g in params["groups"]:
        profiles = select(data.index, params["region"], g["start_date"], g["end_date"])
        labels = np.array(["g"] * len(profiles.rows), dtype=object)
        entry = {"start_date": g["start_date"], "end_date": g["end_date"], "profiles": int(len(profiles.rows)),
                 "profile_set_sha256": profile_set_digest(profiles)}
        if tool == "compute_mld":
            ref = mld_reference(data, profiles, labels, params["qc_policy"]).get("g")
            entry["counts"] = {s: (ref["counts"].get(s, 0) if ref else 0) for s in MLD_STATUSES}
            entry["mld_m"] = ref["mld_m"] if ref else None
            entry["interval_width_m"] = ref["interval_width_m"] if ref else None
        else:
            ref = consistency_reference(data, profiles, params["variable"], labels, params["levels"],
                                        params["qc_policy"], params["temporal_filter"]).get("g")
            entry["excluded_temporal_status"] = ref["excluded_temporal_status"] if ref else 0
            entry["profiles_rejected_qc"] = ref["profiles_rejected_qc"] if ref else 0
            entry["depths"] = ref["depths"] if ref else [
                {"level": k, "n": 0, "bias": None, "mae": None, "rmse": None} for k in params["levels"]]
        out.append(entry)
    return out
