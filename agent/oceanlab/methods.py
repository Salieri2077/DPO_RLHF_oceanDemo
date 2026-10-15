"""Tool-side (vectorised) implementations of the frozen method cards. The scorer checks them against `reference`.

mld_temperature_v1: 10 m reference interpolated from the two bracketing native levels (both accepted, else
no_reference); scan contiguous accepted levels below it; the first level with |theta - theta10| >= 0.2 degC is
intersected with the target theta10 + sign(theta_k - theta10) * 0.2 by linear interpolation in theta against the
previous point (theta10 at 10 m for the first level). A non-accepted level before any crossing stops the scan:
support_interrupted if an accepted level exists deeper, otherwise right_censored; reaching the last accepted level
without crossing is right_censored. Profile-level rejections are profile_excluded.

profile_consistency_v1: per group and depth, common accepted pairs (EN4 accepted, GLORYS finite, index temporal
status 0 = nearest_or_exact); difference = reanalysis - observation; bias, MAE, RMSE and n per depth.
temporal_filter="any" (statuses 0/1/2 with a finite GLORYS value) is a collocation-sensitivity diagnostic only.

Summaries keep every requested group (`labels`), including groups without profiles, which get zero counts.
"""
import numpy as np

REFERENCE_DEPTH = 10.0
THRESHOLD = 0.2
STATUSES = ("estimable", "support_interrupted", "right_censored", "no_reference", "profile_excluded")


def mld_temperature_v1(theta, accepted, depth, profile_ok):
    """theta/accepted: (profiles, levels). Returns dict of per-profile arrays."""
    n, levels = theta.shape
    hi = int(np.searchsorted(depth, REFERENCE_DEPTH, side="right"))  # first native level deeper than 10 m
    lo = hi - 1
    weight = (REFERENCE_DEPTH - depth[lo]) / (depth[hi] - depth[lo])
    has_ref = accepted[:, lo] & accepted[:, hi]
    theta10 = theta[:, lo] + weight * (theta[:, hi] - theta[:, lo])

    below_ok = accepted[:, hi:]
    below_t = theta[:, hi:]
    below_d = depth[hi:]
    m = below_ok.shape[1]
    with np.errstate(invalid="ignore"):
        crossed = below_ok & (np.abs(below_t - theta10[:, None]) >= THRESHOLD)
    first_bad = np.where((~below_ok).any(1), (~below_ok).argmax(1), m)
    first_cross = np.where(crossed.any(1), crossed.argmax(1), m)
    estimable = has_ref & (first_cross < first_bad)

    rows = np.arange(n)
    k = np.minimum(first_cross, m - 1)
    prev_d = np.where(k == 0, REFERENCE_DEPTH, below_d[np.maximum(k - 1, 0)])
    prev_t = np.where(k == 0, theta10, below_t[rows, np.maximum(k - 1, 0)])
    cross_t = below_t[rows, k]
    target = theta10 + np.copysign(THRESHOLD, cross_t - theta10)
    with np.errstate(invalid="ignore", divide="ignore"):
        mld = prev_d + (target - prev_t) / (cross_t - prev_t) * (below_d[k] - prev_d)

    deeper_ok = np.zeros(n, dtype=bool)
    for j in range(m):  # any accepted level strictly below the first non-accepted one
        deeper_ok |= below_ok[:, j] & (j > first_bad)
    last_supported = np.where(first_bad == 0, REFERENCE_DEPTH, below_d[np.clip(first_bad - 1, 0, m - 1)])

    status = np.full(n, "right_censored", dtype=object)
    status[estimable] = "estimable"
    status[~estimable & (first_bad < m) & deeper_ok] = "support_interrupted"
    status[~has_ref] = "no_reference"
    status[~profile_ok] = "profile_excluded"
    est = status == "estimable"
    stopped = np.isin(status, ["support_interrupted", "right_censored"])
    return {
        "status": status,
        "theta10": np.where(has_ref & profile_ok, theta10, np.nan),
        "mld": np.where(est, mld, np.nan),
        "interval_top": np.where(est, prev_d, np.nan),
        "interval_bottom": np.where(est, below_d[k], np.nan),
        "last_supported_depth": np.where(stopped, last_supported, np.nan),
    }


def quartiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    q25, median, q75 = np.percentile(values, [25, 50, 75], method="linear")
    return {"q25": float(q25), "median": float(median), "q75": float(q75)}


def summarize_mld(result, groups, labels=None):
    """Per group: status counts and quartiles of estimable MLD and of crossing-interval widths."""
    out = {}
    for group in (sorted(set(groups)) if labels is None else labels):
        sel = groups == group
        status = result["status"][sel]
        counts = {s: int((status == s).sum()) for s in STATUSES}
        est = status == "estimable"
        width = (result["interval_bottom"] - result["interval_top"])[sel][est]
        out[group] = {"profiles": int(sel.sum()), "counts": counts,
                      "mld_m": quartiles(result["mld"][sel][est]), "interval_width_m": quartiles(width)}
    return out


def consistency(observed, accepted, reanalysis, temporal_status, groups, depth_index,
                temporal_filter="nearest_or_exact", labels=None):
    """Per group and depth level: n, bias, MAE, RMSE of reanalysis - observation on common accepted pairs."""
    if temporal_filter not in ("nearest_or_exact", "any"):
        raise ValueError(f"unknown temporal filter {temporal_filter}")
    usable = temporal_status == 0 if temporal_filter == "nearest_or_exact" else np.ones(len(temporal_status), bool)
    pair = accepted & np.isfinite(reanalysis) & usable[:, None]
    diff = reanalysis - observed
    out = {}
    for group in (sorted(set(groups)) if labels is None else labels):
        sel = groups == group
        rows = []
        for k in depth_index:
            m = pair[sel, k]
            d = diff[sel, k][m]
            if d.size:
                rows.append({"level": int(k), "n": int(d.size), "bias": float(d.mean()),
                             "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean()))})
            else:
                rows.append({"level": int(k), "n": 0, "bias": None, "mae": None, "rmse": None})
        out[group] = {"profiles": int(sel.sum()),
                      "excluded_temporal_status": int((~usable[sel]).sum()), "depths": rows}
    return out
