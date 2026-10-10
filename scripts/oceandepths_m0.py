#!/usr/bin/env python3
"""M0 data check for the OceanDepths analysis-agent tasks (Agent Evaluation/SKILLRL_OCEAN_DESIGN.md).

Reads whole zarr chunks (8192 profiles) sampled evenly per year from the full enriched store and reports what the
two task families depend on: QC codes and their relation to the finite-value masks, 10 m reference support, the
estimable / support-interrupted / right-censored / no-reference split of the temperature-threshold method, EN4-vs-GLORYS
common-valid pair statistics per depth, salinity co-coverage, and region/time query cost. Descriptive only; it does
not generate task answers. The qc1 comparison requires accepted position, variable-profile and variable-level QC.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

STORE = "data/argo_glors_ostia_ssh.zarr"
REPORT_DEPTHS = [0.494, 5.078, 9.573, 47.374, 109.729, 222.475, 541.089, 1062.44]


def mld_threshold(theta, valid, depth, ref_depth=10.0, delta=0.2):
    """Temperature-threshold calculation on one block of profiles (rows).

    Reference: linear interpolation at ref_depth between the two bracketing levels, both required valid.
    Scan the contiguous valid levels below the reference; the first |theta - ref| >= delta is interpolated against the
    previous temperature (the reference itself for the first level). A gap stops the scan without bridging;
    interrupted support is separate from reaching the end without a crossing. Returns statuses and depths.
    """
    n = theta.shape[0]
    valid = valid.astype(bool) & np.isfinite(theta)
    upper = int(np.searchsorted(depth, ref_depth))  # first level deeper than ref_depth
    lower = upper - 1
    status = np.full(n, "no_reference", dtype=object)
    value = np.full(n, np.nan)
    censor_depth = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)
    has_ref = valid[:, lower] & valid[:, upper]
    w = (ref_depth - depth[lower]) / (depth[upper] - depth[lower])
    ref = theta[:, lower] * (1 - w) + theta[:, upper] * w
    for i in np.flatnonzero(has_ref):
        prev_d, prev_t = ref_depth, ref[i]
        status[i], reason[i] = "right_censored", "profile_end"
        last_valid = ref_depth
        for k in range(upper, len(depth)):
            if not valid[i, k]:
                reason[i] = "gap" if valid[i, k:].any() else "profile_end"
                if reason[i] == "gap":
                    status[i] = "support_interrupted"
                break
            t = theta[i, k]
            if abs(t - ref[i]) >= delta:
                # Interpolate theta, since abs(theta - ref) is not linear across a sign change.
                target = ref[i] + np.copysign(delta, t - ref[i])
                frac = (target - prev_t) / (t - prev_t)
                value[i] = prev_d + frac * (depth[k] - prev_d)
                status[i], reason[i] = "estimable", ""
                break
            prev_d, prev_t, last_valid = depth[k], t, depth[k]
        if status[i] in {"support_interrupted", "right_censored"}:
            censor_depth[i] = last_valid
    return status, value, censor_depth, reason


def sample_chunks(year, chunk_size, per_year):
    """Pick distinct whole chunks within each year, away from shared year boundaries."""
    if per_year < 1:
        raise ValueError("chunks_per_year must be positive")
    picks = set()
    for y in np.unique(year):
        rows = np.flatnonzero(year == y)
        first = (int(rows[0]) + chunk_size - 1) // chunk_size
        last = (int(rows[-1]) + 1) // chunk_size - 1
        if first > last:
            continue
        candidates = np.arange(first, last + 1)
        candidates = np.array([c for c in candidates
                               if np.all(year[c * chunk_size:(c + 1) * chunk_size] == y)], dtype=int)
        if not len(candidates):
            continue
        count = min(per_year, len(candidates))
        positions = np.floor((np.arange(count) + .5) * len(candidates) / count).astype(int)
        picks.update(candidates[positions].tolist())
    return sorted(picks)


class PairStats:
    """Running GLORYS-minus-EN4 statistics on common valid pairs, per depth column."""

    def __init__(self, levels):
        self.n, self.s, self.a, self.q, self.big = (np.zeros(levels) for _ in range(5))

    def add(self, observed, reanalysis, mask):
        d = np.where(mask, reanalysis.astype(np.float64) - observed.astype(np.float64), 0.0)
        self.n += mask.sum(0)
        self.s += d.sum(0)
        self.a += np.abs(d).sum(0)
        self.q += (d ** 2).sum(0)
        self.big += (np.abs(d) > 5).sum(0)

    def at(self, k):
        n = self.n[k]
        if not n:
            return {"n": 0}
        return {"n": int(n), "bias": self.s[k] / n, "mae": self.a[k] / n, "rmse": float(np.sqrt(self.q[k] / n)),
                "share_abs_gt_5": self.big[k] / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/anhuang/oceandepths"))
    parser.add_argument("--chunks_per_year", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/oceandepths_m0"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = zarr.open_consolidated(str(args.root / STORE), mode="r")
    depth = root["glorys_depth"][:].astype(np.float64)
    chunk = root["argo_potm_on_glorys_depth"].chunks[0]
    index = pd.read_parquet(args.root / "indices/profiles.parquet",
                            columns=["profile", "profile_date", "latitude", "longitude", "glorys_temporal_status"])
    if not np.array_equal(index["profile"].to_numpy(), np.arange(len(index))):
        raise ValueError("Profile index must match the store row order")
    year = (index["profile_date"] // 10000).to_numpy()

    picks = sample_chunks(year, chunk, args.chunks_per_year)

    qc_valid, qc_invalid, profile_qc = Counter(), Counter(), Counter()
    status_all, reason_all = Counter(), Counter()
    mld_values, censor_depths, mld_glorys, mld_by_year = [], [], [], {}
    status_qc1, reason_qc1, mld_qc1 = Counter(), Counter(), []
    psal_qc_valid = Counter()
    temp_pairs, sal_pairs = PairStats(len(depth)), PairStats(len(depth))
    temp_pairs_qc1, sal_pairs_qc1 = PairStats(len(depth)), PairStats(len(depth))
    ref_support = Counter()
    sal_cover = Counter()
    read_seconds = 0.0
    for c in picks:
        sl = slice(c * chunk, (c + 1) * chunk)
        started = time.time()
        potm = root["argo_potm_on_glorys_depth"][sl]
        pvalid = root["argo_potm_valid_on_glorys_depth"][sl].astype(bool)
        pqc = root["argo_potm_qc_on_glorys_depth"][sl]
        thetao = root["glorys_thetao"][sl]
        psal = root["argo_psal_on_glorys_depth"][sl]
        svalid = root["argo_psal_valid_on_glorys_depth"][sl].astype(bool)
        sqc = root["argo_psal_qc_on_glorys_depth"][sl]
        so = root["glorys_so"][sl]
        mlotst = root["glorys_mlotst"][sl]
        prof_qc = root["argo_profile_potm_qc"][sl]
        sal_prof_qc = root["argo_profile_psal_qc"][sl]
        position_qc = root["argo_position_qc"][sl]
        juld_qc = root["argo_juld_qc"][sl]
        juld = root["profile_juld"][sl]
        depth_qc = root["argo_depth_qc_on_glorys_depth"][sl]
        # Zero-only Zarr status chunks are absent with fill_value=null; slice reads can return uninitialized bytes.
        temporal_status = index["glorys_temporal_status"].iloc[sl].to_numpy()
        if not np.isin(temporal_status, [0, 1, 2]).all():
            raise ValueError("Unknown GLORYS temporal status in profile index")
        dates = root["profile_date"][sl]
        read_seconds += time.time() - started

        qc_valid.update(pqc[pvalid].tolist())
        qc_invalid.update(pqc[~pvalid].tolist())
        profile_qc.update(prof_qc.tolist())
        lower = int(np.searchsorted(depth, 10.0)) - 1
        ref_support.update({"both": int((pvalid[:, lower] & pvalid[:, lower + 1]).sum()),
                            "shallow_only": int((pvalid[:, lower] & ~pvalid[:, lower + 1]).sum()),
                            "deep_only": int((~pvalid[:, lower] & pvalid[:, lower + 1]).sum()),
                            "neither": int((~pvalid[:, lower] & ~pvalid[:, lower + 1]).sum())})
        status, value, censor, reason = mld_threshold(potm.astype(np.float64), pvalid, depth)
        status_all.update(status.tolist())
        reason_all.update(f"{s}:{r}" for s, r in zip(status, reason)
                          if s in {"support_interrupted", "right_censored"})
        est = status == "estimable"
        mld_values += value[est].tolist()
        mld_glorys += mlotst[est].astype(np.float64).tolist()
        stopped = np.isin(status, ["support_interrupted", "right_censored"])
        censor_depths += censor[stopped].tolist()
        for y in np.unique(dates // 10000):
            stats = mld_by_year.setdefault(int(y), Counter())
            stats.update(status[dates // 10000 == y].tolist())

        tmask = pvalid & np.isfinite(thetao)
        smask = svalid & np.isfinite(so)
        sal_cover.update({"profiles": len(dates), "with_any_psal": int(svalid.any(1).sum()),
                          "with_common_psal_so": int(smask.any(1).sum())})
        temp_pairs.add(potm, thetao, tmask)
        sal_pairs.add(psal, so, smask)
        row_ok = ((position_qc == 1) & np.isfinite(juld) & np.isin(juld_qc, [-1, 1])
                  & index["latitude"].iloc[sl].between(-90, 90).to_numpy()
                  & index["longitude"].iloc[sl].between(-180, 180).to_numpy())
        depth_ok = np.isin(depth_qc, [-1, 1])  # -1 is unavailable, not a claim that depth QC passed.
        taccepted = pvalid & np.isfinite(potm) & (pqc == 1) & depth_ok & ((prof_qc == 1) & row_ok)[:, None]
        saccepted = svalid & np.isfinite(psal) & (sqc == 1) & depth_ok & ((sal_prof_qc == 1) & row_ok)[:, None]
        temp_pairs_qc1.add(potm, thetao, taccepted & np.isfinite(thetao) & (temporal_status == 0)[:, None])
        sal_pairs_qc1.add(psal, so, saccepted & np.isfinite(so) & (temporal_status == 0)[:, None])
        psal_qc_valid.update(sqc[svalid].tolist())
        status1, value1, _, reason1 = mld_threshold(potm.astype(np.float64), taccepted, depth)
        status1[(prof_qc != 1) | ~row_ok] = "profile_excluded"
        status_qc1.update(status1.tolist())
        reason_qc1.update(f"{s}:{r}" for s, r in zip(status1, reason1)
                         if s in {"support_interrupted", "right_censored"})
        mld_qc1 += value1[status1 == "estimable"].tolist()

    mld = np.array(mld_values)
    glorys = np.array(mld_glorys)
    ok = np.isfinite(glorys)
    total = sum(status_all.values())

    # Query cost: one region/season request resolved through the index, then reading only the needed chunks.
    started = time.time()
    q = index[(index["profile_date"].between(20180101, 20180331)) & index["latitude"].between(30, 40)
              & index["longitude"].between(140, 160)]
    rows = q["profile"].to_numpy()
    chunks_needed = np.unique(rows // chunk)
    lookup = time.time() - started
    started = time.time()
    potm_q = root["argo_potm_on_glorys_depth"].get_orthogonal_selection((rows, slice(None)))
    read_q = time.time() - started

    summary = {
        "store": STORE, "profiles_total": int(len(index)), "chunk_size": chunk,
        "sampled_chunks": len(picks), "sampled_chunk_ids": picks,
        "sampling": "distinct whole chunks within each year; two interior time clusters per year by default, not population weighted",
        "qc1_policy": "finite, valid, variable level/profile and position QC=1; finite time and valid coordinates; depth/time QC=1 when available (-1 unavailable allowed); GLORYS comparisons also require index temporal_status=0",
        "sampled_profiles": total, "sample_read_seconds": round(read_seconds, 1),
        "depth_levels_m": [round(float(d), 3) for d in depth],
        "reference_10m_bracket_m": [round(float(depth[lower]), 3), round(float(depth[lower + 1]), 3)],
        "potm_qc_on_valid": dict(sorted(qc_valid.items())), "potm_qc_on_invalid": dict(sorted(qc_invalid.items())),
        "profile_potm_qc": dict(sorted(profile_qc.items())),
        "reference_support": {k: v / total for k, v in ref_support.items()},
        "mld_status_share": {k: v / total for k, v in status_all.items()},
        "mld_status_share_qc1": {k: v / total for k, v in status_qc1.items()},
        "mld_estimable_m_qc1": {"median": float(np.median(mld_qc1)), "p25": float(np.percentile(mld_qc1, 25)),
                                "p75": float(np.percentile(mld_qc1, 75))},
        "psal_qc_on_valid": dict(sorted(psal_qc_valid.items())),
        "censored_reason_share": {k: v / total for k, v in reason_all.items()},
        "censored_reason_share_qc1": {k: v / total for k, v in reason_qc1.items()},
        "mld_estimable_m": {"median": float(np.median(mld)), "p25": float(np.percentile(mld, 25)),
                            "p75": float(np.percentile(mld, 75)), "p95": float(np.percentile(mld, 95))},
        "censored_last_verified_depth_m": {"median": float(np.nanmedian(censor_depths)) if censor_depths else None},
        "mld_vs_glorys_mlotst_density_definition": {
            "n": int(ok.sum()), "spearman": float(pd.Series(mld[ok]).corr(pd.Series(glorys[ok]), method="spearman")),
            "median_glorys_minus_threshold_m": float(np.median(glorys[ok] - mld[ok]))},
        "mld_status_by_year": {y: {k: round(v / sum(c.values()), 3) for k, v in c.items()} for y, c in sorted(mld_by_year.items())},
        "temperature_pairs_glorys_minus_en4": {f"{depth[k]:.2f}m": temp_pairs.at(k) for k in range(len(depth))
                                                if any(abs(depth[k] - r) < 0.01 for r in REPORT_DEPTHS)},
        "salinity_pairs_glorys_minus_en4": {f"{depth[k]:.2f}m": sal_pairs.at(k) for k in range(len(depth))
                                             if any(abs(depth[k] - r) < 0.01 for r in REPORT_DEPTHS)},
        "temperature_pairs_qc1": {f"{depth[k]:.2f}m": temp_pairs_qc1.at(k) for k in range(len(depth))
                                  if any(abs(depth[k] - r) < 0.01 for r in REPORT_DEPTHS)},
        "salinity_pairs_qc1": {f"{depth[k]:.2f}m": sal_pairs_qc1.at(k) for k in range(len(depth))
                               if any(abs(depth[k] - r) < 0.01 for r in REPORT_DEPTHS)},
        "salinity_coverage": {k: v / sal_cover["profiles"] for k, v in sal_cover.items() if k != "profiles"},
        "example_query": {"request": "30-40N, 140-160E, 2018-01-01..2018-03-31", "profiles": int(len(rows)),
                          "chunks_touched": int(len(chunks_needed)), "index_lookup_s": round(lookup, 2),
                          "read_potm_s": round(read_q, 2), "shape": list(potm_q.shape)},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "depth_levels_m"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
