#!/usr/bin/env python3
"""Human-checkable spot checks of mld_temperature_v1 on real dev profiles: one profile per status (plus a crossing
reached from the opposite side of theta10, if present), with raw values, QC and every intermediate quantity."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.oceanlab import reference
from agent.oceanlab.data import OceanDepths, Profiles


def main():
    data = OceanDepths()
    requests = [json.loads(l) for l in (ROOT / "data/processed/oceanlab/oceanlab_dev_v1.jsonl").open(encoding="utf-8")]
    depth = data.depth.tolist()
    wanted = {"estimable", "support_interrupted", "right_censored", "no_reference", "profile_excluded", "opposite_side"}
    lines = ["# mld_temperature_v1 抽查（真实开发剖面）", "",
             "每条剖面列出 10m 两侧层、扫描经过的层（到越界层或停止处为止），可手算核对。θ 单位 ℃，深度 m。", ""]
    for request in requests:
        if not wanted or "mld" not in request["analyses"]:
            continue
        (lat, lon), months = request["region"], [m for g in request["groups"] for m in g["months"]]
        start, end = f"{months[0]}-01", f"{months[-1]}-28"
        profiles = data.select(lat, lon, start, end)
        fields = reference.load_profiles(data, profiles, "potential_temperature")
        for i in range(len(profiles.rows)):
            profile_ok, ok = reference._accepted(fields, i)
            theta = fields["value"][i].tolist()
            status, mld, top, bottom, last = reference.mld_one(theta, ok, depth, profile_ok)
            hi = next(k for k, d in enumerate(depth) if d > 10)
            key = status
            if status == "estimable":
                theta10 = float(np.interp(10.0, depth[hi - 1:hi + 1], theta[hi - 1:hi + 1]))
                k = depth.index(bottom)
                prev_t = theta10 if top == 10.0 else theta[k - 1]
                if (prev_t - theta10) * (theta[k] - theta10) < 0:
                    key = "opposite_side"
            if key not in wanted:
                continue
            wanted.discard(key)
            row = profiles.index.iloc[i]
            lines += [f"## {key}（请求 {request['id']}）", "",
                      f"- 记录键：`{row['profile_source_file']}:{row['profile_idx']}`；时间 {row['time']}；"
                      f"位置 {row['latitude']:.3f}, {row['longitude']:.3f}",
                      f"- 位置 QC {int(fields['position_qc'][i])}，势温剖面 QC {int(fields['profile_qc'][i])}，"
                      f"时间 QC {int(fields['juld_qc'][i])}，剖面深度 QC {int(fields['profile_depth_qc'][i])}"
                      f" → 剖面{'合格' if profile_ok else '被排除'}", "",
                      "| 层 | 深度 | θ | 掩膜 | 层 QC | 深度 QC | 合格 |", "|---:|---:|---:|---:|---:|---:|:---:|"]
            stop = len(depth) if status in ("right_censored", "support_interrupted") else (
                depth.index(bottom) + 1 if status == "estimable" else hi + 1)
            stop = min(stop, len(depth), hi + 12 if status != "estimable" else len(depth))
            for k in range(hi - 1, stop):
                lines.append(f"| {k} | {depth[k]:.3f} | {theta[k]:.4f} | {int(fields['mask'][i][k])} | "
                             f"{int(fields['level_qc'][i][k])} | {int(fields['depth_qc'][i][k])} | {'✓' if ok[k] else '✗'} |")
            if status not in ("no_reference", "profile_excluded"):
                w = (10 - depth[hi - 1]) / (depth[hi] - depth[hi - 1])
                theta10 = theta[hi - 1] + w * (theta[hi] - theta[hi - 1])
                lines.append(f"\nθ10 = {theta[hi-1]:.4f} + {w:.4f} × ({theta[hi]:.4f} − {theta[hi-1]:.4f}) = {theta10:.4f}")
            if status == "estimable":
                k = depth.index(bottom)
                goal = theta10 + (0.2 if theta[k] > theta10 else -0.2)
                lines.append(f"首次 |θ−θ10| ≥ 0.2 在 {bottom:.3f} m（θ={theta[k]:.4f}）；目标 θ = {goal:.4f}；"
                             f"上一点 ({top:.3f} m, {prev_t:.4f})；交点 = {top:.3f} + ({goal:.4f} − {prev_t:.4f}) / "
                             f"({theta[k]:.4f} − {prev_t:.4f}) × ({bottom:.3f} − {top:.3f}) = **{mld:.3f} m**，"
                             f"越界区间 [{top:.3f}, {bottom:.3f}]，宽 {bottom - top:.3f} m")
            elif status in ("support_interrupted", "right_censored"):
                lines.append(f"\n结果：**{status}**，最后连续合格深度 {last:.3f} m（不给精确深度）")
            else:
                lines.append(f"\n结果：**{status}**")
            lines.append("")
    out = ROOT / "artifacts/oceanlab/spot_checks_dev_v1.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out, "missing kinds:", sorted(wanted))


if __name__ == "__main__":
    main()
