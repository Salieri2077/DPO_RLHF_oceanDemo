"""Hand-worked cases for the frozen method cards, checked against both the tool-side (`methods`) and the
scorer-side (`reference`) implementations, plus a real-data agreement check when the dataset is available."""
import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agent.oceanlab import methods, reference
from agent.oceanlab.data import partition_of

DEPTH = np.array([5.078, 9.573, 11.405, 20.0, 30.0, 40.0])
HAS_DATA = Path("/data/anhuang/oceandepths/SOURCE_MANIFEST.json").exists() and importlib.util.find_spec("zarr")


def both(theta, ok, profile_ok=True):
    """Run one profile through both implementations; they must agree."""
    tool = methods.mld_temperature_v1(np.array([theta], float), np.array([ok]), DEPTH, np.array([profile_ok]))
    ref = reference.mld_one(theta, ok, DEPTH.tolist(), profile_ok)
    assert tool["status"][0] == ref[0], (tool["status"][0], ref[0])
    if ref[0] == "estimable":
        assert math.isclose(tool["mld"][0], ref[1], rel_tol=1e-12)
        assert (tool["interval_top"][0], tool["interval_bottom"][0]) == (ref[2], ref[3])
    elif ref[0] in ("support_interrupted", "right_censored"):
        assert tool["last_supported_depth"][0] == ref[4]
    return ref


class MixedLayerCases(unittest.TestCase):
    def test_threshold_target_handles_sign_change(self):
        # theta10 = 15.0; 20 m is -0.1, 30 m is +0.3: crossing of 15.2 at 27.5 m (old |dT| interpolation gave 25 m).
        status, mld, top, bottom, _ = both([15.0, 15.0, 15.0, 14.9, 15.3, 15.3], [True] * 6)
        self.assertEqual(status, "estimable")
        self.assertAlmostEqual(mld, 27.5)
        self.assertEqual((top, bottom), (20.0, 30.0))

    def test_crossing_at_first_level_below_reference(self):
        # theta10 interpolated between 9.573 and 11.405; the first scan step starts from (10 m, theta10).
        status, mld, top, bottom, _ = both([15.0, 15.0, 16.0, 16.0, 16.0, 16.0], [True] * 6)
        w = (10 - 9.573) / (11.405 - 9.573)
        theta10 = 15.0 + w
        self.assertEqual((status, top, bottom), ("estimable", 10.0, 11.405))
        self.assertAlmostEqual(mld, 10 + (0.2 / (16.0 - theta10)) * 1.405)

    def test_one_sided_reference_is_no_reference(self):
        self.assertEqual(both([15.0, math.nan, 15.0, 14.0, 14.0, 14.0], [True, False, True, True, True, True])[0],
                         "no_reference")
        self.assertEqual(both([15.0, 15.0, math.nan, 14.0, 14.0, 14.0], [True, True, False, True, True, True])[0],
                         "no_reference")

    def test_internal_gap_interrupts_support(self):
        # The crossing at 30 m is behind a rejected 20 m level: not a first crossing, no depth is returned.
        status, mld, *_, last = both([15.0, 15.0, 15.0, 15.0, 14.0, 14.0], [True, True, True, False, True, True])
        self.assertEqual((status, mld, last), ("support_interrupted", None, 11.405))

    def test_no_crossing_to_last_level_is_right_censored(self):
        status, *_, last = both([15.0, 15.0, 15.0, 15.1, 15.1, 15.1], [True] * 6)
        self.assertEqual((status, last), ("right_censored", 40.0))
        status, *_, last = both([15.0, 15.0, 15.0, 15.1, math.nan, math.nan], [True, True, True, True, False, False])
        self.assertEqual((status, last), ("right_censored", 20.0))

    def test_rejected_profile_is_excluded(self):
        self.assertEqual(both([15.0, 15.0, 15.0, 14.0, 14.0, 14.0], [True] * 6, profile_ok=False)[0],
                         "profile_excluded")

    def test_quantile_type7(self):
        values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
        for p in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(reference.quantile7(values, p), np.percentile(values, 100 * p, method="linear"))


class QualityAndPartition(unittest.TestCase):
    def test_qc_policy_v1(self):
        raw = {"value": [1.0, 2.0, math.nan, 4.0], "mask": [1, 1, 1, 0], "level_qc": [1, 4, 1, 1],
               "depth_qc": [-1, 1, 1, 1]}
        ok, levels = reference.accepted_levels(raw, 1, 1, -1, -1, True, True)
        self.assertTrue(ok)
        self.assertEqual(levels, [True, False, False, False])  # QC 4, non-finite, mask 0 rejected; -1 depth QC unknown
        self.assertFalse(reference.accepted_levels(raw, 1, 4, -1, -1, True, True)[0])  # position rejected
        self.assertFalse(reference.accepted_levels(raw, 4, 1, -1, -1, True, True)[0])  # variable profile rejected
        self.assertFalse(reference.accepted_levels(raw, 1, 1, 4, -1, True, True)[0])  # available time QC must be 1

    def test_consistency_excludes_temporal_status(self):
        observed = np.array([[1.0, 2.0], [1.0, 2.0]])
        reanalysis = np.array([[1.5, 2.5], [9.0, 9.0]])
        accepted = np.ones((2, 2), bool)
        out = methods.consistency(observed, accepted, reanalysis, np.array([0, 1]), np.array(["g", "g"]), [0, 1])
        self.assertEqual(out["g"]["excluded_temporal_status"], 1)
        self.assertEqual([(r["n"], r["bias"]) for r in out["g"]["depths"]], [(1, 0.5), (1, 0.5)])

    def test_partition_v1(self):
        t = pd.to_datetime(["2016-06-01", "2016-12-20", "2017-01-25", "2017-06-01", "2018-06-01", "2018-06-01",
                            "2021-06-01"])
        lat = [0, 0, 0, 0, -30, -20.5, -30]
        lon = [0, 0, 0, 0, -20, -20, -20]
        self.assertEqual(partition_of(t, lat, lon).tolist(),
                         ["train", "buffer", "buffer", "dev", "dev-reserved", "buffer", "test-reserved"])


@unittest.skipUnless(HAS_DATA, "OceanDepths data or zarr not available")
class RealDataAgreement(unittest.TestCase):
    """Tool-side and reference implementations must agree on real profiles."""

    def test_methods_match_reference(self):
        from agent.oceanlab.data import OceanDepths
        data = OceanDepths()
        for lat, lon, start, end in (((30, 40), (140, 150), "2018-01-01", "2018-03-31"),
                                     ((-60, -50), (60, 70), "2017-07-01", "2017-09-30")):
            profiles = data.select(lat, lon, start, end)
            groups = profiles.index["time"].dt.strftime("%Y-%m").to_numpy()
            theta, accepted, profile_ok = data.levels(profiles, "potential_temperature")
            tool = methods.summarize_mld(methods.mld_temperature_v1(theta, accepted, data.depth, profile_ok), groups)
            ref = reference.mld_reference(data, profiles, groups)
            for group, t in tool.items():
                r = ref[group]
                self.assertEqual({k: v for k, v in t["counts"].items() if v}, r["counts"])
                for key in ("mld_m", "interval_width_m"):
                    if t[key] is None:
                        self.assertIsNone(r[key])
                    else:
                        for q in ("q25", "median", "q75"):
                            self.assertAlmostEqual(t[key][q], r[key][q], places=9)
            for variable in ("potential_temperature", "salinity"):
                value, accepted, _ = data.levels(profiles, variable)
                reanalysis = data.read({"potential_temperature": "glorys_thetao", "salinity": "glorys_so"}[variable],
                                       profiles.rows).astype(np.float64)
                status = profiles.index["glorys_temporal_status"].to_numpy()
                depth_index = list(range(4, 25))
                tool = methods.consistency(value, accepted, reanalysis, status, groups, depth_index)
                ref = reference.consistency_reference(data, profiles, variable, groups, depth_index)
                for group in tool:
                    self.assertEqual(tool[group]["excluded_temporal_status"], ref[group]["excluded_temporal_status"])
                    for a, b in zip(tool[group]["depths"], ref[group]["depths"]):
                        self.assertEqual(a["n"], b["n"])
                        for key in ("bias", "mae", "rmse"):
                            if a[key] is not None:
                                self.assertAlmostEqual(a[key], b[key], places=9)


if __name__ == "__main__":
    unittest.main()
