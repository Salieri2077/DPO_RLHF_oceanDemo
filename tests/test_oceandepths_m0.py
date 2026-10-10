import unittest

import numpy as np

from scripts.oceandepths_m0 import mld_threshold, sample_chunks


class MixedLayerTests(unittest.TestCase):
    def calculate(self, theta, valid=None):
        theta = np.array([theta], dtype=float)
        valid = np.ones(theta.shape, dtype=bool) if valid is None else np.array([valid], dtype=bool)
        return mld_threshold(theta, valid, np.array([9., 11., 20., 30.]))

    def test_signed_temperature_crossing(self):
        # The true linear crossing is 27.5 m; interpolating absolute endpoint differences gives 25 m.
        for theta in ([20., 20., 19.9, 20.3], [20., 20., 20.1, 19.7]):
            status, depth, _, _ = self.calculate(theta)
            self.assertEqual(status[0], "estimable")
            self.assertAlmostEqual(depth[0], 27.5)

    def test_reference_is_interpolated_and_needs_both_sides(self):
        status, depth, _, _ = self.calculate([0., .2, .3, .4])
        self.assertEqual(status[0], "estimable")
        self.assertAlmostEqual(depth[0], 20.)
        for valid in ([False, True, True, True], [True, False, True, True]):
            status, _, _, _ = self.calculate([0., .2, .3, .4], valid)
            self.assertEqual(status[0], "no_reference")

    def test_gap_does_not_bridge_to_deeper_crossing(self):
        status, depth, last, reason = self.calculate([20., 20., np.nan, 19.])
        self.assertEqual(status[0], "support_interrupted")
        self.assertEqual(reason[0], "gap")
        self.assertTrue(np.isnan(depth[0]))
        self.assertEqual(last[0], 11.)

    def test_end_without_crossing_is_separate(self):
        status, depth, last, reason = self.calculate([20., 20., 20.05, 20.1])
        self.assertEqual(status[0], "right_censored")
        self.assertEqual(reason[0], "profile_end")
        self.assertTrue(np.isnan(depth[0]))
        self.assertEqual(last[0], 30.)


class SamplingTests(unittest.TestCase):
    def test_year_boundary_chunks_are_not_repeated_or_misattributed(self):
        year = np.repeat([2000, 2001, 2002], 35)
        picks = sample_chunks(year, 8, 2)
        self.assertEqual(len(picks), 6)
        self.assertEqual(len(picks), len(set(picks)))
        for c in picks:
            self.assertEqual(len(np.unique(year[c * 8:(c + 1) * 8])), 1)

    def test_many_requested_draws_cannot_duplicate_small_year(self):
        picks = sample_chunks(np.repeat(2000, 16), 8, 10)
        self.assertEqual(picks, [0, 1])


if __name__ == "__main__":
    unittest.main()
