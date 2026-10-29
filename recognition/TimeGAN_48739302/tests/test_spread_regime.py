import unittest

import numpy as np

from spread_regime_analysis import clock_weights, histogram_kl, sampled_spreads, spread_values, summary


class SpreadRegimeTest(unittest.TestCase):
    def test_clock_weights_handle_ties_and_boundary_crossing(self):
        times = np.array([0.0, 0.0, 5.0, 12.0])
        weights = clock_weights(times, 3, 10, session_end=20)
        np.testing.assert_array_equal(weights, [0, 2, 5, 0])
        self.assertEqual(weights.sum(), 7)

    def test_clock_weighted_summary_uses_duration(self):
        result = summary(np.array([2.0, 10.0]), np.array([9.0, 1.0]))
        self.assertAlmostEqual(result["mean"], 2.8)
        self.assertEqual(result["median"], 2)
        self.assertEqual(result["p95"], 10)

    def test_KL_direction_is_not_interchangeable(self):
        p, q = np.array([1] * 9 + [2]), np.array([1] * 5 + [2] * 5)
        edges = np.array([0.5, 1.5, 2.5])
        self.assertEqual(histogram_kl(p, p, edges), 0)
        self.assertGreater(abs(histogram_kl(p, q, edges) - histogram_kl(q, p, edges)), 0.1)

    def test_sampling_respects_period_boundaries_and_quote_validity(self):
        times = np.arange(20, dtype=float)
        windows, flat = sampled_spreads(times, np.arange(20, dtype=float), 0, 12, 3, seq_len=4)
        self.assertEqual(windows.shape, (3, 4))
        np.testing.assert_array_equal(flat, np.arange(12))
        spreads = spread_values(np.array([[1000, 1, 800, 1], [1e10, 1, 800, 1]]))
        self.assertEqual(spreads[0], 2)
        self.assertTrue(np.isnan(spreads[1]))


if __name__ == "__main__":
    unittest.main()
