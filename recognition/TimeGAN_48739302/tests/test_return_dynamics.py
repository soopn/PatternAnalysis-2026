import unittest

import numpy as np

from evaluate import kl_divergence
from predict import decode_features
from return_dynamics_analysis import (
    RETURN_EDGES, autocorrelation, decoded_quote_mid, return_summary,
    shuffle_increments, zero_jump_kl,
)


class ReturnDynamicsTest(unittest.TestCase):
    def test_zero_jump_decomposition_matches_production_KL(self):
        real = np.array([0] * 90 + [0.5] * 5 + [-0.5] * 5)
        fake = np.array([0] * 80 + [1] * 10 + [-1] * 10)
        result = zero_jump_kl(real, fake)
        self.assertAlmostEqual(result["total"], kl_divergence(real, fake, RETURN_EDGES), places=12)
        self.assertAlmostEqual(result["total"], result["zero_vs_move_frequency"]
                               + result["weighted_nonzero_shape"], places=12)

    def test_shuffling_preserves_jumps_zeros_and_window_drift(self):
        increments = np.array([[0, 0.5, -1, 0, 2], [1, 0, -0.5, 0, 0]])
        shuffled = shuffle_increments(increments, np.random.default_rng(5))
        np.testing.assert_array_equal(np.sort(increments, axis=1), np.sort(shuffled, axis=1))
        np.testing.assert_array_equal(increments.sum(1), shuffled.sum(1))

    def test_no_returns_cross_window_boundaries(self):
        mid = np.array([[100.0] * 100, [1000.0] * 100])
        result = return_summary(mid)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["zero_share"], 1)
        self.assertEqual(result["quiet_window_share"], 1)

    def test_signed_autocorrelation_detects_alternation(self):
        values = np.tile([1.0, -1.0], (3, 50))
        acf = autocorrelation(values)
        self.assertAlmostEqual(acf[0], -1)
        self.assertAlmostEqual(acf[1], 1)

    def test_current_decoder_has_anchor_parity_effect(self):
        # Regression diagnostic: this records existing behaviour, not a fix.
        features = np.ones((1, 3, 42))
        for index in range(40):
            if index % 4 in (1, 3):
                features[..., index] = np.log1p(100)
        features[..., 40] = [0, np.arcsinh(0.5), np.arcsinh(0.5)]
        first = decoded_quote_mid(decode_features(features, np.array([100.5])))
        shifted = decoded_quote_mid(decode_features(features, np.array([101.5])))
        np.testing.assert_array_equal(np.diff(first), [[0, 0]])
        np.testing.assert_array_equal(np.diff(shifted), [[1, 1]])


if __name__ == "__main__":
    unittest.main()
