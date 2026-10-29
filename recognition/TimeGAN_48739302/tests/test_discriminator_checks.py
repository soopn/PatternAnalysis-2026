import unittest

import numpy as np

from discriminator_checks import best_threshold, classification_metrics, partition


class DiscriminatorChecksTest(unittest.TestCase):
    def test_auc_counts_ties_as_half(self):
        labels = np.array([0, 0, 1, 1])
        result = classification_metrics(labels, np.ones(4))
        self.assertEqual(result["auc"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)

    def test_score_orientation_and_threshold(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        threshold = best_threshold(labels, scores)
        result = classification_metrics(labels, scores, threshold)
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(classification_metrics(labels, -scores)["auc"], 0.0)

    def test_temporal_split_has_no_overlapping_windows(self):
        starts = np.arange(1000)
        train, test = partition(starts, seq_len=100)
        self.assertLessEqual(starts[train[-1]] + 100, starts[test[0]])
        self.assertFalse(np.intersect1d(train, test).size)


if __name__ == "__main__":
    unittest.main()
