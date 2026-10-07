import unittest

import numpy as np
import pandas as pd

from utils.correlation_metrics import fisher_average_batch_correlations


class FisherAverageBatchCorrelationsTests(unittest.TestCase):
    def test_fisher_averages_within_batch_correlations(self):
        data = pd.DataFrame(
            {
                "upload_batch_id": ["first"] * 4 + ["second"] * 4,
                "x": [1, 2, 3, 4] * 2,
                "y": [1, 3, 2, 5, 5, 2, 3, 1],
            }
        )
        first_r = data.loc[data["upload_batch_id"] == "first", ["x", "y"]].corr().loc[
            "x", "y"
        ]
        second_r = data.loc[data["upload_batch_id"] == "second", ["x", "y"]].corr().loc[
            "x", "y"
        ]
        expected = np.tanh(
            (np.arctanh(first_r) + np.arctanh(second_r)) / 2
        )

        result = fisher_average_batch_correlations(data, ["x", "y"])

        self.assertAlmostEqual(result.loc["x", "y"], expected)
        self.assertEqual(result.loc["x", "y"], result.loc["y", "x"])
        self.assertEqual(result.loc["x", "x"], 1.0)

    def test_ignores_batches_with_insufficient_or_constant_data(self):
        data = pd.DataFrame(
            {
                "upload_batch_id": ["valid"] * 4 + ["short"] * 2 + ["constant"] * 4,
                "x": [1, 2, 3, 4, 1, 2, 1, 1, 1, 1],
                "y": [1, 3, 2, 5, 1, 2, 1, 2, 3, 4],
            }
        )
        expected = data.loc[data["upload_batch_id"] == "valid", ["x", "y"]].corr().loc[
            "x", "y"
        ]

        result = fisher_average_batch_correlations(data, ["x", "y"])

        self.assertAlmostEqual(result.loc["x", "y"], expected)

    def test_returns_missing_coefficient_when_no_batch_is_eligible(self):
        data = pd.DataFrame(
            {
                "upload_batch_id": ["only"] * 3,
                "x": [1, 1, 1],
                "y": [1, 2, 3],
            }
        )

        result = fisher_average_batch_correlations(data, ["x", "y"])

        self.assertTrue(pd.isna(result.loc["x", "y"]))


if __name__ == "__main__":
    unittest.main()
