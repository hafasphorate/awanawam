import unittest

import numpy as np
import pandas as pd

from utils.grid_normalization import cell_area_m2, resample_to_reference_grid


class GridNormalizationTests(unittest.TestCase):
    def test_cell_area_uses_millimetres(self):
        self.assertAlmostEqual(cell_area_m2(500), 0.25)
        self.assertAlmostEqual(cell_area_m2(1000), 1.0)

    def test_resampling_preserves_density_and_sums_counts(self):
        coords = [(x, y) for x in (0, 500) for y in (0, 500)]
        df = pd.DataFrame(coords, columns=["x", "y"])
        df["crowd_density"] = [4.0, 0.0, 0.0, 0.0]
        df["volume"] = [1.0, 0.0, 0.0, 0.0]
        df["speed_mps"] = [1.2, 9.0, 9.0, 9.0]
        df["degrees_from_spine"] = [350.0, 10.0, 10.0, 10.0]
        df["mean_depth"] = [2.0, 4.0, 4.0, 2.0]

        result, block_index = resample_to_reference_grid(df, 500)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(block_index), [0, 0, 0, 0])
        # 1 person on 1 m² = 1 person/m², matching the mean of four 0.25 m² cells.
        self.assertAlmostEqual(result.loc[0, "crowd_density"], 1.0)
        self.assertAlmostEqual(result.loc[0, "volume"], 1.0)
        # Flow metrics are weighted by where people actually are.
        self.assertAlmostEqual(result.loc[0, "speed_mps"], 1.2)
        self.assertAlmostEqual(result.loc[0, "degrees_from_spine"], 350.0)
        self.assertAlmostEqual(result.loc[0, "mean_depth"], 3.0)
        self.assertEqual(result.loc[0, "grid_size_mm"], 1000.0)

    def test_coarse_grid_is_unchanged(self):
        df = pd.DataFrame({"x": [0, 1000], "y": [0, 0], "crowd_density": [1.0, 2.0]})
        result, block_index = resample_to_reference_grid(df, 1000)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.array_equal(block_index, [0, 1]))


if __name__ == "__main__":
    unittest.main()
