import unittest

import pandas as pd

from utils.derived_vga_metrics import add_comparison_metrics, extract_analysis_area


class DerivedVgaMetricsTests(unittest.TestCase):
    def test_analysis_area_uses_selected_room_geometry(self):
        data = {
            "floorplan": {
                "bounds": [0, 0, 100, 100],
                "selected_rooms": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            }
        }

        self.assertEqual(extract_analysis_area(data), (100.0, "selected room polygons"))

    def test_analysis_area_falls_back_to_floorplan_bounds(self):
        self.assertEqual(
            extract_analysis_area({"floorplan": {"bounds": [0, 0, 20, 10]}}),
            (200.0, "floorplan bounding box"),
        )

    def test_adds_relative_area_z_score_and_normalized_integration(self):
        source = pd.DataFrame(
            {
                "isovist_area": [25.0, 50.0, 75.0],
                "mean_depth": [1.0, 2.0, 3.0],
                "integration": [1.0, 2.0, 3.0],
            }
        )

        result = add_comparison_metrics(source, analysis_area=100.0)

        self.assertEqual(
            result["relative_isovist_area_pct"].tolist(), [25.0, 50.0, 75.0]
        )
        self.assertAlmostEqual(result["mean_depth_z_score"].mean(), 0.0)
        self.assertAlmostEqual(result["mean_depth_z_score"].std(ddof=0), 1.0)
        self.assertTrue(result["nain"].notna().all())

    def test_constant_mean_depth_has_zero_z_scores_and_can_derive_nain(self):
        source = pd.DataFrame({"mean_depth": [2.0, 2.0, 2.0]})

        result = add_comparison_metrics(source, analysis_area=None)

        self.assertEqual(result["mean_depth_z_score"].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(result["nain"].notna().all())


if __name__ == "__main__":
    unittest.main()
