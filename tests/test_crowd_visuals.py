import unittest

from utils.crowd_visuals import crowd_metric_label, deviation_ray_segments, direction_vector


class CrowdVisualsTests(unittest.TestCase):
    def test_metric_label_changes_display_without_changing_metric_key(self):
        metric_key = "peak_density_people_per_m2"

        self.assertEqual(
            crowd_metric_label(metric_key), "Highest Density People Per M2"
        )
        self.assertEqual(metric_key, "peak_density_people_per_m2")

    def test_direction_arrow_respects_spine_bearing_and_orientation(self):
        clockwise = direction_vector(0, 0, 90, 0, clockwise=True)
        anticlockwise = direction_vector(0, 0, 90, 0, clockwise=False)

        self.assertAlmostEqual(clockwise["x_end"], 1.0)
        self.assertAlmostEqual(clockwise["y_end"], 0.0)
        self.assertAlmostEqual(anticlockwise["x_end"], -1.0)
        self.assertAlmostEqual(anticlockwise["y_end"], 0.0)

    def test_direction_arrows_shrink_and_fade_with_uncertainty(self):
        certain = direction_vector(0, 0, 45, 0, standard_deviation=0)
        uncertain = direction_vector(0, 0, 45, 0, standard_deviation=90)

        certain_length = (
            certain["x_end"] ** 2 + certain["y_end"] ** 2
        ) ** 0.5
        uncertain_length = (
            uncertain["x_end"] ** 2 + uncertain["y_end"] ** 2
        ) ** 0.5
        self.assertLess(uncertain_length, certain_length)
        self.assertLess(uncertain["opacity"], certain["opacity"])

    def test_deviation_returns_four_rays_with_expected_bounds(self):
        segments = deviation_ray_segments(2, 3, 90, 0, 2)

        self.assertEqual(len(segments), 4)
        for x_start, y_start, x_end, y_end in segments:
            self.assertEqual((x_start, y_start), (2.0, 3.0))
            self.assertAlmostEqual(
                ((x_end - x_start) ** 2 + (y_end - y_start) ** 2) ** 0.5,
                2.0,
            )


if __name__ == "__main__":
    unittest.main()
