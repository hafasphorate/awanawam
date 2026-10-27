import unittest

import geopandas as gpd
from shapely.geometry import LineString, Point

from utils.urban_paths import (
    entrance_barrier_nodes_within_radius,
    entrance_barrier_paths,
)


class UrbanPathsTests(unittest.TestCase):
    def test_radius_filter_keeps_tagged_points_and_excludes_distant_or_untagged(self):
        nodes = gpd.GeoDataFrame(
            {
                "entrance": ["yes", None, None],
                "barrier": [None, "gate", None],
            },
            geometry=[
                Point(103.8005, 1.3),
                Point(103.801, 1.3),
                Point(103.81, 1.3),
            ],
            crs="EPSG:4326",
        )

        result = entrance_barrier_nodes_within_radius(
            nodes, center_lat=1.3, center_lon=103.8, radius_m=500
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.index), {0, 1})

    def test_connector_uses_nearest_point_on_existing_path(self):
        nodes = gpd.GeoDataFrame(
            geometry=[Point(103.8005, 1.3003)],
            crs="EPSG:4326",
        )
        paths = [
            LineString([(103.8, 1.3), (103.801, 1.3)]),
            LineString([(103.8, 1.301), (103.801, 1.301)]),
        ]

        connectors = entrance_barrier_paths(nodes, paths)

        self.assertEqual(len(connectors), 1)
        connector = connectors[0]
        self.assertAlmostEqual(connector.coords[0][0], 103.8005, places=5)
        self.assertAlmostEqual(connector.coords[0][1], 1.3003, places=5)
        self.assertAlmostEqual(connector.coords[1][1], 1.3, places=5)
        self.assertLess(connector.length, 0.0005)

    def test_connector_is_omitted_when_point_already_lies_on_a_path(self):
        nodes = gpd.GeoDataFrame(
            geometry=[Point(103.8005, 1.3)],
            crs="EPSG:4326",
        )
        paths = [LineString([(103.8, 1.3), (103.801, 1.3)])]

        self.assertEqual(entrance_barrier_paths(nodes, paths), [])

    def test_nodes_tagged_as_no_entrance_are_excluded(self):
        nodes = gpd.GeoDataFrame(
            {"entrance": ["no"], "barrier": [None]},
            geometry=[Point(103.8001, 1.3)],
            crs="EPSG:4326",
        )

        result = entrance_barrier_nodes_within_radius(
            nodes, center_lat=1.3, center_lon=103.8, radius_m=500
        )

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
