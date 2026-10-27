import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, transform, unary_union


def entrance_barrier_nodes_within_radius(
    nodes: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> gpd.GeoDataFrame:
    """Return tagged entrance/barrier points within a metric radius of the map center."""
    nodes = nodes.copy()
    if nodes.crs is None:
        nodes = nodes.set_crs("EPSG:4326")
    nodes = nodes.to_crs("EPSG:4326")

    tag_columns = [column for column in ("entrance", "barrier") if column in nodes.columns]
    if not tag_columns:
        return nodes.iloc[0:0].copy()

    has_tag = pd.concat(
        [
            nodes[column].notna()
            & ~nodes[column].astype(str).str.strip().str.lower().isin(
                {"", "no", "false", "0"}
            )
            for column in tag_columns
        ],
        axis=1,
    ).any(axis=1)
    is_point = nodes.geometry.notna() & (nodes.geometry.geom_type == "Point")
    nodes = nodes.loc[has_tag & is_point].copy()
    if nodes.empty:
        return nodes

    metric_crs = nodes.estimate_utm_crs()
    project_center = Transformer.from_crs(
        "EPSG:4326", metric_crs, always_xy=True
    ).transform
    center = Point(*project_center(center_lon, center_lat))
    project_node = Transformer.from_crs(
        "EPSG:4326", metric_crs, always_xy=True
    ).transform
    within_radius = [
        transform(project_node, geometry).distance(center) <= radius_m
        for geometry in nodes.geometry
    ]
    return nodes.loc[pd.Series(within_radius, index=nodes.index)].copy()


def entrance_barrier_paths(
    nodes: gpd.GeoDataFrame,
    street_paths: list[LineString],
) -> list[LineString]:
    """Create shortest straight connectors from each filtered point to a street path."""
    paths = [
        path for path in street_paths
        if path is not None and not path.is_empty and path.length > 0
    ]
    if nodes.empty or not paths:
        return []

    metric_crs = nodes.estimate_utm_crs()
    project_path = Transformer.from_crs(
        "EPSG:4326", metric_crs, always_xy=True
    ).transform
    projected_nodes = [
        transform(project_path, point) for point in nodes.geometry
    ]
    projected_paths = [transform(project_path, path) for path in paths]
    street_union = unary_union(projected_paths)
    unproject_connector = Transformer.from_crs(
        metric_crs, "EPSG:4326", always_xy=True
    ).transform

    connectors = []
    for point in projected_nodes:
        start, end = nearest_points(point, street_union)
        if start.distance(end) <= 0.01:
            continue
        connector = LineString([start, end])
        connectors.append(transform(unproject_connector, connector))
    return connectors
