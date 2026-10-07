import math
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def _positive_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _polygon_area(coordinates: Any) -> float:
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 3:
        return 0.0

    try:
        points = [(float(point[0]), float(point[1])) for point in coordinates]
    except (TypeError, ValueError, IndexError):
        return 0.0

    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def extract_analysis_area(
    data: Any, node_count: Optional[int] = None
) -> Tuple[Optional[float], Optional[str]]:
    """Return the uploaded analysis footprint area and the source used."""
    if not isinstance(data, dict):
        return None, None

    floorplan = data.get("floorplan")
    floorplan = floorplan if isinstance(floorplan, dict) else {}
    metadata = data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    for source, container in (
        ("uploaded analysis area", data),
        ("uploaded floorplan", floorplan),
        ("uploaded metadata", metadata),
    ):
        area = _positive_number(container.get("analysis_area"))
        if area is not None:
            return area, source

    selected_rooms = floorplan.get("selected_rooms", data.get("selected_rooms"))
    if isinstance(selected_rooms, (list, tuple)) and selected_rooms:
        first = selected_rooms[0]
        polygons = (
            [selected_rooms]
            if isinstance(first, (list, tuple))
            and len(first) >= 2
            and all(isinstance(value, (int, float)) for value in first[:2])
            else selected_rooms
        )
        polygon_areas = [_polygon_area(polygon) for polygon in polygons]
        total_area = sum(polygon_areas)
        if total_area > 0:
            return total_area, "selected room polygons"

    bounds = floorplan.get("bounds", data.get("bounds"))
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            min_x, min_y, max_x, max_y = map(float, bounds)
        except (TypeError, ValueError):
            return None, None
        bounds_area = (max_x - min_x) * (max_y - min_y)
        if math.isfinite(bounds_area) and bounds_area > 0:
            return bounds_area, "floorplan bounding box"

    analysis_settings = data.get("analysis_settings")
    analysis_settings = (
        analysis_settings if isinstance(analysis_settings, dict) else {}
    )
    grid_size = _positive_number(analysis_settings.get("grid_size_mm"))
    if node_count is not None and node_count > 0 and grid_size is not None:
        return node_count * grid_size**2, "estimated from VGA node count and grid spacing"

    return None, None


def add_comparison_metrics(
    data_frame: pd.DataFrame, analysis_area: Optional[float]
) -> pd.DataFrame:
    """Add normalized VGA metrics that can be derived from an uploaded node table."""
    result = data_frame.copy()

    mean_depth_column = next(
        (
            column
            for column in ("mean_depth", "visual_mean_depth")
            if column in result.columns
        ),
        None,
    )
    if mean_depth_column is not None:
        mean_depth = pd.to_numeric(result[mean_depth_column], errors="coerce")
        standard_deviation = mean_depth.std(ddof=0)
        if pd.notna(standard_deviation) and standard_deviation > 0:
            result["mean_depth_z_score"] = (
                mean_depth - mean_depth.mean()
            ) / standard_deviation
        else:
            result["mean_depth_z_score"] = mean_depth.where(mean_depth.isna(), 0.0)

    if analysis_area is not None and analysis_area > 0 and "isovist_area" in result:
        isovist_area = pd.to_numeric(result["isovist_area"], errors="coerce")
        result["relative_isovist_area_pct"] = (
            isovist_area / analysis_area * 100.0
        )

    integration_column = next(
        (
            column
            for column in ("integration", "visual_integration")
            if column in result.columns
        ),
        None,
    )
    if integration_column is not None or mean_depth_column is not None:
        nain = pd.Series(np.nan, index=result.index, dtype=float)
        floor_groups = (
            result.groupby("floor", dropna=False).indices.values()
            if "floor" in result.columns
            else [np.arange(len(result))]
        )

        for positions in floor_groups:
            node_count = len(positions)
            if node_count < 3:
                continue

            depth_normalizer = (
                2.0
                * (
                    node_count
                    * (math.log2((node_count + 2.0) / 3.0) - 1.0)
                    + 1.0
                )
                / ((node_count - 1.0) * (node_count - 2.0))
            )
            if depth_normalizer <= 0:
                continue

            rows = result.iloc[positions]
            if integration_column is not None:
                integration = pd.to_numeric(
                    rows[integration_column], errors="coerce"
                )
            else:
                mean_depth = pd.to_numeric(
                    rows[mean_depth_column], errors="coerce"
                )
                relative_asymmetry = (
                    2.0 * (mean_depth - 1.0) / (node_count - 2.0)
                )
                integration = 1.0 / relative_asymmetry.where(
                    relative_asymmetry > 0
                )

            nain.iloc[positions] = integration / depth_normalizer

        result["nain"] = nain

    return result
