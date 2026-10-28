"""Helpers that make VGA/crowd datasets with different node intervals comparable.

Intensive quantities (density in people/m², speed, angles, isovist metrics, NAIN)
do not depend on the node interval. Extensive quantities (people per cell,
person-frames) scale with cell area. To compare datasets, every dataset is
resampled onto one reference grid: intensive values are averaged (weighted by
crowd density for flow metrics), extensive values are summed.
"""
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.derived_vga_metrics import _infer_grid_size, _positive_number

REFERENCE_GRID_MM = 1000.0
MM_TO_M = 0.001

_COORDINATE_COLUMNS = {"x", "y"}
_IDENTIFIER_COLUMNS = {"node_id", "grid_node_idx", "id", "index", "grid_size_mm"}
_DENSITY_COLUMNS = ("crowd_density", "density", "average_density_people_per_m2")


def cell_area_m2(grid_size_mm: float) -> float:
    """Area in m² of a square grid cell whose side is given in millimetres."""
    return (float(grid_size_mm) * MM_TO_M) ** 2


def infer_grid_interval_mm(
    node_data: Optional[pd.DataFrame], configured: object = None
) -> Optional[float]:
    """Return the configured node interval, else infer it from node coordinates."""
    interval = _positive_number(configured)
    return interval if interval is not None else _infer_grid_size(node_data)


def _is_angular(name: str) -> bool:
    name = name.lower()
    if "deviation" in name:
        return False
    return any(word in name for word in ("direction", "heading", "bearing")) or (
        "degrees_from_spine" in name
    )


def _is_extensive(name: str) -> bool:
    name = name.lower()
    if any(word in name for word in ("density", "speed", "per_m2", "pct", "ratio")):
        return False
    return any(
        word in name
        for word in ("volume", "count", "person_frames", "people", "pedestrian")
    )


def _flow_weights(df: pd.DataFrame) -> pd.Series:
    for column in _DENSITY_COLUMNS:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(1.0, index=df.index)


def resample_to_reference_grid(
    df: pd.DataFrame,
    source_grid_mm: Optional[float],
    reference_grid_mm: float = REFERENCE_GRID_MM,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Aggregate nodes into reference-size blocks.

    Returns the resampled table and, for every input row, the position of its
    block in that table. A source grid that is not finer than the reference grid
    is returned unchanged because it cannot be refined.
    """
    identity = np.arange(len(df))
    if (
        source_grid_mm is None
        or source_grid_mm >= reference_grid_mm * (1 - 1e-9)
        or not _COORDINATE_COLUMNS.issubset(df.columns)
        or df.empty
    ):
        result = df.copy()
        if source_grid_mm is not None:
            result["grid_size_mm"] = float(source_grid_mm)
        return result.reset_index(drop=True), identity

    data = df.reset_index(drop=True)
    x = pd.to_numeric(data["x"], errors="coerce")
    y = pd.to_numeric(data["y"], errors="coerce")
    # A small epsilon keeps nodes on a block edge in the block they start.
    block_x = np.floor((x - x.min()) / reference_grid_mm + 1e-9)
    block_y = np.floor((y - y.min()) / reference_grid_mm + 1e-9)
    ids = data.assign(_bx=block_x, _by=block_y).groupby(["_bx", "_by"]).ngroup()
    group_count = int(ids.max()) + 1

    weights = _flow_weights(data)
    weight_sum = weights.groupby(ids).sum()
    columns = {}
    columns["x"] = x.groupby(ids).mean()
    columns["y"] = y.groupby(ids).mean()

    numeric_columns = [
        column
        for column in data.select_dtypes(include=np.number).columns
        if column not in _COORDINATE_COLUMNS and column.lower() not in _IDENTIFIER_COLUMNS
    ]
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if _is_extensive(column):
            columns[column] = values.groupby(ids).sum(min_count=1)
        elif _is_angular(column):
            radians = np.radians(values)
            valid = values.notna()
            w = weights.where(valid, 0.0)
            # Fall back to equal weights where no crowd was observed in the block.
            w = w.where(weight_sum.reindex(ids).to_numpy() > 0, valid.astype(float))
            sin_sum = (np.sin(radians).fillna(0.0) * w).groupby(ids).sum()
            cos_sum = (np.cos(radians).fillna(0.0) * w).groupby(ids).sum()
            angle = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
            columns[column] = angle.where((sin_sum != 0) | (cos_sum != 0))
        elif column in _DENSITY_COLUMNS or "density" in column.lower():
            columns[column] = values.groupby(ids).mean()
        elif any(word in column.lower() for word in ("speed", "deviation")):
            valid = values.notna()
            w = weights.where(valid, 0.0)
            w = w.where(weight_sum.reindex(ids).to_numpy() > 0, valid.astype(float))
            numerator = (values.fillna(0.0) * w).groupby(ids).sum()
            denominator = w.groupby(ids).sum()
            columns[column] = (numerator / denominator.where(denominator > 0))
        else:
            columns[column] = values.groupby(ids).mean()

    resampled = pd.DataFrame(columns).reindex(range(group_count)).reset_index(drop=True)
    resampled["grid_size_mm"] = float(reference_grid_mm)
    return resampled, ids.to_numpy()
