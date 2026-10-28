import json
import hashlib
from io import BytesIO
import tempfile
import time
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from supabase import Client, create_client
from shapely.geometry import MultiPoint, Point, LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
import streamlit as st
import shapely
from utils.navigation import render_home_button
from utils.export_figures import prepare_export_figure
from utils.plotly_floorplan import configure_floorplan_figure
from utils.dataset_locks import unlocked_records
from utils.derived_vga_metrics import add_comparison_metrics, extract_analysis_area
from utils.grid_normalization import (
    REFERENCE_GRID_MM,
    infer_grid_interval_mm,
    resample_to_reference_grid,
)
from utils.crowd_visuals import (
    add_deviation_rays,
    add_direction_arrows,
    crowd_metric_label,
)
from utils.density_visuals import (
    DENSITY_COLORSCALE,
    DENSITY_SCALE_MAX,
)

from utils.vga_engine import (
    compute_isovist_metrics,
    extract_dxf_walls,
    generate_isovist_polygon,
    process_cad_file,
)

st.set_page_config(page_title="Module 2: Visibility Graph Analysis", layout="wide")
render_home_button()

# Force '+' crosshair cursor on interactive Plotly floorplan canvas
st.markdown(
    """
    <style>
    .js-plotly-plot .plotly .draglayer,
    .js-plotly-plot .plotly .nsewdrag,
    .js-plotly-plot .plotly .cursor-crosshair, 
    .js-plotly-plot .plotly .drag {
        cursor: crosshair !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Module 2: Visibility Graph Analysis (VGA)")
st.markdown(
    "Upload a CAD floorplan (**DXF or DWG**) or a **saved JSON session**. Hover with the **`+` crosshair**, click directly inside any room or corridor zone to highlight it in green, and run spatial metrics strictly for selected areas."
)

# Sidebar Settings
st.sidebar.header("Analysis Settings")
grid_size = st.sidebar.number_input(
    "Grid Dimension (mm)", min_value=200, max_value=5000, value=1000, step=100
)
ray_step = st.sidebar.slider(
    "Ray Angle Step (Degrees)", min_value=1.0, max_value=15.0, value=2.0, step=0.5
)
ray_count = int(360 / ray_step)
door_snap_dist = st.sidebar.slider(
    "Doorway/Corridor Auto-Close Gap (mm)", min_value=100, max_value=3000, value=1200, step=100
)

uploaded_file = None


@st.cache_data
def extract_enclosed_rooms(_wall_lines, snap_distance=1200):
    """Reconstructs enclosed room polygons and corridor spaces with automatic interior hole detection."""
    lines = list(_wall_lines)

    endpoints = sorted(
        {
            tuple(coords)
            for line in lines
            for coords in (line.coords[0], line.coords[-1])
        }
    )
    endpoint_gaps = []
    for i, (x1, y1) in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            x2, y2 = endpoints[j]
            distance = math.hypot(x2 - x1, y2 - y1)
            if 10.0 < distance <= snap_distance:
                endpoint_gaps.append((distance, i, j))

    endpoint_gaps.sort()
    matched_endpoints = set()
    closing_lines = []
    for _, i, j in endpoint_gaps:
        if i not in matched_endpoints and j not in matched_endpoints:
            matched_endpoints.update((i, j))
            closing_lines.append(LineString([endpoints[i], endpoints[j]]))

    merged_walls = unary_union(lines + closing_lines)
    raw_polygons = list(polygonize(merged_walls))

    valid_polygons = [p for p in raw_polygons if p.is_valid and p.area > 100.0]
    return valid_polygons


def compute_graph_topology_with_progress(vga_results, isovist_polys, status_container, progress_bar):
    """Computes inter-isovist graph integration and entropy with active progress tracking."""
    num_nodes = len(vga_results)
    if num_nodes == 0:
        return vga_results

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    start_time = time.time()
    xs = np.array([r["x"] for r in vga_results], dtype=float)
    ys = np.array([r["y"] for r in vga_results], dtype=float)

    for i in range(num_nodes):
        # Vectorised point-in-polygon test of every node against isovist i.
        adj_matrix[i] = shapely.contains_xy(isovist_polys[i], xs, ys)

        completed = i + 1
        progress_ratio = completed / num_nodes
        elapsed = time.time() - start_time
        avg_per_node = elapsed / completed
        remaining_secs = int((num_nodes - completed) * avg_per_node)
        mins, secs = divmod(remaining_secs, 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        if completed % 5 == 0 or completed == num_nodes:
            progress_bar.progress(progress_ratio)
            status_container.markdown(
                f" **Computing Graph Topology (Integration & Entropy)...** Node {completed}/{num_nodes} ({int(progress_ratio * 100)}%) | **Est. remaining:** `{time_str}`"
            )

    # Two nodes are linked when either can see the other (blockwise to avoid a full n² temporary).
    block = 1000
    for start in range(0, num_nodes, block):
        stop = min(start + block, num_nodes)
        adj_matrix[start:stop] |= adj_matrix[:, start:stop].T.copy()
    np.fill_diagonal(adj_matrix, False)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    graph = csr_matrix(adj_matrix)
    del adj_matrix
    status_container.markdown(" **Computing visual depth between nodes...**")
    batch = 250
    for start in range(0, num_nodes, batch):
        indices = np.arange(start, min(start + batch, num_nodes))
        depth_rows = dijkstra(graph, directed=False, unweighted=True, indices=indices)
        for row_number, i in enumerate(indices):
            depths = depth_rows[row_number]
            depths = depths[np.isfinite(depths)]
            total_depth = float(depths.sum())
            reachable_nodes = len(depths)

            if reachable_nodes > 1:
                mean_depth = total_depth / (reachable_nodes - 1)
                integration = 1.0 / (2.0 * (mean_depth - 1.0) / max(1, reachable_nodes - 2)) if mean_depth > 1 else 0.0
            else:
                mean_depth = 0.0
                integration = 0.0

            vga_results[i]["mean_depth"] = round(mean_depth, 3)
            vga_results[i]["integration"] = round(integration, 3)
        progress_bar.progress(min(1.0, (indices[-1] + 1) / num_nodes))

    return vga_results


def poly_to_svg_path(poly):
    """Converts a Shapely Polygon into a properly oriented SVG path string so interior holes remain transparent."""
    oriented_poly = orient(poly, sign=1.0)

    x_poly, y_poly = oriented_poly.exterior.xy
    coords = list(zip(x_poly, y_poly))
    path = f"M {coords[0][0]},{coords[0][1]} "
    for x, y in coords[1:]:
        path += f"L {x},{y} "
    path += "Z "

    for interior in oriented_poly.interiors:
        ix, iy = interior.xy
        icoords = list(zip(ix, iy))
        path += f"M {icoords[0][0]},{icoords[0][1]} "
        for x, y in icoords[1:]:
            path += f"L {x},{y} "
        path += "Z "

    return path


def render_interactive_floorplan(
    wall_lines, bounds, selected_polys=None, selectable_zones=None
):
    """Builds interactive Plotly figure configured with custom crosshair cursor."""
    fig = go.Figure()
    minx, miny, maxx, maxy = bounds

    if selected_polys:
        for idx, poly in enumerate(selected_polys):
            svg_path = poly_to_svg_path(poly)
            fig.add_shape(
                type="path",
                path=svg_path,
                fillcolor="rgba(0, 230, 118, 0.45)",
                line=dict(color="#00FF66", width=3),
                layer="below",
            )

    wall_x, wall_y = [], []
    for line in wall_lines:
        x, y = line.xy
        wall_x.extend([x[0], x[1], None])
        wall_y.extend([y[0], y[1], None])

    fig.add_trace(
        go.Scatter(
            x=wall_x,
            y=wall_y,
            mode="lines",
            line=dict(color="#00ADB5", width=1.5),
            hoverinfo="none",
            showlegend=False,
        )
    )

    plan_area = (maxx - minx) * (maxy - miny)
    grid_step = max(200, math.sqrt(plan_area / 25000))
    gx = np.arange(minx, maxx, grid_step)
    gy = np.arange(miny, maxy, grid_step)
    g_xx, g_yy = np.meshgrid(gx, gy)

    fig.add_trace(
        go.Scatter(
            x=g_xx.flatten(),
            y=g_yy.flatten(),
            mode="markers",
            marker=dict(size=12, color="rgba(0, 0, 0, 0.001)"),
            hoverinfo="none",
            showlegend=False,
            name="sensor_grid",
        )
    )

    if selectable_zones:
        zone_points = [zone.representative_point() for zone in selectable_zones]
        fig.add_trace(
            go.Scatter(
                x=[point.x for point in zone_points],
                y=[point.y for point in zone_points],
                customdata=list(range(len(selectable_zones))),
                mode="markers",
                marker=dict(
                    size=10,
                    color="rgba(255, 193, 7, 0.7)",
                    line=dict(color="white", width=1),
                ),
                hovertemplate="Zone target<extra></extra>",
                showlegend=False,
                name="zone_targets",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            title="X (mm)",
            showgrid=True,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(title="Y (mm)", showgrid=True, zeroline=False),
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        clickmode="event+select",
        dragmode=False,
        hovermode="closest",
    )
    configure_floorplan_figure(fig)
    return fig


def render_vga_heatmap_with_underlay(df, metric_column, wall_lines):
    """Renders VGA metric heatmap with original CAD floorplan wall lines underlaid."""
    fig = go.Figure()
    metric_label = crowd_metric_label(metric_column)

    # 1. Underlay Wall Lines
    wall_x, wall_y = [], []
    for line in wall_lines:
        x, y = line.xy
        wall_x.extend([x[0], x[1], None])
        wall_y.extend([y[0], y[1], None])

    fig.add_trace(
        go.Scatter(
            x=wall_x,
            y=wall_y,
            mode="lines",
            line=dict(color="#666666", width=1.5),
            hoverinfo="none",
            showlegend=False,
            name="CAD Walls",
        )
    )

    # 2. Heatmap Points Overlay
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=8,
                color=df[metric_column],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=metric_label),
                opacity=0.9,
            ),
            text=[f"{metric_label}: {v}" for v in df[metric_column]],
            hoverinfo="x+y+text",
            name="VGA Data",
        )
    )

    fig.update_layout(
        title=dict(text=f"Spatial Map: {metric_label}", x=0.01),
        template="plotly_dark",
        xaxis=dict(
            title="X (mm)",
            showgrid=True,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(title="Y (mm)", showgrid=True, zeroline=False),
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    configure_floorplan_figure(fig)
    return fig


def render_cluster_map(df, wall_lines, selected_group=None, spine_result=None):
    """Render VGA points by cluster while retaining the CAD wall underlay."""
    fig = go.Figure()

    wall_x, wall_y = [], []
    for line in wall_lines:
        x, y = line.xy
        wall_x.extend([x[0], x[1], None])
        wall_y.extend([y[0], y[1], None])

    fig.add_trace(
        go.Scatter(
            x=wall_x,
            y=wall_y,
            mode="lines",
            line=dict(color="#666666", width=1.5),
            hoverinfo="none",
            showlegend=False,
            name="CAD Walls",
        )
    )

    groups = sorted(df["cluster"].unique())
    palette = px.colors.qualitative.Safe
    for index, group in enumerate(groups):
        group_df = df[df["cluster"] == group]
        is_selected = selected_group is None or group == selected_group
        fig.add_trace(
            go.Scatter(
                x=group_df["x"],
                y=group_df["y"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=palette[index % len(palette)] if is_selected else "#777777",
                    opacity=0.95 if is_selected else 0.22,
                    line=dict(width=0.5, color="#111111"),
                ),
                text=[f"Group {group}<br>Metrics: {row}" for row in group_df["cluster"]],
                hovertemplate="x=%{x}<br>y=%{y}<br>%{text}<extra></extra>",
                name=f"Group {group}",
            )
        )

    fig.update_layout(
        title="VGA Metric Clusters",
        template="plotly_dark",
        xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (mm)"),
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    configure_floorplan_figure(fig)
    if spine_result is not None:
        add_cluster_spine_overlay(fig, spine_result)
    return fig


def connected_grid_zones(points, grid_df=None):
    """Return edge-connected grid zones ranked by size and mean integration."""
    integration_lookup = {}
    if "integration" in points.columns:
        integration_lookup = (
            points.assign(
                x=pd.to_numeric(points["x"], errors="coerce"),
                y=pd.to_numeric(points["y"], errors="coerce"),
                integration=pd.to_numeric(points["integration"], errors="coerce"),
            )
            .dropna(subset=["x", "y"])
            .drop_duplicates(subset=["x", "y"])
            .set_index(["x", "y"])["integration"]
            .to_dict()
        )
    points = points[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    if points.empty:
        raise ValueError("The selected group has no usable x/y cells.")

    grid_points = (grid_df if grid_df is not None else points)[["x", "y"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().drop_duplicates()
    if grid_points.empty:
        grid_points = points
    grid_x_steps = np.diff(np.sort(grid_points["x"].unique()))
    grid_y_steps = np.diff(np.sort(grid_points["y"].unique()))
    usable_x_steps = grid_x_steps[grid_x_steps > 0]
    usable_y_steps = grid_y_steps[grid_y_steps > 0]
    fallback_step = min(
        [float(np.min(steps)) for steps in (usable_x_steps, usable_y_steps) if len(steps)]
        or [1.0]
    )
    x_step = float(np.min(usable_x_steps)) if len(usable_x_steps) else fallback_step
    y_step = float(np.min(usable_y_steps)) if len(usable_y_steps) else fallback_step
    x_origin, y_origin = float(grid_points["x"].min()), float(grid_points["y"].min())
    point_lookup = {
        (
            round((float(row.x) - x_origin) / x_step),
            round((float(row.y) - y_origin) / y_step),
        ): (float(row.x), float(row.y))
        for row in points.itertuples()
    }
    point_keys = set(point_lookup)

    components = []
    remaining = set(point_keys)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current_x, current_y = stack.pop()
            for neighbor in (
                (current_x - 1, current_y),
                (current_x + 1, current_y),
                (current_x, current_y - 1),
                (current_x, current_y + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    zones = []
    cluster_center = points[["x", "y"]].mean().to_numpy(dtype=float)
    for component in components:
        exposed_edges = sum(
            (y_step if offset_x else x_step)
            for x_key, y_key in component
            for offset_x, offset_y in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            )
            if (x_key + offset_x, y_key + offset_y) not in component
        )
        area = len(component) * x_step * y_step
        compactness = 4 * math.pi * area / (exposed_edges**2)
        selected_points = np.array([point_lookup[key] for key in sorted(component)], dtype=float)
        integration_values = [
            integration_lookup.get(point_lookup[key], np.nan) for key in component
        ]
        mean_integration = float(np.nanmean(integration_values)) if not np.all(
            np.isnan(integration_values)
        ) else 1.0
        zone_center = selected_points.mean(axis=0)
        center_distance = float(np.linalg.norm(zone_center - cluster_center))
        zones.append(
            {
                "component": component,
                "selected_points": selected_points,
                "size": len(component),
                "compactness": compactness,
                "mean_integration": mean_integration,
                "score": len(component) * mean_integration,
                "center_distance": center_distance,
                "tie_break": min(component),
            }
        )

    selected_zone = max(
        zones,
        key=lambda zone: (
            zone["score"],
            zone["size"],
            zone["mean_integration"],
            -zone["center_distance"],
            tuple(-value for value in zone["tie_break"]),
        ),
    )
    return zones, selected_zone, x_step, y_step


def calculate_cluster_spine(cluster_df, count_df=None, grid_df=None, selected_zone=None):
    """Calculate the spine through the globally selected connected zone."""
    points = cluster_df[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    if points.empty:
        raise ValueError("The selected group has no usable x/y cells.")

    zones, local_zone, x_step, y_step = connected_grid_zones(cluster_df, grid_df)
    selected_zone = selected_zone or local_zone
    selected_points = selected_zone["selected_points"]
    largest_component_size = max(zone["size"] for zone in zones)
    point_geometry = MultiPoint(selected_points.tolist())
    minimum_rectangle = point_geometry.minimum_rotated_rectangle
    if minimum_rectangle.geom_type != "Polygon":
        minimum_rectangle = point_geometry.buffer(max(x_step, y_step) / 2).minimum_rotated_rectangle
    rectangle_coords = list(minimum_rectangle.exterior.coords)
    rectangle_edges = [
        (rectangle_coords[index], rectangle_coords[index + 1])
        for index in range(len(rectangle_coords) - 1)
    ]
    longest_edge = max(
        rectangle_edges,
        key=lambda edge: math.hypot(
            edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]
        ),
    )
    edge_start, edge_end = longest_edge
    edge_dx = edge_end[0] - edge_start[0]
    edge_dy = edge_end[1] - edge_start[1]
    edge_length = math.hypot(edge_dx, edge_dy)
    center_x, center_y = minimum_rectangle.centroid.x, minimum_rectangle.centroid.y
    direction_x, direction_y = edge_dx / edge_length, edge_dy / edge_length
    count_points = (count_df if count_df is not None else cluster_df)[["x", "y"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().to_numpy(dtype=float)
    spine_start = (center_x - direction_x * edge_length / 2, center_y - direction_y * edge_length / 2)
    spine_end = (center_x + direction_x * edge_length / 2, center_y + direction_y * edge_length / 2)
    side_tolerance = max(min(x_step, y_step) * 0.05, 1e-9)
    end_values = direction_x * (count_points[:, 0] - center_x) + direction_y * (count_points[:, 1] - center_y)
    end_positive_count = int(np.sum(end_values > side_tolerance))
    end_negative_count = int(np.sum(end_values < -side_tolerance))
    end_boundary_count = int(np.sum(np.abs(end_values) <= side_tolerance))
    if end_positive_count > end_negative_count:
        point_zero, point_one = spine_end, spine_start
        point_zero_count, point_one_count = end_positive_count, end_negative_count
    else:
        point_zero, point_one = spine_start, spine_end
        point_zero_count, point_one_count = end_negative_count, end_positive_count
    spine_direction_length = math.hypot(
        point_one[0] - point_zero[0],
        point_one[1] - point_zero[1],
    )
    spine_direction_x = (point_one[0] - point_zero[0]) / spine_direction_length
    spine_direction_y = (point_one[1] - point_zero[1]) / spine_direction_length
    side_values = (
        spine_direction_x * (count_points[:, 1] - center_y)
        - spine_direction_y * (count_points[:, 0] - center_x)
    )
    side_labels = np.where(
        side_values > side_tolerance,
        "Side 1",
        np.where(side_values < -side_tolerance, "Side 2", "On spine"),
    )
    positive_count = int(np.sum(side_labels == "Side 1"))
    negative_count = int(np.sum(side_labels == "Side 2"))
    boundary_count = int(np.sum(side_labels == "On spine"))
    if negative_count > positive_count:
        spine_direction = "C"
    elif positive_count > negative_count:
        spine_direction = "AC"
    else:
        spine_direction = "Balanced"
    spine_angle = math.degrees(
        math.atan2(point_one[1] - point_zero[1], point_one[0] - point_zero[0])
    ) % 360
    return {
        "selected_points": selected_points,
        "rectangle_coords": rectangle_coords,
        "spine_start": spine_start,
        "spine_end": spine_end,
        "point_zero": point_zero,
        "point_one": point_one,
        "spine_angle": spine_angle,
        "spine_direction": spine_direction,
        "clockwise_side_count": negative_count,
        "anticlockwise_side_count": positive_count,
        "count_points": count_points,
        "side_labels": side_labels,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "boundary_count": boundary_count,
        "end_positive_count": end_positive_count,
        "end_negative_count": end_negative_count,
        "end_boundary_count": end_boundary_count,
        "point_zero_count": point_zero_count,
        "point_one_count": point_one_count,
        "component_count": len(zones),
        "largest_component_size": largest_component_size,
        "selected_zone_size": selected_zone["size"],
        "selected_zone_compactness": selected_zone["compactness"],
        "selected_zone_mean_integration": selected_zone["mean_integration"],
        "selected_zone_score": selected_zone["score"],
    }


def calculate_centrality_spine(
    df,
    top_fraction=0.15,
    measure="betweenness",
    max_exact_nodes=600,
):
    """Isolate a continuous spine from the top integration cells using graph centrality.

    The highest-integration cells are linked to their 8-connected grid neighbours, the
    strongest connected region is scored with betweenness or closeness centrality, and the
    spine is the lowest-cost path between that region's two extremities, where stepping onto
    a high-centrality cell is cheap. The path then feeds the same bounding-box/side-count
    analysis as the zone-based spine so the results render identically.
    """
    import networkx as nx

    required = {"x", "y", "integration"}
    if not required.issubset(df.columns):
        raise ValueError("Graph-based spine calculation needs x, y and integration columns.")
    if not 0 < top_fraction <= 1:
        raise ValueError("The top fraction must be between 0 and 1.")

    data = df.copy()
    for column in ("x", "y", "integration"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["x", "y", "integration"])
    data = data.drop_duplicates(subset=["x", "y"])
    if len(data) < 2:
        raise ValueError("There are not enough cells with integration values.")

    x_steps = np.diff(np.sort(data["x"].unique()))
    y_steps = np.diff(np.sort(data["y"].unique()))
    steps = [float(steps[steps > 0].min()) for steps in (x_steps, y_steps) if np.any(steps > 0)]
    if not steps:
        raise ValueError("The grid cells do not span more than one position.")
    x_step = float(x_steps[x_steps > 0].min()) if np.any(x_steps > 0) else min(steps)
    y_step = float(y_steps[y_steps > 0].min()) if np.any(y_steps > 0) else min(steps)
    x_origin, y_origin = float(data["x"].min()), float(data["y"].min())

    threshold = float(data["integration"].quantile(1.0 - top_fraction))
    candidates = data[data["integration"] >= threshold].copy()
    candidates["gx"] = np.round((candidates["x"] - x_origin) / x_step).astype(int)
    candidates["gy"] = np.round((candidates["y"] - y_origin) / y_step).astype(int)

    graph = nx.Graph()
    for gx, gy, x_value, y_value, integration_value in zip(
        candidates["gx"], candidates["gy"], candidates["x"], candidates["y"], candidates["integration"]
    ):
        graph.add_node(
            (int(gx), int(gy)), x=float(x_value), y=float(y_value), integration=float(integration_value)
        )
    for gx, gy in list(graph.nodes):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            neighbor = (gx + dx, gy + dy)
            if neighbor in graph:
                graph.add_edge(
                    (gx, gy), neighbor, distance=math.hypot(dx * x_step, dy * y_step)
                )

    components = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)]
    components = [component for component in components if component.number_of_nodes() >= 2]
    if not components:
        raise ValueError(
            "The top-integration cells are not adjacent to each other; try a larger top percentage."
        )
    region = max(
        components,
        key=lambda component: (
            component.number_of_nodes()
            * np.mean([attrs["integration"] for _, attrs in component.nodes(data=True)]),
            component.number_of_nodes(),
        ),
    )

    if measure == "closeness":
        centrality = nx.closeness_centrality(region, distance="distance")
    else:
        sample = region.number_of_nodes() > max_exact_nodes
        centrality = nx.betweenness_centrality(
            region,
            k=max_exact_nodes if sample else None,
            weight="distance",
            seed=42,
            normalized=True,
        )
    centrality_values = np.array(list(centrality.values()), dtype=float)
    centrality_range = np.ptp(centrality_values)
    normalized_centrality = {
        node: (value - centrality_values.min()) / centrality_range if centrality_range > 0 else 1.0
        for node, value in centrality.items()
    }
    for first, second, attrs in region.edges(data=True):
        mean_centrality = (normalized_centrality[first] + normalized_centrality[second]) / 2
        attrs["cost"] = attrs["distance"] / (0.05 + mean_centrality)

    anchor = max(normalized_centrality, key=normalized_centrality.get)
    end_a = max(
        nx.single_source_dijkstra_path_length(region, anchor, weight="distance").items(),
        key=lambda item: item[1],
    )[0]
    end_b = max(
        nx.single_source_dijkstra_path_length(region, end_a, weight="distance").items(),
        key=lambda item: item[1],
    )[0]
    path_nodes = nx.shortest_path(region, end_a, end_b, weight="cost")
    path_points = np.array(
        [[region.nodes[node]["x"], region.nodes[node]["y"]] for node in path_nodes], dtype=float
    )
    path_integration = np.array([region.nodes[node]["integration"] for node in path_nodes])
    path_df = pd.DataFrame(
        {"x": path_points[:, 0], "y": path_points[:, 1], "integration": path_integration}
    )
    zone = {
        "selected_points": path_points,
        "size": len(path_points),
        "compactness": 0.0,
        "mean_integration": float(path_integration.mean()),
        "score": float(len(path_points) * path_integration.mean()),
    }
    result = calculate_cluster_spine(path_df, count_df=data, grid_df=data, selected_zone=zone)
    result.update(
        {
            "path_coords": path_points,
            "candidate_points": candidates[["x", "y"]].to_numpy(dtype=float),
            "top_fraction": top_fraction,
            "integration_threshold": threshold,
            "centrality_measure": measure,
            "candidate_count": len(candidates),
            "region_size": region.number_of_nodes(),
            "path_mean_centrality": float(
                np.mean([normalized_centrality[node] for node in path_nodes])
            ),
        }
    )
    return result


def add_cluster_spine_overlay(fig, spine_result, group_label=None, spine_color="#FFD166"):
    """Overlay the selected cluster bounding box, spine, and side counts."""
    if not spine_result or not {
        "rectangle_coords",
        "count_points",
        "side_labels",
        "point_zero",
        "point_one",
        "point_zero_count",
        "point_one_count",
    }.issubset(spine_result):
        return fig
    rectangle_coords = spine_result["rectangle_coords"]
    fig.add_shape(
        type="path",
        path="M " + " L ".join(f"{point[0]},{point[1]}" for point in rectangle_coords) + " Z",
        line=dict(color="#00E5FF", width=3, dash="dash"),
        fillcolor="rgba(0, 229, 255, 0.08)",
        layer="above",
    )
    side_colors = {"Side 1": "#FF6B6B", "Side 2": "#4D96FF", "On spine": "#FFFFFF"}
    selected_points = spine_result["count_points"]
    side_labels = spine_result["side_labels"]
    for side_label in ("Side 1", "Side 2", "On spine"):
        side_points = selected_points[side_labels == side_label]
        if len(side_points) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=side_points[:, 0],
                y=side_points[:, 1],
                mode="markers",
                marker=dict(size=12, color=side_colors[side_label], line=dict(color="#111111", width=1)),
                name=f"{group_label} {side_label}" if group_label else side_label,
                hovertemplate=(
                    f"{group_label + ' ' if group_label else ''}{side_label}"
                    "<br>x=%{x}<br>y=%{y}<extra></extra>"
                ),
            )
        )
    point_zero = spine_result.get("point_zero", spine_result["spine_start"])
    point_one = spine_result.get("point_one", spine_result["spine_end"])
    if "candidate_points" in spine_result:
        candidate_points = np.asarray(spine_result["candidate_points"])
        fig.add_trace(
            go.Scatter(
                x=candidate_points[:, 0],
                y=candidate_points[:, 1],
                mode="markers",
                marker=dict(size=7, color="#00E5FF", symbol="diamond-open"),
                name=f"Top {spine_result['top_fraction'] * 100:.0f}% integration cells",
                hovertemplate="x=%{x}<br>y=%{y}<extra>Top integration cell</extra>",
            )
        )
    if "path_coords" in spine_result:
        path_coords = np.asarray(spine_result["path_coords"])
        fig.add_trace(
            go.Scatter(
                x=path_coords[:, 0],
                y=path_coords[:, 1],
                mode="lines+markers",
                line=dict(color="#FFFFFF", width=3),
                marker=dict(size=8, color="#FFFFFF", line=dict(color="#111111", width=1)),
                name="Centrality path",
                hovertemplate="x=%{x}<br>y=%{y}<extra>Centrality path</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[point_zero[0], point_one[0]],
            y=[point_zero[1], point_one[1]],
            mode="lines+markers+text",
            line=dict(color=spine_color, width=6),
            marker=dict(size=12, color=spine_color),
            text=["Point 0", "Point 1"],
            textposition="top center",
            name=f"{group_label} spine" if group_label else "Calculated spine",
            hovertemplate=f"Spine angle: {spine_result['spine_angle']:.1f}°<extra></extra>",
        )
    )
    fig.add_annotation(
        x=rectangle_coords[0][0],
        y=rectangle_coords[0][1],
        xanchor="left",
        yanchor="top",
        text=(
            f"{group_label + '<br>' if group_label else ''}"
            f"Spine angle: {spine_result['spine_angle']:.1f}°<br>"
            f"Selected zone: {spine_result['selected_zone_size']} cells | "
            f"Mean integration: {spine_result['selected_zone_mean_integration']:.3f} | "
            f"Score: {spine_result['selected_zone_score']:.3f}<br>"
            f"Side 1 (red): {spine_result['positive_count']} cells | "
            f"Side 2 (blue): {spine_result['negative_count']} cells<br>"
            f"On spine: {spine_result['boundary_count']} cells<br>"
            f"Point 0 end: {spine_result['point_zero_count']} cells | "
            f"Point 1 end: {spine_result['point_one_count']} cells"
        ),
        showarrow=False,
        bgcolor="rgba(17,17,17,0.85)",
        bordercolor="#FFD166",
        borderwidth=1,
        font=dict(color="#FFFFFF"),
    )
    return fig


def render_cluster_radar(group_averages, metric_columns, selected_group=None):
    """Render overlaid, per-metric normalized group averages on a shared 0-1 scale."""
    normalized = group_averages.copy()
    for metric in metric_columns:
        minimum = normalized[metric].min()
        maximum = normalized[metric].max()
        if maximum == minimum:
            normalized[metric] = 0.5
        else:
            normalized[metric] = (normalized[metric] - minimum) / (maximum - minimum)

    angles = metric_columns + [metric_columns[0]]
    fig = go.Figure()
    palette = px.colors.qualitative.Safe
    for index, row in normalized.iterrows():
        group = int(row["cluster"])
        is_selected = selected_group is None or group == selected_group
        values = [float(row[metric]) for metric in metric_columns]
        values.append(values[0])
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=angles,
                mode="lines+markers",
                fill="toself",
                fillcolor=palette[(group - 1) % len(palette)] if is_selected else "#777777",
                line=dict(
                    color=palette[(group - 1) % len(palette)] if is_selected else "#777777",
                    width=2,
                ),
                opacity=0.85 if is_selected else 0.18,
                name=f"Group {group}",
            )
        )

    fig.update_layout(
        title="Average Metrics by Group (normalized)",
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".1f")),
        height=650,
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def matplotlib_color(plotly_color):
    """Convert Plotly's rgb(...) palette values to Matplotlib-compatible hex."""
    if plotly_color.startswith("rgb(") and plotly_color.endswith(")"):
        red, green, blue = (int(value.strip()) for value in plotly_color[4:-1].split(","))
        return f"#{red:02x}{green:02x}{blue:02x}"
    return plotly_color


def render_png_download(fig, label, file_name, key, title=None, subtitle=None):
    """Offer a PNG export for Plotly figures when Kaleido is installed (generated on demand)."""
    state_key = f"{key}_data"
    if st.button(f"Prepare {label.lower()}", key=f"{key}_prepare"):
        try:
            export_fig = prepare_export_figure(fig, title=title, subtitle=subtitle)
            st.session_state[state_key] = export_fig.to_image(format="png", scale=2)
        except Exception:
            st.caption("PNG export requires the `kaleido` package.")
            return
    if state_key in st.session_state:
        st.download_button(label, st.session_state[state_key], file_name, "image/png", key=key)


def cluster_map_png(df, wall_lines, selected_group=None, spine_result=None, subtitle=None):
    """Create a PNG directly with Matplotlib, avoiding Plotly's Chrome dependency."""
    figure, axis = plt.subplots(figsize=(10, 8), facecolor="#111111")
    axis.set_facecolor("#111111")
    for line in wall_lines:
        x, y = line.xy
        axis.plot(x, y, color="#666666", linewidth=1.2)

    palette = px.colors.qualitative.Safe
    for group in sorted(df["cluster"].unique()):
        group_df = df[df["cluster"] == group]
        is_selected = selected_group is None or group == selected_group
        color = (
            matplotlib_color(palette[(group - 1) % len(palette)])
            if is_selected
            else "#777777"
        )
        axis.scatter(
            group_df["x"], group_df["y"], s=48, color=color,
            alpha=0.95 if is_selected else 0.22, label=f"Group {group}",
            edgecolors="#111111", linewidths=0.5,
        )
    if spine_result and {"rectangle_coords", "count_points", "side_labels"}.issubset(spine_result):
        rectangle_coords = np.asarray(spine_result["rectangle_coords"])
        axis.fill(
            rectangle_coords[:, 0],
            rectangle_coords[:, 1],
            color="#00E5FF",
            alpha=0.08,
            zorder=5,
        )
        axis.plot(
            rectangle_coords[:, 0],
            rectangle_coords[:, 1],
            color="#00E5FF",
            linewidth=2.5,
            linestyle="--",
            zorder=6,
        )
        side_colors = {"Side 1": "#FF6B6B", "Side 2": "#4D96FF", "On spine": "#FFFFFF"}
        count_points = spine_result["count_points"]
        side_labels = spine_result["side_labels"]
        for side_label, color in side_colors.items():
            side_points = count_points[side_labels == side_label]
            if len(side_points):
                axis.scatter(
                    side_points[:, 0],
                    side_points[:, 1],
                    s=72,
                    color=color,
                    edgecolors="#111111",
                    linewidths=0.7,
                    label=side_label,
                    zorder=7,
                )
        point_zero = spine_result.get("point_zero", spine_result["spine_start"])
        point_one = spine_result.get("point_one", spine_result["spine_end"])
        axis.plot(
            [point_zero[0], point_one[0]],
            [point_zero[1], point_one[1]],
            color="#FFD166",
            linewidth=4,
            zorder=8,
            label="Calculated spine",
        )
        axis.scatter(
            [point_zero[0], point_one[0]],
            [point_zero[1], point_one[1]],
            s=70,
            color="#FFD166",
            edgecolors="#111111",
            zorder=9,
        )
        axis.text(point_zero[0], point_zero[1], "Point 0", color="white", zorder=10)
        axis.text(point_one[0], point_one[1], "Point 1", color="white", zorder=10)
        axis.text(
            rectangle_coords[0, 0],
            rectangle_coords[0, 1],
            f"Spine angle: {spine_result['spine_angle']:.1f}°\n"
            f"Side 1: {spine_result['positive_count']} | Side 2: {spine_result['negative_count']}\n"
            f"Point 0 end: {spine_result['point_zero_count']} | Point 1 end: {spine_result['point_one_count']}\n"
            f"On spine: {spine_result['boundary_count']}",
            color="white",
            fontsize=8,
            va="top",
            bbox=dict(facecolor="#111111", edgecolor="#FFD166", alpha=0.85),
            zorder=9,
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("VGA Metric Clusters", color="white", fontsize=14, fontweight="bold", loc="left", pad=22 if subtitle else 6)
    if subtitle:
        axis.text(0, 1.015, subtitle, transform=axis.transAxes, color="#cccccc", fontsize=9, va="bottom")
    axis.set_xlabel("X (mm)", color="white")
    axis.set_ylabel("Y (mm)", color="white")
    axis.tick_params(colors="white")
    for spine in axis.spines.values():
        spine.set_color("#555555")
    axis.grid(color="#333333", alpha=0.5)
    axis.legend(facecolor="#222222", labelcolor="white")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    output.seek(0)
    return output.getvalue()


ELBOW_STRENGTH_THRESHOLD = 0.3


def elbow_analysis(inertias):
    """Locate the elbow on min-max normalised axes and report how pronounced it is.

    Strength is the largest gap between the normalised curve and the straight line
    joining its endpoints (0 = straight line, 1 = perfect right angle).
    """
    inertias = np.asarray(inertias, dtype=float)
    if len(inertias) <= 2 or np.ptp(inertias) == 0:
        return 2, 0.0
    x_norm = np.linspace(0.0, 1.0, len(inertias))
    y_norm = (inertias - inertias.min()) / np.ptp(inertias)
    gaps = (1.0 - y_norm) - x_norm
    return int(np.argmax(gaps) + 2), float(gaps.max())


def choose_elbow_k(inertias):
    """Select the elbow k (see elbow_analysis)."""
    return elbow_analysis(inertias)[0]


def cluster_vga_metrics(source_df, requested_k=None):
    """Cluster all usable numeric VGA metrics and return results plus diagnostics."""
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    metric_columns = [
        column for column in source_df.select_dtypes(include=np.number).columns
        if column not in {"x", "y"}
    ]
    metric_columns = [column for column in metric_columns if source_df[column].notna().any()]
    if not metric_columns:
        raise ValueError("No numeric VGA metrics are available for clustering.")

    values = SimpleImputer(strategy="median").fit_transform(source_df[metric_columns])
    scaled_values = StandardScaler().fit_transform(values)
    if len(source_df) < 3:
        raise ValueError("At least three VGA points are required for clustering.")

    max_k = min(10, len(source_df) - 1)
    candidate_ks = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    for k in candidate_ks:
        fitted = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_values)
        k_labels = fitted.labels_
        inertias.append(float(fitted.inertia_))
        silhouettes.append(
            float(silhouette_score(scaled_values, k_labels, sample_size=min(3000, len(scaled_values)), random_state=42))
            if len(set(k_labels)) > 1 else float("nan")
        )

    elbow_k, elbow_strength = elbow_analysis(inertias)
    has_clear_elbow = elbow_strength >= ELBOW_STRENGTH_THRESHOLD
    if requested_k is not None:
        selection_method = "manual"
        selected_k = requested_k
    elif has_clear_elbow:
        selection_method = "elbow"
        selected_k = elbow_k
    else:
        selection_method = "silhouette"
        selected_k = candidate_ks[int(np.nanargmax(silhouettes))]
    if selected_k not in candidate_ks:
        raise ValueError(f"Choose a number of groups between 2 and {max_k}.")
    model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_values)

    clustered_df = source_df.copy()
    clustered_df["cluster"] = labels + 1
    score = silhouette_score(scaled_values, labels) if len(set(labels)) > 1 else float("nan")
    elbow_df = pd.DataFrame({"k": candidate_ks, "inertia": inertias, "silhouette": silhouettes})
    elbow_df.attrs.update(
        elbow_k=elbow_k,
        elbow_strength=elbow_strength,
        has_clear_elbow=has_clear_elbow,
        selection_method=selection_method,
    )
    return clustered_df, metric_columns, selected_k, score, elbow_df


def render_centrality_spine_section(source_df, walls):
    """Alternative spine: top-integration cells -> adjacency graph -> centrality path."""
    st.subheader("Spine Identification: Graph-Based Topographic Centrality")
    st.markdown(
        "The spine is the continuous, most-connected corridor of the space, found from the "
        "highest-integration VGA cells rather than from a hand-drawn line."
    )
    with st.expander("How the spine is calculated", expanded=True):
        st.markdown(
            """
1. **Select candidate cells.** Only cells with an integration value are used. The cells at or above the chosen percentile
   of integration (default: the top 15%) are kept. These are the places that are most visually
   accessible from the rest of the plan.
2. **Build an adjacency graph.** Each kept cell becomes a node. It is linked to the kept cells that touch it
   on the grid (up, down, left, right and diagonals). Edge length is the true centre-to-centre distance, so
   the method works for any node interval, not just 1 m.
3. **Isolate the main region.** The kept cells can break into several disconnected patches. The patch with the
   highest *cell count × mean integration* is chosen, so a large, strongly integrated area wins over small
   isolated hot spots.
4. **Score centrality.** Every cell in that region gets a network centrality score.
   *Betweenness* measures how many shortest routes between other cells pass through a cell, so it highlights
   the backbone of the region. *Closeness* measures how near a cell is to all others, so it highlights the
   geometric core. Scores are rescaled to 0–1.
5. **Trace the continuous spine.** The two cells furthest apart in the region are taken as its extremities.
   The spine is the lowest-cost path between them, where moving through a high-centrality cell is cheap and
   moving through a low-centrality cell is expensive. The path therefore stays on the central corridor and
   is always unbroken.
6. **Describe the spine.** A minimum rotated rectangle is fitted around the path and its long axis gives the
   spine line, **Point 0** and **Point 1**. Point 1 is placed at the end with fewer cells beyond it.
   The **spine angle** is the direction from Point 0 to Point 1, measured anticlockwise from the +X axis.
   All VGA cells are then counted either side of the line: **Side 1** (red) is to the left of
   Point 0 → Point 1 and **Side 2** (blue) is to the right. More cells on the right gives **C** (clockwise),
   more on the left gives **AC** (anticlockwise), and equal counts are **Balanced**.
            """
        )
    if "integration" not in source_df.columns:
        st.info("This method needs an `integration` value for every VGA cell.")
        return
    control_a, control_b = st.columns(2)
    top_percent = control_a.slider(
        "Top integration cells to keep (%)", 5, 50, 15, step=1, key="centrality_spine_percent"
    )
    measure_label = control_b.selectbox(
        "Centrality measure",
        ["Betweenness", "Closeness"],
        key="centrality_spine_measure",
        help="Betweenness favours cells that many shortest routes pass through; closeness favours "
        "cells that are nearest to all other top cells.",
    )
    try:
        result = calculate_centrality_spine(
            source_df,
            top_fraction=top_percent / 100,
            measure=measure_label.lower(),
        )
    except ValueError as error:
        st.warning(f"Could not calculate a graph-based spine: {error}")
        return

    st.caption(
        f"{result['candidate_count']} cells with integration ≥ {result['integration_threshold']:.3f} | "
        f"largest connected region {result['region_size']} cells | spine path "
        f"{result['selected_zone_size']} cells | mean integration "
        f"{result['selected_zone_mean_integration']:.3f}"
    )
    fig = render_vga_heatmap_with_underlay(source_df, "integration", walls)
    add_cluster_spine_overlay(fig, result, group_label="Graph-based spine")
    fig.update_layout(title="Graph-Based Topographic Centrality Spine")
    st.plotly_chart(fig, width="stretch")
    angle_col, direction_col = st.columns(2)
    angle_col.metric("Spine angle", f"{result['spine_angle']:.1f}°")
    direction_col.metric(
        "Spine Direction",
        result["spine_direction"],
        help=(
            "C = Clockwise (more cells to the right of Point 0 → Point 1); "
            "AC = AntiClockwise (more cells to the left). Equal counts are Balanced."
        ),
    )
    render_png_download(
        fig,
        "Download graph-based spine as PNG",
        "vga_centrality_spine.png",
        "centrality_spine_png",
        subtitle=(
            f"Top {top_percent}% integration cells, {measure_label.lower()} centrality | "
            f"path {result['selected_zone_size']} cells | spine angle {result['spine_angle']:.1f}° "
            f"({result['spine_direction']})"
        ),
    )


def render_clustering_tab():
    st.subheader("Metric Analysis")
    clustering_upload = st.file_uploader(
        "Import previously analysed VGA JSON (optional)",
        type=["json"],
        key="clustering_json_uploader",
    )

    clustering_df = st.session_state.get("vga_df")
    clustering_walls = st.session_state.get("wall_lines", [])
    if clustering_upload is not None:
        try:
            imported_data = json.load(clustering_upload)
            imported_rows = (
                imported_data.get("vga_results")
                or imported_data.get("vga_grid")
                or imported_data.get("vga_floorplan_nodes")
            )
            if isinstance(imported_rows, dict):
                imported_rows = imported_rows.get("nodes", imported_rows.get("data", []))
            if not imported_rows:
                raise ValueError("The JSON does not contain VGA result rows.")
            clustering_df = pd.DataFrame(imported_rows)
            if st.session_state.get("clustering_upload_name") != clustering_upload.name:
                st.session_state.pop("clustering_result", None)
                st.session_state["clustering_upload_name"] = clustering_upload.name
            st.session_state["clustering_source_df"] = clustering_df
            imported_walls = imported_data.get("floorplan", {}).get("wall_lines", [])
            clustering_walls = [LineString(coords) for coords in imported_walls]
            if clustering_walls:
                st.session_state["clustering_walls"] = clustering_walls
            st.success(f"Loaded {len(clustering_df)} VGA points from `{clustering_upload.name}`.")
        except Exception as error:
            st.error(f"Could not import clustering data: {error}")
            clustering_df = None

    clustering_df = st.session_state.get("clustering_source_df", clustering_df)
    clustering_walls = st.session_state.get("clustering_walls", clustering_walls)
    if clustering_df is None or clustering_df.empty:
        st.info("Run the visibility analysis above or import a saved VGA JSON session to begin clustering.")
        return
    if not {"x", "y"}.issubset(clustering_df.columns):
        st.error("VGA results must include `x` and `y` coordinates to draw the clustered floorplan.")
        return

    render_centrality_spine_section(clustering_df, clustering_walls)

    st.markdown("---")
    st.subheader("K-Means Clustering of VGA Metrics")
    st.caption("Group cells with similar metric profiles, after the spine has been identified above.")
    requested_k = st.number_input(
        "Number of groups (optional; leave at 0 to use the elbow method)",
        min_value=0,
        max_value=max(0, min(10, len(clustering_df) - 1)),
        value=0,
        step=1,
    )
    if st.button("Run K-Means Clustering", type="primary"):
        try:
            result = cluster_vga_metrics(clustering_df, requested_k or None)
            st.session_state["clustering_result"] = result
        except ValueError as error:
            st.warning(str(error))

    result = st.session_state.get("clustering_result")
    if result is None:
        st.caption("Choose a group count if needed, then press the button to run clustering.")
        return

    clustered_df, metric_columns, selected_k, silhouette, elbow_df = result
    st.markdown(
        "The silhouette coefficient measures how well each point fits its own group versus other groups. "
        "Values near 1 indicate clear separation, values near 0 indicate overlapping groups, and negative "
        "values suggest points may be assigned to the wrong group."
    )
    st.caption(f"Metrics used: {', '.join(metric_columns)}")
    metric_col, score_col = st.columns(2)
    metric_col.metric("Groups", selected_k)
    score_col.metric("Silhouette coefficient", f"{silhouette:.3f}")

    elbow_info = elbow_df.attrs
    method = elbow_info.get("selection_method", "elbow")
    strength = elbow_info.get("elbow_strength", float("nan"))
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(
        x=elbow_df["k"], y=elbow_df["inertia"], mode="lines+markers",
        name="Within-group inertia", line=dict(color="#4D96FF"),
    ))
    if "silhouette" in elbow_df:
        elbow_fig.add_trace(go.Scatter(
            x=elbow_df["k"], y=elbow_df["silhouette"], mode="lines+markers",
            name="Silhouette (right axis)", yaxis="y2",
            line=dict(color="#FFD166", dash="dot"),
        ))
    selected_inertia = float(elbow_df.loc[elbow_df["k"] == selected_k, "inertia"].iloc[0])
    elbow_fig.add_trace(go.Scatter(
        x=[selected_k], y=[selected_inertia], mode="markers",
        name=f"Selected k = {selected_k}",
        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#FF4D4D", width=3)),
    ))
    elbow_fig.update_layout(
        title="Elbow Method", template="plotly_dark", height=420,
        xaxis=dict(title="Number of groups", dtick=1),
        yaxis=dict(title="Within-group inertia"),
        yaxis2=dict(title="Silhouette coefficient", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(elbow_fig, width="stretch")
    if method == "manual":
        st.caption(
            f"Group count set manually. Elbow strength {strength:.2f} "
            f"(≥ {ELBOW_STRENGTH_THRESHOLD:.2f} indicates a clear elbow, suggesting k = {elbow_info.get('elbow_k')})."
        )
    elif method == "elbow":
        st.caption(f"Clear elbow detected at k = {selected_k} (strength {strength:.2f}).")
    else:
        st.warning(
            f"No clear elbow (strength {strength:.2f}, needs ≥ {ELBOW_STRENGTH_THRESHOLD:.2f}); the inertia curve "
            f"falls smoothly. Using k = {selected_k}, the group count with the best silhouette coefficient. "
            "Set a group count manually if you prefer another value."
        )
    render_png_download(
        elbow_fig, "Download elbow chart as PNG", "vga_elbow.png", "clustering_elbow_png",
        subtitle=(
            f"Selected {selected_k} groups ({method}) | elbow strength {strength:.2f} | "
            f"silhouette coefficient {silhouette:.3f}"
        ),
    )
    group_options = ["All groups"] + [f"Group {group}" for group in sorted(clustered_df["cluster"].unique())]
    selected_group_label = st.selectbox("View group", group_options)
    selected_group = None if selected_group_label == "All groups" else int(selected_group_label.split()[-1])

    cluster_fig = render_cluster_map(clustered_df, clustering_walls, selected_group)
    st.plotly_chart(cluster_fig, width="stretch")
    render_png_download(
        cluster_fig, "Download cluster map as PNG", "vga_cluster_map.png", "clustering_map_png",
        subtitle=f"K-Means, {selected_k} groups | showing: {selected_group_label} | {len(clustered_df)} cells",
    )
    st.download_button(
        "Download colour-coded plan as PNG",
        data=cluster_map_png(
            clustered_df, clustering_walls, selected_group,
            subtitle=f"K-Means, {selected_k} groups | showing: {selected_group_label} | {len(clustered_df)} cells",
        ),
        file_name="vga_metric_clusters.png",
        mime="image/png",
    )

    group_averages = clustered_df.groupby("cluster")[metric_columns].mean().reset_index()
    radar_fig = render_cluster_radar(group_averages, metric_columns, selected_group)
    st.plotly_chart(radar_fig, width="stretch")
    render_png_download(
        radar_fig, "Download radar chart as PNG", "vga_cluster_radar.png", "clustering_radar_png",
        subtitle="Each metric is min-max normalised across groups (0 = lowest group average, 1 = highest)",
    )
    group_averages.insert(0, "Group", group_averages.pop("cluster").map(lambda value: f"Group {value}"))
    st.subheader("Average Metrics by Group")
    st.dataframe(group_averages, width="stretch", hide_index=True)


@st.cache_resource
def init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=600)
def fetch_historical_crowd_metrics(_supabase):
    """Fetch unlocked prediction metrics and their record and dataset counts."""
    page_size = 1000
    offset = 0
    records = []
    while True:
        response = (
            _supabase.table("vga_crowd_records")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data or []
        records.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    active_records = unlocked_records(pd.DataFrame(records)) if records else pd.DataFrame()
    metrics = [row.get("metrics_data", {}) for row in active_records.to_dict(orient="records")]
    historical_df = pd.json_normalize(metrics) if metrics else pd.DataFrame()
    dataset_count = (
        active_records["upload_batch_id"].nunique()
        if "upload_batch_id" in active_records.columns
        else 0
    )
    return historical_df, len(active_records), dataset_count


def extract_vga_rows(data):
    """Support the saved-session and common standalone VGA JSON shapes."""
    if isinstance(data, list):
        return data, []
    rows = next(
        (data.get(key) for key in ("vga_results", "vga_grid", "vga_floorplan_nodes", "nodes")
         if isinstance(data.get(key), list)),
        [],
    )
    walls = data.get("floorplan", {}).get("wall_lines", [])
    return rows, [LineString(coords) for coords in walls]


def spine_settings_from_metadata(metadata):
    """Read spine settings while accepting both current and legacy session metadata."""
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        bearing = metadata.get("spine_bearing_from_north")
        if bearing is None and metadata.get("spine_angle_from_horizontal") is not None:
            bearing = (
                90.0 - float(metadata["spine_angle_from_horizontal"])
            ) % 360
        bearing = float(bearing)
        if not np.isfinite(bearing):
            raise ValueError("Spine bearing must be finite.")
        bearing %= 360
    except (TypeError, ValueError):
        bearing = 0.0
    metric_units = metadata.get("metric_units", {})
    if not isinstance(metric_units, dict):
        metric_units = {}
    orientation = metadata.get(
        "direction_orientation", metric_units.get("direction_orientation")
    )
    if orientation not in ("Clockwise", "Anti-clockwise"):
        orientation = "Clockwise"
    return bearing, orientation


def projection_png(projected_df, density_column, walls):
    """Render the projected density map without requiring Plotly's image engine."""
    figure, axis = plt.subplots(figsize=(10, 8), facecolor="#111111")
    axis.set_facecolor("#111111")
    for line in walls:
        x_values, y_values = line.xy
        axis.plot(x_values, y_values, color="#666666", linewidth=1.0)
    density_cmap = LinearSegmentedColormap.from_list(
        "crowd_density",
        [color for _, color in DENSITY_COLORSCALE],
    )
    scatter = axis.scatter(
        projected_df["x"],
        projected_df["y"],
        c=projected_df[density_column],
        cmap=density_cmap,
        norm=Normalize(vmin=0, vmax=DENSITY_SCALE_MAX),
        s=42,
    )
    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        label="Projected people / m²",
        ticks=range(DENSITY_SCALE_MAX + 1),
    )
    colorbar.set_label("Projected people / m²", color="white")
    colorbar.ax.tick_params(colors="white")
    axis.plot([], [], color="#666666", linewidth=1.5, label="CAD walls")
    axis.legend(facecolor="#222222", labelcolor="white", loc="upper right")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(
        "Projected Crowd Density", color="white", fontsize=14, fontweight="bold", loc="left", pad=22
    )
    axis.text(
        0, 1.015,
        f"{len(projected_df)} nodes | colour = predicted people per m² (scale 0-{DENSITY_SCALE_MAX})",
        transform=axis.transAxes, color="#cccccc", fontsize=9, va="bottom",
    )
    axis.set_xlabel("X (mm)", color="white")
    axis.set_ylabel("Y (mm)", color="white")
    axis.tick_params(colors="white")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output.getvalue()


def project_crowd_metrics_knn(source_df, historical_df, feature_columns, target_columns, neighbor_fraction=0.05):
    """Project crowd metrics and weighted standard deviations from nearest VGA neighbors."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    historical_features = historical_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    source_features = source_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    historical_space = scaler.fit_transform(imputer.fit_transform(historical_features))
    source_space = scaler.transform(imputer.transform(source_features))

    neighbor_count = max(1, int(np.ceil(len(historical_df) * neighbor_fraction)))
    neighbor_count = min(neighbor_count, len(historical_df))
    projections = {target: np.full(len(source_df), np.nan) for target in target_columns}
    standard_deviations = {target: np.full(len(source_df), np.nan) for target in target_columns}
    for source_index, source_point in enumerate(source_space):
        distances = np.linalg.norm(historical_space - source_point, axis=1)
        nearest_indices = np.argsort(distances)[:neighbor_count]
        nearest_distances = distances[nearest_indices]
        for target in target_columns:
            target_values = pd.to_numeric(historical_df[target], errors="coerce").to_numpy()
            valid = pd.notna(target_values[nearest_indices])
            if not valid.any():
                continue
            values = target_values[nearest_indices][valid]
            selected_distances = nearest_distances[valid]
            target_name = target.lower()
            is_angular_metric = any(
                word in target_name for word in ("direction", "heading", "bearing")
            ) or "degrees_from_spine" in target_name
            if np.any(selected_distances == 0):
                exact_value = float(values[selected_distances == 0][0])
                projections[target][source_index] = (
                    exact_value % 360 if is_angular_metric else exact_value
                )
                standard_deviations[target][source_index] = 0.0
                continue
            weights = 1.0 / selected_distances
            if is_angular_metric:
                radians = np.radians(values)
                mean = float(
                    np.degrees(
                        np.arctan2(
                            np.average(np.sin(radians), weights=weights),
                            np.average(np.cos(radians), weights=weights),
                        )
                    )
                    % 360
                )
                if np.isclose(mean, 360.0):
                    mean = 0.0
                deviations = (values - mean + 180) % 360 - 180
            else:
                mean = np.average(values, weights=weights)
                deviations = values - mean
            projections[target][source_index] = float(mean)
            standard_deviations[target][source_index] = float(
                np.sqrt(np.average(deviations ** 2, weights=weights))
            )

    return projections, standard_deviations, neighbor_count


def render_projection_tab():
    st.subheader("Projected Crowd Metrics")
    st.caption("Each VGA node is mapped into a standardized 10D VGA space. Its closest 5% of historical nodes estimate crowd metrics using inverse-distance weighting; the reported range is the weighted mean ± one weighted standard deviation.")
    projection_upload = st.file_uploader("Upload previous VGA results (JSON)", type=["json"], key="projection_json_uploader")
    source_df = st.session_state.get("vga_df")
    source_walls = st.session_state.get("wall_lines", [])
    source_metadata = st.session_state.get("vga_source_metadata", {})

    upload_grid_mm = None
    if projection_upload is not None:
        try:
            imported = json.load(projection_upload)
            rows, imported_walls = extract_vga_rows(imported)
            source_df = pd.DataFrame(rows)
            if isinstance(imported, dict):
                source_metadata = imported.get("metadata", {})
                if not isinstance(source_metadata, dict):
                    source_metadata = {}
                upload_settings = imported.get("analysis_settings")
                if not isinstance(upload_settings, dict):
                    upload_settings = source_metadata.get("analysis_settings", {})
                if isinstance(upload_settings, dict):
                    upload_grid_mm = upload_settings.get("grid_size_mm")
            if imported_walls:
                source_walls = imported_walls
            st.success(f"Loaded {len(source_df)} VGA points from `{projection_upload.name}`.")
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Could not load VGA results: {error}")
            return

    if source_df is None or source_df.empty or not {"x", "y"}.issubset(source_df.columns):
        st.info("Upload a VGA JSON session or run VGA analysis above to begin.")
        return

    # Historical data lives on the reference grid, so a finer source grid is
    # resampled onto it for matching and the result is copied back to each node.
    source_grid_mm = infer_grid_interval_mm(
        source_df,
        upload_grid_mm if projection_upload is not None else st.session_state.get("vga_grid_spacing"),
    )
    source_area, _ = extract_analysis_area(
        {"analysis_settings": {"grid_size_mm": source_grid_mm}},
        node_count=len(source_df),
        node_data=source_df,
    )
    model_df, block_index = resample_to_reference_grid(
        add_comparison_metrics(source_df.reset_index(drop=True), source_area),
        source_grid_mm,
    )
    if source_grid_mm and source_grid_mm < REFERENCE_GRID_MM:
        st.caption(
            f"Node interval {source_grid_mm:g} mm is finer than the {REFERENCE_GRID_MM:g} mm reference grid "
            f"used by the historical data: {len(source_df):,} nodes are matched as {len(model_df):,} "
            "reference-grid blocks and each node receives its block's projection."
        )
    elif source_grid_mm and source_grid_mm > REFERENCE_GRID_MM:
        st.warning(
            f"Node interval {source_grid_mm:g} mm is coarser than the {REFERENCE_GRID_MM:g} mm "
            "reference grid; projections may be less reliable."
        )
    try:
        historical_df, historical_node_count, historical_dataset_count = (
            fetch_historical_crowd_metrics(init_supabase())
        )
    except Exception as error:
        st.warning(
            "Could not load historical crowd records. Verify that `SUPABASE_URL` "
            "and `SUPABASE_KEY` are configured and that the `vga_crowd_records` "
            "table exists and is accessible. Supabase error: "
            f"{error}"
        )
        return
    if historical_df.empty:
        st.info("No historical VGA and crowd records are available in Supabase yet.")
        return
    st.caption(
        f"Prediction history: {historical_node_count:,} unlocked node records "
        f"from {historical_dataset_count:,} data sets."
    )

    vga_columns = [
        column
        for column in model_df.select_dtypes(include=np.number).columns
        if column not in {"x", "y", "grid_size_mm"}
    ]
    shared_vga_columns = [
        column
        for column in vga_columns
        if column in historical_df.columns and pd.to_numeric(historical_df[column], errors="coerce").notna().any()
    ]
    crowd_metric_words = (
        "crowd",
        "density",
        "people",
        "person",
        "pedestrian",
        "count",
        "volume",
        "speed",
        "velocity",
        "direction",
        "heading",
        "bearing",
        "deviation",
        "occupancy",
        "flow",
        "spine",
    )
    crowd_columns = [
        column
        for column in historical_df.select_dtypes(include=np.number).columns
        if column not in shared_vga_columns
        and any(word in column.lower() for word in crowd_metric_words)
    ]
    if len(shared_vga_columns) < 10 or not crowd_columns:
        st.warning("The uploaded VGA data and Supabase data must share at least 10 numeric VGA metrics and contain crowd metrics.")
        return

    selected_targets = st.multiselect(
        "Crowd and movement metrics to predict",
        crowd_columns,
        default=crowd_columns,
        key="projection_targets",
    )
    if not selected_targets:
        return

    direction_targets = [
        target
        for target in selected_targets
        if any(word in target.lower() for word in ("direction", "heading", "bearing"))
        or "degrees_from_spine" in target.lower()
    ]
    deviation_targets = [
        target
        for target in selected_targets
        if "deviation" in target.lower() and "spine" in target.lower()
    ]
    spine_bearing_from_north, stored_orientation = spine_settings_from_metadata(
        source_metadata
    )
    if direction_targets or deviation_targets:
        st.markdown("#### Spine reference for direction previews")
        spine_col, orientation_col = st.columns(2)
        with spine_col:
            spine_bearing_from_north = st.number_input(
                "Spine direction from north (degrees)",
                min_value=0.0,
                max_value=359.999,
                value=float(spine_bearing_from_north),
                step=1.0,
                key="projection_spine_bearing",
                help="Prefilled from uploaded session metadata when available.",
            )
        with orientation_col:
            direction_orientation = st.radio(
                "Spine direction convention (C/AC)",
                options=["Clockwise", "Anti-clockwise"],
                format_func=lambda orientation: (
                    "Clockwise (C)"
                    if orientation == "Clockwise"
                    else "AntiClockwise (AC)"
                ),
                index=0 if stored_orientation == "Clockwise" else 1,
                horizontal=True,
                key="projection_direction_orientation",
                help="These settings convert spine-relative values back to floorplan arrows.",
            )
    else:
        direction_orientation = stored_orientation

    feature_columns = shared_vga_columns[:10]
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(source_df, index=True).values.tobytes()
    ).hexdigest()
    history_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            historical_df[feature_columns + selected_targets], index=True
        ).values.tobytes()
    ).hexdigest()
    projection_signature = hashlib.sha256(
        json.dumps(
            {
                "source": source_hash,
                "source_grid_mm": source_grid_mm,
                "history": history_hash,
                "targets": selected_targets,
                "features": feature_columns,
                "spine_bearing_from_north": float(spine_bearing_from_north),
                "direction_orientation": direction_orientation,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if st.button(
        "Generate Crowd Projection",
        type="primary",
        key="generate_crowd_projection",
    ):
        projections, standard_deviations, neighbor_count = project_crowd_metrics_knn(
            model_df,
            historical_df,
            feature_columns,
            selected_targets,
        )
        usable_projections = {}
        usable_standard_deviations = {}
        for target, values in projections.items():
            mean_values = pd.Series(
                np.asarray(values)[block_index], index=source_df.index
            ).clip(lower=0)
            if pd.notna(mean_values).any():
                usable_projections[target] = mean_values
                usable_standard_deviations[target] = pd.Series(
                    np.asarray(standard_deviations[target])[block_index],
                    index=source_df.index,
                )
        st.session_state["crowd_projection_result"] = {
            "signature": projection_signature,
            "projections": usable_projections,
            "standard_deviations": usable_standard_deviations,
            "neighbor_count": neighbor_count,
        }

    projection_result = st.session_state.get("crowd_projection_result")
    if (
        projection_result is None
        or projection_result.get("signature") != projection_signature
    ):
        st.info(
            "Choose the crowd metrics and spine settings, then press "
            "**Generate Crowd Projection** to calculate predictions."
        )
        return

    projections = projection_result["projections"]
    standard_deviations = projection_result["standard_deviations"]
    neighbor_count = projection_result["neighbor_count"]
    if not projections:
        st.warning("No selected crowd metrics have usable historical values in the closest VGA neighbors.")
        return
    projected_df = source_df[["x", "y"]].copy()
    for target, values in projections.items():
        base_field = f"projected_{target}"
        deviation = standard_deviations[target]
        projected_df[base_field] = values
        projected_df[f"{base_field}_std"] = deviation
        projected_df[f"{base_field}_lowest"] = (values - deviation).clip(lower=0)
        projected_df[f"{base_field}_highest"] = values + deviation
    st.caption(f"VGA dimensions: {', '.join(feature_columns)} | Historical neighbors per node: {neighbor_count} of {len(historical_df)}")
    st.dataframe(
        pd.DataFrame(
            {
                "Crowd metric": [
                    crowd_metric_label(metric) for metric in projections
                ],
                "Method": "Inverse-distance weighted mean ± 1 standard deviation",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    x_differences = np.diff(np.sort(projected_df["x"].dropna().unique()))
    y_differences = np.diff(np.sort(projected_df["y"].dropna().unique()))
    spacing_candidates = [
        float(np.median(differences[differences > 0]))
        for differences in (x_differences, y_differences)
        if np.any(differences > 0)
    ]
    vector_length = min(spacing_candidates) * 0.35 if spacing_candidates else 1.0

    metric_tabs = st.tabs([crowd_metric_label(target) for target in projections])
    for metric_index, (target, metric_tab) in enumerate(zip(projections, metric_tabs)):
        with metric_tab:
            projected_metric = f"projected_{target}"
            metric_name = target.lower()
            is_density_metric = "density" in metric_name
            is_direction_metric = target in direction_targets
            is_deviation_metric = target in deviation_targets
            values = projected_df[projected_metric]
            standard_deviation = projected_df[f"{projected_metric}_std"]
            low_values = projected_df[f"{projected_metric}_lowest"]
            high_values = projected_df[f"{projected_metric}_highest"]
            metric_label = crowd_metric_label(target)
            high_density = values > 3 if is_density_metric else None

            if is_density_metric:
                st.metric(
                    "Projected areas above 3 people / m²",
                    f"{int(high_density.sum())} / {len(projected_df)} nodes",
                )
                st.caption("Nodes above 3 people / m² are the mid-to-high density areas.")

            fig_metric = go.Figure()
            for line in source_walls:
                x_values, y_values = line.xy
                fig_metric.add_trace(
                    go.Scatter(
                        x=list(x_values),
                        y=list(y_values),
                        mode="lines",
                        line=dict(color="#666"),
                        showlegend=False,
                    )
                )

            if is_density_metric:
                colorscale = DENSITY_COLORSCALE
                color_min, color_max = 0, DENSITY_SCALE_MAX
                colorbar_title = "people / m²"
            elif is_direction_metric:
                colorscale = "twilight"
                color_min, color_max = 0, 360
                colorbar_title = "degrees from spine"
            elif is_deviation_metric:
                colorscale = [
                    [0.0, "#ffffff"],
                    [0.25, "#fff7bc"],
                    [0.5, "#fec44f"],
                    [0.75, "#d95f0e"],
                    [1.0, "#8c2d04"],
                ]
                color_min, color_max = 0, 90
                colorbar_title = "deviation angle (°)"
            else:
                colorscale = "Viridis"
                color_min = float(values.min())
                color_max = float(values.max())
                if color_min == color_max:
                    color_max = color_min + 1
                colorbar_title = metric_label

            hover_value_label = (
                "predicted density"
                if is_density_metric
                else "predicted direction"
                if is_direction_metric
                else "predicted deviation"
                if is_deviation_metric
                else f"predicted {metric_label}"
            )
            hover_template = (
                f"x=%{{x}}<br>y=%{{y}}<br>{hover_value_label}=%{{marker.color:.2f}}"
                "<br>weighted standard deviation=%{customdata[0]:.2f}"
                "<br>predicted range=%{customdata[1]:.2f}–%{customdata[2]:.2f}"
                "<extra></extra>"
            )
            marker_options = dict(
                size=9,
                color=values.tolist(),
                colorscale=colorscale,
                cmin=color_min,
                cmax=color_max,
                showscale=True,
                colorbar=dict(
                    title=colorbar_title,
                    tick0=0 if is_density_metric else None,
                    dtick=1 if is_density_metric else None,
                ),
            )
            fig_metric.add_trace(
                go.Scatter(
                    x=projected_df["x"],
                    y=projected_df["y"],
                    mode="markers",
                    marker=marker_options,
                    customdata=pd.concat(
                        [standard_deviation, low_values, high_values], axis=1
                    ).to_numpy(),
                    text=(
                        np.where(
                            high_density,
                            "MID-HIGH DENSITY (>3 people/m²)",
                            "Below threshold",
                        ).tolist()
                        if is_density_metric
                        else None
                    ),
                    hovertemplate=(
                        hover_template.replace(
                            "<extra></extra>",
                            "<br>%{text}<extra></extra>",
                        )
                        if is_density_metric
                        else hover_template
                    ),
                    name=f"Predicted {metric_label}",
                )
            )

            if is_direction_metric:
                add_direction_arrows(
                    fig_metric,
                    projected_df["x"],
                    projected_df["y"],
                    values,
                    spine_bearing_from_north,
                    clockwise=direction_orientation == "Clockwise",
                    standard_deviations=standard_deviation,
                    base_length=vector_length,
                )
            elif is_deviation_metric:
                add_deviation_rays(
                    fig_metric,
                    projected_df["x"],
                    projected_df["y"],
                    values,
                    spine_bearing_from_north,
                    length=vector_length,
                )

            fig_metric.update_layout(
                title=(
                    f"Predicted {metric_label} with uncertainty-scaled arrows"
                    if is_direction_metric
                    else f"Predicted {metric_label} with four rays per node"
                    if is_deviation_metric
                    else f"Projected {metric_label}"
                ),
                template="plotly_dark",
                height=620,
                xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Y (mm)"),
            )
            configure_floorplan_figure(fig_metric)
            st.plotly_chart(fig_metric, width="stretch")

            if is_direction_metric:
                st.caption(
                    "Arrow direction uses the spine bearing and selected clockwise/anti-clockwise convention. "
                    "Higher direction standard deviation produces shorter, more transparent arrows."
                )
            elif is_deviation_metric:
                st.caption(
                    "Four rays show both directions along the undirected spine and both "
                    "deviation angles (0–90°) from it."
                )

            render_png_download(
                fig_metric,
                f"Download projected {metric_label} map as PNG",
                f"projected_metric_{metric_index}.png",
                f"projection_metric_png_{metric_index}",
                subtitle=(
                    f"Inverse-distance weighted mean of {neighbor_count} nearest historical nodes | "
                    f"{len(projected_df)} nodes | colour = {colorbar_title}"
                ),
            )

            if is_density_metric:
                st.download_button(
                    "Download projected density map as PNG",
                    projection_png(projected_df, projected_metric, source_walls),
                    "projected_crowd_density.png",
                    "image/png",
                    key=f"projection_map_png_{metric_index}",
                )

    serialized_walls = [list(line.coords) for line in source_walls]
    crowd_projection_export = {
        "metadata": {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_method": "Inverse-distance weighted mean ± 1 standard deviation",
            "vga_dimensions": feature_columns,
            "historical_neighbor_count_per_node": neighbor_count,
            "historical_node_count": len(historical_df),
            "projected_crowd_metrics": list(projections),
            "spine_bearing_from_north": float(spine_bearing_from_north),
            "direction_orientation": direction_orientation,
        },
        "summary": {
            "total_grid_node_count": len(projected_df),
            "projected_metric_count": len(projections),
        },
        "floorplan": {
            "wall_lines": serialized_walls,
            "cad_walls": serialized_walls,
        },
        "wall_lines": serialized_walls,
        "cad_walls": serialized_walls,
        "nodes": json.loads(projected_df.to_json(orient="records")),
    }
    st.download_button(
        "Download projected metrics as JSON",
        json.dumps(crowd_projection_export, indent=2),
        "projected_crowd_metrics.json",
        "application/json",
        key="projection_json",
    )


analysis_tab, clustering_tab, projection_tab = st.tabs(["2.1 VGA Analysis", "2.2 Metric Analysis", "2.3 Crowd Projection"])

@st.cache_resource
def vga_result_store():
    """Process-wide store of finished VGA runs, shared across browser sessions."""
    return {}


def render_analysis_tab():
    uploaded_file = st.file_uploader(
        "Upload CAD Floorplan (DXF, DWG) or Saved Session (JSON)",
        type=["dxf", "dwg", "json"],
        key="vga_analysis_uploader",
    )

    if uploaded_file is None:
        return

    file_ext = "." + uploaded_file.name.split(".")[-1].lower()
    upload_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("selected_rooms_source") != upload_signature:
        st.session_state["selected_rooms"] = []
        st.session_state["selected_rooms_source"] = upload_signature

    if file_ext == ".json":
        # Process imported pre-computed session
        try:
            session_data = json.load(uploaded_file)
            st.session_state["vga_df"] = pd.DataFrame(session_data["vga_results"])
            st.session_state["vga_floorplan_nodes"] = st.session_state["vga_df"].to_dict(orient="records")
            st.session_state["vga_results"] = st.session_state["vga_floorplan_nodes"]
            session_metadata = session_data.get("metadata")
            session_metadata = session_metadata if isinstance(session_metadata, dict) else {}
            st.session_state["vga_source_metadata"] = session_metadata
            analysis_settings = session_metadata.get("analysis_settings", {})
            if isinstance(analysis_settings, dict) and analysis_settings.get("grid_size_mm") is not None:
                st.session_state["vga_grid_spacing"] = analysis_settings["grid_size_mm"]
            else:
                st.session_state.pop("vga_grid_spacing", None)
            st.session_state["clustering_source_df"] = st.session_state["vga_df"]
            
            restored_walls = [
                LineString(coords) for coords in session_data["floorplan"]["wall_lines"]
            ]
            st.session_state["wall_lines"] = restored_walls
            st.session_state["clustering_walls"] = restored_walls
            
            if "selected_rooms" in session_data["floorplan"]:
                st.session_state["selected_rooms"] = [
                    Polygon(coords) for coords in session_data["floorplan"]["selected_rooms"]
                ]
            
            st.success(f"✅ Successfully reimported session data from `{uploaded_file.name}`!")
        except Exception as e:
            st.error(f"❌ Error loading JSON session file: {e}")
            st.stop()
    else:
        # Process raw CAD files (DXF or DWG)
        st.session_state.pop("vga_source_metadata", None)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Parsing CAD wall geometry and building enclosed spatial zones..."):
            try:
                wall_lines = process_cad_file(tmp_path)
                st.session_state["wall_lines"] = wall_lines
                strtree = STRtree(wall_lines)
                enclosed_rooms = extract_enclosed_rooms(wall_lines, snap_distance=door_snap_dist)
            except Exception as e:
                st.error(f"❌ Error parsing CAD file: {e}")
                st.stop()

        st.success(
            f"Extracted {len(wall_lines)} wall boundary segments and detected {len(enclosed_rooms)} spatial zones."
        )

        all_bounds = [w.bounds for w in wall_lines]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        floorplan_bounds = (minx, miny, maxx, maxy)

        st.subheader("Interactive Public Space Selection")
        st.info(
            " **Single Click Selection Active:** Click inside a room or corridor using the **`+` crosshair**. For narrow zones, click an amber target marker. If zones overlap, the smallest containing zone is selected."
        )

        selection_mode_option = st.radio(
            "Selection Mode:",
            ["Full Floorplan", "Click Inside Rooms to Select Zones"],
            horizontal=True,
        )

        if "selected_rooms" not in st.session_state:
            st.session_state["selected_rooms"] = []

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(" Reset Selected Regions"):
                st.session_state["selected_rooms"] = []
                st.rerun()

        selected_polygons = st.session_state["selected_rooms"]

        if selection_mode_option == "Click Inside Rooms to Select Zones":
            fig_plan = render_interactive_floorplan(
                wall_lines,
                floorplan_bounds,
                selected_polys=selected_polygons,
                selectable_zones=enclosed_rooms,
            )

            chart_events = st.plotly_chart(
                fig_plan,
                width="stretch",
                on_select="rerun",
                selection_mode="points",
                key="floorplan_selector",
            )
            render_png_download(
                fig_plan, "Download floorplan as PNG", "vga_floorplan.png", "vga_floorplan_png",
                title="VGA Floorplan and Selected Zones",
                subtitle=f"{len(selected_polygons)} zone(s) selected | {len(wall_lines)} wall segments | axes in mm",
            )

            if chart_events and "selection" in chart_events:
                pts = chart_events["selection"].get("points", [])
                if pts:
                    matched_room = None
                    for point in pts:
                        zone_index = point.get("customdata")
                        if isinstance(zone_index, (list, tuple, np.ndarray)):
                            zone_index = zone_index[0] if len(zone_index) else None
                        if isinstance(zone_index, (int, np.integer)):
                            if 0 <= zone_index < len(enclosed_rooms):
                                matched_room = enclosed_rooms[zone_index]
                                break

                    if matched_room is None:
                        click_point = Point(pts[0]["x"], pts[0]["y"])
                        candidate_rooms = [
                            room for room in enclosed_rooms if room.contains(click_point)
                        ]
                        if candidate_rooms:
                            matched_room = min(candidate_rooms, key=lambda room: room.area)

                    if matched_room is not None and not any(
                        room.equals(matched_room)
                        for room in st.session_state["selected_rooms"]
                    ):
                        st.session_state["selected_rooms"].append(matched_room)
                        st.rerun()

            if selected_polygons:
                total_area = sum(p.area for p in selected_polygons) / 1e6
                st.success(
                    f"✅ **{len(selected_polygons)} Zone(s) Selected & Highlighted!** Combined Area: `{round(total_area, 2)} m²`"
                )

        # Server-side copy of finished runs. A long analysis can outlive the browser
        # connection, which starts a fresh session; this lets the results be recovered.
        run_key = hashlib.sha256(
            json.dumps(
                [
                    upload_signature, float(grid_size), int(ray_count), float(door_snap_dist),
                    selection_mode_option,
                    sorted(p.wkt for p in selected_polygons),
                ],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        result_store = vga_result_store()

        def publish_results(df_results):
            st.session_state["vga_df"] = df_results
            st.session_state["vga_floorplan_nodes"] = df_results.to_dict(orient="records")
            st.session_state["vga_results"] = st.session_state["vga_floorplan_nodes"]
            st.session_state["clustering_source_df"] = df_results
            st.session_state["clustering_walls"] = wall_lines
            st.session_state["vga_grid_spacing"] = float(grid_size)

        if "vga_df" not in st.session_state and run_key in result_store:
            publish_results(result_store[run_key])
            st.info("Restored the finished analysis for this file and settings.")

        if st.button("Run Visibility Analysis"):
            if selected_polygons and selection_mode_option != "Full Floorplan":
                combined_bounds = unary_union(selected_polygons).bounds
                calc_minx, calc_miny, calc_maxx, calc_maxy = combined_bounds
            else:
                calc_minx, calc_miny, calc_maxx, calc_maxy = minx, miny, maxx, maxy

            x_coords = np.arange(calc_minx, calc_maxx, grid_size)
            y_coords = np.arange(calc_miny, calc_maxy, grid_size)

            grid_points = []
            for x in x_coords:
                for y in y_coords:
                    pt = Point(x, y)
                    if selected_polygons and selection_mode_option != "Full Floorplan":
                        if any(poly.contains(pt) for poly in selected_polygons):
                            grid_points.append((x, y))
                    else:
                        grid_points.append((x, y))

            total_points = len(grid_points)

            if total_points == 0:
                st.warning("No grid points generated in selected zone. Try a smaller grid dimension or select a room.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                vga_results = []
                isovist_polys = []
                start_time = time.time()

                for idx, pt in enumerate(grid_points):
                    isovist, occluded_count = generate_isovist_polygon(
                        pt, wall_lines, strtree, num_rays=ray_count
                    )
                    if isovist:
                        metrics = compute_isovist_metrics(
                            isovist, pt, occluded_count, ray_count
                        )
                        metrics["x"] = pt[0]
                        metrics["y"] = pt[1]
                        vga_results.append(metrics)
                        isovist_polys.append(isovist)

                    completed = idx + 1
                    progress_ratio = completed / total_points
                    elapsed_time = time.time() - start_time
                    avg_time_per_pt = elapsed_time / completed
                    remaining_pts = total_points - completed
                    estimated_remaining_seconds = remaining_pts * avg_time_per_pt

                    mins, secs = divmod(int(estimated_remaining_seconds), 60)
                    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                    if completed % 5 == 0 or completed == total_points:
                        progress_bar.progress(progress_ratio)
                        status_text.markdown(
                            f" **Phase 1: Analyzing Isovists {completed}/{total_points}** ({int(progress_ratio * 100)}%) | **Est. time remaining:** `{time_str}`"
                        )

                if vga_results:
                    final_vga_results = compute_graph_topology_with_progress(
                        vga_results, isovist_polys, status_text, progress_bar
                    )

                    progress_bar.empty()
                    total_time = round(time.time() - start_time, 2)
                    status_text.success(
                        f"✅ Analysis complete in **{total_time}s** across **{total_points}** points!"
                    )

                    df_results = pd.DataFrame(final_vga_results)
                    result_store[run_key] = df_results
                    while len(result_store) > 3:
                        result_store.pop(next(iter(result_store)))
                    publish_results(df_results)
                else:
                    progress_bar.empty()
                    st.error("Could not extract valid isovists. Selected points may be inside wall geometry.")

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

        # Offer the session download first, so results can be saved even if
        # rendering the heatmap below is slow or fails for large grids.
        export_key = (id(df), uploaded_file.name, len(st.session_state.get("selected_rooms", [])))
        if st.session_state.get("vga_export_key") != export_key:
            wall_lines_serialized = [
                [list(c) for c in line.coords]
                for line in st.session_state.get("wall_lines", [])
            ]
            selected_rooms_serialized = [
                [list(c) for c in poly.exterior.coords]
                for poly in st.session_state.get("selected_rooms", [])
            ]
            complete_vga_export = {
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "file_name": uploaded_file.name if uploaded_file else "imported_session",
                    "camera_corners": st.session_state.get("four_corners", []),
                },
                "analysis_settings": {
                    "grid_size_mm": grid_size,
                    "ray_step_deg": ray_step,
                    "door_snap_dist_mm": door_snap_dist,
                    "selection_mode": selection_mode_option if 'selection_mode_option' in locals() else "Unknown"
                },
                "floorplan": {
                    "bounds": floorplan_bounds if 'floorplan_bounds' in locals() else None,
                    "wall_lines": wall_lines_serialized,
                    "selected_rooms": selected_rooms_serialized,
                    "camera_corners": st.session_state.get("four_corners", []),
                },
                "camera_corners": st.session_state.get("four_corners", []),
                "vga_results": df.to_dict(orient="records")
            }
            st.session_state["vga_export_json"] = json.dumps(complete_vga_export, default=float)
            st.session_state["vga_export_key"] = export_key

        st.download_button(
            label=" Download Complete VGA Session JSON",
            data=st.session_state["vga_export_json"],
            file_name="vga_complete_session.json",
            mime="application/json",
        )

        st.subheader("VGA Heatmap Visualizer")
        available_metrics = [c for c in df.columns if c not in ["x", "y"]]

        if available_metrics:
            selected_metric = st.selectbox(
                "Select Metric to Render:",
                available_metrics,
                format_func=crowd_metric_label,
            )
            metric_name = selected_metric.lower()
            is_direction_metric = (
                any(
                    word in metric_name
                    for word in ("direction", "heading", "bearing")
                )
                or "degrees_from_spine" in metric_name
            ) and "deviation" not in metric_name
            is_deviation_metric = (
                "deviation" in metric_name and "spine" in metric_name
            )
            if is_direction_metric or is_deviation_metric:
                map_bearing, map_orientation = spine_settings_from_metadata(
                    st.session_state.get("vga_source_metadata", {})
                )
                bearing_col, orientation_col = st.columns(2)
                with bearing_col:
                    map_bearing = st.number_input(
                        "Spine bearing from north (degrees)",
                        min_value=0.0,
                        max_value=359.999,
                        value=float(map_bearing),
                        step=1.0,
                        key="vga_map_spine_bearing",
                    )
                with orientation_col:
                    map_orientation = st.radio(
                        "Direction angle convention",
                        ["Clockwise", "Anti-clockwise"],
                        index=0 if map_orientation == "Clockwise" else 1,
                        horizontal=True,
                        key="vga_map_direction_orientation",
                    )

            if selected_metric in df.columns and "wall_lines" in st.session_state:
                fig_heatmap = render_vga_heatmap_with_underlay(
                    df, selected_metric, st.session_state["wall_lines"]
                )
                x_differences = np.diff(np.sort(df["x"].dropna().unique()))
                y_differences = np.diff(np.sort(df["y"].dropna().unique()))
                spacing_candidates = [
                    float(np.median(differences[differences > 0]))
                    for differences in (x_differences, y_differences)
                    if np.any(differences > 0)
                ]
                vector_length = (
                    min(spacing_candidates) * 0.35 if spacing_candidates else 1.0
                )
                if is_direction_metric:
                    add_direction_arrows(
                        fig_heatmap,
                        df["x"],
                        df["y"],
                        pd.to_numeric(df[selected_metric], errors="coerce"),
                        map_bearing,
                        clockwise=map_orientation == "Clockwise",
                        base_length=vector_length,
                    )
                elif is_deviation_metric:
                    add_deviation_rays(
                        fig_heatmap,
                        df["x"],
                        df["y"],
                        pd.to_numeric(df[selected_metric], errors="coerce"),
                        map_bearing,
                        length=vector_length,
                    )
                st.plotly_chart(fig_heatmap, width="stretch")
                render_png_download(
                    fig_heatmap, "Download heatmap as PNG", "vga_heatmap.png", "vga_heatmap_png",
                    subtitle=f"{len(df)} VGA cells | axes in mm",
                )



with analysis_tab:
    render_analysis_tab()

with clustering_tab:
    render_clustering_tab()

with projection_tab:
    render_projection_tab()
