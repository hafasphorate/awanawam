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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utils.navigation import render_home_button
from utils.plotly_floorplan import configure_floorplan_figure
from utils.dataset_locks import unlocked_records
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

    for i in range(num_nodes):
        poly_i = isovist_polys[i]

        for j in range(i + 1, num_nodes):
            pt_j = Point(vga_results[j]["x"], vga_results[j]["y"])
            poly_j = isovist_polys[j]
            pt_i = Point(vga_results[i]["x"], vga_results[i]["y"])

            if poly_i.contains(pt_j) or poly_j.contains(pt_i):
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True

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

    for i in range(num_nodes):
        visited = {i: 0}
        queue = [i]

        while queue:
            curr = queue.pop(0)
            curr_depth = visited[curr]

            neighbors = np.where(adj_matrix[curr])[0]
            for nxt in neighbors:
                if nxt not in visited:
                    visited[nxt] = curr_depth + 1
                    queue.append(nxt)

        depths = list(visited.values())
        total_depth = sum(depths)
        reachable_nodes = len(depths)

        if reachable_nodes > 1:
            mean_depth = total_depth / (reachable_nodes - 1)
            integration = 1.0 / (2.0 * (mean_depth - 1.0) / max(1, reachable_nodes - 2)) if mean_depth > 1 else 0.0
        else:
            mean_depth = 0.0
            integration = 0.0

        vga_results[i]["mean_depth"] = round(mean_depth, 3)
        vga_results[i]["integration"] = round(integration, 3)

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
    """Return edge-connected grid zones ranked by size and cell-based compactness."""
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
        zone_center = selected_points.mean(axis=0)
        center_distance = float(np.linalg.norm(zone_center - cluster_center))
        zones.append(
            {
                "component": component,
                "selected_points": selected_points,
                "size": len(component),
                "compactness": compactness,
                "score": len(component) * compactness,
                "center_distance": center_distance,
                "tie_break": min(component),
            }
        )

    selected_zone = max(
        zones,
        key=lambda zone: (
            zone["score"],
            zone["size"],
            zone["compactness"],
            -zone["center_distance"],
            tuple(-value for value in zone["tie_break"]),
        ),
    )
    return zones, selected_zone, x_step, y_step


def select_mall_spine_zone(clustered_df):
    """Select one connected zone globally across every K-Means group."""
    all_points = clustered_df[["x", "y"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().drop_duplicates()
    if all_points.empty:
        raise ValueError("There are no usable VGA x/y cells for spine selection.")

    mall_center = all_points[["x", "y"]].mean().to_numpy(dtype=float)
    candidates = []
    for group in sorted(clustered_df["cluster"].dropna().unique()):
        group_df = clustered_df[clustered_df["cluster"] == group]
        zones, _, _, _ = connected_grid_zones(group_df, clustered_df)
        for zone in zones:
            zone_center = zone["selected_points"].mean(axis=0)
            zone["mall_center_distance"] = float(np.linalg.norm(zone_center - mall_center))
            candidates.append((int(group), zone))

    if not candidates:
        raise ValueError("No connected VGA zones are available for spine selection.")

    return max(
        candidates,
        key=lambda candidate: (
            candidate[1]["score"],
            candidate[1]["size"],
            candidate[1]["compactness"],
            -candidate[1]["mall_center_distance"],
            -candidate[0],
            tuple(-value for value in candidate[1]["tie_break"]),
        ),
    )


def calculate_cluster_spine(cluster_df, count_df=None, grid_df=None, selected_zone=None):
    """Calculate the spine through the globally selected connected zone."""
    points = cluster_df[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    if points.empty:
        raise ValueError("The selected group has no usable x/y cells.")

    zones, local_zone, x_step, y_step = connected_grid_zones(points, grid_df)
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
    side_values = direction_x * (count_points[:, 1] - center_y) - direction_y * (count_points[:, 0] - center_x)
    side_tolerance = max(min(x_step, y_step) * 0.05, 1e-9)
    side_labels = np.where(
        side_values > side_tolerance,
        "Side 1",
        np.where(side_values < -side_tolerance, "Side 2", "On spine"),
    )
    positive_count = int(np.sum(side_labels == "Side 1"))
    negative_count = int(np.sum(side_labels == "Side 2"))
    boundary_count = int(np.sum(side_labels == "On spine"))
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
        "selected_zone_score": selected_zone["score"],
    }


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
            f"Compactness: {spine_result['selected_zone_compactness']:.3f} | "
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


def render_cluster_spine_map(df, wall_lines, selected_group, spine_result):
    """Render all groups and overlay the single mall-wide selected zone and spine."""
    fig = render_cluster_map(df, wall_lines)
    add_cluster_spine_overlay(
        fig,
        spine_result,
        group_label=f"Winning zone — Group {selected_group}",
    )
    fig.update_layout(title="Mall-wide Automatic Spine Calculation")
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


def render_png_download(fig, label, file_name, key):
    """Offer a PNG export for Plotly figures when Kaleido is installed."""
    try:
        png_data = fig.to_image(format="png")
    except Exception:
        st.caption("PNG export requires the `kaleido` package.")
        return
    st.download_button(label, png_data, file_name, "image/png", key=key)


def cluster_map_png(df, wall_lines, selected_group=None, spine_result=None):
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
    axis.set_title("VGA Metric Clusters", color="white")
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


def choose_elbow_k(inertias):
    """Select the elbow as the point furthest from the first-to-last line."""
    if len(inertias) <= 2:
        return 2

    points = np.column_stack((np.arange(len(inertias)), inertias))
    start, end = points[0], points[-1]
    line = end - start
    distances = np.abs(np.cross(line, points - start)) / np.linalg.norm(line)
    return int(np.argmax(distances) + 2)


def cluster_vga_metrics(source_df, requested_k=None):
    """Cluster all usable numeric VGA metrics and return results plus diagnostics."""
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
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_values).inertia_ for k in candidate_ks]
    selected_k = requested_k if requested_k is not None else choose_elbow_k(inertias)
    if selected_k not in candidate_ks:
        raise ValueError(f"Choose a number of groups between 2 and {max_k}.")
    model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_values)

    clustered_df = source_df.copy()
    clustered_df["cluster"] = labels + 1
    score = silhouette_score(scaled_values, labels) if len(set(labels)) > 1 else float("nan")
    elbow_df = pd.DataFrame({"k": candidate_ks, "inertia": inertias})
    return clustered_df, metric_columns, selected_k, score, elbow_df


def render_clustering_tab():
    st.subheader("K-Means Clustering of VGA Metrics")
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

    elbow_fig = px.line(
        elbow_df, x="k", y="inertia", markers=True, title="Elbow Method",
        labels={"k": "Number of groups", "inertia": "Within-group inertia"},
    )
    st.plotly_chart(elbow_fig, use_container_width=True)
    render_png_download(elbow_fig, "Download elbow chart as PNG", "vga_elbow.png", "clustering_elbow_png")
    group_options = ["All groups"] + [f"Group {group}" for group in sorted(clustered_df["cluster"].unique())]
    selected_group_label = st.selectbox("View group", group_options)
    selected_group = None if selected_group_label == "All groups" else int(selected_group_label.split()[-1])

    cluster_fig = render_cluster_map(clustered_df, clustering_walls, selected_group)
    st.plotly_chart(cluster_fig, use_container_width=True)
    render_png_download(cluster_fig, "Download cluster map as PNG", "vga_cluster_map.png", "clustering_map_png")
    st.download_button(
        "Download colour-coded plan as PNG",
        data=cluster_map_png(clustered_df, clustering_walls, selected_group),
        file_name="vga_metric_clusters.png",
        mime="image/png",
    )

    st.subheader("Automatic Spine Calculation")
    st.caption(
        "Edge-adjacent VGA cells form separate zones across all groups. One zone is "
        "selected by maximizing cell count × compactness (4π × cell area ÷ perimeter²); "
        "ties favor the zone closest to the mall center."
    )
    try:
        winning_group, winning_zone = select_mall_spine_zone(clustered_df)
        spine_result = calculate_cluster_spine(
            clustered_df[clustered_df["cluster"] == winning_group],
            count_df=clustered_df,
            grid_df=clustered_df,
            selected_zone=winning_zone,
        )
        st.caption(
            f"Selected zone: Group {winning_group} | "
            f"{spine_result['selected_zone_size']} cells | "
            f"compactness {spine_result['selected_zone_compactness']:.3f} | "
            f"score {spine_result['selected_zone_score']:.3f}"
        )
        spine_fig = render_cluster_spine_map(
            clustered_df,
            clustering_walls,
            winning_group,
            spine_result,
        )
        st.plotly_chart(spine_fig, use_container_width=True)
        render_png_download(
            spine_fig,
            "Download spine calculation as PNG",
            "vga_cluster_spine.png",
            "clustering_spine_png",
        )
    except ValueError as error:
        st.warning(f"Could not calculate a mall-wide spine: {error}")

    group_averages = clustered_df.groupby("cluster")[metric_columns].mean().reset_index()
    radar_fig = render_cluster_radar(group_averages, metric_columns, selected_group)
    st.plotly_chart(radar_fig, use_container_width=True)
    render_png_download(radar_fig, "Download radar chart as PNG", "vga_cluster_radar.png", "clustering_radar_png")
    group_averages.insert(0, "Group", group_averages.pop("cluster").map(lambda value: f"Group {value}"))
    st.subheader("Average Metrics by Group")
    st.dataframe(group_averages, use_container_width=True, hide_index=True)


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
    figure.colorbar(
        scatter,
        ax=axis,
        label="Projected people / m²",
        ticks=range(DENSITY_SCALE_MAX + 1),
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("Projected Crowd Density", color="white")
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

    if projection_upload is not None:
        try:
            imported = json.load(projection_upload)
            rows, imported_walls = extract_vga_rows(imported)
            source_df = pd.DataFrame(rows)
            if isinstance(imported, dict):
                source_metadata = imported.get("metadata", {})
                if not isinstance(source_metadata, dict):
                    source_metadata = {}
            if imported_walls:
                source_walls = imported_walls
            st.success(f"Loaded {len(source_df)} VGA points from `{projection_upload.name}`.")
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Could not load VGA results: {error}")
            return

    if source_df is None or source_df.empty or not {"x", "y"}.issubset(source_df.columns):
        st.info("Upload a VGA JSON session or run VGA analysis above to begin.")
        return
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

    vga_columns = [column for column in source_df.select_dtypes(include=np.number).columns if column not in {"x", "y"}]
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
                "Spine bearing from north (degrees)",
                min_value=0.0,
                max_value=359.999,
                value=float(spine_bearing_from_north),
                step=1.0,
                key="projection_spine_bearing",
                help="Prefilled from uploaded session metadata when available.",
            )
        with orientation_col:
            direction_orientation = st.radio(
                "Direction angle convention",
                options=["Clockwise", "Anti-clockwise"],
                index=0 if stored_orientation == "Clockwise" else 1,
                horizontal=True,
                key="projection_direction_orientation",
                help="These settings convert spine-relative values back to floorplan arrows.",
            )
    else:
        direction_orientation = stored_orientation

    feature_columns = shared_vga_columns[:10]
    projections, standard_deviations, neighbor_count = project_crowd_metrics_knn(
        source_df,
        historical_df,
        feature_columns,
        selected_targets,
    )
    usable_projections = {}
    usable_standard_deviations = {}
    for target, values in projections.items():
        mean_values = pd.Series(values, index=source_df.index).clip(lower=0)
        if pd.notna(mean_values).any():
            usable_projections[target] = mean_values
            usable_standard_deviations[target] = pd.Series(
                standard_deviations[target], index=source_df.index
            )
    projections = usable_projections
    standard_deviations = usable_standard_deviations
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
        use_container_width=True,
        hide_index=True,
    )

    density_column = next((column for column in projections if "density" in column.lower()), None)
    if density_column:
        projected_density = f"projected_{density_column}"
        high_density = projected_df[projected_density] > 3
        st.metric("Projected areas above 3 people / m²", f"{int(high_density.sum())} / {len(projected_df)} nodes")
        st.caption("Nodes above 3 people / m² are the mid-to-high density areas.")
        map_fig = go.Figure()
        for line in source_walls:
            x_values, y_values = line.xy
            map_fig.add_trace(go.Scatter(x=list(x_values), y=list(y_values), mode="lines", line=dict(color="#666"), showlegend=False))
        map_fig.add_trace(
            go.Scatter(
                x=projected_df["x"].tolist(),
                y=projected_df["y"].tolist(),
                mode="markers",
                marker=dict(
                    size=9,
                    color=projected_df[projected_density].tolist(),
                    colorscale=DENSITY_COLORSCALE,
                    cmin=0,
                    cmax=DENSITY_SCALE_MAX,
                    showscale=True,
                    colorbar=dict(title="people / m²", tick0=0, dtick=1),
                ),
                customdata=projected_df[
                    [
                        f"{projected_density}_std",
                        f"{projected_density}_lowest",
                        f"{projected_density}_highest",
                    ]
                ].to_numpy(),
                text=np.where(
                    high_density,
                    "MID-HIGH DENSITY (>3 people/m²)",
                    "Below threshold",
                ).tolist(),
                hovertemplate=(
                    "x=%{x}<br>y=%{y}"
                    "<br>mean density=%{marker.color:.2f}"
                    "<br>weighted standard deviation=%{customdata[0]:.2f}"
                    "<br>predicted range=%{customdata[1]:.2f}–%{customdata[2]:.2f} people/m²"
                    "<br>%{text}<extra></extra>"
                ),
                name="Projected density",
            )
        )
        map_fig.update_layout(title="Projected Crowd Density", template="plotly_dark", height=620, xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1), yaxis=dict(title="Y (mm)"))
        configure_floorplan_figure(map_fig)
        st.plotly_chart(map_fig, use_container_width=True)
        st.download_button("Download projected density map as PNG", projection_png(projected_df, projected_density, source_walls), "projected_crowd_density.png", "image/png", key="projection_map_png")
    else:
        st.info("No historical density column was found, so the >3 people / m² map cannot be calculated.")

    x_differences = np.diff(np.sort(projected_df["x"].dropna().unique()))
    y_differences = np.diff(np.sort(projected_df["y"].dropna().unique()))
    spacing_candidates = [
        float(np.median(differences[differences > 0]))
        for differences in (x_differences, y_differences)
        if np.any(differences > 0)
    ]
    vector_length = min(spacing_candidates) * 0.35 if spacing_candidates else 1.0

    for target in direction_targets:
        projected_direction = f"projected_{target}"
        fig_direction = go.Figure()
        for line in source_walls:
            x_values, y_values = line.xy
            fig_direction.add_trace(
                go.Scatter(
                    x=list(x_values),
                    y=list(y_values),
                    mode="lines",
                    line=dict(color="#666"),
                    showlegend=False,
                )
            )
        fig_direction.add_trace(
            go.Scatter(
                x=projected_df["x"],
                y=projected_df["y"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=projected_df[projected_direction],
                    colorscale="twilight",
                    cmin=0,
                    cmax=360,
                    showscale=True,
                    colorbar=dict(title="degrees from spine"),
                ),
                customdata=projected_df[
                    [f"{projected_direction}_std"]
                ].to_numpy(),
                hovertemplate=(
                    "x=%{x}<br>y=%{y}<br>predicted direction=%{marker.color:.2f}°"
                    "<br>weighted standard deviation=%{customdata[0]:.2f}°"
                    "<extra></extra>"
                ),
                name="Predicted direction",
            )
        )
        add_direction_arrows(
            fig_direction,
            projected_df["x"],
            projected_df["y"],
            projected_df[projected_direction],
            spine_bearing_from_north,
            clockwise=direction_orientation == "Clockwise",
            standard_deviations=projected_df[f"{projected_direction}_std"],
            base_length=vector_length,
        )
        fig_direction.update_layout(
            title=f"Predicted {crowd_metric_label(target)} with uncertainty-scaled arrows",
            template="plotly_dark",
            height=620,
            xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (mm)"),
        )
        configure_floorplan_figure(fig_direction)
        st.plotly_chart(fig_direction, use_container_width=True)
        st.caption(
            "Arrow direction uses the spine bearing and selected clockwise/anti-clockwise convention. "
            "Higher direction standard deviation produces shorter, more transparent arrows."
        )

    for target in deviation_targets:
        projected_deviation = f"projected_{target}"
        fig_deviation = go.Figure()
        for line in source_walls:
            x_values, y_values = line.xy
            fig_deviation.add_trace(
                go.Scatter(
                    x=list(x_values),
                    y=list(y_values),
                    mode="lines",
                    line=dict(color="#666"),
                    showlegend=False,
                )
            )
        fig_deviation.add_trace(
            go.Scatter(
                x=projected_df["x"],
                y=projected_df["y"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=projected_df[projected_deviation],
                    colorscale=[
                        [0.0, "#ffffff"],
                        [0.25, "#fff7bc"],
                        [0.5, "#fec44f"],
                        [0.75, "#d95f0e"],
                        [1.0, "#8c2d04"],
                    ],
                    cmin=0,
                    cmax=90,
                    showscale=True,
                    colorbar=dict(title="deviation angle (°)"),
                ),
                customdata=projected_df[
                    [f"{projected_deviation}_std"]
                ].to_numpy(),
                hovertemplate=(
                    "x=%{x}<br>y=%{y}<br>predicted deviation=%{marker.color:.2f}°"
                    "<br>weighted standard deviation=%{customdata[0]:.2f}°"
                    "<extra></extra>"
                ),
                name="Predicted deviation",
            )
        )
        add_deviation_rays(
            fig_deviation,
            projected_df["x"],
            projected_df["y"],
            projected_df[projected_deviation],
            spine_bearing_from_north,
            length=vector_length,
        )
        fig_deviation.update_layout(
            title=f"Predicted {crowd_metric_label(target)} with four rays per node",
            template="plotly_dark",
            height=620,
            xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (mm)"),
        )
        configure_floorplan_figure(fig_deviation)
        st.plotly_chart(fig_deviation, use_container_width=True)
        st.caption(
            "Four rays show both directions along the undirected spine and both "
            "deviation angles (0–90°) from it."
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


analysis_tab, clustering_tab, projection_tab = st.tabs(["2.1 VGA Analysis", "2.2 Metric Clustering", "2.3 Crowd Projection"])

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
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="floorplan_selector",
            )
            render_png_download(fig_plan, "Download floorplan as PNG", "vga_floorplan.png", "vga_floorplan_png")

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
                    st.session_state["vga_df"] = df_results
                    st.session_state["vga_floorplan_nodes"] = df_results.to_dict(orient="records")
                    st.session_state["vga_results"] = st.session_state["vga_floorplan_nodes"]
                    st.session_state["clustering_source_df"] = df_results
                    st.session_state["clustering_walls"] = wall_lines
                    st.session_state["vga_grid_spacing"] = float(grid_size)
                else:
                    progress_bar.empty()
                    st.error("Could not extract valid isovists. Selected points may be inside wall geometry.")

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

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
                st.plotly_chart(fig_heatmap, use_container_width=True)
                render_png_download(fig_heatmap, "Download heatmap as PNG", "vga_heatmap.png", "vga_heatmap_png")

        # Serialize Shapely wall lines to list of coordinate lists
        wall_lines_serialized = []
        if "wall_lines" in st.session_state:
            for line in st.session_state["wall_lines"]:
                wall_lines_serialized.append(list(line.coords))

        # Serialize Shapely selected room polygons
        selected_rooms_serialized = []
        if "selected_rooms" in st.session_state:
            for poly in st.session_state["selected_rooms"]:
                selected_rooms_serialized.append(list(poly.exterior.coords))

        # Build complete JSON package
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

        json_data = json.dumps(complete_vga_export, indent=2)

        st.download_button(
            label=" Download Complete VGA Session JSON",
            data=json_data,
            file_name="vga_complete_session.json",
            mime="application/json",
        )


with analysis_tab:
    render_analysis_tab()

with clustering_tab:
    render_clustering_tab()

with projection_tab:
    render_projection_tab()
