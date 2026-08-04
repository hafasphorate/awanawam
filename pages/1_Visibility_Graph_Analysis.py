import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
import streamlit as st

from utils.vga_engine import (
    compute_isovist_metrics,
    extract_dxf_walls,
    generate_isovist_polygon,
)

st.set_page_config(page_title="Visibility Graph Analysis", layout="wide")

st.title("1. Visibility Graph Analysis (VGA)")
st.markdown(
    "Upload a DXF floorplan, click inside one or more rooms to auto-detect and highlight their enclosed boundaries, and compute spatial metrics."
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

uploaded_file = st.file_uploader("Upload DXF Floorplan", type=["dxf"])


@st.cache_data
def extract_enclosed_rooms(_wall_lines):
    """Reconstructs enclosed room polygons from DXF line segments using Shapely polygonize."""
    merged_walls = unary_union(_wall_lines)
    enclosed_polygons = list(polygonize(merged_walls))
    return enclosed_polygons


def compute_graph_topology_with_progress(vga_results, isovist_polys, status_container, progress_bar):
    """Computes inter-isovist graph integration and entropy with active progress tracking."""
    num_nodes = len(vga_results)
    if num_nodes == 0:
        return vga_results

    # Build adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    start_time = time.time()

    for i in range(num_nodes):
        pt_i = Point(vga_results[i]["x"], vga_results[i]["y"])
        poly_i = isovist_polys[i]

        for j in range(i + 1, num_nodes):
            pt_j = Point(vga_results[j]["x"], vga_results[j]["y"])
            poly_j = isovist_polys[j]

            # Mutual visibility condition
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
                f"⌛ **Computing Graph Topology (Integration & Entropy)...** Node {completed}/{num_nodes} ({int(progress_ratio * 100)}%) | **Est. remaining:** `{time_str}`"
            )

    # Calculate Mean Depth and Integration for each node
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
            # Integration metric (inverse of Relative Asymmetry)
            integration = 1.0 / (2.0 * (mean_depth - 1.0) / max(1, reachable_nodes - 2)) if mean_depth > 1 else 0.0
        else:
            mean_depth = 0.0
            integration = 0.0

        vga_results[i]["mean_depth"] = round(mean_depth, 3)
        vga_results[i]["integration"] = round(integration, 3)

    return vga_results


def render_interactive_floorplan(wall_lines, selected_polys=None):
    """Builds interactive Plotly figure highlighting all selected room zones in green."""
    fig = go.Figure()

    # Draw DXF Wall Lines
    for line in wall_lines:
        x, y = line.xy
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                line=dict(color="#00ADB5", width=1.5),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Highlight all selected room boundaries
    if selected_polys:
        for idx, poly in enumerate(selected_polys):
            x_poly, y_poly = poly.exterior.xy
            fig.add_trace(
                go.Scatter(
                    x=list(x_poly),
                    y=list(y_poly),
                    fill="toself",
                    fillcolor="rgba(0, 255, 0, 0.35)",
                    line=dict(color="#00FF00", width=2),
                    name=f"Zone {idx + 1}",
                    hoverinfo="name",
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
        dragmode="select",
        hovermode="closest",
    )
    return fig


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Parsing DXF wall geometry and building enclosed spatial zones..."):
        wall_lines = extract_dxf_walls(tmp_path)
        strtree = STRtree(wall_lines)
        enclosed_rooms = extract_enclosed_rooms(wall_lines)

    st.success(
        f"Extracted {len(wall_lines)} wall boundary segments and detected {len(enclosed_rooms)} enclosed spatial zones."
    )

    all_bounds = [w.bounds for w in wall_lines]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    st.subheader("Interactive Public Space Selection")
    st.info(
        "💡 **Multi-Room Selection Active:** Click anywhere inside a room to select it. Click additional rooms to select multiple areas at once! Highlighted green areas will be analyzed."
    )

    selection_mode_option = st.radio(
        "Selection Mode:",
        ["Full Floorplan", "Click Inside Rooms to Select Zones"],
        horizontal=True,
    )

    if "selected_rooms" not in st.session_state:
        st.session_state["selected_rooms"] = []

    # Reset Selection Control
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔴 Reset Selected Regions"):
            st.session_state["selected_rooms"] = []
            st.rerun()

    selected_polygons = st.session_state["selected_rooms"]

    if selection_mode_option == "Click Inside Rooms to Select Zones":
        fig_plan = render_interactive_floorplan(wall_lines, selected_polys=selected_polygons)

        chart_events = st.plotly_chart(
            fig_plan,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
        )

        if chart_events and "selection" in chart_events:
            pts = chart_events["selection"].get("points", [])
            if pts:
                click_x = pts[0]["x"]
                click_y = pts[0]["y"]
                click_point = Point(click_x, click_y)

                matched_room = None
                for room in enclosed_rooms:
                    if room.contains(click_point):
                        matched_room = room
                        break

                if matched_room:
                    # Prevent duplicate additions
                    if not any(r.equals(matched_room) for r in st.session_state["selected_rooms"]):
                        st.session_state["selected_rooms"].append(matched_room)
                        st.rerun()
                else:
                    st.warning("Click fell outside valid room boundaries. Click inside an open room area.")

        if selected_polygons:
            total_area = sum(p.area for p in selected_polygons) / 1e6
            st.success(
                f"✅ **{len(selected_polygons)} Zone(s) Selected & Highlighted!** Combined Area: `{round(total_area, 2)} m²`"
            )

    if st.button("Run Visibility Analysis"):
        # Optimization: Restrict grid bounds strictly to selected polygon bounding boxes
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

            # Phase 1: Isovist Field Calculations
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
                        f"⌛ **Phase 1: Analyzing Isovists {completed}/{total_points}** ({int(progress_ratio * 100)}%) | **Est. time remaining:** `{time_str}`"
                    )

            # Phase 2: Topology Metrics with Real-Time Progress Bar
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
            else:
                progress_bar.empty()
                st.error("Could not extract valid isovists. Selected points may be inside wall geometry.")

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

        st.subheader("VGA Heatmap Visualizer")
        available_metrics = [c for c in df.columns if c not in ["x", "y"]]

        if available_metrics:
            selected_metric = st.selectbox("Select Metric to Render:", available_metrics)

            if selected_metric in df.columns:
                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color=selected_metric,
                    color_continuous_scale="Viridis",
                    title=f"Spatial Map: {selected_metric}",
                )
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig, use_container_width=True)

            json_data = df.to_json(orient="records")
            st.download_button(
                label="📥 Download Complete VGA Metrics JSON",
                data=json_data,
                file_name="vga_analysis_results.json",
                mime="application/json",
            )