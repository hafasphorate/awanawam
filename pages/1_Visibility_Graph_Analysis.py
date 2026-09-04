import json
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.vga_engine import (
    compute_isovist_metrics,
    extract_dxf_walls,
    generate_isovist_polygon,
    process_cad_file,
)

st.set_page_config(page_title="Visibility Graph Analysis", layout="wide")

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

st.title("1. Visibility Graph Analysis (VGA)")
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

uploaded_file = st.file_uploader("Upload CAD Floorplan (DXF, DWG) or Saved Session (JSON)", type=["dxf", "dwg", "json"])


@st.cache_data
def extract_enclosed_rooms(_wall_lines, snap_distance=1200):
    """Reconstructs enclosed room polygons and corridor spaces with automatic interior hole detection."""
    lines = list(_wall_lines)
    
    endpoints = []
    for l in lines:
        coords = list(l.coords)
        endpoints.append(Point(coords[0]))
        endpoints.append(Point(coords[-1]))

    closing_lines = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1, p2 = endpoints[i], endpoints[j]
            dist = p1.distance(p2)
            if 10.0 < dist <= snap_distance:
                closing_lines.append(LineString([p1, p2]))

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
                f"⌛ **Computing Graph Topology (Integration & Entropy)...** Node {completed}/{num_nodes} ({int(progress_ratio * 100)}%) | **Est. remaining:** `{time_str}`"
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


def render_interactive_floorplan(wall_lines, bounds, selected_polys=None):
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

    grid_step = max(200, (maxx - minx) / 60)
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
    return fig


def render_vga_heatmap_with_underlay(df, metric_column, wall_lines):
    """Renders VGA metric heatmap with original CAD floorplan wall lines underlaid."""
    fig = go.Figure()

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
                colorbar=dict(title=metric_column),
                opacity=0.9,
            ),
            text=[f"{metric_column}: {v}" for v in df[metric_column]],
            hoverinfo="x+y+text",
            name="VGA Data",
        )
    )

    fig.update_layout(
        title=dict(text=f"Spatial Map: {metric_column}", x=0.01),
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

    return fig


def render_cluster_map(df, wall_lines, selected_group=None):
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
    return fig


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
    st.info(
        "The silhouette coefficient measures how well each point fits its own group versus other groups. "
        "Values near 1 indicate clear separation, values near 0 indicate overlapping groups, and negative "
        "values suggest points may be assigned to the wrong group."
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
    st.caption(f"Metrics used: {', '.join(metric_columns)}")
    metric_col, score_col = st.columns(2)
    metric_col.metric("Groups", selected_k)
    score_col.metric("Silhouette coefficient", f"{silhouette:.3f}")

    elbow_fig = px.line(
        elbow_df, x="k", y="inertia", markers=True, title="Elbow Method",
        labels={"k": "Number of groups", "inertia": "Within-group inertia"},
    )
    st.plotly_chart(elbow_fig, use_container_width=True)
    group_options = ["All groups"] + [f"Group {group}" for group in sorted(clustered_df["cluster"].unique())]
    selected_group_label = st.selectbox("View group", group_options)
    selected_group = None if selected_group_label == "All groups" else int(selected_group_label.split()[-1])
    cluster_fig = render_cluster_map(clustered_df, clustering_walls, selected_group)
    st.plotly_chart(cluster_fig, use_container_width=True)
    try:
        st.download_button(
            "Download colour-coded plan as PNG",
            data=cluster_fig.to_image(format="png"),
            file_name="vga_metric_clusters.png",
            mime="image/png",
        )
    except (ValueError, ImportError):
        st.warning("PNG export requires the `kaleido` package. Install the project requirements and try again.")

    group_averages = clustered_df.groupby("cluster")[metric_columns].mean().reset_index()
    group_averages.insert(0, "Group", group_averages.pop("cluster").map(lambda value: f"Group {value}"))
    st.subheader("Average Metrics by Group")
    st.dataframe(group_averages, use_container_width=True, hide_index=True)


analysis_tab, clustering_tab = st.tabs(["VGA Analysis", "Metric Clustering"])

with clustering_tab:
    render_clustering_tab()

with analysis_tab:
    st.empty()

if uploaded_file is not None:
    file_ext = "." + uploaded_file.name.split(".")[-1].lower()

    if file_ext == ".json":
        # Process imported pre-computed session
        try:
            session_data = json.load(uploaded_file)
            st.session_state["vga_df"] = pd.DataFrame(session_data["vga_results"])
            
            restored_walls = [
                LineString(coords) for coords in session_data["floorplan"]["wall_lines"]
            ]
            st.session_state["wall_lines"] = restored_walls
            
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
            "💡 **Single Click Selection Active:** Target your selection using the **`+` crosshair**. Clicking a corridor selects strictly the corridor space without selecting enclosed interior rooms!"
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
            if st.button("🔴 Reset Selected Regions"):
                st.session_state["selected_rooms"] = []
                st.rerun()

        selected_polygons = st.session_state["selected_rooms"]

        if selection_mode_option == "Click Inside Rooms to Select Zones":
            fig_plan = render_interactive_floorplan(
                wall_lines, floorplan_bounds, selected_polys=selected_polygons
            )

            chart_events = st.plotly_chart(
                fig_plan,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="floorplan_selector",
            )

            if chart_events and "selection" in chart_events:
                pts = chart_events["selection"].get("points", [])
                if pts:
                    click_x = pts[0]["x"]
                    click_y = pts[0]["y"]
                    click_point = Point(click_x, click_y)

                    candidate_rooms = [r for r in enclosed_rooms if r.contains(click_point)]
                    if candidate_rooms:
                        matched_room = candidate_rooms[0]

                        if not any(r.equals(matched_room) for r in st.session_state["selected_rooms"]):
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
                            f"⌛ **Phase 1: Analyzing Isovists {completed}/{total_points}** ({int(progress_ratio * 100)}%) | **Est. time remaining:** `{time_str}`"
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
                else:
                    progress_bar.empty()
                    st.error("Could not extract valid isovists. Selected points may be inside wall geometry.")

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

        st.subheader("VGA Heatmap Visualizer")
        available_metrics = [c for c in df.columns if c not in ["x", "y"]]

        if available_metrics:
            selected_metric = st.selectbox("Select Metric to Render:", available_metrics)

            if selected_metric in df.columns and "wall_lines" in st.session_state:
                fig_heatmap = render_vga_heatmap_with_underlay(
                    df, selected_metric, st.session_state["wall_lines"]
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

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
                "file_name": uploaded_file.name if uploaded_file else "imported_session"
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
                "selected_rooms": selected_rooms_serialized
            },
            "vga_results": df.to_dict(orient="records")
        }

        json_data = json.dumps(complete_vga_export, indent=2)

        st.download_button(
            label="📥 Download Complete VGA Session JSON",
            data=json_data,
            file_name="vga_complete_session.json",
            mime="application/json",
        )


