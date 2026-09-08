import json
from io import BytesIO
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from supabase import Client, create_client
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utils.navigation import render_home_button

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


def cluster_map_png(df, wall_lines, selected_group=None):
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
    """Fetch and flatten the historical metric payload used by Module 4."""
    page_size = 1000
    offset = 0
    records = []
    while True:
        response = (
            _supabase.table("vga_crowd_records")
            .select("metrics_data")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data or []
        records.extend(row.get("metrics_data", {}) for row in page)
        if len(page) < page_size:
            break
        offset += page_size
    return pd.json_normalize(records) if records else pd.DataFrame()


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


def projection_png(projected_df, density_column, walls):
    """Render the projected density map without requiring Plotly's image engine."""
    figure, axis = plt.subplots(figsize=(10, 8), facecolor="#111111")
    axis.set_facecolor("#111111")
    for line in walls:
        x_values, y_values = line.xy
        axis.plot(x_values, y_values, color="#666666", linewidth=1.0)
    scatter = axis.scatter(projected_df["x"], projected_df["y"], c=projected_df[density_column], cmap="YlOrRd", s=42)
    figure.colorbar(scatter, ax=axis, label="Projected people / m²")
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


def render_projection_tab():
    st.subheader("Projected Crowd Metrics")
    st.caption("Historical Pearson correlations from Supabase are used with standardized linear models to estimate crowd metrics for uploaded VGA nodes.")
    projection_upload = st.file_uploader("Upload previous VGA results (JSON)", type=["json"], key="projection_json_uploader")
    source_df = st.session_state.get("vga_df")
    source_walls = st.session_state.get("wall_lines", [])

    if projection_upload is not None:
        try:
            imported = json.load(projection_upload)
            rows, imported_walls = extract_vga_rows(imported)
            source_df = pd.DataFrame(rows)
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
        historical_df = fetch_historical_crowd_metrics(init_supabase())
    except Exception:
        st.warning("Supabase credentials or the `vga_crowd_records` table are unavailable.")
        return
    if historical_df.empty:
        st.info("No historical VGA and crowd records are available in Supabase yet.")
        return

    vga_columns = [column for column in source_df.select_dtypes(include=np.number).columns if column not in {"x", "y"}]
    crowd_columns = [column for column in historical_df.select_dtypes(include=np.number).columns if any(word in column.lower() for word in ("crowd", "density", "people", "pedestrian", "count", "volume"))]
    if not vga_columns or not crowd_columns:
        st.warning("The uploaded VGA data or Supabase data does not contain usable numeric metrics.")
        return

    selected_targets = st.multiselect("Crowd metrics to project", crowd_columns, default=[column for column in crowd_columns if "density" in column.lower()] or crowd_columns[:1], key="projection_targets")
    if not selected_targets:
        return

    projections = {}
    model_rows = []
    for target in selected_targets:
        best = None
        for feature in vga_columns:
            paired = historical_df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) < 3 or paired[feature].nunique() < 2:
                continue
            correlation = paired[feature].corr(paired[target])
            if pd.notna(correlation) and (best is None or abs(correlation) > abs(best["correlation"])):
                best = {"feature": feature, "correlation": float(correlation), "paired": paired}
        if best is None:
            continue
        paired = best["paired"]
        x_values = pd.to_numeric(source_df[best["feature"]], errors="coerce")
        x_std = paired[best["feature"]].std()
        slope = best["correlation"] * paired[target].std() / x_std if x_std else 0.0
        intercept = paired[target].mean() - slope * paired[best["feature"]].mean()
        projections[target] = (intercept + slope * x_values).clip(lower=0)
        model_rows.append({"Crowd metric": target, "Best VGA metric": best["feature"], "Pearson r": best["correlation"]})

    if not projections:
        st.warning("No VGA-to-crowd pairs have at least three complete historical records.")
        return
    projected_df = source_df[["x", "y"]].copy()
    for target, values in projections.items():
        projected_df[f"projected_{target}"] = values
    st.dataframe(pd.DataFrame(model_rows).style.format({"Pearson r": "{:.3f}"}), use_container_width=True, hide_index=True)

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
        map_fig.add_trace(go.Scatter(x=projected_df["x"].tolist(), y=projected_df["y"].tolist(), mode="markers", marker=dict(size=9, color=projected_df[projected_density].tolist(), colorscale="YlOrRd", showscale=True, colorbar=dict(title="people / m²")), text=np.where(high_density, "MID-HIGH DENSITY (>3 people/m²)", "Below threshold").tolist(), hovertemplate="x=%{x}<br>y=%{y}<br>projected density=%{marker.color:.2f}<br>%{text}<extra></extra>", name="Projected density"))
        map_fig.update_layout(title="Projected Crowd Density", template="plotly_dark", height=620, xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1), yaxis=dict(title="Y (mm)"))
        st.plotly_chart(map_fig, use_container_width=True)
        st.download_button("Download projected density map as PNG", projection_png(projected_df, projected_density, source_walls), "projected_crowd_density.png", "image/png", key="projection_map_png")
    else:
        st.info("No historical density column was found, so the >3 people / m² map cannot be calculated.")
    st.download_button("Download projected metrics as CSV", projected_df.to_csv(index=False), "projected_crowd_metrics.csv", "text/csv", key="projection_csv")


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

    if file_ext == ".json":
        # Process imported pre-computed session
        try:
            session_data = json.load(uploaded_file)
            st.session_state["vga_df"] = pd.DataFrame(session_data["vga_results"])
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
            " **Single Click Selection Active:** Target your selection using the **`+` crosshair**. Clicking a corridor selects strictly the corridor space without selecting enclosed interior rooms!"
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
                wall_lines, floorplan_bounds, selected_polys=selected_polygons
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
                    st.session_state["clustering_source_df"] = df_results
                    st.session_state["clustering_walls"] = wall_lines
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


