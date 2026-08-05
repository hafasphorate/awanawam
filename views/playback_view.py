# views/playback_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import io
import time

try:
    import ezdxf
except ImportError:
    ezdxf = None


def render_playback_view(wall_lines=None, tracking_df=None):
    st.subheader("📊 2D Movement Playback & Crowd Metric Heatmaps")

    # Initialize persistent state for mode 3 DXF walls and tracking DF
    if "dxf_walls" not in st.session_state:
        st.session_state.dxf_walls = []
    if "tracking_results_df" not in st.session_state:
        st.session_state.tracking_results_df = None
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    # Sync input walls if provided externally
    if wall_lines:
        st.session_state.dxf_walls = wall_lines
    else:
        wall_lines = st.session_state.dxf_walls

    df = tracking_df if tracking_df is not None else st.session_state.get("tracking_results_df", None)

    # --- UPLOAD SECTION ---
    st.info("💡 **Select your visualization mode below to upload data:**")

    mode = st.radio(
        "Choose Input & Visualization Mode:",
        options=[
            "1️⃣ Upload JSON (Plan View with Embedded CAD/Coordinates)",
            "2️⃣ Upload CSV Only (Standard Bounding Box / Rectangle Format)",
            "3️⃣ Upload CSV + DXF (Playback Overlaid on CAD Floorplan)"
        ],
        key="tab4_upload_mode"
    )

    col_up1, col_up2 = st.columns([3, 1])

    with col_up1:
        if "1️⃣" in mode:
            uploaded_json = st.file_uploader("📂 Upload Tracking JSON", type=["json"], key="tab4_json_upload")
            if uploaded_json:
                try:
                    df = _parse_tracking_file(uploaded_json)
                    st.session_state.tracking_results_df = df
                    st.success(f"✅ Loaded {len(df)} tracking records!")
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")

        elif "2️⃣" in mode:
            uploaded_csv = st.file_uploader("📂 Upload Tracking CSV", type=["csv"], key="tab4_csv_upload")
            if uploaded_csv:
                try:
                    df = _parse_tracking_file(uploaded_csv)
                    st.session_state.tracking_results_df = df
                    st.session_state.dxf_walls = []  # Clear walls for pure CSV mode
                    wall_lines = []
                    st.success(f"✅ Loaded {len(df)} tracking records!")
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")

        elif "3️⃣" in mode:
            col_csv, col_dxf = st.columns(2)
            with col_csv:
                uploaded_csv = st.file_uploader("📂 Upload Tracking CSV", type=["csv"], key="tab4_csv_combo")
            with col_dxf:
                uploaded_dxf = st.file_uploader("📐 Upload CAD Layout (DXF)", type=["dxf"], key="tab4_dxf_combo")

            # Store DXF independently into session state without overwriting CSV tracking
            if uploaded_dxf:
                parsed_walls = _parse_dxf_file(uploaded_dxf)
                if parsed_walls:
                    st.session_state.dxf_walls = parsed_walls
                    wall_lines = parsed_walls
                    st.success("✅ DXF Layout loaded!")

            if uploaded_csv:
                try:
                    df = _parse_tracking_file(uploaded_csv)
                    st.session_state.tracking_results_df = df
                    st.success(f"✅ Loaded {len(df)} tracking records!")
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")

    with col_up2:
        if st.button("🧪 Load Mock Data (CSV + DXF Layout)", use_container_width=True):
            df = _generate_mock_tracking_data()
            st.session_state.tracking_results_df = df
            st.session_state.dxf_walls = _generate_mock_walls()
            wall_lines = st.session_state.dxf_walls
            st.rerun()

    if df is None or df.empty:
        st.warning("⚠️ Please upload your dataset to launch playback.")
        return

    # --- NORMALIZE DATA & METRICS ---
    df.columns = [str(c).lower().strip() for c in df.columns]
    _map_column_aliases(df)

    required_cols = {"frame", "x", "y", "track_id"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        st.error(f"❌ Dataset is missing required columns: `{', '.join(missing_cols)}`")
        if st.button("🔄 Reset Data"):
            st.session_state.tracking_results_df = None
            st.session_state.dxf_walls = []
            st.rerun()
        return

    _calculate_derived_metrics(df)

    # --- CONTROLS BAR ---
    col_mode, col_metric, col_trail = st.columns([1.2, 1.5, 1])

    with col_mode:
        aggregation_mode = st.radio(
            "Heatmap Scope:",
            options=["Frame-by-Frame", "Full Session Aggregated"],
            index=0,
            key="playback_agg_scope"
        )

    with col_metric:
        metric_choice = st.selectbox(
            "Select Crowd Metric Heatmap:",
            options=["None (Trajectory Only)", "Crowd Volume", "Density", "Speed", "Direction"],
            index=1,
            key="playback_metric_selectbox"
        )

    with col_trail:
        trail_length = st.number_input(
            "Trail Length (Frames)",
            min_value=1,
            max_value=100,
            value=15,
            step=5,
            key="playback_trail_input"
        )

    # --- ANIMATION & PLAYBACK CONTROLS ---
    max_frame = int(df["frame"].max())
    min_frame = int(df["frame"].min())

    if "current_playback_frame" not in st.session_state:
        st.session_state.current_playback_frame = min_frame

    col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 1, 3])

    with col_p1:
        if st.button("▶️ Play", use_container_width=True):
            st.session_state.is_playing = True
    with col_p2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.is_playing = False
    with col_p3:
        fps = st.select_slider("Playback Speed (FPS)", options=[1, 2, 5, 10, 15, 30], value=10)

    with col_p4:
        current_frame = st.slider(
            "Frame Slider",
            min_value=min_frame,
            max_value=max_frame,
            value=st.session_state.current_playback_frame,
            step=1,
            key="playback_frame_slider"
        )
        st.session_state.current_playback_frame = current_frame

    # --- MAIN PLAYBACK FIGURE ---
    st.markdown("### 📽️ Interactive Playback")
    fig = go.Figure()

    _draw_cad_walls(fig, wall_lines)

    if metric_choice != "None (Trajectory Only)":
        active_heatmap_df = df if aggregation_mode == "Full Session Aggregated" else df[df["frame"] <= current_frame]
        _add_heatmap_layer(fig, active_heatmap_df, metric_choice)

    start_frame = max(min_frame, current_frame - trail_length)
    frame_window_df = df[(df["frame"] >= start_frame) & (df["frame"] <= current_frame)]

    for track_id, group in frame_window_df.groupby("track_id"):
        fig.add_trace(go.Scatter(
            x=group["x"],
            y=group["y"],
            mode="lines",
            line=dict(width=2),
            hoverinfo="none",
            showlegend=False,
            opacity=0.6
        ))

    active_agents = df[df["frame"] == current_frame]
    if not active_agents.empty:
        fig.add_trace(go.Scatter(
            x=active_agents["x"],
            y=active_agents["y"],
            mode="markers+text",
            marker=dict(size=10, color="#FF007F", symbol="circle", line=dict(color="#FFFFFF", width=1)),
            text=[f"ID {tid}" for tid in active_agents["track_id"]],
            textposition="top center",
            textfont=dict(size=10, color="#FFFFFF"),
            name="Active Pedestrians",
            hoverinfo="text",
            hovertext=[f"Pedestrian #{tid}<br>X: {x:.2f}<br>Y: {y:.2f}" 
                       for tid, x, y in zip(active_agents["track_id"], active_agents["x"], active_agents["y"])]
        ))

    _configure_layout(fig, df, wall_lines)
    st.plotly_chart(fig, use_container_width=True)

    # Frame Loop for Playback Animation
    if st.session_state.is_playing:
        if st.session_state.current_playback_frame < max_frame:
            st.session_state.current_playback_frame += 1
            time.sleep(1.0 / fps)
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.session_state.current_playback_frame = min_frame

    # --- AGGREGATED HEATMAP AT BOTTOM ---
    st.markdown("---")
    st.markdown("### 📈 Aggregated Session Dynamics & Directional Vector Field")
    st.caption("Spatial heatmaps and scaled directional velocity arrows calculated across all frames.")

    col_v1, col_v2, col_v3 = st.columns([1, 1, 2])
    with col_v1:
        show_vectors = st.checkbox("🎯 Show Direction Arrows", value=True, key="show_vector_arrows")
    with col_v2:
        grid_bins = st.slider("Grid Density (Bins)", min_value=8, max_value=30, value=15, step=1, key="vector_grid_bins")
    with col_v3:
        arrow_scale = st.slider("Arrow Length Multiplier", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="vector_scale")

    agg_fig = go.Figure()

    _draw_cad_walls(agg_fig, wall_lines)

    if metric_choice != "None (Trajectory Only)":
        _add_heatmap_layer(agg_fig, df, metric_choice, is_summary=True)
    else:
        _add_heatmap_layer(agg_fig, df, "Crowd Volume", is_summary=True)

    if show_vectors:
        _add_vector_field(agg_fig, df, grid_bins=grid_bins, scale_multiplier=arrow_scale)

    _configure_layout(agg_fig, df, wall_lines)
    st.plotly_chart(agg_fig, use_container_width=True)


# --- HELPER FUNCTIONS ---

def _map_column_aliases(df):
    mappings = {
        "frame": ["frame_idx", "frame_id", "frames", "frame_num", "step", "timestamp"],
        "track_id": ["track_id", "id", "pedestrian_id", "agent_id", "track_idx", "person_id"],
        "x": ["world_x", "pos_x", "x_coord", "x_pos", "px", "x_m"],
        "y": ["world_y", "pos_y", "y_coord", "y_pos", "py", "y_m"]
    }
    for target, aliases in mappings.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df.rename(columns={alias: target}, inplace=True)
                    break


def _calculate_derived_metrics(df):
    df.sort_values(by=["track_id", "frame"], inplace=True)
    df["dx"] = df.groupby("track_id")["x"].diff().fillna(0)
    df["dy"] = df.groupby("track_id")["y"].diff().fillna(0)
    df["dt"] = df.groupby("track_id")["frame"].diff().fillna(1)
    
    if "speed" not in df.columns:
        df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / np.maximum(df["dt"], 1)
    if "angle" not in df.columns:
        df["angle"] = np.degrees(np.arctan2(df["dy"], df["dx"])) % 360


def _draw_cad_walls(fig, wall_lines):
    wall_x, wall_y = [], []
    for line in wall_lines:
        if hasattr(line, "xy"):
            x, y = line.xy
            wall_x.extend([x[0], x[1], None])
            wall_y.extend([y[0], y[1], None])
        elif isinstance(line, (list, tuple)):
            for i in range(len(line) - 1):
                wall_x.extend([line[i][0], line[i+1][0], None])
                wall_y.extend([line[i][1], line[i+1][1], None])

    if wall_x:
        fig.add_trace(go.Scatter(
            x=wall_x, y=wall_y,
            mode="lines",
            line=dict(color="#00ADB5", width=1.8),
            name="CAD Floorplan Walls",
            hoverinfo="none",
            showlegend=True
        ))


def _add_heatmap_layer(fig, target_df, metric_choice, is_summary=False):
    if target_df.empty:
        return

    colorscale = "Jet"
    z_vals = None
    histfunc = "avg"

    if metric_choice in ["Crowd Volume", "None (Trajectory Only)"]:
        z_vals = np.ones(len(target_df))
        colorscale = "Viridis"
        histfunc = "sum"
    elif metric_choice == "Density":
        z_vals = np.ones(len(target_df))
        colorscale = "Hot"
        histfunc = "count"
    elif metric_choice == "Speed":
        z_vals = target_df["speed"]
        colorscale = "Plasma"
        histfunc = "avg"
    elif metric_choice == "Direction":
        z_vals = target_df["angle"]
        colorscale = "HSV"
        histfunc = "avg"

    title_suffix = " (Full Aggregated)" if is_summary else ""

    fig.add_trace(go.Histogram2dContour(
        x=target_df["x"],
        y=target_df["y"],
        z=z_vals,
        histfunc=histfunc,
        colorscale=colorscale,
        opacity=0.6,
        showscale=True,
        name=f"{metric_choice}{title_suffix}",
        contours=dict(coloring="heatmap")
    ))


def _add_vector_field(fig, df, grid_bins=15, scale_multiplier=2.0):
    moving_df = df[(df["dx"] != 0) | (df["dy"] != 0)].copy()
    if moving_df.empty:
        return

    # Spatial dimensions
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    spatial_span = max(x_max - x_min, y_max - y_min)

    # Base scale dynamically calculated from total spatial extent (handles mm, meters, etc.)
    base_arrow_len = (spatial_span / grid_bins) * 0.7 * scale_multiplier

    x_bins = np.linspace(x_min, x_max, grid_bins)
    y_bins = np.linspace(y_min, y_max, grid_bins)

    moving_df["x_bin"] = pd.cut(moving_df["x"], bins=x_bins, labels=False)
    moving_df["y_bin"] = pd.cut(moving_df["y"], bins=y_bins, labels=False)

    binned_vectors = moving_df.groupby(["x_bin", "y_bin"], observed=True).agg(
        avg_dx=("dx", "mean"),
        avg_dy=("dy", "mean"),
        count=("track_id", "count")
    ).reset_index()

    if binned_vectors.empty:
        return

    arrow_x, arrow_y = [], []
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    for _, row in binned_vectors.iterrows():
        bx, by = int(row["x_bin"]), int(row["y_bin"])
        if bx >= len(x_centers) or by >= len(y_centers):
            continue

        cx, cy = x_centers[bx], y_centers[by]
        dx, dy = row["avg_dx"], row["avg_dy"]
        
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude == 0:
            continue

        norm_dx = (dx / magnitude) * base_arrow_len
        norm_dy = (dy / magnitude) * base_arrow_len

        end_x = cx + norm_dx
        end_y = cy + norm_dy

        # Arrow shaft
        arrow_x.extend([cx, end_x, None])
        arrow_y.extend([cy, end_y, None])

        # Arrow head wings
        wing_angle = np.pi / 6
        wing_len = 0.35 * base_arrow_len
        base_angle = np.arctan2(norm_dy, norm_dx)

        left_x = end_x - wing_len * np.cos(base_angle - wing_angle)
        left_y = end_y - wing_len * np.sin(base_angle - wing_angle)
        right_x = end_x - wing_len * np.cos(base_angle + wing_angle)
        right_y = end_y - wing_len * np.sin(base_angle + wing_angle)

        arrow_x.extend([end_x, left_x, None, end_x, right_x, None])
        arrow_y.extend([end_y, left_y, None, end_y, right_y, None])

    if arrow_x:
        fig.add_trace(go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="lines",
            line=dict(color="#00FFFF", width=2.5),
            name="Direction Vector Field",
            hoverinfo="none",
            showlegend=True
        ))


def _configure_layout(fig, df, wall_lines):
    padding = 1.0
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    # Consider DXF bounds if present
    if wall_lines:
        for line in wall_lines:
            if hasattr(line, "xy"):
                x, y = line.xy
                x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
                y_min, y_max = min(y_min, min(y)), max(y_max, max(y))

    fig.update_layout(
        template="plotly_dark",
        height=550,
        xaxis=dict(title="X Position", range=[x_min - padding, x_max + padding], scaleanchor="y", scaleratio=1, showgrid=True),
        yaxis=dict(title="Y Position", range=[y_min - padding, y_max + padding], showgrid=True),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )


def _parse_tracking_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".json"):
        content = json.load(uploaded_file)
        if isinstance(content, list):
            return pd.read_json(io.StringIO(json.dumps(content)))
        elif isinstance(content, dict):
            for key in ["tracks", "data", "records", "results", "detections", "trajectories"]:
                if key in content and isinstance(content[key], list):
                    return pd.DataFrame(content[key])

            flattened_records = []
            for k, v in content.items():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            if "track_id" not in item:
                                item["track_id"] = k
                            flattened_records.append(item)

            if flattened_records:
                return pd.DataFrame(flattened_records)
            return pd.DataFrame.from_dict(content, orient="index").T
    raise ValueError("Unsupported file format. Please upload CSV or JSON.")


def _parse_dxf_file(uploaded_dxf):
    if ezdxf is None:
        st.warning("⚠️ `ezdxf` library is not installed. Run `pip install ezdxf` to parse DXF files.")
        return []

    try:
        dxf_bytes = uploaded_dxf.read()
        doc = ezdxf.read(io.StringIO(dxf_bytes.decode('utf-8', errors='ignore')))
        msp = doc.modelspace()

        lines = []
        for entity in msp.query('LINE LWPOLYLINE POLYLINE'):
            if entity.dxftype() == 'LINE':
                lines.append([(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)])
            elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                points = [(p[0], p[1]) for p in entity.get_points()]
                lines.append(points)
        return lines
    except Exception as e:
        st.error(f"Error parsing DXF file: {e}")
        return []


def _generate_mock_tracking_data():
    np.random.seed(42)
    records = []
    for tid in range(1, 9):
        start_x, start_y = np.random.uniform(100.0, 800.0), np.random.uniform(100.0, 800.0)
        vx, vy = np.random.uniform(-10.0, 10.0), np.random.uniform(-10.0, 10.0)
        x, y = start_x, start_y
        for frame in range(0, 100, 2):
            x += vx + np.random.normal(0, 2.0)
            y += vy + np.random.normal(0, 2.0)
            records.append({
                "frame": frame,
                "track_id": tid,
                "x": float(x),
                "y": float(y)
            })
    return pd.DataFrame(records)


def _generate_mock_walls():
    return [
        [(0, 0), (1000, 0)],
        [(1000, 0), (1000, 1000)],
        [(1000, 1000), (0, 1000)],
        [(0, 1000), (0, 0)]
    ]