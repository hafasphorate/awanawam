import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import io

try:
    import ezdxf
except ImportError:
    ezdxf = None


# =============================================================================
# MAIN VIEW RENDERER
# =============================================================================

def render_playback_view(wall_lines=None, tracking_df=None):
    st.title("📊 2D Movement Playback & Crowd Metric Heatmaps")

    # --- 1. SESSION STATE INITIALIZATION ---
    if "wall_lines" not in st.session_state:
        st.session_state.wall_lines = []
    if "tracking_results_df" not in st.session_state:
        st.session_state.tracking_results_df = None

    if wall_lines:
        st.session_state.wall_lines = wall_lines

    # --- 2. DATA IMPORT SECTION ---
    st.markdown("## 📥 1. Import Data")
    
    upload_mode = st.radio(
        "Select Data Input Method:",
        options=[
            "1️⃣ JSON (Mapped Trajectories + Embedded CAD Floorplan)",
            "2️⃣ CSV Only (Standard Bounding Box / Grid Trajectories)",
            "3️⃣ CSV + DXF Layout (Overlay Trajectories onto CAD Layout)"
        ],
        key="pb_upload_mode"
    )

    col_up, col_mock = st.columns([3, 1])

    with col_up:
        if "1️⃣" in upload_mode:
            uploaded_json = st.file_uploader("Upload JSON File", type=["json"], key="pb_json")
            if uploaded_json:
                parsed_df, extracted_walls = _parse_tracking_json(uploaded_json)
                if parsed_df is not None:
                    st.session_state.tracking_results_df = parsed_df
                    if extracted_walls:
                        st.session_state.wall_lines = extracted_walls
                    st.success(f"✅ Successfully loaded {len(parsed_df)} records from JSON.")

        elif "2️⃣" in upload_mode:
            uploaded_csv = st.file_uploader("Upload Trajectory CSV", type=["csv"], key="pb_csv")
            if uploaded_csv:
                parsed_df = pd.read_csv(uploaded_csv)
                st.session_state.tracking_results_df = parsed_df
                st.session_state.wall_lines = []
                st.success(f"✅ Successfully loaded {len(parsed_df)} CSV records.")

        elif "3️⃣" in upload_mode:
            c1, c2 = st.columns(2)
            with c1:
                uploaded_csv = st.file_uploader("Upload Trajectory CSV", type=["csv"], key="pb_csv_combo")
            with c2:
                uploaded_dxf = st.file_uploader("Upload CAD Layout (.dxf)", type=["dxf"], key="pb_dxf_combo")

            if uploaded_csv:
                st.session_state.tracking_results_df = pd.read_csv(uploaded_csv)
            if uploaded_dxf:
                st.session_state.wall_lines = _parse_dxf_file(uploaded_dxf)

            if uploaded_csv and uploaded_dxf:
                st.success("✅ CSV Trajectories and DXF Layout successfully merged!")

    with col_mock:
        st.markdown("**Test Drive:**")
        if st.button("🧪 Load Sample Data", use_container_width=True):
            st.session_state.tracking_results_df = _generate_mock_tracking_data()
            st.session_state.wall_lines = _generate_mock_walls()
            st.rerun()

    # Get active dataset
    df = tracking_df if tracking_df is not None else st.session_state.get("tracking_results_df", None)
    active_walls = st.session_state.get("wall_lines", [])

    if df is None or df.empty:
        st.info("👆 Please upload data or click **Load Sample Data** to proceed.")
        return

    # --- 3. DATA CLEANING & METRICS CALCULATION ---
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    _map_column_aliases(df)

    required_cols = {"frame", "x", "y", "track_id"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"❌ Missing required columns in data: `{', '.join(missing)}`")
        return

    # Choose time basis without breaking older datasets.
    fps = None
    if "fps" in df.columns:
        fps_series = pd.to_numeric(df["fps"], errors="coerce").dropna()
        if not fps_series.empty:
            fps = float(fps_series.iloc[0])
    elif "frame_rate" in df.columns:
        fps_series = pd.to_numeric(df["frame_rate"], errors="coerce").dropna()
        if not fps_series.empty:
            fps = float(fps_series.iloc[0])

    time_mode = st.radio(
        "Time basis for motion metrics",
        options=[
            "Legacy frame-count (matches old exports)",
            "Seconds (time-normalized, recommended when fps is fixed)",
        ],
        index=0,
        key="pb_time_mode",
        help="Legacy keeps the original frame-based speed calculations. Seconds uses frame rate to convert to real elapsed time when available.",
    )
    normalize_time = time_mode.startswith("Seconds")

    # Compute metrics (Speed, Direction, Density, Volume)
    df = _calculate_derived_metrics(df, fps=fps, normalize_time=normalize_time)

    # --- 4. METRIC DISPLAY CONTROLS ---
    st.markdown("---")
    st.markdown("## ⚙️ 2. Visual & Heatmap Settings")

    c_metric, c_bins, c_trail = st.columns([1.5, 1, 1])
    
    with c_metric:
        metric_choice = st.selectbox(
            "Heatmap Metric:",
            options=["Volume", "Density", "Speed", "Direction"],
            index=0,
            key="pb_selected_metric"
        )
    with c_bins:
        grid_bins = st.slider("Heatmap Resolution (Grid Bins)", min_value=10, max_value=50, value=20, step=5)
    with c_trail:
        trail_length = st.number_input("Trail History Length (Frames)", min_value=0, max_value=50, value=10, step=2)

    # --- 5. DATA VIEWS (Interactive Playback & Summary) ---
    st.markdown("---")
    st.markdown("## 📺 3. View Data")

    tab_video, tab_frame, tab_agg = st.tabs([
        "🎬 Interactive Video Playback",
        "🖼️ Frame-by-Frame Inspector",
        "📈 Aggregated Summary Heatmap"
    ])

    # --- TAB 1: Client-Side Native Plotly Video Playback ---
    with tab_video:
        st.subheader("Animated Movement Playback")
        st.caption("Press Play inside the chart viewport to view dynamic heatmaps in real-time.")
        
        anim_fig = _build_plotly_animation(df, active_walls, metric_choice, trail_length)
        st.plotly_chart(anim_fig, use_container_width=True)

    # --- TAB 2: Frame-by-Frame Manual View ---
    with tab_frame:
        st.subheader("Single Frame Inspector")
        min_f, max_f = int(df["frame"].min()), int(df["frame"].max())
        
        selected_frame = st.slider("Select Frame Index", min_value=min_f, max_value=max_f, value=min_f, step=1)
        
        frame_fig = _build_single_frame_figure(df, active_walls, selected_frame, metric_choice, trail_length)
        st.plotly_chart(frame_fig, use_container_width=True)

    # --- TAB 3: Full Session Aggregated Heatmap ---
    with tab_agg:
        st.subheader("Full Session Aggregated Heatmap & Vector Field")
        
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            show_vectors = st.checkbox("Show Direction Flow Vectors", value=True)
        with c_v2:
            arrow_scale = st.slider("Vector Arrow Scaling", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

        agg_fig = _build_aggregated_figure(df, active_walls, metric_choice, grid_bins, show_vectors, arrow_scale)
        st.plotly_chart(agg_fig, use_container_width=True)

    # --- 6. EXPORT SECTION ---
    st.markdown("---")
    st.markdown("## 📤 4. Export Data")
    
    exp_col1, exp_col2 = st.columns([2, 1])
    with exp_col1:
        st.write("Export the current workspace session (CAD Layout + Trajectories + Computed Metrics) as JSON.")
    with exp_col2:
        json_bytes = _export_to_json(df, active_walls)
        st.download_button(
            label="💾 Download JSON Package",
            data=json_bytes,
            file_name="crowd_analytics_export.json",
            mime="application/json",
            use_container_width=True
        )


# =============================================================================
# COMPUTATION & METRIC HELPERS
# =============================================================================

def _map_column_aliases(df):
    """Normalize user dataset column names into canonical names."""
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


@st.cache_data
def _calculate_derived_metrics(df, fps=None, normalize_time=False):
    """Vectorized calculation of Speed, Angle (Direction), and Spatial Density.

    Default behavior matches historical exports: speed uses frame counts. If
    normalize_time is True and fps is provided, speed is computed in m/s using
    elapsed seconds instead. This keeps legacy work intact while enabling a
    time-normalized mode for videos captured with a fixed camera frame rate.
    """
    df = df.sort_values(by=["track_id", "frame"]).reset_index(drop=True)

    # Spatial Deltas
    df["dx"] = df.groupby("track_id")["x"].diff().fillna(0)
    df["dy"] = df.groupby("track_id")["y"].diff().fillna(0)
    df["frame_dt"] = df.groupby("track_id")["frame"].diff().fillna(0)

    # Time basis: preserve legacy behavior unless explicit seconds-normalization is enabled.
    if normalize_time:
        if fps is None or fps <= 0:
            fps = 30.0
        df["dt"] = np.maximum(df["frame_dt"] / fps, 1.0 / fps)
    else:
        df["dt"] = np.maximum(df["frame_dt"], 1)

    # Speed & Direction Angle
    distance = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["speed"] = distance / np.maximum(df["dt"], 1e-9)
    df["direction"] = np.degrees(np.arctan2(df["dy"], df["dx"])) % 360

    # Local Crowd Density (count of pedestrians in a frame / cell area)
    # Legacy volume remains a count, while density is area-normalized when a cell area is known.
    df["density"] = df.groupby("frame")["track_id"].transform("count")
    df["volume"] = 1.0  # Base unit for volume aggregation
    df["time_basis"] = "seconds" if normalize_time else "frames"
    df["fps_used"] = fps if fps is not None else np.nan

    return df


# =============================================================================
# PLOTLY FIGURE BUILDERS
# =============================================================================

def _build_plotly_animation(df, wall_lines, metric_choice, trail_length):
    """Builds a client-side animated Plotly figure to eliminate Python rerun delays."""
    frames = []
    unique_frames = sorted(df["frame"].unique())
    
    # Downsample frame rendering if video sequence is long (> 150 frames) for butter-smooth animation
    frame_step = 1 if len(unique_frames) <= 150 else int(len(unique_frames) / 100)
    selected_frames = unique_frames[::frame_step]

    metric_col_map = {"Volume": "volume", "Density": "density", "Speed": "speed", "Direction": "direction"}
    z_col = metric_col_map.get(metric_choice, "volume")
    colorscale_map = {"Volume": "Viridis", "Density": "Hot", "Speed": "Plasma", "Direction": "HSV"}

    # Base Layout
    fig = go.Figure()
    _draw_cad_walls(fig, wall_lines)

    # Add Animation Frames
    for f in selected_frames:
        frame_df = df[df["frame"] == f]
        start_f = max(df["frame"].min(), f - trail_length)
        trail_df = df[(df["frame"] >= start_f) & (df["frame"] <= f)]

        frame_data = [
            # Trajectory Trails
            go.Scatter(
                x=trail_df["x"], y=trail_df["y"],
                mode="markers", marker=dict(size=3, color="#00ADB5", opacity=0.4),
                hoverinfo="none"
            ),
            # Current Pedestrian Positions
            go.Scatter(
                x=frame_df["x"], y=frame_df["y"],
                mode="markers+text",
                marker=dict(size=8, color="#FF007F", line=dict(color="#FFFFFF", width=1)),
                text=[f"ID {tid}" for tid in frame_df["track_id"]],
                textposition="top center",
                hoverinfo="text",
                hovertext=[f"Pedestrian #{tid}<br>X: {x:.2f}<br>Y: {y:.2f}" for tid, x, y in zip(frame_df["track_id"], frame_df["x"], frame_df["y"])]
            )
        ]

        # Add Frame Heatmap
        if not frame_df.empty:
            frame_data.insert(0, go.Densitymapbox if False else go.Histogram2dContour(
                x=frame_df["x"], y=frame_df["y"], z=frame_df[z_col],
                histfunc="avg", colorscale=colorscale_map.get(metric_choice, "Viridis"),
                opacity=0.6, showscale=False, contours=dict(coloring="heatmap")
            ))

        frames.append(go.Frame(data=frame_data, name=str(f)))

    # Apply First Frame Data to Figure
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)

    fig.frames = frames

    # Animation Controls Layout
    fig.update_layout(
        template="plotly_dark",
        height=600,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ],
            "x": 0.1, "y": 1.15
        }],
        sliders=[{
            "steps": [{"args": [[str(f)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(f), "method": "animate"} for f in selected_frames],
            "x": 0.1, "y": -0.05, "len": 0.9
        }]
    )
    _configure_axes(fig, df, wall_lines)
    return fig


def _build_single_frame_figure(df, wall_lines, frame_idx, metric_choice, trail_length):
    """Generates static layout for inspectable single frame."""
    fig = go.Figure()
    _draw_cad_walls(fig, wall_lines)

    start_frame = max(df["frame"].min(), frame_idx - trail_length)
    trail_df = df[(df["frame"] >= start_frame) & (df["frame"] <= frame_idx)]
    frame_df = df[df["frame"] == frame_idx]

    _add_heatmap_layer(fig, trail_df, metric_choice)

    # Trails
    for track_id, group in trail_df.groupby("track_id"):
        fig.add_trace(go.Scatter(x=group["x"], y=group["y"], mode="lines", line=dict(width=1.5, color="#00ADB5"), opacity=0.5, showlegend=False))

    # Active Pedestrians
    if not frame_df.empty:
        fig.add_trace(go.Scatter(
            x=frame_df["x"], y=frame_df["y"], mode="markers+text",
            marker=dict(size=10, color="#FF007F", line=dict(color="#FFFFFF", width=1)),
            text=[f"ID {tid}" for tid in frame_df["track_id"]], textposition="top center",
            name="Active Crowd"
        ))

    _configure_axes(fig, df, wall_lines)
    return fig


def _build_aggregated_figure(df, wall_lines, metric_choice, grid_bins, show_vectors, arrow_scale):
    """Builds the overall session summary heatmap and vector field."""
    fig = go.Figure()
    _draw_cad_walls(fig, wall_lines)
    _add_heatmap_layer(fig, df, metric_choice, is_summary=True)

    if show_vectors:
        _add_vector_field(fig, df, grid_bins=grid_bins, scale_multiplier=arrow_scale)

    _configure_axes(fig, df, wall_lines)
    return fig


# =============================================================================
# CAD & DRAWING UTILITIES
# =============================================================================

def _draw_cad_walls(fig, wall_lines):
    """Renders CAD wall lines on the Plotly viewport."""
    if not wall_lines:
        return

    wall_x, wall_y = [], []
    for line in wall_lines:
        if hasattr(line, "xy"):
            x, y = line.xy
            wall_x.extend([x[0], x[1], None])
            wall_y.extend([y[0], y[1], None])
        elif isinstance(line, (list, tuple)):
            for i in range(len(line) - 1):
                p1, p2 = line[i], line[i+1]
                x1 = p1.get("x", p1[0]) if isinstance(p1, (dict, tuple, list)) else p1[0]
                y1 = p1.get("y", p1[1]) if isinstance(p1, (dict, tuple, list)) else p1[1]
                x2 = p2.get("x", p2[0]) if isinstance(p2, (dict, tuple, list)) else p2[0]
                y2 = p2.get("y", p2[1]) if isinstance(p2, (dict, tuple, list)) else p2[1]
                wall_x.extend([x1, x2, None])
                wall_y.extend([y1, y2, None])

    if wall_x:
        fig.add_trace(go.Scatter(
            x=wall_x, y=wall_y, mode="lines",
            line=dict(color="#00E5FF", width=2.0),
            name="CAD Walls", hoverinfo="none"
        ))


def _add_heatmap_layer(fig, target_df, metric_choice, is_summary=False):
    if target_df.empty:
        return

    col_map = {"Volume": "volume", "Density": "density", "Speed": "speed", "Direction": "direction"}
    colorscale_map = {"Volume": "Viridis", "Density": "Hot", "Speed": "Plasma", "Direction": "HSV"}

    z_col = col_map.get(metric_choice, "volume")
    colorscale = colorscale_map.get(metric_choice, "Viridis")

    fig.add_trace(go.Histogram2dContour(
        x=target_df["x"], y=target_df["y"], z=target_df[z_col],
        histfunc="avg", colorscale=colorscale, opacity=0.65,
        showscale=True, name=f"{metric_choice} {'(Summary)' if is_summary else ''}",
        contours=dict(coloring="heatmap")
    ))


def _add_vector_field(fig, df, grid_bins=20, scale_multiplier=1.5):
    """Calculates averaged directional field vectors on grid cells."""
    moving_df = df[(df["dx"] != 0) | (df["dy"] != 0)].copy()
    if moving_df.empty:
        return

    x_bins = np.linspace(df["x"].min(), df["x"].max(), grid_bins)
    y_bins = np.linspace(df["y"].min(), df["y"].max(), grid_bins)

    moving_df["x_bin"] = pd.cut(moving_df["x"], bins=x_bins, labels=False)
    moving_df["y_bin"] = pd.cut(moving_df["y"], bins=y_bins, labels=False)

    binned = moving_df.groupby(["x_bin", "y_bin"], observed=True).agg({"dx": "mean", "dy": "mean"}).reset_index()

    arrow_x, arrow_y = [], []
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    base_len = ((df["x"].max() - df["x"].min()) / grid_bins) * 0.5 * scale_multiplier

    for _, row in binned.iterrows():
        bx, by = int(row["x_bin"]), int(row["y_bin"])
        if bx < len(x_centers) and by < len(y_centers):
            cx, cy = x_centers[bx], y_centers[by]
            dx, dy = row["dx"], row["dy"]
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 0:
                end_x = cx + (dx / mag) * base_len
                end_y = cy + (dy / mag) * base_len
                arrow_x.extend([cx, end_x, None])
                arrow_y.extend([cy, end_y, None])

    if arrow_x:
        fig.add_trace(go.Scatter(x=arrow_x, y=arrow_y, mode="lines", line=dict(color="#00FFFF", width=1.5), name="Flow Vector"))


def _configure_axes(fig, df, wall_lines):
    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="X Position", scaleanchor="y", scaleratio=1, showgrid=True),
        yaxis=dict(title="Y Position", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )


# =============================================================================
# PARSERS & EXPORT HANDLERS
# =============================================================================

def _parse_tracking_json(uploaded_json):
    content = json.load(uploaded_json)
    extracted_df, extracted_walls = None, []

    if isinstance(content, dict):
        for key in ["tracks", "data", "records", "results", "trajectories"]:
            if key in content and isinstance(content[key], list):
                extracted_df = pd.DataFrame(content[key])
                break
        for wall_key in ["walls", "wall_lines", "layout", "cad", "floorplan"]:
            if wall_key in content and isinstance(content[wall_key], list):
                extracted_walls = content[wall_key]
                break

    elif isinstance(content, list):
        extracted_df = pd.DataFrame(content)

    return extracted_df, extracted_walls


def _parse_dxf_file(uploaded_dxf):
    if ezdxf is None:
        st.warning("⚠️ `ezdxf` library missing. Install via `pip install ezdxf`.")
        return []
    try:
        bytes_data = uploaded_dxf.read()
        try:
            text_str = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            text_str = bytes_data.decode("cp1252", errors="ignore")

        doc = ezdxf.read(io.StringIO(text_str))
        msp = doc.modelspace()
        lines = []
        for entity in msp.query("LINE LWPOLYLINE POLYLINE"):
            if entity.dxftype() == "LINE":
                lines.append([(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)])
            elif entity.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                lines.append([(p[0], p[1]) for p in entity.get_points()])
        return lines
    except Exception as e:
        st.error(f"Error parsing DXF: {e}")
        return []


def _export_to_json(df, wall_lines):
    """Packages trajectory data, computed metrics, and CAD walls into JSON format."""
    export_dict = {
        "metadata": {
            "time_basis": df["time_basis"].dropna().iloc[0] if "time_basis" in df.columns and not df.empty else "frames",
            "fps_used": float(df["fps_used"].dropna().iloc[0]) if "fps_used" in df.columns and not df.empty and not pd.isna(df["fps_used"].dropna().iloc[0]) else None,
        },
        "walls": wall_lines,
        "trajectories": df[["frame", "track_id", "x", "y", "speed", "direction", "density", "dt", "time_basis"]].to_dict(orient="records")
    }
    return json.dumps(export_dict, indent=2).encode("utf-8")


def _generate_mock_tracking_data():
    np.random.seed(42)
    records = []
    for tid in range(1, 10):
        x, y = np.random.uniform(10, 90), np.random.uniform(10, 90)
        vx, vy = np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)
        for frame in range(0, 60, 2):
            x += vx + np.random.normal(0, 0.2)
            y += vy + np.random.normal(0, 0.2)
            records.append({"frame": frame, "track_id": tid, "x": float(x), "y": float(y)})
    return pd.DataFrame(records)


def _generate_mock_walls():
    return [[(0, 0), (100, 0)], [(100, 0), (100, 100)], [(100, 100), (0, 100)], [(0, 100), (0, 0)]]