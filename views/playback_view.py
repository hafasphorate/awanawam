# views/playback_view.py
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


def render_playback_view(wall_lines=None, tracking_df=None):
    st.subheader("📊 2D Movement Playback & Crowd Metric Heatmaps")

    if wall_lines is None:
        wall_lines = st.session_state.get("dxf_walls", [])

    # 1. Check Session State for tracking data
    df = tracking_df if tracking_df is not None else st.session_state.get("tracking_results_df", None)

    # 2. Upload Interface (CSV/JSON and optional DXF)
    if df is None or df.empty or (not wall_lines and "dxf_walls" not in st.session_state):
        st.warning("⚠️ Active tracking data or room layout is missing.")
        st.info("💡 **Quick Fix:** Upload your tracking data (CSV or JSON) and CAD layout (DXF), or use mock data to test instantly.")

        col_up1, col_up2 = st.columns(2)

        with col_up1:
            # Combined CSV and JSON Uploader
            uploaded_file = st.file_uploader(
                "📂 Upload Tracking File (CSV or JSON)",
                type=["csv", "json"],
                key="tab4_data_upload"
            )
            
            # Optional DXF Uploader
            uploaded_dxf = st.file_uploader(
                "📐 Upload Floorplan Layout (DXF - Optional)",
                type=["dxf"],
                key="tab4_dxf_upload"
            )

            # Process DXF file
            if uploaded_dxf is not None:
                parsed_walls = _parse_dxf_file(uploaded_dxf)
                if parsed_walls:
                    st.session_state.dxf_walls = parsed_walls
                    wall_lines = parsed_walls
                    st.success("✅ Loaded DXF CAD layout!")

            # Process CSV/JSON file
            if uploaded_file is not None:
                try:
                    df = _parse_tracking_file(uploaded_file)
                    st.session_state.tracking_results_df = df
                    st.success(f"✅ Successfully loaded {len(df)} tracking records!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing tracking file: {e}")

        with col_up2:
            if st.button("🧪 Load Mock Data for Testing", use_container_width=True):
                df = _generate_mock_tracking_data()
                st.session_state.tracking_results_df = df
                if not wall_lines:
                    st.session_state.dxf_walls = _generate_mock_walls()
                st.rerun()

        if df is None or df.empty:
            return

    # Normalize column names
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Map frame variations (including frame_idx) to 'frame'
    frame_aliases = ["frame_idx", "frame_id", "frames", "frame_num", "step", "timestamp"]
    if "frame" not in df.columns:
        for alias in frame_aliases:
            if alias in df.columns:
                df.rename(columns={alias: "frame"}, inplace=True)
                break

    # Map ID variations to 'track_id'
    id_aliases = ["id", "pedestrian_id", "agent_id", "track_idx"]
    if "track_id" not in df.columns:
        for alias in id_aliases:
            if alias in df.columns:
                df.rename(columns={alias: "track_id"}, inplace=True)
                break

    # Validate essential columns
    required_cols = {"frame", "x", "y", "track_id"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        st.error(f"❌ Loaded file is missing required columns: `{', '.join(missing_cols)}`")
        st.write("Found columns:", list(df.columns))

        if st.button("🔄 Reset & Try Again"):
            st.session_state.tracking_results_df = None
            st.rerun()
        return

    # --- CONTROLS BAR ---
    col_metric, col_frame, col_trail = st.columns([1.5, 2, 1])

    with col_metric:
        metric_choice = st.selectbox(
            "Select Crowd Metric Heatmap:",
            options=["None (Trajectory Only)", "Crowd Volume", "Density", "Speed", "Direction"],
            index=1,
            key="playback_metric_selectbox"
        )

    max_frame = int(df["frame"].max())
    min_frame = int(df["frame"].min())

    with col_frame:
        current_frame = st.slider(
            "Frame Playback Slider",
            min_value=min_frame,
            max_value=max_frame,
            value=min_frame,
            step=1,
            key="playback_frame_slider"
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

    # --- PREPARE PLOTLY FIGURE ---
    fig = go.Figure()

    # Draw CAD Walls
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
            line=dict(color="#00ADB5", width=1.5),
            name="CAD Walls",
            hoverinfo="none",
            showlegend=False
        ))

    # Add Heatmap Layer based on Selected Metric
    if metric_choice != "None (Trajectory Only)":
        heatmap_df = df[df["frame"] <= current_frame]

        if not heatmap_df.empty:
            colorscale = "Jet"
            z_vals = None

            if metric_choice == "Crowd Volume":
                z_vals = np.ones(len(heatmap_df))
                colorscale = "Viridis"
            elif metric_choice == "Density":
                z_vals = np.ones(len(heatmap_df))
                colorscale = "Hot"
            elif metric_choice == "Speed":
                z_vals = heatmap_df["speed"] if "speed" in heatmap_df.columns else np.ones(len(heatmap_df))
                colorscale = "Plasma"
            elif metric_choice == "Direction":
                z_vals = heatmap_df["angle"] if "angle" in heatmap_df.columns else np.zeros(len(heatmap_df))
                colorscale = "HSV"

            fig.add_trace(go.Histogram2dContour(
                x=heatmap_df["x"],
                y=heatmap_df["y"],
                z=z_vals,
                histfunc="sum" if metric_choice == "Crowd Volume" else "avg",
                colorscale=colorscale,
                opacity=0.6,
                showscale=True,
                name=metric_choice,
                contours=dict(coloring="heatmap")
            ))

    # Add 2D Human Trajectory Trails
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

    # Add Active Pedestrians at Current Frame
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
            hovertext=[f"Pedestrian #{tid}<br>X: {x:.2f}m<br>Y: {y:.2f}m" 
                       for tid, x, y in zip(active_agents["track_id"], active_agents["x"], active_agents["y"])]
        ))

    fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis=dict(title="X Position (m)", scaleanchor="y", scaleratio=1, showgrid=True),
        yaxis=dict(title="Y Position (m)", showgrid=True),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def _parse_tracking_file(uploaded_file):
    """Parses uploaded CSV or JSON file into a pandas DataFrame."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".json"):
        content = json.load(uploaded_file)
        if isinstance(content, list):
            return pd.DataFrame(content)
        elif isinstance(content, dict):
            # Check for nested data structures like {"tracks": [...]}
            for key in ["tracks", "data", "records", "results"]:
                if key in content and isinstance(content[key], list):
                    return pd.DataFrame(content[key])
            return pd.DataFrame(content)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or JSON.")


def _parse_dxf_file(uploaded_dxf):
    """Parses DXF files into line coordinates using ezdxf."""
    if ezdxf is None:
        st.warning("⚠️ `ezdxf` library is not installed. Run `pip install ezdxf` to parse DXF files.")
        return []

    try:
        # ezdxf requires stream reading
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
    """Generates mock testing tracks."""
    np.random.seed(42)
    records = []
    for tid in range(1, 9):
        start_x, start_y = np.random.uniform(1.0, 8.0), np.random.uniform(1.0, 8.0)
        vx, vy = np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)
        x, y = start_x, start_y
        for frame in range(0, 100, 2):
            x += vx + np.random.normal(0, 0.02)
            y += vy + np.random.normal(0, 0.02)
            records.append({
                "frame": frame,
                "track_id": tid,
                "x": float(x),
                "y": float(y),
                "speed": float(np.sqrt(vx**2 + vy**2) * 10),
                "angle": float(np.degrees(np.arctan2(vy, vx)) % 360)
            })
    return pd.DataFrame(records)


def _generate_mock_walls():
    """Mock room walls if no DXF uploaded."""
    return [
        [(0, 0), (10, 0)],
        [(10, 0), (10, 10)],
        [(10, 10), (0, 10)],
        [(0, 10), (0, 0)]
    ]