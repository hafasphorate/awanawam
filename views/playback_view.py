# views/playback_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

def render_playback_view(wall_lines, tracking_df=None):
    st.subheader("📊 2D Movement Playback & Crowd Metric Heatmaps")

    # 1. Check Session State
    df = tracking_df if tracking_df is not None else st.session_state.get("tracking_results_df", None)

    # 2. If data is missing, render an easy upload interface right inside Tab 4
    if df is None or df.empty:
        st.warning("⚠️ No active tracking session found in memory.")
        st.info("💡 **Quick Fix:** Upload your exported tracking CSV file below, or use mock data to test the playback interface instantly.")

        col_up1, col_up2 = st.columns(2)
        
        with col_up1:
            uploaded_csv = st.file_uploader(
                "📂 Upload Tracking CSV (Exported from Tab 2.3)",
                type=["csv"],
                key="tab4_csv_upload"
            )
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.session_state.tracking_results_df = df
                    st.success(f"✅ Successfully loaded {len(df)} tracking records!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing CSV file: {e}")

        with col_up2:
            if st.button("🧪 Load Mock Playback Data for Testing", use_container_width=True):
                df = _generate_mock_tracking_data()
                st.session_state.tracking_results_df = df
                st.rerun()

        return

    # Normalize column names just in case
    df.columns = [c.lower().strip() for c in df.columns]

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