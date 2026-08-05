# views/playback_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def render_playback_view(wall_lines, tracking_df=None):
    """
    Renders 2D Human Movement Playback & Heatmap Visualizations.
    Expects tracking_df with columns: ['frame', 'track_id', 'x', 'y', 'speed', 'angle']
    """
    st.subheader("📊 2D Movement Playback & Crowd Metric Heatmaps")

    # Guard Clause: Check for tracking data in session state or direct parameter
    df = tracking_df if tracking_df is not None else st.session_state.get("tracking_results_df", None)

    if df is None or df.empty:
        st.info("ℹ️ No tracking data found. Please run video tracking in Tab 2.3 first to view 2D playback & analytics.")
        
        # Mock Data Generator for UI Testing / Preview
        if st.checkbox("🧪 Load Mock Playback Data for Testing", value=False):
            df = _generate_mock_tracking_data()
            st.session_state.tracking_results_df = df
            st.rerun()
        else:
            return

    # --- CONTROLS BAR ---
    col_metric, col_frame, col_trail = st.columns([1.5, 2, 1])

    with col_metric:
        metric_choice = st.selectbox(
            "Select Crowd Metric Heatmap:",
            options=["None (Trajectory Only)", "Crowd Volume", "Density", "Speed", "Direction"],
            index=1
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
            step=5
        )

    # --- PREPARE FIGURE ---
    fig = go.Figure()

    # 1. Unpack & Draw CAD Walls
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

    # 2. Add Heatmap Layer based on Selected Metric
    if metric_choice != "None (Trajectory Only)":
        # Cumulative data up to current frame for overall map or single frame snapshot
        heatmap_df = df[df["frame"] <= current_frame]
        
        if not heatmap_df.empty:
            colorscale = "Jet"
            z_vals = None
            hover_template = None

            if metric_choice == "Crowd Volume":
                z_vals = np.ones(len(heatmap_df))
                colorscale = "Viridis"
                hover_template = "Volume Density"
            elif metric_choice == "Density":
                z_vals = np.ones(len(heatmap_df))
                colorscale = "Hot"
                hover_template = "Occupancy Density"
            elif metric_choice == "Speed":
                z_vals = heatmap_df["speed"] if "speed" in heatmap_df.columns else np.random.uniform(0.5, 2.0, len(heatmap_df))
                colorscale = "Plasma"
                hover_template = "Speed: %{z:.2f} m/s"
            elif metric_choice == "Direction":
                z_vals = heatmap_df["angle"] if "angle" in heatmap_df.columns else np.random.uniform(0, 360, len(heatmap_df))
                colorscale = "HSV"
                hover_template = "Heading Angle: %{z:.1f}°"

            fig.add_trace(go.Densitymapbox if False else go.Histogram2dContour(
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

    # 3. Add 2D Human Trajectories & Current Frame Markers
    start_frame = max(min_frame, current_frame - trail_length)
    frame_window_df = df[(df["frame"] >= start_frame) & (df["frame"] <= current_frame)]

    # Draw Motion Trails
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

    # Draw Current Agents (At Current Frame)
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

    # Layout Styling
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
    """Generates mock tracking trajectories for visualization demo."""
    np.random.seed(42)
    records = []
    num_tracks = 8
    num_frames = 100

    for tid in range(1, num_tracks + 1):
        start_x = np.random.uniform(1.0, 8.0)
        start_y = np.random.uniform(1.0, 8.0)
        vx = np.random.uniform(-0.1, 0.1)
        vy = np.random.uniform(-0.1, 0.1)

        x, y = start_x, start_y
        for frame in range(0, num_frames, 2):
            x += vx + np.random.normal(0, 0.02)
            y += vy + np.random.normal(0, 0.02)
            speed = float(np.sqrt(vx**2 + vy**2) * 10)
            angle = float(np.degrees(np.arctan2(vy, vx)) % 360)
            records.append({
                "frame": frame,
                "track_id": tid,
                "x": float(x),
                "y": float(y),
                "speed": speed,
                "angle": angle
            })

    return pd.DataFrame(records)