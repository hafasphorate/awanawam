# views/tracking_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.tracking_engine import TrackingProcessor

def render_tracking_view(dxf_wall_polygons, vga_grid_df=None):
    st.subheader("Step 2.2: Occupancy Density & Path Tracking")

    # Guard clause: check for calibrated homography matrix
    if "homography_matrix" not in st.session_state or st.session_state.homography_matrix is None:
        st.error("⚠️ Camera Homography Matrix not found! Please complete Step 2.1 Calibration first.")
        return

    col_up, col_opts = st.columns([2, 1])

    with col_up:
        uploaded_csv = st.file_uploader(
            "Upload Tracking Bounding Box CSV", 
            type=["csv"],
            help="CSV must contain: frame_id, track_id, x1, y1, x2, y2"
        )

    with col_opts:
        st.markdown("##### Configuration")
        tracking_point = st.radio(
            "Detection Point",
            options=["Head (Top-Center)", "Feet (Bottom-Center)"],
            index=0,
            help="Default is Head (Top-Center) for crowded environments where feet are occluded."
        )
        show_vga_underlay = st.checkbox("Overlay VGA Integration Heatmap", value=(vga_grid_df is not None))
        show_paths = st.checkbox("Show Head Movement Trajectories", value=True)
        bins_res = st.slider("Heatmap Density Resolution", 30, 150, 80)

    # Load uploaded CSV or fallback to synthetic demo data
    if not uploaded_csv:
        st.info("💡 Upload a CSV or use synthetic crowd tracking data to preview the view.")
        if st.button("🎲 Run with Synthetic Crowd Data"):
            df_raw = generate_synthetic_crowd_data()
        else:
            return
    else:
        df_raw = pd.read_csv(uploaded_csv)

    # Process and transform points
    processor = TrackingProcessor(st.session_state.homography_matrix)
    try:
        df_transformed = processor.parse_and_transform_csv(df_raw, tracking_point=tracking_point)
    except Exception as e:
        st.error(f"Error processing tracking file: {e}")
        return

    # Metrics Summary
    k1, k2, k3 = st.columns(3)
    k1.metric("Tracked Frames", len(df_transformed['frame_id'].unique()))
    k1.caption(f"Using: **{tracking_point}**")
    k2.metric("Unique Person IDs", len(df_transformed['track_id'].unique()))
    k3.metric("Total Detection Points", len(df_transformed))

    # Compute DXF Bounding Box
    if dxf_wall_polygons:
        all_x = [pt[0] for wall in dxf_wall_polygons for pt in wall.exterior.coords]
        all_y = [pt[1] for wall in dxf_wall_polygons for pt in wall.exterior.coords]
        bounds = (min(all_x), max(all_x), min(all_y), max(all_y))
    else:
        bounds = (df_transformed['CAD_X'].min(), df_transformed['CAD_X'].max(), 
                  df_transformed['CAD_Y'].min(), df_transformed['CAD_Y'].max())

    # Compute Density Matrix
    x_centers, y_centers, density_z = processor.compute_occupancy_density(
        df_transformed['CAD_X'].values,
        df_transformed['CAD_Y'].values,
        bounds,
        nbins=bins_res
    )

    # --- PLOTLY DUAL-LAYER MAP ---
    fig = go.Figure()

    # Layer 1: VGA Underlay (Optional)
    if show_vga_underlay and vga_grid_df is not None:
        fig.add_trace(go.Scattergl(
            x=vga_grid_df['x'], y=vga_grid_df['y'],
            mode='markers',
            marker=dict(
                color=vga_grid_df['integration'],
                colorscale='Cividis',
                size=5, opacity=0.35,
                colorbar=dict(title="VGA Integration", x=1.02)
            ),
            name="VGA Integration"
        ))

    # Layer 2: CAD Walls
    for wall in dxf_wall_polygons:
        wx, wy = wall.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(wx), y=list(wy),
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False, hoverinfo="none"
        ))

    # Layer 3: Head Occupancy Density Heatmap
    fig.add_trace(go.Contour(
        x=x_centers, y=y_centers, z=density_z,
        colorscale="YlOrRd", opacity=0.65,
        contours_coloring="heatmap", line_width=0,
        colorbar=dict(title="Crowd Head Density", x=1.14),
        name="Occupancy Density"
    ))

    # Layer 4: Person Movement Trajectories
    if show_paths:
        for track_id, track_df in df_transformed.groupby('track_id'):
            fig.add_trace(go.Scatter(
                x=track_df['CAD_X'], y=track_df['CAD_Y'],
                mode='lines+markers', marker=dict(size=3), line=dict(width=1),
                name=f"Person {track_id}",
                hovertemplate=f"<b>ID:</b> {track_id}<br><b>Frame:</b> %{{text}}<br><b>CAD X:</b> %{{x:.2f}}<br><b>CAD Y:</b> %{{y:.2f}}",
                text=track_df['frame_id']
            ))

    fig.update_layout(
        title="<b>Head-Height Crowd Occupancy Density vs. CAD Layout</b>",
        xaxis=dict(title="CAD X (meters)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="CAD Y (meters)"),
        height=650, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Download Button
    st.download_button(
        label="📥 Download Projected CAD Coordinates (CSV)",
        data=df_transformed.to_csv(index=False),
        file_name="projected_crowd_head_coordinates.csv",
        mime="text/csv",
        use_container_width=True
    )


def generate_synthetic_crowd_data() -> pd.DataFrame:
    """Generates synthetic bounding box tracking logs for demonstration."""
    np.random.seed(42)
    records = []
    
    for person_id in range(1, 7):
        start_u = np.random.uniform(250, 550)
        start_v = np.random.uniform(150, 400)
        
        for frame in range(1, 80, 2):
            start_u += np.random.normal(2, 0.8)
            start_v += np.random.normal(1.5, 0.8)
            
            # Bounding box [x1, y1, x2, y2]
            # y1 represents the top edge (head)
            records.append({
                'frame_id': frame,
                'track_id': person_id,
                'x1': start_u - 15,
                'y1': start_v,
                'x2': start_u + 15,
                'y2': start_v + 75
            })
            
    return pd.DataFrame(records)