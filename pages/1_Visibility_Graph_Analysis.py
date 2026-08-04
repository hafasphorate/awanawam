import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import time
from shapely.strtree import STRtree

from utils.vga_engine import extract_dxf_walls, generate_isovist_polygon, compute_isovist_metrics

st.set_page_config(page_title="Visibility Graph Analysis", layout="wide")

st.title("1. Visibility Graph Analysis (VGA)")
st.markdown("Upload a floorplan DXF file, define grid resolution, and analyze spatial visibility metrics.")

# Sidebar Configuration
st.sidebar.header("Analysis Settings")
grid_size = st.sidebar.number_input(
    "Grid Dimension (mm)", min_value=200, max_value=5000, value=1000, step=100
)

ray_step = st.sidebar.slider(
    "Ray Angle Step (Degrees)",
    min_value=1.0,
    max_value=15.0,
    value=2.0,
    step=0.5,
    help="Smaller values cast more rays for higher visual accuracy (e.g., 1.0° = 360 rays).",
)

ray_count = int(360 / ray_step)

uploaded_file = st.file_uploader("Upload DXF Floorplan", type=["dxf"])

if uploaded_file is not None:
    # Save DXF temporarily to parse with ezdxf
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Parsing DXF wall geometry..."):
        wall_lines = extract_dxf_walls(tmp_path)
        strtree = STRtree(wall_lines)

    st.success(f"Extracted {len(wall_lines)} wall boundary segments.")

    # Get Floorplan Bounds
    all_bounds = [w.bounds for w in wall_lines]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    st.write(f"**Bounding Box Extents:** Width = {maxx - minx:.1f} mm, Height = {maxy - miny:.1f} mm")

    if st.button("Run Visibility Analysis"):
        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)
        grid_points = [(x, y) for x in x_coords for y in y_coords]
        total_points = len(grid_points)

        if total_points == 0:
            st.warning("No grid points generated. Please check your floorplan boundaries or grid size.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            vga_results = []
            start_time = time.time()

            for idx, pt in enumerate(grid_points):
                isovist = generate_isovist_polygon(pt, wall_lines, strtree, num_rays=ray_count)
                if isovist:
                    metrics = compute_isovist_metrics(isovist, pt)
                    metrics["x"] = pt[0]
                    metrics["y"] = pt[1]
                    vga_results.append(metrics)

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
                        f"⌛ **Analyzing grid point {completed} of {total_points}** ({int(progress_ratio * 100)}%) | "
                        f"**Est. time remaining:** `{time_str}`"
                    )

            progress_bar.empty()
            total_time = round(time.time() - start_time, 2)
            status_text.success(f"✅ Visibility Analysis complete in **{total_time} seconds** across **{total_points}** points!")

            df_results = pd.DataFrame(vga_results)
            st.session_state["vga_df"] = df_results

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

        st.subheader("VGA Heatmap Visualizer")
        selected_metric = st.selectbox(
            "Select Metric to Render:",
            ["isovist_area", "isovist_compactness", "isovist_drift_magnitude", "isovist_perimeter"]
        )

        fig = px.scatter(
            df, x="x", y="y", color=selected_metric,
            color_continuous_scale="Viridis",
            title=f"Spatial Map: {selected_metric}"
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

        json_data = df.to_json(orient="records")
        st.download_button(
            label="📥 Download VGA Metrics as JSON",
            data=json_data,
            file_name="vga_analysis_results.json",
            mime="application/json"
        )