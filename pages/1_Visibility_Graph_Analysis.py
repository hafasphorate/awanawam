import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import tempfile
import time
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from utils.vga_engine import (
    extract_dxf_walls, 
    generate_isovist_polygon, 
    compute_isovist_metrics,
    compute_graph_vga_metrics
)

st.set_page_config(page_title="Visibility Graph Analysis", layout="wide")

st.title("1. Visibility Graph Analysis (VGA)")
st.markdown("Upload a floorplan DXF file, select public spaces interactively, set grid resolution, and analyze spatial visibility metrics.")

# Sidebar Settings
st.sidebar.header("Analysis Settings")
grid_size = st.sidebar.number_input("Grid Dimension (mm)", min_value=200, max_value=5000, value=1000, step=100)
ray_step = st.sidebar.slider("Ray Angle Step (Degrees)", min_value=1.0, max_value=15.0, value=2.0, step=0.5)
ray_count = int(360 / ray_step)

uploaded_file = st.file_uploader("Upload DXF Floorplan", type=["dxf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Parsing DXF wall geometry..."):
        wall_lines = extract_dxf_walls(tmp_path)
        strtree = STRtree(wall_lines)

    all_bounds = [w.bounds for w in wall_lines]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    width, height = maxx - minx, maxy - miny

    st.subheader("Select Public Analysis Area")
    st.info("Draw a polygon around the public space where grid points should be evaluated.")

    # Render interactive canvas overlay
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.2)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_color="#1E1E1E",
        height=500,
        width=700,
        drawing_mode="polygon",
        key="canvas",
    )

    # Extract polygon from canvas coordinates back to floorplan mm coordinates
    selected_polygons = []
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "path":
                pts = []
                for p in obj["path"]:
                    if len(p) >= 3 and (p[0] == 'M' or p[0] == 'L'):
                        # Scale canvas (700x500) coords to real DXF dimensions
                        scaled_x = minx + (p[1] / 700.0) * width
                        scaled_y = miny + ((500.0 - p[2]) / 500.0) * height
                        pts.append((scaled_x, scaled_y))
                if len(pts) >= 3:
                    selected_polygons.append(Polygon(pts))

    if st.button("Run Visibility Analysis"):
        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)
        
        # Filter points within selected public polygons (or use bounding box if no selection made)
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                pt = Point(x, y)
                if selected_polygons:
                    if any(poly.contains(pt) for poly in selected_polygons):
                        grid_points.append((x, y))
                else:
                    grid_points.append((x, y))

        total_points = len(grid_points)

        if total_points == 0:
            st.warning("No grid points inside the selected area. Adjust your boundary polygon or grid size.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            vga_results = []
            isovist_polys = []
            start_time = time.time()

            for idx, pt in enumerate(grid_points):
                isovist, occluded_count = generate_isovist_polygon(pt, wall_lines, strtree, num_rays=ray_count)
                if isovist:
                    metrics = compute_isovist_metrics(isovist, pt, occluded_count, ray_count)
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
                        f"⌛ **Analyzing grid point {completed} of {total_points}** ({int(progress_ratio * 100)}%) | "
                        f"**Est. time remaining:** `{time_str}`"
                    )

            # Compute Graph-level VGA Metrics (Integration, Entropy, Mean Depth, Connectivity)
            status_text.markdown("⌛ **Computing graph topology metrics (Integration & Entropy)...**")
            final_vga_results = compute_graph_vga_metrics(vga_results, isovist_polys)

            progress_bar.empty()
            total_time = round(time.time() - start_time, 2)
            status_text.success(f"✅ Visibility Analysis complete in **{total_time} seconds** across **{total_points}** public points!")

            df_results = pd.DataFrame(final_vga_results)
            st.session_state["vga_df"] = df_results

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

        st.subheader("VGA Heatmap Visualizer")
        all_metrics = [
            "visual_integration", "visual_entropy", "visual_mean_depth", "connectivity",
            "isovist_area", "isovist_compactness", "isovist_drift_magnitude", 
            "isovist_min_radial", "isovist_max_radial", "isovist_occlusivity", 
            "isovist_perimeter", "isovist_drift_minima"
        ]
        selected_metric = st.selectbox("Select Metric to Render:", all_metrics)

        fig = px.scatter(
            df, x="x", y="y", color=selected_metric,
            color_continuous_scale="Viridis",
            title=f"Spatial Map: {selected_metric}"
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

        json_data = df.to_json(orient="records")
        st.download_button(
            label="📥 Download Complete VGA Metrics JSON",
            data=json_data,
            file_name="vga_analysis_results.json",
            mime="application/json"
        )