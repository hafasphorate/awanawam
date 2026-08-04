import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import tempfile
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
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
st.markdown("Upload a DXF floorplan, select analysis zones on the plan preview, and compute spatial metrics.")

# Sidebar Settings
st.sidebar.header("Analysis Settings")
grid_size = st.sidebar.number_input("Grid Dimension (mm)", min_value=200, max_value=5000, value=1000, step=100)
ray_step = st.sidebar.slider("Ray Angle Step (Degrees)", min_value=1.0, max_value=15.0, value=2.0, step=0.5)
ray_count = int(360 / ray_step)

uploaded_file = st.file_uploader("Upload DXF Floorplan", type=["dxf"])

def create_floorplan_image(wall_lines, width_px=700, height_px=500):
    """Renders DXF wall segments to a PIL image to use as a canvas background."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')

    for line in wall_lines:
        x, y = line.xy
        ax.plot(x, y, color='#FFFFFF', linewidth=1)

    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Parsing DXF wall geometry..."):
        wall_lines = extract_dxf_walls(tmp_path)
        strtree = STRtree(wall_lines)

    st.success(f"Extracted {len(wall_lines)} wall boundary segments.")

    all_bounds = [w.bounds for w in wall_lines]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    width, height = maxx - minx, maxy - miny

    st.subheader("Interactive Public Space Selection")
    st.info("Click points on the floorplan to define public analysis zones (or leave blank to evaluate the full floorplan).")

    # Render DXF floorplan background image
    bg_image = create_floorplan_image(wall_lines)

    selection_mode = st.radio("Selection Mode:", ["Full Floorplan", "Click Points / Polygon Boundary"], horizontal=True)

    selected_polygons = []

    if selection_mode == "Click Points / Polygon Boundary":
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=bg_image,
            height=500,
            width=700,
            drawing_mode="polygon",
            key="canvas_floorplan",
        )

        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "path":
                    pts = []
                    for p in obj["path"]:
                        if len(p) >= 3 and (p[0] == 'M' or p[0] == 'L'):
                            scaled_x = minx + (p[1] / 700.0) * width
                            scaled_y = miny + ((500.0 - p[2]) / 500.0) * height
                            pts.append((scaled_x, scaled_y))
                    if len(pts) >= 3:
                        selected_polygons.append(Polygon(pts))

    if st.button("Run Visibility Analysis"):
        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)

        grid_points = []
        for x in x_coords:
            for y in y_coords:
                pt = Point(x, y)
                if selected_polygons and selection_mode != "Full Floorplan":
                    if any(poly.contains(pt) for poly in selected_polygons):
                        grid_points.append((x, y))
                else:
                    grid_points.append((x, y))

        total_points = len(grid_points)

        if total_points == 0:
            st.warning("No grid points generated. Check your selected zone or adjust grid size.")
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

            if vga_results:
                status_text.markdown("⌛ **Computing graph topology metrics (Integration & Entropy)...**")
                final_vga_results = compute_graph_vga_metrics(vga_results, isovist_polys)

                progress_bar.empty()
                total_time = round(time.time() - start_time, 2)
                status_text.success(f"✅ Analysis complete in **{total_time}s** across **{total_points}** points!")

                df_results = pd.DataFrame(final_vga_results)
                st.session_state["vga_df"] = df_results
            else:
                progress_bar.empty()
                st.error("Could not extract valid isovists. Points may be located inside solid wall geometry.")

    if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
        df = st.session_state["vga_df"]

        st.subheader("VGA Heatmap Visualizer")

        # Dynamically list only metrics that exist in the calculated DataFrame
        available_metrics = [c for c in df.columns if c not in ["x", "y"]]
        
        if available_metrics:
            selected_metric = st.selectbox("Select Metric to Render:", available_metrics)

            if selected_metric in df.columns:
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