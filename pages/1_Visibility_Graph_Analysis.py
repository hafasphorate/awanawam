import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
import streamlit as st

from utils.vga_engine import (
    compute_graph_vga_metrics,
    compute_isovist_metrics,
    extract_dxf_walls,
    generate_isovist_polygon,
)

st.set_page_config(page_title="Visibility Graph Analysis", layout="wide")

st.title("1. Visibility Graph Analysis (VGA)")
st.markdown(
    "Upload a DXF floorplan, select public analysis zones using the interactive"
    " Box or Lasso tools, and compute spatial visibility metrics."
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

uploaded_file = st.file_uploader("Upload DXF Floorplan", type=["dxf"])


def render_interactive_floorplan(wall_lines):
  """Builds a Plotly figure displaying DXF wall lines with Box and Lasso select enabled."""
  fig = go.Figure()

  for line in wall_lines:
    x, y = line.xy
    fig.add_trace(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            line=dict(color="#00ADB5", width=1.5),
            hoverinfo="none",
            showlegend=False,
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
      dragmode="lasso",  # Default tool active on plot
  )
  return fig


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

  st.subheader("Interactive Public Space Selection")
  st.info(
      "💡 **How to select your space:** Use the **Box Select** or **Lasso"
      " Select** tool from the Plotly toolbar (top right of the plot) to highlight"
      " the public area to analyze."
  )

  selection_mode_option = st.radio(
      "Selection Mode:",
      ["Full Floorplan", "Select Area on Plan (Box / Lasso)"],
      horizontal=True,
  )

  selected_polygon = None

  if selection_mode_option == "Select Area on Plan (Box / Lasso)":
    fig_plan = render_interactive_floorplan(wall_lines)

    # Use valid Streamlit selection modes: "box" and "lasso"
    chart_events = st.plotly_chart(
        fig_plan,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["box", "lasso"],
    )

    # Process selections from user interaction
    if chart_events and "selection" in chart_events:
      selection_data = chart_events["selection"]

      # Handle Box selection
      if "range" in selection_data and selection_data["range"]:
        x_range = selection_data["range"].get("x", [])
        y_range = selection_data["range"].get("y", [])
        if len(x_range) == 2 and len(y_range) == 2:
          bbox_pts = [
              (x_range[0], y_range[0]),
              (x_range[1], y_range[0]),
              (x_range[1], y_range[1]),
              (x_range[0], y_range[1]),
          ]
          selected_polygon = Polygon(bbox_pts)

      # Handle Lasso selection points
      elif "point_indices" in selection_data or "points" in selection_data:
        points = selection_data.get("points", [])
        coords = [(p["x"], p["y"]) for p in points if "x" in p and "y" in p]
        if len(coords) >= 3:
          selected_polygon = Polygon(coords)

    if selected_polygon:
      st.success("✅ Area selection captured successfully!")

  if st.button("Run Visibility Analysis"):
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_points = []
    for x in x_coords:
      for y in y_coords:
        pt = Point(x, y)
        if (
            selected_polygon
            and selection_mode_option != "Full Floorplan"
        ):
          if selected_polygon.contains(pt):
            grid_points.append((x, y))
        else:
          grid_points.append((x, y))

    total_points = len(grid_points)

    if total_points == 0:
      st.warning(
          "No grid points generated. Adjust your Box/Lasso selection or grid"
          " size."
      )
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
              f"⌛ **Analyzing grid point {completed} of {total_points}**"
              f" ({int(progress_ratio * 100)}%) | **Est. time remaining:**"
              f" `{time_str}`"
          )

      if vga_results:
        status_text.markdown(
            "⌛ **Computing graph topology metrics (Integration & Entropy)...**"
        )
        final_vga_results = compute_graph_vga_metrics(
            vga_results, isovist_polys
        )

        progress_bar.empty()
        total_time = round(time.time() - start_time, 2)
        status_text.success(
            f"✅ Analysis complete in **{total_time}s** across"
            f" **{total_points}** points!"
        )

        df_results = pd.DataFrame(final_vga_results)
        st.session_state["vga_df"] = df_results
      else:
        progress_bar.empty()
        st.error(
            "Could not extract valid isovists. Selected points may be inside"
            " solid walls."
        )

  if "vga_df" in st.session_state and not st.session_state["vga_df"].empty:
    df = st.session_state["vga_df"]

    st.subheader("VGA Heatmap Visualizer")
    available_metrics = [c for c in df.columns if c not in ["x", "y"]]

    if available_metrics:
      selected_metric = st.selectbox(
          "Select Metric to Render:", available_metrics
      )

      if selected_metric in df.columns:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=selected_metric,
            color_continuous_scale="Viridis",
            title=f"Spatial Map: {selected_metric}",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

      json_data = df.to_json(orient="records")
      st.download_button(
          label="📥 Download Complete VGA Metrics JSON",
          data=json_data,
          file_name="vga_analysis_results.json",
          mime="application/json",
      )