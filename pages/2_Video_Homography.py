# pages/2_Video_Homography.py
import io
import json
import ezdxf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon
import streamlit as st

from utils.tracking_engine import HomographyCalibrator, extract_frame_from_video
from views.tracking_view import render_tracking_view

st.set_page_config(
    page_title="Module 2: Video Homography & Tracking", layout="wide"
)

st.title("📹 Module 2: Video Homography & Region Selection")

# Initialize Session State
if "dxf_walls" not in st.session_state:
  st.session_state.dxf_walls = []
if "vga_grid_df" not in st.session_state:
  st.session_state.vga_grid_df = None
if "selected_polygon_pts" not in st.session_state:
  st.session_state.selected_polygon_pts = [
      {"X (m)": 0.0, "Y (m)": 0.0},
      {"X (m)": 10.0, "Y (m)": 0.0},
      {"X (m)": 10.0, "Y (m)": 10.0},
      {"X (m)": 0.0, "Y (m)": 10.0},
  ]
if "homography_matrix" not in st.session_state:
  st.session_state.homography_matrix = None
if "selected_frame_idx" not in st.session_state:
  st.session_state.selected_frame_idx = 0

# Navigation Tabs
tab_import, tab_region, tab_tracking = st.tabs([
    "📂 2.1 Import DXF / JSON & Video",
    "📐 2.2 Define ROI Polygon",
    "🔥 2.3 Occupancy Analytics",
])


def parse_dxf_bytes(file_bytes):
  """Parses DXF bytes and extracts wall line geometries."""
  doc = ezdxf.read(io.StringIO(file_bytes.decode("utf-8", errors="ignore")))
  msp = doc.modelspace()
  walls = []

  for e in msp:
    if e.dxftype() == "LINE":
      start, end = e.dxf.start, e.dxf.end
      walls.append(LineString([(start.x, start.y), (end.x, end.y)]))
    elif e.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
      pts = [(p[0], p[1]) for p in e.get_points("xy")]
      if len(pts) >= 2:
        if e.is_closed and len(pts) >= 3:
          walls.append(Polygon(pts))
        else:
          walls.append(LineString(pts))
  return walls


# ==========================================
# TAB 1: FILE & VIDEO IMPORT
# ==========================================
with tab_import:
  st.subheader("Step 2.1: Load DXF/JSON Config & Surveillance Video")

  col_json, col_dxf = st.columns(2)

  # 📄 Option A: JSON Import
  with col_json:
    st.markdown("### 📄 Option A: Import Exported JSON")
    uploaded_json = st.file_uploader(
        "Upload JSON Export (VGA + Polygon Config)",
        type=["json"],
        key="json_uploader",
    )

    if uploaded_json is not None:
      try:
        data = json.load(uploaded_json)

        # 1. Parse VGA Grid Data
        if "vga_grid" in data and data["vga_grid"]:
          st.session_state.vga_grid_df = pd.DataFrame(data["vga_grid"])
          st.success(
              f"✅ Loaded VGA Grid ({len(st.session_state.vga_grid_df)} nodes)"
          )

        # 2. Parse Saved Polygon Points
        if "polygon_points" in data and data["polygon_points"]:
          pts = data["polygon_points"]
          # Convert list of lists [[x, y], ...] to dict list for dataframe editor
          if isinstance(pts[0], list) or isinstance(pts[0], tuple):
            st.session_state.selected_polygon_pts = [
                {"X (m)": p[0], "Y (m)": p[1]} for p in pts
            ]
          else:
            st.session_state.selected_polygon_pts = pts
          st.success(
              f"✅ Loaded {len(st.session_state.selected_polygon_pts)} polygon"
              " vertices"
          )

        # 3. Parse Homography Matrix
        if "homography_matrix" in data and data["homography_matrix"]:
          st.session_state.homography_matrix = np.array(
              data["homography_matrix"]
          )
          st.success("✅ Loaded pre-saved Homography Matrix")

        # 4. Parse Walls if encoded in JSON
        if "dxf_walls" in data and data["dxf_walls"]:
          walls = []
          for w in data["dxf_walls"]:
            if len(w) >= 3:
              walls.append(Polygon(w))
            elif len(w) == 2:
              walls.append(LineString(w))
          st.session_state.dxf_walls = walls
          st.success(
              f"✅ Loaded {len(walls)} wall elements from JSON configuration"
          )

      except Exception as e:
        st.error(f"Error parsing JSON: {e}")

  # 📐 Option B: DXF Import
  with col_dxf:
    st.markdown("### 📐 Option B: Import Raw DXF File")
    uploaded_dxf = st.file_uploader(
        "Upload CAD DXF File", type=["dxf"], key="dxf_uploader"
    )

    if uploaded_dxf is not None:
      try:
        dxf_bytes = uploaded_dxf.read()
        walls = parse_dxf_bytes(dxf_bytes)
        st.session_state.dxf_walls = walls
        st.success(
            f"✅ Successfully parsed DXF! {len(walls)} wall geometries ready"
            " for rendering."
        )
      except Exception as e:
        st.error(f"Failed to parse DXF file: {e}")

  st.markdown("---")

  # 📹 Surveillance Video Upload
  st.markdown("### 📹 Video Stream Target")
  uploaded_video = st.file_uploader(
      "Upload Surveillance Video (.mp4, .avi, .mov)",
      type=["mp4", "avi", "mov"],
      key="video_uploader",
  )

  if uploaded_video:
    st.session_state.uploaded_video_file = uploaded_video
    st.success("✅ Video file attached and ready for tracking calibration.")

    # Frame Picker
    frame_idx = st.slider(
        "Select Calibration Preview Frame",
        min_value=0,
        max_value=1000,
        value=st.session_state.selected_frame_idx,
        step=5,
    )
    st.session_state.selected_frame_idx = frame_idx

    raw_frame_rgb = extract_frame_from_video(
        uploaded_video, frame_number=frame_idx
    )
    if raw_frame_rgb is not None:
      st.image(
          raw_frame_rgb,
          caption=f"Preview Frame (Index #{frame_idx})",
          use_container_width=True,
      )


# ==========================================
# TAB 2: REGION SELECTION & EDITING
# ==========================================
with tab_region:
  st.subheader("Step 2.2: Define Analysis Polygon on Floorplan")

  col_controls, col_plot = st.columns([1, 2])

  with col_controls:
    st.markdown("#### Polygon Vertices (CAD World m)")

    default_df = pd.DataFrame(st.session_state.selected_polygon_pts)

    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        key="poly_editor",
    )

    # Save to session state
    poly_pts_dicts = edited_df.to_dict(orient="records")
    st.session_state.selected_polygon_pts = poly_pts_dicts

    st.markdown("---")

    # JSON Exporter Payload
    poly_list = [[p["X (m)"], p["Y (m)"]] for p in poly_pts_dicts]
    export_payload = {
        "polygon_points": poly_list,
        "vga_grid": (
            st.session_state.vga_grid_df.to_dict(orient="records")
            if st.session_state.vga_grid_df is not None
            else []
        ),
        "homography_matrix": (
            st.session_state.homography_matrix.tolist()
            if st.session_state.homography_matrix is not None
            else None
        ),
    }

    st.download_button(
        label="💾 Export Updated JSON Config",
        data=json.dumps(export_payload, indent=2),
        file_name="floorplan_homography_config.json",
        mime="application/json",
        use_container_width=True,
    )

  with col_plot:
    st.markdown("#### Live Floorplan & ROI Preview")

    fig = go.Figure()

    # 1. Render DXF Walls
    for wall in st.session_state.get("dxf_walls", []):
      if hasattr(wall, "exterior"):
        wx, wy = wall.exterior.xy
      else:
        wx, wy = wall.xy
      fig.add_trace(
          go.Scatter(
              x=list(wx),
              y=list(wy),
              mode="lines",
              line=dict(color="black", width=1.5),
              showlegend=False,
          )
      )

    # 2. Render VGA Nodes
    if st.session_state.vga_grid_df is not None:
      fig.add_trace(
          go.Scatter(
              x=st.session_state.vga_grid_df["x"],
              y=st.session_state.vga_grid_df["y"],
              mode="markers",
              marker=dict(size=4, color="lightgray"),
              name="VGA Grid Nodes",
          )
      )

    # 3. Render Polygon ROI
    if len(poly_list) >= 3:
      px = [p[0] for p in poly_list] + [poly_list[0][0]]
      py = [p[1] for p in poly_list] + [poly_list[0][1]]

      fig.add_trace(
          go.Scatter(
              x=px,
              y=py,
              mode="lines+markers+text",
              fill="toself",
              fillcolor="rgba(255, 0, 0, 0.2)",
              line=dict(color="red", width=2.5),
              marker=dict(size=8, color="red"),
              text=[f"P{i+1}" for i in range(len(poly_list))] + [""],
              textposition="top right",
              name="ROI Polygon",
          )
      )

    fig.update_layout(
        height=500,
        xaxis=dict(title="X (meters)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (meters)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: OCCUPANCY TRACKING VIEW
# ==========================================
with tab_tracking:
  render_tracking_view(
      st.session_state.get("dxf_walls", []),
      st.session_state.get("vga_grid_df", None),
  )