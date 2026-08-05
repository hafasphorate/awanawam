# pages/2_Video_Homography.py
import json
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon
import streamlit as st

# Reuse your working CAD engine and tracking functions
from utils.vga_engine import process_cad_file
from utils.tracking_engine import extract_frame_from_video
from views.tracking_view import render_tracking_view

st.set_page_config(page_title="Module 2: Video Homography & Tracking", layout="wide")

st.title("📹 Module 2: Video Homography & Region Selection")

# Initialize Session State safely
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

# ==========================================
# TAB 1: FILE & VIDEO IMPORT
# ==========================================
with tab_import:
    st.subheader("Step 2.1: Load CAD (DXF/DWG) or JSON Config & Video")

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

                # 1. Parse VGA Grid
                if "vga_grid" in data and data["vga_grid"]:
                    st.session_state.vga_grid_df = pd.DataFrame(data["vga_grid"])
                    st.success(f"✅ Loaded VGA Grid ({len(st.session_state.vga_grid_df)} nodes)")

                # 2. Parse Polygon Points Safely (Prevents KeyError "X (m)")
                if "polygon_points" in data and data["polygon_points"]:
                    raw_pts = data["polygon_points"]
                    formatted_pts = []

                    for pt in raw_pts:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            formatted_pts.append({"X (m)": float(pt[0]), "Y (m)": float(pt[1])})
                        elif isinstance(pt, dict):
                            x_val = pt.get("X (m)", pt.get("x", pt.get("X", 0.0)))
                            y_val = pt.get("Y (m)", pt.get("y", pt.get("Y", 0.0)))
                            formatted_pts.append({"X (m)": float(x_val), "Y (m)": float(y_val)})

                    if formatted_pts:
                        st.session_state.selected_polygon_pts = formatted_pts
                        st.success(f"✅ Loaded {len(formatted_pts)} polygon vertices")

                # 3. Parse Homography Matrix
                if "homography_matrix" in data and data["homography_matrix"]:
                    st.session_state.homography_matrix = np.array(data["homography_matrix"])
                    st.success("✅ Loaded pre-saved Homography Matrix")

                # 4. Parse Walls
                if "dxf_walls" in data and data["dxf_walls"]:
                    walls = []
                    for w in data["dxf_walls"]:
                        if len(w) >= 3:
                            walls.append(Polygon(w))
                        elif len(w) == 2:
                            walls.append(LineString(w))
                    st.session_state.dxf_walls = walls
                    st.success(f"✅ Loaded {len(walls)} wall geometries from JSON")

            except Exception as e:
                st.error(f"Error parsing JSON: {e}")

    # 📐 Option B: CAD Import (DXF / DWG) - Uses same backend as VGA Page
    with col_dxf:
        st.markdown("### 📐 Option B: Import Raw CAD File")
        uploaded_cad = st.file_uploader(
            "Upload CAD Floorplan (DXF or DWG)",
            type=["dxf", "dwg"],
            key="cad_uploader",
        )

        if uploaded_cad is not None:
            file_ext = "." + uploaded_cad.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_cad.getvalue())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Processing CAD file via VGA Engine..."):
                    wall_lines = process_cad_file(tmp_path)
                    st.session_state.dxf_walls = wall_lines
                    st.success(f"✅ Successfully parsed CAD! {len(wall_lines)} wall boundary lines ready.")
            except Exception as e:
                st.error(f"Failed to parse CAD file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

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
        st.success("✅ Video file attached successfully!")

        frame_idx = st.slider(
            "Select Calibration Preview Frame",
            min_value=0,
            max_value=1000,
            value=st.session_state.selected_frame_idx,
            step=5,
        )
        st.session_state.selected_frame_idx = frame_idx

        raw_frame_rgb = extract_frame_from_video(uploaded_video, frame_number=frame_idx)
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
        st.markdown("#### Polygon Vertices (CAD World)")

        # Ensure dataframe columns are standardized and fallback if key missing
        default_df = pd.DataFrame(st.session_state.selected_polygon_pts)
        if "X (m)" not in default_df.columns or "Y (m)" not in default_df.columns:
            default_df = pd.DataFrame([{"X (m)": 0.0, "Y (m)": 0.0}, {"X (m)": 10.0, "Y (m)": 0.0}])

        edited_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            key="poly_editor",
        )

        # Safe key extraction prevents KeyError
        poly_pts_dicts = edited_df.to_dict(orient="records")
        st.session_state.selected_polygon_pts = poly_pts_dicts

        poly_list = []
        for p in poly_pts_dicts:
            x_val = p.get("X (m)", p.get("x", p.get("X", 0.0)))
            y_val = p.get("Y (m)", p.get("y", p.get("Y", 0.0)))
            poly_list.append([float(x_val), float(y_val)])

        st.markdown("---")

        # JSON Exporter
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

        # 1. Render DXF/DWG Walls
        wall_x, wall_y = [], []
        for line in st.session_state.get("dxf_walls", []):
            if hasattr(line, "xy"):
                x, y = line.xy
                wall_x.extend([x[0], x[1], None])
                wall_y.extend([y[0], y[1], None])

        if wall_x:
            fig.add_trace(
                go.Scatter(
                    x=wall_x,
                    y=wall_y,
                    mode="lines",
                    line=dict(color="#00ADB5", width=1.5),
                    name="CAD Walls",
                    showlegend=True,
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
            template="plotly_dark",
            height=500,
            xaxis=dict(title="X Coordinate", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y Coordinate"),
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