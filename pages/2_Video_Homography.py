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
# TAB 2: REGION SELECTION & EDITING (ENHANCED)
# ==========================================
with tab_region:
    st.subheader("Step 2.2: Define 4 ROI Camera Corners on Floorplan")
    st.info(
        "💡 **4-Point Click Selection:** Click **4 points** anywhere on the floorplan canvas to define "
        "the corners of your camera's field of view. Zoom in as needed — the viewport position will remain locked between clicks!"
    )

    if "four_corners" not in st.session_state:
        st.session_state.four_corners = []
    if "editing_point_idx" not in st.session_state:
        st.session_state.editing_point_idx = None

    col_controls, col_plot = st.columns([1.1, 2.4])

    with col_controls:
        st.markdown("#### Corner Coordinates (CAD World)")

        # Global Control Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔴 Reset All", use_container_width=True):
                st.session_state.four_corners = []
                st.session_state.selected_polygon_pts = []
                st.session_state.editing_point_idx = None
                st.rerun()

        with col_btn2:
            if st.session_state.editing_point_idx is not None:
                if st.button("❌ Cancel Edit", use_container_width=True):
                    st.session_state.editing_point_idx = None
                    st.rerun()

        # Display Selection Status
        num_pts = len(st.session_state.four_corners)
        if st.session_state.editing_point_idx is not None:
            edit_num = st.session_state.editing_point_idx + 1
            st.warning(f"🎯 **Editing Mode Active:** Click anywhere on the map to re-position **P{edit_num}**.")
        elif num_pts < 4:
            st.info(f"⚠️ Selected **{num_pts}/4** corners. Click **{4 - num_pts}** more point(s) on the map.")
        else:
            st.success("✅ All 4 Corners Selected!")

        # Per-Point Table with Redo / Edit Buttons
        corner_labels = ["P1 (Top-Left)", "P2 (Top-Right)", "P3 (Bottom-Right)", "P4 (Bottom-Left)"]
        
        if len(st.session_state.four_corners) > 0:
            st.markdown("---")
            st.markdown("##### Selected Points & Individual Redo")
            for idx, pt in enumerate(st.session_state.four_corners[:4]):
                c_lbl = corner_labels[idx] if idx < 4 else f"P{idx+1}"
                col_info, col_act = st.columns([2.5, 1])

                with col_info:
                    st.markdown(f"**{c_lbl}**: `({round(pt[0], 2)}, {round(pt[1], 2)})`")
                with col_act:
                    is_currently_editing = (st.session_state.editing_point_idx == idx)
                    btn_label = "🎯 Target" if is_currently_editing else "✏️ Edit"
                    if st.button(btn_label, key=f"edit_btn_{idx}", use_container_width=True):
                        st.session_state.editing_point_idx = idx
                        st.rerun()

        st.markdown("---")

        # Config Exporter
        export_payload = {
            "polygon_points": st.session_state.four_corners[:4],
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
            label="💾 Export JSON Config",
            data=json.dumps(export_payload, indent=2),
            file_name="floorplan_homography_config.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_plot:
        st.markdown("#### Click Floorplan Canvas")

        fig = go.Figure()

        # 1. Unpack Wall Lines (Supports both live DXF objects and JSON imported dicts/lists)
        wall_x, wall_y = [], []
        all_x, all_y = [], []

        dxf_walls = st.session_state.get("dxf_walls", [])
        for line in dxf_walls:
            # Handles Shapely LineString / Polygon objects from DXF parser
            if hasattr(line, "xy"):
                x, y = line.xy
                wall_x.extend([x[0], x[1], None])
                wall_y.extend([y[0], y[1], None])
                all_x.extend(x)
                all_y.extend(y)
            # Handles raw coordinate tuples/lists parsed from JSON import
            elif isinstance(line, (list, tuple)):
                for pt in line:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        all_x.append(pt[0])
                        all_y.append(pt[1])
                for i in range(len(line) - 1):
                    wall_x.extend([line[i][0], line[i+1][0], None])
                    wall_y.extend([line[i][1], line[i+1][1], None])

        if wall_x:
            fig.add_trace(
                go.Scatter(
                    x=wall_x,
                    y=wall_y,
                    mode="lines",
                    line=dict(color="#00ADB5", width=1.5),
                    name="CAD Walls",
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # 2. Add Click-Receiver Sensor Grid Across Bounding Box
        if all_x and all_y:
            minx, maxx = min(all_x), max(all_x)
            miny, maxy = min(all_y), max(all_y)
            grid_step_x = (maxx - minx) / 60 if (maxx - minx) > 0 else 1.0
            grid_step_y = (maxy - miny) / 60 if (maxy - miny) > 0 else 1.0

            gx = np.arange(minx, maxx, grid_step_x)
            gy = np.arange(miny, maxy, grid_step_y)
            g_xx, g_yy = np.meshgrid(gx, gy)

            fig.add_trace(
                go.Scatter(
                    x=g_xx.flatten(),
                    y=g_yy.flatten(),
                    mode="markers",
                    marker=dict(size=14, color="rgba(0, 0, 0, 0.001)"),
                    hoverinfo="none",
                    showlegend=False,
                    name="click_sensor_grid",
                )
            )

        # 3. Render Selected Corners & ROI Fill Area
        pts = st.session_state.four_corners
        if len(pts) > 0:
            px = [p[0] for p in pts]
            py = [p[1] for p in pts]

            if len(pts) == 4:
                px_closed = px + [px[0]]
                py_closed = py + [py[0]]
                fig.add_trace(
                    go.Scatter(
                        x=px_closed,
                        y=py_closed,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(0, 230, 118, 0.35)",
                        line=dict(color="#00FF66", width=2.5),
                        name="Camera ROI Zone",
                    )
                )

            # Draw Point Markers & Highlight active editing point in yellow
            marker_colors = [
                "#FFD700" if (st.session_state.editing_point_idx == i) else "#00FF66"
                for i in range(len(pts))
            ]

            fig.add_trace(
                go.Scatter(
                    x=px,
                    y=py,
                    mode="markers+text",
                    marker=dict(size=14, color=marker_colors, symbol="circle"),
                    text=[f"P{i+1}" for i in range(len(pts))],
                    textposition="top right",
                    textfont=dict(size=14, color="#FFFFFF"),
                    name="Selected Corners",
                )
            )

        # Layout Configuration with uirevision=True to retain zoom level across reruns
        fig.update_layout(
            template="plotly_dark",
            height=620,
            xaxis=dict(title="X Coordinate", scaleanchor="y", scaleratio=1, showgrid=True),
            yaxis=dict(title="Y Coordinate", showgrid=True),
            margin=dict(l=10, r=10, t=30, b=10),
            clickmode="event+select",
            dragmode=False,
            hovermode="closest",
            uirevision="constant_viewport", # Retains user zoom & pan position
        )

        chart_events = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"four_corner_canvas_{len(st.session_state.four_corners)}_{st.session_state.editing_point_idx}",
        )

        # Event Dispatcher for Direct Map Clicks
        if chart_events and "selection" in chart_events:
            event_pts = chart_events["selection"].get("points", [])
            if event_pts:
                click_x = float(event_pts[0]["x"])
                click_y = float(event_pts[0]["y"])

                # Mode A: Replacing/Redoing a Specific Selected Corner
                if st.session_state.editing_point_idx is not None:
                    target_idx = st.session_state.editing_point_idx
                    st.session_state.four_corners[target_idx] = [click_x, click_y]
                    st.session_state.editing_point_idx = None
                    st.session_state.selected_polygon_pts = [
                        {"X (m)": p[0], "Y (m)": p[1]} for p in st.session_state.four_corners
                    ]
                    st.rerun()

                # Mode B: Adding a New Corner (Up to 4 total)
                elif len(st.session_state.four_corners) < 4:
                    if not st.session_state.four_corners or (
                        st.session_state.four_corners[-1] != [click_x, click_y]
                    ):
                        st.session_state.four_corners.append([click_x, click_y])
                        st.session_state.selected_polygon_pts = [
                            {"X (m)": p[0], "Y (m)": p[1]} for p in st.session_state.four_corners
                        ]
                        st.rerun()

# ==========================================
# TAB 3: OCCUPANCY TRACKING VIEW
# ==========================================
with tab_tracking:
    render_tracking_view(
        st.session_state.get("dxf_walls", []),
        st.session_state.get("vga_grid_df", None),
    )