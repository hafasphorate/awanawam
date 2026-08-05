# pages/2_Video_Homography.py
import json
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon
import streamlit as st

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
if "camera_view_range" not in st.session_state:
    st.session_state.camera_view_range = None
if "last_click_hash" not in st.session_state:
    st.session_state.last_click_hash = None


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

    # Option A: JSON Import
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

                if "vga_grid" in data and data["vga_grid"]:
                    st.session_state.vga_grid_df = pd.DataFrame(data["vga_grid"])
                    st.success(f"✅ Loaded VGA Grid ({len(st.session_state.vga_grid_df)} nodes)")

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
                        st.session_state.four_corners = [[p["X (m)"], p["Y (m)"]] for p in formatted_pts]
                        st.success(f"✅ Loaded {len(formatted_pts)} polygon vertices")

                if "homography_matrix" in data and data["homography_matrix"]:
                    st.session_state.homography_matrix = np.array(data["homography_matrix"])
                    st.success("✅ Loaded pre-saved Homography Matrix")

                if "dxf_walls" in data and data["dxf_walls"]:
                    walls = []
                    for w in data["dxf_walls"]:
                        if len(w) >= 3:
                            walls.append(Polygon(w))
                        elif len(w) == 2:
                            walls.append(LineString(w))
                    st.session_state.dxf_walls = walls
                    st.session_state.camera_view_range = None
                    st.success(f"✅ Loaded {len(walls)} wall geometries from JSON")

            except Exception as e:
                st.error(f"Error parsing JSON: {e}")

    # Option B: CAD Import
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
                    st.session_state.camera_view_range = None
                    st.success(f"✅ Successfully parsed CAD! {len(wall_lines)} wall boundary lines ready.")
            except Exception as e:
                st.error(f"Failed to parse CAD file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.markdown("---")

    # Video Upload
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
    st.subheader("Step 2.2: Define 4 ROI Camera Corners on Floorplan")
    st.info(
        "💡 **4-Point Click Selection:** Click **4 points** on the floorplan to set your camera corners. "
        "Zoom or pan freely — your zoom state is locked across clicks!"
    )

    if "four_corners" not in st.session_state:
        st.session_state.four_corners = []
    if "editing_point_idx" not in st.session_state:
        st.session_state.editing_point_idx = None

    col_controls, col_plot = st.columns([1.1, 2.4])

    with col_controls:
        st.markdown("#### Corner Coordinates (CAD World)")

        # Global Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔴 Reset All", use_container_width=True):
                st.session_state.four_corners = []
                st.session_state.selected_polygon_pts = []
                st.session_state.editing_point_idx = None
                st.session_state.camera_view_range = None
                st.session_state.last_click_hash = None
                st.rerun()

        with col_btn2:
            if st.session_state.editing_point_idx is not None:
                if st.button("❌ Cancel Edit", use_container_width=True):
                    st.session_state.editing_point_idx = None
                    st.rerun()

        # Selection Status
        num_pts = len(st.session_state.four_corners)
        if st.session_state.editing_point_idx is not None:
            edit_num = st.session_state.editing_point_idx + 1
            st.warning(f"🎯 **Editing P{edit_num}:** Click anywhere on the map to set the new position.")
        elif num_pts < 4:
            st.info(f"⚠️ Selected **{num_pts}/4** corners. Click **{4 - num_pts}** more point(s) on the map.")
        else:
            st.success("✅ All 4 Corners Selected!")

        # Point Table with Edit Controls
        corner_labels = ["P1 (Top-Left)", "P2 (Top-Right)", "P3 (Bottom-Right)", "P4 (Bottom-Left)"]
        
        if len(st.session_state.four_corners) > 0:
            st.markdown("---")
            st.markdown("##### Selected Points")
            for idx, pt in enumerate(st.session_state.four_corners[:4]):
                c_lbl = corner_labels[idx] if idx < 4 else f"P{idx+1}"
                col_info, col_act = st.columns([2.5, 1])

                with col_info:
                    st.markdown(f"**{c_lbl}**: `({round(pt[0], 2)}, {round(pt[1], 2)})`")
                with col_act:
                    is_editing = (st.session_state.editing_point_idx == idx)
                    btn_label = "🎯 Active" if is_editing else "✏️ Edit"
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

        # 1. Unpack CAD Walls
        wall_x, wall_y = [], []
        all_x, all_y = [], []

        dxf_walls = st.session_state.get("dxf_walls", [])
        for line in dxf_walls:
            if hasattr(line, "xy"):
                x, y = line.xy
                wall_x.extend([x[0], x[1], None])
                wall_y.extend([y[0], y[1], None])
                all_x.extend(x)
                all_y.extend(y)
            elif isinstance(line, (list, tuple)):
                for pt in line:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        all_x.append(pt[0])
                        all_y.append(pt[1])
                for i in range(len(line) - 1):
                    wall_x.extend([line[i][0], line[i+1][0], None])
                    wall_y.extend([line[i+1][0], None])

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

        # Compute Initial Bounds if not set
        if all_x and all_y:
            minx, maxx = min(all_x), max(all_x)
            miny, maxy = min(all_y), max(all_y)
            pad_x = (maxx - minx) * 0.05 if (maxx - minx) > 0 else 1.0
            pad_y = (maxy - miny) * 0.05 if (maxy - miny) > 0 else 1.0
            init_x_range = [minx - pad_x, maxx + pad_x]
            init_y_range = [miny - pad_y, maxy + pad_y]
        else:
            minx, maxx = -1, 10
            miny, maxy = -1, 10
            init_x_range = [-1, 10]
            init_y_range = [-1, 10]

        if st.session_state.camera_view_range is None:
            st.session_state.camera_view_range = {
                "x": init_x_range,
                "y": init_y_range,
            }

        # 2. Dense Click Target Grid
        if all_x and all_y:
            grid_step_x = (maxx - minx) / 80 if (maxx - minx) > 0 else 1.0
            grid_step_y = (maxy - miny) / 80 if (maxy - miny) > 0 else 1.0

            gx = np.arange(minx, maxx, grid_step_x)
            gy = np.arange(miny, maxy, grid_step_y)
            g_xx, g_yy = np.meshgrid(gx, gy)

            fig.add_trace(
                go.Scatter(
                    x=g_xx.flatten(),
                    y=g_yy.flatten(),
                    mode="markers",
                    marker=dict(size=12, color="rgba(0,0,0,0.001)"),
                    hoverinfo="none",
                    showlegend=False,
                    name="click_sensor_grid",
                )
            )

        # 3. Selected Corners & ROI Fill
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

        # Apply persisted camera ranges so zoom position is preserved perfectly
        fig.update_layout(
            template="plotly_dark",
            height=620,
            xaxis=dict(
                title="X Coordinate",
                range=st.session_state.camera_view_range["x"],
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
            ),
            yaxis=dict(
                title="Y Coordinate",
                range=st.session_state.camera_view_range["y"],
                showgrid=True,
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            clickmode="event+select",
            dragmode="pan",
            hovermode="closest",
            uirevision="constant_lock",
        )

        chart_events = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="roi_floorplan_canvas",
        )

        # Preserve Zoom/Pan position when user zooms or pans
        if chart_events and "relayout" in chart_events:
            relayout = chart_events["relayout"]
            if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
                st.session_state.camera_view_range["x"] = [
                    relayout["xaxis.range[0]"],
                    relayout["xaxis.range[1]"],
                ]
            if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
                st.session_state.camera_view_range["y"] = [
                    relayout["yaxis.range[0]"],
                    relayout["yaxis.range[1]"],
                ]

        # Handle Click Selection & Edit Dispatching
        if chart_events and "selection" in chart_events:
            event_pts = chart_events["selection"].get("points", [])
            if event_pts:
                click_x = float(event_pts[0]["x"])
                click_y = float(event_pts[0]["y"])
                click_hash = f"{click_x:.3f}_{click_y:.3f}_{st.session_state.editing_point_idx}"

                # Only process if this is a fresh new click
                if click_hash != st.session_state.last_click_hash:
                    st.session_state.last_click_hash = click_hash

                    # EDIT MODE: Replace the active target corner
                    if st.session_state.editing_point_idx is not None:
                        target_idx = st.session_state.editing_point_idx
                        st.session_state.four_corners[target_idx] = [click_x, click_y]
                        st.session_state.editing_point_idx = None
                        st.session_state.selected_polygon_pts = [
                            {"X (m)": p[0], "Y (m)": p[1]} for p in st.session_state.four_corners
                        ]
                        st.rerun()

                    # ADD MODE: Add up to 4 corners
                    elif len(st.session_state.four_corners) < 4:
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