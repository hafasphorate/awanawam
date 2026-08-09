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

import json
import re
import tempfile
import cv2
import numpy as np
import PIL.Image
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. HELPER: VIDEO FRAME EXTRACTION
# ==========================================
def extract_frame_from_video(uploaded_file, frame_number=0):
    """Extracts a specific frame (RGB) from a Streamlit UploadedFile object using OpenCV."""
    if uploaded_file is None:
        return None

    try:
        # Streamlit UploadedFile resides in memory; save to temp file for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error reading video frame: {e}")

    return None


import json
import re
import tempfile
import cv2
import numpy as np
import PIL.Image
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. HELPER: VIDEO FRAME EXTRACTION
# ==========================================
def extract_frame_from_video(uploaded_file, frame_number=0):
    """Extracts a specific frame (RGB) from a Streamlit UploadedFile object using OpenCV."""
    if uploaded_file is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error reading video frame: {e}")

    return None


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
for key, default in [
    ("four_corners", []),
    ("exclusion_masks", []),
    ("editing_point_idx", None),
    ("last_click_hash", None),
    ("mask_canvas_key_ver", 0),
    ("selected_frame_idx", 0),
    ("camera_view_range", None),
    ("selected_polygon_pts", []),
    ("vga_grid_df", None),
    ("homography_matrix", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================
# 3. TAB 2: REGION SELECTION & MASKING
# ==========================================
st.subheader("Step 2.2: Video Masking & ROI Corner Calibration")

# --- SECTION A: VIDEO PREVIEW & POLYGON MASKING ---
st.markdown("### 🚫 1. Video Polygon Masking (Exclusion Zones)")
st.info(
    "💡 **Instructions:** Use the draw tool in the Plotly toolbar (top right) "
    "to sketch exclusion zones directly on the video frame. Double-click to close a polygon."
)

if (
    "uploaded_video_file" in st.session_state
    and st.session_state.uploaded_video_file is not None
):
    col_m_slider, col_m_btns = st.columns([2.5, 1.5])

    with col_m_slider:
        frame_idx = st.slider(
            "Calibration Video Frame",
            min_value=0,
            max_value=1000,
            value=st.session_state.get("selected_frame_idx", 0),
            step=5,
        )
        st.session_state.selected_frame_idx = frame_idx

    with col_m_btns:
        st.markdown(
            "<div style='margin-top: 15px;'></div>", unsafe_allow_html=True
        )
        if st.button("🔥 Reset All Masks", use_container_width=True):
            st.session_state.exclusion_masks = []
            st.session_state.mask_canvas_key_ver += 1
            st.rerun()

    # Extract Video Frame
    raw_frame_rgb = extract_frame_from_video(
        st.session_state.uploaded_video_file,
        frame_number=st.session_state.selected_frame_idx,
    )

    if raw_frame_rgb is not None:
        img_h, img_w, _ = raw_frame_rgb.shape
        pil_img = PIL.Image.fromarray(raw_frame_rgb)

        fig_img = go.Figure()

        # Add Frame Image as Canvas Background
        fig_img.add_layout_image(
            dict(
                source=pil_img,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img_w,
                sizey=img_h,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        )

        # Draw Saved Exclusion Masks
        for idx, mask in enumerate(
            st.session_state.get("exclusion_masks", [])
        ):
            if len(mask) >= 3:
                mx = [p[0] for p in mask] + [mask[0][0]]
                my = [p[1] for p in mask] + [mask[0][1]]
                fig_img.add_trace(
                    go.Scatter(
                        x=mx,
                        y=my,
                        mode="lines+markers",
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.45)",
                        line=dict(color="#FF0000", width=3),
                        marker=dict(size=6, color="#FF0000"),
                        name=f"Mask Zone #{idx+1}",
                    )
                )

        fig_img.update_layout(
            template="plotly_dark",
            height=550,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(
                range=[0, img_w],
                showgrid=False,
                zeroline=False,
                constrain="domain",
            ),
            yaxis=dict(
                range=[img_h, 0],
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            dragmode="drawclosedpath",
            newshape=dict(
                fillcolor="rgba(255, 0, 0, 0.4)",
                line=dict(color="#FF0000", width=2),
            ),
            showlegend=False,
            uirevision=f"MASK_REV_{st.session_state.get('mask_canvas_key_ver', 0)}",
        )

        plotly_config = {
            "modeBarButtonsToAdd": [
                "drawclosedpath",
                "drawrect",
                "eraseshape",
            ],
            "displayModeBar": True,
        }

        v_events = st.plotly_chart(
            fig_img,
            use_container_width=True,
            on_select="rerun",
            config=plotly_config,
            key=f"video_mask_canvas_{st.session_state.get('mask_canvas_key_ver', 0)}",
        )

        # Parse Drawn Shapes from Selection
        if v_events and "selection" in v_events:
            shapes = v_events["selection"].get("shapes", [])
            if shapes:
                parsed_masks = []
                for shape in shapes:
                    shape_type = shape.get("type")
                    if shape_type == "path":
                        path_str = shape.get("path", "")
                        tokens = re.findall(
                            r"([MLZz])\s*([-\d\.\,\s]*)", path_str
                        )
                        pts = []
                        for cmd, coords_str in tokens:
                            if cmd in ["M", "L", "m", "l"]:
                                nums = re.findall(r"[-\d\.]+", coords_str)
                                if len(nums) >= 2:
                                    pts.append([float(nums[0]), float(nums[1])])

                        if len(pts) >= 3:
                            step = max(1, len(pts) // 15)
                            parsed_masks.append(pts[::step])

                    elif shape_type == "rect":
                        x0, x1 = float(shape["x0"]), float(shape["x1"])
                        y0, y1 = float(shape["y0"]), float(shape["y1"])
                        parsed_masks.append(
                            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                        )

                if parsed_masks and parsed_masks != st.session_state.get(
                    "exclusion_masks", []
                ):
                    st.session_state.exclusion_masks = parsed_masks
                    st.rerun()

        num_masks = len(st.session_state.get("exclusion_masks", []))
        if num_masks > 0:
            st.success(f"✅ **{num_masks}** Exclusion Zone(s) Active!")
    else:
        st.error("Failed to decode video frame at the selected frame index.")

else:
    st.warning(
        "⚠️ Please upload a video file in Step 2.1 to display the frame preview."
    )

st.markdown("---")

# --- SECTION B: INTERACTIVE FLOORPLAN CORNER CALIBRATION ---
st.markdown("### 📐 2. Camera ROI Corner Mapping")

col_controls, col_plot = st.columns([1.2, 2.8])

with col_controls:
    st.markdown("#### Corner Point Settings")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔴 Clear All Corners", use_container_width=True):
            st.session_state.four_corners = []
            st.session_state.editing_point_idx = None
            st.session_state.last_click_hash = None
            st.rerun()

    with col_btn2:
        if st.session_state.editing_point_idx is not None:
            if st.button("❌ Cancel Edit", use_container_width=True):
                st.session_state.editing_point_idx = None
                st.rerun()

    num_pts = len(st.session_state.four_corners)
    if st.session_state.editing_point_idx is not None:
        st.warning(
            f"🎯 **Editing P{st.session_state.editing_point_idx + 1}:** Click floorplan map to place."
        )
    elif num_pts < 4:
        st.info(
            f"⚠️ Selected **{num_pts}/4** corners. Click **{4 - num_pts}** more point(s) on the map."
        )
    else:
        st.success("✅ All 4 ROI Corners Configured!")

    # Corner Point List Controls
    corner_labels = [
        "P1 (Top-Left)",
        "P2 (Top-Right)",
        "P3 (Bottom-Right)",
        "P4 (Bottom-Left)",
    ]
    if len(st.session_state.four_corners) > 0:
        st.markdown("##### Selected Corners")
        for idx in range(len(st.session_state.four_corners)):
            pt = st.session_state.four_corners[idx]
            c_lbl = corner_labels[idx] if idx < 4 else f"P{idx+1}"

            col_info, col_edit, col_del = st.columns([2.0, 1.0, 0.8])
            with col_info:
                st.markdown(
                    f"**{c_lbl}**: `({round(pt[0], 2)}, {round(pt[1], 2)})`"
                )
            with col_edit:
                is_editing = st.session_state.editing_point_idx == idx
                btn_label = "🎯 Target" if is_editing else "✏️ Edit"
                if st.button(
                    btn_label, key=f"edit_btn_{idx}", use_container_width=True
                ):
                    st.session_state.editing_point_idx = idx
                    st.session_state.last_click_hash = None
                    st.rerun()
            with col_del:
                if st.button(
                    "🗑️",
                    key=f"del_btn_{idx}",
                    help=f"Delete {c_lbl}",
                    use_container_width=True,
                ):
                    st.session_state.four_corners.pop(idx)
                    if st.session_state.editing_point_idx == idx:
                        st.session_state.editing_point_idx = None
                    st.session_state.last_click_hash = None
                    st.rerun()

with col_plot:
    fig = go.Figure()

    # --- ROBUST CAD WALL PARSER ---
    # Look for CAD walls in common session state keys
    dxf_walls = (
        st.session_state.get("wall_lines")
        or st.session_state.get("dxf_walls")
        or st.session_state.get("walls")
        or []
    )
    
    wall_x, wall_y = [], []
    all_x, all_y = [], []

    for line in dxf_walls:
        try:
            # Case 1: Shapely LineString / Geometry with .xy property
            if hasattr(line, "xy"):
                coords_x, coords_y = list(line.xy[0]), list(line.xy[1])
                for i in range(len(coords_x) - 1):
                    wall_x.extend([coords_x[i], coords_x[i + 1], None])
                    wall_y.extend([coords_y[i], coords_y[i + 1], None])
                    all_x.extend([coords_x[i], coords_x[i + 1]])
                    all_y.extend([coords_y[i], coords_y[i + 1]])

            # Case 2: Line object with start/end attributes (ezdxf style)
            elif hasattr(line, "dxf"):
                start = line.dxf.start
                end = line.dxf.end
                wall_x.extend([start[0], end[0], None])
                wall_y.extend([start[1], end[1], None])
                all_x.extend([start[0], end[0]])
                all_y.extend([start[1], end[1]])

            # Case 3: List/Tuple of Point Pairs e.g., [ (x1, y1), (x2, y2) ]
            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                p1, p2 = line[0], line[1]
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
                wall_x.extend([x1, x2, None])
                wall_y.extend([y1, y2, None])
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
        except Exception:
            continue

    # Plot CAD Wall Geometry
    if wall_x and wall_y:
        fig.add_trace(
            go.Scatter(
                x=wall_x,
                y=wall_y,
                mode="lines",
                line=dict(color="#00ADB5", width=1.8),
                name="CAD Floorplan",
                hoverinfo="none",
                showlegend=False,
            )
        )
    else:
        st.warning("⚠️ No floorplan wall vectors detected in `st.session_state`. Please upload or parse your CAD file in Step 1.")

    # --- CLICK SENSOR GRID ---
    if all_x and all_y:
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        pad_x = (maxx - minx) * 0.05 if (maxx - minx) > 0 else 2.0
        pad_y = (maxy - miny) * 0.05 if (maxy - miny) > 0 else 2.0
        bounds_x = [minx - pad_x, maxx + pad_x]
        bounds_y = [miny - pad_y, maxy + pad_y]
    else:
        minx, maxx = -5.0, 60.0
        miny, maxy = -5.0, 60.0
        bounds_x = [-5.0, 60.0]
        bounds_y = [-5.0, 60.0]

    gx = np.linspace(minx, maxx, 80)
    gy = np.linspace(miny, maxy, 80)
    g_xx, g_yy = np.meshgrid(gx, gy)

    fig.add_trace(
        go.Scatter(
            x=g_xx.flatten(),
            y=g_yy.flatten(),
            mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0.001)"),
            hoverinfo="x+y",
            showlegend=False,
            name="click_grid",
        )
    )

    # --- SELECTED ROI CORNERS & POLYGON ---
    pts = st.session_state.four_corners
    if len(pts) > 0:
        px_pts = [p[0] for p in pts]
        py_pts = [p[1] for p in pts]

        if len(pts) == 4:
            fig.add_trace(
                go.Scatter(
                    x=px_pts + [px_pts[0]],
                    y=py_pts + [py_pts[0]],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(0, 230, 118, 0.35)",
                    line=dict(color="#00FF66", width=2.5),
                    name="ROI Polygon",
                )
            )

        marker_colors = [
            "#FFD700" if (st.session_state.editing_point_idx == i) else "#00FF66"
            for i in range(len(pts))
        ]

        fig.add_trace(
            go.Scatter(
                x=px_pts,
                y=py_pts,
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=marker_colors,
                    symbol="circle",
                    line=dict(color="#000000", width=1.5),
                ),
                text=[f"P{i+1}" for i in range(len(pts))],
                textposition="top right",
                textfont=dict(size=14, color="#FFFFFF"),
                name="Corners",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=580,
        xaxis=dict(
            title="X Coordinate (m)",
            range=bounds_x,
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title="Y Coordinate (m)",
            range=bounds_y,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
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

    # Click Handling
    if chart_events and "selection" in chart_events:
        event_pts = chart_events["selection"].get("points", [])
        if event_pts:
            click_x = float(event_pts[0]["x"])
            click_y = float(event_pts[0]["y"])
            click_hash = f"{click_x:.2f}_{click_y:.2f}_{st.session_state.editing_point_idx}"

            if click_hash != st.session_state.last_click_hash:
                st.session_state.last_click_hash = click_hash

                if st.session_state.editing_point_idx is not None:
                    target_idx = st.session_state.editing_point_idx
                    st.session_state.four_corners[target_idx] = [
                        click_x,
                        click_y,
                    ]
                    st.session_state.editing_point_idx = None
                    st.rerun()

                elif len(st.session_state.four_corners) < 4:
                    st.session_state.four_corners.append([click_x, click_y])
                    st.rerun()

# ==========================================
# TAB 3: OCCUPANCY TRACKING VIEW
# ==========================================
with tab_tracking:
    render_tracking_view(
        st.session_state.get("dxf_walls", []),
        st.session_state.get("vga_grid_df", None),
    )