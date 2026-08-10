import json
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import PIL.Image
import re
import streamlit as st
from shapely.geometry import LineString, Polygon


from utils.vga_engine import process_cad_file
from utils.tracking_engine import extract_frame_from_video
from views.tracking_view import render_tracking_view

st.set_page_config(page_title="Module 2: Video Homography & Tracking", layout="wide")

st.title("📹 Module 3: Video Homography & Region Selection")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "wall_lines" not in st.session_state:
    st.session_state.wall_lines = []
if "dxf_walls" not in st.session_state:
    st.session_state.dxf_walls = []
if "vga_grid_df" not in st.session_state:
    st.session_state.vga_grid_df = None
if "selected_polygon_pts" not in st.session_state:
    st.session_state.selected_polygon_pts = []
if "homography_matrix" not in st.session_state:
    st.session_state.homography_matrix = None
if "selected_frame_idx" not in st.session_state:
    st.session_state.selected_frame_idx = 0
if "four_corners" not in st.session_state:
    st.session_state.four_corners = []
if "editing_point_idx" not in st.session_state:
    st.session_state.editing_point_idx = None
if "processed_click_sig" not in st.session_state:
    st.session_state.processed_click_sig = None
if "tracking_results_df" not in st.session_state:
    st.session_state.tracking_results_df = None

# Plot Range Axes State
if "current_x_range" not in st.session_state:
    st.session_state.current_x_range = None
if "current_y_range" not in st.session_state:
    st.session_state.current_y_range = None

# Exclusion Masking State
if "exclusion_masks" not in st.session_state:
    st.session_state.exclusion_masks = []
if "active_mask_pts" not in st.session_state:
    st.session_state.active_mask_pts = []
if "mask_click_sig" not in st.session_state:
    st.session_state.mask_click_sig = None
if "mask_canvas_key_ver" not in st.session_state:
    st.session_state.mask_canvas_key_ver = 0

# Navigation Tabs
tab_import, tab_region, tab_tracking, tab_playback = st.tabs([
    "📂 3.1 Import CAD / Session & Video",
    "📐 3.2 Define ROI & Video Masking",
    "🔥 3.3 Occupancy Analytics",
    "🎬 3.4 2D Playback & Crowd Heatmaps",
])

# ==========================================
# HELPER FUNCTIONS: ROBUST CAD WALL PARSING
# ==========================================
def normalize_line_to_dict(line):
    """Converts Shapely objects, tuples, or dicts to standard dicts {"x": [x1, x2], "y": [y1, y2]}."""
    try:
        if hasattr(line, "xy"):  # Shapely LineString
            x, y = line.xy
            return {"x": [float(x[0]), float(x[1])], "y": [float(y[0]), float(y[1])]}
        elif isinstance(line, dict) and "x" in line and "y" in line:
            return {"x": [float(line["x"][0]), float(line["x"][1])], "y": [float(line["y"][0]), float(line["y"][1])]}
        elif isinstance(line, (list, tuple)) and len(line) >= 2:
            p1, p2 = line[0], line[1]
            if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)):
                return {"x": [float(p1[0]), float(p2[0])], "y": [float(p1[1]), float(p2[1])]}
    except Exception:
        pass
    return None

def extract_walls_from_session(data):
    """Recursively searches for wall lines across common JSON export structures and normalizes them to Shapely LineString objects."""
    normalized_walls = []
    raw_found = []

    if isinstance(data, dict):
        if "floorplan" in data and isinstance(data["floorplan"], dict):
            raw_found.extend(data["floorplan"].get("wall_lines", []))

        for key in ["wall_lines", "dxf_walls", "walls", "cad_walls", "geometry_lines"]:
            if key in data and isinstance(data[key], list):
                raw_found.extend(data[key])

    elif isinstance(data, list):
        raw_found = data

    for item in raw_found:
        try:
            if hasattr(item, "xy"):
                normalized_walls.append(LineString(item.coords))
            elif isinstance(item, dict) and "x" in item and "y" in item:
                xs = item["x"]
                ys = item["y"]
                if len(xs) >= 2 and len(ys) >= 2:
                    normalized_walls.append(LineString([(float(xs[0]), float(ys[0])), (float(xs[1]), float(ys[1]))]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                p1, p2 = item[0], item[1]
                if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)):
                    normalized_walls.append(LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))]))
        except Exception:
            continue

    return normalized_walls

def add_cad_walls_to_fig(fig, wall_color="#00ADB5", width=1.5):
    """Overlays CAD wall lines onto any Plotly canvas using pre-normalized dicts."""
    walls_data = st.session_state.get("wall_lines", []) or st.session_state.get("dxf_walls", [])

    wall_x, wall_y = [], []
    for line in walls_data:
        d = normalize_line_to_dict(line)
        if d:
            wall_x.extend([d["x"][0], d["x"][1], None])
            wall_y.extend([d["y"][0], d["y"][1], None])

    if wall_x:
        fig.add_trace(go.Scatter(
            x=wall_x, y=wall_y,
            mode="lines",
            line=dict(color=wall_color, width=width),
            name="CAD Walls",
            hoverinfo="none",
            showlegend=False
        ))
    return fig

# ==========================================
# TAB 1: FILE & VIDEO IMPORT
# ==========================================
with tab_import:
    st.subheader("Step 3.1: Load CAD (DXF/DWG) or JSON Config & Surveillance Video")

    col_json, col_dxf = st.columns(2)

    with col_json:
        st.markdown("### 📄 Option A: Import Exported JSON Session")
        uploaded_json = st.file_uploader(
            "Upload JSON Floorplan / Export (VGA + Polygon Config)",
            type=["json"],
            key="json_uploader_tab1",
        )

        if uploaded_json is not None:
            try:
                data = json.load(uploaded_json)

                # 1. Extract and normalize CAD walls
                extracted_walls = extract_walls_from_session(data)
                if extracted_walls:
                    st.session_state["wall_lines"] = extracted_walls
                    st.session_state["dxf_walls"] = extracted_walls
                    st.success(f"✅ Loaded and validated {len(extracted_walls)} CAD wall segments!")
                else:
                    st.error("❌ JSON loaded, but zero valid wall segments could be extracted. Please check file formatting.")

                # 2. Extract VGA Grid Data
                vga_raw = data.get("vga_results") or data.get("vga_grid")
                if vga_raw:
                    st.session_state["vga_df"] = pd.DataFrame(vga_raw)
                    st.session_state.vga_grid_df = st.session_state["vga_df"]
                    st.success(f"✅ Loaded VGA Grid ({len(st.session_state.vga_grid_df)} nodes)")

                # 3. Extract ROI Polygon / Corners
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

                if "homography_matrix" in data and data["homography_matrix"]:
                    st.session_state.homography_matrix = np.array(data["homography_matrix"])

                if "exclusion_masks" in data and data["exclusion_masks"]:
                    st.session_state.exclusion_masks = data["exclusion_masks"]

            except Exception as e:
                st.error(f"Error parsing JSON session file: {e}")

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
                    raw_wall_lines = process_cad_file(tmp_path)
                    st.session_state.dxf_walls = raw_wall_lines
                    st.session_state.wall_lines = raw_wall_lines
                    st.success(f"✅ Successfully parsed CAD! {len(raw_wall_lines)} wall boundary lines ready.")
            except Exception as e:
                st.error(f"Failed to parse CAD file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.markdown("---")
    st.markdown("### 📹 Surveillance Video Target")
    uploaded_video = st.file_uploader(
        "Upload Surveillance Video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
    )

    if uploaded_video:
        st.session_state.uploaded_video_file = uploaded_video
        st.success("✅ Video file attached successfully!")

# ==========================================
# 3. TAB 2: REGION SELECTION & MASKING
# ==========================================
with tab_region:
    st.subheader("Step 3.2: Video Masking & ROI Corner Calibration")

    # --- SECTION A: VIDEO PREVIEW & POLYGON MASKING ---
    st.markdown("### 🚫 1. Video Polygon Masking (Exclusion Zones)")
    st.info(
        "💡 **Instructions:** Use the draw tool in the Plotly toolbar (top right) "
        "to sketch exclusion zones directly on the video frame. Double-click to close a polygon."
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
            "⚠️ Please upload a video file in Step 3.1 to display the frame preview."
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
        st.session_state.get("wall_lines", []) or st.session_state.get("dxf_walls", []),
        st.session_state.get("vga_grid_df", None),
    )

# ==========================================
# TAB 4: 2D PLAYBACK & CROWD HEATMAPS
# ==========================================
with tab_playback:
    st.subheader("Step 3.4: 2D Playback & Crowd Trajectory Analytics")

    st.markdown("### 1. Import Tracking Dataset")
    col_up1, col_up2 = st.columns(2)

    def parse_tracking_json(raw_json):
        if isinstance(raw_json, list):
            return pd.DataFrame(raw_json)

        if isinstance(raw_json, dict):
            for key in ["tracking_points", "tracking_results", "pedestrian_trajectories", "trajectories", "tracking_data"]:
                if key in raw_json and isinstance(raw_json[key], list) and len(raw_json[key]) > 0:
                    return pd.DataFrame(raw_json[key])

        return pd.json_normalize(raw_json)

    with col_up1:
        uploaded_tb_json = st.file_uploader("Upload JSON Export (from Step 3.3)", type=["json"], key="tb_json_up")
        if uploaded_tb_json is not None:
            try:
                raw_json = json.load(uploaded_tb_json)
                df_loaded = parse_tracking_json(raw_json)
                st.session_state.tracking_results_df = df_loaded
                st.success(f"✅ Successfully imported {len(df_loaded)} tracking records!")
            except Exception as e:
                st.error(f"Error reading JSON: {e}")

    with col_up2:
        uploaded_tb_csv = st.file_uploader("Upload CSV Tracking Export", type=["csv"], key="tb_csv_up")
        if uploaded_tb_csv is not None:
            try:
                st.session_state.tracking_results_df = pd.read_csv(uploaded_tb_csv)
                st.success("✅ Successfully imported CSV tracking records!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")

    df_track = st.session_state.get("tracking_results_df")

    if df_track is not None and not df_track.empty:
        df_track.columns = [str(c).lower().strip() for c in df_track.columns]

        frame_col = next((c for c in ["frame_idx", "frame", "frame_number", "timestamp"] if c in df_track.columns), None)
        x_col = next((c for c in ["world_x", "x", "x (m)", "x_m", "pos_x", "x_canvas", "img_x"] if c in df_track.columns), None)
        y_col = next((c for c in ["world_y", "y", "y (m)", "y_m", "pos_y", "y_canvas", "img_y"] if c in df_track.columns), None)
        id_col = next((c for c in ["track_id", "id", "person_id"] if c in df_track.columns), "track_id")

        if x_col and y_col:
            if not frame_col:
                df_track["frame_idx"] = 0
                frame_col = "frame_idx"

            if id_col not in df_track.columns:
                df_track[id_col] = 1

            if "speed" not in df_track.columns:
                df_track = df_track.sort_values(by=[id_col, frame_col])
                df_track["dx"] = df_track.groupby(id_col)[x_col].diff().fillna(0)
                df_track["dy"] = df_track.groupby(id_col)[y_col].diff().fillna(0)
                df_track["speed"] = np.sqrt(df_track["dx"]**2 + df_track["dy"]**2)

            st.markdown("### 2. Motion Playback & Frame Analytics")
            frames_available = sorted(df_track[frame_col].unique())
            selected_f = st.slider(
                "Select Frame for Instant Inspection",
                min_value=int(min(frames_available)),
                max_value=int(max(frames_available)),
                value=int(min(frames_available)),
            )

            curr_frame_df = df_track[df_track[frame_col] == selected_f]

            col_fb1, col_fb2 = st.columns(2)

            with col_fb1:
                st.markdown(f"**Pedestrian Plan View (Frame #{selected_f})**")
                fig_play = go.Figure()
                fig_play = add_cad_walls_to_fig(fig_play)

                fig_play.add_trace(go.Scatter(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    mode="markers+text",
                    marker=dict(size=12, color="#FF5722"),
                    text=curr_frame_df[id_col].astype(str),
                    textposition="top center",
                    name="Pedestrians"
                ))
                fig_play.update_layout(
                    template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1)
                )
                st.plotly_chart(fig_play, use_container_width=True)

            with col_fb2:
                st.markdown(f"**Instant Density Heatmap (Frame #{selected_f})**")
                fig_f_hm = go.Figure()
                fig_f_hm = add_cad_walls_to_fig(fig_f_hm, wall_color="#FFFFFF", width=2)

                fig_f_hm.add_trace(go.Histogram2dContour(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    colorscale="Jet",
                    showscale=True
                ))
                fig_f_hm.update_layout(
                    template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1)
                )
                st.plotly_chart(fig_f_hm, use_container_width=True)

            st.markdown("---")

            st.markdown("### 3. Aggregated Crowd Metrics (Entire Video)")

            m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
                "📊 Crowd Volume", "🔥 Density Heatmap", "⚡ Speed Distribution", "🧭 Directional Flow"
            ])

            with m_tab1:
                st.markdown("#### Cumulative Occupancy Heatmap")
                fig_vol = go.Figure()
                fig_vol = add_cad_walls_to_fig(fig_vol, wall_color="#FFFFFF", width=2)
                fig_vol.add_trace(go.Histogram2dContour(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Viridis", showscale=True
                ))
                fig_vol.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_vol, use_container_width=True)

            with m_tab2:
                st.markdown("#### Binned Pedestrian Density Grid")
                fig_dens = go.Figure()
                fig_dens = add_cad_walls_to_fig(fig_dens, wall_color="#FFFFFF", width=2)
                fig_dens.add_trace(go.Histogram2d(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Hot", showscale=True, nbinsx=35, nbinsy=35
                ))
                fig_dens.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_dens, use_container_width=True)

            with m_tab3:
                st.markdown("#### Velocity Heatmap")
                fig_spd = px.scatter(
                    df_track, x=x_col, y=y_col, color="speed", color_continuous_scale="Plasma",
                    title="Pedestrian Speed Distribution"
                )
                fig_spd = add_cad_walls_to_fig(fig_spd)
                fig_spd.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_spd, use_container_width=True)

            with m_tab4:
                st.markdown("#### Movement Direction Vectors")
                fig_dir = px.scatter(
                    df_track, x=x_col, y=y_col, color="dx", color_continuous_scale="RdBu",
                    title="Directional Shift Field (dx)"
                )
                fig_dir = add_cad_walls_to_fig(fig_dir)
                fig_dir.update_layout(template="plotly_dark", height=500, xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_dir, use_container_width=True)

            st.markdown("---")
            st.markdown("### 4. Export Aggregated Analytics")

            crowd_metrics_export = {
                "total_frames": int(df_track[frame_col].nunique()),
                "total_unique_pedestrians": int(df_track[id_col].nunique()),
                "average_speed": float(df_track["speed"].mean()),
                "max_speed": float(df_track["speed"].max()),
                "trajectories": df_track[[frame_col, id_col, x_col, y_col, "speed"]].to_dict(orient="records")
            }

            st.download_button(
                label="💾 Export Analytics JSON",
                data=json.dumps(crowd_metrics_export, indent=2),
                file_name="crowd_analytics.json",
                mime="application/json",
                use_container_width=True,
            )

        else:
            st.error(f"⚠️ Could not resolve coordinate columns in dataset. Found columns: {list(df_track.columns)}")

    else:
        st.info("💡 Upload a JSON/CSV tracking file above or run tracking in Step 3.3 to view movement playback and heatmaps.")