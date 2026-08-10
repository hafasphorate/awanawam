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

import math
from scipy.spatial import KDTree

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


def serialize_session_walls():
    """Convert stored wall geometries into JSON-safe coordinate dictionaries."""
    raw_walls = (
        st.session_state.get("wall_lines")
        or st.session_state.get("dxf_walls")
        or st.session_state.get("cad_walls")
        or st.session_state.get("walls")
        or []
    )

    serialized = []
    for line in raw_walls:
        normalized = normalize_line_to_dict(line)
        if normalized:
            serialized.append(normalized)
    return serialized


def serialize_vga_nodes(vga_nodes):
    """Normalize VGA node payloads that may be a DataFrame, list, or dict."""
    if isinstance(vga_nodes, pd.DataFrame):
        return vga_nodes.to_dict(orient="records")
    if isinstance(vga_nodes, list):
        return vga_nodes
    if isinstance(vga_nodes, dict):
        if "nodes" in vga_nodes and isinstance(vga_nodes["nodes"], list):
            return vga_nodes["nodes"]
        if "vga_floorplan_nodes" in vga_nodes and isinstance(vga_nodes["vga_floorplan_nodes"], list):
            return vga_nodes["vga_floorplan_nodes"]
        return [value for value in vga_nodes.values() if isinstance(value, dict)]
    return []


def build_grid_aligned_crowd_vga_export(vga_nodes, df_track, id_col, frame_col, x_col, y_col):
    """Merge VGA node metadata with per-grid crowd metrics keyed by grid_node_idx."""
    vga_rows = []
    if isinstance(vga_nodes, pd.DataFrame):
        vga_nodes = vga_nodes.to_dict(orient="records")
    elif not isinstance(vga_nodes, list):
        vga_nodes = []

    crowd_by_grid = {}
    if "grid_node_idx" in df_track.columns:
        crowd_by_grid = (
            df_track.groupby("grid_node_idx")
            .agg(
                pedestrian_count=(id_col, "count"),
                unique_pedestrians=(id_col, "nunique"),
                avg_speed=("speed", "mean"),
                mean_heading_deg=("dir_deg_north", "mean"),
                crowd_volume=(id_col, "count"),
                crowd_density=(id_col, "count"),
            )
            .to_dict("index")
        )

    for idx, node in enumerate(vga_nodes):
        row = dict(node) if isinstance(node, dict) else {"node_id": idx}
        row.setdefault("node_id", idx)
        row.setdefault("grid_node_idx", idx)
        grid_key = row.get("grid_node_idx", row.get("node_id", idx))
        if isinstance(grid_key, float) and np.isfinite(grid_key):
            grid_key = int(grid_key)

        agg = crowd_by_grid.get(grid_key, {})
        row.update(
            {
                "pedestrian_count": int(agg.get("pedestrian_count", row.get("pedestrian_count", 0))),
                "unique_pedestrians": int(agg.get("unique_pedestrians", row.get("unique_pedestrians", 0))),
                "avg_speed": float(agg.get("avg_speed", row.get("avg_speed", 0.0))),
                "mean_heading_deg": float(agg.get("mean_heading_deg", row.get("mean_heading_deg", 0.0))),
                "crowd_volume": int(agg.get("crowd_volume", row.get("crowd_volume", row.get("pedestrian_count", 0)))),
                "crowd_density": float(agg.get("crowd_density", row.get("crowd_density", row.get("pedestrian_count", 0.0)))),
                "volume": float(agg.get("crowd_volume", row.get("crowd_volume", row.get("pedestrian_count", 0.0)))),
                "density": float(agg.get("crowd_density", row.get("crowd_density", row.get("pedestrian_count", 0.0)))),
                "speed": float(agg.get("avg_speed", row.get("speed", row.get("avg_speed", 0.0)))),
                "direction": float(agg.get("mean_heading_deg", row.get("direction", row.get("mean_heading_deg", 0.0)))),
            }
        )
        vga_rows.append(row)

    if not vga_rows and "grid_node_idx" in df_track.columns:
        for grid_idx, group in df_track.groupby("grid_node_idx"):
            vga_rows.append(
                {
                    "node_id": int(grid_idx),
                    "grid_node_idx": int(grid_idx),
                    "pedestrian_count": int(group[id_col].count()),
                    "unique_pedestrians": int(group[id_col].nunique()),
                    "avg_speed": float(group["speed"].mean()) if "speed" in group.columns else 0.0,
                    "mean_heading_deg": float(group["dir_deg_north"].mean()) if "dir_deg_north" in group.columns else 0.0,
                    "crowd_volume": int(group[id_col].count()),
                    "crowd_density": float(group[id_col].count()),
                    "volume": float(group[id_col].count()),
                    "density": float(group[id_col].count()),
                    "speed": float(group["speed"].mean()) if "speed" in group.columns else 0.0,
                    "direction": float(group["dir_deg_north"].mean()) if "dir_deg_north" in group.columns else 0.0,
                }
            )

    return vga_rows


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

import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import cKDTree

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def add_cad_walls_to_fig(fig, line_color="#888888", line_width=1.5):
    """Underlays CAD floorplan wall lines from all possible session state keys."""
    # Attempt to retrieve wall geometries from all common key names
    wall_lines = (
        st.session_state.get("wall_lines")
        or st.session_state.get("cad_walls")
        or st.session_state.get("vga_walls")
        or st.session_state.get("walls")
        or []
    )

    if not wall_lines:
        return fig

    wall_x, wall_y = [], []

    for line in wall_lines:
        try:
            # Case 1: Shapely LineString object
            if hasattr(line, "xy"):
                x, y = line.xy
                wall_x.extend([list(x)[0], list(x)[1], None])
                wall_y.extend([list(y)[0], list(y)[1], None])

            # Case 2: Dict schema like {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
            elif isinstance(line, dict):
                x1 = line.get("x1", line.get("start_x"))
                y1 = line.get("y1", line.get("start_y"))
                x2 = line.get("x2", line.get("end_x"))
                y2 = line.get("y2", line.get("end_y"))
                if None not in (x1, y1, x2, y2):
                    wall_x.extend([x1, x2, None])
                    wall_y.extend([y1, y2, None])

            # Case 3: List/Tuple pair of coordinates [ (x1,y1), (x2,y2) ]
            elif isinstance(line, (list, tuple)) and len(line) == 2:
                p1, p2 = line[0], line[1]
                wall_x.extend([p1[0], p2[0], None])
                wall_y.extend([p1[1], p2[1], None])
        except Exception:
            continue

    if wall_x and wall_y:
        fig.add_trace(
            go.Scatter(
                x=wall_x,
                y=wall_y,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hoverinfo="none",
                showlegend=False,
                name="CAD Floorplan",
            )
        )
    return fig


def calculate_bearing_from_north(dx, dy):
    """Calculates directional angle in degrees relative to North (top = 0 deg).

    Angles increase clockwise: North=0, East=90, South=180, West=270.
    """
    if dx == 0 and dy == 0:
        return 0.0
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    return float(angle_deg % 360)


def map_points_to_grid_nodes(df_track, grid_nodes, x_col, y_col):
    """Maps continuous trajectory points to the nearest VGA grid node."""
    if not grid_nodes or df_track.empty:
        return df_track

    # Extract node positions (handles both list of dicts or DataFrame-like dicts)
    node_coords = []
    for n in grid_nodes:
        nx = n.get("x", n.get("world_x", n.get("pos_x")))
        ny = n.get("y", n.get("world_y", n.get("pos_y")))
        if nx is not None and ny is not None:
            node_coords.append([nx, ny])

    if not node_coords:
        return df_track

    node_coords_arr = np.array(node_coords)
    tree = cKDTree(node_coords_arr)

    track_coords = df_track[[x_col, y_col]].values
    distances, indices = tree.query(track_coords)

    df_track["grid_node_idx"] = indices
    df_track["grid_node_x"] = node_coords_arr[indices, 0]
    df_track["grid_node_y"] = node_coords_arr[indices, 1]

    return df_track


# ==========================================
# TAB 4: 2D PLAYBACK & CROWD HEATMAPS
# ==========================================
with tab_playback:
    st.subheader("Step 2.4: 2D Playback & Crowd Trajectory Analytics")

    st.markdown("### 1. Import Tracking Dataset")
    col_up1, col_up2 = st.columns(2)

    def parse_tracking_json(raw_json):
        # Extract VGA Grid Nodes if included in JSON upload
        if isinstance(raw_json, dict):
            if "vga_floorplan_nodes" in raw_json:
                st.session_state.vga_floorplan_nodes = raw_json[
                    "vga_floorplan_nodes"
                ]
            if "vga_results" in raw_json:
                st.session_state.vga_results = raw_json["vga_results"]
            if "wall_lines" in raw_json:
                st.session_state.wall_lines = raw_json["wall_lines"]

        if isinstance(raw_json, list):
            return pd.DataFrame(raw_json)

        if isinstance(raw_json, dict):
            for key in [
                "tracking_points",
                "tracking_results",
                "pedestrian_trajectories",
                "trajectories",
                "tracking_data",
            ]:
                if (
                    key in raw_json
                    and isinstance(raw_json[key], list)
                    and len(raw_json[key]) > 0
                ):
                    return pd.DataFrame(raw_json[key])

        return pd.json_normalize(raw_json)

    with col_up1:
        uploaded_tb_json = st.file_uploader(
            "Upload JSON Export (from Step 2.3 / Spatial Analysis)",
            type=["json"],
            key="tb_json_up",
        )
        if uploaded_tb_json is not None:
            try:
                raw_json = json.load(uploaded_tb_json)
                df_loaded = parse_tracking_json(raw_json)
                st.session_state.tracking_results_df = df_loaded
                st.success(
                    f"✅ Successfully imported {len(df_loaded)} tracking records!"
                )
            except Exception as e:
                st.error(f"Error reading JSON: {e}")

    with col_up2:
        uploaded_tb_csv = st.file_uploader(
            "Upload CSV Tracking Export", type=["csv"], key="tb_csv_up"
        )
        if uploaded_tb_csv is not None:
            try:
                st.session_state.tracking_results_df = pd.read_csv(
                    uploaded_tb_csv
                )
                st.success("✅ Successfully imported CSV tracking records!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Wall upload status indicator
    has_walls = any(
        st.session_state.get(k)
        for k in ["wall_lines", "cad_walls", "vga_walls", "walls"]
    )
    if has_walls:
        st.caption("🟢 CAD Floorplan geometry loaded and active for overlays.")
    else:
        st.caption(
            "🟡 No CAD floorplan geometry found in session. (Upload CAD floorplan in Step 2.1 to display walls)."
        )

    st.markdown("---")

    df_track = st.session_state.get("tracking_results_df")

    if df_track is not None and not df_track.empty:
        df_track.columns = [str(c).lower().strip() for c in df_track.columns]

        frame_col = next(
            (
                c
                for c in ["frame_idx", "frame", "frame_number", "timestamp"]
                if c in df_track.columns
            ),
            None,
        )
        x_col = next(
            (
                c
                for c in [
                    "world_x",
                    "x",
                    "x (m)",
                    "x_m",
                    "pos_x",
                    "x_canvas",
                    "img_x",
                ]
                if c in df_track.columns
            ),
            None,
        )
        y_col = next(
            (
                c
                for c in [
                    "world_y",
                    "y",
                    "y (m)",
                    "y_m",
                    "pos_y",
                    "y_canvas",
                    "img_y",
                ]
                if c in df_track.columns
            ),
            None,
        )
        id_col = next(
            (c for c in ["track_id", "id", "person_id"] if c in df_track.columns),
            "track_id",
        )

        if x_col and y_col:
            if not frame_col:
                df_track["frame_idx"] = 0
                frame_col = "frame_idx"

            if id_col not in df_track.columns:
                df_track[id_col] = 1

            # --- Calculate Motion Metrics ---
            df_track = df_track.sort_values(by=[id_col, frame_col])
            df_track["dx"] = df_track.groupby(id_col)[x_col].diff().fillna(0)
            df_track["dy"] = df_track.groupby(id_col)[y_col].diff().fillna(0)
            df_track["speed"] = np.sqrt(df_track["dx"] ** 2 + df_track["dy"] ** 2)

            # Compass Bearing (0° North)
            df_track["dir_deg_north"] = [
                calculate_bearing_from_north(dx, dy)
                for dx, dy in zip(df_track["dx"], df_track["dy"])
            ]

            # --- Grid Alignment (VGA Node Mapping) ---
            vga_nodes = st.session_state.get("vga_floorplan_nodes", [])
            if not vga_nodes and "vga_results" in st.session_state:
                # Fallback to extracting nodes from vga_results if available
                vga_res = st.session_state.vga_results
                if isinstance(vga_res, list):
                    vga_nodes = vga_res
                elif isinstance(vga_res, dict) and "nodes" in vga_res:
                    vga_nodes = vga_res["nodes"]

            if vga_nodes:
                df_track = map_points_to_grid_nodes(
                    df_track, vga_nodes, x_col, y_col
                )
                st.info(
                    f"🔗 Matched movement tracking data to {len(vga_nodes)} floorplan grid nodes."
                )

            # --- 2. Motion Playback ---
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
                st.markdown(
                    f"**Pedestrian Plan View (Frame #{selected_f})**"
                )
                fig_play = go.Figure()
                fig_play = add_cad_walls_to_fig(fig_play)

                fig_play.add_trace(
                    go.Scatter(
                        x=curr_frame_df[x_col],
                        y=curr_frame_df[y_col],
                        mode="markers+text",
                        marker=dict(size=12, color="#FF5722"),
                        text=curr_frame_df[id_col].astype(str),
                        textposition="top center",
                        name="Pedestrians",
                    )
                )
                fig_play.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )
                st.plotly_chart(fig_play, use_container_width=True)

            with col_fb2:
                st.markdown(
                    f"**Instant Density Heatmap (Frame #{selected_f})**"
                )
                fig_f_hm = go.Figure()
                fig_f_hm = add_cad_walls_to_fig(
                    fig_f_hm, line_color="#FFFFFF", line_width=1.5
                )

                fig_f_hm.add_trace(
                    go.Histogram2dContour(
                        x=curr_frame_df[x_col],
                        y=curr_frame_df[y_col],
                        colorscale="Jet",
                        showscale=True,
                    )
                )
                fig_f_hm.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )
                st.plotly_chart(fig_f_hm, use_container_width=True)

            st.markdown("---")

            # --- 3. Aggregated Metrics ---
            st.markdown("### 3. Aggregated Crowd Metrics (Entire Video)")

            m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(
                [
                    "📊 Crowd Volume",
                    "🔥 Density Heatmap",
                    "⚡ Speed Distribution",
                    "🧭 Directional Flow",
                ]
            )

            with m_tab1:
                st.markdown("#### Cumulative Occupancy Heatmap")
                fig_vol = go.Figure()
                fig_vol = add_cad_walls_to_fig(
                    fig_vol, line_color="#FFFFFF", line_width=1.5
                )
                fig_vol.add_trace(
                    go.Histogram2dContour(
                        x=df_track[x_col],
                        y=df_track[y_col],
                        colorscale="Viridis",
                        showscale=True,
                    )
                )
                fig_vol.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            with m_tab2:
                st.markdown("#### Binned Pedestrian Density Grid")
                fig_dens = go.Figure()
                fig_dens = add_cad_walls_to_fig(
                    fig_dens, line_color="#FFFFFF", line_width=1.5
                )

                plot_x = (
                    df_track["grid_node_x"]
                    if "grid_node_x" in df_track
                    else df_track[x_col]
                )
                plot_y = (
                    df_track["grid_node_y"]
                    if "grid_node_y" in df_track
                    else df_track[y_col]
                )

                fig_dens.add_trace(
                    go.Histogram2d(
                        x=plot_x,
                        y=plot_y,
                        colorscale="Hot",
                        showscale=True,
                        nbinsx=35,
                        nbinsy=35,
                    )
                )
                fig_dens.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )
                st.plotly_chart(fig_dens, use_container_width=True)

            with m_tab3:
                st.markdown("#### Velocity Heatmap")
                fig_spd = px.scatter(
                    df_track,
                    x=x_col,
                    y=y_col,
                    color="speed",
                    color_continuous_scale="Plasma",
                    title="Pedestrian Speed Distribution",
                )
                fig_spd = add_cad_walls_to_fig(fig_spd)
                fig_spd.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )
                st.plotly_chart(fig_spd, use_container_width=True)

            with m_tab4:
                st.markdown("#### Directional Shift Field (Degrees from North)")
                fig_dir = px.scatter(
                    df_track,
                    x=x_col,
                    y=y_col,
                    color="dir_deg_north",
                    color_continuous_scale="twilight",
                    range_color=[0, 360],
                    labels={"dir_deg_north": "Heading (° North)"},
                    title="Movement Direction Relative to North (0° = Up/North)",
                )
                fig_dir = add_cad_walls_to_fig(fig_dir)
                fig_dir.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )
                st.plotly_chart(fig_dir, use_container_width=True)

            st.markdown("---")
            st.markdown(
                "### 4. Export Combined Correlation Dataset (VGA + Crowd)"
            )

            # --- Retrieve Full VGA Analysis Metrics ---
            raw_vga_analysis = st.session_state.get("vga_results", {})

            # Build a single grid-aligned dataset where each node contains both VGA and crowd metrics.
            integrated_correlation_nodes = build_grid_aligned_crowd_vga_export(
                vga_nodes, df_track, id_col, frame_col, x_col, y_col
            )

            wall_lines_serialized = serialize_session_walls()
            vga_nodes_export = integrated_correlation_nodes if integrated_correlation_nodes else serialize_vga_nodes(vga_nodes)
            if not vga_nodes_export and isinstance(raw_vga_analysis, list):
                vga_nodes_export = raw_vga_analysis
            if not vga_nodes_export:
                vga_grid_df = st.session_state.get("vga_grid_df")
                if isinstance(vga_grid_df, pd.DataFrame):
                    vga_nodes_export = vga_grid_df.to_dict(orient="records")

            export_homography = st.session_state.get("homography_matrix")
            if isinstance(export_homography, np.ndarray):
                export_homography = export_homography.tolist()

            crowd_vga_export = {
                "metadata": {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "four_corners_roi": st.session_state.get("four_corners", []),
                    "selected_polygon_pts": st.session_state.get("selected_polygon_pts", []),
                    "frame_column": frame_col,
                    "track_id_column": id_col,
                    "x_column": x_col,
                    "y_column": y_col,
                },
                "summary": {
                    "total_frames": int(df_track[frame_col].nunique()),
                    "total_unique_pedestrians": int(
                        df_track[id_col].nunique()
                    ),
                    "average_speed": float(df_track["speed"].mean()),
                    "max_speed": float(df_track["speed"].max()),
                    "total_grid_nodes": len(integrated_correlation_nodes),
                },
                "floorplan": {
                    "wall_lines": wall_lines_serialized,
                    "cad_walls": wall_lines_serialized,
                    "polygon_points": st.session_state.get("selected_polygon_pts", []),
                    "homography_matrix": export_homography,
                    "exclusion_masks": st.session_state.get("exclusion_masks", []),
                },
                "wall_lines": wall_lines_serialized,
                "cad_walls": wall_lines_serialized,
                "vga_floorplan_nodes": vga_nodes_export,
                "vga_results": raw_vga_analysis,
                "vga_grid": raw_vga_analysis,
                "vga_global_results": raw_vga_analysis,
                "grid_nodes_correlation_data": integrated_correlation_nodes,
                "crowd_metrics_by_grid": integrated_correlation_nodes,
                "trajectories": df_track[
                    [
                        frame_col,
                        id_col,
                        x_col,
                        y_col,
                        "speed",
                        "dir_deg_north",
                    ]
                    + (["grid_node_idx"] if "grid_node_idx" in df_track else [])
                ].to_dict(orient="records"),
            }

            st.download_button(
                label="💾 Export Integrated VGA & Crowd Correlation Dataset (JSON)",
                data=json.dumps(crowd_vga_export, indent=2),
                file_name="integrated_vga_crowd_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

            # Quick preview table for correlation analysis
            if integrated_correlation_nodes:
                st.markdown("#### Preview Node Correlation Data")
                df_corr_preview = pd.DataFrame(integrated_correlation_nodes)
                st.dataframe(df_corr_preview.head(10), use_container_width=True)

        else:
            st.error(
                f"⚠️ Could not resolve coordinate columns in dataset. Found columns: {list(df_track.columns)}"
            )

    else:
        st.info(
            "💡 Upload a JSON/CSV tracking file above or run tracking in Step 2.3 to view movement playback and heatmaps."
        )