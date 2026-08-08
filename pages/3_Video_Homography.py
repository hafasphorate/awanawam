# pages/2_Video_Homography.py
import json
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.vga_engine import process_cad_file
from utils.tracking_engine import extract_frame_from_video
from views.tracking_view import render_tracking_view

st.set_page_config(page_title="Module 2: Video Homography & Tracking", layout="wide")

st.title("📹 Module 2: Video Homography & Region Selection")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "dxf_walls" not in st.session_state:
    st.session_state.dxf_walls = []
if "wall_lines" not in st.session_state:
    st.session_state.wall_lines = []
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

# Exclusion Masking State
if "exclusion_masks" not in st.session_state:
    st.session_state.exclusion_masks = []
if "active_mask_pts" not in st.session_state:
    st.session_state.active_mask_pts = []
if "mask_click_sig" not in st.session_state:
    st.session_state.mask_click_sig = None

# Navigation Tabs
tab_import, tab_region, tab_tracking, tab_playback = st.tabs([
    "📂 2.1 Import CAD / Session & Video",
    "📐 2.2 Define ROI & Video Masking",
    "🔥 2.3 Occupancy Analytics",
    "🎬 2.4 2D Playback & Crowd Heatmaps",
])

# Helper function to unpack wall geometry cleanly
def extract_walls_from_json(data):
    walls = []
    wall_source = data.get("dxf_walls", data.get("wall_lines", data.get("walls", [])))
    for w in wall_source:
        if isinstance(w, dict) and "x" in w and "y" in w:
            walls.append(w)
        elif isinstance(w, (list, tuple)) and len(w) >= 2:
            walls.append({"x": [w[0][0], w[1][0]], "y": [w[0][1], w[1][1]]})
    return walls

# ==========================================
# TAB 1: FILE & VIDEO IMPORT
# ==========================================
with tab_import:
    st.subheader("Step 2.1: Load CAD (DXF/DWG) or JSON Config & Surveillance Video")

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

                # 1. Load VGA Grid
                vga_data = data.get("vga_results", data.get("vga_grid", data.get("vga_floorplan_nodes", [])))
                if vga_data:
                    st.session_state.vga_grid_df = pd.DataFrame(vga_data)
                    st.session_state["vga_df"] = st.session_state.vga_grid_df
                    st.success(f"✅ Loaded VGA Grid ({len(st.session_state.vga_grid_df)} nodes)")

                # 2. Fix 1: Load CAD Walls from JSON Export
                walls = extract_walls_from_json(data)
                if walls:
                    st.session_state.dxf_walls = walls
                    st.session_state.wall_lines = walls
                    st.success(f"✅ Loaded {len(walls)} CAD wall boundaries")

                # 3. Load ROI Polygon / Corners
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
                st.error(f"Error parsing JSON: {e}")

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
                    st.session_state.wall_lines = wall_lines
                    st.success(f"✅ Successfully parsed CAD! {len(wall_lines)} wall boundary lines ready.")
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
# TAB 2: REGION SELECTION & POINT MASKING
# ==========================================
with tab_region:
    st.subheader("Step 2.2: Video Masking & ROI Corner Calibration")

    st.markdown("### 🚫 1. Video Polygon Masking (Exclusion Zones)")
    st.info("💡 Click points directly on the frame to draw a polygon surrounding regions to **EXCLUDE from human tracking**.")

    if "uploaded_video_file" in st.session_state and st.session_state.uploaded_video_file is not None:
        col_m_slider, col_m_btns = st.columns([2.5, 1.5])

        with col_m_slider:
            frame_idx = st.slider(
                "Calibration Video Frame",
                min_value=0,
                max_value=1000,
                value=st.session_state.selected_frame_idx,
                step=5,
            )
            st.session_state.selected_frame_idx = frame_idx

        with col_m_btns:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            c_btn1, c_btn2, c_btn3 = st.columns(3)
            with c_btn1:
                if st.button("💾 Lock Mask", use_container_width=True):
                    if len(st.session_state.active_mask_pts) >= 3:
                        st.session_state.exclusion_masks.append(list(st.session_state.active_mask_pts))
                        st.session_state.active_mask_pts = []
                        st.success("Mask Saved!")
                        st.rerun()
                    else:
                        st.warning("Needs ≥3 points.")
            with c_btn2:
                if st.button("🗑️ Clear Active", use_container_width=True):
                    st.session_state.active_mask_pts = []
                    st.rerun()
            with c_btn3:
                if st.button("🔥 Reset All", use_container_width=True):
                    st.session_state.exclusion_masks = []
                    st.session_state.active_mask_pts = []
                    st.rerun()

        raw_frame_rgb = extract_frame_from_video(st.session_state.uploaded_video_file, frame_number=frame_idx)
        if raw_frame_rgb is not None:
            fig_img = px.imshow(raw_frame_rgb)
            img_h, img_w = raw_frame_rgb.shape[0], raw_frame_rgb.shape[1]

            # Fix 2: Render completed polygon exclusion masks
            shapes_list = []
            for mask in st.session_state.exclusion_masks:
                if len(mask) >= 3:
                    path_str = f"M {mask[0][0]},{mask[0][1]} " + " ".join([f"L {p[0]},{p[1]}" for p in mask[1:]]) + " Z"
                    shapes_list.append(dict(
                        type="path",
                        path=path_str,
                        fillcolor="rgba(255, 0, 0, 0.45)",
                        line=dict(color="#FF0000", width=3),
                    ))

            # Fix 2: Active drawing overlay
            if len(st.session_state.active_mask_pts) > 0:
                amx = [p[0] for p in st.session_state.active_mask_pts]
                amy = [p[1] for p in st.session_state.active_mask_pts]
                amx_line = amx + ([amx[0]] if len(amx) > 1 else [])
                amy_line = amy + ([amy[0]] if len(amy) > 1 else [])

                fig_img.add_trace(go.Scatter(
                    x=amx_line, y=amy_line,
                    mode="markers+lines",
                    marker=dict(color="#FFFF00", size=10, symbol="circle"),
                    line=dict(color="#FFFF00", width=3, dash="dash"),
                    name="Active Mask",
                    showlegend=False
                ))

            fig_img.update_layout(
                shapes=shapes_list,
                margin=dict(l=0, r=0, t=10, b=10),
                height=600,
                xaxis=dict(range=[0, img_w], showgrid=False),
                yaxis=dict(range=[img_h, 0], showgrid=False), # Maintain image axis orientation
                clickmode="event+select",
                dragmode="pan",
                uirevision="VIDEO_CANVAS_PRESERVE"
            )

            v_events = st.plotly_chart(
                fig_img,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="video_mask_canvas"
            )

            if v_events and "selection" in v_events:
                v_pts = v_events["selection"].get("points", [])
                if v_pts:
                    vx, vy = float(v_pts[0]["x"]), float(v_pts[0]["y"])
                    sig = f"vmask_{vx:.2f}_{vy:.2f}"
                    if sig != st.session_state.mask_click_sig:
                        st.session_state.mask_click_sig = sig
                        st.session_state.active_mask_pts.append([vx, vy])
                        st.rerun()
    else:
        st.warning("⚠️ Please upload a surveillance video in Step 2.1 to enable interactive video masking.")

    st.markdown("---")

    # --- BOTTOM SECTION: CORNER COORDINATES & FLOORPLAN MAP ---
    st.markdown("### 📐 2. Camera ROI Corner Mapping")

    col_controls, col_plot = st.columns([1.2, 2.8])

    with col_controls:
        st.markdown("#### Corner Point Settings")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔴 Reset ROI", use_container_width=True):
                st.session_state.four_corners = []
                st.session_state.selected_polygon_pts = []
                st.session_state.editing_point_idx = None
                st.session_state.processed_click_sig = None
                st.rerun()

        with col_btn2:
            if st.session_state.editing_point_idx is not None:
                if st.button("❌ Cancel Edit", use_container_width=True):
                    st.session_state.editing_point_idx = None
                    st.rerun()

        num_pts = len(st.session_state.four_corners)
        if st.session_state.editing_point_idx is not None:
            st.warning(f"🎯 **Editing P{st.session_state.editing_point_idx + 1}:** Click map to re-position.")
        elif num_pts < 4:
            st.info(f"⚠️ Selected **{num_pts}/4** corners. Click floorplan map.")
        else:
            st.success("✅ All 4 ROI Corners Configured!")

        corner_labels = ["P1 (Top-Left)", "P2 (Top-Right)", "P3 (Bottom-Right)", "P4 (Bottom-Left)"]
        if len(st.session_state.four_corners) > 0:
            st.markdown("##### Current Corners")
            for idx, pt in enumerate(st.session_state.four_corners[:4]):
                c_lbl = corner_labels[idx] if idx < 4 else f"P{idx+1}"
                col_info, col_act = st.columns([2.2, 1])
                with col_info:
                    st.markdown(f"**{c_lbl}**: `({round(pt[0], 2)}, {round(pt[1], 2)})`")
                with col_act:
                    is_editing = (st.session_state.editing_point_idx == idx)
                    btn_label = "🎯 Target" if is_editing else "✏️ Edit"
                    if st.button(btn_label, key=f"edit_btn_{idx}", use_container_width=True):
                        st.session_state.editing_point_idx = idx
                        st.session_state.processed_click_sig = None
                        st.rerun()

    with col_plot:
        st.markdown("#### Interactive Floorplan Map")

        fig = go.Figure()

        # Fix 3: Robust CAD Wall Rendering supporting all data formats
        wall_x, wall_y = [], []
        walls_data = st.session_state.get("dxf_walls") or st.session_state.get("wall_lines", [])

        for line in walls_data:
            if hasattr(line, "xy"):  # Shapely LineString
                x, y = line.xy
                wall_x.extend([x[0], x[1], None])
                wall_y.extend([y[0], y[1], None])
            elif isinstance(line, dict) and "x" in line and "y" in line:  # Dict representation
                wall_x.extend([line["x"][0], line["x"][1], None])
                wall_y.extend([line["y"][0], line["y"][1], None])
            elif isinstance(line, (list, tuple)):  # Coordinate tuples
                for i in range(len(line) - 1):
                    wall_x.extend([line[i][0], line[i+1][0], None])
                    wall_y.extend([line[i][1], line[i+1][1], None])

        if wall_x:
            fig.add_trace(
                go.Scatter(
                    x=wall_x, y=wall_y,
                    mode="lines",
                    line=dict(color="#00ADB5", width=1.5),
                    name="CAD Walls",
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # Plot selected ROI corners
        pts = st.session_state.four_corners
        if len(pts) > 0:
            px_pts = [p[0] for p in pts]
            py_pts = [p[1] for p in pts]

            if len(pts) == 4:
                px_closed = px_pts + [px_pts[0]]
                py_closed = py_pts + [py_pts[0]]
                fig.add_trace(
                    go.Scatter(
                        x=px_closed, y=py_closed,
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
                    x=px_pts, y=py_pts,
                    mode="markers+text",
                    marker=dict(size=14, color=marker_colors, symbol="circle"),
                    text=[f"P{i+1}" for i in range(len(pts))],
                    textposition="top right",
                    textfont=dict(size=14, color="#FFFFFF"),
                    name="Selected Corners",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis=dict(title="X Coordinate", scaleanchor="y", scaleratio=1, showgrid=True),
            yaxis=dict(title="Y Coordinate", showgrid=True),
            margin=dict(l=10, r=10, t=30, b=10),
            clickmode="event+select",
            dragmode="pan",
            hovermode="closest",
            uirevision="PERMANENT_CANVAS_LOCK",
        )

        chart_events = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="roi_floorplan_canvas",
        )

        if chart_events and "selection" in chart_events:
            event_pts = chart_events["selection"].get("points", [])
            if event_pts:
                click_x = float(event_pts[0]["x"])
                click_y = float(event_pts[0]["y"])
                click_sig = f"{click_x:.4f}_{click_y:.4f}_{st.session_state.editing_point_idx}"

                if click_sig != st.session_state.processed_click_sig:
                    st.session_state.processed_click_sig = click_sig

                    if st.session_state.editing_point_idx is not None:
                        target_idx = st.session_state.editing_point_idx
                        st.session_state.four_corners[target_idx] = [click_x, click_y]
                        st.session_state.editing_point_idx = None
                        st.session_state.selected_polygon_pts = [
                            {"X (m)": p[0], "Y (m)": p[1]} for p in st.session_state.four_corners
                        ]
                        st.rerun()

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

# ==========================================
# TAB 4: 2D PLAYBACK & CROWD HEATMAPS
# ==========================================
with tab_playback:
    st.subheader("Step 2.4: 2D Playback & Crowd Trajectory Analytics")

    st.markdown("### 1. Import Tracking Dataset")
    col_up1, col_up2 = st.columns(2)

    # Fix 4: Universal JSON tracking parser
    def parse_tracking_json(raw_json):
        if isinstance(raw_json, list):
            return pd.DataFrame(raw_json)

        if isinstance(raw_json, dict):
            # Inspect common tracking payload keys
            for key in ["tracking_points", "tracking_results", "pedestrian_trajectories", "trajectories", "tracking_data"]:
                if key in raw_json and isinstance(raw_json[key], list) and len(raw_json[key]) > 0:
                    return pd.DataFrame(raw_json[key])

        return pd.json_normalize(raw_json)

    with col_up1:
        uploaded_tb_json = st.file_uploader("Upload JSON Export (from Step 2.3)", type=["json"], key="tb_json_up")
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

    # DISPLAY HEATMAPS AND MOTION PLAYBACK
    df_track = st.session_state.get("tracking_results_df")

    if df_track is not None and not df_track.empty:
        df_track.columns = [str(c).lower().strip() for c in df_track.columns]

        # Fix 4: Map coordinate and frame keys flexibly across various schema versions
        frame_col = next((c for c in ["frame", "frame_idx", "frame_number", "timestamp"] if c in df_track.columns), None)
        x_col = next((c for c in ["x", "x (m)", "x_m", "pos_x", "x_canvas"] if c in df_track.columns), None)
        y_col = next((c for c in ["y", "y (m)", "y_m", "pos_y", "y_canvas"] if c in df_track.columns), None)
        id_col = next((c for c in ["track_id", "id", "person_id"] if c in df_track.columns), "track_id")

        if x_col and y_col:
            # Ensure frame column exists
            if not frame_col:
                df_track["frame"] = 0
                frame_col = "frame"

            if id_col not in df_track.columns:
                df_track[id_col] = 1

            if "speed" not in df_track.columns:
                df_track = df_track.sort_values(by=[id_col, frame_col])
                df_track["dx"] = df_track.groupby(id_col)[x_col].diff().fillna(0)
                df_track["dy"] = df_track.groupby(id_col)[y_col].diff().fillna(0)
                df_track["speed"] = np.sqrt(df_track["dx"]**2 + df_track["dy"]**2)

            # --- SECTION 2: FRAME PLAYBACK ---
            st.markdown("### 2. Motion Playback & Frame Analytics")
            frames_available = sorted(df_track[frame_col].unique())
            selected_f = st.slider("Select Frame for Instant Inspection", min_value=int(min(frames_available)), max_value=int(max(frames_available)), value=int(min(frames_available)))

            curr_frame_df = df_track[df_track[frame_col] == selected_f]

            col_fb1, col_fb2 = st.columns(2)

            with col_fb1:
                st.markdown(f"**Pedestrian Plan View (Frame #{selected_f})**")
                fig_play = go.Figure()

                # Add CAD walls as background trace
                walls_data = st.session_state.get("dxf_walls") or st.session_state.get("wall_lines", [])
                for line in walls_data:
                    if hasattr(line, "xy"):
                        wx, wy = line.xy
                        fig_play.add_trace(go.Scatter(x=list(wx), y=list(wy), mode="lines", line=dict(color="#00ADB5", width=1), showlegend=False))
                    elif isinstance(line, dict) and "x" in line:
                        fig_play.add_trace(go.Scatter(x=line["x"], y=line["y"], mode="lines", line=dict(color="#00ADB5", width=1), showlegend=False))

                fig_play.add_trace(go.Scatter(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    mode="markers+text",
                    marker=dict(size=12, color="#FF5722"),
                    text=curr_frame_df[id_col].astype(str),
                    textposition="top center",
                    name="Pedestrians"
                ))
                fig_play.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_play, use_container_width=True)

            with col_fb2:
                st.markdown(f"**Instant Density Heatmap (Frame #{selected_f})**")
                fig_f_hm = go.Figure(go.Histogram2dContour(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    colorscale="Jet",
                    showscale=True
                ))
                fig_f_hm.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_f_hm, use_container_width=True)

            st.markdown("---")

            # --- SECTION 3: FULL HEATMAP ANALYTICS ---
            st.markdown("### 3. Aggregated Crowd Metrics (Entire Video)")

            m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["📊 Crowd Volume", "🔥 Density Heatmap", "⚡ Speed Distribution", "🧭 Directional Flow"])

            with m_tab1:
                st.markdown("#### Cumulative Occupancy Heatmap")
                fig_vol = go.Figure(go.Histogram2dContour(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Viridis", showscale=True
                ))
                fig_vol.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_vol, use_container_width=True)

            with m_tab2:
                st.markdown("#### Binned Pedestrian Density Grid")
                fig_dens = go.Figure(go.Histogram2d(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Hot", showscale=True, nbinsx=35, nbinsy=35
                ))
                fig_dens.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_dens, use_container_width=True)

            with m_tab3:
                st.markdown("#### Velocity Heatmap")
                fig_spd = px.scatter(
                    df_track, x=x_col, y=y_col, color="speed", color_continuous_scale="Plasma",
                    title="Pedestrian Speed Distribution"
                )
                fig_spd.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_spd, use_container_width=True)

            with m_tab4:
                st.markdown("#### Movement Direction Vectors")
                fig_dir = px.scatter(
                    df_track, x=x_col, y=y_col, color="dx", color_continuous_scale="Coolwarm",
                    title="Directional Shift Field (dx)"
                )
                fig_dir.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_dir, use_container_width=True)

            # Export Section
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
            st.error(f"⚠️ Could not resolve coordinate columns (`x`, `y`) in dataset. Found columns: {list(df_track.columns)}")

    else:
        st.info("💡 Upload a JSON/CSV tracking file above or run tracking in Step 2.3 to view movement playback and heatmaps.")