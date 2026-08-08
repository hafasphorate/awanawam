# pages/2_Video_Homography.py
import json
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import LineString, Polygon
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

# Exclusion Masking State (Interactive Clicking)
if "exclusion_masks" not in st.session_state:
    st.session_state.exclusion_masks = []  # Completed masks
if "active_mask_pts" not in st.session_state:
    st.session_state.active_mask_pts = []  # In-progress mask points
if "mask_click_sig" not in st.session_state:
    st.session_state.mask_click_sig = None

# Navigation Tabs
tab_import, tab_region, tab_tracking, tab_playback = st.tabs([
    "📂 2.1 Import CAD / Session & Video",
    "📐 2.2 Define ROI & Video Masking",
    "🔥 2.3 Occupancy Analytics",
    "🎬 2.4 2D Playback & Crowd Heatmaps",
])

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

                vga_data = data.get("vga_results", data.get("vga_grid", []))
                if vga_data:
                    st.session_state.vga_grid_df = pd.DataFrame(vga_data)
                    st.session_state["vga_df"] = st.session_state.vga_grid_df
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
                        st.success(f"✅ Loaded {len(formatted_pts)} ROI corner points")

                if "homography_matrix" in data and data["homography_matrix"]:
                    st.session_state.homography_matrix = np.array(data["homography_matrix"])

                if "exclusion_masks" in data and data["exclusion_masks"]:
                    st.session_state.exclusion_masks = data["exclusion_masks"]

                tracking_data = data.get("tracking_results", data.get("tracking_data", None))
                if tracking_data is not None:
                    st.session_state.tracking_results_df = pd.DataFrame(tracking_data)

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
    st.subheader("Step 2.2: Define 4 ROI Camera Corners & Click Video Masking")

    col_controls, col_video_preview, col_plot = st.columns([1.1, 1.8, 2.2])

    # --- LEFT COLUMN: CONTROLS & ROI CORNERS ---
    with col_controls:
        st.markdown("#### Corner Coordinates")

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
            edit_num = st.session_state.editing_point_idx + 1
            st.warning(f"🎯 **Editing P{edit_num}:** Click map to update position.")
        elif num_pts < 4:
            st.info(f"⚠️ Selected **{num_pts}/4** corners. Click map.")
        else:
            st.success("✅ All 4 Corners Selected!")

        corner_labels = ["P1 (Top-Left)", "P2 (Top-Right)", "P3 (Bottom-Right)", "P4 (Bottom-Left)"]

        if len(st.session_state.four_corners) > 0:
            st.markdown("##### Selected Points")
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

        st.markdown("---")
        st.markdown("#### Masking Controls")
        if st.button("💾 Save Active Mask Zone", use_container_width=True):
            if len(st.session_state.active_mask_pts) >= 3:
                st.session_state.exclusion_masks.append(list(st.session_state.active_mask_pts))
                st.session_state.active_mask_pts = []
                st.success("Mask Saved!")
                st.rerun()
            else:
                st.warning("Mask requires at least 3 points.")

        if st.button("🗑️ Clear Active Mask", use_container_width=True):
            st.session_state.active_mask_pts = []
            st.rerun()

        if st.button("🔥 Clear All Mask Zones", use_container_width=True):
            st.session_state.exclusion_masks = []
            st.session_state.active_mask_pts = []
            st.rerun()

    # --- MIDDLE COLUMN: INTERACTIVE VIDEO FRAME MASKING ---
    with col_video_preview:
        st.markdown("#### Video Calibration Frame & Click Masking")
        if "uploaded_video_file" in st.session_state and st.session_state.uploaded_video_file is not None:
            frame_idx = st.slider(
                "Preview Frame Index",
                min_value=0,
                max_value=1000,
                value=st.session_state.selected_frame_idx,
                step=5,
            )
            st.session_state.selected_frame_idx = frame_idx

            raw_frame_rgb = extract_frame_from_video(st.session_state.uploaded_video_file, frame_number=frame_idx)
            if raw_frame_rgb is not None:
                fig_img = px.imshow(raw_frame_rgb)

                # Overlay Saved Masks
                for m_idx, mask in enumerate(st.session_state.exclusion_masks):
                    mx = [p[0] for p in mask] + [mask[0][0]]
                    my = [p[1] for p in mask] + [mask[0][1]]
                    fig_img.add_trace(go.Scatter(x=mx, y=my, mode="lines", fill="toself", fillcolor="rgba(255, 0, 0, 0.4)", line=dict(color="red"), name=f"Mask {m_idx+1}"))

                # Overlay Active Masking Points
                if len(st.session_state.active_mask_pts) > 0:
                    amx = [p[0] for p in st.session_state.active_mask_pts]
                    amy = [p[1] for p in st.session_state.active_mask_pts]
                    fig_img.add_trace(go.Scatter(x=amx, y=amy, mode="markers+lines", marker=dict(color="yellow", size=10), line=dict(color="yellow", dash="dash"), name="Active Mask Points"))

                fig_img.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=450,
                    clickmode="event+select",
                    dragmode="drawclosedpath",
                    uirevision="VIDEO_LOCKED",
                )

                v_events = st.plotly_chart(fig_img, use_container_width=True, on_select="rerun", selection_mode="points", key="video_mask_canvas")

                # Handle clicks on video frame for polygon masking
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
            st.warning("⚠️ Upload video in Tab 2.1 to enable interactive point masking.")

    # --- RIGHT COLUMN: MAP CANVAS (FIXED ZOOM RESET) ---
    with col_plot:
        st.markdown("#### Interactive Floorplan Map")

        fig = go.Figure()

        wall_x, wall_y = [], []
        dxf_walls = st.session_state.get("dxf_walls") or st.session_state.get("wall_lines", [])
        for line in dxf_walls:
            if hasattr(line, "xy"):
                x, y = line.xy
                wall_x.extend([x[0], x[1], None])
                wall_y.extend([y[0], y[1], None])
            elif isinstance(line, (list, tuple)):
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

        pts = st.session_state.four_corners
        if len(pts) > 0:
            px_pts = [p[0] for p in pts]
            py_pts = [p[1] for p in pts]

            if len(pts) == 4:
                px_closed = px_pts + [px_pts[0]]
                py_closed = py_pts + [py_pts[0]]
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
                    x=px_pts,
                    y=py_pts,
                    mode="markers+text",
                    marker=dict(size=14, color=marker_colors, symbol="circle"),
                    text=[f"P{i+1}" for i in range(len(pts))],
                    textposition="top right",
                    textfont=dict(size=14, color="#FFFFFF"),
                    name="Selected Corners",
                )
            )

        # STRICT ZOOM LOCK: uirevision constant + NO RANGE OVERWRITES
        fig.update_layout(
            template="plotly_dark",
            height=600,
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

        # Handle Plotly Clicks
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

    # 1. IMPORT DATASET DIRECTLY IN 2.4
    st.markdown("### 1. Import Tracking Dataset")
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        uploaded_tb_json = st.file_uploader("Upload JSON Export (from Step 2.3)", type=["json"], key="tb_json_up")
        if uploaded_tb_json is not None:
            try:
                raw_json = json.load(uploaded_tb_json)
                tb_data = raw_json.get("tracking_results", raw_json.get("tracking_data", raw_json))
                st.session_state.tracking_results_df = pd.DataFrame(tb_data)
                st.success("✅ Imported JSON tracking records!")
            except Exception as e:
                st.error(f"Error reading JSON: {e}")

    with col_up2:
        uploaded_tb_csv = st.file_uploader("Upload CSV Tracking Export", type=["csv"], key="tb_csv_up")
        if uploaded_tb_csv is not None:
            try:
                st.session_state.tracking_results_df = pd.read_csv(uploaded_tb_csv)
                st.success("✅ Imported CSV tracking records!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")

    # DISPLAY PLAYBACK AND HEATMAPS IF DATA AVAILABLE
    df_track = st.session_state.get("tracking_results_df")

    if df_track is not None and not df_track.empty:
        # Standardize column naming
        df_track.columns = [c.lower().strip() for c in df_track.columns]

        # Resolve column keys
        frame_col = next((c for c in ["frame", "frame_idx", "timestamp"] if c in df_track.columns), None)
        x_col = next((c for c in ["x", "x (m)", "x_m", "pos_x"] if c in df_track.columns), None)
        y_col = next((c for c in ["y", "y (m)", "y_m", "pos_y"] if c in df_track.columns), None)
        id_col = next((c for c in ["track_id", "id", "person_id"] if c in df_track.columns), "track_id")

        if frame_col and x_col and y_col:
            # Generate Speed & Vector direction metrics if absent
            if "speed" not in df_track.columns:
                df_track = df_track.sort_values(by=[id_col, frame_col])
                df_track["dx"] = df_track.groupby(id_col)[x_col].diff().fillna(0)
                df_track["dy"] = df_track.groupby(id_col)[y_col].diff().fillna(0)
                df_track["speed"] = np.sqrt(df_track["dx"]**2 + df_track["dy"]**2)

            # --- SECTION 2: FRAME-BY-FRAME PLAYBACK & METRICS ---
            st.markdown("### 2. Motion Playback & Frame Analytics")
            frames_available = sorted(df_track[frame_col].unique())
            selected_f = st.slider("Select Frame for Instant Inspection", min_value=int(min(frames_available)), max_value=int(max(frames_available)), value=int(min(frames_available)))

            curr_frame_df = df_track[df_track[frame_col] == selected_f]

            col_fb1, col_fb2 = st.columns(2)

            with col_fb1:
                st.markdown(f"**Human Movement Scatter Plan (Frame #{selected_f})**")
                fig_play = go.Figure()

                # Add CAD Walls
                for line in st.session_state.get("dxf_walls", []):
                    if hasattr(line, "xy"):
                        wx, wy = line.xy
                        fig_play.add_trace(go.Scatter(x=list(wx), y=list(wy), mode="lines", line=dict(color="#00ADB5", width=1), showlegend=False, hoverinfo="none"))

                # Add Active People (Dots)
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
                st.markdown(f"**Frame Density / Volume Heatmap (Frame #{selected_f})**")
                fig_f_hm = px.density_mapbox if False else px.scatter_density if False else None
                fig_f_hm = go.Figure(go.Histogram2dContour(
                    x=curr_frame_df[x_col],
                    y=curr_frame_df[y_col],
                    colorscale="Jet",
                    showscale=True
                ))
                fig_f_hm.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_f_hm, use_container_width=True)

            st.markdown("---")

            # --- SECTION 3: AGGREGATED VIDEO-WIDE CROWD METRICS ---
            st.markdown("### 3. Aggregated Crowd Metrics (Entire Video)")

            m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["📊 Crowd Volume", "🔥 Density Heatmap", "⚡ Speed Analysis", "🧭 Vector Direction/Flow"])

            with m_tab1:
                st.markdown("#### Cumulative Crowd Volume Heatmap")
                fig_vol = go.Figure(go.Histogram2dContour(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Viridis", showscale=True
                ))
                fig_vol.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_vol, use_container_width=True)

            with m_tab2:
                st.markdown("#### Overall Pedestrian Occupancy Density")
                fig_dens = go.Figure(go.Histogram2d(
                    x=df_track[x_col], y=df_track[y_col], colorscale="Hot", showscale=True, nbinsx=30, nbinsy=30
                ))
                fig_dens.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_dens, use_container_width=True)

            with m_tab3:
                st.markdown("#### Spatial Velocity & Speed Distribution (m/s)")
                fig_spd = px.scatter(
                    df_track, x=x_col, y=y_col, color="speed", color_continuous_scale="Plasma",
                    title="Pedestrian Speed Distribution"
                )
                fig_spd.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_spd, use_container_width=True)

            with m_tab4:
                st.markdown("#### Flow Direction Field (Movement Vectors)")
                fig_dir = px.scatter(
                    df_track, x=x_col, y=y_col, color="dx", color_continuous_scale="Coolwarm",
                    title="Directional Velocity Field (X-Vector Shift)"
                )
                fig_dir.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_dir, use_container_width=True)

            # --- SECTION 4: EXPORT CROWD METRICS JSON ---
            st.markdown("---")
            st.markdown("### 4. Export Aggregated Analytics")

            crowd_metrics_export = {
                "total_frames": int(df_track[frame_col].nunique()),
                "total_unique_pedestrians": int(df_track[id_col].nunique()),
                "average_speed": float(df_track["speed"].mean()),
                "max_speed": float(df_track["speed"].max()),
                "frame_density_summary": df_track.groupby(frame_col)[id_col].count().to_dict(),
                "pedestrian_trajectories": df_track[[frame_col, id_col, x_col, y_col, "speed"]].to_dict(orient="records")
            }

            st.download_button(
                label="💾 Export Crowd Metrics JSON",
                data=json.dumps(crowd_metrics_export, indent=2),
                file_name="crowd_analytics_metrics.json",
                mime="application/json",
                use_container_width=True,
            )

        else:
            st.error(f"Missing required coordinate/frame columns in tracking data. Found columns: {list(df_track.columns)}")

    else:
        st.info("💡 Upload a JSON/CSV tracking file above or execute Module 2.3 to view movement playback and analytics.")