# pages/2_Video_Homography.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from utils.tracking_engine import HomographyCalibrator
from views.tracking_view import render_tracking_view

st.set_page_config(page_title="Module 2: Video Homography & Tracking", layout="wide")

# Session State Initialization
if "img_points" not in st.session_state:
    st.session_state.img_points = []
if "cad_points" not in st.session_state:
    st.session_state.cad_points = []
if "homography_matrix" not in st.session_state:
    st.session_state.homography_matrix = None

st.title("📹 Module 2: Video Homography & Person Tracking")

# Navigation Tabs
tab_calib, tab_tracking = st.tabs(["📐 2.1 Head-Height Calibration", "🔥 2.2 Occupancy Analytics"])

# ==========================================
# TAB 1: CALIBRATION INTERFACE
# ==========================================
with tab_calib:
    st.subheader("Step 2.1: Head-Height Calibration")
    st.markdown(
        "💡 **Crowd Mode Strategy**: Select **at least 4 point pairs** at **Head Height** "
        "(~1.7m above floor, e.g., tops of doors, pillars, or signboards) visible in both the camera frame and DXF plan."
    )

    col_file, col_act = st.columns([2, 1])
    with col_file:
        uploaded_frame = st.file_uploader("Upload Camera Reference Frame", type=["png", "jpg", "jpeg"])
    with col_act:
        st.write("### Actions")
        if st.button("🗑️ Clear Calibration Points", use_container_width=True):
            st.session_state.img_points = []
            st.session_state.cad_points = []
            st.session_state.homography_matrix = None
            st.rerun()

    if not uploaded_frame:
        st.info("👆 Upload a camera screenshot above to begin calibration.")
    else:
        camera_img = Image.open(uploaded_frame)
        img_w, img_h = camera_img.size

        col_cam, col_cad = st.columns(2)

        # 1. Camera View Picker
        with col_cam:
            st.markdown("##### 1. Camera View (Head Level Pixels)")
            fig_cam = go.Figure()
            fig_cam.add_trace(go.Image(z=camera_img))

            if st.session_state.img_points:
                u_pts, v_pts = zip(*st.session_state.img_points)
                fig_cam.add_trace(go.Scatter(
                    x=u_pts, y=v_pts, mode="markers+text",
                    marker=dict(color="red", size=12, symbol="cross"),
                    text=[f"P{i+1}" for i in range(len(u_pts))],
                    textposition="top right", name="Selected Heads"
                ))

            fig_cam.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=420,
                xaxis=dict(showgrid=False, range=[0, img_w]),
                yaxis=dict(showgrid=False, range=[img_h, 0]),
                clickmode="event+select"
            )

            cam_event = st.plotly_chart(
                fig_cam, on_select="rerun", selection_mode="points",
                use_container_width=True, key="plotly_cam"
            )

            if cam_event and "selection" in cam_event and cam_event["selection"]["points"]:
                pt = cam_event["selection"]["points"][0]
                nu, nv = round(pt["x"], 2), round(pt["y"], 2)
                if not st.session_state.img_points or (nu, nv) != st.session_state.img_points[-1]:
                    if len(st.session_state.img_points) == len(st.session_state.cad_points):
                        st.session_state.img_points.append((nu, nv))
                        st.rerun()

        # 2. DXF CAD Picker
        with col_cad:
            st.markdown("##### 2. CAD Floorplan (World X, Y)")
            fig_cad = go.Figure()
            
            dxf_walls = st.session_state.get("dxf_walls", [])
            for wall in dxf_walls:
                wx, wy = wall.exterior.xy
                fig_cad.add_trace(go.Scatter(
                    x=list(wx), y=list(wy), mode="lines", 
                    line=dict(color="black", width=1.5), showlegend=False
                ))

            if st.session_state.cad_points:
                cx_pts, cy_pts = zip(*st.session_state.cad_points)
                fig_cad.add_trace(go.Scatter(
                    x=cx_pts, y=cy_pts, mode="markers+text",
                    marker=dict(color="blue", size=12, symbol="circle"),
                    text=[f"P{i+1}" for i in range(len(cx_pts))],
                    textposition="top right", name="Selected CAD Points"
                ))

            fig_cad.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=420,
                yaxis=dict(scaleanchor="x", scaleratio=1),
                clickmode="event+select"
            )

            cad_event = st.plotly_chart(
                fig_cad, on_select="rerun", selection_mode="points",
                use_container_width=True, key="plotly_cad"
            )

            if cad_event and "selection" in cad_event and cad_event["selection"]["points"]:
                pt = cad_event["selection"]["points"][0]
                nx, ny = round(pt["x"], 2), round(pt["y"], 2)
                if len(st.session_state.cad_points) < len(st.session_state.img_points):
                    if not st.session_state.cad_points or (nx, ny) != st.session_state.cad_points[-1]:
                        st.session_state.cad_points.append((nx, ny))
                        st.rerun()

        # Calibration Pair Table
        st.markdown("---")
        n_img, n_cad = len(st.session_state.img_points), len(st.session_state.cad_points)
        
        if n_img > n_cad:
            st.warning(f"📍 Point P{n_img} clicked on Camera view. Now click the matching point on the CAD plan.")
        elif n_img == n_cad and n_img < 4:
            st.info(f"📍 Selected {n_img}/4 point pairs. Need at least {4 - n_img} more pair(s).")
        elif n_img == n_cad and n_img >= 4:
            st.success(f"✅ Ready! {n_img} head-height point pairs selected.")

        if n_img > 0:
            rows = []
            for i in range(max(n_img, n_cad)):
                u, v = st.session_state.img_points[i] if i < n_img else ("-", "-")
                x, y = st.session_state.cad_points[i] if i < n_cad else ("-", "-")
                rows.append({"Point": f"P{i+1}", "Camera u": u, "Camera v": v, "CAD X": x, "CAD Y": y})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Solve Matrix Button
        if n_img >= 4 and n_img == n_cad:
            if st.button("⚡ Solve Homography Matrix (H)", type="primary", use_container_width=True):
                try:
                    calibrator = HomographyCalibrator()
                    H = calibrator.compute_homography(st.session_state.img_points, st.session_state.cad_points)
                    st.session_state.homography_matrix = H
                    st.success("Homography Matrix successfully computed and saved!")
                    st.code(np.array2string(H, precision=4, suppress_small=True))
                except Exception as e:
                    st.error(f"Failed to calculate Homography Matrix: {e}")

# ==========================================
# TAB 2: OCCUPANCY TRACKING VIEW
# ==========================================
with tab_tracking:
    render_tracking_view(
        st.session_state.get("dxf_walls", []),
        st.session_state.get("vga_grid_df", None)
    )