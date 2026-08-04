# pages/2_Video_Homography.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.tracking_engine import HomographyCalibrator, extract_frame_from_video
from views.tracking_view import render_tracking_view

st.set_page_config(page_title="Module 2: Video Homography & Tracking", layout="wide")

# Session State Initialization
if "img_points" not in st.session_state:
    st.session_state.img_points = []
if "cad_points" not in st.session_state:
    st.session_state.cad_points = []
if "homography_matrix" not in st.session_state:
    st.session_state.homography_matrix = None
if "selected_frame_idx" not in st.session_state:
    st.session_state.selected_frame_idx = 0

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
        uploaded_video = st.file_uploader(
            "Upload Surveillance Video (.mp4, .avi, .mov)", 
            type=["mp4", "avi", "mov"]
        )
    with col_act:
        st.write("### Actions")
        if st.button("🗑️ Clear Calibration Points", use_container_width=True):
            st.session_state.img_points = []
            st.session_state.cad_points = []
            st.session_state.homography_matrix = None
            st.rerun()

    if not uploaded_video:
        st.info("👆 Upload a surveillance video file above to begin calibration.")
    else:
        # Frame extraction controller
        col_frame_ctrl, _ = st.columns([3, 1])
        with col_frame_ctrl:
            frame_idx = st.slider(
                "Select Frame for Calibration Target",
                min_value=0,
                max_value=1000,
                value=st.session_state.selected_frame_idx,
                step=5,
                help="Slide to pick a clear frame where reference head-height landmarks are unobstructed."
            )
            if frame_idx != st.session_state.selected_frame_idx:
                st.session_state.selected_frame_idx = frame_idx
                st.rerun()

        camera_frame_rgb = extract_frame_from_video(
            uploaded_video, 
            frame_number=st.session_state.selected_frame_idx
        )

        if camera_frame_rgb is None:
            st.error("Failed to extract frame from video. Please try another frame index or video format.")
        else:
            img_h, img_w, _ = camera_frame_rgb.shape

            col_cam, col_cad = st.columns(2)

            # ----------------------------------------------------
            # 1. Camera View Picker
            # ----------------------------------------------------
            with col_cam:
                st.markdown("##### 1. Camera View (Head Level Pixels)")
                
                # Base Plotly Figure with Explicit Z-Ordering
                fig_cam = go.Figure()
                fig_cam.add_trace(go.Image(z=camera_frame_rgb))

                # Overlay Selected Camera Points & Closed Polygon Wireframe
                if st.session_state.img_points:
                    u_pts, v_pts = zip(*st.session_state.img_points)
                    
                    # Create closed wireframe loop if 3+ points are selected
                    poly_u = list(u_pts)
                    poly_v = list(v_pts)
                    if len(poly_u) >= 3:
                        poly_u.append(poly_u[0])
                        poly_v.append(poly_v[0])

                    fig_cam.add_trace(go.Scatter(
                        x=poly_u, 
                        y=poly_v, 
                        mode="markers+text+lines",
                        marker=dict(
                            color="#FF0000", 
                            size=16, 
                            symbol="cross",
                            line=dict(width=2, color="#FFFFFF")
                        ),
                        line=dict(color="#FF0000", width=2, dash="dash"),
                        text=[f"P{i+1}" for i in range(len(u_pts))],
                        textposition="top right",
                        textfont=dict(color="red", size=14, family="Arial Black"),
                        name="Selected Camera Region",
                        hoverinfo="x+y+text"
                    ))

                # Layout Config for Strict Pointer Behavior & Inverted Y Pixel Coordinates
                fig_cam.update_layout(
                    dragmode="select",
                    hovermode="closest",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450,
                    xaxis=dict(showgrid=False, zeroline=False, range=[0, img_w], autorange=False),
                    yaxis=dict(showgrid=False, zeroline=False, range=[img_h, 0], autorange=False),
                    clickmode="event+select",
                    showlegend=False
                )

                cam_event = st.plotly_chart(
                    fig_cam, 
                    on_select="rerun", 
                    selection_mode="points",
                    use_container_width=True, 
                    key="plotly_cam"
                )

                # Process Camera Click Event
                if cam_event and "selection" in cam_event and cam_event["selection"]["points"]:
                    pt = cam_event["selection"]["points"][0]
                    nu, nv = round(pt["x"], 2), round(pt["y"], 2)
                    
                    # Alternating pick logic: Camera Point -> CAD Point
                    if not st.session_state.img_points or len(st.session_state.img_points) == len(st.session_state.cad_points):
                        if not st.session_state.img_points or (nu, nv) != st.session_state.img_points[-1]:
                            st.session_state.img_points.append((nu, nv))
                            st.rerun()

                # Manual Point Entry Expander for Camera
                with st.expander("➕ Manual Pixel Coordinate Input"):
                    c_u, c_v = st.columns(2)
                    manual_u = c_u.number_input("Pixel U (X)", min_value=0.0, max_value=float(img_w), value=0.0, key="manual_u")
                    manual_v = c_v.number_input("Pixel V (Y)", min_value=0.0, max_value=float(img_h), value=0.0, key="manual_v")
                    if st.button("Add Camera Point Manually"):
                        if len(st.session_state.img_points) == len(st.session_state.cad_points):
                            st.session_state.img_points.append((round(manual_u, 2), round(manual_v, 2)))
                            st.rerun()

            # ----------------------------------------------------
            # 2. DXF CAD Picker
            # ----------------------------------------------------
            with col_cad:
                st.markdown("##### 2. CAD Floorplan (World X, Y)")
                fig_cad = go.Figure()
                
                # Render Wall Polylines
                dxf_walls = st.session_state.get("dxf_walls", [])
                for wall in dxf_walls:
                    wx, wy = wall.exterior.xy
                    fig_cad.add_trace(go.Scatter(
                        x=list(wx), y=list(wy), mode="lines", 
                        line=dict(color="black", width=1.5), showlegend=False
                    ))

                # Overlay Selected CAD Points & Closed Polygon Wireframe
                if st.session_state.cad_points:
                    cx_pts, cy_pts = zip(*st.session_state.cad_points)
                    
                    poly_cx = list(cx_pts)
                    poly_cy = list(cy_pts)
                    if len(poly_cx) >= 3:
                        poly_cx.append(poly_cx[0])
                        poly_cy.append(poly_cy[0])

                    fig_cad.add_trace(go.Scatter(
                        x=poly_cx, 
                        y=poly_cy, 
                        mode="markers+text+lines",
                        marker=dict(
                            color="#0000FF", 
                            size=14, 
                            symbol="circle",
                            line=dict(width=1.5, color="#FFFFFF")
                        ),
                        line=dict(color="#0000FF", width=2, dash="dash"),
                        text=[f"P{i+1}" for i in range(len(cx_pts))],
                        textposition="top right", 
                        textfont=dict(color="blue", size=14, family="Arial Black"),
                        name="Selected CAD Region"
                    ))

                fig_cad.update_layout(
                    dragmode="select",
                    hovermode="closest",
                    margin=dict(l=0, r=0, t=10, b=0), 
                    height=450,
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    clickmode="event+select",
                    showlegend=False
                )

                cad_event = st.plotly_chart(
                    fig_cad, 
                    on_select="rerun", 
                    selection_mode="points",
                    use_container_width=True, 
                    key="plotly_cad"
                )

                # Process CAD Click Event
                if cad_event and "selection" in cad_event and cad_event["selection"]["points"]:
                    pt = cad_event["selection"]["points"][0]
                    nx, ny = round(pt["x"], 2), round(pt["y"], 2)
                    if len(st.session_state.cad_points) < len(st.session_state.img_points):
                        if not st.session_state.cad_points or (nx, ny) != st.session_state.cad_points[-1]:
                            st.session_state.cad_points.append((nx, ny))
                            st.rerun()

                # Manual Point Entry Expander for CAD
                with st.expander("➕ Manual CAD Coordinate Input"):
                    c_x, c_y = st.columns(2)
                    manual_x = c_x.number_input("CAD X (m)", value=0.0, key="manual_cad_x")
                    manual_y = c_y.number_input("CAD Y (m)", value=0.0, key="manual_cad_y")
                    if st.button("Add CAD Point Manually"):
                        if len(st.session_state.cad_points) < len(st.session_state.img_points):
                            st.session_state.cad_points.append((round(manual_x, 2), round(manual_y, 2)))
                            st.rerun()

            # ----------------------------------------------------
            # Calibration Pair Table & Solver
            # ----------------------------------------------------
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
                    rows.append({"Point": f"P{i+1}", "Camera u (px)": u, "Camera v (px)": v, "CAD X (m)": x, "CAD Y (m)": y})
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