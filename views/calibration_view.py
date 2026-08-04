# views/calibration_view.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from utils.homography_engine import HomographyCalibrator

def render_calibration_view(dxf_wall_polygons):
    st.header("📐 Module 2.1: Camera-to-CAD Calibration")
    st.markdown(
        "Select **at least 4 matching point pairs** (e.g., room corners, pillar bases, door thresholds) "
        "between the camera image and the DXF floorplan."
    )

    # Session State Initialization
    if "img_points" not in st.session_state:
        st.session_state.img_points = []
    if "cad_points" not in st.session_state:
        st.session_state.cad_points = []
    if "homography_matrix" not in st.session_state:
        st.session_state.homography_matrix = None

    # --- TOP CONTROL BAR ---
    col_file, col_actions = st.columns([2, 1])
    with col_file:
        uploaded_frame = st.file_uploader(
            "Upload Camera Reference Frame / Screenshot", 
            type=["png", "jpg", "jpeg"]
        )
    
    with col_actions:
        st.write("### Actions")
        if st.button("🗑️ Clear Picked Points", use_container_width=True):
            st.session_state.img_points = []
            st.session_state.cad_points = []
            st.session_state.homography_matrix = None
            st.rerun()

    if not uploaded_frame:
        st.info("👆 Please upload a camera screenshot above to begin calibration.")
        return

    # Load camera image
    camera_img = Image.open(uploaded_frame)
    img_width, img_height = camera_img.size

    # --- SIDE-BY-SIDE PICKING INTERFACE ---
    col_cam, col_dxf = st.columns(2)

    # 1. CAMERA POINT PICKER (Plotly Image Click Event)
    with col_cam:
        st.subheader("1. Camera View (Pixels u, v)")
        
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Image(z=camera_img))
        
        # Overlay existing selected camera points
        if st.session_state.img_points:
            u_coords, v_coords = zip(*st.session_state.img_points)
            fig_cam.add_trace(
                go.Scatter(
                    x=u_coords, y=v_coords,
                    mode="markers+text",
                    marker=dict(color="red", size=12, symbol="cross"),
                    text=[f"P{i+1}" for i in range(len(u_coords))],
                    textposition="top right",
                    name="Selected Points"
                )
            )

        fig_cam.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            xaxis=dict(showgrid=False, zeroline=False, range=[0, img_width]),
            yaxis=dict(showgrid=False, zeroline=False, range=[img_height, 0]), # inverted Y for image space
            clickmode="event+select"
        )

        cam_event = st.plotly_chart(
            fig_cam, 
            on_select="rerun", 
            selection_mode="points",
            use_container_width=True,
            key="plotly_camera"
        )

        # Handle Camera Click
        if cam_event and "selection" in cam_event and cam_event["selection"]["points"]:
            pt = cam_event["selection"]["points"][0]
            new_u, new_v = round(pt["x"], 2), round(pt["y"], 2)
            
            # Prevent duplicate clicks
            if not st.session_state.img_points or (new_u, new_v) != st.session_state.img_points[-1]:
                if len(st.session_state.img_points) == len(st.session_state.cad_points):
                    st.session_state.img_points.append((new_u, new_v))
                    st.rerun()

    # 2. DXF FLOORPLAN POINT PICKER
    with col_dxf:
        st.subheader("2. CAD Plan (World X, Y)")
        
        fig_cad = go.Figure()
        
        # Plot DXF Wall Lines
        for wall in dxf_wall_polygons:
            x, y = wall.exterior.xy
            fig_cad.add_trace(
                go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color="black", width=1.5), showlegend=False)
            )

        # Overlay existing selected CAD points
        if st.session_state.cad_points:
            x_coords, y_coords = zip(*st.session_state.cad_points)
            fig_cad.add_trace(
                go.Scatter(
                    x=x_coords, y=y_coords,
                    mode="markers+text",
                    marker=dict(color="blue", size=12, symbol="circle"),
                    text=[f"P{i+1}" for i in range(len(x_coords))],
                    textposition="top right",
                    name="Selected CAD Points"
                )
            )

        fig_cad.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            clickmode="event+select"
        )

        cad_event = st.plotly_chart(
            fig_cad, 
            on_select="rerun", 
            selection_mode="points",
            use_container_width=True,
            key="plotly_cad"
        )

        # Handle CAD Click
        if cad_event and "selection" in cad_event and cad_event["selection"]["points"]:
            pt = cad_event["selection"]["points"][0]
            new_x, new_y = round(pt["x"], 2), round(pt["y"], 2)
            
            # Enforce sequential paired picking (Camera P_i -> CAD P_i)
            if len(st.session_state.cad_points) < len(st.session_state.img_points):
                if not st.session_state.cad_points or (new_x, new_y) != st.session_state.cad_points[-1]:
                    st.session_state.cad_points.append((new_x, new_y))
                    st.rerun()

    # --- STATUS & PAIRING TABLE ---
    st.markdown("---")
    st.subheader("3. Calibration Point Pairs")

    # Sync Status Message
    n_img, n_cad = len(st.session_state.img_points), len(st.session_state.cad_points)
    if n_img > n_cad:
        st.warning(f"⚠️ Point P{n_img} selected on Camera View. Now click the matching position on the DXF Plan.")
    elif n_img == n_cad and n_img < 4:
        st.info(f"📍 {n_img}/4 point pairs selected. Need at least {4 - n_img} more pair(s).")
    elif n_img == n_cad and n_img >= 4:
        st.success(f"✅ Ready! {n_img} valid point pairs selected.")

    # Render Table of Corresponding Coordinates
    if n_img > 0:
        table_data = []
        for i in range(max(n_img, n_cad)):
            u, v = st.session_state.img_points[i] if i < n_img else ("-", "-")
            x, y = st.session_state.cad_points[i] if i < n_cad else ("-", "-")
            table_data.append({"Point": f"P{i+1}", "Camera Pixel (u)": u, "Camera Pixel (v)": v, "CAD X": x, "CAD Y": y})
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # --- COMPUTE HOMOGRAPHY BUTTON ---
    if n_img >= 4 and n_img == n_cad:
        if st.button("🔥 Compute Homography Matrix (H)", type="primary", use_container_width=True):
            try:
                calibrator = HomographyCalibrator()
                H = calibrator.compute_homography(
                    st.session_state.img_points, 
                    st.session_state.cad_points
                )
                st.session_state.homography_matrix = H
                st.success("Homography Matrix successfully computed and saved to Session State!")
                
                # Display computed 3x3 Matrix
                st.write("**3x3 Transformation Matrix (H):**")
                st.code(np.array2string(H, precision=4, suppress_small=True))
                
            except Exception as e:
                st.error(f"Homography calculation failed: {e}")