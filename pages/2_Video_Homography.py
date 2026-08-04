# pages/2_Video_Homography.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
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


def draw_overlay_on_frame(frame_rgb: np.ndarray, points: list) -> np.ndarray:
    """
    Burns point markers, crosshairs, target numbers, and polygon wireframe
    directly onto the video frame pixels using OpenCV.
    """
    canvas = frame_rgb.copy()
    num_pts = len(points)
    if num_pts == 0:
        return canvas

    # Draw wireframe polygon connecting selected points
    pt_array = np.array(points, dtype=np.int32)
    if num_pts >= 2:
        cv2.polylines(canvas, [pt_array], isClosed=(num_pts >= 3), color=(255, 0, 0), thickness=2)

    # Draw crosshairs and labels for each point
    for idx, (u, v) in enumerate(points):
        x, y = int(u), int(v)
        # Red Circle
        cv2.circle(canvas, (x, y), 8, (255, 0, 0), -1)
        # White Border
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), 2)
        # Crosshair lines
        cv2.line(canvas, (x - 12, y), (x + 12, y), (255, 255, 255), 2)
        cv2.line(canvas, (x, y - 12), (x, y + 12), (255, 255, 255), 2)
        # Label Text P1, P2, etc.
        cv2.putText(
            canvas,
            f"P{idx+1}",
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"P{idx+1}",
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


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
            "Upload Surveillance Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"]
        )
    with col_act:
        st.write("### Quick Actions")
        c_act1, c_act2 = st.columns(2)
        if c_act1.button("↩️ Undo Point", use_container_width=True):
            n_cam = len(st.session_state.img_points)
            n_cad = len(st.session_state.cad_points)
            if n_cam > n_cad:
                st.session_state.img_points.pop()
            elif n_cam == n_cad and n_cam > 0:
                st.session_state.cad_points.pop()
            st.rerun()

        if c_act2.button("🗑️ Clear All", use_container_width=True):
            st.session_state.img_points = []
            st.session_state.cad_points = []
            st.session_state.homography_matrix = None
            st.rerun()

    if not uploaded_video:
        st.info("👆 Upload a surveillance video file above to begin calibration.")
    else:
        # Frame extraction slider
        col_frame_ctrl, _ = st.columns([3, 1])
        with col_frame_ctrl:
            frame_idx = st.slider(
                "Select Frame for Calibration Target",
                min_value=0,
                max_value=1000,
                value=st.session_state.selected_frame_idx,
                step=5,
                help="Slide to pick a clear frame where reference head-height landmarks are unobstructed.",
            )
            if frame_idx != st.session_state.selected_frame_idx:
                st.session_state.selected_frame_idx = frame_idx
                st.rerun()

        raw_frame_rgb = extract_frame_from_video(
            uploaded_video, frame_number=st.session_state.selected_frame_idx
        )

        if raw_frame_rgb is None:
            st.error(
                "Failed to extract frame from video. Please try another frame index or video format."
            )
        else:
            img_h, img_w, _ = raw_frame_rgb.shape

            # Guided Instruction Banner
            n_img = len(st.session_state.img_points)
            n_cad = len(st.session_state.cad_points)

            if n_img == n_cad:
                st.info(
                    f"🎯 **STEP {n_img + 1}**: Click **Point P{n_img + 1}** on the **Left (Camera View)** at Head Level."
                )
            else:
                st.warning(
                    f"🎯 **STEP {n_img}**: Camera Point P{n_img} recorded! Now click the **Matching Point P{n_img}** on the **Right (CAD Plan)**."
                )

            col_cam, col_cad = st.columns(2)

            # ----------------------------------------------------
            # 1. Camera View Picker (Direct Burn Pixel Overlay)
            # ----------------------------------------------------
            with col_cam:
                st.markdown("##### 1. Camera View (Head Level Pixels)")

                # Burn overlay onto frame before Plotly renders
                frame_with_overlay = draw_overlay_on_frame(
                    raw_frame_rgb, st.session_state.img_points
                )

                fig_cam = go.Figure()
                fig_cam.add_trace(go.Image(z=frame_with_overlay))

                fig_cam.update_layout(
                    dragmode=False,
                    hovermode="closest",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450,
                    xaxis=dict(
                        showgrid=False, zeroline=False, range=[0, img_w], autorange=False
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        range=[img_h, 0],
                        autorange=False,
                    ),
                    clickmode="event+select",
                )

                cam_event = st.plotly_chart(
                    fig_cam,
                    on_select="rerun",
                    selection_mode="points",
                    use_container_width=True,
                    key="plotly_cam",
                )

                # Capture Camera Click Event
                if (
                    cam_event
                    and "selection" in cam_event
                    and cam_event["selection"]["points"]
                ):
                    pt = cam_event["selection"]["points"][0]
                    nu, nv = round(pt["x"], 2), round(pt["y"], 2)

                    if len(st.session_state.img_points) == len(st.session_state.cad_points):
                        if (
                            not st.session_state.img_points
                            or (nu, nv) != st.session_state.img_points[-1]
                        ):
                            st.session_state.img_points.append((nu, nv))
                            st.rerun()

                # Manual Point Entry Expander for Camera
                with st.expander("➕ Manual Pixel Coordinate Input"):
                    c_u, c_v = st.columns(2)
                    manual_u = c_u.number_input(
                        "Pixel U (X)",
                        min_value=0.0,
                        max_value=float(img_w),
                        value=0.0,
                        key="manual_u",
                    )
                    manual_v = c_v.number_input(
                        "Pixel V (Y)",
                        min_value=0.0,
                        max_value=float(img_h),
                        value=0.0,
                        key="manual_v",
                    )
                    if st.button("Add Camera Point Manually"):
                        if len(st.session_state.img_points) == len(st.session_state.cad_points):
                            st.session_state.img_points.append(
                                (round(manual_u, 2), round(manual_v, 2))
                            )
                            st.rerun()

            # ----------------------------------------------------
            # 2. DXF CAD Picker
            # ----------------------------------------------------
            with col_cad:
                st.markdown("##### 2. CAD Floorplan (World X, Y)")
                fig_cad = go.Figure()

                # Render DXF Wall Polylines
                dxf_walls = st.session_state.get("dxf_walls", [])
                for wall in dxf_walls:
                    wx, wy = wall.exterior.xy
                    fig_cad.add_trace(
                        go.Scatter(
                            x=list(wx),
                            y=list(wy),
                            mode="lines",
                            line=dict(color="black", width=1.5),
                            showlegend=False,
                        )
                    )

                # Overlay CAD Points and Wireframe
                if st.session_state.cad_points:
                    cx_pts, cy_pts = zip(*st.session_state.cad_points)

                    poly_cx = list(cx_pts)
                    poly_cy = list(cy_pts)
                    if len(poly_cx) >= 3:
                        poly_cx.append(poly_cx[0])
                        poly_cy.append(poly_cy[0])

                    fig_cad.add_trace(
                        go.Scatter(
                            x=poly_cx,
                            y=poly_cy,
                            mode="markers+text+lines",
                            marker=dict(
                                color="#0000FF",
                                size=14,
                                symbol="circle",
                                line=dict(width=1.5, color="#FFFFFF"),
                            ),
                            line=dict(color="#0000FF", width=2, dash="dash"),
                            text=[f"P{i+1}" for i in range(len(cx_pts))],
                            textposition="top right",
                            textfont=dict(color="blue", size=14, family="Arial Black"),
                            name="Selected CAD Region",
                        )
                    )

                fig_cad.update_layout(
                    dragmode=False,
                    hovermode="closest",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450,
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    clickmode="event+select",
                    showlegend=False,
                )

                cad_event = st.plotly_chart(
                    fig_cad,
                    on_select="rerun",
                    selection_mode="points",
                    use_container_width=True,
                    key="plotly_cad",
                )

                # Capture CAD Click Event
                if (
                    cad_event
                    and "selection" in cad_event
                    and cad_event["selection"]["points"]
                ):
                    pt = cad_event["selection"]["points"][0]
                    nx, ny = round(pt["x"], 2), round(pt["y"], 2)
                    if len(st.session_state.cad_points) < len(st.session_state.img_points):
                        if (
                            not st.session_state.cad_points
                            or (nx, ny) != st.session_state.cad_points[-1]
                        ):
                            st.session_state.cad_points.append((nx, ny))
                            st.rerun()

                # Manual Point Entry Expander for CAD
                with st.expander("➕ Manual CAD Coordinate Input"):
                    c_x, c_y = st.columns(2)
                    manual_x = c_x.number_input("CAD X (m)", value=0.0, key="manual_cad_x")
                    manual_y = c_y.number_input("CAD Y (m)", value=0.0, key="manual_cad_y")
                    if st.button("Add CAD Point Manually"):
                        if len(st.session_state.cad_points) < len(
                            st.session_state.img_points
                        ):
                            st.session_state.cad_points.append(
                                (round(manual_x, 2), round(manual_y, 2))
                            )
                            st.rerun()

            # ----------------------------------------------------
            # Calibration Point Pair Table & Solver
            # ----------------------------------------------------
            st.markdown("---")
            n_img, n_cad = len(st.session_state.img_points), len(st.session_state.cad_points)

            if n_img > 0 or n_cad > 0:
                rows = []
                for i in range(max(n_img, n_cad)):
                    u, v = (
                        st.session_state.img_points[i]
                        if i < n_img
                        else ("Waiting...", "Waiting...")
                    )
                    x, y = (
                        st.session_state.cad_points[i]
                        if i < n_cad
                        else ("Waiting...", "Waiting...")
                    )
                    rows.append(
                        {
                            "Point Pair": f"P{i+1}",
                            "Camera U (px)": u,
                            "Camera V (px)": v,
                            "CAD X (m)": x,
                            "CAD Y (m)": y,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Solve Matrix Button
            if n_img >= 4 and n_img == n_cad:
                if st.button(
                    "⚡ Solve Homography Matrix (H)",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        calibrator = HomographyCalibrator()
                        H = calibrator.compute_homography(
                            st.session_state.img_points, st.session_state.cad_points
                        )
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
        st.session_state.get("dxf_walls", []), st.session_state.get("vga_grid_df", None)
    )