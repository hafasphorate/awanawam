# pages/2_Video_Homography.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
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
# HELPER RENDERING FUNCTIONS (DIRECT BURN)
# ==========================================
def draw_camera_overlay(frame_rgb: np.ndarray, points: list) -> np.ndarray:
    """Burns camera points, crosshairs, numbers, and polygon wireframe into RGB pixels."""
    canvas = frame_rgb.copy()
    num_pts = len(points)
    if num_pts == 0:
        return canvas

    # Draw Red Wireframe Polygon
    if num_pts >= 2:
        pt_array = np.array([(int(u), int(v)) for u, v in points], dtype=np.int32)
        cv2.polylines(canvas, [pt_array], isClosed=(num_pts >= 3), color=(255, 0, 0), thickness=3)

    # Draw Point Targets & Labels
    for idx, (u, v) in enumerate(points):
        x, y = int(u), int(v)
        cv2.circle(canvas, (x, y), 7, (255, 0, 0), -1)
        cv2.circle(canvas, (x, y), 9, (255, 255, 255), 2)
        cv2.line(canvas, (x - 15, y), (x + 15, y), (255, 255, 255), 2)
        cv2.line(canvas, (x, y - 15), (x, y + 15), (255, 255, 255), 2)
        
        lbl = f"P{idx+1}"
        cv2.putText(canvas, lbl, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def render_cad_with_polygon(dxf_walls: list, cad_points: list) -> Image.Image:
    """Renders CAD walls + CAD point wireframe directly onto a Matplotlib buffer image."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.set_facecolor("#FFFFFF")
    
    # Draw CAD Walls
    for wall in dxf_walls:
        wx, wy = wall.exterior.xy
        ax.plot(wx, wy, color="black", linewidth=1.2)

    # Draw CAD Polygon Wireframe & Points
    if cad_points:
        cx, cy = zip(*cad_points)
        num_pts = len(cad_points)
        
        # Plot Polygon line
        if num_pts >= 2:
            px_list = list(cx) + ([cx[0]] if num_pts >= 3 else [])
            py_list = list(cy) + ([cy[0]] if num_pts >= 3 else [])
            ax.plot(px_list, py_list, color="blue", linestyle="--", linewidth=2.5, label="CAD Bounds")
            
        # Plot Point Markers and Labels
        ax.scatter(cx, cy, color="blue", s=80, edgecolors="white", zorder=5)
        for i, (x, y) in enumerate(zip(cx, cy)):
            ax.annotate(f"P{i+1}", (x, y), textcoords="offset points", xytext=(8, 8),
                        ha="left", fontsize=11, fontweight="bold", color="blue")

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    # Convert Matplotlib to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ==========================================
# TAB 1: CALIBRATION INTERFACE
# ==========================================
with tab_calib:
    st.subheader("Step 2.1: Head-Height Calibration")
    st.markdown(
        "💡 **Crowd Mode Strategy**: Select **at least 4 point pairs** at **Head Height** "
        "(~1.7m above floor) visible in both the camera frame and DXF plan."
    )

    col_file, col_act = st.columns([2, 1])
    with col_file:
        uploaded_video = st.file_uploader("Upload Surveillance Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    with col_act:
        st.write("### Actions")
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
                min_value=0, max_value=1000,
                value=st.session_state.selected_frame_idx, step=5
            )
            if frame_idx != st.session_state.selected_frame_idx:
                st.session_state.selected_frame_idx = frame_idx
                st.rerun()

        raw_frame_rgb = extract_frame_from_video(uploaded_video, frame_number=st.session_state.selected_frame_idx)

        if raw_frame_rgb is None:
            st.error("Failed to extract frame from video.")
        else:
            img_h, img_w, _ = raw_frame_rgb.shape
            n_img = len(st.session_state.img_points)
            n_cad = len(st.session_state.cad_points)

            # Guided Banner
            if n_img == n_cad:
                st.info(f"🎯 **STEP {n_img + 1}**: Select **Camera Point P{n_img + 1}** on the left.")
            else:
                st.warning(f"🎯 **STEP {n_img}**: Camera P{n_img} saved! Now select matching **CAD Point P{n_img}** on the right.")

            col_cam, col_cad = st.columns(2)

            # ----------------------------------------------------
            # 1. Camera View (Direct Burned OpenCV Canvas)
            # ----------------------------------------------------
            with col_cam:
                st.markdown("##### 1. Camera View (Head Level Pixels)")
                frame_burned = draw_camera_overlay(raw_frame_rgb, st.session_state.img_points)

                fig_cam = go.Figure(go.Image(z=frame_burned))
                fig_cam.update_layout(
                    dragmode=False, margin=dict(l=0, r=0, t=10, b=0), height=420,
                    xaxis=dict(showgrid=False, zeroline=False, range=[0, img_w], autorange=False),
                    yaxis=dict(showgrid=False, zeroline=False, range=[img_h, 0], autorange=False),
                    clickmode="event+select"
                )

                cam_event = st.plotly_chart(fig_cam, on_select="rerun", selection_mode="points", use_container_width=True, key="plotly_cam")

                if cam_event and "selection" in cam_event and cam_event["selection"]["points"]:
                    pt = cam_event["selection"]["points"][0]
                    nu, nv = round(pt["x"], 2), round(pt["y"], 2)
                    if len(st.session_state.img_points) == len(st.session_state.cad_points):
                        if not st.session_state.img_points or (nu, nv) != st.session_state.img_points[-1]:
                            st.session_state.img_points.append((nu, nv))
                            st.rerun()

                with st.expander("➕ Manual Camera Coordinates"):
                    c_u, c_v = st.columns(2)
                    manual_u = c_u.number_input("Pixel U (X)", min_value=0.0, max_value=float(img_w), value=0.0)
                    manual_v = c_v.number_input("Pixel V (Y)", min_value=0.0, max_value=float(img_h), value=0.0)
                    if st.button("Add Camera Point Manually"):
                        if len(st.session_state.img_points) == len(st.session_state.cad_points):
                            st.session_state.img_points.append((round(manual_u, 2), round(manual_v, 2)))
                            st.rerun()

            # ----------------------------------------------------
            # 2. CAD View (Direct Burned Matplotlib Canvas)
            # ----------------------------------------------------
            with col_cad:
                st.markdown("##### 2. CAD Floorplan (World X, Y)")
                dxf_walls = st.session_state.get("dxf_walls", [])
                cad_img_burned = render_cad_with_polygon(dxf_walls, st.session_state.cad_points)

                # Fetch CAD spatial limits for click positioning
                min_x, max_x = -10.0, 50.0
                min_y, max_y = -10.0, 50.0
                if dxf_walls:
                    bounds = [w.bounds for w in dxf_walls]
                    min_x = min(b[0] for b in bounds) - 2.0
                    max_x = max(b[2] for b in bounds) + 2.0
                    min_y = min(b[1] for b in bounds) - 2.0
                    max_y = max(b[3] for b in bounds) + 2.0

                fig_cad = go.Figure(go.Image(z=np.array(cad_img_burned)))
                fig_cad.update_layout(
                    dragmode=False, margin=dict(l=0, r=0, t=10, b=0), height=420,
                    xaxis=dict(showgrid=False, zeroline=False, range=[0, cad_img_burned.width]),
                    yaxis=dict(showgrid=False, zeroline=False, range=[cad_img_burned.height, 0]),
                    clickmode="event+select"
                )

                cad_event = st.plotly_chart(fig_cad, on_select="rerun", selection_mode="points", use_container_width=True, key="plotly_cad")

                if cad_event and "selection" in cad_event and cad_event["selection"]["points"]:
                    pt = cad_event["selection"]["points"][0]
                    # Map click pixel on Matplotlib image back to World CAD coordinates
                    click_px, click_py = pt["x"], pt["y"]
                    cad_x = min_x + (click_px / cad_img_burned.width) * (max_x - min_x)
                    cad_y = max_y - (click_py / cad_img_burned.height) * (max_y - min_y)

                    if len(st.session_state.cad_points) < len(st.session_state.img_points):
                        st.session_state.cad_points.append((round(cad_x, 2), round(cad_y, 2)))
                        st.rerun()

                with st.expander("➕ Manual CAD Coordinates"):
                    c_x, c_y = st.columns(2)
                    manual_x = c_x.number_input("CAD X (m)", value=0.0, key="manual_cad_x")
                    manual_y = c_y.number_input("CAD Y (m)", value=0.0, key="manual_cad_y")
                    if st.button("Add CAD Point Manually"):
                        if len(st.session_state.cad_points) < len(st.session_state.img_points):
                            st.session_state.cad_points.append((round(manual_x, 2), round(manual_y, 2)))
                            st.rerun()

            # ----------------------------------------------------
            # Pair Summary & Matrix Solver
            # ----------------------------------------------------
            st.markdown("---")
            if n_img > 0 or n_cad > 0:
                rows = []
                for i in range(max(n_img, n_cad)):
                    u, v = st.session_state.img_points[i] if i < n_img else ("Waiting...", "Waiting...")
                    x, y = st.session_state.cad_points[i] if i < n_cad else ("Waiting...", "Waiting...")
                    rows.append({"Point Pair": f"P{i+1}", "Camera U (px)": u, "Camera V (px)": v, "CAD X (m)": x, "CAD Y (m)": y})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

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