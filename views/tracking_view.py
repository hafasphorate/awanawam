# views/tracking_view.py
import json
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.homography_engine import compute_homography_matrix
from utils.tracking_engine import extract_frame_from_video, process_video_frame


@st.cache_data(show_spinner=False)
def get_video_frame_count_cached(file_name: str, video_bytes: bytes) -> int:
    """Cached helper to read video frame count once without repeated disk I/O."""
    if not video_bytes:
        return 100
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
            tmp_v.write(video_bytes)
            tmp_path = tmp_v.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return total if total > 0 else 100
    except Exception:
        return 100


def render_tracking_view(dxf_walls: list, vga_grid_df: pd.DataFrame = None):
    """Renders Tab 2.3: Video Homography and Live 2D Motion Mapping."""
    st.subheader("Step 2.3: Occupancy & Human Movement Analytics")

    # Check for Uploaded Video
    uploaded_video = st.session_state.get("uploaded_video_file", None)
    if uploaded_video is None:
        st.warning("⚠️ Please upload a surveillance video file in **Tab 2.1** first.")
        return

    # Check for Floorplan Corners
    four_corners = st.session_state.get("four_corners", [])
    if len(four_corners) < 4:
        st.warning(
            f"⚠️ Please select all **4 ROI corners** on the 2D floorplan in **Tab 2.2** first "
            f"(Currently selected: {len(four_corners)}/4)."
        )
        return

    st.markdown("---")

    # Read video bytes safely for cached frame counting
    uploaded_video.seek(0)
    video_bytes = uploaded_video.read()
    uploaded_video.seek(0)

    max_frames = get_video_frame_count_cached(uploaded_video.name, video_bytes)

    # 🎛️ Advanced Precision & CPU Controls
    with st.expander("🛠️ Detection Sensitivity & CPU Optimization Controls", expanded=True):
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            model_name = st.selectbox(
                "Select Model Architecture:",
                [
                    "yolov8n.pt",       # Fast & CPU-friendly
                    "yolov8s.pt",       # Balanced Accuracy
                    "rtdetr-l.pt",      # Transformer-based
                    "yolov8x-pose.pt",   # Pose/Keypoints
                ],
                index=0,
                help="`yolov8n.pt` is lightweight and recommended for high-density cloud execution.",
            )
            detect_target = st.radio(
                "Tracking Point Target:",
                ["Head", "Feet / Ground"],
                horizontal=True,
            )

        with col_cfg2:
            inference_size = st.selectbox(
                "Inference Resolution (px):",
                [320, 640, 960, 1280],
                index=1,
                help="320/640px drastically cuts CPU load while retaining detection quality.",
            )
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.01,
                max_value=0.50,
                value=0.12,
                step=0.01,
                help="Lower confidence reveals faint/partially occluded individuals.",
            )

        with col_cfg3:
            iou_threshold = st.slider(
                "NMS IoU Overlap Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Higher IoU threshold prevents adjacent crowded people from being merged.",
            )
            frame_skip = st.slider(
                "Frame Skip (CPU Saver)",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Process 1 out of N frames during video tracking to prevent CPU throttling.",
            )

    st.markdown("### 🎞️ Single Frame Preview")
    frame_idx = st.slider(
        "Select Frame to Preview",
        min_value=0,
        max_value=max(1, max_frames - 1),
        value=st.session_state.get("selected_frame_idx", 0),
        step=1,
        key="tracking_frame_slider",
    )

    # Extract single raw frame for preview without running YOLO yet
    raw_frame = extract_frame_from_video(uploaded_video, frame_number=frame_idx)
    if raw_frame is None:
        st.error("❌ Failed to decode frame from video. Try re-uploading the video file in Tab 2.1.")
        return

    img_h, img_w, _ = raw_frame.shape

    # Video corners matching standard 4-point order
    video_src_pts = [
        [0, 0],
        [img_w, 0],
        [img_w, img_h],
        [0, img_h],
    ]

    # Calculate Homography Matrix H
    H = compute_homography_matrix(video_src_pts, four_corners[:4])
    st.session_state.homography_matrix = H

    # Manual button trigger to protect CPU from continuous slider inference
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_preview = st.button("🔍 Run Frame Preview", type="primary", use_container_width=True)
    with col_info:
        st.caption("Click button to execute YOLO model on selected frame. Prevents slider CPU throttling.")

    if run_preview:
        with st.spinner("Processing preview frame..."):
            annotated_frame, df_detections = process_video_frame(
                raw_frame,
                H_matrix=H,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                inference_size=inference_size,
                detect_target=detect_target,
                model_name=model_name,
            )
            st.session_state["preview_annotated"] = annotated_frame
            st.session_state["preview_dets"] = df_detections
    else:
        annotated_frame = st.session_state.get("preview_annotated", raw_frame)
        df_detections = st.session_state.get("preview_dets", pd.DataFrame())

    st.markdown("---")

    # Render Side-by-Side Interface
    col_video, col_plan = st.columns(2)

    # 🎥 Left Column: Video Detection Stream
    with col_video:
        st.markdown("#### 📹 Camera Feed (Detections)")
        st.image(
            annotated_frame,
            caption=f"Frame #{frame_idx} | Detected People: {len(df_detections)}",
            use_container_width=True,
        )

    # 🗺️ Right Column: 2D Floorplan Mapping
    with col_plan:
        st.markdown("#### 🗺️ 2D Floorplan Real-Time Map")

        fig_2d = go.Figure()

        # 1. Render CAD Walls
        wall_x, wall_y = [], []
        for line in dxf_walls:
            if hasattr(line, "xy"):
                x, y = line.xy
                wall_x.extend([x[0], x[1], None])
                wall_y.extend([y[0], y[1], None])
            elif isinstance(line, (list, tuple)):
                for i in range(len(line) - 1):
                    wall_x.extend([line[i][0], line[i + 1][0], None])
                    wall_y.extend([line[i][1], line[i + 1][1], None])

        if wall_x:
            fig_2d.add_trace(
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

        # 2. Render ROI Polygon
        px = [p[0] for p in four_corners[:4]] + [four_corners[0][0]]
        py = [p[1] for p in four_corners[:4]] + [four_corners[0][1]]
        fig_2d.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                fill="toself",
                fillcolor="rgba(0, 230, 118, 0.15)",
                line=dict(color="#00FF66", width=2, dash="dash"),
                name="Camera Field of View",
            )
        )

        # 3. Render Detected Human Positions on Floorplan
        if not df_detections.empty and "world_x" in df_detections.columns:
            valid_dets = df_detections.dropna(subset=["world_x", "world_y"])
            if not valid_dets.empty:
                fig_2d.add_trace(
                    go.Scatter(
                        x=valid_dets["world_x"],
                        y=valid_dets["world_y"],
                        mode="markers+text",
                        marker=dict(
                            size=12,
                            color="#FF007F",
                            symbol="circle",
                            line=dict(width=2, color="#FFFFFF"),
                        ),
                        text=[f"ID:{tid}" for tid in valid_dets["track_id"]],
                        textposition="top center",
                        textfont=dict(color="#FF007F", size=12),
                        name="Tracked Occupants",
                    )
                )

        fig_2d.update_layout(
            template="plotly_dark",
            height=500,
            xaxis=dict(
                title="X (meters)",
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                range=st.session_state.get("current_x_range", None),
            ),
            yaxis=dict(
                title="Y (meters)",
                showgrid=True,
                range=st.session_state.get("current_y_range", None),
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_2d, use_container_width=True)

    # Position table
    if not df_detections.empty:
        st.markdown("### 📊 Single Frame Position Coordinates")
        display_df = df_detections[["track_id", "img_x", "img_y", "world_x", "world_y"]].copy()
        display_df.columns = ["Track ID", "Image X (px)", "Image Y (px)", "Floorplan X (m)", "Floorplan Y (m)"]
        st.dataframe(display_df, use_container_width=True)

    # 🚀 Batch Video Sequence Execution
    st.markdown("---")
    st.markdown("### 🎬 Full Video Tracking Execution")

    if st.button("▶️ Run Batch Video Tracking"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_placeholder = st.empty()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
            tmp_v.write(video_bytes)
            tmp_path = tmp_v.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        curr_frame_idx = 0
        all_tracking_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Frame Skipping Logic
            if curr_frame_idx % frame_skip != 0:
                curr_frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ann_frame, df_dets = process_video_frame(
                frame_rgb,
                H_matrix=H,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                inference_size=inference_size,
                detect_target=detect_target,
                model_name=model_name,
            )

            if not df_dets.empty:
                df_dets["frame_idx"] = curr_frame_idx
                all_tracking_results.append(df_dets)

            video_placeholder.image(
                ann_frame,
                caption=f"Processing Frame {curr_frame_idx}/{total_frames} (Frame Skip: {frame_skip})",
                use_container_width=True,
            )

            progress_bar.progress(min(1.0, curr_frame_idx / max(1, total_frames)))
            status_text.text(f"Processed frame {curr_frame_idx} of {total_frames}...")

            curr_frame_idx += 1

        cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        st.success("✅ Full Video Batch Tracking Completed!")

        if all_tracking_results:
            full_df = pd.concat(all_tracking_results, ignore_index=True)
            st.session_state["full_tracking_df"] = full_df

# Inside views/tracking_view.py, right after tracking finishes processing:

# 1. Create/Format your tracking dataframe (e.g., results_df)
st.session_state.tracking_results_df = results_df  # <--- ADD THIS LINE!

# Now when the user downloads the CSV or changes tabs, the data stays in memory
st.download_button(
    label="📥 Download Tracking CSV",
    data=results_df.to_csv(index=False),
    file_name="tracking_results.csv",
    mime="text/csv",
)

    # 📥 Dual Export Options (CSV & JSON with VGA Metadata)
    full_df = st.session_state.get("full_tracking_df", pd.DataFrame())

    if not full_df.empty:
        st.markdown("#### 📥 Export Tracking & VGA Analytics")
        col_dl1, col_dl2 = st.columns(2)

        # 1. Standard CSV Export
        csv_bytes = full_df.to_csv(index=False).encode("utf-8")
        with col_dl1:
            st.download_button(
                label="📄 Download Motion Data (CSV)",
                data=csv_bytes,
                file_name="human_tracking_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 2. Combined JSON Export (Includes VGA Grid Metadata)
        vga_dict = vga_grid_df.to_dict(orient="records") if vga_grid_df is not None else []

        export_payload = {
            "metadata": {
                "four_corners_roi": st.session_state.get("four_corners", []),
                "total_detections": int(len(full_df)),
                "total_frames_processed": int(full_df["frame_idx"].max()) if "frame_idx" in full_df.columns else 0,
                "model_used": model_name,
                "detect_target": detect_target,
            },
            "vga_floorplan_nodes": vga_dict,
            "tracking_points": full_df.to_dict(orient="records"),
        }

        json_bytes = json.dumps(export_payload, indent=2).encode("utf-8")

        with col_dl2:
            st.download_button(
                label="📦 Download Complete Dataset (JSON + VGA)",
                data=json_bytes,
                file_name="human_tracking_and_vga.json",
                mime="application/json",
                use_container_width=True,
            )