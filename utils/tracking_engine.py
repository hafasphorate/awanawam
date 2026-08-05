# utils/tracking_engine.py
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from cv2 import VideoCapture

# Try importing ultralytics YOLO; fallback gracefully if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils.homography_engine import project_points


@st.cache_resource
def load_yolo_model(model_name: str = "yolov8n.pt"):
    """Loads and caches the YOLO model for detection."""
    if not YOLO_AVAILABLE:
        return None
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model ({model_name}): {e}")
        return None


def extract_frame_from_video(video_file, frame_number: int = 0) -> np.ndarray:
    """Extracts an RGB image frame from a Streamlit uploaded video file."""
    if video_file is None:
        return None

    # Write video stream to temp file for OpenCV consumption
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
        tmp_v.write(video_file.getvalue())
        tmp_path = tmp_v.name

    cap = VideoCapture(tmp_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def process_video_frame(
    frame_rgb: np.ndarray,
    H_matrix: np.ndarray = None,
    conf_threshold: float = 0.3,
    detect_target: str = "Head",  # "Head" or "Feet / Ground"
    model_name: str = "yolov8n.pt",
) -> tuple:
    """Detects people in a frame, estimates head/ground positions, and projects them 

    onto floorplan coordinates via Homography.

    Returns
    -------
    tuple: (annotated_frame_rgb, detections_df)
        - annotated_frame_rgb: OpenCV frame with bounding boxes and keypoint markers
        - detections_df: DataFrame containing track/detection IDs, pixel coordinates, and world coordinates (X, Y)
    """
    if frame_rgb is None:
        return None, pd.DataFrame()

    annotated_frame = frame_rgb.copy()
    h, w, _ = annotated_frame.shape
    model = load_yolo_model(model_name)

    detection_list = []
    pixel_points = []

    if model is not None and YOLO_AVAILABLE:
        # Run tracking / detection with YOLO (classes 0 = person)
        results = model.track(
            annotated_frame,
            classes=[0],
            conf=conf_threshold,
            persist=True,
            verbose=False,
        )

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get bounding box coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get tracking ID if present, otherwise default to 0
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else 0

                # Target point calculation
                if detect_target == "Head":
                    # Center of top bounding box edge (head position)
                    target_x = (x1 + x2) / 2.0
                    target_y = y1
                else:
                    # Bottom center (feet/ground contact position)
                    target_x = (x1 + x2) / 2.0
                    target_y = y2

                pixel_points.append([target_x, target_y])

                # Draw bounding box on frame
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 128),
                    2,
                )
                
                # Draw head/target point marker
                cv2.circle(
                    annotated_frame,
                    (int(target_x), int(target_y)),
                    5,
                    (255, 0, 128),
                    -1,
                )

                # Label text
                label = f"ID:{track_id} ({detect_target})"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 128),
                    2,
                )

                detection_list.append({
                    "track_id": track_id,
                    "img_x": target_x,
                    "img_y": target_y,
                    "bbox": [x1, y1, x2, y2],
                    "world_x": None,
                    "world_y": None,
                })

    # Project pixel points into world coordinates if Homography matrix exists
    if H_matrix is not None and len(pixel_points) > 0:
        world_pts = project_points(pixel_points, H_matrix)
        for idx, w_pt in enumerate(world_pts):
            detection_list[idx]["world_x"] = w_pt[0]
            detection_list[idx]["world_y"] = w_pt[1]

    df_detections = pd.DataFrame(detection_list)
    return annotated_frame, df_detections