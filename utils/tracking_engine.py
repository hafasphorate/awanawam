# utils/tracking_engine.py
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Try importing ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils.homography_engine import project_points


@st.cache_resource
def load_yolo_model(model_name: str = "yolov8n-pose.pt"):
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
    """Extracts an RGB image frame safely from a Streamlit uploaded video file."""
    if video_file is None:
        return None

    try:
        # Reset stream pointer to beginning
        video_file.seek(0)
        video_bytes = video_file.read()
        video_file.seek(0)

        if not video_bytes:
            return None

        # Create a named temp file and explicitly close it so OpenCV can open it cleanly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
            tmp_v.write(video_bytes)
            tmp_path = tmp_v.name

        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

        # Get total frames to prevent out-of-bounds frame seeking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = min(max(0, frame_number), max(0, total_frames - 1))

        # Set position
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        # Fallback: If position seek failed, read sequentially
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            curr = 0
            while cap.isOpened() and curr <= target_frame:
                ret, frame = cap.read()
                if curr == target_frame:
                    break
                curr += 1

        cap.release()

        # Clean up temporary file safely
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    except Exception as e:
        st.error(f"Video Decoding Error: {e}")

    return None


def process_video_frame(
    frame_rgb: np.ndarray,
    H_matrix: np.ndarray = None,
    conf_threshold: float = 0.25,
    detect_target: str = "Head",
    model_name: str = "yolov8n-pose.pt",
) -> tuple:
    """Detects people in a frame using bounding boxes or pose keypoints, 

    estimates head/ground positions, and projects them onto floorplan coordinates.
    """
    if frame_rgb is None:
        return None, pd.DataFrame()

    annotated_frame = frame_rgb.copy()
    h, w, _ = annotated_frame.shape

    if not YOLO_AVAILABLE:
        cv2.putText(
            annotated_frame,
            "Ultralytics YOLO not installed!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )
        return annotated_frame, pd.DataFrame()

    model = load_yolo_model(model_name)
    detection_list = []
    pixel_points = []

    if model is not None:
        # Run inference / tracking
        results = model.track(
            annotated_frame,
            conf=conf_threshold,
            persist=True,
            verbose=False,
        )

        if results and len(results) > 0:
            res = results[0]

            # MODE A: Pose Model Detections (Best for Overhead/Heads)
            if hasattr(res, "keypoints") and res.keypoints is not None and len(res.keypoints) > 0:
                keypoints_data = res.keypoints.xy.cpu().numpy()
                boxes = res.boxes if hasattr(res, "boxes") else None

                for idx, kpts in enumerate(keypoints_data):
                    track_id = idx + 1
                    if boxes is not None and idx < len(boxes) and boxes[idx].id is not None:
                        track_id = int(boxes[idx].id[0].cpu().numpy())

                    # Pose keypoint indices: 0 = Nose, 1 = Left Eye, 2 = Right Eye, 3 = Left Ear, 4 = Right Ear
                    valid_head_pts = [pt for pt in kpts[:5] if pt[0] > 0 and pt[1] > 0]

                    if detect_target == "Head" and len(valid_head_pts) > 0:
                        target_x = float(np.mean([pt[0] for pt in valid_head_pts]))
                        target_y = float(np.mean([pt[1] for pt in valid_head_pts]))
                    elif boxes is not None and idx < len(boxes):
                        # Fallback to bbox
                        xyxy = boxes[idx].xyxy[0].cpu().numpy()
                        target_x = float((xyxy[0] + xyxy[2]) / 2.0)
                        target_y = float(xyxy[1] if detect_target == "Head" else xyxy[3])
                    else:
                        continue

                    pixel_points.append([target_x, target_y])

                    # Draw head point
                    cv2.circle(annotated_frame, (int(target_x), int(target_y)), 7, (255, 0, 128), -1)
                    cv2.putText(
                        annotated_frame,
                        f"ID:{track_id}",
                        (int(target_x) + 8, int(target_y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 128),
                        2,
                    )

                    detection_list.append({
                        "track_id": track_id,
                        "img_x": target_x,
                        "img_y": target_y,
                        "world_x": None,
                        "world_y": None,
                    })

            # MODE B: Standard Object Bounding Box Detections
            elif hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes
                for box in boxes:
                    # Filter for Person class (class_id == 0)
                    cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    if cls_id != 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else 0

                    target_x = (x1 + x2) / 2.0
                    target_y = y1 if detect_target == "Head" else y2

                    pixel_points.append([target_x, target_y])

                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 128), 2)
                    cv2.circle(annotated_frame, (int(target_x), int(target_y)), 6, (255, 0, 128), -1)
                    cv2.putText(
                        annotated_frame,
                        f"ID:{track_id}",
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
                        "world_x": None,
                        "world_y": None,
                    })

    # Project pixel points into world coordinates if Homography matrix H exists
    if H_matrix is not None and len(pixel_points) > 0:
        world_pts = project_points(pixel_points, H_matrix)
        for idx, w_pt in enumerate(world_pts):
            detection_list[idx]["world_x"] = w_pt[0]
            detection_list[idx]["world_y"] = w_pt[1]

    df_detections = pd.DataFrame(detection_list)
    return annotated_frame, df_detections