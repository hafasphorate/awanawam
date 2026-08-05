# utils/tracking_engine.py
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from ultralytics import YOLO, RTDETR
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils.homography_engine import project_points


@st.cache_resource
def load_detection_model(model_name: str = "keremberke/yolov8n-head"):
    """Loads and caches standard YOLO, Head Detectors, or RT-DETR models."""
    if not YOLO_AVAILABLE:
        return None
    try:
        if "rtdetr" in model_name.lower():
            model = RTDETR(model_name)
        else:
            model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model ({model_name}): {e}")
        return None


def process_video_frame(
    frame_rgb: np.ndarray,
    H_matrix: np.ndarray = None,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
    inference_size: int = 1280,
    model_name: str = "keremberke/yolov8n-head",
) -> tuple:
    """Detects people using dedicated Head/Crowd detection models."""
    if frame_rgb is None:
        return None, pd.DataFrame()

    annotated_frame = frame_rgb.copy()

    if not YOLO_AVAILABLE:
        return annotated_frame, pd.DataFrame()

    model = load_detection_model(model_name)
    detection_list = []
    pixel_points = []

    if model is not None:
        results = model.predict(
            annotated_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=inference_size,
            verbose=False,
        )

        if results and len(results) > 0:
            res = results[0]

            # 1. Check for Pose Keypoints (if using pose models)
            if hasattr(res, "keypoints") and res.keypoints is not None and len(res.keypoints) > 0:
                keypoints_data = res.keypoints.xy.cpu().numpy()
                for idx, kpts in enumerate(keypoints_data):
                    valid_head_pts = [pt for pt in kpts[:5] if pt[0] > 0 and pt[1] > 0]
                    if len(valid_head_pts) > 0:
                        target_x = float(np.mean([pt[0] for pt in valid_head_pts]))
                        target_y = float(np.mean([pt[1] for pt in valid_head_pts]))
                        pixel_points.append([target_x, target_y])
                        cv2.circle(annotated_frame, (int(target_x), int(target_y)), 5, (255, 0, 128), -1)
                        detection_list.append({"track_id": idx + 1, "img_x": target_x, "img_y": target_y})

            # 2. Check for Bounding Boxes (Head Detectors / RT-DETR / Standard YOLO)
            elif hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes
                for idx, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Center point of bounding box (ideal for head detectors)
                    target_x = (x1 + x2) / 2.0
                    target_y = (y1 + y2) / 2.0

                    pixel_points.append([target_x, target_y])

                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 128), 2)
                    cv2.circle(annotated_frame, (int(target_x), int(target_y)), 4, (255, 0, 128), -1)

                    detection_list.append({"track_id": idx + 1, "img_x": target_x, "img_y": target_y})

    # Project to 2D Floorplan
    if H_matrix is not None and len(pixel_points) > 0:
        world_pts = project_points(pixel_points, H_matrix)
        for idx, w_pt in enumerate(world_pts):
            detection_list[idx]["world_x"] = w_pt[0]
            detection_list[idx]["world_y"] = w_pt[1]

    df_detections = pd.DataFrame(detection_list)
    return annotated_frame, df_detections