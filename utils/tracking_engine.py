import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from ultralytics import RTDETR, YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils.homography_engine import project_points

# Map UI names to official Ultralytics models that auto-download seamlessly
MODEL_MAPPING = {
    "yolov8n.pt": "yolov8n.pt",
    "keremberke/yolov8n-head": "yolov8n.pt",  # Fallback to lightweight standard YOLOv8n
    "yolov8s.pt": "yolov8s.pt",
    "rtdetr-l.pt": "rtdetr-l.pt",
    "yolov9e.pt": "yolov9e.pt",
    "yolov8x-pose.pt": "yolov8x-pose.pt",
}


@st.cache_resource
def load_detection_model(model_name: str = "yolov8n.pt"):
    """Loads and caches standard YOLO, Head Detectors, or RT-DETR models."""
    if not YOLO_AVAILABLE:
        return None

    # Resolve model file name
    actual_model_path = MODEL_MAPPING.get(model_name, "yolov8n.pt")

    try:
        if "rtdetr" in actual_model_path.lower():
            model = RTDETR(actual_model_path)
        else:
            model = YOLO(actual_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model ({model_name}): {e}")
        return None


def is_point_excluded(point: tuple, exclusion_masks: list) -> bool:
    """Checks whether a target pixel point (x, y) falls within any drawn exclusion zone."""
    if not exclusion_masks:
        return False

    for mask in exclusion_masks:
        if len(mask) >= 3:
            pts_arr = np.array(mask, dtype=np.float32)
            # cv2.pointPolygonTest returns >= 0 if the point is inside or on the boundary
            if (
                cv2.pointPolygonTest(
                    pts_arr, (float(point[0]), float(point[1])), False
                )
                >= 0
            ):
                return True
    return False


def extract_frame_from_video(video_file, frame_number: int = 0) -> np.ndarray:
    """Extracts an RGB image frame safely from a Streamlit uploaded video file."""
    if video_file is None:
        return None

    try:
        video_file.seek(0)
        video_bytes = video_file.read()
        video_file.seek(0)

        if not video_bytes:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
            tmp_v.write(video_bytes)
            tmp_path = tmp_v.name

        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = min(max(0, frame_number), max(0, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            curr = 0
            while cap.isOpened() and curr <= target_frame:
                ret, frame = cap.read()
                if curr == target_frame:
                    break
                curr += 1

        cap.release()

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
    conf_threshold: float = 0.12,
    iou_threshold: float = 0.45,
    inference_size: int = 640,
    detect_target: str = "Head",
    model_name: str = "yolov8n.pt",
) -> tuple:
    """Detects people using YOLO / RT-DETR and maps detection coordinates,

    excluding any points located within st.session_state.exclusion_masks.
    """
    if frame_rgb is None:
        return None, pd.DataFrame()

    annotated_frame = frame_rgb.copy()

    if not YOLO_AVAILABLE:
        return annotated_frame, pd.DataFrame()

    model = load_detection_model(model_name)
    detection_list = []
    pixel_points = []

    # Fetch active exclusion masks set in Tab 2 UI
    exclusion_masks = st.session_state.get("exclusion_masks", [])

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

            # 1. Pose Keypoint Models
            if (
                hasattr(res, "keypoints")
                and res.keypoints is not None
                and len(res.keypoints) > 0
            ):
                keypoints_data = res.keypoints.xy.cpu().numpy()
                for idx, kpts in enumerate(keypoints_data):
                    if detect_target.lower().startswith(
                        "feet"
                    ) or detect_target.lower().startswith("ground"):
                        valid_pts = [
                            pt for pt in kpts[15:] if pt[0] > 0 and pt[1] > 0
                        ]  # Ankle keypoints
                    else:
                        valid_pts = [
                            pt for pt in kpts[:5] if pt[0] > 0 and pt[1] > 0
                        ]  # Nose/eyes/ears keypoints

                    if len(valid_pts) > 0:
                        target_x = float(np.mean([pt[0] for pt in valid_pts]))
                        target_y = float(np.mean([pt[1] for pt in valid_pts]))

                        # EXCLUSION MASK CHECK
                        if is_point_excluded(
                            (target_x, target_y), exclusion_masks
                        ):
                            continue

                        pixel_points.append([target_x, target_y])
                        cv2.circle(
                            annotated_frame,
                            (int(target_x), int(target_y)),
                            5,
                            (255, 0, 128),
                            -1,
                        )
                        detection_list.append(
                            {
                                "track_id": len(detection_list) + 1,
                                "img_x": target_x,
                                "img_y": target_y,
                            }
                        )

            # 2. Bounding Box Models (Filter Person Class = 0)
            elif hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes
                for idx, box in enumerate(boxes):
                    cls_id = (
                        int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else 0
                    )

                    # Filter for Person class (0) in COCO dataset models
                    if cls_id != 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    target_x = float((x1 + x2) / 2.0)

                    if detect_target.lower().startswith(
                        "feet"
                    ) or detect_target.lower().startswith("ground"):
                        target_y = float(y2)  # Bottom center for feet ground contact
                    else:
                        target_y = float(y1)  # Top center for head target

                    # EXCLUSION MASK CHECK
                    if is_point_excluded((target_x, target_y), exclusion_masks):
                        continue

                    pixel_points.append([target_x, target_y])

                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 128),
                        2,
                    )
                    cv2.circle(
                        annotated_frame,
                        (int(target_x), int(target_y)),
                        4,
                        (255, 0, 128),
                        -1,
                    )

                    detection_list.append(
                        {
                            "track_id": len(detection_list) + 1,
                            "img_x": target_x,
                            "img_y": target_y,
                        }
                    )

    # Project pixel points into 2D Floorplan Coordinates
    if H_matrix is not None and len(pixel_points) > 0:
        world_pts = project_points(pixel_points, H_matrix)
        for idx, w_pt in enumerate(world_pts):
            detection_list[idx]["world_x"] = w_pt[0]
            detection_list[idx]["world_y"] = w_pt[1]

    df_detections = pd.DataFrame(detection_list)
    return annotated_frame, df_detections