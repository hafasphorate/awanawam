import os
import re
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

MODEL_MAPPING = {
    "yolov8n.pt": "yolov8n.pt",
    "keremberke/yolov8n-head": "yolov8n.pt",
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


def parse_polygon_mask(mask) -> list:
    """Extracts (x, y) float coordinate pairs from various shape formats."""
    pts = []

    if isinstance(mask, str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", mask)
        if len(nums) >= 6:
            coords = [float(n) for n in nums]
            pts = [[coords[i], coords[i + 1]] for i in range(0, len(coords) - 1, 2)]

    elif isinstance(mask, dict):
        if "x" in mask and "y" in mask:
            pts = [[float(x), float(y)] for x, y in zip(mask["x"], mask["y"])]
        elif "path" in mask and isinstance(mask["path"], str):
            return parse_polygon_mask(mask["path"])

    elif isinstance(mask, (list, tuple)):
        for item in mask:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pts.append([float(item[0]), float(item[1])])
            elif isinstance(item, dict) and "x" in item and "y" in item:
                pts.append([float(item["x"]), float(item["y"])])

    return pts


def get_projected_exclusion_polygons(
    video_masks: list, frame_shape: tuple, H_matrix: np.ndarray
) -> list:
    """Takes polygons drawn on the video image, scales them to native resolution,

    and transforms them into 2D floorplan space via the Homography matrix.
    """
    if H_matrix is None or not video_masks:
        return []

    frame_h, frame_w = frame_shape[:2]
    canvas_w = st.session_state.get("mask_canvas_width", frame_w)
    canvas_h = st.session_state.get("mask_canvas_height", frame_h)

    scale_x = frame_w / float(canvas_w) if canvas_w > 0 else 1.0
    scale_y = frame_h / float(canvas_h) if canvas_h > 0 else 1.0

    projected_polygons = []

    for mask in video_masks:
        raw_pts = parse_polygon_mask(mask)
        if len(raw_pts) >= 3:
            # 1. Scale points to full frame resolution
            scaled_pts = [[pt[0] * scale_x, pt[1] * scale_y] for pt in raw_pts]

            # 2. Map video pixels -> floorplan 2D coordinates using Homography matrix
            world_poly_pts = project_points(scaled_pts, H_matrix)

            if len(world_poly_pts) >= 3:
                projected_polygons.append(
                    np.array(world_poly_pts, dtype=np.float32)
                )

    return projected_polygons


def is_world_point_in_exclusion(
    world_x: float, world_y: float, projected_polygons: list
) -> bool:
    """Checks if a 2D floorplan point falls inside any projected video exclusion zone."""
    for poly_arr in projected_polygons:
        if (
            cv2.pointPolygonTest(poly_arr, (float(world_x), float(world_y)), False)
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
    """Detects people, maps them to 2D floorplan space, and excludes detections

    whose 2D floorplan coordinates land inside video-drawn exclusion zones.
    """
    if frame_rgb is None:
        return None, pd.DataFrame()

    annotated_frame = frame_rgb.copy()

    if not YOLO_AVAILABLE:
        return annotated_frame, pd.DataFrame()

    model = load_detection_model(model_name)
    raw_detections = []
    pixel_points = []

    # Get raw video exclusion masks from session state
    video_masks = st.session_state.get("exclusion_masks", [])

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
                        ]
                    else:
                        valid_pts = [
                            pt for pt in kpts[:5] if pt[0] > 0 and pt[1] > 0
                        ]

                    if len(valid_pts) > 0:
                        target_x = float(np.mean([pt[0] for pt in valid_pts]))
                        target_y = float(np.mean([pt[1] for pt in valid_pts]))

                        pixel_points.append([target_x, target_y])
                        raw_detections.append(
                            {
                                "img_x": target_x,
                                "img_y": target_y,
                                "bbox": None,
                            }
                        )

            # 2. Bounding Box Models (Person Class = 0)
            elif hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes
                for idx, box in enumerate(boxes):
                    cls_id = (
                        int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else 0
                    )

                    if cls_id != 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    target_x = float((x1 + x2) / 2.0)

                    if detect_target.lower().startswith(
                        "feet"
                    ) or detect_target.lower().startswith("ground"):
                        target_y = float(y2)
                    else:
                        target_y = float(y1)

                    pixel_points.append([target_x, target_y])
                    raw_detections.append(
                        {
                            "img_x": target_x,
                            "img_y": target_y,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        }
                    )

    final_detections = []

    if H_matrix is not None and len(pixel_points) > 0:
        # Project raw pixel detection points to 2D floorplan space
        world_pts = project_points(pixel_points, H_matrix)

        # Convert video-drawn masks to floorplan 2D polygons using H_matrix
        projected_exclusion_zones = get_projected_exclusion_polygons(
            video_masks, frame_rgb.shape, H_matrix
        )

        for idx, w_pt in enumerate(world_pts):
            world_x, world_y = float(w_pt[0]), float(w_pt[1])

            # Filter out points that fall inside the projected 2D exclusion zones
            if is_world_point_in_exclusion(
                world_x, world_y, projected_exclusion_zones
            ):
                continue

            det_info = raw_detections[idx]
            det_info["world_x"] = world_x
            det_info["world_y"] = world_y
            det_info["track_id"] = len(final_detections) + 1
            final_detections.append(det_info)

            # Draw only valid, non-excluded detections on video
            if det_info["bbox"]:
                x1, y1, x2, y2 = det_info["bbox"]
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2), (0, 255, 128), 2
                )
            cv2.circle(
                annotated_frame,
                (int(det_info["img_x"]), int(det_info["img_y"])),
                4,
                (255, 0, 128),
                -1,
            )
    else:
        for idx, det_info in enumerate(raw_detections):
            det_info["track_id"] = idx + 1
            final_detections.append(det_info)
            if det_info["bbox"]:
                x1, y1, x2, y2 = det_info["bbox"]
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2), (0, 255, 128), 2
                )

    df_detections = pd.DataFrame(final_detections)
    if "bbox" in df_detections.columns:
        df_detections = df_detections.drop(columns=["bbox"])

    return annotated_frame, df_detections