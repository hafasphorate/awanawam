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
    """Loads and caches YOLO or RT-DETR models."""
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


def extract_raw_shapes_from_session() -> list:
    """Attempts to retrieve drawn shapes across the common session state keys."""
    for key in [
        "exclusion_masks",
        "mask_polygons",
        "drawn_masks",
        "canvas_objects",
    ]:
        masks = st.session_state.get(key, [])
        if isinstance(masks, list) and masks:
            return masks

    canvas_data = st.session_state.get("canvas_result", None)
    if canvas_data and isinstance(canvas_data, dict):
        json_data = canvas_data.get("json_data", {})
        if "objects" in json_data:
            return json_data["objects"]

    return []


def parse_polygon_mask(mask) -> list:
    """Parses SVG path strings, canvas objects, dictionaries, and raw coordinate lists."""
    pts = []

    # Case A: SVG Path String (e.g., Fabric.js canvas paths 'M 10 20 L 30 40 ... Z')
    if isinstance(mask, str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", mask)
        if len(nums) >= 6:
            coords = [float(n) for n in nums]
            pts = [[coords[i], coords[i + 1]] for i in range(0, len(coords) - 1, 2)]

    # Case B: Dictionary (e.g., Fabric.js path/rect object)
    elif isinstance(mask, dict):
        # SVG path inside fabric object
        if "path" in mask and isinstance(mask["path"], list):
            for cmd in mask["path"]:
                if len(cmd) >= 3 and isinstance(cmd[1], (int, float)) and isinstance(cmd[2], (int, float)):
                    pts.append([float(cmd[1]), float(cmd[2])])
        elif "path" in mask and isinstance(mask["path"], str):
            return parse_polygon_mask(mask["path"])
        
        # Rectangles (left, top, width, height)
        elif "left" in mask and "top" in mask and "width" in mask and "height" in mask:
            l, t, w, h = float(mask["left"]), float(mask["top"]), float(mask["width"]), float(mask["height"])
            pts = [[l, t], [l + w, t], [l + w, t + h], [l, t + h]]
            
        # Point arrays
        elif "x" in mask and "y" in mask:
            pts = [[float(x), float(y)] for x, y in zip(mask["x"], mask["y"])]

    # Case C: List of points
    elif isinstance(mask, (list, tuple)):
        for item in mask:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pts.append([float(item[0]), float(item[1])])
            elif isinstance(item, dict):
                p = parse_polygon_mask(item)
                pts.extend(p)

    return pts


def apply_exclusion_masks_to_frame(
    frame_rgb: np.ndarray, video_masks: list, fill_color: tuple = (0, 0, 0)
) -> np.ndarray:
    """Fills drawn polygon regions on the video image with a solid color to block detection."""
    if not video_masks:
        return frame_rgb.copy()

    masked_frame = frame_rgb.copy()
    frame_h, frame_w = frame_rgb.shape[:2]

    canvas_w = st.session_state.get("mask_canvas_width", frame_w)
    canvas_h = st.session_state.get("mask_canvas_height", frame_h)
    scale_x = frame_w / float(canvas_w) if canvas_w > 0 else 1.0
    scale_y = frame_h / float(canvas_h) if canvas_h > 0 else 1.0

    for mask in video_masks:
        raw_pts = parse_polygon_mask(mask)
        if len(raw_pts) >= 3:
            pts = []
            for pt in raw_pts:
                x = float(pt[0]) * scale_x
                y = float(pt[1]) * scale_y
                pts.append([int(np.clip(x, 0, frame_w - 1)), int(np.clip(y, 0, frame_h - 1))])

            if len(pts) >= 3:
                cv2.fillPoly(masked_frame, np.array([pts], dtype=np.int32), fill_color)

    return masked_frame


def extract_frame_from_video(video_file, frame_number: int = 0) -> np.ndarray:
    """Extracts an RGB frame safely from an uploaded video file."""
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
    """Applies blackout masks directly onto the frame, runs YOLO/RT-DETR, and maps detections to 2D space."""
    if frame_rgb is None:
        return None, pd.DataFrame()

    if not YOLO_AVAILABLE:
        return frame_rgb.copy(), pd.DataFrame()

    # Frame annotations are reference-only in the homography workflow.
    video_masks = (
        extract_raw_shapes_from_session()
        if st.session_state.get("use_exclusion_masks", False)
        else []
    )

    # 2. APPLY BLACKOUT MASKS BEFORE AI DETECTION
    inference_frame = apply_exclusion_masks_to_frame(
        frame_rgb, video_masks, fill_color=(0, 0, 0)
    )

    annotated_frame = inference_frame.copy()
    model = load_detection_model(model_name)
    raw_detections = []
    pixel_points = []

    # 3. RUN MODEL PREDICTION ON THE BLACKED-OUT FRAME
    if model is not None:
        results = model.predict(
            inference_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=inference_size,
            verbose=False,
        )

        if results and len(results) > 0:
            res = results[0]

            # Pose Keypoint Models
            if (
                hasattr(res, "keypoints")
                and res.keypoints is not None
                and len(res.keypoints) > 0
            ):
                keypoints_data = res.keypoints.xy.cpu().numpy()
                for idx, kpts in enumerate(keypoints_data):
                    if detect_target.lower().startswith("feet") or detect_target.lower().startswith("ground"):
                        valid_pts = [pt for pt in kpts[15:] if pt[0] > 0 and pt[1] > 0]
                    else:
                        valid_pts = [pt for pt in kpts[:5] if pt[0] > 0 and pt[1] > 0]

                    if len(valid_pts) > 0:
                        target_x = float(np.mean([pt[0] for pt in valid_pts]))
                        target_y = float(np.mean([pt[1] for pt in valid_pts]))

                        pixel_points.append([target_x, target_y])
                        raw_detections.append(
                            {"img_x": target_x, "img_y": target_y, "bbox": None}
                        )

            # Bounding Box Models (Person Class = 0)
            elif hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else 0
                    if cls_id != 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    target_x = float((x1 + x2) / 2.0)

                    if detect_target.lower().startswith("feet") or detect_target.lower().startswith("ground"):
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

    # 4. PROJECT DETECTIONS TO 2D FLOORPLAN
    final_detections = []

    if H_matrix is not None and len(pixel_points) > 0:
        world_pts = project_points(pixel_points, H_matrix)

        for idx, w_pt in enumerate(world_pts):
            det_info = raw_detections[idx]
            det_info["world_x"] = float(w_pt[0])
            det_info["world_y"] = float(w_pt[1])
            det_info["track_id"] = len(final_detections) + 1
            final_detections.append(det_info)

            if det_info["bbox"]:
                x1, y1, x2, y2 = det_info["bbox"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
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
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 128), 2)

    df_detections = pd.DataFrame(final_detections)
    if "bbox" in df_detections.columns:
        df_detections = df_detections.drop(columns=["bbox"])

    return annotated_frame, df_detections