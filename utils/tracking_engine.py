# utils/tracking_engine.py
import tempfile
import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple, Optional
import streamlit as st


def extract_frame_from_video(
    uploaded_video_file, frame_number: int = 0
) -> Optional[np.ndarray]:
    """
    Extracts a specific RGB frame from a Streamlit UploadedFile object.
    
    Uses a temporary file buffer since OpenCV's VideoCapture requires a disk path.
    """
    if uploaded_video_file is None:
        return None

    # Write video bytes to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video_file.getbuffer())
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Error opening uploaded video file.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        st.error("Uploaded video contains no frames.")
        cap.release()
        return None

    # Clamp frame_number to valid video index range
    frame_number = max(0, min(frame_number, total_frames - 1))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        st.error(f"Failed to read frame index {frame_number} from video.")
        return None

    # OpenCV defaults to BGR -> Convert to RGB for Plotly/Streamlit display
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class HomographyCalibrator:
    """Handles 3x3 Homography Matrix calculation and point transformations."""

    def __init__(self, H_matrix: Optional[np.ndarray] = None):
        self.H_matrix = H_matrix

    def compute_homography(
        self,
        img_points: List[Tuple[float, float]],
        cad_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Computes 3x3 Homography Matrix H from matching point pairs.
        img_points: [(u1, v1), (u2, v2), ...]
        cad_points: [(X1, Y1), (X2, Y2), ...]
        """
        if len(img_points) < 4 or len(cad_points) < 4:
            raise ValueError("At least 4 corresponding point pairs are required.")

        pts_src = np.array(img_points, dtype=np.float32)
        pts_dst = np.array(cad_points, dtype=np.float32)

        # Compute Homography using RANSAC for robustness against minor point inaccuracies
        H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        self.H_matrix = H
        return H

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transforms N x 2 pixel points (u, v) into CAD coordinates (X, Y)."""
        if self.H_matrix is None:
            raise RuntimeError("Homography matrix is not calibrated.")

        pts_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        cad_pts = cv2.perspectiveTransform(pts_reshaped, self.H_matrix)
        return cad_pts.reshape(-1, 2)


class TrackingProcessor:
    """Parses tracking logs and computes spatial occupancy heatmaps."""

    def __init__(self, homography_matrix: np.ndarray):
        if homography_matrix is None or homography_matrix.shape != (3, 3):
            raise ValueError("A valid 3x3 Homography Matrix is required.")
        self.calibrator = HomographyCalibrator(homography_matrix)

    def parse_and_transform_csv(
        self, df: pd.DataFrame, tracking_point: str = "Head (Top-Center)"
    ) -> pd.DataFrame:
        """
        Parses tracking DataFrame and appends transformed CAD_X and CAD_Y.
        Expected columns: ['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2']
        """
        required_cols = {"x1", "y1", "x2", "y2"}
        if not required_cols.issubset(df.columns):
            raise KeyError(
                f"CSV missing required bounding box columns: {required_cols - set(df.columns)}"
            )

        df = df.copy()

        # Extract horizontal center
        u = (df["x1"] + df["x2"]) / 2.0

        # Head mode picks top edge (y1) to handle crowded scenes; Feet picks bottom edge (y2)
        if tracking_point == "Head (Top-Center)":
            v = df["y1"]
        else:
            v = df["y2"]

        pixel_pts = np.column_stack((u, v))
        cad_pts = self.calibrator.transform_points(pixel_pts)

        df["point_u"] = u
        df["point_v"] = v
        df["CAD_X"] = cad_pts[:, 0]
        df["CAD_Y"] = cad_pts[:, 1]

        return df

    @staticmethod
    def compute_occupancy_density(
        cad_x: np.ndarray,
        cad_y: np.ndarray,
        grid_bounds: Tuple[float, float, float, float],
        nbins: int = 80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a 2D spatial density histogram for Plotly contour mapping.
        grid_bounds: (min_x, max_x, min_y, max_y)
        """
        min_x, max_x, min_y, max_y = grid_bounds

        density, xedges, yedges = np.histogram2d(
            cad_x,
            cad_y,
            bins=nbins,
            range=[[min_x, max_x], [min_y, max_y]],
        )

        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        return x_centers, y_centers, density.T