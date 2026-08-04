# utils/homography_engine.py
import cv2
import numpy as np
from typing import List, Tuple, Dict

class HomographyCalibrator:
    def __init__(self):
        self.H_matrix: np.ndarray = None

    def compute_homography(
        self, 
        img_points: List[Tuple[float, float]], 
        cad_points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Computes 3x3 Homography Matrix H from matching point pairs.
        img_points: [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]
        cad_points: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
        """
        if len(img_points) < 4 or len(cad_points) < 4:
            raise ValueError("At least 4 corresponding point pairs are required.")

        pts_src = np.array(img_points, dtype=np.float32)
        pts_dst = np.array(cad_points, dtype=np.float32)

        # Compute Homography using RANSAC for robustness against minor point inaccuracies
        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        self.H_matrix = H
        return H

    def transform_point(self, u: float, v: float) -> Tuple[float, float]:
        """Transforms a single (u, v) video pixel coordinate to (X, Y) CAD space."""
        if self.H_matrix is None:
            raise RuntimeError("Homography matrix not calibrated yet.")

        pt_pixel = np.array([[[u, v]]], dtype=np.float32)
        pt_cad = cv2.perspectiveTransform(pt_pixel, self.H_matrix)
        
        return float(pt_cad[0][0][0]), float(pt_cad[0][0][1])

    def transform_batch_feet(self, bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float]]:
        """
        Takes bounding boxes [x1, y1, x2, y2] from object detector,
        extracts bottom-center (foot position: (x1+x2)/2, y2),
        and returns CAD coordinates [(X, Y), ...].
        """
        if not bboxes or self.H_matrix is None:
            return []

        # Foot positions (bottom center of bounding boxes)
        feet_pixels = np.array([
            [[(box[0] + box[2]) / 2.0, box[3]]] for box in bboxes
        ], dtype=np.float32)

        cad_coords = cv2.perspectiveTransform(feet_pixels, self.H_matrix)
        
        return [(float(pt[0][0]), float(pt[0][1])) for pt in cad_coords]