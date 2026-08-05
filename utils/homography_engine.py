# utils/homography_engine.py
import cv2
import numpy as np


def compute_homography_matrix(src_points: list, dst_points: list) -> np.ndarray:
    """Calculates the 3x3 Homography Matrix given source (camera image) 

    and destination (world floorplan) point pairs.

    Parameters
    ----------
    src_points : list
        List of [x, y] coordinates selected on the video frame (pixels).
    dst_points : list
        List of [x, y] coordinates corresponding on the floorplan (meters/CAD units).

    Returns
    -------
    np.ndarray or None
        3x3 Homography transformation matrix H.
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        return None

    src_pts = np.float32(src_points[:4]).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_points[:4]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def project_points(points: list, H: np.ndarray) -> list:
    """Transforms 2D points from source frame coordinates to floorplan world coordinates

    using homography matrix H.

    Parameters
    ----------
    points : list
        List of [x, y] points (e.g. detected head/feet locations in pixels).
    H : np.ndarray
        3x3 Homography matrix.

    Returns
    -------
    list
        List of transformed [X, Y] world coordinates on the floorplan.
    """
    if H is None or len(points) == 0:
        return []

    pts_array = np.float32(points).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts_array, H)

    # Flatten back to list of [X, Y]
    return transformed_pts.reshape(-1, 2).tolist()


def project_single_point(x: float, y: float, H: np.ndarray) -> tuple:
    """Convenience helper to project a single (x, y) point."""
    res = project_points([[x, y]], H)
    if res:
        return res[0][0], res[0][1]
    return None, None