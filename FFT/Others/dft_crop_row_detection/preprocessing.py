"""Image pre-processing from Section 2.2.1 of the paper.

Implements
  * Excess Green Index (ExG) grayscale conversion,
  * camera undistortion (cv2),
  * homography-based inverse perspective mapping (IPM), eq. (1)-(8),
  * maximum axis-aligned inscribed rectangle (MAAIR) ROI extraction,
  * the ROI <-> local map frame transform (scale + translation).

The DFT core in row_detection.py does not require calibration data; this
module provides the documented utility for turning a raw camera image into the
rectified bird's-eye view the core expects. In the tests/demo we feed already
rectified images (synthetic or the aerial example) straight into the core.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------

def exg_gray(rgb: np.ndarray) -> np.ndarray:
    """Excess Green Index: ExG = 2g - r - b with normalized channels (Yang et
    al. 2015), mapped to [0, 255]."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim == 2:
        return rgb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s = r + g + b
    s = np.where(s == 0, 1.0, s)
    exg = 2.0 * g / s - r / s - b / s
    lo, hi = exg.min(), exg.max()
    if hi - lo > 1e-9:
        exg = (exg - lo) / (hi - lo)
    return exg * 255.0


# ---------------------------------------------------------------------------
# Homography / inverse perspective mapping (eq. 2-8)
# ---------------------------------------------------------------------------

def euler_to_rotation(pitch: float, yaw: float, roll: float,
                      degrees: bool = True) -> np.ndarray:
    """Rotation matrix from (pitch, yaw, roll) using the XYZ convention.

    Camera mount convention: pitch = downward tilt (positive tilts the optical
    axis toward the ground), yaw around vertical, roll around optical axis.
    """
    if degrees:
        pitch, yaw, roll = map(np.radians, (pitch, yaw, roll))
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def homography_from_extrinsics(K: np.ndarray, R: np.ndarray,
                               t: np.ndarray) -> np.ndarray:
    """Build the plane homography H = K [r1 r2 t] (eq. 7)."""
    r1, r2 = R[:, 0], R[:, 1]
    return K @ np.column_stack([r1, r2, t])


def apply_ipm(image: np.ndarray, H: np.ndarray,
              out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse perspective mapping via homography H (eq. 8).

    Maps image pixels to the ground plane: [x_m, y_m, 1]^T ~ H^-1 [x_i, y_i, 1]^T.

    Returns
    -------
    warped : out_size array (bird's-eye view)
    valid  : out_size boolean mask of pixels with a source sample
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for IPM")
    warped = cv2.warpPerspective(image, np.linalg.inv(H), out_size,
                                 flags=cv2.INTER_LINEAR)
    # valid mask: warp a white image with border value 0
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(mask, np.linalg.inv(H), out_size,
                                borderValue=0) > 0
    return warped, valid


# ---------------------------------------------------------------------------
# MAAIR: maximum axis-aligned inscribed rectangle of a binary mask
# ---------------------------------------------------------------------------

def _largest_rectangle_in_histogram(heights: np.ndarray) -> tuple[int, int, int]:
    """Largest rectangle in a histogram; returns (start, height, area)."""
    stack = []
    best_start, best_h, best_area = 0, 0, 0
    h = heights
    for i in range(len(h) + 1):
        cur = h[i] if i < len(h) else 0
        while stack and cur < h[stack[-1]]:
            height = h[stack.pop()]
            start = stack[-1] + 1 if stack else 0
            area = height * (i - start)
            if area > best_area:
                best_area, best_start, best_h = area, start, height
        stack.append(i)
    return best_start, best_h, best_area


def maair(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Maximum axis-aligned inscribed rectangle of a boolean mask.

    Returns (row0, col0, height, width). Uses the classic per-row histogram /
    largest-rectangle-in-histogram method over all rows.
    """
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)  # row0, col0, h, w, area
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        start, h, area = _largest_rectangle_in_histogram(heights)
        if area > best[4]:
            best = (r - h + 1, start, h, area // h, area)
    row0, col0, h, w, _ = best
    return row0, col0, h, w


# ---------------------------------------------------------------------------
# ROI <-> map frame transforms
# ---------------------------------------------------------------------------

def roi_to_map_transform(scale: float, c_x: float, c_y: float) -> np.ndarray:
    """ROI-pixel -> map-metre homogeneous affine transform.

    Convention: pixel (x, y) -> map (X, Y) with
        X = scale * (y - c_y),  Y = scale * (c_x - x)
    i.e. +Y is forward (up the image), +X to the right; (c_x, c_y) is the
    origin of the local map frame in pixel coordinates.
    """
    return np.array([[0.0, scale, -scale * c_y],
                     [-scale, 0.0, scale * c_x],
                     [0.0, 0.0, 1.0]])


def map_to_roi_transform(scale: float, c_x: float, c_y: float) -> np.ndarray:
    """Inverse of roi_to_map_transform (map-metre -> ROI-pixel)."""
    return np.linalg.inv(roi_to_map_transform(scale, c_x, c_y))