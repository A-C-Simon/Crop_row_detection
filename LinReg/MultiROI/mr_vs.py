"""
MultiROI Visual Servoing for Furrow Following
==============================================
Navigation stack that consumes the MultiROI navigation line (furrow
corridor centre) and produces rover velocities.  Adapted from
ExG/visual-crop-row-navigation_ros2/src/agribot_vs.cpp visual servoing
but simplified for the furrow centre line.

The navigation line from test_multi_roi.py is:
  nav_line = (w, b) for y = w*x + b  (in cropped image coords)
  nav_curve = [(x,y)] polyline sampled every 4px (if nav_curve enabled)
  crop_offset = (dx, dy) border crop

Desired behaviour: keep the furrow centre line vertical and centred
at the bottom of the image (directly under the rover's nose).  The
rover drives forward at constant speed and steers to null the
lateral and heading errors.

This file is intentionally ROS-free for offline testing and can be
wrapped in a ROS2 node (see run_mr_navigation.py).

Author: derived from agribot_vs.cpp visual servoing
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np


def wrapToPi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


@dataclass
class MRVSParams:
    # Image geometry (after border crop, before any resize)
    width: int = 640
    height: int = 480
    # Control gains - tuned for 0.2 m/s forward
    # err_x is normalized by width (err_x/width), so lambda_x=2 means 100px error -> 0.31 rad/s
    # Theta is in radians, lambda_theta=1 means 10deg error -> 0.17 rad/s
    lambda_x: float = 2.0   # lateral error gain (was 10, too aggressive)
    lambda_theta: float = 1.0  # heading error gain
    # Velocity limits
    vf_des: float = 0.20     # desired forward speed m/s
    w_max: float = 0.60      # max angular rad/s (allow up to ~35deg/s)
    w_min: float = 0.01
    z_min: float = 0.01
    # Camera geometry (for interaction matrix, pinhole approx)
    rho_deg: float = -60.0   # tilt from horizontal
    ty: float = 0.6
    tz: float = 0.7


class MultiROIVS:
    """Visual servoing for MultiROI furrow centre line."""

    def __init__(self, params: Optional[MRVSParams] = None):
        self.params = params or MRVSParams()
        self.rho = math.radians(self.params.rho_deg)
        # For logging
        self.last_F = np.zeros(3)
        self.last_err = np.zeros(2)

    def nav_line_to_feature(self, nav_line, nav_curve, crop_offset, image_shape) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """
        Convert navigation line/curve to visual features.

        Returns:
            F = [X, Y, Theta] in image-centred coords (like agribot_vs)
            P,Q = bottom/top points of the line in full image coords (for drawing)
        """
        h, w = image_shape[:2]
        dx, dy = crop_offset if crop_offset else (0, 0)
        # Prefer curve tangent at base if available (more accurate for curving furrows)
        if nav_curve is not None and len(nav_curve) >= 2:
            # nav_curve is list of (x,y) in cropped coords, y=0 top
            # Take the lowest 2 points (closest to robot, largest y)
            # Curve sampled every 4px from y=0 to h-1, so last point is bottom
            # Use the segment at the bottom for heading
            # Find the two points with largest y
            pts = sorted(nav_curve, key=lambda p: p[1])
            # Bottom segment: last 2 points
            x1, y1 = pts[-1]
            x2, y2 = pts[-2] if len(pts) >= 2 else (pts[-1][0], pts[-1][1] - 10)
            # Convert to full image coords
            x1 += dx; y1 += dy
            x2 += dx; y2 += dy
            P = np.array([x1, y1], dtype=float)
            Q = np.array([x2, y2], dtype=float)
        elif nav_line is not None:
            w_slope, b = nav_line
            # Line y = w*x + b in cropped coords
            # Convert to full image: y_full = y_cropped + dy, x_full = x_cropped + dx
            # So y_cropped = w*x_cropped + b => y_full - dy = w*(x_full - dx) + b
            # => y_full = w*x_full + (b + dy - w*dx)
            b_full = b + dy - w_slope * dx
            # Intersections with top y=0 and bottom y=h-1 in full image
            # x = (y - b_full) / w_slope
            h_full, w_full = h, w  # full image is same as input after border crop is re-added
            # Use image height from bgr.shape, which includes border
            # Actually bgr is the original before crop, so h,w are full
            # nav_line was fitted in cropped coords, so we need to map
            # For simplicity, compute in cropped then add offset
            # Bottom point y = h_cropped -1 in cropped, maps to y_full = h_cropped-1+dy
            # Top point y=0 maps to y_full=dy
            h_cropped = h - 2*dy if dy else h
            # Use the line in cropped coords to get x at bottom/top
            y_bottom_c = h_cropped - 1
            y_top_c = 0
            if abs(w_slope) < 1e-6:
                # Horizontal (should not happen for vertical furrow)
                x_bottom_c = w / 2
                x_top_c = w / 2
            else:
                x_bottom_c = (y_bottom_c - b) / w_slope
                x_top_c = (y_top_c - b) / w_slope
            x_bottom = x_bottom_c + dx
            y_bottom = y_bottom_c + dy
            x_top = x_top_c + dx
            y_top = y_top_c + dy
            P = np.array([x_bottom, y_bottom], dtype=float)
            Q = np.array([x_top, y_top], dtype=float)
        else:
            return None, None

        # Theta: angle of line vs vertical, like agribot_vs compute_Theta
        # agribot_vs: Y = P.y - Q.y, X = Q.x - P.x, phi = atan2(Y, X), Theta = pi/2 - phi
        # This makes vertical line (P bottom, Q top, X=0, Y>0) give Theta=0
        Y = float(P[1] - Q[1])
        X = float(Q[0] - P[0])
        phi = math.atan2(Y, X)
        Theta = wrapToPi(math.pi/2 - phi)  # deviation from vertical

        # Feature F in image-centred coords (origin at image centre)
        # agribot_vs: xi = xc - width/2, yi = yc - height/2
        # Use P (bottom point) as the reference (closest to robot)
        # X = P.x - width/2, Y = P.y - height/2
        # For furrow, X is lateral error at bottom, Theta is heading error
        X = P[0] - w/2.0
        Y = P[1] - h/2.0
        F = np.array([X, Y, Theta], dtype=float)
        return F, (tuple(P), tuple(Q))

    def compute_control(self, F: np.ndarray) -> Tuple[float, float, Dict]:
        """
        Visual servoing control law, simplified from agribot_vs.cpp:Controller

        Input F = [X, Y, Theta], F_des = [0, height/2, 0] ??? For furrow,
        we want X=0 (centred) and Theta=0 (vertical) at the bottom.
        Y is not directly controlled (forward motion).

        Returns (v, w, info dict)
        """
        p = self.params
        if F is None:
            return 0.0, 0.0, {"err_x": 0, "err_theta": 0}

        # Desired feature: centred and vertical at bottom
        # For ExG, F_des = [0, height/2, 0] where height is image height
        # For MultiROI furrow, we also want X=0, Theta=0.
        # Y is not critical, but we keep it for completeness.
        X, Y, Theta = float(F[0]), float(F[1]), float(F[2])
        # Errors
        err_x = X  # pixels, need to convert to meters? For now keep pixels, gains will handle
        err_theta = wrapToPi(Theta)

        # Simple proportional control (like agribot but without full Ls)
        # w = -lambda_x * err_x_norm - lambda_theta * err_theta
        # Normalize err_x by width to make gain independent of resolution
        # Use image width to convert pixels to normalized units
        # For 640 width, 10px error ~ 0.015 normalized
        # We keep err_x in pixels and let lambda handle it, similar to agribot where lambda_x=10
        err_x_norm = err_x / p.width  # normalize

        # Compute angular velocity
        # Use same structure as agribot: w = -Jw_pinv*(lambda*err + Jv*v)
        # Simplified: w = -(lambda_x*err_x_norm + lambda_theta*err_theta)
        # Scale to be in rad/s
        w_raw = -(p.lambda_x * err_x_norm + p.lambda_theta * err_theta)

        # Clamp
        w = max(-p.w_max, min(p.w_max, w_raw))
        if abs(w) < p.w_min:
            w = 0.0
        if abs(w) < p.z_min:
            w = 0.0

        v = p.vf_des

        # If errors are large, reduce linear speed (optional)
        # For now constant

        info = {
            "err_x": float(err_x),
            "err_x_norm": float(err_x_norm),
            "err_theta_deg": float(math.degrees(err_theta)),
            "w_raw": float(w_raw),
            "w": float(w),
            "v": float(v),
        }
        self.last_F = F.copy()
        self.last_err = np.array([err_x, err_theta])
        return v, w, info

    def draw_overlay(self, bgr, P, Q, v, w, info):
        """Draw navigation line and velocity info on image."""
        out = bgr.copy()
        h, w_img = out.shape[:2]
        if P is not None and Q is not None:
            # Draw navigation line red (like agribot) and window
            cv2.line(out, (int(P[0]), int(P[1])), (int(Q[0]), int(Q[1])), (0, 0, 255), 2, cv2.LINE_AA)
            # Draw bottom point
            cv2.circle(out, (int(P[0]), int(P[1])), 8, (0, 0, 255), -1)
            cv2.circle(out, (int(Q[0]), int(Q[1])), 5, (0, 255, 255), -1)
            # Draw image centre
            cv2.circle(out, (w_img//2, h//2), 4, (255, 255, 0), -1)
            cv2.drawMarker(out, (w_img//2, h-20), (255, 255, 0), cv2.MARKER_STAR, 20, 2)
        # Text overlay
        txt = f"v={v:.2f} m/s w={math.degrees(w):.1f} deg/s | err_x={info.get('err_x',0):.0f}px err_theta={info.get('err_theta_deg',0):.1f}deg"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return out
