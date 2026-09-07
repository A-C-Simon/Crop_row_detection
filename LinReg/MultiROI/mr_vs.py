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
    # Vertical coverage: navigation line is drawn only over the bottom
    # `vertical_coverage` fraction of the image (same as detector ROIs).
    # 0.6 means the red line ends at y = h*(1-0.6), where ROI coverage ends.
    vertical_coverage: float = 0.75
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
    # --- Command smoothing (temporal robustness) ---
    # These are applied INSIDE compute_control when smooth=True
    # Keep small/causal to avoid lag but prevent chatter
    w_alpha: float = 0.35         # low-pass on w (0=hold, 1=no smoothing)
    max_w_rate: float = 1.2       # rad/s per second (rate limit)
    w_deadband: float = 0.02      # rad/s; values below -> 0
    v_conf_scale: bool = True
    v_min_scale: float = 0.45     # at confidence 0, v = vf * 0.45


def clip_segment_to_coverage_top(P, Q, h, vertical_coverage):
    """Clip segment P->Q so it never goes above the coverage top.

    Coverage is bottom-anchored: visible y must satisfy y >= h*(1-cov).
    Returns (P_clipped, Q_clipped) as float arrays, or (None, None) if the
    whole segment lies above coverage.
    """
    try:
        cov = float(vertical_coverage)
    except Exception:
        cov = 1.0
    if not math.isfinite(cov):
        cov = 1.0
    cov = min(1.0, max(0.1, cov))
    if cov >= 1.0:
        return np.array(P, dtype=float), np.array(Q, dtype=float)
    y_top = float(h * (1.0 - cov))
    Pp = np.array(P, dtype=float).copy()
    Qq = np.array(Q, dtype=float).copy()
    p_above = Pp[1] < y_top
    q_above = Qq[1] < y_top
    if p_above and q_above:
        return None, None
    if p_above or q_above:
        # interpolate the above endpoint to y=y_top along P->Q
        denom = (Qq[1] - Pp[1])
        if abs(denom) < 1e-9:
            return None, None
        t = (y_top - Pp[1]) / denom  # t=0 at P, t=1 at Q
        Xt = Pp[0] + t * (Qq[0] - Pp[0])
        if p_above:
            Pp = np.array([Xt, y_top], dtype=float)
        else:
            Qq = np.array([Xt, y_top], dtype=float)
    return Pp, Qq


class MultiROIVS:
    """Visual servoing for MultiROI furrow centre line."""

    def __init__(self, params: Optional[MRVSParams] = None):
        self.params = params or MRVSParams()
        self.rho = math.radians(self.params.rho_deg)
        # For logging
        self.last_F = np.zeros(3)
        self.last_err = np.zeros(2)
        # For command smoothing (rate-limit + low-pass)
        self._prev_w: float = 0.0
        self._prev_v: float = float(self.params.vf_des)
        self._prev_time: Optional[float] = None

    def reset_smoother(self):
        self._prev_w = 0.0
        self._prev_v = float(self.params.vf_des)
        self._prev_time = None

    def nav_line_to_feature(self, nav_line, nav_curve, crop_offset, image_shape, vertical_coverage=None) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """
        Convert navigation line/curve to visual features.

        Returns:
            F = [X, Y, Theta] in image-centred coords (like agribot_vs)
            P,Q = bottom/top points of the line in full image coords (for drawing)

        The top point Q stops where vertical coverage ends (bottom-anchored
        `vertical_coverage` fraction), so the drawn line never extends above
        the ROI region. Control features (X at bottom, Theta) are unaffected
        since they depend on the bottom point and line direction only.
        """
        h, w = image_shape[:2]
        dx, dy = crop_offset if crop_offset else (0, 0)
        cov = vertical_coverage
        if cov is None:
            cov = float(getattr(self.params, "vertical_coverage", 0.75))
        try:
            cov = float(cov)
        except Exception:
            cov = 0.75
        if not math.isfinite(cov):
            cov = 0.75
        cov = min(1.0, max(0.1, cov))
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
            # Top stops where vertical coverage ends (bottom-anchored).
            y_bottom_c = h_cropped - 1
            y_top_c = int(round(h_cropped * (1.0 - cov))) if cov < 1.0 else 0
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

    def compute_control(self, F: np.ndarray,
                        dt: Optional[float] = None,
                        confidence: float = 1.0,
                        smooth: bool = True) -> Tuple[float, float, Dict]:
        """
        Visual servoing control law, simplified from agribot_vs.cpp:Controller

        Input F = [X, Y, Theta], F_des = [0, height/2, 0] ??? For furrow,
        we want X=0 (centred) and Theta=0 (vertical) at the bottom.
        Y is not directly controlled (forward motion).

        Smoothing (when smooth=True):
          - low-pass: w_lpf = (1-alpha)*prev_w + alpha*w_raw
          - rate limit: |delta|/dt <= max_w_rate
          - deadband: |w|<w_deadband -> 0
          - confidence-based v scaling

        Returns (v, w, info dict)
        """
        p = self.params
        if F is None:
            # also smooth hold
            if smooth:
                # decay w toward 0 with rate limit
                now = time.perf_counter()
                if dt is None:
                    if self._prev_time is None:
                        dt = 0.05
                    else:
                        dt = now - self._prev_time
                        dt = max(1e-3, min(0.5, dt))
                self._prev_time = now
                max_delta = p.max_w_rate * dt
                # low-pass toward 0
                target = 0.0
                w_lpf = (1 - p.w_alpha) * self._prev_w + p.w_alpha * target
                # rate limit to 0
                delta = w_lpf - self._prev_w
                if abs(delta) > max_delta:
                    w_lpf = self._prev_w + math.copysign(max_delta, delta)
                if abs(w_lpf) < p.w_deadband:
                    w_lpf = 0.0
                self._prev_w = w_lpf
                v = p.vf_des * (p.v_min_scale + (1 - p.v_min_scale) * float(np.clip(confidence, 0, 1))) if p.v_conf_scale else p.vf_des
                self._prev_v = v
                return v, w_lpf, {"err_x": 0, "err_theta": 0, "w_raw": 0.0, "w": float(w_lpf), "v": float(v), "smoothed": True}
            return 0.0, 0.0, {"err_x": 0, "err_theta": 0, "w_raw": 0.0, "w": 0.0, "v": 0.0}

        # Desired feature: centred and vertical at bottom
        X, Y, Theta = float(F[0]), float(F[1]), float(F[2])
        # Errors
        err_x = X  # pixels
        err_theta = wrapToPi(Theta)
        err_x_norm = err_x / p.width  # normalize
        w_raw = -(p.lambda_x * err_x_norm + p.lambda_theta * err_theta)

        # Clamp raw before smoothing (keep limits)
        w_clamped = max(-p.w_max, min(p.w_max, w_raw))
        if abs(w_clamped) < p.w_min:
            w_clamped = 0.0
        if abs(w_clamped) < p.z_min:
            w_clamped = 0.0

        v_raw = p.vf_des

        w_out = w_clamped
        v_out = v_raw

        if smooth:
            now = time.perf_counter()
            if dt is None:
                if self._prev_time is None:
                    dt = 0.05
                else:
                    dt = now - self._prev_time
                    dt = max(1e-3, min(0.5, dt))
            self._prev_time = now
            # low-pass
            w_lpf = (1 - p.w_alpha) * self._prev_w + p.w_alpha * w_clamped
            # rate limit
            max_delta = p.max_w_rate * dt
            delta = w_lpf - self._prev_w
            if abs(delta) > max_delta:
                w_lpf = self._prev_w + math.copysign(max_delta, delta)
            # re-clamp to max
            w_lpf = max(-p.w_max, min(p.w_max, w_lpf))
            if abs(w_lpf) < p.w_deadband:
                w_lpf = 0.0
            w_out = w_lpf
            self._prev_w = w_out
            # confidence-based v scaling
            if p.v_conf_scale:
                scale = p.v_min_scale + (1.0 - p.v_min_scale) * float(np.clip(confidence, 0.0, 1.0))
                v_out = v_raw * scale
            self._prev_v = v_out

        info = {
            "err_x": float(err_x),
            "err_x_norm": float(err_x_norm),
            "err_theta_deg": float(math.degrees(err_theta)),
            "w_raw": float(w_raw),
            "w_clamped": float(w_clamped),
            "w": float(w_out),
            "v": float(v_out),
            "confidence": float(confidence),
        }
        self.last_F = F.copy()
        self.last_err = np.array([err_x, err_theta])
        return v_out, w_out, info

    def draw_overlay(self, bgr, P, Q, v, w, info, vertical_coverage=None):
        """Draw navigation line and velocity info on image.

        The red navigation line is clipped to the vertical-coverage top
        (bottom-anchored `vertical_coverage` fraction), so it ends where
        the ROI coverage ends.
        """
        out = bgr.copy()
        h, w_img = out.shape[:2]
        if P is not None and Q is not None:
            cov = vertical_coverage
            if cov is None:
                cov = float(getattr(self.params, "vertical_coverage", 0.75))
            Pc, Qc = clip_segment_to_coverage_top(P, Q, h, cov)
            if Pc is not None and Qc is not None:
                # Draw navigation line red (like agribot) and window
                cv2.line(out, (int(Pc[0]), int(Pc[1])), (int(Qc[0]), int(Qc[1])), (0, 0, 255), 2, cv2.LINE_AA)
                # Draw bottom point (P may have been clipped if it was above; still show)
                cv2.circle(out, (int(Pc[0]), int(Pc[1])), 8, (0, 0, 255), -1)
                cv2.circle(out, (int(Qc[0]), int(Qc[1])), 5, (0, 255, 255), -1)
            # Draw image centre
            cv2.circle(out, (w_img//2, h//2), 4, (255, 255, 0), -1)
            cv2.drawMarker(out, (w_img//2, h-20), (255, 255, 0), cv2.MARKER_STAR, 20, 2)
        # Text overlay
        txt = f"v={v:.2f} m/s w={math.degrees(w):.1f} deg/s | err_x={info.get('err_x',0):.0f}px err_theta={info.get('err_theta_deg',0):.1f}deg"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return out
