#!/usr/bin/env python3
"""Base-anchored, column-aware sliding window for ExG crop-row navigation.

Window contract
---------------
* The window is pinned to the **base of the frame** - the ground point just
  ahead of the chassis, the same place as the blue star the MultiROI overlay
  draws at ``(width/2, height - base_margin)``. The vertical centre is
  ``Yc = height - base_margin - H/2``, so the bottom edge sits on that
  chassis-forward reference and the window never drifts up the image.
* Acquisition keeps the **column-aware latch**: the window follows the crop
  column nearest the image centre (bottom-band vertical-projection peaks,
  prominence-weighted), with the width taken from the median inter-row gap.
  The ExG vehicle rides *above* the rows, so the band it locks onto is the
  one at the bottom of the image (the plants under/ahead of the chassis).
* When the latched column has been driven to the image centre (the chassis
  aligned with it) for ``lock_frames`` consecutive frames the window
  **locks at bottom-centre**: ``Xc`` is the image centre from then on, the
  permanent blue-marker behaviour.
* While locked, if the column disappears from the base band - the end of the
  crop column, i.e. the headland/lane transition - or the window runs empty
  for ``lost_frames`` frames, the tracker **re-latches** (back to
  acquisition). That is the trigger the sim row-change machine watches.

``BaseColumnWindow`` owns the temporal state; the pure helpers
(``column_profile``, ``pick_column``) are usable standalone when there is no
sequence (e.g. a single still).
"""

from __future__ import annotations

import math

import numpy as np
import cv2

ACQUIRE = "acquire"
LOCKED = "locked"

# params keys read from the ExG params dict, with the historical defaults
DEFAULTS = {
    "width": 640,
    "height": 480,
    "ex_Xc": 320,            # fallback lateral centre when no column is seen
    "ex_Yc": 380,            # kept for reference; Yc now comes from base_margin
    "nh_L": 80,              # fallback window width
    "nh_H": 180,             # window height
    "colaware_y0_frac": 0.55,
    "colaware_peak_dist_frac": 0.045,
    "colaware_prominence": 0.12,
    "colaware_peak_height": 0.15,
    # base anchoring / state machine
    "base_margin": 10.0,     # marker height above the frame bottom (px)
    "latch_tol_px": 24.0,    # |column - centre| counted as aligned
    "lock_frames": 3,        # aligned frames before locking at centre
    "lost_frames": 5,        # column-less frames before re-latching
    "lock_search_px": 60.0,  # a locked window looks for its column this near
    "min_nh_points": 5,      # empty window threshold while locked
}


def _cfg(params, key):
    """params[key] when set, else the module default."""
    if params and params.get(key) is not None:
        return params[key]
    return DEFAULTS[key]


def find_profile_peaks(prof_norm, distance, prominence, height):
    """``scipy.signal.find_peaks`` with a dependency-free fallback."""
    try:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(prof_norm, distance=distance,
                                  prominence=prominence, height=height)
        return list(int(p) for p in peaks), props
    except Exception:
        peaks = []
        for idx in np.argsort(prof_norm)[::-1]:
            if prof_norm[idx] < height:
                break
            if all(abs(int(idx) - p) >= distance for p in peaks):
                peaks.append(int(idx))
        peaks.sort()
        return peaks, {"prominences": np.ones(len(peaks))}


def column_profile(combined_mask, width, height, params=None):
    """Bottom-band vertical projection of the combined HSV mask.

    Returns ``(profile, smooth, peaks, props, median_gap, L_dynamic, y0)``.
    ``peaks`` are candidate crop-column positions, ``L_dynamic`` the
    inter-row-gap-derived window width (falls back to ``nh_L``).
    """
    y0_frac = float(_cfg(params, "colaware_y0_frac"))
    y0 = int(height * y0_frac)
    roi = combined_mask[y0:height, :]
    profile = roi.sum(axis=0).astype(float)
    if profile.max() > 0:
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 5).ravel()
    else:
        smooth = profile
    prof_n = smooth / smooth.max() if smooth.max() > 0 else smooth

    dist = max(25, int(width * float(_cfg(params, "colaware_peak_dist_frac"))))
    peaks, props = find_profile_peaks(
        prof_n, dist,
        float(_cfg(params, "colaware_prominence")),
        float(_cfg(params, "colaware_peak_height")))

    median_gap = None
    L_dynamic = int(_cfg(params, "nh_L"))
    if len(peaks) >= 2:
        gaps = np.diff(sorted(peaks))
        median_gap = float(np.median(gaps)) if len(gaps) else None
        if median_gap and 30 <= median_gap <= 180:
            L_dynamic = int(np.clip(median_gap * 0.65, 60, 110))
    return profile, smooth, peaks, props, median_gap, L_dynamic, y0


def pick_column(peaks, props, width, prominence_weight=30.0):
    """Nearest-to-centre peak, prominence-weighted. ``(index, x)`` or
    ``(None, None)`` when there are no peaks."""
    if not peaks:
        return None, None
    center = width // 2
    prominences = list(props.get("prominences", [])) if props else []
    if len(prominences) != len(peaks):
        prominences = [1.0] * len(peaks)
    scores = [abs(int(x) - center) - prominence_weight * float(p)
              for x, p in zip(peaks, prominences)]
    idx = int(np.argmin(scores))
    return idx, int(peaks[idx])


class BaseColumnWindow:
    """Temporal base-anchored, column-aware window tracker.

    One instance per camera stream; call :meth:`update` once per frame with
    the combined HSV mask (and optionally the contour centres for the empty
    check). Read ``state``/``Xc``/``Yc``/``L``/``H``/``column_x`` off the
    instance (also returned as a dict) to window the detected points.
    """

    def __init__(self, params=None):
        self.params = dict(params or {})
        self.width = int(_cfg(self.params, "width"))
        self.height = int(_cfg(self.params, "height"))
        self.reset()

    # --------------------------------------------------------------
    def reset(self):
        """Back to acquisition (fresh run, or a re-latch)."""
        self.state = ACQUIRE
        self._aligned = 0
        self._lost = 0
        self.relatched = False
        self.Xc = int(_cfg(self.params, "ex_Xc"))
        self.Yc = self.base_yc(int(_cfg(self.params, "nh_H")))
        self.L = int(_cfg(self.params, "nh_L"))
        self.H = int(_cfg(self.params, "nh_H"))
        self.column_x = None
        self.chosen_idx = None
        self.n_nh = 0
        self.peaks = []
        self.median_gap = None
        self.profile = None
        self.smooth = None
        self.y0 = int(self.height * float(_cfg(self.params, "colaware_y0_frac")))

    # --------------------------------------------------------------
    def base_yc(self, H=None):
        """Window centre that puts its bottom edge on the chassis reference."""
        H = int(_cfg(self.params, "nh_H")) if H is None else int(H)
        margin = float(_cfg(self.params, "base_margin"))
        return int(round(self.height - margin - H / 2.0))

    def ref_y(self):
        """Chassis-forward reference row (the blue-marker height)."""
        return int(round(self.height - float(_cfg(self.params, "base_margin"))))

    def nh_count(self, centers):
        """Points inside the current window."""
        if centers is None:
            return None
        hx, hL, hy, hH = self.Xc, self.L / 2.0, self.Yc, self.H / 2.0
        return sum(1 for (x, y) in centers
                   if (hx - hL < x < hx + hL) and (hy - hH < y < hy + hH))

    # --------------------------------------------------------------
    def update(self, combined_mask, centers=None):
        """Advance one frame. Returns the window dict (also stored on self)."""
        width = self.width
        profile, smooth, peaks, props, median_gap, L_dyn, y0 = \
            column_profile(combined_mask, width, self.height, self.params)
        chosen_idx, column_x = pick_column(peaks, props, width)

        H = int(_cfg(self.params, "nh_H"))
        state = self.state
        relatched = False
        aligned = False

        if state == ACQUIRE:
            if column_x is not None:
                L = L_dyn
                half = L // 2
                Xc = int(np.clip(column_x, half + 2, width - half - 2))
                self._lost = 0
                aligned = abs(column_x - width // 2) <= \
                    float(_cfg(self.params, "latch_tol_px"))
                self._aligned = self._aligned + 1 if aligned else 0
                if self._aligned >= int(_cfg(self.params, "lock_frames")):
                    state = LOCKED
                    Xc = width // 2
            else:
                L = int(_cfg(self.params, "nh_L"))
                Xc = int(_cfg(self.params, "ex_Xc"))
                self._aligned = 0
        else:  # LOCKED - window stays at bottom-centre
            L = self.L if self.L > 0 else int(_cfg(self.params, "nh_L"))
            Xc = width // 2
            near = column_x is not None and \
                abs(column_x - width // 2) <= float(_cfg(self.params, "lock_search_px"))
            self.Xc, self.Yc, self.L, self.H = Xc, self.base_yc(H), L, H
            n_nh = self.nh_count(centers) if centers is not None else None
            empty = n_nh is not None and n_nh < int(_cfg(self.params, "min_nh_points"))
            if near and not empty:
                self._lost = 0
            else:
                # no column near the base band (crop column ended) or the
                # window ran empty -> re-latch for the next column/lane
                self._lost += 1
                if self._lost >= int(_cfg(self.params, "lost_frames")):
                    self.reset()
                    self.peaks = peaks
                    self.median_gap = median_gap
                    self.profile = profile
                    self.smooth = smooth
                    self.y0 = y0
                    self.column_x = column_x
                    self.chosen_idx = chosen_idx
                    self.relatched = True
                    return self.as_dict(relatched=True, aligned=False)

        Yc = self.base_yc(H)
        self.Xc, self.Yc, self.L, self.H = Xc, Yc, L, H
        self.state = state
        self.column_x = column_x
        self.chosen_idx = chosen_idx
        self.peaks = peaks
        self.median_gap = median_gap
        self.profile = profile
        self.smooth = smooth
        self.y0 = y0
        self.relatched = relatched
        self.n_nh = self.nh_count(centers) if centers is not None else 0
        return self.as_dict(relatched=relatched, aligned=aligned)

    # --------------------------------------------------------------
    def as_dict(self, relatched=None, aligned=None):
        return {
            "Xc": self.Xc,
            "Yc": self.Yc,
            "L": self.L,
            "H": self.H,
            "state": self.state,
            "locked": self.state == LOCKED,
            "column_x": self.column_x,
            "chosen_idx": self.chosen_idx,
            "ref_y": self.ref_y(),
            "peaks": list(self.peaks),
            "median_gap": self.median_gap,
            "profile": self.profile,
            "smooth": self.smooth,
            "y0": self.y0,
            "n_nh": self.n_nh,
            "relatched": self.relatched if relatched is None else relatched,
            "aligned": aligned,
        }


# ---------------------------------------------------------------------------
# chassis-base servoing (shared by the video runner and the sim nav node)
# ---------------------------------------------------------------------------
def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def line_base_error(fit_info, width, height, y_ref=None, base_margin=10.0):
    """Lateral + heading error of the fitted row at the chassis reference.

    Evaluates the fitted line at the bottom-of-frame chassis reference
    (``y_ref = height - base_margin``, the blue-marker row), not the image
    middle, because the ExG vehicle rides above the rows and steers on the
    plants under/ahead of the chassis.

    Returns ``(err_x_px, err_theta_rad)``: ``err_x > 0`` means the row sits
    right of the chassis, ``err_theta`` is the row's lean from image-vertical
    (positive = top of the row to the right). ``(None, None)`` without a fit.
    """
    if not fit_info:
        return None, None
    try:
        vx, vy, x0, y0 = fit_info["line"]
    except Exception:
        return None, None
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
    if abs(vy) < 1e-6:
        return None, None
    # fitLine's direction is sign-arbitrary; force it down-image so the
    # heading is the row's lean from vertical, not +180 deg.
    if vy < 0:
        vx, vy = -vx, -vy
    if y_ref is None:
        y_ref = height - base_margin
    x_ref = x0 if abs(y_ref - y0) < 1e-9 else x0 + vx * (y_ref - y0) / vy
    err_x = float(x_ref) - width / 2.0
    err_theta = wrap_pi(math.atan2(vx, vy))
    return err_x, err_theta


def steer_from_base(err_x, err_theta, width, kx=0.9, kth=1.0,
                    w_max=0.6, deadband=0.01):
    """Yaw command that drives the row base to the bottom-centre reference.

    Lateral term: ``err_x > 0`` (row base right of the chassis) -> negative w
    (turn right), matching a forward-facing camera. Heading term: a positive
    ``err_theta`` means the row leans up-left, so the chassis must yaw left
    (positive w) to stay on it - the same sign as ``err_theta``, NOT negated
    (negating makes the two terms cancel on a bend). ``deadband`` zeroes the
    command so the straight-ahead wiggle stays out of the wheels.
    """
    if err_x is None:
        return 0.0
    w = -kx * float(err_x) / float(width) + kth * float(err_theta or 0.0)
    w = max(-w_max, min(w_max, w))
    return 0.0 if abs(w) < deadband else float(w)
