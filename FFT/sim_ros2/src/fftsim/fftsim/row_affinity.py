"""Row-affinity corridor tracker for the fftsim DFT rig.

Why this exists: the Gai et al. DFT detector fits ONE plane wave
(fx, fy, phi) to the whole bird's-eye ROI, i.e. a global bundle of straight,
parallel rows. On straight fields that is exactly the ground truth, but on
curved / S-shaped plantings the true local corridor (the furrow the rover is
actually driving in) bends inside the ROI while the global fit averages the
curvature away. The servo then tracks the global corridor center and the
rover drifts off course on curves.

Fix ("affinity for the detected crop rows"): the DFT result is still trusted
for WHAT the rows are (spacing, coarse direction, phase index) - only the
local geometry is re-measured. Around the two crest lines that flank the
robot reference (the same flanking-pair rule the global fit uses), the
actual brightness crests of the windowed BEV ROI are located scanline by
scanline near their predicted wave positions, fitted with quadratic curves
x(y) = c0 + c1 y + c2 y^2, and the local corridor center/heading is taken
between the two fits at the reference point. ey/eth measured there follow
the curvature of the planted rows instead of the global plane wave. The
bend of the centerline across the visible band is also returned, as a
curvature feedforward signal for the shared servo.

Only the fx-dominant case (near-vertical rows, |fx| >= |fy|) is handled;
that is the only geometry the sim rig produces (fixed camera yaw, tangent
ring spawn). Anything else returns None and the pipeline falls back to the
global fit.

All steps are deterministic linear algebra on data the pipeline already
computes (windowed ROI + refined peak + DTFT phase), so the added cost is a
few 1D argmax searches plus two small least-squares fits per frame.

Sign conventions (matched to dft_crop_row_detector.Detection):
  ey_px  e = line - ref along x: corridor center RIGHT of the robot
         reference => positive.
  eth_deg  deviation of the corridor direction from image-vertical,
         leaning right (top of the row to the right of its base) => positive.
  bend   heading change of the centerline from the lower to the upper half
         of the band, radians, mr_vs.curve_bend convention (+ = turning
         right up ahead).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class AffinityResult:
    """Local corridor measurement in BEV ROI pixel coordinates."""

    __slots__ = ("ey_px", "eth_deg", "bend", "residual_px", "n_pts")

    def __init__(self, ey_px: float, eth_deg: float, bend: float,
                 residual_px: float, n_pts: int):
        self.ey_px = float(ey_px)
        self.eth_deg = float(eth_deg)
        self.bend = float(bend)
        self.residual_px = float(residual_px)
        self.n_pts = int(n_pts)

    def __repr__(self):
        return (f"AffinityResult(ey_px={self.ey_px:+.1f}, "
                f"eth={self.eth_deg:+.2f}deg, bend={self.bend:+.3f}rad, "
                f"res={self.residual_px:.1f}px, n={self.n_pts})")


def _measure_crest(imgw: np.ndarray, fx: float, fy: float, phi: float,
                   row_k: int, ys: np.ndarray,
                   period_px: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Locate crest #row_k of the detected plane wave on every scanline in ys.

    Wave model (Detection convention, ROI px coords, y down):
        I(x, y) = A cos(2 pi (fx x + fy y) + phi)
    Crest k: 2 pi (fx x + fy y) + phi = 2 pi k, so on scanline y the
    predicted crossing is x_pred = (k + base - fy*y) / fx with
    base = -phi/2pi (identical to Detection._row_positions:
    pos = (base + ks) / f_ax). The actual crest is the argmax of the
    smoothed scanline within +-0.45 periods of the prediction, refined with
    a parabolic sub-pixel step. Scanlines whose crest is weak relative to
    the local window scale (missing plants, weeds) are dropped.

    Returns (xs, ys) arrays of crest points, or None.
    """
    h, w = imgw.shape
    if abs(fx) < 1e-9:
        return None
    base = -phi / (2.0 * np.pi)
    pts_x, pts_y, vals = [], [], []
    half_win = 0.45 * period_px
    # smoothing kernel ~ 1/6 period to beat plant-scale noise
    ksz = int(max(3, period_px / 6.0)) | 1
    ker = np.exp(-0.5 * ((np.arange(ksz) - ksz // 2) / max(ksz / 6.0, 1.0)) ** 2)
    ker /= ker.sum()
    wy = np.hanning(h)  # y window factor, to normalize crest amplitudes
    for j in ys:
        j = int(j)
        if not (0 <= j < h):
            continue
        line = imgw[j, :]
        sm = np.convolve(line, ker, mode="same")
        pred = (row_k + base - fy * j) / fx
        lo = int(math.floor(pred - half_win))
        hi = int(math.ceil(pred + half_win))
        if lo < 0 or hi > w - 1 or hi <= lo:
            continue
        i = lo + int(np.argmax(sm[lo:hi + 1]))
        if not (1 <= i <= w - 2):
            continue
        # parabolic sub-pixel peak
        ym1, y0, yp1 = sm[i - 1], sm[i], sm[i + 1]
        den = ym1 - 2.0 * y0 + yp1
        delta = 0.5 * (ym1 - yp1) / den if abs(den) > 1e-9 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        pts_x.append(i + delta)
        pts_y.append(float(j))
        vals.append(float(y0) / max(wy[j], 1e-3))  # window-normalized
    if len(pts_x) < 8:
        return None
    vals = np.asarray(vals)
    med = float(np.median(vals))
    keep = vals > 0.5 * max(med, 1e-9)
    if keep.sum() < 8:
        return None
    return np.asarray(pts_x)[keep], np.asarray(pts_y)[keep]


def _fit_quad(py: np.ndarray, px: np.ndarray,
              max_res_px: float) -> Optional[Tuple[float, float, float, float]]:
    """Least-squares quadratic x = c0 + c1*y + c2*y^2 through crest points.

    Returns (c0, c1, c2, rms_residual_px) or None when too few points or
    the fit is too ragged (the local model does not describe the crest)."""
    if len(py) < 8:
        return None
    try:
        c2, c1, c0 = np.polyfit(py, px, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not all(math.isfinite(float(v)) for v in (c0, c1, c2)):
        return None
    r = px - (c0 + c1 * py + c2 * py * py)
    rms = float(np.sqrt(np.mean(r * r)))
    if not math.isfinite(rms) or rms > max_res_px:
        return None
    return float(c0), float(c1), float(c2), rms


def local_corridor(imgw: np.ndarray, fx: float, fy: float, phi: float,
                   spacing_px: float, roi_shape: Tuple[int, int],
                   ref_xy: Tuple[float, float],
                   global_eth_deg: float = 0.0,
                   frac: float = 0.5,
                   max_res_px: float = 6.0,
                   max_eth_delta_deg: float = 35.0,
                   flanks_agree_deg: float = 12.0) -> Optional[AffinityResult]:
    """Local (bottom-of-ROI) corridor center/heading from the DFT-detected
    row crests flanking the reference point.

    imgw          : the *windowed* BEV ROI the detection ran on (float64)
    fx, fy, phi   : refined peak + DTFT phase from Detection
    spacing_px    : Detection.spacing_px
    roi_shape     : (h, w) of the BEV ROI
    ref_xy        : robot reference point in ROI px (same as detect())
    global_eth_deg: Detection.e_theta_deg (loose sanity gate; on curved
                    fields the whole-ROI average heading can legitimately
                    differ from the local one by tens of degrees, so this
                    gate must stay wide)
    frac          : bottom fraction of the ROI used for the local fits
    flanks_agree_deg: the two flank headings must agree this closely -
                    a tight gate against locking onto foreign features
    """
    h, w = roi_shape
    if imgw.shape != roi_shape or imgw.ndim != 2:
        return None
    if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(phi)):
        return None
    fmag = math.hypot(fx, fy)
    if fmag < 1e-9 or not (spacing_px > 0):
        return None
    if abs(fy) >= abs(fx):
        return None  # near-horizontal rows: outside the supported geometry
    ref_x, ref_y = float(ref_xy[0]), float(ref_xy[1])
    band_top = (1.0 - frac) * (h - 1)
    ax_all = np.arange(h, dtype=float)
    ys = ax_all[ax_all >= band_top]
    if len(ys) < 16:
        return None

    # crest indices flanking the reference, from the same wave equation the
    # detector uses for its intersections (base = -phi/2pi)
    base = -phi / (2.0 * np.pi)
    k0 = fx * ref_x + fy * ref_y + phi / (2.0 * np.pi)
    k_left, k_right = int(math.floor(k0)), int(math.floor(k0)) + 1

    fits = []
    for k in (k_left, k_right):
        got = _measure_crest(imgw, fx, fy, phi, k, ys, period_px=spacing_px)
        if got is None:
            return None
        px, py = got
        fit = _fit_quad(py, px, max_res_px)
        if fit is None:
            return None
        fits.append(fit)

    (a0, a1, a2, ra), (b0, b1, b2, rb) = fits
    grid = np.linspace(float(ys[0]), float(ys[-1]), 24)
    xa = a0 + a1 * grid + a2 * grid * grid
    xb = b0 + b1 * grid + b2 * grid * grid
    if (np.min(xa) < 0 or np.max(xa) > w - 1
            or np.min(xb) < 0 or np.max(xb) > w - 1):
        return None
    mid = 0.5 * (xa + xb)  # corridor center polyline over the band

    # sanity gates: the two flank fits must describe one coherent corridor,
    # and only loosely agree with the global plane-wave direction (which
    # averages over the whole ROI and lags on curvature by design)
    m_a = a1 + 2.0 * a2 * ref_y   # dx/dy of flank A at the reference row
    m_b = b1 + 2.0 * b2 * ref_y
    eth_a = -math.degrees(math.atan(m_a))
    eth_b = -math.degrees(math.atan(m_b))
    if abs(eth_a - eth_b) > flanks_agree_deg:
        return None
    if (abs(eth_a - global_eth_deg) > max_eth_delta_deg
            or abs(eth_b - global_eth_deg) > max_eth_delta_deg):
        return None

    # corridor center offset at the reference (e = line - ref, so a center
    # to the RIGHT of the robot reference is positive)
    ey_px = float(0.5 * ((a0 + a1 * ref_y + a2 * ref_y * ref_y)
                         + (b0 + b1 * ref_y + b2 * ref_y * ref_y)) - ref_x)
    eth_deg = 0.5 * (eth_a + eth_b)

    # bend: heading change of the center polyline, lower vs upper half of
    # the visible band (mr_vs.curve_bend convention: atan2(dx, forward),
    # forward = up the image = decreasing y)
    n = len(grid)
    lo_s, up_s = slice(0, n // 2), slice(n // 2, n)
    a_lo = math.atan2(mid[lo_s][0] - mid[lo_s][-1],
                      grid[lo_s][-1] - grid[lo_s][0])
    a_up = math.atan2(mid[up_s][0] - mid[up_s][-1],
                      grid[up_s][-1] - grid[up_s][0])
    bend = _wrap(a_up - a_lo)

    return AffinityResult(ey_px=ey_px, eth_deg=eth_deg, bend=float(bend),
                          residual_px=max(ra, rb), n_pts=n)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def bend_to_ff(bend: float, v: float, gain: float,
               ff_max: float = 0.15) -> float:
    """Curvature feedforward command from the measured corridor bend, same
    convention as run_mr_navigation.process_image: ff = clip(-gain*v*bend)."""
    try:
        b = float(bend)
    except Exception:
        return 0.0
    if not math.isfinite(b):
        return 0.0
    return float(np.clip(-gain * float(v) * b, -ff_max, ff_max))
