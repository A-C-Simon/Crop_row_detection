"""DFT-based crop row detection, following Gai et al. (2024), fft.pdf.

Pipeline (Section 2.2 of the paper):
  1) Pre-processing: Excess-Green grayscale and inverse perspective mapping to
     a metric bird's-eye ROI (MAAIR cropped).
  2) DFT: 2D FFT of the Hanning-windowed ROI.
  3) Peak localization: bandpass filter around the expected row-spacing
     frequency, coarse argmax peak, Taylor-expansion sub-bin refinement,
     DTFT phase at the refined continuous frequency.
  4) Row geometry: wavefront lines (constant phase) give row positions and
     direction; lateral deviation e_y and heading deviation e_theta follow
     from the two rows flanking the robot reference point.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def exg_gray(rgb: np.ndarray) -> np.ndarray:
    """Excess Green Index grayscale, contrast-normalized to [0, 255]."""
    arr = rgb[..., :3].astype(np.float64)
    exg = 2.0 * arr[..., 1] - arr[..., 0] - arr[..., 2]
    lo, hi = np.percentile(exg, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((exg - lo) / (hi - lo) * 255.0, 0.0, 255.0)


def maair(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Maximum axis-aligned inscribed rectangle of a valid region whose row
    spans are contiguous and monotone (trapezoids from IPM):
    returns (row, col, height, width)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, 0, 0
    r0 = int(np.argmax(rows))
    r1 = len(rows) - int(np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = len(cols) - int(np.argmax(cols[::-1]))
    n = r1 - r0
    spans = np.zeros(n, dtype=int)
    lefts = np.zeros(n, dtype=int)
    for i, r in enumerate(range(r0, r1)):
        idx = np.flatnonzero(mask[r, c0:c1])
        if idx.size:
            lefts[i] = idx[0]
            spans[i] = idx[-1] - idx[0] + 1
    sufmin = np.minimum.accumulate(spans[::-1])[::-1]
    heights = np.arange(n, 0, -1)
    min_w, min_h = 0.6 * (c1 - c0), 0.6 * n
    ok = (sufmin >= min_w) & (heights >= min_h)
    if not ok.any():
        return r0, c0, n, c1 - c0
    kk = np.flatnonzero(ok)
    k = int(kk[np.argmax(sufmin[kk] * heights[kk])])
    j = k + int(np.argmin(spans[k:]))
    return r0 + k, c0 + int(lefts[j]), int(heights[k]), int(sufmin[k])


def camera_homography(shape: Tuple[int, int], pitch_deg: float, height_m: float,
                      fov_y_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Ground->image homography of a pinhole camera at altitude height_m.

    World frame: X right, Y forward, Z up, ground plane Z=0. Pitch tilts the
    optical axis away from nadir toward +Y; yaw rotates the tilt direction.
    """
    h, w = shape
    f = (h / 2.0) / np.tan(np.radians(fov_y_deg) / 2.0)
    K = np.array([[f, 0.0, w / 2.0],
                  [0.0, f, h / 2.0],
                  [0.0, 0.0, 1.0]])
    th = np.radians(pitch_deg)
    c, s = np.cos(th), np.sin(th)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, -c, -s],
                   [0.0, s, -c]])
    ps = np.radians(yaw_deg)
    cy, sy = np.cos(ps), np.sin(ps)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    R = Rx @ Rz.T
    t = -R @ np.array([0.0, 0.0, height_m])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def rectify_forward(gray: np.ndarray, pitch_deg: float, height_m: float,
                    fov_y_deg: float, gsd: Optional[float] = None,
                    yaw_deg: float = 0.0,
                    range_m: Optional[float] = None
                    ) -> Tuple[np.ndarray, float]:
    """Inverse perspective mapping of a forward-looking image to a metric
    bird's-eye grid, MAAIR-cropped to fully sampled pixels.

    Returns (roi_gray, gsd_used_in_m_per_px, map_info). map_info carries the
    rect-to-image homography, the ROI crop offset and the full grid size so
    detections can be drawn back onto the original image. gsd=None
    auto-matches the grid to the source image sampling density. range_m
    limits how far ahead of the camera ground is kept.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for rectification")
    Hg = camera_homography(gray.shape, pitch_deg, height_m, fov_y_deg, yaw_deg)
    Tinv = np.linalg.inv(Hg)

    t = np.linspace(0.0, 1.0, 64)
    m_, n_ = gray.shape
    eu = np.concatenate([t, np.ones_like(t), 1 - t, np.zeros_like(t)])
    ev = np.concatenate([np.zeros_like(t), t, np.ones_like(t), 1 - t])
    pts = Tinv @ np.vstack([eu * (n_ - 1), ev * (m_ - 1), np.ones_like(eu)])
    Xc, Yc = pts[0] / pts[2], pts[1] / pts[2]
    ok = np.isfinite(Xc) & np.isfinite(Yc) & (np.abs(Xc) < 80)
    ymax_lim = 80.0 if not range_m or range_m <= 0 else float(range_m)
    ok &= (Yc > 0.15) & (Yc < ymax_lim)
    if ok.sum() < 8:
        raise ValueError("no ground within range; check pitch/height/range")
    xmin, xmax = np.percentile(Xc[ok], [1, 99])
    ymin, ymax = np.percentile(Yc[ok], [1, 99])

    if not gsd or gsd <= 0:
        rr = np.linspace(0, m_ - 1, 400)
        pts = Tinv @ np.vstack([np.full_like(rr, n_ / 2.0), rr,
                                np.ones_like(rr)])
        Yg = pts[1] / pts[2]
        sel = np.isfinite(Yg) & (Yg > max(ymin, 0.15)) & (Yg < ymax)
        if sel.sum() > 2:
            gsd = float(np.clip(np.percentile(1.0 / np.abs(np.gradient(
                Yg[sel])), 80) * 1.2, 0.004, 0.02))
        else:
            gsd = height_m / 400.0

    gsd = max(gsd, (xmax - xmin) / 2000.0, (ymax - ymin) / 2000.0)
    W = int(np.ceil((xmax - xmin) / gsd))
    Ht = int(np.ceil((ymax - ymin) / gsd))
    if W < 16 or Ht < 16:
        raise ValueError(f"degenerate rectified grid ({W}x{Ht} px)")
    A = np.array([[1 / gsd, 0, -xmin / gsd],
                  [0, -1 / gsd, ymax / gsd],
                  [0, 0, 1.0]])
    M = A @ Tinv

    src = np.clip(gray, 0, 255).astype(np.uint8)
    full = cv2.warpPerspective(src, M, (W, Ht), flags=cv2.INTER_LINEAR,
                               borderValue=0)
    valid = cv2.warpPerspective(np.full(gray.shape[:2], 255, np.uint8),
                                M, (W, Ht), borderValue=0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    r0, c0, hh, ww = maair(cv2.erode(valid, k) > 0)
    if hh == 0 or ww == 0:
        raise ValueError("rectification produced no valid ground pixels")
    map_info = {"minv": np.linalg.inv(M), "offset": (int(r0), int(c0)),
                "grid": (int(W), int(Ht))}
    return (full[r0:r0 + hh, c0:c0 + ww].astype(np.float64), float(gsd),
            map_info)


# ---------------------------------------------------------------------------
# DFT detection core
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Detection:
    fx: float                 # refined peak frequency along x (columns), cyc/px
    fy: float                 # refined peak frequency along y (rows), cyc/px
    Tx: float                 # period along x, px (inf when fx == 0)
    Ty: float                 # period along y, px (inf when fy == 0)
    phi: float                # DTFT phase at the peak, rad
    spacing_px: float         # perpendicular row spacing, px
    angle_from_vertical_deg: float
    intersections: np.ndarray # axis crossings of the row lines, px
    direction: np.ndarray     # unit direction of row lines, pointing up
    ref_point: np.ndarray     # robot/reference point in ROI px
    e_indices: np.ndarray     # signed distances of all lines to ref point, px
    ey_px: float              # lateral deviation, px (nan if <2 rows)
    corridor_px: float        # width of the flanked corridor, px
    e_theta_deg: float        # heading deviation vs image-up, deg
    prominence: float         # spectral lock confidence, peak / band median
    magnitude: np.ndarray     # shifted magnitude spectrum (for plotting)
    freq_x: np.ndarray        # shifted frequency axes matching magnitude
    freq_y: np.ndarray
    band_lo: float            # bandpass radii in cyc/px
    band_hi: float

    @property
    def n_rows(self) -> int:
        return len(self.intersections)

    def line_endpoints(self, length_scale: float = 1.6):
        """Segment endpoints (x0, y0, x1, y1) of every detected row line."""
        h, w = self.magnitude.shape
        L = length_scale * float(np.hypot(h, w))
        t = self.direction
        segs = []
        for xi in self.intersections:
            p = np.array([xi, 0.0])
            a, b = p - t * L, p + t * L
            segs.append((a[0], a[1], b[0], b[1]))
        return segs

    def centerline(self):
        """Navigation centerline through the midpoint of the two flanking
        rows: (x0, y0, x1, y1) or None."""
        e = self.e_indices
        if len(e) < 2:
            return None
        order = np.argsort(np.abs(e))
        i, j = order[0], order[1]
        xim = (self.intersections[i] + self.intersections[j]) / 2.0
        h, w = self.magnitude.shape
        L = 1.6 * float(np.hypot(h, w))
        t = self.direction
        a, b = np.array([xim, 0.0]) - t * L, np.array([xim, 0.0]) + t * L
        return a[0], a[1], b[0], b[1]


class DFTRowDetector:
    """Bandpass DFT row detector with Hanning window, Taylor sub-bin peak
    refinement and DTFT phase estimation."""

    def __init__(self,
                 min_period_px: float = 8.0,
                 max_period_px: Optional[float] = None,
                 spacing_prior_px: Optional[float] = None,
                 spacing_tol: float = 0.3):
        self.min_period_px = float(min_period_px)
        self.max_period_px = max_period_px
        self.spacing_prior_px = spacing_prior_px
        self.spacing_tol = float(spacing_tol)

    def detect(self, img: np.ndarray,
               ref_xy: Optional[Tuple[float, float]] = None) -> Detection:
        img = np.asarray(img, dtype=np.float64)
        h, w = img.shape
        if img.ndim != 2 or h < 16 or w < 16:
            raise ValueError("need a 2D image >= 16 px per side")

        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        imgw = img * win

        F = np.fft.fftshift(np.fft.fft2(imgw))
        mag = np.abs(F)
        FX = np.fft.fftshift(np.fft.fftfreq(w))
        FY = np.fft.fftshift(np.fft.fftfreq(h))
        FR = np.hypot(FY[:, None], FX[None, :])

        f_hi = 1.0 / max(self.min_period_px, 2.0)
        max_p = self.max_period_px or min(h, w) / 2.2
        max_p = min(max_p, min(h, w) / 3.0)
        f_lo = 1.0 / max(max_p, 2.0 * self.min_period_px)
        if self.spacing_prior_px:
            f0 = 1.0 / self.spacing_prior_px
            f_lo = max(f_lo, f0 / (1.0 + self.spacing_tol))
            f_hi = min(f_hi, f0 * (1.0 + self.spacing_tol))
        if f_lo >= f_hi:
            raise ValueError("empty bandpass band; check spacing priors")

        half = (FX[None, :] > 0) | ((FX[None, :] == 0) & (FY[:, None] > 0))
        mask = (FR >= f_lo) & (FR <= f_hi) & half
        if not mask.any():
            raise ValueError("bandpass mask empty at this image size")

        masked = np.where(mask, mag, 0.0)
        yi0, xi0 = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak_val = float(mag[yi0, xi0])
        ring_med = float(np.median(mag[mask]))
        prominence = peak_val / max(ring_med, 1e-9)

        dx, dy = self._taylor_refine(mag, int(xi0), int(yi0))
        fx = float(FX[xi0] + dx / w)
        fy = float(FY[yi0] + dy / h)

        u = np.arange(w)
        v = np.arange(h)
        ex = np.exp(-2j * np.pi * u * fx)
        ey_v = np.exp(-2j * np.pi * v * fy)
        Xdtft = (ey_v @ imgw) @ ex
        phi = float(np.angle(Xdtft))

        Tx = 1.0 / fx if abs(fx) > 1e-12 else np.inf
        Ty = 1.0 / fy if abs(fy) > 1e-12 else np.inf

        direction = np.array([-fy, fx])
        nrm = float(np.hypot(*direction))
        direction = direction / nrm if nrm > 0 else np.array([0.0, 1.0])
        if direction[1] > 0:
            direction = -direction

        intersections, e_vals = self._row_positions(fx, fy, phi, w, h,
                                                    ref_xy or (w / 2.0, h - 1.0))

        e_theta = float(np.degrees(np.arctan2(direction[0], -direction[1])))

        order = np.argsort(np.abs(e_vals))
        if len(order) >= 2:
            ey_px = float((e_vals[order[0]] + e_vals[order[1]]) / 2.0)
            corridor = float(abs(e_vals[order[0]] - e_vals[order[1]]))
        else:
            ey_px, corridor = float("nan"), float("nan")

        return Detection(
            fx=fx, fy=fy, Tx=Tx, Ty=Ty, phi=phi,
            spacing_px=1.0 / float(np.hypot(fx, fy)),
            angle_from_vertical_deg=e_theta,
            intersections=intersections,
            direction=direction,
            ref_point=np.array(ref_xy or (w / 2.0, h - 1.0), dtype=float),
            e_indices=e_vals,
            ey_px=ey_px, corridor_px=corridor,
            e_theta_deg=e_theta,
            prominence=prominence,
            magnitude=mag, freq_x=FX, freq_y=FY,
            band_lo=f_lo, band_hi=f_hi,
        )

    @staticmethod
    def _taylor_refine(mag: np.ndarray, xi: int, yi: int) -> Tuple[float, float]:
        """Sub-bin peak bias via second-order Taylor expansion (Eq. 11-12):
        solve Hessian @ delta = -gradient with numerical derivatives."""
        h, w = mag.shape
        if not (1 <= xi <= w - 2 and 1 <= yi <= h - 2):
            return 0.0, 0.0
        f00 = mag[yi, xi]
        gx = 0.5 * (mag[yi, xi + 1] - mag[yi, xi - 1])
        gy = 0.5 * (mag[yi + 1, xi] - mag[yi - 1, xi])
        fxx = mag[yi, xi + 1] + mag[yi, xi - 1] - 2.0 * f00
        fyy = mag[yi + 1, xi] + mag[yi - 1, xi] - 2.0 * f00
        fxy = 0.25 * (mag[yi + 1, xi + 1] + mag[yi - 1, xi - 1]
                      - mag[yi + 1, xi - 1] - mag[yi - 1, xi + 1])
        det = fxx * fyy - fxy * fxy
        if not (fxx < 0 and fyy < 0 and abs(det) > 1e-12):
            return 0.0, 0.0
        dxs = (fxy * gy - fyy * gx) / det
        dys = (fxy * gx - fxx * gy) / det
        return float(np.clip(dxs, -1.0, 1.0)), float(np.clip(dys, -1.0, 1.0))

    @staticmethod
    def _row_positions(fx: float, fy: float, phi: float, w: int, h: int,
                       ref_xy: Tuple[float, float]):
        """Wavefront intersections on the dominant axis (Eq. 18) and their
        signed perpendicular distances to the reference point (Eq. 21)."""
        rx, ry = ref_xy
        direction = np.array([-fy, fx])
        nrm = float(np.hypot(*direction))
        direction = direction / nrm if nrm > 0 else np.array([0.0, 1.0])
        if direction[1] > 0:
            direction = -direction

        if abs(fx) >= abs(fy):
            f_ax, span = fx, w
        else:
            f_ax, span = fy, h
        base = -phi / (2.0 * np.pi)
        kmin = int(np.ceil((0.0 - base) * f_ax))
        kmax = int(np.floor((span - base) * f_ax))
        if kmax < kmin:
            kmax = kmin
        ks = np.arange(kmin, kmax + 1, dtype=np.float64)
        if len(ks) > 600:
            keep = np.linspace(0, len(ks) - 1, 600).round().astype(int)
            ks = ks[keep]
        pos = (base + ks) / f_ax

        if abs(fx) >= abs(fy):
            points = np.column_stack([pos, np.zeros_like(pos)])
        else:
            points = np.column_stack([np.zeros_like(pos), pos])
        e = ((rx - points[:, 0]) * direction[1]
             - (ry - points[:, 1]) * direction[0])
        return pos, e


def corridor_in_image(res: Detection, map_info: Optional[dict]):
    """Back-project the navigation corridor onto the original image.

    Returns a dict with polylines in image pixel coordinates for every
    detected row, the two corridor bordering rows, the navigation centerline,
    the corridor fill polygon and the reference point, or None when the
    detection cannot be mapped (no map info or no flanking row pair).
    """
    if map_info is None:
        return None
    minv = map_info["minv"]
    r0, c0 = map_info["offset"]
    gw, gh = map_info["grid"]
    t = res.direction

    def to_image(pts):
        ph = np.column_stack([pts[:, 0] + c0, pts[:, 1] + r0,
                              np.ones(len(pts))])
        q = ph @ minv.T
        return np.column_stack([q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]])

    def polyline(x_at_y0):
        s = np.linspace(-2.0 * (gw + gh), 2.0 * (gw + gh), 8001)
        xs = x_at_y0 + s * t[0]
        ys = s * t[1]
        ok = ((xs >= -c0) & (xs <= gw - 1.0 - c0)
              & (ys >= -r0) & (ys <= gh - 1.0 - r0))
        if not ok.any():
            return None
        i0 = int(np.argmax(ok))
        i1 = len(ok) - int(np.argmax(ok[::-1]))
        return to_image(np.column_stack([xs[i0:i1], ys[i0:i1]]))

    e = res.e_indices
    if len(e) < 2:
        return None
    order = np.argsort(np.abs(e))
    i, j = int(order[0]), int(order[1])
    rows = [p for p in (polyline(xi) for xi in res.intersections)
            if p is not None]
    b1, b2 = polyline(res.intersections[i]), polyline(res.intersections[j])
    if b1 is None or b2 is None:
        return None
    if len(b1) > len(b2):
        b2 = b2[np.linspace(0, len(b2) - 1, len(b1)).round().astype(int)]
    else:
        b1 = b1[np.linspace(0, len(b1) - 1, len(b2)).round().astype(int)]
    return {
        "rows": rows,
        "borders": [b1, b2],
        "centerline": polyline(0.5 * (res.intersections[i]
                                      + res.intersections[j])),
        "corridor": np.vstack([b1, b2[::-1]]),
        "ref": to_image(np.asarray([res.ref_point], dtype=float))[0],
    }
