"""Frequency-domain crop-row detection (Gai et al., 2024).

Reproduces Sections 2.2.2-2.2.4 of
"Robust Crop Row Detection using Discrete Fourier Transform for Vision-based
In-Field Navigation" (Computers and Electronics in Agriculture, 2024):

  * 2D Hanning window           (eq. 9-10)
  * FFT magnitude/phase spectrum (text after eq. 10)
  * bandpass + coarse peak      (eq. 11 section)
  * Taylor sub-pixel refinement (eq. 11-12)
  * single-point DTFT phase     (eq. 13-14)
  * row geometry: periods, direction, x-intercepts (eq. 15-18)
  * deviations: e_y, e_theta    (eq. 19-23)

Coordinate conventions
----------------------
Arrays are arr[x, y] with x = row index (increasing downward) and y = column
index (increasing rightward), matching the paper's (M, N) = (rows, cols) with
x in [0, M-1] and y in [0, N-1] (eq. 10, 13).

Notational corrections vs. the paper (see kimi.md for the reasoning):
  * eq. 18/20: x-intercepts carry the physically correct 1/(2 pi) factor
        x_k = Tx * (k - phi/(2 pi))
  * eq. 21: d_k is the TRUE signed perpendicular distance p_k . n_hat (unit
    wavefront normal), not the printed dot product with the row direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Windows and spectra
# ---------------------------------------------------------------------------

def hanning_1d(n: int) -> np.ndarray:
    """1D Hanning window, eq. (9): w(n) = 0.5 - 0.5 cos(2 pi n/(N-1))."""
    if n <= 1:
        return np.ones(n)
    nn = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * nn / (n - 1))


def hanning_2d(shape: tuple[int, int]) -> np.ndarray:
    """2D Hanning window, eq. (10): separable product of two 1D windows."""
    m, n = shape
    return np.outer(hanning_1d(m), hanning_1d(n))


def fft_spectrum(image: np.ndarray):
    """Window the image, FFT it and return shifted magnitude spectrum.

    Returns
    -------
    mag : (M, N) float array, shifted magnitude spectrum f(x, y)
    fx  : (M,)   float array, frequencies (cycles/pixel) along axis 0 (x)
    fy  : (N,)   float array, frequencies (cycles/pixel) along axis 1 (y)
    win : (M, N) float array, the applied 2D Hanning window
    """
    win = hanning_2d(image.shape)
    # Remove the window-weighted mean: its spectrum is the window transform
    # scaled by the mean, whose main lobe (2 bins wide for a Hanning window)
    # otherwise swamps genuine peaks when only a few periods fit in the image.
    wmean = np.sum(image * win) / np.sum(win)
    windowed = (image.astype(float) - wmean) * win
    spec = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.abs(spec)
    m, n = image.shape
    fx = np.fft.fftshift(np.fft.fftfreq(m))
    fy = np.fft.fftshift(np.fft.fftfreq(n))
    return mag, fx, fy, win


# ---------------------------------------------------------------------------
# Peak finding: bandpass -> coarse peak -> Taylor refinement -> DTFT phase
# ---------------------------------------------------------------------------

def bandpass_mask(fx: np.ndarray, fy: np.ndarray,
                  row_spacing_px: float,
                  spacing_tolerance: float = 0.15,
                  edge_width_frac: float = 0.10) -> np.ndarray:
    """Soft annular band-pass mask keeping periods near `row_spacing_px`.

    The mask is 1.0 for radial frequencies whose period is within the band
    [1/(P(1+tol)), 1/(P(1-tol))] and tapers to 0 over `edge_width_frac` of the
    band width on each side (raised-cosine edges).
    """
    fr = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
    fp = 1.0 / row_spacing_px
    f_lo = fp / (1.0 + spacing_tolerance)
    f_hi = fp / (1.0 - spacing_tolerance)
    w = edge_width_frac * (f_hi - f_lo)
    if w <= 0:
        w = 1e-9
    up = np.clip((fr - f_lo) / w, 0.0, 1.0)
    up = 0.5 - 0.5 * np.cos(np.pi * up)          # raised-cosine on the low edge
    dn = np.clip((f_hi - fr) / w, 0.0, 1.0)
    dn = 0.5 - 0.5 * np.cos(np.pi * dn)          # raised-cosine on the high edge
    return up * dn


def _central_hessian(f: np.ndarray, i: int, j: int) -> np.ndarray:
    """Central-difference Hessian of f at (i, j), clipped to the array."""
    im, jm = f.shape
    i0, i1 = max(i - 1, 0), min(i + 1, im - 1)
    j0, j1 = max(j - 1, 0), min(j + 1, jm - 1)
    fxx = f[i1, j] - 2.0 * f[i, j] + f[i0, j] if i1 > i0 else 0.0
    fyy = f[i, j1] - 2.0 * f[i, j] + f[i, j0] if j1 > j0 else 0.0
    fxy = (f[i1, j1] - f[i1, j0] - f[i0, j1] + f[i0, j0]) / 4.0 \
        if i1 > i0 and j1 > j0 else 0.0
    return np.array([[fxx, fxy], [fxy, fyy]])


def refine_peak_2d(f: np.ndarray, i0: int, j0: int) -> tuple[float, float]:
    """Sub-pixel refinement by Taylor expansion (eq. 11-12).

    Numeric 1st/2nd derivatives are computed by finite differences; the bias is
    the single Newton step  (dx, dy) = -H^-1 grad(f)  at the coarse peak.
    """
    im, jm = f.shape
    i = min(max(i0, 0), im - 1)
    j = min(max(j0, 0), jm - 1)
    ip = min(i + 1, im - 1)
    jp = min(j + 1, jm - 1)
    im_ = max(i - 1, 0)
    jm_ = max(j - 1, 0)

    gx = (f[ip, j] - f[im_, j]) / (ip - im_)
    gy = (f[i, jp] - f[i, jm_]) / (jp - jm_)
    H = _central_hessian(f, i, j)
    try:
        d = -np.linalg.solve(H, np.array([gx, gy]))
    except np.linalg.LinAlgError:
        d = np.array([0.0, 0.0])
    if not np.all(np.isfinite(d)):
        d = np.array([0.0, 0.0])
    # cap the step to a couple of bins for robustness
    d = np.clip(d, -2.0, 2.0)
    return i + float(d[0]), j + float(d[1])


def dtft_phase(windowed: np.ndarray, fx: float, fy: float) -> float:
    """Single-point DTFT of the windowed image at (fx, fy) (eq. 13-14).

    F(u, v) = sum_m sum_n f(m, n) exp(-j 2 pi (fx m + fy n))
    """
    m = windowed.shape[0]
    n = windowed.shape[1]
    ex = np.exp(-2j * np.pi * fx * np.arange(m))
    ey = np.exp(-2j * np.pi * fy * np.arange(n))
    F = ex @ windowed.astype(float) @ ey
    return float(np.angle(F))


def dtft_magnitude_grid(windowed: np.ndarray, fx_center: float, fy_center: float,
                        span: float = 0.5, steps: int = 21):
    """DTFT magnitude on a fine grid around (fx_center, fy_center).

    Returns (mag_grid, fx_grid, fy_grid) where fx_grid/fy_grid are bin-offset
    coordinates (relative to the FFT grid) and mag_grid is the DTFT magnitude.
    Used to refine the peak location to sub-bin precision before reading off
    the phase; a direct DTFT evaluation (rather than the FFT) has no binning.
    """
    m = windowed.shape[0]
    n = windowed.shape[1]
    offs = np.linspace(-span, span, steps)
    ex = np.exp(-2j * np.pi * (fx_center + offs / m)[:, None] * np.arange(m)[None, :])
    ey = np.exp(-2j * np.pi * (fy_center + offs / n)[None, :] * np.arange(n)[:, None])
    grid = ex @ windowed.astype(float) @ ey     # (steps, steps) complex
    mag = np.abs(grid)
    return mag, offs, offs


# ---------------------------------------------------------------------------
# Row geometry and deviations
# ---------------------------------------------------------------------------

@dataclass
class RowDetection:
    """Result of detecting one crop-row pattern in an image."""

    image_shape: tuple[int, int]
    fx: float                     # fundamental frequency, cycles/pixel (axis 0)
    fy: float                     # fundamental frequency, cycles/pixel (axis 1)
    Tx: float                     # period along x (pixels)
    Ty: float                     # period along y (pixels)
    phi: float                    # phase of the peak (radians)
    row_spacing_px: float         # perpendicular row spacing in pixels
    line_dir: np.ndarray          # unit direction l = (-Tx, Ty)/|.| (ROI frame)
    k_offsets: np.ndarray         # intercepts x_k = Tx (k - phi/2pi), k in range
    e_y: Optional[float] = None   # signed lateral deviation (map units)
    e_theta: Optional[float] = None  # signed heading deviation (radians)
    d_k: np.ndarray = field(default_factory=lambda: np.array([]))
    coarse_peak: tuple[int, int] = field(default_factory=lambda: (0, 0))
    refined_peak: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    @property
    def row_angle_deg(self) -> float:
        """Row direction angle vs. the image x-axis, folded into (-90, 90] deg.

        The modulo fold ((ang+90)%180)-90 ensures the angle is always in
        (-90, 90] regardless of the raw atan2 output, which would otherwise
        wrap unpredictably for rows near the vertical. This convention means
        the row angle in the image frame is consistent with the rover-following-
        furrows use case: the camera always sees the rows as roughly vertical.
        """
        ang = np.degrees(np.arctan2(self.line_dir[1], self.line_dir[0]))
        return float(((ang + 90.0) % 180.0) - 90.0)


def _bilinear_sample(img: np.ndarray, X: np.ndarray,
                     Y: np.ndarray) -> np.ndarray:
    """Bilinearly sample img at (X, Y) coordinate arrays (x=row, y=col)."""
    m, n = img.shape
    x = np.clip(X, 0.0, m - 1.001)
    y = np.clip(Y, 0.0, n - 1.001)
    i0 = x.astype(int)
    j0 = y.astype(int)
    fx = x - i0
    fy = y - j0
    i1 = np.minimum(i0 + 1, m - 1)
    j1 = np.minimum(j0 + 1, n - 1)
    return (img[i0, j0] * (1 - fx) * (1 - fy) +
            img[i1, j0] * fx * (1 - fy) +
            img[i0, j1] * (1 - fx) * fy +
            img[i1, j1] * fx * fy)


def trace_row_curves(image: np.ndarray, res: RowDetection,
                     slab_len: float = 48.0, overlap: float = 0.5,
                     min_amp_frac: float = 0.3,
                     step: float = 2.0) -> list[np.ndarray]:
    """Trace individual rows as curvy, possibly broken polylines.

    The global DFT solution `res` provides the carrier: the image is
    resampled in (along-row s, across-row r) coordinates, split into slabs
    along s, and each slab's perpendicular profile is phase-demodulated at
    the row frequency 1/P. The local phase gives every row's lateral offset
    within the slab, so slow curvature is followed while the global straight-
    line family cannot. Slabs with insufficient modulation amplitude (gaps,
    shadows, missing rows) yield NaN points, breaking the polylines there.

    Returns a list of (K, 2) arrays with (x, y) points per row; invalid
    slab positions are NaN so plotting shows natural gaps.
    """
    img = np.asarray(image, dtype=float)
    m, n = img.shape
    fnorm = np.hypot(res.fx, res.fy)
    if fnorm <= 0 or not np.isfinite(fnorm):
        return []
    P = 1.0 / fnorm
    lx, ly = res.line_dir
    nx, ny = res.fx / fnorm, res.fy / fnorm

    corners = [(0.0, 0.0), (0.0, float(n)), (float(m), 0.0), (float(m), float(n))]
    sv = [cx * lx + cy * ly for cx, cy in corners]
    rv = [cx * nx + cy * ny for cx, cy in corners]
    ss = np.arange(min(sv), max(sv) + step, step)
    rr = np.arange(min(rv), max(rv), 1.0)
    X = ss[None, :] * lx + rr[:, None] * nx
    Y = ss[None, :] * ly + rr[:, None] * ny
    inside = (X >= 0) & (X <= m - 1) & (Y >= 0) & (Y <= n - 1)
    patch = _bilinear_sample(img, X, Y)
    patch[~inside] = 0.0

    L = max(int(round(slab_len / step)), 8)
    hop = max(L // 2, 1) if overlap >= 0.5 else max(L // 4, 1)
    centers, amps, phis, covs = [], [], [], []
    kernel = np.exp(-2j * np.pi * rr / P)
    a = 0
    while a < len(ss):
        b = min(a + L, len(ss))
        if b - a < max(4, L // 4):
            break
        block = patch[:, a:b]
        covs.append(inside[:, a:b].mean())
        prof = block.mean(axis=1)
        prof = prof - prof.mean()
        z = complex(np.sum(prof * kernel))
        centers.append(ss[(a + b - 1) // 2])
        amps.append(abs(z))
        phis.append(np.angle(z))
        if b >= len(ss):
            break
        a += hop
    centers = np.array(centers)
    amps = np.array(amps)
    phis = np.array(phis)
    covs = np.array(covs)
    if len(centers) < 2 or np.median(amps) <= 0:
        return []

    ph = phis.copy()
    for k in range(1, len(ph)):
        ph[k] += 2.0 * np.pi * np.round((ph[k - 1] - ph[k]) / (2.0 * np.pi))
    # slabs cut by the image border see less real data; require a floor so
    # zero padding cannot fake a modulation drop (or a phantom peak)
    good = (amps >= min_amp_frac * np.median(amps)) & (covs > 0.30)

    polys = []
    for k_row in np.arange(np.floor(min(rv) / P) - 1,
                           np.ceil(max(rv) / P) + 2):
        r_vals = P * (k_row - ph / (2.0 * np.pi))
        xs = centers * lx + r_vals * nx
        ys = centers * ly + r_vals * ny
        ok = good & (xs >= -P) & (xs <= m + P) & (ys >= -P) & (ys <= n + P)
        if ok.sum() < 2:
            continue
        polys.append(np.column_stack([np.where(ok, xs, np.nan),
                                      np.where(ok, ys, np.nan)]))
    return polys


def _clip_segment(p0, p1, m: int, n: int) -> Optional[np.ndarray]:
    """Liang-Barsky clip of segment p0->p1 to [0, m-1] x [0, n-1].

    Returns (x0, y0, x1, y1) or None if the segment misses the rectangle.
    Valid for any direction vector (no sign assumptions).
    """
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(p1[0]) - x0, float(p1[1]) - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, (m - 1) - x0), (-dy, y0), (dy, (n - 1) - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
        else:
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None
    return np.array([x0 + t0 * dx, y0 + t0 * dy,
                     x0 + t1 * dx, y0 + t1 * dy])


class RowDetector:
    """Core DFT crop-row detector from the paper.

    Parameters
    ----------
    row_spacing_px : float, optional
        Prior row spacing in pixels. If None, estimated automatically from the
        radial magnitude spectrum.
    spacing_tolerance : float
        Relative half-width of the band-pass filter (default 0.15 -> +/-15%).
    """

    def __init__(self, row_spacing_px: Optional[float] = None,
                 spacing_tolerance: float = 0.15,
                 num_rows: int = 12):
        self.row_spacing_px = row_spacing_px
        self.spacing_tolerance = spacing_tolerance
        self.num_rows = num_rows

    # -- helpers ------------------------------------------------------------

    def _estimate_spacing(self, mag: np.ndarray, fx: np.ndarray,
                          fy: np.ndarray) -> float:
        """Estimate the dominant row period from the strongest spectral peak.

        The DC neighbourhood is masked out up to the frequency whose period
        fits twice into the smaller image dimension (a usable row pattern must
        repeat at least twice in the field of view); the strongest remaining
        peak gives the row frequency. The rows are the sharpest periodic
        structure in a rectified crop image, so the tallest peak outside the
        DC blob is the row fundamental.
        """
        fr = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
        spec = mag.copy()
        spec[fr < 2.0 / min(*mag.shape)] = 0.0
        idx = np.unravel_index(int(np.argmax(spec)), spec.shape)
        return float(1.0 / fr[idx])

    # -- main entry ---------------------------------------------------------

    def detect(self, image: np.ndarray,
               roi_to_map: Optional[np.ndarray] = None) -> RowDetection:
        """Detect the crop-row pattern in a (rectified) grayscale image.

        Parameters
        ----------
        image : (M, N) float array, grayscale bird's-eye view (already
            rectified by the IPM stage; row pattern must be periodic/parallel).
        roi_to_map : (3, 3) array, optional
            Homogeneous affine map ROI-pixel -> local-map-meters (scale +
            translation, see preprocessing.ROI transform). When given, the
            position/heading deviations are computed (eq. 19-23).
        """
        image = np.asarray(image, dtype=float)
        self.image_shape = image.shape
        res = self._detect_once(image, roi_to_map, self.row_spacing_px)
        if self.row_spacing_px is None:
            fp = 1.0 / res.row_spacing_px
            fnorm = float(np.hypot(res.fx, res.fy))
            if (fnorm > 0 and np.isfinite(fnorm)
                    and abs(fnorm - fp) / fp > self.spacing_tolerance):
                # The auto-estimated spacing was off: the refined peak escaped
                # its own band-pass window. Re-lock with the measured
                # frequency as the prior, then keep whichever candidate locks
                # the spectrum more sharply (the escaped one may have been
                # the true rows all along).
                res2 = self._detect_once(image, roi_to_map, 1.0 / fnorm)
                if self._peak_prominence(image, res2) > \
                        self._peak_prominence(image, res):
                    res = res2
        # report the measured perpendicular spacing of the locked peak, not
        # the prior used for band-pass centering (they can disagree when the
        # auto-estimate was off)
        fnorm = float(np.hypot(res.fx, res.fy))
        if fnorm > 0 and np.isfinite(fnorm):
            res.row_spacing_px = 1.0 / fnorm
        return res

    def _peak_prominence(self, image: np.ndarray, res: RowDetection) -> float:
        """Sharpness of the lock: bandpassed peak vs. its spectral ring.

        A genuine row fundamental stands as a narrow spike above the local
        spectral background; drift or edge artefacts form broad blobs. The
        median over a doubled-width annulus is a robust background estimate.
        """
        mag, fx, fy, _ = fft_spectrum(image)
        fnorm = float(np.hypot(res.fx, res.fy))
        if fnorm <= 0 or not np.isfinite(fnorm):
            return 0.0
        fr = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
        inner = bandpass_mask(fx, fy, 1.0 / fnorm, self.spacing_tolerance)
        outer = bandpass_mask(fx, fy, 1.0 / fnorm,
                              min(2.0 * self.spacing_tolerance, 0.95))
        ring = mag[(outer > 0) & (fr > 0)]
        if ring.size == 0:
            return 0.0
        return float((mag * inner).max() / max(np.median(ring), 1e-9))

    def _detect_once(self, image: np.ndarray,
                     roi_to_map: Optional[np.ndarray],
                     spacing_prior: Optional[float]) -> RowDetection:
        mag, fx, fy, win = fft_spectrum(image)

        if spacing_prior is None:
            spacing = self._estimate_spacing(mag, fx, fy)
        else:
            spacing = float(spacing_prior)

        mask = bandpass_mask(fx, fy, spacing, self.spacing_tolerance)
        filtered = mag * mask

        ci = np.unravel_index(int(np.argmax(filtered)), filtered.shape)
        coarse = (int(ci[0]), int(ci[1]))

        # Taylor refinement (eq. 11-12) on the band-passed magnitude
        ri, rj = refine_peak_2d(filtered, coarse[0], coarse[1])

        m, n = image.shape
        fx0 = (ri - m // 2) / m            # cycles/pixel along x (axis 0)
        fy0 = (rj - n // 2) / n            # cycles/pixel along y (axis 1)

        # refine the peak further with a fine DTFT magnitude scan (sub-bin)
        wmean = np.sum(win * image) / np.sum(win)
        zimg = win * (image - wmean)
        mag_grid, off_x, off_y = dtft_magnitude_grid(zimg, fx0, fy0,
                                                     span=0.5, steps=51)
        bi, bj = np.unravel_index(int(np.argmax(mag_grid)), mag_grid.shape)
        fx0 += off_x[bi] / m
        fy0 += off_y[bj] / n

        # DTFT phase at the refined peak (eq. 13-14)
        phi = dtft_phase(zimg, fx0, fy0)

        # canonicalize to the positive-fy peak (the two symmetric peaks carry
        # the same geometry; this keeps Tx/Ty/phi signs deterministic)
        if fy0 < 0:
            fx0, fy0 = -fx0, -fy0
            phi = -phi

        # Periods (eq. 15-16) in pixels. u is the FFT bin index = M*fx, so
        # Tx = M/u = 1/fx.
        Tx = 1.0 / fx0 if abs(fx0) > 1e-9 else float("inf")
        Ty = 1.0 / fy0 if abs(fy0) > 1e-9 else float("inf")

        # Normalized line direction (eq. 17)
        v = np.array([-Tx, Ty]) if np.isfinite(Tx) else np.array([-np.sign(fx0) * 1e9, Ty])
        if np.all(np.isfinite(v)) and np.linalg.norm(v) > 0:
            lvec = v / np.linalg.norm(v)
        else:
            lvec = np.array([0.0, 1.0])

        # x-axis intercepts (eq. 18, corrected with the 1/(2 pi) factor).
        # The k index of a point p is s(p) = fx*px + fy*py + phi/(2pi); the
        # range is chosen to cover every row line that intersects the image
        # (a fixed k window around 0 would miss the image entirely whenever
        # the phase places the family far from the coordinate origin).
        corners = [(0.0, 0.0), (0.0, float(n)),
                   (float(m), 0.0), (float(m), float(n))]
        svals = [fx0 * cx + fy0 * cy + phi / (2.0 * np.pi)
                 for cx, cy in corners]
        k = np.arange(np.floor(min(svals)) - 1, np.ceil(max(svals)) + 2)
        x_k = Tx * (k - phi / (2.0 * np.pi))

        res = RowDetection(
            image_shape=image.shape,
            fx=fx0, fy=fy0, Tx=Tx, Ty=Ty, phi=phi,
            row_spacing_px=spacing,
            line_dir=lvec,
            k_offsets=x_k,
            coarse_peak=coarse,
            refined_peak=(ri + off_x[bi], rj + off_y[bj]),
        )

        if roi_to_map is not None:
            res.d_k, res.e_y, res.e_theta = self._deviations(res, np.asarray(roi_to_map))
        return res

    # -- deviations (eq. 19-23) ---------------------------------------------

    def _deviations(self, res: RowDetection,
                    T: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Signed perpendicular distances and deviations in the map frame."""
        a = 1.0 / res.Tx if np.isfinite(res.Tx) else 0.0
        b = 1.0 / res.Ty if np.isfinite(res.Ty) else 0.0
        Tinv = np.linalg.inv(T)
        p1, p2, p3 = Tinv[0, 0], Tinv[0, 1], Tinv[0, 2]
        p4, p5, p6 = Tinv[1, 0], Tinv[1, 1], Tinv[1, 2]

        # line family in map frame: A X + B Y = C_k
        A = a * p1 + b * p4
        B = a * p2 + b * p5
        denom = np.hypot(A, B)
        if denom == 0:
            return np.array([]), 0.0, 0.0
        # canonical unit normal so that rows on the robot's right (+X in the
        # map frame) get positive signed distance; this makes the convention
        # "e_y > 0 <=> robot is left of the row centreline" hold for any angle
        lM = T[:2, :2] @ res.line_dir
        n_hat = np.array([lM[1], -lM[0]])
        n_hat /= np.linalg.norm(n_hat)
        if n_hat[0] < 0:            # keep the normal pointing toward +X
            n_hat = -n_hat
        sign = np.sign(n_hat @ np.array([A, B]) / denom)
        if sign == 0:
            sign = 1.0

        ks = np.arange(-self.num_rows, self.num_rows + 1)
        d = (ks - res.phi / (2.0 * np.pi) - (a * p3 + b * p6)) / denom
        d = sign * d

        # two nearest rows (by |d|) define the navigation centerline
        order = np.argsort(np.abs(d))
        e_y = float(np.mean(d[order[:2]]))

        # heading deviation (eq. 23) with direction canonicalized forward
        lM = np.array([lM[0], lM[1]])
        if lM[1] < 0:
            lM = -lM
        e_theta = float(-np.arctan2(lM[0], lM[1]))

        return d, e_y, e_theta

    # -- visualization helpers ----------------------------------------------

    def row_lines(self, res: RowDetection,
                  image_shape: Optional[tuple[int, int]] = None) -> list[np.ndarray]:
        """Return each row line as an (x0, y0, x1, y1) segment clipped to the image.

        Uses the normal form fx*x + fy*y = c_k of each row line, so the result
        is well conditioned for any orientation (including rows parallel to an
        image axis, where x-intercepts Tx*(k - phi/2pi) degenerate).
        """
        if image_shape is None:
            image_shape = res.image_shape
        m, n = image_shape
        fnorm = np.hypot(res.fx, res.fy)
        if fnorm <= 0 or not np.isfinite(fnorm):
            return []
        nx, ny = res.fx / fnorm, res.fy / fnorm      # unit wavefront normal
        lx, ly = -ny, nx                             # direction along the rows
        half = float(np.hypot(m, n))
        corners = [(0.0, 0.0), (0.0, float(n)),
                   (float(m), 0.0), (float(m), float(n))]
        svals = [res.fx * cx + res.fy * cy + res.phi / (2.0 * np.pi)
                 for cx, cy in corners]
        ks = np.arange(np.floor(min(svals)) - 1, np.ceil(max(svals)) + 2)
        lines = []
        for c_k in ks - res.phi / (2.0 * np.pi):
            delta = c_k / fnorm                      # signed offset along n
            cx, cy = delta * nx, delta * ny          # closest point to origin
            seg = _clip_segment([cx - half * lx, cy - half * ly],
                                [cx + half * lx, cy + half * ly], m, n)
            if seg is not None:
                lines.append(seg)
        return lines