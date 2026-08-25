"""
CAROLIF: Clustering Algorithm based RObust LIne Fitting
=======================================================
Re-implementation of the crop row detection method from:

    Khan M.N., Rahi A., Rajendran V.P., Al Hasan M., Anwar S. (2024)
    "Real-time crop row detection using computer vision - application in
     agricultural robots", Frontiers in Artificial Intelligence 7:1435686.
     doi: 10.3389/frai.2024.1435686

Pipeline (Algorithm 1 of the paper):
  Step 1  Pre-processing
    1.1  Region of Interest (ROI) selection (bottom part of image, >= 3 rows)
    1.2  Projective transformation (homography) so that converging crop rows
         become parallel straight lines
    1.3  Segmentation with normalized RGB Excess Green (ExG) + Otsu threshold
    1.4  Noise reduction with morphological opening and closing
  Step 2  Clustering
    2.1  HDBSCAN clustering of the white pixels (main tuning parameter:
         min cluster size)
    2.2  If fewer than `expected_min_rows` clusters are found, assume rows have
         merged under weed pressure and iteratively delete outliers
         (low membership-strength points) until they separate
    2.3  Crop row distance check: clusters closer to each other than the row
         distance threshold are weeds -> delete the smaller one (fewer points,
         smaller height)
  Step 3  Line fitting
    3.1  RANSAC line fitting on each cluster (robust against weed outliers)
    3.2  Slope threshold check: keep lines whose angle is within
         [70, 110] degrees; discard the rest
    3.3  Map fitted lines back onto the original image

Author: engineered from the paper description.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:                                    # prefer the reference hdbscan package
    from hdbscan import HDBSCAN
except ImportError:                     # sklearn >= 1.3 fallback
    from sklearn.cluster import HDBSCAN


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class CarolifConfig:
    # ---- Step 1: pre-processing -------------------------------------------
    roi_top_frac: float = 0.42        # ROI starts at this fraction of height
    max_side: int = 1100              # downscale images larger than this
    dst_width_scale: float = 1.35     # width of virtual top-view plane
    vp_fallback_xy: Tuple[float, float] = (0.5, -0.75)   # in ROI units (x*w, y*h)
    vp_bottom_span: Tuple[float, float] = (0.04, 0.96)   # bottom edge of source quad
    # segmentation / noise reduction
    morph_kernel: int = 3
    morph_open_iters: int = 1
    morph_close_iters: int = 2
    # ---- Step 2: clustering -------------------------------------------------
    pool_keep_frac: float = 0.12      # pooled cell counts as occupied above
    min_cluster_size: int = 30        # the paper's main tuning parameter (floor)
    min_cluster_size_frac: float = 0.008    # ...but at least this share of pts
    min_samples: int = 5
    max_cluster_points: int = 60000   # subsample cap for speed (fixed seed)
    expected_min_rows: int = 3        # fewer clusters => merged rows assumption
    refine_iters: int = 8             # iterative outlier deletion budget
    refine_keep_frac: float = 0.85    # fraction kept per refinement iteration
    merge_ratio: float = 2.5          # oversized if > ratio * median cluster size
    row_dist_factor: float = 0.55     # gap < factor*median gap => weed cluster
    min_cluster_points: int = 40      # discard tiny clusters outright
    min_height_frac: float = 0.25     # row clusters must span >= this share of
                                      # the top-view height (weed patches are short)
    # ---- grid pooling (canopy robustness + speed) ---------------------------
    pool_target_cells: int = 150      # aim for ~this many cells across view
    # ---- Step 3: line fitting ----------------------------------------------
    ransac_thresh: float = 3.0        # inlier perpendicular distance (px)
    ransac_iters: int = 600
    ransac_confidence: float = 0.99
    slope_range: Tuple[float, float] = (70.0, 110.0)  # degrees, from horizontal
    seed: int = 42

    timings: Dict[str, float] = field(default_factory=dict, repr=False)


# ----------------------------------------------------------------------------
# Helpers - Step 1
# ----------------------------------------------------------------------------
def normalize_exg(bgr: np.ndarray):
    """Normalized-RGB Excess Green (paper Eqs. 2 & 9): ExG = 2g - r - b."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    eps = 1e-6
    s = r + g + b + eps
    rn, gn, bn = r / s, g / s, b / s
    exg = 2.0 * gn - rn - bn                      # range [-1, 1]
    scaled = np.clip((exg + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return exg, scaled


def segment_green(roi_bgr: np.ndarray, cfg: CarolifConfig) -> np.ndarray:
    """
    ExG + Otsu thresholding followed by morphological opening/closing.
    Otsu is computed over vegetation candidates only (ExG > 0) so that in
    fully-covered canopies it still separates lit rows from shaded ones
    rather than slicing highlight noise out of a uniform green scene.
    """
    exg, scaled = normalize_exg(roi_bgr)
    cand = exg > 0.05                              # vegetation candidates only
    if cand.sum() < 50:                            # essentially no vegetation
        return np.zeros(roi_bgr.shape[:2], np.uint8)
    thr, _ = cv2.threshold(scaled[cand], 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((scaled >= thr) & cand).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                            iterations=cfg.morph_open_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=cfg.morph_close_iters)
    return mask


def estimate_vanishing_point(mask: np.ndarray,
                             cfg: CarolifConfig) -> Tuple[float, float]:
    """
    Estimate the vanishing point of the crop rows inside the ROI.

    Uses Hough segments on the green mask; the vanishing point is the median
    of pairwise segment intersections located above the ROI bottom. Falls back
    to a default point above the image centre when estimation fails.
    """
    h, w = mask.shape[:2]
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=60,
                           minLineLength=int(0.12 * h), maxLineGap=12)
    fallback = (cfg.vp_fallback_xy[0] * w, cfg.vp_fallback_xy[1] * h)
    if segs is None or len(segs) < 2:
        return fallback

    lines = []
    for x1, y1, x2, y2 in segs[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if not (25.0 <= ang <= 155.0):            # near-vertical row segments
            continue
        n = np.hypot(dx, dy) + 1e-9
        lines.append(((x1, y1), (dx / n, dy / n)))
    if len(lines) < 2:
        return fallback

    pts = []
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(lines), size=min(len(lines), 40), replace=False)
    sel = [lines[i] for i in idx]
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            (p1, d1), (p2, d2) = sel[i], sel[j]
            det = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(det) < 1e-6:
                continue
            t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / det
            xi, yi = p1[0] + t * d1[0], p1[1] + t * d1[1]
            if -1.5 * w <= xi <= 2.5 * w and yi <= h:   # above/beside ROI bottom
                pts.append((xi, yi))
    if len(pts) < 5:
        return fallback

    arr = np.array(pts, dtype=np.float64)
    med = np.median(arr, axis=0)
    keep = arr[np.linalg.norm(arr - med, axis=1) < 0.5 * w]
    vx, vy = np.median(keep if len(keep) >= 5 else arr, axis=0)

    # A usable VP must sit above the ROI (rows converge upwards); otherwise
    # the boundary trapezoid would self-intersect -> fall back, keeping the
    # horizontal estimate when it is sane.
    fb_x, fb_y = fallback
    if not (-0.5 * w <= vx <= 1.5 * w):
        vx = fb_x
    if vy >= 0.02 * h:
        vy = fb_y
    return float(vx), float(vy)


def build_homography(roi_shape: Tuple[int, int], vp: Tuple[float, float],
                     cfg: CarolifConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the planar homography H mapping the ROI plane to a virtual top view
    where the (converging) crop rows become parallel vertical lines.

    Boundary points (paper Sec. 3.1.1.1): a trapezoid whose side edges pass
    through the vanishing point, mapped onto a full rectangle.
    Returns (H, H_inv).
    """
    h, w = roi_shape[:2]
    vx, vy = vp
    bx1, bx2 = cfg.vp_bottom_span[0] * w, cfg.vp_bottom_span[1] * w

    def at_y(y: float) -> Tuple[float, float]:
        t = (y - vy) / (h - vy + 1e-9)
        return vx + (bx1 - vx) * t, vx + (bx2 - vx) * t

    y_top = 0.0
    tx1, tx2 = at_y(y_top)
    src = np.array([[bx1, h], [tx1, y_top], [tx2, y_top], [bx2, h]],
                   dtype=np.float32)
    dw = int(round(cfg.dst_width_scale * w))
    dh = int(round(cfg.dst_width_scale * h))
    dst = np.array([[0, dh - 1], [0, 0], [dw - 1, 0], [dw - 1, dh - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H, cv2.invert(H)[1]


# ----------------------------------------------------------------------------
# Helpers - Step 2
# ----------------------------------------------------------------------------
def white_pixel_points(mask: np.ndarray,
                       cap: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates of white pixels; random subsample if above `cap`."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) == 0:
        return pts, np.arange(0)
    if len(pts) > cap:
        sel = rng.choice(len(pts), size=cap, replace=False)
        pts = pts[sel]
    return pts, np.arange(len(pts))


def pool_mask_points(mask: np.ndarray,
                     cfg: CarolifConfig) -> Tuple[np.ndarray, int]:
    """
    Grid-pool the top-view mask before clustering.

    Canopy shadows punch holes into row bands; clustering raw speckle splits a
    single row into many fragments. Pooling to ~`pool_target_cells` columns
    bridges those holes (a cell counts as occupied above `pool_keep_frac`
    coverage) and shrinks the point count so HDBSCAN stays fast.
    """
    h, w = mask.shape[:2]
    cell = max(2, round(min(h, w) / cfg.pool_target_cells))
    gh, gw = max(1, h // cell), max(1, w // cell)
    pooled = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_AREA)
    occ = pooled > 255.0 * cfg.pool_keep_frac
    ys, xs = np.nonzero(occ)
    pts = np.column_stack([xs * cell + cell / 2.0,
                           ys * cell + cell / 2.0]).astype(np.float32)
    return pts, cell


def run_hdbscan(points: np.ndarray, cfg: CarolifConfig,
                min_cluster_size: Optional[int] = None) -> HDBSCAN:
    return HDBSCAN(min_cluster_size=min_cluster_size or cfg.min_cluster_size,
                   min_samples=cfg.min_samples,
                   metric="euclidean",
                   cluster_selection_method="eom").fit(points)

# ----------------------------------------------------------------------------
# Helpers - Step 3
# ----------------------------------------------------------------------------
def ransac_line(points: np.ndarray, cfg: CarolifConfig
                ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """RANSAC straight-line fit; returns two far-apart on-line points."""
    n = len(points)
    if n < 2:
        return None
    rng = np.random.default_rng(cfg.seed)
    best_inliers, best_count = None, 0
    thresh2 = cfg.ransac_thresh ** 2
    for _ in range(cfg.ransac_iters):
        i, j = rng.choice(n, size=2, replace=False)
        p, q = points[i], points[j]
        d = q - p
        norm2 = d[0] * d[0] + d[1] * d[1]
        if norm2 < 1e-9:
            continue
        # squared perpendicular distance of all points from line(p, q)
        diff = points - p
        cross2 = (diff[:, 0] * d[1] - diff[:, 1] * d[0]) ** 2
        inliers = cross2 / norm2 <= thresh2
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count, best_inliers = cnt, inliers
            # early exit per RANSAC confidence bound
            e = max(1e-9, 1.0 - best_count / n)
            if e ** cfg.ransac_iters < 1.0 - cfg.ransac_confidence:
                break
    if best_inliers is None or best_count < 2:
        return None

    pts = points[best_inliers]
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    direction = vt[0]

    # Wide canopy bands give RANSAC an arbitrarily-placed thin inlier slab;
    # keep its orientation but recenter the line on the cluster's median so
    # the fitted line runs along the band centre (robust against outliers).
    normal = np.array([-direction[1], direction[0]])
    c = c + normal * float(np.median((points - c) @ normal))

    t = (points - c) @ direction
    p1, p2 = c + t.min() * direction, c + t.max() * direction
    return p1, p2, best_count / n


def htransform(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N,2) array of points."""
    p = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ H.T
    return p[:, :2] / (p[:, 2:3] + 1e-9)


def extend_line_full_height(p1: np.ndarray, p2: np.ndarray,
                            width: float, height: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extend a fitted line so it spans the whole top-view height (the paper
    superimposes each fitted straight line over the full crop row extent).
    Falls back to the original segment when the line is near-horizontal.
    """
    d = p2 - p1
    n = np.hypot(*d)
    if n < 1e-9:
        return p1, p2
    u = d / n
    c = 0.5 * (p1 + p2)
    if abs(u[1]) < 1e-6:
        return p1, p2
    t_top = (0.0 - c[1]) / u[1]
    t_bot = (height - c[1]) / u[1]
    lo, hi = sorted((t_top, t_bot))
    # clamp horizontally so the drawn segment stays within a generous margin
    pad = 2.0 * max(width, height)
    if abs(u[0]) > 1e-6:
        cand = [((0 - c[0]) / u[0]), ((width - c[0]) / u[0])]
        lo = max(lo, min(cand) - pad)
        hi = min(hi, max(cand) + pad)
    return c + lo * u, c + hi * u


def line_angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle of the line vs horizontal, folded into [0, 180)."""
    ang = np.degrees(np.arctan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])))
    return abs(ang) % 180.0


# ----------------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------------
class CAROLIF:
    """CAROLIF crop row detector (Khan et al., 2024)."""

    def __init__(self, config: Optional[CarolifConfig] = None):
        self.cfg = config or CarolifConfig()

    # ------------------------------------------------------------------ detect
    def _calibrate_vp(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        Automatic boundary-point selection (paper Sec. 3.1.1.1, automated):
        grid-search vanishing point candidates and keep the one whose top view
        yields the most pronounced parallel vertical row structure (peaky
        column projection). Falls back to the Hough estimate / defaults.
        """
        cfg = self.cfg
        h, w = mask.shape[:2]

        small_w = 220
        small = cv2.resize(mask, (small_w, max(24, int(small_w * h / w))),
                           interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        sh, sw = small.shape
        dst_small = CAROLIF._dst_size((sh, sw), cfg)

        def score(vpx: float, vpy: float) -> float:
            Hc, _ = build_homography((sh, sw), (vpx, vpy), cfg)
            warped = cv2.warpPerspective(small, Hc, dst_small,
                                         flags=cv2.INTER_NEAREST)
            area = float((warped > 0).sum())
            if area < 0.02 * warped.size:          # degenerate quad
                return -1.0
            # Parallel vertical rows <=> band boundaries are vertical lines
            # <=> image-gradient energy concentrates along the horizontal.
            gx = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
            mag2 = float((gx * gx + gy * gy).sum()) + 1e-9
            vert_frac = float((gx * gx).sum()) / mag2
            return vert_frac * float(area / warped.size) ** 0.25

        coarse = estimate_vanishing_point(mask, cfg)   # data-driven prior
        candidates = [coarse] if coarse[1] < 0 else []
        vx_c = 0.5 * w
        for fx in (-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25):
            for fy in (-0.40, -0.70, -1.00, -1.50, -2.20, -3.20, -4.50):
                candidates.append((vx_c + fx * w, fy * h))
        best_vp, best_s = None, -1.0
        for vp_c in candidates:
            s = score(vp_c[0], vp_c[1])
            if s > best_s:
                best_vp, best_s = vp_c, s

        # local refinement around the winner (fixes residual row tilt)
        if best_vp is not None:
            for _ in range(2):
                vx, vy = best_vp
                for dx, dy in ((-0.05, 0), (0.05, 0), (0, -0.35 * abs(vy)),
                               (0, 0.35 * abs(vy))):
                    vp_n = (vx + dx * w, vy + dy)
                    s = score(vp_n[0], vp_n[1])
                    if s > best_s:
                        best_vp, best_s = vp_n, s
        return best_vp if best_vp is not None else (vx_c, -1.0 * h)

    def detect(self, bgr: np.ndarray) -> Dict:
        """
        Run the full pipeline on a BGR image.

        Returns a dict with:
          lines_orig   : list of ((x1,y1),(x2,y2)) in ORIGINAL image pixels
          lines_roi    : same, in ROI coordinates
          lines_view   : fitted segments inside the transformed (top) view
          intermediates: visualisation frames (original/roi/mask/view/clusters)
          timings      : ms spent per step
        """
        cfg = self.cfg
        cfg.timings = {}
        rng = np.random.default_rng(cfg.seed)
        T = cfg.timings

        t0 = time.perf_counter()
        scale = 1.0
        if max(bgr.shape[:2]) > cfg.max_side:
            scale = cfg.max_side / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        T["resize"] = (time.perf_counter() - t0) * 1000

        # -- Step 1.1: ROI --------------------------------------------------
        t0 = time.perf_counter()
        H_full, W_full = bgr.shape[:2]
        y0 = int(cfg.roi_top_frac * H_full)
        roi = bgr[y0:H_full, :]
        T["roi"] = (time.perf_counter() - t0) * 1000

        # -- Step 1.3/1.4: segmentation + morphology -------------------------
        t0 = time.perf_counter()
        b, g, r = cv2.split(roi.astype(np.float32))
        eps = 1e-6
        s = r + g + b + eps
        exg_raw = 2.0 * (g / s) - (r / s) - (b / s)
        if float((exg_raw > 0.05).sum()) < 0.06 * roi.shape[0] * roi.shape[1]:
            # no meaningful vegetation (e.g. bare soil) -> nothing to detect
            mask = np.zeros(roi.shape[:2], np.uint8)
            return self._empty_result(bgr, roi, mask, T)
        mask = segment_green(roi, cfg)
        T["segment"] = (time.perf_counter() - t0) * 1000

        # -- Step 1.2: projective transformation ------------------------------
        t0 = time.perf_counter()
        vp = self._calibrate_vp(mask)
        T["vp_calibration"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        Hm, Hinv = build_homography(roi.shape, vp, cfg)
        view_size = self._dst_size(roi.shape, cfg)
        view = cv2.warpPerspective(roi, Hm, view_size,
                                   flags=cv2.INTER_LINEAR)
        mask_t = cv2.warpPerspective(mask, Hm, view_size,
                                     flags=cv2.INTER_NEAREST)
        T["transform"] = (time.perf_counter() - t0) * 1000

        # -- Step 2: clustering -----------------------------------------------
        t0 = time.perf_counter()
        points, cell = pool_mask_points(mask_t, cfg)
        labels, strengths, pidx = self._cluster_with_refinements(points, rng)

        # polish calibration using the found clusters (rows -> vertical), then
        # re-project all working data through the refined homography
        if labels.size and int(labels.max()) >= 0:
            roi_all = htransform(points, Hinv)          # full array, ROI frame
            labels_full = np.full(len(points), -1, dtype=int)
            labels_full[pidx] = labels                  # subset-aligned labels
            vp2 = self._refine_vp_with_clusters(roi_all, labels_full, vp,
                                                roi.shape, cfg)
            if (abs(vp2[0] - vp[0]) > 0.5) or (abs(vp2[1] - vp[1]) > 0.5):
                vp = vp2
                Hm, Hinv = build_homography(roi.shape, vp, cfg)
                view = cv2.warpPerspective(roi, Hm, view_size,
                                           flags=cv2.INTER_LINEAR)
                mask_t = cv2.warpPerspective(mask, Hm, view_size,
                                             flags=cv2.INTER_NEAREST)
                points = htransform(roi_all, Hm)        # same length as before
        clusters = self._geometry_filter(points[pidx], labels, pidx, rng,
                                         float(mask_t.shape[0]))
        T["cluster"] = (time.perf_counter() - t0) * 1000

        # -- Step 3: RANSAC line fitting + slope check ------------------------
        t0 = time.perf_counter()
        lines_view, lines_roi, lines_orig, kept_clusters = [], [], [], []
        for cid, idx in clusters.items():
            fit = ransac_line(points[idx], cfg)
            if fit is None:
                continue
            p1, p2, inlier_ratio = fit
            ang = line_angle_deg(p1, p2)
            lo, hi = cfg.slope_range
            if not (lo <= ang <= hi):
                continue
            p1, p2 = extend_line_full_height(p1, p2, float(mask_t.shape[1]),
                                             float(mask_t.shape[0]))
            seg_v = (tuple(p1), tuple(p2))
            lines_view.append(seg_v + (inlier_ratio,))
            seg_r = tuple(self._to_roi(pt, Hinv) for pt in (p1, p2))
            lines_roi.append(seg_r)
            seg_o = tuple(self._to_original(pt, Hinv, y0, scale)
                          for pt in (p1, p2))
            lines_orig.append(seg_o)
            kept_clusters.append(cid)
        T["fit_lines"] = (time.perf_counter() - t0) * 1000

        intermediates = {
            "roi": roi.copy(),
            "mask": mask.copy(),
            "view": view.copy(),
            "clusters": self._render_clusters(mask_t, points[pidx], labels,
                                              kept_clusters, cell),
            "detection_view": self._draw(view, lines_view),
            "vp": vp,
        }
        return {
            "lines": lines_orig,
            "lines_roi": lines_roi,
            "lines_view": lines_view,
            "n_clusters_raw": int(labels.max() + 1) if labels.size else 0,
            "n_rows": len(lines_orig),
            "intermediates": intermediates,
            "timings": dict(T),
        }

    # ------------------------------------------------------------- internals
    def _refine_vp_with_clusters(self, pts_roi: np.ndarray, labels: np.ndarray,
                                 vp: Tuple[float, float],
                                 roi_shape, cfg: CarolifConfig
                                 ) -> Tuple[float, float]:
        """
        Second calibration pass: nudge the vanishing point so that the already
        clustered row directions become as vertical as possible in the top
        view. Reuses the existing labels (no re-clustering), so it is cheap.
        """
        sizes = {c: int((labels == c).sum()) for c in np.unique(labels) if c >= 0}
        big = [c for c, n in sizes.items() if n >= cfg.min_cluster_points]
        if len(big) < 1:
            return vp

        h_roi = float(roi_shape[0])

        def angular_err(vpx: float, vpy: float) -> float:
            Hc, _ = build_homography(roi_shape, (vpx, vpy), cfg)
            errs = []
            for c in big:
                q = htransform(pts_roi[labels == c], Hc)
                q -= q.mean(axis=0)
                _, _, vt = np.linalg.svd(q, full_matrices=False)
                ang = abs(np.degrees(np.arctan2(vt[0][1], vt[0][0]))) % 180.0
                errs.append(abs(ang - 90.0))
            return float(np.mean(errs)) if errs else 180.0

        best_vp, best_err = vp, angular_err(vp[0], vp[1])
        for _ in range(2):
            vx, vy = best_vp
            improved = False
            for dx, dy in ((-0.03 * h_roi, 0), (0.03 * h_roi, 0),
                           (0, 0.25 * abs(vy)), (0, -0.25 * abs(vy))):
                cand = (vx + dx, vy + dy)
                e = angular_err(cand[0], cand[1])
                if e < best_err - 1e-4:
                    best_vp, best_err, improved = cand, e, True
            if not improved:
                break
        return best_vp

    def _empty_result(self, bgr, roi, mask, T) -> Dict:
        return {
            "lines": [], "lines_roi": [], "lines_view": [],
            "n_clusters_raw": 0, "n_rows": 0,
            "intermediates": {
                "roi": roi.copy(), "mask": mask.copy(),
                "view": np.zeros((1, 1, 3), np.uint8),
                "clusters": np.zeros((1, 1, 3), np.uint8),
                "detection_view": np.zeros((1, 1, 3), np.uint8),
                "vp": (0.0, 0.0),
            },
            "timings": dict(T),
        }

    @staticmethod
    def _dst_size(roi_shape: Tuple[int, ...],
                  cfg: CarolifConfig) -> Tuple[int, int]:
        h, w = roi_shape[:2]
        return (int(round(cfg.dst_width_scale * w)),
                int(round(cfg.dst_width_scale * h)))

    @staticmethod
    def _to_roi(pt: np.ndarray, Hinv: np.ndarray) -> Tuple[int, int]:
        v = Hinv @ np.array([pt[0], pt[1], 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    @staticmethod
    def _to_original(pt: np.ndarray, Hinv: np.ndarray, y0: float,
                     scale: float) -> Tuple[int, int]:
        rx, ry = CAROLIF._to_roi(pt, Hinv)
        return int(round(rx / scale)), int(round((ry + y0) / scale))

    # Step 2 core: clustering + preventive measures (paper Sec. 3.1.2)
    def _cluster_with_refinements(self, points, rng):
        """
        HDBSCAN with the paper's preventive measures:
        - fewer than `expected_min_rows` clusters  => rows merged by weeds,
          iteratively delete outliers (weakest membership) and re-cluster;
        - a cluster holding far more points than typical crop-row clusters is
          itself a merge => keep deleting its outliers until it separates.
        """
        cfg = self.cfg
        if len(points) == 0:
            return np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=int)

        mcs = max(cfg.min_cluster_size,
                  int(cfg.min_cluster_size_frac * len(points)))
        idx_all = np.arange(len(points))
        model = run_hdbscan(points, cfg, mcs)
        labels = model.labels_.astype(int)
        strengths = model.probabilities_.copy()

        for _ in range(cfg.refine_iters):
            n_clusters = int(labels.max() + 1) if labels.size else 0
            counts = {c: int((labels == c).sum()) for c in range(n_clusters)}
            merged = any(self._is_merged(n, counts) for n in counts.values())
            if n_clusters >= cfg.expected_min_rows and not merged:
                break
            thr = np.quantile(strengths, 1.0 - cfg.refine_keep_frac)
            keep = strengths > thr
            if keep.sum() < max(mcs * 2 // 3,
                                cfg.min_cluster_points) or keep.all():
                break
            points, strengths = points[keep], strengths[keep]
            idx_all = idx_all[keep]
            model = run_hdbscan(points, cfg, mcs)
            labels = model.labels_.astype(int)
            strengths = model.probabilities_
        return labels, strengths, idx_all

    def _is_merged(self, n: int, counts: Dict[int, int]) -> bool:
        others = [m for c, m in counts.items() if m != n]
        if not others:
            return False
        med = float(np.median(others))
        return n > self.cfg.merge_ratio * med and n > 3 * self.cfg.min_cluster_size

    # Step 2.3: crop-row-distance check between clusters
    def _geometry_filter(self, points, labels, pidx, rng, view_h: float):
        """
        `points`/`labels` describe the (possibly reduced) clustered subset;
        `pidx` maps each of its rows back to the full point array so returned
        cluster index arrays index the ORIGINAL point list.
        """
        cfg = self.cfg
        if labels.size == 0:
            return {}
        local = {}
        min_height = cfg.min_height_frac * view_h
        for c in range(int(labels.max()) + 1):
            sel = np.nonzero(labels == c)[0]
            if len(sel) < cfg.min_cluster_points:
                continue
            # weed patches are short: crop-row clusters span most of the view
            if points[sel][:, 1].ptp() < min_height:
                continue
            local[c] = sel
        if len(local) < 2:
            return {c: pidx[sel] for c, sel in local.items()}

        centers = {c: float(points[sel][:, 0].mean()) for c, sel in local.items()}
        heights = {c: float(points[sel][:, 1].ptp()) for c, sel in local.items()}
        order = sorted(centers, key=lambda c: centers[c])
        gaps = {order[i]: centers[order[i + 1]] - centers[order[i]]
                for i in range(len(order) - 1)}
        med_gap = float(np.median(list(gaps.values()))) if gaps else 0.0
        if med_gap <= 0:
            return {c: pidx[sel] for c, sel in local.items()}

        changed = True
        while changed and len(local) > 1:
            changed = False
            cs = sorted(local, key=lambda c: centers[c])
            for i in range(len(cs) - 1):
                a, b = cs[i], cs[i + 1]
                if centers[b] - centers[a] < cfg.row_dist_factor * med_gap:
                    # delete the weaker cluster: fewer points AND smaller height
                    weak = a if (len(local[a]), heights[a]) <= \
                        (len(local[b]), heights[b]) else b
                    del local[weak]
                    changed = True
                    break
        return {c: pidx[sel] for c, sel in local.items()}

    @staticmethod
    def _render_clusters(mask_t, points, labels, kept_ids, cell: int = 2):
        vis = cv2.cvtColor(mask_t, cv2.COLOR_GRAY2BGR)
        if labels.size == 0:
            return vis
        kept = set(kept_ids)
        rng = np.random.default_rng(0)
        half = max(1, cell // 2)
        for c in np.unique(labels):
            sel = points[labels == c].astype(int)
            if c < 0:                       # outliers rendered black
                col = (0, 0, 0)
            elif c in kept:
                col = (int(rng.integers(60, 255)), int(rng.integers(60, 255)),
                       int(rng.integers(60, 255)))
            else:
                col = (128, 128, 128)
            for x, y in sel:
                x0, y0 = max(0, x - half), max(0, y - half)
                vis[y0:y0 + cell, x0:x0 + cell] = col
        return vis

    @staticmethod
    def _draw(img, lines):
        out = img.copy()
        for p1, p2, *_ in lines:
            cv2.line(out, (int(round(p1[0])), int(round(p1[1]))),
                     (int(round(p2[0])), int(round(p2[1]))),
                     (0, 0, 255), 3, cv2.LINE_AA)
        return out
