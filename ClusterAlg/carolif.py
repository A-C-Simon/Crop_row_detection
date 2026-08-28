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


# Paper Eq.3 in MultiROI — 4x4 anti-diagonal structuring element
KERNEL_K = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=np.uint8)

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
    # segmentation / noise reduction — "struct" = preclean embedded (default),
    # "paper" = original ExG+Otsu + ellipse open/close (Khan et al.)
    morph: str = "struct"             # "struct" | "paper"
    morph_kernel: int = 3
    morph_open_iters: int = 1
    morph_close_iters: int = 2
    # preclean band-support filter (MultiROI test_multi_roi.py: pre_corridor_clean)
    # Applied INSIDE morphology when morph="struct": every blob not overlapping a
    # per-strip row-candidate band (Eq.5-8) is erased before any corridor exists.
    pre_clean_n_strips: int = 10
    pre_clean_l_frac: float = 0.05
    pre_clean_pad_px: int = 6
    pre_clean_min_area: int = 12
    pre_clean_max_blob_frac: float = 0.05
    # ---- Step 2: clustering -------------------------------------------------
    pool_keep_frac: float = 0.12      # pooled cell counts as occupied above
    min_cluster_size: int = 60        # aggressive: kill weak / weed clusters
    min_cluster_size_frac: float = 0.012    # ...but at least this share of pts
    min_samples: int = 10             # stricter density requirement
    max_cluster_points: int = 60000   # subsample cap for speed (fixed seed)
    expected_min_rows: int = 2        # fewer clusters => merged rows assumption
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
    slope_range: Tuple[float, float] = (5.0, 175.0)   # effectively disabled — all angles pass
    vertical_stretch: float = 2.0     # stretch y before clustering to favor vertical groups
    merge_col_thresh: float = 0.06    # clusters within this fraction of width are same column
    merge_y_gap_frac: float = 0.12    # vertical gap < frac of view_h → merge stacked clusters
    diverge_angle_max: float = 75.0   # lines > this angle from horizon-center direction → delete
    spline_smoothing: float = 0.003   # smoothing factor for UnivariateSpline (0=interpolate)
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


def column_projection(strip: np.ndarray, x_lo: int, x_hi: int) -> np.ndarray:
    """MultiROI Eq.5-6: vertical projection filtered by T=M+E (Eq.7-8).

    Dense-canopy guard: on saturated scenes M+E can exceed max(Z) and wipe
    out the entire profile — then fall back to P90 so strongest columns survive.
    Direct copy of test_multi_roi.py:column_projection.
    """
    window = strip[:, x_lo:x_hi]
    z = (window == 255).sum(axis=0).astype(np.float64)  # Eq.5
    t = z.mean() + z.std()                               # Eq.7-8
    if t > 0 and not np.any(z >= t):
        t = float(np.quantile(z, 0.9))
    z[z < t] = 0.0                                       # Eq.6
    return z


def cluster_feature_columns(z: np.ndarray, x_lo: int, l_thresh: int):
    """MultiROI Sec 2.3.2: group feature columns whose gap < L."""
    cols = np.nonzero(z > 0)[0]
    if cols.size == 0:
        return []
    clusters = []
    start = prev = cols[0]
    for c in cols[1:]:
        if c - prev < l_thresh:
            prev = c
        else:
            clusters.append((start + x_lo, prev + x_lo))
            start = prev = c
    clusters.append((start + x_lo, prev + x_lo))
    return clusters


def pre_corridor_clean(binary: np.ndarray, cfg: CarolifConfig):
    """EXPERIMENTAL preclean embedded in morphology (MultiROI struct).

    Per-strip row-candidate bands from Eq.5-8 define which columns carry row
    structure. Every connected component is judged at blob level: if its
    horizontal extent overlaps a band in its own or adjacent strip it survives,
    otherwise it is erased as off-structure weed. Whole blobs are kept/dropped
    (never shaved), blobs > max_blob_frac of vegetation are always spared,
    strips without bands are left untouched. Returns (cleaned, info dict).
    Direct port of test_multi_roi.py:pre_corridor_clean — validated to prevent
    wrong-lane capture (photo_9/photo_6 eye tests).
    """
    h, w = binary.shape[:2]
    n_strips = cfg.pre_clean_n_strips
    l_frac = cfg.pre_clean_l_frac
    pad_px = cfg.pre_clean_pad_px
    min_area = cfg.pre_clean_min_area
    max_blob_frac = cfg.pre_clean_max_blob_frac

    dh = h // n_strips if n_strips > 0 else h
    l_thresh = max(2, int(l_frac * w))

    # per-strip row-candidate bands
    bands = {}
    for mu in range(1, n_strips + 1):
        y1 = h - mu * dh
        ys, ye = max(0, y1), y1 + dh
        if ye <= ys:
            continue
        z = column_projection(binary[ys:ye, :], 0, w)
        bands[mu] = cluster_feature_columns(z, 0, l_thresh)

    def strip_of(y):
        return min(n_strips, max(1, (h - int(y)) // max(1, dh)))

    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    total = int(np.count_nonzero(binary))
    big_px = max_blob_frac * max(1, total)

    cleaned = binary.copy()
    removed = removed_area = 0
    for i in range(1, n_cc):
        x0, _, bw_, _, area = stats[i]
        cy = centroids[i][1]
        if area < min_area:
            continue  # speckle: opening's job
        mu = strip_of(cy)
        tol = []
        for m_u in (mu - 1, mu, mu + 1):
            tol += [(a - pad_px, b + 1 + pad_px)
                    for a, b in bands.get(m_u, [])]
        if not tol:
            continue  # no structure known here: keep
        if area > big_px:
            continue  # giant cap: too big to condemn
        hi_x = x0 + bw_ + pad_px
        lo_x = x0 - pad_px
        if any(hi_x > lo and lo_x < hi for lo, hi in tol):
            continue  # touches row structure
        cleaned[labels == i] = 0
        removed += 1
        removed_area += area
    return cleaned, {
        "applied": removed > 0,
        "n_removed": removed,
        "removed_px": removed_area,
        "removed_frac": removed_area / max(1, total),
    }


def segment_green(roi_bgr: np.ndarray, cfg: CarolifConfig) -> np.ndarray:
    """
    ExG + Otsu thresholding followed by morphological filtering.

    Two modes (cfg.morph):
      paper  — original CAROLIF: normalized RGB ExG + ellipse open + close
               (Khan et al. Eqs. 7-9, Sec. 3.1.1.3)
      struct — DEFAULT, matches test_multi_roi.py morphology exactly:
               raw ExG = 2G - R - B, Otsu on full image, KERNEL_K (4×4
               anti-diagonal) opening, then pre_corridor_clean band-support
               weed filter. No safety valve — preclean output is final.
    """
    h, w = roi_bgr.shape[:2]
    b, g, r = cv2.split(roi_bgr.astype(np.float32))

    if cfg.morph == "struct":
        # Raw ExG (test_multi_roi.py preprocess, index="raw"): 2G - R - B,
        # clipped to [0, 255].  Matches the MultiROI pipeline exactly.
        exg = 2.0 * g - r - b
        exg8 = np.clip(exg, 0.0, 255.0).astype(np.uint8)
        _, binary = cv2.threshold(exg8, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # KERNEL_K opening (Eq.3 of Zhou et al. 2021)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_K)
        # Band-support weed filter (pre_corridor_clean, no safety valve)
        binary, _ = pre_corridor_clean(binary, cfg)
        return binary
    else:
        # Paper mode: normalized RGB ExG (Khan et al. Eqs. 7-9)
        exg, scaled = normalize_exg(roi_bgr)
        cand = exg > 0.05
        if cand.sum() < 50:
            return np.zeros((h, w), np.uint8)
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
               min_cluster_size: Optional[int] = None,
               vertical_stretch: Optional[float] = None) -> HDBSCAN:
    """HDBSCAN with optional vertical stretch to favor column-like clusters.

    When vertical_stretch > 1 the y-coordinates are scaled up before clustering
    so that two points stacked vertically are treated as closer than two points
    at the same horizontal separation.  This biases HDBSCAN toward tall, narrow
    (column-shaped) clusters — exactly what crop rows look like in the top view.
    The stretch is applied *only* for clustering; the original un-stretched
    coordinates are used everywhere else (RANSAC, rendering, etc.).
    """
    vs = vertical_stretch if vertical_stretch is not None else 1.0
    if vs != 1.0 and len(points) > 0:
        stretched = points.copy()
        stretched[:, 1] *= vs   # emphasise vertical proximity
    else:
        stretched = points
    return HDBSCAN(min_cluster_size=min_cluster_size or cfg.min_cluster_size,
                   min_samples=cfg.min_samples,
                   metric="euclidean",
                   cluster_selection_method="eom").fit(stretched)

# ----------------------------------------------------------------------------
# Helpers - Step 3
# ----------------------------------------------------------------------------
def spline_fit(points: np.ndarray, cfg: CarolifConfig
               ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Fit a smoothing spline through the cluster points (curvilinear).

    Sorts points by y, fits a UnivariateSpline x = f(y) with light
    smoothing, then samples the curve at the top and bottom of the cluster
    to get two endpoint coordinates.  The 'inlier ratio' is estimated as
    the fraction of points within ransac_thresh of the spline.

    Returns (p1, p2, inlier_ratio) or None if fitting fails.
    """
    from scipy.interpolate import UnivariateSpline

    n = len(points)
    if n < 2:
        return None
    if n < 4:
        # too few for a spline — fall back to a straight line via SVD
        c = points.mean(axis=0)
        u, s, vt = np.linalg.svd(points - c, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        c = c + normal * float(np.median((points - c) @ normal))
        t = (points - c) @ direction
        p1, p2 = c + t.min() * direction, c + t.max() * direction
        return p1, p2, 1.0

    # sort by y so spline is single-valued
    order = np.argsort(points[:, 1])
    ys = points[order, 1].astype(np.float64)
    xs = points[order, 0].astype(np.float64)

    # deduplicate y values (UnivariateSpline requires strictly increasing x)
    # if there are ties, average the corresponding x values
    unique_y, inverse = np.unique(ys, return_inverse=True)
    if len(unique_y) < 4:
        # not enough unique y values — fall back to straight line
        c = points.mean(axis=0)
        u, s, vt = np.linalg.svd(points - c, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        c = c + normal * float(np.median((points - c) @ normal))
        t = (points - c) @ direction
        p1, p2 = c + t.min() * direction, c + t.max() * direction
        return p1, p2, 1.0
    unique_x = np.array([xs[inverse == k].mean() for k in range(len(unique_y))])

    # fit x = f(y) with smoothing
    try:
        spline = UnivariateSpline(unique_y, unique_x, k=min(3, len(unique_y) - 1),
                                  s=cfg.spline_smoothing * n)
    except Exception:
        # fallback to straight line
        c = points.mean(axis=0)
        u, s, vt = np.linalg.svd(points - c, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        c = c + normal * float(np.median((points - c) @ normal))
        t = (points - c) @ direction
        p1, p2 = c + t.min() * direction, c + t.max() * direction
        return p1, p2, 1.0

    y_lo, y_hi = float(ys.min()), float(ys.max())
    x_lo_val = float(spline(y_lo))
    x_hi_val = float(spline(y_hi))
    # guard against NaN from spline extrapolation
    if not (np.isfinite(x_lo_val) and np.isfinite(x_hi_val)):
        c = points.mean(axis=0)
        u, s, vt = np.linalg.svd(points - c, full_matrices=False)
        direction = vt[0]
        t = (points - c) @ direction
        p1, p2 = c + t.min() * direction, c + t.max() * direction
        return p1, p2, 1.0
    p1 = np.array([x_lo_val, y_lo], dtype=np.float32)
    p2 = np.array([x_hi_val, y_hi], dtype=np.float32)

    # estimate inlier ratio: how many points are close to the spline curve
    x_eval = spline(ys)
    dists = np.abs(xs - x_eval)
    inlier_ratio = float((dists <= cfg.ransac_thresh).sum()) / n
    return p1, p2, inlier_ratio


# keep a reference to the raw SVD fallback for the "too few points" case
def _svd_line_fallback(points: np.ndarray, cfg: CarolifConfig
                       ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    n = len(points)
    if n < 2:
        return None
    c = points.mean(axis=0)
    u, s, vt = np.linalg.svd(points - c, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    c = c + normal * float(np.median((points - c) @ normal))
    t = (points - c) @ direction
    p1, p2 = c + t.min() * direction, c + t.max() * direction
    return p1, p2, 1.0


def _merge_stacked_clusters(subset_points, labels, pidx, cfg, view_h, view_w):
    """Merge clusters that are stacked vertically in the same column direction.

    After HDBSCAN + geometry filter, two crop-row clusters may sit one above
    the other in the same column.  This function detects such pairs and merges
    them into a single cluster.

    Two clusters are merged when:
      - their horizontal centres are within merge_col_thresh * view_w
      - their vertical gap is < merge_y_gap_frac * view_h

    subset_points: the points array aligned with labels (i.e. points[pidx]).
    """
    if labels.size == 0:
        return labels, pidx

    unique_labels = sorted(set(labels) - {-1})
    if len(unique_labels) < 2:
        return labels, pidx

    # compute cluster stats using subset_points (aligned with labels)
    stats = {}
    for c in unique_labels:
        sel = np.nonzero(labels == c)[0]
        xc = float(subset_points[sel][:, 0].mean())
        y_lo = float(subset_points[sel][:, 1].min())
        y_hi = float(subset_points[sel][:, 1].max())
        stats[c] = {'sel': sel, 'xc': xc, 'y_lo': y_lo, 'y_hi': y_hi}

    # find merge pairs: same column, small vertical gap
    merge_map = {}  # child → parent
    sorted_clusters = sorted(stats.keys(), key=lambda c: stats[c]['y_lo'])
    col_thresh = cfg.merge_col_thresh * view_w
    y_gap_max = cfg.merge_y_gap_frac * view_h

    for i in range(len(sorted_clusters)):
        ci = sorted_clusters[i]
        if ci in merge_map:
            continue
        for j in range(i + 1, len(sorted_clusters)):
            cj = sorted_clusters[j]
            if cj in merge_map:
                continue
            dx = abs(stats[ci]['xc'] - stats[cj]['xc'])
            # vertical gap: bottom of upper cluster to top of lower cluster
            gap = stats[ci]['y_lo'] - stats[cj]['y_hi']  # negative = overlap
            if dx < col_thresh and gap < y_gap_max:
                # same column, vertically adjacent → merge cj into ci
                # (ci is the upper one — keep as parent)
                merge_map[cj] = ci

    if not merge_map:
        return labels, pidx

    # apply merge map (chain resolution)
    def resolve(c):
        while c in merge_map:
            c = merge_map[c]
        return c

    new_labels = labels.copy()
    for c in unique_labels:
        parent = resolve(c)
        if parent != c:
            new_labels[labels == c] = parent

    return new_labels, pidx


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


def line_intersection(p1a, p2a, p1b, p2b):
    """Find the intersection of two line segments (extended to infinite lines).
    Returns (x, y) or None if parallel."""
    da = np.array(p2a, dtype=float) - np.array(p1a, dtype=float)
    db = np.array(p2b, dtype=float) - np.array(p1b, dtype=float)
    det = da[0] * db[1] - da[1] * db[0]
    if abs(det) < 1e-9:
        return None
    t = (((np.array(p1b, dtype=float)[0] - p1a[0]) * db[1] -
          (np.array(p1b, dtype=float)[1] - p1a[1]) * db[0]) / det)
    ix = p1a[0] + t * da[0]
    iy = p1a[1] + t * da[1]
    return (ix, iy)


def angle_between_lines(p1a, p2a, p1b, p2b) -> float:
    """Angle in degrees between the directions of two lines."""
    da = np.array(p2a, dtype=float) - np.array(p1a, dtype=float)
    db = np.array(p2b, dtype=float) - np.array(p1b, dtype=float)
    cos_a = np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(cos_a), 0, 1))))


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

    def detect(self, bgr: np.ndarray, debug: bool = False) -> Dict:
        """
        Run the full pipeline on a BGR image.

        Returns a dict with:
          lines_orig   : list of ((x1,y1),(x2,y2)) in ORIGINAL image pixels
          lines_roi    : same, in ROI coordinates
          lines_view   : fitted segments inside the transformed (top) view
          intermediates: visualisation frames (original/roi/mask/view/clusters)
          timings      : ms spent per step

        When debug=True, intermediates also contains before/after images for
        every decision point (mask_before_open, mask_after_open, raw_clusters,
        refined_clusters, geo_before, geo_after, ransac_before, ransac_after,
        rejection_log).
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
        # Quick vegetation guard — raw ExG for struct mode, normalized for paper
        if cfg.morph == "struct":
            exg_guard = 2.0 * g - r - b
        else:
            eps = 1e-6
            s = r + g + b + eps
            exg_guard = 2.0 * (g / s) - (r / s) - (b / s)
        if float((exg_guard > 0.05).sum()) < 0.06 * roi.shape[0] * roi.shape[1]:
            # no meaningful vegetation (e.g. bare soil) -> nothing to detect
            mask = np.zeros(roi.shape[:2], np.uint8)
            return self._empty_result(bgr, roi, mask, T)
        mask = segment_green(roi, cfg)
        T["segment"] = (time.perf_counter() - t0) * 1000

        # -- Debug: capture intermediate masks for every decision point ------
        dbg = {}
        if debug:
            # 1) ExG + Otsu only (before any morphology)
            b_ch, g_ch, r_ch = cv2.split(roi.astype(np.float32))
            if cfg.morph == "struct":
                exg_dbg = 2.0 * g_ch - r_ch - b_ch
                exg8_dbg = np.clip(exg_dbg, 0.0, 255.0).astype(np.uint8)
            else:
                exg_dbg, exg8_dbg_norm = normalize_exg(roi)
                exg8_dbg = exg8_dbg_norm
            _, mask_otsu = cv2.threshold(exg8_dbg, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dbg["mask_otsu"] = mask_otsu.copy()
            # 2) After KERNEL_K opening (before preclean)
            mask_opened = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, KERNEL_K)
            dbg["mask_opened"] = mask_opened.copy()
            # 3) After preclean = final mask
            dbg["mask_preclean"] = mask.copy()

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

        # Debug: capture raw HDBSCAN labels before refinement
        if debug and len(points) > 0:
            mcs_dbg = max(cfg.min_cluster_size,
                          int(cfg.min_cluster_size_frac * len(points)))
            model_dbg = run_hdbscan(points, cfg, mcs_dbg,
                                    vertical_stretch=cfg.vertical_stretch)
            dbg["raw_labels"] = model_dbg.labels_.astype(int).copy()
            dbg["raw_points"] = points.copy()
        else:
            dbg["raw_labels"] = np.zeros(0, dtype=int)
            dbg["raw_points"] = np.zeros((0, 2), np.float32)

        labels, strengths, pidx = self._cluster_with_refinements(points, rng)

        # Debug: capture labels after refinement
        if debug:
            dbg["refined_labels"] = labels.copy()
            dbg["refined_points"] = points[pidx].copy() if pidx.size else np.zeros((0, 2), np.float32)
            dbg["pidx"] = pidx.copy()

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

        # -- Step 2.5: merge vertically stacked clusters in same column -------
        labels, pidx = _merge_stacked_clusters(
            points[pidx] if pidx.size else points, labels, pidx, cfg,
            float(mask_t.shape[0]), float(mask_t.shape[1]))

        if debug:
            dbg["merged_labels"] = labels.copy()
            dbg["merged_points"] = points[pidx].copy() if pidx.size else np.zeros((0, 2), np.float32)

        # Debug: capture geometry filter before-state (all clusters passing
        # min_points + min_height, before the gap-based weed deletion)
        if debug:
            view_h_dbg = float(mask_t.shape[0])
            min_height_dbg = cfg.min_height_frac * view_h_dbg
            geo_before = {}
            rejection_log = []
            for c in range(int(labels.max()) + 1) if labels.size else range(0):
                sel = np.nonzero(labels == c)[0]
                if len(sel) < cfg.min_cluster_points:
                    rejection_log.append((c, f"too few pts ({len(sel)}<{cfg.min_cluster_points})"))
                    continue
                if points[sel][:, 1].ptp() < min_height_dbg:
                    rejection_log.append((c, f"too short ({points[sel][:,1].ptp():.0f}<{min_height_dbg:.0f}px)"))
                    continue
                geo_before[c] = sel
            dbg["geo_before"] = geo_before
            dbg["rejection_log"] = rejection_log

        clusters = self._geometry_filter(points[pidx], labels, pidx, rng,
                                         float(mask_t.shape[0]))

        # Debug: capture geometry filter after-state + gap deletions
        if debug:
            dbg["geo_after"] = clusters
            # record which clusters were deleted by gap check
            gap_deleted = set(geo_before.keys()) - set(clusters.keys())
            for c in gap_deleted:
                dbg["rejection_log"].append((c, "gap too narrow (weed neighbor)"))

        T["cluster"] = (time.perf_counter() - t0) * 1000

        # -- Step 2.6: select the 2 center clusters (Zhou flanking style) ----
        selected_clusters, forfeited_clusters = self._select_center_2(
            clusters, points, mask_t)

        # -- Step 3: Spline line fitting + intersection voting ---------------
        t0 = time.perf_counter()
        lines_view, lines_roi, lines_orig, kept_clusters = [], [], [], []
        ransac_all = []   # debug: all fits
        ransac_reject = []  # debug: rejected lines
        rejected_spline_curves = []  # for red vis: voted-out lines
        rejected_line_segs = []      # for red vis: voted-out extended lines
        view_w = float(mask_t.shape[1])
        view_h = float(mask_t.shape[0])
        horizon_center = np.array([view_w / 2.0, 0.0], dtype=np.float32)

        # Phase 1: fit a spline to the 2 center clusters only
        raw_fits = []  # (cid, p1, p2, inlier_ratio)
        raw_spline_curves = {}  # cid → sorted cluster points (for vis)
        for cid, idx in selected_clusters.items():
            fit = spline_fit(points[idx], cfg)
            if fit is None:
                if debug:
                    ransac_reject.append((cid, "spline fit failed (<4 pts)"))
                continue
            p1, p2, inlier_ratio = fit
            p1e, p2e = extend_line_full_height(p1, p2, view_w, view_h)
            raw_fits.append((cid, p1e, p2e, inlier_ratio))
            cl_pts = points[idx]
            if len(cl_pts) >= 2:
                order = np.argsort(cl_pts[:, 1])
                raw_spline_curves[cid] = cl_pts[order].astype(np.int32)

        # Phase 2: assemble — keep ALL fitted splines (no voting/rejection)
        spline_curves = []
        for i in range(len(raw_fits)):
            cid, p1, p2, inlier_ratio = raw_fits[i]
            ang = line_angle_deg(p1, p2)
            curve = raw_spline_curves.get(cid)
            p1_safe = tuple(int(float(x)) for x in p1)
            p2_safe = tuple(int(float(x)) for x in p2)
            lines_view.append((p1_safe, p2_safe, inlier_ratio))
            seg_r = tuple(self._to_roi(pt, Hinv) for pt in (p1, p2))
            lines_roi.append(seg_r)
            seg_o = tuple(self._to_original(pt, Hinv, y0, scale)
                          for pt in (p1, p2))
            lines_orig.append(seg_o)
            kept_clusters.append(cid)
            if debug:
                ransac_all.append((p1_safe, p2_safe, cid, inlier_ratio, ang, True))
            if curve is not None:
                spline_curves.append(curve)
        T["fit_lines"] = (time.perf_counter() - t0) * 1000

        # Full-image binary mask for composite display (panel 03)
        mask_full = segment_green(bgr, cfg)

        try:
            clusters_img = self._render_clusters(mask_t, points[pidx], labels,
                                                  kept_clusters, cell,
                                                  forfeited_ids=set(
                                                      forfeited_clusters.keys()))
        except Exception:
            clusters_img = cv2.cvtColor(mask_t, cv2.COLOR_GRAY2BGR)
        try:
            det_view = self._draw(view, lines_view, spline_curves,
                                  rejected_lines=rejected_line_segs,
                                  rejected_splines=rejected_spline_curves)
        except Exception:
            det_view = view.copy()

        intermediates = {
            "roi": roi.copy(),
            "mask": mask.copy(),
            "mask_full": mask_full.copy(),
            "view": view.copy(),
            "clusters": clusters_img,
            "detection_view": det_view,
            "spline_curves": spline_curves,
            "rejected_spline_curves": rejected_spline_curves,
            "rejected_line_segs": rejected_line_segs,
            "forfeited_clusters": forfeited_clusters,
            "vp": vp,
        }
        if debug:
            intermediates.update(dbg)
            intermediates["ransac_all"] = ransac_all
            intermediates["ransac_reject"] = ransac_reject
            intermediates["kept_clusters"] = kept_clusters
            intermediates["lines_view"] = lines_view
            intermediates["mask_t"] = mask_t.copy()
            intermediates["points"] = points[pidx].copy() if pidx.size else np.zeros((0, 2), np.float32)
            intermediates["labels"] = labels.copy() if labels.size else np.zeros(0, dtype=int)
            intermediates["cell"] = cell
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
        model = run_hdbscan(points, cfg, mcs,
                            vertical_stretch=cfg.vertical_stretch)
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
            model = run_hdbscan(points, cfg, mcs,
                                vertical_stretch=cfg.vertical_stretch)
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

    def _select_center_2(self, clusters, points, mask_t):
        """Select the 2 clusters closest to the horizontal center.

        Follows the Zhou et al. (2021) flanking-cluster style: use column
        projection on the top-view mask to find the 2 row-candidate bands
        closest to the image centre, then match HDBSCAN clusters to those
        bands.  Outer clusters are forfeited — in imperfect BEV views they
        frequently merge into one unreliable blob.

        Returns (selected_clusters, forfeited_clusters).
        """
        if len(clusters) <= 2:
            return clusters, {}

        view_h, view_w = mask_t.shape[:2]
        center_x = view_w / 2.0

        # --- column projection (Zhou Eq.5-6) ---
        z = (mask_t == 255).sum(axis=0).astype(np.float64)
        t = z.mean() + z.std()                   # T = M + E (Eq.7-8)
        z_filt = z.copy()
        z_filt[z_filt < t] = 0.0

        # contiguous feature-column bands (Sec.2.3.2, gap < L)
        cols = np.nonzero(z_filt > 0)[0]
        l_thresh = max(2, int(0.05 * view_w))
        bands = []
        if len(cols) > 0:
            start = prev = cols[0]
            for c in cols[1:]:
                if c - prev < l_thresh:
                    prev = c
                else:
                    bands.append((start, prev))
                    start = prev = c
            bands.append((start, prev))

        # --- find the 2 target bands (Zhou flanking_clusters) ---
        target_bands = None
        if len(bands) >= 2:
            band_centers = [(a + b) / 2.0 for a, b in bands]
            left_bands = [(i, bc) for i, bc in enumerate(band_centers)
                          if bc < center_x]
            right_bands = [(i, bc) for i, bc in enumerate(band_centers)
                           if bc >= center_x]
            if left_bands and right_bands:
                c_left_i = max(left_bands, key=lambda x: x[1])[0]
                c_right_i = min(right_bands, key=lambda x: x[1])[0]
                target_bands = [bands[c_left_i], bands[c_right_i]]
            else:
                # all on one side — pick 2 closest to center
                si = sorted(range(len(bands)),
                            key=lambda i: abs(band_centers[i] - center_x))
                target_bands = [bands[si[0]], bands[si[1]]]

        # --- match HDBSCAN clusters to target bands ---
        if target_bands is not None:
            selected, forfeited = {}, {}
            for cid, idx in clusters.items():
                cx = float(points[idx][:, 0].mean())
                matched = any(lo - 30 <= cx <= hi + 30
                             for lo, hi in target_bands)
                (selected if matched else forfeited)[cid] = idx
            if len(selected) >= 1:
                # if we got 1 by projection match but there are >2 total,
                # fill the other slot with the next-closest-to-center
                if len(selected) == 1 and len(clusters) > 1:
                    remaining = {c: idx for c, idx in clusters.items()
                                 if c not in selected}
                    closest = min(remaining,
                                  key=lambda c: abs(
                                      float(points[remaining[c]][:, 0].mean())
                                      - center_x))
                    selected[closest] = remaining[closest]
                    forfeited.pop(closest, None)
                if len(selected) == 2:
                    return selected, forfeited

        # --- fallback: simply 2 closest to horizontal center ---
        info = []
        for cid, idx in clusters.items():
            cx = float(points[idx][:, 0].mean())
            info.append((cid, idx, abs(cx - center_x)))
        info.sort(key=lambda x: x[2])
        selected = {c[0]: c[1] for c in info[:2]}
        forfeited = {c[0]: c[1] for c in info[2:]}
        return selected, forfeited

    @staticmethod
    def _render_clusters(mask_t, points, labels, kept_ids, cell: int = 2,
                         forfeited_ids=None):
        vis = cv2.cvtColor(mask_t, cv2.COLOR_GRAY2BGR)
        if labels.size == 0:
            return vis
        kept = set(kept_ids)
        forfeited = set(forfeited_ids) if forfeited_ids else set()
        rng = np.random.default_rng(0)
        half = max(1, cell // 2)
        for c in np.unique(labels):
            sel = points[labels == c].astype(int)
            if c < 0:                       # outliers rendered black
                col = (0, 0, 0)
            elif c in kept:
                col = (int(rng.integers(60, 255)), int(rng.integers(60, 255)),
                       int(rng.integers(60, 255)))
            elif c in forfeited:
                col = (0, 165, 255)  # orange — forfeited by center-2 selector
            else:
                col = (128, 128, 128)
            for x, y in sel:
                x0, y0 = max(0, x - half), max(0, y - half)
                vis[y0:y0 + cell, x0:x0 + cell] = col
        return vis

    @staticmethod
    def _draw(img, lines, spline_points=None, rejected_lines=None,
              rejected_splines=None):
        """Draw detection lines on image.
        - Green: surviving spline curves + extended lines
        - Red: voted-out / rejected lines (shown for debugging)
        """
        out = img.copy()
        # rejected lines in red first (so green draws on top)
        if rejected_splines is not None:
            for pts in rejected_splines:
                if len(pts) >= 2:
                    cv2.polylines(out, [pts.astype(np.int32)], False,
                                  (0, 0, 220), 1, cv2.LINE_AA)
        if rejected_lines is not None:
            for item in rejected_lines:
                try:
                    p1, p2 = item[0], item[1]
                    pt1 = (int(float(p1[0])), int(float(p1[1])))
                    pt2 = (int(float(p2[0])), int(float(p2[1])))
                    cv2.line(out, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
        # surviving lines in green
        if spline_points is not None and len(spline_points) > 0:
            for pts in spline_points:
                if len(pts) >= 2:
                    cv2.polylines(out, [pts.astype(np.int32)], False,
                                  (0, 220, 0), 2, cv2.LINE_AA)
        for item in lines:
            try:
                p1, p2 = item[0], item[1]
                pt1 = (int(float(p1[0])), int(float(p1[1])))
                pt2 = (int(float(p2[0])), int(float(p2[1])))
                cv2.line(out, pt1, pt2, (0, 220, 0), 2, cv2.LINE_AA)
            except Exception:
                pass
        return out
