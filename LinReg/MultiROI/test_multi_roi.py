"""
Autonomous detection of crop rows based on adaptive multi-ROI in maize fields
=============================================================================
Re-implementation of:
    Zhou Y, Yang Y, Zhang B L, Wen X, Yue X, Chen L Q.
    "Autonomous detection of crop rows based on adaptive multi-ROI in maize
    fields." Int J Agric & Biol Eng, 2021; 14(4): 217-225.
    DOI: 10.25165/j.ijabe.20211404.6315

Pipeline (paper sections):
  2.2.1  Normalized improved ExG (Eq. 1-2):  ExG = 1.8g - 1.1r - 0.8b
         + Otsu binarization
  2.2.2  Morphological open, kernel K = 4x4 anti-diagonal (Eq. 3)
  2.3.1  Divide image into N=10 horizontal strips (Eq. 4)
  2.3.2  Initial ROI: vertical column projection (Eq. 5), noise filter
         T = M + E (Eq. 6-8), clustering of feature columns (gap < L),
         two clusters flanking the midpoint -> CLeft/CRight,
         D = max(CRight) - min(CLeft), W_ROI = 1.2*D (Eq. 9),
         renewed midpoint (Eq. 10) stored in point set Q
  2.3.3  Multi-ROI: shift window up strip by strip, repeat projection +
         clustering inside the current ROI; fall back to previous ROI window
         when no feature points are found
  2.4    Navigation line: least squares y = w*x + b over Q (Eq. 11-14),
          initial estimated midpoint excluded
  3      Detection lines: least squares per crop row across the multi-ROI

EXPERIMENTAL (this copy only): struct-clean morphology is the DEFAULT
  Segmentation runs Eq. (3) and then a band-support weed filter BEFORE
  any corridor is found: per-strip column projections (Eq. 5-8, with a
  P90 fallback for saturated canopies) define row-candidate bands; every
  connected component overlapping a band survives, isolated off-structure
  blobs are erased. Wrong-lane capture therefore cannot happen - decoys
  are gone at strip 1 (validated by eye on photo_9/photo_6).

  The two-pass anchor scrub (--scorer if/madz) is OPT-IN via --post_scrub:
  it was observed to delete real crop rows (photo_11/17) and only ever
  added value where weeds are band-aligned yet globally off-corridor
  (bev6). Other experiments (robust oriented openings, single-pass inline
  cleaning) remain available behind flags as documented negative results.

Usage:
    python test_multi_roi.py --input ../../Photos --results_dir ./result_test
"""

import argparse
import csv
import glob
import math
import os
import time
import warnings

import cv2
import numpy as np
from sklearn.ensemble import IsolationForest

# Morphological kernel K, Eq. (3): 4x4 anti-diagonal of ones
KERNEL_K = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=np.uint8)

# Colors BGR
COLOR_NAV_LINE = (255, 200, 0)     # blue - navigation line (Fig. 12)
COLOR_DET_LINE = (150, 50, 0)      # dark blue - detection lines (Fig. 12)
COLOR_ROI      = (255, 255, 255)   # white - ROI rectangles (Fig. 9/11)


def preprocess(bgr, border_frac=0.02, index="raw", morph="paper",
               open_len=9, fence_k=1.5, morph_fence="if", n_strips=10,
               l_frac=0.05):
    """Section 2.2: edge crop + ExG + Otsu + morphology.

    index:
      "raw"        classic Woebbecke ExG = 2G - R - B on the raw 0..255
                   channels (ref [30] of the paper). Robust on our dataset;
                   used as the default.
      "normalized" the paper's Eq. (1)-(2): channel-normalized improved
                   ExG = 1.8g - 1.1r - 0.8b, mapped from [0, 1.8] to 0..255.
                   Illumination invariant, but degenerates on images where
                   crop and background are both strongly green (e.g. aerial
                   views): normalization erases the brightness contrast and
                   Otsu then fails to separate crop from soil.

    Returns (binary mask with crop=255, ExG grayscale, Otsu threshold,
    (dx, dy) crop offset).
    """
    h, w = bgr.shape[:2]
    dx, dy = int(w * border_frac), int(h * border_frac)
    img = bgr[dy:h - dy, dx:w - dx] if dx > 0 and dy > 0 else bgr

    b, g, r = cv2.split(img.astype(np.float32))
    if index == "normalized":
        s = r + g + b
        s[s == 0] = 1.0  # guard Eq. (1) division for black pixels
        bn, gn, rn = b / s, g / s, r / s      # Eq. (1): normalized channels
        exg = 1.8 * gn - 1.1 * rn - 0.8 * bn  # Eq. (2): improved ExG
        exg8 = (np.clip(exg, 0.0, 1.8) / 1.8 * 255.0).astype(np.uint8)
    else:
        exg = 2.0 * g - r - b
        exg8 = np.clip(exg, 0.0, 255.0).astype(np.uint8)

    otsu_t, binary = cv2.threshold(exg8, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if morph == "robust":
        binary = robust_morphology(binary, open_len,
                                   fence=morph_fence,
                                   fence_k=fence_k)   # EXPERIMENTAL
    elif morph == "struct":
        # EXPERIMENTAL: paper opening + pre_clean-style band-support
        # filtering embedded directly in segmentation
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_K)
        binary, _ = pre_corridor_clean(binary, n_strips, l_frac)
    else:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_K)  # Eq. (3)
    return binary, exg8, float(otsu_t), (dx, dy)


def column_projection(strip, x_lo, x_hi):
    """Eq. (5)-(6): vertical projection of a strip within [x_lo, x_hi).

    Z(i) = number of 255-valued pixels in column i (a = 1), then filtered
    to 0 wherever Z(i) < T with T = M + E (Eq. 7-8).

    Dense-canopy guard: on overhead/full-canopy scenes the column counts
    saturate near the strip height, so M + E can exceed max(Z) and wipe
    out the entire profile (photo_7-class frames). In that case - and
    only then - the threshold falls back to the 90th percentile so the
    strongest columns (the visible rows) survive.
    """
    window = strip[:, x_lo:x_hi]
    z = (window == 255).sum(axis=0).astype(np.float64)  # Eq. (5)
    t = z.mean() + z.std()                              # Eq. (7)-(8)
    if t > 0 and not np.any(z >= t):
        t = float(np.quantile(z, 0.9))
    z[z < t] = 0.0                                      # Eq. (6)
    return z


def cluster_feature_columns(z, x_lo, l_thresh):
    """Section 2.3.2: group feature columns (Z != 0) whose gap < L.

    Returns clusters as (c_min, c_max) in absolute image coordinates.
    """
    cols = np.nonzero(z > 0)[0]
    if cols.size == 0:
        return []
    clusters = []
    start = prev = cols[0]
    for c in cols[1:]:
        if c - prev < l_thresh:  # neighboring points closer than L group up
            prev = c
        else:
            clusters.append((start + x_lo, prev + x_lo))
            start = prev = c
    clusters.append((start + x_lo, prev + x_lo))
    return clusters


def flanking_clusters(clusters, mo_x):
    """The two clusters nearest to MO along the column axis -> (CLeft, CRight)."""
    left = [c for c in clusters if (c[0] + c[1]) / 2.0 < mo_x]
    right = [c for c in clusters if (c[0] + c[1]) / 2.0 >= mo_x]
    c_left = max(left, key=lambda c: c[1]) if left else None
    c_right = min(right, key=lambda c: c[0]) if right else None
    return c_left, c_right


class MultiROIDetector:
    """Adaptive multi-ROI crop row detector (Zhou et al. 2021)."""

    def __init__(self, n_strips=10, l_frac=0.05, border_frac=0.02,
                 min_points_per_row=2, index="raw", min_flank_frac=0.01,
                 nav_include_first=False, morph="paper", open_len=9,
                 fence_k=1.5, morph_fence="if", nav_curve=True):
        self.n = n_strips
        self.l_frac = l_frac             # clustering distance L as fraction of W
        self.border_frac = border_frac
        self.min_points_per_row = min_points_per_row
        self.index = index               # ExG variant: "raw" or "normalized"
        # a cluster must hold at least this share of the strip's surviving
        # projection mass to be eligible as a flanking row; otherwise thin
        # weed spikes can become the corridor border and freeze the ROI
        # around the wrong lane (cf. photo_9's 1-px sliver at strip 1)
        self.min_flank_frac = min_flank_frac
        # same rationale as the detection-chain deviation: the full-width
        # initial strip can anchor on rows outside the travelling area
        # (photo_6: its only qualifying right cluster was the THIRD row),
        # and its midpoint sits at maximal leverage over the nav-line
        # slope - so it is excluded from the fit unless asked otherwise
        self.nav_include_first = nav_include_first
        # segmentation morphology: "paper" = Eq. (3) single anti-diagonal
        # opening; "robust" = close + union of oriented line openings +
        # area floor (see robust_morphology)
        self.morph = morph
        self.open_len = open_len
        self.fence_k = fence_k
        self.morph_fence = morph_fence
        # EXPERIMENTAL: smoothed-spline navigation curve through the
        # accepted Q dots (with safety envelope vs the straight TLS line);
        # falls back to the straight line when evidence is thin.
        self.nav_curve = nav_curve

    def detect(self, bgr):
        """Run the full pipeline on one BGR image; returns results dict."""
        t0 = time.perf_counter()
        binary, exg8, otsu_t, (dx, dy) = preprocess(bgr, self.border_frac,
                                                    self.index, self.morph,
                                                    self.open_len,
                                                    self.fence_k,
                                                    self.morph_fence,
                                                    self.n, self.l_frac)
        core = self._run_core(binary)
        out = dict(core)
        out.update({
            "binary": binary,
            "exg": exg8,
            "otsu_t": otsu_t,
            "crop_offset": (dx, dy),
        })
        out["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    def detect_from_binary(self, binary, crop_offset):
        """Re-run the strip climb + fits on a pre-segmented (cleaned) mask.

        Used as pass 2 after Isolation-Forest scrubbing. The returned dict
        has the same keys as detect(), except exg/otsu_t are None.
        """
        t0 = time.perf_counter()
        core = self._run_core(binary)
        out = dict(core)
        out.update({
            "binary": binary,
            "exg": None,
            "otsu_t": None,
            "crop_offset": crop_offset,
            "time_ms": (time.perf_counter() - t0) * 1000.0,
        })
        return out

    def _run_core(self, binary):
        """Sections 2.3-2.4 + 3 on an already-segmented binary mask."""
        h, w = binary.shape[:2]

        dh = h // self.n                     # Eq. (4): strip height dh = H/N
        l_thresh = max(2, int(self.l_frac * w))
        rois = []            # per-strip (x_lo, x_hi, y_top, y_bot), bottom->top
        q = []               # point set Q of renewed midpoints (Eq. 11)
        q_feed = []          # (mo_x, my, flank span or None) evidence tags
        left_feed = []       # (y, CLeft center, D) candidates, two-sided only
        right_feed = []      # same for the right chain

        mo_x = w / 2.0                       # Sec. 2.3.2: initial midpoint MO
        x_lo, x_hi = 0, w                    # initial ROI = full strip width

        # Weed-pressure accumulators. Two complementary signals, combined
        # by max():
        #   occupancy - fraction of filtered projection mass INSIDE the
        #               furrow (between the flanking clusters): scattered
        #               in-furrow weeds forming fake columns.
        #   flank_fail - fraction of strips (2..N) failing to find BOTH
        #               flanking clusters: weed so dense it chokes the
        #               T=M+E filter or bridges the rows until the
        #               corridor collapses (cf. photo_9-style scenes).
        # Free to compute - reuses per-strip projections already made.
        furrow_px = furrow_tot = 0.0
        flank_ok = flank_tried = 0
        feed_d = []          # (y, flank span) of two-sided picks: chain filter

        for mu in range(1, self.n + 1):      # strip 1 = bottom strip
            y1 = h - mu * dh                 # top row of strip mu
            d_strip = None                   # flank span evidence this strip

            strip = binary[max(0, y1):y1 + dh, :]

            z = column_projection(strip, x_lo, x_hi)
            clusters = cluster_feature_columns(z, x_lo, l_thresh)

            # mass gate: drop clusters too weak to be a real row segment
            if clusters and self.min_flank_frac > 0.0:
                tot = float(z.sum())
                strong = []
                for a, b in clusters:
                    lo_i = max(0, int(a - x_lo))
                    hi_i = min(len(z), int(b - x_lo) + 1)
                    if hi_i > lo_i and \
                            float(z[lo_i:hi_i].sum()) >= self.min_flank_frac * tot:
                        strong.append((a, b))
                clusters = strong

            if clusters:
                c_left, c_right = flanking_clusters(clusters, mo_x)
                # Window + corridor renewal ONLY on a two-sided pick.
                # A lone cluster is insufficient evidence: sliding mo_x
                # (or the ROI) onto it collapses the navigation line onto
                # that single row on sparse/young-crop fields where the
                # two rows alternate visibility strip by strip
                # (photo_11/photo_17 failure mode). Insufficient evidence
                # reuses the previous window and Mx - the same fall-back
                # spirit as Sec. 2.3.3's empty-strip rule. The chain feed
                # below still records the pick for the detection lines.
                if c_left is not None and c_right is not None:
                    # Eq. (9): D = max(CRight) - min(CLeft)
                    d = c_right[1] - c_left[0]
                    d_strip = d
                    if d <= 0:
                        d = max(4, x_hi - x_lo)
                    # Fig. 8b: ROI = [min(CLeft) - 0.1D, max(CRight) + 0.1D]
                    new_lo = int(max(0, c_left[0] - 0.1 * d))
                    new_hi = int(min(w, c_right[1] + 0.1 * d))
                    if new_hi - new_lo >= 4:     # keep sane windows only
                        x_lo, x_hi = new_lo, new_hi
                    # Eq. (10): renewed Mx = midpoint between the two
                    # flanking rows (cluster centers, not outer edges, so
                    # merged clusters cannot yank the corridor sideways)
                    mo_x = ((c_left[0] + c_left[1]) / 2.0
                            + (c_right[0] + c_right[1]) / 2.0) / 2.0
                # weed pressure: occupancy of the furrow cross-section.
                # Bare furrow -> clear gap between CLeft/CRight -> low;
                # scattered weeds inside the gap -> their projection mass
                # counts; weeds so dense they BRIDGE the two clusters ->
                # gap gone -> strip counts as fully occupied (worst case,
                # cf. photo_9-style scenes).
                if c_left is not None and c_right is not None:
                    tot = float(z.sum())
                    flank_ok += 1
                    if c_right[0] > c_left[1]:
                        lo_i = int(c_left[1] - x_lo) + 1
                        hi_i = int(c_right[0] - x_lo)
                        if tot > 0.0 and hi_i > lo_i:
                            furrow_px += float(z[lo_i:hi_i].sum())
                            furrow_tot += tot
                    elif tot > 0.0:
                        furrow_px += tot          # bridged: 100% occupied
                        furrow_tot += tot
                if mu > 1:
                    flank_tried += 1
            else:
                if mu > 1:
                    flank_tried += 1
            # else / one-sided: previous ROI window & Mx reused

            my = h - mu * dh                 # Eq. (10): My(mu) = My(mu-1) - dh
            rois.append((x_lo, x_hi, max(0, y1), y1 + dh))
            q.append((mo_x, my))             # renewed MO into Q
            q_feed.append((mo_x, my, d_strip))  # with its evidence tag

            # Per-strip flanking cluster centers, used for the detection
            # lines (Section 3): the CLeft chain and CRight chain across the
            # strips are the crop rows bordering the travelling area. Strip 1
            # is skipped: with the full-width initial view its flanking pick
            # can anchor on rows outside the travelling area. When only one
            # cluster is found it continues the side its center falls on
            # (rows converge but do not swap sides).
            # Chain feed (Section 3): only two-sided picks contribute.
            # One-sided centers come from ambiguous strips (wide inherited
            # windows, merged bands) and their leverage at the image base
            # tilted entire lines (photo_17 failure mode). Missing
            # evidence is preferred over wrong evidence.
            cy = y1 + dh / 2.0
            if clusters and mu > 1 and \
                    c_left is not None and c_right is not None:
                d_here = c_right[1] - c_left[0]
                left_feed.append((cy, (c_left[0] + c_left[1]) / 2.0, d_here))
                right_feed.append((cy, (c_right[0] + c_right[1]) / 2.0,
                                   d_here))
                feed_d.append((cy, d_here))

        # Section 2.4: navigation line - EVIDENCE-GATED Q, same standard
        # as the detection chains. A midpoint is a measurement only if its
        # strip produced a two-sided pick with plausible flank span
        # (D <= 1.6 x median). Synthetic dots (one-sided/empty strips under
        # option B) and merged-band picks (wide D) are excluded - it would
        # be inconsistent to bar those boxes from the chains yet let their
        # dots vote on navigation. Strip 1 stays excluded as well (its
        # full-width view predates any corridor knowledge). Fallback: if
        # no measured dot survives, fit the priors rather than return no
        # line (all-red visualization makes that explicit).
        nav_pts, q_rejected = [], []
        measured = [(x, y, d) for x, y, d in q_feed if d is not None]
        if measured:
            med_d = float(np.median([d for _, _, d in measured]))
            max_d = 1.6 * med_d
        else:
            max_d = float("inf")
        for i, (x, y, d) in enumerate(q_feed):
            worthy = (d is not None and d <= max_d
                      and (i > 0 or self.nav_include_first))
            if worthy:
                nav_pts.append((x, y))
            else:
                q_rejected.append((x, y))
        fallback = len(nav_pts) < 2
        if fallback:
            nav_pts = list(q)
            q_rejected = []

        q_accepted = []
        if len(nav_pts) >= 2:
            nav_line, acc_rel, _ = self._fit_line_tls(nav_pts)
            q_accepted = [nav_pts[i] for i in acc_rel]
            q_rejected += [nav_pts[i] for i in range(len(nav_pts))
                           if i not in acc_rel]

        # Navigation CURVE (EXPERIMENTAL): smoothed cubic regression spline
        # through the accepted dots - follows genuinely curving corridors
        # instead of cutting across them. Safety envelope: the curve may
        # never deviate more than 15% of the median corridor span from the
        # straight TLS reference, so it cannot whip regardless of what the
        # dots suggest. Falls back to the straight line when evidence is
        # thin (<4 dots) or the spline fails to construct.
        nav_curve = None
        nav_base_angle = float("nan")
        if self.nav_curve and len(q_accepted) >= 4:
            env_cap = 0.15 * (med_d if measured else 0.10 * w)
            curve = self._fit_nav_curve(q_accepted, h, env_cap)
            if curve is not None:
                nav_curve, nav_base_angle = curve

        if nav_line is not None and not math.isfinite(nav_base_angle):
            nav_base_angle = math.degrees(math.atan(abs(1.0 / nav_line[0])))

        # Section 3: detection lines, one least-squares line per row chain.
        # Corridor-width consistency filter first: a two-sided pick whose
        # flank span D dwarfs the scene's median span came from merged
        # multi-row bands (photo_17 s3: D=576 vs median 180) and its
        # centers describe OTHER rows - feeding them tilts the whole line
        # even through the robust fitter, because a single far outlier in
        # an otherwise near-vertical chain is ill-conditioned in OLS.
        left_pts, right_pts = [], []
        if feed_d:
            max_d = 1.6 * float(np.median([d for _, d in feed_d]))
            for cy, cx, d_here in left_feed:
                if d_here <= max_d:
                    left_pts.append((cx, cy))
            for cy, cx, d_here in right_feed:
                if d_here <= max_d:
                    right_pts.append((cx, cy))
        # Same total-least-squares fitter as the nav line: detection
        # chains are frequently near-vertical (converged rows, sparse
        # scenes), where y-on-x regression is ill-conditioned.
        det_lines = []
        for pts in (left_pts, right_pts):
            if len(pts) < self.min_points_per_row:
                continue
            fit_tls, _, _ = self._fit_line_tls(pts)
            if fit_tls is not None:
                det_lines.append((fit_tls[0], fit_tls[1], len(pts)))

        occupancy = (furrow_px / furrow_tot) if furrow_tot > 0.0 else 0.0
        fail_rate = (1.0 - flank_ok / flank_tried) if flank_tried > 0 else 0.0

        # Curvature proxy: angle between the nav-line slopes fitted to the
        # lower and upper half of Q. High bend = geometrically curved row,
        # NOT weeds - such corridors must not be scrubbed against their
        # own straight anchors (the straight-line model is the limit
        # there, cf. photo_2-class fields). 999 marks "unmeasurable".
        bend = 999.0
        try:
            h_mid = h / 2.0
            xs = np.array([p[0] for p in q], dtype=np.float64)
            ys = np.array([p[1] for p in q], dtype=np.float64)
            lo_m = ys >= h_mid
            hi_m = ~lo_m
            if lo_m.sum() >= 2 and hi_m.sum() >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    k_lo = float(np.polyfit(xs[lo_m], ys[lo_m], 1)[0])
                    k_hi = float(np.polyfit(xs[hi_m], ys[hi_m], 1)[0])
                if abs(k_lo) > 1e-9 and abs(k_hi) > 1e-9:
                    a_lo = math.atan(1.0 / k_lo)
                    a_hi = math.atan(1.0 / k_hi)
                    bend = math.degrees(abs(a_lo - a_hi))
        except Exception:
            pass

        return {
            "rois": rois,
            "q": q,
            "q_accepted": q_accepted,
            "q_rejected": q_rejected,
            "nav_line": nav_line,    # (w, b) straight TLS reference
            "nav_curve": nav_curve,  # sampled [(x, y)] polyline or None
            "nav_base_angle_deg": nav_base_angle,
            "det_lines": det_lines,  # list of (w, b, n_pts)
            "weed_pressure": max(occupancy, fail_rate),
        }

    def _fit_nav_curve(self, pts, img_h, env_cap):
        """Smoothed cubic regression spline through evidence-gated dots.

        Returns (samples, base_tangent_deg) or None. Interior knots scale
        with point count (0-3), so sparse scenes fit nearly straight
        lines; the safety envelope caps deviation from the straight TLS
        reference at env_cap px; outside the data range the curve extends
        along its end tangents so the drawn path spans the full image.
        """
        try:
            from scipy.interpolate import LSQUnivariateSpline
        except ImportError:
            return None
        P = sorted(pts, key=lambda p: p[1])
        vy, vx = [], []
        for x, y in P:                       # average duplicate rows
            if vy and abs(y - vy[-1]) < 1e-6:
                vx[-1] = (vx[-1] + x) / 2.0
                continue
            vy.append(y)
            vx.append(x)
        if len(vy) < 4:
            return None
        ys = np.array(vy, dtype=np.float64)   # image rows (increasing)
        xs = np.array(vx, dtype=np.float64)   # lateral positions

        def spline_at(yy):
            n_int = max(0, min(3, (len(ys) - 3) // 2))
            t = np.linspace(ys[0], ys[-1], n_int + 2)[1:-1] if n_int else \
                np.array([])
            sp = LSQUnivariateSpline(ys, xs, t, k=3)
            return sp(yy)

        # robust refit: drop worst residuals vs the curve (max 2 rounds)
        keep = list(range(len(ys)))
        for _ in range(2):
            if len(keep) <= 4:
                break
            vals = np.array([float(spline_at(ys[i])) for i in keep])
            res = np.abs(np.array([xs[i] for i in keep]) - vals)
            worst = int(np.argmax(res))
            if res[worst] <= 2.5 * max(float(res.std()), 1e-6):
                break
            keep.pop(worst)

        yy = np.arange(0, img_h, 4, dtype=np.float64)
        sx = spline_at(np.clip(yy, ys[0], ys[-1]))
        # linear extension beyond the data range along end tangents
        step = max(1.0, (ys[-1] - ys[0]) / 20.0)
        m_lo = (float(spline_at(ys[0] + step)) - float(spline_at(ys[0]))) / step
        m_hi = (float(spline_at(ys[-1])) - float(spline_at(ys[-1] - step))) / step
        # boundary splines can wiggle; cap extension slope for drawing
        m_lo = float(np.clip(m_lo, -6.0, 6.0))
        m_hi = float(np.clip(m_hi, -6.0, 6.0))
        for i, yv in enumerate(yy):
            if yv < ys[0]:
                sx[i] += m_lo * (yv - ys[0])
            elif yv > ys[-1]:
                sx[i] += m_hi * (yv - ys[-1])

        # safety envelope vs the straight TLS reference line.
        # Evaluate the reference geometrically as x(y) = (y - b)/w: this
        # is numerically stable even for the near-vertical w=1e6
        # representation, where naive w*y + b overflows to ~1e8 and would
        # snap the clamped curve onto an exploding line.
        ref = self._fit_line_tls([(x, y) for x, y in zip(xs[keep], ys[keep])])
        w_r, b_r = ref[0]
        ref_x = (yy - b_r) / w_r
        dev = np.abs(sx - ref_x)
        if env_cap > 0 and float(dev.max()) > env_cap:
            f = env_cap / float(dev.max())
            sx = ref_x + (sx - ref_x) * f

        samples = [(float(x), float(y)) for x, y in zip(sx, yy)]
        # steering-relevant tangent: derivative at the LAST MEASURED row
        # (deepest evidence), not in the extrapolated tail
        dxdy = (float(spline_at(ys[-1] + 1e-3))
                - float(spline_at(ys[-1] - 1e-3))) / 2e-3
        dxdy = float(np.clip(dxdy, -6.0, 6.0))
        base_ang = math.degrees(math.atan(abs(dxdy)))
        return samples, base_ang

    @staticmethod
    def _fit_line(points):
        """Least squares y = w*x + b (Eq. 12-14). None if degenerate."""
        pts = [(float(x), float(y)) for x, y in points
               if np.isfinite(x) and np.isfinite(y)]
        if len(pts) < 2:
            return None
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if float(xs.max() - xs.min()) < 1e-6:
            # Perfectly vertical point set (straight corridor): y = w*x + b
            # is undefined; represent x = mx as an extremely steep line
            mx = float(xs.mean())
            return 1e6, -1e6 * mx
        try:
            w_slope, b = np.polyfit(xs, ys, 1)
        except np.RankWarning:
            return None
        return float(w_slope), float(b)

    @staticmethod
    def _fit_line_tls(points, max_reject=3, n_sigma=2.5):
        """Total-least-squares line through points, perpendicular residuals.

        Returns (w, b, accepted_indices). The line minimizes the sum of
        squared PERPENDICULAR distances - the intuitive 'line of best
        fit' - and remains stable for perfectly vertical corridors where
        y-on-x regression degenerates. Up to max_reject points are then
        dropped at n_sigma of the perpendicular residual spread.
        """
        pts = [(float(x), float(y)) for x, y in points
               if np.isfinite(x) and np.isfinite(y)]
        if len(pts) < 2:
            return None, [], []
        P = np.array(pts)
        acc = list(range(len(pts)))
        c = d = None
        rejects = 0
        while True:
            Q = P[acc]
            c = Q.mean(axis=0)
            _, _, vt = np.linalg.svd(Q - c)
            d = vt[0]                        # principal direction
            n_vec = np.array([-d[1], d[0]])  # unit normal
            resid = np.abs((P - c) @ n_vec)
            r_acc = resid[acc]
            if len(acc) <= 4 or rejects >= max_reject:
                break
            worst = int(np.argmax(r_acc))
            if r_acc[worst] <= n_sigma * max(float(r_acc.std()), 1e-6):
                break
            del acc[worst]
            rejects += 1
        dx, dy = float(d[0]), float(d[1])
        if abs(dx) < 1e-9:                   # vertical corridor
            w_out, b_out = 1e6 * (1.0 if dy >= 0 else -1.0), -1e6 * c[0]
        else:
            w_out = dy / dx
            b_out = c[1] - w_out * c[0]
        return (w_out, b_out), acc, [i for i in range(len(pts))
                                     if i not in acc]

    def _fit_line_robust(self, points, max_reject=3, n_sigma=2.5):
        """Least squares with conservative outlier rejection on the midpoints."""
        pts = list(points)
        fit = self._fit_line(pts)
        rejected = 0
        while fit is not None and len(pts) > 4 and rejected < max_reject:
            w_slope, b = fit
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            resid = np.abs(ys - (w_slope * xs + b))
            worst = int(np.argmax(resid))
            if resid.max() <= n_sigma * max(resid.std(), 1e-6):
                break
            pts.pop(worst)
            rejected += 1
            fit = self._fit_line(pts)
        return fit


def line_se(length, angle_deg):
    """Binary line structuring element of given length and orientation."""
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2.0
    r = math.radians(angle_deg)
    dx, dy = math.cos(r), math.sin(r)
    p1 = (int(round(c - dx * (length - 1) / 2)),
          int(round(c - dy * (length - 1) / 2)))
    p2 = (int(round(c + dx * (length - 1) / 2)),
          int(round(c + dy * (length - 1) / 2)))
    cv2.line(k, p1, p2, 1, 1)
    return k


def robust_morphology(binary, open_len=9, min_area=12,
                      fence="if", fence_k=1.5, max_fence_frac=0.5,
                      if_contamination=0.15, if_seed=42):
    """EXPERIMENTAL: weed-robust replacement for Eq. (3)'s single opening.

    Opening-only chain:

      1. UNION OF ORIENTED LINE OPENINGS over six directions: a pixel
         survives only where the mask contains a straight run of
         `open_len` px in at least one direction. Row segments qualify;
         round/irregular weed blobs cannot hold such a line and vanish -
         pure shape prior, no corridor knowledge. The paper's Eq. (3) is
         the special case of ONE diagonal at ONE scale.
      2. ATTRIBUTE AREA OPENING - components below `min_area` px are
         removed regardless of orientation.
      3. EMBEDDED OUTLIER FENCE on component shape attributes
         [log-area, elongation, extent]:
           fence="if":  IsolationForest over the three attributes
                        (multivariate; catches jointly odd blobs)
           fence="iqr": one-sided Tukey fences - only unusually small
                        or unusually round blobs are condemned
         Safety: blobs larger than 5% of total vegetation are never
         removed, and a valve aborts when > max_fence_frac of the
         vegetation would be erased (population not crop-like).
    """
    acc = None
    for ang in (0, 30, 60, 90, 120, 150):
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  line_se(open_len, ang))
        acc = opened if acc is None else (acc | opened)

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(acc, 8)
    ids, feats = [], []
    total_area = 0
    for i in range(1, n_cc):
        x, y, bw, bh, area = stats[i]
        total_area += area
        if area < min_area:
            continue                     # attribute area opening
        elong = max(bw, bh) / max(1.0, float(min(bw, bh)))
        extent = area / max(1.0, float(bw * bh))
        feats.append([math.log10(max(1, area)), elong, extent])
        ids.append(i)

    if fence != "off" and len(feats) >= 8:
        F = np.array(feats, dtype=np.float64)
        if fence == "if":
            iso = IsolationForest(n_estimators=100,
                                  contamination=if_contamination,
                                  random_state=if_seed)
            flagged = iso.fit_predict(F) == -1
        else:
            flagged = np.zeros(len(F), dtype=bool)
            for col in range(F.shape[1]):
                q1, q3 = np.quantile(F[:, col], [0.25, 0.75])
                lo = q1 - fence_k * (q3 - q1)
                flagged |= F[:, col] < lo
        big_px = 0.05 * max(1, total_area)
        veg_flagged = int(sum(stats[i][4] for i, fl in zip(ids, flagged)
                              if fl and stats[i][4] <= big_px))
        if veg_flagged <= max_fence_frac * max(1, total_area):
            for i, fl in zip(ids, flagged):
                if fl and stats[i][4] <= big_px:
                    acc[labels == i] = 0
    return acc


def pre_corridor_clean(binary, n_strips=10, l_frac=0.05, pad_px=6,
                       min_area=12, max_blob_frac=0.05):
    """EXPERIMENTAL: structural clean BEFORE any corridor is found.

    Per strip, the paper's own feature extraction (Eq. 5-8) defines which
    COLUMNS carry row structure: the T = M + E filter plus gap-L
    clustering yields row-candidate bands. Every connected component is
    then judged at blob level: if its horizontal extent overlaps a band
    of its own or an adjacent strip, it stays - otherwise it is erased
    as unaligned weed. Whole blobs are kept or dropped (never shaved),
    and blobs larger than max_blob_frac of all vegetation are always
    spared: a merged canopy spanning several rows legitimately overlaps
    many bands via its extent, while an isolated mid-furrow weed touches
    none. Components in strips without any bands are left alone rather
    than guessed away.

    Returns (cleaned_mask, info dict).
    """
    h, w = binary.shape[:2]
    dh = h // n_strips
    l_thresh = max(2, int(l_frac * w))

    # per-strip row-candidate bands from Eq. 5-8
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
            continue                     # speckle: leave (opening's job)
        mu = strip_of(cy)
        tol = []
        for m_u in (mu - 1, mu, mu + 1):
            tol += [(a - pad_px, b + 1 + pad_px)
                    for a, b in bands.get(m_u, [])]
        if not tol:
            continue                     # no structure known here: keep
        if area > big_px:
            continue                     # giant cap: too big to condemn
        hi_x = x0 + bw_ + pad_px
        lo_x = x0 - pad_px
        if any(hi_x > lo and lo_x < hi for lo, hi in tol):
            continue                     # touches row structure
        cleaned[labels == i] = 0
        removed += 1
        removed_area += area
    return cleaned, {
        "applied": removed > 0,
        "n_removed": removed,
        "removed_px": removed_area,
        "removed_frac": removed_area / max(1, total),
    }


def isolate_forest_clean(binary, exg8, otsu_t, anchors,
                         contamination=0.30, seed=42, min_components=10,
                         min_area=12, keep_dist_frac=0.02,
                         max_removed_frac=0.85, max_blob_frac=0.05,
                         scorer="if", z_thresh=3.5):
    """EXPERIMENTAL: Isolation-Forest removal of off-corridor vegetation.

    Aggressive mode. The pass-1 anchor lines protect a generous band:
    components whose centroid lies within keep_dist_frac * width of an
    anchor - the row/corridor neighbourhood itself, so crop-row segments
    are never candidates and cannot come out broken - or tiny speckle
    (< min_area px) are preserved; everything else is scored and the
    weediest-looking candidates get erased.

    scorer="if":   IsolationForest (contamination=0.30) over 5 features:
                   [d_anchor, log_area, elongation, exg_mean, exg_margin].
    scorer="madz": one-sided robust fence over the CANDIDATE pool's own
                   d_anchor values: cutoff = median + z_thresh * MAD.
                   z_thresh is in raw MAD units, so lowering it pulls the
                   fence toward the crop rows (the keep band already
                   protects everything closest to them). Deterministic
                   and much cheaper than a forest.

    Common features per blob:
      d_anchor   perpendicular centroid distance to the nearest anchor,
                 normalized by image width (core signal)
      log_area   log10 of component pixel count
      elongation max(bw,bh)/min(bw,bh) (row-aligned growth is stretched)
      exg_mean   mean ExG intensity inside the component
      exg_margin mean ExG minus the Otsu threshold (weak vs strong green)

    A safety valve still aborts the cleaning of an image when more than
    max_removed_frac of the candidate pixels would be erased - at that
    point the anchors themselves are suspect, not the blobs. Additionally,
    no single component larger than max_blob_frac * total vegetation is
    ever erased: a blob that big may be a row segment curving away from
    the anchors, and removing it can gut a sparse mask (aerial views).

    Returns (cleaned_binary, info dict).
    """
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if n_cc <= 1:
        return binary.copy(), {"applied": False, "reason": "no components"}

    width = binary.shape[1]
    keep_px = keep_dist_frac * width
    total_area = cand_area = 0
    comp_ids, feats = [], []         # candidates: scored AND removable
    for i in range(1, n_cc):
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]
        total_area += area
        d_min = min(abs(w_s * cx - cy + b) / math.hypot(w_s, 1.0)
                    for w_s, b in anchors)
        if area < min_area or d_min <= keep_px:
            continue                     # protected: speckle / corridor band
        elong = max(bw, bh) / max(1.0, float(min(bw, bh)))
        m = labels[y:y + bh, x:x + bw] == i
        exg_mean = float(exg8[y:y + bh, x:x + bw][m].mean())
        feats.append([d_min / width, math.log10(max(1, area)), elong,
                      exg_mean, exg_mean - otsu_t])
        comp_ids.append(i)
        cand_area += area

    if len(feats) < min_components:
        return binary.copy(), {"applied": False,
                               "reason": f"only {len(feats)} candidates"}

    X = np.array(feats, dtype=np.float64)
    extra = {}
    if scorer == "madz":
        # one-sided fence over the candidate pool's own distances; the
        # keep band has already removed everything closest to the rows,
        # so lowering z_thresh walks the cutoff back toward the corridor
        Xd = X[:, 0]
        med = float(np.median(Xd))
        mad = float(np.median(np.abs(Xd - med)))
        cutoff = med + z_thresh * max(mad, 1e-6)
        pred = np.where(Xd > cutoff, -1, 1)
        extra = {"madz_median": round(med, 4), "madz_mad": round(mad, 4),
                 "madz_cutoff": round(cutoff, 4)}
    else:
        iso = IsolationForest(n_estimators=100, contamination=contamination,
                              random_state=seed)
        pred = iso.fit_predict(X)

    cleaned = binary.copy()
    removed = removed_area = spared_big = 0
    big_px = max_blob_frac * max(1, total_area)
    for i, p in zip(comp_ids, pred):
        if p != -1:
            continue
        area_i = stats[i][4]
        if area_i > big_px:
            spared_big += 1          # too big to condemn confidently
            continue
        cleaned[labels == i] = 0
        removed += 1
        removed_area += area_i
    if removed_area > max_removed_frac * max(1, cand_area):
        # IF condemned most of the contested area -> anchors were likely
        # wrong, do not trust the scrub
        return binary.copy(), {"applied": False,
                               "reason": "safety valve: removal too large"}
    return cleaned, {
        "applied": removed > 0,
        "n_components": len(comp_ids),
        "n_candidates_px": cand_area,
        "n_removed": removed,
        "n_spared_giants": spared_big,
        "removed_frac": removed_area / max(1, total_area),
        **extra,
    }


def nav_angle(nav_line):
    """Navigation-line angle from vertical in degrees (nan if no line)."""
    if nav_line is None:
        return float("nan")
    return math.degrees(math.atan(abs(1.0 / nav_line[0])))


def nav_report_angle(res):
    """Steering-relevant angle: curve tangent at the image base when a
    navigation curve exists, else the straight-line angle."""
    a = res.get("nav_base_angle_deg", float("nan"))
    if isinstance(a, float) and math.isfinite(a):
        return a
    return nav_angle(res.get("nav_line"))


def draw_results(bgr, res, draw_rois=False):
    """Figure 11/12 style visualization.

    Returns (overlay_on_original, overlay_on_binary). Lines are drawn in the
    cropped coordinate frame, then shifted back by the preprocessing offset.
    """
    dx, dy = res["crop_offset"]
    original = bgr.copy()
    bh, bw = res["binary"].shape[:2]
    binary_vis = cv2.cvtColor(res["binary"], cv2.COLOR_GRAY2BGR)

    if draw_rois:
        for i, (x_lo, x_hi, y1, y2) in enumerate(res["rois"], start=1):
            cv2.rectangle(binary_vis, (int(x_lo), int(y1)),
                          (int(x_hi), int(y2)), COLOR_ROI, 2)
            cv2.putText(binary_vis, str(i), (int(x_lo) + 4, int(y2) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Q midpoints: bright green = used by the nav fit,
        # red = rejected/excluded (strip-1 dot or outlier)
        for qx, qy in res.get("q_rejected", []):
            cv2.circle(binary_vis, (int(qx), int(qy)), 3, (0, 0, 255), -1)
        for qx, qy in res.get("q_accepted", []):
            cv2.circle(binary_vis, (int(qx), int(qy)), 3, (0, 255, 0), -1)

    def draw_line(img, w_slope, b, color, thickness=2):
        h = img.shape[0]
        # fit is y = w*x + b  ->  x = (y - b) / w
        if abs(w_slope) < 1e-6:  # degenerate horizontal fit
            cv2.line(img, (0, int(b) + dy), (img.shape[1], int(b) + dy),
                     color, thickness)
            return
        p1 = (int((0 - b) / w_slope), 0)             # x at y = 0
        p2 = (int(((h - 1) - b) / w_slope), h - 1)   # x at y = h-1
        cv2.line(img, (p1[0] + dx, p1[1] + dy),
                 (p2[0] + dx, p2[1] + dy), color, thickness)

    for w_slope, b, _ in res["det_lines"]:
        draw_line(binary_vis, w_slope, b, COLOR_DET_LINE, 2)
        draw_line(original, w_slope, b, COLOR_DET_LINE, 2)
    if res.get("nav_curve"):
        poly = np.array([(int(x) + dx, int(y) + dy)
                         for x, y in res["nav_curve"]], dtype=np.int32)
        cv2.polylines(binary_vis, [poly], False, COLOR_NAV_LINE, 2,
                      cv2.LINE_AA)
        cv2.polylines(original, [poly], False, COLOR_NAV_LINE, 2,
                      cv2.LINE_AA)
    elif res["nav_line"] is not None:
        w_slope, b = res["nav_line"]
        draw_line(binary_vis, w_slope, b, COLOR_NAV_LINE, 2)
        draw_line(original, w_slope, b, COLOR_NAV_LINE, 2)
    return original, binary_vis


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive multi-ROI crop row detection (Zhou et al. 2021)")
    parser.add_argument("--input", default="../../Photos",
                        help="Folder or glob of images")
    parser.add_argument("--results_dir", default="./result_test")
    parser.add_argument("--n_strips", type=int, default=10,
                        help="N horizontal image strips (paper: N=10)")
    parser.add_argument("--l_frac", type=float, default=0.05,
                        help="Clustering distance L as fraction of width")
    parser.add_argument("--border_frac", type=float, default=0.02,
                        help="Edge cropping fraction (Sec. 2.2)")
    parser.add_argument("--max_side", type=int, default=1920,
                        help="Downscale so the longest side <= this "
                             "(paper used 1920x1080; 0 disables)")
    parser.add_argument("--index", choices=("raw", "normalized"),
                        default="raw",
                        help="ExG color index: 'raw' = 2G-R-B (robust, "
                             "default), 'normalized' = paper Eq.(1)-(2)")
    parser.add_argument("--rotate_names", default="bev5,bev6,bev7",
                        help="Comma-separated image-name prefixes that must "
                             "be rotated before detection (rows run "
                             "perpendicular to the travel direction); "
                             "empty string disables")
    parser.add_argument("--rotate_deg", type=int, default=90,
                        choices=(90, 180, 270),
                        help="Rotation applied to --rotate_names images, "
                             "clockwise (default 90)")
    parser.add_argument("--post_scrub", action="store_true",
                        help="OPT-IN: run the anchor-based outlier scrub "
                             "(--scorer if/madz) after detection. Off by "
                             "default - the embedded struct-clean handles "
                             "weed removal, and the scrub was observed to "
                             "delete real crop rows (photo_11/17)")
    parser.add_argument("--if_contamination", type=float, default=0.30,
                        help="IsolationForest contamination (expected "
                             "outlier fraction), default 0.30")
    parser.add_argument("--if_seed", type=int, default=42,
                        help="IsolationForest random_state for "
                             "reproducibility")
    parser.add_argument("--if_keep_dist", type=float, default=0.02,
                        help="Anchor protection band as fraction of image "
                             "width; blobs whose centroid lies closer to an "
                             "anchor are never removed - keeps crop-row "
                             "segments intact (default 0.02)")
    parser.add_argument("--if_min_area", type=int, default=12,
                        help="Components smaller than this many pixels are "
                             "never removed (default 12)")
    parser.add_argument("--if_max_removed", type=float, default=0.85,
                        help="Safety valve: abort cleaning when more than "
                             "this fraction of candidate pixels would be "
                             "erased (default 0.85)")
    parser.add_argument("--if_max_blob", type=float, default=0.05,
                        help="Never erase a single component larger than "
                             "this fraction of total vegetation (default "
                             "0.05)")
    parser.add_argument("--scorer", choices=("if", "madz"), default="if",
                        help="Outlier scorer for the scrub: 'if' = "
                             "IsolationForest over 5 features (default), "
                             "'madz' = deterministic robust z-score on "
                             "distance-to-anchor only - cheaper, suited "
                             "to crop-dominated scenes")
    parser.add_argument("--madz_thresh", type=float, default=0.5,
                        help="Fence position for --scorer madz, in MAD "
                             "units above the candidate median; smaller = "
                             "tighter to the crop rows (default 0.5)")
    parser.add_argument("--min_flank_frac", type=float, default=0.01,
                        help="Minimum share of a strip's surviving "
                             "projection mass for a cluster to qualify as "
                             "a flanking row - blocks 1-px weed spikes "
                             "(measured: slivers <=0.006, weakest real "
                             "rows >=0.015) without starving weak upper "
                             "strips (default 0.01)")
    parser.add_argument("--nav_include_first", action="store_true",
                        help="Include strip 1's midpoint in the navigation "
                             "line fit (default: excluded - the full-width "
                             "initial view can anchor outside the corridor "
                             "and its midpoint has maximal slope leverage)")
    parser.add_argument("--morph", choices=("paper", "robust", "struct"),
                        default="struct",
                        help="Segmentation morphology: 'struct' (default) "
                             "= Eq. (3) opening + embedded band-support "
                             "weed removal before any corridor is found; "
                             "'paper' = Eq. (3) only; 'robust' = oriented "
                             "openings + attribute fence experiments")
    parser.add_argument("--open_len", type=int, default=9,
                        help="Line length for --morph robust oriented "
                             "openings; structures shorter than this in "
                             "every direction are removed (default 9)")
    parser.add_argument("--fence_k", type=float, default=1.5,
                        help="Tukey IQR multiplier for the classical "
                             "outlier fence embedded in --morph robust "
                             "(default 1.5)")
    parser.add_argument("--morph_fence", choices=("if", "iqr"),
                        default="if",
                        help="Outlier fence embedded in --morph robust: "
                             "'if' = IsolationForest over shape "
                             "attributes (default), 'iqr' = one-sided "
                             "Tukey fences")
    parser.add_argument("--nav_curve", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Draw the navigation path as a smoothed "
                             "regression spline through the accepted dots "
                             "(safety-envelope capped) instead of a "
                             "straight line; CSV angle = tangent at image "
                             "base (default: on)")
    parser.add_argument("--weed_gate", type=float, default=0.35,
                        help="Run the Isolation-Forest scrub only when the "
                             "weed-pressure metric (max of in-furrow "
                             "vegetation occupancy and flanking-pick "
                             "failure rate) reaches this threshold; 0 "
                             "forces cleaning on every image "
                             "(default 0.35)")
    parser.add_argument("--max_bend", type=float, default=45.0,
                        help="Skip the scrub when the corridor bends more "
                             "than this many degrees between image halves "
                             "- curved rows are geometry, not weeds "
                             "(default 45)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.png"))
                       + glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        paths = sorted(glob.glob(args.input))
    if not paths:
        print(f"No images found for input: {args.input}")
        return

    overlays_dir = os.path.join(args.results_dir, "overlays")
    multiroi_dir = os.path.join(args.results_dir, "multiroi_masks")
    binary_dir = os.path.join(args.results_dir, "binary")
    for d in (overlays_dir, multiroi_dir, binary_dir):
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "detection_data.csv")

    detector = MultiROIDetector(n_strips=args.n_strips, l_frac=args.l_frac,
                                border_frac=args.border_frac, index=args.index,
                                min_flank_frac=args.min_flank_frac,
                                nav_include_first=args.nav_include_first,
                                morph=args.morph, open_len=args.open_len,
                                fence_k=args.fence_k,
                                morph_fence=args.morph_fence,
                                nav_curve=args.nav_curve)
    rotate_prefixes = tuple(p.strip() for p in args.rotate_names.split(",")
                            if p.strip())
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE,
               180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "filename",
            "nav_angle_pass1_deg", "nav_angle_final_deg",
            "n_nav_points", "n_detection_lines",
            "weed_pressure", "if_removed_components", "if_removed_veg_frac",
            "time_ms"])

    ok = fail = 0
    times = []
    deltas = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"[ERROR] Cannot read '{path}'")
            fail += 1
            continue
        if name.startswith(rotate_prefixes):
            bgr = cv2.rotate(bgr, rot_map[args.rotate_deg])   # rows vertical
        scale = 1.0
        if args.max_side > 0:
            side = max(bgr.shape[:2])
            if side > args.max_side:
                scale = args.max_side / side
                bgr = cv2.resize(bgr, (int(round(bgr.shape[1] * scale)),
                                       int(round(bgr.shape[0] * scale))),
                                 interpolation=cv2.INTER_AREA)
        try:
            # Pass 1: base pipeline; its lines become the IF anchors and
            # feed the cheap weed-pressure gate. Always computed so the
            # CSV keeps the uncleaned reference angle.
            res1 = detector.detect(bgr)
            ang1 = nav_report_angle(res1)

            final = res1
            info = {"applied": False}
            pressure = float(res1.get("weed_pressure", 0.0))
            bend = float(res1.get("nav_bend_deg", 999.0))
            t_if = time.perf_counter()
            if args.post_scrub:
                anchors = [tuple(line[:2]) for line in res1["det_lines"]]
                if res1["nav_line"] is not None:
                    anchors.append(tuple(res1["nav_line"]))
                if anchors:
                    # Gate 1: scrub only on substantial weed pressure.
                    # Gate 2: never scrub a strongly bent corridor - there
                    # the straight anchors are wrong, not the vegetation.
                    if bend > args.max_bend:
                        info = {"applied": False,
                                "reason": f"curved corridor ({bend:.0f} deg)"
                                          " - anchors untrustworthy"}
                    elif pressure < args.weed_gate:
                        info = {"applied": False,
                                "reason": f"weed gate {pressure:.3f} < "
                                          f"{args.weed_gate:.3f}"}
                    else:
                        cleaned, info = isolate_forest_clean(
                            res1["binary"], res1["exg"], res1["otsu_t"],
                            anchors,
                            contamination=args.if_contamination,
                            seed=args.if_seed,
                            min_area=args.if_min_area,
                            keep_dist_frac=args.if_keep_dist,
                            max_removed_frac=args.if_max_removed,
                            max_blob_frac=args.if_max_blob,
                            scorer=args.scorer,
                            z_thresh=args.madz_thresh)
                        if info.get("applied"):
                            final = detector.detect_from_binary(
                                cleaned, res1["crop_offset"])
                            final["exg"] = res1["exg"]
                            final["otsu_t"] = res1["otsu_t"]
            total_ms = (res1["time_ms"]
                        + final.get("time_ms", 0.0) * (final is not res1)
                        + (time.perf_counter() - t_if) * 1000.0)

            overlay, mask_vis = draw_results(bgr, final, draw_rois=True)
            cv2.imwrite(os.path.join(overlays_dir, f"{name}_result.png"), overlay)
            cv2.imwrite(os.path.join(multiroi_dir, f"{name}_multiroi.png"), mask_vis)
            cv2.imwrite(os.path.join(binary_dir, f"{name}_binary.png"),
                        final["binary"])

            ang_f = nav_report_angle(final)
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    name, f"{ang1:.3f}", f"{ang_f:.3f}",
                    len(final["q"]), len(final["det_lines"]),
                    f"{pressure:.4f}" if pressure == pressure else "nan",
                    info.get("n_removed", 0),
                    f"{info.get('removed_frac', 0.0):.4f}",
                    f"{total_ms:.1f}"])
            times.append(total_ms)
            if info.get("applied"):
                deltas.append(abs(ang_f - ang1))
            if info.get("applied"):
                deltas.append(abs(ang_f - ang1))
            scrub = (f"scrub removed {info['n_removed']}"
                     f"/{info['n_components']} comps "
                     f"({info['removed_frac']*100:.1f}% veg)"
                     if info.get("applied")
                     else f"scrub skipped ({info.get('reason', 'nothing flagged')})")
            print(f"[OK] {name}: weed={pressure:5.3f} "
                  f"nav angle {ang1:6.2f} -> {ang_f:6.2f} deg"
                  f" from vertical, {len(final['det_lines'])} det lines, "
                  f"{scrub}, {total_ms:.1f} ms")
            ok += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Failed on '{name}': {e}")
            fail += 1

    if times:
        print(f"\nDone. {ok} succeeded, {fail} failed.")
        print(f"Mean computation time: {np.mean(times):.1f} ms "
              f"(std {np.std(times):.1f}) over {len(times)} images "
              f"(paper reports 240.8 ms on 1920x1080)")
        if deltas:
            print(f"IF changed the nav angle on {len(deltas)} images "
                  f"(mean |delta| {np.mean(deltas):.2f} deg)")
    print(f"Results saved to: {os.path.abspath(args.results_dir)}")


if __name__ == "__main__":
    main()
