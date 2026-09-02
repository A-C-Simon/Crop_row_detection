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

Usage:
    python multi_roi.py --input ../../Photos --results_dir ./result
"""

import argparse
import csv
import glob
import math
import os
import time

import cv2
import numpy as np

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


def preprocess(bgr, border_frac=0.02, index="raw"):
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

    Returns (binary mask with crop=255, ExG grayscale, (dx, dy) crop offset).
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

    _, binary = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_K)  # Eq. (3)
    return binary, exg8, (dx, dy)


def column_projection(strip, x_lo, x_hi):
    """Eq. (5)-(6): vertical projection of a strip within [x_lo, x_hi).

    Z(i) = number of 255-valued pixels in column i (a = 1), then filtered
    to 0 wherever Z(i) < T with T = M + E (Eq. 7-8).
    """
    window = strip[:, x_lo:x_hi]
    z = (window == 255).sum(axis=0).astype(np.float64)  # Eq. (5)
    t = z.mean() + z.std()                              # Eq. (7)-(8)
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
                 min_points_per_row=2, index="raw"):
        self.n = n_strips
        self.l_frac = l_frac             # clustering distance L as fraction of W
        self.border_frac = border_frac
        self.min_points_per_row = min_points_per_row
        self.index = index               # ExG variant: "raw" or "normalized"

    def detect(self, bgr):
        """Run the full pipeline on one BGR image; returns results dict."""
        t0 = time.perf_counter()
        binary, exg8, (dx, dy) = preprocess(bgr, self.border_frac, self.index)
        # All strip/ROI geometry lives in the CROPPED binary frame; the crop
        # offset is only re-applied when drawing on the original image.
        h, w = binary.shape[:2]

        dh = h // self.n                     # Eq. (4): strip height dh = H/N
        l_thresh = max(2, int(self.l_frac * w))
        rois = []            # per-strip (x_lo, x_hi, y_top, y_bot), bottom->top
        q = []               # point set Q of renewed midpoints (Eq. 11)
        left_pts = []        # CLeft chain across strips (detection line)
        right_pts = []       # CRight chain across strips (detection line)

        mo_x = w / 2.0                       # Sec. 2.3.2: initial midpoint MO
        x_lo, x_hi = 0, w                    # initial ROI = full strip width

        for mu in range(1, self.n + 1):      # strip 1 = bottom strip
            y1 = h - mu * dh                 # top row of strip mu
            strip = binary[max(0, y1):y1 + dh, :]

            z = column_projection(strip, x_lo, x_hi)
            clusters = cluster_feature_columns(z, x_lo, l_thresh)

            if clusters:
                c_left, c_right = flanking_clusters(clusters, mo_x)
                # Eq. (9): D = max(CRight) - min(CLeft)
                if c_left is not None and c_right is not None:
                    d = c_right[1] - c_left[0]
                elif c_right is not None:
                    d = c_right[1] - c_right[0]
                else:
                    d = c_left[1] - c_left[0]
                if d <= 0:
                    d = max(4, x_hi - x_lo)
                # Fig. 8b: ROI = [min(CLeft) - 0.1D, max(CRight) + 0.1D]
                base_lo = c_left[0] if c_left is not None else c_right[0]
                base_hi = c_right[1] if c_right is not None else c_left[1]
                new_lo = int(max(0, base_lo - 0.1 * d))
                new_hi = int(min(w, base_hi + 0.1 * d))
                if new_hi - new_lo >= 4:     # keep sane windows only
                    x_lo, x_hi = new_lo, new_hi
                # Eq. (10): renewed Mx = midpoint between the two flanking
                # rows. Cluster centers are used rather than outer edges so
                # wide (merged) clusters cannot yank the corridor sideways.
                c_left_c = (c_left[0] + c_left[1]) / 2.0 if c_left else None
                c_right_c = (c_right[0] + c_right[1]) / 2.0 if c_right else None
                if c_left_c is not None and c_right_c is not None:
                    mo_x = (c_left_c + c_right_c) / 2.0
                else:
                    mo_x = c_left_c if c_left_c is not None else c_right_c
            # else Sec. 2.3.3: no feature points -> previous ROI window & Mx

            my = h - mu * dh                 # Eq. (10): My(mu) = My(mu-1) - dh
            rois.append((x_lo, x_hi, max(0, y1), y1 + dh))
            q.append((mo_x, my))             # renewed MO into Q

            # Per-strip flanking cluster centers, used for the detection
            # lines (Section 3): the CLeft chain and CRight chain across the
            # strips are the crop rows bordering the travelling area. Strip 1
            # is skipped: with the full-width initial view its flanking pick
            # can anchor on rows outside the travelling area. When only one
            # cluster is found it continues the side its center falls on
            # (rows converge but do not swap sides).
            cy = y1 + dh / 2.0
            if clusters and mu > 1:
                only = c_left if c_right is None else c_right
                if c_left is not None and c_right is not None:
                    left_pts.append(((c_left[0] + c_left[1]) / 2.0, cy))
                    right_pts.append(((c_right[0] + c_right[1]) / 2.0, cy))
                else:
                    side_pts = left_pts if only is c_left else right_pts
                    side_pts.append(((only[0] + only[1]) / 2.0, cy))

        # Section 2.4: navigation line over Q (initial estimate not in Q).
        # A robust refinement of the paper's least squares: strips where the
        # flanking-cluster pick jumps to a wrong row pair produce outlier
        # midpoints; iteratively reject the worst residual (max 3 points,
        # 2.5 sigma) so the line follows the stable corridor majority.
        nav_line = self._fit_line_robust(q)

        # Section 3: detection lines, one least-squares line per row chain
        det_lines = []
        for pts in (left_pts, right_pts):
            if len(pts) < self.min_points_per_row:
                continue
            fit = self._fit_line(pts)
            if fit is not None:
                det_lines.append((fit[0], fit[1], len(pts)))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "binary": binary,
            "exg": exg8,
            "crop_offset": (dx, dy),
            "rois": rois,
            "q": q,
            "nav_line": nav_line,    # (w, b) for y = w*x + b or None
            "det_lines": det_lines,  # list of (w, b, n_pts)
            "time_ms": elapsed_ms,
        }

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
        for qx, qy in res["q"]:
            cv2.circle(binary_vis, (int(qx), int(qy)), 3, (0, 0, 255), -1)

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
    if res["nav_line"] is not None:
        w_slope, b = res["nav_line"]
        draw_line(binary_vis, w_slope, b, COLOR_NAV_LINE, 2)
        draw_line(original, w_slope, b, COLOR_NAV_LINE, 2)
    return original, binary_vis


def make_composite(bgr, res):
    """Vertical single-window composite: binary (top) / mask (middle) / overlay
    (bottom) stacked vertically to maximize vertical size on portrait pages and
    screens.

    All three panels are scaled to the full overlay width so each view remains
    large when the composite is fitted to page width; stacked vertically the
    window fills the page height instead of shrinking to an ultra-wide strip.
    """
    overlay, mask_vis = draw_results(bgr, res, draw_rois=True)
    binary_bgr = cv2.cvtColor(res["binary"], cv2.COLOR_GRAY2BGR)

    w_target = overlay.shape[1]

    def resize_to_w(img, w):
        if img.shape[1] == w:
            return img
        scale = w / img.shape[1]
        new_h = int(round(img.shape[0] * scale))
        return cv2.resize(img, (w, new_h), interpolation=cv2.INTER_NEAREST)

    mask_resized = resize_to_w(mask_vis, w_target)
    binary_resized = resize_to_w(binary_bgr, w_target)

    bar_h = 36

    def with_label(img, text):
        w = img.shape[1]
        bar = np.full((bar_h, w, 3), (32, 32, 32), dtype=np.uint8)
        cv2.putText(bar, text, (10, bar_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return np.vstack([bar, img])

    binary_labeled = with_label(binary_resized, "Binary (ExG + Otsu)")
    mask_labeled = with_label(mask_resized, "MultiROI mask + ROIs")
    overlay_labeled = with_label(overlay, "Overlay (original)")

    sep_h = 4
    sep = np.full((sep_h, w_target, 3), (255, 255, 255), dtype=np.uint8)
    composite = np.vstack([binary_labeled, sep, mask_labeled, sep, overlay_labeled])
    return composite


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive multi-ROI crop row detection (Zhou et al. 2021)")
    parser.add_argument("--input", default="../../Photos",
                        help="Folder or glob of images")
    parser.add_argument("--results_dir", default="./result")
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
    args = parser.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.png"))
                       + glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        paths = sorted(glob.glob(args.input))
    if not paths:
        print(f"No images found for input: {args.input}")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "detection_data.csv")

    detector = MultiROIDetector(n_strips=args.n_strips, l_frac=args.l_frac,
                                border_frac=args.border_frac, index=args.index)
    rotate_prefixes = tuple(p.strip() for p in args.rotate_names.split(",")
                            if p.strip())
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE,
               180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "filename", "nav_angle_from_vertical_deg", "n_nav_points",
            "n_detection_lines", "time_ms"])

    ok = fail = 0
    times = []
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
            res = detector.detect(bgr)
            composite = make_composite(bgr, res)
            cv2.imwrite(os.path.join(args.results_dir, f"{name}.png"), composite)

            if res["nav_line"] is not None:
                ang = math.degrees(math.atan(abs(1.0 / res["nav_line"][0])))
            else:
                ang = float("nan")
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    name, f"{ang:.3f}", len(res["q"]),
                    len(res["det_lines"]), f"{res['time_ms']:.1f}"])
            times.append(res["time_ms"])
            print(f"[OK] {name}: nav angle={ang:6.2f} deg from vertical, "
                  f"{len(res['det_lines'])} detection lines, "
                  f"{res['time_ms']:.1f} ms")
            ok += 1
        except Exception as e:
            print(f"[ERROR] Failed on '{name}': {e}")
            fail += 1

    if times:
        print(f"\nDone. {ok} succeeded, {fail} failed.")
        print(f"Mean computation time: {np.mean(times):.1f} ms "
              f"(std {np.std(times):.1f}) over {len(times)} images "
              f"(paper reports 240.8 ms on 1920x1080)")
    print(f"Results saved to: {os.path.abspath(args.results_dir)}")


if __name__ == "__main__":
    main()
