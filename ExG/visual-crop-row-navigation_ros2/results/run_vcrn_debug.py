#!/usr/bin/env python3
"""
VCRN DEBUG — Visual Crop Row Navigation (Bonn) stage-by-stage debug visualizer
================================================================================
Replicates `src/agribot_vs.cpp: CropRowFeatures / getContureCenters /
is_in_neigbourhood / FitLineOnContures` in Python and renders every decision
point in a 4x4 debug composite, mirroring `ClusterAlg/run_carolif.py` debug.

Pipeline (agribot_vs_nodehandler.cpp:49 CropRow_Tracking):
  1 Resize to 640x480 (params/agribot_vs_run.yaml width/height)
  2 BGR -> HSV -> split H/S/V -> inRange per channel -> combined mask
  3 findContours -> getContureCenters (approxPolyDP+minEnclosingCircle)
  4 filterContures (degenerate as shipped) + is_in_neigbourhood window
  5 FitLineOnContures (cv::fitLine DIST_L2) -> homogeneous border clip

Usage:
  python3 run_vcrn_debug.py [--input DIR] [--output DIR] [--pattern GLOB]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans

# ---------------------------------------------------------------------------
# params
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = Path(__file__).parent.parent / "params" / "agribot_vs_run.yaml"

def load_params(yaml_path: Path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    p = data.get("agribot_vs", {}).get("ros__parameters", data)
    # fallback defaults matching agribot_vs.cpp: readRUNParmas
    return {
        "min_Hue": int(p.get("min_Hue", 40)),
        "max_Hue": int(p.get("max_Hue", 80)),
        "min_Saturation": int(p.get("min_Saturation", 50)),
        "max_Saturation": int(p.get("max_Saturation", 255)),
        "min_Value": int(p.get("min_Value", 100)),
        "max_Value": int(p.get("max_Value", 150)),
        "width": int(p.get("width", 640)),
        "height": int(p.get("height", 480)),
        "Scale": float(p.get("Scale", 0.7)),
        "minContourSize": float(p.get("minContourSize", 2.0)),
        "ex_Xc": int(p.get("ex_Xc", 320)),
        "ex_Yc": int(p.get("ex_Yc", 240)),
        "nh_L": int(p.get("nh_L", 120)),
        "nh_H": int(p.get("nh_H", 250)),
        "nh_offset": int(p.get("nh_offset", 200)),
        # Isolation Forest inside window (anomaly removal before fitLine)
        "iso_enabled": bool(p.get("iso_enabled", True)),
        "iso_contamination": float(p.get("iso_contamination", 0.15)),
        "iso_min_points": int(p.get("iso_min_points", 12)),
        "iso_n_estimators": int(p.get("iso_n_estimators", 100)),
        # Column-aware window spawning (dynamic Xc/L near chassis)
        "colaware_enabled": bool(p.get("colaware_enabled", True)),
        "colaware_y0_frac": float(p.get("colaware_y0_frac", 0.55)),
        # Gap-based multi-row inside window (keep central cluster)
        "gap_enabled": bool(p.get("gap_enabled", True)),
        "gap_eps": float(p.get("gap_eps", 14)),
        "gap_min_samples": int(p.get("gap_min_samples", 6)),
        "gap_min_points": int(p.get("gap_min_points", 12)),
    }

# ---------------------------------------------------------------------------
# helpers - replicate C++ logic
# ---------------------------------------------------------------------------
def hom2euc(v):
    return np.array([v[0]/v[2], v[1]/v[2]], dtype=np.float64)

def fit_line_clip(nh_points, width, height):
    """Replicate FitLineOnContures:247-291. Returns dict with raw line and clipped."""
    if len(nh_points) == 0:
        return None
    pts = np.array(nh_points, dtype=np.float32)
    # cv2.fitLine returns (vx,vy,x0,y0)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])

    # P1 = vx + x0, vy + y0 ; P2 = x0,y0 as in C++
    P1 = np.array([vx + x0, vy + y0, 1.0])
    P2 = np.array([x0, y0, 1.0])
    # border lines homogeneous
    top_left = np.array([0,0,1], dtype=float)
    top_right = np.array([width,0,1], dtype=float)
    bottom_left = np.array([0,height,1], dtype=float)
    bottom_right = np.array([width,height,1], dtype=float)
    l_ib = np.cross(bottom_left, bottom_right)
    l_it = np.cross(top_left, top_right)
    l_il = np.cross(top_left, bottom_left)
    l_ir = np.cross(top_right, bottom_right)
    l = np.cross(P1, P2)

    def inter(l1,l2):
        v = np.cross(l1,l2)
        if abs(v[2]) < 1e-9:
            return np.array([np.nan, np.nan])
        return hom2euc(v)

    R_t = inter(l, l_it)
    R_l = inter(l, l_il)
    R_b = inter(l, l_ib)
    R_r = inter(l, l_ir)
    R = np.vstack([R_t, R_l, R_b, R_r])  # 4x2
    # is_in_image_point strict 0<=x<=width,0<=y<=height
    inside = []
    for i in range(4):
        x,y = R[i]
        if 0 <= x <= width and 0 <= y <= height and np.isfinite(x):
            inside.append((float(x), float(y)))
    # also tolerate 2px variant for debug note
    inside_tol = []
    for i in range(4):
        x,y = R[i]
        if -2 <= x <= width+2 and -2 <= y <= height+2 and np.isfinite(x):
            inside_tol.append((float(x), float(y)))

    raw_p1 = (float(P1[0]), float(P1[1]))
    raw_p2 = (float(P2[0]), float(P2[1]))
    # direction for drawing long line
    norm = np.hypot(vx, vy) + 1e-9
    ux, uy = vx/norm, vy/norm
    # extend to image diagonal
    diag = np.hypot(width, height)*1.5
    ext_p1 = (float(x0 + ux*diag), float(y0 + uy*diag))
    ext_p2 = (float(x0 - ux*diag), float(y0 - uy*diag))

    return {
        "line": (vx,vy,x0,y0),
        "P1": raw_p1, "P2": raw_p2,
        "R": R,
        "inside": inside,
        "inside_tol": inside_tol,
        "ext_p1": ext_p1, "ext_p2": ext_p2,
        "avg_line": inside[:2] if len(inside)>=2 else [],
    }

def draw_window(img_bgr, Xc, Yc, L, H, color=(255,204,102), thick=3):
    out = img_bgr.copy()
    x = int(Xc - L/2)
    y = int(Yc - H/2)
    cv2.rectangle(out, (x,y), (x+L, y+H), color, thick)
    return out

def detect_column_aware_window(combined_mask, centers, params, width, height):
    """
    Column-aware window spawning: find dominant row column via vertical projection,
    spawn window centered on that column near chassis. Window width dynamic from inter-row gap.
    Returns (Xc, L, H, profile, peak_xs, chosen_idx, median_gap)
    """
    from scipy.signal import find_peaks
    # ROI near chassis: bottom 40% (y 0.6*H to H) where rows are widest and non-converging
    # For BEV full height also works, but base ROI still captures rows.
    y0 = int(height * 0.55)  # 55% from top = bottom 45% (like trace_rows NEAR_FRAC 0.55)
    # Use combined mask for profile (more robust than sparse centers when n small)
    roi_mask = combined_mask[y0:height, :]
    # column profile: sum of white pixels per column
    profile = roi_mask.sum(axis=0).astype(float)  # 0..(roi_h*255)
    # also consider centers contribution if mask sparse? we combine both?
    # Smooth with 1D Gaussian (sigma 5)
    if profile.max() > 0:
        prof_smooth = cv2.GaussianBlur(profile.reshape(1, -1), (0,0), 5).ravel()
    else:
        prof_smooth = profile
    # normalize for peak detection
    if prof_smooth.max() > 0:
        prof_n = prof_smooth / prof_smooth.max()
    else:
        prof_n = prof_smooth
    # peak distance: assume at least 30px between rows (640/15 rows ~42)
    peak_dist = max(25, int(width * 0.045))  # ~28 for 640
    peaks, props = find_peaks(prof_n, distance=peak_dist, prominence=0.12, height=0.15)
    peak_xs = peaks.tolist()
    median_gap = None
    chosen_idx = None
    chosen_x = None
    # dynamic L from median gap
    L_dynamic = params["nh_L"]
    if len(peak_xs) >= 2:
        gaps = np.diff(sorted(peak_xs))
        median_gap = float(np.median(gaps)) if len(gaps)>0 else None
        if median_gap and 30 <= median_gap <= 180:
            L_dynamic = int(np.clip(median_gap * 0.65, 60, 110))
    # choose peak closest to image center (robot center) among prominent peaks
    if len(peak_xs) > 0:
        # sort peaks by distance to center, then by prominence/height
        center = width // 2
        # prominence and height from props if available
        prominences = props.get("prominences", np.ones(len(peak_xs)))
        heights = props.get("peak_heights", np.ones(len(peak_xs)))
        # score = distance penalty + prominence bonus - choose min distance with high prominence
        # Rank by distance, but filter weak peaks: keep only top 60% prominence
        # Simpler: choose closest to center among peaks with height > median height
        # For now: closest to center
        distances = [abs(x - center) for x in peak_xs]
        # If multiple close, pick highest prominence among those within 60px of closest?
        # Use weighted score: distance - 30*prominence (prominence 0-1)
        scores = [d - 30*p for d, p in zip(distances, prominences if len(prominences)==len(peak_xs) else distances)]
        chosen_idx = int(np.argmin(scores))
        chosen_x = int(peak_xs[chosen_idx])
    else:
        chosen_x = width // 2
    # clamp Xc so window stays fully inside image
    half = L_dynamic // 2
    Xc = int(np.clip(chosen_x, half + 2, width - half - 2))
    # Yc stays at params ex_Yc (chassis base), but we keep it low 380
    Yc = params["ex_Yc"]
    H = params["nh_H"]
    # If no peaks, fallback to params Xc
    if len(peak_xs) == 0:
        Xc = params["ex_Xc"]
        L_dynamic = params["nh_L"]
    return Xc, L_dynamic, H, profile, prof_smooth, peak_xs, chosen_idx, median_gap, y0

def filter_gap_clusters(points, Xc, Yc, H, params):
    """
    Gap-based multi-row filter inside window - KMeans K=2 - Strategy 2 Closest to Center (Anti-Crop-Killer).
    If window contains 2 row clusters separated by a uniform gap, keep only the cluster whose centroid
    is closest to the bottom-center of the sliding window (rover nose, Xc, Yc+H/2), not just Xc.
    Calculates geometric center of all dots, runs KMeans K=2 on x, finds centroids, measures Euclidean
    distance to bottom-center. Keeps the cluster on the row the rover is physically driving on.
    Returns (kept_points, removed_points, n_clusters, labels, gap_info)
    """
    if not params.get("gap_enabled", True):
        return points, [], 1, None, {"reason": "disabled"}
    if len(points) < params.get("gap_min_points", 12):
        return points, [], 1, None, {"reason": f"n<{params.get('gap_min_points',12)}"}
    X = np.array([p[0] for p in points], dtype=float).reshape(-1,1)
    # geometric center of all dots (for logging, rover window center reference)
    all_cx = float(np.mean([p[0] for p in points]))
    all_cy = float(np.mean([p[1] for p in points]))
    # bottom-center of sliding window (rover nose): Xc, Yc+H/2
    # Yc and H are passed for Strategy 2
    bottom_cx = float(Xc)
    bottom_cy = float(Yc + H/2.0)
    # KMeans K=2 on x (lateral separation)
    try:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
        labels = kmeans.labels_
    except Exception as e:
        return points, [], 1, None, {"reason": f"kmeans failed {e}"}
    uniq = sorted(set(labels))
    n_clusters = len(uniq)
    if n_clusters != 2:
        return points, [], 1, labels, {"n_clusters": n_clusters, "all_center": (all_cx, all_cy), "bottom_center": (bottom_cx, bottom_cy)}
    # per-cluster centroids (mean x,y) and sizes
    centroids = {}
    medians = {}
    sizes = {}
    stds = {}
    for c in uniq:
        pts_c = [p for p,lb in zip(points, labels) if lb==c]
        xs = np.array([p[0] for p in pts_c], dtype=float)
        ys = np.array([p[1] for p in pts_c], dtype=float)
        centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
        medians[c] = float(np.median(xs))
        sizes[c] = int(len(pts_c))
        stds[c] = float(np.std(xs)) if len(xs)>1 else 0.0
    # gap between medians (uniform gap)
    c0, c1 = uniq[0], uniq[1]
    gap = abs(medians[c0] - medians[c1])
    avg_std = (stds[c0] + stds[c1]) / 2.0
    L = params.get("nh_L", 80)
    thresh = max(18.0, float(params.get("gap_eps", 14)), 2.0*avg_std + 4.0, float(L)*0.30)
    min_size_ratio = 0.15
    small_ratio = min(sizes.values()) / max(1, len(points))
    if gap < thresh or small_ratio < min_size_ratio:
        return points, [], 1, labels, {"n_clusters": 1, "gap": gap, "thresh": thresh, "medians": medians, "sizes": sizes, "stds": stds, "centroids": centroids, "all_center": (all_cx, all_cy), "bottom_center": (bottom_cx, bottom_cy), "reason": f"gap {gap:.0f} < thresh {thresh:.0f} or small_ratio {small_ratio:.2f} -> single row"}
    # Strategy 2: keep cluster whose centroid is closest to bottom-center (rover nose)
    dists = {}
    for c in uniq:
        cx, cy = centroids[c]
        dists[c] = float(np.hypot(cx - bottom_cx, cy - bottom_cy))
    # also compute distance to geometric center for logging
    dists_to_all = {c: float(np.hypot(centroids[c][0]-all_cx, centroids[c][1]-all_cy)) for c in uniq}
    sorted_c = sorted(uniq, key=lambda c: (dists[c], -sizes[c]))
    kept = sorted_c[0]
    kept_centroid = centroids[kept]
    kept_median = medians[kept]
    removed = [p for p,lb in zip(points, labels) if lb != kept]
    kept_points = [p for p,lb in zip(points, labels) if lb == kept]
    info = {"n_clusters": 2, "gap": gap, "thresh": thresh, "kept": kept, "kept_centroid": kept_centroid,
            "kept_median": kept_median, "medians": medians, "sizes": sizes, "stds": stds,
            "centroids": centroids, "dists_bottom": dists, "dists_all": dists_to_all,
            "all_center": (all_cx, all_cy), "bottom_center": (bottom_cx, bottom_cy), "labels": labels, "L": L}
    return kept_points, removed, 2, labels, info

# ---------------------------------------------------------------------------
# debug composite
# ---------------------------------------------------------------------------
def save_debug_composite(path: Path, bgr_orig: np.ndarray, intermediates: dict,
                         stem: str, params: dict):
    # unpack
    bgr_resized = intermediates["bgr_resized"]
    h_mask = intermediates["h_mask"]
    s_mask = intermediates["s_mask"]
    v_mask = intermediates["v_mask"]
    combined = intermediates["combined"]
    contours_img = intermediates["contours_img"]
    centers_img = intermediates["centers_img"]
    window_img = intermediates["window_img"]
    nh_img = intermediates["nh_img"]
    iso_img = intermediates.get("iso_img", nh_img)
    profile_img = intermediates.get("profile_img", np.zeros((80, params["width"],3),dtype=np.uint8))
    raw_line_img = intermediates["raw_line_img"]
    clipped_line_img = intermediates["clipped_line_img"]
    final_img = intermediates["final_img"]
    rejection_log = intermediates["rejection_log"]
    timings = intermediates["timings"]
    n_contours = intermediates["n_contours"]
    n_centers = intermediates["n_centers"]
    n_nh = intermediates["n_nh"]
    n_iso_in = intermediates.get("n_iso_in", n_nh)
    n_iso_out = intermediates.get("n_iso_out", 0)
    iso_applied = intermediates.get("iso_applied", False)
    # gap filter removed - keep for compatibility but not used
    n_gap_in = intermediates.get("n_gap_in", n_iso_in)
    n_gap_out = intermediates.get("n_gap_out", 0)
    gap_n_clusters = intermediates.get("gap_n_clusters", 1)
    width = params["width"]; height = params["height"]
    # dynamic window values (col-aware)
    Xc = intermediates.get("window_Xc", params["ex_Xc"])
    Yc = intermediates.get("window_Yc", params["ex_Yc"])
    L = intermediates.get("window_L", params["nh_L"])
    H = intermediates.get("window_H", params["nh_H"])
    colaware_peaks = intermediates.get("colaware_peaks", [])
    colaware_chosen = intermediates.get("colaware_chosen", None)
    colaware_median_gap = intermediates.get("colaware_median_gap", None)
    # original static for reference
    L_orig = params["nh_L"]
    H_orig = params["nh_H"]

    def bgr2rgb(img):
        if img is None or img.size==0:
            return np.zeros((10,10,3),dtype=np.uint8)
        if len(img.shape)==2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    iso_label = f"IsolationForest {params.get('iso_contamination',0.15)}" if iso_applied else "IsolationForest skipped"
    gap_s = f"{colaware_median_gap:.0f}" if colaware_median_gap is not None else "0"
    chosen_s = f"{colaware_peaks[colaware_chosen]}" if colaware_chosen is not None and colaware_chosen < len(colaware_peaks) else "none"
    colaware_label = f"09 Column Profile\n{len(colaware_peaks)} peaks gap={gap_s} chosen={chosen_s}"
    win_label = f"10 Window dynamic\n{L}x{H} @({Xc},{Yc}) orig{L_orig}x{H_orig}@{params['ex_Xc']},{params['ex_Yc']}"
    panels = [
        (bgr2rgb(bgr_orig), "01 Original\n(full res)", False),
        (bgr2rgb(bgr_resized), f"02 Resized {width}x{height}\n(pipeline input)", False),
        (h_mask, f"03 Hue Mask\nH {params['min_Hue']}-{params['max_Hue']}", True),
        (s_mask, f"04 Sat Mask\nS {params['min_Saturation']}-{params['max_Saturation']}", True),
        (v_mask, f"05 Val Mask\nV {params['min_Value']}-{params['max_Value']}", True),
        (combined, "06 Combined Mask\nH&S&V (173)", True),
        (bgr2rgb(contours_img), f"07 Contours\nfindContours N={n_contours}", False),
        (bgr2rgb(centers_img), f"08 All Centers\nN={n_centers}", False),
        (bgr2rgb(profile_img), colaware_label, False),
        (bgr2rgb(window_img), win_label, False),
        (bgr2rgb(nh_img), f"11 Inside Window\nYELLOW nh={n_nh} | before IF", False),
        (bgr2rgb(iso_img), f"12 {iso_label}\nYELLOW in={n_iso_in} RED out={n_iso_out}", False),
        (bgr2rgb(raw_line_img), "13 Raw fitLine (inliers)\nBLUE infinite", False),
        (bgr2rgb(clipped_line_img), "14 Clipped AvgLine (inliers)\nRED", False),
        (bgr2rgb(final_img), "15 Final Result\nGREEN all + YEL inliers + RED", False),
    ]

    fig, axes = plt.subplots(5, 4, figsize=(28,24), constrained_layout=True)
    axes_flat = axes.flatten()
    for idx, (img, title, is_gray) in enumerate(panels):
        ax = axes_flat[idx]
        if is_gray:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=7, pad=3)
        ax.axis("off")

    # 16 Rejection Log
    ax_log = axes_flat[15]
    ax_log.axis("off")
    log_text = "=== REJECTION LOG ===\n"
    if not rejection_log:
        log_text += "  (none - line found)\n"
    else:
        for line in rejection_log:
            log_text += f"  • {line}\n"
    # add caveats
    log_text += "\n=== CAVEATS (results/README.txt) ===\n"
    log_text += "  • filterContures degenerate (center_min_off=0)\n"
    log_text += "  • FitLine clips strictly 0..W; +2px tol noted\n"
    log_text += "  • ONE line per frame (servoing)\n"
    log_text += "  • bev6: HSV mask zero with defaults\n"
    ax_log.text(0.02,0.98, log_text, transform=ax_log.transAxes, va="top", ha="left",
                fontsize=7, family="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.95, edgecolor="gray", boxstyle="round,pad=0.4"))
    ax_log.set_title("16 Rejection / Notes", fontsize=8)

    # 17 Summary
    ax_sum = axes_flat[16]
    ax_sum.axis("off")
    def fmt(k):
        v = timings.get(k,0)
        return f"{v:.1f}" if isinstance(v,float) else str(v)
    total = sum([timings.get(k,0) for k in ["resize","hsv","contours","centers","neighbourhood","iso","gap","fitline"]])
    gap_str2 = f"{colaware_median_gap:.0f}" if colaware_median_gap is not None else "0"
    summary = (
        f"{stem}\n"
        f"W={width} H={height} Scale={params['Scale']}  Win {L}x{H} @({Xc},{Yc}) dyn gap={gap_str2}\n"
        f"HSV: H{params['min_Hue']}-{params['max_Hue']} S{params['min_Saturation']}-{params['max_Saturation']} V{params['min_Value']}-{params['max_Value']}\n"
        f"Contours: {n_contours} | Centers: {n_centers} | Inside: {n_nh} | IF in:{n_iso_in} out:{n_iso_out} | Gap in:{n_gap_in} out:{n_gap_out} clusters={gap_n_clusters}\n"
        f"IF: {'ON' if iso_applied else 'OFF'} cont={params.get('iso_contamination',0.15)} | Gap: {'ON' if params.get('gap_enabled') else 'OFF'} eps={params.get('gap_eps',14)} | col-aware {'ON' if params.get('colaware_enabled') else 'OFF'}\n"
        f"--- timings (ms) ---\n"
        f"resize:{fmt('resize')} hsv:{fmt('hsv')} contours:{fmt('contours')}\n"
        f"centers:{fmt('centers')} win:{fmt('neighbourhood')} iso:{fmt('iso')} gap:{fmt('gap')} fit:{fmt('fitline')}\n"
        f"TOTAL: {total:.1f} ms  ({1000/total:.1f} FPS)\n"
        f"Line: {'FOUND' if intermediates['has_line'] else 'NONE'} (on gap-kept)\n"
    )
    if intermediates['has_line'] and intermediates['fit_info']:
        info = intermediates['fit_info']
        inside = info['inside']
        summary += f"AvgLine: {inside[0] if len(inside)>0 else ' - '} -> {inside[1] if len(inside)>1 else ' - '}\n"
        vx,vy,x0,y0 = info['line']
        summary += f"fitLine: vx={vx:.3f} vy={vy:.3f} x0={x0:.0f} y0={y0:.0f}\n"
    ax_sum.text(0.02,0.98, summary, transform=ax_sum.transAxes, va="top", ha="left",
                fontsize=7, family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.4"))
    ax_sum.set_title("17 Summary", fontsize=8)
    # hide remaining cells 18-20 (20 cells total, used 0-16)
    for idx in [17,18,19]:
        axes_flat[idx].axis("off")

    fig.suptitle(f"VCRN DEBUG — {stem}  |  ONE line + IF + column-aware window (KMeans removed)  |  Stage-by-stage",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# per-image processing
# ---------------------------------------------------------------------------
def process_image(path: Path, params: dict, out_dir: Path):
    t_all = time.perf_counter()
    timings = {}
    rejection_log = []

    bgr_orig = cv2.imread(str(path))
    if bgr_orig is None:
        raise IOError(f"cannot read {path}")
    stem = path.stem

    # 1 resize to WxH
    t0 = time.perf_counter()
    width, height = params["width"], params["height"]
    bgr_resized = cv2.resize(bgr_orig, (width, height), interpolation=cv2.INTER_AREA)
    timings["resize"] = (time.perf_counter()-t0)*1000

    # 2 HSV split + inRange
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    h_mask = cv2.inRange(h_chan, params["min_Hue"], params["max_Hue"])
    s_mask = cv2.inRange(s_chan, params["min_Saturation"], params["max_Saturation"])
    v_mask = cv2.inRange(v_chan, params["min_Value"], params["max_Value"])
    combined = cv2.bitwise_and(h_mask, cv2.bitwise_and(s_mask, v_mask))
    timings["hsv"] = (time.perf_counter()-t0)*1000
    if int(cv2.countNonZero(combined)) == 0:
        rejection_log.append(f"Combined mask empty (0 px) — H {params['min_Hue']}-{params['max_Hue']} S {params['min_Saturation']}-{params['max_Saturation']} V {params['min_Value']}-{params['max_Value']} (bev6 typical)")

    # 3 findContours
    t0 = time.perf_counter()
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    timings["contours"] = (time.perf_counter()-t0)*1000
    n_contours = len(contours)
    if n_contours==0:
        rejection_log.append("findContours: 0 contours")

    # contours visualization
    contours_img = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    for i,cnt in enumerate(contours):
        col = (int(rng.integers(0,255)), int(rng.integers(0,255)), int(rng.integers(0,255)))
        cv2.drawContours(contours_img, contours, i, col, 1, cv2.LINE_8, hierarchy, 0)

    # 4 getContureCenters
    t0 = time.perf_counter()
    centers = []
    radii = []
    centers_img_base = bgr_resized.copy()
    for cnt in contours:
        # approxPolyDP eps 2 true as in 211
        poly = cv2.approxPolyDP(cnt, 2, True)
        (x,y), rad = cv2.minEnclosingCircle(poly)
        centers.append((float(x), float(y)))
        radii.append(float(rad))
        cv2.circle(centers_img_base, (int(round(x)), int(round(y))), 3, (51,204,51), cv2.FILLED, 8, 0)
        # also radius circle faint?
    n_centers = len(centers)
    # draw count text on centers_img? we'll add in composite title
    centers_img = centers_img_base
    if n_centers==0:
        rejection_log.append("getContureCenters: 0 centers")
    timings["centers"] = (time.perf_counter()-t0)*1000

    # 4b filterContures degenerate - log but not needed, we compute filtered? radius check
    # In C++ 228: radius >= minContourSize (2.0) and x in [cols/2 -off, cols/2+off] off=0 => never true unless x exactly center
    # So we note it
    # For debug we show that filter does nothing: we could compute but not filter
    # Instead we treat all centers as points (as nodehandler does: getContureCenters then filterContures? Actually nodehandler calls getContureCenters -> points, filterContures -> nh_points?? Wait code: imageFrontCallback CropRow_Tracking: src.points = getContureCenters ; src.nh_points = filterContures ; is_in_neigbourhood(src)
    # But is_in_neigbourhood overwrites nh_points based on window, so filterContures result is overwritten? Let's follow nodehandler:51-58
    # Actually CropRow_Tracking: src.points = getContureCenters ; src.nh_points = filterContures ; agribotVS.is_in_neigbourhood(src) which clears nh_points and refills based on points? No it iterates I.points, not I.nh_points? Let's check: is_in_neigbourhood:673 iterates I.points, fills I.nh_points. So filterContures is discarded! So we note.
    # We'll still compute filtered for display but window is the real filter.

    # 5 is_in_neigbourhood — column-aware dynamic spawning
    t0 = time.perf_counter()
    # detect column-aware window if enabled
    if params.get("colaware_enabled", True):
        Xc_dyn, L_dyn, H_dyn, profile_raw, profile_smooth, peak_xs, chosen_idx, median_gap, y0_roi = detect_column_aware_window(combined, centers, params, width, height)
        # log dynamic choice
        gap_str = f"{median_gap:.0f}" if median_gap is not None else "0"
        if peak_xs and chosen_idx is not None:
            chosen_x = peak_xs[chosen_idx]
            rejection_log.append(f"Column-aware window: peak at x={chosen_x} (closest to center) from {len(peak_xs)} peaks median_gap={gap_str} -> Xc={Xc_dyn} L={L_dyn} (orig L={params['nh_L']})")
        elif not peak_xs:
            rejection_log.append(f"Column-aware window: no peaks found -> fallback Xc={Xc_dyn} L={L_dyn}")
            # also fallback median_gap
        Xc, L, H = Xc_dyn, L_dyn, H_dyn
        # store for debug panel
        colaware_profile = profile_smooth
        colaware_peaks = peak_xs
        colaware_chosen = chosen_idx
        colaware_median_gap = median_gap
        colaware_y0 = y0_roi
    else:
        Xc, Yc, L, H = params["ex_Xc"], params["ex_Yc"], params["nh_L"], params["nh_H"]
        colaware_profile = np.zeros(width, dtype=float)
        colaware_peaks = []
        colaware_chosen = None
        colaware_median_gap = None
        colaware_y0 = int(height*0.55)
        # keep Yc from params if not colaware
        Yc = params["ex_Yc"]
    # Ensure Yc is still base (from params), but colaware uses detected Xc/L
    if not params.get("colaware_enabled", True):
        Yc = params["ex_Yc"]
    else:
        Yc = params["ex_Yc"]  # keep base Yc fixed at 380 as tuned
    # Build profile image for debug panel 09
    # Create 100px tall profile plot on black background width=640
    prof_h = 80
    profile_img = np.zeros((prof_h, width, 3), dtype=np.uint8)
    # draw profile as white line normalized to prof_h
    if colaware_profile.max() > 0:
        prof_norm = colaware_profile / colaware_profile.max() * (prof_h-10)
    else:
        prof_norm = colaware_profile
    for x in range(width-1):
        y1 = int(prof_h - 5 - prof_norm[x])
        y2 = int(prof_h - 5 - prof_norm[x+1])
        cv2.line(profile_img, (x,y1), (x+1,y2), (255,255,255), 1)
    # draw peaks: red for all, green for chosen
    for i, px in enumerate(colaware_peaks):
        col = (0,255,0) if i==colaware_chosen else (0,0,255)
        cv2.circle(profile_img, (int(px), int(prof_h -5 - (colaware_profile[px]/colaware_profile.max()* (prof_h-10) if colaware_profile.max()>0 else 0))), 4, col, -1)
        cv2.line(profile_img, (int(px),0), (int(px),prof_h), (50,50,50), 1)
    # draw center line
    cv2.line(profile_img, (width//2,0), (width//2,prof_h), (255,255,0), 1)
    # draw window Xc/L as cyan band
    cv2.rectangle(profile_img, (int(Xc - L/2), 0), (int(Xc + L/2), prof_h), (255,204,102), 1)
    # add text overlay? we will title panel instead

    # window image: orange rectangle + all centers (gray outside, green inside)
    window_img = bgr_resized.copy()
    cv2.rectangle(window_img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 3)
    # draw all centers as small gray dots on window_img as context
    for (x,y) in centers:
        cv2.circle(window_img, (int(round(x)), int(round(y))), 2, (120,120,120), cv2.FILLED)

    # nh_points = those inside window
    nh_points = []
    outside_points = []
    for (x,y) in centers:
        if (Xc - L/2 < x < Xc + L/2) and (Yc - H/2 < y < Yc + H/2):
            nh_points.append((x,y))
        else:
            outside_points.append((x,y))
    n_nh = len(nh_points)
    # nh_img: green inside, gray outside, orange window
    nh_img = bgr_resized.copy()
    cv2.rectangle(nh_img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 3)
    for (x,y) in outside_points:
        cv2.circle(nh_img, (int(round(x)), int(round(y))), 3, (100,100,100), cv2.FILLED)
    for (x,y) in nh_points:
        cv2.circle(nh_img, (int(round(x)), int(round(y))), 5, (0,204,255), cv2.FILLED)
    timings["neighbourhood"] = (time.perf_counter()-t0)*1000
    if n_nh==0:
        rejection_log.append(f"is_in_neigbourhood (col-aware): 0 points inside {L}x{H} window @({Xc},{Yc}) — {n_centers} total outside | peaks={len(colaware_peaks)} chosen={colaware_chosen}")

    # 5b Isolation Forest inside window (anomaly removal before fitLine)
    t0 = time.perf_counter()
    iso_inliers = nh_points
    iso_outliers = []
    iso_applied = False
    n_iso_in = n_nh
    n_iso_out = 0
    iso_img = bgr_resized.copy()
    cv2.rectangle(iso_img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 2)
    # draw outside gray context
    for (x,y) in outside_points:
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 2, (70,70,70), cv2.FILLED)
    if params.get("iso_enabled", True) and n_nh >= params.get("iso_min_points", 12):
        try:
            pts = np.array(nh_points, dtype=np.float32)
            # Use (x,y) 2D - isolation in image plane. No scaling needed, but row is vertical so y spread large.
            iso = IsolationForest(contamination=params.get("iso_contamination", 0.15),
                                  n_estimators=params.get("iso_n_estimators", 100),
                                  random_state=42)
            pred = iso.fit_predict(pts)  # 1 inlier, -1 outlier
            mask_in = pred == 1
            iso_inliers = [tuple(p) for p, m in zip(nh_points, mask_in) if m]
            iso_outliers = [tuple(p) for p, m in zip(nh_points, mask_in) if not m]
            n_iso_in = len(iso_inliers)
            n_iso_out = len(iso_outliers)
            iso_applied = True
            if n_iso_out > 0:
                rejection_log.append(f"IsolationForest inside window: removed {n_iso_out}/{n_nh} outliers ({n_iso_out/n_nh*100:.0f}%) contamination={params.get('iso_contamination',0.15)} -> {n_iso_in} inliers kept for fitLine")
            else:
                rejection_log.append(f"IsolationForest inside window: 0 outliers flagged (contamination {params.get('iso_contamination',0.15)})")
        except Exception as e:
            rejection_log.append(f"IsolationForest failed: {e} -> using all {n_nh} points")
            iso_inliers = nh_points
            iso_outliers = []
            iso_applied = False
            n_iso_in = n_nh
    else:
        if n_nh > 0 and n_nh < params.get("iso_min_points", 12):
            rejection_log.append(f"IsolationForest skipped: n_nh={n_nh} < iso_min_points={params.get('iso_min_points',12)} -> using all points")
        elif not params.get("iso_enabled", True):
            rejection_log.append("IsolationForest disabled -> using all points")
        iso_inliers = nh_points
        iso_outliers = []
        n_iso_in = n_nh
    # visualize iso result: inliers yellow, outliers red
    for (x,y) in iso_inliers:
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 5, (0,204,255), cv2.FILLED)
    for (x,y) in iso_outliers:
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 5, (0,0,255), cv2.FILLED)  # red outliers
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 7, (0,0,255), 1)
    timings["iso"] = (time.perf_counter()-t0)*1000

    # 5c Gap filter removed per user request - stick to Isolation Forest only (KMeans removed)
    # KMeans gap-based multi-row filter disabled; Isolation Forest remains the sole anomaly filter inside window
    gap_kept = iso_inliers
    gap_removed = []
    gap_n_clusters = 1
    gap_info = {"reason": "KMeans removed per user request - stick to IF only", "n_clusters": 1}
    gap_img = iso_img.copy()
    n_gap_in = n_iso_in
    n_gap_out = 0
    timings["gap"] = 0.0
    rejection_log.append("Gap filter KMeans removed per user request - stick to Isolation Forest only")
    final_inliers = gap_kept
    final_outliers = iso_outliers

    # 6 FitLineOnContures (on gap-kept inliers)
    t0 = time.perf_counter()
    fit_info = None
    has_line = False
    raw_line_img = bgr_resized.copy()
    clipped_line_img = bgr_resized.copy()
    final_img = bgr_resized.copy()
    # draw window on raw/clipped/final
    for img in [raw_line_img, clipped_line_img, final_img]:
        cv2.rectangle(img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 2)
    # draw all points: final inliers yellow, gap-removed magenta, iso outliers red, outside gray
    for im in [raw_line_img, clipped_line_img, final_img]:
        for (x,y) in final_inliers:
            cv2.circle(im, (int(round(x)), int(round(y))), 4, (0,204,255), cv2.FILLED)
        for (x,y) in gap_removed:
            cv2.circle(im, (int(round(x)), int(round(y))), 4, (255,0,255), cv2.FILLED)
        for (x,y) in iso_outliers:
            cv2.circle(im, (int(round(x)), int(round(y))), 4, (0,0,255), cv2.FILLED)
        for (x,y) in outside_points:
            cv2.circle(im, (int(round(x)), int(round(y))), 2, (70,70,70), cv2.FILLED)

    if n_gap_in>0:
        fit_info = fit_line_clip(final_inliers, width, height)
        if fit_info is None:
            rejection_log.append("FitLine: no fit info")
        else:
            # draw raw line (extended)
            cv2.line(raw_line_img, (int(round(fit_info["ext_p1"][0])), int(round(fit_info["ext_p1"][1]))),
                     (int(round(fit_info["ext_p2"][0])), int(round(fit_info["ext_p2"][1]))), (255,0,0), 2, cv2.LINE_AA)
            # also draw R points (4 border intersections) as small circles
            for (x,y) in fit_info["R"]:
                if np.isfinite(x):
                    cv2.circle(raw_line_img, (int(round(x)), int(round(y))), 6, (255,0,255), 2)
            # clipped
            inside = fit_info["inside"]
            inside_tol = fit_info["inside_tol"]
            if len(inside)>=2:
                has_line=True
                cv2.line(clipped_line_img, (int(round(inside[0][0])), int(round(inside[0][1]))),
                         (int(round(inside[1][0])), int(round(inside[1][1]))), (0,0,255), 2, cv2.LINE_AA)
                # draw intersections
                for (x,y) in fit_info["R"]:
                    col = (0,255,0) if (x,y) in inside else (0,0,255)
                    if np.isfinite(x):
                        cv2.circle(clipped_line_img, (int(round(x)), int(round(y))), 5, col, 2)
            else:
                rejection_log.append(f"FitLineOnContures: {len(inside)} intersections inside strict [0,W]x[0,H] (need 2) — {len(inside_tol)} with +2px tol — NO AvgLine (results/README.txt:17)")
                # show tolerant line faint
                if len(inside_tol)>=2:
                    cv2.line(clipped_line_img, (int(round(inside_tol[0][0])), int(round(inside_tol[0][1]))),
                             (int(round(inside_tol[1][0])), int(round(inside_tol[1][1]))), (0,0,255), 1, cv2.LINE_AA)
                    for (x,y) in fit_info["R"]:
                        cv2.circle(clipped_line_img, (int(round(x)), int(round(y))), 5, (128,128,128), 1)
            # final
            if has_line:
                cv2.line(final_img, (int(round(inside[0][0])), int(round(inside[0][1]))),
                         (int(round(inside[1][0])), int(round(inside[1][1]))), (0,0,255), 2, cv2.LINE_AA)
            # also draw centers on final as green? Already drawn as orange, add green for all centers?
            # final overlay legend: green dots = all centers, red line, orange rect – keep as is
    else:
        rejection_log.append(f"FitLine: skipped — no inliers (n_nh={n_nh} n_iso={n_iso_in} n_gap={n_gap_in})")
        fit_info = None

    timings["fitline"] = (time.perf_counter()-t0)*1000

    # also final: add green dots for all centers as in node.cpp getContureCenters draws green 3 px – we already have orange for nh, but add small green for all?
    # For final we want green dots for all centers (as per legend green dots = contour centers) + orange nh + red line
    # Redraw green dots on final for all centers on top
    for (x,y) in centers:
        cv2.circle(final_img, (int(round(x)), int(round(y))), 3, (51,204,51), cv2.FILLED)

    intermediates = {
        "bgr_resized": bgr_resized,
        "h_mask": h_mask,
        "s_mask": s_mask,
        "v_mask": v_mask,
        "combined": combined,
        "contours_img": contours_img,
        "centers_img": centers_img,
        "window_img": window_img,
        "nh_img": nh_img,
        "iso_img": iso_img,
        "gap_img": gap_img,
        "profile_img": profile_img,
        "raw_line_img": raw_line_img,
        "clipped_line_img": clipped_line_img,
        "final_img": final_img,
        "rejection_log": rejection_log,
        "timings": timings,
        "n_contours": n_contours,
        "n_centers": n_centers,
        "n_nh": n_nh,
        "n_iso_in": n_iso_in,
        "n_iso_out": n_iso_out,
        "iso_applied": iso_applied,
        "iso_inliers": iso_inliers,
        "iso_outliers": iso_outliers,
        "n_gap_in": n_gap_in,
        "n_gap_out": n_gap_out,
        "gap_n_clusters": gap_n_clusters,
        "gap_info": gap_info,
        "gap_kept": gap_kept,
        "gap_removed": gap_removed,
        "has_line": has_line,
        "fit_info": fit_info,
        "window_Xc": Xc,
        "window_Yc": Yc,
        "window_L": L,
        "window_H": H,
        "colaware_peaks": colaware_peaks,
        "colaware_chosen": colaware_chosen,
        "colaware_median_gap": colaware_median_gap,
        "colaware_profile": colaware_profile,
        "colaware_y0": colaware_y0,
    }
    return intermediates

# ---------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("/home/ac/Crop_Row_Detection_Techniques/Photos"))
    ap.add_argument("--output", type=Path, default=here)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    args = ap.parse_args()

    params = load_params(args.params)
    print(f"Params from {args.params}: {params}")

    images = sorted(args.input.glob(args.pattern))
    if not images:
        print(f"no images matching {args.pattern} in {args.input}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images, 1):
        try:
            inter = process_image(img_path, params, args.output)
            out_path = args.output / f"{img_path.stem}_debug.png"
            save_debug_composite(out_path, cv2.imread(str(img_path)), inter, img_path.stem, params)
            print(f"[{i}/{len(images)}] {img_path.name}: contours={inter['n_contours']} centers={inter['n_centers']} nh={inter['n_nh']} line={'YES' if inter['has_line'] else 'NO'} -> {out_path.name}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{i}/{len(images)}] {img_path.name}: ERROR {e}")

    return 0

if __name__ == "__main__":
    main()
