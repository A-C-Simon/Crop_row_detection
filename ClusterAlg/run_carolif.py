#!/usr/bin/env python3
"""
Batch runner for CAROLIF crop row detection (Khan et al., 2024).

Runs the detector over every image in an input directory and stores, per
image, all intermediate stages of Algorithm 1 plus the final overlay, along
with per-step processing times (cf. Figures 1 & 10 of the paper) and a
summary CSV.

Usage:
    python3 run_carolif.py [--input DIR] [--output DIR] [--pattern GLOB]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carolif import CAROLIF, CarolifConfig


def annotate_original(bgr: np.ndarray, lines) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    for (p1, p2) in lines:
        try:
            pt1 = (int(float(p1[0])), int(float(p1[1])))
            pt2 = (int(float(p2[0])), int(float(p2[1])))
            # clamp to image bounds
            pt1 = (max(0, min(w-1, pt1[0])), max(0, min(h-1, pt1[1])))
            pt2 = (max(0, min(w-1, pt2[0])), max(0, min(h-1, pt2[1])))
            cv2.line(out, pt1, pt2, (0, 0, 255), max(2, w // 400),
                     cv2.LINE_AA)
        except Exception:
            pass
    return out


def save_composite(path: Path, bgr: np.ndarray, intermediates: dict,
                   result_overlay: np.ndarray, stem: str,
                   n_rows: int, n_clusters_raw: int,
                   timings: dict, vp: tuple, morph: str = "struct") -> None:
    """Save ONE window with all pipeline stages (2x4 grid) instead of 7 separate images.

    Panels: 0 Original, 1 ROI, 2 Binary Mask (preclean-embedded when morph=struct),
            3 Top View, 4 Clusters, 5 Detection View, 6 Final Result, 7 Summary.
    """
    roi = intermediates["roi"]
    mask = intermediates.get("mask_full", intermediates["mask"])  # full-image mask for display
    view = intermediates["view"]
    clusters = intermediates["clusters"]
    det_view = intermediates["detection_view"]

    def bgr2rgb(img):
        if img is None or img.size == 0:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare panel list: (image_array, title, is_gray)
    morph_tag = "STRUCT preclean" if morph == "struct" else "PAPER"
    panels = [
        (bgr2rgb(bgr), "01 Original", False),
        (bgr2rgb(roi), "02 ROI\n(Sec 3.1.1)", False),
        (mask, f"03 Binary Mask\nExG+Otsu+{morph_tag} (Eq.9)", True),
        (bgr2rgb(view), "04 Top View\nProjective (Sec 3.1.1.1)", False),
        (bgr2rgb(clusters), f"05 Clusters\nHDBSCAN raw={n_clusters_raw}", False),
        (bgr2rgb(det_view), "06 Detection View\nRANSAC (Sec 3.1.3)", False),
        (bgr2rgb(result_overlay), f"07 Result\n{n_rows} row(s) detected", False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for idx, (img, title, is_gray) in enumerate(panels):
        ax = axes[idx]
        if is_gray:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis("off")

    # Summary text panel (last axes[7])
    ax = axes[7]
    ax.axis("off")
    # Build summary text
    vp_str = f"({vp[0]:.1f}, {vp[1]:.1f})" if vp else "N/A"
    # timings already rounded; ensure keys exist
    def fmt(k):
        v = timings.get(k, 0)
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    txt = (
        f"{stem}\n"
        f"VP: {vp_str}  |  Rows: {n_rows}  |  Clusters raw: {n_clusters_raw}\n"
        f"Morph: {morph} ({'K+preclean' if morph=='struct' else 'ellipse open+close'})\n"
        f"--- timings (ms) ---\n"
        f"resize: {fmt('resize')}  roi: {fmt('roi')}  segment: {fmt('segment')}\n"
        f"vp_calib: {fmt('vp_calibration')}  transform: {fmt('transform')}\n"
        f"cluster: {fmt('cluster')}  fit_lines: {fmt('fit_lines')}\n"
        f"TOTAL: {fmt('total_pipeline_ms')} ms  ({fmt('fps')} FPS)\n"
        f"--- pipeline ---\n"
        f"ROI → ExG/Otsu+{morph_tag} → Homography → HDBSCAN\n"
        f"→ Spline curve → intersect-vote → overlay"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.4"))
    ax.set_title("08 Summary", fontsize=9)

    fig.suptitle(        f"CAROLIF — {stem}  |  Center-2 flanking + spline [{morph}] — one window",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Debug composite: every decision point in one 4×4 window
# --------------------------------------------------------------------------
def _render_cluster_panel(mask_t, points, labels, cell=2, highlight_ids=None):
    """Render a cluster visualization: colored by cluster ID.
    highlight_ids: if given, only these clusters get bright colors;
    others are gray. If None, all non-noise clusters are bright."""
    vis = cv2.cvtColor(mask_t, cv2.COLOR_GRAY2BGR)
    if labels.size == 0:
        return vis
    rng = np.random.default_rng(0)
    half = max(1, cell // 2)
    highlight = set(highlight_ids) if highlight_ids is not None else None
    for c in np.unique(labels):
        sel = points[labels == c].astype(int)
        if c < 0:
            col = (0, 0, 0)
        elif highlight is None or c in highlight:
            col = (int(rng.integers(60, 255)), int(rng.integers(60, 255)),
                   int(rng.integers(60, 255)))
        else:
            col = (100, 100, 100)
        for x, y in sel:
            x0, y0 = max(0, x - half), max(0, y - half)
            vis[y0:y0 + cell, x0:x0 + cell] = col
    return vis


def _render_center2_panel(mask_t, all_pts, all_labels, cell,
                          selected_ids, forfeited_ids):
    """Render center-2 selection: bright for selected, orange for forfeited."""
    vis = cv2.cvtColor(mask_t, cv2.COLOR_GRAY2BGR)
    if all_labels.size == 0:
        return vis
    sel_set = set(selected_ids) if selected_ids else set()
    for_set = set(forfeited_ids) if forfeited_ids else set()
    half = max(1, cell // 2)
    for c in np.unique(all_labels):
        sel = all_pts[all_labels == c].astype(int)
        if c < 0:
            col = (0, 0, 0)
        elif c in sel_set:
            col = (0, 255, 0)      # bright green — selected center-2
        elif c in for_set:
            col = (0, 165, 255)    # orange — forfeited
        else:
            col = (80, 80, 80)     # dark gray — geometry filtered
        for x, y in sel:
            x0, y0 = max(0, x - half), max(0, y - half)
            vis[y0:y0 + cell, x0:x0 + cell] = col
    # Draw column-projection center line
    vh, vw = mask_t.shape[:2]
    cx = vw // 2
    cv2.line(vis, (cx, 0), (cx, vh), (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def _render_ransac_panel(mask_t, ransac_all, w, h):
    """Draw all spline fits: green = kept, red = rejected by vote/divergence."""
    vis = cv2.cvtColor(mask_t, cv2.COLOR_GRAY2BGR)
    for item in ransac_all:
        try:
            p1, p2, cid, ratio, ang, accepted = item
            pt1 = (int(float(p1[0])), int(float(p1[1])))
            pt2 = (int(float(p2[0])), int(float(p2[1])))
            color = (0, 200, 0) if accepted else (0, 0, 220)
            thick = 2 if accepted else 1
            cv2.line(vis, pt1, pt2, color, thick, cv2.LINE_AA)
            if accepted:
                mx = int((pt1[0] + pt2[0]) / 2)
                my = int((pt1[1] + pt2[1]) / 2)
                cv2.putText(vis, f"{ang:.0f}d", (mx+4, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1)
        except Exception:
            pass
    return vis


def save_debug_composite(path: Path, bgr: np.ndarray, intermediates: dict,
                         result_overlay: np.ndarray, stem: str,
                         n_rows: int, n_clusters_raw: int,
                         timings: dict, vp: tuple, morph: str = "struct") -> None:
    """4×4 debug grid showing every decision point with before/after + reasons."""
    roi = intermediates["roi"]
    mask_full = intermediates.get("mask_full", intermediates["mask"])
    view = intermediates["view"]
    mask_t = intermediates.get("mask_t", intermediates["mask"])

    def bgr2rgb(img):
        if img is None or img.size == 0:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- build visualizations for each panel ---
    # Panel 03: ExG + Otsu (before morphology)
    p_otsu = intermediates.get("mask_otsu", np.zeros_like(mask_t))
    # Panel 04: After KERNEL_K (before preclean)
    p_opened = intermediates.get("mask_opened", np.zeros_like(mask_t))
    # Panel 05: After preclean (final mask)
    p_preclean = intermediates.get("mask_preclean", np.zeros_like(mask_t))
    # Panel 07: HDBSCAN raw clusters
    raw_labels = intermediates.get("raw_labels", np.zeros(0, dtype=int))
    raw_points = intermediates.get("raw_points", np.zeros((0,2), np.float32))
    cell = intermediates.get("cell", 2)
    p_raw_clusters = _render_cluster_panel(mask_t, raw_points, raw_labels, cell)
    # Panel 08: After refinement
    ref_labels = intermediates.get("refined_labels", np.zeros(0, dtype=int))
    ref_points = intermediates.get("refined_points", np.zeros((0,2), np.float32))
    p_ref_clusters = _render_cluster_panel(mask_t, ref_points, ref_labels, cell)
    # Panel 09: Geometry before (all clusters passing min_points/height)
    geo_before = intermediates.get("geo_before", {})
    kept_ids = list(geo_before.keys())
    all_pts = intermediates.get("points", np.zeros((0,2), np.float32))
    all_labels = intermediates.get("labels", np.zeros(0, dtype=int))
    p_geo_before = _render_cluster_panel(mask_t, all_pts, all_labels, cell,
                                          highlight_ids=kept_ids)
    # Panel 10: Geometry after (kept clusters)
    geo_after = intermediates.get("geo_after", {})
    p_geo_after = _render_cluster_panel(mask_t, all_pts, all_labels, cell,
                                         highlight_ids=list(geo_after.keys()))
    # Panel 11: Center-2 selection (Zhou flanking style)
    forfeited = intermediates.get("forfeited_clusters", {})
    kept_clusters_dbg = intermediates.get("kept_clusters", [])
    p_center2 = _render_center2_panel(mask_t, all_pts, all_labels, cell,
                                       kept_clusters_dbg, list(forfeited.keys()))
    # Panel 12: RANSAC all lines
    ransac_all = intermediates.get("ransac_all", [])
    p_ransac = _render_ransac_panel(mask_t, ransac_all,
                                    mask_t.shape[1], mask_t.shape[0])

    # --- assemble panels ---
    morph_tag = "struct" if morph == "struct" else "paper"
    n_forfeited = len(forfeited)
    panels = [
        (bgr2rgb(bgr), "01 Original", False),
        (bgr2rgb(roi), "02 ROI", False),
        (p_otsu, "03 After Otsu\n(before morphology)", True),
        (p_opened, "04 After KERNEL_K\n(before preclean)", True),
        (p_preclean, "05 After Preclean\n(final mask)", True),
        (bgr2rgb(view), "06 Top View\n(homography)", False),
        (bgr2rgb(p_raw_clusters), f"07 HDBSCAN Raw\n{n_clusters_raw} clusters found", False),
        (bgr2rgb(p_ref_clusters), "08 After Refinement\n+ vertical merge", False),
        (bgr2rgb(p_geo_before), f"09 Geo Filter Before\n{len(geo_before)} pass size/height", False),
        (bgr2rgb(p_geo_after), f"10 Geo Filter After\n{len(geo_after)} kept", False),
        (bgr2rgb(p_center2), f"11 Center-2 Select\n{2} kept, {n_forfeited} forfeited", False),
        (bgr2rgb(p_ransac), "12 Spline Fits\nGREEN=kept RED=rejected", False),
        (bgr2rgb(result_overlay), f"13 Final Result\n{n_rows} row(s)", False),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(28, 20), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (img, title, is_gray) in enumerate(panels):
        ax = axes_flat[idx]
        if is_gray:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    # Panel 14: Rejection log
    ax_log = axes_flat[13]
    ax_log.axis("off")
    rej = intermediates.get("rejection_log", [])
    ransac_rej = intermediates.get("ransac_reject", [])
    log_lines = ["=== CLUSTER REJECTIONS ==="]
    for cid, reason in rej:
        log_lines.append(f"  cluster {cid}: {reason}")
    if not rej:
        log_lines.append("  (none)")
    log_lines.append("")
    log_lines.append("=== SPLINE / VOTE REJECTIONS ===")
    for cid, reason in ransac_rej:
        log_lines.append(f"  cluster {cid}: {reason}")
    if not ransac_rej:
        log_lines.append("  (none)")
    log_text = "\n".join(log_lines)
    ax_log.text(0.02, 0.98, log_text, transform=ax_log.transAxes,
                va="top", ha="left", fontsize=8, family="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.95, edgecolor="gray",
                          boxstyle="round,pad=0.4"))
    ax_log.set_title("14 Rejection Log", fontsize=9)

    # Panel 15: Summary
    ax_sum = axes_flat[14]
    ax_sum.axis("off")
    vp_str = f"({vp[0]:.1f}, {vp[1]:.1f})" if vp else "N/A"
    def fmt(k):
        v = timings.get(k, 0)
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    summary = (
        f"{stem}\n"
        f"VP: {vp_str}  |  Rows: {n_rows}  |  Raw clusters: {n_clusters_raw}\n"
        f"Morph: {morph_tag}\n"
        f"--- timings (ms) ---\n"
        f"resize: {fmt('resize')}  roi: {fmt('roi')}  segment: {fmt('segment')}\n"
        f"vp_calib: {fmt('vp_calibration')}  transform: {fmt('transform')}\n"
        f"cluster: {fmt('cluster')}  fit_lines: {fmt('fit_lines')}\n"
        f"TOTAL: {fmt('total_pipeline_ms')} ms  ({fmt('fps')} FPS)"
    )
    ax_sum.text(0.02, 0.98, summary, transform=ax_sum.transAxes,
                va="top", ha="left", fontsize=8.5, family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                          boxstyle="round,pad=0.4"))
    ax_sum.set_title("15 Summary", fontsize=9)

    # hide remaining axis
    for idx in [15]:
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"CAROLIF DEBUG — {stem}  |  Center-2 flanking + every decision point  [{morph}]",
        fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def process_image(detector: CAROLIF, path: Path, out_dir: Path, debug: bool = False) -> dict:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise IOError(f"cannot read {path}")

    t0 = time.perf_counter()
    res = detector.detect(bgr, debug=debug)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    stem = path.stem
    rows = res["lines"]
    result = annotate_original(bgr, rows)

    timings = dict(res["timings"])
    timings["total_pipeline_ms"] = wall_ms
    timings["fps"] = 1000.0 / wall_ms if wall_ms > 0 else float("nan")

    vp = res["intermediates"]["vp"]
    rounded_timings = {k: round(v, 2) for k, v in timings.items()}

    # Standard composite (always generated)
    save_composite(out_dir / f"{stem}_result.png",
                   bgr, res["intermediates"], result,
                   stem, len(rows), res["n_clusters_raw"],
                   rounded_timings, vp, morph=detector.cfg.morph)

    # Debug composite (only when --debug is passed)
    if debug:
        save_debug_composite(out_dir / f"{stem}_debug.png",
                             bgr, res["intermediates"], result,
                             stem, len(rows), res["n_clusters_raw"],
                             rounded_timings, vp, morph=detector.cfg.morph)

    # also keep timings with raw values for CSV
    return {
        "image": path.name,
        "n_rows": len(rows),
        "n_clusters_raw": res["n_clusters_raw"],
        "vanishing_point": tuple(round(v, 1) for v in
                                 res["intermediates"]["vp"]),
        **{k: round(v, 2) for k, v in timings.items()},
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path(
        "/home/ac/Crop_Row_Detection_Techniques/Photos"))
    ap.add_argument("--output", type=Path, default=here / "result")
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--morph", choices=("struct", "paper"), default="struct",
                    help="Morphology: 'struct' = preclean-embedded DEFAULT (K + band-support filter, opening-only, validated on photo_9/6), 'paper' = original ellipse open+close")
    ap.add_argument("--debug", action="store_true",
                    help="Generate a 4x4 debug composite per image showing every decision point (before/after + rejection reasons)")
    args = ap.parse_args()

    images = sorted(args.input.glob(args.pattern))
    if not images:
        print(f"no images matching {args.pattern} in {args.input}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = CarolifConfig(morph=args.morph)
    detector = CAROLIF(cfg)
    fields = ["image", "n_rows", "n_clusters_raw", "vanishing_point",
              "resize", "roi", "segment", "transform", "cluster",
              "fit_lines", "total_pipeline_ms", "fps"]
    records = []
    for i, img_path in enumerate(images, 1):
        try:
            rec = process_image(detector, img_path, args.output, debug=args.debug)
        except Exception as exc:                       # keep batch going
            rec = {"image": img_path.name, "error": str(exc)}
            print(f"[{i}/{len(images)}] {img_path.name}: ERROR {exc}")
        else:
            print(f"[{i}/{len(images)}] {img_path.name}: "
                  f"{rec['n_rows']} rows | "
                  f"{rec['total_pipeline_ms']:.0f} ms "
                  f"({rec['fps']:.1f} FPS)")
        records.append(rec)

    csv_path = args.output / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    ok = [r for r in records if "error" not in r]
    if ok:
        tot = [r["total_pipeline_ms"] for r in ok]
        print(f"\n{len(ok)}/{len(records)} images processed | "
              f"avg {np.mean(tot):.0f} ms/img | "
              f"{1000.0 / np.mean(tot):.1f} FPS | rows detected total: "
              f"{sum(r['n_rows'] for r in ok)}")
    print(f"results saved to: {args.output}\nsummary: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
