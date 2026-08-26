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
    for (p1, p2) in lines:
        cv2.line(out, p1, p2, (0, 0, 255), max(2, out.shape[1] // 400),
                 cv2.LINE_AA)
    return out


def save_composite(path: Path, bgr: np.ndarray, intermediates: dict,
                   result_overlay: np.ndarray, stem: str,
                   n_rows: int, n_clusters_raw: int,
                   timings: dict, vp: tuple) -> None:
    """Save ONE window with all pipeline stages (2x4 grid) instead of 7 separate images.

    Panels: 0 Original, 1 ROI, 2 Binary Mask, 3 Top View, 4 Clusters,
            5 Detection View, 6 Final Result, 7 Summary text.
    """
    roi = intermediates["roi"]
    mask = intermediates["mask"]
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
    panels = [
        (bgr2rgb(bgr), "01 Original", False),
        (bgr2rgb(roi), "02 ROI\n(Sec 3.1.1)", False),
        (mask, "03 Binary Mask\nExG+Otsu+Morph (Eq.9)", True),
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
        f"--- timings (ms) ---\n"
        f"resize: {fmt('resize')}  roi: {fmt('roi')}  segment: {fmt('segment')}\n"
        f"vp_calib: {fmt('vp_calibration')}  transform: {fmt('transform')}\n"
        f"cluster: {fmt('cluster')}  fit_lines: {fmt('fit_lines')}\n"
        f"TOTAL: {fmt('total_pipeline_ms')} ms  ({fmt('fps')} FPS)\n"
        f"--- pipeline ---\n"
        f"ROI → ExG/Otsu → Homography → HDBSCAN\n"
        f"→ RANSAC → slope [70,110]° → overlay"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.4"))
    ax.set_title("08 Summary", fontsize=9)

    fig.suptitle(f"CAROLIF — {stem}  |  Khan et al. 2024 (Alg.1)  —  all stages in one window",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_image(detector: CAROLIF, path: Path, out_dir: Path) -> dict:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise IOError(f"cannot read {path}")

    t0 = time.perf_counter()
    res = detector.detect(bgr)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    stem = path.stem
    rows = res["lines"]
    result = annotate_original(bgr, rows)

    timings = dict(res["timings"])
    timings["total_pipeline_ms"] = wall_ms
    timings["fps"] = 1000.0 / wall_ms if wall_ms > 0 else float("nan")

    # ONE composite window per image (user request: don't split into 6 separate files)
    vp = res["intermediates"]["vp"]
    rounded_timings = {k: round(v, 2) for k, v in timings.items()}
    save_composite(out_dir / f"{stem}_result.png",
                   bgr, res["intermediates"], result,
                   stem, len(rows), res["n_clusters_raw"],
                   rounded_timings, vp)

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
    args = ap.parse_args()

    images = sorted(args.input.glob(args.pattern))
    if not images:
        print(f"no images matching {args.pattern} in {args.input}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = CarolifConfig()
    detector = CAROLIF(cfg)
    fields = ["image", "n_rows", "n_clusters_raw", "vanishing_point",
              "resize", "roi", "segment", "transform", "cluster",
              "fit_lines", "total_pipeline_ms", "fps"]
    records = []
    for i, img_path in enumerate(images, 1):
        try:
            rec = process_image(detector, img_path, args.output)
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
