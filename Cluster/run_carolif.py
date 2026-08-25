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

from carolif import CAROLIF, CarolifConfig


def save_panel(path: Path, title: str, img: np.ndarray, height: int = 420) -> None:
    h, w = img.shape[:2]
    if h > height:
        s = height / h
        img = cv2.resize(img, (int(w * s), height), interpolation=cv2.INTER_AREA)
    put_text(img, title)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"failed to write {path}")


def put_text(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)


def annotate_original(bgr: np.ndarray, lines) -> np.ndarray:
    out = bgr.copy()
    for (p1, p2) in lines:
        cv2.line(out, p1, p2, (0, 0, 255), max(2, out.shape[1] // 400),
                 cv2.LINE_AA)
    return out


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

    save_panel(out_dir / f"{stem}_01_original.png", "Original", bgr)
    save_panel(out_dir / f"{stem}_02_roi.png",
               "Step 1.1 ROI", res["intermediates"]["roi"])
    save_panel(out_dir / f"{stem}_03_binary_mask.png",
               "Step 1.3/1.4 ExG+Otsu+morphology",
               cv2.cvtColor(res["intermediates"]["mask"], cv2.COLOR_GRAY2BGR))
    save_panel(out_dir / f"{stem}_04_top_view.png",
               "Step 1.2 Projective transform (rows parallel)",
               res["intermediates"]["view"])
    save_panel(out_dir / f"{stem}_05_clusters.png",
               "Step 2 HDBSCAN clusters (grey=dropped, black=outliers)",
               res["intermediates"]["clusters"])
    save_panel(out_dir / f"{stem}_06_detection_view.png",
               "Step 3 RANSAC lines in top view",
               res["intermediates"]["detection_view"])
    save_panel(out_dir / f"{stem}_07_result.png",
               f"CAROLIF result - {len(rows)} row(s)", result)

    timings = dict(res["timings"])
    timings["total_pipeline_ms"] = wall_ms
    timings["fps"] = 1000.0 / wall_ms if wall_ms > 0 else float("nan")
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
    ap.add_argument("--output", type=Path, default=here / "results")
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
