"""Batch-process a folder of real images through the DFT row detector.

Bird's-eye-view files are detected directly; forward-looking photos are
rectified with an automatic pitch scan first (see run_real_image.py).

Writes into the output directory (default ./results):
  * <stem>_rows.png   overlay per image (input + traced rows + centerline)
  * metrics.csv       one row per image with spacing/angle/deviations
  * summary.md        human-readable table + parameter notes

Usage:
    python3 run_batch.py [PHOTO_DIR] [--out DIR] [--height M]
                         [--scan-pitch LO:HI:STEP] [--no-trace]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time

import numpy as np

from run_real_image import load_gray, rectify_forward, pitch_score, overlay
from row_detection import RowDetector, trace_row_curves
from preprocessing import roi_to_map_transform


def detect_image(path: str, bev: bool, args) -> dict:
    """Run the full pipeline on one image; returns a metrics dict."""
    t0 = time.time()
    img = load_gray(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    if bev:
        scale = 26.0 / max(img.shape)
        T = roi_to_map_transform(scale, img.shape[0] / 2, img.shape[1] / 2)
        mode = f"BEV ({scale*100:.2f} cm/px assumed)"
    else:
        lo, hi, step = map(float, args.scan_pitch.split(":"))
        best = None
        for p in np.arange(lo, hi + 1e-9, step):
            try:
                r, _, _ = rectify_forward(img, p, args.height, args.fov,
                                          None, 0.0, args.range_m)
                d_p = RowDetector(spacing_tolerance=args.tolerance)
                res_p = d_p.detect(r)
                score, _ = pitch_score(r, d_p, res_p,
                                       args.verticality_sigma)
            except Exception:
                continue
            if best is None or score > best[0]:
                best = (score, p)
        if best is None:
            raise RuntimeError("pitch scan found no usable rectification")
        pitch = float(best[1])
        img, (r0, c0), gsd = rectify_forward(img, pitch, args.height,
                                             args.fov, None, 0.0,
                                             args.range_m)
        scale = gsd
        T = roi_to_map_transform(gsd, img.shape[0] / 2 + r0,
                                 img.shape[1] / 2 + c0)
        mode = (f"forward rectified @ {pitch:.0f} deg pitch, "
                f"h={args.height} m, gsd={gsd*100:.1f} cm/px")

    det = RowDetector(spacing_tolerance=args.tolerance)
    res = det.detect(img, T)

    traces = None
    n_pts = 0
    if not args.no_trace:
        traces = trace_row_curves(img, res)
        if traces:
            n_pts = int(sum(np.isfinite(p[:, 0]).sum() for p in traces))

    prom = det._peak_prominence(img, res)
    conf = pitch_confidence_local(img, det, res)
    out_png = os.path.join(args.out, stem + "_rows.png")
    overlay(det, res, img, scale, out_png,
            "detected rows + centerline (cyan)", traces)

    return dict(
        file=os.path.basename(path), mode=mode,
        spacing_px=round(res.row_spacing_px, 2),
        spacing_cm=round(res.row_spacing_px * scale * 100, 1),
        angle_deg=round(res.row_angle_deg, 2),
        e_y_cm=round(res.e_y * scale * 100, 1),
        e_theta_deg=round(math.degrees(res.e_theta), 2),
        prominence=round(prom, 1),
        polylines=len(traces) if traces else 0,
        trace_points=n_pts,
        seconds=round(time.time() - t0, 1),
        warning=("weak lock" if prom < 3.0 else
                 ("texture-scale lock" if res.row_spacing_px < 8 else "")),
    )


def pitch_confidence_local(rect, det, res):
    from run_real_image import pitch_confidence
    return pitch_confidence(rect, det, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photos", nargs="?", default="/home/ac/Crop_Row_"
                    "Detection_Techniques/Photos")
    ap.add_argument("--out", default=None,
                    help="output directory (default: ./results)")
    ap.add_argument("--height", type=float, default=0.8,
                    help="forward mode camera height in m")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--range", type=float, default=10.0, dest="range_m")
    ap.add_argument("--scan-pitch", default="25:55:5")
    ap.add_argument("--verticality-sigma", type=float, default=12.0)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--bev-pattern", default=r"(?i)bev",
                    help="filenames matching this regex are treated as "
                         "bird's-eye views")
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()

    args.out = args.out or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "results")
    os.makedirs(args.out, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = sorted(f for f in os.listdir(args.photos)
                   if f.lower().endswith(exts))
    if not files:
        raise SystemExit(f"no images found in {args.photos}")

    bev_re = re.compile(args.bev_pattern)
    rows, failures = [], []
    for i, f in enumerate(files, 1):
        bev = bool(bev_re.search(f))
        print(f"\n[{i}/{len(files)}] {f}  ({'BEV' if bev else 'forward'})",
              flush=True)
        try:
            rows.append(detect_image(os.path.join(args.photos, f),
                                     bev, args))
            r = rows[-1]
            print(f"  spacing {r['spacing_px']} px ({r['spacing_cm']} cm), "
                  f"angle {r['angle_deg']} deg, "
                  f"prominence {r['prominence']}x {r['warning']}".rstrip())
        except Exception as e:
            failures.append((f, repr(e)))
            print(f"  FAILED: {e!r}")

    # ---- CSV ---------------------------------------------------------------
    csv_path = os.path.join(args.out, "metrics.csv")
    fields = ["file", "mode", "spacing_px", "spacing_cm", "angle_deg",
              "e_y_cm", "e_theta_deg", "prominence", "polylines",
              "trace_points", "seconds", "warning"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- Markdown summary ---------------------------------------------------
    md_path = os.path.join(args.out, "summary.md")
    with open(md_path, "w") as fh:
        fh.write("# DFT crop-row detection - batch results\n\n")
        fh.write(f"Source: `{args.photos}` ({len(files)} images)\n\n")
        fh.write("Forward-looking photos were rectified with an automatic "
                 f"pitch scan {args.scan_pitch} deg (yaw=0 verticality prior, "
                 f"sigma={args.verticality_sigma} deg); BEV files were "
                 "detected directly.\n\n")
        fh.write("| file | mode | spacing px | spacing cm | angle deg | "
                 "e_y cm | e_theta deg | prominence | polylines | warning |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['file']} | {r['mode']} | {r['spacing_px']} | "
                     f"{r['spacing_cm']} | {r['angle_deg']} | {r['e_y_cm']} | "
                     f"{r['e_theta_deg']} | {r['prominence']}x | "
                     f"{r['polylines']} | {r['warning']} |\n")
        if failures:
            fh.write("\n## Failures\n\n")
            for f, err in failures:
                fh.write(f"* `{f}`: {err}\n")

    print(f"\n{len(rows)} detected, {len(failures)} failed.")
    print(f"results written to {args.out} "
          f"(metrics.csv, summary.md, *_rows.png)")


if __name__ == "__main__":
    main()
