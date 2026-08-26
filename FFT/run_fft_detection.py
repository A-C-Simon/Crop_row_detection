"""Batch test of the paper's DFT crop row detector (fft.pdf) on Photos/.

Forward-looking images are rectified to a metric bird's-eye ROI with a
pitch-scanning inverse perspective mapping; images named 'bev*' are treated
as already top-down. Every image gets a 4-panel figure (input, rectified ROI
with detected rows + navigation centerline, bandpassed log spectrum with the
locked peak, metric summary) and a row in results_fft/metrics.csv.

Usage:
    python3 run_fft_detection.py [--photos DIR] [--out DIR]
        [--height M] [--fov DEG] [--scan LO:HI:STEP] [--range M]
        [--spacing-prior-m M] [--bev-width-m M] [--gsd M]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dft_crop_row_detector import (DFTRowDetector, Detection, exg_gray,
                                   rectify_forward)

METRIC_FIELDS = ["image", "mode", "pitch_deg", "yaw_deg", "gsd_m_per_px",
                 "roi_w", "roi_h", "row_spacing_px", "row_spacing_m",
                 "angle_dev_deg", "fx_cyc_per_px", "fy_cyc_per_px", "phi_rad",
                 "n_rows", "ey_cm", "e_theta_deg", "corridor_m", "prominence",
                 "detect_ms", "status"]


def spacing_band_px(gsd: float | None, args) -> tuple[float, float, float | None]:
    """(min_period, max_period, prior) in pixels for the bandpass filter."""
    if gsd:
        lo = max(args.min_spacing_m / gsd, 6.0)
        hi = args.max_spacing_m / gsd
        prior = args.spacing_prior_m / gsd if args.spacing_prior_m else None
    else:
        lo, hi, prior = 10.0, None, None
    return lo, hi, prior


def detect_roi(roi: np.ndarray, gsd: float | None, ref_xy, args):
    lo, hi, prior = spacing_band_px(gsd, args)
    det = DFTRowDetector(min_period_px=lo, max_period_px=hi,
                         spacing_prior_px=prior)
    t0 = time.perf_counter()
    res = det.detect(roi, ref_xy=ref_xy)
    ms = (time.perf_counter() - t0) * 1000.0
    return det, res, ms


def pitch_scan_score(res: Detection, sigma_deg: float) -> float:
    dev = abs(res.angle_from_vertical_deg)
    weight = float(np.exp(-0.5 * (dev / sigma_deg) ** 2))
    return res.prominence * weight


def process_bev(rgb: np.ndarray, args):
    gray = exg_gray(rgb)
    h, w = gray.shape
    gsd = args.bev_width_m / w
    roi = gray
    _, res, ms = detect_roi(roi, gsd, (w / 2.0, h - 1.0), args)
    info = {"mode": "BEV", "pitch": None, "yaw": None, "gsd": gsd}
    return rgb, roi, res, ms, info


def process_forward(rgb: np.ndarray, args):
    gray = exg_gray(rgb)
    cands = []
    yaws = ([0.0] if args.yaw_scan is None else
            list(np.arange(args.yaw_scan[0], args.yaw_scan[1] + 1e-9,
                           args.yaw_scan[2])))
    for pitch in np.arange(args.scan[0], args.scan[1] + 1e-9, args.scan[2]):
        for yaw in yaws:
            try:
                rect, r_gsd = rectify_forward(gray, float(pitch), args.height,
                                              args.fov, args.gsd,
                                              yaw_deg=float(yaw),
                                              range_m=args.range_m)
                lo, hi, prior = spacing_band_px(r_gsd, args)
                d = DFTRowDetector(min_period_px=lo, max_period_px=hi,
                                   spacing_prior_px=prior)
                r = d.detect(rect, ref_xy=(rect.shape[1] / 2.0,
                                           rect.shape[0] - 1.0))
                if r.n_rows < 3 or r.prominence < 15.0:
                    continue
                score = pitch_scan_score(r, args.verticality_sigma)
            except Exception:
                continue
            print(f"    pitch {pitch:5.1f} yaw {yaw:6.1f}: "
                  f"spacing {r.spacing_px * r_gsd * 100:6.1f} cm, "
                  f"dev {r.angle_from_vertical_deg:7.2f} deg, "
                  f"prom {r.prominence:7.1f}, score {score:9.1f}")
            cands.append((score, r.prominence, float(pitch), float(yaw)))
    if not cands:
        raise RuntimeError("no usable pitch/yaw in scan range")
    best_vert = max(cands, key=lambda c: c[0])
    max_prom = max(c[1] for c in cands)
    if best_vert[0] < 8.0:
        chosen = max(cands, key=lambda c: c[1])
        print(f"    no near-vertical lock; falling back to dominant "
              f"pattern (prominence {chosen[1]:.1f}x)")
    else:
        chosen = best_vert
    _, _, pitch, yaw = chosen
    roi, gsd = rectify_forward(gray, pitch, args.height, args.fov, args.gsd,
                               yaw_deg=yaw, range_m=args.range_m)
    _, res, ms = detect_roi(roi, gsd, (roi.shape[1] / 2.0,
                                       roi.shape[0] - 1.0), args)
    info = {"mode": "FORWARD", "pitch": pitch, "yaw": yaw, "gsd": gsd}
    return rgb, roi, res, ms, info


def draw_spectrum(ax, res: Detection):
    mag = res.magnitude
    fx_ax, fy_ax = res.freq_x, res.freq_y
    span = res.band_hi * 1.35
    ax.imshow(np.log1p(mag), cmap="magma",
              extent=[fy_ax[0], fy_ax[-1], fx_ax[-1], fx_ax[0]],
              aspect="auto")
    th = np.linspace(0, 2 * np.pi, 256)
    for r, ls in ((res.band_lo, "--"), (res.band_hi, "--")):
        ax.plot(r * np.cos(th), r * np.sin(th), "w:", lw=0.8, alpha=0.8)
    for sgn in (1, -1):
        ax.plot(sgn * res.fy, sgn * res.fx, "c+", ms=14, mew=2)
    ax.set_xlim(-span, span)
    ax.set_ylim(span, -span)
    ax.set_xlabel("f_y (cyc/px)")
    ax.set_ylabel("f_x (cyc/px)")
    ax.set_title(f"bandpassed log spectrum, peak prominence "
                 f"{res.prominence:.1f}x")


def draw_roi(ax, roi: np.ndarray, res: Detection, title: str):
    ax.imshow(roi, cmap="gray")
    for x0, y0, x1, y1 in res.line_endpoints():
        ax.plot([x0, x1], [y0, y1], "r-", lw=1.4, alpha=0.85, zorder=3)
    cl = res.centerline()
    if cl:
        ax.plot([cl[0], cl[2]], [cl[1], cl[3]], "c-", lw=2.2, zorder=4)
    ax.plot(*res.ref_point, marker="*", color="yellow", ms=14,
            mec="k", zorder=5)
    h, w = roi.shape
    ax.set_xlim(-w * 0.03, w * 1.03)
    ax.set_ylim(h * 1.03, -h * 0.03)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def summary_text(name, info, res: Detection, ms: float, gsd: float | None):
    lines = [
        name,
        f"mode           : {info['mode']}"
        + (f", pitch {info['pitch']:.1f} deg" if info["pitch"] else "")
        + (f", yaw {info['yaw']:.1f} deg" if info.get("yaw") else ""),
        f"ROI            : {res.magnitude.shape[1]} x "
        f"{res.magnitude.shape[0]} px"
        + (f", gsd {gsd * 100:.2f} cm/px" if gsd else ""),
        f"peak (fx, fy)  : ({res.fx:.5f}, {res.fy:.5f}) cyc/px",
        f"periods (Tx,Ty): ({res.Tx:.1f}, {res.Ty:.1f}) px",
        f"phase phi      : {res.phi:.3f} rad",
        f"row spacing    : {res.spacing_px:.1f} px"
        + (f" = {res.spacing_px * gsd * 100:.1f} cm" if gsd else ""),
        f"rows detected  : {res.n_rows}",
        f"heading dev e_t: {res.e_theta_deg:.2f} deg",
        f"lateral dev e_y: "
        + (f"{res.ey_px * gsd * 100:.1f} cm" if gsd and np.isfinite(res.ey_px)
           else f"{res.ey_px:.1f} px"),
        f"corridor width : "
        + (f"{res.corridor_px * gsd:.2f} m" if gsd and np.isfinite(res.corridor_px)
           else f"{res.corridor_px:.1f} px"),
        f"prominence     : {res.prominence:.1f}x",
        f"detect time    : {ms:.0f} ms",
    ]
    return "\n".join(lines)


def save_figure(path, name, rgb, roi, res, ms, info, status="OK"):
    gsd = info.get("gsd")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{name}  [{info['mode']}]", fontsize=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    title = (f"rectified BEV ROI: {res.n_rows} rows, "
             f"spacing "
             + (f"{res.spacing_px * gsd * 100:.0f} cm, " if gsd else "")
             + f"e_theta {res.e_theta_deg:.2f} deg")
    draw_roi(axes[0, 1], roi, res, title)
    draw_spectrum(axes[1, 0], res)
    axes[1, 1].axis("off")
    txt = summary_text(name, info, res, ms, gsd)
    if status != "OK":
        txt += f"\nSTATUS: {status}"
    axes[1, 1].text(0.02, 0.98, txt, va="top", family="monospace",
                    fontsize=10)
    axes[1, 1].set_title("DFT crop-row detection summary", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", default="/home/ac/Crop_Row_Detection_Techniques/Photos")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "results_fft"))
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--scan", default="20:60:5")
    ap.add_argument("--yaw-scan", default="-30:30:10", dest="yaw_scan")
    ap.add_argument("--range", type=float, default=10.0, dest="range_m")
    ap.add_argument("--gsd", type=float, default=None)
    ap.add_argument("--min-spacing-m", type=float, default=0.15)
    ap.add_argument("--max-spacing-m", type=float, default=2.0)
    ap.add_argument("--spacing-prior-m", type=float, default=None)
    ap.add_argument("--verticality-sigma", type=float, default=12.0)
    ap.add_argument("--bev-width-m", type=float, default=26.0)
    args = ap.parse_args(argv)
    args.scan = tuple(float(x) for x in args.scan.split(":"))
    args.yaw_scan = (None if str(args.yaw_scan).lower() in ("none", "")
                     else tuple(float(x) for x in args.yaw_scan.split(":")))

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.photos)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        wr.writeheader()

        for fname in files:
            print(f"[{fname}]")
            rgb = np.asarray(Image.open(os.path.join(args.photos, fname))
                             .convert("RGB"))
            stem = os.path.splitext(fname)[0]
            is_bev = stem.lower().startswith("bev")
            try:
                if is_bev:
                    rgb, roi, res, ms, info = process_bev(rgb, args)
                else:
                    rgb, roi, res, ms, info = process_forward(rgb, args)
                status = ("weak lock" if res.prominence < 3.0 else
                      "angle>30deg" if abs(res.angle_from_vertical_deg) > 30.0
                      else "OK")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                wr.writerow({"image": fname, "status": f"FAILED: {exc}"})
                continue

            out_png = os.path.join(args.out, f"{stem}_result.png")
            save_figure(out_png, fname, rgb, roi, res, ms, info, status)
            gsd = info.get("gsd")
            wr.writerow({
                "image": fname, "mode": info["mode"],
                "pitch_deg": info.get("pitch"), "yaw_deg": info.get("yaw"),
                "gsd_m_per_px": gsd,
                "roi_w": res.magnitude.shape[1],
                "roi_h": res.magnitude.shape[0],
                "row_spacing_px": round(res.spacing_px, 2),
                "row_spacing_m": (round(res.spacing_px * gsd, 4)
                                  if gsd else None),
                "angle_dev_deg": round(res.angle_from_vertical_deg, 2),
                "fx_cyc_per_px": round(res.fx, 6),
                "fy_cyc_per_px": round(res.fy, 6),
                "phi_rad": round(res.phi, 4),
                "n_rows": res.n_rows,
                "ey_cm": (round(res.ey_px * gsd * 100, 2)
                          if gsd and np.isfinite(res.ey_px) else None),
                "e_theta_deg": round(res.e_theta_deg, 2),
                "corridor_m": (round(res.corridor_px * gsd, 3)
                               if gsd and np.isfinite(res.corridor_px) else None),
                "prominence": round(res.prominence, 1),
                "detect_ms": round(ms),
                "status": status,
            })
            print(f"  rows {res.n_rows}, spacing "
                  f"{res.spacing_px:.1f}px, e_theta {res.e_theta_deg:.2f} deg"
                  + (f", e_y {res.ey_px * gsd * 100:.1f} cm" if gsd else "")
                  + f", prom {res.prominence:.1f}x -> {os.path.basename(out_png)}")

    print(f"\ndone: {len(files)} images, results in {args.out}")


if __name__ == "__main__":
    main()
