"""Run the DFT crop row detector on a video file or camera stream.

The rectification pitch and yaw are calibrated once on the first usable
frame and reused afterwards, so the per frame cost is one rectification
plus one DFT. Lateral and heading deviations are smoothed with an
exponential moving average. Outputs an annotated side-by-side video and
a per frame CSV in the output directory.

Usage:
    python3 run_fft_video.py VIDEO [options]
    python3 run_fft_video.py 0 [options]        (webcam index 0)
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import cv2
import numpy as np

from dft_crop_row_detector import DFTRowDetector, exg_gray, rectify_forward
from run_fft_detection import pitch_scan_score, spacing_band_px


def ema(prev, new, alpha):
    if new is None or np.isnan(new):
        return prev
    if prev is None or np.isnan(prev):
        return new
    return (1.0 - alpha) * prev + alpha * new


def ema_angle(prev, new, alpha):
    if new is None or np.isnan(new):
        return prev
    if prev is None or np.isnan(prev):
        return new
    z = ((1.0 - alpha) * np.exp(1j * np.radians(prev))
         + alpha * np.exp(1j * np.radians(new)))
    return float(np.degrees(np.angle(z)))


def calibrate(gray, args):
    """Pick rectification pitch and yaw on one frame (same rules as the
    photo batch runner)."""
    cands = []
    yaws = ([0.0] if args.yaw_scan is None else
            list(np.arange(args.yaw_scan[0], args.yaw_scan[1] + 1e-9,
                           args.yaw_scan[2])))
    for pitch in np.arange(args.scan[0], args.scan[1] + 1e-9, args.scan[2]):
        for yaw in yaws:
            try:
                rect, gsd = rectify_forward(gray, float(pitch), args.height,
                                            args.fov, args.gsd,
                                            yaw_deg=float(yaw),
                                            range_m=args.range_m)
                lo, hi, prior = spacing_band_px(gsd, args)
                d = DFTRowDetector(min_period_px=lo, max_period_px=hi,
                                   spacing_prior_px=prior)
                r = d.detect(rect, ref_xy=(rect.shape[1] / 2.0,
                                           rect.shape[0] - 1.0))
                if r.n_rows < 3 or r.prominence < 15.0:
                    continue
                score = pitch_scan_score(r, args.verticality_sigma)
            except Exception:
                continue
            cands.append((score, r.prominence, float(pitch), float(yaw)))
    if not cands:
        raise RuntimeError("calibration failed: no usable pitch/yaw on "
                           "the first frame")
    best_vert = max(cands, key=lambda c: c[0])
    if best_vert[0] < 8.0:
        chosen = max(cands, key=lambda c: c[1])
        print("    no near-vertical lock; using dominant pattern")
    else:
        chosen = best_vert
    return chosen[2], chosen[3]


def draw_panel(bgr, roi, res, gsd, ey_s, eth_s, idx, t, status, height=480):
    def resize_h(img):
        s = height / img.shape[0]
        return cv2.resize(img, (max(int(img.shape[1] * s), 1), height))

    left = resize_h(bgr)
    vis = cv2.cvtColor(np.clip(roi, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in res.line_endpoints():
        cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                 (0, 0, 255), 1)
    cl = res.centerline()
    if cl:
        cv2.line(vis, (int(cl[0]), int(cl[1])), (int(cl[2]), int(cl[3])),
                 (255, 255, 0), 2)
    rx, ry = res.ref_point
    cv2.drawMarker(vis, (int(rx), int(ry)), (0, 255, 255),
                   cv2.MARKER_STAR, 14, 2)
    vis = resize_h(vis)

    bar = np.full((height, 380, 3), 30, np.uint8)
    ey_txt = (f"e_y smoothed : {ey_s:.1f} cm"
              if ey_s is not None and np.isfinite(ey_s) else "e_y smoothed : n/a")
    lines = [
        f"frame {idx}   t = {t:.1f} s",
        f"rows found   : {res.n_rows}",
        f"row spacing  : {res.spacing_px * gsd * 100:.0f} cm",
        ey_txt,
        f"e_theta smth : {eth_s:.2f} deg",
        f"prominence   : {res.prominence:.1f}x",
        f"status       : {status}",
    ]
    y = 30
    for ln in lines:
        cv2.putText(bar, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 30
    return np.hstack([left, vis, bar])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="video file path or camera index")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "results_fft"))
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N processed frames (0 = all)")
    ap.add_argument("--smooth", type=float, default=0.35,
                    help="EMA weight of the new measurement (0..1)")
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
    ap.add_argument("--show", action="store_true",
                    help="live preview window (press q to quit)")
    ap.add_argument("--no-video", action="store_true",
                    help="do not write the overlay mp4")
    args = ap.parse_args(argv)
    args.scan = tuple(float(x) for x in args.scan.split(":"))
    args.yaw_scan = (None if str(args.yaw_scan).lower() in ("none", "")
                     else tuple(float(x) for x in str(args.yaw_scan).split(":")))

    os.makedirs(args.out, exist_ok=True)
    src = int(args.video) if str(args.video).isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, bgr = cap.read()
    if not ok:
        raise SystemExit("no frames readable")
    gray0 = exg_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    t0 = time.perf_counter()
    pitch, yaw = calibrate(gray0, args)
    print(f"calibrated frame 0: pitch {pitch:.1f} deg, yaw {yaw:.1f} deg "
          f"({time.perf_counter() - t0:.1f} s)")

    stem = (f"cam{src}" if isinstance(src, int)
            else os.path.splitext(os.path.basename(src))[0])
    csv_path = os.path.join(args.out, f"{stem}_video_metrics.csv")
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["frame", "time_s", "pitch_deg", "yaw_deg", "row_spacing_m",
                 "n_rows", "ey_cm_smoothed", "e_theta_deg_smoothed",
                 "prominence", "status"])

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(args.out, f"{stem}_overlay.mp4")
        writer = {"obj": None, "path": out_path, "fourcc": fourcc}

    ey_s, eth_s, last = None, None, None
    idx, n_proc = 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    t_start = time.perf_counter()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % max(args.stride, 1) != 0:
            idx += 1
            continue
        t = idx / fps
        gray = exg_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        status = "OK"
        res = None
        try:
            roi, gsd = rectify_forward(gray, pitch, args.height, args.fov,
                                       args.gsd, yaw_deg=yaw,
                                       range_m=args.range_m)
            lo, hi, prior = spacing_band_px(gsd, args)
            d = DFTRowDetector(min_period_px=lo, max_period_px=hi,
                               spacing_prior_px=prior)
            res = d.detect(roi, ref_xy=(roi.shape[1] / 2.0,
                                        roi.shape[0] - 1.0))
            last = res
        except Exception as exc:
            if last is None:
                idx += 1
                continue
            status = f"HELD ({exc})"
            res = last
            roi, gsd = None, None
        if roi is None:
            try:
                roi, gsd = rectify_forward(gray, pitch, args.height,
                                           args.fov, args.gsd, yaw_deg=yaw,
                                           range_m=args.range_m)
            except Exception:
                idx += 1
                continue

        ey_s = ema(ey_s, res.ey_px * gsd * 100.0, args.smooth)
        eth_s = ema_angle(eth_s, res.e_theta_deg, args.smooth)
        panel = draw_panel(bgr, roi, res, gsd, ey_s, eth_s, idx, t, status)
        if writer is not None and writer["obj"] is None:
            writer["obj"] = cv2.VideoWriter(
                writer["path"], writer["fourcc"],
                fps / max(args.stride, 1),
                (panel.shape[1], panel.shape[0]))
        if writer is not None and writer["obj"] is not None:
            writer["obj"].write(panel)
        wr.writerow([idx, round(t, 3), pitch, yaw,
                     round(res.spacing_px * gsd, 4), res.n_rows,
                     None if ey_s is None else round(ey_s, 2),
                     round(eth_s, 2), round(res.prominence, 1), status])
        if args.show:
            cv2.imshow("DFT crop rows (q quits)", panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1
        n_proc += 1
        if args.max_frames and n_proc >= args.max_frames:
            break

    cap.release()
    if writer is not None and writer["obj"] is not None:
        writer["obj"].release()
        print(f"overlay video : {writer['path']}")
    fh.close()
    print(f"per frame csv : {csv_path}")
    print(f"processed {n_proc} frames in {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
