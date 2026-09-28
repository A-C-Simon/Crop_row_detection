"""Run the DFT crop row detector on a video file or camera stream.

The rectification pitch and yaw are calibrated once on the first usable
frame and reused afterwards, so the per frame cost is one rectification
plus one DFT. A temporal row-lock filter (RowLockFilter) keeps a
continuous corridor state across frames: small innovations are followed
with an EMA, while jumps larger than a fraction of the row spacing
(jitter, half-period flips, harmonic spacing jumps) are held instead of
followed, and only accepted as genuine lane changes after several
mutually-consistent frames. Outputs an annotated side-by-side video
(original frame with photo-style corridor overlay | BEV with locked
lines | metrics) and a per frame CSV in the output directory.

Usage:
    python3 run_fft_video.py VIDEO [options]
    python3 run_fft_video.py 0 [options]        (webcam index 0)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import replace

import cv2
import numpy as np

from dft_crop_row_detector import (DFTRowDetector, corridor_in_image,
                                   exg_gray, rectify_forward)
from run_fft_detection import pitch_scan_score, spacing_band_px


def _wrap_deg(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


class RowLockFilter:
    """Temporal row-lock for video DFT tracking.

    Per-frame DFT detections are independent, so phase jitter, a
    half-period flip or a harmonic spacing jump can teleport the
    navigation line onto a neighbouring row (or onto the crop). The
    filter keeps a continuous corridor state (centerline base-x,
    heading, spacing) and gates each raw detection against it:

    - innovation within gate (fraction of spacing), heading and spacing
      within gates -> EMA update, status OK;
    - otherwise the raw frame is treated as a jitter outlier and HELD
      (previous state is output, so the drawn lines stand still);
    - outliers that stay mutually consistent for `persist_frames`
      frames are accepted as a genuine lane change -> RELOCK.

    A physical lane change therefore takes ~persist_frames to follow,
    while single-frame jitter never moves the lines.
    """

    def __init__(self, smooth=0.35, row_gate_frac=0.35,
                 max_spacing_change=0.20, max_heading_jump=8.0,
                 persist_frames=5, min_gate_px=8.0):
        self.smooth = float(smooth)
        self.row_gate_frac = float(row_gate_frac)
        self.max_spacing_change = float(max_spacing_change)
        self.max_heading_jump = float(max_heading_jump)
        self.persist_frames = int(persist_frames)
        self.min_gate_px = float(min_gate_px)
        self.reset()

    def reset(self):
        self.init = False
        self.filt_cx = 0.0
        self.filt_eth = 0.0
        self.filt_S = 0.0
        self.pending = 0
        self.cand_cx = 0.0
        self.cand_eth = 0.0
        self.cand_S = 0.0

    @staticmethod
    def _base_x(cl, h):
        x0, y0, x1, y1 = cl
        if abs(y1 - y0) < 1e-9:
            return (x0 + x1) / 2.0
        return x0 + (x1 - x0) * ((h - 1.0 - y0) / (y1 - y0))

    def _commit(self, cx, eth, S):
        self.filt_cx, self.filt_eth, self.filt_S = cx, eth, S

    def update(self, res, roi_shape):
        h, w = roi_shape
        cl = res.centerline() if res is not None else None
        if cl is None:
            if not self.init:
                self._commit(w / 2.0, 0.0, 50.0)
                self.init = True
                status = "INIT (no line)"
            else:
                status = "HELD (no line)"
            return self._out(res, roi_shape, status, raw_cx=None)

        raw_cx = self._base_x(cl, h)
        raw_eth = float(res.e_theta_deg)
        raw_S = float(res.spacing_px)
        if not self.init or not (np.isfinite(raw_cx) and np.isfinite(
                raw_eth) and np.isfinite(raw_S) and raw_S > 1e-6):
            if np.isfinite(raw_cx) and np.isfinite(raw_eth):
                self._commit(raw_cx, raw_eth, raw_S if raw_S > 1e-6 else 50.0)
                self.init = True
            return self._out(res, roi_shape, "INIT", raw_cx=raw_cx)

        gate = max(self.row_gate_frac * self.filt_S, self.min_gate_px)
        innov = raw_cx - self.filt_cx
        deth = _wrap_deg(raw_eth - self.filt_eth)
        dsp = abs(raw_S - self.filt_S) / max(self.filt_S, 1e-9)
        if abs(innov) <= gate and abs(deth) <= self.max_heading_jump \
                and dsp <= self.max_spacing_change:
            a = self.smooth
            z = ((1.0 - a) * np.exp(1j * np.radians(self.filt_eth))
                 + a * np.exp(1j * np.radians(raw_eth)))
            self._commit((1.0 - a) * self.filt_cx + a * raw_cx,
                         float(np.degrees(np.angle(z))),
                         (1.0 - a) * self.filt_S + a * raw_S)
            self.pending = 0
            return self._out(res, roi_shape, "OK", raw_cx=raw_cx)

        # outlier: check whether it continues a pending lane-change stream
        if self.pending == 0:
            self.cand_cx, self.cand_eth, self.cand_S = raw_cx, raw_eth, raw_S
            self.pending = 1
        else:
            c_gate = max(0.5 * gate, self.min_gate_px)
            if abs(raw_cx - self.cand_cx) <= c_gate \
                    and abs(_wrap_deg(raw_eth - self.cand_eth)) \
                    <= self.max_heading_jump \
                    and abs(raw_S - self.cand_S) / max(self.cand_S, 1e-9) \
                    <= self.max_spacing_change:
                self.cand_cx = 0.5 * self.cand_cx + 0.5 * raw_cx
                z = (0.5 * np.exp(1j * np.radians(self.cand_eth))
                     + 0.5 * np.exp(1j * np.radians(raw_eth)))
                self.cand_eth = float(np.degrees(np.angle(z)))
                self.cand_S = 0.5 * self.cand_S + 0.5 * raw_S
                self.pending += 1
            else:
                self.cand_cx, self.cand_eth, self.cand_S = raw_cx, raw_eth, raw_S
                self.pending = 1
        if self.pending >= self.persist_frames:
            self._commit(self.cand_cx, self.cand_eth, self.cand_S)
            self.pending = 0
            return self._out(res, roi_shape, "RELOCK", raw_cx=raw_cx)
        d = ("row jump" if abs(innov) > gate
             else "heading jump" if abs(deth) > self.max_heading_jump
             else "spacing jump")
        return self._out(res, roi_shape, f"HELD ({d})", raw_cx=raw_cx)

    def _out(self, res, roi_shape, status, raw_cx):
        h, w = roi_shape
        th = math.radians(self.filt_eth)
        t = np.array([math.sin(th), -math.cos(th)])
        rx = w / 2.0
        filt_ey = (rx - self.filt_cx) * t[1] - ((h - 1.0) - (h - 1.0)) * t[0]
        delta = None if raw_cx is None else self.filt_cx - raw_cx
        return {"filt_cx": float(self.filt_cx), "filt_eth": float(self.filt_eth),
                "filt_S": float(self.filt_S), "filt_ey_px": float(filt_ey),
                "filt_t": t, "delta_px": delta, "raw_cx": raw_cx,
                "status": status, "pending": int(self.pending)}


def draw_original_overlay(bgr, res, filt, map_info):
    """Photo-style overlay on the original frame: shaded corridor, orange
    bordering rows, thin red rows, cyan navigation centerline, robot star.

    Uses the same corridor_in_image() projection as the photo runner
    (lines clipped to the full rectified grid, not the valid ROI), so the
    overlay spans the frame exactly like the photo figures. All lines come
    from the row-locked corridor (raw grid translated laterally by the
    filter delta, locked heading)."""
    ov = bgr.copy()
    delta = filt["delta_px"] or 0.0
    draw_res = replace(res, intersections=np.asarray(res.intersections)
                       + delta, direction=np.asarray(filt["filt_t"]))
    cor = corridor_in_image(draw_res, map_info)
    if cor is None:
        return ov
    for p in cor["rows"]:
        cv2.polylines(ov, [p.astype(np.int32)], False, (0, 0, 255), 2)
    if cor["corridor"] is not None:
        shade = ov.copy()
        cv2.fillConvexPoly(shade, cor["corridor"].astype(np.int32),
                           (255, 255, 0))
        cv2.addWeighted(shade, 0.18, ov, 0.82, 0, ov)
    for p in cor["borders"]:
        cv2.polylines(ov, [p.astype(np.int32)], False, (0, 165, 255), 3)
    if cor["centerline"] is not None:
        cv2.polylines(ov, [cor["centerline"].astype(np.int32)], False,
                      (255, 255, 0), 4)
    rx, ry = cor["ref"]
    cv2.drawMarker(ov, (int(rx), int(ry)), (0, 255, 255),
                   cv2.MARKER_STAR, 18, 2)
    return ov


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
                rect, gsd, _ = rectify_forward(gray, float(pitch), args.height,
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


def _raw_passthrough(res, roi_shape):
    """Filter-shaped dict straight from the raw detection (--no-rowlock)."""
    import math as _math
    h, w = roi_shape
    cl = res.centerline()
    raw_cx = (RowLockFilter._base_x(cl, h) if cl is not None else w / 2.0)
    th = _math.radians(float(res.e_theta_deg))
    t = np.array([_math.sin(th), -_math.cos(th)])
    return {"filt_cx": float(raw_cx), "filt_eth": float(res.e_theta_deg),
            "filt_S": float(res.spacing_px),
            "filt_ey_px": float(res.ey_px), "filt_t": t, "delta_px": 0.0,
            "raw_cx": float(raw_cx), "status": "RAW", "pending": 0}


def draw_panel(bgr, roi, res, filt, map_info, gsd, idx, t, status,
               height=480):
    def resize_h(img):
        s = height / img.shape[0]
        return cv2.resize(img, (max(int(img.shape[1] * s), 1), height))

    h, w = roi.shape
    # left: original frame with the photo-style locked overlay
    try:
        left_full = draw_original_overlay(bgr, res, filt, map_info)
    except Exception:
        left_full = bgr
    left = resize_h(left_full)
    # middle: BEV with stabilized rows + locked corridor
    vis = cv2.cvtColor(np.clip(roi, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    delta = filt["delta_px"] or 0.0
    tdir = filt["filt_t"]
    cx, S = filt["filt_cx"], filt["filt_S"]
    L = 1.6 * float(np.hypot(h, w))
    rt = res.direction
    for xi in res.intersections + delta:
        p = np.array([xi, 0.0])
        a, b = p - rt * L, p + rt * L
        cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 (0, 0, 255), 1)
    perp = np.array([-tdir[1], tdir[0]])
    for sgn in (-1.0, 1.0):
        p = np.array([cx + sgn * 0.5 * S * perp[0],
                      (h - 1.0) + sgn * 0.5 * S * perp[1]])
        a, b = p - tdir * L, p + tdir * L
        cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 (0, 165, 255), 2)
    p = np.array([cx, h - 1.0])
    a, b = p - tdir * L, p + tdir * L
    cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
             (255, 255, 0), 2)
    if filt.get("raw_cx") is not None:
        pr = np.array([filt["raw_cx"], h - 1.0])
        ar, br = pr - rt * L, pr + rt * L
        cv2.line(vis, (int(ar[0]), int(ar[1])), (int(br[0]), int(br[1])),
                 (200, 200, 200), 1)
    rx, ry = res.ref_point
    cv2.drawMarker(vis, (int(rx), int(ry)), (0, 255, 255),
                   cv2.MARKER_STAR, 14, 2)
    cv2.circle(vis, (int(cx), int(h - 1)), 5, (255, 0, 255), 2)
    vis = resize_h(vis)

    bar = np.full((height, 400, 3), 30, np.uint8)
    ey_txt = (f"e_y locked  : {filt['filt_ey_px'] * gsd * 100:.1f} cm"
              if np.isfinite(filt["filt_ey_px"]) else "e_y locked  : n/a")
    lines = [
        f"frame {idx}   t = {t:.1f} s",
        f"rows found   : {res.n_rows}",
        f"row spacing  : {res.spacing_px * gsd * 100:.0f} cm",
        ey_txt,
        f"e_theta lock : {filt['filt_eth']:.2f} deg",
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
                    help="row-lock EMA weight of accepted measurements (0..1)")
    ap.add_argument("--persist-frames", type=int, default=5,
                    help="consistent outlier frames before a lane change is "
                         "accepted (default 5)")
    ap.add_argument("--row-gate-frac", type=float, default=0.35,
                    help="centerline jump gate as fraction of spacing "
                         "(default 0.35)")
    ap.add_argument("--heading-gate", type=float, default=8.0,
                    help="per-frame heading jump gate in deg (default 8)")
    ap.add_argument("--spacing-gate", type=float, default=0.20,
                    help="spacing change gate as fraction (default 0.20)")
    ap.add_argument("--no-rowlock", action="store_true",
                    help="disable the temporal row-lock filter (draw raw)")
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

    rowlock = (None if args.no_rowlock else RowLockFilter(
        smooth=args.smooth, row_gate_frac=args.row_gate_frac,
        max_spacing_change=args.spacing_gate,
        max_heading_jump=args.heading_gate,
        persist_frames=args.persist_frames))
    if args.no_rowlock:
        print("row-lock filter disabled (--no-rowlock): drawing raw detections")

    last, last_map = None, None
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
        map_info = last_map
        try:
            roi, gsd, map_info = rectify_forward(gray, pitch, args.height,
                                                 args.fov, args.gsd,
                                                 yaw_deg=yaw,
                                                 range_m=args.range_m)
            lo, hi, prior = spacing_band_px(gsd, args)
            d = DFTRowDetector(min_period_px=lo, max_period_px=hi,
                               spacing_prior_px=prior)
            res = d.detect(roi, ref_xy=(roi.shape[1] / 2.0,
                                        roi.shape[0] - 1.0))
            last, last_map = res, map_info
        except Exception as exc:
            if last is None:
                idx += 1
                continue
            status = f"HELD ({exc})"
            res = last
            map_info = last_map
            roi, gsd = None, None
        if roi is None:
            try:
                roi, gsd, map_info = rectify_forward(gray, pitch, args.height,
                                                     args.fov, args.gsd,
                                                     yaw_deg=yaw,
                                                     range_m=args.range_m)
            except Exception:
                idx += 1
                continue

        if rowlock is None:
            f = _raw_passthrough(res, roi.shape)
        else:
            f = rowlock.update(res, roi.shape)
            if status == "OK":
                status = f["status"]
        panel = draw_panel(bgr, roi, res, f, map_info, gsd, idx, t, status)
        if writer is not None and writer["obj"] is None:
            writer["obj"] = cv2.VideoWriter(
                writer["path"], writer["fourcc"],
                fps / max(args.stride, 1),
                (panel.shape[1], panel.shape[0]))
        if writer is not None and writer["obj"] is not None:
            writer["obj"].write(panel)
        wr.writerow([idx, round(t, 3), pitch, yaw,
                     round(f["filt_S"] * gsd, 4), res.n_rows,
                     round(f["filt_ey_px"] * gsd * 100.0, 2),
                     round(f["filt_eth"], 2), round(res.prominence, 1),
                     status])
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
