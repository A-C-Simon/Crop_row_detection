#!/usr/bin/env python3
"""
ExG Video Navigation Runner
===========================
Runs the ExG visual servoing pipeline (column-aware dynamic window +
Isolation Forest) on video files, producing control commands and
overlay videos.

This is the video counterpart to run_vcrn_debug.py (which is for Photos).
It reuses the same perception and control logic but processes a video
stream frame-by-frame and writes an output video with the vs overlay.

Usage:
  python3 run_exg_video.py --input ../../Photos/test_video1.mp4 --output ./video_output
  python3 run_exg_video.py --input 0 --output ./video_output --show  # webcam

Outputs per video:
  - <name>_exg_nav.mp4 : BGR overlay video (final image with red line, window, dots)
  - <name>_exg_nav.csv : per-frame v,w,err_x,err_theta,has_line, time
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import math

# Import the debug pipeline logic (process_image, load_params)
from run_vcrn_debug import process_image, load_params
from run_vcrn_debug import save_debug_composite  # not used for video, but available


def run_video(input_path: str, output_dir: Path, params, show: bool = False):
    # Try camera index
    try:
        cam_idx = int(input_path)
        cap = cv2.VideoCapture(cam_idx)
        is_camera = True
        name = f"camera{cam_idx}"
    except ValueError:
        cap = cv2.VideoCapture(input_path)
        is_camera = False
        name = Path(input_path).stem

    if not cap.isOpened():
        print(f"Cannot open {input_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_exg_nav.csv"
    # Writer will be created after first frame to know size (final overlay is 640x480)
    writer = None
    writer_w = writer_h = None
    out_path = output_dir / f"{name}_exg_nav.mp4" if not is_camera else None

    # For video, we use the same image size as the pipeline (640x480)
    # The final overlay is always 640x480, so writer size is fixed

    with open(csv_path, "w", newline="") as f:
        csvw = csv.writer(f)
        csvw.writerow(["frame", "timestamp", "v_mps", "w_radps", "w_degps", "err_x_px", "err_theta_deg", "has_line", "n_nh", "n_inliers", "time_ms"])

        frame_idx = 0
        while True:
            ret, bgr_full = cap.read()
            if not ret:
                break
            # The pipeline expects BGR and will resize to 640x480 internally
            # But for video, the input may already be 700x370 or 960x540, we let process_image handle resize
            t0 = time.perf_counter()
            # Use process_image from run_vcrn_debug which does resize to 640x480 and all steps
            # We need to pass a Path-like, but process_image expects a Path and does cv2.imread.
            # For video, we need a direct version that takes BGR array.
            # So we call a helper that mimics process_image but with BGR input.
            # Instead, we will directly use the logic from process_image but with BGR array.
            # To avoid duplication, we create a temporary file or just call the internal logic.
            # For now, we will save the frame to a temp file and call process_image, but that's slow.
            # Better to directly call the pipeline steps.

            # Direct pipeline for video frame (avoid disk I/O)
            # We replicate the core of process_image but with bgr array input
            # For simplicity, we will call process_image_via_array
            res = process_image_array(bgr_full, params)
            dt = (time.perf_counter() - t0) * 1000.0

            # Control is already computed inside res? No, process_image_array returns intermediates with fit
            # The fitLine is the navigation line, and control would be based on it.
            # For ExG, the navigation line is the fitted line through inliers.
            # We need to compute v,w from the line. The debug pipeline already does fitLine,
            # but not the control law. For video, we should compute v,w similar to mr_vs.

            # Extract fit info and compute simple control (similar to mr_vs but for ExG row)
            has_line = res["has_line"]
            fit_info = res["fit_info"]
            n_nh = res["n_nh"]
            n_in = res.get("n_iso_in", n_nh)

            # Simple control: use the same logic as mr_vs but for ExG row
            # For ExG, the line is the crop row itself, we want to keep it centred.
            # Use the bottom point of the fitted line as reference.
            if has_line and fit_info is not None:
                inside = fit_info["inside"]
                if len(inside) >= 2:
                    # line endpoints in 640x480
                    P = np.array(inside[0], dtype=float)  # one end
                    Q = np.array(inside[1], dtype=float)
                    # Ensure P is bottom (larger y)
                    if P[1] < Q[1]:
                        P, Q = Q, P
                    # Feature
                    w_img, h_img = 640, 480
                    X = P[0] - w_img/2.0
                    Y = P[1] - h_img/2.0
                    # Theta
                    Yv = P[1] - Q[1]
                    Xv = Q[0] - P[0]
                    phi = math.atan2(Yv, Xv)
                    Theta = phi - math.pi/2
                    # wrap
                    while Theta > math.pi:
                        Theta -= 2*math.pi
                    while Theta < -math.pi:
                        Theta += 2*math.pi
                    # Control gains (from MRVS but for ExG)
                    err_x = X
                    err_theta = Theta
                    err_x_norm = err_x / w_img
                    w_raw = -(2.0 * err_x_norm + 1.0 * err_theta)
                    w_max = 0.6
                    w_ang = max(-w_max, min(w_max, w_raw))
                    if abs(w_ang) < 0.01:
                        w_ang = 0.0
                    v = 0.20
                else:
                    v, w_ang = 0.0, 0.0
                    err_x = err_theta = 0.0
            else:
                v, w_ang = 0.0, 0.0
                err_x = err_theta = 0.0

            csvw.writerow([frame_idx, time.time(), f"{v:.4f}", f"{w_ang:.4f}", f"{math.degrees(w_ang):.2f}",
                           f"{err_x:.1f}", f"{math.degrees(err_theta):.2f}" if has_line else "0",
                           int(has_line), n_nh, n_in, f"{dt:.1f}"])

            # For video, use the final overlay image (640x480) which already has red line, window, dots
            overlay = res["final_img"]  # BGR 640x480
            # Add text overlay with v,w
            txt = f"v={v:.2f} w={math.degrees(w_ang):.1f}deg err_x={err_x:.0f} Theta={math.degrees(err_theta) if has_line else 0:.1f}"
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            # Add frame number
            cv2.putText(overlay, f"frame {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            if writer is None and not is_camera:
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                if fps < 1 or fps > 120:
                    fps = 20.0
                writer_h, writer_w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (writer_w, writer_h))
                if not writer.isOpened():
                    print(f"[WARN] VideoWriter failed for {writer_w}x{writer_h}")

            if writer is not None:
                if overlay.shape[1] != writer_w or overlay.shape[0] != writer_h:
                    overlay = cv2.resize(overlay, (writer_w, writer_h), interpolation=cv2.INTER_AREA)
                writer.write(overlay)

            if show:
                preview = overlay
                if preview.shape[1] > 1280:
                    scale = 1280 / preview.shape[1]
                    preview = cv2.resize(preview, (1280, int(preview.shape[0]*scale)), interpolation=cv2.INTER_AREA)
                cv2.imshow("ExG Navigation - Video", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: v={v:.2f} w={math.degrees(w_ang):.1f}deg has_line={has_line} nh={n_nh} in={n_in} {dt:.1f}ms")

    cap.release()
    if writer:
        writer.release()
        print(f"Video saved to {out_path} ({frame_idx} frames)")
    print(f"CSV saved to {csv_path}")
    print(f"Saved to {output_dir}")


def process_image_array(bgr, params):
    """
    Version of process_image that takes a BGR array directly (for video),
    without reading from disk. It replicates the logic of run_vcrn_debug.process_image
    but with array input.
    """
    import time
    import cv2
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from run_vcrn_debug import detect_column_aware_window, fit_line_clip

    t_all = time.perf_counter()
    timings = {}
    rejection_log = []

    bgr_orig = bgr
    width, height = params["width"], params["height"]
    # Resize to WxH (like run_vcrn_debug)
    t0 = time.perf_counter()
    bgr_resized = cv2.resize(bgr_orig, (width, height), interpolation=cv2.INTER_AREA)
    timings["resize"] = (time.perf_counter()-t0)*1000

    # HSV
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    h_mask = cv2.inRange(h_chan, params["min_Hue"], params["max_Hue"])
    s_mask = cv2.inRange(s_chan, params["min_Saturation"], params["max_Saturation"])
    v_mask = cv2.inRange(v_chan, params["min_Value"], params["max_Value"])
    combined = cv2.bitwise_and(h_mask, cv2.bitwise_and(s_mask, v_mask))
    timings["hsv"] = (time.perf_counter()-t0)*1000
    if int(cv2.countNonZero(combined)) == 0:
        rejection_log.append("Combined mask empty")

    # Contours
    t0 = time.perf_counter()
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    timings["contours"] = (time.perf_counter()-t0)*1000
    n_contours = len(contours)
    contours_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i,cnt in enumerate(contours):
        cv2.drawContours(contours_img, contours, i, (0,255,0), 1)

    # Centers
    t0 = time.perf_counter()
    centers = []
    for cnt in contours:
        poly = cv2.approxPolyDP(cnt, 2, True)
        (x,y), rad = cv2.minEnclosingCircle(poly)
        centers.append((float(x), float(y)))
    n_centers = len(centers)
    centers_img = bgr_resized.copy()
    for (x,y) in centers:
        cv2.circle(centers_img, (int(round(x)), int(round(y))), 3, (51,204,51), cv2.FILLED)
    timings["centers"] = (time.perf_counter()-t0)*1000

    # Window - column-aware
    t0 = time.perf_counter()
    if params.get("colaware_enabled", True):
        Xc_dyn, L_dyn, H_dyn, profile_raw, profile_smooth, peak_xs, chosen_idx, median_gap, y0_roi = detect_column_aware_window(combined, centers, params, width, height)
        Xc, L, H = Xc_dyn, L_dyn, H_dyn
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
        Yc = params["ex_Yc"]
    if not params.get("colaware_enabled", True):
        Yc = params["ex_Yc"]
    else:
        Yc = params["ex_Yc"]
    # Profile image (not needed for video, but keep for consistency)
    prof_h = 80
    profile_img = np.zeros((prof_h, width, 3), dtype=np.uint8)
    if colaware_profile.max() > 0:
        prof_norm = colaware_profile / colaware_profile.max() * (prof_h-10)
    else:
        prof_norm = colaware_profile
    for x in range(width-1):
        y1 = int(prof_h - 5 - prof_norm[x])
        y2 = int(prof_h - 5 - prof_norm[x+1])
        cv2.line(profile_img, (x,y1), (x+1,y2), (255,255,255), 1)
    for i, px in enumerate(colaware_peaks):
        col = (0,255,0) if i==colaware_chosen else (0,0,255)
        cv2.circle(profile_img, (int(px), int(prof_h -5 - (colaware_profile[px]/colaware_profile.max()* (prof_h-10) if colaware_profile.max()>0 else 0))), 4, col, -1)
    cv2.line(profile_img, (width//2,0), (width//2,prof_h), (255,255,0), 1)
    cv2.rectangle(profile_img, (int(Xc - L/2), 0), (int(Xc + L/2), prof_h), (255,204,102), 1)

    window_img = bgr_resized.copy()
    cv2.rectangle(window_img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 3)
    for (x,y) in centers:
        cv2.circle(window_img, (int(round(x)), int(round(y))), 2, (120,120,120), cv2.FILLED)

    nh_points = []
    outside_points = []
    for (x,y) in centers:
        if (Xc - L/2 < x < Xc + L/2) and (Yc - H/2 < y < Yc + H/2):
            nh_points.append((x,y))
        else:
            outside_points.append((x,y))
    n_nh = len(nh_points)
    nh_img = bgr_resized.copy()
    cv2.rectangle(nh_img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 3)
    for (x,y) in outside_points:
        cv2.circle(nh_img, (int(round(x)), int(round(y))), 3, (100,100,100), cv2.FILLED)
    for (x,y) in nh_points:
        cv2.circle(nh_img, (int(round(x)), int(round(y))), 5, (0,204,255), cv2.FILLED)
    timings["neighbourhood"] = (time.perf_counter()-t0)*1000
    if n_nh==0:
        rejection_log.append(f"is_in_neigbourhood: 0 points inside {L}x{H} window @({Xc},{Yc})")

    # Isolation Forest
    t0 = time.perf_counter()
    iso_inliers = nh_points
    iso_outliers = []
    iso_applied = False
    n_iso_in = n_nh
    n_iso_out = 0
    iso_img = bgr_resized.copy()
    cv2.rectangle(iso_img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 2)
    for (x,y) in outside_points:
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 2, (70,70,70), cv2.FILLED)
    if params.get("iso_enabled", True) and n_nh >= params.get("iso_min_points", 12):
        try:
            pts = np.array(nh_points, dtype=np.float32)
            iso = IsolationForest(contamination=params.get("iso_contamination", 0.15),
                                  n_estimators=params.get("iso_n_estimators", 100),
                                  random_state=42)
            pred = iso.fit_predict(pts)
            mask_in = pred == 1
            iso_inliers = [tuple(p) for p, m in zip(nh_points, mask_in) if m]
            iso_outliers = [tuple(p) for p, m in zip(nh_points, mask_in) if not m]
            n_iso_in = len(iso_inliers)
            n_iso_out = len(iso_outliers)
            iso_applied = True
        except Exception as e:
            iso_inliers = nh_points
            iso_outliers = []
            iso_applied = False
            n_iso_in = n_nh
    else:
        iso_inliers = nh_points
        iso_outliers = []
        n_iso_in = n_nh
    for (x,y) in iso_inliers:
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 5, (0,204,255), cv2.FILLED)
    for (x,y) in iso_outliers:
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 5, (0,0,255), cv2.FILLED)
        cv2.circle(iso_img, (int(round(x)), int(round(y))), 7, (0,0,255), 1)
    timings["iso"] = (time.perf_counter()-t0)*1000

    # Gap disabled
    gap_kept = iso_inliers
    gap_removed = []
    n_gap_in = n_iso_in
    n_gap_out = 0
    gap_info = {"n_clusters":1}
    gap_img = iso_img.copy()
    final_inliers = gap_kept
    final_outliers = iso_outliers
    timings["gap"] = 0.0

    # FitLine
    t0 = time.perf_counter()
    fit_info = None
    has_line = False
    raw_line_img = bgr_resized.copy()
    clipped_line_img = bgr_resized.copy()
    final_img = bgr_resized.copy()
    for img in [raw_line_img, clipped_line_img, final_img]:
        cv2.rectangle(img, (int(Xc - L/2), int(Yc - H/2)), (int(Xc + L/2), int(Yc + H/2)), (255,204,102), 2)
    for im in [raw_line_img, clipped_line_img, final_img]:
        for (x,y) in final_inliers:
            cv2.circle(im, (int(round(x)), int(round(y))), 4, (0,204,255), cv2.FILLED)
        for (x,y) in iso_outliers:
            cv2.circle(im, (int(round(x)), int(round(y))), 4, (0,0,255), cv2.FILLED)
        for (x,y) in outside_points:
            cv2.circle(im, (int(round(x)), int(round(y))), 2, (70,70,70), cv2.FILLED)

    if n_gap_in>0:
        fit_info = fit_line_clip(final_inliers, width, height)
        if fit_info is not None:
            inside = fit_info["inside"]
            if len(inside)>=2:
                has_line=True
                cv2.line(clipped_line_img, (int(round(inside[0][0])), int(round(inside[0][1]))),
                         (int(round(inside[1][0])), int(round(inside[1][1]))), (0,0,255), 2, cv2.LINE_AA)
                cv2.line(final_img, (int(round(inside[0][0])), int(round(inside[0][1]))),
                         (int(round(inside[1][0])), int(round(inside[1][1]))), (0,0,255), 2, cv2.LINE_AA)
                cv2.line(raw_line_img, (int(round(fit_info["ext_p1"][0])), int(round(fit_info["ext_p1"][1]))),
                         (int(round(fit_info["ext_p2"][0])), int(round(fit_info["ext_p2"][1]))), (255,0,0), 2, cv2.LINE_AA)
    timings["fitline"] = (time.perf_counter()-t0)*1000

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
        "has_line": has_line,
        "fit_info": fit_info,
        "window_Xc": Xc,
        "window_Yc": Yc,
        "window_L": L,
        "window_H": H,
        "colaware_peaks": colaware_peaks,
        "colaware_chosen": colaware_chosen,
        "colaware_median_gap": colaware_median_gap if 'colaware_median_gap' in locals() else None,
    }
    return intermediates


def main():
    parser = argparse.ArgumentParser(description="ExG video navigation")
    parser.add_argument("--input", required=True, help="Video file or camera index")
    parser.add_argument("--output", default="./video_output", help="Output directory")
    parser.add_argument("--params", default=str(Path(__file__).resolve().parent.parent / "params" / "agribot_vs_run.yaml"), help="Params yaml")
    parser.add_argument("--show", action="store_true", help="Show preview")
    args = parser.parse_args()

    import yaml

    # Load params
    yaml_path = Path(args.params)
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        p = data.get("agribot_vs", {}).get("ros__parameters", data)
        params = {
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
            "ex_Yc": int(p.get("ex_Yc", 380)),
            "nh_L": int(p.get("nh_L", 80)),
            "nh_H": int(p.get("nh_H", 180)),
            "nh_offset": int(p.get("nh_offset", 200)),
            "iso_enabled": bool(p.get("iso_enabled", True)),
            "iso_contamination": float(p.get("iso_contamination", 0.15)),
            "iso_min_points": int(p.get("iso_min_points", 12)),
            "iso_n_estimators": int(p.get("iso_n_estimators", 100)),
            "colaware_enabled": bool(p.get("colaware_enabled", True)),
            "colaware_y0_frac": float(p.get("colaware_y0_frac", 0.55)),
            "gap_enabled": bool(p.get("gap_enabled", False)),
            "gap_eps": float(p.get("gap_eps", 14)),
            "gap_min_samples": int(p.get("gap_min_samples", 6)),
            "gap_min_points": int(p.get("gap_min_points", 12)),
        }
    else:
        print(f"Params not found {yaml_path}, using defaults")
        from run_vcrn_debug import load_params
        # fallback to default location
        default_path = Path(__file__).parent.parent / "params" / "agribot_vs_run.yaml"
        params = load_params(default_path)

    run_video(args.input, Path(args.output), params, show=args.show)


if __name__ == "__main__":
    main()
