#!/usr/bin/env python3
"""
MultiROI Navigation Runner - Furrow following visual servoing
==============================================================
Runs the MultiROI detector and the visual servoing controller on a
folder of images or a video, producing control commands for a rover.

This is the "rest of the navigation stack" after the navigation line
is already available (furrow centre).  It implements the actual rover
movement: converts the image line into v,w velocities.

Temporal robustness:
  A TemporalNavigationFilter sits between the raw detector and the
  controller.  It filters X / Theta / width with innovation gating
  and persistence, so transient missing-plant jumps do not cause
  violent steering.  The controller's w is additionally low-pass /
  rate-limited.  Raw and filtered states are both logged / drawn.

Usage:
  # On Photos (offline, for analysis)
  python3 run_mr_navigation.py --input ../../Photos --output ./nav_output

  # On video (temporal enabled by default)
  python3 run_mr_navigation.py --input /path/to/video.mp4 --output ./nav_output --video

  # On video looped forever until stopped (q or Ctrl-C)
  python3 run_mr_navigation.py --input /path/to/video.mp4 --output ./nav_output --video --loop
  python3 run_mr_navigation.py --input /path/to/video.mp4 --output /tmp --video --show --loop

  # Live camera (ROS-free, OpenCV capture)
  python3 run_mr_navigation.py --input 0 --output ./nav_output --video --show

  # Folder with temporal forced on (if images are a sequence)
  python3 run_mr_navigation.py --input ../../Photos --output ./nav_output --temporal

Outputs:
  - nav_output/<name>_nav.png : overlay with navigation line and velocity
  - nav_output/nav_commands.csv : per-frame v,w, errors, line params
  - nav_output/nav_video.mp4 (if --video)

The controller is intentionally simple and does not require ROS.
For ROS2 integration, wrap MultiROIVS in a node subscribing to
/front/image_raw and publishing /cmd_vel (see bottom of file).

Author: adapted from ExG/agribot_vs.cpp
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from test_multi_roi import MultiROIDetector
from mr_vs import MultiROIVS, MRVSParams
from temporal_filter import TemporalNavigationFilter, TemporalFilterParams
from lookahead_corridor_map import LookaheadCorridorMap, LookaheadParams


def _build_raw_dict(res, F, PQ, image_shape):
    """Create raw measurement dict for TemporalNavigationFilter from detector + vs output."""
    h, w = image_shape[:2]
    has_line = F is not None and PQ is not None
    if has_line:
        P, Q = PQ
        # bottom x in full image coords
        raw_bottom_x = float(P[0])
        raw_theta = float(F[2])
        raw_X = float(F[0])
        raw_err_x = float(raw_X)
        raw_err_theta = float(math.degrees(raw_theta))
        # width from detector diagnostics (cropped == full approximately)
        raw_width = float(res.get("median_width", 0) or res.get("bottom_width", 0) or 120.0)
        if not math.isfinite(raw_width) or raw_width < 5:
            raw_width = 120.0
    else:
        raw_bottom_x = w / 2.0
        raw_theta = 0.0
        raw_X = 0.0
        raw_err_x = 0.0
        raw_err_theta = 0.0
        raw_width = float(res.get("median_width", 0) or 120.0)
        if not math.isfinite(raw_width) or raw_width < 5:
            raw_width = 120.0

    return {
        "raw_bottom_x": float(raw_bottom_x),
        "raw_theta": float(raw_theta),
        "raw_width": float(raw_width),
        "raw_X": float(raw_X),
        "raw_err_x": float(raw_err_x),
        "raw_err_theta_deg": float(raw_err_theta),
        "has_line": bool(has_line),
        "n_two_sided": int(res.get("n_two_sided", 0)),
        "n_q_accepted": int(res.get("n_q_accepted", 0)),
        "n_q_rejected": int(res.get("n_q_rejected", 0)),
        "weed_pressure": float(res.get("weed_pressure", 0.0)),
        "median_width": float(res.get("median_width", raw_width)),
        "bottom_width": float(res.get("bottom_width", raw_width)),
    }


def process_image(bgr, detector, vs, draw=True, t_filter=None, dt=None, last_w=None, lookahead_map=None):
    """Run detection + (optionally lookahead map + temporal filter) + visual servoing.

    Pipeline:
      detector -> lookahead_map (corrects local bottom false expansion using visible future)
               -> temporal_filter (innovation gating, persistence, EMA)
               -> controller (low-pass/rate-limited w)

    When lookahead_map is supplied the raw strip observations are validated against
    the remembered future corridor before reaching the temporal filter.
    When t_filter is supplied the (corrected) feature is gated temporally.
    """
    t0 = time.perf_counter()
    h, w = bgr.shape[:2]
    # lookahead prior for detector ROI hold (boxes/midpoints stay steady)
    lookahead_prior = None
    if lookahead_map is not None:
        try:
            # predicted future corridor at current y (motion-compensated)
            pred_bins = lookahead_map.get_prediction((h, w))
            lookahead_prior = {"pred_bins": pred_bins}
        except Exception:
            lookahead_prior = None
    res = detector.detect(bgr, lookahead_prior=lookahead_prior)
    crop_offset = res.get("crop_offset", (0, 0))

    # Get navigation line/curve
    nav_line = res.get("nav_line")
    nav_curve = res.get("nav_curve")

    F_raw, PQ_raw = vs.nav_line_to_feature(nav_line, nav_curve, crop_offset, (h, w))

    # Build raw dict for filter / fallback control
    raw_dict = _build_raw_dict(res, F_raw, PQ_raw, (h, w))

    # --- lookahead corridor memory (spatial) ---
    map_out = None
    corrected_for_temporal = raw_dict
    if lookahead_map is not None:
        strip_profile = res.get("strip_profile", [])
        map_out = lookahead_map.update(strip_profile, res, (h, w), dt=dt, last_w=last_w)
        # Build corrected dict that temporal filter will consume
        # Preserve original raw for diagnostics but feed corrected values as "raw" to temporal
        corrected_for_temporal = {
            "raw_bottom_x": float(map_out["corrected_bottom_x"]),
            "raw_theta": float(map_out["corrected_theta"]),
            "raw_width": float(map_out["corrected_width"]),
            "raw_X": float(map_out["corrected_X"]),
            "raw_err_x": float(map_out["corrected_X"]),
            "raw_err_theta_deg": float(math.degrees(map_out["corrected_theta"])),
            "has_line": bool(raw_dict["has_line"] or map_out["map_confidence"] > 0.15),
            "n_two_sided": int(raw_dict["n_two_sided"]),
            "n_q_accepted": int(raw_dict["n_q_accepted"]),
            "n_q_rejected": int(raw_dict["n_q_rejected"]),
            "weed_pressure": float(raw_dict["weed_pressure"]),
            "median_width": float(map_out["corrected_width"]),
            "bottom_width": float(map_out["corrected_width"]),
            # diagnostics: keep original raw for logging
            "orig_raw_bottom_x": float(raw_dict["raw_bottom_x"]),
            "orig_raw_width": float(raw_dict["raw_width"]),
            "orig_raw_theta": float(raw_dict["raw_theta"]),
        }

    # choose which dict the temporal filter sees
    filter_input = corrected_for_temporal if lookahead_map is not None else raw_dict

    # temporal filter path
    filt_out = None
    if t_filter is not None:
        filt_out = t_filter.update(filter_input, dt=dt, last_w=last_w)
        F_used = filt_out["filt_F"]
        # P/Q for filtered – use filter's synthesized points (full image coords)
        P_filt = np.array(filt_out["filt_P"], dtype=float)
        Q_filt = np.array(filt_out["filt_Q"], dtype=float)
        # compute control from filtered feature; smoother inside vs uses confidence
        # combine confidences: temporal conf * map conf if map exists
        eff_conf = float(filt_out["confidence"])
        if map_out is not None:
            eff_conf = float(min(eff_conf, map_out["map_confidence"] * 0.6 + 0.4))  # map influences but not dominates
            # also if map is holding/reject, reduce confidence slightly
            if map_out["map_status"] in ("map_hold", "occlusion_hold", "map_reject", "map_pending"):
                eff_conf = float(min(eff_conf, map_out["map_confidence"] * 0.8 + 0.2))
        v, w_ang, info = vs.compute_control(F_used, dt=dt, confidence=eff_conf, smooth=True)
        # augment info with filtered/raw diagnostics + map
        info = dict(info)
        info.update({
            "filt_err_x": float(filt_out["filt_X"]),
            "filt_err_theta_deg": float(filt_out["filt_theta_deg"]),
            "filt_width": float(filt_out["filt_width"]),
            "filt_bottom_x": float(filt_out["filt_bottom_x"]),
            "confidence": float(eff_conf),
            "raw_confidence": float(filt_out["raw_confidence"]),
            "status": filt_out["status"],
            "is_large": bool(filt_out["is_large"]),
            "innovation_x": float(filt_out["innovation_x"]),
            "innovation_theta_deg": float(filt_out["innovation_theta_deg"]),
            "innovation_width": float(filt_out["innovation_width"]),
            "pending_count": int(filt_out["pending_count"]),
            "raw_err_x": float(raw_dict["raw_err_x"]),
            "raw_err_theta_deg": float(raw_dict["raw_err_theta_deg"]),
            "orig_raw_err_x": float(raw_dict["raw_err_x"]),
            "n_two_sided": int(raw_dict["n_two_sided"]),
            "n_q_accepted": int(raw_dict["n_q_accepted"]),
            "median_width": float(raw_dict["median_width"]),
        })
        # add map diagnostics if present
        if map_out is not None:
            info.update({
                "map_status": map_out["map_status"],
                "map_confidence": float(map_out["map_confidence"]),
                "map_bottom_x": float(map_out["corrected_bottom_x"]),
                "map_width": float(map_out["corrected_width"]),
                "raw_map_innovation_x": float(map_out["raw_map_innovation_x"]),
                "raw_map_innovation_width": float(map_out["raw_map_innovation_width"]),
                "spatial_old": int(map_out["spatial_support_old"]),
                "spatial_new": int(map_out["spatial_support_new"]),
                "map_pred_bottom_x": float(map_out["pred_bottom_x"]),
                "map_shift_px": float(map_out["shift_px"]),
                "eff_confidence": float(eff_conf),
            })
        P, Q = P_filt, Q_filt
        # also keep raw for overlay comparison
        P_raw, Q_raw = (PQ_raw if PQ_raw is not None else (None, None))
        if PQ_raw is not None:
            P_raw_np, Q_raw_np = np.array(PQ_raw[0], dtype=float), np.array(PQ_raw[1], dtype=float)
        else:
            P_raw_np = Q_raw_np = None
        F = F_used
    else:
        # legacy non-temporal path (but still may have lookahead correction)
        if lookahead_map is not None:
            # use corrected_for_temporal directly to compute control without temporal EMA
            # synthesize a feature from corrected values
            # F_corrected = [X, Y, Theta] with X from corrected_bottom_x
            F_corr = np.array([float(map_out["corrected_X"]), h/4.0, float(map_out["corrected_theta"])], dtype=float)
            # synthesize P/Q from corrected
            # we can use map_out's corrected positions for drawing
            P_c = np.array([float(map_out["corrected_bottom_x"]), float(h-1)], dtype=float)
            # Q via theta
            tan_t = math.tan(float(np.clip(map_out["corrected_theta"], -math.radians(45), math.radians(45))))
            Q_c = np.array([float(map_out["corrected_bottom_x"] + h * tan_t), 0.0], dtype=float)
            eff_conf = float(map_out["map_confidence"])
            v, w_ang, info = vs.compute_control(F_corr, dt=dt, confidence=eff_conf, smooth=True if lookahead_map is not None else False)
            info = dict(info)
            info.update({
                "filt_err_x": float(map_out["corrected_X"]),
                "filt_err_theta_deg": float(math.degrees(map_out["corrected_theta"])),
                "filt_width": float(map_out["corrected_width"]),
                "filt_bottom_x": float(map_out["corrected_bottom_x"]),
                "confidence": float(eff_conf),
                "map_status": map_out["map_status"],
                "map_confidence": float(map_out["map_confidence"]),
                "raw_map_innovation_x": float(map_out["raw_map_innovation_x"]),
                "raw_map_innovation_width": float(map_out["raw_map_innovation_width"]),
                "spatial_old": int(map_out["spatial_support_old"]),
                "spatial_new": int(map_out["spatial_support_new"]),
                "raw_err_x": float(raw_dict["raw_err_x"]),
                "raw_err_theta_deg": float(raw_dict["raw_err_theta_deg"]),
                "orig_raw_err_x": float(raw_dict["raw_err_x"]),
                "n_two_sided": int(raw_dict["n_two_sided"]),
                "n_q_accepted": int(raw_dict["n_q_accepted"]),
                "median_width": float(raw_dict["median_width"]),
                "status": map_out["map_status"],
                "is_large": False,
                "pending_count": int(map_out.get("conflict_frames",0)),
                "innovation_x": float(map_out["raw_map_innovation_x"]),
                "innovation_theta_deg": 0.0,
                "innovation_width": float(map_out["raw_map_innovation_width"]),
            })
            P, Q = P_c, Q_c
            F = F_corr
            P_raw_np, Q_raw_np = (np.array(PQ_raw[0], dtype=float), np.array(PQ_raw[1], dtype=float)) if PQ_raw is not None else (None, None)
            # expose map_out as filt_out for uniform handling (overlay)
            filt_out = map_out  # trick for overlay code below to show map info via filt_out-like
        else:
            F = F_raw
            P_raw_np = Q_raw_np = None
            P_raw = Q_raw = None
            if F_raw is not None:
                v, w_ang, info = vs.compute_control(F_raw, dt=dt, confidence=1.0, smooth=False)
                P, Q = PQ_raw
            else:
                v, w_ang, info = 0.0, 0.0, {"err_x": 0, "err_theta_deg": 0}
                P = Q = None
            # add raw diagnostics for uniform CSV
            info = dict(info)
            info.update({
                "filt_err_x": float(info.get("err_x", 0)),
                "filt_err_theta_deg": float(info.get("err_theta_deg", 0)),
                "filt_width": float(raw_dict["raw_width"]),
                "filt_bottom_x": float(raw_dict["raw_bottom_x"]),
                "confidence": 1.0 if raw_dict["has_line"] else 0.0,
                "raw_confidence": 1.0,
                "status": "accepted" if raw_dict["has_line"] else "held_no_line",
                "is_large": False,
                "innovation_x": 0.0,
                "innovation_theta_deg": 0.0,
                "innovation_width": 0.0,
                "pending_count": 0,
                "raw_err_x": float(raw_dict["raw_err_x"]),
                "raw_err_theta_deg": float(raw_dict["raw_err_theta_deg"]),
                "n_two_sided": int(raw_dict["n_two_sided"]),
                "n_q_accepted": int(raw_dict["n_q_accepted"]),
                "median_width": float(raw_dict["median_width"]),
            })
            filt_out = None
            map_out = None

    dt_ms = (time.perf_counter() - t0) * 1000.0

    overlay = None
    if draw:
        overlay = vs.draw_overlay(bgr, P, Q, v, w_ang, info)
        # when filtered, also draw raw line faintly (yellow) and annotation
        if P_raw_np is not None and Q_raw_np is not None and (filt_out is not None or map_out is not None):
            # draw raw line in cyan dashed style (thin)
            try:
                cv2.line(overlay, (int(P_raw_np[0]), int(P_raw_np[1])), (int(Q_raw_np[0]), int(Q_raw_np[1])), (255, 255, 0), 1, cv2.LINE_AA)
                # mark raw bottom small
                cv2.circle(overlay, (int(P_raw_np[0]), int(P_raw_np[1])), 4, (255, 255, 0), 1)
            except Exception:
                pass
        # draw lookahead map corridor as green dashed boxes/center points
        if map_out is not None:
            try:
                # draw map bins as semi-transparent green rectangles for visible future
                h_o, w_o = bgr.shape[:2]
                cov = float(getattr(lookahead_map.params, "vertical_coverage", 0.75)) if lookahead_map is not None else 0.75
                dh = (h_o * cov) / max(1, len(map_out.get("bins", [])) or 10)
                for i, b in enumerate(map_out.get("bins", [])):
                    if b.confidence < 0.15:
                        continue
                    # bin y: bottom strip mu=1 is at y = h - dh (bottom-anchored 3/4 coverage)
                    y_top = int(max(0, h_o - (i+1)*dh))
                    y_bot = int(y_top+dh)
                    x_lo = int(max(0, b.center_x - b.width/2))
                    x_hi = int(min(w_o, b.center_x + b.width/2))
                    # green dashed: thin outline
                    color = (0, 255, 0) if b.confidence > 0.4 else (0, 180, 0)
                    cv2.rectangle(overlay, (x_lo, y_top), (x_hi, y_bot), color, 1)
                    # center dot
                    cv2.circle(overlay, (int(b.center_x), int((y_top+y_bot)/2)), 2, color, -1)
                # mark predicted bottom
                if "pred_bottom_x" in map_out:
                    px = int(map_out["pred_bottom_x"])
                    cv2.circle(overlay, (px, h_o-6), 5, (0, 255, 255), 1)
            except Exception:
                pass
        if filt_out is not None or map_out is not None:
            # second line: confidence/status
            try:
                # choose which status to show: map_status if map exists else temporal status
                if map_out is not None:
                    txt2 = f"map={map_out['map_status']} mc={map_out['map_confidence']:.2f} sp_old={map_out['spatial_support_old']} sp_new={map_out['spatial_support_new']} shift={map_out['shift_px']:.0f}px"
                    cv2.putText(overlay, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
                    if t_filter is not None and filt_out is not None and filt_out is not map_out:
                        txt3 = f"tmp={filt_out['status']} tc={filt_out['confidence']:.2f} innov={filt_out['innovation_x']:+.0f}px | map_innov={map_out['raw_map_innovation_x']:+.0f}px w{map_out['raw_map_innovation_width']:+.0f}px"
                        cv2.putText(overlay, txt3, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        txt3 = f"raw->map innov x{map_out['raw_map_innovation_x']:+.0f}px w{map_out['raw_map_innovation_width']:+.0f}px raw_err{raw_dict['raw_err_x']:+.0f} filt_err{info.get('filt_err_x',0):+.0f}"
                        cv2.putText(overlay, txt3, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1, cv2.LINE_AA)
                else:
                    h_o = overlay.shape[0]
                    txt2 = f"conf={filt_out['confidence']:.2f} status={filt_out['status']} pend={filt_out['pending_count']} innov_x={filt_out['innovation_x']:+.0f}px innov_th={filt_out['innovation_theta_deg']:+.1f}deg"
                    cv2.putText(overlay, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                    txt3 = f"raw_w={raw_dict['median_width']:.0f}px filt_w={filt_out['filt_width']:.0f}px raw_err={raw_dict['raw_err_x']:+.0f} filt_err={filt_out['filt_X']:+.0f}"
                    cv2.putText(overlay, txt3, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
            except Exception:
                pass

    return {
        "res": res,
        "F": F,
        "F_raw": F_raw,
        "P": P,
        "Q": Q,
        "v": v,
        "w": w_ang,
        "info": info,
        "overlay": overlay,
        "time_ms": dt_ms,
        "filt_out": filt_out,
        "raw_dict": raw_dict,
        "map_out": map_out,
    }


def make_side_by_side(nav_bgr, comp_bgr, border=8, max_height=800):
    """Create a single window with nav overlay (left) and MultiROI composite (right) side-by-side."""
    if len(nav_bgr.shape) == 2:
        nav_bgr = cv2.cvtColor(nav_bgr, cv2.COLOR_GRAY2BGR)
    if len(comp_bgr.shape) == 2:
        comp_bgr = cv2.cvtColor(comp_bgr, cv2.COLOR_GRAY2BGR)

    h1, w1 = nav_bgr.shape[:2]
    h2, w2 = comp_bgr.shape[:2]
    target_h = min(max(h1, h2), max_height)
    def scale_to_h(img, th):
        h, w = img.shape[:2]
        if h == th:
            return img
        scale = th / h
        new_w = int(round(w * scale))
        return cv2.resize(img, (new_w, th), interpolation=cv2.INTER_AREA)
    nav_s = scale_to_h(nav_bgr, target_h)
    comp_s = scale_to_h(comp_bgr, target_h)

    label_h = 28
    def add_label(img, text):
        w = img.shape[1]
        bar = np.full((label_h, w, 3), (32, 32, 32), dtype=np.uint8)
        cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    nav_l = add_label(nav_s, "Navigation (filtered red, raw cyan) - star is rover nose")
    comp_l = add_label(comp_s, "MultiROI Composite - binary / mask+ROIs / overlay")

    h_n, w_n = nav_l.shape[:2]
    h_c, w_c = comp_l.shape[:2]
    if h_n != h_c:
        if h_n < h_c:
            pad = np.full((h_c - h_n, w_n, 3), (255, 255, 255), dtype=np.uint8)
            nav_l = np.vstack([nav_l, pad])
        else:
            pad = np.full((h_n - h_c, w_c, 3), (255, 255, 255), dtype=np.uint8)
            comp_l = np.vstack([comp_l, pad])
    border_img = np.full((nav_l.shape[0], border, 3), (255, 255, 255), dtype=np.uint8)
    combined = np.hstack([nav_l, border_img, comp_l])
    return combined


def run_folder(input_path, output_dir, detector, vs, show=False, use_temporal=False, t_filter=None, use_lookahead=False, lookahead_map=None):
    """Process a folder of images. Temporal/lookahead disabled by default for unrelated images."""
    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "*.png")) +
                       glob.glob(os.path.join(input_path, "*.jpg")) +
                       glob.glob(os.path.join(input_path, "*.jpeg")))
    else:
        paths = sorted(glob.glob(input_path))

    if not paths:
        print(f"No images found for {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "nav_commands.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "v_mps", "w_radps", "w_degps",
                         "err_x_px", "err_theta_deg", "filt_err_x", "filt_err_theta_deg",
                         "confidence", "status", "innovation_x", "innovation_theta_deg",
                         "median_width", "filt_width", "n_two_sided", "has_line", "time_ms",
                         "map_status", "map_conf", "map_innov_x", "map_innov_w", "support_old", "support_new"])

        # For folders with temporal/lookahead enabled, keep one state across the sorted list.
        local_filter = t_filter if use_temporal else None
        local_map = lookahead_map if use_lookahead else None
        last_w = 0.0
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            bgr = cv2.imread(p)
            if bgr is None:
                print(f"[WARN] cannot read {p}")
                continue
            out = process_image(bgr, detector, vs, draw=True, t_filter=local_filter, dt=None, last_w=last_w, lookahead_map=local_map)
            last_w = float(out["w"])
            from test_multi_roi import make_composite
            comp = make_composite(bgr, out["res"])
            if out["overlay"] is not None:
                combined = make_side_by_side(out["overlay"], comp)
                cv2.imwrite(os.path.join(output_dir, f"{name}_combined.png"), combined)
                cv2.imwrite(os.path.join(output_dir, f"{name}_nav.png"), out["overlay"])
                cv2.imwrite(os.path.join(output_dir, f"{name}_composite.png"), comp)
            else:
                cv2.imwrite(os.path.join(output_dir, f"{name}_composite.png"), comp)

            has_line = out["res"]["nav_line"] is not None
            nav_w, nav_b = out["res"]["nav_line"] if has_line else (float("nan"), float("nan"))
            info = out["info"]
            writer.writerow([os.path.basename(p),
                             f"{out['v']:.4f}", f"{out['w']:.4f}", f"{math.degrees(out['w']):.2f}",
                             f"{info.get('raw_err_x', info.get('err_x',0)):.1f}", f"{info.get('raw_err_theta_deg', info.get('err_theta_deg',0)):.2f}",
                             f"{info.get('filt_err_x',0):.1f}", f"{info.get('filt_err_theta_deg',0):.2f}",
                             f"{info.get('confidence',1.0):.2f}", info.get("status",""),
                             f"{info.get('innovation_x',0):.1f}", f"{info.get('innovation_theta_deg',0):.2f}",
                             f"{info.get('median_width',0):.0f}", f"{info.get('filt_width',0):.0f}",
                             int(info.get("n_two_sided",0)), int(has_line), f"{out['time_ms']:.1f}",
                             info.get("map_status",""), f"{info.get('map_confidence',0):.2f}",
                             f"{info.get('raw_map_innovation_x',0):.1f}", f"{info.get('raw_map_innovation_width',0):.1f}",
                             int(info.get("spatial_old",0)), int(info.get("spatial_new",0))])
            map_txt = f" map={info.get('map_status','')}" if info.get("map_status") else ""
            print(f"[{name}] v={out['v']:.2f} w={math.degrees(out['w']):.1f}deg/s raw_err={info.get('raw_err_x',0):+.0f} filt_err={info.get('filt_err_x',0):+.0f} conf={info.get('confidence',0):.2f} status={info.get('status','')}{map_txt} {out['time_ms']:.1f}ms")
            if show and out["overlay"] is not None:
                cv2.imshow("MultiROI Navigation", out["overlay"])
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
    print(f"\nSaved to {os.path.abspath(output_dir)}")
    print(f"CSV: {csv_path}")


def run_video(input_path, output_dir, detector, vs, show=False, use_temporal=True, t_filter=None, loop=False, use_lookahead=True, lookahead_map=None):
    """Process a video file or camera index. Temporal+lookahead enabled by default.

    If loop=True and input is a video file, the video rewinds to the first
    frame and continues forever until the user quits (q) or interrupts
    (Ctrl-C). Camera streams already run forever and ignore loop.
    """
    try:
        cam_idx = int(input_path)
        cap = cv2.VideoCapture(cam_idx)
        is_camera = True
        name = f"camera{cam_idx}"
    except ValueError:
        cap = cv2.VideoCapture(input_path)
        is_camera = False
        name = os.path.splitext(os.path.basename(input_path))[0]

    if not cap.isOpened():
        print(f"Cannot open video/camera {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{name}_nav.csv")
    writer = None
    writer_w = writer_h = None
    out_path = os.path.join(output_dir, f"{name}_nav.mp4") if not is_camera else None
    if loop and not is_camera:
        print(f"[LOOP] enabled — video will rewind and run until stopped (q / Ctrl-C)")

    # One continuous temporal+lookahead state for the whole video
    if use_temporal and t_filter is None:
        # create with image shape hint – will be corrected on first frame if needed
        t_filter = TemporalNavigationFilter(TemporalFilterParams(image_width=vs.params.width, image_height=vs.params.height, n_strips=detector.n))
    elif not use_temporal:
        t_filter = None
    if use_lookahead and lookahead_map is None:
        lookahead_map = LookaheadCorridorMap(LookaheadParams(image_width=vs.params.width, image_height=vs.params.height, n_bins=detector.n))
    elif not use_lookahead:
        lookahead_map = None

    # ensure vs smoother is reset for new sequence
    vs.reset_smoother()

    with open(csv_path, "w", newline="") as f:
        csvw = csv.writer(f)
        csvw.writerow(["frame", "timestamp", "v_mps", "w_radps",
                       "raw_err_x_px", "raw_err_theta_deg", "filt_err_x_px", "filt_err_theta_deg",
                       "confidence", "status", "innovation_x", "innovation_theta_deg",
                       "median_width", "filt_width", "n_two_sided", "has_line", "pending_count",
                       "map_status", "map_conf", "map_innov_x", "map_innov_w", "support_old", "support_new", "shift_px"])

        frame_idx = 0
        loop_count = 0
        last_time = time.perf_counter()
        last_w = 0.0
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    if loop and not is_camera:
                        loop_count += 1
                        print(f"[LOOP] end of video — rewinding (loop {loop_count})")
                        # rewind: try seek, else reopen
                        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if not ok:
                            cap.release()
                            cap = cv2.VideoCapture(input_path)
                            if not cap.isOpened():
                                print("[LOOP] failed to reopen video")
                                break
                        # reset temporal+lookahead state across discontinuity so the
                        # last->first frame jump is not treated as a perception spike
                        if t_filter is not None:
                            t_filter.reset()
                        if lookahead_map is not None:
                            lookahead_map.reset()
                        vs.reset_smoother()
                        last_w = 0.0
                        last_time = time.perf_counter()
                        ret, bgr = cap.read()
                        if not ret:
                            print("[LOOP] rewound but still no frame, stopping")
                            break
                    else:
                        break
                now = time.perf_counter()
                dt = now - last_time
                dt = max(1e-3, min(0.5, dt))
                last_time = now

                # update filter/map image size on first frame if mismatch
                if t_filter is not None and frame_idx == 0:
                    h, w = bgr.shape[:2]
                    t_filter.params.image_width = w
                    t_filter.params.image_height = h
                if lookahead_map is not None and frame_idx == 0:
                    h, w = bgr.shape[:2]
                    lookahead_map.params.image_width = w
                    lookahead_map.params.image_height = h

                out = process_image(bgr, detector, vs, draw=True, t_filter=t_filter, dt=dt, last_w=last_w, lookahead_map=lookahead_map)
                last_w = float(out["w"])
                info = out["info"]
                csvw.writerow([frame_idx, time.time(),
                               f"{out['v']:.4f}", f"{out['w']:.4f}",
                               f"{info.get('raw_err_x',0):.1f}", f"{info.get('raw_err_theta_deg',0):.2f}",
                               f"{info.get('filt_err_x',0):.1f}", f"{info.get('filt_err_theta_deg',0):.2f}",
                               f"{info.get('confidence',0):.2f}", info.get("status",""),
                               f"{info.get('innovation_x',0):.1f}", f"{info.get('innovation_theta_deg',0):.2f}",
                               f"{info.get('median_width',0):.0f}", f"{info.get('filt_width',0):.0f}",
                               int(info.get("n_two_sided",0)), int(out["res"]["nav_line"] is not None),
                               int(info.get("pending_count",0)),
                               info.get("map_status",""), f"{info.get('map_confidence',0):.2f}",
                               f"{info.get('raw_map_innovation_x',0):.1f}", f"{info.get('raw_map_innovation_width',0):.1f}",
                               int(info.get("spatial_old",0)), int(info.get("spatial_new",0)), f"{info.get('map_shift_px',0):.1f}"])
                from test_multi_roi import make_composite
                comp = make_composite(bgr, out["res"])
                combined = make_side_by_side(out["overlay"] if out["overlay"] is not None else bgr, comp) if out["overlay"] is not None else comp

                if writer is None and not is_camera:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    if fps < 1 or fps > 120:
                        fps = 20.0
                    writer_h, writer_w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (writer_w, writer_h))
                    if not writer.isOpened():
                        print(f"[WARN] VideoWriter failed for {writer_w}x{writer_h} fps {fps}")

                if writer is not None:
                    if combined.shape[1] != writer_w or combined.shape[0] != writer_h:
                        combined = cv2.resize(combined, (writer_w, writer_h), interpolation=cv2.INTER_AREA)
                    writer.write(combined)
                if show:
                    preview = combined
                    if preview.shape[1] > 1280:
                        scale = 1280 / preview.shape[1]
                        preview = cv2.resize(preview, (1280, int(preview.shape[0]*scale)), interpolation=cv2.INTER_AREA)
                    cv2.imshow("MultiROI Navigation - Combined (nav | composite) - q to quit", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                if frame_idx % 30 == 0:
                    map_txt = f" map={info.get('map_status','')} mc={info.get('map_confidence',0):.2f}" if info.get('map_status') else ""
                    print(f"Frame {frame_idx}: v={out['v']:.2f} w={math.degrees(out['w']):.1f} deg/s raw_err={info.get('raw_err_x',0):+.0f} filt_err={info.get('filt_err_x',0):+.0f} conf={info.get('confidence',0):.2f} {info.get('status','')}{map_txt}")
        except KeyboardInterrupt:
            print("\n[LOOP] interrupted by user (Ctrl-C)")

    cap.release()
    if writer:
        writer.release()
        print(f"Video saved to {out_path} ({frame_idx} frames, {loop_count} loops)")
    print(f"Saved to {output_dir}")
    print(f"CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="MultiROI furrow following navigation")
    parser.add_argument("--input", default="../../Photos", help="Image folder, glob, video file or camera index")
    parser.add_argument("--output", default="./nav_output", help="Output directory")
    parser.add_argument("--video", action="store_true", help="Treat input as video/camera")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--loop", action="store_true", help="Loop video forever until stopped (q / Ctrl-C); only for --video files, camera already loops")
    parser.add_argument("--line", action="store_true", help="Use straight line instead of smoothing spline (default is smoothing spline / curvilinear fit)")
    parser.add_argument("--width", type=int, default=640, help="Expected image width for normalization")
    parser.add_argument("--height", type=int, default=480, help="Expected height")
    parser.add_argument("--vf", type=float, default=0.20, help="Forward speed m/s")
    parser.add_argument("--wmax", type=float, default=0.50, help="Max angular rad/s")
    parser.add_argument("--lambdax", type=float, default=10.0, help="Lateral gain")
    parser.add_argument("--lambdatheta", type=float, default=1.0, help="Heading gain")
    parser.add_argument("--n_strips", type=int, default=10) 
    parser.add_argument("--l_frac", type=float, default=0.05)
    parser.add_argument("--index", choices=("raw","normalized"), default="raw")
    # --- temporal / smoothing flags ---
    parser.add_argument("--temporal", action="store_true", help="Enable temporal filtering for folder mode (video enables by default)")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal filtering (even for video)")
    parser.add_argument("--persist-frames", type=int, default=4, help="Frames of persistent large innovation before accepting (default 4)")
    parser.add_argument("--max-jump-frac", type=float, default=0.10, help="Max bottom jump as fraction of image width before gating (default 0.10)")
    parser.add_argument("--max-jump-width-frac", type=float, default=0.45, help="Max bottom jump as fraction of corridor width (default 0.45)")
    parser.add_argument("--max-heading-jump", type=float, default=12.0, help="Max heading jump deg per frame (default 12)")
    parser.add_argument("--max-width-change", type=float, default=0.30, help="Max corridor width change fraction per frame (default 0.30)")
    parser.add_argument("--w-alpha", type=float, default=0.35, help="Low-pass alpha for w (default 0.35)")
    parser.add_argument("--max-w-rate", type=float, default=1.2, help="Max w rate rad/s per sec (default 1.2)")
    parser.add_argument("--w-deadband", type=float, default=0.02, help="Deadband for w in rad/s (default 0.02)")
    parser.add_argument("--alpha-x", type=float, default=0.35, help="EMA alpha for bottom_x (default 0.35)")
    parser.add_argument("--alpha-theta", type=float, default=0.35, help="EMA alpha for heading (default 0.35)")
    # --- lookahead / visible-future map flags ---
    parser.add_argument("--lookahead", action="store_true", help="Enable lookahead corridor map for folder mode (video enables by default)")
    parser.add_argument("--no-lookahead", action="store_true", help="Disable lookahead corridor map (even for video)")
    parser.add_argument("--lookahead-bins", type=int, default=10, help="Number of vertical bins for lookahead map (default 10 = strips)")
    parser.add_argument("--lookahead-shift-px", type=float, default=14.0, help="Forward motion pixels per frame for map shift (default 14)")
    parser.add_argument("--lookahead-center-gate", type=float, default=0.35, help="Max center gate as fraction of width for map consistency (default 0.35)")
    parser.add_argument("--lookahead-width-gate", type=float, default=0.35, help="Max width gate fraction for map consistency (default 0.35)")
    parser.add_argument("--lookahead-accept-frames", type=int, default=4, help="Frames of spatial support before accepting new corridor (default 4)")
    parser.add_argument("--lookahead-accept-bins", type=int, default=3, help="Upper bins supporting new geometry to accept (default 3)")
    parser.add_argument("--lookahead-conf-decay", type=float, default=0.97, help="Per-frame confidence decay for map (default 0.97)")
    parser.add_argument("--lookahead-overlay", action=argparse.BooleanOptionalAction, default=True, help="Draw lookahead map overlay (green boxes)")
    parser.add_argument("--ignore-initial", type=int, default=0, help="Ignore the first N bottom ROI boxes/midpoints (e.g. 3) for nav; robot follows center star until upper good evidence (approach phase, noisy entry)")
    parser.add_argument("--vertical-coverage", type=float, default=0.75, help="Vertical fraction of image covered by ROIs from bottom (0.75 = bottom 3/4, top 1/4 has no boxes; default 0.75 per request)")
    parser.add_argument("--roi-draw-frac", type=float, default=1.0, help="Height fraction of each white ROI box relative to strip height (1.0 = full strip, 0.75 = 3/4 height centered; default 1.0)")
    args = parser.parse_args()

    # Detector with default robust params (from test_multi_roi.py)
    # --line = straight line only, default = smoothing spline (nav_curve=True)
    detector = MultiROIDetector(n_strips=args.n_strips, l_frac=args.l_frac, index=args.index,
                                nav_curve=(not args.line),
                                ignore_initial=args.ignore_initial,
                                vertical_coverage=args.vertical_coverage,
                                roi_draw_frac=args.roi_draw_frac)
    if args.line:
        print("[LINE] straight-line mode enabled (smoothing spline disabled)")
    else:
        print("[LINE] smoothing spline (curvilinear) mode enabled (default); use --line for straight line")

    # Visual servoing params
    vs_params = MRVSParams(
        width=args.width, height=args.height,
        vf_des=args.vf, w_max=args.wmax,
        lambda_x=args.lambdax, lambda_theta=args.lambdatheta,
        w_alpha=args.w_alpha, max_w_rate=args.max_w_rate, w_deadband=args.w_deadband,
    )
    vs = MultiROIVS(vs_params)

    os.makedirs(args.output, exist_ok=True)

    use_temporal = False
    t_filter = None
    if args.no_temporal:
        use_temporal = False
    elif args.video:
        use_temporal = True
    elif args.temporal:
        use_temporal = True

    if use_temporal:
        t_params = TemporalFilterParams(
            image_width=args.width, image_height=args.height,
            max_bottom_jump_frac=args.max_jump_frac,
            max_bottom_jump_width_frac=args.max_jump_width_frac,
            max_heading_jump_deg=args.max_heading_jump,
            max_width_change_frac=args.max_width_change,
            persist_frames=args.persist_frames,
            alpha_x=args.alpha_x, alpha_theta=args.alpha_theta,
            n_strips=args.n_strips,
        )
        t_filter = TemporalNavigationFilter(t_params)

    # --- lookahead map ---
    use_lookahead = False
    lookahead_map = None
    if args.no_lookahead:
        use_lookahead = False
    elif args.video:
        use_lookahead = True
    elif args.lookahead:
        use_lookahead = True

    if use_lookahead:
        l_params = LookaheadParams(
            image_width=args.width, image_height=args.height,
            n_bins=args.lookahead_bins,
            shift_px=args.lookahead_shift_px,
            max_center_gate_frac=args.lookahead_center_gate,
            max_width_gate_frac=args.lookahead_width_gate,
            accept_frames=args.lookahead_accept_frames,
            accept_bins=args.lookahead_accept_bins,
            conf_decay=args.lookahead_conf_decay,
            ignore_initial=args.ignore_initial,
            vertical_coverage=args.vertical_coverage,
        )
        lookahead_map = LookaheadCorridorMap(l_params)
        print(f"[LOOKAHEAD] enabled (shift {args.lookahead_shift_px:.1f}px, gates {args.lookahead_center_gate:.2f}/{args.lookahead_width_gate:.2f}, accept {args.lookahead_accept_frames}f/{args.lookahead_accept_bins}b)")
    else:
        print("[LOOKAHEAD] disabled")
    if args.ignore_initial:
        print(f"[IGNORE] ignoring first {args.ignore_initial} bottom ROI boxes/midpoints for nav (approach phase) – following center star when upper insufficient")

    if args.loop and not args.video:
        print("[WARN] --loop only applies with --video; ignoring for folder mode")
    if args.video:
        run_video(args.input, args.output, detector, vs, show=args.show, use_temporal=use_temporal, t_filter=t_filter, loop=args.loop, use_lookahead=use_lookahead, lookahead_map=lookahead_map)
    else:
        run_folder(args.input, args.output, detector, vs, show=args.show, use_temporal=use_temporal, t_filter=t_filter, use_lookahead=use_lookahead, lookahead_map=lookahead_map)


if __name__ == "__main__":
    main()

"""
ROS2 Node Wrapper (example, not executed in offline mode):

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class MultiROINavNode(Node):
    def __init__(self):
        super().__init__('multiroi_nav')
        self.bridge = CvBridge()
        self.detector = MultiROIDetector()
        self.vs = MultiROIVS()
        self.t_filter = TemporalNavigationFilter()
        self.sub = self.create_subscription(Image, '/front/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        out = process_image(bgr, self.detector, self.vs, draw=False, t_filter=self.t_filter, dt=0.05, last_w=self.t_filter._last_w)
        twist = Twist()
        twist.linear.x = float(out['v'])
        twist.angular.z = float(out['w'])
        self.pub.publish(twist)

# To run: colcon build, source install/setup.bash, ros2 run <pkg> multoroi_nav
"""
