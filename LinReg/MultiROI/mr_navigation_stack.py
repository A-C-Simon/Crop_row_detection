"""
MultiROI Navigation Stack - Rover Control
==========================================
Implements the full navigation stack for a rover using the MultiROI
furrow detector. This is the "rest from there as to an actual rover
moving" after the navigation line is available.

The stack consists of:
1. Perception: MultiROI detector (already available, furrow centre line)
2. State Estimation: Visual features (X, Theta) from navigation line
3. Control: Visual servoing controller that outputs v,w
4. Actuation: Rover kinematics (differential drive) update

This is designed to be used both offline (on Photos) and online
(with ROS2 or with a simulated rover in Gazebo).

The rover is modeled as a differential drive with:
  x_dot = v * cos(theta)
  y_dot = v * sin(theta)
  theta_dot = w

Where (x,y,theta) is the rover pose in the world frame.

For the visual servoing, we use the same interaction matrix approach
as in ExG/agribot_vs.cpp but simplified for the furrow.

Author: adapted from ExG navigation stack for MultiROI furrow
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from test_multi_roi import MultiROIDetector
from mr_vs import MultiROIVS, MRVSParams, wrapToPi
from temporal_filter import TemporalNavigationFilter, TemporalFilterParams
from lookahead_corridor_map import LookaheadCorridorMap, LookaheadParams
import time as _time


@dataclass
class RoverPose:
    x: float = 0.0  # meters, forward is +x in world? Use standard
    y: float = 0.0
    theta: float = 0.0  # radians, 0 is along +x

    def copy(self):
        return RoverPose(self.x, self.y, self.theta)


@dataclass
class RoverCommand:
    v: float  # linear m/s
    w: float  # angular rad/s
    err_x: float
    err_theta: float
    has_line: bool


class RoverSimulator:
    """Simple differential drive rover simulator."""

    def __init__(self, initial_pose: Optional[RoverPose] = None, dt: float = 0.1):
        self.pose = initial_pose or RoverPose()
        self.dt = dt
        self.path: List[RoverPose] = [self.pose.copy()]
        self.commands: List[RoverCommand] = []

    def update(self, v: float, w: float):
        """Update pose using differential drive kinematics."""
        # Simple Euler integration
        self.pose.x += v * math.cos(self.pose.theta) * self.dt
        self.pose.y += v * math.sin(self.pose.theta) * self.dt
        self.pose.theta = wrapToPi(self.pose.theta + w * self.dt)
        self.path.append(self.pose.copy())
        return self.pose.copy()

    def get_path_array(self):
        """Return path as numpy array for plotting."""
        return np.array([[p.x, p.y, p.theta] for p in self.path])


class MultiROINavigationStack:
    """
    Full navigation stack: perception + control + actuation.

    Usage:
        detector = MultiROIDetector()
        vs = MultiROIVS()
        nav_stack = MultiROINavigationStack(detector, vs)
        # For each image:
        cmd, overlay = nav_stack.step(bgr_image)
        # cmd is RoverCommand with v,w
        # Update simulator or publish to /cmd_vel
        simulator.update(cmd.v, cmd.w)

    Temporal robustness: a TemporalNavigationFilter sits between
    detector and controller.  Raw and filtered states are both
    available via `self.last_raw` / `self.last_filt`.
    """

    def __init__(self, detector: Optional[MultiROIDetector] = None,
                 vs: Optional[MultiROIVS] = None,
                 simulator: Optional[RoverSimulator] = None,
                 use_temporal: bool = True,
                 t_params: Optional[TemporalFilterParams] = None,
                 use_lookahead: bool = True,
                 l_params: Optional[LookaheadParams] = None,
                 ignore_initial: int = 0):
        # handle ignore_initial passed via detector/l_params as well
        self.ignore_initial = int(ignore_initial)
        if detector is None:
            detector = MultiROIDetector(ignore_initial=self.ignore_initial)
        elif self.ignore_initial:
            detector.ignore_initial = self.ignore_initial
        self.detector = detector
        self.vs = vs or MultiROIVS()
        self.simulator = simulator or RoverSimulator()
        self.step_count = 0
        self.last_cmd = RoverCommand(0, 0, 0, 0, False)
        self.use_temporal = use_temporal
        self.t_filter: Optional[TemporalNavigationFilter] = None
        if use_temporal:
            tp = t_params or TemporalFilterParams(
                image_width=self.vs.params.width,
                image_height=self.vs.params.height,
                n_strips=self.detector.n)
            # will be lazily corrected to actual image shape on first step
            self.t_filter = TemporalNavigationFilter(tp)
        self.use_lookahead = use_lookahead
        self.lookahead_map: Optional[LookaheadCorridorMap] = None
        if use_lookahead:
            lp = l_params or LookaheadParams(
                image_width=self.vs.params.width,
                image_height=self.vs.params.height,
                n_bins=self.detector.n,
                ignore_initial=self.ignore_initial,
                vertical_coverage=float(getattr(self.detector, "vertical_coverage", 0.75)))
            # ensure sync
            lp.ignore_initial = self.ignore_initial
            lp.vertical_coverage = float(getattr(self.detector, "vertical_coverage", 0.75))
            self.lookahead_map = LookaheadCorridorMap(lp)
        self.last_raw: Optional[Dict] = None
        self.last_filt: Optional[Dict] = None
        self.last_map: Optional[Dict] = None
        self._prev_time: Optional[float] = None

    def step(self, bgr, draw=True):
        """
        One navigation step: detect + lookahead map + temporal filter + control.

        Returns (cmd, overlay, res) where
          cmd is RoverCommand
          overlay is BGR image with debug overlay (filtered red, raw cyan, map green)
          res is the raw detector result dict
        Also updates self.last_raw / self.last_filt / self.last_map for diagnostics.
        """
        h, w = bgr.shape[:2]
        # Detect with lookahead prior so ROI boxes / midpoints stay steady through gaps
        lookahead_prior = None
        if self.lookahead_map is not None:
            try:
                pred_bins = self.lookahead_map.get_prediction((h, w))
                lookahead_prior = {"pred_bins": pred_bins}
            except Exception:
                lookahead_prior = None
        res = self.detector.detect(bgr, lookahead_prior=lookahead_prior)
        crop_offset = res.get("crop_offset", (0, 0))
        nav_line = res.get("nav_line")
        nav_curve = res.get("nav_curve")

        # Visual features - raw
        F_raw, PQ_raw = self.vs.nav_line_to_feature(nav_line, nav_curve, crop_offset, (h, w))

        # timing for dt / rate-limit prediction
        now = _time.perf_counter()
        if self._prev_time is None:
            dt = 0.05
        else:
            dt = now - self._prev_time
            dt = max(1e-3, min(0.5, dt))
        self._prev_time = now

        # Correct filter/map image sizes on first frame if mismatch
        if self.t_filter is not None and self.step_count == 0:
            self.t_filter.params.image_width = w
            self.t_filter.params.image_height = h
        if self.lookahead_map is not None and self.step_count == 0:
            self.lookahead_map.params.image_width = w
            self.lookahead_map.params.image_height = h

        # build raw dict
        has_raw = F_raw is not None and PQ_raw is not None
        if has_raw:
            P_raw, Q_raw = PQ_raw
            raw_bottom_x = float(P_raw[0])
            raw_theta = float(F_raw[2])
            raw_X = float(F_raw[0])
            raw_width = float(res.get("median_width", 0) or res.get("bottom_width", 0) or 120.0)
        else:
            raw_bottom_x = w / 2.0
            raw_theta = 0.0
            raw_X = 0.0
            raw_width = float(res.get("median_width", 0) or 120.0)
            if not math.isfinite(raw_width) or raw_width < 5:
                raw_width = 120.0
        raw_dict = {
            "raw_bottom_x": float(raw_bottom_x),
            "raw_theta": float(raw_theta),
            "raw_width": float(raw_width),
            "raw_X": float(raw_X),
            "raw_err_x": float(raw_X),
            "raw_err_theta_deg": float(math.degrees(raw_theta)) if has_raw else 0.0,
            "has_line": bool(has_raw),
            "n_two_sided": int(res.get("n_two_sided", 0)),
            "n_q_accepted": int(res.get("n_q_accepted", 0)),
            "n_q_rejected": int(res.get("n_q_rejected", 0)),
            "weed_pressure": float(res.get("weed_pressure", 0.0)),
            "median_width": float(res.get("median_width", raw_width)),
            "bottom_width": float(res.get("bottom_width", raw_width)),
        }

        # --- lookahead map (spatial) ---
        map_out = None
        filter_input = raw_dict
        if self.lookahead_map is not None:
            strip_profile = res.get("strip_profile", [])
            map_out = self.lookahead_map.update(strip_profile, res, (h, w), dt=dt, last_w=self.last_cmd.w)
            # corrected dict for temporal filter
            filter_input = {
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
                "orig_raw_bottom_x": float(raw_dict["raw_bottom_x"]),
                "orig_raw_width": float(raw_dict["raw_width"]),
            }
            self.last_map = map_out
        else:
            self.last_map = None

        # --- temporal filter (scalar) ---
        if self.t_filter is not None:
            filt_out = self.t_filter.update(filter_input, dt=dt, last_w=self.last_cmd.w)
            F = filt_out["filt_F"]
            P = np.array(filt_out["filt_P"], dtype=float)
            Q = np.array(filt_out["filt_Q"], dtype=float)
            # combine confidences if map present
            eff_conf = float(filt_out["confidence"])
            if map_out is not None:
                eff_conf = float(min(eff_conf, map_out["map_confidence"] * 0.6 + 0.4))
                if map_out["map_status"] in ("map_hold", "occlusion_hold", "map_reject", "map_pending"):
                    eff_conf = float(min(eff_conf, map_out["map_confidence"] * 0.8 + 0.2))
            v, w_ang, info = self.vs.compute_control(F, dt=dt, confidence=eff_conf, smooth=True)
            info = dict(info)
            info.update({
                "filt_err_x": float(filt_out["filt_X"]),
                "filt_err_theta_deg": float(filt_out["filt_theta_deg"]),
                "confidence": float(eff_conf),
                "status": filt_out["status"],
                "innovation_x": float(filt_out["innovation_x"]),
                "innovation_theta_deg": float(filt_out["innovation_theta_deg"]),
                "pending_count": int(filt_out["pending_count"]),
                "raw_err_x": float(raw_dict["raw_err_x"]),
                "raw_err_theta_deg": float(raw_dict["raw_err_theta_deg"]),
            })
            if map_out is not None:
                info.update({
                    "map_status": map_out["map_status"],
                    "map_confidence": float(map_out["map_confidence"]),
                    "raw_map_innovation_x": float(map_out["raw_map_innovation_x"]),
                    "raw_map_innovation_width": float(map_out["raw_map_innovation_width"]),
                    "spatial_old": int(map_out["spatial_support_old"]),
                    "spatial_new": int(map_out["spatial_support_new"]),
                })
            has_line = bool(filt_out["has_line"] or has_raw)
            self.last_raw = raw_dict
            self.last_filt = filt_out
        else:
            # legacy or lookahead-only path
            if map_out is not None:
                # use corrected map directly
                F = np.array([float(map_out["corrected_X"]), h/4.0, float(map_out["corrected_theta"])], dtype=float)
                P = np.array([float(map_out["corrected_bottom_x"]), float(h-1)], dtype=float)
                tan_t = math.tan(float(np.clip(map_out["corrected_theta"], -math.radians(45), math.radians(45))))
                Q = np.array([float(map_out["corrected_bottom_x"] + h * tan_t), 0.0], dtype=float)
                eff_conf = float(map_out["map_confidence"])
                v, w_ang, info = self.vs.compute_control(F, dt=dt, confidence=eff_conf, smooth=True)
                info = dict(info)
                info.update({
                    "filt_err_x": float(map_out["corrected_X"]),
                    "filt_err_theta_deg": float(math.degrees(map_out["corrected_theta"])),
                    "confidence": float(eff_conf),
                    "status": map_out["map_status"],
                    "map_status": map_out["map_status"],
                    "map_confidence": float(map_out["map_confidence"]),
                    "raw_map_innovation_x": float(map_out["raw_map_innovation_x"]),
                    "raw_map_innovation_width": float(map_out["raw_map_innovation_width"]),
                    "spatial_old": int(map_out["spatial_support_old"]),
                    "spatial_new": int(map_out["spatial_support_new"]),
                    "raw_err_x": float(raw_dict["raw_err_x"]),
                    "raw_err_theta_deg": float(raw_dict["raw_err_theta_deg"]),
                })
                has_line = True
                self.last_raw = raw_dict
                self.last_filt = None
                filt_out = None
            else:
                self.last_raw = None
                self.last_filt = None
                self.last_map = None
                if F_raw is not None:
                    v, w_ang, info = self.vs.compute_control(F_raw, dt=dt, confidence=1.0, smooth=False)
                    P, Q = PQ_raw[0], PQ_raw[1]
                    P = np.array(P, dtype=float); Q = np.array(Q, dtype=float)
                    has_line = True
                else:
                    v, w_ang, info = 0.0, 0.0, {"err_x": 0, "err_theta_deg": 0}
                    P = Q = None
                    has_line = False
                info = dict(info)
                info.setdefault("confidence", 1.0 if has_line else 0.0)
                info.setdefault("status", "accepted" if has_line else "held_no_line")
                filt_out = None

        cmd = RoverCommand(v=v, w=w_ang,
                           err_x=info.get("filt_err_x", info.get("err_x", 0)),
                           err_theta=info.get("filt_err_theta_deg", info.get("err_theta_deg", 0)),
                           has_line=has_line)

        # Update simulator
        self.simulator.update(cmd.v, cmd.w)
        self.last_cmd = cmd
        self.step_count += 1

        # Draw overlay
        overlay = None
        if draw:
            # Draw filtered line (red) via vs overlay
            if P is not None and Q is not None:
                overlay = self.vs.draw_overlay(bgr, tuple(P), tuple(Q), v, w_ang, info)
            else:
                overlay = bgr.copy()
                cv2.putText(overlay, f"v={v:.2f} w={math.degrees(w_ang):.1f}deg/s no line", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            # When temporal active, also draw raw faint cyan
            if F_raw is not None and PQ_raw is not None:
                try:
                    Pr, Qr = PQ_raw
                    cv2.line(overlay, (int(Pr[0]), int(Pr[1])), (int(Qr[0]), int(Qr[1])), (255, 255, 0), 1, cv2.LINE_AA)
                except Exception:
                    pass
            # draw map bins if present
            if self.lookahead_map is not None and self.last_map is not None:
                try:
                    h_o, w_o = bgr.shape[:2]
                    dh = h_o / max(1, self.lookahead_map.params.n_bins)
                    for i, b in enumerate(self.last_map.get("bins", [])):
                        if b.confidence < 0.15:
                            continue
                        y_top = int(max(0, h_o - (i+1)*dh))
                        y_bot = int(y_top+dh)
                        x_lo = int(max(0, b.center_x - b.width/2))
                        x_hi = int(min(w_o, b.center_x + b.width/2))
                        color = (0, 255, 0) if b.confidence > 0.4 else (0, 180, 0)
                        cv2.rectangle(overlay, (x_lo, y_top), (x_hi, y_bot), color, 1)
                        cv2.circle(overlay, (int(b.center_x), int((y_top+y_bot)/2)), 2, color, -1)
                    if "pred_bottom_x" in self.last_map:
                        cv2.circle(overlay, (int(self.last_map["pred_bottom_x"]), h_o-6), 5, (0, 255, 255), 1)
                except Exception:
                    pass
            if self.t_filter is not None and self.last_filt is not None:
                try:
                    filt = self.last_filt
                    txt2 = f"conf={filt['confidence']:.2f} {filt['status']} pend={filt['pending_count']} innov_x={filt['innovation_x']:+.0f}"
                    cv2.putText(overlay, txt2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass
            if self.lookahead_map is not None and self.last_map is not None:
                try:
                    m = self.last_map
                    txt3 = f"map={m['map_status']} mc={m['map_confidence']:.2f} old={m['spatial_support_old']} new={m['spatial_support_new']}"
                    cv2.putText(overlay, txt3, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                except Exception:
                    pass
            pose = self.simulator.pose
            y_txt = 84 if self.lookahead_map is not None else (66 if self.t_filter is not None else 60)
            txt2 = f"Pose x={pose.x:.2f} y={pose.y:.2f} th={math.degrees(pose.theta):.1f}deg | Step {self.step_count}"
            cv2.putText(overlay, txt2, (10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        return cmd, overlay, res

    def reset(self, pose: Optional[RoverPose] = None):
        """Reset the stack and simulator."""
        self.simulator = RoverSimulator(initial_pose=pose)
        self.step_count = 0
        self.last_cmd = RoverCommand(0, 0, 0, 0, False)
        self.last_raw = None
        self.last_filt = None
        self.last_map = None
        self._prev_time = None
        if self.t_filter is not None:
            self.t_filter.reset()
        if self.lookahead_map is not None:
            self.lookahead_map.reset()
        self.vs.reset_smoother()


def demo_offline(photos_dir="../../Photos", output_dir="./nav_stack_output"):
    """Demo: run navigation stack on all Photos as if they were sequential frames."""
    import glob
    import os
    import csv

    detector = MultiROIDetector()
    vs = MultiROIVS()
    nav_stack = MultiROINavigationStack(detector, vs)

    paths = sorted(glob.glob(os.path.join(photos_dir, "*.png")) +
                   glob.glob(os.path.join(photos_dir, "*.jpg")))
    if not paths:
        print(f"No images in {photos_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "rover_path.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "filename", "v", "w_deg", "err_x", "err_theta_deg",
                         "pose_x", "pose_y", "pose_theta_deg", "has_line"])

        for i, p in enumerate(paths):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            name = os.path.basename(p)
            cmd, overlay, res = nav_stack.step(bgr, draw=True)
            # Save overlay
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(name)[0]}_nav.png"), overlay)
            pose = nav_stack.simulator.pose
            writer.writerow([i, name, f"{cmd.v:.3f}", f"{math.degrees(cmd.w):.2f}",
                             f"{cmd.err_x:.1f}", f"{cmd.err_theta:.2f}",
                             f"{pose.x:.3f}", f"{pose.y:.3f}", f"{math.degrees(pose.theta):.1f}",
                             int(cmd.has_line)])
            print(f"[{i}] {name}: v={cmd.v:.2f} w={math.degrees(cmd.w):.1f}deg err_x={cmd.err_x:.0f} err_theta={cmd.err_theta:.1f} -> pose x={pose.x:.2f} y={pose.y:.2f} th={math.degrees(pose.theta):.1f}")

    # Save path plot
    try:
        import matplotlib.pyplot as plt
        path_arr = nav_stack.simulator.get_path_array()
        plt.figure(figsize=(10, 6))
        plt.plot(path_arr[:, 0], path_arr[:, 1], "-o", markersize=3, label="Rover path")
        plt.xlabel("X (m) forward")
        plt.ylabel("Y (m) lateral")
        plt.title("Rover Path - MultiROI Furrow Following (Photos as sequence)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "rover_path.png"), dpi=150)
        plt.close()
        print(f"Path plot saved to {output_dir}/rover_path.png")
    except Exception as e:
        print(f"Could not plot path: {e}")

    print(f"\nDone. Outputs in {os.path.abspath(output_dir)}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MultiROI Navigation Stack Demo")
    parser.add_argument("--input", default="../../Photos", help="Photos folder")
    parser.add_argument("--output", default="./nav_stack_output", help="Output folder")
    args = parser.parse_args()
    demo_offline(args.input, args.output)
