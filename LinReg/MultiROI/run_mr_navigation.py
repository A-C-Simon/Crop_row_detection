#!/usr/bin/env python3
"""
MultiROI Navigation Runner - Furrow following visual servoing
==============================================================
Runs the MultiROI detector and the visual servoing controller on a
folder of images or a video, producing control commands for a rover.

This is the "rest of the navigation stack" after the navigation line
is already available (furrow centre).  It implements the actual rover
movement: converts the image line into v,w velocities.

Usage:
  # On Photos (offline, for analysis)
  python3 run_mr_navigation.py --input ../../Photos --output ./nav_output

  # On video
  python3 run_mr_navigation.py --input /path/to/video.mp4 --output ./nav_output --video

  # Live camera (ROS-free, OpenCV capture)
  python3 run_mr_navigation.py --input 0 --output ./nav_output --video --show

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


def process_image(bgr, detector, vs, draw=True):
    """Run detection + visual servoing on one BGR image."""
    t0 = time.perf_counter()
    res = detector.detect(bgr)
    h, w = bgr.shape[:2]
    crop_offset = res.get("crop_offset", (0, 0))

    # Get navigation line/curve
    nav_line = res.get("nav_line")
    nav_curve = res.get("nav_curve")

    F, PQ = vs.nav_line_to_feature(nav_line, nav_curve, crop_offset, (h, w))
    if F is not None:
        v, w_ang, info = vs.compute_control(F)
        P, Q = PQ
    else:
        v, w_ang, info = 0.0, 0.0, {"err_x": 0, "err_theta_deg": 0}
        P = Q = None

    dt = (time.perf_counter() - t0) * 1000.0

    overlay = None
    if draw:
        overlay = vs.draw_overlay(bgr, P, Q, v, w_ang, info)
        # Also draw the MultiROI overlay for reference (faint)
        # The vs overlay already has the red line, but we can blend
        pass

    return {
        "res": res,
        "F": F,
        "P": P,
        "Q": Q,
        "v": v,
        "w": w_ang,
        "info": info,
        "overlay": overlay,
        "time_ms": dt,
    }


def make_side_by_side(nav_bgr, comp_bgr, border=8, max_width=1200):
    """Create a single window with nav overlay (top) and MultiROI composite (bottom) stacked vertically.

    Both images are scaled to the same width (the wider one, capped to max_width)
    while preserving aspect ratio, then concatenated vertically with a white
    border and labels. This keeps the combined image at a reasonable size and
    avoids having to memorize two separate windows per frame. For video, the
    size is fixed after the first frame.
    """
    # Ensure both are 3-channel BGR
    if len(nav_bgr.shape) == 2:
        nav_bgr = cv2.cvtColor(nav_bgr, cv2.COLOR_GRAY2BGR)
    if len(comp_bgr.shape) == 2:
        comp_bgr = cv2.cvtColor(comp_bgr, cv2.COLOR_GRAY2BGR)

    h1, w1 = nav_bgr.shape[:2]
    h2, w2 = comp_bgr.shape[:2]
    # Target width is max of the two, capped
    target_w = min(max(w1, w2), max_width)
    # Scale both to target_w
    def scale_to_w(img, tw):
        h, w = img.shape[:2]
        if w == tw:
            return img
        scale = tw / w
        new_h = int(round(h * scale))
        return cv2.resize(img, (tw, new_h), interpolation=cv2.INTER_AREA)
    nav_s = scale_to_w(nav_bgr, target_w)
    comp_s = scale_to_w(comp_bgr, target_w)

    # Add labels on top
    label_h = 28
    def add_label(img, text):
        w = img.shape[1]
        bar = np.full((label_h, w, 3), (32, 32, 32), dtype=np.uint8)
        cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    nav_l = add_label(nav_s, "Navigation (vs overlay) - red furrow line, star is rover nose")
    comp_l = add_label(comp_s, "MultiROI Composite - binary / mask+ROIs / overlay")

    # White horizontal border between
    border_img = np.full((border, target_w, 3), (255, 255, 255), dtype=np.uint8)
    combined = np.vstack([nav_l, border_img, comp_l])
    return combined


def run_folder(input_path, output_dir, detector, vs, show=False):
    """Process a folder of images."""
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
                         "err_x_px", "err_theta_deg",
                         "nav_w", "nav_b", "has_line", "time_ms"])

        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            bgr = cv2.imread(p)
            if bgr is None:
                print(f"[WARN] cannot read {p}")
                continue
            out = process_image(bgr, detector, vs, draw=True)
            # Create single-window combined view: nav + composite side-by-side
            from test_multi_roi import make_composite
            comp = make_composite(bgr, out["res"])
            if out["overlay"] is not None:
                combined = make_side_by_side(out["overlay"], comp)
                cv2.imwrite(os.path.join(output_dir, f"{name}_combined.png"), combined)
                # Also keep individual for debugging if needed
                cv2.imwrite(os.path.join(output_dir, f"{name}_nav.png"), out["overlay"])
                cv2.imwrite(os.path.join(output_dir, f"{name}_composite.png"), comp)
            else:
                cv2.imwrite(os.path.join(output_dir, f"{name}_composite.png"), comp)

            has_line = out["res"]["nav_line"] is not None
            nav_w, nav_b = out["res"]["nav_line"] if has_line else (float("nan"), float("nan"))
            writer.writerow([os.path.basename(p),
                             f"{out['v']:.4f}", f"{out['w']:.4f}", f"{math.degrees(out['w']):.2f}",
                             f"{out['info'].get('err_x',0):.1f}", f"{out['info'].get('err_theta_deg',0):.2f}",
                             f"{nav_w:.4f}", f"{nav_b:.2f}", int(has_line), f"{out['time_ms']:.1f}"])
            print(f"[{name}] v={out['v']:.2f} w={math.degrees(out['w']):.1f}deg/s err_x={out['info'].get('err_x',0):.0f}px err_theta={out['info'].get('err_theta_deg',0):.1f}deg has_line={has_line} {out['time_ms']:.1f}ms")
            if show and out["overlay"] is not None:
                cv2.imshow("MultiROI Navigation", out["overlay"])
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
    print(f"\nSaved to {os.path.abspath(output_dir)}")
    print(f"CSV: {csv_path}")


def run_video(input_path, output_dir, detector, vs, show=False):
    """Process a video file or camera index."""
    # Try to interpret as camera index
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
    # Video writer (if not camera) - will be initialized after first frame to know combined size
    writer = None
    first_frame = True

    with open(csv_path, "w", newline="") as f:
        csvw = csv.writer(f)
        csvw.writerow(["frame", "timestamp", "v_mps", "w_radps", "err_x_px", "err_theta_deg", "has_line"])

        frame_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            out = process_image(bgr, detector, vs, draw=True)
            csvw.writerow([frame_idx, time.time(),
                           f"{out['v']:.4f}", f"{out['w']:.4f}",
                           f"{out['info'].get('err_x',0):.1f}", f"{out['info'].get('err_theta_deg',0):.1f}",
                           int(out["res"]["nav_line"] is not None)])
            # Create combined view for video/show
            from test_multi_roi import make_composite
            comp = make_composite(bgr, out["res"])
            combined = make_side_by_side(out["overlay"] if out["overlay"] is not None else bgr, comp) if out["overlay"] is not None else comp

            if writer is None and not is_camera and out["overlay"] is not None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                h_c, w_c = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join(output_dir, f"{name}_nav.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w_c, h_c))

            if writer is not None:
                # Ensure writer size matches combined
                if writer.isOpened():
                    # If size mismatch, re-create writer
                    w_wr = int(writer.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(writer, 'get') else combined.shape[1]
                    h_wr = int(writer.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(writer, 'get') else combined.shape[0]
                    if w_wr != combined.shape[1] or h_wr != combined.shape[0]:
                        writer.release()
                        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = os.path.join(output_dir, f"{name}_nav.mp4")
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (combined.shape[1], combined.shape[0]))
                writer.write(combined)
            if show:
                cv2.imshow("MultiROI Navigation - Combined (nav | composite)", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: v={out['v']:.2f} w={math.degrees(out['w']):.1f} deg/s")

    cap.release()
    if writer:
        writer.release()
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MultiROI furrow following navigation")
    parser.add_argument("--input", default="../../Photos", help="Image folder, glob, video file or camera index")
    parser.add_argument("--output", default="./nav_output", help="Output directory")
    parser.add_argument("--video", action="store_true", help="Treat input as video/camera")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--width", type=int, default=640, help="Expected image width for normalization")
    parser.add_argument("--height", type=int, default=480, help="Expected height")
    parser.add_argument("--vf", type=float, default=0.20, help="Forward speed m/s")
    parser.add_argument("--wmax", type=float, default=0.50, help="Max angular rad/s")
    parser.add_argument("--lambdax", type=float, default=10.0, help="Lateral gain")
    parser.add_argument("--lambdatheta", type=float, default=1.0, help="Heading gain")
    parser.add_argument("--n_strips", type=int, default=10)
    parser.add_argument("--l_frac", type=float, default=0.05)
    parser.add_argument("--index", choices=("raw","normalized"), default="raw")
    args = parser.parse_args()

    # Detector with default robust params (from test_multi_roi.py)
    detector = MultiROIDetector(n_strips=args.n_strips, l_frac=args.l_frac, index=args.index)

    # Visual servoing params - adapted from agribot_vs_run.yaml
    vs_params = MRVSParams(
        width=args.width, height=args.height,
        vf_des=args.vf, w_max=args.wmax,
        lambda_x=args.lambdax, lambda_theta=args.lambdatheta,
    )
    vs = MultiROIVS(vs_params)

    os.makedirs(args.output, exist_ok=True)

    if args.video:
        run_video(args.input, args.output, detector, vs, show=args.show)
    else:
        run_folder(args.input, args.output, detector, vs, show=args.show)


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
        self.sub = self.create_subscription(Image, '/front/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        out = process_image(bgr, self.detector, self.vs, draw=False)
        twist = Twist()
        twist.linear.x = float(out['v'])
        twist.angular.z = float(out['w'])
        self.pub.publish(twist)

# To run: colcon build, source install/setup.bash, ros2 run <pkg> multoroi_nav
"""
