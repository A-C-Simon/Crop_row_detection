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
from typing import Optional, Tuple, List

import cv2
import numpy as np

from test_multi_roi import MultiROIDetector
from mr_vs import MultiROIVS, MRVSParams, wrapToPi


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
    """

    def __init__(self, detector: Optional[MultiROIDetector] = None,
                 vs: Optional[MultiROIVS] = None,
                 simulator: Optional[RoverSimulator] = None):
        self.detector = detector or MultiROIDetector()
        self.vs = vs or MultiROIVS()
        self.simulator = simulator or RoverSimulator()
        self.step_count = 0
        self.last_cmd = RoverCommand(0, 0, 0, 0, False)

    def step(self, bgr, draw=True):
        """
        One navigation step: detect + control.

        Returns (cmd, overlay, res) where
          cmd is RoverCommand
          overlay is BGR image with debug overlay
          res is the raw detector result dict
        """
        h, w = bgr.shape[:2]
        # Detect
        res = self.detector.detect(bgr)
        crop_offset = res.get("crop_offset", (0, 0))
        nav_line = res.get("nav_line")
        nav_curve = res.get("nav_curve")

        # Visual features
        F, PQ = self.vs.nav_line_to_feature(nav_line, nav_curve, crop_offset, (h, w))
        if F is not None:
            v, w_ang, info = self.vs.compute_control(F)
            P, Q = PQ
            has_line = True
        else:
            # No line: stop or search
            v, w_ang, info = 0.0, 0.0, {"err_x": 0, "err_theta_deg": 0}
            P = Q = None
            has_line = False
            # Could implement search behavior: rotate slowly
            # w_ang = 0.2  # search

        cmd = RoverCommand(v=v, w=w_ang,
                           err_x=info.get("err_x", 0),
                           err_theta=info.get("err_theta_deg", 0),
                           has_line=has_line)

        # Update simulator
        self.simulator.update(cmd.v, cmd.w)
        self.last_cmd = cmd
        self.step_count += 1

        # Draw overlay
        overlay = None
        if draw:
            overlay = self.vs.draw_overlay(bgr, P, Q, v, w_ang, info)
            # Add detector overlay as well (faint)
            # We already have the vs line in red, but also show the MultiROI composite
            # For now just the vs overlay is enough
            # Add text for step count and pose
            pose = self.simulator.pose
            txt2 = f"Pose x={pose.x:.2f} y={pose.y:.2f} th={math.degrees(pose.theta):.1f}deg | Step {self.step_count}"
            cv2.putText(overlay, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            # Draw path history as small dots on top-down view (optional)
            # For now just the overlay

        return cmd, overlay, res

    def reset(self, pose: Optional[RoverPose] = None):
        """Reset the stack and simulator."""
        self.simulator = RoverSimulator(initial_pose=pose)
        self.step_count = 0
        self.last_cmd = RoverCommand(0, 0, 0, 0, False)


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
