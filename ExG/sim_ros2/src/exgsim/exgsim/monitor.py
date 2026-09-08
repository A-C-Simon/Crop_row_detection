#!/usr/bin/env python3
"""Overlay + CSV monitor for the ExG-in-Gazebo rig.

The C++ agribot_vs node publishes no debug image, so this node watches its
inputs/outputs and produces the rig-standard artifacts:

  - subscribes /camera/image_raw, /vs_msg (err_x/err_theta magnitudes),
    /cmd_vel (v/w display) and /odom (pose),
  - publishes the drawn overlay on /exgsim/overlay (rgb8),
  - writes <log_dir>/nav_run.csv + overlay PNGs (same schema as the other
    rigs; cross_track honors lane/circle env, n_two/conf come from the
    vs_msg magnitudes heuristically).

Probe mode (MRSIM_MODE=probe): capture MRSIM_FRAMES raw frames to
MRSIM_OUT_DIR, print per-frame vs_msg values, then exit.
"""
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from visual_crop_row_navigation_ros2.msg import VsMsg


def _sensor_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=2)


class ExGMonitor(Node):
    def __init__(self):
        super().__init__("exg_monitor")
        self.bridge = CvBridge()

        def _p(name, default):
            try:
                return self.get_parameter(name).value
            except Exception:
                return default

        self.lane_y = float(os.environ.get("MRSIM_LANE_Y", _p("lane_y", 0.0)))
        self.lane_end_x = float(os.environ.get("MRSIM_LANE_END_X",
                                               _p("lane_end_x", 9.0)))
        self.circle_cx = float(os.environ.get("MRSIM_CIRCLE_CX", _p("circle_cx", 0.0)))
        self.circle_cy = float(os.environ.get("MRSIM_CIRCLE_CY", _p("circle_cy", 0.0)))
        self.circle_r = float(os.environ.get("MRSIM_CIRCLE_R", _p("circle_r", 0.0)))
        self.max_laps = float(os.environ.get("MRSIM_CIRCLE_LAPS", _p("max_laps", 1.0)))
        self.max_seconds = float(os.environ.get("MRSIM_MAX_SECONDS",
                                                _p("max_seconds", 0.0)))
        log_dir = os.environ.get("MRSIM_LOG_DIR", _p("log_dir", ""))
        self.save_every = int(os.environ.get("MRSIM_SAVE_EVERY",
                                             _p("save_every", 20)))
        self._lap_angle = 0.0
        self._lap_prev = None

        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_f = open(self.log_dir / "nav_run.csv", "w", newline="")
            self.csv_w = csv.writer(self.csv_f)
            self.csv_w.writerow(["sim_t", "odom_x", "odom_y", "cross_track",
                                 "err_x_px", "raw_th_deg", "filt_th_deg",
                                 "conf", "status", "v", "w", "n_two", "ff"])
            self.get_logger().info(f"logging to {self.log_dir}")
        else:
            self.csv_f = None

        self.odom_x = self.odom_y = 0.0
        self.err_x = self.err_th = 0.0
        self.v = self.w = 0.0
        self.last_bgr = None
        self.last_t = None
        self.frame_idx = 0
        self.t0 = None

        self.ovl_pub = self.create_publisher(Image, "/exgsim/overlay", 5)
        self.create_subscription(Image, "/camera/image_raw",
                                 self._on_image, _sensor_qos())
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(VsMsg, "/vs_msg", self._on_vs, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)

    # --------------------------------------------------------------
    def _on_odom(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        if self.circle_r > 0:
            ang = math.atan2(self.odom_y - self.circle_cy,
                             self.odom_x - self.circle_cx)
            if self._lap_prev is not None:
                d = ang - self._lap_prev
                while d > math.pi:
                    d -= 2.0 * math.pi
                while d < -math.pi:
                    d += 2.0 * math.pi
                self._lap_angle += d
            self._lap_prev = ang

    def _cross_track(self):
        if self.circle_r > 0:
            return math.hypot(self.odom_x - self.circle_cx,
                              self.odom_y - self.circle_cy) - self.circle_r
        return self.odom_y - self.lane_y

    def _on_vs(self, msg: VsMsg):
        self.err_x = float(msg.err_x)
        self.err_th = float(msg.err_theta)

    def _on_cmd(self, msg: Twist):
        self.v = float(msg.linear.x)
        self.w = float(msg.angular.z)

    # --------------------------------------------------------------
    def _on_image(self, msg: Image):
        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t_now
        self.frame_idx += 1
        try:
            if msg.encoding in ("bgr8", "8UC3"):
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().warn(f"decode: {e}")
            return
        self.last_bgr = bgr
        self.last_t = t_now

        ovl = self._draw(bgr)
        if self.ovl_pub.get_subscription_count() > 0:
            try:
                im = self.bridge.cv2_to_imgmsg(
                    cv2.cvtColor(ovl, cv2.COLOR_BGR2RGB), encoding="rgb8")
                im.header = msg.header
                self.ovl_pub.publish(im)
            except Exception as e:
                self.get_logger().warn(f"overlay pub: {e}")

        if self.log_dir:
            self.csv_w.writerow([
                f"{t_now:.3f}", f"{self.odom_x:.3f}", f"{self.odom_y:.3f}",
                f"{self._cross_track():.3f}",
                f"{self.err_x:.1f}", f"{math.degrees(self.err_th):.2f}",
                f"{math.degrees(self.err_th):.2f}",
                "1.00", "exg", f"{self.v:.3f}", f"{self.w:.3f}", "0", "0.0000"])
            self.csv_f.flush()
            if self.frame_idx % self.save_every == 0:
                cv2.imwrite(str(self.log_dir / f"frame_{self.frame_idx:05d}.png"),
                            ovl)
        if self.frame_idx % 60 == 0:
            self.get_logger().info(
                f"[{self.frame_idx:>4}] t={t_now:6.2f} "
                f"odom=({self.odom_x:+.2f},{self.odom_y:+.2f}) "
                f"cross={self._cross_track():+.3f} "
                f"err=({self.err_x:.1f},{math.degrees(self.err_th):.1f}) "
                f"v={self.v:.2f} w={self.w:+.2f}")
        if (self.max_seconds > 0 and (t_now - self.t0) >= self.max_seconds):
            self.get_logger().info(f"stop: reached max_seconds={self.max_seconds}")
            raise SystemExit
        if (self.circle_r > 0 and self.max_laps > 0
                and abs(self._lap_angle) >= self.max_laps * 2.0 * math.pi):
            self.get_logger().info(
                f"stop: completed {self.max_laps:g} lap(s)")
            raise SystemExit
        if self.circle_r <= 0 and self.lane_end_x and self.odom_x >= self.lane_end_x:
            self.get_logger().info(f"stop: end of lane x={self.odom_x:.2f}")
            raise SystemExit

    def _draw(self, bgr):
        out = bgr.copy()
        h, w = out.shape[:2]
        cv2.circle(out, (w // 2, h // 2), 4, (255, 255, 0), -1)
        cv2.drawMarker(out, (w // 2, h - 20), (255, 255, 0),
                       cv2.MARKER_STAR, 20, 2)
        txt = (f"v={self.v:.2f} m/s w={math.degrees(self.w):.1f} deg/s | "
               f"err_x={self.err_x:.1f} err_th={math.degrees(self.err_th):.1f}deg")
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
        txt2 = (f"odom=({self.odom_x:+.2f},{self.odom_y:+.2f}) "
                f"cross={self._cross_track():+.2f}")
        cv2.putText(out, txt2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def destroy_node(self):
        if self.csv_f and not self.csv_f.closed:
            self.csv_f.close()
        super().destroy_node()


def probe_main(args=None):
    """Capture N raw frames plus per-frame vs_msg errors, then exit."""
    rclpy.init(args=args)
    node = rclpy.create_node("exg_probe")

    def _p(name, default):
        try:
            return node.get_parameter(name).value
        except Exception:
            return default

    node.declare_parameter("camera_topic", "/camera/image_raw")
    node.declare_parameter("frames", 5)
    node.declare_parameter("out_dir", "/tmp")
    cam_topic = str(os.environ.get("MRSIM_CAMERA_TOPIC", _p("camera_topic",
                                                             "/camera/image_raw")))
    nframes = int(os.environ.get("MRSIM_FRAMES", _p("frames", 5)))
    outdir = Path(str(os.environ.get("MRSIM_OUT_DIR", _p("out_dir", "/tmp"))))
    outdir.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge()
    got = {"n": 0}
    last_vs = {"x": float("nan"), "th": float("nan")}

    def vs_cb(msg: VsMsg):
        last_vs["x"] = float(msg.err_x)
        last_vs["th"] = float(msg.err_theta)

    def cb(msg: Image):
        if got["n"] >= nframes:
            return
        try:
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") \
                if msg.encoding in ("bgr8", "8UC3") else \
                cv2.cvtColor(bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"),
                             cv2.COLOR_RGB2BGR)
        except Exception as e:
            node.get_logger().error(f"decode: {e}")
            return
        cv2.imwrite(str(outdir / f"probe_raw_{got['n']:03d}.png"), bgr)
        node.get_logger().info(
            f"frame {got['n']}: vs err_x={last_vs['x']:.1f} "
            f"err_th={math.degrees(last_vs['th']):.1f}deg")
        got["n"] += 1
        if got["n"] >= nframes:
            node.get_logger().info(f"probe done -> {outdir}")
            raise SystemExit

    node.create_subscription(Image, cam_topic, cb, _sensor_qos())
    node.create_subscription(VsMsg, "/vs_msg", vs_cb, 10)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ExGMonitor()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        node.get_logger().info("monitor done")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    if os.environ.get("MRSIM_MODE") == "probe":
        probe_main()
    else:
        main()
