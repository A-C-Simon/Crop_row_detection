#!/usr/bin/env python3
"""MultiROI crop-row navigation node for the Gazebo test rig.

Subscribes to the simulated rover's front camera (/camera/image_raw), runs
the MultiROI detection + temporal filter + visual servoing pipeline
(process_image from LinReg/MultiROI/run_mr_navigation.py), and publishes
/cmd_vel to the gazebo diff-drive plugin - closed loop.

Also:
  - subscribes /odom to log world pose / cross-track error (world lane center
    is `lane_y`; cross-track = odom.y - lane_y),
  - publishes the drawn overlay on /multiroi/overlay (rgb8),
  - writes a CSV + overlay PNGs into log_dir ("" disables).

Usage (after colcon build + source):
  ros2 run mrsim multiroi_nav --ros-args -p log_dir:=/tmp/mrsim_log
  or see launch/farm.launch.py.
Probe mode (camera/calibration only, no control):
  ros2 run mrsim multiroi_probe --ros-args -p out_dir:=/tmp -p frames:=5
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


def _find_multiroi_dir():
    env = os.environ.get("MULTIROI_DIR")
    if env and (Path(env) / "run_mr_navigation.py").exists():
        return str(Path(env).resolve())
    # walk up from this file looking for the marker
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "run_mr_navigation.py").exists():
            return str(p)
        p = p.parent
    return None


_MULTIROI = _find_multiroi_dir()
if _MULTIROI is None:
    raise RuntimeError("cannot locate MultiROI source dir; set MULTIROI_DIR env")
sys.path.insert(0, _MULTIROI)

# nav-mode pipeline comes from the algorithm-selection seam (default: multiroi)
from pipeline import build_pipeline, current_algorithm  # noqa: E402


def _sensor_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST,
                      depth=2,
                      durability=DurabilityPolicy.VOLATILE)


class MultiROINavNode(Node):
    def __init__(self):
        super().__init__("multiroi_nav")
        self.bridge = CvBridge()

        # configuration: env vars when launched from farm.launch.py, with
        # ros parameters as fallback (for `ros2 run mrsim multiroi_nav -p ...`)
        def _p(name, default):
            try:
                return self.get_parameter(name).value
            except Exception:
                return default

        cam_topic = str(os.environ.get("MRSIM_CAMERA_TOPIC",
                                        _p("camera_topic", "/camera/image_raw")))
        cmd_topic = str(os.environ.get("MRSIM_CMD_TOPIC",
                                       _p("cmd_vel_topic", "/cmd_vel")))
        odom_topic = str(os.environ.get("MRSIM_ODOM_TOPIC",
                                        _p("odom_topic", "/odom")))
        self.lane_y = float(os.environ.get("MRSIM_LANE_Y", _p("lane_y", 0.0)))
        self.lane_end_x = float(os.environ.get("MRSIM_LANE_END_X",
                                               _p("lane_end_x", 9.0)))
        self.max_seconds = float(os.environ.get("MRSIM_MAX_SECONDS",
                                                _p("max_seconds", 0.0)))
        # circle-lane mode (concentric ring field): the diff-drive plugin
        # initializes odometry at the spawn (world) pose, so odom doubles as
        # world xy; cross-track becomes radial error and the run ends after
        # max_laps angle swept around (cx, cy). R<=0 keeps straight behavior.
        self.circle_cx = float(os.environ.get("MRSIM_CIRCLE_CX", _p("circle_cx", 0.0)))
        self.circle_cy = float(os.environ.get("MRSIM_CIRCLE_CY", _p("circle_cy", 0.0)))
        self.circle_r = float(os.environ.get("MRSIM_CIRCLE_R", _p("circle_r", 0.0)))
        self.max_laps = float(os.environ.get("MRSIM_CIRCLE_LAPS", _p("max_laps", 1.0)))
        self._lap_angle = 0.0
        self._lap_prev = None
        if self.circle_r > 0:
            self.get_logger().info(
                f"circle lane: center=({self.circle_cx:.2f},{self.circle_cy:.2f}) "
                f"R={self.circle_r:.2f} max_laps={self.max_laps:g} "
                f"(cross_track=radial error)")
        log_dir = os.environ.get("MRSIM_LOG_DIR", _p("log_dir", ""))
        self.save_every = int(os.environ.get("MRSIM_SAVE_EVERY",
                                             _p("save_every", 20)))
        self.v_max = float(os.environ.get("MRSIM_V_MAX", _p("v_max", 0.22)))
        # teleop mode: keep detection/logging alive but never publish /cmd_vel
        # (the teleop node owns the topic). Env MRSIM_NAV_IDLE=1 from launch.
        self.idle = str(os.environ.get("MRSIM_NAV_IDLE", _p("nav_idle", ""))) == "1"
        if self.idle:
            self.get_logger().info("nav node IDLE (teleop mode): no /cmd_vel published")

        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_f = open(self.log_dir / "nav_run.csv", "w", newline="")
            self.csv_w = csv.writer(self.csv_f)
            self.csv_w.writerow(["sim_t", "odom_x", "odom_y", "cross_track",
                                 "err_x_px", "raw_th_deg", "filt_th_deg",
                                 "conf", "status", "v", "w", "n_two", "ff"])
            self.get_logger().info(f"logging to {self.log_dir}")

        # --- algorithm pipeline (selection seam; default: multiroi) ---
        self.algorithm = str(os.environ.get("MRSIM_ALGORITHM",
                                            _p("algorithm", "multiroi")))
        self.pipeline = build_pipeline(self.algorithm)
        self.get_logger().info(f"algorithm: {self.pipeline.name} "
                               f"(vf_des={self.v_max})")
        # mr_vs vf_des == v_max: rebuild vs with the rover's speed cap
        if self.pipeline.name == "multiroi" and hasattr(self.pipeline, "vs"):
            self.pipeline.vs.params.vf_des = self.v_max
        self.last_w = 0.0

        # --- ROS plumbing ---
        self.cam_sub = self.create_subscription(
            Image, cam_topic, self._on_image, _sensor_qos())
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.ovl_pub = self.create_publisher(Image, "/multiroi/overlay", 5)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 10)

        self.odom_x = self.odom_y = 0.0
        self.last_img_t = None
        self.frame_idx = 0
        self.pub_count = 0
        self._last_cmd = Twist()
        self._last_cmd.linear.x = 0.0
        self._last_cmd.angular.z = 0.0
        self._cmd_seq = 0

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

    # --------------------------------------------------------------
    def _stop_robot(self, reason: str):
        self.get_logger().info(f"stop: {reason}")
        zero = Twist()
        self.cmd_pub.publish(zero)
        self._last_cmd = zero
        if self.log_dir:
            self.csv_f.close()
            self.get_logger().info(f"CSV closed -> {self.log_dir / 'nav_run.csv'}")
        self.destroy_node()

    # --------------------------------------------------------------
    def _on_image(self, msg: Image):
        # sim-time delta between frames
        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_img_t is not None:
            dt = t_now - self.last_img_t
            if dt <= 0.0 or dt > 0.5:
                dt = 0.05
        else:
            dt = 0.05
        self.last_img_t = t_now

        if self.frame_idx == 0:
            self.t0 = t_now
        self.frame_idx += 1

        # decode to BGR
        enc = msg.encoding
        if enc in ("bgr8", "8UC3"):
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        elif enc in ("rgb8",):
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            self.get_logger().warn(f"unsupported encoding {enc}")
            return

        try:
            out = self.pipeline.process(bgr)
        except Exception as e:
            self.get_logger().error(f"pipeline {self.pipeline.name} failed: {e}")
            return
        self.last_w = float(out["w"])
        info = out["info"]

        # --- command (skipped in teleop idle mode) ---
        twist = Twist()
        twist.linear.x = float(np.clip(out["v"], 0.0, self.v_max))
        twist.angular.z = float(out["w"])
        if not self.idle:
            self.cmd_pub.publish(twist)
            self._last_cmd = twist
        self._cmd_seq += 1

        # --- overlay ---
        ovl = out.get("overlay")
        if ovl is not None and self.ovl_pub.get_subscription_count() > 0:
            try:
                rgb = cv2.cvtColor(ovl, cv2.COLOR_BGR2RGB)
                im = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
                im.header = msg.header
                self.ovl_pub.publish(im)
            except Exception as e:  # pragma: no cover
                self.get_logger().warn(f"overlay pub: {e}")

        # --- logging ---
        cross = self._cross_track()
        if self.log_dir:
            self.csv_w.writerow([
                f"{t_now:.3f}", f"{self.odom_x:.3f}", f"{self.odom_y:.3f}",
                f"{cross:.3f}",
                f"{info.get('err_x', 0):.1f}",
                f"{info.get('raw_err_theta_deg', info.get('err_theta_deg', 0)):.2f}",
                f"{info.get('filt_err_theta_deg', 0):.2f}",
                f"{info.get('confidence', 0):.2f}", str(info.get("status", "")),
                f"{out['v']:.3f}", f"{out['w']:.3f}",
                int(info.get("n_two_sided", 0)), f"{info.get('ff', 0):.4f}"])
            self.csv_f.flush()
            if ovl is not None and self.frame_idx % self.save_every == 0:
                cv2.imwrite(str(self.log_dir / f"frame_{self.frame_idx:05d}.png"), ovl)

        if self.frame_idx % 60 == 0:
            lap_txt = (f" lap={abs(self._lap_angle) / (2.0 * math.pi):.2f}"
                       if self.circle_r > 0 else "")
            self.get_logger().info(
                f"[{self.frame_idx:>4}] t={t_now:6.2f} odom=({self.odom_x:+.2f},"
                f"{self.odom_y:+.2f}) cross={cross:+.3f}{lap_txt} "
                f"w={twist.angular.z:+.2f} conf={info.get('confidence', 0):.2f} "
                f"status={info.get('status', '')}")

        # auto-termination is disabled while idling (teleop decides when done)
        if self.idle:
            return

        # termination conditions
        if self.max_seconds > 0 and (t_now - self.t0) >= self.max_seconds:
            self._stop_robot(f"reached max_seconds={self.max_seconds}")
        elif self.circle_r > 0:
            if self.max_laps > 0 and abs(self._lap_angle) >= self.max_laps * 2.0 * math.pi:
                self._stop_robot(
                    f"completed {self.max_laps:g} lap(s) "
                    f"(radial err {cross:+.2f} m)")
            elif self.frame_idx > 60000:
                self._stop_robot("frame cap")
        elif self.odom_x >= self.lane_end_x:
            self._stop_robot(f"reached end of lane (x={self.odom_x:.2f})")
        elif self.frame_idx > 20000:
            self._stop_robot("frame cap")

    def destroy_node(self):
        if self.log_dir and not self.csv_f.closed:
            self.csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiROINavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    finally:
        node._stop_robot("shutdown") if hasattr(node, "_stop_robot") else None
        if rclpy.ok():
            rclpy.shutdown()


def probe_main(args=None):
    """Camera calibration probe: capture N frames, run detector, save PNGs."""
    rclpy.init(args=args)
    node = rclpy.create_node("multiroi_probe")
    # env vars when launched from farm_probe.launch.py, params as fallback
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

    # probe uses the multiroi pipeline (detector-level calibration view)
    from pipeline import build_pipeline
    pl = build_pipeline("multiroi")
    detector, vs = pl.detector, pl.vs
    got = {"n": 0}

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
        res = detector.detect(bgr)
        F, PQ = vs.nav_line_to_feature(res.get("nav_line"), res.get("nav_curve"),
                                       res.get("crop_offset", (0, 0)),
                                       bgr.shape[:2])
        th = math.degrees(F[2]) if F is not None else float("nan")
        node.get_logger().info(
            f"frame {got['n']}: n_two={res.get('n_two_sided')} "
            f"median_w={float(res.get('median_width', 0)):.0f}px "
            f"theta={th:+.1f}deg nav_line={'yes' if F is not None else 'NO'}")
        got["n"] += 1
        if got["n"] >= nframes:
            node.get_logger().info(f"probe done -> {outdir}")
            raise SystemExit

    node.create_subscription(Image, cam_topic, cb, _sensor_qos())
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    # farm.launch.py / farm_probe.launch.py launch this file directly with
    # MRSIM_MODE set; plain `python3 nav_node.py` runs the nav loop.
    if os.environ.get("MRSIM_MODE") == "probe":
        probe_main()
    else:
        main()
