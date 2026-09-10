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
        # row changing (ExG-style headland turns, straight fields only):
        # at each lane end, bulb-turn into the adjacent furrow and drive it
        # back the other way. lane_index tracks the current furrow in the
        # sidecar furrow list; turn_dir walks it, flipping at the edges.
        self.row_change = str(os.environ.get("MRSIM_ROW_CHANGE",
                                             _p("row_change", ""))).lower() \
            in ("1", "true", "yes")
        self.max_lanes = int(float(os.environ.get("MRSIM_MAX_LANES",
                                                  _p("max_lanes", 2))))
        self.lane_start_x = float(os.environ.get("MRSIM_LANE_START_X",
                                                 _p("lane_start_x", -8.0)))
        self.lane_index = int(float(os.environ.get("MRSIM_LANE_INDEX",
                                                   _p("lane_index", 0))))
        try:
            self.furrows = [float(c) for c in
                            str(os.environ.get("MRSIM_FURROWS", "0.0")).split(",")]
        except Exception:
            self.furrows = [0.0]
        self.turn_dir = 1
        self.lanes_done = 0
        self.drive_dir = 1
        # turn_mode: "bulb" (odometry push/spin/slide/spin, validated),
        # "fishtail" (rear-guided reverse-in: push, arc, reverse), or
        # "shuttle" (no turns at all: vision row-end, lateral jog one
        # spacing, lanes alternate forward/front-camera and
        # backward/rear-camera with a primary/secondary camera swap).
        self.turn_mode = str(os.environ.get("MRSIM_TURN_MODE",
                                            _p("turn_mode", "bulb"))).lower()
        if self.turn_mode not in ("bulb", "fishtail", "shuttle"):
            self.get_logger().warn(
                f"unknown turn_mode '{self.turn_mode}'; using bulb")
            self.turn_mode = "bulb"
        self.phase = "follow"
        self.phase_t0 = 0.0
        self.phase_x0 = self.phase_y0 = self.phase_yaw0 = 0.0
        self.phase_target_y = self.lane_y
        self.phase_target_yaw = 0.0
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
        if self.row_change:
            if self.circle_r > 0:
                self.get_logger().warn(
                    "row_change is for straight fields; ignoring on the ring")
                self.row_change = False
            elif len(self.furrows) < 2:
                self.get_logger().warn(
                    "row_change needs 2+ furrows; single lane, driving through")
                self.row_change = False
            else:
                self.lane_index = max(0, min(len(self.furrows) - 1,
                                             self.lane_index))
                self.lane_y = float(self.furrows[self.lane_index])
                self.phase_target_y = self.lane_y
                self.get_logger().info(
                    f"row change on: {len(self.furrows)} furrows, start lane "
                    f"{self.lane_index} (y={self.lane_y:+.2f}), "
                    f"max_lanes={self.max_lanes:g} turn={self.turn_mode}")
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
                                 "conf", "status", "v", "w", "n_two", "ff",
                                 "rear_err_x_px", "rear_th_deg", "rear_conf"])
            self.get_logger().info(f"logging to {self.log_dir}")

        # --- algorithm pipeline (selection seam; default: multiroi) ---
        self.algorithm = str(os.environ.get("MRSIM_ALGORITHM",
                                            _p("algorithm", "multiroi")))
        self.pipeline = build_pipeline(self.algorithm)
        self.get_logger().info(f"algorithm: {self.pipeline.name} "
                               f"(vf_des={self.v_max})")
        if getattr(self.pipeline, "line_fit", False):
            self.get_logger().info("nav fit: straight line (no spline)")
        # --- rear algorithm pipeline (fishtail + shuttle): a second,
        # independent MultiROI instance on the rear camera. While reversing,
        # the rover moves the way the rear camera faces: the image mirror
        # and the reversed motion cancel, so its (v, w) applies with +sign
        # (v negated for travel, w kept). Verified by derivation; the
        # shuttle backward legs prove it in sim (rear_w_sign=+1).
        self.rear_pipeline = None
        self.rear_out = None
        self.rear_t = None
        self.rear_frame_idx = 0
        self.rear_ms = 0.0
        if self.turn_mode in ("fishtail", "shuttle"):
            try:
                self.rear_pipeline = build_pipeline(self.algorithm)
                if (self.pipeline.name == "multiroi"
                        and hasattr(self.rear_pipeline, "vs")):
                    self.rear_pipeline.vs.params.vf_des = self.v_max
                self.get_logger().info("rear pipeline: on "
                                       "(/camera_back/image_raw)")
            except Exception as e:
                self.get_logger().warn(f"rear pipeline failed: {e}")
        try:
            self.rear_w_sign = float(os.environ.get("MRSIM_REAR_W_SIGN",
                                                    _p("rear_w_sign", 1.0)))
        except Exception:
            self.rear_w_sign = 1.0
        self.last_w = 0.0
        # front freshness (the secondary-camera check needs it on backward
        # shuttle legs) + leg odometry for the shuttle vision trigger guard
        self.front_info = None
        self.front_t = None
        self.leg_x0 = self.lane_start_x
        self.leg_yaw = 0.0
        self.end_empty_n = 0

        # --- ROS plumbing ---
        self.cam_sub = self.create_subscription(
            Image, cam_topic, self._on_image, _sensor_qos())
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.ovl_pub = self.create_publisher(Image, "/multiroi/overlay", 5)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 10)
        self.rear_sub = None
        self.rear_ovl_pub = None
        if self.rear_pipeline is not None:
            rear_topic = str(os.environ.get("MRSIM_REAR_CAMERA_TOPIC",
                                            _p("rear_camera_topic",
                                               "/camera_back/image_raw")))
            self.rear_sub = self.create_subscription(
                Image, rear_topic, self._on_rear_image, _sensor_qos())
            self.rear_ovl_pub = self.create_publisher(
                Image, "/multiroi/overlay_rear", 5)

        self.odom_x = self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_img_t = None
        self.frame_idx = 0
        self.pub_count = 0
        self._last_cmd = Twist()
        self._last_cmd.linear.x = 0.0
        self._last_cmd.angular.z = 0.0
        self._cmd_seq = 0

    # --------------------------------------------------------------
    @staticmethod
    def _quat_to_yaw(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _ang_diff(a, b):
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def _on_odom(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        try:
            self.odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)
        except Exception:
            pass
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
    # Row changing: headland turns between adjacent furrows. At each lane
    # end the rover pushes past the rows, turns into the next furrow and
    # drives it back the other way. Detection keeps running for the overlay
    # the whole time. Two styles (turn_mode): "bulb" scripts
    # push/spin1/slide/spin2 on odometry; "fishtail" arcs forward away from
    # the target furrow then reverses into it, steered by the rear camera
    # when it locks (negated servo output) with an odometry crab fallback.
    # Straight fields only.
    TURN_PUSH_M = 1.3
    TURN_RATE = 0.5
    TURN_DRIVE_V = 0.18
    TURN_TOL_YAW = 0.12
    TURN_TOL_Y = 0.06
    TURN_TIMEOUT = 15.0
    # fishtail (rear-guided reverse-in) tuning: forward arc swings the nose
    # toward the next furrow, then the rover backs into it steered by the
    # rear camera. No in-place spinning.
    FISHTAIL_ARC_YAW = 1.5
    FISHTAIL_PUSH_M = 0.7
    FISHTAIL_REVERSE_V = 0.15
    FISHTAIL_CRAB_GAIN = 1.2
    FISHTAIL_TOL_YAW = 0.20
    FISHTAIL_TOL_Y = 0.10
    FISHTAIL_REAR_TOL_PX = 25.0
    FISHTAIL_MIN_REVERSE_M = 0.4
    FISHTAIL_TIMEOUT = 25.0

    def _reset_perception(self):
        try:
            self.pipeline.reset()
        except Exception:
            pass
        try:
            if hasattr(self.pipeline, "t_filter"):
                self.pipeline.t_filter.reset()
        except Exception:
            pass
        try:
            if self.rear_pipeline is not None:
                self.rear_pipeline.reset()
                if hasattr(self.rear_pipeline, "t_filter"):
                    self.rear_pipeline.t_filter.reset()
        except Exception:
            pass
        self.last_w = 0.0
        self.rear_out = None
        self.rear_t = None

    def _lane_end_reached(self):
        if self.drive_dir > 0:
            return self.odom_x >= self.lane_end_x
        return self.odom_x <= self.lane_start_x

    def _enter_turn(self, t_now):
        nxt = self.lane_index + self.turn_dir
        if not (0 <= nxt < len(self.furrows)):
            self.turn_dir *= -1
            nxt = self.lane_index + self.turn_dir
        self.phase_target_y = float(self.furrows[nxt])
        self.phase = "push"
        self.phase_t0 = t_now
        self.phase_x0, self.phase_y0 = self.odom_x, self.odom_y
        self.phase_yaw0 = self.odom_yaw
        self.phase_slide_yaw = 0.0
        # fishtail bookkeeping: side (+1 target above, -1 below), leg yaw,
        # arc-end yaw and final (next-leg) yaw. Arc carries the rover toward
        # the target furrow while moving along the current leg direction.
        self.phase_side = 1.0 if self.phase_target_y >= self.odom_y else -1.0
        self.phase_yaw_leg = self.odom_yaw
        # Fishtail geometry (verified sign analysis, straight fields): the
        # forward arc swings the nose AWAY from the target furrow, then the
        # reversing rover backs TOWARD it while the nose comes around to the
        # next-leg heading. (Arcing toward the target first backs away from
        # it and strands the rover - seen in sim.)
        self.phase_arc_yaw = (self.phase_yaw_leg - self.phase_side
                              * self.drive_dir * self.FISHTAIL_ARC_YAW)
        self.phase_final_yaw = (self.phase_yaw_leg - self.phase_side
                                * self.drive_dir * math.pi)
        self.phase_no_rear_t = None
        self._no_rear_warned = False
        self.get_logger().info(
            f"row change: lane {self.lane_index} -> {nxt} "
            f"(y {self.lane_y:+.2f} -> {self.phase_target_y:+.2f}) "
            f"turn={self.turn_mode}")

    def _spin_toward(self, target_yaw):
        d = self._ang_diff(target_yaw, self.odom_yaw)
        if abs(d) < self.TURN_TOL_YAW:
            return 0.0, True
        return math.copysign(self.TURN_RATE, d), False

    def _turn_twist(self, t_now):
        """Scripted headland maneuver. Returns (twist, done, Tenth-leg info).
        done True means FOLLOW resumed (lane fields already updated)."""
        tw = Twist()
        if t_now - self.phase_t0 > self.TURN_TIMEOUT:
            return tw, "timeout", f"turn timeout in {self.phase}"
        if self.phase == "push":
            return self._push_twist(t_now)
        elif self.phase == "spin1":
            w, done = self._spin_toward(self.phase_target_yaw)
            tw.angular.z = w
            if done:
                self.phase_slide_yaw = self.phase_target_yaw
                self.phase, self.phase_t0 = "slide", t_now
        elif self.phase == "slide":
            err = self.phase_target_y - self.odom_y
            if abs(err) < self.TURN_TOL_Y:
                self.phase_target_yaw = 0.0 if self.drive_dir < 0 else math.pi
                self.phase, self.phase_t0 = "spin2", t_now
            else:
                tw.linear.x = self.TURN_DRIVE_V
                tw.angular.z = float(np.clip(
                    -1.5 * self._ang_diff(self.odom_yaw, self.phase_slide_yaw),
                    -0.4, 0.4))
        elif self.phase == "spin2":
            w, done = self._spin_toward(self.phase_target_yaw)
            tw.angular.z = w
            if done:
                return self._finish_turn(tw)
        else:
            return tw, "timeout", f"bad turn phase {self.phase}"
        return tw, "", ""

    def _push_twist(self, t_now):
        """Shared push-past-the-rows phase; branches to arc (fishtail) or
        spin1 (bulb) once past."""
        tw = Twist()
        tw.linear.x = self.TURN_DRIVE_V
        tw.angular.z = float(np.clip(
            -1.5 * self._ang_diff(self.odom_yaw, self.phase_yaw0),
            -0.4, 0.4))
        push_m = self.FISHTAIL_PUSH_M \
            if self.turn_mode == "fishtail" else self.TURN_PUSH_M
        if (self.odom_x - self.phase_x0) * self.drive_dir >= push_m:
            if self.turn_mode == "fishtail":
                self.phase, self.phase_t0 = "arc", t_now
                self.get_logger().info(
                    f"fishtail arc: side={self.phase_side:+.0f} "
                    f"to yaw {self.phase_arc_yaw:+.2f}")
            else:
                side = 1.0 if self.phase_target_y >= self.odom_y else -1.0
                self.phase_target_yaw = side * math.pi / 2.0
                self.phase, self.phase_t0 = "spin1", t_now
        return tw, "", ""

    def _finish_turn(self, tw):
        """Shared lane bookkeeping when a turn completes; resumes FOLLOW."""
        self.lane_index += self.turn_dir
        self.lane_y = float(self.furrows[self.lane_index])
        self.drive_dir *= -1
        self._reset_perception()
        self.phase = "follow"
        self.leg_x0 = self.odom_x
        self.leg_yaw = self.odom_yaw
        self.end_empty_n = 0
        self.get_logger().info(
            f"row change done: lane {self.lane_index} "
            f"(y={self.lane_y:+.2f}) dir={self.drive_dir:+d}")
        return tw, "follow", ""

    # --------------------------------------------------------------
    # Shuttle (no-turn row switching). Lanes alternate forward/front and
    # backward/rear; the primary camera swaps every lane. At the row end
    # the primary runs out of crops (empty); the rover drives straight
    # until the secondary agrees it is fully out, jogs laterally one
    # furrow spacing, swaps primary and drives back. The chassis never
    # turns around: yaw stays ~0 on every leg.
    SHUTTLE_END_FRAMES = 10
    SHUTTLE_END_CONF = 0.15
    SHUTTLE_EXIT_V = 0.15
    SHUTTLE_EXIT_M = 1.8
    SHUTTLE_BACKSTOP_M = 2.5
    SHUTTLE_JOG_V = 0.15
    SHUTTLE_CRAB = 0.45
    SHUTTLE_TOL_Y = 0.06
    SHUTTLE_TOL_YAW = 0.10
    SHUTTLE_TIMEOUT = 30.0

    @staticmethod
    def _pipe_locked(pipe_out, pipe_t, t_now):
        """Fresh, lined detection on a pipeline, else None."""
        if pipe_out is None or pipe_t is None:
            return None
        try:
            if (t_now - pipe_t) > 0.5:
                return None
            info = pipe_out.get("info", {})
            if not info.get("has_line", False):
                return None
            return info
        except Exception:
            return None

    def _secondary_empty(self, t_now):
        """True when the secondary camera sees no crops (row fully exited).

        Uses confidence, not has_line: at the row end the detector often
        sees crop fragments with confidence but forms no line, and that
        still means "not fully out". Stale/missing counts as empty.
        """
        try:
            if self.drive_dir > 0:
                out, pt = self.rear_out, self.rear_t
            else:
                out = {"info": self.front_info} if self.front_info else None
                pt = self.front_t
            if out is None or pt is None or (t_now - pt) > 0.5:
                return True
            return float(out["info"].get("confidence", 0)) < 0.25
        except Exception:
            return True

    def _primary_empty(self, out, t_now):
        """True when the primary camera sees no more crops this frame.

        Uses the filter status + confidence, not has_line: has_line
        flickers false for stretches mid-lane (nav fit gaps) while status
        stays accepted, which false-triggered row ends in sim. A true row
        end is persistently pending + low confidence.
        """
        try:
            if self.drive_dir > 0:
                info = out.get("info", {})
            else:
                locked = self._pipe_locked(self.rear_out, self.rear_t, t_now)
                info = locked if locked is not None else {}
                if locked is None:
                    return True
            if str(info.get("status", "")) != "accepted":
                return True
            return float(info.get("confidence", 0)) < self.SHUTTLE_END_CONF
        except Exception:
            return False

    def _shuttle_end_triggered(self, out, t_now):
        # vision trigger: primary empty for END_FRAMES straight, only
        # past 70% of the leg (row ends are at lane ends; vision picks
        # the exact point late in the leg, odometry vetoes mid-lane
        # false trips from filter flicker). Odometry backstop forces it
        # past the sidecar lane end (safety if rows outrun the map).
        if self.drive_dir > 0:
            if self.odom_x >= self.lane_end_x + self.SHUTTLE_BACKSTOP_M:
                return True
            if self.odom_x < self.leg_x0 + 0.7 * (self.lane_end_x
                                                 - self.leg_x0):
                self.end_empty_n = 0
                return False
        else:
            if self.odom_x <= self.lane_start_x - self.SHUTTLE_BACKSTOP_M:
                return True
            if self.odom_x > self.leg_x0 + 0.7 * (self.lane_start_x
                                                 - self.leg_x0):
                self.end_empty_n = 0
                return False
        if self._primary_empty(out, t_now):
            self.end_empty_n += 1
        else:
            self.end_empty_n = 0
        return self.end_empty_n >= self.SHUTTLE_END_FRAMES

    def _enter_exit(self, t_now, nxt):
        self.phase_target_y = float(self.furrows[nxt])
        self.phase = "exit"
        self.phase_t0 = t_now
        self.phase_x0, self.phase_y0 = self.odom_x, self.odom_y
        self.phase_yaw0 = self.odom_yaw
        self.leg_yaw = self.odom_yaw
        prim = "front" if self.drive_dir > 0 else "rear"
        sec = "rear" if self.drive_dir > 0 else "front"
        self.get_logger().info(
            f"row end ({prim} empty): exiting "
            f"{'+x' if self.drive_dir > 0 else '-x'}, secondary={sec}")

    def _shuttle_twist(self, t_now):
        """Exit straight, then crab one spacing sideways. Returns
        (twist, done, Tenth-leg info) like the other machines."""
        tw = Twist()
        if t_now - self.phase_t0 > self.SHUTTLE_TIMEOUT:
            return tw, "timeout", f"turn timeout in {self.phase}"
        if self.phase == "exit":
            # straight out, nose held; done when the secondary agrees the
            # row is fully out (no crops anywhere in its view) or the
            # overshoot cap.
            tw.linear.x = self.drive_dir * self.SHUTTLE_EXIT_V
            tw.angular.z = float(np.clip(
                -1.5 * self._ang_diff(self.odom_yaw, self.phase_yaw0),
                -0.4, 0.4))
            exited = self._secondary_empty(t_now)
            over = abs(self.odom_x - self.phase_x0) >= self.SHUTTLE_EXIT_M
            if exited or over:
                self.phase, self.phase_t0 = "jog", t_now
                self.get_logger().info(
                    f"shuttle jog: y {self.odom_y:+.2f} -> "
                    f"{self.phase_target_y:+.2f} "
                    f"({'secondary empty' if exited else 'overshoot cap'})")
        elif self.phase == "jog":
            # lateral crab toward the next furrow, backing toward the
            # field (not away): the jog ends parked at the row ends with
            # the rear camera facing the rows, so the backward leg starts
            # with a rear lock. The nose never turns around.
            side = 1.0 if self.phase_target_y >= self.odom_y else -1.0
            y_ok = abs(self.phase_target_y - self.odom_y) < self.SHUTTLE_TOL_Y
            if y_ok:
                yaw_sp = self.leg_yaw
            else:
                yaw_sp = self.leg_yaw - side * self.drive_dir * self.SHUTTLE_CRAB
            tw.linear.x = -self.drive_dir * self.SHUTTLE_JOG_V
            tw.angular.z = float(np.clip(
                0.8 * self._ang_diff(yaw_sp, self.odom_yaw), -0.4, 0.4))
            if y_ok and abs(self._ang_diff(self.odom_yaw, self.leg_yaw)) \
                    < self.SHUTTLE_TOL_YAW:
                return self._finish_turn(tw)
        else:
            return tw, "timeout", f"bad turn phase {self.phase}"
        return tw, "", ""

    def _lane_change_prelude(self):
        """Shared leg-completion prologue. Returns (next lane, stop_reason);
        stop_reason '' means keep going."""
        self.lanes_done += 1
        if self.max_lanes > 0 and self.lanes_done >= self.max_lanes:
            return None, (f"covered {self.lanes_done} lane(s), "
                          f"last y={self.lane_y:+.2f}")
        nxt = self.lane_index + self.turn_dir
        if not (0 <= nxt < len(self.furrows)):
            self.turn_dir *= -1
            nxt = self.lane_index + self.turn_dir
            if not (0 <= nxt < len(self.furrows)):
                return None, "no adjacent furrow to change into"
        return nxt, ""

    def _rear_base_err(self, t_now):
        """Lateral px error of the rear corridor at its base (image
        bottom = crops by the chassis), vs image center. Uses the raw
        accepted strip dots, not the fitted/filtered line: the fit goes
        diagonal on rear views while the dots sit in the corridor.
        None when the rear is stale or shows no low dots."""
        try:
            if self.rear_out is None or self.rear_t is None:
                return None
            if (t_now - self.rear_t) > 0.5:
                return None
            res = self.rear_out.get("res")
            if not res:
                return None
            dots = res.get("q_accepted", []) or []
            binary = res.get("binary")
            bh = binary.shape[0] if binary is not None else 480
            dx, _dy = res.get("crop_offset", (0, 0))
            low = [float(x) for x, y in dots if float(y) > 0.75 * float(bh)]
            if not low:
                return None
            return sum(low) / len(low) + float(dx) - 320.0
        except Exception:
            return None

    def _step_shuttle_follow(self, out, t_now):
        """One control tick on a shuttle leg. Forward legs servo the front
        camera; backward legs servo the rear corridor base (raw dots) and
        creep straight until dots show."""
        if self._shuttle_end_triggered(out, t_now):
            nxt, stop = self._lane_change_prelude()
            if stop:
                return Twist(), stop
            self._enter_exit(t_now, nxt)
            return Twist(), ""
        tw = Twist()
        if self.drive_dir > 0:
            tw.linear.x = float(np.clip(out["v"], 0.0, self.v_max))
            tw.angular.z = float(out["w"])
        else:
            base = self._rear_base_err(t_now)
            if base is not None:
                # reversing: the lane at +Y appears at +px, and moving
                # -x needs yaw<0 to gain +Y, so steer -k*err with a weak
                # nose hold for damping.
                w_lat = -1.5 * (float(base) / 320.0)
                w_hold = -0.3 * self._ang_diff(self.odom_yaw, self.leg_yaw)
                tw.linear.x = -0.15
                tw.angular.z = float(np.clip(w_lat + w_hold, -0.6, 0.6))
            else:
                tw.linear.x = -0.10
                tw.angular.z = float(np.clip(
                    -1.5 * self._ang_diff(self.odom_yaw, self.leg_yaw),
                    -0.4, 0.4))
        return tw, ""

    def _rear_locked(self, t_now):
        """Fresh rear detection usable for reverse steering, else None."""
        if self.rear_pipeline is None or self.rear_out is None:
            return None
        try:
            if self.rear_t is None or (t_now - self.rear_t) > 0.5:
                return None
            info = self.rear_out.get("info", {})
            if not info.get("has_line", False):
                return None
            return info
        except Exception:
            return None

    def _fishtail_twist(self, t_now):
        """Rear-guided reverse-in turn (no spinning). Returns (twist, done,
        Tenth-leg info); done True means FOLLOW resumed."""
        tw = Twist()
        if t_now - self.phase_t0 > self.FISHTAIL_TIMEOUT:
            return tw, "timeout", f"turn timeout in {self.phase}"
        if self.phase == "push":
            return self._push_twist(t_now)
        if self.phase == "arc":
            # forward arc AWAY from the next furrow until the nose has swung
            tw.linear.x = self.TURN_DRIVE_V
            tw.angular.z = -self.phase_side * self.drive_dir * self.TURN_RATE
            if abs(self._ang_diff(self.odom_yaw, self.phase_arc_yaw)) \
                    < self.FISHTAIL_TOL_YAW:
                self.phase, self.phase_t0 = "reverse", t_now
                self.phase_x0, self.phase_y0 = self.odom_x, self.odom_y
                try:
                    if self.rear_pipeline is not None:
                        self.rear_pipeline.reset()
                except Exception:
                    pass
                self.rear_out = None
                self.get_logger().info("fishtail reverse: backing in")
        elif self.phase == "reverse":
            tw.linear.x = -self.FISHTAIL_REVERSE_V
            info = self._rear_locked(t_now)
            if info is not None:
                # reversing moves the way the rear camera faces: reuse its
                # forward servo output negated (sign verified in sim).
                try:
                    tw.angular.z = float(np.clip(
                        self.rear_w_sign * float(self.rear_out["w"]),
                        -0.6, 0.6))
                except Exception:
                    tw.angular.z = 0.0
            else:
                # no rear lock yet: crab toward the furrow while converging
                # on the final heading. Reversing at yaw th moves laterally
                # vy = v*sin(th) with v<0, so hold the nose off the final
                # heading by d*k*e_y (e_y = target - y); that gains y at
                # ~|v|*k*e_y while the heading still converges.
                e_y = self.phase_target_y - self.odom_y
                crab = float(np.clip(self.drive_dir * self.FISHTAIL_CRAB_GAIN
                                     * e_y, -0.5, 0.5))
                yaw_sp = self.phase_final_yaw + crab
                d = self._ang_diff(yaw_sp, self.odom_yaw)
                tw.angular.z = float(np.clip(0.8 * d, -0.4, 0.4))
            backed = math.hypot(self.odom_x - self.phase_x0,
                                self.odom_y - self.phase_y0)
            yaw_ok = abs(self._ang_diff(self.odom_yaw,
                                        self.phase_final_yaw)) \
                < self.FISHTAIL_TOL_YAW
            y_ok = abs(self.phase_target_y - self.odom_y) < self.FISHTAIL_TOL_Y
            if self.rear_pipeline is not None:
                rear_ok = (info is not None
                           and abs(float(info.get("err_x", 1e9)))
                           < self.FISHTAIL_REAR_TOL_PX)
                # robust fallback: if the rear never locks (headland view,
                # CPU starvation), complete on odometry after a grace period
                # rather than timing out mid-field.
                if info is None:
                    if self.phase_no_rear_t is None:
                        self.phase_no_rear_t = t_now
                    elif t_now - self.phase_no_rear_t > 8.0:
                        if not getattr(self, "_no_rear_warned", False):
                            self._no_rear_warned = True
                            self.get_logger().warn(
                                "fishtail: no rear lock for 8s, finishing "
                                "on odometry")
                        rear_ok = True
                else:
                    self.phase_no_rear_t = None
            else:
                rear_ok = True
            if backed >= self.FISHTAIL_MIN_REVERSE_M and yaw_ok and y_ok \
                    and rear_ok:
                return self._finish_turn(tw)
        else:
            return tw, "timeout", f"bad turn phase {self.phase}"
        return tw, "", ""

    def _step_row_change(self, out, t_now):
        """One control tick in row-change mode. Returns (twist, stop_reason).
        stop_reason "" means keep driving."""
        if self.phase == "follow":
            if self.turn_mode == "shuttle":
                return self._step_shuttle_follow(out, t_now)
            if self._lane_end_reached():
                nxt, stop = self._lane_change_prelude()
                if stop:
                    return Twist(), stop
                self._enter_turn(t_now)
                tw = Twist()
                return tw, ""
            tw = Twist()
            tw.linear.x = float(np.clip(out["v"], 0.0, self.v_max))
            tw.angular.z = float(out["w"])
            return tw, ""
        if self.turn_mode == "fishtail":
            tw, status, reason = self._fishtail_twist(t_now)
        elif self.turn_mode == "shuttle":
            tw, status, reason = self._shuttle_twist(t_now)
        else:
            tw, status, reason = self._turn_twist(t_now)
        if status == "timeout":
            return Twist(), reason
        return tw, ""

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
    def _decode_bgr(self, msg: Image):
        """Image msg -> BGR array, or None for unsupported encodings."""
        enc = msg.encoding
        if enc in ("bgr8", "8UC3"):
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if enc in ("rgb8",):
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.get_logger().warn(f"unsupported encoding {enc}")
        return None

    def _on_rear_image(self, msg: Image):
        if self.rear_pipeline is None:
            return
        # the rear view is only processed when some consumer needs it:
        # fishtail arc/reverse phases (gated so the front keeps full CPU
        # while following). Shuttle always runs it: the filter must be
        # warm on clear rows mid-lane, otherwise it cold-starts at the
        # row end and never locks (seen in sim).
        if self.turn_mode == "fishtail":
            if self.phase not in ("arc", "reverse"):
                return
        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            bgr = self._decode_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"rear decode: {e}")
            return
        if bgr is None:
            return
        try:
            t0 = time.perf_counter()
            out = self.rear_pipeline.process(bgr)
            self.rear_ms = 1000.0 * (time.perf_counter() - t0)
        except Exception as e:
            self.get_logger().error(f"rear pipeline failed: {e}")
            return
        self.rear_out = out
        self.rear_t = t_now
        self.rear_frame_idx += 1
        # save rear views through the turn (front overlay keeps publishing,
        # but nothing subscribes to the rear one mid-run)
        try:
            if self.log_dir and self.rear_frame_idx % 10 == 0:
                cv2.imwrite(str(self.log_dir /
                                f"rear_{self.rear_frame_idx:05d}.png"),
                            out["overlay"])
        except Exception:
            pass
        if self.rear_ovl_pub is not None \
                and self.rear_ovl_pub.get_subscription_count() > 0:
            try:
                rgb = cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB)
                im = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
                im.header = msg.header
                self.rear_ovl_pub.publish(im)
            except Exception as e:  # pragma: no cover
                self.get_logger().warn(f"rear overlay pub: {e}")

    def _rear_csv(self):
        try:
            if self.rear_out is not None:
                info = self.rear_out.get("info", {})
                return [f"{info.get('err_x', 0):.1f}",
                        f"{info.get('raw_err_theta_deg', info.get('err_theta_deg', 0)):.2f}",
                        f"{info.get('confidence', 0):.2f}"]
        except Exception:
            pass
        return ["", "", ""]

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
        try:
            bgr = self._decode_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"decode: {e}")
            return
        if bgr is None:
            return

        try:
            out = self.pipeline.process(bgr)
        except Exception as e:
            self.get_logger().error(f"pipeline {self.pipeline.name} failed: {e}")
            return
        self.last_w = float(out["w"])
        info = out["info"]
        self.front_info = info
        self.front_t = t_now

        # --- command (skipped in teleop idle mode) ---
        twist = Twist()
        twist.linear.x = float(np.clip(out["v"], 0.0, self.v_max))
        twist.angular.z = float(out["w"])
        stop_reason = ""
        if self.row_change and self.circle_r <= 0.0 and not self.idle:
            twist, stop_reason = self._step_row_change(out, t_now)
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

        # --- logging (v/w are the commanded twist, so backward shuttle
        # legs read correctly too) ---
        cross = self._cross_track()
        if self.log_dir:
            self.csv_w.writerow([
                f"{t_now:.3f}", f"{self.odom_x:.3f}", f"{self.odom_y:.3f}",
                f"{cross:.3f}",
                f"{info.get('err_x', 0):.1f}",
                f"{info.get('raw_err_theta_deg', info.get('err_theta_deg', 0)):.2f}",
                f"{info.get('filt_err_theta_deg', 0):.2f}",
                f"{info.get('confidence', 0):.2f}", str(info.get("status", "")),
                f"{self._last_cmd.linear.x:.3f}",
                f"{self._last_cmd.angular.z:.3f}",
                int(info.get("n_two_sided", 0)), f"{info.get('ff', 0):.4f}",
                *self._rear_csv()])
            self.csv_f.flush()
            if ovl is not None and self.frame_idx % self.save_every == 0:
                cv2.imwrite(str(self.log_dir / f"frame_{self.frame_idx:05d}.png"), ovl)

        if self.frame_idx % 60 == 0:
            lap_txt = (f" lap={abs(self._lap_angle) / (2.0 * math.pi):.2f}"
                       if self.circle_r > 0 else "")
            leg_txt = (f" leg={self.lanes_done + 1} lane={self.lane_index} "
                       f"ph={self.phase}"
                       if self.row_change and self.circle_r <= 0.0 else "")
            if self.turn_mode == "shuttle" and self.row_change \
                    and self.circle_r <= 0.0:
                leg_txt += f" prim={'R' if self.drive_dir < 0 else 'F'}"
            rear_txt = ""
            if self.turn_mode in ("fishtail", "shuttle") and (
                    self.phase in ("arc", "reverse", "exit", "jog")
                    or (self.turn_mode == "shuttle" and self.drive_dir < 0)):
                try:
                    rinfo = (self.rear_out or {}).get("info", {})
                    age = (t_now - self.rear_t) if self.rear_t else -1.0
                    base = self._rear_base_err(t_now)
                    btxt = f"{base:+.0f}" if base is not None else "--"
                    rear_txt = (f" rear_c={rinfo.get('confidence', 0):.2f} "
                                f"rbase={btxt} rage={age:.1f}s "
                                f"rn={self.rear_frame_idx} "
                                f"rms={self.rear_ms:.0f}ms")
                except Exception:
                    pass
            self.get_logger().info(
                f"[{self.frame_idx:>4}] t={t_now:6.2f} odom=({self.odom_x:+.2f},"
                f"{self.odom_y:+.2f}) cross={cross:+.3f}{lap_txt}{leg_txt} "
                f"w={twist.angular.z:+.2f} conf={info.get('confidence', 0):.2f} "
                f"status={info.get('status', '')}{rear_txt}")

        # auto-termination is disabled while idling (teleop decides when done)
        if self.idle:
            return

        # termination conditions
        if self.max_seconds > 0 and (t_now - self.t0) >= self.max_seconds:
            self._stop_robot(f"reached max_seconds={self.max_seconds}")
        elif stop_reason:
            self._stop_robot(stop_reason)
        elif self.circle_r > 0:
            if self.max_laps > 0 and abs(self._lap_angle) >= self.max_laps * 2.0 * math.pi:
                self._stop_robot(
                    f"completed {self.max_laps:g} lap(s) "
                    f"(radial err {cross:+.2f} m)")
            elif self.frame_idx > 60000:
                self._stop_robot("frame cap")
        elif self.row_change:
            if self.frame_idx > 60000:
                self._stop_robot("frame cap")
            # lane ends handled by the row-change machine (legs/lanes bound it)
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
