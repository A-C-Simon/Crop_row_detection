#!/usr/bin/env python3
"""ExG crop-row navigation node for the Gazebo test rig.

Subscribes to the simulated rover's front camera (/camera/image_raw), runs the
Python ExG pipeline - `results/run_vcrn_debug.py` + `results/exg_window.py` -
and publishes /cmd_vel to the gazebo diff-drive plugin, closed loop.

Window behaviour (`results/exg_window.py`, the base-anchored column-aware
tracker):

  * the window is **pinned to the base of the frame** - its bottom edge sits on
    the chassis-forward reference (the MultiROI blue star at
    (W/2, H - base_margin)) - and never drifts up the image;
  * **acquisition** latches the crop column nearest the image centre
    (bottom-band projection peaks, width from the median inter-row gap);
  * steering while driving forward slides that latched column to the
    bottom-centre reference, aligning the chassis with it;
  * once aligned for `lock_frames`, the window **locks at bottom-centre**;
  * the window **re-latches** when the column leaves the base band (end of the
    crop column, i.e. the headland/lane transition) or the window runs empty.

Steering uses the fitted row's position at the bottom-of-frame chassis
reference (`line_base_error`), because the ExG vehicle rides above the rows.

Also:
  - publishes VsMsg on /vs_msg so the rig monitor works unchanged,
  - publishes the drawn overlay on /exgsim/nav_overlay (rgb8),
  - writes <log_dir>/exg_nav.csv (window state per frame) + overlay PNGs.

Usage (after colcon build + source, with EXG_DIR set):
  ros2 run exgsim exg_nav --ros-args -p log_dir:=/tmp/exgsim_nav
  or see launch/farm.launch.py (nav:=exg).
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

try:  # VsMsg is optional: the monitor uses it, driving does not need it
    from visual_crop_row_navigation_ros2.msg import VsMsg
except Exception:  # pragma: no cover - exg_ws not sourced
    VsMsg = None


# ---------------------------------------------------------------------------
# locate the ExG Python pipeline (results/) the same way the rig finds sources
# ---------------------------------------------------------------------------
def _find_results_dir():
    env = os.environ.get("EXG_DIR")
    if env:
        c = Path(env) / "visual-crop-row-navigation_ros2" / "results"
        if (c / "exg_window.py").exists():
            return c
    p = Path(__file__).resolve()
    for _ in range(8):
        c = p / "visual-crop-row-navigation_ros2" / "results"
        if (c / "exg_window.py").exists():
            return c
        p = p.parent
    return None


_RESULTS = _find_results_dir()
if _RESULTS is None:
    raise RuntimeError("cannot locate ExG results/ pipeline; set EXG_DIR env")
sys.path.insert(0, str(_RESULTS))

from exg_window import (BaseColumnWindow, ACQUIRE, LOCKED,  # noqa: E402
                        line_base_error, steer_from_base)
from run_vcrn_debug import fit_line_clip, load_params  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402


def _sensor_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST,
                      depth=2,
                      durability=DurabilityPolicy.VOLATILE)


class ExGNavNode(Node):
    def __init__(self):
        super().__init__("exg_nav")
        self.bridge = CvBridge()

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
        # teleop/demo: keep detection alive but never publish /cmd_vel
        self.idle = str(os.environ.get("MRSIM_NAV_IDLE", _p("nav_idle", ""))) == "1"
        if self.idle:
            self.get_logger().info("nav node IDLE (teleop/demo): no /cmd_vel")

        # parameters: sim-tuned params by default, results/ params as fallback
        default_params = str(
            Path(__file__).resolve().parent.parent / "params" / "exgsim_run.yaml")
        params_file = str(os.environ.get(
            "MRSIM_EXG_PARAMS", _p("params_file", default_params)))
        if not Path(params_file).exists():
            params_file = str(_RESULTS.parent / "params" / "agribot_vs_run.yaml")
        self.params = load_params(Path(params_file))
        self.get_logger().info(f"ExG params: {params_file}")

        self.tracker = BaseColumnWindow(self.params)
        self.v_max = float(os.environ.get("MRSIM_V_MAX",
                                          _p("v_max", self.params["vf_des"])))

        log_dir = os.environ.get("MRSIM_LOG_DIR", _p("log_dir", ""))
        self.save_every = int(os.environ.get("MRSIM_SAVE_EVERY",
                                             _p("save_every", 20)))
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_f = open(self.log_dir / "exg_nav.csv", "w", newline="")
            self.csv_w = csv.writer(self.csv_f)
            self.csv_w.writerow(["sim_t", "odom_x", "odom_y", "cross_track",
                                 "err_x_px", "err_theta_deg", "v", "w",
                                 "state", "locked", "column_x", "n_nh",
                                 "relatched"])
            self.get_logger().info(f"logging to {self.log_dir}")
        else:
            self.csv_f = None

        self.odom_x = self.odom_y = 0.0
        self.last_img_t = None
        self.t0 = None
        self.frame_idx = 0
        self._relatch_n = 0
        self._last_cmd = Twist()
        # when the column vanishes mid-lane (transient) creep forward to
        # re-acquire instead of stalling; stop once the search times out or
        # the lane end is reached.
        self.search_v = float(os.environ.get("MRSIM_SEARCH_V",
                                             _p("search_v", 0.06)))
        self.search_seconds = float(os.environ.get("MRSIM_SEARCH_SECONDS",
                                                   _p("search_seconds", 4.0)))
        self._lost_since = None
        self._searching = False

        self.cam_sub = self.create_subscription(
            Image, cam_topic, self._on_image, _sensor_qos())
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.ovl_pub = self.create_publisher(Image, "/exgsim/nav_overlay", 5)
        self.vs_pub = self.create_publisher(VsMsg, "/vs_msg", 10) \
            if VsMsg is not None else None
        if self.vs_pub is None:
            self.get_logger().warn(
                "VsMsg unavailable (source exg_ws); monitor will see no /vs_msg")
        self.get_logger().info(
            f"ExG nav on '{cam_topic}' -> '{cmd_topic}' | window base-anchored "
            f"(state={self.tracker.state}) v_max={self.v_max:.2f}")

    # --------------------------------------------------------------
    def _on_odom(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def _cross_track(self):
        return self.odom_y - self.lane_y

    def _decode_bgr(self, msg: Image):
        if msg.encoding in ("bgr8", "8UC3"):
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if msg.encoding == "rgb8":
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.get_logger().warn(f"unsupported encoding {msg.encoding}")
        return None

    # --------------------------------------------------------------
    def _detect(self, bgr):
        """One pipeline pass: mask -> centres -> base window -> fit -> command."""
        p = self.params
        width, height = p["width"], p["height"]
        img = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        combined = cv2.inRange(h_chan, p["min_Hue"], p["max_Hue"])
        combined = cv2.bitwise_and(combined, cv2.inRange(
            s_chan, p["min_Saturation"], p["max_Saturation"]))
        combined = cv2.bitwise_and(combined, cv2.inRange(
            v_chan, p["min_Value"], p["max_Value"]))

        contours, _ = cv2.findContours(combined, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            poly = cv2.approxPolyDP(cnt, 2, True)
            (cx, cy), _rad = cv2.minEnclosingCircle(poly)
            centers.append((float(cx), float(cy)))

        win = self.tracker.update(combined, centers)
        Xc, Yc, L, H = win["Xc"], win["Yc"], win["L"], win["H"]
        nh_points = [(x, y) for (x, y) in centers
                     if (Xc - L / 2 < x < Xc + L / 2)
                     and (Yc - H / 2 < y < Yc + H / 2)]

        inliers = nh_points
        if (p.get("iso_enabled", True)
                and len(nh_points) >= p.get("iso_min_points", 12)):
            try:
                pts = np.array(nh_points, dtype=np.float32)
                iso = IsolationForest(
                    contamination=p.get("iso_contamination", 0.15),
                    n_estimators=p.get("iso_n_estimators", 100),
                    random_state=42)
                pred = iso.fit_predict(pts)
                inliers = [tuple(q) for q, m in zip(nh_points, pred == 1) if m]
            except Exception:
                inliers = nh_points

        fit_info = fit_line_clip(inliers, width, height) if inliers else None
        has_line = bool(fit_info and len(fit_info["inside"]) >= 2)

        err_x = err_theta = None
        w_cmd = 0.0
        v_cmd = 0.0
        if has_line:
            err_x, err_theta = line_base_error(
                fit_info, width, height, y_ref=win["ref_y"],
                base_margin=p.get("base_margin", 10.0))
            if err_x is not None:
                w_cmd = steer_from_base(
                    err_x, err_theta, width,
                    kx=p.get("base_kx", 0.9), kth=p.get("base_kth", 1.0),
                    w_max=p.get("base_w_max", 0.6))
                v_cmd = min(self.v_max, p.get("vf_des", 0.2))

        overlay = self._draw(img, centers, nh_points, inliers, fit_info,
                             win, v_cmd, w_cmd, err_x, err_theta)
        return dict(win=win, n_nh=len(nh_points), fit_info=fit_info,
                    has_line=has_line, err_x=err_x, err_theta=err_theta,
                    v=v_cmd, w=w_cmd, overlay=overlay)

    def _draw(self, img, centers, nh_points, inliers, fit_info, win,
              v, w, err_x, err_theta):
        out = img.copy()
        W, H = self.params["width"], self.params["height"]
        Xc, Yc, L, Hw = win["Xc"], win["Yc"], win["L"], win["H"]
        # base reference (blue star) + window
        cv2.drawMarker(out, (W // 2, win["ref_y"]), (255, 255, 0),
                       cv2.MARKER_STAR, 20, 2)
        colour = (0, 255, 0) if win["locked"] else (0, 165, 255)
        cv2.rectangle(out, (int(Xc - L / 2), int(Yc - Hw / 2)),
                      (int(Xc + L / 2), int(Yc + Hw / 2)), colour, 3)
        for (x, y) in centers:
            cv2.circle(out, (int(round(x)), int(round(y))), 2, (110, 110, 110),
                       cv2.FILLED)
        for (x, y) in nh_points:
            cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 204, 255),
                       cv2.FILLED)
        if fit_info is not None:
            inside = fit_info["inside"]
            if len(inside) >= 2:
                cv2.line(out, (int(round(inside[0][0])), int(round(inside[0][1]))),
                         (int(round(inside[1][0])), int(round(inside[1][1]))),
                         (0, 0, 255), 2, cv2.LINE_AA)
        if win.get("column_x") is not None and not win["locked"]:
            cv2.line(out, (int(win["column_x"]), 0),
                     (int(win["column_x"]), H), (0, 255, 255), 1)
        txt = (f"{win['state']} v={v:.2f} w={math.degrees(w):.1f}deg "
               f"base_err={err_x:.0f}px" if err_x is not None
               else f"{win['state']} no line")
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"nh={len(nh_points)} "
                         f"col={win.get('column_x')}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return out

    # --------------------------------------------------------------
    def _on_image(self, msg: Image):
        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t_now
        self.frame_idx += 1
        try:
            bgr = self._decode_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"decode: {e}")
            return
        if bgr is None:
            return

        try:
            out = self._detect(bgr)
        except Exception as e:
            self.get_logger().error(f"pipeline failed: {e}")
            return

        win = out["win"]
        if win.get("relatched"):
            self._relatch_n += 1
            self.get_logger().info(
                f"re-latch #{self._relatch_n}: column lost, acquiring again "
                f"(x={self.odom_x:.2f})")

        # no line: creep straight to re-acquire, or stop once the column has
        # been gone past the search window (a true lane end)
        if out["has_line"]:
            self._lost_since = None
            self._searching = False
        else:
            if self._lost_since is None:
                self._lost_since = t_now
            searching = (t_now - self._lost_since) <= self.search_seconds
            if searching != self._searching:
                self._searching = searching
                self.get_logger().info(
                    "no line: " + ("creeping to re-acquire" if searching
                                   else "search timed out"))
            out["v"] = self.search_v if searching else 0.0
            out["w"] = 0.0

        twist = Twist()
        if not self.idle:
            twist.linear.x = float(out["v"])
            twist.angular.z = float(out["w"])
            self.cmd_pub.publish(twist)
            self._last_cmd = twist

        # /vs_msg so the rig monitor logs err_x / err_theta unchanged
        if self.vs_pub is not None:
            vm = VsMsg()
            vm.err_x = float(out["err_x"] or 0.0)
            vm.err_theta = float(out["err_theta"] or 0.0)
            self.vs_pub.publish(vm)

        ovl = out["overlay"]
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
                f"{out['err_x']:.1f}" if out["err_x"] is not None else "",
                f"{math.degrees(out['err_theta']):.2f}"
                if out["err_theta"] is not None else "",
                f"{out['v']:.3f}", f"{out['w']:.3f}", win["state"],
                int(win["locked"]),
                win.get("column_x") if win.get("column_x") is not None else "",
                out["n_nh"], int(bool(win.get("relatched")))])
            self.csv_f.flush()
            if self.frame_idx % self.save_every == 0:
                cv2.imwrite(str(self.log_dir / f"exg_{self.frame_idx:05d}.png"),
                            ovl)

        if self.frame_idx % 60 == 0:
            self.get_logger().info(
                f"[{self.frame_idx:>4}] t={t_now:6.2f} "
                f"odom=({self.odom_x:+.2f},{self.odom_y:+.2f}) "
                f"cross={self._cross_track():+.3f} state={win['state']} "
                f"col={win.get('column_x')} nh={out['n_nh']} "
                f"err={out['err_x'] if out['err_x'] is None else round(out['err_x'],1)}px "
                f"v={out['v']:.2f} w={out['w']:+.2f}")

        if self.idle:
            return
        if self.max_seconds > 0 and (t_now - self.t0) >= self.max_seconds:
            self._stop(f"reached max_seconds={self.max_seconds}")
        elif self.odom_x >= self.lane_end_x:
            self._stop(f"reached end of lane (x={self.odom_x:.2f})")
        elif (self._lost_since is not None
              and (t_now - self._lost_since) > self.search_seconds
              and not out["has_line"]) and self.frame_idx > 10:
            self._stop("column lost (crop column ended)")
        elif self.frame_idx > 60000:
            self._stop("frame cap")

    def _stop(self, reason):
        self.get_logger().info(f"stop: {reason}")
        self.cmd_pub.publish(Twist())
        if self.csv_f:
            self.csv_f.close()
        self.destroy_node()

    def destroy_node(self):
        if self.csv_f and not self.csv_f.closed:
            self.csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExGNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
