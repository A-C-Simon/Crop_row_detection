#!/usr/bin/env python3
"""ToF crop-safety guard for the Gazebo rover rigs (mrsim / fftsim / exgsim).

Four Time-of-Flight rangers watch the crop rows: sideways
(``/tof/left``, ``/tof/right``) plus headlamp-style front modules angled
+/-35 deg outward (``/tof/front_left``, ``/tof/front_right``), which see
head-on drift into plants at bends ~0.7 m ahead, before the side rangers
react. All publish ``sensor_msgs/Range`` (infrared). While every reading
stays above the safety threshold the guard is transparent: the navigation
stack's command on ``<in>`` (default ``/cmd_vel_raw``) is republished
unchanged on ``<out>`` (default ``/cmd_vel``).

When a side violates the minimum clearance the guard OVERRIDES the
angular command (crop safety is high priority - the vision servo output
is discarded for that tick) and steers away from the close side. Each
side uses the nearer of its two rangers (side or front):

    deficit_L = max(0, tof_min - min(left, front_left))
    deficit_R = max(0, tof_min - min(right, front_right))
    w_safe    = clip(-gain * deficit_L + gain * deficit_R, +/- max_w)
    v_safe    = min(raw_v, tof_v)

i.e. too close on the left -> negative (rightward) yaw, too close on the
right -> positive (leftward) yaw, both sides -> slow creep toward the
larger clearance. Linear speed is capped at ``tof_v`` while overriding.

The ranger topics are subscribed with ``raw=True`` and parsed by hand
(see ``parse_range_cdr``). This is deliberate: the Gazebo ray plugin
reports "no return" as float32 FLT_MAX, and this image's
``rosidl_generator_py`` Range converter rejects exactly that value (its
float32 bound check ``<= 3.402823466e+38`` excludes true FLT_MAX
``3.4028234663852886e+38``), which kills any normally-subscribed Python
node - including ``ros2 topic echo /tof/left`` - with
``RuntimeError: Unable to convert call argument to Python object``.
Parsing the CDR bytes with ``struct`` has no such check, so the guard is
immune. (Use the guard's ``tof_guard.csv`` for telemetry, not echo.)

Configuration is via env vars (set by farm.launch.py ``tof*`` args, with
sane defaults so ``ros2 run <pkg> tof_guard`` also works standalone):

    MRSIM_TOF_MIN     minimum side clearance in m (default 0.35)
    MRSIM_TOF_GAIN    yaw gain rad/s per m of deficit (default 2.0)
    MRSIM_TOF_MAX_W   |w| clamp while overriding (default 0.6)
    MRSIM_TOF_V       linear cap while overriding (default 0.12)
    MRSIM_TOF_IN      raw cmd topic (default /cmd_vel_raw)
    MRSIM_TOF_OUT     safe cmd topic (default /cmd_vel)
    MRSIM_TOF_LEFT    left ranger topic (default /tof/left)
    MRSIM_TOF_RIGHT   right ranger topic (default /tof/right)
    MRSIM_TOF_FL      front-left ranger topic (default /tof/front_left)
    MRSIM_TOF_FR      front-right ranger topic (default /tof/front_right)
    MRSIM_TOF_STALE   ranger staleness timeout s (default 0.5)
    MRSIM_LOG_DIR     when set, appends tof_guard.csv telemetry
"""
import csv
import math
import os
import struct
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range  # only for topic type matching (raw take)
from geometry_msgs.msg import Twist


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return str(v) if v != "" else default


def parse_range_cdr(data: bytes, expect_frame=None):
    """Parse CDR (little-endian) bytes of a sensor_msgs/Range message.

    Returns the range in meters (``inf`` = nothing within max_range =
    clear) or ``None`` when the payload cannot be parsed/validated.

    Layout: encap(4) + stamp(8) + strlen u32 + frame chars + null +
    radiation_type u8, aligned to 4, then fov/min/max/range as f32 LE.
    ``struct`` converts inf/FLT_MAX/NaN without complaint.
    """
    try:
        # CDR encapsulation header, first two bytes: 0x0001 means
        # little-endian plain CDR, which is what FastDDS writes here.
        # Anything else (big-endian, delimited CDR) is rejected as stale
        # rather than misparsed.
        if len(data) < 16 or data[0] != 0 or data[1] != 1:
            return None
        (strlen,) = struct.unpack_from("<I", data, 12)
        if not (1 <= strlen <= 128):
            return None
        fend = 16 + strlen
        off = (fend + 1 + 3) & ~3  # radiation_type byte, then align to 4
        # radiation_type + 4 floats after the null-terminated frame
        if len(data) < off + 16:
            return None
        frame = data[16:fend - 1].decode("ascii")
        if expect_frame is not None and frame != expect_frame:
            return None
        _fov, mn, mx, rng = struct.unpack_from("<4f", data, off)
        if not all(math.isfinite(v) for v in (mn, mx)):
            return None
        if not (0.0 <= mn < mx <= 100.0):
            return None
        if math.isnan(rng):
            return math.inf  # invalid reading: fail open (as stale would)
        if math.isinf(rng):
            return math.inf if rng > 0 else None
        if rng < 0.0:
            return None
        if rng >= mx - 1e-6:
            return math.inf  # at/above max range (incl. FLT_MAX) = clear
        return float(rng)
    except Exception:
        return None


def compute_override(left: float, right: float, raw_v: float, raw_w: float,
                     tof_min: float, gain: float, max_w: float,
                     v_cap: float):
    """Pure override law. ``left``/``right`` are meters (inf = clear).

    Returns (v_out, w_out, overriding: bool, reason: str).
    """
    dl = max(0.0, tof_min - left) if math.isfinite(left) else 0.0
    dr = max(0.0, tof_min - right) if math.isfinite(right) else 0.0
    if dl <= 0.0 and dr <= 0.0:
        return raw_v, raw_w, False, ""
    w_safe = -gain * dl + gain * dr
    w_safe = float(max(-max_w, min(max_w, w_safe)))
    v_safe = float(min(raw_v, v_cap))
    side = "left" if dl > dr else ("right" if dr > dl else "both")
    return v_safe, w_safe, True, f"tof {side} close"


def nearest_fresh(pairs, stale: float, now: float) -> float:
    """Nearest reading among (value, timestamp) pairs that is fresh
    (age <= stale s), else inf. Pure function for testability."""
    best = math.inf
    for v, t in pairs:
        if t is not None and (now - t) <= stale and v < best:
            best = v
    return best


class ToFGuard(Node):
    def __init__(self):
        super().__init__("tof_guard")
        self.tof_min = _env_float("MRSIM_TOF_MIN", 0.35)
        self.gain = _env_float("MRSIM_TOF_GAIN", 2.0)
        self.max_w = _env_float("MRSIM_TOF_MAX_W", 0.6)
        self.v_cap = _env_float("MRSIM_TOF_V", 0.12)
        self.stale = _env_float("MRSIM_TOF_STALE", 0.5)
        in_topic = _env_str("MRSIM_TOF_IN", "/cmd_vel_raw")
        out_topic = _env_str("MRSIM_TOF_OUT", "/cmd_vel")
        left_topic = _env_str("MRSIM_TOF_LEFT", "/tof/left")
        right_topic = _env_str("MRSIM_TOF_RIGHT", "/tof/right")
        fl_topic = _env_str("MRSIM_TOF_FL", "/tof/front_left")
        fr_topic = _env_str("MRSIM_TOF_FR", "/tof/front_right")

        self.left = self.right = self.fl = self.fr = math.inf
        self.left_t = self.right_t = self.fl_t = self.fr_t = None
        self.n_raw = 0
        self.n_override = 0
        self.n_bad = 0
        self._was_overriding = False
        self._last_warn = 0.0

        log_dir = os.environ.get("MRSIM_LOG_DIR", "")
        self.csv_w = None
        self.csv_f = None
        if log_dir:
            try:
                p = Path(log_dir)
                p.mkdir(parents=True, exist_ok=True)
                self.csv_f = open(p / "tof_guard.csv", "w", newline="")
                self.csv_w = csv.writer(self.csv_f)
                self.csv_w.writerow(["t", "left_m", "right_m", "fl_m",
                                     "fr_m", "raw_v", "raw_w", "out_v",
                                     "out_w", "override"])
                self.csv_f.flush()
            except Exception as e:
                self.get_logger().warn(f"tof guard csv disabled: {e}")

        self.pub = self.create_publisher(Twist, out_topic, 10)
        # raw takes: immune to the rosidl FLT_MAX assert bug (see module
        # docstring); the Range class is only used for DDS type matching.
        self.create_subscription(Range, left_topic, self._on_left_raw, 10,
                                 raw=True)
        self.create_subscription(Range, right_topic, self._on_right_raw, 10,
                                 raw=True)
        self.create_subscription(Range, fl_topic, self._on_fl_raw, 10,
                                 raw=True)
        self.create_subscription(Range, fr_topic, self._on_fr_raw, 10,
                                 raw=True)
        self.create_subscription(Twist, in_topic, self._on_cmd, 10)
        self.get_logger().info(
            f"ToF guard on: in={in_topic} out={out_topic} "
            f"left={left_topic} right={right_topic} "
            f"front_left={fl_topic} front_right={fr_topic} "
            f"min={self.tof_min:.2f}m gain={self.gain:.2f} "
            f"max_w={self.max_w:.2f} v_cap={self.v_cap:.2f}")

    # --------------------------------------------------------------
    _FRAMES = {"left": "tof_left_link", "right": "tof_right_link",
               "fl": "tof_fl_link", "fr": "tof_fr_link"}

    def _on_range_bytes(self, data: bytes, side: str):
        rng = parse_range_cdr(bytes(data),
                              expect_frame=self._FRAMES.get(side))
        now = time.monotonic()
        if rng is None:
            self.n_bad += 1
            if now - self._last_warn > 5.0:
                self._last_warn = now
                self.get_logger().warn(
                    f"tof guard: bad {side} payload "
                    f"(n_bad={self.n_bad}), treating as stale")
            return
        if side == "left":
            self.left, self.left_t = rng, now
        elif side == "right":
            self.right, self.right_t = rng, now
        elif side == "fl":
            self.fl, self.fl_t = rng, now
        else:
            self.fr, self.fr_t = rng, now

    def _on_left_raw(self, data: bytes):
        self._on_range_bytes(data, "left")

    def _on_right_raw(self, data: bytes):
        self._on_range_bytes(data, "right")

    def _on_fl_raw(self, data: bytes):
        self._on_range_bytes(data, "fl")

    def _on_fr_raw(self, data: bytes):
        self._on_range_bytes(data, "fr")

    def _fresh(self, t):
        if t is None:
            return False
        return (time.monotonic() - t) <= self.stale

    @staticmethod
    def _fm(v):
        return f"{v:.2f}m" if math.isfinite(v) else "inf"

    def _on_cmd(self, msg: Twist):
        now = time.monotonic()
        # Fuse each side down to one clearance: the nearer of its side and
        # front ranger wins, so a bend seen only by the angled front module
        # still triggers the override for that side. Stale or missing
        # rangers read as inf (fail open) instead of blocking the nav.
        left = nearest_fresh(((self.left, self.left_t),
                              (self.fl, self.fl_t)), self.stale, now)
        right = nearest_fresh(((self.right, self.right_t),
                               (self.fr, self.fr_t)), self.stale, now)
        fl = self.fl if self._fresh(self.fl_t) else math.inf
        fr = self.fr if self._fresh(self.fr_t) else math.inf
        if (self.left_t is None or self.right_t is None
                or self.fl_t is None or self.fr_t is None) \
                and now - self._last_warn > 5.0:
            self._last_warn = now
            self.get_logger().warn(
                "tof guard: no ranger msgs yet, passing through")
        raw_v = float(msg.linear.x)
        raw_w = float(msg.angular.z)
        # compute_override returns its own side label, but here the log
        # names the closest violating ranger instead (see `who` below),
        # so the label is intentionally discarded.
        v, w, overriding, _ = compute_override(
            left, right, raw_v, raw_w,
            self.tof_min, self.gain, self.max_w, self.v_cap)
        out = Twist()
        out.linear.x = v
        out.angular.z = w
        self.pub.publish(out)
        self.n_raw += 1
        if overriding:
            self.n_override += 1
        if overriding != self._was_overriding:
            self._was_overriding = overriding
            if overriding:
                # name the closest violating ranger for the log
                cands = (("left", self.left if self._fresh(self.left_t)
                          else math.inf),
                         ("right", self.right if self._fresh(self.right_t)
                          else math.inf),
                         ("front-left", fl), ("front-right", fr))
                who = min(
                    (n for n, d in cands
                     if math.isfinite(d) and d < self.tof_min),
                    default="tof")
                self.get_logger().warn(
                    f"ToF OVERRIDE {who} close: "
                    f"L={self._fm(left)} R={self._fm(right)} "
                    f"FL={self._fm(fl)} FR={self._fm(fr)} "
                    f"raw=({raw_v:.2f},{raw_w:+.2f}) -> "
                    f"safe=({v:.2f},{w:+.2f})")
            else:
                self.get_logger().info("ToF clear: resuming nav commands")
        if self.csv_w is not None:
            try:
                self.csv_w.writerow([
                    f"{now:.3f}",
                    f"{left:.3f}" if math.isfinite(left) else "inf",
                    f"{right:.3f}" if math.isfinite(right) else "inf",
                    f"{fl:.3f}" if math.isfinite(fl) else "inf",
                    f"{fr:.3f}" if math.isfinite(fr) else "inf",
                    f"{raw_v:.3f}", f"{raw_w:.3f}",
                    f"{v:.3f}", f"{w:.3f}", int(overriding)])
                if self.n_raw % 20 == 0:
                    self.csv_f.flush()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = ToFGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            zero = Twist()
            node.pub.publish(zero)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
