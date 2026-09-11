#!/usr/bin/env python3
"""Rover reset helper shared by the mrsim / fftsim / exgsim rigs.

Two uses:

1. ``r`` key in the keyboard teleop nodes: ``from rover_reset import
   Respawn, spawn_pose`` then ``Respawn(node).respawn(*spawn_pose())``.
2. Standalone listener (this file's ``main``): subscribes
   ``/reset_rover`` (std_msgs/Empty) and respawns on every message, and
   also watches the launch terminal for an ``r`` keypress, so any mode -
   including autonomous, where no keyboard node runs - can reset without
   relaunching::

     ros2 topic pub --once /reset_rover std_msgs/msg/Empty "{}"
     # or just press r in the launch terminal

Mechanically the respawn uses Gazebo's ``/delete_entity`` +
``/spawn_entity`` services with the same model file the launch uses
(``rover.sdf`` when present, else ``rover.urdf`` - the SDF keeps mrsim's
rear camera, which live URDF parsing drops). Initial pose comes from the
``MRSIM_SPAWN`` env var (``"x,y,yaw"``, set by farm.launch.py).
"""
import math
import os
import pathlib
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Empty
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

ROVER_ENTITY = "rover"
RESET_TOPIC = "/reset_rover"


def _model_xml() -> str:
    """Rover model XML the same way the launches pick it (SDF first)."""
    here = pathlib.Path(__file__).resolve()
    src_urdf = here.parent.parent / "urdf"
    cands = [src_urdf / "rover.sdf", src_urdf / "rover.urdf"]
    own = here.parent.parent.name
    for prefix in (here.parents[3], here.parents[4], here.parents[5]):
        for sub in (own, "mrsim", "fftsim", "exgsim"):
            for ext in ("rover.sdf", "rover.urdf"):
                cands.append(prefix / "share" / sub / "urdf" / ext)
    for p in cands:
        try:
            if p.is_file():
                return p.read_text()
        except Exception:
            continue
    raise RuntimeError("cannot find rover.sdf/rover.urdf for respawn")


def spawn_pose() -> tuple:
    """Initial spawn pose (x, y, yaw) from MRSIM_SPAWN (else lane default)."""
    raw = os.environ.get("MRSIM_SPAWN", "-8.0,0.0,0.0")
    try:
        x, y, yaw = (float(v) for v in raw.split(","))
        return x, y, yaw
    except ValueError:
        print(f"[respawn] bad MRSIM_SPAWN='{raw}', using -8,0,0", flush=True)
        return -8.0, 0.0, 0.0


class Respawn:
    """Service clients + delete/spawn helpers for rover reset."""

    def __init__(self, node: Node):
        self.node = node
        self.xml = _model_xml()
        self._spawn = node.create_client(SpawnEntity, "/spawn_entity")
        self._delete = node.create_client(DeleteEntity, "/delete_entity")
        node.get_logger().info(
            "respawn: waiting for /spawn_entity and /delete_entity services"
        )
        for c in (self._spawn, self._delete):
            while not c.service_is_ready():
                rclpy.spin_once(node, timeout_sec=0.2)
        node.get_logger().info("respawn: services ready")

    def delete_rover(self) -> bool:
        self._last_delete_msg = ""
        try:
            fut = self._delete.call_async(
                DeleteEntity.Request(name=ROVER_ENTITY))
            # service calls run outside subscription callbacks, so spinning
            # right here until the response arrives (or timeout) is safe
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=10)
            if not fut.done():
                self.node.get_logger().error("respawn: delete_entity timed out")
                return False
            resp = fut.result()
            self._last_delete_msg = str(resp.status_message)
            if resp.success:
                self.node.get_logger().info("respawn: deleted rover")
            else:
                self.node.get_logger().warn(
                    f"respawn: delete_entity failed: {resp.status_message}"
                )
            return resp.success
        except Exception as e:
            self.node.get_logger().error(f"respawn: delete_entity exception: {e}")
            return False

    def spawn_rover(self, x: float, y: float, yaw: float) -> bool:
        quat = Quaternion()
        s, c_ = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        quat.x, quat.y, quat.z, quat.w = 0.0, 0.0, s, c_
        req = SpawnEntity.Request()
        req.name = ROVER_ENTITY
        req.xml = self.xml
        req.robot_namespace = ""
        req.initial_pose = Pose(
            position=Point(x=x, y=y, z=0.0), orientation=quat
        )
        req.reference_frame = "world"
        try:
            fut = self._spawn.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=10)
            if not fut.done():
                self.node.get_logger().error("respawn: spawn_entity timed out")
                return False
            resp = fut.result()
            if resp.success:
                self.node.get_logger().info(
                    f"respawn: rover back at ({x:.2f},{y:.2f}) "
                    f"yaw={math.degrees(yaw):.1f}"
                )
            else:
                self.node.get_logger().warn(
                    f"respawn: spawn_entity failed: {resp.status_message}"
                )
            return resp.success
        except Exception as e:
            self.node.get_logger().error(f"respawn: spawn_entity exception: {e}")
            return False

    def respawn(self, x: float, y: float, yaw: float) -> bool:
        # order matters: delete first to avoid name collisions with the
        # existing model; then spawn at the initial pose. Gazebo will
        # re-advertise /odom shortly after.
        if not self.delete_rover():
            msg = getattr(self, "_last_delete_msg", "")
            if "not exist" in msg.lower():
                # rover is already gone (e.g. a previous attempt deleted it
                # but died before respawning): still (re)spawn it
                self.node.get_logger().warn(
                    "respawn: rover absent; spawning it at the start pose"
                )
            else:
                # spawn would refuse (entity still exists): stop instead of
                # leaving a half-done state.
                return False
        if not self.spawn_rover(x, y, yaw):
            return False
        # be tidy: no residual velocity after a respawn. Stop both the raw
        # topic (nav/teleop republish over it) and the guarded one, so the
        # rover lands stopped even with the ToF guard in the loop.
        try:
            zero = Twist()
            raw_topic = os.environ.get("MRSIM_CMD_TOPIC", "/cmd_vel")
            self.node.create_publisher(Twist, raw_topic, 10).publish(zero)
            self.node.create_publisher(Twist, "/cmd_vel", 10).publish(zero)
        except Exception:
            pass
        return True


class ResetListener(Node):
    """/reset_rover (Empty) listener: respawn on every message.

    Also watches the launch terminal for an ``r`` keypress (same /dev/tty
    trick as the keyboard teleop: ros2 launch does not forward stdin), so
    reset works by pressing ``r`` right where the sim was launched - in
    every mode, including autonomous where no teleop node runs. The one
    exception is fftsim teleop mode, where the keyboard teleop already owns
    the terminal keys (it has its own ``r``); there the listener stays
    topic-only to avoid double respawns.
    """

    # settle time after a respawn: lets delete/spawn propagate through
    # Gazebo (plugin re-init, topic re-advertise) before another reset is
    # accepted. Overlapping delete/spawn cycles corrupt sensor state.
    SETTLE_S = 3.0

    def __init__(self):
        super().__init__("rover_reset")
        self.respawn = Respawn(self)
        self._pending = False
        self._busy_until = 0.0
        self.create_subscription(Empty, RESET_TOPIC, self._on_reset, 5)
        self._key_fd = None
        self._key_old = None
        pkg = pathlib.Path(__file__).resolve().parent.parent.name
        teleop_owns_keys = (pkg == "fftsim"
                            and os.environ.get("MRSIM_SIM_MODE", "auto")
                            == "teleop")
        if not teleop_owns_keys:
            self._grab_keys()
        if self._key_fd is not None:
            hint = "press 'r' here to reset, or publish " + RESET_TOPIC
        else:
            hint = (f"no terminal for keys; reset via `ros2 topic pub --once "
                    f"{RESET_TOPIC} std_msgs/msg/Empty \"{{}}\"`")
        self.get_logger().info(f"reset listener on {RESET_TOPIC} ({hint})")

    # --------------------------------------------------------------
    # launch-terminal 'r' key (inactive when another keyboard node owns it)
    def _grab_keys(self):
        """Open /dev/tty non-blocking, raw input / cooked output (Ctrl-C
        keeps working, launch output unaffected). None on failure."""
        import termios
        try:
            src = open("/dev/tty", "rb", buffering=0)
        except OSError:
            return
        fd = src.fileno()
        try:
            old = termios.tcgetattr(fd)
            a = termios.tcgetattr(fd)
            a[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK
                      | termios.ISTRIP | termios.IXON)
            a[1] |= (termios.OPOST | termios.ONLCR)
            a[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN)
            a[3] |= termios.ISIG  # keep Ctrl-C live: SIGINT reaches launch
            a[6][termios.VMIN] = 1
            a[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSADRAIN, a)
        except termios.error:
            try:
                src.close()
            except Exception:
                pass
            return
        self._key_fd = fd
        self._key_src = src
        self._key_old = old

    def restore_keys(self):
        import termios
        if self._key_fd is not None:
            try:
                termios.tcsetattr(self._key_fd, termios.TCSADRAIN,
                                  self._key_old)
            except Exception:
                pass
            try:
                self._key_src.close()
            except Exception:
                pass
            self._key_fd = None

    def poll_keys(self) -> None:
        """Non-blocking: any 'r'/'R' pending on the launch terminal flags a
        reset; all other keys are ignored (never consumed as driving)."""
        if self._key_fd is None:
            return
        import select
        try:
            while select.select([self._key_fd], [], [], 0)[0]:
                try:
                    chunk = os.read(self._key_fd, 8)
                except OSError:
                    break
                if not chunk:
                    break
                if b"r" in chunk or b"R" in chunk:
                    self._pending = True
        except Exception:
            pass

    def _on_reset(self, _msg: Empty):
        # flag only: the service calls run in the main loop below, never
        # nested inside a callback (nested spinning can deadlock).
        self._pending = True

    def service_pending(self) -> bool:
        if self._pending:
            self._pending = False
            now = time.monotonic()
            if now < self._busy_until:
                self.get_logger().warn(
                    "reset ignored: previous respawn still settling")
                return False
            x, y, yaw = spawn_pose()
            self.get_logger().info(
                f"reset requested -> start ({x:.2f},{y:.2f}) "
                f"yaw={math.degrees(yaw):.1f}")
            ok = self.respawn.respawn(x, y, yaw)
            self._busy_until = time.monotonic() + self.SETTLE_S
            if not ok:
                self.get_logger().warn("reset failed; try again")
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = ResetListener()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            node.poll_keys()
            node.service_pending()
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_keys()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
