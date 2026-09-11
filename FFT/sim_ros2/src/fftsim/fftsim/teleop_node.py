#!/usr/bin/env python3
"""Keyboard teleop for the Gazebo rover (fftsim test rig).

Publishes /cmd_vel from keyboard input. Run it in the terminal that has
focus, then press keys to drive:

    w / up-arrow     forward        s / down-arrow  back
    a / left-arrow   turn left      d / right-arrow turn right
    q / e            strafe-left/right  (diff-drive: same as a/d)
    space            stop            x / ctrl-c      quit
    r                respawn the rover at the initial spawn pose
                     (MRSIM_SPAWN "x,y,yaw"; no relaunch needed)

Keys ramp speed smoothly (up to v_max / omega_max). Auto-repeat keeps the
rover moving while a key is held; release = stop.

Modes:
  normal : reads keys from the terminal - prefers stdin when it is a tty
           (standalone run), else the controlling terminal /dev/tty, so
           keys also work as a `ros2 launch` child (which does not forward
           stdin). Launch via farm.launch.py with mode:=teleop.
  demo   : MRSIM_DEMO_KEYS="w w a" (space-separated, each held 1 s) plays a
           scripted key sequence - used for headless tests.

Launch:
  ros2 launch fftsim farm.launch.py mode:=teleop gui:=true
"""
import math
import os
import select
import sys
import termios
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# rover_reset lives next to this file. Direct script runs
# (python3 .../teleop_node.py) import it as a sibling; `ros2 run`
# console scripts import it through the installed package instead.
try:
    from rover_reset import Respawn, spawn_pose
except ImportError:  # pragma: no cover - ros2 run package context
    from fftsim.rover_reset import Respawn, spawn_pose

KEYMAP = {
    "w": ("fwd", 1.0), "W": ("fwd", 1.0),
    "s": ("fwd", -1.0), "S": ("fwd", -1.0),
    "a": ("turn", 1.0), "A": ("turn", 1.0),
    "d": ("turn", -1.0), "D": ("turn", -1.0),
    "q": ("turn", 1.0), "Q": ("turn", 1.0),      # q/e alias a/d
    "e": ("turn", -1.0), "E": ("turn", -1.0),
    "\x1b[A": ("fwd", 1.0),    # up
    "\x1b[B": ("fwd", -1.0),   # down
    "\x1b[C": ("turn", -1.0),  # right
    "\x1b[D": ("turn", 1.0),   # left
}
MAX_LIN = 0.5    # m/s  (diff-drive limit in rover.urdf)
MAX_ANG = 1.5    # rad/s
RAMP = 0.35      # per-second rate limit toward a held key


class TeleopNode(Node):
    def __init__(self):
        super().__init__("fft_teleop")
        topic = os.environ.get("MRSIM_CMD_TOPIC", "/cmd_vel")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.lin = 0.0
        self.ang = 0.0
        self.last_t = time.time()
        self._respawn = None  # lazy: service wait happens on first 'r'
        self.get_logger().info(
            f"teleop ready on '{topic}' | keys: w/s fwd, a/d turn, space stop, "
            f"r respawn at start, x quit "
            f"(v_max={MAX_LIN}, w_max={MAX_ANG})")

    def publish(self, lin: float, ang: float):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.pub.publish(t)

    def ramp_to(self, target_lin: float, target_ang: float, dt: float):
        """Move current speed toward the target at RAMP rate."""
        max_dl = RAMP * MAX_LIN * dt
        max_da = RAMP * MAX_ANG * dt
        self.lin += max(-max_dl, min(max_dl, target_lin - self.lin))
        self.ang += max(-max_da, min(max_da, target_ang - self.ang))

    def step(self, held: set, dt: float):
        """One control tick: ramp toward the held keys, publish."""
        tl = ta = 0.0
        for key in held:
            kind, sign = KEYMAP[key]
            if kind == "fwd":
                tl += sign
            else:
                ta += sign
        # normalize diagonals
        if tl != 0.0 and ta != 0.0:
            tl *= 0.7
            ta *= 0.7
        self.ramp_to(tl * MAX_LIN, ta * MAX_ANG, dt)
        self.publish(self.lin, self.ang)
        return self.lin, self.ang


def _demo_keys(node: TeleopNode):
    """Scripted keys (headless test): 'w w a s' style, each held ~0.8 s."""
    seq = os.environ.get("MRSIM_DEMO_KEYS", "w w a a s").split()
    node.get_logger().info(f"demo mode: playing keys {seq}")
    for key in seq:
        held = {key} if key in KEYMAP else set()
        if not held:
            node.get_logger().warn(f"demo: unknown key '{key}' skipped")
            continue
        t0 = time.time()
        while time.time() - t0 < 0.8:
            node.step(held, 0.05)
            time.sleep(0.05)
    node.step(set(), 0.05)  # release
    node.get_logger().info("demo done")


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    if os.environ.get("MRSIM_DEMO_KEYS") is not None:
        try:
            _demo_keys(node)
        finally:
            node.publish(0.0, 0.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return

    # interactive keyboard loop. Preferred input is stdin when it is a tty
    # (standalone run); otherwise the controlling terminal /dev/tty, which
    # is how keys keep working as a `ros2 launch` child (launch does not
    # forward stdin). Piped stdin without a terminal still works for
    # scripted input; with no terminal at all the node idles.
    src = None
    if sys.stdin.isatty():
        src = sys.stdin
    else:
        try:
            src = open("/dev/tty", "rb", buffering=0)
        except OSError:
            if not sys.stdin.closed:
                src = sys.stdin
    if src is None:
        node.get_logger().warn(
            "teleop: no terminal for keyboard input; keys unavailable")
        try:
            while rclpy.ok():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        node.publish(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    fd = src.fileno()
    old = None
    raw = False
    try:
        # raw INPUT (per-key, unbuffered) but cooked OUTPUT: when this node
        # shares the launch terminal (via /dev/tty), gzserver/launch output
        # keeps its newline translation and Ctrl-C keeps working. ECHO off
        # so held keys do not spam the shared terminal.
        old = termios.tcgetattr(fd)
        a = termios.tcgetattr(fd)
        a[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK
                  | termios.ISTRIP | termios.IXON)
        a[1] |= (termios.OPOST | termios.ONLCR)
        a[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN)
        a[3] |= termios.ISIG  # keep Ctrl-C live: SIGINT reaches the launch
        a[6][termios.VMIN] = 1
        a[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSADRAIN, a)
        raw = True
    except termios.error:
        pass  # piped stdin: no terminal control, keys still parse
    try:
        held = set()
        last = time.time()
        while True:
            # read any pending keys (os.read: no buffering surprises)
            while select.select([fd], [], [], 0)[0]:
                try:
                    ch = os.read(fd, 1).decode("utf-8", "ignore")
                except OSError:
                    break
                if ch == "":  # EOF (closed pipe): idle instead of spinning
                    time.sleep(0.1)
                    break
                if ch == "\x1b":  # escape sequence (arrows)
                    more = os.read(fd, 2).decode("utf-8", "ignore") \
                        if select.select([fd], [], [], 0.05)[0] else ""
                    ch += more
                if ch in ("x", "X", "\x04"):  # x / ctrl-d quit
                    node.get_logger().info("teleop quit")
                    raise KeyboardInterrupt
                if ch == "\x03":  # ctrl-c (only seen without ISIG)
                    raise KeyboardInterrupt
                if ch == " ":
                    held.clear()
                elif ch in KEYMAP:
                    held.add(ch)
                elif ch in ("r", "R"):
                    # 'r' = respawn at the initial spawn pose (discrete action)
                    held.clear()  # land stopped, not still driving
                    try:
                        if node._respawn is None:
                            node._respawn = Respawn(node)
                        x, y, yaw = spawn_pose()
                        node.get_logger().info(
                            f"respawn: returning rover to start "
                            f"({x:.2f},{y:.2f})")
                        if node._respawn.respawn(x, y, yaw):
                            node.lin = 0.0
                            node.ang = 0.0
                            node.publish(0.0, 0.0)
                    except Exception as e:
                        node.get_logger().warn(f"respawn failed: {e}")
                    continue
            now = time.time()
            node.step(held, min(0.2, now - last))
            last = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if raw and old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if src is not sys.stdin:
            try:
                src.close()
            except Exception:
                pass
        node.publish(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
