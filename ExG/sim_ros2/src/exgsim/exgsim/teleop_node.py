#!/usr/bin/env python3
"""Keyboard teleop for the Gazebo rover (exgsim test rig).

Publishes /cmd_vel from keyboard input. Run it in the terminal that has
focus, then press keys to drive:

    w / up-arrow     forward        s / down-arrow  back
    a / left-arrow   turn left      d / right-arrow turn right
    q / e            strafe-left/right  (diff-drive: same as a/d)
    space            stop            x / ctrl-c      quit

Keys ramp speed smoothly (up to v_max / omega_max). Auto-repeat keeps the
rover moving while a key is held; release = stop.

Modes:
  normal : reads keys from the terminal (raw tty) - launch via farm.launch.py
           with mode:=teleop.
  demo   : MRSIM_DEMO_KEYS="w w a" (space-separated, each held 1 s) plays a
           scripted key sequence - used for headless tests.

Launch:
  ros2 launch exgsim farm.launch.py mode:=teleop gui:=true
"""
import math
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

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
        super().__init__("exg_teleop")
        topic = os.environ.get("MRSIM_CMD_TOPIC", "/cmd_vel")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.lin = 0.0
        self.ang = 0.0
        self.last_t = time.time()
        self.get_logger().info(
            f"teleop ready on '{topic}' | keys: w/s fwd, a/d turn, space stop, x quit "
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

    # interactive keyboard loop (raw tty)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        held = set()
        last = time.time()
        while True:
            # read any pending keys
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":  # escape sequence (arrows)
                    more = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.05)[0] else ""
                    ch += more
                if ch in ("x", "X", "\x03"):  # x / ctrl-c quit
                    node.get_logger().info("teleop quit")
                    raise KeyboardInterrupt
                if ch == " ":
                    held.clear()
                elif ch in KEYMAP:
                    held.add(ch)
            now = time.time()
            node.step(held, min(0.2, now - last))
            last = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.publish(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
