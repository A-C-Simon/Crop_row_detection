"""Launch the exgsim closed-loop test: maize farm + rover + ExG C++ stack.

    export EXG_DIR=/abs/path/to/ExG   # (only used for messages; the C++
    nodes come from its colcon install sourced by run_sim.sh)
    ros2 launch exgsim farm.launch.py log_dir:=/tmp/exgsim_log
    ros2 launch exgsim farm.launch.py field:=zigzag5 log_dir:=/tmp/z

Starts gzserver headless on the selected field, spawns the rover in the
configured furrow, bridges Gazebo topics to the names the C++ agribot_vs
node expects (/front/rgb/image_raw, /odometry/raw), runs that node (HSV
detect + servo -> /cmd_vel) plus a monitor (overlay + CSV, same schema as
the other rigs). The launch shuts down when the monitor exits (lane end /
laps / max time).

Fields (world + spawn/lane defaults travel together; explicit robot_*/
lane/circle args always win):
    circle    2-row ring R12 (default)      curve     2-row S-bend
    straight  2-row straight                curve5    5-row S-bend
    straight5 5-row straight                zigzag5   5-row gentle zigzag
Custom world file: world:=/path/to.world (sibling .spawn.json used when
present, else the field sidecar).

Modes (MRSIM_SIM_MODE env, set by run scripts):
  auto   - C++ stack drives (mask_tune false)
  teleop - C++ stack idles (mask_tune true, no cmd), keyboard teleop owns
           /cmd_vel (run exg_teleop by hand in the GUI terminal)
  demo   - scripted teleop keys (headless check), C++ stack idles
"""
import os
import re
import sys
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            Shutdown, AppendEnvironmentVariable,
                            SetEnvironmentVariable,
                            RegisterEventHandler, ExecuteProcess,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PACKAGE = "exgsim"
EXG_PKG = "visual_crop_row_navigation_ros2"

FIELDS = {
    "circle": "farm_maize",
    "curve": "farm_curve",
    "straight": "farm_straight",
    "curve5": "farm_curve5",
    "straight5": "farm_straight5",
    "zigzag5": "farm_zigzag5",
}

_EXG_WS = os.environ.get("EXG_WS_DIR", "")
_EXG_PKG_SHARE = ""
if _EXG_WS:
    _EXG_PKG_SHARE = str(Path(_EXG_WS) / "install"
                         / EXG_PKG / "share" / EXG_PKG)
def _exgsim_src():
    """Source tree holding the python nodes (bridge/monitor/teleop)."""
    env = os.environ.get("EXG_DIR", "")
    if env:
        c = Path(env) / "sim_ros2" / "src" / "exgsim"
        if (c / "exgsim" / "bridge.py").exists():
            return c
    for a in Path(__file__).resolve().parents:
        c = a / "sim_ros2" / "src" / "exgsim"
        if (c / "exgsim" / "bridge.py").exists():
            return c
    raise RuntimeError("exgsim nodes not found; export EXG_DIR=/abs/path/to/ExG")


_EXGSIM_SRC = _exgsim_src()
_BRIDGE_PY = _EXGSIM_SRC / "exgsim" / "bridge.py"
_MONITOR_PY = _EXGSIM_SRC / "exgsim" / "monitor.py"
_TELEOP_PY = _EXGSIM_SRC / "exgsim" / "teleop_node.py"
for _p in (_BRIDGE_PY, _MONITOR_PY, _TELEOP_PY):
    if not _p.exists():
        raise RuntimeError(f"exgsim nodes not found under {_EXGSIM_SRC}")


def _read_sidecar(path):
    try:
        p = Path(path)
        if p.is_file():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}


def _load_sidecar(pkg_share, field, world):
    """Spawn/lane defaults for a field.

    Precedence per key: explicit launch arg > sibling sidecar of an
    explicitly passed world file > the field preset sidecar > hardcoded
    fallback. Returns (world_file, defaults dict).
    """
    worlds = Path(pkg_share) / "worlds"
    stem = FIELDS.get(field, "farm_maize")
    world_file = world or str(worlds / (stem + ".world"))
    chain = []
    if world:
        sib = str(Path(world).with_suffix("")) + ".spawn.json"
        chain.append(_read_sidecar(sib))
    chain.append(_read_sidecar(worlds / (stem + ".spawn.json")))
    defaults = {"robot_x": "-8.0", "robot_y": "0.0", "robot_yaw": "0.0",
                "lane_y": "0.0", "lane_end_x": "9.0",
                "circle_cx": "0.0", "circle_cy": "0.0", "circle_r": "0.0",
                "max_laps_default": "1"}
    for sc in reversed(chain):
        for k in ("robot_x", "robot_y", "robot_yaw", "lane_y",
                  "lane_end_x", "circle_cx", "circle_cy", "circle_r"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
        if sc.get("max_laps_default") is not None:
            defaults["max_laps_default"] = str(sc["max_laps_default"])
    return world_file, defaults


def _display_alive():
    """True if $DISPLAY points at a reachable local X server.

    The C++ node calls imshow() unconditionally; a stale DISPLAY (set but
    dead) stalls every loop iteration instead of failing fast.
    """
    d = os.environ.get("DISPLAY", "")
    m = re.match(r"^(.*):(\d+)(?:\.\d+)?$", d)
    if not m:
        return False
    host, num = m.group(1), m.group(2)
    if host in ("", "localhost", "unix"):
        return Path(f"/tmp/.X11-unix/X{num}").exists()
    return True


_NEED_XVFB = not _display_alive()


def _setup(context):
    cfg = context.launch_configurations
    pkg_share = get_package_share_directory(PACKAGE)
    gazebo_share = get_package_share_directory("gazebo_ros")
    field = cfg.get("field", "circle")
    world_file, sidecar = _load_sidecar(
        pkg_share, field, cfg.get("world", ""))

    def val(name, sidecar_key=None, fallback=""):
        v = cfg.get(name, "")
        if v == "" and sidecar_key and sidecar.get(sidecar_key) is not None:
            return str(sidecar[sidecar_key])
        return v if v != "" else fallback

    robot_x = val("robot_x", "robot_x", "-8.0")
    robot_y = val("robot_y", "robot_y", "0.0")
    robot_yaw = val("robot_yaw", "robot_yaw", "0.0")
    lane_y = val("lane_y", "lane_y", "0.0")
    lane_end_x = val("lane_end_x", "lane_end_x", "9.0")
    circle_cx = val("circle_cx", "circle_cx", "0.0")
    circle_cy = val("circle_cy", "circle_cy", "0.0")
    circle_r = val("circle_r", "circle_r", "0.0")
    max_laps = val("max_laps", "max_laps_default", "1")

    # sim-tuned params ship with THIS package (vendor file keeps upstream values)
    param_file = str(Path(pkg_share) / "params" / "exgsim_run.yaml")

    sim_mode = os.environ.get("MRSIM_SIM_MODE", "auto")
    idle = sim_mode != "auto"  # teleop/demo: C++ stack must not drive

    monitor = ExecuteProcess(
        cmd=[sys.executable, str(_MONITOR_PY)],
        output="screen",
        additional_env={
            "MRSIM_LANE_Y": lane_y,
            "MRSIM_LANE_END_X": lane_end_x,
            "MRSIM_CIRCLE_CX": circle_cx,
            "MRSIM_CIRCLE_CY": circle_cy,
            "MRSIM_CIRCLE_R": circle_r,
            "MRSIM_CIRCLE_LAPS": max_laps,
            "MRSIM_LOG_DIR": cfg.get("log_dir", "/tmp/exgsim_log"),
            "MRSIM_MAX_SECONDS": cfg.get("max_seconds", "0"),
        })

    bridge = ExecuteProcess(
        cmd=[sys.executable, str(_BRIDGE_PY)],
        output="screen")

    vs_node = Node(
        package=EXG_PKG, executable="agribot_vs_node",
        output="screen",
        # imshow() needs a live X server; virtual one unless the real
        # display is reachable
        prefix=["xvfb-run", "-a"] if _NEED_XVFB else [],
        parameters=[param_file,
                    {"mask_tune": idle,
                     "publish_cmd_vel": (not idle)}],
    )

    demo = ExecuteProcess(
        cmd=[sys.executable, str(_TELEOP_PY)],
        output="screen",
        condition=IfCondition("1" if sim_mode == "demo" else "0"),
        additional_env={"MRSIM_DEMO_KEYS": cfg.get("demo_keys", "w w w a a s s")})

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
            launch_arguments={
                "world": world_file,
                "gui": cfg.get("gui", "false"),
                "verbose": "false",
            }.items(),
        ),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen",
             parameters=[{
                 "robot_description":
                     (Path(pkg_share) / "urdf" / "rover.urdf").read_text()}]),
        Node(package="gazebo_ros", executable="spawn_entity.py",
             output="screen",
             arguments=[
                 "-topic", "robot_description",
                 "-entity", "rover",
                 "-x", robot_x,
                 "-y", robot_y,
                 "-z", "0.0",
                 "-Y", robot_yaw,
             ]),

        bridge,
        vs_node,
        monitor,
        demo,
        RegisterEventHandler(OnProcessExit(target_action=monitor,
                                           on_exit=[Shutdown(reason="monitor done")])),
        RegisterEventHandler(OnProcessExit(target_action=demo,
                                           on_exit=[Shutdown(reason="demo done")])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "field", default_value="circle", choices=list(FIELDS) + ["custom"],
            description="field preset (world + spawn/lane defaults travel "
                        "together); use field:=custom with world:= for other files"),
        DeclareLaunchArgument("world", default_value="",
                              description="world file override (empty = the "
                                          "field preset world; a sibling "
                                          ".spawn.json seeds defaults)"),
        DeclareLaunchArgument("mode", default_value="auto",
                              choices=["auto", "teleop", "demo"],
                              description="auto=ExG drives; "
                                          "teleop=keyboard node drives "
                                          "(C++ stack idles, GUI view); "
                                          "demo=scripted keys (headless test)"),
        DeclareLaunchArgument("demo_keys", default_value="w w w a a s s",
                              description="scripted key sequence for mode:=demo"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("robot_x", default_value="",
                              description="spawn x (empty = field default)"),
        DeclareLaunchArgument("robot_y", default_value="",
                              description="spawn y (empty = field default)"),
        DeclareLaunchArgument("robot_yaw", default_value="",
                              description="spawn yaw (empty = field default)"),
        DeclareLaunchArgument("lane_y", default_value="",
                              description="furrow center y (empty = field default)"),
        DeclareLaunchArgument("lane_end_x", default_value="9.0"),
        DeclareLaunchArgument("circle_cx", default_value="",
                              description="ring-field center x (empty = field default)"),
        DeclareLaunchArgument("circle_cy", default_value="",
                              description="ring-field center y (empty = field default)"),
        DeclareLaunchArgument("circle_r", default_value="",
                              description="ring furrow radius; <=0 = straight lane "
                                          "(end at lane_end_x; empty = field default)"),
        DeclareLaunchArgument("max_laps", default_value="",
                              description="ring-field laps before auto-stop; "
                                          "<=0 = loop forever (empty = field default)"),
        DeclareLaunchArgument("max_seconds", default_value="0"),
        DeclareLaunchArgument("log_dir", default_value="/tmp/exgsim_log"),
        OpaqueFunction(function=_setup),
    ])
