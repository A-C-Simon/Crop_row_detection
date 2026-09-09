"""Probe launch: farm + rover + ExG stack with wheels disabled.

The C++ detector cannot run offline, so the probe runs the real stack with
`publish_cmd_vel:=false`: detection runs, /vs_msg flows, the rover never
moves. The monitor captures MRSIM_FRAMES raw frames plus per-frame vs
errors, then everything shuts down.

    export EXG_DIR=/abs/path/to/ExG
    ros2 launch exgsim farm_probe.launch.py out_dir:=/tmp/p frames:=5
    ros2 launch exgsim farm_probe.launch.py field:=zigzag5 out_dir:=/tmp/p
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
                            RegisterEventHandler, ExecuteProcess,
                            OpaqueFunction)
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

def _exgsim_src():
    """Source tree holding the python nodes (bridge/monitor)."""
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
for _p in (_BRIDGE_PY, _MONITOR_PY):
    if not _p.exists():
        raise RuntimeError(f"exgsim nodes not found under {_EXGSIM_SRC}")


def _display_alive():
    """True if $DISPLAY points at a reachable local X server (see
    farm.launch.py: the C++ node imshows unconditionally)."""
    d = os.environ.get("DISPLAY", "")
    m = re.match(r"^(.*):(\d+)(?:\.\d+)?$", d)
    if not m:
        return False
    host, num = m.group(1), m.group(2)
    if host in ("", "localhost", "unix"):
        return Path(f"/tmp/.X11-unix/X{num}").exists()
    return True


_NEED_XVFB = not _display_alive()


def _field_sidecar(pkg_share, field, world):
    """World file + spawn defaults for a field preset (same precedence as
    farm.launch.py)."""
    worlds = Path(pkg_share) / "worlds"
    stem = FIELDS.get(field, "farm_maize")
    world_file = world or str(worlds / (stem + ".world"))
    chain = []
    if world:
        sib = str(Path(world).with_suffix("")) + ".spawn.json"
        try:
            p = Path(sib)
            if p.is_file():
                chain.append(json.loads(p.read_text()))
        except Exception:
            pass
    try:
        p = worlds / (stem + ".spawn.json")
        if p.is_file():
            chain.append(json.loads(p.read_text()))
    except Exception:
        pass
    defaults = {"robot_x": "-8.0", "robot_y": "0.0", "robot_yaw": "0.0"}
    for sc in reversed(chain):
        for k in defaults:
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
    return world_file, defaults


def _setup(context):
    cfg = context.launch_configurations
    pkg_share = get_package_share_directory(PACKAGE)
    gazebo_share = get_package_share_directory("gazebo_ros")
    # sim-tuned params ship with THIS package (vendor file keeps upstream values)
    param_file = str(Path(pkg_share) / "params" / "exgsim_run.yaml")
    world_file, sidecar = _field_sidecar(
        pkg_share, cfg.get("field", "circle"), cfg.get("world", ""))

    def val(name, fallback):
        v = cfg.get(name, "")
        return v if v != "" else sidecar.get(name, fallback)

    probe = ExecuteProcess(
        cmd=[sys.executable, str(_MONITOR_PY)],
        output="screen",
        additional_env={
            "MRSIM_MODE": "probe",
            "MRSIM_OUT_DIR": cfg.get("out_dir", "/tmp"),
            "MRSIM_FRAMES": cfg.get("frames", "5"),
        })

    bridge = ExecuteProcess(
        cmd=[sys.executable, str(_BRIDGE_PY)],
        output="screen")

    vs_node = Node(
        package=EXG_PKG, executable="agribot_vs_node",
        output="screen",
        prefix=["xvfb-run", "-a"] if _NEED_XVFB else [],
        parameters=[param_file,
                    {"mask_tune": False,
                     "publish_cmd_vel": False}],
    )

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
             parameters=[{"robot_description":
                         (Path(pkg_share) / "urdf" / "rover.urdf").read_text()}]),
        Node(package="gazebo_ros", executable="spawn_entity.py",
             output="screen",
             arguments=["-topic", "robot_description", "-entity", "rover",
                        "-x", val("robot_x", "-8.0"),
                        "-y", val("robot_y", "0.0"),
                        "-z", "0.0", "-Y", val("robot_yaw", "0.0")]),
        bridge,
        vs_node,
        probe,
        # when the probe exits -> tear the whole sim down
        RegisterEventHandler(OnProcessExit(target_action=probe,
                                           on_exit=[Shutdown(reason="probe done")])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "field", default_value="circle", choices=list(FIELDS) + ["custom"],
            description="field preset (world + spawn defaults travel together)"),
        DeclareLaunchArgument("world", default_value="",
                              description="world file override (empty = the "
                                          "field preset world)"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("robot_x", default_value="",
                              description="spawn x (empty = field default)"),
        DeclareLaunchArgument("robot_y", default_value="",
                              description="spawn y (empty = field default)"),
        DeclareLaunchArgument("robot_yaw", default_value="",
                              description="spawn yaw (empty = field default)"),
        DeclareLaunchArgument("out_dir", default_value="/tmp"),
        DeclareLaunchArgument("frames", default_value="5"),
        OpaqueFunction(function=_setup),
    ])
