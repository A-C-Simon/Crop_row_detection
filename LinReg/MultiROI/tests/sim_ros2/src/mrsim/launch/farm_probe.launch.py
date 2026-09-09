"""Probe launch: farm + rover, capture camera frames through the MultiROI
detector (no control). Exits when the probe captured the requested frames.

    export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI
    ros2 launch mrsim farm_probe.launch.py out_dir:=/tmp/p frames:=5
    ros2 launch mrsim farm_probe.launch.py field:=zigzag5 out_dir:=/tmp/p

Note: the probe is launched as `python3 nav_node.py` with env vars instead
of a console script, because colcon (modern setuptools) installs console
scripts under bin/ where launch_ros cannot resolve them.
"""
import os
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

PACKAGE = "mrsim"

FIELDS = {
    "circle": "farm_maize",
    "curve": "farm_curve",
    "straight": "farm_straight",
    "curve5": "farm_curve5",
    "straight5": "farm_straight5",
    "zigzag5": "farm_zigzag5",
}


def _resolve_multiroi():
    cands = []
    env = os.environ.get("MULTIROI_DIR")
    if env:
        cands.append(Path(env))
    try:  # installed layout: <MultiROI>/tests/sim_ros2/install/.../share/mrsim
        cands.append(Path(get_package_share_directory(PACKAGE)).parents[5])
    except Exception:
        pass
    cands.append(Path(__file__).resolve().parents[5])  # source layout
    for c in cands:
        try:
            if (c / "run_mr_navigation.py").exists():
                if env and Path(env) != c:
                    print(f"[mrsim] ignoring bad MULTIROI_DIR='{env}', using '{c}'",
                          file=sys.stderr)
                return str(c)
        except Exception:
            continue
    raise RuntimeError(
        "MULTIROI_DIR must point to LinReg/MultiROI (where run_mr_navigation.py lives); "
        f"got '{env}'")


_MULTIROI = _resolve_multiroi()
_AGRIBOT_MODELS = str(Path(_MULTIROI) / "tests" / "agribot" / "agribot_gazebo" / "models")
_NAV_PY = Path(_MULTIROI) / "tests" / "sim_ros2" / "src" / "mrsim" / "mrsim" / "nav_node.py"


def _field_sidecar(pkg_share, field, world):
    """World file + spawn defaults for a field preset (same precedence as
    farm.launch.py: explicit world file's sibling sidecar, else the field
    preset sidecar, else hardcoded fallback)."""
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
    defaults = {"robot_x": "-8.0", "robot_y": "0.0", "robot_yaw": "0.0",
                "furrows": "[0.0]"}
    for sc in reversed(chain):
        for k in ("robot_x", "robot_y", "robot_yaw"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
        if sc.get("furrows") is not None:
            try:
                defaults["furrows"] = json.dumps(
                    [float(c) for c in sc["furrows"]])
            except Exception:
                pass
    return world_file, defaults


def _spawn_row_y(sidecar, spawn_row):
    """Y of the Nth furrow from the left (1-based), or raises."""
    try:
        n = int(spawn_row)
    except (ValueError, TypeError):
        raise RuntimeError(f"spawn_row must be an integer, got '{spawn_row}'")
    try:
        furrows = [float(c) for c in json.loads(sidecar.get("furrows", "[0.0]"))]
    except Exception:
        furrows = [0.0]
    if not (1 <= n <= len(furrows)):
        raise RuntimeError(
            f"spawn_row {n} out of range for {len(furrows)} furrow(s) "
            f"at {sorted(furrows)}")
    return f"{furrows[n - 1]:.3f}"


def _setup(context):
    cfg = context.launch_configurations
    pkg_share = get_package_share_directory(PACKAGE)
    gazebo_share = get_package_share_directory("gazebo_ros")
    world_file, sidecar = _field_sidecar(
        pkg_share, cfg.get("field", "circle"), cfg.get("world", ""))

    def val(name, fallback):
        v = cfg.get(name, "")
        return v if v != "" else sidecar.get(name, fallback)

    robot_y = val("robot_y", "0.0")
    if cfg.get("spawn_row", "") != "" and cfg.get("robot_y", "") == "":
        robot_y = _spawn_row_y(sidecar, cfg.get("spawn_row", ""))

    probe = ExecuteProcess(
        cmd=[sys.executable, str(_NAV_PY)],
        output="screen",
        additional_env={
            "MRSIM_MODE": "probe",
            "MRSIM_OUT_DIR": cfg.get("out_dir", "/tmp"),
            "MRSIM_FRAMES": cfg.get("frames", "5"),
        })

    return [
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH",
                                  f"{_AGRIBOT_MODELS}:/usr/share/gazebo-11/models"),
        AppendEnvironmentVariable("MULTIROI_DIR", _MULTIROI),

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
                        "-y", robot_y,
                        "-z", "0.0", "-Y", val("robot_yaw", "0.0")]),
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
        DeclareLaunchArgument("spawn_row", default_value="",
                              description="spawn in the Nth furrow from the left "
                                          "(1-based; overrides robot_y unless set)"),
        DeclareLaunchArgument("out_dir", default_value="/tmp"),
        DeclareLaunchArgument("frames", default_value="5"),
        OpaqueFunction(function=_setup),
    ])
