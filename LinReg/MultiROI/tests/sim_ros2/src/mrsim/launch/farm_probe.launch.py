"""Probe launch: farm + rover, capture camera frames through the MultiROI
detector (no control). Exits when the probe captured the requested frames.

    export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI
    ros2 launch mrsim farm_probe.launch.py out_dir:=/tmp/p frames:=5

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
                            RegisterEventHandler, ExecuteProcess)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

PACKAGE = "mrsim"


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


def generate_launch_description():
    pkg_share = get_package_share_directory(PACKAGE)
    gazebo_share = get_package_share_directory("gazebo_ros")
    # spawn defaults follow the committed world (see farm.launch sidecar)
    sidecar = {"robot_x": "-8.0", "robot_y": "0.0", "robot_yaw": "0.0"}
    try:
        sc = Path(pkg_share) / "worlds" / "farm_maize.spawn.json"
        if sc.is_file():
            raw = json.loads(sc.read_text())
            for k in sidecar:
                if raw.get(k) is not None:
                    sidecar[k] = str(raw[k])
    except Exception:
        pass

    probe = ExecuteProcess(
        cmd=[sys.executable, str(_NAV_PY)],
        output="screen",
        additional_env={
            "MRSIM_MODE": "probe",
            "MRSIM_OUT_DIR": LaunchConfiguration("out_dir"),
            "MRSIM_FRAMES": LaunchConfiguration("frames"),
        })

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=PathJoinSubstitution(
            [pkg_share, "worlds", "farm_maize.world"])),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("robot_x", default_value=sidecar["robot_x"]),
        DeclareLaunchArgument("robot_y", default_value=sidecar["robot_y"]),
        DeclareLaunchArgument("robot_yaw", default_value=sidecar["robot_yaw"]),
        DeclareLaunchArgument("out_dir", default_value="/tmp"),
        DeclareLaunchArgument("frames", default_value="5"),

        AppendEnvironmentVariable("GAZEBO_MODEL_PATH",
                                  f"{_AGRIBOT_MODELS}:/usr/share/gazebo-11/models"),
        AppendEnvironmentVariable("MULTIROI_DIR", _MULTIROI),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
            launch_arguments={
                "world": LaunchConfiguration("world"),
                "gui": LaunchConfiguration("gui"),
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
                        "-x", LaunchConfiguration("robot_x"),
                        "-y", LaunchConfiguration("robot_y"),
                        "-z", "0.0", "-Y", LaunchConfiguration("robot_yaw")]),
        probe,
        # when the probe exits -> tear the whole sim down
        RegisterEventHandler(OnProcessExit(target_action=probe,
                                           on_exit=[Shutdown(reason="probe done")])),
    ])
