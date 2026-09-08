"""Probe launch: farm + rover, capture camera frames through the FFT
detector (no control). Exits when the probe captured the requested frames.

    export FFT_DIR=/abs/path/to/FFT
    ros2 launch fftsim farm_probe.launch.py out_dir:=/tmp/p frames:=5

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
                            SetEnvironmentVariable,
                            RegisterEventHandler, ExecuteProcess)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

PACKAGE = "fftsim"


def _resolve_src_dir(env_name, marker, pkg_parents_up):
    cands = []
    env = os.environ.get(env_name)
    if env:
        cands.append(Path(env))
    try:  # installed layout: <root>/sim_ros2/install/.../share/<pkg>
        cands.append(Path(get_package_share_directory(PACKAGE)).parents[pkg_parents_up])
    except Exception:
        pass
    cands.append(Path(__file__).resolve().parents[5])  # source layout
    for c in cands:
        try:
            if (c / marker).exists():
                if env and Path(env) != c:
                    print(f"[fftsim] ignoring bad {env_name}='{env}', using '{c}'",
                          file=sys.stderr)
                return str(c)
        except Exception:
            continue
    raise RuntimeError(
        f"{env_name} must point to the dir containing {marker}; got '{env}'")


_FFT = _resolve_src_dir("FFT_DIR", "dft_crop_row_detector.py", 4)
_MULTIROI = _resolve_src_dir("MULTIROI_DIR", "mr_vs.py", 99)
_AGRIBOT_MODELS = str(Path(_MULTIROI) / "tests" / "agribot" / "agribot_gazebo" / "models")
_NAV_PY = Path(_FFT) / "sim_ros2" / "src" / "fftsim" / "fftsim" / "nav_node.py"


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
        SetEnvironmentVariable("MULTIROI_DIR", _MULTIROI),
        SetEnvironmentVariable("FFT_DIR", _FFT),

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
