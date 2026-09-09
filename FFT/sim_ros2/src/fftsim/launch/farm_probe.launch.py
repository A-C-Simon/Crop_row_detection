"""Probe launch: farm + rover, capture camera frames through the FFT
detector (no control). Exits when the probe captured the requested frames.

    export FFT_DIR=/abs/path/to/FFT
    ros2 launch fftsim farm_probe.launch.py out_dir:=/tmp/p frames:=5
    ros2 launch fftsim farm_probe.launch.py field:=zigzag5 out_dir:=/tmp/p

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
                            RegisterEventHandler, ExecuteProcess,
                            OpaqueFunction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PACKAGE = "fftsim"

FIELDS = {
    "circle": "farm_maize",
    "curve": "farm_curve",
    "straight": "farm_straight",
    "curve5": "farm_curve5",
    "straight5": "farm_straight5",
    "zigzag5": "farm_zigzag5",
}


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
    world_file, sidecar = _field_sidecar(
        pkg_share, cfg.get("field", "circle"), cfg.get("world", ""))

    def val(name, fallback):
        v = cfg.get(name, "")
        return v if v != "" else sidecar.get(name, fallback)

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
        SetEnvironmentVariable("MULTIROI_DIR", _MULTIROI),
        SetEnvironmentVariable("FFT_DIR", _FFT),

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
