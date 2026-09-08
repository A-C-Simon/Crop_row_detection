"""Launch the fftsim closed-loop test: maize farm + rover + FFT nav node.

    export FFT_DIR=/abs/path/to/FFT
    export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI   # shared servo only
    ros2 launch fftsim farm.launch.py log_dir:=/tmp/fftsim_log

Starts gzserver headless on the generated farm world, spawns the rover in
the furrow between two plant rows, and runs `nav_node.py` (camera ->
DFT detector -> shared visual servo -> /cmd_vel). The launch shuts
everything down when the nav node exits (end of lane / max time / Ctrl-C).

Note: the nav node is launched as `python3 nav_node.py` with env vars
instead of a console script, because colcon (modern setuptools) installs
console scripts under bin/ where launch_ros cannot resolve them.
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
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

PACKAGE = "fftsim"

# FFT source dir: /abs/path/to/FFT (contains dft_crop_row_detector.py).
# MultiROI dir is needed only for the shared visual servo (mr_vs).
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
if not (Path(_AGRIBOT_MODELS) / "big_plant").is_dir():
    raise RuntimeError(f"agribot plant models not found at {_AGRIBOT_MODELS}")

# node sources: live in the FFT sim workspace (src tree).
_SIM_SRC = Path(_FFT) / "sim_ros2" / "src" / "fftsim"
_NAV_PY = _SIM_SRC / "fftsim" / "nav_node.py"
_TELEOP_PY = _SIM_SRC / "fftsim" / "teleop_node.py"
if not (_NAV_PY.exists() and _TELEOP_PY.exists()):
    raise RuntimeError(f"fftsim nodes not found under {_SIM_SRC}")


def _load_sidecar(pkg_share):
    """Spawn/lane defaults for the committed world (written by
    gen_farm_world.py as <world-stem>.spawn.json). Custom worlds passed via
    world:= should set robot_x/y/yaw and circle_* explicitly."""
    defaults = {"robot_x": "-8.0", "robot_y": "0.0", "robot_yaw": "0.0",
                "lane_y": "0.0", "lane_end_x": "9.0",
                "circle_cx": "0.0", "circle_cy": "0.0", "circle_r": "0.0",
                "max_laps_default": "1", "trim_default": "0.0"}
    try:
        sc = Path(pkg_share) / "worlds" / "farm_maize.spawn.json"
        if sc.is_file():
            raw = json.loads(sc.read_text())
            for k in ("robot_x", "robot_y", "robot_yaw", "lane_y",
                      "lane_end_x", "circle_cx", "circle_cy", "circle_r"):
                if raw.get(k) is not None:
                    defaults[k] = str(raw[k])
            if raw.get("max_laps_default") is not None:
                defaults["max_laps_default"] = str(raw["max_laps_default"])
            if raw.get("trim_default") is not None:
                defaults["trim_default"] = str(raw["trim_default"])
    except Exception:
        pass
    return defaults


def generate_launch_description():
    pkg_share = get_package_share_directory(PACKAGE)
    gazebo_share = get_package_share_directory("gazebo_ros")
    gazebo_default_models = "/usr/share/gazebo-11/models"
    sidecar = _load_sidecar(pkg_share)

    sim_mode = LaunchConfiguration("mode")
    # idle env: nav node runs detection/overlay but never publishes /cmd_vel
    # unless mode:=auto (so teleop/demo own the topic)
    nav_idle = "1" if os.environ.get("MRSIM_SIM_MODE", "auto") != "auto" else ""
    nav = ExecuteProcess(
        cmd=[sys.executable, str(_NAV_PY)],
        output="screen",
        additional_env={
            "MRSIM_MODE": "nav",
            "MRSIM_NAV_IDLE": nav_idle,
            "MRSIM_LANE_Y": LaunchConfiguration("lane_y"),
            "MRSIM_LANE_END_X": LaunchConfiguration("lane_end_x"),
            "MRSIM_CIRCLE_CX": LaunchConfiguration("circle_cx"),
            "MRSIM_CIRCLE_CY": LaunchConfiguration("circle_cy"),
            "MRSIM_CIRCLE_R": LaunchConfiguration("circle_r"),
            "MRSIM_CIRCLE_LAPS": LaunchConfiguration("max_laps"),
            "MRSIM_LAMBDA_X": LaunchConfiguration("lambda_x"),
            "MRSIM_LAMBDA_THETA": LaunchConfiguration("lambda_theta"),
            "MRSIM_HEADING_GATE": LaunchConfiguration("heading_gate"),
            "MRSIM_FF_GAIN": LaunchConfiguration("ff_gain"),
            "FFT_TRIM_M": LaunchConfiguration("trim"),
            "MRSIM_LOG_DIR": LaunchConfiguration("log_dir"),
            "MRSIM_MAX_SECONDS": LaunchConfiguration("max_seconds"),
        })

    # mode:=demo - scripted keys, headless-verifiable. The demo process exits
    # after its sequence; when it does, the handler below shuts the launch
    # down (nav idles, so it would otherwise run forever).
    demo = ExecuteProcess(
        cmd=[sys.executable, str(_TELEOP_PY)],
        output="screen",
        condition=IfCondition("1" if os.environ.get("MRSIM_SIM_MODE", "auto") == "demo" else "0"),
        additional_env={"MRSIM_DEMO_KEYS": LaunchConfiguration("demo_keys")})

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=PathJoinSubstitution(
            [pkg_share, "worlds", "farm_maize.world"])),
        DeclareLaunchArgument("mode", default_value="auto",
                              choices=["auto", "teleop", "demo"],
                              description="auto=FFT drives; "
                                          "teleop=keyboard node drives "
                                          "(nav node idles, GUI view); "
                                          "demo=scripted keys (headless test)"),
        DeclareLaunchArgument("demo_keys", default_value="w w w a a s s",
                              description="scripted key sequence for mode:=demo"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("robot_x", default_value=sidecar["robot_x"]),
        DeclareLaunchArgument("robot_y", default_value=sidecar["robot_y"]),
        DeclareLaunchArgument("robot_yaw", default_value=sidecar["robot_yaw"]),
        DeclareLaunchArgument("lane_y", default_value=sidecar["lane_y"]),
        DeclareLaunchArgument("lane_end_x", default_value=sidecar["lane_end_x"]),
        DeclareLaunchArgument("circle_cx", default_value=sidecar["circle_cx"],
                              description="ring-field center x (0 = straight mode off)"),
        DeclareLaunchArgument("circle_cy", default_value=sidecar["circle_cy"],
                              description="ring-field center y"),
        DeclareLaunchArgument("circle_r", default_value=sidecar["circle_r"],
                              description="ring furrow radius; <=0 = straight lane "
                                          "(end at lane_end_x)"),
        DeclareLaunchArgument("max_laps", default_value=sidecar["max_laps_default"],
                              description="ring-field laps before auto-stop; "
                                          "<=0 = loop forever"),
        DeclareLaunchArgument("lambda_x", default_value="2.0",
                              description="servo lateral gain (validated default)"),
        DeclareLaunchArgument("lambda_theta", default_value="1.0",
                              description="servo heading gain (validated default; "
                                          "try 0.5 to reduce inside-cut on rings)"),
        DeclareLaunchArgument("heading_gate", default_value="0.1",
                              description="lateral-priority gate width (normalized "
                                          "lateral error at which heading authority "
                                          "halves; <=0 disables)"),
        DeclareLaunchArgument("ff_gain", default_value="0.0",
                              description="curvature-feedforward gain (experiment, "
                                          "default off)"),
        DeclareLaunchArgument("trim", default_value=sidecar["trim_default"],
                              description="static lateral bias trim in meters, "
                                          "subtracted from raw ey (BEV reference "
                                          "calibration)"),
        DeclareLaunchArgument("max_seconds", default_value="0"),
        DeclareLaunchArgument("log_dir", default_value="/tmp/fftsim_log"),

        # env for gazebo + spawn
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH",
                                  f"{_AGRIBOT_MODELS}:{gazebo_default_models}"),
        SetEnvironmentVariable("MULTIROI_DIR", _MULTIROI),
        SetEnvironmentVariable("FFT_DIR", _FFT),

        # 1) gazebo headless with the generated farm world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
            launch_arguments={
                "world": LaunchConfiguration("world"),
                "gui": LaunchConfiguration("gui"),
                "verbose": "false",
            }.items(),
        ),

        # 2) robot_state_publisher: puts rover URDF on /robot_description
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen",
             parameters=[{
                 "robot_description":
                     (Path(pkg_share) / "urdf" / "rover.urdf").read_text()}]),

        # 3) spawn the rover into the furrow
        Node(package="gazebo_ros", executable="spawn_entity.py",
             output="screen",
             arguments=[
                 "-topic", "robot_description",
                 "-entity", "rover",
                 "-x", LaunchConfiguration("robot_x"),
                 "-y", LaunchConfiguration("robot_y"),
                 "-z", "0.0",
                 "-Y", LaunchConfiguration("robot_yaw"),
             ]),

        # 4) FFT navigation node (idles its /cmd_vel unless mode:=auto)
        nav,
        # 5) mode:=demo - scripted teleop keys (headless check); its exit
        #    shuts the launch down (second handler below)
        demo,
        # when the nav node exits (finished run), shut the whole launch down
        RegisterEventHandler(OnProcessExit(target_action=nav,
                                           on_exit=[Shutdown(reason="nav done")])),
        RegisterEventHandler(OnProcessExit(target_action=demo,
                                           on_exit=[Shutdown(reason="demo done")])),
    ])
