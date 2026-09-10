"""Launch the fftsim closed-loop test: maize farm + rover + FFT nav node.

    export FFT_DIR=/abs/path/to/FFT
    export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI   # shared servo only
    ros2 launch fftsim farm.launch.py log_dir:=/tmp/fftsim_log
    ros2 launch fftsim farm.launch.py field:=zigzag5 log_dir:=/tmp/z

Starts gzserver headless on the selected field, spawns the rover in the
configured furrow, and runs `nav_node.py` (camera -> DFT detector ->
shared visual servo -> /cmd_vel). The launch shuts everything down when
the nav node exits (end of lane / laps / max time / Ctrl-C).

Fields (world + spawn/lane defaults travel together; explicit robot_*/
lane/circle args always win):
    circle    2-row ring R12 (default)      curve     2-row S-bend
    straight  2-row straight                curve5    5-row S-bend
    straight5 5-row straight                zigzag5   5-row gentle zigzag
Custom world file: world:=/path/to.world (sibling .spawn.json used when
present, else the field sidecar).

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
                            RegisterEventHandler, ExecuteProcess,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
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


FIELDS = {
    "circle": "farm_maize",
    "curve": "farm_curve",
    "straight": "farm_straight",
    "curve5": "farm_curve5",
    "straight5": "farm_straight5",
    "zigzag5": "farm_zigzag5",
}


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
                "max_laps_default": "1", "trim_default": "0.0"}
    for sc in reversed(chain):
        for k in ("robot_x", "robot_y", "robot_yaw", "lane_y",
                  "lane_end_x", "circle_cx", "circle_cy", "circle_r"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
        for k in ("max_laps_default", "trim_default"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
    return world_file, defaults


def _setup(context):
    cfg = context.launch_configurations
    pkg_share = get_package_share_directory(PACKAGE)
    gazebo_share = get_package_share_directory("gazebo_ros")
    gazebo_default_models = "/usr/share/gazebo-11/models"
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
    # default start is lane 1 (first furrow of the sidecar) unless the
    # rover is placed explicitly.
    if cfg.get("robot_y", "") == "" and cfg.get("lane_y", "") == "":
        try:
            sib = str(Path(world_file).with_suffix("")) + ".spawn.json"
            furrows = [float(c) for c in
                       json.loads(Path(sib).read_text()).get("furrows", [])]
            if furrows:
                robot_y = lane_y = f"{furrows[0]:.3f}"
        except Exception:
            pass
    lane_end_x = val("lane_end_x", "lane_end_x", "9.0")
    circle_cx = val("circle_cx", "circle_cx", "0.0")
    circle_cy = val("circle_cy", "circle_cy", "0.0")
    circle_r = val("circle_r", "circle_r", "0.0")
    max_laps = val("max_laps", "max_laps_default", "1")
    trim = val("trim", "trim_default", "0.0")

    # idle env: nav node runs detection/overlay but never publishes /cmd_vel
    # unless mode:=auto (so teleop/demo own the topic)
    nav_idle = "1" if os.environ.get("MRSIM_SIM_MODE", "auto") != "auto" else ""
    nav = ExecuteProcess(
        cmd=[sys.executable, str(_NAV_PY)],
        output="screen",
        additional_env={
            "MRSIM_MODE": "nav",
            "MRSIM_NAV_IDLE": nav_idle,
            "MRSIM_LANE_Y": lane_y,
            "MRSIM_LANE_END_X": lane_end_x,
            "MRSIM_CIRCLE_CX": circle_cx,
            "MRSIM_CIRCLE_CY": circle_cy,
            "MRSIM_CIRCLE_R": circle_r,
            "MRSIM_CIRCLE_LAPS": max_laps,
            "MRSIM_LAMBDA_X": cfg.get("lambda_x", "2.0"),
            "MRSIM_LAMBDA_THETA": cfg.get("lambda_theta", "1.0"),
            "MRSIM_HEADING_GATE": cfg.get("heading_gate", "0.1"),
            "MRSIM_FF_GAIN": cfg.get("ff_gain", "0.0"),
            "FFT_TRIM_M": trim,
            "MRSIM_LOG_DIR": cfg.get("log_dir", "/tmp/fftsim_log"),
            "MRSIM_MAX_SECONDS": cfg.get("max_seconds", "0"),
        })

    # mode:=demo - scripted keys, headless-verifiable. The demo process exits
    # after its sequence; when it does, the handler below shuts the launch
    # down (nav idles, so it would otherwise run forever).
    demo = ExecuteProcess(
        cmd=[sys.executable, str(_TELEOP_PY)],
        output="screen",
        condition=IfCondition("1" if os.environ.get("MRSIM_SIM_MODE", "auto") == "demo" else "0"),
        additional_env={"MRSIM_DEMO_KEYS": cfg.get("demo_keys", "w w w a a s s")})

    return [
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
                "world": world_file,
                "gui": cfg.get("gui", "false"),
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
                 "-x", robot_x,
                 "-y", robot_y,
                 "-z", "0.0",
                 "-Y", robot_yaw,
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
                              description="auto=FFT drives; "
                                          "teleop=keyboard node drives "
                                          "(nav node idles, GUI view); "
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
        DeclareLaunchArgument("trim", default_value="",
                              description="static lateral bias trim in meters "
                                          "(empty = field default)"),
        DeclareLaunchArgument("max_seconds", default_value="0"),
        DeclareLaunchArgument("log_dir", default_value="/tmp/fftsim_log"),
        OpaqueFunction(function=_setup),
    ])
