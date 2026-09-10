"""Launch the mrsim closed-loop test: maize farm + rover + MultiROI nav node.

    export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI
    ros2 launch mrsim farm.launch.py log_dir:=/tmp/mrsim_log
    ros2 launch mrsim farm.launch.py field:=zigzag5 log_dir:=/tmp/z

Starts gzserver headless on the selected field, spawns the rover in the
configured furrow, and runs `nav_node.py` (camera -> MultiROI
detector/temporal filter/mr_vs -> /cmd_vel). The launch shuts everything
down when the nav node exits (end of lane / laps / max time / Ctrl-C).

Fields (world + spawn/lane defaults travel together; explicit robot_*/
lane/circle args always win):
    circle    2-row ring R12 (default)      curve     2-row S-bend
    straight  2-row straight                curve5    5-row S-bend
    straight5 5-row straight                zigzag5   5-row gentle zigzag
Custom world file: world:=/path/to.world (sibling .spawn.json used when
present, else the field sidecar).
Drive another furrow: spawn_row:=2 (1-based from the left). Chain furrows
with row_change:=true (headland turn at each lane end, max_lanes bounds
the demo).

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
                            RegisterEventHandler, ExecuteProcess,
                            OpaqueFunction)
from launch.conditions import IfCondition
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

# MultiROI source dir: /abs/path/to/LinReg/MultiROI (contains run_mr_navigation.py)
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
if not (Path(_AGRIBOT_MODELS) / "big_plant").is_dir():
    raise RuntimeError(f"agribot plant models not found at {_AGRIBOT_MODELS}")

# node sources: live in the MultiROI sim workspace (src tree).
_SIM_SRC = Path(_MULTIROI) / "tests" / "sim_ros2" / "src" / "mrsim"
_NAV_PY = _SIM_SRC / "mrsim" / "nav_node.py"
_TELEOP_PY = _SIM_SRC / "mrsim" / "teleop_node.py"
if not (_NAV_PY.exists() and _TELEOP_PY.exists()):
    raise RuntimeError(f"mrsim nodes not found under {_SIM_SRC}")


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
                "max_laps_default": "1", "vertical_coverage_default": "0.75",
                "n_rows": "2", "row_spacing": "1.1", "furrows": "[0.0]"}
    for sc in reversed(chain):
        for k in ("robot_x", "robot_y", "robot_yaw", "lane_y",
                  "lane_end_x", "circle_cx", "circle_cy", "circle_r"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
        for k in ("max_laps_default", "vertical_coverage_default"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
        for k in ("n_rows", "row_spacing"):
            if sc.get(k) is not None:
                defaults[k] = str(sc[k])
        if sc.get("furrows") is not None:
            try:
                defaults["furrows"] = json.dumps(
                    [float(c) for c in sc["furrows"]])
            except Exception:
                pass
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
    lane_end_x = val("lane_end_x", "lane_end_x", "9.0")
    circle_cx = val("circle_cx", "circle_cx", "0.0")
    circle_cy = val("circle_cy", "circle_cy", "0.0")
    circle_r = val("circle_r", "circle_r", "0.0")
    max_laps = val("max_laps", "max_laps_default", "1")
    vertical_coverage = val("vertical_coverage", "vertical_coverage_default",
                            "0.75")

    # --spawn N: drive the Nth furrow from the left (1-based). Overrides
    # the sidecar spawn/lane unless robot_y/lane_y were passed explicitly.
    try:
        furrows = [float(c) for c in json.loads(sidecar.get("furrows", "[0.0]"))]
    except Exception:
        furrows = [0.0]
    lane_index = 0
    try:
        lane_index = min(range(len(furrows)),
                         key=lambda i: abs(furrows[i] - float(lane_y)))
    except Exception:
        pass
    spawn_row = cfg.get("spawn_row", "")
    if spawn_row != "":
        try:
            n = int(spawn_row)
        except ValueError:
            raise RuntimeError(f"spawn_row must be 1..{len(furrows)}, got '{spawn_row}'")
        if not (1 <= n <= len(furrows)):
            raise RuntimeError(
                f"spawn_row {n} out of range for {len(furrows)} furrow(s) "
                f"at {sorted(furrows)}")
        lane_index = n - 1
        if cfg.get("robot_y", "") == "":
            robot_y = f"{furrows[lane_index]:.3f}"
        if cfg.get("lane_y", "") == "":
            lane_y = f"{furrows[lane_index]:.3f}"

    # idle env: nav node runs detection/overlay but never publishes /cmd_vel
    # unless mode:=auto (so teleop/demo own the topic)
    nav_idle = "1" if os.environ.get("MRSIM_SIM_MODE", "auto") != "auto" else ""
    nav = ExecuteProcess(
        cmd=[sys.executable, str(_NAV_PY)],
        output="screen",
        additional_env={
            "MRSIM_MODE": "nav",
            "MRSIM_ALGORITHM": os.environ.get("MRSIM_ALGORITHM", "multiroi"),
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
            "MRSIM_KI": cfg.get("ki", "0.3"),
            "MRSIM_FF_GAIN": cfg.get("ff_gain", "0.0"),
            "MRSIM_VERTICAL_COVERAGE": vertical_coverage,
            "MRSIM_LINE_FIT": cfg.get("line_fit", "false"),
            "MRSIM_INIT_WINDOW": cfg.get("init_window", "1.0"),
            "MRSIM_ROW_CHANGE": cfg.get("row_change", "false"),
            "MRSIM_MAX_LANES": cfg.get("max_lanes", "2"),
            "MRSIM_LANE_START_X": robot_x,
            "MRSIM_LANE_INDEX": str(lane_index),
            "MRSIM_FURROWS": ",".join(f"{c:.3f}" for c in furrows),
            "MRSIM_LOG_DIR": cfg.get("log_dir", "/tmp/mrsim_log"),
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
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH",
                                  f"{_AGRIBOT_MODELS}:{gazebo_default_models}"),
        AppendEnvironmentVariable("MULTIROI_DIR", _MULTIROI),

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

        # 3) spawn the rover into the furrow. NOTE: spawned from rover.sdf,
        # not the URDF topic: Gazebo 11's live URDF parsing silently drops
        # the second camera sensor, while the converted SDF keeps both.
        # After editing rover.urdf, regenerate with:
        #   gz sdf -p src/mrsim/urdf/rover.urdf > src/mrsim/urdf/rover.sdf
        # (robot_state_publisher still uses the URDF for TF.)
        Node(package="gazebo_ros", executable="spawn_entity.py",
             output="screen",
             arguments=[
                 "-file", str(Path(pkg_share) / "urdf" / "rover.sdf"),
                 "-entity", "rover",
                 "-x", robot_x,
                 "-y", robot_y,
                 "-z", "0.0",
                 "-Y", robot_yaw,
             ]),

        # 4) MultiROI navigation node (idles its /cmd_vel unless mode:=auto)
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
                              description="auto=MultiROI drives; "
                                          "teleop=keyboard node drives "
                                          "(nav node idles, GUI view); "
                                          "demo=scripted keys (headless test)"),
        DeclareLaunchArgument("algorithm", default_value="multiroi",
                              description="detection algorithm: multiroi (only "
                                          "one wired so far; see pipeline.py)"),
        DeclareLaunchArgument("demo_keys", default_value="w w w a a s s",
                              description="scripted key sequence for mode:=demo"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("robot_x", default_value="",
                              description="spawn x (empty = field default)"),
        DeclareLaunchArgument("robot_y", default_value="",
                              description="spawn y (empty = field default)"),
        DeclareLaunchArgument("robot_yaw", default_value="",
                              description="spawn yaw (empty = field default)"),
        DeclareLaunchArgument("spawn_row", default_value="",
                              description="drive the Nth furrow from the left "
                                          "(1-based; empty = sidecar spawn; "
                                          "overrides robot_y/lane_y unless set)"),
        DeclareLaunchArgument("row_change", default_value="false",
                              description="at the lane end, headland-turn into "
                                          "the next furrow and keep going "
                                          "(straight fields only)"),
        DeclareLaunchArgument("max_lanes", default_value="2",
                              description="lanes to cover with row_change on "
                                          "(0 = until Ctrl-C)"),
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
        DeclareLaunchArgument("ki", default_value="0.3",
                              description="lateral integral gain against steady "
                                          "inside-cut (0 = off)"),
        DeclareLaunchArgument("ff_gain", default_value="0.0",
                              description="curvature-feedforward gain (experiment, "
                                          "default off)"),
        DeclareLaunchArgument("line_fit", default_value="false",
                              description="straight least-squares nav line "
                                          "instead of the smoothing spline "
                                          "(same switch as --line offline)"),
        DeclareLaunchArgument("init_window", default_value="1.0",
                              description="strip-1 search width as a fraction "
                                          "of image width (narrower anchors "
                                          "the driven furrow when several "
                                          "identical corridors are visible)"),
        DeclareLaunchArgument("vertical_coverage", default_value="",
                              description="ROI height fraction from the image "
                                          "bottom (empty = field default)"),
        DeclareLaunchArgument("max_seconds", default_value="0"),
        DeclareLaunchArgument("log_dir", default_value="/tmp/mrsim_log"),
        OpaqueFunction(function=_setup),
    ])
