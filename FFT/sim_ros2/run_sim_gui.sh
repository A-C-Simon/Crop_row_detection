#!/usr/bin/env bash
# Run the sim with a live Gazebo GUI AND interactive keyboard teleop, for use
# at the machine's real display (see tests/README.md).
#
#   ./run_sim_gui.sh               # GUI + teleop (nav node idles)
#   ./run_sim_gui.sh --auto        # GUI + autonomous FFT driving
#   ./run_sim_gui.sh --demo        # GUI + scripted keys (no keyboard needed)
#   extras: --x/--y/--yaw spawn override (defaults follow the committed
#           world sidecar), --laps N (ring-field laps; default 0.5 = half-ring
#           demo, 0 = forever), --world PATH (e.g. src/fftsim/worlds/farm_curve.world
#           for the S-bend field; then also pass its spawn explicitly)
#
# The teleop keys are typed into the terminal that runs this script
# (w/s = fwd/back, a/d = turn, space = stop, x = quit).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_DIR="$(cd "${HERE}/.." && pwd)"                  # .../FFT
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)/LinReg/MultiROI"  # shared servo
MODE="teleop"
DEMO_KEYS=""
LOG_DIR="/tmp/fftsim_gui"
LAPS="0.5"
SPAWN_ARGS=()
WORLD_ARG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto) MODE="auto" ; shift ;;
    --demo) MODE="demo" ; shift ;;
    --keys) DEMO_KEYS="$2" ; shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2") ; shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2") ; shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2") ; shift 2 ;;
    --laps) LAPS="$2" ; shift 2 ;;
    --world) WORLD_ARG=(world:="$2") ; shift 2 ;;
    *) echo "unknown arg $1 (--auto|--demo|--keys|--x|--y|--yaw|--laps|--world)"; exit 1 ;;
  esac
done

source /opt/ros/humble/setup.bash
export FFT_DIR
export MULTIROI_DIR
export MRSIM_SIM_MODE="${MODE}"   # read by farm.launch.py to idle the nav node
export GAZEBO_MODEL_PATH="${MULTIROI_DIR}/tests/agribot/agribot_gazebo/models:/usr/share/gazebo-11/models"

echo "== building fftsim workspace =="
cd "${HERE}"
colcon build --symlink-install --base-paths src 2>&1 | tail -2
source install/setup.bash

echo "== launching Gazebo GUI + mode=${MODE} (Ctrl-C stops) =="
ARGS=(mode:="${MODE}" gui:=true log_dir:="${LOG_DIR}" max_laps:="${LAPS}")
[ -n "${DEMO_KEYS}" ] && ARGS+=(demo_keys:="${DEMO_KEYS}")
ros2 launch fftsim farm.launch.py "${ARGS[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}"
