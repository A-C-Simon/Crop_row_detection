#!/usr/bin/env bash
# Run the sim with a live Gazebo GUI AND interactive keyboard teleop, for use
# at the machine's real display (see tests/README.md).
#
#   ./run_sim_gui.sh               # GUI + teleop (nav node idles)
#   ./run_sim_gui.sh --auto        # GUI + autonomous MultiROI driving
#   ./run_sim_gui.sh --demo        # GUI + scripted keys (no keyboard needed)
#   extras: --x/--y/--yaw spawn override (defaults follow the committed
#           world sidecar), --laps N (ring-field laps; default 0.5 = half-ring
#           demo, 0 = forever)
#
# The teleop keys are typed into the terminal that runs this script
# (w/s = fwd/back, a/d = turn, space = stop, x = quit).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)"          # .../LinReg/MultiROI
MODE="teleop"
DEMO_KEYS=""
ALGO="multiroi"
LOG_DIR="/tmp/mrsim_gui"
LAPS="0.5"
SPAWN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto) MODE="auto" ; shift ;;
    --demo) MODE="demo" ; shift ;;
    --keys) DEMO_KEYS="$2" ; shift 2 ;;
    --algo) ALGO="$2" ; shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2") ; shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2") ; shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2") ; shift 2 ;;
    --laps) LAPS="$2" ; shift 2 ;;
    *) echo "unknown arg $1 (--auto|--demo|--keys|--algo|--x|--y|--yaw|--laps)"; exit 1 ;;
  esac
done

source /opt/ros/humble/setup.bash
export MULTIROI_DIR
export MRSIM_SIM_MODE="${MODE}"   # read by farm.launch.py to idle the nav node
export MRSIM_ALGORITHM="${ALGO}"   # detection algorithm (default: multiroi)
export GAZEBO_MODEL_PATH="${MULTIROI_DIR}/tests/agribot/agribot_gazebo/models:/usr/share/gazebo-11/models"

echo "== building mrsim workspace =="
cd "${HERE}"
colcon build --symlink-install --base-paths src 2>&1 | tail -2
source install/setup.bash

echo "== launching Gazebo GUI + mode=${MODE} (Ctrl-C stops) =="
ARGS=(mode:="${MODE}" gui:=true log_dir:="${LOG_DIR}" max_laps:="${LAPS}")
[ -n "${DEMO_KEYS}" ] && ARGS+=(demo_keys:="${DEMO_KEYS}")
ros2 launch mrsim farm.launch.py "${ARGS[@]}" "${SPAWN_ARGS[@]}"
