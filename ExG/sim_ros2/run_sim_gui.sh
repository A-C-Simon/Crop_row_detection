#!/usr/bin/env bash
# Run the ExG sim with a live Gazebo GUI AND interactive keyboard teleop, for
# use at the machine's real display.
#
#   ./run_sim_gui.sh               # GUI + teleop (C++ stack idles)
#   ./run_sim_gui.sh --auto        # GUI + autonomous ExG driving
#   ./run_sim_gui.sh --demo        # GUI + scripted keys (no keyboard needed)
#   extras: --x/--y/--yaw spawn override (defaults follow the committed
#           world sidecar), --laps N (ring-field laps; default 0.5 = half-ring
#           demo, 0 = forever)
#
# In teleop mode run the keyboard driver by hand in this terminal after the
# sim is up:  source install/setup.bash && ros2 run exgsim exg_teleop
# (w/s = fwd/back, a/d = turn, space = stop, x = quit).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXG_DIR="$(cd "${HERE}/.." && pwd)"                  # .../ExG
REPO_DIR="$(cd "${HERE}/../.." && pwd)"              # repo root
MODE="teleop"
DEMO_KEYS=""
LOG_DIR="/tmp/exgsim_gui"
LAPS="0.5"
SPAWN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto) MODE="auto" ; shift ;;
    --demo) MODE="demo" ; shift ;;
    --keys) DEMO_KEYS="$2" ; shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2") ; shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2") ; shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2") ; shift 2 ;;
    --laps) LAPS="$2" ; shift 2 ;;
    *) echo "unknown arg $1 (--auto|--demo|--keys|--x|--y|--yaw|--laps)"; exit 1 ;;
  esac
done

source /opt/ros/humble/setup.bash
export EXG_DIR
export MRSIM_SIM_MODE="${MODE}"   # read by farm.launch.py to idle the C++ stack
export GAZEBO_MODEL_PATH="${REPO_DIR}/LinReg/MultiROI/tests/agribot/agribot_gazebo/models:/usr/share/gazebo-11/models"

echo "== building exgsim workspace =="
cd "${HERE}"
if [[ ! -x "exg_ws/install/visual_crop_row_navigation_ros2/lib/visual_crop_row_navigation_ros2/agribot_vs_node" ]]; then
  echo "-- C++ stack missing, building it first (one time, ~1 min)"
  mkdir -p exg_ws && cd exg_ws && \
    colcon build --base-paths ../../visual-crop-row-navigation_ros2 \
      --build-base build --install-base install 2>&1 | tail -2
  cd "${HERE}"
fi
colcon build --symlink-install --base-paths src 2>&1 | tail -2
source exg_ws/install/setup.bash
source install/setup.bash

echo "== launching Gazebo GUI + mode=${MODE} (Ctrl-C stops) =="
ARGS=(mode:="${MODE}" gui:=true log_dir:="${LOG_DIR}" max_laps:="${LAPS}")
[ -n "${DEMO_KEYS}" ] && ARGS+=(demo_keys:="${DEMO_KEYS}")
ros2 launch exgsim farm.launch.py "${ARGS[@]}" "${SPAWN_ARGS[@]}"
