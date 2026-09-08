#!/usr/bin/env bash
# Build + run the closed-loop ExG-in-Gazebo test in one shot.
#
#   ./run_sim.sh [--probe] [--seconds 90] [--out /tmp/exgsim_log] [--x 12] [--y 0] [--yaw 1.57] [--laps 1]
#                [--world src/exgsim/worlds/farm_curve.world]
#
# Spawn/--laps defaults follow the committed world's sidecar
# (src/exgsim/worlds/farm_maize.spawn.json); pass explicitly to override.
# The C++ stack drives in auto mode; gains live in
# src/exgsim/params/exgsim_run.yaml (copied from the vendor defaults).
#
# --probe : detection check - runs the real C++ stack with publish_cmd_vel
#           disabled (rover never moves), captures frames plus per-frame
#           vs errors through the monitor.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXG_DIR="$(cd "${HERE}/.." && pwd)"                  # .../ExG
REPO_DIR="$(cd "${HERE}/../.." && pwd)"              # repo root
LOG_DIR="/tmp/exgsim_log"
SECONDS_ARG=""
WORLD_ARG=()
LAPS_ARG=()
PROBE=0
SPAWN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe) PROBE=1; shift ;;
    --seconds) SECONDS_ARG="max_seconds:=${2}"; shift 2 ;;
    --out) LOG_DIR="$2"; shift 2 ;;
    --world) WORLD_ARG=(world:="$2"); shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2"); shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2"); shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2"); shift 2 ;;
    --laps) LAPS_ARG=(max_laps:="$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

source /opt/ros/humble/setup.bash
export EXG_DIR
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

mkdir -p "${LOG_DIR}"
if [[ "$PROBE" == "1" ]]; then
  echo "== probe mode: C++ stack detects, wheels disabled =="
  timeout 120 ros2 launch exgsim farm_probe.launch.py \
    out_dir:="${LOG_DIR}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" || true
else
  echo "== closed loop: ExG stack drives the furrow =="
  timeout "${SIM_TIMEOUT:-600}" ros2 launch exgsim farm.launch.py \
    log_dir:="${LOG_DIR}" \
    ${SECONDS_ARG:+${SECONDS_ARG}} "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" "${LAPS_ARG[@]}" || true
fi

echo
echo "== results in ${LOG_DIR} =="
ls -la "${LOG_DIR}" | head
