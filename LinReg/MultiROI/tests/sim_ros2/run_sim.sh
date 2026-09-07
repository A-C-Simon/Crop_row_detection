#!/usr/bin/env bash
# Build + run the closed-loop MultiROI-in-Gazebo test in one shot.
#
#   ./run_sim.sh [--probe] [--seconds 90] [--out /tmp/mrsim_log] [--x -8] [--y 0.4] [--yaw 0.2] [--laps 2]
#
# Spawn/--laps defaults follow the committed world's sidecar
# (src/mrsim/worlds/farm_maize.spawn.json); pass explicitly to override.
#
# --probe : calibration mode - spawns the farm + rover, captures a few camera
#           frames through the detector and prints diagnostics (no driving).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)"          # .../LinReg/MultiROI
LOG_DIR="/tmp/mrsim_log"
SECONDS_ARG=""
LAPS_ARG=()
GAIN_ARGS=()
PROBE=0
SPAWN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe) PROBE=1; shift ;;
    --seconds) SECONDS_ARG="max_seconds:=${2}"; shift 2 ;;
    --out) LOG_DIR="$2"; shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2"); shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2"); shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2"); shift 2 ;;
    --laps) LAPS_ARG=(max_laps:="$2"); shift 2 ;;
    --lambdax) GAIN_ARGS+=(lambda_x:="$2"); shift 2 ;;
    --lambdat) GAIN_ARGS+=(lambda_theta:="$2"); shift 2 ;;
    --ff) GAIN_ARGS+=(ff_gain:="$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

source /opt/ros/humble/setup.bash
export MULTIROI_DIR
export GAZEBO_MODEL_PATH="${MULTIROI_DIR}/tests/agribot/agribot_gazebo/models:/usr/share/gazebo-11/models"

echo "== building mrsim workspace =="
cd "${HERE}"
colcon build --symlink-install --base-paths src 2>&1 | tail -3
source install/setup.bash

mkdir -p "${LOG_DIR}"
if [[ "$PROBE" == "1" ]]; then
  echo "== probe mode: capture frames through the detector =="
  timeout 120 ros2 launch mrsim farm_probe.launch.py \
    out_dir:="${LOG_DIR}" "${SPAWN_ARGS[@]}" || true
else
  echo "== closed loop: nav node drives the furrow =="
  timeout "${SIM_TIMEOUT:-600}" ros2 launch mrsim farm.launch.py \
    log_dir:="${LOG_DIR}" \
    ${SECONDS_ARG:+${SECONDS_ARG}} "${SPAWN_ARGS[@]}" "${LAPS_ARG[@]}" "${GAIN_ARGS[@]}" || true
fi

echo
echo "== results in ${LOG_DIR} =="
ls -la "${LOG_DIR}" | head
