#!/usr/bin/env bash
# Build + run the closed-loop ExG-in-Gazebo test in one shot.
# Headless (no Gazebo window); for the live view use run_sim_gui.sh.
#
#   ./run_sim.sh [--probe] [--seconds 90] [--out /tmp/exgsim_log] [--straight 5]
#
# Launch arguments (passed through to farm.launch.py unless noted):
#   Fields (world + spawn/lane defaults travel together):
#   --circle (default)  --curve[N]  --straight[N]  --zigzag[N]
#     N = row count, e.g. --straight 5. Bare --straight/--curved mean N=2,
#     bare --zigzag means N=5. N=2 and N=5 use committed snapshots; any
#     other count 2..10 is generated on demand into ~/.cache/crop-row-fields.
#   --field NAME   raw preset name (circle, curve, straight, curve5,
#     straight5, zigzag5, custom).
#   --probe        detection check: runs the real C++ stack with
#     publish_cmd_vel disabled (rover never moves), captures frames plus
#     per-frame vs errors through the monitor.
#   --seconds N    stop after N sim-seconds (0 = off; lane end/laps stop first).
#   --out DIR      log dir for nav_run.csv + overlay PNGs.
#   --x/--y/--yaw  spawn pose override (defaults follow the field sidecar).
#   --laps N       ring-field laps before auto-stop, 0 = loop forever.
#   --world PATH   custom world file (sibling .spawn.json seeds defaults).
#   --tof          crop-safety guard: side + angled-front ToF rangers override vision
#     steering when closer than --tof-min (default 0.35 m). High priority:
#     the guard angular command replaces the stack output while violated.
#   --tof-min M / --tof-gain G / --tof-max-w W / --tof-v V  guard tuning
#     (min clearance m, yaw gain rad/s per m, |w| clamp, linear cap m/s).
# The C++ stack drives in auto mode; gains live in
# src/exgsim/params/exgsim_run.yaml (copied from the vendor defaults).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXG_DIR="$(cd "${HERE}/.." && pwd)"                  # .../ExG
REPO_DIR="$(cd "${HERE}/../.." && pwd)"              # repo root
GEN="${REPO_DIR}/LinReg/MultiROI/tests/sim_ros2/src/mrsim/scripts/gen_farm_world.py"
LOG_DIR="/tmp/exgsim_log"
SECONDS_ARG=""
FIELD_ARG=()
WORLD_ARG=()
LAPS_ARG=()
TOF_ARGS=()
PROBE=0
SPAWN_ARGS=()

# resolve_row_field <shape> <n>: committed snapshot for N=2/N=5, generated
# cache world otherwise. Sets FIELD_ARG and/or WORLD_ARG.
resolve_row_field() {
  local shape="$1" n="$2" stem dir world spacing extra
  case "${shape}" in
    straight)
      if [[ "${n}" == "2" ]]; then FIELD_ARG=(field:=straight); return; fi
      if [[ "${n}" == "5" ]]; then FIELD_ARG=(field:=straight5); return; fi
      extra="--shape straight --curve-amp 0" ;;
    curved)
      if [[ "${n}" == "2" ]]; then FIELD_ARG=(field:=curve); return; fi
      if [[ "${n}" == "5" ]]; then FIELD_ARG=(field:=curve5); return; fi
      extra="--shape straight" ;;
    zigzag)
      if [[ "${n}" == "5" ]]; then FIELD_ARG=(field:=zigzag5); return; fi
      extra="--shape zigzag" ;;
  esac
  stem="farm_${shape}_r${n}"
  dir="${XDG_CACHE_HOME:-$HOME/.cache}/crop-row-fields"
  world="${dir}/${stem}.world"
  if [[ ! -f "${world}" ]]; then
    echo "-- generating ${n}-row ${shape} field (one time) -> ${world}"
    mkdir -p "${dir}"
    if [[ "${n}" -gt 2 ]]; then spacing="0.2"; else spacing="0.12"; fi
    # shellcheck disable=SC2086
    python3 "${GEN}" \
      --rows "${n}" --plant-spacing "${spacing}" ${extra} \
      --out "${world}" || { echo "field generation failed"; exit 1; }
  fi
  WORLD_ARG=(world:="${world}")
}

# _take_number <default>: consume $2 into N if it is a positive integer.
_take_number() {
  if [[ "${2:-}" =~ ^[0-9]+$ ]]; then N="$2"; return 0; fi
  N="$1"; return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe) PROBE=1; shift ;;
    --seconds) SECONDS_ARG="max_seconds:=${2}"; shift 2 ;;
    --out) LOG_DIR="$2"; shift 2 ;;
    --world) WORLD_ARG=(world:="$2"); shift 2 ;;
    --circle) FIELD_ARG=(field:=circle); shift ;;
    --curve|--curved)
      if _take_number 2 "$2"; then shift 2; else shift; fi
      if [[ "${N}" -lt 2 || "${N}" -gt 10 ]]; then
        echo "--curved needs a row count 2..10"; exit 1; fi
      resolve_row_field curved "${N}" ;;
    --straight)
      if _take_number 2 "$2"; then shift 2; else shift; fi
      if [[ "${N}" -lt 2 || "${N}" -gt 10 ]]; then
        echo "--straight needs a row count 2..10"; exit 1; fi
      resolve_row_field straight "${N}" ;;
    --zigzag)
      if _take_number 5 "$2"; then shift 2; else shift; fi
      if [[ "${N}" -lt 2 || "${N}" -gt 10 ]]; then
        echo "--zigzag needs a row count 2..10"; exit 1; fi
      resolve_row_field zigzag "${N}" ;;
    --field) FIELD_ARG=(field:="$2"); shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2"); shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2"); shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2"); shift 2 ;;
    --laps) LAPS_ARG=(max_laps:="$2"); shift 2 ;;
    --tof) TOF_ARGS+=(tof:=true); shift ;;
    --tof-min) TOF_ARGS+=(tof_min:="$2"); shift 2 ;;
    --tof-gain) TOF_ARGS+=(tof_gain:="$2"); shift 2 ;;
    --tof-max-w) TOF_ARGS+=(tof_max_w:="$2"); shift 2 ;;
    --tof-v) TOF_ARGS+=(tof_v:="$2"); shift 2 ;;
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
    out_dir:="${LOG_DIR}" "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" || true
else
  echo "== closed loop: ExG stack drives the furrow =="
  timeout "${SIM_TIMEOUT:-600}" ros2 launch exgsim farm.launch.py \
    log_dir:="${LOG_DIR}" \
    ${SECONDS_ARG:+${SECONDS_ARG}} "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" "${LAPS_ARG[@]}" "${TOF_ARGS[@]}" || true
fi

echo
echo "== results in ${LOG_DIR} =="
ls -la "${LOG_DIR}" | head
