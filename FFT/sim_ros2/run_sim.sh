#!/usr/bin/env bash
# Build + run the closed-loop FFT-in-Gazebo test in one shot.
# Headless (no Gazebo window); for the live view use run_sim_gui.sh.
#
#   ./run_sim.sh [--probe] [--seconds 90] [--out /tmp/fftsim_log] [--straight 5]
#
# Launch arguments (passed through to farm.launch.py unless noted):
#   Fields (world + spawn/lane defaults travel together):
#   --circle (default)  --curve[N]  --straight[N]  --zigzag[N]
#     N = row count, e.g. --straight 5. Bare --straight/--curved mean N=2,
#     bare --zigzag means N=5. N=2 and N=5 use committed snapshots; any
#     other count 2..10 is generated on demand into ~/.cache/crop-row-fields.
#   --field NAME   raw preset name (circle, curve, straight, curve5,
#     straight5, zigzag5, custom).
#   --probe        calibration mode: spawn world+rover, capture a few camera
#     frames through the detector and print diagnostics (no driving).
#   --seconds N    stop after N sim-seconds (0 = off; lane end/laps stop first).
#   --out DIR      log dir for nav_run.csv + overlay PNGs.
#   --x/--y/--yaw  spawn pose override (defaults follow the field sidecar).
#   --laps N       ring-field laps before auto-stop, 0 = loop forever.
#   --lambdax/--lambdat/--gate/--ff/--trim  servo and DFT tuning (validated
#     defaults; see farm.launch.py descriptions).
#   --tof          crop-safety guard: side + angled-front ToF rangers override vision
#     steering when closer than --tof-min (default 0.35 m). High priority:
#     the guard angular command replaces the servo output while violated.
#   --tof-min M / --tof-gain G / --tof-max-w W / --tof-v V  guard tuning
#     (min clearance m, yaw gain rad/s per m, |w| clamp, linear cap m/s).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_DIR="$(cd "${HERE}/.." && pwd)"                  # .../FFT
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)/LinReg/MultiROI"  # shared servo
LOG_DIR="/tmp/fftsim_log"
SECONDS_ARG=""
FIELD_ARG=()
WORLD_ARG=()
LAPS_ARG=()
GAIN_ARGS=()
TRIM_ARG=()
TOF_ARGS=()
PROBE=0
SPAWN_ARGS=()

# resolve_row_field <shape> <n>: committed snapshot for N=2/N=5, generated
# cache world otherwise. Sets FIELD_ARG and/or WORLD_ARG. Generation reuses
# the MultiROI field generator (same geometry everywhere).
resolve_row_field() {
  local shape="$1" n="$2" stem dir world spacing extra gen
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
    gen="${MULTIROI_DIR}/tests/sim_ros2/src/mrsim/scripts/gen_farm_world.py"
    # shellcheck disable=SC2086
    python3 "${gen}" \
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
    --lambdax) GAIN_ARGS+=(lambda_x:="$2"); shift 2 ;;
    --lambdat) GAIN_ARGS+=(lambda_theta:="$2"); shift 2 ;;
    --gate) GAIN_ARGS+=(heading_gate:="$2"); shift 2 ;;
    --ff) GAIN_ARGS+=(ff_gain:="$2"); shift 2 ;;
    --trim) TRIM_ARG=(trim:="$2"); shift 2 ;;
    --tof) TOF_ARGS+=(tof:=true); shift ;;
    --tof-min) TOF_ARGS+=(tof_min:="$2"); shift 2 ;;
    --tof-gain) TOF_ARGS+=(tof_gain:="$2"); shift 2 ;;
    --tof-max-w) TOF_ARGS+=(tof_max_w:="$2"); shift 2 ;;
    --tof-v) TOF_ARGS+=(tof_v:="$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

source /opt/ros/humble/setup.bash
export FFT_DIR
export MULTIROI_DIR
export GAZEBO_MODEL_PATH="${MULTIROI_DIR}/tests/agribot/agribot_gazebo/models:/usr/share/gazebo-11/models"

echo "== building fftsim workspace =="
cd "${HERE}"
colcon build --symlink-install --base-paths src 2>&1 | tail -3
source install/setup.bash

mkdir -p "${LOG_DIR}"
if [[ "$PROBE" == "1" ]]; then
  echo "== probe mode: capture frames through the detector =="
  timeout 120 ros2 launch fftsim farm_probe.launch.py \
    out_dir:="${LOG_DIR}" "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" || true
else
  echo "== closed loop: nav node drives the furrow =="
  timeout "${SIM_TIMEOUT:-600}" ros2 launch fftsim farm.launch.py \
    log_dir:="${LOG_DIR}" \
    ${SECONDS_ARG:+${SECONDS_ARG}} "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" "${LAPS_ARG[@]}" "${GAIN_ARGS[@]}" "${TRIM_ARG[@]}" "${TOF_ARGS[@]}" || true
fi

echo
echo "== results in ${LOG_DIR} =="
ls -la "${LOG_DIR}" | head
