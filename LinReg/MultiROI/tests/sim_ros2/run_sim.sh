#!/usr/bin/env bash
# Build + run the closed-loop MultiROI-in-Gazebo test in one shot.
# Headless (no Gazebo window); for the live view use run_sim_gui.sh.
#
#   ./run_sim.sh [--probe] [--seconds 90] [--out /tmp/mrsim_log] [--field zigzag5]
#
# Launch arguments (passed through to farm.launch.py unless noted):
#   Fields (world + spawn/lane defaults travel together):
#   --circle (default)  --curve[N]  --straight[N]  --zigzag[N]
#     N = row count, e.g. --straight 5. Bare --straight/--curved mean N=2,
#     bare --zigzag means N=5. N=2 and N=5 use committed snapshots; any
#     other count 2..10 is generated on demand into ~/.cache/crop-row-fields
#     (deterministic seed, reused when present).
#   --field NAME   raw preset name (circle, curve, straight, curve5,
#     straight5, zigzag5, custom).
#   --probe        calibration mode: spawn world+rover, capture a few camera
#     frames through the detector and print diagnostics (no driving).
#   --seconds N    stop after N sim-seconds (0 = off; lane end/laps stop first).
#   --out DIR      log dir for nav_run.csv + overlay PNGs.
#   --x/--y/--yaw  spawn pose override (defaults follow the field sidecar).
#   --spawn N      drive the Nth furrow from the left, 1-based (default
#     lane 1; overrides the sidecar spawn/lane unless --x/--y/--lane-y
#     are also given).
#   --row-change / --rows-change 0|1: headland-turn into the next furrow at
#     each lane end (straight fields only; odometry-scripted bulb turn,
#     vision keeps drawing but does not steer during the maneuver).
#     Automatic on fields with 2+ furrows unless --rows-change 0 is given.
#   --max-lanes N  lanes to cover before auto-stop (0 = until Ctrl-C;
#     defaults to a full sweep of the field).
#   --turn-mode bulb|fishtail|shuttle: headland turn style (bulb = odometry
#     push/spin/slide/spin; fishtail = rear-guided reverse-in, no
#     spinning; shuttle = no turns: vision row-end, lateral jog, lanes
#     alternate forward/backward; default bulb).
#   --laps N       ring-field laps before auto-stop, 0 = loop forever.
#   --line         straight least-squares nav line instead of the spline.
#   --coverage F   ROI height fraction from the image bottom (shorter
#     lookahead cuts inside less on curves, sees less ahead).
#   --init-window F  strip-1 search width as image fraction (narrower
#     anchors the driven furrow when identical corridors compete).
#   --lambdax/--lambdat/--gate/--ki/--ff  servo tuning (validated defaults;
#     see farm.launch.py descriptions).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)"          # .../LinReg/MultiROI
LOG_DIR="/tmp/mrsim_log"
SECONDS_ARG=""
FIELD_ARG=()
WORLD_ARG=()
LAPS_ARG=()
GAIN_ARGS=()
EXTRA_ARGS=()
PROBE=0
SPAWN_ARGS=()
HAVE_RC=0
RC_VALUE=""
HAVE_MAXLANES=0

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
    python3 "${HERE}/src/mrsim/scripts/gen_farm_world.py" \
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
    --spawn) SPAWN_ARGS+=(spawn_row:="$2"); shift 2 ;;
    --row-change) HAVE_RC=1; RC_VALUE=1; shift ;;
    --rows-change)
      if [[ "$2" != "0" && "$2" != "1" ]]; then
        echo "--rows-change takes 0 or 1"; exit 1; fi
      HAVE_RC=1; RC_VALUE="$2"; shift 2 ;;
    --max-lanes) EXTRA_ARGS+=(max_lanes:="$2"); HAVE_MAXLANES=1; shift 2 ;;
    --turn-mode)
      if [[ "$2" != "bulb" && "$2" != "fishtail" && "$2" != "shuttle" ]]; then
        echo "--turn-mode takes bulb, fishtail or shuttle"; exit 1; fi
      EXTRA_ARGS+=(turn_mode:="$2"); shift 2 ;;
    --line) EXTRA_ARGS+=(line_fit:=true); shift ;;
    --coverage) EXTRA_ARGS+=(vertical_coverage:="$2"); shift 2 ;;
    --init-window) EXTRA_ARGS+=(init_window:="$2"); shift 2 ;;
    --lambdax) GAIN_ARGS+=(lambda_x:="$2"); shift 2 ;;
    --lambdat) GAIN_ARGS+=(lambda_theta:="$2"); shift 2 ;;
    --gate) GAIN_ARGS+=(heading_gate:="$2"); shift 2 ;;
    --ki) GAIN_ARGS+=(ki:="$2"); shift 2 ;;
    --ff) GAIN_ARGS+=(ff_gain:="$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

# Row-change defaults (closed loop only; the probe never drives): with 2+
# furrows available the demo turns into the next furrow at each lane end
# and covers them all, unless --rows-change 0 was passed explicitly. An
# explicit --row-change/--rows-change value always wins; --rows-change 0
# disables changing (and --max-lanes then has nothing to bound).
if [[ "$PROBE" == "0" ]]; then
  if [[ "${HAVE_RC}" == "1" ]]; then
    if [[ "${RC_VALUE}" == "1" ]]; then
      EXTRA_ARGS+=(row_change:=true)
    else
      EXTRA_ARGS+=(row_change:=false)
    fi
  else
    _WORLD_FILE=""
    if [[ "${#WORLD_ARG[@]}" -gt 0 ]]; then
      _WORLD_FILE="${WORLD_ARG[0]#world:=}"
    elif [[ "${#FIELD_ARG[@]}" -gt 0 ]]; then
      case "${FIELD_ARG[0]#field:=}" in
        circle) _STEM="farm_maize" ;;
        curve) _STEM="farm_curve" ;;
        straight) _STEM="farm_straight" ;;
        curve5) _STEM="farm_curve5" ;;
        straight5) _STEM="farm_straight5" ;;
        zigzag5) _STEM="farm_zigzag5" ;;
        *) _STEM="" ;;
      esac
      [[ -n "${_STEM:-}" ]] && _WORLD_FILE="${HERE}/src/mrsim/worlds/${_STEM}.world"
    else
      _WORLD_FILE="${HERE}/src/mrsim/worlds/farm_maize.world"
    fi
    _SIDECAR=""
    [[ -n "${_WORLD_FILE}" ]] && _SIDECAR="${_WORLD_FILE%.world}.spawn.json"
    _N_FURROWS=1
    if [[ -n "${_SIDECAR}" && -f "${_SIDECAR}" ]]; then
      _N_FURROWS=$(python3 -c "
import json, sys
try:
    print(len(json.load(open(sys.argv[1])).get('furrows', [0.0])))
except Exception:
    print(1)
" "${_SIDECAR}")
    fi
    if [[ "${_N_FURROWS}" -ge 2 ]]; then
      EXTRA_ARGS+=(row_change:=true)
      echo "-- ${_N_FURROWS} furrows available: row changing on"
      if [[ "${HAVE_MAXLANES}" == "0" ]]; then
        EXTRA_ARGS+=(max_lanes:=$((2 * _N_FURROWS - 2)))
        echo "-- covering up to $((2 * _N_FURROWS - 2)) lanes"
      fi
    fi
  fi
fi

source /opt/ros/humble/setup.bash
export MULTIROI_DIR
export GAZEBO_MODEL_PATH="${MULTIROI_DIR}/tests/agribot/agribot_gazebo/models:/usr/share/gazebo-11/models"

echo "== building mrsim workspace =="
cd "${HERE}"
colcon build --symlink-install --base-paths src 2>&1 | tail -3
source install/setup.bash

mkdir -p "${LOG_DIR}"
if pgrep -x gzserver > /dev/null; then
  echo "refusing to start: a gzserver process is already running."
  echo "It would fight this run over ROS topics; kill leftovers first:"
  echo "  pgrep -af 'gzserver|nav_node.py|farm.launch.py'   # inspect"
  echo "  pkill -x gzserver   # then relaunch"
  exit 1
fi
if [[ "$PROBE" == "1" ]]; then
  echo "== probe mode: capture frames through the detector =="
  timeout 120 ros2 launch mrsim farm_probe.launch.py \
    out_dir:="${LOG_DIR}" "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" || true
else
  echo "== closed loop: nav node drives the furrow =="
  timeout "${SIM_TIMEOUT:-600}" ros2 launch mrsim farm.launch.py \
    log_dir:="${LOG_DIR}" \
    ${SECONDS_ARG:+${SECONDS_ARG}} "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" "${LAPS_ARG[@]}" "${GAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" || true
fi

echo
echo "== results in ${LOG_DIR} =="
ls -la "${LOG_DIR}" | head
