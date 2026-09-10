#!/usr/bin/env bash
# Run the sim with a live Gazebo GUI AND interactive keyboard teleop, for use
# at the machine's real display (see tests/README.md).
# Same test as run_sim.sh but with the Gazebo window open. In teleop mode
# the nav node idles (detection + overlay keep running, no /cmd_vel); drive
# with the keyboard node in a second terminal:
#   python3 src/mrsim/mrsim/teleop_node.py
# (w/s = fwd/back, a/d = turn, space = stop, x = quit).
#
#   ./run_sim_gui.sh               # GUI + teleop (nav node idles)
#   ./run_sim_gui.sh --auto        # GUI + autonomous MultiROI driving
#   ./run_sim_gui.sh --demo        # GUI + scripted keys (no keyboard needed)
#   ./run_sim_gui.sh --auto --straight 5   # GUI + auto, 5 straight rows
#   ./run_sim_gui.sh --auto --straight 5 --spawn 2 --row-change  # drive
#     furrow 2, headland-turn into the next furrow at the lane end
#
# Launch arguments (passed through to farm.launch.py unless noted):
#   --auto         autonomous driving (default mode is teleop/manual).
#   --demo         scripted key sequence instead of keyboard (headless test);
#     --keys "w w w a d" overrides the sequence (each key held ~0.8 s).
#   --algo NAME    detection algorithm (only multiroi is wired).
#   Fields (world + spawn/lane defaults travel together):
#   --circle (default)  --curve[N]  --straight[N]  --zigzag[N]
#     N = row count, e.g. --straight 5. Bare --straight/--curved mean N=2,
#     bare --zigzag means N=5. N=2 and N=5 use committed snapshots; any
#     other count 2..10 is generated on demand into ~/.cache/crop-row-fields.
#   --field NAME   raw preset name (circle, curve, straight, curve5,
#     straight5, zigzag5, custom).
#   --x/--y/--yaw  spawn pose override (defaults follow the field sidecar).
#   --spawn N      drive the Nth furrow from the left, 1-based.
#   --row-change / --rows-change 0|1: headland-turn into the next furrow
#     at each lane end. Automatic on fields with 2+ furrows unless
#     --rows-change 0 is given.
#   --max-lanes N  lanes to cover before auto-stop (0 = until Ctrl-C;
#     defaults to a full sweep of the field).
#   --turn-mode bulb|fishtail|shuttle: headland turn style (default bulb).
#   --laps N       ring-field laps before auto-stop, 0 = loop forever.
#   --line         straight nav-line fit instead of the spline.
#   --coverage F / --init-window F / --world PATH  see run_sim.sh header.
#
# The teleop keys are typed into the teleop terminal
# (w/s = fwd/back, a/d = turn, space = stop, x = quit).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)"          # .../LinReg/MultiROI
MODE="teleop"
DEMO_KEYS=""
ALGO="multiroi"
LOG_DIR="/tmp/mrsim_gui"
LAPS=""
FIELD_ARG=()
WORLD_ARG=()
SPAWN_ARGS=()
EXTRA_ARGS=()
WORLD_FILE=""
HAVE_X=0; HAVE_Y=0; HAVE_YAW=0; HAVE_LAPS=0
HAVE_RC=0; RC_VALUE=""; HAVE_MAXLANES=0

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
    --auto) MODE="auto" ; shift ;;
    --demo) MODE="demo" ; shift ;;
    --keys) DEMO_KEYS="$2" ; shift 2 ;;
    --algo) ALGO="$2" ; shift 2 ;;
    --circle) FIELD_ARG=(field:=circle) ; shift ;;
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
    --field) FIELD_ARG=(field:="$2") ; shift 2 ;;
    --x) SPAWN_ARGS+=(robot_x:="$2") ; HAVE_X=1 ; shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2") ; HAVE_Y=1 ; shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2") ; HAVE_YAW=1 ; shift 2 ;;
    --laps) LAPS="$2" ; HAVE_LAPS=1 ; shift 2 ;;
    --spawn) SPAWN_ARGS+=(spawn_row:="$2") ; shift 2 ;;
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
    --line) EXTRA_ARGS+=(line_fit:=true) ; shift ;;
    --coverage) EXTRA_ARGS+=(vertical_coverage:="$2") ; shift 2 ;;
    --world) WORLD_ARG=(world:="$2") ; WORLD_FILE="$2" ; shift 2 ;;
    *) echo "unknown arg $1 (--auto|--demo|--keys|--algo|--circle|--curve|--straight|--zigzag|--field|--x|--y|--yaw|--laps|--spawn|--row-change|--max-lanes|--turn-mode|--line|--coverage|--world)"; exit 1 ;;
  esac
done

# A --world file with a sibling .spawn.json sidecar lends its spawn/lane
# defaults for anything not passed explicitly (so custom worlds behave
# like the committed one with no extra flags).
if [[ -n "${WORLD_FILE}" && -f "${WORLD_FILE%.world}.spawn.json" ]]; then
  while IFS="=" read -r key val; do
    case "${key}" in
      robot_x) [[ "${HAVE_X}" == "0" ]] && SPAWN_ARGS+=(robot_x:="${val}") ;;
      robot_y) [[ "${HAVE_Y}" == "0" ]] && SPAWN_ARGS+=(robot_y:="${val}") ;;
      robot_yaw) [[ "${HAVE_YAW}" == "0" ]] && SPAWN_ARGS+=(robot_yaw:="${val}") ;;
      circle_r) EXTRA_ARGS+=(circle_r:="${val}") ;;
      max_laps_default) [[ "${HAVE_LAPS}" == "0" ]] && LAPS="${val}" ;;
    esac
  done < <(WORLD_SIDECAR="${WORLD_FILE%.world}.spawn.json" python3 -c "
import json, os
d = json.load(open(os.environ['WORLD_SIDECAR']))
for k in ('robot_x', 'robot_y', 'robot_yaw', 'circle_r', 'max_laps_default'):
    if d.get(k) is not None:
        print(f'{k}={d[k]}')
")
fi
[[ -z "${LAPS}" ]] && LAPS="0.5"

# Row-change defaults: with 2+ furrows available the demo turns into the
# next furrow at each lane end and covers them all, unless --rows-change 0
# was passed explicitly. An explicit --row-change/--rows-change value
# always wins; --rows-change 0 disables changing (and --max-lanes then has
# nothing to bound). Teleop/demo modes never auto-enable (nothing drives
# closed loop there).
if [[ "${MODE}" == "auto" ]]; then
  if [[ "${HAVE_RC}" == "1" ]]; then
    if [[ "${RC_VALUE}" == "1" ]]; then
      EXTRA_ARGS+=(row_change:=true)
    else
      EXTRA_ARGS+=(row_change:=false)
    fi
  else
    _WORLD_FILE="${WORLD_FILE}"
    if [[ -z "${_WORLD_FILE}" && "${#FIELD_ARG[@]}" -gt 0 ]]; then
      case "${FIELD_ARG[0]#field:=}" in
        circle) _STEM="farm_maize" ;;
        curve) _STEM="farm_curve" ;;
        straight) _STEM="farm_straight" ;;
        curve5) _STEM="farm_curve5" ;;
        straight5) _STEM="farm_straight5" ;;
        zigzag5) _STEM="farm_zigzag5" ;;
        *) _STEM="" ;;
      esac
      [[ -n "${_STEM:-}" ]] && \
        _WORLD_FILE="${HERE}/src/mrsim/worlds/${_STEM}.world"
    fi
    [[ -z "${_WORLD_FILE}" ]] && \
      _WORLD_FILE="${HERE}/src/mrsim/worlds/farm_maize.world"
    _SIDECAR="${_WORLD_FILE%.world}.spawn.json"
    _N_FURROWS=1
    if [[ -f "${_SIDECAR}" ]]; then
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

if pgrep -x gzserver > /dev/null; then
  echo "refusing to start: a gzserver process is already running."
  echo "Kill leftovers first (pkill -x gzserver), then relaunch."
  exit 1
fi

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
ros2 launch mrsim farm.launch.py "${ARGS[@]}" "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" "${EXTRA_ARGS[@]}"
