#!/usr/bin/env bash
# Run the sim with a live Gazebo GUI AND interactive keyboard teleop, for use
# at the machine's real display (see README.md).# Same test as run_sim.sh but with the Gazebo window open. In teleop mode
# the nav node idles (detection + overlay keep running, no /cmd_vel) and the
# keyboard teleop node starts automatically with the launch - type into this
# terminal to drive:
# (w/s = fwd/back, a/d = turn, space = stop, x = quit; 'r' respawns the rover
#  at the initial spawn pose, no relaunch needed).
#
#   ./run_sim_gui.sh               # GUI + teleop (nav node idles)
#   ./run_sim_gui.sh --auto        # GUI + autonomous FFT driving
#   ./run_sim_gui.sh --demo        # GUI + scripted keys (no keyboard needed)
#   ./run_sim_gui.sh --auto --straight 5   # GUI + auto, 5 straight rows
#
# Launch arguments (passed through to farm.launch.py unless noted):
#   --auto         autonomous driving (default mode is teleop/manual).
#   --demo         scripted key sequence instead of keyboard (headless test);
#     --keys "w w w a d" overrides the sequence (each key held ~0.8 s).
#   Fields (world + spawn/lane defaults travel together):
#   --circle (default)  --curve[N]  --straight[N]  --zigzag[N]
#     N = row count, e.g. --straight 5. Bare --straight/--curved mean N=2,
#     bare --zigzag means N=5. N=2 and N=5 use committed snapshots; any
#     other count 2..10 is generated on demand into ~/.cache/crop-row-fields.
#   --field NAME   raw preset name (circle, curve, straight, curve5,
#     straight5, zigzag5, custom).
#   --x/--y/--yaw  spawn pose override (defaults follow the field sidecar).
#   --laps N       ring-field laps before auto-stop, 0 = loop forever.
#   Reset in any mode (no relaunch): press r in the teleop terminal, or run
#     ros2 topic pub --once /reset_rover std_msgs/msg/Empty "{}"
#   --world PATH   custom world file (sibling .spawn.json seeds defaults).
#   --tof / --tof-min M / --tof-gain G / --tof-max-w W / --tof-v V:
#     crop-safety guard (side + angled-front ToF rangers override vision steering when
#     closer than the min clearance; high priority).
#
# The teleop keys are typed into this launch terminal
# (w/s = fwd/back, a/d = turn, space = stop, x = quit; 'r' = respawn at start).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_DIR="$(cd "${HERE}/.." && pwd)"                  # .../FFT
MULTIROI_DIR="$(cd "${HERE}/../.." && pwd)/LinReg/MultiROI"  # shared servo
MODE="teleop"
DEMO_KEYS=""
LOG_DIR="/tmp/fftsim_gui"
LAPS="0.5"
FIELD_ARG=()
SPAWN_ARGS=()
WORLD_ARG=()
TOF_ARGS=()

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
    --auto) MODE="auto" ; shift ;;
    --demo) MODE="demo" ; shift ;;
    --keys) DEMO_KEYS="$2" ; shift 2 ;;
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
    --x) SPAWN_ARGS+=(robot_x:="$2") ; shift 2 ;;
    --y) SPAWN_ARGS+=(robot_y:="$2") ; shift 2 ;;
    --yaw) SPAWN_ARGS+=(robot_yaw:="$2") ; shift 2 ;;
    --laps) LAPS="$2" ; shift 2 ;;
    --world) WORLD_ARG=(world:="$2") ; shift 2 ;;
    --tof) TOF_ARGS+=(tof:=true) ; shift ;;
    --tof-min) TOF_ARGS+=(tof_min:="$2") ; shift 2 ;;
    --tof-gain) TOF_ARGS+=(tof_gain:="$2") ; shift 2 ;;
    --tof-max-w) TOF_ARGS+=(tof_max_w:="$2") ; shift 2 ;;
    --tof-v) TOF_ARGS+=(tof_v:="$2") ; shift 2 ;;
    *) echo "unknown arg $1 (--auto|--demo|--keys|--circle|--curve|--straight|--zigzag|--field|--x|--y|--yaw|--laps|--world|--tof|--tof-min|--tof-gain|--tof-max-w|--tof-v)"; exit 1 ;;
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
[[ " ${TOF_ARGS[*]} " == *"tof:=true"* ]] && echo "-- ToF crop-safety guard ON (${TOF_ARGS[*]})"
ros2 launch fftsim farm.launch.py "${ARGS[@]}" "${FIELD_ARG[@]}" "${WORLD_ARG[@]}" "${SPAWN_ARGS[@]}" "${TOF_ARGS[@]}"
