# MultiROI in Gazebo (ROS 2) — test rig

This folder lets you test the MultiROI crop-row detection + navigation stack
(`LinReg/MultiROI`) in a simulated field, running the **same** pipeline that
`run_mr_navigation.py` runs on real video:

```
simulated camera → MultiROI detector → temporal filter → mr_vs → /cmd_vel
                      (LinReg/MultiROI sources, unchanged)
```

The simulator is **Gazebo Classic 11 on ROS 2 Humble** (headless), with a
4-wheel diff-drive rover carrying a downward-pitched RGB camera.

## Layout

```
tests/
├── agribot/              PRBonn/agribot repo (fetched as reference; our rig
│                         reuses its big_plant/small_plant STL maize models)
└── sim_ros2/             self-contained colcon workspace (the actual rig)
    ├── run_sim.sh        one-shot: build + probe or closed-loop run (headless)
    ├── run_sim_gui.sh    one-command: Gazebo GUI + keyboard teleop/auto/demo
    ├── src/mrsim/
    │   ├── scripts/gen_farm_world.py   generates the field .world from the
    │   │                                agribot plant STLs (deterministic seed)
    │   │                                incl. the gzclient GUI camera pose
    │   ├── urdf/rover.urdf             rover: 4-wheel diff drive + RGB camera
    │   ├── launch/farm.launch.py       closed loop: gzserver(world) + spawn
    │   │                                rover + nav node; shuts down on finish
    │   ├── launch/farm_probe.launch.py camera calibration probe (no driving)
    │   ├── mrsim/nav_node.py           camera→MultiROI→cmd_vel node + CSV log
    │   ├── mrsim/pipeline.py           algorithm-selection seam (multiroi now;
    │   │                                FFT/CAROLIF/ExG hook points documented)
    │   ├── mrsim/teleop_node.py        keyboard teleop (/cmd_vel) + scripted
    │   │                                demo keys for headless checks
    │   ├── worlds/farm_maize.world     generated field (committed snapshot)
    │   └── package.xml, setup.py
    └── tools/             small helpers (grab_img.py, check_frame.py)
```

Why not run `PRBonn/agribot` as-is: it is a ROS 1 (catkin/Melodic) repo and
this machine runs ROS 2 Humble, so the farm worlds and plant models were
ported to a ROS 2 rig. Only its `agribot_gazebo` assets are vendored here
(the rest of the reference tree stays local); the rig reuses the
big_plant/small_plant STL maize models.

## What is simulated

- **Field**: two concentric circular maize rows by default (ring furrow of
  radius 12 m around the origin, 1.1 m row spacing, plants every 0.2 m,
  ~0.45 m canopy, ~0.67 m tall, small gaussian planting noise), so the rover
  must hold a constant bend — straight-line driving scores nothing. The rover
  spawns on the ring at `(12, 0)` with a look-into-the-turn yaw and laps
  counter-clockwise.
  Regenerate with different geometry:
  `python3 src/mrsim/scripts/gen_farm_world.py --shape straight ...` (rows
  along +x with an optional S-bend: `--curve-amp 1 --curve-period 18`),
  `--shape circle --circle-r 12 --plant-spacing 0.2 ...`
  (`python3 src/mrsim/scripts/gen_farm_world.py -h` for all options).
  The generator writes a `farm_maize.spawn.json` sidecar next to the world;
  launch files use it for spawn/lane defaults, so plain commands keep working
  after regenerating.
- **Rover**: ~0.6 m track, 4 driven wheels
  (`libgazebo_ros_diff_drive`, `/cmd_vel` → `/odom` + tf), front RGB camera
  (640x480, ~65° HFOV) mounted ~1.4 m high, pitched ~24° down so both
  furrow walls stay in view for the detector's full ROI.
- **Nav node** (`multiroi_nav`): subscribes `/camera/image_raw`, runs the
  exact MultiROI `process_image()` from `run_mr_navigation.py` every frame
  (detector + temporal filter + mr_vs, with spike rejection), publishes
  `/cmd_vel`, and logs `nav_run.csv` + overlay PNGs to `log_dir`.

> Camera mounting note (URDF): Gazebo renders a camera sensor along the
> **+X axis of the link the sensor block is attached to** (verified
> empirically with a colored-box world). So the sensor lives on
> `camera_link` (whose joint holds the downward pitch), and `camera_optical`
> is kept purely as the TF frame published in the image header.

## Quick start

Prereqs (already present on this machine): ROS 2 Humble, Gazebo Classic 11,
`colcon`, `gazebo_ros` (spawn_entity, diff_drive, camera plugins). The nav
node needs the agribot models dir on `GAZEBO_MODEL_PATH`, which
`run_sim.sh`/launch files set up from `MULTIROI_DIR`.

```bash
cd LinReg/MultiROI/tests/sim_ros2
export MULTIROI_DIR=$(pwd)/../..      # .../LinReg/MultiROI (or export beforehand)

# 1) camera calibration probe: spawn world+rover, grab 5 frames through the
#    detector, print n_two / median row width / heading; PNGs in --out.
./run_sim.sh --probe --out /tmp/mrsim_probe

# 2) closed-loop run: rover laps the ring (stops after the sidecar lap
#    default, 0.5; max_laps:=0 loops forever). Results in /tmp/mrsim_log.
./run_sim.sh --out /tmp/mrsim_log

# fields (world + spawn/lane defaults travel together):
#   --circle (default)  --curve[N]  --straight[N]  --zigzag[N]
# examples: --straight 5  --straight 2  --curved 8 (bare flag: N=2, N=5
# for zigzag). Snapshots cover N=2 and N=5; other counts generate on
# demand into ~/.cache/crop-row-fields (2..10 rows, shared by all rigs).
# options: --seconds 90  (stop after N s)   --laps 2 (ring laps)
#          --x/--y/--yaw (spawn override; defaults follow the field sidecar)
#          --line (straight nav fit)  --coverage 0.6  --init-window 0.4
```

Or launch manually (field presets work the same way):

```bash
source /opt/ros/humble/setup.bash
export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI
cd tests/sim_ros2 && colcon build --symlink-install --base-paths src && source install/setup.bash
ros2 launch mrsim farm.launch.py log_dir:=/tmp/mrsim_log    # closed loop
ros2 launch mrsim farm.launch.py field:=zigzag5 log_dir:=/tmp/z
ros2 launch mrsim farm_probe.launch.py out_dir:=/tmp/p frames:=5
ros2 launch mrsim farm_probe.launch.py field:=curve5 out_dir:=/tmp/p
```

Field notes: the 2-row fields track (ring half-lap and S-bend validated).
Straight 5-row tracks its lane once spawned inside a real furrow (mean
0.01 m). Two related gotchas, both fixed: the generator used to shift rows
by the lane offset, parking the spawn on top of a row; and identical
neighboring furrows can still pull the fit, so `--spawn N` (1-based from
the left) picks the driven furrow explicitly.

Row changing: on fields with 2+ furrows the demo turns into the next
furrow at each lane end and covers them all, with no extra flags
(`max_lanes` defaults to a full sweep). `--rows-change 0|1` forces it
off/on (`--row-change` means 1); with 0, `--max-lanes` does nothing.
Turns are odometry-scripted by default (bulb: push, spin, slide, spin
with per-phase timeouts that stop safe); `--turn-mode fishtail` backs
into the next furrow instead (push, forward arc away, rear-camera-guided
reverse-in, no spinning; the rear steers when it locks, an odometry crab
finishes otherwise). Detection keeps drawing throughout.
Measured on straight 3-row auto sweep: lane 1 out, turn, lane 0 back
(mean 0.053 m), clean stop. Fishtail on straight 4-row middle furrow:
legs 0.029 m and 0.057 m, bulb parity, clean `covered 2 lane(s)` stop.
Straight fields only; needs 2+ furrows.
On bending 5-row fields (curve5, zigzag5) the corridor fit can still walk
or lag: identical competing furrows plus lookahead hold. Lane anchoring
there is open work.

## Outputs and metrics

Each nav frame appends a row to `<log_dir>/nav_run.csv`:

```
sim_t, odom_x, odom_y, cross_track, err_x_px, raw_th_deg, filt_th_deg,
conf, status, v, w, n_two, ff
```

- `cross_track` = distance from the furrow centerline (goal: near 0). On a
  straight world that is `odom_y - lane_y`; on the ring world it is the
  radial error `hypot(odom - center) - R` (the nav log also prints lap
  progress; the run stops after `max_laps`).
- `err_x_px` / `filt_th_deg` / `conf` / `status` / `n_two` come straight
  from the MultiROI pipeline (`process_image()` info dict).
- Every 20th frame the drawn overlay (red nav line etc.) is saved as
  `frame_XXXXX.png` in the same log dir.

Reference result from the committed defaults: one continuous half-lap of the
ring (the `completed 0.5 lap(s)` stop fires at 180° swept), with lateral
priority (heading gate) and a shortened 0.6 lookahead on the ring:

| metric | value |
|---|---|
| path covered | ~36.7 m of constant bend (R = 12 m) |
| mean \|radial error\| | 0.06 m (max 0.15 m, centered: no wall contact) |
| corridor evidence | `n_two` 9–10 throughout, no dropouts |
| steering | `w` active throughout, never saturating |
| half-lap stop | fires cleanly at 180.0° |

Climbing the inner wall on bends was the steady inside-cut of the
P-servo. Three fixes stack to remove it: lateral priority (heading gate
0.1, keeps working back to center first), a shorter 0.6 lookahead on the
ring (the fitted line bends less across a shorter window), and a gated
lateral integral (`ki` 0.3, integrates only when confident, clamped and
decaying) that winds away any remaining steady offset. Straights are
unaffected (mean cross-track 0.02 m with the integral on).

Earlier baselines for comparison: no gate and full 0.75 lookahead cut
~0.5 m inside by mid-lap. Shortening the lookahead trims the cut because
the fitted line bends less across a shorter window; the overlay `err_x`
still reads a few dozen px on bends (tilt projection, not displacement).

Known limit: raw heading flicker near dropout patches can still exit the
ring, so full rings are beyond the default half-lap; the default
`max_laps` is 0.5. Tuning history: heading-gain halving (no effect, the
tilt state just re-deepens), spline-bend feedforward (`ff_gain`, default
off, exited earlier via a flicker patch, kept as plumbing), shorter
lookahead (kept: 0.6 on the ring via the world sidecar) and lateral
priority (kept: heading gate 0.1). Launch args include `line_fit`
(straight nav line, no spline), `vertical_coverage`, `heading_gate`,
`lambda_x/theta` and `ff_gain`; `run_sim_gui.sh` takes `--line`,
`--world`, `--x/--y/--yaw`, `--laps`.

## Live GUI + keyboard teleop (at the machine's display)

Run this on the machine whose screen you want to watch (the sim rig opens a
Gazebo window there). One command builds and launches everything:

```bash
cd LinReg/MultiROI/tests/sim_ros2
./run_sim_gui.sh            # GUI + keyboard teleop (nav node idles, keeps drawing)
./run_sim_gui.sh --auto     # GUI + MultiROI drives the furrow autonomously
./run_sim_gui.sh --demo     # GUI + scripted keys (no keyboard needed)
# extras: --keys "w w w a d"   (demo sequence)   --algo multiroi   (see below)
```

The Gazebo window opens with the camera already framing the rover start down
between the two crop rows (world `<gui><camera>` pose in `farm_maize.world`
— regenerate with `gen_farm_world.py` to move it). Orbit / pan / zoom in the
view with the mouse (left drag orbit, middle drag pan, scroll zoom, right
click for the context menu).

**Teleop keys** (type into the terminal that runs the sim when in teleop
mode):

| key | action |
|---|---|
| `w` / `s` | forward / reverse |
| `a` / `d` | turn left / right |
| `q` / `e` | turn aliases (left/right) |
| `space` | stop |
| `x` | quit teleop |

Speeds ramp smoothly toward `v_max=0.5 m/s` / `w_max=1.5 rad/s`; release the
key to stop. While the keyboard drives, the nav node keeps running its
detection + overlay logging but publishes nothing (`mode!=auto` → it logs
`nav node IDLE`), so `/cmd_vel` stays exclusively yours. In `--auto` mode the
nav node drives and the keyboard is ignored.

Headless automation of the same thing (no display needed):

```bash
# scripted drive for CI/verification: plays 0.8 s per key then shuts down
MRSIM_SIM_MODE=demo ros2 launch mrsim farm.launch.py \
    demo_keys:="w w w w a a d d s s" log_dir:=/tmp/mrsim_demo gui:=false
```

## Choosing the detection algorithm

Every nav node run goes through one pipeline selected by `algorithm:=`
(default and only wired option: **`multiroi`**):

```bash
ros2 launch mrsim farm.launch.py algorithm:=multiroi gui:=true
./run_sim_gui.sh --algo multiroi
```

The seam is `src/mrsim/mrsim/pipeline.py`: it builds a pipeline object whose
`process(bgr)` returns the `{v, w, info, overlay}` dict each frame. The other
detectors in this repo (`FFT/` DFT-row detector, `ClusterAlg/` CAROLIF,
`ExG/visual-crop-row-navigation_ros2`) are **not** wired yet — each has a
different output contract (BEV rectification, clustered splines, its own
C++ ROS2 control node), so wiring one means adding a builder + adapter in
`pipeline.py` (hook points are documented there).

## Running headless

`run_sim.sh` and the launch files default to `gui:=false` (gzserver only) so
everything runs without a display. To watch it on a screen, pass
`gui:=true` (or use `run_sim_gui.sh` above):

```bash
ros2 launch mrsim farm.launch.py gui:=true log_dir:=/tmp/mrsim_log
```

(If you have no real display, an X server like Xvfb is needed for the GUI.)

## Debugging / tuning checklist

- **Rows not detected (`n_two=0`)?** Check the camera view first with the
  probe PNGs: both green furrow walls should be visible in most of the lower
  ~75 % of the frame, with the soil corridor between them. Camera height /
  pitch and row spacing/canopy width are the knobs (URDF camera joint rpy,
  `gen_farm_world.py` args).
- **Rover wanders / loses the lane**: correlate with `nav_run.csv` — check
  `n_two`/`conf` dropping and `err_x_px` swings; also see the temporal
  filter spike-rejection flags (`--spike-confirm-frames` etc. in
  `run_mr_navigation.py`).
- **Rover doesn't move**: the diff-drive plugin needs
  `<num_wheel_pairs>2</num_wheel_pairs>` when 4 wheels are listed (ROS 2
  `gazebo_ros_diff_drive` requirement); check gzserver logs for
  "Inconsistent number of joints".
- **Clean rebuild**: `rm -rf build install log && ./run_sim.sh --probe`
