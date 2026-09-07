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

# 2) closed-loop run: rover laps the ring (stops after 1 lap by default;
#    max_laps:=0 loops forever). Results in /tmp/mrsim_log.
./run_sim.sh --out /tmp/mrsim_log

# options: --seconds 90  (stop after N s)   --laps 2 (ring laps)
#          --x/--y/--yaw (spawn override; defaults follow the world sidecar)
```

Or launch manually:

```bash
source /opt/ros/humble/setup.bash
export MULTIROI_DIR=/abs/path/to/LinReg/MultiROI
cd tests/sim_ros2 && colcon build --symlink-install --base-paths src && source install/setup.bash
ros2 launch mrsim farm.launch.py log_dir:=/tmp/mrsim_log    # closed loop
ros2 launch mrsim farm_probe.launch.py out_dir:=/tmp/p frames:=5
```

## Outputs and metrics

Each nav frame appends a row to `<log_dir>/nav_run.csv`:

```
sim_t, odom_x, odom_y, cross_track, err_x_px, raw_th_deg, filt_th_deg,
conf, status, v, w, n_two
```

- `cross_track` = distance from the furrow centerline (goal: near 0). On a
  straight world that is `odom_y - lane_y`; on the ring world it is the
  radial error `hypot(odom - center) - R` (the nav log also prints lap
  progress; the run stops after `max_laps`).
- `err_x_px` / `filt_th_deg` / `conf` / `status` / `n_two` come straight
  from the MultiROI pipeline (`process_image()` info dict).
- Every 20th frame the drawn overlay (red nav line etc.) is saved as
  `frame_XXXXX.png` in the same log dir.

Reference result from the committed defaults — one continuous half-lap of the
ring (the `completed 0.5 lap(s)` stop fires at 180° swept):

| metric | value |
|---|---|
| path covered | ~36.7 m of constant bend (R = 12 m) |
| mean \|radial error\| | 0.362 m (inside-cutting, max 0.499 m) |
| corridor evidence | `n_two` ≥ 8 in ≥ 80% of frames per 30° sector |
| steering | `w` active throughout (−0.33…+0.33 rad/s, never saturating) |
| half-lap stop | fires cleanly at 180.0° |

Known limit: on constant curvature the P-servo slowly cuts inside (a few dm
per quarter lap — alternating bends self-cancel, so straight/S fields are
unaffected), and raw heading flicker near dropout patches can exit the ring,
so full rings are beyond it; the default `max_laps` is 0.5. Experiments so
far: heading-gain halving (no effect — the tilt state just re-deepens) and a
spline-bend feedforward (`ff_gain`, default off — exited earlier via a
flicker patch, kept as plumbing for tuning). Full-ring centering needs deeper
work (lateral integral with anti-windup, or curve-aware detection).

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
