# ExG crop-row navigation in Gazebo (ROS 2) — test rig

Same rig shape as `LinReg/MultiROI/tests/sim_ros2`. Two drivers are
selectable with `nav:` (`--nav exg|vendor`); **`vendor` is the default**
until the Python path finishes validation:

```
                 ┌ exgsim/nav_node.py   (nav:=exg)
                 │   ExG/visual-crop-row-navigation_ros2/results/:
                 │   base-anchored column-aware window → /cmd_vel
simulated camera┤
   ↓ bridge      └ agribot_vs_node      (nav:=vendor, default)
                 ↘ monitor → /exgsim/overlay + nav_run.csv
```

* `nav:=exg` runs the repo's own Python ExG navigation
  (`ExG/visual-crop-row-navigation_ros2/results/`, the same pipeline the
  offline debug and video tools use) closed-loop in Gazebo. Its window is
  pinned to the base of the frame and latches crop columns - see
  "Base-anchored column-aware window" below.
* `nav:=vendor` runs the upstream C++ `agribot_vs_node` unchanged.

Package `exgsim` (Gazebo Classic 11, ROS 2 Humble). The C++ package is built
once into `exg_ws/` by the run scripts.

## Quick start

```bash
cd ExG/sim_ros2
./run_sim.sh --straight 5 --out /tmp/exg_s5 --seconds 120   # Python ExG nav
./run_sim.sh --straight 5 --nav vendor --out /tmp/exg_v --seconds 90
./run_sim.sh --probe --out /tmp/exg_probe                  # C++ detection check
./run_sim_gui.sh --auto                     # GUI + autonomous driving
```

### Manual driving + respawn (`r`)

`run_sim_gui.sh` without `--auto` starts in teleop mode: the C++ stack
idles and the keyboard node (`teleop_node.py`) starts automatically with the
launch and reads the launch terminal itself, so just type into that
terminal: w/s/a/d drive, space stops, x quits, and `r` respawns the rover at
the initial spawn pose (delete_entity + spawn_entity at the MRSIM_SPAWN
pose; handy after driving into the crops, no relaunch needed). Manual
driving routes through the ToF guard when `--tof` is on. The same reset is
available in every mode via
`ros2 topic pub --once /reset_rover std_msgs/msg/Empty "{}"`.

Fields work like the MultiROI rig: `--circle` (default), `--curve[N]`,
`--straight[N]`, `--zigzag[N]` (bare flag: N=2, N=5 for zigzag; other
counts generate on demand into a shared fields cache). Explicit
`--x/--y/--yaw/--laps` always win; `--world` selects a custom file.

Spawn/termination defaults follow the field sidecars (e.g.
`src/exgsim/worlds/farm_maize.spawn.json`, ring R = 12 m). With the default
`nav:=exg` the rover starts **on a crop row** (the ExG algorithm rides above
the rows), not in the furrow; `--nav vendor` keeps the historical furrow
start, and the ring world always keeps its radius/furrow start.

### Base-anchored column-aware window (`nav:=exg`)

The Python ExG driver wraps the same window state machine as the offline
tools (`results/exg_window.py`, used by `run_vcrn_debug.py` and
`run_exg_video.py`):

1. **Acquire** - the window is pinned to the base of the frame (its bottom
   edge on the chassis-forward reference, the MultiROI blue-star row) and
   latches the crop column nearest the image centre, using the bottom-band
   vertical projection peaks; the window width comes from the median
   inter-row gap.
2. **Align** - steering while driving forward slides that latched column to
   the bottom-centre reference, i.e. the chassis turns onto the column.
3. **Lock** - after `lock_frames` aligned frames the window locks at
   bottom-centre (the permanent blue-marker behaviour) and the fitted row is
   evaluated at the base reference (`line_base_error`), not the image
   middle, because the vehicle rides above the rows.
4. **Re-latch** - when the column leaves the base band (the end of the crop
   column / headland transition) or the window runs empty for
   `lost_frames`, the tracker goes back to acquisition for the next column.
   The node creeps forward for `search_seconds` to find it, then stops - the
   sim’s end-of-lane trigger.

Tuning lives in `src/exgsim/params/exgsim_run.yaml` (`base_margin`,
`latch_tol_px`, `lock_frames`, `lost_frames`, `base_kx`/`base_kth`,
`base_w_max`), a sim-tuned copy of the vendor params. Per-frame window state
is written to `exg_nav.csv` and overdrawn frames to `exg_*.png` in the log
dir; the drawn overlay is published on `/exgsim/nav_overlay`.

### Crop-safety ToF guard

Same rig-wide guard as MultiROI (see its tests README): four rangers
(`/tof/left`, `/tof/right`, `/tof/front_left`, `/tof/front_right`, the
front pair yawed +/-35 deg for bend lookahead) feed `tof_guard.py`. With
`tof:=true` the active driver (C++ stack, Python `nav_node.py`, or the
teleop node) publishes to `/cmd_vel_raw` and the guard owns the wheels,
republishing the safety-overridden command on `/cmd_vel`. Enable with `./run_sim.sh --tof` plus
`--tof-min`, `--tof-gain`, `--tof-max-w`, `--tof-v` tuning; per-frame
telemetry goes to `tof_guard.csv`. In any mode,
`ros2 topic pub --once /reset_rover std_msgs/msg/Empty "{}"` (or `r` in a
keyboard terminal) teleports the rover back to the start pose.

## How it fits together

* `bridge.py`: `/camera/image_raw` (BestEffort) → `/front/rgb/image_raw`
  (Reliable) and `/odom` → `/odometry/raw`, the topic names the C++ node
  hardcodes. The Python `nav_node.py` subscribes to the raw Gazebo topics
  directly, so it does not need the bridge; the bridge stays for `nav:=vendor`
  and the monitor.
* `nav_node.py` (`nav:=exg`): the `results/` pipeline drives
  `/cmd_vel`, publishes `/vs_msg` for the monitor and `/exgsim/nav_overlay`.
  It locates the pipeline via `$EXG_DIR` (set by the run scripts) or the repo
  layout. In `mode:=teleop/demo` it logs detection but publishes no `/cmd_vel`.
* `farm.launch.py`: gzserver + spawn + bridge + one of the two drivers +
  monitor. `mode:=teleop/demo` sets `mask_tune` so the C++ stack idles and
  publishes nothing; the teleop node then starts with the launch and reads
  its terminal (no separate `ros2 run exg_teleop`).
* `monitor.py`: overlay on `/exgsim/overlay` plus `nav_run.csv` in the
  shared schema (`cross_track` honors lane/circle env). Probe mode captures
  raw frames plus per-frame `/vs_msg` errors.
* The C++ node calls `imshow` unconditionally, so the launch puts it under
  `xvfb-run` unless `$DISPLAY` points at a live X server.

## Sim tuning (all in `src/exgsim/params/exgsim_run.yaml`, vendor file untouched)

* HSV Value floor 100 → 70: Gazebo greens sit at V~75-100 and the vendor
  window starved the mask to ~1%.
* Camera geometry to the rover: `tz` 0.7 → 1.36 m, `ty` 0.6 → 0.28 m,
  front tilt `rho_f` -60 → -66 deg.
* `max_row_num` 20 → 1000000: the node counts loop iterations, not rows,
  so 20 stops all processing after ~1 s.

## Reference results

`nav:=exg` (Python pipeline, spawn on a crop row):

* Straight 5-row field, 120 s: 15.9 m traversed (`-8.0 → +7.9`), mean
  |cross-track| 0.063 m, max 0.117 m, 5 re-latches (2 transient mid-lane,
  the rest at the lane end), clean `max_seconds` stop.
* S-bend (`farm_curve.world`): full lane traversed (`-8.0 → +8.4`),
  ~65% of frames `locked`, re-latches on the bends and clusters at the
  headland, then the end-of-lane stop (`column lost`). Bend-following is
  looser than the straight case (mean |cross-track| 0.42 m) - the servo is
  deliberately conservative (narrow base window, capped `base_w_max`) so it
  re-latches rather than cutting the bend.

`nav:=vendor` (upstream C++, furrow spawn):

* S-bend world: 14 m traversed (`-8 → +6.2`), mean |cross-track| 0.25 m,
  clean `max_seconds` stop.
* Ring world: detects (points flow, errors nonzero) but the vendor tuning
  dives inside and loses the lane; rings need a tuning pass. Probe with
  `publish_cmd_vel` disabled for safe checks.
