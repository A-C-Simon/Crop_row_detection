# ExG crop-row navigation in Gazebo (ROS 2) — test rig

Same rig shape as `LinReg/MultiROI/tests/sim_ros2`, but the driver is the
vendor C++ stack (`ExG/visual-crop-row-navigation_ros2`: HSV detection +
visual servoing in `agribot_vs_node`) instead of a Python pipeline:

```
simulated camera → bridge → agribot_vs_node → /cmd_vel
                  ↘ monitor → /exgsim/overlay + nav_run.csv
```

Package `exgsim` (Gazebo Classic 11, ROS 2 Humble). The C++ package is built
once into `exg_ws/` by the run scripts.

## Quick start

```bash
cd ExG/sim_ros2
./run_sim.sh --probe --out /tmp/exg_probe
./run_sim.sh --straight 5 --out /tmp/exg_s5 --seconds 90
./run_sim_gui.sh --auto                     # GUI + autonomous driving
```

Fields work like the MultiROI rig: `--circle` (default), `--curve[N]`,
`--straight[N]`, `--zigzag[N]` (bare flag: N=2, N=5 for zigzag; other
counts generate on demand into a shared fields cache). Explicit
`--x/--y/--yaw/--laps` always win; `--world` selects a custom file.

Spawn/termination defaults follow the field sidecars (e.g.
`src/exgsim/worlds/farm_maize.spawn.json`, ring R = 12 m).

## How it fits together

* `bridge.py`: `/camera/image_raw` (BestEffort) → `/front/rgb/image_raw`
  (Reliable) and `/odom` → `/odometry/raw`, the topic names the C++ node
  hardcodes. IMU/AMCL inputs stay unconnected; the node tolerates them.
* `farm.launch.py`: gzserver + spawn + bridge + `agribot_vs_node` (with
  `src/exgsim/params/exgsim_run.yaml`, a sim-tuned copy of the vendor
  params) + monitor. `mode:=teleop/demo` sets `mask_tune` so the C++
  stack idles and publishes nothing.
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

* S-bend world (`farm_curve.world`, straight spawn): 14 m traversed
  (`-8 → +6.2`), mean |cross-track| 0.25 m, clean `max_seconds` stop.
* Ring world: detects (points flow, errors nonzero) but the vendor tuning
  dives inside and loses the lane; rings need a tuning pass. Probe with
  `publish_cmd_vel` disabled for safe checks.
