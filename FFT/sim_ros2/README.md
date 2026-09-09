# FFT crop-row detection in Gazebo (ROS 2) — test rig

Same rig shape as `LinReg/MultiROI/tests/sim_ros2`, but the perception is
Gai et al.'s DFT detector (`FFT/dft_crop_row_detector.py`) behind the shared
visual servo (`LinReg/MultiROI/mr_vs.py`), so algorithms compare behind
identical control.

```
simulated camera → DFT detector → EMA smooth → shared servo → /cmd_vel
```

Package `fftsim` (Gazebo Classic 11, ROS 2 Humble). Needs the agribot maize
models via `GAZEBO_MODEL_PATH` (set by the run scripts from the MultiROI
tree) and `MULTIROI_DIR` for the shared servo.

## Quick start

```bash
cd FFT/sim_ros2
./run_sim.sh --probe --out /tmp/fft_probe   # 5 frames: n_rows/ey/theta/prominence
./run_sim.sh --out /tmp/fft_log             # closed loop, 0.5-lap default
./run_sim.sh --straight 5 --out /tmp/fft_s5 # numbered rows: --straight/--curved/--zigzag
./run_sim_gui.sh --auto                     # GUI + autonomous driving
```

Fields work like the MultiROI rig: `--circle` (default), `--curve[N]`,
`--straight[N]`, `--zigzag[N]` (bare flag: N=2, N=5 for zigzag; other
counts generate on demand into a shared fields cache). Explicit
`--x/--y/--yaw/--laps` always win; `--world` selects a custom file.

Spawn/termination defaults follow the field sidecars (e.g.
`src/fftsim/worlds/farm_maize.spawn.json`).
Note the ring spawn yaw is a look-into-the-turn lead (1.8308), same as the
MultiROI rig. Probes print `n_rows ey theta prom status`.

## Pipeline notes (`src/fftsim/fftsim/pipeline.py`)

* Fixed BEV geometry for the sim camera: pitch 66 deg from nadir, 1.4 m
  high, 51 deg vertical FOV, 5 m range (env overrides `FFT_PITCH_DEG`,
  `FFT_HEIGHT_M`, `FFT_FOV_Y_DEG`, `FFT_YAW_DEG`, `FFT_RANGE_M`). The short
  range is deliberate: a 10 m window lets S-bends smear the spectrum
  (spawn read −0.79 m at prom 138 with full confidence), while 5 m sees
  straighter rows (−0.22 m, prom 85) and runs ~4x faster.
* Accepts a frame when `n_rows >= 2` and prominence >= 10, else holds the
  last detection (status `HELD`). Lateral error in meters × 300 px/m feeds
  the shared servo; confidence scales forward speed.
* **Row affinity** (`src/fftsim/fftsim/row_affinity.py`, on by default,
  `FFT_AFFINITY=0` disables): the DFT detector fits ONE plane wave to the
  whole ROI, i.e. straight parallel rows. On curved / S-shaped plantings
  that global fit averages the corridor bend away and the rover drifts off
  course. After each accepted detection the tracker re-measures the two
  row crests flanking the reference directly in the windowed BEV ROI (scan
  the predicted wave positions, argmax + sub-pixel peak per scanline,
  quadratic `x(y)` fit per flank), and ey/eth are taken from that local
  corridor instead of the global fit. Falls back to the global ey/eth on
  any rejection (ragged crest fit, flank disagreement > 12 deg, local
  heading > 35 deg away from the global fit, corridor leaving the ROI).
  Diagnostics: `info['affinity']`, `info['aff_bend']` (corridor bend across
  the visible band, radians).
* **Curvature feedforward** (`ff_gain` launch arg, env `MRSIM_FF_GAIN`,
  default 0 = off): adds `-gain * vf * bend` to the servo command so the
  P-terms do not hold a steady-state offset on constant-curvature paths
  (same convention as the MultiROI lookahead FF). Suggested 0.5 for the
  ring world once affinity is on.
* BEV reference trim (`trim` launch arg, per-world `trim_default` in the
  spawn sidecar, currently 0.0): subtracts a static bias from raw ey.
  Needed at 10 m range (0.17 m bias); the 5 m window is near-unbiased so
  the default stays 0.
* Overlay (`/fftsim/overlay`): back-projected row borders/centerline from
  `corridor_in_image` plus `ey/eth/rows/prominence` text. CSV schema matches
  the MultiROI rig with `cross_track` in meters of lateral deviation.
* Runs ~75 ms/frame (~7-13 Hz); fast enough for the 0.2 m/s rover.

## Reference results (5 m range, trim 0.0)

Ring (R = 12 m): full half-lap, clean stop at −179.8 deg. Reported `err_x`
mean 21 px, max 67 px; true radial error mean 0.11 m, max 0.20 m; `n_rows`
3+, no dropouts. A tangent spawn also tracks stably here; the yaw lead is
shared with the MultiROI rig for consistent spawn geometry.

S-bend (`farm_curve.world`, straight spawn): 13 m traversed (`−8 → +5`),
`|err_x|` mean 13 px, max 43 px, no dropouts, clean `max_seconds` stop.
Plus safety nets that earned their keep: the straddle gate rejects
wrong-furrow locks (same-side row pairs), and the rover stands still
(`v = w = 0`) after ~30 consecutively held frames instead of driving
blind into the crops.

## Verification (offline, no Gazebo needed)

`src/fftsim/fftsim/test_row_affinity.py` builds synthetic BEV fields with
known geometry and compares the global DFT corridor against the affinity
corridor at the reference point (truth from the planted crest positions):

```
straight   gt −6.0px   global −6.0px   affinity −6.0px   (identical)
arc R=900  gt −0.1px   global −13.0px  affinity −0.2px   (global drifts 13px)
S curve    gt +12.6px  global +14.4px  affinity +11.3px
```

Run it with `python3 test_row_affinity.py` from `src/fftsim/fftsim/`.

## Known limitations (affinity)

* Only near-vertical rows (|fx| >= |fy|) are re-measured locally; anything
  else falls back to the global fit (the sim camera is fixed-yaw, so this
  is the only geometry the rig produces).
* The flank fits are straight-line-quadratic locals: extreme curvature or
  heavy dropouts inside the band reject the frame to the global estimate.
* The `trim` calibration is a static bias: on the ring the affinity
  corridor should remove most of the curved-path drift, but re-check
  `trim_default` after enabling `ff_gain` (the FF changes the settle point).
