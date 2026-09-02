# Adaptive Multi-ROI Crop Row Detection — Weed-Robust Edition

Re-implementation + hardening of:

> Zhou Y, Yang Y, Zhang B L, Wen X, Yue X, Chen L Q. **Autonomous detection of
> crop rows based on adaptive multi-ROI in maize fields.** Int J Agric & Biol
> Eng, 2021; 14(4): 217-225. DOI: [10.25165/j.ijabe.20211404.6315](https://doi.org/10.25165/j.ijabe.20211404.6315)

This directory contains two programs:

| File | Role |
|---|---|
| `multi_roi.py` | faithful re-implementation of the paper (frozen reference) |
| `test_multi_roi.py` | **the hardened pipeline — current default** (this README documents it) |

Supporting docs: `audit.md` (implementation-vs-paper audit),
`EXPERIMENT_NOTES.md` (every strategy tested, kept or rejected, with data),
`result_*/` (six versioned result sets, each with its own README).

---

## Final pipeline (`python test_multi_roi.py`, zero flags)

```
BGR image
  ▼
ExG color index → Otsu → anti-diagonal opening (paper Eq. 1–3)
  ▼
EMBEDDED STRUCT-CLEAN ──────────────────────────────────────────┐
 per strip: column projection (Eq. 5–8, P90 fallback for        │
 saturated canopies) → row-candidate bands                      │
 every connected component overlapping a band survives;         │
 isolated off-structure blobs are erased BEFORE any corridor    │
 exists → wrong-lane capture is impossible by construction      │
  ▼                                                             │
STRIP CLIMB (Eq. 4–10), bottom → top                            │
 • clusters below 1% of strip projection mass are rejected as   │
   weed slivers (--min_flank_frac)                              │
 • ONE-SIDED strips freeze window + corridor estimate           │
   (a lone row can no longer drag navigation onto itself)       │
  ▼
EVIDENCE-GATED NAVIGATION FIT
 • only two-sided picks with plausible flank span               │
   (D ≤ 1.6 × median) count as measurements                     │
 • synthetic/frozen dots + wide-band picks are excluded (red)   │
 • TOTAL LEAST SQUARES (perpendicular residuals) — correct for  │
   near-vertical corridors where classic OLS degenerates        │
 • conservative outlier rejection in perpendicular space        │
  ▼
NAVIGATION CURVE (default on)
 smoothed cubic regression spline through the accepted dots;
 SAFETY ENVELOPE: never deviates > 15% of median corridor span
 from the straight TLS reference; falls back to that straight
 line when < 4 dots survive
  ▼
DETECTION LINES (straight, TLS)
 built only from two-sided picks whose span passes the same
 corridor-width filter; robust fitter as backstop
```

**Visualization contract** (`multiroi_masks/`): 🟢 green dots = measured
midpoints used by the navigation fit · 🔴 red dots = excluded (synthetic,
wide-span, strip-1, or perpendicular outliers) · blue polyline = navigation
curve · dark-blue lines = detection lines · white boxes = per-strip ROIs.

---

## Why it diverges from the paper

Every deviation below fixed an observed failure; full evidence in
`EXPERIMENT_NOTES.md` (failure modes F1–F10).

| # | Deviation | Failure it fixes |
|---|---|---|
| 1 | Raw ExG default (normalized ExG optional) | normalized index collapses on dense-green frames |
| 2 | Embedded band-support clean in segmentation | weeds created fake rows / wrong-lane corridors (photo_9, photo_6) |
| 3 | Flank-candidate mass gate (1% of strip mass) | 1-px weed slivers froze the ROI around the wrong lane |
| 4 | One-sided strips freeze instead of slide | sparse young-crop fields: nav collapsed onto the single visible row |
| 5 | P90 fallback when M+E annihilates a profile | full-canopy frames lost ALL structure (photo_7) |
| 6 | Evidence-gated Q (two-sided + plausible span only) | unworthy boxes voted on navigation while barred from det chains |
| 7 | Total least squares everywhere (nav + chains) | y-on-x OLS is ill-conditioned on near-vertical dot clouds (photo_8: 87° for a vertical corridor) |
| 8 | Corridor-width filter on chain points (D ≤ 1.6×median) | merged multi-row bands tilted whole det lines (photo_17) |
| 9 | Strip-1 midpoint excluded from nav fit | initial full-width view anchored outside the corridor at maximal leverage |
| 10 | Navigation path = safety-envelope spline | straight best-fit cuts across curving furrows; unsafe for steering |

Known remaining limitations: strongly curved fields exceed the linear
detection-line model (photo_2); extreme sparse scenes may yield few worthy
strips; a weed sitting exactly on a row line is geometrically indistinguishable
from crop.

---

## Results (22 images; bev6 removed as degenerate)

Navigation angle = tangent of the nav curve at the image base (steering
semantics). Baseline comparison is indicative only — the metric itself
improved (see deviation 7).

| Image | baseline | final | | Image | baseline | final |
|---|---|---|---|---|---|---|
| bev | 3.72 | 4.63 | | photo_10 | 14.66 | 8.89 |
| bev2 | 2.49 | 3.71 | | photo_11 | 1.68 | 0.76 |
| bev4 | 5.38 | 2.67 | | photo_12 | 15.99 | 26.36 |
| bev5 | 8.67 | 5.79 | | photo_13 | 10.00 | 7.39 |
| bev7 | 0.00 | 0.58 | | photo_14 | 7.14 | 0.18 |
| photo_1 | 15.47 | 18.87 | | photo_15 | 4.91 | 3.36 |
| photo_2 | 19.11 | 54.02 ⚠ | | photo_16 | 15.00 | 15.30 |
| photo_3 | 18.82 | 23.06 | | photo_17 | 7.73 | 18.29 |
| photo_4 | 11.06 | 17.42 | | photo_5 | 17.74 | 2.36 |
| photo_6 | 3.76 | 5.05 | | photo_7 | 8.96 | 12.54 |
| photo_8 | 3.37 | 6.24 | | photo_9 | 7.79 | 9.53 |

⚠ photo_2 is an acknowledged anomalous curved field. Eye-tested approvals:
photo_9, photo_6, photo_7, photo_11, photo_17.

Performance: ~35 ms mean, ~20 ms typical ground frame (baseline 11.5 ms;
paper reports 240.8 ms).

---

## Usage

```bash
# default (struct morph, navigation curve; no post-scrub)
python test_multi_roi.py --input ../../Photos --results_dir ./result_02_DEFAULT_struct

# add the dormant anchor-based scrub (fires only under high weed pressure)
python test_multi_roi.py --post_scrub --scorer madz          # or --scorer if

# ablations / experiments
python test_multi_roi.py --morph paper                       # paper segmentation only
python test_multi_roi.py --morph robust --morph_fence if     # oriented openings experiment
python test_multi_roi.py --no-nav_curve                      # straight TLS navigation
```

Key flags: `--n_strips --l_frac --min_flank_frac --weed_gate --max_bend
--nav_curve/--no-nav_curve --post_scrub --scorer {if,madz} --morph
{struct,paper,robust}` (see `--help` for the full set).

## Result directories

| Directory | Content |
|---|---|
| `result_01_baseline_paper/` | untouched `multi_roi.py` output |
| `result_02_DEFAULT_struct/` | **current default** (this README's subject) |
| `result_03_struct_plus_madz/` | default + gated MAD-z scrub |
| `result_04_if_scrub_papermorph/` | paper morph + gated IF scrub (superseded) |
| `result_05_madz_scrub_papermorph/` | paper morph + gated MAD-z scrub (superseded) |
| `result_06_robustmorph_iffence/` | oriented-openings experiment (rejected) |

Each directory has its own README describing exactly how it was produced.

---

## Navigation Stack - Furrow Following Visual Servoing (2026-08)

This directory now includes a complete navigation stack that turns the
already available navigation line (furrow corridor centre) into actual
rover motion. The navigation line for MultiROI is along the furrow
between the two crop rows, while ExG follows the crop row itself, but
the control principle is identical. We reuse the visual servoing
formulation from `ExG/visual-crop-row-navigation_ros2/src/agribot_vs.cpp`.

### Stack Overview

```
Image -> MultiROI Detector (furrow centre nav_line/nav_curve)
  -> Visual Features (X, Theta) at bottom of image
  -> Visual Servoing Controller (v, w)
  -> Rover Kinematics (differential drive)
  -> /cmd_vel or simulated pose
```

**Files:**

| File | Role |
|---|---|
| `mr_vs.py` | Visual servoing for furrow. Converts `nav_line` `(w,b)` or `nav_curve` `[(x,y)]` to feature `F=[X,Y,Theta]` and computes `v,w` via `w = -(lambda_x * X/width + lambda_theta * Theta)` |
| `mr_navigation_stack.py` | Full stack `MultiROINavigationStack`: `detector.detect()` -> `vs.nav_line_to_feature()` -> `vs.compute_control()` -> `simulator.update()`. Handles `crop_offset`, curve tangent at base, and draws overlay |
| `run_mr_navigation.py` | Offline runner for Photos or video. For each frame produces `*_nav.png` overlay and `nav_commands.csv` with `v,w,err_x,err_theta` |

**Visual features** `mr_vs.py:74` `nav_line_to_feature`:

* `nav_curve` preferred if available: bottom two points of the spline give `P` (bottom) and `Q` (top) in full image coords
* Else `nav_line` `y=w*x+b` in cropped coords converted to full image via `b_full = b + dy - w*dx`, intersections at `y=0` and `y=h-1`
* `X = P.x - width/2` lateral error at bottom (pixels, negative is left)
* `Y = P.y - height/2`
* `Theta = pi/2 - atan2(P.y - Q.y, Q.x - P.x)` deviation from vertical, `wrapToPi`, `0` is vertical

Desired `F_des = [0, _, 0]` - centred and vertical at the bottom. The rover's nose is at `width/2, height-20` (star marker).

**Control law** `mr_vs.py:152` `compute_control`:

```
err_x = X, err_theta = Theta
err_x_norm = err_x / width
w_raw = -(lambda_x * err_x_norm + lambda_theta * err_theta)
w = clamp(w_raw, -w_max, w_max), zero if |w|<w_min
v = vf_des (constant 0.20 m/s)
```

Defaults `MRVSParams:46` `lambda_x=2.0` `lambda_theta=1.0` `vf_des=0.20` `w_max=0.60` `w_min=0.01` tuned so `100px` lateral (`0.156` norm) gives `0.31` rad/s plus `10 deg` heading (`0.17` rad) stays within `0.60`. Same structure as `agribot_vs.cpp:385` `w = -Jw_pinv*(lambda*err + Jv*v)` but simplified.

If no line, `v=w=0` (could search with `w=0.2`).

**Rover kinematics** `mr_navigation_stack.py:58` `RoverSimulator`:

```
x += v * cos(theta) * dt
y += v * sin(theta) * dt
theta = wrapToPi(theta + w*dt)
dt=0.1
```

Path is logged and plotted to `rover_path.png`.

### Usage

```bash
# Offline on Photos - Produces nav_output with overlays and CSV
#   --input points to the 23 test images, --output creates nav_output/
#   For each image it runs MultiROI detection to get the furrow centre
#   nav_line/nav_curve, converts it to visual features X,Theta at the
#   bottom of the image, then computes v,w with visual servoing.
#   Output per image is nav_output/<name>_nav.png (red furrow line,
#   star at rover nose, v,w,err) and nav_commands.csv with all
#   velocities and line params. Mean time about 10 ms detector plus
#   1 ms control per image.
python3 run_mr_navigation.py --input ../../Photos --output ./nav_output

# On video file - Processes a recorded video frame by frame
#   --video tells the runner to treat input as video, not folder.
#   It reads each frame, runs the same detection and control, writes
#   a side-by-side video nav_output/<video>_nav.mp4 (left nav overlay,
#   right MultiROI composite) and nav_output/<video>_nav.csv per frame.
#   Use this to test the stack on test_video1.mp4 or test_video2.mp4.
python3 run_mr_navigation.py --input /path/to/video.mp4 --output ./nav_output --video

# Live camera 0 with preview, q to quit - Runs on a real rover camera
#   --input 0 selects /dev/video0, --video plus --show opens a live
#   OpenCV window with the combined nav/composite view. The same v,w
#   that would be published to /cmd_vel are displayed. Press q to quit.
#   For ROS2, wrap the same MultiROIVS in MultiROINavNode subscribing
#   to /front/image_raw.
python3 run_mr_navigation.py --input 0 --output ./nav_output --video --show

# Tune gains - Adjusts the visual servoing response
#   --vf forward speed m/s, --wmax max angular rad/s, --lambdax lateral
#   gain on err_x/width, --lambdatheta heading gain on Theta. Higher
#   lambdax gives stronger correction for off-centre furrow, higher
#   lambdatheta for tilted furrow. Default 0.20, 0.60, 2.0, 1.0 keeps
#   100px lateral within 0.60 rad/s limit. Use this to tune for your
#   rover wheelbase and camera tilt.
python3 run_mr_navigation.py --input ../../Photos --vf 0.20 --wmax 0.60 --lambdax 2.0 --lambdatheta 1.0

# Different detector settings still work - Passes through to MultiROI
#   --n_strips number of horizontal strips, --l_frac clustering distance
#   as fraction of width, --index raw or normalized ExG. The navigation
#   stack will then follow whatever furrow the detector finds, so you
#   can test robustness to different segmentation or strip settings
#   without changing the controller.
python3 run_mr_navigation.py --input ../../Photos --n_strips 10 --l_frac 0.05 --index raw
```

Outputs per run:

* `nav_output/<name>_nav.png` - BGR with red navigation line, bottom/top points, centre star, and text `v, w, err_x, err_theta`
* `nav_output/<name>_composite.png` - original MultiROI composite for reference
* `nav_output/nav_commands.csv` - `filename, v, w, err_x, err_theta, nav_w, nav_b, has_line, time_ms`
* `nav_output/rover_path.csv` and `rover_path.png` - simulated path when run as sequence (Photos as frames)

For the 23 Photos as a pseudo-sequence, the stack yields `22/23` lines, mean `~10 ms` detector + `~1 ms` control, path `0.46 m` forward with heading `~1.1 deg` final, velocities mostly within limits (e.g. `bev` `err_x -221` `err_theta 4.6` `w 28.6 deg/s` at limit, `photo_8` `err_x -10` `err_theta 5.5` `w 3.1 deg/s`).

### ROS2 Integration

`mr_vs.py` bottom contains a commented `MultiROINavNode` example:

```python
import rclpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class MultiROINavNode(Node):
    def __init__(self):
        super().__init__('multiroi_nav')
        self.bridge = CvBridge()
        self.detector = MultiROIDetector()
        self.vs = MultiROIVS()
        self.sub = self.create_subscription(Image, '/front/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
    def cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        res = self.detector.detect(bgr)
        F,_ = self.vs.nav_line_to_feature(res["nav_line"], res["nav_curve"], res["crop_offset"], bgr.shape)
        v,w,_ = self.vs.compute_control(F) if F is not None else (0.0,0.0,{})
        twist = Twist(); twist.linear.x=float(v); twist.angular.z=float(w)
        self.pub.publish(twist)
```

Build as a ROS2 package, `colcon build`, `ros2 launch` similar to `ExG`.

This completes the navigation stack from the already available furrow line to actual rover motion, analogous to `ExG` but for the MultiROI corridor. The same `IsolationForest` and `column-aware` improvements from `ExG` can be ported here if needed, but the current `test_multi_roi.py` already includes embedded struct-clean and TLS robustness.
