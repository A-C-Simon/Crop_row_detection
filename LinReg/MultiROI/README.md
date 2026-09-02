# Adaptive Multi-ROI Crop Row Detection — Weed-Robust Edition

Re-implementation + hardening of:

> Zhou Y, Yang Y, Zhang B L, Wen X, Yue X, Chen L Q. **Autonomous detection of crop rows based on adaptive multi-ROI in maize fields.** Int J Agric & Biol Eng, 2021; 14(4): 217-225. DOI: [10.25165/j.ijabe.20211404.6315](https://doi.org/10.25165/j.ijabe.20211404.6315)

This directory contains the hardened detector and the full navigation stack:

| File | Role |
|---|---|
| `multi_roi.py` | faithful re-implementation of the paper (frozen reference) |
| `test_multi_roi.py` | **hardened pipeline — current default** (this README documents it) |
| `lookahead_corridor_map.py` | visible-future corridor memory (spatiotemporal, motion-compensated) |
| `temporal_filter.py` | scalar temporal filter + command smoother for `X/Theta/width` and `w` |
| `mr_vs.py` | visual servoing for furrow: `nav_line/nav_curve` → `F=[X,Y,Theta]` → `v,w` |
| `mr_navigation_stack.py` | full stack `detector → lookahead → temporal → vs → simulator` |
| `run_mr_navigation.py` | offline runner for Photos or video (with loop, line, lookahead) |
| `test_temporal_filter.py` | synthetic tests for temporal and lookahead behavior |

Supporting docs: `audit.md` (implementation-vs-paper audit), `EXPERIMENT_NOTES.md` (every strategy tested, kept or rejected, with data), `result_*/` (versioned result sets).

---

## Final pipeline (`python test_multi_roi.py`, zero flags)

```
BGR image
  ▼
ExG color index → Otsu → anti-diagonal opening (paper Eq. 1–3)
  ▼
EMBEDDED STRUCT-CLEAN
 per strip: column projection (Eq. 5–8, P90 fallback for saturated canopies)
 → row-candidate bands
 every connected component overlapping a band survives;
 isolated off-structure blobs are erased BEFORE any corridor exists
 → wrong-lane capture is impossible by construction
  ▼
STRIP CLIMB (Eq. 4–10), bottom → top, bottom-anchored vertical coverage 0.75
 effective_h = 0.75*h, dh = effective_h//10, so ROIs cover bottom 3/4 only
 top 1/4 has no boxes; each white box height = dh * roi_draw_frac (default 1.0)
 • clusters below 1% of strip projection mass are rejected as weed slivers (--min_flank_frac)
 • ONE-SIDED strips freeze window + corridor estimate (lone row cannot drag nav onto itself)
 • lookahead prior seeding: if a high-confidence future corridor is predicted
   for this strip, a sudden raw width/center jump is held (ROI/midpoint not updated)
  ▼
EVIDENCE-GATED NAVIGATION FIT
 • only two-sided picks with plausible flank span (D <= 1.6 × median) count
   synthetic/frozen dots + wide-band picks + suppressed strips are excluded (red)
 • bottom N strips can be forced ignored (--ignore-initial 3) for noisy entry:
   those midpoints are marked suppressed/red and the robot follows centre star
 • TOTAL LEAST SQUARES (perpendicular residuals) — correct for near-vertical corridors
 • conservative outlier rejection in perpendicular space (up to 3 points, 2.5*std)
  ▼
NAVIGATION CURVE (default on, --line for straight)
 smoothed cubic regression spline through accepted dots;
 SAFETY ENVELOPE: never deviates > 15% of median corridor width from TLS reference;
 falls back to straight line when < 4 dots survive
  ▼
DETECTION LINES (straight, TLS)
 built only from two-sided picks whose span passes same corridor-width filter
 and not suppressed; robust fitter as backstop
```

**Per-strip profile exposed for lookahead** (`res["strip_profile"]`): `mu, y_top/y_bot/y_center, center_x, width, left_x/right_x, two_sided, accepted_nav, suppressed, roi`. Also `q_feed, left_feed, right_feed, feed_d, median_width, bottom_width, n_two_sided, raw_bottom_x`.

**Visualization contract** (`multiroi_masks/` and composite):
- middle panel white boxes = per-strip ROIs (now bottom 3/4 only, `36px` vs `48px` for `480h`), green dots `q_accepted`, red dots `q_rejected`/`suppressed`/`ignore_initial`
- blue polyline = navigation curve (spline) or straight TLS when `--line`
- dark-blue lines = detection lines
- overlay bottom panel: filtered red line `P->Q` (8px red bottom, 5px yellow top), raw thin cyan line, green lookahead boxes + centre dots, yellow `pred_bottom_x` (lookahead prediction), centre star `width/2, height-20`, text `v,w,err`

---

## Why it diverges from the paper

Every deviation fixed an observed failure; full evidence in `EXPERIMENT_NOTES.md`.

| # | Deviation | Failure fixed |
|---|---|---|
| 1 | Raw ExG default (normalized optional) | normalized collapses on dense-green frames |
| 2 | Embedded band-support clean in segmentation | weeds created fake rows / wrong-lane corridors (photo_9, photo_6) |
| 3 | Flank mass gate 1% | 1-px weed slivers froze ROI around wrong lane |
| 4 | One-sided strips freeze instead of slide | sparse young-crop: nav collapsed onto single visible row |
| 5 | P90 fallback when M+E annihilates profile | full-canopy frames lost all structure (photo_7) |
| 6 | Evidence-gated Q (two-sided + plausible span only) | unworthy boxes voted on nav while barred from chains |
| 7 | Total least squares everywhere | y-on-x OLS ill-conditioned on near-vertical clouds (photo_8) |
| 8 | Corridor-width filter on chain points (D ≤1.6×median) | merged multi-row bands tilted det lines (photo_17) |
| 9 | Strip-1 midpoint excluded from nav fit | initial full-width view anchored outside corridor at maximal leverage |
| 10 | Navigation path = safety-envelope spline | straight best-fit cuts across curving furrows |
| 11 | Vertical coverage 0.75 (bottom 3/4 only) | top boxes at horizon are noisy/perspective-compressed; user request to keep boxes in lower 3/4 |
| 12 | Ignore initial N bottom strips (`--ignore-initial 3`) | damaged/trampled entry rows grab wide; robot follows centre star until upper good evidence |
| 13 | Lookahead prior gating inside strip climb | missing-plant gap at bottom would expand ROI to 3 rows and shift midpoint 50px (crops.mp4) |
| 14 | Lookahead corridor map (visible-future memory) | long gaps persist 10+ frames and would otherwise be accepted after `persist_frames` |
| 15 | Scalar temporal filter + command smoother | single-frame jumps became violent `w` spikes and lane drift |

Known remaining: strongly curved fields exceed linear det-line model (photo_2); extreme sparse may yield few worthy strips; weed exactly on row line is indistinguishable.

---

## Navigation stack (lookahead + temporal + vs)

```
Image -> MultiROI Detector (strip_profile, nav_line/curve, 0.75 coverage, ignore N)
  -> LookaheadCorridorMap (visible-future memory, motion-compensated)
  -> TemporalNavigationFilter (innovation gating, persistence, EMA)
  -> MultiROIVS (low-pass/rate-limit/deadband/conf speed scale)
  -> Rover kinematics / /cmd_vel
```

**LookaheadCorridorMap** (`lookahead_corridor_map.py:40`):
- Bins `n=10` matching strips, each `center_x/width/left/right/confidence/age`. `vertical_coverage=0.75` so `dh = h*0.75/n`, bottom-anchored. `shift_px=14` forward per frame: `pred_bins[i] = old_bins[i+shift/dh]` interpolated with `conf*0.97` decay.
- Per-strip gate `|raw_c-pred_c| <= max(12,0.35*pred_w)` and `|raw_w-pred_w| <=0.35*pred_w`; if high-conf `pred` and raw exceeds gate → `suppress` (hold ROI/midpoint, `d_strip=None` so nav ignores it).
- Spatial support: `support_old` = upper bins consistent with old map, `support_new` = upper bins consistent with bottom raw new. Bottom conflict with `support_old>=3` and `support_new<3` → `map_hold` (keep old, `corr=pred`); with `support_new>=3` → `map_pending` → after `accept_frames=4` → `map_switch` (blend `0.35` toward new). Missing bottom → `occlusion_hold` (use `pred`). Effective bottom is `ignore_initial` (e.g. `3 → mu=4`) so approach phase uses higher, cleaner strips.

**TemporalNavigationFilter** (`temporal_filter.py:62`):
- Filters stable features `bottom_x (=w/2+X)`, `Theta`, `width` (never raw `w,b`). Constant prediction + optional `motion_gain`. Innovation `raw-pred` vs `max(18px, max(0.10*width,0.45*pred_w))`, `12 deg`, `0.30*pred_w`. Large → `pending` (tiny `0.08` nudge, conf drops), same `pending` for `persist_frames=4` within `0.5*thresh` → `pending_accepted` (EMA `0.35` toward pending). Else EMA `0.35/0.35/0.30` and conf EMA `0.40`.

**Controller smoothing** (`mr_vs.py:46`):
- `w_raw = -(lambda_x*X/width + lambda_theta*Theta)`, clamp `w_max 0.60`, `w_min 0.01`
- If `smooth`: `w_lpf = (1-alpha)*prev + alpha*w` (`w_alpha 0.35`), rate limit `|Δw| <= max_w_rate*dt` (`1.2 rad/s²`), deadband `0.02`, `v = vf*(0.45+0.55*conf)` when `v_conf_scale`.

---

## Usage

```bash
# default (struct morph, spline, lookahead+temporal on for video, off for folder,
#          vertical coverage 0.75 bottom 3/4, ignore 0)
python test_multi_roi.py --input ../../Photos --results_dir ./result_test
python LinReg/MultiROI/run_mr_navigation.py --input ../../Photos --output ./nav_output

# video (lookahead+temporal on by default)
python LinReg/MultiROI/run_mr_navigation.py --input LinReg/crop_line_detector_cv/images/crops.mp4 --output /tmp --video --show
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --loop --show   # loop forever until q/Ctrl-C

# straight line instead of spline
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --line
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --show --line --loop

# noisy entry: ignore bottom 3 boxes/midpoints, follow centre star until upper good
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --ignore-initial 3
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --show --ignore-initial 3 --vertical-coverage 0.75

# explicit lookahead/temporal control
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --no-lookahead
python LinReg/MultiROI/run_mr_navigation.py --input ../../Photos --output /tmp --lookahead --temporal
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --lookahead-shift-px 14 --lookahead-center-gate 0.35 --lookahead-accept-frames 4

# tune gains / coverage / strip geometry
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --vf 0.20 --wmax 0.60 --lambdax 2.0 --lambdatheta 1.0 --n_strips 10 --l_frac 0.05 --vertical-coverage 0.75 --roi-draw-frac 1.0
python LinReg/MultiROI/run_mr_navigation.py --input crops.mp4 --output /tmp --video --no-temporal --vertical-coverage 1.0  # full height for comparison

# detector ablations
python test_multi_roi.py --morph paper --no-nav_curve
python test_multi_roi.py --post_scrub --scorer madz
```

Key flags `run_mr_navigation.py:646`:
- `--video` / `--loop` / `--line` / `--show` / `--width/height`
- `--ignore-initial N` (0 default, 3 for noisy entry), `--vertical-coverage 0.75` (bottom fraction), `--roi-draw-frac 1.0`
- `--temporal/--no-temporal` (`video` on), `--lookahead/--no-lookahead` (`video` on)
- `--persist-frames 4 --max-jump-frac 0.10 --max-jump-width-frac 0.45 --max-heading-jump 12 --max-width-change 0.30 --w-alpha 0.35 --max-w-rate 1.2 --w-deadband 0.02 --alpha-x 0.35 --alpha-theta 0.35`
- `--lookahead-shift-px 14 --lookahead-center-gate 0.35 --lookahead-width-gate 0.35 --lookahead-accept-frames 4 --lookahead-accept-bins 3 --lookahead-conf-decay 0.97 --lookahead-overlay`
- detector: `--n_strips --l_frac --min_flank_frac --index {raw,normalized} --morph {struct,paper,robust}`

Outputs per run:
- `*_nav.png` overlay (filtered red `P->Q` 8/5px, raw thin cyan, green map boxes + centre dots, yellow `pred_bottom_x`, centre star, text `v,w,err,conf,status,map_status`)
- `*_composite.png` (binary / mask+ROIs+dots / overlay stacked)
- `nav_commands.csv` / `*_nav.csv` (`filename/frame, v, w, raw_err, filt_err, conf, status, innov, median/filt_width, n_two_sided, has_line, time_ms, map_status/conf/innov, support_old/new`) and `nav_video.mp4` for video (side-by-side).

---

## Result directories

| Directory | Content |
|---|---|
| `result_01_baseline_paper/` | untouched `multi_roi.py` |
| `result_02_DEFAULT_struct/` | current default (this README) |
| `result_03_struct_plus_madz/` | + gated MAD-z scrub |
| `result_04_if_scrub_papermorph/` | paper morph + IF scrub (superseded) |
| `result_05_madz_scrub_papermorph/` | (superseded) |
| `result_06_robustmorph_iffence/` | oriented openings (rejected) |

---

## ROS2 Integration

`mr_vs.py` bottom `MultiROINavNode` example now uses `process_image(..., lookahead_map, t_filter)`:

```python
from test_multi_roi import MultiROIDetector
from lookahead_corridor_map import LookaheadCorridorMap
from temporal_filter import TemporalNavigationFilter

detector = MultiROIDetector(vertical_coverage=0.75, ignore_initial=0)
lookahead = LookaheadCorridorMap()
t_filter = TemporalNavigationFilter()
# in callback:
pred = lookahead.get_prediction(bgr.shape)
res = detector.detect(bgr, lookahead_prior={"pred_bins": pred})
map_out = lookahead.update(res["strip_profile"], res, bgr.shape, dt=0.05, last_w=self.last_w)
# feed corrected to temporal then vs...
```

Build as ROS2 package, `colcon build`, `ros2 launch` similar to `ExG`.

Performance: detector `~35 ms` mean, `~20 ms` typical ground frame (`240.8 ms` paper), plus lookahead+temporal `<1 ms`, control `~1 ms`.

