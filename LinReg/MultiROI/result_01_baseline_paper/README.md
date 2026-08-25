# result_01_baseline_paper — Untouched Paper Baseline

**Producer**: `multi_roi.py` (faithful re-implementation of Zhou et al. 2021)
**Command**: `python multi_roi.py --input ../../Photos --results_dir ./result_01_baseline_paper`
**Date**: 2026-08-24 · **Images**: 23/23 succeeded · Mean time 11.5 ms

## Process
Paper pipeline only: ExG → Otsu → anti-diagonal opening (Eq. 3) → 10-strip
adaptive multi-ROI → least-squares nav line + detection lines.
No weed handling beyond morphology. Serves as the reference all
`test_multi_roi.py` experiments are compared against (`EXPERIMENT_NOTES.md`).

## Known weaknesses (motivated every later experiment)
- Weed slivers can anchor the ROI at strip 1 (photo_9 right detection line
  sits on a weed blob; true row further right)
- photo_6's original corridor was anchored on a 0.004-mass weed spike
- photo_7's saturated canopy froze the ROI at full image width
- Dense-weed scenes (photo_9) corrupt the navigation angle

## Contents
- `overlays/` — nav + detection lines on original images
- `multiroi_masks/` — binary mask + ROI staircase + Q midpoints
- `binary/` — post-morphology binary masks
- `detection_data.csv` — filename, nav_angle_from_vertical_deg,
  n_nav_points, n_detection_lines, time_ms
