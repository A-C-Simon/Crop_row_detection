# result_02_DEFAULT_struct — Current Default Pipeline (FINAL)

**Producer**: `python test_multi_roi.py` (zero flags — defaults as of 2026-08-25)
**Images**: 22/22 succeeded (bev6 removed from dataset as degenerate)
**Mean time**: ~38 ms (typical ground frame ~20–30 ms; paper reference 240.8 ms)

## Process (final architecture)

1. **Segmentation**: ExG → Otsu → Eq. (3) opening → **embedded band-support
   weed clean** (per-strip Eq. 5–8 bands, P90 canopy fallback; blobs
   overlapping no band are erased before any corridor exists)
2. **Strip climb** with sliver mass gate (`--min_flank_frac 0.01`) and
   one-sided freeze (a lone row cannot drag window or corridor estimate)
3. **Evidence-gated navigation fit**: only two-sided picks with plausible
   flank span vote; **total least squares** (perpendicular residuals) +
   conservative rejection in perpendicular space
4. **Navigation curve**: smoothed cubic regression spline through accepted
   dots, clamped to a ±15%-of-corridor-span safety envelope around the
   straight TLS reference; falls back to that line when <4 dots survive
5. **Detection lines**: straight TLS through two-sided chain points filtered
   by corridor-width consistency (D ≤ 1.6×median), robust fitter as backstop

## Visualization contract

- 🟢 bright green dots — midpoints used by the navigation fit
- 🔴 red dots — excluded (synthetic/frozen, wide-band, strip-1, outliers)
- blue polyline — navigation curve (or line when fallback)
- dark-blue lines — detection lines · white boxes — per-strip ROIs

## Eye-test status

✅ photo_9, photo_6, photo_7, photo_11, photo_17 (detection-line rebuild),
plus corridor-quality checks on photo_5/8/1/3.

## Final base-tangent angles (deg from vertical)

bev 4.63 · bev2 3.71 · bev4 2.67 · bev5 5.79 · bev7 0.58 · photo_1 18.87 ·
photo_2 54.02 ⚠ · photo_3 23.06 · photo_4 17.42 · photo_5 2.36 · photo_6 5.05 ·
photo_7 12.54 · photo_8 6.24 · photo_9 9.53 · photo_10 8.89 · photo_11 0.76 ·
photo_12 26.36 · photo_13 7.39 · photo_14 0.18 · photo_15 3.36 · photo_16 15.30 ·
photo_17 18.29

(⚠ photo_2 = anomalous strongly curved field; metric is now the curve tangent
at image base — not directly comparable to straight-line-era numbers.)

## Contents

Same layout as sibling directories. `multiroi_masks/` carries the full audit
trail: green/red evidence dots, ROIs, curve and detection lines.
