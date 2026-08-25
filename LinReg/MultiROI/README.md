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
