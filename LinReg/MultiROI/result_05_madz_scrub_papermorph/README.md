# result_05_madz_scrub_papermorph — Paper Morph + Gated MAD-z Scrub

**Producer**: `test_multi_roi.py --morph paper --post_scrub --scorer madz`
**Date**: 2026-08-25 refresh · **Images**: 23/23 · Mean time 16.3 ms

## Process
Paper segmentation, then the two-pass anchor scrub using a one-sided
robust fence instead of a forest: candidates (blobs away from pass-1
anchor lines) with `d_anchor > median + z_thresh·MAD` are erased; strip
climb re-runs on the cleaned mask. Gated by weed pressure and bend guard.
Deterministic and ~10× cheaper than the IF variant.

## Historical role
Cheapest working scrub; matched IF's angles on 21/23 images after the
infrastructure fixes (sliver mass gate, nav-skip-first, P90 fallback).

## Why it was rejected as default
Eye tests showed it deleting actual crop rows on photo_11 and photo_17
(heavy scrubs: 27.5%/19–25% of vegetation removed). The embedded
struct-clean achieves equivalent corridor quality without touching row
material — see `result_02_DEFAULT_struct`.

## Contents
Same layout as sibling directories. CSV includes weed_pressure,
if_removed_components, if_removed_veg_frac columns.
