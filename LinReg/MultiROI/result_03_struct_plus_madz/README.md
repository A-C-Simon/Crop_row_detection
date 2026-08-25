# result_03_struct_plus_madz — Struct Morph + Gated MAD-z Scrub

**Producer**: `test_multi_roi.py --post_scrub --scorer madz`
**Date**: 2026-08-25 · **Images**: 23/23 succeeded · Mean time 45.3 ms

## Process
`result_02_DEFAULT_struct` pipeline, plus the anchor-based MAD-z outlier
scrub running after detection — gated by weed pressure (≥0.35) and corridor
bend (<45°), so it stays dormant unless needed.

## Result
The scrub fired on 6 images and changed exactly ONE: bev6 (7.54 → **4.33°**),
where weeds are band-aligned in their strip but globally off-corridor —
the one failure class per-strip bands cannot see. All other 22 images are
identical to `result_02_DEFAULT_struct`.

## Status
Optional safety net on top of the default. Kept available via `--post_scrub`
because it is free when dormant; NOT default because the same mechanism
deleted real crop rows on photo_11/17 in other runs.

## Contents
Same layout as sibling directories.
