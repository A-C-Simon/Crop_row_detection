# result_04_if_scrub_papermorph — Paper Morph + Gated IF Scrub

**Producer**: `test_multi_roi.py --morph paper --post_scrub` (scorer `if`)
**Date**: 2026-08-25 refresh · **Images**: 23/23 · Mean time 49.2 ms

## Process
Original paper segmentation (no struct-clean), then a two-pass scheme:
pass-1 lines act as anchors; IsolationForest scores candidate blobs on
[distance-to-anchor, log-area, elongation, ExG mean, ExG margin]; flagged
weeds erased; strip climb re-run on the cleaned mask. Gated by weed
pressure and bend guard.

## Historical role
First working weed-removal strategy (photo_9 eye-test approved at 10.28° in
its day). Superseded after eye tests showed the IF scrub deleting real crop
rows on photo_11/17, and after the embedded struct-clean proved stronger.

## Known issues visible here
- photo_2 22.01° / photo_8 7.75° — nav-fit relocations from excluding
  strip-1 midpoint (see EXPERIMENT_NOTES.md OQ-3)
- No structural fix: decoys still reach detection at strip 1

## Contents
Same layout as sibling directories. CSV includes weed_pressure,
if_removed_components, if_removed_veg_frac columns.
