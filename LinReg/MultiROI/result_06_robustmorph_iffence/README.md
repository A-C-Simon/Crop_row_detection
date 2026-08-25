# result_06_robustmorph_iffence — Oriented Openings + Embedded IF Fence

**Producer**:
`test_multi_roi.py --morph robust --post_scrub --morph_fence if --scorer madz`
**Date**: 2026-08-25 · **Images**: 23/23 · Mean time 216 ms

## Process
Segmentation replaced by an experimental opening-only chain:
1. Union of oriented line openings (6 directions × 9 px) — keeps anything
   containing a straight run in any direction; round weeds vanish
2. Attribute area opening (components < 12 px dropped)
3. Embedded IsolationForest fence over shape attributes
   [log-area, elongation, extent] — jointly unusual blobs erased
(gated MAD-z scrub may then run afterwards)

## Status: EXPERIMENT — not recommended as default
Mixed outcomes vs baseline:
- photo_9 10.73° / bev6 4.06° / photo_6 3.67° — good
- **photo_5 56.50° 🚨** — openings fragment this scene; failure persists in
  every fence variant (openings-stage issue, see notes OQ-1)
- photo_2 34.84° ⚠ · bev 10.93° 🚨 regression
- Costliest configuration (~216 ms mean; per-image forest fits inside
  segmentation)

## Lessons retained (see EXPERIMENT_NOTES.md)
Closing hurts (user-confirmed); fences disagree scene-by-scene (no free
lunch); kept behind flags as documented negative results.

## Contents
Same layout as sibling directories.
