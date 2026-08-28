# CAROLIF + Preclean-Embedded Morphology — Single-Window Results (23 images)

**Paper:** Khan et al. 2024 (Alg.1 CAROLIF) + MultiROI preclean framework (LinReg/MultiROI/test_multi_roi.py)

All stages for **one input image are in ONE window** (2×4 grid) — no scrolling.

## What was implemented (your preferred preclean embedding)

- `carolif.py:55 KERNEL_K` kept for reference; **default struct uses isotropic ellipse + preclean** (photo_17: K 4×4 anti-diagonal shredded sparse rows → VP (502,-509) → 0 rows; ellipse preserves row → VP (251,-2292) → 1 row, fixed)
- `carolif.py:129 column_projection` Eq.5-8 + P90 fallback
- `carolif.py:148 cluster_feature_columns` gap<L grouping
- `carolif.py:165 pre_corridor_clean` band-support: per-strip N=10, bands → component overlap (+6px, ±1 strip) → isolated blobs erased, giants >5% spared, no-band strips untouched
- `carolif.py:228 segment_green` branch: `struct` → `ellipse MORPH_OPEN (3) + pre_corridor_clean` (opening-only, closing removed per EXPERIMENT_NOTES.md) with safety valve (if retained <40% or <500px or removed_frac>0.5 → revert to opened mask) for sparse scenes like photo_17; `paper` → ellipse open+close
- `run_carolif.py:39 save_composite` single-window; `run_carolif.py:152 --morph struct|paper`

Embedding is **inside morphological processing** BEFORE homography — wrong-lane capture impossible (MultiROI photo_9/6 eye tests).

## Layout per `*_result.png`

01 Original | 02 ROI Sec3.1.1 | 03 Binary Mask K+preclean | 04 Top View homography | 05 Clusters HDBSCAN raw=N | 06 Detection RANSAC slope[70,110]° | 07 Result red rows | 08 Summary (VP, rows, timings, morph tag)

## Results 23 Photos

**Default `struct` (this folder, FIXED photo_17):** 23/23 OK, 56 rows total, avg 845 ms — **photo_17 now 1 row** (was 0 with K kernel), VP (251,-2292), corrected

**Paper ablation `result_paper/`:** 58 rows total

Delta struct-paper (selected):
- photo_2 4→1, photo_7 3→2, photo_14 3→1, photo_15 5→3, photo_12 2→1 : weed false positives removed by band filter
- photo_3 1→2, photo_4 1→5, bev2 4→5, bev 12→13 : recovered true rows previously merged
- photo_17 1→1 (fixed, was 0 with K) — sparse young-crop field, preclean now conservative via ellipse + valve

Photo_17 detail: K open 7239 px → VP 502,-509 → 0 rows; ellipse open 7820 px → retained → preclean 1 blob (19px) → 1 row stable.

## Reproduce

```bash
cd /home/ac/Crop_Row_Detection_Techniques/ClusterAlg
python3 run_carolif.py --input ../Photos --output ./result --morph struct   # preclean DEFAULT (fixed)
python3 run_carolif.py --input ../Photos --output ./result_paper --morph paper
```

Config: `CarolifConfig(morph="struct", pre_clean_n_strips=10, l_frac=0.05, pad=6, min_area=12, max_blob_frac=0.05)`
