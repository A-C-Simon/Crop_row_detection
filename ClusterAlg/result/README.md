# CAROLIF — Single-Window Results (21 images)

**Paper:** Khan et al. 2024, Frontiers in AI 7:1435686 — Algorithm 1 CAROLIF

All pipeline stages for **one input image are now in ONE window** (2×4 grid, 20×10 inch, 150 dpi), as requested.

## Layout per `*_result.png` (one file per input)

| Panel | Title | Paper step |
|---|---|---|
| 0 | 01 Original | input |
| 1 | 02 ROI (Sec 3.1.1) | bottom 58% of frame (≥3 rows) |
| 2 | 03 Binary Mask ExG+Otsu+Morph (Eq.9) | `normalize_exg` + Otsu + open/close |
| 3 | 04 Top View Projective (Sec 3.1.1.1) | homography H (Eq.1-6) auto VP |
| 4 | 05 Clusters HDBSCAN raw=N | Sec 3.1.2 |
| 5 | 06 Detection View RANSAC (Sec 3.1.3) | per-cluster line fit |
| 6 | 07 Result N row(s) | red lines overlaid on original |
| 7 | 08 Summary | VP, n_rows, timings, pipeline string |

No more scrolling through 7 separate files — open e.g. `bev_result.png` and see everything.

## Files

- `*_result.png` : 21 composite figures (one per photo in `/Photos`)
  - `bev*.png` → bird's-eye / top-down scenes
  - `photo_*.png` → perspective field photos
- `summary.csv` : per-image n_rows, n_clusters_raw, VP, per-step ms, FPS

## Reproduce

```bash
cd /home/ac/Crop_Row_Detection_Techniques/ClusterAlg
python3 run_carolif.py --input /home/ac/Crop_Row_Detection_Techniques/Photos --output ./result
```

Config: `carolif.py:55 CarolifConfig` — roi_top_frac=0.42, HDBSCAN min_cluster_size, RANSAC thresh 3px, slope [70,110]°.

## Stats (this run, CPU)

- 21/21 processed, 54 rows total
- avg 949 ms/img (1.1 FPS) — paper reports 108 ms at 120×80 ROI; our ROI is ~1100px (single-threaded, no GPU)
- bare-soil `bev5` correctly 0 rows via vegetation guard

