# CAROLIF — Crop Row Detection (Clustering Algorithm based RObust LIne Fitting)

Python re-implementation of the crop row detection method from the paper in
this repository:

> Khan M.N., Rahi A., Rajendran V.P., Al Hasan M., Anwar S. (2024).
> **"Real-time crop row detection using computer vision — application in
> agricultural robots."** *Frontiers in Artificial Intelligence* 7:1435686.
> doi: [10.3389/frai.2024.1435686](https://doi.org/10.3389/frai.2024.1435686)

The pipeline follows Algorithm 1 (CAROLIF) of the paper step by step.

## Pipeline (mapping to the paper)

| Step | Paper section | Implementation (`carolif.py`) |
|------|---------------|-------------------------------|
| 1.1 Region of Interest (≥ 3 rows, ~bottom of frame) | Sec. 3.1.1 | `detect()` — bottom `(1 - roi_top_frac)` part of the image |
| 1.2 Projective transformation (rows become parallel) | Sec. 3.1.1.1 | `estimate_vanishing_point`, `_calibrate_vp`, `build_homography` |
| 1.3 Segmentation: normalized-RGB ExG + Otsu | Sec. 3.1.1.2 (Eqs. 2, 9) | `normalize_exg`, `segment_green` |
| 1.4 Noise reduction: morphological opening/closing | Sec. 3.1.1.3 | `segment_green` |
| 2 HDBSCAN clustering of white pixels | Sec. 3.1.2 | `run_hdbscan`, `pool_mask_points` |
| 2.2 < 3 clusters ⇒ merged rows: iterative outlier deletion | Sec. 3.1.2 | `_cluster_with_refinements` |
| 2.3 crop-row distance check: delete weed clusters (fewer points & smaller height) | Sec. 3.1.2 / Fig. 1 | `_geometry_filter` |
| 3 RANSAC line fitting per cluster | Sec. 3.1.3 | `ransac_line` |
| 3.2 slope threshold [70°, 110°], discard the rest | Sec. 3.3 / Fig. 12 | `detect()` + `line_angle_deg` |
| 3.3 superimpose fitted lines on the crop rows | Algorithm 1 | `extend_line_full_height` + inverse homography |

### Engineering additions (kept faithful to the paper)

The paper selects the four homography boundary points **manually** per scene.
For batch processing they are estimated automatically:

* `_calibrate_vp` — grid search over candidate vanishing points; the winner is
  the one whose top view has the most vertical band boundaries
  (gradient-orientation score), followed by a local refinement.
* `_refine_vp_with_clusters` — second pass that nudges the vanishing point so
  the already-clustered rows become as vertical as possible (no re-clustering).
* `pool_mask_points` — grid pooling of the top-view mask before clustering so
  canopy shadows do not shatter row bands into speckle (also a big speed-up).
* Vegetation guard — images with no meaningful green content (bare soil) are
  rejected early via the raw ExG fraction.

## Requirements

* Python 3.10+
* `opencv-python`, `numpy`, `scikit-learn` (>= 1.3) **or** the `hdbscan`
  package, `scipy` (pulled in by sklearn)

## Usage

```bash
# run on every image in the Photos folder, write results to ./results
python3 run_carolif.py --input /home/ac/Crop_Row_Detection_Techniques/Photos \
                       --output ./results

# single image / custom pattern
python3 run_carolif.py --input path/to/dir --pattern 'bev*.png' --output out
```

Library use:

```python
import cv2
from carolif import CAROLIF, CarolifConfig

detector = CAROLIF(CarolifConfig())
res = detector.detect(cv2.imread("field.png"))
for (p1, p2) in res["lines"]:        # crop rows in original image pixels
    ...
```

## Main tuning parameters (`CarolifConfig`)

| Parameter | Default | Meaning (paper reference) |
|-----------|---------|---------------------------|
| `roi_top_frac` | 0.42 | ROI = image below this fraction (Sec. 3.1.1) |
| `min_cluster_size` / `min_cluster_size_frac` | 30 / 0.008 | HDBSCAN main tuning parameter (Sec. 3.1.2) |
| `expected_min_rows` | 3 | fewer clusters ⇒ assume merged rows, start outlier deletion |
| `merge_ratio` | 2.5 | cluster much larger than median ⇒ merged rows |
| `row_dist_factor` | 0.55 | clusters closer than this × median gap ⇒ weed cluster (Sec. 3.1.2) |
| `min_height_frac` | 0.25 | row clusters must span ≥ this share of top-view height |
| `ransac_thresh` | 3.0 px | RANSAC inlier distance (Sec. 3.1.3) |
| `slope_range` | (70°, 110°) | fitted-line slope threshold (Sec. 3.3) |

## Outputs (`results/`)

Per image, every stage of Algorithm 1 is stored:

```
<stem>_01_original.png          input image
<stem>_02_roi.png               Step 1.1  ROI crop
<stem>_03_binary_mask.png       Step 1.3/1.4  ExG+Otsu+morphology mask
<stem>_04_top_view.png          Step 1.2  projective transform (parallel rows)
<stem>_05_clusters.png          Step 2  HDBSCAN clusters (grey=dropped, black=outliers)
<stem>_06_detection_view.png    Step 3  RANSAC lines in the top view
<stem>_07_result.png            final overlay on the original image
summary.csv                     per-image row count, vanishing point, per-step timings
```

## Results on `Photos/` (23 images)

* 23/23 images processed without failure; bare-soil images (`bev5`, `bev6`)
  correctly return 0 rows via the vegetation guard.
* 56 crop rows detected in total; bird's-eye fields (`bev2`, `bev4`) and
  perspective fields (`photo_*`) get near-vertical, row-centred lines.
* Average pipeline time ≈ **0.63 s/image (~1.6 FPS)** at ~1100 px on a CPU
  (paper reports 108 ms at a 120×80 ROI on a Core i5 — our ROI is far larger
  and the code is single-threaded, unoptimised).

## Known limitations

* The paper calibrates the homography with **manually picked** boundary
  points; the automatic vanishing-point search can be a few degrees off on
  strongly curved or oblique scenes (e.g. distant hilly fields such as
  `photo_2`), which the slope filter then partially rejects.
* Dense closed canopies with thin dark drill gaps (`photo_1`) invert the
  usual crop-on-soil contrast the ExG segmentation expects; detections are
  correspondingly sparser.
* Straight-line fitting assumes rows are straight inside the ROI (paper
  Sec. 3.1.1), so strongly curved rows are only approximated.
