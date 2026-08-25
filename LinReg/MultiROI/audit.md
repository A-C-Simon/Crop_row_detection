# Audit: MultiROI Implementation vs. Zhou et al. 2021

**Paper**: Zhou Y, Yang Y, Zhang B L, Wen X, Yue X, Chen L Q. "Autonomous detection
of crop rows based on adaptive multi-ROI in maize fields." *Int J Agric & Biol Eng*,
2021; 14(4): 217–225. DOI: [10.25165/j.ijabe.20211404.6315](https://doi.org/10.25165/j.ijabe.20211404.6315)

**Audited file**: `multi_roi.py`
**Date**: 2026-08-24

---

## What's correct

| Step | Paper section | Status |
|---|---|---|
| Edge cropping | Sec 2.2 | Correct (2% border) |
| ExG → Otsu → morph open | Sec 2.2.1–2.2.2 | Correct for `--index raw` |
| Kernel K (4×4 anti-diagonal) | Eq. 3 | Exact match |
| N=10 horizontal strips | Sec 2.3.1 / Eq. 4 | Correct |
| Column projection Z(i) | Eq. 5 | Correct |
| Filter T = M + E | Eq. 6–8 | Correct |
| Clustering (gap < L) | Sec 2.3.2 | Correct |
| Flanking clusters CLeft/CRight | Sec 2.3.2 | Correct |
| D = CRight.max − CLeft.min | Eq. 9 | Correct |
| ROI = [base_lo − 0.1D, base_hi + 0.1D] | Fig 8b | Correct |
| Window reuse on empty strip | Sec 2.3.3 | Correct |
| Strip loop bottom→top | — | Correct |
| Detection lines skip strip 1 | Section 3 | Correct (matches README) |

---

## Intentional deviations (documented)

1. **Default ExG**: Uses raw `2G−R−B` instead of the paper's normalized Eq. 1–2.
   Documented in README as a robustness improvement — reasonable.

2. **Midpoint uses cluster *centers*** instead of outer edges (Eq. 10). Prevents
   wide/merged clusters from yanking the corridor. Reasonable engineering choice.

3. **Outlier rejection** on the navigation line fit (`_fit_line_robust`): up to 3
   points rejected at 2.5σ. The paper does not mention this — it is a pure addition
   for robustness. Fine.

---

## Bugs / Issues found

### 1. `--index normalized` ExG scaling clips the negative tail (bug)

```python
exg8 = (np.clip(exg, 0.0, 1.8) / 1.8 * 255.0).astype(np.uint8)
```

The paper's Eq. 2 gives ExG ∈ [−1.8, 1.8]. The code clips the *negative* tail to 0,
destroying contrast between soil (negative ExG) and shadow/transition pixels. The paper
likely maps the full range linearly to 0–255, not just the positive half. On images with
strong soil signal, this collapses most of the histogram near 0 and pushes Otsu into a
degenerate threshold. This is why the normalized mode fails on aerial views — but the
bug makes it fail on more images than it should.

**Suggested fix**: Replace the clipping with a full-range normalization:

```python
exg8 = ((exg + 1.8) / 3.6 * 255.0).astype(np.uint8)  # maps [-1.8, 1.8] → [0, 255]
```

### 2. Vertical-line numerics in `_fit_line_robust` (edge case)

`_fit_line` returns `w_slope = 1e6` for perfectly vertical point sets (straight
corridors). The outlier rejection in `_fit_line_robust` then computes
`resid = np.abs(ys - (w_slope * xs + b))`. With `w_slope = 1e6`, the term
`1e6 * xs` can overflow for large x, producing inf residuals. NumPy handles inf
without crashing, but this could incorrectly reject valid points on vertical corridors.

### 3. Degenerate aerial images (`bev7` → 0 detection lines, angle 0.000°)

`bev7` produces 0 detection lines and a navigation angle of exactly 0.000° with 10
nav points, suggesting the corridor collapsed to a degenerate case. The aerial view
likely has no visible soil corridor at the default `--l_frac 0.05`. The README
suggests `--l_frac 0.03` for this image, but this is not handled automatically.

### 4. Curved rows exceed straight least-squares model (`photo_2` → 19.1°, 1 detection line)

Acknowledged in the README. The least-squares fit `y = wx + b` cannot represent curved
corridors. The paper's own accuracy is 95.3% on 1920×1080 images, and curved fields
are the main failure mode. Not a bug, but a fundamental limitation.

---

## Summary

| Category | Count |
|---|---|
| Faithfully implemented | ~90% of the algorithm |
| Intentional deviations (documented) | 3 |
| Actual bugs | 1 (ExG normalization clipping) |
| Edge-case issues | 2 (vertical line numerics, degenerate aerial) |
