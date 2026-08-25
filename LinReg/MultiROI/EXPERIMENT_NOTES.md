# Experiment Notes: Weed-Robust Variants of the Multi-ROI Pipeline

**Test file**: `test_multi_roi.py` (safe copy of `multi_roi.py`; original results
untouched in `result/`)
**Period**: 2026-08-24/25 · **Dataset**: 23 shared Photos (17 ground-level,
6 aerial `bev*`)
**Reference**: Zhou et al. 2021, "Autonomous detection of crop rows based on
adaptive multi-ROI in maize fields" (see `audit.md`)

Ground truth does not exist for this dataset; "better" below means stable
corridors + eye-test approval on calibration images (photo_9 primary).

---

## 1. Strategies implemented

| Flag | Strategy | Where it acts |
|---|---|---|
| *(base)* | paper pipeline + robust nav fit | — |
| `--pre_clean` | per-strip band-support filter before detection | segmentation → detection |
| `--scorer madz` | gated MAD-z fence vs pass-1 anchors | between two passes |
| `--scorer if` (default) | gated IsolationForest vs pass-1 anchors | between two passes |
| `--morph struct` | **pre_clean embedded in segmentation** + gated scrub | inside preprocessing |
| `--morph robust --morph_fence iqr\|if` | oriented line openings + area floor + shape-attribute fence | inside preprocessing |

Shared safety machinery (all strategies): anchor keep-band, speckle floor
(`min_area=12`), giant-blob cap (>5% veg never condemned), removal valve
(abort if >70–85% of candidates erased), weed-pressure gate (`--weed_gate`
0.35), curvature guard (`--max_bend` 45° — never scrub a bent corridor).

## 2. Failure modes discovered & countermeasures

| # | Failure mode | Evidence | Countermeasure (flag) |
|---|---|---|---|
| F1 | 1-px weed sliver becomes a flanking row; ROI freezes around it | photo_9 strip-1 cluster `[419-419]`; photo_6 `[622-622]` | mass gate `--min_flank_frac 0.01` (measured: slivers ≤0.006 of strip mass, real rows ≥0.015) |
| F2 | Strip-1 midpoint at maximal leverage bends the whole nav line | photo_6 first ROI reached 3rd row → 21° | exclude strip-1 midpoint from Q fit (default; `--nav_include_first` reverts) |
| F3 | M+E threshold annihilates saturated profiles | photo_7: max(z)=47 < M+E≈48.5 → zero survivors → full-width frozen ROI | P90 fallback in `column_projection`, only when nothing survives |
| F4 | Off-structure weeds create fake clusters / pull picks | photo_10 −4° after cleaning alone | `--pre_clean` / gated scrubs |
| F5 | Pixel-level cleaning breaks rows into dashed lines | first IF run removed 91–99% veg incl. rows | always decide at connected-component level |
| F6 | Lone-cluster strips slide the nav line onto a single row | photo_11/17 nav coincided with a row (sparse young crops) | one-sided strips reuse previous window + Mx (option B) and feed det chains only on two-sided picks |
| F7 | Wide early-strip boxes feed cross-row centers into det chains | photo_17 left line near-horizontal, D=576 vs median 180 | corridor-width consistency filter (drop picks with D > 1.6×median) + robust chain fit |
| F8 | y-on-x OLS is ill-conditioned on near-vertical dot clouds | photo_8: dots form a vertical line, OLS claimed 87°; drawn lines visually not best-fit | all line fits (nav + chains) switched to total least squares with perpendicular-space rejection |
| F9 | Dots from unworthy strips still voted on navigation | same wide boxes barred from chains had green nav dots | evidence-gated Q: only two-sided, plausible-span midpoints fit; synthetic/wide dots red |
| F10 | Straight best-fit cuts across curving furrows | user safety requirement for curved corridors | nav path = smoothed cubic regression spline clamped to ±15% of median corridor span around the straight TLS reference; base-tangent CSV metric |

**Rejected experiments**: predictive convergence clamp (see above);
single-pass inline anchoring; closing in robust morphology; interpolating
(unpenalized) splines — smoothing + envelope chosen instead.

**Final default (2026-08-25)**: struct morph + option-B climb + mass gate +
evidence-gated TLS navigation + envelope spline curve + filtered TLS det
chains. Post-scrub opt-in only.

**Rejected experiment — predictive convergence clamp** (2026-08-25): narrowing
upper-strip windows along extrapolated row trajectories produced textbook box
convergence on forward images (photo_1 widths 124→29 px) and was a no-op on
aerials, but shifted 17 corridors including large regressions (photo_10
12.7→33.0°, photo_6 5.4→9.2°). Removed; code not retained.

## 3. Results snapshot (final angles, degrees from vertical)

| Image | orig | pre_clean | madz gated | morph+IQR | morph+IF | morph struct |
|---|---|---|---|---|---|---|
| photo_9 ✓eye | 7.79 | **9.02** | 8.99 | 1.44 ⚠ | 10.73 | 9.02 |
| photo_6 ✓eye | 3.76 | 5.42 | 5.42 | 3.67 | 3.67 | 5.44 |
| photo_7 fixed (F3) | 8.96 | 9.30 | 9.69 | 13.65 ⚠ | 13.65 ⚠ | 9.31 |
| photo_11 | 1.68 | 2.50 | 1.63 | 1.52 | 1.52 | 2.50 |
| photo_14 | 7.14 | 7.56 | 7.56 | 12.31 ⚠ | 3.89 ⚠ | 7.31 |
| photo_5 🚨 | 17.74 | 18.51 | 18.51 | 56.50 | 56.50 | 18.51 |
| bev6 | 11.48 | 7.54 | 7.73 | 18.84 ⚠ | 4.06 | 4.33 |
| bev 🚨 | 3.72 | 3.73 | 3.73 | 3.70 | 10.93 | 3.73 |
| photo_2 ⚠ | 19.11 | 28.03 | 22.01 | 34.84 | 34.84 | 28.03 |
| photo_8 ⚠ | 3.37 | 7.75 | 7.75 | 8.18 | 8.18 | 7.75 |

Mean runtime: base ~11 ms · madz gated ~16 ms · pre_clean ~50 ms ·
morph+IF ~216 ms · morph struct ~45 ms.

## 4. Findings

1. **pre_clean is the strongest strategy** (user eye test). Mechanism:
   judge each blob against per-strip row-candidate bands (Eq. 5–8 applied
   destructively) *before* any corridor exists — wrong-lane capture becomes
   impossible because decoys are gone at strip 1.
   `--morph struct` embeds exactly this in segmentation and adds the gated
   scrub on top (bev6 7.54→4.33); it inherits pre_clean's numbers elsewhere.
2. **Closing hurts, opening only**: user-confirmed. Closing welded weed
   masses into row bands (photo_6 distortion); all kept variants are
   opening-only. The paper's single anti-diagonal opening silently shaves
   non-`╱`-oriented row segments; union-of-oriented-openings fixes that but
   fragments dense scenes (photo_5) — see OQ-1.
3. **Fence choice is scene-dependent (no free lunch)**: IQR vs IF inside
   robust-morph flipped photo_9 (IF better) and bev (IQR better).
   Univariate fences fail on multi-row distance plateaus (MAD-z v1 cut-off
   landed beyond physical range: median d=0.20, MAD=0.11 over full frames).
4. **Statistical filters need the right reference population**. Global
   frame statistics describe other rows' geometry, not crop-vs-weed.
   Per-strip bands (pre_clean) or corridor-relative distances (gated
   scorers) provide valid populations.
5. **Component-level decisions are mandatory** (F5): every pixel-shaving
   variant produced broken rows.

## 5. Open questions (need labeled data / eye tests)

- **OQ-1**: photo_5 fragmentation under oriented openings (56° persists in
  every fence variant) — openings-stage issue, not fence issue.
- **OQ-2**: photo_2 curved field: baseline itself may be wrong (19°);
  variants give 22–35°. Bend guard prevents scrubbing; a curved-aware
  corridor model (piecewise/polynomial) is the principled fix.
- **OQ-3**: photo_8/bev6 post-nav-skip relocations (~±4°) — plausible
  outlier-midpoint corrections, unverified.
- **OQ-4**: bev6 degeneracy (no visible furrow, README-noted) — all
  strategies disagree (2.7–18.8°); candidate for exclusion or special
  handling in evaluation.

## 6. Result directories & reproduction commands

**DEFAULTS (since 2026-08-25)**: `--morph struct` is on by default; the
anchor scrub is opt-in via `--post_scrub` (rejected as default after eye
tests showed it deleting real crop rows on photo_11/17).

| Directory | Process | Old name |
|---|---|---|
| `result_01_baseline_paper/` | untouched multi_roi.py output | result |
| `result_02_DEFAULT_struct/` | **current default**: struct morph, no scrub | result_default |
| `result_03_struct_plus_madz/` | struct morph + gated MAD-z scrub | result_mstruct |
| `result_04_if_scrub_papermorph/` | paper morph + gated IF scrub | result_test |
| `result_05_madz_scrub_papermorph/` | paper morph + gated MAD-z scrub | result_madz |
| `result_06_robustmorph_iffence/` | oriented openings + embedded IF fence | result_morph |

Deleted as redundant/rejected: `result_structonly` (identical to 02),
`result_preclean` (legacy flag, superseded by struct default),
`result_sp` (rejected single-pass experiment; see §4/F5 notes).

```bash
python multi_roi.py     --input ../../Photos --results_dir ./result_01_baseline_paper
python test_multi_roi.py --input ../../Photos --results_dir ./result_02_DEFAULT_struct
python test_multi_roi.py --input ../../Photos --results_dir ./result_03_struct_plus_madz --post_scrub --scorer madz
python test_multi_roi.py --input ../../Photos --results_dir ./result_04_if_scrub_papermorph --morph paper --post_scrub
python test_multi_roi.py --input ../../Photos --results_dir ./result_05_madz_scrub_papermorph --morph paper --post_scrub --scorer madz
python test_multi_roi.py --input ../../Photos --results_dir ./result_06_robustmorph_iffence --morph robust --post_scrub [--morph_fence iqr]
```

Single-pass inline anchoring was also tested and rejected: slower (per-strip
forest refits) and photo_10 hijacked to 32–38° via noisy inherited anchors;
code was removed after the experiment.
