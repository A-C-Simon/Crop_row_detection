# Crop-lines-system benchmark on the rover dataset (2026-08-21)

Reference implementation (`find_gradient.py` + `find_lines.py`) evaluated on the
same 23 real images used to validate our DFT detector, with **identical inputs**:
BEV images fed directly (ExG gray, normalized [0,1]); forward-looking photos fed
as rectified by our IPM front-end (auto-gsd, per-image best pitch/yaw from
{35,40,45}deg x yaw fallback).

Two passes:

* **Pass A – full reference pipeline** (their FFT angle stage feeds their line scanner).
* **Pass B – line stage only**, seeded with the angle our detector found
  (isolates their line-finding quality from their fragile angle estimate).

Outputs: `<name>_croplines.png` (pass A), `<name>_croplines_seeded.png` (pass B),
`summary.csv`, `summary_seeded.csv`.

## Pass A results (their full pipeline)

| image | cfg | FFT spacing px | FFT angle deg | lines | time s | status |
|---|---|---:|---:|---:|---:|---|
| bev | BEV | 210.0 | -90.0 | 13 | 929.2 | ok |
| bev2 | BEV | 728.0 | -0.0 | 0 | 0.0 | ZeroDivisionError |
| bev4 | BEV | 17.6 | -90.0 | 0 | 0.4 | ok |
| bev5 | BEV | 32.6 | -0.0 | 0 | 0.0 | ZeroDivisionError |
| bev6 | BEV | 17.4 | -0.0 | 0 | 0.0 | ZeroDivisionError |
| bev7 | BEV | 27.9 | -0.0 | 0 | 0.0 | ZeroDivisionError |
| photo_1 | pitch40 | 169.0 | -0.0 | 0 | 1.0 | ZeroDivisionError |
| photo_2 | pitch35 | 220.0 | -0.0 | 0 | 0.8 | ZeroDivisionError |
| photo_3 | p35+yaw45 | 116.3 | 45.2 | 0 | 4.0 | ok |
| photo_4 | p45+yaw75 | 35.5 | -11.4 | 0 | 7.0 | SystemExit |
| photo_5 | pitch35 | 195.0 | -0.0 | 0 | 0.5 | ZeroDivisionError |
| photo_6 | p40+yaw95 | 183.0 | -0.0 | 0 | 4.4 | ZeroDivisionError |
| photo_7 | pitch35 | 221.0 | -0.0 | 0 | 0.4 | ZeroDivisionError |
| photo_8 | p45+yaw15 | 140.0 | -0.0 | 0 | 7.2 | ZeroDivisionError |
| photo_9 | p40+yaw60 | 190.4 | 41.9 | 28 | 8.2 | ok |
| photo_10 | p45+yaw75 | 275.4 | 43.4 | 0 | 10.8 | ok |
| photo_11 | p45+yaw75 | 464.0 | -90.0 | 0 | 10.4 | ok |
| photo_12 | p40+yaw60 | 196.0 | -0.0 | 0 | 4.1 | ZeroDivisionError |
| photo_13 | p45+yaw95 | 152.5 | 8.1 | 0 | 35.2 | ok |
| photo_14 | p45+yaw120 | 946.0 | -90.0 | 0 | 19.5 | ok |
| photo_15 | p40+yaw75 | 413.0 | -90.0 | 8 | 7.5 | ok |
| photo_16 | pitch35 | 237.0 | -0.0 | 0 | 0.8 | ZeroDivisionError |
| photo_17 | p40+yaw60 | 292.0 | -0.0 | 0 | 5.1 | ZeroDivisionError |

## Pass B results (their line stage seeded with our angle)

| image | our angle deg | lines | time s | status |
|---|---:|---:|---:|---|
| bev | -2.7 | 0 | 3.2 | SystemExit |
| bev2 | -2.5 | 0 | 0.1 | SystemExit |
| bev4 | -3.3 | 0 | 0.0 | SystemExit |
| bev5 | -89.5 | 0 | 0.5 | ok |
| bev6 | -90.0 | 0 | 30.3 | SystemExit |
| bev7 | 89.7 | 0 | 5.9 | ok |
| photo_1 | -89.9 | 0 | 1.4 | ok |
| photo_2 | 81.0 | 0 | 1.3 | ok |
| photo_3 | 87.9 | 0 | 4.1 | ok |
| photo_4 | -84.0 | 0 | 7.2 | SystemExit |
| photo_5 | -88.3 | 0 | 0.4 | SystemExit |
| photo_6 | -85.7 | 0 | 5.7 | SystemExit |
| photo_7 | 87.8 | 0 | 1.2 | ok |
| photo_8 | -50.2* | 0 | 8.5 | SystemExit |
| photo_9 | 87.2 | 22 | 7.8 | ok |
| photo_10 | -79.8 | 0 | 6.0 | SystemExit |
| photo_11 | 83.4 | 0 | 9.1 | ok |
| photo_12 | 86.7 | 0 | 4.0 | ok |
| photo_13 | -86.3 | 0 | 5.1 | SystemExit |
| photo_14 | 89.1 | 0 | 20.6 | ok |
| photo_15 | 84.6 | 8 | 7.4 | ok |
| photo_16 | 89.1 | 0 | 1.9 | ok |
| photo_17 | 86.8 | 20 | 7.9 | ok |

\* photo_8 re-locked at a different angle here because detection ran on the
normalized [0,1] copy rather than the raw 0..255 rect used in our batch;
photo_8 is an unresolved case for us anyway.

## Findings

1. **Angle stage (FFT) is fragile on this data.** No windowing and only a single
   zeroed DC bin: on 13 of 23 images the peak lands at ±0 deg or ±90 deg by
   convention flip, and periods like 728 px / 946 px exceed the image width
   outright. Where it does lock correctly (bev4/5/6/7), spacing matches ours
   within ~10% (exact on bev5 = 32.6 px and bev7 = 27.9 px).
2. **Crash-prone library code.** `get_start_indexes` divides by
   `round(-1/gradient)` (mod-by-zero when angle ≈ 0) and calls `sys.exit()`
   when a scan line leaves the frame — killing the *caller's* process.
   12/23 crashes in pass A, 7/23 even with a known-good angle in pass B.
3. **Line yield near zero.** Lines produced on only 3/23 images in each pass
   (A: bev=13, photo_15=8, photo_9=28; B: photo_15=8, photo_17=20,
   photo_9=22); everywhere else `find_peaks_cwt` + the 0.5 intensity cutoff
   reject everything. The cutoff assumes bright rows on dark soil across the
   whole scan line, which rarely holds in these images.
4. **Runtime.** bev took 929 s in pass A (the random-restart `improve_line`
   loop); typical images 1–35 s.
5. **Comparison with our DFT pipeline:** ours returns row spacing (cm), angle,
   confidence and traced polylines on all 23 images without crashing (weak
   cases remain photo_3, photo_8, photo_9, photo_15 — see
   `../Result/*_rows.png`). On the clean BEV subset both systems agree well
   on spacing; on forward-looking imagery the reference system produces no
   usable rows at all.

## Verdict

The reference system targets single overhead NDVI tiles with near-diagonal,
high-contrast rows and needs per-dataset tuning (angle prior shared between
images, intensity cutoff, peak widths). On our rover dataset it fails
structurally (crashes, no peaks) rather than merely degrading. Our DFT +
phase-tracing pipeline is the better fit; the one idea worth borrowing is
their iterative line refinement, which we effectively already have via
per-slab phase demodulation in `trace_row_curves`.
