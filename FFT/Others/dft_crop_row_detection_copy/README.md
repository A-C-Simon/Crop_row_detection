# DFT Crop Row Detection (Gai et al. 2024)

Python reproduction of the frequency-domain crop-row detection algorithm from

> J. Gai, H. Guo, R. Raj, L. Tang, *Robust Crop Row Detection using Discrete
> Fourier Transform for Vision-based In-Field Navigation*, Computers and
> Electronics in Agriculture, 2024.

Implements the full pipeline of Sections 2.2.2-2.2.4 plus the LQG navigation
controller of Section 3.2: Hanning windowing, FFT spectrum, band-pass + coarse
peak, Taylor-expansion sub-pixel refinement, single-point DTFT phase, row-line
geometry, and position/heading deviation estimation.

On top of the paper it adds the pieces needed to run on a real rover camera:

* automatic row-spacing estimation (no prior needed),
* an inverse-perspective rectification front-end for **forward-looking**
  cameras (tilt/pitch, height, FOV, yaw — no calibration data required),
* near-field limiting (rows are sharpest at the bottom of the photo),
* per-row **curve tracing**: rows are drawn as curvy, broken polylines that
  follow the actual crop lines instead of one global straight-line family,
* **robustness fixes** (validated on real images):
  - auto-spacing rejects degenerate texture locks (periods < 8 px),
  - curve tracing gates on *local* contrast (windowed median) so uniformly
    dim-but-trackable regions are no longer dropped, plus an absolute
    amplitude floor against pure-noise neighbourhoods,
  - `--scan-pitch` (yaw = 0) scores pitches with a Gaussian verticality
    prior: correctly rectified rows come out parallel to travel (~90 deg),
    so strong peaks on horizontal artefacts no longer win the scan,
  - overlays clip traced rows/centerline to the image frame; optional
    `--spectrum` panel marks the locked peak on the log-FFT magnitude;
  - every run prints a lock-confidence report (peak/ring prominence) with
    explicit warnings for weak (<3x) or texture-scale (<8 px) locks.

## Files

| file                   | contents |
|------------------------|----------|
| `row_detection.py`     | the core DFT row detector (windows, FFT, peak refinement, DTFT phase, geometry, deviations) + `trace_row_curves` |
| `run_real_image.py`    | CLI for real photos: BEV mode and forward-looking mode with IPM rectification, overlay rendering |
| `run_batch.py`         | batch-process a folder of images into overlays + `metrics.csv` + `summary.md` |
| `synthetic.py`         | synthetic bird's-eye crop-row image generator with known ground truth |
| `preprocessing.py`     | ExG grayscale, camera undistortion, homography IPM, MAAIR ROI extraction |
| `lqg_controller.py`    | discrete LQR + Kalman filter + closed-loop simulation |
| `test_detection.py`    | validation suite against the synthetic ground truth (9 tests) |
| `demo_example_image.py`| demo on the aerial example image (no calibration needed) |

## Requirements

Python 3.10+, numpy, scipy. Optional: OpenCV (rectification of forward-looking
images), matplotlib (overlays/demos), Pillow (loading arbitrary image formats).
The core detector only needs numpy.

## Usage

```bash
# validate the detector on synthetic images (all tests should PASS)
python3 test_detection.py

# demo on the aerial example image (writes demo_output.png)
python3 demo_example_image.py

# real image, already bird's-eye view
python3 run_real_image.py field_bev.png --scale 0.02

# real image from a forward-looking camera on the rover front
python3 run_real_image.py photo.png --pitch 40 --height 0.8 --fov 70

# automatically try several pitches and keep the best-locked one
python3 run_real_image.py photo.png --scan-pitch 20:60:5 --height 0.8

# camera yawed relative to the rows
python3 run_real_image.py photo.png --pitch 35 --yaw 85 --height 0.8

# batch-process a whole folder into ./results (overlays + metrics + summary)
python3 run_batch.py /path/to/photos --out results
```

Useful options: `--range M` keeps only the nearest M metres ahead when
rectifying (default 10, `0` = all); `--no-trace` draws only the global
straight-line fit; `--spacing PX` supplies a known row spacing in pixels;
`--out PNG` sets the overlay path.

## How the core works

1. 2D Hanning window `w(x,y) = 1/4 (1-cos 2πx/(M-1))(1-cos 2πy/(N-1))` is
   applied to the rectified grayscale ROI image (eq. 9-10) to suppress spectral
   leakage. The window-weighted mean is removed first so the window's own DC
   main lobe cannot bias nearby peaks (important when only a few row periods
   fit into the image).
2. The FFT magnitude spectrum is band-pass filtered around the row spacing
   (auto-estimated if not given) and the peak is located coarsely.
3. A Taylor-expansion Newton step on the band-passed magnitude gives a sub-pixel
   peak frequency (eq. 11-12). A fine DTFT-magnitude scan then pins the peak to
   sub-bin accuracy, and the phase is read off with a single DTFT evaluation
   (eq. 13-14).
4. The peak frequency gives the row periods `Tx = 1/fx`, `Ty = 1/fy` (eq. 15-16)
   and the line direction `l = (-Tx, Ty)/|.|` (eq. 17). The x-axis intercepts
   are `x_k = Tx (k - φ/(2π))` (eq. 18); the k range is derived from the image
   corners so every row intersecting the frame is generated.
5. Mapping the rows back to the local map frame and taking the two rows
   bracketing the robot gives the lateral deviation `e_y` and heading deviation
   `e_θ` (eq. 19-23).

### Automatic spacing estimation

With no prior, the strongest spectral peak outside the DC neighbourhood
(`fr < 2/min(M,N)` masked out — a usable pattern must repeat at least twice)
gives the band-pass centre. If the refined peak later escapes that band, the
detector re-locks once with the measured frequency as prior and keeps the
candidate with the sharper spectral prominence (bandpassed peak vs. median of
its spectral ring). The reported `row_spacing_px` is always the measured
perpendicular spacing of the locked peak, never the stale prior.

### Curve tracing (`trace_row_curves`)

The global DFT solution is a straight-line family; real rows bend. The tracer
uses that solution as a carrier:

1. resample the image into `(along-row s, across-row r)` coordinates,
2. split into overlapping slabs along `s`,
3. demodulate each slab's perpendicular profile at the row frequency `1/P`
   (single-bin DTFT): amplitude = local contrast, phase = local lateral offset,

so every row becomes a polyline `r_m(s)` that follows slow curvature. Slabs
with insufficient amplitude or too little image coverage yield NaN points,
which break the polylines exactly where rows fade or disappear. In the overlay
traced rows are solid red; the global straight fit is a faint dotted reference.

### Forward-looking mode (IPM rectification)

For a camera mounted on the rover looking down and ahead, the image is first
rectified to a metric ground grid:

* pinhole camera model, world frame X right / Y forward / Z up, camera at
  `(0, 0, h)`; `--pitch` tilts the optical axis away from nadir toward +Y,
  `--yaw` rotates the tilt direction horizontally; focal length from `--fov`;
* inverse homography warps the photo onto a grid at `--gsd` m/px (+Y forward
  points up in the rectified view);
* `--range M` discards ground farther than M metres ahead — far rows fade and
  only add noise, while the nearest rows (bottom of the photo) are sharpest;
* the valid-pixel mask is warped alongside and eroded before MAAIR cropping so
  border artefacts cannot enter the ROI.

**Sanity check:** with correct `--pitch/--height/--fov/--yaw`, detected rows
come out parallel to travel, i.e. angle ≈ ±90°, and the traced polylines hug
the bright crop bands. If the angle is near 0° or the spacing looks wrong,
adjust the parameters (`--scan-pitch` automates pitch; yaw usually needs manual
tuning). For deployment, measure the mount's pitch/height/FOV once and hard-
code them.

### Notational corrections vs. the paper

Two printed formulas in the paper are dimensionally inconsistent; we implement
the physically correct versions (see `kimi.md` for the reasoning):

* eq. 18/20 omit a factor `1/(2π)` in the x-intercepts; we use
  `x_k = Tx (k - φ/(2π))`.
* eq. 21 uses a dot product with the row direction; the signed perpendicular
  distance is `d_k = p_k · n̂` with the unit wavefront normal `n̂` (we use a
  canonical normal pointing toward the robot's right so that `e_y > 0` means the
  robot is left of the row centreline).

## LQG controller (Section 3.2)

State `x = (e_y, e_y', e_θ, e_θ')`, input = curvature κ, measurements `(ê_y, ê_θ)`:

```
A = [[1, dt, 0, 0],        B = [0]         C = [1 0 0 0]
     [0,  0, s, 0],            [0]             [0 0 1 0]
     [0,  0, 1, dt],           [0]
     [0,  0, 0, 0]]            [s]
```

The paper prints a Greek nu in the last entry of B; physically it is the
curvature-to-heading-rate gain equal to the forward speed `s` (we use `s`).

## Validation results

Synthetic images (1200x1200 px, 0.76 m rows, 5° tilt):

* spacing recovered to <0.1%, phase to ~0.05 rad, direction to ~0.1°;
* lateral deviation `e_y` recovered to ~2 mm and heading deviation `e_θ` to
  ~0.1°;
* robust under additive noise, weed blobs, missing-plant gaps and illumination
  gradients (deviations within the paper's reported field accuracy of
  6.4 cm / 1.48°);
* the LQG loop drives an initial 0.5 m / 10° disturbance to <5 cm / <3°;
* curved rows (14 px bow): tracer follows to <3 px where the global straight
  family misses by >5 px (`test_trace_curved_rows`);
* end-to-end forward-mode round trip: synthetic ground truth rectified through
  a 35° virtual camera recovers the true spacing within ~10%.

Real rover images (`../Photos`):

* all bird's-eye views lock the correct spacing and angle, traced into 8-27
  polylines each;
* forward-looking photos at pitch 40 deg, height 0.8 m recover standard row
  spacings (~51-75 cm) at ≈90 deg with visible breaks where rows fade; a
  strongly yawed shot needs `--yaw 85` to come out vertical.

## Limitations

* **Rows must be parallel and periodic.** The whole image is described by one
  global sinusoid: equal spacing, one direction. Headlands, row terminations,
  curving rows, or two field blocks at different angles violate the model —
  the fit averages over them and the band-pass may lock onto the wrong
  structure. `trace_row_curves` only follows *gentle* curvature around the
  globally locked direction; it cannot track rows that genuinely converge or
  branch.
* **Forward mode lives on uncalibrated extrinsics.** Pitch, height, FOV and
  yaw are hand-supplied guesses; a wrong value still rectifies into a
  plausible-looking bird's-eye view with confidently wrong spacing and angle.
  Pitch can be scanned automatically, but yaw currently cannot — a mount that
  yaws relative to the rows needs manual tuning per pass.
* **It needs enough periods in view, and visible ones.** Spectral resolution
  scales with aperture: when only one or two row periods fit in the ROI, peaks
  broaden and harmonics or plant-scale periodicity can out-compete the row
  fundamental (we observed a spurious 3/4-spacing lock on a tight near-field
  crop). Conversely, the signal is pure intensity contrast — once the canopy
  closes and the inter-row bands disappear, there is nothing left to detect.
