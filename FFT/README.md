# DFT Crop Row Detection

Implementation of the crop row detection algorithm from:

J. Gai, Z. Guo, A. Raj, L. Tang, "Robust Crop Row Detection using Discrete
Fourier Transform (DFT) for Vision-based In-Field Navigation",
Computers and Electronics in Agriculture, 2024.
https://www.sciencedirect.com/science/article/pii/S0168169924010573

The idea of the paper in one sentence: crop rows are planted parallel and
equally spaced, so in the frequency domain they produce one dominant peak
whose radius gives the row spacing, whose angle gives the row direction and
whose phase gives the row positions. Finding that peak is cheaper and more
robust against weeds, glare and missing plants than detecting individual
plants and fitting lines to them.

## Files

| File | Purpose |
|---|---|
| `dft_crop_row_detector.py` | Core algorithm: ExG, inverse perspective mapping, MAAIR, Hanning window, FFT, bandpass, Taylor peak refinement, DTFT phase, row geometry, deviations |
| `run_fft_detection.py` | Batch runner for a folder of photos, writes one figure per photo plus `metrics.csv` |
| `run_fft_video.py` | Runner for video files and camera streams, writes an annotated overlay video plus a per frame CSV |
| `results_fft/` | All results (figures, `metrics.csv`, video outputs) |

## Requirements

Python 3.10 with numpy, opencv-python, matplotlib, Pillow. Tested with
numpy 1.26 and OpenCV 4.11.

## How the algorithm works

```
 input frame (RGB)
        |
        v
 [1] ExG grayscale ................ 2G - R - B, percentile normalized
        |
        v
 [2] Inverse perspective mapping .. homography to a metric bird's eye grid,
        |                          valid mask, MAAIR crop  =>  ROI
        v
 [3] Hanning window + 2D FFT ..... windowed ROI -> magnitude spectrum
        |
        v
 [4] Bandpass filter ............. keep frequencies whose period matches
        |                          plausible row spacings (annulus mask)
        v
 [5] Coarse peak ................. argmax of masked magnitude  -> (Px, Py)
        |
        v
 [6] Taylor refinement ........... 2nd order fit, sub-bin offset  -> (fx, fy)
        |
        v
 [7] DTFT phase .................. single complex sum at (fx, fy) -> phi
        |
        v
 [8] Row geometry ................ direction t, axis crossings x_i, lines
        |
        v
 [9] Deviations .................. e_y (lateral), e_theta (heading)
```

What each block does and why it is there:

1. **ExG grayscale.** Vegetation is highlighted with the Excess Green index
   `ExG = 2*G - R - B`, then contrast normalized between the 1st and 99.5th
   percentile. This makes the pipeline insensitive to overall brightness and
   color casts. Function: `exg_gray`.

2. **Bird's eye rectification.** The DFT model behind the method assumes
   parallel, equally spaced rows. A perspective photo violates this: rows
   converge to a vanishing point. The frame is therefore warped to a metric
   ground grid with a homography built from a pinhole camera model
   (`camera_homography`) using pitch, height, field of view and yaw. Pixels
   with no source in the original image are masked out and the ROI is cut to
   the largest useful axis aligned rectangle (`maair`). Since the test photos
   come with no calibration data, the pitch and yaw are found by scanning
   (see the photo runner section below). Function: `rectify_forward`.

3. **Hanning window and FFT.** The image border acts as a rectangular
   window and spreads energy across the spectrum (spectral leakage), which
   would bury the row peak. A 2D Hanning window, the outer product of two
   1D Hanning windows, suppresses that (paper Eq. 9 and 10). A 2D FFT then
   produces the magnitude spectrum. Function: `DFTRowDetector.detect`.

4. **Bandpass filter.** Only frequencies whose period lies between
   `--min-spacing-m` and `--max-spacing-m` (converted to pixels with the
   ground sample distance) are kept, optionally narrowed around a known row
   spacing with `--spacing-prior-m`. This kills the DC component, slow
   shading gradients and fine weed texture in one step (paper Section
   2.2.3, step 1).

5. **Coarse peak.** The largest magnitude inside the band, searched in the
   half plane `fx > 0` so each spatial frequency is visited once.

6. **Taylor refinement.** The FFT bin spacing is too coarse for centimeter
   level row positions. A second order Taylor expansion around the coarse
   peak, with numerical first and second derivatives, gives a sub-bin offset
   by solving `H * delta = -grad` (paper Eq. 11 and 12). The result is the
   continuous frequency `(fx, fy)` in cycles per pixel.

7. **DTFT phase.** The FFT phase at a bin is only valid at the bin center,
   so the phase of the exact refined frequency is computed with a direct
   DTFT, one complex weighted sum over the windowed image (paper Eq. 13).
   Its angle is the phase `phi` (paper Eq. 14).

8. **Row geometry.** The row pattern is a plane wave. Lines of constant
   phase are the plant rows. Their direction is perpendicular to `(fx, fy)`
   and their crossings of the dominant axis are `x_i = (k - phi/2pi) / f`
   (paper Eq. 15 to 18). Functions: `_row_positions`, `line_endpoints`.

9. **Deviations.** The reference point (bottom center of the ROI, the
   position the robot is assumed to occupy) is compared with every detected
   line. The signed perpendicular distances are sorted and the two rows
   flanking the reference are taken. Their mean signed distance is the
   lateral deviation `e_y`, their corridor width is a sanity check, and the
   row direction angle relative to the image vertical is the heading
   deviation `e_theta` (paper Eq. 21 to 23). Distances are converted to
   meters with the ground sample distance.

The LQG controller of the paper (Section 3.2) is not part of this code, the
output stops at the two deviations, which is exactly what such a controller
would consume.

## Running on photos

```
python3 run_fft_detection.py
```

This processes every image in `/home/ac/Crop_Row_Detection_Techniques/Photos`
and writes the results to `results_fft/`. Photos whose name starts with
`bev` are treated as already top-down; everything else goes through the
rectification path.

Useful options:

```
--photos DIR            input folder
--out DIR               output folder (default results_fft)
--height 1.0            assumed camera height in m (rectification)
--fov 70.0              assumed vertical field of view in deg
--scan 20:60:5          pitch candidates lo:hi:step
--yaw-scan -30:30:10    yaw candidates, "none" disables
--range 10.0            keep only ground within this distance ahead
--min-spacing-m 0.15    smallest row spacing considered
--max-spacing-m 2.0     largest row spacing considered
--spacing-prior-m M     narrow the band around a known spacing
--bev-width-m 26.0      assumed real width of top-down images (metric scale)
```

How the pitch and yaw scan picks the rectification: every candidate pair is
rectified and detected, then scored by `prominence * exp(-0.5*(dev/sigma)^2)`
where prominence is the spectral peak contrast and dev is how far the
detected rows lean away from vertical. Rows that cannot be made vertical by
any yaw (curved or radiating rows) fall back to the strongest spectral
pattern and are flagged in the CSV. The scan dominates the runtime, roughly
2 s per photo; the detection itself takes 25 to 45 ms.

## Running on video

```
python3 run_fft_video.py /path/to/video.mp4 --out results_fft
python3 run_fft_video.py 0 --show          # webcam 0 with live preview
```

Video differs from the photo batch in three ways:

1. **Calibrate once.** The pitch and yaw scan runs a single time, on the
   first frame. Every following frame reuses those rectification parameters,
   so a frame costs about 80 ms instead of seconds. If the camera mount
   moves during recording, recalibrate by restarting on a representative
   frame.

2. **Temporal smoothing.** `e_y` and `e_theta` are smoothed with an
   exponential moving average, `--smooth` sets the weight of the new
   measurement (default 0.35, lower means smoother). Angles are smoothed on
   the unit circle so the ±180 deg wrap cannot cause jumps.

3. **Hold on failure.** If a frame cannot be rectified or detected (motion
   blur, frame with no ground), the last valid detection is held and the
   row is marked `HELD` in the CSV instead of dropping the frame.

Options beyond the photo runner:

```
--stride N        process every Nth frame (output video plays at fps/N)
--max-frames N    stop after N processed frames
--smooth A        EMA weight, 0..1
--show            live preview window, q quits
--no-video        skip writing the overlay mp4
```

Outputs per video, written to `--out`:

- `<name>_overlay.mp4`: side by side panels, original frame, rectified bird's
  eye view with detected rows in red, the navigation centerline in cyan and
  the reference point as a yellow star, plus a text panel with the current
  metrics.
- `<name>_video_metrics.csv`: one row per processed frame with frame index,
  timestamp, rectification parameters, row spacing, row count, smoothed
  `e_y` in cm, smoothed `e_theta` in deg, prominence and status.

## Outputs of the photo runner

For every photo, `results_fft/<name>_result.png` contains four panels:

1. the input photo,
2. the rectified bird's eye ROI with all detected row lines in red, the
   navigation centerline (center of the two rows flanking the reference
   point) in cyan, and the reference point as a yellow star,
3. the log magnitude spectrum with the bandpass annulus and the locked peak
   with its mirror marked,
4. a text summary of all numbers.

`results_fft/metrics.csv` has one row per photo: mode, chosen pitch and yaw,
ground sample distance, ROI size, row spacing in px and m, row angle
deviation from vertical, peak frequencies, DTFT phase, row count, `e_y` in
cm, `e_theta` in deg, corridor width, prominence, detection time and status.

Status values: `OK`, `weak lock` (prominence below 3, treat with care) and
`angle>30deg` (the scene has no near-vertical row direction, usually curved
or radiating rows on hillsides).

## Metric scale caveats

The test photos have no calibration data, so metric numbers rest on the
assumed camera model: height 1.0 m, vertical fov 70 deg. If the real camera
differed, spacings and `e_y` scale roughly linearly with the height error,
while angles and row counts stay correct. For a real robot, replace the
pinhole guess in `camera_homography` with the calibrated intrinsic matrix
and the measured camera pose, as done in the paper, and the metric outputs
become absolute.

## Verification

The core was checked on synthetic images with known spacing, angle and
phase: recovered spacing matched to 0.005 percent, line positions to 0.1 px,
and the detected lines sit on the brightness crests of the synthetic rows.

## Known limitations

- Rows must be reasonably straight and parallel inside the ROI. Strongly
  curved scenes (hillside vineyards, radiating broadcast rows) have no
  single global direction and are flagged rather than silently wrong.
- The bandpass needs at least about three row periods inside the ROI.
- Metric accuracy depends on the assumed camera model, see above.
- The reference point is fixed at the bottom center of the ROI. With real
  extrinsics the paper maps the ROI back to the robot frame instead.
