"""Run the DFT row detector on a real image.

Two modes:

* BEV (default): the image is already a bird's-eye view (camera facing
  straight down). Rows must be parallel and periodic.

* Forward-looking (--pitch > 0): the camera is mounted on the rover looking
  down and ahead. The image is rectified to a metric ground grid with an
  inverse-perspective homography built from --pitch/--height/--fov (pinhole
  model, no calibration data), cropped to the largest fully-sampled rectangle
  (MAAIR) and detected there.

Usage:
    python3 run_real_image.py IMAGE [options]

Options:
    --pitch DEG      camera tilt away from straight-down (0 = BEV, default)
    --scan-pitch L:H:S  try several pitches, keep the best-locked one
    --height M       camera height above ground (default 1.0 m)
    --fov DEG        vertical field of view of the camera (default 70 deg)
    --yaw DEG        camera yaw relative to the rows (default 0)
    --gsd M          rectified ground sample distance in m/px
                     (default: height/400)
    --range M        forward mode: rectify only the nearest M metres ahead
                     (rows are sharpest near the camera; default 10, 0 = all)
    --no-trace       draw only the global straight-line fit, skip per-row
                     curve tracing
    --scale M        metres per pixel for BEV mode, used only to report
                     deviations in metres (default: guess 26 m image width)
    --spacing PX     prior row spacing in pixels (default: auto-estimate)
    --tolerance F    bandpass relative tolerance (default 0.15)
    --out PNG        overlay output path (default: <image>_rows.png)
    --spectrum       add a log-FFT panel with the locked peak to the overlay
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import exg_gray, roi_to_map_transform, maair
from row_detection import (RowDetector, fft_spectrum, bandpass_mask,
                           trace_row_curves)

try:
    import cv2
except ImportError:
    cv2 = None


def load_gray(path: str) -> np.ndarray:
    from PIL import Image
    arr = np.asarray(Image.open(path), dtype=float)
    if arr.ndim == 3:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return exg_gray(arr[..., :3])
    return arr


def camera_homography(shape, pitch_deg: float, height_m: float,
                      fov_y_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Ground->image homography for a pinhole camera at (0, 0, height).

    World frame: X right, Y forward, Z up; ground plane Z=0. The optical axis
    points straight down at pitch 0 and tilts toward +Y (forward) with
    increasing pitch; yaw rotates the tilt direction horizontally.
    """
    m, n = shape
    f = (m / 2.0) / np.tan(np.radians(fov_y_deg) / 2.0)
    K = np.array([[f, 0, n / 2.0], [0, f, m / 2.0], [0, 0, 1.0]])
    th = np.radians(pitch_deg)
    c, s = np.cos(th), np.sin(th)
    B = np.array([[1.0, 0.0, 0.0],
                  [0.0, -c, -s],
                  [0.0, s, -c]])
    ps = np.radians(yaw_deg)
    cy, sy = np.cos(ps), np.sin(ps)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    R = B @ Rz.T
    t = -R @ np.array([0.0, 0.0, height_m])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def rectify_forward(gray: np.ndarray, pitch_deg: float, height_m: float,
                    fov_y_deg: float, gsd: float | None, yaw_deg: float = 0.0,
                    range_m: float | None = None):
    """IPM rectification of a forward-looking image to a ground grid.

    Returns (rect, offset, gsd) where `rect` is the MAAIR-cropped bird's-eye
    view, `offset` the (row, col) of the crop inside the full rectified grid,
    and `gsd` the ground sample distance actually used (m/px): pass `None` to
    pick it automatically from the source image's own sampling density, which
    prevents the far field from being magnified past its pixel content.
    `range_m` limits the rectification to the nearest ground (small Y =
    bottom of the photo), where rows are sharpest; far-field rows fade and
    only add noise. None/0 keeps the whole visible ground.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --pitch rectification")
    H = camera_homography(gray.shape, pitch_deg, height_m, fov_y_deg, yaw_deg)
    Tinv = np.linalg.inv(H)

    # Ground footprint of the image: project a dense sample of the four
    # edges (four bare corners are fragile - they can tie in Y or drop out
    # and collapse the bounding box to nothing).
    t = np.linspace(0.0, 1.0, 64)
    m_, n_ = gray.shape
    edge_u = np.concatenate([t, np.ones_like(t), 1 - t, np.zeros_like(t)])
    edge_v = np.concatenate([np.zeros_like(t), t, np.ones_like(t), 1 - t])
    pts = Tinv @ np.vstack([edge_u * (n_ - 1), edge_v * (m_ - 1),
                            np.ones_like(edge_u)])
    Xc, Yc = pts[0] / pts[2], pts[1] / pts[2]
    ok = np.isfinite(Xc) & np.isfinite(Yc) & (np.abs(Xc) < 80)
    ymax_lim = 80.0 if not range_m or range_m <= 0 else float(range_m)
    ok &= (Yc > 0.15) & (Yc < ymax_lim)
    if ok.sum() < 8:
        raise ValueError("--range too small: no ground within that distance")
    xmin, xmax = np.percentile(Xc[ok], [1, 99])
    ymin, ymax = np.percentile(Yc[ok], [1, 99])

    if not gsd or gsd <= 0:
        # Match the grid to the source's own sampling density: project the
        # centre column onto the ground and measure metres-per-source-pixel
        # across the kept span. Without this, distant ground (coarse in the
        # photo) gets blown up far past its pixel content and looks smeared.
        rr = np.linspace(0, m_ - 1, 400)
        pts = Tinv @ np.vstack([np.full_like(rr, n_ / 2.0),
                                rr, np.ones_like(rr)])
        Yg = pts[1] / pts[2]
        sel = np.isfinite(Yg) & (Yg > max(ymin, 0.15)) & (Yg < ymax)
        if sel.sum() > 2:
            # bias toward the coarse far-field sampling so NO region of the
            # rectified image is magnified past its source pixel content
            # (a finer grid only adds interpolation smear there)
            step = float(np.percentile(
                np.abs(np.gradient(Yg[sel], rr[sel])), 80))
            gsd = float(np.clip(step * 1.2, 0.004, 0.02))
        else:
            gsd = height_m / 400.0

    # Fit the whole span into <=2000 px per axis by coarsening the grid
    # rather than silently cropping the field of view.
    gsd = max(gsd, (xmax - xmin) / 2000.0, (ymax - ymin) / 2000.0)
    W = int(np.ceil((xmax - xmin) / gsd))
    Ht = int(np.ceil((ymax - ymin) / gsd))
    if W < 16 or Ht < 16:
        raise ValueError("rectified grid degenerate "
                         f"({W}x{Ht} px); check --pitch/--height/--fov")
    A = np.array([[1 / gsd, 0, -xmin / gsd],
                  [0, -1 / gsd, ymax / gsd],
                  [0, 0, 1.0]])                    # +Y forward becomes up
    M = A @ Tinv                                    # image px -> rect px

    src = np.clip(gray, 0, 255).astype(np.uint8)
    full = cv2.warpPerspective(src, M, (W, Ht),
                               flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpPerspective(np.full(gray.shape[:2], 255, np.uint8),
                                M, (W, Ht), borderValue=0) > 0
    # erode so border pixels (warp edge artefacts, partial samples) cannot
    # enter the ROI: they form strong artificial edges/bands
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    r0, c0, h, w = maair(cv2.erode(valid.astype(np.uint8), k) > 0)
    if h == 0 or w == 0:
        raise ValueError("rectification produced no valid ground pixels; "
                         "check --pitch/--height/--fov")
    return full[r0:r0 + h, c0:c0 + w], (r0, c0), float(gsd)


def overlay(det: RowDetector, res, img: np.ndarray, scale: float,
            out_path: str, title: str, traces=None, spectrum=False):
    m, n = img.shape
    if spectrum:
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        mag, fx, fy, _ = fft_spectrum(img)
        logmag = np.log1p(mag)
        axes[2].imshow(logmag, cmap="magma", extent=[fy[0], fy[-1],
                                                     fx[-1], fx[0]])
        fnorm = np.hypot(res.fx, res.fy)
        if fnorm > 0:
            for sgn in (1, -1):     # mark the locked peak and its mirror
                axes[2].plot(sgn * res.fy, sgn * res.fx, "c+", ms=14, mew=2)
        axes[2].set_title("log FFT magnitude (+ = locked peak)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input (ExG)")
    ax = axes[1]
    ax.imshow(img, cmap="gray")
    for line in det.row_lines(res):
        ax.plot([line[1], line[3]], [line[0], line[2]],
                color="0.35", ls=":", lw=1.2, zorder=2)
    if traces:
        # NaN gaps inside each polyline break it naturally where rows fade;
        # clip to the image so near-border points never draw outside the axes
        for poly in traces:
            p = poly.copy()
            ok = np.isfinite(p[:, 0])
            p[ok, 0] = np.clip(p[ok, 0], 0, m - 1)
            p[ok, 1] = np.clip(p[ok, 1], 0, n - 1)
            ax.plot(p[:, 1], p[:, 0], "r-", lw=2, zorder=3)
        title += " - red: traced rows, dotted: global fit"
    d = np.asarray(res.d_k)
    lx, ly = res.line_dir
    if len(d):
        i = np.argsort(np.abs(d))[:2]
        d_mid = d[i].mean()
        c = np.array([n / 2, m / 2]) + d_mid / scale * np.array([ly, -lx])
        half = float(np.hypot(m, n)) / 2 + abs(d_mid) / scale
        t = np.array([-half, half])
        seg = [(c[0] + t[0] * lx, c[1] - t[0] * ly),
               (c[0] + t[1] * lx, c[1] - t[1] * ly)]
        # Liang-Barsky-style clip of the centreline to the frame
        x0, y0 = seg[0]
        x1, y1 = seg[1]
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p_, q_ in ((-dx, x0), (dx, (n - 1) - x0),
                       (-dy, y0), (dy, (m - 1) - y0)):
            if p_ == 0:
                if q_ < 0:
                    t0, t1 = 1.0, 0.0
                    break
            else:
                r_ = q_ / p_
                if p_ < 0:
                    t0 = max(t0, r_)
                else:
                    t1 = min(t1, r_)
        if t0 <= t1:
            ax.plot([x0 + t0 * dx, x0 + t1 * dx],
                    [y0 + t0 * dy, y0 + t1 * dy], "c-", lw=2, zorder=4)
    ax.set_title(title)
    for a in axes[:2]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(-n * 0.02, n * 1.02)
        a.set_ylim(m * 1.02, -m * 0.02)
    if spectrum:
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved overlay to {out_path}")


def pitch_confidence(rect: np.ndarray, det: RowDetector, res) -> float:
    """Peak sharpness of the bandpassed spectrum (higher = rows locked in)."""
    mag, fx, fy, _ = fft_spectrum(rect)
    mask = bandpass_mask(fx, fy, res.row_spacing_px, det.spacing_tolerance)
    return float((mag * mask).max() / max(mag[mag > 0].mean(), 1e-9))


def pitch_score(rect: np.ndarray, det: RowDetector, res,
                sigma_deg: float) -> tuple[float, float]:
    """Scan score = spectral confidence x Gaussian verticality prior.

    With yaw = 0 a correctly rectified image has rows parallel to travel,
    i.e. angle ~ +/-90 deg; a strong peak on horizontal structure is far
    more likely field-edge/shading artefacts than rows. Returns
    (score, raw_confidence).
    """
    conf = pitch_confidence(rect, det, res)
    ang = abs(res.row_angle_deg)
    dev = 90.0 - ang                     # 0 deg when rows are vertical
    weight = float(np.exp(-0.5 * (dev / sigma_deg) ** 2))
    return conf * weight, conf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="camera tilt away from straight-down, degrees")
    ap.add_argument("--scan-pitch", default=None, metavar="LO:HI:STEP",
                    help="try several pitches and keep the best-locked one, "
                         "e.g. --scan-pitch 20:60:5")
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="camera yaw relative to the rows, degrees")
    ap.add_argument("--gsd", type=float, default=None,
                    help="rectified m/px (forward mode); default: auto-match "
                         "to the source image sampling density")
    ap.add_argument("--range", type=float, default=10.0, dest="range_m",
                    help="forward mode: keep only this many metres ahead "
                         "(near rows are sharpest, far rows fade); "
                         "0 keeps everything")
    ap.add_argument("--no-trace", action="store_true",
                    help="skip per-row curve tracing (global lines only)")
    ap.add_argument("--scale", type=float, default=None,
                    help="BEV mode: metres per pixel for metric reporting")
    ap.add_argument("--spacing", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--spectrum", action="store_true",
                    help="add a log-FFT-magnitude panel with the locked "
                         "peak marked to the overlay")
    ap.add_argument("--verticality-sigma", type=float, default=12.0,
                    help="pitch scan (yaw=0 only): degrees of row-angle "
                         "deviation from vertical at which the verticality "
                         "prior drops to exp(-0.5); 0 disables the prior")
    args = ap.parse_args()

    img = load_gray(args.image)

    if args.scan_pitch:
        lo, hi, step = map(float, args.scan_pitch.split(":"))
        prefer_vertical = (args.yaw == 0.0) and args.verticality_sigma > 0
        best = None
        for p in np.arange(lo, hi + 1e-9, step):
            try:
                r, _, _ = rectify_forward(img, p, args.height, args.fov,
                                          args.gsd, args.yaw, args.range_m)
                d = RowDetector(row_spacing_px=args.spacing,
                                spacing_tolerance=args.tolerance)
                res_p = d.detect(r)
                if prefer_vertical:
                    score, conf = pitch_score(r, d, res_p,
                                              args.verticality_sigma)
                else:
                    score = conf = pitch_confidence(r, d, res_p)
            except Exception as e:
                print(f"  pitch {p:5.1f}: skipped ({e})")
                continue
            extra = (f", score {score:8.1f}" if prefer_vertical else "")
            print(f"  pitch {p:5.1f}: spacing {res_p.row_spacing_px:6.1f} px, "
                  f"angle {res_p.row_angle_deg:7.2f}, confidence {conf:8.1f}"
                  f"{extra}")
            if best is None or score > best[0]:
                best = (score, p)
        if best is None:
            raise SystemExit("no usable pitch in scan range")
        args.pitch = float(best[1])
        print(f"selected pitch {args.pitch:.1f} deg "
              f"(score {best[0]:.1f}"
              f"{', verticality-weighted' if prefer_vertical else ''})")

    if args.pitch > 0:
        img, (r0, c0), gsd = rectify_forward(img, args.pitch, args.height,
                                             args.fov, args.gsd, args.yaw,
                                             args.range_m)
        scale = gsd
        T = roi_to_map_transform(gsd, img.shape[0] / 2 + r0,
                                 img.shape[1] / 2 + c0)
        gsd_txt = ("auto" if not args.gsd else f"{gsd*100:.1f} cm/px")
        mode = (f"rectified: pitch {args.pitch} deg, yaw {args.yaw} deg, "
                f"height {args.height} m, fov {args.fov} deg, "
                f"gsd {gsd_txt}")
    else:
        scale = args.scale if args.scale is not None \
            else 26.0 / max(img.shape)
        T = roi_to_map_transform(scale, img.shape[0] / 2, img.shape[1] / 2)
        mode = f"BEV, assumed {scale*100:.2f} cm/px"

    det = RowDetector(row_spacing_px=args.spacing,
                      spacing_tolerance=args.tolerance)
    res = det.detect(img, T)

    traces = None
    if not args.no_trace:
        traces = trace_row_curves(img, res)
        n_pts = sum(int(np.isfinite(p[:, 0]).sum()) for p in traces) \
            if traces else 0
        print(f"traced row curves    : {len(traces or [])} polylines, "
              f"{n_pts} sampled points")

    print(f"image                : {args.image}")
    print(f"mode                 : {mode}")
    print(f"row spacing          : {res.row_spacing_px:.2f} px "
          f"= {res.row_spacing_px*scale*100:.1f} cm")
    print(f"row angle (vs x-axis): {res.row_angle_deg:.2f} deg")
    print(f"frequency (fx, fy)   : ({res.fx:.5f}, {res.fy:.5f}) c/px")
    print(f"periods (Tx, Ty)     : ({res.Tx:.1f}, {res.Ty:.1f}) px")
    print(f"phase phi            : {res.phi:.3f} rad")
    print(f"heading deviation e_t: {np.degrees(res.e_theta):.2f} deg")
    print(f"lateral deviation e_y: {res.e_y*scale*100:.1f} cm")

    conf = pitch_confidence(img, det, res)
    prom = det._peak_prominence(img, res)
    print(f"lock confidence     : {conf:.1f} (peak/ring prominence "
          f"{prom:.1f}x)")
    if res.row_spacing_px < 8.0:
        print("WARNING: locked spacing below 8 px - likely texture noise, "
              "not rows; supply --spacing if known")
    elif prom < 3.0:
        print("WARNING: weak spectral lock (prominence < 3x ring median); "
              "rows may be faint or the model violated (curved/missing rows)")

    out = args.out or os.path.splitext(args.image)[0] + "_rows.png"
    overlay(det, res, img, scale, out,
            "detected rows + centerline (cyan)", traces, args.spectrum)


if __name__ == "__main__":
    main()
