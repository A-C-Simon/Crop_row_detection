"""Validation of the DFT crop-row detector on synthetic images.

Run:  python3 test_detection.py
Exit code 0 on success, 1 on failure. Prints a per-test PASS/FAIL summary.

Checks
------
  1. clean recovery of fx, fy, phi, spacing, angle
  2. detected row family coincides with the true row family
  3. position/heading deviations e_y, e_theta match ground truth
  4. robustness to noise / weeds / gaps / illumination gradient
  5. Hanning window improves peak localisation
  6. recovery across different spacing / angle / offset settings
  7. LQG closed-loop simulation drives deviations to zero
"""

from __future__ import annotations

import numpy as np

from preprocessing import roi_to_map_transform
from row_detection import (RowDetector, refine_peak_2d, bandpass_mask,
                           trace_row_curves)
from synthetic import generate_row_image
from lqg_controller import LQGController


def _tol(a, b, rel=1e-3, abs_=1e-9):
    return abs(a - b) <= abs_ + rel * max(abs(a), abs(b))


def _angular_diff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_recovery():
    img, gt = generate_row_image(seed=0)
    det = RowDetector(row_spacing_px=gt.row_spacing_px).detect(img)
    # frequencies recovered to ~2-3% (0.03 bins on a 1200px grid)
    assert _tol(det.fx, gt.fx, rel=3e-2, abs_=1e-5), (det.fx, gt.fx)
    assert _tol(det.fy, gt.fy, rel=3e-2, abs_=1e-5), (det.fy, gt.fy)
    assert abs(_angular_diff(det.phi, gt.phi0)) < 0.08, (det.phi, gt.phi0)
    assert _tol(det.row_spacing_px, gt.row_spacing_px, rel=5e-3), \
        (det.row_spacing_px, gt.row_spacing_px)
    assert _tol(abs(det.row_angle_deg), abs(gt.row_angle_deg), rel=5e-2, abs_=0.3), \
        (det.row_angle_deg, gt.row_angle_deg)


def test_line_family():
    img, gt = generate_row_image(seed=0)
    T = roi_to_map_transform(0.01, img.shape[0] / 2, img.shape[1] / 2)
    det = RowDetector(row_spacing_px=gt.row_spacing_px, num_rows=24).detect(img, T)
    # every row of the true family near the robot must be detected to 2 cm
    # (the full set of far-away rows is not compared: a tiny direction error
    #  rotates the family and displaces far rows without affecting navigation)
    family = gt.line_consts
    near = family[np.abs(family) < 3.0]
    for d in near:
        if not np.any(np.abs(det.d_k - d) < 0.02):
            raise AssertionError(f"true row at {d:.4f} m not detected")
    # and the detected spacing must match the true spacing
    assert _tol(np.median(np.diff(np.sort(det.d_k))), gt.row_spacing_px * 0.01,
                rel=5e-2), np.median(np.diff(np.sort(det.d_k)))


def test_deviations():
    for e_y in (-0.30, -0.05, 0.0, 0.12, 0.35):
        for beta in (3.0, 5.0, -7.0, 12.0):
            img, gt = generate_row_image(e_y_m=e_y, row_angle_deg=beta, seed=1)
            T = roi_to_map_transform(0.01, img.shape[0] / 2, img.shape[1] / 2)
            det = RowDetector(row_spacing_px=gt.row_spacing_px).detect(img, T)
            assert abs(det.e_y - gt.e_y) < 0.02, (det.e_y, gt.e_y, e_y, beta)
            assert abs(_angular_diff(det.e_theta, gt.e_theta)) < np.radians(0.5), \
                (det.e_theta, gt.e_theta, beta)


def test_robustness():
    cases = [
        dict(noise=0.05, seed=2),
        dict(noise=0.15, seed=3),
        dict(weed_frac=0.15, seed=4),
        dict(gap_frac=0.10, seed=5),
        dict(gradient=0.35, seed=6),
        dict(noise=0.05, weed_frac=0.10, gap_frac=0.05, gradient=0.20, seed=7),
    ]
    for kw in cases:
        img, gt = generate_row_image(e_y_m=0.10, row_angle_deg=6.0, **kw)
        T = roi_to_map_transform(0.01, img.shape[0] / 2, img.shape[1] / 2)
        det = RowDetector(row_spacing_px=gt.row_spacing_px).detect(img, T)
        assert abs(det.e_y - gt.e_y) < 0.06, (det.e_y, gt.e_y, kw)
        assert abs(_angular_diff(det.e_theta, gt.e_theta)) < np.radians(2.0), \
            (det.e_theta, gt.e_theta, kw)


def test_hanning_improves_peak():
    img, gt = generate_row_image(seed=8)
    det_win = RowDetector(row_spacing_px=gt.row_spacing_px).detect(img)
    # without the window the coarse peak is a sinc peak; quantify the error
    # by the frequency residual after refinement on the unwindowed spectrum
    mag_uw = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    m, n = img.shape
    fx = np.fft.fftshift(np.fft.fftfreq(m))
    fy = np.fft.fftshift(np.fft.fftfreq(n))
    mask = bandpass_mask(fx, fy, gt.row_spacing_px)
    filt = mag_uw * mask
    ci = np.unravel_index(np.argmax(filt), filt.shape)
    ri, rj = refine_peak_2d(filt, int(ci[0]), int(ci[1]))
    err_uw = np.hypot((ri - m // 2) / m - gt.fx, (rj - n // 2) / n - gt.fy)
    err_w = np.hypot(det_win.fx - gt.fx, det_win.fy - gt.fy)
    assert err_w <= err_uw, (err_w, err_uw)


def test_parameter_sweep():
    for spacing in (0.56, 0.76, 0.90):
        for beta in (-12.0, -4.0, 4.0, 10.0):
            img, gt = generate_row_image(spacing_m=spacing, row_angle_deg=beta,
                                         e_y_m=0.08, seed=9)
            det = RowDetector(row_spacing_px=gt.row_spacing_px).detect(img)
            assert _tol(det.row_spacing_px, gt.row_spacing_px, rel=5e-3), \
                (det.row_spacing_px, gt.row_spacing_px)
            assert abs(_angular_diff(det.phi, gt.phi0)) < 0.08


def test_auto_spacing():
    for kw in (dict(seed=11), dict(noise=0.15, seed=12),
               dict(weed_frac=0.15, gap_frac=0.10, gradient=0.25, seed=13)):
        img, gt = generate_row_image(**kw)
        det = RowDetector(row_spacing_px=None)
        res = det.detect(img)
        assert _tol(res.row_spacing_px, gt.row_spacing_px, rel=0.10), \
            (res.row_spacing_px, gt.row_spacing_px, kw)


def test_trace_curved_rows():
    """Curved rows: the tracer must follow the curve; straight lines cannot."""
    rows, cols, P = 480, 480, 60.0
    amp = 14.0                       # lateral bow (px) across the image
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    delta = amp * ((ii - rows / 2) / (rows / 2)) ** 2   # shift vs row index x
    img = 0.5 + 0.5 * np.cos(2.0 * np.pi * (jj + delta) / P)
    det = RowDetector(row_spacing_px=P)
    res = det.detect(img)
    assert abs(res.row_angle_deg) < 1.0, res.row_angle_deg
    curves = trace_row_curves(img, res)
    assert curves, "no polylines traced"
    # central row: true centreline is y(x) = k*P - delta(x); take the traced
    # polyline closest to the image centre and check it tracks the curve,
    # while the global straight family misses by ~amp at the image ends
    best = min(curves, key=lambda c: abs(np.nanmean(c[:, 1]) - cols / 2))
    pts = best[np.isfinite(best[:, 0])]
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    assert len(pts) > 10, len(pts)
    x_grid = np.arange(rows)
    interp = np.interp(x_grid, pts[:, 0], pts[:, 1])
    k_true = np.round(interp[rows // 2] / P)
    true_y = k_true * P - amp * ((x_grid - rows / 2) / (rows / 2)) ** 2
    err_trace = np.abs(interp - true_y).max()
    straight = k_true * P            # global straight-line model
    err_straight_ends = max(abs(straight - true_y[0]), abs(straight - true_y[-1]))
    assert err_trace < 3.0, err_trace
    assert err_straight_ends > 5.0, err_straight_ends


def test_lqg_convergence():
    speed, dt = 0.8, 0.125
    Qx = np.diag([1.0, 0.1, 1.0, 0.1])
    Ru = 0.1
    Qw = np.diag([1e-4, 1e-3, 1e-4, 1e-3])
    Rv = np.diag([0.0643 ** 2, np.radians(1.48) ** 2])
    ctrl = LQGController(speed, dt, Qx, Ru, Qw, Rv)
    x0 = np.array([0.50, 0.0, np.radians(10.0), 0.0])
    hist = ctrl.simulate(300, x0, seed=10)
    ey = np.abs(hist["x"][-1, 0])
    eth = np.abs(hist["x"][-1, 2])
    assert ey < 0.05, ey
    assert eth < np.radians(3.0), np.degrees(eth)
    # max overshoot of position error should stay bounded
    assert np.abs(hist["x"][:, 0]).max() < 0.6


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())