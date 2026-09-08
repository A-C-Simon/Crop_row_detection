"""Offline check for the row-affinity tracker (no Gazebo needed).

Builds synthetic bird's-eye fields (straight / arc / S) with the same wave
model the DFT detector assumes, runs the real DFTRowDetector plus the new
local_corridor affinity measurement, and compares both against the ground
truth corridor center at the reference point (bottom center of the ROI).

    python3 test_row_affinity.py
"""
import math
import sys
from pathlib import Path

import numpy as np

_FFT = Path(__file__).resolve()
for _ in range(6):
    if (_FFT / "dft_crop_row_detector.py").exists():
        break
    _FFT = _FFT.parent
sys.path.insert(0, str(_FFT))
from dft_crop_row_detector import DFTRowDetector  # noqa: E402

from row_affinity import local_corridor  # noqa: E402


def _rows_field(h=320, w=260, period=44.0, lateral_fn=None, phase0=0.0,
                noise=0.06, seed=7, duty=0.5):
    """Bright row crests on dark soil. lateral_fn(y) gives the x-offset of
    the row family (curve); rows at x = k*period + lateral_fn(y)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    lat = lateral_fn(yy)
    s = xx - lat
    img = 0.5 + 0.5 * np.cos(2.0 * np.pi * s / period + phase0)
    # missing plants: zero out a few crest patches
    for _ in range(6):
        cy = rng.integers(0, h)
        cx = rng.integers(0, w)
        img[cy:cy + 18, cx:cx + 30] = 0.0
    img += noise * rng.standard_normal((h, w))
    return img


def run_case(name, lateral_fn, kind="straight"):
    h, w = 320, 260
    period = 44.0
    img = _rows_field(h, w, period, lateral_fn)
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    imgw = img * win
    ref = (w / 2.0, h - 1.0)
    det = DFTRowDetector(min_period_px=12.0, max_period_px=90.0)
    res = det.detect(img, ref_xy=ref)

    # ground truth: crest positions at the reference row, using the
    # detector's convention e = line_x - ref_x (center RIGHT of ref > 0).
    lat_ref = lateral_fn(np.array([ref[1]]))[0]
    crest_x = np.arange(-4 * period, w + 4 * period, period) + lat_ref
    e = crest_x - ref[0]
    left = e[e < 0]   # crest left of the ref
    right = e[e > 0]  # crest right of the ref
    gt = 0.5 * (left[np.argmin(np.abs(left))]
                + right[np.argmin(np.abs(right))])

    aff = local_corridor(imgw, res.fx, res.fy, res.phi, res.spacing_px,
                         roi_shape=img.shape, ref_xy=ref,
                         global_eth_deg=res.e_theta_deg)

    g_ey = float(res.ey_px)
    a_ey = float(aff.ey_px) if aff is not None else float("nan")
    print(f"[{name}] rows={res.n_rows} prom={res.prominence:.1f} "
          f"eth={res.e_theta_deg:+.2f}deg")
    print(f"    gt={gt:+.1f}px  global={g_ey:+.1f}px  "
          f"affinity={'-' if aff is None else f'{a_ey:+.1f}px'}")
    if aff is not None:
        print(f"    aff eth={aff.eth_deg:+.2f}deg bend={aff.bend:+.3f}rad "
              f"res={aff.residual_px:.1f}px")
    return gt, g_ey, a_ey, (res.e_theta_deg if aff is None else aff.eth_deg)


def main():
    # straight rows, shifted family: both methods should nail it
    run_case("straight", lambda y: 14.0 + 0.0 * y)
    # constant-curvature arc: x(y) = R*(1 - cos(y/R)); R=900 drifts ~56 px
    # across the ROI (~1.3 periods) and leans ~19 deg at the top - the
    # global plane-wave fit must average this away, the affinity fit must
    # not
    run_case("arc (R=900)",
             lambda y: 8.0 + 900.0 * (1.0 - np.cos(np.asarray(y) / 900.0)))
    # S-shape: bend changes sign inside the ROI
    run_case("S curve", lambda y: 10.0 + 26.0 * np.tanh((y - 200.0) / 90.0))


if __name__ == "__main__":
    main()
