"""Synthetic bird's-eye crop-row images with known ground truth.

Used to validate the DFT-based detector from row_detection.py against a
perfectly known periodic row pattern.

Geometry / conventions
----------------------
The image is a bird's-eye view of the ground. The robot sits at the image
centre, which is the origin of the local map frame:

    X (right),  Y (forward/up),  scale = `scale` metres per pixel
    pixel (i, j)  ->  X = (j - c_j) * scale,  Y = (c_i - i) * scale

Rows are parallel straight lines whose direction makes angle `row_angle_deg`
with the forward axis (+Y). The two rows bracketing the robot lie at signed
perpendicular distances  e_y +/- spacing/2  from the origin along the wavefront
normal n_hat = (cos(beta), -sin(beta)) (positive => centreline is to the right
of the robot, i.e. e_y > 0 means the robot is to the LEFT of the centreline).

The intensity profile across a row is 0.5 + 0.5 cos(psi) with

    psi(x, y) = 2 pi (x sin(beta) + y cos(beta) - C) / P_px

so row centres (maxima) sit at psi = 2 pi k, i.e.
    x/Tx + y/Ty = k - phi0/(2 pi)
with  Tx = P_px / sin(beta),  Ty = P_px / cos(beta),  phi0 = -2 pi C / P_px.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RowGroundTruth:
    """Known parameters of the generated row pattern."""

    fx: float                     # fundamental frequency along x (c/px)
    fy: float                     # fundamental frequency along y (c/px)
    Tx: float                     # period along x (px)
    Ty: float                     # period along y (px)
    phi0: float                   # phase of the pattern (rad)
    row_spacing_px: float         # perpendicular spacing (px)
    row_angle_deg: float          # beta, angle of rows vs. forward (+Y)
    e_y: float                    # placed lateral deviation (m)
    e_theta: float                # placed heading deviation (rad) = -beta
    line_consts: np.ndarray       # C_k for the true row family in map frame (m)
    n_hat: np.ndarray             # wavefront normal in map frame (X, Y)
    l_map: np.ndarray             # row direction in map frame (X, Y), unit


def generate_row_image(
    rows: int = 1200,
    cols: int = 1200,
    scale: float = 0.01,          # metres per pixel
    spacing_m: float = 0.76,      # perpendicular row spacing in metres
    row_angle_deg: float = 5.0,   # rows vs. forward (+Y), degrees
    e_y_m: float = 0.10,          # lateral deviation of the robot (m)
    row_width_frac: float = 1.0,  # 1.0 => full sinusoidal profile; <1 strips
    noise: float = 0.0,           # additive zero-mean Gaussian noise (std)
    weed_frac: float = 0.0,       # fraction of random weed blobs added
    gap_frac: float = 0.0,        # fraction of row pixels removed (gaps)
    gradient: float = 0.0,        # relative illumination ramp across image
    seed: Optional[int] = None,
) -> tuple[np.ndarray, RowGroundTruth]:
    """Generate a synthetic bird's-eye crop-row image and its ground truth."""
    rng = np.random.default_rng(seed)

    p_px = spacing_m / scale
    beta = np.radians(row_angle_deg)
    c_i, c_j = rows / 2.0, cols / 2.0

    # rows at d_k = delta0 + k*P (perpendicular distance along n_hat);
    # the two bracketing rows are delta0 - P/2 and delta0 + P/2,
    # so delta0 = e_y + P/2 (in metres), i.e. delta0_px = e_y/scale + P_px/2.
    delta0_px = e_y_m / scale + p_px / 2.0

    # pixel phase: psi = 2 pi (x sin b + y cos b - C)/P_px
    # x = row index = i, y = col index = j ;  C = c_i sin b + c_j cos b + delta0_px
    sinb, cosb = np.sin(beta), np.cos(beta)
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows), indexing="xy")
    # ii = row index (x), jj = col index (y)
    phase = 2.0 * np.pi * (ii * sinb + jj * cosb) / p_px
    phase -= 2.0 * np.pi * (c_i * sinb + c_j * cosb + delta0_px) / p_px

    # row profile (bright at phase = 2 pi k)
    if row_width_frac >= 1.0:
        img = 0.5 + 0.5 * np.cos(phase)
    else:
        w_half = np.pi * row_width_frac
        img = (np.cos(phase) > np.cos(w_half)).astype(float)

    # gaps: remove random patches of the rows
    if gap_frac > 0:
        n_gaps = int(gap_frac * rows * cols / (30 * 30)) + 1
        for _ in range(n_gaps):
            gi = int(rng.integers(0, rows - 30))
            gj = int(rng.integers(0, cols - 30))
            img[gi:gi + 30, gj:gj + 30] *= rng.uniform(0.0, 0.1)

    # weeds: bright random blobs between rows
    if weed_frac > 0:
        n_weeds = int(weed_frac * rows * cols / (20 * 20))
        ys, xs = rng.integers(0, rows, n_weeds), rng.integers(0, cols, n_weeds)
        img[ys, xs] += 1.0
        img = _blur(img, 3)

    # illumination gradient (glare / shadow ramp across the image)
    if gradient > 0:
        ramp = np.linspace(1.0 - gradient, 1.0 + gradient, cols)
        img = img * ramp[None, :]

    # sensor noise
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape)

    img = np.clip(img, 0.0, None)

    # ---- ground truth -----------------------------------------------------
    Tx = p_px / sinb if abs(sinb) > 1e-9 else float("inf")
    Ty = p_px / cosb
    phi0 = -2.0 * np.pi * (c_i * sinb + c_j * cosb + delta0_px) / p_px
    fx = sinb / p_px
    fy = cosb / p_px

    # map frame transforms (see row_detection.py coordinate notes)
    # X = scale*(y - c_j), Y = scale*(c_i - x)
    T = np.array([[0.0, scale, -scale * c_j],
                  [-scale, 0.0, scale * c_i],
                  [0.0, 0.0, 1.0]])
    n_hat = np.array([cosb, -sinb])            # wavefront normal in map frame
    l_map = np.array([sinb, cosb])             # row direction in map frame

    # true row lines in map frame: q . n_hat = d_k, d_k = delta0 + k*P
    ks = np.arange(-12, 13)
    line_consts = delta0_px * scale + ks * spacing_m

    gt = RowGroundTruth(
        fx=fx, fy=fy, Tx=Tx, Ty=Ty, phi0=phi0,
        row_spacing_px=p_px, row_angle_deg=row_angle_deg,
        e_y=e_y_m, e_theta=-beta, line_consts=line_consts,
        n_hat=n_hat, l_map=l_map,
    )
    return img, gt


def _blur(img: np.ndarray, k: int) -> np.ndarray:
    """Simple box blur via shifted slices (no scipy needed)."""
    k = int(k)
    kernel = np.ones((k, k), dtype=float) / (k * k)
    pad = k // 2
    out = np.zeros_like(img)
    rows, cols = img.shape
    for di in range(k):
        for dj in range(k):
            out[pad:pad + rows - (k - 1),
                pad:pad + cols - (k - 1)] += kernel[di, dj] * img[di:di + rows - (k - 1),
                                                                   dj:dj + cols - (k - 1)]
    return out