"""Demo: run the DFT row detector on the aerial example image.

The example image (Crop-lines-system/example_image.png) is an aerial NDVI crop
field tile, already ortho-rectified, so no camera calibration / IPM is needed
(the IPM stage in preprocessing.py is only for the paper's forward-looking
camera geometry). The row spacing is estimated automatically from the spectrum.

Usage:  python3 demo_example_image.py
Output: prints the detected spacing / angle / deviations and saves an overlay
        to demo_output.png (detected row lines in red, navigation centerline in
        cyan).
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from row_detection import RowDetector
from preprocessing import roi_to_map_transform


def main():
    img = plt.imread("../Crop-lines-system/example_image.png")
    if img.ndim == 3:
        img = img[..., 0]
    img = np.asarray(img, dtype=float)

    # estimate the scale assuming the image spans ~26 m (grid tiles are ~1 ha)
    # this is only used to report deviations in metres; the detector itself is
    # scale-free.
    scale = 26.0 / img.shape[0]

    detector = RowDetector(row_spacing_px=None, spacing_tolerance=0.15)
    T = roi_to_map_transform(scale, img.shape[0] / 2, img.shape[1] / 2)
    res = detector.detect(img, T)

    print(f"image size           : {img.shape[0]}x{img.shape[1]} px")
    print(f"row spacing          : {res.row_spacing_px:.2f} px = "
          f"{res.row_spacing_px * scale * 100:.1f} cm")
    print(f"row angle (vs x-axis): {res.row_angle_deg:.2f} deg")
    print(f"frequency (fx, fy)   : ({res.fx:.5f}, {res.fy:.5f}) c/px")
    print(f"periods (Tx, Ty)     : ({res.Tx:.1f}, {res.Ty:.1f}) px")
    print(f"phase phi            : {res.phi:.3f} rad")
    print(f"heading deviation e_t: {np.degrees(res.e_theta):.2f} deg")
    print(f"lateral deviation e_y: {res.e_y * 100:.1f} cm")

    # overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input image")

    axes[1].imshow(img, cmap="gray")
    for line in detector.row_lines(res):
        axes[1].plot([line[1], line[3]], [line[0], line[2]], "r-", lw=1)
    # navigation centreline (midway between the two nearest rows)
    lx, ly = res.line_dir
    n_hat = np.array([ly, -lx])
    d = np.sort(res.d_k)
    i = np.argsort(np.abs(d))[:2]
    d_mid = d[i].mean()
    c = np.array([img.shape[1] / 2, img.shape[0] / 2]) + d_mid / scale * n_hat
    t = np.linspace(-img.shape[0], img.shape[0], 2)
    axes[1].plot([c[0] + t[0] * lx, c[0] + t[1] * lx],
                 [c[1] - t[0] * ly, c[1] - t[1] * ly], "c-", lw=2)
    axes[1].set_title("detected rows")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("demo_output.png", dpi=120)
    print("saved overlay to demo_output.png")


if __name__ == "__main__":
    main()