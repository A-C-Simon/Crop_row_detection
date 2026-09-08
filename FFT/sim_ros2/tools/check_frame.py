#!/usr/bin/env python3
"""Run a saved farm camera frame through the FFT pipeline and report the
detection, to sanity-check the sim view.

    FFT_DIR=/abs/path/to/FFT MULTIROI_DIR=/abs/path/to/LinReg/MultiROI \
        python3 check_frame.py frame.png
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent / "src" / "fftsim" / "fftsim"))

from pipeline import build_pipeline  # noqa: E402


def main():
    img = cv2.imread(sys.argv[1])
    pl = build_pipeline()
    out = pl.process(img)
    info = out["info"]
    print("ey=%+.2fm eth=%+.1fdeg n_rows=%s prom=%.1f status=%s v=%.2f w=%+.3f"
          % (info.get("ey_m", float("nan")),
             info.get("filt_err_theta_deg", float("nan")),
             info.get("n_two_sided"), info.get("prominence", 0),
             info.get("status"), out["v"], out["w"]))
    ovl = out.get("overlay")
    if ovl is not None:
        Path("/tmp").mkdir(exist_ok=True)
        cv2.imwrite("/tmp/frame_overlay.png", ovl)
        b = ovl[:, :, 0].astype(int)
        g = ovl[:, :, 1].astype(int)
        r = ovl[:, :, 2].astype(int)
        for name, m in [("red", (r > 150) & (g < 90) & (b < 90)),
                        ("blue", (b > 150) & (g < 90) & (r < 90)),
                        ("green", (g > 150) & (r < 90) & (b < 90))]:
            ys, xs = np.where(m)
            if len(ys):
                print("%s px=%d x=%d..%d y=%d..%d" % (name, len(ys), xs.min(),
                                                      xs.max(), ys.min(), ys.max()))
            else:
                print("%s px=0" % name)


if __name__ == "__main__":
    main()
