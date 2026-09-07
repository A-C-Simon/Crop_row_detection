#!/usr/bin/env python3
"""Run a saved farm camera frame through the MultiROI pipeline (draw=True)
and report where the drawn nav/det lines sit, to sanity-check the sim view."""
import sys
from pathlib import Path

import cv2
import numpy as np

_MULTIROI = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _MULTIROI)

from test_multi_roi import MultiROIDetector
from mr_vs import MultiROIVS, MRVSParams
from temporal_filter import TemporalNavigationFilter, TemporalFilterParams
from run_mr_navigation import process_image


def main():
    img = cv2.imread(sys.argv[1])
    detector = MultiROIDetector()
    vs = MultiROIVS(MRVSParams(width=640, height=480, vertical_coverage=0.75))
    tf = TemporalNavigationFilter(
        TemporalFilterParams(image_width=640, image_height=480,
                             n_strips=detector.n))
    vs.reset_smoother()
    out = process_image(img, detector, vs, draw=True, t_filter=tf,
                        dt=0.05, last_w=0.0, lookahead_map=None)
    info = out["info"]
    print("err_x=%.1f raw_th=%.1f filt_th=%.1f conf=%.2f n_two=%s status=%s"
          % (info.get("err_x", 0),
             info.get("raw_err_theta_deg", info.get("err_theta_deg", 0)),
             info.get("filt_err_theta_deg", 0),
             info.get("confidence", 0), info.get("n_two_sided"), info.get("status")))
    res = detector.last_result if hasattr(detector, "last_result") else None
    ovl = out.get("overlay")
    if ovl is not None:
        Path("/tmp").mkdir(exist_ok=True)
        cv2.imwrite("/tmp/frame_overlay.png", ovl)
        # where are red (0,0,255-ish) and blue (255,0,0-ish) pixels?
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
