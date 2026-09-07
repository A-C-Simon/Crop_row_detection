"""Algorithm selection seam for the mrsim test rig.

`mode` (nav_node.py) runs a full detection+servoing pipeline per frame. The
rig currently ships one algorithm:

    multiroi  - MultiROI strip detector + temporal filter + mr_vs visual
                servoing (LinReg/MultiROI; the `process_image` pipeline).

Other detectors live in the repo but are NOT yet wired for closed-loop
driving (each has a different output contract - FFT works on a rectified
BEV image, CAROLIF emits clustered splines, ExG is a separate C++ ROS2
package). When one is adapted it gets a builder here plus an entry in
`ALGORITHMS` so it can be selected with `algorithm:=<name>`.

Selection comes from the `MRSIM_ALGORITHM` env var (set by farm.launch.py's
`algorithm:=` argument or run_sim.sh's `--algo`).
"""
import os
import sys
from pathlib import Path

import cv2
import math
import numpy as np

# Same colors as test_multi_roi.draw_results (BGR) so the sim view matches
# the offline algorithm figures.
_COLOR_NAV = (255, 200, 0)    # light blue - nav curve/line
_COLOR_DET = (150, 50, 0)     # dark blue - per-row detection lines
_COLOR_ROI = (255, 255, 255)  # white - ROI boxes


def _draw_detector_on_overlay(overlay, res):
    """Fuse detector viz (ROIs, Q dots, det/nav lines) onto the vs overlay.

    Mirrors test_multi_roi.draw_results(..., draw_rois=True) but draws on the
    full-size nav overlay (which already carries the filtered red line +
    control text), shifting cropped coords back by crop_offset.
    Returns the same array (drawn in place on a copy by the caller).
    """
    try:
        dx, dy = res.get("crop_offset", (0, 0))
        dx, dy = int(dx), int(dy)
    except Exception:
        dx, dy = 0, 0
    h, w = overlay.shape[:2]
    try:
        cov = float(res.get("vertical_coverage", 0.75))
    except Exception:
        cov = 0.75
    if not math.isfinite(cov):
        cov = 0.75
    cov = min(1.0, max(0.1, cov))
    # coverage top in CROPPED coords (binary frame); detector binary is the
    # cropped image, i.e. smaller than the overlay by (dx, dy) per side.
    try:
        bh, bw = res.get("binary").shape[:2]
    except Exception:
        bh, bw = h - 2 * dy, w - 2 * dx
    y_top_c = int(round(bh * (1.0 - cov))) if cov < 1.0 else 0
    y_bot_c = bh - 1

    def _clip_y(y):
        return max(0, min(h - 1, int(y)))

    # --- ROI boxes (white) + strip numbers ---
    try:
        for i, (x_lo, x_hi, y1, y2) in enumerate(res.get("rois", []), start=1):
            pt1 = (int(x_lo) + dx, _clip_y(int(y1) + dy))
            pt2 = (int(x_hi) + dx, _clip_y(int(y2) + dy))
            cv2.rectangle(overlay, pt1, pt2, _COLOR_ROI, 1)
            cv2.putText(overlay, str(i), (pt1[0] + 3, pt2[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    except Exception:
        pass

    # --- Q midpoints: green = used in nav fit, red = rejected ---
    try:
        for qx, qy in res.get("q_rejected", []):
            cv2.circle(overlay, (int(qx) + dx, _clip_y(int(qy) + dy)),
                       3, (0, 0, 255), -1)
        for qx, qy in res.get("q_accepted", []):
            cv2.circle(overlay, (int(qx) + dx, _clip_y(int(qy) + dy)),
                       3, (0, 255, 0), -1)
    except Exception:
        pass

    def _draw_line_cropped(w_slope, b, color, thickness=1):
        try:
            if abs(float(w_slope)) < 1e-6:
                yb = int(float(b)) + dy
                if yb < y_top_c + dy or yb > y_bot_c + dy:
                    return
                cv2.line(overlay, (0, _clip_y(yb)),
                         (w - 1, _clip_y(yb)), color, thickness)
                return
            x_top = (y_top_c - float(b)) / float(w_slope) + dx
            x_bot = (y_bot_c - float(b)) / float(w_slope) + dx
            cv2.line(overlay,
                     (int(x_top), _clip_y(y_top_c + dy)),
                     (int(x_bot), _clip_y(y_bot_c + dy)),
                     color, thickness, cv2.LINE_AA)
        except Exception:
            pass

    # --- per-row detection lines (dark blue, thin so red servo stays dominant)
    try:
        for w_slope, b, _n in res.get("det_lines", []):
            _draw_line_cropped(w_slope, b, _COLOR_DET, 1)
    except Exception:
        pass

    # --- nav curve (light blue polyline) or straight nav line ---
    try:
        curve = res.get("nav_curve")
        if curve:
            pts = [(int(x) + dx, _clip_y(int(y) + dy))
                   for x, y in curve if float(y) >= y_top_c]
            if len(pts) >= 2:
                cv2.polylines(overlay, [np.array(pts, dtype=np.int32)],
                              False, _COLOR_NAV, 1, cv2.LINE_AA)
        elif res.get("nav_line") is not None:
            w_slope, b = res["nav_line"]
            _draw_line_cropped(w_slope, b, _COLOR_NAV, 1)
    except Exception:
        pass
    return overlay


def _multiroi_dir() -> str:
    env = os.environ.get("MULTIROI_DIR")
    if env and (Path(env) / "run_mr_navigation.py").exists():
        return str(Path(env).resolve())
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "run_mr_navigation.py").exists():
            return str(p)
        p = p.parent
    raise RuntimeError("cannot locate MultiROI source dir; set MULTIROI_DIR")


def build_pipeline(algorithm: str):
    """Return a pipeline object for `algorithm` (raises for unimplemented).

    The returned object must expose, for a BGR frame:
        process(bgr) -> out dict  with v, w, info, overlay  (nav mode)
    MultiROI's implementation adapts the existing `process_image()`.
    """
    algo = (algorithm or "multiroi").lower()
    if algo == "multiroi":
        sys.path.insert(0, _multiroi_dir())
        from mr_vs import MultiROIVS, MRVSParams
        from temporal_filter import TemporalNavigationFilter, TemporalFilterParams
        from test_multi_roi import MultiROIDetector
        from run_mr_navigation import process_image

        class _MultiROIPipeline:
            name = "multiroi"

            def __init__(self):
                self.detector = MultiROIDetector()
                # servo gains overridable for experiments (defaults = the
                # validated values); e.g. a smaller lambda_theta leans on the
                # local lateral error instead of the lookahead heading, which
                # reduces inside-cutting on constant-curvature paths.
                try:
                    lx = float(os.environ.get("MRSIM_LAMBDA_X", "2.0"))
                except ValueError:
                    lx = 2.0
                try:
                    lt = float(os.environ.get("MRSIM_LAMBDA_THETA", "1.0"))
                except ValueError:
                    lt = 1.0
                try:
                    fg = float(os.environ.get("MRSIM_FF_GAIN", "0.0"))
                except ValueError:
                    fg = 0.0
                self.ff_gain = fg
                self.ff_mem = {}
                self.vs = MultiROIVS(MRVSParams(
                    width=640, height=480, vertical_coverage=0.75,
                    lambda_x=lx, lambda_theta=lt))
                self.t_filter = TemporalNavigationFilter(
                    TemporalFilterParams(image_width=640, image_height=480,
                                         n_strips=self.detector.n))
                self.vs.reset_smoother()
                self.last_w = 0.0

            def process(self, bgr):
                out = process_image(bgr, self.detector, self.vs, draw=True,
                                    t_filter=self.t_filter, last_w=self.last_w,
                                    lookahead_map=None,
                                    use_ff=True, ff_gain=self.ff_gain,
                                    ff_mem=self.ff_mem)
                self.last_w = float(out["w"])
                # Fuse the detector visualization onto the servo overlay so
                # /multiroi/overlay (and the saved frame_*.png) shows what
                # the offline algorithm figures show: white ROI boxes +
                # strip numbers, green/red Q dots, dark-blue row lines and
                # light-blue nav curve - on top of the red filtered line.
                try:
                    ovl = out.get("overlay")
                    res = out.get("res")
                    if ovl is not None and res is not None:
                        out["overlay"] = _draw_detector_on_overlay(
                            ovl, res)
                except Exception:
                    pass
                return out

            def reset(self):  # for odometry/teleop mode changes
                self.last_w = 0.0
                self.vs.reset_smoother()

        return _MultiROIPipeline()

    raise NotImplementedError(
        f"algorithm '{algorithm}' is not wired into the sim rig yet. "
        "Available: multiroi. (FFT/CAROLIF/ExG detectors exist in the repo "
        "but need a closed-loop adapter - see pipeline.py docs.)")


def current_algorithm() -> str:
    return os.environ.get("MRSIM_ALGORITHM", "multiroi")
