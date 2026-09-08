"""DFT crop-row pipeline for the fftsim test rig.

Runs Gai et al.'s DFT detector (FFT/dft_crop_row_detector.py) on each camera
frame and steers with the shared visual servo (LinReg/MultiROI/mr_vs.py), so
different perceptions are compared behind identical control.

Per frame (mirrors run_fft_video.py, fixed geometry instead of a scan):
  gray = ExG -> rectify_forward (pitch/yaw fixed for the sim camera) ->
  DFTRowDetector bandpass around the expected row spacing -> detect() ->
  EMA smooth ey/eth (hold-last on reject) -> mr_vs.compute_control.

The adapter exposes the same contract as the MultiROI pipeline:
    process(bgr) -> {"v", "w", "info", "overlay"}
with info keys err_x/err_theta_deg (raw), filt_err_x/filt_err_theta_deg
(smoothed), confidence, status, n_two_sided (= n_rows), median_width,
ey_m and prominence.
"""
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _fft_dir() -> str:
    env = os.environ.get("FFT_DIR")
    if env and (Path(env) / "dft_crop_row_detector.py").exists():
        return str(Path(env).resolve())
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "dft_crop_row_detector.py").exists():
            return str(p)
        p = p.parent
    raise RuntimeError("cannot locate FFT source dir; set FFT_DIR")


def _multiroi_dir() -> str:
    env = os.environ.get("MULTIROI_DIR")
    if env and (Path(env) / "mr_vs.py").exists():
        return str(Path(env).resolve())
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "mr_vs.py").exists():
            return str(p)
        p = p.parent
    raise RuntimeError("cannot locate MultiROI source dir (shared servo); "
                       "set MULTIROI_DIR")


sys.path.insert(0, _fft_dir())
sys.path.insert(0, _multiroi_dir())

from dft_crop_row_detector import (  # noqa: E402
    DFTRowDetector, corridor_in_image, exg_gray, rectify_forward,
)
from mr_vs import MultiROIVS, MRVSParams  # noqa: E402  (shared servo)
try:  # local corridor tracker (row affinity for curved rows)
    from row_affinity import bend_to_ff, local_corridor  # noqa: E402
except ImportError:  # imported as a package (console_scripts), not a script
    try:
        from .row_affinity import bend_to_ff, local_corridor  # noqa: E402
    except Exception:  # pragma: no cover - keep the rig runnable without it
        bend_to_ff = local_corridor = None


def spacing_band_px(gsd, min_spacing_m=0.15, max_spacing_m=2.0,
                    spacing_prior_m=None):
    """(min_period, max_period, prior) in pixels for the bandpass filter
    (mirrors spacing_band_px in run_fft_detection.py without its CLI)."""
    if gsd:
        lo = max(min_spacing_m / gsd, 6.0)
        hi = max_spacing_m / gsd
        prior = spacing_prior_m / gsd if spacing_prior_m else None
    else:
        lo, hi, prior = 10.0, None, None
    return lo, hi, prior


# Sim camera geometry (must match the rover URDF): 640x480, ~65 deg HFOV
# (~51 deg vertical), 1.4 m high, pitched ~24 deg down = 66 deg from nadir.
CAM_PITCH_DEG = float(os.environ.get("FFT_PITCH_DEG", "66.0"))
CAM_HEIGHT_M = float(os.environ.get("FFT_HEIGHT_M", "1.4"))
CAM_FOV_Y_DEG = float(os.environ.get("FFT_FOV_Y_DEG", "51.0"))
CAM_YAW_DEG = float(os.environ.get("FFT_YAW_DEG", "0.0"))
CAM_RANGE_M = float(os.environ.get("FFT_RANGE_M", "10.0"))
# Lateral scale for the shared servo (px per meter): picked so a 1.1 m
# corridor spans ~330 px, the same operating point as the MultiROI overlay.
PX_PER_M = 330.0 / 1.1
# Static bias trim (meters, subtracted from raw ey): the BEV reference
# point sits off the true straight-ahead by a near-constant amount for a
# fixed camera, which would otherwise settle as a steady offset.
TRIM_M = float(os.environ.get("FFT_TRIM_M", "0.0"))
PROM_MIN = 10.0      # below this the frame is held, not trusted
EMA_ALPHA = 0.35
# Row affinity: re-measure the corridor locally from the detected row crests
# in the bottom fraction of the BEV ROI (the global DFT fit is a straight
# plane wave and averages away curvature on curved/S-shaped fields).
AFFINITY = os.environ.get("FFT_AFFINITY", "1") not in ("0", "false", "False")
AFF_FRAC = float(os.environ.get("FFT_AFF_FRAC", "0.5"))
# Curvature feedforward gain (launch arg ff_gain / env MRSIM_FF_GAIN):
# anticipatory turn on the measured corridor bend so the P-terms do not
# hold a steady-state offset on constant-curvature paths.
FF_GAIN = float(os.environ.get("MRSIM_FF_GAIN", "0.0") or 0.0)


def _ema(prev, new, alpha=EMA_ALPHA):
    if new is None or (isinstance(new, float) and math.isnan(new)):
        return prev
    if prev is None or (isinstance(prev, float) and math.isnan(prev)):
        return new
    return (1.0 - alpha) * prev + alpha * new


def _ema_angle(prev, new, alpha=EMA_ALPHA):
    if new is None or (isinstance(new, float) and math.isnan(new)):
        return prev
    if prev is None or (isinstance(prev, float) and math.isnan(prev)):
        return new
    z = ((1.0 - alpha) * np.exp(1j * np.radians(prev))
         + alpha * np.exp(1j * np.radians(new)))
    return float(np.degrees(np.angle(z)))


def _band_args():
    return types.SimpleNamespace(min_spacing_m=0.15, max_spacing_m=2.0,
                                 spacing_prior_m=None)


class FFTPipeline:
    name = "fft"

    def __init__(self):
        try:
            lx = float(os.environ.get("MRSIM_LAMBDA_X", "2.0"))
        except ValueError:
            lx = 2.0
        try:
            lt = float(os.environ.get("MRSIM_LAMBDA_THETA", "1.0"))
        except ValueError:
            lt = 1.0
        try:
            hg = float(os.environ.get("MRSIM_HEADING_GATE", "0.1"))
        except ValueError:
            hg = 0.1
        self.vs = MultiROIVS(MRVSParams(
            width=640, height=480, vertical_coverage=0.75,
            lambda_x=lx, lambda_theta=lt, heading_gate=hg))
        self.vs.reset_smoother()
        self.ey_s = None
        self.eth_s = None
        self.last_res = None
        self.last_gsd = None
        self._last_straddle = None
        self._last_aff = None   # last accepted AffinityResult (diagnostics)
        self.held_frames = 0

    def _detect(self, bgr):
        t0 = time.perf_counter()
        gray = exg_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        roi, gsd, map_info = rectify_forward(
            gray, CAM_PITCH_DEG, CAM_HEIGHT_M, CAM_FOV_Y_DEG,
            None, yaw_deg=CAM_YAW_DEG, range_m=CAM_RANGE_M)
        lo, hi, prior = spacing_band_px(gsd)
        det = DFTRowDetector(min_period_px=lo, max_period_px=hi,
                             spacing_prior_px=prior)
        res = det.detect(roi, ref_xy=(roi.shape[1] / 2.0, roi.shape[0] - 1.0))
        # the windowed ROI is what the detector's DTFT phase refers to;
        # rebuild it here for the local crest tracker (cheap: one multiply)
        h, w = roi.shape
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        imgw = roi.astype(np.float64) * win
        ms = (time.perf_counter() - t0) * 1000.0
        return res, gsd, map_info, ms, imgw

    @staticmethod
    def _straddle_ey(res):
        """Corridor-center offset from a flanking row pair, in px.

        Returns (e_left + e_right) / 2 for the nearest line on each side of
        the reference, or None when both nearest lines sit on one side
        (wrong-furrow lock) or lines are missing.
        """
        try:
            e = np.asarray(res.e_indices, dtype=float)
            xi = np.asarray(res.intersections, dtype=float)
        except Exception:
            return None
        if e.size < 2 or xi.size != e.size:
            return None
        left = e[e < 0.0]
        right = e[e > 0.0]
        if left.size == 0 or right.size == 0:
            return None
        el = float(left[np.argmin(np.abs(left))])
        er = float(right[np.argmin(np.abs(right))])
        if not (math.isfinite(el) and math.isfinite(er)):
            return None
        return (el + er) / 2.0

    def process(self, bgr):
        h, w = bgr.shape[:2]
        status, res, gsd, map_info = "OK", None, self.last_gsd, None
        aff = None
        try:
            res, gsd, map_info, _ms, imgw = self._detect(bgr)
            ok = (res.n_rows >= 2 and res.prominence >= PROM_MIN
                  and math.isfinite(res.ey_px) and math.isfinite(res.e_theta_deg))
            if ok:
                # straddle gate: the corridor center must come from a row
                # pair flanking the reference (one line each side). Two
                # nearest lines on the SAME side mean a wrong-furrow lock.
                straddled = self._straddle_ey(res)
                if straddled is None:
                    status = "HELD one-sided (no flanking pair)"
                    ok = False
            if ok:
                # row affinity: re-measure the corridor locally from the
                # detected crests (follows curvature the global plane-wave
                # fit averages away). Falls back to the global fit on any
                # rejection (ragged crest fit, heading sanity gate, ...).
                if local_corridor is not None and AFFINITY:
                    aff = local_corridor(
                        imgw, res.fx, res.fy, res.phi, res.spacing_px,
                        roi_shape=imgw.shape,
                        ref_xy=(imgw.shape[1] / 2.0, imgw.shape[0] - 1.0),
                        global_eth_deg=res.e_theta_deg, frac=AFF_FRAC)
            if not ok and status == "OK":
                status = f"HELD weak (n={res.n_rows} prom={res.prominence:.1f})"
            if ok:
                self.last_res = res
                self.last_gsd = gsd
                self._last_straddle = straddled
                self._last_aff = aff
            else:
                res = self.last_res
                if res is None:
                    raise ValueError("no lock yet")
        except Exception as e:
            if self.last_res is None:
                raise
            status = f"HELD ({e})"
            res = self.last_res
            gsd = self.last_gsd
        if status == "OK":
            self.held_frames = 0
            # prefer the local (affinity) corridor offset when available;
            # it tracks the actual planted rows instead of the global fit
            if aff is not None:
                raw_ey_m = float(aff.ey_px) * gsd - TRIM_M
            else:
                raw_ey_m = float(self._last_straddle * gsd) - TRIM_M
        else:
            self.held_frames += 1
            aff = self._last_aff if status.startswith("HELD") else None
            raw_ey_m = float(self.ey_s) if self.ey_s is not None else 0.0
        if status == "OK" and aff is not None:
            raw_eth = float(aff.eth_deg)
        else:
            raw_eth = float(res.e_theta_deg)
        self.ey_s = _ema(self.ey_s, raw_ey_m)
        self.eth_s = _ema_angle(self.eth_s, raw_eth)
        ey_m = float(self.ey_s) if self.ey_s is not None else 0.0
        eth_deg = float(self.eth_s) if self.eth_s is not None else 0.0
        conf = float(np.clip((res.prominence - PROM_MIN) / 20.0, 0.0, 1.0))
        if status != "OK":
            conf = min(conf, 0.4)

        err_x = ey_m * PX_PER_M
        F = np.array([err_x, h / 2.0, math.radians(eth_deg)], dtype=float)
        # curvature feedforward from the local corridor bend (0 when the
        # affinity measurement is unavailable or the gain is off)
        v_ref = float(getattr(self.vs.params, "vf_des", 0.2))
        ff = 0.0
        if FF_GAIN > 0.0 and aff is not None:
            ff = bend_to_ff(aff.bend, v_ref, FF_GAIN)
        v_ref = float(getattr(self.vs.params, "vf_des", 0.2))
        v, w_ang, info = self.vs.compute_control(F, dt=None, confidence=conf,
                                                 smooth=True, ff=ff)
        if status != "OK" and self.held_frames > 30:
            # lost for seconds: stand still and keep sensing instead of
            # driving blind into the crops on stale estimates
            v = 0.0
            w_ang = 0.0
        info = dict(info)
        info.update({
            "filt_err_x": float(err_x),
            "filt_err_theta_deg": float(eth_deg),
            "filt_width": float(res.corridor_px * gsd),
            "filt_bottom_x": float(w / 2.0 + err_x),
            "raw_err_x": float(raw_ey_m * PX_PER_M),
            "raw_err_theta_deg": float(raw_eth),
            "confidence": float(conf),
            "status": status,
            "n_two_sided": int(res.n_rows),
            "n_q_accepted": int(res.n_rows),
            "median_width": float(res.corridor_px),
            "ey_m": float(ey_m),
            "prominence": float(res.prominence),
            "affinity": bool(aff is not None),
            "aff_bend": float(aff.bend) if aff is not None else 0.0,
            "ff": float(ff),
        })
        overlay = self._draw(bgr, res, map_info, ey_m, eth_deg, v, w_ang,
                             info)
        return {"v": v, "w": w_ang, "info": info, "overlay": overlay,
                "res": res, "gsd": gsd}

    def _draw(self, bgr, res, map_info, ey_m, eth_deg, v, w_ang, info):
        out = bgr.copy()
        h, w = out.shape[:2]
        try:
            corr = corridor_in_image(res, map_info) if map_info else None
        except Exception:
            corr = None
        if corr:
            try:
                for poly in corr.get("rows", []):
                    p = np.asarray(poly)[::100]
                    if len(p) >= 2:
                        cv2.polylines(out, [p.astype(np.int32)], False,
                                      (150, 50, 0), 1, cv2.LINE_AA)
                for poly in corr.get("borders", []):
                    p = np.asarray(poly)[::50]
                    if len(p) >= 2:
                        cv2.polylines(out, [p.astype(np.int32)], False,
                                      (255, 200, 0), 2, cv2.LINE_AA)
                cl = corr.get("centerline")
                if cl is not None:
                    p = np.asarray(cl)[::50]
                    if len(p) >= 2:
                        cv2.polylines(out, [p.astype(np.int32)], False,
                                      (0, 0, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
        cv2.circle(out, (w // 2, h // 2), 4, (255, 255, 0), -1)
        cv2.drawMarker(out, (w // 2, h - 20), (255, 255, 0),
                       cv2.MARKER_STAR, 20, 2)
        txt = (f"v={v:.2f} m/s w={math.degrees(w_ang):.1f} deg/s | "
               f"ey={ey_m:+.2f}m eth={eth_deg:+.1f}deg")
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
        txt2 = (f"rows={res.n_rows} prom={res.prominence:.1f} "
                f"status={info.get('status', '')}")
        cv2.putText(out, txt2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def reset(self):
        self.ey_s = self.eth_s = None
        self.last_res = None
        self._last_straddle = None
        self._last_aff = None
        self.held_frames = 0
        self.vs.reset_smoother()


def build_pipeline(algorithm: str = "fft"):
    algo = (algorithm or "fft").lower()
    if algo != "fft":
        raise NotImplementedError(
            f"algorithm '{algorithm}' unknown here; this rig runs 'fft'")
    return FFTPipeline()


def current_algorithm() -> str:
    return os.environ.get("MRSIM_ALGORITHM", "fft")
