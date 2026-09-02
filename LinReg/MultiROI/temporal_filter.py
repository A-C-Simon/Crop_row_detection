"""
TemporalNavigationFilter + CommandSmoother
==========================================
Temporal robustness layer for MultiROI crop-row navigation.

Sits between the raw MultiROI detector (test_multi_roi.MultiROIDetector)
and the visual-servoing controller (mr_vs.MultiROIVS).

Design principle:
  Trust the past enough to reject transient anomalies, but allow
  persistent, coherent, multi-frame evidence to override the past.

What is filtered:
  Control-relevant quantities in STABLE image space:
    - bottom_x   : x-position of the corridor centre at the image bottom (px)
                   (== w/2 + err_x).  Never filter raw slope w,b (1e6 issue).
    - theta      : heading error vs vertical (rad), from F.Theta
    - width      : corridor width D (px), median across two-sided strips
  NOT filtered: raw slope/intercept (w,b) directly.

Prediction:
  Constant-position model by default.  If last_w and dt are supplied
  the prediction optionally nudges bottom_x / theta using a tiny
  motion model (purely optional, small gain).

Innovation gating:
  innovation = raw - predicted
  thresholds are relative to image width / corridor width / degrees.

Persistence:
  Large innovations are marked suspicious.  They are NOT applied.
  Instead they are stored as a pending candidate.  If the same
  deviation persists for `persist_frames` consecutive frames within a
  coherence window, it is accepted (gradual EMA towards it).

Confidence:
  Combines: n_two_sided, n_q_accepted, weed_pressure / fail rate,
            width plausibility, innovation magnitude, hold/pending.
  Exposed for logging and for optional speed scaling.

CommandSmoother:
  Low-pass + rate-limit + deadband on w, optional confidence-based
  v scaling.  Small / causal, no large buffers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import numpy as np

def wrapToPi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

@dataclass
class TemporalFilterParams:
    # image geometry – used to normalise thresholds
    image_width: int = 640
    image_height: int = 480

    # --- innovation gating ---
    # max bottom-center jump that is accepted in one step WITHOUT persistence.
    # expressed two ways; the effective threshold is max of both.
    max_bottom_jump_frac: float = 0.10        # fraction of image width  (0.10*640=64px)
    max_bottom_jump_width_frac: float = 0.45  # fraction of corridor width (0.45*180≈81px)
    max_heading_jump_deg: float = 12.0        # deg per frame
    max_width_change_frac: float = 0.30       # relative change in width per frame
    # absolute minimal thresholds to avoid zero-width lock
    min_bottom_jump_px: float = 18.0
    min_width_px: float = 10.0

    # --- persistence ---
    persist_frames: int = 4                   # same large jump for N frames -> accept
    pending_coherence_frac: float = 0.5       # two pending raws within 50% of thresh count as same
    pending_alpha: float = 0.08               # tiny nudge while pending (hold vs slow drift)

    # --- EMA smoothing when accepted ---
    alpha_x: float = 0.35       # how fast bottom_x follows accepted raw
    alpha_theta: float = 0.35
    alpha_width: float = 0.30
    alpha_conf: float = 0.40    # confidence EMA

    # --- confidence ---
    # raw confidence weights (informational, not hard gate)
    # confidence stays [0,1]; display/helpers use it for speed scaling.

    # --- prediction ---
    # if last_w available, predict lateral drift: dX = -k * w * dt * width
    # k is very small.  Disabled when k==0.
    motion_gain_x: float = 0.0
    motion_gain_theta: float = 0.0

    # for logging
    n_strips: int = 10

@dataclass
class CommandSmootherParams:
    w_alpha: float = 0.35          # low-pass on w (0=hold, 1=no smoothing)
    max_w_rate: float = 1.2        # rad/s per second, i.e. delta_w / dt
    w_deadband: float = 0.02       # rad/s; values inside deadband -> 0
    v_conf_scale: bool = True
    v_min_scale: float = 0.4       # at confidence 0, v = vf * 0.4
    # time source – if None use wall clock; else per-frame dt supplied

@dataclass
class TemporalState:
    filt_bottom_x: float = 0.0
    filt_theta: float = 0.0        # rad
    filt_width: float = 0.0
    filt_X: float = 0.0            # filt_bottom_x - w/2
    confidence: float = 0.7
    frame_idx: int = 0
    status: str = "init"           # accepted | held | pending | pending_accepted | held_no_line | init
    pending_count: int = 0
    time_last: float = field(default_factory=time.perf_counter)

class TemporalNavigationFilter:
    """
    Maintains filtered corridor state.
    Call `update(raw, ...)` each frame.
    Call `reset()` when sequence breaks (e.g. new folder).
    `filtered_F` is suitable for MultiROIVS.compute_control().
    """

    def __init__(self, params: Optional[TemporalFilterParams] = None):
        self.params = params or TemporalFilterParams()
        self.state: Optional[TemporalState] = None
        # pending candidate – the suspicious large jump we are watching
        self._pending_raw: Optional[Dict] = None
        self._pending_count: int = 0
        self._last_w: float = 0.0
        self._frame: int = 0

    def reset(self):
        self.state = None
        self._pending_raw = None
        self._pending_count = 0
        self._frame = 0

    # --------------------------------------------------------------
    def _compute_raw_confidence(self, raw: Dict) -> float:
        p = self.params
        n = p.n_strips
        n_two = int(raw.get("n_two_sided", 0))
        n_q = int(raw.get("n_q_accepted", 0))
        weed = float(raw.get("weed_pressure", 0.0))
        if not math.isfinite(weed):
            weed = 0.0
        weed = max(0.0, min(1.0, weed))

        # base from evidence count
        # expected two-sided: at least n-1 (strip1 excluded) ideally; use ratio
        denom_two = max(1, n - 1)
        r_two = n_two / denom_two
        r_q = n_q / max(1, n)   # up to 10 accepted points

        base = 0.55 * r_two + 0.35 * r_q + 0.10 * (1.0 if raw.get("has_line") else 0.0)
        # penalise weed pressure
        base *= (1.0 - 0.45 * weed)
        # penalise many rejected q points
        n_rej = int(raw.get("n_q_rejected", 0))
        if n_rej > 3:
            base *= 0.85
        return float(np.clip(base, 0.05, 1.0))

    # --------------------------------------------------------------
    def _predict(self, dt: float, last_w: float) -> Tuple[float, float, float]:
        """Return predicted (bottom_x, theta, width). Constant model + optional motion."""
        assert self.state is not None
        p = self.params
        pred_x = self.state.filt_bottom_x
        pred_theta = self.state.filt_theta
        pred_w = self.state.filt_width
        # optional motion compensation – very small
        if p.motion_gain_x and last_w and dt:
            pred_x += -p.motion_gain_x * last_w * dt * p.image_width
        if p.motion_gain_theta and last_w and dt:
            pred_theta = wrapToPi(pred_theta + p.motion_gain_theta * last_w * dt)
        return pred_x, pred_theta, pred_w

    # --------------------------------------------------------------
    def update(self, raw: Dict, dt: Optional[float] = None, last_w: Optional[float] = None) -> Dict:
        """
        Args:
            raw: dict with keys:
                raw_bottom_x, raw_theta (rad), raw_width, raw_X,
                has_line (bool), n_two_sided, n_q_accepted, n_q_rejected,
                weed_pressure, raw_err_x, raw_err_theta_deg, ...
            dt: seconds since last frame (for rate limit / prediction). If None, wall clock.
            last_w: previous angular velocity (for optional prediction)
        Returns:
            dict with filtered state + diagnostics:
                filt_bottom_x, filt_theta, filt_width, filt_X, filt_F (np array),
                confidence, status, innovation_x, innovation_theta, innovation_width,
                is_large, pending_count, raw, ...
        """
        p = self.params
        self._frame += 1
        now = time.perf_counter()

        if dt is None:
            if self.state is None:
                dt = 0.05
            else:
                dt = now - self.state.time_last
                dt = max(1e-3, min(0.5, dt))

        if last_w is not None:
            self._last_w = float(last_w)

        has_line = bool(raw.get("has_line", False))
        raw_bottom_x = float(raw.get("raw_bottom_x", p.image_width / 2.0))
        raw_theta = float(raw.get("raw_theta", 0.0))
        raw_width = float(raw.get("raw_width", 0.0))
        if not math.isfinite(raw_width) or raw_width < 1:
            raw_width = float(self.state.filt_width) if self.state and self.state.filt_width > 1 else 120.0
        raw_X = float(raw.get("raw_X", raw_bottom_x - p.image_width / 2.0))

        raw_conf = self._compute_raw_confidence(raw)

        # first frame or no line previously – initialise
        if self.state is None:
            if not has_line:
                # no measurement at all – stay uninitialised but report held
                filt_x = p.image_width / 2.0
                filt_theta = 0.0
                filt_w = raw_width if raw_width > 1 else 120.0
                conf = 0.2
                self.state = TemporalState(
                    filt_bottom_x=filt_x, filt_theta=filt_theta,
                    filt_width=filt_w, filt_X=0.0,
                    confidence=conf, frame_idx=self._frame,
                    status="held_no_line", time_last=now)
                innov_x = innov_theta = innov_w = 0.0
                is_large = False
                status = "held_no_line"
            else:
                # accept first measurement directly
                self.state = TemporalState(
                    filt_bottom_x=raw_bottom_x,
                    filt_theta=wrapToPi(raw_theta),
                    filt_width=raw_width,
                    filt_X=raw_X,
                    confidence=float(np.clip(0.6 * raw_conf + 0.4 * 0.7, 0.0, 1.0)),
                    frame_idx=self._frame,
                    status="accepted",
                    time_last=now)
                innov_x = innov_theta = innov_w = 0.0
                is_large = False
                status = "accepted"
                self._pending_raw = None
                self._pending_count = 0
            # build output
            return self._build_output(raw, raw_conf, innov_x, innov_theta, innov_w, is_large, status)

        # has a previous state – predict
        pred_x, pred_theta, pred_w = self._predict(dt, self._last_w)

        if not has_line:
            # no detection this frame – decay confidence, hold prediction
            # slowly drift confidence down
            new_conf = float(np.clip(self.state.confidence * 0.92, 0.05, 1.0))
            # hold filtered state (maybe tiny decay towards centre)
            # keep filt as pred
            self.state.filt_bottom_x = pred_x
            self.state.filt_theta = pred_theta
            # width holds
            self.state.confidence = new_conf
            self.state.frame_idx = self._frame
            self.state.status = "held_no_line"
            self.state.time_last = now
            # reset pending? keep it – missing line doesn't prove pending wrong
            # but increment age? we keep count; if missing persists, pending will timeout eventually
            innov_x = raw_bottom_x - pred_x if has_line else 0.0
            innov_theta = 0.0
            innov_w = 0.0
            is_large = False
            status = "held_no_line"
            return self._build_output(raw, raw_conf, innov_x, innov_theta, innov_w, is_large, status)

        # compute innovations
        innov_x = raw_bottom_x - pred_x
        innov_theta = wrapToPi(raw_theta - pred_theta)
        # width innovation relative
        innov_w = raw_width - pred_w

        # thresholds
        thresh_x = max(p.min_bottom_jump_px,
                       max(p.max_bottom_jump_frac * p.image_width,
                           p.max_bottom_jump_width_frac * max(p.min_width_px, pred_w)))
        thresh_theta = math.radians(p.max_heading_jump_deg)
        thresh_w = p.max_width_change_frac * max(p.min_width_px, pred_w)

        is_large_x = abs(innov_x) > thresh_x
        is_large_theta = abs(innov_theta) > thresh_theta
        is_large_w = abs(innov_w) > thresh_w

        is_large = is_large_x or is_large_theta or is_large_w

        # confidence gating: low raw_conf also counts as large
        # but we already handle via innovation; still drop confidence when raw_conf very low
        low_evidence = raw_conf < 0.35

        if is_large or low_evidence:
            # check persistence – is this same large jump as pending?
            same_as_pending = False
            if self._pending_raw is not None:
                dx_p = abs(raw_bottom_x - self._pending_raw["raw_bottom_x"])
                dt_p = abs(wrapToPi(raw_theta - self._pending_raw["raw_theta"]))
                dw_p = abs(raw_width - self._pending_raw["raw_width"])
                # coherence window = fraction of thresholds
                if (dx_p < p.pending_coherence_frac * thresh_x and
                    dt_p < p.pending_coherence_frac * thresh_theta + 1e-6 and
                    dw_p < p.pending_coherence_frac * thresh_w + 1e-6):
                    same_as_pending = True

            if self._pending_raw is None or not same_as_pending:
                # new suspicious candidate
                self._pending_raw = {"raw_bottom_x": raw_bottom_x,
                                     "raw_theta": wrapToPi(raw_theta),
                                     "raw_width": raw_width}
                self._pending_count = 1
            else:
                # same candidate persists
                # running average for pending to allow slow drift
                pr = self._pending_raw
                a = 0.3
                pr["raw_bottom_x"] = (1 - a) * pr["raw_bottom_x"] + a * raw_bottom_x
                pr["raw_theta"] = wrapToPi((1 - a) * pr["raw_theta"] + a * wrapToPi(raw_theta))
                pr["raw_width"] = (1 - a) * pr["raw_width"] + a * raw_width
                self._pending_count += 1

            if self._pending_count >= p.persist_frames:
                # persistent coherent large deviation -> accept gradually
                # EMA towards pending average
                ax = 0.35  # slightly faster to catch up
                self.state.filt_bottom_x = (1 - ax) * pred_x + ax * self._pending_raw["raw_bottom_x"]
                self.state.filt_theta = wrapToPi((1 - ax) * pred_theta + ax * self._pending_raw["raw_theta"])
                self.state.filt_width = (1 - p.alpha_width) * pred_w + p.alpha_width * self._pending_raw["raw_width"]
                self.state.filt_X = self.state.filt_bottom_x - p.image_width / 2.0
                # confidence recovers somewhat – persistent evidence is trustworthy
                new_conf = float(np.clip(0.6 * raw_conf + 0.4 * self.state.confidence, 0.0, 1.0))
                # boost a little because persistence beats noise
                new_conf = float(np.clip(new_conf + 0.08, 0.0, 1.0))
                self.state.confidence = (1 - p.alpha_conf) * self.state.confidence + p.alpha_conf * new_conf
                self.state.status = "pending_accepted"
                status = "pending_accepted"
                # reset pending after accept – but keep state, next large will be new candidate
                self._pending_raw = None
                self._pending_count = 0
            else:
                # hold / tiny nudge
                ap = p.pending_alpha
                self.state.filt_bottom_x = (1 - ap) * pred_x + ap * raw_bottom_x
                self.state.filt_theta = wrapToPi((1 - ap) * pred_theta + ap * wrapToPi(raw_theta))
                self.state.filt_width = (1 - 0.05) * pred_w + 0.05 * raw_width
                self.state.filt_X = self.state.filt_bottom_x - p.image_width / 2.0
                # confidence drops
                new_conf = float(np.clip(raw_conf * 0.5 + self.state.confidence * 0.5 * 0.7, 0.0, 1.0))
                self.state.confidence = float(np.clip((1 - p.alpha_conf) * self.state.confidence + p.alpha_conf * new_conf * 0.7, 0.05, 1.0))
                self.state.status = "pending"
                status = "pending"
        else:
            # normal small innovation – accept with EMA
            self.state.filt_bottom_x = (1 - p.alpha_x) * pred_x + p.alpha_x * raw_bottom_x
            self.state.filt_theta = wrapToPi((1 - p.alpha_theta) * pred_theta + p.alpha_theta * wrapToPi(raw_theta))
            self.state.filt_width = (1 - p.alpha_width) * pred_w + p.alpha_width * raw_width
            self.state.filt_X = self.state.filt_bottom_x - p.image_width / 2.0
            # confidence EMA
            new_conf = float(np.clip(raw_conf, 0.0, 1.0))
            self.state.confidence = (1 - p.alpha_conf) * self.state.confidence + p.alpha_conf * new_conf
            # boost toward 1 if evidence strong and innovation tiny
            if abs(innov_x) < 0.4 * thresh_x and abs(innov_theta) < 0.4 * thresh_theta:
                self.state.confidence = float(np.clip(self.state.confidence + 0.02, 0.0, 1.0))
            self.state.status = "accepted"
            status = "accepted"
            self._pending_raw = None
            self._pending_count = 0

        self.state.frame_idx = self._frame
        self.state.time_last = now
        self.state.pending_count = self._pending_count

        return self._build_output(raw, raw_conf, innov_x, innov_theta, innov_w, is_large, status)

    # --------------------------------------------------------------
    def _build_output(self, raw, raw_conf, innov_x, innov_theta, innov_w, is_large, status):
        s = self.state
        # feature for controller: F = [X, Y, Theta] ; Y is not critical, use 0 or h/4
        # controller uses X and Theta. Use image_height/4 as placeholder Y.
        F = np.array([s.filt_X, self.params.image_height / 4.0, s.filt_theta], dtype=float)
        # filtered bottom point for drawing: (filt_bottom_x, bottom_y)
        bottom_y = self.params.image_height - 1  # full image bottom (will add offset handling outside)
        # filtered top point via heading: Q.x = P.x + Y_span * tan(theta)
        # Y_span = image_height (approx). Use safe tan clip.
        y_span = self.params.image_height
        tan_t = math.tan(float(np.clip(s.filt_theta, -math.radians(45), math.radians(45))))
        top_x = s.filt_bottom_x + y_span * tan_t * -1  # because Y = P.y - Q.y positive upward; Q.x = P.x + Y * tan(theta)? sign check from mr_vs.
        # In mr_vs: X = Q.x - P.x = Y*tan(theta) . So Q.x = P.x + Y*tan(theta)
        # Our top is upward, so Q.x = P.x + y_span * tan(theta)
        # But filtered theta sign: wrap. Keep same formula.
        top_x = s.filt_bottom_x + (-y_span) * tan_t  # actually Y = P.y - Q.y = y_span, so Q.x = P.x + y_span*tan(theta)? test with theta>0 (line tilts right at top): Q.x > P.x, tan positive -> Q.x > P.x correct. So top_x = filt_bottom_x + y_span*tan(theta)
        # correct sign: top_x = filt_bottom_x + y_span*tan(theta)  (since Y = y_span)
        top_x = s.filt_bottom_x + y_span * tan_t

        return {
            "filt_bottom_x": float(s.filt_bottom_x),
            "filt_theta": float(s.filt_theta),
            "filt_theta_deg": float(math.degrees(s.filt_theta)),
            "filt_width": float(s.filt_width),
            "filt_X": float(s.filt_X),
            "filt_F": F,
            "filt_P": (float(s.filt_bottom_x), float(bottom_y)),
            "filt_Q": (float(top_x), 0.0),
            "confidence": float(s.confidence),
            "raw_confidence": float(raw_conf),
            "status": status,
            "is_large": bool(is_large),
            "innovation_x": float(innov_x),
            "innovation_theta_deg": float(math.degrees(innov_theta)),
            "innovation_width": float(innov_w),
            "pending_count": int(self._pending_count),
            "frame_idx": int(self._frame),
            "raw": raw,
            "has_line": bool(raw.get("has_line", False)),
        }

    def get_state(self) -> Optional[TemporalState]:
        return self.state

class CommandSmoother:
    """
    Smooths w command: low-pass + rate-limit + deadband + optional confidence-based v scaling.
    Stateful per sequence.
    """
    def __init__(self, params: Optional[CommandSmootherParams] = None):
        self.params = params or CommandSmootherParams()
        self.prev_w: float = 0.0
        self.prev_v: float = 0.0
        self.prev_time: Optional[float] = None

    def reset(self):
        self.prev_w = 0.0
        self.prev_v = 0.0
        self.prev_time = None

    def smooth(self, w_raw: float, v_raw: float, confidence: float = 1.0, dt: Optional[float] = None) -> Tuple[float, float, Dict]:
        p = self.params
        now = time.perf_counter()
        if dt is None:
            if self.prev_time is None:
                dt = 0.05
            else:
                dt = now - self.prev_time
                dt = max(1e-3, min(0.5, dt))
        self.prev_time = now

        # low-pass
        w_lpf = (1 - p.w_alpha) * self.prev_w + p.w_alpha * w_raw

        # rate limit
        max_delta = p.max_w_rate * dt
        delta = w_lpf - self.prev_w
        if abs(delta) > max_delta:
            w_lpf = self.prev_w + math.copysign(max_delta, delta)

        # deadband
        if abs(w_lpf) < p.w_deadband:
            w_lpf = 0.0

        # confidence-based v scaling
        v_out = v_raw
        if p.v_conf_scale:
            # linear scale from v_min_scale at conf 0 to 1 at conf 1
            scale = p.v_min_scale + (1.0 - p.v_min_scale) * float(np.clip(confidence, 0.0, 1.0))
            v_out = v_raw * scale

        info = {
            "w_raw": float(w_raw),
            "w_lpf": float(w_lpf),
            "delta": float(delta),
            "dt": float(dt),
            "confidence": float(confidence),
            "v_raw": float(v_raw),
            "v_out": float(v_out),
        }
        self.prev_w = w_lpf
        self.prev_v = v_out
        return w_lpf, v_out, info
