#!/usr/bin/env python3
"""Weed-robust crop-row detection using crop colour + global row geometry.

This is a companion to the adaptive MultiROI implementation.  It addresses
two cases in which vegetation-density/outlier logic is fundamentally weak:

* weeds are as dense as (or denser than) the maize;
* a straight bare footpath competes with genuinely curved crop rows.

The detector first builds a *maize-likelihood* image.  Vegetation is gated by
ExG, while yellow-green/light pixels receive more weight than the darker,
bluer grass in this field.  It then finds candidate row ridges in horizontal
strips and uses dynamic programming to select a globally coherent pair.  The
energy combines crop evidence, the repeated-row pattern, perspective spacing,
centre/width continuity, and (for video) a weak temporal prior.  Curves are
fit as x(y), so near-vertical and curved rows remain well conditioned.

No statistical assumption is made that weeds are rare.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class RowState:
    left: float
    right: float
    score: float

    @property
    def center(self) -> float:
        return 0.5 * (self.left + self.right)

    @property
    def width(self) -> float:
        return self.right - self.left


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def maize_likelihood(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (soft maize likelihood, general vegetation mask, crop mask).

    Lab b* and HSV hue separate the visibly yellow-green maize from the
    darker/bluer grass.  ExG prevents light soil from being classified as
    crop.  Both absolute and frame-adaptive terms are used: the absolute
    terms encode the supplied field's appearance, while robust centring
    tolerates exposure changes along the video.
    """
    small_blur = cv2.GaussianBlur(bgr, (3, 3), 0)
    bf, gf, rf = cv2.split(small_blur.astype(np.float32))
    exg = 2.0 * gf - rf - bf
    exg8 = np.clip(exg, 0, 255).astype(np.uint8)
    otsu, veg = cv2.threshold(exg8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(small_blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(small_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab_b = lab[..., 2]

    vsel = veg > 0
    if np.count_nonzero(vsel) > 200:
        # The lower-hue / higher-b* tail contains the lighter maize.  Robust
        # frame statistics keep the rule stable as auto-exposure changes.
        h_med = float(np.median(hue[vsel]))
        b_med = float(np.median(lab_b[vsel]))
        v_med = float(np.median(val[vsel]))
        h_mad = max(2.5, 1.4826 * float(np.median(np.abs(hue[vsel] - h_med))))
        b_mad = max(4.0, 1.4826 * float(np.median(np.abs(lab_b[vsel] - b_med))))
        v_mad = max(15.0, 1.4826 * float(np.median(np.abs(val[vsel] - v_med))))
    else:
        h_med, b_med, v_med = 46.0, 150.0, 130.0
        h_mad, b_mad, v_mad = 5.0, 8.0, 30.0

    adaptive = (0.95 * (h_med - hue) / h_mad
                + 0.90 * (lab_b - b_med) / b_mad
                + 0.18 * (val - v_med) / v_mad)
    # Absolute separation observed in this maize/grass sequence.  It is a
    # soft cue, not a hard threshold: geometry may retain shaded maize.
    absolute = ((45.5 - hue) / 4.5 + (lab_b - 150.0) / 8.0
                + 0.10 * (val - 125.0) / 35.0)
    green_gate = _sigmoid((exg - max(5.0, float(otsu))) / 7.0)
    crop_prob = green_gate * _sigmoid(0.70 * adaptive + 0.55 * absolute)
    crop_prob *= np.clip((sat + 35.0) / 150.0, 0.35, 1.0)

    # Retain the strongest crop-colour third of vegetation.  This threshold
    # does not presume weeds are sparse; it only selects the crop-like mode.
    if np.count_nonzero(vsel) > 200:
        cut = max(0.42, float(np.quantile(crop_prob[vsel], 0.62)))
    else:
        cut = 0.5
    crop_mask = ((crop_prob >= cut) & vsel).astype(np.uint8) * 255
    crop_mask = cv2.morphologyEx(
        crop_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return crop_prob.astype(np.float32), veg, crop_mask


class ColorGeometryRowDetector:
    """Select the central crop-row pair by global dynamic programming."""

    def __init__(self, n_strips: int = 18, top_frac: float = 0.25,
                 max_side: int = 960, temporal_weight: float = 0.55):
        self.n_strips = max(10, int(n_strips))
        # Bottom-anchored processing coverage.  The default top_frac=0.25
        # means the blurry/far upper quarter is neither scored nor drawn;
        # curves cover y=[0.25H, H), matching MultiROI's 0.75 coverage.
        self.top_frac = float(np.clip(top_frac, 0.0, 0.90))
        self.max_side = max(0, int(max_side))
        self.temporal_weight = float(max(0.0, temporal_weight))
        self._previous: list[tuple[float, float, float]] | None = None

    def reset(self) -> None:
        self._previous = None

    @staticmethod
    def _sample_profile(profile: np.ndarray, x: float) -> float:
        return float(np.interp(x, np.arange(profile.size), profile))

    def _states_for_strip(self, profile: np.ndarray, t: float,
                          previous: tuple[float, float] | None) -> list[RowState]:
        w = profile.size
        vmax = max(float(profile.max()), 1e-6)
        norm = profile / vmax
        min_peak_dist = max(7, int(w * (0.022 + 0.012 * t)))
        peaks, props = find_peaks(norm, distance=min_peak_dist,
                                  prominence=0.025, height=0.055)
        # Keep high-evidence candidates but always include local maxima near
        # the temporal row predictions and the image centre neighbourhood.
        order = list(peaks[np.argsort(norm[peaks])[::-1]][:18])
        probes = [w * q for q in np.linspace(0.08, 0.92, 13)]
        if previous is not None:
            probes += [previous[0], previous[1]]
        radius = max(8, int(0.035 * w))
        for p in probes:
            lo, hi = max(0, int(p) - radius), min(w, int(p) + radius + 1)
            if hi > lo:
                order.append(lo + int(np.argmax(norm[lo:hi])))
        xs = sorted(set(int(x) for x in order if 2 <= x < w - 2))

        # Perspective-aware but deliberately broad limits.  Global
        # continuity determines the actual row spacing.
        min_sep = w * (0.035 + 0.105 * t)
        max_sep = w * (0.18 + 0.38 * t)
        states: list[RowState] = []
        for ia, a in enumerate(xs):
            for b in xs[ia + 1:]:
                sep = b - a
                if sep < min_sep or sep > max_sep:
                    continue
                c = 0.5 * (a + b)
                evidence = norm[a] + norm[b]
                # Repeated-row support: a real adjacent pair is commonly
                # accompanied by another ridge one spacing away.  A bare
                # path has no crop-coloured boundary evidence and gets none.
                repeat = 0.5 * (self._sample_profile(norm, a - sep)
                                + self._sample_profile(norm, b + sep))
                center_pen = ((c - 0.5 * w) / (0.27 * w)) ** 2
                score = 2.35 * evidence + 0.55 * repeat - 1.65 * t * center_pen
                # Prefer adjacent rows: vegetation evidence halfway between
                # two alleged rows suggests the pair skipped a real row.
                il = int(round(a + 0.22 * sep))
                ir = int(round(b - 0.22 * sep)) + 1
                middle = float(norm[il:ir].max()) if ir > il else self._sample_profile(norm, c)
                score -= 1.35 * max(0.0, middle - 0.55 * min(norm[a], norm[b]))
                states.append(RowState(float(a), float(b), float(score)))

        if previous is not None:
            pl, pr = previous
            # Guarantee a state close to the temporal prediction even if a
            # shaded row does not pass peak prominence in this frame.
            a = int(np.clip(pl, 2, w - 3))
            b = int(np.clip(pr, a + min_sep, w - 3))
            if b > a:
                ev = norm[a] + norm[b]
                states.append(RowState(float(a), float(b), float(1.7 * ev)))
        return sorted(states, key=lambda s: s.score, reverse=True)[:80]

    def _select_path(self, profiles: list[np.ndarray], ys: np.ndarray,
                     h: int, w: int) -> list[RowState] | None:
        # Layers are bottom -> top.  That anchors the selected corridor at
        # the robot/camera and lets row spacing converge toward the horizon.
        layers: list[list[RowState]] = []
        for k, (profile, y) in enumerate(zip(profiles, ys)):
            t = float(y / max(1, h - 1))
            prev_pair = None
            if self._previous and k < len(self._previous):
                prev_pair = (self._previous[k][0], self._previous[k][1])
            layer = self._states_for_strip(profile, t, prev_pair)
            if not layer:
                return None
            layers.append(layer)

        # Beam-form dynamic programming retains the preceding direction as
        # state.  This adds a second-difference/curvature term: the optimum
        # may bend smoothly, but cannot migrate one row sideways over a few
        # strips merely because the horizon is ambiguous.
        beams = []
        for j, s in enumerate(layers[0]):
            cost = -s.score
            if self._previous:
                pl, pr, _ = self._previous[0]
                cost += self.temporal_weight * (
                    ((s.left - pl) / (0.08 * w)) ** 2
                    + ((s.right - pr) / (0.08 * w)) ** 2)
            beams.append((float(cost), [j], 0.0, 0.0))

        for k in range(1, len(layers)):
            cur, prev = layers[k], layers[k - 1]
            expanded = []
            for cost, indices, last_dc, last_dw in beams:
                p = prev[indices[-1]]
                for j, s in enumerate(cur):
                    dc = s.center - p.center
                    dw = s.width - p.width
                    expand_up = max(0.0, dw)
                    trans = (1.35 * (dc / (0.075 * w)) ** 2
                             + 0.85 * (dw / max(18.0, 0.28 * p.width)) ** 2
                             + 1.8 * (expand_up / max(18.0, 0.22 * p.width)) ** 2)
                    if k >= 2:
                        trans += (1.15 * ((dc - last_dc) / (0.045 * w)) ** 2
                                  + 0.30 * ((dw - last_dw) /
                                            max(14.0, 0.20 * p.width)) ** 2)
                    if abs(dc) > 0.18 * w or abs(dw) > 0.55 * p.width:
                        trans += 25.0
                    value = cost + trans - s.score
                    if self._previous and k < len(self._previous):
                        pl, pr, _ = self._previous[k]
                        value += self.temporal_weight * 0.35 * (
                            ((s.left - pl) / (0.10 * w)) ** 2
                            + ((s.right - pr) / (0.10 * w)) ** 2)
                    expanded.append((float(value), indices + [j], dc, dw))
            expanded.sort(key=lambda item: item[0])
            # Preserve endpoint diversity; otherwise hundreds of beams can
            # collapse onto one locally attractive (but premature) state.
            per_end: dict[int, int] = {}
            beams = []
            for item in expanded:
                end = item[1][-1]
                if per_end.get(end, 0) >= 2:
                    continue
                beams.append(item)
                per_end[end] = per_end.get(end, 0) + 1
                if len(beams) >= 120:
                    break
            if not beams:
                return None

        best = min(beams, key=lambda item: item[0])
        return [layers[k][j] for k, j in enumerate(best[1])]

    @staticmethod
    def _smooth_curve(ys_bottom_up: np.ndarray, xs_bottom_up: np.ndarray,
                      h: int, w: int, y_min: int = 0) -> list[tuple[float, float]]:
        # Smooth the discrete globally-selected ridge first, then use PCHIP.
        # Unlike an unconstrained cubic smoothing spline, PCHIP cannot ring
        # or create a sideways hook between otherwise valid strip points.
        order = np.argsort(ys_bottom_up)
        ya = np.asarray(ys_bottom_up, np.float64)[order]
        xa = np.asarray(xs_bottom_up, np.float64)[order]
        xa = gaussian_filter1d(xa, sigma=1.15, mode="nearest")
        spl = PchipInterpolator(ya, xa, extrapolate=True)
        yy = np.arange(max(0, int(y_min)), h, 4, dtype=np.float64)
        # PCHIP extrapolation over the half-strip from the first strip centre
        # to the exact coverage boundary is shape preserving and short.
        xx = spl(np.minimum(yy, ya[-1]))
        # Linear tangent extension to the image bottom.
        if yy.size and ya[-1] < h - 1:
            d = float(spl.derivative()(ya[-1]))
            m = yy > ya[-1]
            xx[m] = float(spl(ya[-1])) + np.clip(d, -2.0, 2.0) * (yy[m] - ya[-1])
        xx = np.clip(xx, 0, w - 1)
        return [(float(x), float(y)) for x, y in zip(xx, yy)]

    @staticmethod
    def _tls_line(points: list[tuple[float, float]]) -> tuple[float, float] | None:
        if len(points) < 2:
            return None
        p = np.asarray(points, np.float64)
        mean = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - mean, full_matrices=False)
        dx, dy = vt[0]
        if abs(dx) < 1e-7:
            slope = 1e6
        else:
            slope = float(dy / dx)
        return slope, float(mean[1] - slope * mean[0])

    def detect(self, bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        oh, ow = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(oh, ow)) if self.max_side else 1.0
        if scale < 1.0:
            work = cv2.resize(bgr, (int(round(ow * scale)), int(round(oh * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            work = bgr
        h, w = work.shape[:2]
        prob, veg, crop = maize_likelihood(work)

        y_top = int(round(self.top_frac * h))
        edges = np.linspace(y_top, h, self.n_strips + 1).astype(int)
        profiles: list[np.ndarray] = []
        ys: list[float] = []
        # bottom -> top
        for k in range(self.n_strips - 1, -1, -1):
            y1, y2 = int(edges[k]), int(edges[k + 1])
            band_prob = prob[y1:y2]
            band_crop = crop[y1:y2].astype(np.float32) / 255.0
            # Crop colour dominates; binary support preserves shaded leaf
            # fragments.  A mild horizontal blur forms row ridges.
            p = (0.78 * band_prob.sum(axis=0)
                 + 0.22 * band_crop.sum(axis=0)) / max(1, y2 - y1)
            sigma = max(2.0, w * (0.006 + 0.004 * ((y1 + y2) / (2.0 * h))))
            profiles.append(gaussian_filter1d(p, sigma=sigma, mode="nearest"))
            ys.append(0.5 * (y1 + y2))
        ys_arr = np.asarray(ys, np.float64)
        path = self._select_path(profiles, ys_arr, h, w)

        if path is None:
            # Safe fallback: centred corridor.  This is preferable to
            # following a no-evidence bare path.
            widths = w * (0.12 + 0.32 * ys_arr / h)
            path = [RowState(w / 2 - d / 2, w / 2 + d / 2, 0.0) for d in widths]

        # Temporal EMA only after global selection; it cannot make weeds win
        # the current-frame energy and merely removes video jitter.
        if self._previous and len(self._previous) == len(path):
            alpha = 0.60
            smoothed = []
            for s, (pl, pr, _) in zip(path, self._previous):
                smoothed.append(RowState(alpha * s.left + (1 - alpha) * pl,
                                         alpha * s.right + (1 - alpha) * pr,
                                         s.score))
            path = smoothed
        self._previous = [(s.left, s.right, s.score) for s in path]

        left_x = np.array([s.left for s in path])
        right_x = np.array([s.right for s in path])
        center_x = 0.5 * (left_x + right_x)
        left_curve = self._smooth_curve(ys_arr, left_x, h, w, int(edges[0]))
        right_curve = self._smooth_curve(ys_arr, right_x, h, w, int(edges[0]))
        nav_curve = self._smooth_curve(ys_arr, center_x, h, w, int(edges[0]))

        # Convert work coordinates back to original image coordinates.
        inv = 1.0 / scale
        def unscale_curve(curve):
            return [(x * inv, y * inv) for x, y in curve]
        left_curve_o = unscale_curve(left_curve)
        right_curve_o = unscale_curve(right_curve)
        nav_curve_o = unscale_curve(nav_curve)
        q = [(float(x * inv), float(y * inv)) for x, y in zip(center_x, ys_arr)]
        left_pts = [(float(x * inv), float(y * inv)) for x, y in zip(left_x, ys_arr)]
        right_pts = [(float(x * inv), float(y * inv)) for x, y in zip(right_x, ys_arr)]
        nav_line = self._tls_line(q)
        det_lines = []
        for pts in (left_pts, right_pts):
            line = self._tls_line(pts)
            if line is not None:
                det_lines.append((line[0], line[1], len(pts)))

        rois = []
        profile = []
        for i, (s, y) in enumerate(zip(path, ys_arr)):
            half_h = (edges[1] - edges[0]) / 2.0
            y1, y2 = max(0.0, y - half_h), min(float(h), y + half_h)
            margin = 0.10 * s.width
            rois.append(((s.left - margin) * inv, (s.right + margin) * inv,
                         y1 * inv, y2 * inv))
            profile.append({
                "mu": i + 1, "y_top": int(y1 * inv), "y_bot": int(y2 * inv),
                "y_center": float(y * inv), "center_x": float(s.center * inv),
                "width": float(s.width * inv), "left_x": float(s.left * inv),
                "right_x": float(s.right * inv), "left_edge": float(s.left * inv),
                "right_edge": float(s.right * inv), "two_sided": True,
                "accepted_nav": True, "suppressed": False,
                "roi": rois[-1], "crop_energy": float(s.score),
            })

        # Upscale diagnostic masks once, keeping rendering compatible with
        # the original-resolution input.
        if scale < 1.0:
            veg_o = cv2.resize(veg, (ow, oh), interpolation=cv2.INTER_NEAREST)
            crop_o = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_NEAREST)
            prob_o = cv2.resize((prob * 255).astype(np.uint8), (ow, oh),
                                interpolation=cv2.INTER_LINEAR)
        else:
            veg_o, crop_o, prob_o = veg, crop, (prob * 255).astype(np.uint8)

        widths_o = [(s.width * inv) for s in path]
        result = {
            "binary": crop_o,
            "binary_raw": veg_o,
            "binary_opened": veg_o,
            "crop_likelihood": prob_o,
            "crop_offset": (0, 0),
            "rois": rois,
            "q": q, "q_feed": [(x, y, widths_o[i]) for i, (x, y) in enumerate(q)],
            "q_accepted": q, "q_rejected": [],
            "nav_line": nav_line, "nav_curve": nav_curve_o,
            "left_curve": left_curve_o, "right_curve": right_curve_o,
            "det_lines": det_lines, "strip_profile": profile,
            "vertical_coverage": 1.0 - self.top_frac,
            "median_width": float(np.median(widths_o)),
            "bottom_width": float(widths_o[0]),
            "raw_bottom_x": float(q[0][0]),
            "n_two_sided": len(path), "n_q_accepted": len(path),
            "n_q_rejected": 0, "weed_pressure": float(np.mean(veg_o > 0)
                                                        - np.mean(crop_o > 0)),
            "time_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return result


def draw_result(bgr: np.ndarray, res: dict) -> tuple[np.ndarray, np.ndarray]:
    overlay = bgr.copy()
    mask = cv2.cvtColor(res["binary"], cv2.COLOR_GRAY2BGR)

    def poly(curve, color, thick):
        pts = np.asarray([(round(x), round(y)) for x, y in curve], np.int32)
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, color, thick, cv2.LINE_AA)
            cv2.polylines(mask, [pts], False, color, thick, cv2.LINE_AA)

    poly(res["left_curve"], (255, 80, 0), 4)
    poly(res["right_curve"], (255, 80, 0), 4)
    # Match the original MultiROI navigation-line convention: light blue
    # in BGR, rather than the former yellow line.
    nav_color = (255, 200, 0)
    poly(res["nav_curve"], nav_color, 5)
    for p in res["strip_profile"]:
        c = (int(round(p["center_x"])), int(round(p["y_center"])))
        cv2.circle(overlay, c, 4, nav_color, -1, cv2.LINE_AA)
        cv2.circle(mask, c, 4, nav_color, -1, cv2.LINE_AA)

    cv2.putText(overlay, "blue: crop rows  light blue: navigation centre",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 4,
                cv2.LINE_AA)
    cv2.putText(overlay, "blue: crop rows  light blue: navigation centre",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                cv2.LINE_AA)
    return overlay, mask


def make_composite(bgr: np.ndarray, res: dict) -> np.ndarray:
    overlay, mask = draw_result(bgr, res)
    veg = cv2.cvtColor(res["binary_raw"], cv2.COLOR_GRAY2BGR)
    crop = cv2.applyColorMap(res["crop_likelihood"], cv2.COLORMAP_TURBO)
    h, w = bgr.shape[:2]

    def label(im, text):
        out = im.copy()
        cv2.rectangle(out, (0, 0), (w, 38), (30, 30, 30), -1)
        cv2.putText(out, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.67,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    top = np.hstack([label(veg, "All vegetation (ExG)"),
                     label(crop, "Maize likelihood: colour, not density")])
    bottom = np.hstack([label(mask, "Global curved row-pair solution"),
                        label(overlay, "Crop-row overlay")])
    sep = np.full((4, top.shape[1], 3), 255, np.uint8)
    return np.vstack([top, sep, bottom])


def run_image(path: str, output_dir: str, detector: ColorGeometryRowDetector) -> str:
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    res = detector.detect(image)
    out = os.path.join(output_dir, f"{Path(path).stem}_color_geometry_composite.png")
    if not cv2.imwrite(out, make_composite(image, res)):
        raise RuntimeError(f"Could not write image: {out}")
    return out


def run_video(path: str, output_dir: str, detector: ColorGeometryRowDetector,
              preview: bool = False) -> tuple[str, str]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    stem = Path(path).stem
    out_path = os.path.join(output_dir, f"{stem}_color_geometry_composite.mp4")
    csv_path = os.path.join(output_dir, f"{stem}_color_geometry.csv")
    writer = None
    frame_no = 0
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detector.detect(frame)
        comp = make_composite(frame, res)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (comp.shape[1], comp.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video: {out_path}")
        writer.write(comp)
        rows.append((frame_no, frame_no / fps, res["raw_bottom_x"],
                     res["bottom_width"], res["median_width"],
                     res["time_ms"], res["weed_pressure"]))
        frame_no += 1
        if preview:
            cv2.imshow("Colour + global geometry crop rows", cv2.resize(comp, None,
                       fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 100 == 0:
            print(f"processed {frame_no} frames", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    with open(csv_path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "time_s", "bottom_x", "bottom_width",
                      "median_width", "processing_ms", "weed_pressure"])
        out.writerows(rows)
    return out_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image or video")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--n-strips", type=int, default=18)
    parser.add_argument("--vertical-coverage", type=float, default=0.75,
                        help="Bottom-anchored image fraction used for detection and drawing (default: 0.75)")
    parser.add_argument("--top-frac", type=float, default=None,
                        help="Deprecated inverse of --vertical-coverage; when supplied it takes precedence")
    parser.add_argument("--max-side", type=int, default=960,
                        help="Internal processing size; output stays original")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    top_frac = (args.top_frac if args.top_frac is not None
                else 1.0 - float(np.clip(args.vertical_coverage, 0.10, 1.0)))
    detector = ColorGeometryRowDetector(
        n_strips=args.n_strips, top_frac=top_frac, max_side=args.max_side,
        temporal_weight=0.0 if args.no_temporal else 0.55)
    suffix = Path(args.input).suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
        video, data = run_video(args.input, args.output, detector, args.preview)
        print(f"video: {video}\ncsv:   {data}")
    else:
        print(f"image: {run_image(args.input, args.output, detector)}")


if __name__ == "__main__":
    main()
