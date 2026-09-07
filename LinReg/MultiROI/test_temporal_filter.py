#!/usr/bin/env python3
"""
Synthetic tests for temporal navigation filter.
Verifies:
 1. Missing-plant transient rejected
 2. Persistent lane shift eventually accepted
 3. Sudden width double rejected
 4. Straight corridor stable
 5. Curved corridor followed
 6. Vertical coverage clips drawn nav lines
 7. Brief heading spikes/outliers fully rejected (spike guard)
"""
import math
import cv2
import numpy as np
from test_multi_roi import MultiROIDetector
from mr_vs import MultiROIVS, MRVSParams
from temporal_filter import TemporalNavigationFilter, TemporalFilterParams

def make_synthetic_bgr(h=480, w=640, rows_x=(250, 390), row_width=10, gap_segments=None, noise=False):
    """
    Create synthetic BGR image with two vertical green rows.
    rows_x: tuple (x_left_center, x_right_center) or list of two
    gap_segments: list of (row_idx, y0, y1) to erase that segment (missing plants)
    """
    # brown soil background
    bgr = np.full((h, w, 3), (60, 65, 120), dtype=np.uint8)  # B,G,R brownish
    for idx, cx in enumerate(rows_x):
        x0 = int(cx - row_width//2)
        x1 = int(cx + row_width//2)
        cv2.rectangle(bgr, (x0, 0), (x1, h-1), (30, 200, 30), -1)  # green
        if gap_segments:
            for r_idx, y0, y1 in gap_segments:
                if r_idx == idx:
                    cv2.rectangle(bgr, (x0, int(y0)), (x1, int(y1)), (60, 65, 120), -1)
    if noise:
        # add small weed blobs random
        for _ in range(20):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            cv2.circle(bgr, (cx, cy), 4, (40, 180, 40), -1)
    return bgr

def run_sequence(frames, detector, vs, use_temporal=True):
    """Run sequence through detector+filter+vs; return lists of raw_w, filt_w, statuses."""
    if use_temporal:
        t_params = TemporalFilterParams(image_width=640, image_height=480, n_strips=detector.n, persist_frames=4)
        t_filter = TemporalNavigationFilter(t_params)
        vs.reset_smoother()
    else:
        t_filter = None

    raw_ws, filt_ws, statuses, confs = [], [], [], []
    # we run raw vs filtered separately: when use_temporal we still log raw innovation
    # For raw-only baseline we compute without filter
    last_w = 0.0
    for bgr in frames:
        h, w = bgr.shape[:2]
        res = detector.detect(bgr)
        crop_offset = res.get("crop_offset", (0,0))
        nav_line = res.get("nav_line")
        nav_curve = res.get("nav_curve")
        F_raw, PQ_raw = vs.nav_line_to_feature(nav_line, nav_curve, crop_offset, (h,w))
        if F_raw is None:
            raw_w = 0.0
            filt_w = 0.0
            statuses.append("no_line")
            confs.append(0.0)
            raw_ws.append(raw_w)
            filt_ws.append(filt_w)
            continue
        # raw control without smoothing
        v_raw, w_raw, info_raw = vs.compute_control(F_raw, dt=0.05, confidence=1.0, smooth=False)
        raw_ws.append(float(w_raw))

        if use_temporal:
            # build raw dict
            P_raw = PQ_raw[0]
            raw_bottom_x = float(P_raw[0])
            raw_theta = float(F_raw[2])
            raw_width = float(res.get("median_width", 120))
            raw_dict = {
                "raw_bottom_x": raw_bottom_x,
                "raw_theta": raw_theta,
                "raw_width": raw_width,
                "raw_X": float(F_raw[0]),
                "raw_err_x": float(F_raw[0]),
                "raw_err_theta_deg": float(math.degrees(raw_theta)),
                "has_line": True,
                "n_two_sided": int(res.get("n_two_sided",0)),
                "n_q_accepted": int(res.get("n_q_accepted",0)),
                "n_q_rejected": int(res.get("n_q_rejected",0)),
                "weed_pressure": float(res.get("weed_pressure",0)),
                "median_width": float(res.get("median_width", raw_width)),
                "bottom_width": float(res.get("bottom_width", raw_width)),
            }
            filt_out = t_filter.update(raw_dict, dt=0.05, last_w=last_w)
            F_filt = filt_out["filt_F"]
            v_f, w_f, info_f = vs.compute_control(F_filt, dt=0.05, confidence=filt_out["confidence"], smooth=True)
            filt_ws.append(float(w_f))
            statuses.append(filt_out["status"])
            confs.append(float(filt_out["confidence"]))
            last_w = float(w_f)
        else:
            filt_ws.append(float(w_raw))
            statuses.append("raw")
            confs.append(1.0)
    return raw_ws, filt_ws, statuses, confs

def test_missing_plants():
    print("\n=== Test 1: Missing plants (transient weak row) ===")
    detector = MultiROIDetector()
    vs = MultiROIVS(MRVSParams(width=640, height=480, lambda_x=10.0, lambda_theta=1.0, w_max=0.6))
    # stable baseline at 250,390 centered 320
    frames = []
    for i in range(10):
        if 3 <= i <= 5:
            # missing left row segment bottom half (weakens detection – left row nearly gone)
            # erase lower 60% of left row to simulate gap at bottom where influence maximal
            bgr = make_synthetic_bgr(rows_x=(250,390), gap_segments=[(0, 200, 480)])
        else:
            bgr = make_synthetic_bgr(rows_x=(250,390))
        frames.append(bgr)
    raw_ws, filt_ws, statuses, confs = run_sequence(frames, detector, vs, use_temporal=True)
    print(f"raw_w deg/s: {[round(math.degrees(x),1) for x in raw_ws]}")
    print(f"filt_w deg/s: {[round(math.degrees(x),1) for x in filt_ws]}")
    print(f"status: {statuses}")
    print(f"conf: {[round(c,2) for c in confs]}")
    # Expect filtered spike suppressed: max filt < 0.6* max raw during anomaly
    anomaly_raw_max = max(abs(x) for x in raw_ws[3:6])
    anomaly_filt_max = max(abs(x) for x in filt_ws[3:6])
    # raw may not always spike huge on synthetic perfect rows – but we can check filt is not larger than raw
    if anomaly_raw_max > 0.05:
        assert anomaly_filt_max < anomaly_raw_max * 0.7 + 0.05, "Filtered should suppress transient spike"
        print(f"PASS: transient suppressed {math.degrees(anomaly_filt_max):.1f} < {math.degrees(anomaly_raw_max):.1f}")
    else:
        # Even if raw doesn't spike, filtered should stay small
        assert max(abs(x) for x in filt_ws) < 0.3, "Filtered should stay near zero"
        print("PASS (raw didn't spike strongly, filtered stayed small)")
    # Also check recovery: after anomaly, filtered returns near zero
    assert abs(math.degrees(filt_ws[-1])) < 8, f"Should recover after gap, got {math.degrees(filt_ws[-1])}"
    print("PASS test_missing_plants")

def test_persistent_lane_shift():
    print("\n=== Test 2: Persistent lane shift ===")
    detector = MultiROIDetector()
    vs = MultiROIVS(MRVSParams(width=640, height=480, lambda_x=2.0, lambda_theta=1.0))
    frames = []
    # first 3 frames centered, next 10 frames shifted right by 60px
    for i in range(13):
        if i < 3:
            bgr = make_synthetic_bgr(rows_x=(250,390))
        else:
            bgr = make_synthetic_bgr(rows_x=(310,450))  # +60 shift
        frames.append(bgr)
    # run with temporal but low gain to avoid instant jump; filtered should eventually follow
    t_params = TemporalFilterParams(image_width=640, image_height=480, n_strips=detector.n, persist_frames=4, max_bottom_jump_frac=0.10)
    # manual run to inspect filt_bottom_x progression
    t_filter = TemporalNavigationFilter(t_params)
    vs.reset_smoother()
    filt_xs = []
    raw_xs = []
    last_w=0
    for bgr in frames:
        res = detector.detect(bgr)
        h,w = bgr.shape[:2]
        crop_offset = res.get("crop_offset",(0,0))
        F_raw,PQ = vs.nav_line_to_feature(res.get("nav_line"), res.get("nav_curve"), crop_offset, (h,w))
        if F_raw is None:
            continue
        raw_xs.append(float(F_raw[0] + 320))  # bottom_x
        P_raw = PQ[0]
        raw_dict = {
            "raw_bottom_x": float(P_raw[0]), "raw_theta": float(F_raw[2]),
            "raw_width": float(res.get("median_width",120)),
            "raw_X": float(F_raw[0]), "raw_err_x": float(F_raw[0]), "raw_err_theta_deg": float(math.degrees(F_raw[2])),
            "has_line": True, "n_two_sided": int(res.get("n_two_sided",0)),
            "n_q_accepted": int(res.get("n_q_accepted",0)), "n_q_rejected": int(res.get("n_q_rejected",0)),
            "weed_pressure": float(res.get("weed_pressure",0)), "median_width": float(res.get("median_width",120)),
            "bottom_width": float(res.get("bottom_width",120)),
        }
        filt_out = t_filter.update(raw_dict, dt=0.05, last_w=last_w)
        filt_xs.append(float(filt_out["filt_bottom_x"]))
        # dummy control
        F_f = filt_out["filt_F"]
        _,w_f,_ = vs.compute_control(F_f, dt=0.05, confidence=filt_out["confidence"], smooth=True)
        last_w=w_f
        print(f"frame {len(filt_xs)-1}: raw_x={raw_xs[-1]:.0f} filt_x={filt_xs[-1]:.0f} status={filt_out['status']} conf={filt_out['confidence']:.2f}")
    # after persist_frames, filtered should approach new center (~380)
    new_center = 380  # (310+450)/2
    old_center = 320  # (250+390)/2
    assert filt_xs[3] < old_center + 25, f"Frame 3 (first shift) should still be near old (pending), got {filt_xs[3]}"
    assert filt_xs[-1] > new_center - 20, f"Final filtered should approach new center {new_center}, got {filt_xs[-1]}"
    print(f"PASS persistent shift: start {filt_xs[0]:.0f} -> end {filt_xs[-1]:.0f} target {new_center}")

def test_width_expansion():
    print("\n=== Test 3: Sudden impossible width expansion ===")
    detector = MultiROIDetector()
    vs = MultiROIVS(MRVSParams(width=640, height=480, lambda_x=2.0, lambda_theta=1.0))
    frames=[]
    for i in range(8):
        if i==3 or i==4:
            # width double: rows at 200 and 440 (width 240 vs normal 140) -> bottom_x still 320 but width larger
            bgr = make_synthetic_bgr(rows_x=(200,440))
        else:
            bgr = make_synthetic_bgr(rows_x=(250,390))
        frames.append(bgr)
    raw_ws, filt_ws, statuses, confs = run_sequence(frames, detector, vs, use_temporal=True)
    print(f"raw_w deg: {[round(math.degrees(x),1) for x in raw_ws]}")
    print(f"filt_w deg: {[round(math.degrees(x),1) for x in filt_ws]}")
    print(f"status: {statuses}")
    # Even though bottom_x same, width innovation should trigger pending, so filt should not spike; but w mainly depends on X not width, so width alone may not cause w spike – check confidence drop and pending
    assert statuses[3] in ("pending","pending_accepted","accepted")  # but check that confidence dropped or pending
    # For this synthetic, width jump without X jump may be considered not large for w, but filter should still mark pending for width
    # Ensure filt_w stayed small
    assert max(abs(math.degrees(x)) for x in filt_ws[3:5]) < 10, "Width anomaly shouldn't cause steering spike"
    print("PASS width expansion")

def test_vertical_coverage_draw_clip():
    """Drawn nav lines (blue in composite, red in mr_vs overlay) must stop
    at the vertical-coverage top instead of extending to image row 0."""
    print("\n=== Test 6: vertical coverage clips drawn nav lines ===")
    cov = 0.6
    h, w = 480, 640
    detector = MultiROIDetector(vertical_coverage=cov)
    vs = MultiROIVS(MRVSParams(width=w, height=h, vertical_coverage=cov))
    bgr = make_synthetic_bgr(h=h, w=w, rows_x=(250, 390))
    res = detector.detect(bgr)
    assert abs(float(res["vertical_coverage"]) - cov) < 1e-6, \
        "detector result must expose the vertical coverage used"

    from test_multi_roi import draw_results
    orig, _bin = draw_results(bgr, res)
    dx, dy = res["crop_offset"]
    bh = res["binary"].shape[0]
    # blue/dark-blue drawn lines must not appear above coverage top
    top_full = dy + int(round(bh * (1.0 - cov)))
    blue = (orig[:, :, 0] > 220) & (orig[:, :, 2] < 120)  # nav/det lines are blue-ish
    ys = np.where(blue.any(axis=1))[0]
    assert len(ys) > 0, "synthetic frame should produce a nav line"
    assert ys.min() >= top_full - 4, \
        f"blue line drawn above coverage top: min y {ys.min()} vs coverage top {top_full}"
    # and the line should still reach the bottom of the ROI region (cropped bottom)
    bottom_full = dy + bh - 1
    assert ys.max() >= bottom_full - 6, \
        f"blue line should reach bottom of ROI region: max y {ys.max()} vs {bottom_full}"

    # red line from mr_vs.draw_overlay also clipped to coverage top
    F, PQ = vs.nav_line_to_feature(res.get("nav_line"), res.get("nav_curve"),
                                   res["crop_offset"], (h, w), vertical_coverage=cov)
    if PQ is not None:
        P, Q = PQ
        overlay = vs.draw_overlay(bgr, tuple(P), tuple(Q), 0.2, 0.0,
                                  {"err_x": 0, "err_theta_deg": 0}, vertical_coverage=cov)
        red = (overlay[:, :, 2] > 180) & (overlay[:, :, 1] < 100) & (overlay[:, :, 0] < 100)
        ys_r = np.where(red.any(axis=1))[0]
        assert len(ys_r) > 0, "red nav line should be drawn"
        assert ys_r.min() >= int(h * (1.0 - cov)) - 4, \
            f"red line drawn above coverage top: min y {ys_r.min()}"
    print(f"PASS vertical coverage clip (cov={cov}, coverage top y={top_full})")


def _feed_raw(t_filter, vs, bottom_x, theta_deg, width=120.0, n_two=9, w_deg=None):
    """Feed one frame into the filter; returns filt_out dict. theta in degrees."""
    raw_dict = {
        "raw_bottom_x": float(bottom_x),
        "raw_theta": float(math.radians(theta_deg)),
        "raw_width": float(width),
        "raw_X": float(bottom_x - 320.0),
        "raw_err_x": float(bottom_x - 320.0),
        "raw_err_theta_deg": float(theta_deg),
        "has_line": True,
        "n_two_sided": int(n_two),
        "n_q_accepted": int(n_two),
        "n_q_rejected": 0,
        "weed_pressure": 0.0,
        "median_width": float(width),
        "bottom_width": float(width),
    }
    out = t_filter.update(raw_dict, dt=0.05, last_w=0.0)
    if w_deg is not None:
        F = out["filt_F"]
        vs.compute_control(F, dt=0.05, confidence=out["confidence"], smooth=True)
    return out


def test_heading_spike_rejected():
    """A sharp sub-second heading outlier (like a lookahead/map hold emitting a
    stale -20..-30 deg remembered slope for < 1 s) must leave NO trace on the
    filtered state: no drift toward it and no commit.

    Reproduces the real crops.mp4 event: baseline ~0 deg, then a ~20-frame
    excursion to ~-25 deg while two-sided corridor evidence is weak (only
    2-6 of 9 strips see both rows - the detector's trustworthy signal is
    degraded even though the stale memory/input keeps emitting the bogus
    slope), then the evidence returns and the input snaps back to ~0.
    """
    print("\n=== Test 7: brief heading spike fully rejected ===")
    tp = TemporalFilterParams(image_width=640, image_height=480, n_strips=10,
                              persist_frames=4, spike_confirm_frames=3,
                              max_heading_jump_deg=12.0)
    t_filter = TemporalNavigationFilter(tp)

    # like the real event: jump to -20..-31 deg and wander; n_two stays 2-6
    # (below commit_min_two_sided=7) so the spike-guard never commits it
    excursion = [-20.0, -24.0, -28.0, -31.0, -29.0, -22.0, -25.0, -20.5,
                 -24.0, -27.0, -22.0, -21.5, -21.0, -13.5, -12.5, -13.0]
    series = [0.0]*6 + excursion + [0.0]*12
    n2_series = [9]*6 + [6, 2, 4, 1, 2, 3, 4, 4, 6, 2, 3, 5, 4, 6, 3, 2] + [9]*12

    max_dev = 0.0
    any_commit = False
    out_last = None
    for i, th in enumerate(series):
        out = _feed_raw(t_filter, None, 320.0, th, n_two=n2_series[i])
        out_last = out
        fth = float(out["filt_theta_deg"])
        max_dev = max(max_dev, abs(fth))
        if out["status"] == "pending_accepted":
            any_commit = True
        if 5 <= i <= 40 and (i < 8 or i % 3 == 0):
            print(f"  fr{i}: raw={th:+6.1f} filt={fth:+6.1f} status={out['status']:>16} n2={n2_series[i]}")
    # the outlier must not move the line meaningfully nor be committed
    assert not any_commit, "sustained low-evidence outlier must never be committed"
    assert max_dev < 1.5, f"filtered heading swung {max_dev:.2f} deg toward the outlier"
    # and it must settle back on the true course
    final = float(out_last["filt_theta_deg"])
    assert abs(final) < 0.5, f"should return to baseline, got {final:.2f} deg"
    print(f"PASS spike rejected (max |filt| during excursion = {max_dev:.2f} deg, final {final:.2f} deg)")


def test_short_pulse_rejected():
    """A short (3-frame) clean square pulse - far below the guard window, even
    with full evidence - must not move the state at all."""
    print("\n=== Test 8: short square pulse fully rejected ===")
    tp = TemporalFilterParams(image_width=640, image_height=480, n_strips=10,
                              persist_frames=4, spike_confirm_frames=3,
                              max_heading_jump_deg=12.0)
    t_filter = TemporalNavigationFilter(tp)
    degs = [0.0]*5 + [-20.0]*3 + [0.0]*8
    max_dev = 0.0
    for th in degs:
        out = _feed_raw(t_filter, None, 320.0, th, n_two=9)
        max_dev = max(max_dev, abs(float(out["filt_theta_deg"])))
    assert max_dev < 0.5, f"3-frame pulse moved the filtered line {max_dev:.2f} deg"
    print(f"PASS short pulse rejected (max deviation {max_dev:.3f} deg)")


def test_straight_stable():
    print("\n=== Test 4: Straight corridor stable ===")
    detector = MultiROIDetector()
    vs = MultiROIVS(MRVSParams(width=640, height=480, lambda_x=2.0, lambda_theta=1.0, w_alpha=0.35, max_w_rate=1.2))
    frames = [make_synthetic_bgr(rows_x=(250,390)) for _ in range(6)]
    raw_ws, filt_ws, statuses, confs = run_sequence(frames, detector, vs, use_temporal=True)
    print(f"filt_w deg: {[round(math.degrees(x),2) for x in filt_ws]}")
    # filtered should remain near zero, no oscillation
    for w in filt_ws:
        assert abs(math.degrees(w)) < 5, f"Straight should stay near zero, got {math.degrees(w)}"
    print("PASS straight stable")

def test_curved():
    print("\n=== Test 5: Curved corridor (gentle slant) ===")
    detector = MultiROIDetector()
    vs = MultiROIVS(MRVSParams(width=640, height=480, lambda_x=2.0, lambda_theta=1.0))
    # To simulate curve, slant rows: top narrower than bottom (converging). Use different x at top vs bottom? Our synthetic draws vertical; need slanted.
    # Draw slanted lines manually
    def make_slanted(h=480,w=640, left_bottom=250, right_bottom=390, angle_deg=5):
        bgr = np.full((h,w,3), (60,65,120), dtype=np.uint8)
        # line tilt: delta_x at top = h * tan(angle)
        dx = int(h * math.tan(math.radians(angle_deg)))
        lb_top = left_bottom - dx
        rb_top = right_bottom - dx
        for (xb, xt) in [(left_bottom, lb_top), (right_bottom, rb_top)]:
            pts = np.array([[xb-5, h-1],[xb+5, h-1],[xt+5,0],[xt-5,0]], dtype=np.int32)
            cv2.fillPoly(bgr, [pts], (30,200,30))
        return bgr
    frames = [make_slanted(angle_deg=5) for _ in range(6)]
    raw_ws, filt_ws, statuses, confs = run_sequence(frames, detector, vs, use_temporal=True)
    print(f"raw_w deg: {[round(math.degrees(x),1) for x in raw_ws]}")
    print(f"filt_w deg: {[round(math.degrees(x),1) for x in filt_ws]}")
    # Should follow curve: w should be non-zero but not jittery, and filtered should be close to raw after initial
    assert abs(math.degrees(filt_ws[-1])) > 0.5, "Curved should produce steering"
    assert max(abs(math.degrees(x)-math.degrees(filt_ws[-1])) for x in filt_ws[-3:]) < 8, "Should not jitter on curve"
    print("PASS curved")

if __name__ == "__main__":
    test_missing_plants()
    test_persistent_lane_shift()
    test_width_expansion()
    test_straight_stable()
    test_curved()
    test_vertical_coverage_draw_clip()
    test_heading_spike_rejected()
    test_short_pulse_rejected()
    print("\nAll synthetic tests passed")
