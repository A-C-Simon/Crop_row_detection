"""Per-row crop tracing - v5: rotated-slab profile tracking.

Topology tracing fails on dense canopy (the green mask is one connected
mesh), so instead:

  1. ExG -> Otsu green mask            (the part that already works well)
  2. estimate dominant row orientation (rotation search maximising
                                        column-profile peakedness)
  3. rotate so rows stand vertical
  4. horizontal slabs -> column profile -> peaks = row centres
  5. link peaks across slabs           -> one polyline per row (curvature ok,
                                          converging rows handled)
  6. rotate back, draw red lines on the green-tinted photo
"""
import os
import time

import numpy as np
import cv2
from scipy.signal import find_peaks

SLAB_H = 64          # px at working scale
OVERLAP = 0.5
PEAK_MIN_DIST = 18   # px between adjacent row centres
TRACK_TOL = 26.0     # max lateral jump between consecutive slabs
MIN_SLABS = 2        # a track must appear in >= this many slabs

# Forward-facing shots: rows converge to a vanishing point near the top of
# the frame. Navigation only needs the near field, so analysis is restricted
# to the bottom NEAR_FRAC of those images; the convergence zone is ignored.
NEAR_FRAC = 0.55
FORWARD_PREFIXES = ("photo",)
MERGE_TOL = 26.0     # tracks closer than this (px) are the same row


def exg_mask(rgb):
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), \
        rgb[..., 2].astype(float)
    s = np.where(r + g + b == 0, 1.0, r + g + b)
    exg = 2.0 * g / s - r / s - b / s
    exg8 = np.clip((exg - exg.min()) / max(exg.ptp(), 1e-9) * 255,
                   0, 255).astype(np.uint8)
    _, bw = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def clean(mask):
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h)), M


def profile_peakedness(prof):
    prof = prof - prof.min()
    if prof.max() <= 0:
        return 0.0
    pk, props = find_peaks(prof, distance=PEAK_MIN_DIST,
                           prominence=0.02 * prof.max())
    prom = np.sort(props["prominences"])[::-1]
    return float(prom[:12].sum())


def estimate_angle(mask):
    best_deg, best_score = 0.0, -1.0
    for deg in range(-90, 91, 3):
        rot, _ = rotate(mask, deg)
        prof = rot.mean(axis=0)
        sc = profile_peakedness(prof)
        if sc > best_score:
            best_score, best_deg = sc, deg
    return best_deg


def slab_peaks(rot_mask):
    h, w = rot_mask.shape
    step = max(int(SLAB_H * (1 - OVERLAP)), 8)
    detections = []          # list over slabs of [(col, strength)]
    ycenters = []
    y = 0
    while y < h:
        y2 = min(y + SLAB_H, h)
        if y2 - y < 12:
            break
        prof = rot_mask[y:y2].mean(axis=0)
        prof = prof - prof.min()
        if prof.max() > 1e-6:
            prof_s = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 3).ravel()
            pk, props = find_peaks(prof_s, distance=PEAK_MIN_DIST,
                                   prominence=0.07 * prof_s.max())
            detections.append(list(zip(pk.tolist(),
                                       props["prominences"].tolist())))
        else:
            detections.append([])
        ycenters.append((y + y2) / 2)
        if y2 >= h:
            break
        y += step
    return ycenters, detections


def link_tracks(ycenters, detections):
    """Greedy nearest-neighbour tracking down the slabs."""
    tracks = []              # each: list of (slab_idx, col)
    active = []              # [track_idx, last_col, last_slab]
    for si, det in enumerate(detections):
        cols = sorted(det, key=lambda t: -t[1])
        claimed = set()
        # extend existing tracks first
        for ai in range(len(active)):
            ti, last_col, last_si = active[ai]
            if si - last_si > 2:      # track stale
                continue
            tol = TRACK_TOL * (si - last_si)
            cands = [c for c, _ in cols
                     if abs(c - last_col) <= tol and c not in claimed]
            if cands:
                c = min(cands, key=lambda c: abs(c - last_col))
                tracks[ti].append((si, c))
                claimed.add(c)
                active[ai] = [ti, c, si]
        # new tracks from unclaimed strong peaks
        for c, s in cols:
            if c in claimed:
                continue
            tracks.append([(si, c)])
            active.append([len(tracks) - 1, c, si])
        # drop dead tracks from active
        active = [a for a in active if si - a[2] <= 2]

    # collapse duplicate/fragmented tracks: union-find over pairs whose mean
    # column distance on shared slabs is below MERGE_TOL, then average
    # columns per slab so each row yields exactly one polyline
    n = len(tracks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    maps = [dict(tr) for tr in tracks]
    for i in range(n):
        for j in range(i + 1, n):
            common = set(maps[i]) & set(maps[j])
            if not common:
                continue
            d = np.mean([abs(maps[i][s] - maps[j][s]) for s in common])
            if d < MERGE_TOL:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for members in clusters.values():
        by_slab = {}
        for m in members:
            for s, c in tracks[m]:
                by_slab.setdefault(s, []).append(c)
        combined = sorted((s, float(np.mean(cs)))
                          for s, cs in by_slab.items())
        merged.append(combined)
    return merged


def main():
    photos = sorted(os.listdir("/home/ac/Crop_Row_Detection_Techniques/Photos"))
    outdir = ("/home/ac/Crop_Row_Detection_Techniques/ExG/"
              "visual-crop-row-navigation_ros2/results")
    print(f"{'image':40s} {'angle':>6s} {'rows':>5s} {'cov':>5s} {'secs':>6s}")
    for p in photos:
        t0 = time.time()
        bgr = cv2.imread(os.path.join(
            "/home/ac/Crop_Row_Detection_Techniques/Photos", p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        sc = min(1.0, 1000.0 / max(h, w))
        rgb_s = cv2.resize(rgb, (int(w * sc), int(h * sc))) if sc < 1 else rgb.copy()

        mask = clean(exg_mask(rgb_s))
        h_s, w_s = mask.shape
        forward = os.path.basename(p).lower().startswith(FORWARD_PREFIXES)
        y0 = int((1.0 - NEAR_FRAC) * h_s) if forward else 0
        roi = mask[y0:, :]

        deg = estimate_angle(roi)
        rot, M = rotate(roi, deg)

        ycenters, dets = slab_peaks(rot)
        tracks = link_tracks(ycenters, dets)

        # build polylines in rotated frame, map back to original
        Minv = cv2.invertAffineTransform(M)
        polys = []
        for tr in tracks:
            if len(tr) < MIN_SLABS:
                continue
            pts_r = np.array([[c, ycenters[si]] for si, c in tr],
                             dtype=np.float32)
            n = len(pts_r)
            if n >= 4:      # light smoothing along the track
                x = pts_r[:, 0].copy()
                k = np.ones(3) / 3
                x[1:-1] = np.convolve(x, k, "valid")
                pts_r[:, 0] = x
            pts_o = cv2.transform(pts_r.reshape(-1, 1, 2), Minv).ravel()
            pts_o = pts_o.reshape(-1, 2)
            pts_o[:, 1] += y0          # back to full-frame coordinates
            polys.append(pts_o)

        # coverage: fraction of each polyline riding on/near green
        md = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
        covs = []
        for pl in polys:
            pi = pl.astype(np.float32)
            seg = np.sqrt((np.diff(pi[:, 0]) ** 2 +
                           np.diff(pi[:, 1]) ** 2))
            L = seg.sum()
            if L < 1:
                continue
            n = max(int(L / 4), 2)
            t = np.linspace(0, len(pi) - 1, n)
            xs = np.interp(t, np.arange(len(pi)), pi[:, 0])
            ys = np.interp(t, np.arange(len(pi)), pi[:, 1])
            xi = np.clip(xs.round().astype(int), 0, md.shape[1] - 1)
            yi = np.clip(ys.round().astype(int), 0, md.shape[0] - 1)
            covs.append(md[yi, xi].mean())
        cov = float(np.mean(covs)) if covs else 0.0

        vis = rgb_s.copy()
        tint = vis.copy()
        tint[mask > 0] = (0.6 * tint[mask > 0] +
                          0.4 * np.array([80, 220, 80])).astype(np.uint8)
        vis = tint
        for pl in polys:
            cv2.polylines(vis, [pl.astype(np.int32)], False,
                          (255, 0, 0), 3, cv2.LINE_AA)
        name = os.path.splitext(p)[0]
        cv2.imwrite(os.path.join(outdir, f"{name}_traced.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"{name:40s} {deg:6d} {len(polys):5d} "
              f"{cov*100:5.0f}% {time.time()-t0:6.1f}")


if __name__ == "__main__":
    main()
