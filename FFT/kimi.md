# kimi.md — Reproducing "Robust Crop Row Detection using DFT for Vision-based In-Field Navigation"

**Goal (from user):** reproduce the code/algorithms from `fft.pdf` (Gai, Guo, Raj, Tang — *Computers and Electronics in Agriculture*, 2024, Elsevier user license). The `Crop-lines-system/` git repo is a related-but-simpler reference implementation (FFT angle detection + line profiling, no sub-pixel refinement, no phase-based positioning) — useful for ideas, NOT a reproduction of this paper.

**Progress log and handover notes are appended at the bottom ("Log" section).**

---

## 1. What the paper does (algorithm summary)

Pipeline (Section 2.2 of the paper):

1. **Image pre-processing**: undistort (camera calibration), then inverse perspective mapping (IPM) via homography H → bird's-eye view; extract MAAIR (maximum axis-aligned inscribed rectangle) as ROI; ROI spatial resolution 0.01 m/pixel; convert to grayscale with Excess Green Index (ExG). Max look-ahead: 16 m for heading, 8 m for position deviation.
2. **2D Hanning window + FFT** → magnitude spectrum f(x,y), phase spectrum p(x,y).
3. **Peak extraction**:
   - Bandpass filter the magnitude spectrum to keep only components with periods close to the known row spacing (76 cm in the paper).
   - Coarse peak = argmax → (x0, y0).
   - Sub-pixel refinement via Taylor expansion (solve H·Δ = −∇f with numeric derivatives) → (u, v) = (x0+Δx, y0+Δy).
   - Phase at (u,v) computed with a single DTFT evaluation → φ.
4. **Row geometry**: periods Tx = M/u, Ty = N/v; line direction l = (−Tx, Ty)/||·||; x-axis intercepts p_k from phase; map back to local map frame; take two rows bracketing the robot (origin), signed perpendicular distances d_k, d_{k+1}.
5. **Deviations**: e_y = (d_k + d_{k+1})/2; e_θ = −atan(l_xM / l_yM).
6. **(Navigation demo)** LQG = LQR + Kalman filter on state (e_y, e_y', e_θ, e_θ'), input = curvature κ, measurements (ê_y, ê_θ).

## 2. Equations (transcribed from rendered PDF pages 10–15, 20)

Frames: F_I image, F_C camera, F_M local map (origin = robot nav reference, ground plane), F_ROI ROI image.

- (2) K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
- (3) s·[xI,yI,1]ᵀ = K·[xC,yC,zC]ᵀ
- (4) [xC,yC,zC,1]ᵀ = T_C^M·[xM,yM,zM,1]ᵀ = (T_M^C)⁻¹·[xM,yM,zM,1]ᵀ
- (5) s·[xI,yI,1]ᵀ = K·[I|0]·T_C^M·[xM,yM,0,1]ᵀ   (I = 3×3 identity)
- (6) T_C^M = [R₃ₓ₃ T₃ₓ₁; 0 1], R elements r_ij, translation (tx,ty,tz)
- (7) H = K·[[r11,r12,tx],[r21,r22,ty],[r31,r32,tz]]
- (8) [xM,yM,1]ᵀ = (1/s)·H⁻¹·[xI,yI,1]ᵀ  ← inverse perspective mapping
- (9) 1D Hanning: w(n) = 0.5 − 0.5·cos(2πn/(M−1)), 0 ≤ n ≤ M−1
- (10) 2D Hanning: w(x,y) = ¼[1−cos(2πx/(M−1))][1−cos(2πy/(N−1))], (M,N) = image size, x ∈ [0,M−1], y ∈ [0,N−1]
- (11) Taylor: ∇f(x0,y0) + H_f(x0,y0)·(Δx,Δy)ᵀ = 0
- (12) (Δx,Δy)ᵀ = −H_f(x0,y0)⁻¹·∇f(x0,y0); derivatives computed numerically; refined peak (u,v) = (x0+Δx, y0+Δy)
- (13) DTFT: F(u,v) = Σ_{m=0}^{M}Σ_{n=0}^{N} f(m,n)·e^(−j2π(um/M + vn/N))  ← f(m,n) here is the WINDOWED ROI image (notation overloaded in paper)
- (14) φ = phase(F(u,v))
- (15) Tx = M/u, (16) Ty = N/v  (pixels)
- (17) l = (lx,ly) = (−Tx,Ty)/||(−Tx,Ty)||  (normalized row direction in ROI frame)
- (18) p_k = [Tx·(2kπ − φ), 0], k ∈ ℤ  ← x-axis intercepts of row lines. **DIMENSIONALLY SUSPECT**: phase zero-crossing condition 2π(um/M+vn/N)+φ = 2kπ gives x-intercept = Tx·(k − φ/2π) = Tx·(2kπ−φ)/(2π). The paper appears to drop the 2π factor (or defines φ in cycles). **Decision: implement the physically correct x_k = Tx·(2kπ−φ)/(2π)** and validate on synthetic data.
- (19) l_{M} = normalize(T_M^ROI·(lx,ly,0)ᵀ)
- (20) p_{kM} = T_M^ROI·(Tx·(2kπ−φ), 0, 1)ᵀ  (same 2π note as above)
- (21) d_k = (x_{kM},y_{kM})·(l_{xM},l_{yM})  ← as printed, dot with the ROW DIRECTION, which is NOT the perpendicular distance unless rows are ⊥ x-axis. **Decision: implement d_k as the true signed perpendicular distance = p_{kM}·n, with n the unit wavefront normal n ∝ (1/Tx, 1/Ty) (mapped to map frame).** This converges to the paper's formula in the near-ideal case and is geometrically correct in general. Logged as a paper typo/inconsistency.
- (22) e_y = (d_k + d_{k+1})/2  (two rows with smallest |d| bracketing the robot)
- (23) e_θ = −atan(l_{xM}/l_{yM})
- (24) r = Pearson correlation; (25) RMSE (standard definitions)
- (26) x_{n+1} = A x_n + B u_n + w;  y_n = C x_n + v;  cov(w)=Q, cov(v)=R
- (27) x = (e_y, e_y', e_θ, e_θ')ᵀ; (28) u = (κ); (29) y = (ê_y, ê_θ)ᵀ
- (30) A = [[1,Δt,0,0],[0,0,s,0],[0,0,1,Δt],[0,0,0,0]], B = (0,0,0,ν)ᵀ·κ — the last B entry is printed as Greek nu; physically it is the curvature→heading-rate gain (≈ forward speed s for a car-like model, e_θ' = s·κ). **Decision: B = (0,0,0,s)ᵀ by default, configurable.** C = [[1,0,0,0],[0,0,1,0]] (implied by eq 29).

Paper parameters: row spacing 76 cm; ROI 0.01 m/px; look-ahead 16 m (heading) / 8 m (position); camera 2048×1536, 4 mm lens, 2.1 m height, 37° downward tilt; robot speed 0.8 m/s; camera 3–4 fps; LQG at 8 Hz. Results: RMSE 6.43 cm / 1.48°, mean abs tracking error 3.74 cm with LQG.

## 3. Implementation plan

New directory `dft_crop_row_detection/` (keep paper, pdf text dump, and reference repo untouched):

```
dft_crop_row_detection/
├── preprocessing.py    # ExG grayscale; undistort+homography IPM (eq 1–8); MAAIR extraction
├── row_detection.py    # Hanning 2D (eq 9–10), FFT, bandpass, coarse peak,
│                       # Taylor refinement (eq 11–12), DTFT phase (eq 13–14),
│                       # row lines (eq 15–18), deviations (eq 19–23)
├── lqg_controller.py   # LQR (discrete Riccati) + Kalman filter (eq 26–30)
├── synthetic.py        # synthetic crop-row image generator (stripes + noise/weeds/gaps) with known ground truth
├── test_detection.py   # validation: recover spacing/angle/phase/deviations on synthetic images
├── demo_example_image.py # run detector on Crop-lines-system/example_image.png (no calibration → skip IPM)
└── README.md           # short usage notes
```

Key engineering decisions / constraints:

- We have **no camera calibration or robot pose data**, so the homography IPM is implemented as a documented utility (build H from K and extrinsics per eq 2–8) but the validation path feeds synthetic / already-rectified images directly into the frequency-domain core, which is the paper's actual contribution.
- Pure Python + numpy/scipy (+opencv if available, else fallback). The paper used "Python using OpenCV and Eigen" (sic).
- Validate the core on synthetic images: known spacing/angle/offset must be recovered to sub-pixel accuracy; robustness check with noise, weed blobs, missing plants, glare/shadow gradients.
- Demo on `Crop-lines-system/example_image.png` (aerial NDVI image, rows visible) — treat row spacing as a configurable bandpass prior.
- LQG: implement + small closed-loop simulation (kinematic car-like model tracking a row centerline) to show deviations converge.

## 4. Log (append-only handover notes)

- **[done]** Extracted paper text (`pdftotext fft.pdf fft.txt`, 2542 lines) and rendered pages 9–16, 20 to `pdf_pages/pg-*.png` (110 dpi) to read equations that the text layer mangles. All equations 2–30 transcribed in §2 above. Paper typos found and decisions logged: eq 18/20 missing 2π factor; eq 21 dot-product with row direction instead of wavefront normal; eq 30 B-matrix "ν" ambiguous (use s).
- **[done]** Read `Crop-lines-system/README.md`. That repo does 2D FFT for row ANGLE only + line profiling for positions; no Hanning window, no Taylor sub-pixel refinement, no DTFT phase, no homography, no deviation/LQG. Reference only.
- **[next]** Check python env (numpy/scipy/cv2), skim `Crop-lines-system/find_gradient.py`/`find_lines.py`, then implement per §3.
