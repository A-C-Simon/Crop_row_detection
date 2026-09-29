# Weed-robust crop-row detector

`color_geometry_crop_rows.py` is a field-adapted companion to the original
MultiROI detector. It fixes the supplied failure case without assuming that
weeds are rare.

## Why the old method fails

ExG plus column density describes *vegetation*, not maize. When grass is the
majority class, an Isolation Forest correctly treats maize as the unusual
population, which is the opposite of the desired semantic decision. Local
MultiROI updates can then lock onto the wrong band, and unconstrained linear
regression can prefer a long straight footpath over curved rows.

## New scoring model

1. ExG only separates vegetation from soil.
2. A soft maize likelihood combines HSV hue, Lab b* (yellow/blue), brightness,
   and frame-adaptive robust statistics. This uses the observed lighter,
   yellower maize versus darker grass; it does not use population size.
3. Each horizontal strip supplies possible crop-row ridges.
4. Dynamic programming selects an adjacent row pair across all strips. Its
   energy rewards crop-colour evidence and repeated row spacing, and penalizes
   off-centre selection, implausible perspective changes, and curvature
   discontinuity.
5. Shape-preserving PCHIP curves are fitted as x(y). This remains conditioned
   for near-vertical rows and cannot overshoot like a free cubic spline.
6. Only the bottom-anchored 75% of each frame is processed and drawn. The
   blurry/far upper 25% cannot distort the row curves or rover command.
7. Video adds a weak temporal prior and EMA after spatial selection.

The dark-blue curves in outputs are selected crop rows. The light-blue curve is
their navigation centre, matching the original MultiROI colour convention. The
diagnostic panels deliberately show all ExG vegetation beside maize likelihood
so failures can be audited.

## Reproduce

```bash
python3 color_geometry_crop_rows.py \
  --input /home/ac/Crop_Row_Detection_Techniques/CropRowNav/Dataset/document_5314402612911580775.mp4 \
  --output . --max-side 512 --vertical-coverage 0.75

python3 color_geometry_crop_rows.py \
  --input /home/ac/Crop_Row_Detection_Techniques/CropRowNav/Dataset/photo_5314402613371544472_y.jpg \
  --output . --no-temporal
```

## Generated validation artifacts

- `document_5314402612911580775_color_geometry_composite.mp4`
- `document_5314402612911580775_color_geometry.csv`
- `photo_5314402613371544472_y_color_geometry_composite.png`

The validated video contains 1,563 frames at 3747/125 fps and is 52.1417 s.

## Research basis

- Vidovic, Cupec, and Hocenski, *Crop row detection by global energy
  minimization*, Pattern Recognition 55 (2016), 68-86,
  DOI 10.1016/j.patcog.2016.01.013: dynamic programming combines image
  evidence with geometric priors and supports curved rows.
- Montalvo et al., *Automatic detection of crop rows in maize fields with high
  weeds pressure*, Expert Systems with Applications 39 (2012): vegetation
  indices plus double thresholding before row fitting.
- Zheng et al., *Maize and weed classification using color indices with
  support vector data description in outdoor fields*, Computers and
  Electronics in Agriculture 141 (2017), 215-222: crop/weed colour indices can
  remain discriminative when ordinary RGB/vegetation segmentation cannot.

## Limitations and next step

The maize-colour cue is calibrated to this camera/field and adapts to exposure,
but a major change in cultivar, growth stage, lighting spectrum, or white
balance can reduce separation. For deployment across fields, annotate crop and
weed pixels from multiple days and replace `maize_likelihood()` with a trained
semantic model; keep the global row-pair optimizer as the geometry/safety
layer. Quantitative accuracy still requires manually labelled row curves.
