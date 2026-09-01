Crop-row detection overlays from visual-crop-row-navigation_ros2 on ../Photos (23 images).

Method: detection core extracted verbatim from src/agribot_vs_nodehandler.cpp
CropRow_Tracking() into a standalone binary (no ROS needed):
  HSV inRange (H 40-80, S 50-255, V 100-150) -> findContours -> contour
  centers -> neighbourhood window (120x250 at image centre) -> cv::fitLine.
Params are the repo's own params/agribot_vs_run.yaml; frames resized to the
configured 640x480. Extracted source: /tmp/opencode/vcrn_core/vcrn_core.cpp

Overlay legend:
  green dots = contour centers, red line = fitted row/steering line,
  orange rectangle = tracking window.

Notes / caveats found during testing:
- bev6: HSV mask matches zero pixels with default params -> no line.
- filterContures is degenerate as shipped (center_min_off/max_off hardcoded 0).
- FitLineOnContures emits no line when border intersections land a fraction of
  a pixel outside the frame; a 2 px tolerance was applied to test the method.
- Output is ONE steering line per frame (visual servoing), not per-row maps.

------------------------------------------------------------------------------
PER-ROW TRACES (*_traced.png, one red polyline per crop row)
------------------------------------------------------------------------------
The repo's servoing design only ever draws a single steering line, so per-row
traces were produced separately (script: /tmp/opencode/trace_rows_v5.py):

  1. ExG (2g/(r+g+b) excess green) -> Otsu threshold -> 5x5 close / 3x3 open.
  2. Dominant row orientation found by rotating -90..+90 deg in 3 deg steps
     and maximising column-profile peak prominence.
  3. Rotated so rows stand vertical; horizontal slabs (64 px, 50% overlap);
     per-slab column profile -> Gaussian smooth -> scipy find_peaks
     (distance 18 px, prominence 7% of max).
  4. Peaks linked across slabs greedily (nearest neighbour, 26 px/slab
     tolerance); duplicate/fragmented tracks collapsed by union-find
     (mean column distance < 26 px on shared slabs) and columns averaged
     per slab, so each row yields exactly one polyline; >= 2 slabs required.
  5. Polylines mapped back through the inverse rotation and drawn in red on
     the green-tinted photo.

Near-field priority (rover navigation): forward-facing shots (photo_*) are
analysed on the bottom 55% of the frame only (NEAR_FRAC=0.55). The upper part,
where rows converge to the vanishing point, is ignored - decisions are made
from the soil closest to the robot frame; far points are handled as the rover
approaches. BEV frames are processed full-frame.

Results (angle = estimated dominant orientation of the analysed region,
cov = % of each traced line riding on/near the green mask):
  bev 0deg 15 rows | bev2 3deg 11 | bev4 9deg 5 | bev5 -90deg 6 |
  bev6 -90deg 10 | bev7 90deg 12 | photo_1 -3deg 19 | photo_2 -72deg 2 |
  photo_3 -9deg 18 | photo_4 3deg 11 | photo_5 -9deg 14 | photo_6 18deg 10 |
  photo_7 -21deg 16 | photo_8 24deg 14 | photo_9 12deg 23 | photo_10 -3deg 23 |
  photo_11 -6deg 9 | photo_12 3deg 9 | photo_13 3deg 10 | photo_14 -12deg 22 |
  photo_15 -15deg 22 | photo_16 -3deg 25 | photo_17 -3deg 14
Coverage 88-100%. See _contact_sheet.png for an overview.
