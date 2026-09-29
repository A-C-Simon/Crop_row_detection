# Color-geometry crop-row results — 75% vertical coverage

This folder contains a fixed-parameter run over every supported file in
`CropRowNav/Dataset`:

- 9 JPG images → 9 diagnostic PNG composites
- 3 MP4 videos → 3 H.264 diagnostic MP4 composites
- 3 per-frame CSV files

## Configuration

- `n_strips=18`
- `vertical_coverage=0.75`, bottom anchored
- upper 25% excluded from both detection and drawing
- internal processing `max_side=960` for images and `512` for videos
- color-geometry global optimizer enabled
- temporal prior enabled for videos and disabled for independent images
- dark blue: selected crop-row boundaries
- light blue: navigation centre

## Video verification

| Input suffix | Frames | Duration (s) | CSV rows | P95 centre jump (px) | Mean processing (ms) | Output size |
|---|---:|---:|---:|---:|---:|---:|
| `0750` | 837 | 27.9037 | 837 | 16.92 | 418.0 | 54.6 MiB |
| `0762` | 661 | 22.0040 | 661 | 17.71 | 390.9 | 50.2 MiB |
| `0775` | 1563 | 52.1417 | 1563 | 13.66 | 447.5 | 114.1 MiB |

All videos were decoded after H.264 compression and their frame counts match
their diagnostic CSV row counts. Representative frames distributed across each
video were visually checked after encoding.

## Reproduction

```bash
python3 ../color_geometry_crop_rows.py \
  --input /path/to/image-or-video \
  --output . \
  --vertical-coverage 0.75 \
  --max-side 512
```

Add `--no-temporal` for independent still images. The generated OpenCV video
can optionally be transcoded with FFmpeg/libx264 for smaller storage, as done
for the three delivered videos.
