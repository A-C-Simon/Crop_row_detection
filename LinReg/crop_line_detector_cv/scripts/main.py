from cropLineDetector import *

import cv2
import numpy as np
import argparse
import csv
import glob
import os

# The interpolated polynomial's degree
POLY_DEGREE = 1


def process_video(path, viz_options, display):
    """Run the detector on a video file, frame by frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open video: {path}")
        return

    detector = None
    while True:
        ret, image = cap.read()
        if not ret:
            break

        if detector is None:
            detector = cropLineDetector(original_frame=image,
                                        poly_deg=POLY_DEGREE,
                                        viz_options=viz_options,
                                        display=display)

        # The purpose of this whole class is to determine the heading angle error
        heading_angle_error = detector.get_heading_angle_error(image)

        # Correct using the determined angle!
        print(heading_angle_error)

        # This waits for a key press to advance for debugging purposes
        if display and cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    if display:
        cv2.destroyAllWindows()


def process_images(paths, results_dir, viz_options):
    """Run the detector on a set of standalone images and save the overlays.

    Every photo is an independent scene (no temporal continuity), so each one
    gets its own detector instance. Runs fully headless: results are written
    to <results_dir> instead of being shown in GUI windows.
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "crop_line_detection_data.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "heading_angle_error_rad"])

    ok, skipped = 0, 0
    for path in paths:
        img = cv2.imread(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if img is None:
            print(f"[ERROR] Cannot read '{path}'")
            skipped += 1
            continue

        try:
            detector = cropLineDetector(original_frame=img,
                                        poly_deg=POLY_DEGREE,
                                        viz_options=viz_options,
                                        display=False)
            angle = detector.get_heading_angle_error(img)

            out_path = os.path.join(results_dir, f"{name}_result.png")
            result = detector.last_frame
            if result is None:
                # DRAW_FINAL_RESULT not enabled; grab it by enabling the flag
                detector = cropLineDetector(original_frame=img,
                                            poly_deg=POLY_DEGREE,
                                            viz_options=DRAW_FINAL_RESULT,
                                            display=False)
                detector.get_heading_angle_error(img)
                result = detector.last_frame
            cv2.imwrite(out_path, result)

            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([name, f"{angle:.6f}"])
            print(f"[OK] {name}: heading angle error = {np.degrees(angle):.2f} deg")
            ok += 1
        except Exception as e:
            print(f"[ERROR] Failed on '{name}': {e}")
            skipped += 1

    print(f"\nDone. {ok} succeeded, {skipped} failed/skipped.")
    print(f"Results saved to: {os.path.abspath(results_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop line detection on images or video.")
    parser.add_argument("--input",
                        help="Video file OR folder/glob of images "
                             "(default: repo's own demo video)")
    parser.add_argument("--results_dir", default="./results",
                        help="Folder for saved overlays + CSV (default: ./results)")
    parser.add_argument("--display", action="store_true",
                        help="Show GUI windows (video mode only)")
    args = parser.parse_args()

    if args.input is None:
        input_path = os.path.join(os.path.dirname(__file__), "..", "images", "crops.mp4")
    else:
        input_path = args.input

    viz_options = (DRAW_FINAL_RESULT | DRAW_CENTER_ESTIMATIONS |
                   DRAW_ANGLE_ERROR_ON_IMAGE)

    if os.path.isfile(input_path) and input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(input_path, viz_options, display=args.display or False)
        return

    if os.path.isdir(input_path):
        paths = sorted(
            glob.glob(os.path.join(input_path, "*.png"))
            + glob.glob(os.path.join(input_path, "*.jpg"))
            + glob.glob(os.path.join(input_path, "*.jpeg")))
    else:
        paths = sorted(glob.glob(input_path))

    if not paths:
        print(f"No images/video found for input: {input_path}")
        return
    process_images(paths, args.results_dir, viz_options)


if __name__ == '__main__':
    main()
