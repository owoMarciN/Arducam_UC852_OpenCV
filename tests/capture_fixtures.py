"""
capture_fixtures.py — Standalone script to capture 20 fixture images
from the Arducam UC852 (OV9782) at 1280x800.

Usage:
    python3 capture_fixtures.py

Images are saved to ./fixture_images/fixture_00.jpg … fixture_19.jpg
"""

import os
import cv2

CAMERA_INDEX = 2
WIDTH = 1280
HEIGHT = 800
COUNT = 20
OUTPUT_DIR = "fixture_images"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {CAMERA_INDEX}. "
            "Ensure the Arducam UC852 is connected and /dev/video2 exists."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened. Resolution: {actual_w}x{actual_h}")

    saved = []
    i = 0
    while i < COUNT:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [WARNING] Frame {i} read failed, retrying…")
            continue

        path = os.path.join(OUTPUT_DIR, f"fixture_{i:02d}.jpg")
        ok = cv2.imwrite(path, frame)
        if ok:
            print(f"  Saved {path}")
            saved.append(path)
            i += 1
        else:
            print(f"  [WARNING] Failed to write {path}, retrying…")

    cap.release()
    print(f"\nDone. {len(saved)}/{COUNT} images saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()