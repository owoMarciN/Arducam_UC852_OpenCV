"""
generate_surface_fixtures.py — Generates synthetic surface fixture images
for testing SurfaceDetector, standing in for real factory photographs
until physical samples are available.

Produces 10 grayscale images (640x400):
    5 clean   (PASS)  — surface_clean_01.jpg   .. surface_clean_05.jpg
    3 scratch (FAIL)  — surface_scratch_01.jpg .. surface_scratch_03.jpg
    2 dust    (FAIL)  — surface_dust_01.jpg    .. surface_dust_02.jpg

Ground truth is encoded in the filename prefix (clean / scratch / dust)
so tests can parse expected pass/fail without a separate manifest.

Usage:
    python3 generate_surface_fixtures.py
"""

import os

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 400
OUTPUT_DIR = "../fixtures_surface"
SEED = 42

CLEAN_COUNT   = 5
SCRATCH_COUNT = 3
DUST_COUNT    = 2


def make_clean_surface(rng: np.random.Generator) -> np.ndarray:
    """Flat metallic-looking surface: low-amplitude noise, heavily blurred."""
    base = np.full((HEIGHT, WIDTH), 150, dtype=np.uint8)
    noise = rng.normal(0, 4, (HEIGHT, WIDTH)).astype(np.int16)
    frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    return frame


def add_scratch(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one long thin dark line across the surface."""
    out = frame.copy()
    x1 = int(rng.integers(0, WIDTH // 3))
    y1 = int(rng.integers(0, HEIGHT))
    x2 = int(rng.integers(2 * WIDTH // 3, WIDTH))
    y2 = int(rng.integers(0, HEIGHT))
    cv2.line(out, (x1, y1), (x2, y2), color=40, thickness=2)
    return out


def add_dust(frame: np.ndarray, rng: np.random.Generator, count: int = 4) -> np.ndarray:
    """Draw several small compact dark specks scattered on the surface."""
    out = frame.copy()
    for _ in range(count):
        cx = int(rng.integers(20, WIDTH - 20))
        cy = int(rng.integers(20, HEIGHT - 20))
        radius = int(rng.integers(4, 7))
        cv2.circle(out, (cx, cy), radius, color=30, thickness=-1)
    return out


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    saved = []

    for i in range(1, CLEAN_COUNT + 1):
        frame = make_clean_surface(rng)
        path = os.path.join(OUTPUT_DIR, f"surface_clean_{i:02d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)

    for i in range(1, SCRATCH_COUNT + 1):
        frame = make_clean_surface(rng)
        frame = add_scratch(frame, rng)
        path = os.path.join(OUTPUT_DIR, f"surface_scratch_{i:02d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)

    for i in range(1, DUST_COUNT + 1):
        frame = make_clean_surface(rng)
        frame = add_dust(frame, rng)
        path = os.path.join(OUTPUT_DIR, f"surface_dust_{i:02d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)

    print(f"{len(saved)} fixtures saved to ./{OUTPUT_DIR}/")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
