"""
generate_color_brightness_fixtures.py — Synthetic fixtures for ColorDetector
and BrightnessDetector, standing in for real factory photographs until
physical samples are available.

Color fixtures (BGR, 640x400) in tests/fixtures/color/:
    3 normal   (PASS) — color_normal_01.jpg   .. color_normal_03.jpg
    3 anomaly  (FAIL) — color_anomaly_01.jpg  .. color_anomaly_03.jpg

Brightness fixtures (grayscale, 640x400) in tests/fixtures/brightness/:
    2 normal      (PASS) — brightness_normal_01.jpg .. brightness_normal_02.jpg
    1 overexposed (FAIL) — brightness_over_01.jpg
    1 underexposed(FAIL) — brightness_under_01.jpg

Ground truth is encoded in the filename prefix. Output paths are resolved
relative to this script's own location, independent of pytest's working
directory.

Usage:
    python3 generate_color_brightness_fixtures.py
"""

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 400
COLOR_DIR      = Path(__file__).resolve().parent / "fixtures" / "color"
BRIGHTNESS_DIR = Path(__file__).resolve().parent / "fixtures" / "brightness"
SEED = 99

# Matches ColorDetector's default expected hue range (90-130 = blue-ish coating)
NORMAL_HUE  = 110
ANOMALY_HUE = 20   # orange/red — clearly outside the expected coating range


def make_normal_color_panel(rng: np.random.Generator) -> np.ndarray:
    """A uniformly coated panel within the expected HSV range."""
    hsv = np.zeros((HEIGHT, WIDTH, 3), dtype=np.int16)
    hsv[:, :, 0] = NORMAL_HUE + rng.integers(-3, 3, (HEIGHT, WIDTH))
    hsv[:, :, 1] = 180 + rng.integers(-10, 10, (HEIGHT, WIDTH))
    hsv[:, :, 2] = 160 + rng.integers(-10, 10, (HEIGHT, WIDTH))
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.GaussianBlur(bgr, (5, 5), 0)


def add_color_anomaly(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint a wrong-color patch onto an otherwise normal panel."""
    out = frame.copy()
    x = int(rng.integers(80, WIDTH - 220))
    y = int(rng.integers(60, HEIGHT - 140))
    w, h = 140, 100
    patch_hsv = np.zeros((h, w, 3), dtype=np.uint8)
    patch_hsv[:, :, 0] = ANOMALY_HUE
    patch_hsv[:, :, 1] = 200
    patch_hsv[:, :, 2] = 170
    patch_bgr = cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2BGR)
    out[y:y + h, x:x + w] = patch_bgr
    return out


def make_brightness_frame(rng: np.random.Generator, base_level: int, noise_sigma: float = 5.0) -> np.ndarray:
    """Grayscale frame at a fixed base intensity, used for normal/over/under variants."""
    base = np.full((HEIGHT, WIDTH), base_level, dtype=np.int16)
    noise = rng.normal(0, noise_sigma, (HEIGHT, WIDTH)).astype(np.int16)
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def main() -> None:
    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    BRIGHTNESS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    saved = []

    for i in range(1, 4):
        frame = make_normal_color_panel(rng)
        path = COLOR_DIR / f"color_normal_{i:02d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)

    for i in range(1, 4):
        frame = add_color_anomaly(make_normal_color_panel(rng), rng)
        path = COLOR_DIR / f"color_anomaly_{i:02d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)

    for i in range(1, 3):
        frame = make_brightness_frame(rng, base_level=130)
        path = BRIGHTNESS_DIR / f"brightness_normal_{i:02d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)

    frame = make_brightness_frame(rng, base_level=235)
    path = BRIGHTNESS_DIR / "brightness_over_01.jpg"
    cv2.imwrite(str(path), frame)
    saved.append(path)

    frame = make_brightness_frame(rng, base_level=20)
    path = BRIGHTNESS_DIR / "brightness_under_01.jpg"
    cv2.imwrite(str(path), frame)
    saved.append(path)

    print(f"{len(saved)} fixtures saved.")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
