"""
test_color_brightness_detector.py — Unit tests for color.py and brightness.py.

Includes the Week 4 accuracy gates: both detectors must correctly classify
>= 80% of their respective fixture sets.

Run with:
    pytest test_color_brightness_detector.py -v -s
"""

import glob
import os

import cv2
import numpy as np
import pytest

from detectors.base import DetectionResult, DetectorError
from detectors.color import ColorDetector
from detectors.brightness import BrightnessDetector
from generate_color_brightness_fixtures import (
    main as generate_fixtures,
    COLOR_DIR,
    BRIGHTNESS_DIR,
)

ACCURACY_TARGET = 0.80


# ---------------------------------------------------------------------------
# Ensure fixture images exist before this module's tests run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_fixture_images():
    color_missing = not os.path.isdir(COLOR_DIR) or len(glob.glob(f"{COLOR_DIR}/*.jpg")) == 0
    brightness_missing = not os.path.isdir(BRIGHTNESS_DIR) or len(glob.glob(f"{BRIGHTNESS_DIR}/*.jpg")) == 0
    if color_missing or brightness_missing:
        generate_fixtures()


# ---------------------------------------------------------------------------
# In-memory synthetic frames for unit-level tests
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_color_frame():
    """640x400 BGR frame within the expected coating color range."""
    rng = np.random.default_rng(3)
    hsv = np.zeros((400, 640, 3), dtype=np.int16)
    hsv[:, :, 0] = 110 + rng.integers(-3, 3, (400, 640))
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 160
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.fixture
def anomaly_color_frame(normal_color_frame):
    """Normal panel with a wrong-color patch painted on it."""
    out = normal_color_frame.copy()
    patch = np.zeros((100, 140, 3), dtype=np.uint8)
    patch[:, :, 0] = 20   # hue
    patch[:, :, 1] = 200  # sat
    patch[:, :, 2] = 170  # val
    out[100:200, 200:340] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)
    return out


@pytest.fixture
def normal_brightness_frame():
    """640x400 grayscale frame with a mid-range mean intensity."""
    rng = np.random.default_rng(11)
    base = np.full((400, 640), 130, dtype=np.int16)
    noise = rng.normal(0, 5, (400, 640)).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def overexposed_frame():
    rng = np.random.default_rng(12)
    base = np.full((400, 640), 235, dtype=np.int16)
    noise = rng.normal(0, 5, (400, 640)).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def underexposed_frame():
    rng = np.random.default_rng(13)
    base = np.full((400, 640), 20, dtype=np.int16)
    noise = rng.normal(0, 5, (400, 640)).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ColorDetector — pipeline steps
# ---------------------------------------------------------------------------

class TestColorDetectorSteps:
    def test_name(self):
        assert ColorDetector().name == "color"

    def test_requires_color_flag(self):
        assert ColorDetector.requires_color is True

    def test_anomaly_mask_shape(self, normal_color_frame):
        det = ColorDetector()
        mask = det.compute_anomaly_mask(normal_color_frame)
        assert mask.shape == normal_color_frame.shape[:2]

    def test_find_regions_empty_on_blank_mask(self):
        det = ColorDetector()
        blank = np.zeros((400, 640), dtype=np.uint8)
        assert det.find_regions(blank) == []

    def test_build_region_type_is_color(self):
        region = ColorDetector.build_region((10, 10, 50, 50))
        assert region.type == "color"

    def test_compute_score_no_defects(self):
        assert ColorDetector.compute_score([]) == 1.0

    def test_invalid_frame_raises_on_grayscale_input(self):
        det = ColorDetector()
        with pytest.raises(DetectorError):
            det.detect(np.zeros((400, 640), dtype=np.uint8))  # missing color channel


# ---------------------------------------------------------------------------
# ColorDetector — end to end
# ---------------------------------------------------------------------------

class TestColorDetectorEndToEnd:
    def test_normal_panel_passes(self, normal_color_frame):
        result = ColorDetector().detect(normal_color_frame)
        assert result.passed is True
        assert result.score == 1.0

    def test_anomaly_panel_fails(self, anomaly_color_frame):
        result = ColorDetector().detect(anomaly_color_frame)
        assert result.passed is False
        assert any(d.type == "color" for d in result.defects)

    def test_returns_detection_result(self, normal_color_frame):
        result = ColorDetector().detect(normal_color_frame)
        assert isinstance(result, DetectionResult)


# ---------------------------------------------------------------------------
# BrightnessDetector — pipeline steps
# ---------------------------------------------------------------------------

class TestBrightnessDetectorSteps:
    def test_name(self):
        assert BrightnessDetector().name == "brightness"

    def test_requires_color_flag(self):
        assert BrightnessDetector.requires_color is False

    def test_classify_normal_returns_none(self, normal_brightness_frame):
        det = BrightnessDetector()
        mean_val = det.compute_mean(normal_brightness_frame)
        assert det.classify(mean_val, 0.0, 0.0) is None

    def test_classify_overexposed(self):
        det = BrightnessDetector()
        assert det.classify(mean_val=240, high_frac=0.0, low_frac=0.0) == "overexposed"

    def test_classify_underexposed(self):
        det = BrightnessDetector()
        assert det.classify(mean_val=10, high_frac=0.0, low_frac=0.0) == "underexposed"

    def test_invalid_frame_raises_on_color_input(self):
        det = BrightnessDetector()
        with pytest.raises(DetectorError):
            det.detect(np.zeros((400, 640, 3), dtype=np.uint8))  # not grayscale


# ---------------------------------------------------------------------------
# BrightnessDetector — end to end
# ---------------------------------------------------------------------------

class TestBrightnessDetectorEndToEnd:
    def test_normal_frame_passes(self, normal_brightness_frame):
        result = BrightnessDetector().detect(normal_brightness_frame)
        assert result.passed is True
        assert result.score == 1.0

    def test_overexposed_frame_fails(self, overexposed_frame):
        result = BrightnessDetector().detect(overexposed_frame)
        assert result.passed is False
        assert result.defects[0].type == "brightness"

    def test_underexposed_frame_fails(self, underexposed_frame):
        result = BrightnessDetector().detect(underexposed_frame)
        assert result.passed is False
        assert result.defects[0].type == "brightness"

    def test_full_frame_region_on_defect(self, overexposed_frame):
        result = BrightnessDetector().detect(overexposed_frame)
        d = result.defects[0]
        h, w = overexposed_frame.shape
        assert (d.w, d.h) == (w, h)


# ---------------------------------------------------------------------------
# Week 4 deliverable gate — accuracy >= 80% on each fixture set
# ---------------------------------------------------------------------------

class TestFixtureAccuracy:
    def test_color_accuracy_at_least_80_percent(self):
        det = ColorDetector()
        paths = sorted(glob.glob(f"{COLOR_DIR}/*.jpg"))
        assert len(paths) == 6, f"Expected 6 color fixtures, found {len(paths)}"

        correct = 0
        for path in paths:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            result = det.detect(frame)
            expected_pass = "normal" in os.path.basename(path)
            if result.passed == expected_pass:
                correct += 1

        accuracy = correct / len(paths)
        print(f"\nColorDetector fixture accuracy: {correct}/{len(paths)} = {accuracy:.1%}")
        assert accuracy >= ACCURACY_TARGET

    def test_brightness_accuracy_at_least_80_percent(self):
        det = BrightnessDetector()
        paths = sorted(glob.glob(f"{BRIGHTNESS_DIR}/*.jpg"))
        assert len(paths) == 4, f"Expected 4 brightness fixtures, found {len(paths)}"

        correct = 0
        for path in paths:
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            result = det.detect(frame)
            expected_pass = "normal" in os.path.basename(path)
            if result.passed == expected_pass:
                correct += 1

        accuracy = correct / len(paths)
        print(f"BrightnessDetector fixture accuracy: {correct}/{len(paths)} = {accuracy:.1%}")
        assert accuracy >= ACCURACY_TARGET
