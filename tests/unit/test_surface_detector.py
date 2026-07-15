"""
test_surface_detector.py — Unit tests for base.py and surface.py.

Includes the Week 3 accuracy gate: SurfaceDetector must score >= 80%
correct PASS/FAIL classification on the fixture set.

Run with:
    pytest test_surface_detector.py -v -s
"""

import glob
import os

import cv2
import numpy as np
import pytest

from detectors.base import DefectRegion, Detector, DetectionResult, DetectorError
from detectors.surface import SurfaceDetector
from generate_surface_fixtures import main as generate_fixtures, OUTPUT_DIR

ACCURACY_TARGET = 0.80


# ---------------------------------------------------------------------------
# Fixtures (pytest fixtures, not to be confused with the image fixture set)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_fixture_images():
    """Generate the synthetic fixture image set once before this module runs."""
    if not os.path.isdir(OUTPUT_DIR) or len(glob.glob(f"{OUTPUT_DIR}/*.jpg")) == 0:
        generate_fixtures()


@pytest.fixture
def clean_frame():
    """640x400 uint8 grayscale frame with no defects."""
    rng = np.random.default_rng(1)
    base = np.full((400, 640), 150, dtype=np.uint8)
    noise = rng.normal(0, 4, (400, 640)).astype(np.int16)
    frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(frame, (9, 9), 0)


@pytest.fixture
def scratch_frame(clean_frame):
    """Clean frame with one elongated, near-horizontal scratch drawn across it."""
    out = clean_frame.copy()
    cv2.line(out, (50, 180), (590, 220), color=40, thickness=2)
    return out


@pytest.fixture
def dust_frame(clean_frame):
    """Clean frame with several small compact specks drawn on it."""
    out = clean_frame.copy()
    for cx, cy in [(100, 100), (300, 200), (500, 300), (200, 350)]:
        cv2.circle(out, (cx, cy), 5, color=30, thickness=-1)
    return out


# ---------------------------------------------------------------------------
# base.py — DefectRegion
# ---------------------------------------------------------------------------

class TestDefectRegion:
    def test_valid_region(self):
        d = DefectRegion(x=1, y=2, w=10, h=5, confidence=0.9, type="scratch")
        assert d.area == 50

    def test_invalid_type_raises(self):
        with pytest.raises(DetectorError):
            DefectRegion(x=0, y=0, w=1, h=1, confidence=0.5, type="rust")

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(DetectorError):
            DefectRegion(x=0, y=0, w=1, h=1, confidence=1.5, type="dust")

    def test_nonpositive_dimensions_raise(self):
        with pytest.raises(DetectorError):
            DefectRegion(x=0, y=0, w=0, h=1, confidence=0.5, type="dust")


# ---------------------------------------------------------------------------
# base.py — DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def test_passed_true_when_no_defects(self):
        r = DetectionResult(score=1.0, defects=[])
        assert r.passed is True

    def test_passed_false_when_defects_present(self):
        d = DefectRegion(x=0, y=0, w=5, h=5, confidence=0.8, type="dust")
        r = DetectionResult(score=0.5, defects=[d])
        assert r.passed is False

    def test_score_out_of_range_raises(self):
        with pytest.raises(DetectorError):
            DetectionResult(score=1.5, defects=[])


# ---------------------------------------------------------------------------
# base.py — Detector ABC
# ---------------------------------------------------------------------------

class TestDetectorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Detector()

    def test_subclass_must_implement_detect_and_name(self):
        class Incomplete(Detector):
            pass
        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# surface.py — pipeline steps
# ---------------------------------------------------------------------------

class TestSurfaceDetectorSteps:
    def test_name(self):
        assert SurfaceDetector().name == "surface"

    def test_detect_edges_output_shape(self, clean_frame):
        det = SurfaceDetector()
        edges = det.detect_edges(clean_frame)
        assert edges.shape == clean_frame.shape

    def test_apply_morphology_output_shape(self, clean_frame):
        det = SurfaceDetector()
        edges = det.detect_edges(clean_frame)
        closed = det.apply_morphology(edges)
        assert closed.shape == edges.shape

    def test_find_regions_empty_on_blank_mask(self):
        det = SurfaceDetector()
        blank = np.zeros((400, 640), dtype=np.uint8)
        assert det.find_regions(blank) == []

    def test_classify_region_scratch(self):
        det = SurfaceDetector()
        region = det.classify_region((10, 10, 100, 4))  # long, thin
        assert region.type == "scratch"

    def test_classify_region_dust(self):
        det = SurfaceDetector()
        region = det.classify_region((10, 10, 6, 6))  # compact
        assert region.type == "dust"

    def test_compute_score_no_defects(self):
        assert SurfaceDetector.compute_score([]) == 1.0

    def test_compute_score_decreases_with_defects(self):
        d = DefectRegion(x=0, y=0, w=10, h=10, confidence=1.0, type="dust")
        score = SurfaceDetector.compute_score([d])
        assert 0.0 <= score < 1.0

    def test_invalid_frame_raises(self):
        det = SurfaceDetector()
        with pytest.raises(DetectorError):
            det.detect(np.zeros((10, 10, 3), dtype=np.uint8))  # not grayscale


# ---------------------------------------------------------------------------
# surface.py — end-to-end detection
# ---------------------------------------------------------------------------

class TestSurfaceDetectorEndToEnd:
    def test_clean_frame_passes(self, clean_frame):
        result = SurfaceDetector().detect(clean_frame)
        assert result.passed is True
        assert result.score == 1.0

    def test_scratch_frame_fails(self, scratch_frame):
        result = SurfaceDetector().detect(scratch_frame)
        assert result.passed is False
        assert any(d.type == "scratch" for d in result.defects)

    def test_dust_frame_fails(self, dust_frame):
        result = SurfaceDetector().detect(dust_frame)
        assert result.passed is False
        assert any(d.type == "dust" for d in result.defects)

    def test_duration_ms_recorded(self, clean_frame):
        result = SurfaceDetector().detect(clean_frame)
        assert result.duration_ms > 0.0

    def test_returns_detection_result(self, clean_frame):
        result = SurfaceDetector().detect(clean_frame)
        assert isinstance(result, DetectionResult)


# ---------------------------------------------------------------------------
# Week 3 deliverable gate — accuracy >= 80% on the fixture set
# ---------------------------------------------------------------------------

class TestFixtureAccuracy:
    def test_accuracy_at_least_80_percent(self):
        det = SurfaceDetector()
        paths = sorted(glob.glob(f"{OUTPUT_DIR}/*.jpg"))
        assert len(paths) == 10, f"Expected 10 fixtures, found {len(paths)}"

        correct = 0
        for path in paths:
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            result = det.detect(frame)
            expected_pass = "clean" in os.path.basename(path)
            if result.passed == expected_pass:
                correct += 1

        accuracy = correct / len(paths)
        print(f"\nSurfaceDetector fixture accuracy: {correct}/{len(paths)} = {accuracy:.1%}")
        assert accuracy >= ACCURACY_TARGET, (
            f"Accuracy {accuracy:.1%} below target {ACCURACY_TARGET:.0%}"
        )

    def test_defect_type_matches_filename(self):
        """Fixtures named 'scratch' should classify primarily as scratch, 'dust' as dust."""
        det = SurfaceDetector()
        for path in sorted(glob.glob(f"{OUTPUT_DIR}/surface_scratch_*.jpg")):
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            result = det.detect(frame)
            assert any(d.type == "scratch" for d in result.defects), path

        for path in sorted(glob.glob(f"{OUTPUT_DIR}/surface_dust_*.jpg")):
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            result = det.detect(frame)
            assert any(d.type == "dust" for d in result.defects), path