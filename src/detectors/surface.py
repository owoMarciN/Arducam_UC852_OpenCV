"""
surface.py — Surface defect detector (scratch / dust) via Canny edges
and morphological filtering.

Pipeline:
    grayscale frame (uint8)
    → Canny edge detection
    → morphological closing (connects broken edge fragments)
    → contour extraction
    → filter by minimum area
    → classify each contour as 'scratch' (elongated) or 'dust' (compact)

Usage:
    from base import Detector
    from surface import SurfaceDetector

    det = SurfaceDetector()
    result = det.detect(gray_frame)
    if not result.passed:
        for d in result.defects:
            print(d.type, d.confidence)
"""

import time

import cv2
import numpy as np

from detectors.base import DefectRegion, Detector, DetectionResult, DetectorError

# Canny edge thresholds
CANNY_LOW  = 50
CANNY_HIGH = 150

# Morphological closing kernel — connects fragmented edges into solid blobs
MORPH_KERNEL_SIZE = (3, 3)

# Contours smaller than this (pixels^2) are treated as texture noise, not defects
MIN_CONTOUR_AREA = 20

# Aspect ratio (long side / short side) at or above this is classified as a scratch,
# below this is classified as dust
SCRATCH_ASPECT_RATIO = 2.5

# Contour area at or above this maps to full confidence (1.0)
CONFIDENCE_AREA_CAP = 400

# Per-defect score penalty, floor 0.0
SCORE_PENALTY_PER_DEFECT = 0.25


class SurfaceDetector(Detector):
    """Detects scratches and dust particles on a flat surface."""

    def __init__(
        self,
        canny_low: int = CANNY_LOW,
        canny_high: int = CANNY_HIGH,
        min_area: int = MIN_CONTOUR_AREA,
        scratch_aspect_ratio: float = SCRATCH_ASPECT_RATIO,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.scratch_aspect_ratio = scratch_aspect_ratio

    @property
    def name(self) -> str:
        return "surface"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run the full surface-defect pipeline on one grayscale frame."""
        _validate_frame(frame)
        t0 = time.perf_counter()

        edges  = self.detect_edges(frame)
        closed = self.apply_morphology(edges)
        boxes  = self.find_regions(closed)

        defects = [self.classify_region(box) for box in boxes]
        score   = self.compute_score(defects)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(score=score, defects=defects, duration_ms=duration_ms)

    # ------------------------------------------------------------------
    # Pipeline steps (static — no internal state required)
    # ------------------------------------------------------------------

    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        return cv2.Canny(frame, self.canny_low, self.canny_high)

    @staticmethod
    def apply_morphology(edges: np.ndarray) -> np.ndarray:
        """Close small gaps in edge fragments so contours form solid regions."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    def find_regions(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find bounding boxes of contours above the minimum area threshold."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            boxes.append(cv2.boundingRect(c))  # (x, y, w, h)
        return boxes

    def classify_region(self, box: tuple[int, int, int, int]) -> DefectRegion:
        """Classify a bounding box as 'scratch' (elongated) or 'dust' (compact)."""
        x, y, w, h = box
        long_side, short_side = max(w, h), max(1, min(w, h))
        aspect_ratio = long_side / short_side

        defect_type = "scratch" if aspect_ratio >= self.scratch_aspect_ratio else "dust"
        area = w * h
        confidence = min(1.0, area / CONFIDENCE_AREA_CAP)

        return DefectRegion(x=x, y=y, w=w, h=h, confidence=confidence, type=defect_type)

    @staticmethod
    def compute_score(defects: list[DefectRegion]) -> float:
        """
        Aggregate a 0.0-1.0 surface quality score.
        Each defect subtracts a fixed penalty, weighted by its confidence.
        """
        if not defects:
            return 1.0
        penalty = sum(SCORE_PENALTY_PER_DEFECT * d.confidence for d in defects)
        return max(0.0, 1.0 - penalty)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _validate_frame(frame: np.ndarray) -> None:
    """Raise DetectorError if frame is not a valid single-channel uint8 array."""
    if not isinstance(frame, np.ndarray):
        raise DetectorError(f"Expected np.ndarray, got {type(frame)}")
    if frame.dtype != np.uint8:
        raise DetectorError(f"Expected uint8 frame, got dtype {frame.dtype}")
    if frame.ndim != 2:
        raise DetectorError(f"Expected 2D grayscale frame, got shape {frame.shape}")
    if frame.size == 0:
        raise DetectorError("Frame is empty (zero size).")
