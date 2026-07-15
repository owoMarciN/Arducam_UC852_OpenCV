"""
color.py — Color / coating anomaly detector via HSV range masking.

Pipeline:
    BGR frame (uint8)
    → convert to HSV
    → mask pixels OUTSIDE the expected coating color range
    → morphological opening (remove single-pixel noise from the mask)
    → contour extraction
    → filter by minimum area
    → each surviving contour becomes a 'color' DefectRegion

Unlike SurfaceDetector and BrightnessDetector, ColorDetector needs the
original color frame, not the grayscale output of Preprocessor. It sets
requires_color = True so Ensemble knows to route the pre-grayscale frame
to it instead.

Usage:
    from detectors.color import ColorDetector

    det = ColorDetector()
    result = det.detect(bgr_frame)
"""

import time

import cv2
import numpy as np

from detectors.base import DefectRegion, Detector, DetectionResult, DetectorError

# Expected coating color range in HSV (OpenCV: H 0-179, S/V 0-255).
# Tune these per product / paint spec.
HUE_RANGE = (90, 130)
SAT_RANGE = (40, 255)
VAL_RANGE = (40, 255)

# Morphological opening kernel — removes isolated noise pixels from the mask
MORPH_KERNEL_SIZE = (3, 3)

# Contours smaller than this (pixels^2) are treated as compression/lighting
# noise, not a real coating defect.
MIN_CONTOUR_AREA = 150

# Contour area at or above this maps to full confidence (1.0)
CONFIDENCE_AREA_CAP = 5000

# Per-defect score penalty, floor 0.0
SCORE_PENALTY_PER_DEFECT = 0.35


class ColorDetector(Detector):
    """Detects patches whose color falls outside the expected coating range."""

    requires_color = True

    def __init__(
        self,
        hue_range: tuple[int, int] = HUE_RANGE,
        sat_range: tuple[int, int] = SAT_RANGE,
        val_range: tuple[int, int] = VAL_RANGE,
        min_area: int = MIN_CONTOUR_AREA,
    ) -> None:
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range
        self.min_area = min_area

    @property
    def name(self) -> str:
        return "color"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run the full color-anomaly pipeline on one BGR frame."""
        _validate_frame(frame)
        t0 = time.perf_counter()

        mask   = self.compute_anomaly_mask(frame)
        opened = self.apply_morphology(mask)
        boxes  = self.find_regions(opened)

        defects = [self.build_region(box) for box in boxes]
        score   = self.compute_score(defects)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(score=score, defects=defects, duration_ms=duration_ms)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def compute_anomaly_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a binary mask of pixels OUTSIDE the expected color range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_range[0], self.sat_range[0], self.val_range[0]], dtype=np.uint8)
        upper = np.array([self.hue_range[1], self.sat_range[1], self.val_range[1]], dtype=np.uint8)
        in_range = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_not(in_range)

    @staticmethod
    def apply_morphology(mask: np.ndarray) -> np.ndarray:
        """Remove isolated noise pixels via morphological opening."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def find_regions(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find bounding boxes of contours above the minimum area threshold."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            boxes.append(cv2.boundingRect(c))
        return boxes

    @staticmethod
    def build_region(box: tuple[int, int, int, int]) -> DefectRegion:
        """Build a 'color' DefectRegion from a bounding box, confidence scaled by area."""
        x, y, w, h = box
        confidence = min(1.0, (w * h) / CONFIDENCE_AREA_CAP)
        return DefectRegion(x=x, y=y, w=w, h=h, confidence=confidence, type="color")

    @staticmethod
    def compute_score(defects: list[DefectRegion]) -> float:
        """Aggregate a 0.0-1.0 color quality score from detected anomalies."""
        if not defects:
            return 1.0
        penalty = sum(SCORE_PENALTY_PER_DEFECT * d.confidence for d in defects)
        return max(0.0, 1.0 - penalty)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _validate_frame(frame: np.ndarray) -> None:
    """Raise DetectorError if frame is not a valid 3-channel BGR uint8 array."""
    if not isinstance(frame, np.ndarray):
        raise DetectorError(f"Expected np.ndarray, got {type(frame)}")
    if frame.dtype != np.uint8:
        raise DetectorError(f"Expected uint8 frame, got dtype {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise DetectorError(f"Expected 3-channel BGR frame, got shape {frame.shape}")
    if frame.size == 0:
        raise DetectorError("Frame is empty (zero size).")
