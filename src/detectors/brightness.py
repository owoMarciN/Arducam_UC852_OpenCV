"""
brightness.py — Over/under exposure detector via global intensity statistics.

Pipeline:
    grayscale frame (uint8)
    → compute mean intensity and saturated-pixel fractions
    → classify as overexposed / underexposed / normal
    → if abnormal, emit one full-frame DefectRegion (type='brightness')

Brightness is a global property of the frame, not a localized defect, so
unlike SurfaceDetector and ColorDetector there is at most one DefectRegion,
covering the entire frame.

Usage:
    from detectors.brightness import BrightnessDetector

    det = BrightnessDetector()
    result = det.detect(gray_frame)
"""

import time

import numpy as np

from detectors.base import DefectRegion, Detector, DetectionResult, DetectorError

# Mean intensity thresholds (0-255 grayscale)
MEAN_LOW  = 60    # mean at or below this -> underexposed
MEAN_HIGH = 200   # mean at or above this -> overexposed

# Fraction of pixels at the extreme ends that independently trigger a flag,
# even if the mean is within range (catches blown highlights / crushed blacks
# concentrated in part of the frame).
SATURATED_HIGH_FRACTION = 0.05  # >=5% of pixels near 255
SATURATED_LOW_FRACTION  = 0.05  # >=5% of pixels near 0


class BrightnessDetector(Detector):
    """Detects globally overexposed or underexposed frames."""

    requires_color = False

    def __init__(
        self,
        mean_low: float = MEAN_LOW,
        mean_high: float = MEAN_HIGH,
        saturated_high_fraction: float = SATURATED_HIGH_FRACTION,
        saturated_low_fraction: float = SATURATED_LOW_FRACTION,
    ) -> None:
        self.mean_low = mean_low
        self.mean_high = mean_high
        self.saturated_high_fraction = saturated_high_fraction
        self.saturated_low_fraction = saturated_low_fraction

    @property
    def name(self) -> str:
        return "brightness"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Classify frame exposure and return a DetectionResult."""
        _validate_frame(frame)
        t0 = time.perf_counter()

        mean_val  = self.compute_mean(frame)
        high_frac = self.compute_saturated_fraction(frame, high=True)
        low_frac  = self.compute_saturated_fraction(frame, high=False)

        condition = self.classify(mean_val, high_frac, low_frac)

        defects = []
        if condition is not None:
            h, w = frame.shape
            confidence = self.compute_confidence(mean_val, condition)
            defects.append(DefectRegion(x=0, y=0, w=w, h=h, confidence=confidence, type="brightness"))

        score = self.compute_score(defects)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(score=score, defects=defects, duration_ms=duration_ms)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mean(frame: np.ndarray) -> float:
        return float(np.mean(frame))

    @staticmethod
    def compute_saturated_fraction(frame: np.ndarray, high: bool) -> float:
        """Fraction of pixels within 1 gray level of the extreme (255 or 0)."""
        if high:
            return float(np.mean(frame >= 254))
        return float(np.mean(frame <= 1))

    def classify(self, mean_val: float, high_frac: float, low_frac: float) -> str | None:
        """Return 'overexposed', 'underexposed', or None."""
        if mean_val >= self.mean_high or high_frac >= self.saturated_high_fraction:
            return "overexposed"
        if mean_val <= self.mean_low or low_frac >= self.saturated_low_fraction:
            return "underexposed"
        return None

    def compute_confidence(self, mean_val: float, condition: str) -> float:
        """Scale confidence by how far the mean sits past its threshold."""
        if condition == "overexposed":
            span = max(1.0, 255 - self.mean_high)
            return min(1.0, max(0.0, (mean_val - self.mean_high) / span))
        span = max(1.0, self.mean_low)
        return min(1.0, max(0.0, (self.mean_low - mean_val) / span))

    @staticmethod
    def compute_score(defects: list[DefectRegion]) -> float:
        """Aggregate a 0.0-1.0 exposure quality score."""
        if not defects:
            return 1.0
        return max(0.0, 1.0 - defects[0].confidence)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _validate_frame(frame: np.ndarray) -> None:
    """Raise DetectorError if frame is not a valid 2D grayscale uint8 array."""
    if not isinstance(frame, np.ndarray):
        raise DetectorError(f"Expected np.ndarray, got {type(frame)}")
    if frame.dtype != np.uint8:
        raise DetectorError(f"Expected uint8 frame, got dtype {frame.dtype}")
    if frame.ndim != 2:
        raise DetectorError(f"Expected 2D grayscale frame, got shape {frame.shape}")
    if frame.size == 0:
        raise DetectorError("Frame is empty (zero size).")
