"""
ensemble.py — Combines multiple detector scores into a single PASS/FAIL decision.

Mirrors cpp/include/detectors/ensemble.hpp: EnsembleConfig (weights + pass
threshold) and FinalDecision (pass, confidence, all defects, JSON export).

Detectors that need the original color frame (ColorDetector) set
`requires_color = True`; Ensemble routes the raw frame to those and the
grayscale conversion (via Preprocessor.to_grayscale) to everything else,
so each detector always receives the representation it expects.

Usage:
    from detectors.ensemble import Ensemble, EnsembleConfig
    from detectors.surface import SurfaceDetector
    from detectors.color import ColorDetector
    from detectors.brightness import BrightnessDetector

    config = EnsembleConfig(weights={"surface": 0.5, "color": 0.3, "brightness": 0.2})
    ensemble = Ensemble(config)
    ensemble.add(SurfaceDetector())
    ensemble.add(ColorDetector())
    ensemble.add(BrightnessDetector())

    decision = ensemble.run(bgr_frame)
    if not decision.passed:
        print(decision.result_json())
"""

import json
from dataclasses import dataclass, field

import numpy as np

from detectors.base import DefectRegion, Detector, DetectorError
from preprocessor import Preprocessor

DEFAULT_PASS_THRESHOLD = 0.7


@dataclass
class EnsembleConfig:
    """Per-detector weights and the score threshold for a PASS decision."""

    weights: dict[str, float]
    pass_threshold: float = DEFAULT_PASS_THRESHOLD

    def __post_init__(self) -> None:
        if not self.weights:
            raise DetectorError("EnsembleConfig.weights must not be empty")
        if any(w < 0 for w in self.weights.values()):
            raise DetectorError("EnsembleConfig.weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise DetectorError("EnsembleConfig.weights must sum to a positive value")
        if not (0.0 <= self.pass_threshold <= 1.0):
            raise DetectorError(f"pass_threshold must be in [0.0, 1.0], got {self.pass_threshold}")


@dataclass
class FinalDecision:
    """Aggregated outcome of running every registered detector on one frame."""

    passed: bool
    confidence: float
    all_defects: list[DefectRegion] = field(default_factory=list)

    def result_json(self) -> str:
        """Serialize to JSON for the Python/C++ or API bridge."""
        return json.dumps({
            "pass": self.passed,
            "confidence": round(self.confidence, 4),
            "defects": [
                {
                    "x": d.x, "y": d.y, "w": d.w, "h": d.h,
                    "confidence": round(d.confidence, 4),
                    "type": d.type,
                }
                for d in self.all_defects
            ],
        })


class Ensemble:
    """Registers detectors and combines their scores into one decision."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self._detectors: list[Detector] = []

    def add(self, detector: Detector) -> None:
        """Register a detector. Its name must have a configured weight."""
        if detector.name not in self.config.weights:
            raise DetectorError(
                f"No weight configured for detector '{detector.name}'. "
                f"Known weights: {list(self.config.weights)}"
            )
        self._detectors.append(detector)

    def run(self, frame: np.ndarray) -> FinalDecision:
        """
        Run every registered detector on the frame and combine their scores.

        `frame` should be the original captured frame (BGR, 3-channel).
        Grayscale-only detectors receive an internally derived grayscale
        copy; detectors with requires_color=True receive the original.
        """
        if not self._detectors:
            raise DetectorError("Ensemble has no detectors registered")

        gray_frame = frame if frame.ndim == 2 else Preprocessor.to_grayscale(frame)

        weighted_sum = 0.0
        weight_total = 0.0
        all_defects: list[DefectRegion] = []

        for detector in self._detectors:
            detector_frame = frame if getattr(detector, "requires_color", False) else gray_frame
            result = detector.detect(detector_frame)

            weight = self.config.weights[detector.name]
            weighted_sum += result.score * weight
            weight_total += weight
            all_defects.extend(result.defects)

        confidence = weighted_sum / weight_total if weight_total > 0 else 0.0
        passed = confidence >= self.config.pass_threshold

        return FinalDecision(passed=passed, confidence=confidence, all_defects=all_defects)
