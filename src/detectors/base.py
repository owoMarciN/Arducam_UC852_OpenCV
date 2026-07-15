"""
base.py — Abstract detector interface and shared data structures.

Defines the contract every concrete detector (surface, color, brightness)
must implement. Mirrors the C++ Detector / DetectionResult interface from
the technical reference so the Week 5 C++ port is a direct translation.

Usage:
    from base import Detector, DetectionResult, DefectRegion

    class MyDetector(Detector):
        def detect(self, frame: np.ndarray) -> DetectionResult:
            ...
        @property
        def name(self) -> str:
            return "my_detector"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

VALID_DEFECT_TYPES = ("scratch", "dust", "color", "brightness")


class DetectorError(ValueError):
    """Raised when a detector receives an invalid frame or configuration."""


@dataclass
class DefectRegion:
    """A single detected defect, bounded by an axis-aligned box."""

    x: int
    y: int
    w: int
    h: int
    confidence: float  # 0.0 - 1.0
    type: str          # one of VALID_DEFECT_TYPES

    def __post_init__(self) -> None:
        if self.type not in VALID_DEFECT_TYPES:
            raise DetectorError(
                f"Invalid defect type '{self.type}', expected one of {VALID_DEFECT_TYPES}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise DetectorError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.w <= 0 or self.h <= 0:
            raise DetectorError(f"width/height must be positive, got w={self.w}, h={self.h}")

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class DetectionResult:
    """Output of a single detector run on a single frame."""

    score: float                                   # 0.0 = bad, 1.0 = perfect
    defects: list[DefectRegion] = field(default_factory=list)
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise DetectorError(f"score must be in [0.0, 1.0], got {self.score}")

    @property
    def passed(self) -> bool:
        """True when no defects were found."""
        return len(self.defects) == 0


class Detector(ABC):
    """Abstract base class every concrete detector must implement."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame and return a DetectionResult."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in ensemble weighting and logging."""
        raise NotImplementedError
