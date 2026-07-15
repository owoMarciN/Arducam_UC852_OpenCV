"""
test_ensemble.py — Unit tests for ensemble.py.

Covers EnsembleConfig validation, weighted-score aggregation using fake
detectors (deterministic, no image thresholds involved), and one
integration test combining the real SurfaceDetector, ColorDetector, and
BrightnessDetector end to end — the Week 4 "integration test with
ensemble" deliverable.

Run with:
    pytest test_ensemble.py -v -s
"""

import json

import cv2
import numpy as np
import pytest

from detectors.base import DefectRegion, Detector, DetectionResult, DetectorError
from detectors.ensemble import Ensemble, EnsembleConfig, FinalDecision
from detectors.surface import SurfaceDetector
from detectors.color import ColorDetector
from detectors.brightness import BrightnessDetector
from generate_color_brightness_fixtures import make_normal_color_panel, add_color_anomaly


# ---------------------------------------------------------------------------
# Fake detector — deterministic test double, no real image processing
# ---------------------------------------------------------------------------

class FakeDetector(Detector):
    """Returns a fixed score/defects and records the frame it received."""

    def __init__(self, name, score, defects=None, requires_color=False):
        self._name = name
        self._score = score
        self._defects = defects or []
        self.requires_color = requires_color
        self.received_frame = None

    @property
    def name(self) -> str:
        return self._name

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self.received_frame = frame
        return DetectionResult(score=self._score, defects=list(self._defects))


# ---------------------------------------------------------------------------
# EnsembleConfig validation
# ---------------------------------------------------------------------------

class TestEnsembleConfig:
    def test_valid_config(self):
        config = EnsembleConfig(weights={"surface": 0.5, "color": 0.5})
        assert config.pass_threshold == 0.7

    def test_empty_weights_raises(self):
        with pytest.raises(DetectorError):
            EnsembleConfig(weights={})

    def test_negative_weight_raises(self):
        with pytest.raises(DetectorError):
            EnsembleConfig(weights={"surface": -0.1})

    def test_zero_sum_weights_raises(self):
        with pytest.raises(DetectorError):
            EnsembleConfig(weights={"surface": 0.0, "color": 0.0})

    def test_invalid_pass_threshold_raises(self):
        with pytest.raises(DetectorError):
            EnsembleConfig(weights={"surface": 1.0}, pass_threshold=1.5)


# ---------------------------------------------------------------------------
# Ensemble.add
# ---------------------------------------------------------------------------

class TestEnsembleAdd:
    def test_add_registers_known_detector(self):
        config = EnsembleConfig(weights={"fake": 1.0})
        ens = Ensemble(config)
        ens.add(FakeDetector("fake", score=1.0))
        assert len(ens._detectors) == 1

    def test_add_raises_for_unweighted_detector(self):
        config = EnsembleConfig(weights={"surface": 1.0})
        ens = Ensemble(config)
        with pytest.raises(DetectorError):
            ens.add(FakeDetector("color", score=1.0))


# ---------------------------------------------------------------------------
# Ensemble.run — weighted math, using fakes for deterministic results
# ---------------------------------------------------------------------------

class TestEnsembleRunMath:
    def test_no_detectors_raises(self):
        config = EnsembleConfig(weights={"surface": 1.0})
        ens = Ensemble(config)
        with pytest.raises(DetectorError):
            ens.run(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_weighted_average_matches_expected(self):
        config = EnsembleConfig(weights={"a": 0.5, "b": 0.3, "c": 0.2}, pass_threshold=0.7)
        ens = Ensemble(config)
        ens.add(FakeDetector("a", score=1.0))
        ens.add(FakeDetector("b", score=0.5))
        ens.add(FakeDetector("c", score=0.0))

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        decision = ens.run(frame)

        expected = 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.0  # 0.65
        assert decision.confidence == pytest.approx(expected)
        assert decision.passed is False  # 0.65 < 0.7

    def test_all_scores_perfect_passes(self):
        config = EnsembleConfig(weights={"a": 0.6, "b": 0.4})
        ens = Ensemble(config)
        ens.add(FakeDetector("a", score=1.0))
        ens.add(FakeDetector("b", score=1.0))

        decision = ens.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert decision.confidence == 1.0
        assert decision.passed is True

    def test_all_defects_aggregated(self):
        d1 = DefectRegion(x=0, y=0, w=5, h=5, confidence=0.9, type="scratch")
        d2 = DefectRegion(x=1, y=1, w=6, h=6, confidence=0.8, type="color")

        config = EnsembleConfig(weights={"a": 0.5, "b": 0.5})
        ens = Ensemble(config)
        ens.add(FakeDetector("a", score=0.6, defects=[d1]))
        ens.add(FakeDetector("b", score=0.6, defects=[d2]))

        decision = ens.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(decision.all_defects) == 2
        assert {d.type for d in decision.all_defects} == {"scratch", "color"}

    def test_requires_color_detector_receives_original_frame(self):
        color_fake = FakeDetector("color", score=1.0, requires_color=True)
        config = EnsembleConfig(weights={"color": 1.0})
        ens = Ensemble(config)
        ens.add(color_fake)

        bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        ens.run(bgr)
        assert color_fake.received_frame.ndim == 3
        assert color_fake.received_frame.shape == bgr.shape

    def test_gray_only_detector_receives_grayscale_frame(self):
        surface_fake = FakeDetector("surface", score=1.0, requires_color=False)
        config = EnsembleConfig(weights={"surface": 1.0})
        ens = Ensemble(config)
        ens.add(surface_fake)

        bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        ens.run(bgr)
        assert surface_fake.received_frame.ndim == 2
        assert surface_fake.received_frame.shape == (20, 30)

    def test_already_grayscale_input_is_passed_through(self):
        surface_fake = FakeDetector("surface", score=1.0, requires_color=False)
        config = EnsembleConfig(weights={"surface": 1.0})
        ens = Ensemble(config)
        ens.add(surface_fake)

        gray = np.zeros((20, 30), dtype=np.uint8)
        ens.run(gray)
        assert surface_fake.received_frame.ndim == 2


# ---------------------------------------------------------------------------
# FinalDecision.result_json
# ---------------------------------------------------------------------------

class TestFinalDecisionJson:
    def test_result_json_is_valid_and_has_expected_keys(self):
        d = DefectRegion(x=1, y=2, w=3, h=4, confidence=0.5, type="dust")
        decision = FinalDecision(passed=False, confidence=0.42, all_defects=[d])
        parsed = json.loads(decision.result_json())

        assert parsed["pass"] is False
        assert parsed["confidence"] == 0.42
        assert len(parsed["defects"]) == 1
        assert parsed["defects"][0]["type"] == "dust"


# ---------------------------------------------------------------------------
# Integration test — real detectors combined via Ensemble (Week 4 deliverable)
# ---------------------------------------------------------------------------

class TestEnsembleIntegrationRealDetectors:
    @staticmethod
    def _build_ensemble():
        config = EnsembleConfig(weights={"surface": 0.5, "color": 0.3, "brightness": 0.2})
        ens = Ensemble(config)
        ens.add(SurfaceDetector())
        ens.add(ColorDetector())
        ens.add(BrightnessDetector())
        return ens

    def test_clean_frame_passes(self):
        rng = np.random.default_rng(1)
        frame = make_normal_color_panel(rng)
        decision = self._build_ensemble().run(frame)
        assert decision.passed is True
        assert decision.confidence >= 0.7

    def test_heavily_defective_frame_fails(self):
        """Scratch + color anomaly + overexposure combined should fail confidently."""
        rng = np.random.default_rng(5)
        frame = make_normal_color_panel(rng)
        frame = add_color_anomaly(frame, rng)
        cv2.line(frame, (40, 180), (600, 220), color=(0, 0, 0), thickness=3)
        frame = np.clip(frame.astype(np.int16) + 90, 0, 255).astype(np.uint8)

        decision = self._build_ensemble().run(frame)
        assert decision.passed is False
        assert decision.confidence < 0.6
        assert len(decision.all_defects) > 0

    def test_result_json_serializable_for_real_decision(self):
        rng = np.random.default_rng(1)
        frame = make_normal_color_panel(rng)
        decision = self._build_ensemble().run(frame)
        parsed = json.loads(decision.result_json())
        assert "pass" in parsed and "confidence" in parsed and "defects" in parsed
