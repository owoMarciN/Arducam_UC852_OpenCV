"""
test_capture_preprocessor.py — Unit tests for capture.py and preprocessor.py.

Tests run without physical hardware by mocking cv2.VideoCapture.
The timing test (< 10ms) runs on a synthetic 1280x800 frame using
the real OpenCV pipeline — no mocking.

Run with:
    pytest test_capture_preprocessor.py -v
"""

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from capture import CameraCapture, CaptureError
from preprocessor import Preprocessor, PreprocessError, _validate_frame

# ---------------------------------------------------------------------------
# Shared synthetic frames
# ---------------------------------------------------------------------------

@pytest.fixture
def bgr_frame():
    """1280x800 BGR uint8 frame with real content (not blank, not saturated)."""
    rng = np.random.default_rng(42)
    return rng.integers(10, 245, (800, 1280, 3), dtype=np.uint8)


@pytest.fixture
def gray_frame():
    """640x400 grayscale uint8 frame."""
    rng = np.random.default_rng(7)
    return rng.integers(10, 245, (400, 640), dtype=np.uint8)


@pytest.fixture
def mock_cap(bgr_frame):
    """cv2.VideoCapture mock that returns a valid frame on read()."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, bgr_frame)
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH:  1280.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 800.0,
    }.get(prop, 0.0)
    return cap


# ---------------------------------------------------------------------------
# CameraCapture tests
# ---------------------------------------------------------------------------

class TestCameraCaptureOpen:
    def test_open_succeeds(self, mock_cap):
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            cam = CameraCapture()
            cam.open()
            assert cam.is_open
            cam.close()

    def test_open_raises_when_device_missing(self):
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False
        with patch("capture.cv2.VideoCapture", return_value=bad_cap):
            with pytest.raises(CaptureError, match="Cannot open camera"):
                CameraCapture().open()

    def test_context_manager_closes_on_exit(self, mock_cap):
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            with CameraCapture() as cam:
                assert cam.is_open
        mock_cap.release.assert_called_once()

    def test_context_manager_closes_on_exception(self, mock_cap):
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(ValueError):
                with CameraCapture() as cam:
                    raise ValueError("deliberate")
        mock_cap.release.assert_called_once()


class TestCameraCaptureProperties:
    def test_actual_width(self, mock_cap):
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            with CameraCapture() as cam:
                assert cam.actual_width == 1280

    def test_actual_height(self, mock_cap):
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            with CameraCapture() as cam:
                assert cam.actual_height == 800

    def test_property_raises_when_closed(self):
        cam = CameraCapture()
        with pytest.raises(CaptureError, match="not open"):
            _ = cam.actual_width


class TestCameraCaptureReadFrame:
    def test_read_frame_returns_ndarray(self, mock_cap, bgr_frame):
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            with CameraCapture() as cam:
                frame = cam.read_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == bgr_frame.shape

    def test_read_frame_raises_on_failure(self, mock_cap):
        mock_cap.read.return_value = (False, None)
        with patch("capture.cv2.VideoCapture", return_value=mock_cap):
            with CameraCapture() as cam:
                with pytest.raises(CaptureError, match="read\\(\\) failed"):
                    cam.read_frame()

    def test_read_frame_raises_when_closed(self):
        cam = CameraCapture()
        with pytest.raises(CaptureError, match="not open"):
            cam.read_frame()


# ---------------------------------------------------------------------------
# Preprocessor — individual steps
# ---------------------------------------------------------------------------

class TestResize:
    def test_output_shape(self, bgr_frame):
        out = Preprocessor.resize(bgr_frame, (640, 400))
        assert out.shape == (400, 640, 3)

    def test_grayscale_resize(self, gray_frame):
        out = Preprocessor.resize(gray_frame, (320, 200))
        assert out.shape == (200, 320)

    def test_invalid_size_raises(self, bgr_frame):
        with pytest.raises(PreprocessError):
            Preprocessor.resize(bgr_frame, (0, 400))

    def test_dtype_preserved(self, bgr_frame):
        out = Preprocessor.resize(bgr_frame, (640, 400))
        assert out.dtype == np.uint8


class TestToGrayscale:
    def test_bgr_to_gray(self, bgr_frame):
        out = Preprocessor.to_grayscale(bgr_frame)
        assert out.ndim == 2

    def test_already_gray_passthrough(self, gray_frame):
        out = Preprocessor.to_grayscale(gray_frame)
        assert out.ndim == 2
        assert out is gray_frame  # same object, no copy

    def test_invalid_channels_raises(self):
        bad = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(PreprocessError):
            Preprocessor.to_grayscale(bad)


class TestGaussianBlur:
    def test_output_shape_unchanged(self, gray_frame):
        out = Preprocessor.gaussian_blur(gray_frame)
        assert out.shape == gray_frame.shape

    def test_output_dtype(self, gray_frame):
        out = Preprocessor.gaussian_blur(gray_frame)
        assert out.dtype == np.uint8

    def test_blur_smooths_noise(self):
        noisy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blurred = Preprocessor.gaussian_blur(noisy)
        assert float(np.std(blurred)) < float(np.std(noisy))


class TestNormalizeFrame:
    def test_output_range(self, gray_frame):
        out = Preprocessor.normalize_frame(gray_frame)
        assert int(out.min()) == 0
        assert int(out.max()) == 255

    def test_uniform_frame_stays_uint8(self):
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        out = Preprocessor.normalize_frame(uniform)
        assert out.dtype == np.uint8

    def test_output_dtype(self, gray_frame):
        out = Preprocessor.normalize_frame(gray_frame)
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# Preprocessor — full pipeline
# ---------------------------------------------------------------------------

class TestPreprocessorPipeline:
    def test_output_is_2d(self, bgr_frame):
        pp = Preprocessor()
        result, _ = pp.run(bgr_frame)
        assert result.ndim == 2

    def test_output_shape(self, bgr_frame):
        pp = Preprocessor(resize_to=(640, 400))
        result, _ = pp.run(bgr_frame)
        assert result.shape == (400, 640)

    def test_output_dtype(self, bgr_frame):
        pp = Preprocessor()
        result, _ = pp.run(bgr_frame)
        assert result.dtype == np.uint8

    def test_elapsed_is_float(self, bgr_frame):
        pp = Preprocessor()
        _, elapsed = pp.run(bgr_frame)
        assert isinstance(elapsed, float)

    def test_no_resize_option(self, bgr_frame):
        pp = Preprocessor(resize_to=None, blur=False, normalize=False)
        result, _ = pp.run(bgr_frame)
        assert result.shape == (800, 1280)

    def test_invalid_input_raises(self):
        pp = Preprocessor()
        with pytest.raises(PreprocessError):
            pp.run(np.zeros((100, 100, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Timing — target < 10ms on a 1280x800 frame
# ---------------------------------------------------------------------------

class TestPreprocessorTiming:
    RUNS        = 50
    TARGET_MS   = 10.0
    WARMUP_RUNS = 5

    def test_mean_time_under_10ms(self, bgr_frame):
        pp = Preprocessor()

        # warmup — let OpenCV and the OS settle
        for _ in range(self.WARMUP_RUNS):
            pp.run(bgr_frame)

        times = []
        for _ in range(self.RUNS):
            _, elapsed = pp.run(bgr_frame)
            times.append(elapsed)

        mean_ms = float(np.mean(times))
        p95_ms  = float(np.percentile(times, 95))

        print(
            f"\nPreprocessor timing over {self.RUNS} runs:"
            f"\n  mean : {mean_ms:.3f} ms"
            f"\n  p95  : {p95_ms:.3f} ms"
            f"\n  min  : {min(times):.3f} ms"
            f"\n  max  : {max(times):.3f} ms"
        )

        assert mean_ms < self.TARGET_MS, (
            f"Mean preprocess time {mean_ms:.3f} ms exceeds target {self.TARGET_MS} ms"
        )

    def test_p95_under_20ms(self, bgr_frame):
        """p95 guard — occasional spikes should still be reasonable."""
        pp = Preprocessor()
        times = [pp.run(bgr_frame)[1] for _ in range(self.RUNS)]
        p95_ms = float(np.percentile(times, 95))
        assert p95_ms < 20.0, (
            f"p95 preprocess time {p95_ms:.3f} ms exceeds 20 ms guard"
        )


# ---------------------------------------------------------------------------
# _validate_frame helper
# ---------------------------------------------------------------------------

class TestValidateFrame:
    def test_valid_frame_passes(self, bgr_frame):
        _validate_frame(bgr_frame)  # no exception

    def test_non_ndarray_raises(self):
        with pytest.raises(PreprocessError):
            _validate_frame([[1, 2], [3, 4]])

    def test_wrong_dtype_raises(self):
        with pytest.raises(PreprocessError):
            _validate_frame(np.zeros((100, 100, 3), dtype=np.float32))

    def test_empty_array_raises(self):
        with pytest.raises(PreprocessError):
            _validate_frame(np.zeros((0, 0, 3), dtype=np.uint8))
