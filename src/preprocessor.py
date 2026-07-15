"""
preprocessor.py — Image preprocessing pipeline for Arducam UC852 frames.

Each step is a pure function operating on numpy arrays so they are
independently testable and composable.

Pipeline (default):
    BGR frame (1280x800 uint8)
    → resize to 640x400
    → convert BGR → grayscale
    → Gaussian blur  (5x5, sigma=0)
    → normalize      (0–255 uint8)

Usage:
    from preprocessor import Preprocessor

    pp = Preprocessor()
    result = pp.run(frame)          # full pipeline
    gray   = pp.to_grayscale(frame) # individual step
"""

import time
import cv2
import numpy as np

# Default pipeline parameters
DEFAULT_RESIZE  = (640, 400)       # (width, height)
BLUR_KERNEL     = (5, 5)
BLUR_SIGMA      = 0                # let OpenCV derive sigma from kernel size
NORM_ALPHA      = 0
NORM_BETA       = 255
NORM_TYPE       = cv2.NORM_MINMAX
NORM_DTYPE      = cv2.CV_8U


class PreprocessError(ValueError):
    """Raised when a frame fails a precondition check."""


class Preprocessor:
    """
    Stateless image preprocessing pipeline.

    Parameters
    ----------
    resize_to : (width, height) target after resize, or None to skip
    blur      : True to apply Gaussian blur
    normalize : True to apply min-max normalization
    """

    def __init__(
        self,
        resize_to: tuple[int, int] | None = DEFAULT_RESIZE,
        blur: bool      = True,
        normalize: bool = True,
    ) -> None:
        self.resize_to = resize_to
        self.blur      = blur
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Execute the full preprocessing pipeline.

        Returns
        -------
        result   : processed grayscale frame (uint8)
        elapsed  : wall-clock time in milliseconds
        """
        _validate_frame(frame)
        t0 = time.perf_counter()

        img = frame

        if self.resize_to is not None:
            img = self.resize(img, self.resize_to)

        img = self.to_grayscale(img)

        if self.blur:
            img = self.gaussian_blur(img)

        if self.normalize:
            img = self.normalize_frame(img)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return img, elapsed_ms

    # ------------------------------------------------------------------
    # Individual steps (static — no internal state required)
    # ------------------------------------------------------------------

    @staticmethod
    def resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Resize frame to (width, height).
        Uses INTER_AREA for downscaling (best quality, fastest for this ratio).
        """
        _validate_frame(frame)
        w, h = size
        if w <= 0 or h <= 0:
            raise PreprocessError(f"resize size must be positive, got {size}")
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert BGR or grayscale frame to single-channel grayscale."""
        _validate_frame(frame)
        if frame.ndim == 2:
            return frame  # already grayscale
        if frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise PreprocessError(
            f"Expected 1 or 3 channel frame, got shape {frame.shape}"
        )

    @staticmethod
    def gaussian_blur(frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur with a 5x5 kernel."""
        _validate_frame(frame)
        return cv2.GaussianBlur(frame, BLUR_KERNEL, BLUR_SIGMA)

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Min-max normalize pixel values to 0–255 uint8."""
        _validate_frame(frame)
        out = np.empty_like(frame)
        cv2.normalize(frame, out, NORM_ALPHA, NORM_BETA, NORM_TYPE, NORM_DTYPE)
        return out


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _validate_frame(frame: np.ndarray) -> None:
    """Raise PreprocessError if frame is not a valid uint8 numpy array."""
    if not isinstance(frame, np.ndarray):
        raise PreprocessError(f"Expected np.ndarray, got {type(frame)}")
    if frame.dtype != np.uint8:
        raise PreprocessError(f"Expected uint8 frame, got dtype {frame.dtype}")
    if frame.size == 0:
        raise PreprocessError("Frame is empty (zero size).")
