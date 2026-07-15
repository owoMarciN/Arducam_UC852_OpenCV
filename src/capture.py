"""
capture.py — Arducam UC852 (OV9782) camera abstraction.

Provides a CameraCapture class that manages device lifecycle,
format negotiation, and single-frame reads.

Usage:
    from capture import CameraCapture

    with CameraCapture() as cam:
        frame = cam.read_frame()
"""

import cv2
import numpy as np

CAMERA_INDEX = 2
TARGET_WIDTH  = 1280
TARGET_HEIGHT = 800
FOURCC        = "MJPG"


class CaptureError(RuntimeError):
    """Raised when the camera cannot be opened or a frame read fails."""


class CameraCapture:
    """
    Context-manager wrapper around cv2.VideoCapture for the Arducam UC852.

    Parameters
    ----------
    index  : V4L2 device index (default 2 → /dev/video2)
    width  : requested frame width  (default 1280)
    height : requested frame height (default 800)
    fourcc : pixel format negotiated with the driver (default 'MJPG')
    """

    def __init__(
        self,
        index: int  = CAMERA_INDEX,
        width: int  = TARGET_WIDTH,
        height: int = TARGET_HEIGHT,
        fourcc: str = FOURCC,
    ) -> None:
        self.index  = index
        self.width  = width
        self.height = height
        self.fourcc = fourcc
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def open(self) -> "CameraCapture":
        """Open the device and negotiate format + resolution."""
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CaptureError(
                f"Cannot open camera at index {self.index}. "
                "Check that /dev/video2 exists and the Arducam UC852 is connected."
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        return self

    def close(self) -> None:
        """Release the device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_width(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self) -> np.ndarray:
        """
        Capture and return one BGR frame as a numpy uint8 array.

        Raises CaptureError if the read fails.
        """
        self._require_open()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CaptureError("camera.read() failed — no frame returned.")
        return frame

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self.is_open:
            raise CaptureError("Camera is not open. Call open() or use as context manager.")
