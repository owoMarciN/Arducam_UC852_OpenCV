"""
test_camera.py — Arducam UC852 (OV9782) validation suite
Run with: 
    pytest test_camera.py -v
Run only 20 fixtures:
    python3 capture_fixtures.py
"""

import cv2
import numpy as np
import pytest

CAMERA_INDEX = 2
TARGET_WIDTH = 1280
TARGET_HEIGHT = 800
SAMPLE_COUNT = 20
OUTPUT_DIR = "../fixture_images"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def camera():
    """Open the camera once for the entire module, release after all tests."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    assert cap.isOpened(), (
        f"Cannot open camera at index {CAMERA_INDEX}. "
        "Check that the Arducam UC852 is connected and /dev/video0 is accessible."
    )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    yield cap

    cap.release()


@pytest.fixture(scope="module")
def single_frame(camera):
    """Capture one frame and return it for reuse across tests."""
    ret, frame = camera.read()
    assert ret, "camera.read() returned False — no frame captured."
    assert frame is not None, "Captured frame is None."
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCameraConnection:
    def test_camera_opens(self, camera):
        assert camera.isOpened()

    def test_resolution_width(self, camera):
        actual = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        assert actual == TARGET_WIDTH, (
            f"Expected width {TARGET_WIDTH}, got {actual}"
        )

    def test_resolution_height(self, camera):
        actual = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert actual == TARGET_HEIGHT, (
            f"Expected height {TARGET_HEIGHT}, got {actual}"
        )


class TestFrameProperties:
    def test_frame_captured(self, single_frame):
        assert single_frame is not None

    def test_frame_shape(self, single_frame):
        h, w, c = single_frame.shape
        assert w == TARGET_WIDTH, f"Frame width {w} != {TARGET_WIDTH}"
        assert h == TARGET_HEIGHT, f"Frame height {h} != {TARGET_HEIGHT}"

    def test_frame_is_color(self, single_frame):
        assert single_frame.ndim == 3, "Expected 3-channel (color) frame"
        assert single_frame.shape[2] == 3, "Expected BGR — 3 channels"

    def test_frame_dtype(self, single_frame):
        assert single_frame.dtype == np.uint8

    def test_frame_not_blank(self, single_frame):
        """Fail if the frame is completely black (sensor not sending data)."""
        assert single_frame.max() > 0, "Frame is entirely black"

    def test_frame_not_saturated(self, single_frame):
        """Fail if every pixel is at maximum (sensor overexposed / stuck)."""
        assert single_frame.min() < 255, "Frame is entirely white / saturated"

    def test_frame_has_variance(self, single_frame):
        """Ensure there is real content — not a flat uniform color."""
        std = float(np.std(single_frame))
        assert std > 1.0, f"Frame has no variance (std={std:.4f}) — may be a solid color"


class TestSampleCapture:
    def test_capture_20_samples(self, camera, tmp_path):
        """
        Capture 20 frames and save them as JPEG fixtures.
        Images land in ./fixture_images/ relative to the working directory,
        and also in pytest's tmp_path for assertion purposes.
        """
        import os

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        saved = []

        for i in range(SAMPLE_COUNT):
            ret, frame = camera.read()
            assert ret, f"Frame {i} read failed"
            assert frame is not None, f"Frame {i} is None"

            filename = f"fixture_{i:02d}.jpg"
            local_path = os.path.join(OUTPUT_DIR, filename)
            tmp_file = str(tmp_path / filename)

            ok_local = cv2.imwrite(local_path, frame)
            ok_tmp = cv2.imwrite(tmp_file, frame)

            assert ok_local, f"Failed to write {local_path}"
            assert ok_tmp, f"Failed to write {tmp_file}"

            saved.append(local_path)

        assert len(saved) == SAMPLE_COUNT, (
            f"Expected {SAMPLE_COUNT} images, saved {len(saved)}"
        )
        print(f"\n{SAMPLE_COUNT} fixture images saved to ./{OUTPUT_DIR}/")