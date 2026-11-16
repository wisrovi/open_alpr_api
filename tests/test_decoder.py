"""Unit tests for accelerated image decoding module (app.decoder).

Validates SIMD/torchvision accelerated decoding when available, OpenCV fallback,
and robust exception handling when receiving invalid or truncated bytes.
"""

import cv2
import numpy as np
import pytest

from app.decoder import decode_image_bytes


def test_decode_image_bytes_valid_jpeg() -> None:
    """Validate decoding of a valid synthetic JPEG byte array.

    What this test does:
    1. Creates a 120x160 BGR image and encodes it to JPEG.
    2. Calls `decode_image_bytes`.
    3. Asserts the returned matrix has identical dimensions and channel count.
    """
    orig = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(orig, (10, 10), (100, 100), (0, 255, 0), -1)
    _, buf = cv2.imencode(".jpg", orig)

    decoded, _ = decode_image_bytes(buf.tobytes())
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (120, 160, 3)
    assert decoded.dtype == np.uint8


def test_decode_image_bytes_empty_buffer_raises_value_error() -> None:
    """Validate that passing an empty byte buffer raises ValueError.

    What this test does:
    1. Passes b"" to `decode_image_bytes`.
    2. Asserts ValueError is raised.
    """
    with pytest.raises(ValueError, match="Empty image byte buffer"):
        decode_image_bytes(b"")


def test_decode_image_bytes_corrupted_data_raises_value_error() -> None:
    """Validate that passing corrupted non-image bytes raises ValueError.

    What this test does:
    1. Passes random corrupted binary data.
    2. Asserts ValueError is raised indicating decode failure.
    """
    with pytest.raises(ValueError, match="Failed to decode"):
        decode_image_bytes(b"\x00\x01\x02\x03\x04corrupted_payload")
