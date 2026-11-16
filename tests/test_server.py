"""Unit tests for OpenALPRLitAPI and server payload handlers.

This module validates LitServe integration, request decoding (multipart, base64, raw bytes),
accelerated SIMD decoding, error handling on corrupted image inputs, response formatting,
and server factory registration.
"""

import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.server import OpenALPRLitAPI, create_server


@pytest.fixture
def api_instance() -> OpenALPRLitAPI:
    """Fixture providing an uninitialized OpenALPRLitAPI instance with mocked pipeline."""
    api = OpenALPRLitAPI(api_path="/test_predict")
    api.pipeline = MagicMock()
    return api


@pytest.fixture
def synthetic_jpeg_bytes() -> bytes:
    """Fixture providing valid JPEG encoded bytes for testing."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.circle(img, (32, 32), 16, (0, 255, 0), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_decode_request_raw_bytes(
    api_instance: OpenALPRLitAPI, synthetic_jpeg_bytes: bytes
) -> None:
    """Validate request decoding when input is a dict containing raw bytes.

    What this test does:
    1. Passes `{'image': bytes}` to `decode_request`.
    2. Validates that it returns the original bytes and a decoded OpenCV matrix.
    """
    raw_bytes, matrix, annotate = api_instance.decode_request({"image": synthetic_jpeg_bytes})
    assert raw_bytes == synthetic_jpeg_bytes
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (64, 64, 3)
    assert annotate is False


def test_decode_request_base64_string(
    api_instance: OpenALPRLitAPI, synthetic_jpeg_bytes: bytes
) -> None:
    """Validate request decoding when input is a base64 encoded string with annotate=True.

    What this test does:
    1. Encodes JPEG bytes to base64 string with data URI prefix.
    2. Passes `{'image': b64_string, 'annotate': True}`.
    3. Validates that base64 is decoded properly and annotate is parsed as True.
    """
    b64_str = "data:image/jpeg;base64," + base64.b64encode(synthetic_jpeg_bytes).decode("utf-8")
    raw_bytes, matrix, annotate = api_instance.decode_request(
        {"image": b64_str, "annotate": True}
    )
    assert raw_bytes == synthetic_jpeg_bytes
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (64, 64, 3)
    assert annotate is True


def test_decode_request_invalid_bytes_raises_error(api_instance: OpenALPRLitAPI) -> None:
    """Validate that invalid/corrupt image bytes raise a descriptive ValueError.

    What this test does:
    1. Passes invalid binary data that cannot be decoded as an image.
    2. Verifies that `ValueError` is raised with an informative message.
    """
    with pytest.raises(ValueError, match="Failed to decode"):
        api_instance.decode_request({"image": b"corrupted_non_image_data"})


def test_encode_response_formatting(
    api_instance: OpenALPRLitAPI, synthetic_jpeg_bytes: bytes
) -> None:
    """Validate response encoding into standard JSON serializable dict.

    What this test does:
    1. Supplies sample pipeline output dictionary including ROI crop metadata.
    2. Asserts standard fields are preserved and pre-annotated image is base64 encoded.
    """
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    sample_output = {
        "status": "success",
        "license_plate": {"plate": "7777XYZ", "confidence": 98.0},
        "primary_vehicle": {"bbox": [0, 0, 30, 30], "class": 2},
        "all_vehicles": [{"bbox": [0, 0, 30, 30], "class": 2}],
        "latency_ms": 35.5,
        "roi_cropped": True,
        "crop_bbox": [0, 0, 30, 30],
        "backend_error": "Notice: low contrast",
        "annotated_image": img,
    }

    # Test default strict legacy response (only command_error, command_output, status)
    encoded = api_instance.encode_response(sample_output)
    assert encoded["status"] == "success"
    assert encoded["command_output"] == [["7777XYZ", "98.000"]]
    assert encoded["command_error"] == ["Notice: low contrast"]
    assert "annotated_image_base64" in encoded
    assert "license_plate" not in encoded
    assert "primary_vehicle" not in encoded

    # Test extended metadata response when explicitly requested
    sample_output_extended = {**sample_output, "extended": True}
    encoded_ext = api_instance.encode_response(sample_output_extended)
    assert encoded_ext["status"] == "success"
    assert encoded_ext["command_output"] == [["7777XYZ", "98.000"]]
    assert encoded_ext["license_plate"]["plate"] == "7777XYZ"
    assert encoded_ext["latency_ms"] == 35.5
    assert encoded_ext["roi_cropped"] is True


def test_predict_delegates_to_pipeline(
    api_instance: OpenALPRLitAPI, synthetic_jpeg_bytes: bytes
) -> None:
    """Validate that predict() forwards decoded components to pipeline.process_single.

    What this test does:
    1. Sets mock return value on api_instance.pipeline.process_single.
    2. Calls `predict((synthetic_jpeg_bytes, dummy_matrix, False))`.
    3. Verifies process_single was invoked with identical arguments.
    """
    dummy_matrix = np.zeros((10, 10, 3), dtype=np.uint8)
    api_instance.pipeline.process_single.return_value = {"status": "success"}

    result = api_instance.predict((synthetic_jpeg_bytes, dummy_matrix, False))
    api_instance.pipeline.process_single.assert_called_once_with(
        image_bytes=synthetic_jpeg_bytes,
        image_bgr=dummy_matrix,
        annotate=False,
    )
    assert result == {"status": "success"}


def test_create_server_initialization() -> None:
    """Validate LitServer instantiation and WebSocket stream route registration.

    What this test does:
    1. Calls `create_server(api_path='/custom_predict')`.
    2. Validates that LitServer instance is created and FastAPI routes include WebSocket stream.
    """
    server = create_server(api_path="/custom_predict")
    assert server is not None
    # Verify that '/ws/stream' route was added to the FastAPI app
    route_paths = [r.path for r in server.app.routes]
    assert "/ws/stream" in route_paths
