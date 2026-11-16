"""Unit tests for ALPRPipeline.

This module validates the concurrent pipeline connecting vehicle detection (wyolo)
and plate recognition (OpenALPR backend). It verifies class filtering, sorting by
bounding box area, ROI pre-cropping, parallel execution, structured output synthesis,
and image annotation.
"""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.client import OpenALPRClient
from app.pipeline import ALPRPipeline


@pytest.fixture
def mock_yolo() -> MagicMock:
    """Fixture providing a mock YOLOInference object."""
    yolo = MagicMock()
    return yolo


@pytest.fixture
def mock_alpr() -> MagicMock:
    """Fixture providing a mock OpenALPRClient object."""
    alpr = MagicMock(spec=OpenALPRClient)
    return alpr


@pytest.fixture
def sample_image() -> np.ndarray:
    """Fixture providing a synthetic 100x100 BGR test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    return img


@pytest.fixture
def sample_image_bytes(sample_image: np.ndarray) -> bytes:
    """Fixture providing encoded JPEG bytes of sample_image."""
    _, buf = cv2.imencode(".jpg", sample_image)
    return buf.tobytes()


def test_pipeline_filters_and_sorts_vehicles(
    mock_yolo: MagicMock, mock_alpr: MagicMock, sample_image_bytes: bytes, sample_image: np.ndarray
) -> None:
    """Validate that the pipeline correctly filters non-vehicle classes and sorts by area.

    What this test does:
    1. Simulates YOLO detections containing non-vehicle (class 0) and vehicle classes (2 and 7).
    2. Runs `process_single` with `enable_roi_crop=False`.
    3. Asserts that non-vehicle classes are ignored and primary vehicle is the largest vehicle.
    """
    mock_yolo.predict.return_value = [
        {"bbox": [0, 0, 10, 20], "conf": 0.9, "class": 0, "class_name": "person"},
        {"bbox": [10, 10, 20, 20], "conf": 0.8, "class": 2, "class_name": "car"},
        {"bbox": [20, 20, 50, 50], "conf": 0.85, "class": 7, "class_name": "truck"},
    ]
    mock_alpr.process_image_bytes.return_value = (
        "ABC1234",
        95.0,
        [{"plate": "ABC1234", "confidence": 95.0}],
        None,
    )

    pipeline = ALPRPipeline(
        yolo_predictor=mock_yolo, alpr_client=mock_alpr, enable_roi_crop=False
    )
    result = pipeline.process_single(image_bytes=sample_image_bytes, image_bgr=sample_image)

    assert result["status"] == "success"
    assert len(result["all_vehicles"]) == 2
    for v in result["all_vehicles"]:
        assert v["class"] in {2, 3, 5, 7}

    assert result["primary_vehicle"] is not None
    assert result["primary_vehicle"]["class"] == 7
    assert result["primary_vehicle"]["area"] == 900
    assert result["license_plate"]["plate"] == "ABC1234"


def test_pipeline_roi_crop_executed(
    mock_yolo: MagicMock, mock_alpr: MagicMock, sample_image_bytes: bytes, sample_image: np.ndarray
) -> None:
    """Validate ROI pre-cropping when a vehicle is detected.

    What this test does:
    1. Simulates a vehicle detected by wyolo at [20, 20, 80, 80].
    2. Simulates OpenALPR successfully reading plate from the crop.
    3. Asserts that `roi_cropped` is True and crop coordinates are padded.
    """
    mock_yolo.predict.return_value = [
        {"bbox": [20, 20, 80, 80], "conf": 0.92, "class": 2, "class_name": "car"}
    ]
    mock_alpr.process_image_bytes.return_value = ("ROI999", 98.0, [], None)

    pipeline = ALPRPipeline(
        yolo_predictor=mock_yolo, alpr_client=mock_alpr, enable_roi_crop=True
    )
    result = pipeline.process_single(image_bytes=sample_image_bytes, image_bgr=sample_image)

    assert result["status"] == "success"
    assert result["roi_cropped"] is True
    assert result["crop_bbox"] is not None
    # Crop bbox should expand beyond [20, 20, 80, 80] due to padding
    assert result["crop_bbox"][0] < 20
    assert result["crop_bbox"][1] < 20
    assert result["crop_bbox"][2] > 80
    assert result["crop_bbox"][3] > 80
    assert result["license_plate"]["plate"] == "ROI999"


def test_compute_padded_roi_boundary_clamping(mock_yolo: MagicMock, mock_alpr: MagicMock) -> None:
    """Validate that padded ROI coordinates never exceed image dimensions or go negative.

    What this test does:
    1. Tests corner cases where bbox is at edge [0, 0, 50, 50] on a 100x100 image.
    2. Confirms x1 and y1 clamp at 0, and x2 and y2 clamp at 100.
    """
    pipeline = ALPRPipeline(
        yolo_predictor=mock_yolo, alpr_client=mock_alpr, roi_padding=0.2
    )
    c_x1, c_y1, c_x2, c_y2 = pipeline._compute_padded_roi([0, 0, 95, 95], (100, 100))
    assert c_x1 == 0
    assert c_y1 == 0
    assert c_x2 == 100
    assert c_y2 == 100


def test_pipeline_handles_no_detections(
    mock_yolo: MagicMock, mock_alpr: MagicMock, sample_image_bytes: bytes, sample_image: np.ndarray
) -> None:
    """Validate behavior when neither vehicles nor plates are detected in the frame.

    What this test does:
    1. Returns empty detection lists from both YOLO and OpenALPR.
    2. Asserts that the pipeline status is 'empty', primary_vehicle is None,
       and latency_ms is recorded properly.
    """
    mock_yolo.predict.return_value = []
    mock_alpr.process_image_bytes.return_value = (None, None, [], None)

    pipeline = ALPRPipeline(yolo_predictor=mock_yolo, alpr_client=mock_alpr)
    result = pipeline.process_single(image_bytes=sample_image_bytes, image_bgr=sample_image)

    assert result["status"] == "empty"
    assert result["primary_vehicle"] is None
    assert result["all_vehicles"] == []
    assert result["license_plate"]["plate"] is None
    assert result["latency_ms"] >= 0.0


def test_pipeline_annotation_generation(
    mock_yolo: MagicMock, mock_alpr: MagicMock, sample_image_bytes: bytes, sample_image: np.ndarray
) -> None:
    """Validate that image annotation generates a valid modified OpenCV matrix.

    What this test does:
    1. Calls `process_single` with `annotate=True`.
    2. Confirms that `annotated_image` is present in the output dictionary.
    3. Confirms that the returned matrix has identical dimensions and channel count.
    """
    mock_yolo.predict.return_value = [
        {"bbox": [10, 10, 40, 40], "conf": 0.88, "class": 2, "class_name": "car"}
    ]
    mock_alpr.process_image_bytes.return_value = ("9999XYZ", 91.2, [], None)

    pipeline = ALPRPipeline(yolo_predictor=mock_yolo, alpr_client=mock_alpr)
    result = pipeline.process_single(
        image_bytes=sample_image_bytes, image_bgr=sample_image, annotate=True
    )

    assert "annotated_image" in result
    annotated = result["annotated_image"]
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == sample_image.shape


def test_pipeline_adaptive_crop_rescaling(
    mock_yolo: MagicMock, mock_alpr: MagicMock
) -> None:
    """Validate that overly large vehicle crops are adaptively rescaled before OpenALPR query.

    What this test does:
    1. Creates a large 1000x1000 image with a large vehicle detection [100, 100, 900, 900].
    2. Initializes ALPRPipeline with max_crop_dim=500.
    3. Verifies that the crop sent to mock_alpr has dimensions constrained to max_crop_dim.
    """
    large_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", large_img)
    large_bytes = buf.tobytes()

    mock_yolo.predict.return_value = [
        {"bbox": [100, 100, 900, 900], "conf": 0.95, "class": 2, "class_name": "car"}
    ]
    captured_bytes = []

    def capture_alpr_call(crop_data: bytes) -> tuple:
        captured_bytes.append(crop_data)
        return ("RESCALE123", 95.0, [], None)

    mock_alpr.process_image_bytes.side_effect = capture_alpr_call

    pipeline = ALPRPipeline(
        yolo_predictor=mock_yolo,
        alpr_client=mock_alpr,
        max_crop_dim=500,
        enable_roi_crop=True,
    )
    result = pipeline.process_single(image_bytes=large_bytes, image_bgr=large_img)

    assert result["status"] == "success"
    assert len(captured_bytes) == 1
    # Decode captured bytes and verify dimension was rescaled <= 500
    nparr = np.frombuffer(captured_bytes[0], np.uint8)
    decoded_crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert max(decoded_crop.shape[:2]) <= 500
