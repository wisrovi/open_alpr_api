"""Unit tests for GPU-accelerated ALPR Engine (YOLO-LPD + Deep Learning OCR)."""

from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.gpu_alpr import GPUALPREngine, setup_nvidia_cuda_libraries


def test_setup_nvidia_cuda_libraries_executes_safely() -> None:
    """Ensure setup_nvidia_cuda_libraries runs without raising exceptions."""
    setup_nvidia_cuda_libraries()


def test_gpu_alpr_empty_input_returns_empty() -> None:
    """Validate that None or empty image input gracefully returns empty list."""
    with patch("fast_alpr.ALPR"):
        engine = GPUALPREngine(device="cpu")
        assert engine.predict_bgr(None) == []
        assert engine.predict_bgr(np.array([])) == []
        plate, conf, candidates, err = engine.process_image_bgr(None)
        assert plate is None
        assert conf is None
        assert candidates == []
        assert err is None


def test_gpu_alpr_mock_inference_success() -> None:
    """Validate successful prediction mapping from fast_alpr result structures."""
    mock_ocr = MagicMock()
    mock_ocr.text = "1234-XYZ"
    mock_ocr.confidence = [0.95, 0.96, 0.94]

    mock_bbox = MagicMock()
    mock_bbox.x1, mock_bbox.y1, mock_bbox.x2, mock_bbox.y2 = 10, 20, 100, 50

    mock_detection = MagicMock()
    mock_detection.bounding_box = mock_bbox

    mock_res = MagicMock()
    mock_res.ocr = mock_ocr
    mock_res.detection = mock_detection

    with patch("fast_alpr.ALPR") as MockALPR:
        mock_instance = MockALPR.return_value
        mock_instance.predict.return_value = [mock_res]

        engine = GPUALPREngine(device="cpu")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        plate, conf, candidates, err = engine.process_image_bgr(dummy_img)

        assert plate == "1234XYZ"
        assert conf == pytest.approx(95.0, rel=1e-2)
        assert len(candidates) == 1
        assert candidates[0]["bbox"] == [10, 20, 100, 50]
        assert err is None


def test_gpu_alpr_process_image_bytes_decodes() -> None:
    """Validate process_image_bytes successfully decodes JPEG bytes."""
    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)

    with patch("fast_alpr.ALPR") as MockALPR:
        mock_instance = MockALPR.return_value
        mock_instance.predict.return_value = []

        engine = GPUALPREngine(device="cpu")
        plate, conf, candidates, err = engine.process_image_bytes(encoded.tobytes())

        assert plate is None
        assert conf is None
        assert candidates == []
        assert err is None

        # Test invalid corrupted bytes
        p, c, cands, err_msg = engine.process_image_bytes(b"not_an_image")
        assert err_msg == "Failed to decode image bytes"


def test_gpu_alpr_real_image_inference() -> None:
    """Validate end-to-end inference against the test image rojo_4.png."""
    import os

    test_img_path = "test/rojo_4.png"
    if not os.path.exists(test_img_path):
        pytest.skip("rojo_4.png test image not found")

    img = cv2.imread(test_img_path)
    engine = GPUALPREngine(device="cuda")
    plate, conf, candidates, err = engine.process_image_bgr(img)

    assert err is None
    assert plate == "2000DFL"
    assert conf is not None
    assert conf > 50.0
    assert len(candidates) >= 1
