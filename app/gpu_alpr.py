"""GPU-accelerated ALPR Engine utilizing YOLO-LPD detector and Deep Learning OCR (MobileViT/CRNN)."""

import ctypes
import glob
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def setup_nvidia_cuda_libraries() -> None:
    """Preload NVIDIA CUDA/cuDNN shared libraries bundled with pip/conda if needed."""
    for base in sys.path:
        nvidia_path = os.path.join(base, "nvidia")
        if os.path.isdir(nvidia_path):
            for lib_dir in sorted(glob.glob(os.path.join(nvidia_path, "*/lib"))):
                for so_file in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                    try:
                        ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


class GPUALPREngine:
    """End-to-end Deep Learning License Plate Recognition running on NVIDIA GPU."""

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "european-plates-mobile-vit-v2-model",
        device: str = "cuda",
        conf_thresh: float = 0.35,
    ) -> None:
        """Initialize YOLO-LPD detector and OCR recognizer on GPU with CPU fallback.

        Args:
            detector_model: ONNX/PyTorch model identifier for plate localization.
            ocr_model: OCR model identifier for character sequence recognition.
            device: Target compute device ("cuda", "cpu", or "auto").
            conf_thresh: Detection confidence threshold.
        """
        setup_nvidia_cuda_libraries()
        self.device = device
        self.conf_thresh = conf_thresh

        try:
            from fast_alpr import ALPR

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.alpr = ALPR(
                detector_model=detector_model,
                detector_conf_thresh=conf_thresh,
                detector_providers=providers,
                ocr_model=ocr_model,
                ocr_device=device,
                ocr_providers=providers,
            )
            logger.info(
                "GPU ALPR Engine successfully initialized with %s + %s on %s",
                detector_model,
                ocr_model,
                device,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize GPU ALPR Engine on %s: %s. Falling back to CPU.",
                device,
                exc,
            )
            from fast_alpr import ALPR

            self.alpr = ALPR(
                detector_model=detector_model,
                detector_conf_thresh=conf_thresh,
                detector_providers=["CPUExecutionProvider"],
                ocr_model=ocr_model,
                ocr_device="cpu",
                ocr_providers=["CPUExecutionProvider"],
            )

    def predict_bgr(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run plate detection and OCR on a BGR numpy array.

        Args:
            image_bgr: Source image matrix in BGR format.

        Returns:
            List of detected plate structures with text, confidence, bbox, and candidates.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        try:
            results = self.alpr.predict(image_bgr)
        except Exception as exc:
            logger.error("GPU ALPR inference error: %s", exc)
            return []

        output: List[Dict[str, Any]] = []
        for res in results:
            if not res or not hasattr(res, "ocr") or not res.ocr:
                continue

            text = getattr(res.ocr, "text", "")
            if not text:
                continue

            # Clean and normalize plate string
            cleaned_text = "".join(ch for ch in text if ch.isalnum()).upper()
            if not cleaned_text:
                continue

            conf_scores = getattr(res.ocr, "confidence", [0.8])
            if isinstance(conf_scores, (list, tuple)):
                mean_conf = float(np.mean(conf_scores)) * 100.0 if conf_scores else 80.0
            else:
                mean_conf = float(conf_scores) * 100.0 if float(conf_scores) <= 1.0 else float(conf_scores)

            det_box = getattr(res, "detection", None)
            bbox = None
            if det_box and hasattr(det_box, "bounding_box"):
                bb = det_box.bounding_box
                bbox = [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)]

            output.append(
                {
                    "plate": cleaned_text,
                    "confidence": round(mean_conf, 2),
                    "bbox": bbox,
                }
            )

        # Sort highest confidence first
        output.sort(key=lambda item: item["confidence"], reverse=True)
        return output

    def process_image_bgr(
        self, image_bgr: np.ndarray
    ) -> Tuple[Optional[str], Optional[float], List[Dict[str, Any]], Optional[str]]:
        """Process image BGR array returning plate, confidence, candidates, and error.

        Matches OpenALPRClient signature.

        Args:
            image_bgr: Source OpenCV BGR image array.

        Returns:
            Tuple of (best_plate_text, best_confidence, all_plates_list, error_message)
        """
        plates = self.predict_bgr(image_bgr)
        if not plates:
            return None, None, [], None

        best = plates[0]
        return best["plate"], best["confidence"], plates, None

    def process_image_bytes(
        self, image_bytes: bytes
    ) -> Tuple[Optional[str], Optional[float], List[Dict[str, Any]], Optional[str]]:
        """Decode image bytes and process on GPU ALPR.

        Matches OpenALPRClient signature.

        Args:
            image_bytes: Raw JPEG/PNG image bytes.

        Returns:
            Tuple of (best_plate_text, best_confidence, all_plates_list, error_message)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, [], "Failed to decode image bytes"

        return self.process_image_bgr(img)
