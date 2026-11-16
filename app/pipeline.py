"""Concurrent inference pipeline combining wyolo vehicle detection, ROI cropping, and OpenALPR."""

from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from wyolo.inference import YOLOInference

from .client import OpenALPRClient
from .config import settings
from .decoder import decode_image_bytes

logger = logging.getLogger(__name__)


class ALPRPipeline:
    """Coordinates concurrent vehicle detection, ROI pre-cropping, and license plate OCR."""

    def __init__(
        self,
        yolo_predictor: YOLOInference,
        alpr_client: OpenALPRClient,
        vehicle_classes: Optional[Set[int]] = None,
        enable_roi_crop: bool = True,
        roi_padding: float = 0.15,
        max_workers: int = 4,
    ) -> None:
        """Initialize pipeline with optional ROI pre-cropping optimization.

        Args:
            yolo_predictor: Initialized wyolo YOLOInference engine.
            alpr_client: Initialized OpenALPRClient instance.
            vehicle_classes: Set of COCO class IDs considered vehicles.
            enable_roi_crop: Whether to crop vehicle ROI before querying OpenALPR.
            roi_padding: Percentage padding to add around vehicle bounding box.
            max_workers: ThreadPoolExecutor worker count for concurrent dispatch.
        """
        self.yolo = yolo_predictor
        self.alpr = alpr_client
        self.vehicle_classes = vehicle_classes or settings.vehicle_classes
        self.enable_roi_crop = enable_roi_crop
        self.roi_padding = roi_padding
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _infer_vehicles(self, image_input: Any) -> List[Dict[str, Any]]:
        """Run YOLO inference and filter for vehicle classes.

        Args:
            image_input: Decoded image array or source.

        Returns:
            List of detected vehicle bounding boxes and metadata sorted by size.
        """
        raw_detections = self.yolo.predict(image_input)
        filtered_vehicles: List[Dict[str, Any]] = []

        for det in raw_detections:
            cls_id = int(det.get("class", -1))
            if cls_id in self.vehicle_classes:
                x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
                area = max(0, x2 - x1) * max(0, y2 - y1)
                det_copy = dict(det)
                det_copy["area"] = area
                filtered_vehicles.append(det_copy)

        # Sort descending by bounding box area (largest vehicle first)
        filtered_vehicles.sort(key=lambda item: item["area"], reverse=True)
        return filtered_vehicles

    def _compute_padded_roi(
        self, bbox: List[int], img_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Compute bounding box coordinates with contextual padding.

        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates.
            img_shape: (height, width) of the source image.

        Returns:
            Tuple of clamped (x1, y1, x2, y2) with padding.
        """
        h_img, w_img = img_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = int(bw * self.roi_padding)
        pad_y = int(bh * self.roi_padding)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w_img, x2 + pad_x)
        crop_y2 = min(h_img, y2 + pad_y)

        return crop_x1, crop_y1, crop_x2, crop_y2

    def process_single(
        self,
        image_bytes: bytes,
        image_bgr: Optional[np.ndarray] = None,
        annotate: bool = False,
    ) -> Dict[str, Any]:
        """Execute vehicle detection and plate OCR with optional ROI pre-cropping.

        If `enable_roi_crop` is active, YOLO detects vehicles first (~8ms), crops the
        primary vehicle (~400x300 instead of full 1080p/4K), and submits the lightweight
        crop to OpenALPR (~15ms instead of 75ms).

        Args:
            image_bytes: Raw JPEG bytes for fallback or direct OpenALPR processing.
            image_bgr: Decoded OpenCV image matrix.
            annotate: Whether to generate an annotated image with drawn boxes.

        Returns:
            Structured dictionary with detection, recognition, and latency metrics.
        """
        t_start = time.perf_counter()

        if image_bgr is None:
            image_bgr, _ = decode_image_bytes(image_bytes)

        vehicles: List[Dict[str, Any]] = []
        plate_text: Optional[str] = None
        plate_conf: Optional[float] = None
        all_plates: List[Dict[str, Any]] = []
        alpr_error: Optional[str] = None
        roi_cropped = False
        crop_bbox: Optional[List[int]] = None

        if self.enable_roi_crop:
            # 1. Detect vehicles using wyolo (fast TensorRT/FP16)
            vehicles = self._infer_vehicles(image_bgr)

            if vehicles:
                # 2. Extract primary vehicle and crop ROI
                primary = vehicles[0]
                orig_bbox = primary.get("bbox", [0, 0, 0, 0])
                c_x1, c_y1, c_x2, c_y2 = self._compute_padded_roi(
                    orig_bbox, image_bgr.shape
                )
                crop_bgr = image_bgr[c_y1:c_y2, c_x1:c_x2]

                if crop_bgr.size > 0:
                    _, crop_encoded = cv2.imencode(
                        ".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    crop_bytes = crop_encoded.tobytes()

                    # Query OpenALPR with the lightweight vehicle crop
                    plate_text, plate_conf, all_plates, alpr_error = (
                        self.alpr.process_image_bytes(crop_bytes)
                    )
                    roi_cropped = True
                    crop_bbox = [c_x1, c_y1, c_x2, c_y2]

            # If no plate found in crop or no vehicle detected, fallback to full image
            if not plate_text and not alpr_error:
                plate_text, plate_conf, all_plates, alpr_error = (
                    self.alpr.process_image_bytes(image_bytes)
                )

        else:
            # Parallel execution without pre-cropping
            fut_yolo = self.executor.submit(self._infer_vehicles, image_bgr)
            fut_alpr = self.executor.submit(self.alpr.process_image_bytes, image_bytes)

            vehicles = fut_yolo.result()
            plate_text, plate_conf, all_plates, alpr_error = fut_alpr.result()

        total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        primary_vehicle = vehicles[0] if vehicles else None

        result: Dict[str, Any] = {
            "status": "success" if (plate_text or vehicles) else "empty",
            "license_plate": {
                "plate": plate_text,
                "confidence": plate_conf,
                "candidates": all_plates,
            },
            "primary_vehicle": primary_vehicle,
            "all_vehicles": vehicles,
            "latency_ms": total_latency_ms,
            "roi_cropped": roi_cropped,
            "crop_bbox": crop_bbox,
            "backend_error": alpr_error,
        }

        if annotate and image_bgr is not None:
            annotated_bgr = self.annotate_image(
                image_bgr=image_bgr,
                vehicles=vehicles,
                plate_text=plate_text,
                plate_conf=plate_conf,
            )
            result["annotated_image"] = annotated_bgr

        return result

    @staticmethod
    def annotate_image(
        image_bgr: np.ndarray,
        vehicles: List[Dict[str, Any]],
        plate_text: Optional[str] = None,
        plate_conf: Optional[float] = None,
    ) -> np.ndarray:
        """Draw bounding boxes and recognition labels onto the image.

        Args:
            image_bgr: OpenCV BGR image matrix.
            vehicles: List of detected vehicles sorted by size.
            plate_text: Recognized license plate text.
            plate_conf: Recognition confidence percentage.

        Returns:
            Annotated OpenCV BGR image matrix.
        """
        annotated = image_bgr.copy()

        for vehicle in vehicles:
            bbox = vehicle.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if vehicles:
            prim = vehicles[0]
            x1, y1, x2, y2 = prim.get("bbox", [0, 0, 0, 0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f"Plate: {plate_text or 'N/A'}"
            if plate_conf is not None:
                label += f" ({plate_conf:.1f}%)"

            (w, h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            text_y = max(y1 - 10, h + 10)
            cv2.rectangle(
                annotated,
                (x1, text_y - h - baseline),
                (x1 + w, text_y + baseline),
                (0, 0, 255),
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated
