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
from .tracker import VehicleTracker

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
        max_workers: int = 16,
        max_crop_dim: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
        max_concurrent_vehicles: int = 4,
        enable_tracking: bool = False,
        enhance_contrast: bool = True,
    ) -> None:
        """Initialize pipeline with optional ROI pre-cropping optimization.

        Args:
            yolo_predictor: Initialized wyolo YOLOInference engine.
            alpr_client: Initialized OpenALPRClient instance.
            vehicle_classes: Set of COCO class IDs considered vehicles.
            enable_roi_crop: Whether to crop vehicle ROI before querying OpenALPR.
            roi_padding: Percentage padding to add around vehicle bounding box.
            max_workers: ThreadPoolExecutor worker count for concurrent dispatch.
            max_crop_dim: Maximum width/height dimension for vehicle crop before OpenALPR.
            jpeg_quality: JPEG compression quality for cropped payload (1-100).
            max_concurrent_vehicles: Maximum vehicles to query concurrently per frame.
            enable_tracking: Whether to track vehicles across frames and cache plates.
            enhance_contrast: Whether to apply CLAHE contrast enhancement on crops.
        """
        self.yolo = yolo_predictor
        self.alpr = alpr_client
        self.vehicle_classes = vehicle_classes or settings.vehicle_classes
        self.enable_roi_crop = enable_roi_crop
        self.roi_padding = roi_padding
        self.max_crop_dim = max_crop_dim or settings.max_crop_dim
        self.jpeg_quality = jpeg_quality or settings.jpeg_quality
        self.max_concurrent_vehicles = max_concurrent_vehicles
        self.enable_tracking = enable_tracking
        self.enhance_contrast = enhance_contrast
        self.tracker = VehicleTracker()
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
            cls_id = int(det.get("class_id", det.get("class", -1)))
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
            detected_plates_map: List[Tuple[str, float]] = []

            if vehicles:
                tracked_info = (
                    self.tracker.update(vehicles)
                    if self.enable_tracking
                    else [(v, None, None, None) for v in vehicles]
                )

                # Prepare crops for concurrent dispatch across available vehicles
                crops_to_query: List[Tuple[Optional[int], bytes, Dict[str, Any]]] = []
                for det, track_id, cached_plate, cached_conf in tracked_info[: self.max_concurrent_vehicles]:
                    # Fast-path: if plate was already recognized with high confidence, reuse it
                    if cached_plate and cached_conf:
                        detected_plates_map.append((cached_plate, cached_conf))
                        continue

                    orig_bbox = det.get("bbox", [0, 0, 0, 0])
                    c_x1, c_y1, c_x2, c_y2 = self._compute_padded_roi(
                        orig_bbox, image_bgr.shape
                    )
                    crop_bgr = image_bgr[c_y1:c_y2, c_x1:c_x2]

                    if crop_bgr.size > 0:
                        # CLAHE contrast enhancement for challenging lighting conditions
                        if self.enhance_contrast:
                            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            cl = clahe.apply(l)
                            crop_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

                        # Adaptive crop rescaling: cap maximum dimension
                        ch, cw = crop_bgr.shape[:2]
                        if max(ch, cw) > self.max_crop_dim:
                            scale = self.max_crop_dim / float(max(ch, cw))
                            crop_bgr = cv2.resize(
                                crop_bgr,
                                (int(cw * scale), int(ch * scale)),
                                interpolation=cv2.INTER_AREA,
                            )

                        crops_to_query.append((track_id, crop_bgr, det))

                # Dispatch queries in parallel via thread pool executor
                if crops_to_query:
                    def _query_alpr(img_data: Any):
                        if hasattr(self.alpr, "process_image_bgr") and isinstance(img_data, np.ndarray):
                            res = self.alpr.process_image_bgr(img_data)
                            if isinstance(res, (tuple, list)) and len(res) == 4:
                                return res
                        if isinstance(img_data, np.ndarray):
                            _, enc = cv2.imencode(
                                ".jpg", img_data, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                            )
                            return self.alpr.process_image_bytes(enc.tobytes())
                        return self.alpr.process_image_bytes(img_data)

                    futures = [
                        (item[0], item[2], self.executor.submit(_query_alpr, item[1]))
                        for item in crops_to_query
                    ]
                    for track_id, det, fut in futures:
                        p_text, p_conf, p_candidates, err = fut.result()
                        if p_text and p_conf:
                            detected_plates_map.append((p_text, p_conf))
                            if self.enable_tracking and track_id is not None:
                                self.tracker.record_plate(track_id, p_text, p_conf)
                        if err and not alpr_error:
                            alpr_error = err

                roi_cropped = len(crops_to_query) > 0
                if vehicles:
                    primary_bbox = vehicles[0].get("bbox", [0, 0, 0, 0])
                    crop_bbox = list(self._compute_padded_roi(primary_bbox, image_bgr.shape))

            # If no plate found in any vehicle crop, fallback to full image
            if not detected_plates_map and not alpr_error:
                res = None
                if hasattr(self.alpr, "process_image_bgr") and image_bgr is not None:
                    candidate_res = self.alpr.process_image_bgr(image_bgr)
                    if isinstance(candidate_res, (tuple, list)) and len(candidate_res) == 4:
                        res = candidate_res
                if res is None:
                    res = self.alpr.process_image_bytes(image_bytes)
                p_text, p_conf, p_candidates, alpr_error = res
                if p_text and p_conf:
                    detected_plates_map.append((p_text, p_conf))

            plate_text = detected_plates_map[0][0] if detected_plates_map else None
            plate_conf = detected_plates_map[0][1] if detected_plates_map else None
            all_plates = [{"plate": p, "confidence": c} for p, c in detected_plates_map]

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
