"""LitServe API implementation with SIMD decoding, ROI pre-cropping, and WebSocket streaming."""

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from fastapi import Request, Response, WebSocket, WebSocketDisconnect
import litserve as ls
import numpy as np
from wyolo.inference import YOLOInference

from .client import OpenALPRClient
from .config import settings
from .decoder import decode_image_bytes
from .pipeline import ALPRPipeline

logger = logging.getLogger(__name__)


class OpenALPRLitAPI(ls.LitAPI):
    """LitServe API for high-performance vehicle detection and license plate recognition."""

    def __init__(
        self,
        api_path: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        batch_timeout: Optional[float] = None,
    ) -> None:
        """Initialize LitAPI with custom endpoint path and dynamic batching parameters."""
        super().__init__()
        self.api_path = api_path or settings.api_path
        self.max_batch_size = max_batch_size or settings.max_batch_size
        self.batch_timeout = batch_timeout or settings.batch_timeout
        self.alpr_client: Optional[Any] = None
        self.yolo: Optional[YOLOInference] = None
        self.pipeline: Optional[ALPRPipeline] = None

    def setup(self, device: str) -> None:
        """Initialize models, clients, and execution pipeline on allocated device.

        Args:
            device: Accelerator device string provided by LitServe (e.g. 'cuda:0', 'cpu').
        """
        logger.info(f"Initializing OpenALPRLitAPI on device '{device}'...")

        # Initialize ALPR engine (GPU deep learning or C++ OpenALPR backend)
        if settings.alpr_engine == "gpu":
            from .gpu_alpr import GPUALPREngine

            gpu_device = "cuda" if "cuda" in str(device).lower() else settings.gpu_device
            logger.info(
                "Initializing GPU ALPR Engine (YOLO-LPD + MobileViT OCR) on %s...", gpu_device
            )
            self.alpr_client = GPUALPREngine(
                detector_model=settings.gpu_detector_model,
                ocr_model=settings.gpu_ocr_model,
                device=gpu_device,
            )
        else:
            logger.info("Initializing legacy OpenALPR C++ client connecting to %s...", settings.openalpr_endpoint)
            self.alpr_client = OpenALPRClient(
                endpoint=settings.openalpr_endpoint,
                timeout=settings.openalpr_timeout_sec,
            )

        # Ensure model file exists or download via Ultralytics
        if not os.path.exists(settings.yolo_model) and settings.yolo_model.endswith(".pt"):
            try:
                from ultralytics import YOLO
                YOLO(settings.yolo_model)
            except Exception as exc:
                logger.warning(f"Could not pre-download model: {exc}")

        # Initialize wyolo unified predictor (supports FP16 TensorRT or PyTorch)
        self.yolo = YOLOInference(
            model_path=settings.yolo_model,
            conf=settings.conf_threshold,
            imgsz=settings.imgsz,
            quantization=settings.quantization,
            auto_engine=True,
        )

        # Build pipeline with ROI pre-cropping
        self.pipeline = ALPRPipeline(
            yolo_predictor=self.yolo,
            alpr_client=self.alpr_client,
            vehicle_classes=settings.vehicle_classes,
            enable_roi_crop=settings.enable_roi_crop,
            roi_padding=settings.roi_padding,
        )

        # Run model warmup to eliminate initial kernel compilation latency
        try:
            logger.info("Executing pipeline warmup...")
            dummy_img = np.zeros((settings.imgsz, settings.imgsz, 3), dtype=np.uint8)
            self.yolo.predict(dummy_img)
            logger.info("Pipeline warmup completed successfully.")
        except Exception as exc:
            logger.warning(f"Warmup notice: {exc}")

    def decode_request(
        self, request: Request, **kwargs: Any
    ) -> Tuple[bytes, np.ndarray, bool]:
        """Decode incoming request payload into image bytes and decoded OpenCV matrix.

        Utilizes SIMD/torchvision accelerated decoding for up to 4x faster image decode.

        Args:
            request: Incoming Starlette/FastAPI request payload or dictionary.

        Returns:
            Tuple of (raw_image_bytes, image_bgr, annotate_flag).
        """
        raw_data: Any = None
        annotate: bool = False

        if isinstance(request, dict):
            annotate = bool(request.get("annotate", False))
            raw_data = request.get("image")
            if raw_data is None and "file" in request:
                raw_data = request.get("file")
        elif hasattr(request, "get"):
            raw_data = request.get("image")
        else:
            raw_data = request

        # Extract bytes from Starlette UploadFile if present
        if hasattr(raw_data, "file"):
            raw_data.file.seek(0)
            image_bytes = raw_data.file.read()
        elif hasattr(raw_data, "read"):
            image_bytes = raw_data.read()
        elif isinstance(raw_data, (bytes, bytearray)):
            image_bytes = bytes(raw_data)
        elif isinstance(raw_data, str):
            clean_b64 = raw_data.split(",", 1)[-1] if "," in raw_data else raw_data
            image_bytes = base64.b64decode(clean_b64)
        else:
            raise ValueError(
                "Invalid payload: expected 'image' field with binary bytes or base64 string."
            )

        image_bgr, _ = decode_image_bytes(image_bytes)
        return image_bytes, image_bgr, annotate

    def predict(
        self,
        decoded_input: Union[
            Tuple[bytes, np.ndarray, bool], List[Tuple[bytes, np.ndarray, bool]]
        ],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute inference pipeline on preprocessed input (handles single or batched inputs).

        Args:
            decoded_input: Either a single tuple or a list of tuples: (raw_image_bytes, image_bgr, annotate_flag).

        Returns:
            Pipeline result dictionary or list of result dictionaries when batched.
        """
        if self.pipeline is None:
            raise RuntimeError("API pipeline is not initialized. Run setup() first.")

        if isinstance(decoded_input, list):
            futures = [
                self.pipeline.executor.submit(
                    self.pipeline.process_single,
                    image_bytes=item[0],
                    image_bgr=item[1],
                    annotate=item[2],
                )
                for item in decoded_input
            ]
            return [f.result() for f in futures]

        image_bytes, image_bgr, annotate = decoded_input
        return self.pipeline.process_single(
            image_bytes=image_bytes,
            image_bgr=image_bgr,
            annotate=annotate,
        )

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format pipeline output with 100% backward-compatible keys and v1.1.0 extensions.

        Ensures full compatibility with legacy consumers expecting:
            - "status": "success" | "error" | "empty"
            - "command_output": [ [plate_str, confidence_str], ... ]
            - "command_error": [ ... ]
        While simultaneously providing rich v1.1.0 vehicle and latency metadata.

        Args:
            output: Result dictionary from pipeline.

        Returns:
            JSON-serializable response dictionary.
        """
        # Build 100% backward-compatible command_output: List[List[str]]
        # Returns one entry [plate, confidence] per detected vehicle plate
        command_output: List[List[str]] = []
        candidates = output.get("license_plate", {}).get("candidates", [])

        if candidates:
            for c in candidates:
                p = c.get("plate")
                conf = c.get("confidence")
                if p:
                    command_output.append(
                        [str(p), f"{conf:.3f}" if conf is not None else "0.0"]
                    )
        elif output.get("license_plate", {}).get("plate"):
            best_plate = output["license_plate"]["plate"]
            best_conf = output["license_plate"].get("confidence")
            command_output.append(
                [str(best_plate), f"{best_conf:.3f}" if best_conf is not None else "0.0"]
            )

        # Build 100% backward-compatible command_error: List[str]
        command_error: List[str] = []
        if output.get("backend_error"):
            command_error.append(str(output["backend_error"]))

        status_str = (
            "success"
            if (command_output or output.get("status") == "success")
            else "error"
            if command_error
            else "empty"
        )

        # 100% Strict Legacy API Schema
        response: Dict[str, Any] = {
            "command_error": command_error,
            "command_output": command_output,
            "status": status_str,
        }

        # Include extended debug metadata only if explicitly requested
        if output.get("extended"):
            response["license_plate"] = output.get("license_plate", {})
            response["primary_vehicle"] = output.get("primary_vehicle")
            response["all_vehicles"] = output.get("all_vehicles", [])
            response["latency_ms"] = output.get("latency_ms", 0.0)
            response["roi_cropped"] = output.get("roi_cropped", False)
            response["crop_bbox"] = output.get("crop_bbox")

        if "annotated_image" in output and output["annotated_image"] is not None:
            _, buffer = cv2.imencode(
                ".jpg", output["annotated_image"], [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            response["annotated_image_base64"] = base64.b64encode(buffer).decode("utf-8")

        return response


def create_server(api_path: Optional[str] = None) -> ls.LitServer:
    """Instantiate, configure LitServer, and register WebSocket streaming endpoint.

    Args:
        api_path: Custom endpoint path. Defaults to settings.api_path.

    Returns:
        Configured LitServer instance with WebSocket stream enabled.
    """
    api = OpenALPRLitAPI(
        api_path=api_path,
        max_batch_size=settings.max_batch_size,
        batch_timeout=settings.batch_timeout,
    )
    server = ls.LitServer(
        api,
        accelerator=settings.accelerator,
        workers_per_device=settings.workers_per_device,
    )

    # Register route aliases for seamless backward compatibility across all client paths
    api_target = api_path or settings.api_path
    handler = None
    for r in server.app.routes:
        if hasattr(r, "path") and r.path == api_target:
            handler = r.endpoint
            break
    if handler:
        for alias in ["/process_image", "/process_image/", "/predict"]:
            if alias != api_target:
                server.app.add_api_route(alias, handler, methods=["POST"])

    # Register Real-Time Bi-Directional WebSocket Streaming Route
    @server.app.websocket("/ws/stream")
    async def websocket_stream_endpoint(websocket: WebSocket) -> None:
        """Stream continuous frames to the pipeline via WebSocket connection."""
        await websocket.accept()
        logger.info("Real-time WebSocket client connected for frame streaming.")
        try:
            while True:
                # Receive binary frame from client
                frame_bytes = await websocket.receive_bytes()
                if not frame_bytes:
                    continue

                # Process through pipeline
                if api.pipeline is not None:
                    res = api.pipeline.process_single(image_bytes=frame_bytes)
                    resp = api.encode_response(res)
                    await websocket.send_json(resp)
                else:
                    await websocket.send_json({"error": "Pipeline not ready"})

        except WebSocketDisconnect:
            logger.info("WebSocket streaming client disconnected normally.")
        except Exception as exc:
            logger.warning(f"WebSocket streaming connection error: {exc}")
            try:
                await websocket.close()
            except Exception:
                pass

    return server


def main() -> None:
    """Run LitServe API server with command-line entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(f"Starting open_alpr_api LitServer on port {settings.port}...")
    server = create_server()
    server.run(host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
