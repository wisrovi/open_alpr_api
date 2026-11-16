"""Configuration settings for open_alpr_api."""

import os
from dataclasses import dataclass, field
from typing import Set


@dataclass
class Settings:
    """Application runtime settings loaded from environment variables."""

    # Server configuration
    host: str = field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("API_PORT", "1037"))
    )
    api_path: str = field(
        default_factory=lambda: os.getenv("API_PATH", "/process_image/")
    )

    # OpenALPR backend service configuration
    openalpr_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "OPENALPR_ENDPOINT",
            os.getenv("API_ENDPOINT", "http://alpr:8000/process_image/"),
        )
    )
    openalpr_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("OPENALPR_TIMEOUT_SEC", "15.0"))
    )

    # Model and inference configuration
    yolo_model: str = field(
        default_factory=lambda: os.getenv("YOLO_MODEL", "yolo11n.pt")
    )
    imgsz: int = field(
        default_factory=lambda: int(os.getenv("IMGSZ", "640"))
    )
    conf_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONF_THRESHOLD", "0.25"))
    )
    quantization: int = field(
        default_factory=lambda: int(os.getenv("QUANTIZATION", "16"))
    )
    max_batch_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "8"))
    )
    batch_timeout: float = field(
        default_factory=lambda: float(os.getenv("BATCH_TIMEOUT", "0.0"))
    )
    workers_per_device: int = field(
        default_factory=lambda: int(os.getenv("WORKERS_PER_DEVICE", "1"))
    )
    accelerator: str = field(
        default_factory=lambda: os.getenv("ACCELERATOR", "auto")
    )

    # Adaptive Crop and Encoding Optimizations
    max_crop_dim: int = field(
        default_factory=lambda: int(os.getenv("MAX_CROP_DIM", "680"))
    )
    jpeg_quality: int = field(
        default_factory=lambda: int(os.getenv("JPEG_QUALITY", "80"))
    )

    # Target COCO classes for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: Set[int] = field(
        default_factory=lambda: {2, 3, 5, 7}
    )

    # ROI Assisted Pre-Cropping optimization
    enable_roi_crop: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ROI_CROP", "true").lower() in ("true", "1", "yes")
    )
    roi_padding: float = field(
        default_factory=lambda: float(os.getenv("ROI_PADDING", "0.15"))
    )


settings = Settings()
