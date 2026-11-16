"""High-performance thread-safe HTTP client for the OpenALPR backend."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)


class OpenALPRClient:
    """Thread-safe persistent client managing pooled connections to OpenALPR."""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 15.0,
        pool_connections: int = 20,
        pool_maxsize: int = 50,
        max_retries: int = 2,
    ) -> None:
        """Initialize OpenALPR HTTP client with persistent connection pooling.

        Args:
            endpoint: URL of the OpenALPR backend /process_image/ endpoint.
            timeout: Request timeout in seconds.
            pool_connections: Number of connection pools to cache.
            pool_maxsize: Maximum number of connections to save in the pool.
            max_retries: Number of retry attempts for idempotent operations.
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = requests.Session()

        retries = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retries,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def process_image_bytes(
        self, image_bytes: bytes, filename: str = "frame.jpg"
    ) -> Tuple[Optional[str], Optional[float], List[Dict[str, Any]], Optional[str]]:
        """Send raw image bytes to OpenALPR backend and parse plate results.

        Args:
            image_bytes: Raw JPEG or PNG bytes.
            filename: Virtual filename for multipart payload.

        Returns:
            Tuple containing:
                - best_plate (Optional[str]): Most confident license plate text.
                - best_confidence (Optional[float]): Confidence percentage (0-100).
                - all_plates (List[Dict[str, Any]]): All detected candidate plates.
                - error_message (Optional[str]): Error description if request failed.
        """
        files = {"image": (filename, image_bytes, "image/jpeg")}

        try:
            response = self.session.post(
                self.endpoint, files=files, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "error":
                err = data.get("message", "OpenALPR backend returned error status")
                logger.warning(f"OpenALPR backend error: {err}")
                return None, None, [], str(err)

            command_output = data.get("command_output")
            if not command_output or not isinstance(command_output, list):
                logger.info("OpenALPR backend returned empty results.")
                return None, None, [], None

            candidates: List[Dict[str, Any]] = []
            for item in command_output:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    plate_text = str(item[0]).strip()
                    try:
                        confidence = float(item[1])
                    except (ValueError, TypeError):
                        confidence = 0.0
                    candidates.append({"plate": plate_text, "confidence": confidence})

            if not candidates:
                return None, None, [], None

            best = candidates[0]
            return best["plate"], best["confidence"], candidates, None

        except requests.exceptions.Timeout:
            err = f"OpenALPR request timed out after {self.timeout}s"
            logger.error(err)
            return None, None, [], err
        except requests.exceptions.RequestException as exc:
            err = f"OpenALPR connection failed: {exc}"
            logger.error(err)
            return None, None, [], err
        except Exception as exc:
            err = f"Unexpected error during OpenALPR query: {exc}"
            logger.error(err)
            return None, None, [], err

    def process_image_bgr(
        self, image_bgr: Any, jpeg_quality: int = 80
    ) -> Tuple[Optional[str], Optional[float], List[Dict[str, Any]], Optional[str]]:
        """Encode OpenCV BGR image matrix and process via OpenALPR.

        Args:
            image_bgr: OpenCV image matrix in BGR format.
            jpeg_quality: JPEG compression quality (1-100).

        Returns:
            Tuple of (winning_plate, winning_confidence, candidate_list, error_message)
        """
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            return None, None, [], "Failed to encode image to JPEG"
        return self.process_image_bytes(encoded.tobytes())

    def close(self) -> None:
        """Close underlying HTTP session and release sockets."""
        self.session.close()
