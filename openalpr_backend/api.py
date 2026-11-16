"""Ultra-fast in-memory OpenALPR API backend.

Replaces slow disk writes and subprocess spawns with direct in-memory
C++ Python bindings (openalpr.Alpr) and a thread-safe instance pool.
"""

import logging
import os
import queue
import sys
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request

# Ensure OpenALPR Python bindings are on sys.path
OPENALPR_BINDINGS = "/srv/openalpr/src/bindings/python/openalpr"
if OPENALPR_BINDINGS not in sys.path and os.path.exists(OPENALPR_BINDINGS):
    sys.path.append(OPENALPR_BINDINGS)

import openalpr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("openalpr_inmemory_backend")

app = Flask(__name__)

# Configurable OpenALPR paths
OPENALPR_COUNTRY = os.getenv("OPENALPR_COUNTRY", "eu")
OPENALPR_CONF = os.getenv("OPENALPR_CONF", "/etc/openalpr/openalpr.conf")
OPENALPR_RUNTIME_DATA = os.getenv(
    "OPENALPR_RUNTIME_DATA", "/usr/share/openalpr/runtime_data"
)
POOL_SIZE = int(os.getenv("OPENALPR_POOL_SIZE", "4"))


class AlprPool:
    """Thread-safe pool of OpenALPR C++ engine instances loaded in memory."""

    def __init__(self, size: int = 4) -> None:
        self.pool = queue.Queue(maxsize=size)
        logger.info(
            f"Initializing OpenALPR in-memory engine pool (size={size}, country={OPENALPR_COUNTRY})..."
        )
        for i in range(size):
            instance = openalpr.Alpr(
                OPENALPR_COUNTRY, OPENALPR_CONF, OPENALPR_RUNTIME_DATA
            )
            if not instance.is_loaded():
                logger.error(f"Failed to load OpenALPR instance #{i}")
                raise RuntimeError("Failed to load OpenALPR C++ engine.")
            instance.set_top_n(25)
            self.pool.put(instance)
        logger.info(f"OpenALPR in-memory engine pool of {size} instances ready.")

    def get(self) -> openalpr.Alpr:
        return self.pool.get()

    def put(self, instance: openalpr.Alpr) -> None:
        self.pool.put(instance)


alpr_pool = AlprPool(size=POOL_SIZE)


def filter_spanish_plates(
    res: Dict[str, Any]
) -> Tuple[List[List[str]], List[str]]:
    """Filter and format OpenALPR candidate results for Spanish plate patterns.

    For each detected plate region (vehicle), selects the single best-matching candidate
    following Spanish plate formats, avoiding duplicate OCR candidate readings:
      - Old format: 8 characters (2 letters, 4 digits, 2 letters) e.g., 'MA1234AB'
      - New format: 7 characters (4 digits, 3 letters) e.g., '2000DFL'
    """
    rows: List[List[str]] = []

    for result in res.get("results", []):
        best_match: Optional[List[str]] = None
        for candidate in result.get("candidates", []):
            plate = str(candidate.get("plate", "")).strip().upper()
            conf = candidate.get("confidence", 0.0)

            # Old format: 2 letters, 4 numbers, 2 letters (8 chars)
            is_valid_old = (
                len(plate) == 8
                and plate[0:2].isalpha()
                and plate[2:6].isdigit()
                and plate[6:8].isalpha()
            )

            # New format: 4 numbers, 3 letters (7 chars)
            is_valid_new = (
                len(plate) == 7
                and plate[0:4].isdigit()
                and plate[4:7].isalpha()
            )

            if is_valid_old or is_valid_new:
                best_match = [plate, f"{conf:.3f}"]
                break  # Candidates are sorted descending by confidence, pick the winner

        if best_match:
            rows.append(best_match)

    return rows, []


@app.route("/process_image/", methods=["POST"])
@app.route("/process_image", methods=["POST"])
def process_image() -> Any:
    """Process image bytes directly in memory with pooled C++ OpenALPR engine."""
    try:
        image_file = request.files.get("image") or request.files.get("file")
        if not image_file:
            return (
                jsonify({"status": "error", "message": "No se proporcionó una imagen"}),
                400,
            )

        image_bytes = image_file.read()
        if not image_bytes:
            return (
                jsonify({"status": "error", "message": "Imagen vacía"}),
                400,
            )

        # Lease an engine instance from the pool
        engine = alpr_pool.get()
        try:
            # Recognize directly in memory from raw JPEG/PNG bytes
            alpr_result = engine.recognize_array(image_bytes)
        finally:
            # Always return instance to the pool
            alpr_pool.put(engine)

        rows, errors = filter_spanish_plates(alpr_result)

        response_data = {
            "status": "success",
            "command_output": rows,
            "command_error": errors,
        }
        return jsonify(response_data)

    except Exception as exc:
        logger.error(f"In-memory OpenALPR error: {exc}", exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/test_page/", methods=["GET"])
def test_page() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Prueba de API OpenALPR In-Memory</title></head>
    <body>
        <h1>Prueba de API OpenALPR In-Memory</h1>
        <p>Motor OpenALPR ultra-rápido ejecutándose directamente en memoria RAM (C++ Bindings).</p>
        <form action="/process_image/" method="post" enctype="multipart/form-data">
            <label for="image">Selecciona una imagen:</label><br>
            <input type="file" id="image" name="image"><br><br>
            <input type="submit" value="Procesar imagen">
        </form>
    </body>
    </html>
    """


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
