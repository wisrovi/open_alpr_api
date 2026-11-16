<p align="center">
  <a href="https://linkedin.com/in/wisrovi-rodriguez"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://wisrovi.dev"><img src="https://img.shields.io/badge/Author-wisrovi.dev-111827?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portal" /></a>
  <a href="https://orcid.org/0009-0005-0710-1861"><img src="https://img.shields.io/badge/ORCID-0009--0005--0710--1861-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" /></a>
  <img src="https://img.shields.io/badge/version-1.1.0-blue.svg?style=for-the-badge" alt="Version 1.1.0" />
</p>

# open_alpr_api

> High-throughput, low-latency Automatic License Plate Recognition (ALPR) and Vehicle Detection API powered by **LitServe**, **wyolo** (TensorRT / FP16), accelerated SIMD decoding, ROI pre-cropping, and real-time WebSocket streaming.

---

## 👤 Author & Official Affiliation

* **William Steve Rodriguez Villamizar (Wisrovi)**
* **Title:** AI Leader & Solutions Architect at eCaptureDtech | Scientific Researcher
* 📧 **Email:** [wisrovi.rodriguez@gmail.com](mailto:wisrovi.rodriguez@gmail.com)
* 🌐 **Official Portal:** [wisrovi.dev](https://wisrovi.dev)
* 💼 **LinkedIn:** [wisrovi-rodriguez](https://www.linkedin.com/in/wisrovi-rodriguez/)
* 🆔 **ORCID:** [0009-0005-0710-1861](https://orcid.org/0009-0005-0710-1861)
* 📦 **PyPI:** [pypi.org/user/wisrovi/](https://pypi.org/user/wisrovi/)
* 🐙 **GitHub:** [@wisrovi](https://github.com/wisrovi)

---

## 🛠 Key Technologies and Libraries

The following core technologies and libraries drive the `open_alpr_api` platform:

* **[LitServe](https://github.com/Lightning-AI/LitServe)**: High-performance serving framework from Lightning AI designed for AI/ML inference workloads. Delivers native dynamic batching, multi-worker GPU process orchestration, and zero-copy streaming via FastAPI and uvloop.
* **[wyolo](https://pypi.org/project/wyolo/)**: Corporate enterprise vision library providing automated TensorRT compilation, FP16/INT8 engine optimization, latency profiling, and unified object detection.
* **[TorchVision SIMD IO](https://pytorch.org/vision/stable/io.html)**: High-throughput C++/SIMD image decoding pipeline (`torchvision.io.decode_image`), providing up to 4x faster image decode speeds than traditional OpenCV.
* **[In-Memory OpenALPR C++ Engine](https://github.com/openalpr/openalpr)**: Dedicated in-memory OCR backend using native precompiled C++ bindings (`openalpr.Alpr`) and an instance pool (`AlprPool`) for zero-disk-I/O high-throughput recognition.
* **[OpenCV Headless (opencv-python-headless)](https://pypi.org/project/opencv-python-headless/)**: High-performance computer vision library optimized for server containers without X11/GUI runtime dependencies.
* **[FastAPI & WebSockets](https://fastapi.tiangolo.com/)**: Asynchronous web server layer providing REST endpoints and bi-directional WebSocket streaming (`/ws/stream`) for RTSP camera feeds.
* **[Gradio](https://www.gradio.app/)**: Interactive web UI client decoupled from the ML engine for demonstration and rapid visual inspection.
* **[Requests & Urllib3](https://requests.readthedocs.io/)**: Resilient HTTP client with thread-safe connection pooling (`HTTPAdapter`), keep-alive persistence, and automated retry policies.
* **[Pytest & Pytest-Cov](https://docs.pytest.org/)**: Robust automated testing and coverage suite verifying pipeline concurrency, response schemas, and error boundaries.

---

## 🏛 Architecture Overview

In version **1.1.0**, the inference pipeline combines:

```
[ Client / Camera / RTSP Stream ]
         │
         ├──► HTTP POST /process_image/ ──┐
         │                                ▼
         └──► WS Stream /ws/stream ──► [ Accelerated SIMD Decoder (TorchVision/OpenCV) ]
                                                 │ (~3 ms)
                                                 ▼
                                       [ wyolo TensorRT FP16 ]
                                                 │ (~10 ms)
                                                 ▼
                                       [ Adaptive ROI Pre-Cropping Engine (<=680px) ]
                                                 │ (Crops and scales vehicle bounding box)
                                                 ▼
                                       [ In-Memory OpenALPR C++ Pool (RAM) ]
                                                 │ (~60-70 ms direct in memory)
                                                 ▼
                                       [ 100% Backward-Compatible JSON Response ]
```

1. **Adaptive ROI Pre-Cropping (`ENABLE_ROI_CROP=true`, `MAX_CROP_DIM=680`)**: `wyolo` detects vehicle coordinates in ~10ms, crops the vehicle with 15% contextual padding, and proportionally rescales oversized bounding boxes to a max dimension of 680px before OCR. This avoids searching through empty street background and drops OCR scanning latency by over **55%**.
2. **In-Memory OpenALPR C++ Bindings (`openalpr_backend`)**: Replaces slow disk writes (`tempfile.mkdtemp()`) and per-request OS sub-processes (`subprocess.run(["alpr"])`) with direct memory calls (`engine.recognize_array(bytes)`) across an in-memory instance pool (`AlprPool`), cutting OCR latency from ~260ms down to **~60-70ms**.
3. **Accelerated SIMD Image Decoding**: Uses SIMD-accelerated C++ decoding (`torchvision.io.decode_image`) in ~3.5ms compared to ~13ms in standard OpenCV.
4. **Real-Time WebSocket Streaming (`/ws/stream`)**: Direct bi-directional binary frame streaming for continuous video feeds and security cameras without HTTP connection teardown overhead.
5. **Zero Schema Drift**: 100% backward-compatible JSON schema returning exact `{"command_error": [], "command_output": [["PLATE", "CONF"]], "status": "success"}` on port `1037` and route `/process_image/`.

---

## 🚀 Quickstart & Deployment

### 1. Launch via Docker Compose (Recommended)

To run the complete ecosystem (OpenALPR backend + LitServe API + Gradio UI):

```bash
docker-compose up --build -d
```

* **LitServe API**: Accessible at `http://localhost:1037/process_image/`
* **WebSocket Stream**: Accessible at `ws://localhost:1037/ws/stream`
* **Gradio Web Interface**: Accessible at `http://localhost:7860`

### 2. Standalone CLI Client (`image.py`)

Send an image directly to the LitServe inference API:

```bash
# Basic detection (queries http://localhost:1037/process_image/ by default)
python image.py test/rojo_4.png

# Target custom endpoint
python image.py test/rojo_4.png --endpoint http://localhost:1037/process_image/
```

---

## 📡 API Reference

### 1. Endpoint: `POST /process_image/` (also aliased to `/process_image` and `/predict`)

#### Request (Multipart Form Data):
* `image`: Binary image file (JPEG / PNG).

#### Response (100% Backward-Compatible JSON):
```json
{
  "command_error": [],
  "command_output": [
    [
      "2000DFL",
      "91.582"
    ]
  ],
  "status": "success"
}
```

### 2. WebSocket Endpoint: `ws://localhost:1037/ws/stream`

Connect via WebSocket client, send binary JPEG frames sequentially, and receive structured JSON responses instantly per frame without HTTP framing overhead.

---

## 🧪 Testing and Code Coverage

### Run Unit Tests Locally:
```bash
pytest tests/ -v
```

### Calculate Test Coverage:
```bash
./calculate_coverage.sh
```
This generates a detailed terminal summary and an interactive HTML report under `htmlcov/index.html`.

### Run Unit Tests in Docker:
Per corporate quality assurance standards, all tests can be verified within an isolated Docker environment:
```bash
./run_tests_docker.sh
```

---

## 📦 Docker Images & Tags

* **API Engine**: `wisrovi/open_alpr_api:1.1.0` (and `wisrovi/open_alpr_api:latest`)
* **Web UI Frontend**: `wisrovi/open_alpr_frontend:1.1.0` (and `wisrovi/open_alpr_frontend:latest`)
* **Backend OCR**: `wisrovi/open_alpr:1.0`

To build and push images:
```bash
./scripts/build_and_push.sh
```

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
