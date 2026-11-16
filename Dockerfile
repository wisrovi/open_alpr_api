# ==============================================================================
# Dockerfile for open_alpr_api:1.3.0 (LitServe + wyolo + GPU ALPR)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install minimal runtime dependencies in a single layer to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO weights and GPU ALPR models during build for instant startup
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && \
    python3 -c "from fast_alpr import ALPR; ALPR(detector_model='yolo-v9-t-384-license-plate-end2end', ocr_model='european-plates-mobile-vit-v2-model')" || true

# Copy application source code
COPY app/ ./app/
COPY main.py .

# Expose LitServe HTTP port
EXPOSE 8000

# Healthcheck to verify service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch LitServe API server
CMD ["python", "main.py"]
