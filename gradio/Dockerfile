# ==============================================================================
# Dockerfile for alpr_frontend:1.1.0 (Lightweight Gradio Web UI)
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Minimal system runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight web frontend requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy frontend source code
COPY api.py .

# Expose Gradio port
EXPOSE 7860

# Launch Gradio interface
CMD ["python", "api.py"]
