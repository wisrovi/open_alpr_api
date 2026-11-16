#!/usr/bin/env bash
# ==============================================================================
# Build and Push script for open_alpr_api and frontend v1.1.0
# ==============================================================================
set -euo pipefail

VERSION="1.3.0"
ALPR_IMAGE="wisrovi/open_alpr:${VERSION}"
ALPR_LATEST="wisrovi/open_alpr:latest"
API_IMAGE="wisrovi/open_alpr_api:${VERSION}"
API_LATEST="wisrovi/open_alpr_api:latest"
FRONTEND_IMAGE="wisrovi/open_alpr_frontend:${VERSION}"
FRONTEND_LATEST="wisrovi/open_alpr_frontend:latest"

echo "=========================================================="
echo " Building and Tagging Docker Images for v${VERSION}"
echo "=========================================================="

echo "[1/6] Building OpenALPR Backend image (${ALPR_IMAGE})..."
docker build -t "${ALPR_IMAGE}" -t "${ALPR_LATEST}" -f openalpr_backend/Dockerfile openalpr_backend/

echo "[2/6] Building API Docker image (${API_IMAGE})..."
docker build -t "${API_IMAGE}" -t "${API_LATEST}" -f Dockerfile .

echo "[3/6] Building Frontend Docker image (${FRONTEND_IMAGE})..."
docker build -t "${FRONTEND_IMAGE}" -t "${FRONTEND_LATEST}" -f gradio/Dockerfile gradio/

echo "[4/6] Pushing OpenALPR Backend image to Docker Hub..."
docker push "${ALPR_IMAGE}"
docker push "${ALPR_LATEST}"

echo "[5/6] Pushing API image to Docker Hub..."
docker push "${API_IMAGE}"
docker push "${API_LATEST}"

echo "[6/6] Pushing Frontend image to Docker Hub..."
docker push "${FRONTEND_IMAGE}"
docker push "${FRONTEND_LATEST}"

echo "=========================================================="
echo " Build & Push completed successfully!"
echo "=========================================================="
