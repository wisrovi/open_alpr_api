#!/usr/bin/env bash
# ==============================================================================
# Script to execute unit tests inside an isolated Docker container
# ==============================================================================
set -euo pipefail

IMAGE_NAME="open_alpr_api_test_runner:latest"

echo "=========================================================="
echo " Building Test Runner Docker Image..."
echo "=========================================================="

docker build -t "${IMAGE_NAME}" -f Dockerfile .

echo "=========================================================="
echo " Executing Pytest inside Docker Container..."
echo "=========================================================="

docker run --rm \
    -v "$(pwd)/tests:/app/tests:ro" \
    "${IMAGE_NAME}" \
    pytest tests/ -v
