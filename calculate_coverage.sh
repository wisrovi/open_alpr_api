#!/usr/bin/env bash
# ==============================================================================
# Script to compute and report code coverage using pytest-cov
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " Running Unit Tests with Code Coverage"
echo "=========================================================="

pytest tests/ \
    --cov=app \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml \
    -v

echo "=========================================================="
echo " Coverage report generated in htmlcov/index.html"
echo "=========================================================="
