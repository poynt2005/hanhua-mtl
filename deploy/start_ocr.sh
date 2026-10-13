#!/usr/bin/env bash
set -euo pipefail

# Manga OCR server for vast.ai: comic-text-detector (boxes) + manga-ocr (recognition)
# usage: OCR_TOKEN=secret bash start_ocr.sh
#        DEVICE=cuda PORT=8080 bash start_ocr.sh

PORT="${PORT:-8080}"
DEVICE="${DEVICE:-cuda}"
export PORT DEVICE

pip install -q fastapi "uvicorn[standard]" pillow numpy manga-ocr comic-text-detector

echo "starting OCR server on 0.0.0.0:${PORT} (device=${DEVICE})"
exec python ocr_server.py
