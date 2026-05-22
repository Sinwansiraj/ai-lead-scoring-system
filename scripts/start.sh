#!/bin/sh
# Entrypoint: train model then start the API.
# Used by the Render deployment (and any Docker run without a CMD override).
set -e

echo "==> Training model..."
python scripts/train.py

echo "==> Starting API on port ${PORT:-8000}..."
exec python -m uvicorn lead_scoring.api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1
