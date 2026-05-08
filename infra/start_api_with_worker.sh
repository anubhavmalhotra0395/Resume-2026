#!/usr/bin/env bash
set -euo pipefail

# Run RQ worker in background for single-service free-tier deploys.
python -m processor.worker &

# Render injects PORT dynamically.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
