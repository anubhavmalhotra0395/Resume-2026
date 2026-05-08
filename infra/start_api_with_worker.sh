#!/usr/bin/env bash
set -euo pipefail

# Optional worker in same container (disabled by default for free memory limits).
if [ "${APP_ENABLE_WORKER:-0}" = "1" ]; then
  python -m processor.worker &
fi

# Render injects PORT dynamically.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
