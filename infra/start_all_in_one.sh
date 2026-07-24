#!/usr/bin/env bash
set -euo pipefail

# All-in-one boot: Redis + RQ worker + API in a single container.
# Suits free hosts (Hugging Face Spaces, Oracle free VM, Render).

mkdir -p "${APP_STORAGE_ROOT:-/data}"

# 1) Redis (local, ephemeral — queue state only)
redis-server --daemonize yes --save "" --appendonly no --port 6379

# 2) Worker (skippable with APP_ENABLE_WORKER=0 on tiny instances)
if [ "${APP_ENABLE_WORKER:-1}" = "1" ]; then
  python -m processor.worker &
fi

# 3) API (HF Spaces injects PORT=7860)
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
