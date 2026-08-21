Reference-Based Vocal Style Transfer (DSP-first)
================================================
This repo is a minimal, transparent implementation of the guide you described:
upload a reference track + a dry vocal, analyze the reference for mix “style,”
and apply a similar EQ/comp/reverb/saturation chain to the dry vocal.

What’s here
-----------
- `frontend/` — zero-build static UI for upload, polling, and download.
- `api/` — FastAPI service for uploads, validation, status, download, and static hosting.
- `processor/` — Python DSP pipeline (analysis + processing) and worker entrypoint.
- `docs/` — Architecture notes and pipeline overview.
- `infra/` — Dockerfiles + docker-compose for API, worker, Redis.
- `tests/` — A couple of sanity tests for the DSP blocks.

The stack is torch-free: vocal separation runs UVR's Kim_Vocal_2 MDX-Net
model directly through onnxruntime (~67 MB, auto-downloaded on first use).

Quickstart (local)
------------------
1) Install deps (Python 3.10+): `pip install -r requirements.txt`
2) Run Redis for the queue (e.g., `docker run -p 6379:6379 redis:7`)
3) Start API: `uvicorn api.main:app --reload`
4) Start worker: `python -m processor.worker`

Production-style (Docker)
-------------------------
1) `cd infra`
2) `docker-compose up --build`
3) API/UI at `http://localhost:8000` (uploads + polling). Redis exposed at 6379.
   Storage is volume-mounted to `infra/storage/`.

API sketch
----------
- `POST /jobs` — multipart form with `reference` **or** `reference_url`, plus `dry`; returns `job_id`.
- `GET /jobs/{job_id}` — status + download URL (if done).
- `GET /outputs/{job_id}.wav` — download.
- `GET /healthz`

Processing flow
---------------
1) FFmpeg normalizes to mono/44.1kHz WAV and LUFS-safe level.
2) Analysis (`processor.analysis.style_extractor`) derives a recipe:
   EQ curve, compression settings, reverb decay/mix, saturation hint.
3) DSP chain (`processor.dsp.chain`) applies EQ → compression → reverb
   → saturation → limiter/normalize.
4) Output WAV is written to `storage/outputs/{job_id}.wav`.

Notes
-----
- Pure DSP, no black-box ML. SciPy for effects, librosa for analysis.
- Replace storage/queue with your infra (S3/MinIO, Celery/RQ, etc.) if needed.
- Files are TTL’d (`delete_after_hours` in `processor/config.py`), and uploads are size/duration validated via ffprobe.
- CORS origins can be set with `APP_CORS_ORIGINS` (comma-separated) for a hosted frontend.


