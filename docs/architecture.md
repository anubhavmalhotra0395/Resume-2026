Architecture Overview
=====================

Goals
-----
- DSP-first vocal style transfer: copy mix style (EQ/dynamics/space), not voice.
- Predictable, explainable, cheap to run; ML can be added later.

Main pieces
-----------
- Frontend: upload reference + dry vocal, show progress, download result.
- API: receives uploads, validates (size/duration/MIME), enqueues jobs, exposes status, serves static UI.
- Queue: Redis + RQ (swap for Celery if you prefer).
- Worker: Python DSP pipeline, reads inputs, writes output, sweeps old files.
- Storage: local `storage/` here; swap for S3/MinIO in prod.

Processing pipeline
-------------------
1) Normalize with FFmpeg (mono, 44.1kHz, loudnorm).
2) Analyze reference with librosa:
   - Spectral tilt/bands → EQ recipe.
   - RMS variance → compression threshold/ratio/attack/release heuristics.
   - Energy decay (RT60-ish) → reverb decay/mix; harmonic emphasis → saturation hint.
3) Apply DSP chain (SciPy):
   EQ → compression → reverb (FFT convolution) → soft saturation → limiter/normalize.
4) Save WAV output; update job status.

Queues & storage
----------------
- `processor/config.py` holds Redis URL, storage paths, retention settings.
- `processor/worker.py` registers the RQ worker and job handler.
- `processor/cleanup.py` TTL sweeper (runs after jobs; can be cron’d).

Testing
-------
- `tests/` include sanity checks for EQ/comp transfer functions and the pipeline.

Production checklist
--------------------
- Env-driven settings (`APP_*`) for storage, Redis, CORS.
- Dockerized API + worker + Redis (`infra/docker-compose.yml`).
- Static UI bundled with API; CORS configurable for external hosting.
- Validation with ffprobe (size/duration) + TTL cleanup.
- Replace `storage/` with S3/MinIO and signed URLs when deploying multi-host.

