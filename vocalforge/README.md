# VocalForge — standalone site

VocalForge as its own site, separate from the 3D room portfolio
(which keeps its embedded, access-gated copy untouched).

Single static file (`index.html`), talks to the existing FastAPI backend.

## Features (full parity with the API)

- Reference track + dry vocal upload (drag & drop or picker), or reference URL
- Optional vocal-layer analysis (`POST /analyze-layers`) with selectable layers
- AI Tuning: plain-English prompt → DSP config via `POST /api/ai-audio`
  (Groq · Llama 3.3, falls back safely) — config rides along with the job
- Full effect chain toggles mirroring the API defaults (width, spectral
  refiner, de-esser, harmony, chorus/flanger, delay, gate, doubler, exciter,
  tape, parallel comp, M/S EQ, transient shaper, multiband, adaptive DSP,
  ML refine/RVC)
- Job submission (`POST /jobs`), live progress via SSE
  (`GET /jobs/{id}/stream`) with polling fallback (`GET /jobs/{id}`)
- Result playback + WAV download (`/outputs/…`) + recipe JSON (`/recipes/…`)
- API health badge (`/healthz`)

## Run

Easiest — served by the API itself (mounted at `/vocalforge`):

    uvicorn api.main:app --reload
    # open http://localhost:8000/vocalforge/

(Requires Redis + worker for actual processing — see the repo root README.)

Or serve statically from anywhere:

    npx serve vocalforge -l 5000
    # open http://localhost:5000/?api=http://localhost:8000

The `?api=` query param (or the ⚙ API button) sets the backend base URL,
persisted in localStorage. CORS is already open on the API.
