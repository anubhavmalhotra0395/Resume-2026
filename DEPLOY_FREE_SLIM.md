# Deploying VocalForge online for free (slim build)

The original image is ~14 GB because it installs CUDA torch + the optional
RVC stack. The slim build (`infra/Dockerfile.slim`) uses CPU-only torch and
skips RVC extras → **~2.5–3 GB**, small enough for free hosts. Separation
models (Kim_Vocal_2, demucs weights) download on first use into `/data`,
so they never bloat the image.

## What the slim build contains

- `requirements-core.txt` — everything except RVC (CPU torch via the
  `download.pytorch.org/whl/cpu` index; no `nvidia-*` packages)
- `requirements-rvc.txt` — optional extras (torchaudio, transformers,
  pyworld, crepe) only if you want `ml_refine=true`
- `infra/start_all_in_one.sh` — Redis + RQ worker + API in one container
- Serves the frontends too: `/vocalforge/` (standalone) and `/ui/` (room)

Build & run locally:

    docker build -f infra/Dockerfile.slim -t vocalforge-slim .
    docker run -p 7860:7860 -e GROQ_API_KEY=... -v vf-data:/data vocalforge-slim
    # http://localhost:7860/vocalforge/

## Option A — Hugging Face Spaces (recommended)

Free tier: 2 vCPU, 16 GB RAM, 50 GB ephemeral disk — enough for CPU demucs.

1. Create a **Docker Space** at huggingface.co/new-space.
2. Push this repo to the Space (or mirror from GitHub). Add a top-level
   `Dockerfile` containing just:

       FROM scratch
       # (not used — point the Space at infra/Dockerfile.slim instead)

   …or simpler: copy `infra/Dockerfile.slim` to the repo root as
   `Dockerfile` (Spaces builds the root Dockerfile; the COPY paths
   already match the repo layout).
3. In Space settings → Variables, add `GROQ_API_KEY` (secret).
4. Space listens on port 7860 — already the default in the image.
5. App URL: `https://<user>-<space>.hf.space/vocalforge/`

Notes: free Spaces sleep after ~48 h idle and the disk is ephemeral
(models re-download after a rebuild — ~1–2 min, cached in /data between
requests while alive).

## Option B — Oracle Cloud Always Free (most powerful, forever)

Always-free ARM VM: up to 4 OCPU / 24 GB RAM. Runs the FULL compose stack
unchanged (there's already an `infra/oracle/` folder started).

1. Create an Ampere A1 instance (Ubuntu), open port 80/443.
2. Install docker + compose, clone the repo.
3. `docker compose -f infra/docker-compose.yml up -d --build`
   (on ARM the CPU torch wheels install fine; build takes a while once).
4. Put Caddy or nginx in front for HTTPS.

This is the only free option with a persistent disk and no sleep.

## Option C — Render.com free web service

Works with `infra/Dockerfile.slim` as-is (`start_all_in_one.sh` already
reads `$PORT`). Free tier sleeps after 15 min idle and has 512 MB RAM —
enough for the API + AI tuning, but **too small for demucs separation**;
set `APP_ENABLE_WORKER=0` and treat it as a demo/API-only deploy.

## Frontends

Keep the room + VocalForge on Vercel (free, instant) and point them at the
API host with `?api=https://your-api-host` — the standalone UI persists it.
Same-origin also works since the API serves `/vocalforge/` and `/ui/` itself.

## Cost-free checklist

- [ ] `GROQ_API_KEY` set (AI tuning; falls back gracefully without it)
- [ ] First job after a cold start is slow (model download) — expected
- [ ] Keep `ml_refine` off unless you installed `requirements-rvc.txt`
