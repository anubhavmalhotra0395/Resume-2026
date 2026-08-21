# Deploying Doctavox online for free

The stack is now **torch-free**: separation runs Kim_Vocal_2.onnx directly
via onnxruntime (`processor/utils/mdx_onnx.py`), and the RVC / neural-refiner
extras were removed. The image (`infra/Dockerfile.slim`) is ~1 GB (was
~2.5-3 GB with CPU torch, 14 GB with CUDA), and RAM needs dropped to
~1-2 GB per job. The ~67 MB separation model downloads on first use into
`/data` so it never bloats the image.

Build & run locally:

    docker build -f infra/Dockerfile.slim -t doctavox .
    docker run -p 7860:7860 -e GROQ_API_KEY=... -v dv-data:/data doctavox
    # http://localhost:7860/vocalforge/

Low-memory hosts: set `APP_LOW_MEMORY=1` to replace ML separation with a
bandpass fallback (lower quality, but survives 512 MB).

## Option A — Hugging Face Spaces (NO LONGER FREE, as of 2026-08)

**Docker Spaces now require a PRO subscription ($9/mo).** Creating one on
free `cpu-basic` returns HTTP 402: *"Static Spaces are free for everyone,
but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
subscription."* Only static sites remain free. The steps below still apply
if you subscribe — otherwise use Option B (Oracle), which is the only
remaining free host with enough RAM for demucs.

Paid tier: 2 vCPU, 16 GB RAM, 50 GB ephemeral disk — enough for CPU demucs.

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

## Option B — Oracle Cloud Always Free (recommended; the only free option left)

Always-free ARM VM: up to 4 OCPU / 24 GB RAM, persistent disk, no sleep.

Fully scripted — see **`infra/oracle/`** (compose stack + Caddy/HTTPS +
`bootstrap.sh`). Short version:

    ssh ubuntu@<PUBLIC_IP>
    git clone --depth 1 <this repo> ~/doctavox
    GROQ_API_KEY=... bash ~/doctavox/infra/oracle/bootstrap.sh

`bootstrap.sh` installs Docker, opens the host firewall, builds, and gets a
Let's Encrypt cert for a `<dashed-ip>.sslip.io` hostname (no domain needed).

**The catch:** free Ampere capacity is scarce and region-locked to your
tenancy's *home* region — "Out of host capacity" is routine and can persist
for hours or days. The fix is a patient retry loop, not a config change.
Always Free resources cannot be created outside the home region.

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
