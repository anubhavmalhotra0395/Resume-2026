# Free Deployment Guide (Render + Upstash Redis)

## Render pipeline minutes exhausted?

Render’s **Dockerfile builds** (torch, demucs, …) burn **pipeline minutes** fast. Use this instead:

1. **GitHub Actions** (this repo) builds the image and pushes to **GHCR** — see [`.github/workflows/publish-api-image.yml`](.github/workflows/publish-api-image.yml) (free for public repos; image is **linux/amd64** as required by [Render prebuilt images](https://render.com/docs/deploy-an-image)).
2. **`render.yaml`** uses **`runtime: image`** so Render **pulls** the image instead of building it on Render (saves build pipeline time).
3. Edit `render.yaml` → `image.url` to your real package, e.g. `ghcr.io/myuser/myrepo:latest` (must be **lowercase**; match the package name after the first Actions run).
4. In GitHub → **Packages** → set the container package to **Public** (simplest), or add **GHCR credentials** in Render and `image.creds` per Render docs.
5. **Redeploy:** Prebuilt image services do not always pick up a new `:latest` automatically. In Render: **Manual Deploy → Deploy latest reference**, or add a **Deploy Hook** (service **Settings**) and store its URL as GitHub secret **`RENDER_DEPLOY_HOOK_URL`** — the publish workflow will `curl` it after each successful push.

If you already had a **Dockerfile-based** Render service, you usually must **create a new web service** from the updated Blueprint (runtime is fixed at creation).

---

This project has:
- FastAPI web service (`api.main:app`)
- background worker (`processor.worker`)
- Redis queue (RQ)

The easiest free setup is:
- **Render** for the API (prebuilt image from GHCR — see top of this file)
- **Upstash Redis** free tier for the queue
- **Vercel** (or static host) for the frontend — the API image does not include `frontend/`, so use your deployed site URL for the UI.

## 1) Push code to GitHub

Render reads `render.yaml` from GitHub. **Run GitHub Actions “Publish API image” once** so GHCR has an image, then set `image.url` in `render.yaml` to match.

## 2) Create free Redis in Upstash

1. Go to [https://upstash.com](https://upstash.com)
2. Create a Redis database (free tier)
3. Copy the Redis connection URL (starts with `redis://...` or `rediss://...`)

Save it as:
- `APP_REDIS_URL`

## 3) Deploy on Render with Blueprint

1. Go to [https://render.com](https://render.com)
2. New + → **Blueprint**
3. Select your GitHub repo
4. Render detects `render.yaml` and creates **one** service: `doctasky-api` (web).

## 4) Set required environment variables

In Render dashboard for **doctasky-api**, set:

- `APP_REDIS_URL` = your Upstash redis URL
- `GROQ_API_KEY` = your key (if AI endpoints are used)

Already provided in `render.yaml`:
- `APP_STORAGE_ROOT=/tmp/storage`
- `APP_DELETE_AFTER_HOURS=24`
- `APP_CORS_ORIGINS=*` (tighten to your Vercel URL in production)

Background jobs: `render.yaml` sets `APP_ENABLE_WORKER=0` on the free web instance (saves RAM). For a separate worker service you’d add another Render service later.

## 5) Open the app

API base URL:

- `https://<your-api-service>.onrender.com/`

Health check:

- `https://<your-api-service>.onrender.com/healthz`

Use your **Vercel** URL for the UI; point the frontend’s API base URL at the Render URL above.

## Important Free-Tier Notes

- Free services can sleep when idle (cold start delay).
- This stack is ML/audio-heavy (`torch`, `demucs`, etc.), so free tier may be slow or hit memory limits for large jobs.
- `/tmp/storage` is ephemeral; files can disappear after restarts.

If you want, next step I can also add a **light mode** toggle to skip heavy processors for more reliable free-tier runs.
