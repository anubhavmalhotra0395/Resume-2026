# Free Deployment Guide (Render + Upstash Redis)

This project has:
- FastAPI web service (`api.main:app`)
- background worker (`processor.worker`)
- Redis queue (RQ)

The easiest free setup is:
- **Render** for API + worker
- **Upstash Redis** free tier for queue

## 1) Push code to GitHub

Render deploys from GitHub, so first push this repo to GitHub.

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
4. Render detects `render.yaml` and creates:
   - `doctasky-api` (web service)
   - `doctasky-worker` (worker service)

## 4) Set required environment variables (both services)

In Render dashboard, set:

- `APP_REDIS_URL` = your Upstash redis URL
- `GROQ_API_KEY` = your key (if AI endpoints are used)

Already provided in `render.yaml`:
- `APP_STORAGE_ROOT=/tmp/storage`
- `APP_DELETE_AFTER_HOURS=24`
- `APP_CORS_ORIGINS=*`

## 5) Open the app

After deploy, open:

- `https://<your-api-service>.onrender.com/ui`

The API root is:
- `https://<your-api-service>.onrender.com/`

## Important Free-Tier Notes

- Free services can sleep when idle (cold start delay).
- This stack is ML/audio-heavy (`torch`, `demucs`, etc.), so free tier may be slow or hit memory limits for large jobs.
- `/tmp/storage` is ephemeral; files can disappear after restarts.

If you want, next step I can also add a **light mode** toggle to skip heavy processors for more reliable free-tier runs.
