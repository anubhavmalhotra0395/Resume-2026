# Quick Start Guide

## Option 1: Docker (Recommended - Easiest)

### Prerequisites
- Docker Desktop installed and running
- Docker Compose (usually included with Docker Desktop)

### Steps

1. **Navigate to the infra directory**:
   ```powershell
   cd infra
   ```

2. **Start all services**:
   ```powershell
   docker compose up --build
   ```

   This will:
   - Build the API and worker containers
   - Start Redis
   - Start the API server (port 8000)
   - Start the worker

3. **Access the web UI**:
   - Open your browser: http://localhost:8000/ui
   - Or visit: http://localhost:8000

4. **Check health**:
   ```powershell
   curl http://localhost:8000/healthz
   ```

### Stop the services
Press `Ctrl+C` in the terminal, or:
```powershell
docker compose down
```

---

## Option 2: Local Development

### Prerequisites
- Python 3.11+
- Redis running (port 6379)
- FFmpeg installed and on PATH

### Steps

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Start Redis** (if not running):
   ```powershell
   # Option A: Using Docker
   docker run -d -p 6379:6379 redis:7

   # Option B: Install Redis locally and run
   redis-server
   ```

3. **Start the API server** (in one terminal):
   ```powershell
   uvicorn api.main:app --reload --port 8000
   ```

4. **Start the worker** (in another terminal):
   ```powershell
   python -m processor.worker
   ```

5. **Access the web UI**:
   - Open: http://localhost:8000/ui

---

## First Steps After Starting

### 1. Test the Setup

Run the smoke test:
```powershell
python scripts/test_rvc_setup.py
```

### 2. Check Model Status

```powershell
curl http://localhost:8000/health/models
```

### 3. Use the Web UI

1. Go to http://localhost:8000/ui
2. Upload a reference track (or paste YouTube/Spotify link)
3. Upload your dry vocal
4. (Optional) Enable "AI Voice Style Transfer (RVC)" if you have a model
5. Click "Start Job"
6. Wait for processing
7. Download the result

---

## Troubleshooting

### Docker Issues

**"docker compose not found"**:
- Use `docker compose` (v2) instead of `docker-compose` (v1)
- Or install Docker Desktop

**"Port already in use"**:
- Stop other services using port 8000 or 6379
- Or change ports in `docker-compose.yml`

**"Redis connection error"**:
- Wait a few seconds for Redis to start
- Check Redis is healthy: `docker compose ps`

### Local Development Issues

**"Redis connection error"**:
- Make sure Redis is running: `redis-cli ping`
- Check `APP_REDIS_URL` in environment or config

**"FFmpeg not found"**:
- Install FFmpeg and add to PATH
- Or use Docker instead

**"Module not found"**:
- Install dependencies: `pip install -r requirements.txt`

---

## What's Running?

After starting, you should have:

- ✅ **API Server** - http://localhost:8000
- ✅ **Web UI** - http://localhost:8000/ui
- ✅ **Worker** - Processing jobs in background
- ✅ **Redis** - Job queue (port 6379)

---

## Next Steps

1. **Add RVC Model** (optional):
   - Place model at `models/rvc/pretrained.pth`
   - See `ADD_RVC_MODEL.md` for details

2. **Test the API**:
   ```powershell
   python scripts/test_api_job.py
   ```

3. **View Metrics**:
   ```powershell
   curl http://localhost:8000/metrics
   ```

---

## Quick Commands Reference

```powershell
# Start everything (Docker)
cd infra
docker compose up --build

# Stop everything (Docker)
docker compose down

# View logs
docker compose logs -f

# Restart worker only
docker compose restart worker

# Check status
docker compose ps

# Health check
curl http://localhost:8000/healthz

# Model status
curl http://localhost:8000/health/models
```

---

**Ready to go!** 🚀

Visit http://localhost:8000/ui to start processing vocals.

