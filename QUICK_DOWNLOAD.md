# Quick Guide: Download the RVC Model

You provided this URL:
```
https://huggingface.co/AnonUser123/generic_rvc_voice/resolve/main/rvc_generic.pth
```

## ⚠️ Issue: Repository Requires Authentication

The repository appears to be private or requires login. Here are your options:

## Option 1: Download Manually (Easiest)

1. **Open the repository page**:
   - Go to: https://huggingface.co/AnonUser123/generic_rvc_voice
   - Log in to HuggingFace if needed

2. **Download the file**:
   - Click on `rvc_generic.pth`
   - Click "Download" button
   - Save the file

3. **Place it in the project**:
   ```bash
   # Copy the downloaded file to:
   models/rvc/pretrained.pth
   ```

   Or use PowerShell:
   ```powershell
   Copy-Item "C:\path\to\downloaded\rvc_generic.pth" "models\rvc\pretrained.pth"
   ```

## Option 2: Use HuggingFace CLI

If you have HuggingFace account:

```bash
# Login first
huggingface-cli login

# Then download
python download_model_with_auth.py
```

Or in Docker:
```bash
docker compose -f infra/docker-compose.yml exec worker huggingface-cli login
docker compose -f infra/docker-compose.yml exec worker python /app/download_model_with_auth.py
```

## Option 3: Direct Download Script

If you're logged in to HuggingFace in your browser, you can try:

```bash
# In Docker
docker compose -f infra/docker-compose.yml exec worker python download_model_with_auth.py
```

## Verify the Model

Once the file is in place, test it:

```bash
python scripts/test_rvc_setup.py
```

Or check manually:
```bash
# Check if file exists
ls -lh models/rvc/pretrained.pth

# Should show file size (RVC models are usually 50-500 MB)
```

## Expected File Location

```
models/rvc/pretrained.pth
```

The directory structure is already set up - just place the file there!

