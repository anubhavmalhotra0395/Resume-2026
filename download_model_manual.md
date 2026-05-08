# Manual Model Download Instructions

The HuggingFace repository `AnonUser123/generic_rvc_voice` appears to be:
- Private (requires authentication)
- Or doesn't exist
- Or requires login

## Option 1: Download Manually from HuggingFace

1. **Visit the repository page**:
   - Go to: https://huggingface.co/AnonUser123/generic_rvc_voice
   - You may need to log in to HuggingFace

2. **Download the model file**:
   - Click on `rvc_generic.pth` file
   - Click "Download" button
   - Save the file

3. **Place it in the correct location**:
   ```bash
   # Copy the downloaded file to:
   models/rvc/pretrained.pth
   ```

## Option 2: Use a Different Public Model

If this repository is not accessible, try these public RVC models:

1. **Search for public RVC models**:
   - Visit: https://huggingface.co/models?search=rvc
   - Filter by: "Public" models
   - Look for models with `.pth` files

2. **Download and place**:
   ```bash
   # After downloading, place at:
   models/rvc/pretrained.pth
   ```

## Option 3: Use HuggingFace CLI (if authenticated)

If you have HuggingFace account:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login (if needed)
huggingface-cli login

# Download using Python
python download_hf_model.py
```

## Option 4: Create Placeholder Model

For testing purposes, create a placeholder:

```bash
docker compose -f infra/docker-compose.yml exec worker python scripts/create_placeholder_rvc_model.py
```

## Current Status

The model file should be at:
```
models/rvc/pretrained.pth
```

Once you have the file in place, test it:
```bash
python scripts/test_rvc_setup.py
```

