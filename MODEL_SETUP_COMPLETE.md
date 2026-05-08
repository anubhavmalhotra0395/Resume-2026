# Model Setup Guide - Complete

## Folder Structure Created ✅

```
models/
  rvc/
    pretrained.pth    # RVC generator model (required)
    vocoder.pth       # NSF-HifiGAN vocoder (optional)
  hubert/
    hubert-base-ls960.pt  # HuBERT encoder (auto-downloaded if missing)
```

## Paths Configured ✅

The system now uses these paths:

- **HuBERT**: `models/hubert/hubert-base-ls960.pt`
- **RVC Model**: `models/rvc/pretrained.pth`
- **Vocoder**: `models/rvc/vocoder.pth`

## Auto-Download Behavior

### HuBERT
- **First use**: Downloads from HuggingFace automatically
- **Saves locally**: Saves to `models/hubert/hubert-base-ls960.pt` for future use
- **Manual download**: Run `python scripts/download_hubert.py`

### RVC Model
- **Required**: Must be placed manually at `models/rvc/pretrained.pth`
- **No auto-download**: You need to provide your trained model

### Vocoder
- **Optional**: Place at `models/rvc/vocoder.pth` if available
- **Fallback**: Uses Griffin-Lim if vocoder not found

## Setup Steps

### Step 1: Download HuBERT (Automatic)

The first time you use RVC, HuBERT will download automatically. Or download manually:

```bash
python scripts/download_hubert.py
```

### Step 2: Add RVC Model

Place your trained RVC model at:
```
models/rvc/pretrained.pth
```

### Step 3: Add Vocoder (Optional)

Place vocoder at:
```
models/rvc/vocoder.pth
```

### Step 4: Verify Setup

Check model status:
```bash
curl http://localhost:8000/health/models
```

Expected response:
```json
{
  "hubert": true,
  "hubert_file": true,
  "rvc_model": true,
  "rvc_file": true,
  "vocoder": true,
  "vocoder_file": true,
  "device": "cpu",
  "models_loaded": true,
  "msg": "All models available and loaded"
}
```

## Health Check Response Fields

- **hubert**: Model loaded in memory
- **hubert_file**: File exists on disk
- **rvc_model**: Model loaded in memory
- **rvc_file**: File exists on disk
- **vocoder**: Model loaded in memory
- **vocoder_file**: File exists on disk
- **device**: Processing device (cpu/cuda/mps)
- **models_loaded**: All critical models loaded
- **msg**: Human-readable status
- **paths**: File paths being used

## Troubleshooting

### "hubert": false
- **Solution**: HuBERT will auto-download on first use
- **Or**: Run `python scripts/download_hubert.py`

### "rvc_model": false but "rvc_file": true
- **Issue**: Model file exists but architecture doesn't match
- **Solution**: Adapt `_create_rvc_generator()` in `rvc_refiner.py` to your model format

### "vocoder": false
- **Not critical**: System uses Griffin-Lim fallback
- **To fix**: Place vocoder at `models/rvc/vocoder.pth`

## File Sizes

- **HuBERT**: ~300 MB (auto-downloaded)
- **RVC Model**: Varies (50-500 MB typically)
- **Vocoder**: Varies (20-100 MB typically)

## Next Steps

1. ✅ Folders created
2. ✅ Paths configured
3. ⏳ Download/add models
4. ⏳ Test with health check
5. ⏳ Process your first job!

---

**Status**: Infrastructure ready. Add your RVC model to enable full voice conversion! 🚀

