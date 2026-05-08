# Model Setup - Quick Reference

## ✅ Folders Created

```
models/
  rvc/
    pretrained.pth    ← Place your RVC model here
    vocoder.pth       ← Place vocoder here (optional)
  hubert/
    hubert-base-ls960.pt  ← Auto-downloaded on first use
```

## Paths Configured ✅

All paths are now set correctly:

- **HuBERT**: `models/hubert/hubert-base-ls960.pt`
- **RVC Model**: `models/rvc/pretrained.pth`  
- **Vocoder**: `models/rvc/vocoder.pth`

## Quick Setup

### 1. Download HuBERT (Automatic)

HuBERT will auto-download on first use. Or download manually:

```bash
python scripts/download_hubert.py
```

### 2. Add RVC Model

Place your trained RVC model at:
```
models/rvc/pretrained.pth
```

### 3. Add Vocoder (Optional)

Place vocoder at:
```
models/rvc/vocoder.pth
```

### 4. Verify

```bash
curl http://localhost:8000/health/models
```

Expected:
```json
{
  "hubert": true,
  "rvc_model": true,
  "vocoder": true,
  "models_loaded": true
}
```

## What Changed

✅ **Folders created**: `models/rvc/` and `models/hubert/`  
✅ **Paths updated**: All model paths point to correct locations  
✅ **Auto-download**: HuBERT downloads automatically if missing  
✅ **Health check**: Enhanced to show file existence and loading status  

## Next Steps

1. Add your RVC model file to `models/rvc/pretrained.pth`
2. (Optional) Add vocoder to `models/rvc/vocoder.pth`
3. Test: `curl http://localhost:8000/health/models`
4. Use RVC in your jobs!

---

**Status**: ✅ **Ready** - Just add your RVC model file!

