# How to Add an RVC Model

I've set up all the infrastructure for RVC models. Here are your options:

## ✅ Quick Option: Create Placeholder Model

This creates a dummy model that loads without errors (for testing):

### In Docker (Recommended):
```bash
cd infra
docker compose run --rm worker python scripts/create_placeholder_rvc_model.py
```

### Or Locally (if Python is installed):
```bash
python scripts/create_placeholder_rvc_model.py
```

Or use the simple script:
```bash
python create_rvc_model.py
```

## 📥 Option 2: Download a Real Model

### From HuggingFace:
1. Visit: https://huggingface.co/models?search=rvc
2. Find a model (look for `.pth` files)
3. Download it
4. Place at: `models/rvc/pretrained.pth`

Or use the download script:
```bash
python scripts/download_rvc_model.py <model_url>
```

### Example HuggingFace Models:
- Search for "RVC" or "voice conversion"
- Look for models with `.pth` checkpoint files
- Download and place in `models/rvc/`

## 📁 Option 3: Use Your Own Model

If you have a trained RVC model file:

```bash
# Copy it to the right location
python scripts/download_rvc_model.py /path/to/your/model.pth

# Or manually:
cp /path/to/your/model.pth models/rvc/pretrained.pth
```

## ✅ Verify It Works

Test your setup:
```bash
python scripts/test_rvc_setup.py
```

Or in Docker:
```bash
cd infra
docker compose exec worker python scripts/test_rvc_setup.py
```

## 📍 Model Location

The model should be at:
```
models/rvc/pretrained.pth
```

This path is:
- ✅ Already created
- ✅ Mounted in Docker
- ✅ Configured in the app

## ⚠️ Important Note

Even with a real model file, RVC won't perform actual voice conversion until you implement the inference code in `processor/ml_refine/rvc_refiner.py`. The current code is a scaffold that loads the model but uses pass-through.

## 🚀 Next Steps After Adding Model

1. **Test loading**: Run `python scripts/test_rvc_setup.py`
2. **Implement inference**: Complete the code in `processor/ml_refine/rvc_refiner.py`
3. **Use in app**: Enable RVC checkbox in the web UI

## 📚 More Info

- See `models/rvc/README.md` for detailed setup
- See `docs/RVC_MODEL_SETUP.md` for complete guide
- See `README_RVC_MODEL.md` for quick reference

---

**Current Status**: Model directory is ready. Just add your model file!

