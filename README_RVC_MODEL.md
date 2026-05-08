# Quick Guide: Adding an RVC Model

## Option 1: Create Placeholder Model (Quick Test)

Run this inside Docker:
```bash
docker compose exec worker python scripts/create_placeholder_rvc_model.py
```

Or locally (if Python is installed):
```bash
python scripts/create_placeholder_rvc_model.py
```

This creates a dummy model that loads without errors (but won't do real voice conversion).

## Option 2: Download a Real Model

### From HuggingFace:
1. Visit https://huggingface.co/models?search=rvc
2. Find a model you want
3. Download the `.pth` file
4. Place it at: `models/rvc/pretrained.pth`

Or use the script:
```bash
python scripts/download_rvc_model.py <model_url>
```

### From RVC-Project:
1. Visit https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
2. Follow their model download instructions
3. Place the model at: `models/rvc/pretrained.pth`

## Option 3: Use Your Own Trained Model

If you have a trained RVC model:
```bash
python scripts/download_rvc_model.py /path/to/your/model.pth
```

Or manually copy:
```bash
cp /path/to/your/model.pth models/rvc/pretrained.pth
```

## Verify Setup

Test if your model is set up correctly:
```bash
python scripts/test_rvc_setup.py
```

Or in Docker:
```bash
docker compose exec worker python scripts/test_rvc_setup.py
```

## Important Notes

⚠️ **Even with a real model file, RVC won't work until you implement the inference code.**

The current implementation is a scaffold. You need to:
1. Implement model loading in `processor/ml_refine/rvc_refiner.py` → `_load_rvc_model_pytorch()`
2. Implement inference in `processor/ml_refine/rvc_refiner.py` → `_apply_rvc_pytorch()`

See `docs/RVC_MODEL_SETUP.md` for complete details.

