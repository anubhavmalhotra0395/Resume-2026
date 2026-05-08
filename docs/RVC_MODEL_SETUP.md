# RVC Model Setup Guide

## Overview

RVC (Retrieval Voice Conversion) support has been added to the vocal style transfer app. This guide explains how to set up and use an RVC model.

## What Was Added

1. **Model Directory Structure**: `models/rvc/` directory created
2. **Model Loading Infrastructure**: Code to load PyTorch (`.pth`) or ONNX (`.onnx`) models
3. **Helper Scripts**:
   - `scripts/download_rvc_model.py` - Download or copy RVC models
   - `scripts/test_rvc_setup.py` - Test if your model is set up correctly
4. **Docker Support**: Models directory mounted in docker-compose
5. **Documentation**: `models/rvc/README.md` with detailed instructions

## Current Status

⚠️ **Important**: The RVC implementation is currently a **scaffold**. This means:

- ✅ Model file detection and loading infrastructure
- ✅ F0 extraction (PyWorld/CREPE)
- ✅ Content feature extraction (placeholder)
- ⚠️ **Model architecture loading** - needs implementation
- ⚠️ **Inference code** - needs implementation

Even with a model file, RVC will currently use **pass-through** (no voice conversion) until the inference code is implemented.

## Quick Start

### Step 1: Get an RVC Model

You have three options:

#### Option A: Download from HuggingFace
```bash
# Visit https://huggingface.co/models?search=rvc
# Download a .pth or .onnx model
# Then:
python scripts/download_rvc_model.py <model_url>
```

#### Option B: Use Your Own Model
```bash
# If you have a trained model file:
python scripts/download_rvc_model.py /path/to/your/model.pth
```

#### Option C: Manual Setup
```bash
# Just place your model file at:
models/rvc/pretrained.pth
# or
models/rvc/pretrained.onnx
```

### Step 2: Test the Setup

```bash
python scripts/test_rvc_setup.py
```

This will verify:
- Model file exists
- Model can be loaded
- Basic structure is correct

### Step 3: Implement Inference Code

To make RVC actually work, you need to implement:

1. **Model Architecture Loading** (`processor/ml_refine/rvc_refiner.py`):
   - Function: `_load_rvc_model_pytorch()`
   - Adapt to your RVC framework's model structure
   - Example frameworks: RVC-Project/RVC, so-vits-svc, etc.

2. **Inference Code** (`processor/ml_refine/rvc_refiner.py`):
   - Function: `_apply_rvc_pytorch()` or `_apply_rvc_onnx()`
   - Implement the actual voice conversion logic
   - Use f0 and content features already extracted

## Configuration

### Environment Variables

```bash
# Set custom model path
export APP_RVC_MODEL_PATH=models/rvc/my_model.pth

# Enable/disable GPU
export APP_RVC_ENABLE_GPU=true
```

### In Code

Edit `processor/config.py`:
```python
rvc_model_path: str = "models/rvc/pretrained.pth"
rvc_enable_gpu: bool = True
```

## Docker Setup

The `models` directory is automatically mounted in Docker:

```yaml
volumes:
  - ../models:/app/models
```

So you can:
1. Place your model in `models/rvc/` on your host machine
2. It will be available inside the Docker container at `/app/models/rvc/`

## Using RVC in the App

1. **Enable RVC checkbox** in the web UI
2. The app will attempt to load and use the model
3. If model is missing or inference fails, it falls back to pass-through (no conversion)

## Troubleshooting

### "RVC model not found"
- Check that model file exists at `models/rvc/pretrained.pth`
- Run `python scripts/test_rvc_setup.py` to verify

### "RVC model architecture not defined"
- This means the model file exists but loading code needs implementation
- See `_load_rvc_model_pytorch()` in `processor/ml_refine/rvc_refiner.py`

### "RVC PyTorch inference not fully implemented"
- Model loaded but inference code is placeholder
- See `_apply_rvc_pytorch()` in `processor/ml_refine/rvc_refiner.py`

### Model format issues
- Ensure model matches your RVC framework version
- Check if model is PyTorch (`.pth`) or ONNX (`.onnx`)
- Verify model structure matches expected format

## Next Steps

1. **Get a model**: Download or train an RVC model
2. **Place it**: Put model file in `models/rvc/pretrained.pth`
3. **Implement loading**: Adapt `_load_rvc_model_pytorch()` to your model format
4. **Implement inference**: Complete `_apply_rvc_pytorch()` with actual conversion logic
5. **Test**: Use the app and verify voice conversion works

## Resources

- [RVC-Project GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [HuggingFace RVC Models](https://huggingface.co/models?search=rvc)
- [RVC Documentation](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki)

## Support

For issues:
1. Check `models/rvc/README.md` for detailed setup
2. Run `python scripts/test_rvc_setup.py` to diagnose
3. Check worker logs for RVC-related messages

