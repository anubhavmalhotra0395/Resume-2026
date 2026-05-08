# QA and Testing Guide

## Overview

This document describes the testing infrastructure, QA procedures, and validation steps for the vocal style transfer system.

## Test Structure

```
tests/
  test_dsp_eq.py          # EQ unit tests
  test_compressor.py      # Compressor unit tests
  test_reverb.py          # Reverb unit tests
  test_rvc_basic.py       # RVC basic tests
  test_regression.py      # Golden reference tests
  golden/                 # Golden reference files (optional)
    dry.wav
    reference.wav
    expected_style.wav

scripts/
  test_rvc_setup.py      # Smoke test for RVC setup
  test_api_job.py        # API integration test
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
python -m pytest tests/

# Run specific test file
python tests/test_dsp_eq.py
python tests/test_compressor.py
python tests/test_reverb.py
python tests/test_rvc_basic.py
```

### Smoke Tests

```bash
# Test RVC setup
python scripts/test_rvc_setup.py

# Test API integration
python scripts/test_api_job.py
```

### Regression Tests

```bash
# Run golden reference tests
python tests/test_regression.py
```

## Metrics

### Processing Metrics

Each job records the following metrics (stored in recipe JSON):

- **processing_time_total** - Total processing time (seconds)
- **processing_time_rvc** - RVC processing time (seconds)
- **processing_time_dsp** - DSP chain time (seconds)
- **spectral_distance** - Spectral distance to reference (lower is better)
- **lufs_reference** - Reference LUFS loudness
- **lufs_output** - Output LUFS loudness

### Accessing Metrics

```bash
# View metrics for a job
curl http://localhost:8000/recipes/{job_id}.json | jq .metrics

# View aggregated metrics
curl http://localhost:8000/metrics
```

## Validation

### File Validation

The system validates uploaded files with:

- **Size limit**: 50MB default (configurable via `APP_MAX_FILE_MB`)
- **Duration limit**: 600s default, 30s for RVC jobs (configurable)
- **Minimum duration**: 0.5s
- **Sample rate**: Auto-resampled to 44.1kHz if needed
- **Channels**: Auto-converted to mono if needed

### Validation Errors

- `400 Bad Request` - File too large, too long, or too short
- `415 Unsupported Media Type` - Invalid file type

## Health Checks

### API Health

```bash
# Basic health check
curl http://localhost:8000/healthz

# Model availability check
curl http://localhost:8000/health/models
```

### Model Health Response

```json
{
  "hubert": true,
  "rvc_model": true,
  "vocoder": false,
  "device": "cuda",
  "models_loaded": true,
  "msg": "All models available"
}
```

## Smoke Test Instructions

### RVC Setup Test

1. **Run the test**:
   ```bash
   python scripts/test_rvc_setup.py
   ```

2. **Expected output**:
   - ✓ Models loaded
   - ✓ Processing completed
   - Output file created at `test_outputs/test_rvc_out.wav`

3. **Troubleshooting**:
   - If models not found: Place RVC model at `models/rvc/pretrained.pth`
   - If HuBERT fails: Install `transformers` library
   - If processing fails: Check logs for errors

### API Integration Test

1. **Start the API**:
   ```bash
   docker compose up
   ```

2. **Run the test**:
   ```bash
   python scripts/test_api_job.py
   ```

3. **Expected output**:
   - ✓ Job submitted
   - ✓ Job completed
   - ✓ Output downloaded and verified

## Metric Definitions

### Spectral Distance

- **Definition**: L1 distance between log-magnitude STFTs
- **Range**: 0-100 (lower is more similar)
- **Typical values**: 
  - Very similar: < 2.0
  - Similar: 2.0-5.0
  - Different: > 5.0

### LUFS (Loudness Units)

- **Definition**: ITU-R BS.1770 loudness measurement
- **Range**: -60 to 0 dB (0 is maximum)
- **Typical values**:
  - Very quiet: < -30
  - Quiet: -30 to -20
  - Normal: -20 to -10
  - Loud: > -10

## Deployment Notes

### Pre-Deployment Checklist

- [ ] All unit tests pass
- [ ] Smoke tests pass
- [ ] Regression tests pass (if golden files exist)
- [ ] Health checks return OK
- [ ] Metrics endpoint accessible
- [ ] Model files in place
- [ ] Validation limits configured

### Monitoring

1. **Health Checks**: Set up monitoring for `/healthz` and `/health/models`
2. **Metrics**: Monitor `/metrics` endpoint for processing times
3. **Logs**: Check worker logs for errors and warnings
4. **Job Status**: Monitor job success/failure rates

## Updating Models

### RVC Model

1. Place new model at `models/rvc/pretrained.pth`
2. Restart worker: `docker compose restart worker`
3. Verify: `curl http://localhost:8000/health/models`

### Vocoder

1. Place new vocoder at `models/rvc/vocoder.pth`
2. Restart worker
3. Verify health check

### HuBERT

- Automatically downloaded from HuggingFace on first use
- No manual update needed (uses `facebook/hubert-base-ls960`)

## Troubleshooting

### Tests Fail

1. **Check dependencies**: `pip install -r requirements.txt`
2. **Check model files**: Verify models exist at expected paths
3. **Check logs**: Look for error messages in test output
4. **Check API**: Ensure API is running for integration tests

### Metrics Missing

1. Check worker logs for metric computation errors
2. Verify `processor/utils/metrics.py` is imported
3. Check recipe JSON file exists for the job

### Health Check Fails

1. Check model files exist
2. Check GPU availability (if using GPU)
3. Check worker logs for model loading errors
4. Verify dependencies installed (`transformers`, `pyworld`, etc.)

## Performance Benchmarks

### Expected Processing Times

- **RVC processing**: 2-5s per second of audio (CPU), 0.5-1s (GPU)
- **DSP chain**: 0.1-0.5s per second of audio
- **Total pipeline**: 3-6s per second of audio (CPU), 1-2s (GPU)

### Resource Usage

- **CPU**: 2-4 cores during processing
- **Memory**: 2-4GB (with models loaded)
- **GPU**: 2-4GB VRAM (if using GPU)

## Continuous Integration

### CI Pipeline (Example)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/
      - run: python scripts/test_rvc_setup.py
```

## Additional Resources

- **Architecture**: See `docs/architecture.md`
- **RVC Setup**: See `docs/RVC_MODEL_SETUP.md`
- **Implementation**: See `docs/RVC_IMPLEMENTATION.md`

