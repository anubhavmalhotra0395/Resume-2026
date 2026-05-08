#!/usr/bin/env python3
"""
Smoke test for RVC setup - tests model loading and basic processing.

This script:
- Loads RVCRefiner with get_rvc_refiner()
- Processes a small dry vocal + reference vocal
- Writes output to test_outputs/test_rvc_out.wav
- Logs availability of HuBERT, RVC model, vocoder, device info, processing duration
- Checks audio duration and prints summary
"""
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import librosa
from processor.ml_refine.rvc_refiner import get_rvc_refiner
from processor.config import settings

def create_test_audio(duration_sec=2.0, sr=44100, freq=440.0):
    """Create a simple test tone."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    # Add some harmonics for more realistic vocal-like sound
    audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
    audio += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
    return audio.astype(np.float32)

def main():
    print("=" * 60)
    print("RVC Setup Smoke Test")
    print("=" * 60)
    print()
    
    # Create test output directory
    test_output_dir = Path("test_outputs")
    test_output_dir.mkdir(exist_ok=True)
    
    # Create test audio
    print("Creating test audio...")
    dry_audio = create_test_audio(duration_sec=2.0, sr=44100, freq=440.0)
    ref_audio = create_test_audio(duration_sec=2.0, sr=44100, freq=440.0)
    
    # Save test files
    dry_path = test_output_dir / "test_dry.wav"
    ref_path = test_output_dir / "ref.wav"
    sf.write(dry_path, dry_audio, 44100)
    sf.write(ref_path, ref_audio, 44100)
    print(f"✓ Test audio created: {dry_path}")
    print()
    
    # Get RVC refiner
    print("Initializing RVC refiner...")
    model_path = settings.rvc_model_path
    print(f"  Model path: {model_path}")
    print(f"  Model exists: {Path(model_path).exists()}")
    
    try:
        refiner = get_rvc_refiner(
            model_path=model_path if Path(model_path).exists() else None,
            enable_gpu=True,
        )
        print(f"  Device: {refiner.device}")
        print()
        
        # Check model availability
        print("Checking model availability...")
        hubert_available = refiner.hubert_model is not None
        rvc_available = refiner.rvc_model is not None
        vocoder_available = refiner.vocoder is not None
        
        print(f"  HuBERT encoder: {'✓ Available' if hubert_available else '✗ Not available'}")
        print(f"  RVC model: {'✓ Available' if rvc_available else '✗ Not available'}")
        print(f"  Vocoder: {'✓ Available' if vocoder_available else '✗ Not available (will use Griffin-Lim)'}")
        print()
        
        # Try to load models
        print("Loading models...")
        start_load = time.time()
        models_loaded = refiner.load_models()
        load_time = time.time() - start_load
        
        if models_loaded:
            print(f"✓ Models loaded in {load_time:.2f}s")
        else:
            print(f"⚠ Some models failed to load (took {load_time:.2f}s)")
        print()
        
        # Process audio
        print("Processing audio with RVC...")
        start_process = time.time()
        processed = refiner.process(dry_audio, sr=44100)
        process_time = time.time() - start_process
        
        if processed is not None and len(processed) > 0:
            print(f"✓ Processing completed in {process_time:.2f}s")
            print(f"  Input length: {len(dry_audio)} samples ({len(dry_audio)/44100:.2f}s)")
            print(f"  Output length: {len(processed)} samples ({len(processed)/44100:.2f}s)")
            
            # Check audio quality
            max_val = np.max(np.abs(processed))
            rms = np.sqrt(np.mean(processed**2))
            print(f"  Max amplitude: {max_val:.4f}")
            print(f"  RMS: {rms:.4f}")
            
            # Save output
            output_path = test_output_dir / "test_rvc_out.wav"
            sf.write(output_path, processed, 44100)
            print(f"✓ Output saved to: {output_path}")
            print()
            
            # Summary
            print("=" * 60)
            print("Test Summary")
            print("=" * 60)
            print(f"Status: {'✓ PASS' if max_val > 0.01 else '⚠ WARNING (very quiet output)'}")
            print(f"HuBERT: {'✓' if hubert_available else '✗'}")
            print(f"RVC Model: {'✓' if rvc_available else '✗'}")
            print(f"Vocoder: {'✓' if vocoder_available else '✗ (using fallback)'}")
            print(f"Device: {refiner.device}")
            print(f"Load time: {load_time:.2f}s")
            print(f"Process time: {process_time:.2f}s")
            print(f"Total time: {load_time + process_time:.2f}s")
            print()
            
            if not rvc_available:
                print("⚠ RVC model not found. Processing used pass-through.")
                print(f"   Place model at: {model_path}")
            
            return 0 if max_val > 0.01 else 1
        else:
            print("✗ Processing failed - no output generated")
            return 1
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
