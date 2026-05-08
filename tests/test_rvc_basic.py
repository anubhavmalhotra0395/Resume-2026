"""
Basic unit tests for RVC processing.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from processor.ml_refine.rvc_refiner import get_rvc_refiner


def test_rvc_loading():
    """Test that RVCRefiner can be loaded."""
    refiner = get_rvc_refiner(enable_gpu=False)
    assert refiner is not None
    assert refiner.device is not None
    print("✓ RVCRefiner loaded")


def test_rvc_process_pass_through():
    """Test RVC process() with pass-through (no model)."""
    refiner = get_rvc_refiner(enable_gpu=False)
    
    # Create 1-second test clip
    sr = 44100
    duration = 1.0
    test_audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    
    # Process (will use pass-through if no model)
    processed = refiner.process(test_audio, sr)
    
    # Should return audio (even if pass-through)
    assert processed is not None
    assert len(processed) > 0
    assert np.max(np.abs(processed)) > 0
    
    print("✓ RVC process() works (pass-through acceptable)")


def test_rvc_output_file():
    """Test that RVC can produce output file."""
    refiner = get_rvc_refiner(enable_gpu=False)
    
    # Create test audio
    sr = 44100
    duration = 1.0
    test_audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    
    # Process
    processed = refiner.process(test_audio, sr)
    
    if processed is not None:
        # Save to file
        output_path = Path("test_outputs") / "test_rvc_basic.wav"
        output_path.parent.mkdir(exist_ok=True)
        sf.write(output_path, processed, sr)
        
        # Verify file exists
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        print(f"✓ Output file created: {output_path}")


if __name__ == "__main__":
    test_rvc_loading()
    test_rvc_process_pass_through()
    test_rvc_output_file()
    print("✓ All RVC basic tests passed")

