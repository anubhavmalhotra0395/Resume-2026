"""
Golden reference regression tests.

Compares processed output against expected baseline to detect regressions.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from processor.utils.metrics import compute_spectral_distance


def test_golden_reference():
    """
    Test against golden reference files.
    
    Expected structure:
    tests/golden/
        dry.wav
        reference.wav
        expected_style.wav  (optional baseline)
    """
    golden_dir = Path("tests/golden")
    
    if not golden_dir.exists():
        print("⚠ Golden reference directory not found. Skipping regression test.")
        print(f"   Create {golden_dir} with dry.wav and reference.wav to enable.")
        return True
    
    dry_path = golden_dir / "dry.wav"
    ref_path = golden_dir / "reference.wav"
    expected_path = golden_dir / "expected_style.wav"
    
    if not dry_path.exists() or not ref_path.exists():
        print("⚠ Golden reference files not found. Skipping regression test.")
        print(f"   Required: {dry_path}, {ref_path}")
        return True
    
    # Load golden files
    dry_audio, sr = sf.read(dry_path)
    ref_audio, _ = sf.read(ref_path)
    
    if len(dry_audio.shape) > 1:
        dry_audio = dry_audio[:, 0]
    if len(ref_audio.shape) > 1:
        ref_audio = ref_audio[:, 0]
    
    # Process (simplified - would normally use full pipeline)
    from processor.analysis.style_extractor import analyze_reference
    from processor.dsp.chain import apply_chain
    from processor.dsp.eq import EqBand
    
    # Analyze reference
    recipe = analyze_reference(ref_audio, sr, dry_y=dry_audio)
    
    # Apply DSP chain
    processed = apply_chain(
        dry_audio,
        sr,
        eq_bands=recipe.eq,
        comp=recipe.compressor,
        reverb=recipe.reverb,
        saturation_drive=recipe.saturation_drive,
        width=recipe.width,
        reference=ref_audio,
    )
    
    # Save output for inspection
    output_path = Path("test_outputs") / "regression_output.wav"
    output_path.parent.mkdir(exist_ok=True)
    sf.write(output_path, processed, sr)
    print(f"✓ Regression test output: {output_path}")
    
    # Compare with expected baseline if available
    if expected_path.exists():
        expected_audio, _ = sf.read(expected_path)
        if len(expected_audio.shape) > 1:
            expected_audio = expected_audio[:, 0]
        
        # Compute spectral distance
        distance = compute_spectral_distance(output_path, expected_path, sr=sr)
        
        # Threshold for regression (adjust based on your requirements)
        threshold = 5.0  # Lower is more similar
        
        print(f"  Spectral distance to baseline: {distance:.2f} (threshold: {threshold:.2f})")
        
        if distance > threshold:
            print(f"⚠ WARNING: Spectral distance exceeds threshold!")
            print(f"   This may indicate a regression.")
            return False
        else:
            print(f"✓ Spectral distance within acceptable range")
            return True
    else:
        print("⚠ No expected baseline found. Output saved for future baseline creation.")
        print(f"   Copy {output_path} to {expected_path} to create baseline.")
        return True


if __name__ == "__main__":
    success = test_golden_reference()
    sys.exit(0 if success else 1)

