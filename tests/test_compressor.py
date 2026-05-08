"""
Unit tests for compressor processing.
"""
import numpy as np
import pytest
from processor.dsp.compressor import apply_compressor, CompressorSettings


def test_compressor_reduces_crest_factor():
    """Test that compressor reduces crest factor (dynamic range)."""
    sr = 44100
    duration = 1.0
    # Create high-dynamic signal (loud peaks, quiet sections)
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # Add loud peaks
    audio[::1000] = 1.0  # Every 1000 samples, set to 1.0
    
    # Compute crest factor before
    rms_before = np.sqrt(np.mean(audio ** 2))
    peak_before = np.max(np.abs(audio))
    crest_before = peak_before / rms_before if rms_before > 0 else 0
    
    # Apply compressor
    comp_settings = CompressorSettings(
        threshold_db=-12.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=50.0,
        makeup_db=0.0,
        knee_db=3.0,
    )
    processed = apply_compressor(audio, sr, comp_settings)
    
    # Compute crest factor after
    rms_after = np.sqrt(np.mean(processed ** 2))
    peak_after = np.max(np.abs(processed))
    crest_after = peak_after / rms_after if rms_after > 0 else 0
    
    # Crest factor should decrease (compression reduces dynamic range)
    assert crest_after < crest_before, f"Compressor failed: {crest_after} >= {crest_before}"


def test_compressor_preserves_audio():
    """Test that compressor doesn't silence audio."""
    sr = 44100
    duration = 0.5
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3
    
    comp_settings = CompressorSettings(
        threshold_db=-18.0,
        ratio=3.0,
        attack_ms=10.0,
        release_ms=100.0,
    )
    processed = apply_compressor(audio, sr, comp_settings)
    
    # Should not be silent
    assert np.max(np.abs(processed)) > 0.01
    # Should have similar length
    assert abs(len(processed) - len(audio)) < 10


if __name__ == "__main__":
    test_compressor_reduces_crest_factor()
    test_compressor_preserves_audio()
    print("✓ All compressor tests passed")

