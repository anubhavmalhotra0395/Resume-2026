"""
Unit tests for EQ processing.
"""
import numpy as np
import pytest
from processor.dsp.eq import apply_eq, EqBand


def test_eq_boost():
    """Test that EQ boost increases RMS in boosted band."""
    # Create white noise
    sr = 44100
    duration = 1.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    
    # Measure RMS in 1kHz band before
    from scipy.signal import butter, lfilter
    b, a = butter(4, [900/(sr/2), 1100/(sr/2)], btype='band')
    band_before = lfilter(b, a, audio)
    rms_before = np.sqrt(np.mean(band_before ** 2))
    
    # Apply 6dB boost at 1kHz
    bands = [EqBand(f=1000, gain_db=6.0, q=1.0)]
    processed = apply_eq(audio, sr, bands)
    
    # Measure RMS in same band after
    band_after = lfilter(b, a, processed)
    rms_after = np.sqrt(np.mean(band_after ** 2))
    
    # Boosted band should have higher RMS
    assert rms_after > rms_before * 1.5, f"EQ boost failed: {rms_after} <= {rms_before * 1.5}"


def test_eq_cut():
    """Test that EQ cut decreases RMS in cut band."""
    # Create white noise
    sr = 44100
    duration = 1.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    
    # Measure RMS in 1kHz band before
    from scipy.signal import butter, lfilter
    b, a = butter(4, [900/(sr/2), 1100/(sr/2)], btype='band')
    band_before = lfilter(b, a, audio)
    rms_before = np.sqrt(np.mean(band_before ** 2))
    
    # Apply -6dB cut at 1kHz
    bands = [EqBand(f=1000, gain_db=-6.0, q=1.0)]
    processed = apply_eq(audio, sr, bands)
    
    # Measure RMS in same band after
    band_after = lfilter(b, a, processed)
    rms_after = np.sqrt(np.mean(band_after ** 2))
    
    # Cut band should have lower RMS
    assert rms_after < rms_before * 0.7, f"EQ cut failed: {rms_after} >= {rms_before * 0.7}"


def test_eq_multiple_bands():
    """Test applying multiple EQ bands."""
    sr = 44100
    duration = 1.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    
    # Apply multiple bands
    bands = [
        EqBand(f=200, gain_db=3.0, q=0.7),
        EqBand(f=1000, gain_db=-3.0, q=1.0),
        EqBand(f=5000, gain_db=2.0, q=1.5),
    ]
    
    processed = apply_eq(audio, sr, bands)
    
    # Should not be silent
    assert np.max(np.abs(processed)) > 0.01
    # Should have similar length
    assert abs(len(processed) - len(audio)) < 100


if __name__ == "__main__":
    test_eq_boost()
    test_eq_cut()
    test_eq_multiple_bands()
    print("✓ All EQ tests passed")

