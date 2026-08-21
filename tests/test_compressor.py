"""
Unit tests for compressor processing.
"""
import numpy as np
import pytest
from processor.dsp.compressor import apply_compressor, CompressorSettings


def test_compressor_reduces_dynamic_range():
    """Loud passages must come down more than quiet ones.

    (The old version of this test injected single-sample spikes and expected
    a lower crest factor — but an RMS compressor with a 5 ms attack passes
    single-sample transients by design; catching those is a limiter's job.
    No correct compressor could satisfy it.)
    """
    sr = 44100
    t = np.arange(sr * 2) / sr
    carrier = np.sin(2 * np.pi * 440 * t)
    # Alternating loud (0.7) and quiet (0.07) passages — 20 dB apart
    loud_mask = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(np.float64)
    audio = (carrier * (0.07 + 0.63 * loud_mask)).astype(np.float32)

    comp_settings = CompressorSettings(
        threshold_db=-18.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=50.0,
        makeup_db=0.0,
        knee_db=3.0,
    )
    processed = apply_compressor(audio, sr, comp_settings)

    def rms_db(seg):
        return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)

    loud = loud_mask > 0.5
    gap_before = rms_db(audio[loud]) - rms_db(audio[~loud])
    gap_after = rms_db(processed[loud]) - rms_db(processed[~loud])

    # 20 dB of range through a 4:1 compressor must shrink substantially
    assert gap_after < gap_before - 3.0, (
        f"Dynamic range not reduced: {gap_before:.1f} dB -> {gap_after:.1f} dB"
    )
    # And the loud material specifically must be attenuated
    assert rms_db(processed[loud]) < rms_db(audio[loud]) - 2.0


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

