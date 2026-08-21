import numpy as np
from processor.dsp.analysis.reverb_analysis import estimate_reverb_params, ReverbProfile
from processor.dsp.effects.apply_reverb import apply_reverb
from processor.dsp.reverb import ReverbSettings as _Settings


def test_reverb_profile_estimation_and_apply():
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t) * np.exp(-3.0 * t)
    y = y.astype(np.float32)

    profile = estimate_reverb_params(y, sr)
    assert isinstance(profile, ReverbProfile)
    assert 0.0 < profile.rt60 < 10.0

    # Convert the analysis profile to apply settings, as the worker does —
    # the old test passed the profile object straight in, which apply_reverb
    # has never accepted.
    settings = _Settings(decay_s=profile.rt60, mix=max(profile.wet, 0.1),
                         pre_delay_ms=profile.predelay_ms)
    out = apply_reverb(y, sr, settings)
    assert len(out) == len(y)
    assert not np.any(np.isnan(out))


if __name__ == "__main__":
    test_reverb_profile_estimation_and_apply()
    print("✓ reverb tests passed")
"""
Unit tests for reverb processing.
"""
import numpy as np
import pytest
from processor.dsp.reverb import apply_reverb, ReverbSettings


def test_reverb_impulse_response():
    """Test that reverb produces tail on impulse."""
    sr = 44100
    # Create impulse (single sample spike)
    audio = np.zeros(int(sr * 0.1), dtype=np.float32)
    audio[100] = 1.0  # Impulse at sample 100
    
    # Apply reverb with long decay
    reverb_settings = ReverbSettings(
        decay_s=1.0,
        mix=0.5,
        pre_delay_ms=10.0,
    )
    processed = apply_reverb(audio, sr, reverb_settings)
    
    # Check that there's energy after the impulse (reverb tail)
    tail_start = 200  # After impulse
    tail = processed[tail_start:]
    tail_energy = np.sqrt(np.mean(tail ** 2))
    
    # Tail should have some energy (reverb decay)
    assert tail_energy > 0.001, f"Reverb tail too quiet: {tail_energy}"


def test_reverb_preserves_audio():
    """Test that reverb doesn't silence audio."""
    sr = 44100
    duration = 0.5
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3
    
    reverb_settings = ReverbSettings(
        decay_s=0.5,
        mix=0.3,
        pre_delay_ms=20.0,
    )
    processed = apply_reverb(audio, sr, reverb_settings)
    
    # Should not be silent
    assert np.max(np.abs(processed)) > 0.01
    # Should have similar or longer length (reverb adds tail)
    assert len(processed) >= len(audio) * 0.9


if __name__ == "__main__":
    test_reverb_impulse_response()
    test_reverb_preserves_audio()
    print("✓ All reverb tests passed")

