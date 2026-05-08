import numpy as np
from processor.dsp.effects.apply_harmony import apply_harmony
from processor.dsp.analysis.harmony_analysis import HarmonyProfile


def test_apply_harmony_basic():
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Dry vocal: sine at 220 Hz
    dry = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    # Simple profile: +4 semitones, medium strength, center pan, no offset
    profile = HarmonyProfile(
        intervals_semitones=[4],
        strengths=[0.5],
        pans=[0.0],
        timing_offsets=[0.0],
    )

    out = apply_harmony(dry, sr, profile)

    # Length should match
    assert len(out.shape) == 1
    assert len(out) == len(dry)

    # Should not be silent
    assert np.max(np.abs(out)) > 0.01

    # Should be peak-normalized to <= 1
    assert np.max(np.abs(out)) <= 1.0


if __name__ == "__main__":
    test_apply_harmony_basic()
    print("✓ apply_harmony basic test passed")

