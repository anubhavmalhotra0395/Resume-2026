import numpy as np
from processor.dsp.analysis.harmony_analysis import HarmonyProfile
from processor.dsp.effects.apply_harmony import apply_harmony


def test_harmony_generation_basic():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    dry = 0.3 * np.sin(2 * np.pi * 220 * t)

    profile = HarmonyProfile(
        intervals_semitones=[4, 7],
        strengths=[0.5, 0.4],
        pans=[-0.5, 0.5],
        timing_offsets=[0.0, 0.01],
    )

    out = apply_harmony(dry, sr, profile)

    assert len(out) == len(dry)
    assert not np.isnan(out).any()
    assert np.max(np.abs(out)) <= 1.0


if __name__ == "__main__":
    test_harmony_generation_basic()
    print("✓ Harmony integration test passed")

