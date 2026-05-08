import numpy as np
import librosa
from processor.dsp.analysis.harmony_analysis import detect_harmonies


def test_detect_harmony_simple():
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Fundamental at A3 (220 Hz)
    fundamental = 0.5 * np.sin(2 * np.pi * 220 * t)
    # Harmony at +7 semitones (~330 Hz)
    harmony = 0.25 * np.sin(2 * np.pi * 220 * (2 ** (7 / 12)) * t)

    ref = fundamental + harmony

    profile = detect_harmonies(ref, sr)

    assert 7 in profile.intervals_semitones, f"Expected +7 semitone harmony, got {profile.intervals_semitones}"
    assert len(profile.intervals_semitones) == len(profile.strengths) == len(profile.pans) == len(profile.timing_offsets)
    assert all(np.isfinite(profile.strengths)), "Strengths contain non-finite values"


if __name__ == "__main__":
    test_detect_harmony_simple()
    print("✓ Harmony detection test passed")

