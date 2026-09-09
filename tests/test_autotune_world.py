"""Guards for the WORLD-vocoder autotune path.

The properties here are the ones that were each broken at some point while
this path was built, and none of them are visible in a "does it run" check:
a correction can be perfect on pitch and still ruin the performance.
"""
import numpy as np
import pytest

from processor.dsp.autotune import _world_correct, _scale_notes_in_range

pyworld = pytest.importorskip("pyworld")

SR = 22050
CHROMATIC = _scale_notes_in_range(0, "chromatic")


def _sung(dev_semitones=0.35, seconds=3.0, sr=SR):
    """A detuned vowel with a loud onset, an unvoiced burst and a decay."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 220.0 * 2 ** (dev_semitones / 12.0)
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = np.exp(-t * 0.6)
    env[: int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))
    y = y * env
    # unvoiced consonant in the middle
    a, b = int(1.4 * sr), int(1.55 * sr)
    y[a:b] = np.random.RandomState(0).randn(b - a) * 0.2
    return (y / np.max(np.abs(y)) * 0.7).astype(np.float32)


def _cents_off(y, sr=SR):
    import librosa
    f0, _, _ = librosa.pyin(y.astype(float), fmin=80, fmax=800, sr=sr,
                            frame_length=2048)
    f = f0[np.isfinite(f0)]
    assert len(f) > 20
    midi = 69 + 12 * np.log2(f / 440.0)
    return float(np.median(np.abs(midi - np.round(midi)) * 100))


def test_pulls_a_detuned_note_onto_the_grid():
    y = _sung(dev_semitones=0.35)
    out = _world_correct(y, SR, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    assert _cents_off(out) < _cents_off(y) / 2


def test_strength_zero_leaves_pitch_alone():
    y = _sung(dev_semitones=0.35)
    out = _world_correct(y, SR, strength=0.0, scale_notes=CHROMATIC)
    assert out is not None
    assert abs(_cents_off(out) - _cents_off(y)) < 12.0


def test_keeps_the_input_loudness_contour():
    """WORLD rebuilds amplitude from a smoothed envelope, which flattened the
    micro-dynamics and cost 4.4 points of transient fidelity downstream."""
    y = _sung()
    out = _world_correct(y, SR, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    win = 512
    n = min(len(y), len(out)) // win
    ry = np.array([np.sqrt(np.mean(y[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    ro = np.array([np.sqrt(np.mean(out[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    assert np.corrcoef(ry, ro)[0, 1] > 0.95


def test_unvoiced_material_survives():
    """Consonants have no pitch to correct and only smear if resynthesised."""
    y = _sung()
    out = _world_correct(y, SR, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    a, b = int(1.42 * SR), int(1.53 * SR)
    assert np.corrcoef(y[a:b], out[a:b])[0, 1] > 0.9


def test_output_is_finite_and_unclipped():
    y = _sung()
    out = _world_correct(y, SR, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    assert np.isfinite(out).all()
    assert len(out) == len(y)
    assert np.max(np.abs(out)) <= 1.0


def test_too_short_input_returns_none_so_the_caller_can_fall_back():
    assert _world_correct(np.zeros(1000, np.float32), SR, scale_notes=CHROMATIC) is None


def test_does_not_add_noise_to_the_gaps():
    """Reported by ear: "a bit of noise in the processed one".

    dio marks the odd frame voiced on nothing but noise, and WORLD then
    rebuilds it from a spectral envelope, putting hiss between phrases where
    the input had silence. Correcting pitch must not raise the noise floor.
    """
    sr = SR
    n = int(4.0 * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    # Two sung phrases with a genuinely silent gap between them.
    for start in (0.2, 2.4):
        a, b = int(start * sr), int((start + 1.2) * sr)
        seg = np.arange(b - a) / sr
        tone = sum(np.sin(2 * np.pi * 220 * 2 ** (0.3 / 12) * k * seg) / k
                   for k in range(1, 10))
        y[a:b] = tone * np.sin(np.pi * np.linspace(0, 1, b - a))
    y = (y / np.max(np.abs(y)) * 0.7).astype(np.float32)

    gap = slice(int(1.6 * sr), int(2.3 * sr))
    quiet_in = float(np.sqrt(np.mean(y[gap] ** 2)))

    out = _world_correct(y, sr, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    quiet_out = float(np.sqrt(np.mean(out[gap] ** 2)))
    # Allow a whisker of leakage from the phrase tails, not a noise bed.
    assert quiet_out <= max(quiet_in * 2.0, 1e-4), (
        f"gap noise rose from {quiet_in:.2e} to {quiet_out:.2e}")


def _f0_flatness(y, sr):
    """Fraction of frames whose pitch exactly repeats the previous frame.

    A human never does this; a vocoder writing a piecewise-constant contour
    does it constantly.
    """
    import librosa
    f0, _, _ = librosa.pyin(y.astype(float), fmin=80, fmax=800, sr=sr,
                            frame_length=2048)
    f = f0[np.isfinite(f0)]
    assert len(f) > 20
    midi = 69 + 12 * np.log2(f / 440.0)
    return float(np.mean(np.abs(np.diff(midi)) < 0.005))


def _vibrato(seconds=3.0, sr=SR, rate_hz=5.5, depth_cents=45.0, detune_st=0.35):
    """A detuned note WITH vibrato -- the expression that must survive."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    cents = depth_cents * np.sin(2 * np.pi * rate_hz * t)
    f0 = 220.0 * 2 ** ((detune_st + cents / 100.0) / 12.0)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.ones(n)
    ramp = int(0.05 * sr)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return (y * env / np.max(np.abs(y * env)) * 0.7).astype(np.float32)


def test_correction_keeps_the_singers_vibrato():
    """Reported by ear: "too robotic when autotune is there".

    Writing the median-filtered target straight out makes pitch piecewise
    constant -- flat inside every note. The note CENTRE should be snapped and
    the motion around it preserved.
    """
    y = _vibrato()
    out = _world_correct(y, SR, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    assert _f0_flatness(out, SR) < _f0_flatness(y, SR) + 0.20, \
        "pitch contour went flat -- vibrato was discarded"


def test_correction_still_moves_the_note_onto_the_grid():
    """The vibrato fix must not stop it correcting: centre still snaps."""
    y = _vibrato(detune_st=0.35, depth_cents=20.0)
    out = _world_correct(y, SR, strength=1.0, scale_notes=CHROMATIC)
    assert out is not None
    assert _cents_off(out) < _cents_off(y)
