"""Chorus: detector must not invent an effect, applier must not comb filter.

Reported by ear as a robotic, warbling vocal and confirmed by A/B render --
disabling chorus was the single change that removed it.
"""
import numpy as np

from processor.dsp.analysis.chorus_analysis import detect_chorus
from processor.dsp.effects.apply_chorus import apply_chorus

SR = 22050


def _vocal_like(seconds=6.0, sr=SR):
    """Phrases separated by gaps -- the phrasing that used to trip the gate."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 10))
    env = np.zeros(n)
    for start in (0.2, 2.2, 4.2):
        a, b = int(start * sr), min(n, int((start + 1.3) * sr))
        if b - a > 2:
            env[a:b] = np.sin(np.pi * np.linspace(0, 1, b - a))
    y = tone * env
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.7).astype(np.float32)


def test_untreated_mono_reports_no_chorus():
    """The old code fabricated a side channel from a 2 ms delayed copy and
    reported mix=0.250 -- its maximum -- on a raw dry acapella."""
    assert detect_chorus(_vocal_like(), SR).mix == 0.0


def test_untreated_stereo_reports_no_chorus():
    m = _vocal_like()
    rng = np.random.RandomState(0)
    wide = np.stack([m, np.roll(m, 13) * 0.98 + rng.randn(len(m)) * 1e-4])
    assert detect_chorus(wide, SR).mix == 0.0


def test_silence_and_noise_report_no_chorus():
    rng = np.random.RandomState(1)
    for sig in (np.zeros(SR * 2, np.float32),
                (rng.randn(SR * 2) * 0.1).astype(np.float32)):
        assert detect_chorus(sig, SR).mix == 0.0


def test_applier_is_mono_compatible():
    """The applier summed both modulated delay lines with the dry signal in
    ONE channel: comb filtering with a sweeping notch, i.e. robotic by
    construction. The wet pair must cancel in the mono sum instead.
    """
    x = _vocal_like()
    out = apply_chorus(x.copy(), SR, rate_hz=1.2, depth=0.6, mix=0.3)
    assert out.ndim == 2 and out.shape[0] == 2
    mono = out.mean(axis=0)
    gain = float(np.dot(mono, x) / np.dot(x, x))
    residual = mono - gain * x
    rel = float(np.sqrt(np.mean(residual ** 2)) / (np.sqrt(np.mean(mono ** 2)) + 1e-12))
    assert rel < 0.02, f"wet did not cancel in mono (residual {rel:.4f})"


def test_applier_puts_the_modulation_in_the_stereo_field():
    """Mono-compatible must not mean 'does nothing'."""
    x = _vocal_like()
    out = apply_chorus(x.copy(), SR, rate_hz=1.2, depth=0.6, mix=0.3)
    side = (out[0] - out[1]) / 2.0
    mono = out.mean(axis=0)
    ratio = np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mono ** 2)) + 1e-12)
    assert ratio > 0.05, "no stereo modulation produced"
