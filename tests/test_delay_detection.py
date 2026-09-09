"""Delay detection by cancellation.

Reported by ear: "there is delay in the reference but none in the processed".
The previous detector used onset autocorrelation, which cannot tell an echo
from the song's rhythm -- it read ~400 ms periodicity in the reference AND in
the dry vocal (same song, same tempo), so the rhythm-cancellation step meant
to fix that removed the echo too and the result fell just under the gate.

An echo is the SAME waveform attenuated and delayed, while repeated musical
content is merely similar, so least-squares cancellation separates them.
"""
import numpy as np

from processor.dsp.delay import detect_delay

SR = 22050


def _phrases(seconds=30.0, sr=SR, seed=0):
    """Rhythmic phrases at a fixed tempo, with NO echo -- the case the old
    detector mistook for one."""
    rng = np.random.RandomState(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float64)
    beat = int(0.4 * sr)                      # 150 bpm, the "400 ms" trap
    for k in range(int(seconds / 0.4)):
        a = k * beat
        b = min(n, a + int(0.25 * sr))
        if b - a < 8:
            continue
        t = np.arange(b - a) / sr
        # each phrase is DIFFERENT content at the same rhythm
        f = 180 + rng.randint(-40, 40)
        y[a:b] = (np.sin(2 * np.pi * f * t) * np.hanning(b - a)
                  + rng.randn(b - a) * 0.02)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.7


def _with_echo(x, sr, ms, level):
    d = int(ms * sr / 1000)
    e = np.zeros_like(x)
    e[d:] = x[:-d] * level
    return x + e


def test_rhythm_alone_is_not_reported_as_delay():
    r = detect_delay(_phrases(), SR)
    assert r is None or r.get("type") == "none", r


def test_a_known_echo_is_recovered():
    x = _phrases()
    for ms, level in ((300.0, 0.35), (450.0, 0.25)):
        r = detect_delay(_with_echo(x, SR, ms, level), SR)
        assert r and r.get("type") != "none", f"missed {ms}ms echo"
        assert abs(r["delay_ms"] - ms) <= 8, f"lag {r['delay_ms']} vs {ms}"
        assert abs(r["echo_level"] - level) < 0.12, \
            f"level {r['echo_level']:.3f} vs {level}"


def test_echo_on_top_of_rhythm_is_still_found():
    """The real failure: an echo at a musical interval, where the rhythm sits."""
    x = _phrases()
    r = detect_delay(_with_echo(x, SR, 400.0, 0.30), SR)
    assert r and r.get("type") != "none"
    assert abs(r["delay_ms"] - 400.0) <= 8


def test_confidence_rises_with_echo_level():
    x = _phrases()
    c = [detect_delay(_with_echo(x, SR, 250.0, lv), SR)["confidence"]
         for lv in (0.15, 0.30, 0.45)]
    assert c[0] <= c[1] <= c[2], c


def test_slapback_is_refused_rather_than_guessed():
    """Below ~180 ms a voice cancels against its own pitch period, so this
    method cannot tell a slapback from a sustained note. It must decline."""
    x = _phrases()
    r = detect_delay(_with_echo(x, SR, 90.0, 0.30), SR)
    assert r is None or r.get("type") == "none" or r["delay_ms"] >= 180


def test_a_sustained_tone_is_not_an_echo():
    """A periodic waveform cancels against itself at multiples of its pitch
    period -- a synthetic sung melody with no echo cancelled 55.7% of its
    energy. The transient requirement is what keeps this method inside the
    material it was calibrated on."""
    sr = SR
    t = np.arange(int(20 * sr)) / sr
    # a sung melody: sustained notes that CHANGE pitch, so any self-similarity
    # lag moves with the tune
    f = np.repeat([196.0, 220.0, 247.0, 165.0, 208.0], int(len(t) / 5) + 1)[:len(t)]
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * 0.7
    r = detect_delay(y.astype(np.float64), sr)
    assert r is None or r.get("type") == "none", r


def test_silence_returns_nothing():
    assert detect_delay(np.zeros(SR * 3), SR) is None
