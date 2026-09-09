import numpy as np
from processor.dsp.delay import detect_delay, apply_delay


def test_delay_detection():
    """Echo detection needs transients — a continuous sine has no onsets, so
    the old version of this test could never pass. Use a bursty signal with a
    planted 420 ms echo and verify the *measured* lag is close.

    30 s, not 12: detection now requires the lag to hold across segments of
    the take (a periodic waveform cancels against itself at multiples of its
    pitch period, so a single window cannot tell an echo from a sustained
    note). Judging that needs several seconds per segment.
    """
    sr = 44100
    t = np.arange(sr * 30) / sr
    bursts = (np.sin(2 * np.pi * 1.1 * t) > 0.55).astype(float)
    # Each burst differs slightly. Detection works by cancelling a delayed
    # copy, and a bit-exactly repeating waveform IS an echo by that
    # definition -- no method can tell them apart, and no recording repeats
    # exactly anyway. Vary the tone so the control is realistic.
    rng = np.random.RandomState(0)
    wobble = np.interp(t, np.linspace(0, t[-1], 40), 300 + rng.randn(40) * 25)
    dry = (np.sin(2 * np.pi * np.cumsum(wobble) / sr) * bursts * 0.5).astype(np.float32)
    d = int(0.420 * sr)
    ref = dry.copy()
    ref[d:] += dry[:-d] * 0.45

    res = detect_delay(ref, sr, dry=dry)
    assert res["delay_ms"] > 0, f"echo not detected: {res}"
    assert abs(res["delay_ms"] - 420) < 40, f"lag off: {res['delay_ms']:.0f} ms"

    # And the rhythm-only control must NOT trigger
    res2 = detect_delay(dry, sr, dry=dry)
    assert res2["delay_ms"] == 0.0, f"false positive on echo-free signal: {res2}"


def test_delay_apply_shape():
    sr = 48000
    x = np.random.randn(sr).astype(np.float32)
    y = apply_delay(x, sr, delay_ms=200)
    assert len(x) == len(y)
    assert not np.isnan(y).any()


if __name__ == "__main__":
    test_delay_detection()
    test_delay_apply_shape()
    print("✓ delay tests passed")

