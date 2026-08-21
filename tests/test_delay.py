import numpy as np
from processor.dsp.delay import detect_delay, apply_delay


def test_delay_detection():
    """Echo detection needs transients — a continuous sine has no onsets, so
    the old version of this test could never pass. Use a bursty signal with a
    planted 420 ms echo and verify the *measured* lag is close."""
    sr = 44100
    t = np.arange(sr * 12) / sr
    bursts = (np.sin(2 * np.pi * 1.1 * t) > 0.55).astype(float)
    dry = (np.sin(2 * np.pi * 300 * t) * bursts * 0.5).astype(np.float32)
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

