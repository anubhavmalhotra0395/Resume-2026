import numpy as np
from processor.dsp.delay import detect_delay, apply_delay


def test_delay_detection():
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 440 * t)

    # make an artificial delay
    delay_samples = int(0.25 * sr)  # 250 ms at 1s duration
    y = np.copy(x)
    y[delay_samples:] += 0.3 * x[:-delay_samples]

    res = detect_delay(y, sr)
    assert res["delay_ms"] > 0
    assert res["confidence"] >= 0


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

