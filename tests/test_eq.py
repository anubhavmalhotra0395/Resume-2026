import numpy as np

from processor.dsp.eq import EqBand, apply_eq


def test_eq_passes_audio_shape():
    x = np.zeros(44100, dtype=np.float32)
    bands = [EqBand(f=1000, gain_db=3.0, q=1.0)]
    y = apply_eq(x, 44100, bands)
    assert y.shape == x.shape


