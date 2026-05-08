import numpy as np

from processor.dsp.chain import apply_chain
from processor.dsp.compressor import CompressorSettings
from processor.dsp.eq import EqBand
from processor.dsp.reverb import ReverbSettings


def test_chain_runs():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = apply_chain(
        x,
        sr,
        eq_bands=[EqBand(f=1000, gain_db=0.0, q=1.0)],
        comp=CompressorSettings(
            threshold_db=-18, ratio=3.0, attack_ms=10, release_ms=120, makeup_db=0
        ),
        reverb=ReverbSettings(decay_s=0.5, mix=0.2),
        saturation_drive=1.0,
    )
    assert y.shape == x.shape
    assert np.isfinite(y).all()


