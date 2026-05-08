"""
Vocal Doubler — creates opposite-polarity pitch-shifted L/R copies for thick doubling.
Distinct from Width/ADT: uses real pitch shifting to simulate a second take.
"""
from dataclasses import dataclass

import numpy as np
import librosa


@dataclass
class DoublerSettings:
    detune_cents_l: float = -8.0   # left copy detuned down
    detune_cents_r: float = 8.0    # right copy detuned up
    delay_ms_l: float = 18.0
    delay_ms_r: float = 22.0
    mix: float = 0.35


def _pitch_shift_fast(y: np.ndarray, sr: int, cents: float) -> np.ndarray:
    """Pitch shift by cents using librosa. Positive = up."""
    n_steps = cents / 100.0
    if abs(n_steps) < 0.01:
        return y.copy()
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)
    except Exception:
        return y.copy()


def apply_doubler(x: np.ndarray, sr: int, cfg: DoublerSettings) -> np.ndarray:
    """
    Apply vocal doubling. Returns mono sum (L+R blended) to stay compatible
    with the mono chain. The stereo spread effect comes from opposite detune.
    """
    if cfg.mix <= 0:
        return x

    delay_l = int(sr * cfg.delay_ms_l / 1000.0)
    delay_r = int(sr * cfg.delay_ms_r / 1000.0)

    shifted_l = _pitch_shift_fast(x, sr, cfg.detune_cents_l)
    shifted_r = _pitch_shift_fast(x, sr, cfg.detune_cents_r)

    # Apply delays
    def _delay(sig: np.ndarray, d: int) -> np.ndarray:
        out = np.zeros_like(sig)
        if d < len(sig):
            out[d:] = sig[:-d] if d > 0 else sig
        return out

    double_l = _delay(shifted_l, delay_l)
    double_r = _delay(shifted_r, delay_r)

    # Sum L+R doubles and blend with dry
    doubles = (double_l + double_r) * 0.5
    result = (1.0 - cfg.mix) * x + cfg.mix * doubles
    return result.astype(np.float32)
