"""
Mid-Side EQ — apply separate EQ curves to the mid and side channels of a stereo signal.
Mono input is passed through unchanged (no side channel to process).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from processor.dsp.eq import EqBand, apply_eq


@dataclass
class MsEqSettings:
    mid_bands:  List[EqBand] = field(default_factory=list)
    side_bands: List[EqBand] = field(default_factory=list)


def apply_ms_eq(x: np.ndarray, sr: int, cfg: MsEqSettings) -> np.ndarray:
    """
    Encode to M-S, EQ each channel independently, decode back.
    If input is mono, returns as-is.
    """
    if x.ndim == 1:
        # Mono — M-S is meaningless, apply mid EQ only to the mono signal
        if cfg.mid_bands:
            return apply_eq(x, sr, cfg.mid_bands)
        return x

    if x.ndim == 2 and x.shape[0] == 2:
        left, right = x[0], x[1]
    elif x.ndim == 2 and x.shape[1] == 2:
        left, right = x[:, 0], x[:, 1]
    else:
        return x

    mid  = (left + right) * 0.5
    side = (left - right) * 0.5

    if cfg.mid_bands:
        mid = apply_eq(mid, sr, cfg.mid_bands)
    if cfg.side_bands:
        side = apply_eq(side, sr, cfg.side_bands)

    new_left  = mid + side
    new_right = mid - side

    if x.ndim == 2 and x.shape[0] == 2:
        return np.stack([new_left, new_right], axis=0).astype(np.float32)
    else:
        return np.stack([new_left, new_right], axis=1).astype(np.float32)
