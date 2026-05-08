"""
Parallel Compression (New York compression) — compresses a duplicate of the signal
heavily, then blends it back with the dry signal to preserve transients while
adding density and perceived loudness.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ParallelCompSettings:
    threshold_db: float = -30.0
    ratio: float = 10.0
    attack_ms: float = 2.0
    release_ms: float = 150.0
    blend: float = 0.3        # amount of compressed copy added to dry (0–1)


def _compress(x: np.ndarray, sr: int, threshold_db: float, ratio: float,
              attack_ms: float, release_ms: float) -> np.ndarray:
    """Simple peak-following compressor (no lookahead)."""
    threshold_lin = 10 ** (threshold_db / 20.0)
    attack_coef   = np.exp(-1.0 / (sr * attack_ms  / 1000.0 + 1e-9))
    release_coef  = np.exp(-1.0 / (sr * release_ms / 1000.0 + 1e-9))

    env = 0.0
    out = np.empty_like(x)
    for i in range(len(x)):
        level = abs(x[i])
        if level > env:
            env = attack_coef * env + (1.0 - attack_coef) * level
        else:
            env = release_coef * env + (1.0 - release_coef) * level

        if env > threshold_lin:
            gain_reduction = threshold_lin * ((env / threshold_lin) ** (1.0 / ratio)) / env
        else:
            gain_reduction = 1.0
        out[i] = x[i] * gain_reduction

    return out.astype(np.float32)


def apply_parallel_comp(x: np.ndarray, sr: int, cfg: ParallelCompSettings) -> np.ndarray:
    """
    Sum compressed copy at `blend` ratio back to the dry signal.
    Both copies are at unity — no wet/dry crossfade; they are additive.
    """
    if cfg.blend <= 0:
        return x

    compressed = _compress(
        x, sr,
        threshold_db=cfg.threshold_db,
        ratio=cfg.ratio,
        attack_ms=cfg.attack_ms,
        release_ms=cfg.release_ms,
    )
    blend = float(np.clip(cfg.blend, 0.0, 1.0))
    result = x + blend * compressed

    # Normalise to prevent clipping
    peak = float(np.max(np.abs(result)))
    if peak > 0.99:
        result = result / peak * 0.99

    return result.astype(np.float32)
