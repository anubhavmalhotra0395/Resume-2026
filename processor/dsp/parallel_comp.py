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
    """Heavy compression pass — delegates to the block-based compressor
    (the old per-sample Python loop here ran slower than realtime)."""
    from processor.dsp.compressor import CompressorSettings, apply_compressor

    return apply_compressor(x, sr, CompressorSettings(
        threshold_db=threshold_db, ratio=ratio,
        attack_ms=attack_ms, release_ms=release_ms,
        makeup_db=0.0, knee_db=0.0,
    ))


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
