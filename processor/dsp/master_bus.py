"""
Master bus — the final polish stage, applied after all vocal effects.

Two units, in series:
  1. Glue compressor — gentle 2:1 linked-stereo bus compression (slow attack,
     programme-dependent release). This is the "everything belongs together"
     density a mixed record has and a raw effects chain doesn't.
  2. Lookahead limiter — transparent peak control to a fixed ceiling, so the
     output can sit at the reference's loudness without clipping.

Both operate on (N,) mono or (N, 2) stereo (channels-last, like the rest of
the chain's output path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processor.dsp.compressor import CompressorSettings, compressor_gain


@dataclass
class MasterBusSettings:
    glue_threshold_db: float = -18.0
    glue_ratio: float = 2.0
    glue_attack_ms: float = 30.0     # slow — lets transients breathe
    glue_release_ms: float = 250.0
    glue_makeup_db: float = 2.0
    ceiling: float = 0.97
    limiter_release_ms: float = 80.0


def _limiter_gain(peak_per_block: np.ndarray, block: int, sr: int,
                  ceiling: float, release_ms: float) -> np.ndarray:
    """Block-rate limiter gain: instant (lookahead) attack, smooth release."""
    need = np.minimum(1.0, ceiling / np.maximum(peak_per_block, 1e-9))

    # Lookahead: a block's reduction starts one block early so the attack
    # never lets a peak through.
    look = np.minimum(need, np.roll(need, -1))
    look[-1] = need[-1]

    # Release: gain may only *rise* at the release rate; drops are instant.
    rise = float(np.exp(block / (0.001 * max(release_ms, 1.0) * sr)))
    g = np.empty_like(look)
    prev = 1.0
    for i in range(len(look)):
        prev = min(look[i], prev * rise)
        g[i] = prev
    return g


def apply_master_bus(y: np.ndarray, sr: int,
                     cfg: MasterBusSettings | None = None) -> np.ndarray:
    cfg = cfg or MasterBusSettings()
    if len(y) == 0:
        return y

    stereo = y.ndim == 2
    work = np.asarray(y, dtype=np.float64)
    mid = work.mean(axis=1) if stereo else work

    # Material-relative threshold: sit ~4 dB under the programme RMS so any
    # input level gets the same gentle few dB of glue — a fixed threshold
    # would crush quiet mixes and miss loud ones.
    rms = float(np.sqrt(np.mean(mid ** 2)))
    if rms > 1e-9:
        rms_db = 20.0 * np.log10(rms)
        threshold_db = float(np.clip(rms_db - 4.0, -34.0, -10.0))
    else:
        threshold_db = cfg.glue_threshold_db

    # ── 1. Glue compression (linked stereo: one gain from the mid signal) ──
    gain = compressor_gain(mid, sr, CompressorSettings(
        threshold_db=threshold_db,
        ratio=cfg.glue_ratio,
        attack_ms=cfg.glue_attack_ms,
        release_ms=cfg.glue_release_ms,
        makeup_db=0.0,
        knee_db=6.0,   # wide knee — bus compression should be invisible
    ))
    work = work * (gain[:, None] if stereo else gain)
    work *= 10 ** (cfg.glue_makeup_db / 20.0)

    # ── 2. Lookahead limiter to the ceiling ────────────────────────────────
    block = max(1, int(sr * 0.001))
    n_blocks = int(np.ceil(len(work) / block))
    flat = np.abs(work).max(axis=1) if stereo else np.abs(work)
    padded = np.zeros(n_blocks * block)
    padded[: len(flat)] = flat
    peaks = padded.reshape(n_blocks, block).max(axis=1)

    g_blocks = _limiter_gain(peaks, block, sr, cfg.ceiling, cfg.limiter_release_ms)
    block_pos = (np.arange(n_blocks) + 0.5) * block
    g = np.interp(np.arange(len(work)), block_pos, g_blocks)
    work = work * (g[:, None] if stereo else g)

    # Hard safety clip — the interpolated gain can overshoot by a hair
    work = np.clip(work, -cfg.ceiling, cfg.ceiling)
    return work.astype(np.float32, copy=False)
