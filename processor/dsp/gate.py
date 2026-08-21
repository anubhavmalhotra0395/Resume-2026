"""
Noise Gate — removes room noise and breath between words.
Applied as the first stage in the vocal chain.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter


@dataclass
class GateSettings:
    threshold_db: float = -50.0
    attack_ms: float = 2.0
    release_ms: float = 100.0
    hold_ms: float = 50.0


def apply_gate(x: np.ndarray, sr: int, cfg: GateSettings) -> np.ndarray:
    """
    Apply a noise gate. Signal below threshold_db is attenuated.
    Uses smooth attack/release envelope following to avoid clicking.
    """
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    threshold_lin = 10 ** (cfg.threshold_db / 20.0)

    # Block-based (1 ms) envelope follower — same behaviour as the old
    # per-sample loop but ~100x faster on minutes-long vocals.
    block = max(1, int(sr * 0.001))
    n_blocks = int(np.ceil(len(x) / block))
    padded = np.zeros(n_blocks * block, dtype=np.float64)
    padded[: len(x)] = np.abs(np.asarray(x, dtype=np.float64))
    level = padded.reshape(n_blocks, block).max(axis=1)  # peak per block

    attack_coef = float(np.exp(-block / (sr * cfg.attack_ms / 1000.0 + 1e-9)))
    release_coef = float(np.exp(-block / (sr * cfg.release_ms / 1000.0 + 1e-9)))
    hold_blocks = max(0, int(cfg.hold_ms / 1000.0 * sr / block))

    open_ = level >= threshold_lin
    env_blocks = np.empty(n_blocks)
    env = 0.0
    hold = 0
    for i in range(n_blocks):
        if open_[i]:
            env = attack_coef * env + (1.0 - attack_coef)
            hold = hold_blocks
        elif hold > 0:
            hold -= 1
            env = attack_coef * env + (1.0 - attack_coef)
        else:
            env = release_coef * env
        env_blocks[i] = env

    block_pos = (np.arange(n_blocks) + 0.5) * block
    gain = np.interp(np.arange(len(x)), block_pos, env_blocks)
    return (x * gain).astype(np.float32)
