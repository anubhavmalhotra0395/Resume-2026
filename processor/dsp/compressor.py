from dataclasses import dataclass

import numpy as np


@dataclass
class CompressorSettings:
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_db: float = 0.0
    knee_db: float = 3.0


def _db_to_lin(db: np.ndarray) -> np.ndarray:
    return 10 ** (db / 20.0)


def _lin_to_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-12))


def _gain_curve(x: np.ndarray, sr: int, cfg: CompressorSettings) -> np.ndarray:
    """Per-sample gain envelope: block-RMS detector, attack/release smoothing
    in dB at 1 ms block rate, soft-knee gain computer, linear interpolation."""
    block = max(1, int(sr * 0.001))  # 1 ms detector hop
    n_blocks = int(np.ceil(len(x) / block))
    padded = np.zeros(n_blocks * block, dtype=np.float64)
    padded[: len(x)] = np.asarray(x, dtype=np.float64)

    rms = np.sqrt(np.mean(padded.reshape(n_blocks, block) ** 2, axis=1))
    rms_db = _lin_to_db(rms)

    attack_coeff = float(np.exp(-block / (0.001 * max(cfg.attack_ms, 0.1) * sr)))
    release_coeff = float(np.exp(-block / (0.001 * max(cfg.release_ms, 1.0) * sr)))
    env_db = np.empty(n_blocks)
    e = rms_db[0]
    for i in range(n_blocks):
        v = rms_db[i]
        c = attack_coeff if v > e else release_coeff
        e = c * e + (1.0 - c) * v
        env_db[i] = e

    thr, knee, ratio = cfg.threshold_db, cfg.knee_db, max(cfg.ratio, 1.0)
    over = env_db - thr
    gain_db = np.zeros(n_blocks)
    if knee > 0:
        in_knee = np.abs(over) <= knee / 2
        above = over > knee / 2
        gain_db[in_knee] = (1.0 / ratio - 1.0) * (over[in_knee] + knee / 2) ** 2 / (2.0 * knee)
        gain_db[above] = (1.0 / ratio - 1.0) * over[above]
    else:
        above = over > 0
        gain_db[above] = (1.0 / ratio - 1.0) * over[above]

    block_pos = (np.arange(n_blocks) + 0.5) * block
    return np.interp(np.arange(len(x)), block_pos, _db_to_lin(gain_db))


def apply_compressor(x: np.ndarray, sr: int, cfg: CompressorSettings) -> np.ndarray:
    """
    Feed-forward compressor: short-window RMS detector, attack/release
    smoothing at 1 ms block rate, soft-knee gain computer, linear-interpolated
    gain applied to the full-rate signal.

    The previous per-sample implementation tracked |x| directly, so its
    envelope pumped within each waveform cycle and undershot on transients —
    measured on a burst signal it *raised* the crest factor from 8 dB to
    21 dB (a compressor must lower it) while running at 0.5x realtime.
    """
    if len(x) == 0:
        return x
    y = np.asarray(x, dtype=np.float64) * _gain_curve(x, sr, cfg)
    if cfg.makeup_db:
        y *= _db_to_lin(cfg.makeup_db)
    return y.astype(np.float32, copy=False)


def compressor_gain(sidechain: np.ndarray, sr: int, cfg: CompressorSettings) -> np.ndarray:
    """The gain curve alone, computed from `sidechain` — for linked-stereo
    use: detect on a mono downmix, apply the identical gain to each channel."""
    if len(sidechain) == 0:
        return np.ones(0)
    return _gain_curve(sidechain, sr, cfg)
