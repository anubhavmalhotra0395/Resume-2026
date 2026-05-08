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


def apply_compressor(x: np.ndarray, sr: int, cfg: CompressorSettings) -> np.ndarray:
    """
    Feed-forward compressor with soft knee and RMS detector.
    """
    # Detector settings
    attack_coeff = np.exp(-1.0 / (0.001 * cfg.attack_ms * sr))
    release_coeff = np.exp(-1.0 / (0.001 * cfg.release_ms * sr))

    env = 0.0
    gain = np.zeros_like(x)

    thr = cfg.threshold_db
    knee = cfg.knee_db
    ratio = cfg.ratio

    for i, sample in enumerate(x):
        rect = sample * sample  # power detector
        rect = np.sqrt(rect)

        if rect > env:
            env = attack_coeff * env + (1 - attack_coeff) * rect
        else:
            env = release_coeff * env + (1 - release_coeff) * rect

        env_db = _lin_to_db(np.array([env]))[0]

        # Soft knee
        if knee > 0:
            lower = thr - knee / 2
            upper = thr + knee / 2
            if env_db < lower:
                gain_db = env_db
            elif env_db > upper:
                gain_db = thr + (env_db - thr) / ratio
            else:
                # Within knee region
                t = (env_db - lower) / knee
                soft = env_db + (1 / ratio - 1) * (env_db - thr) * t * t / 2
                gain_db = soft
        else:
            if env_db > thr:
                gain_db = thr + (env_db - thr) / ratio
            else:
                gain_db = env_db

        gain[i] = _db_to_lin(gain_db - env_db)

    y = x * gain
    if cfg.makeup_db:
        y *= _db_to_lin(cfg.makeup_db)
    return y.astype(x.dtype)


