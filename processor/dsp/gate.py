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

    attack_coef  = np.exp(-1.0 / (sr * cfg.attack_ms  / 1000.0 + 1e-9))
    release_coef = np.exp(-1.0 / (sr * cfg.release_ms / 1000.0 + 1e-9))
    hold_samples = int(sr * cfg.hold_ms / 1000.0)

    env = 0.0
    hold_counter = 0
    gain = np.ones(len(x), dtype=np.float32)

    for i in range(len(x)):
        level = abs(x[i])
        if level >= threshold_lin:
            env = attack_coef * env + (1.0 - attack_coef) * 1.0
            hold_counter = hold_samples
        else:
            if hold_counter > 0:
                hold_counter -= 1
                env = attack_coef * env + (1.0 - attack_coef) * 1.0
            else:
                env = release_coef * env
        gain[i] = float(env)

    return (x * gain).astype(np.float32)
