"""
Tape Emulation — soft saturation + high-frequency roll-off to mimic analogue tape warmth.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class TapeSettings:
    drive: float = 0.3          # 0.0–1.0 saturation amount
    hf_rolloff_hz: float = 14000.0  # start of HF roll-off
    mix: float = 0.4            # wet/dry blend


def apply_tape(x: np.ndarray, sr: int, cfg: TapeSettings) -> np.ndarray:
    """
    Tape emulation: slightly asymmetric tanh saturation (emphasises 2nd harmonic) +
    gentle low-pass above hf_rolloff_hz, blended at mix.
    """
    if cfg.mix <= 0:
        return x

    drive = float(np.clip(cfg.drive, 0.01, 1.0))
    gain  = 1.0 + drive * 3.0

    # Slightly asymmetric tanh → more even-order (2nd) harmonics like tape
    warm = (np.tanh(x * gain * 1.05) * 0.6 + np.tanh(x * gain * 0.95) * 0.4) / gain

    # High-frequency roll-off above hf_rolloff_hz
    nyq = sr / 2.0
    rolloff = min(float(cfg.hf_rolloff_hz), nyq * 0.95)
    if rolloff < nyq * 0.95:
        sos = butter(2, rolloff / nyq, btype="low", output="sos")
        warm = sosfilt(sos, warm).astype(np.float32)

    mix = float(np.clip(cfg.mix, 0.0, 1.0))
    result = (1.0 - mix) * x + mix * warm
    return result.astype(np.float32)
