"""
Exciter / Harmonic Enhancer — adds synthetic high-frequency harmonics via soft saturation.
Crossover-limited so it only acts above freq_hz to avoid muddying the mids.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class ExciterSettings:
    drive: float = 0.3      # 0.0–1.0
    mix: float = 0.2        # wet/dry blend
    freq_hz: float = 6000.0  # only excite above this frequency


def apply_exciter(x: np.ndarray, sr: int, cfg: ExciterSettings) -> np.ndarray:
    """
    High-pass at freq_hz, apply soft tanh saturation at drive, blend at mix.
    """
    if cfg.mix <= 0:
        return x

    # High-pass filter to isolate high-frequency content
    nyq = sr / 2.0
    freq = min(cfg.freq_hz, nyq * 0.95)
    sos = butter(4, freq / nyq, btype="high", output="sos")
    hf = sosfilt(sos, x).astype(np.float32)

    # Soft saturation — tanh with adjustable drive
    drive = float(np.clip(cfg.drive, 0.01, 1.0))
    gain = 1.0 + drive * 4.0          # boost before tanh so harmonics generate
    saturated = np.tanh(hf * gain) / gain  # normalise back approximately

    # Blend with dry
    mix = float(np.clip(cfg.mix, 0.0, 1.0))
    result = x + mix * saturated
    return result.astype(np.float32)
