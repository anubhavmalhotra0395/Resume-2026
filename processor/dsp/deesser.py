"""
Split-band de-esser: compresses ONLY the sibilance band and sums it back.

The previous version computed a brightness-keyed gain and applied it to the
FULL-BAND signal — any 5-10 kHz energy above -30 dB ducked the entire vocal
by up to 4x. That is a pumping "dullness machine", not a de-esser; it was a
major reason processed vocals lost the reference's presence.
"""
import numpy as np
from scipy.signal import butter, sosfilt, lfilter

from processor.dsp.compressor import CompressorSettings, compressor_gain


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply bandpass filter (kept for external callers)."""
    ny = sr / 2
    b, a = butter(order, [low / ny, high / ny], btype='band')
    return lfilter(b, a, x)


def apply_deesser(x: np.ndarray, sr: int, thresh_db: float = -26.0, ratio: float = 3.0) -> np.ndarray:
    """
    Reduce sibilance without touching the rest of the spectrum.

    Split at 5.5 kHz; the high band is compressed (fast attack, short release)
    with a gain curve computed from its own energy; low band passes untouched.
    """
    split_hz = 5500.0
    nyq = sr / 2.0
    sos = butter(4, split_hz / nyq, btype="high", output="sos")
    high = sosfilt(sos, np.asarray(x, dtype=np.float64))
    low = np.asarray(x, dtype=np.float64) - high

    gain = compressor_gain(high.astype(np.float32), sr, CompressorSettings(
        threshold_db=thresh_db,
        ratio=ratio,
        attack_ms=1.5,     # sibilance is fast
        release_ms=60.0,
        makeup_db=0.0,
        knee_db=4.0,
    ))

    result = low + high * gain
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # Safety: never let the de-esser meaningfully change overall level
    if np.max(np.abs(result)) < np.max(np.abs(x)) * 0.5:
        return x
    return result.astype(np.float32, copy=False)
