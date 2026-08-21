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


def measure_sibilance_db(x: np.ndarray, sr: int) -> float:
    """Energy of the sibilance band relative to the full signal, in dB.
    Used to match the de-esser's depth to the reference's actual character."""
    sos = butter(4, 5500.0 / (sr / 2.0), btype="high", output="sos")
    high = sosfilt(sos, np.asarray(x, dtype=np.float64))
    full = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) + 1e-12
    band = float(np.sqrt(np.mean(high ** 2))) + 1e-12
    return 20.0 * np.log10(band / full)


def apply_deesser(x: np.ndarray, sr: int, thresh_db: float = -26.0, ratio: float = 3.0,
                  ref_sibilance_db: float | None = None) -> np.ndarray:
    """
    Reduce sibilance without touching the rest of the spectrum.

    Split at 5.5 kHz; the high band is compressed (fast attack, short release)
    with a gain curve computed from its own energy; low band passes untouched.

    ref_sibilance_db: the reference's measured sibilance-to-full ratio. When
    given, the ratio adapts: if the dry vocal is already no more sibilant
    than the reference, de-essing goes gentle (1.5:1); if it is much more
    sibilant, it deepens (up to 5:1).
    """
    if ref_sibilance_db is not None:
        own = measure_sibilance_db(x, sr)
        excess = own - ref_sibilance_db  # +ve = we are more sibilant than ref
        ratio = float(np.clip(1.5 + excess * 0.5, 1.5, 5.0))
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
