"""
Multiband compressor for frequency-dependent dynamic processing.
"""
import numpy as np
from scipy.signal import butter, lfilter
from typing import List, Tuple


def butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    """Design bandpass filter."""
    ny = 0.5 * fs
    b, a = butter(order, [low / ny, high / ny], btype='band')
    return b, a


def apply_band(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply bandpass filter."""
    return lfilter(b, a, x)


def simple_compress(x: np.ndarray, threshold_db: float = -18, ratio: float = 3.0) -> np.ndarray:
    """Simple static compressor in dB domain."""
    eps = 1e-9
    mag = np.abs(x)
    db = 20 * np.log10(np.maximum(mag, eps))
    over = db - threshold_db
    gain_db = np.zeros_like(over)
    mask = over > 0
    gain_db[mask] = -(1 - 1 / ratio) * over[mask]
    
    # Clamp gain_db to prevent Inf/NaN
    gain_db = np.clip(gain_db, -60.0, 60.0)
    
    gain = 10 ** (gain_db / 20.0)
    
    # Safety check for NaN/Inf
    gain = np.nan_to_num(gain, nan=1.0, posinf=1.0, neginf=1.0)
    gain = np.clip(gain, 0.0, 10.0)  # Reasonable gain limit
    
    result = x * gain
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def multiband_compress(
    x: np.ndarray,
    sr: int,
    bands: List[Tuple[float, float]] = None,
    thresholds: List[float] = None,
    ratios: List[float] = None,
) -> np.ndarray:
    """
    Apply multiband compression.
    
    Args:
        x: Input audio
        sr: Sample rate
        bands: List of (low_freq, high_freq) tuples
        thresholds: List of threshold_db for each band
        ratios: List of compression ratios for each band
    
    Returns:
        Compressed audio
    """
    if bands is None:
        bands = [(20, 200), (200, 2000), (2000, 12000)]
    if thresholds is None:
        thresholds = [-24, -18, -12]
    if ratios is None:
        ratios = [2, 3, 4]
    
    out = np.zeros_like(x)
    
    for (low, high), th, r in zip(bands, thresholds, ratios):
        try:
            b, a = butter_bandpass(low, high, sr)
            band = apply_band(b, a, x)
            c = simple_compress(band, threshold_db=th, ratio=r)
            # Safety check before adding
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            out += c
        except Exception as e:
            # If band processing fails, skip this band
            continue
    
    # Safety check: if out is all zeros or invalid, return original
    if np.max(np.abs(out)) < 1e-9 or not np.all(np.isfinite(out)):
        return x
    
    # Blend with original to prevent holes
    result = 0.9 * out + 0.1 * x
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final safety: if result is invalid, return original
    if not np.all(np.isfinite(result)) or np.max(np.abs(result)) < np.max(np.abs(x)) * 0.01:
        return x
    
    return result

