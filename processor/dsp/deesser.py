"""
Improved de-esser for sibilance reduction.
"""
import numpy as np
from scipy.signal import butter, lfilter


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply bandpass filter."""
    ny = sr / 2
    b, a = butter(order, [low / ny, high / ny], btype='band')
    return lfilter(b, a, x)


def apply_deesser(x: np.ndarray, sr: int, thresh_db: float = -30, ratio: float = 4.0) -> np.ndarray:
    """
    Apply de-esser to reduce sibilance.
    
    Args:
        x: Input audio
        sr: Sample rate
        thresh_db: Threshold in dB for sibilance detection
        ratio: Compression ratio for sibilant regions
    
    Returns:
        De-essed audio
    """
    # Focus on sibilance band (5-10kHz)
    s = bandpass(x, sr, 5000, 10000)
    env = np.abs(s)
    env_db = 20 * np.log10(np.maximum(env, 1e-9))
    
    mask = env_db > thresh_db
    
    # Compute local gain per sample
    gain = np.ones_like(x)
    
    if mask.any() and env_db.max() > thresh_db:
        # Simple: reduce amplitude where band energy high
        reduction = (env_db - thresh_db) / (env_db.max() + 1e-9)
        reduction = np.clip(reduction, 0, 1)
        
        # Map to gain attenuation based on ratio (clamp to prevent complete silence)
        attenuation = 1.0 / (1.0 + (ratio - 1.0) * reduction)
        attenuation = np.clip(attenuation, 0.1, 1.0)  # Never reduce below 10%
        
        # Smooth and apply to whole signal (conservative)
        window_size = min(256, len(attenuation) // 4)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            att_smooth = np.convolve(attenuation, kernel, mode='same')
        else:
            att_smooth = attenuation
        
        gain *= att_smooth
    
    result = x * gain
    
    # Safety check: ensure we didn't silence the audio
    if np.max(np.abs(result)) < np.max(np.abs(x)) * 0.01:
        # If de-esser made audio 100x quieter, something went wrong - return original
        return x
    
    return result

