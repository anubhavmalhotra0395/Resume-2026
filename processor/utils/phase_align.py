"""
Phase alignment utility to reduce phasing artifacts.
"""
import numpy as np
import librosa
from typing import Tuple


def align_by_crosscorr(ref: np.ndarray, target: np.ndarray, sr: int = 44100, max_shift_ms: float = 50) -> np.ndarray:
    """
    Align target audio to reference using cross-correlation.
    
    Args:
        ref: Reference audio signal
        target: Target audio signal to align
        sr: Sample rate
        max_shift_ms: Maximum shift in milliseconds
    
    Returns:
        Aligned target audio
    """
    # Cross-correlate and align by max correlation within window
    max_shift = int(sr * max_shift_ms / 1000)
    
    # Use shorter segments for efficiency if signals are long
    if len(ref) > 44100 * 10:  # > 10 seconds
        ref_seg = ref[:44100 * 5]  # Use first 5 seconds
        target_seg = target[:44100 * 5]
    else:
        ref_seg = ref
        target_seg = target
    
    corr = np.correlate(target_seg, ref_seg, "full")
    center = len(corr) // 2
    window = corr[center - max_shift : center + max_shift + 1]
    shift = np.argmax(window) - max_shift
    
    # Apply shift to full signal
    if shift > 0:
        aligned = np.concatenate([np.zeros(shift, dtype=target.dtype), target[:-shift]])
    elif shift < 0:
        aligned = target[-shift:]
        aligned = np.concatenate([aligned, np.zeros(-shift, dtype=target.dtype)])
    else:
        aligned = target
    
    # Ensure same length as original
    if len(aligned) != len(target):
        if len(aligned) > len(target):
            aligned = aligned[:len(target)]
        else:
            aligned = np.pad(aligned, (0, len(target) - len(aligned)), mode='constant')
    
    return aligned

