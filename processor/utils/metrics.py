"""
Metrics computation utilities for audio processing.

Provides:
- compute_spectral_distance() - Compare two audio files
- compute_lufs() - Compute LUFS loudness
- timer() - Context manager for timing
"""
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str = "operation"):
    """
    Context manager for timing operations.
    
    Usage:
        with timer("RVC processing"):
            result = process_audio()
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.debug(f"{name} took {elapsed:.3f}s")


def compute_spectral_distance(
    path_a: Path | str,
    path_b: Path | str,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """
    Compute spectral distance between two audio files.
    
    Uses log1p-STFT difference as distance metric.
    
    Args:
        path_a: Path to first audio file
        path_b: Path to second audio file
        sr: Sample rate for comparison
        n_fft: FFT size
        hop_length: Hop length for STFT
        
    Returns:
        Spectral distance (lower is more similar, typically 0-100)
    """
    try:
        # Load audio
        audio_a, sr_a = librosa.load(path_a, sr=sr, mono=True)
        audio_b, sr_b = librosa.load(path_b, sr=sr, mono=True)
        
        # Ensure same length
        min_len = min(len(audio_a), len(audio_b))
        audio_a = audio_a[:min_len]
        audio_b = audio_b[:min_len]
        
        # Compute STFT
        stft_a = librosa.stft(audio_a, n_fft=n_fft, hop_length=hop_length)
        stft_b = librosa.stft(audio_b, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to magnitude and apply log1p
        mag_a = np.abs(stft_a)
        mag_b = np.abs(stft_b)
        
        log_mag_a = np.log1p(mag_a)
        log_mag_b = np.log1p(mag_b)
        
        # Compute L1 distance (mean absolute difference)
        distance = np.mean(np.abs(log_mag_a - log_mag_b))
        
        return float(distance)
        
    except Exception as e:
        logger.warning(f"Failed to compute spectral distance: {e}")
        return float('inf')


def compute_lufs(
    path: Path | str,
    sr: Optional[int] = None,
) -> float:
    """
    Compute LUFS (Loudness Units relative to Full Scale) using ITU-R BS.1770.
    
    This is a simplified implementation. For production, use pyloudnorm or similar.
    
    Args:
        path: Path to audio file
        sr: Sample rate (auto-detect if None)
        
    Returns:
        LUFS value (typically -60 to 0, where 0 is maximum)
    """
    try:
        # Load audio
        audio, file_sr = librosa.load(path, sr=sr, mono=True)
        
        if sr and sr != file_sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        
        # Simplified LUFS calculation
        # Real implementation would use K-weighting filter
        # This is an approximation using RMS
        
        # Compute RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Convert to dB
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -120  # Very quiet
        
        # Approximate LUFS (RMS-based, not true LUFS)
        # True LUFS would be ~3-6 dB higher typically
        lufs = db + 3.0  # Rough approximation
        
        return float(lufs)
        
    except Exception as e:
        logger.warning(f"Failed to compute LUFS: {e}")
        return float('-inf')


def compute_crest_factor(audio: np.ndarray) -> float:
    """
    Compute crest factor (peak-to-RMS ratio).
    
    Args:
        audio: Audio signal
        
    Returns:
        Crest factor (typically 3-20 for vocals)
    """
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    if rms > 0:
        return float(peak / rms)
    return 0.0


def compute_dynamic_range(audio: np.ndarray) -> float:
    """
    Compute dynamic range (difference between peak and RMS in dB).
    
    Args:
        audio: Audio signal
        
    Returns:
        Dynamic range in dB
    """
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    if rms > 0 and peak > 0:
        rms_db = 20 * np.log10(rms)
        peak_db = 20 * np.log10(peak)
        return float(peak_db - rms_db)
    return 0.0

