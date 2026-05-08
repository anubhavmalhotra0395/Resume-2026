"""
Phrase/segment detection for adaptive per-segment DSP processing.
"""
import librosa
import numpy as np
from typing import List, Tuple


def detect_phrases(
    y: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    energy_thresh_db: float = -40,
    min_silence_ms: int = 120,
) -> List[Tuple[int, int]]:
    """
    Detect phrases/segments in audio using energy envelope.
    
    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: STFT hop length
        energy_thresh_db: Energy threshold in dB
        min_silence_ms: Minimum silence duration to split segments (ms)
    
    Returns:
        List of (start_sample, end_sample) tuples
    """
    # Energy envelope based phrase detection
    frame_len = 1024
    hop = hop_length
    
    S = librosa.stft(y, n_fft=2048, hop_length=hop)
    energy = np.log1p((np.abs(S) ** 2).sum(axis=0))
    
    # Normalize
    energy_mean = energy.mean()
    energy_std = energy.std() + 1e-9
    energy = (energy - energy_mean) / energy_std
    
    # Convert to dB
    energy_db = 20 * np.log10(np.abs(energy) + 1e-9)
    
    # Threshold
    threshold = energy_mean + (energy_thresh_db / 20.0)  # Approximate dB threshold
    mask = energy > threshold
    
    # Group contiguous frames into segments
    segments = []
    start = None
    
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            end = i
            segments.append((start * hop, end * hop))
            start = None
    
    if start is not None:
        segments.append((start * hop, len(y)))
    
    # Merge short gaps
    merged = []
    min_silence_samples = int((min_silence_ms / 1000.0) * sr)
    
    for s, e in segments:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s - pe < min_silence_samples:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
    
    # Filter out very short segments (< 100ms)
    min_segment_samples = int(0.1 * sr)
    filtered = [(s, e) for s, e in merged if (e - s) >= min_segment_samples]
    
    return filtered

