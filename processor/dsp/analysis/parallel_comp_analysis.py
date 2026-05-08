"""
Parallel compression analysis — detect upward compression signature via crest factor.
When quiet sections are louder than expected relative to loud sections, a parallel
compressor (New York compression) was likely applied.
Returns None if crest factor is normal.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


def detect_parallel_comp(reference_audio: np.ndarray, sr: int) -> Optional["ParallelCompSettings"]:
    """
    Compare crest factor across frames to infer parallel compression.
    Returns ParallelCompSettings if upward compression signature is detected.
    """
    from processor.dsp.parallel_comp import ParallelCompSettings

    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
    else:
        mono = reference_audio

    # Frame-level RMS
    frame_len = 2048
    hop = 512
    rms = librosa.feature.rms(y=mono, frame_length=frame_len, hop_length=hop)[0]

    if len(rms) < 10:
        return None

    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))

    # Crest factor = peak / RMS; low crest factor = highly compressed
    peak_rms = float(np.percentile(rms_db, 95))
    low_rms  = float(np.percentile(rms_db, 20))
    crest_db = peak_rms - low_rms

    # Natural crest factor for vocals ~15–25 dB; parallel comp reduces to ~8–15 dB
    if crest_db > 15.0:
        return None  # not significantly compressed

    # Estimate blend from how compressed it is
    compression_depth = float(np.clip((15.0 - crest_db) / 10.0, 0.05, 0.5))

    return ParallelCompSettings(
        threshold_db=-30.0,
        ratio=8.0,
        attack_ms=2.0,
        release_ms=150.0,
        blend=compression_depth,
    )
