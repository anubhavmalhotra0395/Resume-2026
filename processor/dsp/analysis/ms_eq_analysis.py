"""
Mid-Side EQ analysis — compare mid and side spectral envelopes.
If they differ by > 2 dB in any band, M-S EQ was likely applied.
Returns None for mono inputs or when mid/side spectra are similar.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import librosa

from processor.dsp.eq import EqBand


def detect_ms_eq(reference_audio: np.ndarray, sr: int) -> Optional["MsEqSettings"]:
    """
    Detect M-S EQ from stereo reference by comparing mid vs side frequency content.
    Returns None if input is mono or spectra are too similar.
    """
    from processor.dsp.ms_eq import MsEqSettings

    if reference_audio.ndim != 2 or reference_audio.shape[0] != 2:
        return None

    left, right = reference_audio[0], reference_audio[1]

    mid  = (left + right) * 0.5
    side = (left - right) * 0.5

    side_energy = float(np.mean(side ** 2))
    mid_energy  = float(np.mean(mid  ** 2)) + 1e-12
    if side_energy / mid_energy < 0.02:
        return None  # near-mono, skip

    n_fft = 2048
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def _mag(y: np.ndarray) -> np.ndarray:
        stft = np.abs(librosa.stft(y, n_fft=n_fft))
        return np.mean(stft, axis=1)

    mid_mag  = _mag(mid)
    side_mag = _mag(side)

    # Analyse 5 octave bands
    band_edges = [100, 300, 1000, 3000, 8000, min(16000, sr // 2 - 100)]
    mid_bands:  List[EqBand] = []
    side_bands: List[EqBand] = []

    for i in range(len(band_edges) - 1):
        lo, hi = band_edges[i], band_edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue

        m_db  = float(20 * np.log10(np.mean(mid_mag[mask])  + 1e-9))
        s_db  = float(20 * np.log10(np.mean(side_mag[mask]) + 1e-9))
        diff_db = s_db - m_db
        fc = float(np.sqrt(lo * hi))
        q  = 0.7

        # Only add a band if mid/side differ by more than 2 dB
        if abs(diff_db) < 2.0:
            continue

        # The mid needs the opposite correction: if side is boosted, mid is cut
        mid_gain  = float(np.clip(-diff_db * 0.5, -6.0, 6.0))
        side_gain = float(np.clip( diff_db * 0.5, -6.0, 6.0))

        if abs(mid_gain) > 0.5:
            mid_bands.append(EqBand(f=fc, gain_db=mid_gain, q=q))
        if abs(side_gain) > 0.5:
            side_bands.append(EqBand(f=fc, gain_db=side_gain, q=q))

    if not mid_bands and not side_bands:
        return None

    return MsEqSettings(mid_bands=mid_bands, side_bands=side_bands)
