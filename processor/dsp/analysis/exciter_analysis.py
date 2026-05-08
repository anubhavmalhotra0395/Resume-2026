"""
Exciter analysis — detect disproportionate HF harmonic energy in the reference vocal.
Returns None if no evidence of HF harmonic enhancement.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


def detect_exciter(reference_audio: np.ndarray, sr: int) -> Optional["ExciterSettings"]:
    """
    Compare high-frequency (8–16 kHz) energy to a smoothed low-frequency baseline.
    If reference has significantly more HF energy than expected from the fundamentals
    alone, an exciter/harmonic enhancer was likely used.
    """
    from processor.dsp.exciter import ExciterSettings

    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
    else:
        mono = reference_audio

    # Compute full power spectrum
    n_fft = 2048
    stft = np.abs(librosa.stft(mono, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Energy in bands
    def _band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            return 0.0
        return float(np.mean(stft[mask] ** 2))

    mid_energy = _band_energy(1000, 6000)
    hf_energy  = _band_energy(8000, min(16000, sr / 2 - 100))

    if mid_energy < 1e-12:
        return None

    # Expected HF falloff ratio (natural roll-off ~ -6 dB per octave from 6kHz to 12kHz)
    expected_ratio = 0.25
    actual_ratio   = hf_energy / mid_energy

    excess_ratio = actual_ratio / (expected_ratio + 1e-9)

    # If actual HF is more than 40% above expected, an exciter was used
    if excess_ratio < 1.4:
        return None

    drive = float(np.clip((excess_ratio - 1.4) * 0.4, 0.1, 0.8))
    mix   = float(np.clip((excess_ratio - 1.0) * 0.1, 0.1, 0.35))

    return ExciterSettings(
        drive=drive,
        mix=mix,
        freq_hz=6000.0,
    )
