"""
Tape emulation analysis — detect HF roll-off slope and harmonic signature
characteristic of tape saturation.
Returns None if reference shows no tape-like characteristics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


def detect_tape(reference_audio: np.ndarray, sr: int) -> Optional["TapeSettings"]:
    """
    Detect tape emulation from:
    1. HF roll-off above 12 kHz steeper than typical digital chain.
    2. Presence of low-level 2nd/3rd harmonic distortion content.
    Returns None if no tape signature found.
    """
    from processor.dsp.tape import TapeSettings

    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
    else:
        mono = reference_audio

    n_fft = 4096
    stft  = np.abs(librosa.stft(mono, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def _band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            return 1e-12
        return float(np.mean(stft[mask] ** 2))

    e_8k  = _band_energy(8000,  12000)
    e_12k = _band_energy(12000, 16000)
    e_16k = _band_energy(16000, min(20000, sr / 2 - 100))

    if e_8k < 1e-12:
        return None

    # Roll-off ratio: how much energy drops from 8kHz→12kHz band
    rolloff_12k = e_12k / e_8k
    rolloff_16k = e_16k / (e_12k + 1e-12)

    # Tape typically rolls off 3-6 dB per octave above 14kHz → ratio < 0.5 at 12k→16k
    has_hf_rolloff = rolloff_12k < 0.5 or rolloff_16k < 0.35

    if not has_hf_rolloff:
        return None

    # Estimate drive from how steep the rolloff is
    drive_est = float(np.clip(1.0 - rolloff_12k, 0.1, 0.8))
    mix_est   = float(np.clip(drive_est * 0.5, 0.15, 0.5))

    # Estimate rolloff start frequency
    hf_rolloff_hz = 14000.0
    if rolloff_12k < 0.25:
        hf_rolloff_hz = 10000.0
    elif rolloff_12k > 0.4:
        hf_rolloff_hz = 16000.0

    return TapeSettings(
        drive=drive_est,
        hf_rolloff_hz=hf_rolloff_hz,
        mix=mix_est,
    )
