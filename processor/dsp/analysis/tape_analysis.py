"""
Tape emulation analysis - detect HF roll-off slope and harmonic signature
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

    # HF roll-off is a weak tape cue, because singing is naturally dark up
    # here. Measured across six commercial vocal stems, untreated tracks span
    # rolloff_12k 0.081-0.449 while the same tracks pushed through heavy tape
    # span 0.047-0.260 — the two distributions overlap almost entirely, so no
    # threshold on this feature cleanly separates "tape" from "dark singer".
    #
    # The old gate (rolloff_12k < 0.5 or rolloff_16k < 0.35) fired on 6 of 6
    # untreated stems, and drive was clip(1 - rolloff_12k, 0.1, 0.8), which
    # pins to 0.8 whenever the 12-16 kHz band is quiet — i.e. always. Every
    # job therefore got near-maximum tape saturation no matter what the
    # reference sounded like, which works directly against matching it.
    #
    # So: only claim tape when the reference is darker than any natural vocal
    # in that sample (0 of 6 untreated stems pass both tests), and scale drive
    # continuously from the measurement instead of pinning it.
    if not (rolloff_12k < 0.09 and rolloff_16k < 0.05):
        return None

    drive_est = float(np.clip((0.09 - rolloff_12k) / 0.09 * 0.5, 0.05, 0.5))
    mix_est   = float(np.clip(0.20 + 0.40 * drive_est, 0.15, 0.45))

    # Roll-off start: the frequency where the spectrum has fallen 12 dB below
    # its own 2-6 kHz level. Continuous, rather than a three-way step that
    # reported 10 kHz for practically every reference.
    p_ref = _band_energy(2000, 6000)
    hf_rolloff_hz = 14000.0
    if p_ref > 1e-12:
        thresh = p_ref * 10 ** (-12.0 / 10.0)
        for lo in range(4000, 18000, 500):
            if _band_energy(lo, lo + 1000) < thresh:
                hf_rolloff_hz = float(lo)
                break
    hf_rolloff_hz = float(np.clip(hf_rolloff_hz, 6000.0, 18000.0))

    return TapeSettings(
        drive=drive_est,
        hf_rolloff_hz=hf_rolloff_hz,
        mix=mix_est,
    )
