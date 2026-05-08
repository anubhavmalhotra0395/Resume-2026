"""
Vocal Doubler analysis — detect doubling from stereo spread and pitch modulation.
Returns None if reference appears mono or has minimal spread.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


def detect_doubler(reference_audio: np.ndarray, sr: int) -> Optional["DoublerSettings"]:
    """
    Measure side/mid energy ratio to detect whether a doubler was applied.
    If side/mid energy > 0.15, doubling is assumed.
    """
    from processor.dsp.doubler import DoublerSettings

    if reference_audio.ndim == 2 and reference_audio.shape[0] == 2:
        mid  = (reference_audio[0] + reference_audio[1]) * 0.5
        side = (reference_audio[0] - reference_audio[1]) * 0.5
    else:
        # Mono — no stereo spread to measure
        return None

    mid_energy  = float(np.mean(mid  ** 2)) + 1e-12
    side_energy = float(np.mean(side ** 2))
    spread_ratio = side_energy / mid_energy

    if spread_ratio < 0.15:
        return None  # near-mono, no doubler

    # Estimate detune depth from pitch modulation in the mid channel
    detune_cents = 8.0  # default
    try:
        f0, voiced_flag, _ = librosa.pyin(
            mid,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag & np.isfinite(f0)]
        if len(f0_voiced) > 10:
            semitone_hz = np.diff(f0_voiced)
            pitch_jitter = float(np.std(semitone_hz) / (np.mean(f0_voiced) + 1e-6) * 1200)
            detune_cents = float(np.clip(pitch_jitter * 2.0, 4.0, 20.0))
    except Exception:
        pass

    mix = float(np.clip(spread_ratio * 1.5, 0.2, 0.5))

    return DoublerSettings(
        detune_cents_l=-detune_cents,
        detune_cents_r=detune_cents,
        delay_ms_l=18.0,
        delay_ms_r=22.0,
        mix=mix,
    )
