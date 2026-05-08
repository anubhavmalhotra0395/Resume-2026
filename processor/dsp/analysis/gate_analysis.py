"""
Noise gate analysis — estimate threshold from the reference vocal noise floor.
Returns None if the reference is already very clean (floor < -55 dBFS).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


def detect_gate(reference_audio: np.ndarray, sr: int) -> Optional["GateSettings"]:
    """
    Measure the noise floor in quiet inter-word gaps and return GateSettings
    that would replicate the gating level found in the reference.
    Returns None if the reference floor is already below -55 dBFS (no gate needed).
    """
    from processor.dsp.gate import GateSettings

    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
    else:
        mono = reference_audio

    # Frame-wise RMS
    frame_len = 2048
    hop = 512
    rms = librosa.feature.rms(y=mono, frame_length=frame_len, hop_length=hop)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))

    # ── Find the noise floor from GENUINE SILENT GAPS only ──────────────────
    # Step 1: find the vocal signal ceiling (90th percentile loud frames)
    loud_db = float(np.percentile(rms_db, 90))
    # Step 2: quiet frames are those at least 30 dB below the loudest frames
    quiet_threshold = loud_db - 30.0
    quiet_frames = rms_db[rms_db < quiet_threshold]

    if len(quiet_frames) < 10:
        # Not enough silence to measure a meaningful gate — skip
        return None

    # Noise floor = median of quiet frames (robust to outliers)
    floor_db = float(np.median(quiet_frames))

    # ── Skip if already very clean ─────────────────────────────────────────
    # -55 dBFS noise floor means the room/mic is already quiet enough
    if floor_db < -55.0:
        return None

    # ── Only apply gate if there is a meaningful noise problem ────────────
    # If silence sits between -55 and -45 dBFS, a gentle gate is appropriate.
    # Below -45 we skip — the reference is already clean.
    if floor_db < -45.0:
        return None

    # Gate opens 10 dB above measured floor, conservatively capped at -35 dBFS
    # (never above -35 — that would chop off soft sung notes)
    threshold_db = float(np.clip(floor_db + 10.0, -60.0, -35.0))

    # Estimate attack/release from transition shape around gate threshold
    thresh_lin = 10 ** (threshold_db / 20.0)
    above = rms > thresh_lin

    # Find rising and falling edges
    edges_up   = np.where(np.diff(above.astype(int)) > 0)[0]
    edges_down = np.where(np.diff(above.astype(int)) < 0)[0]

    attack_ms  = 2.0
    release_ms = 100.0

    if len(edges_up) > 0:
        # Average attack: frames from below→above threshold
        slopes = []
        for idx in edges_up[:5]:
            start = max(0, idx - 3)
            if rms[start] > 0 and rms[min(idx + 1, len(rms) - 1)] > 0:
                delta_db = rms_db[min(idx + 1, len(rms) - 1)] - rms_db[start]
                frames = max(1, idx - start)
                ms_per_frame = hop / sr * 1000.0
                if delta_db > 1:
                    slopes.append(frames * ms_per_frame)
        if slopes:
            attack_ms = float(np.clip(np.mean(slopes), 0.5, 20.0))

    if len(edges_down) > 0:
        slopes = []
        for idx in edges_down[:5]:
            end = min(len(rms) - 1, idx + 5)
            delta_db = rms_db[idx] - rms_db[end]
            frames = max(1, end - idx)
            ms_per_frame = hop / sr * 1000.0
            if delta_db > 1:
                slopes.append(frames * ms_per_frame)
        if slopes:
            release_ms = float(np.clip(np.mean(slopes), 20.0, 300.0))

    return GateSettings(
        threshold_db=threshold_db,
        attack_ms=attack_ms,
        release_ms=release_ms,
        hold_ms=50.0,
    )
