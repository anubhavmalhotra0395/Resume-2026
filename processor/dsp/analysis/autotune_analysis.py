"""
Autotune analysis — detect pitch correction from semitone clustering in the reference vocal.

Detection logic:
  - Use librosa.pyin on the reference vocal to get voiced pitch frames.
  - Compute how tightly frames cluster to chromatic semitones (cents deviation).
  - If median deviation < 15 cents → pitch correction was applied.
  - Estimate strength (how aggressive) and retune_ms (how fast).
  - Returns AutotuneSettings (auto-applied), or None if no correction detected.

The KEY used for correction is detected from the dry vocal at apply time,
not from the reference — so notes snap to the dry vocal's own scale.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


@dataclass
class AutotuneSettings:
    strength: float = 0.7   # 0.0–1.0
    retune_ms: float = 20.0  # how fast correction applies (ms)


def detect_autotune(reference_audio: np.ndarray, sr: int) -> Optional[AutotuneSettings]:
    """
    Measure how tightly voiced frames cluster to semitone boundaries using librosa.pyin.

    Returns AutotuneSettings if the reference vocal appears pitch-corrected
    (median deviation from nearest semitone < 15 cents), None otherwise.

    The returned settings carry strength/speed derived from the reference.
    The KEY is not stored here — it is detected from the dry vocal at apply time.
    """
    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
    else:
        mono = reference_audio

    # Limit to 30s for speed — pyin is very slow on long audio
    _MAX_SAMPLES = sr * 30
    if len(mono) > _MAX_SAMPLES:
        # Take from the middle of the track (more likely to have vocals)
        start = (len(mono) - _MAX_SAMPLES) // 2
        mono = mono[start: start + _MAX_SAMPLES]

    # Downsample to 16kHz for pyin — pitch detection doesn't need full SR
    _TARGET_SR = 16000
    if sr != _TARGET_SR:
        import librosa as _lb
        mono = _lb.resample(mono, orig_sr=sr, target_sr=_TARGET_SR)
        _pyin_sr = _TARGET_SR
    else:
        _pyin_sr = sr

    try:
        f0, voiced_flag, _ = librosa.pyin(
            mono,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=_pyin_sr,
            hop_length=128,
        )
    except Exception:
        return None

    voiced = f0[voiced_flag & np.isfinite(f0) & (f0 > 0)]
    if len(voiced) < 20:
        return None

    # Convert to MIDI semitones
    midi = librosa.hz_to_midi(voiced)
    # Deviation from nearest chromatic semitone (in cents: 0–50)
    deviations_cents = np.abs(midi - np.round(midi)) * 100.0
    median_dev = float(np.median(deviations_cents))

    # Natural unprocessed singing: median deviation typically 20–50 cents
    # Auto-tuned singing: deviation typically < 15 cents
    if median_dev > 15.0:
        return None  # vocal not pitch-corrected in reference

    # Strength: how tightly notes are locked to grid (lower dev = stronger correction)
    strength = float(np.clip(1.0 - median_dev / 15.0, 0.3, 1.0))

    # Retune speed: estimated from how fast pitch transitions happen
    retune_ms = 20.0
    try:
        hop_ms = 256.0 / sr * 1000.0
        midi_diff = np.abs(np.diff(midi))
        # Large jumps (> 0.5 semitone) indicate a retune event
        jump_frames = midi_diff[midi_diff > 0.5]
        if len(jump_frames) > 3:
            avg_frames_between_jumps = len(midi) / max(len(jump_frames), 1)
            retune_ms = float(np.clip(avg_frames_between_jumps * hop_ms * 0.4, 5.0, 80.0))
    except Exception:
        pass

    return AutotuneSettings(strength=strength, retune_ms=retune_ms)
