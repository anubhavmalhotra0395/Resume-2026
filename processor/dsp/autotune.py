"""
Key-aware autotune — detects the musical key of the dry vocal and snaps pitch
to only the notes in that scale. This prevents out-of-key artefacts.

The autotune_analysis module decides STRENGTH and SPEED from the reference.
This module applies the correction to the dry vocal in its own key.
"""
from __future__ import annotations

import numpy as np
import librosa
from typing import List, Optional


# Chromatic note names
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale intervals relative to root (semitones) for major and natural minor
_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def _detect_key(y: np.ndarray, sr: int) -> tuple[int, str]:
    """
    Detect the musical key (root + mode) of the audio using chromagram.
    Returns (root_semitone, mode) where mode is "major" or "minor".
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)

    best_score = -np.inf
    best_root = 0
    best_mode = "major"

    for root in range(12):
        for mode, intervals in [("major", _MAJOR_INTERVALS), ("minor", _MINOR_INTERVALS)]:
            mask = np.zeros(12)
            for i in intervals:
                mask[(root + i) % 12] = 1.0
            score = float(np.dot(chroma_mean, mask))
            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode

    return best_root, best_mode


def _scale_notes_in_range(root: int, mode: str, lo_midi: int = 36, hi_midi: int = 96) -> List[float]:
    """
    Return all MIDI note numbers belonging to root/mode scale between lo and hi.
    """
    intervals = _MAJOR_INTERVALS if mode == "major" else _MINOR_INTERVALS
    notes = []
    for midi in range(lo_midi, hi_midi + 1):
        if (midi - root) % 12 in intervals:
            notes.append(float(midi))
    return notes


def _nearest_scale_note(midi: float, scale_notes: List[float]) -> float:
    """Snap a MIDI pitch to the nearest note in the scale."""
    arr = np.array(scale_notes)
    idx = int(np.argmin(np.abs(arr - midi)))
    return scale_notes[idx]


def estimate_f0(y: np.ndarray, sr: int) -> np.ndarray:
    """Estimate f0 using YIN; returns Hz array with unvoiced as nan."""
    f0 = librosa.yin(y, fmin=80, fmax=1200, sr=sr, frame_length=2048, hop_length=256)
    f0[f0 <= 0] = np.nan
    return f0


def apply_autotune(
    y: np.ndarray,
    sr: int,
    retune_ms: float = 20.0,
    strength: float = 0.7,
    scale_root: Optional[int] = None,
    scale_mode: Optional[str] = None,
) -> np.ndarray:
    """
    Key-aware autotune.

    1. Detect the musical key from the dry vocal (or use supplied scale_root/mode).
    2. Build the scale note grid for that key.
    3. For each voiced frame, find the nearest *in-key* semitone.
    4. Compute the shift needed and apply at the given strength.

    Args:
        y            : dry mono vocal
        sr           : sample rate
        retune_ms    : how quickly pitch correction is applied (smoothing window)
        strength     : 0–1, how hard to pull toward perfect pitch
        scale_root   : MIDI note class 0–11 (C=0). If None, detected from y.
        scale_mode   : "major" or "minor". If None, detected from y.
    """
    f0 = estimate_f0(y, sr)
    voiced_mask = ~np.isnan(f0)
    if not np.any(voiced_mask):
        return y

    # Detect key from the dry vocal if not supplied
    if scale_root is None or scale_mode is None:
        try:
            scale_root, scale_mode = _detect_key(y, sr)
        except Exception:
            scale_root, scale_mode = 0, "major"

    scale_notes = _scale_notes_in_range(scale_root, scale_mode)
    if not scale_notes:
        return y

    # Convert f0 to MIDI
    midi = np.where(voiced_mask, librosa.hz_to_midi(np.where(voiced_mask, f0, 440.0)), np.nan)

    # Snap each voiced frame to nearest in-key note
    target_midi = np.copy(midi)
    for i in range(len(midi)):
        if voiced_mask[i] and np.isfinite(midi[i]):
            target_midi[i] = _nearest_scale_note(midi[i], scale_notes)

    target_hz = np.where(voiced_mask, librosa.midi_to_hz(np.nan_to_num(target_midi)), f0)

    # Smooth the target over retune_ms to avoid zipper noise
    hop_samples = 256
    retune_samples = max(1, int(sr * retune_ms / 1000.0 / hop_samples))
    if retune_samples > 1:
        kernel = np.ones(retune_samples) / retune_samples
        target_hz = np.convolve(target_hz, kernel, mode="same")

    # Shift in semitones (positive = up, negative = down)
    safe_f0 = np.where(voiced_mask & (f0 > 0), f0, 1.0)
    shift_semitones = 12.0 * np.log2(np.maximum(target_hz, 1.0) / safe_f0)
    shift_semitones[~voiced_mask] = 0.0

    # Apply strength
    applied_shift = shift_semitones * float(np.clip(strength, 0.0, 1.0))

    # Use the median voiced shift for global pitch correction (simple but effective)
    voiced_shifts = applied_shift[voiced_mask & np.isfinite(applied_shift)]
    if len(voiced_shifts) == 0:
        return y
    median_shift = float(np.nanmedian(voiced_shifts))
    if abs(median_shift) < 0.02:
        return y

    y_tuned = librosa.effects.pitch_shift(y, sr=sr, n_steps=median_shift, bins_per_octave=12)

    # Crossfade edges to avoid clicks
    fade_samps = min(int(sr * retune_ms / 1000.0), len(y) // 4)
    if fade_samps > 0:
        window = np.linspace(0.0, 1.0, fade_samps)
        out = y_tuned.copy()
        out[:fade_samps]  = window * y_tuned[:fade_samps]  + (1.0 - window) * y[:fade_samps]
        out[-fade_samps:] = (1.0 - window) * y_tuned[-fade_samps:] + window * y[-fade_samps:]
        return out.astype(np.float32)

    return y_tuned.astype(np.float32)
