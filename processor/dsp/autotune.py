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

    # Snap each voiced frame to the nearest in-key note
    target_midi = np.copy(midi)
    for i in range(len(midi)):
        if voiced_mask[i] and np.isfinite(midi[i]):
            target_midi[i] = _nearest_scale_note(midi[i], scale_notes)

    # Per-frame correction in semitones. (The previous implementation
    # computed this too, then collapsed it to ONE median shift for the whole
    # track — a static transpose, i.e. no per-note tuning ever happened.)
    shift = np.where(
        voiced_mask & np.isfinite(midi) & np.isfinite(target_midi),
        (target_midi - midi) * float(np.clip(strength, 0.0, 1.0)),
        0.0,
    )
    shift = np.nan_to_num(shift, nan=0.0)
    # Ignore octave-level detection glitches; tuning corrects small errors
    shift = np.where(np.abs(shift) > 1.5, 0.0, shift)
    if not np.any(np.abs(shift) > 0.02):
        return y

    HOP = 256  # estimate_f0 hop

    # ── Segment into notes: contiguous runs needing the same correction ──
    # Hard tune (small retune_ms) keeps segments exact; softer settings
    # smooth the shift curve first so corrections glide instead of snapping.
    retune_frames = max(1, int(retune_ms / 1000.0 * sr / HOP))
    if retune_frames > 1 and strength < 0.9:
        k = np.ones(retune_frames) / retune_frames
        shift = np.convolve(shift, k, mode="same")

    segments = []  # (start_frame, end_frame, shift_semitones)
    MIN_FRAMES = max(3, int(0.06 * sr / HOP))  # >= 60 ms per tuned piece
    i = 0
    n = len(shift)
    while i < n:
        if abs(shift[i]) < 0.03:
            i += 1
            continue
        j = i + 1
        while j < n and abs(shift[j] - shift[i]) < 0.35 and abs(shift[j]) >= 0.03:
            j += 1
        if j - i >= MIN_FRAMES:
            segments.append((i, j, float(np.median(shift[i:j]))))
        i = j
    if not segments:
        return y

    # Merge segments so each pitch_shift call gets enough context and the
    # total call count stays sane on long vocals
    merged = [list(segments[0])]
    for s0, s1, sh in segments[1:]:
        if s0 - merged[-1][1] <= MIN_FRAMES and abs(sh - merged[-1][2]) < 0.2:
            merged[-1][1] = s1
        else:
            merged.append([s0, s1, sh])

    # ── Apply: pitch-shift each note region, equal-power crossfade back ──
    out = np.array(y, dtype=np.float64, copy=True)
    fade = max(32, int(0.012 * sr))  # 12 ms joins
    for s0, s1, sh in merged:
        a = max(0, s0 * HOP - fade)
        b = min(len(y), s1 * HOP + fade)
        if b - a < fade * 3:
            continue
        piece = np.asarray(y[a:b], dtype=np.float32)
        try:
            tuned = librosa.effects.pitch_shift(piece, sr=sr, n_steps=float(sh),
                                                bins_per_octave=12)
        except Exception:
            continue
        if len(tuned) != len(piece):
            tuned = tuned[: len(piece)] if len(tuned) > len(piece) else np.pad(
                tuned, (0, len(piece) - len(tuned)))
        w = np.ones(len(piece))
        ramp = np.linspace(0.0, 1.0, fade)
        w[:fade] = ramp
        w[-fade:] = ramp[::-1]
        out[a:b] = out[a:b] * (1.0 - w) + tuned.astype(np.float64) * w

    return out.astype(np.float32)
