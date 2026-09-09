"""
Autotune analysis - detect pitch correction from semitone clustering in the reference vocal.

Detection logic:
  - Use librosa.pyin on the reference vocal to get voiced pitch frames.
  - Compute how tightly frames cluster to chromatic semitones (cents deviation).
  - If median deviation < 15 cents -> pitch correction was applied.
  - Estimate strength (how aggressive) and retune_ms (how fast).
  - Returns AutotuneSettings (auto-applied), or None if no correction detected.

The KEY used for correction is detected from the dry vocal at apply time,
not from the reference - so notes snap to the dry vocal's own scale.
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
    # How far off the grid the reference itself sits, in cents. This is the
    # tightness to aim for: a hard-tuned lead measures under a cent, a lightly
    # tuned one 10-15. Without it there is no way to tell 'snap everything'
    # from 'nudge the worst notes', and the correction floor cannot adapt.
    ref_cents: float = 10.0
    # Pitch classes (0-11) the REFERENCE actually sings, measured rather than
    # inferred from a key template — see scale_from_reference. None means snap
    # chromatically, which is safe but keeps notes that are wrong for the song.
    scale_pcs: Optional[list] = None


def detect_autotune(reference_audio: np.ndarray, sr: int) -> Optional[AutotuneSettings]:
    """
    Measure how tightly voiced frames cluster to semitone boundaries using librosa.pyin.

    Returns AutotuneSettings if the reference vocal appears pitch-corrected
    (median deviation from nearest semitone < 15 cents), None otherwise.

    The returned settings carry strength/speed derived from the reference.
    The KEY is not stored here - it is detected from the dry vocal at apply time.
    """
    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
    else:
        mono = reference_audio

    # 30s window. Shortening this to 15s to save ~12s of detection time
    # changed the ANSWER: on the same reference it reported strength 0.33
    # instead of 1.00, and the chain then barely tuned at all. Pitch
    # clustering needs enough voiced material to be stable.
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

    return AutotuneSettings(strength=strength, retune_ms=retune_ms,
                            ref_cents=median_dev)


def scale_from_reference(ref_vocal, sr, coverage: float = 0.85,
                         max_notes: int = 8):
    """The note set the reference actually SINGS, as pitch classes 0-11.

    Key detection from a chord template was tried and is not safe here: on the
    Hide reference the mix scored C# major over G# major by a margin of 0.003
    -- a coin flip -- while the vocal alone said D# and the dry said G#.
    Snapping to a wrongly chosen key pulls correct notes to wrong pitches,
    which is why chromatic snapping became the default.

    Measuring which notes the reference uses avoids the guess entirely. Its
    top 7 pitch classes cover 92.1% of all sung frames, so the scale is not
    ambiguous at all -- only the label for it is.

    Returns (pitch_classes, coverage_fraction), or (None, 0.0) when the
    distribution is too flat to call a scale.
    """
    import librosa

    m = np.asarray(ref_vocal, dtype=np.float32)
    if m.ndim > 1:
        m = m.mean(axis=0 if m.shape[0] <= 2 else 1)
    if len(m) < sr:
        return None, 0.0
    try:
        mm = librosa.resample(np.asarray(m, dtype=float), orig_sr=sr,
                              target_sr=16000)
        f0, _, _ = librosa.pyin(mm, fmin=65, fmax=1000, sr=16000,
                                frame_length=2048)
    except Exception:
        return None, 0.0
    f = f0[np.isfinite(f0)]
    if len(f) < 100:
        return None, 0.0

    midi = 69 + 12 * np.log2(f / 440.0)
    hist = np.bincount(np.round(midi).astype(int) % 12, minlength=12).astype(float)
    total = hist.sum()
    if total <= 0:
        return None, 0.0
    hist /= total

    # Take notes in descending use until they explain `coverage` of the singing.
    order = np.argsort(hist)[::-1]
    chosen, acc = [], 0.0
    for idx in order:
        chosen.append(int(idx))
        acc += hist[idx]
        if acc >= coverage or len(chosen) >= max_notes:
            break
    # A flat distribution means chromatic singing (or a bad f0 track): no scale.
    if acc < coverage or len(chosen) >= 11:
        return None, float(acc)
    return sorted(chosen), float(acc)
