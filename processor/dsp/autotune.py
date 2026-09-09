"""
Key-aware autotune — detects the musical key of the dry vocal and snaps pitch
to only the notes in that scale. This prevents out-of-key artefacts.

The autotune_analysis module decides STRENGTH and SPEED from the reference.
This module applies the correction to the dry vocal in its own key.
"""
from __future__ import annotations

import os

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
    if mode == "chromatic":
        # Every semitone: correction is never more than 50 cents and can
        # never pick a wrong pitch class — immune to key-detection errors.
        return [float(m) for m in range(lo_midi, hi_midi + 1)]
    intervals = _MAJOR_INTERVALS if mode == "major" else _MINOR_INTERVALS
    notes = []
    for midi in range(lo_midi, hi_midi + 1):
        if (midi - root) % 12 in intervals:
            notes.append(float(midi))
    return notes


def _notes_from_pitch_classes(pcs, lo_midi: int = 36, hi_midi: int = 96):
    """Every octave of the given pitch classes, as MIDI numbers.

    Used to snap a take to the notes the REFERENCE sings. Chromatic snapping
    cannot fix a note that is in tune but wrong for the song; this can, without
    the risk of a mis-detected key, because the note set is measured from the
    reference rather than inferred from a template.
    """
    keep = {int(p) % 12 for p in (pcs or [])}
    if not keep:
        return []
    return [float(m) for m in range(lo_midi, hi_midi + 1) if (m % 12) in keep]


def _nearest_scale_note(midi: float, scale_notes: List[float]) -> float:
    """Snap a MIDI pitch to the nearest note in the scale."""
    arr = np.array(scale_notes)
    idx = int(np.argmin(np.abs(arr - midi)))
    return scale_notes[idx]


def estimate_f0(y: np.ndarray, sr: int) -> np.ndarray:
    """Estimate f0 using pYIN; returns Hz array with unvoiced frames as nan.

    pYIN rather than YIN because YIN has no voicing decision at all -- it
    returns a pitch for every frame including consonants, breaths and silence.
    On a real vocal it called 3446 of 3446 frames voiced where pYIN found 2082,
    so the corrector was grain-shifting sibilance and room tone as if they were
    sung notes. It also tunes slightly better (9.2 vs 10.8 cents on the same
    take). It costs about 20s on a 2-minute vocal, which is affordable now that
    separation no longer dominates the job.
    """
    try:
        f0, _voiced, _p = librosa.pyin(y, fmin=80, fmax=1200, sr=sr,
                                       frame_length=2048, hop_length=256)
        if np.isfinite(f0).sum() >= 8:
            return f0
    except Exception:
        pass
    f0 = librosa.yin(y, fmin=80, fmax=1200, sr=sr, frame_length=2048, hop_length=256)
    f0[f0 <= 0] = np.nan
    return f0


# Smallest correction worth making, in semitones (12 cents).
_MIN_AUDIBLE_ST = 0.12


# PSOLA is the correction engine; APP_AUTOTUNE_PSOLA=0 falls back to the
# older segment-wise phase-vocoder path.
_USE_PSOLA = os.environ.get("APP_AUTOTUNE_PSOLA", "1") != "0"
# WORLD vocoder resynthesis, tried before PSOLA. Set APP_AUTOTUNE_WORLD=0 to
# A/B against the PSOLA path.
_USE_WORLD = os.environ.get("APP_AUTOTUNE_WORLD", "1") != "0"
# Unvoiced gaps shorter than this are treated as part of the surrounding note.
_UNVOICED_CLOSE_MS = float(os.environ.get("APP_AUTOTUNE_CLOSE_MS", "60"))
_UNVOICED_RAMP_MS = float(os.environ.get("APP_AUTOTUNE_RAMP_MS", "10"))
# Frames this far below programme are silence; the vocoder never writes there.
_SILENCE_BELOW_DB = float(os.environ.get("APP_AUTOTUNE_SILENCE_DB", "40"))
# Above this the vocoder's output is discarded and the input's own content
# kept: nothing musical lives there and resynthesis only adds fold-down.
_HF_KEEP_HZ = float(os.environ.get("APP_AUTOTUNE_HF_KEEP_HZ", "16000"))


def apply_autotune(
    y: np.ndarray,
    sr: int,
    retune_ms: float = 20.0,
    strength: float = 0.7,
    ref_cents: Optional[float] = None,
    scale_root: Optional[int] = None,
    scale_mode: Optional[str] = None,
    scale_pcs: Optional[List[int]] = None,
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
    # ---- Vocoder correction (WORLD) ---------------------------------------
    # First choice, and tried before anything else is computed: WORLD runs its
    # own pitch tracker, so the estimate_f0 + key detection below are pure
    # waste on this path (35s of a 45s call on a 30s vocal).
    #
    # PSOLA re-spaces the existing glottal pulses, so any wobble its tracker
    # misses survives the shift -- it plateaued at ~10 cents with 20 cents
    # still moving inside each note. WORLD rebuilds the signal from a pitch
    # contour we write ourselves, so a note comes out exactly as steady as it
    # is specified: 0.0 within-note motion, matching the reference lead, where
    # PSOLA left 10.0. Formants are a separate channel and measured a smaller
    # shift than PSOLA's (0.68 dB vs 1.86 dB).
    if _USE_WORLD:
        try:
            _sr_notes = (_notes_from_pitch_classes(scale_pcs) if scale_pcs
                         else _scale_notes_in_range(
                             0 if scale_root is None else scale_root,
                             "chromatic" if scale_mode is None else scale_mode))
            _w = _world_correct(y, sr, strength=float(np.clip(strength, 0.0, 1.0)),
                                scale_notes=_sr_notes)
            if _w is not None:
                return _w.astype(np.float32)
        except Exception as _we:
            print(f"  WORLD correction unavailable ({_we}); trying PSOLA")

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

    # ---- Continuous correction (PSOLA) ------------------------------------
    # Preferred path. The segment method below applies one constant shift per
    # note, so pitch keeps drifting inside the note -- against a hard-tuned
    # lead (0.8 cents off, no motion within a note) it stalled at 20 cents of
    # in-note drift whatever the segment size. Correcting every pitch period
    # pins the note instead, and PSOLA re-spaces the glottal pulses rather than
    # resynthesising the spectrum, so formants survive.
    if _USE_PSOLA:
        try:
            from scipy.ndimage import median_filter
            _ok = voiced_mask & np.isfinite(midi)
            if _ok.sum() >= 8:
                # One steady target per note: snapping each frame on its own
                # lets the target follow the singer's wobble, which is exactly
                # the wobble we are trying to remove.
                _k = max(1, int(0.15 * sr / 256)) | 1
                _sm = median_filter(np.where(_ok, midi, np.nanmedian(midi[_ok])), size=_k)
                _tm = np.array([_nearest_scale_note(v, scale_notes) for v in _sm])
                _tgt_hz = np.where(_ok, librosa.midi_to_hz(_tm), np.nan)
                _f0_hz = np.where(_ok, f0, np.nan)
                _out = _psola_correct(y, sr, _f0_hz, _tgt_hz, 256,
                                      strength=float(np.clip(strength, 0.0, 1.0)))
                if np.isfinite(_out).all() and float(np.max(np.abs(_out))) > 1e-6:
                    return _out.astype(np.float32)
        except Exception as _pe:
            print(f"  PSOLA correction unavailable ({_pe}); using segment method")


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
    # Correction floor follows the reference's own tightness.
    #
    # A fixed 12-cent floor cannot reproduce a hard-tuned lead: refusing to
    # correct anything under 12 cents guarantees the output stalls around 10,
    # while the Hide lead sits at 0.8 cents -- essentially perfectly snapped.
    # Aiming at half the reference's own error lets a tightly tuned reference
    # pull everything onto the grid, and still spares a loosely tuned one from
    # being resynthesised to fix errors nobody can hear.
    _floor = _MIN_AUDIBLE_ST if ref_cents is None else float(
        np.clip(ref_cents / 200.0, 0.01, _MIN_AUDIBLE_ST))
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
    # >= 100 ms per tuned piece. Each region is pitch-shifted independently and
    # joined by a crossfade, so short regions buy little correction and pay the
    # full artifact cost of another join.
    MIN_FRAMES = max(3, int(0.10 * sr / HOP))
    i = 0
    n = len(shift)
    while i < n:
        # Skip corrections too small to hear.
        #
        # This was 0.03 semitones -- THREE CENTS. Nobody hears a three cent
        # error, but everybody hears the phase vocoder invoked to fix it, so
        # most of the vocal was being resynthesised to correct errors below the
        # threshold of perception. At 12 cents the pitch result is identical
        # (10.8 cents median, matching the reference exactly) while the amount
        # of audio put through the shifter drops from 55% to 24%.
        if abs(shift[i]) < _floor:
            i += 1
            continue
        j = i + 1
        while j < n and abs(shift[j] - shift[i]) < 0.35 and abs(shift[j]) >= _floor:
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
    # 30 ms joins, not 12. Each region is pitch-shifted independently by a
    # phase vocoder, so neighbouring regions do not share phase and a short
    # crossfade between them warbles audibly.
    fade = max(32, int(0.030 * sr))
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


def _psola_correct(y, sr, f0_frames, target_frames, hop, strength=1.0):
    """
    Continuous pitch correction by time-domain pitch-synchronous overlap-add.

    The segment-based corrector above applies ONE constant shift per note, so
    pitch keeps drifting inside the note: measured against a hard-tuned lead
    (0.8 cents off, zero motion within a note) it stalled at 20 cents of
    in-note drift no matter how the segments were sized. Correcting every
    period instead is what pins a note to the grid.

    PSOLA rather than a phase vocoder because it re-spaces the glottal pulses
    themselves: formants stay where they are, so the voice keeps its character
    instead of acquiring the smeared, watery quality that comes from
    resynthesising the spectrum frame by frame.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 4 * hop:
        return y

    fpos = np.arange(len(f0_frames)) * hop
    voiced = np.isfinite(f0_frames) & np.isfinite(target_frames) & (f0_frames > 0)
    if voiced.sum() < 8:
        return y

    idx = np.arange(n)
    f0_s = np.interp(idx, fpos[voiced], f0_frames[voiced])
    ratio = np.clip(target_frames[voiced] / f0_frames[voiced], 0.85, 1.18)
    # Partial correction stays available: 1.0 pins to the grid, less leaves
    # some of the singer's own pitch in place.
    ratio = 1.0 + (ratio - 1.0) * float(np.clip(strength, 0.0, 1.0))
    ratio_s = np.interp(idx, fpos[voiced], ratio)
    tgt_s = np.interp(idx, fpos[voiced], target_frames[voiced])
    voiced_s = np.interp(idx, fpos, voiced.astype(float)) > 0.5

    period = np.clip(sr / np.clip(f0_s, 60.0, 1100.0), 32, sr // 50)

    # Analysis marks: walk one period at a time, snapping each to the nearest
    # energy peak so grains stay pitch-synchronous.
    marks = []
    t = 0
    while t < n - 2:
        T = int(period[min(t, n - 1)])
        if voiced_s[min(t, n - 1)]:
            w = max(2, T // 4)
            lo, hi = max(0, t - w), min(n, t + w + 1)
            t = lo + int(np.argmax(np.abs(y[lo:hi]))) if hi > lo else t
        marks.append(t)
        t += T
    if len(marks) < 4:
        return y
    marks = np.asarray(marks)

    out = np.zeros(n)
    norm = np.zeros(n)
    # Synthesis position is tracked as a float and only rounded when a grain is
    # placed. Advancing by whole samples quantises every period ratio -- with a
    # ~196 sample period a wanted 1.023 becomes 1.021, and the error accumulates
    # -- which left about 9 cents on the table even on a clean tone.
    s_f = float(marks[0])
    while s_f < n - 2:
        s = int(round(s_f))
        j = int(np.clip(np.searchsorted(marks, s), 0, len(marks) - 1))
        a = int(marks[j])
        T = int(period[min(a, n - 1)])
        L = 2 * T
        lo, hi = a - T, a - T + L
        o_lo, o_hi = s - T, s - T + L
        if lo >= 0 and hi <= n and o_lo >= 0 and o_hi <= n:
            win = np.hanning(L)
            out[o_lo:o_hi] += y[lo:hi] * win
            norm[o_lo:o_hi] += win
        # Space the next grain at the TARGET period, not the measured period
        # scaled by a ratio. Those are algebraically the same only if both are
        # read at the same instant; period came from the analysis mark and the
        # ratio from the synthesis position, so every frame of pitch-tracking
        # noise leaked straight into the output. Driving the spacing from the
        # target alone pins the note -- which is what hard tuning does, and why
        # a hard-tuned lead shows zero pitch motion inside a note.
        if voiced_s[min(a, n - 1)]:
            _tgt = tgt_s[min(a, n - 1)]
            _step = sr / _tgt if _tgt > 1.0 else float(period[min(a, n - 1)])
            # strength blends between the sung period and the target period
            _st = float(np.clip(strength, 0.0, 1.0))
            _step = float(period[min(a, n - 1)]) * (1.0 - _st) + _step * _st
        else:
            _step = float(period[min(a, n - 1)])
        s_f += max(1.0, _step)

    norm[norm < 1e-6] = 1.0
    res = out / norm
    # Unvoiced material is copied through untouched -- consonants have no
    # period to re-space and only degrade if they are grain-shifted.
    res = np.where(voiced_s, res, y)
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def _world_correct(y, sr, strength=1.0, scale_notes=None, note_ms=150.0):
    """
    Pitch correction by WORLD vocoder resynthesis.

    PSOLA and phase-vocoder methods SHIFT the recording, so whatever wobble the
    pitch tracker fails to see survives into the output -- which is why they
    plateau around 9 cents with ~20 cents of drift still inside each note.
    WORLD instead decomposes the voice into three independent channels:

        f0            the pitch contour
        spectral env  the formants, i.e. who the singer is
        aperiodicity  breath and noise

    and rebuilds the audio from them. The pitch is therefore SPECIFIED rather
    than nudged: writing a constant f0 across a note produces a note that is
    exactly in tune and exactly flat, which is what a hard-tuned lead measures
    like (0.8 cents, no motion within the note). Formants are untouched because
    the envelope is a separate channel, so the voice keeps its identity.

    Returns None if WORLD is unavailable or the analysis is unusable, so the
    caller can fall back.
    """
    try:
        import pyworld
    except ImportError:
        return None

    x = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < sr // 4:
        return None

    try:
        frame_ms = 5.0
        # dio, not harvest. harvest is the more accurate tracker but it ran
        # 167s of DSP on a 4-minute vocal -- most of a job's whole budget.
        # stonemask refines dio's estimate, and the contour is then median
        # filtered to one steady value per note, so the extra precision
        # harvest buys is discarded two lines later anyway.
        f0, t = pyworld.dio(x, sr, f0_floor=65.0, f0_ceil=1100.0,
                            frame_period=frame_ms)
        f0 = pyworld.stonemask(x, f0, t, sr)          # refine the estimate
        sp = pyworld.cheaptrick(x, f0, t, sr)
        ap = pyworld.d4c(x, f0, t, sr)
    except Exception:
        return None

    voiced = f0 > 0
    if voiced.sum() < 8:
        return None

    notes = scale_notes if scale_notes else _scale_notes_in_range(0, "chromatic")
    midi = np.zeros_like(f0)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

    # One steady target per note. Snapping frame by frame would let the target
    # chase the singer's own drift -- the very thing being removed.
    from scipy.ndimage import median_filter
    k = max(1, int(note_ms / frame_ms)) | 1
    smooth = median_filter(np.where(voiced, midi, np.nan_to_num(np.nanmedian(midi[voiced]))), size=k)
    target = np.array([_nearest_scale_note(v, notes) for v in smooth])

    st = float(np.clip(strength, 0.0, 1.0))
    # Correct the note's CENTRE; keep the singer's motion around it.
    #
    # `midi * (1-st) + target * st` writes the median-filtered target straight
    # out, so at full strength the pitch is piecewise constant -- dead flat
    # inside every note. That is the robot sound, and it is not what a tuned
    # record does: measured against Hide's lead, this output sat at 0.8 cents
    # from the grid with 62% of frames repeating the previous frame's pitch
    # exactly, where the reference is 9.2 cents and 47%.
    #
    # Splitting the contour into a note centre (`smooth`) and everything
    # faster than a note (vibrato, scoops, micro-variation) lets the centre be
    # snapped while the expression rides on top untouched.
    residual = midi - smooth
    new_midi = smooth + (target - smooth) * st + residual
    f0_new = np.zeros_like(f0)
    f0_new[voiced] = 440.0 * (2.0 ** ((new_midi[voiced] - 69.0) / 12.0))

    try:
        out = pyworld.synthesize(np.ascontiguousarray(f0_new),
                                 np.ascontiguousarray(sp),
                                 np.ascontiguousarray(ap), sr, frame_ms)
    except Exception:
        return None

    out = np.asarray(out, dtype=np.float64)
    if len(out) < len(x):
        out = np.pad(out, (0, len(x) - len(out)))
    out = out[: len(x)]
    if not np.isfinite(out).all() or float(np.max(np.abs(out))) < 1e-6:
        return None
    # Resynthesis sets its own absolute level; match the input's.
    out = out * (float(np.max(np.abs(x))) / (float(np.max(np.abs(out))) + 1e-12))

    # Keep the ORIGINAL unvoiced audio.
    #
    # Consonants have no pitch to correct, and rebuilding them from a spectral
    # envelope plus a noise channel smears their attack: resynthesising
    # everything scored 4.4/10 on phrasing density and 7.3 on transients,
    # against 9.7 and 9.7 for the untouched dry. Crossfading the vocoder in
    # only where the voice is actually pitched keeps the consonants intact and
    # costs nothing in tuning, since unvoiced frames were never corrected.
    # dio's voicing flag flickers frame to frame in the middle of sustained
    # notes. Ramping straight off it modulates the amplitude every few
    # milliseconds, which reads as smeared attacks (transient fidelity 4.2/10).
    # Close the short gaps first so the mask holds steady across a note and
    # only opens for genuine unvoiced stretches.
    # Put the dry signal's own loudness contour back.
    #
    # WORLD rebuilds amplitude from a smoothed spectral envelope, so the
    # micro-dynamics that make a vocal read as "attacking" get flattened:
    # correcting pitch alone cost 4.4 points of transient fidelity, 2.0 of
    # phrasing density and 1.7 of dynamic range downstream. Pitch correction
    # has no business altering loudness, so the short-time gain is forced back
    # to the input's frame by frame. Pitch comes from the vocoder, dynamics
    # from the original.
    _fl = max(64, int(0.020 * sr))          # 20 ms window
    _hp = max(32, _fl // 2)
    _pad = _fl // 2
    _xp = np.pad(x, (_pad, _fl), mode="constant")
    _op = np.pad(out, (_pad, _fl), mode="constant")
    _n = 1 + (len(x) + _pad) // _hp
    _rx = np.empty(_n); _ro = np.empty(_n)
    for _i in range(_n):
        _a = _i * _hp
        _rx[_i] = np.sqrt(np.mean(_xp[_a:_a + _fl] ** 2) + 1e-20)
        _ro[_i] = np.sqrt(np.mean(_op[_a:_a + _fl] ** 2) + 1e-20)
    # Only correct where there is signal to correct; clamp so a near-silent
    # frame cannot produce a huge gain.
    _g = np.clip(_rx / _ro, 0.25, 4.0)
    _g[_rx < 1e-5] = 1.0
    _gs = np.interp(np.arange(len(x)), np.arange(_n) * _hp, _g)
    out = out * _gs

    vflag = voiced.astype(np.float64)
    close = max(1, int(_UNVOICED_CLOSE_MS / frame_ms)) | 1
    vflag = (median_filter(vflag, size=close) > 0.5).astype(np.float64)

    # Never synthesise into silence.
    #
    # dio marks the odd frame voiced on nothing but noise, and WORLD then
    # rebuilds that frame from a spectral envelope -- putting audible hiss in
    # the gaps between phrases where the input had none. Measured, autotune
    # raised the gap level 2.8 dB over the same render with it off. Where the
    # input is essentially silent, the input IS the right answer.
    hop_n = max(1, int(round(sr * frame_ms / 1000.0)))
    n_fr = len(vflag)
    pad = np.pad(np.abs(x), (0, hop_n * n_fr))
    fr_rms = np.sqrt(np.array([
        np.mean(pad[i * hop_n:(i + 1) * hop_n] ** 2) for i in range(n_fr)]) + 1e-20)
    fr_db = 20.0 * np.log10(fr_rms)
    silent = fr_db < (np.percentile(fr_db, 95) - _SILENCE_BELOW_DB)
    vflag[silent] = 0.0

    hop = sr * frame_ms / 1000.0
    idx = np.clip((np.arange(len(x)) / hop).astype(np.int64), 0, len(vflag) - 1)
    mask = vflag[idx]
    # Short ramp, so each splice is a fade rather than a step.
    ramp = max(1, int(_UNVOICED_RAMP_MS / 1000.0 * sr))
    mask = np.convolve(mask, np.ones(ramp) / ramp, mode="same")
    out = out * mask + x * (1.0 - mask)

    if not np.isfinite(out).all():
        return None

    # Keep the ORIGINAL's top octave.
    #
    # WORLD rebuilds the signal from an impulse train plus a noise channel, and
    # once the pitch is moved the harmonics move with it -- near Nyquist they
    # fold. Probed through the chain, autotune was adding +7.2 dB above 16 kHz
    # (the tone-matching EQ, the obvious suspect, added only +0.9), and that is
    # audible as fizz on loud notes. A vocal carries nothing musical up there,
    # and correcting pitch is no reason to alter it, so splice the input's own
    # top octave back on. Zero-phase filters, so the two halves recombine
    # without smearing.
    if sr > 2 * _HF_KEEP_HZ:
        try:
            from scipy.signal import butter, sosfiltfilt
            _w = _HF_KEEP_HZ / (sr / 2.0)
            _lp = butter(8, _w, btype="low", output="sos")
            _hp = butter(8, _w, btype="high", output="sos")
            out = sosfiltfilt(_lp, out) + sosfiltfilt(_hp, x[:len(out)])
        except Exception as _hf:
            print(f"  WORLD top-octave splice skipped ({_hf})")

    # Re-match the peak LAST. The earlier normalisation happens before the
    # envelope is restored and the unvoiced material is blended back, and both
    # of those add level -- a synthetic case came out at 1.14, i.e. clipped,
    # and everything downstream would have inherited it.
    _pk = float(np.max(np.abs(out)))
    _target_pk = float(np.max(np.abs(x)))
    if _pk > 1e-9 and _pk > _target_pk:
        out = out * (_target_pk / _pk)
    return out.astype(np.float32)
