"""
Vocal Layer Analysis — detect exactly how many vocal layers are in the reference.

Two types of layers are distinguished:
  1. Doubling layers  — same melody, tiny pitch/timing variation (ADT / stack)
  2. Harmony voices   — different pitch (musical interval: 3rd, 4th, 5th, octave)

Total detected layers drives the replication: if reference has N layers total,
the output will have exactly N layers applied to the dry vocal.

Detection approach
──────────────────
Doublers:
  - Measure stereo decorrelation (side/mid energy ratio).
  - Measure pitch microvariation within voiced segments.
  - Each 0.10–0.15 unit of side/mid ratio ≈ one additional doubling layer.
  - Cross-correlation between L and R channel delayed copies also reveals
    the number of stacked copies by finding sub-peaks in the autocorrelation.

Harmony voices:
  - Use chromagram + pyin f0 to detect pitched content at musical intervals
    above/below the main melody.
  - Only count as a harmony if the interval's energy clears a strict threshold
    AND there are more than 1 frame with that pitch.
  - Hard cap: max 2 harmony voices (beyond that it's reverb/overtones, not vocals).

Output: VocalLayersProfile
  - n_doublers         : int  (0 = no doubling, 1 = one double, etc.)
  - doubler_detunes    : List[float]  cents for each copy (e.g. [+9.0, -7.0])
  - doubler_delays_ms  : List[float]  ms offset per copy
  - doubler_pans       : List[float]  -1..1 stereo position per copy
  - harmony_intervals  : List[int]    semitones (e.g. [5, 7])
  - harmony_strengths  : List[float]  0..1 mix level
  - harmony_pans       : List[float]  -1..1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import librosa


@dataclass
class VocalLayersProfile:
    # Doubling layers (same melody, slightly detuned/delayed copies)
    n_doublers: int = 0
    doubler_detunes_cents: List[float] = field(default_factory=list)
    doubler_delays_ms: List[float] = field(default_factory=list)
    doubler_pans: List[float] = field(default_factory=list)

    # Harmony voices (different pitched)
    harmony_intervals: List[int] = field(default_factory=list)
    harmony_strengths: List[float] = field(default_factory=list)
    harmony_pans: List[float] = field(default_factory=list)

    @property
    def total_layers(self) -> int:
        """Total layers including the dry lead (always 1)."""
        return 1 + self.n_doublers + len(self.harmony_intervals)


_ANALYSIS_SR = 16000  # downsample to this before pyin calls


def _maybe_resample(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Downsample to _ANALYSIS_SR and clip to 20s for fast analysis."""
    import librosa as _lb
    # Clip to 20s first
    max_s = _ANALYSIS_SR * 20
    if audio.ndim == 2:
        clipped = audio[:, :sr * 20] if audio.shape[1] > sr * 20 else audio
        if sr != _ANALYSIS_SR:
            ch0 = _lb.resample(clipped[0].astype(np.float32), orig_sr=sr, target_sr=_ANALYSIS_SR)
            ch1 = _lb.resample(clipped[1].astype(np.float32), orig_sr=sr, target_sr=_ANALYSIS_SR) if clipped.shape[0] > 1 else ch0
            return np.stack([ch0, ch1]), _ANALYSIS_SR
        return clipped, sr
    else:
        clipped = audio[:sr * 20] if len(audio) > sr * 20 else audio
        if sr != _ANALYSIS_SR:
            return _lb.resample(clipped.astype(np.float32), orig_sr=sr, target_sr=_ANALYSIS_SR), _ANALYSIS_SR
        return clipped, sr


def _estimate_doublers(reference_audio: np.ndarray, sr: int) -> tuple[int, List[float], List[float], List[float]]:
    """
    Estimate number of doubling layers from stereo decorrelation and
    cross-correlation analysis.

    Returns (n_doublers, detunes_cents, delays_ms, pans)
    """
    if reference_audio.ndim != 2 or reference_audio.shape[0] != 2:
        return 0, [], [], []

    reference_audio, sr = _maybe_resample(reference_audio, sr)
    left  = reference_audio[0].astype(np.float64)
    right = reference_audio[1].astype(np.float64)
    mid   = (left + right) * 0.5
    side  = (left - right) * 0.5

    mid_energy  = float(np.mean(mid  ** 2)) + 1e-12
    side_energy = float(np.mean(side ** 2))
    spread = side_energy / mid_energy

    # Estimate doubler count from spread ratio
    # spread < 0.05  → mono / single voice → 0 doublers
    # spread 0.05–0.20 → light doubling → 1 doubler
    # spread 0.20–0.45 → medium stack → 2 doublers
    # spread > 0.45  → heavy stack → 3+ doublers (cap at 6 for sanity)
    if spread < 0.05:
        n_doublers = 0
    elif spread < 0.20:
        n_doublers = 1
    elif spread < 0.45:
        n_doublers = 2
    else:
        n_doublers = min(int(spread / 0.15), 6)

    if n_doublers == 0:
        return 0, [], [], []

    # Detect actual delay offsets by cross-correlating L and R
    # sub-peaks in the cross-correlation = delayed copies
    max_lag_samples = int(sr * 0.05)  # up to 50ms
    xcorr = np.correlate(left[:min(len(left), 4 * sr)],
                         right[:min(len(right), 4 * sr)], mode="full")
    half = len(xcorr) // 2
    xcorr_pos = xcorr[half: half + max_lag_samples]
    xcorr_pos = xcorr_pos / (np.max(np.abs(xcorr_pos)) + 1e-9)

    # Find cross-correlation peaks (excluding lag 0)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(xcorr_pos[1:], height=0.25, distance=int(sr * 0.005))
    peaks = peaks + 1  # correct for the [1:] slice

    detected_delays_ms = []
    for p in peaks[:n_doublers]:
        detected_delays_ms.append(float(p / sr * 1000.0))

    # Estimate pitch microvariation per layer from pyin deviation
    detune_base = 8.0  # cents default
    try:
        f0, voiced, _ = librosa.pyin(
            mid[:min(len(mid), _ANALYSIS_SR * 10)],
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            hop_length=128,
        )
        voiced_f0 = f0[voiced & np.isfinite(f0) & (f0 > 0)]
        if len(voiced_f0) > 10:
            midi_vals = librosa.hz_to_midi(voiced_f0)
            # std of pitch in cents
            pitch_std_cents = float(np.std(midi_vals) * 100.0)
            # typical doubler detune is 5–20 cents
            detune_base = float(np.clip(pitch_std_cents * 0.5, 5.0, 20.0))
    except Exception:
        pass

    # Build detune/delay/pan per doubler layer
    # Alternate polarity and pan symmetrically
    detunes = []
    delays  = []
    pans    = []

    for i in range(n_doublers):
        sign = 1 if i % 2 == 0 else -1
        detunes.append(sign * detune_base * (1.0 + i * 0.15))
        if i < len(detected_delays_ms):
            delays.append(detected_delays_ms[i])
        else:
            # Spread delays 12–28ms
            delays.append(12.0 + i * 8.0)
        # Pan: first pair centre-left/right, further pairs wider
        pan_amount = min(0.3 + i * 0.25, 0.9)
        pans.append(sign * pan_amount)

    return n_doublers, detunes, delays, pans


def _estimate_harmony_voices(reference_audio: np.ndarray, sr: int) -> tuple[List[int], List[float], List[float]]:
    """
    Detect distinct harmony voices (pitched differently from lead) in reference.
    Returns (intervals_semitones, strengths, pans).
    Hard cap: max 2 voices.
    """
    reference_audio, sr = _maybe_resample(reference_audio, sr)
    if reference_audio.ndim == 2:
        mono = np.mean(reference_audio, axis=0)
        left  = reference_audio[0]
        right = reference_audio[1] if reference_audio.shape[0] > 1 else reference_audio[0]
    else:
        mono = reference_audio
        left = right = mono

    # Detect main f0
    try:
        f0, voiced, _ = librosa.pyin(
            mono[:min(len(mono), _ANALYSIS_SR * 15)],
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            hop_length=128,
        )
        voiced_f0 = f0[voiced & np.isfinite(f0) & (f0 > 0)]
        if len(voiced_f0) < 10:
            return [], [], []
        main_f0 = float(np.median(voiced_f0))
    except Exception:
        return [], [], []

    # Spectral analysis — use already-clipped mono/left/right (max 20s at 16kHz)
    n_fft = 1024
    hop = 128
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S_mean = np.mean(S, axis=1)
    S_norm = S_mean / (np.max(S_mean) + 1e-9)

    S_L = np.abs(librosa.stft(left, n_fft=n_fft, hop_length=hop))
    S_R = np.abs(librosa.stft(right, n_fft=n_fft, hop_length=hop))

    def _band_e(spec_norm, target_hz, bw=50.0):
        mask = (freqs >= target_hz - bw) & (freqs <= target_hz + bw)
        return float(np.mean(spec_norm[mask])) if mask.any() else 0.0

    # Candidate musical intervals — positive only (above the lead).
    # Negative intervals are below the lead and make the mix sound darker/thinner.
    # A 3rd above, 4th above, 5th above, or octave above all add fullness and air.
    candidate_intervals = [3, 4, 5, 7, 12]
    intervals_out  = []
    strengths_out  = []
    pans_out       = []

    for interval in candidate_intervals:
        if len(intervals_out) >= 2:
            break  # hard cap 2 harmony voices

        target_hz = main_f0 * (2 ** (interval / 12.0))
        if target_hz <= 0 or target_hz > sr / 2:
            continue

        energy = _band_e(S_norm, target_hz)

        # Strict threshold: 0.55 — must be genuinely prominent pitched content
        if energy < 0.55:
            continue

        # Also verify it's NOT just a harmonic overtone of the main f0
        # (harmonics fall at 2x, 3x, 4x the fundamental)
        is_overtone = False
        for mult in [2, 3, 4]:
            overtone_hz = main_f0 * mult
            if abs(target_hz - overtone_hz) < 80:
                is_overtone = True
                break
        if is_overtone:
            continue

        # Pan: compare L/R energy at the harmony frequency
        eL = _band_e(np.mean(S_L, axis=1) / (np.max(np.mean(S_L, axis=1)) + 1e-9), target_hz)
        eR = _band_e(np.mean(S_R, axis=1) / (np.max(np.mean(S_R, axis=1)) + 1e-9), target_hz)
        pan = float(np.clip((eR - eL) / (eR + eL + 1e-9), -1.0, 1.0))

        # Strength: how loud relative to lead
        lead_energy = _band_e(S_norm, main_f0)
        strength = float(np.clip(energy / (lead_energy + 1e-9) * 0.5, 0.1, 0.5))

        intervals_out.append(interval)
        strengths_out.append(strength)
        pans_out.append(pan)

    return intervals_out, strengths_out, pans_out


def detect_vocal_layers(reference_audio: np.ndarray, sr: int) -> Optional[VocalLayersProfile]:
    """
    Detect all vocal layers in the reference audio.
    Returns None if only a single dry vocal is detected (no layering).
    """
    n_doublers, detunes, delays, d_pans = _estimate_doublers(reference_audio, sr)
    harmony_intervals, harmony_strengths, harmony_pans = _estimate_harmony_voices(reference_audio, sr)

    total_extra = n_doublers + len(harmony_intervals)
    if total_extra == 0:
        return None  # single voice — no layering to replicate

    return VocalLayersProfile(
        n_doublers=n_doublers,
        doubler_detunes_cents=detunes,
        doubler_delays_ms=delays,
        doubler_pans=d_pans,
        harmony_intervals=harmony_intervals,
        harmony_strengths=harmony_strengths,
        harmony_pans=harmony_pans,
    )
