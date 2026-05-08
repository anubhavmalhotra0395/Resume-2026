import numpy as np
import librosa
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class HarmonyProfile:
    intervals_semitones: List[int]
    strengths: List[float]
    pans: List[float]           # -1.0 (L) to 1.0 (R)
    timing_offsets: List[float] # seconds

    def as_dict(self):
        return asdict(self)


def _detect_main_f0(ref_mono: np.ndarray, sr: int) -> float:
    """Estimate main melody f0 (median of voiced frames)."""
    try:
        f0, _, _ = librosa.pyin(
            ref_mono,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0 = np.nan_to_num(f0)
        voiced = f0[f0 > 0]
        if voiced.size == 0:
            return 0.0
        return float(np.median(voiced))
    except Exception:
        return 0.0


def _band_energy(S: np.ndarray, freqs: np.ndarray, target_hz: float, bandwidth_hz: float = 30.0) -> float:
    """Sum magnitude around target_hz within bandwidth."""
    if target_hz <= 0:
        return 0.0
    idx = np.where((freqs >= target_hz - bandwidth_hz) & (freqs <= target_hz + bandwidth_hz))[0]
    if idx.size == 0:
        return 0.0
    return float(np.mean(S[idx]))


def detect_harmonies(reference_audio: np.ndarray, sr: int) -> HarmonyProfile:
    """
    Detect simple harmony intervals from a reference vocal.

    Args:
        reference_audio: np.ndarray, shape (n,) or (2, n)
        sr: int, sample rate

    Returns:
        HarmonyProfile dataclass
    """
    # Convert to mono for pitch detection
    if reference_audio.ndim == 2:
        ref_mono = np.mean(reference_audio, axis=0)
    else:
        ref_mono = reference_audio

    main_f0 = _detect_main_f0(ref_mono, sr)
    if main_f0 <= 0:
        return HarmonyProfile([], [], [], [])

    # STFT for spectral inspection
    n_fft = 2048
    hop = 256
    S = np.abs(librosa.stft(ref_mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S_mean = np.mean(S, axis=1)
    S_mean_norm = S_mean / (np.max(S_mean) + 1e-9)

    intervals_semitones: List[int] = []
    strengths: List[float] = []
    pans: List[float] = []
    timing_offsets: List[float] = []

    candidate_intervals = [3, 4, 5, 7, 12]  # minor/major 3rd, 4th, 5th, octave

    # If stereo, prepare L/R spectra for pan estimation
    S_L = S_R = None
    if reference_audio.ndim == 2 and reference_audio.shape[0] > 1:
        S_L = np.abs(librosa.stft(reference_audio[0], n_fft=n_fft, hop_length=hop))
        S_R = np.abs(librosa.stft(reference_audio[1], n_fft=n_fft, hop_length=hop))

    for interval in candidate_intervals:
        target_hz = main_f0 * (2 ** (interval / 12.0))
        energy = _band_energy(S_mean_norm, freqs, target_hz, bandwidth_hz=40.0)

        # High threshold (0.55) to avoid adding harmonies that are just overtones.
        # Also hard-cap at 2 detected voices max — real produced harmony parts are
        # usually just one or two counter-melodies, not all five intervals at once.
        if energy > 0.55 and len(intervals_semitones) < 2:
            intervals_semitones.append(interval)
            strengths.append(float(np.clip(energy, 0.0, 1.0)))

            # Pan estimate: compare L/R energy in that band if stereo
            if S_L is not None and S_R is not None:
                eL = _band_energy(np.mean(S_L, axis=1), freqs, target_hz, bandwidth_hz=40.0)
                eR = _band_energy(np.mean(S_R, axis=1), freqs, target_hz, bandwidth_hz=40.0)
                pan = (eR - eL) / (eR + eL + 1e-9)
                pan = float(np.clip(pan, -1.0, 1.0))
            else:
                pan = 0.0
            pans.append(pan)

            # Timing offset: not estimated here; default 0
            timing_offsets.append(0.0)

    return HarmonyProfile(
        intervals_semitones=intervals_semitones,
        strengths=strengths,
        pans=pans,
        timing_offsets=timing_offsets,
    )

