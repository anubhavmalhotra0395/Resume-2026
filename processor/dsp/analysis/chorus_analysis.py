import numpy as np
import librosa
from dataclasses import dataclass, asdict
from scipy.signal import find_peaks


@dataclass
class ChorusProfile:
    rate_hz: float
    depth: float  # 0..1
    mix: float    # 0..1

    def as_dict(self):
        return asdict(self)


def _autocorr_rate(signal: np.ndarray, sr: int, min_hz: float = 0.2, max_hz: float = 5.0) -> float:
    """
    Estimate modulation rate from a 1D signal via autocorrelation.
    """
    # Remove DC
    sig = signal - np.mean(signal)
    if np.allclose(sig, 0):
        return 0.8  # default gentle chorus

    # Autocorrelation
    corr = np.correlate(sig, sig, mode="full")
    corr = corr[len(corr) // 2 :]  # positive lags

    # Convert target Hz to lag range
    max_lag = int(sr / min_hz)
    min_lag = int(sr / max_hz)
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return 0.8

    search_region = corr[min_lag:max_lag]
    if search_region.size == 0:
        return 0.8

    peaks, _ = find_peaks(search_region)
    if peaks.size == 0:
        return 0.8

    # Strongest peak
    best_peak_idx = peaks[np.argmax(search_region[peaks])]
    lag = min_lag + best_peak_idx
    if lag == 0:
        return 0.8
    return float(sr / lag)


def detect_chorus(reference: np.ndarray, sr: int) -> ChorusProfile:
    """
    Detect chorus-like MODULATION from reference audio.

    Key distinction: chorus is a time-varying pitch/delay modulation effect.
    Simple stereo width (L≠R) from a doubler or room is NOT chorus.

    Two-gate approach:
      1. Side energy > 10% of mid  →  stereo spread exists (necessary but not sufficient)
      2. Side energy envelope has a periodic modulation at 0.2–5 Hz  →  real chorus LFO

    Only if BOTH conditions are met do we report a non-zero mix.
    Mix is capped at 0.25 for vocals.
    """
    # ── Mid/Side split ─────────────────────────────────────────────────────
    if reference.ndim == 2 and reference.shape[0] > 1:
        mid  = np.mean(reference, axis=0)
        side = (reference[0] - reference[1]) / 2.0
    elif reference.ndim == 2 and reference.shape[0] == 1:
        mid  = reference[0]
        side = np.zeros_like(mid)
    else:
        mid = reference
        delay = int(0.002 * sr)
        side = np.zeros_like(mid)
        if delay < len(mid):
            side[delay:] = mid[:-delay]
            side = side - mid

    mid_energy  = float(np.mean(mid  ** 2)) + 1e-9
    side_energy = float(np.mean(side ** 2)) + 1e-9
    side_ratio  = side_energy / mid_energy

    # Gate 1: must have meaningful stereo spread (>10% side/mid energy)
    if side_ratio < 0.10:
        return ChorusProfile(rate_hz=0.8, depth=0.2, mix=0.0)

    # ── Modulation analysis on the side envelope ──────────────────────────
    hop = 512
    side_env = np.abs(side)
    # Smooth to get a slow amplitude envelope
    from scipy.ndimage import uniform_filter1d
    win = max(1, sr // 100)   # 10 ms window
    side_env_smooth = uniform_filter1d(side_env, size=win)

    # Downsample to ~200 Hz for LFO analysis
    ds_factor = max(1, sr // 200)
    env_ds = side_env_smooth[::ds_factor]
    env_sr_ds = sr / ds_factor

    rate_hz = _autocorr_rate(env_ds, int(env_sr_ds), min_hz=0.2, max_hz=5.0)
    rate_hz = float(np.clip(rate_hz, 0.2, 5.0))

    # Modulation depth: coefficient of variation of the smoothed envelope
    env_cv = float(np.std(env_ds) / (np.mean(env_ds) + 1e-9))

    # Gate 2: must have periodic modulation (CV > 0.15 means the side level is
    # genuinely fluctuating rhythmically, not just constant stereo spread)
    MODULATION_CV_THRESHOLD = 0.15
    if env_cv < MODULATION_CV_THRESHOLD:
        return ChorusProfile(rate_hz=rate_hz, depth=float(np.clip(env_cv, 0.0, 1.0)), mix=0.0)

    # ── Confirmed chorus — scale mix from modulation depth ────────────────
    depth = float(np.clip(env_cv, 0.0, 1.0))
    # Mix: proportional to side spread but anchored to modulation evidence
    # Cap at 0.25 for vocals — any higher washes out intelligibility
    raw_mix = side_ratio / (1.0 + side_ratio)   # compress from 0..∞ to 0..1
    mix = float(np.clip(raw_mix * 0.5, 0.05, 0.25))

    return ChorusProfile(rate_hz=rate_hz, depth=depth, mix=mix)

