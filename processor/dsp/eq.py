from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import iirpeak, lfilter


@dataclass
class EqBand:
    f: float
    gain_db: float
    q: float


def apply_eq(x: np.ndarray, sr: int, bands: List[EqBand]) -> np.ndarray:
    """Apply a stack of peaking EQ filters (RBJ biquads)."""
    y = np.copy(x)
    for band in bands:
        # iirpeak expects normalized frequency (0..1, where 1 = Nyquist)
        w0 = band.f / (sr / 2.0)
        w0 = float(np.clip(w0, 1e-4, 0.9999))
        try:
            b, a = iirpeak(w0, band.q)
            # Apply gain by scaling numerator
            gain_linear = 10 ** (band.gain_db / 20.0)
            b = b * gain_linear
            y = lfilter(b, a, y)
        except ValueError:
            # Skip unstable/out-of-range bands
            continue
    return y


def match_spectral_tilt(ref_mag: np.ndarray, freqs: np.ndarray) -> List[EqBand]:
    """
    Derive EQ curve by comparing reference to typical vocal spectrum.
    More conservative and accurate matching.
    """
    bands: List[EqBand] = []
    # 8 bands covering vocal range
    edges = np.array([60, 120, 250, 500, 1000, 2000, 4000, 8000, min(freqs[-1], 20000)])
    
    # Get reference band energies (use median for stability)
    ref_energies_db = []
    centers = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            continue
        # Use median for stable estimate (less sensitive to peaks)
        energy = np.median(ref_mag[idx])
        ref_energies_db.append(20 * np.log10(max(energy, 1e-9)))
        centers.append(np.sqrt(lo * hi))
    
    if not ref_energies_db:
        return bands
    
    # Typical vocal spectrum (pink noise-like with slight presence boost)
    # This is a baseline to compare against
    typical_vocal = np.array([
        -3.0,  # 60-120: sub (rolled off)
        -1.0,  # 120-250: low (slight presence)
        0.0,   # 250-500: low-mid (body)
        1.0,   # 500-1000: mid (presence)
        2.0,   # 1000-2000: high-mid (clarity)
        1.5,   # 2000-4000: presence (sibilance)
        0.0,   # 4000-8000: air (rolled off)
        -2.0,  # 8000+: high (rolled off)
    ])
    
    # Compare reference to typical vocal
    # Only use as many bands as we have
    n_bands = min(len(ref_energies_db), len(typical_vocal))
    typical = typical_vocal[:n_bands]
    ref_db = np.array(ref_energies_db[:n_bands])
    
    # Normalize both to zero-mean for fair comparison
    ref_mean = np.mean(ref_db)
    typical_mean = np.mean(typical)
    ref_normalized = ref_db - ref_mean
    typical_normalized = typical - typical_mean
    
    # Difference = what to apply to match reference
    diff = ref_normalized - typical_normalized
    
    # Apply conservative smoothing and clamping
    # Smooth with moving average to avoid extreme jumps
    if len(diff) > 2:
        smoothed = np.convolve(diff, [0.25, 0.5, 0.25], mode='same')
        diff = smoothed
    
    # Clamp to reasonable range (±6 dB max, more conservative)
    for i, (c, gain_diff) in enumerate(zip(centers[:n_bands], diff)):
        gain_db = float(np.clip(gain_diff, -6.0, 6.0))
        # Skip bands with very small changes (< 0.5 dB)
        if abs(gain_db) < 0.5:
            continue
        # Adaptive Q: wider for lows, tighter for highs
        q = float(np.clip(0.8 + (c / 10000.0) * 0.4, 0.8, 1.2))
        bands.append(EqBand(f=c, gain_db=gain_db, q=q))
    
    return bands


def design_eq_from_mel_diff(
    mel_ref: np.ndarray,
    mel_dry: np.ndarray,
    mel_frequencies: np.ndarray,
    sr: int,
    num_bands: int = 16,
    max_gain_db: float = 6.0,
) -> List[EqBand]:
    """
    Derive parametric EQ bands from mel-band differences between reference and dry.

    mel_ref / mel_dry are power spectra (mel_spectrogram with power=2.0).
    We convert to dB before differencing so the gain values are in real dB.

    Uses 16 bands by default (up from 8) for finer spectral resolution, with
    narrower Q values in the mid/high range to avoid broad smearing.
    """
    bands: List[EqBand] = []

    if len(mel_ref) < 2 or len(mel_dry) < 2:
        return bands

    # Convert power mel bands to dB (10 * log10 for power)
    ref_db = 10.0 * np.log10(np.maximum(mel_ref, 1e-10))
    dry_db = 10.0 * np.log10(np.maximum(mel_dry, 1e-10))

    # Difference in dB = gain to apply
    diff = ref_db - dry_db

    # Remove overall loudness offset — tonal shaping only, not level change
    diff = diff - np.mean(diff)

    # Smooth with Savitzky-Golay to reduce noisy spikes between adjacent mel bands
    try:
        from scipy.signal import savgol_filter
        win = min(11, len(diff) if len(diff) % 2 == 1 else len(diff) - 1)
        if win >= 5:
            diff = savgol_filter(diff, win, 3)
    except Exception:
        pass

    # Pick the most significant boosts and cuts (top num_bands/2 each)
    n_each = num_bands // 2
    pos_indices = np.argsort(diff)[::-1]
    neg_indices = np.argsort(diff)

    selected = []
    seen_freqs: List[float] = []

    def _too_close(f: float, thresh_oct: float = 0.25) -> bool:
        """Skip a frequency that's within thresh_oct octaves of an already-chosen band."""
        for sf in seen_freqs:
            if sf > 0 and abs(np.log2(f / sf)) < thresh_oct:
                return True
        return False

    for idx in pos_indices[:n_each * 2]:          # overshoot, filter duplicates
        if len([x for x in selected if x[1] > 0]) >= n_each:
            break
        f = float(mel_frequencies[idx])
        if f < 30 or f > sr / 2:
            continue
        if not _too_close(f):
            selected.append((idx, diff[idx]))
            seen_freqs.append(f)

    for idx in neg_indices[:n_each * 2]:
        if len([x for x in selected if x[1] < 0]) >= n_each:
            break
        f = float(mel_frequencies[idx])
        if f < 30 or f > sr / 2:
            continue
        if not _too_close(f):
            selected.append((idx, diff[idx]))
            seen_freqs.append(f)

    # Scale gains proportionally — preserve curve shape, no hard clipping
    raw_vals = [v for _, v in selected]
    if raw_vals:
        abs_max = max(abs(v) for v in raw_vals)
        scale = (max_gain_db / abs_max) if abs_max > max_gain_db else 1.0
    else:
        scale = 1.0

    for idx, val in selected:
        f = float(mel_frequencies[idx])
        if f < 30 or f > sr / 2:
            continue
        gain_db = float(val * scale)
        if abs(gain_db) < 0.25:          # skip negligible gains
            continue
        # Narrower Q at higher frequencies for surgical precision:
        #   sub/low:     Q ≈ 0.7  (wide, musical)
        #   low-mid:     Q ≈ 1.0
        #   mid/presence:Q ≈ 1.4
        #   high:        Q ≈ 1.8  (tight, surgical)
        q = float(np.clip(0.7 + (np.log10(max(f, 60)) / np.log10(sr / 2)) * 1.1, 0.7, 1.8))
        bands.append(EqBand(f=f, gain_db=gain_db, q=q))

    return bands


