import numpy as np
import librosa
from dataclasses import dataclass, asdict
from scipy.signal import find_peaks

# How concentrated the side-envelope spectrum must be in the 0.2-5 Hz band
# before it counts as an LFO rather than ordinary phrasing. Peak-to-mean of a
# flat (phrasing-driven) spectrum sits near 5-10; a real chorus LFO spikes far
# above that.
_LFO_PEAKINESS_THRESHOLD = 18.0


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
    DISCONNECTED 2026-09-09: always reports no chorus.

    Reported by ear as a robotic, warbling vocal, and confirmed by A/B render:
    disabling chorus was the only change that removed it.

    Why this is off rather than retuned:

    1. It fired at MAXIMUM on untreated material. `_detect_chorus` passes the
       centre-isolated lead, which is MONO. With no stereo to analyse the old
       code fabricated a side channel by subtracting a 2 ms delayed copy of
       the signal from itself -- a comb filter, large for any broadband source
       and with an envelope that tracks ordinary phrasing. Both gates then
       passed and the mix pinned to its 0.25 cap. Measured: a raw dry acapella
       and a lead vocal with no chorus on it BOTH reported mix=0.250.
    2. Gate 2 measured VARIANCE, not periodicity. A vocal's side level is near
       zero between phrases and loud while singing, so its coefficient of
       variation is high for every real vocal, LFO or not.
    3. No replacement statistic works. Two principled candidates were measured
       against real material with and without a known chorus:
         - side-envelope spectral peakiness: reference lead 7.9 WITHOUT chorus
           and 7.1-7.4 WITH it -- adding chorus slightly LOWERS it.
         - inter-channel cross-correlation lag wobble: 0.30-0.90 ms on
           untreated stems, 0.01-0.17 ms with a real chorus -- inverted.
       Neither responds to the thing it claims to measure.
    4. The applier is mono anyway: apply_chorus sums both modulated delay
       lines and mixes them with the dry signal in one channel, which is comb
       filtering with a moving notch -- robotic by construction, and unable to
       produce the stereo modulation this detector looks for.

    Detection would need a real method (e.g. tracking per-partial frequency
    modulation, or the delay trajectory of a matched second voice). Until then
    the honest output is "no evidence", per the project rule to prefer no
    correction over a fabricated number. The gate code below is preserved for
    whoever implements that.
    """
    return ChorusProfile(rate_hz=0.8, depth=0.0, mix=0.0)


def _detect_chorus_unused(reference: np.ndarray, sr: int) -> ChorusProfile:
    """Preserved for reference; see detect_chorus for why it is not used."""

    # ── Mid/Side split ─────────────────────────────────────────────────────
    if reference.ndim == 2 and reference.shape[0] > 1:
        mid  = np.mean(reference, axis=0)
        side = (reference[0] - reference[1]) / 2.0
    elif reference.ndim == 2 and reference.shape[0] == 1:
        mid  = reference[0]
        side = np.zeros_like(mid)
    else:
        # MONO input carries no stereo modulation evidence, so there is
        # nothing here to detect and the honest answer is "none".
        #
        # This used to fabricate a side channel by subtracting a 2 ms delayed
        # copy of the signal from itself -- a comb filter whose output is
        # large for any broadband source and whose envelope tracks ordinary
        # phrasing. Both gates then passed and the mix pinned to its 0.25 cap.
        # Measured: an untreated dry acapella and a lead vocal with no chorus
        # on it BOTH reported mix=0.250, the maximum. Every job was getting a
        # full-depth chorus, which is audible as a robotic, warbling vocal.
        return ChorusProfile(rate_hz=0.8, depth=0.0, mix=0.0)

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

    # Gate 2: the modulation must be PERIODIC, not merely large.
    #
    # The coefficient of variation was used here, but variance is not
    # periodicity: a vocal's side level is near zero between phrases and loud
    # while singing, so its CV is high for every real vocal whether or not an
    # LFO is present. Test instead whether the envelope's spectrum has a
    # concentrated peak in the LFO band -- an actual oscillator puts its energy
    # at one rate, while phrasing spreads it across the band.
    env_cv = float(np.std(env_ds) / (np.mean(env_ds) + 1e-9))
    _e = env_ds - np.mean(env_ds)
    if len(_e) < 64 or not np.any(_e):
        return ChorusProfile(rate_hz=rate_hz, depth=0.0, mix=0.0)
    _spec = np.abs(np.fft.rfft(_e * np.hanning(len(_e))))
    _fr = np.fft.rfftfreq(len(_e), 1.0 / env_sr_ds)
    _band = (_fr >= 0.2) & (_fr <= 5.0)
    if not _band.any() or _spec[_band].sum() <= 0:
        return ChorusProfile(rate_hz=rate_hz, depth=0.0, mix=0.0)
    # Peak-to-mean of the LFO band: ~1 is flat (phrasing), a real LFO spikes.
    peakiness = float(_spec[_band].max() / (np.mean(_spec[_band]) + 1e-12))
    rate_hz = float(np.clip(_fr[_band][np.argmax(_spec[_band])], 0.2, 5.0))
    if peakiness < _LFO_PEAKINESS_THRESHOLD:
        return ChorusProfile(rate_hz=rate_hz, depth=float(np.clip(env_cv, 0.0, 1.0)),
                             mix=0.0)

    # ── Confirmed chorus — scale mix from modulation depth ────────────────
    depth = float(np.clip(env_cv, 0.0, 1.0))
    # Mix: proportional to side spread but anchored to modulation evidence
    # Cap at 0.25 for vocals — any higher washes out intelligibility
    raw_mix = side_ratio / (1.0 + side_ratio)   # compress from 0..∞ to 0..1
    mix = float(np.clip(raw_mix * 0.5, 0.05, 0.25))

    return ChorusProfile(rate_hz=rate_hz, depth=depth, mix=mix)

