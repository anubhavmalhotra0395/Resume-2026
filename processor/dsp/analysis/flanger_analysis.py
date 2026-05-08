import numpy as np
import librosa
from dataclasses import dataclass, asdict
from scipy.signal import find_peaks


@dataclass
class FlangerProfile:
    rate_hz: float
    depth_ms: float
    base_delay_ms: float
    feedback: float
    mix: float

    def as_dict(self):
        return asdict(self)


def _estimate_base_delay_ms(spec_mag: np.ndarray, sr: int, n_fft: int) -> float:
    """
    Estimate base delay from comb spacing in the magnitude spectrum.
    Uses autocorrelation across frequency to find periodic notches.
    """
    # Average over time
    avg_spec = np.mean(spec_mag, axis=1)
    avg_spec = avg_spec / (np.max(avg_spec) + 1e-9)

    # Autocorrelation across frequency bins
    corr = np.correlate(avg_spec, avg_spec, mode="full")
    corr = corr[len(corr) // 2 :]

    # Ignore lag 0; search a reasonable spacing range (100 Hz to 10 kHz)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    min_hz = 100.0
    max_hz = 10000.0
    min_lag = max(1, int((min_hz / (sr / 2)) * (len(avg_spec))))
    max_lag = int((max_hz / (sr / 2)) * (len(avg_spec)))
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return 1.0

    search = corr[min_lag:max_lag]
    peaks, _ = find_peaks(search)
    if peaks.size == 0:
        return 1.0

    best_peak = peaks[np.argmax(search[peaks])]
    lag_bins = min_lag + best_peak
    # Convert lag bins to frequency spacing
    spacing_hz = (lag_bins / len(avg_spec)) * (sr / 2)
    if spacing_hz <= 0:
        return 1.0
    delay_s = 1.0 / spacing_hz
    delay_ms = delay_s * 1000.0
    return float(np.clip(delay_ms, 0.1, 5.0))


def _estimate_modulation_rate(delay_series_ms: np.ndarray, sr_env: float) -> float:
    """
    Estimate modulation rate from a delay vs time series via autocorrelation.
    """
    if len(delay_series_ms) < 4:
        return 0.5
    series = delay_series_ms - np.mean(delay_series_ms)
    if np.allclose(series, 0):
        return 0.5
    corr = np.correlate(series, series, mode="full")
    corr = corr[len(corr) // 2 :]
    # Search 0.05–10 Hz
    min_hz, max_hz = 0.05, 10.0
    min_lag = int(sr_env / max_hz)
    max_lag = int(sr_env / min_hz)
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return 0.5
    search = corr[min_lag:max_lag]
    peaks, _ = find_peaks(search)
    if peaks.size == 0:
        return 0.5
    best_peak = peaks[np.argmax(search[peaks])]
    lag = min_lag + best_peak
    rate_hz = sr_env / lag
    return float(np.clip(rate_hz, 0.1, 5.0))


def detect_flanger(reference_audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> FlangerProfile:
    """
    Detect flanger-like modulation characteristics from reference audio.
    """
    # Ensure mono for analysis
    if reference_audio.ndim > 1:
        ref = np.mean(reference_audio, axis=0)
    else:
        ref = reference_audio

    # STFT
    S = librosa.stft(ref, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)

    # Base delay from comb spacing
    base_delay_ms = _estimate_base_delay_ms(mag, sr, n_fft)

    # Per-frame delay estimate: use spectral autocorr per frame to build series
    delays = []
    for i in range(mag.shape[1]):
        d = _estimate_base_delay_ms(mag[:, i : i + 1], sr, n_fft)
        delays.append(d)
    delays = np.array(delays)
    sr_env = sr / hop_length

    # Modulation rate
    rate_hz = _estimate_modulation_rate(delays, sr_env)

    # Depth: peak-to-peak of delay series normalized by 5 ms max
    if delays.size > 0:
        depth_ms = float(np.clip((np.max(delays) - np.min(delays)), 0.1, 5.0))
    else:
        depth_ms = 1.0

    depth_norm = float(np.clip(depth_ms / 5.0, 0.0, 1.0))

    # Feedback estimate: spectral sharpness (kurtosis proxy)
    # Hard cap at 0.5 — values near 0.95 cause near-oscillation and metallic artifacts
    avg_spec = np.mean(mag, axis=1)
    avg_spec = avg_spec / (np.max(avg_spec) + 1e-9)
    feedback = float(np.clip((np.max(avg_spec) - np.mean(avg_spec)) / (np.std(avg_spec) + 1e-6), 0.0, 0.5))

    # Mix estimate: side energy vs total (heuristic)
    # Cap at 0.4 and only apply if meaningful stereo decorrelation is present
    if reference_audio.ndim > 1 and reference_audio.shape[0] > 1:
        mid = np.mean(reference_audio, axis=0)
        side = (reference_audio[0] - reference_audio[1]) / 2.0
        mid_e = np.mean(mid ** 2) + 1e-9
        side_e = np.mean(side ** 2) + 1e-9
        raw_mix = side_e / (mid_e + side_e)
        mix = float(np.clip(raw_mix, 0.0, 0.4))
        if mix < 0.08:  # mono or near-mono source: skip flanger
            mix = 0.0
    else:
        mix = 0.0  # mono source: don't apply flanger

    return FlangerProfile(
        rate_hz=rate_hz,
        depth_ms=depth_ms,
        base_delay_ms=base_delay_ms,
        feedback=feedback,
        mix=mix,
    )

