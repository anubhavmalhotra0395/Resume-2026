from dataclasses import dataclass
from typing import List, Optional

import librosa
import numpy as np
from scipy.signal import find_peaks

from processor.dsp.compressor import CompressorSettings
from processor.dsp.eq import EqBand, match_spectral_tilt, design_eq_from_mel_diff
from processor.dsp.reverb import ReverbSettings


@dataclass
class Recipe:
    eq: Optional[List[EqBand]]          # None → no meaningful EQ difference detected
    compressor: Optional[CompressorSettings]  # None → not significantly compressed
    reverb: Optional[ReverbSettings]    # None → no reverb detected
    saturation_drive: Optional[float]   # None → no saturation detected
    width: Optional[dict] = None        # None → mono/no stereo width detected


def _estimate_compression(y: np.ndarray, sr: int, dry_y: np.ndarray | None = None) -> CompressorSettings:
    """
    Estimate compression by comparing reference to dry vocal.
    Lower variance in reference = more compression applied.
    """
    ref_rms = librosa.feature.rms(y=y, hop_length=512, frame_length=2048)[0]
    ref_rms_db = 20 * np.log10(np.maximum(ref_rms, 1e-9))
    
    if dry_y is not None and len(dry_y) > 0:
        # Compare to dry vocal to see what compression was applied
        dry_rms = librosa.feature.rms(y=dry_y, hop_length=512, frame_length=2048)[0]
        dry_rms_db = 20 * np.log10(np.maximum(dry_rms, 1e-9))
        
        # Normalize lengths for comparison
        min_len = min(len(ref_rms_db), len(dry_rms_db))
        ref_rms_db = ref_rms_db[:min_len]
        dry_rms_db = dry_rms_db[:min_len]
        
        # Variance reduction = compression amount (more sensitive calculation)
        ref_var = float(np.var(ref_rms_db))
        dry_var = float(np.var(dry_rms_db))
        var_reduction = (dry_var - ref_var) / (dry_var + 1e-9)
        var_reduction = float(np.clip(var_reduction, 0.0, 1.0))
        
        # Crest factor reduction = compression strength
        ref_crest = float(np.percentile(ref_rms_db, 90) - np.percentile(ref_rms_db, 10))
        dry_crest = float(np.percentile(dry_rms_db, 90) - np.percentile(dry_rms_db, 10))
        crest_reduction = (dry_crest - ref_crest) / (dry_crest + 1e-9)
        crest_reduction = float(np.clip(crest_reduction, 0.0, 1.0))
        
        # Average level difference = makeup gain
        ref_mean = float(np.mean(ref_rms_db))
        dry_mean = float(np.mean(dry_rms_db))
        makeup_db = ref_mean - dry_mean
        
        # Estimate threshold more accurately
        # Find where compression starts by looking at RMS difference
        diff = ref_rms_db - dry_rms_db
        # Compression starts where ref is consistently lower than dry
        compressed_regions = np.where(diff < -1.0)[0]  # Where ref is 1dB+ lower
        if len(compressed_regions) > 0:
            compressed_rms = ref_rms_db[compressed_regions]
            threshold = float(np.percentile(compressed_rms, 50))  # Median of compressed regions
        else:
            threshold = float(np.percentile(ref_rms_db, 30))
        
        # CRITICAL: Clamp threshold to reasonable range for vocals (-40 to -8 dB)
        threshold = float(np.clip(threshold, -40.0, -8.0))
        
        # Ratio: more sensitive to variance reduction (wider range)
        # Map var_reduction to ratio: 0.0 -> 2.0, 1.0 -> 8.0
        ratio = float(2.0 + var_reduction * 6.0)
        ratio = float(np.clip(ratio, 2.0, 8.0))
        
        # Attack: more sensitive to crest reduction
        # High crest reduction = fast attack (tight compression)
        attack = float(np.clip(2 + (1 - crest_reduction) * 18, 2, 20))
        
        # Release: based on remaining variance (more sensitive)
        # Lower remaining variance = longer release (smoother)
        remaining_var = ref_var / (dry_var + 1e-9)
        release = float(np.clip(60 + (1 - remaining_var) * 180, 60, 240))
        
    else:
        # Fallback: analyze reference alone
        p40 = float(np.percentile(ref_rms_db, 40))
        p90 = float(np.percentile(ref_rms_db, 90))
        crest = p90 - p40
        var = float(np.var(ref_rms_db))
        
        threshold = float(np.clip(p40, -40.0, -8.0))
        ratio = np.clip(1.5 + crest * 0.15, 2.0, 6.5)
        attack = np.clip(4 + (crest * 1.5), 2, 25)
        release = np.clip(80 + var * 6, 60, 240)
        makeup_db = np.clip((0 - threshold) * 0.25, 0, 6)
    
    # Clamp makeup gain conservatively
    makeup_db = float(np.clip(makeup_db, 0, 5))
    
    return CompressorSettings(
        threshold_db=float(threshold),
        ratio=ratio,
        attack_ms=attack,
        release_ms=release,
        makeup_db=makeup_db,
    )


def _estimate_reverb(y: np.ndarray, sr: int, dry_y: np.ndarray | None = None) -> ReverbSettings:
    """
    Estimate reverb by comparing reference to dry vocal.
    Extra energy/tail in reference = reverb that was added.
    """
    if dry_y is not None and len(dry_y) > 0:
        # Compare reference to dry to find reverb characteristics
        min_len = min(len(y), len(dry_y))
        ref_comp = y[:min_len]
        dry_comp = dry_y[:min_len]
        
        # Analyze spectral difference (reverb adds energy, especially in highs)
        S_ref = np.abs(librosa.stft(ref_comp, n_fft=2048, hop_length=512))
        S_dry = np.abs(librosa.stft(dry_comp, n_fft=2048, hop_length=512))
        
        # High frequencies show reverb more clearly
        high_freq_start = S_ref.shape[0] // 3
        ref_high = S_ref[high_freq_start:, :]
        dry_high = S_dry[high_freq_start:, :]
        
        ref_high_energy = np.sum(ref_high, axis=0)
        dry_high_energy = np.sum(dry_high, axis=0)
        
        # Normalize lengths
        min_frames = min(len(ref_high_energy), len(dry_high_energy))
        ref_high_energy = ref_high_energy[:min_frames]
        dry_high_energy = dry_high_energy[:min_frames]
        
        # Extra energy in reference = reverb tail
        extra_energy = ref_high_energy - dry_high_energy
        extra_energy = np.maximum(extra_energy, 0)  # Only positive differences
        
        # Energy decay curve (EDC) to estimate RT60
        edc = np.flip(np.cumsum(np.flip(extra_energy)))
        edc_db = librosa.amplitude_to_db(edc, ref=np.max(edc) + 1e-9)
        t = np.arange(len(edc_db)) / (sr / 512)
        if len(edc_db) > 10:
            slope, _ = np.polyfit(t, edc_db, 1)
            # Cap decay at 1.5s for vocals (longer decays drown out the signal)
            rt60 = float(np.clip(-60.0 / (slope + 1e-9), 0.2, 1.5))
        else:
            rt60 = 0.8
        decay = rt60
        
        # Mix: ratio of extra energy to total reference energy (more sensitive)
        total_ref_energy = np.sum(ref_high_energy)
        total_extra = np.sum(extra_energy)
        if total_ref_energy > 1e-9:
            mix_ratio = float(total_extra / total_ref_energy)
            # Cap reverb mix at 0.25 (25%) for vocals - higher values drown out the dry signal
            mix = float(np.clip(mix_ratio * 1.5, 0.05, 0.25))
        else:
            mix = 0.15
        
        # Predelay: simple heuristic based on peaks
        dry_peaks, _ = find_peaks(dry_high_energy, height=np.percentile(dry_high_energy, 70))
        if len(dry_peaks) > 0:
            pre = float(np.clip(10 + len(dry_peaks) * 2, 5, 35))
        else:
            pre = 10.0
    
    else:
        # Fallback: analyze reference alone (conservative)
        rms = librosa.feature.rms(y=y, hop_length=512, frame_length=2048)[0]
        if len(rms) < 10:
            return ReverbSettings(decay_s=0.8, mix=0.15, pre_delay_ms=10)
        
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        peaks, _ = find_peaks(rms_db, height=np.percentile(rms_db, 60), distance=10)
        
        if len(peaks) == 0:
            return ReverbSettings(decay_s=0.8, mix=0.15, pre_delay_ms=10)
        
        decay_times = []
        for peak_idx in peaks[:min(10, len(peaks))]:
            tail = rms_db[peak_idx:]
            below_20 = np.where(tail < rms_db[peak_idx] - 20)[0]
            if below_20.size > 0:
                decay_samples = below_20[0]
                decay_time = float(decay_samples * 512 / sr)
                if 0.2 < decay_time < 2.0:
                    decay_times.append(decay_time)
        
        if decay_times:
            median_decay = float(np.median(decay_times))
            decay = float(np.clip(median_decay * 0.6, 0.4, 1.2))
        else:
            decay = 0.8
        
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        high_freq = S[S.shape[0]//3:, :]
        high_energy = np.sum(high_freq, axis=0)
        
        early_ms = 30
        early_frames = int(early_ms * sr / 1000 / 512)
        if len(high_energy) > early_frames * 2:
            early = np.mean(high_energy[:early_frames])
            late = np.mean(high_energy[early_frames:])
            late_ratio = float(late / (early + 1e-9))
        else:
            late_ratio = 0.1
        
        mix = float(np.clip(0.10 + late_ratio * 0.15, 0.10, 0.25))
        pre = float(np.clip(5 + late_ratio * 15, 5, 25))
    
    return ReverbSettings(decay_s=decay, mix=mix, pre_delay_ms=pre)


def _estimate_eq(y: np.ndarray, sr: int) -> List[EqBand]:
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    mag = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return match_spectral_tilt(mag, freqs)


def _estimate_saturation(y: np.ndarray, dry_y: np.ndarray | None = None) -> float:
    if dry_y is not None and len(dry_y) > 0:
        min_len = min(len(y), len(dry_y))
        ref_comp = y[:min_len]
        dry_comp = dry_y[:min_len]
        # Normalize both to same RMS so we compare timbre not level
        for sig in [ref_comp, dry_comp]:
            rms = np.sqrt(np.mean(sig ** 2))
            if rms > 1e-9:
                sig /= rms
        ref_harm = librosa.effects.harmonic(ref_comp)
        ref_res = ref_comp - ref_harm
        dry_harm = librosa.effects.harmonic(dry_comp)
        dry_res = dry_comp - dry_harm
        ref_ratio = float(np.mean(np.abs(ref_harm)) / (np.mean(np.abs(ref_res)) + 1e-6))
        dry_ratio = float(np.mean(np.abs(dry_harm)) / (np.mean(np.abs(dry_res)) + 1e-6))
        # Higher harmonic ratio in reference means saturation was applied
        ratio_increase = max(0.0, ref_ratio - dry_ratio)
        drive = float(np.clip(1.0 + ratio_increase * 1.2, 1.0, 3.0))
    else:
        harm = librosa.effects.harmonic(y)
        residual = y - harm
        ratio = float(np.mean(np.abs(harm)) / (np.mean(np.abs(residual)) + 1e-6))
        drive = float(np.clip(1.0 + (ratio - 1.0) * 1.5, 1.0, 3.0))
    return drive


def _estimate_width(y: np.ndarray, sr: int) -> dict:
    # Use transient density to set width and delay
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0:
        return {"delay_ms": 12.0, "detune_cents": 4.0, "mix": 0.3}
    peaks, _ = find_peaks(onset_env, height=np.percentile(onset_env, 70))
    density = len(peaks) / max(len(onset_env), 1)
    delay_ms = float(np.clip(8 + density * 30, 8, 22))
    detune_cents = float(np.clip(2 + density * 12, 2, 10))
    mix = float(np.clip(0.25 + density * 0.4, 0.2, 0.6))
    return {"delay_ms": delay_ms, "detune_cents": detune_cents, "mix": mix}


def _has_meaningful_eq(bands: List[EqBand], min_gain_db: float = 0.5) -> bool:
    """Return True if at least one band has a gain change worth applying."""
    return any(abs(b.gain_db) >= min_gain_db for b in bands)


def _has_meaningful_compression(settings: CompressorSettings, dry_y: np.ndarray | None) -> bool:
    """
    Return True if reference shows actual compression relative to the dry vocal.
    A ratio of 2:1 or less on a nearly-flat dynamic suggests no real compression.
    """
    if dry_y is None:
        return True  # Can't tell without dry — assume it's there
    return settings.ratio >= 2.5


def _has_meaningful_reverb(settings: ReverbSettings) -> bool:
    """Return True only if reverb mix is above a perceptible threshold."""
    return settings.mix >= 0.07


def _has_meaningful_saturation(drive: float) -> bool:
    """Return True if saturation drive is above unity (i.e. something was actually added)."""
    return drive > 1.15


def _has_meaningful_width(width: dict) -> bool:
    """Return True if the reference shows stereo spread worth replicating."""
    return width.get("mix", 0) >= 0.2


def analyze_reference(y: np.ndarray, sr: int, dry_y: np.ndarray | None = None) -> Recipe:
    """
    Analyze reference to extract processing recipe.
    If dry_y is provided, compare directly for more accurate matching.
    Each effect field is set to None if the reference does not meaningfully have it.
    """
    if dry_y is not None and len(dry_y) > 0:
        min_len = min(len(y), len(dry_y))
        y_comp = y[:min_len]
        dry_comp = dry_y[:min_len]

        ref_rms = np.sqrt(np.mean(y_comp ** 2))
        dry_rms = np.sqrt(np.mean(dry_comp ** 2))
        if ref_rms > 1e-9 and dry_rms > 1e-9:
            target_rms = 10 ** (-20 / 20.0)
            y_comp = y_comp * (target_rms / ref_rms)
            dry_comp = dry_comp * (target_rms / dry_rms)

        mel_ref = librosa.feature.melspectrogram(
            y=y_comp, sr=sr, n_fft=4096, hop_length=512, n_mels=64, power=2.0
        )
        mel_dry = librosa.feature.melspectrogram(
            y=dry_comp, sr=sr, n_fft=4096, hop_length=512, n_mels=64, power=2.0
        )
        mel_f = librosa.mel_frequencies(n_mels=64, fmin=30, fmax=sr / 2)
        eq_bands_raw = design_eq_from_mel_diff(
            mel_ref=np.mean(mel_ref, axis=1),
            mel_dry=np.mean(mel_dry, axis=1),
            mel_frequencies=mel_f,
            sr=sr,
            num_bands=16,
            max_gain_db=6.0,
        )

        compressor_raw = _estimate_compression(y_comp, sr, dry_y=dry_comp)
        reverb_raw     = _estimate_reverb(y_comp, sr, dry_y=dry_comp)
    else:
        eq_bands_raw   = _estimate_eq(y, sr)
        compressor_raw = _estimate_compression(y, sr, dry_y=None)
        reverb_raw     = _estimate_reverb(y, sr, dry_y=None)

    saturation_raw = _estimate_saturation(y, dry_y=dry_y)
    width_raw      = _estimate_width(y, sr)

    # Guard each effect: only include it if meaningfully detected
    eq_out         = eq_bands_raw   if _has_meaningful_eq(eq_bands_raw)                    else None
    compressor_out = compressor_raw if _has_meaningful_compression(compressor_raw, dry_y)  else None
    reverb_out     = reverb_raw     if _has_meaningful_reverb(reverb_raw)                  else None
    saturation_out = saturation_raw if _has_meaningful_saturation(saturation_raw)          else None
    width_out      = width_raw      if _has_meaningful_width(width_raw)                    else None

    return Recipe(
        eq=eq_out,
        compressor=compressor_out,
        reverb=reverb_out,
        saturation_drive=saturation_out,
        width=width_out,
    )


def _estimate_eq_from_comparison(ref_mag: np.ndarray, dry_mag: np.ndarray, freqs: np.ndarray) -> List[EqBand]:
    """
    Derive EQ by directly comparing reference to dry vocal.
    More accurate matching using spectral envelope comparison.
    """
    from processor.dsp.eq import EqBand
    
    bands: List[EqBand] = []
    edges = np.array([60, 120, 250, 500, 1000, 2000, 4000, 8000, min(freqs[-1], 20000)])
    
    # Convert to dB for more accurate comparison
    ref_db = 20 * np.log10(ref_mag + 1e-9)
    dry_db = 20 * np.log10(dry_mag + 1e-9)
    
    # Normalize both to zero-mean (removes overall level differences)
    ref_mean = np.mean(ref_db)
    dry_mean = np.mean(dry_db)
    ref_normalized = ref_db - ref_mean
    dry_normalized = dry_db - dry_mean
    
    # Smooth both spectra to reduce noise/artifacts
    from scipy.signal import savgol_filter
    try:
        window = min(51, len(ref_normalized) // 4)
        if window > 5 and window % 2 == 1:
            ref_smooth = savgol_filter(ref_normalized, window, 3)
            dry_smooth = savgol_filter(dry_normalized, window, 3)
        else:
            ref_smooth = ref_normalized
            dry_smooth = dry_normalized
    except:
        ref_smooth = ref_normalized
        dry_smooth = dry_normalized
    
    ref_energies = []
    dry_energies = []
    centers = []
    
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            continue
        
        # Use median of smoothed spectrum for stability
        ref_energy = np.median(ref_smooth[idx])
        dry_energy = np.median(dry_smooth[idx])
        
        ref_energies.append(ref_energy)
        dry_energies.append(dry_energy)
        centers.append(np.sqrt(lo * hi))
    
    if not ref_energies:
        return bands
    
    # Calculate gain needed: difference in dB
    for center, ref_e, dry_e in zip(centers, ref_energies, dry_energies):
        gain_db = ref_e - dry_e  # Direct dB difference
        
        # Clamp to reasonable range
        gain_db = float(np.clip(gain_db, -6.0, 6.0))
        
        # Include all meaningful changes (lower threshold)
        if abs(gain_db) < 0.2:
            continue
        
        # Adaptive Q: wider for lows, tighter for highs
        q = float(np.clip(0.7 + (center / 10000.0) * 0.5, 0.7, 1.3))
        bands.append(EqBand(f=center, gain_db=gain_db, q=q))
    
    return bands


