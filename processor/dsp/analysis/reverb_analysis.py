"""
Reverb analysis utilities.

Estimate RT60, predelay, early/late energy split, and suggested wet amount.
Uses Schroeder method (energy decay curve) and conservative fallbacks.
"""

import numpy as np
import librosa
from dataclasses import dataclass, asdict
from scipy import stats
import logging

log = logging.getLogger("reverb_analysis")
log.setLevel(logging.INFO)


@dataclass
class ReverbProfile:
    rt60: float = 0.8             # seconds
    predelay_ms: float = 20.0     # ms
    early_ratio: float = 0.3      # fraction of early energy (0..1)
    wet: float = 0.25             # suggested wet mix (0..1)
    confidence: float = 0.0

    def as_dict(self):
        return asdict(self)


def _energy_decay_curve(y):
    """Return Schroeder energy decay curve (EDC) in linear units."""
    e = y.astype(np.float64) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc = edc / (np.max(edc) + 1e-12)
    return edc


def estimate_reverb_params(reference_audio: np.ndarray, sr: int) -> ReverbProfile:
    """
    Estimate RT60, predelay, early/late ratio, and wet amount from a reference vocal stem.
    Returns ReverbProfile with conservative defaults on failure.
    """
    profile = ReverbProfile()
    try:
        if reference_audio.ndim == 2:
            mono = np.mean(reference_audio, axis=0)
        else:
            mono = reference_audio

        env = librosa.onset.onset_strength(y=mono, sr=sr)
        if env.size == 0:
            mono_clip = mono
        else:
            idx = int(len(env) * 0.5)
            hop = 512
            center_sample = idx * hop
            start = max(0, center_sample - sr // 2)
            end = min(len(mono), center_sample + sr // 2)
            mono_clip = mono[start:end]

        if mono_clip.size < 1024:
            mono_clip = mono

        edc = _energy_decay_curve(mono_clip)
        edc_db = 10.0 * np.log10(edc + 1e-12)

        max_db = edc_db[0]
        target1 = max_db - 5.0
        target2 = max_db - 35.0

        try:
            i1 = np.where(edc_db <= target1)[0][0]
            i2 = np.where(edc_db <= target2)[0][0]
        except Exception:
            n = len(edc_db)
            i1 = int(n * 0.05)
            i2 = int(n * 0.9)

        if i2 <= i1:
            i2 = min(len(edc_db) - 1, i1 + 10)

        times = np.arange(len(edc_db)) * (len(mono_clip) / float(len(edc_db))) / float(sr)
        x = times[i1:i2]
        ydb = edc_db[i1:i2]

        if len(x) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ydb)
            if slope < -0.01:
                rt60 = -60.0 / slope
                # Vocal reverb rarely exceeds 2.5s — cap tighter than the old 10s
                profile.rt60 = float(np.clip(rt60, 0.1, 2.5))
                profile.confidence = float(np.clip(abs(r_value), 0.0, 1.0))
            else:
                profile.rt60 = 0.8
                profile.confidence = 0.2
        else:
            profile.rt60 = 0.8
            profile.confidence = 0.1

        search_ms = int(min(len(mono_clip), int(sr * 0.1)))
        if search_ms < 128:
            profile.predelay_ms = 10.0
        else:
            onset_env = np.abs(librosa.util.normalize(mono_clip[:search_ms]))
            peak_idx = int(np.argmax(onset_env))
            # Cap predelay at 40 ms — beyond that it sounds like a slap-back delay, not reverb
            profile.predelay_ms = float(np.clip((peak_idx / float(sr)) * 1000.0, 0.0, 40.0))

        early_ms = 50
        early_samples = int(sr * early_ms / 1000.0)
        total_samples = min(len(mono_clip), int(sr * profile.rt60 * 1.2))
        early_energy = np.sum(mono_clip[:early_samples] ** 2) + 1e-12
        late_energy = np.sum(mono_clip[early_samples:total_samples] ** 2) + 1e-12
        early_ratio = float(early_energy / (early_energy + late_energy))
        profile.early_ratio = float(np.clip(early_ratio, 0.05, 0.8))

        # Wet: scale with RT60 but hard-cap at 0.20 for vocals
        # Higher wet values drown the dry signal and make the vocal sound distant
        wet = 0.08 + (profile.rt60 / 2.5) * 0.12
        wet *= (1.0 - profile.early_ratio * 0.3)
        profile.wet = float(np.clip(wet, 0.05, 0.20))

        # Low-confidence measurement: if the EDC didn't fit a clean line,
        # trust the result less and snap to conservative defaults
        if profile.confidence < 0.5:
            profile.rt60 = float(np.clip(profile.rt60, 0.3, 1.2))
            profile.wet  = float(np.clip(profile.wet,  0.05, 0.15))

    except Exception as e:
        log.warning(f"Reverb analysis failed: {e}")
        return ReverbProfile()

    return profile

