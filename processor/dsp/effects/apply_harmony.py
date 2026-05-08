import numpy as np
import librosa
from processor.dsp.analysis.harmony_analysis import HarmonyProfile


def _pan_stereo(layer: np.ndarray, pan: float) -> np.ndarray:
    """
    Simple stereo panning. pan in [-1,1]; -1=left, +1=right.
    Returns stereo signal (2, N).
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    # Constant-power panning
    left_gain = np.cos((pan + 1) * np.pi / 4)
    right_gain = np.sin((pan + 1) * np.pi / 4)
    return np.vstack([layer * left_gain, layer * right_gain])


def apply_harmony(dry_vocal: np.ndarray, sr: int, profile: HarmonyProfile) -> np.ndarray:
    """
    Generate harmony layers and mix with dry vocal.

    Args:
        dry_vocal: np.ndarray, shape (n,) or (2, n)
        sr: int, sample rate
        profile: HarmonyProfile from detect_harmonies()

    Returns:
        np.ndarray: harmonized output, stereo if input is stereo
    """
    # Ensure mono for processing layers
    if dry_vocal.ndim == 2:
        dry_mono = np.mean(dry_vocal, axis=0)
    else:
        dry_mono = dry_vocal

    harmonized = dry_mono.copy()
    layers = []

    for interval, strength, pan, offset in zip(
        profile.intervals_semitones,
        profile.strengths,
        profile.pans,
        profile.timing_offsets,
    ):
        # Pitch-shift dry vocal
        try:
            harmony_layer = librosa.effects.pitch_shift(dry_mono, sr=sr, n_steps=interval)
        except Exception:
            # Fallback: if pitch_shift fails, skip this layer
            continue

        # Apply strength (volume)
        harmony_layer = harmony_layer * float(np.clip(strength, 0.0, 1.0))

        # Apply offset (circular shift)
        offset_samples = int(offset * sr)
        if offset_samples > 0:
            harmony_layer = np.pad(harmony_layer, (offset_samples, 0))[: len(harmonized)]
        elif offset_samples < 0:
            harmony_layer = np.pad(harmony_layer, (0, -offset_samples))[-offset_samples : len(harmonized)]
        else:
            harmony_layer = harmony_layer[: len(harmonized)]

        layers.append((harmony_layer, pan))

    # Mix layers with dry vocal (mono bus)
    for layer, _pan in layers:
        harmonized[: len(layer)] += layer

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmonized)) + 1e-9
    if peak > 1.0:
        harmonized = harmonized / peak * 0.95

    # If input was stereo, return stereo and pan each layer
    if dry_vocal.ndim == 2:
        # Start with dry vocal stereo
        out = dry_vocal.copy()
        # Add panned layers
        for layer, _pan in layers:
            layer_st = _pan_stereo(layer, _pan)
            # Match length
            min_len = min(out.shape[1], layer_st.shape[1])
            out[:, :min_len] += layer_st[:, :min_len]
        # Normalize stereo
        peak = np.max(np.abs(out)) + 1e-9
        if peak > 1.0:
            out = out / peak * 0.95
        return out.astype(np.float32)

    return harmonized.astype(np.float32)

