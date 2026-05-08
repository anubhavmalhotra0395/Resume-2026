"""
Vocal Layer Replicator — applies exactly the number of layers detected from the reference.

Two layer types:
  1. Doubling layers  — same pitch, tiny detune + delay + pan (ADT stack)
  2. Harmony voices   — different pitch, mixed at the detected strength

The output is always stereo (2, N) so each layer can be independently panned.
If the input is mono, it is first widened to stereo at centre (0 pan).
"""
from __future__ import annotations

import numpy as np
import librosa
from typing import Optional

from processor.dsp.analysis.vocal_layers_analysis import VocalLayersProfile


def _to_stereo(y: np.ndarray) -> np.ndarray:
    """Ensure signal is (2, N) stereo."""
    if y.ndim == 1:
        return np.stack([y, y], axis=0).astype(np.float32)
    if y.ndim == 2 and y.shape[0] == 2:
        return y.astype(np.float32)
    if y.ndim == 2 and y.shape[1] == 2:
        return y.T.astype(np.float32)
    return np.stack([y[0], y[0]], axis=0).astype(np.float32)


def _pan_layer(mono_layer: np.ndarray, pan: float, n: int) -> np.ndarray:
    """Pan a mono layer to stereo (2, N). pan: -1=left, 0=centre, +1=right."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * np.pi / 4.0  # 0..π/2
    left_gain  = float(np.cos(angle))
    right_gain = float(np.sin(angle))
    layer = mono_layer[:n]
    pad   = n - len(layer)
    if pad > 0:
        layer = np.pad(layer, (0, pad))
    return np.stack([layer * left_gain, layer * right_gain], axis=0).astype(np.float32)


def _pitch_shift_safe(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Pitch shift with fallback if steps are too large."""
    if abs(n_steps) < 0.01:
        return y.copy()
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)
    except Exception:
        return y.copy()


def _detune_layer(y: np.ndarray, sr: int, cents: float, delay_ms: float) -> np.ndarray:
    """
    Create one ADT double: pitch shift by `cents` + apply `delay_ms` offset.
    Keeps the same length as input.
    """
    n_steps = cents / 100.0
    shifted = _pitch_shift_safe(y, sr, n_steps)

    delay_samples = int(sr * delay_ms / 1000.0)
    if delay_samples > 0 and delay_samples < len(shifted):
        out = np.zeros_like(shifted)
        out[delay_samples:] = shifted[:-delay_samples]
        return out
    return shifted


def build_vocal_layer_stems(
    dry: np.ndarray,
    sr: int,
    profile: VocalLayersProfile,
    doubler_mix: float = 0.40,
) -> dict[str, np.ndarray]:
    """
    Build per-layer stems (channels-last) for preview/audition.

    Returns:
        dict with keys like:
          - lead
          - doubler_1, doubler_2, ...
          - harmony_1, harmony_2, ...
    """
    st = _to_stereo(dry)
    n = st.shape[1]
    mono = np.mean(st, axis=0).astype(np.float32)

    stems: dict[str, np.ndarray] = {
        "lead": st.T.astype(np.float32),
    }

    for i in range(profile.n_doublers):
        if i >= len(profile.doubler_detunes_cents):
            break
        cents = profile.doubler_detunes_cents[i]
        delay_ms = profile.doubler_delays_ms[i] if i < len(profile.doubler_delays_ms) else 12.0 + i * 6.0
        pan = profile.doubler_pans[i] if i < len(profile.doubler_pans) else (0.5 if i % 2 == 0 else -0.5)
        layer_mono = _detune_layer(mono, sr, cents, delay_ms)
        layer_st = _pan_layer(layer_mono, pan, n) * float(np.clip(doubler_mix, 0.0, 1.0))
        stems[f"doubler_{i+1}"] = layer_st.T.astype(np.float32)

    for i, (interval, strength, pan) in enumerate(
        zip(profile.harmony_intervals, profile.harmony_strengths, profile.harmony_pans),
        start=1,
    ):
        harmony_mono = _pitch_shift_safe(mono, sr, float(interval))
        harmony_st = _pan_layer(harmony_mono, pan, n) * float(np.clip(strength, 0.05, 0.5))
        stems[f"harmony_{i}"] = harmony_st.T.astype(np.float32)

    return stems


def filter_vocal_layers_profile(profile: VocalLayersProfile, selected_layers: list[str]) -> VocalLayersProfile | None:
    """
    Keep only selected doubler/harmony layers.
    `lead` is always part of the base signal, so it is not filtered here.
    """
    if not selected_layers:
        return profile

    selected = set(selected_layers)
    keep_d = [f"doubler_{i+1}" in selected for i in range(profile.n_doublers)]
    keep_h = [f"harmony_{i+1}" in selected for i in range(len(profile.harmony_intervals))]

    n_doublers = sum(keep_d)
    h_count = sum(keep_h)
    if n_doublers == 0 and h_count == 0:
        return None

    return VocalLayersProfile(
        n_doublers=n_doublers,
        doubler_detunes_cents=[v for v, keep in zip(profile.doubler_detunes_cents, keep_d) if keep],
        doubler_delays_ms=[v for v, keep in zip(profile.doubler_delays_ms, keep_d) if keep],
        doubler_pans=[v for v, keep in zip(profile.doubler_pans, keep_d) if keep],
        harmony_intervals=[v for v, keep in zip(profile.harmony_intervals, keep_h) if keep],
        harmony_strengths=[v for v, keep in zip(profile.harmony_strengths, keep_h) if keep],
        harmony_pans=[v for v, keep in zip(profile.harmony_pans, keep_h) if keep],
    )


def apply_vocal_layers(
    dry: np.ndarray,
    sr: int,
    profile: VocalLayersProfile,
    doubler_mix: float = 0.40,   # how loud each doubling layer is relative to lead
) -> np.ndarray:
    """
    Apply exactly the layers described in `profile` to the dry vocal.

    Args:
        dry          : input vocal, mono (N,) or stereo (2, N)
        sr           : sample rate
        profile      : VocalLayersProfile from detect_vocal_layers()
        doubler_mix  : volume of each doubling copy (0–1)

    Returns:
        stereo (2, N) array with all layers mixed together
    """
    # Work in stereo throughout
    st = _to_stereo(dry)
    n  = st.shape[1]

    # Mono lead for generating layers
    mono = np.mean(st, axis=0).astype(np.float32)

    # Start with the dry lead at centre
    out = st.copy()

    # ── Doubling layers ────────────────────────────────────────────────────
    for i in range(profile.n_doublers):
        if i >= len(profile.doubler_detunes_cents):
            break
        cents    = profile.doubler_detunes_cents[i]
        delay_ms = profile.doubler_delays_ms[i] if i < len(profile.doubler_delays_ms) else 12.0 + i * 6.0
        pan      = profile.doubler_pans[i]       if i < len(profile.doubler_pans)      else (0.5 if i % 2 == 0 else -0.5)

        layer_mono = _detune_layer(mono, sr, cents, delay_ms)
        layer_st   = _pan_layer(layer_mono, pan, n)
        out += layer_st * doubler_mix

    # ── Harmony voices ─────────────────────────────────────────────────────
    for interval, strength, pan in zip(
        profile.harmony_intervals,
        profile.harmony_strengths,
        profile.harmony_pans,
    ):
        harmony_mono = _pitch_shift_safe(mono, sr, float(interval))
        harmony_st   = _pan_layer(harmony_mono, pan, n)
        out += harmony_st * float(np.clip(strength, 0.05, 0.5))

    # Normalise to prevent clipping while preserving relative levels
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out = out / peak * 0.97

    # Return (N, 2) — soundfile / downstream code expects channels-last
    return out.T.astype(np.float32)
