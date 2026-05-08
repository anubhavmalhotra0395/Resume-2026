"""
Convolution reverb using pre-synthesised impulse responses (IRs).

Four types bundled as synthesised IRs (no external files needed):
  - plate    : bright, dense, fast-diffusing — classic vocal plate sound
  - room     : small-medium live room
  - hall     : larger space, longer tail
  - chamber  : echo chamber, coloured mid density

The IRs are generated at runtime from carefully tuned Schroeder + FDN
parameters that model the acoustic behaviour of each space. This avoids
shipping large .wav files while still sounding significantly more realistic
than a simple single-comb reverb.

Usage:
    from processor.dsp.conv_reverb import ConvReverbSettings, apply_conv_reverb

    cfg = ConvReverbSettings(reverb_type="plate", rt60=1.2, wet=0.15, pre_delay_ms=20)
    y_wet = apply_conv_reverb(y, sr, cfg)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)

ReverbType = Literal["plate", "room", "hall", "chamber"]

# Cache synthesised IRs so we don't regenerate on every call
_ir_cache: dict[tuple, np.ndarray] = {}


@dataclass
class ConvReverbSettings:
    reverb_type: ReverbType = "plate"
    rt60: float = 1.2          # reverberation time in seconds
    wet: float = 0.15          # wet/dry mix  (0–0.30)
    pre_delay_ms: float = 15.0 # pre-delay in milliseconds


# ---------------------------------------------------------------------------
# IR synthesis helpers
# ---------------------------------------------------------------------------

def _noise_burst(length: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length).astype(np.float32)


def _decay_envelope(length: int, sr: int, rt60: float) -> np.ndarray:
    """Exponential decay envelope that reaches –60 dB at rt60 seconds."""
    t = np.arange(length) / sr
    return np.exp(-6.908 * t / max(rt60, 0.01)).astype(np.float32)  # 6.908 = ln(10^3)


def _comb(sig: np.ndarray, delay: int, g: float) -> np.ndarray:
    out = np.zeros_like(sig)
    for n in range(len(sig)):
        d = n - delay
        out[n] = sig[n] + g * (out[d] if d >= 0 else 0.0)
    return out


def _allpass(sig: np.ndarray, delay: int, g: float) -> np.ndarray:
    out = np.zeros_like(sig)
    for n in range(len(sig)):
        d = n - delay
        xd = sig[d] if d >= 0 else 0.0
        od = out[d] if d >= 0 else 0.0
        out[n] = -g * sig[n] + xd + g * od
    return out


def _build_ir(
    sr: int,
    rt60: float,
    pre_delay_ms: float,
    reverb_type: ReverbType,
) -> np.ndarray:
    cache_key = (sr, round(rt60, 2), round(pre_delay_ms, 1), reverb_type)
    if cache_key in _ir_cache:
        return _ir_cache[cache_key]

    length = int((rt60 + 0.1) * sr)

    # ── Type-specific parameters ──────────────────────────────────────────
    if reverb_type == "plate":
        # Bright dense plate — short delays, high diffusion
        comb_delays_ms  = [25.3, 26.9, 28.7, 30.5]
        comb_feedbacks  = [0.76, 0.74, 0.72, 0.71]
        allpass_delays_ms = [5.0, 1.7]
        allpass_g = 0.65
        hf_damping = 0.55        # moderate HF damping
        density_noise = 0.5

    elif reverb_type == "room":
        # Small-medium live room — slightly longer delays, coloured
        comb_delays_ms  = [30.1, 34.3, 38.9, 41.2]
        comb_feedbacks  = [0.78, 0.76, 0.74, 0.72]
        allpass_delays_ms = [6.0, 2.1]
        allpass_g = 0.70
        hf_damping = 0.40
        density_noise = 0.35

    elif reverb_type == "hall":
        # Large hall — long, smooth tail
        comb_delays_ms  = [39.2, 44.1, 49.8, 53.7]
        comb_feedbacks  = [0.82, 0.80, 0.78, 0.77]
        allpass_delays_ms = [8.3, 3.0]
        allpass_g = 0.72
        hf_damping = 0.30        # less HF damping → airier
        density_noise = 0.20

    else:  # chamber
        # Echo chamber — darker, more mid-centric
        comb_delays_ms  = [27.0, 31.5, 36.0, 40.5]
        comb_feedbacks  = [0.79, 0.77, 0.75, 0.73]
        allpass_delays_ms = [5.5, 1.9]
        allpass_g = 0.68
        hf_damping = 0.65        # darker character
        density_noise = 0.45

    # Scale feedback to match rt60 (longer rt60 → higher feedback)
    rt60_ref = 1.2
    fb_scale = float(np.clip(rt60 / rt60_ref, 0.5, 1.5))
    comb_feedbacks = [float(np.clip(g * fb_scale, 0.0, 0.97)) for g in comb_feedbacks]

    # Start with impulse + white noise for early reflections
    ir = np.zeros(length, dtype=np.float32)
    ir[0] = 1.0
    # Add low-level noise burst for diffuse early reflections
    noise = _noise_burst(length) * density_noise
    # Taper noise quickly (only first ~50 ms relevant for early reflections)
    early_len = int(0.05 * sr)
    taper = np.concatenate([np.ones(early_len), np.zeros(length - early_len)]).astype(np.float32)
    ir = ir + noise * taper

    # Comb filter bank
    comb_sum = np.zeros(length, dtype=np.float32)
    for ms, g in zip(comb_delays_ms, comb_feedbacks):
        d = max(1, int(ms * 0.001 * sr))
        comb_sum += _comb(ir, d, g)
    comb_sum /= len(comb_delays_ms)

    # HF damping on comb output (one-pole low-pass per reflection)
    from scipy.signal import lfilter as _lf
    lp_coeff = 1.0 - hf_damping
    comb_sum = _lf([lp_coeff], [1.0, -(1.0 - lp_coeff)], comb_sum).astype(np.float32)

    # Allpass cascade for diffusion
    ap = comb_sum
    for ms, in zip(allpass_delays_ms,):
        d = max(1, int(ms * 0.001 * sr))
        ap = _allpass(ap, d, allpass_g)

    # Apply decay envelope
    env = _decay_envelope(len(ap), sr, rt60)
    ap = ap * env

    # Pre-delay
    pre_samples = int(pre_delay_ms * 0.001 * sr)
    if pre_samples > 0:
        ap = np.concatenate([np.zeros(pre_samples, dtype=np.float32), ap])

    # Normalise IR peak to 1 so wet/dry mix is meaningful
    peak = np.max(np.abs(ap))
    if peak > 1e-9:
        ap = ap / peak

    _ir_cache[cache_key] = ap.astype(np.float32)
    return _ir_cache[cache_key]


# ---------------------------------------------------------------------------
# Public apply function
# ---------------------------------------------------------------------------

def apply_conv_reverb(
    x: np.ndarray,
    sr: int,
    cfg: ConvReverbSettings,
) -> np.ndarray:
    """
    Apply convolution reverb to mono audio.

    Args:
        x:   Mono audio float32 array.
        sr:  Sample rate.
        cfg: ConvReverbSettings instance.

    Returns:
        Wet+dry mixed audio, same length as input.
    """
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak_in = np.max(np.abs(x))
    if peak_in > 1.0:
        x = x / peak_in

    wet_mix = float(np.clip(cfg.wet, 0.0, 0.30))
    if wet_mix < 0.01:
        return x.astype(np.float32)

    try:
        ir = _build_ir(sr, cfg.rt60, cfg.pre_delay_ms, cfg.reverb_type)
        wet = fftconvolve(x, ir, mode="full")[: len(x)]
        wet = np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise wet level to match dry before mixing
        wet_peak = np.max(np.abs(wet))
        if wet_peak > 1e-9:
            wet = wet / wet_peak * peak_in

        result = (1.0 - wet_mix) * x + wet_mix * wet
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        r_peak = np.max(np.abs(result))
        if r_peak > 1.0:
            result = result / r_peak * 0.98

        return result.astype(np.float32)

    except Exception as e:
        logger.warning(f"Convolution reverb failed: {e}; returning dry")
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: choose reverb type from RT60 value
# ---------------------------------------------------------------------------

def reverb_type_from_rt60(rt60: float) -> ReverbType:
    """Pick the most appropriate IR type based on measured RT60."""
    if rt60 < 0.8:
        return "plate"
    elif rt60 < 1.4:
        return "room"
    elif rt60 < 2.2:
        return "hall"
    else:
        return "chamber"
