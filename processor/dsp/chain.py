from typing import List, Optional

import numpy as np

from processor.dsp.compressor import CompressorSettings, apply_compressor
from processor.dsp.eq import EqBand, apply_eq
from processor.dsp.reverb import ReverbSettings, apply_reverb
from processor.dsp.conv_reverb import ConvReverbSettings, apply_conv_reverb, reverb_type_from_rt60
from processor.dsp.saturation import normalize_peak, soft_clip
from processor.dsp.width import apply_width
from processor.dsp.deesser import apply_deesser
from processor.dsp.transient_shaper import transient_shaper
from processor.dsp.multiband_compressor import multiband_compress
from processor.dsp.analysis.chorus_analysis import ChorusProfile
from processor.dsp.effects.apply_chorus import apply_chorus
from processor.dsp.analysis.flanger_analysis import FlangerProfile
from processor.dsp.effects.apply_flanger import apply_flanger
from processor.dsp.gate import GateSettings, apply_gate
from processor.dsp.parallel_comp import ParallelCompSettings, apply_parallel_comp
from processor.dsp.tape import TapeSettings, apply_tape
from processor.dsp.exciter import ExciterSettings, apply_exciter
from processor.dsp.doubler import DoublerSettings, apply_doubler
from processor.dsp.ms_eq import MsEqSettings, apply_ms_eq
from processor.dsp.autotune import apply_autotune
from processor.dsp.analysis.autotune_analysis import AutotuneSettings


def normalize_rms(x: np.ndarray, target_db: float = -18.0) -> np.ndarray:
    """Simple RMS normalization to a target dBFS."""
    # Use safe square calculation to prevent overflow
    x_safe = np.clip(x, -1.0, 1.0)
    rms = np.sqrt(np.mean(np.clip(np.square(x_safe), 0, 1.0))) + 1e-9
    target_lin = 10 ** (target_db / 20.0)
    gain = target_lin / rms
    
    # Clamp gain to prevent excessive amplification
    gain = np.clip(gain, 0.1, 100.0)
    
    result = (x * gain).astype(np.float32)
    
    # Safety: ensure result isn't too loud
    max_result = np.max(np.abs(result))
    if max_result > 1.0:
        result = result / max_result * 0.95
    
    return result


def match_loudness(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match output loudness to reference loudness."""
    # Use safe square calculation to prevent overflow
    ref_safe = np.clip(reference, -1.0, 1.0)
    x_safe = np.clip(x, -1.0, 1.0)
    
    ref_rms = np.sqrt(np.mean(np.clip(np.square(ref_safe), 0, 1.0))) + 1e-9
    out_rms = np.sqrt(np.mean(np.clip(np.square(x_safe), 0, 1.0))) + 1e-9
    
    gain = ref_rms / out_rms
    
    # Clamp gain to prevent excessive amplification or attenuation
    # Don't make audio quieter than -30 dBFS or louder than 0 dBFS
    min_gain = 0.032  # -30 dBFS minimum
    max_gain = 1.0    # 0 dBFS maximum
    gain = np.clip(gain, min_gain, max_gain)
    
    result = (x * gain).astype(np.float32)
    
    # Safety: ensure result isn't too loud
    max_result = np.max(np.abs(result))
    if max_result > 1.0:
        result = result / max_result * 0.95
    
    return result


def apply_chain(
    x: np.ndarray,
    sr: int,
    eq_bands: List[EqBand],
    comp: Optional[CompressorSettings],
    reverb: Optional[ReverbSettings],
    saturation_drive: Optional[float] = None,
    width: Optional[dict] = None,
    chorus_profile: Optional[ChorusProfile] = None,
    flanger_profile: Optional[FlangerProfile] = None,
    reference: Optional[np.ndarray] = None,
    enable_deesser: bool = True,
    enable_transient_shaper: bool = False,
    enable_multiband: bool = False,
    segments: Optional[List] = None,
    # New professional effects
    gate: Optional[GateSettings] = None,
    parallel_comp: Optional[ParallelCompSettings] = None,
    tape: Optional[TapeSettings] = None,
    exciter: Optional[ExciterSettings] = None,
    doubler: Optional[DoublerSettings] = None,
    ms_eq: Optional[MsEqSettings] = None,
    autotune: Optional[AutotuneSettings] = None,
) -> np.ndarray:
    """
    Apply full DSP chain with optional enhancements.
    
    Args:
        x: Input audio
        sr: Sample rate
        eq_bands: EQ bands to apply
        comp: Compressor settings
        reverb: Reverb settings
        saturation_drive: Saturation drive
        width: Width/ADT settings
        reference: Reference audio for loudness matching
        enable_deesser: Enable de-esser
        enable_transient_shaper: Enable transient shaper
        enable_multiband: Use multiband compressor instead of single-band
        segments: Optional list of (start, end) tuples for per-segment processing
    
    Returns:
        Processed audio
    """
    y = x

    if segments is not None and len(segments) > 0:
        pass  # future: per-segment adaptive processing

    # Ensure finite values
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 1. Noise Gate (pre-chain) ──────────────────────────────────────────
    if gate is not None:
        try:
            y_before = y.copy()
            y = apply_gate(y, sr, gate)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Gate produced silent output. Skipping gate.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Gate failed ({e}). Skipping gate.")

    # EQ
    y_before = y.copy()
    y = apply_eq(y, sr, eq_bands)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(np.abs(y)) < 1e-9:
        print("⚠ WARNING: EQ produced silent output. Skipping EQ.")
        y = y_before
    
    # Compression (multiband or single-band) — skip if not detected in reference
    if comp is not None or enable_multiband:
        y_before = y.copy()
        if enable_multiband:
            y = multiband_compress(y, sr)
        else:
            y = apply_compressor(y, sr, comp)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(y)) < 1e-9:
            print("⚠ WARNING: Compressor produced silent output. Skipping compression.")
            y = y_before

    # ── Parallel Compression (after main compressor) ───────────────────────
    if parallel_comp is not None:
        try:
            y_before = y.copy()
            y = apply_parallel_comp(y, sr, parallel_comp)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Parallel comp produced silent output. Skipping.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Parallel comp failed ({e}). Skipping.")

    # De-esser (before reverb to reduce sibilance in reverb tail)
    if enable_deesser:
        y_before = y.copy()
        y = apply_deesser(y, sr)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(y)) < 1e-9:
            print("⚠ WARNING: De-esser produced silent output. Skipping de-esser.")
            y = y_before
    
    # Reverb — convolution reverb first, algorithmic as fallback
    if reverb is not None:
        y_before = y.copy()
        try:
            # Use convolution reverb for a more realistic space
            rv_type = reverb_type_from_rt60(reverb.decay_s)
            conv_cfg = ConvReverbSettings(
                reverb_type=rv_type,
                rt60=reverb.decay_s,
                wet=reverb.mix,
                pre_delay_ms=reverb.pre_delay_ms,
            )
            # Conv reverb expects mono; handle stereo inputs
            if y.ndim == 2:
                ch0 = apply_conv_reverb(y[:, 0], sr, conv_cfg)
                ch1 = apply_conv_reverb(y[:, 1], sr, conv_cfg)
                y_wet = np.stack([ch0, ch1], axis=1)
            else:
                y_wet = apply_conv_reverb(y, sr, conv_cfg)
            y_wet = np.nan_to_num(y_wet, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y_wet)) > 1e-9:
                y = y_wet
                print(f"  Convolution reverb ({rv_type}): rt60={reverb.decay_s:.2f}s wet={reverb.mix:.2f}")
            else:
                raise ValueError("conv reverb produced silence")
        except Exception as e:
            print(f"  Conv reverb failed ({e}), falling back to algorithmic reverb")
            y = apply_reverb(y, sr, reverb)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(y)) < 1e-9:
            print("⚠ WARNING: Reverb produced silent output. Skipping reverb.")
            y = y_before
    
    # Transient shaper (optional)
    if enable_transient_shaper:
        y_before = y.copy()
        y = transient_shaper(y, sr, amount=0.3)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(y)) < 1e-9:
            print("⚠ WARNING: Transient shaper produced silent output. Skipping transient shaper.")
            y = y_before
    
    # Saturation — skip if not detected in reference
    if saturation_drive is not None:
        y_before = y.copy()
        y = soft_clip(y, drive=saturation_drive)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(y)) < 1e-9:
            print("⚠ WARNING: Saturation produced silent output. Skipping saturation.")
            y = y_before

    # ── Tape Emulation (after saturation, before spatial effects) ─────────
    if tape is not None:
        try:
            y_before = y.copy()
            y = apply_tape(y, sr, tape)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Tape emulation produced silent output. Skipping.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Tape emulation failed ({e}). Skipping.")

    # ── Exciter (after tape, adds HF harmonics) ───────────────────────────
    if exciter is not None:
        try:
            y_before = y.copy()
            y = apply_exciter(y, sr, exciter)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Exciter produced silent output. Skipping.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Exciter failed ({e}). Skipping.")

    # Width/ADT
    if width and width.get("mix", 0) > 0:
        y = apply_width(
            y,
            sr,
            delay_ms=float(width.get("delay_ms", 12.0)),
            detune_cents=float(width.get("detune_cents", 4.0)),
            mix=float(width.get("mix", 0.35)),
        )

    # ── Vocal Doubler (after width, before modulation) ────────────────────
    if doubler is not None and doubler.mix > 0:
        try:
            y_before = y.copy()
            y = apply_doubler(y, sr, doubler)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Doubler produced silent output. Skipping.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Doubler failed ({e}). Skipping.")

    # ── Mid-Side EQ (after doubler, before chorus/flanger) ───────────────
    if ms_eq is not None and (ms_eq.mid_bands or ms_eq.side_bands):
        try:
            y_before = y.copy()
            y = apply_ms_eq(y, sr, ms_eq)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: M-S EQ produced silent output. Skipping.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: M-S EQ failed ({e}). Skipping.")

    # Chorus (after width / modulation effects) — skip if mix is zero
    if chorus_profile is not None and chorus_profile.mix > 0:
        try:
            y_before = y.copy()
            y = apply_chorus(
                y,
                sr,
                rate_hz=chorus_profile.rate_hz,
                depth=chorus_profile.depth,
                mix=chorus_profile.mix,
            )
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Chorus produced silent output. Skipping chorus.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Chorus failed ({e}). Skipping chorus.")

    # Flanger (after chorus) — skip if mix is zero
    if flanger_profile is not None and flanger_profile.mix > 0:
        try:
            y_before = y.copy()
            y = apply_flanger(y, sr, flanger_profile)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Flanger produced silent output. Skipping flanger.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Flanger failed ({e}). Skipping flanger.")
    
    # ── Autotune — key detected from the dry input (x), applied to processed ──
    # We use x (original dry) for key detection so the key is accurate,
    # then apply correction to the processed signal y.
    if autotune is not None:
        try:
            from processor.dsp.autotune import _detect_key
            y_before = y.copy()
            mono_dry = x if x.ndim == 1 else np.mean(x, axis=0)
            scale_root, scale_mode = _detect_key(mono_dry, sr)
            print(f"  Autotune: key={['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][scale_root]} {scale_mode} "
                  f"strength={autotune.strength:.2f} retune={autotune.retune_ms:.0f}ms")

            mono_y = y if y.ndim == 1 else np.mean(y, axis=0)
            tuned = apply_autotune(
                mono_y, sr,
                retune_ms=autotune.retune_ms,
                strength=autotune.strength,
                scale_root=scale_root,
                scale_mode=scale_mode,
            )
            if y.ndim == 2:
                y = np.stack([tuned, tuned], axis=0).astype(np.float32)
            else:
                y = tuned
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(np.abs(y)) < 1e-9:
                print("⚠ WARNING: Autotune produced silent output. Skipping.")
                y = y_before
        except Exception as e:
            print(f"⚠ WARNING: Autotune failed ({e}). Skipping.")

    # Match loudness to reference if provided, otherwise normalize to -18 dBFS
    if reference is not None and len(reference) > 0:
        # Normalize lengths for comparison
        min_len = min(len(y), len(reference))
        ref_seg = reference[:min_len]
        y_seg = y[:min_len]
        
        # Safety check: ensure reference has audio
        ref_rms = np.sqrt(np.mean(np.clip(np.square(ref_seg), 0, 1.0)))
        y_rms = np.sqrt(np.mean(np.clip(np.square(y_seg), 0, 1.0)))
        
        # Minimum RMS threshold to prevent making audio too quiet
        min_rms = 0.01  # -40 dBFS minimum
        
        if ref_rms > min_rms and y_rms > min_rms:
            y_matched = match_loudness(y_seg, ref_seg)
            # Safety: ensure matched audio isn't too quiet
            matched_rms = np.sqrt(np.mean(np.clip(np.square(y_matched), 0, 1.0)))
            
            # If reference is very quiet, don't match to it - use reasonable target instead
            if ref_rms < 0.05:  # Reference is very quiet (< -26 dBFS)
                # Use -18 dBFS target instead of matching quiet reference
                y = normalize_rms(y_seg, target_db=-18.0)
            elif matched_rms > min_rms:
                y = y_matched
            else:
                # Matched audio is too quiet, use RMS normalization
                y = normalize_rms(y_seg, target_db=-18.0)
        else:
            # Reference or output is too quiet, use RMS normalization
            y = normalize_rms(y_seg, target_db=-18.0)
        
        if len(y) < len(x):
            # Pad if needed (shouldn't happen, but safety)
            y = np.pad(y, (0, len(x) - len(y)), mode='constant')
    else:
        y = normalize_rms(y, target_db=-18.0)
    
    # Final peak normalization
    y = normalize_peak(y, peak=0.99)
    
    # Final safety check: ensure we have valid audio
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    max_final = np.max(np.abs(y))
    rms_final = np.sqrt(np.mean(np.clip(np.square(y), 0, 1.0)))
    
    # Check both peak and RMS to ensure audio is audible
    if max_final < 0.01 or rms_final < 0.005:  # Very quiet threshold
        # If somehow we got zeros or very quiet audio, boost it
        print(f"⚠ WARNING: DSP chain produced very quiet output (max={max_final:.6f}, rms={rms_final:.6f}). Boosting...")
        if max_final > 1e-9:
            # Boost to reasonable level
            target_max = 0.5
            y = y / max_final * target_max
        else:
            # If completely silent, return original input
            print("⚠ WARNING: DSP chain produced silent output. Returning original input.")
            return x.astype(np.float32)
    
    # Ensure final output has reasonable level (at least -30 dBFS RMS)
    rms_final = np.sqrt(np.mean(np.clip(np.square(y), 0, 1.0)))
    if rms_final < 0.032:  # -30 dBFS
        boost = 0.032 / (rms_final + 1e-9)
        boost = np.clip(boost, 1.0, 10.0)  # Don't boost more than 10x
        y = y * boost
        # Re-normalize peak if needed
        max_y = np.max(np.abs(y))
        if max_y > 1.0:
            y = y / max_y * 0.95
    
    return y.astype(np.float32)


