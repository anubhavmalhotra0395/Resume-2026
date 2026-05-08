from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve


@dataclass
class ReverbSettings:
    decay_s: float
    mix: float
    pre_delay_ms: float = 0.0


def _comb_filter(input_signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    output = np.zeros_like(input_signal)
    for n in range(len(input_signal)):
        delayed = output[n - delay_samples] if n - delay_samples >= 0 else 0.0
        output[n] = input_signal[n] + feedback * delayed
    return output


def _allpass_filter(input_signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    output = np.zeros_like(input_signal)
    for n in range(len(input_signal)):
        delayed = output[n - delay_samples] if n - delay_samples >= 0 else 0.0
        input_delayed = input_signal[n - delay_samples] if n - delay_samples >= 0 else 0.0
        output[n] = -feedback * input_signal[n] + input_delayed + feedback * delayed
    return output


def build_impulse_response(sr: int, decay_s: float, pre_delay_ms: float) -> np.ndarray:
    # Schroeder reverb approximation: sum of comb + allpass
    length = int((decay_s + pre_delay_ms * 0.001) * sr)
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    # Pre-delay
    pre_samples = int(pre_delay_ms * 0.001 * sr)
    if pre_samples > 0:
        impulse = np.concatenate([np.zeros(pre_samples, dtype=np.float32), impulse])

    # Comb filters
    comb_delays = [int(sr * t) for t in [0.0297, 0.0371, 0.0411, 0.0437]]
    comb_feedback = 0.805  # decay factor
    comb_sum = np.zeros_like(impulse)
    for d in comb_delays:
        comb_sum += _comb_filter(impulse, max(1, d), comb_feedback)
    comb_sum /= len(comb_delays)

    # Allpass filters
    allpass_delays = [int(sr * t) for t in [0.005, 0.0017]]
    allpass_feedback = 0.7
    ap = comb_sum
    for d in allpass_delays:
        ap = _allpass_filter(ap, max(1, d), allpass_feedback)

    # Apply decay envelope
    t = np.arange(len(ap)) / sr
    env = np.exp(-3 * t / decay_s)
    ir = ap * env
    return ir.astype(np.float32)


def apply_reverb(x: np.ndarray, sr: int, cfg: ReverbSettings) -> np.ndarray:
    # Safety: ensure input is finite and reasonable
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clamp input to prevent overflow (normalize if too loud)
    max_val = np.max(np.abs(x))
    if max_val > 1.0:
        x = x / max_val * 0.95
    
    # Safety: clamp mix to reasonable range (never exceed 30% for vocals)
    mix = float(np.clip(cfg.mix, 0.0, 0.30))
    
    # Safety: clamp decay to prevent excessive tail
    decay = float(np.clip(cfg.decay_s, 0.2, 1.5))
    
    try:
        ir = build_impulse_response(sr, decay, cfg.pre_delay_ms)
        wet = fftconvolve(x, ir, mode="full")[: len(x)]
        
        # Ensure wet signal is valid and prevent overflow
        wet = np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp wet signal to prevent overflow
        max_wet = np.max(np.abs(wet))
        if max_wet > 1.0:
            wet = wet / max_wet * 0.95
        
        result = (1 - mix) * x + mix * wet
        
        # Safety check: ensure we didn't lose too much signal (use safer calculation)
        try:
            # Use a more stable calculation to avoid overflow
            dry_level = np.sqrt(np.mean(np.clip(np.square(x), 0, 1.0)))
            result_level = np.sqrt(np.mean(np.clip(np.square(result), 0, 1.0)))
            
            if result_level < dry_level * 0.3:  # If result is 70% quieter, something's wrong
                # Fallback: use less reverb
                mix = min(mix, 0.15)
                result = (1 - mix) * x + mix * wet
        except (OverflowError, RuntimeWarning):
            # If calculation fails, use conservative mix
            mix = min(mix, 0.15)
            result = (1 - mix) * x + mix * wet
        
        # Final safety: ensure result is valid
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        max_result = np.max(np.abs(result))
        if max_result > 1.0:
            result = result / max_result * 0.95
        
        return result.astype(np.float32)
    except Exception as e:
        # If reverb fails completely, return dry signal
        import logging
        logging.warning(f"Reverb processing failed: {e}. Returning dry signal.")
        return x.astype(np.float32)


