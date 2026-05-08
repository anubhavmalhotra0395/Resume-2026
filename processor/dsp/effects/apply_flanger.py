import numpy as np
from dataclasses import dataclass
from math import sin, pi
from processor.dsp.analysis.flanger_analysis import FlangerProfile


def _process_channel(y: np.ndarray, sr: int, profile: FlangerProfile, phase_offset: float = 0.0) -> np.ndarray:
    """
    Apply flanger to a single channel using a circular buffer with modulated fractional delay.
    """
    n = len(y)
    out = np.zeros_like(y, dtype=np.float32)

    base_delay_samp = profile.base_delay_ms * sr / 1000.0
    depth_samp = profile.depth_ms * sr / 1000.0
    max_delay = int(np.ceil(base_delay_samp + depth_samp)) + 2
    if max_delay < 4:
        max_delay = 4

    buffer = np.zeros(max_delay, dtype=np.float32)
    feedback = float(np.clip(profile.feedback, 0.0, 0.95))
    mix = float(np.clip(profile.mix, 0.0, 1.0))
    rate = float(np.clip(profile.rate_hz, 0.05, 5.0))

    omega = 2 * pi * rate

    for i in range(n):
        x_n = y[i]

        # LFO
        lfo = sin(omega * (i / sr) + phase_offset)
        cur_delay = base_delay_samp + depth_samp * lfo
        # Clamp delay
        if cur_delay < 0.1:
            cur_delay = 0.1
        if cur_delay > max_delay - 2:
            cur_delay = max_delay - 2

        # Fractional read position
        read_pos = (i % max_delay) - cur_delay
        if read_pos < 0:
            read_pos += max_delay

        idx0 = int(np.floor(read_pos)) % max_delay
        idx1 = (idx0 + 1) % max_delay
        frac = read_pos - np.floor(read_pos)
        delayed = (1 - frac) * buffer[idx0] + frac * buffer[idx1]

        # Write with feedback
        buffer[i % max_delay] = x_n + delayed * feedback

        wet = delayed
        out[i] = (1 - mix) * x_n + mix * wet

    # Safety
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(np.abs(out)) + 1e-9
    if max_val > 1.0:
        out = out / max_val * 0.95
    return out.astype(np.float32)


def apply_flanger(y: np.ndarray, sr: int, profile: FlangerProfile) -> np.ndarray:
    """
    Apply flanger effect to mono or stereo signal.
    Stereo is processed with L/R phase offset for width.
    """
    if y.ndim == 1:
        return _process_channel(y, sr, profile, phase_offset=0.0)
    elif y.ndim == 2:
        # Process each channel with small phase offset for stereo spread
        left = _process_channel(y[0], sr, profile, phase_offset=0.0)
        right = _process_channel(y[1], sr, profile, phase_offset=pi / 4)
        return np.vstack([left, right])
    else:
        # Unsupported shape; return input
        return y

