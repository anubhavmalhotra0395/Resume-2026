import numpy as np
from dataclasses import dataclass
from math import sin, pi


@dataclass
class ChorusParams:
    rate_hz: float = 0.8
    depth: float = 0.3   # 0..1
    mix: float = 0.3     # 0..1
    base_delay_ms: float = 12.0  # base delay
    mod_depth_ms: float = 8.0    # modulation amplitude


def _modulated_delay(signal: np.ndarray, sr: int, params: ChorusParams, phase_offset: float = 0.0) -> np.ndarray:
    """
    Apply a single modulated delay line with LFO.
    Uses linear interpolation for fractional delay samples.
    """
    n_samples = len(signal)
    output = np.zeros_like(signal, dtype=np.float32)

    # Convert ms to samples
    base_delay = params.base_delay_ms / 1000.0 * sr
    mod_depth = params.mod_depth_ms / 1000.0 * sr * params.depth  # scale by depth

    # Ensure buffer is large enough
    max_delay = int(np.ceil(base_delay + mod_depth)) + 2
    buffer = np.zeros(max_delay, dtype=np.float32)

    # LFO angular frequency
    omega = 2 * pi * params.rate_hz

    for n in range(n_samples):
        # Current input sample
        x_n = signal[n]

        # LFO value
        lfo = sin(omega * (n / sr) + phase_offset)

        # Current delay in samples
        cur_delay = base_delay + mod_depth * lfo
        if cur_delay < 1.0:
            cur_delay = 1.0
        if cur_delay > max_delay - 2:
            cur_delay = max_delay - 2

        # Read position
        read_pos = n % max_delay - cur_delay
        if read_pos < 0:
            read_pos += max_delay

        # Fractional delay: linear interpolation
        idx0 = int(np.floor(read_pos)) % max_delay
        idx1 = (idx0 + 1) % max_delay
        frac = read_pos - np.floor(read_pos)
        delayed = (1 - frac) * buffer[idx0] + frac * buffer[idx1]

        # Write current sample into buffer
        buffer[n % max_delay] = x_n

        output[n] = delayed

    return output


def apply_chorus(signal: np.ndarray, sr: int, rate_hz: float, depth: float, mix: float) -> np.ndarray:
    """
    Apply stereo-style chorus to a mono signal using two modulated delay lines.

    Args:
        signal: Input mono signal
        sr: Sample rate
        rate_hz: LFO rate in Hz
        depth: 0..1 modulation depth
        mix: 0..1 wet/dry mix
    """
    params = ChorusParams(rate_hz=rate_hz, depth=np.clip(depth, 0.0, 1.0), mix=np.clip(mix, 0.0, 1.0))

    # Two delay lines with phase offset
    wet1 = _modulated_delay(signal, sr, params, phase_offset=0.0)
    wet2 = _modulated_delay(signal, sr, params, phase_offset=pi / 2)

    wet = 0.5 * (wet1 + wet2)

    # Mix
    out = (1 - params.mix) * signal + params.mix * wet

    # Safety
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(np.abs(out)) + 1e-9
    if max_val > 1.0:
        out = out / max_val * 0.95
    return out.astype(np.float32)

