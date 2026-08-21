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
    Single modulated delay line (feedforward tap, linear interpolation).
    Vectorised: the tap position is n - delay(n), so the whole line is one
    np.interp over the signal — same math as the old per-sample loop,
    orders of magnitude faster.
    """
    n_samples = len(signal)
    if n_samples == 0:
        return signal.astype(np.float32)

    base_delay = params.base_delay_ms / 1000.0 * sr
    mod_depth = params.mod_depth_ms / 1000.0 * sr * params.depth

    n = np.arange(n_samples)
    delay = base_delay + mod_depth * np.sin(2 * pi * params.rate_hz * (n / sr) + phase_offset)
    delay = np.clip(delay, 1.0, None)
    read_pos = n - delay
    return np.interp(read_pos, n, signal, left=0.0, right=0.0).astype(np.float32)


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

