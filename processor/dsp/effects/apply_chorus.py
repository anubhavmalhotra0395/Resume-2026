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
    np.interp over the signal - same math as the old per-sample loop,
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

    # Two delay lines in ANTI-PHASE, one per channel.
    #
    # These used to be summed into a single channel and mixed with the dry
    # signal in mono. Summing a signal with delayed copies of itself is comb
    # filtering, and because the delay is swept the notches sweep with it --
    # which is heard as a robotic, warbling vocal, by construction. It also
    # produced no stereo modulation at all, so the "stereo-style chorus" in
    # this docstring never existed and detect_chorus could not have seen its
    # own applier's output.
    #
    # Keeping one line per channel puts the modulation in the STEREO FIELD
    # instead: the mono sum stays close to dry, so the comb filtering that
    # caused the artefact does not happen on fold-down either.
    wet1 = _modulated_delay(signal, sr, params, phase_offset=0.0)
    wet2 = _modulated_delay(signal, sr, params, phase_offset=pi)

    # The wet pair goes in ANTI-PHASE so it cancels in the mono sum.
    #
    # Adding wet to both channels leaves it in the mid, which still comb
    # filters on fold-down: measured 22.5 dB of spectral ripple against the
    # dry signal even with the two lines split across channels. Sending
    # +wet left and -wet right puts the whole effect in the side, so the mono
    # sum stays a clean (scaled) copy of the input and the swept notches
    # never appear.
    wet = 0.5 * (wet1 - wet2)
    if signal.ndim == 1:
        dry = (1.0 - params.mix) * signal
        out = np.stack([dry + params.mix * wet, dry - params.mix * wet], axis=0)
    else:
        ch = 0 if signal.shape[0] <= 2 else 1
        a = signal[0] if ch == 0 else signal[:, 0]
        b = signal[1] if ch == 0 else signal[:, 1]
        mid = 0.5 * (a + b)
        w = 0.5 * (_modulated_delay(mid, sr, params, phase_offset=0.0)
                   - _modulated_delay(mid, sr, params, phase_offset=pi))
        out = np.stack([a - params.mix * (a - mid) + params.mix * w,
                        b - params.mix * (b - mid) - params.mix * w], axis=ch)

    # Safety
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(np.abs(out)) + 1e-9
    if max_val > 1.0:
        out = out / max_val * 0.95
    return out.astype(np.float32)

