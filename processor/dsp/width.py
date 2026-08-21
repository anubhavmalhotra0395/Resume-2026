import numpy as np


def apply_width(
    y: np.ndarray,
    sr: int,
    delay_ms: float = 12.0,
    detune_cents: float = 4.0,
    mix: float = 0.35,
) -> np.ndarray:
    """
    Mono-compatible ADT/double-tracking widener.

    The detuned copy comes from a slowly *modulated* delay line (classic tape
    ADT), so the perceived detune is constant but the copy stays anchored to
    the lead. The previous implementation resampled the whole copy at the
    detune ratio — a 4-cent detune is a 0.23% speed difference, so the copy
    drifted ~140 ms per minute and turned into an audible second vocal by the
    end of the song ("sounds doubled after a while").
    """
    if mix <= 0 or len(y) == 0:
        return y

    n = np.arange(len(y))
    base_s = max(delay_ms, 1.0) / 1000.0

    # Depth chosen so the delay modulation's slope equals the requested
    # detune at its steepest point: pitch ratio deviation = d(delay)/dt.
    rate_hz = 0.35  # slow wander — heard as detune, not vibrato
    ratio_dev = 2 ** (abs(detune_cents) / 1200.0) - 1.0
    depth_s = ratio_dev / (2 * np.pi * rate_hz)

    # 1 - cos keeps the delay >= base at all times (never reads the future)
    delay_samples = (base_s + depth_s * (1.0 - np.cos(2 * np.pi * rate_hz * n / sr))) * sr
    pos = n - delay_samples
    copy = np.interp(pos, n, y, left=0.0, right=0.0)

    out = (1 - mix) * y + mix * copy
    return out.astype(np.float32)
