import numpy as np


def apply_width(
    y: np.ndarray,
    sr: int,
    delay_ms: float = 12.0,
    detune_cents: float = 4.0,
    mix: float = 0.35,
) -> np.ndarray:
    """
    Simple mono-compatible ADT/double-tracking style widener.
    - Adds a delayed, lightly detuned copy and mixes back.
    - Keeps output mono to avoid downstream surprises.
    """
    delay_samples = int(sr * (delay_ms / 1000.0))
    if delay_samples <= 0 or mix <= 0:
        return y

    # Detune factor
    detune_ratio = 2 ** (detune_cents / 1200.0)
    x_detune = np.interp(
        np.arange(0, len(y)) * detune_ratio,
        np.arange(0, len(y)),
        y,
        left=0,
        right=0,
    )
    pad = np.zeros(delay_samples, dtype=y.dtype)
    delayed = np.concatenate([pad, x_detune])[: len(y)]

    out = (1 - mix) * y + mix * delayed
    return out.astype(np.float32)

