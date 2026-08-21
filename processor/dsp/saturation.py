import numpy as np


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """tanh saturation, loudness-preserving: adds harmonics without changing
    the signal's RMS, so saturation never doubles as a (mis)gain stage."""
    drive = float(np.clip(drive, 0.5, 2.0))
    x_safe = np.clip(x, -2.0, 2.0)
    y = np.tanh(x_safe * drive)

    in_rms = float(np.sqrt(np.mean(x_safe ** 2)))
    out_rms = float(np.sqrt(np.mean(y ** 2)))
    if in_rms > 1e-9 and out_rms > 1e-9:
        y = y * (in_rms / out_rms)

    max_y = np.max(np.abs(y))
    if max_y > 1.0:
        y = y / max_y * 0.98
    return y.astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    current = np.max(np.abs(x)) + 1e-9
    return (x / current) * peak


def drive_from_harmonic_ratio(ratio: float) -> float:
    # Map harmonic ratio to drive between 1.0 and 3.0
    return float(np.clip(1.0 + (ratio - 1.0) * 1.5, 1.0, 3.0))

