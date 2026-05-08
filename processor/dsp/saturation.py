import numpy as np


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    # Clamp drive to reasonable range
    drive = float(np.clip(drive, 0.5, 3.0))
    
    # Clamp input to prevent overflow
    x_safe = np.clip(x, -2.0, 2.0)
    
    y = np.tanh(x_safe * drive)
    
    # Normalize output to prevent excessive gain reduction
    # tanh with high drive can reduce overall level significantly
    if drive > 2.0:
        # Compensate for level loss from high drive
        compensation = 1.0 + (drive - 2.0) * 0.2
        y = y * compensation
    
    # Ensure output is reasonable
    max_y = np.max(np.abs(y))
    if max_y > 1.0:
        y = y / max_y * 0.95
    
    return y.astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    current = np.max(np.abs(x)) + 1e-9
    return (x / current) * peak


def drive_from_harmonic_ratio(ratio: float) -> float:
    # Map harmonic ratio to drive between 1.0 and 3.0
    return float(np.clip(1.0 + (ratio - 1.0) * 1.5, 1.0, 3.0))

