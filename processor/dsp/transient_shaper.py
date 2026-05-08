"""
Transient shaper for accentuation/softening of transients.
"""
import numpy as np
import librosa


def transient_shaper(x: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """
    Shape transients in audio.
    
    Args:
        x: Input audio
        sr: Sample rate
        amount: Amount of shaping (-1 to 1, positive = accentuate, negative = soften)
    
    Returns:
        Shaped audio
    """
    # Replace non-finite values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Split into transient vs sustain via onset strength
    env = librosa.onset.onset_strength(y=x, sr=sr)
    
    # Upsample env to len(x)
    env_up = np.interp(np.linspace(0, len(env), len(x)), np.arange(len(env)), env)
    
    # Identify transient regions (top 25% of onset strength)
    threshold = np.percentile(env_up, 75)
    mask = env_up > threshold
    
    out = x.copy()
    
    # Accentuate transients, soften sustain
    if amount > 0:
        out[mask] *= (1.0 + amount)
        out[~mask] *= (1.0 - amount * 0.2)
    else:
        # Soften transients
        out[mask] *= (1.0 + amount)  # amount is negative
        out[~mask] *= (1.0 - amount * 0.1)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(out)) + 1e-9
    out = out / max_val * 0.95
    
    return out

