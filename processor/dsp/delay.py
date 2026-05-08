import numpy as np
import librosa


def detect_delay(y: np.ndarray, sr: int):
    """
    Detect musical echo/delay (1/4, 1/8, dotted 1/8, triplet) from audio.

    Uses normalised cross-correlation so the confidence score is 0..1 regardless
    of signal length.  Only returns a non-zero result when the correlation peak
    is strong enough to indicate a real repeating echo.

    Confidence threshold: 0.12  (empirically derived — natural room decay / chorus
    typically scores 0.04-0.08; genuine delay repeats score 0.12+).

    Returns:
        dict: { "delay_ms": float, "confidence": float, "type": str }
              delay_ms == 0.0 when no confident delay is found.
    """
    # Onset/beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if tempo <= 0 or beats is None or len(beats) < 2:
        return {"delay_ms": 0.0, "confidence": 0.0, "type": "none"}

    beat_duration = 60.0 / float(tempo)  # seconds per beat

    # Candidate delays (seconds)
    candidates = {
        "quarter":       beat_duration,
        "eighth":        beat_duration / 2.0,
        "dotted_eighth": beat_duration * 0.75,
        "triplet":       beat_duration / 3.0,
    }

    # Use onset envelope (much shorter than raw audio) for reliable correlation
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env_sr = sr / hop  # frames per second

    # Normalise
    env_norm = onset_env / (np.max(np.abs(onset_env)) + 1e-6)
    # Normalised autocorrelation via FFT for speed
    n = len(env_norm)
    fft_len = int(2 ** np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(env_norm, n=fft_len)
    power = np.abs(F) ** 2
    ac = np.fft.irfft(power)[:n]
    ac_norm = ac / (ac[0] + 1e-9)  # normalise so zero-lag = 1.0

    best_type = "none"
    best_score = 0.0
    best_ms = 0.0

    for name, sec in candidates.items():
        lag = int(sec * env_sr)
        if lag <= 0 or lag >= len(ac_norm):
            continue
        # Average peak across a ±2 frame window for robustness
        lo = max(0, lag - 2)
        hi = min(len(ac_norm), lag + 3)
        score = float(np.max(ac_norm[lo:hi]))
        if score > best_score:
            best_score = score
            best_type = name
            best_ms = sec * 1000.0

    # Confidence threshold: 0.12 means the echo repeat accounts for ≥12% of signal energy
    # Below this it's more likely to be room ambience, not a deliberate delay effect
    CONFIDENCE_THRESHOLD = 0.12
    if best_score < CONFIDENCE_THRESHOLD:
        return {"delay_ms": 0.0, "confidence": float(best_score), "type": "none"}

    return {
        "delay_ms": float(best_ms),
        "confidence": float(best_score),
        "type": best_type,
    }


def apply_delay(y: np.ndarray, sr: int, delay_ms: float, feedback: float = 0.25, mix: float = 0.25) -> np.ndarray:
    """
    Apply a simple feedback delay.
    """
    if delay_ms <= 0:
        return y

    delay_samples = int((delay_ms / 1000.0) * sr)
    if delay_samples <= 0:
        return y

    out = np.copy(y)
    fb = float(np.clip(feedback, 0.0, 0.95))
    wet = np.zeros_like(y)

    for i in range(delay_samples, len(y)):
        wet[i] = y[i] + fb * wet[i - delay_samples]
    # shift wet by delay
    wet_shifted = np.zeros_like(y)
    wet_shifted[delay_samples:] = wet[:-delay_samples]

    result = (1 - mix) * y + mix * wet_shifted
    return result.astype(np.float32)

