import numpy as np
import librosa


def _onset_ac(y: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    """Normalised autocorrelation of the onset-strength envelope."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env_norm = onset_env / (np.max(np.abs(onset_env)) + 1e-6)
    n = len(env_norm)
    fft_len = int(2 ** np.ceil(np.log2(max(2 * n, 2))))
    F = np.fft.rfft(env_norm, n=fft_len)
    ac = np.fft.irfft(np.abs(F) ** 2)[:n]
    return ac / (ac[0] + 1e-9)


def detect_delay(y: np.ndarray, sr: int, dry: np.ndarray | None = None):
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
    env_sr = sr / hop  # frames per second
    ac_norm = _onset_ac(y, sr, hop)

    # Rhythm cancellation: a rhythmic vocal autocorrelates at the beat even
    # with NO delay effect — which is what the old detector kept "finding".
    # The dry vocal shares the rhythm but not the effect, so subtracting its
    # autocorrelation leaves (approximately) just the echo.
    ac_dry = _onset_ac(dry, sr, hop) if dry is not None and len(dry) > sr else None

    # MEASURE the echo lag from the autocorrelation itself — the strongest
    # peak in the plausible vocal-delay range. (Previously the lag came from
    # beat-tracked tempo and the AC only scored it; beat-tracking an isolated
    # vocal octave-errs easily, which made the same song flip between 406 ms
    # and 604 ms depending on the analysis window.)
    from scipy.signal import find_peaks

    lag_lo = max(1, int(0.080 * env_sr))          # <80 ms is slapback/comb territory
    lag_hi = min(int(0.800 * env_sr), len(ac_norm) - 1)
    if lag_hi <= lag_lo:
        return {"delay_ms": 0.0, "confidence": 0.0, "type": "none"}

    seg = ac_norm[lag_lo:lag_hi].copy()
    if ac_dry is not None:
        m = min(len(seg), max(0, len(ac_dry) - lag_lo))
        if m > 0:
            seg[:m] = seg[:m] - ac_dry[lag_lo:lag_lo + m]
    peaks, props = find_peaks(seg, height=0.0)
    if peaks.size == 0:
        return {"delay_ms": 0.0, "confidence": 0.0, "type": "none"}
    best_idx = int(peaks[np.argmax(props["peak_heights"])]) + lag_lo
    best_score = float(seg[best_idx - lag_lo])
    best_ms = best_idx / env_sr * 1000.0

    # Confidence threshold: 0.12 means the echo repeat accounts for ≥12% of signal energy
    # Below this it's more likely to be room ambience, not a deliberate delay effect
    CONFIDENCE_THRESHOLD = 0.12
    if best_score < CONFIDENCE_THRESHOLD:
        return {"delay_ms": 0.0, "confidence": float(best_score), "type": "none"}

    # Name the measured lag musically; snap only when nearly exact (4%) so a
    # bad tempo estimate can never drag the measured echo time around
    best_type = "echo"
    for name, sec in {**candidates, "dotted_quarter": beat_duration * 1.5}.items():
        cand_ms = sec * 1000.0
        if cand_ms > 0 and abs(best_ms - cand_ms) / cand_ms < 0.04:
            best_type = name
            best_ms = cand_ms
            break

    # ── Echo character, measured from the same autocorrelation ─────────────
    # First-repeat level ≈ AC peak height at the lag (validated on synthetic
    # echoes: planted 0.45 reads ~0.44). Feedback ≈ how much of that repeat
    # survives to the double lag: AC(2L)/AC(L).
    echo_level = float(np.clip(best_score, 0.0, 0.6))
    feedback = 0.25
    lag2 = 2 * best_idx
    if lag2 < len(ac_norm):
        ac2 = float(ac_norm[lag2])
        if ac_dry is not None and lag2 < len(ac_dry):
            ac2 -= float(ac_dry[lag2])
        if best_score > 1e-6:
            feedback = float(np.clip(ac2 / best_score, 0.1, 0.6))

    return {
        "delay_ms": float(best_ms),
        "confidence": float(best_score),
        "type": best_type,
        "echo_level": echo_level,
        "feedback": feedback,
    }


def apply_delay(y: np.ndarray, sr: int, delay_ms: float, feedback: float = 0.25,
                mix: float = 0.25, wet_lowpass_hz: float = 7000.0) -> np.ndarray:
    """
    Feedback delay. `mix` sets the first repeat's level relative to the dry
    signal (so a measured echo_level maps straight onto it); `feedback` sets
    how much each repeat carries to the next; the wet path is low-passed —
    like every studio vocal delay — so repeats sit behind the voice instead
    of clashing with it.
    """
    if delay_ms <= 0:
        return y

    delay_samples = int((delay_ms / 1000.0) * sr)
    if delay_samples <= 0:
        return y

    fb = float(np.clip(feedback, 0.0, 0.95))
    # Feedback comb y[n] = x[n] + fb*y[n-d]  →  H(z) = 1 / (1 - fb z^-d),
    # then delayed by d to become the echo tail. (The old per-sample loop
    # was slower than realtime and left the first delay period silent.)
    from scipy.signal import lfilter, butter, sosfilt
    a = np.zeros(delay_samples + 1)
    a[0], a[delay_samples] = 1.0, -fb
    wet = lfilter([1.0], a, y)
    wet_shifted = np.zeros_like(y)
    wet_shifted[delay_samples:] = wet[: len(y) - delay_samples]

    nyq = sr / 2.0
    if wet_lowpass_hz and wet_lowpass_hz < nyq * 0.95:
        sos = butter(2, wet_lowpass_hz / nyq, btype="low", output="sos")
        wet_shifted = sosfilt(sos, wet_shifted)

    # Additive echo: dry stays at unity, repeats at `mix` — matching how the
    # echo level was measured (relative to the dry signal).
    result = y + float(np.clip(mix, 0.0, 0.6)) * wet_shifted
    peak = float(np.max(np.abs(result)))
    if peak > 0.99:
        result = result * (0.99 / peak)
    return result.astype(np.float32)

