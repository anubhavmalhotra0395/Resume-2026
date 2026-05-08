"""
Apply a generated algorithmic/convolutional reverb to a signal.

We construct a synthetic impulse response (IR) from the ReverbProfile:
- early reflections: sparse short taps
- late reverb: filtered exponentially decaying noise (tail shaped by RT60)
Then convolve the dry vocal with the IR (fftconvolve).
"""

import numpy as np
from scipy.signal import fftconvolve, lfilter
import logging
from processor.dsp.analysis.reverb_analysis import ReverbProfile

log = logging.getLogger("apply_reverb")
log.setLevel(logging.INFO)


def _make_early_reflections(sr, predelay_ms, early_ratio):
    """
    Create a small array of early reflection taps based on early_ratio.
    early_ratio controls relative energy; returns (ir, offset_samples)
    """
    predelay_s = predelay_ms / 1000.0
    predelay_samples = int(round(predelay_s * sr))
    tap_times_ms = [5, 15, 30, 55]
    taps = np.zeros(predelay_samples + int(sr * 0.1) + 1)
    base_gain = 1.0 * early_ratio
    for t in tap_times_ms:
        idx = predelay_samples + int(round((t / 1000.0) * sr))
        if idx < taps.shape[0]:
            decay = base_gain * (0.6 ** (t / 20.0))
            taps[idx] += decay
    return taps


def _make_late_tail(sr, rt60, length_s=None):
    """
    Create late tail as exponentially decaying filtered noise.
    length_s defaults to max(2*rt60, 3s)
    """
    if length_s is None:
        length_s = max(2.0 * rt60, 3.0)
    n = int(round(length_s * sr))
    if n <= 0:
        n = int(round(1.0 * sr))
    noise = np.random.randn(n) * 0.001
    t = np.arange(n) / float(sr)
    amp = 10 ** (-60.0 * t / (20.0 * rt60 + 1e-9))
    tail = noise * amp
    b = [0.5]
    a = [1.0, -0.5]
    tail = lfilter(b, a, tail)
    return tail


def make_synthetic_ir(sr, profile: ReverbProfile):
    """
    Build IR = early taps + late tail
    """
    early = _make_early_reflections(sr, profile.predelay_ms, profile.early_ratio)
    late = _make_late_tail(sr, profile.rt60)
    ir = np.concatenate([early, late])
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12)
    return ir


def apply_reverb(y: np.ndarray, sr: int, profile: ReverbProfile) -> np.ndarray:
    """
    Apply generated reverb based on profile via convolution.
    y: mono (n,) or stereo (2,n): returns same shape as input.
    """
    if y.ndim == 2:
        mono = np.mean(y, axis=0)
        stereo_flag = True
    else:
        mono = y
        stereo_flag = False

    ir = make_synthetic_ir(sr, profile)
    wet = fftconvolve(mono, ir, mode="full")[: len(mono)]
    dry = mono
    wet_level = profile.wet
    out_mono = (1.0 - wet_level) * dry + wet_level * wet

    peak = np.max(np.abs(out_mono)) + 1e-12
    if peak > 1.0:
        out_mono = out_mono / peak

    if stereo_flag:
        wet_l = fftconvolve(y[0], ir, mode="full")[: y.shape[1]]
        wet_r = fftconvolve(y[1], ir, mode="full")[: y.shape[1]]
        out_l = (1.0 - profile.wet) * y[0] + profile.wet * wet_l
        out_r = (1.0 - profile.wet) * y[1] + profile.wet * wet_r
        out = np.vstack([out_l, out_r])
        peak = np.max(np.abs(out)) + 1e-12
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)

    return out_mono.astype(np.float32)

