"""
Stereo image synthesis and matching.

The vocal chain processes mono (one signal, one set of filters). Stereo used
to appear only when panned vocal layers were added — so a lead-only render
came out mono, and layered renders got their "width" from amplitude-panned
copies of the *same* signal (L/R correlation 1.0), which headphones hear as
off-centre rather than wide.

This module builds a genuine stereo image from a mono vocal:

    D = allpass_decorrelate(mono)      # same timbre, scrambled phase
    L = mono + a·D
    R = mono - a·D

Properties that make this the right construction:
  * side content is a·D, mid is exactly `mono` → the width knob `a` maps
    directly onto a measurable side/mid ratio, so it can be *matched* to a
    reference instead of guessed;
  * L+R = 2·mono, so a mono fold-down is bit-for-bit the original — no
    phase cancellation on club systems or phone speakers;
  * `a` is applied per frequency band, because real records are narrow in
    the lows and wide up top, and that ratio is measurable per band.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt

# Band edges for width measurement/matching (Hz)
BANDS = [(0, 250, "low"), (250, 4000, "mid"), (4000, 20000, "high")]

# Allpass diffusor: prime-ish delays in ms, alternating sign, moderate g.
# Short enough to read as width rather than echo on a vocal.
_AP = ((6.7, 0.62), (11.3, -0.58), (17.9, 0.55), (26.3, -0.5))


def _allpass(x: np.ndarray, sr: int, delay_ms: float, g: float) -> np.ndarray:
    """y[n] = -g·x[n] + x[n-d] + g·y[n-d]  →  H(z) = (-g + z^-d)/(1 - g·z^-d)"""
    d = max(1, int(sr * delay_ms / 1000.0))
    b = np.zeros(d + 1); b[0], b[d] = -g, 1.0
    a = np.zeros(d + 1); a[0], a[d] = 1.0, -g
    return lfilter(b, a, x)


def decorrelate(mono: np.ndarray, sr: int) -> np.ndarray:
    """Phase-scrambled twin of `mono`: same spectrum, near-zero correlation."""
    d = np.asarray(mono, dtype=np.float64)
    for delay_ms, g in _AP:
        d = _allpass(d, sr, delay_ms, g)
    # allpass chains preserve energy, but normalise anyway so `a` means what
    # it says regardless of material
    rms_in = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(d ** 2)))
    if rms_out > 1e-9 and rms_in > 1e-9:
        d *= rms_in / rms_out
    return d


def _band_split(x: np.ndarray, sr: int):
    """Split into (low, mid, high) with complementary Butterworth filters."""
    nyq = sr / 2.0
    # 8th-order: with 4th-order skirts, mid-band side energy leaked into the
    # low-band measurement and made narrow lows unachievable.
    lo_sos = butter(8, min(250.0 / nyq, 0.99), btype="low", output="sos")
    hi_sos = butter(8, min(4000.0 / nyq, 0.99), btype="high", output="sos")
    # Zero-phase: causal filters leave phase-driven leakage in the residual
    # (mid band carried ~-9 dB of sub-250 Hz energy purely from phase lag)
    low = sosfiltfilt(lo_sos, x)
    high = sosfiltfilt(hi_sos, x)
    mid = np.asarray(x, dtype=np.float64) - low - high
    return low, mid, high


def measure_stereo_profile(stereo: np.ndarray, sr: int) -> dict:
    """
    Per-band side/mid ratio (dB) plus overall L/R correlation of a stereo
    signal. Accepts (2, N) or (N, 2). Returns {} for mono input.
    """
    y = np.asarray(stereo, dtype=np.float64)
    if y.ndim != 2:
        return {}
    if y.shape[0] == 2:
        L, R = y[0], y[1]
    elif y.shape[1] == 2:
        L, R = y[:, 0], y[:, 1]
    else:
        return {}
    if len(L) < sr // 2:
        return {}

    mid = (L + R) / 2.0
    side = (L - R) / 2.0
    prof: dict = {}
    m_bands = _band_split(mid, sr)
    s_bands = _band_split(side, sr)
    for (m, s, (_, _, name)) in zip(m_bands, s_bands, BANDS):
        m_rms = float(np.sqrt(np.mean(m ** 2)))
        s_rms = float(np.sqrt(np.mean(s ** 2)))
        if m_rms < 1e-9:
            prof[name] = -60.0
        else:
            prof[name] = float(np.clip(20 * np.log10((s_rms + 1e-12) / m_rms), -60.0, 6.0))
    try:
        prof["correlation"] = float(np.corrcoef(L, R)[0, 1])
    except Exception:
        prof["correlation"] = 1.0
    return prof


def apply_stereo_image(mono: np.ndarray, sr: int, target: dict | None = None,
                       strength: float = 1.0) -> np.ndarray:
    """
    Turn a mono vocal into true stereo, matching `target` (as produced by
    measure_stereo_profile). Without a target, applies a musical default
    (narrow lows, moderate mids, wide highs).

    Returns (N, 2) channels-last, mono-compatible: L+R == 2·mono.
    """
    y_in = np.asarray(mono, dtype=np.float64)
    side_exist = None
    if y_in.ndim == 2:
        # Already stereo (panned layers): keep that image, and add
        # decorrelated width on top until the target is met. Mid stays the
        # mono sum, so fold-down remains clean.
        L0 = y_in[:, 0] if y_in.shape[1] == 2 else y_in[0]
        R0 = y_in[:, 1] if y_in.shape[1] == 2 else y_in[1]
        x = (L0 + R0) / 2.0
        side_exist = (L0 - R0) / 2.0
    else:
        x = y_in
    if len(x) < sr // 4:
        return np.stack([x, x], axis=1).astype(np.float32)

    defaults = {"low": -20.0, "mid": -12.0, "high": -7.0}
    tgt = {**defaults, **{k: v for k, v in (target or {}).items() if k in defaults}}

    d = decorrelate(x, sr)
    # Synthesis bands carry a guard gap (250-400 Hz unused, 4-5k eased):
    # measurement-filter skirts otherwise pick up neighbouring-band side
    # energy that no amount of gain adjustment or trimming can remove.
    nyq = sr / 2.0
    _lo = butter(8, min(250.0 / nyq, 0.99), btype="low", output="sos")
    _mid_band = butter(6, [min(400.0 / nyq, 0.98), min(4000.0 / nyq, 0.985)], btype="band", output="sos")
    _hi = butter(8, min(5000.0 / nyq, 0.99), btype="high", output="sos")
    d_bands = (sosfiltfilt(_lo, d), sosfiltfilt(_mid_band, d), sosfiltfilt(_hi, d))

    # Per-band width gains, calibrated in a closed loop: the analysis and
    # synthesis band filters overlap, so an open-loop a = 10^(target/20)
    # lands several dB off (measured: -20 dB request → -9 dB result). Two
    # correction passes put the *measured* ratio on target instead.
    caps = {"low": 0.35, "mid": 0.85, "high": 0.95}
    gains = {name: float(np.clip(10 ** (float(tgt[name]) / 20.0) * float(strength), 0.0, caps[name]))
             for (_, _, name) in BANDS}

    def build(g: dict) -> np.ndarray:
        side = np.zeros_like(x) if side_exist is None else side_exist.copy()
        for band, (_, _, name) in zip(d_bands, BANDS):
            side += g[name] * band
        return side

    side = build(gains)
    for _ in range(2):
        prof = measure_stereo_profile(np.stack([x + side, x - side], axis=1), sr)
        if not prof:
            break
        adjusted = False
        for (_, _, name) in BANDS:
            err = float(tgt[name]) - float(prof.get(name, tgt[name]))
            if abs(err) > 0.7 and gains[name] > 1e-6:
                gains[name] = float(np.clip(gains[name] * 10 ** (err / 20.0), 0.0, caps[name]))
                adjusted = True
        if not adjusted:
            break
        side = build(gains)

    # Subtractive trim: gains can only add width per decorrelator band, but
    # filter skirts leak (e.g. mid-band side energy reads as "low" width and
    # overshoots a narrow-lows target). Carve the composed side per measured
    # band until each lands on target.
    for _ in range(3):
        prof = measure_stereo_profile(np.stack([x + side, x - side], axis=1), sr)
        if not prof:
            break
        trimmed = False
        side_bands = _band_split(side, sr)
        for band, (_, _, name) in zip(side_bands, BANDS):
            excess = float(prof.get(name, -60.0)) - float(tgt[name])
            if excess > 1.0:
                k = 1.0 - 10 ** (-excess / 20.0)
                side = side - k * band
                trimmed = True
        if not trimmed:
            break

    L = x + side
    R = x - side
    peak = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))))
    if peak > 0.99:
        k = 0.99 / peak
        L, R = L * k, R * k
    out = np.stack([L, R], axis=1).astype(np.float32)
    out.setflags(write=True)
    return out
