"""
Dynamics profile transfer — makes the processed vocal *ride* like the
reference rides.

Global density matching (one glue ratio) matches the p90/p10 spread but not
the shape between: a reference that hovers dense-and-loud with short dips is
different from one that swings evenly, even at the same spread. This stage
quantile-maps the short-term loudness envelope of the output onto the
reference's envelope distribution — fader automation computed from the
reference, applied smoothly, with silence left untouched.

Timeline-free: distributions are matched, not timestamps, so the reference
and the dry vocal can be entirely different songs.
"""
from __future__ import annotations

import numpy as np


def _frame_rms_db(mono: np.ndarray, sr: int, frame_s: float = 0.05):
    frame = max(1, int(sr * frame_s))
    n = len(mono) // frame
    if n < 8:
        return None, frame
    rms = np.sqrt(np.mean(mono[: n * frame].reshape(n, frame) ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-9), frame


def loudness_quantiles_centered(mono: np.ndarray, sr: int) -> list | None:
    """The 41-point centered loudness quantile curve — a reference's 'ride
    fingerprint'. Stored in recipes so presets can replay the dynamics feel
    without the reference audio."""
    db, _ = _frame_rms_db(np.asarray(mono, dtype=np.float64), sr)
    if db is None:
        return None
    act = db > (np.percentile(db, 95) - 40.0)
    if act.sum() < 8:
        return None
    qs = np.linspace(0.0, 100.0, 41)
    q = np.percentile(db[act], qs)
    return [round(float(v - np.median(db[act])), 2) for v in q]


def match_dynamics(
    y: np.ndarray,
    sr: int,
    ref_mono: np.ndarray | None,
    strength: float = 0.7,
    max_gain_db: float = 8.0,
    ref_quantiles: list | None = None,
) -> np.ndarray:
    """
    Quantile-map y's active-frame loudness distribution onto ref_mono's.

    strength: 0..1 — how far each frame moves toward its mapped target.
    Gains are clipped to ±max_gain_db and smoothed (~300 ms) so the
    automation is inaudible as such.
    """
    if len(y) == 0 or strength <= 0:
        return y

    mono = y.mean(axis=1) if y.ndim == 2 else y
    out_db, frame = _frame_rms_db(mono.astype(np.float64), sr)
    if out_db is None:
        return y
    ref_db = None
    if ref_mono is not None:
        ref_db, _ = _frame_rms_db(np.asarray(ref_mono, dtype=np.float64), sr)
    if ref_db is None and not ref_quantiles:
        return y

    # Active = within 40 dB of each signal's own loud passages (relative
    # gate, same reasoning as the density measurement: absolute gates count
    # separation bleed as programme).
    def active_mask(db):
        return db > (np.percentile(db, 95) - 40.0)

    out_act = active_mask(out_db)
    if out_act.sum() < 8:
        return y

    # Quantile mapping: a frame at the q-th loudness percentile of the
    # output moves toward the reference's q-th percentile...
    qs = np.linspace(0.0, 100.0, 41)
    out_q = np.percentile(out_db[out_act], qs)
    if ref_db is not None:
        ref_act = active_mask(ref_db)
        if ref_act.sum() < 8:
            return y
        ref_q = np.percentile(ref_db[ref_act], qs)
        ref_centered = ref_q - np.median(ref_db[ref_act])
    else:
        ref_centered = np.asarray(ref_quantiles, dtype=np.float64)
        if len(ref_centered) != len(qs):
            return y
    # ...but only the SHAPE is transferred, not absolute level — LUFS parity
    # is the loudness stage's job. Center on the output's median.
    ref_q_centered = ref_centered + np.median(out_db[out_act])

    target_db = np.interp(out_db, out_q, ref_q_centered)
    gain_db = np.where(out_act, (target_db - out_db) * float(strength), 0.0)
    gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)

    # Smooth (~300 ms) so the automation never pumps
    k = max(1, int(0.3 / 0.05))
    kernel = np.ones(k) / k
    gain_db = np.convolve(gain_db, kernel, mode="same")

    block_pos = (np.arange(len(gain_db)) + 0.5) * frame
    gain = 10 ** (np.interp(np.arange(len(mono)), block_pos, gain_db) / 20.0)
    out = np.asarray(y, dtype=np.float64) * (gain[:, None] if y.ndim == 2 else gain)
    return out.astype(np.float32, copy=False)


def dynamics_profile_gap_db(a_mono: np.ndarray, b_mono: np.ndarray, sr: int) -> float:
    """Mean |gap| between two signals' centered loudness quantile curves —
    0 means they ride identically. Used by the match report."""
    a_db, _ = _frame_rms_db(np.asarray(a_mono, dtype=np.float64), sr)
    b_db, _ = _frame_rms_db(np.asarray(b_mono, dtype=np.float64), sr)
    if a_db is None or b_db is None:
        return 0.0
    qs = np.linspace(5.0, 95.0, 19)

    def centered(db):
        act = db > (np.percentile(db, 95) - 40.0)
        if act.sum() < 8:
            return None
        q = np.percentile(db[act], qs)
        return q - np.median(db[act])

    ca, cb = centered(a_db), centered(b_db)
    if ca is None or cb is None:
        return 0.0
    return float(np.mean(np.abs(ca - cb)))
