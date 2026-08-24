"""Stereo image matching + per-note autotune — the 2026-08-24 quality pass."""
import numpy as np
import pytest

from processor.dsp.stereo import apply_stereo_image, measure_stereo_profile
from processor.dsp.autotune import apply_autotune, estimate_f0


def test_stereo_width_matches_target_and_folds_to_mono():
    sr = 44100
    rng = np.random.default_rng(3)
    # vocal-ish: filtered noise burst train (broadband but not white)
    t = np.arange(sr * 6) / sr
    mono = (np.sin(2 * np.pi * 220 * t) * 0.2
            + rng.standard_normal(len(t)) * 0.05).astype(np.float32)

    target = {"low": -18.0, "mid": -9.0, "high": -5.0}
    out = apply_stereo_image(mono, sr, target=target)
    assert out.shape == (len(mono), 2)

    prof = measure_stereo_profile(out, sr)
    for band in ("mid", "high"):
        assert abs(prof[band] - target[band]) < 3.0, f"{band}: {prof[band]}"
    # lows must at least be far narrower than mids (guard band working)
    assert prof["low"] < prof["mid"] - 3.0

    fold = out.mean(axis=1)
    assert np.max(np.abs(fold - mono)) < 1e-4, "mono fold-down must be exact"


def test_autotune_corrects_opposite_directions():
    """The old implementation applied ONE global median shift — it could
    never fix a sharp note and a flat note in the same take."""
    sr = 22050

    def tone(f, dur):
        t = np.arange(int(sr * dur)) / sr
        return (np.sin(2 * np.pi * f * t) * np.hanning(len(t)) ** 0.2).astype(np.float32)

    sharp_a = 220 * 2 ** (0.40 / 12)   # 40 cents sharp of A
    flat_e = 330 * 2 ** (-0.45 / 12)   # 45 cents flat of E
    y = np.concatenate([tone(sharp_a, 1.2), tone(flat_e, 1.2)])

    out = apply_autotune(y, sr, retune_ms=10.0, strength=1.0,
                         scale_root=9, scale_mode="minor")  # A minor

    def med_cents(sig, lo, hi, ref):
        f0 = estimate_f0(sig[int(lo * sr):int(hi * sr)], sr)
        f0 = f0[np.isfinite(f0)]
        return 1200 * np.log2(np.median(f0) / ref)

    assert abs(med_cents(out, 0.2, 1.0, 220)) < 15, "sharp note not pulled down"
    assert abs(med_cents(out, 1.4, 2.2, 330)) < 15, "flat note not pulled up"
