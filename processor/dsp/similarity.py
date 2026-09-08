"""
Similarity measurement suite - everything we know how to compare between two
vocals, in one place.

The matching stages could only ever be as complete as the set of properties we
measure: anything absent from this file is, by definition, unmatched and also
invisible in any score we report. So this is deliberately broader than what the
chain currently corrects. Dimensions we can measure but not yet correct are
still worth having, because they show where the real remaining gap is.

Everything is level-invariant (shapes and ratios, mean-removed where relevant)
except `lufs` itself, since absolute level is owned by the final loudness match.
"""
from __future__ import annotations

import numpy as np
import librosa

_HOP = 512
_NFFT = 4096

# 32 log-spaced edges: finer than the 12 the correction loop uses, so the score
# can see fine structure the corrector averages away.
_EDGES32 = np.geomspace(60.0, 16000.0, 33)
_EDGES_W = np.geomspace(100.0, 12000.0, 9)     # per-band width profile
_MOD_EDGES = np.geomspace(0.5, 32.0, 9)        # envelope modulation, Hz


def _mono(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y
    return y.mean(axis=1) if y.shape[0] > y.shape[1] else y.mean(axis=0)


def _stereo(y):
    """Return (L, R) or (mono, mono)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y, y
    a = y if y.shape[0] == 2 else y.T
    if a.shape[0] < 2:
        return a[0], a[0]
    return a[0], a[1]


def _band_shape(power, freqs, edges):
    out = np.empty(len(edges) - 1)
    for i in range(len(out)):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = 10 * np.log10(float(np.sum(power[m])) + 1e-20) if m.any() else -200.0
    ok = out > -190.0
    if ok.any():
        out = out - float(np.mean(out[ok]))
    return out


def _frame_db(mono, sr):
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=_HOP)[0]
    return 20 * np.log10(np.maximum(rms, 1e-9))


def measure(y, sr: int) -> dict:
    """Full feature set for one signal."""
    m = _mono(y)
    L, R = _stereo(y)
    n = min(len(L), len(R), len(m))
    m, L, R = m[:n], L[:n], R[:n]
    f = {}

    S = np.abs(librosa.stft(np.ascontiguousarray(m), n_fft=_NFFT, hop_length=_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    power = np.mean(S, axis=1)

    # ---- spectrum -------------------------------------------------------
    f["bands32"] = _band_shape(power, freqs, _EDGES32)
    tot = float(np.sum(power)) + 1e-20
    hi = (freqs >= 10000)
    f["air_db"] = 10 * np.log10(float(np.sum(power[hi])) / tot + 1e-20)
    f["centroid_hz"] = float(np.sum(freqs * power) / tot)
    f["flatness"] = float(np.mean(librosa.feature.spectral_flatness(
        S=np.sqrt(S) + 1e-12)))
    f["rolloff95_hz"] = float(np.interp(
        0.95, np.cumsum(power) / tot, freqs))

    # ---- level / dynamics ----------------------------------------------
    db = _frame_db(m, sr)
    live = db > (np.percentile(db, 95) - 45.0)
    act = db[live] if live.any() else db
    peak = 20 * np.log10(float(np.max(np.abs(m))) + 1e-12)
    rms = 20 * np.log10(float(np.sqrt(np.mean(m ** 2))) + 1e-12)
    f["crest_db"] = peak - rms
    f["lra_db"] = float(np.percentile(act, 95) - np.percentile(act, 10))
    f["spread_db"] = float(np.percentile(act, 90) - np.percentile(act, 10))
    # Noise floor relative to programme — separation bleed and room tone.
    f["floor_rel_db"] = float(np.percentile(db, 5) - np.percentile(db, 95))
    # Ride fingerprint: centred loudness quantiles.
    q = np.percentile(act, np.linspace(0, 100, 21))
    f["ride_q"] = q - np.median(act)

    # ---- stereo ---------------------------------------------------------
    mid, side = (L + R) * 0.5, (L - R) * 0.5
    f["side_pct"] = 100.0 * float(np.sqrt(np.mean(side ** 2))) / (
        float(np.sqrt(np.mean(mid ** 2))) + 1e-12)
    denom = np.sqrt(np.mean(L ** 2) * np.mean(R ** 2)) + 1e-20
    f["lr_corr"] = float(np.mean(L * R) / denom)
    SM = np.abs(librosa.stft(np.ascontiguousarray(mid), n_fft=_NFFT, hop_length=_HOP)) ** 2
    SS = np.abs(librosa.stft(np.ascontiguousarray(side), n_fft=_NFFT, hop_length=_HOP)) ** 2
    pm, ps = np.mean(SM, axis=1), np.mean(SS, axis=1)
    wb = np.empty(len(_EDGES_W) - 1)
    for i in range(len(wb)):
        msk = (freqs >= _EDGES_W[i]) & (freqs < _EDGES_W[i + 1])
        wb[i] = (10 * np.log10((float(np.sum(ps[msk])) + 1e-20) /
                               (float(np.sum(pm[msk])) + 1e-20)) if msk.any() else -60.0)
    f["width_bands"] = np.clip(wb, -60.0, 20.0)

    # ---- sibilance ------------------------------------------------------
    # Measured on the brightest frames only, so it reports how sibilants are
    # handled rather than how bright the whole vocal is.
    zcr = librosa.feature.zero_crossing_rate(m, frame_length=2048, hop_length=_HOP)[0]
    k = min(len(zcr), S.shape[1])
    sib_fr = np.argsort(zcr[:k])[-max(1, k // 10):]
    band_s = (freqs >= 5000) & (freqs < 9000)
    band_b = (freqs >= 1000) & (freqs < 4000)
    seg = S[:, sib_fr]
    f["sibilance_db"] = 10 * np.log10(
        (float(np.sum(seg[band_s])) + 1e-20) / (float(np.sum(seg[band_b])) + 1e-20))

    # ---- transients -----------------------------------------------------
    env = librosa.onset.onset_strength(y=m, sr=sr, hop_length=_HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=_HOP,
                                        units="frames", backtrack=False)
    rises = []
    for o in onsets:
        a, b = max(0, o - 6), min(len(db), o + 12)
        seg_db = db[a:b]
        if len(seg_db) < 4:
            continue
        lo, hi_ = np.min(seg_db), np.max(seg_db)
        if hi_ - lo < 6:
            continue
        rises.append((hi_ - lo) / max(1, (np.argmax(seg_db) + 1)))  # dB per frame
    f["attack_db_per_frame"] = float(np.median(rises)) if rises else 0.0
    f["onset_rate_hz"] = float(len(onsets) / (len(m) / sr + 1e-9))

    # ---- modulation spectrum -------------------------------------------
    # How the loudness envelope fluctuates. Reverb and compression both damp
    # fast modulation; this is the dimension the amplitude-envelope reverb
    # measures could not capture.
    e = 10 ** (db / 20.0)
    e = e - np.mean(e)
    fr = sr / _HOP
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    mf = np.fft.rfftfreq(len(e), d=1.0 / fr)
    mod = np.empty(len(_MOD_EDGES) - 1)
    for i in range(len(mod)):
        msk = (mf >= _MOD_EDGES[i]) & (mf < _MOD_EDGES[i + 1])
        mod[i] = 10 * np.log10(float(np.sum(spec[msk])) + 1e-20) if msk.any() else -200.0
    ok = mod > -190
    if ok.any():
        mod = mod - float(np.mean(mod[ok]))
    f["mod_spec"] = mod

    return f


# Per-dimension weighting is deliberately absent: there is no principled way to
# trade a dB of sibilance against a percentage point of width, so the report
# lists dimensions separately rather than inventing a single score.
_VECTORS = ("bands32", "width_bands", "mod_spec", "ride_q")
_SCALARS = ("crest_db", "lra_db", "spread_db", "floor_rel_db", "air_db",
            "sibilance_db", "side_pct", "lr_corr", "flatness", "centroid_hz",
            "rolloff95_hz", "attack_db_per_frame", "onset_rate_hz")


def distances(a: dict, b: dict) -> dict:
    """Per-dimension distance between two measure() results."""
    d = {}
    for k in _VECTORS:
        d[k] = float(np.mean(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
    for k in _SCALARS:
        d[k] = float(abs(a[k] - b[k]))
    # Percent-scale features are easier to read normalised.
    d["centroid_hz"] /= 1000.0
    d["rolloff95_hz"] /= 1000.0
    return d
