"""Late-chain matching stages.

These four properties are all measured on the FINAL signal, which is why
correcting them earlier in the chain does not stick: `match_crest` ran before
reverb, width and the doubler, and every one of those puts peaks back. Crest
is peak-minus-RMS, i.e. scale invariant, so no amount of normalisation moves
it -- only limiting does, and it has to be the last thing that happens.

Every target here is measured from the reference being matched. There are no
fixed targets: a sparse ballad and a dense rap reference pull these in
opposite directions and both are correct.

Measured on the Hide pair (processed output vs the separated reference lead),
each stage in isolation:

    crest      0.0 -> 9.7      floor  0.0 -> 9.3
    ride       3.4 -> 7.0      sibilance 5.8 -> 9.3

They interact, though -- the ride match lifts LRA to 8.4 and the floor
expander then drops it to 4.3 -- so the caller applies them through the same
accept-if-better gate the rest of the chain uses rather than unconditionally.
"""
from __future__ import annotations

import os
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

_HOP = 512
_NFFT = 2048
# Matches similarity.measure(): frames this far below the 95th percentile are
# not programme. Keep in sync or the ride match optimises a different set of
# frames from the one being scored.
_LIVE_WINDOW_DB = 45.0
# How quickly the ride gain recovers. Long releases hold the gain down into the
# next phrase entry and cost transient fidelity.
_RIDE_RELEASE_MS = float(os.environ.get("APP_RIDE_RELEASE_MS", "60"))


def _frame_db(m: np.ndarray) -> np.ndarray:
    r = librosa.feature.rms(y=np.ascontiguousarray(m), frame_length=_NFFT,
                            hop_length=_HOP)[0]
    return 20 * np.log10(np.maximum(r, 1e-9))


def _chan_axis(y: np.ndarray) -> int:
    """Which axis holds channels.

    chain.py is not consistent about this -- the reverb block indexes y[:, 0]
    (samples first) while the autotune block does np.mean(y, axis=0)
    (channels first). They only coexist because y is mono in practice. Rather
    than pick one and be silently wrong on stereo, infer it: a channel axis is
    1 or 2 long, a sample axis is thousands.
    """
    return 0 if y.shape[0] <= 2 else 1


def _mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return y.mean(axis=_chan_axis(y))


def _smooth_gain(g: np.ndarray, sr: int, atk_ms: float = 20.0,
                 rel_ms: float = 150.0) -> np.ndarray:
    """One-pole attack/release on a per-frame linear gain curve.

    Fast to duck and slow to recover. An unsmoothed curve chatters frame to
    frame, which is what made the old gate audible as fades.
    """
    a = float(np.exp(-1.0 / max(1e-6, atk_ms / 1000.0 * sr / _HOP)))
    r = float(np.exp(-1.0 / max(1e-6, rel_ms / 1000.0 * sr / _HOP)))
    out = np.empty_like(g)
    s = float(g[0])
    for i, v in enumerate(g):
        c = a if v < s else r
        s = c * s + (1.0 - c) * float(v)
        out[i] = s
    return out


def _broadcast(y: np.ndarray, per_sample: np.ndarray) -> np.ndarray:
    """Shape a per-sample curve so it multiplies y along its sample axis."""
    if y.ndim == 1:
        return per_sample
    return per_sample[None, :] if _chan_axis(y) == 0 else per_sample[:, None]


def _n_samples(y: np.ndarray) -> int:
    return len(y) if y.ndim == 1 else y.shape[1 - _chan_axis(y)]


def _apply_frame_gain(y: np.ndarray, gf: np.ndarray) -> np.ndarray:
    n = _n_samples(y)
    g = np.interp(np.arange(n), np.arange(len(gf)) * _HOP, gf)
    return y * _broadcast(y, g)


def match_ride(y: np.ndarray, sr: int, ref_ride_q) -> np.ndarray:
    """Histogram-match the active-frame loudness distribution to the reference.

    `ride_q` is that distribution's shape (21 quantiles, centred on its own
    median), so mapping our quantiles onto the reference's matches it directly.
    The reference's centre is deliberately NOT imported -- only its shape --
    because absolute level is the loudness stage's job.
    """
    m = _mono(y)
    db = _frame_db(m)
    live = db > (np.percentile(db, 95) - _LIVE_WINDOW_DB)
    act = db[live] if live.any() else db
    src_q = np.percentile(act, np.linspace(0, 100, 21))
    dst_q = float(np.median(act)) + np.asarray(ref_ride_q, dtype=float)
    # Quantile curves must be increasing for interp to behave.
    src_q = np.maximum.accumulate(src_q)
    dst_q = np.maximum.accumulate(dst_q)
    gdb = np.clip(np.interp(db, src_q, dst_q) - db, -12.0, 12.0)
    # Sparing the gaps and releasing faster were both tried, to stop this
    # undoing the gap ducking and blunting attacks. Measured through the chain
    # they cost more than they gained (Hide 6.89 -> 6.74), so the plain
    # quantile map stands.
    g = _smooth_gain(10 ** (gdb / 20.0), sr, atk_ms=30.0, rel_ms=200.0)
    return _apply_frame_gain(y, g)


def expand_floor(y: np.ndarray, sr: int, target_floor_rel_db: float,
                 fraction: float = 1.0) -> np.ndarray:
    """Downward expansion below a searched threshold, to match the reference's
    gap-to-programme ratio.

    SUPERSEDED in the chain by `reduce_gap_noise`, which uses a gate-style
    envelope (fast open, slow close) instead of this one's fast-duck /
    slow-recover, and so does not duck the start of every phrase. Kept because
    it is the plain expander and the tests pin its behaviour.

    A separated reference has unnaturally clean gaps -- the separator emits
    near-silence where there is no voice -- while our output has reverb tails
    there. On the Hide pair that was a 13 dB difference, the single largest
    gap of any dimension. `fraction` allows closing only part of it, since
    expansion trades against LRA.
    """
    m = _mono(y)
    db = _frame_db(m)
    cur = float(np.percentile(db, 5) - np.percentile(db, 95))
    target = cur + float(fraction) * (float(target_floor_rel_db) - cur)
    p95 = float(np.percentile(db, 95))
    best_g, best_err = None, 1e9
    for thr_rel in np.arange(-60.0, -6.0, 3.0):
        over = db - (p95 + thr_rel)
        for ratio in (1.5, 2.0, 3.0, 4.0):
            gdb = np.maximum(np.where(over < 0, over * (ratio - 1.0), 0.0), -40.0)
            g = _smooth_gain(10 ** (gdb / 20.0), sr)
            nd = db + 20 * np.log10(np.maximum(g, 1e-9))
            err = abs(float(np.percentile(nd, 5) - np.percentile(nd, 95)) - target)
            if err < best_err:
                best_err, best_g = err, g
    if best_g is None:
        return y
    return _apply_frame_gain(y, best_g)


def match_sibilance(y: np.ndarray, sr: int, cur_db: float,
                    target_db: float) -> np.ndarray:
    """Match 5-9 kHz energy against 1-4 kHz, in sibilant frames only.

    The metric looks at the top decile of frames by zero-crossing rate, so a
    broadband high shelf is the wrong tool: it closed sibilance but cost 0.6
    on the 32-band curve and 0.9 on air, because it brightened the whole
    vocal instead of the esses. Band-limit and frame-gate it.

    This corrects in BOTH directions. Hide's reference lead is 5 dB *more*
    sibilant than the dry, so the right move there is a boost -- a de-esser
    alone can never reach it.
    """
    gap = float(target_db) - float(cur_db)
    if abs(gap) < 0.3:
        return y
    m = _mono(y)
    sos = butter(2, [5000.0 / (sr / 2), 9000.0 / (sr / 2)], btype="band",
                 output="sos")
    band = sosfilt(sos, y, axis=(-1 if y.ndim == 1 else 1 - _chan_axis(y)))
    zcr = librosa.feature.zero_crossing_rate(np.ascontiguousarray(m),
                                             frame_length=_NFFT,
                                             hop_length=_HOP)[0]
    sel = (zcr >= np.percentile(zcr, 85)).astype(np.float64)
    sel = _smooth_gain(np.maximum(sel, 1e-6), sr, atk_ms=5.0, rel_ms=30.0)
    sel_s = np.interp(np.arange(_n_samples(y)), np.arange(len(sel)) * _HOP, sel)
    g = 10 ** (np.clip(gap, -12.0, 12.0) / 20.0) - 1.0
    return y + g * band * _broadcast(y, sel_s)


def limit_crest(y: np.ndarray, sr: int, target_crest_db: float) -> np.ndarray:
    """Reduce peak-to-RMS toward the reference's crest, WITHOUT waveshaping.

    Must run last. Crest is peak-minus-RMS and therefore scale invariant, so
    the final loudness match cannot disturb it, but any stage that adds peaks
    can -- which is how the mid-chain version was being undone.

    The first implementation computed a per-sample gain,

        gr = where(|x| > thr, thr + (|x| - thr) * 0.15, |x|) / |x|

    which is an instantaneous nonlinearity: a mathematically clean harmonic
    tone with no content above 6 kHz at all (-207 dB) came out with that band
    at -34.6 dB. Reported by ear as "crackle / distortion / fizz", worst on
    loud notes, which is exactly where a waveshaper is driven hardest.

    So the gain is computed per FRAME from the peak envelope and smoothed with
    a limiter's attack and release before it is applied. Gain that moves
    slowly relative to the waveform scales the signal; gain that moves at
    sample rate reshapes it, and reshaping is what generates harmonics.
    """
    m = _mono(y)
    peak = float(np.max(np.abs(m)))
    if peak < 1e-9:
        return y
    n_fr = max(2, 1 + len(m) // _HOP)
    pad = np.pad(np.abs(m), (0, n_fr * _HOP))
    pk = np.array([pad[i * _HOP:(i + 1) * _HOP].max() for i in range(n_fr)])

    def crest_of(v):
        # Judge on the SAME central excerpt the accept-if-better gate uses.
        #
        # Crest is dominated by the single loudest sample, so the figure for a
        # whole track and for a 30 s window are different numbers. Optimising
        # the full-length one while the gate scored an excerpt meant the stage
        # could land its target exactly (16.29 against 16.18) and still be
        # rejected as a regression.
        mm = _mono(v)
        cap = int(_CREST_MEASURE_S * sr)
        if len(mm) > cap:
            st = (len(mm) - cap) // 2
            mm = mm[st:st + cap]
        return (20 * np.log10(float(np.max(np.abs(mm))) + 1e-12)
                - 20 * np.log10(float(np.sqrt(np.mean(mm ** 2))) + 1e-12))

    best, best_err = y, abs(crest_of(y) - float(target_crest_db))
    for t in np.linspace(0.95, 0.15, 33):
        thr = peak * t
        gr = np.where(pk > thr, (thr + (pk - thr) * 0.15) / np.maximum(pk, 1e-12), 1.0)
        # Fast to duck, slower to recover -- a limiter's envelope.
        g = _smooth_gain(gr, sr, atk_ms=2.0, rel_ms=60.0)
        yy = _apply_frame_gain(y, g)
        err = abs(crest_of(yy) - float(target_crest_db))
        if err < best_err:
            best_err, best = err, yy
    return best



# Same edges the score uses. A corrector on a COARSER basis than the score
# cannot reach it: apply_stereo_image controls three bands (low/mid/high)
# while width is measured across these eight, which is why width per band sat
# at 4.4/10 no matter how the three were set.
_EDGES_W = np.geomspace(100.0, 12000.0, 9)
_WIN = 2048


def match_width_bands(y: np.ndarray, sr: int, target_width_bands) -> np.ndarray:
    """Match the reference's side-to-mid ratio in each of eight bands.

    Works on the mid/side spectrum directly, so the correction is applied at
    exactly the resolution it is scored at, and the mid channel is left
    untouched -- mono fold-down is unchanged.

    A band with no side energy cannot be widened by gain alone (there is
    nothing to amplify), so those are left alone rather than multiplied by a
    huge number; creating width is apply_stereo_image's job, this stage only
    shapes it.
    """
    if y.ndim != 2:
        return y
    ax = _chan_axis(y)
    L = y[0] if ax == 0 else y[:, 0]
    R = y[1] if ax == 0 else y[:, 1]
    mid = (L + R) * 0.5
    side = (L - R) * 0.5
    if float(np.sqrt(np.mean(side ** 2))) < 1e-7:
        return y

    Sm = librosa.stft(np.ascontiguousarray(mid), n_fft=_WIN, hop_length=_HOP)
    Ss = librosa.stft(np.ascontiguousarray(side), n_fft=_WIN, hop_length=_HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_WIN)
    pm = np.abs(Sm) ** 2
    ps = np.abs(Ss) ** 2

    tgt = np.asarray(target_width_bands, dtype=float)
    # Build a SMOOTH gain curve across frequency.
    #
    # Setting a constant gain per band steps the curve at every band edge, and
    # an STFT multiplied by a stepped curve is a brick-wall filter: its impulse
    # response rings, which is heard as a metallic or robotic edge. Solve each
    # band's correction at its centre frequency, then interpolate between the
    # centres in log-frequency so the filter stays gentle.
    centres, corrections = [], []
    for i in range(len(_EDGES_W) - 1):
        msk = (freqs >= _EDGES_W[i]) & (freqs < _EDGES_W[i + 1])
        if not msk.any() or i >= len(tgt):
            continue
        s_e = float(np.sum(ps[msk])) + 1e-20
        m_e = float(np.sum(pm[msk])) + 1e-20
        cur = 10 * np.log10(s_e / m_e)
        centres.append(np.sqrt(_EDGES_W[i] * _EDGES_W[i + 1]))
        # A band with no side content cannot be widened by gain; ask for none.
        corrections.append(0.0 if cur <= -59.0
                           else float(np.clip(tgt[i] - cur, -24.0, 24.0)))
    if not centres:
        return y
    lf = np.log(np.maximum(freqs, 1.0))
    g_db = np.interp(lf, np.log(np.asarray(centres)), np.asarray(corrections))
    gains = 10 ** (g_db / 20.0)

    side_new = librosa.istft(Ss * gains[:, None], hop_length=_HOP,
                             n_fft=_WIN, length=len(side))
    Ln = mid + side_new
    Rn = mid - side_new
    out = np.stack([Ln, Rn], axis=ax)
    return out.astype(y.dtype, copy=False)


# How far below programme a frame must sit before it counts as a gap, and the
# most a candidate may disturb the singing before it is rejected outright.
_GAP_BELOW_DB = 30.0
_MAX_PROGRAMME_DISTURB_DB = 0.25



def _gap_mask(db: np.ndarray):
    """Frames that sit between phrases, and how far down that turned out to be.

    Adaptive on purpose. A fixed "30 dB below programme" rule finds nothing on
    a noisy source -- whose gaps might only be 20 dB down -- and that is
    exactly the material that needs gap treatment most. Returns (mask, depth)
    or (None, None) when there is nothing usable.
    """
    prog = float(np.percentile(db, 95))
    need = max(4, int(0.02 * len(db)))
    for cand in (_GAP_BELOW_DB, 25.0, 20.0, 15.0):
        sel = db < (prog - cand)
        if sel.sum() >= need:
            return sel, cand
    return None, None

def reduce_gap_noise(y: np.ndarray, sr: int, target_floor_rel_db: float):
    """Bring the level BETWEEN phrases down to the reference's, and nothing else.

    Reported by ear -- "a bit of noise in the processed one" -- and confirmed:
    the gaps sat 16 dB louder than the reference's, and their spectral tilt had
    gone from -4.0 dB (dark, like the dry input) to -0.1 (flat), because the
    chain's high-frequency boost lifts the input's own noise along with the
    voice. The dry acapella is itself noisy, so this is partly inherited and
    partly ours.

    Two things make this safe where the old expander was not:

    * The envelope OPENS fast and CLOSES slowly. `expand_floor` did the
      reverse -- 20 ms to duck, 150-400 ms to recover -- so the gain was still
      down when the next phrase started and every entry was ducked by 7-22 dB.
      That is the "fades" failure mode from earlier in this project, and it is
      what a gate's attack/release is for.
    * Candidates that move programme frames at all are rejected. The chosen
      setting must be inaudible on the voice, not merely good on average, so
      this is judged directly rather than through the composite score -- which
      rated the audible noise as an acceptable trade and was wrong.

    Returns (audio, info) where info records what was applied.
    """
    m = _mono(y)
    db = _frame_db(m)
    if len(db) < 8:
        return y, {"applied": False, "reason": "too short"}
    prog = float(np.percentile(db, 95))
    loud = db > (prog - 12.0)
    gaps, gap_below = _gap_mask(db)
    if gaps is None or not loud.any():
        return y, {"applied": False, "reason": "no gaps"}

    cur = float(np.percentile(db[gaps], 50) - prog)
    target = float(target_floor_rel_db)
    if cur <= target + 1.0:
        return y, {"applied": False, "reason": "already quiet enough",
                   "gap_rel_db": cur}

    best, best_err, best_info = None, 1e9, None
    for thr_rel in (-24.0, -30.0, -36.0, -42.0):
        over = db - (prog + thr_rel)
        for ratio in (1.5, 2.0, 3.0, 4.0, 6.0):
            gdb = np.maximum(np.where(over < 0, over * (ratio - 1.0), 0.0), -30.0)
            # atk_ms is the coefficient used while the gain FALLS, rel_ms while
            # it rises: slow to close, fast to open.
            g = _smooth_gain(10 ** (gdb / 20.0), sr, atk_ms=200.0, rel_ms=5.0)
            gs = 20 * np.log10(np.maximum(g, 1e-9))
            if float(np.min(gs[loud])) < -_MAX_PROGRAMME_DISTURB_DB:
                continue                      # would be audible on the voice
            got = float(np.percentile((db + gs)[gaps], 50) - prog)
            err = abs(got - target)
            if err < best_err:
                best_err, best = err, g
                best_info = {"threshold_rel_db": thr_rel, "ratio": ratio,
                             "gap_rel_db": got, "from_gap_rel_db": cur,
                             "gap_defined_below_db": gap_below,
                             "programme_dip_db": float(np.min(gs[loud]))}
    if best is None:
        return y, {"applied": False, "reason": "no setting spared the voice"}
    best_info["applied"] = True
    return _apply_frame_gain(y, best), best_info


_TILT_SPLIT_HZ = 5000.0
_TILT_MEASURE_S = 45.0
# Matches the accept-if-better gate's excerpt length in worker.py.
_CREST_MEASURE_S = 30.0
# Never darken the gaps by more than this, however far off the target reads.
# The reference tilt measured -9.6 on the 120s mono analysis signal against
# -4.8 on the full stereo stem, and chasing the larger figure over-cut: it
# took the noise floor score from 7.3 down to 6.3 by overshooting the
# reference's gap level rather than landing on it.
_MAX_TILT_CUT_DB = 5.0


def gap_tilt(y: np.ndarray, sr: int, gap_below_db: float | None = None) -> float:
    """Spectral tilt of the material BETWEEN phrases, in dB (HF minus mid).

    Bright noise reads as hiss; dark noise reads as room. Two signals can hold
    identical gap ENERGY and sound completely different, so the level alone is
    not enough to judge by.
    """
    m = _mono(y)
    # Bounded excerpt: tilt is distributional, and measuring it over the whole
    # file cost 35s across the correction's four passes.
    _cap = int(_TILT_MEASURE_S * sr)
    if len(m) > _cap:
        _st = (len(m) - _cap) // 2
        m = m[_st:_st + _cap]
    db = _frame_db(m)
    if gap_below_db is None:
        sel, _ = _gap_mask(db)
        if sel is None:
            return float("nan")
    else:
        sel = db < (float(np.percentile(db, 95)) - gap_below_db)
    S = np.abs(librosa.stft(np.ascontiguousarray(m), n_fft=_NFFT, hop_length=_HOP))
    n = min(S.shape[1], len(sel))
    G = S[:, :n][:, sel[:n]]
    if G.shape[1] < 5:
        return float("nan")
    fr = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    hi = float(G[(fr >= 6000) & (fr < 16000)].sum())
    lo = float(G[(fr >= 200) & (fr < 2000)].sum())
    return 20.0 * np.log10((hi + 1e-12) / (lo + 1e-12))


def match_gap_tilt(y: np.ndarray, sr: int, target_tilt_db: float,
                   gap_below_db: float | None = None):
    """Darken the residual noise between phrases to the reference's colour.

    The chain's high-frequency boost is applied to match the reference's
    brightness, but it lifts the INPUT's noise along with the voice: measured
    on Hide, gap tilt went from -4.0 dB on the dry to -0.1 on the output,
    against -4.8 on the reference. Cutting the top only where there is no
    voice removes the hiss without touching the singing's air.

    Returns (audio, info).
    """
    cur = gap_tilt(y, sr, gap_below_db)
    if not np.isfinite(cur) or not np.isfinite(target_tilt_db):
        return y, {"applied": False, "reason": "tilt unmeasurable"}
    excess = float(cur) - float(target_tilt_db)
    if excess <= 0.5:
        return y, {"applied": False, "reason": "already dark enough",
                   "gap_tilt_db": cur}

    m = _mono(y)
    db = _frame_db(m)
    if gap_below_db is None:
        sel, gap_below_db = _gap_mask(db)
        if sel is None:
            return y, {"applied": False, "reason": "no gaps"}
    else:
        sel = db < (float(np.percentile(db, 95)) - gap_below_db)
    # 1 where there is no voice, 0 over programme.
    gapness = sel.astype(np.float64)
    # Engages in ~50 ms once the voice stops and releases in 5 ms when it
    # returns. The LEVEL gate has to close slowly or it chops reverb tails,
    # but a brief high-frequency dip in a gap is inaudible, and closing slowly
    # here left most gap frames only partly cut: tilt moved -0.6 -> -1.5
    # against a -4.8 target.
    gapness = 1.0 - _smooth_gain(1.0 - gapness, sr, atk_ms=50.0, rel_ms=5.0)

    sos = butter(2, min(_TILT_SPLIT_HZ / (sr / 2), 0.99), btype="high",
                 output="sos")
    g_s = np.interp(np.arange(_n_samples(y)), np.arange(len(gapness)) * _HOP,
                    gapness)
    gb = _broadcast(y, g_s)

    # One pass closes roughly two thirds of the gap, because the cut only acts
    # where the mask is open while the tilt is measured over every gap frame.
    # Iterate to convergence rather than over-cutting in a single step.
    out = y
    passes = 0
    for _ in range(4):
        now = gap_tilt(out, sr, gap_below_db)
        excess = now - float(target_tilt_db)
        remaining = _MAX_TILT_CUT_DB - (cur - now)      # budget already spent
        if not np.isfinite(excess) or excess <= 0.5 or remaining <= 0.25:
            break
        step = min(excess, remaining, 12.0)
        band = sosfilt(sos, out, axis=(-1 if out.ndim == 1 else 1 - _chan_axis(out)))
        out = out - (1.0 - 10 ** (-step / 20.0)) * band * gb
        passes += 1
    return out, {"applied": passes > 0, "gap_tilt_db": cur,
                 "final_tilt_db": gap_tilt(out, sr, gap_below_db),
                 "target_tilt_db": float(target_tilt_db), "passes": passes}


# Harmonics 2-8 against the fundamental, measured on voiced frames. This is
# what "rich" means for a voice: a thin vocal has the fundamental and little
# else, a rich one has a stack of overtones above it.
_HARM_NFFT = 4096
_HARM_TOLERANCE = 0.15
# Drive amounts searched, gentlest first: this should be the least saturation
# that reaches the reference, not the most the curve can produce.
_HARM_DRIVES = (0.6, 0.9, 1.3, 1.8, 2.5, 3.5, 5.0)


def harmonic_richness(y: np.ndarray, sr: int) -> float:
    """Energy in harmonics 2-8 relative to the fundamental."""
    m = _mono(y)
    m = np.ascontiguousarray(np.asarray(m, dtype=np.float32))
    if len(m) < sr:
        return float("nan")
    r = librosa.feature.rms(y=m, frame_length=_NFFT, hop_length=_HOP)[0]
    db = 20 * np.log10(np.maximum(r, 1e-9))
    voiced = db > (np.percentile(db, 95) - 15.0)
    S = np.abs(librosa.stft(m, n_fft=_HARM_NFFT, hop_length=_HOP))
    n = min(S.shape[1], len(voiced))
    V = S[:, :n][:, voiced[:n]]
    if V.shape[1] < 4:
        return float("nan")
    # Track the fundamental rather than assuming one: a fixed f0 makes the
    # measure move when the singer's range differs from the reference's.
    try:
        f0, _, _ = librosa.pyin(m[: sr * 45].astype(float), fmin=65, fmax=600,
                                sr=sr, frame_length=2048)
        f0m = float(np.nanmedian(f0))
    except Exception:
        f0m = float("nan")
    if not np.isfinite(f0m) or f0m <= 0:
        return float("nan")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_HARM_NFFT)
    band = lambda a, b: float(V[(freqs >= a) & (freqs < b)].sum())
    fund = band(f0m * 0.8, f0m * 1.3) + 1e-12
    return (band(f0m * 1.7, f0m * 8.5) + 1e-12) / fund


# NOTE: a match_harmonics() corrector was written here and REMOVED. The
# measure above does not respond to the thing that adds harmonics: pushing the
# output through asymmetric soft clipping at drives from 0.6 to 20 moved
# richness 4.87 -> 4.86 -> 4.82 -> 4.73 -> 4.63, i.e. DOWN, because saturation
# also fills the fundamental band and compresses the peaks the measure is
# taken over. Driving a correction from it would optimise the wrong thing --
# the same trap as the chorus detector. Richness is exposed as a user control
# instead, until a measure exists that rises when overtones are added.
