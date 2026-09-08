"""
Closed-loop matching: measure the OUTPUT, compare it to the REFERENCE, correct,
repeat.

The rest of the chain is open-loop - analyse the reference, estimate a
parameter, apply it once. That makes output quality a function of estimator
accuracy, and blind estimation of a mix parameter from a finished vocal is
ill-posed: many (dry, chain) pairs produce the same wet result, so no
estimator can be relied on to recover the right one.

Closing the loop sidesteps that. We do not need to know that the reference has
a 1.8 s tail; we need the setting whose *output* measures the same tail the
reference measures. An estimator that is biased short still converges, because
its output is measured and corrected rather than trusted.

Nothing here holds a target of its own. Every target is measured off this
reference on this job, so the behaviour stays genre-dynamic by construction -
strictly more so than detection, which at least assumes the effect model.
"""
from __future__ import annotations

import time

import numpy as np
import librosa

# Log-spaced analysis bands spanning the vocal range. Wider than the six-band
# gap map the one-shot pass used, so the loop can see (and fix) shapes that
# coarser bands average away.
#
# 12, not 24, and that was measured rather than assumed. Scoring on an
# independent 32-band basis, a 24-band corrector halves the spectral residual
# in isolation (6.06 -> 1.61 dB vs 6.06 -> 4.46) — but end to end it made
# things worse overall: bands32 improved 32% -> 44% while LRA went -9% -> -22%,
# spread -7% -> -20% and crest +10% -> +2%. Mean across 16 dimensions fell from
# 20% to 18%. Correcting the spectrum harder perturbs peak-to-RMS faster than
# the later stages can repair, so the extra resolution is not free.
_BAND_EDGES = np.array(
    [60, 110, 190, 320, 520, 820, 1300, 2000, 3100, 4700, 7000, 10500, 16000],
    dtype=float,
)
_BAND_Q = 1.6


def _band_shape_db(mono: np.ndarray, sr: int, edges=None) -> np.ndarray:
    """
    Per-band energy in dB, mean-removed.

    Mean-removed so this describes spectral *shape* only: absolute level is
    owned by the final LUFS match, and leaving it in would make the loop
    fight the loudness stage.
    """
    if mono.ndim > 1:
        mono = mono.mean(axis=-1)
    n_fft = 4096
    spec = np.abs(librosa.stft(np.ascontiguousarray(mono), n_fft=n_fft)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = np.mean(spec, axis=1)
    edges = _BAND_EDGES if edges is None else np.asarray(edges, dtype=float)
    out = np.empty(len(edges) - 1)
    for i in range(len(out)):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = 10.0 * np.log10(float(np.sum(power[m])) + 1e-20) if m.any() else -200.0
    valid = out > -190.0
    if valid.any():
        out = out - float(np.mean(out[valid]))
    return out


def spectral_shape_gap_db(a_mono: np.ndarray, b_mono: np.ndarray, sr: int) -> float:
    """Mean absolute per-band difference in spectral shape. 0 = identical."""
    return float(np.mean(np.abs(_band_shape_db(a_mono, sr) - _band_shape_db(b_mono, sr))))


def match_spectrum(
    y: np.ndarray,
    sr: int,
    ref_mono: np.ndarray,
    iters: int = 4,
    tol_db: float = 0.30,
    max_step_db: float = 3.5,
    damping: float = 0.75,
    edges=None,
    q=None,
):
    """
    Iteratively EQ `y` until its band shape matches `ref_mono`'s.

    Each pass measures the remaining gap and corrects a damped fraction of it,
    which converges without ringing from one big correction. Returns
    (audio, info).
    """
    from processor.dsp.eq import EqBand, apply_eq
    from processor.dsp.deesser import apply_deesser, measure_sibilance_db

    edges = _BAND_EDGES if edges is None else np.asarray(edges, dtype=float)
    q = _BAND_Q if q is None else float(q)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric centres
    ref_shape = _band_shape_db(ref_mono, sr, edges)
    history = []

    for _ in range(max(1, iters)):
        mono = y if y.ndim == 1 else y.mean(axis=1)
        gap = ref_shape - _band_shape_db(mono, sr, edges)
        worst = float(np.max(np.abs(gap)))
        history.append(round(float(np.mean(np.abs(gap))), 3))
        if worst < tol_db:
            break
        bands = [
            EqBand(f=float(c), gain_db=float(np.clip(g * damping, -max_step_db, max_step_db)),
                   q=q)
            for c, g in zip(centers, gap)
            if abs(g) >= tol_db
        ]
        if not bands:
            break
        if y.ndim == 2:
            y = np.stack([apply_eq(y[:, 0], sr, bands), apply_eq(y[:, 1], sr, bands)], axis=1)
        else:
            y = apply_eq(y, sr, bands)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    mono = y if y.ndim == 1 else y.mean(axis=1)
    final = float(np.mean(np.abs(ref_shape - _band_shape_db(mono, sr, edges))))
    return y, {"gap_history_db": history, "final_gap_db": round(final, 3)}


def _crest_db(y: np.ndarray) -> float:
    a = np.abs(y)
    peak = float(np.max(a)) + 1e-12
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-12
    return 20.0 * np.log10(peak / rms)


def match_crest(
    y: np.ndarray,
    sr: int,
    ref_mono: np.ndarray,
    iters: int = 5,
    tol_db: float = 0.5,
    target_crest_db: float | None = None,
):
    """
    Compress until the output's crest factor matches the reference's.

    Crest is monotone in compression amount, so this converges by bisection on
    threshold. Only ever reduces crest - restoring peaks to an over-compressed
    signal would mean inventing transients that aren't there.
    """
    from processor.dsp.master_bus import _limiter_gain

    # target_crest_db lets a caller aim somewhere between the current crest
    # and the reference's, for a partial amount rather than a full match.
    target = _crest_db(ref_mono) if target_crest_db is None else float(target_crest_db)
    start = _crest_db(y if y.ndim == 1 else y.mean(axis=1))
    if start <= target + tol_db:
        return y, {"skipped": "output already at or below reference crest",
                   "target": round(target, 2), "current": round(start, 2)}

    # Peak limiting, not RMS compression.
    #
    # Crest is peak-to-RMS, and the chain's compressor detects on block RMS --
    # so it never sees the isolated sample peaks that set the peak term. Swept
    # across every threshold it left crest at 22.7 dB or pushed it to 25.6,
    # never down: compressing lowers RMS while the peak stays where the
    # limiter left it, which WIDENS the ratio. Pulling the peaks down and
    # making the level back up is what actually closes it.
    def limited(ceil_db):
        ceiling = 10 ** (ceil_db / 20.0)
        block = max(1, int(sr * 0.001))
        flat = np.abs(y).max(axis=1) if y.ndim == 2 else np.abs(y)
        nb = int(np.ceil(len(flat) / block))
        pad = np.zeros(nb * block)
        pad[: len(flat)] = flat
        g = _limiter_gain(pad.reshape(nb, block).max(axis=1), block, sr, ceiling, 80.0)
        gi = np.interp(np.arange(len(flat)), (np.arange(nb) + 0.5) * block, g)
        out = y * (gi[:, None] if y.ndim == 2 else gi)
        pk_in = float(np.max(np.abs(y))) + 1e-12
        pk_out = float(np.max(np.abs(out))) + 1e-12
        return out * (pk_in / pk_out)   # make the level back up

    peak_db = 20.0 * np.log10(float(np.max(np.abs(y))) + 1e-12)
    lo, hi = peak_db - 24.0, peak_db - 0.2   # lo = hard limiting, hi = none
    best, best_info = y, {"target": round(target, 2), "current": round(start, 2),
                          "note": "no threshold reached target"}
    best_err = abs(start - target)
    for _ in range(max(1, iters)):
        mid = 0.5 * (lo + hi)
        cand = limited(mid)
        got = _crest_db(cand if cand.ndim == 1 else cand.mean(axis=1))
        # Keep the closest candidate seen, not simply the last one tried --
        # bisection can end on a worse guess than one it already passed.
        if abs(got - target) < best_err:
            best, best_err = cand, abs(got - target)
            best_info = {"target": round(target, 2), "reached": round(got, 2),
                         "threshold_db": round(mid, 1)}
        if abs(got - target) < tol_db:
            break
        if got > target:
            hi = mid          # still too peaky -> limit harder
        else:
            lo = mid
    return np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0), best_info


# ---------------------------------------------------------------------------
# Not implemented on purpose: closed-loop reverb amount.
#
# The obvious next stage would search the reverb blend whose output tail energy
# matches the reference's. It was built and measured, and it does not work:
# on real vocal stems (which already carry the mix's reverb) adding a plate
# barely moves any envelope-domain wetness measure, and not in a consistent
# direction. Sweeping blend 0 -> 0.45 on four stems:
#
#   reverb_tail_ratio  non-monotone on 3 of 4 (adding reverb LOWERED it on
#                      Al James, 0.118 -> 0.104, because reverb fills the
#                      inter-word gaps and lifts the trailing peak that the
#                      decay test measures against)
#   gap-only ratio     non-monotone on 4 of 4
#   envelope range     non-monotone on 4 of 4
#   envelope autocorr  monotone on 4 of 4, but the SIGN flips per track
#                      (rises on Timboz, falls on Speak Softly), so it cannot
#                      tell the loop which way to move
#
# A usable control signal needs a measure that is both sensitive and
# consistently signed on already-reverberant material — likely a modulation-
# spectrum feature rather than anything computed from the amplitude envelope.
# Until then reverb stays open-loop on the estimator in reverb_analysis.
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Joint refinement over every scored dimension
# ---------------------------------------------------------------------------
#
# Typical distance between two UNRELATED commercial vocal stems, over 120 pairs
# from 16 tracks. This puts dimensions on one scale: 1.0 means "as wrong as a
# random other vocal", 0.0 means identical. Without a common scale there is no
# principled way to weigh a dB of crest against a point of width, and the loop
# cannot judge whether a trade is worth taking.
_NORM = {
    "bands32": 6.589, "width_bands": 5.509, "mod_spec": 2.708, "ride_q": 2.481,
    "crest_db": 2.208, "lra_db": 5.604, "spread_db": 6.048, "floor_rel_db": 8.162,
    "air_db": 5.415, "sibilance_db": 11.703, "side_pct": 11.468, "lr_corr": 0.107,
    "centroid_hz": 0.283, "rolloff95_hz": 1.125, "attack_db_per_frame": 0.461,
    "onset_rate_hz": 1.543,
}
_ALL_KEYS = tuple(_NORM.keys())

# A finer spectral basis, offered as a candidate rather than used outright. On
# its own it more than halves the spectral residual but costs more in dynamics
# than it gains in tone; inside this loop it is taken only when the whole score
# improves.
_FINE_EDGES = np.geomspace(60.0, 16000.0, 25)
_FINE_Q = 1.0 / (float(_FINE_EDGES[1] / _FINE_EDGES[0])
                 - float(_FINE_EDGES[0] / _FINE_EDGES[1]))


def _composite(d, keys=_ALL_KEYS) -> float:
    return float(np.mean([d[k] / _NORM[k] for k in keys if k in d and _NORM.get(k)]))


def refine_match(y, sr, ref_mono, max_rounds: int = 16, score_seconds: float = 20.0,
                 ref_dyn=None, time_budget_s: float = 45.0):
    """
    Greedy coordinate descent over tone AND dynamics, scored on exactly the
    measurements the match report uses.

    Three properties, each answering a way the one-shot stages failed:

    * It optimises the SCORED definition of each property. The dynamics stage
      matched loudness range over a 25 dB window on 50 ms frames while the score
      measured a 45 dB window on 2048-sample frames - so it could converge on
      its own terms and move the score not at all.
    * A candidate is kept only when the composite over ALL dimensions improves,
      so no stage can buy its own metric by wrecking another. Every earlier
      single-dimension correction here did exactly that: the spectral loop cost
      crest, the crest loop cost everything, the finer basis cost dynamics.
    * Nothing holds a target of its own - every target is measured off this
      reference, on this job.
    """
    from processor.dsp.similarity import measure, distances
    from processor.dsp.compressor import CompressorSettings, apply_compressor
    from processor.dsp.transient_shaper import transient_shaper
    from processor.dsp.dynamics_transfer import match_dynamics
    from processor.dsp.eq import EqBand, apply_eq

    ref_mono = np.asarray(ref_mono, dtype=np.float64)
    if ref_mono.ndim > 1:
        ref_mono = ref_mono.mean(axis=-1)

    # Scoring on a bounded excerpt keeps this affordable on long vocals; every
    # property here is distributional, so a representative slice suffices.
    n_score = int(score_seconds * sr)

    def clip_for_score(sig):
        if len(sig) <= n_score:
            return sig
        start = max(0, (len(sig) - n_score) // 2)
        return sig[start:start + n_score]

    ref_f = measure(clip_for_score(ref_mono), sr)
    # Dynamics may be scored against a different signal from tone: the centre
    # mask that isolates the lead also reshapes the envelope, so the lead
    # represents tone better and the whole vocal represents dynamics better.
    if ref_dyn is None:
        ref_dyn_f = ref_f
    else:
        rd = np.asarray(ref_dyn, dtype=np.float64)
        if rd.ndim > 1:
            rd = rd.mean(axis=-1)
        ref_dyn_f = measure(clip_for_score(rd), sr)
    _DYN = ("crest_db", "lra_db", "spread_db", "ride_q",
            "attack_db_per_frame", "mod_spec", "onset_rate_hz", "floor_rel_db")
    ref_dyn_src = ref_mono if ref_dyn is None else np.asarray(ref_dyn, dtype=np.float64)
    if ref_dyn_src.ndim > 1:
        ref_dyn_src = ref_dyn_src.mean(axis=-1)

    def score(sig):
        f = measure(clip_for_score(sig), sr)
        d_tone = distances(f, ref_f)
        d_dyn = distances(f, ref_dyn_f)
        d = {k: (d_dyn[k] if k in _DYN else d_tone[k]) for k in d_tone}
        return _composite(d)

    def stereo_op(sig, fn):
        if sig.ndim == 2:
            return np.stack([fn(sig[:, c]) for c in range(sig.shape[1])], axis=1)
        return fn(sig)

    def eq_op(sig, bands):
        return stereo_op(sig, lambda c: apply_eq(c, sr, bands))

    # Search on a short excerpt, then replay the winning moves ONCE at full
    # length. Previously every candidate was rendered over the whole vocal and
    # only scored on an excerpt, so a 3-minute take meant ~125 full-length
    # renders and the stage effectively hung. The moves are chosen from
    # distributional properties, so a representative slice picks the same ones
    # at a fraction of the cost.
    search_y = y
    if len(y) > n_score:
        st = max(0, (len(y) - n_score) // 2)
        search_y = y[st:st + n_score]

    best = search_y
    best_score = score(best)
    start_score = best_score
    log = []
    chosen = []          # (name, fn) replayed on the full-length signal
    t0 = time.monotonic()
    # How many times each family of move may be used.
    #
    # Without this the search stacks the same stage over and over, because
    # doing so keeps nudging the score: observed runs applied ride at full
    # strength FOUR times, piled three +2 dB shelves onto 6.5 kHz, and put
    # three compressors in series. The measurements improved and the audio did
    # not — the output picked up a 14.9 dB gain ride and 22 frames pushed more
    # than 45 dB down, heard as pumping and words fading out. A stage may now
    # be used once (twice for the spectral match, which is genuinely iterative
    # and gentle), and the whole search is capped at MAX_MOVES.
    used = {}
    LIMITS = {"spec": 2}
    DEFAULT_LIMIT = 1
    MAX_MOVES = 5

    def family(name):
        for f in ("spec", "ride", "comp", "trans", "mod", "deess", "air", "sib"):
            if name.startswith(f):
                return f
        return name

    for _ in range(max(1, max_rounds)):
        if len(log) >= MAX_MOVES or time.monotonic() - t0 > time_budget_s:
            break
        peak_db = 20.0 * np.log10(float(np.max(np.abs(best))) + 1e-12)
        cands = [
            ("spec12", lambda s: match_spectrum(s, sr, ref_mono)[0]),
            ("spec24", lambda s: match_spectrum(s, sr, ref_mono, edges=_FINE_EDGES,
                                                q=_FINE_Q, damping=0.5)[0]),
        ]
        for st in (0.4, 0.7, 1.0):
            cands.append(("ride%.1f" % st,
                          lambda s, st=st: match_dynamics(s, sr, ref_mono, strength=st)))
        for off in (4.0, 8.0, 14.0, 20.0):
            cfg = CompressorSettings(threshold_db=peak_db - off, ratio=3.0,
                                     attack_ms=8.0, release_ms=140.0, makeup_db=0.0)
            cands.append(("comp-%.0f" % off,
                          lambda s, cfg=cfg: stereo_op(s, lambda c: apply_compressor(c, sr, cfg))))
        for amt in (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8):
            cands.append(("trans%+.1f" % amt,
                          lambda s, a=amt: stereo_op(s, lambda c: transient_shaper(c, sr, amount=a))))
        # A real de-esser, aimed at the reference's own sibilance level. The
        # static bell below can only tilt the whole band; this rides only the
        # frames where sibilance actually occurs, which is what the score
        # measures (brightest-decile frames).
        try:
            _sib_target = measure_sibilance_db(ref_mono, sr)
            for thr in (-30.0, -24.0, -18.0):
                cands.append(("deess%.0f" % thr,
                              lambda s, t=thr: stereo_op(
                                  s, lambda c: apply_deesser(c, sr, thresh_db=t, ratio=3.0,
                                                             ref_sibilance_db=_sib_target))))
        except Exception:
            pass
        # Wider compression search: depth was reaching only about half its
        # ceiling with a single ratio.
        for ratio in (2.0, 6.0):
            for off in (6.0, 12.0):
                cfg2 = CompressorSettings(threshold_db=peak_db - off, ratio=ratio,
                                          attack_ms=15.0, release_ms=200.0, makeup_db=0.0)
                cands.append(("comp%.0f:1-%.0f" % (ratio, off),
                              lambda s, cfg=cfg2: stereo_op(s, lambda c: apply_compressor(c, sr, cfg))))
        for mg in (2.5, 5.0):
            cands.append(("mod%.1f" % mg,
                          lambda s, mg=mg: match_modulation(s, sr, ref_dyn_src, max_gain_db=mg)))
        for g in (-2.0, 2.0):
            cands.append(("air%+.0f" % g,
                          lambda s, g=g: eq_op(s, [EqBand(f=11000.0, gain_db=g, q=0.7)])))
            cands.append(("sib%+.0f" % g,
                          lambda s, g=g: eq_op(s, [EqBand(f=6500.0, gain_db=g, q=1.4)])))

        # Best improvement, not first. Taking the first candidate that helped
        # meant the loop was biased by list order: the spectral moves are
        # listed first and almost always help a little, so it took one,
        # re-planned, and exited before a transient or modulation move was
        # ever evaluated. Those two dimensions sat at exactly 0 while the
        # search never reached them.
        best_name, best_cand, best_new, best_fn = None, None, best_score, None
        for name, fn in cands:
            if used.get(family(name), 0) >= LIMITS.get(family(name), DEFAULT_LIMIT):
                continue
            try:
                cand = np.nan_to_num(fn(best), nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                continue
            if not np.isfinite(cand).all() or float(np.max(np.abs(cand))) < 1e-6:
                continue
            sc = score(cand)
            if sc < best_new - 1e-4:
                best_name, best_cand, best_new, best_fn = name, cand, sc, fn
        if best_cand is None:
            break
        best, best_score = best_cand, best_new
        used[family(best_name)] = used.get(family(best_name), 0) + 1
        chosen.append((best_name, best_fn))
        log.append("%s->%.4f" % (best_name, best_new))

    out = best
    if search_y is not y and chosen:
        out = y
        for _nm, _fn in chosen:
            try:
                out = np.nan_to_num(_fn(out), nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
    return out, {"start": round(start_score, 4), "final": round(best_score, 4),
                 "applied": log, "searched_on_excerpt": search_y is not y}


def _env_db_for_mod(mono, sr, hop=512):
    rms = librosa.feature.rms(y=np.ascontiguousarray(mono), frame_length=2048,
                              hop_length=hop)[0]
    return 20.0 * np.log10(np.maximum(rms, 1e-9)), hop


_MOD_EDGES = np.geomspace(0.5, 32.0, 9)


def match_modulation(y, sr, ref_mono, max_gain_db: float = 5.0, smooth_ms: float = 120.0):
    """
    Match how the loudness envelope FLUCTUATES, not just where it sits.

    Compression, riding and reverb all show up as energy at particular
    envelope-modulation rates - roughly 0.5-32 Hz - and that pattern is what a
    listener reads as "pumping" or "movement". Nothing else in the chain
    touches it: level matching sets the envelope's position, dynamics transfer
    sets its distribution, and neither controls its rate content, which is why
    that dimension sat at zero while everything around it moved.

    The envelope is taken to the modulation domain, each band's magnitude is
    scaled toward the reference's band profile with the phase left alone (the
    two are different performances, so only the profile transfers, never the
    timing), and the difference is applied back as a smoothed gain ride.
    """
    mono = y if y.ndim == 1 else y.mean(axis=1)
    ref_mono = np.asarray(ref_mono, dtype=np.float64)
    if ref_mono.ndim > 1:
        ref_mono = ref_mono.mean(axis=-1)

    eo, hop = _env_db_for_mod(mono, sr)
    er, _ = _env_db_for_mod(ref_mono, sr)
    if len(eo) < 32 or len(er) < 32:
        return y

    ao = eo - float(np.mean(eo))
    ar = er - float(np.mean(er))
    fr = sr / hop

    def band_profile(a):
        spec = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
        f = np.fft.rfftfreq(len(a), d=1.0 / fr)
        out = np.empty(len(_MOD_EDGES) - 1)
        for i in range(len(out)):
            m = (f >= _MOD_EDGES[i]) & (f < _MOD_EDGES[i + 1])
            out[i] = 10 * np.log10(float(np.sum(spec[m])) + 1e-20) if m.any() else -200.0
        ok = out > -190
        if ok.any():
            out = out - float(np.mean(out[ok]))
        return out

    want, have = band_profile(ar), band_profile(ao)
    gap = np.clip(want - have, -max_gain_db * 2, max_gain_db * 2)

    F = np.fft.rfft(ao)
    f = np.fft.rfftfreq(len(ao), d=1.0 / fr)
    scale = np.ones(len(F))
    for i in range(len(gap)):
        m = (f >= _MOD_EDGES[i]) & (f < _MOD_EDGES[i + 1])
        if m.any():
            scale[m] = 10 ** (gap[i] / 20.0)   # amplitude, gap is a power ratio in dB
    corrected = np.fft.irfft(F * scale, n=len(ao))

    ride = np.clip(corrected - ao, -max_gain_db, max_gain_db)
    k = max(1, int(smooth_ms / (hop / sr * 1000.0)))
    if k > 1:
        ride = np.convolve(ride, np.ones(k) / k, mode="same")

    pos = (np.arange(len(ride)) + 0.5) * hop
    g = 10 ** (np.interp(np.arange(len(mono)), pos, ride) / 20.0)
    out = np.asarray(y, dtype=np.float64) * (g[:, None] if y.ndim == 2 else g)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Not implemented on purpose: parameter-space joint search.
#
# refine_match is greedy in SIGNAL space — it picks a move, applies it, and
# re-plans from the result — which looked like the reason the score plateaued:
# path dependence, where an early spectral move is taken first and every later
# move then works on the EQ'd signal and cannot undo it.
#
# A parameter-space version was built and measured, and it is WORSE: coordinate
# descent over one parameter per stage, re-rendering from the same input each
# evaluation, scored 1.231 against greedy's 1.151 on the same six pairs, at
# roughly four times the cost. The reason is the thing the redesign threw away:
# greedy can apply the SAME stage repeatedly, and compounding two or three
# spectral passes is worth more than being able to revisit an earlier choice.
#
# Nor is the search stopping early. Raising max_rounds 8 -> 16 -> 30 gives
# 1.151 -> 1.138 -> 1.138, converging at 8 accepted moves. The plateau is a
# genuine local optimum of this candidate set, not a search limitation, so
# more search effort is not where the remaining gain is.
# ---------------------------------------------------------------------------
