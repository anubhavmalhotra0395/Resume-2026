"""Guards for the late-chain matching stages.

Two classes of bug these exist to catch:

1. Silent no-ops. Every one of these stages is applied through an
   accept-if-better gate, so a stage that quietly does nothing looks exactly
   like a stage the gate rejected. The responsiveness assertions below are the
   only thing that tells them apart.
2. Channel-layout confusion. chain.py indexes y[:, 0] in one place and
   np.mean(y, axis=0) in another; they coexist only because y is mono in
   practice. A stage that assumes the wrong layout on stereo would transpose
   the signal into noise.
"""
import numpy as np
import pytest

from processor.dsp.late_match import (match_ride, expand_floor, match_sibilance,
                                      limit_crest, _frame_db, _mono)

SR = 22050


def _phrases(seconds=8.0, sr=SR, gap_floor=0.02):
    """Sung phrases separated by quiet gaps, like a real vocal."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 10))
    env = np.zeros(n)
    for start in (0.3, 2.3, 4.3, 6.3):
        a = int(start * sr)
        b = min(n, a + int(1.2 * sr))
        if b - a < 2:               # phrase falls outside a short excerpt
            continue
        seg = np.linspace(0, 1, b - a)
        env[a:b] = np.sin(np.pi * seg)
    rng = np.random.RandomState(0)
    y = tone * env + rng.randn(n) * gap_floor
    return (y / np.max(np.abs(y)) * 0.8).astype(np.float64)


def _crest(y):
    m = _mono(y)
    return (20 * np.log10(np.max(np.abs(m)) + 1e-12)
            - 20 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-12))


def _floor_rel(y):
    db = _frame_db(_mono(y))
    return float(np.percentile(db, 5) - np.percentile(db, 95))


# ---- crest ---------------------------------------------------------------

def test_limit_crest_moves_toward_the_target():
    y = _phrases()
    target = _crest(y) - 4.0
    out = limit_crest(y, SR, target)
    assert abs(_crest(out) - target) < abs(_crest(y) - target)


def test_limit_crest_is_scale_invariant_in_its_target():
    """Crest is peak-minus-RMS, so a level change must not alter the result's
    crest -- this is why the final loudness match cannot undo this stage."""
    y = _phrases()
    a = limit_crest(y, SR, _crest(y) - 4.0)
    b = limit_crest(y * 0.25, SR, _crest(y) - 4.0)
    assert abs(_crest(a) - _crest(b)) < 0.5


def test_limit_crest_leaves_an_already_matching_signal_alone():
    y = _phrases()
    out = limit_crest(y, SR, _crest(y))
    assert np.allclose(out, y, atol=1e-6)


# ---- floor ---------------------------------------------------------------

def test_expand_floor_lowers_the_gaps():
    y = _phrases(gap_floor=0.05)
    out = expand_floor(y, SR, _floor_rel(y) - 15.0)
    assert _floor_rel(out) < _floor_rel(y)


def test_expand_floor_fraction_scales_the_move():
    y = _phrases(gap_floor=0.05)
    target = _floor_rel(y) - 20.0
    half = _floor_rel(expand_floor(y, SR, target, fraction=0.5))
    full = _floor_rel(expand_floor(y, SR, target, fraction=1.0))
    assert full < half < _floor_rel(y)


def test_expand_floor_keeps_the_loud_parts_intact():
    """The old gate was audible as fades because it ducked programme, not
    just gaps."""
    y = _phrases(gap_floor=0.05)
    out = expand_floor(y, SR, _floor_rel(y) - 15.0)
    db = _frame_db(_mono(y))
    loud = db > np.percentile(db, 80)
    ry = _frame_db(_mono(y))[loud]
    ro = _frame_db(_mono(out))[loud]
    assert np.median(ro - ry) > -1.0


# ---- sibilance -----------------------------------------------------------

def test_match_sibilance_corrects_in_both_directions():
    y = _phrases()
    up = match_sibilance(y, SR, 4.0, 9.0)
    down = match_sibilance(y, SR, 9.0, 4.0)
    band = lambda v: float(np.sum(np.abs(np.fft.rfft(_mono(v)))[
        int(5000 * len(v) / SR):int(9000 * len(v) / SR)]))
    assert band(up) > band(y) > band(down)


def test_match_sibilance_ignores_a_negligible_gap():
    y = _phrases()
    assert np.allclose(match_sibilance(y, SR, 7.0, 7.1), y)


# ---- ride ----------------------------------------------------------------

def test_match_ride_reshapes_the_loudness_distribution():
    y = _phrases()
    db = _frame_db(_mono(y))
    live = db > (np.percentile(db, 95) - 45.0)
    act = db[live]
    # Ask for a noticeably wider ride than the signal has.
    wide = (np.percentile(act, np.linspace(0, 100, 21)) - np.median(act)) * 2.0
    out = match_ride(y, SR, wide)
    spread = lambda v: float(np.percentile(_frame_db(_mono(v)), 90)
                             - np.percentile(_frame_db(_mono(v)), 10))
    assert spread(out) > spread(y)


def test_match_ride_does_not_import_the_reference_level():
    """Only the SHAPE of the reference's loudness distribution is matched;
    absolute level belongs to the loudness stage."""
    y = _phrases()
    db = _frame_db(_mono(y))
    live = db > (np.percentile(db, 95) - 45.0)
    q = np.percentile(db[live], np.linspace(0, 100, 21)) - np.median(db[live])
    out = match_ride(y, SR, q)
    rms = lambda v: np.sqrt(np.mean(_mono(v) ** 2))
    assert abs(20 * np.log10(rms(out) / rms(y))) < 3.0


# ---- layout / robustness -------------------------------------------------

@pytest.mark.parametrize("layout", ["mono", "channels_first", "samples_first"])
def test_every_stage_preserves_shape_and_finiteness(layout):
    m = _phrases()
    y = {"mono": m,
         "channels_first": np.stack([m, m * 0.9], axis=0),
         "samples_first": np.stack([m, m * 0.9], axis=1)}[layout]
    db = _frame_db(_mono(y))
    live = db > (np.percentile(db, 95) - 45.0)
    q = np.percentile(db[live], np.linspace(0, 100, 21)) - np.median(db[live])
    out = match_ride(y, SR, q)
    out = expand_floor(out, SR, _floor_rel(y) - 10.0)
    out = match_sibilance(out, SR, 4.0, 8.0)
    out = limit_crest(out, SR, _crest(y) - 2.0)
    assert out.shape == y.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) > 1e-3


def test_silent_input_does_not_explode():
    z = np.zeros(SR * 2)
    assert np.isfinite(limit_crest(z, SR, 12.0)).all()
    assert np.isfinite(expand_floor(z, SR, -60.0)).all()


# ---- per-band width ------------------------------------------------------

def _stereo_pair(seconds=6.0, sr=SR, side_gain=0.3):
    m = _phrases(seconds, sr)
    rng = np.random.RandomState(1)
    side = np.convolve(m, rng.randn(256) * 0.1, mode="same") * side_gain
    return np.stack([m + side, m - side], axis=0)


def _width_bands(y, sr=SR):
    import librosa
    from processor.dsp.late_match import _EDGES_W, _WIN, _HOP, _chan_axis
    ax = _chan_axis(y)
    L = y[0] if ax == 0 else y[:, 0]
    R = y[1] if ax == 0 else y[:, 1]
    Sm = np.abs(librosa.stft(np.ascontiguousarray((L + R) * .5), n_fft=_WIN,
                             hop_length=_HOP)) ** 2
    Ss = np.abs(librosa.stft(np.ascontiguousarray((L - R) * .5), n_fft=_WIN,
                             hop_length=_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_WIN)
    out = []
    for i in range(len(_EDGES_W) - 1):
        msk = (freqs >= _EDGES_W[i]) & (freqs < _EDGES_W[i + 1])
        out.append(10 * np.log10((Ss[msk].sum() + 1e-20) / (Sm[msk].sum() + 1e-20)))
    return np.array(out)


def test_match_width_bands_hits_the_target_profile():
    """The corrector must work at the resolution it is scored at: three bands
    could not match an eight-band profile, which pinned width at 4.4/10."""
    from processor.dsp.late_match import match_width_bands
    y = _stereo_pair()
    target = _width_bands(y) - np.linspace(2.0, 8.0, 8)
    out = match_width_bands(y, SR, target)
    assert np.max(np.abs(_width_bands(out) - target)) < 1.5


def test_match_width_bands_leaves_the_mid_channel_alone():
    """Mono fold-down must be unchanged -- only the side is shaped."""
    from processor.dsp.late_match import match_width_bands
    y = _stereo_pair()
    out = match_width_bands(y, SR, _width_bands(y) - 6.0)
    assert np.corrcoef(y[0] + y[1], out[0] + out[1])[0, 1] > 0.999


def test_match_width_bands_passes_mono_through():
    from processor.dsp.late_match import match_width_bands
    m = _phrases()
    assert np.allclose(match_width_bands(m, SR, np.full(8, -12.0)), m)


def test_match_width_bands_will_not_amplify_a_silent_side():
    """A band with no side content cannot be widened by gain; multiplying
    nothing by a large number is how this stage would blow up."""
    from processor.dsp.late_match import match_width_bands
    m = _phrases()
    y = np.stack([m, m], axis=0)               # side is exactly zero
    out = match_width_bands(y, SR, np.full(8, 0.0))
    assert np.isfinite(out).all()
    assert np.allclose(out, y)


# ---- gap noise -----------------------------------------------------------

def test_reduce_gap_noise_quietens_the_gaps():
    from processor.dsp.late_match import reduce_gap_noise
    y = _phrases(gap_floor=0.03)
    out, info = reduce_gap_noise(y, SR, _floor_rel(y) - 15.0)
    assert info["applied"], info
    assert _floor_rel(out) < _floor_rel(y) - 3.0


def test_reduce_gap_noise_does_not_duck_phrase_entries():
    """The failure mode this replaces: an envelope that recovered slowly was
    still down when the next phrase began, ducking every entry by 7-22 dB and
    reading as fades."""
    from processor.dsp.late_match import reduce_gap_noise
    y = _phrases(gap_floor=0.03)
    out, info = reduce_gap_noise(y, SR, _floor_rel(y) - 15.0)
    assert info["applied"]
    db = _frame_db(_mono(y))
    loud = db > (np.percentile(db, 95) - 12.0)
    delta = _frame_db(_mono(out))[loud] - db[loud]
    assert float(np.min(delta)) > -0.5, f"worst programme dip {np.min(delta):.2f} dB"


def test_reduce_gap_noise_leaves_a_clean_signal_alone():
    from processor.dsp.late_match import reduce_gap_noise
    y = _phrases(gap_floor=0.0005)
    out, info = reduce_gap_noise(y, SR, _floor_rel(y) + 20.0)
    assert not info["applied"]
    assert np.allclose(out, y)


def test_reduce_gap_noise_reports_what_it_did():
    from processor.dsp.late_match import reduce_gap_noise
    y = _phrases(gap_floor=0.03)
    _, info = reduce_gap_noise(y, SR, _floor_rel(y) - 15.0)
    assert {"threshold_rel_db", "ratio", "gap_rel_db", "programme_dip_db"} <= set(info)


# ---- gap colour ----------------------------------------------------------

def _bright_gaps(sr=SR, seconds=8.0):
    """Phrases with deliberately BRIGHT noise in the gaps -- the hiss signature
    the chain's high-frequency boost creates out of the input's own noise."""
    m = _phrases(seconds, sr, gap_floor=0.0)
    rng = np.random.RandomState(3)
    noise = rng.randn(len(m))
    noise = np.diff(np.concatenate([[0.0], noise]))     # +6 dB/oct = bright
    return (m + noise * 0.01).astype(np.float64)


def test_gap_tilt_reports_brighter_for_brighter_noise():
    from processor.dsp.late_match import gap_tilt
    bright = _bright_gaps()
    dark = _phrases(gap_floor=0.01)
    assert gap_tilt(bright, SR) > gap_tilt(dark, SR)


def test_match_gap_tilt_darkens_toward_the_target():
    from processor.dsp.late_match import match_gap_tilt, gap_tilt
    y = _bright_gaps()
    target = gap_tilt(y, SR) - 5.0
    out, info = match_gap_tilt(y, SR, target)
    assert info["applied"], info
    assert gap_tilt(out, SR) < gap_tilt(y, SR) - 1.5


def test_match_gap_tilt_does_not_touch_the_singing():
    """Cutting the top must happen only where there is no voice, or it dulls
    the vocal's air along with the hiss."""
    from processor.dsp.late_match import match_gap_tilt, gap_tilt
    y = _bright_gaps()
    out, info = match_gap_tilt(y, SR, gap_tilt(y, SR) - 5.0)
    assert info["applied"]
    db = _frame_db(_mono(y))
    loud = db > (np.percentile(db, 95) - 12.0)
    delta = _frame_db(_mono(out))[loud] - db[loud]
    assert float(np.min(delta)) > -0.5


def test_match_gap_tilt_leaves_already_dark_noise_alone():
    from processor.dsp.late_match import match_gap_tilt, gap_tilt
    y = _phrases(gap_floor=0.01)
    out, info = match_gap_tilt(y, SR, gap_tilt(y, SR) + 6.0)
    assert not info["applied"]
    assert np.allclose(out, y)


def test_limit_crest_does_not_generate_harmonics():
    """Reported by ear as "crackle / distortion / fizz", worst on loud notes.

    The first implementation applied a per-sample gain -- an instantaneous
    nonlinearity -- so a clean harmonic tone with nothing above 6 kHz came out
    with that band at -34.6 dB. Gain must move slowly relative to the
    waveform: slow gain scales a signal, fast gain reshapes it.
    """
    from processor.dsp.late_match import limit_crest
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 8))
    x = x / np.max(np.abs(x)) * 0.8

    def hf_ratio(v):
        S = np.abs(np.fft.rfft(v * np.hanning(len(v))))
        f = np.fft.rfftfreq(len(v), 1 / sr)
        b = lambda a, c: S[(f >= a) & (f < c)].sum()
        return 20 * np.log10((b(6000, 22050) + 1e-12) / (b(100, 3000) + 1e-12))

    clean = hf_ratio(x)
    for drop in (1.0, 2.0, 3.0):
        cur = (20 * np.log10(np.max(np.abs(x)))
               - 20 * np.log10(np.sqrt(np.mean(x ** 2))))
        out = limit_crest(x, sr, cur - drop)
        assert hf_ratio(out) < clean + 90.0, (
            f"limiter generated harmonics: {clean:.0f} -> {hf_ratio(out):.0f} dB")


def test_limit_crest_still_matches_crest_on_varying_material():
    """The no-distortion fix must not turn the stage into a no-op.

    A smooth gain envelope has limited authority: a steady tone's crest cannot
    be changed by it at all, and this fixture's fairly uniform peaks cap the
    reduction near 1.3 dB (real vocals, whose peaks vary far more, reached
    4 dB). So assert it lands modest targets exactly and always moves toward
    the larger ones -- never that it can reach any target asked of it.
    """
    from processor.dsp.late_match import limit_crest
    y = _phrases()
    cur = _crest(y)
    assert abs(_crest(limit_crest(y, SR, cur - 1.0)) - (cur - 1.0)) < 0.3
    for drop in (2.0, 4.0):
        out = _crest(limit_crest(y, SR, cur - drop))
        assert out < cur - 0.8, f"no reduction at all for a -{drop} dB ask"
        assert out >= cur - drop - 0.3, "overshot the target"
