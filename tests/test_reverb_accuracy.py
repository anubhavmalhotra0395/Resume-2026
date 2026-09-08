"""
Ground-truth guards for the reverb path and the detectors that feed it.

Every assertion here corresponds to a defect that shipped:

  * _build_ir applied a decay envelope on top of an already-decaying comb
    bank. Decays multiply, so their dB slopes add, and every IR came out
    2-5x shorter than requested (1.2 s -> 0.38 s).
  * estimate_reverb_params fitted RT60 to a mid-phrase slice, where there is
    no tail to fit. It returned ~1.77 s for a bone-dry signal and did not
    move when reverb was added.
  * wet was 0.08 + (rt60/2.5)*0.12 — a function of RT60, never a measurement,
    landing near 0.14 for every reference.
  * detect_tape fired on 6 of 6 untreated commercial vocal stems, with drive
    pinned at 0.8, so every job got near-maximum tape saturation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from processor.dsp.conv_reverb import ConvReverbSettings, _build_ir, apply_conv_reverb
from processor.dsp.analysis.reverb_analysis import estimate_reverb_params, reverb_tail_ratio
from processor.dsp.analysis.tape_analysis import detect_tape

SR = 44100


def _dry_vocal(seconds=12.0, seed=7):
    """Harmonic notes with clean releases and real silence between them.
    Dry by construction, so any measured tail comes from what we added."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    out = np.zeros(n)
    pos = 0.0
    while pos < seconds - 1.0:
        dur = float(rng.uniform(0.28, 0.75))
        f0 = float(rng.choice([220., 247., 262., 294., 330., 349., 392., 440.]))
        i0, i1 = int(pos * SR), min(n, int((pos + dur) * SR))
        ln = i1 - i0
        if ln < 64:
            break
        t = np.arange(ln) / SR
        sig = sum(np.sin(2 * np.pi * f0 * h * t) / h ** 1.25 for h in range(1, 22))
        sig /= np.max(np.abs(sig)) + 1e-9
        env = np.ones(ln)
        a, r = int(0.02 * SR), int(0.10 * SR)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] *= np.linspace(1, 0, r) ** 1.6
        out[i0:i1] += sig * env * float(rng.uniform(0.55, 1.0))
        pos += dur + float(rng.uniform(0.12, 0.45))
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.7).astype(np.float32)


def _measure_rt60(ir):
    """Schroeder backward integration, fitted over -5..-25 dB."""
    e = np.asarray(ir, dtype=np.float64) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edb = 10 * np.log10(edc / (edc[0] + 1e-20) + 1e-20)
    i1 = np.where(edb <= -5)[0][0]
    i2 = np.where(edb <= -25)[0][0]
    x = np.arange(i1, i2) / SR
    slope, _ = np.polyfit(x, edb[i1:i2], 1)
    return 60.0 / abs(slope)


def test_ir_realises_requested_rt60():
    """The IR must actually decay at the rate it was asked for."""
    for rt60 in (0.3, 0.6, 1.2, 1.8, 2.4):
        measured = _measure_rt60(_build_ir(SR, rt60, 20.0, "plate"))
        assert abs(measured - rt60) < 0.20 * rt60 + 0.08, (
            f"IR for rt60={rt60}s measured {measured:.2f}s "
            f"(the old compounding-decay bug gave ~{rt60 / 3:.2f}s)"
        )


def test_reverb_detector_responds_to_rt60():
    """Detected RT60 must rise with real added reverb, not sit at a constant."""
    dry = _dry_vocal()
    short = apply_conv_reverb(dry, SR, ConvReverbSettings("plate", rt60=0.4, wet=0.25, pre_delay_ms=20))
    long_ = apply_conv_reverb(dry, SR, ConvReverbSettings("plate", rt60=2.4, wet=0.25, pre_delay_ms=20))
    to_mono = lambda a: a if a.ndim == 1 else a.mean(1)
    rt_short = estimate_reverb_params(to_mono(short), SR).rt60
    rt_long = estimate_reverb_params(to_mono(long_), SR).rt60
    assert rt_long > rt_short + 0.4, (
        f"detector reported {rt_short:.2f}s for a 0.4s tail and {rt_long:.2f}s "
        f"for a 2.4s tail — it is not tracking reverb"
    )


def test_wet_is_measured_not_constant():
    """Wet must be read off the reference, not derived from RT60."""
    dry = _dry_vocal()
    to_mono = lambda a: a if a.ndim == 1 else a.mean(1)
    wets = [
        estimate_reverb_params(
            to_mono(apply_conv_reverb(dry, SR, ConvReverbSettings("plate", rt60=1.2, wet=w, pre_delay_ms=20))), SR
        ).wet
        for w in (0.10, 0.35)
    ]
    # The old formula returned an identical wet for both (it never looked at
    # the audio), so any strict increase is the property under test.
    assert wets[1] > wets[0] + 0.02, (
        f"wet={wets[0]:.2f} for a 0.10 mix and {wets[1]:.2f} for a 0.35 mix — "
        f"not responding to the reference"
    )
    # And a dry reference must not be called wet.
    assert estimate_reverb_params(dry, SR).wet < 0.15


def test_tail_ratio_tracks_wetness():
    """The tail measurement backing wet must separate dry from washed-out."""
    dry = _dry_vocal()
    to_mono = lambda a: a if a.ndim == 1 else a.mean(1)
    washed = to_mono(apply_conv_reverb(dry, SR, ConvReverbSettings("plate", rt60=1.6, wet=0.35, pre_delay_ms=20)))
    assert reverb_tail_ratio(washed, SR) > reverb_tail_ratio(dry, SR) + 0.01


def test_dry_baseline_is_independent_of_phrasing():
    """
    The dry reading must not depend on how densely the singer articulates.

    An earlier version counted any frame >120 ms after an onset as tail, so a
    sparsely-phrased dry take read 0.73 against a densely-phrased dry take's
    0.60 — the 'wetness' of a dry vocal moved with its phrasing.
    """
    sparse_src = _dry_vocal(seed=7)
    dense_src = _dry_vocal(seconds=12.0, seed=23)
    a, b = reverb_tail_ratio(sparse_src, SR), reverb_tail_ratio(dense_src, SR)
    assert abs(a - b) < 0.03, f"dry baselines diverge by phrasing: {a:.3f} vs {b:.3f}"


def test_tape_does_not_fire_on_a_plain_vocal():
    """A natural, untreated vocal must not be reported as tape-saturated.

    Vocals are naturally dark above 8 kHz; the old rolloff gate read that as
    tape on every reference and applied drive=0.8 to all of them.
    """
    assert detect_tape(_dry_vocal(), SR) is None
