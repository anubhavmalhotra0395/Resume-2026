"""The tone matcher must not amplify the top octave.

Reported by ear as "crackle / distortion / fizz". A vocal carries almost
nothing above ~12 kHz; what lives there is codec and separation artefacts, and
the reference and the dry vocal have DIFFERENT artefacts. Matching that
difference boosts our own noise -- measured at +8.1 dB above 16 kHz on the
Hide pair, audible, and invisible to all sixteen similarity dimensions.
"""
import numpy as np

from processor.dsp.eq import (_limit_hf_boost, _HF_TRUST_HZ,
                              _HF_BOOST_CEILING_DB, design_eq_from_mel_diff)


def test_boosts_below_the_trust_frequency_are_untouched():
    for f in (100.0, 1000.0, 8000.0, _HF_TRUST_HZ):
        assert _limit_hf_boost(f, 6.0) == 6.0


def test_boosts_above_it_are_tapered():
    """Monotone while the taper is running, then flat at the ceiling. Do not
    assert 20 kHz is strictly quieter than 16 kHz: both are clamped by then,
    and that assertion tests the taper's shape rather than the requirement."""
    assert _limit_hf_boost(13000.0, 6.0) < 6.0
    assert _limit_hf_boost(16000.0, 6.0) <= _limit_hf_boost(13000.0, 6.0)
    assert _limit_hf_boost(20000.0, 6.0) <= _limit_hf_boost(16000.0, 6.0)
    assert _limit_hf_boost(16000.0, 6.0) <= _HF_BOOST_CEILING_DB + 1e-6


def test_the_ceiling_is_never_exceeded_at_the_top():
    for g in (3.0, 6.0, 24.0):
        assert _limit_hf_boost(21000.0, g) <= _HF_BOOST_CEILING_DB + 1e-6


def test_cuts_are_never_restrained():
    """Removing noise up there is always safe -- only boosting is limited."""
    for f in (12000.0, 16000.0, 20000.0):
        assert _limit_hf_boost(f, -6.0) == -6.0
        assert _limit_hf_boost(f, -12.0) == -12.0


def test_designed_eq_never_boosts_the_top_octave_hard():
    """End to end: a reference with far more top-octave energy than the dry
    must not produce a large boost there."""
    sr = 44100
    n = 48
    freqs = np.geomspace(60.0, 20000.0, n)
    dry = np.full(n, 1e-4)
    ref = dry.copy()
    ref[freqs > 15000] *= 400.0            # a big artefact-driven difference
    bands = design_eq_from_mel_diff(ref, dry, freqs, sr, max_gain_db=6.0)
    top = [b for b in bands if b.f > 15000 and b.gain_db > 0]
    assert all(b.gain_db <= _HF_BOOST_CEILING_DB + 1e-6 for b in top), \
        [(b.f, b.gain_db) for b in top]
