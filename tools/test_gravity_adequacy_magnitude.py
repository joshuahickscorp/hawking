"""Regression test for the campaign's most expensive scar: an adequacy check
scored by cosine alone is scale-invariant, so a candidate that keeps every
direction and destroys every magnitude (Wh = 0.01*W) scores 1.000000 on it.
That exact construction shipped once (see MEMORY.md: "the adequacy gate was
BLIND TO MAGNITUDE all campaign"). This test is what the campaign lacked --
it pins the 0.01*W negative control so a future edit cannot silently
reintroduce the blindness.

Findings this test encodes (2026-09-05 audit):

  tools/gravity_doctor_gate.py (the load-bearing Doctor gate -- imported by
  gravity_allocator, gravity_alignment, gravity_scale_choice, gravity_seal_artifact,
  gravity_endpoint_alloc, gravity_joint_scale, gravity_gate_calibration,
  gravity_tabula_probe, flash_gravity_doctor_cycle) is ALREADY magnitude-aware.
  Its `observed`/`probed`/`worst_unit` axes are scale-invariant by construction
  (proven below), but its `gain` axis (added for exactly this reason, per its
  own docstring) is not, and `gate()` rejects a 0.01x/100x scale on `gain`
  before the other three axes get a vote.

  tools/gravity_xform_hadamard.py's score() (used by gravity_planes_ladder.py)
  pairs `hold_cosine` with `rel_fro`, so it was already magnitude-aware too.

  tools/flash_doctor_bank_screen.py and tools/flash_doctor_ngram_screen.py
  (Doctor's Stage-A screens for Qwen3.8-Flash-Next, a named operator target)
  were NOT: their `sample_cosine` field, and the doctor_funnel early_rejection
  policy built on it, had no magnitude term. Fixed here by adding `sample_rel_fro`
  alongside the existing `sample_cosine` (metric added, not removed).

Mutation check performed by hand while writing this test (not left in source):
commenting out the `gn = ...` line inside gravity_doctor_gate.axes() and
reading `gain` back as the untouched 1.0 makes test_gate_rejects_scaled_weight
fail on the `worst_axis == "gain"` assertion, confirming the assertions are
load-bearing rather than vacuous.
"""
from __future__ import annotations

import numpy as np

from tools import flash_doctor_bank_screen, gravity_doctor_gate, gravity_xform_hadamard


def _toy(seed=0, d_out=64, d_in=128, n=96):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    X = rng.standard_normal((n, d_in)).astype(np.float32)
    return W, X


# --------------------------------------------------------------------------
# tools/gravity_doctor_gate.py -- already fixed; pin it so it stays fixed.
# --------------------------------------------------------------------------

def test_doctor_gate_original_three_axes_are_scale_invariant():
    """Document the scar directly: cosine-based axes cannot see magnitude."""
    W, X = _toy()
    for scale in (0.01, 100.0):
        a = gravity_doctor_gate.axes(W, (scale * W).astype(np.float32), X, seed=1)
        assert a["observed"] > 0.9999, a
        assert a["probed"] > 0.9999, a
        assert a["worst_unit"] > 0.9999, a


def test_doctor_gate_gain_axis_catches_the_same_scale():
    """The magnitude term the scar was missing: gain = min(r, 1/r) on the
    per-row/per-unit norm ratio, so both shrink and blow-up read low."""
    W, X = _toy()
    for scale in (0.01, 100.0):
        a = gravity_doctor_gate.axes(W, (scale * W).astype(np.float32), X, seed=1)
        assert a["gain"] < 0.05, f"scale={scale} gain={a['gain']} should be ~{min(scale, 1/scale):.4f}"


def test_doctor_gate_rejects_scaled_weight():
    """End-to-end: the actual pass/fail gate() call, against a real reference,
    must refuse a 0.01x-magnitude candidate and must name gain as the reason."""
    W, X = _toy()
    ref = gravity_doctor_gate.axes(
        W, gravity_doctor_gate.c_faithful_q4(W, group=32), X, seed=2)
    g = gravity_doctor_gate.gate(W, (0.01 * W).astype(np.float32), X, ref=ref, seed=2)
    assert g["healthy"] is False, g
    assert g["worst_axis"] == "gain", g


def test_doctor_gate_accepts_an_honest_codec():
    """Positive control: gate() must not reject faithful quantization outright
    (a test that only ever fails candidates proves nothing)."""
    W, X = _toy()
    ref = gravity_doctor_gate.axes(
        W, gravity_doctor_gate.c_faithful_q4(W, group=32), X, seed=3)
    g = gravity_doctor_gate.gate(W, gravity_doctor_gate.c_faithful_q4(W, group=32), X, ref=ref, seed=3)
    assert g["healthy"] is True, g


# --------------------------------------------------------------------------
# tools/gravity_xform_hadamard.py -- already fixed (cosine paired with rel_fro).
# --------------------------------------------------------------------------

def test_xform_hadamard_rel_fro_catches_scale_while_cosine_does_not():
    W, _ = _toy()
    Wh = (0.01 * W).astype(np.float32)
    assert gravity_xform_hadamard.cosine(W, Wh) > 0.9999
    assert gravity_xform_hadamard.rel_fro(W, Wh) > 0.9, "rel_fro must flag a 100x magnitude error"


# --------------------------------------------------------------------------
# tools/flash_doctor_bank_screen.py -- fixed by this change (added sample_rel_fro).
# --------------------------------------------------------------------------

def test_flash_bank_screen_cosine_alone_is_scale_invariant():
    """Same scar, reproduced on the raw formula the Stage-A screen's
    early_rejection policy is written against, to show WHY the fix is needed."""
    W, _ = _toy(d_out=32, d_in=256)
    a, b = W.ravel(), (0.01 * W).ravel()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, "cosine alone would have let a 0.01x-magnitude candidate through"


def test_flash_bank_screen_quant_error_now_reports_rel_fro():
    """The fix: quant_error() must return a magnitude-aware third value, and
    an honest quantizer at real bit-widths must score it near zero (positive
    control -- a check that only ever fails cannot be trusted)."""
    W, _ = _toy(d_out=32, d_in=256)
    cosine, mae, rel_fro = flash_doctor_bank_screen.quant_error(W, bits=4, group=64)
    assert 0.0 <= rel_fro < 0.2, f"honest q4 should hold magnitude reasonably, got rel_fro={rel_fro}"
    assert cosine > 0.99


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
