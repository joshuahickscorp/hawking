"""G119 tests: four perturbation kinds, matched by construction, and every
refusal raises. The expensive replay is NOT exercised here - `perturb` is pure
so the maths can be checked without a two-minute CPU replay per assertion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perturbation_workunit as pw  # noqa: E402


def W():
    rng = np.random.default_rng(0)
    return rng.normal(size=(16, 24))


@pytest.mark.parametrize("kind", pw.KINDS)
def test_every_kind_touches_the_same_elements_at_the_same_fraction(kind):
    """fraction means the share of rows SELECTED in every kind, so two kinds at
    one fraction are matched by construction. That is what makes them
    comparable at all."""
    _, n, _ = pw.perturb(W(), "rows", 0.5, kind, 3)
    _, n0, _ = pw.perturb(W(), "rows", 0.5, "zero", 3)
    assert n == n0


@pytest.mark.parametrize("kind", pw.KINDS)
def test_the_original_is_never_mutated(kind):
    w = W()
    before = w.copy()
    pw.perturb(w, "rows", 0.4, kind, 1)
    assert np.array_equal(w, before), "damage must be applied to a copy"


@pytest.mark.parametrize("kind", pw.KINDS)
def test_only_the_selected_slice_changes(kind):
    w = W()
    w2, _, d = pw.perturb(w, "rows", 0.25, kind, 5)
    changed_rows = {int(i) for i in np.where((w2 != w).any(axis=1))[0]}
    assert len(changed_rows) <= d["selected"]


def test_zero_actually_zeroes():
    w = W()
    w2, _, _ = pw.perturb(w, "rows", 0.5, "zero", 2)
    touched = (w2 != w).any(axis=1)
    assert np.allclose(w2[touched], 0.0)


def test_quantize_stays_inside_the_slice_range():
    """A quantizer that leaves the data's own range is not quantizing it."""
    w = W()
    w2, _, d = pw.perturb(w, "rows", 0.5, "quantize", 2)
    rows = np.where((w2 != w).any(axis=1))[0]
    for r in rows:
        assert w2[r].min() >= w[r].min() - 1e-9
        assert w2[r].max() <= w[r].max() + 1e-9
    assert d["bits"] == 2


def test_quantize_is_gentler_than_zero():
    """If coarse quantization damaged as much as deletion, it would not be a
    separate question."""
    w = W()
    zq = np.abs(pw.perturb(w, "rows", 0.5, "zero", 4)[0] - w).sum()
    qq = np.abs(pw.perturb(w, "rows", 0.5, "quantize", 4)[0] - w).sum()
    assert qq < zq


def test_noise_is_scaled_to_the_slice_not_absolute():
    """A fixed sigma destroys a small-magnitude row and barely touches a large
    one, which would confound magnitude with importance."""
    w = W()
    _, _, d1 = pw.perturb(w, "rows", 0.5, "noise", 6)
    _, _, d2 = pw.perturb(w * 100.0, "rows", 0.5, "noise", 6)
    assert d2["sigma"] == pytest.approx(d1["sigma"] * 100.0, rel=1e-6)


def test_low_rank_keeps_most_of_the_energy():
    w = W()
    _, _, d = pw.perturb(w, "rows", 0.5, "low_rank", 8)
    assert 0 < d["rank"] <= d["of_rank"]
    assert 0.0 < d["energy_kept"] <= 1.0 + 1e-9


def test_low_rank_is_a_real_rank_reduction():
    w = W()
    w2, _, d = pw.perturb(w, "rows", 0.5, "low_rank", 8)
    rows = np.where((w2 != w).any(axis=1))[0]
    assert np.linalg.matrix_rank(w2[rows]) <= d["rank"]


def test_the_same_seed_selects_the_same_slice():
    a, _, _ = pw.perturb(W(), "cols", 0.3, "zero", 11)
    b, _, _ = pw.perturb(W(), "cols", 0.3, "zero", 11)
    assert np.array_equal(a, b)


def test_rows_and_cols_at_one_fraction_touch_the_SAME_count():
    """Worth pinning because it is counter-intuitive: f*R rows of C columns and
    f*C columns of R rows are both f*R*C elements. The axis does not change the
    budget, so a rows-vs-cols comparison IS matched on element count."""
    _, nr, _ = pw.perturb(W(), "rows", 0.5, "zero", 1)
    _, nc, _ = pw.perturb(W(), "cols", 0.5, "zero", 1)
    assert nr == nc


def test_matched_FRACTION_across_different_tensors_is_not_matched_ELEMENTS():
    """The scar this campaign already paid for. gate and down are different
    shapes, so 40% of each destroys different amounts of information, and a
    fraction-matched comparison between them is not a control."""
    small = np.zeros((16, 24))
    large = np.zeros((64, 24))
    _, ns, _ = pw.perturb(small, "rows", 0.5, "zero", 1)
    _, nl, _ = pw.perturb(large, "rows", 0.5, "zero", 1)
    assert ns != nl
    assert nl == 4 * ns


@pytest.mark.parametrize("bad", ["delete", "", "ZERO", None])
def test_an_unknown_kind_raises(bad):
    with pytest.raises(pw.PerturbRefused, match="kind"):
        pw.perturb(W(), "rows", 0.5, bad, 1)


def test_run_refuses_outside_its_authority():
    for kwargs in (
        dict(tensor="q_proj", layer=0, side="rows", fraction=0.5),
        dict(tensor="up", layer=0, side="diagonal", fraction=0.5),
        dict(tensor="up", layer=0, side="rows", fraction=1.5),
        dict(tensor="up", layer=0, side="rows", fraction=0.5, kind="delete"),
    ):
        with pytest.raises(pw.PerturbRefused):
            pw.run(**kwargs)


def test_the_four_kinds_s031_names_are_all_present():
    assert set(pw.KINDS) == {"zero", "quantize", "noise", "low_rank"}
    assert pw.DEFAULT_KIND == "zero"


def test_the_contract_names_all_four_kinds_and_what_each_asks():
    c = pw.contract()
    assert set(c["kinds"]) == set(pw.KINDS)
    assert "REMOVED" in c["kinds"]["zero"]
    assert "CHEAPER" in c["kinds"]["quantize"]
    assert "redundant rather than absent" in c["kinds"]["low_rank"]


def test_the_contract_refuses_to_call_its_output_capability():
    c = pw.contract()
    assert c["measured_level"] == "LOCAL_FUNCTIONAL_FIDELITY"
    assert "is not capability" in c["not_capability"]
    assert "CAPABILITY_STAGES" in c["not_capability"]


def test_the_contract_states_the_matched_element_caveat():
    c = pw.contract()
    assert "DIFFERENT-SIZED tensors it does not" in c["matched_by_construction"]


def test_the_contract_inputs_match_the_code():
    c = pw.contract()["inputs"]
    assert c["tensor"] == list(pw.TENSORS)
    assert c["side"] == list(pw.SIDES)
    assert c["kind"] == list(pw.KINDS)


def test_the_receipt_parses_and_names_the_obligation():
    import json as _j
    from _common import REPO
    d = _j.loads((REPO / "receipts/future/COMPONENT_PERTURBATION.json").read_text())
    assert d["obligation"] == "G119"
    assert set(d["kinds"]) == set(pw.KINDS)
