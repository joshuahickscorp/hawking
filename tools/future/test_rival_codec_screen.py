"""Tests for the rival-codec screen.

Negative controls that must actually fire:
  * the reference family is reproduced against the committed receipt (or the
    harness voids rivals when it cannot)
  * n_fit < fitted dimension is REFUSED, not scored
  * Q4 far from the committed ~0.101 voids the harness
  * a miss is never labelled as a contract pass
  * Q4 does not pass the coherence contract against itself
A skipped test is a P0. Absent corpus is a recorded refusal, not an omitted case.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import rival_codec_screen as rcs
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def _contract(**overrides):
    base = {
        "ok": True,
        "reason": "test contract",
        "min_heldout_cosine": 999 / 1000,
        "max_heldout_relative_fro_error": 1 / 20,
        "must_beat_per_expert_q4": True,
        "fit_holdout_required": True,
    }
    base.update(overrides)
    return base


def _tiny(n_rows=40, n_experts=3, d_out=8, d_in=64, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n_rows, d_in)).astype(np.float32)
    weights = rng.normal(size=(n_experts, d_out, d_in)).astype(np.float32)
    return states, weights


def test_holdout_split_matches_the_reference_screen():
    fit, heldout = rcs.heldout_split(4)
    assert fit.tolist() == [False, True, True, True]
    assert heldout.tolist() == [True, False, False, False]
    fit1024, held1024 = rcs.heldout_split(1024)
    assert int(fit1024.sum()) == 819
    assert int(held1024.sum()) == 205


def test_q4_factor_bytes_are_the_campaign_4_25_identity():
    n_params = 64 * 100
    factor_bytes = rcs.quant_factor_bytes(n_params, 4, group=64)
    bpw = rcs.diagnostic_bpw(factor_bytes, n_params)
    assert bpw == 4.25
    assert rcs.quant_factor_bytes(n_params, 3, group=64) < factor_bytes
    assert rcs.quant_factor_bytes(n_params, 2, group=64) < rcs.quant_factor_bytes(n_params, 3, group=64)


def test_q4_matches_qn_bits_4_and_is_lossy_on_gaussian_weights():
    """NEGATIVE CONTROL: Q4 of random weights is not a zero-error coding."""
    weights = np.random.default_rng(1).normal(size=(2, 8, 64)).astype(np.float32)
    q4 = rcs.symmetric_group_q4(weights)
    qn = rcs.symmetric_group_qn(weights, 4)
    assert np.allclose(q4, qn)
    err = float(np.linalg.norm((q4 - weights).astype(np.float64)))
    assert err > 0.0
    q2 = rcs.symmetric_group_qn(weights, 2)
    err2 = float(np.linalg.norm((q2 - weights).astype(np.float64)))
    assert err2 > err


def test_gate_can_pass_and_can_reject():
    """The gate must be able to return both True and False."""
    contract = _contract()
    passed = rcs.apply_surface_gates(
        heldout_error=0.04,
        heldout_cosine=0.9995,
        q4_error=0.10,
        contract=contract,
    )
    assert passed["surface_gate_pass"] is True
    assert passed["labelled"] == "SCORED_PASSED_CONTRACT"
    assert passed["surface_failure_gates"] == []
    assert passed["beats_per_expert_q4_on_heldout"] is True

    by_error = rcs.apply_surface_gates(
        heldout_error=0.20,
        heldout_cosine=0.9995,
        q4_error=0.10,
        contract=contract,
    )
    assert by_error["surface_gate_pass"] is False
    assert "held-out function error" in by_error["surface_failure_gates"]
    assert by_error["labelled"] == "SCORED_FAILED_CONTRACT"

    by_cosine = rcs.apply_surface_gates(
        heldout_error=0.01,
        heldout_cosine=0.90,
        q4_error=0.10,
        contract=contract,
    )
    assert by_cosine["surface_gate_pass"] is False
    assert "held-out function cosine" in by_cosine["surface_failure_gates"]

    by_q4 = rcs.apply_surface_gates(
        heldout_error=0.04,
        heldout_cosine=0.9995,
        q4_error=0.03,
        contract=contract,
    )
    assert by_q4["surface_gate_pass"] is False
    assert "does not beat per-expert Q4" in by_q4["surface_failure_gates"]
    assert by_q4["beats_per_expert_q4_on_heldout"] is False


def test_q4_control_does_not_pass_the_contract_against_itself():
    """NEGATIVE CONTROL: the comparator is not allowed to mark itself a pass."""
    contract = _contract()
    # Committed Q4 is ~0.101, above the 0.05 error gate, and error == q4 so it
    # does not beat itself either.
    row = rcs.apply_surface_gates(
        heldout_error=0.10143323614929661,
        heldout_cosine=0.995,
        q4_error=0.10143323614929661,
        contract=contract,
    )
    assert row["surface_gate_pass"] is False
    assert row["beats_per_expert_q4_on_heldout"] is False
    assert "does not beat per-expert Q4" in row["surface_failure_gates"]
    assert "held-out function error" in row["surface_failure_gates"]


def test_mislabelled_pass_is_caught():
    """NEGATIVE CONTROL: a validator nobody has watched reject is not a validator."""
    contract = _contract()
    forged = [
        {
            "family": "forged_pass",
            "scored": True,
            "surface_gate_pass": True,
            "heldout_relative_fro_error": 0.40,
            "heldout_cosine": 0.70,
            "per_expert_q4_heldout_relative_fro_error": 0.10,
            "beats_per_expert_q4_on_heldout": False,
            "labelled": "SCORED_PASSED_CONTRACT",
        }
    ]
    judgement = rcs.none_mislabelled(forged, contract)
    assert judgement["ok"] is False
    assert judgement["violations"]
    honest_fail = [
        {
            "family": "honest_fail",
            "scored": True,
            "surface_gate_pass": False,
            "heldout_relative_fro_error": 0.40,
            "heldout_cosine": 0.70,
            "per_expert_q4_heldout_relative_fro_error": 0.10,
            "beats_per_expert_q4_on_heldout": False,
            "labelled": "SCORED_FAILED_CONTRACT",
        }
    ]
    assert rcs.none_mislabelled(honest_fail, contract)["ok"] is True


def test_underdetermined_rank_is_refused_not_scored():
    """NEGATIVE CONTROL: n_fit < rank must not emit a starved score (NS-014)."""
    states, weights = _tiny(n_rows=10, n_experts=2, d_out=8, d_in=64)
    fit, held = rcs.heldout_split(states.shape[0])
    assert int(fit.sum()) < 16
    with pytest.raises(rcs.UnderdeterminedFitError):
        rcs.fit_shared_latent_program(
            states, weights, rank=16, fit_rows=fit, heldout_rows=held
        )
    teacher = rcs.projected_outputs(states, weights)
    full = rcs.attempt_full_dim_activation_map(states, teacher, fit, held)
    assert full["status"] == "REFUSED_UNDERDETERMINED"
    assert full["scored"] is False
    assert full["surface_gate_pass"] is False
    assert full["heldout_relative_fro_error"] is None
    assert full["fitted_dimension"] == 64
    assert full["n_fit"] == int(fit.sum())
    assert full["n_fit"] < full["fitted_dimension"]


def test_full_dim_map_scores_when_n_fit_meets_width():
    """The underdetermined guard must not refuse a determined full-width map."""
    states, weights = _tiny(n_rows=100, n_experts=2, d_out=4, d_in=64)
    fit, held = rcs.heldout_split(states.shape[0])
    assert int(fit.sum()) >= 64
    teacher = rcs.projected_outputs(states, weights)
    row = rcs.attempt_full_dim_activation_map(states, teacher, fit, held)
    assert row["scored"] is True
    assert row["heldout_relative_fro_error"] is not None
    assert row["fitted_dimension"] == 64


def test_q4_far_from_committed_voids_harness():
    """NEGATIVE CONTROL: a Q4 that is not ~0.101 means the harness is wrong."""
    committed = [
        {
            "rank": 4,
            "heldout_relative_fro_error": 0.6677643943246043,
            "heldout_cosine": 0.7444498406100204,
            "fit_relative_fro_error": 0.660940946465921,
            "diagnostic_factor_equivalent_bpw": 0.02515923566878981,
            "per_expert_q4_heldout_relative_fro_error": 0.10143323614929661,
        }
    ]
    reproduced = [
        {
            "rank": 4,
            "heldout_relative_fro_error": 0.6677643943246043,
            "heldout_cosine": 0.7444498406100204,
            "fit_relative_fro_error": 0.660940946465921,
            "diagnostic_factor_equivalent_bpw": 0.02515923566878981,
        }
    ]
    bad = rcs.judge_harness(
        reproduced=reproduced,
        committed=committed,
        q4_error=0.50,
        expected_q4=0.10143323614929661,
    )
    assert bad["ok"] is False
    assert bad["rivals_publishable"] is False
    assert bad["q4_ok"] is False
    good = rcs.judge_harness(
        reproduced=reproduced,
        committed=committed,
        q4_error=0.10143323614929661,
        expected_q4=0.10143323614929661,
    )
    assert good["ok"] is True
    assert good["rivals_publishable"] is True


def test_reference_mismatch_voids_rivals():
    """NEGATIVE CONTROL: a harness that cannot reproduce the known result publishes nothing."""
    committed = [
        {
            "rank": 4,
            "heldout_relative_fro_error": 0.6677643943246043,
            "heldout_cosine": 0.7444498406100204,
            "fit_relative_fro_error": 0.660940946465921,
            "diagnostic_factor_equivalent_bpw": 0.02515923566878981,
            "per_expert_q4_heldout_relative_fro_error": 0.10143323614929661,
        }
    ]
    reproduced = [
        {
            "rank": 4,
            "heldout_relative_fro_error": 0.10,
            "heldout_cosine": 0.99,
            "fit_relative_fro_error": 0.09,
            "diagnostic_factor_equivalent_bpw": 0.02515923566878981,
        }
    ]
    harness = rcs.judge_harness(
        reproduced=reproduced,
        committed=committed,
        q4_error=0.10143323614929661,
        expected_q4=0.10143323614929661,
    )
    assert harness["ok"] is False
    assert harness["reference_ok"] is False
    assert harness["rivals_publishable"] is False


def test_absent_contract_refuses_the_screen():
    """NEGATIVE CONTROL: missing contract is not a defaulted table."""
    states, weights = _tiny()
    result = rcs.screen_on_arrays(
        states=states,
        weights=weights,
        contract={"ok": False, "reason": "absent"},
        ranks=(4,),
    )
    assert result["status"] == "REFUSED_NO_CONTRACT"
    assert result["rivals_published"] is not True
    assert result["families"] == []


def test_tiny_screen_q4_does_not_pass_and_labelling_holds():
    states, weights = _tiny()
    expert_ids = [10, 20, 30]
    fit_routes = [expert_ids for _ in range(states.shape[0])]
    result = rcs.screen_on_arrays(
        states=states,
        weights=weights,
        contract=_contract(),
        ranks=(4, 8),
        fit_route_ids=fit_routes,
        expert_ids=expert_ids,
        backbone=weights.mean(axis=0),
    )
    assert result["status"] == "RIVAL_FAMILIES_SCORED_OFFLINE_META_SURFACE"
    assert result["labelling"]["ok"] is True
    q4 = result["families"][0]["rows"][0]
    assert q4["family"] == "per_expert_q4_control"
    assert q4["scored"] is True
    assert q4["surface_gate_pass"] is False
    assert q4["beats_per_expert_q4_on_heldout"] is False
    ids = [f["family"] for f in result["families"]]
    assert "per_expert_q3_control" in ids
    assert "per_expert_q2_control" in ids
    assert "common_left_subspace_plus_expert_local_core" in ids
    assert "common_right_subspace_plus_expert_local_core" in ids
    assert "clustered_subspaces_route_conditioned" in ids
    assert "dictionary_plus_per_expert_sparse_residual" in ids
    assert "expert_local_small_core_plus_shared_decoder" in ids
    assert "sparse_residual_on_cheap_backbone" in ids
    assert rcs.REFERENCE_KIND in ids
    for name in (
        "common_left_subspace_plus_expert_local_core",
        "common_right_subspace_plus_expert_local_core",
        "clustered_subspaces_route_conditioned",
        "dictionary_plus_per_expert_sparse_residual",
        "expert_local_small_core_plus_shared_decoder",
        "sparse_residual_on_cheap_backbone",
        rcs.REFERENCE_KIND,
    ):
        block = next(f for f in result["families"] if f["family"] == name)
        assert block["n_scored"] == 2, (name, [r.get("reason") for r in block["rows"]])
        assert all(r.get("scored") is True for r in block["rows"])
    for row in result["flat_rows"]:
        if row.get("surface_gate_pass") is True:
            assert row.get("scored") is True
            assert row["heldout_relative_fro_error"] <= 1 / 20
            assert row["heldout_cosine"] >= 999 / 1000
            assert row["beats_per_expert_q4_on_heldout"] is True
        if row.get("scored") is False:
            assert row.get("surface_gate_pass") is not True


def test_left_basis_is_the_output_side():
    """A left-space gram taken on the input axis is a different (wrong) object."""
    _states, weights = _tiny()
    u = rcs.left_basis(weights, 4)
    v = rcs.right_basis(weights, 4)
    assert u.shape == (weights.shape[1], 4)
    assert v.shape == (4, weights.shape[2])
    assert u.shape[0] != v.shape[1] or weights.shape[1] == weights.shape[2]


def test_absent_backbone_is_refused_not_replaced_with_mean_expert():
    states, weights = _tiny()
    result = rcs.screen_on_arrays(
        states=states,
        weights=weights,
        contract=_contract(),
        ranks=(4,),
        fit_route_ids=None,
        expert_ids=None,
        backbone=None,
    )
    block = next(
        f for f in result["families"] if f["family"] == "sparse_residual_on_cheap_backbone"
    )
    assert all(r["status"] == "REFUSED_ABSENT_INPUT" for r in block["rows"])
    assert all(r["surface_gate_pass"] is False for r in block["rows"])
    assert "mean expert" in block["rows"][0]["reason"]
    clustered = next(
        f for f in result["families"] if f["family"] == "clustered_subspaces_route_conditioned"
    )
    assert clustered["rows"][0]["status"] == "REFUSED_ABSENT_INPUT"


def test_common_right_full_available_rank_beats_rank_one():
    """A one-space factor that is too skinny must be able to lose to a fuller one."""
    states, weights = _tiny(n_rows=40, n_experts=3, d_out=8, d_in=64, seed=2)
    fit, held = rcs.heldout_split(states.shape[0])
    teacher = rcs.projected_outputs(states, weights)
    skinny = rcs.fit_common_right(
        states, weights, teacher, fit, held, rank=1
    )
    fuller = rcs.fit_common_right(
        states, weights, teacher, fit, held, rank=8
    )
    assert fuller["heldout_relative_fro_error"] < skinny["heldout_relative_fro_error"]
    assert skinny["scored"] is True
    assert fuller["scored"] is True


def test_contract_is_read_from_screen_not_invented():
    absent = rcs.contract_from_screen(None)
    assert absent["ok"] is False
    assert absent["min_heldout_cosine"] is None
    screen = {
        "coherence_contract": {
            "min_heldout_cosine": 999 / 1000,
            "max_heldout_relative_fro_error": 1 / 20,
            "must_beat_per_expert_q4": True,
        }
    }
    got = rcs.contract_from_screen(screen)
    assert got["ok"] is True
    assert got["min_heldout_cosine"] == 999 / 1000
    assert got["max_heldout_relative_fro_error"] == 1 / 20
    assert got["must_beat_per_expert_q4"] is True
    broken = rcs.contract_from_screen({"coherence_contract": {"min_heldout_cosine": 0.999}})
    assert broken["ok"] is False


def test_build_copes_with_or_without_the_corpus():
    """build() must write a sealed receipt either way. Never skip."""
    out = rcs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RIVAL_CODEC_SCREEN.json"
    assert doc["schema"] == rcs.SCHEMA
    assert doc["version"] == 1
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["promotion_allowed"] is False
    assert doc["a_pass_is_not_a_win"] is True
    assert "Static sidecar" in doc["claim_boundary"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_REPRESENTATION.meta-gates-3-9"
    assert doc["status"] in {
        "RIVAL_FAMILIES_SCORED_OFFLINE_META_SURFACE",
        "HARNESS_INVALID_RIVALS_VOID",
        "REFUSED_ABSENT_INPUT",
        "REFUSED_INSUFFICIENT_TEACHER_COVERAGE",
        "REFUSED_NO_CONTRACT",
    }
    if doc["status"] == "RIVAL_FAMILIES_SCORED_OFFLINE_META_SURFACE":
        assert doc["harness"]["ok"] is True
        assert doc["rivals_published"] is True
        q4 = doc["q4_error"]
        expected = (
            ((doc.get("harness") or {}).get("expected_q4"))
            or 0.10143323614929661
        )
        assert abs(q4 - expected) <= rcs.Q4_ABS_TOL
        assert doc["labelling"]["ok"] is True
        for fam in doc["families"]:
            for row in fam["rows"]:
                if row.get("surface_gate_pass") is True:
                    assert row.get("scored") is True
                    assert row["heldout_relative_fro_error"] <= doc["contract"][
                        "max_heldout_relative_fro_error"
                    ]
                    assert row["heldout_cosine"] >= doc["contract"]["min_heldout_cosine"]
                    assert row["beats_per_expert_q4_on_heldout"] is True
                if row.get("scored") is False:
                    assert row.get("surface_gate_pass") is not True
        under = doc["underdetermined_control"]
        assert under["status"] == "REFUSED_UNDERDETERMINED"
        assert under["scored"] is False
        # Reproduced reference family must sit on the committed numbers.
        ref = next(f for f in doc["families"] if f["family"] == rcs.REFERENCE_KIND)
        committed = rcs.committed_reference_rows(rcs.load_named(rcs.SCREEN_REL)[0] or {})
        if committed:
            assert rcs.judge_harness(
                reproduced=ref["rows"],
                committed=committed,
                q4_error=q4,
                expected_q4=expected,
            )["ok"] is True
    else:
        assert doc["rivals_published"] is False
        assert doc["reason"]


def test_selftest_is_callable():
    assert callable(rcs.selftest)


def test_threaded_expert_factoring_is_bit_identical_to_the_serial_path(monkeypatch):
    """Parallelism here must be a scheduling change, never a numerical one.

    Each expert's gram and eigendecomposition is independent, so factoring them
    across threads has to return exactly what the serial loop returned, in the
    same expert order. A result that merely agrees to a tolerance, or that comes
    back in completion order, is a different screen.
    """
    rng = np.random.default_rng(7)
    residuals = rng.standard_normal((9, 48, 61), dtype=np.float32)

    monkeypatch.setenv("RCS_EIGH_WORKERS", "1")
    assert rcs._eigh_workers() == 1
    serial_u, serial_v = rcs.residual_factors_batch(residuals, 6)

    monkeypatch.setenv("RCS_EIGH_WORKERS", "4")
    assert rcs._eigh_workers() == 4
    threaded_u, threaded_v = rcs.residual_factors_batch(residuals, 6)

    assert np.array_equal(serial_u, threaded_u), "threading moved the left factors"
    assert np.array_equal(serial_v, threaded_v), "threading moved the right factors"

    # Order, not just contents: a shuffled batch must NOT compare equal, or the
    # assertions above would pass for a result reassembled in completion order.
    shuffled_u, _ = rcs.residual_factors_batch(residuals[::-1], 6)
    assert not np.array_equal(serial_u, shuffled_u)


def test_rank_sweep_reuses_one_eigendecomposition_and_nothing_else(monkeypatch):
    """The sweep must pay for the bank's eigenvectors once, and only for that bank.

    right_basis() is asked for ranks 4, 8, 16, 32, 64 of the SAME stacked bank. The
    gram and its eigendecomposition do not depend on the rank -- only the trailing
    slice does -- so recomputing them per rank is pure waste. Reuse is only sound if
    it is keyed on the identity of THIS bank: a different array of the same shape
    must not be served the first one's eigenvectors.
    """
    rng = np.random.default_rng(11)
    bank = rng.standard_normal((5, 7, 24), dtype=np.float32)

    calls = []
    real_eigh = np.linalg.eigh
    monkeypatch.setattr(
        np.linalg, "eigh", lambda g: (calls.append(g.shape), real_eigh(g))[1]
    )

    swept = {rank: rcs.right_basis(bank, rank) for rank in (4, 8, 16)}
    assert len(calls) == 1, f"the rank sweep decomposed the same bank {len(calls)} times"

    # Reuse must not change the answer: every rank is still the trailing slice.
    full = swept[16]
    for rank in (4, 8):
        assert np.array_equal(swept[rank], full[-rank:]), f"rank {rank} drifted"

    # A DIFFERENT bank of the same shape must be decomposed on its own terms.
    other = rng.standard_normal((5, 7, 24), dtype=np.float32)
    other_v = rcs.right_basis(other, 4)
    assert len(calls) == 2, "a different bank was served the first bank's eigenvectors"
    assert not np.array_equal(other_v, swept[4])

    # And the original bank, asked again after eviction, still answers correctly.
    monkeypatch.setattr(np.linalg, "eigh", real_eigh)
    assert np.array_equal(rcs.right_basis(bank, 4), swept[4])


def test_top_eigh_is_the_TOP_eigenspace_not_merely_some_k_columns():
    """The defining property, which nothing else in this file pins.

    _top_eigh underwrites left_basis, right_basis, residual_factors and
    residual_factors_batch -- every factorization here. Inverting its slice to
    `evecs[:, :rank]` returns the eigenvectors of the SMALLEST eigenvalues: the
    worst rank-k subspace instead of the best. That mutation survived all 19 fast
    tests in this file, because test_left_basis_is_the_output_side asserts only
    SHAPES (both slices are the same shape) and the threading test compares the
    serial path against the threaded one (both would be equally wrong).

    A comparison of two paths through the same function cannot detect a fault in
    that function. This asserts the property against an independent reference.
    """
    rng = np.random.default_rng(3)
    a = rng.standard_normal((40, 12), dtype=np.float32).astype(np.float64)
    gram = a @ a.T
    rank = 4

    got = rcs._top_eigh(gram, rank)
    assert got.shape == (40, rank)

    evals, evecs = np.linalg.eigh(gram)
    top = evecs[:, -rank:]
    bottom = evecs[:, :rank]

    # Energy captured, computed independently of which columns were returned.
    def captured(basis):
        basis = basis.astype(np.float64, copy=False)
        return float(np.trace(basis.T @ gram @ basis))

    assert captured(got) > captured(bottom), "returned the WORST rank-k subspace"
    # rel=1e-6, not tighter: _top_eigh returns float32, the reference is float64,
    # and the gap between them here is 4e-9 relative -- float32 rounding, not a
    # different subspace. The load-bearing comparison is the one above, which the
    # inverted slice fails by a factor of ~100.
    assert captured(got) == pytest.approx(captured(top), rel=1e-6)

    # And it really is the maximiser: no other rank-k eigen-subspace beats it.
    best = float(np.sort(evals)[-rank:].sum())
    assert captured(got) == pytest.approx(best, rel=1e-6)


def test_residual_factors_absorb_the_projection_into_V_single_and_batch():
    """V must be U^T R, not just some r rows of R.

    residual_factors and residual_factors_batch return R ~ U @ V with U spanning
    the top-r left eigenspace. Replacing V with `R[:r]` -- discarding the
    projection entirely -- passed all 20 fast tests in this file. The batch case
    survived even though the threading test calls it, because that test compares
    the serial path to the threaded one and both would carry the same wrong V.

    Asserted here against the property itself: V is the projection, and U @ V is
    the orthogonal projection of R onto U's span, which strictly beats taking r
    raw rows of R.
    """
    rng = np.random.default_rng(5)
    residual = rng.standard_normal((30, 41), dtype=np.float32)
    rank = 5

    u, v = rcs.residual_factors(residual, rank)
    assert u.shape == (30, rank)
    assert v.shape == (rank, 41)

    # V is the projection of R onto U, not an arbitrary slice of R.
    assert np.allclose(v, u.T @ residual, atol=1e-4), "V is not U^T R"

    # U @ V is therefore the orthogonal projection U U^T R.
    assert np.allclose(u @ v, (u @ u.T) @ residual, atol=1e-4)

    # And that beats the mutant that drops the projection.
    def err(pred):
        return float(np.linalg.norm(pred - residual))

    assert err(u @ v) < err(u @ residual[:rank]), "the projection buys nothing"

    # The batch path must carry the same property, per expert.
    batch = np.stack([residual, residual[::-1] * 0.5], axis=0)
    bu, bv = rcs.residual_factors_batch(batch, rank)
    assert bu.shape == (2, 30, rank)
    assert bv.shape == (2, rank, 41)
    for i in range(2):
        assert np.allclose(bv[i], bu[i].T @ batch[i], atol=1e-4), f"expert {i}: V is not U^T R"
        projected = float(np.linalg.norm(bu[i] @ bv[i] - batch[i]))
        raw_rows = float(np.linalg.norm(bu[i] @ batch[i][:rank] - batch[i]))
        assert projected < raw_rows, f"expert {i}: the projection buys nothing"


def test_fused_metrics_are_bit_identical_to_the_separate_ones():
    """Removing duplicated work must not move a single bit.

    relative_fro and cosine each upcast the SAME two arrays to float64
    independently, so scoring one pair paid for four temporaries where two
    suffice. The fused form must produce exactly what the separate calls did --
    not "within a tolerance", exactly, because a screen that shifts under a
    refactor cannot be compared against its own committed numbers.
    """
    rng = np.random.default_rng(19)
    for shape in ((5, 31, 17), (2, 64, 8), (9, 3, 40)):
        pred = rng.standard_normal(shape, dtype=np.float32)
        teacher = rng.standard_normal(shape, dtype=np.float32)
        fused = rcs._relative_fro_and_cosine(pred, teacher)
        assert fused["heldout_relative_fro_error"] == rcs.relative_fro(pred, teacher)
        assert fused["heldout_cosine"] == rcs.cosine(pred, teacher)


def test_the_masked_gather_is_reused_only_for_the_SAME_teacher_and_mask():
    """A 660 MB gather may be reused, but never across different inputs."""
    rng = np.random.default_rng(23)
    teacher = rng.standard_normal((4, 12, 6), dtype=np.float32)
    mask = np.zeros(12, dtype=bool)
    mask[::2] = True

    first = rcs._masked_columns(teacher, mask)
    again = rcs._masked_columns(teacher, mask)
    assert again is first, "the identical gather was recomputed"
    assert np.array_equal(first, teacher[:, mask])

    other_mask = np.zeros(12, dtype=bool)
    other_mask[1::2] = True
    third = rcs._masked_columns(teacher, other_mask)
    assert third is not first, "a different mask was served the cached gather"
    assert np.array_equal(third, teacher[:, other_mask])

    other_teacher = rng.standard_normal((4, 12, 6), dtype=np.float32)
    fourth = rcs._masked_columns(other_teacher, other_mask)
    assert np.array_equal(fourth, other_teacher[:, other_mask]), (
        "a different teacher was served another teacher's columns"
    )
