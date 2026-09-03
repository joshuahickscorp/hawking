"""Tests for the sparse residual-rescue census.

Load-bearing negatives a guard nobody has watched fail is not a guard:

  * a sparse residual whose gather indices are not billed is REFUSED
  * a train-set figure cannot be reported as held-out
  * uniform and capability-allocated budgets are both reported
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import executable_economics as ee
from tools.future import mlp_sparse_residual as msr
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def _fixture(**kwargs):
    return msr.make_fixture(**kwargs)


def test_unbilled_residual_index_is_refused():
    """NEGATIVE CONTROL: a free gather index is a fabrication, not a candidate."""
    br = msr.residual_byte_breakdown(k=8, n_layers=2, hidden=16)
    added = msr.bytes_added_from_breakdown(br)
    msr.validate_billing(
        {"k": 8, "byte_breakdown": br, "bytes_added": added, "dispatch_delta": 2.0}
    )
    assert br["index_bytes"] == 8 * msr.INDEX_BYTES * 2
    assert added["metadata"] >= br["index_bytes"]
    assert added["residuals"] == br["value_bytes"]
    assert added["total"] == sum(added[k] for k in ee.BYTES_ADDED_FIELDS)

    stolen = dict(added)
    stolen["metadata"] = int(br["metadata_base_bytes"])
    stolen["total"] = sum(int(stolen[k]) for k in ee.BYTES_ADDED_FIELDS)
    with pytest.raises(msr.UnbilledResidualIndex, match="index"):
        msr.validate_billing(
            {
                "k": 8,
                "byte_breakdown": br,
                "bytes_added": stolen,
                "dispatch_delta": 2.0,
            }
        )

    zero_br = dict(br)
    zero_br["index_bytes"] = 0
    with pytest.raises(msr.UnbilledResidualIndex, match="not billed"):
        msr.validate_billing(
            {
                "k": 8,
                "byte_breakdown": zero_br,
                "bytes_added": added,
                "dispatch_delta": 2.0,
            }
        )


def test_waved_dispatch_is_refused():
    br = msr.residual_byte_breakdown(k=4, n_layers=2, hidden=16)
    added = msr.bytes_added_from_breakdown(br)
    with pytest.raises(msr.UnbilledDispatch, match="dispatch"):
        msr.validate_billing(
            {
                "k": 4,
                "byte_breakdown": br,
                "bytes_added": added,
                "dispatch_delta": 0.0,
            }
        )


def test_k_zero_may_omit_indices():
    br = msr.residual_byte_breakdown(k=0, n_layers=2, hidden=16)
    added = msr.bytes_added_from_breakdown(br)
    assert br["index_bytes"] == 0
    msr.validate_billing(
        {"k": 0, "byte_breakdown": br, "bytes_added": added, "dispatch_delta": 0.0}
    )


def test_train_set_figure_cannot_be_reported_as_held_out():
    """NEGATIVE CONTROL: a fit-set number labelled held-out must refuse."""
    fx = _fixture()
    ho = msr.function_error(fx["Yho"], fx["Yho"], split="hold", report_as="held_out")
    assert ho["held_out_split"] == "hold"
    assert ho["error_authority"] == "held_out_relative_l2"
    assert ho["held_out_relative_l2"] == pytest.approx(0.0, abs=1e-12)

    tr = msr.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="train")
    assert "held_out_relative_l2" not in tr
    assert tr["train_split"] == "train"

    with pytest.raises(msr.TrainReportedAsHeldOut, match="cannot be reported as held-out"):
        msr.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="held_out")

    with pytest.raises(msr.TrainReportedAsHeldOut):
        msr.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="hold")

    forged = {
        "held_out_relative_l2": tr["train_relative_l2_diagnostic"],
        "held_out_split": "train",
        "error_authority": "held_out_relative_l2",
    }
    with pytest.raises(msr.TrainReportedAsHeldOut, match="held_out_split"):
        msr.validate_error_authority(forged)

    leaked = {
        "held_out_relative_l2": 0.01,
        "held_out_split": "hold",
        "fitted_on": "hold",
        "error_authority": "held_out_relative_l2",
    }
    with pytest.raises(msr.TrainReportedAsHeldOut):
        msr.validate_error_authority(leaked)


def test_consumer_is_atlas_gather_and_add():
    sketch = msr.residual_consumer_sketch(k=32)
    assert sketch["primitive"] == msr.GATHER_ADD_PRIMITIVE
    assert sketch["primitive"] in ATLAS_PRIMITIVES
    assert sketch["index_billed"] is True
    assert sketch["dispatch_delta"] == float(msr.N_LAYERS)
    assert sketch["rematerialize_dense_W"] is False
    for name in sketch["also"]:
        assert name in ATLAS_PRIMITIVES
    dead = msr.residual_consumer_sketch(k=32, rematerialize_dense_W=True)
    assert dead["status"] == msr.REJECTED_DENSE_REMAT


def test_concentration_curve_is_sparse_on_a_sparse_fixture():
    rng = np.random.default_rng(0)
    rho = np.zeros((32, 16), dtype=np.float64)
    rho[:, :2] = rng.standard_normal((32, 2))
    conc = msr.residual_concentration(rho)
    oc = conc["axes"]["output_coords"]
    assert oc["dense"] is False
    assert oc["frac_kept_for_50pct_energy"] <= 2.0 / 16.0 + 1e-9
    # Two of 16 coords carry the residual; 10% kept is 2 coords.
    assert oc["frac_energy_at_10pct"] > 0.9


def test_concentration_curve_is_dense_on_isotropic_residual():
    rng = np.random.default_rng(1)
    rho = rng.standard_normal((200, 64)).astype(np.float32)
    conc = msr.residual_concentration(rho)
    oc = conc["axes"]["output_coords"]
    assert oc["dense"] is True
    assert oc["frac_energy_at_1pct"] <= msr.DENSE_ENERGY_AT_1PCT
    assert oc["frac_energy_at_10pct"] <= msr.DENSE_ENERGY_AT_10PCT
    assert oc["frac_kept_for_50pct_energy"] > 0.3
    tok = conc["axes"]["tokens"]
    assert tok["axis"] == "tokens"


def test_allocate_uniform_and_capability_differ_when_prior_is_skewed():
    energy = np.arange(16, dtype=np.float64)
    cap = msr.fixture_capability_weights(16)
    u = msr.allocate_coords(energy, 4, policy=msr.UNIFORM, weights=cap["coord_weights"])
    c = msr.allocate_coords(energy, 4, policy=msr.CAPABILITY, weights=cap["coord_weights"])
    g = msr.allocate_coords(energy, 4, policy=msr.ENERGY_GREEDY)
    assert len(u) == 4
    assert len(c) == 4
    assert set(g.tolist()) == {12, 13, 14, 15}
    # Quiet last block is down-weighted under capability, equal under uniform.
    last = set(range(12, 16))
    assert len(set(u.tolist()) & last) == 1
    assert len(set(c.tolist()) & last) == 0


def test_uniform_and_capability_both_reported():
    fx = _fixture(hidden=16, sparse_k=2, n_train=40, n_hold=12, rank=3)
    cap = msr.fixture_capability_weights(16)
    budgets = [
        {"name": "bulk_only", "mlp_frac_requested": 0.0, "k": 0},
        {"name": "k2", "mlp_frac_requested": 0.0, "k": 2},
        {"name": "k4", "mlp_frac_requested": 0.0, "k": 4},
    ]
    out = msr.measure(
        pack=fx,
        cap=cap,
        bulk_ids=["shared_input_r64"],
        n_layers=2,
        budgets=budgets,
        fit_gram=True,
        rank=3,
    )
    allocs = {p["allocation"] for b in out["bulks"] for p in b["budget_sweep"]}
    assert msr.UNIFORM in allocs
    assert msr.CAPABILITY in allocs
    assert out["school"]["status"]
    for bulk in out["bulks"]:
        assert "map_value_vs_uniform" in bulk
        ks = {row["k"] for row in bulk["map_value_vs_uniform"]}
        assert ks == {0, 2, 4}
        for point in bulk["budget_sweep"]:
            assert point["held_out_split"] == "hold"
            assert point["error_authority"] == "held_out_relative_l2"
            assert "held_out_relative_l2" in point
            assert "train_relative_l2_diagnostic" in point
            assert point["held_out_relative_l2"] != point.get("train_split")
            msr.validate_error_authority(point)
            msr.validate_billing(point)
            if int(point["k"]) > 0:
                assert point["byte_breakdown"]["index_bytes"] > 0
                assert point["bytes_added"]["metadata"] >= point["byte_breakdown"]["index_bytes"]
                assert point["dispatch_delta"] > 0.0
                assert point["economics"]["dispatch_delta"] > 0.0
                assert point["economics"]["terms"]["dispatch_ms_delta"] > 0.0
                assert point["economics"]["bytes_added"]["metadata"] >= point["byte_breakdown"][
                    "index_bytes"
                ]
                assert point["consumer"]["primitive"] in ATLAS_PRIMITIVES


def test_selftest_fires_the_guards():
    out = msr.selftest()
    assert out["held_out_leak_refused"] is True
    assert out["unbilled_residual_index_refused"] is True
    assert out["unbilled_dispatch_refused"] is True


def test_k_from_mlp_frac_bills_index_and_value():
    k1 = msr.k_from_mlp_frac(msr.mlp_frac_of_k(1))
    assert k1 == 1
    per = msr.per_coord_bytes()
    assert per == msr.N_LAYERS * (msr.HIDDEN * msr.ELEMENT_BYTES + msr.INDEX_BYTES)
    # One coordinate is more than 1e-4 of MLP bytes once indices are billed.
    assert msr.k_from_mlp_frac(1e-4) == 0
    assert msr.k_from_mlp_frac(0.001) >= 1
    licensed = 2_785_280 / ee.MLP_ACTIVE_BYTES
    assert msr.k_from_mlp_frac(licensed) >= 1


def test_economics_scores_index_and_dispatch_not_bytes_alone():
    br = msr.residual_byte_breakdown(k=81)
    added = msr.bytes_added_from_breakdown(br)
    scored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: added[k] for k in ee.BYTES_ADDED_FIELDS},
        extra_flops_per_output_element=msr.extra_flops_per_output_element(81),
        dispatch_delta=msr.dispatch_delta_for_k(81),
        consuming_primitive=msr.GATHER_ADD_PRIMITIVE,
        organ="mlp",
        stream_class="weight_codes",
        reusable_family=True,
        high_information_falsifier=True,
        status=msr.OPEN,
    )
    assert scored["bytes_added"]["metadata"] >= br["index_bytes"]
    assert scored["bytes_added"]["residuals"] == br["value_bytes"]
    assert scored["dispatch_delta"] == float(msr.N_LAYERS)
    assert scored["terms"]["dispatch_ms_delta"] > 0.0
    assert scored["extra_flops_per_output_element"] > 0.0
    assert scored["terms"]["flop_ms_delta"] > 0.0
    free = dict(added)
    free["metadata"] = 0
    with pytest.raises(msr.UnbilledResidualIndex):
        msr.validate_billing(
            {
                "k": 81,
                "byte_breakdown": br,
                "bytes_added": free,
                "dispatch_delta": float(msr.N_LAYERS),
            }
        )


def test_oracle_correct_of_all_coords_is_exact():
    fx = _fixture()
    idx = np.arange(fx["Yho"].shape[1])
    got = msr.oracle_correct(np.zeros_like(fx["Yho"]), fx["Yho"], idx)
    assert msr.mean_l2_ratio(got, fx["Yho"]) == pytest.approx(0.0, abs=1e-12)


def test_build_emits_sealed_receipt():
    out = msr.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "MLP_SPARSE_RESIDUAL.json"
    assert doc["schema"] == msr.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc
    assert doc["selftest"]["unbilled_residual_index_refused"] is True
    assert doc["selftest"]["held_out_leak_refused"] is True
    assert doc["metric"]["authority"] == "held_out_relative_l2"
    assert doc["corpus"]["split_unit"] == "prompt_id"
    assert doc["corpus"]["disjoint"] is True
    assert doc["consuming_primitive"]["name"] in ATLAS_PRIMITIVES
    assert doc["consuming_primitive"]["index_dtype_bytes"] == msr.INDEX_BYTES


def test_receipt_reports_both_allocations_and_held_out_by_prompt():
    out = msr.build()
    doc = json.loads(out.read_text())
    assert doc["bulks"]
    ids = {b["id"] for b in doc["bulks"]}
    assert "shared_input_r64" in ids
    assert "shared_output_r64" in ids
    assert "shared_both_r64" in ids
    assert "oracle_pca_r64" in ids
    allocs = set()
    for bulk in doc["bulks"]:
        assert bulk["held_out_split"] == "hold"
        assert bulk["error_authority"] == "held_out_relative_l2"
        assert "concentration" in bulk
        axes = bulk["concentration"]["axes"]
        for name in ("output_coords", "tokens", "w_blocks_output"):
            assert name in axes
            assert "curve" in axes[name]
            assert "frac_energy_at_1pct" in axes[name]
        assert "input_directions" in axes
        assert "map_value_vs_uniform" in bulk
        for point in bulk["budget_sweep"]:
            allocs.add(point["allocation"])
            assert point["held_out_split"] == "hold"
            assert point["error_authority"] == "held_out_relative_l2"
            assert "held_out_relative_l2" in point
            assert "train_relative_l2_diagnostic" in point
            assert point["train_split"] == "train"
            assert "held_out_relative_l2" in point
            assert "train_relative_l2_diagnostic" in point
            msr.validate_error_authority(point)
            msr.validate_billing(point)
            if int(point["k"]) > 0:
                assert point["byte_breakdown"]["index_bytes"] > 0
                assert point["dispatch_delta"] > 0.0
            if bulk["billed"] and int(point["k"]) > 0:
                assert point["economics"] is not None
                assert point["economics"]["bytes_added"]["metadata"] >= point["byte_breakdown"][
                    "index_bytes"
                ]
                assert point["economics"]["dispatch_delta"] > 0.0
                assert point["economics"]["consuming_primitive"] in ATLAS_PRIMITIVES
            if not bulk["billed"]:
                assert point["economics"] is None
    assert msr.UNIFORM in allocs
    assert msr.CAPABILITY in allocs
    assert doc["school"]["status"] in {
        msr.RESIDUAL_DENSE_CLOSED,
        msr.RESIDUAL_RESCUE_CLOSED,
        msr.OPEN,
    }
    assert "is_the_residual_after_the_best_bulk_sparse" in doc["answers"]


def test_residual_map_uses_cholesky_back_substitution_not_a_general_solve():
    """L is triangular; solving it with a general LU is the wrong algorithm.

    factor["L"] comes from np.linalg.cholesky. np.linalg.solve would ask LAPACK
    for a general LU factorization with pivoting -- O(n^3) to answer what
    back-substitution answers in O(n^2), twice per call. This pins BOTH facts:
    the result must match a general solve (it is the same system), and the
    routine must be the triangular one (it is the same answer for less work).
    """
    import numpy as np

    from tools.future import mlp_sparse_residual as m

    rng = np.random.default_rng(17)
    n, k, c = 64, 9, 5
    x = rng.standard_normal((n, k))
    g = x.T @ x + np.eye(k) * 1e-3
    factor = {"L": np.linalg.cholesky(g), "X": x}
    r_cols = rng.standard_normal((n, c))

    got = m.solve_residual_map(factor, r_cols)

    # Independent reference: the general solve this replaced.
    rhs = factor["X"].T @ np.ascontiguousarray(r_cols, dtype=np.float64)
    ref = np.linalg.solve(factor["L"].T, np.linalg.solve(factor["L"], rhs))
    assert np.allclose(got, ref, atol=1e-9), "the triangular solve changed the answer"

    # And it really solves the normal equations: g @ result == X.T @ r_cols.
    assert np.allclose(g @ got, rhs, atol=1e-8), "the result does not satisfy the system"
