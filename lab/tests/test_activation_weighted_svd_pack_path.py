"""Codec + surplus-first selection coverage for activation_weighted_svd_low_rank_q."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import lab.operators.ascension_dual_gravity_worker as dual
from lab.operators.ascension_dual_gravity_worker import (
    ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
    EXPANDED_REPRESENTATIONS,
    MAGIC_ACT_SVD,
    Proposal,
    Target,
    _activation_capture_binding,
    _activation_weighted_svd_low_rank_codec,
    _decode_activation_weighted_svd_low_rank_codec,
    _encode,
    _parse_container,
    _representation_config,
)
from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    BUDGET_POINTS,
    select_budget_for_organ,
)


def test_activation_weighted_family_is_first_class_but_not_in_v2_auto_schedule() -> None:
    assert ACTIVATION_WEIGHTED_SVD_REPRESENTATION not in EXPANDED_REPRESENTATIONS
    assert ACTIVATION_WEIGHTED_SVD_REPRESENTATION in dual.EXPLICIT_PACK_REPRESENTATIONS
    config = _representation_config(ACTIVATION_WEIGHTED_SVD_REPRESENTATION, generation=1)
    assert config["selection_metric"] == "surplus_over_null"
    assert config["weight_cosine_role"] == "secondary_and_distribution_local_guard"
    assert config["activation_capture"]["sha256"]
    assert config["activation_capture"]["not_synthetic_unit_direction"] is True
    assert config["activation_capture"]["fit_kind"] == "real_routed_activation_capture"


def test_activation_weighted_codec_binds_capture_and_decodes_physical_bytes() -> None:
    rng = np.random.default_rng(7)
    # Low-rank plant so the fit has a real surplus over a constant-mean null.
    true_left = rng.standard_normal((32, 4), dtype=np.float32)
    true_right = rng.standard_normal((4, 48), dtype=np.float32)
    W = true_left @ true_right
    X = rng.standard_normal((64, 48), dtype=np.float32)
    capture = {
        "path": "/tmp/fake-capture",
        "capture_result_path": "/tmp/fake-capture/capture-result.json",
        "sha256": "a" * 64,
        "schema": "test.capture.v1",
        "status": "TEST",
        "fit_kind": "real_routed_activation_capture",
        "not_synthetic_unit_direction": True,
    }
    codec = _activation_weighted_svd_low_rank_codec(
        W, rank=4, bits=4, X_fit=X[:48], capture_identity=capture, X_hold=X[48:]
    )
    header, body = _parse_container(codec.payload, expected_magic=MAGIC_ACT_SVD)
    decoded = _decode_activation_weighted_svd_low_rank_codec(codec.payload)

    assert header["schema"] == "hawking.gravity.activation_weighted_svd_low_rank.v1"
    assert header["representation"] == ACTIVATION_WEIGHTED_SVD_REPRESENTATION
    assert header["activation_capture"]["sha256"] == "a" * 64
    assert header["selection_metric"]["primary"] == "surplus_over_null"
    assert header["activation_quality"]["surplus_over_null"] is not None
    assert len(body) == header["left_body_bytes"] + header["right_body_bytes"]
    assert codec.reconstruction.shape == W.shape
    np.testing.assert_allclose(decoded, codec.reconstruction, rtol=1e-5, atol=1e-5)


def test_activation_weighted_codec_refuses_synthetic_capture_binding() -> None:
    W = np.eye(8, dtype=np.float32)
    X = np.eye(8, dtype=np.float32)
    with pytest.raises(dual.DualGravityError, match="synthetic"):
        _activation_weighted_svd_low_rank_codec(
            W,
            rank=2,
            bits=3,
            X_fit=X,
            capture_identity={
                "path": "x",
                "sha256": "b" * 64,
                "fit_kind": "synthetic_unit_direction",
            },
        )


def test_encode_requires_real_activation_rows_for_activation_weighted_family() -> None:
    values = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 100.0
    capture = _representation_config(ACTIVATION_WEIGHTED_SVD_REPRESENTATION, generation=0)
    proposal = Proposal(
        sequence=0,
        generation=0,
        target=Target("unit.weight", "routed_expert_gate", True, "unit"),
        representation=ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
        config=capture,
        candidate_id="unit-act-svd",
    )
    with pytest.raises(dual.DualGravityError, match="activation_rows"):
        _encode(values, proposal)

    X = np.random.default_rng(1).standard_normal((20, 16), dtype=np.float32)
    codec, training = _encode(values, proposal, activation_rows=X[:15], activation_hold_rows=X[15:])
    assert codec.metadata["schema"] == "hawking.gravity.activation_weighted_svd_low_rank.v1"
    assert training["status"] == "FIT_ON_BOUND_REAL_ACTIVATION_CAPTURE"
    assert training["selection_metric"] == "surplus_over_null"


def test_select_budget_prefers_surplus_over_weight_cosine(tmp_path: Path) -> None:
    """Selection metric is surplus-over-null, not weight cosine.

    Geometry matches a Q30 routed expert projection (768×2048) so under-ceiling
    low-rank budgets are reachable the same way the measured probe observed.
    """
    rng = np.random.default_rng(11)
    out_dim, in_dim, plant_rank = 768, 2048, 32
    true_left = rng.standard_normal((out_dim, plant_rank), dtype=np.float32)
    true_right = rng.standard_normal((plant_rank, in_dim), dtype=np.float32)
    W = true_left @ true_right
    X = rng.standard_normal((96, in_dim), dtype=np.float32)
    # Codec only checks fit_kind + sha presence for this unit-test identity.
    capture = {
        "path": str(tmp_path),
        "capture_result_path": str(tmp_path / "capture-result.json"),
        "sha256": "c" * 64,
        "schema": "test",
        "status": "TEST",
        "fit_kind": "real_routed_activation_capture",
        "not_synthetic_unit_direction": True,
    }
    X_fit, X_hold = X[:72], X[72:]
    winner = select_budget_for_organ(W=W, X_fit=X_fit, X_hold=X_hold, capture_identity=capture)
    assert winner["under_ceiling"] is True
    assert winner["component_bpw"] <= 1.5 + 1e-9
    assert winner["budget_label"] in {b["label"] for b in BUDGET_POINTS}
    assert "surplus_over_null" in winner
    assert "weight_cosine" in winner
    # Every under-ceiling sweep row should not beat the winner on surplus.
    under = [r for r in winner["sweep"] if r["under_ceiling"]]
    assert winner["surplus_over_null"] == max(r["surplus_over_null"] for r in under)


def test_real_capture_binding_when_asset_present() -> None:
    capture_run = dual.DEFAULT_Q30_ACTIVATION_CAPTURE_RUN
    if not (capture_run / "capture-result.json").is_file():
        pytest.skip("broad activation capture not present on this host")
    bound = _activation_capture_binding(capture_run)
    assert len(bound["sha256"]) == 64
    assert bound["fit_kind"] == "real_routed_activation_capture"
    # Mismatch must fail closed.
    with pytest.raises(dual.DualGravityError, match="sha256 mismatch"):
        _activation_capture_binding(capture_run, capture_result_sha256="0" * 64)
