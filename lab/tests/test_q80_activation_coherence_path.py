"""Q80 coherence path: readiness, all-layer requirement, surplus-first reuse."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_activation_weighted_svd_repack as q80_repack
from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    select_budget_for_organ,
)
from lab.operators.q80_activation_capture_readiness import assess
import numpy as np


def test_readiness_verdict_is_blocked_and_lists_gqa() -> None:
    design = assess()
    assert design["verdict"] == "ALL_LAYER_ACTIVATION_CAPTURE_NOT_YET_POSSIBLE"
    missing_ids = [m["id"] for m in design["missing_exact"]]
    assert "gqa_full_layer_same_runtime_encode" in missing_ids
    assert "broad_activation_capture_binary" in missing_ids
    assert design["claim_boundary"]["fitting_on_layer0_or_component_captures_is_refused"] is True
    assert design["selection_policy_when_capture_lands"]["primary"] == "surplus_over_null"
    assert design["selection_policy_when_capture_lands"]["require_all_layer_capture"] is True
    bpw = design["baseline_artifact"]["complete_physical_bpw"]
    assert bpw is not None
    assert float(bpw) <= 1.5


def test_repack_refuses_missing_capture(tmp_path: Path) -> None:
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(q80_repack.ActivationWeightedRepackError, match="capture-result"):
        q80_repack.Qwen80ActivationWeightedSvdRepack(
            capture_run=empty,
            root=tmp_path / "out",
            require_all_layer_capture=True,
        ).run()


def test_repack_refuses_partial_layer_capture(tmp_path: Path) -> None:
    run = tmp_path / "l0-only"
    run.mkdir()
    (run / "capture-result.json").write_text(
        json.dumps(
            {
                "schema": "hawking.ascension.qwen80_something_l0_only.v1",
                "status": "TEST",
                "probes": [
                    {
                        "probe_id": "p0",
                        "steps": [
                            {
                                "position": 0,
                                "selected_expert_ids": [0, 1],
                                "router_input_hidden_f32le": {
                                    "relative_path": "h.f32le",
                                    "elements": 4,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "h.f32le").write_bytes(np.zeros(4, dtype="<f4").tobytes())
    with pytest.raises(q80_repack.ActivationWeightedRepackError, match="all-layer"):
        q80_repack.Qwen80ActivationWeightedSvdRepack(
            capture_run=run,
            root=tmp_path / "out",
            require_all_layer_capture=True,
        ).run()


def test_preflight_coverage_missing_capture(tmp_path: Path) -> None:
    cov = q80_repack.preflight_coverage_from_capture(tmp_path / "nope")
    assert cov["status"] == "CAPTURE_MISSING"
    assert cov["cannot_be_coherent"] is True


def test_surplus_first_selection_still_primary() -> None:
    """Reuse measured Q30 geometry: surplus is the selection key under 1.5 BPW."""
    rng = np.random.default_rng(11)
    # Q80 expert gate is 512×2048; plant low-rank so under-ceiling budgets exist.
    out_dim, in_dim, plant_rank = 512, 2048, 32
    W = rng.standard_normal((out_dim, plant_rank), dtype=np.float32) @ rng.standard_normal(
        (plant_rank, in_dim), dtype=np.float32
    )
    X = rng.standard_normal((96, in_dim), dtype=np.float32)
    capture = {
        "path": "/tmp/fake",
        "capture_result_path": "/tmp/fake/capture-result.json",
        "sha256": "d" * 64,
        "schema": "test",
        "status": "TEST",
        "fit_kind": "real_routed_activation_capture",
        "not_synthetic_unit_direction": True,
    }
    winner = select_budget_for_organ(
        W=W, X_fit=X[:72], X_hold=X[72:], capture_identity=capture
    )
    assert winner["under_ceiling"] is True
    assert winner["component_bpw"] <= 1.5 + 1e-9
    under = [r for r in winner["sweep"] if r["under_ceiling"]]
    assert winner["surplus_over_null"] == max(r["surplus_over_null"] for r in under)


def test_null_first_refuses_missing_capture(tmp_path: Path) -> None:
    from lab.operators import q80_activation_null_first_report as null_mod
    import sys

    out = tmp_path / "null.json"
    argv = [
        "q80_activation_null_first_report",
        "--capture-run",
        str(tmp_path / "missing"),
        "--label",
        "test",
        "--out-json",
        str(out),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        code = null_mod.main()
    finally:
        sys.argv = old
    assert code == 2
    doc = json.loads(out.read_text())
    assert doc["status"] == "REFUSED_CAPTURE_MISSING"
    assert doc["headline"]["mean_null_all_scored"] is None
