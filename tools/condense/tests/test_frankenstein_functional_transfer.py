"""Functional-transfer scaffold: aligner, cartography, gates, adapters, promotion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_ablation as ablation  # noqa: E402
from lab.operators import frankenstein_aligner as aligner  # noqa: E402
from lab.operators import frankenstein_baseline_freeze as baseline  # noqa: E402
from lab.operators import frankenstein_bridges as bridges  # noqa: E402
from lab.operators import frankenstein_cartography as carto  # noqa: E402
from lab.operators import frankenstein_functional_transfer as ft  # noqa: E402
from lab.operators import frankenstein_gates as gates  # noqa: E402
from lab.operators import frankenstein_promotion_gate as promo  # noqa: E402
from lab.operators import frankenstein_trace_format as traces  # noqa: E402
from lab.operators import frankenstein_transfer as xfer  # noqa: E402
from lab.operators import frankenstein_verifier_loop as vloop  # noqa: E402
from lab.operators.frankenstein_gates import (  # noqa: E402
    LINEAR_SUBSPACE_INITIALIZATION,
    REQUIRES_GLM_RUNTIME,
    REQUIRES_TRAINING_LOOP,
    REQUIRES_VERIFIER,
)
from lab.receipts import verify  # noqa: E402


# ---------------------------------------------------------------------------
# Relabel: linear init, never PROTO complete
# ---------------------------------------------------------------------------


def test_linear_module_role_is_initialization_not_proto() -> None:
    extraction = xfer.extract_from_synthetic_weights(rank=8, seed=7)
    module = xfer.build_transfer_module(extraction=extraction)
    assert module["role"] == LINEAR_SUBSPACE_INITIALIZATION
    assert module["proto_frankenstein_complete"] is False
    assert module["inheritance_status"] == "NOT_INHERITANCE_LINEAR_INIT_ONLY"
    assert module["capability_claim"] is False
    assert module["capability_status"] == "UNVALIDATED_WEIGHT_ONLY_DERIVED"
    boundary = module["claim_boundary_linear_init"]
    assert boundary["is_inheritance"] is False
    assert boundary["is_proto_frankenstein"] is False


def test_forbidden_proto_complete_label_refused() -> None:
    with pytest.raises(ValueError, match="LINEAR_SUBSPACE_INITIALIZATION"):
        gates.assert_not_proto_complete("PROTO_FRANKENSTEIN_COMPLETE")


# ---------------------------------------------------------------------------
# Aligner: fixtures without token-ID matching
# ---------------------------------------------------------------------------


def test_aligner_decoded_spans_on_fixtures() -> None:
    left = [
        traces.make_decoded_span(text="use induction", byte_start=0, byte_end=13),
        traces.make_decoded_span(text="base case n=0", byte_start=14, byte_end=27),
        traces.make_decoded_span(text="inductive step", byte_start=28, byte_end=42),
    ]
    right = [
        traces.make_decoded_span(text="Use Induction", byte_start=0, byte_end=13),
        traces.make_decoded_span(text="inductive step", byte_start=40, byte_end=54),
        traces.make_decoded_span(text="base case n = 0", byte_start=20, byte_end=35),
    ]
    aligned = aligner.align_decoded_spans(left, right, min_score=0.5)
    assert len(aligned) >= 2
    methods = {a["method"] for a in aligned}
    assert methods == {"decoded_span_text"}
    # Must not require equal token ids / equal byte ranges across tokenizers.
    texts = {(a["left_text"].lower(), a["right_text"].lower()) for a in aligned}
    assert any("induction" in l and "induction" in r for l, r in texts)


def test_aligner_refuses_token_id_sequences() -> None:
    left = [{"token_id": 101}, {"token_id": 202}]
    right = [{"token_id": 55}, {"token_id": 66}]
    with pytest.raises(aligner.AlignerError, match="token-ID"):
        aligner.align_decoded_spans(left, right)


def test_aligner_formal_actions_and_tools() -> None:
    left_actions = [
        traces.make_formal_action(action_type="lemma", payload={"name": "L1"}),
        traces.make_formal_action(action_type="apply", payload={"thm": "nat.add_comm"}),
    ]
    right_actions = [
        traces.make_formal_action(action_type="apply", payload={"thm": "nat.add_comm"}),
        traces.make_formal_action(action_type="lemma", payload={"name": "L1"}),
    ]
    aa = aligner.align_formal_actions(left_actions, right_actions)
    assert len(aa) == 2
    assert all(a["score"] >= 0.6 for a in aa)

    left_tools = [traces.make_tool_event(tool_name="lean_check", args={"goal": "1"})]
    right_tools = [traces.make_tool_event(tool_name="lean_check", args={"goal": "1", "x": 2})]
    ta = aligner.align_tool_events(left_tools, right_tools)
    assert len(ta) == 1
    assert ta[0]["tool_name"] == "lean_check"


def test_align_paired_sides_report() -> None:
    left = {
        "decoded_spans": [
            traces.make_decoded_span(text="hello world", byte_start=0, byte_end=11)
        ],
        "formal_actions": [
            traces.make_formal_action(action_type="prove", payload={"n": 1})
        ],
        "tool_events": [],
    }
    right = {
        "decoded_spans": [
            traces.make_decoded_span(text="hello world", byte_start=0, byte_end=11)
        ],
        "formal_actions": [
            traces.make_formal_action(action_type="prove", payload={"n": 1})
        ],
        "tool_events": [],
    }
    report = aligner.align_paired_sides(left, right)
    forbidden = report["method_policy"]["forbidden"]
    assert "token_ids" in forbidden
    assert "vocab_index_match" in forbidden
    assert "token_id_to_token_id" in forbidden
    assert report["summary"]["n_span_alignments"] == 1
    assert report["summary"]["n_action_alignments"] == 1


# ---------------------------------------------------------------------------
# Cartography on synthetic paired matrices
# ---------------------------------------------------------------------------


def test_cka_and_procrustes_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 16))
    # Linear CKA is invariant to orthogonal transforms + isotropic scale.
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    y_scaled = (x @ q) * 2.5
    y_ortho = x @ q
    cka = carto.linear_cka(x, y_scaled)
    assert cka > 0.99
    # Orthogonal Procrustes (no scale) matches y_ortho closely.
    proc = carto.procrustes_similarity(x, y_ortho)
    assert proc["relative_residual_energy"] < 0.05
    assert proc["correlation"] > 0.99
    cca = carto.cca_similarity(x, y_scaled, n_components=4)
    assert cca["mean_top_k"] > 0.9
    # Unrelated noise → low CKA
    z = rng.standard_normal((128, 16))
    assert carto.linear_cka(x, z) < 0.3


def test_correspondence_matrix_recovers_planted_map() -> None:
    glm, dsv = carto.synthetic_paired_layers(
        n_glm=6, n_dsv=4, n_samples=80, d_glm=24, d_dsv=16, seed=1
    )
    mat = carto.correspondence_matrix(glm, dsv, metric="cka")
    assert mat.shape == (6, 4)
    phases = carto.functional_phase_map(mat, glm_layer_count=6, dsv4f_layer_count=4)
    assert phases["ratio_map_rejected"] is True
    # Each dsv layer should prefer its planted glm source (approx).
    for j in range(4):
        src = min(int(j * 6 / 4), 5)
        assert int(np.argmax(mat[:, j])) == src

    report = carto.build_correspondence_report(glm, dsv, metric="cka", source="synthetic")
    verify(report, label="cartography")
    assert report["status"] == "OK"
    assert report["fabricated"] is False
    assert report["ratio_map_rejected"] is True


def test_live_glm_cartography_fails_closed() -> None:
    report = carto.build_correspondence_report(
        None,
        None,
        source="live_glm",
        glm_runtime_available=False,
    )
    verify(report, label="cartography gated")
    assert report["status"] == "FAIL_CLOSED"
    assert report["gate"] == REQUIRES_GLM_RUNTIME
    assert report["executed"] is False
    assert report["fabricated"] is False


# ---------------------------------------------------------------------------
# Trace format + membership + capture gate
# ---------------------------------------------------------------------------


def test_membership_disjoint_and_trace_validate() -> None:
    mgr = traces.MembershipManager()
    mgr.assign("ex-1", "train")
    mgr.assign("ex-2", "hidden_test")
    with pytest.raises(traces.TraceFormatError, match="disjoint"):
        mgr.assign("ex-1", "public_test")
    sealed = mgr.seal_document()
    verify(sealed, label="membership")
    assert sealed["disjoint"] is True
    assert sealed["counts"]["train"] == 1

    tr = traces.build_paired_trace(
        example_id="ex-1",
        membership="train",
        prompt_text="Prove 1+1=2",
        decoded_spans=[
            traces.make_decoded_span(text="Prove 1+1=2", byte_start=0, byte_end=11)
        ],
        formal_actions=[
            traces.make_formal_action(action_type="goal", payload={"s": "1+1=2"})
        ],
        dsv4f_side={**traces.empty_side("dsv4f"), "present": True},
    )
    verify(tr, label="trace")
    assert tr["alignment_policy"]["never_align_on"] == [
        "token_ids",
        "incompatible_vocab_indices",
    ]
    assert tr["fabricated"] is False


def test_glm_capture_fails_closed_without_runtime() -> None:
    result = traces.capture_glm_trajectory(
        example_id="x",
        prompt_text="p",
        membership="train",
        glm_runtime=None,
    )
    assert result["status"] == "FAIL_CLOSED"
    assert result["gate"] == REQUIRES_GLM_RUNTIME
    assert result["executed"] is False
    assert result["fabricated"] is False

    paired = traces.capture_paired_evidence(
        example_id="x",
        prompt_text="p",
        membership="calibration",
        glm_runtime=None,
    )
    verify(paired, label="partial pair")
    assert paired["meta"]["complete_pair"] is False
    assert paired["sides"]["glm"]["capture_status"] == "FAIL_CLOSED"


# ---------------------------------------------------------------------------
# Bridges / adapters: apply+revert + gravity accounting + train gate
# ---------------------------------------------------------------------------


def test_reversible_bridge_apply_revert() -> None:
    bridge = bridges.ReversibleBridge.initialize(
        d_model=64, d_hidden=32, rank=4, seed=3, scale=0.15
    )
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 64))
    y = bridge.apply(x)
    assert y.shape == x.shape
    assert float(np.max(np.abs(y - x))) > 1e-3
    x_back, info = bridge.revert(y, n_iters=24, atol=1e-6)
    assert info["recon_error"] < 1e-3
    assert np.allclose(x_back, x, atol=1e-3)
    grav = bridge.gravity_accounting()
    assert grav["parameter_bytes"] > 0
    assert grav["hash_bound"] == bridge.content_hash()
    assert grav["ablatable"] is True
    spec = bridge.to_spec()
    verify(spec, label="bridge spec")
    assert spec["trained"] is False
    assert spec["linear_init_role"] == LINEAR_SUBSPACE_INITIALIZATION


def test_adapter_bank_apply_revert_and_byte_accounting() -> None:
    bank = bridges.build_adapter_bank(d_model=64, rank=4, seed=2)
    verify({k: v for k, v in bank.items() if not k.startswith("_")}, label="bank")
    assert bank["trained"] is False
    assert bank["glm_router_weights_copied"] is False
    assert bank["total_parameter_bytes"] > 0
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 64))
    y, meta = bridges.apply_adapter_bank(x, bank)
    assert meta["applied"]
    x_back = bridges.revert_adapter_bank(y, bank, applied=meta["applied"])
    assert np.allclose(x_back, x, atol=1e-5)
    # Independent ablation: skip one adapter
    y2, meta2 = bridges.apply_adapter_bank(
        x, bank, skip=["GLM_METHOD_ADAPTER"]
    )
    assert "GLM_METHOD_ADAPTER" not in meta2["applied"]


def test_fit_bridge_fails_closed_without_training_loop() -> None:
    bridge = bridges.ReversibleBridge.initialize(d_model=32, d_hidden=16, rank=2, seed=0)
    result = bridges.fit_bridge(bridge)
    assert result["status"] == "FAIL_CLOSED"
    assert result["gate"] == REQUIRES_TRAINING_LOOP
    assert result["executed"] is False
    assert result["trained"] is False
    assert result["fabricated"] is False

    bank = bridges.build_adapter_bank(d_model=32, rank=2, seed=0)
    result2 = bridges.fit_adapters(bank)
    assert result2["status"] == "FAIL_CLOSED"
    assert result2["gate"] == REQUIRES_TRAINING_LOOP


# ---------------------------------------------------------------------------
# Verifier loop gate
# ---------------------------------------------------------------------------


def test_verifier_loop_fails_closed() -> None:
    spec = vloop.loop_interface_spec()
    verify(spec, label="verifier interface")
    assert spec["status"] == "INTERFACE_ONLY"
    result = vloop.run_verified_expert_iteration(problem={"id": "p0"})
    verify(result, label="verifier run")
    assert result["status"] == "FAIL_CLOSED"
    assert result["gate"] == REQUIRES_VERIFIER
    assert result["executed"] is False
    assert result["fabricated"] is False


# ---------------------------------------------------------------------------
# A–G ablation + promotion PENDING
# ---------------------------------------------------------------------------


def test_ag_ablation_framework_pending() -> None:
    report = ablation.run_ag_ablation(arm_scores=None)
    verify(report, label="ag ablation")
    assert report["verdict"] == "PENDING"
    assert report["fabricated_scores"] is False
    assert len(report["arms"]) == 7
    assert report["arms"][1]["arm"] == ablation.ARM_FT_B
    assert "LINEAR" in report["note"] or report["claim_boundary"][
        "proto_complete_from_linear"
    ] is False


def test_ag_ablation_reject_on_secondary_and_imitation() -> None:
    base = ablation.default_score_template(0.70)
    good = ablation.default_score_template(0.70)
    for d in ablation.MATH_DOMAINS:
        good["math"][d] = 0.85
    bad = ablation.default_score_template(0.70)
    for d in ablation.MATH_DOMAINS:
        bad["math"][d] = 0.95
    bad["secondary"]["coding_and_repository_work"] = 0.50

    scores = {
        ablation.ARM_FT_A: {**base, "bench_scope": "FIXTURE"},
        ablation.ARM_FT_B: {**good, "bench_scope": "FIXTURE"},
        ablation.ARM_FT_C: {
            **good,
            "bench_scope": "FIXTURE",
            "imitation_only_without_proof": True,
        },
        ablation.ARM_FT_D: {**bad, "bench_scope": "FIXTURE"},
    }
    report = ablation.run_ag_ablation(arm_scores=scores)
    verify(report, label="ag scored")
    assert report["reject_rule_fired"] is True
    assert report["verdict"] == "REJECT"
    by_arm = {r["arm"]: r for r in report["arms"]}
    assert by_arm[ablation.ARM_FT_C]["verdict"] == "REJECT"
    assert by_arm[ablation.ARM_FT_D]["verdict"] == "REJECT"
    assert by_arm[ablation.ARM_FT_B]["verdict"] == "ACCEPT"


def test_promotion_gate_returns_pending_honestly() -> None:
    result = promo.evaluate_promotion()
    verify(result, label="promotion")
    assert result["verdict"] == "PENDING"
    assert result["fabricated_scores"] is False
    assert result["claim_boundary"]["linear_init_cannot_promote"] is True
    assert result["targets"]["math_gap_recovery_min"] == 0.70
    assert "PENDING" in result["reason"] or "incomplete" in result["reason"].lower()
    # Explicit FAIL scores → REJECT
    reject = promo.evaluate_promotion(
        scores={
            "math": {"gap_recovery": 0.2},
            "secondary": {
                axis: {"gate": "PASS"} for axis in promo.SECONDARY_AXES
            },
            "hidden_eval": {"pass": True},
            "proof_computation_repair": {"pass": False},
        },
        provenance={"complete": True},
        gravity_accounting={"parameter_bytes": 1, "tps": 1.0, "p99": 10.0},
        routing={"stable": True},
        independent_challenge={"pass": True},
        ablation_verdict="ACCEPT",
    )
    assert reject["verdict"] == "REJECT"


def test_baseline_freeze_measurable_fields() -> None:
    doc = baseline.freeze_base_dsv4f_baseline(write=False)
    verify(doc, label="baseline")
    assert doc["name"] == "BASE_DSV4F"
    assert doc["student_body"]["num_hidden_layers"] == 43
    assert doc["runtime"]["tps_mean"]["status"] == "PENDING"
    assert doc["claim_boundary"]["fabricated_scores"] is False
    assert doc["role_of_linear_mapping"] == LINEAR_SUBSPACE_INITIALIZATION


def test_program_seal_and_inventory() -> None:
    doc = ft.build_program_document()
    verify(doc, label="program")
    assert doc["status"] == "SCAFFOLD_SEALED"
    assert doc["proto_frankenstein_complete"] is False
    assert doc["linear_mapping"]["role"] == LINEAR_SUBSPACE_INITIALIZATION
    assert doc["promotion_gate"]["current_verdict"] == "PENDING"
    assert len(doc["seven_layer_transfer"]) == 7
    assert len(doc["ablation_ag"]["arms"]) == 7
    assert REQUIRES_GLM_RUNTIME in doc["missing_infra"]
    assert REQUIRES_TRAINING_LOOP in doc["missing_infra"]
    assert REQUIRES_VERIFIER in doc["missing_infra"]
    assert doc["fabricated_training_or_eval"] is False


def test_secondary_suite_framework() -> None:
    suite = promo.secondary_non_regression_suite_framework()
    verify(suite, label="secondary suite")
    assert suite["status"] == "FRAMEWORK_ONLY"
    assert suite["fabricated_scores"] is False
