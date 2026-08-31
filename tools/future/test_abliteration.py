"""Tests for abliteration as a candidate transformation generator.

Negative controls are the point: an artifact that claims refusals were removed
must be refused, a direction selected on refusal suppression alone must be
underdetermined, a plan naming a specimen that is not whole-tree verified must
be refused, and no code path may write a hardware or capability number.

Never skip. If the verification receipt is absent, plan() must refuse and the
test asserts that refusal. Sparse checkout is not evidence of absence.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from hcli.workunit import WorkUnit, is_ready
from tools.future import abliteration as ab
from tools.future import tabula as tb
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)
from tools.future.ebpw_categories import CategoryError


def _evals(completion: str, harmless: str, loss: str) -> dict[str, Any]:
    return {
        "completion": {"gate": completion},
        "harmless": {"gate": harmless},
        "loss": {"gate": loss},
    }


def _verified_row(name: str, *, bytes_hashed: int = 1519209243, n_files: int = 10) -> dict[str, Any]:
    return {
        "specimen": name,
        "status": "WHOLE_TREE_VERIFIED",
        "whole_tree_verified": True,
        "n_files": n_files,
        "verified": n_files,
        "mismatched": 0,
        "no_remote_digest": 0,
        "unrecognized_digest": 0,
        "skipped_time_budget": 0,
        "bytes_hashed": bytes_hashed,
        "owner": "modellake",
    }


def _partial_row(name: str) -> dict[str, Any]:
    return {
        "specimen": name,
        "status": "PARTIAL_NO_REMOTE_DIGEST",
        "whole_tree_verified": False,
        "n_files": 4,
        "verified": 3,
        "mismatched": 0,
        "no_remote_digest": 1,
        "unrecognized_digest": 0,
        "skipped_time_budget": 0,
        "bytes_hashed": 100,
        "owner": "modellake",
    }


def test_build_emits_sealed_static_receipt():
    out = ab.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ABLITERATION.json"
    assert doc["schema"] == "hawking.future.abliteration.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["weights_modified"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert "tools/future/tabula.py" in " ".join(doc["recovered_implementation"])
    rc = doc["resident_callable"]
    assert rc["entry_point"] == "tools.future.abliteration.method()"
    assert rc["receipt"] == "receipts/future/ABLITERATION.json"
    assert rc["frontier"] == "FT.MODEL_CAPABILITY.hard-gates"
    assert rc["this_lane_writes_frontier"] is False
    assert "CPU_ANALYSIS" in rc["workunit"]
    assert "future.abliteration.method" in rc["workunit_emitted"]
    assert "future.abliteration.run-specimen" in rc["workunit_emitted"]
    _assert_no_hardware_claims(doc)


def test_method_recovers_generation_selection_projection_scoping():
    m = ab.method()
    ids = [s["id"] for s in m["stages"]]
    assert ids[:4] == [
        "generate_directions",
        "select_direction",
        "projection",
        "layer_scoping",
    ]
    assert m["required_evals"] == ["completion", "harmless", "loss"]
    assert m["not_a_rival_floor"] is True
    assert m["belongs_to"].startswith("Tabula")
    assert "andyrdt/refusal_direction" in m["upstream"]["scientific"]["repo"]
    assert "NousResearch/llm-abliteration" in m["upstream"]["operational"]["repo"]
    assert "FailSpy/abliterator" in m["upstream"]["workbench"]["repo"]
    assert "does not guarantee" in m["upstream"]["operational"]["guarantee"].lower()
    assert m["recovered_method_defaults"]["measured_here"] is False
    assert m["weights_modified"] is False
    assert m["gpu_authority"] is False


def test_contracts_require_both_sets_and_three_evals():
    c = ab.contracts()
    assert c["dataset"]["harmful_set"]["required"] is True
    assert c["dataset"]["harmless_set"]["required"] is True
    assert c["dataset"]["both_required"] is True
    assert c["dataset"]["harmful_only_is_underdetermined"] is True
    for name in ("completion_eval", "harmless_eval", "loss_eval"):
        assert c["evaluation"][name]["required"] is True
    assert "REFUSE" in c["refusal_contracts"]["empty_harmful_after_filter"]
    assert c["projection"]["distinct_norm_preserves"] is True
    assert c["outer_scorer"]["module"] == "tools.future.tabula.evaluate"
    assert c["outer_scorer"]["axes"] == list(tb.SCORE_AXES)


def test_claim_boundary_forbids_uncensoring_switch():
    b = ab.claim_boundary()
    assert "candidate direction" in b["object"]
    joined = " ".join(b["must_not_assert"]).lower()
    assert "abliterated" in joined
    assert "refusals were removed" in joined
    assert "does not guarantee" in b["upstream_quote"].lower()
    assert b["status_labels_are_hypotheses"] is True
    assert b["gpu_authority"] is False


def test_negative_artifact_claiming_refusals_removed_is_refused():
    with pytest.raises(ab.ClaimBoundaryError, match="refusals_removed"):
        ab.admit_result({"status": "ok", "refusals_removed": True})
    with pytest.raises(ab.ClaimBoundaryError, match="uncensoring switch"):
        ab.admit_result({"status": "abliterated"})
    with pytest.raises(ab.ClaimBoundaryError, match="uncensored"):
        ab.admit_result({"status": "uncensored"})
    with pytest.raises(ab.ClaimBoundaryError, match="refusals were removed"):
        ab.admit_result({"status": "CANDIDATE", "notes": "refusals were removed"})
    with pytest.raises(ab.ClaimBoundaryError, match="gpu_authority"):
        ab.admit_result({"status": "CANDIDATE", "gpu_authority": True})
    ok = ab.admit_result(
        {
            "status": "CANDIDATE",
            "selected_id": "cand-mid",
            "projection": "projected",
            "gpu_authority": False,
        }
    )
    assert ok["admitted"] is True
    assert ok["not_as"] == "uncensoring_switch"


def test_negative_direction_selected_on_refusal_suppression_alone():
    with pytest.raises(ab.UnderdeterminedSelection, match="refusal suppression alone"):
        ab.select_by_refusal_suppression_alone([{"id": "d0"}])
    with pytest.raises(ab.UnderdeterminedSelection, match="underdetermined"):
        ab.select([{"id": "d0", "evals": {"completion": {"gate": "PASS"}}}])
    with pytest.raises(ab.MissingEvalError, match="harmless"):
        ab.EvalBundle.from_mapping({"completion": {"gate": "PASS"}, "loss": {"gate": "PASS"}})
    with pytest.raises(ab.MissingEvalError, match="loss"):
        ab.EvalBundle.from_mapping(
            {"completion": {"gate": "PASS"}, "harmless": {"gate": "PASS"}}
        )
    with pytest.raises(ab.MissingEvalError, match="PASS/FAIL"):
        ab.EvalBundle.from_mapping(
            {
                "completion": {"gate": "PASS"},
                "harmless": {"gate": "PASS"},
                "loss": {},
            }
        )


def test_select_refuses_leftovers_when_every_gate_fails():
    with pytest.raises(ab.SelectionEmpty, match="first-class gate"):
        ab.select(
            [
                {
                    "id": "only-refusal",
                    "source_layer": 4,
                    "n_layers": 12,
                    "evals": _evals("PASS", "FAIL", "FAIL"),
                }
            ]
        )
    with pytest.raises(ab.SelectionEmpty, match="first-class gate"):
        ab.select(
            [
                {
                    "id": "kill-quality",
                    "source_layer": 4,
                    "n_layers": 12,
                    "evals": _evals("PASS", "PASS", "FAIL"),
                }
            ]
        )
    with pytest.raises(ab.SelectionEmpty, match="first-class gate"):
        ab.select(
            [
                {
                    "id": "over-refuse",
                    "source_layer": 4,
                    "n_layers": 12,
                    "evals": _evals("PASS", "FAIL", "PASS"),
                }
            ]
        )


def test_select_prunes_last_fraction_as_source_even_if_gates_pass():
    with pytest.raises(ab.SelectionEmpty, match="source-layer prune"):
        ab.select(
            [
                {
                    "id": "tail",
                    "source_layer": 11,
                    "source_position": -1,
                    "n_layers": 12,
                    "evals": _evals("PASS", "PASS", "PASS"),
                }
            ]
        )


def test_select_surviving_candidate_is_a_tie_break_not_a_quality_rank():
    mid = {
        "id": "cand-mid",
        "source_layer": 4,
        "source_position": -1,
        "n_layers": 12,
        "evals": _evals("PASS", "PASS", "PASS"),
    }
    earlier = {
        "id": "cand-early",
        "source_layer": 3,
        "source_position": -1,
        "n_layers": 12,
        "evals": _evals("PASS", "PASS", "PASS"),
    }
    tail = {
        "id": "cand-tail",
        "source_layer": 11,
        "source_position": -1,
        "n_layers": 12,
        "evals": _evals("PASS", "PASS", "PASS"),
    }
    out = ab.select([tail, mid, earlier])
    assert out["selected_id"] == "cand-early"
    assert out["n_surviving"] == 2
    assert "cand-tail" in out["discarded"]
    assert out["ranking_by_refusal_suppression"] is False
    assert out["tie_break_is_not_measured_quality"] is True
    assert out["gpu_authority"] is False
    assert "uncensoring" in out["claim"]
    assert out["claim"].startswith("candidate direction")


def test_negative_plan_unverified_specimen_is_refused():
    fake = {"results": [_partial_row("foo@abc")]}
    with pytest.raises(ab.PlanRefusal, match="whole-tree verified"):
        ab.plan("foo@abc", verification=fake, scars=[])
    with pytest.raises(ab.PlanRefusal, match="whole-tree verified"):
        ab.plan("missing@000", verification=fake, scars=[])


def test_plan_unlocated_verification_is_refused_not_skipped(monkeypatch):
    monkeypatch.setattr(ab, "load_verification_doc", lambda: None)
    monkeypatch.setattr(ab, "git", lambda *_a, **_k: "")
    with pytest.raises(ab.PlanRefusal, match="unlocated|unverified"):
        ab.plan("Qwen--Qwen3-0.6B@c1899de289a0")


def test_plan_smallest_eligible_whole_tree_verified():
    fake = {
        "results": [
            _partial_row("ignore-me@1"),
            _verified_row("tencent--HunyuanVideo@6204ad6aea1a", bytes_hashed=23782, n_files=2),
            _verified_row("microsoft--bitnet-b1.58-2B-4T@04c3b9ad9361", bytes_hashed=1187777736, n_files=10),
            _verified_row("Qwen--Qwen3-0.6B@c1899de289a0", bytes_hashed=1519209243, n_files=10),
            _verified_row(
                "Qwen--Qwen3-0.6B@c1899de289a0#partial",
                bytes_hashed=1519209243,
                n_files=10,
            ),
            _verified_row(
                "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb",
                bytes_hashed=14_000_000_000,
                n_files=20,
            ),
        ]
    }
    pick = ab.smallest_eligible(fake)
    assert pick["specimen"] == "Qwen--Qwen3-0.6B@c1899de289a0"
    planned = ab.plan(verification=fake, scars=[])
    assert planned["specimen"] == "Qwen--Qwen3-0.6B@c1899de289a0"
    assert planned["status"] == "PLAN_ONLY"
    assert planned["ran"] is False
    assert planned["weights_modified"] is False
    assert planned["gpu_authority"] is False
    assert planned["resource_class"] == "GPU_EXCLUSIVE"
    assert planned["sleep_state"] == "SLEEPING"
    assert planned["verification"]["whole_tree_verified"] is True
    assert "Tabula" in planned["outer_scorer"] or "tabula" in planned["outer_scorer"]


def test_plan_refuses_ineligible_architecture_even_when_verified():
    fake = {
        "results": [
            _verified_row("tencent--HunyuanVideo@6204ad6aea1a", bytes_hashed=23782, n_files=2)
        ]
    }
    with pytest.raises(ab.PlanRefusal, match="not eligible"):
        ab.plan("tencent--HunyuanVideo@6204ad6aea1a", verification=fake, scars=[])


def test_quantized_parent_is_a_category_error():
    kind = ab.classify_specimen("microsoft--bitnet-b1.58-2B-4T@04c3b9ad9361")
    assert kind["eligible"] is False
    assert kind["weight_space"] == "quantized"
    with pytest.raises(CategoryError, match="full-weight"):
        ab.require_full_weight(kind)


def test_unclassified_architecture_fails_closed():
    kind = ab.classify_specimen("unknown-lab--mystery-9B@deadbeef")
    assert kind["eligible"] is False
    assert kind["architecture"] == "unclassified"
    fake = {"results": [_verified_row("unknown-lab--mystery-9B@deadbeef")]}
    with pytest.raises(ab.PlanRefusal, match="not eligible"):
        ab.plan("unknown-lab--mystery-9B@deadbeef", verification=fake, scars=[])


def test_named_abliterated_is_a_naming_fact_not_a_result():
    kind = ab.classify_specimen("qwen3.8-27b-abliterated-bf16@local")
    assert kind["named_abliterated"] is True
    assert kind["eligible"] is True
    fake = {"results": [_verified_row("qwen3.8-27b-abliterated-bf16@local", bytes_hashed=55_000_000_000)]}
    planned = ab.plan("qwen3.8-27b-abliterated-bf16@local", verification=fake, scars=[])
    assert planned["named_abliterated_is_not_a_result"] is True
    assert planned["status"] == "PLAN_ONLY"
    with pytest.raises(ab.ClaimBoundaryError):
        ab.admit_result({"status": "abliterated", "specimen": planned["specimen"]})


def test_projection_reuses_tabula_and_distinguishes_row_norm():
    proof = ab.projection_reuses_tabula()
    assert proof["reused"] == "tools.future.tabula.project"
    assert proof["conventional_left_null"] is True
    assert proof["frobenius_restore_error_below_1e8"] is True
    assert proof["row_norm_and_frobenius_are_distinct_operators"] is True
    assert proof["weights_modified_on_a_real_specimen"] is False
    v = np.array([1.0, 0.0, 0.0])
    h = np.array([1.0, 1.0, 0.0])
    out = ab.orthogonalize_against(v, h)
    h_hat = h / np.linalg.norm(h)
    assert abs(float(out @ h_hat)) < 1e-12
    with pytest.raises(ab.UnderdeterminedSelection, match="zero direction"):
        ab.orthogonalize_against(np.zeros(3), h)


def test_scoped_layers_source_and_destination_are_different_knobs():
    dest = ab.scoped_layers(8, role="destination")
    src = ab.scoped_layers(8, role="source")
    assert dest["role"] == "destination"
    assert src["role"] == "source"
    assert 0 in dest["blocked"]
    assert 7 in dest["blocked"]
    assert 7 in src["blocked"]
    assert 0 not in src["blocked"]
    with pytest.raises(ab.PlanRefusal, match="zero layers"):
        ab.scoped_layers(4, role="destination", whitelist=[], blacklist=[0, 1, 2, 3])
    with pytest.raises(ab.PlanRefusal, match="source or destination"):
        ab.scoped_layers(8, role="both")


def test_run_and_weights_are_frozen():
    with pytest.raises(ab.RunRefused, match="does not run"):
        ab.run("Qwen--Qwen3-0.6B@c1899de289a0")
    with pytest.raises(tb.WeightsFrozen):
        ab.apply_to_weights("language_model.model.layers.0.mlp.down_proj.weight")


def test_sleeping_run_unit_is_not_ready_and_round_trips_hcli():
    units = ab.emit_workunits()
    by_id = {row["id"]: row for row in units}
    assert by_id["future.abliteration.method"]["status"] == "pending"
    assert by_id["future.abliteration.select"]["resource_class"] == "STATIC_ANALYSIS"
    run = by_id["future.abliteration.run-specimen"]
    assert run["status"] == "sleeping"
    assert run["classification"] == "SLEEPING"
    assert run["resource_class"] == "GPU_EXCLUSIVE"
    assert run["weights_modified"] is False
    assert run["may_promote"] is False
    mapped = {row["id"]: WorkUnit.from_dict(dict(row)) for row in units}
    assert mapped[run["id"]].status == "sleeping"
    assert is_ready(mapped[run["id"]], mapped) is False
    assert tb.sleeping_unit_is_not_ready(units) is True
    assert is_ready(mapped["future.abliteration.method"], mapped) is True


def test_no_code_path_writes_a_hardware_or_capability_number():
    doc = json.loads(ab.build().read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert not isinstance(doc.get(key), (int, float))
    for key in ("refusal_rate", "refusal_score", "ce_loss", "perplexity", "tps"):
        assert not isinstance(doc.get(key), (int, float))
    assert doc["gpu_authority"] is False
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 12.0})
    with pytest.raises(ab.ClaimBoundaryError, match="capability/hardware"):
        ab.admit_result({"status": "CANDIDATE", "refusal_rate": 0.0})
    with pytest.raises(ab.ClaimBoundaryError, match="capability/hardware"):
        ab.admit_result({"status": "CANDIDATE", "tps": 8})


def test_outer_scorer_is_tabula_not_a_rival():
    """Hitting the behavioural target while destroying tool use remains Tabula FAILURE."""
    vec = tb.ScoreVector(
        behavioral=0.95,
        capability=0.10,
        tool_use=-0.80,
        reasoning=0.05,
        instruction_following=0.04,
    )
    verdict = tb.evaluate(vec)
    assert verdict.outcome == "FAILURE"
    assert "tool_use" in verdict.regressions
    assert ab.contracts()["outer_scorer"]["module"] == "tools.future.tabula.evaluate"


def test_negative_controls_in_build_all_fired():
    doc = json.loads(ab.build().read_text())
    trials = {r["trial"]: r for r in doc["refusals_proven"]}
    for name in (
        "refusals_removed_artifact",
        "status_abliterated",
        "select_by_refusal_suppression_alone",
        "select_missing_harmless_and_loss",
        "select_all_gates_fail",
        "plan_unverified_specimen",
        "quantized_parent",
        "run_refused",
        "apply_to_weights",
    ):
        assert trials[name]["refused"] is True
    assert doc["negative_index"]["invoked"] is True
