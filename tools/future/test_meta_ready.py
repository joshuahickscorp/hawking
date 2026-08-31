"""Tests for Flash meta downstream readiness.

The load-bearing negative control: any structural fixture this module can
produce is REFUSED by teacher_corpus.validate_corpus as not-real-capture.
No path through this module advances gate 2 without a genuine corpus.
A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.future import meta_funnel as mf
from tools.future import meta_ready as mr
from tools.future import teacher_corpus as tc
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims, load_json


def test_build_emits_sealed_receipt():
    out = mr.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "META_DOWNSTREAM_READY.json"
    assert doc["schema"] == "hawking.future.meta_ready.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert doc["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "evidence_source" in doc
    assert load_json(out)["schema"] == doc["schema"]


def test_selftest_refuses_structural_fixture():
    result = mr.selftest()
    assert result["structural_fixture_refused_by_validate_corpus"] is True
    assert result["simulate_wired"] is True
    assert result["simulate_gate2_advanced"] is False
    assert result["n_structural_rows"] <= 16


def test_stage_dossiers_gates_3_to_9():
    dossiers = mr.stage_dossiers()
    ids = [d["gate_id"] for d in dossiers]
    assert ids == [3, 4, 5, 6, 7, 8, 9]
    names = [d["gate_name"] for d in dossiers]
    assert names == [
        "held_out_numerical",
        "route_stability",
        "logit_token_validation",
        "bounded_capability",
        "physical_nr_lowering",
        "complete_nx",
        "ebpw",
    ]
    required = {
        "exact_inputs",
        "null_it_tests",
        "kill_criterion",
        "command_that_would_run_it",
        "already_built",
        "still_missing",
        "predecessor_gate_id",
        "predecessor_output",
        "needs_nothing_but_predecessor_output",
        "funnel_input_shape",
        "wired_in_funnel",
    }
    for row in dossiers:
        missing = required - set(row)
        assert not missing, (row["gate_id"], missing)
        assert row["needs_nothing_but_predecessor_output"] is True
        assert row["wired_in_funnel"] is True
        assert row["predecessor_gate_id"] == row["gate_id"] - 1
        assert row["can_proceed_without_corpus"] is False
        assert row["already_built"]
        assert row["still_missing"]
        assert row["exact_inputs"]
        assert row["kill_criterion"]
        assert "python3 tools/future/" in row["command_that_would_run_it"]


def test_pipeline_wiring_is_a_chain():
    wiring = mr.pipeline_wiring()
    assert wiring["wired"] is True
    assert wiring["gate_2_needs_only_corpus"] is True
    assert wiring["gate_2_required_input"] == "teacher_corpus"
    assert wiring["gates_3_to_9_need_nothing_but_predecessor"] is True
    assert wiring["funnel_enforces_no_skip"] is True
    assert wiring["unwired"] == []
    assert len(wiring["edges"]) == 7
    for i, edge in enumerate(wiring["edges"]):
        assert edge["wired"] is True
        assert edge["from_gate"] == i + 2
        assert edge["to_gate"] == i + 3
        assert edge["from_output"] == mf.GATES[i + 1].required_input
        assert edge["to_input"] == mf.GATES[i + 2].required_input


def test_corpus_arrival_contract_shape():
    contract = mr.corpus_arrival_contract()
    assert contract["one_call"].endswith("admit_capture(rows, envelope=...)")
    assert contract["validator"] == "tools.future.teacher_corpus.validate_corpus"
    min_rows = contract["min_rows"]
    assert isinstance(min_rows, int) and min_rows > 0
    spec = contract["specimen_binding"]
    assert spec["model"] == tc.FLASH_SPECIMEN["model"]
    assert spec["pinned_revision"] == tc.FLASH_SPECIMEN["pinned_revision"]
    assert spec["seal_sha256"] == tc.FLASH_SPECIMEN["seal_sha256"]
    assert "layer" in contract["layer_surface_binding"]["required_per_row"]
    assert "surface" in contract["layer_surface_binding"]["required_per_row"]
    assert contract["per_row_route_ids"]["required"] is True
    for name in (
        "row_diversity",
        "prompt_diversity",
        "token_position_diversity",
        "route_diversity",
        "capability_domain_diversity",
    ):
        assert name in contract["diversity_thresholds"]
        assert contract["diversity_thresholds"][name]["threshold"] is not None
    assert contract["source_authority_capture"]["required"] is True
    assert contract["anti_fabrication"]["sacred_guard"] == "THRESHOLD_MET_ONLY_BY_DUPLICATION"
    # Guard still lives in teacher_corpus, not a fork.
    from pathlib import Path as _Path
    assert "THRESHOLD_MET_ONLY_BY_DUPLICATION" in _Path(tc.__file__).read_text()


def test_negative_control_structural_fixture_refused_by_validate_corpus():
    """NEGATIVE CONTROL: the fixture this module produces must fail the sacred guard.

    validate_corpus must REFUSE it as not-real-capture. If this fixture starts
    passing, the fixture is the bug, not the guard.
    """
    fixture = mr.make_structural_fixture()
    assert fixture
    assert len(fixture) <= 16
    assert all(r["fixture_kind"] == mr.STRUCTURAL_FIXTURE_KIND for r in fixture)
    assert all(str(r["payload"]).startswith(mr.STRUCTURAL_PAYLOAD_PREFIX) for r in fixture)
    assert all((r["specimen"]["model"]).startswith("fixture/") for r in fixture)
    assert all((r.get("provenance") or {}).get("kind") == "structural_fixture" for r in fixture)

    min_rows = mr.contract_min_rows()
    with pytest.raises(tc.CorpusRefused) as ei:
        tc.validate_corpus(fixture, min_rows=min_rows, raise_on_refuse=True)
    err = ei.value
    assert "REFUSED" in str(err)
    assert err.codes, "validate_corpus must fire at least one refusal code"
    assert "MISSING_SPECIMEN_OR_PROVENANCE_BINDING" in err.codes
    assert err.result["accepted"] is False

    admission = mr.admit_capture(
        fixture,
        envelope={
            "fixture_kind": mr.STRUCTURAL_FIXTURE_KIND,
            "source_authority_capture": False,
        },
    )
    assert admission["accepted"] is False
    assert admission["structural_fixture"] is True
    assert admission["reason"] != "ADMITTED"
    assert "STRUCTURAL_FIXTURE_NOT_REAL_CAPTURE" in admission["codes"]


def test_no_path_advances_gate2_without_genuine_corpus():
    """simulate-arrival, bind, and the funnel must not PASS gate 2 on a fixture."""
    sim = mr.simulate_arrival()
    assert sim["fabricated_training_rows"] is False
    assert sim["gate2_advanced"] is False
    assert sim["bound_gate2_passed"] is False
    assert sim["structural_fixture"] is True
    assert sim["admission"]["accepted"] is False
    assert sim["wiring"]["wired"] is True
    assert sim["validate_corpus_refused_fixture"] is True
    assert sim["recovered_families"]["all_stall_at_gate_2"] is True

    families, _ = mf.recover_families()
    bound, info = mr.bind_corpus_to_families(families, sim["admission"])
    assert info["bound"] is False
    assert info["gate2_advanced"] is False
    # Recovered families still have teacher_corpus NOT_BUILT, not a fixture.
    funnel = mf.Funnel()
    for cand in bound:
        g1 = funnel.advance(cand, 1)
        assert g1.verdict in {"PASSED", "REFUSED", "KILLED"}
        if g1.verdict != "PASSED":
            continue
        g2 = funnel.advance(cand, 2)
        assert g2.verdict != "PASSED"
        assert 2 not in (cand.get("passed_gates") or [])


def test_admit_capture_is_one_call_with_clear_reason():
    fixture = mr.make_structural_fixture()
    result = mr.admit_capture(fixture, envelope={"source_authority_capture": False})
    assert "reason" in result
    assert "codes" in result
    assert result["accepted"] is False
    assert isinstance(result["reason"], str) and result["reason"]
    # Duplicated 256-row pad: validate_corpus refuses, admit reports that code.
    duped = tc.make_duplicated_corpus(32, unique=4)
    dup_result = mr.admit_capture(duped, envelope={"source_authority_capture": True}, min_rows=32)
    assert dup_result["accepted"] is False
    assert (
        "THRESHOLD_MET_ONLY_BY_DUPLICATION" in dup_result["codes"]
        or "STRUCTURAL_FIXTURE_NOT_REAL_CAPTURE" in dup_result["codes"]
        or "SPECIMEN_BINDING_MISMATCH" in dup_result["codes"]
    )


def test_diverse_flash_labelled_fixture_still_not_admitted():
    """A 32-row diverse STATIC_ONLY fixture with Flash specimen still is not a capture.

    validate_corpus may accept it at min_rows=32; admit_capture must not, because
    source_authority_capture is false and provenance is a fixture.
    """
    rows = tc.make_diverse_corpus(32, specimen=tc.FLASH_SPECIMEN)
    vc = tc.validate_corpus(rows, min_rows=32, raise_on_refuse=True)
    assert vc["accepted"] is True
    admission = mr.admit_capture(
        rows,
        envelope={"source_authority_capture": False},
        min_rows=32,
    )
    assert admission["accepted"] is False
    assert "SOURCE_AUTHORITY_CAPTURE_FALSE" in admission["codes"] or (
        "STRUCTURAL_FIXTURE_NOT_REAL_CAPTURE" in admission["codes"]
    )


def test_simulate_arrival_does_not_fabricate_rows():
    sim = mr.simulate_arrival()
    assert sim["mode"] == "SIMULATE_ARRIVAL_WIRING_ONLY"
    assert sim["fabricated_training_rows"] is False
    assert sim["n_rows_inspected"] <= 16
    assert sim["n_rows_inspected"] < mr.contract_min_rows()
    # Must not write a fake capture under Codex's receipts/headless.
    fake = mr.REPO / "receipts" / "headless" / "FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json"
    # Cope with either visibility: we did not create or overwrite it.
    if fake.is_file():
        doc = json.loads(fake.read_text())
        assert doc.get("schema") != mr.SCHEMA


def test_simulate_arrival_caller_corpus_path(tmp_path):
    fixture = mr.make_structural_fixture()
    path = tmp_path / "caller.json"
    path.write_text(json.dumps({"rows": fixture, "source_authority_capture": False}))
    sim = mr.simulate_arrival(corpus_path=path)
    assert sim["row_source"] == "caller_supplied_path"
    assert sim["fabricated_training_rows"] is False
    assert sim["admission"]["accepted"] is False
    assert sim["gate2_advanced"] is False


def test_blocking_chain_names_metal_gpu_and_splits_work():
    chain = mr.blocking_chain()
    cap = chain["capture"]
    assert cap["lookup_copes"] is True
    assert "Metal" in cap["blocker"]
    # If the live receipt is visible, the documented fields hold. If not,
    # the chain still names the GPU blocker rather than treating invisibility
    # as a different science.
    if cap["present"]:
        assert cap["status"] == "BLOCKED_NO_METAL_GPU"
        assert cap["teacher_rows_written"] == 0
        assert cap["failure_stage"] == "dense_source_bf16_prefix_initialization"
        assert cap["source_authority_capture"] is False
        assert cap["evidence_source"] == "live_headless"
    else:
        assert cap["failure_stage"] == "dense_source_bf16_prefix_initialization"
    assert chain["gate_1_analytical"]["can_proceed_without_corpus"] is True
    assert chain["gate_2_teacher_fit"]["can_proceed_without_corpus"] is False
    assert chain["sidecar_gpu_authority"] is False
    names = {x["what"] for x in chain["analytical_screens_that_can_run_now"]}
    assert any("EBPW" in n or "ebpw" in n.lower() or "category" in n.lower() for n in names)
    cannot = chain["genuinely_cannot_without_corpus"]
    assert any("teacher_fit" in c or "gate 2" in c for c in cannot)
    assert any("promotion" in c for c in cannot)


def test_lookup_capture_boundary_copes_with_either_visibility():
    hit = mr.lookup_capture_boundary()
    assert hit["lookup_copes"] is True
    assert hit["evidence_source"] in {"live_headless", "pinned_snapshot"}
    if hit["present"]:
        assert hit["status"] == "BLOCKED_NO_METAL_GPU"
        assert hit["requested_rows"] == hit["minimum_rows"]
        assert hit["teacher_rows_written"] == 0
        assert hit["failure_stage"] == "dense_source_bf16_prefix_initialization"
        assert hit["source_authority_capture"] is False
    else:
        assert "Metal" in (hit.get("blocker_even_if_receipt_unseen") or "")


def test_ranked_first_experiments_admit_then_largest_family():
    ranked = mr.ranked_first_experiments()
    assert ranked
    assert ranked[0]["experiment"] == "admit_capture"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["requires_gpu"] is False
    budgets = mr.load_family_budgets()
    if budgets.get("present") and budgets.get("families"):
        first_fit = next(r for r in ranked if r["experiment"].startswith("gate_2_teacher_fit."))
        assert first_fit["family"] == budgets["families"][0]["family"]
        n_fit = sum(1 for r in ranked if r["experiment"].startswith("gate_2_teacher_fit."))
        assert n_fit == budgets["n_families"]
        # Derived from data, not a capped convenience integer.
        assert budgets["n_families"] == len(budgets["families"])
    experiments = [r["experiment"] for r in ranked]
    assert any(e.startswith("gate_3_") for e in experiments)
    assert any(e.startswith("gate_9_") for e in experiments)
    for row in ranked:
        assert "why" in row and row["why"]
        assert "cost_class" in row
        assert "command" in row


def test_family_count_is_derived_from_budget_not_hardcoded():
    budgets = mr.load_family_budgets()
    if budgets.get("present"):
        names = [f["family"] for f in budgets["families"]]
        assert len(set(names)) == len(names)
        assert budgets["n_families"] == len(names)
        # Ranked experiments must not freeze a 9-family assumption in the
        # contract layer; they follow whatever the budget currently lists.
        ranked = mr.ranked_first_experiments(budgets=budgets)
        n_fit = sum(1 for r in ranked if r["experiment"].startswith("gate_2_teacher_fit."))
        assert n_fit == len(names)


def test_evidence_source_is_pinned_or_live_per_input():
    doc = mr.build_document()
    src = doc["evidence_source"]
    assert isinstance(src, dict) and src
    for path, kind in src.items():
        assert kind in {"pinned_snapshot", "live_headless"}, (path, kind)
    # Capture boundary is current state → live_headless when used.
    assert src[mr.REL_CAPTURE_BOUNDARY] == "live_headless"
    # SUB1 / coherence are stable → pinned when the snapshot has them.
    if doc["family_budgets_derived"]["n_families"]:
        assert src[mr.REL_META_SUB1] == "pinned_snapshot"


def test_receipt_contains_required_substantive_sections():
    out = mr.build()
    doc = json.loads(out.read_text())
    for key in (
        "stage_dossiers",
        "corpus_arrival_contract",
        "simulate_arrival",
        "blocking_chain",
        "ranked_first_experiments",
        "recovered_implementation",
        "gaps_closed",
        "negative_findings",
        "evidence_source",
        "pipeline_wiring",
    ):
        assert key in doc and doc[key], key
    assert len(doc["stage_dossiers"]) == 7
    assert doc["simulate_arrival"]["gate2_advanced"] is False
    assert doc["simulate_arrival"]["fabricated_training_rows"] is False
    assert doc["pipeline_wiring"]["wired"] is True
    assert doc["current_funnel_stall"]["all_recovered_stall_at_gate_2"] is True
    assert doc["era_vocabulary"]["eras"] == 5
    assert doc["era_vocabulary"]["odysseys"] == 3
    assert "FPGA" in doc["era_vocabulary"]["fpga_is"] or "fpga" in doc["era_vocabulary"]["fpga_is"].lower()


def test_hardware_claim_still_blocked_on_this_receipt():
    doc = mr.build_document()
    _assert_no_hardware_claims(doc)
    with pytest.raises(HardwareClaimError):
        poisoned = dict(doc)
        poisoned["tps"] = 123.0
        _assert_no_hardware_claims(poisoned)


def test_funnel_still_refuses_skip_after_this_module():
    """Downstream readiness must not create a skip around teacher fit."""
    funnel = mf.Funnel()
    plan = {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": False,
        "regions": [
            {"kind": "shared_generator", "bits_class": "shared", "family": "x", "organ": "routed_experts"}
        ],
    }
    cand = {
        "id": "skip.via.ready",
        "family": "x",
        "organ": "routed_experts",
        "technique": "x",
        "model": mf.FLASH_MODEL,
        "allocation_plan": plan,
        "inputs": mf._default_inputs(
            allocation_plan=plan,
            teacher_corpus="NOT_BUILT",
            held_out_numerical={"status": "PASSED"},
        ),
        "passed_gates": [],
    }
    assert funnel.advance(cand, 1).verdict == "PASSED"
    assert funnel.advance(cand, 2).verdict == "REFUSED"
    assert funnel.advance(cand, 3).verdict == "REFUSED"
    assert cand.get("died_at") is None


def test_structural_fixture_refuses_to_emit_a_training_sized_corpus():
    with pytest.raises(ValueError):
        mr.make_structural_fixture(n=256)
    with pytest.raises(ValueError):
        mr.make_structural_fixture(n=32)
