"""Tests for the MoE physical-execution school.

Negative control (must actually fail if the guard is removed):
  * a PHYSICAL_INVARIANT with no falsifier is refused
  * at least one observation is honestly a Metal ACCIDENT, with the
    distinguishing experiment named
A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import moe_physical_school as mps
from tools.future import static_skeleton as sk
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _doc() -> dict:
    out = mps.build()
    return json.loads(out.read_text())


def test_build_emits_sealed_receipt():
    out = mps.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "MOE_PHYSICAL_SCHOOL.json"
    assert doc["schema"] == "hawking.future.moe_physical.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert len(doc["seal_sha256"]) == 64
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "Static sidecar" in doc["claim_boundary"]
    assert doc["measurement_classes"]["this_module"] == "STATIC_ONLY"
    assert doc["measurement_classes"]["DIAGNOSTIC_RELATIVE"] == "not produced"
    assert doc["measurement_classes"]["PROTECTED_ABSOLUTE"] == "not produced"
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["era_vocabulary"]["no_era_vi"] is True
    assert doc["era_vocabulary"]["no_odyssey_iv"] is True
    assert doc["era_vocabulary"]["fpga_is_not_its_own_civilization"] is True


def test_selftest_aliases_build():
    assert mps.selftest().name == "MOE_PHYSICAL_SCHOOL.json"


def test_taxonomy_covers_the_contracted_forms():
    forms = mps.execution_taxonomy()
    ids = [f["id"] for f in forms]
    assert ids == [
        "gather_scatter",
        "dense_with_mask",
        "sorted_by_expert",
        "per_expert_dispatch",
        "fused_routed_dispatch",
        "route_before_payload",
        "payload_then_select",
        "static_skeleton_expert_id",
        "data_dependent_topology",
    ]
    for form in forms:
        pc = form["physical_consequence"]
        for key in mps.CONSEQUENCE_KEYS:
            assert pc[key], f"{form['id']} missing {key}"
        assert form["description"]
        assert "flash_layer46" in form
        assert "replayable_with_expert_id_only" in form


def test_observed_flash_form_is_gather_route_before_fused():
    doc = _doc()
    observed = doc["observed_flash_form"]
    assert observed["geometry"] == "gather_scatter"
    assert observed["launch"] == "fused_routed_dispatch"
    assert observed["order"] == "route_before_payload"
    assert observed["skeleton"] == "static_skeleton_expert_id"
    assert observed["shared_expert"] == "static_always_on"
    kernels = observed["routed_path_kernels"]
    assert "qwen_next_bf16_expert_gate_up_swiglu" in kernels
    assert "qwen_next_bf16_expert_down" in kernels
    assert "moe_topk_gate" in kernels
    # P6 per-expert launches are a different observed form, not layer-46.
    assert observed["p6_hash_graph_is_a_different_observed_form"]["launch"] == "per_expert_dispatch"


def test_repeated_work_is_grounded_not_theoretical():
    doc = _doc()
    census = doc["grounding"]["organ_census"]
    assert census["reachable"] is True
    routed = census["routed_experts"]
    assert routed["family"] == "routed_experts"
    # ~68.5% is the census fraction; read-from-receipt, not a measurement.
    frac = routed["fraction"]
    assert frac["provenance"] == "read_from_receipt"
    assert frac["not_a_measurement"] is True
    assert 0.68 < float(frac["value"]) < 0.70

    items = {r["id"]: r for r in doc["repeated_work_census"]}
    assert "RW-INDEPENDENT-W-E" in items
    assert "RW-SAME-X-MANY-EXPERTS" in items
    same_x = items["RW-SAME-X-MANY-EXPERTS"]
    assert "DISPATCH FUSED" in same_x["layer46_status"]
    assert "COMPUTE-ONE-X-MANY-EXPERTS" in same_x["already_named_by"]
    # Cosine is a structure metric: experts are not copies.
    doctor = doc["grounding"]["doctor_screen"]
    if doctor.get("reachable"):
        assert abs(float(doctor["cross_expert_gate_up_mean_cosine"])) < 0.02
    not_repeat = items["RW-WHAT-IS-NOT-REPEATED"]
    assert "schedule shape" in not_repeat["layer46_status"]


def test_compute_sharing_is_cited_not_forked():
    doc = _doc()
    cited = doc["cited_compute_sharing"]
    kinds = [c["kind"] for c in cited]
    assert kinds == list(mps.ebs.REQUIRED_COMPUTE_KINDS)
    by_kind = {c["kind"]: c for c in cited}
    assert by_kind["one_hidden_vector_many_experts"]["layer46_execution_status"] == (
        "DISPATCH_FUSED_PAYLOAD_INDEPENDENT"
    )
    assert by_kind["latent_weighted_reduction"]["layer46_execution_status"] == "NOT_IN_GRAPH"
    # The school must not re-emit expert-bank candidates as its own.
    src = Path(mps.__file__).read_text()
    assert "COMPUTE-ONE-X-MANY-EXPERTS" in src
    assert "def generate_compute" not in src


def test_ceremony_vs_work_uses_the_ledger():
    doc = _doc()
    cvw = doc["ceremony_vs_work"]
    assert cvw["law_cited"] == "a dispatch is not a unit of cost"
    arith_kernels = {r["kernel"] for r in cvw["arithmetic"]["rows"]}
    assert "qwen_next_bf16_expert_gate_up_swiglu" in arith_kernels
    assert "qwen_next_bf16_expert_down" in arith_kernels
    control_kernels = {r["kernel"] for r in cvw["control"]["rows"]}
    assert "moe_topk_gate" in control_kernels
    combine_kernels = {r["kernel"] for r in cvw["combine"]["rows"]}
    assert "qwen_next_moe_weighted_sum" in combine_kernels
    # HyperConnection consume of moe_output is not the routed organ.
    arith_and_combine = {r["kernel"] for r in cvw["arithmetic"]["rows"]} | combine_kernels
    assert "qwen_next_hyperconnection_combine" not in arith_and_combine
    sched = cvw["scheduling_overhead"]
    assert sched["fusion_candidates_still_open_on_routed_path"]
    assert "TokenCommandBuffer" in sched["token_command_buffer"]
    assert "synchronization" in sched["critical_path_ceremony_owners"]["layer10"]
    assert "command submission" in sched["critical_path_ceremony_owners"]["layer10"]
    # Ledger dispatch count is labelled, not claimed as a measurement.
    dc = cvw["ledger_dispatch_count"]
    assert dc["provenance"] == "read_from_receipt"
    assert dc["not_a_measurement"] is True
    assert dc["value"] == 35


def test_every_observation_has_falsifier_and_distinguish():
    obs = mps.observations()
    assert obs, "school must emit observations"
    classes = {o["classification"] for o in obs}
    assert "PHYSICAL_INVARIANT" in classes
    assert "METAL_ACCIDENT" in classes
    for o in obs:
        for field in mps.OBSERVATION_FIELDS:
            assert o.get(field), f"{o.get('id')} missing {field}"
        assert o["evidence_class"] == "STATIC_ONLY"
        assert o["classification"] in mps.CLASSIFICATIONS
        assert o["transfer_scope"] in mps.TRANSFER_SCOPES
        if o["classification"] == "PHYSICAL_INVARIANT":
            assert o["transfer_scope"] in mps.INVARIANT_SCOPES
        else:
            assert o["transfer_scope"] in mps.ACCIDENT_SCOPES


def test_negative_control_refuses_invariant_without_falsifier():
    """Guard nobody has watched fail is not a guard."""
    probe = {
        "id": "PROBE-TEST-INVARIANT-NO-FALSIFIER",
        "claim": "routed MoE is always faster fused",
        "classification": "PHYSICAL_INVARIANT",
        "transfer_scope": "MOE_FAMILY",
        "cheapest_falsifier": "",
        "distinguish_experiment": "run it on CUDA",
        "evidence": "none",
        "physical_consequence": mps._consequence("a", "b", "c", "d"),
    }
    with pytest.raises(mps.ClaimRefused) as ei:
        mps.admit_observation(probe)
    assert "no falsifier" in str(ei.value)
    assert "PROBE-TEST-INVARIANT-NO-FALSIFIER" in str(ei.value)

    # A well-formed invariant still admits, so the guard is discriminating.
    live = next(o for o in mps.observations() if o["classification"] == "PHYSICAL_INVARIANT")
    admitted = mps.admit_observation(live)
    assert admitted["id"] == live["id"]


def test_negative_control_honest_metal_accident_with_distinguish():
    """At least one observation is honestly a Metal ACCIDENT, experiment named."""
    accidents = [o for o in mps.observations() if o["classification"] == "METAL_ACCIDENT"]
    assert accidents, "school must classify at least one observation a Metal accident"
    # Dual command-buffer wait is the textbook Metal recording accident:
    # layer-46 already uses one TokenCommandBuffer; P6 historically used two.
    dual = next(o for o in accidents if o["id"] == "ACC-P6-DUAL-COMMANDBUFFER")
    assert dual["transfer_scope"] == "METAL_BACKEND"
    assert "CUDA graphs" in dual["distinguish_experiment"] or "CUDA" in dual["distinguish_experiment"]
    assert "FPGA" in dual["distinguish_experiment"]
    assert dual["cheapest_falsifier"]
    # An accident with no distinguishing experiment is refused — the other
    # half of the guard.
    with pytest.raises(mps.ClaimRefused) as ei:
        mps.admit_observation(
            {
                "id": "PROBE-ACCIDENT-NO-DISTINGUISH",
                "claim": "35 launches are a Metal accident",
                "classification": "METAL_ACCIDENT",
                "transfer_scope": "METAL_BACKEND",
                "cheapest_falsifier": "fused parity fails",
                "distinguish_experiment": "",
                "evidence": "ledger",
                "physical_consequence": mps._consequence("a", "b", "c", "d"),
            }
        )
    assert "distinguishing experiment" in str(ei.value)


def test_refusal_controls_in_receipt_actually_fired():
    rows = mps.assert_guards_fire()
    assert len(rows) == 3
    assert all(r["refused"] is True for r in rows)
    assert {r["probe_id"] for r in rows} == {
        "PROBE-INVARIANT-WITHOUT-FALSIFIER",
        "PROBE-ACCIDENT-WITHOUT-DISTINGUISH",
        "PROBE-ACCIDENT-WITH-INVARIANT-SCOPE",
    }
    doc = _doc()
    assert doc["counts"]["refusal_controls_fired"] == 3
    assert doc["counts"]["metal_accidents"] >= 1
    assert doc["counts"]["physical_invariants"] >= 1


def test_accident_cannot_wear_an_invariant_scope():
    with pytest.raises(mps.ClaimRefused) as ei:
        mps.admit_observation(
            {
                "id": "PROBE-SCOPE-CONFUSION",
                "claim": "Metal command buffers are a law of MoE",
                "classification": "METAL_ACCIDENT",
                "transfer_scope": "GENERAL_PHYSICAL",
                "cheapest_falsifier": "a non-Metal backend still needs them",
                "distinguish_experiment": "CUDA graphs",
                "evidence": "none",
                "physical_consequence": mps._consequence("a", "b", "c", "d"),
            }
        )
    assert "cannot have invariant scope" in str(ei.value)


def test_skeleton_cross_reference_replayable_vs_gated():
    xref = mps.skeleton_cross_reference()
    assert xref["expert_id_is_permitted"] is True
    assert xref["permitted_dynamic_slots"] == list(sk.SLOT_KINDS)
    accepted = {r["form_id"]: r for r in xref["replayable_with_expert_id_only"]}
    for form_id in mps.REPLAYABLE_FORMS:
        assert accepted[form_id]["accepted"] is True
        assert accepted[form_id]["errors"] == []
    refused = {r["form_id"]: r for r in xref["require_data_dependent_topology"]}
    for form_id in mps.NOT_REPLAYABLE_FORMS:
        assert refused[form_id]["accepted"] is False
        blob = " ".join(refused[form_id]["errors"])
        assert "VALUE_GATED" in blob

    # Direct: gather/scatter replays; data-dependent does not.
    gather = sk.validate(mps.skeleton_for_form("gather_scatter"))
    assert gather.accepted is True
    gated = sk.validate(mps.skeleton_for_form("data_dependent_topology"))
    assert gated.accepted is False
    # The legal twin still passes, so the validator is discriminating.
    assert sk.validate(sk.legal_expert_id_skeleton()).accepted is True


def test_payload_then_select_static_replays_gated_does_not():
    static = sk.validate(mps.skeleton_for_form("payload_then_select_static"))
    assert static.accepted is True
    gated = sk.validate(mps.skeleton_for_form("payload_then_select_gated"))
    assert gated.accepted is False


def test_routed_candidates_are_derived_not_hardcoded():
    ev = mps.load_named("ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json")
    assert ev["reachable"] is True
    expected = mps.select_routed_candidates(ev["doc"])
    doc = _doc()
    got = doc["routed_candidates"]
    assert got["count_is_derived"] is True
    assert got["count"] == len(expected)
    assert got["count"] == len(got["rows"])
    assert [r["candidate_id"] for r in got["rows"]] == [c["candidate_id"] for c in expected]
    ids = [r["candidate_id"] for r in got["rows"]]
    assert any(i.startswith("flash-p6-") for i in ids)
    assert any(i.startswith("flash-routed-") or i == "flash-routed-fp4-gate-up-swiglu-fused" for i in ids)
    assert any(i.startswith("flash-compact-moe-") for i in ids)
    assert "flash-router-topk-fusion" in ids
    for row in got["rows"]:
        assert row["exact_mutation"]
        assert row["expected_eliminated_work"]
    # Queue total is derived from the receipt, not a campaign-era constant.
    total = got["queue_total_candidates"]
    assert total["provenance"] == "read_from_receipt"
    assert total["value"] == len(ev["doc"]["candidates"])


def test_receipt_does_not_claim_hardware_numbers():
    doc = _doc()

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_MOE_PHYSICAL_SCHOOL_HARDWARE_PROBE.json",
            {"schema": "probe", "version": 1, "tps": 12.0},
            "tools/future/test_moe_physical_school.py",
        )


def test_ledger_gpu_ns_is_not_restated_even_as_null_claim():
    doc = _doc()
    for row in doc["grounding"]["layer46_routed_rows"]:
        assert "gpu_ns" not in row
        assert "host_encode_us" not in row
        assert "kernel" in row
    for owner in doc["grounding"]["critical_path_layer10"].get("owners") or []:
        assert "ns" not in owner or owner.get("ns") is None
        assert "ns_populated" in owner
        assert owner["ns_populated"] is False


def test_evidence_source_is_labelled_per_input():
    doc = _doc()
    block = doc["evidence_source"]
    assert block["overall"] in {"pinned_snapshot", "live_headless", "mixed_pinned_and_live", "unavailable"}
    # This checkout has the pinned snapshot; the school must record that
    # path when it takes it. If a future checkout only has live headless,
    # overall would be live_headless — both are honest.
    for name in mps.EVIDENCE_NAMES:
        row = block["per_input"][name]
        assert row["evidence_source"] in {"pinned_snapshot", "live_headless", "unavailable"}
        assert isinstance(row["reachable"], bool)
        if row["reachable"]:
            assert row["path"]
            assert row["sha256"]
            assert len(row["sha256"]) == 64


def test_load_named_copes_with_either_state():
    """Do not encode the checkout. Assert the loader records which path it took."""
    hit = mps.load_named("FLASH_ORGAN_CENSUS.json")
    assert hit["evidence_source"] in {"pinned_snapshot", "live_headless", "unavailable"}
    if hit["reachable"]:
        assert isinstance(hit["doc"], dict)
        assert hit["doc"].get("schema")
    else:
        assert hit["doc"] is None
        assert "coped" in hit

    # A name this school invented: the module must cope, and the test must
    # not treat that as evidence a real campaign file is absent.
    probe = mps.load_named("MOE_PHYSICAL_SCHOOL_NO_SUCH_INPUT.json")
    assert probe["evidence_source"] == "unavailable"
    assert probe["reachable"] is False
    assert probe["coped"]


def test_moe_negatives_are_queried_not_rederived():
    neg = mps.consult_moe_negatives()
    assert "index_reachable" in neg
    assert list(neg["families_queried"]) == list(mps.MOE_NEGATIVE_FAMILIES)
    # Cope with either checkout: index may or may not be on disk.
    if neg["index_reachable"]:
        assert neg["hit_count"] == len(neg.get("sample_scar_ids") or []) or neg["hit_count"] >= 0
        assert "cross_expert_structure" in neg["families_queried"]
        assert "megakernel" in neg["implication"].lower() or "Megakernel" in neg["implication"]
        assert "similar" in neg["implication"]
    else:
        assert "coped" in neg


def test_meta_contract_route_before_payload_is_cited():
    doc = _doc()
    budget = doc["grounding"]["family_budget"]
    if budget.get("reachable"):
        acc = budget["accelerator_contract"]
        assert acc["route_before_payload"] is True
        assert acc["dense_rematerialization"] is False
        assert budget["routed_experts"]["runtime_shape"]


def test_form_missing_consequence_is_refused():
    with pytest.raises(mps.FormSchemaError):
        mps._require_form({"id": "broken", "description": "x", "physical_consequence": {}})
