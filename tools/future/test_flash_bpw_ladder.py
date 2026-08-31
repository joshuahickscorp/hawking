"""Negative controls for the complete-executable EBPW ladder.

A guard nobody has watched fail is not a guard. These tests prove the
ladder can refuse: a rung claimed on prospective_meta_bpw, a plan that is
smaller but rematerializes a dense parent, a plan that is smaller but
multiplies FLOPs past its byte saving, and the current tree (no qualified
physical evidence) marking every rung UNTESTED or REFUSED and none REACHED.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.future import ebpw_categories as ec
from tools.future import flash_bpw_ladder as ladder
from tools.future import flash_nr_complete as fn
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)


def test_seven_quantities_are_imported_not_redefined():
    assert ladder.IMPORTED_SEVEN == (
        fn.SourceControlEbpw,
        fn.StaticActiveEbpwEstimate,
        fn.StaticCompleteEbpwEstimate,
        fn.ProspectiveMetaBpw,
        fn.SerializedNrInformation,
        fn.SerializedNxEbpw,
        fn.QualifiedCompletePhysicalEbpw,
    )
    assert [cls.category for cls in ladder.IMPORTED_SEVEN] == list(fn.SEVEN_TYPES)
    assert ladder.QualifiedCompletePhysicalEbpw is fn.QualifiedCompletePhysicalEbpw
    assert ladder.ProspectiveMetaBpw is fn.ProspectiveMetaBpw
    assert ladder.REQUIRED_QUANTITY == "qualified_complete_physical_ebpw"
    assert "SourceControlEbpw" not in ladder.__dict__ or ladder.SourceControlEbpw is fn.SourceControlEbpw


def test_rungs_are_the_2_25_to_sub1_ladder():
    rows = ladder.rungs()
    assert [r.id for r in rows] == [
        "bpw_2_25",
        "bpw_2_00",
        "bpw_1_75",
        "bpw_1_50",
        "bpw_1_25",
        "bpw_1_00",
        "bpw_sub1",
    ]
    assert [r.target_bpw for r in rows] == [2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0]
    assert rows[-1].exclusive is True
    assert all(r.exclusive is False for r in rows[:-1])
    for r in rows:
        d = r.as_dict()
        assert d["required_quantity"] == ladder.REQUIRED_QUANTITY
        assert d["required_evidence_class"] == "PROTECTED_ABSOLUTE"
        ids = [a["id"] for a in d["required_artifacts"]]
        assert "qualified_complete_physical_ebpw" in ids
        assert "source_independent_nx" in ids
        assert "coherence_held" in ids
        assert d["prospective_meta_bpw_role"] == fn.RESEARCH_TARGET
    assert ladder.resolve_rung(2.25).id == "bpw_2_25"
    assert ladder.resolve_rung("sub-1").id == "bpw_sub1"
    assert ladder.resolve_rung(0.5).id == "bpw_sub1"
    with pytest.raises(ladder.UnknownRungError):
        ladder.resolve_rung("ERA_VI_DOES_NOT_EXIST")


def test_status_prospective_meta_bpw_is_refused():
    """NEGATIVE CONTROL: a rung claimed on prospective_meta_bpw is REFUSED."""
    qty = fn.ProspectiveMetaBpw(0.887, evidence=fn.RESEARCH_TARGET)
    row = ladder.status("bpw_sub1", qty)
    assert row["verdict"] == ladder.REFUSED
    assert row["verdict"] != ladder.REACHED
    assert row["required_quantity"] == "qualified_complete_physical_ebpw"
    assert row["claimed_quantity"] == "prospective_meta_bpw"
    assert "qualified_complete_physical_ebpw" in row["reason"]
    flagged = ladder.status(
        "bpw_sub1",
        {"quantity": "prospective_meta_bpw", "value": 0.887, "promotion_allowed": True},
    )
    assert flagged["verdict"] == ladder.REFUSED
    assert flagged["required_quantity"] == "qualified_complete_physical_ebpw"


def test_status_diagnostic_factor_bpw_is_refused():
    """NEGATIVE CONTROL: the L4 0.0254/0.5284 screen cannot mark sub-1 REACHED."""
    claim = {
        "quantity": "diagnostic_factor_equivalent_bpw",
        "value": 0.0254,
        "heldout_relative_fro_error": 0.5284,
        "surface_gate_pass": False,
        "schema": "hawking.flash.meta_coherence_screen.v1",
    }
    row = ladder.status("bpw_sub1", claim)
    assert row["verdict"] == ladder.REFUSED
    assert row["required_quantity"] == "qualified_complete_physical_ebpw"
    assert row["claimed_quantity"] == "diagnostic_factor_equivalent_bpw"


def test_status_wrong_six_types_are_refused():
    """NEGATIVE CONTROL: none of the other six typed quantities can mark a rung."""
    values = {
        "source_control_ebpw": fn.SourceControlEbpw(16.0),
        "static_active_ebpw_estimate": fn.StaticActiveEbpwEstimate(0.01),
        "static_complete_ebpw_estimate": fn.StaticCompleteEbpwEstimate(16.0),
        "serialized_nr_information": fn.SerializedNrInformation(1024),
        "serialized_nx_ebpw": fn.SerializedNxEbpw(0.5),
        "prospective_meta_bpw": fn.ProspectiveMetaBpw(0.5),
    }
    for name, qty in values.items():
        row = ladder.status("bpw_1_00", qty)
        assert row["verdict"] == ladder.REFUSED, name
        assert row["required_quantity"] == "qualified_complete_physical_ebpw", name
        assert row["claimed_quantity"] == name


def test_current_tree_has_no_reached_rung():
    """NEGATIVE CONTROL: no qualified physical evidence ⇒ none REACHED."""
    rows = ladder.ladder_status()
    assert len(rows) == 7
    for row in rows:
        assert row["verdict"] in {ladder.UNTESTED, ladder.REFUSED}
        assert row["verdict"] != ladder.REACHED
        assert row["required_quantity"] == "qualified_complete_physical_ebpw"
        assert "qualified_complete_physical_ebpw" in row["missing_artifacts"] or (
            row["verdict"] == ladder.UNTESTED
        )
    assert not any(row["verdict"] == ladder.REACHED for row in rows)


def test_dominated_smaller_but_rematerializing():
    """NEGATIVE CONTROL: smaller + dense parent rematerialization is dominated."""
    verdict = ladder.dominated(ladder.remat_plan())
    assert verdict["dominated"] is True
    assert verdict["verdict"] == ladder.DOMINATED
    assert ladder.RULE_REMAT in verdict["fired"]
    assert ladder.RULE_INCOHERENT not in verdict["fired"]
    assert ladder.RULE_COMPUTE not in verdict["fired"]


def test_dominated_smaller_but_compute_multiplied():
    """NEGATIVE CONTROL: 0.5x bytes / 3.0x FLOPs is dominated."""
    verdict = ladder.dominated(ladder.compute_trap_plan())
    assert verdict["dominated"] is True
    assert verdict["verdict"] == ladder.DOMINATED
    assert ladder.RULE_COMPUTE in verdict["fired"]
    assert ladder.RULE_REMAT not in verdict["fired"]
    assert ladder.RULE_INCOHERENT not in verdict["fired"]


def test_dominated_smaller_but_incoherent_screen_shape():
    """NEGATIVE CONTROL: smaller + held-out failure is dominated.

    Prefers the real L4 REAL256 receipt. If that file is unreadable in this
    checkout, the cited 0.0254 / 0.5284 shape still has to fire — the test
    copes, it does not skip.
    """
    screen = ladder.load_screen()
    if screen.get("reachable"):
        plan = ladder.screen_trap_plan(screen)
        incumbent = plan.get("incumbent") if isinstance(plan.get("incumbent"), dict) else None
        verdict = ladder.dominated(plan, incumbent)
        assert plan["bpw"] is not None
        assert plan["heldout_relative_fro_error"] is not None
        assert plan["heldout_relative_fro_error"] > ladder.MAX_HELDOUT_RELATIVE_FRO
        assert plan["bpw"] < 1.0
    else:
        plan = {
            "id": "cited_l4_shape",
            "quantity": "diagnostic_factor_equivalent_bpw",
            "bpw": 0.0254,
            "storage_bytes": 1_218_560,
            "flop_milli": ladder.FLOOR_CONTROL_FLOP_MILLI,
            "heldout_relative_fro_error": 0.5284,
            "surface_gate_pass": False,
            "coherent": False,
            "rematerializes_dense_parent": False,
            "path_kind": ladder.PRODUCTION,
        }
        incumbent = {
            "id": "dense",
            "storage_bytes": 766_771_200,
            "flop_milli": ladder.FLOOR_CONTROL_FLOP_MILLI,
            "coherent": True,
        }
        verdict = ladder.dominated(plan, incumbent)
    assert verdict["dominated"] is True
    assert verdict["verdict"] == ladder.DOMINATED
    assert ladder.RULE_INCOHERENT in verdict["fired"]


def test_dominated_honest_smaller_stays_on_the_front():
    """Positive combinator: dominated() is not a constant True."""
    verdict = ladder.dominated(ladder.honest_smaller_plan())
    assert verdict["dominated"] is False
    assert verdict["verdict"] == ladder.ON_FRONT
    assert verdict["fired"] == []
    front = ladder.pareto_front(
        [ladder.honest_smaller_plan(), ladder.compute_trap_plan(), ladder.remat_plan()]
    )
    ids = [p["id"] for p in front]
    assert "honest_smaller_direct" in ids
    assert "half_bytes_triple_flops" not in ids
    assert "smaller_dense_remat" not in ids


def test_dominated_refuses_a_plan_with_no_size():
    with pytest.raises(ladder.LadderRefuse):
        ladder.dominated({})
    verdict = ladder.dominated({"id": "no-size", "coherent": True})
    assert verdict["verdict"] == ladder.REFUSED
    assert verdict["dominated"] is True
    assert verdict["on_pareto_front"] is False


def test_budget_is_heterogeneous_not_uniform():
    for r in ladder.rungs():
        b = ladder.budget(r)
        assert b["forces_uniform_bpw"] is False
        assert b["uniform_refused"] is True
        assert b["unit"] == "TOTAL_EXECUTABLE_INFORMATION"
        assert b["not_a_measurement"] is True
        assert b["evidence_class"] == "STATIC_ONLY"
        assert b["gpu_authority"] is False
        values = [o["proposed_bpw"] for o in b["organs"] if o["proposed_bpw"] is not None]
        assert values, f"{r.id} emitted no proposed_bpw"
        assert len(set(round(v, 6) for v in values)) >= 2, f"{r.id} collapsed to uniform"
        by_name = {o["organ"]: o for o in b["organs"]}
        routed = by_name.get("routed_experts")
        router = by_name.get("router")
        if routed and router and routed["proposed_bpw"] is not None and router["proposed_bpw"] is not None:
            assert router["proposed_bpw"] >= routed["proposed_bpw"]
            assert router["bit_class"] == "CONTROL_PREMIUM"
            assert routed["bit_class"] == "PREDICTABLE_BULK"
        for organ in b["organs"]:
            assert organ["not_qualified_complete_physical_ebpw"] is True
            assert organ["quantity"] == "search_pressure_allocation"


def test_status_combinator_can_reach_and_above_target_refuses():
    """The gate can open. A physical number above the bound still refuses."""
    opened = ladder.status("bpw_2_25", ladder.combinator_reached_evidence("bpw_2_25"))
    assert opened["verdict"] == ladder.REACHED
    assert opened["not_a_measurement"] is True
    assert opened["claimed_quantity"] == "qualified_complete_physical_ebpw"
    above = ladder.status("bpw_2_00", ladder.combinator_reached_evidence("bpw_2_25"))
    assert above["verdict"] == ladder.REFUSED
    assert above["verdict"] != ladder.REACHED
    sub1 = ladder.status("bpw_sub1", ladder.combinator_reached_evidence("bpw_2_25"))
    assert sub1["verdict"] == ladder.REFUSED


def test_cross_category_arithmetic_still_raises():
    with pytest.raises(fn.CategoryError):
        _ = ladder.ProspectiveMetaBpw(0.887) + ladder.QualifiedCompletePhysicalEbpw(0.887)
    with pytest.raises(ec.CategoryError):
        _ = fn.ProspectiveMetaBpw(0.887) + fn.QualifiedCompletePhysicalEbpw(2.25)


def test_build_emits_sealed_receipt():
    out = ladder.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FLASH_BPW_LADDER.json"
    assert doc["schema"] == "hawking.future.flash_bpw_ladder.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert not isinstance(doc.get(field), (int, float))
    assert doc["disk_reached_count"] == 0
    assert len(doc["rungs"]) == 7
    assert doc["required_quantity"] == "qualified_complete_physical_ebpw"
    assert doc["selftest"]["meta_claim_verdict"] == ladder.REFUSED
    assert "RULE_REMAT" in str(doc["selftest"]["remat_fired"]) or ladder.RULE_REMAT in doc["selftest"]["remat_fired"]
    assert ladder.RULE_COMPUTE in doc["selftest"]["compute_fired"]
    assert ladder.RULE_INCOHERENT in doc["selftest"]["screen_fired"]
    assert doc["selftest"]["honest_verdict"] == ladder.ON_FRONT
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_REPRESENTATION.meta-gates-3-9"
    assert "CPU_ANALYSIS" in doc["resident_callable"]["workunit"]
    assert doc["resident_callable"]["receipt"] == "receipts/future/FLASH_BPW_LADDER.json"
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["seven_quantities"] == list(fn.SEVEN_TYPES)


def test_selftest_aliases_are_watched():
    controls = ladder.selftest()
    assert controls["meta_claim_verdict"] == ladder.REFUSED
    assert controls["disk_reached"] == []
    assert controls["combinator_is_not_a_measurement"] is True
    assert controls["cross_category_still_raises"] is True
    assert controls["unknown_rung_raises"] is True
    assert controls["empty_plan_raises"] is True
    assert controls["seven_imported_not_redefined"] is True


def test_receipt_rejects_numeric_hardware_fields():
    with pytest.raises(HardwareClaimError):
        from tools.future._common import write_receipt

        write_receipt(
            "_LADDER_HARDWARE_PROBE.json",
            {"schema": "probe", "tps": 1.0},
            "tools/future/flash_bpw_ladder.py",
        )
