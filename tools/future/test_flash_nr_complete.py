"""Negative controls for continuous Flash NR + seven typed EBPW quantities.

A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.future import ebpw_categories as ec
from tools.future import flash_nr_complete as fn
from tools.future._common import RECEIPTS


def test_build_emits_sealed_receipt():
    out = fn.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FLASH_NR_COMPLETE.json"
    assert doc["schema"] == "hawking.future.flash_nr_complete.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["selftest"]["cross_category_arithmetic_raises"] is True
    assert doc["selftest"]["cross_category_assignment_raises"] is True
    assert doc["selftest"]["meta_only_research_target_refused"] is True
    assert doc["selftest"]["unbilled_runtime_nr_rejected"] is True
    assert doc["selftest"]["dense_production_rejected"] is True
    assert doc["prospective_meta_bpw_role"] == "RESEARCH_TARGET"
    assert doc["qualified_complete_physical_ebpw_state"] == "UNKNOWN"
    assert doc["can_promote"] is False
    assert "RESEARCH_TARGET" in doc["can_promote_reason"] or "UNKNOWN" in doc["can_promote_reason"]
    assert doc["resident_callable"]["can_hcli_invoke"] is True
    assert doc["resident_callable"]["receipt"] == "receipts/future/FLASH_NR_COMPLETE.json"
    assert "future.flash_nr_complete.build" in doc["resident_callable"]["work_units_emitted"]
    assert (
        "future.flash_nr_complete.qualified_physical_ebpw"
        in doc["resident_callable"]["work_units_emitted"]
    )


def test_seven_categories_are_distinct_types():
    values = [
        fn.SourceControlEbpw(16.0),
        fn.StaticActiveEbpwEstimate(0.01),
        fn.StaticCompleteEbpwEstimate(16.0),
        fn.ProspectiveMetaBpw(0.887),
        fn.SerializedNrInformation(1024),
        fn.SerializedNxEbpw(4.0),
        fn.QualifiedCompletePhysicalEbpw(2.4),
    ]
    types = [type(v) for v in values]
    assert len(set(types)) == 7
    assert [v.category for v in values] == list(fn.SEVEN_TYPES)
    assert set(fn.SEVEN_TYPES) == {
        "source_control_ebpw",
        "static_active_ebpw_estimate",
        "static_complete_ebpw_estimate",
        "prospective_meta_bpw",
        "serialized_nr_information",
        "serialized_nx_ebpw",
        "qualified_complete_physical_ebpw",
    }


def test_same_category_arithmetic_is_ok():
    a = fn.SourceControlEbpw(10)
    b = fn.SourceControlEbpw(3)
    assert (a + b).value == 13
    assert type(a + b) is fn.SourceControlEbpw


def test_cross_category_assignment_and_arithmetic_raise():
    """NEGATIVE CONTROL: laundering one EBPW quantity into another is a type error."""
    meta = fn.ProspectiveMetaBpw(0.887, evidence=fn.RESEARCH_TARGET)
    phys = fn.QualifiedCompletePhysicalEbpw(0.887, evidence="launder attempt")
    src = fn.SourceControlEbpw(16.0, evidence=fn.SCIENCE_ONLY)
    est = fn.StaticCompleteEbpwEstimate(16.0, evidence="STATIC")
    with pytest.raises(fn.CategoryError, match="not interchangeable"):
        _ = meta + phys
    with pytest.raises(fn.CategoryError, match="not interchangeable"):
        _ = src + est
    with pytest.raises(fn.CategoryError):
        _ = src == est
    with pytest.raises(fn.CategoryError):
        _ = float(meta)
    with pytest.raises(fn.CategoryError):
        _ = meta + 0.887
    ledger = fn.SevenLedger()
    ledger.prospective_meta_bpw = meta
    with pytest.raises(fn.CategoryError, match="type error"):
        ledger.qualified_complete_physical_ebpw = meta
    with pytest.raises(fn.CategoryError, match="type error"):
        ledger.source_control_ebpw = est
    with pytest.raises(fn.CategoryError, match="type error"):
        fn.bind(fn.QualifiedCompletePhysicalEbpw, meta)
    with pytest.raises(fn.CategoryError, match="type error"):
        fn.bind(fn.QualifiedCompletePhysicalEbpw, ec.CompletePhysicalEbpw(0.887))


def test_can_promote_refuses_prospective_meta_bpw_alone():
    """NEGATIVE CONTROL: 0.887 RESEARCH_TARGET never promotes."""
    qty = fn.ProspectiveMetaBpw(0.887, evidence=fn.RESEARCH_TARGET)
    ok, reason = fn.can_promote(qty)
    assert ok is False
    assert "RESEARCH_TARGET" in reason
    assert "never promotes alone" in reason

    ok2, reason2 = fn.can_promote({"prospective_meta_bpw": 0.887})
    assert ok2 is False
    assert "RESEARCH_TARGET" in reason2

    ok3, reason3 = fn.can_promote(
        {"prospective_meta_bpw": 0.887, "promotion_allowed": True, "force_promote": True}
    )
    assert ok3 is False
    assert "RESEARCH_TARGET" in reason3

    ledger = fn.SevenLedger(prospective_meta_bpw=qty)
    ok4, reason4 = fn.can_promote(ledger)
    assert ok4 is False
    assert "RESEARCH_TARGET" in reason4


def test_nx_referencing_nr_without_billing_is_rejected():
    """NEGATIVE CONTROL: a quiet runtime NR pointer without billed bytes is refused."""
    nx = fn.unbilled_runtime_nr_nx()
    with pytest.raises(fn.NrBillingError, match="not billed"):
        fn.check_nr_billing(nx, fn.unbilled_runtime_nr_ledger())
    with pytest.raises(fn.NrBillingError, match="not billed"):
        fn.check_nr_billing(nx, None)
    ok = fn.check_nr_billing(nx, fn.billed_runtime_nr_ledger())
    assert ok["ok"] is True
    assert ok["references_nr_at_runtime"] is True
    assert ok["nr_bytes_billed"] is True


def test_live_metadata_nx_lowers_nr_is_not_a_runtime_reference():
    """Cope with either checkout: V0 NX names an NR at compile time, it does not run."""
    docs = fn.load_docs()
    nx = docs.get("nx_v0")
    if not isinstance(nx, dict):
        nx = {
            "schema": "hawking.flash.nx_genome.v1",
            "status": "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
            "lowers_nr": {"path": "receipts/headless/FLASH_COMPLETE_V0.nr.json"},
        }
    assert nx.get("lowers_nr")
    assert fn.nx_references_nr_at_runtime(nx) is False
    result = fn.check_nr_billing(nx, {"nr_bytes_billed": False})
    assert result["ok"] is True
    assert result["references_nr_at_runtime"] is False


def test_dense_rematerializing_production_path_is_rejected():
    """NEGATIVE CONTROL: production reconstruct-dense-checkpoint is refused."""
    with pytest.raises(fn.DenseRematError, match="dense"):
        fn.reject_dense_production(fn.dense_production_nx())
    verify = fn.reject_dense_production(fn.verifying_dense_nx())
    assert verify["ok"] is True
    assert verify["path_kind"] == fn.VERIFICATION
    ok, reason = fn.can_promote(
        {
            "qualified_complete_physical_ebpw": fn.QualifiedCompletePhysicalEbpw(
                2.4, evidence="synthetic"
            ),
            "path_kind": fn.VERIFICATION,
            "executable_byte_ledger": {
                "self_contained": True,
                "for_this_executable": True,
                "complete_storage_bytes": 4096,
            },
            "capability_preserving_runtime": True,
            "physical_measurement_authority": fn.PROTECTED,
            "bench_state": "PROTECTED",
            "measurement_state": fn.PROTECTED,
            "consumes_representation_directly": True,
        }
    )
    assert ok is False
    assert "verification" in reason


def test_continuous_nr_covers_every_census_organ():
    docs = fn.load_docs()
    nr = fn.build_continuous_nr(docs)
    required = fn.required_organs(docs)
    assert required, "census family_summary must yield at least one organ"
    fn.assert_complete(nr, required)
    present = [s["organ"] for s in nr["organs"]]
    assert present == required
    assert nr["complete"] is True
    assert nr["organ_count"] == len(required)
    assert nr["promotion_status"] == fn.NOT_PROMOTABLE
    assert "derived" in nr["promotion_reason"]
    for slot in nr["organs"]:
        assert slot["occupying"]["name"]
        assert slot["promotion_status"] == fn.NOT_PROMOTABLE
        occupying = slot["occupying"]
        if occupying["kind"] == fn.EXACT_FALLBACK:
            assert occupying["science_mark"] == fn.SCIENCE_ONLY
            assert occupying["representation"] == fn.EXACT_REPR


def test_missing_organ_is_rejected():
    """NEGATIVE CONTROL: completeness fails closed when an organ is dropped."""
    with pytest.raises(fn.IncompleteNrError, match="missing organs"):
        fn.assert_complete(
            {"complete": True, "organs": [{"organ": "norm"}]},
            ["norm", "routed_experts"],
        )
    docs = fn.load_docs()
    required = fn.required_organs(docs)
    if len(required) < 2:
        pytest.skip("need at least two census organs to drop one")
    nr = fn.build_continuous_nr(docs)
    dropped = dict(nr)
    dropped["organs"] = [s for s in nr["organs"] if s["organ"] != required[0]]
    with pytest.raises(fn.IncompleteNrError, match=required[0]):
        fn.assert_complete(dropped, required)


def test_qualified_physical_stays_unknown_and_is_not_0_887():
    docs = fn.load_docs()
    seven = fn.seven_from_docs(docs)
    q = seven.qualified_complete_physical_ebpw
    assert q is not None
    assert q.value is None
    assert q.category == "qualified_complete_physical_ebpw"
    p = seven.prospective_meta_bpw
    assert p is not None
    assert p.value is not None
    assert p.value < 1.0
    assert "RESEARCH_TARGET" in p.evidence
    src = seven.source_control_ebpw
    assert src is not None
    assert src.value is not None
    with pytest.raises(fn.CategoryError):
        _ = src == fn.StaticCompleteEbpwEstimate(src.value)
    dumped = seven.as_dict()
    assert dumped["prospective_meta_bpw"]["role"] == "RESEARCH_TARGET"
    assert dumped["qualified_complete_physical_ebpw"]["state"] == "UNKNOWN"
    assert dumped["qualified_complete_physical_ebpw"]["value"] is None


def test_receipt_does_not_copy_handoff_hardware_fields():
    out = fn.build()
    doc = json.loads(out.read_text())
    flash = doc["current_flash_state"]
    assert "GPU_ns" not in flash
    assert "complete_wall_ns" not in flash
    assert "accepted_tps" not in flash
    assert flash["current_qualified_physical_ebpw"] in {fn.UNKNOWN, "UNKNOWN"}
    for unit in doc["work_units"]:
        if unit["id"] == "future.flash_nr_complete.qualified_physical_ebpw":
            assert unit["status"] == "sleeping"
            assert unit["classification"] == "SLEEPING"
            assert unit["resource_class"] == "GPU_EXCLUSIVE"
            assert unit["must_not_synthesize_result"] is True


def test_byte_ledger_absence_is_recorded_not_asserted():
    """Sparse checkout: missing BYTE_LEDGER is a recorded path, not a proof of absence."""
    docs = fn.load_docs()
    row = docs["resolution"]["ledger"]
    assert "present" in row
    assert row["resolved_via"] in {"pinned", "headless", "primary_worktree", "git:HEAD", "missing"}
    nr = fn.build_continuous_nr(docs)
    assert nr["complete"] is True


def test_positive_combinator_opens_and_is_not_a_measurement():
    full = {
        "qualified_complete_physical_ebpw": fn.QualifiedCompletePhysicalEbpw(
            2.4, evidence="synthetic combinator control (not a measurement)"
        ),
        "source_control_ebpw": fn.SourceControlEbpw(16.0, evidence=fn.SCIENCE_ONLY),
        "executable_byte_ledger": {
            "self_contained": True,
            "for_this_executable": True,
            "complete_storage_bytes": 4096,
        },
        "capability_preserving_runtime": True,
        "physical_measurement_authority": fn.PROTECTED,
        "bench_state": "PROTECTED",
        "measurement_state": fn.PROTECTED,
        "path_kind": fn.PRODUCTION,
        "dense_rematerialization": False,
        "consumes_representation_directly": True,
        "nr_complete": True,
    }
    ok, reason = fn.can_promote(full)
    assert ok is True, reason
    assert "all promotion predicates held" in reason
    missing = dict(full)
    missing["capability_preserving_runtime"] = False
    ok2, reason2 = fn.can_promote(missing)
    assert ok2 is False
    assert "capability-preserving runtime" in reason2


def test_selftest_watches_the_guards_fail():
    controls = fn.selftest()
    assert controls["cross_category_arithmetic_raises"] is True
    assert controls["cross_category_assignment_raises"] is True
    assert controls["landed_complete_physical_cannot_bind_qualified"] is True
    assert controls["meta_only_research_target_refused"] is True
    assert controls["meta_with_flag_refused"] is True
    assert controls["unbilled_runtime_nr_rejected"] is True
    assert controls["billed_runtime_nr_accepted"] is True
    assert controls["dense_production_rejected"] is True
    assert controls["verification_may_reconstruct"] is True
    assert controls["missing_organ_rejected"] is True
    assert controls["positive_combinator_opens"] is True
    assert controls["positive_combinator_is_not_a_measurement"] is True
    assert controls["equal_numbers_are_not_interchangeable"] is True
