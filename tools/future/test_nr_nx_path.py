"""Negative controls for the NR/NX path map.

A requirement reported MET without its artifact, a physical miss labelled
pending, a silent disagreement with the audit, or a physical_ebpw value
would be the campaign repeating STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM.
These tests make each of those refusals fire. An absent receipt is a
recorded refusal, never pytest.skip.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import flash_nx_audit as nx_audit
from tools.future import nr_nx_path as nnp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _receipt() -> dict:
    path = nnp.build()
    return json.loads(path.read_text())


def test_build_emits_sealed_static_receipt():
    out = nnp.build()
    assert out.name == "NR_NX_PATH.json"
    assert out.parent == RECEIPTS
    doc = json.loads(out.read_text())
    assert doc["schema"] == nnp.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["physical_ebpw"] is None
    assert doc["physical_ebpw_written"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == "receipts/future/NR_NX_PATH.json"
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_EXECUTION.complete-token"
    assert doc["seven_all_met"] is False
    assert doc["criterion"]["id"] == "nr_nx_path_callable"
    assert doc["criterion"]["met"] is False


def test_seven_named_in_audit_order_and_states_read_from_audit():
    doc = _receipt()
    names = [r["requirement"] for r in doc["seven"]]
    assert names == list(nx_audit.SEVEN_REQUIREMENTS)
    audit_doc, via = nnp.load_rel(nnp.REL_AUDIT)
    if audit_doc is None:
        assert doc["audit"]["refused"] is True
        assert via == "missing"
        for row in doc["seven"]:
            assert row["ok"] is False
            assert row.get("audit_state") is None
        return
    audit_rows = {r["requirement"]: r for r in audit_doc["seven_requirements"]}
    for row in doc["seven"]:
        assert row["audit_state"] == audit_rows[row["requirement"]]["state"]
        assert row["audit_cited"] == audit_rows[row["requirement"]]["cited"]
        assert row["ok"] is False


def test_requirement_never_met_without_artifact_on_disk():
    """NEGATIVE CONTROL: a synthetic promotable NX that never touched disk is not MET."""
    synth = nx_audit.synthetic_promotable_nx()
    for req in nx_audit.SEVEN_REQUIREMENTS:
        judged = nnp.judge_requirement(req, nx=synth, nx_path=None)
        assert judged["ok"] is False
        assert judged["state"] == "REFUSED_WITHOUT_ARTIFACT"
        assert judged["artifact"] is None
    judged = nnp.judge_requirement(
        "capability",
        nx=synth,
        nx_path="/definitely/not/a/real/nx.json",
    )
    assert judged["ok"] is False
    assert judged["state"] == "REFUSED_WITHOUT_ARTIFACT"


def test_requirement_can_be_met_when_artifact_is_on_disk(tmp_path):
    """Inverse of the refusal: the checker must still be able to return MET."""
    synth = nx_audit.synthetic_promotable_nx()
    path = tmp_path / "promotable.nx.json"
    path.write_text(json.dumps(synth))
    judged = nnp.judge_requirement("capability", nx=synth, nx_path=path)
    assert judged["ok"] is True
    assert judged["state"] == "MET"
    assert judged["artifact"] == str(path)
    live = nx_audit.evidence_path(nx_audit.REL_NX_V0)
    if live is None:
        refused = nnp.judge_requirement("capability", nx=synth, nx_path=None)
        assert refused["ok"] is False
        return
    v0 = json.loads(live.read_text())
    live_judged = nnp.judge_requirement("capability", nx=v0, nx_path=live)
    assert live_judged["ok"] is False


def test_metadata_seal_cannot_satisfy_even_if_checker_is_lied_to(tmp_path):
    """A metadata NX on disk is still not a satisfying artifact."""
    path = nx_audit.evidence_path(nx_audit.REL_NX_V0)
    if path is None:
        judged = nnp.judge_requirement("self_contained_dependencies", nx={}, nx_path=None)
        assert judged["ok"] is False
        return
    nx = json.loads(path.read_text())
    assert nx_audit._status_is_metadata_only(nx)
    for req in nx_audit.SEVEN_REQUIREMENTS:
        judged = nnp.judge_requirement(req, nx=nx, nx_path=path)
        assert judged["ok"] is False


def test_physical_requirements_are_sleeping_never_pending():
    """NEGATIVE CONTROL: hardware-blocked is SLEEPING with a wake, never pending."""
    doc = _receipt()
    physical = {"accepted_generation", "capability", "protected_performance"}
    for row in doc["seven"]:
        if row["requirement"] not in physical:
            continue
        assert row["schedule"] == nnp.SLEEPING
        assert row["schedule"] not in nnp.FORBIDDEN_PHYSICAL_SCHEDULES
        assert "pending" not in str(row["schedule"]).lower()
        assert row["wake_conditions"], f"{row['requirement']} has no wake conditions"
        wu = row["sleeping_workunit"]
        assert wu is not None
        assert wu["status"] == "sleeping"
        assert wu["status"] not in nnp.FORBIDDEN_PHYSICAL_SCHEDULES
        assert wu["classification"] == "SLEEPING"
        assert wu["synthetic_result_forbidden"] is True
        assert any(not w["holds"] for w in row["wake_conditions"]), (
            f"{row['requirement']} is SLEEPING but every wake condition holds"
        )


def test_metal_gpu_present_does_not_flip_protected_performance_to_pending():
    """The whole point of today's re-derivation."""
    doc = _receipt()
    host = doc["today"]["host"]
    row = next(r for r in doc["seven"] if r["requirement"] == "protected_performance")
    if host.get("verdict") == "FALSIFIED_AS_A_HOST_PROPERTY":
        metal_wake = next(w for w in row["wake_conditions"] if w["id"] == "metal_gpu_present")
        assert metal_wake["holds"] is True
        assert row["schedule"] == nnp.SLEEPING
        assert "HOST_HAS_NO_GPU" not in row["blocker_classes"]
        assert nnp.PHYSICAL_AUTHORITY in row["blocker_classes"]
        lease_wake = next(w for w in row["wake_conditions"] if w["id"] == "gpu_lease_proven_holder")
        assert lease_wake["holds"] is False
    else:
        assert row["schedule"] == nnp.SLEEPING
        assert "pending" not in str(row["schedule"]).lower()


def test_teacher_1024_is_not_accepted_generation():
    """NEGATIVE CONTROL: a real capture must still fail the NX generation predicate."""
    facts = nnp.today_teacher_facts()
    judged = nnp.teacher_satisfies_accepted_generation(facts)
    assert judged["ok"] is False
    if facts.get("capture_present"):
        assert "source-BF16" in judged["why"] or "mlp_input" in judged["why"] or "oracle" in judged["why"]
        doc = _receipt()
        row = next(r for r in doc["seven"] if r["requirement"] == "accepted_generation")
        assert row["ok"] is False
        assert row["audit_state"] != "MET"
        assert row["schedule"] == nnp.SLEEPING
    else:
        assert "not on disk" in judged["why"]


def test_disagreement_with_audit_is_reported_not_resolved():
    """If the audit and this module disagree, that IS the finding."""
    doc = _receipt()
    audit_doc, via = nnp.load_rel(nnp.REL_AUDIT)
    if audit_doc is None:
        assert doc["audit"]["refused"] is True
        assert via == "missing"
        return
    state_disagreements = [
        d for d in doc["audit_disagreements"] if d.get("kind") == "requirement_state"
    ]
    for row in doc["seven"]:
        if row["state_agreement"] == "DISAGREE":
            assert any(d["requirement"] == row["requirement"] for d in state_disagreements)
            assert row["ok"] is False
        elif row["state_agreement"] == "AGREE":
            assert row["audit_state"] == row["live_checker_state"]
    host = doc["today"]["host"]
    if host.get("verdict") == "FALSIFIED_AS_A_HOST_PROPERTY":
        causal = [d for d in doc["audit_disagreements"] if d.get("kind") == "causal_label"]
        assert causal, "Metal GPU is present; the audit's 'sidecar has no GPU' causal label must be reported"
        assert any(d.get("verdict") == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM" for d in causal)
        for d in causal:
            assert d["resolution"] == "REPORTED_NOT_RESOLVED"


def test_inherited_no_gpu_causal_claim_is_rejected_when_metal_is_present():
    host = {
        "verdict": "FALSIFIED_AS_A_HOST_PROPERTY",
        "device_name": "Apple M3 Ultra",
        "gpu_authority": False,
    }
    claim = nnp.reclassify_causal_claim(
        "sidecar has no GPU; all numbers that would require a lease remain unclaimed here",
        host,
    )
    assert claim["verdict"] == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM"
    claim2 = nnp.reclassify_causal_claim(
        "MetalContext reports no Metal-capable GPU on the host of record",
        host,
    )
    assert claim2["verdict"] == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM"
    clean = nnp.reclassify_causal_claim(
        "sidecar has no GPU authority and will not flock the bench lock",
        host,
    )
    assert clean["verdict"] == "NOT_A_HOST_ABSENCE_CLAIM"


def test_host_absence_confirmed_is_the_negative_of_falsified():
    """NEGATIVE CONTROL: the classifier can still return CONFIRMED."""
    host = {"verdict": "CONFIRMED", "device_name": None, "gpu_authority": False}
    claim = nnp.reclassify_causal_claim("this host has no Metal-capable GPU", host)
    assert claim["verdict"] == "HOST_ABSENCE_CONFIRMED_BY_PROBE"
    untested = nnp.reclassify_causal_claim("this host has no Metal-capable GPU", {"verdict": "UNTESTED"})
    assert untested["verdict"] == "HOST_ABSENCE_UNTESTED"


def test_no_code_path_writes_a_physical_ebpw_value():
    """NEGATIVE CONTROL: record_physical_ebpw always raises."""
    with pytest.raises(nnp.PhysicalEbpwForbidden):
        nnp.record_physical_ebpw(0.887)
    with pytest.raises(nnp.PhysicalEbpwForbidden):
        nnp.record_physical_ebpw(16.0)
    with pytest.raises(nnp.PhysicalEbpwForbidden):
        nnp.record_physical_ebpw(None)
    doc = _receipt()
    nnp.assert_no_physical_ebpw(doc)
    with pytest.raises(nnp.PhysicalEbpwForbidden):
        nnp.assert_no_physical_ebpw({"physical_ebpw": 0.5})
    with pytest.raises(nnp.PhysicalEbpwForbidden):
        nnp.assert_no_physical_ebpw({"nested": {"qualified_complete_physical_ebpw": 1.0}})
    src = Path(nnp.__file__).read_text()
    tree = ast.parse(src)
    assigned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in nnp.PHYSICAL_EBPW_KEYS:
                    assigned.append(t.id)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in nnp.PHYSICAL_EBPW_KEYS:
                assigned.append(node.target.id)
    assert assigned == []


def test_fallback_zero_refused_on_metadata_nx():
    """NEGATIVE CONTROL: stamping 0 on the V0 metadata NX is a lie and must raise."""
    path = nx_audit.evidence_path(nx_audit.REL_NX_V0)
    if path is None:
        disc = nnp.fallback_disclosure({"status": nx_audit.METADATA_ONLY})
        assert disc["ok"] is False
        assert disc["would_be_a_lie"] is True
        with pytest.raises(nnp.FallbackDisclosureError, match="lie"):
            nnp.stamp_fallback_count({"status": nx_audit.METADATA_ONLY}, 0)
        return
    nx = json.loads(path.read_text())
    disc = nnp.fallback_disclosure(nx)
    assert disc["ok"] is False
    assert disc["would_be_a_lie"] is True
    with pytest.raises(nnp.FallbackDisclosureError, match="lie"):
        nnp.stamp_fallback_count(nx, 0)
    with pytest.raises(nnp.FallbackDisclosureError):
        nnp.stamp_fallback_count(nx, 1)


def test_fallback_disclosure_can_accept_a_real_zero():
    """Inverse: a source-independent NX with disclosed 0 is allowed by the judge."""
    synth = nx_audit.synthetic_promotable_nx()
    disc = nnp.fallback_disclosure(synth)
    assert disc["ok"] is True
    assert disc["fallback_count"] == 0
    missing = nnp.fallback_disclosure({"status": "SOURCE_INDEPENDENT_COMPLETE", "source_independent": True})
    assert missing["ok"] is False
    assert missing["would_be_a_lie"] is False


def test_complete_system_ledger_contract_does_not_close():
    docs_ok = True
    try:
        docs = nx_audit._load_all()
    except FileNotFoundError:
        docs_ok = False
        docs = {}
    contract = nnp.complete_system_ledger_contract(
        docs.get("ledger") if docs_ok else None,
        docs.get("executable") if docs_ok else None,
    )
    assert contract["closed"] is False
    assert contract["refused"] is True
    assert contract["missing_fields"]
    if docs_ok:
        assert contract["complete_storage_bytes_is_null"] is True
        assert contract["promotion_allowed"] is False
        assert "launder" in contract["refused_to_fill_from_exact_control"]


def test_check_nx_still_rejects_v0_requirements_were_not_lowered():
    path = nx_audit.evidence_path(nx_audit.REL_NX_V0)
    if path is None:
        judged = nnp.judge_requirement("protected_performance", nx=None, nx_path=None)
        assert judged["ok"] is False
        return
    nx = json.loads(path.read_text())
    ledger_p = nx_audit.evidence_path(nx_audit.REL_LEDGER)
    ctx = {"byte_ledger": json.loads(ledger_p.read_text())} if ledger_p else None
    result = nx_audit.check_nx(nx, context=ctx)
    assert result["promotable"] is False
    assert set(result["failed_requirements"]) == set(nx_audit.SEVEN_REQUIREMENTS)
    doc = _receipt()
    assert doc["live_nx"]["promotable"] is False
    assert doc["seven_all_met"] is False


def test_cpu_next_is_never_used_for_physical_requirements():
    doc = _receipt()
    for row in doc["seven"]:
        if row["requirement"] in nnp.PHYSICAL_TO_SATISFY:
            assert row["schedule"] != nnp.CPU_NEXT
            assert row["schedule"] != "pending"
            assert nnp.PHYSICAL_AUTHORITY in row["blocker_classes"]


def test_self_contained_is_work_not_done_not_host_gpu_absence():
    doc = _receipt()
    row = next(r for r in doc["seven"] if r["requirement"] == "self_contained_dependencies")
    assert nnp.WORK_NOT_DONE in row["blocker_classes"]
    assert "HOST_HAS_NO_GPU" not in row["blocker_classes"]
    assert row["ok"] is False
    assert row["schedule"] in {nnp.CPU_NEXT, nnp.BLOCKED_ON_PRIOR}


def test_missing_audit_is_a_refusal_not_a_default_met():
    got = nnp.seven_from_audit(None, "missing")
    assert got["refused"] is True
    assert got["seven_all_met"] is not True
    assert got["requirements"] == []
    got2 = nnp.seven_from_audit({"seven_all_met": True, "seven_requirements": []}, "short")
    assert got2["refused"] is True
    assert got2["seven_all_met"] is not False or got2["refused"] is True
    assert got2["requirements"] == []


def test_receipt_refuses_hardware_claims():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_nr_nx_path_hardware_probe.json",
            {"schema": "probe", "tps": 12.0},
            "tools/future/test_nr_nx_path.py",
        )
    src = Path(nnp.__file__).read_text()
    tree = ast.parse(src)
    numeric_hw = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in HARDWARE_FIELDS:
                    if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)):
                        numeric_hw.append((k.value, v.value))
    assert numeric_hw == [], f"module literals hardware fields {numeric_hw}"


def test_kernel_overlay_does_not_treat_source_compile_ok_as_a_wall():
    host = nnp.today_host_facts()
    try:
        docs = nx_audit._load_all()
    except FileNotFoundError:
        overlay = nnp.kernel_catalog_overlay({}, host)
        assert overlay["still_plan_only"] is not True or overlay["native_kernels_status"] is None
        return
    overlay = nnp.kernel_catalog_overlay(docs, host)
    if host.get("source_compile") == "OK":
        assert overlay["metal_compile_is_a_host_wall"] is False
        assert overlay["still_plan_only"] is True
        assert overlay["not_built_count"] > 0
    else:
        assert overlay["metal_compile_is_a_host_wall"] in {True, None, False}


def test_no_skipped_tests_and_no_stubs_in_module():
    src = Path(nnp.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            raise AssertionError(f"pass statement at line {node.lineno}")
        if isinstance(node, ast.Constant) and node.value == "TODO":
            raise AssertionError(f"TODO literal at line {node.lineno}")
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            fn = node.exc.func
            if isinstance(fn, ast.Name) and fn.id == "NotImplementedError":
                raise AssertionError(f"NotImplementedError at line {node.lineno}")
    test_tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "skip":
                raise AssertionError(f"skip call at line {node.lineno}")
            if isinstance(fn, ast.Name) and fn.id == "skip":
                raise AssertionError(f"skip call at line {node.lineno}")
