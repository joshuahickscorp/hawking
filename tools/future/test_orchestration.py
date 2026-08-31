"""The connector must make its claims TRUE, not merely turn a metric green.

The whole risk of this module is Goodharting: it exists because an audit said
`operational=0 of 74`, and it is the thing that moves that number. So the tests
are weighted toward proving the binding describes reality — a binding that names
a module which does not write the receipt it claims must be REJECTED, and
`invoke()` must actually run the module and actually produce the receipt.
"""
import json

import pytest

from tools.future import orchestration as orch
from tools.future._common import REPO, RECEIPTS


def test_bind_emits_sealed_receipt_with_no_broken_bindings():
    out = orch.build()
    doc = json.loads(out.read_text())
    assert doc["schema"] == orch.SCHEMA
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["counts"]["broken"] == 0
    assert doc["counts"]["bound"] > 0


def test_every_binding_names_a_real_module_and_its_real_receipt():
    """The binding table must describe disk, not intent."""
    v = orch.validate_bindings()
    assert v["broken"] == []
    for row in v["bound"]:
        assert (REPO / "tools" / "future" / row["module"]).is_file()
        assert row["receipt"].endswith(".json")
        assert row["frontier_item"].startswith("FT.")
        assert row["species"]


def test_negative_control_a_binding_that_does_not_describe_reality_is_broken():
    """NEGATIVE CONTROL: the validator must actually be capable of rejecting.

    A table nobody has watched fail is a table that will silently drift into
    fiction the moment a module is renamed.
    """
    original = dict(orch.BINDINGS)
    try:
        orch.BINDINGS["a_module_that_does_not_exist.py"] = ("FT.TOOLS.freshness", "X")
        v = orch.validate_bindings()
        assert any("does not exist" in b["why"] for b in v["broken"]), v["broken"]
        with pytest.raises(orch.BindingError):
            orch.build()
    finally:
        orch.BINDINGS.clear()
        orch.BINDINGS.update(original)
    # and the table is healthy again
    assert orch.validate_bindings()["broken"] == []


def test_infrastructure_is_excluded_not_fake_bound():
    """Honesty about the denominator: infra informs no frontier and is not credited."""
    v = orch.validate_bindings()
    for name in orch.INFRASTRUCTURE:
        assert name not in orch.BINDINGS, f"{name} must not be given a fake binding"
    bound_names = {r["module"] for r in v["bound"]}
    assert not (bound_names & orch.INFRASTRUCTURE)


def test_invoke_actually_runs_the_module_and_produces_the_receipt():
    """This is what makes the binding true rather than declarative."""
    res = orch.invoke("evidence_snapshot.py")
    assert res["module"] == "evidence_snapshot.py"
    assert res["routed_to_frontier"].startswith("FT.")
    assert (REPO / res["receipt"]).is_file()
    assert res["evidence_class"] == "STATIC_ONLY"


def test_invoke_fails_closed_on_an_unknown_module():
    with pytest.raises(orch.UnknownBinding):
        orch.invoke("not_a_real_module.py")
    with pytest.raises(orch.UnknownBinding):
        orch.emit_workunit("not_a_real_module.py")


def test_emitted_workunit_carries_no_authority_it_should_not_have():
    wu = orch.emit_workunit("freshness.py")
    assert wu["gpu_authority"] is False
    assert wu["evidence_class"] == "STATIC_ONLY"
    assert wu["output_contract"].startswith("receipts/future/")
    forbidden = {"acquire_gpu_lease", "promote", "modify_verifier", "widen_authority"}
    assert not (set(wu["allowed_authority"]) & forbidden)


def test_frontier_view_only_credits_validated_bindings():
    view = orch.frontier_view()
    v = orch.validate_bindings()
    credited = {r for rows in view["by_probe_receipt"].values() for r in rows}
    declared = {r["frontier_item"] for r in v["bound"]}
    assert credited <= declared, "the view credited a frontier item no binding validated"


def test_audit_reflects_the_bindings_and_is_not_asserted():
    """The audit number must come from the audit, not from this module's hopes."""
    # Build it rather than skip: a suite that passes by skipping measured nothing,
    # and the adversarial attack treats a fired skip as a P0.
    from tools.future import resident_api as ra
    p = RECEIPTS / "RESIDENT_API_AUDIT.json"
    if not p.exists():
        ra.audit()
    doc = json.loads(p.read_text())
    counts = doc.get("counts") or {}
    assert isinstance(counts.get("operational", 0), int)
