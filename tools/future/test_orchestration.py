"""The connector must make its claims TRUE, not merely turn a metric green.

The whole risk of this module is Goodharting: it exists because an audit said
`operational=0 of 74`, and it is the thing that moves that number. So the tests
are weighted toward proving the binding describes reality — a binding that names
a module which does not write the receipt it claims must be REJECTED, and
`invoke()` must actually run the module and actually produce the receipt.
"""
import json
from pathlib import Path

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


_LOCK_RELS = (
    ".hcli/locks/protected-accelerator-bench.lock",
    ".hcli/locks/qwen-protected-bench.lock",
)


def _lock_mtime_snapshot() -> dict[str, tuple[int, int] | None]:
    paths = [REPO / rel for rel in _LOCK_RELS]
    paths.append(Path("/tmp/hawking_protected_window.lease"))
    out: dict[str, tuple[int, int] | None] = {}
    for p in paths:
        try:
            st = p.stat()
            out[str(p)] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            out[str(p)] = None
    return out


def test_protected_scheduler_is_bound_and_callable_through_orchestration():
    """An unbound module is not resident-callable. This one is bound and called."""
    key = "protected_scheduler.py"
    assert key in orch.BINDINGS, "protected_scheduler must be in BINDINGS"
    frontier, species = orch.BINDINGS[key]
    assert frontier == "FT.GPU_KERNELS.ready-protected"
    assert species == "PROTECTED_SCHEDULER"

    locks_dir = REPO / ".hcli" / "locks"
    existed = locks_dir.exists()
    before = _lock_mtime_snapshot()

    # Resident path: BINDINGS allowlist, then the module's decision API.
    report = orch.call_bound(key, "capability_report")
    assert report["PROTECTED_SCHEDULER_CAPABLE"] is True
    assert report["PROTECTED_WINDOW_AVAILABLE"] is False
    assert report["did_not_fabricate_lease"] is True
    assert report["did_not_flock"] is True
    assert report["did_not_touch_lock_file"] is True

    probe = {
        "id": "future.orchestration.protected-probe",
        "resource_class": "GPU_EXCLUSIVE",
        "requires_quiescence": True,
    }
    decision = orch.call_bound(
        key,
        "decide",
        probe,
        contamination={"contamination_class": "HEAVY"},
        lease={"present": False, "holders": {"pids": []}},
    )
    assert decision["scheduler_capable"] is True
    assert decision["window_available"] is False
    assert decision["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    parked = orch.call_bound(
        key,
        "park",
        probe,
        contamination={"contamination_class": "HEAVY"},
        lease={"present": False, "holders": {"pids": []}},
    )
    assert parked["parked"] is True
    assert parked["scheduler_capable"] is True

    from tools.future import protected_scheduler as ps
    from tools.future import qualification_pipeline as qp

    with pytest.raises(ps.SchedulerRefused):
        orch.call_bound(key, "acquire_lease")
    with pytest.raises(ps.SchedulerRefused):
        orch.call_bound(key, "seize_lease")
    with pytest.raises(qp.AuthorityBoundaryError):
        orch.call_bound(key, "refuse_flock")

    after = _lock_mtime_snapshot()
    assert after == before
    if not existed:
        assert not locks_dir.exists(), "bound call must not create .hcli/locks"

    # invoke() would call build() and rewrite PROTECTED_SCHEDULER.json from a
    # module this lane cannot edit. Routing is still declared on the binding.
    wu = orch.emit_workunit(key)
    assert wu["frontier_item"] == frontier
    assert wu["species"] == species
    assert wu["module"] == "tools/future/protected_scheduler.py"
    assert wu["gpu_authority"] is False
    assert wu["evidence_class"] == "STATIC_ONLY"
    assert wu["output_contract"] == "receipts/future/PROTECTED_SCHEDULER.json"
    receipt = REPO / "receipts" / "future" / "PROTECTED_SCHEDULER.json"
    assert receipt.is_file()
    doc = json.loads(receipt.read_text())
    assert doc["capability"]["PROTECTED_SCHEDULER_CAPABLE"] is True
    assert doc["capability"]["PROTECTED_WINDOW_AVAILABLE"] is False
    assert doc["gpu_authority"] is False
    assert doc["did_not_fabricate_lease"] is True
    assert doc["did_not_flock"] is True
    assert doc["did_not_touch_lock_file"] is True

    after_emit = _lock_mtime_snapshot()
    assert after_emit == before
    if not existed:
        assert not locks_dir.exists()


def test_call_bound_fails_closed_on_unbound_or_missing_fn():
    with pytest.raises(orch.UnknownBinding):
        orch.call_bound("not_a_real_module.py", "capability_report")
    with pytest.raises(orch.UnknownBinding):
        orch.bound_module("not_a_real_module.py")
    with pytest.raises(orch.BindingError):
        orch.call_bound("protected_scheduler.py", "this_function_does_not_exist")


def test_mutation_engine_is_bound_and_callable_through_orchestration(tmp_path):
    """An unbound engine is not resident-callable. This one is bound and called.

    autonomy_run.py is not edited. The resident reaches propose/apply/
    evidence/rollback/verdict through BINDINGS (resident_mutation_engine
    / call_bound). invoke() would rewrite MUTATION_ENGINE.json from a
    module this lane cannot edit.
    """
    key = "mutation_engine.py"
    assert key in orch.BINDINGS, "mutation_engine must be in BINDINGS"
    frontier, species = orch.BINDINGS[key]
    assert frontier == "FT.HCLI_SELF.emit-workunits"
    assert species == "MUTATION_ENGINE"

    locks_dir = REPO / ".hcli" / "locks"
    existed = locks_dir.exists()
    before = _lock_mtime_snapshot()

    from tools.future import mutation_engine as me

    me.unbind()
    try:
        engine = orch.resident_mutation_engine(tmp_path)
        assert engine is me._need()
        proposed = orch.call_bound(key, "propose", "FT.HCLI_SELF.emit-workunits")
        assert proposed["mutation_class"] == me.PIPELINE_SELF
        assert proposed["state"] == "PROPOSED"
        applied = orch.call_bound(key, "apply", proposed)
        assert applied["state"] == "APPLIED"
        assert applied["before_digest"] != applied["after_digest"]
        ev = orch.call_bound(key, "evidence", proposed)
        assert ev["digest_changed"] is True
        decided = orch.call_bound(key, "verdict", proposed)
        assert decided["verdict"] in me.VERDICTS
        if decided["verdict"] == me.VERDICT_KEPT:
            undone = orch.call_bound(key, "rollback", proposed)
            assert undone["digest_match"] is True
        else:
            assert decided["verdict"] == me.VERDICT_ROLLED_BACK
            assert decided.get("digest_match") is True
    finally:
        orch.call_bound(key, "unbind")

    after = _lock_mtime_snapshot()
    assert after == before
    if not existed:
        assert not locks_dir.exists(), "bound mutation_engine must not create .hcli/locks"

    wu = orch.emit_workunit(key)
    assert wu["frontier_item"] == frontier
    assert wu["species"] == species
    assert wu["module"] == "tools/future/mutation_engine.py"
    assert wu["output_contract"] == "receipts/future/MUTATION_ENGINE.json"
    assert wu["gpu_authority"] is False
    receipt = REPO / "receipts" / "future" / "MUTATION_ENGINE.json"
    assert receipt.is_file()
