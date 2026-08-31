"""Negative controls for the resident mutation cycle.

A mutation engine nobody has watched refuse is a way to break the system
autonomously. These tests prove rollback by digest, prove KEPT cannot be
minted on dirty evidence, prove a Codex target is refused before a write,
prove same-file conflict, and prove no path emits a hardware number.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import mutation_engine as me
from tools.future import mutation_surface as ms
from tools.future import dirty_measure as dm
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)
from tools.future.contamination import PromotionRefused


@pytest.fixture
def engine(tmp_path: Path) -> me.MutationEngine:
    return me.MutationEngine(tmp_path)


def test_build_emits_sealed_receipt():
    out = me.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == me.RECEIPT
    assert doc["schema"] == me.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["proofs"]["all_hold"] is True
    assert doc["completable_here"] == [me.PIPELINE_SELF]
    assert doc["resident_callable"]["frontier"] == "FT.HCLI_SELF.emit-workunits"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["proofs"]["pipeline_self"]["verdict"] == me.VERDICT_KEPT
    assert doc["proofs"]["pipeline_self"]["rollback_digest_match"] is True


def test_rollback_restores_byte_identical_prior_state(engine: me.MutationEngine):
    """NEGATIVE CONTROL: undo is measured by digest, not declared."""
    proposed = engine.propose("FT.HCLI_SELF.emit-workunits")
    target = engine.scope / proposed["target"]
    before_bytes = target.read_bytes()
    before = hashlib.sha256(before_bytes).hexdigest()
    engine.apply(proposed)
    assert hashlib.sha256(target.read_bytes()).hexdigest() != before
    rb = engine.rollback(proposed)
    assert rb["digest_match"] is True
    assert rb["byte_identical"] is True
    after_bytes = target.read_bytes() if target.is_file() else b""
    assert hashlib.sha256(after_bytes).hexdigest() == before
    assert after_bytes == before_bytes
    assert rb["restored_digest"] == before
    assert rb["verdict"] == me.VERDICT_ROLLED_BACK


def test_protected_mutation_never_reports_kept_on_dirty_evidence(engine: me.MutationEngine):
    """NEGATIVE CONTROL: KERNEL_OR_GPU cannot KEPT without a protected window."""
    proposed = engine.propose("FT.GPU_KERNELS.ready-protected")
    assert proposed["needs_protected_window"] is True
    engine.apply(proposed)
    ev = engine.evidence(proposed)
    assert ev["evidence_class"] == dm.EVIDENCE_CLASS
    assert ev["parking"] == me.PARK_PROTECTED
    decided = engine.verdict(proposed)
    assert decided["verdict"] != me.VERDICT_KEPT
    assert decided["verdict"] == me.VERDICT_INCONCLUSIVE
    assert decided["parking"] == me.PARK_PROTECTED
    assert decided["promotable"] is False
    with pytest.raises(PromotionRefused):
        dm.offer_for_promotion(
            {
                "evidence_class": ev["evidence_class"],
                "measurement_class": "STATIC_ONLY",
                "gpu_authority": False,
            }
        )


def test_token_rate_and_representation_are_inconclusive(engine: me.MutationEngine):
    for frontier in ("FT.TPS.protected-tps", "FT.MODEL_REPRESENTATION.ngram-school"):
        m = engine.propose(frontier)
        engine.apply(m)
        v = engine.verdict(m)
        assert v["verdict"] == me.VERDICT_INCONCLUSIVE
        assert v["verdict"] != me.VERDICT_KEPT
        engine.rollback(m)


def test_mutation_outside_sidecar_refused_before_apply(engine: me.MutationEngine):
    """NEGATIVE CONTROL: Codex target never reaches a write."""
    crate = "crates/hawking-core/src/engine.rs"
    assert ms.owner(crate) == "CODEX"
    assert ms.intersects_codex(crate) is True
    dest = engine.scope / crate
    with pytest.raises(me.PartitionRefused, match="refused before apply"):
        engine.propose("FT.GPU_KERNELS.static-warnings", target=crate)
    assert dest.exists() is False
    with pytest.raises(me.PartitionRefused):
        me.refuse_if_outside_partition("hcli/mutation.py")
    with pytest.raises(me.PartitionRefused):
        me.refuse_if_outside_partition("tools/accelerator/bench.py")
    with pytest.raises(me.PartitionRefused):
        me.refuse_if_outside_partition("receipts/headless/ACCELERATOR_SCOREBOARD.json")


def test_same_file_conflict_detected(engine: me.MutationEngine):
    """NEGATIVE CONTROL: two mutations cannot occupy one file."""
    a = engine.propose("FT.HCLI_SELF.emit-workunits")
    b = engine.propose("FT.TOOLS.frontiers-refill")
    assert a["target"] == b["target"]
    engine.apply(a)
    with pytest.raises(me.MutationConflictError, match="both touch"):
        engine.apply(b)
    assert isinstance(me.MutationConflictError, type)
    assert issubclass(me.MutationConflictError, me.cp.IncompatibleMutationError)


def test_no_code_path_writes_a_hardware_performance_number(engine: me.MutationEngine):
    """NEGATIVE CONTROL: stuffing a hardware field is refused, not stored."""
    with pytest.raises((HardwareClaimError, dm.DirtyMagnitudeRefused)):
        engine.propose("FT.HCLI_SELF.emit-workunits", change={"tps": 120.0})
    with pytest.raises((HardwareClaimError, dm.DirtyMagnitudeRefused)):
        engine.propose("FT.TPS.protected-tps", change={"accepted_tps": 8})
    with pytest.raises((HardwareClaimError, dm.DirtyMagnitudeRefused)):
        engine.propose(
            "FT.GPU_KERNELS.ready-protected", change={"gpu_ns": 12, "fusion": "on"}
        )
    m = engine.propose("FT.HCLI_SELF.emit-workunits")
    applied = engine.apply(m)
    ev = engine.evidence(m)
    v = engine.verdict(m)
    rb = engine.rollback(m)
    for node in (m, applied, ev, v, rb):
        _assert_no_hardware_claims(node)
        for key in HARDWARE_FIELDS:
            if key in node and isinstance(node[key], (int, float)):
                raise AssertionError(f"{key} = {node[key]!r} leaked into a public record")


def test_pipeline_self_end_to_end_kept_and_undoable(engine: me.MutationEngine):
    cycle = me.pipeline_self_cycle(engine)
    assert cycle["verdict"]["verdict"] == me.VERDICT_KEPT
    work = cycle["evidence"]["work"]
    assert work["unique_frontier_ids_after"] >= work["unique_frontier_ids_before"]
    assert work["units_queued_after"] < work["units_queued_before"]
    assert work["replays_skipped_after"] > work["replays_skipped_before"]
    assert work["units_queued_before"] == me.TRIAL_REFILL_COUNT * len(me.TRIAL_REFILL_IDS)
    assert work["units_queued_after"] == len(me.TRIAL_REFILL_IDS)
    assert cycle["rollback_digest_match"] is True
    policy = json.loads((engine.scope / me.POLICY_NAME).read_text())
    assert policy["refill_identity"] == me.RECOVERED_REFILL_IDENTITY


def test_harmful_pipeline_self_is_rolled_back(engine: me.MutationEngine):
    """NEGATIVE CONTROL: a mutation that loses unique work cannot KEPT."""
    m = engine.propose(
        "FT.VERIFICATION.repro",
        change={"stop_after_first": True, "refill_identity": "frontier_module"},
    )
    engine.apply(m)
    v = engine.verdict(m)
    assert v["verdict"] == me.VERDICT_ROLLED_BACK
    assert v["verdict"] != me.VERDICT_KEPT
    assert v["digest_match"] is True


def test_resident_artifact_inconclusive_via_succession_vocabulary(engine: me.MutationEngine):
    m = engine.propose("FT.CHILD_RESIDENT.install-dry-run")
    assert m["mutation_class"] == me.RESIDENT_ARTIFACT
    engine.apply(m)
    v = engine.verdict(m)
    assert v["verdict"] == me.VERDICT_INCONCLUSIVE
    child = json.loads((engine.scope / m["target"]).read_text())
    assert child["succession_verdict"] == me.succ.VERDICT_INSUFFICIENT
    engine.rollback(m)


def test_absent_inputs_refuse_rather_than_succeed(engine: me.MutationEngine):
    with pytest.raises(me.MutationRefused, match="absent frontier"):
        engine.propose("")
    with pytest.raises(me.MutationRefused, match="no id"):
        engine.apply({})
    m = engine.propose("FT.HCLI_SELF.emit-workunits")
    with pytest.raises(me.MutationRefused, match="evidence requires"):
        engine.evidence(m)
    with pytest.raises(me.MutationRefused, match="verdict requires"):
        engine.verdict(m)
    with pytest.raises(me.MutationRefused, match="rollback is for APPLIED"):
        engine.rollback(m)
    with pytest.raises(me.MutationRefused, match="empty change"):
        engine.propose("FT.HCLI_SELF.emit-workunits", change={})
    with pytest.raises(me.MutationRefused, match="unknown mutation class"):
        engine.propose("FT.HCLI_SELF.emit-workunits", mutation_class="NOT_A_CLASS")


def test_unbound_module_level_propose_refuses():
    """NEGATIVE CONTROL: no bound scope means no live-tree mutation."""
    me.unbind()
    with pytest.raises(me.MutationRefused, match="no engine bound"):
        me.propose("FT.HCLI_SELF.emit-workunits")


def test_apply_never_calls_acquire_lease(
    engine: me.MutationEngine, monkeypatch: pytest.MonkeyPatch
):
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("apply seized a lease")

    monkeypatch.setattr(me.pw, "acquire_lease", boom)
    m = engine.propose("FT.GPU_KERNELS.ready-protected")
    engine.apply(m)
    assert called["n"] == 0


def test_refuse_protected_lease_is_the_named_raise():
    with pytest.raises(me.pw.WindowRefused):
        me.refuse_protected_lease()


def test_all_five_classes_are_proposeable(engine: me.MutationEngine):
    mapping = {
        me.KERNEL_OR_GPU: "FT.GPU_KERNELS.ready-protected",
        me.REPRESENTATION_BPW: "FT.MODEL_REPRESENTATION.ngram-school",
        me.TOKEN_RATE: "FT.TPS.protected-tps",
        me.PIPELINE_SELF: "FT.HCLI_SELF.emit-workunits",
        me.RESIDENT_ARTIFACT: "FT.CHILD_RESIDENT.install-dry-run",
    }
    for klass, frontier in mapping.items():
        m = engine.propose(frontier)
        assert m["mutation_class"] == klass
        assert m["state"] == "PROPOSED"
        assert m["gpu_authority"] is False


def test_module_has_no_placeholder_and_no_skip():
    src = Path(me.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [
                n
                for n in node.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
            ]
            assert not (
                len(body) == 1 and isinstance(body[0], ast.Pass)
            ), f"{node.name} body is pass"
            for n in body:
                if isinstance(n, ast.Raise) and "NotImplementedError" in ast.dump(n):
                    raise AssertionError(f"{node.name} raises NotImplementedError")
    test_src = Path(__file__).read_text()
    test_tree = ast.parse(test_src)
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Call):
            dump = ast.dump(node)
            assert "pytest.skip" not in dump
            assert "skip(" not in dump or "pytest" not in dump


def test_recovered_constants_match_autonomy_run():
    policy = me.recovered_pipeline_policy()
    assert policy["refill_watermark"] == me.ar.REFILL_WATERMARK
    assert policy["refill_every"] == me.ar.REFILL_EVERY
    assert policy["refill_interval_s"] == me.ar.REFILL_INTERVAL_S
    assert policy["refill_identity"] == me.RECOVERED_REFILL_IDENTITY
    recovered = me.simulate_trial_refills(policy)
    assert recovered["units_queued"] == 100
    assert recovered["unique_frontier_ids"] == 25
    assert recovered["busywork"] is True
    improved = dict(policy)
    improved["refill_identity"] = "frontier_module"
    improved["identity_committed_at"] = "queue"
    after = me.simulate_trial_refills(improved)
    assert after["units_queued"] == 25
    assert after["replays_skipped"] == 75
    assert after["busywork"] is False
