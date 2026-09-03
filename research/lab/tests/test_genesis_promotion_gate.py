"""Every promotion-gate clause is watched FAIL. Self-cert is refused."""
from __future__ import annotations

from typing import Any, Callable

import pytest

from lab.lineage.canon import labeled_sha
from lab.lineage.identity import Invoker, make_qwen38_genesis
from lab.lineage.promotion import (
    ALL_CLAUSES,
    CLAUSE_ARTIFACT_IDENTITY,
    CLAUSE_BENCHMARK_UNCHANGED,
    CLAUSE_BPW_UP_TOKEN_DOWN,
    CLAUSE_CAPABILITY,
    CLAUSE_COMPLETE_TOKEN_MATERIAL,
    CLAUSE_NO_NEW_SILENT_FALLBACK,
    CLAUSE_PROTECTED_TESTS,
    CLAUSE_REPRESENTATION_BPW,
    CLAUSE_ROLLBACK_ARTIFACT,
    CLAUSE_RUNTIME_GENOME,
    CLAUSE_STATE_TRANSFER,
    CLAUSE_TPS_UP_CAP_DOWN,
    SelfCertificationRefused,
    clause_status,
    evaluate_promotion,
)
from lab.receipts import verify
from lab.lineage.testing import (
    CHILD_WALL,
    PARENT_REPS,
    armed_lineage,
    make_child,
    passing_evidence,
)


def _accept_bundle():
    return armed_lineage()


def test_happy_path_accepts_when_every_clause_passes() -> None:
    state, parent, child, ev, inv = _accept_bundle()
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    verify(verdict, label="promotion")
    assert verdict["verdict"] == "ACCEPT"
    assert verdict["fabricated_accept"] is False
    assert verdict["authority_level"] == "authoritative"
    assert [c["name"] for c in verdict["checks"]] == list(ALL_CLAUSES)
    assert {c["status"] for c in verdict["checks"]} == {"PASS"}


def test_missing_evidence_is_pending_never_accept() -> None:
    state, parent, child, _ev, inv = _accept_bundle()
    verdict = evaluate_promotion(parent=parent, child=child, evidence={}, invoker=inv, lineage=state)
    assert verdict["verdict"] == "PENDING"
    assert verdict["fabricated_accept"] is False
    assert verdict["authority_level"] == "pending"


def test_parent_self_certification_refused() -> None:
    state, parent, child, ev, _inv = _accept_bundle()
    rogue = Invoker(principal="parent", identity=parent.instance_id, acting_as="parent")
    with pytest.raises(SelfCertificationRefused, match="self-certification refused"):
        evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=rogue, lineage=state)
    # Identity match even with an otherwise-legal principal name is refused.
    impersonate = Invoker(principal="lineage_gate", identity=parent.instance_id)
    with pytest.raises(SelfCertificationRefused, match="parent or the child"):
        evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=impersonate, lineage=state)


def test_child_self_certification_refused() -> None:
    state, parent, child, ev, _inv = _accept_bundle()
    rogue = Invoker(principal="child", identity=child.instance_id, acting_as="child")
    with pytest.raises(SelfCertificationRefused, match="self-certification refused"):
        evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=rogue, lineage=state)
    impersonate = Invoker(principal="protected_controller", identity=child.instance_id)
    with pytest.raises(SelfCertificationRefused, match="parent or the child"):
        evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=impersonate, lineage=state)


def test_sandbox_model_cannot_invoke_gate() -> None:
    state, parent, child, ev, _inv = _accept_bundle()
    with pytest.raises(SelfCertificationRefused, match="sandbox_model"):
        evaluate_promotion(
            parent=parent,
            child=child,
            evidence=ev,
            invoker=Invoker(principal="sandbox_model", identity="qwen38"),
            lineage=state,
        )


Corruptor = Callable[[Any, Any, dict[str, Any]], tuple[Any, dict[str, Any]]]


def _cap_down(parent, child, ev):
    weak = make_child(parent, capability={**child.capability, "engineering": 0.0})
    return weak, ev


def _new_fallback(parent, child, ev):
    fallen = make_child(parent, silent_fallback_ids=("silent_cpu_path",))
    ev = dict(ev)
    ev["new_silent_fallbacks"] = ["silent_cpu_path"]
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["artifact_sha"] = fallen.artifact_sha
    ev["artifact_receipt"] = {"sha": fallen.artifact_sha}
    ev["genome"] = {
        "runtime_sha": fallen.runtime_sha,
        "kernel_genome_sha": fallen.kernel_genome_sha,
    }
    ev["representation"] = {"bpw": fallen.representation_bpw}
    return fallen, ev


def _token_not_better(parent, child, ev):
    ev = dict(ev)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["complete_token_ns_reps"] = [40_000_000, 40_100_000, 39_900_000]
    return child, ev


def _token_not_material(parent, child, ev):
    ev = dict(ev)
    ev["measurement"] = dict(ev["measurement"])
    # ~0.01% faster — not material.
    parent_mean = sum(PARENT_REPS) / 3
    tiny = int(parent_mean - 1_000)
    ev["measurement"]["complete_token_ns_reps"] = [tiny, tiny + 50, tiny - 50]
    return child, ev


def _artifact_swap(parent, child, ev):
    ev = dict(ev)
    ev["artifact_receipt"] = {"sha": labeled_sha("someone-elses-artifact")}
    return child, ev


def _bpw_mismatch(parent, child, ev):
    ev = dict(ev)
    ev["representation"] = {"bpw": 1.111111}
    return child, ev


def _genome_swap(parent, child, ev):
    ev = dict(ev)
    ev["genome"] = {
        "runtime_sha": labeled_sha("other-runtime"),
        "kernel_genome_sha": child.kernel_genome_sha,
    }
    return child, ev


def _protected_fail(parent, child, ev):
    ev = dict(ev)
    ev["protected_tests"] = [
        {"name": "coherence_greedy_ids", "status": "FAIL"},
        {"name": "complete_token_ledger_closed", "status": "PASS"},
        {"name": "no_silent_fallback", "status": "PASS"},
    ]
    return child, ev


def _protected_skip(parent, child, ev):
    ev = dict(ev)
    ev["protected_tests"] = [
        {"name": "coherence_greedy_ids", "status": "SKIP"},
        {"name": "complete_token_ledger_closed", "status": "PASS"},
        {"name": "no_silent_fallback", "status": "PASS"},
    ]
    return child, ev


def _transfer_unverified(parent, child, ev):
    ev = dict(ev)
    ev["state_transfer"] = {"checksum_verified": False, "checksum_sha256": ""}
    return child, ev


def _benchmark_changed(parent, child, ev):
    other = make_child(parent, benchmark_fingerprint=labeled_sha("different-bench"))
    ev = passing_evidence(parent, other)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["benchmark_fingerprint"] = other.benchmark_fingerprint
    ev["benchmark_changed"] = True
    return other, ev


def _bpw_win_token_loss(parent, child, ev):
    slimmer = make_child(parent, representation_bpw=3.0, complete_token_ns=parent.complete_token_ns + 5_000_000)
    ev = passing_evidence(parent, slimmer)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["complete_token_ns_reps"] = [40_000_000, 40_200_000, 39_800_000]
    ev["child_tps"] = slimmer.tps
    return slimmer, ev


CLAUSE_CASES: list[tuple[str, str, Corruptor]] = [
    ("capability_below_contract", CLAUSE_CAPABILITY, _cap_down),
    ("new_silent_fallback", CLAUSE_NO_NEW_SILENT_FALLBACK, _new_fallback),
    ("complete_token_worse", CLAUSE_COMPLETE_TOKEN_MATERIAL, _token_not_better),
    ("complete_token_not_material", CLAUSE_COMPLETE_TOKEN_MATERIAL, _token_not_material),
    ("artifact_identity_swapped", CLAUSE_ARTIFACT_IDENTITY, _artifact_swap),
    ("representation_bpw_mismatch", CLAUSE_REPRESENTATION_BPW, _bpw_mismatch),
    ("runtime_genome_mismatch", CLAUSE_RUNTIME_GENOME, _genome_swap),
    ("protected_test_failed", CLAUSE_PROTECTED_TESTS, _protected_fail),
    ("protected_test_skipped", CLAUSE_PROTECTED_TESTS, _protected_skip),
    ("state_transfer_unverified", CLAUSE_STATE_TRANSFER, _transfer_unverified),
    ("benchmark_changed", CLAUSE_BENCHMARK_UNCHANGED, _benchmark_changed),
    ("bpw_improved_token_worse", CLAUSE_BPW_UP_TOKEN_DOWN, _bpw_win_token_loss),
]


@pytest.mark.parametrize("label,clause,corrupt", CLAUSE_CASES, ids=[c[0] for c in CLAUSE_CASES])
def test_each_clause_rejects_when_input_corrupted(label: str, clause: str, corrupt: Corruptor) -> None:
    state, parent, child, ev, inv = _accept_bundle()
    out = corrupt(parent, child, ev)
    if len(out) == 3:
        child, ev, state = out
    else:
        child, ev = out
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT", (label, verdict)
    assert clause_status(verdict, clause) == "FAIL", (label, verdict["checks"])
    assert verdict["fabricated_accept"] is False


def test_rollback_artifact_clause_rejects_when_lkg_cleared() -> None:
    state, parent, child, ev, inv = _accept_bundle()
    state._put("LAST_KNOWN_GOOD", None)
    ev = dict(ev)
    ev.pop("rollback_artifact", None)
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_ROLLBACK_ARTIFACT) == "FAIL"


def test_tps_up_capability_down_named_reject() -> None:
    state, parent, child, ev, inv = _accept_bundle()
    # Child is faster (TPS up) and drops a required axis.
    weak = make_child(parent, capability={**child.capability, "coherence": 0.0})
    ev = passing_evidence(parent, weak)
    verdict = evaluate_promotion(parent=parent, child=weak, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_TPS_UP_CAP_DOWN) == "FAIL"
    assert clause_status(verdict, CLAUSE_CAPABILITY) == "FAIL"


def test_cpu_wait_proxy_timing_rejected() -> None:
    state, parent, child, ev, inv = _accept_bundle()
    ev = dict(ev)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["timing_authority"] = "cpu_wait_proxy"
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_COMPLETE_TOKEN_MATERIAL) == "FAIL"
    detail = next(c["detail"] for c in verdict["checks"] if c["name"] == CLAUSE_COMPLETE_TOKEN_MATERIAL)
    assert "CPU-wait proxy" in detail


def test_component_win_without_token_move_is_reject() -> None:
    """A component improvement that does not move the complete token is a negative."""
    state, parent, child, ev, inv = _accept_bundle()
    ev = dict(ev)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["complete_token_ns_reps"] = list(PARENT_REPS)
    ev["component_win_ns"] = 2_000_000  # claimed, ignored
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_COMPLETE_TOKEN_MATERIAL) == "FAIL"


def test_external_invoker_can_reject_and_accept() -> None:
    state, parent, child, ev, inv = _accept_bundle()
    assert inv.identity != parent.instance_id
    assert inv.identity != child.instance_id
    red = dict(ev)
    red["representation"] = {"bpw": 9.999}
    rejected = evaluate_promotion(parent=parent, child=child, evidence=red, invoker=inv, lineage=state)
    assert rejected["verdict"] == "REJECT"
    accepted = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert accepted["verdict"] == "ACCEPT"


def test_qwen38_parent_identity_is_the_seated_genesis() -> None:
    parent = make_qwen38_genesis()
    assert parent.complete_token_ns == 35_227_918
    assert parent.identity["binary"] == "ascension_qwen38_hybrid_greedy"
    assert parent.generation == 0
    # CHILD_WALL is used by the passing child so the gate has a real improvement.
    assert CHILD_WALL < parent.complete_token_ns
