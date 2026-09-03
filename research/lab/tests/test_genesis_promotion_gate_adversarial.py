"""The six forged accepts, and every new clause, watched going red.

These cases were ACCEPT against the paperwork gate. Each must REJECT.
A test that only walks the happy path is what opened the hole.
"""
from __future__ import annotations

from typing import Any

import pytest

from lab.lineage.canon import labeled_sha
from lab.lineage.identity import DEFAULT_BENCHMARK_FINGERPRINT
from lab.lineage.promotion import (
    ALL_CLAUSES,
    CLAUSE_ARTIFACT_IDENTITY,
    CLAUSE_CAPABILITY,
    CLAUSE_COMPLETE_TOKEN_MATERIAL,
    CLAUSE_GENERATION,
    CLAUSE_GREEDY_TOKEN_IDS,
    CLAUSE_MODEL_IDENTITY,
    CLAUSE_STATE_TRANSFER,
    CLAUSE_TPS_UP_CAP_DOWN,
    NS_PER_SECOND,
    REQUIRED_PROTECTED_TESTS,
    clause_status,
    evaluate_promotion,
)
from lab.lineage.testing import (
    CHILD_REPS,
    PARENT_REPS,
    armed_lineage,
    make_child,
    passing_evidence,
)
from lab.lineage.transfer import payload_checksum, parent_research_payload


def _legacy_paperwork_evidence(parent, child) -> dict[str, Any]:
    """The evidence shape that produced the six ACCEPTs: asserted, not computed."""
    return {
        "measurement": {
            "artifact_sha": child.artifact_sha,
            "complete_token_ns_reps": list(CHILD_REPS),
            "parent_complete_token_ns_reps": list(PARENT_REPS),
            "regime": "warm",
            "timing_authority": "MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "benchmark_fingerprint": DEFAULT_BENCHMARK_FINGERPRINT,
            "paired": True,
            "alternating_reps": 3,
        },
        "representation": {"bpw": child.representation_bpw, "receipt_ref": "rep-child-g1"},
        "genome": {
            "runtime_sha": child.runtime_sha,
            "kernel_genome_sha": child.kernel_genome_sha,
            "receipt_ref": "genome-child-g1",
        },
        "artifact_receipt": {"sha": child.artifact_sha},
        "protected_tests": [{"name": name, "status": "PASS"} for name in REQUIRED_PROTECTED_TESTS],
        "state_transfer": {
            "checksum_verified": True,
            "checksum_sha256": labeled_sha("xfer/child-g1"),
        },
        "rollback_artifact": {"valid": True, "instance_id": parent.instance_id},
        "parent_contract": {
            "capability": dict(parent.capability),
            "benchmark_fingerprint": parent.benchmark_fingerprint,
            "silent_fallback_ids": list(parent.silent_fallback_ids),
            "complete_token_ns": parent.complete_token_ns,
            "representation_bpw": parent.representation_bpw,
            "tps": parent.tps,
        },
        "child_tps": child.tps,
        "new_silent_fallbacks": [],
    }


def _fails(verdict: dict[str, Any]) -> list[str]:
    return [row["name"] for row in verdict["checks"] if row["status"] == "FAIL"]


def _detail(verdict: dict[str, Any], name: str) -> str:
    for row in verdict["checks"]:
        if row["name"] == name:
            return row["detail"]
    raise AssertionError(f"clause {name} missing")


def test_paperwork_only_child_is_reject() -> None:
    """Attack 1: capability floats + three PASS strings, model never ran."""
    state, parent, child, ev, inv = armed_lineage()
    ev = dict(ev)
    ev.pop("greedy_token_ids", None)
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_GREEDY_TOKEN_IDS) == "FAIL"
    assert "required evidence" in _detail(verdict, CLAUSE_GREEDY_TOKEN_IDS)


def test_lowered_contract_and_forged_child_tps_is_reject() -> None:
    """Attack 2: engineering 0.0 hidden by contract=0 and child_tps = parent.tps."""
    state, parent, child, _ev, inv = armed_lineage()
    weak = make_child(parent, capability={**child.capability, "engineering": 0.0})
    ev = passing_evidence(parent, weak)
    ev["parent_contract"] = dict(ev["parent_contract"])
    ev["parent_contract"]["capability"] = {axis: 0.0 for axis in parent.capability}
    ev["child_tps"] = parent.tps
    verdict = evaluate_promotion(parent=parent, child=weak, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_CAPABILITY) == "FAIL"
    assert clause_status(verdict, CLAUSE_TPS_UP_CAP_DOWN) == "FAIL"
    assert "lowered" in _detail(verdict, CLAUSE_CAPABILITY)
    tps_detail = _detail(verdict, CLAUSE_TPS_UP_CAP_DOWN)
    assert "improves TPS" in tps_detail
    # Derived from walls, not the forged child_tps == parent.tps.
    assert f"{parent.tps:.4f} > {parent.tps:.4f}" not in tps_detail


def test_missing_timing_authority_with_invented_reps_is_reject() -> None:
    """Attack 3: absent timing_authority plus 1 ms reps must FAIL, never skip."""
    state, parent, child, ev, inv = armed_lineage()
    ev = dict(ev)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"].pop("timing_authority", None)
    ev["measurement"].pop("parent_timing_authority", None)
    ev["measurement"].pop("child_timing_authority", None)
    ev["measurement"]["complete_token_ns_reps"] = [1_000_000, 1_000_000, 1_000_000]
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_COMPLETE_TOKEN_MATERIAL) == "FAIL"
    assert "missing is FAIL" in _detail(verdict, CLAUSE_COMPLETE_TOKEN_MATERIAL)


def test_mixed_stopwatches_cross_authority_is_reject() -> None:
    """Attack 4: 38.217 ms complete-wall vs 35.228 ms encode+submit+wait."""
    state, parent, child, ev, inv = armed_lineage()
    ev = dict(ev)
    ev["measurement"] = dict(ev["measurement"])
    ev["measurement"]["parent_complete_token_ns_reps"] = [38_217_000, 38_217_000, 38_217_000]
    ev["measurement"]["complete_token_ns_reps"] = [35_228_000, 35_228_000, 35_228_000]
    ev["measurement"]["parent_timing_authority"] = "complete-wall stopwatch"
    ev["measurement"]["child_timing_authority"] = "encode+submit+wait stopwatch"
    ev["measurement"]["timing_authority"] = "encode+submit+wait stopwatch"
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_COMPLETE_TOKEN_MATERIAL) == "FAIL"
    detail = _detail(verdict, CLAUSE_COMPLETE_TOKEN_MATERIAL)
    assert "cross-authority" in detail
    assert "complete-wall" in detail
    assert "encode+submit+wait" in detail


def test_generation_zero_different_model_is_reject() -> None:
    """Attack 5: generation-0 child of a different model."""
    state, parent, _child, _ev, inv = armed_lineage()
    alien = make_child(parent, generation=0)
    alien.identity["model"] = "totally-different-weights"
    ev = passing_evidence(parent, alien)
    verdict = evaluate_promotion(parent=parent, child=alien, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_GENERATION) == "FAIL"
    assert clause_status(verdict, CLAUSE_MODEL_IDENTITY) == "FAIL"
    assert "totally-different-weights" in _detail(verdict, CLAUSE_MODEL_IDENTITY)
    assert "strictly increase" in _detail(verdict, CLAUSE_GENERATION)


def test_forged_checksum_verified_true_is_reject() -> None:
    """Attack 6: checksum_sha256 = 'ab'*32 with verified: true."""
    state, parent, child, ev, inv = armed_lineage()
    ev = dict(ev)
    ev["state_transfer"] = {"checksum_verified": True, "checksum_sha256": "ab" * 32}
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_STATE_TRANSFER) == "FAIL"
    assert "untrusted" in _detail(verdict, CLAUSE_STATE_TRANSFER) or "mismatch" in _detail(
        verdict, CLAUSE_STATE_TRANSFER
    )


@pytest.mark.parametrize(
    "label,corrupt,clause,needle",
    [
        (
            "generation_not_increased",
            lambda p, c, ev: (make_child(p, generation=p.generation), ev),
            CLAUSE_GENERATION,
            "strictly increase",
        ),
        (
            "model_swapped",
            lambda p, c, ev: _swap_model(p, c, ev),
            CLAUSE_MODEL_IDENTITY,
            "identity.model mismatch",
        ),
        (
            "greedy_only_two_prompts",
            lambda p, c, ev: (c, _two_prompts(ev)),
            CLAUSE_GREEDY_TOKEN_IDS,
            "requires >= 3",
        ),
        (
            "greedy_ids_disagree",
            lambda p, c, ev: (c, _disagree(ev)),
            CLAUSE_GREEDY_TOKEN_IDS,
            "disagree",
        ),
        (
            "artifact_preimage_missing",
            lambda p, c, ev: (c, _drop_preimage(ev)),
            CLAUSE_ARTIFACT_IDENTITY,
            "computed from a preimage",
        ),
        (
            "artifact_preimage_wrong",
            lambda p, c, ev: (c, _wrong_preimage(ev)),
            CLAUSE_ARTIFACT_IDENTITY,
            "mismatch",
        ),
        (
            "checksum_payload_tampered",
            lambda p, c, ev: (c, _tamper_payload(ev)),
            CLAUSE_STATE_TRANSFER,
            "mismatch",
        ),
        (
            "contract_floor_lowered_child_ok",
            lambda p, c, ev: (c, _lower_contract(ev, p)),
            CLAUSE_CAPABILITY,
            "lowered",
        ),
    ],
)
def test_each_new_clause_rejects_when_input_corrupted(label, corrupt, clause, needle) -> None:
    state, parent, child, ev, inv = armed_lineage()
    child, ev = corrupt(parent, child, ev)
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT", (label, _fails(verdict), verdict["checks"])
    assert clause_status(verdict, clause) == "FAIL", (label, verdict["checks"])
    assert needle in _detail(verdict, clause), (label, _detail(verdict, clause))
    assert verdict["fabricated_accept"] is False


def test_child_tps_is_derived_from_wall_reps_not_evidence() -> None:
    state, parent, child, _ev, inv = armed_lineage()
    weak = make_child(parent, capability={**child.capability, "engineering": 0.0})
    ev = passing_evidence(parent, weak)
    ev["child_tps"] = parent.tps
    ev["parent_contract"] = dict(ev["parent_contract"])
    ev["parent_contract"]["tps"] = parent.tps
    verdict = evaluate_promotion(parent=parent, child=weak, evidence=ev, invoker=inv, lineage=state)
    assert clause_status(verdict, CLAUSE_TPS_UP_CAP_DOWN) == "FAIL"
    child_wall = sum((30_010_000, 29_980_000, 30_010_000)) / 3
    derived = NS_PER_SECOND / child_wall
    assert f"{derived:.4f}" in _detail(verdict, CLAUSE_TPS_UP_CAP_DOWN)


def test_forged_checksum_with_payload_still_recomputed() -> None:
    state, parent, child, ev, inv = armed_lineage()
    ev = dict(ev)
    payload = parent_research_payload(next_bottleneck="real-science-next-bottleneck")
    ev["state_transfer"] = {
        "payload": payload,
        "checksum_sha256": "ab" * 32,
        "checksum_verified": True,
    }
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_STATE_TRANSFER) == "FAIL"
    assert payload_checksum(payload) in _detail(verdict, CLAUSE_STATE_TRANSFER)
    assert "ab" * 32 in _detail(verdict, CLAUSE_STATE_TRANSFER)


def test_missing_identity_model_is_fail() -> None:
    state, parent, child, ev, inv = armed_lineage()
    child.identity.pop("model", None)
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    assert clause_status(verdict, CLAUSE_MODEL_IDENTITY) == "FAIL"
    assert "missing is FAIL" in _detail(verdict, CLAUSE_MODEL_IDENTITY)


def test_new_clauses_are_in_all_clauses() -> None:
    assert CLAUSE_GENERATION in ALL_CLAUSES
    assert CLAUSE_MODEL_IDENTITY in ALL_CLAUSES
    assert CLAUSE_GREEDY_TOKEN_IDS in ALL_CLAUSES


def _swap_model(parent, child, ev):
    other = make_child(parent)
    other.identity["model"] = "totally-different-weights"
    return other, passing_evidence(parent, other)


def _two_prompts(ev):
    ev = dict(ev)
    ev["greedy_token_ids"] = ev["greedy_token_ids"][:2]
    return ev


def _disagree(ev):
    ev = dict(ev)
    rows = [dict(row) for row in ev["greedy_token_ids"]]
    rows[0]["child_ids"] = list(rows[0]["parent_ids"]) + [999]
    ev["greedy_token_ids"] = rows
    return ev


def _drop_preimage(ev):
    ev = dict(ev)
    ev["artifact_receipt"] = {"sha": ev["artifact_receipt"]["sha"]}
    return ev


def _wrong_preimage(ev):
    ev = dict(ev)
    ev["artifact_receipt"] = dict(ev["artifact_receipt"])
    ev["artifact_receipt"]["preimage"] = "not-the-bytes-that-hash-to-the-claimed-sha"
    return ev


def _tamper_payload(ev):
    ev = dict(ev)
    st = dict(ev["state_transfer"])
    payload = dict(st["payload"])
    payload["NEXT_BOTTLENECK"] = "silently rewritten"
    st["payload"] = payload
    ev["state_transfer"] = st
    return ev


def _lower_contract(ev, parent):
    ev = dict(ev)
    ev["parent_contract"] = dict(ev["parent_contract"])
    ev["parent_contract"]["capability"] = {axis: 0.0 for axis in parent.capability}
    return ev


def test_original_six_accepts_are_now_reject() -> None:
    """The exact forged inputs that the paperwork gate accepted."""
    state, parent, child, _ev, inv = armed_lineage()

    v1 = evaluate_promotion(
        parent=parent,
        child=child,
        evidence=_legacy_paperwork_evidence(parent, child),
        invoker=inv,
        lineage=state,
    )
    assert v1["verdict"] == "REJECT", v1
    assert clause_status(v1, CLAUSE_GREEDY_TOKEN_IDS) == "FAIL"

    weak = make_child(parent, capability={**child.capability, "engineering": 0.0})
    ev2 = _legacy_paperwork_evidence(parent, weak)
    ev2["parent_contract"] = dict(ev2["parent_contract"])
    ev2["parent_contract"]["capability"] = {axis: 0.0 for axis in parent.capability}
    ev2["child_tps"] = parent.tps
    v2 = evaluate_promotion(parent=parent, child=weak, evidence=ev2, invoker=inv, lineage=state)
    assert v2["verdict"] == "REJECT"
    assert clause_status(v2, CLAUSE_CAPABILITY) == "FAIL"
    assert clause_status(v2, CLAUSE_TPS_UP_CAP_DOWN) == "FAIL"

    ev3 = _legacy_paperwork_evidence(parent, child)
    ev3["measurement"] = dict(ev3["measurement"])
    ev3["measurement"].pop("timing_authority", None)
    ev3["measurement"]["complete_token_ns_reps"] = [1_000_000, 1_000_000, 1_000_000]
    v3 = evaluate_promotion(parent=parent, child=child, evidence=ev3, invoker=inv, lineage=state)
    assert v3["verdict"] == "REJECT"
    assert clause_status(v3, CLAUSE_COMPLETE_TOKEN_MATERIAL) == "FAIL"
    assert "missing is FAIL" in _detail(v3, CLAUSE_COMPLETE_TOKEN_MATERIAL)

    ev4 = _legacy_paperwork_evidence(parent, child)
    ev4["measurement"] = dict(ev4["measurement"])
    ev4["measurement"]["parent_complete_token_ns_reps"] = [38_217_000, 38_217_000, 38_217_000]
    ev4["measurement"]["complete_token_ns_reps"] = [35_228_000, 35_228_000, 35_228_000]
    ev4["measurement"]["timing_authority"] = "encode+submit+wait stopwatch"
    ev4["measurement"]["parent_timing_authority"] = "complete-wall stopwatch"
    v4 = evaluate_promotion(parent=parent, child=child, evidence=ev4, invoker=inv, lineage=state)
    assert v4["verdict"] == "REJECT"
    assert clause_status(v4, CLAUSE_COMPLETE_TOKEN_MATERIAL) == "FAIL"
    assert "cross-authority" in _detail(v4, CLAUSE_COMPLETE_TOKEN_MATERIAL)

    alien = make_child(parent, generation=0)
    alien.identity["model"] = "totally-different-weights"
    v5 = evaluate_promotion(
        parent=parent,
        child=alien,
        evidence=_legacy_paperwork_evidence(parent, alien),
        invoker=inv,
        lineage=state,
    )
    assert v5["verdict"] == "REJECT"
    assert clause_status(v5, CLAUSE_GENERATION) == "FAIL"
    assert clause_status(v5, CLAUSE_MODEL_IDENTITY) == "FAIL"

    ev6 = _legacy_paperwork_evidence(parent, child)
    ev6["state_transfer"] = {"checksum_verified": True, "checksum_sha256": "ab" * 32}
    v6 = evaluate_promotion(parent=parent, child=child, evidence=ev6, invoker=inv, lineage=state)
    assert v6["verdict"] == "REJECT"
    assert clause_status(v6, CLAUSE_STATE_TRANSFER) == "FAIL"
