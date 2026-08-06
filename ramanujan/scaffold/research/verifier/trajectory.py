"""Emit verified trajectories in the scaffold paired-trace / TraceRecord shape.

Aligned with ``odyssey.TraceRecord`` keep-fields and the corpora provenance
stamp used by D1–D7.  A record is only marked ``admitted=true`` when a real
verifier returned ACCEPTED.  Unverified attempts are never admitted as training
positives.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ramanujan.verifier.base import VerificationResult, Verdict, content_hash

VERIFIED_TRAJECTORY_SCHEMA = "hawking.ramanujan.verified_trajectory.v1"
PAIRED_TRACE_SCHEMA = "hawking.ramanujan.paired_trace.v1"
# Matches RamanujanProtoProgram.trace_schema keep set.
TRACE_KEEP_FIELDS = (
    "statement_hash",
    "plan",
    "subgoals",
    "formal_states",
    "actions",
    "tool_calls",
    "verifier_outcomes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_verified_trajectory(
    *,
    problem_id: str,
    statement: str,
    attempts: Sequence[Mapping[str, Any]],
    final_answer: str | None,
    final_result: VerificationResult | None,
    plan: Sequence[str] = (),
    subgoals: Sequence[str] = (),
    formal_states: Sequence[str] = (),
    actions: Sequence[str] = (),
    tool_calls: Sequence[Mapping[str, Any]] = (),
    teacher_critique: Mapping[str, Any] | None = None,
    membership: str = "fixture",
    method_label: str = "expert_iteration_exact",
) -> dict[str, Any]:
    """Build a verified-trajectory training record.

    ``admitted`` is True only when ``final_result.verdict is ACCEPTED``.
    """
    statement_hash = content_hash({"problem_id": problem_id, "statement": statement})
    accepted = bool(final_result is not None and final_result.verdict is Verdict.ACCEPTED)
    outcomes: list[dict[str, Any]] = []
    for attempt in attempts:
        vr = attempt.get("verification")
        if isinstance(vr, VerificationResult):
            outcomes.append(vr.as_verifier_outcome())
        elif isinstance(vr, Mapping):
            outcomes.append(dict(vr))
    if final_result is not None and (
        not outcomes or outcomes[-1].get("detail") != final_result.detail
    ):
        outcomes.append(final_result.as_verifier_outcome())

    disposition = (
        "EXACTLY_REPRODUCED"
        if accepted and final_result and final_result.outcome_kind == "independent_exact_check"
        else "LEAN_VERIFIED"
        if accepted and final_result and final_result.outcome_kind == "lean_replay"
        else "REJECTED"
        if final_result and final_result.verdict is Verdict.REJECTED
        else "PLAUSIBLE_UNVERIFIED"
    )

    body = {
        "schema": VERIFIED_TRAJECTORY_SCHEMA,
        "id": f"vt:{problem_id}:{statement_hash[:12]}",
        "problem_id": problem_id,
        "statement": statement,
        "statement_hash": statement_hash,
        "problem_hash": statement_hash,
        "membership": membership,
        "method_label": method_label,
        "plan": list(plan) or ["attempt", "verify", "repair_if_needed", "accept_only_if_verified"],
        "subgoals": list(subgoals) or [statement],
        "formal_states": list(formal_states),
        "actions": list(actions)
        or [f"claim:{a.get('answer')}" for a in attempts if isinstance(a, Mapping)],
        "tool_calls": [dict(t) for t in tool_calls],
        "verifier_outcomes": outcomes,
        "disposition": disposition,
        "attempts": [
            {
                "round": int(a.get("round", i)),
                "answer": a.get("answer"),
                "verdict": (
                    a["verification"].verdict.value
                    if isinstance(a.get("verification"), VerificationResult)
                    else (a.get("verification") or {}).get("verdict")
                ),
                "backend_id": (
                    a["verification"].backend_id
                    if isinstance(a.get("verification"), VerificationResult)
                    else (a.get("verification") or {}).get("backend_id")
                ),
                "detail": (
                    a["verification"].detail
                    if isinstance(a.get("verification"), VerificationResult)
                    else (a.get("verification") or {}).get("detail")
                ),
            }
            for i, a in enumerate(attempts)
        ],
        "final_answer": final_answer if accepted else None,
        "teacher_critique": dict(teacher_critique) if teacher_critique else None,
        "admitted": accepted,
        "source_id": "expert_iteration",
        "split": "train" if accepted else "rejected",
        "provenance": {
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "at": _utc_now(),
            "extraction_method": "verifier_expert_iteration_loop",
            "teacher_from_math_preserve": False,
            "verified_only_admission": True,
        },
    }
    # Drop non-keep free-form model dumps; keep structured fields.
    body["text"] = (
        f"statement: {statement}\n"
        f"final: {final_answer if accepted else '(not admitted)'}\n"
        f"disposition: {disposition}\n"
        f"rounds: {len(attempts)}"
    )
    body["content_hash"] = content_hash(
        {k: body[k] for k in (
            "problem_id", "statement", "statement_hash", "disposition",
            "attempts", "final_answer", "verifier_outcomes", "admitted",
        )}
    )
    return body


def emit_paired_trace_record(
    trajectory: Mapping[str, Any],
    *,
    wrong_attempt: str | None = None,
    verified_answer: str | None = None,
) -> dict[str, Any]:
    """Scaffold paired-trace: (failed attempt | problem) → verified target.

    Mirrors the D4 repair-pair idea and the functional paired-trace discipline:
    input surface is the broken attempt, target surface is the verified repair.
    Only emitted with ``admitted=true`` when the trajectory itself was admitted.
    """
    admitted = bool(trajectory.get("admitted"))
    attempts = list(trajectory.get("attempts") or [])
    if wrong_attempt is None:
        for row in attempts:
            if row.get("verdict") == Verdict.REJECTED.value:
                wrong_attempt = str(row.get("answer") or "")
                break
    if verified_answer is None and admitted:
        verified_answer = trajectory.get("final_answer")

    input_side = {
        "role": "wrong_attempt" if wrong_attempt else "problem",
        "problem_id": trajectory.get("problem_id"),
        "statement": trajectory.get("statement"),
        "answer": wrong_attempt,
        "statement_hash": trajectory.get("statement_hash"),
    }
    target_side = {
        "role": "verified_repair" if admitted else "unverified",
        "problem_id": trajectory.get("problem_id"),
        "statement": trajectory.get("statement"),
        "answer": verified_answer if admitted else None,
        "disposition": trajectory.get("disposition"),
        "verifier_outcomes": list(trajectory.get("verifier_outcomes") or []),
        "statement_hash": trajectory.get("statement_hash"),
    }
    record = {
        "schema": PAIRED_TRACE_SCHEMA,
        "id": f"pt:{trajectory.get('id', content_hash(dict(trajectory))[:16])}",
        "source_id": "expert_iteration",
        "split": "train" if admitted else "rejected",
        "admitted": admitted,
        "input_surface": "wrong_attempt",
        "target_surface": "verified_repair",
        "input": input_side,
        "target": target_side,
        # TraceRecord-compatible keep fields for the verified side.
        "statement_hash": trajectory.get("statement_hash"),
        "plan": list(trajectory.get("plan") or []),
        "subgoals": list(trajectory.get("subgoals") or []),
        "formal_states": list(trajectory.get("formal_states") or []),
        "actions": list(trajectory.get("actions") or []),
        "tool_calls": list(trajectory.get("tool_calls") or []),
        "verifier_outcomes": list(trajectory.get("verifier_outcomes") or []),
        "disposition": trajectory.get("disposition"),
        "text": (
            f"input: {wrong_attempt!r}\n"
            f"target: {verified_answer!r}\n"
            f"admitted: {admitted}\n"
            f"statement: {trajectory.get('statement')}"
        ),
        "provenance": dict(trajectory.get("provenance") or {})
        | {
            "paired_from": trajectory.get("id"),
            "at": _utc_now(),
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        },
    }
    record["content_hash"] = content_hash(
        {
            "input": input_side,
            "target": target_side,
            "admitted": admitted,
            "statement_hash": record["statement_hash"],
        }
    )
    return record
