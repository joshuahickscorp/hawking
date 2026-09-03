"""Expert-iteration harness: attempt → verify → (teacher critique) → repair → accept.

Only verified answers become training data.  The GLM teacher-critique step is
gated behind ``REQUIRES_GLM_ACCESS`` and is never faked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ramanujan.verifier.base import (
    VerificationRequest,
    VerificationResult,
    Verdict,
)
from ramanujan.verifier.registry import VerifierRegistry, default_registry
from ramanujan.verifier.trajectory import emit_paired_trace_record, emit_verified_trajectory

# Explicit gate token — callers and receipts must surface this string.
REQUIRES_GLM_ACCESS = "REQUIRES_GLM_ACCESS"
GLM_ACCESS_ENV = "HAWKING_GLM_TEACHER_ACCESS"


@dataclass(frozen=True)
class StudentAttempt:
    answer: str
    plan: tuple[str, ...] = ()
    subgoals: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class TeacherCritiqueGate:
    """Result of requesting a teacher critique.

    When GLM is unavailable the gate is closed; the harness may still attempt a
    local repair callback, but must record that no teacher critique was obtained.
    """

    available: bool
    status: str  # "OK" | "REQUIRES_GLM_ACCESS" | "REFUSED"
    critique: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "critique": self.critique,
            "detail": self.detail,
            "gate": REQUIRES_GLM_ACCESS if not self.available else "OPEN",
        }


def glm_access_granted() -> bool:
    """True only when the operator explicitly enables GLM teacher access."""
    return os.environ.get(GLM_ACCESS_ENV, "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def request_teacher_critique(
    *,
    statement: str,
    failed_answer: str,
    verification: VerificationResult,
    glm_callback: Callable[[Mapping[str, Any]], str] | None = None,
) -> TeacherCritiqueGate:
    """Teacher critique step.  Fails closed without GLM access; never invents critique."""
    if not glm_access_granted():
        return TeacherCritiqueGate(
            available=False,
            status=REQUIRES_GLM_ACCESS,
            critique=None,
            detail=(
                f"teacher critique gated: set {GLM_ACCESS_ENV}=1 and provide a "
                "glm_callback to obtain real GLM critique. No critique was invented."
            ),
        )
    if glm_callback is None:
        return TeacherCritiqueGate(
            available=False,
            status=REQUIRES_GLM_ACCESS,
            critique=None,
            detail="GLM access flag set but no glm_callback provided; refuse fake critique",
        )
    try:
        text = glm_callback(
            {
                "statement": statement,
                "failed_answer": failed_answer,
                "verification": verification.as_dict(),
            }
        )
    except Exception as exc:  # real GLM path failed — still not a fake critique
        return TeacherCritiqueGate(
            available=False,
            status="REFUSED",
            critique=None,
            detail=f"glm_callback raised: {exc}",
        )
    if not isinstance(text, str) or not text.strip():
        return TeacherCritiqueGate(
            available=False,
            status="REFUSED",
            critique=None,
            detail="glm_callback returned empty critique",
        )
    return TeacherCritiqueGate(
        available=True,
        status="OK",
        critique=text.strip(),
        detail="critique obtained from glm_callback under access gate",
    )


@dataclass
class ExpertIterationResult:
    accepted: bool
    trajectory: dict[str, Any]
    paired_trace: dict[str, Any]
    rounds: list[dict[str, Any]] = field(default_factory=list)
    teacher_critique: dict[str, Any] | None = None
    stop_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "stop_reason": self.stop_reason,
            "rounds": list(self.rounds),
            "teacher_critique": self.teacher_critique,
            "trajectory": self.trajectory,
            "paired_trace": self.paired_trace,
            # Hard invariant echo for fixtures/tests.
            "never_accepted_unverified": True,
            "admitted_implies_verified": (
                (not self.trajectory.get("admitted"))
                or self.accepted
            ),
        }


StudentFn = Callable[[Mapping[str, Any]], StudentAttempt]
RepairFn = Callable[[Mapping[str, Any]], StudentAttempt]


class ExpertIterationHarness:
    """Closed loop for functional-transfer verified trajectories.

    Parameters
    ----------
    student
        Callable producing an initial attempt from the problem mapping.
    repair
        Callable producing a repair attempt given problem + last failure +
        optional critique.  Fixture tests inject a deterministic repairer.
    registry
        Verifier registry (defaults to exact + sympy + lean).
    max_repairs
        Maximum repair rounds after the initial attempt.
    """

    def __init__(
        self,
        *,
        student: StudentFn,
        repair: RepairFn | None = None,
        registry: VerifierRegistry | None = None,
        max_repairs: int = 2,
        glm_callback: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> None:
        self.student = student
        self.repair = repair
        self.registry = registry or default_registry()
        self.max_repairs = max(0, int(max_repairs))
        self.glm_callback = glm_callback

    def run(self, problem: Mapping[str, Any]) -> ExpertIterationResult:
        problem_id = str(problem.get("id") or problem.get("problem_id") or "unknown")
        statement = str(problem.get("statement") or "")
        kind = str(problem.get("kind") or "exact_numeric")
        payload = dict(problem.get("payload") or {})
        # Allow flat fixture fields to populate payload.
        for key in ("expression", "expected_expr", "proof_lean", "capsule_path"):
            if key in problem and key not in payload:
                payload[key] = problem[key]

        rounds: list[dict[str, Any]] = []
        attempts_for_trace: list[dict[str, Any]] = []
        teacher_gate: TeacherCritiqueGate | None = None
        all_actions: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        attempt = self.student(dict(problem))
        current = attempt
        final_result: VerificationResult | None = None
        final_answer: str | None = None
        accepted = False
        stop_reason = "exhausted"

        for round_idx in range(self.max_repairs + 1):
            request = VerificationRequest(
                problem_id=problem_id,
                statement=statement,
                kind=kind,
                claimed_answer=current.answer,
                payload=payload,
                checker_id="host_verifier",
            )
            result = self.registry.verify(request)
            tool_calls.append(
                {
                    "tool": "exact-checker" if result.backend_id != "lean" else "lean",
                    "backend_id": result.backend_id,
                    "args": {"problem_id": problem_id, "round": round_idx},
                }
            )
            row = {
                "round": round_idx,
                "answer": current.answer,
                "verification": result.as_dict(),
                "phase": "initial" if round_idx == 0 else "repair",
            }
            rounds.append(row)
            attempts_for_trace.append(
                {"round": round_idx, "answer": current.answer, "verification": result}
            )
            all_actions.extend(current.actions or (f"claim:{current.answer}",))
            final_result = result
            final_answer = current.answer

            if result.verdict is Verdict.ACCEPTED:
                accepted = True
                stop_reason = "verified"
                break

            # Failure or uncertainty → teacher critique (gated), then repair.
            if round_idx >= self.max_repairs:
                stop_reason = f"max_repairs_exhausted_last={result.verdict.value}"
                break

            teacher_gate = request_teacher_critique(
                statement=statement,
                failed_answer=current.answer,
                verification=result,
                glm_callback=self.glm_callback,
            )
            if self.repair is None:
                stop_reason = f"no_repair_fn_after_{result.verdict.value}"
                break

            repair_ctx = {
                "problem": dict(problem),
                "failed_answer": current.answer,
                "verification": result.as_dict(),
                "teacher_critique": teacher_gate.as_dict(),
                "round": round_idx,
            }
            current = self.repair(repair_ctx)

        trajectory = emit_verified_trajectory(
            problem_id=problem_id,
            statement=statement,
            attempts=attempts_for_trace,
            final_answer=final_answer if accepted else None,
            final_result=final_result if accepted else final_result,
            plan=tuple(attempt.plan)
            or ("attempt", "verify", "repair_if_needed", "accept_only_if_verified"),
            subgoals=tuple(attempt.subgoals) or (statement,),
            actions=tuple(all_actions),
            tool_calls=tool_calls,
            teacher_critique=teacher_gate.as_dict() if teacher_gate else None,
            method_label="expert_iteration_exact",
        )
        # Belt-and-suspenders: never admit without ACCEPTED.
        if accepted and final_result is not None and final_result.verdict is Verdict.ACCEPTED:
            trajectory["admitted"] = True
        else:
            trajectory["admitted"] = False
            trajectory["final_answer"] = None
            if trajectory.get("disposition") == "EXACTLY_REPRODUCED":
                trajectory["disposition"] = "REJECTED"
            accepted = False

        paired = emit_paired_trace_record(trajectory)
        return ExpertIterationResult(
            accepted=accepted,
            trajectory=trajectory,
            paired_trace=paired,
            rounds=rounds,
            teacher_critique=teacher_gate.as_dict() if teacher_gate else None,
            stop_reason=stop_reason,
        )


def fixture_wrong_then_right_student(
    wrong: str,
    right: str,
) -> tuple[StudentFn, RepairFn]:
    """Deterministic student that is wrong first, then repairs to ``right``."""

    def student(_problem: Mapping[str, Any]) -> StudentAttempt:
        return StudentAttempt(answer=wrong, plan=("guess",), actions=(f"claim:{wrong}",))

    def repair(ctx: Mapping[str, Any]) -> StudentAttempt:
        return StudentAttempt(
            answer=right,
            plan=("repair_from_verifier",),
            actions=(f"repair:{right}",),
            notes=str((ctx.get("teacher_critique") or {}).get("status") or ""),
        )

    return student, repair
