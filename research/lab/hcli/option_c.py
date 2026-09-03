"""Option-C sandbox (Bible §24) — standing subsystem, not a one-off habit.

Structural identity with tonight's Claude + Grok delegate/audit pattern:

  ┌──────────────────────┬────────────────────────────┬─────────────────────┐
  │ Option-C role        │ Tonight's reference impl   │ Authority           │
  ├──────────────────────┼────────────────────────────┼─────────────────────┤
  │ 30B executor         │ Grok Build (`delegate`)    │ isolated worktree,  │
  │                      │                            │ edit/compile/test,  │
  │                      │                            │ emit candidate only │
  │ 80B reviewer         │ Grok `audit` (read-only,   │ independent inspect,│
  │                      │ independent of executor)   │ challenge, no edit  │
  │ protected controller │ Claude (architect + final  │ parity / held-out / │
  │                      │ authority; never transfers)│ CLEAN / sign promote│
  │                      │                            │ or rollback         │
  └──────────────────────┴────────────────────────────┴─────────────────────┘

Option-C is *logical*, not necessarily simultaneous: executor → reviewer →
controller may run as sequential phases (see residency modes Mode B/C).

Mandatory independent review before controller decision for the categories in
``MANDATORY_REVIEW_CATEGORIES``. Sandbox models never sign their own results,
never merge themselves, never modify protected oracles or promotion thresholds.

Reference implementation (habits formalised here):
  - ``~/.claude/skills/grok-orchestration/SKILL.md``
  - ``delegate.md`` — contract → isolated worktree → report → review checklist
  - ``audit.md`` — independent criteria, no executor conclusions, synthesis by controller
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
import hashlib
import json
import uuid

from lab.receipts import seal

SCHEMA = "hawking.hcli.option_c.v1"
CANDIDATE_SCHEMA = "hawking.hcli.option_c.candidate.v1"
REVIEW_SCHEMA = "hawking.hcli.option_c.review.v1"
DECISION_SCHEMA = "hawking.hcli.option_c.decision.v1"

# Bootstrap models (logical roles; no weight loading in this scaffold).
EXECUTOR_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
REVIEWER_MODEL_ID = "Qwen/Qwen3-Coder-Next"  # 80B-class reviewer


class Role(str, Enum):
    EXECUTOR = "executor_30b"
    REVIEWER = "reviewer_80b"
    PROTECTED_CONTROLLER = "protected_controller"


class CandidatePhase(str, Enum):
    """Logical Option-C lifecycle for one candidate experiment."""

    IDLE = "idle"
    EXECUTING = "executing"
    CANDIDATE_EMITTED = "candidate_emitted"
    REVIEWING = "reviewing"
    REVIEW_EMITTED = "review_emitted"
    CONTROLLER_EVAL = "controller_eval"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Bible §24 mandatory review surface.
MANDATORY_REVIEW_CATEGORIES: frozenset[str] = frozenset(
    {
        "kernel_promotion",
        "quantization_change",
        "routing_change",
        "benchmark_change",
        "runtime_scheduling",
        "storage_deletion",
        "artifact_promotion",
        "effect_authority",
    }
)

# Result classification (Bible §22) applied at controller decision.
RESULT_CLASSES: frozenset[str] = frozenset(
    {
        "PROMOTED_MECHANISM",
        "REJECTED_MECHANISM",
        "TOOL_DEFECT",
        "PLANNING_DEFECT",
        "VERIFIER_DEFECT",
        "ENVIRONMENT_DEFECT",
        "INSUFFICIENT_EVIDENCE",
    }
)

# Executor may / may not (Bible §21) — encoded as policy, not prose alone.
EXECUTOR_ALLOWED: frozenset[str] = frozenset(
    {
        "read_source",
        "inspect_public_profiles",
        "edit_owned_worktree",
        "compile",
        "run_allowed_tests",
        "request_protected_benchmark",
        "request_approved_downloads",
        "emit_candidate_report",
    }
)

EXECUTOR_FORBIDDEN: frozenset[str] = frozenset(
    {
        "modify_protected_oracle",
        "modify_held_out_prompts",
        "modify_promotion_thresholds",
        "merge_self",
        "sign_own_results",
        "delete_stable_artifacts",
        "read_or_print_credentials",
    }
)

REVIEWER_ALLOWED: frozenset[str] = frozenset(
    {
        "independently_inspect",
        "challenge_parity",
        "challenge_benchmark",
        "challenge_architecture",
        "request_distinguishing_tests",
        "emit_review_report",
    }
)

REVIEWER_FORBIDDEN: frozenset[str] = frozenset(
    {
        "edit_worktree",
        "merge",
        "sign_promotion",
        "delete_artifacts",
        "see_controller_conclusions_before_review",  # independent by construction
    }
)


class OptionCRefusal(ValueError):
    """Unlawful Option-C transition or authority violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass
class CandidateReport:
    """30B executor output. A claim, not a receipt — controller must verify."""

    candidate_id: str
    worktree_ref: str
    category: str
    summary: str
    changes: Mapping[str, Any]
    tests_run: Sequence[str]
    evidence_paths: Sequence[str]
    executor_model_id: str = EXECUTOR_MODEL_ID
    emitted_at: str = field(default_factory=_utc_now)
    # Structural mirror of grok-report.md + diff.patch
    report_sha256: str = ""
    diff_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.report_sha256:
            self.report_sha256 = _digest(
                {
                    "candidate_id": self.candidate_id,
                    "summary": self.summary,
                    "changes": dict(self.changes),
                    "tests_run": list(self.tests_run),
                }
            )
        if not self.diff_sha256:
            self.diff_sha256 = _digest(dict(self.changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "worktree_ref": self.worktree_ref,
            "category": self.category,
            "summary": self.summary,
            "changes": dict(self.changes),
            "tests_run": list(self.tests_run),
            "evidence_paths": list(self.evidence_paths),
            "executor_model_id": self.executor_model_id,
            "emitted_at": self.emitted_at,
            "report_sha256": self.report_sha256,
            "diff_sha256": self.diff_sha256,
            "claim_boundary": "executor_report_is_claim_not_receipt",
        }


@dataclass
class ReviewReport:
    """80B reviewer output. Independent of executor conclusions."""

    candidate_id: str
    challenges: Sequence[Mapping[str, Any]]
    distinguishing_tests_requested: Sequence[str]
    severity_findings: Sequence[Mapping[str, Any]]
    recommendation: str  # APPROVE_WITH_GATES | REQUEST_CHANGES | REJECT | ABSTAIN
    reviewer_model_id: str = REVIEWER_MODEL_ID
    # Independence fence: reviewer must not be fed controller conclusions.
    received_controller_conclusions: bool = False
    emitted_at: str = field(default_factory=_utc_now)
    review_sha256: str = ""

    def __post_init__(self) -> None:
        if self.received_controller_conclusions:
            raise OptionCRefusal(
                "reviewer independence violated: must not receive controller conclusions"
            )
        if not self.review_sha256:
            self.review_sha256 = _digest(
                {
                    "candidate_id": self.candidate_id,
                    "challenges": list(self.challenges),
                    "findings": list(self.severity_findings),
                    "recommendation": self.recommendation,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REVIEW_SCHEMA,
            "candidate_id": self.candidate_id,
            "challenges": [dict(c) for c in self.challenges],
            "distinguishing_tests_requested": list(self.distinguishing_tests_requested),
            "severity_findings": [dict(f) for f in self.severity_findings],
            "recommendation": self.recommendation,
            "reviewer_model_id": self.reviewer_model_id,
            "received_controller_conclusions": self.received_controller_conclusions,
            "emitted_at": self.emitted_at,
            "review_sha256": self.review_sha256,
            "claim_boundary": "review_is_independent_challenge_not_authority",
        }


@dataclass
class ControllerDecision:
    candidate_id: str
    result_class: str
    action: str  # PROMOTE | ROLLBACK | REJECT | HOLD
    protected_parity: Mapping[str, Any]
    held_out_capability: Mapping[str, Any]
    clean_benchmark: Mapping[str, Any]
    signed_by: str
    review_required_and_present: bool
    reason: str
    decided_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "candidate_id": self.candidate_id,
            "result_class": self.result_class,
            "action": self.action,
            "protected_parity": dict(self.protected_parity),
            "held_out_capability": dict(self.held_out_capability),
            "clean_benchmark": dict(self.clean_benchmark),
            "signed_by": self.signed_by,
            "review_required_and_present": self.review_required_and_present,
            "reason": self.reason,
            "decided_at": self.decided_at,
        }


@dataclass
class CandidateSession:
    """One Option-C candidate lifecycle instance."""

    candidate_id: str
    category: str
    phase: CandidatePhase = CandidatePhase.IDLE
    worktree_ref: str | None = None
    candidate_report: CandidateReport | None = None
    review_report: ReviewReport | None = None
    decision: ControllerDecision | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def _note(self, event: str, **extra: Any) -> None:
        self.history.append({"event": event, "at": _utc_now(), **extra})

    def requires_mandatory_review(self) -> bool:
        return self.category in MANDATORY_REVIEW_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "category": self.category,
            "phase": self.phase.value,
            "worktree_ref": self.worktree_ref,
            "requires_mandatory_review": self.requires_mandatory_review(),
            "candidate_report": self.candidate_report.to_dict() if self.candidate_report else None,
            "review_report": self.review_report.to_dict() if self.review_report else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "history": list(self.history),
        }


class OptionCSandbox:
    """Executor + reviewer surfaces (stubs). No model loading."""

    def __init__(self) -> None:
        self.sessions: dict[str, CandidateSession] = {}

    def open_candidate(
        self,
        *,
        category: str,
        worktree_ref: str,
        candidate_id: str | None = None,
    ) -> CandidateSession:
        if not worktree_ref or not str(worktree_ref).strip():
            raise OptionCRefusal("worktree_ref required (isolated worktree only)")
        cid = candidate_id or f"oc-{uuid.uuid4().hex[:12]}"
        if cid in self.sessions:
            raise OptionCRefusal(f"candidate {cid!r} already open")
        session = CandidateSession(
            candidate_id=cid,
            category=category,
            worktree_ref=worktree_ref,
            phase=CandidatePhase.IDLE,
        )
        session._note("opened", worktree_ref=worktree_ref, category=category)
        self.sessions[cid] = session
        return session

    # --- Executor surface (Bible §24 / delegate pattern) ----------------------

    def executor_start(self, candidate_id: str, *, actor: str = Role.EXECUTOR.value) -> CandidateSession:
        s = self._get(candidate_id)
        if s.phase not in {CandidatePhase.IDLE}:
            raise OptionCRefusal(f"executor_start requires IDLE, got {s.phase.value}")
        self._assert_action_allowed(Role.EXECUTOR, "edit_owned_worktree")
        s.phase = CandidatePhase.EXECUTING
        s._note("executor_start", actor=actor)
        return s

    def executor_emit(
        self,
        candidate_id: str,
        *,
        summary: str,
        changes: Mapping[str, Any],
        tests_run: Sequence[str],
        evidence_paths: Sequence[str] | None = None,
        actor: str = Role.EXECUTOR.value,
    ) -> CandidateReport:
        s = self._get(candidate_id)
        if s.phase != CandidatePhase.EXECUTING:
            raise OptionCRefusal(f"executor_emit requires EXECUTING, got {s.phase.value}")
        for forbidden in EXECUTOR_FORBIDDEN:
            if forbidden in changes:
                raise OptionCRefusal(f"executor may not perform {forbidden!r}")
        report = CandidateReport(
            candidate_id=candidate_id,
            worktree_ref=s.worktree_ref or "",
            category=s.category,
            summary=summary,
            changes=dict(changes),
            tests_run=list(tests_run),
            evidence_paths=list(evidence_paths or ()),
        )
        s.candidate_report = report
        s.phase = CandidatePhase.CANDIDATE_EMITTED
        s._note("candidate_emitted", report_sha256=report.report_sha256, actor=actor)
        return report

    # --- Reviewer surface (Bible §24 / audit pattern) ------------------------

    def reviewer_start(
        self,
        candidate_id: str,
        *,
        actor: str = Role.REVIEWER.value,
        controller_conclusions: Mapping[str, Any] | None = None,
    ) -> CandidateSession:
        s = self._get(candidate_id)
        if s.phase != CandidatePhase.CANDIDATE_EMITTED:
            raise OptionCRefusal(
                f"reviewer_start requires CANDIDATE_EMITTED, got {s.phase.value}"
            )
        if controller_conclusions:
            raise OptionCRefusal(
                "reviewer must not receive controller conclusions (independence fence)"
            )
        if s.requires_mandatory_review() is False:
            # Still allowed to review; just not mandatory.
            pass
        s.phase = CandidatePhase.REVIEWING
        s._note("reviewer_start", actor=actor, independent=True)
        return s

    def reviewer_emit(
        self,
        candidate_id: str,
        *,
        challenges: Sequence[Mapping[str, Any]],
        distinguishing_tests_requested: Sequence[str],
        severity_findings: Sequence[Mapping[str, Any]],
        recommendation: str,
        actor: str = Role.REVIEWER.value,
    ) -> ReviewReport:
        s = self._get(candidate_id)
        if s.phase != CandidatePhase.REVIEWING:
            raise OptionCRefusal(f"reviewer_emit requires REVIEWING, got {s.phase.value}")
        report = ReviewReport(
            candidate_id=candidate_id,
            challenges=list(challenges),
            distinguishing_tests_requested=list(distinguishing_tests_requested),
            severity_findings=list(severity_findings),
            recommendation=recommendation,
        )
        s.review_report = report
        s.phase = CandidatePhase.REVIEW_EMITTED
        s._note("review_emitted", review_sha256=report.review_sha256, actor=actor)
        return report

    def _get(self, candidate_id: str) -> CandidateSession:
        if candidate_id not in self.sessions:
            raise OptionCRefusal(f"unknown candidate {candidate_id!r}")
        return self.sessions[candidate_id]

    @staticmethod
    def _assert_action_allowed(role: Role, action: str) -> None:
        if role is Role.EXECUTOR and action in EXECUTOR_FORBIDDEN:
            raise OptionCRefusal(f"executor forbidden: {action}")
        if role is Role.REVIEWER and action in REVIEWER_FORBIDDEN:
            raise OptionCRefusal(f"reviewer forbidden: {action}")

    @staticmethod
    def role_map_to_tonight() -> dict[str, Any]:
        """Standing mapping from Option-C to the grok-orchestration pattern."""
        return {
            "schema": SCHEMA,
            "structural_identity": True,
            "mapping": {
                Role.EXECUTOR.value: {
                    "reference": "grok-run delegate",
                    "skill": "grok-orchestration/delegate.md",
                    "isolation": "git worktree on grok/<task-id>",
                    "artifacts": ["grok-report.md", "diff.patch"],
                    "may_edit": True,
                },
                Role.REVIEWER.value: {
                    "reference": "grok-run audit",
                    "skill": "grok-orchestration/audit.md",
                    "isolation": "read-only sandbox (kernel-enforced)",
                    "artifacts": ["audit findings"],
                    "may_edit": False,
                    "independence": "does not receive controller conclusions",
                },
                Role.PROTECTED_CONTROLLER.value: {
                    "reference": "Claude (architect + final authority)",
                    "skill": "grok-orchestration/SKILL.md",
                    "authority": [
                        "protected_parity",
                        "held_out_capability",
                        "clean_benchmark",
                        "sign_promotion_or_rollback",
                    ],
                    "never_transfers": True,
                    "claim_vs_receipt": "Grok report is a claim; controller verifies the artifact",
                },
            },
            "logical_not_simultaneous": True,
            "mandatory_review_categories": sorted(MANDATORY_REVIEW_CATEGORIES),
        }


class OptionCController:
    """Protected controller: only authority that may promote or roll back."""

    ROLE = Role.PROTECTED_CONTROLLER

    def __init__(self, sandbox: OptionCSandbox, *, controller_id: str = "protected_controller") -> None:
        self.sandbox = sandbox
        self.controller_id = controller_id

    def begin_eval(self, candidate_id: str) -> CandidateSession:
        s = self.sandbox._get(candidate_id)
        if s.requires_mandatory_review() and s.review_report is None:
            raise OptionCRefusal(
                f"category {s.category!r} requires mandatory 80B review before controller eval"
            )
        if s.phase not in {
            CandidatePhase.REVIEW_EMITTED,
            CandidatePhase.CANDIDATE_EMITTED,  # only if review not mandatory
        }:
            raise OptionCRefusal(
                f"controller eval requires REVIEW_EMITTED (or CANDIDATE_EMITTED when "
                f"review not mandatory); got {s.phase.value}"
            )
        if s.phase == CandidatePhase.CANDIDATE_EMITTED and s.requires_mandatory_review():
            raise OptionCRefusal("mandatory review missing")
        s.phase = CandidatePhase.CONTROLLER_EVAL
        s._note("controller_eval_start", actor=self.controller_id)
        return s

    def decide(
        self,
        candidate_id: str,
        *,
        protected_parity: Mapping[str, Any] | None = None,
        held_out_capability: Mapping[str, Any] | None = None,
        clean_benchmark: Mapping[str, Any] | None = None,
        result_class: str | None = None,
        force_action: str | None = None,
    ) -> dict[str, Any]:
        """Run protected gates and sign promotion or rollback.

        Missing evidence → INSUFFICIENT_EVIDENCE / HOLD (honest PENDING), never
        fabricated PROMOTE. Mirrors frankenstein_promotion_gate honesty.
        """
        s = self.sandbox._get(candidate_id)
        if s.phase != CandidatePhase.CONTROLLER_EVAL:
            self.begin_eval(candidate_id)

        review_required = s.requires_mandatory_review()
        review_present = s.review_report is not None
        if review_required and not review_present:
            raise OptionCRefusal("cannot decide without mandatory review")

        checks: list[dict[str, Any]] = []

        def _gate(name: str, bundle: Mapping[str, Any] | None) -> str:
            if bundle is None:
                checks.append({"name": name, "status": "PENDING", "detail": "not provided"})
                return "PENDING"
            ok = bool(bundle.get("pass"))
            status = "PASS" if ok else "FAIL"
            checks.append({"name": name, "status": status, "detail": str(bundle)})
            return status

        p_status = _gate("protected_parity", protected_parity)
        h_status = _gate("held_out_capability", held_out_capability)
        c_status = _gate("clean_benchmark", clean_benchmark)

        statuses = {p_status, h_status, c_status}
        if force_action is not None:
            action = force_action
            if action == "PROMOTE" and "FAIL" in statuses:
                raise OptionCRefusal("cannot force PROMOTE when a protected gate FAILed")
            if action == "PROMOTE" and "PENDING" in statuses:
                raise OptionCRefusal("cannot force PROMOTE with PENDING evidence")
        elif "FAIL" in statuses:
            action = "REJECT"
        elif statuses == {"PASS"}:
            # Reviewer hard reject still blocks promotion.
            if (
                s.review_report is not None
                and s.review_report.recommendation == "REJECT"
            ):
                action = "REJECT"
            else:
                action = "PROMOTE"
        else:
            action = "HOLD"

        if result_class is None:
            result_class = {
                "PROMOTE": "PROMOTED_MECHANISM",
                "REJECT": "REJECTED_MECHANISM",
                "ROLLBACK": "REJECTED_MECHANISM",
                "HOLD": "INSUFFICIENT_EVIDENCE",
            }.get(action, "INSUFFICIENT_EVIDENCE")
        if result_class not in RESULT_CLASSES:
            raise OptionCRefusal(f"unknown result_class {result_class!r}")

        # Executor / reviewer may never be the signer.
        if self.controller_id in {
            Role.EXECUTOR.value,
            Role.REVIEWER.value,
            "executor",
            "reviewer",
        }:
            raise OptionCRefusal("sandbox roles cannot sign controller decisions")

        decision = ControllerDecision(
            candidate_id=candidate_id,
            result_class=result_class,
            action=action,
            protected_parity=dict(protected_parity or {}),
            held_out_capability=dict(held_out_capability or {}),
            clean_benchmark=dict(clean_benchmark or {}),
            signed_by=self.controller_id,
            review_required_and_present=review_required and review_present,
            reason=self._reason(action, checks, s),
        )
        s.decision = decision
        s.phase = {
            "PROMOTE": CandidatePhase.PROMOTED,
            "REJECT": CandidatePhase.REJECTED,
            "ROLLBACK": CandidatePhase.ROLLED_BACK,
            "HOLD": CandidatePhase.INSUFFICIENT_EVIDENCE,
        }.get(action, CandidatePhase.INSUFFICIENT_EVIDENCE)
        s._note("decision", action=action, result_class=result_class)

        document = {
            "schema": DECISION_SCHEMA,
            "decision": decision.to_dict(),
            "checks": checks,
            "session": {
                "candidate_id": s.candidate_id,
                "category": s.category,
                "phase": s.phase.value,
                "candidate_report_sha256": (
                    s.candidate_report.report_sha256 if s.candidate_report else None
                ),
                "review_report_sha256": (
                    s.review_report.review_sha256 if s.review_report else None
                ),
            },
            "fabricated_promote": False,
            "authority": Role.PROTECTED_CONTROLLER.value,
            "reference_pattern": "claude_controller_plus_grok_delegate_and_audit",
        }
        return seal(document)

    @staticmethod
    def _reason(action: str, checks: Sequence[Mapping[str, Any]], s: CandidateSession) -> str:
        if action == "PROMOTE":
            return "all protected gates PASS; controller signs promotion"
        if action == "HOLD":
            pending = [c["name"] for c in checks if c["status"] == "PENDING"]
            return f"evidence incomplete: {pending}; HOLD not PROMOTE"
        if action == "REJECT" and s.review_report and s.review_report.recommendation == "REJECT":
            return "reviewer recommended REJECT under mandatory review"
        fails = [c["name"] for c in checks if c["status"] == "FAIL"]
        if fails:
            return f"protected gate FAIL: {fails}"
        return f"action={action}"
