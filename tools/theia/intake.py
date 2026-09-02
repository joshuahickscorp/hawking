"""H.2 bounty intake pipeline as an explicit state machine.

Security-class bounties stop at AUTHORITY/SCOPE RESOLUTION: either
BLOCKED_RIGHTS or HALTED_BEFORE_ACTIVE_TEST. They never enter cheap
reproduction, artifacts, or any path that could probe a target.

Self-bounties run the full pipeline against a local receipt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from tools.theia.authority import (
    AuthorizationDecision,
    BlockedRights,
    resolve,
)
from tools.theia.bounty import Bounty, SECURITY_BOUNTY_CLASSES
from tools.theia.labs import LabKind, SelfBountyKind
from tools.theia.security import (
    ActiveTestRefused,
    LAST_LEGAL_STATE,
    SecurityMachine,
    SecurityState,
)
from tools.theia.value import (
    ScheduleScore,
    ValueRefused,
    VerifiedResult,
    accept_as_verified,
    bounty_value,
)


class IntakeStage(Enum):
    DISCOVER = "DISCOVER"
    AUTHORITY_SCOPE_RESOLUTION = "AUTHORITY/SCOPE RESOLUTION"
    DUPLICATE_KNOWN_SOLUTION_CHECK = "DUPLICATE / KNOWN-SOLUTION CHECK"
    VERIFIER_DEFINITION = "VERIFIER DEFINITION"
    COST_VALUE_ESTIMATE = "COST / VALUE ESTIMATE"
    CHEAP_REPRODUCTION_SCREEN = "CHEAP REPRODUCTION / SCREEN"
    PLAN = "PLAN"
    BOUNDED_WORKUNITS = "BOUNDED WORKUNITS"
    ARTIFACTS_TESTS_PROOFS_MEASUREMENTS = "ARTIFACTS / TESTS / PROOFS / MEASUREMENTS"
    INDEPENDENT_VERIFICATION = "INDEPENDENT VERIFICATION"
    SUBMISSION_PROMOTION = "SUBMISSION / PROMOTION"
    TRAJECTORY_METHOD_NEGATIVE_SCIENCE = "TRAJECTORY + METHOD + NEGATIVE SCIENCE"


INTAKE_ORDER: tuple[IntakeStage, ...] = tuple(IntakeStage)


class IntakeRefused(ValueError):
    pass


class VerificationFailed(ValueError):
    pass


@dataclass(frozen=True)
class HaltedBeforeActiveTest:
    last_legal_h3_state: str
    reason: str
    status: str = "HALTED_BEFORE_ACTIVE_TEST"


@dataclass
class IntakeResult:
    bounty_id: str
    source: str
    stages_visited: tuple[str, ...]
    final_stage: IntakeStage
    schedule_score: ScheduleScore | None
    verified_result: VerifiedResult | None
    blocked: BlockedRights | None
    security_halt: HaltedBeforeActiveTest | None
    notes: dict[str, Any] = field(default_factory=dict)
    lab: str = ""
    self_bounty_kind: str | None = None
    exit_code: int = 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "bounty_id": self.bounty_id,
            "source": self.source,
            "stages_visited": list(self.stages_visited),
            "final_stage": self.final_stage.value,
            "schedule_score": (
                self.schedule_score.to_json_dict()
                if self.schedule_score is not None
                else None
            ),
            "verified_result": (
                {
                    "kind": self.verified_result.kind,
                    "artifact": self.verified_result.artifact,
                    "detail": self.verified_result.detail,
                    "type": type(self.verified_result).__name__,
                }
                if self.verified_result is not None
                else None
            ),
            "blocked": (
                {
                    "status": self.blocked.status,
                    "reason": self.blocked.reason,
                    "detail": self.blocked.detail,
                }
                if self.blocked
                else None
            ),
            "security_halt": (
                {
                    "status": self.security_halt.status,
                    "last_legal_h3_state": self.security_halt.last_legal_h3_state,
                    "reason": self.security_halt.reason,
                }
                if self.security_halt
                else None
            ),
            "lab": self.lab,
            "self_bounty_kind": self.self_bounty_kind,
            "notes": self.notes,
            "exit_code": self.exit_code,
        }


def local_artifact(source: str) -> Path:
    if "://" in source:
        raise IntakeRefused("intake reads local artifacts only; refused URL source")
    path = Path(source)
    if not path.is_file():
        raise IntakeRefused(f"intake source is not a local file: {source}")
    return path


def recompute_seal(doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def verify_receipt(path: Path, *, expected_schema: str | None = None) -> VerifiedResult:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise VerificationFailed(f"{path} is not a JSON object")
    schema = doc.get("schema")
    if expected_schema is not None and schema != expected_schema:
        raise VerificationFailed(f"schema {schema!r} != {expected_schema!r}")
    claimed = doc.get("seal_sha256")
    if not isinstance(claimed, str) or not claimed:
        raise VerificationFailed(f"{path} has no seal_sha256")
    actual = recompute_seal(doc)
    if claimed != actual:
        raise VerificationFailed(f"seal mismatch for {path}")
    detail: dict[str, Any] = {"schema": schema, "seal_sha256": actual}
    if schema == "hawking.theia.math_formal.v1":
        from tools.theia.math_lab import verify_math_receipt

        detail.update(verify_math_receipt(path, doc))
    elif schema == "hawking.theia.systems_compiler.v1":
        from tools.theia.systems_lab import verify_systems_receipt

        detail.update(verify_systems_receipt(path, doc))
    else:
        from tools.theia.self_bounty import SCHEMA_KIND, independent_self_bounty_checks

        if schema in SCHEMA_KIND:
            detail.update(independent_self_bounty_checks(path, doc))
    return VerifiedResult(
        kind="receipt_seal",
        artifact=str(path),
        detail=detail,
    )


def run_intake(
    bounty: Bounty,
    *,
    value_inputs_factory,
    authority_file: Path | None = None,
    declared_target: str | None = None,
    known_ids: tuple[str, ...] = (),
    self_bounty_kind: SelfBountyKind | None = None,
    expected_schema: str | None = None,
) -> IntakeResult:
    visited: list[str] = []
    notes: dict[str, Any] = {}
    score: ScheduleScore | None = None
    verified: VerifiedResult | None = None

    def finish(
        stage: IntakeStage,
        *,
        blocked: BlockedRights | None = None,
        halt: HaltedBeforeActiveTest | None = None,
        exit_code: int = 1,
    ) -> IntakeResult:
        return IntakeResult(
            bounty_id=bounty.id,
            source=bounty.source,
            stages_visited=tuple(visited),
            final_stage=stage,
            schedule_score=score,
            verified_result=verified,
            blocked=blocked,
            security_halt=halt,
            notes=notes,
            lab=bounty.lab,
            self_bounty_kind=self_bounty_kind.value if self_bounty_kind else None,
            exit_code=exit_code,
        )

    # DISCOVER
    visited.append(IntakeStage.DISCOVER.value)
    try:
        artifact = local_artifact(bounty.source)
    except IntakeRefused as e:
        return finish(
            IntakeStage.DISCOVER,
            blocked=BlockedRights(reason="discover_failed", detail=str(e)),
        )
    notes["discovered"] = str(artifact)

    # AUTHORITY / SCOPE RESOLUTION
    visited.append(IntakeStage.AUTHORITY_SCOPE_RESOLUTION.value)
    if bounty.bounty_class in SECURITY_BOUNTY_CLASSES:
        decision: AuthorizationDecision = resolve(
            authority_file=authority_file,
            declared_target=declared_target,
            bounty_text=bounty.question_or_target,
        )
        notes["authorization"] = {
            "status": decision.status,
            "reason": decision.reason,
            "detail": decision.detail,
        }
        if decision.status == "BLOCKED_RIGHTS":
            return finish(
                IntakeStage.AUTHORITY_SCOPE_RESOLUTION,
                blocked=BlockedRights(
                    reason=decision.reason, detail=decision.detail
                ),
            )
        machine = SecurityMachine()
        if authority_file is not None:
            machine.pin_from_file(authority_file)
        machine.walk_to_last_legal()
        refused = False
        try:
            machine.advance(SecurityState.ACTIVE_TEST)
        except ActiveTestRefused:
            refused = True
        if not refused:
            raise RuntimeError("ACTIVE_TEST advance did not refuse")
        halt = HaltedBeforeActiveTest(
            last_legal_h3_state=LAST_LEGAL_STATE.value,
            reason=(
                "H.3 modeled through RATE/IMPACT_POLICY_PASS; ACTIVE_TEST "
                "refused; this scaffold will not take a security bounty into "
                "H.2 work stages"
            ),
        )
        notes["h3_state"] = machine.state.value
        return finish(
            IntakeStage.AUTHORITY_SCOPE_RESOLUTION,
            halt=halt,
            exit_code=0,
        )
    if bounty.authorization_scope.kind != "HAWKING_INTERNAL":
        return finish(
            IntakeStage.AUTHORITY_SCOPE_RESOLUTION,
            blocked=BlockedRights(
                reason="unpinned",
                detail="non-security bounty without HAWKING_INTERNAL scope",
            ),
        )
    notes["authorization"] = {
        "status": "HAWKING_INTERNAL",
        "reason": "hawking-internal laboratory work; not a security class",
    }

    # DUPLICATE / KNOWN-SOLUTION CHECK
    visited.append(IntakeStage.DUPLICATE_KNOWN_SOLUTION_CHECK.value)
    known = bounty.id in known_ids
    notes["known_solution"] = known
    # Known solutions still archive; the check ran.

    # VERIFIER DEFINITION
    visited.append(IntakeStage.VERIFIER_DEFINITION.value)
    if bounty.verifier != "tools.theia.intake.verify_receipt":
        return finish(
            IntakeStage.VERIFIER_DEFINITION,
            blocked=BlockedRights(
                reason="verifier",
                detail=f"unknown verifier {bounty.verifier!r}",
            ),
        )
    notes["verifier"] = bounty.verifier

    # COST / VALUE ESTIMATE — schedule only, never truth
    visited.append(IntakeStage.COST_VALUE_ESTIMATE.value)
    try:
        inputs = value_inputs_factory(artifact)
        score = bounty_value(inputs)
    except ValueRefused as e:
        return finish(
            IntakeStage.COST_VALUE_ESTIMATE,
            blocked=BlockedRights(reason="value_refused", detail=str(e)),
        )
    notes["schedule_declares_result_true"] = False

    # CHEAP REPRODUCTION / SCREEN
    visited.append(IntakeStage.CHEAP_REPRODUCTION_SCREEN.value)
    doc = json.loads(artifact.read_text())
    if not isinstance(doc, dict) or "schema" not in doc:
        return finish(
            IntakeStage.CHEAP_REPRODUCTION_SCREEN,
            blocked=BlockedRights(
                reason="screen", detail="receipt is not a schema-bearing object"
            ),
        )
    if expected_schema is not None and doc.get("schema") != expected_schema:
        return finish(
            IntakeStage.CHEAP_REPRODUCTION_SCREEN,
            blocked=BlockedRights(
                reason="screen",
                detail=f"schema {doc.get('schema')!r} != {expected_schema!r}",
            ),
        )
    notes["screen_schema"] = doc.get("schema")

    # PLAN
    visited.append(IntakeStage.PLAN.value)
    notes["plan"] = {
        "lab": bounty.lab,
        "kind": self_bounty_kind.value if self_bounty_kind else None,
        "actions": ["verify_receipt", "score_schedule", "stage"],
        "not_actions": ["train", "ACTIVE_TEST", "network", "scan"],
    }

    # BOUNDED WORKUNITS
    visited.append(IntakeStage.BOUNDED_WORKUNITS.value)
    notes["workunits"] = [
        {
            "id": f"{bounty.id}:ingest",
            "bound": bounty.budget.workunits,
            "lane": "CPU_ANALYSIS",
        }
    ]

    # ARTIFACTS / TESTS / PROOFS / MEASUREMENTS
    visited.append(IntakeStage.ARTIFACTS_TESTS_PROOFS_MEASUREMENTS.value)
    notes["artifacts"] = [str(artifact)]

    # INDEPENDENT VERIFICATION — the only place VerifiedResult is born
    visited.append(IntakeStage.INDEPENDENT_VERIFICATION.value)
    try:
        verified = accept_as_verified(
            verify_receipt(artifact, expected_schema=expected_schema)
        )
    except (VerificationFailed, TypeError) as e:
        return finish(
            IntakeStage.INDEPENDENT_VERIFICATION,
            blocked=BlockedRights(reason="verification_failed", detail=str(e)),
        )

    # SUBMISSION / PROMOTION — stage locally; do not publish
    visited.append(IntakeStage.SUBMISSION_PROMOTION.value)
    notes["submission"] = bounty.submission_policy

    # TRAJECTORY + METHOD + NEGATIVE SCIENCE
    visited.append(IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE.value)
    laws = []
    if isinstance(doc.get("scars"), list):
        laws = [s.get("law") for s in doc["scars"] if s.get("law")]
    if doc.get("general_law"):
        laws.append(doc["general_law"])
    notes["trajectory"] = {
        "method": "local receipt intake; no training; no active test",
        "laws": laws,
        "negative_science": bool(laws),
    }

    if len(visited) != len(INTAKE_ORDER):
        return finish(
            IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE,
            blocked=BlockedRights(
                reason="incomplete",
                detail=f"visited {len(visited)} != {len(INTAKE_ORDER)} H.2 stages",
            ),
        )
    return finish(
        IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE, exit_code=0
    )
