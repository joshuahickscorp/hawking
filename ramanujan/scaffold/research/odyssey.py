#!/usr/bin/env python3.12
"""Fixture-only Odyssey control plane for the compact Ramanujan scaffold.

This is intentionally a *control plane*, not a research launcher.  It gives the
pre-sandbox campaign one auditable, dependency-light home for the parts that did
not fit the original Q0--Q6 scaffold: T0--T12 sequencing, questioning,
intervention/debt records, structured traces, transfer rehearsal, cold storage,
F0--F12 and Q0--Q12 fixture runners, and attack injection.

The module has three deliberately hard boundaries:

* it admits only ``fixture_only`` Director environments;
* it never sets or accepts ``RAMANUJAN_RESEARCH_AUTHORIZED=true``;
* it never promotes a claim.  Existing Stores, independent verifiers, and the
  Tribunal remain the only promotion surface.

Keeping these mechanisms together avoids creating a second sprawling campaign
tree while leaving the sealed Q0 receipts and their hash-bound modules intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ramanujan.evidence import Tier
from ramanujan.ledger import Ledger
from ramanujan.limits import LimitRegistry
from ramanujan.stores import Stores, TribunalRefused


AUTHORITY = "NON_PRODUCTION_AUTHORITY"
RESEARCH_AUTHORIZED = False
SCHEMA = "hawking.ramanujan.odyssey_fixture.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OdysseyRefused(RuntimeError):
    """A fixture controller was asked to exceed its evidence or authority boundary."""


class StorageRefused(OdysseyRefused):
    """The phase/storage discipline cannot safely satisfy a requested operation."""


class TransitionRefused(OdysseyRefused):
    """A T/F/Q phase was skipped, repeated, or attempted out of order."""


def _canonical_value(value: Any) -> Any:
    """Reduce the small record vocabulary to deterministic JSON-safe values."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_bytes(value: Any) -> bytes:
    """Stable bytes for identities, receipts, shards, and delta checkpoints."""
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OdysseyRefused(f"{name} must be a lowercase SHA-256 hex digest")


def _safe_token(value: str, name: str = "identifier") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise OdysseyRefused(f"{name} must be a non-empty safe string")
    if value.startswith(("/", "\\")) or ".." in Path(value).parts:
        raise OdysseyRefused(f"{name} may not be absolute or traverse a parent directory")
    return value


class AuthorityBasis(str, Enum):
    """Where an answer derives authority; a role name is intentionally absent."""

    FORMAL_LIBRARY = "FORMAL_LIBRARY"
    PRIMARY_SOURCE = "PRIMARY_SOURCE"
    EXACT_COMPUTATION = "EXACT_COMPUTATION"
    METHOD_CAPSULE = "METHOD_CAPSULE"
    MODEL_INFERENCE_ONLY = "MODEL_INFERENCE_ONLY"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class OdysseyTier(IntEnum):
    """Extended reporting lattice layered over the sealed legacy Tier 0--3 API."""

    ASSERTED = 0
    EMPIRICALLY_SUPPORTED = 1
    FORMALIZED_OR_CERTIFIED = 2
    PROVEN_AND_REPLAYED = 3
    INDEPENDENTLY_REPRODUCED = 4
    EXPERT_REVIEWED_FOR_SCOPE_AND_SIGNIFICANCE = 5


_TIER_REQUIREMENTS: dict[OdysseyTier, tuple[AuthorityBasis, ...]] = {
    OdysseyTier.EMPIRICALLY_SUPPORTED: (AuthorityBasis.EXACT_COMPUTATION,),
    OdysseyTier.FORMALIZED_OR_CERTIFIED: (
        AuthorityBasis.FORMAL_LIBRARY,
        AuthorityBasis.EXACT_COMPUTATION,
    ),
    OdysseyTier.PROVEN_AND_REPLAYED: (AuthorityBasis.FORMAL_LIBRARY,),
    OdysseyTier.INDEPENDENTLY_REPRODUCED: (
        AuthorityBasis.FORMAL_LIBRARY,
        AuthorityBasis.EXACT_COMPUTATION,
    ),
    OdysseyTier.EXPERT_REVIEWED_FOR_SCOPE_AND_SIGNIFICANCE: (AuthorityBasis.HUMAN_REVIEW,),
}


@dataclass(frozen=True)
class EvidenceRecord:
    """One typed item in the extended evidence lattice.

    The record is for reporting and audit.  It cannot manufacture a legacy
    ``VerifierEvent`` or alter a ``Stores`` claim by itself.
    """

    tier: OdysseyTier
    basis: AuthorityBasis
    actor: str
    artifact_hash: str
    independent: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_hash(self.artifact_hash, "evidence artifact_hash")
        if not self.actor.strip():
            raise OdysseyRefused("evidence actor must be non-empty")
        if self.tier == OdysseyTier.ASSERTED:
            return
        if self.basis not in _TIER_REQUIREMENTS[self.tier]:
            raise OdysseyRefused(
                f"{self.tier.name} requires one of "
                f"{[item.value for item in _TIER_REQUIREMENTS[self.tier]]}, not {self.basis.value}"
            )
        if self.tier in {
            OdysseyTier.INDEPENDENTLY_REPRODUCED,
            OdysseyTier.EXPERT_REVIEWED_FOR_SCOPE_AND_SIGNIFICANCE,
        } and not self.independent:
            raise OdysseyRefused(f"{self.tier.name} requires an independent actor")


@dataclass
class EvidenceLattice:
    """Strict T0--T5 sequence; Tier 5 cannot substitute for replay correctness."""

    records: list[EvidenceRecord] = field(default_factory=list)

    @property
    def tier(self) -> OdysseyTier:
        return self.records[-1].tier if self.records else OdysseyTier.ASSERTED

    @property
    def legacy_tier(self) -> Tier:
        """Compatibility view; never reaches past the existing clean-replay Tier 3."""
        return Tier(min(int(self.tier), int(Tier.PROVEN)))

    def advance(self, record: EvidenceRecord) -> OdysseyTier:
        record.validate()
        current = self.tier
        if current is OdysseyTier.EXPERT_REVIEWED_FOR_SCOPE_AND_SIGNIFICANCE:
            raise OdysseyRefused("evidence lattice is already at its terminal Tier 5")
        if record.tier != OdysseyTier(int(current) + 1):
            raise OdysseyRefused(
                f"evidence tiers advance one step at a time ({current.name} -> "
                f"{OdysseyTier(int(current) + 1).name})"
            )
        self.records.append(record)
        return self.tier


@dataclass(frozen=True)
class EnvironmentFreeze:
    """T0/T1 identity lock for the only environment this local scaffold may admit."""

    run_id: str
    director_hash: str
    toolchain_hash: str
    corpus_manifest_hash: str
    membership_hash: str
    contamination_hash: str
    storage_receipt_hash: str
    director_kind: str = "fixture"
    fixture_only: bool = True
    immutable_director: bool = True
    research_authorized: bool = False

    def validate(self) -> None:
        _safe_token(self.run_id, "run_id")
        for name in (
            "director_hash",
            "toolchain_hash",
            "corpus_manifest_hash",
            "membership_hash",
            "contamination_hash",
            "storage_receipt_hash",
        ):
            _require_hash(getattr(self, name), name)
        if self.director_kind != "fixture" or not self.fixture_only:
            raise OdysseyRefused("this repository may scaffold only a fixture Director environment")
        if not self.immutable_director:
            raise OdysseyRefused("the Director must be immutable and mounted read-only")
        if self.research_authorized or RESEARCH_AUTHORIZED:
            raise OdysseyRefused("RAMANUJAN_RESEARCH_AUTHORIZED must remain false")

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "director_hash": self.director_hash,
            "director_kind": self.director_kind,
            "fixture_only": True,
            "immutable_director": True,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "freeze_hash": content_hash(asdict(self)),
        }


def freeze_environment(**kwargs: Any) -> EnvironmentFreeze:
    """Create and validate a fixture-only T1 environment freeze."""
    freeze = EnvironmentFreeze(**kwargs)
    freeze.validate()
    return freeze


def admit_substrate(freeze: EnvironmentFreeze) -> dict[str, Any]:
    """T0 admission receipt; a fixture admission is never a production claim."""
    return {
        **freeze.receipt(),
        "stage": "T0",
        "status": "ADMITTED_FIXTURE_SUBSTRATE",
        "production_authority": False,
        "research_authority": False,
    }


ODYSSEY_STAGES = tuple(f"T{number}" for number in range(13))
TRAINING_STAGES = tuple(f"F{number}" for number in range(13))
QUALIFICATION_STAGES = tuple(f"Q{number}" for number in range(13))

T_STAGE_PURPOSE = {
    "T0": "substrate_admission",
    "T1": "environment_and_corpus_freeze",
    "T2": "institute_bootstrap",
    "T3": "verified_teacher_traces",
    "T4": "supervised_behavior",
    "T5": "solved_case_reconstruction",
    "T6": "expert_questioning",
    "T7": "expert_iteration",
    "T8": "falsification_and_repair",
    "T9": "verifier_guided_preference_and_rl",
    "T10": "transfer_and_perturbation",
    "T11": "endurance_rehearsal",
    "T12": "checkpoint_tournament",
}


@dataclass(frozen=True)
class StageRecord:
    family: str
    stage: str
    artifacts_hash: str
    note: str
    fixture_only: bool = True


@dataclass
class StageMachine:
    """Append-only ordered campaign state for T, F, or Q stages."""

    family: str
    stages: tuple[str, ...]
    ledger: Ledger | None = None
    records: list[StageRecord] = field(default_factory=list)

    def _expected(self) -> str:
        if len(self.records) >= len(self.stages):
            raise TransitionRefused(f"{self.family} is already complete")
        return self.stages[len(self.records)]

    def advance(self, stage: str, *, artifacts: Mapping[str, Any], note: str) -> StageRecord:
        expected = self._expected()
        if stage != expected:
            raise TransitionRefused(f"expected {expected}, received {stage}; campaign stages cannot be skipped")
        if artifacts.get("RAMANUJAN_RESEARCH_AUTHORIZED") is True or artifacts.get("research_authority") is True:
            raise OdysseyRefused("a fixture stage may not carry research authority")
        record = StageRecord(self.family, stage, content_hash(dict(artifacts)), str(note), True)
        self.records.append(record)
        if self.ledger is not None:
            self.ledger.append(
                "checkpoint",
                {
                    "record_type": "odyssey_stage",
                    "family": self.family,
                    "stage": stage,
                    "artifacts_hash": record.artifacts_hash,
                    "fixture_only": True,
                },
                actor="odyssey-controller",
            )
        return record

    def checkpoint(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "family": self.family,
            "stages": list(self.stages),
            "records": [asdict(record) for record in self.records],
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        }
        return {**body, "checkpoint_hash": content_hash(body)}

    @classmethod
    def resume(cls, checkpoint: Mapping[str, Any], *, ledger: Ledger | None = None) -> "StageMachine":
        unsigned = {key: value for key, value in checkpoint.items() if key != "checkpoint_hash"}
        if checkpoint.get("schema") != SCHEMA or checkpoint.get("checkpoint_hash") != content_hash(unsigned):
            raise TransitionRefused("stage checkpoint hash or schema mismatch")
        if checkpoint.get("RAMANUJAN_RESEARCH_AUTHORIZED") is not False:
            raise OdysseyRefused("resumed checkpoint illegally grants research authority")
        stages = tuple(checkpoint.get("stages") or ())
        records = [StageRecord(**dict(row)) for row in checkpoint.get("records") or ()]
        if tuple(record.stage for record in records) != stages[: len(records)]:
            raise TransitionRefused("resumed stage records are not an ordered prefix")
        return cls(str(checkpoint["family"]), stages, ledger=ledger, records=records)


class QuestionCategory(str, Enum):
    SCOPE = "scope"
    LANDSCAPE = "landscape"
    METHOD = "method"
    COUNTEREXAMPLE = "counterexample"
    PROOF = "proof"
    FORMALIZATION = "formalization"
    COMPUTATION = "computation"
    SEMINAR = "seminar"
    NOVELTY = "novelty"
    ABANDONMENT = "abandonment"


_QUESTION_BANK: dict[QuestionCategory, str] = {
    QuestionCategory.SCOPE: "What exact domains, quantifiers, edge cases, and non-scope are asserted?",
    QuestionCategory.LANDSCAPE: "What known theorem, counterexample, and literature boundary overlap this claim?",
    QuestionCategory.METHOD: "Why this method, what cheap experiment discriminates it, and what kills it?",
    QuestionCategory.COUNTEREXAMPLE: "What is the smallest adversarial or degenerate counterexample search?",
    QuestionCategory.PROOF: "Which decisive lemma is fragile, and what is its dependency chain?",
    QuestionCategory.FORMALIZATION: "Does the formal statement preserve the informal theorem without hidden assumptions?",
    QuestionCategory.COMPUTATION: "Is arithmetic exact, reproducible, independently checkable, and range-complete?",
    QuestionCategory.SEMINAR: "Explain the hardest step, failed natural method, and nearest false generalization.",
    QuestionCategory.NOVELTY: "What provenance and qualified review support the novelty and attribution boundary?",
    QuestionCategory.ABANDONMENT: "What would bury this branch, and what evidence would reopen it?",
}


@dataclass(frozen=True)
class Question:
    id: str
    category: QuestionCategory
    prompt: str
    answer: str | None = None
    bases: tuple[AuthorityBasis, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    @property
    def certifiable(self) -> bool:
        return bool(self.answer and self.evidence_refs and any(
            basis is not AuthorityBasis.MODEL_INFERENCE_ONLY for basis in self.bases
        ))


@dataclass(frozen=True)
class QuestionRevision:
    question_id: str
    prior_answer_hash: str
    replacement_answer_hash: str
    prior_bases: tuple[AuthorityBasis, ...]
    replacement_bases: tuple[AuthorityBasis, ...]


@dataclass
class QuestionGraph:
    """Explicit questioning surface; model inference may hypothesize but never close it."""

    case_id: str
    questions: dict[str, Question] = field(default_factory=dict)
    answer_history: list[QuestionRevision] = field(default_factory=list)

    @classmethod
    def for_case(cls, case_id: str) -> "QuestionGraph":
        graph = cls(case_id)
        for category, prompt in _QUESTION_BANK.items():
            qid = f"{case_id}:{category.value}"
            graph.questions[qid] = Question(qid, category, prompt)
        return graph

    def answer(
        self,
        question_id: str,
        answer: str,
        *,
        bases: Sequence[AuthorityBasis],
        evidence_refs: Sequence[str],
        certifying: bool = False,
    ) -> Question:
        if question_id not in self.questions:
            raise KeyError(f"unknown question {question_id!r}")
        if not isinstance(answer, str) or not answer.strip():
            raise OdysseyRefused("question answers must be non-empty")
        typed_bases = tuple(AuthorityBasis(item) for item in bases)
        refs = tuple(_safe_token(str(item), "evidence reference") for item in evidence_refs)
        if not typed_bases:
            raise OdysseyRefused("every answer must disclose an authority basis")
        if certifying and (
            not refs or all(item is AuthorityBasis.MODEL_INFERENCE_ONLY for item in typed_bases)
        ):
            raise OdysseyRefused("MODEL_INFERENCE_ONLY cannot certify a Question Graph answer")
        prior = self.questions[question_id]
        answered = replace(prior, answer=answer.strip(), bases=typed_bases, evidence_refs=refs)
        if prior.answer is not None:
            self.answer_history.append(
                QuestionRevision(
                    question_id,
                    text_hash(prior.answer),
                    text_hash(answered.answer or ""),
                    prior.bases,
                    answered.bases,
                )
            )
        self.questions[question_id] = answered
        return answered

    def unanswered(self) -> list[Question]:
        return [question for question in self.questions.values() if question.answer is None]

    def uncertified(self) -> list[Question]:
        return [question for question in self.questions.values() if question.answer is not None and not question.certifiable]

    def coverage(self) -> dict[str, Any]:
        total = len(self.questions)
        answered = total - len(self.unanswered())
        certified = sum(question.certifiable for question in self.questions.values())
        return {
            "case_id": self.case_id,
            "total": total,
            "answered": answered,
            "certified": certified,
            "unanswered_categories": [question.category.value for question in self.unanswered()],
            "uncertified_categories": [question.category.value for question in self.uncertified()],
            "answer_revisions": len(self.answer_history),
        }


INTERVENTION_KINDS = frozenset(
    {
        "problem_selection",
        "source_selection",
        "representation_choice",
        "formal_language_choice",
        "evaluator_design",
        "algorithm_skeleton",
        "hint",
        "retry_instruction",
        "branch_pruning",
        "attempt_selection",
        "clarification_request",
        "proof_rewrite",
        "external_expert_comment",
        "certificate_correction",
        "scope_narrowing",
    }
)


@dataclass(frozen=True)
class InterventionEvent:
    actor: str
    kind: str
    input_state_hash: str
    intervention: str
    reason: str
    branch_change: str
    autonomy_affected: bool
    ledger_seq: int | None = None


@dataclass
class InterventionLedger:
    """Records human/external steering without pretending it was autonomous."""

    ledger: Ledger | None = None
    events: list[InterventionEvent] = field(default_factory=list)

    def record(
        self,
        *,
        actor: str,
        kind: str,
        input_state: Mapping[str, Any],
        intervention: str,
        reason: str,
        branch_change: str,
        autonomy_affected: bool = True,
    ) -> InterventionEvent:
        if kind not in INTERVENTION_KINDS:
            raise OdysseyRefused(f"unknown intervention kind {kind!r}")
        if not actor.strip() or actor.lower().startswith(("model", "director", "student")):
            raise OdysseyRefused("an Intervention Ledger event must identify a human or external actor")
        if not all(isinstance(value, str) and value.strip() for value in (intervention, reason, branch_change)):
            raise OdysseyRefused("intervention, reason, and branch change must be non-empty")
        state_hash = content_hash(dict(input_state))
        seq = None
        if self.ledger is not None:
            row = self.ledger.append(
                "checkpoint",
                {
                    "record_type": "human_intervention",
                    "kind": kind,
                    "input_state_hash": state_hash,
                    "branch_change": branch_change,
                    "autonomy_affected": bool(autonomy_affected),
                },
                actor=actor,
            )
            seq = row.seq
        event = InterventionEvent(
            actor, kind, state_hash, intervention, reason, branch_change, bool(autonomy_affected), seq
        )
        self.events.append(event)
        return event

    def autonomy_label(self) -> str:
        kinds = {event.kind for event in self.events if event.autonomy_affected}
        if "external_expert_comment" in kinds:
            return "EXPERT_VERIFIED" if "attempt_selection" not in kinds else "HUMAN_AI_COLLABORATION"
        if "attempt_selection" in kinds:
            return "HUMAN_SELECTED"
        if kinds:
            return "HUMAN_GUIDED"
        return "AUTONOMOUS_WITH_FIXED_ENVIRONMENT"


DEBT_KINDS = frozenset(
    {
        "undefined_specialized_term",
        "unverified_literature_claim",
        "unformalized_decisive_lemma",
        "unreproduced_computation",
        "unknown_standard_counterexample",
        "uncertain_novelty_boundary",
        "missing_domain_reviewer",
    }
)


@dataclass(frozen=True)
class ExpertiseDebtItem:
    id: str
    branch_id: str
    kind: str
    severity: int
    description_hash: str
    resolved_by: AuthorityBasis | None = None

    @property
    def open(self) -> bool:
        return self.resolved_by is None


@dataclass
class ExpertiseDebtTracker:
    """Economist-visible debt has metadata only; statement prose never enters allocation."""

    ledger: Ledger | None = None
    items: dict[str, ExpertiseDebtItem] = field(default_factory=dict)

    def add(self, branch_id: str, kind: str, description: str, *, severity: int = 1, actor: str = "skeptic") -> ExpertiseDebtItem:
        _safe_token(branch_id, "branch_id")
        if kind not in DEBT_KINDS or not description.strip() or not 1 <= int(severity) <= 5:
            raise OdysseyRefused("expertise debt needs a known kind, non-empty description, and severity 1..5")
        item_id = content_hash({"branch": branch_id, "kind": kind, "description": description, "n": len(self.items)})[:20]
        item = ExpertiseDebtItem(item_id, branch_id, kind, int(severity), text_hash(description))
        self.items[item_id] = item
        if self.ledger is not None:
            self.ledger.append(
                "objection",
                {"record_type": "expertise_debt", "id": item_id, "branch": branch_id, "kind": kind, "severity": severity},
                actor=actor,
            )
        return item

    def resolve(self, item_id: str, basis: AuthorityBasis, *, actor: str = "librarian") -> ExpertiseDebtItem:
        item = self.items[item_id]
        if basis is AuthorityBasis.MODEL_INFERENCE_ONLY:
            raise OdysseyRefused("model inference alone cannot retire Expertise Debt")
        if not item.open:
            raise OdysseyRefused(f"expertise debt {item_id} is already resolved")
        resolved = replace(item, resolved_by=basis)
        self.items[item_id] = resolved
        if self.ledger is not None:
            self.ledger.append(
                "verifier_event",
                {"record_type": "expertise_debt_resolution", "id": item_id, "basis": basis.value},
                actor=actor,
            )
        return resolved

    def summary(self, branch_id: str) -> dict[str, Any]:
        rows = [item for item in self.items.values() if item.branch_id == branch_id]
        unresolved = [item for item in rows if item.open]
        return {
            "branch_id": branch_id,
            "open_count": len(unresolved),
            "debt_score": sum(item.severity for item in unresolved),
            "kinds": sorted(item.kind for item in unresolved),
        }


@dataclass(frozen=True)
class BudgetDecision:
    branch_id: str
    granted: bool
    units: int
    reason: str
    metadata: Mapping[str, Any]


@dataclass
class OdysseyEconomist:
    """Metadata-only compute allocator; convincing claim text is not an input."""

    debt: ExpertiseDebtTracker
    max_debt_score: int = 8
    ledger: Ledger | None = None

    def allocate(
        self,
        branch_id: str,
        *,
        requested_units: int,
        verification_score: float,
        scope_changes: int,
    ) -> BudgetDecision:
        if requested_units <= 0 or not 0.0 <= verification_score <= 1.0 or scope_changes < 0:
            raise OdysseyRefused("budget allocation accepts only bounded metadata")
        debt = self.debt.summary(branch_id)
        metadata = {**debt, "verification_score": verification_score, "scope_changes": scope_changes}
        if debt["debt_score"] > self.max_debt_score:
            decision = BudgetDecision(branch_id, False, 0, "rising_expertise_debt", metadata)
        elif verification_score <= 0.0 and debt["open_count"]:
            decision = BudgetDecision(branch_id, False, 0, "no_verification_signal", metadata)
        elif scope_changes > 2:
            decision = BudgetDecision(branch_id, False, 0, "repeated_scope_changes", metadata)
        else:
            decision = BudgetDecision(branch_id, True, requested_units, "metadata_gate_passed", metadata)
        if self.ledger is not None:
            self.ledger.append(
                "budget_grant",
                {"record_type": "odyssey_economist", "granted": decision.granted, "units": decision.units, **metadata},
                actor="economist",
            )
        return decision


class BranchStatus(str, Enum):
    OPEN = "OPEN"
    NARROWED = "NARROWED"
    BURIED = "BURIED"
    REOPENED = "REOPENED"
    HALTED = "HALTED"


@dataclass(frozen=True)
class ResearchBranch:
    """Odyssey branch metadata linked to the existing claim/Graveyard stores.

    This object deliberately keeps no proof prose.  Its public economics view is
    therefore safe to hand to the Economist without exposing claim content.
    """

    id: str
    problem_id: str
    claim_id: str
    parent_id: str | None
    method_family: str
    falsification_plan: str
    reopen_condition: str
    authority_bases: tuple[AuthorityBasis, ...] = ()
    status: BranchStatus = BranchStatus.OPEN
    failure_reason: str | None = None

    def metadata_for_economist(self, debt: ExpertiseDebtTracker, *, verification_score: float, scope_changes: int) -> dict[str, Any]:
        if not 0.0 <= verification_score <= 1.0 or scope_changes < 0:
            raise OdysseyRefused("branch economics metadata is malformed")
        return {
            "branch_id": self.id,
            "status": self.status.value,
            "method_family": self.method_family,
            "authority_bases": [basis.value for basis in self.authority_bases],
            "verification_score": verification_score,
            "scope_changes": scope_changes,
            **debt.summary(self.id),
        }

    def bury(self, reason: str) -> "ResearchBranch":
        if self.status is BranchStatus.BURIED:
            raise OdysseyRefused("an already buried branch cannot be buried again")
        if not reason.strip():
            raise OdysseyRefused("a burial needs a concrete failure reason")
        return replace(self, status=BranchStatus.BURIED, failure_reason=reason)

    def reopen(self, evidence: EvidenceRecord) -> "ResearchBranch":
        if self.status is not BranchStatus.BURIED:
            raise OdysseyRefused("only a buried branch can reopen")
        evidence.validate()
        if evidence.basis is AuthorityBasis.MODEL_INFERENCE_ONLY:
            raise OdysseyRefused("a buried branch cannot reopen on model inference alone")
        return replace(self, status=BranchStatus.REOPENED, failure_reason=None)


@dataclass(frozen=True)
class CaseEvidence:
    basis: AuthorityBasis
    locator: str
    artifact_hash: str

    def validate(self) -> None:
        _safe_token(self.locator, "case evidence locator")
        _require_hash(self.artifact_hash, "case evidence artifact_hash")


@dataclass(frozen=True)
class CaseStudy:
    """Known/bounded case data; it deliberately has no field for a production novelty claim."""

    id: str
    honest_statement: str
    scope: tuple[str, ...]
    provenance: tuple[CaseEvidence, ...]
    known_solution_visibility: str
    expected_structure: tuple[str, ...]
    disclosure_cards: tuple[tuple[str, str], ...]
    contamination_risk: str
    transfer_kinds: tuple[str, ...] = (
        "same_method_new_parameters",
        "same_obstruction_new_domain",
        "false_nearby_conjecture",
        "weakened_true_theorem",
        "strengthened_false_theorem",
        "different_representation",
    )

    @property
    def hash(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class CaseAdmission:
    case_id: str
    accepted: bool
    evidence_grade: str
    reasons: tuple[str, ...]
    case_hash: str | None = None


def ingest_case(case: CaseStudy) -> CaseAdmission:
    """Validate case provenance and grade evidence without claiming mathematical novelty."""
    reasons: list[str] = []
    if not case.id or not case.honest_statement.strip() or not case.scope:
        reasons.append("missing_honest_statement_or_scope")
    if case.known_solution_visibility not in {"blind", "progressive", "known", "hidden_transfer"}:
        reasons.append("invalid_known_solution_visibility")
    if not case.expected_structure:
        reasons.append("missing_structural_evaluation_target")
    if not case.disclosure_cards:
        reasons.append("missing_progressive_disclosure_cards")
    for evidence in case.provenance:
        try:
            evidence.validate()
        except OdysseyRefused as exc:
            reasons.append(str(exc))
    bases = {evidence.basis for evidence in case.provenance}
    if not bases or bases == {AuthorityBasis.MODEL_INFERENCE_ONLY}:
        reasons.append("model_inference_only_is_not_case_provenance")
    if AuthorityBasis.FORMAL_LIBRARY in bases or AuthorityBasis.EXACT_COMPUTATION in bases:
        grade = "CERTIFICATE_BACKED"
    elif AuthorityBasis.PRIMARY_SOURCE in bases:
        grade = "SOURCE_BACKED"
    else:
        grade = "PROVISIONAL"
    return CaseAdmission(case.id, not reasons, grade, tuple(reasons), case.hash if not reasons else None)


@dataclass(frozen=True)
class MethodCapsule:
    """Structural learning target; canonical-proof wording is intentionally absent."""

    problem_id: str
    provenance: tuple[str, ...]
    honest_statement: str
    definitions: tuple[str, ...]
    known_landscape: tuple[str, ...]
    obstruction: str
    failed_natural_approaches: tuple[str, ...]
    decisive_observation: str
    method_selection_evidence: tuple[str, ...]
    winning_method: str
    critical_lemma_graph: tuple[tuple[str, tuple[str, ...]], ...]
    instruments: tuple[str, ...]
    scope: tuple[str, ...]
    non_scope: tuple[str, ...]
    human_interventions: tuple[str, ...]
    alternative_solutions: tuple[str, ...]
    transfer_variants: tuple[str, ...]
    contamination_risk: str
    reopen_conditions: tuple[str, ...]

    def validate(self) -> None:
        required_scalars = {
            "problem_id": self.problem_id,
            "honest_statement": self.honest_statement,
            "obstruction": self.obstruction,
            "decisive_observation": self.decisive_observation,
            "winning_method": self.winning_method,
            "contamination_risk": self.contamination_risk,
        }
        if any(not isinstance(value, str) or not value.strip() for value in required_scalars.values()):
            raise OdysseyRefused("Method Capsule has an empty required scalar")
        required_lists = {
            "provenance": self.provenance,
            "definitions": self.definitions,
            "known_landscape": self.known_landscape,
            "failed_natural_approaches": self.failed_natural_approaches,
            "method_selection_evidence": self.method_selection_evidence,
            "critical_lemma_graph": self.critical_lemma_graph,
            "instruments": self.instruments,
            "scope": self.scope,
            "non_scope": self.non_scope,
            "transfer_variants": self.transfer_variants,
            "reopen_conditions": self.reopen_conditions,
        }
        if any(not value for value in required_lists.values()):
            raise OdysseyRefused("Method Capsule has an empty required structural field")
        names = [node for node, _deps in self.critical_lemma_graph]
        if len(names) != len(set(names)):
            raise OdysseyRefused("critical lemma graph has duplicate node names")

    @property
    def hash(self) -> str:
        self.validate()
        return content_hash(asdict(self))


@dataclass(frozen=True)
class TransferVariant:
    id: str
    source_case_id: str
    kind: str
    prompt: str
    expected_structure: tuple[str, ...]
    known_solution_visible: bool = False
    requires_verification: bool = True


class PerturbationGenerator:
    """Deterministic structural variants, explicitly not a source of unverified mathematics."""

    def generate(self, case: CaseStudy, *, kind: str, seed: int) -> TransferVariant:
        if kind not in case.transfer_kinds:
            raise OdysseyRefused(f"transfer kind {kind!r} is not declared by case {case.id!r}")
        nonce = content_hash({"case": case.hash, "kind": kind, "seed": int(seed)})[:12]
        prompt = (
            f"Fixture transfer {nonce}: preserve the structural method for {case.id}, "
            f"but apply the declared perturbation {kind.replace('_', ' ')}. "
            "This variant needs independent verification before any mathematical use."
        )
        return TransferVariant(
            id=f"{case.id}:{kind}:{nonce}",
            source_case_id=case.id,
            kind=kind,
            prompt=prompt,
            expected_structure=case.expected_structure,
        )


@dataclass(frozen=True)
class ReconstructionResult:
    case_id: str
    disclosed_cards: tuple[str, ...]
    structural_recall: float
    structural_precision: float
    leaked_solution: bool


@dataclass
class ProgressiveDisclosureHarness:
    """Evaluates invariant/reduction alignment, never lexical similarity to a reference proof."""

    def reveal(self, case: CaseStudy, level: int) -> dict[str, Any]:
        if level < 0:
            raise OdysseyRefused("disclosure level must be non-negative")
        cards = case.disclosure_cards[:level]
        forbidden = {"canonical_proof", "full_solution", "reference_proof"}
        if any(name in forbidden for name, _value in cards):
            raise OdysseyRefused("progressive disclosure may not leak a canonical solution")
        return {
            "case_id": case.id,
            "statement": case.honest_statement,
            "scope": list(case.scope),
            "cards": list(cards),
            "known_solution_visible": False,
        }

    def evaluate(self, case: CaseStudy, proposed_structure: Iterable[str], *, disclosed_level: int) -> ReconstructionResult:
        proposed = {str(item).strip() for item in proposed_structure if str(item).strip()}
        expected = set(case.expected_structure)
        overlap = proposed & expected
        recall = len(overlap) / len(expected) if expected else 0.0
        precision = len(overlap) / len(proposed) if proposed else 0.0
        cards = tuple(name for name, _value in case.disclosure_cards[:disclosed_level])
        return ReconstructionResult(case.id, cards, recall, precision, False)


class SeminarVerdict(str, Enum):
    RETURN_FOR_REPAIR = "RETURN_FOR_REPAIR"
    NARROW = "NARROW"
    REQUEST_EXTERNAL_EXPERT = "REQUEST_EXTERNAL_EXPERT"
    PROVISIONAL_TO_TRIBUNAL = "PROVISIONAL_TO_TRIBUNAL"


@dataclass(frozen=True)
class SeminarRecord:
    case_id: str
    verdict: SeminarVerdict
    unanswered_categories: tuple[str, ...]
    uncertified_categories: tuple[str, ...]
    transfer_score: float
    fragile_lemma: str
    note: str


class SeminarRunner:
    """Seven-round oral-defense simulation; recommendation only, never admission."""

    def run(
        self,
        case: CaseStudy,
        graph: QuestionGraph,
        *,
        fragile_lemma: str,
        transfer: ReconstructionResult,
    ) -> SeminarRecord:
        if graph.case_id != case.id:
            raise OdysseyRefused("Question Graph and case mismatch")
        coverage = graph.coverage()
        unanswered = tuple(coverage["unanswered_categories"])
        uncertified = tuple(coverage["uncertified_categories"])
        if unanswered:
            verdict = SeminarVerdict.RETURN_FOR_REPAIR
            note = "Question Graph has unanswered defense lines."
        elif QuestionCategory.NOVELTY.value in uncertified:
            verdict = SeminarVerdict.REQUEST_EXTERNAL_EXPERT
            note = "Novelty/attribution is not grounded in qualified external authority."
        elif uncertified or transfer.structural_recall < 1.0:
            verdict = SeminarVerdict.NARROW
            note = "Scope, evidence, or structural transfer needs narrowing before review."
        else:
            verdict = SeminarVerdict.PROVISIONAL_TO_TRIBUNAL
            note = "Fixture seminar passed; existing independent Tribunal gates still apply."
        return SeminarRecord(
            case.id,
            verdict,
            unanswered,
            uncertified,
            transfer.structural_recall,
            fragile_lemma,
            note,
        )


def tribunal_adjudicate(
    stores: Stores,
    claim_id: str,
    *,
    admitting_actor: str = "tribunal",
    external_expert_gate: bool,
    review_packet_hash: str,
) -> dict[str, Any]:
    """Bridge a qualified external-review gate to the existing separated Tribunal.

    It delegates rather than recreates claim promotion.  The caller must provide
    an external gate and a hash-bound review packet; author self-admission and
    insufficient legacy evidence remain refused by ``Stores``.
    """
    _require_hash(review_packet_hash, "review_packet_hash")
    if not external_expert_gate:
        raise TribunalRefused("Tribunal admission requires an explicit external expert gate")
    stores.tribunal_admit(claim_id, admitting_actor=admitting_actor, human_expert_gate=True)
    return {
        "claim_id": claim_id,
        "status": "ADMITTED_BY_EXISTING_TRIBUNAL",
        "review_packet_hash": review_packet_hash,
        "production_authority": False,
        "research_authority": False,
    }


@dataclass(frozen=True)
class ExactCertificate:
    generator_id: str
    input_hash: str
    result: Any
    certificate: Mapping[str, Any]
    certificate_hash: str


class ExactComputationAdapter:
    """Generator/certificate split.  Exactness must be declared, then independently checked."""

    def compute(
        self,
        inputs: Mapping[str, Any],
        generator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        generator_id: str,
    ) -> ExactCertificate:
        _safe_token(generator_id, "generator_id")
        raw = dict(generator(dict(inputs)))
        if raw.get("arithmetic") != "exact" or "certificate" not in raw:
            raise OdysseyRefused("exact computation must declare arithmetic='exact' and return a certificate")
        certificate = dict(raw["certificate"])
        return ExactCertificate(
            generator_id=generator_id,
            input_hash=content_hash(dict(inputs)),
            result=raw.get("result"),
            certificate=certificate,
            certificate_hash=content_hash(certificate),
        )


class IndependentChecker:
    """A checker is intentionally distinct from the certificate generator."""

    def verify(
        self,
        certificate: ExactCertificate,
        checker: Callable[[Mapping[str, Any]], bool],
        *,
        checker_id: str,
    ) -> EvidenceRecord:
        _safe_token(checker_id, "checker_id")
        if checker_id == certificate.generator_id:
            raise OdysseyRefused("a certificate generator cannot be its own independent checker")
        if not bool(checker(certificate.certificate)):
            raise OdysseyRefused("independent checker rejected the exact certificate")
        return EvidenceRecord(
            OdysseyTier.EMPIRICALLY_SUPPORTED,
            AuthorityBasis.EXACT_COMPUTATION,
            actor=checker_id,
            artifact_hash=certificate.certificate_hash,
            independent=True,
            detail={"generator_id": certificate.generator_id, "input_hash": certificate.input_hash},
        )


class LeanAdapter:
    """Injectable pinned-Lean adapter; it never silently falls back to a host proof check."""

    def replay(
        self,
        capsule: Mapping[str, Any],
        replay: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        source = str(capsule.get("proof_lean") or "")
        if re.search(r"\b(sorry|admit|axiom)\b", source):
            raise OdysseyRefused("Lean adapter refuses holes or locally declared axioms")
        result = dict(replay(dict(capsule)))
        if result.get("environment") != "pinned_clean_container":
            raise OdysseyRefused("Lean replay must identify the pinned clean container")
        if result.get("ok") is not True:
            raise OdysseyRefused("pinned Lean replay did not succeed")
        _require_hash(str(result.get("container_hash") or ""), "container_hash")
        return result


class TraceDisposition(str, Enum):
    LEAN_VERIFIED = "LEAN_VERIFIED"
    EXACTLY_REPRODUCED = "EXACTLY_REPRODUCED"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    PLAUSIBLE_UNVERIFIED = "PLAUSIBLE_UNVERIFIED"
    NEGATIVE = "NEGATIVE"
    REJECTED = "REJECTED"


_TRAINABLE_DISPOSITIONS = frozenset(
    {TraceDisposition.LEAN_VERIFIED, TraceDisposition.EXACTLY_REPRODUCED, TraceDisposition.HUMAN_REVIEWED}
)

DEFAULT_TOOL_ALLOWLIST = frozenset({"lean", "exact-checker", "fixture-search", "retrieval"})


@dataclass(frozen=True)
class ExternalInputAssessment:
    digest: str
    trusted_as_instruction: bool
    suspicious_patterns: tuple[str, ...]


class SandboxGuard:
    """Small pre-sandbox guard for untrusted papers, tool output, and tool requests.

    It does not claim OS isolation.  It makes the safe local default explicit:
    retrieved text is evidence to inspect, never executable instruction, and every
    tool request must be on a closed allowlist with safe path-like arguments.
    """

    _INJECTION_PATTERNS = {
        "ignore_previous": re.compile(r"ignore\s+(all\s+)?previous", re.I),
        "tool_directive": re.compile(r"(?:run|execute|shell|curl|wget)\s+[\w./-]+", re.I),
        "prompt_override": re.compile(r"system\s+prompt|developer\s+message", re.I),
    }

    def inspect_external_text(self, text: str) -> ExternalInputAssessment:
        if not isinstance(text, str):
            raise OdysseyRefused("external text must be text")
        patterns = tuple(name for name, rule in self._INJECTION_PATTERNS.items() if rule.search(text))
        return ExternalInputAssessment(text_hash(text), False, patterns)

    def validate_tool_call(self, call: Mapping[str, Any], *, allowlist: frozenset[str] = DEFAULT_TOOL_ALLOWLIST) -> None:
        tool = call.get("tool")
        if tool not in allowlist:
            raise OdysseyRefused(f"tool {tool!r} is not on the pre-sandbox allowlist")
        args = call.get("args") or {}
        if not isinstance(args, Mapping):
            raise OdysseyRefused("tool args must be a mapping")
        for value in args.values():
            if isinstance(value, str) and ("\x00" in value or value.startswith(("/", "\\")) or ".." in Path(value).parts):
                raise OdysseyRefused("tool arguments may not include unsafe path traversal")


@dataclass(frozen=True)
class TraceRecord:
    """Auditable trace target without raw hidden states, logits, KV caches, or free-form CoT."""

    id: str
    problem_hash: str
    statement_hash: str
    membership: str
    director_hash: str
    template_hash: str
    retrieval_set: tuple[str, ...]
    method_label: str
    plan: tuple[str, ...]
    subgoals: tuple[str, ...]
    formal_states: tuple[str, ...]
    actions: tuple[str, ...]
    tool_calls: tuple[Mapping[str, Any], ...]
    verifier_outcomes: tuple[Mapping[str, Any], ...]
    disposition: TraceDisposition = TraceDisposition.PLAUSIBLE_UNVERIFIED
    student_failure_class: str | None = None

    @property
    def hash(self) -> str:
        return content_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["disposition"] = self.disposition.value
        return body


@dataclass(frozen=True)
class TraceAssessment:
    trace: TraceRecord
    reasons: tuple[str, ...]

    @property
    def trainable(self) -> bool:
        return self.trace.disposition in _TRAINABLE_DISPOSITIONS


class TraceVerifier:
    """Dispositioner for structured traces; prose plausibility cannot enter student training."""

    def __init__(self, guard: SandboxGuard | None = None) -> None:
        self.guard = guard or SandboxGuard()

    def assess(self, trace: TraceRecord) -> TraceAssessment:
        reasons: list[str] = []
        for name in ("problem_hash", "statement_hash", "director_hash", "template_hash"):
            try:
                _require_hash(getattr(trace, name), name)
            except OdysseyRefused as exc:
                reasons.append(str(exc))
        if not trace.membership or not trace.method_label or not trace.plan or not trace.subgoals:
            reasons.append("trace lacks required structured plan/subgoal/membership fields")
        for item in trace.retrieval_set:
            try:
                _safe_token(item, "retrieval id")
            except OdysseyRefused as exc:
                reasons.append(str(exc))
        for call in trace.tool_calls:
            try:
                if not isinstance(call, Mapping):
                    raise OdysseyRefused("tool call must be a mapping")
                self.guard.validate_tool_call(call)
            except OdysseyRefused as exc:
                reasons.append(str(exc))
        outcomes = [row for row in trace.verifier_outcomes if isinstance(row, Mapping)]
        kinds = {str(row.get("kind")) for row in outcomes}
        lean_replay = next((row for row in outcomes if row.get("kind") == "lean_replay"), None)
        exact_check = next((row for row in outcomes if row.get("kind") == "independent_exact_check"), None)
        human_review = next((row for row in outcomes if row.get("kind") == "human_review"), None)
        if "lean_replay" in kinds:
            try:
                _require_hash(str((lean_replay or {}).get("container_hash") or ""), "Lean replay container_hash")
            except OdysseyRefused as exc:
                reasons.append(str(exc))
        if "independent_exact_check" in kinds:
            if not (exact_check or {}).get("checker_id") or (exact_check or {}).get("checker_id") == (exact_check or {}).get("generator_id"):
                reasons.append("independent exact check needs a distinct checker_id")
        if "human_review" in kinds:
            if not (human_review or {}).get("reviewer") or (human_review or {}).get("external") is not True:
                reasons.append("human review needs an identified external reviewer")
        if reasons:
            disposition = TraceDisposition.REJECTED
        elif "negative" in kinds:
            disposition = TraceDisposition.NEGATIVE
        elif "lean_replay" in kinds:
            disposition = TraceDisposition.LEAN_VERIFIED
        elif "independent_exact_check" in kinds:
            disposition = TraceDisposition.EXACTLY_REPRODUCED
        elif "human_review" in kinds:
            disposition = TraceDisposition.HUMAN_REVIEWED
        else:
            disposition = TraceDisposition.PLAUSIBLE_UNVERIFIED
        return TraceAssessment(replace(trace, disposition=disposition), tuple(reasons))


class StreamingDirectorTraceExecutor:
    """Streams a fixture Director callback into a structured trace; no parent can be mounted."""

    def __init__(self, *, limits: LimitRegistry | None = None) -> None:
        self.limits = limits or LimitRegistry()

    def stream(
        self,
        freeze: EnvironmentFreeze,
        case: CaseStudy,
        producer: Callable[[CaseStudy], Iterable[Mapping[str, Any]]],
    ) -> Iterator[Mapping[str, Any]]:
        freeze.validate()
        if not freeze.fixture_only or freeze.director_kind != "fixture":
            raise OdysseyRefused("streaming a real Director is outside the local scaffold")
        verdict = self.limits.consult("local_compute", role_id="director")
        if not verdict.allowed:
            raise OdysseyRefused(f"fixture local compute blocked by {verdict.blocking_limit}")
        for fragment in producer(case):
            if not isinstance(fragment, Mapping):
                raise OdysseyRefused("Director trace fragments must be structured mappings")
            # A fixture trace may carry plans/actions, never an authority-bearing answer field.
            if "research_authorized" in fragment or "production_authority" in fragment:
                raise OdysseyRefused("trace fragment attempted to grant authority")
            yield dict(fragment)

    def collect(
        self,
        freeze: EnvironmentFreeze,
        case: CaseStudy,
        *,
        membership: str,
        template_hash: str,
        producer: Callable[[CaseStudy], Iterable[Mapping[str, Any]]],
    ) -> TraceRecord:
        _require_hash(template_hash, "template_hash")
        _safe_token(membership, "membership")
        merged: dict[str, Any] = {}
        for fragment in self.stream(freeze, case, producer):
            merged.update(fragment)
        needed = ("method_label", "plan", "subgoals")
        if any(not merged.get(name) for name in needed):
            raise OdysseyRefused(f"streamed trace is missing required fields: {needed}")
        statement_hash = str(merged.get("statement_hash") or case.hash)
        _require_hash(statement_hash, "statement_hash")
        trace_body = {
            "problem_hash": case.hash,
            "statement_hash": statement_hash,
            "membership": membership,
            "director_hash": freeze.director_hash,
            "template_hash": template_hash,
            "retrieval_set": tuple(str(item) for item in merged.get("retrieval_set") or ()),
            "method_label": str(merged["method_label"]),
            "plan": tuple(str(item) for item in merged["plan"]),
            "subgoals": tuple(str(item) for item in merged["subgoals"]),
            "formal_states": tuple(str(item) for item in merged.get("formal_states") or ()),
            "actions": tuple(str(item) for item in merged.get("actions") or ()),
            "tool_calls": tuple(dict(item) for item in merged.get("tool_calls") or ()),
            "verifier_outcomes": tuple(dict(item) for item in merged.get("verifier_outcomes") or ()),
            "student_failure_class": merged.get("student_failure_class"),
        }
        trace_id = content_hash(trace_body)[:24]
        return TraceRecord(id=trace_id, **trace_body)


@dataclass(frozen=True)
class TraceShard:
    sha256: str
    trace_hashes: tuple[str, ...]
    byte_count: int
    cold_key: str | None
    sealed: bool = True


class MockColdStore:
    """Content-addressed local cold store suitable for fixture tests and recovery rehearsal."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        _require_hash(key, "cold-store key")
        if self.root.is_symlink():
            raise StorageRefused("cold-store root may not be a symlink")
        bucket = self.root / key[:2]
        if bucket.exists() and bucket.is_symlink():
            raise StorageRefused("cold-store bucket may not be a symlink")
        return bucket / key

    def put(self, payload: bytes) -> str:
        key = hashlib.sha256(payload).hexdigest()
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise StorageRefused("existing cold-store object is unsafe or hash-inconsistent")
            return key
        temp = path.with_suffix(".tmp")
        if temp.exists() and temp.is_symlink():
            raise StorageRefused("cold-store temporary path may not be a symlink")
        temp.write_bytes(payload)
        os.replace(temp, path)
        return key

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file() or path.is_symlink():
            raise StorageRefused("cold-store object is absent or unsafe")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != key:
            raise StorageRefused("cold-store object hash mismatch")
        return payload

    def verify(self, key: str) -> bool:
        self.get(key)
        return True


class TraceCompactor:
    """Verifies, removes untrainable expansion, hashes, reloads, then seals a trace shard."""

    def __init__(self, verifier: TraceVerifier | None = None) -> None:
        self.verifier = verifier or TraceVerifier()

    def seal(self, traces: Iterable[TraceRecord], *, cold_store: MockColdStore | None = None) -> TraceShard:
        assessed = [self.verifier.assess(trace) for trace in traces]
        if not assessed:
            raise StorageRefused("cannot seal an empty trace shard")
        rejected = [assessment for assessment in assessed if not assessment.trainable]
        if rejected:
            statuses = [assessment.trace.disposition.value for assessment in rejected]
            raise StorageRefused(f"only independently dispositioned traces may enter a training shard: {statuses}")
        rows = [assessment.trace.as_dict() for assessment in assessed]
        payload = canonical_bytes(rows)
        digest = hashlib.sha256(payload).hexdigest()
        # The byte payload is immediately reparsed before it can be called sealed.
        restored = json.loads(payload.decode("utf-8"))
        if canonical_bytes(restored) != payload:
            raise StorageRefused("trace shard reload is not byte-stable")
        cold_key = cold_store.put(payload) if cold_store is not None else None
        if cold_store is not None and cold_store.get(cold_key) != payload:
            raise StorageRefused("cold-store reload differs from sealed trace shard")
        return TraceShard(digest, tuple(assessment.trace.hash for assessment in assessed), len(payload), cold_key, True)


@dataclass(frozen=True)
class StoragePlan:
    system_floor: int
    active_model: int
    optimizer_or_runtime: int
    one_checkpoint: int
    one_trace_shard: int
    toolchain_and_index: int
    scratch: int

    @property
    def required_bytes(self) -> int:
        values = asdict(self).values()
        if any(int(value) < 0 for value in values):
            raise StorageRefused("storage preflight values must be non-negative")
        return sum(int(value) for value in values)


def preflight_storage(plan: StoragePlan, *, free_bytes: int | None = None, path: Path | None = None) -> dict[str, int]:
    """Enforce the master-plan free-space equation before any epoch starts."""
    available = int(free_bytes) if free_bytes is not None else shutil.disk_usage(path or Path.cwd()).free
    required = plan.required_bytes
    if available < required:
        raise StorageRefused(f"storage preflight failed: free={available} < required={required}")
    return {"free_bytes": available, "required_bytes": required, "headroom_bytes": available - required}


class Epoch(str, Enum):
    IDLE = "idle"
    DIRECTOR = "director"
    STUDENT = "student"
    REFRESH = "refresh"


@dataclass
class PhaseSeparatedScheduler:
    """Three-body law with the stronger local rule: Director and student never co-reside."""

    active: set[str] = field(default_factory=set)
    epoch: Epoch = Epoch.IDLE
    history: list[dict[str, Any]] = field(default_factory=list)

    def _enter(self, epoch: Epoch, plan: StoragePlan, *, free_bytes: int | None) -> None:
        preflight = preflight_storage(plan, free_bytes=free_bytes)
        if self.active:
            raise StorageRefused("an epoch must finish before a different body can be mounted")
        body = "director" if epoch in {Epoch.DIRECTOR, Epoch.REFRESH} else "student"
        self.active = {body}
        self.epoch = epoch
        self.history.append({"event": "enter", "epoch": epoch.value, "active": sorted(self.active), **preflight})

    def finish(self) -> None:
        if self.epoch is Epoch.IDLE:
            raise StorageRefused("no active epoch to finish")
        self.history.append({"event": "finish", "epoch": self.epoch.value, "active": sorted(self.active)})
        self.active.clear()
        self.epoch = Epoch.IDLE

    def attach_trace_shard(self, shard: TraceShard) -> None:
        if self.epoch is Epoch.IDLE or not shard.sealed:
            raise StorageRefused("only a sealed shard may be attached to an active epoch")
        if len(self.active) >= 3:
            raise StorageRefused("three-body storage law would be exceeded")
        self.active.add("trace_shard")
        if {"director", "student"}.issubset(self.active):
            raise StorageRefused("Director and student cannot co-reside")

    def run_director_epoch(
        self,
        plan: StoragePlan,
        action: Callable[[], Any],
        *,
        free_bytes: int | None = None,
    ) -> Any:
        self._enter(Epoch.DIRECTOR, plan, free_bytes=free_bytes)
        try:
            return action()
        finally:
            self.finish()

    def run_student_epoch(
        self,
        plan: StoragePlan,
        shard: TraceShard,
        action: Callable[[TraceShard], Any],
        *,
        free_bytes: int | None = None,
    ) -> Any:
        self._enter(Epoch.STUDENT, plan, free_bytes=free_bytes)
        try:
            self.attach_trace_shard(shard)
            return action(shard)
        finally:
            self.finish()

    def run_refresh_epoch(
        self,
        plan: StoragePlan,
        action: Callable[[], Any],
        *,
        free_bytes: int | None = None,
    ) -> Any:
        self._enter(Epoch.REFRESH, plan, free_bytes=free_bytes)
        try:
            return action()
        finally:
            self.finish()


@dataclass(frozen=True)
class DeltaCheckpoint:
    slot: str
    parent_hash: str | None
    sha256: str
    cold_key: str
    metadata: Mapping[str, Any]


@dataclass
class DeltaCheckpointManager:
    """Retains logical current/best/rollback pointers and cold-stores every immutable delta."""

    cold_store: MockColdStore
    pointers: dict[str, DeltaCheckpoint] = field(default_factory=dict)
    history: list[DeltaCheckpoint] = field(default_factory=list)

    def write_delta(self, slot: str, delta: Mapping[str, Any], *, parent_hash: str | None = None) -> DeltaCheckpoint:
        if slot not in {"current", "best", "rollback"}:
            raise StorageRefused("only current, best, and rollback checkpoint pointers are retained")
        if parent_hash is not None:
            _require_hash(parent_hash, "parent_hash")
        body = {"parent_hash": parent_hash, "delta": dict(delta), "fixture_only": True}
        payload = canonical_bytes(body)
        digest = hashlib.sha256(payload).hexdigest()
        key = self.cold_store.put(payload)
        checkpoint = DeltaCheckpoint(slot, parent_hash, digest, key, {"fixture_only": True})
        self.pointers[slot] = checkpoint
        self.history.append(checkpoint)
        return checkpoint

    def cold_restore(self, slot: str) -> dict[str, Any]:
        checkpoint = self.pointers.get(slot)
        if checkpoint is None:
            raise StorageRefused(f"no {slot!r} checkpoint pointer exists")
        payload = self.cold_store.get(checkpoint.cold_key)
        if hashlib.sha256(payload).hexdigest() != checkpoint.sha256:
            raise StorageRefused("cold-restored checkpoint hash differs from pointer")
        result = json.loads(payload.decode("utf-8"))
        if result.get("fixture_only") is not True:
            raise StorageRefused("checkpoint lacks fixture-only boundary")
        return result


@dataclass(frozen=True)
class StudentShard:
    source_shard: str
    sha256: str
    cold_key: str
    records: int


class StudentShardExecutor:
    """Converts a sealed trace shard into a compact auditable student input shard."""

    def __init__(self, cold_store: MockColdStore) -> None:
        self.cold_store = cold_store

    def convert(self, shard: TraceShard) -> StudentShard:
        if not shard.sealed or not shard.cold_key:
            raise StorageRefused("student conversion requires a cold-sealed trace shard")
        raw = self.cold_store.get(shard.cold_key)
        rows = json.loads(raw.decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            raise StorageRefused("trace shard does not reload to records")
        # Keep auditable structured targets, not raw prompts/KV/logits/rejected transcripts.
        compact = [
            {
                "id": row["id"],
                "problem_hash": row["problem_hash"],
                "statement_hash": row["statement_hash"],
                "membership": row["membership"],
                "method_label": row["method_label"],
                "plan": row["plan"],
                "subgoals": row["subgoals"],
                "actions": row["actions"],
                "disposition": row["disposition"],
            }
            for row in rows
        ]
        payload = canonical_bytes(compact)
        key = self.cold_store.put(payload)
        return StudentShard(shard.sha256, hashlib.sha256(payload).hexdigest(), key, len(compact))


# Future Ramanujan-proto program -------------------------------------------------
#
# This is a deliberately executable *fixture algorithm* for the future program,
# not a hidden launcher for any of its large models.  Its important property is
# physical: each teacher produces compact, verifier-dispositioned trace shards in
# a separate epoch; only then may the strict V4 Flash student be mounted.

PROTO_SCHEMA = "hawking.ramanujan.proto_shardstream.v1"
GIB = 1024**3


class ProtoModelRole(str, Enum):
    STUDENT = "STUDENT"
    TEACHER = "TEACHER"


class ProtoTeacher(str, Enum):
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"
    GLM_MATH_DIRECTOR = "glm-math-director"
    KIMI_K3 = "kimi-k3"


class TeacherPass(str, Enum):
    PRIMARY_PLAN_AND_FORMALIZE = "PRIMARY_PLAN_AND_FORMALIZE"
    METHOD_CRITIQUE_AND_REPAIR = "METHOD_CRITIQUE_AND_REPAIR"
    INDEPENDENT_ALTERNATIVE_AND_FALSIFIER = "INDEPENDENT_ALTERNATIVE_AND_FALSIFIER"


@dataclass(frozen=True)
class ProtoModelPin:
    """A source-admission target, never a claim that its weights are mounted.

    ``estimated_source_bytes`` is planning input only.  The exact source manifest
    hash is intentionally absent until an owner-approved future source admission
    supplies it.  This keeps the current scaffold honest while still pinning the
    non-negotiable Flash student identity and architecture.
    """

    model_id: str
    revision: str
    architecture: str
    role: ProtoModelRole
    total_parameters: int | None = None
    active_parameters: int | None = None
    estimated_source_bytes: int | None = None

    def validate(self) -> None:
        _safe_token(self.model_id, "model_id")
        _safe_token(self.revision, "model revision")
        _safe_token(self.architecture, "model architecture")
        if self.total_parameters is not None and self.total_parameters <= 0:
            raise OdysseyRefused("model total_parameters must be positive when declared")
        if self.active_parameters is not None and self.active_parameters <= 0:
            raise OdysseyRefused("model active_parameters must be positive when declared")
        if (
            self.total_parameters is not None
            and self.active_parameters is not None
            and self.active_parameters > self.total_parameters
        ):
            raise OdysseyRefused("active parameters cannot exceed total parameters")
        if self.estimated_source_bytes is not None and self.estimated_source_bytes <= 0:
            raise OdysseyRefused("estimated_source_bytes must be positive when declared")

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(
            {
                "model_id": self.model_id,
                "revision": self.revision,
                "architecture": self.architecture,
                "role": self.role.value,
            }
        )

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": self.architecture,
            "role": self.role.value,
            "total_parameters": self.total_parameters,
            "active_parameters": self.active_parameters,
            "estimated_source_bytes": self.estimated_source_bytes,
            "identity_hash": self.identity_hash,
            "source_admission": "REQUIRED_BEFORE_ANY_REAL_MOUNT",
        }


# The Flash revision is the existing local admission target.  The teacher
# revisions remain explicit future-admission placeholders rather than invented
# source hashes or a claim that a current source is locally available.
STRICT_V4_FLASH = ProtoModelPin(
    "deepseek-ai/DeepSeek-V4-Flash",
    "60d8d70770c6776ff598c94bb586a859a38244f1",
    "deepseek_v4",
    ProtoModelRole.STUDENT,
    total_parameters=284_000_000_000,
    active_parameters=13_000_000_000,
    estimated_source_bytes=159_609_485_896,
)
V4_PRO_TEACHER = ProtoModelPin(
    "deepseek-ai/DeepSeek-V4-Pro",
    "future-source-admission-required",
    "deepseek_v4",
    ProtoModelRole.TEACHER,
    total_parameters=1_600_000_000_000,
    active_parameters=49_000_000_000,
    estimated_source_bytes=864_700_000_000,
)
GLM_MATH_TEACHER = ProtoModelPin(
    "zai-org/GLM-5.2",
    "future-math-director-admission-required",
    "glm_moe_dsa",
    ProtoModelRole.TEACHER,
)
KIMI_K3_TEACHER = ProtoModelPin(
    "moonshotai/Kimi-K3",
    "future-source-admission-required",
    "kimi_k3",
    ProtoModelRole.TEACHER,
    total_parameters=2_800_000_000_000,
    active_parameters=104_000_000_000,
    # 2.8T MXFP4 weights are roughly 1.4 TB before scales/indexes; this is a
    # storage-planning estimate, not a downloaded-source claim.
    estimated_source_bytes=1_400_000_000_000,
)


@dataclass(frozen=True)
class StrictV4Student:
    """The future proto base: Flash only, with route-aware rollout learning.

    The existing independent-layer cascade finding forbids the old local-fit
    objective.  The final model must preserve downstream routing over rollouts;
    the trace program therefore distils behavior/tool policy, not teacher weight
    slices, hidden states, or unrestricted prose reasoning.
    """

    model: ProtoModelPin = STRICT_V4_FLASH
    objective: str = "route_aware_rollout_structured_trace_distillation"
    adapter_mode: str = "reversible_math_route_adapters"
    preserve_next_router: bool = True
    independent_layerwise_distillation: bool = False

    def validate(self) -> None:
        self.model.validate()
        if self.model != STRICT_V4_FLASH or self.model.role is not ProtoModelRole.STUDENT:
            raise OdysseyRefused("Ramanujan-proto requires the exact pinned DeepSeek V4 Flash student")
        if self.model.architecture != "deepseek_v4":
            raise OdysseyRefused("Ramanujan-proto student must remain on the deepseek_v4 architecture")
        if not self.preserve_next_router or self.independent_layerwise_distillation:
            raise OdysseyRefused("the Flash student must use a route-aware rollout objective, never independent layers")
        _safe_token(self.objective, "student objective")
        _safe_token(self.adapter_mode, "student adapter_mode")

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(asdict(self))


@dataclass(frozen=True)
class TeacherAssignment:
    teacher: ProtoTeacher
    model: ProtoModelPin
    pass_kind: TeacherPass
    purpose: str
    depends_on: ProtoTeacher | None = None

    def validate(self) -> None:
        self.model.validate()
        if self.model.role is not ProtoModelRole.TEACHER:
            raise OdysseyRefused("teacher assignment may not mount a student body")
        _safe_token(self.purpose, "teacher purpose")
        expected = {
            ProtoTeacher.DEEPSEEK_V4_PRO: (V4_PRO_TEACHER, TeacherPass.PRIMARY_PLAN_AND_FORMALIZE, None),
            ProtoTeacher.GLM_MATH_DIRECTOR: (
                GLM_MATH_TEACHER,
                TeacherPass.METHOD_CRITIQUE_AND_REPAIR,
                ProtoTeacher.DEEPSEEK_V4_PRO,
            ),
            ProtoTeacher.KIMI_K3: (KIMI_K3_TEACHER, TeacherPass.INDEPENDENT_ALTERNATIVE_AND_FALSIFIER, None),
        }
        wanted_model, wanted_pass, wanted_dependency = expected[self.teacher]
        if self.model != wanted_model or self.pass_kind is not wanted_pass or self.depends_on is not wanted_dependency:
            raise OdysseyRefused(f"{self.teacher.value} assignment violates the fixed Ramanujan teacher mix")


@dataclass(frozen=True)
class TeacherMix:
    """A data-level ensemble; model weights are never averaged or merged."""

    assignments: tuple[TeacherAssignment, ...]

    @classmethod
    def ramanujan_default(cls) -> "TeacherMix":
        return cls(
            (
                TeacherAssignment(
                    ProtoTeacher.DEEPSEEK_V4_PRO,
                    V4_PRO_TEACHER,
                    TeacherPass.PRIMARY_PLAN_AND_FORMALIZE,
                    "primary structured decomposition, formalization, and proof-state proposal",
                ),
                TeacherAssignment(
                    ProtoTeacher.GLM_MATH_DIRECTOR,
                    GLM_MATH_TEACHER,
                    TeacherPass.METHOD_CRITIQUE_AND_REPAIR,
                    "mathematical method critique, repair, and scope checks over compact V4 candidates",
                    depends_on=ProtoTeacher.DEEPSEEK_V4_PRO,
                ),
                TeacherAssignment(
                    ProtoTeacher.KIMI_K3,
                    KIMI_K3_TEACHER,
                    TeacherPass.INDEPENDENT_ALTERNATIVE_AND_FALSIFIER,
                    "independent alternative methods, discriminating experiments, and falsifiers",
                ),
            )
        )

    def validate(self) -> None:
        expected = tuple(ProtoTeacher)
        if tuple(item.teacher for item in self.assignments) != expected:
            raise OdysseyRefused("teacher mix must be V4 Pro -> GLM math -> K3 in three sequential passes")
        for item in self.assignments:
            item.validate()

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(self.assignments)


@dataclass(frozen=True)
class TeacherJob:
    round_id: str
    problem_hash: str
    assignment: TeacherAssignment
    prior_teacher: ProtoTeacher | None = None
    prior_trace_hash: str | None = None

    def validate(self) -> None:
        _safe_token(self.round_id, "round_id")
        _require_hash(self.problem_hash, "teacher job problem_hash")
        self.assignment.validate()
        if self.prior_teacher is not self.assignment.depends_on:
            raise OdysseyRefused("teacher job dependency differs from its declared mix policy")
        if self.prior_teacher is None and self.prior_trace_hash is not None:
            raise OdysseyRefused("an independent teacher may not receive a prior-teacher trace")
        if self.prior_trace_hash is not None:
            _require_hash(self.prior_trace_hash, "prior_trace_hash")

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(
            {
                "round_id": self.round_id,
                "problem_hash": self.problem_hash,
                "teacher": self.assignment.teacher.value,
                "pass": self.assignment.pass_kind.value,
                "prior_teacher": None if self.prior_teacher is None else self.prior_teacher.value,
                "prior_trace_hash": self.prior_trace_hash,
            }
        )


@dataclass(frozen=True)
class TeacherContribution:
    job_hash: str
    teacher: ProtoTeacher
    source_identity_hash: str
    fragment_hash: str
    trace_hash: str
    prior_trace_hash: str | None
    disposition: TraceDisposition
    accepted: bool
    reasons: tuple[str, ...] = ()

    def validate(self) -> None:
        for name in ("job_hash", "source_identity_hash", "fragment_hash", "trace_hash"):
            _require_hash(getattr(self, name), name)
        if self.prior_trace_hash is not None:
            _require_hash(self.prior_trace_hash, "prior_trace_hash")


@dataclass(frozen=True)
class RamanujanProtoProgram:
    """Frozen design identity for the future Flash-based proto family."""

    student: StrictV4Student = StrictV4Student()
    teachers: TeacherMix = field(default_factory=TeacherMix.ramanujan_default)
    trace_schema: str = "statement-hash-structured-plan-subgoals-formal-states-actions-tools-verifier-outcomes"
    no_raw_logits_or_kv: bool = True
    no_teacher_weight_merge: bool = True
    research_authorized: bool = False

    def validate(self) -> None:
        self.student.validate()
        self.teachers.validate()
        if not self.no_raw_logits_or_kv or not self.no_teacher_weight_merge:
            raise OdysseyRefused("proto program must retain only compact structured targets and never merge teacher weights")
        if self.research_authorized or RESEARCH_AUTHORIZED:
            raise OdysseyRefused("this proto program is a future plan, not research authorization")
        _safe_token(self.trace_schema, "trace_schema")

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(asdict(self))

    def teacher_jobs(self, round_id: str, problem_hashes: Iterable[str]) -> tuple[TeacherJob, ...]:
        self.validate()
        _safe_token(round_id, "round_id")
        jobs: list[TeacherJob] = []
        for problem_hash in problem_hashes:
            _require_hash(problem_hash, "problem_hash")
            for assignment in self.teachers.assignments:
                job = TeacherJob(round_id, problem_hash, assignment, assignment.depends_on)
                job.validate()
                jobs.append(job)
        return tuple(jobs)

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PROTO_SCHEMA,
            "status": "FUTURE_UNAUTHORIZED_PROTO_PROGRAM",
            "program_hash": self.identity_hash,
            "student": {
                **self.student.model.manifest(),
                "objective": self.student.objective,
                "adapter_mode": self.student.adapter_mode,
                "preserve_next_router": True,
                "independent_layerwise_distillation": False,
            },
            "teacher_mix": [
                {
                    "teacher": assignment.teacher.value,
                    "model": assignment.model.manifest(),
                    "pass": assignment.pass_kind.value,
                    "purpose": assignment.purpose,
                    "depends_on": None if assignment.depends_on is None else assignment.depends_on.value,
                }
                for assignment in self.teachers.assignments
            ],
            "trace_policy": {
                "schema": self.trace_schema,
                "keep": ["statement_hash", "plan", "subgoals", "formal_states", "actions", "tool_calls", "verifier_outcomes"],
                "discard": ["raw_prompt", "raw_response", "reasoning_content", "hidden_states", "kv_cache", "full_logits"],
                "teacher_weight_merge": False,
            },
            "scheduler": "V4 primary -> verifier-dispositioned compact trace reference -> GLM critique; K3 independent -> same-statement arbitration -> sealed trace shard -> Flash student epoch",
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "production_authority": False,
        }


_TEACHER_FRAGMENT_FIELDS = frozenset(
    {
        "method_label",
        "statement_hash",
        "plan",
        "subgoals",
        "formal_states",
        "actions",
        "tool_calls",
        "verifier_outcomes",
        "retrieval_set",
        "student_failure_class",
    }
)


@dataclass(frozen=True)
class ProtoDistillationRound:
    """Receipt for one fixture-only sequential teacher/student round."""

    program_hash: str
    round_id: str
    trace_shard: TraceShard
    student_shard: StudentShard
    checkpoint: DeltaCheckpoint
    contributions: tuple[TeacherContribution, ...]
    arbitrations: tuple[TeacherArbitration, ...]
    mix_receipt_hash: str
    mix_receipt_cold_key: str
    scheduler_history: tuple[Mapping[str, Any], ...]
    fixture_only: bool = True
    research_authorized: bool = False

    def teacher_triangulation_evidence(self) -> PreservationEvidence:
        """Collapse per-problem arbitration receipts into the Condense axis."""
        if not self.arbitrations:
            raise OdysseyRefused("a proto round without teacher arbitrations cannot satisfy the Condense teacher gate")
        return PreservationEvidence(
            PreservationAxis.TEACHER_TRIANGULATION,
            baseline=1.0,
            candidate=1.0 if all(item.triangulated for item in self.arbitrations) else 0.0,
            evidence_hash=content_hash([item.receipt_hash for item in self.arbitrations]),
            independent=True,
            note="aggregate of all same-statement three-teacher arbitration receipts in this round",
        )


class ShardStreamDistiller:
    """Phase-separated future algorithm, executable only with fixture callbacks.

    Real mounts must be implemented later behind source admission, operator, and
    authorization gates.  This class nevertheless fixes the algorithm now:
    three serialized teacher passes per problem; each emits only structured
    targets; independent verifiers disposition them; immutable shards feed a
    route-aware Flash student after all teacher windows are gone.
    """

    def __init__(
        self,
        program: RamanujanProtoProgram | None = None,
        *,
        verifier: TraceVerifier | None = None,
    ) -> None:
        self.program = program or RamanujanProtoProgram()
        self.program.validate()
        self.verifier = verifier or TraceVerifier()
        self.arbiter = TeacherArbiter(self.verifier)

    @staticmethod
    def _trace_from_fragment(job: TeacherJob, template_hash: str, fragment: Mapping[str, Any]) -> TraceRecord:
        job.validate()
        _require_hash(template_hash, "template_hash")
        if not isinstance(fragment, Mapping):
            raise OdysseyRefused("teacher output must be a structured mapping")
        unknown = set(fragment) - _TEACHER_FRAGMENT_FIELDS
        if unknown:
            raise OdysseyRefused(
                "teacher output contains disallowed raw or opaque fields: " + ", ".join(sorted(map(str, unknown)))
            )
        if not all(fragment.get(name) for name in ("statement_hash", "method_label", "plan", "subgoals")):
            raise OdysseyRefused("teacher output needs statement_hash, method_label, plan, and subgoals")
        statement_hash = str(fragment["statement_hash"])
        _require_hash(statement_hash, "teacher statement_hash")
        body = {
            "problem_hash": job.problem_hash,
            "statement_hash": statement_hash,
            "membership": "future-ramanujan-proto-train",
            "director_hash": job.assignment.model.identity_hash,
            "template_hash": template_hash,
            "retrieval_set": tuple(str(item) for item in fragment.get("retrieval_set") or ()),
            "method_label": str(fragment["method_label"]),
            "plan": tuple(str(item) for item in fragment["plan"]),
            "subgoals": tuple(str(item) for item in fragment["subgoals"]),
            "formal_states": tuple(str(item) for item in fragment.get("formal_states") or ()),
            "actions": tuple(str(item) for item in fragment.get("actions") or ()),
            "tool_calls": tuple(dict(item) for item in fragment.get("tool_calls") or ()),
            "verifier_outcomes": tuple(dict(item) for item in fragment.get("verifier_outcomes") or ()),
            "student_failure_class": fragment.get("student_failure_class"),
        }
        return TraceRecord(id=content_hash({"job": job.identity_hash, "body": body})[:24], **body)

    def run_fixture_round(
        self,
        *,
        round_id: str,
        problem_hashes: Iterable[str],
        template_hash: str,
        storage_plan: StoragePlan,
        cold_store: MockColdStore,
        checkpoint_manager: DeltaCheckpointManager,
        teacher_output: Callable[[TeacherJob], Mapping[str, Any]],
        student_train: Callable[[StrictV4Student, StudentShard], Mapping[str, Any]],
        free_bytes: int | None = None,
    ) -> ProtoDistillationRound:
        """Exercise the exact future scheduling protocol using fixture callbacks.

        ``teacher_output`` and ``student_train`` are deliberately injected: this
        method has no network/client/model-loading capability and cannot turn a
        future source declaration into a live teacher call.
        """
        self.program.validate()
        _safe_token(round_id, "round_id")
        _require_hash(template_hash, "template_hash")
        jobs = self.program.teacher_jobs(round_id, problem_hashes)
        if not jobs:
            raise StorageRefused("a proto distillation round needs at least one problem")
        scheduler = PhaseSeparatedScheduler()
        submitted: dict[str, dict[ProtoTeacher, tuple[TeacherJob, Mapping[str, Any], TraceAssessment]]] = {}
        accepted: list[TraceRecord] = []
        contributions: list[TeacherContribution] = []
        for job in jobs:
            if job.prior_teacher is not None:
                prior = submitted.get(job.problem_hash, {}).get(job.prior_teacher)
                if prior is None or not prior[2].trainable:
                    raise StorageRefused("GLM critique requires a verifier-dispositioned compact V4 primary trace")
                job = replace(job, prior_trace_hash=prior[2].trace.hash)
                job.validate()
            fragment = scheduler.run_director_epoch(
                storage_plan,
                lambda job=job: dict(teacher_output(job)),
                free_bytes=free_bytes,
            )
            trace = self._trace_from_fragment(job, template_hash, fragment)
            assessment = self.verifier.assess(trace)
            submitted.setdefault(job.problem_hash, {})[job.assignment.teacher] = (job, fragment, assessment)
        arbitrations: list[TeacherArbitration] = []
        for problem_hash, by_teacher in submitted.items():
            candidates = tuple(
                TeacherTraceCandidate(teacher, row[2].trace, row[2].trace.statement_hash)
                for teacher, row in by_teacher.items()
            )
            arbitration = self.arbiter.adjudicate(candidates)
            arbitrations.append(arbitration)
            trainable_teachers = {row.teacher for row in arbitration.accepted} if arbitration.triangulated else set()
            for teacher, (job, fragment, assessment) in by_teacher.items():
                admitted = teacher in trainable_teachers
                reasons = assessment.reasons + (() if admitted else ("complete three-teacher statement arbitration required",))
                contribution = TeacherContribution(
                    job.identity_hash,
                    teacher,
                    job.assignment.model.identity_hash,
                    content_hash(fragment),
                    assessment.trace.hash,
                    job.prior_trace_hash,
                    assessment.trace.disposition,
                    admitted,
                    reasons,
                )
                contribution.validate()
                contributions.append(contribution)
                if admitted:
                    accepted.append(assessment.trace)
        if not accepted:
            raise StorageRefused("no complete three-teacher statement set survived independent verifier disposition")
        trace_shard = TraceCompactor(self.verifier).seal(accepted, cold_store=cold_store)

        def train(sealed: TraceShard) -> tuple[StudentShard, DeltaCheckpoint]:
            student_shard = StudentShardExecutor(cold_store).convert(sealed)
            delta = dict(student_train(self.program.student, student_shard))
            checkpoint = checkpoint_manager.write_delta(
                "current",
                {
                    "program_hash": self.program.identity_hash,
                    "student_hash": self.program.student.identity_hash,
                    "student_shard": student_shard.sha256,
                    "route_aware": True,
                    "teacher_weight_merge": False,
                    "delta": delta,
                },
            )
            return student_shard, checkpoint

        student_shard, checkpoint = scheduler.run_student_epoch(
            storage_plan, trace_shard, train, free_bytes=free_bytes
        )
        mix_receipt = {
            "schema": PROTO_SCHEMA,
            "status": "FIXTURE_SHARD_STREAM_DISTILLATION_ROUND",
            "program_hash": self.program.identity_hash,
            "round_id": round_id,
            "trace_shard": trace_shard.sha256,
            "student_shard": student_shard.sha256,
            "checkpoint": checkpoint.sha256,
            "contributions": [
                {
                    "job_hash": item.job_hash,
                    "teacher": item.teacher.value,
                    "source_identity_hash": item.source_identity_hash,
                    "fragment_hash": item.fragment_hash,
                    "trace_hash": item.trace_hash,
                    "prior_trace_hash": item.prior_trace_hash,
                    "disposition": item.disposition.value,
                    "accepted": item.accepted,
                    "reasons": list(item.reasons),
                }
                for item in contributions
            ],
            "teacher_arbitrations": [
                {
                    "statement_hash": item.statement_hash,
                    "triangulated": item.triangulated,
                    "receipt_hash": item.receipt_hash,
                    "accepted": [row.teacher.value for row in item.accepted],
                    "rejected": [row.teacher.value for row in item.rejected],
                }
                for item in arbitrations
            ],
            "teacher_weight_merge": False,
            "fixture_only": True,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        }
        payload = canonical_bytes(mix_receipt)
        key = cold_store.put(payload)
        return ProtoDistillationRound(
            self.program.identity_hash,
            round_id,
            trace_shard,
            student_shard,
            checkpoint,
            tuple(contributions),
            tuple(arbitrations),
            hashlib.sha256(payload).hexdigest(),
            key,
            tuple(dict(row) for row in scheduler.history),
        )


@dataclass(frozen=True)
class ProtoFootprint:
    """Conservative future 1-BPW sizing model for the full strict Flash proto.

    It models a compact full-model artifact, not the failed independent-layer
    functional student.  Context/KV and workspace values are explicit budgets so
    operators can revise them after a real DeepSeek V4 Gravity runtime exists.
    """

    total_parameters: int = STRICT_V4_FLASH.total_parameters or 0
    target_bpw: float = 1.0
    artifact_overhead_fraction: float = 0.20
    runtime_workspace_bytes: int = 8 * GIB
    kv_cache_budget_bytes: int = 16 * GIB
    safety_margin_bytes: int = 12 * GIB

    def validate(self) -> None:
        if self.total_parameters <= 0 or self.target_bpw <= 0:
            raise StorageRefused("footprint requires positive parameter count and BPW")
        if not 0 <= self.artifact_overhead_fraction <= 1:
            raise StorageRefused("artifact overhead fraction must be in [0, 1]")
        if min(self.runtime_workspace_bytes, self.kv_cache_budget_bytes, self.safety_margin_bytes) < 0:
            raise StorageRefused("footprint budgets must be non-negative")

    @property
    def weight_payload_bytes(self) -> int:
        self.validate()
        return math.ceil(self.total_parameters * self.target_bpw / 8)

    @property
    def artifact_bytes(self) -> int:
        return math.ceil(self.weight_payload_bytes * (1 + self.artifact_overhead_fraction))

    @property
    def warm_ram_bytes(self) -> int:
        return self.artifact_bytes + self.runtime_workspace_bytes + self.kv_cache_budget_bytes

    @property
    def recommended_ram_bytes(self) -> int:
        return self.warm_ram_bytes + self.safety_margin_bytes

    def estimate(self, *, host_ram_bytes: int, free_disk_bytes: int) -> dict[str, Any]:
        self.validate()
        if host_ram_bytes < 0 or free_disk_bytes < 0:
            raise StorageRefused("host capacity inputs must be non-negative")
        return {
            "schema": PROTO_SCHEMA,
            "scope": "PREDICTION_NOT_MEASUREMENT",
            "target_bpw": self.target_bpw,
            "total_parameters": self.total_parameters,
            "weight_payload_bytes": self.weight_payload_bytes,
            "weight_payload_gib": round(self.weight_payload_bytes / GIB, 2),
            "artifact_bytes": self.artifact_bytes,
            "artifact_gib": round(self.artifact_bytes / GIB, 2),
            "warm_ram_bytes": self.warm_ram_bytes,
            "warm_ram_gib": round(self.warm_ram_bytes / GIB, 2),
            "recommended_ram_bytes": self.recommended_ram_bytes,
            "recommended_ram_gib": round(self.recommended_ram_bytes / GIB, 2),
            "host_ram_gib": round(host_ram_bytes / GIB, 2),
            "free_disk_gib": round(free_disk_bytes / GIB, 2),
            "fits_one_resident_body": host_ram_bytes >= self.recommended_ram_bytes,
            "fits_artifact_on_disk": free_disk_bytes >= self.artifact_bytes + self.safety_margin_bytes,
            "max_parallel_model_bodies": host_ram_bytes // self.recommended_ram_bytes,
            "parallelism_note": "Parallel requests may share one resident body; this is not permission to load multiple model bodies.",
        }


@dataclass(frozen=True)
class ProtoGravityRenderPlan:
    """A render contract that refuses to impersonate an emitted .gravity artifact."""

    program_hash: str
    checkpoint_hash: str | None
    artifact_name: str
    footprint: Mapping[str, Any]
    required_gates: tuple[str, ...]
    deferred_runtime_gates: tuple[str, ...] = ("hawking_measured_tps_receipt",)
    status: str = "FUTURE_RENDER_PLAN_ONLY"
    production_authority: bool = False


class ProtoGravityRenderer:
    """Produces the future Gravity handoff contract, never a fake model file."""

    REQUIRED_GATES = (
        "pinned_deepseek_v4_source_admission",
        "deepseek_v4_gravity_adapter_and_dtype_parity",
        "condense_capability_receipt",
        "teacher_arbitration_receipt",
        "route_aware_rollout_retention",
        "lean_and_exact_math_retention",
        "checkpoint_tournament_winner",
        "owner_authorized_render_window",
    )

    def plan(
        self,
        program: RamanujanProtoProgram,
        *,
        checkpoint_hash: str | None = None,
        target_bpw: float = 1.0,
        host_ram_bytes: int = 96 * GIB,
        free_disk_bytes: int = 342 * GIB,
    ) -> ProtoGravityRenderPlan:
        program.validate()
        if checkpoint_hash is not None:
            _require_hash(checkpoint_hash, "proto checkpoint_hash")
        footprint = ProtoFootprint(target_bpw=target_bpw).estimate(
            host_ram_bytes=host_ram_bytes, free_disk_bytes=free_disk_bytes
        )
        return ProtoGravityRenderPlan(
            program.identity_hash,
            checkpoint_hash,
            f"ramanujan-proto-v4flash-{target_bpw:g}bpw.gravity",
            footprint,
            self.REQUIRED_GATES,
        )


class PreservationAxis(str, Enum):
    """Capabilities that must survive a Condense rung before further reduction."""

    ROUTER_TOP1 = "router_top1_agreement"
    ROUTE_AWARE_ROLLOUT = "route_aware_rollout"
    STATEMENT_FIDELITY = "statement_fidelity"
    LEAN_REPLAY = "lean_replay"
    EXACT_REPRODUCTION = "exact_reproduction"
    PROOF_REPAIR = "proof_repair"
    FALSE_LEMMA_REJECTION = "false_lemma_rejection"
    PERTURBED_TRANSFER = "perturbed_transfer"
    TEACHER_TRIANGULATION = "teacher_triangulation"
    ARTIFACT_RELOAD = "artifact_reload"
    RUNTIME_TPS = "runtime_tps"


@dataclass(frozen=True)
class PreservationRequirement:
    """A normalized [0, 1] capability gate, deliberately separate from loss."""

    axis: PreservationAxis
    minimum_candidate: float
    minimum_retention: float
    independent: bool = True

    def validate(self) -> None:
        for value, name in ((self.minimum_candidate, "minimum_candidate"), (self.minimum_retention, "minimum_retention")):
            if not 0 <= value <= 1:
                raise OdysseyRefused(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PreservationEvidence:
    """One measured parent-vs-candidate capability observation.

    Scores are intentionally normalized by the future evaluator.  The scaffold
    does not pretend that a model loss, a teacher opinion, or an unpinned public
    benchmark is evidence of capability retention.
    """

    axis: PreservationAxis
    baseline: float
    candidate: float
    evidence_hash: str
    independent: bool = True
    note: str = ""

    def validate(self) -> None:
        for value, name in ((self.baseline, "baseline"), (self.candidate, "candidate")):
            if not 0 <= value <= 1:
                raise OdysseyRefused(f"{name} must be a normalized score in [0, 1]")
        _require_hash(self.evidence_hash, "preservation evidence_hash")

    @property
    def retention(self) -> float:
        self.validate()
        # A zero parent score has no capability to preserve; the candidate must
        # still independently clear the requirement's absolute floor.
        return 1.0 if self.baseline == 0 else self.candidate / self.baseline


@dataclass(frozen=True)
class CondenseRungContext:
    """Immutable parent/candidate binding for one reduction step.

    A good score from a prior candidate cannot be carried forward: every
    decision names the exact parent state, candidate state, and frozen evaluator
    suite that produced it.
    """

    parent_bpw: float
    candidate_bpw: float
    parent_state_hash: str
    candidate_state_hash: str
    evaluator_suite_hash: str

    def validate(self) -> None:
        if self.parent_bpw <= self.candidate_bpw or self.candidate_bpw <= 0:
            raise OdysseyRefused("a Condense rung must reduce to a positive lower BPW")
        for name in ("parent_state_hash", "candidate_state_hash", "evaluator_suite_hash"):
            _require_hash(getattr(self, name), name)
        if self.parent_state_hash == self.candidate_state_hash:
            raise OdysseyRefused("a Condense candidate must not reuse the parent-state hash")

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(asdict(self))


@dataclass(frozen=True)
class CondenseCapabilityDecision:
    status: str  # PASS | REFUSED
    passed: tuple[PreservationAxis, ...]
    failed: tuple[PreservationAxis, ...]
    deferred: tuple[PreservationAxis, ...]
    rung_context_hash: str
    receipt_hash: str
    fixture_only: bool = True
    research_authorized: bool = False

    @property
    def promotable(self) -> bool:
        return self.status == "PASS" and not self.failed


class MathPreservationContract:
    """Fail-closed Condense gate for the six material capability risks.

    TPS deliberately stays outside Condense promotion: it is a Hawking runtime
    receipt and is deferred rather than simulated.  Every other axis must have a
    fresh, independently attributable measurement at every quantization rung.
    """

    DEFAULT_REQUIREMENTS = (
        PreservationRequirement(PreservationAxis.ROUTER_TOP1, 0.995, 0.995),
        PreservationRequirement(PreservationAxis.ROUTE_AWARE_ROLLOUT, 0.98, 0.98),
        PreservationRequirement(PreservationAxis.STATEMENT_FIDELITY, 1.0, 1.0),
        PreservationRequirement(PreservationAxis.LEAN_REPLAY, 0.99, 0.99),
        PreservationRequirement(PreservationAxis.EXACT_REPRODUCTION, 0.99, 0.99),
        PreservationRequirement(PreservationAxis.PROOF_REPAIR, 0.95, 0.95),
        PreservationRequirement(PreservationAxis.FALSE_LEMMA_REJECTION, 0.99, 0.99),
        PreservationRequirement(PreservationAxis.PERTURBED_TRANSFER, 0.95, 0.95),
        PreservationRequirement(PreservationAxis.TEACHER_TRIANGULATION, 1.0, 1.0),
        PreservationRequirement(PreservationAxis.ARTIFACT_RELOAD, 1.0, 1.0),
    )

    def __init__(self, requirements: Sequence[PreservationRequirement] | None = None) -> None:
        self.requirements = tuple(requirements or self.DEFAULT_REQUIREMENTS)
        if not self.requirements:
            raise OdysseyRefused("Condense needs at least one capability-preservation requirement")
        axes = [item.axis for item in self.requirements]
        if len(axes) != len(set(axes)) or PreservationAxis.RUNTIME_TPS in axes:
            raise OdysseyRefused("Condense requirements must be unique and may not fake a TPS receipt")
        for item in self.requirements:
            item.validate()

    def assess(
        self,
        context: CondenseRungContext,
        evidence: Iterable[PreservationEvidence],
    ) -> CondenseCapabilityDecision:
        context.validate()
        rows = tuple(evidence)
        by_axis: dict[PreservationAxis, PreservationEvidence] = {}
        for row in rows:
            row.validate()
            if row.axis in by_axis:
                raise OdysseyRefused(f"duplicate preservation evidence for {row.axis.value}")
            by_axis[row.axis] = row
        expected_axes = {requirement.axis for requirement in self.requirements}
        unexpected = set(by_axis) - expected_axes
        if unexpected:
            labels = ", ".join(sorted(axis.value for axis in unexpected))
            raise OdysseyRefused(f"unexpected Condense evidence ({labels}); TPS remains a Hawking runtime receipt")
        passed: list[PreservationAxis] = []
        failed: list[PreservationAxis] = []
        receipt_rows: list[dict[str, Any]] = []
        for requirement in self.requirements:
            row = by_axis.get(requirement.axis)
            ok = bool(
                row is not None
                and row.candidate >= requirement.minimum_candidate
                and row.retention >= requirement.minimum_retention
                and (not requirement.independent or row.independent)
            )
            (passed if ok else failed).append(requirement.axis)
            receipt_rows.append(
                {
                    "axis": requirement.axis.value,
                    "passed": ok,
                    "minimum_candidate": requirement.minimum_candidate,
                    "minimum_retention": requirement.minimum_retention,
                    "evidence_hash": None if row is None else row.evidence_hash,
                }
            )
        status = "PASS" if not failed else "REFUSED"
        return CondenseCapabilityDecision(
            status,
            tuple(passed),
            tuple(failed),
            (PreservationAxis.RUNTIME_TPS,),
            context.identity_hash,
            content_hash(
                {
                    "schema": PROTO_SCHEMA,
                    "status": status,
                    "rung_context": context.identity_hash,
                    "rows": receipt_rows,
                }
            ),
        )


@dataclass(frozen=True)
class ProtectedSurface:
    """A component class that cannot be casually labelled non-math and removed."""

    name: str
    policy: str  # SOURCE_PRECISION_LOCKED | EVIDENCE_DRIVEN_ALLOCATION
    required_axes: tuple[PreservationAxis, ...]

    def validate(self) -> None:
        _safe_token(self.name, "protected surface")
        if self.policy not in {"SOURCE_PRECISION_LOCKED", "EVIDENCE_DRIVEN_ALLOCATION"}:
            raise OdysseyRefused("protected surface policy is invalid")
        if not self.required_axes:
            raise OdysseyRefused("protected surface needs measured preservation axes")


@dataclass(frozen=True)
class QuantizationRung:
    bpw: float
    name: str

    def validate(self) -> None:
        if self.bpw <= 0:
            raise OdysseyRefused("quantization BPW must be positive")
        _safe_token(self.name, "quantization rung")


@dataclass(frozen=True)
class QuantizationLadder:
    """Progressive Condense schedule; jumping directly to 1 BPW is forbidden."""

    rungs: tuple[QuantizationRung, ...] = (
        QuantizationRung(4.0, "source-adjacent"),
        QuantizationRung(3.0, "three-bpw"),
        QuantizationRung(2.0, "two-bpw"),
        QuantizationRung(1.5, "one-point-five-bpw"),
        QuantizationRung(1.25, "one-point-two-five-bpw"),
        QuantizationRung(1.0, "one-bpw-target"),
    )

    def validate(self) -> None:
        if len(self.rungs) < 2 or self.rungs[-1].bpw != 1.0:
            raise OdysseyRefused("quantization ladder must descend to the explicit 1-BPW target")
        values = [item.bpw for item in self.rungs]
        if values != sorted(values, reverse=True) or len(values) != len(set(values)):
            raise OdysseyRefused("quantization ladder BPW values must be strictly descending")
        for item in self.rungs:
            item.validate()

    def next_rung(self, context: CondenseRungContext, decision: CondenseCapabilityDecision) -> QuantizationRung:
        self.validate()
        if not decision.promotable:
            raise OdysseyRefused("a failed math-preservation contract blocks the next Condense rung")
        context.validate()
        if decision.rung_context_hash != context.identity_hash:
            raise OdysseyRefused("Condense decision does not bind the parent/candidate rung being promoted")
        for index, rung in enumerate(self.rungs[:-1]):
            if rung.bpw == context.parent_bpw:
                next_rung = self.rungs[index + 1]
                if next_rung.bpw != context.candidate_bpw:
                    raise OdysseyRefused("Condense may advance only one declared BPW rung at a time")
                return next_rung
        raise OdysseyRefused("Condense parent BPW is not an admitted quantization rung")


@dataclass(frozen=True)
class TeacherTraceCandidate:
    """An independently attributed candidate; different methods may coexist."""

    teacher: ProtoTeacher
    trace: TraceRecord
    statement_hash: str

    def validate(self) -> None:
        _require_hash(self.statement_hash, "teacher candidate statement_hash")
        expected = {
            ProtoTeacher.DEEPSEEK_V4_PRO: V4_PRO_TEACHER,
            ProtoTeacher.GLM_MATH_DIRECTOR: GLM_MATH_TEACHER,
            ProtoTeacher.KIMI_K3: KIMI_K3_TEACHER,
        }[self.teacher]
        if self.trace.director_hash != expected.identity_hash:
            raise OdysseyRefused("teacher candidate does not identify its declared pinned teacher")


@dataclass(frozen=True)
class TeacherArbitration:
    accepted: tuple[TeacherTraceCandidate, ...]
    rejected: tuple[TeacherTraceCandidate, ...]
    statement_hash: str
    triangulated: bool
    receipt_hash: str

    def capability_evidence(self) -> PreservationEvidence:
        """Bind the three-teacher disposition receipt into the Condense gate."""
        return PreservationEvidence(
            PreservationAxis.TEACHER_TRIANGULATION,
            baseline=1.0,
            candidate=1.0 if self.triangulated else 0.0,
            evidence_hash=self.receipt_hash,
            independent=True,
            note="all fixed teachers supplied verifier-dispositioned variants for one statement",
        )


class TeacherArbiter:
    """Keeps a three-teacher mix from becoming an untraceable blended opinion."""

    def __init__(self, verifier: TraceVerifier | None = None) -> None:
        self.verifier = verifier or TraceVerifier()

    def adjudicate(self, candidates: Iterable[TeacherTraceCandidate]) -> TeacherArbitration:
        rows = tuple(candidates)
        if {row.teacher for row in rows} != set(ProtoTeacher) or len(rows) != len(ProtoTeacher):
            raise OdysseyRefused("teacher arbitration requires exactly one V4 Pro, GLM Math, and K3 candidate")
        for row in rows:
            row.validate()
        statement_hashes = {row.statement_hash for row in rows}
        if len(statement_hashes) != 1:
            raise OdysseyRefused("teacher disagreement on the formalized statement blocks distillation")
        accepted: list[TeacherTraceCandidate] = []
        rejected: list[TeacherTraceCandidate] = []
        for row in rows:
            assessed = self.verifier.assess(row.trace)
            target = TeacherTraceCandidate(row.teacher, assessed.trace, row.statement_hash)
            (accepted if assessed.trainable else rejected).append(target)
        return TeacherArbitration(
            tuple(accepted),
            tuple(rejected),
            next(iter(statement_hashes)),
            len(accepted) == len(ProtoTeacher),
            content_hash(
                {
                    "statement_hash": next(iter(statement_hashes)),
                    "accepted": [(row.teacher.value, row.trace.hash) for row in accepted],
                    "rejected": [(row.teacher.value, row.trace.hash) for row in rejected],
                }
            ),
        )


@dataclass(frozen=True)
class RamanujanCondenseSpec:
    """Capability-first Condense contract for the future strict Flash proto."""

    program: RamanujanProtoProgram = field(default_factory=RamanujanProtoProgram)
    protected_surfaces: tuple[ProtectedSurface, ...] = (
        ProtectedSurface(
            "routers-and-route-scores",
            "SOURCE_PRECISION_LOCKED",
            (PreservationAxis.ROUTER_TOP1, PreservationAxis.ROUTE_AWARE_ROLLOUT),
        ),
        ProtectedSurface(
            "attention-context-and-shared-path",
            "SOURCE_PRECISION_LOCKED",
            (PreservationAxis.ROUTE_AWARE_ROLLOUT, PreservationAxis.PERTURBED_TRANSFER),
        ),
        ProtectedSurface(
            "embedding-norm-lmhead-and-tool-grammar",
            "SOURCE_PRECISION_LOCKED",
            (
                PreservationAxis.STATEMENT_FIDELITY,
                PreservationAxis.LEAN_REPLAY,
                PreservationAxis.EXACT_REPRODUCTION,
                PreservationAxis.ARTIFACT_RELOAD,
            ),
        ),
        ProtectedSurface(
            "routed-experts-and-math-candidates",
            "EVIDENCE_DRIVEN_ALLOCATION",
            (PreservationAxis.PROOF_REPAIR, PreservationAxis.FALSE_LEMMA_REJECTION, PreservationAxis.PERTURBED_TRANSFER),
        ),
    )
    ladder: QuantizationLadder = field(default_factory=QuantizationLadder)
    contract: MathPreservationContract = field(default_factory=MathPreservationContract)
    runtime_tps_deferred_to_hawking: bool = True
    research_authorized: bool = False

    def validate(self) -> None:
        self.program.validate()
        self.ladder.validate()
        if self.research_authorized or RESEARCH_AUTHORIZED:
            raise OdysseyRefused("Condense spec cannot grant research authority")
        if not self.runtime_tps_deferred_to_hawking:
            raise OdysseyRefused("TPS must remain an explicit Hawking runtime receipt, not a Condense simulation")
        names = [item.name for item in self.protected_surfaces]
        if len(names) != len(set(names)):
            raise OdysseyRefused("protected surfaces must be uniquely named")
        for item in self.protected_surfaces:
            item.validate()
        contract_axes = {item.axis for item in self.contract.requirements}
        expected_axes = {item.axis for item in MathPreservationContract.DEFAULT_REQUIREMENTS}
        if contract_axes != expected_axes:
            raise OdysseyRefused("Ramanujan Condense requires the complete fixed math-preservation contract")
        surface_axes = {axis for item in self.protected_surfaces for axis in item.required_axes}
        material_axes = expected_axes - {PreservationAxis.TEACHER_TRIANGULATION}
        if not material_axes <= surface_axes:
            raise OdysseyRefused("every non-teacher capability gate must protect at least one model surface")

    @property
    def identity_hash(self) -> str:
        self.validate()
        return content_hash(
            {
                "program": self.program.identity_hash,
                "surfaces": self.protected_surfaces,
                "ladder": self.ladder,
                "requirements": self.contract.requirements,
            }
        )

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PROTO_SCHEMA,
            "status": "FUTURE_CAPABILITY_FIRST_CONDENSE_SPEC",
            "spec_hash": self.identity_hash,
            "student": self.program.student.model.manifest(),
            "no_assumed_math_core": True,
            "protected_surfaces": [
                {"name": item.name, "policy": item.policy, "required_axes": [axis.value for axis in item.required_axes]}
                for item in self.protected_surfaces
            ],
            "quantization_ladder": [{"bpw": item.bpw, "name": item.name} for item in self.ladder.rungs],
            "promotion_contract": [
                {
                    "axis": item.axis.value,
                    "minimum_candidate": item.minimum_candidate,
                    "minimum_retention": item.minimum_retention,
                    "independent": item.independent,
                }
                for item in self.contract.requirements
            ],
            "teacher_conflict_policy": "same independently assessed statement hash; retain only independently verifier-dispositioned variants",
            "rung_binding": "every promotion receipt binds exact parent state, candidate state, and frozen evaluator-suite hashes",
            "runtime": {
                "tps": "DEFERRED_TO_HAWKING_RUNTIME_RECEIPT",
                "artifact_reload": "REQUIRED_BEFORE_CONDENSE_PROMOTION",
            },
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "production_authority": False,
        }


def build_ramanujan_condense_spec() -> RamanujanCondenseSpec:
    """Build and validate the future capability-preserving Condense contract."""
    spec = RamanujanCondenseSpec()
    spec.validate()
    return spec


def build_ramanujan_proto_program() -> RamanujanProtoProgram:
    """One explicit construction point for the future strict-Flash program."""
    program = RamanujanProtoProgram()
    program.validate()
    return program


F_STAGE_PURPOSE = {
    "F0": "instrumentation",
    "F1": "retrieval",
    "F2": "method_selection",
    "F3": "formalization",
    "F4": "supervised_proving",
    "F5": "verified_director_distillation",
    "F6": "repair",
    "F7": "expert_iteration",
    "F8": "preference",
    "F9": "bounded_verifier_rl",
    "F10": "reversible_director_adapters",
    "F11": "composer_solver",
    "F12": "checkpoint_tournament",
}


@dataclass(frozen=True)
class TrainingStageResult:
    stage: str
    purpose: str
    status: str
    artifacts_hash: str
    authority: str = AUTHORITY


@dataclass
class FTrainer:
    """F0--F12 fixture stage controller.  It tracks readiness; it does not train a real parent/student."""

    machine: StageMachine = field(default_factory=lambda: StageMachine("F", TRAINING_STAGES))

    def run(self, stage: str, artifacts: Mapping[str, Any]) -> TrainingStageResult:
        if stage not in F_STAGE_PURPOSE:
            raise TransitionRefused(f"unknown training stage {stage!r}")
        trace_shard = artifacts.get("verified_trace_shard")
        if stage == "F5" and (
            not isinstance(trace_shard, TraceShard) or not trace_shard.sealed or not trace_shard.cold_key
        ):
            raise OdysseyRefused("F5 requires a sealed dispositioned trace shard")
        if stage == "F10" and not artifacts.get("reversible"):
            raise OdysseyRefused("F10 requires a reversible, hash-bound adapter")
        if stage == "F10":
            _require_hash(str(artifacts.get("adapter_hash") or ""), "adapter_hash")
        frontier = artifacts.get("pareto_frontier")
        if stage == "F12" and (
            not isinstance(frontier, Sequence)
            or isinstance(frontier, (str, bytes))
            or not frontier
            or not all(isinstance(candidate, CandidateCheckpoint) for candidate in frontier)
        ):
            raise OdysseyRefused("F12 requires a checkpoint Pareto frontier")
        record = self.machine.advance(
            stage,
            artifacts={**dict(artifacts), "RAMANUJAN_RESEARCH_AUTHORIZED": False},
            note=F_STAGE_PURPOSE[stage],
        )
        return TrainingStageResult(stage, F_STAGE_PURPOSE[stage], "FIXTURE_REHEARSAL_RECORDED", record.artifacts_hash)


@dataclass(frozen=True)
class CandidateCheckpoint:
    id: str
    metrics: Mapping[str, float]
    hard_gates: Mapping[str, bool]


class CheckpointTournament:
    """Pareto selection with hard gates; no scalar score can hide a failed verifier gate."""

    # Larger is better except for these operational costs.
    LOWER_IS_BETTER = frozenset({"storage", "runtime", "cost"})

    def frontier(self, candidates: Sequence[CandidateCheckpoint]) -> list[CandidateCheckpoint]:
        eligible = [candidate for candidate in candidates if candidate.hard_gates and all(candidate.hard_gates.values())]
        result: list[CandidateCheckpoint] = []
        for candidate in eligible:
            dominated = False
            for other in eligible:
                if other is candidate:
                    continue
                keys = set(candidate.metrics) | set(other.metrics)
                not_worse = all(
                    (other.metrics.get(key, float("-inf")) >= candidate.metrics.get(key, float("-inf")))
                    if key not in self.LOWER_IS_BETTER
                    else (other.metrics.get(key, float("inf")) <= candidate.metrics.get(key, float("inf")))
                    for key in keys
                )
                strictly_better = any(
                    (other.metrics.get(key, float("-inf")) > candidate.metrics.get(key, float("-inf")))
                    if key not in self.LOWER_IS_BETTER
                    else (other.metrics.get(key, float("inf")) < candidate.metrics.get(key, float("inf")))
                    for key in keys
                )
                if not_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                result.append(candidate)
        return sorted(result, key=lambda candidate: candidate.id)


class ComposerSolver:
    """F11 interface: Composer proposes structured subgoals; Solver acts; verifier adjudicates."""

    def run(
        self,
        problem: Mapping[str, Any],
        composer: Callable[[Mapping[str, Any]], Sequence[str]],
        solver: Callable[[str], Mapping[str, Any]],
        verifier: Callable[[Mapping[str, Any]], bool],
    ) -> dict[str, Any]:
        subgoals = tuple(str(item) for item in composer(dict(problem)) if str(item).strip())
        if not subgoals:
            raise OdysseyRefused("Composer must emit structured subgoals")
        attempts = [dict(solver(subgoal)) for subgoal in subgoals]
        accepted = [attempt for attempt in attempts if verifier(attempt)]
        return {
            "subgoals": subgoals,
            "accepted": len(accepted),
            "rejected": len(attempts) - len(accepted),
            "authority": AUTHORITY,
        }


ATTACKS = (
    "prompt_injection",
    "malicious_latex",
    "malicious_lean_import",
    "path_traversal",
    "symlink_escape",
    "tool_output_injection",
    "hidden_test_leakage",
    "retrieval_poisoning",
    "checkpoint_poisoning",
    "ledger_truncation",
    "verifier_spoofing",
    "role_self_promotion",
    "tribunal_collusion",
    "fork_bomb_or_orphan_process",
    "disk_exhaustion",
)


@dataclass(frozen=True)
class AttackResult:
    attack: str
    status: str  # CONTAINED | BREACH | UNEXERCISED
    evidence_hash: str | None = None
    note: str = ""


@dataclass
class AttackHarness:
    """Attack orchestration is fail-closed: omitted probes are UNEXERCISED, never passed."""

    def run(self, probes: Mapping[str, Callable[[], bool | Mapping[str, Any]]]) -> list[AttackResult]:
        unknown = set(probes) - set(ATTACKS)
        if unknown:
            raise OdysseyRefused(f"unknown attack probes: {sorted(unknown)}")
        results: list[AttackResult] = []
        for attack in ATTACKS:
            probe = probes.get(attack)
            if probe is None:
                results.append(AttackResult(attack, "UNEXERCISED", note="no probe supplied"))
                continue
            try:
                output = probe()
                if isinstance(output, Mapping):
                    contained = output.get("contained") is True
                    evidence = output.get("evidence")
                    evidence_hash = content_hash(evidence) if evidence is not None else None
                    note = str(output.get("note") or "")
                else:
                    contained = output is True
                    evidence_hash, note = None, "boolean probe"
                results.append(AttackResult(attack, "CONTAINED" if contained else "BREACH", evidence_hash, note))
            except Exception as exc:  # A crashing defense is not evidence of containment.
                results.append(AttackResult(attack, "BREACH", note=f"probe raised {type(exc).__name__}: {exc}"))
        return results

    @staticmethod
    def all_contained(results: Sequence[AttackResult]) -> bool:
        return len(results) == len(ATTACKS) and all(result.status == "CONTAINED" for result in results)


Q_STAGE_PURPOSE = {
    "Q0": "deterministic_recovery",
    "Q1": "corpus_integrity",
    "Q2": "statement_fidelity",
    "Q3": "retrieval_and_landscape",
    "Q4": "formal_pipeline",
    "Q5": "computational_pipeline",
    "Q6": "proof_repair",
    "Q7": "numerical_repair",
    "Q8": "adversarial_epistemics",
    "Q9": "expert_seminar",
    "Q10": "known_solution_reconstruction",
    "Q11": "transfer",
    "Q12": "multi_day_sealed_rehearsal",
}

_Q_HASH_EVIDENCE = {
    "Q1": "corpus_integrity_hash",
    "Q2": "statement_fidelity_hash",
    "Q3": "landscape_retrieval_hash",
    "Q4": "lean_replay_hash",
    "Q5": "exact_certificate_hash",
    "Q6": "repair_replay_hash",
    "Q7": "numerical_repair_hash",
    "Q10": "reconstruction_hash",
    "Q12": "sealed_rehearsal_hash",
}


@dataclass(frozen=True)
class QualificationResult:
    stage: str
    status: str
    evidence_hash: str
    note: str
    authority: str = AUTHORITY


@dataclass
class QRunner:
    """Q0--Q12 controller.  A full fixture pass is a rehearsal, never sandbox authorization."""

    machine: StageMachine = field(default_factory=lambda: StageMachine("Q", QUALIFICATION_STAGES))

    def run(self, stage: str, evidence: Mapping[str, Any]) -> QualificationResult:
        if stage not in Q_STAGE_PURPOSE:
            raise TransitionRefused(f"unknown qualification stage {stage!r}")
        if evidence.get("RAMANUJAN_RESEARCH_AUTHORIZED") is True:
            raise OdysseyRefused("Q stages cannot grant research authority")
        if stage == "Q0":
            restore = evidence.get("cold_restore")
            if not isinstance(restore, Mapping) or restore.get("hash_verified") is not True:
                raise OdysseyRefused("Q0 requires a hash-verified deterministic cold restore receipt")
            _require_hash(str(restore.get("checkpoint_hash") or ""), "Q0 checkpoint_hash")
        if stage in _Q_HASH_EVIDENCE:
            _require_hash(str(evidence.get(_Q_HASH_EVIDENCE[stage]) or ""), _Q_HASH_EVIDENCE[stage])
        if stage == "Q8":
            results = evidence.get("attacks")
            if (
                not isinstance(results, Sequence)
                or isinstance(results, (str, bytes))
                or not all(isinstance(result, AttackResult) for result in results)
                or not AttackHarness.all_contained(results)
            ):
                raise OdysseyRefused("Q8 requires every declared attack to be independently contained")
        if stage == "Q9":
            _require_hash(str(evidence.get("seminar_record_hash") or ""), "Q9 seminar_record_hash")
            if evidence.get("seminar_verdict") not in {
                SeminarVerdict.PROVISIONAL_TO_TRIBUNAL.value,
                SeminarVerdict.NARROW.value,
            }:
                raise OdysseyRefused("Q9 requires an auditable seminar result")
        if stage == "Q10" and evidence.get("blind_reconstruction") is not True:
            raise OdysseyRefused("Q10 requires an explicitly blind reconstruction receipt")
        if stage == "Q11":
            transfer = evidence.get("transfer")
            if not isinstance(transfer, ReconstructionResult) or transfer.leaked_solution or transfer.structural_recall < 1.0:
                raise OdysseyRefused("Q11 requires a leak-free complete structural transfer on the fixture variant")
        record = self.machine.advance(
            stage,
            artifacts={**dict(evidence), "RAMANUJAN_RESEARCH_AUTHORIZED": False},
            note=Q_STAGE_PURPOSE[stage],
        )
        return QualificationResult(stage, "FIXTURE_REHEARSAL_RECORDED", record.artifacts_hash, Q_STAGE_PURPOSE[stage])

    def summary(self) -> dict[str, Any]:
        completed = tuple(record.stage for record in self.machine.records)
        return {
            "completed": completed,
            "fixture_rehearsal_complete": completed == QUALIFICATION_STAGES,
            "RAMANUJAN_SANDBOX_READY": False,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "note": "Fixture success is not a true-sandbox readiness or research authorization claim.",
        }


def build_expert_review_packet(
    case: CaseStudy,
    capsule: MethodCapsule,
    graph: QuestionGraph,
    *,
    trace: TraceRecord | None = None,
    interventions: InterventionLedger | None = None,
) -> dict[str, Any]:
    """Prepare a concrete external-review request instead of inventing expert conclusions."""
    capsule.validate()
    admission = ingest_case(case)
    if not admission.accepted:
        raise OdysseyRefused(f"cannot build review packet from inadmissible case: {admission.reasons}")
    coverage = graph.coverage()
    return {
        "schema": SCHEMA,
        "status": "REQUEST_EXTERNAL_REVIEW",
        "fixture_only": True,
        "production_authority": False,
        "research_authority": False,
        "claim": {"statement": case.honest_statement, "scope": list(case.scope), "case_hash": case.hash},
        "definitions": list(capsule.definitions),
        "dependency_graph": [[node, list(deps)] for node, deps in capsule.critical_lemma_graph],
        "critical_lemmas": [node for node, _deps in capsule.critical_lemma_graph],
        "instruments": list(capsule.instruments),
        "known_failure_modes": list(capsule.failed_natural_approaches),
        "prior_art_and_provenance": list(capsule.provenance),
        "questions_requiring_expertise": coverage["unanswered_categories"] + coverage["uncertified_categories"],
        "trace_hash": None if trace is None else trace.hash,
        "method_capsule_hash": capsule.hash,
        "intervention_label": None if interventions is None else interventions.autonomy_label(),
        "reproduction_command": "python3.12 -m ramanujan.odyssey --fixture-selftest",
        "note": "Packet requests qualified review; it does not assert novelty, correctness, or significance.",
    }


@dataclass
class OdysseyController:
    """Small composition root for a T0--T12 fixture rehearsal."""

    freeze: EnvironmentFreeze
    ledger: Ledger | None = None
    t_machine: StageMachine = field(init=False)
    interventions: InterventionLedger = field(init=False)
    debt: ExpertiseDebtTracker = field(init=False)
    economist: OdysseyEconomist = field(init=False)

    def __post_init__(self) -> None:
        self.freeze.validate()
        self.t_machine = StageMachine("T", ODYSSEY_STAGES, ledger=self.ledger)
        self.interventions = InterventionLedger(self.ledger)
        self.debt = ExpertiseDebtTracker(self.ledger)
        self.economist = OdysseyEconomist(self.debt, ledger=self.ledger)

    def advance(self, stage: str, artifacts: Mapping[str, Any], *, note: str | None = None) -> StageRecord:
        if stage not in T_STAGE_PURPOSE:
            raise TransitionRefused(f"unknown Odyssey stage {stage!r}")
        return self.t_machine.advance(
            stage,
            artifacts={**dict(artifacts), "RAMANUJAN_RESEARCH_AUTHORIZED": False},
            note=note or T_STAGE_PURPOSE[stage],
        )

    def checkpoint_resume(self) -> dict[str, Any]:
        return self.t_machine.checkpoint()


def run_fixture_rehearsal() -> dict[str, Any]:
    """Run an accelerated, entirely synthetic T0--T12/F0--F12/Q0--Q12 rehearsal.

    This is intentionally an integration test with disposable local storage, not a
    campaign launch.  Every "verifier" is a named fixture and the result explicitly
    remains non-authorizing even when every simulated gate is exercised.
    """
    seed = text_hash("ramanujan-odyssey-full-fixture")
    freeze = freeze_environment(
        run_id="full-fixture-rehearsal",
        director_hash=seed,
        toolchain_hash=text_hash("toolchain"),
        corpus_manifest_hash=text_hash("corpus"),
        membership_hash=text_hash("membership"),
        contamination_hash=text_hash("contamination"),
        storage_receipt_hash=text_hash("storage"),
    )
    case = CaseStudy(
        id="full-fixture-case",
        honest_statement="Fixture addition is commutative over fixture naturals.",
        scope=("fixture naturals",),
        provenance=(CaseEvidence(AuthorityBasis.FORMAL_LIBRARY, "fixture-lean", text_hash("case-proof")),),
        known_solution_visibility="progressive",
        expected_structure=("invariant", "reduction"),
        disclosure_cards=(("definitions", "fixture arithmetic"),),
        contamination_risk="synthetic fixture only",
    )
    if not ingest_case(case).accepted:
        raise AssertionError("internal fixture case must be admissible")
    graph = QuestionGraph.for_case(case.id)
    for question in graph.questions.values():
        graph.answer(
            question.id,
            "fixture-backed answer",
            bases=(AuthorityBasis.HUMAN_REVIEW if question.category is QuestionCategory.NOVELTY else AuthorityBasis.FORMAL_LIBRARY,),
            evidence_refs=("fixture-evidence",),
            certifying=True,
        )
    transfer = ProgressiveDisclosureHarness().evaluate(case, case.expected_structure, disclosed_level=1)
    seminar = SeminarRunner().run(case, graph, fragile_lemma="fixture-swap", transfer=transfer)
    if seminar.verdict is not SeminarVerdict.PROVISIONAL_TO_TRIBUNAL:
        raise AssertionError("internal fixture seminar must be structurally complete")

    plan = StoragePlan(1, 1, 1, 1, 1, 1, 1)
    with tempfile.TemporaryDirectory(prefix="ramanujan_odyssey_fixture_") as directory:
        cold = MockColdStore(Path(directory) / "cold")
        scheduler = PhaseSeparatedScheduler()
        executor = StreamingDirectorTraceExecutor()
        trace = scheduler.run_director_epoch(
            plan,
            lambda: executor.collect(
                freeze,
                case,
                membership="fixture-train",
                template_hash=text_hash("fixture-template"),
                producer=lambda _case: iter(
                    (
                        {"method_label": "invariant", "plan": ["reduce"], "subgoals": ["swap"]},
                        {
                            "retrieval_set": ["fixture-lean"],
                            "actions": ["simp"],
                            "tool_calls": [{"tool": "lean"}],
                            "verifier_outcomes": [{"kind": "lean_replay", "container_hash": text_hash("fixture-container")}],
                        },
                    )
                ),
            ),
            free_bytes=16,
        )
        shard = TraceCompactor().seal([trace], cold_store=cold)
        student_shard = scheduler.run_student_epoch(
            plan, shard, lambda sealed: StudentShardExecutor(cold).convert(sealed), free_bytes=16
        )
        checkpoints = DeltaCheckpointManager(cold)
        checkpoint = checkpoints.write_delta("current", {"student_shard": student_shard.sha256})
        restore = checkpoints.cold_restore("current")

        controller = OdysseyController(freeze)
        t_artifacts = {
            "T0": admit_substrate(freeze),
            "T1": freeze.receipt(),
            "T2": {"roles": 10, "stores": 7, "question_graph": True},
            "T3": {"trace_shard_hash": shard.sha256},
            "T4": {"student_shard_hash": student_shard.sha256},
            "T5": {"reconstruction_hash": content_hash(transfer)},
            "T6": {"seminar": seminar.verdict.value},
            "T7": {"repair_pairs": "fixture"},
            "T8": {"attack_injection": "fixture"},
            "T9": {"verifier_preference": "fixture"},
            "T10": {"transfer_hash": content_hash(transfer)},
            "T11": {"checkpoint_hash": checkpoint.sha256},
            "T12": {"tournament": "fixture"},
        }
        for stage in ODYSSEY_STAGES:
            controller.advance(stage, t_artifacts[stage])

        trainer = FTrainer()
        frontier = (CandidateCheckpoint("fixture-student", {"correctness": 1.0, "storage": 1.0}, {"formal": True}),)
        for stage in TRAINING_STAGES:
            inputs: dict[str, Any] = {}
            if stage == "F5":
                inputs["verified_trace_shard"] = shard
            elif stage == "F10":
                inputs.update({"reversible": True, "adapter_hash": text_hash("fixture-adapter")})
            elif stage == "F12":
                inputs["pareto_frontier"] = frontier
            trainer.run(stage, inputs)

        attack_results = AttackHarness().run(
            {attack: (lambda name=attack: {"contained": True, "evidence": {"fixture_attack": name}}) for attack in ATTACKS}
        )
        qualifier = QRunner()
        for stage in QUALIFICATION_STAGES:
            evidence: dict[str, Any] = {}
            if stage == "Q0":
                evidence["cold_restore"] = {"hash_verified": restore["fixture_only"] is True, "checkpoint_hash": checkpoint.sha256}
            elif stage in _Q_HASH_EVIDENCE:
                evidence[_Q_HASH_EVIDENCE[stage]] = text_hash(f"fixture-{stage}")
            if stage == "Q8":
                evidence["attacks"] = attack_results
            elif stage == "Q9":
                evidence.update({"seminar_verdict": seminar.verdict.value, "seminar_record_hash": content_hash(seminar)})
            elif stage == "Q10":
                evidence["blind_reconstruction"] = True
            elif stage == "Q11":
                evidence["transfer"] = transfer
            qualifier.run(stage, evidence)
        qualification = qualifier.summary()

    return {
        "schema": SCHEMA,
        "status": "ACCELERATED_FIXTURE_REHEARSAL_COMPLETE",
        "odyssey_stages": [record.stage for record in controller.t_machine.records],
        "training_stages": [record.stage for record in trainer.machine.records],
        "qualification_stages": list(qualification["completed"]),
        "trace_shard_hash": shard.sha256,
        "student_shard_hash": student_shard.sha256,
        "all_fixture_attacks_contained": AttackHarness.all_contained(attack_results),
        "RAMANUJAN_SANDBOX_READY": False,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "production_authority": False,
    }


def _fixture_selftest() -> dict[str, Any]:
    """A no-network smoke check that demonstrates the boundary without consuming a model."""
    seed = text_hash("ramanujan-odyssey-fixture")
    freeze = freeze_environment(
        run_id="fixture-selftest",
        director_hash=seed,
        toolchain_hash=text_hash("toolchain"),
        corpus_manifest_hash=text_hash("corpus"),
        membership_hash=text_hash("membership"),
        contamination_hash=text_hash("contamination"),
        storage_receipt_hash=text_hash("storage"),
    )
    controller = OdysseyController(freeze)
    controller.advance("T0", admit_substrate(freeze))
    controller.advance("T1", freeze.receipt())
    controller.advance("T2", {"roles": 10, "fixture": True})
    return {
        "status": "FIXTURE_SELFTEST_OK",
        "stages": [record.stage for record in controller.t_machine.records],
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ramanujan fixture-only Odyssey control plane")
    parser.add_argument("--fixture-selftest", action="store_true", help="run a no-network T0--T2 boundary smoke check")
    parser.add_argument("--fixture-rehearsal", action="store_true", help="run an accelerated disposable T0--T12/F0--F12/Q0--Q12 fixture rehearsal")
    parser.add_argument("--proto-plan", action="store_true", help="print the future strict-Flash shard-stream distillation manifest")
    parser.add_argument("--proto-footprint", action="store_true", help="print the conservative future 1-BPW Flash artifact/RAM prediction")
    parser.add_argument("--proto-gravity-plan", action="store_true", help="print the future-only Gravity render contract and its blockers")
    parser.add_argument("--proto-condense-spec", action="store_true", help="print the capability-first Condense promotion contract")
    args = parser.parse_args(argv)
    selected = sum(
        bool(item)
        for item in (
            args.fixture_selftest,
            args.fixture_rehearsal,
            args.proto_plan,
            args.proto_footprint,
            args.proto_gravity_plan,
            args.proto_condense_spec,
        )
    )
    if selected != 1:
        parser.error("select exactly one fixture or future-plan command")
    if args.fixture_selftest:
        result: Any = _fixture_selftest()
    elif args.fixture_rehearsal:
        result = run_fixture_rehearsal()
    else:
        if args.proto_condense_spec:
            result = build_ramanujan_condense_spec().manifest()
        else:
            program = build_ramanujan_proto_program()
            if args.proto_plan:
                result = program.manifest()
            elif args.proto_footprint:
                result = ProtoFootprint().estimate(host_ram_bytes=96 * GIB, free_disk_bytes=shutil.disk_usage(Path.cwd()).free)
            else:
                result = asdict(ProtoGravityRenderer().plan(program))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
