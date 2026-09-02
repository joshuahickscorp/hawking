"""§19.1 Bounty ontology and the 17 closed classes of §19.2.

Routing fields `bounty_class` and `lab` are not in the roadmap struct; they
exist so a Bounty can attach to exactly one class and one laboratory. The
named §19.1 fields are otherwise exact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping


class BountyClass(Enum):
    """Closed set. Values are the §19.2 phrases."""

    OPEN_SOURCE_BUG_ISSUE = "open-source bug / issue"
    COMPILER_RUNTIME_BUG = "compiler/runtime bug"
    GPU_KERNEL_OPTIMIZATION = "GPU kernel optimization challenge"
    AI_MODEL_OPTIMIZATION = "AI/model optimization bounty"
    FORMAL_PROOF_OR_COUNTEREXAMPLE = "formal proof or counterexample"
    MATH_RESEARCH_CHALLENGE = "math research challenge"
    SCIENTIFIC_REPLICATION = "scientific replication"
    SCIENTIFIC_ANOMALY_INVESTIGATION = "scientific anomaly investigation"
    PHYSICS_MODEL_DISCRIMINATION = "physics model discrimination"
    DATA_ANALYSIS_CHALLENGE = "data-analysis challenge"
    AUTHORIZED_BUG_BOUNTY_PROGRAM = "authorized bug-bounty program"
    CTF_INTENTIONALLY_VULNERABLE_LAB = "CTF / intentionally vulnerable lab"
    PERFORMANCE_ENERGY_CHALLENGE = "performance/energy challenge"
    PROTOCOL_TOOLING_INTEROPERABILITY = "protocol/tooling interoperability"
    HAWKING_INTERNAL_SELF_BOUNTY = "Hawking internal self-bounty"
    LITERATURE_EQUIVALENCE_NOVELTY_CHECK = "literature equivalence/novelty check"
    REPRODUCIBILITY_BOUNTY = "reproducibility bounty"


assert len(BountyClass) == 17, "§19.2 lists 17 classes; this enum is closed"


SECURITY_BOUNTY_CLASSES: frozenset[BountyClass] = frozenset(
    {
        BountyClass.AUTHORIZED_BUG_BOUNTY_PROGRAM,
        BountyClass.CTF_INTENTIONALLY_VULNERABLE_LAB,
    }
)


class PublicOrPrivate(Enum):
    PUBLIC = "public"
    PRIVATE = "private"


@dataclass(frozen=True)
class Budget:
    workunits: int
    notes: str = "bounded; this scaffold does not launch unbounded work"

    def __post_init__(self) -> None:
        if self.workunits < 1:
            raise ValueError("budget.workunits must be >= 1")


@dataclass(frozen=True)
class AuthorizationScope:
    """Who authorized what. Never parsed from question_or_target."""

    kind: str  # HAWKING_INTERNAL | PROGRAM | UNPINNED
    program_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"HAWKING_INTERNAL", "PROGRAM", "UNPINNED"}:
            raise ValueError(f"unknown authorization_scope.kind: {self.kind!r}")


def hawking_internal_scope() -> AuthorizationScope:
    return AuthorizationScope(kind="HAWKING_INTERNAL", program_id="hawking.self")


def unpinned_scope() -> AuthorizationScope:
    return AuthorizationScope(kind="UNPINNED", program_id=None)


INTERNAL_VERIFIER = "tools.theia.intake.verify_receipt"

_INTERNAL_RULES = (
    "local artifact only",
    "no network",
    "no ACTIVE_TEST",
    "no training",
)


def make_internal_bounty(
    *,
    id: str,
    source: str,
    domain: str,
    question_or_target: str,
    nonmonetary_value: str,
    bounty_class: BountyClass,
    lab: str,
    extra_rules: tuple[str, ...] = (),
    evidence_required: tuple[str, ...] = ("receipt schema", "seal_sha256"),
) -> "Bounty":
    """Hawking-internal bounty. Not a security class. Never infers scope from text."""
    return Bounty(
        id=id,
        source=source,
        domain=domain,
        question_or_target=question_or_target,
        monetary_reward=None,
        nonmonetary_value=nonmonetary_value,
        authorization_scope=hawking_internal_scope(),
        rules=_INTERNAL_RULES + extra_rules,
        evidence_required=evidence_required,
        verifier=INTERNAL_VERIFIER,
        budget=Budget(workunits=1),
        deadline=None,
        public_or_private=PublicOrPrivate.PRIVATE,
        submission_policy="stage into receipts/future; do not publish externally",
        success_conditions=(
            "intake reaches TRAJECTORY + METHOD + NEGATIVE SCIENCE",
            "schedule score produced",
            "independent receipt verification",
        ),
        stop_conditions=("BLOCKED_RIGHTS", "missing receipt", "unreadable json"),
        bounty_class=bounty_class,
        lab=lab,
    )


@dataclass(frozen=True)
class Bounty:
    id: str
    source: str
    domain: str
    question_or_target: str
    monetary_reward: Fraction | None
    nonmonetary_value: str
    authorization_scope: AuthorizationScope
    rules: tuple[str, ...]
    evidence_required: tuple[str, ...]
    verifier: str
    budget: Budget
    deadline: str | None
    public_or_private: PublicOrPrivate
    submission_policy: str
    success_conditions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    bounty_class: BountyClass
    lab: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Bounty.id is required")
        if not self.verifier:
            raise ValueError("Bounty.verifier is required")
        if not self.success_conditions:
            raise ValueError("Bounty.success_conditions is required")
        if not self.stop_conditions:
            raise ValueError("Bounty.stop_conditions is required")

    def to_json_dict(self) -> dict[str, Any]:
        reward: Any = None
        if self.monetary_reward is not None:
            reward = {
                "numerator": self.monetary_reward.numerator,
                "denominator": self.monetary_reward.denominator,
            }
        return {
            "id": self.id,
            "source": self.source,
            "domain": self.domain,
            "question_or_target": self.question_or_target,
            "monetary_reward": reward,
            "nonmonetary_value": self.nonmonetary_value,
            "authorization_scope": asdict(self.authorization_scope),
            "rules": list(self.rules),
            "evidence_required": list(self.evidence_required),
            "verifier": self.verifier,
            "budget": asdict(self.budget),
            "deadline": self.deadline,
            "public_or_private": self.public_or_private.value,
            "submission_policy": self.submission_policy,
            "success_conditions": list(self.success_conditions),
            "stop_conditions": list(self.stop_conditions),
            "bounty_class": self.bounty_class.value,
            "lab": self.lab,
        }

    @classmethod
    def from_json_dict(cls, d: Mapping[str, Any]) -> "Bounty":
        reward = d.get("monetary_reward")
        money: Fraction | None
        if reward is None:
            money = None
        elif isinstance(reward, Mapping):
            money = Fraction(int(reward["numerator"]), int(reward["denominator"]))
        else:
            raise TypeError("monetary_reward must be null or {numerator, denominator}")
        scope = d["authorization_scope"]
        return cls(
            id=str(d["id"]),
            source=str(d["source"]),
            domain=str(d["domain"]),
            question_or_target=str(d["question_or_target"]),
            monetary_reward=money,
            nonmonetary_value=str(d["nonmonetary_value"]),
            authorization_scope=AuthorizationScope(
                kind=str(scope["kind"]),
                program_id=scope.get("program_id"),
            ),
            rules=tuple(d["rules"]),
            evidence_required=tuple(d["evidence_required"]),
            verifier=str(d["verifier"]),
            budget=Budget(
                workunits=int(d["budget"]["workunits"]),
                notes=str(d["budget"].get("notes") or Budget.notes),
            ),
            deadline=d.get("deadline"),
            public_or_private=PublicOrPrivate(d["public_or_private"]),
            submission_policy=str(d["submission_policy"]),
            success_conditions=tuple(d["success_conditions"]),
            stop_conditions=tuple(d["stop_conditions"]),
            bounty_class=BountyClass(d["bounty_class"]),
            lab=str(d["lab"]),
        )
