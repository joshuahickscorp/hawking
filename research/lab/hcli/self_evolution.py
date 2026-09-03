"""Self-evolution admission pipeline (Bible §23) — interface + stub.

Verified trajectories may propose memory / rule / skill / tool / index /
benchmark / anti-pattern updates. Admission is multi-stage and never accepts
from a single noisy result.

Negative science reuses the Ramanujan graveyard *laws* (read-only reference
from ``ramanujan.scaffold.core.stores``):

  - burial is not deletion — rejected proposals stay auditable
  - free resurrection is refused — revive only with a premise-changing event
  - the proposer is never the admitter (tribunal separation)

Promotion-gate honesty mirrors ``lab.operators.frankenstein_promotion_gate``:
missing evidence yields PENDING, never a fabricated ACCEPT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import uuid

from lab.receipts import seal

SCHEMA = "hawking.hcli.self_evolution.v1"
LEDGER_SCHEMA = "hawking.hcli.evolution_ledger.v1"

# Minimum independent historical-trace replays before admission may leave PENDING.
MIN_REPLAY_TRACES = 2
# Minimum distinct outcomes (accept or reject) that must accumulate before a
# *class* of proposal may auto-bias future routing. Single-result rewrites are banned.
MIN_CLASS_SAMPLES_FOR_BIAS = 3


class ProposalKind(str, Enum):
    MEMORY_UPDATE = "memory_update"
    BEHAVIORAL_RULE = "behavioral_rule"
    SKILL_UPDATE = "skill_update"
    TOOL_WRAPPER = "tool_wrapper"
    TOOL_RETIREMENT = "tool_retirement"
    SEARCH_INDEX_UPDATE = "search_index_update"
    NEW_BENCHMARK = "new_benchmark"
    NEW_ANTI_PATTERN = "new_anti_pattern"


PROPOSAL_KINDS: tuple[ProposalKind, ...] = tuple(ProposalKind)


class AdmissionStage(str, Enum):
    """Linear admission path; only PROTECTED_CONTROLLER advances past COMPAT."""

    PROPOSED = "proposed"
    REPLAY = "replay_on_historical_traces"
    HIDDEN_VALIDATION = "hidden_validation"
    COMPAT_TEST = "compatibility_test"
    PROTECTED_ADMISSION = "protected_admission"
    # Terminals
    ADMITTED = "admitted"
    BURIED = "buried"
    PENDING = "pending"


_STAGE_ORDER: tuple[AdmissionStage, ...] = (
    AdmissionStage.PROPOSED,
    AdmissionStage.REPLAY,
    AdmissionStage.HIDDEN_VALIDATION,
    AdmissionStage.COMPAT_TEST,
    AdmissionStage.PROTECTED_ADMISSION,
)

_TERMINALS = frozenset(
    {
        AdmissionStage.ADMITTED,
        AdmissionStage.BURIED,
        AdmissionStage.PENDING,
    }
)

# Ledger event kinds that can license revival of a buried proposal (premise change).
PREMISE_CHANGE_KINDS = frozenset(
    {
        "new_historical_trace_corpus",
        "hidden_validation_redesign",
        "compatibility_surface_change",
        "controller_policy_revision",
    }
)


class EvolutionRefusal(ValueError):
    """Caller asked for an unlawful admission transition."""


class GraveyardRefused(EvolutionRefusal):
    """Free resurrection, double-burial, or destructive rewrite refused."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass
class Proposal:
    """A candidate self-modification. Never mutates production state itself."""

    kind: ProposalKind
    body: Mapping[str, Any]
    proposer: str
    source_trajectory_id: str
    proposal_id: str = field(default_factory=lambda: f"evp-{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=_utc_now)
    stage: AdmissionStage = AdmissionStage.PROPOSED
    evidence: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None  # ACCEPT | REJECT | PENDING
    bury_reason: str | None = None
    buried_at_seq: int | None = None
    premise_change_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind.value,
            "body": dict(self.body),
            "proposer": self.proposer,
            "source_trajectory_id": self.source_trajectory_id,
            "created_at": self.created_at,
            "stage": self.stage.value,
            "evidence": dict(self.evidence),
            "verdict": self.verdict,
            "bury_reason": self.bury_reason,
            "buried_at_seq": self.buried_at_seq,
            "premise_change_seq": self.premise_change_seq,
        }


@dataclass
class LedgerRow:
    seq: int
    kind: str
    payload: dict[str, Any]
    actor: str
    recorded_at: str


@dataclass
class EvolutionLedger:
    """Append-only outcome ledger. Learns from accept *and* reject.

    There is deliberately no delete/edit. Corrections are new rows; refutations
    are burials. This is the self-evolution analogue of Ramanujan's graveyard.
    """

    rows: list[LedgerRow] = field(default_factory=list)
    _superseded: set[int] = field(default_factory=set)

    def append(self, kind: str, payload: Mapping[str, Any], *, actor: str) -> LedgerRow:
        row = LedgerRow(
            seq=len(self.rows),
            kind=kind,
            payload=dict(payload),
            actor=actor,
            recorded_at=_utc_now(),
        )
        self.rows.append(row)
        return row

    def supersede(self, seq: int, *, actor: str, reason: str) -> LedgerRow:
        if not 0 <= seq < len(self.rows):
            raise EvolutionRefusal(f"cannot supersede missing seq {seq}")
        self._superseded.add(seq)
        return self.append(
            "supersession",
            {"supersedes": seq, "reason": reason},
            actor=actor,
        )

    def superseded_seqs(self) -> frozenset[int]:
        return frozenset(self._superseded)

    def outcomes_for_kind(self, kind: ProposalKind) -> list[LedgerRow]:
        return [
            r
            for r in self.rows
            if r.kind in {"admitted", "buried"}
            and r.payload.get("proposal_kind") == kind.value
            and r.seq not in self._superseded
        ]

    def class_sample_count(self, kind: ProposalKind) -> int:
        return len(self.outcomes_for_kind(kind))

    def may_bias_routing(self, kind: ProposalKind) -> bool:
        """Never rewrite behaviour from one noisy result."""
        return self.class_sample_count(kind) >= MIN_CLASS_SAMPLES_FOR_BIAS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "rows": [
                {
                    "seq": r.seq,
                    "kind": r.kind,
                    "payload": r.payload,
                    "actor": r.actor,
                    "recorded_at": r.recorded_at,
                }
                for r in self.rows
            ],
            "superseded": sorted(self._superseded),
        }


@dataclass
class StageResult:
    stage: AdmissionStage
    status: str  # PASS | FAIL | PENDING
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


class AdmissionPipeline:
    """propose → replay → hidden validation → compat test → protected admission.

    Pure logic / stubs. Stage runners accept injected evidence maps so tests
    can drive PASS/FAIL/PENDING without live model work.
    """

    def __init__(self, ledger: EvolutionLedger | None = None) -> None:
        self.ledger = ledger or EvolutionLedger()
        self.proposals: dict[str, Proposal] = {}

    def propose(
        self,
        *,
        kind: ProposalKind | str,
        body: Mapping[str, Any],
        proposer: str,
        source_trajectory_id: str,
        proposal_id: str | None = None,
    ) -> Proposal:
        if not proposer or not str(proposer).strip():
            raise EvolutionRefusal("proposer is required")
        if not source_trajectory_id or not str(source_trajectory_id).strip():
            raise EvolutionRefusal("source_trajectory_id is required (verified trajectory)")
        if not isinstance(body, Mapping) or not body:
            raise EvolutionRefusal("proposal body must be a non-empty mapping")
        pk = kind if isinstance(kind, ProposalKind) else ProposalKind(kind)
        kwargs: dict[str, Any] = {
            "kind": pk,
            "body": dict(body),
            "proposer": proposer,
            "source_trajectory_id": source_trajectory_id,
        }
        if proposal_id is not None:
            kwargs["proposal_id"] = proposal_id
        p = Proposal(**kwargs)
        if p.proposal_id in self.proposals:
            raise EvolutionRefusal(f"proposal {p.proposal_id!r} already exists; never overwrite")
        self.proposals[p.proposal_id] = p
        self.ledger.append(
            "proposed",
            {
                "proposal_id": p.proposal_id,
                "proposal_kind": p.kind.value,
                "source_trajectory_id": source_trajectory_id,
                "body_sha256": _digest(body),
            },
            actor=proposer,
        )
        return p

    def _require(self, proposal_id: str) -> Proposal:
        if proposal_id not in self.proposals:
            raise EvolutionRefusal(f"unknown proposal {proposal_id!r}")
        p = self.proposals[proposal_id]
        if p.stage == AdmissionStage.BURIED:
            raise GraveyardRefused(
                f"proposal {proposal_id!r} is buried; revive with a premise change first"
            )
        if p.stage == AdmissionStage.ADMITTED:
            raise EvolutionRefusal(f"proposal {proposal_id!r} is already admitted")
        return p

    def run_replay(
        self,
        proposal_id: str,
        *,
        historical_trace_ids: Sequence[str],
        outcomes: Mapping[str, str] | None = None,
        actor: str = "replay_runner",
    ) -> StageResult:
        """Replay proposal on historical traces. Stub: outcomes injected."""
        p = self._require(proposal_id)
        if p.stage not in {AdmissionStage.PROPOSED, AdmissionStage.PENDING}:
            raise EvolutionRefusal(
                f"replay requires PROPOSED/PENDING, got {p.stage.value}"
            )
        traces = list(historical_trace_ids)
        if len(traces) < MIN_REPLAY_TRACES:
            result = StageResult(
                stage=AdmissionStage.REPLAY,
                status="PENDING",
                detail=(
                    f"need >= {MIN_REPLAY_TRACES} historical traces; got {len(traces)}. "
                    "Never admit from a single noisy result."
                ),
                metrics={"trace_count": len(traces)},
            )
            p.stage = AdmissionStage.PENDING
            p.verdict = "PENDING"
            p.evidence["replay"] = result.to_dict()
            self.ledger.append(
                "stage_result",
                {"proposal_id": proposal_id, **result.to_dict()},
                actor=actor,
            )
            return result

        outcomes = dict(outcomes or {})
        fails = [t for t, s in outcomes.items() if s == "FAIL"]
        pending = [t for t in traces if t not in outcomes]
        if pending:
            status = "PENDING"
            detail = f"missing outcomes for traces: {pending}"
        elif fails:
            status = "FAIL"
            detail = f"replay failed on {fails}"
        else:
            status = "PASS"
            detail = f"replay passed on {len(traces)} traces"

        result = StageResult(
            stage=AdmissionStage.REPLAY,
            status=status,
            detail=detail,
            metrics={"trace_count": len(traces), "fails": fails, "pending": pending},
        )
        p.evidence["replay"] = result.to_dict()
        if status == "FAIL":
            self.bury(proposal_id, reason=detail, actor=actor)
        elif status == "PENDING":
            p.stage = AdmissionStage.PENDING
            p.verdict = "PENDING"
        else:
            p.stage = AdmissionStage.REPLAY
        self.ledger.append(
            "stage_result",
            {"proposal_id": proposal_id, **result.to_dict()},
            actor=actor,
        )
        return result

    def run_hidden_validation(
        self,
        proposal_id: str,
        *,
        hidden_bundle: Mapping[str, Any] | None = None,
        actor: str = "hidden_validator",
    ) -> StageResult:
        p = self._require(proposal_id)
        if p.stage not in {AdmissionStage.REPLAY, AdmissionStage.PENDING}:
            # Allow advance only after successful replay.
            if p.evidence.get("replay", {}).get("status") != "PASS":
                raise EvolutionRefusal(
                    f"hidden validation requires successful replay; stage={p.stage.value}"
                )
        if p.evidence.get("replay", {}).get("status") != "PASS":
            raise EvolutionRefusal("hidden validation requires replay PASS")

        if hidden_bundle is None:
            result = StageResult(
                stage=AdmissionStage.HIDDEN_VALIDATION,
                status="PENDING",
                detail="hidden validation bundle not provided (honest PENDING, never fabricate)",
            )
            p.stage = AdmissionStage.PENDING
            p.verdict = "PENDING"
            p.evidence["hidden_validation"] = result.to_dict()
            self.ledger.append(
                "stage_result",
                {"proposal_id": proposal_id, **result.to_dict()},
                actor=actor,
            )
            return result

        ok = bool(hidden_bundle.get("pass"))
        disjoint = hidden_bundle.get("train_eval_disjoint", True)
        if not disjoint:
            ok = False
        status = "PASS" if ok else "FAIL"
        detail = str(hidden_bundle.get("detail", hidden_bundle))
        result = StageResult(
            stage=AdmissionStage.HIDDEN_VALIDATION,
            status=status,
            detail=detail,
            metrics={"train_eval_disjoint": disjoint},
        )
        p.evidence["hidden_validation"] = result.to_dict()
        if status == "FAIL":
            self.bury(proposal_id, reason=f"hidden validation failed: {detail}", actor=actor)
        else:
            p.stage = AdmissionStage.HIDDEN_VALIDATION
        self.ledger.append(
            "stage_result",
            {"proposal_id": proposal_id, **result.to_dict()},
            actor=actor,
        )
        return result

    def run_compat_test(
        self,
        proposal_id: str,
        *,
        compat_report: Mapping[str, Any] | None = None,
        actor: str = "compat_runner",
    ) -> StageResult:
        p = self._require(proposal_id)
        if p.evidence.get("hidden_validation", {}).get("status") != "PASS":
            raise EvolutionRefusal("compat test requires hidden_validation PASS")

        if compat_report is None:
            result = StageResult(
                stage=AdmissionStage.COMPAT_TEST,
                status="PENDING",
                detail="compatibility report not provided",
            )
            p.stage = AdmissionStage.PENDING
            p.verdict = "PENDING"
            p.evidence["compat_test"] = result.to_dict()
            self.ledger.append(
                "stage_result",
                {"proposal_id": proposal_id, **result.to_dict()},
                actor=actor,
            )
            return result

        ok = bool(compat_report.get("pass"))
        status = "PASS" if ok else "FAIL"
        detail = str(compat_report.get("detail", compat_report))
        result = StageResult(
            stage=AdmissionStage.COMPAT_TEST,
            status=status,
            detail=detail,
            metrics=dict(compat_report.get("metrics") or {}),
        )
        p.evidence["compat_test"] = result.to_dict()
        if status == "FAIL":
            self.bury(proposal_id, reason=f"compat failed: {detail}", actor=actor)
        else:
            p.stage = AdmissionStage.COMPAT_TEST
        self.ledger.append(
            "stage_result",
            {"proposal_id": proposal_id, **result.to_dict()},
            actor=actor,
        )
        return result

    def protected_admit(
        self,
        proposal_id: str,
        *,
        admitter: str,
        sign: bool = True,
    ) -> dict[str, Any]:
        """Protected controller admission. Proposer may never admit their own proposal."""
        p = self._require(proposal_id)
        if admitter == p.proposer:
            raise EvolutionRefusal(
                "tribunal separation: the system that produces a proposal is never "
                "the system that admits it"
            )
        if p.evidence.get("compat_test", {}).get("status") != "PASS":
            # Honest PENDING rather than ACCEPT when evidence incomplete.
            missing = [
                s
                for s in ("replay", "hidden_validation", "compat_test")
                if p.evidence.get(s, {}).get("status") != "PASS"
            ]
            p.stage = AdmissionStage.PENDING
            p.verdict = "PENDING"
            document = {
                "schema": SCHEMA,
                "proposal_id": proposal_id,
                "verdict": "PENDING",
                "reason": f"admission evidence incomplete: {missing}",
                "proposal": p.to_dict(),
                "fabricated_accept": False,
            }
            sealed = seal(document) if sign else document
            self.ledger.append(
                "admission_pending",
                {"proposal_id": proposal_id, "missing": missing},
                actor=admitter,
            )
            return sealed

        # Multi-trace + multi-sample fence against single-noise rewrites.
        replay_count = int(p.evidence.get("replay", {}).get("metrics", {}).get("trace_count", 0))
        if replay_count < MIN_REPLAY_TRACES:
            raise EvolutionRefusal(
                f"refuse single-noise admission: replay_count={replay_count} "
                f"< {MIN_REPLAY_TRACES}"
            )

        p.stage = AdmissionStage.ADMITTED
        p.verdict = "ACCEPT"
        self.ledger.append(
            "admitted",
            {
                "proposal_id": proposal_id,
                "proposal_kind": p.kind.value,
                "body_sha256": _digest(p.body),
                "source_trajectory_id": p.source_trajectory_id,
            },
            actor=admitter,
        )
        document = {
            "schema": SCHEMA,
            "proposal_id": proposal_id,
            "verdict": "ACCEPT",
            "reason": "all admission stages passed under protected controller",
            "proposal": p.to_dict(),
            "class_sample_count": self.ledger.class_sample_count(p.kind),
            "may_bias_routing": self.ledger.may_bias_routing(p.kind),
            "fabricated_accept": False,
        }
        return seal(document) if sign else document

    def bury(self, proposal_id: str, *, reason: str, actor: str) -> Proposal:
        """Move proposal to graveyard. Remains readable; nothing is deleted."""
        if proposal_id not in self.proposals:
            raise EvolutionRefusal(f"unknown proposal {proposal_id!r}")
        p = self.proposals[proposal_id]
        if p.stage == AdmissionStage.BURIED:
            raise GraveyardRefused(f"proposal {proposal_id!r} is already buried")
        row = self.ledger.append(
            "buried",
            {
                "proposal_id": proposal_id,
                "proposal_kind": p.kind.value,
                "buried_because": reason,
                "source_trajectory_id": p.source_trajectory_id,
            },
            actor=actor,
        )
        p.stage = AdmissionStage.BURIED
        p.verdict = "REJECT"
        p.bury_reason = reason
        p.buried_at_seq = row.seq
        return p

    def revive(
        self,
        proposal_id: str,
        *,
        because: str,
        actor: str,
        premise_change_seq: int | None = None,
    ) -> Proposal:
        """Revive only when a post-burial premise-changing ledger event exists."""
        if proposal_id not in self.proposals:
            raise EvolutionRefusal(f"unknown proposal {proposal_id!r}")
        p = self.proposals[proposal_id]
        if p.stage != AdmissionStage.BURIED:
            raise EvolutionRefusal(f"proposal {proposal_id!r} is not buried")
        if premise_change_seq is None:
            raise GraveyardRefused(
                f"cannot resurrect proposal {proposal_id!r} without a premise-changing "
                "ledger event. Burial is not deletion, and free resurrection is not revival."
            )
        self._require_valid_premise_change(p, premise_change_seq)
        self.ledger.append(
            "revived",
            {
                "proposal_id": proposal_id,
                "revived_because": because,
                "premise_change_seq": premise_change_seq,
            },
            actor=actor,
        )
        p.stage = AdmissionStage.PROPOSED
        p.verdict = None
        p.bury_reason = None
        p.premise_change_seq = premise_change_seq
        # Clear stage evidence so the pipeline re-runs under new premises.
        p.evidence = {"prior_burial_premise_change_seq": premise_change_seq}
        p.buried_at_seq = None
        return p

    def _require_valid_premise_change(self, p: Proposal, seq: int) -> None:
        rows = self.ledger.rows
        if not 0 <= seq < len(rows):
            raise GraveyardRefused(f"premise_change_seq {seq} does not refer to a ledger row")
        row = rows[seq]
        if row.kind not in PREMISE_CHANGE_KINDS:
            raise GraveyardRefused(
                f"row {seq} kind={row.kind!r} cannot change premises; "
                f"need one of {sorted(PREMISE_CHANGE_KINDS)}"
            )
        if p.buried_at_seq is None or seq <= p.buried_at_seq:
            raise GraveyardRefused(
                f"premise_change_seq {seq} must be strictly after burial seq "
                f"{p.buried_at_seq}; earlier evidence does not un-bury a proposal"
            )
        if seq in self.ledger.superseded_seqs():
            raise GraveyardRefused(
                f"premise_change_seq {seq} was superseded; a dead premise change "
                "cannot license revival"
            )

    def graveyard(self) -> Iterable[Proposal]:
        return (p for p in self.proposals.values() if p.stage == AdmissionStage.BURIED)

    def live(self) -> Iterable[Proposal]:
        return (p for p in self.proposals.values() if p.stage != AdmissionStage.BURIED)

    def evaluate_status(self, proposal_id: str) -> dict[str, Any]:
        """Summarise pipeline status without advancing (controller inspection)."""
        if proposal_id not in self.proposals:
            raise EvolutionRefusal(f"unknown proposal {proposal_id!r}")
        p = self.proposals[proposal_id]
        checks = []
        for key, stage in (
            ("replay", AdmissionStage.REPLAY),
            ("hidden_validation", AdmissionStage.HIDDEN_VALIDATION),
            ("compat_test", AdmissionStage.COMPAT_TEST),
        ):
            ev = p.evidence.get(key)
            if ev is None:
                checks.append({"name": key, "status": "PENDING", "detail": "not run"})
            else:
                checks.append(
                    {"name": key, "status": ev.get("status", "PENDING"), "detail": ev.get("detail", "")}
                )
        statuses = {c["status"] for c in checks}
        if p.stage == AdmissionStage.ADMITTED:
            overall = "ACCEPT"
        elif p.stage == AdmissionStage.BURIED or "FAIL" in statuses:
            overall = "REJECT"
        elif statuses == {"PASS"} and p.stage == AdmissionStage.COMPAT_TEST:
            overall = "READY_FOR_PROTECTED_ADMISSION"
        else:
            overall = "PENDING"
        return {
            "schema": SCHEMA,
            "proposal_id": proposal_id,
            "stage": p.stage.value,
            "overall": overall,
            "checks": checks,
            "fabricated_accept": False,
        }


class SelfEvolutionEngine:
    """Standing subsystem façade over the admission pipeline + outcome ledger."""

    def __init__(self) -> None:
        self.pipeline = AdmissionPipeline()

    @property
    def ledger(self) -> EvolutionLedger:
        return self.pipeline.ledger

    def propose_from_trajectory(
        self,
        *,
        kind: ProposalKind | str,
        body: Mapping[str, Any],
        proposer: str,
        trajectory_id: str,
    ) -> Proposal:
        return self.pipeline.propose(
            kind=kind,
            body=body,
            proposer=proposer,
            source_trajectory_id=trajectory_id,
        )

    def learn_summary(self) -> dict[str, Any]:
        """Aggregate accepted *and* rejected experiments for class-level learning."""
        by_kind: dict[str, dict[str, int]] = {}
        for kind in ProposalKind:
            outcomes = self.ledger.outcomes_for_kind(kind)
            admitted = sum(1 for r in outcomes if r.kind == "admitted")
            buried = sum(1 for r in outcomes if r.kind == "buried")
            by_kind[kind.value] = {
                "admitted": admitted,
                "buried": buried,
                "total": admitted + buried,
                "may_bias_routing": self.ledger.may_bias_routing(kind),
            }
        return seal(
            {
                "schema": LEDGER_SCHEMA,
                "by_kind": by_kind,
                "law": "never_rewrite_from_single_noisy_result",
                "min_class_samples_for_bias": MIN_CLASS_SAMPLES_FOR_BIAS,
                "min_replay_traces": MIN_REPLAY_TRACES,
            }
        )
