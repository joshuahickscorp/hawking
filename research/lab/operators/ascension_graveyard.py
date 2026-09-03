"""Negative-science Graveyard for Ascension (Bible §32).

Every proposal checks the Graveyard first. Record:
  mechanism, model/geometry, measured outcome, failure reason, reopen condition.

Known failure classes (Bible §32):
  prompt-independent collapse, beats-null misuse, median masking,
  fewer waits but slower wall time, unmeasured GPU claims, circular parity
  oracle, synthetic activation mismatch, cold-miss masking, storage
  accumulation, ignored eviction, capability loss after compression.

## Existing vs new

``ramanujan.scaffold.core.stores.Stores`` already implements a full Graveyard
with ``bury`` / ``revive`` / premise-change gating (``PREMISE_CHANGE_KINDS`` =
verifier_event | literature_query; free resurrection refused; nothing deleted;
claims stay readable). Odyssey ``ResearchBranch`` adds ``reopen_condition`` +
``bury``/``reopen`` with evidence authority bases.

This module does **not** write into ramanujan stores (read-only reference per
task contract). It reuses the *semantics*:

- bury records a failure; nothing is deleted
- revive/reopen requires a materially new premise (explicit reopen_condition
  + premise-change evidence), never "I changed my mind"
- known failure classes are first-class query keys so proposals fail closed
  when they re-state a buried mechanism without a new premise

Durable sealing into ramanujan or a campaign ledger is a later integration step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class GraveyardError(RuntimeError):
    """Graveyard invariant failed closed."""


class FailureClass(str, Enum):
    """Bible §32 known negative-science classes + extensible OTHER."""

    PROMPT_INDEPENDENT_COLLAPSE = "prompt_independent_collapse"
    BEATS_NULL_MISUSE = "beats_null_misuse"
    MEDIAN_MASKING = "median_masking"
    FEWER_WAITS_SLOWER_WALL = "fewer_waits_but_slower_wall_time"
    UNMEASURED_GPU_CLAIMS = "unmeasured_gpu_claims"
    CIRCULAR_PARITY_ORACLE = "circular_parity_oracle"
    SYNTHETIC_ACTIVATION_MISMATCH = "synthetic_activation_mismatch"
    COLD_MISS_MASKING = "cold_miss_masking"
    STORAGE_ACCUMULATION = "storage_accumulation"
    IGNORED_EVICTION = "ignored_eviction"
    CAPABILITY_LOSS_AFTER_COMPRESSION = "capability_loss_after_compression"
    OTHER = "other"


BIBLE_FAILURE_CLASSES: tuple[FailureClass, ...] = tuple(
    fc for fc in FailureClass if fc is not FailureClass.OTHER
)


@dataclass(frozen=True)
class BurialRecord:
    """One negative-science burial (Bible §32 field set)."""

    burial_id: str
    mechanism: str
    model_geometry: str
    measured_outcome: str
    failure_reason: str
    reopen_condition: str
    failure_class: FailureClass
    related_research_item_ids: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    actor: str = "adversary"
    buried: bool = True
    premise_change_evidence: str | None = None
    revived_because: str | None = None

    def __post_init__(self) -> None:
        for label in (
            "burial_id",
            "mechanism",
            "model_geometry",
            "measured_outcome",
            "failure_reason",
            "reopen_condition",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise GraveyardError(f"{label} must be a non-empty string")
        if not isinstance(self.failure_class, FailureClass):
            raise GraveyardError("failure_class must be a FailureClass")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "hawking.ascension.graveyard_burial.v1",
            "burial_id": self.burial_id,
            "mechanism": self.mechanism,
            "model_geometry": self.model_geometry,
            "measured_outcome": self.measured_outcome,
            "failure_reason": self.failure_reason,
            "reopen_condition": self.reopen_condition,
            "failure_class": self.failure_class.value,
            "related_research_item_ids": list(self.related_research_item_ids),
            "citations": list(self.citations),
            "actor": self.actor,
            "buried": self.buried,
            "premise_change_evidence": self.premise_change_evidence,
            "revived_because": self.revived_because,
        }


@dataclass
class AscensionGraveyard:
    """Negative-science store for Ascension mechanisms.

    Semantic kinship with ramanujan ``Stores.bury`` / ``Stores.revive``:
    free resurrection is refused; a premise-changing evidence string is required
    to revive. Does not touch ramanujan write paths.
    """

    burials: dict[str, BurialRecord] = field(default_factory=dict)

    def bury(self, record: BurialRecord) -> BurialRecord:
        if record.burial_id in self.burials and self.burials[record.burial_id].buried:
            raise GraveyardError(f"burial {record.burial_id!r} is already buried")
        if not record.buried:
            raise GraveyardError("bury() requires buried=True")
        # Freeze a clean burial (clear any accidental revive fields).
        clean = BurialRecord(
            burial_id=record.burial_id,
            mechanism=record.mechanism,
            model_geometry=record.model_geometry,
            measured_outcome=record.measured_outcome,
            failure_reason=record.failure_reason,
            reopen_condition=record.reopen_condition,
            failure_class=record.failure_class,
            related_research_item_ids=record.related_research_item_ids,
            citations=record.citations,
            actor=record.actor,
            buried=True,
            premise_change_evidence=None,
            revived_because=None,
        )
        self.burials[clean.burial_id] = clean
        return clean

    def revive(
        self,
        burial_id: str,
        *,
        because: str,
        premise_change_evidence: str,
        actor: str = "research",
    ) -> BurialRecord:
        """Revive only with a materially new premise (ramanujan semantics)."""
        if burial_id not in self.burials:
            raise GraveyardError(f"unknown burial {burial_id!r}")
        current = self.burials[burial_id]
        if not current.buried:
            raise GraveyardError(f"burial {burial_id!r} is not currently buried")
        if not because.strip():
            raise GraveyardError("revival requires a non-empty because")
        if not premise_change_evidence.strip():
            raise GraveyardError(
                "cannot resurrect without premise_change_evidence "
                "(verifier result or literature change after burial); "
                "free resurrection is refused — ramanujan Stores.revive pattern"
            )
        revived = BurialRecord(
            burial_id=current.burial_id,
            mechanism=current.mechanism,
            model_geometry=current.model_geometry,
            measured_outcome=current.measured_outcome,
            failure_reason=current.failure_reason,
            reopen_condition=current.reopen_condition,
            failure_class=current.failure_class,
            related_research_item_ids=current.related_research_item_ids,
            citations=current.citations,
            actor=actor,
            buried=False,
            premise_change_evidence=premise_change_evidence.strip(),
            revived_because=because.strip(),
        )
        self.burials[burial_id] = revived
        return revived

    def active(self) -> list[BurialRecord]:
        return [b for b in self.burials.values() if b.buried]

    def by_failure_class(self, failure_class: FailureClass) -> list[BurialRecord]:
        return [b for b in self.active() if b.failure_class is failure_class]

    def by_mechanism(self, mechanism: str) -> list[BurialRecord]:
        key = mechanism.strip().lower()
        return [b for b in self.active() if b.mechanism.strip().lower() == key]

    def check_proposal(
        self,
        *,
        mechanism: str,
        new_premise: str | None = None,
    ) -> dict[str, Any]:
        """Bible §32: every proposal checks the Graveyard first.

        Returns a machine-readable gate result. If matching active burials
        exist and ``new_premise`` is absent/blank, status is REFUSED.
        """
        hits = self.by_mechanism(mechanism)
        if not hits:
            return {
                "status": "CLEAR",
                "mechanism": mechanism,
                "matching_burials": [],
                "may_proceed": True,
            }
        if not new_premise or not new_premise.strip():
            return {
                "status": "REFUSED",
                "mechanism": mechanism,
                "matching_burials": [b.as_dict() for b in hits],
                "may_proceed": False,
                "reason": (
                    "mechanism matches active graveyard burials; "
                    "supply a materially new premise or do not repeat"
                ),
                "reopen_conditions": [b.reopen_condition for b in hits],
            }
        return {
            "status": "PREMISE_REVIEW_REQUIRED",
            "mechanism": mechanism,
            "matching_burials": [b.as_dict() for b in hits],
            "may_proceed": False,  # human/tribunal still required; scaffold never auto-clears
            "proposed_new_premise": new_premise.strip(),
            "reason": (
                "materially new premise claimed; tribunal/human gate must confirm "
                "before re-running a buried mechanism (ramanujan tribunal kinship)"
            ),
            "reopen_conditions": [b.reopen_condition for b in hits],
        }

    def seed_known_classes_index(self) -> dict[str, str]:
        """Return the Bible §32 class catalogue for docs and preflight UIs."""
        return {fc.value: fc.name for fc in BIBLE_FAILURE_CLASSES}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "hawking.ascension.graveyard.v1",
            "burial_count": len(self.burials),
            "active_count": len(self.active()),
            "known_failure_classes": [fc.value for fc in BIBLE_FAILURE_CLASSES],
            "semantics": {
                "nothing_deleted": True,
                "free_resurrection_refused": True,
                "premise_change_required_to_revive": True,
                "ramanujan_stores_write_paths_untouched": True,
                "adapts": "ramanujan.scaffold.core.stores.Stores.bury/revive + Odyssey reopen_condition",
            },
            "burials": [b.as_dict() for b in self.burials.values()],
        }


def burial_from_mapping(value: Mapping[str, Any]) -> BurialRecord:
    fc_raw = value.get("failure_class", FailureClass.OTHER.value)
    if isinstance(fc_raw, FailureClass):
        fc = fc_raw
    else:
        fc = FailureClass(str(fc_raw))
    return BurialRecord(
        burial_id=str(value["burial_id"]),
        mechanism=str(value["mechanism"]),
        model_geometry=str(value["model_geometry"]),
        measured_outcome=str(value["measured_outcome"]),
        failure_reason=str(value["failure_reason"]),
        reopen_condition=str(value["reopen_condition"]),
        failure_class=fc,
        related_research_item_ids=tuple(value.get("related_research_item_ids") or ()),
        citations=tuple(value.get("citations") or ()),
        actor=str(value.get("actor") or "adversary"),
        buried=bool(value.get("buried", True)),
        premise_change_evidence=value.get("premise_change_evidence"),
        revived_because=value.get("revived_because"),
    )


def ensure_graveyard_checked(
    graveyard: AscensionGraveyard,
    *,
    mechanism: str,
    new_premise: str | None = None,
) -> dict[str, Any]:
    """Convenience gate used by research-registry admission paths."""
    result = graveyard.check_proposal(mechanism=mechanism, new_premise=new_premise)
    if result["status"] == "REFUSED":
        raise GraveyardError(result["reason"])
    return result
