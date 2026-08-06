"""Research registry: every item ends ADMIT_TO_* / DEFER / REJECT (Bible §4).

Generalises the research-verdict shape already used in tonight's
``FRANKENSTEIN_ARCHITECTURE_OPTIONS.md``:

  technique → mechanism/hypothesis
  feasibility against constraints → expected B/F/U/R/D/S/K + capability risk
  verdict (PRIMARY / BEST V1+ / DEFER / RULED OUT / NOT APPLICABLE / POOR FIT)
    → ADMIT_TO_GRAVITY | ADMIT_TO_RUNTIME | ADMIT_TO_KERNEL | DEFER | REJECT
  citations / local sealed contracts → source geometry + prototype refs
  reopen language ("only worth building if V0 …") → reopen_condition

The research phase is complete only when every mechanism that could materially
affect Gravity packing has a decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class ResearchRegistryError(RuntimeError):
    """Research registry invariant failed."""


class ResearchVerdict(str, Enum):
    ADMIT_TO_GRAVITY = "ADMIT_TO_GRAVITY"
    ADMIT_TO_RUNTIME = "ADMIT_TO_RUNTIME"
    ADMIT_TO_KERNEL = "ADMIT_TO_KERNEL"
    DEFER = "DEFER"
    REJECT = "REJECT"


# B/F/U/R/D/S/K — bandwidth, flops, utilisation, reuse, depth, synchronisation, kernel
BFURDSK_AXES = ("B", "F", "U", "R", "D", "S", "K")


@dataclass(frozen=True)
class BFURDSKEffect:
    """Expected effect on each B/F/U/R/D/S/K axis (qualitative or numeric)."""

    B: str = "unmeasured"
    F: str = "unmeasured"
    U: str = "unmeasured"
    R: str = "unmeasured"
    D: str = "unmeasured"
    S: str = "unmeasured"
    K: str = "unmeasured"

    def __post_init__(self) -> None:
        for axis in BFURDSK_AXES:
            value = getattr(self, axis)
            if not isinstance(value, str) or not value.strip():
                raise ResearchRegistryError(f"BFURDSK axis {axis} must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return {axis: getattr(self, axis) for axis in BFURDSK_AXES}


@dataclass(frozen=True)
class ResearchItem:
    """One research decision with the full Bible §4 field set."""

    item_id: str
    mechanism: str
    hypothesis: str
    expected_bfurdsk: BFURDSKEffect
    source_geometry: str
    prototype: str
    measured_result: str
    capability_risk: str
    gravity_implication: str
    runtime_implication: str
    reopen_condition: str
    verdict: ResearchVerdict
    # Extra provenance, generalising FRANKENSTEIN_ARCHITECTURE_OPTIONS citations
    citations: tuple[str, ...] = ()
    constraints_checked: tuple[str, ...] = ()
    feasibility_notes: str = ""
    actor: str = "research"
    related_graveyard_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label in (
            "item_id",
            "mechanism",
            "hypothesis",
            "source_geometry",
            "prototype",
            "measured_result",
            "capability_risk",
            "gravity_implication",
            "runtime_implication",
            "reopen_condition",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ResearchRegistryError(f"{label} must be a non-empty string")
        if not isinstance(self.verdict, ResearchVerdict):
            raise ResearchRegistryError("verdict must be a ResearchVerdict")
        if not isinstance(self.expected_bfurdsk, BFURDSKEffect):
            raise ResearchRegistryError("expected_bfurdsk must be a BFURDSKEffect")
        if self.verdict is ResearchVerdict.REJECT and self.reopen_condition.strip().lower() in {
            "n/a",
            "none",
            "never",
        }:
            # Reject still needs a reopen condition per Bible §4 / §32 — "never"
            # is allowed only when explicit and non-empty; the above rejects
            # underspecified placeholders that are not real conditions.
            if self.reopen_condition.strip().lower() != "never":
                raise ResearchRegistryError(
                    "REJECT requires an explicit reopen_condition "
                    "(use 'never' only when permanently closed)"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "hawking.ascension.research_item.v1",
            "item_id": self.item_id,
            "mechanism": self.mechanism,
            "hypothesis": self.hypothesis,
            "expected_bfurdsk_effect": self.expected_bfurdsk.as_dict(),
            "source_geometry": self.source_geometry,
            "prototype": self.prototype,
            "measured_result": self.measured_result,
            "capability_risk": self.capability_risk,
            "gravity_implication": self.gravity_implication,
            "runtime_implication": self.runtime_implication,
            "reopen_condition": self.reopen_condition,
            "verdict": self.verdict.value,
            "citations": list(self.citations),
            "constraints_checked": list(self.constraints_checked),
            "feasibility_notes": self.feasibility_notes,
            "actor": self.actor,
            "related_graveyard_ids": list(self.related_graveyard_ids),
        }


# Map FRANKENSTEIN_ARCHITECTURE_OPTIONS free-text verdicts onto Bible §4 enums.
FRANKENSTEIN_VERDICT_MAP: dict[str, ResearchVerdict] = {
    "PRIMARY": ResearchVerdict.ADMIT_TO_RUNTIME,
    "BEST POST-V0 UPGRADE": ResearchVerdict.DEFER,
    "BEST V1+": ResearchVerdict.DEFER,
    "HIGH RISK / DEFER": ResearchVerdict.DEFER,
    "DEFER": ResearchVerdict.DEFER,
    "RULED OUT": ResearchVerdict.REJECT,
    "NOT APPLICABLE": ResearchVerdict.REJECT,
    "POOR FIT": ResearchVerdict.REJECT,
    "OUT": ResearchVerdict.REJECT,
    "OPTIONAL LATER": ResearchVerdict.DEFER,
}


def map_frankenstein_verdict(text: str) -> ResearchVerdict:
    """Best-effort map from tonight's research-doc wording to §4 verdicts."""
    key = text.strip().upper()
    if key in FRANKENSTEIN_VERDICT_MAP:
        return FRANKENSTEIN_VERDICT_MAP[key]
    for prefix, verdict in FRANKENSTEIN_VERDICT_MAP.items():
        if key.startswith(prefix):
            return verdict
    raise ResearchRegistryError(f"unmapped frankenstein verdict text: {text!r}")


@dataclass
class ResearchRegistry:
    """In-memory research registry. Durable store is a future seal path."""

    items: dict[str, ResearchItem] = field(default_factory=dict)

    def record(self, item: ResearchItem) -> ResearchItem:
        if item.item_id in self.items:
            raise ResearchRegistryError(
                f"research item {item.item_id!r} already recorded; "
                "amend via a new item_id or explicit supersede path"
            )
        self.items[item.item_id] = item
        return item

    def get(self, item_id: str) -> ResearchItem:
        try:
            return self.items[item_id]
        except KeyError as exc:
            raise ResearchRegistryError(f"unknown research item {item_id!r}") from exc

    def by_verdict(self, verdict: ResearchVerdict) -> list[ResearchItem]:
        return [i for i in self.items.values() if i.verdict is verdict]

    def incomplete_mechanisms(self, required: Iterable[str]) -> list[str]:
        """Return required mechanism names still lacking a decision."""
        decided = {i.mechanism for i in self.items.values()}
        return [m for m in required if m not in decided]

    def research_phase_complete(self, required_mechanisms: Iterable[str]) -> bool:
        """Bible §4: complete only when all material Gravity mechanisms have a decision."""
        return not self.incomplete_mechanisms(required_mechanisms)

    def as_dict(self) -> dict[str, Any]:
        counts = {v.value: 0 for v in ResearchVerdict}
        for item in self.items.values():
            counts[item.verdict.value] += 1
        return {
            "schema": "hawking.ascension.research_registry.v1",
            "item_count": len(self.items),
            "counts_by_verdict": counts,
            "items": [i.as_dict() for i in self.items.values()],
        }


def item_from_mapping(value: Mapping[str, Any]) -> ResearchItem:
    """Load a ResearchItem from a JSON-shaped mapping."""
    bf = value.get("expected_bfurdsk_effect") or value.get("expected_bfurdsk") or {}
    if not isinstance(bf, Mapping):
        raise ResearchRegistryError("expected_bfurdsk_effect must be a mapping")
    effect = BFURDSKEffect(**{axis: str(bf.get(axis, "unmeasured")) for axis in BFURDSK_AXES})
    verdict_raw = value.get("verdict")
    if isinstance(verdict_raw, ResearchVerdict):
        verdict = verdict_raw
    else:
        verdict = ResearchVerdict(str(verdict_raw))
    return ResearchItem(
        item_id=str(value["item_id"]),
        mechanism=str(value["mechanism"]),
        hypothesis=str(value["hypothesis"]),
        expected_bfurdsk=effect,
        source_geometry=str(value["source_geometry"]),
        prototype=str(value["prototype"]),
        measured_result=str(value["measured_result"]),
        capability_risk=str(value["capability_risk"]),
        gravity_implication=str(value["gravity_implication"]),
        runtime_implication=str(value["runtime_implication"]),
        reopen_condition=str(value["reopen_condition"]),
        verdict=verdict,
        citations=tuple(value.get("citations") or ()),
        constraints_checked=tuple(value.get("constraints_checked") or ()),
        feasibility_notes=str(value.get("feasibility_notes") or ""),
        actor=str(value.get("actor") or "research"),
        related_graveyard_ids=tuple(value.get("related_graveyard_ids") or ()),
    )
