#!/usr/bin/env python3.12
"""Declarative experiment / campaign specification schema.

A retired campaign keeps an ExperimentSpec, a small fixture, its receipt, a
reproduction command, and a reopen condition. It does not keep a bespoke
controller: the engine runs the spec.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "hawking.condense.experiment_spec.v1"
SPECS_DIR = Path(__file__).resolve().parent / "specs"


class SpecError(ValueError):
    """Raised when a spec is malformed or incomplete."""


class CampaignPhase(str, Enum):
    """Canonical lifecycle phases — one set for every campaign."""

    PRECHECK = "precheck"
    MEASURE = "measure"
    ALLOCATE = "allocate"
    PACK = "pack"
    SEAL = "seal"
    MONITOR = "monitor"
    RESUME = "resume"
    REPORT = "report"


class ResourceClass(str, Enum):
    LIGHT_READONLY = "light-readonly"
    LIGHT = "light"
    HEAVY = "heavy"
    GPU_LAB = "gpu-lab"
    NETWORK = "network"
    DETACHED = "detached"


# Legal phase order for a forward campaign (resume is orthogonal).
DEFAULT_PHASE_ORDER: tuple[CampaignPhase, ...] = (
    CampaignPhase.PRECHECK,
    CampaignPhase.MEASURE,
    CampaignPhase.ALLOCATE,
    CampaignPhase.PACK,
    CampaignPhase.SEAL,
    CampaignPhase.MONITOR,
    CampaignPhase.REPORT,
)


@dataclass(frozen=True)
class StepSpec:
    """One unit of work inside a phase."""

    id: str
    phase: str
    description: str = ""
    # Optional handler key resolved by the runtime registry; empty = pure record.
    handler: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    idempotent: bool = True
    optional: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "description": self.description,
            "handler": self.handler,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "idempotent": self.idempotent,
            "optional": self.optional,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StepSpec":
        if not isinstance(raw, Mapping):
            raise SpecError("step must be an object")
        sid = raw.get("id")
        phase = raw.get("phase")
        if not isinstance(sid, str) or not sid:
            raise SpecError("step.id must be a non-empty string")
        if not isinstance(phase, str) or not phase:
            raise SpecError(f"step {sid!r}: phase must be a non-empty string")
        try:
            CampaignPhase(phase)
        except ValueError as exc:
            raise SpecError(
                f"step {sid!r}: unknown phase {phase!r}; "
                f"allowed={[p.value for p in CampaignPhase]}"
            ) from exc
        params = raw.get("params") or {}
        if not isinstance(params, Mapping):
            raise SpecError(f"step {sid!r}: params must be an object")
        return cls(
            id=sid,
            phase=phase,
            description=str(raw.get("description") or ""),
            handler=str(raw.get("handler") or ""),
            inputs=tuple(str(x) for x in (raw.get("inputs") or ())),
            outputs=tuple(str(x) for x in (raw.get("outputs") or ())),
            idempotent=bool(raw.get("idempotent", True)),
            optional=bool(raw.get("optional", False)),
            params=dict(params),
        )


@dataclass(frozen=True)
class ReopenCondition:
    """When a sealed-negative / retired campaign may be reopened."""

    id: str
    description: str
    # Machine-checkable predicate name (registry) or free-text gate.
    predicate: str = ""
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "predicate": self.predicate,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReopenCondition":
        if not isinstance(raw, Mapping):
            raise SpecError("reopen_condition must be an object")
        rid = raw.get("id")
        if not isinstance(rid, str) or not rid:
            raise SpecError("reopen_condition.id must be a non-empty string")
        return cls(
            id=rid,
            description=str(raw.get("description") or ""),
            predicate=str(raw.get("predicate") or ""),
            evidence=tuple(str(x) for x in (raw.get("evidence") or ())),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """Full declarative campaign / experiment specification."""

    schema: str
    campaign_id: str
    title: str
    family: str
    status: str  # live | retired | sealed_negative | fixture_only
    resource_class: str
    phases: tuple[str, ...]
    steps: tuple[StepSpec, ...]
    receipt: str = ""
    fixture: str = ""
    reproduction: str = ""
    reopen: tuple[ReopenCondition, ...] = ()
    lease_name: str = ""
    checkpoint_name: str = ""
    authorization_fences: tuple[str, ...] = ()
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "title": self.title,
            "family": self.family,
            "status": self.status,
            "resource_class": self.resource_class,
            "phases": list(self.phases),
            "steps": [s.to_dict() for s in self.steps],
            "receipt": self.receipt,
            "fixture": self.fixture,
            "reproduction": self.reproduction,
            "reopen": [r.to_dict() for r in self.reopen],
            "lease_name": self.lease_name,
            "checkpoint_name": self.checkpoint_name,
            "authorization_fences": list(self.authorization_fences),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    def steps_for(self, phase: str | CampaignPhase) -> list[StepSpec]:
        name = phase.value if isinstance(phase, CampaignPhase) else phase
        return [s for s in self.steps if s.phase == name]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentSpec":
        return validate_spec(raw)


_ALLOWED_STATUS = frozenset(
    {"live", "retired", "sealed_negative", "fixture_only", "historical"}
)


def validate_spec(raw: Mapping[str, Any]) -> ExperimentSpec:
    """Validate and return a typed ExperimentSpec. Fail closed on drift."""
    if not isinstance(raw, Mapping):
        raise SpecError("spec root must be an object")
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise SpecError(f"unsupported schema {schema!r}; expected {SCHEMA!r}")
    campaign_id = raw.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise SpecError("campaign_id must be a non-empty string")
    title = str(raw.get("title") or campaign_id)
    family = str(raw.get("family") or "unknown")
    status = str(raw.get("status") or "historical")
    if status not in _ALLOWED_STATUS:
        raise SpecError(
            f"status {status!r} not in {sorted(_ALLOWED_STATUS)}"
        )
    resource_class = str(raw.get("resource_class") or ResourceClass.LIGHT.value)
    try:
        ResourceClass(resource_class)
    except ValueError as exc:
        raise SpecError(f"unknown resource_class {resource_class!r}") from exc

    phases_raw = raw.get("phases")
    if not isinstance(phases_raw, Sequence) or isinstance(phases_raw, (str, bytes)):
        raise SpecError("phases must be a list of phase names")
    phases: list[str] = []
    for item in phases_raw:
        if not isinstance(item, str):
            raise SpecError("each phase must be a string")
        try:
            CampaignPhase(item)
        except ValueError as exc:
            raise SpecError(f"unknown phase {item!r}") from exc
        phases.append(item)
    if not phases:
        raise SpecError("phases must be non-empty")

    steps_raw = raw.get("steps") or []
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        raise SpecError("steps must be a list")
    steps = tuple(StepSpec.from_dict(s) for s in steps_raw)
    seen_ids: set[str] = set()
    for step in steps:
        if step.id in seen_ids:
            raise SpecError(f"duplicate step id {step.id!r}")
        seen_ids.add(step.id)
        if step.phase not in phases and step.phase != CampaignPhase.RESUME.value:
            raise SpecError(
                f"step {step.id!r} phase {step.phase!r} not listed in phases"
            )

    reopen_raw = raw.get("reopen") or []
    if not isinstance(reopen_raw, Sequence) or isinstance(reopen_raw, (str, bytes)):
        raise SpecError("reopen must be a list")
    reopen = tuple(ReopenCondition.from_dict(r) for r in reopen_raw)

    fences = tuple(str(x) for x in (raw.get("authorization_fences") or ()))
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise SpecError("metadata must be an object")

    return ExperimentSpec(
        schema=SCHEMA,
        campaign_id=campaign_id,
        title=title,
        family=family,
        status=status,
        resource_class=resource_class,
        phases=tuple(phases),
        steps=steps,
        receipt=str(raw.get("receipt") or ""),
        fixture=str(raw.get("fixture") or ""),
        reproduction=str(raw.get("reproduction") or ""),
        reopen=reopen,
        lease_name=str(raw.get("lease_name") or f"{campaign_id}.lease"),
        checkpoint_name=str(
            raw.get("checkpoint_name") or f"{campaign_id}.checkpoint.json"
        ),
        authorization_fences=fences,
        notes=str(raw.get("notes") or ""),
        metadata=dict(metadata),
    )


def load_spec(raw: Mapping[str, Any] | str | Path) -> ExperimentSpec:
    """Load from mapping or JSON path."""
    if isinstance(raw, (str, Path)):
        return load_spec_path(Path(raw))
    return validate_spec(raw)


def load_spec_path(path: Path) -> ExperimentSpec:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read spec {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SpecError(f"spec root must be an object: {path}")
    return validate_spec(data)


def list_specs(directory: Path | None = None) -> list[Path]:
    directory = Path(directory) if directory else SPECS_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def load_all_specs(directory: Path | None = None) -> list[ExperimentSpec]:
    return [load_spec_path(p) for p in list_specs(directory)]
