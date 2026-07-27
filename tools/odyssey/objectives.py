#!/usr/bin/env python3.12
"""Objective registry for the Odyssey trainer apparatus.

Objectives are registered by name with an explicit schema. An unregistered
objective is REFUSED — never defaulted. This is the law the dry-run skeleton
could not enforce because it never ran a trainer.
"""
from __future__ import annotations

from typing import Any, Callable

SCHEMA = "hawking.odyssey.objective_registry.v1"


class UnregisteredObjective(RuntimeError):
    """Raised when a caller asks for an objective that is not in the registry."""


class ObjectiveRefused(RuntimeError):
    """Raised when an objective is known but not admissible for the requested stage."""


# Each entry: name -> {schema, stages, description, kind, loss_fn_name, notes}
_REGISTRY: dict[str, dict[str, Any]] = {
    "capability_weighted_ce": {
        "schema": "hawking.odyssey.objective.capability_weighted_ce.v1",
        "stages": ["T1", "FIXTURE"],
        "kind": "loss",
        "description": "Capability-weighted cross-entropy under the selected profile.",
        "implements": "toy_ce_step",
        "note": "Full CE against real corpora is gated by data acquisition; toy step is FIXTURE only.",
    },
    "fake_qat_distill": {
        "schema": "hawking.odyssey.objective.fake_qat_distill.v1",
        "stages": ["T2", "FIXTURE"],
        "kind": "fake_simulation",
        "description": "FIXTURE: fake-quant distillation operator on small tensors. Not a measurement.",
        "implements": "fake_qat_step",
        "note": "Honestly named fake. Results must never be reported as QAT measurements on a parent.",
    },
    "trajectory_divergence": {
        "schema": "hawking.odyssey.objective.trajectory_divergence.v1",
        "stages": ["T3", "FIXTURE"],
        "kind": "interface",
        "description": "Trajectory divergence penalty against parent traces.",
        "implements": "trajectory_loss_shape",
        "note": (
            "Real T3 needs parent trajectory traces. Teacher ledger today: 122 lines, "
            "118 per-layer captures, 0 trajectory traces. Interface only until traces exist."
        ),
    },
    "false_refusal_reduction": {
        "schema": "hawking.odyssey.objective.false_refusal.v1",
        "stages": ["T4"],
        "kind": "declared",
        "description": "False-refusal reduction on verified permitted tasks.",
        "implements": None,
        "note": "Requires sovereignty corpus (DECLARED_NOT_PRESENT) and a served capable model.",
    },
}


def list_objectives() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_objective(name: str) -> dict[str, Any]:
    """Return the registered objective schema, or refuse."""
    if name not in _REGISTRY:
        raise UnregisteredObjective(
            f"objective {name!r} is not registered; refusing rather than defaulting. "
            f"known={list_objectives()}"
        )
    entry = dict(_REGISTRY[name])
    entry["name"] = name
    entry["registry_schema"] = SCHEMA
    return entry


def require_objective(name: str, *, stage: str | None = None) -> dict[str, Any]:
    """Resolve and optionally check stage admissibility.

    An objective is admissible for `stage` only if `stage` appears in its
    registered stages list. There is no silent default to another objective.
    """
    entry = get_objective(name)
    if stage is not None and stage not in entry["stages"]:
        raise ObjectiveRefused(
            f"objective {name!r} not admissible for stage {stage!r}; "
            f"allowed={entry['stages']}"
        )
    return entry


def register_objective(name: str, entry: dict[str, Any], *, overwrite: bool = False) -> None:
    """Test/extension hook. Production path uses the sealed registry above."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"objective {name!r} already registered")
    required = {"schema", "stages", "kind", "description"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"objective entry missing {missing}")
    _REGISTRY[name] = dict(entry)


def unregister_objective(name: str) -> None:
    """Test hook only."""
    _REGISTRY.pop(name, None)
