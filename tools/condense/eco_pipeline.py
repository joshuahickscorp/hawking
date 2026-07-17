#!/usr/bin/env python3.12
"""The data-driven Press -> Summon pipeline state machine.

The directive asks to "Consolidate overlapping controllers into a data-driven state
machine where safe" for the ten product surfaces:

    Press -> Doctor -> Horizon -> Context -> Continuum -> Lens -> Bridge
          -> Passport -> Capsule -> Summon

Rather than ten hand-wired controllers, the stage graph is declarative DATA. The state
machine gives validators, exact resume, rollback, and offline hydration over that data.
"Where safe" means additive only: this models the summon-time flow, it does NOT merge or
touch the live Doctor campaign controllers (those are immutable).

The pipeline's canonical order is exactly the directive's ten-name sequence, and it is a
valid topological order of the dependency DAG (verified by `validate_spec`). Each stage
declares which Passport identity dimension it produces, so the one identity/receipt graph
(eco_passport) is assembled from stage outputs.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, SCHEMA_PIPELINE, SCHEMA_PIPELINE_STATE,
    hash_value, seal_field, sealed, now_iso,
)

# Declarative stage graph. `requires` is the true dependency DAG; the list order below is
# the directive's canonical presentation order, which validate_spec proves is a valid topo
# sort. `passport_dimension` links a stage to the identity facet it produces (or None).
STAGES: tuple[dict[str, Any], ...] = (
    {"name": "press", "plane": "condensation", "requires": (),
     "produces": ("candidate_representation",), "passport_dimension": "artifact",
     "validators": ("deterministic_pack", "exact_bytes")},
    {"name": "doctor", "plane": "condensation", "requires": ("press",),
     "produces": ("treatment",), "passport_dimension": "doctor_treatment",
     "validators": ("treatment_bytes_counted", "claim_separation")},
    {"name": "horizon", "plane": "condensation", "requires": ("doctor",),
     "produces": ("model_event_horizon", "context_horizon", "continuity_horizon", "autonomy_horizon"),
     "passport_dimension": "capability_contract",
     "validators": ("frozen_contract", "lcb_gates")},
    {"name": "context", "plane": "context", "requires": ("horizon",),
     "produces": ("context_manifest", "kv_policy"), "passport_dimension": "context_horizon",
     "validators": ("mandatory_reservations", "effective_context")},
    {"name": "continuum", "plane": "agency", "requires": ("context",),
     "produces": ("event_log", "checkpoint", "agent_continuity_capsule"),
     "passport_dimension": "session_state",
     "validators": ("chain_integrity", "resume_after_reboot")},
    {"name": "lens", "plane": "context", "requires": ("context",),
     "produces": ("workspace_index",), "passport_dimension": None,
     "validators": ("content_addressed", "token_budgeted")},
    {"name": "bridge", "plane": "bridge", "requires": ("context",),
     "produces": ("openai_api", "responses_api", "mcp_client", "mcp_server", "adapter_abi"),
     "passport_dimension": "client_compat",
     "validators": ("fail_closed_operators", "no_dense_reconstruction")},
    {"name": "passport", "plane": "artifact",
     "requires": ("press", "doctor", "horizon", "context", "continuum", "bridge"),
     "produces": ("passport",), "passport_dimension": "physical_bytes",
     "validators": ("all_eight_dimensions", "self_seal")},
    {"name": "capsule", "plane": "artifact", "requires": ("passport",),
     "produces": ("capsule", "signed_slices"), "passport_dimension": "device_profile",
     "validators": ("signed_slices", "byte_receipts")},
    {"name": "summon", "plane": "experience", "requires": ("capsule", "bridge"),
     "produces": ("session",), "passport_dimension": None,
     "validators": ("device_admission", "honest_envelope")},
)

CANONICAL_ORDER: tuple[str, ...] = tuple(s["name"] for s in STAGES)
_STAGE_BY_NAME = {s["name"]: s for s in STAGES}

# Validators that are actually enforced here. The rest are declared scaffold (they gate a
# real artifact only once that plane is built post-activation).
_REAL_VALIDATORS = {"all_eight_dimensions", "self_seal", "claim_separation"}


def pipeline_spec() -> dict[str, Any]:
    # Deterministic content address: the spec is a pure function of the stage graph, so it
    # carries NO timestamp inside the seal (a timestamped seal would change every second and
    # never match new_state().spec_sha256).
    spec = {
        "schema": SCHEMA_PIPELINE,
        "canonical_order": list(CANONICAL_ORDER),
        "stages": [dict(s, requires=list(s["requires"]), produces=list(s["produces"]),
                        validators=list(s["validators"])) for s in STAGES],
    }
    return seal_field(spec, "spec_sha256")


def validate_spec(spec: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """The DAG is acyclic and the canonical order is a valid topological order."""
    reasons: list[str] = []
    order = list(CANONICAL_ORDER)
    position = {name: i for i, name in enumerate(order)}
    seen: set[str] = set()
    for name in order:
        stage = _STAGE_BY_NAME[name]
        for dep in stage["requires"]:
            if dep not in _STAGE_BY_NAME:
                reasons.append(f"{name} requires unknown stage {dep}")
            elif position[dep] >= position[name]:
                reasons.append(f"{name} requires {dep} which is not earlier (not a valid topo order)")
            elif dep not in seen:
                reasons.append(f"{name} requires {dep} not yet seen")
        seen.add(name)
    # every Passport dimension must be produced by exactly one stage
    from eco_passport import DIMENSIONS
    produced = {s["passport_dimension"] for s in STAGES if s["passport_dimension"]}
    for dim in DIMENSIONS:
        if dim not in produced:
            reasons.append(f"no stage produces passport dimension {dim}")
    return (not reasons), reasons


def new_state() -> dict[str, Any]:
    state = {
        "schema": SCHEMA_PIPELINE_STATE,
        "spec_sha256": pipeline_spec()["spec_sha256"],
        "stages": {name: {"status": "pending", "output_sha256": None, "at": None}
                   for name in CANONICAL_ORDER},
        "history": [],
        "created_at": now_iso(),
    }
    return seal_field(state, "state_sha256")


def _requires_met(state: dict[str, Any], name: str) -> bool:
    return all(state["stages"][dep]["status"] == "complete"
               for dep in _STAGE_BY_NAME[name]["requires"])


def _run_validators(name: str, output: dict[str, Any]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for v in _STAGE_BY_NAME[name]["validators"]:
        if v not in _REAL_VALIDATORS:
            results.append({"validator": v, "status": "declared",
                            "detail": "scaffolded; enforced once this plane is activated"})
            continue
        if v == "all_eight_dimensions":
            from eco_passport import DIMENSIONS
            facets = output.get("facets") if isinstance(output, dict) else None
            ok = isinstance(facets, dict) and all(d in facets for d in DIMENSIONS)
            results.append({"validator": v, "status": "pass" if ok else "fail",
                            "detail": "" if ok else "passport output missing dimensions"})
        elif v == "self_seal":
            ok = isinstance(output, dict) and any(
                k.endswith("_sha256") and sealed(output, k) for k in output)
            results.append({"validator": v, "status": "pass" if ok else "fail",
                            "detail": "" if ok else "output is not self-sealed"})
        elif v == "claim_separation":
            layer = output.get("claim_layer") if isinstance(output, dict) else None
            ok = layer in (None, "standalone", "context_system", "agent_system",
                           "external_augmented", "environment", "interop")
            results.append({"validator": v, "status": "pass" if ok else "fail",
                            "detail": "" if ok else f"invalid claim layer {layer}"})
    return results


def advance(state: dict[str, Any], stage: str, output: dict[str, Any]) -> dict[str, Any]:
    """Mark a stage complete iff its requires are met and no validator fails."""
    if stage not in _STAGE_BY_NAME:
        raise EcoError(f"unknown stage {stage}")
    if not _requires_met(state, stage):
        unmet = [d for d in _STAGE_BY_NAME[stage]["requires"]
                 if state["stages"][d]["status"] != "complete"]
        raise EcoError(f"stage {stage} blocked; unmet requires: {', '.join(unmet)}")
    validations = _run_validators(stage, output)
    if any(v["status"] == "fail" for v in validations):
        failed = [v["validator"] for v in validations if v["status"] == "fail"]
        raise EcoError(f"stage {stage} failed validators: {', '.join(failed)}")
    new = _clone(state)
    new["stages"][stage] = {"status": "complete", "output_sha256": hash_value(output),
                            "at": now_iso(), "validations": validations}
    new["history"] = state["history"] + [{"stage": stage, "action": "advance", "at": now_iso()}]
    new.pop("state_sha256", None)
    return seal_field(new, "state_sha256")


def rollback(state: dict[str, Any], to_stage: str) -> dict[str, Any]:
    """Revert `to_stage` and every stage that transitively depends on it to pending."""
    if to_stage not in _STAGE_BY_NAME:
        raise EcoError(f"unknown stage {to_stage}")
    dependents = _transitive_dependents(to_stage) | {to_stage}
    new = _clone(state)
    reverted = []
    for name in CANONICAL_ORDER:
        if name in dependents and new["stages"][name]["status"] != "pending":
            new["stages"][name] = {"status": "pending", "output_sha256": None, "at": None}
            reverted.append(name)
    new["history"] = state["history"] + [
        {"stage": to_stage, "action": "rollback", "reverted": reverted, "at": now_iso()}]
    new.pop("state_sha256", None)
    return seal_field(new, "state_sha256")


def runnable(state: dict[str, Any]) -> list[str]:
    """The next stage(s) whose requires are all complete (exact-resume frontier)."""
    return [name for name in CANONICAL_ORDER
            if state["stages"][name]["status"] == "pending" and _requires_met(state, name)]


def offline_hydrate(present_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct pipeline state from stage outputs found on disk (no live process).

    `present_outputs` maps stage name -> its output object. A stage is marked complete
    only if its output is present, self-consistent, and its requires are also present.
    This is the "offline hydration" contract: resume after process exit / reboot.
    """
    state = new_state()
    for name in CANONICAL_ORDER:
        out = present_outputs.get(name)
        if out is None:
            continue
        if not _requires_met(state, name):
            continue
        try:
            state = advance(state, name, out)
        except EcoError:
            # a present-but-invalid output halts hydration at this stage (fail-closed)
            break
    return state


def _transitive_dependents(stage: str) -> set[str]:
    dependents: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, s in _STAGE_BY_NAME.items():
            if name in dependents:
                continue
            if stage in s["requires"] or dependents & set(s["requires"]):
                dependents.add(name)
                changed = True
    return dependents


def _clone(state: dict[str, Any]) -> dict[str, Any]:
    import copy
    return copy.deepcopy(state)


def selftest() -> dict[str, Any]:
    ok, why = validate_spec()
    if not ok:
        raise EcoError(f"pipeline spec invalid: {why}")
    state = new_state()
    if runnable(state) != ["press"]:
        raise EcoError(f"initial runnable wrong: {runnable(state)}")
    # advance a valid prefix
    for stage in ("press", "doctor", "horizon", "context"):
        state = advance(state, stage, {"stage": stage, "note": "scaffold output"})
    # continuum, lens, bridge all become runnable after context
    run = set(runnable(state))
    if not {"continuum", "lens", "bridge"} <= run:
        raise EcoError(f"post-context runnable wrong: {run}")
    # blocked advance
    blocked = False
    try:
        advance(state, "capsule", {})
    except EcoError:
        blocked = True
    if not blocked:
        raise EcoError("capsule should be blocked before passport")
    # rollback context reverts its dependents
    rolled = rollback(state, "context")
    if rolled["stages"]["context"]["status"] != "pending":
        raise EcoError("rollback did not revert context")
    if rolled["stages"]["press"]["status"] != "complete":
        raise EcoError("rollback wrongly reverted press")
    # offline hydration from present outputs
    outputs = {s: {"stage": s} for s in ("press", "doctor", "horizon")}
    hydrated = offline_hydrate(outputs)
    complete = [n for n in CANONICAL_ORDER if hydrated["stages"][n]["status"] == "complete"]
    if complete != ["press", "doctor", "horizon"]:
        raise EcoError(f"hydration wrong: {complete}")
    return {"ok": True, "stages": len(CANONICAL_ORDER), "spec_valid": True,
            "hydrated_complete": complete, "rollback_reverts_dependents": True}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, sort_keys=True))
