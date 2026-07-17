#!/usr/bin/env python3.12
"""Adaptive experiment engine: acquisition, materialization, validation, ingestion.

Turns the advisory eco_planner frontier into ACTION descriptors. Master goal section 7:
the planner is one component; the engine selects the next experiment by frontier value and
information gain, materializes a source-bound executable program, validates it against the
real adapter admission, and (only for a genuinely lightweight, read-only step) dispatches it
to prove the complete lifecycle. Heavy full-model launches are GATED: they never run while
the legacy campaign is active or the adapter is not fully execution-ready.

This module launches NOTHING heavy. Its dispatch path either runs a lightweight read-only
adapter subcommand (preflight/build-spec/capabilities) or, in tests, an injected fake.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, seal_field, sealed, hash_value, now_iso  # noqa: E402

PROGRAM_SCHEMA = "hawking.successor.program.v1"
RESULT_SCHEMA = "hawking.successor.experiment_result.v1"

# Lightweight, read-only adapter subcommands that may be dispatched without heavy compute.
LIGHTWEIGHT_SUBCOMMANDS = frozenset({"capabilities", "build-spec", "preflight", "selftest"})


class EngineError(EcoError):
    """Fail-closed engine error."""


def acquisition_score(probe: dict[str, Any], *, feasible_prob: float, hv_gain: float,
                      info_gain: float, boundary_uncertainty: float, transfer: float,
                      cost: float, weights: dict[str, float] | None = None) -> float:
    """Master goal 7.4: A(e) = P(feasible)*E[dHV] + lambda_I*I + lambda_B*U_boundary
    + lambda_T*U_transfer - lambda_C*Cost. Deterministic, recorded weights."""
    w = weights or {"info": 0.5, "boundary": 0.7, "transfer": 0.2, "cost": 0.3}
    return (feasible_prob * hv_gain + w["info"] * info_gain + w["boundary"] * boundary_uncertainty
            + w["transfer"] * transfer - w["cost"] * cost)


def next_experiment(plan: dict[str, Any], *, seed: int = 20260717) -> dict[str, Any] | None:
    """Select the highest-value next experiment across all parents' boundary probes.

    Deterministic: recorded weights, a fixed seed for tie-breaking, and an explicit rationale.
    """
    candidates: list[dict[str, Any]] = []
    for parent in plan.get("parents", []):
        label = parent["binding"]["model_label"]
        params_b = parent.get("params_b") or 0.0
        for probe in parent.get("boundary_probes_needed", []):
            rate = probe["rate_bpw"]
            # Heuristic priors before a learned selector exists (7.4 permits calibrated heuristics).
            feasible = 0.9 if probe["current_verdict"] == "INCONCLUSIVE" else 0.6
            # boundary uncertainty is highest near the bracket edge
            bracket = parent.get("event_horizon_bracket", {})
            lp = bracket.get("lowest_pass_bpw")
            boundary_u = 1.0 / (1.0 + abs((lp or rate) - rate))
            info = 1.0 if probe.get("next_feasibility_tier") == "F3_full_model_quality" else 0.5
            transfer = math.log10(max(params_b, 1e-9) * 1e9) / 12.0  # bigger parents inform more
            cost = rate  # cheap proxy: higher bpw ~ more bytes/time (refined by succ_eta)
            score = acquisition_score(probe, feasible_prob=feasible, hv_gain=1.0, info_gain=info,
                                      boundary_uncertainty=boundary_u, transfer=transfer, cost=cost)
            candidates.append({
                "model_label": label, "rate_bpw": rate, "verdict": probe["current_verdict"],
                "feasibility_tier": probe.get("next_feasibility_tier"),
                "doctor_program": probe.get("doctor_program"), "acquisition": round(score, 6),
            })
    if not candidates:
        return None
    # deterministic: sort by score desc, then label, then rate; seed only documents the policy
    candidates.sort(key=lambda c: (-c["acquisition"], c["model_label"], c["rate_bpw"]))
    best = candidates[0]
    return {"selected": best, "considered": len(candidates), "policy_seed": seed,
            "weights": {"info": 0.5, "boundary": 0.7, "transfer": 0.2, "cost": 0.3},
            "rationale": "highest acquisition score across all parent boundary probes"}


def materialize_program(experiment: dict[str, Any], admission: dict[str, Any],
                        *, source_manifest_sha256: str | None,
                        controls: list[str] | None = None) -> dict[str, Any]:
    """Compile a source-bound executable program descriptor (master goal 6.1 MATERIALIZE)."""
    sel = experiment["selected"]
    program = {
        "schema": PROGRAM_SCHEMA,
        "model_label": sel["model_label"],
        "rate_bpw": sel["rate_bpw"],
        "fidelity_tier": sel.get("feasibility_tier", "F0"),
        "adapter_id": admission.get("adapter_id"),
        "adapter_source_sha256": admission.get("adapter_source_sha256"),
        "source_manifest_sha256": source_manifest_sha256,
        "doctor_program": sel.get("doctor_program"),
        "required_controls": controls or ["zero_treatment", "equal_byte_codec"],
        "created_at": now_iso(),
    }
    return seal_field(program, "program_sha256")


def validate_program(program: dict[str, Any], admission: dict[str, Any]) -> tuple[bool, list[str]]:
    """Master goal 6.1 VALIDATE_PROGRAM: source-bound + adapter admits + controls present."""
    reasons: list[str] = []
    if not sealed(program, "program_sha256"):
        reasons.append("program self-seal invalid")
    if program.get("source_manifest_sha256") is None:
        reasons.append("program is not source-bound (no source_manifest_sha256)")
    if program.get("adapter_id") != admission.get("adapter_id"):
        reasons.append("program adapter does not match admission")
    if not admission.get("ready_for_execution"):
        reasons.append(f"adapter not execution-ready: {admission.get('blockers')}")
    if not program.get("required_controls"):
        reasons.append("no causal controls declared")
    return (not reasons), reasons


def dispatch_lightweight(program: dict[str, Any], adapter_path: str, subcommand: str,
                         *, runner: Callable[[list[str]], dict[str, Any]] | None = None,
                         legacy_active: bool = True) -> dict[str, Any]:
    """Dispatch ONLY a lightweight, read-only adapter subcommand to prove the lifecycle.

    Heavy execution is refused: never dispatch a codec bake while the legacy campaign is
    active. `runner` is injectable for tests; the default runs a bounded subprocess.
    """
    if subcommand not in LIGHTWEIGHT_SUBCOMMANDS:
        raise EngineError(f"refusing non-lightweight dispatch of {subcommand!r} "
                          f"(heavy execution is gated while the campaign runs)")
    if runner is None:
        def runner(argv: list[str]) -> dict[str, Any]:  # noqa: E306
            out = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            try:
                parsed = json.loads(out.stdout) if out.stdout.strip() else {}
            except json.JSONDecodeError:
                parsed = {"stdout_head": out.stdout[:2000]}
            return {"returncode": out.returncode, "parsed": parsed}
    argv = ["python3.12", adapter_path, subcommand]
    outcome = runner(argv)
    result = {
        "schema": RESULT_SCHEMA,
        "program_sha256": program.get("program_sha256"),
        "dispatched_subcommand": subcommand,
        "lightweight": True,
        "heavy_execution_gated": legacy_active,
        "outcome": outcome,
        "at": now_iso(),
    }
    return seal_field(result, "result_sha256")


def ingest_result(frontier: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Idempotent result ingestion into a frontier ledger keyed by result_sha256."""
    ledger = dict(frontier)
    seen = ledger.setdefault("ingested", {})
    key = result.get("result_sha256")
    if key in seen:
        ledger["last_action"] = "idempotent_skip"
        return ledger
    seen[key] = {"at": now_iso(), "program": result.get("program_sha256")}
    ledger["last_action"] = "ingested"
    return ledger


def selftest() -> dict[str, Any]:
    plan = {
        "parents": [{
            "binding": {"model_label": "32B"}, "params_b": 32.5,
            "event_horizon_bracket": {"lowest_pass_bpw": None},
            "boundary_probes_needed": [
                {"rate_bpw": 4.0, "current_verdict": "INCONCLUSIVE",
                 "next_feasibility_tier": "F3_full_model_quality",
                 "doctor_program": {"promote": ["doctor_static"]}},
                {"rate_bpw": 2.0, "current_verdict": "UNPROVEN",
                 "next_feasibility_tier": "F0_byte_feasibility", "doctor_program": {"promote": []}},
            ],
        }],
    }
    pick = next_experiment(plan)
    if pick is None or pick["selected"]["model_label"] != "32B":
        raise EngineError("next_experiment failed")
    admission = {"adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
                 "adapter_source_sha256": "a" * 64, "ready_for_execution": True, "blockers": []}
    prog = materialize_program(pick, admission, source_manifest_sha256="b" * 64,
                               controls=["zero_treatment", "equal_byte_codec", "smaller_higher_bit"])
    ok, why = validate_program(prog, admission)
    if not ok:
        raise EngineError(f"program should validate: {why}")
    # not source-bound -> refused
    prog2 = dict(prog); prog2["source_manifest_sha256"] = None
    ok2, _ = validate_program(prog2, admission)
    if ok2:
        raise EngineError("unbound program must fail validation")
    # heavy dispatch refused
    heavy_refused = False
    try:
        dispatch_lightweight(prog, "adapter.py", "run", runner=lambda a: {})
    except EngineError:
        heavy_refused = True
    if not heavy_refused:
        raise EngineError("heavy dispatch not refused")
    # lightweight dispatch via injected runner proves the lifecycle
    res = dispatch_lightweight(prog, "adapter.py", "capabilities",
                               runner=lambda a: {"returncode": 0, "parsed": {"ok": True}})
    if not sealed(res, "result_sha256"):
        raise EngineError("result not sealed")
    # idempotent ingest
    f = ingest_result({}, res)
    f2 = ingest_result(f, res)
    if f["last_action"] != "ingested" or f2["last_action"] != "idempotent_skip":
        raise EngineError("ingest not idempotent")
    return {"ok": True, "acquisition_selects": pick["selected"]["model_label"],
            "source_bound_enforced": True, "heavy_dispatch_gated": True,
            "lightweight_lifecycle_proven": True, "ingest_idempotent": True}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
