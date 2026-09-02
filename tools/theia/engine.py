"""Theia bounty-engine execution path.

Half 1 (this module) is live: ingest a local Hawking receipt as a self-bounty,
run H.2, score with H.1. Half 2 (the model ladder) is BLOCKED_EXTERNAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import REPO, write_receipt
from tools.theia.intake import IntakeResult, run_intake
from tools.theia.labs import (
    DECLARED_IDLE_REASON,
    LABS,
    LabKind,
    SelfBountyKind,
    is_runnable,
    runnable_lab_kinds,
)
from tools.theia.ladder import STAGES, evaluate_wake
from tools.theia.self_bounty import (
    bounty_from_receipt,
    receipt_for_kind,
    value_inputs_from_receipt,
)

DEFINITION = (
    "Theia is Hawking's locally runnable generalist model trained to chase "
    "verified intellectual bounties: monetary or non-monetary problems where "
    "success can be grounded in artifacts, tests, proofs, measurements, "
    "reproductions or authorized program rules."
)
RECORDED_BY = "tools/theia/engine.py"
RECEIPT_NAME = "THEIA_BOUNTY_ENGINE.json"
DEFAULT_SELF_BOUNTY = REPO / "receipts" / "future" / "AUTONOMY_SCARS.json"

REUSE = {
    "reused": [
        "tools.future._common.write_receipt / REPO / RECEIPTS — receipt seal and write path",
        "tools.future.autonomy_scars.scars() — independent check that AUTONOMY_SCARS.json agrees with the module that produced it",
        "complete_ebpw missing-input doctrine — H.1 refuses a missing/zero factor rather than defaulting it to 1",
    ],
    "could_not_reuse": [
        "tools/odyssey/pareto_archive.py dominates()/composite_wus_per_hour_per_GB — axes are complete_ebpw, TPOT, TTFT, capability_passed, hcli_wus_per_hour for resident-body selection, not H.1 bounty factors",
        "tools/future/complete_ebpw.py cost() — bills representation parts in bytes/ms/bpw, not verified_reward/information_gain/risk",
        "tools/future/scar_reevaluator.py FAMILY_IMPL_COST_RANK — ordinal ranks over codec families, not H.1 cost terms",
        "tools/gravity_cost_vector.py / tools/cost_vector_t.py — B/M/F/L/R/T representation vectors, not bounty value",
        "tools/future/autonomy_scars.SCARS Python tuple as the artifact — the contract requires a receipts/ artifact; the JSON is the artifact, the module is only the independent verifier",
    ],
}

SECURITY_STATEMENT = {
    "network_egress": False,
    "credential_handling": False,
    "active_test": False,
    "payload_generation": False,
    "scanning": False,
    "ACTIVE_TEST": "modeled; transition refused; cannot be forced",
    "scope": (
        "immutable once pinned; loaded only from an operator-supplied "
        "authority file; never derived from bounty text; fail closed"
    ),
}


def run(
    receipt_path: Path, kind: SelfBountyKind | None = None
) -> IntakeResult:
    path = Path(receipt_path)
    bounty, resolved, doc = bounty_from_receipt(path, kind)

    def inputs(_artifact: Path):
        return value_inputs_from_receipt(path, doc, resolved)

    return run_intake(
        bounty,
        value_inputs_factory=inputs,
        self_bounty_kind=resolved,
        expected_schema=str(doc.get("schema")),
    )


def run_self_bounty_kind(kind: SelfBountyKind) -> IntakeResult:
    return run(receipt_for_kind(kind), kind)


def run_math_lab() -> IntakeResult:
    from tools.theia.math_lab import run_math_bounty

    return run_math_bounty(write=True)


def run_systems_lab() -> IntakeResult:
    from tools.theia.systems_lab import run_systems_bounty

    return run_systems_bounty(write=True)


def execute_lab(kind: LabKind) -> IntakeResult:
    if kind is LabKind.HAWKING_SELF_BOUNTY:
        return run(DEFAULT_SELF_BOUNTY)
    if kind is LabKind.MATH_FORMAL:
        return run_math_lab()
    if kind is LabKind.SYSTEMS_COMPILER:
        return run_systems_lab()
    if kind is LabKind.AUTHORIZED_SECURITY:
        raise RuntimeError(
            "AUTHORIZED SECURITY is refusal-only; ACTIVE_TEST is unimplemented"
        )
    reason = DECLARED_IDLE_REASON.get(kind, "lab is declared idle")
    raise RuntimeError(f"{kind.value} is declared, not executed: {reason}")


def ladder_snapshot() -> list[dict[str, Any]]:
    out = []
    for stage in STAGES:
        ev = evaluate_wake(stage)
        out.append(
            {
                "name": stage.name,
                "size_hint": stage.size_hint,
                "purpose": stage.purpose,
                "status": stage.status,
                "blocker": stage.blocker,
                "wake_condition": stage.wake_condition,
                "wake_satisfied": ev.satisfied,
                "wake_missing": list(ev.missing),
            }
        )
    return out


def laboratories_snapshot(
    lab_runs: Mapping[str, IntakeResult] | None = None,
) -> dict[str, Any]:
    runs = lab_runs or {}
    out: dict[str, Any] = {}
    for kind, spec in LABS.items():
        ran = runs.get(kind.value)
        out[kind.value] = {
            "executable_work": list(spec.executable_work),
            "refused_work": list(spec.refused_work),
            "bounty_classes": [c.value for c in spec.bounty_classes],
            "runnable": is_runnable(kind),
            "declared_idle_reason": DECLARED_IDLE_REASON.get(kind),
            "ran": ran is not None,
            "result": ran.to_json_dict() if ran is not None else None,
        }
    return out


def build_receipt_doc(
    result: IntakeResult,
    lab_runs: Mapping[str, IntakeResult] | None = None,
) -> dict[str, Any]:
    runs = dict(lab_runs or {})
    if result.lab and result.lab not in runs:
        runs[result.lab] = result
    kind_runs = {}
    prefix = LabKind.HAWKING_SELF_BOUNTY.value + ":"
    for name, r in runs.items():
        if name.startswith(prefix):
            kind_runs[name[len(prefix) :]] = {
                "bounty_id": r.bounty_id,
                "source": r.source,
                "exit_code": r.exit_code,
                "final_stage": r.final_stage.value,
                "schedule_score": (
                    r.schedule_score.to_json_dict()
                    if r.schedule_score is not None
                    else None
                ),
                "independent_module": (
                    (r.verified_result.detail or {}).get("independent_module")
                    if r.verified_result is not None
                    else None
                ),
            }
    return {
        "schema": "hawking.theia.bounty_engine.v1",
        "version": 1,
        "recorded_by": RECORDED_BY,
        "definition": DEFINITION,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": (
            "Bounty-engine sidecar. H.1 scores schedule work and do not "
            "declare a result true. No hardware measurement. No network "
            "egress, credential handling, scanning, payload generation, or "
            "ACTIVE_TEST. Model-ladder stages are BLOCKED_EXTERNAL."
        ),
        "halves": {
            "bounty_engine": {"status": "LIVE", "path": "tools/theia"},
            "model_ladder": {
                "status": "BLOCKED_EXTERNAL",
                "stages": [s.name for s in STAGES],
            },
        },
        "reuse": REUSE,
        "security": SECURITY_STATEMENT,
        "model_ladder": ladder_snapshot(),
        "self_bounty_run": result.to_json_dict(),
        "self_bounty_kinds": kind_runs,
        "runnable_labs": [k.value for k in runnable_lab_kinds()],
        "laboratories": laboratories_snapshot(runs),
    }


def write_engine_receipt(
    result: IntakeResult,
    lab_runs: Mapping[str, IntakeResult] | None = None,
) -> Path:
    return write_receipt(
        RECEIPT_NAME,
        build_receipt_doc(result, lab_runs=lab_runs),
        recorded_by=RECORDED_BY,
    )


def _lab_choice(value: str) -> LabKind:
    key = value.replace("-", "_").replace(" ", "_").upper()
    aliases = {
        "MATH": LabKind.MATH_FORMAL,
        "MATH_FORMAL": LabKind.MATH_FORMAL,
        "SYSTEMS": LabKind.SYSTEMS_COMPILER,
        "SYSTEMS_COMPILER": LabKind.SYSTEMS_COMPILER,
        "SELF": LabKind.HAWKING_SELF_BOUNTY,
        "HAWKING_SELF_BOUNTY": LabKind.HAWKING_SELF_BOUNTY,
        "SELF_BOUNTY": LabKind.HAWKING_SELF_BOUNTY,
    }
    if key in aliases:
        return aliases[key]
    raise argparse.ArgumentTypeError(
        f"unknown lab {value!r}; runnable: MATH_FORMAL, SYSTEMS_COMPILER, "
        "HAWKING_SELF_BOUNTY"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tools.theia")
    parser.add_argument(
        "--self-bounty",
        default=str(DEFAULT_SELF_BOUNTY),
        help="local receipts/ artifact to ingest as a Hawking self-bounty",
    )
    parser.add_argument(
        "--self-bounty-kind",
        choices=[k.value for k in SelfBountyKind],
        default=None,
        help="select a §19.12 self-bounty kind (uses that kind's default receipt unless --self-bounty is also given)",
    )
    parser.add_argument(
        "--lab",
        action="append",
        type=_lab_choice,
        default=None,
        help="run a promoted lab (repeatable): MATH_FORMAL, SYSTEMS_COMPILER, HAWKING_SELF_BOUNTY",
    )
    parser.add_argument(
        "--run-runnable",
        action="store_true",
        help="run every lab that can execute locally today (self-bounty kinds + MATH + SYSTEMS)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="do not write receipts/future/THEIA_BOUNTY_ENGINE.json",
    )
    args = parser.parse_args(argv)
    lab_runs: dict[str, IntakeResult] = {}
    primary: IntakeResult | None = None

    if args.run_runnable:
        for kind in SelfBountyKind:
            lab_runs[f"HAWKING SELF-BOUNTY:{kind.value}"] = run_self_bounty_kind(kind)
        lab_runs[LabKind.MATH_FORMAL.value] = run_math_lab()
        lab_runs[LabKind.SYSTEMS_COMPILER.value] = run_systems_lab()
        primary = lab_runs[f"HAWKING SELF-BOUNTY:{SelfBountyKind.NEGATIVE_SCIENCE.value}"]
        lab_runs[LabKind.HAWKING_SELF_BOUNTY.value] = primary
    elif args.lab:
        for kind in args.lab:
            result = execute_lab(kind)
            lab_runs[kind.value] = result
            primary = result
    elif args.self_bounty_kind:
        kind = SelfBountyKind(args.self_bounty_kind)
        path = Path(args.self_bounty)
        if str(path) == str(DEFAULT_SELF_BOUNTY):
            path = receipt_for_kind(kind)
        primary = run(path, kind)
        lab_runs[LabKind.HAWKING_SELF_BOUNTY.value] = primary
    else:
        primary = run(Path(args.self_bounty))
        lab_runs[LabKind.HAWKING_SELF_BOUNTY.value] = primary

    assert primary is not None
    payload = primary.to_json_dict()
    payload["definition"] = DEFINITION
    payload["model_ladder"] = {
        s["name"]: {
            "status": s["status"],
            "wake_condition_id": s["wake_condition"]["id"],
        }
        for s in ladder_snapshot()
    }
    payload["lab_runs"] = {
        name: {
            "lab": r.lab,
            "bounty_id": r.bounty_id,
            "exit_code": r.exit_code,
            "final_stage": r.final_stage.value,
            "schedule_value": (
                {
                    "numerator": r.schedule_score.value.numerator,
                    "denominator": r.schedule_score.value.denominator,
                }
                if r.schedule_score is not None
                else None
            ),
        }
        for name, r in lab_runs.items()
    }
    if not args.no_write:
        out = write_engine_receipt(primary, lab_runs=lab_runs)
        payload["engine_receipt"] = str(out)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if all(r.exit_code == 0 for r in lab_runs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
