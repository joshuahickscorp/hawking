"""Expose a qualified KIMI candidate to Odyssey as bounded WorkUnit scope.

This is a plan-only adapter. Odyssey remains the campaign owner, HCLI remains
the authority boundary, and ModelLake remains the provenance source. A refused
KIMI promotion gate produces no KIMI worker scope.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.kimi.odyssey_bridge.v1"
DEFAULT_SCOPE = (
    "model.inspect",
    "evidence.record",
    "gravity.measure",
    "campaign.inspect",
)


def plan(
    gate: Mapping[str, Any],
    candidate_id: str,
    *,
    requested_scope: Sequence[str] | None = None,
) -> dict[str, Any]:
    candidate = next(
        (row for row in gate.get("candidates", [])
         if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id),
        None,
    )
    if candidate is None:
        return {
            "schema": SCHEMA,
            "status": "KIMI_WORKER_WITHHELD",
            "candidate_id": candidate_id,
            "reason": "candidate is absent from the promotion gate receipt",
            "scope": [],
        }
    missing = list(dict.fromkeys(
        candidate.get("missing_gates") or gate.get("missing_gates") or []
    ))
    if gate.get("status") != "PROMOTION_ALLOWED" or missing:
        return {
            "schema": SCHEMA,
            "status": "KIMI_WORKER_WITHHELD",
            "candidate_id": candidate_id,
            "missing_gates": missing,
            "scope": [],
            "odyssey_owner": "Odyssey campaign / HCLI WorkUnit emitter",
            "modellake_owner": "ModelLake seal and provenance consumer",
            "reason": (
                "KIMI may not become an Odyssey worker before the same gates that "
                "govern promotion close. Existing ModelLake WorkUnits continue."
            ),
            "external_actions": False,
        }
    scope = tuple(dict.fromkeys(str(x) for x in (requested_scope or DEFAULT_SCOPE)))
    forbidden = {"artifact.promote", "authority.widen", "external.write", "credential.use"}
    if forbidden.intersection(scope):
        return {
            "schema": SCHEMA,
            "status": "KIMI_WORKER_WITHHELD",
            "candidate_id": candidate_id,
            "missing_scope_controls": sorted(forbidden.intersection(scope)),
            "scope": [],
            "reason": "requested scope contains an authority-expanding operation",
            "external_actions": False,
        }
    return {
        "schema": SCHEMA,
        "status": "READY_FOR_BOUNDED_ODYSSEY_WORKUNITS",
        "candidate_id": candidate_id,
        "scope": list(scope),
        "odyssey_owner": "Odyssey campaign / HCLI WorkUnit emitter",
        "modellake_owner": "ModelLake seal and provenance consumer",
        "external_actions": False,
        "requires_hcli_confirmation_for": ["campaign.advance", "campaign.record"],
        "claim_boundary": (
            "Plan only. This does not execute KIMI, create WorkUnits, widen HCLI "
            "authority, write ModelLake, or promote an artifact."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--candidate", default="KIMI_OPERATOR_CANDIDATE_OPA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(json.loads(args.gate.read_text()), args.candidate)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["gate_receipt"] = str(args.gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, indent=2))
    return 0 if result["status"] == "READY_FOR_BOUNDED_ODYSSEY_WORKUNITS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
