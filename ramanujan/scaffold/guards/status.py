"""Read-only local status for the Ramanujan dependency boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ramanujan.layout import BOUNDARY_ROOT, REPO_ROOT


GATE_PATH = BOUNDARY_ROOT / "HAWKING_COMPLETION_GATE.json"
GOVERNANCE_STATUS_PATH = BOUNDARY_ROOT / "RAMANUJAN_GOVERNANCE_STATUS.json"
GREEN_LIGHT_PATH = BOUNDARY_ROOT / "RAMANUJAN_GREEN_LIGHT_TRANSITION.json"
HANDOFF_CONTRACT_PATH = REPO_ROOT / "workspace/campaign/evidence/systems/ramanujan/RAMANUJAN_HANDOFF_CONTRACT.json"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable status input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"status input is not a JSON object: {path}")
    return value


def local_status() -> dict[str, Any]:
    """Return the current non-promoting local status plane."""
    gate = _read(GATE_PATH)
    governance = _read(GOVERNANCE_STATUS_PATH)
    transition = _read(GREEN_LIGHT_PATH)
    handoff = _read(HANDOFF_CONTRACT_PATH)
    authority = gate.get("authority")
    dependency = gate.get("dependency")
    handoff_gate = handoff.get("execution_gate")
    if (
        gate.get("schema") != "hawking.ramanujan.hawking_completion_gate.v1"
        or gate.get("status") != "BLOCKED_ON_HAWKING_COMPLETION"
        or not isinstance(authority, dict)
        or authority.get("ramanujan_research_authorized") is not False
        or authority.get("production_authority") is not False
        or authority.get("self_promotion_forbidden") is not True
        or not isinstance(dependency, dict)
        or not isinstance(dependency.get("handoff_contract"), dict)
        or dependency["handoff_contract"].get("trigger") != "HAWKING_EVOLUTION_COMPLETE"
        or not isinstance(handoff_gate, dict)
        or handoff_gate.get("trigger") != "HAWKING_EVOLUTION_COMPLETE"
        or handoff.get("status") != "PREPARED_NOT_EXECUTED"
        or handoff_gate.get("may_execute_now") is not False
    ):
        raise ValueError("Hawking completion gate is malformed or grants authority")
    if governance.get("RAMANUJAN_RESEARCH_AUTHORIZED") is not False:
        raise ValueError("governance status no longer proves research authorization is false")
    if transition.get("production_authority") is not False:
        raise ValueError("green-light transition no longer proves production authority is false")
    return {
        "schema": "hawking.ramanujan.local_status.v1",
        "status": "BUILDABLE_BUT_BLOCKED_ON_HAWKING_COMPLETION",
        "hawking_completion_gate": {
            "status": gate["status"],
            "path": "ramanujan/governance/boundary/HAWKING_COMPLETION_GATE.json",
            "transition_rule": gate["transition_rule"],
        },
        "handoff": {
            "status": handoff["status"],
            "trigger": handoff_gate["trigger"],
            "may_execute_now": handoff_gate["may_execute_now"],
            "path": "workspace/campaign/evidence/systems/ramanujan/RAMANUJAN_HANDOFF_CONTRACT.json",
        },
        "authority": {
            "ramanujan_research_authorized": governance["RAMANUJAN_RESEARCH_AUTHORIZED"],
            "production_authority": transition["production_authority"],
            "self_promotion_forbidden": authority["self_promotion_forbidden"],
        },
        "safe_local_work": gate["safe_before_completion"],
        "blocked_work": gate["blocked_until_completion"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-hawking-complete",
        action="store_true",
        help="exit nonzero while the intentionally blocked Hawking gate remains closed",
    )
    args = parser.parse_args(argv)
    try:
        report = local_status()
    except ValueError as exc:
        print(f"INVALID: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 1 if args.require_hawking_complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
