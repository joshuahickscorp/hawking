"""Fail-closed KIMI -> Gravity representation -> NX promotion coordinator.

The bridge only assembles a plan. It never edits KIMI_BASE and never creates an
NX artifact while any gate is missing. Once a candidate has independent live
evidence, this is the single place that binds its operator receipt to Gravity's
Noetic Representation substrate and hands a complete manifest to
``nx_promotion.evaluate``. The historical ``noetic_family`` receipt key remains
for compatibility; Noetic is not a peer optimizer or science surface.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey import noetic_compiler as nc
from tools.future.gravity_nova_lineage import validate as validate_nova_lineage

FAMILY_ID = "kimi_operator_projection_sim"


def plan(gate: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    candidate = next(
        (row for row in gate.get("candidates", [])
         if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id),
        None,
    )
    if candidate is None:
        return {
            "status": "PROMOTION_REFUSED",
            "candidate_id": candidate_id,
            "reason": "candidate is absent from the gate receipt",
        }
    missing = list(candidate.get("missing_gates") or [])
    if gate.get("status") != "PROMOTION_ALLOWED" or missing:
        return {
            "status": "PROMOTION_REFUSED",
            "candidate_id": candidate_id,
            "missing_gates": missing or list(gate.get("missing_gates") or []),
            "noetic_family": FAMILY_ID,
            "reason": "Noetic/NX handoff is downstream of every independent gate",
            "weights_written": False,
            "artifact_created": False,
        }
    evidence = gate.get("evidence") or {}
    provenance_missing = [
        name for name, value in {
            "candidate_artifact_hash": candidate.get("artifact_hash"),
            "evaluation_receipt": evidence.get("evaluation_receipt"),
            "nova_lineage_receipt": candidate.get("nova_lineage_receipt"),
        }.items() if not value
    ]
    if provenance_missing:
        return {
            "status": "PROMOTION_REFUSED",
            "candidate_id": candidate_id,
            "missing_provenance": provenance_missing,
            "noetic_family": FAMILY_ID,
            "reason": (
                "An allowed gate is not enough: copied-artifact hash, evaluation "
                "receipt, and Nova parent-to-descendant lineage are required before "
                "a Noetic/NX manifest can be constructed."
            ),
            "weights_written": False,
            "artifact_created": False,
        }
    lineage_check = validate_nova_lineage(
        candidate["nova_lineage_receipt"],
        expected_parent_identity="KIMI_BASE",
        expected_descendant_hash=str(candidate["artifact_hash"]),
    )
    if lineage_check["status"] != "LINEAGE_ACCEPTED":
        return {
            "status": "PROMOTION_REFUSED",
            "candidate_id": candidate_id,
            "missing_provenance": ["valid_nova_lineage"],
            "nova_lineage_check": lineage_check,
            "noetic_family": FAMILY_ID,
            "reason": "Nova lineage is present but incomplete or inconsistent",
            "weights_written": False,
            "artifact_created": False,
        }
    family = nc.get_family(FAMILY_ID)
    return {
        "status": "READY_FOR_NX_MANIFEST",
        "candidate_id": candidate_id,
        "noetic_family": FAMILY_ID,
        "noetic_chain": nc.chain_status(family),
        "provenance": {
            "candidate_artifact_hash": candidate["artifact_hash"],
            "evaluation_receipt": evidence["evaluation_receipt"],
            "nova_lineage_receipt": candidate["nova_lineage_receipt"],
            "nova_lineage_check": lineage_check,
        },
        "weights_written": False,
        "artifact_created": False,
        "next": "construct a complete copied-artifact manifest and call nx_promotion.evaluate",
    }


def plan_from_path(gate_path: Path, candidate_id: str) -> dict[str, Any]:
    import json
    return plan(json.loads(gate_path.read_text()), candidate_id)


def main() -> int:
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--candidate", default="KIMI_OPERATOR_CANDIDATE_OPA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan_from_path(args.gate, args.candidate)
    result.update({
        "schema": "hawking.kimi.promotion_bridge.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_receipt": str(args.gate),
        "claim_boundary": (
            "Plan-only bridge. This command cannot write KIMI weights or create "
            "an NX artifact while the gate is refused."
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, indent=2))
    return 0 if result["status"] == "READY_FOR_NX_MANIFEST" else 2


if __name__ == "__main__":
    raise SystemExit(main())
