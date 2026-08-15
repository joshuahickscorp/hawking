"""Global audit, external review, Apple release, and post-release frontier flow."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.operators.ascension_lifecycle import FAMILY_RULES
from lab.receipts import seal


SCHEMA = "hawking.ascension.global_release_workflow.v1"
FILENAME = "ASCENSION_V3_GLOBAL_RELEASE_WORKFLOW.json"

GLOBAL_REVIEW_PACKET_REQUIREMENTS: tuple[str, ...] = (
    "V3 constitution implemented in code",
    "manager 30B source/1.5/TG3/capability evidence",
    "manager 80B source/1.5/TG3/capability evidence",
    "manager tournament and winner",
    "alternate manager offloaded",
    "sandbox operating under the winner",
    "Agent OS production evidence",
    "evolutionary Gravity evidence",
    "exact-model compiler evidence",
    "Knowledge Plane evidence",
    "all required core families launch-qualified",
    "every advertised exact model qualified",
    "complete BPW matrix",
    "TG matrix",
    "capability matrix",
    "resource and energy evidence",
    "storage and garbage evidence",
    "recovery evidence",
    "Apple install/update/product evidence",
    "all roadblocks resolved",
    "no launch exception",
    "external review request",
)

GLOBAL_AUDIT_ARTIFACTS: tuple[str, ...] = (
    "GENERIC_HF_REFERENCE_READY",
    "ASCENSION_V3_FAMILY_MATRIX",
    "ASCENSION_V3_MODEL_MATRIX",
    "ASCENSION_V3_DENSITY_MATRIX",
    "ASCENSION_V3_TG_MATRIX",
    "ASCENSION_V3_RESOURCE_ATLAS",
    "ASCENSION_V3_POWER_PROFILES",
    "ASCENSION_V3_STORAGE_LEASES",
    "ASCENSION_V3_GARBAGE_AUDIT",
    "ASCENSION_V3_RECOVERY_TEST",
    "HAWKING_APPLE_V3_INSTALL_TEST",
    "HAWKING_APPLE_V3_UPDATE_TEST",
    "HAWKING_APPLE_V3_HCLI_PRODUCT_TEST",
    "HAWKING_APPLE_V3_RELEASE_AUDIT",
    "ASCENSION_V3_CORE_FAMILY_MATRIX_READY",
    "ASCENSION_V3_ALL_ADVERTISED_MODELS_QUALIFIED",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_release_workflow(*, states: Mapping[str, Mapping[str, Any]], launch_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Build the release pathway while refusing to infer any missing evidence."""

    families = [state for state, _completion, _family in FAMILY_RULES]
    family_statuses = {state: str(states.get(state, {}).get("status") or "ABSENT") for state in families}
    global_status = str(states.get("GLOBAL_LAUNCH_AUDIT", {}).get("status") or "ABSENT")
    external_status = str(states.get("EXTERNAL_REVIEW", {}).get("status") or "ABSENT")
    release_status = str(states.get("APPLE_RELEASE", {}).get("status") or "ABSENT")
    frontier_status = str(states.get("TG2_TG1_FRONTIER", {}).get("status") or "ABSENT")
    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_WORKFLOW_ONLY",
            "recorded_at": _utc_now(),
            "global_audit": {
                "state": "GLOBAL_LAUNCH_AUDIT",
                "status": global_status,
                "required_artifacts": list(GLOBAL_AUDIT_ARTIFACTS),
                "family_statuses": family_statuses,
                "generic_hf_reference_is_additive_not_substitution": True,
            },
            "external_review": {
                "state": "EXTERNAL_REVIEW",
                "status": external_status,
                "packet_requirements": list(GLOBAL_REVIEW_PACKET_REQUIREMENTS),
                "must_be_compact_linked_and_inspectable": True,
                "review_packet_is_not_launch_approval": True,
                "findings_must_be_repaired_or_human_waived_then_affected_gates_rerun": True,
            },
            "apple_release": {
                "state": "APPLE_RELEASE",
                "status": release_status,
                "completion_state": "HAWKING_APPLE_V3_PRODUCTION_RELEASE_READY",
                "requires_all_launch_gates_and_accepted_external_review": True,
                "no_exception_path": True,
            },
            "post_release_frontier": {
                "state": "TG2_TG1_FRONTIER",
                "status": frontier_status,
                "may_begin_only_after_apple_release_certified": True,
                "proto_frankenstein_restore_requires_mature_deepseek_path": True,
                "cuda_is_separate_funded_programme": True,
            },
            "derived_launch_gate": dict(launch_gate),
            "claim_boundary": {
                "workflow_is_not_global_audit": True,
                "workflow_is_not_external_review_acceptance": True,
                "workflow_is_not_apple_release": True,
                "no_timeline_or_exception_can_bypass_required_receipts": True,
            },
        }
    )


def write_release_workflow(
    root: str | Path, *, states: Mapping[str, Mapping[str, Any]], launch_gate: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    document = build_release_workflow(states=states, launch_gate=launch_gate)
    _atomic_json(resolved / FILENAME, document)
    return document


__all__ = [
    "FILENAME",
    "GLOBAL_AUDIT_ARTIFACTS",
    "GLOBAL_REVIEW_PACKET_REQUIREMENTS",
    "SCHEMA",
    "build_release_workflow",
    "write_release_workflow",
]
