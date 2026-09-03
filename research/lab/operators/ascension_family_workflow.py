"""Future-proof family-matrix workflow for the post-manager Ascension sandbox.

The V3 Bible forbids treating a generic loader, a metadata record, or one
qualified Qwen as broad production support.  This module makes every required
family, its exact research loop, selection criteria, rotation state, and
launch proof visible in one controller-owned artifact.  It contains no source
selection or model execution and cannot certify any family.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.operators.ascension_kernel_registry import FAMILY_STARTING_DOCTRINES
from lab.operators.ascension_lifecycle import FAMILY_RULES
from lab.receipts import seal


SCHEMA = "hawking.ascension.family_campaign_workflow.v1"
FILENAME = "ASCENSION_V3_FAMILY_CAMPAIGN_WORKFLOW.json"

PER_MODEL_LOOP: tuple[str, ...] = (
    "DISCOVER",
    "METADATA_PREFLIGHT",
    "FINGERPRINT",
    "SCIENTIFIC_DISTINCTION",
    "QUERY_KNOWLEDGE_PLANE",
    "CHOOSE_ACQUISITION_MODE",
    "REPRESENTATIVE_RESEARCH",
    "GRAVITY_POPULATION",
    "MINI_PACK",
    "SAMPLE_VERIFY",
    "DOCTOR_QAT",
    "FULL_PACK",
    "EXACT_MODEL_CODEGEN",
    "COMPLETE_RUNTIME",
    "PROFILE",
    "KERNEL_MUTATE",
    "GRAVITY_MUTATE",
    "SCHEDULER_MUTATE",
    "TG_ASCENT",
    "CAPABILITY",
    "SEAL",
    "NOTIFY",
    "OFFLOAD_EVICT",
    "ROTATE",
)

REPRESENTATIVE_SELECTION_CRITERIA: tuple[str, ...] = (
    "family_semantic_coverage",
    "capability",
    "source_availability",
    "license",
    "architecture_distinction",
    "gravity_feasibility",
    "tg_feasibility",
    "local_runtime_value",
)

REPRESENTATIVE_LAUNCH_GATES: tuple[str, ...] = (
    "official_high_precision_source",
    "complete_bpw_le_1_5",
    "tg3",
    "capability",
    "parity",
    "hcli_integration_when_applicable",
    "restore",
)

ROTATION_STATES: tuple[str, ...] = (
    "ACTIVE",
    "PAUSED_FOR_MECHANISM",
    "WAITING_FOR_EXTERNAL_REVIEW",
    "WAITING_FOR_STORAGE",
)

_DOCTRINE_KEYS: dict[str, str] = {
    "FAMILY_QWEN": "QWEN",
    "FAMILY_LLAMA": "LLAMA",
    "FAMILY_MISTRAL": "MISTRAL_MIXTRAL",
    "FAMILY_DEEPSEEK": "DEEPSEEK",
    "FAMILY_GLM": "GLM",
    "FAMILY_KIMI": "KIMI",
    "FAMILY_GEMMA": "GEMMA",
    "FAMILY_HYBRID": "STATE_SPACE_OR_LINEAR_ATTENTION_HYBRID",
}


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


def build_family_workflow(*, states: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the complete future family campaign graph from current state."""

    sandbox_status = str(states.get("SANDBOX_ACTIVATION", {}).get("status") or "ABSENT")
    families: list[dict[str, Any]] = []
    for state_id, completion, family_label in FAMILY_RULES:
        doctrine_key = _DOCTRINE_KEYS[state_id]
        state_status = str(states.get(state_id, {}).get("status") or "ABSENT")
        if state_status == "CERTIFIED":
            operation_status = "FAMILY_EVIDENCE_COMPLETE"
        elif sandbox_status == "CERTIFIED":
            operation_status = "READY_FOR_EXACT_REPRESENTATIVE_EVIDENCE"
        else:
            operation_status = "WAITING_FOR_SEALED_MANAGER_SANDBOX"
        families.append(
            {
                "state": state_id,
                "family": family_label,
                "completion_state": completion,
                "state_status": state_status,
                "operation_status": operation_status,
                "starting_doctrine": list(FAMILY_STARTING_DOCTRINES[doctrine_key]),
                "selection_criteria": list(REPRESENTATIVE_SELECTION_CRITERIA),
                "representative_launch_gates": list(REPRESENTATIVE_LAUNCH_GATES),
                "per_model_loop": list(PER_MODEL_LOOP),
                "rotation_states": list(ROTATION_STATES),
                "failure_handling": [
                    "seal_failure",
                    "classify_failure",
                    "query_graveyard",
                    "choose_materially_different_mutation",
                    "continue",
                ],
                "generic_fallback_forbidden": True,
                "metadata_only_support_forbidden": True,
            }
        )
    all_families_complete = all(row["state_status"] == "CERTIFIED" for row in families)
    global_status = str(states.get("GLOBAL_LAUNCH_AUDIT", {}).get("status") or "ABSENT")
    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_WORKFLOW_ONLY",
            "recorded_at": _utc_now(),
            "sandbox_dependency": {
                "state": "SANDBOX_ACTIVATION",
                "status": sandbox_status,
                "only_sealed_manager_may_be_active": True,
            },
            "families": families,
            "generic_hf_reference": {
                "required": True,
                "completion_state": "GENERIC_HF_REFERENCE_READY",
                "owned_by_global_audit_state": "GLOBAL_LAUNCH_AUDIT",
                "is_not_core_family_substitute": True,
            },
            "all_advertised_models_law": {
                "each_advertised_exact_model_requires": list(REPRESENTATIVE_LAUNCH_GATES),
                "permitted_non_launch_labels": [
                    "DECLARED",
                    "REFERENCE",
                    "EXPERIMENTAL",
                    "UNVERIFIED",
                ],
            },
            "matrix_handoff": {
                "all_required_family_states_certified": all_families_complete,
                "global_audit_status": global_status,
                "required_global_artifacts": [
                    "ASCENSION_V3_FAMILY_MATRIX",
                    "ASCENSION_V3_MODEL_MATRIX",
                    "ASCENSION_V3_DENSITY_MATRIX",
                    "ASCENSION_V3_TG_MATRIX",
                    "ASCENSION_V3_RESOURCE_ATLAS",
                    "ASCENSION_V3_POWER_PROFILES",
                    "ASCENSION_V3_STORAGE_LEASES",
                    "ASCENSION_V3_GARBAGE_AUDIT",
                    "ASCENSION_V3_RECOVERY_TEST",
                ],
            },
            "manager_replacement_rule": {
                "separate_candidate_lane": True,
                "replacement_requires": [
                    "manager_source",
                    "complete_bpw_le_1_5",
                    "tg3",
                    "capability",
                    "agent_os",
                    "tournament_comparison",
                    "external_review",
                ],
                "cannot_replace_active_manager_automatically": True,
            },
            "claim_boundary": {
                "workflow_is_not_family_source_selection": True,
                "workflow_is_not_exact_model_qualification": True,
                "workflow_does_not_start_downloads_or_model_bodies": True,
                "workflow_does_not_claim_generic_reference_is_production_support": True,
            },
        }
    )


def write_family_workflow(root: str | Path, *, states: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    document = build_family_workflow(states=states)
    _atomic_json(resolved / FILENAME, document)
    return document


__all__ = [
    "FILENAME",
    "PER_MODEL_LOOP",
    "REPRESENTATIVE_LAUNCH_GATES",
    "REPRESENTATIVE_SELECTION_CRITERIA",
    "ROTATION_STATES",
    "SCHEMA",
    "build_family_workflow",
    "write_family_workflow",
]
