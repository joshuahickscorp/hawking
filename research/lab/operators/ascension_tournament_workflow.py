"""Protected Manager Tournament and post-winner transition contract.

It intentionally keeps the competition deterministic without pretending that a
configuration file can run a valid benchmark.  The fixed candidate order,
dimensions, frozen-catalog binding, winning authority, alternate-offload
proof, and sandbox-start fence are all materialized for the persistent
supervisor to preserve across restarts.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.operators.ascension_lifecycle import (
    MANAGER_CANDIDATE_ORDER,
    MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR,
    QWEN30_GRAVITY_MANAGER_ARTIFACT,
    QWEN80_GRAVITY_MANAGER_ARTIFACT,
    TOURNAMENT_CANDIDATE_ORDER,
    TOURNAMENT_DIMENSIONS,
)
from lab.operators.ascension_manager_tournament_protocol import (
    SCHEMA as FINAL_MANAGER_PROTOCOL_SCHEMA,
    build_final_manager_tournament_protocol,
)
from lab.receipts import seal


SCHEMA = "hawking.ascension.manager_tournament_workflow.v1"
FILENAME = "ASCENSION_MANAGER_TOURNAMENT_WORKFLOW.json"

TOURNAMENT_RECEIPTS: tuple[str, ...] = (
    "ASCENSION_MANAGER_TOURNAMENT",
    "ASCENSION_MANAGER_WINNER",
    "ASCENSION_ALTERNATE_OFFLOAD",
)

ALTERNATE_OFFLOAD_PROOFS: tuple[str, ...] = (
    "winner_model_binding",
    "alternate_model_binding",
    "remote_hash",
    "one_command_restore",
    "alternate_local_body_evicted",
    "small_fixtures_retained",
    "no_permanent_second_local_reviewer",
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


def build_tournament_workflow(
    *, states: Mapping[str, Mapping[str, Any]], tournament: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive deterministic run conditions without scoring or promotion."""

    final_manager_protocol = build_final_manager_tournament_protocol()
    qwen30 = str(states.get("MANAGER_30B_AGENT", {}).get("status") or "ABSENT")
    qwen80 = str(states.get("MANAGER_80B_AGENT", {}).get("status") or "ABSENT")
    tournament_state = str(states.get("MANAGER_TOURNAMENT", {}).get("status") or "ABSENT")
    armed_status = str(tournament.get("status") or "NOT_ARMED")
    candidates_qualified = qwen30 == "CERTIFIED" and qwen80 == "CERTIFIED"
    if tournament_state == "CERTIFIED":
        phase = "SEALED_COMPLETE_NO_RERUN"
    elif candidates_qualified and armed_status.startswith("ARMED"):
        phase = "READY_FOR_PROTECTED_EXECUTION_RECEIPT"
    elif armed_status.startswith("ARMED"):
        phase = "ARMED_WAITING_FOR_QUALIFICATIONS"
    else:
        phase = "NOT_ARMED"
    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_WORKFLOW_ONLY",
            "recorded_at": _utc_now(),
            "runtime_phase": phase,
            "arming": dict(tournament),
            "candidates": [
                {
                    "tournament_artifact_id": QWEN30_GRAVITY_MANAGER_ARTIFACT,
                    "source_teacher_model_id": MANAGER_CANDIDATE_ORDER[0],
                    "agent_qualification_state": qwen30,
                },
                {
                    "tournament_artifact_id": QWEN80_GRAVITY_MANAGER_ARTIFACT,
                    "source_teacher_model_id": MANAGER_CANDIDATE_ORDER[1],
                    "agent_qualification_state": qwen80,
                },
            ],
            "deterministic_comparison_contract": {
                "fixed_candidate_order": list(TOURNAMENT_CANDIDATE_ORDER),
                "raw_bf16_models_are_source_authorities_and_not_participants": True,
                "each_participant_requires_complete_gravity_artifact_at_most_1_5_bpw": True,
                "frozen_hidden_task_catalog_required": True,
                "catalog_hash_required": True,
                "comparison_dimensions": list(TOURNAMENT_DIMENSIONS),
                "all_dimensions_must_be_measured": True,
                "no_blended_or_unmeasured_dimension_may_choose_a_winner": True,
            },
            "final_manager_selection_protocol": {
                "schema": FINAL_MANAGER_PROTOCOL_SCHEMA,
                "seal_sha256": final_manager_protocol["seal_sha256"],
                "protocol_identity_sha256": final_manager_protocol["protocol_identity_sha256"],
                "protocol": final_manager_protocol,
                "binding_required_in_scored_tournament_receipt": True,
                "runs_only_after_both_candidates_are_frozen_at_complete_manager_floor": True,
                "does_not_create_a_new_lifecycle_state_or_weaken_tg3": True,
            },
            "pre_tournament_kernel_gate": {
                "each_candidate_requires_exact_model_custom_kernel": True,
                "minimum_operational_base_true_tps": MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR,
                "operational_floor_is_not_a_tg3_substitute": True,
                "tg3_base_true_tps": 333.0,
                "required_manager_receipts": [
                    "QWEN30_MANAGER_KERNEL_OPERATIONAL",
                    "QWEN80_MANAGER_KERNEL_OPERATIONAL",
                ],
            },
            "protected_authority": {
                "models_may_not_self_promote": True,
                "winner_must_be_certified_by": ["protected_controller", "human_operator"],
                "required_receipts": list(TOURNAMENT_RECEIPTS),
                "winner_designation": "ASCENSION_MANAGER",
            },
            "alternate_offload_contract": {
                "required_proofs": list(ALTERNATE_OFFLOAD_PROOFS),
                "must_precede_sandbox_activation": True,
                "does_not_allow_hidden_second_manager": True,
            },
            "sandbox_start_contract": {
                "requires_sealed_manager_tournament": True,
                "requires_sealed_winner": True,
                "requires_alternate_offload": True,
                "only_winner_active": True,
                "second_local_manager_active": False,
                "losing_gravity_artifact_cold_stored_then_locally_evicted": True,
            },
            "claim_boundary": {
                "workflow_does_not_execute_tasks": True,
                "workflow_does_not_score_subjective_dimensions": True,
                "workflow_does_not_choose_winner": True,
                "workflow_does_not_evict_alternate": True,
                "workflow_does_not_activate_sandbox": True,
            },
        }
    )


def write_tournament_workflow(
    root: str | Path,
    *,
    states: Mapping[str, Mapping[str, Any]],
    tournament: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    document = build_tournament_workflow(states=states, tournament=tournament)
    _atomic_json(resolved / FILENAME, document)
    return document


__all__ = [
    "ALTERNATE_OFFLOAD_PROOFS",
    "FILENAME",
    "SCHEMA",
    "TOURNAMENT_RECEIPTS",
    "build_tournament_workflow",
    "write_tournament_workflow",
]
