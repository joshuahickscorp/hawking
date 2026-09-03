"""Deterministic, two-manager work graph for Ascension V3.

This module turns the Bible's 30B-versus-80B sequence into a durable handoff
contract.  It consumes only sealed *candidate* source metadata and lifecycle
state, writes a controller-owned workflow manifest, and never upgrades that
metadata into source authority or permission to download/load a model body.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.ascension_lifecycle import (
    MODEL_30B,
    MODEL_80B,
    QWEN30_GRAVITY_MANAGER_ARTIFACT,
    QWEN80_GRAVITY_MANAGER_ARTIFACT,
)
from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.dual_manager_workflow.v1"
FILENAME = "ASCENSION_DUAL_MANAGER_WORKFLOW.json"


@dataclass(frozen=True)
class ManagerWorkflowSpec:
    key: str
    model_id: str
    tournament_artifact_id: str
    role: str
    source_candidate_filename: str
    source_candidate_id: str
    stage_ids: tuple[str, str, str]
    required_artifacts: tuple[str, ...]
    implementation_surfaces: tuple[tuple[str, str], ...]
    local_source_path: Path | None = None


REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGERS: tuple[ManagerWorkflowSpec, ...] = (
    ManagerWorkflowSpec(
        key="qwen30",
        model_id=MODEL_30B,
        tournament_artifact_id=QWEN30_GRAVITY_MANAGER_ARTIFACT,
        role="executor",
        source_candidate_filename="QWEN30_SOURCE_METADATA_CANDIDATE.json",
        source_candidate_id="QWEN30_SOURCE_METADATA_CANDIDATE",
        stage_ids=("MANAGER_30B_DENSITY", "MANAGER_30B_TG", "MANAGER_30B_AGENT"),
        required_artifacts=(
            "QWEN30_MANAGER_SOURCE",
            "QWEN30_MANAGER_CAPABILITY_ANCHOR",
            "QWEN30_MANAGER_GRAVITY",
            "QWEN30_MANAGER_TG3",
            "QWEN30_MANAGER_KERNEL_OPERATIONAL",
            "QWEN30_MANAGER_AGENT_OS",
        ),
        implementation_surfaces=(
            ("source_admission", "lab/operators/ascension_source_admission.py"),
            ("bounded_gravity_probe", "lab/operators/qwen30b_gravity_pack.py"),
            ("parity_ladder_contract", "lab/operators/ascension_parity_ladder.py"),
            ("tg_gauntlet_contract", "lab/operators/ascension_tg_gauntlet.py"),
            ("agent_os_policy", "lab/hcli/option_c.py"),
            ("residency_policy", "lab/hcli/residency.py"),
        ),
        local_source_path=REPO_ROOT
        / "workspace"
        / "campaign"
        / "records"
        / "runs"
        / "qwen-30b"
        / "Qwen3-Coder-30B-A3B-Instruct",
    ),
    ManagerWorkflowSpec(
        key="qwen80",
        model_id=MODEL_80B,
        tournament_artifact_id=QWEN80_GRAVITY_MANAGER_ARTIFACT,
        role="reviewer",
        source_candidate_filename="QWEN80_SOURCE_METADATA_CANDIDATE.json",
        source_candidate_id="QWEN80_SOURCE_METADATA_CANDIDATE",
        stage_ids=("MANAGER_80B_DENSITY", "MANAGER_80B_TG", "MANAGER_80B_AGENT"),
        required_artifacts=(
            "QWEN80_MANAGER_SOURCE",
            "QWEN80_MANAGER_CAPABILITY_ANCHOR",
            "QWEN80_MANAGER_GRAVITY",
            "QWEN80_MANAGER_TG3",
            "QWEN80_MANAGER_KERNEL_OPERATIONAL",
            "QWEN80_MANAGER_AGENT_OS",
        ),
        implementation_surfaces=(
            ("source_admission", "lab/operators/ascension_source_admission.py"),
            ("hybrid_parity_ladder_contract", "lab/operators/ascension_parity_ladder.py"),
            ("tg_gauntlet_contract", "lab/operators/ascension_tg_gauntlet.py"),
            ("agent_os_policy", "lab/hcli/option_c.py"),
            ("residency_policy", "lab/hcli/residency.py"),
        ),
    ),
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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _candidate_source(root: Path, spec: ManagerWorkflowSpec) -> dict[str, Any]:
    """Expose only safe identity metadata from a valid candidate record."""

    path = root / "source-admission" / spec.source_candidate_filename
    document = _read_json(path)
    if document is None:
        return {"status": "ABSENT", "path": str(path), "candidate_only": True}
    try:
        checked = verify(document, label=str(path))
    except SealIntegrityError as exc:
        return {
            "status": "INVALID",
            "path": str(path),
            "reason": str(exc),
            "candidate_only": True,
        }
    target = checked.get("target") if isinstance(checked.get("target"), Mapping) else {}
    source = checked.get("source") if isinstance(checked.get("source"), Mapping) else {}
    architecture = checked.get("architecture") if isinstance(checked.get("architecture"), Mapping) else {}
    valid = (
        checked.get("artifact_id") == spec.source_candidate_id
        and checked.get("status") == "CANDIDATE_METADATA_CAPTURED"
        and target.get("model_id") == spec.model_id
        and isinstance(source.get("revision"), str)
        and len(source.get("revision")) == 40
    )
    return {
        "status": "PINNED_CANDIDATE_METADATA" if valid else "INVALID_OR_MISMATCHED_CANDIDATE",
        "path": str(path),
        "candidate_only": True,
        "not_source_authority": True,
        "repository": target.get("repository"),
        "revision": source.get("revision"),
        "config_sha256": architecture.get("config_sha256"),
        "license": source.get("license_card_value"),
        "model_type": architecture.get("model_type"),
        "architectures": architecture.get("architectures"),
        "seal_sha256": checked.get("seal_sha256"),
        "no_model_body_downloaded": (checked.get("claim_boundary") or {}).get(
            "no_model_body_downloaded"
        ) is True,
    }


def _manager_row(
    root: Path, spec: ManagerWorkflowSpec, states: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    source = _candidate_source(root, spec)
    stage_statuses = {state: states.get(state, {}).get("status", "ABSENT") for state in spec.stage_ids}
    implementation = [
        {
            "role": role,
            "path": path,
            "present": (REPO_ROOT / path).is_file(),
        }
        for role, path in spec.implementation_surfaces
    ]
    local_source = {
        "known_path": str(spec.local_source_path) if spec.local_source_path else None,
        "present": spec.local_source_path.is_dir() if spec.local_source_path else False,
        "is_not_admission_or_qualification": True,
    }
    if spec.key == "qwen80":
        local_source["body_execution_adapter_status"] = "NOT_YET_MEASURED_OR_CERTIFIED"
    else:
        local_source["body_execution_adapter_status"] = "RESEARCH_PROBE_EXISTS_NOT_LAUNCH_QUALIFICATION"
    return {
        "key": spec.key,
        "source_teacher_model_id": spec.model_id,
        "tournament_artifact_id": spec.tournament_artifact_id,
        "role": spec.role,
        "source_candidate": source,
        "stage_order": list(spec.stage_ids),
        "stage_statuses": stage_statuses,
        "required_controller_evidence": list(spec.required_artifacts),
        "implementation_surfaces": implementation,
        "local_source": local_source,
        "transition_policy": {
            "source_anchor_then_gravity_then_tg3_then_100tps_custom_kernel_then_agent_os": True,
            "candidate_completion_requires_final_agent_state_certification": True,
            "candidate_metadata_is_not_download_permission": True,
            "all_model_body_jobs_require_independent_controller_evidence": True,
            "raw_bf16_source_is_authority_teacher_not_tournament_participant": True,
            "tournament_artifact_requires_complete_gravity_at_most_1_5_bpw": True,
        },
    }


def build_dual_manager_workflow(
    root: str | Path, *, states: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the restart-safe 30B → 80B → protected tournament handoff map."""

    resolved = Path(root).expanduser().resolve()
    rows = [_manager_row(resolved, spec, states) for spec in MANAGERS]
    by_key = {row["key"]: row for row in rows}
    qwen30_ready = by_key["qwen30"]["stage_statuses"].get("MANAGER_30B_AGENT") == "CERTIFIED"
    qwen80_ready = by_key["qwen80"]["stage_statuses"].get("MANAGER_80B_AGENT") == "CERTIFIED"
    tournament_phase = (
        "READY_FOR_PROTECTED_TOURNAMENT_RECEIPT" if qwen30_ready and qwen80_ready else "WAITING_FOR_BOTH_QUALIFIED_MANAGERS"
    )
    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_WORKFLOW_ONLY",
            "recorded_at": _utc_now(),
            "fixed_candidate_order": [spec.tournament_artifact_id for spec in MANAGERS],
            "raw_bf16_models_are_source_authorities_not_tournament_participants": True,
            "managers": rows,
            "handoff": {
                "qwen30_candidate_certified": qwen30_ready,
                "qwen80_may_begin_after_qwen30_agent_certified": qwen30_ready,
                "qwen80_candidate_certified": qwen80_ready,
                "tournament_phase": tournament_phase,
                "tournament_requires": ["MANAGER_30B_AGENT", "MANAGER_80B_AGENT"],
                "post_tournament": [
                    "ASCENSION_MANAGER_WINNER protected certification",
                    "ASCENSION_ALTERNATE_OFFLOAD remote hash + restore + local eviction proof",
                    "SANDBOX_ACTIVATION only under sealed winner",
                ],
            },
            "resource_and_authority_policy": {
                "one_heavy_model_body_job_at_a_time": True,
                "minimum_operational_custom_kernel_base_true_tps": 100.0,
                "tg3_remains_the_stricter_tournament_runtime_floor": 333.0,
                "use_bounded_process_runner_for_body_work": "workspace/ops/ascension/bounded_process_runner.py",
                "source_lifecycle_requires_stream_verify_transform_seal_evict": "lab/operators/credential_broker/lifecycle.py",
                "executor_and_reviewer_may_emit_candidate_reports_only": "lab/hcli/option_c.py",
                "protected_controller_or_human_certifies_terminal_receipts": True,
            },
            "claim_boundary": {
                "workflow_is_not_source_authority": True,
                "workflow_is_not_model_body_download_authorization": True,
                "workflow_is_not_a_runtime_measurement": True,
                "workflow_is_not_a_tournament_result": True,
                "workflow_does_not_start_or_evict_models": True,
            },
        }
    )


def write_dual_manager_workflow(
    root: str | Path, *, states: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    document = build_dual_manager_workflow(resolved, states=states)
    _atomic_json(resolved / FILENAME, document)
    return document


__all__ = [
    "FILENAME",
    "MANAGERS",
    "ManagerWorkflowSpec",
    "SCHEMA",
    "build_dual_manager_workflow",
    "write_dual_manager_workflow",
]
