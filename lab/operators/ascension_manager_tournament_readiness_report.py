"""Read-only, fail-closed final-manager tournament readiness snapshot.

This module is deliberately an observer.  It reads the sealed lifecycle,
physical qualification, and prepared-contract records and returns a sealed
snapshot describing the *conjunction* still required before a protected
manager tournament could be considered.  It never writes campaign state,
creates tasks, starts a process, acquires a lease, or chooses a winner.

The Rust paired-cognition contracts are represented twice in the snapshot:
their source-definition identities make the exact prepared implementation
auditable, while optional sealed materializations under ``authority_root``
prove that the prepared contracts have actually been instantiated.  Source
code alone never substitutes for a sealed materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.ascension_manager_tournament_protocol import (
    SCHEMA as FINAL_MANAGER_PROTOCOL_SCHEMA,
    build_final_manager_tournament_protocol,
)
from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.manager_tournament_readiness_report.v1"
STATUS_PREPARED = "PREPARED_NOT_ACTIVE_MANAGER_TOURNAMENT_READINESS_CONJUNCTION_COMPLETE"
STATUS_REFUSED = "REFUSED_MANAGER_TOURNAMENT_READINESS_CONJUNCTION_INCOMPLETE_OR_UNTRUSTED"

LIFECYCLE_STATE_SCHEMA = "hawking.ascension.v3_state.v1"
LIFECYCLE_WORKFLOW_SCHEMA = "hawking.ascension.manager_tournament_workflow.v1"
LIFECYCLE_CONTROLLER_SCHEMA = "hawking.ascension.manager_tournament_controller.v1"
PHYSICAL_GATE_SCHEMA = "hawking.ascension.physical_tournament_gate.v1"
OPERATIONAL_ASCENT_SCHEMA = "hawking.ascension.physical_operational_ascent.v1"
EXPECTED_PROTOCOL_STATUS = "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED"
# The identity is intentionally pinned rather than recomputed from a changed
# implementation.  A protocol edit therefore fails closed until an explicit,
# separately reviewed update changes this contract too.
EXPECTED_PROTOCOL_IDENTITY_SHA256 = (
    "8e3684af0b7de53690a9c88ce0d52b0cae019e0d798bc2748b5b1556211facf8"
)

QWEN30 = "qwen30"
QWEN80 = "qwen80"
CANDIDATE_ARTIFACTS: tuple[str, str] = (
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
)

FLOOR_REQUIREMENTS: tuple[str, ...] = (
    "verified_raw_source_identity",
    "current_source_revalidation",
    "complete_admitted_artifact_at_most_1_5_bpw",
    "native_exact_full_token_runtime",
    "measured_hcli",
    "custom_kernel_operational_at_least_100_tps",
    "capability_and_evaluation_receipt",
    "final_manager_operations_agent_os_session_restart_residency_rollback_storage",
)
TG10_REQUIREMENT = "tg10_operational_exact_model_100_tps"
TG3_REQUIREMENT = "tg3_at_least_333_tps"


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    artifact_id: str
    manager_state: str
    tg_status_filename: str


CANDIDATES: tuple[CandidateSpec, CandidateSpec] = (
    CandidateSpec(
        key=QWEN30,
        artifact_id=CANDIDATE_ARTIFACTS[0],
        manager_state="MANAGER_30B_AGENT",
        tg_status_filename="QWEN30_TG3_ASCENT_STATUS.json",
    ),
    CandidateSpec(
        key=QWEN80,
        artifact_id=CANDIDATE_ARTIFACTS[1],
        manager_state="MANAGER_80B_AGENT",
        tg_status_filename="QWEN80_TG3_ASCENT_STATUS.json",
    ),
)


@dataclass(frozen=True)
class AuthoritySpec:
    key: str
    filename: str
    source_relative_path: str
    expected_schema: str
    expected_status: str
    binds_final_protocol: bool = False


AUTHORITY_SPECS: tuple[AuthoritySpec, ...] = (
    AuthoritySpec(
        key="lane",
        filename="PAIRED_COGNITION_LANE_AUTHORITY.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_paired_cognition_lane_authority_contract.rs"
        ),
        expected_schema="hawking.ascension.paired_cognition_lane_namespace_mission_authority.v1",
        expected_status=(
            "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_TWO_SEALED_CANDIDATE_WORLDS_"
            "NO_RUNTIME_SERVER_OR_TOURNAMENT"
        ),
    ),
    AuthoritySpec(
        key="mutation",
        filename="PAIRED_COGNITION_MUTATION_AUTHORITY.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_paired_cognition_mutation_authority_contract.rs"
        ),
        expected_schema=(
            "hawking.ascension.paired_cognition_proposal_review_falsification_"
            "primary_acceptance_authority.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_PRIMARY_ONLY_CHAMPION_MUTATION_"
            "PROMOTION_NO_MANAGER_OR_TOURNAMENT_SELECTION"
        ),
    ),
    AuthoritySpec(
        key="knowledge",
        filename="PAIRED_COGNITION_KNOWLEDGE_AUTHORITY.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_paired_cognition_knowledge_plane_"
            "release_gate.rs"
        ),
        expected_schema=(
            "hawking.ascension.paired_cognition_knowledge_plane_generic_release_authority.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_GENERIC_ONLY_INDEPENDENTLY_"
            "VERIFIED_APPEND_ONLY_KNOWLEDGE_PLANE_NO_RUNTIME_OR_TOURNAMENT"
        ),
    ),
    AuthoritySpec(
        key="scheduler",
        filename="PAIRED_COGNITION_ROLE_RESOURCE_SCHEDULER_AUTHORITY.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_paired_cognition_role_resource_"
            "scheduler_authority.rs"
        ),
        expected_schema=(
            "hawking.ascension.paired_cognition_one_body_many_logical_session_role_"
            "resource_scheduler_authority.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_ONE_Q30_ONE_Q80_FOUR_ISOLATED_"
            "LOGICAL_ROLES_NO_RUNTIME_OR_WINNER_SELECTION"
        ),
    ),
    AuthoritySpec(
        key="tg10_development",
        filename="PAIRED_COGNITION_TG10_DEVELOPMENT_ACTIVATION.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_paired_cognition_tg10_development_"
            "activation_state_machine.rs"
        ),
        expected_schema=(
            "hawking.ascension.paired_cognition_both_tg10_development_activation_"
            "state_machine.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_EXACT_TG10_OPERATIONAL_"
            "RECEIPTS_BOUND_NO_RUNTIME_OR_TOURNAMENT"
        ),
    ),
    AuthoritySpec(
        key="tg3_freeze",
        filename="PAIRED_COGNITION_TG3_FREEZE_FINAL_COMPARISON_AUTHORITY.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_paired_cognition_tg3_freeze_"
            "final_comparison_authority.rs"
        ),
        expected_schema=(
            "hawking.ascension.paired_cognition_tg3_freeze_final_comparison_authority.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_TG3_FROZEN_FINAL_MANAGER_"
            "COMPARISON_RESERVED"
        ),
        binds_final_protocol=True,
    ),
    AuthoritySpec(
        key="protected_corpus",
        filename="ASCENSION_MANAGER_TOURNAMENT_PROTECTED_TASK_CORPUS_COMMITMENT.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_manager_tournament_protected_task_"
            "corpus_commitment.rs"
        ),
        expected_schema=(
            "hawking.ascension.manager_tournament_protected_task_corpus_commitment_"
            "authority.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PROTECTED_REAL_HAWKING_TASK_CORPUS_METADATA_"
            "COMMITTED_NO_HIDDEN_TASKS_OR_SCORED_EXECUTION"
        ),
        binds_final_protocol=True,
    ),
    AuthoritySpec(
        key="scorecard_adjudication",
        filename="ASCENSION_MANAGER_TOURNAMENT_SCORECARD_ADJUDICATION.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_manager_tournament_scorecard_"
            "adjudication_contract.rs"
        ),
        expected_schema=(
            "hawking.ascension.manager_tournament_scorecard_adjudication_contract.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_MANAGER_TOURNAMENT_SCORECARD_ADJUDICATION_PENDING_"
            "PROTECTED_SELECTION"
        ),
        binds_final_protocol=True,
    ),
    AuthoritySpec(
        key="selection_recovery",
        filename="ASCENSION_MANAGER_TOURNAMENT_PROTECTED_SELECTION_AND_RECOVERY.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_manager_tournament_protected_"
            "selection_and_recovery_contract.rs"
        ),
        expected_schema=(
            "hawking.ascension.manager_tournament_protected_selection_and_recovery_"
            "contract.v1"
        ),
        expected_status=(
            "PREPARED_NOT_ACTIVE_PROTECTED_SELECTION_AND_RECOVERY_CONTRACT_COMPLETE_"
            "NO_WINNER_SELECTED"
        ),
        binds_final_protocol=True,
    ),
    AuthoritySpec(
        key="final_report",
        filename="ASCENSION_MANAGER_TOURNAMENT_FINAL_REPORT_CONTRACT.json",
        source_relative_path=(
            "crates/hawking-core/examples/ascension_manager_tournament_final_report_"
            "contract.rs"
        ),
        expected_schema="hawking.ascension.manager_tournament_final_report_contract.v1",
        expected_status=(
            "PREPARED_NOT_ACTIVE_PROTECTED_FINAL_MANAGER_SIDE_BY_SIDE_REPORT_"
            "RESERVED_NO_WINNER_OR_EXTERNAL_EMISSION"
        ),
        binds_final_protocol=True,
    ),
)


@dataclass(frozen=True)
class ReadinessPaths:
    """Locations read by the observer; none are ever written by it."""

    repository_root: Path
    lifecycle_root: Path
    physical_root: Path
    authority_root: Path

    @classmethod
    def defaults(cls) -> "ReadinessPaths":
        repository_root = Path(__file__).resolve().parents[2]
        sandbox_root = repository_root / "workspace" / "campaign" / "records" / "ascension-sandbox"
        return cls(
            repository_root=repository_root,
            lifecycle_root=sandbox_root / "lifecycle",
            physical_root=sandbox_root / "physical",
            authority_root=sandbox_root / "lifecycle" / "paired-authorities",
        )

    @classmethod
    def from_roots(
        cls,
        *,
        repository_root: str | Path,
        lifecycle_root: str | Path,
        physical_root: str | Path,
        authority_root: str | Path | None = None,
    ) -> "ReadinessPaths":
        resolved_lifecycle = Path(lifecycle_root).expanduser().resolve()
        return cls(
            repository_root=Path(repository_root).expanduser().resolve(),
            lifecycle_root=resolved_lifecycle,
            physical_root=Path(physical_root).expanduser().resolve(),
            authority_root=(
                Path(authority_root).expanduser().resolve()
                if authority_root is not None
                else resolved_lifecycle / "paired-authorities"
            ),
        )

    @property
    def lifecycle_state(self) -> Path:
        return self.lifecycle_root / "ASCENSION_V3_STATE.json"

    @property
    def lifecycle_workflow(self) -> Path:
        return self.lifecycle_root / "ASCENSION_MANAGER_TOURNAMENT_WORKFLOW.json"

    @property
    def lifecycle_controller(self) -> Path:
        return self.lifecycle_root / "ASCENSION_MANAGER_TOURNAMENT_CONTROLLER.json"

    @property
    def physical_gate(self) -> Path:
        return self.physical_root / "lifecycle" / "ASCENSION_PHYSICAL_TOURNAMENT_GATE_STATUS.json"

    @property
    def operational_ascent(self) -> Path:
        return self.physical_root / "lifecycle" / "ASCENSION_OPERATIONAL_ASCENT_STATUS.json"

    def tg_status(self, candidate: CandidateSpec) -> Path:
        return self.physical_root / candidate.key / "tg3" / candidate.tg_status_filename

    def authority_document(self, authority: AuthoritySpec) -> Path:
        return self.authority_root / authority.filename


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except OSError as error:
        return None, f"unreadable:{error.__class__.__name__}"
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(value, Mapping):
        return None, "not_object"
    return dict(value), None


def _sealed_document(path: Path, *, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    document, error = _read_json(path)
    if document is None:
        return None, [f"{label}: {error}"]
    try:
        return verify(document, label=label), []
    except SealIntegrityError as error:
        return None, [f"{label}: unsealed_or_changed ({error})"]


def _document_identity(
    path: Path,
    document: Mapping[str, Any] | None,
    issues: Sequence[str],
) -> dict[str, Any]:
    raw_sha256: str | None = None
    try:
        raw_sha256 = _sha256_bytes(path.read_bytes())
    except OSError:
        pass
    return {
        "path": str(path),
        "readable": document is not None,
        "raw_sha256": raw_sha256,
        "seal_sha256": document.get("seal_sha256") if document is not None else None,
        "schema": document.get("schema") if document is not None else None,
        "status": document.get("status") if document is not None else None,
        "issues": list(issues),
    }


def _state_rows(document: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    issues: list[str] = []
    rows = document.get("states")
    if not isinstance(rows, list):
        return {}, ["lifecycle state document lacks states array"]
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        mapped = _mapping(row)
        state_id = mapped.get("id")
        if not isinstance(state_id, str) or not state_id:
            issues.append("lifecycle state document has a state without id")
            continue
        if state_id in by_id:
            issues.append(f"lifecycle state document duplicates {state_id}")
            continue
        by_id[state_id] = mapped
    return by_id, issues


def _candidate_rows(document: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    issues: list[str] = []
    rows = document.get("models")
    if not isinstance(rows, list):
        return {}, ["physical gate lacks models array"]
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        mapped = _mapping(row)
        key = mapped.get("key")
        if not isinstance(key, str) or not key:
            issues.append("physical gate has a model without key")
            continue
        if key in by_key:
            issues.append(f"physical gate duplicates {key}")
            continue
        by_key[key] = mapped
    return by_key, issues


def _require_schema(
    document: Mapping[str, Any] | None,
    expected: str,
    *,
    label: str,
) -> list[str]:
    if document is None:
        return [f"{label} is unavailable"]
    if document.get("schema") != expected:
        return [f"{label} schema mismatch"]
    return []


def _verify_protocol_document(document: Mapping[str, Any], *, label: str) -> list[str]:
    issues: list[str] = []
    if document.get("schema") != FINAL_MANAGER_PROTOCOL_SCHEMA:
        issues.append(f"{label} protocol schema mismatch")
    if document.get("status") != EXPECTED_PROTOCOL_STATUS:
        issues.append(f"{label} protocol status mismatch")
    if document.get("protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY_SHA256:
        issues.append(f"{label} protocol identity mismatch")
    try:
        verify(document, label=label)
    except SealIntegrityError as error:
        issues.append(f"{label} protocol is unsealed_or_changed ({error})")
    return issues


def _protocol_snapshot(
    *,
    lifecycle_state: Mapping[str, Any] | None,
    workflow: Mapping[str, Any] | None,
    controller: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    implementation = build_final_manager_tournament_protocol()
    implementation_identity = implementation.get("protocol_identity_sha256")
    if implementation_identity != EXPECTED_PROTOCOL_IDENTITY_SHA256:
        issues.append("canonical final-manager protocol implementation identity changed")
    if implementation.get("schema") != FINAL_MANAGER_PROTOCOL_SCHEMA:
        issues.append("canonical final-manager protocol implementation schema changed")
    if implementation.get("status") != EXPECTED_PROTOCOL_STATUS:
        issues.append("canonical final-manager protocol implementation status changed")
    try:
        verify(implementation, label="canonical final-manager protocol implementation")
    except SealIntegrityError as error:
        issues.append(f"canonical final-manager protocol implementation is unsealed ({error})")

    state_tournament = _mapping(lifecycle_state.get("tournament") if lifecycle_state else None)
    if state_tournament.get("final_manager_protocol_schema") != FINAL_MANAGER_PROTOCOL_SCHEMA:
        issues.append("lifecycle controller state protocol schema mismatch")
    if state_tournament.get("final_manager_protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY_SHA256:
        issues.append("lifecycle controller state protocol identity mismatch")

    if controller is None:
        issues.append("lifecycle tournament controller is unavailable")
    else:
        if controller.get("schema") != LIFECYCLE_CONTROLLER_SCHEMA:
            issues.append("lifecycle tournament controller schema mismatch")
        if controller.get("final_manager_protocol_schema") != FINAL_MANAGER_PROTOCOL_SCHEMA:
            issues.append("lifecycle tournament controller protocol schema mismatch")
        if controller.get("final_manager_protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY_SHA256:
            issues.append("lifecycle tournament controller protocol identity mismatch")
        if tuple(controller.get("candidate_order") or ()) != CANDIDATE_ARTIFACTS:
            issues.append("lifecycle tournament controller candidate order mismatch")

    workflow_protocol = _mapping(
        _mapping(workflow.get("final_manager_selection_protocol") if workflow else None).get("protocol")
    )
    if not workflow_protocol:
        issues.append("lifecycle workflow lacks embedded sealed final-manager protocol")
    else:
        issues.extend(_verify_protocol_document(workflow_protocol, label="lifecycle workflow protocol"))
    workflow_binding = _mapping(workflow.get("final_manager_selection_protocol") if workflow else None)
    if workflow_binding.get("schema") != FINAL_MANAGER_PROTOCOL_SCHEMA:
        issues.append("lifecycle workflow protocol binding schema mismatch")
    if workflow_binding.get("protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY_SHA256:
        issues.append("lifecycle workflow protocol binding identity mismatch")

    return (
        {
            "expected_schema": FINAL_MANAGER_PROTOCOL_SCHEMA,
            "expected_identity_sha256": EXPECTED_PROTOCOL_IDENTITY_SHA256,
            "implementation_identity_sha256": implementation_identity,
            "implementation_seal_sha256": implementation.get("seal_sha256"),
            "lifecycle_state_identity_sha256": state_tournament.get(
                "final_manager_protocol_identity_sha256"
            ),
            "workflow_identity_sha256": workflow_binding.get("protocol_identity_sha256"),
            "controller_identity_sha256": (
                controller.get("final_manager_protocol_identity_sha256") if controller else None
            ),
            "trusted": not issues,
        },
        issues,
    )


def _source_identity(spec: AuthoritySpec, repository_root: Path) -> tuple[dict[str, Any], list[str]]:
    source_path = repository_root / spec.source_relative_path
    issues: list[str] = []
    try:
        content = source_path.read_bytes()
    except FileNotFoundError:
        return (
            {
                "source_path": str(source_path),
                "source_sha256": None,
                "source_readable": False,
                "expected_schema": spec.expected_schema,
                "expected_status": spec.expected_status,
                "binds_final_protocol": spec.binds_final_protocol,
                "prepared_definition_verified": False,
            },
            [f"{spec.key} authority source is missing"],
        )
    except OSError as error:
        return (
            {
                "source_path": str(source_path),
                "source_sha256": None,
                "source_readable": False,
                "expected_schema": spec.expected_schema,
                "expected_status": spec.expected_status,
                "binds_final_protocol": spec.binds_final_protocol,
                "prepared_definition_verified": False,
            },
            [f"{spec.key} authority source is unreadable ({error.__class__.__name__})"],
        )
    text = content.decode("utf-8", errors="replace")
    if spec.expected_schema not in text:
        issues.append(f"{spec.key} authority source does not declare its expected schema")
    if spec.expected_status not in text:
        issues.append(f"{spec.key} authority source does not declare its expected prepared status")
    if spec.binds_final_protocol and EXPECTED_PROTOCOL_IDENTITY_SHA256 not in text:
        issues.append(f"{spec.key} authority source does not pin the final-manager protocol identity")
    return (
        {
            "source_path": str(source_path),
            "source_sha256": _sha256_bytes(content),
            "source_readable": True,
            "expected_schema": spec.expected_schema,
            "expected_status": spec.expected_status,
            "binds_final_protocol": spec.binds_final_protocol,
            "prepared_definition_verified": not issues,
        },
        issues,
    )


def _materialized_authority(
    spec: AuthoritySpec,
    authority_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    path = authority_root / spec.filename
    document, issues = _sealed_document(path, label=f"{spec.key} authority")
    report = _document_identity(path, document, issues)
    report.update(
        {
            "expected_schema": spec.expected_schema,
            "expected_status": spec.expected_status,
            "prepared": False,
            "authority_boundary_safe": False,
        }
    )
    if document is None:
        return report, list(issues)
    local_issues = list(issues)
    if document.get("schema") != spec.expected_schema:
        local_issues.append(f"{spec.key} authority materialization schema mismatch")
    if document.get("status") != spec.expected_status:
        local_issues.append(f"{spec.key} authority materialization status is not prepared")
    if document.get("prepared") is not True:
        local_issues.append(f"{spec.key} authority materialization is not marked prepared")
    authority_boundary = _mapping(document.get("authority_boundary"))
    if not authority_boundary:
        local_issues.append(f"{spec.key} authority materialization lacks authority boundary")
    else:
        for field in (
            "new_physical_model_processes_authorized",
            "server_starts_authorized",
            "port_binds_authorized",
            "gpu_leases_authorized",
            "tournament_state_mutations_authorized",
        ):
            if authority_boundary.get(field, 0) != 0:
                local_issues.append(f"{spec.key} authority grants {field}")
        if authority_boundary.get("paired_world_activation_authorized") is True:
            local_issues.append(f"{spec.key} authority grants paired-world activation")
    for field in (
        "paired_candidate_worlds_active",
        "tournament_active",
        "scored_task_execution_active",
        "scorecards_executed_by_this_contract",
        "winner_selected",
        "hidden_tasks_created",
        "scored_execution_started",
        "candidate_or_red_team_hidden_access_granted",
    ):
        if document.get(field) is True:
            local_issues.append(f"{spec.key} authority claims forbidden active state {field}")
    report["issues"] = local_issues
    report["prepared"] = not local_issues
    report["authority_boundary_safe"] = not any(
        "grants " in issue or "forbidden active state" in issue for issue in local_issues
    )
    return report, local_issues


def _unsealed_tg_snapshot(path: Path) -> dict[str, Any]:
    """Expose live bootstrap status only as observational, never qualifying evidence."""

    document, error = _read_json(path)
    return {
        "path": str(path),
        "readable": document is not None,
        "observational_only": True,
        "schema": document.get("schema") if document else None,
        "lane": document.get("lane") if document else None,
        "phase": document.get("phase") if document else None,
        "current_artifact": document.get("current_artifact") if document else None,
        "read_error": error,
        "does_not_count_as_qualification": True,
    }


def _candidate_qualification(
    *,
    candidate: CandidateSpec,
    lifecycle_states: Mapping[str, Mapping[str, Any]],
    workflow: Mapping[str, Any] | None,
    physical_models: Mapping[str, Mapping[str, Any]],
    operational_ascent: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    lifecycle_row = _mapping(lifecycle_states.get(candidate.manager_state))
    lifecycle_certified = lifecycle_row.get("status") == "CERTIFIED"
    if not lifecycle_certified:
        issues.append(f"{candidate.key} lifecycle manager floor is not certified")

    candidate_rows = workflow.get("candidates") if workflow else []
    if not isinstance(candidate_rows, list):
        candidate_rows = []
        issues.append("lifecycle workflow lacks candidate qualification rows")
    workflow_rows = {
        str(_mapping(row).get("tournament_artifact_id")): _mapping(row)
        for row in candidate_rows
        if isinstance(row, Mapping)
    }
    workflow_row = _mapping(workflow_rows.get(candidate.artifact_id))
    workflow_certified = workflow_row.get("agent_qualification_state") == "CERTIFIED"
    if not workflow_certified:
        issues.append(f"{candidate.key} workflow manager qualification is not certified")

    model = _mapping(physical_models.get(candidate.key))
    if model.get("gravity_artifact_id") != candidate.artifact_id:
        issues.append(f"{candidate.key} physical gate artifact identity mismatch")
    requirements = _mapping(model.get("requirements"))
    floor_checks: dict[str, bool] = {}
    for requirement in FLOOR_REQUIREMENTS:
        passed = _mapping(requirements.get(requirement)).get("state") == "PASS"
        floor_checks[requirement] = passed
        if not passed:
            issues.append(f"{candidate.key} missing manager-floor requirement {requirement}")
    physical_floor = model.get("pre_final_review_qualification") == "PASS"
    if not physical_floor:
        issues.append(f"{candidate.key} physical complete-manager floor is not PASS")

    tg10_passed = _mapping(requirements.get(TG10_REQUIREMENT)).get("state") == "PASS"
    if not tg10_passed:
        issues.append(f"{candidate.key} TG10 operational requirement is not PASS")
    tg3_passed = _mapping(requirements.get(TG3_REQUIREMENT)).get("state") == "PASS"
    if not tg3_passed:
        issues.append(f"{candidate.key} TG3 requirement is not PASS")
    operational_models = _mapping(_mapping(operational_ascent.get("evidence") if operational_ascent else None).get("models"))
    tg10_seal = _mapping(operational_models.get(candidate.key)).get("tg10_receipt_seal_sha256")
    if not isinstance(tg10_seal, str) or len(tg10_seal) != 64:
        issues.append(f"{candidate.key} operational ascent lacks a sealed TG10 receipt identity")

    return (
        {
            "candidate_artifact_id": candidate.artifact_id,
            "lifecycle_state": {
                "id": candidate.manager_state,
                "status": lifecycle_row.get("status"),
                "certified": lifecycle_certified,
            },
            "workflow_state": {
                "agent_qualification_state": workflow_row.get("agent_qualification_state"),
                "certified": workflow_certified,
            },
            "physical_manager_floor": {
                "pre_final_review_qualification": model.get("pre_final_review_qualification"),
                "passed": physical_floor and all(floor_checks.values()),
                "requirements": floor_checks,
            },
            "tg10": {
                "requirement_passed": tg10_passed,
                "operational_receipt_seal_sha256": tg10_seal,
                "passed": tg10_passed and isinstance(tg10_seal, str) and len(tg10_seal) == 64,
            },
            "tg3": {
                "requirement_passed": tg3_passed,
                "passed": tg3_passed,
            },
            "complete_manager_floor_passed": (
                lifecycle_certified
                and workflow_certified
                and physical_floor
                and all(floor_checks.values())
            ),
            "issues": issues,
        },
        issues,
    )


def build_readiness_report(
    paths: ReadinessPaths | None = None,
    *,
    claimed_ready: bool = False,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build a sealed, in-memory readiness snapshot without changing any state."""

    paths = paths or ReadinessPaths.defaults()
    lifecycle_state, state_issues = _sealed_document(paths.lifecycle_state, label="lifecycle state")
    workflow, workflow_issues = _sealed_document(paths.lifecycle_workflow, label="lifecycle workflow")
    controller, controller_issues = _sealed_document(
        paths.lifecycle_controller, label="lifecycle tournament controller"
    )
    physical_gate, physical_gate_issues = _sealed_document(paths.physical_gate, label="physical gate")
    operational_ascent, operational_issues = _sealed_document(
        paths.operational_ascent, label="operational ascent"
    )

    lifecycle_schema_issues = _require_schema(
        lifecycle_state, LIFECYCLE_STATE_SCHEMA, label="lifecycle state"
    )
    workflow_schema_issues = _require_schema(
        workflow, LIFECYCLE_WORKFLOW_SCHEMA, label="lifecycle workflow"
    )
    controller_schema_issues = _require_schema(
        controller, LIFECYCLE_CONTROLLER_SCHEMA, label="lifecycle tournament controller"
    )
    physical_gate_schema_issues = _require_schema(
        physical_gate, PHYSICAL_GATE_SCHEMA, label="physical gate"
    )
    operational_schema_issues = _require_schema(
        operational_ascent, OPERATIONAL_ASCENT_SCHEMA, label="operational ascent"
    )
    protocol, protocol_issues = _protocol_snapshot(
        lifecycle_state=lifecycle_state,
        workflow=workflow,
        controller=controller,
    )

    lifecycle_states, lifecycle_state_row_issues = _state_rows(lifecycle_state or {})
    physical_models, physical_model_issues = _candidate_rows(physical_gate or {})
    candidate_reports: dict[str, dict[str, Any]] = {}
    candidate_issues: list[str] = []
    for candidate in CANDIDATES:
        report, issues = _candidate_qualification(
            candidate=candidate,
            lifecycle_states=lifecycle_states,
            workflow=workflow,
            physical_models=physical_models,
            operational_ascent=operational_ascent,
        )
        report["live_tg_status"] = _unsealed_tg_snapshot(paths.tg_status(candidate))
        candidate_reports[candidate.key] = report
        candidate_issues.extend(issues)

    physical_gate_mapping = physical_gate or {}
    operational_mapping = operational_ascent or {}
    exact_candidate_order = tuple(physical_gate_mapping.get("fixed_candidate_order") or ()) == CANDIDATE_ARTIFACTS
    if not exact_candidate_order:
        physical_model_issues.append("physical gate candidate order mismatch")
    suite_preflight = _mapping(physical_gate_mapping.get("frozen_protected_tournament_suite_preflight"))
    suite_frozen = suite_preflight.get("state") == "PASS"
    precondition_issues: list[str] = []
    if not suite_frozen:
        precondition_issues.append("frozen protected tournament suite preflight is not PASS")
    operational_tg10 = (
        operational_mapping.get("both_valid_tg10_receipts") is True
        and operational_mapping.get("status") == "BOTH_VALID_TG10_OPERATIONAL_RECEIPTS_BOUND"
    )
    if not operational_tg10:
        precondition_issues.append("both exact sealed TG10 operational receipts are not bound")
    if _mapping(operational_mapping.get("protected_tournament")).get("tg3_remains_required") is not True:
        precondition_issues.append("operational ascent no longer preserves TG3 requirement")

    source_authorities: dict[str, dict[str, Any]] = {}
    materialized_authorities: dict[str, dict[str, Any]] = {}
    authority_issues: list[str] = []
    for authority in AUTHORITY_SPECS:
        source_identity, source_issues = _source_identity(authority, paths.repository_root)
        materialized, materialized_issues = _materialized_authority(authority, paths.authority_root)
        source_authorities[authority.key] = source_identity
        materialized_authorities[authority.key] = materialized
        authority_issues.extend(source_issues)
        authority_issues.extend(materialized_issues)

    complete_manager_floor = all(
        candidate_reports[candidate.key]["complete_manager_floor_passed"] for candidate in CANDIDATES
    )
    both_tg10 = operational_tg10 and all(
        candidate_reports[candidate.key]["tg10"]["passed"] for candidate in CANDIDATES
    )
    both_tg3 = all(candidate_reports[candidate.key]["tg3"]["passed"] for candidate in CANDIDATES)
    protected_contract_identities = not authority_issues
    tg3_freeze_prepared = materialized_authorities["tg3_freeze"]["prepared"]
    protected_final_documents_prepared = all(
        materialized_authorities[key]["prepared"]
        for key in ("protected_corpus", "scorecard_adjudication", "selection_recovery", "final_report")
    )
    all_lifecycle_documents_trusted = not (
        state_issues
        + workflow_issues
        + controller_issues
        + physical_gate_issues
        + operational_issues
        + lifecycle_schema_issues
        + workflow_schema_issues
        + physical_gate_schema_issues
        + operational_schema_issues
        + controller_schema_issues
        + lifecycle_state_row_issues
        + physical_model_issues
    )

    conjunctive_prerequisites = [
        {
            "id": "fixed_final_manager_protocol_identity",
            "required": "the canonical protocol and every live controller/workflow binding match the pinned identity",
            "satisfied": protocol["trusted"],
        },
        {
            "id": "trusted_live_lifecycle_and_physical_records",
            "required": "sealed lifecycle, workflow, controller, physical-gate, and operational-ascent records",
            "satisfied": all_lifecycle_documents_trusted,
        },
        {
            "id": "both_complete_manager_floors",
            "required": "both candidate lifecycle/workflow floors and all canonical physical manager-floor receipts",
            "satisfied": complete_manager_floor,
        },
        {
            "id": "both_exact_tg10_operational_receipts",
            "required": "both exact sealed BASE_TRUE_TPS TG10 operational receipts",
            "satisfied": both_tg10,
        },
        {
            "id": "both_tg3_qualifications",
            "required": "both canonical physical TG3 >=333 full-model qualifications",
            "satisfied": both_tg3,
        },
        {
            "id": "frozen_protected_tournament_suite",
            "required": "the sealed protected suite preflight remains PASS",
            "satisfied": suite_frozen,
        },
        {
            "id": "prepared_paired_and_protected_contract_identities",
            "required": "every paired, TG3/freeze, corpus, scorecard, selection/recovery, and report contract source and sealed materialization",
            "satisfied": protected_contract_identities,
        },
        {
            "id": "sealed_post_tg3_freeze_authority",
            "required": "the prepared sealed TG3/freeze final-comparison authority",
            "satisfied": tg3_freeze_prepared,
        },
        {
            "id": "protected_corpus_scorecard_selection_and_report_contracts",
            "required": "prepared sealed corpus, scorecard, selection/recovery, and final-report contracts",
            "satisfied": protected_final_documents_prepared,
        },
    ]
    missing_prerequisites = [
        row["id"] for row in conjunctive_prerequisites if row["satisfied"] is not True
    ]
    if claimed_ready:
        missing_prerequisites.append("caller_readiness_claim_rejected")

    blockers = sorted(
        set(
            state_issues
            + workflow_issues
            + controller_issues
            + physical_gate_issues
            + operational_issues
            + lifecycle_schema_issues
            + workflow_schema_issues
            + physical_gate_schema_issues
            + operational_schema_issues
            + controller_schema_issues
            + protocol_issues
            + lifecycle_state_row_issues
            + physical_model_issues
            + precondition_issues
            + candidate_issues
            + authority_issues
            + (["caller readiness claim is rejected by this read-only observer"] if claimed_ready else [])
        )
    )
    ready = not missing_prerequisites and not blockers
    status = STATUS_PREPARED if ready else STATUS_REFUSED
    report = {
        "schema": SCHEMA,
        "status": status,
        "recorded_at": recorded_at or _utc_now(),
        "prepared_not_active": ready,
        "readiness_claimed_by_caller": claimed_ready,
        "readiness_claim_accepted": False,
        "tournament_ready_for_execution": False,
        "winner_selected": False,
        "external_report_emitted": False,
        "protocol": protocol,
        "live_records": {
            "lifecycle_state": _document_identity(paths.lifecycle_state, lifecycle_state, state_issues),
            "lifecycle_workflow": _document_identity(paths.lifecycle_workflow, workflow, workflow_issues),
            "lifecycle_controller": _document_identity(
                paths.lifecycle_controller, controller, controller_issues
            ),
            "physical_gate": _document_identity(paths.physical_gate, physical_gate, physical_gate_issues),
            "operational_ascent": _document_identity(
                paths.operational_ascent, operational_ascent, operational_issues
            ),
        },
        "candidate_qualification": candidate_reports,
        "prepared_authority_source_identities": source_authorities,
        "prepared_authority_materializations": materialized_authorities,
        "conjunctive_prerequisites": conjunctive_prerequisites,
        "missing_prerequisites": missing_prerequisites,
        "blockers": blockers,
        "claim_boundary": {
            "read_only": True,
            "creates_tasks": False,
            "activates_tournament": False,
            "starts_model_process": False,
            "starts_server": False,
            "starts_gpu_work": False,
            "starts_watcher": False,
            "acquires_lease": False,
            "measures_tps": False,
            "chooses_winner": False,
            "writes_campaign_state": False,
        },
    }
    return seal(report)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one report to stdout; there is intentionally no output-file option."""

    defaults = ReadinessPaths.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(defaults.repository_root))
    parser.add_argument("--lifecycle-root", default=str(defaults.lifecycle_root))
    parser.add_argument("--physical-root", default=str(defaults.physical_root))
    parser.add_argument("--authority-root", default=str(defaults.authority_root))
    parser.add_argument(
        "--claim-ready",
        action="store_true",
        help="exercise the fail-closed rejection path; this never authorizes readiness",
    )
    args = parser.parse_args(argv)
    paths = ReadinessPaths.from_roots(
        repository_root=args.repository_root,
        lifecycle_root=args.lifecycle_root,
        physical_root=args.physical_root,
        authority_root=args.authority_root,
    )
    print(json.dumps(build_readiness_report(paths, claimed_ready=args.claim_ready), indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUTHORITY_SPECS",
    "CANDIDATES",
    "EXPECTED_PROTOCOL_IDENTITY_SHA256",
    "ReadinessPaths",
    "SCHEMA",
    "STATUS_PREPARED",
    "STATUS_REFUSED",
    "build_readiness_report",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
