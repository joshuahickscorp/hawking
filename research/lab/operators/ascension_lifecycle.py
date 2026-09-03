"""Fail-closed, restart-safe Ascension V3 lifecycle and tournament controller.

The Ascension Bible deliberately makes this more than a progress checklist.  A
state may advance only from sealed, controller- or human-certified evidence;
the controller never turns a model report, a plan, an elapsed time, or a
running daemon into a certification.  This module is therefore safe to leave
running for a long campaign: it continuously recomputes the next admissible
piece of work, preserves every pending condition, and arms the Manager
Tournament without ever inventing a winner.

Evidence is intentionally intake-only.  Future workers write sealed documents
to ``controller-evidence`` using the exact artifact identities listed in the
Bible.  The controller validates those documents and advances the graph.  It
does not download model bodies, run a model, delete data, evict an alternate,
or promote a candidate itself.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.operators.ascension_kernel_registry import (
    ARCHITECTURE_FINGERPRINT_FIELDS,
    FAMILY_PLUGINS,
    GRAVITY_COMPONENTS,
    MODEL_PROGRAM_KEY_FIELDS,
    REPRESENTATION_TOURNAMENT_CLASSES,
    SHARED_PRIMITIVES,
)
from lab.operators.ascension_manager_tournament_protocol import (
    SCHEMA as FINAL_MANAGER_TOURNAMENT_PROTOCOL_SCHEMA,
    build_final_manager_tournament_protocol,
    validate_final_manager_tournament_result,
)
from lab.operators.ascension_foundation_contracts import (
    AGENT_OS_PERFORMANCE_GATES,
    AGENT_ROLES,
    AGENT_SCHEDULER_CAPABILITIES,
    CONTEXT_COMPILER_INPUTS,
    CONTEXT_COMPILER_PROPERTIES,
    ENERGY_EVIDENCE_OUTPUTS,
    GROK_LANE_CLASSES,
    GROK_LANE_CONTRACT_FIELDS,
    GROK_RESOURCE_CLASSES,
    GROK_SCHEDULER_RULES,
    KV_STATE_CAPABILITIES,
    PRESSURE_MODES,
    RESOURCE_TELEMETRY_FIELDS,
    SCHEDULER_SELECTION_OBJECTIVES,
    TASK_SCHEDULER_FIELDS,
)
from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.v3_lifecycle.v1"
STATE_SCHEMA = "hawking.ascension.v3_state.v1"
TOURNAMENT_CONTROLLER_SCHEMA = "hawking.ascension.manager_tournament_controller.v1"
CONSTITUTION_SCHEMA = "hawking.ascension.v3_constitution_controller.v1"
WORK_QUEUE_SCHEMA = "hawking.ascension.v3_work_queue.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIBLE = REPO_ROOT.parent / "bible.md"
DEFAULT_SANDBOX_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox"
)
DEFAULT_ROOT = DEFAULT_SANDBOX_ROOT / "lifecycle"

MODEL_30B = "Qwen3-Coder-30B-A3B-Instruct"
MODEL_80B = "Qwen3-Coder-Next-80B"
# Raw BF16 models remain source authorities / teachers.  The protected
# tournament compares only these completed, independently-qualified Gravity
# artifacts.  Keeping the identities separate prevents a raw source body from
# ever being substituted for its manager artifact in a receipt.
QWEN30_GRAVITY_MANAGER_ARTIFACT = "Qwen30-Gravity-Manager-Artifact"
QWEN80_GRAVITY_MANAGER_ARTIFACT = "Qwen80-Gravity-Manager-Artifact"
MANAGER_CANDIDATE_ORDER = (MODEL_30B, MODEL_80B)
TOURNAMENT_CANDIDATE_ORDER = (
    QWEN30_GRAVITY_MANAGER_ARTIFACT,
    QWEN80_GRAVITY_MANAGER_ARTIFACT,
)
MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR = 100.0

CANONICAL_STATES: tuple[str, ...] = (
    "V3_ADOPT",
    "V3_SEED_ARCHIVE",
    "V3_AUTHORITY_FREEZE",
    "V3_GROK_BUILD_FABRIC",
    "V3_AGENT_OS",
    "V3_KNOWLEDGE_PLANE",
    "V3_GRAVITY",
    "V3_METAL_COMPILER",
    "MANAGER_30B_DENSITY",
    "MANAGER_30B_TG",
    "MANAGER_30B_AGENT",
    "MANAGER_80B_DENSITY",
    "MANAGER_80B_TG",
    "MANAGER_80B_AGENT",
    "MANAGER_TOURNAMENT",
    "SANDBOX_ACTIVATION",
    "FAMILY_QWEN",
    "FAMILY_LLAMA",
    "FAMILY_MISTRAL",
    "FAMILY_DEEPSEEK",
    "FAMILY_GLM",
    "FAMILY_KIMI",
    "FAMILY_GEMMA",
    "FAMILY_HYBRID",
    "GLOBAL_LAUNCH_AUDIT",
    "EXTERNAL_REVIEW",
    "APPLE_RELEASE",
    "TG2_TG1_FRONTIER",
)

EXACT_CONTINUATION_OUTPUTS: tuple[str, ...] = (
    "ASCENSION_V3_STATE.json",
    "ASCENSION_V3_NEXT_COMMAND.sh",
    "ASCENSION_V3_ACTIVE_LANES.json",
    "ASCENSION_V3_GATE_MATRIX.json",
    "ASCENSION_V3_REVIEW_INDEX.json",
)

TOURNAMENT_DIMENSIONS: tuple[str, ...] = (
    "hard_gate_compliance",
    "manager_capability",
    "solo_manager_capability",
    "manager_as_orchestrator_capability",
    "verified_tasks_per_hour",
    "architecture_kernel_research",
    "gravity_research_quality",
    "long_horizon_reliability",
    "tool_recovery",
    "search_retrieval",
    "memory_context",
    "hcli_latency",
    "multi_agent_throughput",
    "resident_memory",
    "active_bytes_per_token",
    "energy_per_verified_task",
    "thermal_stability",
    "error_rate",
    "receipt_quality",
    "adversarial_review_quality",
    "failure_recovery",
    "benchmark_honesty",
    "security_effect_discipline",
    "release_integration",
)

CONTROLLER_AUTHORITIES = frozenset({"protected_controller", "human_operator"})


class AscensionLifecycleError(RuntimeError):
    """The controller cannot safely create or evaluate its own state."""


@dataclass(frozen=True)
class ArtifactRule:
    """One source-of-truth artifact required by a V3 state."""

    artifact_id: str
    filename: str
    description: str
    check: str = "generic"
    model_id: str | None = None
    family: str | None = None
    source_bound: bool = False
    format: str = "json"


@dataclass(frozen=True)
class StageSpec:
    """A canonical state plus its non-negotiable evidence dependencies."""

    state_id: str
    completion_states: tuple[str, ...]
    description: str
    prerequisites: tuple[str, ...]
    artifacts: tuple[ArtifactRule, ...]
    resource_class: str


@dataclass(frozen=True)
class LifecyclePaths:
    """Controller-owned locations.  Evidence is never written by the controller."""

    root: Path
    evidence_root: Path
    state_path: Path
    active_lanes_path: Path
    gate_matrix_path: Path
    review_index_path: Path
    next_command_path: Path
    constitution_path: Path
    launch_gate_path: Path
    work_queue_path: Path
    tournament_controller_path: Path
    fidelity_path: Path
    lock_path: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "LifecyclePaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            evidence_root=resolved / "controller-evidence",
            state_path=resolved / "ASCENSION_V3_STATE.json",
            active_lanes_path=resolved / "ASCENSION_V3_ACTIVE_LANES.json",
            gate_matrix_path=resolved / "ASCENSION_V3_GATE_MATRIX.json",
            review_index_path=resolved / "ASCENSION_V3_REVIEW_INDEX.json",
            next_command_path=resolved / "ASCENSION_V3_NEXT_COMMAND.sh",
            constitution_path=resolved / "ASCENSION_V3_CONSTITUTION.json",
            launch_gate_path=resolved / "ASCENSION_V3_LAUNCH_GATE.py",
            work_queue_path=resolved / "ASCENSION_V3_WORK_QUEUE.json",
            tournament_controller_path=resolved / "ASCENSION_MANAGER_TOURNAMENT_CONTROLLER.json",
            fidelity_path=resolved / "ASCENSION_V3_FIDELITY_REPORT.json",
            lock_path=resolved / ".lifecycle.lock",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_json(value)
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _atomic_text(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        if mode is not None:
            os.chmod(path, mode)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        mode=0o640,
    )


def _ensure_real_directory(path: Path, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise AscensionLifecycleError(f"lifecycle path must be a real directory: {path}")
    os.chmod(path, mode)


def _state_machine_from_bible(text: str) -> tuple[str, ...]:
    """Extract only §17.1, so a random token elsewhere cannot satisfy the check."""

    match = re.search(r"^# 17\.1\b.*?(?=^# 17\.2\b)", text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        return ()
    values: list[str] = []
    for line in match.group(0).splitlines():
        token = line.strip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]+", token):
            values.append(token)
    return tuple(values)


def audit_bible(bible_path: str | Path = DEFAULT_BIBLE) -> dict[str, Any]:
    """Return a direct V3 contract audit without interpreting narrative status."""

    path = Path(bible_path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "state_machine_matches": False,
            "issues": [f"cannot read Bible: {exc}"],
        }
    observed = _state_machine_from_bible(text)
    issues: list[str] = []
    if observed != CANONICAL_STATES:
        issues.append(
            "Bible §17.1 differs from the controller's exact canonical V3 state machine"
        )
    for output in EXACT_CONTINUATION_OUTPUTS:
        if output not in text:
            issues.append(f"Bible continuation output missing from source text: {output}")
    return {
        "path": str(path),
        "exists": True,
        "sha256": _file_sha256(path),
        "observed_state_machine": list(observed),
        "expected_state_machine": list(CANONICAL_STATES),
        "state_machine_matches": observed == CANONICAL_STATES,
        "issues": issues,
    }


def _rule(
    artifact_id: str,
    description: str,
    *,
    filename: str | None = None,
    check: str = "generic",
    model_id: str | None = None,
    family: str | None = None,
    source_bound: bool = False,
    format: str = "json",
) -> ArtifactRule:
    suffix = ".jsonl" if format == "jsonl" else ".json"
    return ArtifactRule(
        artifact_id=artifact_id,
        filename=filename or f"{artifact_id}{suffix}",
        description=description,
        check=check,
        model_id=model_id,
        family=family,
        source_bound=source_bound,
        format=format,
    )


SEED_ARCHIVE_ARTIFACTS = (
    _rule("ASCENSION_V3_SEED_ARCHIVE", "sealed V3 Seed Archive"),
    _rule("ASCENSION_V3_PROTECTED_OBJECTS", "protected-object inventory"),
    _rule("ASCENSION_V3_PROCESS_AUDIT", "live process audit"),
    _rule(
        "ASCENSION_V3_GRAVEYARD_IMPORT",
        "negative-science import ledger",
        format="jsonl",
    ),
    _rule(
        "ASCENSION_V3_KERNEL_GENOME_IMPORT",
        "kernel-genome import ledger",
        format="jsonl",
    ),
    _rule(
        "ASCENSION_V3_REPRESENTATION_GENOME_IMPORT",
        "representation-genome import ledger",
        format="jsonl",
    ),
    _rule(
        "ASCENSION_V3_RESTORE",
        "tested Seed Archive restore command",
        filename="ASCENSION_V3_RESTORE.sh",
        format="script",
    ),
    _rule("ASCENSION_V3_STORAGE_RECOVERY_RECEIPT", "storage recovery receipt"),
)

AGENT_OS_ARTIFACTS = (
    _rule("HCLI_AGENT_OS_V3_STATUS", "live Agent OS V3 status", check="agent_os_foundation"),
    _rule("HCLI_CONTEXT_COMPILER_RECEIPT", "context compiler receipt"),
    _rule("HCLI_KV_STATE_MANAGER_RECEIPT", "KV/state manager receipt"),
    _rule("HCLI_CONTINUOUS_BATCHING_SCOREBOARD", "continuous batching scoreboard"),
    _rule("HCLI_VERIFIED_TASKS_PER_HOUR", "verified tasks/hour receipt"),
    _rule("HCLI_RESTART_AND_RECOVERY", "HCLI restart and recovery receipt"),
)

KNOWLEDGE_PLANE_ARTIFACTS = (
    _rule("ASCENSION_KERNEL_GENOME", "kernel genome", format="jsonl"),
    _rule("ASCENSION_REPRESENTATION_GENOME", "representation genome", format="jsonl"),
    _rule("ASCENSION_SCHEDULER_GENOME", "scheduler genome", format="jsonl"),
    _rule("ASCENSION_NEGATIVE_SCIENCE", "negative-science graveyard", format="jsonl"),
    _rule("ASCENSION_TRANSFER_MATRIX", "cross-family transfer matrix"),
    _rule(
        "ASCENSION_MECHANISM_INDEX",
        "mechanism-index SQLite database",
        filename="ASCENSION_MECHANISM_INDEX.sqlite",
        format="sqlite",
    ),
)

GRAVITY_ARTIFACTS = (
    _rule("GRAVITY_V3_FRONTIER", "Gravity V3 frontier", check="gravity_foundation"),
    _rule("GRAVITY_V3_QAT_DOCTOR_INDEX", "QAT/Doctor index"),
)

METAL_ARTIFACTS = (
    _rule("METAL_FAMILY_PLUGIN_MATRIX", "Metal family plugin matrix", check="metal_foundation"),
    _rule("METAL_EXACT_MODEL_COMPILER_STATUS", "exact-model compiler status", check="metal_foundation"),
)


def _manager_artifacts(prefix: str, model_id: str) -> tuple[ArtifactRule, ...]:
    return (
        _rule(
            f"{prefix}_MANAGER_SOURCE",
            f"official high-precision source authority for {model_id}",
            check="manager_source",
            model_id=model_id,
            source_bound=True,
        ),
        _rule(
            f"{prefix}_MANAGER_CAPABILITY_ANCHOR",
            f"frozen capability anchor for {model_id}",
            check="manager_anchor",
            model_id=model_id,
            source_bound=True,
        ),
        _rule(
            f"{prefix}_MANAGER_GRAVITY",
            f"complete-BPW Gravity qualification for {model_id}",
            check="manager_gravity",
            model_id=model_id,
            source_bound=True,
        ),
        _rule(
            f"{prefix}_MANAGER_TG3",
            f"TG3 CLEAN runtime qualification for {model_id}",
            check="manager_tg3",
            model_id=model_id,
            source_bound=True,
        ),
        _rule(
            f"{prefix}_MANAGER_KERNEL_OPERATIONAL",
            f"exact-model custom-kernel operational gate for {model_id}",
            check="manager_kernel_operational",
            model_id=model_id,
            source_bound=True,
        ),
        _rule(
            f"{prefix}_MANAGER_AGENT_OS",
            f"HCLI, residency, and manager-contract qualification for {model_id}",
            check="manager_agent",
            model_id=model_id,
            source_bound=True,
        ),
    )


MANAGER_30_ARTIFACTS = _manager_artifacts("QWEN30", MODEL_30B)
MANAGER_80_ARTIFACTS = _manager_artifacts("QWEN80", MODEL_80B)

FAMILY_RULES: tuple[tuple[str, str, str], ...] = (
    ("FAMILY_QWEN", "QWEN_V3_LAUNCH_READY", "QWEN"),
    ("FAMILY_LLAMA", "LLAMA_V3_LAUNCH_READY", "LLAMA"),
    ("FAMILY_MISTRAL", "MISTRAL_MIXTRAL_V3_LAUNCH_READY", "MISTRAL_MIXTRAL"),
    ("FAMILY_DEEPSEEK", "DEEPSEEK_V3_LAUNCH_READY", "DEEPSEEK"),
    ("FAMILY_GLM", "GLM_V3_LAUNCH_READY", "GLM"),
    ("FAMILY_KIMI", "KIMI_V3_LAUNCH_READY", "KIMI"),
    ("FAMILY_GEMMA", "GEMMA_V3_LAUNCH_READY", "GEMMA"),
    ("FAMILY_HYBRID", "HYBRID_V3_LAUNCH_READY", "STATE_SPACE_OR_LINEAR_ATTENTION_HYBRID"),
)


def _build_stage_specs() -> tuple[StageSpec, ...]:
    specs: list[StageSpec] = [
        StageSpec(
            "V3_ADOPT",
            ("ASCENSION_V3_ADOPTED",),
            "Adopt the supplied Ascension V3 Bible as the sole canonical programme.",
            (),
            (_rule("ASCENSION_V3_ADOPTED", "human/controller V3 adoption", check="adoption"),),
            "CONTROLLER",
        ),
        StageSpec(
            "V3_SEED_ARCHIVE",
            ("ASCENSION_V3_SEED_ARCHIVE_SEALED",),
            "Archive live state, protected objects, prior evidence, and restoration proof.",
            ("V3_ADOPT",),
            SEED_ARCHIVE_ARTIFACTS,
            "DISK_HEAVY",
        ),
        StageSpec(
            "V3_AUTHORITY_FREEZE",
            ("ASCENSION_V3_AUTHORITY_FROZEN",),
            "Freeze authorities, hidden memberships, rollback, and deletion policy.",
            ("V3_SEED_ARCHIVE",),
            (_rule("ASCENSION_V3_AUTHORITY_FROZEN", "authority freeze", check="authority_freeze"),),
            "CONTROLLER",
        ),
        StageSpec(
            "V3_GROK_BUILD_FABRIC",
            ("ASCENSION_V3_GROK_BUILD_FABRIC_READY", "ASCENSION_V3_RESOURCE_GOVERNOR_READY"),
            "Install isolated build lanes plus resource/thermal governance.",
            ("V3_AUTHORITY_FREEZE",),
            (
                _rule("ASCENSION_V3_GROK_BUILD_FABRIC_READY", "Grok build fabric", check="build_fabric"),
                _rule("ASCENSION_V3_RESOURCE_GOVERNOR_READY", "resource governor", check="resource_governor"),
            ),
            "CPU_HEAVY",
        ),
        StageSpec(
            "V3_AGENT_OS",
            ("HCLI_AGENT_OS_V3_READY",),
            "Wire scheduler, context, tools, memory, recovery, and HCLI into the live product path.",
            ("V3_GROK_BUILD_FABRIC",),
            AGENT_OS_ARTIFACTS,
            "CPU_HEAVY",
        ),
        StageSpec(
            "V3_KNOWLEDGE_PLANE",
            ("ASCENSION_KNOWLEDGE_PLANE_READY",),
            "Build reusable Kernel, Representation, Scheduler, and negative-science knowledge planes.",
            ("V3_AGENT_OS",),
            KNOWLEDGE_PLANE_ARTIFACTS,
            "CPU_HEAVY",
        ),
        StageSpec(
            "V3_GRAVITY",
            ("GRAVITY_V3_READY",),
            "Establish the evolutionary Gravity frontier and Doctor/QAT evidence discipline.",
            ("V3_KNOWLEDGE_PLANE",),
            GRAVITY_ARTIFACTS,
            "GPU_HEAVY",
        ),
        StageSpec(
            "V3_METAL_COMPILER",
            ("METAL_EXACT_MODEL_COMPILER_V3_READY",),
            "Establish family semantics and exact-model Metal generation.",
            ("V3_GRAVITY",),
            METAL_ARTIFACTS,
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_30B_DENSITY",
            ("QWEN30_MANAGER_DENSITY_GATE_READY",),
            "Freeze the Qwen 30B source/capability anchor and earn <=1.5 complete BPW.",
            ("V3_METAL_COMPILER",),
            MANAGER_30_ARTIFACTS[:3],
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_30B_TG",
            ("QWEN30_MANAGER_TG3_GATE_READY",),
            "Earn Qwen 30B TG3 on a CLEAN, real-Metal, no-fallback runtime.",
            ("MANAGER_30B_DENSITY",),
            (MANAGER_30_ARTIFACTS[3],),
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_30B_AGENT",
            ("QWEN30_MANAGER_CANDIDATE",),
            "Qualify Qwen 30B's HCLI/Agent OS, residency, recovery, and unattended campaign behaviour.",
            ("MANAGER_30B_TG",),
            MANAGER_30_ARTIFACTS[4:],
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_80B_DENSITY",
            ("QWEN80_MANAGER_DENSITY_GATE_READY",),
            "Freeze the Qwen Next 80B source/capability anchor and earn <=1.5 complete BPW.",
            ("MANAGER_30B_AGENT",),
            MANAGER_80_ARTIFACTS[:3],
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_80B_TG",
            ("QWEN80_MANAGER_TG3_GATE_READY",),
            "Earn Qwen Next 80B TG3 on its exact hybrid runtime.",
            ("MANAGER_80B_DENSITY",),
            (MANAGER_80_ARTIFACTS[3],),
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_80B_AGENT",
            ("QWEN80_MANAGER_CANDIDATE",),
            "Qualify Qwen Next 80B's HCLI/Agent OS, residency, recovery, and unattended campaign behaviour.",
            ("MANAGER_80B_TG",),
            MANAGER_80_ARTIFACTS[4:],
            "GPU_HEAVY",
        ),
        StageSpec(
            "MANAGER_TOURNAMENT",
            (
                "ASCENSION_MANAGER_TOURNAMENT_READY",
                "ASCENSION_MANAGER_SEALED",
                "ASCENSION_ALTERNATE_OFFLOADED",
            ),
            "Run the fixed, protected two-manager tournament, seal its winner, and offload the alternate.",
            ("MANAGER_30B_AGENT", "MANAGER_80B_AGENT"),
            (
                _rule("ASCENSION_MANAGER_TOURNAMENT", "fixed protected manager tournament", check="tournament"),
                _rule("ASCENSION_MANAGER_WINNER", "protected controller manager winner", check="winner"),
                _rule("ASCENSION_ALTERNATE_OFFLOAD", "alternate cold-store and local eviction", check="alternate_offload"),
            ),
            "GPU_HEAVY",
        ),
        StageSpec(
            "SANDBOX_ACTIVATION",
            ("ASCENSION_SANDBOX_ACTIVE",),
            "Activate the production Ascension sandbox only under the sealed winner.",
            ("MANAGER_TOURNAMENT",),
            (_rule("ASCENSION_SANDBOX_ACTIVE", "external sandbox activation receipt", check="sandbox_activation"),),
            "GPU_HEAVY",
        ),
    ]
    for state_id, completion, family in FAMILY_RULES:
        specs.append(
            StageSpec(
                state_id,
                (completion,),
                f"Qualify an exact production representative for the {family} semantic family.",
                ("SANDBOX_ACTIVATION",),
                (
                    _rule(
                        completion,
                        f"exact-model {family} launch qualification",
                        check="family_launch",
                        family=family,
                        source_bound=True,
                    ),
                ),
                "GPU_HEAVY",
            )
        )
    family_states = tuple(state for state, _completion, _family in FAMILY_RULES)
    specs.extend(
        (
            StageSpec(
                "GLOBAL_LAUNCH_AUDIT",
                (
                    "ASCENSION_V3_CORE_FAMILY_MATRIX_READY",
                    "ASCENSION_V3_ALL_ADVERTISED_MODELS_QUALIFIED",
                ),
                "Complete all family, model, density, TG, resource, recovery, and product matrices.",
                family_states,
                (
                    _rule("GENERIC_HF_REFERENCE_READY", "generic Hugging Face intake reference", check="generic_reference"),
                    _rule("ASCENSION_V3_FAMILY_MATRIX", "core family matrix", check="matrix"),
                    _rule("ASCENSION_V3_MODEL_MATRIX", "advertised model matrix", check="matrix"),
                    _rule("ASCENSION_V3_DENSITY_MATRIX", "complete-BPW matrix", check="matrix"),
                    _rule("ASCENSION_V3_TG_MATRIX", "TG matrix", check="matrix"),
                    _rule("ASCENSION_V3_RESOURCE_ATLAS", "resource atlas", check="matrix"),
                    _rule("ASCENSION_V3_POWER_PROFILES", "power profiles", check="matrix"),
                    _rule("ASCENSION_V3_STORAGE_LEASES", "storage leases", check="matrix"),
                    _rule("ASCENSION_V3_GARBAGE_AUDIT", "garbage audit", check="matrix"),
                    _rule("ASCENSION_V3_RECOVERY_TEST", "recovery test", check="matrix"),
                    _rule("HAWKING_APPLE_V3_INSTALL_TEST", "Apple install test", check="product"),
                    _rule("HAWKING_APPLE_V3_UPDATE_TEST", "Apple update test", check="product"),
                    _rule("HAWKING_APPLE_V3_HCLI_PRODUCT_TEST", "HCLI product test", check="product"),
                    _rule("HAWKING_APPLE_V3_RELEASE_AUDIT", "Apple release audit", check="product"),
                    _rule("ASCENSION_V3_CORE_FAMILY_MATRIX_READY", "certified core family matrix", check="global_audit"),
                    _rule("ASCENSION_V3_ALL_ADVERTISED_MODELS_QUALIFIED", "certified advertised-model matrix", check="global_audit"),
                ),
                "CONTROLLER",
            ),
            StageSpec(
                "EXTERNAL_REVIEW",
                ("HAWKING_ASCENSION_V3_REVIEW_REQUIRED",),
                "Publish a compact V3 review packet and receive independent external review.",
                ("GLOBAL_LAUNCH_AUDIT",),
                (
                    _rule("HAWKING_ASCENSION_V3_REVIEW_REQUIRED", "external review packet", check="external_review"),
                    _rule("HAWKING_ASCENSION_V3_EXTERNAL_REVIEW_ACCEPTED", "accepted external review", check="external_review_acceptance"),
                ),
                "CONTROLLER",
            ),
            StageSpec(
                "APPLE_RELEASE",
                ("HAWKING_APPLE_V3_PRODUCTION_RELEASE_READY",),
                "Seal the Apple-first production release only after every launch gate and review pass.",
                ("EXTERNAL_REVIEW",),
                (_rule("HAWKING_APPLE_V3_PRODUCTION_RELEASE_READY", "Apple production release", check="apple_release"),),
                "CONTROLLER",
            ),
            StageSpec(
                "TG2_TG1_FRONTIER",
                ("HAWKING_TG2_TG1_FRONTIER_ACTIVE",),
                "Open post-release TG2/TG1 research without delaying launch completion.",
                ("APPLE_RELEASE",),
                (_rule("HAWKING_TG2_TG1_FRONTIER_ACTIVE", "post-release TG frontier", check="frontier"),),
                "GPU_HEAVY",
            ),
        )
    )
    result = tuple(specs)
    if tuple(spec.state_id for spec in result) != CANONICAL_STATES:
        raise AssertionError("stage specification order must match Bible §17.1 exactly")
    return result


STAGE_SPECS = _build_stage_specs()
STAGE_BY_ID = {spec.state_id: spec for spec in STAGE_SPECS}


def _normal(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _same_model(value: object, expected: str) -> bool:
    return _normal(value) == _normal(expected)


def _hash_like(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", value) is not None


def _lookup(document: Mapping[str, Any], *paths: str) -> Any:
    for dotted in paths:
        cursor: Any = document
        found = True
        for key in dotted.split("."):
            if isinstance(cursor, Mapping) and key in cursor:
                cursor = cursor[key]
            else:
                found = False
                break
        if found:
            if isinstance(cursor, Mapping) and "value" in cursor:
                return cursor["value"]
            return cursor
    return None


def _truth(document: Mapping[str, Any], *paths: str) -> bool:
    return _lookup(document, *paths) is True


def _number(document: Mapping[str, Any], *paths: str) -> float | None:
    value = _lookup(document, *paths)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _string(document: Mapping[str, Any], *paths: str) -> str | None:
    value = _lookup(document, *paths)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _certifier(document: Mapping[str, Any]) -> str | None:
    value = _string(
        document,
        "certified_by",
        "certifier",
        "certification.principal",
        "certification.certified_by",
        "authority.principal",
    )
    return value.lower() if value else None


def _controller_certified(document: Mapping[str, Any]) -> bool:
    status_values = (
        _string(document, "status"),
        _string(document, "certification.status"),
        _string(document, "authority.status"),
    )
    return any(value == "CONTROLLER_CERTIFIED" for value in status_values if value)


def _artifact_identity(document: Mapping[str, Any], artifact_id: str) -> bool:
    candidates = (
        _string(document, "artifact_id"),
        _string(document, "completion_state"),
        _string(document, "artifact"),
        _string(document, "id"),
    )
    return any(value == artifact_id for value in candidates if value)


def _common_issues(rule: ArtifactRule, document: Mapping[str, Any], bible: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not _artifact_identity(document, rule.artifact_id):
        issues.append(f"artifact identity must be exactly {rule.artifact_id}")
    if not _controller_certified(document):
        issues.append("artifact is not CONTROLLER_CERTIFIED")
    certifier = _certifier(document)
    if certifier not in CONTROLLER_AUTHORITIES:
        issues.append("artifact certifier must be protected_controller or human_operator")
    if str(document.get("authority_level") or "").lower() == "candidate":
        issues.append("candidate/report-only artifact cannot certify a V3 state")
    if _truth(document, "timeline_based_completion", "certification.timeline_based_completion"):
        issues.append("timeline-based completion is forbidden")
    if str(_lookup(document, "evidence_basis") or "").lower() == "timeline":
        issues.append("timeline is not admissible evidence")
    if rule.model_id is not None:
        observed = _string(document, "model_id", "model.identity", "source.model_id")
        if not _same_model(observed, rule.model_id):
            issues.append(f"artifact is not bound to exact model {rule.model_id}")
        if certifier is not None and _same_model(certifier, rule.model_id):
            issues.append("a manager model cannot certify its own promotion")
    if rule.source_bound:
        if not _truth(document, "source_bound", "source.source_bound"):
            issues.append("source-bound evidence is required")
        if not _truth(
            document,
            "official_high_precision_source",
            "source.official_high_precision_source",
        ):
            issues.append("official high-precision source authority is required")
        source_hash = _string(
            document,
            "source_hash",
            "source.sha256",
            "source.revision_hash",
            "source.revision",
        )
        if not _hash_like(source_hash):
            issues.append("immutable source hash is required")
    if rule.check == "adoption":
        if not _truth(document, "adopted_as_sole_canonical_programme"):
            issues.append("V3 must be adopted as the sole canonical programme")
        if _string(document, "bible_sha256") != bible.get("sha256"):
            issues.append("adoption receipt Bible hash does not match the live Bible")
    return issues


def _require_true(document: Mapping[str, Any], issues: list[str], label: str, *paths: str) -> None:
    if not _truth(document, *paths):
        issues.append(f"{label} must be true")


def _require_number_at_least(
    document: Mapping[str, Any], issues: list[str], label: str, minimum: float, *paths: str
) -> None:
    value = _number(document, *paths)
    if value is None or value + 1e-9 < minimum:
        issues.append(f"{label} must be >= {minimum:g}")


def _require_number_at_most(
    document: Mapping[str, Any], issues: list[str], label: str, maximum: float, *paths: str
) -> None:
    value = _number(document, *paths)
    if value is None or value > maximum + 1e-9:
        issues.append(f"{label} must be <= {maximum:g}")


def _require_members(
    document: Mapping[str, Any],
    issues: list[str],
    *,
    field: str,
    required: Sequence[str],
    label: str,
) -> None:
    """Require an exact, named contract set without accepting vague booleans."""

    value = _lookup(document, field)
    if not isinstance(value, (list, tuple, set)):
        issues.append(f"{label} must list every required entry")
        return
    observed = {str(item) for item in value}
    missing = [item for item in required if item not in observed]
    if missing:
        issues.append(f"{label} missing required entries: {', '.join(missing)}")


def _specific_issues(rule: ArtifactRule, document: Mapping[str, Any]) -> list[str]:
    """Enforce the hard Bible conditions associated with each evidence role."""

    issues: list[str] = []
    check = rule.check
    if check == "authority_freeze":
        _require_true(document, issues, "hidden memberships frozen", "hidden_memberships_frozen")
        _require_true(document, issues, "deletion policy frozen", "deletion_policy_frozen")
        _require_true(document, issues, "rollback policy frozen", "rollback_policy_frozen")
    elif check == "build_fabric":
        _require_true(document, issues, "isolated worktree contracts", "isolated_worktree_contracts")
        _require_true(document, issues, "report-only model authority", "report_only_model_authority")
        _require_members(
            document,
            issues,
            field="lane_classes",
            required=GROK_LANE_CLASSES,
            label="Grok lane classes",
        )
        _require_members(
            document,
            issues,
            field="lane_contract_fields",
            required=GROK_LANE_CONTRACT_FIELDS,
            label="Grok lane contract fields",
        )
        _require_members(
            document,
            issues,
            field="resource_classes",
            required=GROK_RESOURCE_CLASSES,
            label="Grok resource classes",
        )
        _require_members(
            document,
            issues,
            field="scheduler_rules",
            required=GROK_SCHEDULER_RULES,
            label="Grok scheduler rules",
        )
    elif check == "resource_governor":
        _require_true(document, issues, "resource telemetry", "resource_telemetry")
        _require_true(document, issues, "pressure governor", "pressure_governor")
        _require_true(document, issues, "safe storage ownership", "storage_ownership")
        _require_members(
            document,
            issues,
            field="telemetry_fields",
            required=RESOURCE_TELEMETRY_FIELDS,
            label="resource telemetry fields",
        )
        _require_members(
            document,
            issues,
            field="task_scheduler_fields",
            required=TASK_SCHEDULER_FIELDS,
            label="critical-path task fields",
        )
        _require_members(
            document,
            issues,
            field="scheduler_selection_objectives",
            required=SCHEDULER_SELECTION_OBJECTIVES,
            label="scheduler selection objectives",
        )
        _require_members(
            document,
            issues,
            field="pressure_modes",
            required=PRESSURE_MODES,
            label="pressure modes",
        )
        _require_members(
            document,
            issues,
            field="energy_evidence_outputs",
            required=ENERGY_EVIDENCE_OUTPUTS,
            label="energy evidence outputs",
        )
    elif check == "agent_os_foundation":
        for label, path in (
            ("scheduler live", "scheduler_live"),
            ("tool gateway live", "tool_gateway_live"),
            ("memory live", "memory_live"),
            ("recovery live", "recovery_live"),
        ):
            _require_true(document, issues, label, path)
        if rule.artifact_id == "HCLI_AGENT_OS_V3_STATUS":
            _require_members(
                document,
                issues,
                field="agent_roles",
                required=AGENT_ROLES,
                label="Agent OS roles",
            )
            _require_members(
                document,
                issues,
                field="scheduler_capabilities",
                required=AGENT_SCHEDULER_CAPABILITIES,
                label="Agent OS scheduler capabilities",
            )
            _require_members(
                document,
                issues,
                field="agent_os_performance_gates",
                required=AGENT_OS_PERFORMANCE_GATES,
                label="Agent OS performance gates",
            )
        elif rule.artifact_id == "HCLI_CONTEXT_COMPILER_RECEIPT":
            _require_members(
                document,
                issues,
                field="context_inputs",
                required=CONTEXT_COMPILER_INPUTS,
                label="context compiler inputs",
            )
            _require_members(
                document,
                issues,
                field="context_properties",
                required=CONTEXT_COMPILER_PROPERTIES,
                label="context compiler properties",
            )
        elif rule.artifact_id == "HCLI_KV_STATE_MANAGER_RECEIPT":
            _require_members(
                document,
                issues,
                field="kv_state_capabilities",
                required=KV_STATE_CAPABILITIES,
                label="KV/state capabilities",
            )
    elif check == "gravity_foundation":
        _require_true(document, issues, "direct-execution law", "direct_execution_law")
        _require_true(document, issues, "complete-BPW accounting", "complete_bpw_accounting")
        _require_true(document, issues, "negative-science retrieval", "negative_science_retrieval")
        if rule.artifact_id == "GRAVITY_V3_FRONTIER":
            _require_members(
                document,
                issues,
                field="gravity_components",
                required=GRAVITY_COMPONENTS,
                label="Gravity V3 components",
            )
            _require_members(
                document,
                issues,
                field="representation_tournament_classes",
                required=REPRESENTATION_TOURNAMENT_CLASSES,
                label="Gravity representation tournament classes",
            )
    elif check == "metal_foundation":
        _require_true(document, issues, "exact-model code generation", "exact_model_codegen")
        _require_true(document, issues, "family semantic binding", "family_semantic_binding")
        if rule.artifact_id == "METAL_FAMILY_PLUGIN_MATRIX":
            _require_members(
                document,
                issues,
                field="shared_primitives",
                required=SHARED_PRIMITIVES,
                label="shared compiler primitives",
            )
            _require_members(
                document,
                issues,
                field="family_plugins",
                required=FAMILY_PLUGINS,
                label="required family plugins",
            )
        if rule.artifact_id == "METAL_EXACT_MODEL_COMPILER_STATUS":
            _require_members(
                document,
                issues,
                field="architecture_fingerprint_fields",
                required=ARCHITECTURE_FINGERPRINT_FIELDS,
                label="architecture fingerprint fields",
            )
            _require_members(
                document,
                issues,
                field="model_program_key_fields",
                required=MODEL_PROGRAM_KEY_FIELDS,
                label="exact-model program key fields",
            )
    elif check == "manager_source":
        _require_true(document, issues, "source inventory frozen", "source_inventory_frozen")
        _require_true(document, issues, "tokenizer/template frozen", "tokenizer_template_frozen")
    elif check == "manager_anchor":
        _require_true(document, issues, "capability anchor frozen", "capability_anchor_frozen")
        _require_true(document, issues, "capability anchor passed", "capability_anchor_passed")
        task_hash = _string(document, "frozen_task_catalog_sha256", "task_catalog_sha256")
        if not _hash_like(task_hash):
            issues.append("frozen manager task catalogue hash is required")
    elif check == "manager_gravity":
        _require_number_at_most(document, issues, "complete_bpw", 1.5, "complete_bpw", "metrics.complete_bpw")
        artifact = _string(document, "gravity_manager_artifact_id", "artifact_binding.gravity_manager_artifact_id")
        expected_artifact = (
            QWEN30_GRAVITY_MANAGER_ARTIFACT
            if rule.model_id == MODEL_30B
            else QWEN80_GRAVITY_MANAGER_ARTIFACT
        )
        if artifact != expected_artifact:
            issues.append(f"Gravity receipt must bind tournament artifact {expected_artifact}")
        for label, path in (
            ("native direct execution", "native_direct_execution"),
            ("loadable artifact", "artifact_loadable"),
            ("no hidden dense shadow", "no_hidden_dense_shadow"),
            ("rollback", "rollback_available"),
        ):
            _require_true(document, issues, label, path)
    elif check == "manager_tg3":
        artifact = _string(document, "gravity_manager_artifact_id", "artifact_binding.gravity_manager_artifact_id")
        expected_artifact = (
            QWEN30_GRAVITY_MANAGER_ARTIFACT
            if rule.model_id == MODEL_30B
            else QWEN80_GRAVITY_MANAGER_ARTIFACT
        )
        if artifact != expected_artifact:
            issues.append(f"TG3 receipt must bind tournament artifact {expected_artifact}")
        _require_number_at_least(
            document,
            issues,
            "BASE_TRUE_TPS",
            333.0,
            "base_true_tps",
            "metrics.BASE_TRUE_TPS",
            "scoreboard.metrics.BASE_TRUE_TPS",
        )
        fallback = _number(document, "fallback_count", "metrics.fallback_count")
        if fallback is None or fallback != 0:
            issues.append("fallback_count must be exactly 0")
        for label, paths in (
            ("real Metal runtime", ("real_metal_runtime", "runtime.real_metal_runtime")),
            ("real GPU dispatch", ("real_gpu_dispatch", "runtime.real_gpu_dispatch")),
            ("stable p99", ("stable_p99", "metrics.stable_p99")),
            ("complete-token timing", ("complete_token_timing",)),
            ("batch-1 base runtime", ("batch_1_base_runtime",)),
            ("prompt-dependent coherent generation", ("prompt_dependent_coherent_generation",)),
            ("same exact model", ("same_exact_model",)),
            ("same capability tier", ("same_capability_tier",)),
            ("TG3 review approved", ("tg3_review_approved",)),
        ):
            _require_true(document, issues, label, *paths)
        dispatches = _number(document, "gpu_dispatches", "metrics.gpu_dispatches")
        if dispatches is None or dispatches <= 0:
            issues.append("gpu_dispatches must be positive")
    elif check == "manager_kernel_operational":
        artifact = _string(document, "gravity_manager_artifact_id", "artifact_binding.gravity_manager_artifact_id")
        expected_artifact = (
            QWEN30_GRAVITY_MANAGER_ARTIFACT
            if rule.model_id == MODEL_30B
            else QWEN80_GRAVITY_MANAGER_ARTIFACT
        )
        if artifact != expected_artifact:
            issues.append(f"operational kernel receipt must bind tournament artifact {expected_artifact}")
        _require_number_at_least(
            document,
            issues,
            "operational exact-model BASE_TRUE_TPS",
            MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR,
            "operational_base_true_tps",
            "metrics.operational_base_true_tps",
        )
        fallback = _number(document, "fallback_count", "metrics.fallback_count")
        if fallback is None or fallback != 0:
            issues.append("operational kernel fallback_count must be exactly 0")
        for label, paths in (
            ("exact-model custom kernel", ("exact_model_custom_kernel",)),
            ("native direct execution", ("native_direct_execution",)),
            ("real Metal runtime", ("real_metal_runtime", "runtime.real_metal_runtime")),
            ("real GPU dispatch", ("real_gpu_dispatch", "runtime.real_gpu_dispatch")),
            ("HCLI live path", ("hcli_live_path",)),
            ("kernel parity", ("kernel_parity_passed",)),
            ("stable prompt-dependent output", ("prompt_dependent_coherent_generation",)),
        ):
            _require_true(document, issues, label, *paths)
        dispatches = _number(document, "gpu_dispatches", "metrics.gpu_dispatches")
        if dispatches is None or dispatches <= 0:
            issues.append("operational kernel gpu_dispatches must be positive")
    elif check == "manager_agent":
        artifact = _string(document, "gravity_manager_artifact_id", "artifact_binding.gravity_manager_artifact_id")
        expected_artifact = (
            QWEN30_GRAVITY_MANAGER_ARTIFACT
            if rule.model_id == MODEL_30B
            else QWEN80_GRAVITY_MANAGER_ARTIFACT
        )
        if artifact != expected_artifact:
            issues.append(f"manager receipt must bind tournament artifact {expected_artifact}")
        _require_number_at_least(
            document,
            issues,
            "HCLI/raw decode ratio",
            0.95,
            "hcli_raw_decode_ratio",
            "metrics.hcli_raw_decode_ratio",
        )
        for label, path in (
            ("HCLI Agent OS integration", "hcli_agent_os_integrated"),
            ("manager capability contract", "manager_capability_contract_passed"),
            ("residency safety", "residency_safe"),
            ("restart and rollback", "restart_and_rollback_passed"),
            ("long unattended campaign", "long_unattended_campaign_passed"),
        ):
            _require_true(document, issues, label, path)
    elif check == "tournament":
        candidates = document.get("candidates")
        if not isinstance(candidates, list) or tuple(str(item) for item in candidates) != TOURNAMENT_CANDIDATE_ORDER:
            issues.append("tournament candidates must be the fixed Qwen30-Gravity then Qwen80-Gravity artifact order")
        task_hash = _string(document, "frozen_task_catalog_sha256")
        if not _hash_like(task_hash):
            issues.append("tournament requires a frozen hidden task catalogue hash")
        _require_true(document, issues, "hidden comparison tasks frozen", "hidden_comparison_tasks_frozen")
        comparisons = document.get("comparison_results")
        if not isinstance(comparisons, Mapping):
            issues.append("tournament comparison_results must be an object")
        else:
            for dimension in TOURNAMENT_DIMENSIONS:
                row = comparisons.get(dimension)
                if not isinstance(row, Mapping) or row.get("measured") is not True:
                    issues.append(f"tournament comparison is missing measured {dimension}")
        # The generic comparison matrix above is deliberately not enough to
        # choose Hawking's manager.  The fixed protocol requires both solo and
        # symmetric orchestration modes, conjunctive qualification gates,
        # protected blind/long-horizon work, fair resources, adversarial
        # review, Pareto analysis, and a protected report.  It returns only
        # blockers and never scores either candidate.
        issues.extend(validate_final_manager_tournament_result(document))
    elif check == "winner":
        winner = _string(document, "winner_model", "winner")
        if winner is None or not any(_same_model(winner, item) for item in TOURNAMENT_CANDIDATE_ORDER):
            issues.append("winner must be exactly one qualified Gravity manager artifact")
        if _string(document, "designation") != "ASCENSION_MANAGER":
            issues.append("winner designation must be ASCENSION_MANAGER")
        if not _hash_like(_string(document, "tournament_seal_sha256")):
            issues.append("winner must bind the tournament seal")
    elif check == "alternate_offload":
        for label, path in (
            ("alternate local body evicted", "alternate_local_body_evicted"),
            ("small fixtures retained", "small_fixtures_retained"),
            ("no permanent second local reviewer", "permanent_second_local_reviewer_required"),
        ):
            if path == "permanent_second_local_reviewer_required":
                if _lookup(document, path) is not False:
                    issues.append("a hidden permanent second local reviewer is forbidden")
            else:
                _require_true(document, issues, label, path)
        if not _hash_like(_string(document, "remote_hash", "remote.sha256")):
            issues.append("alternate remote hash is required")
        if _string(document, "restore_command") is None:
            issues.append("alternate restore command is required")
    elif check == "sandbox_activation":
        for label, path in (
            ("external manager gate", "external_manager_gate_passed"),
            ("HCLI Agent OS manager ready", "hcli_agent_os_manager_ready"),
            ("only winner active", "only_winner_active"),
        ):
            _require_true(document, issues, label, path)
        if _lookup(document, "second_local_manager_active") is not False:
            issues.append("second local manager must be inactive at sandbox activation")
    elif check == "family_launch":
        expected_family = rule.family or ""
        if _normal(_string(document, "family") or "") != _normal(expected_family):
            issues.append(f"family qualification must bind exact family {expected_family}")
        for label, path in (
            ("exact-model qualification", "exact_model_qualified"),
            ("capability", "capability_passed"),
            ("parity", "parity_passed"),
            ("recovery", "recovery_passed"),
            ("TG3", "tg3_passed"),
        ):
            _require_true(document, issues, label, path)
        _require_number_at_most(document, issues, "complete_bpw", 1.5, "complete_bpw", "metrics.complete_bpw")
        if _lookup(document, "generic_fallback_used") is not False:
            issues.append("generic fallback cannot substitute for a core family")
    elif check == "generic_reference":
        _require_true(document, issues, "generic reference intake", "generic_reference_intake_ready")
        _require_true(document, issues, "not a core-family substitute", "not_core_family_substitute")
    elif check == "matrix":
        _require_true(document, issues, "matrix complete", "matrix_complete")
        _require_true(document, issues, "direct evidence only", "direct_evidence_only")
    elif check == "product":
        _require_true(document, issues, "product test passed", "product_test_passed")
        _require_true(document, issues, "real product path", "real_product_path")
    elif check == "global_audit":
        _require_true(document, issues, "all advertised models qualified", "all_advertised_models_qualified")
        _require_true(document, issues, "no launch exception", "no_launch_exception")
    elif check == "external_review":
        _require_true(document, issues, "review packet complete", "review_packet_complete")
        _require_true(document, issues, "external review requested", "external_review_requested")
    elif check == "external_review_acceptance":
        _require_true(document, issues, "external review accepted", "external_review_accepted")
        _require_true(document, issues, "findings repaired or waived by human", "findings_repaired_or_human_waived")
    elif check == "apple_release":
        _require_true(document, issues, "all launch gates true", "all_launch_gates_true")
        _require_true(document, issues, "external review accepted", "external_review_accepted")
        _require_true(document, issues, "Apple production package", "apple_production_package_ready")
    elif check == "frontier":
        _require_true(document, issues, "post-release authorization", "post_release_authorized")
        _require_true(document, issues, "TG2/TG1 research active", "tg2_tg1_research_active")
    return issues


def _evaluate_json_rule(
    rule: ArtifactRule, path: Path, bible: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    document = _read_json(path)
    if document is None:
        return (
            {
                "artifact_id": rule.artifact_id,
                "path": str(path),
                "status": "MISSING_OR_UNREADABLE",
                "issues": ["sealed JSON evidence file is absent or unreadable"],
            },
            None,
        )
    try:
        verified = verify(document, label=str(path))
    except SealIntegrityError as exc:
        return (
            {
                "artifact_id": rule.artifact_id,
                "path": str(path),
                "status": "INVALID_SEAL",
                "issues": [str(exc)],
            },
            None,
        )
    issues = _common_issues(rule, verified, bible) + _specific_issues(rule, verified)
    return (
        {
            "artifact_id": rule.artifact_id,
            "path": str(path),
            "status": "PASS" if not issues else "REJECTED",
            "issues": issues,
            "seal_sha256": verified.get("seal_sha256"),
        },
        verified if not issues else None,
    )


def _evaluate_jsonl_rule(
    rule: ArtifactRule, path: Path, bible: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        lines = []
    if not lines:
        return (
            {
                "artifact_id": rule.artifact_id,
                "path": str(path),
                "status": "MISSING_OR_UNREADABLE",
                "issues": ["sealed JSONL evidence ledger is absent or empty"],
            },
            None,
        )
    row_issues: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    for number, raw in enumerate(lines, start=1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            row_issues.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, Mapping):
            row_issues.append(f"line {number}: must be an object")
            continue
        try:
            verified = verify(row, label=f"{path}:{number}")
        except SealIntegrityError as exc:
            row_issues.append(f"line {number}: {exc}")
            continue
        identity = _string(verified, "artifact_id", "ledger_id", "source_artifact")
        if identity != rule.artifact_id:
            row_issues.append(f"line {number}: artifact identity must be {rule.artifact_id}")
            continue
        row_common = _common_issues(rule, verified, bible)
        # JSONL rows identify the ledger through ``ledger_id`` rather than the
        # generic document identity accepted by _common_issues above.
        row_common = [
            issue
            for issue in row_common
            if issue != f"artifact identity must be exactly {rule.artifact_id}"
        ]
        if row_common:
            row_issues.extend(f"line {number}: {issue}" for issue in row_common)
            continue
        valid_rows.append(verified)
    return (
        {
            "artifact_id": rule.artifact_id,
            "path": str(path),
            "status": "PASS" if valid_rows and not row_issues else "REJECTED",
            "issues": row_issues,
            "rows": len(valid_rows),
            "ledger_sha256": _file_sha256(path) if path.exists() else None,
        },
        {"rows": valid_rows} if valid_rows and not row_issues else None,
    )


def _evaluate_file_rule(
    rule: ArtifactRule, path: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Check a required executable/database object without pretending it is a receipt.

    Bible §2.5 deliberately names a restore *script*, and §9.7 names a SQLite
    mechanism index.  They cannot carry our JSON receipt seal themselves, so a
    related sealed JSON artifact binds their digest in ``_cross_stage_issues``.
    This check establishes only object integrity/shape and never certifies a
    V3 state by itself.
    """

    try:
        node = os.lstat(path)
    except OSError:
        node = None
    if node is None or stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        return (
            {
                "artifact_id": rule.artifact_id,
                "path": str(path),
                "status": "MISSING_OR_UNREADABLE",
                "issues": ["required file artifact is absent, symlinked, or not regular"],
            },
            None,
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return (
            {
                "artifact_id": rule.artifact_id,
                "path": str(path),
                "status": "MISSING_OR_UNREADABLE",
                "issues": [f"cannot read required file artifact: {exc}"],
            },
            None,
        )
    issues: list[str] = []
    if rule.format == "script":
        if not raw.startswith(b"#!"):
            issues.append("restore script must have an interpreter shebang")
        if not (stat.S_IMODE(node.st_mode) & 0o111):
            issues.append("restore script must be executable")
    elif rule.format == "sqlite":
        if not raw.startswith(b"SQLite format 3\x00"):
            issues.append("mechanism index must be a SQLite 3 database")
    return (
        {
            "artifact_id": rule.artifact_id,
            "path": str(path),
            "status": "PASS" if not issues else "REJECTED",
            "issues": issues,
            "sha256": _digest(raw),
            "bytes": len(raw),
        },
        {"artifact_id": rule.artifact_id, "sha256": _digest(raw), "path": str(path)}
        if not issues
        else None,
    )


def evaluate_artifact(
    rule: ArtifactRule,
    *,
    evidence_root: str | Path,
    bible: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = Path(evidence_root).expanduser().resolve() / rule.filename
    if rule.format == "jsonl":
        return _evaluate_jsonl_rule(rule, path, bible)
    if rule.format in {"script", "sqlite"}:
        return _evaluate_file_rule(rule, path)
    return _evaluate_json_rule(rule, path, bible)


def _cross_stage_issues(stage_id: str, documents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Check bindings that require more than one sealed receipt."""

    issues: list[str] = []
    if stage_id == "V3_SEED_ARCHIVE":
        archive = documents.get("ASCENSION_V3_SEED_ARCHIVE")
        restore = documents.get("ASCENSION_V3_RESTORE")
        if archive and restore and archive.get("restore_script_sha256") != restore.get("sha256"):
            issues.append("Seed Archive receipt must bind the exact restore script hash")
    elif stage_id == "V3_KNOWLEDGE_PLANE":
        transfer = documents.get("ASCENSION_TRANSFER_MATRIX")
        mechanism_index = documents.get("ASCENSION_MECHANISM_INDEX")
        if transfer and mechanism_index and transfer.get("mechanism_index_sha256") != mechanism_index.get("sha256"):
            issues.append("transfer matrix must bind the exact mechanism-index SQLite hash")
    elif stage_id == "MANAGER_TOURNAMENT":
        tournament = documents.get("ASCENSION_MANAGER_TOURNAMENT")
        winner = documents.get("ASCENSION_MANAGER_WINNER")
        alternate = documents.get("ASCENSION_ALTERNATE_OFFLOAD")
        if tournament and winner:
            if winner.get("tournament_seal_sha256") != tournament.get("seal_sha256"):
                issues.append("winner receipt does not bind the exact tournament seal")
        if tournament and winner and alternate:
            winner_model = _string(winner, "winner_model", "winner")
            alternate_model = _string(alternate, "alternate_model", "alternate")
            expected_alternate = next(
                (candidate for candidate in TOURNAMENT_CANDIDATE_ORDER if not _same_model(candidate, winner_model)),
                None,
            )
            if _string(alternate, "winner_model") is None or not _same_model(
                _string(alternate, "winner_model"), winner_model
            ):
                issues.append("alternate offload receipt must bind the sealed winner")
            if expected_alternate is None or not _same_model(alternate_model, expected_alternate):
                issues.append("alternate offload receipt must name the losing qualified manager")
    elif stage_id == "SANDBOX_ACTIVATION":
        sandbox = documents.get("ASCENSION_SANDBOX_ACTIVE")
        winner = documents.get("ASCENSION_MANAGER_WINNER")
        alternate = documents.get("ASCENSION_ALTERNATE_OFFLOAD")
        if sandbox and winner:
            if not _same_model(_string(sandbox, "manager_model", "manager"), _string(winner, "winner_model", "winner")):
                issues.append("sandbox activation must bind the sealed manager winner")
        if sandbox and alternate and alternate.get("alternate_local_body_evicted") is not True:
            issues.append("sandbox activation cannot follow an unevicted alternate manager")
    return issues


def _stage_state(
    spec: StageSpec,
    *,
    prior: Mapping[str, Mapping[str, Any]],
    bible: Mapping[str, Any],
    evidence_root: Path,
    documents: dict[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing_prerequisites = [
        state for state in spec.prerequisites if prior.get(state, {}).get("status") != "CERTIFIED"
    ]
    if missing_prerequisites:
        return {
            "id": spec.state_id,
            "status": "PENDING_PREREQUISITES",
            "completion_states": list(spec.completion_states),
            "description": spec.description,
            "resource_class": spec.resource_class,
            "prerequisites": list(spec.prerequisites),
            "blockers": [f"requires certified state {value}" for value in missing_prerequisites],
            "evidence": [],
        }

    evidence: list[dict[str, Any]] = []
    local_documents: dict[str, Mapping[str, Any]] = {}
    for rule in spec.artifacts:
        report, document = evaluate_artifact(rule, evidence_root=evidence_root, bible=bible)
        evidence.append(report)
        if document is not None:
            local_documents[rule.artifact_id] = document
            documents[rule.artifact_id] = document

    blockers = [
        issue
        for report in evidence
        for issue in report.get("issues", [])
        if isinstance(issue, str) and issue
    ]
    if spec.state_id == "V3_ADOPT" and bible.get("state_machine_matches") is not True:
        blockers.extend(str(item) for item in bible.get("issues", []))
    # Tournament/sandbox bindings can refer to documents from an earlier state,
    # so use the accumulated document map rather than only this stage's files.
    blockers.extend(_cross_stage_issues(spec.state_id, documents | local_documents))
    return {
        "id": spec.state_id,
        "status": "CERTIFIED" if not blockers else "BLOCKED",
        "completion_states": list(spec.completion_states),
        "description": spec.description,
        "resource_class": spec.resource_class,
        "prerequisites": list(spec.prerequisites),
        "blockers": list(dict.fromkeys(blockers)),
        "evidence": evidence,
    }


def _read_tournament_controller(paths: LifecyclePaths) -> dict[str, Any]:
    document = _read_json(paths.tournament_controller_path)
    if document is None:
        return {
            "status": "NOT_ARMED",
            "path": str(paths.tournament_controller_path),
            "reason": "tournament controller has not been armed",
        }
    try:
        verified = verify(document, label=str(paths.tournament_controller_path))
    except SealIntegrityError as exc:
        return {
            "status": "INVALID_ARMING_RECORD",
            "path": str(paths.tournament_controller_path),
            "reason": str(exc),
        }
    if verified.get("schema") != TOURNAMENT_CONTROLLER_SCHEMA or verified.get("armed") is not True:
        return {
            "status": "NOT_ARMED",
            "path": str(paths.tournament_controller_path),
            "reason": "no valid armed tournament controller record",
        }
    if tuple(verified.get("candidate_order") or ()) != TOURNAMENT_CANDIDATE_ORDER:
        return {
            "status": "INVALID_ARMING_RECORD",
            "path": str(paths.tournament_controller_path),
            "reason": "armed controller candidate order drifted from Bible §3",
        }
    if verified.get("final_manager_protocol_schema") != FINAL_MANAGER_TOURNAMENT_PROTOCOL_SCHEMA:
        return {
            "status": "INVALID_ARMING_RECORD",
            "path": str(paths.tournament_controller_path),
            "reason": "armed controller does not bind the final-manager tournament protocol",
        }
    protocol_identity = verified.get("final_manager_protocol_identity_sha256")
    if not isinstance(protocol_identity, str) or not _hash_like(protocol_identity):
        return {
            "status": "INVALID_ARMING_RECORD",
            "path": str(paths.tournament_controller_path),
            "reason": "armed controller lacks a stable final-manager protocol identity",
        }
    return {
        "status": "ARMED",
        "path": str(paths.tournament_controller_path),
        "armed_at": verified.get("armed_at"),
        "seal_sha256": verified.get("seal_sha256"),
        "candidate_order": list(TOURNAMENT_CANDIDATE_ORDER),
        "dimensions": list(TOURNAMENT_DIMENSIONS),
        "final_manager_protocol_schema": FINAL_MANAGER_TOURNAMENT_PROTOCOL_SCHEMA,
        "final_manager_protocol_identity_sha256": protocol_identity,
    }


def _tournament_runtime_state(
    controller: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if controller.get("status") != "ARMED":
        return dict(controller)
    # Once the receipt-bound Manager Tournament state is certified, an armed
    # controller is evidence of persistent configuration only.  It must not
    # continue to advertise itself as awaiting execution or invite a rerun that
    # could create a conflicting winner/offload lineage.
    if states.get("MANAGER_TOURNAMENT", {}).get("status") == "CERTIFIED":
        return {
            **dict(controller),
            "status": "ARMED_COMPLETE_SEALED",
            "blockers": [],
            "claim_boundary": {
                "sealed_receipts_remain_the_tournament_authority": True,
                "does_not_rerun_or_rescore_a_completed_tournament": True,
                "does_not_select_a_new_winner": True,
            },
        }
    missing = [
        state
        for state in ("MANAGER_30B_AGENT", "MANAGER_80B_AGENT")
        if states.get(state, {}).get("status") != "CERTIFIED"
    ]
    if missing:
        return {
            **dict(controller),
            "status": "ARMED_BLOCKED_UNQUALIFIED_CANDIDATES",
            "blockers": [f"requires certified {state}" for state in missing],
            "claim_boundary": {
                "does_not_run_candidate_models": True,
                "does_not_select_a_winner": True,
                "does_not_activate_sandbox": True,
            },
        }
    return {
        **dict(controller),
        "status": "ARMED_READY_FOR_PROTECTED_EXECUTION",
        "blockers": [
            "awaiting sealed frozen-task tournament receipt from protected execution"
        ],
        "claim_boundary": {
            "does_not_auto_score_subjective_dimensions": True,
            "does_not_auto_select_a_winner": True,
            "models_cannot_certify_their_own_tournament": True,
        },
    }


def _next_active_state(states: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for state in states:
        if state.get("status") == "BLOCKED":
            return state
    for state in states:
        if state.get("status") == "PENDING_PREREQUISITES":
            return state
    return None


def _command_for(paths: LifecyclePaths, *, stage_id: str | None) -> str:
    executable = sys.executable
    cli = REPO_ROOT / "tools" / "condense" / "ascension_lifecycle.py"
    arguments = [
        executable,
        str(cli),
        "work",
        "--root",
        str(paths.root),
    ]
    if stage_id:
        arguments.extend(("--stage", stage_id))
    return shlex.join(arguments)


def _build_work_queue(
    paths: LifecyclePaths,
    *,
    states: Sequence[Mapping[str, Any]],
    tournament: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for state in states:
        if state.get("status") == "CERTIFIED":
            continue
        ready = state.get("status") == "BLOCKED"
        rows.append(
            {
                "action_id": f"intake-{str(state['id']).lower()}",
                "state": state["id"],
                "status": "READY_FOR_EVIDENCE" if ready else "WAITING_DEPENDENCY",
                "resource_class": state["resource_class"],
                "goal": state["description"],
                "requires": list(state.get("prerequisites") or ()),
                "completion_states": list(state.get("completion_states") or ()),
                "required_artifacts": [
                    item.get("artifact_id") for item in state.get("evidence", []) if item.get("artifact_id")
                ]
                or [rule.artifact_id for rule in STAGE_BY_ID[str(state["id"])].artifacts],
                "blockers": list(state.get("blockers") or ()),
                "next_command": _command_for(paths, stage_id=str(state["id"])),
                "authority": {
                    "model_may_emit": "sealed candidate evidence only",
                    "certification_required_from": sorted(CONTROLLER_AUTHORITIES),
                    "model_may_not_self_promote": True,
                },
                "auto_dispatch": "monitor_and_validate_only",
            }
        )
    if tournament.get("status", "").startswith("ARMED"):
        rows.append(
            {
                "action_id": "protected-manager-tournament",
                "state": "MANAGER_TOURNAMENT",
                "status": tournament["status"],
                "resource_class": "GPU_HEAVY",
                "goal": "run the fixed protected manager tournament only after both candidates qualify",
                "requires": ["MANAGER_30B_AGENT", "MANAGER_80B_AGENT"],
                "comparison_dimensions": list(TOURNAMENT_DIMENSIONS),
                "next_command": _command_for(paths, stage_id="MANAGER_TOURNAMENT"),
                "authority": {
                    "model_may_emit": "candidate reports and bounded task outputs",
                    "certification_required_from": sorted(CONTROLLER_AUTHORITIES),
                    "auto_select_winner": False,
                },
            }
        )
    return seal(
        {
            "schema": WORK_QUEUE_SCHEMA,
            "recorded_at": _utc_now(),
            "root": str(paths.root),
            "items": rows,
            "claim_boundary": {
                "no_timeline_completion": True,
                "no_model_body_action_by_controller": True,
                "no_deletion_or_eviction_by_controller": True,
                "no_manager_promotion_by_model": True,
            },
        }
    )


def _build_review_index(
    states: Mapping[str, Mapping[str, Any]], *, tournament: Mapping[str, Any]
) -> dict[str, Any]:
    def status_for(state: str) -> str:
        return str(states.get(state, {}).get("status") or "PENDING")

    reviews = [
        {
            "review": "QWEN30_TG3",
            "state": "MANAGER_30B_TG",
            "status": "SATISFIED" if status_for("MANAGER_30B_TG") == "CERTIFIED" else "PENDING",
            "required_authority": "protected_controller_or_human",
            "reason": "TG3 is a stop-for-review threshold, never silent promotion.",
        },
        {
            "review": "QWEN80_TG3",
            "state": "MANAGER_80B_TG",
            "status": "SATISFIED" if status_for("MANAGER_80B_TG") == "CERTIFIED" else "PENDING",
            "required_authority": "protected_controller_or_human",
            "reason": "TG3 is a stop-for-review threshold, never silent promotion.",
        },
        {
            "review": "MANAGER_TOURNAMENT",
            "state": "MANAGER_TOURNAMENT",
            "status": tournament.get("status"),
            "required_authority": "protected_controller_or_human",
            "reason": "both independently qualified managers must be compared on frozen tasks.",
        },
        {
            "review": "ALTERNATE_OFFLOAD",
            "state": "MANAGER_TOURNAMENT",
            "status": "SATISFIED" if status_for("MANAGER_TOURNAMENT") == "CERTIFIED" else "PENDING",
            "required_authority": "protected_controller_or_human",
            "reason": "remote hash, restore command, eviction, retained fixtures, and no hidden reviewer are mandatory.",
        },
        {
            "review": "GLOBAL_EXTERNAL_REVIEW",
            "state": "EXTERNAL_REVIEW",
            "status": status_for("EXTERNAL_REVIEW"),
            "required_authority": "external_reviewer_and_human",
            "reason": "a V3 review packet is not itself launch approval.",
        },
        {
            "review": "APPLE_RELEASE",
            "state": "APPLE_RELEASE",
            "status": status_for("APPLE_RELEASE"),
            "required_authority": "protected_controller_or_human",
            "reason": "release follows accepted external review and complete matrix evidence.",
        },
    ]
    return seal(
        {
            "schema": "hawking.ascension.v3_review_index.v1",
            "recorded_at": _utc_now(),
            "reviews": reviews,
            "claim_boundary": {"review_index_is_not_certification": True},
        }
    )


def evaluate_launch_gate(states: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Produce a no-exception V3 launch predicate from state, never prose."""

    required = (
        "MANAGER_TOURNAMENT",
        "SANDBOX_ACTIVATION",
        *(state for state, _completion, _family in FAMILY_RULES),
        "GLOBAL_LAUNCH_AUDIT",
        "EXTERNAL_REVIEW",
        "APPLE_RELEASE",
    )
    missing = [state for state in required if states.get(state, {}).get("status") != "CERTIFIED"]
    return {
        "status": "READY" if not missing else "BLOCKED",
        "required_states": list(required),
        "missing_states": missing,
        "launch_exception_permitted": False,
        "derived_only_from_controller_state": True,
    }


def _constitution_document(paths: LifecyclePaths, bible: Mapping[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": CONSTITUTION_SCHEMA,
            "status": "CONTROLLER_CONFIGURATION_ONLY",
            "recorded_at": _utc_now(),
            "bible": {
                "path": bible.get("path"),
                "sha256": bible.get("sha256"),
                "state_machine_matches": bible.get("state_machine_matches"),
            },
            "canonical_states": list(CANONICAL_STATES),
            "exact_continuation_outputs": list(EXACT_CONTINUATION_OUTPUTS),
            "evidence_intake_root": str(paths.evidence_root),
            "manager_tournament": {
                "candidate_order": list(TOURNAMENT_CANDIDATE_ORDER),
                "comparison_dimensions": list(TOURNAMENT_DIMENSIONS),
                "requires_both_qualified": True,
                "winner_must_be_protected_controller_or_human_certified": True,
                "alternate_must_be_offloaded_before_sandbox": True,
            },
            "no_drift_policy": {
                "sandbox_before_tournament": "REFUSE",
                "manager_above_1_5_bpw": "REFUSE",
                "manager_below_tg3": "REFUSE",
                "family_without_exact_model": "REFUSE",
                "model_self_promotion": "REFUSE",
                "unverified_deletion": "REFUSE",
                "timeline_completion": "REFUSE",
                "generic_fallback_core_family": "REFUSE",
                "hidden_second_reviewer": "REFUSE",
            },
            "claim_boundary": {
                "not_a_human_adoption_receipt": True,
                "not_a_model_qualification_receipt": True,
                "not_a_tournament_result": True,
                "not_a_sandbox_activation_receipt": True,
            },
        }
    )


def _launch_gate_wrapper() -> str:
    root_literal = repr(str(REPO_ROOT))
    return f'''#!/usr/bin/env python3
"""Generated V3 launch-gate entry point; the implementation remains repo-owned."""
from __future__ import annotations
import sys

ROOT = {root_literal}
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lab.operators.ascension_lifecycle import main

if __name__ == "__main__":
    raise SystemExit(main())
'''


def bootstrap_layout(
    root: str | Path = DEFAULT_ROOT, *, bible_path: str | Path = DEFAULT_BIBLE
) -> LifecyclePaths:
    """Create controller-owned files only; no certification evidence is fabricated."""

    paths = LifecyclePaths.from_root(root)
    _ensure_real_directory(paths.root, 0o750)
    _ensure_real_directory(paths.evidence_root, 0o750)
    bible = audit_bible(bible_path)
    _atomic_json(paths.constitution_path, _constitution_document(paths, bible))
    _atomic_text(paths.launch_gate_path, _launch_gate_wrapper(), mode=0o750)
    return paths


def evaluate_lifecycle(
    root: str | Path = DEFAULT_ROOT,
    *,
    bible_path: str | Path = DEFAULT_BIBLE,
) -> dict[str, Any]:
    """Evaluate every canonical V3 state against sealed intake evidence."""

    paths = bootstrap_layout(root, bible_path=bible_path)
    bible = audit_bible(bible_path)
    state_map: dict[str, dict[str, Any]] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    ordered_states: list[dict[str, Any]] = []
    for spec in STAGE_SPECS:
        state = _stage_state(
            spec,
            prior=state_map,
            bible=bible,
            evidence_root=paths.evidence_root,
            documents=documents,
        )
        state_map[spec.state_id] = state
        ordered_states.append(state)

    controller = _read_tournament_controller(paths)
    tournament = _tournament_runtime_state(controller, state_map)
    next_state = _next_active_state(ordered_states)
    launch_gate = evaluate_launch_gate(state_map)
    decision_body = {
        "bible_sha256": bible.get("sha256"),
        "states": [
            {"id": state["id"], "status": state["status"], "blockers": state["blockers"]}
            for state in ordered_states
        ],
        "tournament": {
            "status": tournament.get("status"),
            "blockers": tournament.get("blockers", []),
        },
        "launch_gate": launch_gate,
    }
    decision_sha256 = _digest(decision_body)
    state_document = seal(
        {
            "schema": STATE_SCHEMA,
            "recorded_at": _utc_now(),
            "decision_sha256": decision_sha256,
            "bible": bible,
            "canonical_states": list(CANONICAL_STATES),
            "states": ordered_states,
            "first_unmet_state": next_state["id"] if next_state else None,
            "next_command": _command_for(paths, stage_id=str(next_state["id"]) if next_state else None),
            "tournament": tournament,
            "launch_gate": launch_gate,
            "claim_boundary": {
                "controller_does_not_certify_missing_evidence": True,
                "controller_does_not_run_model_bodies": True,
                "controller_does_not_delete_or_evict": True,
                "controller_does_not_auto_select_manager": True,
                "controller_does_not_activate_production_sandbox": True,
            },
        }
    )
    work_queue = _build_work_queue(paths, states=ordered_states, tournament=tournament)
    active_lanes = seal(
        {
            "schema": "hawking.ascension.v3_active_lanes.v1",
            "recorded_at": _utc_now(),
            "decision_sha256": decision_sha256,
            "active_lanes": [
                {
                    "state": item["state"],
                    "status": item["status"],
                    "resource_class": item["resource_class"],
                    "blockers": item["blockers"],
                    "required_artifacts": item["required_artifacts"],
                }
                for item in work_queue["items"]
                if item.get("action_id") != "protected-manager-tournament"
            ],
            "tournament": tournament,
        }
    )
    gate_matrix = seal(
        {
            "schema": "hawking.ascension.v3_gate_matrix.v1",
            "recorded_at": _utc_now(),
            "decision_sha256": decision_sha256,
            "states": ordered_states,
            "launch_gate": launch_gate,
            "no_drift_enforced": _constitution_document(paths, bible)["no_drift_policy"],
        }
    )
    review_index = _build_review_index(state_map, tournament=tournament)
    next_command = state_document["next_command"]
    _atomic_json(paths.state_path, state_document)
    _atomic_json(paths.active_lanes_path, active_lanes)
    _atomic_json(paths.gate_matrix_path, gate_matrix)
    _atomic_json(paths.review_index_path, review_index)
    _atomic_json(paths.work_queue_path, work_queue)
    _atomic_text(
        paths.next_command_path,
        "#!/bin/sh\nset -eu\nexec " + next_command + "\n",
        mode=0o750,
    )
    # Import lazily: the report imports the immutable lifecycle constants for
    # traceability, while the state evaluator remains the sole transition
    # authority.  It runs after the continuation outputs are durable so the
    # report's artifact coverage describes this invocation, not the prior one.
    from lab.operators.ascension_fidelity import build_fidelity_report

    fidelity = build_fidelity_report(
        paths,
        bible=bible,
        states=ordered_states,
        tournament=tournament,
        launch_gate=launch_gate,
    )
    return {
        "schema": SCHEMA,
        "root": str(paths.root),
        "evidence_root": str(paths.evidence_root),
        "state_path": str(paths.state_path),
        "next_command_path": str(paths.next_command_path),
        "work_queue_path": str(paths.work_queue_path),
        "fidelity_path": str(paths.fidelity_path),
        "decision_sha256": decision_sha256,
        "first_unmet_state": state_document["first_unmet_state"],
        "tournament": tournament,
        "launch_gate": launch_gate,
        "certified_states": [
            state["id"] for state in ordered_states if state.get("status") == "CERTIFIED"
        ],
        "state_counts": {
            "certified": sum(1 for state in ordered_states if state.get("status") == "CERTIFIED"),
            "blocked": sum(1 for state in ordered_states if state.get("status") == "BLOCKED"),
            "pending_prerequisites": sum(
                1 for state in ordered_states if state.get("status") == "PENDING_PREREQUISITES"
            ),
        },
        "fidelity": {
            "overall_status": fidelity["fidelity"]["overall_status"],
            "live_receipt_completion": fidelity["fidelity"]["live_receipt_completion"],
            "bible_execution_sequence": fidelity["fidelity"]["bible_execution_sequence"],
        },
    }


def arm_tournament(
    root: str | Path = DEFAULT_ROOT,
    *,
    bible_path: str | Path = DEFAULT_BIBLE,
) -> dict[str, Any]:
    """Persistently arm the protected tournament controller without running models.

    This is the safe meaning of launching a tournament before qualification: the
    judge, exact candidate order, frozen dimensions, and no-autopromotion rule
    are live and restart-safe.  The controller reports an honest block until
    both manager candidates have independently earned their hard gates.
    """

    paths = bootstrap_layout(root, bible_path=bible_path)
    final_manager_protocol = build_final_manager_tournament_protocol()
    document = seal(
        {
            "schema": TOURNAMENT_CONTROLLER_SCHEMA,
            "armed": True,
            "armed_at": _utc_now(),
            "candidate_order": list(TOURNAMENT_CANDIDATE_ORDER),
            "comparison_dimensions": list(TOURNAMENT_DIMENSIONS),
            "final_manager_protocol_schema": FINAL_MANAGER_TOURNAMENT_PROTOCOL_SCHEMA,
            "final_manager_protocol_identity_sha256": final_manager_protocol["protocol_identity_sha256"],
            "execution_contract": {
                "requires_both_independently_qualified": True,
                "hidden_task_catalog_must_be_frozen": True,
                "protected_controller_or_human_certifies_result": True,
                "models_cannot_self_promote": True,
                "winner_not_auto_selected": True,
                "alternate_offload_required_before_sandbox": True,
            },
            "claim_boundary": {
                "armed_is_not_tournament_complete": True,
                "armed_is_not_manager_winner": True,
                "armed_is_not_sandbox_activation": True,
                "does_not_launch_model_body": True,
            },
        }
    )
    _atomic_json(paths.tournament_controller_path, document)
    return evaluate_lifecycle(paths.root, bible_path=bible_path)


def attest_human_adoption(
    root: str | Path = DEFAULT_ROOT,
    *,
    bible_path: str | Path = DEFAULT_BIBLE,
    instruction_sha256: str,
) -> dict[str, Any]:
    """Record an explicit human directive adopting the current Bible.

    This command is deliberately narrow.  It may only write the constitutional
    adoption receipt, binds that receipt to both the exact Bible bytes and a
    digest of the user's instruction, and then re-evaluates the state graph.
    It cannot certify a technical or model-measurement gate.
    """

    if not _hash_like(instruction_sha256):
        raise AscensionLifecycleError("instruction_sha256 must be a 64-character sha256")
    paths = bootstrap_layout(root, bible_path=bible_path)
    bible = audit_bible(bible_path)
    if bible.get("state_machine_matches") is not True or not isinstance(bible.get("sha256"), str):
        raise AscensionLifecycleError("cannot attest adoption against an unreadable or drifted Bible")
    document = seal(
        {
            "schema": "hawking.ascension.human_adoption_attestation.v1",
            "artifact_id": "ASCENSION_V3_ADOPTED",
            "status": "CONTROLLER_CERTIFIED",
            "certified_by": "human_operator",
            "authority_level": "human",
            "recorded_at": _utc_now(),
            "evidence_basis": "explicit_human_instruction",
            "instruction_sha256": instruction_sha256.lower(),
            "bible_sha256": bible["sha256"],
            "adopted_as_sole_canonical_programme": True,
            "claim_boundary": {
                "attests_only_human_constitutional_adoption": True,
                "does_not_certify_any_model_measurement": True,
                "does_not_waive_any_v3_gate": True,
                "does_not_select_a_manager_or_activate_sandbox": True,
            },
        }
    )
    _atomic_json(paths.evidence_root / "ASCENSION_V3_ADOPTED.json", document)
    return evaluate_lifecycle(paths.root, bible_path=bible_path)


@contextlib.contextmanager
def _exclusive_watch_lock(paths: LifecyclePaths) -> Iterator[None]:
    paths.lock_path.touch(exist_ok=True)
    with paths.lock_path.open("r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AscensionLifecycleError(
                f"another lifecycle controller owns {paths.lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def watch(
    root: str | Path = DEFAULT_ROOT,
    *,
    bible_path: str | Path = DEFAULT_BIBLE,
    interval_seconds: float = 30.0,
) -> int:
    """Standalone monitor; the detached sandbox controller normally calls tick itself."""

    if not 5.0 <= float(interval_seconds) <= 3600.0:
        raise AscensionLifecycleError("interval_seconds must be between 5 and 3600")
    paths = bootstrap_layout(root, bible_path=bible_path)
    stopping = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    import signal

    old_term = signal.signal(signal.SIGTERM, request_stop)
    old_int = signal.signal(signal.SIGINT, request_stop)
    try:
        with _exclusive_watch_lock(paths):
            while not stopping:
                result = evaluate_lifecycle(paths.root, bible_path=bible_path)
                print(
                    json.dumps(
                        {
                            "first_unmet_state": result["first_unmet_state"],
                            "tournament": result["tournament"].get("status"),
                            "launch_gate": result["launch_gate"].get("status"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                deadline = time.monotonic() + float(interval_seconds)
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def _work_order(root: str | Path, *, bible_path: str | Path, stage: str | None) -> dict[str, Any]:
    result = evaluate_lifecycle(root, bible_path=bible_path)
    paths = LifecyclePaths.from_root(root)
    queue = _read_json(paths.work_queue_path) or {}
    items = queue.get("items") if isinstance(queue.get("items"), list) else []
    target = stage or result.get("first_unmet_state")
    selected = next((item for item in items if item.get("state") == target), None)
    return {
        "schema": "hawking.ascension.v3_work_order.v1",
        "state": target,
        "work_order": selected,
        "claim_boundary": {
            "this_command_does_not_certify_or_modify_evidence": True,
            "this_command_does_not_load_models_or_delete_data": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        item.add_argument("--bible", type=Path, default=DEFAULT_BIBLE)

    init = sub.add_parser("init", help="create controller-owned V3 continuation files")
    common(init)
    tick = sub.add_parser("tick", help="evaluate all receipt-bound V3 states once")
    common(tick)
    status = sub.add_parser("status", help="print the last V3 state without evaluating")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arm = sub.add_parser("arm-tournament", help="arm the persistent protected tournament controller")
    common(arm)
    adopt = sub.add_parser("attest-adoption", help="record an explicit human adoption attestation")
    common(adopt)
    adopt.add_argument("--instruction-sha256", required=True)
    work = sub.add_parser("work", help="print a deterministic work order for the current state")
    common(work)
    work.add_argument("--stage", choices=CANONICAL_STATES)
    daemon = sub.add_parser("watch", help="standalone lifecycle monitor")
    common(daemon)
    daemon.add_argument("--interval-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        paths = bootstrap_layout(args.root, bible_path=args.bible)
        result = evaluate_lifecycle(paths.root, bible_path=args.bible)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "tick":
        print(json.dumps(evaluate_lifecycle(args.root, bible_path=args.bible), indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        paths = LifecyclePaths.from_root(args.root)
        status = _read_json(paths.state_path)
        if status is None:
            print(json.dumps({"state": "ABSENT", "state_path": str(paths.state_path)}))
            return 2
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    if args.command == "arm-tournament":
        print(json.dumps(arm_tournament(args.root, bible_path=args.bible), indent=2, sort_keys=True))
        return 0
    if args.command == "attest-adoption":
        print(
            json.dumps(
                attest_human_adoption(
                    args.root,
                    bible_path=args.bible,
                    instruction_sha256=args.instruction_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "work":
        print(json.dumps(_work_order(args.root, bible_path=args.bible, stage=args.stage), indent=2, sort_keys=True))
        return 0
    if args.command == "watch":
        return watch(args.root, bible_path=args.bible, interval_seconds=args.interval_seconds)
    raise AssertionError(f"unknown lifecycle command {args.command!r}")


__all__ = [
    "AGENT_OS_ARTIFACTS",
    "ArtifactRule",
    "AscensionLifecycleError",
    "CANONICAL_STATES",
    "DEFAULT_BIBLE",
    "DEFAULT_ROOT",
    "EXACT_CONTINUATION_OUTPUTS",
    "FAMILY_RULES",
    "LifecyclePaths",
    "MANAGER_30_ARTIFACTS",
    "MANAGER_80_ARTIFACTS",
    "MANAGER_CANDIDATE_ORDER",
    "TOURNAMENT_CANDIDATE_ORDER",
    "QWEN30_GRAVITY_MANAGER_ARTIFACT",
    "QWEN80_GRAVITY_MANAGER_ARTIFACT",
    "MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR",
    "MODEL_30B",
    "MODEL_80B",
    "STAGE_SPECS",
    "TOURNAMENT_DIMENSIONS",
    "arm_tournament",
    "attest_human_adoption",
    "audit_bible",
    "bootstrap_layout",
    "evaluate_artifact",
    "evaluate_launch_gate",
    "evaluate_lifecycle",
    "watch",
]


if __name__ == "__main__":
    raise SystemExit(main())
