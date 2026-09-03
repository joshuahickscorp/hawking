"""Structured Ascension Knowledge Plane contract (Bible §9)."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import seal


SCHEMA = "hawking.ascension.knowledge_plane_contract.v1"
FILENAME = "ASCENSION_V3_KNOWLEDGE_PLANE_CONTRACT.json"

KERNEL_GENOME_FIELDS: tuple[str, ...] = (
    "operator",
    "model_family",
    "tensor_geometry",
    "representation",
    "kernel_grammar",
    "tile",
    "threadgroup",
    "memory_layout",
    "command_graph",
    "measured_latency",
    "bandwidth",
    "occupancy",
    "energy",
    "parity",
    "capability",
    "hardware",
    "source_revision",
    "validity_scope",
)

REPRESENTATION_GENOME_FIELDS: tuple[str, ...] = (
    "tensor_or_organ",
    "model_family",
    "geometry",
    "representation",
    "precision",
    "codebook_or_basis",
    "residual",
    "doctor_qat",
    "complete_bpw_delta",
    "capability_delta",
    "runtime_delta",
    "kernel_requirements",
    "failure_modes",
    "reopen_conditions",
)

SCHEDULER_GENOME_FIELDS: tuple[str, ...] = (
    "task_class",
    "resource_class",
    "dependency",
    "critical_path_status",
    "preemptibility",
    "checkpoint_cost",
    "measured_progress_per_watt",
    "contention_outcome",
    "validity_scope",
)

NEGATIVE_SCIENCE_FIELDS: tuple[str, ...] = (
    "mechanism",
    "model_geometry",
    "measured_outcome",
    "failure_reason",
    "reopen_condition",
    "evidence_binding",
)

TRANSFER_MATRIX_FIELDS: tuple[str, ...] = (
    "source_family",
    "target_family",
    "mechanism",
    "transfer_status",
    "compatibility_conditions",
    "required_validation",
    "negative_science_link",
)

MECHANISM_INHERITANCE_RULES: tuple[str, ...] = (
    "accepted_kernel_representation_scheduler_result_is_indexed",
    "cross_family_transfer_requires_explicit_compatibility_conditions",
    "unverified_transfer_cannot_skip_exact_model_parity_capability_or_tg",
    "negative_science_is_retrieved_before_repeat_experiment",
    "sqlite_mechanism_index_hash_is_bound_by_transfer_matrix",
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


def knowledge_plane_contract(*, bible_sha256: str | None = None) -> dict[str, Any]:
    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_CONFIGURATION_ONLY",
            "recorded_at": _utc_now(),
            "bible_sha256": bible_sha256,
            "outputs": {
                "ASCENSION_KERNEL_GENOME": {"format": "jsonl", "fields": list(KERNEL_GENOME_FIELDS)},
                "ASCENSION_REPRESENTATION_GENOME": {"format": "jsonl", "fields": list(REPRESENTATION_GENOME_FIELDS)},
                "ASCENSION_SCHEDULER_GENOME": {"format": "jsonl", "fields": list(SCHEDULER_GENOME_FIELDS)},
                "ASCENSION_NEGATIVE_SCIENCE": {"format": "jsonl", "fields": list(NEGATIVE_SCIENCE_FIELDS)},
                "ASCENSION_TRANSFER_MATRIX": {"format": "json", "fields": list(TRANSFER_MATRIX_FIELDS)},
                "ASCENSION_MECHANISM_INDEX": {"format": "sqlite", "hash_binding_required": True},
            },
            "mechanism_inheritance_rules": list(MECHANISM_INHERITANCE_RULES),
            "claim_boundary": {
                "configuration_is_not_knowledge_plane_evidence": True,
                "no_candidate_or_implementation_may_skip_direct_model_validation": True,
                "negative_science_is_not_deleted_or_silently_ignored": True,
            },
        }
    )


def write_knowledge_plane_contract(
    root: str | Path, *, bible_sha256: str | None = None
) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    document = knowledge_plane_contract(bible_sha256=bible_sha256)
    _atomic_json(resolved / FILENAME, document)
    return document


__all__ = [
    "FILENAME",
    "KERNEL_GENOME_FIELDS",
    "MECHANISM_INHERITANCE_RULES",
    "NEGATIVE_SCIENCE_FIELDS",
    "REPRESENTATION_GENOME_FIELDS",
    "SCHEMA",
    "SCHEDULER_GENOME_FIELDS",
    "TRANSFER_MATRIX_FIELDS",
    "knowledge_plane_contract",
    "write_knowledge_plane_contract",
]
