"""LPC_DATASET — experiment data contract for a learned physical compiler.

Data before ML. A row binds every physical experiment to the identity and
measurement fields required to learn from it. Missing keys are REJECTED.
Unmeasured values are null with a reason code and are never imputed to 0.

This sidecar does not measure. Ingest is a field-presence projection of
existing receipts. Hardware numbers that happen to be present on a source
row stay in memory for baseline tests; they are never sealed into the
receipt under a hardware field name.

    python3 tools/future/lpc_dataset.py --build
    python3 tools/future/lpc_dataset.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "LPC_DATASET.json"
SCHEMA = "hawking.future.lpc_dataset.v1"

# Reused verbatim from tools/accelerator/physical_qualification.py
# MEASUREMENT_FIELDS / queue measurement_contract.required_fields.
MEASUREMENT_CONTRACT_FIELDS: tuple[str, ...] = (
    "total_nx_bytes",
    "resident_bytes",
    "active_representation_bytes_per_token",
    "actual_read_bytes_per_token",
    "transient_bytes_per_token",
    "gpu_ns_per_token",
    "complete_wall_ns_per_accepted_token",
    "dispatches_per_token",
    "sync_ns_per_token",
    "accepted_tps",
    "fallback_count",
)

# Binding the contract asked for. Names that already exist on the Accelerator
# scoreboard are reused exactly; the rest are the gaps the scoreboard does not bind.
REQUIRED_FIELDS: tuple[str, ...] = (
    "model",
    "organ_fingerprint",
    "representation",
    "machine_genome",
    "physical_graph_identity",
    "backend",
    "layout",
    "tile",
    "grouping",
    "fusion",
    "persistent_resources",
    "active_bytes",
    "resident_bytes",
    "dispatches",
    "synchronization",
    "latency",
    "complete_token_effect",
    "contamination_class",
    "capability",
)

NUMERIC_FIELDS: tuple[str, ...] = (
    "active_bytes",
    "resident_bytes",
    "dispatches",
    "synchronization",
    "latency",
)

CONTAMINATION_CLASSES: tuple[str, ...] = (
    "PROTECTED_ABSOLUTE",
    "DIAGNOSTIC_RELATIVE",
    "STATIC_ONLY",
    "UNKNOWN",
)

# QUALIFIED_PROTECTED is the scoreboard/HCLI spelling of a protected run.
_PROTECTED_SOURCE_CLASSES = frozenset({"PROTECTED_ABSOLUTE", "QUALIFIED_PROTECTED"})
_DIAGNOSTIC_SOURCE_CLASSES = frozenset({"DIAGNOSTIC_RELATIVE", "DIAGNOSTIC_CONTAMINATED"})

NULL_REASONS: tuple[str, ...] = (
    "UNMEASURED",
    "NOT_IN_SOURCE",
    "AWAITING_PROTECTED_RECEIPT",
    "HARDWARE_AUTHORITY_REQUIRED",
    "SPARSE_OR_UNTRACKED_SOURCE",
    "PARTIAL_IDENTITY",
    "NOT_APPLICABLE",
    "STATIC_PLAN_ONLY",
    "DIAGNOSTIC_NOT_PROMOTABLE",
    "UNKNOWN_FIELD",
)

SOURCE_SCOREBOARD = "receipts/headless/ACCELERATOR_SCOREBOARD.json"
SOURCE_QUEUE = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
SOURCE_BUDGET = "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json"
SOURCE_GENOME = "receipts/headless/MACHINE_GENOME.json"

FIELD_ORIGINS: dict[str, str] = {
    "model": "scoreboard.model / queue.candidates[].model / QWEN27_TOKEN_NS_BUDGET.model",
    "organ_fingerprint": (
        "queue.affected_physical_region / QWEN27_TOKEN_NS_BUDGET.organs[].organ; "
        "scoreboard does not bind an organ"
    ),
    "representation": "scoreboard.representation / budget.baseline.representation",
    "machine_genome": (
        "scoreboard.machine is a partial identity; MACHINE_GENOME.json is dataset "
        "context and is not auto-attached (that would fabricate a binding)"
    ),
    "physical_graph_identity": "scoreboard.executable_id / hcli.physical_graph fingerprint",
    "backend": "scoreboard.backend / queue.work_units[].preferred_backend",
    "layout": "absent from scoreboard; physical_graph layout algebra is the intended source",
    "tile": "absent from scoreboard; qualification mutation notes are the intended source",
    "grouping": "absent from scoreboard; qualification mutation notes are the intended source",
    "fusion": "queue.exact_mutation.child_fusion_env",
    "persistent_resources": "absent from scoreboard and queue",
    "active_bytes": (
        "scoreboard.active_bytes_per_token / "
        "measurement_contract.active_representation_bytes_per_token"
    ),
    "resident_bytes": "scoreboard.resident_bytes / measurement_contract.resident_bytes",
    "dispatches": "scoreboard.dispatches / measurement_contract.dispatches_per_token",
    "synchronization": "scoreboard.synchronization_ns / measurement_contract.sync_ns_per_token",
    "latency": (
        "scoreboard.complete_token_ns or wall_ns_per_token / "
        "measurement_contract.complete_wall_ns_per_accepted_token; "
        "stored under 'latency' so a sidecar receipt cannot claim token_ns/gpu_ns/wall_ns"
    ),
    "complete_token_effect": "scoreboard.capability_verified combined with fallback_count",
    "contamination_class": (
        "scoreboard.benchmark_class mapped onto PROTECTED_ABSOLUTE | "
        "DIAGNOSTIC_RELATIVE | STATIC_ONLY | UNKNOWN"
    ),
    "capability": "scoreboard.capability_verified",
}

# Recover-time snapshot against parent working-tree disk on 2026-08-29.
# Those three receipts are campaign evidence: present on disk in the parent
# tree, absent from git HEAD, and not materialized in this sparse worktree.
# Live ingest below never reads outside REPO.
RECOVERED_PARENT_CENSUS: dict[str, Any] = {
    "note": (
        "Mandatory recover, 2026-08-29. Scoreboard/queue/budget are untracked "
        "campaign evidence: not in git HEAD, not in this sparse worktree. "
        "Live ingest uses only REPO-relative paths. These counts are a recover "
        "snapshot, not a sealed hardware claim."
    ),
    "scoreboard_schema": "hawking.accelerator.scoreboard.v1",
    "scoreboard_module": "tools/accelerator/scoreboard.py",
    "scoreboard_rows": 13,
    "scoreboard_rows_with_model": 3,
    "scoreboard_rows_with_backend": 6,
    "scoreboard_rows_with_representation": 3,
    "scoreboard_rows_with_organ_fingerprint": 0,
    "scoreboard_rows_with_complete_token_ns": 1,
    "scoreboard_rows_with_resident_bytes": 0,
    "scoreboard_rows_with_accepted_tps": 0,
    "scoreboard_rows_with_capability_verified": 0,
    "scoreboard_lpc_complete_rows": 0,
    "queue_schema": "hawking.accelerator.physical_qualification_queue.v1",
    "queue_module": "tools/accelerator/physical_qualification.py",
    "queue_candidates": 30,
    "queue_measured": 0,
    "queue_by_status": {"BLOCKED": 14, "READY_PROTECTED": 12, "STATIC_ONLY": 4},
    "budget_schema": "hawking.accelerator.qwen27_token_ns_budget.v1",
    "budget_module": "tools/accelerator/qwen27_token_budget.py",
    "budget_organs": 9,
    "budget_organs_measured": 0,
    "finding": (
        "0 LPC-complete rows on recover. The learned compiler is data-starved; "
        "that is the condition this contract exists to make visible."
    ),
}


class SchemaError(ValueError):
    """A row violates the LPC data contract."""


class ImputationError(ValueError):
    """Raised when a null field would be silently filled with 0."""


def _in_git_head(rel: str) -> bool:
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def read_json_source(rel: str) -> tuple[Any | None, dict[str, Any]]:
    """Load a JSON document from REPO disk, else from git HEAD. Never from a sibling tree."""
    path = REPO / rel
    meta: dict[str, Any] = {
        "path": rel,
        "on_disk": path.is_file(),
        "in_git_HEAD": _in_git_head(rel),
        "loaded_from": None,
    }
    if path.is_file():
        meta["loaded_from"] = "disk"
        return load_json(path), meta
    if meta["in_git_HEAD"]:
        raw = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if raw.returncode == 0 and raw.stdout:
            meta["loaded_from"] = "git_HEAD"
            return json.loads(raw.stdout), meta
    return None, meta


def contamination_from_benchmark_class(value: Any) -> str:
    label = str(value or "UNKNOWN").upper()
    if label in _PROTECTED_SOURCE_CLASSES:
        return "PROTECTED_ABSOLUTE"
    if label in _DIAGNOSTIC_SOURCE_CLASSES:
        return "DIAGNOSTIC_RELATIVE"
    if label == "STATIC_ONLY":
        return "STATIC_ONLY"
    return "UNKNOWN"


def _canonical(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and value == "":
            return None
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(value)
        return text or None


def _identity_missing(value: Any) -> bool:
    return _canonical(value) is None


def _number(value: Any) -> int | float | None:
    """Parse a scalar. Measured zero is kept. None stays None. Bool is not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def as_numeric(row: Mapping[str, Any], field: str) -> int | float | None:
    """Return a numeric field or None. Never coerces None to 0."""
    return _number(row.get(field))


def forbid_zero_imputation(row: Mapping[str, Any], field: str) -> int | float:
    """Numeric accessor that refuses to invent a value for a null field."""
    value = as_numeric(row, field)
    if value is None:
        reason = (row.get("absence_reasons") or {}).get(field) or "UNMEASURED"
        raise ImputationError(
            f"{field} is null (reason={reason}); refusing to impute 0"
        )
    return value


def row_template(
    *,
    reasons_for_missing: str | None = "UNMEASURED",
    **fields: Any,
) -> dict[str, Any]:
    """Build a row that contains every required key.

    Remaining nulls get `reasons_for_missing` unless the caller already set
    absence_reasons. Pass reasons_for_missing=None to leave silent nulls
    (useful only for negative tests).
    """
    row: dict[str, Any] = {name: fields.get(name, None) for name in REQUIRED_FIELDS}
    reasons = dict(fields.get("absence_reasons") or {})
    if reasons_for_missing:
        for name in REQUIRED_FIELDS:
            if row[name] is None and name not in reasons:
                reasons[name] = reasons_for_missing
    row["absence_reasons"] = reasons
    for key, value in fields.items():
        if key in REQUIRED_FIELDS or key == "absence_reasons":
            continue
        row[key] = value
    return row


def validate_row(row: Any) -> dict[str, Any]:
    """Classify a row: REJECTED / INVALID_NULL / VALID, plus completeness."""
    if not isinstance(row, Mapping):
        return {"status": "REJECTED", "why": "not_an_object", "complete": False}
    missing = [name for name in REQUIRED_FIELDS if name not in row]
    if missing:
        return {
            "status": "REJECTED",
            "why": "missing_keys",
            "missing": missing,
            "complete": False,
        }
    reasons = row.get("absence_reasons") if isinstance(row.get("absence_reasons"), Mapping) else {}
    silent: list[str] = []
    unknown: list[tuple[str, Any]] = []
    for name in REQUIRED_FIELDS:
        if row[name] is None:
            reason = reasons.get(name)
            if not reason:
                silent.append(name)
            elif reason not in NULL_REASONS:
                unknown.append((name, reason))
    if silent:
        return {
            "status": "INVALID_NULL",
            "why": "silent_null",
            "fields": silent,
            "complete": False,
        }
    if unknown:
        return {
            "status": "INVALID_NULL",
            "why": "unknown_reason",
            "fields": unknown,
            "complete": False,
        }
    klass = row.get("contamination_class")
    if klass not in CONTAMINATION_CLASSES:
        return {
            "status": "INVALID_NULL",
            "why": "unknown_contamination_class",
            "fields": ["contamination_class"],
            "complete": False,
        }
    complete = (
        all(row[name] is not None for name in REQUIRED_FIELDS)
        and klass == "PROTECTED_ABSOLUTE"
    )
    return {"status": "VALID", "why": None, "complete": complete}


def _set_field(
    row: dict[str, Any],
    reasons: dict[str, str],
    sources: dict[str, str],
    name: str,
    value: Any,
    *,
    source_field: str | None,
    null_reason: str,
) -> None:
    if name in NUMERIC_FIELDS:
        parsed = _number(value)
        if parsed is None and value is not None and not _identity_missing(value):
            parsed = None
        row[name] = parsed
        if parsed is None:
            reasons[name] = null_reason
        elif source_field:
            sources[name] = source_field
        return
    if _identity_missing(value):
        row[name] = None
        reasons[name] = null_reason
        return
    row[name] = value if not isinstance(value, (dict, list)) else json.loads(
        json.dumps(value, sort_keys=True, default=str)
    )
    if source_field:
        sources[name] = source_field


def _complete_token_effect(capability: Any, fallback_count: Any) -> tuple[Any, str | None]:
    fallback = _number(fallback_count)
    if capability is True and fallback == 0:
        return True, None
    if capability is False or (fallback is not None and fallback > 0):
        return False, None
    if fallback == 0 and capability is None:
        return None, "PARTIAL_IDENTITY"
    return None, "NOT_IN_SOURCE"


def project_scoreboard_row(
    source: Mapping[str, Any],
    *,
    row_id: str,
) -> dict[str, Any]:
    """Project one Accelerator scoreboard row onto the LPC contract.

    Reuses scoreboard field names. Does not invent organs, layouts, tiles,
    genomes, or measurements the source did not bind.
    """
    reasons: dict[str, str] = {}
    sources: dict[str, str] = {}
    row: dict[str, Any] = {name: None for name in REQUIRED_FIELDS}
    row["row_id"] = row_id
    row["source"] = "scoreboard"
    row["source_receipt"] = source.get("receipt")

    _set_field(row, reasons, sources, "model", source.get("model"),
               source_field="model", null_reason="NOT_IN_SOURCE")
    _set_field(row, reasons, sources, "organ_fingerprint", None,
               source_field=None, null_reason="NOT_IN_SOURCE")
    _set_field(row, reasons, sources, "representation", source.get("representation"),
               source_field="representation", null_reason="NOT_IN_SOURCE")
    machine = source.get("machine")
    _set_field(
        row, reasons, sources, "machine_genome", machine,
        source_field="machine",
        null_reason="NOT_IN_SOURCE" if _identity_missing(machine) else "PARTIAL_IDENTITY",
    )
    if row["machine_genome"] is not None:
        # scoreboard.machine is not a MachineGenome; keep the value but mark it partial.
        reasons["machine_genome"] = "PARTIAL_IDENTITY"
    _set_field(
        row, reasons, sources, "physical_graph_identity", source.get("executable_id"),
        source_field="executable_id", null_reason="NOT_IN_SOURCE",
    )
    _set_field(row, reasons, sources, "backend", source.get("backend"),
               source_field="backend", null_reason="NOT_IN_SOURCE")
    for name in ("layout", "tile", "grouping", "fusion", "persistent_resources"):
        _set_field(row, reasons, sources, name, None,
                   source_field=None, null_reason="NOT_IN_SOURCE")

    active = _number(source.get("active_bytes_per_token"))
    active_src = "active_bytes_per_token"
    if active is None:
        active = _number(source.get("active_representation_bytes_per_token"))
        active_src = "active_representation_bytes_per_token"
    if active is None:
        active = _number(source.get("active_weight_bytes_per_generated_token"))
        active_src = "active_weight_bytes_per_generated_token"
    _set_field(row, reasons, sources, "active_bytes", active,
               source_field=active_src if active is not None else None,
               null_reason="UNMEASURED")

    _set_field(
        row, reasons, sources, "resident_bytes", source.get("resident_bytes"),
        source_field="resident_bytes", null_reason="UNMEASURED",
    )
    disp = _number(source.get("dispatches_per_token"))
    disp_src = "dispatches_per_token"
    if disp is None:
        disp = _number(source.get("dispatches"))
        disp_src = "dispatches"
    _set_field(row, reasons, sources, "dispatches", disp,
               source_field=disp_src if disp is not None else None,
               null_reason="UNMEASURED")
    sync = _number(source.get("synchronization_ns"))
    sync_src = "synchronization_ns"
    if sync is None:
        sync = _number(source.get("sync_ns_per_token"))
        sync_src = "sync_ns_per_token"
    _set_field(row, reasons, sources, "synchronization", sync,
               source_field=sync_src if sync is not None else None,
               null_reason="UNMEASURED")

    latency = _number(source.get("complete_token_ns"))
    lat_src = "complete_token_ns"
    if latency is None:
        latency = _number(source.get("complete_wall_ns_per_accepted_token"))
        lat_src = "complete_wall_ns_per_accepted_token"
    if latency is None:
        latency = _number(source.get("wall_ns_per_token"))
        lat_src = "wall_ns_per_token"
    if latency is None:
        latency = _number(source.get("gpu_ns_per_token"))
        lat_src = "gpu_ns_per_token"
    _set_field(row, reasons, sources, "latency", latency,
               source_field=lat_src if latency is not None else None,
               null_reason="UNMEASURED")

    effect, effect_reason = _complete_token_effect(
        source.get("capability_verified"), source.get("fallback_count")
    )
    _set_field(
        row, reasons, sources, "complete_token_effect", effect,
        source_field="capability_verified+fallback_count" if effect is not None else None,
        null_reason=effect_reason or "NOT_IN_SOURCE",
    )
    _set_field(
        row, reasons, sources, "contamination_class",
        contamination_from_benchmark_class(
            source.get("benchmark_class") or source.get("evidence_mode")
        ),
        source_field="benchmark_class",
        null_reason="NOT_IN_SOURCE",
    )
    _set_field(
        row, reasons, sources, "capability", source.get("capability_verified"),
        source_field="capability_verified", null_reason="NOT_IN_SOURCE",
    )
    row["absence_reasons"] = {k: reasons[k] for k in sorted(reasons)}
    row["field_sources"] = sources
    return row


def project_queue_candidate(
    candidate: Mapping[str, Any],
    *,
    row_id: str,
    backend: Any = None,
) -> dict[str, Any]:
    """Project one physical-qualification candidate. Plans are not measurements."""
    reasons: dict[str, str] = {}
    sources: dict[str, str] = {}
    row: dict[str, Any] = {name: None for name in REQUIRED_FIELDS}
    row["row_id"] = row_id
    row["source"] = "qualification_queue"
    row["source_receipt"] = candidate.get("candidate_id")
    measurements = candidate.get("measurements") if isinstance(candidate.get("measurements"), Mapping) else {}
    absence = measurements.get("absence_reasons") if isinstance(measurements.get("absence_reasons"), Mapping) else {}

    _set_field(row, reasons, sources, "model", candidate.get("model"),
               source_field="model", null_reason="NOT_IN_SOURCE")
    _set_field(
        row, reasons, sources, "organ_fingerprint", candidate.get("affected_physical_region"),
        source_field="affected_physical_region", null_reason="NOT_IN_SOURCE",
    )
    _set_field(row, reasons, sources, "representation", None,
               source_field=None, null_reason="NOT_IN_SOURCE")
    _set_field(row, reasons, sources, "machine_genome", None,
               source_field=None, null_reason="STATIC_PLAN_ONLY")
    _set_field(row, reasons, sources, "physical_graph_identity", None,
               source_field=None, null_reason="STATIC_PLAN_ONLY")
    _set_field(row, reasons, sources, "backend", backend,
               source_field="work_units.preferred_backend", null_reason="NOT_IN_SOURCE")
    for name in ("layout", "tile", "grouping", "persistent_resources"):
        _set_field(row, reasons, sources, name, None,
                   source_field=None, null_reason="NOT_IN_SOURCE")
    mutation = candidate.get("exact_mutation")
    fusion = None
    if isinstance(mutation, Mapping):
        fusion = mutation.get("child_fusion_env", mutation)
    _set_field(row, reasons, sources, "fusion", fusion,
               source_field="exact_mutation.child_fusion_env", null_reason="NOT_IN_SOURCE")

    def _metric(contract_name: str, lpc_name: str) -> None:
        value = measurements.get(contract_name)
        reason = absence.get(contract_name) or "AWAITING_PROTECTED_RECEIPT"
        # Translate queue English into a closed reason code when it matches the
        # standard waiting phrase; otherwise keep AWAITING_PROTECTED_RECEIPT.
        if value is None:
            reason_code = "AWAITING_PROTECTED_RECEIPT"
            if "diagnostic" in str(reason).lower():
                reason_code = "DIAGNOSTIC_NOT_PROMOTABLE"
            _set_field(row, reasons, sources, lpc_name, None,
                       source_field=contract_name, null_reason=reason_code)
        else:
            _set_field(row, reasons, sources, lpc_name, value,
                       source_field=contract_name, null_reason="UNMEASURED")

    _metric("active_representation_bytes_per_token", "active_bytes")
    _metric("resident_bytes", "resident_bytes")
    _metric("dispatches_per_token", "dispatches")
    _metric("sync_ns_per_token", "synchronization")
    _metric("complete_wall_ns_per_accepted_token", "latency")
    _set_field(
        row, reasons, sources, "complete_token_effect", None,
        source_field=None, null_reason="AWAITING_PROTECTED_RECEIPT",
    )
    _set_field(
        row, reasons, sources, "contamination_class", "STATIC_ONLY",
        source_field="queue_status", null_reason="STATIC_PLAN_ONLY",
    )
    _set_field(
        row, reasons, sources, "capability", None,
        source_field=None, null_reason="AWAITING_PROTECTED_RECEIPT",
    )
    row["absence_reasons"] = {k: reasons[k] for k in sorted(reasons)}
    row["field_sources"] = sources
    row["queue_status"] = candidate.get("status")
    return row


def project_budget_organ(
    organ: Mapping[str, Any],
    *,
    model: Any,
    representation: Any,
    row_id: str,
) -> dict[str, Any]:
    """Project one Qwen27 token-ns budget organ. The ledger is planned, not measured."""
    reasons: dict[str, str] = {}
    sources: dict[str, str] = {}
    row: dict[str, Any] = {name: None for name in REQUIRED_FIELDS}
    row["row_id"] = row_id
    row["source"] = "qwen27_token_ns_budget"
    actual = organ.get("actual") if isinstance(organ.get("actual"), Mapping) else {}
    absence = organ.get("absence_reasons") if isinstance(organ.get("absence_reasons"), Mapping) else {}

    _set_field(row, reasons, sources, "model", model,
               source_field="model", null_reason="NOT_IN_SOURCE")
    _set_field(row, reasons, sources, "organ_fingerprint", organ.get("organ"),
               source_field="organ", null_reason="NOT_IN_SOURCE")
    _set_field(row, reasons, sources, "representation", representation,
               source_field="baseline.representation", null_reason="NOT_IN_SOURCE")
    for name, reason in (
        ("machine_genome", "STATIC_PLAN_ONLY"),
        ("physical_graph_identity", "STATIC_PLAN_ONLY"),
        ("backend", "NOT_IN_SOURCE"),
        ("layout", "NOT_IN_SOURCE"),
        ("tile", "NOT_IN_SOURCE"),
        ("grouping", "NOT_IN_SOURCE"),
        ("fusion", "NOT_IN_SOURCE"),
        ("persistent_resources", "NOT_IN_SOURCE"),
    ):
        _set_field(row, reasons, sources, name, None, source_field=None, null_reason=reason)

    mapping = (
        ("active_representation_bytes_per_token", "active_bytes"),
        ("resident_bytes", "resident_bytes"),
        ("dispatches_per_token", "dispatches"),
        ("sync_ns_per_token", "synchronization"),
        ("complete_wall_ns_per_accepted_token", "latency"),
    )
    for src, dest in mapping:
        value = actual.get(src)
        _set_field(
            row, reasons, sources, dest, value,
            source_field=src,
            null_reason="AWAITING_PROTECTED_RECEIPT" if value is None else "UNMEASURED",
        )
        if value is None and src in absence:
            reasons[dest] = "AWAITING_PROTECTED_RECEIPT"
    _set_field(row, reasons, sources, "complete_token_effect", None,
               source_field=None, null_reason="AWAITING_PROTECTED_RECEIPT")
    _set_field(row, reasons, sources, "contamination_class", "STATIC_ONLY",
               source_field="budget.status", null_reason="STATIC_PLAN_ONLY")
    _set_field(row, reasons, sources, "capability", None,
               source_field=None, null_reason="AWAITING_PROTECTED_RECEIPT")
    row["absence_reasons"] = {k: reasons[k] for k in sorted(reasons)}
    row["field_sources"] = sources
    return row


def ingest_scoreboard(doc: Mapping[str, Any] | None, *, prefix: str = "scoreboard") -> list[dict[str, Any]]:
    if not isinstance(doc, Mapping):
        return []
    rows_in = doc.get("rows")
    if not isinstance(rows_in, list):
        return []
    out = []
    for i, raw in enumerate(rows_in):
        if not isinstance(raw, Mapping):
            continue
        receipt = str(raw.get("receipt") or i)
        out.append(project_scoreboard_row(raw, row_id=f"lpc:{prefix}:{i:04d}:{receipt}"))
    return out


def ingest_queue(doc: Mapping[str, Any] | None, *, prefix: str = "queue") -> list[dict[str, Any]]:
    if not isinstance(doc, Mapping):
        return []
    candidates = doc.get("candidates")
    if not isinstance(candidates, list):
        return []
    backend_by_id: dict[str, Any] = {}
    work_units = doc.get("work_units")
    if isinstance(work_units, list):
        for unit in work_units:
            if isinstance(unit, Mapping) and unit.get("candidate_id"):
                backend_by_id[str(unit["candidate_id"])] = unit.get("preferred_backend")
    out = []
    for i, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            continue
        cid = str(raw.get("candidate_id") or i)
        out.append(
            project_queue_candidate(
                raw,
                row_id=f"lpc:{prefix}:{i:04d}:{cid}",
                backend=backend_by_id.get(cid),
            )
        )
    return out


def ingest_budget(doc: Mapping[str, Any] | None, *, prefix: str = "budget") -> list[dict[str, Any]]:
    if not isinstance(doc, Mapping):
        return []
    organs = doc.get("organs")
    if not isinstance(organs, list):
        return []
    model = doc.get("model")
    representation = None
    baseline = doc.get("baseline")
    if isinstance(baseline, Mapping):
        representation = baseline.get("representation")
    out = []
    for i, raw in enumerate(organs):
        if not isinstance(raw, Mapping):
            continue
        organ = str(raw.get("organ") or i)
        out.append(
            project_budget_organ(
                raw,
                model=model,
                representation=representation,
                row_id=f"lpc:{prefix}:{i:04d}:{organ}",
            )
        )
    return out


def classify_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    tallies = {
        "n": len(rows),
        "valid": 0,
        "complete": 0,
        "rejected": 0,
        "invalid_null": 0,
        "by_contamination": {c: 0 for c in CONTAMINATION_CLASSES},
        "by_source": {},
        "field_presence": {
            name: {"non_null": 0, "n": len(rows)} for name in REQUIRED_FIELDS
        },
    }
    for row in rows:
        verdict = validate_row(row)
        status = verdict["status"]
        if status == "VALID":
            tallies["valid"] += 1
            if verdict["complete"]:
                tallies["complete"] += 1
        elif status == "REJECTED":
            tallies["rejected"] += 1
        else:
            tallies["invalid_null"] += 1
        if isinstance(row, Mapping):
            klass = row.get("contamination_class")
            if klass in tallies["by_contamination"]:
                tallies["by_contamination"][klass] += 1
            src = str(row.get("source") or "unknown")
            tallies["by_source"][src] = tallies["by_source"].get(src, 0) + 1
            for name in REQUIRED_FIELDS:
                if row.get(name) is not None:
                    tallies["field_presence"][name]["non_null"] += 1
    tallies["by_source"] = dict(sorted(tallies["by_source"].items()))
    return tallies


def genome_identity(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Device identity only. Measured bandwidth is a hardware claim and is dropped."""
    if not isinstance(doc, Mapping):
        return None
    gpu_cores = doc.get("gpu_cores")
    if isinstance(gpu_cores, Mapping):
        gpu_cores = None
    return {
        "schema": doc.get("schema"),
        "soc": doc.get("soc"),
        "arch": doc.get("arch"),
        "cpu_cores": doc.get("cpu_cores"),
        "perf_cores": doc.get("perf_cores"),
        "efficiency_cores": doc.get("efficiency_cores"),
        "gpu_cores": gpu_cores,
        "memory_bytes": doc.get("memory_bytes"),
        "os": doc.get("os"),
        "knowledge_level": doc.get("knowledge_level"),
        "measured_bandwidth": None,
        "measured_bandwidth_reason": "HARDWARE_AUTHORITY_REQUIRED",
    }


def ingest_from_disk() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project every on-disk/git-HEAD source that exists in this worktree."""
    report: dict[str, Any] = {"sources": [], "rows": None}
    rows: list[dict[str, Any]] = []

    spec = (
        (SOURCE_SCOREBOARD, ingest_scoreboard),
        (SOURCE_QUEUE, ingest_queue),
        (SOURCE_BUDGET, ingest_budget),
    )
    for rel, projector in spec:
        doc, meta = read_json_source(rel)
        projected = projector(doc) if doc is not None else []
        verdicts = classify_rows(projected)
        entry = {
            **meta,
            "projected": verdicts["n"],
            "valid": verdicts["valid"],
            "complete": verdicts["complete"],
            "rejected": verdicts["rejected"],
            "invalid_null": verdicts["invalid_null"],
        }
        report["sources"].append(entry)
        rows.extend(projected)

    genome_doc, genome_meta = read_json_source(SOURCE_GENOME)
    report["machine_genome_source"] = genome_meta
    report["machine_genome_identity"] = genome_identity(genome_doc)
    report["machine_genome_attached_to_rows"] = False
    report["machine_genome_attach_reason"] = (
        "Scoreboard rows do not carry a genome fingerprint. Attaching this "
        "identity to every row would fabricate a binding."
    )
    report["inventory"] = classify_rows(rows)
    return rows, report


def _selftest_rows() -> None:
    missing = row_template(model="x")
    missing.pop("backend")
    assert validate_row(missing)["status"] == "REJECTED"

    valid = row_template(model="x", contamination_class="STATIC_ONLY")
    verdict = validate_row(valid)
    assert verdict["status"] == "VALID" and verdict["complete"] is False

    silent = row_template(reasons_for_missing=None, model="x", contamination_class="STATIC_ONLY")
    assert validate_row(silent)["status"] == "INVALID_NULL"

    zero = row_template(dispatches=0, contamination_class="STATIC_ONLY")
    assert as_numeric(zero, "dispatches") == 0
    assert forbid_zero_imputation(zero, "dispatches") == 0
    try:
        forbid_zero_imputation(valid, "dispatches")
        raise AssertionError("null dispatches must not impute")
    except ImputationError:
        pass

    fixture = {
        "receipt": "fixture.json",
        "model": "qwen3.8-27b-sealed-3.14",
        "backend": "hawking_native",
        "representation": "native-packed",
        "machine": "Apple M3 Ultra",
        "benchmark_class": "QUALIFIED_PROTECTED",
        "complete_token_ns": 1,
        "dispatches": 0,
        "fallback_count": 0,
        "capability_verified": None,
        "resident_bytes": None,
        "executable_id": "abc",
    }
    projected = project_scoreboard_row(fixture, row_id="lpc:fixture:0")
    v = validate_row(projected)
    assert v["status"] == "VALID"
    assert v["complete"] is False
    assert projected["dispatches"] == 0
    assert projected["contamination_class"] == "PROTECTED_ABSOLUTE"
    assert projected["organ_fingerprint"] is None
    assert projected["absence_reasons"]["organ_fingerprint"] == "NOT_IN_SOURCE"
    assert as_numeric(projected, "resident_bytes") is None


def selftest() -> None:
    _selftest_rows()
    from tools.future import lpc_baselines as lb
    lb.selftest()


def build() -> Path:
    selftest()
    from tools.future import lpc_baselines as lb

    _rows, ingest_report = ingest_from_disk()
    inventory = ingest_report["inventory"]
    complete = inventory["complete"]
    finding = (
        f"{complete} LPC-complete row(s) in this worktree. "
        + (
            "The learned compiler is data-starved; that is the condition this "
            "contract exists to make visible."
            if complete == 0
            else "Complete rows still cannot promote a sidecar prediction over a "
            "PROTECTED_ABSOLUTE measurement."
        )
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Experiment data contract binding every physical experiment to the "
            "identity and measurement fields required to learn from it, plus "
            "honest baselines with calibrated uncertainty and ABSTENTION."
        ),
        "head": git("rev-parse", "HEAD"),
        "row_schema": {
            "required_fields": list(REQUIRED_FIELDS),
            "numeric_fields": list(NUMERIC_FIELDS),
            "null_reasons": list(NULL_REASONS),
            "contamination_classes": list(CONTAMINATION_CLASSES),
            "field_origins": FIELD_ORIGINS,
            "rejection_rule": "a row missing a REQUIRED key is REJECTED",
            "null_policy": (
                "A field that is genuinely unmeasured is null with a reason code. "
                "A row may be VALID with nulls. A null field is never imputed to 0. "
                "Measured zero is kept."
            ),
            "complete_row": (
                "VALID, every required field non-null, contamination_class "
                "PROTECTED_ABSOLUTE. DIAGNOSTIC_RELATIVE never completes."
            ),
        },
        "measurement_contract_reused": {
            "source": (
                "tools/accelerator/physical_qualification.py MEASUREMENT_FIELDS / "
                "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json "
                "measurement_contract.required_fields"
            ),
            "fields": list(MEASUREMENT_CONTRACT_FIELDS),
            "null_policy_verbatim": (
                "missing physical metrics remain null until a native protected "
                "complete-token receipt records them"
            ),
            "protected_pass_requires_all_fields": True,
            "metric_scope": "accepted complete generated token; not isolated kernel time",
        },
        "ingest": ingest_report,
        "inventory": inventory,
        "finding": finding,
        "baselines": lb.describe(),
        "authority_rule": (
            "A PROTECTED_ABSOLUTE measurement always outranks a model prediction, "
            "including a confident one that disagrees. The sidecar cannot produce "
            "PROTECTED_ABSOLUTE evidence."
        ),
        "recovered_implementation": {
            "scoreboard": {
                "path": SOURCE_SCOREBOARD,
                "module": "tools/accelerator/scoreboard.py",
                "schema": "hawking.accelerator.scoreboard.v1",
                "role": (
                    "Derived view of sealed receipts. Row shape is the closest "
                    "existing LPC row. Nulls stay None (never zero). Not a dataset "
                    "contract: no organ/layout/tile/grouping/fusion/genome binding, "
                    "no abstention, no held-out splits."
                ),
            },
            "qualification_queue": {
                "path": SOURCE_QUEUE,
                "module": "tools/accelerator/physical_qualification.py",
                "schema": "hawking.accelerator.physical_qualification_queue.v1",
                "role": (
                    "Names measurement_contract.required_fields and the null policy "
                    "this module reuses. Candidates currently carry NOT_MEASURED "
                    "metrics with absence_reasons."
                ),
            },
            "qwen27_token_ns_budget": {
                "path": SOURCE_BUDGET,
                "module": "tools/accelerator/qwen27_token_budget.py",
                "schema": "hawking.accelerator.qwen27_token_ns_budget.v1",
                "role": (
                    "Organ list and the same measurement_contract field names, all "
                    "null until a native protected complete-token receipt exists."
                ),
            },
            "perf_model": {
                "path": "tools/accelerator/perf_model.py",
                "receipt": "receipts/headless/ACCELERATOR_PERF_MODEL.json",
                "role": (
                    "Four-feature ridge on threadgroup/ept milliseconds. Smallest "
                    "thing that asks whether a cliff is predictable. Not a dataset "
                    "contract and not an abstaining compiler."
                ),
            },
            "machine_genome": {
                "path": "tools/accelerator/machine_genome.py",
                "receipt": SOURCE_GENOME,
                "schema": "hawking.accelerator.machine_genome.v1",
                "role": (
                    "Device identity of this machine. Bandwidth is measured under "
                    "a protected window and is not copied into this sidecar receipt."
                ),
            },
            "physical_graph": {
                "path": "hcli/physical_graph.py",
                "role": (
                    "Defines PROTECTED_ABSOLUTE / DIAGNOSTIC_RELATIVE vocabulary "
                    "and a layout algebra; not a per-experiment LPC row."
                ),
            },
            "frontier_entry": {
                "id": "F007",
                "title": "No Learned Physical Compiler dataset contract",
                "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            },
            "parent_disk_census": RECOVERED_PARENT_CENSUS,
        },
        "gaps_closed": [
            "LPC row schema with required-key rejection and a closed null-reason set",
            "null policy that refuses to impute 0, with a helper that raises instead",
            "ingest adapters for scoreboard, qualification queue, and Qwen27 organ budget",
            "honest completeness census (complete rows currently expected to be 0)",
            "nearest measured neighbour with an explicit distance metric",
            "transparent rule cost model that abstains on null inputs",
            "uncertainty + ABSTENTION as a first-class outcome",
            "held-out splits by architecture, organ, and device",
            "authority helper: PROTECTED_ABSOLUTE always outranks a model",
        ],
        "negative_findings": [
            "ACCELERATOR_SCOREBOARD.json is not in git HEAD and is not materialized here",
            "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json is not in git HEAD and is not materialized here",
            "QWEN27_TOKEN_NS_BUDGET.json is not in git HEAD and is not materialized here",
            "scoreboard rows do not bind organ_fingerprint, layout, tile, grouping, fusion, or persistent_resources",
            "no LPC-complete row existed on recover against parent disk (13 scoreboard rows, 0 complete)",
            "queue candidates are 0/30 measured",
            "perf_model.py exists but is not a dataset contract",
            "this sidecar cannot emit PROTECTED_ABSOLUTE or DIAGNOSTIC_RELATIVE measurements",
            "no organ_fingerprint module exists; organ identity is recovered from queue/budget names only",
        ],
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. Inventory counts "
            "are field-presence census, not timings. Fixture baselines used by "
            "selftest are not physical predictions and are not sealed as rows."
        ),
    }
    return write_receipt(RECEIPT, doc, "tools/future/lpc_dataset.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest and not a.build:
        selftest()
        print("selftest ok")
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
