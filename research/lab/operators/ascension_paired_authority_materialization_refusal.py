"""Create a sealed, read-only refusal when paired authorities cannot be honest.

This is intentionally *not* a shortcut materializer for the paired-cognition
or final-tournament contracts.  It observes the fixed, public lifecycle
records and the ten contract definitions, then writes one create-new refusal
record when their sealed prerequisites are absent.  In particular, it never
manufactures a Q30/Q80 topology assertion, TG10 receipt, manager
qualification, hidden task, or candidate score.

The refusal record lives beside (but is not one of) the ten expected prepared
authority documents.  The readiness observer continues to treat every absent
prepared document as absent; this record only makes that decision auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_manager_tournament_readiness_report import (
    AUTHORITY_SPECS,
    CANDIDATES,
    ReadinessPaths,
    build_readiness_report,
)
from lab.operators.ascension_manager_tournament_readiness_report import (
    STATUS_PREPARED as READINESS_PREPARED_STATUS,
)
from lab.operators.ascension_manager_tournament_readiness_report import (
    STATUS_REFUSED as READINESS_REFUSED_STATUS,
)
from lab.receipts import SealIntegrityError, seal, verify

SCHEMA = "hawking.ascension.paired_authority_materialization_refusal_readiness.v1"
STATUS = "REFUSED_PAIRED_AUTHORITY_MATERIALIZATION_MISSING_TRUSTED_SEALED_INPUTS"
OUTPUT_FILENAME = "PAIRED_AUTHORITY_MATERIALIZATION_REFUSAL_READINESS.json"

OPERATIONAL_ASCENT_SCHEMA = "hawking.ascension.physical_operational_ascent.v1"
OPERATIONAL_WAITING_STATUS = "WAITING_FOR_BOTH_VALID_TG10_OPERATIONAL_RECEIPTS"


class PairedAuthorityMaterializationError(ValueError):
    """The input snapshot or create-new output boundary is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _fixed_document_metadata(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read exactly one declared JSON record; never recurse or follow symlinks."""

    metadata: dict[str, Any] = {
        "path": str(path),
        "read_scope": "fixed_declared_metadata_path_only",
        "exists": False,
        "regular_file": False,
        "sealed": False,
        "schema": None,
        "status_or_phase": None,
        "seal_sha256": None,
        "canonical_document_sha256": None,
        "issues": [],
    }
    try:
        node = path.lstat()
    except FileNotFoundError:
        metadata["issues"].append("missing")
        return metadata, None
    except OSError as error:
        metadata["issues"].append(f"unstatable:{error.__class__.__name__}")
        return metadata, None
    metadata["exists"] = True
    if stat.S_ISLNK(node.st_mode):
        metadata["issues"].append("symlink_refused")
        return metadata, None
    if not stat.S_ISREG(node.st_mode):
        metadata["issues"].append("not_regular_file")
        return metadata, None
    metadata["regular_file"] = True
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        metadata["issues"].append(
            f"unreadable_or_invalid_json:{error.__class__.__name__}"
        )
        return metadata, None
    if not isinstance(raw, Mapping):
        metadata["issues"].append("not_json_object")
        return metadata, None
    document = dict(raw)
    metadata["schema"] = document.get("schema")
    metadata["status_or_phase"] = document.get("status", document.get("phase"))
    metadata["seal_sha256"] = document.get("seal_sha256")
    metadata["canonical_document_sha256"] = _canonical_sha256(document)
    try:
        verify(document, label=str(path))
    except SealIntegrityError as error:
        metadata["issues"].append(f"unsealed_or_changed:{error}")
        return metadata, document
    metadata["sealed"] = True
    return metadata, document


def _observed_paths(paths: ReadinessPaths) -> dict[str, Path]:
    candidates = {candidate.key: candidate for candidate in CANDIDATES}
    return {
        "operational_ascent": paths.operational_ascent,
        "qwen30_exact_full_token_runtime": (
            paths.physical_root
            / "qwen30"
            / "complete-runtime"
            / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
        ),
        "qwen30_native_http_adapter": (
            paths.physical_root
            / "qwen30"
            / "complete-runtime"
            / "QWEN30_NATIVE_HTTP_ADAPTER_STATUS.json"
        ),
        "qwen30_base_true_tps_gate": (
            paths.physical_root
            / "qwen30"
            / "tps-gate"
            / "QWEN30_BASE_TRUE_TPS_GATE_STATUS.json"
        ),
        "qwen30_tg3": paths.tg_status(candidates["qwen30"]),
        "qwen80_complete_decoder_readiness": (
            paths.physical_root
            / "qwen80"
            / "complete-runtime"
            / "QWEN80_COMPLETE_DECODER_READINESS_AFTER_SOURCE_TOKEN_L0_20260809T063300Z.json"
        ),
        "qwen80_base_true_tps_gate": (
            paths.physical_root
            / "qwen80"
            / "tps-gate"
            / "QWEN80_BASE_TRUE_TPS_GATE_STATUS.json"
        ),
        "qwen80_tg3": paths.tg_status(candidates["qwen80"]),
    }


def _topology_requirements() -> dict[str, Any]:
    """The exact non-cloning topology required by the lane contract."""

    return {
        "qwen30": {
            "required_sealed_bindings": ["activation", "memory", "logical_session"],
            "topology_assertion": {
                "resident_model_processes": 1,
                "immutable_weight_copies": 1,
                "endpoint": {"host": "127.0.0.1", "port": 18430},
                "logical_session_policy": "many_logical_sessions",
            },
        },
        "qwen80": {
            "required_sealed_bindings": ["activation", "memory", "logical_session"],
            "topology_assertion": {
                "resident_model_processes": 1,
                "immutable_weight_copies": 1,
                "endpoint": {"host": "127.0.0.1", "port": 18480},
                "logical_session_policy": "many_logical_sessions",
            },
            "known_qwen80_contract_grammars": {
                "activation": {
                    "schema": "hawking.ascension.qwen80_resident_server_activation_result.v1",
                    "required_status": (
                        "ELIGIBLE_QWEN80_ONE_RESIDENT_SERVER_AUTOMATIC_LAUNCH_"
                        "PRECONDITION_ONLY"
                    ),
                },
                "memory": {
                    "schema": "hawking.ascension.qwen80_resident_memory_envelope_receipt.v1",
                    "required_status": (
                        "PREPARED_INCOMPLETE_QWEN80_ONE_RESIDENT_MANY_LOGICAL_SESSIONS_"
                        "MEMORY_ENVELOPE_PREFLIGHT_NO_RUNTIME_SERVER_OR_TPS"
                    ),
                },
                "logical_session": {
                    "schema": "hawking.ascension.qwen80_one_resident_session_scheduler_contract.v1",
                    "required_status": (
                        "PREPARED_INCOMPLETE_QWEN80_ONE_RESIDENT_MANY_LOGICAL_SESSIONS_"
                        "SCHEDULER_NO_RUNTIME_SERVER_OR_TPS"
                    ),
                },
            },
        },
        "invariant": "one Q30 body plus one Q80 body; no endpoint, weight, body, or session-namespace cloning",
    }


def _authority_input_grammar() -> dict[str, dict[str, Any]]:
    """Public, non-secret input grammar extracted from the ten Rust contracts."""

    return {
        "lane": {
            "sealed_inputs": [
                "three matching Q30 activation/memory/session bindings with one-body topology",
                "three matching Q80 activation/memory/session bindings with one-body topology",
            ],
            "policy_inputs": [
                "two lane definitions",
                "cross-lane read denial",
                "zero-action boundaries",
            ],
        },
        "mutation": {
            "sealed_inputs": ["prepared lane authority"],
            "policy_inputs": [
                "two lane action policies",
                "immutable protected-record policy",
                "final-selection reservation",
            ],
        },
        "knowledge": {
            "sealed_inputs": ["prepared lane authority", "prepared mutation authority"],
            "policy_inputs": [
                "generic-only knowledge policy",
                "optional independently redacted/provenanced/verified/published generic discovery releases",
            ],
        },
        "scheduler": {
            "sealed_inputs": [
                "prepared lane authority",
                "prepared mutation authority",
                "prepared knowledge authority",
            ],
            "policy_inputs": [
                "one-body resource budgets",
                "four non-cloning primary/helper role-session assignments",
                "bounded content-free queue/fairness telemetry",
                "TG10/TG3/final-mode reservations",
            ],
        },
        "tg10_development": {
            "sealed_inputs": [
                "prepared lane/mutation/knowledge/scheduler chain",
                "canonical operational-ascent status",
                "one exact, fresh Q30 TG10 operational receipt",
                "one exact, fresh Q80 TG10 operational receipt",
            ],
            "policy_inputs": [
                "paired-development activation request must remain false"
            ],
        },
        "tg3_freeze": {
            "sealed_inputs": [
                "pinned final-manager protocol",
                "prepared lane/mutation/knowledge/scheduler/TG10 chain",
                "two complete-manager qualification and TG3/freeze evidence sets",
            ],
            "policy_inputs": [
                "two equal evaluation-mode reservations",
                "protected corpus/fairness/selection/recovery reservations",
            ],
        },
        "protected_corpus": {
            "sealed_inputs": [
                "pinned final-manager protocol",
                "prepared paired chain through TG10",
                "immutable public corpus identity and protected membership commitment",
            ],
            "policy_inputs": [
                "15 public family metadata rows",
                "6 campaign metadata rows",
                "controller-only hidden namespace",
                "fair-envelope and read-only adversarial hooks",
            ],
        },
        "scorecard_adjudication": {
            "sealed_inputs": [
                "pinned final-manager protocol",
                "prepared TG3/freeze authority",
                "prepared protected corpus",
                "equalized resource/asymmetry ledger",
                "two-candidate × two-mode verified scorecards",
                "protected Pareto frontier",
            ],
            "policy_inputs": ["selection request remains false"],
        },
        "selection_recovery": {
            "sealed_inputs": [
                "pinned final-manager protocol",
                "prepared TG3/freeze",
                "prepared corpus",
                "prepared scorecard adjudication",
            ],
            "policy_inputs": [
                "protected selector identity",
                "winner immutable-seal reservation",
                "loser seal/cold-store/hash/restore/recovery reservation",
            ],
        },
        "final_report": {
            "sealed_inputs": [
                "pinned final-manager protocol",
                "prepared corpus",
                "prepared scorecard",
                "prepared selection/recovery",
            ],
            "policy_inputs": [
                "complete two-candidate side-by-side fields",
                "protected authorship/verifier",
                "decision narrative reservation",
            ],
        },
    }


def _operational_tg10_snapshot(document: Mapping[str, Any] | None) -> dict[str, Any]:
    document = _mapping(document)
    models = _mapping(_mapping(document.get("evidence")).get("models"))
    return {
        "schema_matches": document.get("schema") == OPERATIONAL_ASCENT_SCHEMA,
        "status": document.get("status"),
        "waiting_status_observed": document.get("status") == OPERATIONAL_WAITING_STATUS,
        "both_valid_tg10_receipts": document.get("both_valid_tg10_receipts") is True,
        "qwen30_tg10_receipt_seal_sha256": _mapping(models.get("qwen30")).get(
            "tg10_receipt_seal_sha256"
        ),
        "qwen80_tg10_receipt_seal_sha256": _mapping(models.get("qwen80")).get(
            "tg10_receipt_seal_sha256"
        ),
    }


def _expected_authority_paths(paths: ReadinessPaths) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for spec in AUTHORITY_SPECS:
        path = paths.authority_document(spec)
        metadata, document = _fixed_document_metadata(path)
        prepared_present = (
            document is not None
            and metadata["sealed"] is True
            and document.get("schema") == spec.expected_schema
            and document.get("status") == spec.expected_status
            and document.get("prepared") is True
        )
        records[spec.key] = {
            "path": str(path),
            "expected_schema": spec.expected_schema,
            "expected_prepared_status": spec.expected_status,
            "exists": metadata["exists"],
            "sealed": metadata["sealed"],
            "prepared_authority_present_before_refusal": prepared_present,
            "prepared_authority_created_by_this_refusal": False,
            "issues": metadata["issues"],
        }
    return records


def _missing_model_authorities(
    observed: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Do not turn an unrelated or unsealed runtime status into topology proof."""

    q30_runtime = observed["qwen30_exact_full_token_runtime"]
    q30_adapter = observed["qwen30_native_http_adapter"]
    q80_decoder = observed["qwen80_complete_decoder_readiness"]
    missing: list[dict[str, str]] = []
    for model_key, binding, evidence in (
        ("qwen30", "activation", q30_adapter),
        ("qwen30", "memory", q30_runtime),
        ("qwen30", "logical_session", q30_adapter),
        ("qwen80", "activation", q80_decoder),
        ("qwen80", "memory", q80_decoder),
        ("qwen80", "logical_session", q80_decoder),
    ):
        missing.append(
            {
                "model_key": model_key,
                "binding": binding,
                "reason": (
                    "no matching sealed one-body/many-logical-session topology authority; "
                    f"observed fixed record is schema={evidence.get('schema')!r}, "
                    f"sealed={evidence.get('sealed')!r}"
                ),
            }
        )
    return missing


def build_materialization_refusal(
    paths: ReadinessPaths,
    *,
    recorded_at: str | None = None,
    readiness_builder: Callable[..., Mapping[str, Any]] = build_readiness_report,
) -> dict[str, Any]:
    """Build a sealed refusal snapshot; never produce any prepared authority."""

    readiness = dict(readiness_builder(paths, claimed_ready=False))
    try:
        verify(readiness, label="read-only readiness snapshot")
    except SealIntegrityError as error:
        raise PairedAuthorityMaterializationError(
            f"readiness snapshot is unsealed or changed: {error}"
        ) from error
    if (
        readiness.get("schema")
        != "hawking.ascension.manager_tournament_readiness_report.v1"
    ):
        raise PairedAuthorityMaterializationError(
            "refusal helper requires the exact sealed manager-tournament readiness schema"
        )
    if readiness.get("status") == READINESS_PREPARED_STATUS:
        raise PairedAuthorityMaterializationError(
            "refusal helper cannot materialize a prepared authority chain; invoke each reviewed contract with its real sealed inputs"
        )
    if readiness.get("status") != READINESS_REFUSED_STATUS:
        raise PairedAuthorityMaterializationError(
            "refusal helper requires the canonical refused manager-tournament readiness status"
        )

    observed: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any] | None] = {}
    for key, path in _observed_paths(paths).items():
        metadata, document = _fixed_document_metadata(path)
        observed[key] = metadata
        documents[key] = document

    operational = _operational_tg10_snapshot(documents["operational_ascent"])
    expected_paths = _expected_authority_paths(paths)
    materialized_before = {
        key: _mapping(value).get("prepared") is True
        for key, value in _mapping(
            readiness.get("prepared_authority_materializations")
        ).items()
    }
    missing_prerequisites = list(readiness.get("missing_prerequisites") or [])
    blockers = list(readiness.get("blockers") or [])
    sources = _mapping(readiness.get("prepared_authority_source_identities"))

    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": recorded_at or _utc_now(),
        "prepared": False,
        "authority_chain_materialized": False,
        "expected_authority_documents_written": [],
        "refusal_record_is_not_an_authority_materialization": True,
        "paired_candidate_worlds_active": False,
        "paired_development_active": False,
        "tournament_active": False,
        "hidden_task_plaintext_read": False,
        "winner_selected": False,
        "observed_readiness_snapshot": {
            "schema": readiness.get("schema"),
            "status": readiness.get("status"),
            "seal_sha256": readiness.get("seal_sha256"),
            "canonical_document_sha256": _canonical_sha256(readiness),
            "missing_prerequisites": missing_prerequisites,
        },
        "fixed_observed_records": observed,
        "operational_tg10_snapshot": operational,
        "required_one_body_topology": _topology_requirements(),
        "missing_sealed_model_authorities": _missing_model_authorities(observed),
        "ten_contract_source_audit": {
            key: {
                "source_path": _mapping(sources.get(key)).get("source_path"),
                "source_sha256": _mapping(sources.get(key)).get("source_sha256"),
                "expected_schema": _mapping(sources.get(key)).get("expected_schema"),
                "expected_prepared_status": _mapping(sources.get(key)).get(
                    "expected_status"
                ),
                "prepared_definition_verified": _mapping(sources.get(key)).get(
                    "prepared_definition_verified"
                ),
                "input_grammar": _authority_input_grammar().get(key),
            }
            for key in _authority_input_grammar()
        },
        "expected_prepared_authority_paths": expected_paths,
        "readiness_report_delta": {
            "before_status": readiness.get("status"),
            "after_status": readiness.get("status"),
            "before_missing_prerequisites": missing_prerequisites,
            "after_missing_prerequisites": missing_prerequisites,
            "prepared_authorities_before": materialized_before,
            "prepared_authorities_after": materialized_before,
            "new_prepared_authorities_created": [],
            "delta_is_no_change": True,
            "reason": "a refusal record is intentionally not accepted as any of the ten prepared authority documents",
        },
        "binding_blockers": sorted(
            set(
                [
                    "canonical operational ascent does not bind both exact sealed TG10 operational receipts",
                    "no sealed Q30 activation/memory/many-logical-session topology binding is present",
                    "no sealed Q80 activation/memory/many-logical-session topology binding is present",
                    "none of the ten expected prepared authority documents may be fabricated from source definitions or unsealed observations",
                ]
                + missing_prerequisites
                + blockers
            )
        ),
        "authority_boundary": {
            "new_physical_model_processes_authorized": 0,
            "server_starts_authorized": 0,
            "port_binds_authorized": 0,
            "gpu_leases_authorized": 0,
            "tournament_state_mutations_authorized": 0,
            "paired_world_activation_authorized": False,
        },
        "execution_boundary": {
            "fixed_metadata_records_read_only": True,
            "live_artifact_scan_performed": False,
            "model_weights_loaded": False,
            "metal_device_or_dispatch_performed": False,
            "gpu_lease_or_registry_mutated": False,
            "model_or_decoder_token_executed": False,
            "logical_session_created": False,
            "runtime_watcher_or_server_started": False,
            "port_bound_or_listener_created": False,
            "hcli_executed": False,
            "tps_or_tg_measured": False,
            "hidden_task_created_or_read": False,
            "tournament_state_mutated": False,
        },
        "claim_boundary": [
            "This record is a sealed read-only refusal, not a prepared paired authority.",
            "One Q30 and one Q80 body with many logical sessions remains the only allowed topology.",
            "No model, device, server, watcher, lease, task corpus plaintext, TPS/TG, or tournament action occurred.",
        ],
    }
    return seal(report)


def write_new_refusal(paths: ReadinessPaths, report: Mapping[str, Any]) -> Path:
    """Create the one fixed refusal record without replacing any authority document."""

    output = paths.authority_root / OUTPUT_FILENAME
    if not output.is_absolute():
        raise PairedAuthorityMaterializationError(
            "authority root must resolve to an absolute path"
        )
    expected_authority_paths = {
        paths.authority_document(spec) for spec in AUTHORITY_SPECS
    }
    if output in expected_authority_paths:
        raise PairedAuthorityMaterializationError(
            "refusal output must not overlap a prepared authority path"
        )
    paths.authority_root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(paths.authority_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite existing refusal record: {output}"
        ) from None
    return output


def main(argv: Sequence[str] | None = None) -> int:
    defaults = ReadinessPaths.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(defaults.repository_root))
    parser.add_argument("--lifecycle-root", default=str(defaults.lifecycle_root))
    parser.add_argument("--physical-root", default=str(defaults.physical_root))
    parser.add_argument("--authority-root", default=str(defaults.authority_root))
    parser.add_argument("--recorded-at", default=None)
    arguments = parser.parse_args(argv)
    paths = ReadinessPaths.from_roots(
        repository_root=arguments.repository_root,
        lifecycle_root=arguments.lifecycle_root,
        physical_root=arguments.physical_root,
        authority_root=arguments.authority_root,
    )
    try:
        report = build_materialization_refusal(paths, recorded_at=arguments.recorded_at)
        output = write_new_refusal(paths, report)
    except (OSError, PairedAuthorityMaterializationError, SealIntegrityError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "schema": SCHEMA,
                "status": STATUS,
                "seal_sha256": report["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "OUTPUT_FILENAME",
    "SCHEMA",
    "STATUS",
    "PairedAuthorityMaterializationError",
    "build_materialization_refusal",
    "main",
    "write_new_refusal",
]


if __name__ == "__main__":  # pragma: no cover - exercised through main.
    raise SystemExit(main())
