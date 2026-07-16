#!/usr/bin/env python3.12
"""Fail-closed authority for two pending Doctor V5 GC successor transitions.

Packed-payload GC receipts deliberately bind the exact runtime spec of their
immediate successor.  A pending-only acceleration changed that execution
identity for two already-GC'd predecessors without changing either successor's
semantic program or campaign cell identity.  This module records and validates
that very narrow historical transition.  It does not weaken the GC receipt,
authorize any deletion, or authorize a scientific/quality claim.

The authority intentionally does *not* bind a future runtime-spec file hash.
Instead, the future spec binds this authority and this module as ordinary
inputs, while :func:`validate_transition` proves that its semantic program is
the exact historical program recorded by the original receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODULE_PATH = Path(__file__).resolve()
WRAPPER_PATH = HERE / "doctor_v5_qwen_treatment_block_parallel_adapter.py"
PLAN_REL = "reports/condense/doctor_v5_ultra/campaign_plan.json"
STATE_REL = "reports/condense/doctor_v5_ultra/queue_state.json"
DEFAULT_AUTHORITY_PATH = (
    ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/"
    "gc_runtime_transition_authority.json"
)
EXPECTED_PLAN_SHA256 = "3d254b5f7fcc5f02b55f2a71f306f7f6852839b699fd14ab4ddf5a05dbaa0106"
SCHEMA = "hawking.doctor_v5_gc_runtime_transition_authority.v1"
SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 64 * 1024 * 1024


class GCRuntimeTransitionError(RuntimeError):
    """The narrow historical transition could not be proven exactly."""


_INCIDENTS: tuple[dict[str, Any], ...] = (
    {
        "incident_id": "3b-3bpw-full-after-conditional-gc",
        "consumer": {
            "cell_id": "qwen2-5-3b__3bpw__doctor-full",
            "cell_identity_sha256": (
                "f5013423f72383e7903dcf38b9c5f95788067e3fc6ccca6470f4b5311b5efb90"
            ),
            "program_spec_sha256": (
                "5a23ec4520115a11dd5d054548896a57ab9ffa494b75fe5de434cf43e7437e9c"
            ),
            "branch": "doctor_full", "label": "3B", "rate_id": "3",
        },
        "predecessor": {
            "cell_id": "qwen2-5-3b__3bpw__doctor-conditional",
            "cell_identity_sha256": (
                "8ef3c75e85a35eb7249568ec9d672cdaaddf3e928157dfa806618f0db8deebdf"
            ),
            "result_sha256": (
                "18820d66f30eba53e35697e9b9225238e490a5373d86dd16560b867a60dbf788"
            ),
        },
        "receipt_rel": (
            "reports/condense/doctor_v5_ultra/results/"
            "qwen2-5-3b__3bpw__doctor-conditional/packed_gc_receipt.json"
        ),
        "receipt_file_sha256": (
            "3aff4dce2b9ad991e4c772d943c2c802adb53b4f5e670fe8b5846dad8433050f"
        ),
        "receipt_bytes": 2222,
        "receipt_sha256": (
            "5ddfc27d6bde034c3f9c6d808543d50adccd015c0d7cb8cb54833a85f109afa1"
        ),
        "old_spec_rel": (
            "reports/condense/doctor_v5_ultra/staged_acceleration/pending_runtime/"
            "808d8533d8658222d2f730fd540cc61b023abacac6a6beb131a0153ec3d4c34d/"
            "rollback/1784050520599938000-93160897ec0cc54a/"
            "spec-qwen2-5-3b__3bpw__doctor-full.json"
        ),
        "successor": {
            "cell_id": "qwen2-5-3b__3bpw__doctor-full",
            "cell_identity_sha256": (
                "f5013423f72383e7903dcf38b9c5f95788067e3fc6ccca6470f4b5311b5efb90"
            ),
            "runtime_spec_path": (
                "reports/condense/doctor_v5_ultra/runtime_specs/"
                "qwen2-5-3b__3bpw__doctor-full.json"
            ),
            "runtime_spec_sha256": (
                "d2b56fb9f46b3d73e5769869a2fd383b78115e978bbca2dd4800642f6dd10fed"
            ),
            "runtime_spec_bytes": 10876,
            "program_spec_sha256": (
                "5a23ec4520115a11dd5d054548896a57ab9ffa494b75fe5de434cf43e7437e9c"
            ),
        },
    },
    {
        "incident_id": "32b-4bpw-static-after-codec-gc",
        "consumer": {
            "cell_id": "qwen2-5-32b__4bpw__doctor-static",
            "cell_identity_sha256": (
                "43cc2cb366acfbe8b1d0737d158dcff71d365b49ea8a9c2e008369179b6988fa"
            ),
            "program_spec_sha256": (
                "e1c4c404ddf3151df91423ec25c0d2a97c12189add35954afffe5341348cac64"
            ),
            "branch": "doctor_static", "label": "32B", "rate_id": "4",
        },
        "predecessor": {
            "cell_id": "qwen2-5-32b__4bpw__codec-control",
            "cell_identity_sha256": (
                "b0ff78a85c371684d936b13994812d4f813e56806dbbccf6845fd3f4af738864"
            ),
            "result_sha256": (
                "24876b1d5fc8d663e469bd807bfbcce926aef3ddbeaf810d77135682a4788d05"
            ),
        },
        "receipt_rel": (
            "reports/condense/doctor_v5_ultra/results/"
            "qwen2-5-32b__4bpw__codec-control/packed_gc_receipt.json"
        ),
        "receipt_file_sha256": (
            "3a2c2c6d8952a240ae28ed3ce0722c8703f704957341546b15aa9a48d402cadb"
        ),
        "receipt_bytes": 7240,
        "receipt_sha256": (
            "dcf6c7f8c18febfc3ef5c02c0972290ed03eb8eea0ca06362d5c21f66e50ba6c"
        ),
        "old_spec_rel": (
            "reports/condense/doctor_v5_ultra/staged_acceleration/pending_runtime/"
            "808d8533d8658222d2f730fd540cc61b023abacac6a6beb131a0153ec3d4c34d/"
            "rollback/1784050520599938000-93160897ec0cc54a/"
            "spec-qwen2-5-32b__4bpw__doctor-static.json"
        ),
        "successor": {
            "cell_id": "qwen2-5-32b__4bpw__doctor-static",
            "cell_identity_sha256": (
                "43cc2cb366acfbe8b1d0737d158dcff71d365b49ea8a9c2e008369179b6988fa"
            ),
            "runtime_spec_path": (
                "reports/condense/doctor_v5_ultra/runtime_specs/"
                "qwen2-5-32b__4bpw__doctor-static.json"
            ),
            "runtime_spec_sha256": (
                "a4064b007bd0df1ef9ba961754628f088824a9f81c3aa8cbcc0a3e25298a0ad2"
            ),
            "runtime_spec_bytes": 12896,
            "program_spec_sha256": (
                "e1c4c404ddf3151df91423ec25c0d2a97c12189add35954afffe5341348cac64"
            ),
        },
    },
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _workspace_path(raw: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve(strict=must_exist)
        resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise GCRuntimeTransitionError(f"path escapes or is missing from workspace: {path}") \
            from exc
    return resolved


def _read_record(path: Path) -> tuple[bytes, dict[str, Any]]:
    path = _workspace_path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise GCRuntimeTransitionError(f"cannot open bound artifact: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
            raise GCRuntimeTransitionError(f"bound artifact is not a small regular file: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk); size += len(chunk)
        after = os.fstat(fd)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if identity(before) != identity(after) or size != after.st_size:
            raise GCRuntimeTransitionError(f"bound artifact changed while reading: {path}")
        raw = b"".join(chunks)
        return raw, {
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": size,
        }
    finally:
        os.close(fd)


def _json_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, artifact = _read_record(path)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GCRuntimeTransitionError(f"bound JSON is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GCRuntimeTransitionError(f"bound JSON root is not an object: {path}")
    return value, artifact


def artifact(path: Path) -> dict[str, Any]:
    """Return a stable, workspace-confined file artifact row."""
    return _read_record(path)[1]


def _runtime_program_payload(spec: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "schema", "inputs", "resources", "resource_admission",
        "program_spec_sha256", "resource_admission_sha256",
    }
    return {key: value for key, value in spec.items() if key not in excluded}


def _validate_program(spec: dict[str, Any], incident: dict[str, Any],
                      *, context: str) -> None:
    consumer = incident["consumer"]
    program = spec.get("program_spec_sha256")
    if program != consumer["program_spec_sha256"] \
            or _hash_value(_runtime_program_payload(spec)) != program:
        raise GCRuntimeTransitionError(f"{context} semantic program hash differs")
    binding = spec.get("campaign_binding")
    expected = {
        "cell_id": consumer["cell_id"],
        "cell_identity_sha256": consumer["cell_identity_sha256"],
        "branch": consumer["branch"], "label": consumer["label"],
        "target_rate_id": consumer["rate_id"],
    }
    if not isinstance(binding, dict) \
            or any(binding.get(key) != value for key, value in expected.items()):
        raise GCRuntimeTransitionError(f"{context} campaign cell identity differs")
    if spec.get("source_deletion_permitted") is not False \
            or spec.get("quality_claims_permitted") is not False:
        raise GCRuntimeTransitionError(f"{context} claim/deletion boundary differs")


def _validate_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, plan_artifact = _json_record(_workspace_path(PLAN_REL))
    if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256 \
            or _hash_value(_without(plan, "plan_sha256")) != EXPECTED_PLAN_SHA256:
        raise GCRuntimeTransitionError("live campaign plan hash differs from the exact campaign")
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise GCRuntimeTransitionError("live campaign plan has no cell matrix")
    by_id = {row.get("cell_id"): row for row in cells if isinstance(row, dict)}
    for incident in _INCIDENTS:
        expected = incident["consumer"]
        row = by_id.get(expected["cell_id"])
        if not isinstance(row, dict) \
                or row.get("cell_identity_sha256") != expected["cell_identity_sha256"] \
                or row.get("branch") != expected["branch"] \
                or row.get("model_label") != expected["label"] \
                or row.get("rate_id") != expected["rate_id"] \
                or row.get("source_deletion_permitted") is not False \
                or row.get("quality_claims_permitted") is not False:
            raise GCRuntimeTransitionError(
                f"live plan consumer differs: {expected['cell_id']}"
            )
    return plan, plan_artifact


def _validate_receipt(incident: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _workspace_path(incident["receipt_rel"])
    receipt, receipt_artifact = _json_record(path)
    if receipt_artifact != {
        "path": str(path), "sha256": incident["receipt_file_sha256"],
        "bytes": incident["receipt_bytes"],
    }:
        raise GCRuntimeTransitionError(
            f"original GC receipt artifact changed: {incident['incident_id']}"
        )
    predecessor = incident["predecessor"]
    if receipt.get("schema") != "hawking.doctor_v5_packed_gc_receipt.v2" \
            or receipt.get("cell_id") != predecessor["cell_id"] \
            or receipt.get("cell_identity_sha256") != predecessor["cell_identity_sha256"] \
            or receipt.get("result_sha256") != predecessor["result_sha256"] \
            or receipt.get("successor") != incident["successor"] \
            or receipt.get("parent_source_deleted") is not False \
            or receipt.get("receipt_sha256") != incident["receipt_sha256"] \
            or _hash_value(_without(receipt, "receipt_sha256")) \
            != incident["receipt_sha256"]:
        raise GCRuntimeTransitionError(
            f"original GC receipt canonical identity changed: {incident['incident_id']}"
        )
    return receipt, receipt_artifact


def _validate_old_spec(incident: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _workspace_path(incident["old_spec_rel"])
    old_spec, old_artifact = _json_record(path)
    successor = incident["successor"]
    if old_artifact != {
        "path": str(path), "sha256": successor["runtime_spec_sha256"],
        "bytes": successor["runtime_spec_bytes"],
    }:
        raise GCRuntimeTransitionError(
            f"archived historical successor spec changed: {incident['incident_id']}"
        )
    _validate_program(old_spec, incident, context="archived historical successor")
    return old_spec, old_artifact


def _pending_snapshot() -> dict[str, dict[str, Any]]:
    state, _ = _json_record(_workspace_path(STATE_REL))
    if state.get("plan_sha256") != EXPECTED_PLAN_SHA256 \
            or state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        raise GCRuntimeTransitionError("live queue state is not bound to the exact campaign")
    rows = state.get("cells")
    if not isinstance(rows, dict):
        raise GCRuntimeTransitionError("live queue state has no cell rows")
    snapshots: dict[str, dict[str, Any]] = {}
    for incident in _INCIDENTS:
        cell_id = incident["consumer"]["cell_id"]
        row = rows.get(cell_id)
        if not isinstance(row, dict) \
                or row.get("status") not in {
                    "pending", "blocked-execution", "blocked-dependency",
                } \
                or row.get("result_sha256") is not None \
                or row.get("execution_receipt_sha256") is not None:
            raise GCRuntimeTransitionError(
                f"runtime transition is not pending-only: {cell_id}"
            )
        result_root = ROOT / "reports/condense/doctor_v5_ultra/results" / cell_id
        if (result_root / "result.json").exists() \
                or (result_root / "execution_receipt.json").exists():
            raise GCRuntimeTransitionError(
                f"completed consumer evidence already exists: {cell_id}"
            )
        snapshots[cell_id] = {
            "status": row["status"], "attempts": row.get("attempts"),
            "row_sha256": _hash_value(row), "result_sha256": None,
            "execution_receipt_sha256": None,
        }
    return snapshots


def _source_artifacts() -> dict[str, dict[str, Any]]:
    return {"wrapper": artifact(WRAPPER_PATH), "module": artifact(MODULE_PATH)}


def _incident_authority(incident: dict[str, Any],
                        snapshot: dict[str, Any]) -> dict[str, Any]:
    _, receipt_artifact = _validate_receipt(incident)
    _, old_artifact = _validate_old_spec(incident)
    return {
        "incident_id": incident["incident_id"],
        "consumer": dict(incident["consumer"]),
        "predecessor": dict(incident["predecessor"]),
        "original_gc_receipt": {
            **receipt_artifact, "receipt_sha256": incident["receipt_sha256"],
        },
        "archived_old_runtime_spec": old_artifact,
        "historical_successor": dict(incident["successor"]),
        "pending_snapshot": dict(snapshot),
        "permitted_delta": "execution_inputs_and_resources_only",
    }


def build_authority() -> dict[str, Any]:
    """Build an authority document after checking the two live incidents."""
    _, plan_artifact = _validate_plan()
    snapshots = _pending_snapshot()
    sources = _source_artifacts()
    transitions = [
        _incident_authority(row, snapshots[row["consumer"]["cell_id"]])
        for row in _INCIDENTS
    ]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "plan": {**plan_artifact, "plan_sha256": EXPECTED_PLAN_SHA256},
        "implementation": sources,
        "transitions": transitions,
        "policy": {
            "pending_only": True, "listed_transitions_only": True,
            "completed_evidence_mutation_permitted": False,
            "source_deletion_permitted": False,
            "quality_claims_permitted": False,
            "future_runtime_file_hash_bound_here": False,
            "future_runtime_must_bind_authority_and_module": True,
        },
    }
    document["authority_sha256"] = _hash_value(document)
    return document


def _atomic_write_once_or_equal(path: Path, document: dict[str, Any]) -> None:
    path = _workspace_path(path, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing, _ = _json_record(path)
        if existing != document:
            raise GCRuntimeTransitionError(f"refusing to replace different authority: {path}")
        return
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def generate_authority(output_path: Path = DEFAULT_AUTHORITY_PATH) -> dict[str, Any]:
    """Generate, atomically write, and revalidate the immutable authority."""
    document = build_authority()
    _atomic_write_once_or_equal(output_path, document)
    validate_authority(output_path)
    return document


def _expected_transition_rows(snapshots: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _incident_authority(row, snapshots[row["consumer"]["cell_id"]])
        for row in _INCIDENTS
    ]


def validate_authority(authority_path: Path = DEFAULT_AUTHORITY_PATH) -> dict[str, Any]:
    """Validate the authority and every immutable historical/source binding."""
    document, _ = _json_record(authority_path)
    if set(document) != {
        "schema", "plan", "implementation", "transitions", "policy",
        "authority_sha256",
    } or document.get("schema") != SCHEMA \
            or document.get("authority_sha256") \
            != _hash_value(_without(document, "authority_sha256")):
        raise GCRuntimeTransitionError("runtime-transition authority envelope/hash is invalid")
    _, live_plan_artifact = _validate_plan()
    if document.get("plan") != {
        **live_plan_artifact, "plan_sha256": EXPECTED_PLAN_SHA256,
    }:
        raise GCRuntimeTransitionError("authority binds a different live campaign plan")
    if document.get("implementation") != _source_artifacts():
        raise GCRuntimeTransitionError("authority wrapper/module source artifacts changed")
    policy = document.get("policy")
    if policy != {
        "pending_only": True, "listed_transitions_only": True,
        "completed_evidence_mutation_permitted": False,
        "source_deletion_permitted": False,
        "quality_claims_permitted": False,
        "future_runtime_file_hash_bound_here": False,
        "future_runtime_must_bind_authority_and_module": True,
    }:
        raise GCRuntimeTransitionError("authority policy boundary differs")
    transitions = document.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != len(_INCIDENTS):
        raise GCRuntimeTransitionError("authority transition allowlist differs")
    snapshots: dict[str, Any] = {}
    for row in transitions:
        if not isinstance(row, dict) or not isinstance(row.get("consumer"), dict) \
                or not isinstance(row.get("pending_snapshot"), dict):
            raise GCRuntimeTransitionError("authority transition row is invalid")
        snapshots[row["consumer"].get("cell_id")] = row["pending_snapshot"]
    if document["transitions"] != _expected_transition_rows(snapshots):
        raise GCRuntimeTransitionError("authority transition evidence differs")
    for row in document["transitions"]:
        snapshot = row["pending_snapshot"]
        if set(snapshot) != {
            "status", "attempts", "row_sha256", "result_sha256",
            "execution_receipt_sha256",
        } or snapshot["status"] not in {
            "pending", "blocked-execution", "blocked-dependency",
        } or snapshot["result_sha256"] is not None \
                or snapshot["execution_receipt_sha256"] is not None \
                or not isinstance(snapshot["row_sha256"], str) \
                or SHA256_RE.fullmatch(snapshot["row_sha256"]) is None:
            raise GCRuntimeTransitionError("authority was not issued pending-only")
    return document


def _authority_input(authority_path: Path) -> dict[str, Any]:
    return {"role": "gc_runtime_transition_authority", **artifact(authority_path)}


def authority_input_row(authority_path: Path = DEFAULT_AUTHORITY_PATH) -> dict[str, Any]:
    """Return the exact runtime-spec input row for a validated authority."""
    validate_authority(authority_path)
    return _authority_input(authority_path)


def module_input_row() -> dict[str, Any]:
    """Return the exact runtime-spec input row for this validator module."""
    return {"role": "gc_runtime_transition_module", **artifact(MODULE_PATH)}


def _input_by_role(spec: dict[str, Any], role: str) -> dict[str, Any]:
    rows = spec.get("inputs")
    matches = [row for row in rows if isinstance(row, dict) and row.get("role") == role] \
        if isinstance(rows, list) else []
    if len(matches) != 1:
        raise GCRuntimeTransitionError(f"consumer has no unique {role} input")
    return matches[0]


def _incident_for_receipt(receipt_path: Path) -> dict[str, Any]:
    resolved = _workspace_path(receipt_path)
    matches = [row for row in _INCIDENTS
               if resolved == _workspace_path(row["receipt_rel"])]
    if len(matches) != 1:
        raise GCRuntimeTransitionError("GC receipt is not one of the two listed transitions")
    return matches[0]


def validate_transition(*, authority_path: Path = DEFAULT_AUTHORITY_PATH,
                        receipt_path: Path, consumer_spec: dict[str, Any]) \
        -> dict[str, Any]:
    """Validate one exact transition and return the narrow hash virtualization.

    The returned historical hash/size may be presented only for the exact live
    successor path after the caller independently confirms that the live file
    still has ``current_runtime``'s identity.
    """
    authority = validate_authority(authority_path)
    incident = _incident_for_receipt(receipt_path)
    incident_rows = [row for row in authority["transitions"]
                     if row["incident_id"] == incident["incident_id"]]
    if len(incident_rows) != 1:
        raise GCRuntimeTransitionError("listed transition is absent from authority")
    _validate_receipt(incident)
    _validate_old_spec(incident)
    if not isinstance(consumer_spec, dict):
        raise GCRuntimeTransitionError("current consumer spec is not an object")
    _validate_program(consumer_spec, incident, context="current successor")

    successor = incident["successor"]
    current_path = _workspace_path(successor["runtime_spec_path"])
    live_spec, current_artifact = _json_record(current_path)
    if live_spec != consumer_spec:
        raise GCRuntimeTransitionError("current successor file differs from loaded consumer")
    if current_artifact["sha256"] == successor["runtime_spec_sha256"] \
            and current_artifact["bytes"] == successor["runtime_spec_bytes"]:
        raise GCRuntimeTransitionError("historical successor has no runtime transition")

    result_root = ROOT / "reports/condense/doctor_v5_ultra/results" \
        / incident["consumer"]["cell_id"]
    if (result_root / "result.json").exists() \
            or (result_root / "execution_receipt.json").exists():
        raise GCRuntimeTransitionError("transition cannot apply to completed consumer evidence")

    expected_sources = authority["implementation"]
    if _input_by_role(consumer_spec, "adapter_source") != {
        "role": "adapter_source", **expected_sources["wrapper"],
    }:
        raise GCRuntimeTransitionError("consumer does not bind the authorized wrapper")
    if _input_by_role(consumer_spec, "gc_runtime_transition_module") != {
        "role": "gc_runtime_transition_module", **expected_sources["module"],
    }:
        raise GCRuntimeTransitionError("consumer does not bind the transition module")
    if _input_by_role(consumer_spec, "gc_runtime_transition_authority") \
            != _authority_input(authority_path):
        raise GCRuntimeTransitionError("consumer does not bind the transition authority")
    return {
        "incident_id": incident["incident_id"],
        "successor_path": str(current_path),
        "historical_runtime": {
            "sha256": successor["runtime_spec_sha256"],
            "bytes": successor["runtime_spec_bytes"],
        },
        "current_runtime": current_artifact,
        "program_spec_sha256": incident["consumer"]["program_spec_sha256"],
        "cell_identity_sha256": incident["consumer"]["cell_identity_sha256"],
        "source_deletion_permitted": False,
        "quality_claims_permitted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, default=DEFAULT_AUTHORITY_PATH)
    validate = sub.add_parser("validate")
    validate.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY_PATH)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            document = generate_authority(args.output)
        else:
            document = validate_authority(args.authority)
        print(json.dumps({
            "status": "pass", "authority_sha256": document["authority_sha256"],
            "transition_count": len(document["transitions"]),
            "source_deletion_permitted": False, "quality_claims_permitted": False,
        }, sort_keys=True))
        return 0
    except (GCRuntimeTransitionError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
