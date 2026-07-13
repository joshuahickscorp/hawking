#!/usr/bin/env python3.12
"""Canonical, interruption-safe bit-floor row storage and completion bindings.

The fitted curve is a shared JSONL updated by several model workers.  A model completion must not
bind directly to that mutable file's whole-file hash: adding a different model would invalidate an
otherwise unchanged result.  Instead, writers update the shared curve under a separate advisory
lock and Studio publishes an immutable, canonical one-row JSONL proof for each model.  Completion
receipts bind both that exact proof file and the same unique row in the shared curve.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pathlib
import re


FLOOR_POINT_SCHEMA = "hawking.bit_floor_point.v2"


def canonical_json(row: dict) -> str:
    """Return the one accepted byte representation for a floor row."""
    if not isinstance(row, dict):
        raise ValueError("floor row must be a JSON object")
    return json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def canonical_row_sha256(row: dict) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_text(path, text: str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_rows(path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"malformed floor JSONL line {line_number}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(row, dict) or not isinstance(row.get("model"), str) \
                    or not row["model"]:
                raise ValueError(f"floor JSONL line {line_number} has no model identity")
            if row["model"] in seen:
                raise ValueError(f"floor JSONL has duplicate model row: {row['model']}")
            seen.add(row["model"])
            rows.append(row)
    return rows


def locked_upsert_floor_row(path, label: str, row: dict) -> dict:
    """Atomically replace one model row while serializing the shared JSONL RMW.

    The lock is on a stable sidecar inode because the data file itself is atomically replaced.
    Malformed or duplicate existing rows fail closed instead of being silently discarded.
    """
    path = pathlib.Path(path)
    if row.get("model") != label:
        raise ValueError(f"floor row model={row.get('model')!r}, expected {label!r}")
    canonical_json(row)  # reject NaN/non-JSON values before taking the lock
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = pathlib.Path(str(path) + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = [existing for existing in _read_rows(path) if existing["model"] != label]
            rows.append(row)
            _atomic_text(path, "".join(canonical_json(item) + "\n" for item in rows))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return row


def unique_model_row(path, label: str) -> dict:
    matches = [row for row in _read_rows(path) if row.get("model") == label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label!r} floor row, found {len(matches)}")
    return matches[0]


def proof_path(root, lane: str, label: str) -> pathlib.Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return pathlib.Path(root) / f"reports/cron/floor_points/{lane}_{safe_label}.jsonl"


def _display_path(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def create_floor_binding(root, lane: str, label: str, curve_path, audit_path) -> dict:
    """Publish the immutable per-model JSONL proof and return completion fields."""
    root = pathlib.Path(root)
    curve_path = pathlib.Path(curve_path)
    if not curve_path.is_absolute():
        curve_path = root / curve_path
    audit_path = pathlib.Path(audit_path)
    if not audit_path.is_absolute():
        audit_path = root / audit_path
    lock_path = pathlib.Path(str(curve_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            row = unique_model_row(curve_path, label)
            _validate_row_source(row, label, audit_path)
            canonical = canonical_json(row)
            point_path = proof_path(root, lane, label)
            _atomic_text(point_path, canonical + "\n")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "floor_curve_jsonl": _display_path(curve_path, root),
        "floor_jsonl": _display_path(point_path, root),
        "floor_jsonl_sha256": sha256_file(point_path),
        "floor_row": row,
        "floor_row_canonical": canonical,
        "floor_row_sha256": canonical_row_sha256(row),
    }


def _resolve(path_value, root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(str(path_value))
    return path if path.is_absolute() else root / path


def _validate_row_source(row: dict, label: str, audit_path: pathlib.Path) -> None:
    if row.get("schema") != FLOOR_POINT_SCHEMA or row.get("model") != label:
        raise ValueError("floor point schema/model mismatch")
    recorded = pathlib.Path(str(row.get("audit_jsonl", "")))
    if not recorded.is_absolute():
        recorded = pathlib.Path.cwd() / recorded
    if recorded.resolve(strict=False) != audit_path.resolve(strict=False):
        raise ValueError("floor point is not bound to the expected audit JSONL")
    if not audit_path.is_file() or row.get("audit_sha256") != sha256_file(audit_path):
        raise ValueError("floor point audit hash does not match current audit bytes")


def validate_floor_binding(
    completion: dict, root, lane: str, label: str, expected_curve, expected_audit,
) -> tuple[bool, list[str]]:
    """Validate exact proof bytes/hash, canonical row/hash, and current curve membership."""
    root = pathlib.Path(root)
    problems: list[str] = []
    expected_curve = _resolve(expected_curve, root)
    expected_audit = _resolve(expected_audit, root)
    expected_proof = proof_path(root, lane, label)
    proof = _resolve(completion.get("floor_jsonl", ""), root)
    curve = _resolve(completion.get("floor_curve_jsonl", ""), root)
    if proof.resolve(strict=False) != expected_proof.resolve(strict=False):
        problems.append("completion floor_jsonl is not the canonical per-model proof path")
    if curve.resolve(strict=False) != expected_curve.resolve(strict=False):
        problems.append("completion floor_curve_jsonl is not the expected shared curve")
    row = completion.get("floor_row")
    try:
        canonical = canonical_json(row)
    except Exception as exc:
        canonical = ""
        problems.append(f"completion floor_row is not canonical JSON: {exc}")
    if completion.get("floor_row_canonical") != canonical:
        problems.append("completion floor_row_canonical does not match floor_row")
    if canonical and completion.get("floor_row_sha256") != hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest():
        problems.append("completion floor_row_sha256 does not match floor_row")
    try:
        raw = proof.read_bytes()
        if completion.get("floor_jsonl_sha256") != sha256_file(proof):
            problems.append("completion floor_jsonl_sha256 does not match proof bytes")
        if raw != (canonical + "\n").encode("utf-8"):
            problems.append("per-model floor JSONL is not the exact canonical row")
    except Exception as exc:
        problems.append(f"per-model floor JSONL missing/unreadable: {type(exc).__name__}: {exc}")
    try:
        _validate_row_source(row, label, expected_audit)
    except Exception as exc:
        problems.append(f"floor row source binding invalid: {exc}")
    try:
        curve_row = unique_model_row(curve, label)
        if canonical_json(curve_row) != canonical:
            problems.append("shared floor curve row differs from canonical completion row")
    except Exception as exc:
        problems.append(f"shared floor curve row invalid: {type(exc).__name__}: {exc}")
    return not problems, problems


def validate_receipt_floor_row(receipt: dict, row: dict) -> tuple[bool, list[str]]:
    """Bind a schema-constrained official receipt to the canonical selected floor point."""
    problems: list[str] = []
    if not isinstance(receipt, dict) or not isinstance(row, dict):
        return False, ["receipt/floor row must be JSON objects"]
    floor_bpw = row.get("floor_bpw")
    expected_bpw = floor_bpw if floor_bpw is not None else row.get("best_measured_bpw")
    expected_config = row.get("winning_config") if floor_bpw is not None \
        else row.get("best_measured_config")
    try:
        if float(receipt.get("effective_bpw")) != float(expected_bpw):
            problems.append("receipt effective_bpw differs from canonical floor row")
    except (TypeError, ValueError):
        problems.append("receipt/floor row effective_bpw is not numeric")
    artifact = str(receipt.get("condensed_artifact", ""))
    if not isinstance(expected_config, str) or not artifact.startswith(f"{expected_config} @ "):
        problems.append("receipt condensed_artifact does not name canonical floor config")
    try:
        expected_hash = canonical_row_sha256(row)
    except Exception:
        expected_hash = None
    if receipt.get("floor_point_sha256") != expected_hash:
        problems.append("receipt floor_point_sha256 differs from canonical floor row")
    return not problems, problems
