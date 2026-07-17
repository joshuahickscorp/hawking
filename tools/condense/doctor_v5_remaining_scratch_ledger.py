#!/usr/bin/env python3.12
"""Read-only, default-off Doctor V5 remaining-scratch observer.

This module has no queue import and no mutation command.  It reads one frozen
Qwen strand-ladder worker request, derives ``checkpoint.json`` and every
countable artifact path from that request's ``output_root``, and emits a
self-hashed audit receipt on stdout.

Large payloads are never content-hashed.  Existing hashes are syntax-checked
and bound through the exact checkpoint-file hash; open file descriptors are
used only to prove regular-file identity and size without following symlinks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
REQUEST_SCHEMA = "hawking.doctor_v5_strand_ladder_request.v1"
CHECKPOINT_SCHEMA = "hawking.doctor_v5_strand_ladder_checkpoint.v1"
RECEIPT_SCHEMA = "hawking.doctor_v5_remaining_scratch_ledger.v1"
VERSION = "2026-07-14.1"
DISK_RESERVE_BYTES = 50_000_000_000
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")
ORDINAL_UNIT_RE = re.compile(r"(passthrough|encode|attest|decode):(\d{5})")

REQUEST_KEYS = {
    "schema", "request_id", "label", "model_family", "campaign_binding",
    "codec", "source", "parameter_manifest", "execution", "evaluation",
    "doctor_hook", "resources", "output_root", "evidence_policy",
}
SOURCE_KEYS = {
    "census_path", "census_sha256", "model_dir", "shards",
    "source_manifest_sha256",
}
SOURCE_SHARD_KEYS = {"bytes", "name", "ordinal", "path", "sha256"}
CHECKPOINT_KEYS = {
    "schema", "request_sha256", "created_at", "updated_at", "status",
    "plan", "completed_units", "units", "stop_requested",
}
FIXED_PREFIX = ["preflight", "metadata"]
DEFERRED_SUFFIX = ["bundle_manifest", "receipt"]
RESIDENT_SUFFIX = [
    "bundle_manifest", "override_manifest", "baseline_ppl",
    "reconstruction_ppl", "baseline_capability",
    "reconstruction_capability", "ephemeral_cleanup", "receipt",
]


class ScratchLedgerError(RuntimeError):
    """A frozen input is unsafe, ambiguous, or outside the ledger contract."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _lexical_path(raw: str | os.PathLike[str], root: Path, *,
                  require_absolute: bool = True) -> Path:
    text = os.fspath(raw)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise ScratchLedgerError("path is empty or contains NUL")
    candidate = Path(text)
    if require_absolute and not candidate.is_absolute():
        raise ScratchLedgerError(f"path must be absolute: {text}")
    if ".." in candidate.parts:
        raise ScratchLedgerError(f"path traversal is forbidden: {text}")
    lexical = Path(os.path.abspath(text))
    root_lexical = Path(os.path.abspath(root))
    try:
        lexical.relative_to(root_lexical)
    except ValueError as exc:
        raise ScratchLedgerError(f"path escapes workspace: {text}") from exc
    return lexical


def _no_symlink_components(path: Path, root: Path, *, leaf_may_be_missing: bool = False) \
        -> None:
    root = Path(os.path.abspath(root))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ScratchLedgerError(f"path escapes workspace: {path}") from exc
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise ScratchLedgerError(f"workspace root is unavailable: {root}: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ScratchLedgerError(f"workspace root is not a real directory: {root}")
    cursor = root
    for index, component in enumerate(relative.parts):
        cursor /= component
        final = index == len(relative.parts) - 1
        try:
            info = os.lstat(cursor)
        except FileNotFoundError:
            if final and leaf_may_be_missing:
                return
            raise ScratchLedgerError(f"required path is missing: {cursor}")
        except OSError as exc:
            raise ScratchLedgerError(f"cannot inspect path component {cursor}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ScratchLedgerError(f"symlink path component is forbidden: {cursor}")
        if not final and not stat.S_ISDIR(info.st_mode):
            raise ScratchLedgerError(f"non-directory path component: {cursor}")


def _open_stable(path: Path, root: Path, *, directory: bool = False) \
        -> tuple[int, os.stat_result]:
    """Open ``path`` through an O_NOFOLLOW directory-descriptor chain."""
    root = Path(os.path.abspath(root))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ScratchLedgerError(f"path escapes workspace: {path}") from exc
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    opened_directories: list[int] = []
    descriptor: int | None = None
    try:
        root_before = os.lstat(root)
        parent = os.open(root, directory_flags)
        opened_directories.append(parent)
        if _stat_identity(root_before) != _stat_identity(os.fstat(parent)):
            raise ScratchLedgerError("workspace root changed while opening")
        parts = relative.parts
        if not parts:
            descriptor = os.dup(parent)
        else:
            for component in parts[:-1]:
                component_before = os.stat(
                    component, dir_fd=parent, follow_symlinks=False
                )
                if stat.S_ISLNK(component_before.st_mode) \
                        or not stat.S_ISDIR(component_before.st_mode):
                    raise ScratchLedgerError(
                        f"symlink or non-directory path component: {component}"
                    )
                child = os.open(component, directory_flags, dir_fd=parent)
                if _stat_identity(component_before) != _stat_identity(os.fstat(child)):
                    os.close(child)
                    raise ScratchLedgerError(
                        f"path component changed while opening: {component}"
                    )
                opened_directories.append(child)
                parent = child
            leaf = parts[-1]
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            wanted = stat.S_ISDIR(before.st_mode) if directory \
                else stat.S_ISREG(before.st_mode)
            if stat.S_ISLNK(before.st_mode) or not wanted:
                kind = "directory" if directory else "regular file"
                raise ScratchLedgerError(
                    f"path is not a non-symlink {kind}: {path}"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
                | getattr(os, "O_NOFOLLOW", 0)
            if directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(leaf, flags, dir_fd=parent)
            opened = os.fstat(descriptor)
            if _stat_identity(before) != _stat_identity(opened):
                raise ScratchLedgerError(f"path identity changed while opening: {path}")
            return descriptor, opened
        before = os.lstat(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        opened = os.fstat(descriptor)
    except ScratchLedgerError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ScratchLedgerError(f"cannot open stable path {path}: {exc}") from exc
    finally:
        for directory_descriptor in reversed(opened_directories):
            os.close(directory_descriptor)
    assert descriptor is not None
    if _stat_identity(before) != _stat_identity(opened):
        os.close(descriptor)
        raise ScratchLedgerError(f"path identity changed while opening: {path}")
    return descriptor, opened


def _finish_stable(descriptor: int, path: Path, opened: os.stat_result) -> os.stat_result:
    try:
        fd_after = os.fstat(descriptor)
        path_after = os.lstat(path)
    except OSError as exc:
        raise ScratchLedgerError(f"path changed while observing {path}: {exc}") from exc
    if _stat_identity(opened) != _stat_identity(fd_after) \
            or _stat_identity(opened) != _stat_identity(path_after):
        raise ScratchLedgerError(f"path changed while observing: {path}")
    return fd_after


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_stable(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor, opened = _open_stable(path, root)
    try:
        if opened.st_size > MAX_JSON_BYTES:
            raise ScratchLedgerError(f"JSON exceeds {MAX_JSON_BYTES} bytes: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_JSON_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_JSON_BYTES:
                raise ScratchLedgerError(f"JSON exceeds {MAX_JSON_BYTES} bytes: {path}")
        _finish_stable(descriptor, path, opened)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise ScratchLedgerError(f"short JSON read: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=lambda name: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {name}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ScratchLedgerError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScratchLedgerError(f"JSON root is not an object: {path}")
    return value, {
        "path": _relative(path, root),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "identity": {
            "device": opened.st_dev, "inode": opened.st_ino,
            "links": opened.st_nlink, "size": opened.st_size,
            "mtime_ns": opened.st_mtime_ns,
            "ctime_ns": opened.st_ctime_ns,
        },
    }


def _stable_regular_identity(path: Path, root: Path, *, expected_bytes: int,
                             checkpoint_sha256: str) -> dict[str, Any]:
    descriptor, opened = _open_stable(path, root)
    try:
        _finish_stable(descriptor, path, opened)
    finally:
        os.close(descriptor)
    if opened.st_size != expected_bytes:
        raise ScratchLedgerError(
            f"checkpoint byte count differs from regular file size: {path}"
        )
    if opened.st_nlink != 1:
        raise ScratchLedgerError(f"hard-linked or unlinked artifact is forbidden: {path}")
    return {
        "path": _relative(path, root), "bytes": opened.st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "content_rehashed": False,
        "identity": {
            "device": opened.st_dev, "inode": opened.st_ino,
            "links": opened.st_nlink, "size": opened.st_size,
            "mtime_ns": opened.st_mtime_ns,
            "ctime_ns": opened.st_ctime_ns,
        },
    }


def _scan_no_partials(path: Path, root: Path) -> None:
    try:
        present = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ScratchLedgerError(f"cannot inspect artifact directory {path}: {exc}") from exc
    if stat.S_ISLNK(present.st_mode):
        raise ScratchLedgerError(f"symlink artifact directory is forbidden: {path}")
    descriptor, opened = _open_stable(path, root, directory=True)
    try:
        names = os.listdir(descriptor)
        if len(names) != len(set(names)):
            raise ScratchLedgerError(f"directory contains duplicate names: {path}")
        for name in names:
            if ".partial." in name or name.endswith(".partial"):
                raise ScratchLedgerError(f"partial artifact is present: {path / name}")
            try:
                child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise ScratchLedgerError(
                    f"directory entry raced while scanning {path / name}: {exc}"
                ) from exc
            if stat.S_ISLNK(child.st_mode):
                raise ScratchLedgerError(f"symlink artifact is forbidden: {path / name}")
        _finish_stable(descriptor, path, opened)
    finally:
        os.close(descriptor)


def _validate_request(request: dict[str, Any], request_path: Path, root: Path) \
        -> tuple[Path, list[int], int, str]:
    if set(request) != REQUEST_KEYS or request.get("schema") != REQUEST_SCHEMA:
        raise ScratchLedgerError("worker request schema or keys are not frozen")
    output_raw = request.get("output_root")
    if not isinstance(output_raw, str):
        raise ScratchLedgerError("worker output_root is not a string")
    output_root = _lexical_path(output_raw, root)
    _no_symlink_components(output_root, root)
    try:
        output_info = os.lstat(output_root)
    except OSError as exc:
        raise ScratchLedgerError(f"output_root is unavailable: {exc}") from exc
    if not stat.S_ISDIR(output_info.st_mode) or stat.S_ISLNK(output_info.st_mode):
        raise ScratchLedgerError("output_root is not a real directory")
    if request_path != output_root / "request.json":
        raise ScratchLedgerError("request path is not output_root/request.json")

    evaluation = request.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {
            "mode", "retain_dense_reconstruction"} \
            or evaluation.get("mode") not in {"resident", "deferred"} \
            or evaluation.get("retain_dense_reconstruction") is not False:
        raise ScratchLedgerError("unknown or weakened evaluation mode")
    resources = request.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
            "disk_reserve_bytes", "scratch_budget_bytes"}:
        raise ScratchLedgerError("worker resource declaration is not exact")
    if resources.get("disk_reserve_bytes") != DISK_RESERVE_BYTES:
        raise ScratchLedgerError("disk reserve is not exactly 150 decimal GB")
    declared = resources.get("scratch_budget_bytes")
    if not _integer(declared, minimum=1):
        raise ScratchLedgerError("declared scratch is not a positive integer")

    source = request.get("source")
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS \
            or not _valid_sha(source.get("census_sha256")) \
            or not _valid_sha(source.get("source_manifest_sha256")):
        raise ScratchLedgerError("worker source inventory is invalid")
    shards = source.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ScratchLedgerError("worker source shard inventory is empty")
    ordinals: list[int] = []
    names: set[str] = set()
    paths: set[Path] = set()
    for row in shards:
        if not isinstance(row, dict) or set(row) != SOURCE_SHARD_KEYS \
                or not _integer(row.get("ordinal")) \
                or not _integer(row.get("bytes"), minimum=1) \
                or not _valid_sha(row.get("sha256")) \
                or not isinstance(row.get("name"), str) or not row["name"] \
                or not isinstance(row.get("path"), str):
            raise ScratchLedgerError("worker source shard row is invalid")
        source_path = _lexical_path(row["path"], root)
        _no_symlink_components(source_path, root)
        if source_path.name != row["name"]:
            raise ScratchLedgerError("source shard name/path binding differs")
        if row["name"] in names or source_path in paths:
            raise ScratchLedgerError("duplicate source shard name or path")
        names.add(row["name"]); paths.add(source_path); ordinals.append(row["ordinal"])
    if sorted(ordinals) != list(range(len(shards))) or len(ordinals) != len(set(ordinals)):
        raise ScratchLedgerError("source shard ordinals are duplicate or non-contiguous")
    return output_root, sorted(ordinals), declared, evaluation["mode"]


def _validate_plan(plan: Any, ordinals: list[int], mode: str) -> list[str]:
    if not isinstance(plan, list) or any(not isinstance(unit, str) for unit in plan) \
            or len(plan) != len(set(plan)):
        raise ScratchLedgerError("checkpoint plan is not a unique string list")
    suffix = RESIDENT_SUFFIX if mode == "resident" else DEFERRED_SUFFIX
    if plan[:len(FIXED_PREFIX)] != FIXED_PREFIX or plan[-len(suffix):] != suffix:
        raise ScratchLedgerError("checkpoint plan prefix/suffix differs from worker mode")
    middle = plan[len(FIXED_PREFIX):-len(suffix)]
    cursor = 0
    for ordinal in ordinals:
        expected = [
            f"passthrough:{ordinal:05d}", f"encode:{ordinal:05d}",
            f"attest:{ordinal:05d}",
        ]
        if middle[cursor:cursor + 3] != expected:
            raise ScratchLedgerError("checkpoint ordinal phase order is invalid")
        cursor += 3
        decode = f"decode:{ordinal:05d}"
        if cursor < len(middle) and middle[cursor] == decode:
            if mode != "resident":
                raise ScratchLedgerError("deferred evaluation cannot contain decode units")
            cursor += 1
    if cursor != len(middle):
        unknown = middle[cursor] if cursor < len(middle) else "<missing>"
        raise ScratchLedgerError(f"unknown or duplicate checkpoint unit: {unknown}")
    return plan


def _artifact_refs(value: Any, unit: str) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if "sha256" in value or "bytes" in value:
            artifact_shape = set(value) == {"path", "sha256", "bytes"}
            metadata_shape = set(value) == {"name", "sha256", "bytes"}
            if artifact_shape and isinstance(value.get("path"), str) \
                    and _valid_sha(value.get("sha256")) \
                    and _integer(value.get("bytes")):
                yield unit, {"path": value["path"], "sha256": value["sha256"],
                             "bytes": value["bytes"]}
            elif metadata_shape and isinstance(value.get("name"), str) \
                    and value["name"] not in {"", ".", ".."} \
                    and Path(value["name"]).name == value["name"] \
                    and "\x00" not in value["name"] \
                    and _valid_sha(value.get("sha256")) \
                    and _integer(value.get("bytes")):
                # Shared-baseline receipts carry immutable source metadata by
                # basename because the separately bound source identity owns
                # its directory.  Validate that shape, but do not count it as
                # a materialized output-root artifact or require a fabricated
                # absolute path.
                pass
            else:
                raise ScratchLedgerError(
                    f"checkpoint artifact identity syntax is invalid in {unit}"
                )
        for child in value.values():
            yield from _artifact_refs(child, unit)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_refs(child, unit)


def _expected_artifact(evidence: Any, key: str, expected: Path, output_root: Path,
                       unit: str) -> dict[str, Any]:
    row = evidence.get(key) if isinstance(evidence, dict) else None
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"} \
            or not isinstance(row.get("path"), str) \
            or not _valid_sha(row.get("sha256")) \
            or not _integer(row.get("bytes")):
        raise ScratchLedgerError(f"{unit} has no exact {key} artifact identity")
    observed = _lexical_path(row["path"], output_root)
    if observed != expected:
        raise ScratchLedgerError(f"{unit} artifact path differs from its ordinal")
    if ".partial." in observed.name or observed.name.endswith(".partial"):
        raise ScratchLedgerError(f"{unit} points to a partial artifact")
    return {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}


def build_ledger(request_path: Path, *, projected_packed_output_bytes: int,
                 workspace_root: Path = ROOT) -> dict[str, Any]:
    """Build an inert receipt without writing or importing campaign control code."""
    if not _integer(projected_packed_output_bytes):
        raise ScratchLedgerError("projected packed output must be a nonnegative integer")
    root = Path(os.path.abspath(workspace_root))
    request_path = _lexical_path(request_path, root, require_absolute=False)
    request, request_ref = _read_json_stable(request_path, root)
    output_root, ordinals, declared, mode = _validate_request(
        request, request_path, root
    )
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint, checkpoint_ref = _read_json_stable(checkpoint_path, root)
    if set(checkpoint) != CHECKPOINT_KEYS or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ScratchLedgerError("worker checkpoint schema or keys are not frozen")
    if checkpoint.get("request_sha256") != request_ref["sha256"]:
        raise ScratchLedgerError("checkpoint request-file hash binding differs")
    if checkpoint.get("status") not in {"running", "checkpointed-stop", "complete"} \
            or not isinstance(checkpoint.get("stop_requested"), bool):
        raise ScratchLedgerError("checkpoint lifecycle state is invalid")
    plan = _validate_plan(checkpoint.get("plan"), ordinals, mode)
    completed = checkpoint.get("completed_units")
    if not isinstance(completed, list) or completed != plan[:len(completed)] \
            or len(completed) != len(set(completed)):
        raise ScratchLedgerError("checkpoint completion is not an exact plan prefix")
    units = checkpoint.get("units")
    if not isinstance(units, dict) or set(units) != set(completed) \
            or any(not isinstance(units.get(unit), dict) for unit in completed):
        raise ScratchLedgerError("checkpoint unit evidence is incomplete or duplicated")

    _scan_no_partials(output_root / "bundle/shards", root)
    _scan_no_partials(output_root / "evaluation/reconstruction", root)

    # Validate every artifact-shaped identity in every completed unit.  Aliases
    # such as encode.artifact == attest.archive are permitted only when their
    # checkpoint hash and byte count are identical.
    references: dict[Path, dict[str, Any]] = {}
    reference_units: dict[Path, set[str]] = {}
    for unit in completed:
        for owner, row in _artifact_refs(units[unit], unit):
            # Completed evaluation units can bind immutable auxiliary receipts
            # (for example the shared-baseline cache) outside this worker's
            # output root.  They remain confined to and identity-checked inside
            # the workspace, but only exact ordinal artifacts under output_root
            # are eligible for remaining-scratch credit below.
            artifact_path = _lexical_path(row["path"], root)
            if ".partial." in artifact_path.name or artifact_path.name.endswith(".partial"):
                raise ScratchLedgerError(f"checkpoint references a partial artifact: {unit}")
            prior = references.get(artifact_path)
            identity = {"sha256": row["sha256"], "bytes": row["bytes"]}
            if prior is not None and prior != identity:
                raise ScratchLedgerError(
                    f"duplicate artifact path has conflicting identities: {artifact_path}"
                )
            references[artifact_path] = identity
            reference_units.setdefault(artifact_path, set()).add(owner)

    stable_artifacts: dict[Path, dict[str, Any]] = {}
    filesystem_identities: dict[tuple[int, int], Path] = {}
    for artifact_path, row in sorted(references.items(), key=lambda item: str(item[0])):
        observation = _stable_regular_identity(
            artifact_path, root, expected_bytes=row["bytes"],
            checkpoint_sha256=row["sha256"],
        )
        filesystem_identity = (
            observation["identity"]["device"], observation["identity"]["inode"]
        )
        prior_path = filesystem_identities.get(filesystem_identity)
        if prior_path is not None and prior_path != artifact_path:
            raise ScratchLedgerError(
                "distinct artifact paths share a duplicate filesystem identity"
            )
        filesystem_identities[filesystem_identity] = artifact_path
        stable_artifacts[artifact_path] = observation

    completed_set = set(completed)
    durable_materialized = 0
    durable_packed = 0
    rows: list[dict[str, Any]] = []
    packed_identities: set[Path] = set()
    reconstruction_identities: set[Path] = set()
    for ordinal in ordinals:
        encode = f"encode:{ordinal:05d}"
        attest = f"attest:{ordinal:05d}"
        decode = f"decode:{ordinal:05d}"
        encode_done, attest_done, decode_done = (
            encode in completed_set, attest in completed_set, decode in completed_set
        )
        packed_ref: dict[str, Any] | None = None
        reconstruction_ref: dict[str, Any] | None = None
        packed_durable = False
        reconstruction_counted = False
        if encode_done:
            evidence = units[encode]
            if evidence.get("skipped") is not True:
                packed_ref = _expected_artifact(
                    evidence, "artifact",
                    output_root / f"bundle/shards/{ordinal:05d}.strand",
                    output_root, encode,
                )
        if attest_done:
            evidence = units[attest]
            if evidence.get("skipped") is not True:
                archive = _expected_artifact(
                    evidence, "archive",
                    output_root / f"bundle/shards/{ordinal:05d}.strand",
                    output_root, attest,
                )
                if packed_ref is None or archive != packed_ref:
                    raise ScratchLedgerError(
                        f"encode/attest archive identity differs for ordinal {ordinal}"
                    )
                packed_path = _lexical_path(archive["path"], output_root)
                if packed_path in packed_identities:
                    raise ScratchLedgerError("duplicate packed ordinal artifact")
                packed_identities.add(packed_path)
                packed_durable = True
                durable_packed += archive["bytes"]
        if decode_done:
            if mode != "resident" or not encode_done or not attest_done or not packed_durable:
                raise ScratchLedgerError(
                    f"decode {ordinal} lacks completed encode+attest prerequisites"
                )
            evidence = units[decode]
            reconstruction_ref = _expected_artifact(
                evidence, "artifact",
                output_root / f"evaluation/reconstruction/{ordinal:05d}.safetensors",
                output_root, decode,
            )
            reconstruction_path = _lexical_path(reconstruction_ref["path"], output_root)
            if reconstruction_path in reconstruction_identities:
                raise ScratchLedgerError("duplicate reconstruction ordinal artifact")
            reconstruction_identities.add(reconstruction_path)
            reconstruction_counted = True
            durable_materialized += reconstruction_ref["bytes"]
        rows.append({
            "ordinal": ordinal,
            "encode_completed": encode_done, "attest_completed": attest_done,
            "decode_completed": decode_done,
            "packed_archive_durable": packed_durable,
            "reconstruction_counted": reconstruction_counted,
            "packed_bytes": packed_ref["bytes"] if packed_durable and packed_ref else 0,
            "reconstruction_bytes": (
                reconstruction_ref["bytes"] if reconstruction_counted
                and reconstruction_ref else 0
            ),
        })

    if durable_materialized > declared:
        raise ScratchLedgerError(
            "durable reconstruction bytes exceed declared total scratch budget"
        )
    if durable_packed > projected_packed_output_bytes:
        raise ScratchLedgerError(
            "durable packed bytes exceed projected whole packed output"
        )
    remaining_scratch = max(0, declared - durable_materialized)
    projected_remaining_packed = max(
        0, projected_packed_output_bytes - durable_packed
    )
    required_free = (
        DISK_RESERVE_BYTES + remaining_scratch + projected_remaining_packed
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "version": VERSION, "observed_at": _now(),
        "mode": "unbound-default-off-read-only",
        "request": request_ref, "checkpoint": checkpoint_ref,
        "output_root": _relative(output_root, root),
        "evaluation_mode": mode,
        "disk_reserve_bytes": DISK_RESERVE_BYTES,
        "declared_total_scratch_bytes": declared,
        "durable_materialized_bytes": durable_materialized,
        "remaining_scratch_bytes": remaining_scratch,
        "projected_whole_packed_output_bytes": projected_packed_output_bytes,
        "durable_attested_packed_bytes": durable_packed,
        "projected_remaining_packed_output_bytes": projected_remaining_packed,
        "required_free_bytes": required_free,
        "ordinals": rows,
        "artifact_identity_observations": [
            {**stable_artifacts[path],
             "checkpoint_units": sorted(reference_units[path])}
            for path in sorted(stable_artifacts, key=str)
        ],
        "validation_contract": {
            "request_and_checkpoint_stable_no_follow_reads": True,
            "checkpoint_request_file_sha256_matches": True,
            "all_completed_artifact_identities_syntax_valid": True,
            "all_completed_artifacts_regular_and_size_matched": True,
            "artifact_payload_content_rehashed": False,
            "resident_reduction_requires_exact_encode_attest_decode": True,
            "packed_output_projection_accounted_separately": True,
        },
        "isolation": {
            "activation_permitted": False, "queue_imported": False,
            "queue_mutated": False, "request_mutated": False,
            "checkpoint_mutated": False, "results_mutated": False,
            "runtime_defaults_changed": False,
        },
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    errors = validate_receipt(receipt)
    if errors:
        raise ScratchLedgerError("generated receipt is invalid: " + "; ".join(errors))
    return receipt


def validate_receipt(receipt: Any) -> list[str]:
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA \
            or receipt.get("version") != VERSION:
        return ["remaining-scratch receipt schema/version mismatch"]
    errors: list[str] = []
    expected_top = {
        "schema", "version", "observed_at", "mode", "request", "checkpoint",
        "output_root", "evaluation_mode", "disk_reserve_bytes",
        "declared_total_scratch_bytes", "durable_materialized_bytes",
        "remaining_scratch_bytes", "projected_whole_packed_output_bytes",
        "durable_attested_packed_bytes", "projected_remaining_packed_output_bytes",
        "required_free_bytes", "ordinals", "artifact_identity_observations",
        "validation_contract", "isolation", "receipt_sha256",
    }
    if set(receipt) != expected_top:
        errors.append("remaining-scratch receipt top-level keys are not exact")
    if not _valid_sha(receipt.get("receipt_sha256")):
        errors.append("receipt self-hash syntax is invalid")
    else:
        try:
            if receipt["receipt_sha256"] != _hash_value(_without(receipt, "receipt_sha256")):
                errors.append("receipt self-hash mismatch")
        except (TypeError, ValueError):
            errors.append("receipt is not canonically hashable")
    if receipt.get("mode") != "unbound-default-off-read-only":
        errors.append("receipt mode is not inert")
    evaluation_mode = receipt.get("evaluation_mode")
    if evaluation_mode not in {"resident", "deferred"}:
        errors.append("receipt evaluation mode is invalid")
    isolation = receipt.get("isolation")
    expected_isolation = {
        "activation_permitted": False, "queue_imported": False,
        "queue_mutated": False, "request_mutated": False,
        "checkpoint_mutated": False, "results_mutated": False,
        "runtime_defaults_changed": False,
    }
    if isolation != expected_isolation:
        errors.append("receipt isolation proof is weakened")
    contract = receipt.get("validation_contract")
    expected_contract_keys = {
        "request_and_checkpoint_stable_no_follow_reads",
        "checkpoint_request_file_sha256_matches",
        "all_completed_artifact_identities_syntax_valid",
        "all_completed_artifacts_regular_and_size_matched",
        "artifact_payload_content_rehashed",
        "resident_reduction_requires_exact_encode_attest_decode",
        "packed_output_projection_accounted_separately",
    }
    if not isinstance(contract, dict) or set(contract) != expected_contract_keys \
            or contract.get("artifact_payload_content_rehashed") is not False \
            or any(contract.get(key) is not True for key in (
                "request_and_checkpoint_stable_no_follow_reads",
                "checkpoint_request_file_sha256_matches",
                "all_completed_artifact_identities_syntax_valid",
                "all_completed_artifacts_regular_and_size_matched",
                "resident_reduction_requires_exact_encode_attest_decode",
                "packed_output_projection_accounted_separately",
            )):
        errors.append("receipt validation contract is invalid")
    fields = (
        "disk_reserve_bytes", "declared_total_scratch_bytes",
        "durable_materialized_bytes", "remaining_scratch_bytes",
        "projected_whole_packed_output_bytes", "durable_attested_packed_bytes",
        "projected_remaining_packed_output_bytes", "required_free_bytes",
    )
    if any(not _integer(receipt.get(field)) for field in fields):
        errors.append("receipt byte accounting is not nonnegative integer data")
        return errors
    if receipt["disk_reserve_bytes"] != DISK_RESERVE_BYTES:
        errors.append("receipt disk reserve is not exactly 150 decimal GB")
    if receipt["durable_materialized_bytes"] > receipt["declared_total_scratch_bytes"]:
        errors.append("receipt durable scratch exceeds declared total")
    if receipt["durable_attested_packed_bytes"] \
            > receipt["projected_whole_packed_output_bytes"]:
        errors.append("receipt durable packed bytes exceed projection")
    if receipt["remaining_scratch_bytes"] != max(
            0, receipt["declared_total_scratch_bytes"]
            - receipt["durable_materialized_bytes"]):
        errors.append("receipt remaining-scratch equation differs")
    if receipt["projected_remaining_packed_output_bytes"] != max(
            0, receipt["projected_whole_packed_output_bytes"]
            - receipt["durable_attested_packed_bytes"]):
        errors.append("receipt remaining-packed equation differs")
    if receipt["required_free_bytes"] != (
            DISK_RESERVE_BYTES + receipt["remaining_scratch_bytes"]
            + receipt["projected_remaining_packed_output_bytes"]):
        errors.append("receipt required-free equation differs")
    rows = receipt.get("ordinals")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        errors.append("receipt ordinal ledger is invalid")
    else:
        observed = [row.get("ordinal") for row in rows]
        if any(not _integer(value) for value in observed) \
                or observed != sorted(set(observed)):
            errors.append("receipt ordinal ledger is duplicate or unordered")
        durable = 0
        packed = 0
        for row in rows:
            if set(row) != {
                    "ordinal", "encode_completed", "attest_completed",
                    "decode_completed", "packed_archive_durable",
                    "reconstruction_counted", "packed_bytes",
                    "reconstruction_bytes"}:
                errors.append("receipt ordinal row keys are not exact")
                continue
            if any(not isinstance(row.get(field), bool) for field in (
                    "encode_completed", "attest_completed", "decode_completed",
                    "packed_archive_durable", "reconstruction_counted")) \
                    or not _integer(row.get("packed_bytes")) \
                    or not _integer(row.get("reconstruction_bytes")):
                errors.append("receipt ordinal row types are invalid")
                continue
            if row["packed_archive_durable"] and not (
                    row["encode_completed"] and row["attest_completed"]):
                errors.append("receipt counts packed bytes without encode+attest")
            if row["reconstruction_counted"] and not (
                    row["encode_completed"] and row["attest_completed"]
                    and row["decode_completed"]):
                errors.append("receipt counts reconstruction without exact prerequisites")
            if evaluation_mode == "deferred" and (
                    row["decode_completed"] or row["reconstruction_counted"]):
                errors.append("deferred receipt contains reconstruction progress")
            if not row["packed_archive_durable"] and row["packed_bytes"] != 0:
                errors.append("receipt has bytes for a non-durable packed ordinal")
            if not row["reconstruction_counted"] and row["reconstruction_bytes"] != 0:
                errors.append("receipt has bytes for an uncounted reconstruction")
            packed += row["packed_bytes"]
            durable += row["reconstruction_bytes"]
        if packed != receipt["durable_attested_packed_bytes"]:
            errors.append("receipt packed ordinal sum differs")
        if durable != receipt["durable_materialized_bytes"]:
            errors.append("receipt reconstruction ordinal sum differs")
        if evaluation_mode == "deferred" and durable != 0:
            errors.append("deferred receipt reduces declared scratch")

    def valid_identity(value: Any) -> bool:
        return isinstance(value, dict) and set(value) == {
            "device", "inode", "links", "size", "mtime_ns", "ctime_ns"
        } and all(_integer(value.get(field)) for field in value) \
            and value.get("links") == 1

    for field in ("request", "checkpoint"):
        row = receipt.get(field)
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes", "identity"} \
                or not isinstance(row.get("path"), str) or not row.get("path") \
                or not _valid_sha(row.get("sha256")) \
                or not _integer(row.get("bytes"), minimum=1) \
                or not valid_identity(row.get("identity")) \
                or row["identity"].get("size") != row.get("bytes"):
            errors.append(f"receipt {field} binding is invalid")
    artifacts = receipt.get("artifact_identity_observations")
    if not isinstance(artifacts, list):
        errors.append("receipt artifact identity observations are invalid")
    else:
        paths: list[str] = []
        for row in artifacts:
            if not isinstance(row, dict) or set(row) != {
                    "path", "bytes", "checkpoint_sha256", "content_rehashed",
                    "identity", "checkpoint_units"} \
                    or not isinstance(row.get("path"), str) \
                    or not _valid_sha(row.get("checkpoint_sha256")) \
                    or not _integer(row.get("bytes")) \
                    or row.get("content_rehashed") is not False \
                    or not valid_identity(row.get("identity")) \
                    or row["identity"].get("size") != row.get("bytes") \
                    or not isinstance(row.get("checkpoint_units"), list) \
                    or row["checkpoint_units"] != sorted(set(row["checkpoint_units"])):
                errors.append("receipt artifact observation row is invalid")
                continue
            paths.append(row["path"])
        if paths != sorted(set(paths)):
            errors.append("receipt artifact observations are duplicate or unordered")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path,
                        help="frozen strand_ladder/request.json inside the workspace")
    parser.add_argument("--projected-packed-output-bytes", type=int, required=True,
                        help="frozen whole packed-output projection; never inferred")
    args = parser.parse_args(argv)
    try:
        receipt = build_ledger(
            args.request,
            projected_packed_output_bytes=args.projected_packed_output_bytes,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (OSError, TypeError, ValueError, ScratchLedgerError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
