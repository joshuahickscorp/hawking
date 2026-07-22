#!/usr/bin/env python3.12
"""Exact, confirmation-gated Phase-2 release of Kimi-K2.6 source weights.

The audit commands are read-only.  ``execute`` is the sole destructive entry
point and requires a literal token derived from the sealed audit.  It removes
only the 64 verified weight snapshot symlinks, their 64 dedicated blobs, and
the descriptor-enumerated contents of the dedicated Xet directory.  It never
uses globs, recursive deletion, path-following removal, or a shared cache.
Every attempted deletion is covered by a private, hash-chained, fsynced
write-ahead PREPARE/COMMIT journal and a durable terminal receipt; interrupted
attempts reconcile without performing any additional deletion.

This module deliberately does not run itself and is not imported by any live
supervisor.  Adding it does not authorize or perform a source release.
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
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

try:
    from tools.condense import kimi_k26_release_cycle as phase1
except ModuleNotFoundError:  # direct script execution
    import kimi_k26_release_cycle as phase1  # type: ignore[no-redef]


INVENTORY_SCHEMA = "hawking.kimi_k26.phase2_release.inventory.v1"
AUDIT_SCHEMA = "hawking.kimi_k26.phase2_release.audit.v1"
BUNDLE_SCHEMA = "hawking.kimi_k26.phase2_release.bundle.v1"
RECEIPT_SCHEMA = "hawking.kimi_k26.phase2_release.receipt.v1"
CONFIRMATION_SCHEMA = "hawking.kimi_k26.phase2_release.confirmation.v1"
JOURNAL_RECORD_SCHEMA = "hawking.kimi_k26.phase2_release.journal_record.v1"
RELEASE_LEASE_NAME = ".kimi-k26-phase2-release.lease"
DOWNLOAD_LEASE_NAME = ".kimi-k26-download-supervisor.lease"
LEGACY_HEAVY_LEASE_NAME = "kimi_k26.heavy.lease"
JOURNAL_PREFIX = ".kimi-k26-phase2-release-"
WEIGHT_RE = re.compile(r"model-(\d{5})-of-000064\.safetensors\Z")
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
CONFIRM_PREFIX = "CONFIRM-KIMI-K26-PHASE2-"
MAX_JSON_BYTES = 64 * 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class Phase2ReleaseError(RuntimeError):
    """A Phase-2 release was blocked or failed closed."""


class Phase2PartialReleaseError(Phase2ReleaseError):
    """A release stopped after durable progress and has a sealed receipt."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = _clone(dict(receipt))


class _SimulatedHardCrash(BaseException):
    """Fake-only fault used to leave a realistic unterminated journal."""


@dataclass(frozen=True)
class LeaseHold:
    path: Path
    descriptor: int
    identity: dict[str, Any]


class AuditProbe(Protocol):
    def inspect(
        self,
        layout: phase1.SessionLayout,
        *,
        release_roots: Sequence[Path],
        queue_roots: Sequence[Path],
        lease_paths: Sequence[Path],
        owned_lease_paths: Sequence[Path],
        mop_root: Path,
        shared_xet: Path,
        repo_root: Path,
    ) -> Mapping[str, Any]: ...


Verifier = Callable[[phase1.SessionLayout], Mapping[str, Any]]
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _fail(message: str) -> None:
    raise Phase2ReleaseError(message)


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _sha(value: Any) -> str:
    return hashlib.sha256(phase1.canonical_json(value)).hexdigest()


def _same_node(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_uid == right.st_uid
        and left.st_nlink == right.st_nlink
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _fail(
            f"{label} fields differ: missing={sorted(expected - set(value))} "
            f"unknown={sorted(set(value) - expected)}"
        )


def _verify_sealed(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        candidate = _clone(dict(value))
        phase1.verify_sealed_document(candidate, label=label)
        return candidate
    except (phase1.ReleaseCycleError, TypeError, ValueError) as exc:
        raise Phase2ReleaseError(f"{label} is invalid: {exc}") from exc


def _safe_parts(value: object, *, label: str) -> tuple[str, ...]:
    try:
        return phase1._relative_parts(value, label=label)  # noqa: SLF001
    except phase1.ReleaseCycleError as exc:
        raise Phase2ReleaseError(str(exc)) from exc


def _relative_to_session(layout: phase1.SessionLayout, path: Path) -> str:
    try:
        relative = path.relative_to(layout.session)
    except ValueError as exc:
        raise Phase2ReleaseError(f"release path escapes session: {path}") from exc
    parts = _safe_parts(PurePosixPath(relative.as_posix()), label="release path")
    return "/".join(parts)


def _hash_fd(descriptor: int, size: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode("ascii"))  # noqa: S324
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            _fail("regular file truncated during exact hashing")
        sha256.update(chunk)
        git.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        _fail("regular file grew during exact hashing")
    return sha256.hexdigest(), git.hexdigest()


def _node_row(layout: phase1.SessionLayout, path: Path, *, category: str) -> dict[str, Any]:
    relative = _relative_to_session(layout, path)
    root_fd = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
    parent_fds: list[int] = []
    descriptor = -1
    try:
        parent_fds, leaf = _open_relative_parent(root_fd, relative)
        parent_fd = parent_fds[-1]
        named_pre = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        kind = (
            "regular" if stat.S_ISREG(named_pre.st_mode)
            else "symlink" if stat.S_ISLNK(named_pre.st_mode)
            else "directory" if stat.S_ISDIR(named_pre.st_mode)
            else "special"
        )
        if kind == "special":
            _fail(f"special release node is forbidden: {path}")
        if int(named_pre.st_uid) != int(os.getuid()):
            _fail(f"release node owner differs from current uid: {path}")
        if int(named_pre.st_nlink) != 1 and kind != "directory":
            _fail(f"release leaf has unsafe hard-link count: {path}")
        digest: str | None = None
        git_digest: str | None = None
        link_target: str | None = None
        if kind == "regular":
            descriptor = os.open(
                leaf,
                os.O_RDONLY | _NOFOLLOW | _CLOEXEC,
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            if not phase1._identity_equal(named_pre, opened):  # noqa: SLF001
                _fail(f"release file changed while opening: {path}")
            digest, git_digest = _hash_fd(descriptor, int(opened.st_size))
            named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not phase1._identity_equal(opened, named_post):  # noqa: SLF001
                _fail(f"release file changed while hashing: {path}")
            named_pre = named_post
        elif kind == "symlink":
            link_target = os.readlink(leaf, dir_fd=parent_fd)
            named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not phase1._identity_equal(named_pre, named_post):  # noqa: SLF001
                _fail(f"release symlink changed while reading: {path}")
        return {
            "relative_path": relative,
            "category": category,
            "type": kind,
            "device": int(named_pre.st_dev),
            "inode": int(named_pre.st_ino),
            "uid": int(named_pre.st_uid),
            "mode": stat.S_IMODE(named_pre.st_mode),
            "hard_links": int(named_pre.st_nlink),
            "logical_bytes": int(named_pre.st_size),
            "allocated_bytes": int(named_pre.st_blocks) * 512,
            "sha256": digest,
            "git_blob_sha1": git_digest,
            "link_target": link_target,
            "depth": len(_safe_parts(relative, label="node depth")),
        }
    except OSError as exc:
        raise Phase2ReleaseError(f"cannot inspect release node {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for parent_fd in reversed(parent_fds):
            os.close(parent_fd)
        os.close(root_fd)


def _directory_boundary(path: Path, *, label: str) -> dict[str, Any]:
    try:
        descriptor = phase1._open_absolute_directory(path)  # noqa: SLF001
    except (OSError, phase1.ReleaseCycleError) as exc:
        raise Phase2ReleaseError(f"cannot anchor {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        return {
            "path": os.fspath(path),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "uid": int(metadata.st_uid),
            "mode": stat.S_IMODE(metadata.st_mode),
        }
    finally:
        os.close(descriptor)


def _validate_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    value = _verify_sealed(manifest, label="Kimi official manifest")
    if value.get("schema") != "hawking.kimi_k26.official_manifest.v1" \
            or value.get("repo") != phase1.KIMI_REPO \
            or value.get("sha") != phase1.KIMI_REVISION:
        _fail("manifest source identity changed")
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != 96:
        _fail("manifest must contain exactly 96 files")
    weights: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths: list[str] = []
    ordinals: set[int] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict) or set(raw) != {"blob_id", "path", "sha256", "size"}:
            _fail(f"manifest row {index} shape changed")
        path = raw.get("path")
        blob_id = raw.get("blob_id")
        digest = raw.get("sha256")
        size = raw.get("size")
        if not isinstance(path, str) or path in seen:
            _fail("manifest paths are invalid or duplicated")
        _safe_parts(path, label="manifest path")
        if not isinstance(blob_id, str) or HEX40_RE.fullmatch(blob_id) is None:
            _fail(f"manifest blob id is invalid: {path}")
        if digest is not None and (
            not isinstance(digest, str) or HEX64_RE.fullmatch(digest) is None
        ):
            _fail(f"manifest SHA-256 is invalid: {path}")
        if type(size) is not int or size < 0:
            _fail(f"manifest size is invalid: {path}")
        seen.add(path)
        paths.append(path)
        row = dict(raw)
        match = WEIGHT_RE.fullmatch(path)
        if match:
            ordinal = int(match.group(1))
            if not 1 <= ordinal <= 64 or ordinal in ordinals or digest is None:
                _fail(f"weight shard sequence/hash is invalid: {path}")
            ordinals.add(ordinal)
            weights.append(row)
        else:
            metadata.append(row)
    if paths != sorted(paths):
        _fail("manifest file paths are not in canonical sorted order")
    if ordinals != set(range(1, 65)) or len(weights) != 64 or len(metadata) != 32:
        _fail("manifest is not exactly 64 weights plus 32 metadata entries")
    if value.get("file_count") != 96 or value.get("weight_shards") != 64:
        _fail("manifest count summary changed")
    return value, sorted(weights, key=lambda row: row["path"]), sorted(
        metadata, key=lambda row: row["path"]
    )


def _verify_evidence_inputs(
    source: Mapping[str, Any], capsule: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_value = _verify_sealed(source, label="Kimi source verification")
    capsule_value = _verify_sealed(capsule, label="Kimi rollback capsule")
    if source_value.get("schema") != "hawking.kimi_k26.release_cycle.source_verification.v1" \
            or source_value.get("status") != "PASS_EXACT_IMMUTABLE_SOURCE" \
            or source_value.get("repo") != phase1.KIMI_REPO \
            or source_value.get("revision") != phase1.KIMI_REVISION \
            or source_value.get("manifest_seal_sha256") != manifest.get("seal_sha256") \
            or source_value.get("file_count") != 96 \
            or source_value.get("weight_shards") != 64 \
            or source_value.get("logical_bytes") != manifest.get("total_bytes") \
            or source_value.get("weight_bytes") != manifest.get("weight_bytes"):
        _fail("source verification is not exact Phase-1 source evidence")
    if capsule_value.get("schema") != "hawking.kimi_k26.release_cycle.rollback_capsule.v1" \
            or capsule_value.get("status") != "PASS_EXACT_PAYLOAD_RESULT_CAPTURE" \
            or capsule_value.get("mop_touched") is not False:
        _fail("rollback capsule verification is not exact Phase-1 evidence")
    return source_value, capsule_value


def _snapshot_entry(
    layout: phase1.SessionLayout, manifest_row: Mapping[str, Any], *, category: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = layout.snapshot.joinpath(*_safe_parts(manifest_row["path"], label="snapshot path"))
    link_row = _node_row(layout, path, category=category + "_SYMLINK")
    if link_row["type"] != "symlink":
        _fail(f"snapshot entry is not a symlink: {manifest_row['path']}")
    target_text = link_row["link_target"]
    if not isinstance(target_text, str) or os.path.isabs(target_text) or "\x00" in target_text:
        _fail(f"snapshot link is unsafe: {manifest_row['path']}")
    target = Path(os.path.normpath(os.path.join(path.parent, target_text)))
    expected_id = manifest_row["sha256"] or manifest_row["blob_id"]
    if target.parent != layout.blobs or target.name != expected_id:
        _fail(f"snapshot entry points outside its exact dedicated blob: {manifest_row['path']}")
    blob_row = _node_row(layout, target, category=category + "_BLOB")
    if blob_row["type"] != "regular" \
            or blob_row["logical_bytes"] != manifest_row["size"]:
        _fail(f"snapshot blob type/size differs: {manifest_row['path']}")
    if manifest_row["sha256"] is not None:
        if blob_row["sha256"] != manifest_row["sha256"]:
            _fail(f"snapshot blob SHA-256 differs: {manifest_row['path']}")
    elif blob_row["git_blob_sha1"] != manifest_row["blob_id"]:
        _fail(f"snapshot Git blob identity differs: {manifest_row['path']}")
    return link_row, blob_row


def _inventory_xet(layout: phase1.SessionLayout) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leaves: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    root_fd = phase1._open_absolute_directory(layout.xet)  # noqa: SLF001

    def walk(descriptor: int, path: Path) -> None:
        for name in sorted(os.listdir(descriptor)):
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                _fail("unsafe Xet directory entry")
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_path = path / name
            if stat.S_ISDIR(named.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
                try:
                    if not phase1._identity_equal(named, os.fstat(child)):  # noqa: SLF001
                        _fail(f"Xet directory changed while opening: {child_path}")
                    walk(child, child_path)
                finally:
                    os.close(child)
                directories.append(_node_row(layout, child_path, category="XET_DIRECTORY"))
            elif stat.S_ISREG(named.st_mode) or stat.S_ISLNK(named.st_mode):
                leaves.append(_node_row(layout, child_path, category="XET_CONTENT"))
            else:
                _fail(f"special Xet entry is forbidden: {child_path}")

    try:
        walk(root_fd, layout.xet)
    finally:
        os.close(root_fd)
    leaves.sort(key=lambda row: row["relative_path"])
    directories.sort(key=lambda row: (-row["depth"], row["relative_path"]))
    return leaves, directories


def _require_exact_blob_directory(
    layout: phase1.SessionLayout, manifest_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Reject every orphan/partial blob; none may become an implicit target."""
    expected = sorted({str(row["sha256"] or row["blob_id"]) for row in manifest_rows})
    descriptor = phase1._open_absolute_directory(layout.blobs)  # noqa: SLF001
    try:
        observed = sorted(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        incomplete = [name for name in extra if name.endswith(".incomplete")]
        _fail(
            "dedicated blob directory is not the exact sealed 96-file source; "
            f"missing={missing[:3]} extra={extra[:3]} "
            f"incomplete_transfer_artifacts={incomplete[:3]}"
        )


def build_exact_inventory(
    layout: phase1.SessionLayout,
    manifest: Mapping[str, Any],
    source_verification: Mapping[str, Any],
    capsule_verification: Mapping[str, Any],
    *,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Build the single authoritative delete/preserve view without deleting."""
    phase1.validate_layout(layout, mop_root=Path(mop_root), shared_xet=Path(shared_xet))
    manifest_value, weights, metadata = _validate_manifest(manifest)
    source, capsule = _verify_evidence_inputs(
        source_verification, capsule_verification, manifest_value
    )
    if capsule.get("session") != os.fspath(layout.session):
        _fail("rollback capsule belongs to another session")
    _require_exact_blob_directory(layout, [*weights, *metadata])
    weight_links: list[dict[str, Any]] = []
    weight_blobs: list[dict[str, Any]] = []
    metadata_links: list[dict[str, Any]] = []
    metadata_blobs: list[dict[str, Any]] = []
    for row in weights:
        link, blob = _snapshot_entry(layout, row, category="WEIGHT")
        weight_links.append(link)
        weight_blobs.append(blob)
    for row in metadata:
        link, blob = _snapshot_entry(layout, row, category="METADATA")
        metadata_links.append(link)
        metadata_blobs.append(blob)
    weight_blob_ids = {(row["device"], row["inode"]) for row in weight_blobs}
    metadata_blob_ids = {(row["device"], row["inode"]) for row in metadata_blobs}
    if len(weight_blob_ids) != 64 or weight_blob_ids & metadata_blob_ids:
        _fail("weight blobs are not 64 unique objects dedicated away from metadata")
    if len(metadata_links) != 32 or len(metadata_blob_ids) != 32:
        _fail("metadata preserve view is not exactly 32 dedicated snapshot/blob entries")
    xet_leaves, xet_directories = _inventory_xet(layout)
    delete_entries = [*weight_links, *weight_blobs, *xet_leaves, *xet_directories]
    retained_entries = [*metadata_links, *metadata_blobs]
    retained_root_paths = (
        layout.session,
        layout.hub,
        layout.model_cache,
        layout.model_cache / "snapshots",
        layout.snapshot,
        layout.blobs,
        layout.xet,
        layout.build,
        layout.tmp,
        layout.hf_home,
        layout.capsule,
        layout.recovery,
        layout.evidence,
    )
    retained_roots = [
        _directory_boundary(path, label=f"retained root {path}")
        for path in retained_root_paths
    ]
    mop_boundary = _directory_boundary(Path(mop_root), label="MOP root")
    shared_boundary = _directory_boundary(Path(shared_xet), label="shared Xet root")
    target_ids = {(row["device"], row["inode"]) for row in delete_entries}
    protected_ids = {
        (row["device"], row["inode"])
        for row in [*retained_roots, mop_boundary, shared_boundary]
    }
    if target_ids & protected_ids:
        _fail("release target aliases a retained/protected root")
    target_paths = [row["relative_path"] for row in delete_entries]
    if len(target_paths) != len(set(target_paths)):
        _fail("release target paths are duplicated")
    document = {
        "schema": INVENTORY_SCHEMA,
        "status": "PASS_EXACT_64_WEIGHT_RELEASE_VIEW",
        "session": os.fspath(layout.session),
        "repo": phase1.KIMI_REPO,
        "revision": phase1.KIMI_REVISION,
        "manifest_seal_sha256": manifest_value["seal_sha256"],
        "source_verification_seal_sha256": source["seal_sha256"],
        "capsule_verification_seal_sha256": capsule["seal_sha256"],
        "manifest_file_count": 96,
        "weight_symlink_count": len(weight_links),
        "weight_blob_count": len(weight_blobs),
        "metadata_symlink_count_retained": len(metadata_links),
        "metadata_blob_count_retained": len(metadata_blob_ids),
        "xet_leaf_count": len(xet_leaves),
        "xet_directory_count": len(xet_directories),
        "delete_entry_count": len(delete_entries),
        "delete_entries": delete_entries,
        "retained_metadata_entries": retained_entries,
        "retained_roots": retained_roots,
        "mop_boundary": mop_boundary,
        "shared_xet_boundary": shared_boundary,
        "target_logical_bytes": sum(row["logical_bytes"] for row in delete_entries),
        "target_allocated_bytes": sum(row["allocated_bytes"] for row in delete_entries),
        "authoritative_delete_views": 1,
        "globs_used": False,
        "recursive_delete_used": False,
        "deletion_performed": False,
    }
    return phase1.seal_document(document)


_INVENTORY_FIELDS = {
    "schema", "status", "session", "repo", "revision", "manifest_seal_sha256",
    "source_verification_seal_sha256", "capsule_verification_seal_sha256",
    "manifest_file_count", "weight_symlink_count", "weight_blob_count",
    "metadata_symlink_count_retained", "metadata_blob_count_retained",
    "xet_leaf_count", "xet_directory_count", "delete_entry_count", "delete_entries",
    "retained_metadata_entries", "retained_roots", "mop_boundary",
    "shared_xet_boundary", "target_logical_bytes", "target_allocated_bytes",
    "authoritative_delete_views", "globs_used", "recursive_delete_used",
    "deletion_performed", "seal_sha256",
}


def verify_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _verify_sealed(value, label="Phase-2 exact inventory")
    _exact_keys(inventory, _INVENTORY_FIELDS, "Phase-2 exact inventory")
    entries = inventory.get("delete_entries")
    retained = inventory.get("retained_metadata_entries")
    if inventory.get("schema") != INVENTORY_SCHEMA \
            or inventory.get("status") != "PASS_EXACT_64_WEIGHT_RELEASE_VIEW" \
            or inventory.get("repo") != phase1.KIMI_REPO \
            or inventory.get("revision") != phase1.KIMI_REVISION \
            or inventory.get("weight_symlink_count") != 64 \
            or inventory.get("weight_blob_count") != 64 \
            or inventory.get("metadata_symlink_count_retained") != 32 \
            or inventory.get("metadata_blob_count_retained") != 32 \
            or inventory.get("authoritative_delete_views") != 1 \
            or inventory.get("globs_used") is not False \
            or inventory.get("recursive_delete_used") is not False \
            or inventory.get("deletion_performed") is not False:
        _fail("Phase-2 inventory boundary/counts changed")
    if not isinstance(entries, list) or not all(isinstance(row, dict) for row in entries) \
            or inventory.get("delete_entry_count") != len(entries):
        _fail("Phase-2 delete entries are malformed")
    if not isinstance(retained, list) or not all(isinstance(row, dict) for row in retained):
        _fail("Phase-2 retained metadata entries are malformed")
    if inventory.get("target_logical_bytes") != sum(row.get("logical_bytes", -1) for row in entries) \
            or inventory.get("target_allocated_bytes") != sum(
                row.get("allocated_bytes", -1) for row in entries
            ):
        _fail("Phase-2 target byte accounting changed")
    paths = [row.get("relative_path") for row in entries]
    if any(not isinstance(path, str) for path in paths) or len(paths) != len(set(paths)):
        _fail("Phase-2 target paths are invalid or duplicated")
    return inventory


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LC_ALL": "C",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )


def _check_result(result: subprocess.CompletedProcess[str], label: str) -> str:
    if result.returncode != 0:
        _fail(f"{label} exited {result.returncode}")
    return result.stdout or ""


def _audit_readers(roots: Sequence[Path], runner: Runner) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    failures: list[str] = []
    for root in roots:
        result = runner(("/usr/sbin/lsof", "-nP", "-F0puafn", "+D", os.fspath(root)))
        if result.returncode not in {0, 1}:
            failures.append(f"LSOF_EXIT_{result.returncode}:{root}")
            continue
        pids = {
            int(match)
            for match in re.findall(r"(?:^|\n)p(\d+)\x00", result.stdout or "")
            if int(match) != os.getpid()
        }
        if pids:
            matches.append({"root": os.fspath(root), "pids": sorted(pids)})
    return {
        "status": "PASS" if not matches and not failures else "BLOCKED",
        "matches": matches,
        "failures": failures,
    }


def _audit_processes(layout: phase1.SessionLayout, runner: Runner) -> dict[str, Any]:
    result = runner(("/bin/ps", "-axo", "pid=,ppid=,command="))
    if result.returncode != 0:
        return {"status": "BLOCKED", "matches": [], "failures": [f"PS_EXIT_{result.returncode}"]}
    matches: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or not fields[0].isdigit() or int(fields[0]) == os.getpid():
            continue
        try:
            tokens = shlex.split(fields[2])
        except ValueError:
            continue
        token_values = [
            token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
            for token in tokens
        ]
        session_text = os.fspath(layout.session)
        session_bound = any(
            token == session_text or token.startswith(session_text + os.sep)
            for token in token_values
        )
        cache_bound = False
        if "--cache-dir" in tokens:
            index = tokens.index("--cache-dir")
            cache_bound = tokens[index + 1:index + 2] == [os.fspath(layout.hub)]
        revision_bound = False
        if "--revision" in tokens:
            index = tokens.index("--revision")
            revision_bound = tokens[index + 1:index + 2] == [phase1.KIMI_REVISION]
        is_download = "download" in tokens and phase1.KIMI_REPO in tokens
        source_bound = (
            phase1.KIMI_REPO in token_values
            and phase1.KIMI_REVISION in token_values
        )
        if session_bound or (cache_bound and revision_bound and is_download) or source_bound:
            matches.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "argv_sha256": _sha(tokens),
                    "session_bound": session_bound,
                    "source_bound": source_bound,
                }
            )
    return {"status": "PASS" if not matches else "BLOCKED", "matches": matches, "failures": []}


def _audit_launchd(layout: phase1.SessionLayout, runner: Runner) -> dict[str, Any]:
    result = runner(("/bin/launchctl", "print", f"gui/{os.getuid()}"))
    if result.returncode != 0:
        return {
            "status": "BLOCKED",
            "matching_configuration": False,
            "output_sha256": _sha(result.stdout or ""),
            "failures": [f"LAUNCHCTL_EXIT_{result.returncode}"],
        }
    output = result.stdout or ""
    identifiers = (
        os.fspath(layout.session),
        os.fspath(layout.hub),
        phase1.KIMI_REPO,
        phase1.KIMI_REVISION,
    )
    matched = any(identifier in output for identifier in identifiers)
    matching_lines = sorted(
        line.strip()
        for line in output.splitlines()
        if any(identifier in line for identifier in identifiers)
    )
    return {
        "status": "BLOCKED" if matched else "PASS",
        "matching_configuration": matched,
        # Bind only relevant configuration so unrelated launchd churn cannot
        # invalidate the just-confirmed audit between audit and execution.
        "output_sha256": _sha(matching_lines),
        "failures": [],
    }


def _audit_queues(queue_roots: Sequence[Path]) -> dict[str, Any]:
    pending: list[str] = []
    failures: list[str] = []
    for path in queue_roots:
        try:
            named = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
            failures.append(f"UNSAFE_QUEUE_ROOT:{path}")
            continue
        try:
            descriptor = phase1._open_absolute_directory(path)  # noqa: SLF001
        except (OSError, phase1.ReleaseCycleError):
            failures.append(f"UNOPENABLE_QUEUE_ROOT:{path}")
            continue
        try:
            pending.extend(os.fspath(path / name) for name in sorted(os.listdir(descriptor)))
        finally:
            os.close(descriptor)
    return {
        "status": "PASS" if not pending and not failures else "BLOCKED",
        "pending_entries": pending,
        "failures": failures,
    }


def _open_existing_lease(path: Path) -> int | None:
    try:
        parent_fd = phase1._open_absolute_directory(path.parent)  # noqa: SLF001
    except FileNotFoundError:
        return None
    try:
        try:
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(named.st_mode) \
                or named.st_uid != os.getuid() \
                or named.st_nlink != 1:
            _fail(f"lease node is unsafe: {path}")
        descriptor = os.open(
            path.name, os.O_RDWR | _NOFOLLOW | _CLOEXEC, dir_fd=parent_fd
        )
        opened = os.fstat(descriptor)
        if not phase1._identity_equal(named, opened):  # noqa: SLF001
            os.close(descriptor)
            _fail(f"lease changed while opening: {path}")
        return descriptor
    finally:
        os.close(parent_fd)


def _audit_leases(paths: Sequence[Path], owned: Sequence[Path]) -> dict[str, Any]:
    owned_set = {os.fspath(path) for path in owned}
    conflicts: list[str] = []
    failures: list[str] = []
    for path in paths:
        if os.fspath(path) in owned_set:
            continue
        try:
            descriptor = _open_existing_lease(path)
            if descriptor is None:
                continue
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    conflicts.append(os.fspath(path))
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        except (OSError, Phase2ReleaseError) as exc:
            failures.append(f"{path}:{type(exc).__name__}")
    return {
        "status": "PASS" if not conflicts and not failures else "BLOCKED",
        "checked_paths": [os.fspath(path) for path in paths],
        "conflicts": conflicts,
        "failures": failures,
    }


def _audit_git_push(repo_root: Path, runner: Runner) -> dict[str, Any]:
    prefix = ("/usr/bin/git", "-C", os.fspath(repo_root))
    failures: list[str] = []
    try:
        status = _check_result(
            runner((*prefix, "status", "--porcelain=v1", "--untracked-files=all")),
            "git status",
        )
        head = _check_result(runner((*prefix, "rev-parse", "HEAD")), "git HEAD").strip()
        branch = _check_result(runner((*prefix, "symbolic-ref", "--short", "HEAD")), "git branch").strip()
        upstream = _check_result(
            runner((*prefix, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")),
            "git upstream",
        ).strip()
        remote, remote_branch = upstream.split("/", 1)
        remote_output = _check_result(
            runner((*prefix, "ls-remote", "--heads", "--exit-code", remote, remote_branch)),
            "git remote head",
        )
        remote_fields = remote_output.strip().split()
        remote_head = remote_fields[0] if len(remote_fields) == 2 else ""
    except (Phase2ReleaseError, ValueError) as exc:
        failures.append(str(exc))
        status = "UNKNOWN"
        head = branch = upstream = remote_head = ""
    clean = status == ""
    pushed = HEX40_RE.fullmatch(head) is not None and head == remote_head
    return {
        "status": "PASS" if clean and pushed and not failures else "BLOCKED",
        "head": head,
        "branch": branch,
        "upstream": upstream,
        "remote_head": remote_head,
        "worktree_clean": clean,
        "head_pushed_exactly": pushed,
        "failures": failures,
    }


class SystemAuditProbe:
    """Read-only production probes; tests inject a fake implementation."""

    def __init__(self, runner: Runner = _default_runner) -> None:
        self.runner = runner

    def inspect(
        self,
        layout: phase1.SessionLayout,
        *,
        release_roots: Sequence[Path],
        queue_roots: Sequence[Path],
        lease_paths: Sequence[Path],
        owned_lease_paths: Sequence[Path],
        mop_root: Path,
        shared_xet: Path,
        repo_root: Path,
    ) -> Mapping[str, Any]:
        return {
            "readers": _audit_readers(release_roots, self.runner),
            "processes": _audit_processes(layout, self.runner),
            "launchd": _audit_launchd(layout, self.runner),
            "queues": _audit_queues(queue_roots),
            "leases": _audit_leases(lease_paths, owned_lease_paths),
            "git_push": _audit_git_push(repo_root, self.runner),
            "mop": {"status": "PASS", "boundary": _directory_boundary(mop_root, label="MOP root")},
            "shared_xet": {
                "status": "PASS",
                "boundary": _directory_boundary(shared_xet, label="shared Xet root"),
            },
        }


_CHECK_NAMES = {
    "readers", "processes", "launchd", "queues", "leases", "git_push", "mop",
    "shared_xet",
}


def _validate_checks(checks: Mapping[str, Any]) -> dict[str, Any]:
    value = _clone(dict(checks))
    _exact_keys(value, _CHECK_NAMES, "Phase-2 system checks")
    for name, check in value.items():
        if not isinstance(check, dict) or check.get("status") not in {"PASS", "BLOCKED"}:
            _fail(f"Phase-2 system check is malformed: {name}")
    return value


def default_queue_roots(layout: phase1.SessionLayout) -> tuple[Path, ...]:
    return (
        phase1.LEGACY_RUNTIME_ROOT / "queue",
        phase1.LEGACY_RUNTIME_ROOT / "outbox",
        layout.session / "queue",
        layout.session / "outbox",
    )


def default_lease_paths(layout: phase1.SessionLayout) -> tuple[Path, ...]:
    return (
        layout.evidence / RELEASE_LEASE_NAME,
        layout.evidence / DOWNLOAD_LEASE_NAME,
        phase1.LEGACY_RUNTIME_ROOT / LEGACY_HEAVY_LEASE_NAME,
    )


def build_release_audit(
    layout: phase1.SessionLayout,
    inventory: Mapping[str, Any],
    *,
    probe: AuditProbe,
    queue_roots: Sequence[Path] | None = None,
    lease_paths: Sequence[Path] | None = None,
    owned_lease_paths: Sequence[Path] = (),
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
    repo_root: Path = phase1.REPO_ROOT,
) -> dict[str, Any]:
    verified = verify_inventory(inventory)
    if verified["session"] != os.fspath(layout.session):
        _fail("inventory belongs to another Kimi session")
    queues = tuple(queue_roots) if queue_roots is not None else default_queue_roots(layout)
    leases = tuple(lease_paths) if lease_paths is not None else default_lease_paths(layout)
    checks = _validate_checks(
        probe.inspect(
            layout,
            release_roots=(layout.snapshot, layout.blobs, layout.xet),
            queue_roots=queues,
            lease_paths=leases,
            owned_lease_paths=owned_lease_paths,
            mop_root=Path(mop_root),
            shared_xet=Path(shared_xet),
            repo_root=Path(repo_root),
        )
    )
    if checks["mop"].get("boundary") != verified["mop_boundary"]:
        checks["mop"]["status"] = "BLOCKED"
        checks["mop"]["boundary_mismatch"] = True
    if checks["shared_xet"].get("boundary") != verified["shared_xet_boundary"]:
        checks["shared_xet"]["status"] = "BLOCKED"
        checks["shared_xet"]["boundary_mismatch"] = True
    blockers = sorted(name.upper() for name, check in checks.items() if check["status"] != "PASS")
    body = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS_CONFIRMATION_REQUIRED" if not blockers else "BLOCKED",
        "session": os.fspath(layout.session),
        "source_verification_seal_sha256": verified["source_verification_seal_sha256"],
        "capsule_verification_seal_sha256": verified["capsule_verification_seal_sha256"],
        "inventory_seal_sha256": verified["seal_sha256"],
        "target_allocated_bytes": verified["target_allocated_bytes"],
        "checks": checks,
        "blockers": blockers,
        "confirmation_required": True,
        "deletion_authorized_without_confirmation": False,
        "deletion_performed": False,
    }
    return phase1.seal_document(body)


def derive_confirmation_token(audit: Mapping[str, Any]) -> str:
    value = _verify_sealed(audit, label="Phase-2 release audit")
    if value.get("schema") != AUDIT_SCHEMA or value.get("status") != "PASS_CONFIRMATION_REQUIRED" \
            or value.get("blockers") != []:
        _fail("confirmation token is unavailable for a blocked audit")
    digest = _sha(
        {
            "schema": CONFIRMATION_SCHEMA,
            "audit_seal_sha256": value["seal_sha256"],
            "session": value["session"],
            "inventory_seal_sha256": value["inventory_seal_sha256"],
            "action": "UNLINK_EXACT_64_WEIGHT_LINKS_BLOBS_AND_DEDICATED_XET_CONTENTS",
        }
    )
    return CONFIRM_PREFIX + digest


def build_release_bundle(
    layout: phase1.SessionLayout,
    manifest: Mapping[str, Any],
    source_verification: Mapping[str, Any],
    capsule_verification: Mapping[str, Any],
    *,
    probe: AuditProbe,
    queue_roots: Sequence[Path] | None = None,
    lease_paths: Sequence[Path] | None = None,
    owned_lease_paths: Sequence[Path] = (),
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
    repo_root: Path = phase1.REPO_ROOT,
) -> dict[str, Any]:
    inventory = build_exact_inventory(
        layout,
        manifest,
        source_verification,
        capsule_verification,
        mop_root=mop_root,
        shared_xet=shared_xet,
    )
    audit = build_release_audit(
        layout,
        inventory,
        probe=probe,
        queue_roots=queue_roots,
        lease_paths=lease_paths,
        owned_lease_paths=owned_lease_paths,
        mop_root=mop_root,
        shared_xet=shared_xet,
        repo_root=repo_root,
    )
    body = {
        "schema": BUNDLE_SCHEMA,
        "status": audit["status"],
        "session": os.fspath(layout.session),
        "source_verification": _clone(dict(source_verification)),
        "capsule_verification": _clone(dict(capsule_verification)),
        "inventory": inventory,
        "audit": audit,
        "source_verification_seal_sha256": inventory["source_verification_seal_sha256"],
        "capsule_verification_seal_sha256": inventory["capsule_verification_seal_sha256"],
        "inventory_seal_sha256": inventory["seal_sha256"],
        "audit_seal_sha256": audit["seal_sha256"],
        "confirmation_token": None if audit["status"] == "BLOCKED" else derive_confirmation_token(audit),
        "deletion_performed": False,
    }
    return phase1.seal_document(body)


_BUNDLE_FIELDS = {
    "schema", "status", "session", "source_verification", "capsule_verification",
    "inventory", "audit", "source_verification_seal_sha256",
    "capsule_verification_seal_sha256", "inventory_seal_sha256", "audit_seal_sha256",
    "confirmation_token", "deletion_performed", "seal_sha256",
}


def verify_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _verify_sealed(value, label="Phase-2 release bundle")
    _exact_keys(bundle, _BUNDLE_FIELDS, "Phase-2 release bundle")
    source = _verify_sealed(bundle.get("source_verification", {}), label="bundled source evidence")
    capsule = _verify_sealed(bundle.get("capsule_verification", {}), label="bundled capsule evidence")
    inventory = verify_inventory(bundle.get("inventory", {}))
    audit = _verify_sealed(bundle.get("audit", {}), label="bundled release audit")
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("inventory_seal_sha256") != inventory["seal_sha256"]:
        _fail("bundled audit/inventory binding changed")
    expected_status = audit.get("status")
    expected_token = None if expected_status == "BLOCKED" else derive_confirmation_token(audit)
    if bundle.get("schema") != BUNDLE_SCHEMA \
            or bundle.get("status") != expected_status \
            or bundle.get("session") != inventory["session"] \
            or bundle.get("source_verification_seal_sha256") != source["seal_sha256"] \
            or bundle.get("capsule_verification_seal_sha256") != capsule["seal_sha256"] \
            or bundle.get("inventory_seal_sha256") != inventory["seal_sha256"] \
            or bundle.get("audit_seal_sha256") != audit["seal_sha256"] \
            or bundle.get("confirmation_token") != expected_token \
            or inventory["source_verification_seal_sha256"] != source["seal_sha256"] \
            or inventory["capsule_verification_seal_sha256"] != capsule["seal_sha256"] \
            or audit.get("source_verification_seal_sha256") != source["seal_sha256"] \
            or audit.get("capsule_verification_seal_sha256") != capsule["seal_sha256"] \
            or bundle.get("deletion_performed") is not False:
        _fail("Phase-2 bundle seal bindings changed")
    return bundle


def _create_or_open_private_lease(path: Path) -> int:
    parent = path.parent
    descriptor = phase1._open_absolute_directory(parent)  # noqa: SLF001
    try:
        fd = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | _NOFOLLOW | _CLOEXEC,
            0o600,
            dir_fd=descriptor,
        )
        metadata = os.fstat(fd)
        named = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) \
                or metadata.st_uid != os.getuid() \
                or metadata.st_nlink != 1 \
                or stat.S_IMODE(metadata.st_mode) != 0o600 \
                or not phase1._identity_equal(metadata, named):  # noqa: SLF001
            os.close(fd)
            _fail(f"release lease is unsafe: {path}")
        os.fsync(descriptor)
        return fd
    finally:
        os.close(descriptor)


def _lease_identity(path: Path, descriptor: int) -> dict[str, Any]:
    parent_fd = phase1._open_absolute_directory(path.parent)  # noqa: SLF001
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if not phase1._identity_equal(named, opened):  # noqa: SLF001
            _fail(f"held release lease identity changed: {path}")
        if not stat.S_ISREG(opened.st_mode) \
                or opened.st_uid != os.getuid() \
                or opened.st_nlink != 1 \
                or stat.S_IMODE(opened.st_mode) != 0o600:
            _fail(f"held release lease metadata changed: {path}")
        return {
            "path": os.fspath(path),
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "uid": int(opened.st_uid),
            "mode": stat.S_IMODE(opened.st_mode),
            "hard_links": int(opened.st_nlink),
        }
    finally:
        os.close(parent_fd)


def _verify_lease_holds(holds: Sequence[LeaseHold]) -> list[dict[str, Any]]:
    rows = [_lease_identity(hold.path, hold.descriptor) for hold in holds]
    if rows != [hold.identity for hold in holds]:
        _fail("exclusive release lease identities changed")
    return rows


@contextlib.contextmanager
def _exclusive_release_leases(
    layout: phase1.SessionLayout, lease_paths: Sequence[Path]
) -> Iterator[tuple[LeaseHold, ...]]:
    required = (
        layout.evidence / RELEASE_LEASE_NAME,
        layout.evidence / DOWNLOAD_LEASE_NAME,
    )
    paths = tuple(dict.fromkeys([*required, *lease_paths]))
    holds: list[LeaseHold] = []
    try:
        for path in paths:
            if path.parent == phase1.LEGACY_RUNTIME_ROOT and not path.parent.exists():
                continue
            fd = _create_or_open_private_lease(path)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise Phase2ReleaseError(f"release lease is already held: {path}") from exc
            holds.append(LeaseHold(path=path, descriptor=fd, identity=_lease_identity(path, fd)))
        yield tuple(holds)
    finally:
        for hold in reversed(holds):
            with contextlib.suppress(OSError):
                fcntl.flock(hold.descriptor, fcntl.LOCK_UN)
            os.close(hold.descriptor)


def _open_relative_parent(root_fd: int, relative: str) -> tuple[list[int], str]:
    parts = _safe_parts(relative, label="sealed release target")
    fds = [os.dup(root_fd)]
    try:
        for component in parts[:-1]:
            named = os.stat(component, dir_fd=fds[-1], follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
                _fail(f"release parent is not a no-follow directory: {relative}")
            child = os.open(
                component,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                dir_fd=fds[-1],
            )
            if not phase1._identity_equal(named, os.fstat(child)):  # noqa: SLF001
                os.close(child)
                _fail(f"release parent changed while opening: {relative}")
            fds.append(child)
        return fds, parts[-1]
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _metadata_matches(row: Mapping[str, Any], metadata: os.stat_result) -> bool:
    kind = row.get("type")
    expected_kind = (
        stat.S_ISREG(metadata.st_mode) if kind == "regular"
        else stat.S_ISLNK(metadata.st_mode) if kind == "symlink"
        else stat.S_ISDIR(metadata.st_mode) if kind == "directory"
        else False
    )
    stable = (
        ("device", int(metadata.st_dev)),
        ("inode", int(metadata.st_ino)),
        ("uid", int(metadata.st_uid)),
        ("mode", stat.S_IMODE(metadata.st_mode)),
    )
    if not expected_kind or not all(row.get(key) == value for key, value in stable):
        return False
    # Directory size/link/block accounting changes legitimately as the already
    # inventoried children are removed.  Identity, owner, mode, and emptiness
    # are the authoritative rmdir predicates.  Leaf metadata remains exact.
    if kind == "directory":
        return True
    return all(
        row.get(key) == value
        for key, value in (
            ("hard_links", int(metadata.st_nlink)),
            ("logical_bytes", int(metadata.st_size)),
            ("allocated_bytes", int(metadata.st_blocks) * 512),
        )
    )


def _revalidate_leaf(parent_fd: int, leaf: str, row: Mapping[str, Any]) -> None:
    named_pre = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    if not _metadata_matches(row, named_pre):
        _fail(f"release target metadata changed: {row.get('relative_path')}")
    if row["type"] == "regular":
        descriptor = os.open(leaf, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=parent_fd)
        try:
            opened = os.fstat(descriptor)
            if not _metadata_matches(row, opened) \
                    or not phase1._identity_equal(named_pre, opened):  # noqa: SLF001
                _fail(f"release file descriptor identity changed: {row['relative_path']}")
            digest, git_digest = _hash_fd(descriptor, int(opened.st_size))
            if digest != row.get("sha256") or git_digest != row.get("git_blob_sha1"):
                _fail(f"release file hash changed: {row['relative_path']}")
            named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not _metadata_matches(row, named_post) \
                    or not phase1._identity_equal(opened, named_post):  # noqa: SLF001
                _fail(f"release file changed immediately before unlink: {row['relative_path']}")
        finally:
            os.close(descriptor)
    elif row["type"] == "symlink":
        target = os.readlink(leaf, dir_fd=parent_fd)
        named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if target != row.get("link_target") or not _metadata_matches(row, named_post):
            _fail(f"release symlink changed immediately before unlink: {row['relative_path']}")
    else:
        _fail("directory passed to leaf unlink verifier")


def _unlink_exact(root_fd: int, row: Mapping[str, Any]) -> dict[str, Any]:
    fds, leaf = _open_relative_parent(root_fd, str(row["relative_path"]))
    try:
        parent_fd = fds[-1]
        _revalidate_leaf(parent_fd, leaf, row)
        os.unlink(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail(f"release target remained after exact unlink: {row['relative_path']}")
        return {
            "relative_path": row["relative_path"],
            "category": row["category"],
            "type": row["type"],
            "device": row["device"],
            "inode": row["inode"],
            "logical_bytes": row["logical_bytes"],
            "allocated_bytes": row["allocated_bytes"],
            "sha256": row["sha256"],
        }
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _rmdir_exact(root_fd: int, row: Mapping[str, Any]) -> dict[str, Any]:
    fds, leaf = _open_relative_parent(root_fd, str(row["relative_path"]))
    child_fd = -1
    try:
        parent_fd = fds[-1]
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not _metadata_matches(row, named) or not stat.S_ISDIR(named.st_mode):
            _fail(f"Xet directory metadata changed before release: {row['relative_path']}")
        child_fd = os.open(
            leaf,
            os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
            dir_fd=parent_fd,
        )
        if not phase1._identity_equal(named, os.fstat(child_fd)):  # noqa: SLF001
            _fail(f"Xet directory changed while opening: {row['relative_path']}")
        if os.listdir(child_fd):
            _fail(f"Xet directory is not empty before exact rmdir: {row['relative_path']}")
        os.close(child_fd)
        child_fd = -1
        named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not _metadata_matches(row, named_post):
            _fail(f"Xet directory changed immediately before rmdir: {row['relative_path']}")
        os.rmdir(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return {
            "relative_path": row["relative_path"],
            "category": row["category"],
            "type": "directory",
            "device": row["device"],
            "inode": row["inode"],
            "logical_bytes": row["logical_bytes"],
            "allocated_bytes": row["allocated_bytes"],
            "sha256": None,
        }
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        for fd in reversed(fds):
            os.close(fd)


def _verify_preserved_node(root_fd: int, row: Mapping[str, Any]) -> None:
    fds, leaf = _open_relative_parent(root_fd, str(row["relative_path"]))
    try:
        _revalidate_leaf(fds[-1], leaf, row)
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _assert_relative_absent(root_fd: int, relative: str) -> None:
    parts = _safe_parts(relative, label="released path")
    descriptors = [os.dup(root_fd)]
    try:
        for component in parts[:-1]:
            try:
                named = os.stat(
                    component, dir_fd=descriptors[-1], follow_symlinks=False
                )
            except FileNotFoundError:
                return
            if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
                _fail(f"released path parent was replaced: {relative}")
            child = os.open(
                component,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                dir_fd=descriptors[-1],
            )
            descriptors.append(child)
        try:
            os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        except FileNotFoundError:
            return
        _fail(f"released path reappeared: {relative}")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _verify_boundary(expected: Mapping[str, Any], *, label: str) -> None:
    actual = _directory_boundary(Path(str(expected["path"])), label=label)
    if actual != dict(expected):
        _fail(f"{label} identity changed during release")


def _space_sample(descriptor: int) -> dict[str, int]:
    sample = os.fstatvfs(descriptor)
    fragment = int(sample.f_frsize)
    return {
        "free_bytes": int(sample.f_bavail) * fragment,
        "allocated_bytes": (int(sample.f_blocks) - int(sample.f_bfree)) * fragment,
    }


def _free_bytes(descriptor: int) -> int:
    return _space_sample(descriptor)["free_bytes"]


def _attempt_artifacts(
    layout: phase1.SessionLayout, bundle_seal: str
) -> tuple[Path, Path]:
    if HEX64_RE.fullmatch(bundle_seal) is None:
        _fail("bundle seal is invalid for durable release artifacts")
    stem = JOURNAL_PREFIX + bundle_seal
    return (
        layout.evidence / f"{stem}.journal.jsonl",
        layout.evidence / f"{stem}.receipt.json",
    )


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            _fail("short write to durable Phase-2 evidence")
        view = view[written:]


def _private_leaf_exists(root: Path, name: str) -> bool:
    root_fd = phase1._open_absolute_directory(root)  # noqa: SLF001
    try:
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(root_fd)


def _read_private_leaf(root: Path, name: str, *, label: str) -> bytes:
    root_fd = phase1._open_absolute_directory(root)  # noqa: SLF001
    descriptor = -1
    try:
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(named.st_mode) \
                or named.st_uid != os.getuid() \
                or named.st_nlink != 1 \
                or stat.S_IMODE(named.st_mode) != 0o600:
            _fail(f"{label} is not a private append-only artifact")
        descriptor = os.open(
            name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=root_fd
        )
        opened = os.fstat(descriptor)
        if not phase1._identity_equal(named, opened):  # noqa: SLF001
            _fail(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_JSON_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_JSON_BYTES:
                _fail(f"{label} exceeds the maximum durable evidence size")
        named_post = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not phase1._identity_equal(opened, named_post):  # noqa: SLF001
            _fail(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_fd)


def _write_private_leaf_once(root: Path, name: str, raw: bytes, *, label: str) -> None:
    root_fd = phase1._open_absolute_directory(root)  # noqa: SLF001
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
            0o600,
            dir_fd=root_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) \
                or opened.st_uid != os.getuid() \
                or opened.st_nlink != 1 \
                or stat.S_IMODE(opened.st_mode) != 0o600:
            _fail(f"new {label} is not a private single-link file")
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not _same_node(opened, named):
            _fail(f"new {label} changed while writing")
        os.fsync(root_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_fd)


class _JournalWriter:
    """O_APPEND journal anchored beneath the private evidence directory."""

    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name
        self.root_fd = phase1._open_absolute_directory(root)  # noqa: SLF001
        self.descriptor = -1
        self.record_index = 0
        self.head: str | None = None
        try:
            self.descriptor = os.open(
                name,
                os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL
                | _NOFOLLOW | _CLOEXEC,
                0o600,
                dir_fd=self.root_fd,
            )
            self.identity = os.fstat(self.descriptor)
            named = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
            if not _same_node(self.identity, named) \
                    or not stat.S_ISREG(self.identity.st_mode) \
                    or self.identity.st_uid != os.getuid() \
                    or self.identity.st_nlink != 1 \
                    or stat.S_IMODE(self.identity.st_mode) != 0o600:
                _fail("new Phase-2 release journal is unsafe")
            os.fsync(self.root_fd)
            self.size = int(self.identity.st_size)
        except BaseException:
            self.close()
            raise

    def append(self, body: Mapping[str, Any]) -> dict[str, Any]:
        reserved = {"schema", "record_index", "previous_record_sha256", "seal_sha256"}
        if reserved & set(body):
            _fail("journal event attempted to replace chain fields")
        document = phase1.seal_document(
            {
                "schema": JOURNAL_RECORD_SCHEMA,
                "record_index": self.record_index,
                "previous_record_sha256": self.head,
                **_clone(dict(body)),
            }
        )
        raw = phase1.canonical_json(document) + b"\n"
        before = os.fstat(self.descriptor)
        if int(before.st_size) != self.size:
            _fail("Phase-2 release journal size changed before append")
        _write_all(self.descriptor, raw)
        os.fsync(self.descriptor)
        named = os.stat(self.name, dir_fd=self.root_fd, follow_symlinks=False)
        opened = os.fstat(self.descriptor)
        if not _same_node(self.identity, opened) \
                or not _same_node(opened, named) \
                or stat.S_IMODE(opened.st_mode) != 0o600 \
                or int(opened.st_size) != self.size + len(raw) \
                or int(named.st_size) != int(opened.st_size) \
                or os.pread(self.descriptor, len(raw), self.size) != raw:
            _fail("Phase-2 release journal identity changed while appending")
        self.size = int(opened.st_size)
        self.head = document["seal_sha256"]
        self.record_index += 1
        return document

    @classmethod
    def resume(
        cls, root: Path, name: str, records: Sequence[Mapping[str, Any]]
    ) -> _JournalWriter:
        if not records:
            _fail("cannot resume an empty Phase-2 journal")
        writer = cls.__new__(cls)
        writer.root = root
        writer.name = name
        writer.root_fd = phase1._open_absolute_directory(root)  # noqa: SLF001
        writer.descriptor = -1
        try:
            named = os.stat(name, dir_fd=writer.root_fd, follow_symlinks=False)
            writer.descriptor = os.open(
                name,
                os.O_RDWR | os.O_APPEND | _NOFOLLOW | _CLOEXEC,
                dir_fd=writer.root_fd,
            )
            writer.identity = os.fstat(writer.descriptor)
            if not _same_node(writer.identity, named) \
                    or not stat.S_ISREG(writer.identity.st_mode) \
                    or writer.identity.st_uid != os.getuid() \
                    or writer.identity.st_nlink != 1 \
                    or stat.S_IMODE(writer.identity.st_mode) != 0o600:
                _fail("existing Phase-2 release journal is unsafe")
            writer.record_index = len(records)
            writer.head = str(records[-1]["seal_sha256"])
            writer.size = int(writer.identity.st_size)
            return writer
        except BaseException:
            writer.close()
            raise

    def close(self) -> None:
        if getattr(self, "descriptor", -1) >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        if getattr(self, "root_fd", -1) >= 0:
            os.close(self.root_fd)
            self.root_fd = -1

    def __enter__(self) -> _JournalWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _read_journal(path: Path) -> list[dict[str, Any]]:
    raw = _read_private_leaf(path.parent, path.name, label="Phase-2 release journal")
    if not raw or not raw.endswith(b"\n"):
        _fail("Phase-2 release journal is empty or has a torn final record")
    lines = raw.splitlines()
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for index, line in enumerate(lines):
        value = phase1.strict_json_bytes(line, label=f"journal record {index}")
        sealed = _verify_sealed(value, label=f"journal record {index}")
        if sealed.get("schema") != JOURNAL_RECORD_SCHEMA \
                or sealed.get("record_index") != index \
                or sealed.get("previous_record_sha256") != previous:
            _fail(f"Phase-2 journal hash chain changed at record {index}")
        previous = sealed["seal_sha256"]
        records.append(sealed)
    return records


def _receipt_delete_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": row["relative_path"],
        "category": row["category"],
        "type": row["type"],
        "device": row["device"],
        "inode": row["inode"],
        "logical_bytes": row["logical_bytes"],
        "allocated_bytes": row["allocated_bytes"],
        "sha256": row["sha256"],
    }


def _directory_stable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "relative_path", "category", "type", "device", "inode", "uid", "mode"
        )
    }


def _verify_remaining_directory(
    layout: phase1.SessionLayout, row: Mapping[str, Any]
) -> None:
    current = _node_row(
        layout,
        layout.session.joinpath(*_safe_parts(row["relative_path"], label="directory")),
        category=str(row["category"]),
    )
    if _directory_stable_row(current) != _directory_stable_row(row):
        _fail(f"remaining release directory changed: {row['relative_path']}")


def _expected_snapshot_directories(paths: Sequence[str]) -> set[str]:
    directories = {""}
    for raw in paths:
        parts = _safe_parts(raw, label="expected snapshot path")
        for stop in range(1, len(parts)):
            directories.add("/".join(parts[:stop]))
    return directories


def _verify_partial_tree_exactness(
    layout: phase1.SessionLayout,
    inventory: Mapping[str, Any],
    completed_count: int,
) -> None:
    remaining = inventory["delete_entries"][completed_count:]
    retained = inventory["retained_metadata_entries"]
    expected_snapshot = sorted(
        row["relative_path"].removeprefix(
            _relative_to_session(layout, layout.snapshot) + "/"
        )
        for row in [*retained, *remaining]
        if row["category"] in {"METADATA_SYMLINK", "WEIGHT_SYMLINK"}
    )
    actual_snapshot, actual_directories = phase1._list_tree_files(  # noqa: SLF001
        layout.snapshot
    )
    if actual_snapshot != expected_snapshot:
        _fail("snapshot tree differs from exact journal progress")
    actual_directory_names = {str(row["relative_path"]) for row in actual_directories}
    if actual_directory_names != _expected_snapshot_directories(expected_snapshot):
        _fail("snapshot directory tree differs from exact journal progress")

    expected_blobs = sorted(
        PurePosixPath(str(row["relative_path"])).name
        for row in [*retained, *remaining]
        if row["category"] in {"METADATA_BLOB", "WEIGHT_BLOB"}
    )
    blobs_fd = phase1._open_absolute_directory(layout.blobs)  # noqa: SLF001
    try:
        actual_blobs = sorted(os.listdir(blobs_fd))
    finally:
        os.close(blobs_fd)
    if actual_blobs != expected_blobs:
        _fail("blob directory differs from exact journal progress")

    expected_xet_leaves = [
        row for row in remaining if row["category"] == "XET_CONTENT"
    ]
    expected_xet_directories = [
        row for row in remaining if row["category"] == "XET_DIRECTORY"
    ]
    actual_xet_leaves, actual_xet_directories = _inventory_xet(layout)
    if phase1.canonical_json(actual_xet_leaves) != phase1.canonical_json(
        expected_xet_leaves
    ):
        _fail("dedicated Xet leaves differ from exact journal progress")
    if [_directory_stable_row(row) for row in actual_xet_directories] != [
        _directory_stable_row(row) for row in expected_xet_directories
    ]:
        _fail("dedicated Xet directories differ from exact journal progress")


def _verify_progress_state(
    layout: phase1.SessionLayout,
    inventory: Mapping[str, Any],
    completed_count: int,
    *,
    capsule_verifier: Verifier,
    expected_capsule_seal: str,
    lease_holds: Sequence[LeaseHold],
    expected_lease_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = inventory["delete_entries"]
    if type(completed_count) is not int or not 0 <= completed_count <= len(entries):
        _fail("journal completed count is outside the exact delete inventory")
    session_fd = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
    try:
        for row in entries[:completed_count]:
            _assert_relative_absent(session_fd, str(row["relative_path"]))
        for row in entries[completed_count:]:
            if row["type"] == "directory":
                _verify_remaining_directory(layout, row)
            else:
                _verify_preserved_node(session_fd, row)
        for row in inventory["retained_metadata_entries"]:
            _verify_preserved_node(session_fd, row)
    finally:
        os.close(session_fd)
    _verify_partial_tree_exactness(layout, inventory, completed_count)
    for root in inventory["retained_roots"]:
        _verify_boundary(root, label=f"retained root {root['path']}")
    _verify_boundary(inventory["mop_boundary"], label="MOP root")
    _verify_boundary(inventory["shared_xet_boundary"], label="shared Xet root")
    capsule = _verify_sealed(
        capsule_verifier(layout), label="journal reconciliation rollback capsule"
    )
    if capsule.get("seal_sha256") != expected_capsule_seal:
        _fail("rollback capsule changed relative to the sealed release bundle")
    lease_identities = _verify_lease_holds(lease_holds)
    if lease_identities != [dict(row) for row in expected_lease_identities]:
        _fail("held lease identities differ from the durable journal")
    return {
        "status": "PASS",
        "completed_targets_absent": completed_count,
        "unattempted_targets_exact": len(entries) - completed_count,
        "retained_metadata_nodes_exact": len(inventory["retained_metadata_entries"]),
        "retained_roots_exact": len(inventory["retained_roots"]),
        "complete_xet_tree_reconciled": True,
        "mop_boundary_exact": True,
        "shared_xet_boundary_exact": True,
        "capsule_verified": True,
        "lease_identities_exact": True,
        "failures": [],
    }


def _capture_progress_state(
    layout: phase1.SessionLayout,
    inventory: Mapping[str, Any],
    completed_count: int,
    *,
    capsule_verifier: Verifier,
    expected_capsule_seal: str,
    lease_holds: Sequence[LeaseHold],
    expected_lease_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        return _verify_progress_state(
            layout,
            inventory,
            completed_count,
            capsule_verifier=capsule_verifier,
            expected_capsule_seal=expected_capsule_seal,
            lease_holds=lease_holds,
            expected_lease_identities=expected_lease_identities,
        )
    except BaseException as exc:
        return {
            "status": "BLOCKED",
            "completed_targets_absent": None,
            "unattempted_targets_exact": None,
            "retained_metadata_nodes_exact": None,
            "retained_roots_exact": None,
            "complete_xet_tree_reconciled": False,
            "mop_boundary_exact": False,
            "shared_xet_boundary_exact": False,
            "capsule_verified": False,
            "lease_identities_exact": False,
            "failures": [f"{type(exc).__name__}: {exc}"[:2_000]],
        }


def _late_execution_fence(
    layout: phase1.SessionLayout,
    inventory: Mapping[str, Any],
    *,
    expected_capsule: Mapping[str, Any],
    capsule_verifier: Verifier,
    lease_holds: Sequence[LeaseHold],
    expected_lease_identities: Sequence[Mapping[str, Any]],
) -> None:
    """Fast protected-state fence after the slow full inventory hash pass."""
    if _verify_lease_holds(lease_holds) != [
        dict(row) for row in expected_lease_identities
    ]:
        _fail("exclusive leases changed before the late protected-state fence")
    # Close the analogous source-tree gap without rehashing the 595 GB weight
    # payload a third time: require the exact snapshot/blob name sets and the
    # complete content-hashed Xet tree.  Every large weight blob is still
    # rehashed from an O_NOFOLLOW descriptor immediately before its own unlink.
    _verify_partial_tree_exactness(layout, inventory, 0)
    session_fd = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
    try:
        for row in inventory["retained_metadata_entries"]:
            _verify_preserved_node(session_fd, row)
    finally:
        os.close(session_fd)
    for root in inventory["retained_roots"]:
        _verify_boundary(root, label=f"late retained root {root['path']}")
    _verify_boundary(inventory["mop_boundary"], label="late MOP root")
    _verify_boundary(inventory["shared_xet_boundary"], label="late shared Xet root")

    late_capsule = _verify_sealed(
        capsule_verifier(layout), label="late pre-START rollback capsule"
    )
    if phase1.canonical_json(late_capsule) != phase1.canonical_json(expected_capsule):
        _fail("rollback capsule changed during the slow final inventory hash pass")

    # The capsule verifier itself can read for a short interval.  Re-anchor all
    # protected roots and held leases once more after that final content read.
    for root in inventory["retained_roots"]:
        _verify_boundary(root, label=f"post-capsule retained root {root['path']}")
    _verify_boundary(inventory["mop_boundary"], label="post-capsule MOP root")
    _verify_boundary(
        inventory["shared_xet_boundary"], label="post-capsule shared Xet root"
    )
    if _verify_lease_holds(lease_holds) != [
        dict(row) for row in expected_lease_identities
    ]:
        _fail("exclusive leases changed during the late protected-state fence")


def _exception_row(exc: BaseException | None) -> dict[str, str] | None:
    if exc is None:
        return None
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:2_000],
    }


def _build_terminal_receipt(
    layout: phase1.SessionLayout,
    bundle: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    terminal_status: str,
    completed_count: int,
    confirmation_token: str,
    journal_path: Path,
    receipt_path: Path,
    journal_head: str,
    journal_record_count: int,
    lease_identities: Sequence[Mapping[str, Any]],
    intent_outcomes: Sequence[Mapping[str, Any]],
    space_before: Mapping[str, int],
    space_after: Mapping[str, int],
    exception: BaseException | None,
    preserved: Mapping[str, Any],
) -> dict[str, Any]:
    entries = inventory["delete_entries"]
    completed_rows = _clone(entries[:completed_count])
    unattempted_rows = _clone(entries[completed_count:])
    deleted = [_receipt_delete_row(row) for row in completed_rows]
    success = terminal_status == "SUCCESS"
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": (
            "PASS_EXACT_PHASE2_SOURCE_RELEASE"
            if success
            else "PARTIAL_FAILURE_EXACT_PHASE2_SOURCE_RELEASE"
        ),
        "terminal_status": terminal_status,
        "session": os.fspath(layout.session),
        "repo": phase1.KIMI_REPO,
        "revision": phase1.KIMI_REVISION,
        "source_verification_seal_sha256": bundle["source_verification_seal_sha256"],
        "capsule_verification_seal_sha256": bundle["capsule_verification_seal_sha256"],
        "inventory_seal_sha256": bundle["inventory_seal_sha256"],
        "audit_seal_sha256": bundle["audit_seal_sha256"],
        "bundle_seal_sha256": bundle["seal_sha256"],
        "post_release_capsule_seal_sha256": (
            bundle["capsule_verification_seal_sha256"]
            if preserved.get("capsule_verified") is True
            else None
        ),
        "post_release_capsule_verified": preserved.get("capsule_verified") is True,
        "confirmation_token_sha256": hashlib.sha256(
            confirmation_token.encode("utf-8")
        ).hexdigest(),
        "journal_path": os.fspath(journal_path),
        "receipt_path": os.fspath(receipt_path),
        "journal_head_sha256": journal_head,
        "journal_record_count_before_terminal": journal_record_count,
        "lease_identities": _clone(list(lease_identities)),
        "intent_outcomes": _clone(list(intent_outcomes)),
        "completed_count": completed_count,
        "completed_rows": completed_rows,
        "next_row": None if not unattempted_rows else _clone(unattempted_rows[0]),
        "unattempted_rows": unattempted_rows,
        "deleted_entry_count": len(deleted),
        "deleted_entries": deleted,
        "weight_symlink_count_deleted": sum(
            row["category"] == "WEIGHT_SYMLINK" for row in completed_rows
        ),
        "weight_blob_count_deleted": sum(
            row["category"] == "WEIGHT_BLOB" for row in completed_rows
        ),
        "xet_leaf_count_deleted": sum(
            row["category"] == "XET_CONTENT" for row in completed_rows
        ),
        "xet_directory_count_deleted": sum(
            row["category"] == "XET_DIRECTORY" for row in completed_rows
        ),
        "metadata_symlink_count_retained": inventory["metadata_symlink_count_retained"],
        "metadata_blob_count_retained": inventory["metadata_blob_count_retained"],
        "target_logical_bytes": inventory["target_logical_bytes"],
        "target_allocated_bytes": inventory["target_allocated_bytes"],
        "target_allocated_bytes_completed": sum(
            row["allocated_bytes"] for row in completed_rows
        ),
        "target_allocated_bytes_remaining": sum(
            row["allocated_bytes"] for row in unattempted_rows
        ),
        "filesystem_space_before": dict(space_before),
        "filesystem_space_after": dict(space_after),
        "filesystem_free_bytes_delta": (
            space_after["free_bytes"] - space_before["free_bytes"]
        ),
        "filesystem_allocated_bytes_delta": (
            space_after["allocated_bytes"] - space_before["allocated_bytes"]
        ),
        "free_bytes_before": space_before["free_bytes"],
        "free_bytes_after": space_after["free_bytes"],
        "free_bytes_delta": space_after["free_bytes"] - space_before["free_bytes"],
        "exception": _exception_row(exception),
        "preserved_node_status": _clone(dict(preserved)),
        "mop_touched": False if preserved.get("mop_boundary_exact") is True else None,
        "shared_xet_touched": (
            False if preserved.get("shared_xet_boundary_exact") is True else None
        ),
        "capsule_retained": preserved.get("capsule_verified") is True,
        "recovery_retained": preserved.get("retained_roots_exact") is not None,
        "evidence_retained": preserved.get("retained_roots_exact") is not None,
        "globs_used": False,
        "recursive_delete_used": False,
        "deletion_performed": completed_count > 0,
    }
    return phase1.seal_document(body)


def _persist_terminal(
    writer: _JournalWriter,
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    raw = phase1.canonical_json(dict(receipt)) + b"\n"
    _write_private_leaf_once(
        receipt_path.parent, receipt_path.name, raw, label="Phase-2 terminal receipt"
    )
    return writer.append(
        {
            "event": "TERMINAL",
            "terminal_status": receipt["terminal_status"],
            "receipt_path": os.fspath(receipt_path),
            "receipt_seal_sha256": receipt["seal_sha256"],
            "receipt_journal_head_sha256": receipt["journal_head_sha256"],
            "completed_count": receipt["completed_count"],
            "intent_outcome_count": len(receipt["intent_outcomes"]),
            "next_index": (
                None
                if receipt["next_row"] is None
                else int(receipt["completed_count"])
            ),
            "preserved_node_status": receipt["preserved_node_status"]["status"],
        }
    )


_JOURNAL_COMMON_FIELDS = {
    "schema", "record_index", "previous_record_sha256", "event", "seal_sha256",
}
_JOURNAL_START_FIELDS = _JOURNAL_COMMON_FIELDS | {
    "bundle_seal_sha256", "inventory_seal_sha256", "audit_seal_sha256",
    "source_verification_seal_sha256", "capsule_verification_seal_sha256",
    "confirmation_token_sha256", "delete_entry_count", "journal_path",
    "receipt_path", "lease_identities", "filesystem_space_before",
}
_JOURNAL_PREPARE_FIELDS = _JOURNAL_COMMON_FIELDS | {
    "delete_index", "row",
}
_JOURNAL_COMMIT_FIELDS = _JOURNAL_COMMON_FIELDS | {
    "delete_index", "row", "deleted_entry", "commit_basis",
}
_JOURNAL_ABORT_FIELDS = _JOURNAL_COMMON_FIELDS | {
    "delete_index", "row", "outcome",
}
_JOURNAL_TERMINAL_FIELDS = _JOURNAL_COMMON_FIELDS | {
    "terminal_status", "receipt_path", "receipt_seal_sha256",
    "receipt_journal_head_sha256", "completed_count", "next_index",
    "preserved_node_status", "intent_outcome_count",
}


def _valid_space_sample(value: object) -> bool:
    return isinstance(value, dict) \
        and set(value) == {"free_bytes", "allocated_bytes"} \
        and all(type(item) is int and item >= 0 for item in value.values())


def _validate_journal_records(
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    confirmation_token: str,
    *,
    journal_path: Path,
    receipt_path: Path,
    lease_identities: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    int,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    if not records:
        _fail("Phase-2 release journal has no START record")
    start = dict(records[0])
    _exact_keys(start, _JOURNAL_START_FIELDS, "Phase-2 journal START")
    inventory = bundle["inventory"]
    if start.get("event") != "START" \
            or start.get("bundle_seal_sha256") != bundle["seal_sha256"] \
            or start.get("inventory_seal_sha256") != bundle["inventory_seal_sha256"] \
            or start.get("audit_seal_sha256") != bundle["audit_seal_sha256"] \
            or start.get("source_verification_seal_sha256") != bundle[
                "source_verification_seal_sha256"
            ] \
            or start.get("capsule_verification_seal_sha256") != bundle[
                "capsule_verification_seal_sha256"
            ] \
            or start.get("confirmation_token_sha256") != hashlib.sha256(
                confirmation_token.encode("utf-8")
            ).hexdigest() \
            or start.get("delete_entry_count") != inventory["delete_entry_count"] \
            or start.get("journal_path") != os.fspath(journal_path) \
            or start.get("receipt_path") != os.fspath(receipt_path) \
            or start.get("lease_identities") != [dict(row) for row in lease_identities] \
            or not _valid_space_sample(start.get("filesystem_space_before")):
        _fail("Phase-2 journal START binding changed")
    completed = 0
    pending: dict[str, Any] | None = None
    halted = False
    intent_outcomes: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    for record in records[1:]:
        event = record.get("event")
        if terminal is not None:
            _fail("Phase-2 journal has records after its terminal record")
        if event == "PREPARE":
            _exact_keys(record, _JOURNAL_PREPARE_FIELDS, "Phase-2 journal PREPARE")
            if halted or pending is not None \
                    or completed >= len(inventory["delete_entries"]) \
                    or record.get("delete_index") != completed \
                    or record.get("row") != inventory["delete_entries"][completed]:
                _fail(f"Phase-2 journal PREPARE binding changed at index {completed}")
            pending = dict(record)
        elif event == "COMMIT":
            _exact_keys(record, _JOURNAL_COMMIT_FIELDS, "Phase-2 journal COMMIT")
            if pending is None \
                    or record.get("delete_index") != completed \
                    or record.get("row") != pending["row"] \
                    or record.get("deleted_entry") != _receipt_delete_row(pending["row"]) \
                    or record.get("commit_basis") not in {
                        "NORMAL_POST_UNLINK_FSYNC",
                        "RECONCILED_ABSENT_AFTER_DURABLE_PREPARE",
                    }:
                _fail(f"Phase-2 journal COMMIT binding changed at index {completed}")
            if record["commit_basis"] != "NORMAL_POST_UNLINK_FSYNC":
                intent_outcomes.append(
                    {
                        "delete_index": completed,
                        "row": _clone(pending["row"]),
                        "outcome": record["commit_basis"],
                        "certainty": (
                            "PATH_ABSENCE_CONFIRMED_ORIGINAL_INODE_UNLINK_INFERRED"
                        ),
                        "journal_record_seal_sha256": record["seal_sha256"],
                    }
                )
            pending = None
            completed += 1
        elif event == "ABORT":
            _exact_keys(record, _JOURNAL_ABORT_FIELDS, "Phase-2 journal ABORT")
            if pending is None \
                    or record.get("delete_index") != completed \
                    or record.get("row") != pending["row"] \
                    or record.get("outcome") not in {
                        "PREPARED_TARGET_REMAINS_EXACT_NOT_DELETED",
                        "PREPARED_TARGET_MISMATCH_FAIL_CLOSED",
                    }:
                _fail(f"Phase-2 journal ABORT binding changed at index {completed}")
            intent_outcomes.append(
                {
                    "delete_index": completed,
                    "row": _clone(pending["row"]),
                    "outcome": record["outcome"],
                    "certainty": (
                        "EXACT_TARGET_CONFIRMED_PRESENT"
                        if record["outcome"]
                        == "PREPARED_TARGET_REMAINS_EXACT_NOT_DELETED"
                        else "PREPARED_TARGET_STATE_MISMATCH_UNRESOLVED"
                    ),
                    "journal_record_seal_sha256": record["seal_sha256"],
                }
            )
            pending = None
            halted = True
        elif event == "TERMINAL":
            _exact_keys(record, _JOURNAL_TERMINAL_FIELDS, "Phase-2 journal TERMINAL")
            expected_next = None if completed == len(inventory["delete_entries"]) else completed
            if pending is not None \
                    or record.get("terminal_status") not in {"SUCCESS", "PARTIAL_FAILURE"} \
                    or record.get("receipt_path") != os.fspath(receipt_path) \
                    or not isinstance(record.get("receipt_seal_sha256"), str) \
                    or record.get("receipt_journal_head_sha256") != record.get(
                        "previous_record_sha256"
                    ) \
                    or record.get("completed_count") != completed \
                    or record.get("intent_outcome_count") != len(intent_outcomes) \
                    or record.get("next_index") != expected_next \
                    or record.get("preserved_node_status") not in {"PASS", "BLOCKED"}:
                _fail("Phase-2 journal terminal binding changed")
            terminal = dict(record)
        else:
            _fail(f"unknown Phase-2 journal event: {event!r}")
    return start, completed, pending, terminal, intent_outcomes


def _read_durable_receipt(path: Path) -> dict[str, Any]:
    raw = _read_private_leaf(path.parent, path.name, label="Phase-2 terminal receipt")
    if not raw.endswith(b"\n"):
        _fail("Phase-2 terminal receipt has a torn final write")
    return phase1.strict_json_bytes(raw[:-1], label="Phase-2 terminal receipt")


def _sample_session_space(layout: phase1.SessionLayout) -> dict[str, int]:
    descriptor = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
    try:
        return _space_sample(descriptor)
    finally:
        os.close(descriptor)


def _relative_exists_no_follow(root_fd: int, relative: str) -> bool:
    parts = _safe_parts(relative, label="prepared release target")
    descriptors = [os.dup(root_fd)]
    try:
        for component in parts[:-1]:
            try:
                named = os.stat(
                    component, dir_fd=descriptors[-1], follow_symlinks=False
                )
            except FileNotFoundError:
                return False
            if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
                _fail(f"prepared target parent was replaced: {relative}")
            child = os.open(
                component,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                dir_fd=descriptors[-1],
            )
            if not phase1._identity_equal(named, os.fstat(child)):  # noqa: SLF001
                os.close(child)
                _fail(f"prepared target parent changed while opening: {relative}")
            descriptors.append(child)
        try:
            os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _resolve_pending_prepare(
    writer: _JournalWriter,
    layout: phase1.SessionLayout,
    pending: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    row = pending["row"]
    index = int(pending["delete_index"])
    session_fd = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
    try:
        exists = _relative_exists_no_follow(session_fd, str(row["relative_path"]))
        if not exists:
            record = writer.append(
                {
                    "event": "COMMIT",
                    "delete_index": index,
                    "row": row,
                    "deleted_entry": _receipt_delete_row(row),
                    "commit_basis": "RECONCILED_ABSENT_AFTER_DURABLE_PREPARE",
                }
            )
            outcome = "RECONCILED_ABSENT_AFTER_DURABLE_PREPARE"
            committed = True
        else:
            exact = True
            try:
                if row["type"] == "directory":
                    _verify_remaining_directory(layout, row)
                else:
                    _verify_preserved_node(session_fd, row)
            except BaseException:
                exact = False
            outcome = (
                "PREPARED_TARGET_REMAINS_EXACT_NOT_DELETED"
                if exact
                else "PREPARED_TARGET_MISMATCH_FAIL_CLOSED"
            )
            record = writer.append(
                {
                    "event": "ABORT",
                    "delete_index": index,
                    "row": row,
                    "outcome": outcome,
                }
            )
            committed = False
        return committed, {
            "delete_index": index,
            "row": _clone(row),
            "outcome": outcome,
            "certainty": (
                "PATH_ABSENCE_CONFIRMED_ORIGINAL_INODE_UNLINK_INFERRED"
                if committed
                else (
                    "EXACT_TARGET_CONFIRMED_PRESENT"
                    if outcome == "PREPARED_TARGET_REMAINS_EXACT_NOT_DELETED"
                    else "PREPARED_TARGET_STATE_MISMATCH_UNRESOLVED"
                )
            ),
            "journal_record_seal_sha256": record["seal_sha256"],
        }
    finally:
        os.close(session_fd)


def _reconcile_existing_attempt(
    layout: phase1.SessionLayout,
    bundle: Mapping[str, Any],
    confirmation_token: str,
    *,
    capsule_verifier: Verifier,
    lease_holds: Sequence[LeaseHold],
) -> dict[str, Any]:
    journal_path, receipt_path = _attempt_artifacts(layout, bundle["seal_sha256"])
    records = _read_journal(journal_path)
    current_lease_identities = _verify_lease_holds(lease_holds)
    start, completed, pending, terminal, intent_outcomes = _validate_journal_records(
        records,
        bundle,
        confirmation_token,
        journal_path=journal_path,
        receipt_path=receipt_path,
        lease_identities=current_lease_identities,
    )
    receipt_exists = _private_leaf_exists(receipt_path.parent, receipt_path.name)
    if pending is not None:
        if terminal is not None or receipt_exists:
            _fail("pending Phase-2 PREPARE conflicts with terminal evidence")
        with _JournalWriter.resume(journal_path.parent, journal_path.name, records) as writer:
            _resolve_pending_prepare(writer, layout, pending)
        records = _read_journal(journal_path)
        start, completed, pending, terminal, intent_outcomes = _validate_journal_records(
            records,
            bundle,
            confirmation_token,
            journal_path=journal_path,
            receipt_path=receipt_path,
            lease_identities=current_lease_identities,
        )
        if pending is not None or terminal is not None:
            _fail("pending Phase-2 PREPARE did not resolve exactly")
    preserved = _capture_progress_state(
        layout,
        bundle["inventory"],
        completed,
        capsule_verifier=capsule_verifier,
        expected_capsule_seal=bundle["capsule_verification_seal_sha256"],
        lease_holds=lease_holds,
        expected_lease_identities=current_lease_identities,
    )
    if terminal is not None:
        if not receipt_exists:
            _fail("terminal Phase-2 journal is missing its durable receipt")
        receipt = verify_receipt(_read_durable_receipt(receipt_path), bundle)
        if terminal["receipt_seal_sha256"] != receipt["seal_sha256"] \
                or terminal["terminal_status"] != receipt["terminal_status"] \
                or terminal["completed_count"] != receipt["completed_count"] \
                or terminal["receipt_journal_head_sha256"] != receipt[
                    "journal_head_sha256"
                ] \
                or terminal["record_index"] != receipt[
                    "journal_record_count_before_terminal"
                ] \
                or terminal["preserved_node_status"] != receipt[
                    "preserved_node_status"
                ]["status"] \
                or terminal["intent_outcome_count"] != len(receipt["intent_outcomes"]) \
                or receipt["intent_outcomes"] != intent_outcomes:
            _fail("terminal journal and durable receipt differ")
        if preserved["status"] != "PASS":
            _fail("durable release progress no longer reconciles with the filesystem")
        return receipt

    if receipt_exists:
        receipt = verify_receipt(_read_durable_receipt(receipt_path), bundle)
        if receipt.get("journal_head_sha256") != records[-1]["seal_sha256"] \
                or receipt.get("journal_record_count_before_terminal") != len(records) \
                or receipt.get("completed_count") != completed \
                or receipt.get("intent_outcomes") != intent_outcomes:
            _fail("unterminated journal receipt does not bind its exact chain head")
    else:
        interrupted = Phase2ReleaseError(
            "previous release attempt ended without a terminal journal record"
        )
        receipt = _build_terminal_receipt(
            layout,
            bundle,
            bundle["inventory"],
            terminal_status="PARTIAL_FAILURE",
            completed_count=completed,
            confirmation_token=confirmation_token,
            journal_path=journal_path,
            receipt_path=receipt_path,
            journal_head=str(records[-1]["seal_sha256"]),
            journal_record_count=len(records),
            lease_identities=current_lease_identities,
            intent_outcomes=intent_outcomes,
            space_before=start["filesystem_space_before"],
            space_after=_sample_session_space(layout),
            exception=interrupted,
            preserved=preserved,
        )
    with _JournalWriter.resume(journal_path.parent, journal_path.name, records) as writer:
        _persist_terminal(writer, receipt_path, receipt) if not receipt_exists else writer.append(
            {
                "event": "TERMINAL",
                "terminal_status": receipt["terminal_status"],
                "receipt_path": os.fspath(receipt_path),
                "receipt_seal_sha256": receipt["seal_sha256"],
                "receipt_journal_head_sha256": receipt["journal_head_sha256"],
                "completed_count": receipt["completed_count"],
                "intent_outcome_count": len(receipt["intent_outcomes"]),
                "next_index": (
                    None if receipt["next_row"] is None else receipt["completed_count"]
                ),
                "preserved_node_status": receipt["preserved_node_status"]["status"],
            }
        )
    if preserved["status"] != "PASS":
        _fail("interrupted release journal was sealed but filesystem reconciliation failed")
    return receipt


def execute_release(
    layout: phase1.SessionLayout,
    bundle: Mapping[str, Any],
    *,
    confirmation_token: str,
    manifest: Mapping[str, Any],
    source_verifier: Verifier,
    capsule_verifier: Verifier,
    probe: AuditProbe,
    queue_roots: Sequence[Path] | None = None,
    lease_paths: Sequence[Path] | None = None,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
    repo_root: Path = phase1.REPO_ROOT,
    fault_after: int | None = None,
    fault_after_unlink_before_commit: int | None = None,
) -> dict[str, Any]:
    """Execute or reconcile one journaled release under all exclusive leases."""
    value = verify_bundle(bundle)
    if value["status"] != "PASS_CONFIRMATION_REQUIRED" \
            or value["audit"]["blockers"] != []:
        _fail("blocked Phase-2 bundle cannot execute")
    if value["session"] != os.fspath(layout.session):
        _fail("Phase-2 bundle belongs to another session")
    expected_token = derive_confirmation_token(value["audit"])
    if not isinstance(confirmation_token, str) \
            or not __import__("hmac").compare_digest(confirmation_token, expected_token):
        _fail("explicit confirmation token does not match the sealed audit")
    entries = value["inventory"]["delete_entries"]
    if fault_after is not None \
            and (type(fault_after) is not int or not 0 <= fault_after <= len(entries)):
        _fail("fault injection index is outside the fake exact inventory")
    if fault_after_unlink_before_commit is not None \
            and (
                type(fault_after_unlink_before_commit) is not int
                or not 0 <= fault_after_unlink_before_commit < len(entries)
            ):
        _fail("pre-commit crash injection index is outside the fake exact inventory")
    leases = tuple(lease_paths) if lease_paths is not None else default_lease_paths(layout)
    journal_path, receipt_path = _attempt_artifacts(layout, value["seal_sha256"])
    with _exclusive_release_leases(layout, leases) as lease_holds:
        lease_identities = _verify_lease_holds(lease_holds)
        if _private_leaf_exists(journal_path.parent, journal_path.name):
            receipt = _reconcile_existing_attempt(
                layout,
                value,
                confirmation_token,
                capsule_verifier=capsule_verifier,
                lease_holds=lease_holds,
            )
            if receipt["terminal_status"] == "PARTIAL_FAILURE":
                raise Phase2PartialReleaseError(
                    "existing Phase-2 attempt reconciled to durable PARTIAL_FAILURE",
                    receipt,
                )
            return receipt
        if _private_leaf_exists(receipt_path.parent, receipt_path.name):
            _fail("durable Phase-2 receipt exists without its append-only journal")

        current_source = _clone(dict(source_verifier(layout)))
        current_capsule = _clone(dict(capsule_verifier(layout)))
        current = build_release_bundle(
            layout,
            manifest,
            current_source,
            current_capsule,
            probe=probe,
            queue_roots=queue_roots,
            lease_paths=leases,
            owned_lease_paths=tuple(hold.path for hold in lease_holds),
            mop_root=mop_root,
            shared_xet=shared_xet,
            repo_root=repo_root,
        )
        if phase1.canonical_json(current) != phase1.canonical_json(value):
            _fail("Phase-2 source/capsule/inventory/audit changed before execution")
        inventory = verify_inventory(value["inventory"])

        # The probe above is intentionally allowed to be slow.  Once it has
        # finished, rebuild every source/capsule/inventory fact and compare the
        # complete sealed view plus every held lease immediately before the
        # journal START and first unlink.
        final_source = _clone(dict(source_verifier(layout)))
        final_capsule = _clone(dict(capsule_verifier(layout)))
        if phase1.canonical_json(final_source) != phase1.canonical_json(
            value["source_verification"]
        ) or phase1.canonical_json(final_capsule) != phase1.canonical_json(
            value["capsule_verification"]
        ):
            _fail("source or rollback capsule changed after the slow live probes")
        final_inventory = build_exact_inventory(
            layout,
            manifest,
            final_source,
            final_capsule,
            mop_root=mop_root,
            shared_xet=shared_xet,
        )
        if phase1.canonical_json(final_inventory) != phase1.canonical_json(inventory):
            _fail("full exact inventory/protected state changed after the slow live probes")
        if _verify_lease_holds(lease_holds) != lease_identities:
            _fail("exclusive lease identities changed after the slow live probes")

        session_fd = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
        try:
            session_identity = os.fstat(session_fd)
            named_session = layout.session.lstat()
            if not phase1._identity_equal(session_identity, named_session):  # noqa: SLF001
                _fail("session root changed before exact release")
            space_before = _space_sample(session_fd)
            _late_execution_fence(
                layout,
                inventory,
                expected_capsule=value["capsule_verification"],
                capsule_verifier=capsule_verifier,
                lease_holds=lease_holds,
                expected_lease_identities=lease_identities,
            )
            completed = 0
            with _JournalWriter(journal_path.parent, journal_path.name) as writer:
                writer.append(
                    {
                        "event": "START",
                        "bundle_seal_sha256": value["seal_sha256"],
                        "inventory_seal_sha256": value["inventory_seal_sha256"],
                        "audit_seal_sha256": value["audit_seal_sha256"],
                        "source_verification_seal_sha256": value[
                            "source_verification_seal_sha256"
                        ],
                        "capsule_verification_seal_sha256": value[
                            "capsule_verification_seal_sha256"
                        ],
                        "confirmation_token_sha256": hashlib.sha256(
                            confirmation_token.encode("utf-8")
                        ).hexdigest(),
                        "delete_entry_count": len(entries),
                        "journal_path": os.fspath(journal_path),
                        "receipt_path": os.fspath(receipt_path),
                        "lease_identities": lease_identities,
                        "filesystem_space_before": space_before,
                    }
                )
                terminal_persisted = False
                intent_outcomes: list[dict[str, Any]] = []
                try:
                    if fault_after == 0:
                        raise RuntimeError("injected fake Phase-2 fault after 0 deletes")
                    for index, row in enumerate(entries):
                        writer.append(
                            {
                                "event": "PREPARE",
                                "delete_index": index,
                                "row": row,
                            }
                        )
                        deleted = (
                            _rmdir_exact(session_fd, row)
                            if row["type"] == "directory"
                            else _unlink_exact(session_fd, row)
                        )
                        if fault_after_unlink_before_commit == index:
                            raise _SimulatedHardCrash(
                                "injected fake hard crash after unlink/fsync before COMMIT"
                            )
                        writer.append(
                            {
                                "event": "COMMIT",
                                "delete_index": index,
                                "row": row,
                                "deleted_entry": deleted,
                                "commit_basis": "NORMAL_POST_UNLINK_FSYNC",
                            }
                        )
                        completed += 1
                        if fault_after == completed:
                            raise RuntimeError(
                                f"injected fake Phase-2 fault after {completed} deletes"
                            )
                    preserved = _verify_progress_state(
                        layout,
                        inventory,
                        completed,
                        capsule_verifier=capsule_verifier,
                        expected_capsule_seal=value["capsule_verification_seal_sha256"],
                        lease_holds=lease_holds,
                        expected_lease_identities=lease_identities,
                    )
                    named_session_post = layout.session.lstat()
                    if not phase1._identity_equal(  # noqa: SLF001
                        session_identity, named_session_post
                    ):
                        _fail("session root changed during exact release")
                    space_after = _space_sample(session_fd)
                    receipt = _build_terminal_receipt(
                        layout,
                        value,
                        inventory,
                        terminal_status="SUCCESS",
                        completed_count=completed,
                        confirmation_token=confirmation_token,
                        journal_path=journal_path,
                        receipt_path=receipt_path,
                        journal_head=str(writer.head),
                        journal_record_count=writer.record_index,
                        lease_identities=lease_identities,
                        intent_outcomes=intent_outcomes,
                        space_before=space_before,
                        space_after=space_after,
                        exception=None,
                        preserved=preserved,
                    )
                    _persist_terminal(writer, receipt_path, receipt)
                    terminal_persisted = True
                    return verify_receipt(receipt, value)
                except BaseException as exc:
                    if isinstance(exc, _SimulatedHardCrash):
                        raise
                    if terminal_persisted:
                        raise
                    records_now = _read_journal(journal_path)
                    (
                        _start_now,
                        completed,
                        pending,
                        terminal_now,
                        intent_outcomes,
                    ) = _validate_journal_records(
                        records_now,
                        value,
                        confirmation_token,
                        journal_path=journal_path,
                        receipt_path=receipt_path,
                        lease_identities=lease_identities,
                    )
                    if terminal_now is not None:
                        raise Phase2ReleaseError(
                            "unexpected terminal journal record inside active attempt"
                        ) from exc
                    if pending is not None:
                        _resolve_pending_prepare(writer, layout, pending)
                        records_now = _read_journal(journal_path)
                        (
                            _start_now,
                            completed,
                            pending,
                            terminal_now,
                            intent_outcomes,
                        ) = _validate_journal_records(
                            records_now,
                            value,
                            confirmation_token,
                            journal_path=journal_path,
                            receipt_path=receipt_path,
                            lease_identities=lease_identities,
                        )
                        if pending is not None or terminal_now is not None:
                            raise Phase2ReleaseError(
                                "active PREPARE could not be reconciled before terminal receipt"
                            ) from exc
                    space_after = _space_sample(session_fd)
                    preserved = _capture_progress_state(
                        layout,
                        inventory,
                        completed,
                        capsule_verifier=capsule_verifier,
                        expected_capsule_seal=value["capsule_verification_seal_sha256"],
                        lease_holds=lease_holds,
                        expected_lease_identities=lease_identities,
                    )
                    receipt = _build_terminal_receipt(
                        layout,
                        value,
                        inventory,
                        terminal_status="PARTIAL_FAILURE",
                        completed_count=completed,
                        confirmation_token=confirmation_token,
                        journal_path=journal_path,
                        receipt_path=receipt_path,
                        journal_head=str(writer.head),
                        journal_record_count=writer.record_index,
                        lease_identities=lease_identities,
                        intent_outcomes=intent_outcomes,
                        space_before=space_before,
                        space_after=space_after,
                        exception=exc,
                        preserved=preserved,
                    )
                    try:
                        _persist_terminal(writer, receipt_path, receipt)
                    except BaseException as persistence_error:
                        raise Phase2ReleaseError(
                            "PARTIAL_FAILURE receipt/journal persistence failed; "
                            f"reconcile {journal_path} before any further action: "
                            f"{type(persistence_error).__name__}: {persistence_error}"
                        ) from exc
                    verified = verify_receipt(receipt, value)
                    raise Phase2PartialReleaseError(
                        f"Phase-2 release stopped after {completed} exact deletions; "
                        f"durable receipt: {receipt_path}",
                        verified,
                    ) from exc
        finally:
            os.close(session_fd)


_RECEIPT_FIELDS = {
    "schema", "status", "terminal_status", "session", "repo", "revision",
    "source_verification_seal_sha256", "capsule_verification_seal_sha256",
    "inventory_seal_sha256", "audit_seal_sha256", "bundle_seal_sha256",
    "post_release_capsule_seal_sha256", "post_release_capsule_verified",
    "confirmation_token_sha256", "journal_path", "receipt_path",
    "journal_head_sha256", "journal_record_count_before_terminal",
    "lease_identities", "intent_outcomes", "completed_count", "completed_rows", "next_row",
    "unattempted_rows", "deleted_entry_count", "deleted_entries",
    "weight_symlink_count_deleted", "weight_blob_count_deleted",
    "xet_leaf_count_deleted", "xet_directory_count_deleted",
    "metadata_symlink_count_retained", "metadata_blob_count_retained",
    "target_logical_bytes", "target_allocated_bytes",
    "target_allocated_bytes_completed", "target_allocated_bytes_remaining",
    "filesystem_space_before", "filesystem_space_after", "filesystem_free_bytes_delta",
    "filesystem_allocated_bytes_delta", "free_bytes_before", "free_bytes_after",
    "free_bytes_delta", "exception", "preserved_node_status", "mop_touched",
    "shared_xet_touched",
    "capsule_retained", "recovery_retained", "evidence_retained", "globs_used",
    "recursive_delete_used", "deletion_performed", "seal_sha256",
}


def verify_receipt(receipt: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    value = _verify_sealed(receipt, label="Phase-2 release receipt")
    bound = verify_bundle(bundle)
    if bound.get("status") != "PASS_CONFIRMATION_REQUIRED" \
            or not isinstance(bound.get("confirmation_token"), str):
        _fail("a release receipt cannot bind to a blocked bundle")
    _exact_keys(value, _RECEIPT_FIELDS, "Phase-2 release receipt")
    inventory = bound["inventory"]
    terminal_status = value.get("terminal_status")
    expected_status = (
        "PASS_EXACT_PHASE2_SOURCE_RELEASE"
        if terminal_status == "SUCCESS"
        else "PARTIAL_FAILURE_EXACT_PHASE2_SOURCE_RELEASE"
    )
    if value.get("schema") != RECEIPT_SCHEMA \
            or terminal_status not in {"SUCCESS", "PARTIAL_FAILURE"} \
            or value.get("status") != expected_status \
            or value.get("session") != bound["session"] \
            or value.get("repo") != phase1.KIMI_REPO \
            or value.get("revision") != phase1.KIMI_REVISION \
            or value.get("bundle_seal_sha256") != bound["seal_sha256"] \
            or value.get("source_verification_seal_sha256") != bound[
                "source_verification_seal_sha256"
            ] \
            or value.get("capsule_verification_seal_sha256") != bound[
                "capsule_verification_seal_sha256"
            ] \
            or value.get("inventory_seal_sha256") != bound["inventory_seal_sha256"] \
            or value.get("audit_seal_sha256") != bound["audit_seal_sha256"] \
            or value.get("confirmation_token_sha256") != hashlib.sha256(
                bound["confirmation_token"].encode("utf-8")
            ).hexdigest() \
            or value.get("metadata_symlink_count_retained") != inventory[
                "metadata_symlink_count_retained"
            ] \
            or value.get("metadata_blob_count_retained") != inventory[
                "metadata_blob_count_retained"
            ] \
            or value.get("target_logical_bytes") != inventory["target_logical_bytes"] \
            or value.get("target_allocated_bytes") != inventory["target_allocated_bytes"] \
            or value.get("globs_used") is not False \
            or value.get("recursive_delete_used") is not False:
        _fail("Phase-2 release receipt binding/counts changed")
    completed_count = value.get("completed_count")
    if type(completed_count) is not int \
            or not 0 <= completed_count <= inventory["delete_entry_count"]:
        _fail("Phase-2 receipt completed count is invalid")
    expected_completed = inventory["delete_entries"][:completed_count]
    expected_unattempted = inventory["delete_entries"][completed_count:]
    expected_next = None if not expected_unattempted else expected_unattempted[0]
    expected_deleted = [_receipt_delete_row(row) for row in expected_completed]
    if value.get("completed_rows") != expected_completed \
            or value.get("unattempted_rows") != expected_unattempted \
            or value.get("next_row") != expected_next \
            or value.get("deleted_entry_count") != completed_count \
            or value.get("deleted_entries") != expected_deleted:
        _fail("Phase-2 receipt deletion list is malformed")
    for category, field in (
        ("WEIGHT_SYMLINK", "weight_symlink_count_deleted"),
        ("WEIGHT_BLOB", "weight_blob_count_deleted"),
        ("XET_CONTENT", "xet_leaf_count_deleted"),
        ("XET_DIRECTORY", "xet_directory_count_deleted"),
    ):
        if value.get(field) != sum(row["category"] == category for row in expected_completed):
            _fail(f"Phase-2 receipt category count changed: {field}")
    completed_allocated = sum(row["allocated_bytes"] for row in expected_completed)
    remaining_allocated = sum(row["allocated_bytes"] for row in expected_unattempted)
    if value.get("target_allocated_bytes_completed") != completed_allocated \
            or value.get("target_allocated_bytes_remaining") != remaining_allocated \
            or completed_allocated + remaining_allocated != inventory["target_allocated_bytes"]:
        _fail("Phase-2 receipt target allocation accounting changed")
    before_space = value.get("filesystem_space_before")
    after_space = value.get("filesystem_space_after")
    if not _valid_space_sample(before_space) or not _valid_space_sample(after_space):
        _fail("Phase-2 receipt filesystem allocation sample is malformed")
    before = value.get("free_bytes_before")
    after = value.get("free_bytes_after")
    delta = value.get("free_bytes_delta")
    if type(before) is not int or before < 0 \
            or type(after) is not int or after < 0 \
            or type(delta) is not int or delta != after - before \
            or before != before_space["free_bytes"] \
            or after != after_space["free_bytes"] \
            or value.get("filesystem_free_bytes_delta") != (
                after_space["free_bytes"] - before_space["free_bytes"]
            ) \
            or value.get("filesystem_allocated_bytes_delta") != (
                after_space["allocated_bytes"] - before_space["allocated_bytes"]
            ):
        _fail("Phase-2 receipt free-space delta arithmetic changed")
    journal_path, receipt_path = _attempt_artifacts(
        phase1.SessionLayout(
            parent=Path(bound["session"]).parent,
            session=Path(bound["session"]),
            hub=Path(bound["session"]) / "hub",
            xet=Path(bound["session"]) / "xet",
            build=Path(bound["session"]) / "build",
            tmp=Path(bound["session"]) / "build" / "tmp",
            hf_home=Path(bound["session"]) / "build" / "hf-home",
            recovery=Path(bound["session"]) / "recovery",
            evidence=Path(bound["session"]) / "evidence",
        ),
        bound["seal_sha256"],
    )
    preserved = value.get("preserved_node_status")
    if value.get("journal_path") != os.fspath(journal_path) \
            or value.get("receipt_path") != os.fspath(receipt_path) \
            or not isinstance(value.get("journal_head_sha256"), str) \
            or HEX64_RE.fullmatch(value["journal_head_sha256"]) is None \
            or type(value.get("journal_record_count_before_terminal")) is not int \
            or value["journal_record_count_before_terminal"] < 1 \
            or not isinstance(value.get("lease_identities"), list) \
            or not isinstance(value.get("intent_outcomes"), list) \
            or not isinstance(preserved, dict) \
            or preserved.get("status") not in {"PASS", "BLOCKED"}:
        _fail("Phase-2 receipt journal/preserved-state binding changed")
    for outcome in value["intent_outcomes"]:
        if not isinstance(outcome, dict) \
                or set(outcome) != {
                    "delete_index", "row", "outcome", "certainty",
                    "journal_record_seal_sha256"
                } \
                or type(outcome.get("delete_index")) is not int \
                or not 0 <= outcome["delete_index"] < inventory["delete_entry_count"] \
                or outcome.get("row") != inventory["delete_entries"][
                    outcome["delete_index"]
                ] \
                or outcome.get("outcome") not in {
                    "RECONCILED_ABSENT_AFTER_DURABLE_PREPARE",
                    "PREPARED_TARGET_REMAINS_EXACT_NOT_DELETED",
                    "PREPARED_TARGET_MISMATCH_FAIL_CLOSED",
                } \
                or outcome.get("certainty") not in {
                    "PATH_ABSENCE_CONFIRMED_ORIGINAL_INODE_UNLINK_INFERRED",
                    "EXACT_TARGET_CONFIRMED_PRESENT",
                    "PREPARED_TARGET_STATE_MISMATCH_UNRESOLVED",
                } \
                or not isinstance(outcome.get("journal_record_seal_sha256"), str) \
                or HEX64_RE.fullmatch(outcome["journal_record_seal_sha256"]) is None:
            _fail("Phase-2 receipt intent outcome is malformed")
        committed_outcome = (
            outcome["outcome"] == "RECONCILED_ABSENT_AFTER_DURABLE_PREPARE"
        )
        expected_certainty = (
            "PATH_ABSENCE_CONFIRMED_ORIGINAL_INODE_UNLINK_INFERRED"
            if committed_outcome
            else (
                "EXACT_TARGET_CONFIRMED_PRESENT"
                if outcome["outcome"]
                == "PREPARED_TARGET_REMAINS_EXACT_NOT_DELETED"
                else "PREPARED_TARGET_STATE_MISMATCH_UNRESOLVED"
            )
        )
        if outcome["certainty"] != expected_certainty:
            _fail("Phase-2 receipt intent certainty contradicts its outcome")
        if committed_outcome is not (outcome["delete_index"] < completed_count):
            _fail("Phase-2 receipt intent outcome contradicts completed progress")
    if preserved.get("status") == "PASS" and (
        value.get("post_release_capsule_seal_sha256")
        != bound["capsule_verification_seal_sha256"]
        or value.get("post_release_capsule_verified") is not True
        or value.get("mop_touched") is not False
        or value.get("shared_xet_touched") is not False
        or value.get("capsule_retained") is not True
        or value.get("recovery_retained") is not True
        or value.get("evidence_retained") is not True
    ):
        _fail("Phase-2 receipt contradicts its passing preserved-node status")
    if terminal_status == "SUCCESS":
        if completed_count != inventory["delete_entry_count"] \
                or value.get("exception") is not None \
                or preserved.get("status") != "PASS" \
                or value.get("deletion_performed") is not True:
            _fail("successful Phase-2 receipt is not terminally complete")
    else:
        exception = value.get("exception")
        if not isinstance(exception, dict) \
                or set(exception) != {"type", "message"} \
                or value.get("deletion_performed") is not (completed_count > 0):
            _fail("partial Phase-2 receipt does not identify its failure")
    return value


def reconcile_release(
    layout: phase1.SessionLayout,
    bundle: Mapping[str, Any],
    *,
    confirmation_token: str,
    capsule_verifier: Verifier,
    lease_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Reconcile a durable terminal/interrupted journal without deleting."""
    value = verify_bundle(bundle)
    if value.get("status") != "PASS_CONFIRMATION_REQUIRED" \
            or value.get("session") != os.fspath(layout.session):
        _fail("only the exact passing session bundle can be reconciled")
    expected_token = derive_confirmation_token(value["audit"])
    if not isinstance(confirmation_token, str) \
            or not __import__("hmac").compare_digest(confirmation_token, expected_token):
        _fail("reconciliation confirmation token does not match the sealed audit")
    leases = tuple(lease_paths) if lease_paths is not None else default_lease_paths(layout)
    journal_path, _receipt_path = _attempt_artifacts(layout, value["seal_sha256"])
    with _exclusive_release_leases(layout, leases) as lease_holds:
        if not _private_leaf_exists(journal_path.parent, journal_path.name):
            _fail("no durable Phase-2 journal exists for this sealed bundle")
        return _reconcile_existing_attempt(
            layout,
            value,
            confirmation_token,
            capsule_verifier=capsule_verifier,
            lease_holds=lease_holds,
        )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = phase1._read_regular_bytes(  # noqa: SLF001
        path,
        label=label,
        maximum_bytes=MAX_JSON_BYTES,
        expected_uid=os.getuid(),
    )
    return phase1.strict_json_bytes(raw, label=label)


def _load_official_manifest() -> dict[str, Any]:
    raw = phase1._read_regular_bytes(  # noqa: SLF001
        phase1.OFFICIAL_MANIFEST,
        label="official Kimi manifest",
        maximum_bytes=1_000_000,
        expected_uid=os.getuid(),
    )
    return phase1._manifest_from_bytes(raw, label="official Kimi manifest")  # noqa: SLF001


def _print(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(phase1.canonical_json(dict(value)) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="build sealed read-only release bundle")
    audit.add_argument("--session", type=Path, required=True)
    token = sub.add_parser("confirm-token", help="derive explicit token from a passing bundle")
    token.add_argument("--bundle", type=Path, required=True)
    execute = sub.add_parser("execute", help="execute exact release after explicit confirmation")
    execute.add_argument("--session", type=Path, required=True)
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--confirm", required=True)
    reconcile = sub.add_parser(
        "reconcile", help="reconcile a durable journal without any new deletion"
    )
    reconcile.add_argument("--session", type=Path, required=True)
    reconcile.add_argument("--bundle", type=Path, required=True)
    reconcile.add_argument("--confirm", required=True)
    verify = sub.add_parser("verify-receipt", help="verify a sealed receipt against its bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            layout = phase1.layout_for(args.session)
            phase1.validate_layout(layout)
            manifest = _load_official_manifest()
            source = phase1.verify_source(layout)
            capsule = phase1.verify_payload_result_capture(layout)
            value = build_release_bundle(
                layout,
                manifest,
                source,
                capsule,
                probe=SystemAuditProbe(),
            )
        elif args.command == "confirm-token":
            bundle = verify_bundle(_read_json(args.bundle, label="Phase-2 bundle"))
            value = phase1.seal_document(
                {
                    "schema": CONFIRMATION_SCHEMA,
                    "status": "EXPLICIT_CONFIRMATION_TOKEN_DERIVED",
                    "audit_seal_sha256": bundle["audit_seal_sha256"],
                    "confirmation_token": derive_confirmation_token(bundle["audit"]),
                }
            )
        elif args.command == "execute":
            layout = phase1.layout_for(args.session)
            phase1.validate_layout(layout)
            bundle = _read_json(args.bundle, label="Phase-2 bundle")
            manifest = _load_official_manifest()
            value = execute_release(
                layout,
                bundle,
                confirmation_token=args.confirm,
                manifest=manifest,
                source_verifier=lambda candidate: phase1.verify_source(candidate),
                capsule_verifier=lambda candidate: phase1.verify_payload_result_capture(candidate),
                probe=SystemAuditProbe(),
            )
        elif args.command == "reconcile":
            layout = phase1.layout_for(args.session)
            phase1.validate_layout(layout)
            bundle = _read_json(args.bundle, label="Phase-2 bundle")
            value = reconcile_release(
                layout,
                bundle,
                confirmation_token=args.confirm,
                capsule_verifier=lambda candidate: phase1.verify_payload_result_capture(
                    candidate
                ),
            )
        else:
            bundle = _read_json(args.bundle, label="Phase-2 bundle")
            receipt = _read_json(args.receipt, label="Phase-2 receipt")
            value = verify_receipt(receipt, bundle)
        _print(value)
        if value.get("status") == "PARTIAL_FAILURE_EXACT_PHASE2_SOURCE_RELEASE":
            return 3
        return 0 if value.get("status") != "BLOCKED" else 2
    except Phase2PartialReleaseError as exc:
        _print(exc.receipt)
        return 3
    except (OSError, subprocess.SubprocessError, phase1.ReleaseCycleError, Phase2ReleaseError) as exc:
        execution_may_be_partial = args.command == "execute"
        _print(
            phase1.seal_document(
                {
                    "schema": "hawking.kimi_k26.phase2_release.error.v1",
                    "status": "BLOCKED",
                    "error": f"{type(exc).__name__}: {exc}"[:2_000],
                    "live_release_completed": None if execution_may_be_partial else False,
                    "execution_state": (
                        "UNKNOWN_PARTIAL_RELEASE_POSSIBLE_CHECK_EXACT_INVENTORY"
                        if execution_may_be_partial
                        else "NO_RELEASE_ATTEMPTED"
                    ),
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuditProbe",
    "Phase2PartialReleaseError",
    "Phase2ReleaseError",
    "SystemAuditProbe",
    "build_exact_inventory",
    "build_release_audit",
    "build_release_bundle",
    "derive_confirmation_token",
    "execute_release",
    "reconcile_release",
    "verify_bundle",
    "verify_inventory",
    "verify_receipt",
]
