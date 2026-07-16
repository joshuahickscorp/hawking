#!/usr/bin/env python3.12
"""Fail-closed builders for the deferred Appendix physical release packet.

This module is the production-side complement to
``appendix_physical_evidence_gate``.  It owns the provenance artifacts that the
aggregate gate intentionally only validates:

* a Doctor-final, owner-free release-boundary observation and attestation;
* an exact clean-worktree source manifest;
* an exact two-probe release-build receipt;
* an owner-free corpus freeze and byte-for-byte verification attestation; and
* deterministic evidence manifests and aggregate packet assembly.

Every command that reads corpus bytes, hashes release inputs, builds Rust, or
assembles physical results first acquires the same exclusive heavy lease used by
the TQ runners.  Admission is rechecked under that lease.  Status, dry-run, and
self-tests are deliberately non-executing and are safe while Doctor owns the
machine.  Nothing in this module changes a runtime default.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import shlex
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import appendix_contract
import appendix_corpus
import appendix_physical_evidence_gate as evidence_gate
import physical_counter_attestation
import ram_scheduler
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
POST_ROOT = ROOT / "reports" / "condense" / "doctor_v5_ultra" / "post_120b"
OBSERVER_PATH = POST_ROOT / "observer_state.json"
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
REPORT_ROOT = ROOT / "reports" / "appendix" / "physical_release"

BOUNDARY_OBSERVATION_SCHEMA = "hawking.appendix_release_boundary_observation.v1"
CORPUS_RECEIPT_SCHEMA = "hawking.appendix_corpus_verification_receipt.v2"
EVIDENCE_MANIFEST_SCHEMA = "hawking.appendix_physical_evidence_manifest.v1"
ASSEMBLY_RECEIPT_SCHEMA = "hawking.appendix_physical_packet_assembly_receipt.v1"
STATUS_SCHEMA = "hawking.appendix_physical_packet_builder_status.v1"
PREPARED_RELEASE_SCHEMA = "hawking.appendix_prepared_release.v1"

FINAL_PACKET_SCHEMA = "hawking.doctor_v5_final_interpretation_packet.v1"
OBSERVER_SCHEMA = "hawking.doctor_v5_post_120b_observer_state.v1"
HEX64 = evidence_gate.HEX64
BUILD_CONTEXT_SCHEMA = "hawking.appendix_release_build_context.v1"
DETERMINISTIC_BUILD_ENVIRONMENT_KEYS = frozenset({
    "CARGO_HOME",
    "CARGO_INCREMENTAL",
    "CARGO_NET_OFFLINE",
    "CARGO_TARGET_DIR",
    "CARGO_TERM_COLOR",
    "HOME",
    ram_scheduler.HEAVY_LEASE_FD_ENV,
    "LANG",
    "LC_ALL",
    "PATH",
    "RUSTC",
    "RUSTUP_HOME",
    "SOURCE_DATE_EPOCH",
    "TERM",
    "TMPDIR",
    "TZ",
    "ZERO_AR_DATE",
})
RUNTIME_ORDER = {name: index for index, name in enumerate(evidence_gate.RUNTIME_PATHS)}


class ReleaseBlocked(RuntimeError):
    """The immutable release boundary is not currently open."""


class EvidenceError(ValueError):
    """An input cannot support a physical-evidence claim."""


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def _hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not isinstance(claimed, str) or HEX64.fullmatch(claimed) is None:
        return [f"{label}.{field} is invalid"]
    if claimed != canonical_sha256(unstamped):
        return [f"{label}.{field} mismatch"]
    return []


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    _atomic_json_group(((path, value),))


def _atomic_json_group(rows: Sequence[tuple[pathlib.Path, Any]]) -> None:
    """Install a set of immutable JSON files without replacing any prior byte."""
    _atomic_bytes_group(tuple((path, _json_bytes(value)) for path, value in rows))


def _lexical_absolute(path: pathlib.Path) -> pathlib.Path:
    """Return a normalized absolute path without following any symlink."""
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _open_dir_nofollow(path: pathlib.Path, *, create: bool = False) -> int:
    """Open a directory by walking from ``/`` with retained, no-follow dirfds."""
    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"immutable evidence output is not regular: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise EvidenceError(f"immutable evidence output changed while reading: {name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_immutable_regular_at(
    directory_fd: int, name: str, *, expected: bytes,
) -> tuple[int, int]:
    """Prove one dirfd-relative output is byte-exact and non-writable.

    Content equality alone is not an immutable seal: a pre-existing matching
    file may still be owner-writable, and a freshly linked 0600 temporary would
    otherwise remain mutable after this function returned.  Keep the check
    relative to the retained directory descriptor so a pathname replacement
    cannot redirect it.
    """
    observed = _read_regular_at(directory_fd, name)
    if observed != expected:
        raise EvidenceError(f"immutable evidence output changed after install: {name}")
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode):
        raise EvidenceError(f"immutable evidence output is not regular: {name}")
    if stat.S_IMODE(current.st_mode) & 0o222:
        raise EvidenceError(f"immutable evidence output remains writable: {name}")
    return current.st_dev, current.st_ino


def _atomic_bytes_group(rows: Sequence[tuple[pathlib.Path, bytes]]) -> None:
    """Install an immutable group through retained no-follow directory handles.

    All existence checks, temporary creation, hard-link installation, fsyncs, and
    rollback happen relative to retained dirfds.  Replacing a parent directory or
    inserting a symlink after validation therefore cannot redirect evidence.
    """
    normalized = [(_lexical_absolute(path), raw) for path, raw in rows]
    if len({path for path, _raw in normalized}) != len(rows):
        raise EvidenceError("immutable output paths must be distinct")
    parents: dict[pathlib.Path, int] = {}
    entries: list[tuple[pathlib.Path, bytes, int, str]] = []
    temporaries: list[tuple[int, str]] = []
    installed: list[tuple[int, str, tuple[int, int]]] = []
    try:
        for path, raw in normalized:
            if path.parent not in parents:
                parents[path.parent] = _open_dir_nofollow(path.parent, create=True)
            directory_fd = parents[path.parent]
            entries.append((path, raw, directory_fd, path.name))
        missing: list[tuple[pathlib.Path, bytes, int, str]] = []
        for path, raw, directory_fd, name in entries:
            try:
                observed = _read_regular_at(directory_fd, name)
            except FileNotFoundError:
                missing.append((path, raw, directory_fd, name))
            else:
                if observed != raw:
                    raise EvidenceError(f"refusing to replace different immutable evidence: {path}")
                _require_immutable_regular_at(directory_fd, name, expected=raw)
        prepared: list[tuple[pathlib.Path, bytes, int, str, str]] = []
        for ordinal, (path, raw, directory_fd, name) in enumerate(missing):
            temporary = f".{name}.{os.getpid()}.{ordinal}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
            temporaries.append((directory_fd, temporary))
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while preparing immutable evidence")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            prepared.append((path, raw, directory_fd, name, temporary))
        for path, raw, directory_fd, name, temporary in prepared:
            try:
                os.link(
                    temporary, name, src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd, follow_symlinks=False,
                )
            except FileExistsError:
                if _read_regular_at(directory_fd, name) != raw:
                    raise EvidenceError(f"concurrent writer claimed immutable output: {path}")
                _require_immutable_regular_at(directory_fd, name, expected=raw)
            else:
                installed_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                installed.append((directory_fd, name, (installed_stat.st_dev, installed_stat.st_ino)))
                os.chmod(name, 0o400, dir_fd=directory_fd, follow_symlinks=False)
                _require_immutable_regular_at(directory_fd, name, expected=raw)
        for parent, descriptor in parents.items():
            try:
                current_descriptor = _open_dir_nofollow(parent)
            except OSError as exc:
                raise EvidenceError(
                    f"immutable output parent changed or became unsafe: {parent}: {exc}"
                ) from exc
            try:
                retained = os.fstat(descriptor)
                current = os.fstat(current_descriptor)
                if (retained.st_dev, retained.st_ino) != (current.st_dev, current.st_ino):
                    raise EvidenceError(f"immutable output parent was replaced: {parent}")
            finally:
                os.close(current_descriptor)
            os.fsync(descriptor)
    except BaseException:
        for directory_fd, name, installed_id in reversed(installed):
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == installed_id:
                    os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        for directory_fd, temporary in temporaries:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        for descriptor in parents.values():
            os.close(descriptor)


def _immutable_text(path: pathlib.Path, value: str) -> None:
    _atomic_bytes_group(((path, value.encode("utf-8")),))


def _byte_binding(path: pathlib.Path, raw: bytes) -> dict[str, Any]:
    if not raw:
        raise EvidenceError(f"cannot bind empty evidence bytes: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _stable_file_identity(path: pathlib.Path, *, require_executable: bool = False) -> dict[str, Any]:
    """Hash one regular, non-symlink file and reject an in-flight rewrite."""
    lexical = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise EvidenceError(f"symlink is not an immutable evidence file: {path}") from exc
        raise
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise EvidenceError(f"evidence file is absent, empty, or non-regular: {path}")
        if require_executable and before.st_mode & 0o111 == 0:
            raise EvidenceError(f"release probe is not executable: {path}")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(lexical, follow_symlinks=False)
    finally:
        os.close(descriptor)
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_id = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if before_id != after_id or before_id != current_id:
        raise EvidenceError(f"evidence file changed/replaced while hashing: {path}")
    return {"path": str(lexical), "sha256": digest.hexdigest(), "size_bytes": before.st_size}


def _raw_identity(path: pathlib.Path) -> tuple[str, int]:
    identity = _stable_file_identity(path)
    return identity["sha256"], identity["size_bytes"]


def _load_bound_json(path: pathlib.Path, binding: dict[str, Any], *, label: str) -> Any:
    lexical = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(lexical, follow_symlinks=False)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
            or identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
        raise EvidenceError(f"{label} changed/replaced while loading")
    if len(raw) != binding.get("bytes", binding.get("size_bytes")) \
            or hashlib.sha256(raw).hexdigest() != binding.get("sha256"):
        raise EvidenceError(f"{label} byte identity mismatch while loading")
    try:
        return json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not valid JSON: {exc}") from exc


def _safe_resolve(base: pathlib.Path, raw: Any, *, label: str) -> pathlib.Path:
    if not isinstance(raw, str) or not raw:
        raise EvidenceError(f"{label} path is invalid")
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError(f"{label} path escapes its declared base")
    base_resolved = base.resolve(strict=True)
    candidate = base_resolved
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise EvidenceError(f"{label} path contains a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(base_resolved)
    except ValueError as exc:
        raise EvidenceError(f"{label} path escapes its declared base") from exc
    return path


def _verify_reference(reference: Any, *, base: pathlib.Path, label: str) -> dict[str, Any]:
    if not isinstance(reference, dict):
        raise EvidenceError(f"{label} reference must be an object")
    if not {"path", "sha256", "bytes"}.issubset(reference):
        raise EvidenceError(f"{label} reference is incomplete")
    path = _safe_resolve(base, reference.get("path"), label=label)
    digest, size = _raw_identity(path)
    if digest != reference.get("sha256") or size != reference.get("bytes"):
        raise EvidenceError(f"{label} byte identity mismatch")
    return {"label": label, "path": str(path), "sha256": digest, "bytes": size}


def _observer_errors(observer: Any) -> list[str]:
    if not isinstance(observer, dict):
        return ["Doctor observer state is missing"]
    errors: list[str] = []
    if observer.get("schema") != OBSERVER_SCHEMA:
        errors.append("Doctor observer schema is invalid")
    errors.extend(_hash_errors(observer, "state_sha256", label="observer_state"))
    if observer.get("final_interpretation_ready") is not True:
        errors.append("Doctor final_interpretation_ready is false")
    if observer.get("source_deletion_permitted") is not False:
        errors.append("Doctor observer weakened source preservation")
    if not isinstance(observer.get("final_interpretation_packet"), dict):
        errors.append("Doctor final interpretation packet reference is absent")
    return errors


def verify_final_packet_references(
    observer: Any, *, post_root: pathlib.Path = POST_ROOT,
    workspace_root: pathlib.Path = ROOT,
) -> tuple[list[str], dict[str, Any] | None]:
    """Verify every file identity recorded by the Doctor final packet.

    This function can read a large completed corpus and therefore is called by
    production only after the heavy lease is held.  Tests may point it at a
    synthetic ``post_root``.
    """
    errors = _observer_errors(observer)
    if errors:
        return errors, None
    assert isinstance(observer, dict)
    references: list[dict[str, Any]] = []
    try:
        packet_ref = observer["final_interpretation_packet"]
        verified_packet_ref = _verify_reference(
            packet_ref, base=post_root, label="final_interpretation_packet",
        )
        references.append(verified_packet_ref)
        packet_path = pathlib.Path(verified_packet_ref["path"])
        packet = _load_bound_json(
            packet_path, verified_packet_ref, label="final_interpretation_packet",
        )
        if not isinstance(packet, dict) or packet.get("schema") != FINAL_PACKET_SCHEMA:
            raise EvidenceError("Doctor final packet schema is invalid")
        if packet.get("ready") is not True or packet.get("source_deletion_permitted") is not False:
            raise EvidenceError("Doctor final packet is not ready/source-preserving")
        errors.extend(_hash_errors(packet, "packet_sha256", label="final_packet"))
        bases_raw = packet.get("artifact_path_bases")
        if not isinstance(bases_raw, dict):
            raise EvidenceError("Doctor final packet artifact bases are absent")
        expected_bases = {"source_observation", "frozen_inputs", "reports_and_checkpoints"}
        if set(bases_raw) != expected_bases:
            raise EvidenceError("Doctor final packet artifact bases are incomplete or unexpected")
        bases: dict[str, pathlib.Path] = {}
        required_bases = {
            "source_observation": post_root.resolve(strict=True),
            "frozen_inputs": post_root.resolve(strict=True),
            "reports_and_checkpoints": workspace_root.resolve(strict=True),
        }
        for name, raw in bases_raw.items():
            if not isinstance(raw, str):
                raise EvidenceError(f"Doctor final packet base {name} is invalid")
            path = pathlib.Path(raw)
            if not path.is_absolute() or not path.is_dir() or path.is_symlink():
                raise EvidenceError(f"Doctor final packet base {name} is unsafe")
            bases[name] = path.resolve(strict=True)
            if bases[name] != required_bases[name]:
                raise EvidenceError(f"Doctor final packet base {name} is not the exact trusted base")

        references.append(_verify_reference(
            packet.get("source_observation"), base=bases["source_observation"],
            label="source_observation",
        ))
        frozen = packet.get("frozen_inputs")
        if not isinstance(frozen, dict) or set(frozen) != {"campaign_plan", "campaign", "report_index"}:
            raise EvidenceError("Doctor final frozen-input set is incomplete")
        frozen_paths: dict[str, pathlib.Path] = {}
        for name in sorted(frozen):
            row = _verify_reference(
                frozen[name], base=bases["frozen_inputs"], label=f"frozen_inputs.{name}",
            )
            references.append(row)
            frozen_paths[name] = pathlib.Path(row["path"])
        frozen_plan_ref = next(
            row for row in references if row["label"] == "frozen_inputs.campaign_plan"
        )
        frozen_campaign_ref = next(
            row for row in references if row["label"] == "frozen_inputs.campaign"
        )
        frozen_plan = _load_bound_json(
            frozen_paths["campaign_plan"], frozen_plan_ref,
            label="frozen_inputs.campaign_plan",
        )
        frozen_campaign = _load_bound_json(
            frozen_paths["campaign"], frozen_campaign_ref,
            label="frozen_inputs.campaign",
        )
        if not isinstance(frozen_plan, dict) or frozen_plan.get("plan_sha256") != packet.get("plan_sha256"):
            raise EvidenceError("Doctor final packet plan hash differs from frozen input")
        if not isinstance(frozen_campaign, dict) or frozen_campaign.get("campaign_sha256") != packet.get("campaign_sha256"):
            raise EvidenceError("Doctor final packet campaign hash differs from frozen input")

        reports = packet.get("reports")
        if not isinstance(reports, dict) or set(reports) != {"sub-120B", "120B"}:
            raise EvidenceError("Doctor final report set is incomplete")
        verified_reports: dict[str, dict[str, Any]] = {}
        for group in ("sub-120B", "120B"):
            report = reports[group]
            if not isinstance(report, dict) or report.get("complete") is not True:
                raise EvidenceError(f"Doctor final {group} report is not complete")
            row = _verify_reference(
                report, base=bases["reports_and_checkpoints"], label=f"reports.{group}",
            )
            references.append(row)
            verified_reports[group] = row

        checkpoints = packet.get("accepted_report_checkpoints")
        groups = packet.get("accepted_report_checkpoint_groups")
        if groups != ["120B", "sub-120B"] or not isinstance(checkpoints, dict) \
                or set(checkpoints) != {"sub-120B", "120B"}:
            raise EvidenceError("Doctor final checkpoint set is incomplete")
        for group in ("sub-120B", "120B"):
            checkpoint_ref = checkpoints[group]
            row = _verify_reference(
                checkpoint_ref, base=bases["reports_and_checkpoints"],
                label=f"accepted_report_checkpoints.{group}",
            )
            references.append(row)
            checkpoint = _load_bound_json(
                pathlib.Path(row["path"]), row,
                label=f"accepted_report_checkpoints.{group}",
            )
            if not isinstance(checkpoint, dict):
                raise EvidenceError(f"Doctor {group} checkpoint is not an object")
            errors.extend(_hash_errors(checkpoint, "checkpoint_sha256", label=f"checkpoint.{group}"))
            expected_report = {
                "path": reports[group].get("path"),
                "sha256": reports[group].get("sha256"),
                "bytes": reports[group].get("bytes"),
            }
            if checkpoint.get("verified") is not True \
                    or checkpoint.get("source_deletion_permitted") is not False \
                    or checkpoint.get("report_artifact") != expected_report \
                    or checkpoint_ref.get("checkpoint_sha256") != checkpoint.get("checkpoint_sha256"):
                raise EvidenceError(f"Doctor {group} checkpoint contract is invalid")
        if errors:
            return errors, None
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceError) as exc:
        errors.append(str(exc))
        return errors, None

    reference_digest = canonical_sha256([
        {"label": row["label"], "path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}
        for row in sorted(references, key=lambda value: value["label"])
    ])
    return [], {
        "final_packet": packet,
        "final_packet_file_sha256": verified_packet_ref["sha256"],
        "final_packet_bytes": verified_packet_ref["bytes"],
        "verified_reference_count": len(references),
        "verified_references_sha256": reference_digest,
        "references": sorted(references, key=lambda value: value["label"]),
    }


def _guard_healthy(resource: Any) -> bool:
    return (
        isinstance(resource, dict)
        and resource.get("ok") is True
        and ram_scheduler.classify_resource_state(resource) == "green"
    )


def current_status(
    *, observer: Any | None = None, owners: list[dict[str, Any]] | None = None,
    resource: Any | None = None,
) -> dict[str, Any]:
    """Cheap status projection: never hashes corpus, model, or probe files."""
    if observer is None:
        try:
            observer = _load(OBSERVER_PATH)
        except (OSError, UnicodeError, json.JSONDecodeError):
            observer = None
    owner_rows = spec_reentry_scaffold.active_heavy_owners() if owners is None else owners
    sample = ram_scheduler.resource_snapshot(ROOT) if resource is None else resource
    observer_errors = _observer_errors(observer)
    blockers = [*observer_errors]
    if owner_rows:
        blockers.append(f"{len(owner_rows)} heavy owner(s) remain")
    if not _guard_healthy(sample):
        blockers.append("RAM/swap guard is not green")
    return _stamp({
        "schema": STATUS_SCHEMA,
        "default_off": True,
        "execution_capability": False,
        "opens_or_hashes_corpus": False,
        "runs_cargo_or_probe": False,
        "final_interpretation_ready": (
            isinstance(observer, dict) and observer.get("final_interpretation_ready") is True
        ),
        "observer_structurally_valid": not observer_errors,
        "active_heavy_owner_count": len(owner_rows),
        "ram_swap_guard_healthy": _guard_healthy(sample),
        "prelease_admission_ready": not blockers,
        "shared_heavy_lease_required": True,
        "blockers": blockers,
    }, "status_sha256")


def _recheck_under_lease(
    expected_observer_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    try:
        observer = _load(OBSERVER_PATH)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBlocked(f"Doctor observer cannot be reread under lease: {exc}") from exc
    errors = _observer_errors(observer)
    if errors:
        raise ReleaseBlocked("; ".join(errors))
    if observer.get("state_sha256") != expected_observer_sha256:
        raise ReleaseBlocked("Doctor observer changed during the leased release operation")
    owners = spec_reentry_scaffold.active_heavy_owners()
    if owners:
        raise ReleaseBlocked("heavy owners appeared during the leased release operation")
    resource = ram_scheduler.resource_snapshot(ROOT)
    if not _guard_healthy(resource):
        raise ReleaseBlocked("RAM/swap guard ceased to be green during the leased release operation")
    return observer, owners, resource


@dataclass
class ReleaseAdmission:
    lease: Any
    observer: dict[str, Any]
    boundary_observation: dict[str, Any]
    boundary_attestation: dict[str, Any]


def _build_boundary_documents(
    *, observer: dict[str, Any], verified: dict[str, Any], owners: list[dict[str, Any]],
    resource: dict[str, Any], lock_stat: os.stat_result, observed_at_unix_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    owner_snapshot = {
        "schema": "hawking.appendix_owner_snapshot.v1",
        "exclusive_shared_heavy_lease_held": True,
        "lock_path": str(HEAVY_LOCK),
        "lock_device": lock_stat.st_dev,
        "lock_inode": lock_stat.st_ino,
        "owners": owners,
        "observed_at_unix_ns": observed_at_unix_ns,
    }
    owner_sha = canonical_sha256(owner_snapshot)
    observation = _stamp({
        "schema": BOUNDARY_OBSERVATION_SCHEMA,
        "observer_state_sha256": observer["state_sha256"],
        "final_packet_file_sha256": verified["final_packet_file_sha256"],
        "final_packet_canonical_sha256": verified["final_packet"]["packet_sha256"],
        "all_recorded_hashes_verified": True,
        "verified_reference_count": verified["verified_reference_count"],
        "verified_references_sha256": verified["verified_references_sha256"],
        "owner_snapshot": owner_snapshot,
        "owner_snapshot_sha256": owner_sha,
        "resource_snapshot": resource,
        "ram_swap_guard_healthy": _guard_healthy(resource),
        "observed_at_unix_ns": observed_at_unix_ns,
        "default_mutation_requested": False,
    }, "observation_sha256")
    attestation = _stamp({
        "schema": evidence_gate.RELEASE_BOUNDARY_SCHEMA,
        "final_interpretation_ready": True,
        "final_packet_sha256": verified["final_packet_file_sha256"],
        "observer_state_sha256": observer["state_sha256"],
        "all_recorded_hashes_verified": True,
        "active_heavy_owner_count": len(owners),
        "owner_snapshot_sha256": owner_sha,
        "ram_swap_guard_healthy": _guard_healthy(resource),
        "observed_at_unix_ns": observed_at_unix_ns,
    }, "attestation_sha256")
    return observation, attestation


def validate_release_boundary_attestation(
    attestation: Any, *, observation: Any | None = None,
    observer: Any | None = None,
) -> list[str]:
    errors = evidence_gate._validate_release_boundary(attestation)
    if observation is not None:
        expected_fields = {
            "schema", "observer_state_sha256", "final_packet_file_sha256",
            "final_packet_canonical_sha256", "all_recorded_hashes_verified",
            "verified_reference_count", "verified_references_sha256", "owner_snapshot",
            "owner_snapshot_sha256", "resource_snapshot", "ram_swap_guard_healthy",
            "observed_at_unix_ns", "default_mutation_requested", "observation_sha256",
        }
        if not isinstance(observation, dict) or set(observation) != expected_fields:
            errors.append("release boundary observation is malformed")
        else:
            errors.extend(_hash_errors(
                observation, "observation_sha256", label="release_boundary_observation",
            ))
            snapshot = observation.get("owner_snapshot")
            if not isinstance(snapshot, dict) \
                    or snapshot.get("exclusive_shared_heavy_lease_held") is not True \
                    or snapshot.get("owners") != []:
                errors.append("release boundary observation does not prove an exclusive owner-free lease")
            elif canonical_sha256(snapshot) != observation.get("owner_snapshot_sha256"):
                errors.append("release boundary owner snapshot hash mismatch")
            if not _guard_healthy(observation.get("resource_snapshot")) \
                    or observation.get("ram_swap_guard_healthy") is not True:
                errors.append("release boundary observation RAM/swap guard is not green")
            if observation.get("all_recorded_hashes_verified") is not True \
                    or not isinstance(observation.get("verified_reference_count"), int) \
                    or observation.get("verified_reference_count", 0) < 8 \
                    or not isinstance(observation.get("verified_references_sha256"), str) \
                    or HEX64.fullmatch(observation.get("verified_references_sha256", "")) is None:
                errors.append("release boundary observation did not verify the complete final reference set")
            if observation.get("default_mutation_requested") is not False:
                errors.append("release boundary observation requests a default mutation")
            if isinstance(attestation, dict):
                comparisons = {
                    "observer_state_sha256": observation.get("observer_state_sha256"),
                    "final_packet_sha256": observation.get("final_packet_file_sha256"),
                    "owner_snapshot_sha256": observation.get("owner_snapshot_sha256"),
                    "ram_swap_guard_healthy": observation.get("ram_swap_guard_healthy"),
                    "observed_at_unix_ns": observation.get("observed_at_unix_ns"),
                }
                for field, expected in comparisons.items():
                    if attestation.get(field) != expected:
                        errors.append(f"release boundary attestation is not bound to observation field {field}")
    if observer is not None:
        observer_errors = _observer_errors(observer)
        errors.extend(f"current observer: {error}" for error in observer_errors)
        if isinstance(observer, dict) and isinstance(attestation, dict) \
                and attestation.get("observer_state_sha256") != observer.get("state_sha256"):
            errors.append("release boundary attestation is stale relative to current observer")
    return errors


@contextlib.contextmanager
def release_admission() -> Iterator[ReleaseAdmission]:
    """Acquire the shared heavy lease and re-prove the release boundary under it."""
    status = current_status()
    if not status["prelease_admission_ready"]:
        raise ReleaseBlocked("; ".join(status["blockers"]))
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = HEAVY_LOCK.open("a+")
    try:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseBlocked("shared heavy lease is held") from exc
        try:
            observer = _load(OBSERVER_PATH)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseBlocked(f"Doctor observer cannot be read under lease: {exc}") from exc
        structural = _observer_errors(observer)
        owners = spec_reentry_scaffold.active_heavy_owners()
        resource = ram_scheduler.resource_snapshot(ROOT)
        if structural:
            raise ReleaseBlocked("; ".join(structural))
        if owners:
            raise ReleaseBlocked("heavy owners remain after lease acquisition")
        if not _guard_healthy(resource):
            raise ReleaseBlocked("RAM/swap guard is not green after lease acquisition")
        reference_errors, verified = verify_final_packet_references(observer)
        if reference_errors or verified is None:
            raise ReleaseBlocked(
                "Doctor final packet hash verification failed: " + "; ".join(reference_errors)
            )
        observer, owners, resource = _recheck_under_lease(observer["state_sha256"])
        observed_at = time.time_ns()
        observation, attestation = _build_boundary_documents(
            observer=observer, verified=verified, owners=owners, resource=resource,
            lock_stat=os.fstat(lease.fileno()), observed_at_unix_ns=observed_at,
        )
        boundary_errors = validate_release_boundary_attestation(
            attestation, observation=observation, observer=observer,
        )
        if boundary_errors:
            raise ReleaseBlocked("release boundary construction failed: " + "; ".join(boundary_errors))
        yield ReleaseAdmission(lease, observer, observation, attestation)
    finally:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        finally:
            lease.close()


def _git(*argv: str) -> str:
    process = subprocess.run(
        ["git", *argv], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if process.returncode != 0:
        raise EvidenceError(f"git {' '.join(argv)} failed: {process.stderr.strip()}")
    return process.stdout


def build_clean_source_manifest(
    *, release_boundary: dict[str, Any], boundary_observation: dict[str, Any],
) -> dict[str, Any]:
    """Snapshot the exact release-critical source capsule.

    The user's unrelated dirty/untracked files are intentionally outside this
    claim.  Every in-scope path is hashed stably and the complete capsule is
    compared byte-for-byte before and after the release build.
    """
    commit = _git("rev-parse", "HEAD").strip()
    boundary_errors = validate_release_boundary_attestation(
        release_boundary, observation=boundary_observation,
    )
    if boundary_errors:
        raise EvidenceError("source capsule boundary is invalid: " + "; ".join(boundary_errors))

    def capture() -> list[dict[str, Any]]:
        rows = []
        for relative in sorted(evidence_gate.REQUIRED_SOURCE_PATHS):
            identity = _stable_file_identity(ROOT / relative)
            rows.append({
                "path": relative,
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            })
        return rows

    entries = capture()
    if capture() != entries:
        raise EvidenceError("critical source capsule changed during its two-pass capture")
    manifest = _stamp({
        "schema": evidence_gate.SOURCE_MANIFEST_SCHEMA,
        "source_base_commit": commit,
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "scope": "isolated-exact-critical-source-capsule",
        "release_boundary_attestation_sha256": release_boundary["attestation_sha256"],
        "release_boundary_observation_sha256": boundary_observation["observation_sha256"],
        "required_paths_sha256": canonical_sha256(sorted(evidence_gate.REQUIRED_SOURCE_PATHS)),
        "entry_count": len(entries),
        "symlink_count": 0,
        "entries": entries,
        "capsule_sha256": canonical_sha256(entries),
    }, "manifest_sha256")
    errors = validate_clean_source_manifest(manifest, verify_current=True)
    if errors:
        raise EvidenceError("constructed source manifest is invalid: " + "; ".join(errors))
    return manifest


def validate_clean_source_manifest(
    value: Any, *, release_boundary: dict[str, Any] | None = None,
    boundary_observation: dict[str, Any] | None = None,
    verify_current: bool = False,
) -> list[str]:
    errors = evidence_gate._validate_source_manifest(value, verify_files=verify_current)
    if not isinstance(value, dict):
        return errors
    entries = value.get("entries")
    paths = {
        row.get("path") for row in entries if isinstance(row, dict)
    } if isinstance(entries, list) else set()
    if paths != evidence_gate.REQUIRED_SOURCE_PATHS:
        errors.append("source capsule must contain exactly the required release sources")
    if release_boundary is not None:
        if value.get("release_boundary_attestation_sha256") != release_boundary.get("attestation_sha256"):
            errors.append("source capsule is not bound to the exact release boundary")
    if boundary_observation is not None:
        if value.get("release_boundary_observation_sha256") != boundary_observation.get("observation_sha256"):
            errors.append("source capsule is not bound to the exact boundary observation")
    if verify_current:
        try:
            commit = _git("rev-parse", "HEAD").strip()
        except EvidenceError as exc:
            errors.append(str(exc))
        else:
            if value.get("source_base_commit") != commit:
                errors.append("source capsule base commit differs from current HEAD")
    return errors


def _unique_target_dir(
    release_boundary: dict[str, Any], source_manifest: dict[str, Any],
) -> pathlib.Path:
    digest = canonical_sha256({
        "release_boundary_attestation_sha256": release_boundary["attestation_sha256"],
        "source_capsule_sha256": source_manifest["capsule_sha256"],
    })
    return (ROOT / "target" / "appendix-release" / digest).resolve()


def _exact_build_argv(
    cargo_path: str, target_host: str, target_directory: pathlib.Path,
) -> list[str]:
    return [
        cargo_path, "build", "--locked", "--release", "--target", target_host,
        "--target-dir", str(target_directory),
        "-p", "hawking", "--features", "tq", "--bin",
        "hawking-tq-device-probe", "--bin", "hawking-tq-spec-probe",
        "--message-format=json-render-diagnostics",
    ]


def _selection_environment() -> dict[str, str]:
    home = pathlib.Path.home().resolve(strict=True)
    cargo_home = pathlib.Path(
        os.environ.get("CARGO_HOME", str(home / ".cargo"))
    ).expanduser().resolve(strict=True)
    rustup_home = _lexical_absolute(pathlib.Path(
        os.environ.get("RUSTUP_HOME", str(home / ".rustup"))
    ).expanduser())
    return {
        "CARGO_HOME": str(cargo_home),
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "RUSTUP_HOME": str(rustup_home),
        "TZ": "UTC",
    }


def _tool_binding(name: str) -> dict[str, Any]:
    invocation = shutil.which(name)
    if invocation is None:
        raise EvidenceError(f"required release tool is absent from PATH: {name}")
    discovered_path = pathlib.Path(invocation).absolute()
    discovered_resolved = discovered_path.resolve(strict=True)
    selection_environment = _selection_environment()
    selector_argv: list[str] | None = None
    selected = discovered_resolved
    selection_mode = "direct"
    if discovered_resolved.name == "rustup":
        selector_argv = [str(discovered_resolved), "which", name]
        selection = subprocess.run(
            selector_argv, cwd=ROOT, env=selection_environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
        )
        if selection.returncode != 0 or len(selection.stdout.splitlines()) != 1:
            raise EvidenceError(f"rustup could not select exact {name}: {selection.stderr.strip()}")
        selected = pathlib.Path(selection.stdout.strip()).resolve(strict=True)
        selection_mode = "rustup-which-to-direct-binary"
    if selected.is_symlink() or not selected.is_file() or not os.access(selected, os.X_OK):
        raise EvidenceError(f"selected {name} tool is not an executable regular file")
    invocation_path = selected
    version_argv = [str(invocation_path), "-Vv"] if name == "cargo" else [str(invocation_path), "-vV"]
    version_environment = {
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin", "TZ": "UTC",
    }
    process = subprocess.run(
        version_argv, cwd=ROOT, env=version_environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30, check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        raise EvidenceError(f"{name} verbose version probe failed: {process.stderr.strip()}")
    return {
        "invocation_path": str(invocation_path),
        "resolved_binary": _stable_file_identity(selected),
        "version_verbose": process.stdout,
        "version_verbose_sha256": canonical_sha256(process.stdout),
        "selection": {
            "mode": selection_mode,
            "discovered_invocation_path": str(discovered_path),
            "discovered_resolved_binary": _stable_file_identity(discovered_resolved),
            "selection_environment": selection_environment,
            "selector_argv_sha256": (
                canonical_sha256(selector_argv) if selector_argv is not None else None
            ),
            "selected_invocation_path": str(invocation_path),
            "version_probe_environment": version_environment,
            "version_probe_argv_sha256": canonical_sha256(version_argv),
        },
    }


def _rustc_host(toolchain: dict[str, Any]) -> str:
    version = toolchain.get("rustc", {}).get("version_verbose", "")
    hosts = [line.removeprefix("host: ").strip() for line in version.splitlines() if line.startswith("host: ")]
    if len(hosts) != 1 or not hosts[0]:
        raise EvidenceError("bound rustc -vV output has no unique host triple")
    return hosts[0]


def _build_environment(
    toolchain: dict[str, Any], target_directory: pathlib.Path,
    *, cargo_home: pathlib.Path | None = None, lease_fd: int = 0,
) -> dict[str, str]:
    if cargo_home is None:
        cargo_home = pathlib.Path(_selection_environment()["CARGO_HOME"])
    path_directories: list[str] = []
    for candidate in (
        pathlib.Path(toolchain["cargo"]["invocation_path"]).parent,
        pathlib.Path(toolchain["rustc"]["invocation_path"]).parent,
        pathlib.Path("/usr/bin"), pathlib.Path("/bin"),
    ):
        try:
            rendered = str(candidate.resolve(strict=True))
        except OSError:
            # Only synthetic receipt construction reaches this branch; production
            # tool bindings are required to resolve to executable regular files.
            rendered = str(candidate.absolute())
        if rendered not in path_directories:
            path_directories.append(rendered)
    environment = {
        "CARGO_HOME": str(cargo_home.resolve(strict=True)),
        "CARGO_INCREMENTAL": "0",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TARGET_DIR": str(target_directory),
        "CARGO_TERM_COLOR": "never",
        "HOME": str(target_directory / ".isolated-home"),
        ram_scheduler.HEAVY_LEASE_FD_ENV: str(lease_fd),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.pathsep.join(path_directories),
        "RUSTC": toolchain["rustc"]["invocation_path"],
        "RUSTUP_HOME": str(target_directory / ".isolated-rustup-home"),
        "SOURCE_DATE_EPOCH": "1",
        "TERM": "dumb",
        "TMPDIR": str(target_directory / ".isolated-tmp"),
        "TZ": "UTC",
        "ZERO_AR_DATE": "1",
    }
    if set(environment) != DETERMINISTIC_BUILD_ENVIRONMENT_KEYS:
        raise AssertionError("deterministic release-build environment key drift")
    return environment


def _directory_identity(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink():
        raise EvidenceError(f"build context directory is a symlink: {path}")
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(value.st_mode):
        raise EvidenceError(f"build context directory is not a directory: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": stat.S_IMODE(value.st_mode),
    }


def _path_snapshot(value: str) -> dict[str, Any]:
    directories: list[dict[str, Any]] = []
    for raw in value.split(os.pathsep):
        path = pathlib.Path(raw)
        identity = _directory_identity(path)
        entries: list[dict[str, Any]] = []
        for entry in sorted(path.iterdir(), key=lambda candidate: candidate.name):
            try:
                executable = entry.is_file() and os.access(entry, os.X_OK)
            except OSError as exc:
                raise EvidenceError(f"cannot census PATH entry {entry}: {exc}") from exc
            if not executable:
                continue
            resolved = entry.resolve(strict=True)
            entries.append({
                "name": entry.name,
                "link_target": os.readlink(entry) if entry.is_symlink() else None,
                "resolved_binary": _stable_file_identity(resolved),
            })
        directories.append({
            **identity,
            "executable_entries": entries,
            "executable_entries_sha256": canonical_sha256(entries),
        })
    return {
        "value": value,
        "directories": directories,
        "snapshot_sha256": canonical_sha256(directories),
    }


def _configuration_candidates(cargo_home: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    rows: list[tuple[str, pathlib.Path]] = []
    current = ROOT.resolve(strict=True)
    while True:
        rows.extend((
            ("cargo", current / ".cargo" / "config.toml"),
            ("cargo", current / ".cargo" / "config"),
            ("rustup", current / "rust-toolchain.toml"),
            ("rustup", current / "rust-toolchain"),
        ))
        if current.parent == current:
            break
        current = current.parent
    rows.extend((
        ("cargo", cargo_home / "config.toml"),
        ("cargo", cargo_home / "config"),
    ))
    deduplicated: dict[str, tuple[str, pathlib.Path]] = {}
    for kind, path in rows:
        deduplicated[str(_lexical_absolute(path))] = (kind, _lexical_absolute(path))
    return [deduplicated[key] for key in sorted(deduplicated)]


def _configuration_snapshot(cargo_home: pathlib.Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for kind, path in _configuration_candidates(cargo_home):
        if path.is_symlink():
            raise EvidenceError(f"release build configuration candidate is a symlink: {path}")
        if path.exists():
            binding: dict[str, Any] | None = _stable_file_identity(path)
        else:
            binding = None
        entries.append({"kind": kind, "path": str(path), "binding": binding})
    return {"entries": entries, "snapshot_sha256": canonical_sha256(entries)}


def _create_isolated_build_directories(target_directory: pathlib.Path) -> None:
    expected_root = _lexical_absolute(ROOT / "target" / "appendix-release")
    target = _lexical_absolute(target_directory)
    try:
        target.relative_to(expected_root)
    except ValueError as exc:
        raise EvidenceError("release target escapes its exact unbound target root") from exc
    parent_fd = _open_dir_nofollow(target.parent, create=True)
    try:
        try:
            os.mkdir(target.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise EvidenceError(
                "unique boundary+capsule Cargo target already exists; cached target credit is forbidden"
            ) from exc
        target_fd = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            for name in (".isolated-home", ".isolated-rustup-home", ".isolated-tmp"):
                os.mkdir(name, 0o700, dir_fd=target_fd)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _build_execution_context(
    environment: dict[str, str], *, toolchain: dict[str, Any],
) -> dict[str, Any]:
    cargo_home = pathlib.Path(environment["CARGO_HOME"])
    isolated = {
        name: _directory_identity(pathlib.Path(environment[name]))
        for name in ("HOME", "RUSTUP_HOME", "TMPDIR")
    }
    context = _stamp({
        "schema": BUILD_CONTEXT_SCHEMA,
        "environment_keys": sorted(environment),
        "environment_sha256": canonical_sha256(environment),
        "path_snapshot": _path_snapshot(environment["PATH"]),
        "configuration_snapshot": _configuration_snapshot(cargo_home),
        "cargo_home": _directory_identity(cargo_home),
        "isolated_directories": isolated,
        "toolchain_selection_sha256": canonical_sha256({
            name: toolchain[name]["selection"] for name in ("cargo", "rustc")
        }),
        "ambient_environment_inherited": False,
    }, "context_sha256")
    return context


def _validate_build_execution_context_live(
    context: Any, *, environment: dict[str, str], toolchain: dict[str, Any],
) -> list[str]:
    expected = {
        "schema", "environment_keys", "environment_sha256", "path_snapshot",
        "configuration_snapshot", "cargo_home", "isolated_directories",
        "toolchain_selection_sha256", "ambient_environment_inherited", "context_sha256",
    }
    if not isinstance(context, dict) or set(context) != expected:
        return ["release build execution context is malformed"]
    errors = _hash_errors(context, "context_sha256", label="release_build_context")
    if context.get("schema") != BUILD_CONTEXT_SCHEMA:
        errors.append("release build execution context schema is invalid")
    if set(environment) != DETERMINISTIC_BUILD_ENVIRONMENT_KEYS \
            or context.get("environment_keys") != sorted(environment) \
            or context.get("environment_sha256") != canonical_sha256(environment):
        errors.append("release build execution context does not bind the exact environment")
    if context.get("ambient_environment_inherited") is not False:
        errors.append("release build execution context inherited ambient environment")
    try:
        if context.get("path_snapshot") != _path_snapshot(environment["PATH"]):
            errors.append("release build PATH executable namespace changed")
        cargo_home = pathlib.Path(environment["CARGO_HOME"])
        if context.get("configuration_snapshot") != _configuration_snapshot(cargo_home):
            errors.append("release build Cargo/rustup configuration changed")
        if context.get("cargo_home") != _directory_identity(cargo_home):
            errors.append("release build CARGO_HOME directory identity changed")
        expected_isolated = {
            name: _directory_identity(pathlib.Path(environment[name]))
            for name in ("HOME", "RUSTUP_HOME", "TMPDIR")
        }
        if context.get("isolated_directories") != expected_isolated:
            errors.append("release build isolated HOME/TMP/RUSTUP directory identity changed")
    except (OSError, EvidenceError) as exc:
        errors.append(f"release build execution context cannot be verified: {exc}")
    selection_sha = canonical_sha256({
        name: toolchain.get(name, {}).get("selection") for name in ("cargo", "rustc")
    })
    if context.get("toolchain_selection_sha256") != selection_sha:
        errors.append("release build execution context toolchain selector binding changed")
    return errors


def _verify_toolchain_live(toolchain: dict[str, Any]) -> None:
    for name in ("cargo", "rustc"):
        row = toolchain.get(name)
        if not isinstance(row, dict):
            raise EvidenceError(f"{name} toolchain binding is absent")
        resolved = pathlib.Path(row["invocation_path"]).resolve(strict=True)
        if str(resolved) != row.get("resolved_binary", {}).get("path"):
            raise EvidenceError(f"{name} invocation was substituted after toolchain capture")
        if _stable_file_identity(resolved) != row.get("resolved_binary"):
            raise EvidenceError(f"{name} bytes changed after toolchain capture")
        selection = row.get("selection")
        if not isinstance(selection, dict):
            raise EvidenceError(f"{name} toolchain selector binding is absent")
        discovered = pathlib.Path(selection.get("discovered_invocation_path", ""))
        discovered_resolved = discovered.resolve(strict=True)
        if _stable_file_identity(discovered_resolved) != selection.get("discovered_resolved_binary"):
            raise EvidenceError(f"{name} selector bytes/path changed after toolchain capture")


def build_release_build_receipt(
    *, source_manifest: dict[str, Any], build_log_binding: dict[str, Any],
    release_boundary: dict[str, Any],
    toolchain: dict[str, Any], target_host: str,
    target_directory: pathlib.Path,
    build_environment: dict[str, str], build_execution_context: dict[str, Any],
    compiler_artifacts: dict[str, Any],
    compiled_source_closures: dict[str, Any],
    built_at_unix_ns: int,
    verify_output_files: bool,
) -> dict[str, Any]:
    source_errors = validate_clean_source_manifest(
        source_manifest, release_boundary=release_boundary, verify_current=True,
    )
    if source_errors:
        raise EvidenceError("source manifest is not current/clean: " + "; ".join(source_errors))
    manifest_by_path = {row["path"]: row for row in source_manifest["entries"]}
    device_artifact = compiler_artifacts.get("hawking-tq-device-probe")
    spec_artifact = compiler_artifacts.get("hawking-tq-spec-probe")
    if not isinstance(device_artifact, dict) or not isinstance(spec_artifact, dict):
        raise EvidenceError("Cargo did not emit both exact release probe artifacts")
    receipt = _stamp({
        "schema": evidence_gate.RELEASE_BUILD_SCHEMA,
        "source_base_commit": source_manifest["source_base_commit"],
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_authority_capsule_sha256": source_manifest["capsule_sha256"],
        "cargo_lock_sha256": manifest_by_path["Cargo.lock"]["sha256"],
        "release_boundary_attestation_sha256": release_boundary["attestation_sha256"],
        "build_argv_sha256": canonical_sha256(_exact_build_argv(
            toolchain["cargo"]["invocation_path"], target_host, target_directory,
        )),
        "profile": "release",
        "features": ["tq"],
        "target_host": target_host,
        "target_directory": str(target_directory),
        "toolchain": toolchain,
        "build_environment": build_environment,
        "build_execution_context": build_execution_context,
        "compiler_artifacts": compiler_artifacts,
        "compiled_source_closures": compiled_source_closures,
        "success": True,
        "built_at_unix_ns": built_at_unix_ns,
        "build_log": build_log_binding,
        "probes": {
            "device": device_artifact["executable"],
            "spec": spec_artifact["executable"],
        },
        "runtime_defaults_changed": False,
    }, "receipt_sha256")
    errors = validate_release_build_receipt(
        receipt, source_manifest=source_manifest, release_boundary=release_boundary,
        verify_current=verify_output_files,
    )
    if errors:
        raise EvidenceError("constructed release build receipt is invalid: " + "; ".join(errors))
    return receipt


def validate_release_build_receipt(
    value: Any, *, source_manifest: dict[str, Any],
    release_boundary: dict[str, Any] | None = None,
    verify_current: bool = False,
) -> list[str]:
    errors = evidence_gate._validate_release_build(
        value, source_manifest=source_manifest, release_boundary=release_boundary,
        verify_files=verify_current,
    )
    if not isinstance(value, dict):
        return errors
    manifest_entries = source_manifest.get("entries") if isinstance(source_manifest, dict) else None
    cargo_row = next(
        (row for row in manifest_entries if isinstance(row, dict) and row.get("path") == "Cargo.lock"),
        None,
    ) if isinstance(manifest_entries, list) else None
    if not isinstance(cargo_row, dict) or value.get("cargo_lock_sha256") != cargo_row.get("sha256"):
        errors.append("release build Cargo.lock hash is not bound to the source manifest")
    environment = value.get("build_environment")
    toolchain = value.get("toolchain")
    if isinstance(environment, dict) and isinstance(toolchain, dict):
        errors.extend(_validate_build_execution_context_live(
            value.get("build_execution_context"),
            environment=environment, toolchain=toolchain,
        ) if verify_current else [])
    return errors


def _revalidate_build_outputs_before_seal(
    receipt: dict[str, Any], *, source_manifest: dict[str, Any],
    release_boundary: dict[str, Any], boundary_observation: dict[str, Any],
) -> None:
    current_source = build_clean_source_manifest(
        release_boundary=release_boundary, boundary_observation=boundary_observation,
    )
    if current_source != source_manifest:
        raise EvidenceError("release source capsule changed before immutable seal")
    _verify_toolchain_live(receipt["toolchain"])
    context_errors = _validate_build_execution_context_live(
        receipt.get("build_execution_context"),
        environment=receipt.get("build_environment", {}),
        toolchain=receipt.get("toolchain", {}),
    )
    if context_errors:
        raise EvidenceError("release build context changed before immutable seal: " + "; ".join(context_errors))
    bindings: list[tuple[str, dict[str, Any]]] = []
    probes = receipt.get("probes")
    if isinstance(probes, dict):
        bindings.extend((f"probe {name}", probes[name]) for name in ("device", "spec"))
    closures = receipt.get("compiled_source_closures")
    if isinstance(closures, dict):
        for name, closure in closures.items():
            if isinstance(closure, dict):
                bindings.append((f"{name} dep-info", closure.get("dep_info")))
                bindings.extend(
                    (f"{name} source", row) for row in closure.get("entries", [])
                    if isinstance(row, dict)
                )
    for label, binding in bindings:
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise EvidenceError(f"release build {label} binding is malformed before seal")
        if _stable_file_identity(pathlib.Path(binding["path"])) != binding:
            raise EvidenceError(f"release build {label} changed before immutable seal")


def _compiler_artifacts(stdout: str, *, target_directory: pathlib.Path) -> dict[str, Any]:
    wanted = {"hawking-tq-device-probe", "hawking-tq-spec-probe"}
    messages: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target")
        name = target.get("name") if isinstance(target, dict) else None
        if name not in wanted or target.get("kind") != ["bin"]:
            continue
        executable = message.get("executable")
        if name in messages:
            raise EvidenceError(f"Cargo emitted duplicate compiler artifacts for {name}")
        if not isinstance(executable, str):
            raise EvidenceError(f"Cargo artifact for {name} has no executable path")
        path = pathlib.Path(executable)
        expected_root = target_directory.resolve()
        try:
            path.resolve(strict=True).relative_to(expected_root)
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"Cargo artifact for {name} escapes pinned target dir") from exc
        messages[name] = {
            "target_name": name,
            "target_kind": ["bin"],
            "fresh": message.get("fresh"),
            "executable": _stable_file_identity(path, require_executable=True),
        }
    if set(messages) != wanted:
        raise EvidenceError(
            "Cargo output did not identify exactly both probe executables "
            f"(found={sorted(messages)})"
        )
    return messages


def _compiled_source_closure(artifact: dict[str, Any]) -> dict[str, Any]:
    executable = pathlib.Path(artifact["executable"]["path"])
    dep_info = executable.with_suffix(".d")
    dep_binding = _stable_file_identity(dep_info)
    text = dep_info.read_text(encoding="utf-8")
    logical = text.replace("\\\n", " ")
    if ":" not in logical:
        raise EvidenceError(f"Cargo dep-info has no target separator: {dep_info}")
    _target, dependency_text = logical.split(":", 1)
    try:
        tokens = shlex.split(dependency_text, posix=True)
    except ValueError as exc:
        raise EvidenceError(f"Cargo dep-info cannot be parsed: {dep_info}: {exc}") from exc
    paths: dict[str, pathlib.Path] = {}
    for token in tokens:
        candidate = pathlib.Path(token)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise EvidenceError(f"Cargo dep-info source is missing: {token}: {exc}") from exc
        if resolved.is_symlink() or not resolved.is_file():
            raise EvidenceError(f"Cargo dep-info source is unsafe: {resolved}")
        paths[str(resolved)] = resolved
    if not paths:
        raise EvidenceError(f"Cargo dep-info source closure is empty: {dep_info}")
    entries = [_stable_file_identity(path) for path in paths.values()]
    entries.sort(key=lambda row: row["path"])
    return {
        "dep_info": dep_binding,
        "entry_count": len(entries),
        "entries": entries,
        "closure_sha256": canonical_sha256(entries),
    }


def _compiled_source_closures(artifacts: dict[str, Any]) -> dict[str, Any]:
    closures = {
        name: _compiled_source_closure(artifacts[name])
        for name in ("hawking-tq-device-probe", "hawking-tq-spec-probe")
    }
    required = {
        "hawking-tq-device-probe": str(
            (ROOT / "crates/hawking/src/tq_device_probe.rs").resolve()
        ),
        "hawking-tq-spec-probe": str(
            (ROOT / "crates/hawking/src/tq_spec_probe.rs").resolve()
        ),
    }
    for name, required_path in required.items():
        if required_path not in {row["path"] for row in closures[name]["entries"]}:
            raise EvidenceError(f"Cargo dep-info closure for {name} omits its probe source")
    return closures


def run_release_build(
    admission: ReleaseAdmission, *, source_manifest: dict[str, Any],
    build_log: pathlib.Path, write_log: bool = True,
) -> tuple[dict[str, Any], str]:
    before = build_clean_source_manifest(
        release_boundary=admission.boundary_attestation,
        boundary_observation=admission.boundary_observation,
    )
    if before != source_manifest:
        raise EvidenceError("supplied source manifest differs from fresh pre-build manifest")
    toolchain = {"cargo": _tool_binding("cargo"), "rustc": _tool_binding("rustc")}
    _verify_toolchain_live(toolchain)
    target_host = _rustc_host(toolchain)
    target_directory = _unique_target_dir(
        admission.boundary_attestation, source_manifest,
    )
    _create_isolated_build_directories(target_directory)
    build_environment = _build_environment(
        toolchain, target_directory, lease_fd=admission.lease.fileno(),
    )
    build_execution_context = _build_execution_context(
        build_environment, toolchain=toolchain,
    )
    context_errors = _validate_build_execution_context_live(
        build_execution_context, environment=build_environment, toolchain=toolchain,
    )
    if context_errors:
        raise EvidenceError("release build context is not stable before Cargo: " + "; ".join(context_errors))
    command = _exact_build_argv(
        toolchain["cargo"]["invocation_path"], target_host, target_directory,
    )
    env = dict(build_environment)
    process = subprocess.run(
        command, cwd=ROOT, env=env, pass_fds=(admission.lease.fileno(),),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    log_text = (
        "$ " + " ".join(command) + "\n"
        + "[stdout]\n" + process.stdout + "\n[stderr]\n" + process.stderr
        + f"\n[exit_code]\n{process.returncode}\n"
    )
    log_raw = log_text.encode("utf-8")
    if write_log:
        _immutable_text(build_log, log_text)
    if process.returncode != 0:
        raise EvidenceError(f"exact release build failed with exit code {process.returncode}")
    artifacts = _compiler_artifacts(process.stdout, target_directory=target_directory)
    closures = _compiled_source_closures(artifacts)
    _verify_toolchain_live(toolchain)
    context_errors = _validate_build_execution_context_live(
        build_execution_context, environment=build_environment, toolchain=toolchain,
    )
    if context_errors:
        raise EvidenceError("release build context changed during Cargo: " + "; ".join(context_errors))
    after = build_clean_source_manifest(
        release_boundary=admission.boundary_attestation,
        boundary_observation=admission.boundary_observation,
    )
    if before != after:
        raise EvidenceError("source identity changed during release build")
    _recheck_under_lease(admission.observer["state_sha256"])
    receipt = build_release_build_receipt(
        source_manifest=after,
        build_log_binding=(
            _stable_file_identity(build_log) if write_log
            else _byte_binding(build_log, log_raw)
        ),
        release_boundary=admission.boundary_attestation,
        toolchain=toolchain, target_host=target_host,
        target_directory=target_directory,
        build_environment=build_environment,
        build_execution_context=build_execution_context,
        compiler_artifacts=artifacts,
        compiled_source_closures=closures,
        built_at_unix_ns=time.time_ns(),
        verify_output_files=write_log,
    )
    return receipt, log_text


def _within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _require_release_outputs(paths: Sequence[pathlib.Path]) -> None:
    report_root = _lexical_absolute(REPORT_ROOT)
    normalized: set[pathlib.Path] = set()
    for path in paths:
        candidate = _lexical_absolute(path)
        try:
            candidate.relative_to(report_root)
        except ValueError as exc:
            raise EvidenceError(
                f"physical release output must stay in the unbound release tree: {path}"
            ) from exc
        if candidate in normalized:
            raise EvidenceError("physical release output paths must be distinct")
        normalized.add(candidate)


def build_corpus_verification(
    index: dict[str, Any], *, boundary_attestation: dict[str, Any],
    boundary_observation: dict[str, Any], verified_at_unix_ns: int,
    verification_phase: str,
    parent_verification_receipt: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    boundary_errors = validate_release_boundary_attestation(
        boundary_attestation, observation=boundary_observation,
    )
    if boundary_errors:
        raise EvidenceError("release boundary is invalid: " + "; ".join(boundary_errors))
    structure_errors = evidence_gate._validate_corpus_index(index)
    if structure_errors:
        raise EvidenceError("corpus index is invalid: " + "; ".join(structure_errors))
    root_raw = index.get("root")
    if not isinstance(root_raw, str) or not pathlib.Path(root_raw).is_absolute():
        raise EvidenceError("corpus index root must be absolute")
    root = pathlib.Path(root_raw)
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError("corpus index root is absent or unsafe")
    entries = index.get("entries")
    assert isinstance(entries, list)
    observed_rows: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    missing: list[str] = []
    changed: list[str] = []
    symlinks: list[str] = []
    for entry in entries:
        relative = entry["path"]
        expected_paths.add(relative)
        path = root / pathlib.Path(*pathlib.PurePosixPath(relative).parts)
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        identity = _stable_file_identity(path)
        match = identity["sha256"] == entry["sha256"] and identity["size_bytes"] == entry["size"]
        if not match:
            changed.append(relative)
        observed_rows.append({
            "path": relative,
            "kind": entry["kind"],
            "expected_sha256": entry["sha256"],
            "expected_size": entry["size"],
            "observed_sha256": identity["sha256"],
            "observed_size": identity["size_bytes"],
            "semantics": entry["semantics"],
            "matched": match,
        })
    # Repeat every stable read before the directory census.  This detects a
    # writer that replaces an already-hashed file while the first pass is still
    # traversing the corpus.
    first_by_path = {row["path"]: row for row in observed_rows}
    for relative in sorted(expected_paths):
        row = first_by_path.get(relative)
        if row is None:
            continue
        path = root / pathlib.Path(*pathlib.PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            missing.append(relative)
            continue
        identity = _stable_file_identity(path)
        if identity["sha256"] != row["observed_sha256"] \
                or identity["size_bytes"] != row["observed_size"]:
            changed.append(relative)
    current_paths: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        directory = pathlib.Path(dirpath)
        for name in sorted(filenames):
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
            else:
                current_paths.add(relative)
    added = sorted(current_paths - expected_paths)
    owners = spec_reentry_scaffold.active_heavy_owners()
    if verification_phase == "pre_release_build":
        parent_sha = None
        if parent_verification_receipt is not None:
            raise EvidenceError("pre-build corpus verification cannot have a parent")
    elif verification_phase == "post_release_build":
        if not isinstance(parent_verification_receipt, dict):
            raise EvidenceError("post-build corpus verification requires the pre-build receipt")
        parent_sha = parent_verification_receipt.get("verification_receipt_sha256")
        if not isinstance(parent_sha, str) or HEX64.fullmatch(parent_sha) is None:
            raise EvidenceError("post-build corpus verification parent hash is invalid")
    else:
        raise EvidenceError("corpus verification phase must be pre_release_build or post_release_build")
    receipt = _stamp({
        "schema": CORPUS_RECEIPT_SCHEMA,
        "index_sha256": index["index_sha256"],
        "release_boundary_attestation_sha256": boundary_attestation["attestation_sha256"],
        "release_boundary_observation_sha256": boundary_observation["observation_sha256"],
        "root": str(root.resolve()),
        "verified_at_unix_ns": verified_at_unix_ns,
        "verification_phase": verification_phase,
        "parent_verification_receipt_sha256": parent_sha,
        "exclusive_shared_heavy_lease_held": True,
        "active_heavy_owners_after_hashing": owners,
        "entries": sorted(observed_rows, key=lambda row: row["path"]),
        "changed_files": sorted(set(changed)),
        "missing_files": sorted(set(missing)),
        "added_files": added,
        "symlinks": sorted(set(symlinks)),
        "verified_semantic_counts": index["semantic_counts"],
        "default_mutation_requested": False,
    }, "verification_receipt_sha256")
    attestation = _stamp({
        "schema": evidence_gate.CORPUS_VERIFICATION_SCHEMA,
        "index_sha256": index["index_sha256"],
        "verified_at_unix_ns": verified_at_unix_ns,
        "active_heavy_owner_count": len(owners),
        "file_count": index["file_count"],
        "total_bytes": index["total_bytes"],
        "changed_files": len(receipt["changed_files"]),
        "missing_files": len(receipt["missing_files"]),
        "added_files": len(receipt["added_files"]),
        "symlinks": len(receipt["symlinks"]),
        "semantic_counts": index["semantic_counts"],
        "all_censused_semantics_verified": True,
        "verification_receipt_sha256": receipt["verification_receipt_sha256"],
    }, "attestation_sha256")
    errors = validate_corpus_verification(
        attestation, receipt=receipt, index=index,
        boundary_attestation=boundary_attestation,
        boundary_observation=boundary_observation,
        required_phase=verification_phase,
        parent_verification_receipt=parent_verification_receipt,
    )
    if errors:
        raise EvidenceError("corpus verification is not green: " + "; ".join(errors))
    return receipt, attestation


def validate_corpus_verification(
    attestation: Any, *, receipt: Any, index: dict[str, Any],
    boundary_attestation: dict[str, Any], boundary_observation: dict[str, Any],
    required_phase: str = "post_release_build",
    parent_verification_receipt: dict[str, Any] | None = None,
) -> list[str]:
    errors = evidence_gate._validate_corpus_index(index)
    errors.extend(evidence_gate._validate_corpus_verification(attestation, index=index))
    errors.extend(validate_release_boundary_attestation(
        boundary_attestation, observation=boundary_observation,
    ))
    expected_receipt_fields = {
        "schema", "index_sha256", "release_boundary_attestation_sha256",
        "release_boundary_observation_sha256", "root", "verified_at_unix_ns",
        "verification_phase", "parent_verification_receipt_sha256",
        "exclusive_shared_heavy_lease_held", "active_heavy_owners_after_hashing",
        "entries", "changed_files", "missing_files", "added_files", "symlinks",
        "verified_semantic_counts", "default_mutation_requested",
        "verification_receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_receipt_fields:
        errors.append("corpus verification receipt is malformed")
        return errors
    errors.extend(_hash_errors(receipt, "verification_receipt_sha256", label="corpus_receipt"))
    if receipt.get("schema") != CORPUS_RECEIPT_SCHEMA:
        errors.append("corpus verification receipt schema is invalid")
    if receipt.get("index_sha256") != index.get("index_sha256"):
        errors.append("corpus receipt is not bound to its exact index")
    if receipt.get("release_boundary_attestation_sha256") != boundary_attestation.get("attestation_sha256") \
            or receipt.get("release_boundary_observation_sha256") != boundary_observation.get("observation_sha256"):
        errors.append("corpus receipt is not bound to the exact release boundary")
    if receipt.get("exclusive_shared_heavy_lease_held") is not True \
            or receipt.get("active_heavy_owners_after_hashing") != []:
        errors.append("corpus receipt was not produced owner-free under the shared lease")
    for field in ("changed_files", "missing_files", "added_files", "symlinks"):
        if receipt.get(field) != []:
            errors.append(f"corpus receipt {field} must be empty")
    rows = receipt.get("entries")
    if not isinstance(rows, list) or len(rows) != index.get("file_count") \
            or any(not isinstance(row, dict) or row.get("matched") is not True for row in rows):
        errors.append("corpus receipt does not prove every indexed file matched")
    if receipt.get("verified_semantic_counts") != index.get("semantic_counts"):
        errors.append("corpus receipt semantic census differs from index")
    if receipt.get("verification_phase") != required_phase:
        errors.append(f"corpus receipt phase must be {required_phase}")
    expected_parent = (
        parent_verification_receipt.get("verification_receipt_sha256")
        if isinstance(parent_verification_receipt, dict) else None
    )
    if required_phase == "post_release_build":
        if expected_parent is None \
                or receipt.get("parent_verification_receipt_sha256") != expected_parent:
            errors.append("post-build corpus receipt is not bound to exact pre-build receipt")
    elif receipt.get("parent_verification_receipt_sha256") is not None:
        errors.append("pre-build corpus receipt unexpectedly has a parent")
    if receipt.get("default_mutation_requested") is not False:
        errors.append("corpus receipt requests a default mutation")
    if isinstance(attestation, dict):
        if attestation.get("verification_receipt_sha256") != receipt.get("verification_receipt_sha256"):
            errors.append("corpus attestation is not bound to the verification receipt")
        if attestation.get("verified_at_unix_ns") != receipt.get("verified_at_unix_ns"):
            errors.append("corpus attestation timestamp differs from receipt")
    return errors


def _revalidate_final_assembly_cas(
    *, current_observer: dict[str, Any], boundary_attestation: dict[str, Any],
    boundary_observation: dict[str, Any], corpus_index: dict[str, Any],
    corpus_prebuild_receipt: dict[str, Any], corpus_receipt: dict[str, Any],
    corpus_attestation: dict[str, Any],
) -> None:
    """Recreate every mutable release parent from current bytes under the lease."""
    boundary_errors = validate_release_boundary_attestation(
        boundary_attestation, observation=boundary_observation, observer=current_observer,
    )
    if boundary_errors:
        raise EvidenceError("final assembly boundary CAS failed: " + "; ".join(boundary_errors))
    reference_errors, verified = verify_final_packet_references(current_observer)
    if reference_errors or verified is None:
        raise EvidenceError(
            "final assembly Doctor reference CAS failed: " + "; ".join(reference_errors)
        )
    expected_final = {
        "final_packet_file_sha256": verified["final_packet_file_sha256"],
        "final_packet_canonical_sha256": verified["final_packet"]["packet_sha256"],
        "verified_reference_count": verified["verified_reference_count"],
        "verified_references_sha256": verified["verified_references_sha256"],
    }
    for field, observed in expected_final.items():
        if boundary_observation.get(field) != observed:
            raise EvidenceError(f"final assembly Doctor reference CAS changed {field}")

    try:
        fresh_pre, _fresh_pre_attestation = build_corpus_verification(
            corpus_index,
            boundary_attestation=boundary_attestation,
            boundary_observation=boundary_observation,
            verified_at_unix_ns=corpus_prebuild_receipt["verified_at_unix_ns"],
            verification_phase="pre_release_build",
        )
    except (KeyError, EvidenceError) as exc:
        raise EvidenceError(f"final assembly pre-build corpus CAS failed: {exc}") from exc
    if fresh_pre != corpus_prebuild_receipt:
        raise EvidenceError("final assembly pre-build corpus receipt is stale")
    try:
        fresh_post, fresh_attestation = build_corpus_verification(
            corpus_index,
            boundary_attestation=boundary_attestation,
            boundary_observation=boundary_observation,
            verified_at_unix_ns=corpus_receipt["verified_at_unix_ns"],
            verification_phase="post_release_build",
            parent_verification_receipt=corpus_prebuild_receipt,
        )
    except (KeyError, EvidenceError) as exc:
        raise EvidenceError(f"final assembly post-build corpus CAS failed: {exc}") from exc
    if fresh_post != corpus_receipt or fresh_attestation != corpus_attestation:
        raise EvidenceError("final assembly post-build corpus receipt/attestation is stale")


def build_evidence_manifest(kind: str, paths: Sequence[pathlib.Path]) -> dict[str, Any]:
    if kind not in {"device", "spec"}:
        raise EvidenceError("evidence manifest kind must be device or spec")
    if not paths:
        raise EvidenceError("evidence manifest cannot be empty")
    entries = [_stable_file_identity(path) for path in paths]
    entries.sort(key=lambda row: (row["sha256"], row["path"]))
    manifest = _stamp({
        "schema": EVIDENCE_MANIFEST_SCHEMA,
        "kind": kind,
        "entries": entries,
        "default_mutation_requested": False,
    }, "manifest_sha256")
    errors = validate_evidence_manifest(manifest, kind=kind, verify_files=True)
    if errors:
        raise EvidenceError("constructed evidence manifest is invalid: " + "; ".join(errors))
    return manifest


def validate_evidence_manifest(
    value: Any, *, kind: str, verify_files: bool = False,
) -> list[str]:
    expected = {"schema", "kind", "entries", "default_mutation_requested", "manifest_sha256"}
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != expected:
        return [f"{kind} evidence manifest is malformed"]
    errors.extend(_hash_errors(value, "manifest_sha256", label=f"{kind}_evidence_manifest"))
    if value.get("schema") != EVIDENCE_MANIFEST_SCHEMA or value.get("kind") != kind:
        errors.append(f"{kind} evidence manifest schema/kind is invalid")
    if value.get("default_mutation_requested") is not False:
        errors.append(f"{kind} evidence manifest requests a default mutation")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{kind} evidence manifest entries are empty")
        return errors
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for row in entries:
        file_errors = evidence_gate._file_binding_errors(row, label=f"{kind} evidence entry")
        errors.extend(file_errors)
        if not isinstance(row, dict):
            continue
        path_raw, digest = row.get("path"), row.get("sha256")
        if path_raw in seen_paths or digest in seen_hashes:
            errors.append(f"{kind} evidence manifest reuses a path or file hash")
        if isinstance(path_raw, str):
            seen_paths.add(path_raw)
        if isinstance(digest, str):
            seen_hashes.add(digest)
        if verify_files and isinstance(path_raw, str):
            try:
                observed = _stable_file_identity(pathlib.Path(path_raw))
            except (OSError, EvidenceError) as exc:
                errors.append(str(exc))
            else:
                if observed != row:
                    errors.append(f"{kind} evidence file differs from manifest: {path_raw}")
    if entries != sorted(entries, key=lambda row: (row.get("sha256", ""), row.get("path", ""))):
        errors.append(f"{kind} evidence manifest entries are not canonical")
    return errors


def _load_evidence_items(manifest: dict[str, Any], *, kind: str) -> list[dict[str, Any]]:
    errors = validate_evidence_manifest(manifest, kind=kind, verify_files=True)
    if errors:
        raise EvidenceError(f"{kind} evidence manifest is invalid: " + "; ".join(errors))
    items: list[dict[str, Any]] = []
    for row in manifest["entries"]:
        value = _load_bound_json(
            pathlib.Path(row["path"]), row,
            label=f"{kind} evidence item {row['path']}",
        )
        if not isinstance(value, dict):
            raise EvidenceError(f"{kind} evidence item is not an object: {row['path']}")
        items.append(value)
    if kind == "device":
        items.sort(key=lambda row: str(row.get("cell_id", "")))
    else:
        items.sort(key=lambda row: RUNTIME_ORDER.get(row.get("runtime_path"), 999))
    return items


def assemble_physical_packet(
    *, boundary_attestation: dict[str, Any], boundary_observation: dict[str, Any],
    corpus_index: dict[str, Any], corpus_attestation: dict[str, Any],
    corpus_prebuild_receipt: dict[str, Any], corpus_receipt: dict[str, Any],
    source_manifest: dict[str, Any],
    release_build: dict[str, Any], device_manifest: dict[str, Any],
    spec_manifest: dict[str, Any], spec_label: str = "CORPUS",
    current_observer: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if current_observer is not None:
        _revalidate_final_assembly_cas(
            current_observer=current_observer,
            boundary_attestation=boundary_attestation,
            boundary_observation=boundary_observation,
            corpus_index=corpus_index,
            corpus_prebuild_receipt=corpus_prebuild_receipt,
            corpus_receipt=corpus_receipt,
            corpus_attestation=corpus_attestation,
        )
    errors = validate_release_boundary_attestation(
        boundary_attestation, observation=boundary_observation, observer=current_observer,
    )
    errors.extend(validate_corpus_verification(
        corpus_attestation, receipt=corpus_receipt, index=corpus_index,
        boundary_attestation=boundary_attestation,
        boundary_observation=boundary_observation,
        required_phase="post_release_build",
        parent_verification_receipt=corpus_prebuild_receipt,
    ))
    errors.extend(validate_clean_source_manifest(
        source_manifest, release_boundary=boundary_attestation,
        boundary_observation=boundary_observation, verify_current=True,
    ))
    errors.extend(validate_release_build_receipt(
        release_build, source_manifest=source_manifest,
        release_boundary=boundary_attestation, verify_current=True,
    ))
    errors.extend(validate_evidence_manifest(device_manifest, kind="device", verify_files=True))
    errors.extend(validate_evidence_manifest(spec_manifest, kind="spec", verify_files=True))
    if errors:
        raise EvidenceError("assembly parents are invalid: " + "; ".join(errors))
    device_hashes = {row["sha256"] for row in device_manifest["entries"]}
    spec_hashes = {row["sha256"] for row in spec_manifest["entries"]}
    if device_hashes & spec_hashes:
        raise EvidenceError("device and speculative evidence manifests cross-credit file bytes")
    device_items = _load_evidence_items(device_manifest, kind="device")
    spec_items = _load_evidence_items(spec_manifest, kind="spec")
    packet = evidence_gate.stamp({
        "schema": evidence_gate.SCHEMA,
        "release_boundary": boundary_attestation,
        "corpus_index": corpus_index,
        "corpus_verification": corpus_attestation,
        "source_manifest": source_manifest,
        "release_build": release_build,
        "cpu_error_policy": evidence_gate.CPU_ERROR_POLICY,
        "spec_label": spec_label,
        "device_evidence": device_items,
        "spec_evidence": spec_items,
        "default_mutation_requested": False,
    })
    gate_errors = evidence_gate.validate_gate(packet, verify_counter_files=True)
    if gate_errors:
        raise EvidenceError("aggregate physical evidence gate is red: " + "; ".join(gate_errors))
    receipt = _stamp({
        "schema": ASSEMBLY_RECEIPT_SCHEMA,
        "release_boundary_attestation_sha256": boundary_attestation["attestation_sha256"],
        "release_boundary_observation_sha256": boundary_observation["observation_sha256"],
        "corpus_index_sha256": corpus_index["index_sha256"],
        "corpus_verification_attestation_sha256": corpus_attestation["attestation_sha256"],
        "corpus_prebuild_verification_receipt_sha256": corpus_prebuild_receipt["verification_receipt_sha256"],
        "corpus_verification_receipt_sha256": corpus_receipt["verification_receipt_sha256"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "release_build_sha256": release_build["receipt_sha256"],
        "device_evidence_manifest_sha256": device_manifest["manifest_sha256"],
        "spec_evidence_manifest_sha256": spec_manifest["manifest_sha256"],
        "physical_packet_gate_sha256": packet["gate_sha256"],
        "device_evidence_count": len(device_items),
        "spec_evidence_count": len(spec_items),
        "default_off": True,
        "activation_requested": False,
    }, "assembly_receipt_sha256")
    if current_observer is not None:
        _revalidate_final_assembly_cas(
            current_observer=current_observer,
            boundary_attestation=boundary_attestation,
            boundary_observation=boundary_observation,
            corpus_index=corpus_index,
            corpus_prebuild_receipt=corpus_prebuild_receipt,
            corpus_receipt=corpus_receipt,
            corpus_attestation=corpus_attestation,
        )
    return packet, receipt


def prepare_release(
    admission: ReleaseAdmission, *, corpus_root: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    """Prepare every pre-physical parent under one held lease and one boundary."""
    output_dir = output_dir.resolve()
    paths = {
        "boundary_observation": output_dir / "release_boundary_observation.json",
        "boundary_attestation": output_dir / "release_boundary_attestation.json",
        "source_capsule": output_dir / "critical_source_capsule.json",
        "corpus_index": output_dir / "corpus_index.json",
        "corpus_prebuild_receipt": output_dir / "corpus_prebuild_verification_receipt.json",
        "release_build_log": output_dir / "release_build.log",
        "release_build_receipt": output_dir / "release_build_receipt.json",
        "corpus_postbuild_receipt": output_dir / "corpus_postbuild_verification_receipt.json",
        "corpus_attestation": output_dir / "corpus_verification_attestation.json",
        "prepared_release": output_dir / "prepared_release.json",
    }
    _require_release_outputs(tuple(paths.values()))
    if any(_within(path, corpus_root) for path in paths.values()):
        raise EvidenceError("prepared release outputs must live outside the frozen corpus")

    boundary = admission.boundary_attestation
    observation = admission.boundary_observation
    source = build_clean_source_manifest(
        release_boundary=boundary, boundary_observation=observation,
    )
    index = appendix_corpus.build_index(
        corpus_root, active_owners=[], source_base_commit=source["source_base_commit"],
    )
    pre_receipt, _pre_attestation = build_corpus_verification(
        index, boundary_attestation=boundary, boundary_observation=observation,
        verified_at_unix_ns=time.time_ns(),
        verification_phase="pre_release_build",
    )
    build_receipt, build_log_text = run_release_build(
        admission, source_manifest=source,
        build_log=paths["release_build_log"], write_log=False,
    )
    post_receipt, corpus_attestation = build_corpus_verification(
        index, boundary_attestation=boundary, boundary_observation=observation,
        verified_at_unix_ns=time.time_ns(),
        verification_phase="post_release_build",
        parent_verification_receipt=pre_receipt,
    )
    _recheck_under_lease(admission.observer["state_sha256"])
    _revalidate_final_assembly_cas(
        current_observer=admission.observer,
        boundary_attestation=boundary,
        boundary_observation=observation,
        corpus_index=index,
        corpus_prebuild_receipt=pre_receipt,
        corpus_receipt=post_receipt,
        corpus_attestation=corpus_attestation,
    )
    _revalidate_build_outputs_before_seal(
        build_receipt, source_manifest=source,
        release_boundary=boundary, boundary_observation=observation,
    )
    _recheck_under_lease(admission.observer["state_sha256"])
    prepared = _stamp({
        "schema": PREPARED_RELEASE_SCHEMA,
        "release_boundary_attestation_sha256": boundary["attestation_sha256"],
        "release_boundary_observation_sha256": observation["observation_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "corpus_index_sha256": index["index_sha256"],
        "corpus_prebuild_verification_receipt_sha256": pre_receipt["verification_receipt_sha256"],
        "release_build_receipt_sha256": build_receipt["receipt_sha256"],
        "corpus_postbuild_verification_receipt_sha256": post_receipt["verification_receipt_sha256"],
        "corpus_verification_attestation_sha256": corpus_attestation["attestation_sha256"],
        "unique_target_directory": build_receipt["target_directory"],
        "phase_order": [
            "boundary", "critical_source_capsule", "corpus_freeze",
            "corpus_prebuild_verification", "unique_target_release_build",
            "corpus_postbuild_verification", "final_live_cas", "atomic_seal",
        ],
        "default_off": True,
        "activation_requested": False,
    }, "prepared_release_sha256")
    payloads: dict[str, Any] = {
        "boundary_observation": observation,
        "boundary_attestation": boundary,
        "source_capsule": source,
        "corpus_index": index,
        "corpus_prebuild_receipt": pre_receipt,
        "release_build_receipt": build_receipt,
        "corpus_postbuild_receipt": post_receipt,
        "corpus_attestation": corpus_attestation,
        "prepared_release": prepared,
    }
    byte_rows = [
        (paths[name], _json_bytes(value)) for name, value in payloads.items()
    ]
    byte_rows.append((paths["release_build_log"], build_log_text.encode("utf-8")))
    _atomic_bytes_group(tuple(byte_rows))
    return prepared


def _dry_run(action: str) -> dict[str, Any]:
    status = current_status()
    return {
        "schema": "hawking.appendix_physical_packet_dry_run.v1",
        "action": action,
        "would_acquire_shared_heavy_lease": True,
        "would_recheck_final_ready_owner_free_guard_under_lease": True,
        "would_run_cargo": action in {"release-build", "prepare-release"},
        "would_hash_corpus": action in {"freeze-corpus", "prepare-release"},
        "would_run_model_or_gpu": False,
        "would_mutate_runtime_default": False,
        "currently_admissible_before_lease": status["prelease_admission_ready"],
        "blockers": status["blockers"],
    }


def _selftest() -> int:
    resource = {
        "ok": True, "pressure_level": 1, "swap_used_mb": 0.0,
    }
    observer = {
        "schema": OBSERVER_SCHEMA,
        "final_interpretation_ready": True,
        "final_interpretation_packet": {"path": "x", "sha256": "a" * 64, "bytes": 1},
        "source_deletion_permitted": False,
    }
    observer = _stamp(observer, "state_sha256")
    status = current_status(observer=observer, owners=[], resource=resource)
    assert status["prelease_admission_ready"] is True
    assert status["execution_capability"] is False
    assert _dry_run("release-build")["would_run_model_or_gpu"] is False
    assert _exact_build_argv(
        "/tool/cargo", "arm64-apple-darwin", ROOT / "target" / "unique",
    )[0] == "/tool/cargo"
    print("appendix_physical_release_packet.py selftest OK")
    return 0


def _write_boundary(admission: ReleaseAdmission, observation: pathlib.Path, attestation: pathlib.Path) -> None:
    _atomic_json(observation, admission.boundary_observation)
    _atomic_json(attestation, admission.boundary_attestation)


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        return _selftest()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    dry = sub.add_parser("dry-run")
    dry.add_argument("action", choices=(
        "attest-boundary", "source-manifest", "release-build", "freeze-corpus",
        "prepare-release", "evidence-manifest", "assemble",
    ))
    boundary = sub.add_parser("attest-boundary")
    boundary.add_argument("--observation-output", type=pathlib.Path, required=True)
    boundary.add_argument("--attestation-output", type=pathlib.Path, required=True)
    source = sub.add_parser("source-manifest")
    source.add_argument("--output", type=pathlib.Path, required=True)
    build = sub.add_parser("release-build")
    build.add_argument("--source-manifest", type=pathlib.Path, required=True)
    build.add_argument("--build-log", type=pathlib.Path, required=True)
    build.add_argument("--output", type=pathlib.Path, required=True)
    corpus = sub.add_parser("freeze-corpus")
    corpus.add_argument("--root", type=pathlib.Path, default=appendix_corpus.DEFAULT_CORPUS)
    corpus.add_argument("--index-output", type=pathlib.Path, required=True)
    corpus.add_argument("--receipt-output", type=pathlib.Path, required=True)
    corpus.add_argument("--attestation-output", type=pathlib.Path, required=True)
    prepare = sub.add_parser("prepare-release")
    prepare.add_argument("--root", type=pathlib.Path, default=appendix_corpus.DEFAULT_CORPUS)
    prepare.add_argument("--output-dir", type=pathlib.Path, required=True)
    evidence = sub.add_parser("evidence-manifest")
    evidence.add_argument("--kind", choices=("device", "spec"), required=True)
    evidence.add_argument("--item", action="append", type=pathlib.Path, required=True)
    evidence.add_argument("--output", type=pathlib.Path, required=True)
    assembly = sub.add_parser("assemble")
    for name in (
        "boundary-attestation", "boundary-observation", "corpus-index",
        "corpus-attestation", "corpus-prebuild-receipt", "corpus-receipt",
        "source-manifest", "release-build",
        "device-manifest", "spec-manifest",
    ):
        assembly.add_argument(f"--{name}", type=pathlib.Path, required=True)
    assembly.add_argument("--spec-label", default="CORPUS")
    assembly.add_argument("--output", type=pathlib.Path, required=True)
    assembly.add_argument("--receipt-output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "status":
        print(json.dumps(current_status(), indent=2, sort_keys=True))
        return 0
    if args.command == "dry-run":
        print(json.dumps(_dry_run(args.action), indent=2, sort_keys=True))
        return 0
    try:
        with release_admission() as admission:
            if args.command in {
                "attest-boundary", "source-manifest", "release-build", "freeze-corpus",
            }:
                raise EvidenceError(
                    "standalone release-parent issuance is disabled; use prepare-release "
                    "so one held lease and boundary bind every parent"
                )
            if args.command == "prepare-release":
                value = prepare_release(
                    admission, corpus_root=args.root, output_dir=args.output_dir,
                )
                print(json.dumps(value, indent=2, sort_keys=True))
            elif args.command == "evidence-manifest":
                _require_release_outputs((args.output,))
                manifest = build_evidence_manifest(args.kind, args.item)
                _recheck_under_lease(admission.observer["state_sha256"])
                _atomic_json(args.output, manifest)
            elif args.command == "assemble":
                _require_release_outputs((args.output, args.receipt_output))
                boundary_attestation = _load(args.boundary_attestation)
                boundary_observation = _load(args.boundary_observation)
                corpus_index = _load(args.corpus_index)
                corpus_attestation = _load(args.corpus_attestation)
                corpus_prebuild_receipt = _load(args.corpus_prebuild_receipt)
                corpus_receipt = _load(args.corpus_receipt)
                source_manifest = _load(args.source_manifest)
                release_build = _load(args.release_build)
                device_manifest = _load(args.device_manifest)
                spec_manifest = _load(args.spec_manifest)
                packet, receipt = assemble_physical_packet(
                    boundary_attestation=boundary_attestation,
                    boundary_observation=boundary_observation,
                    corpus_index=corpus_index,
                    corpus_attestation=corpus_attestation,
                    corpus_prebuild_receipt=corpus_prebuild_receipt,
                    corpus_receipt=corpus_receipt,
                    source_manifest=source_manifest,
                    release_build=release_build,
                    device_manifest=device_manifest,
                    spec_manifest=spec_manifest,
                    spec_label=args.spec_label,
                    current_observer=admission.observer,
                )
                _recheck_under_lease(admission.observer["state_sha256"])
                _revalidate_final_assembly_cas(
                    current_observer=admission.observer,
                    boundary_attestation=boundary_attestation,
                    boundary_observation=boundary_observation,
                    corpus_index=corpus_index,
                    corpus_prebuild_receipt=corpus_prebuild_receipt,
                    corpus_receipt=corpus_receipt,
                    corpus_attestation=corpus_attestation,
                )
                _atomic_json_group(((args.output, packet), (args.receipt_output, receipt)))
            else:  # pragma: no cover - argparse makes this unreachable.
                raise AssertionError(args.command)
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceError, ReleaseBlocked) as exc:
        print(str(exc), file=sys.stderr)
        return 75 if isinstance(exc, ReleaseBlocked) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
