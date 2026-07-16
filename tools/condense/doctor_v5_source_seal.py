#!/usr/bin/env python3.12
"""Immutable source seals for one-pass Doctor V5 source verification.

Legacy cells re-hash every multi-GB source shard before the quantizer reads it.
A seal pays that pass once, records the census digest plus an exact filesystem
identity tuple, and lets pending accelerated cells use O(stat) verification.
Any inode, size, mtime, or ctime drift fails closed.

``freeze`` refuses while the Studio heavy lease is owned.  It is intended for
the drained acceleration checkpoint, never the active campaign.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Callable
import uuid


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "reports/condense/doctor_v5_pass_b/parameter_manifests"
SEAL_ROOT = ROOT / "reports/condense/doctor_v5_acceleration/source_seals"
HEAVY_LOCK = ROOT / "reports/cron/studio_heavy.lock"
SCHEMA_V1 = "hawking.doctor_v5_source_seal.v1"
SCHEMA = "hawking.doctor_v5_source_seal.v2"
MIGRATION_SCHEMA = "hawking.doctor_v5_source_seal_v1_to_v2_migration.v1"
MAX_JSON_BYTES = 64 * 1024 * 1024

# Darwin getattrlist(2) volume attributes.  st_dev is a kernel-local mount
# number and may be renumbered across a reboot.  APFS' volume UUID is the
# persistent filesystem identity that lets a seal distinguish a harmless
# remount from a different volume without weakening any file-level checks.
_ATTR_BIT_MAP_COUNT = 5
_ATTR_VOL_UUID = 0x00040000
_ATTR_VOL_INFO = 0x80000000


class _AttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


class _Fsid(ctypes.Structure):
    _fields_ = [("val", ctypes.c_int32 * 2)]


class _Statfs(ctypes.Structure):
    # Darwin 64-bit struct statfs from <sys/mount.h>.
    _fields_ = [
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64),
        ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64),
        ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64),
        ("f_fsid", _Fsid),
        ("f_owner", ctypes.c_uint32),
        ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32),
        ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * 16),
        ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    ]


V1_VERIFICATION = {
    "content_hashed_once": True,
    "structural_reuse_fields": ["st_dev", "st_ino", "st_size",
                                 "st_mtime_ns", "st_ctime_ns"],
    "fallback_on_mismatch": "fail_closed",
    "source_deletion_permitted": False,
}
V2_VERIFICATION = {
    "content_hash_authority": "full_sha256_at_freeze_or_v1_migration",
    "structural_reuse_fields": ["apfs_volume_uuid", "st_ino", "st_size",
                                 "st_mtime_ns", "st_ctime_ns"],
    "path_fd_identity_required": True,
    "observed_device_policy": (
        "st_dev_at_seal_is_audit_only; renumbering is accepted only with the exact "
        "APFS volume UUID and every stable file field unchanged"
    ),
    "fallback_on_mismatch": "fail_closed",
    "source_deletion_permitted": False,
}


class SourceSealError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: row for name, row in value.items() if name != key}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceSealError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceSealError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _identity(row: os.stat_result) -> dict[str, int]:
    """Return the exact live identity used to bind a pathname to an open fd."""
    return {
        "st_dev": row.st_dev, "st_ino": row.st_ino, "st_size": row.st_size,
        "st_mtime_ns": row.st_mtime_ns, "st_ctime_ns": row.st_ctime_ns,
    }


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(character in "0123456789abcdef" for character in value)


def _open_regular(path: Path) -> tuple[Path, int, os.stat_result]:
    """Open a canonical workspace file and prove path/fd identity before use."""
    raw = Path(path)
    try:
        if raw.is_symlink():
            raise SourceSealError(f"symlinked source is forbidden: {raw}")
        resolved = raw.resolve(strict=True)
        resolved.relative_to(ROOT.resolve())
        path_row = os.lstat(resolved)
    except (OSError, ValueError) as exc:
        raise SourceSealError(f"source path is invalid: {raw}: {exc}") from exc
    if not stat.S_ISREG(path_row.st_mode):
        raise SourceSealError(f"source is not a regular file: {resolved}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise SourceSealError(f"cannot open source: {resolved}: {exc}") from exc
    try:
        fd_row = os.fstat(fd)
        if not stat.S_ISREG(fd_row.st_mode) or _identity(path_row) != _identity(fd_row):
            raise SourceSealError(f"source path/fd identity differs: {resolved}")
        return resolved, fd, fd_row
    except BaseException:
        os.close(fd)
        raise


def _verify_open_regular(path: Path, fd: int, expected: os.stat_result) -> os.stat_result:
    """Re-prove the same exact path/fd tuple after an operation."""
    try:
        path_row = os.lstat(path)
        fd_row = os.fstat(fd)
    except OSError as exc:
        raise SourceSealError(f"cannot revalidate open source {path}: {exc}") from exc
    expected_identity = _identity(expected)
    if not stat.S_ISREG(path_row.st_mode) or not stat.S_ISREG(fd_row.st_mode) \
            or _identity(path_row) != expected_identity \
            or _identity(fd_row) != expected_identity:
        raise SourceSealError(f"source path/fd identity changed: {path}")
    return fd_row


def _volume_identity(fd: int) -> dict[str, str]:
    """Return the persistent APFS volume UUID for an already verified fd."""
    if sys.platform != "darwin":
        raise SourceSealError("source seal v2 requires Darwin APFS volume identity")
    libc = ctypes.CDLL(None, use_errno=True)
    statfs_row = _Statfs()
    fstatfs = libc.fstatfs
    fstatfs.argtypes = [ctypes.c_int, ctypes.POINTER(_Statfs)]
    fstatfs.restype = ctypes.c_int
    if fstatfs(fd, ctypes.byref(statfs_row)) != 0:
        error = ctypes.get_errno()
        raise SourceSealError(f"fstatfs failed for source fd: {os.strerror(error)}")
    filesystem = bytes(statfs_row.f_fstypename).split(b"\0", 1)[0].decode(
        "ascii", errors="strict"
    )
    if filesystem != "apfs":
        raise SourceSealError(
            f"source seal v2 requires APFS, observed {filesystem or '<empty>'}"
        )

    attributes = _AttrList()
    attributes.bitmapcount = _ATTR_BIT_MAP_COUNT
    attributes.volattr = _ATTR_VOL_INFO | _ATTR_VOL_UUID
    output = (ctypes.c_ubyte * 20)()
    fgetattrlist = libc.fgetattrlist
    fgetattrlist.argtypes = [ctypes.c_int, ctypes.POINTER(_AttrList), ctypes.c_void_p,
                             ctypes.c_size_t, ctypes.c_ulong]
    fgetattrlist.restype = ctypes.c_int
    if fgetattrlist(fd, ctypes.byref(attributes), output, ctypes.sizeof(output), 0) != 0:
        error = ctypes.get_errno()
        raise SourceSealError(
            f"fgetattrlist volume UUID failed for source fd: {os.strerror(error)}"
        )
    raw = bytes(output)
    returned = int.from_bytes(raw[:4], sys.byteorder)
    if returned != len(raw):
        raise SourceSealError(f"unexpected APFS volume UUID receipt length: {returned}")
    try:
        volume_uuid = str(uuid.UUID(bytes=raw[4:20]))
    except (ValueError, AttributeError) as exc:
        raise SourceSealError("invalid APFS volume UUID receipt") from exc
    return {"filesystem": "apfs", "kind": "darwin-apfs-volume-uuid",
            "uuid": volume_uuid}


def _sealed_identity(row: os.stat_result, volume: dict[str, str]) -> dict[str, Any]:
    return {
        "st_dev_at_seal": row.st_dev,
        "st_ino": row.st_ino,
        "st_size": row.st_size,
        "st_mtime_ns": row.st_mtime_ns,
        "st_ctime_ns": row.st_ctime_ns,
        "volume": dict(volume),
    }


def _hash_regular(path: Path) -> tuple[str, int, dict[str, Any]]:
    resolved, fd, before = _open_regular(path)
    try:
        volume_before = _volume_identity(fd)
        digest, size = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, 16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = _verify_open_regular(resolved, fd, before)
        volume_after = _volume_identity(fd)
        if size != after.st_size or volume_after != volume_before:
            raise SourceSealError(f"source changed while sealing: {resolved}")
        return digest.hexdigest(), size, _sealed_identity(after, volume_after)
    finally:
        os.close(fd)


def _read_regular_bytes(path: Path, *, maximum: int = MAX_JSON_BYTES) -> bytes:
    resolved, fd, before = _open_regular(path)
    try:
        if before.st_size > maximum:
            raise SourceSealError(f"artifact exceeds byte ceiling: {resolved}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise SourceSealError(f"artifact exceeds byte ceiling: {resolved}")
        after = _verify_open_regular(resolved, fd, before)
        if size != after.st_size:
            raise SourceSealError(f"artifact changed while reading: {resolved}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _atomic_bytes_once(path: Path, value: bytes) -> None:
    """Create an immutable archive, accepting only an already equal file."""
    path = path.resolve(strict=False)
    path.relative_to(ROOT.resolve())
    if path.exists():
        if path.is_symlink() or _read_regular_bytes(path) != value:
            raise SourceSealError(f"write-once archive differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _artifact(path: Path) -> dict[str, Any]:
    digest, size, _ = _hash_regular(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def default_path(label: str) -> Path:
    if not isinstance(label, str) or not label or "/" in label or ".." in label:
        raise SourceSealError("model label is unsafe")
    return SEAL_ROOT / f"{label}.json"


def _acquire_exclusive_heavy_lease() -> Any:
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = HEAVY_LOCK.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SourceSealError("active heavy owner prevents source sealing") from exc
    return handle


def freeze(label: str, *, workers: int = 4, output: Path | None = None) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise SourceSealError("hash workers must be in [1,8]")
    lease = _acquire_exclusive_heavy_lease()
    try:
        parameter_path = MANIFEST_ROOT / f"{label}.json"
        parameter = _read_json(parameter_path)
        census_path = Path(parameter.get("census_path", "")).resolve(strict=True)
        census_path.relative_to(ROOT.resolve())
        census = _read_json(census_path)
        model_dir = Path(parameter.get("model_dir", "")).resolve(strict=True)
        model_dir.relative_to(ROOT.resolve())
        rows = census.get("source", {}).get("shards")
        if census.get("status") != "complete" or census.get("label") != label \
                or not isinstance(rows, list) or not rows:
            raise SourceSealError("completed census shard inventory is required")

        jobs: list[tuple[int, Path, dict[str, Any]]] = []
        for ordinal, row in enumerate(rows):
            name = row.get("name") if isinstance(row, dict) else None
            if not isinstance(name, str) or Path(name).name != name \
                    or row.get("ordinal") != ordinal:
                raise SourceSealError("census shard ordering/path is invalid")
            path = (model_dir / name).resolve(strict=True)
            path.relative_to(model_dir)
            jobs.append((ordinal, path, row))

        def one(job: tuple[int, Path, dict[str, Any]]) -> dict[str, Any]:
            ordinal, path, census_row = job
            digest, size, identity = _hash_regular(path)
            if digest != census_row.get("file_sha256") or size != census_row.get("bytes"):
                raise SourceSealError(f"source differs from census: {path}")
            return {"ordinal": ordinal, "name": path.name, "path": str(path),
                    "sha256": digest, "bytes": size, "identity": identity}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            sealed = list(pool.map(one, jobs))
        sealed.sort(key=lambda row: row["ordinal"])
        doc = {
            "schema": SCHEMA, "label": label, "created_at": _now(),
            "source_manifest_sha256": census["source"]["source_manifest_sha256"],
            "parameter_manifest": _artifact(parameter_path),
            "census": _artifact(census_path), "shards": sealed,
            "verification": dict(V2_VERIFICATION),
            "migration": None,
            "source_deletion_permitted": False,
        }
        doc["seal_sha256"] = _hash_value(doc)
        errors = validate_document(doc, verify_structural=True)
        if errors:
            raise SourceSealError("built invalid source seal: " + "; ".join(errors))
        _atomic_json(output or default_path(label), doc)
        return doc
    finally:
        lease.close()


def _artifact_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"path", "sha256", "bytes"} \
        and isinstance(value.get("path"), str) and Path(value["path"]).is_absolute() \
        and _is_sha(value.get("sha256")) and _is_int(value.get("bytes"))


def _volume_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"filesystem", "kind", "uuid"} \
            or value.get("filesystem") != "apfs" \
            or value.get("kind") != "darwin-apfs-volume-uuid" \
            or not isinstance(value.get("uuid"), str):
        return False
    try:
        return str(uuid.UUID(value["uuid"])) == value["uuid"]
    except (ValueError, AttributeError):
        return False


def _v1_identity_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
    } and all(_is_int(value[name]) for name in value)


def _v2_identity_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "st_dev_at_seal", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "volume"
    } and all(_is_int(value[name]) for name in (
        "st_dev_at_seal", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
    )) and _volume_valid(value.get("volume"))


def _migration_valid(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict) or set(value) != {
            "schema", "migrated_at", "source_v1_seal", "content_revalidated",
            "revalidation", "device_transitions", "source_deletion_permitted"}:
        return False
    source = value.get("source_v1_seal")
    if value.get("schema") != MIGRATION_SCHEMA \
            or not isinstance(value.get("migrated_at"), str) \
            or value.get("content_revalidated") is not True \
            or value.get("revalidation") != "full_sha256_all_shards" \
            or value.get("source_deletion_permitted") is not False \
            or not isinstance(source, dict) or set(source) != {
                "path", "sha256", "bytes", "seal_sha256", "schema"} \
            or not isinstance(source.get("path"), str) \
            or not Path(source["path"]).is_absolute() \
            or not _is_sha(source.get("sha256")) or not _is_int(source.get("bytes")) \
            or not _is_sha(source.get("seal_sha256")) \
            or source.get("schema") != SCHEMA_V1:
        return False
    transitions = value.get("device_transitions")
    return isinstance(transitions, list) and all(
        isinstance(row, dict) and set(row) == {
            "ordinal", "st_dev_v1", "st_dev_at_migration"
        } and _is_int(row["ordinal"]) and _is_int(row["st_dev_v1"]) \
        and _is_int(row["st_dev_at_migration"])
        for row in transitions
    )


def _verify_v1_row(path: Path, identity: dict[str, Any]) -> None:
    resolved, fd, before = _open_regular(path)
    try:
        if _identity(before) != identity:
            raise SourceSealError(f"sealed v1 source identity changed: {resolved}")
        _verify_open_regular(resolved, fd, before)
    finally:
        os.close(fd)


def _verify_v2_row(path: Path, identity: dict[str, Any]) -> None:
    resolved, fd, before = _open_regular(path)
    try:
        volume_before = _volume_identity(fd)
        stable_live = {
            "st_ino": before.st_ino,
            "st_size": before.st_size,
            "st_mtime_ns": before.st_mtime_ns,
            "st_ctime_ns": before.st_ctime_ns,
        }
        stable_sealed = {name: identity[name] for name in stable_live}
        if volume_before != identity["volume"] or stable_live != stable_sealed:
            raise SourceSealError(f"sealed v2 source identity changed: {resolved}")
        _verify_open_regular(resolved, fd, before)
        if _volume_identity(fd) != volume_before:
            raise SourceSealError(f"sealed v2 volume identity changed: {resolved}")
    finally:
        os.close(fd)


def validate_document(doc: Any, *, verify_structural: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["source seal keys are invalid"]
    schema = doc.get("schema")
    common = {"schema", "label", "created_at", "source_manifest_sha256",
              "parameter_manifest", "census", "shards", "verification",
              "source_deletion_permitted", "seal_sha256"}
    required = common if schema == SCHEMA_V1 else common | {"migration"}
    if schema not in {SCHEMA_V1, SCHEMA} or set(doc) != required:
        return ["source seal keys are invalid"]
    if doc.get("source_deletion_permitted") is not False:
        errors.append("source seal schema/policy is invalid")
    if doc.get("seal_sha256") != _hash_value(_without(doc, "seal_sha256")):
        errors.append("source seal hash mismatch")
    expected_verification = V1_VERIFICATION if schema == SCHEMA_V1 else V2_VERIFICATION
    if doc.get("verification") != expected_verification:
        errors.append("source seal verification contract is invalid")
    if not isinstance(doc.get("label"), str) or not doc["label"] \
            or "/" in doc["label"] or ".." in doc["label"] \
            or not isinstance(doc.get("created_at"), str) \
            or not _is_sha(doc.get("source_manifest_sha256")):
        errors.append("source seal authority fields are invalid")
    for name in ("parameter_manifest", "census"):
        if not _artifact_valid(doc.get(name)):
            errors.append(f"source seal {name} artifact is invalid")
    if schema == SCHEMA and not _migration_valid(doc.get("migration")):
        errors.append("source seal migration receipt is invalid")
    rows = doc.get("shards")
    if not isinstance(rows, list) or not rows:
        errors.append("source seal shard inventory is empty")
        return errors
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "ordinal", "name", "path", "sha256", "bytes", "identity"} \
                or row.get("ordinal") != ordinal:
            errors.append(f"source seal shard[{ordinal}] keys/order invalid")
            continue
        identity = row.get("identity")
        identity_valid = _v1_identity_valid(identity) if schema == SCHEMA_V1 \
            else _v2_identity_valid(identity)
        if not isinstance(row.get("name"), str) \
                or Path(row.get("name", "")).name != row.get("name") \
                or not isinstance(row.get("path"), str) \
                or not Path(row["path"]).is_absolute() \
                or not _is_sha(row.get("sha256")) or not _is_int(row.get("bytes")) \
                or not identity_valid:
            errors.append(f"source seal shard[{ordinal}] identity is invalid")
            continue
        try:
            path = Path(row["path"]).resolve(strict=False)
            path.relative_to(ROOT.resolve())
            if path.name != row["name"] or row["bytes"] != row["identity"]["st_size"]:
                errors.append(f"source seal shard[{ordinal}] path/size mismatch")
            if verify_structural:
                if schema == SCHEMA_V1:
                    _verify_v1_row(path, identity)
                else:
                    _verify_v2_row(path, identity)
        except (OSError, KeyError, TypeError, ValueError, SourceSealError):
            errors.append(f"source seal shard[{ordinal}] identity is invalid")
    return errors


def lookup(path: Path, *, seal_root: Path = SEAL_ROOT) -> tuple[str, int] | None:
    """Return a digest after strict v1 or stable-volume v2 fd/path verification."""
    try:
        if Path(path).is_symlink():
            return None
        resolved = path.resolve(strict=True)
        resolved.relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return None
    if not seal_root.is_dir() or seal_root.is_symlink():
        return None
    for seal_path in sorted(seal_root.glob("*.json")):
        doc = _read_json(seal_path)
        errors = validate_document(doc, verify_structural=False)
        if errors:
            raise SourceSealError(f"invalid source seal {seal_path}: {'; '.join(errors)}")
        for row in doc["shards"]:
            if Path(row["path"]).resolve(strict=False) != resolved:
                continue
            if doc["schema"] == SCHEMA_V1:
                _verify_v1_row(resolved, row["identity"])
            else:
                _verify_v2_row(resolved, row["identity"])
            return row["sha256"], row["bytes"]
    return None


def migrate_v1(seal_path: Path, *, output: Path | None = None,
               archive: Path | None = None, workers: int = 4) -> dict[str, Any]:
    """Fully revalidate a v1 seal once and replace it with an APFS-bound v2 seal.

    This operation intentionally requires the exclusive heavy lease.  It never
    runs implicitly in ``lookup`` or a worker process: migration is a controlled
    drained-checkpoint action because every source byte is read again.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise SourceSealError("migration hash workers must be in [1,8]")
    lease = _acquire_exclusive_heavy_lease()
    try:
        seal_path = Path(seal_path).resolve(strict=True)
        seal_path.relative_to(ROOT.resolve())
        old_raw = _read_regular_bytes(seal_path)
        try:
            old = json.loads(old_raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SourceSealError(f"cannot parse v1 seal {seal_path}: {exc}") from exc
        errors = validate_document(old, verify_structural=False)
        if errors or old.get("schema") != SCHEMA_V1:
            detail = "; ".join(errors) if errors else "seal is not v1"
            raise SourceSealError(f"v1 migration input is invalid: {detail}")

        old_file_sha = hashlib.sha256(old_raw).hexdigest()
        archive_path = (Path(archive) if archive is not None else
                        SEAL_ROOT / "v1_archive" /
                        f"{old['label']}-{old_file_sha}.json").resolve(strict=False)
        archive_path.relative_to(ROOT.resolve())
        _atomic_bytes_once(archive_path, old_raw)
        archived_raw = _read_regular_bytes(archive_path)
        if hashlib.sha256(archived_raw).hexdigest() != old_file_sha:
            raise SourceSealError("archived v1 seal hash differs after durable write")

        def migrate_row(item: tuple[int, dict[str, Any]]) \
                -> tuple[dict[str, Any], dict[str, int]]:
            ordinal, row = item
            path = Path(row["path"])
            digest, size, identity = _hash_regular(path)
            prior = row["identity"]
            stable_names = ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if digest != row["sha256"] or size != row["bytes"] \
                    or any(prior[name] != identity[name] for name in stable_names):
                raise SourceSealError(
                    f"v1 source changed beyond device renumbering: {path.resolve(strict=False)}"
                )
            transition = {
                "ordinal": ordinal,
                "st_dev_v1": prior["st_dev"],
                "st_dev_at_migration": identity["st_dev_at_seal"],
            }
            sealed_row = {
                "ordinal": ordinal, "name": row["name"],
                "path": str(path.resolve(strict=True)), "sha256": digest,
                "bytes": size, "identity": identity,
            }
            return sealed_row, transition

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            migrated_rows = list(pool.map(migrate_row, enumerate(old["shards"])))
        sealed = [row for row, _transition in migrated_rows]
        transitions = [transition for _row, transition in migrated_rows]

        migrated_at = _now()
        doc = {
            "schema": SCHEMA,
            "label": old["label"],
            "created_at": migrated_at,
            "source_manifest_sha256": old["source_manifest_sha256"],
            "parameter_manifest": dict(old["parameter_manifest"]),
            "census": dict(old["census"]),
            "shards": sealed,
            "verification": dict(V2_VERIFICATION),
            "migration": {
                "schema": MIGRATION_SCHEMA,
                "migrated_at": migrated_at,
                "source_v1_seal": {
                    "path": str(archive_path),
                    "sha256": old_file_sha,
                    "bytes": len(old_raw),
                    "seal_sha256": old["seal_sha256"],
                    "schema": SCHEMA_V1,
                },
                "content_revalidated": True,
                "revalidation": "full_sha256_all_shards",
                "device_transitions": transitions,
                "source_deletion_permitted": False,
            },
            "source_deletion_permitted": False,
        }
        doc["seal_sha256"] = _hash_value(doc)
        errors = validate_document(doc, verify_structural=True)
        if errors:
            raise SourceSealError("built invalid migrated seal: " + "; ".join(errors))
        output_path = (Path(output) if output is not None else seal_path).resolve(
            strict=False
        )
        output_path.relative_to(ROOT.resolve())
        if output_path.exists() and output_path.is_symlink():
            raise SourceSealError(f"migration output is symlinked: {output_path}")
        _atomic_json(output_path, doc)
        persisted = _read_json(output_path)
        errors = validate_document(persisted, verify_structural=True)
        if errors or persisted != doc:
            raise SourceSealError(
                "persisted migrated seal failed verification: " + "; ".join(errors)
            )
        return doc
    finally:
        lease.close()


def install_hash_reuse(base_helper: Any, *, seal_root: Path = SEAL_ROOT,
                       attribute: str = "_hash_file") \
        -> Callable[[Path], tuple[str, int]]:
    original = getattr(base_helper, attribute)
    if getattr(original, "_doctor_v5_source_seal_reuse", False):
        return getattr(original, "_doctor_v5_source_seal_original")

    def sealed_hash(path: Path) -> tuple[str, int]:
        match = lookup(Path(path), seal_root=seal_root)
        return match if match is not None else original(path)

    sealed_hash._doctor_v5_source_seal_reuse = True  # type: ignore[attr-defined]
    sealed_hash._doctor_v5_source_seal_original = original  # type: ignore[attr-defined]
    setattr(base_helper, attribute, sealed_hash)
    return original


def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT) as raw:
        root = Path(raw)
        source = root / "source.bin"
        source.write_bytes(b"doctor-v5-source-seal" * 1024)
        digest, size, identity = _hash_regular(source)
        doc = {
            "schema": SCHEMA, "label": "fixture", "created_at": _now(),
            "source_manifest_sha256": "0" * 64,
            "parameter_manifest": {"path": str(source), "sha256": digest, "bytes": size},
            "census": {"path": str(source), "sha256": digest, "bytes": size},
            "shards": [{"ordinal": 0, "name": source.name, "path": str(source),
                        "sha256": digest, "bytes": size, "identity": identity}],
            "verification": dict(V2_VERIFICATION),
            "migration": None,
            "source_deletion_permitted": False,
        }
        doc["seal_sha256"] = _hash_value(doc)
        assert not validate_document(doc, verify_structural=True)
        seal_root = root / "seals"
        _atomic_json(seal_root / "fixture.json", doc)
        assert lookup(source, seal_root=seal_root) == (digest, size)
        source.write_bytes(source.read_bytes() + b"drift")
        try:
            lookup(source, seal_root=seal_root)
        except SourceSealError:
            pass
        else:
            raise AssertionError("structural drift did not fail closed")
    print(json.dumps({"status": "ok", "schema": SCHEMA}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--label", required=True)
    freeze_parser.add_argument("--workers", type=int, default=4)
    freeze_parser.add_argument("--output", type=Path)
    migrate = sub.add_parser("migrate-v1")
    migrate.add_argument("--seal", required=True, type=Path)
    migrate.add_argument("--output", type=Path)
    migrate.add_argument("--archive", type=Path)
    migrate.add_argument("--workers", type=int, default=4)
    verify = sub.add_parser("verify")
    verify.add_argument("--seal", required=True, type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze(args.label, workers=args.workers, output=args.output),
                         indent=2, sort_keys=True))
    elif args.command == "migrate-v1":
        print(json.dumps(migrate_v1(args.seal, output=args.output, archive=args.archive,
                                    workers=args.workers),
                         indent=2, sort_keys=True))
    elif args.command == "verify":
        errors = validate_document(_read_json(args.seal), verify_structural=True)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 2
    else:
        _selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
