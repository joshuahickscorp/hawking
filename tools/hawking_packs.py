#!/usr/bin/env python3
"""Fetch, hydrate, and verify immutable Hawking source packs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "packs.lock.json"
MANIFEST_NAME = "pack.manifest.json"
PAYLOAD = "payload"
PACK_SCHEMA = "hawking.source_pack.v1"
LOCK_SCHEMA = "hawking.pack_lock.v1"
IGNORED_PARTS = {"target", "__pycache__", ".pytest_cache"}
ALLOWED_DESTINATION_ROOTS = {"crates", "tools", "vendor"}
MAX_MANIFEST_BYTES = 4 * 1024 * 1024

def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _locked_archive(path: Path, row: dict[str, object]) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != row["archive_bytes"]:
                return False
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest() == row["archive_sha256"]
    except OSError:
        return False

def load_lock(path: Path = LOCK_PATH) -> dict[str, object]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA or not isinstance(lock.get("packs"), list):
        raise ValueError(f"invalid pack lock: {path}")
    if safe_relative(lock.get("hydrate_root"), label="hydrate_root").as_posix() != ".hawking/packs":
        raise ValueError("hydrate_root must be .hawking/packs")
    safe_relative(lock.get("default_cache"), label="default_cache")
    rows = [row for row in lock["packs"] if isinstance(row, dict)]  # type: ignore[index]
    if len(rows) != len(lock["packs"]) or len({row.get("id") for row in rows}) != len(rows):  # type: ignore[arg-type]
        raise ValueError("pack rows and ids must be unique objects")
    for row in rows:
        validate_lock_row(row)
    return lock

def primary_lines(path: Path, suffixes: set[str]) -> int:
    if path.suffix.lower() not in suffixes:
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)

def source_inventory(source: Path, suffixes: set[str]) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    total_lines = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source packs cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        parts = PurePosixPath(relative).parts
        if relative == MANIFEST_NAME or any(part in IGNORED_PARTS for part in parts):
            continue
        raw = path.read_bytes()
        lines = primary_lines(path, suffixes)
        total_lines += lines
        rows.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "mode": 0o755 if path.stat().st_mode & 0o111 else 0o644,
                "primary_lines": lines,
            }
        )
    return rows, total_lines

def safe_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"unsafe {label}: {value}")
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"unsafe {label}: {value}")
    return path

def _single_name(value: object, *, label: str) -> str:
    path = safe_relative(value, label=label)
    if len(path.parts) != 1:
        raise ValueError(f"{label} must be one filename: {value}")
    return path.as_posix()

def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value)

def _repo_path(relative: PurePosixPath, *, allowed: set[str] | None = None) -> Path:
    if allowed is not None and relative.parts[0] not in allowed:
        raise ValueError(f"protected repository destination: {relative}")
    return _child_path(ROOT, relative)

def _child_path(root: Path, relative: PurePosixPath) -> Path:
    current = root
    if current.is_symlink():
        raise ValueError(f"symlinked path is forbidden: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked path is forbidden: {current}")
    return current

def validate_lock_row(row: dict[str, object]) -> None:
    for key in ("id", "version", "archive_sha256", "manifest_sha256", "tree_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid pack lock field {key}")
    _single_name(row.get("id"), label="id")
    _single_name(row.get("archive_name"), label="archive_name")
    _single_name(row.get("hydrate_name"), label="hydrate_name")
    for key in ("archive_bytes", "primary_lines", "control_primary_lines"):
        if not isinstance(row.get(key), int) or int(row[key]) < 0:
            raise ValueError(f"invalid pack lock field {key}")
    for key in ("archive_sha256", "manifest_sha256", "tree_sha256",
                "control_inventory_sha256"):
        if not _hex(row.get(key), 64):
            raise ValueError(f"invalid pack lock field {key}")
    for key in ("source_commit", "source_tree"):
        if not _hex(row.get(key), 40):
            raise ValueError(f"invalid pack lock field {key}")
    repository = row.get("source_repository")
    if not isinstance(repository, str) or not repository.startswith("https://"):
        raise ValueError("source_repository must be HTTPS")
    _single_name(row.get("source_tag"), label="source_tag")
    control_files = row.get("control_files")
    if not isinstance(control_files, list):
        raise ValueError(f"{row['id']}: control_files must be a list")
    normalized = []
    for control in control_files:
        if not isinstance(control, dict):
            raise ValueError(f"{row['id']}: invalid control file")
        path = safe_relative(control.get("path"), label="control file").as_posix()
        size, lines = control.get("bytes"), control.get("lines")
        if type(size) is not int or size < 0 or type(lines) is not int or lines < 0 \
                or not _hex(control.get("sha256"), 64):
            raise ValueError(f"{row['id']}: invalid control file metadata")
        normalized.append(
            {"path": path, "bytes": size, "sha256": control["sha256"], "lines": lines}
        )
    if normalized != sorted(normalized, key=lambda value: value["path"]) \
            or len({value["path"] for value in normalized}) != len(normalized):
        raise ValueError(f"{row['id']}: control_files must be sorted and unique")
    if sum(value["lines"] for value in normalized) != row["control_primary_lines"] \
            or sha256_bytes(canonical_json(normalized)) != row["control_inventory_sha256"]:
        raise ValueError(f"{row['id']}: control inventory differs from lock")
    mirrors = row.get("mirrors")
    if not isinstance(mirrors, list) or not mirrors \
            or not all(isinstance(mirror, str) and mirror and
                       ("://" not in mirror or mirror.startswith(("https://", "file://")))
                       for mirror in mirrors):
        raise ValueError(f"{row['id']}: mirrors must be nonempty strings")
    mappings = row.get("materialize")
    if not isinstance(mappings, list):
        raise ValueError(f"{row['id']}: materialize must be a list")
    destinations: list[PurePosixPath] = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError(f"{row['id']}: invalid materialization mapping")
        safe_relative(mapping.get("source"), label="materialization source")
        destination = safe_relative(mapping.get("destination"),
                                    label="materialization destination")
        if destination.parts[0] not in ALLOWED_DESTINATION_ROOTS:
            raise ValueError(f"protected materialization destination: {destination}")
        if any(destination.parts[:len(other.parts)] == other.parts
               or other.parts[:len(destination.parts)] == destination.parts
               for other in destinations):
            raise ValueError(f"overlapping materialization destination: {destination}")
        destinations.append(destination)

def locked_manifest(row: dict[str, object],
                    manifest: dict[str, object]) -> list[dict[str, object]]:
    rows = validated_manifest(manifest)
    checks = {
        "pack_id": row["id"],
        "version": row["version"],
        "tree_sha256": row["tree_sha256"],
        "primary_lines": row["primary_lines"],
    }
    if any(str(manifest.get(key)) != str(value) for key, value in checks.items()):
        raise ValueError(f"{row['id']}: hydrated manifest differs from lock")
    if sha256_bytes(canonical_json(manifest)) != row["manifest_sha256"]:
        raise ValueError(f"{row['id']}: hydrated manifest SHA-256 differs from lock")
    return rows

def validated_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    if manifest.get("schema") != PACK_SCHEMA or not isinstance(manifest.get("files"), list):
        raise ValueError("invalid pack manifest")
    rows = manifest["files"]
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pack manifest rows must be objects")
        relative = safe_relative(row.get("path"), label="manifest path").as_posix()
        paths.append(relative)
        if (
            row.get("path") != relative
            or not isinstance(row.get("bytes"), int)
            or int(row["bytes"]) < 0
            or not isinstance(row.get("primary_lines"), int)
            or int(row["primary_lines"]) < 0
            or row.get("mode") not in (0o644, 0o755)
            or not isinstance(row.get("sha256"), str)
            or len(str(row["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in str(row["sha256"]))
        ):
            raise ValueError(f"invalid manifest row: {relative}")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("manifest paths must be unique and sorted")
    if sha256_bytes(canonical_json(rows)) != manifest.get("tree_sha256"):
        raise ValueError("manifest tree SHA-256 differs")
    if sum(int(row["primary_lines"]) for row in rows) != manifest.get("primary_lines"):
        raise ValueError("manifest primary LOC differs")
    return rows

def _archive_manifest(archive: tarfile.TarFile, label: Path) -> dict[str, object]:
    members = [member for member in archive.getmembers() if member.name == MANIFEST_NAME]
    if len(members) != 1 or not members[0].isreg() \
            or members[0].size > MAX_MANIFEST_BYTES:
        raise ValueError(f"{label}: expected one bounded regular {MANIFEST_NAME}")
    handle = archive.extractfile(members[0])
    if handle is None:
        raise ValueError(f"{label}: missing {MANIFEST_NAME}")
    manifest = json.loads(handle.read(MAX_MANIFEST_BYTES + 1))
    if not isinstance(manifest, dict):
        raise ValueError(f"{label}: invalid pack manifest")
    validated_manifest(manifest)
    return manifest

def verify_tree(root: Path, manifest: dict[str, object], suffixes: set[str]) -> dict[str, object]:
    expected_rows = validated_manifest(manifest)
    actual_rows, actual_lines = source_inventory(root, suffixes)
    expected_tree = str(manifest["tree_sha256"])
    actual_tree = sha256_bytes(canonical_json(actual_rows))
    errors: list[str] = []
    if actual_rows != expected_rows:
        errors.append("file inventory differs")
    if actual_tree != expected_tree:
        errors.append("tree SHA-256 differs")
    if actual_lines != manifest.get("primary_lines"):
        errors.append("primary LOC differs")
    return {
        "valid": not errors,
        "errors": errors,
        "tree_sha256": actual_tree,
        "primary_lines": actual_lines,
        "file_count": len(actual_rows),
    }

def _safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or member.issym()
        or member.islnk()
        or not member.isreg()
    ):
        raise ValueError(f"unsafe archive member: {member.name}")
    if not (path == PurePosixPath(MANIFEST_NAME) or path.parts[:1] == (PAYLOAD,)):
        raise ValueError(f"unexpected archive member: {member.name}")
    return path

def _archive_payload(archive: tarfile.TarFile, manifest: dict[str, object],
                     ) -> list[tuple[tarfile.TarInfo, dict[str, object]]]:
    expected = {f"{PAYLOAD}/{row['path']}": row
                for row in validated_manifest(manifest)}
    seen: set[str] = set()
    payload: list[tuple[tarfile.TarInfo, dict[str, object]]] = []
    for member in archive.getmembers():
        path = _safe_member(member)
        name = path.as_posix()
        if name in seen:
            raise ValueError(f"duplicate archive member: {name}")
        seen.add(name)
        if name == MANIFEST_NAME:
            continue
        row = expected.get(name)
        if row is None:
            raise ValueError(f"unexpected archive payload: {name}")
        if member.size != row["bytes"] or member.mode & 0o777 != row["mode"]:
            raise ValueError(f"archive metadata differs: {name}")
        payload.append((member, row))
    missing = sorted(set(expected) - seen)
    if MANIFEST_NAME not in seen or missing:
        raise ValueError(f"archive payload incomplete: {', '.join(missing)}")
    return payload

@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"pack lock is not regular: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

def hydrate_archive(archive_path: Path, destination: Path, *,
                    row: dict[str, object], suffixes: set[str],
                    force: bool = False) -> dict[str, object]:
    validate_lock_row(row)
    staging: Path | None = None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(archive_path, flags)
        with os.fdopen(descriptor, "rb") as raw_archive:
            archive_stat = os.fstat(raw_archive.fileno())
            if not stat.S_ISREG(archive_stat.st_mode) or archive_stat.st_size != row["archive_bytes"]:
                raise ValueError(f"{archive_path}: archive size/type differs from lock")
            digest = hashlib.sha256()
            for chunk in iter(lambda: raw_archive.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != row["archive_sha256"]:
                raise ValueError(f"{archive_path}: archive SHA-256 differs from lock")
            raw_archive.seek(0)
            with tarfile.open(fileobj=raw_archive, mode="r:") as archive:
                manifest = _archive_manifest(archive, archive_path)
                locked_manifest(row, manifest)
                if destination.exists() and not force:
                    current = verify_tree(destination, manifest, suffixes)
                    if current["valid"]:
                        return {"changed": False, "destination": str(destination), **current}
                    raise ValueError(f"{destination}: existing hydration is not the pinned tree")
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.",
                                               dir=destination.parent))
                for member, payload_row in _archive_payload(archive, manifest):
                    relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                    target = staging / relative.as_posix()
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError(f"cannot read archive member: {member.name}")
                    raw_bytes = handle.read(int(payload_row["bytes"]) + 1)
                    if sha256_bytes(raw_bytes) != payload_row["sha256"]:
                        raise ValueError(f"archive payload SHA-256 differs: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw_bytes)
                    target.chmod(member.mode & 0o777)
        (staging / MANIFEST_NAME).write_bytes(canonical_json(manifest))
        verified = verify_tree(staging, manifest, suffixes)
        if not verified["valid"]:
            raise ValueError(f"hydrated tree failed verification: {verified['errors']}")
        backup = destination.with_name(f".{destination.name}.{os.getpid()}.bak")
        _remove(backup)
        moved_old = False
        try:
            if destination.exists():
                os.replace(destination, backup)
                moved_old = True
            os.replace(staging, destination)
            if moved_old:
                _remove(backup)
        except Exception:
            if moved_old and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        return {"changed": True, "destination": str(destination), **verified}
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

def cache_root(lock: dict[str, object]) -> Path:
    env_name = str(lock.get("cache_env") or "HAWKING_PACK_CACHE")
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT / str(lock.get("default_cache") or ".hawking-pack-cache")).resolve()

def pack_destination(lock: dict[str, object], row: dict[str, object]) -> Path:
    root = safe_relative(lock.get("hydrate_root"), label="hydrate_root")
    name = _single_name(row.get("hydrate_name"), label="hydrate_name")
    return _repo_path(root / name)

def cached_archive(lock: dict[str, object], row: dict[str, object]) -> Path:
    return cache_root(lock) / _single_name(row.get("archive_name"), label="archive_name")

def _stream_archive(source, target, row: dict[str, object]) -> None:
    expected_bytes = int(row["archive_bytes"])
    digest = hashlib.sha256()
    written = 0
    while True:
        chunk = source.read(min(1024 * 1024, expected_bytes - written + 1))
        if not chunk:
            break
        written += len(chunk)
        if written > expected_bytes:
            raise ValueError("archive exceeds locked byte count")
        digest.update(chunk)
        target.write(chunk)
    if written != expected_bytes or digest.hexdigest() != row["archive_sha256"]:
        raise ValueError("downloaded archive bytes or SHA-256 differ")

def fetch_pack(lock: dict[str, object], row: dict[str, object], *,
               offline: bool = False) -> dict[str, object]:
    destination = cached_archive(lock, row)
    if _locked_archive(destination, row):
        return {"changed": False, "archive": str(destination)}
    mirrors = row.get("mirrors")
    if not isinstance(mirrors, list) or not mirrors:
        raise ValueError(f"{row['id']}: no mirrors and no valid offline cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for mirror in mirrors:
        temporary: Path | None = None
        try:
            source = str(mirror)
            remote = source.startswith("https://")
            if offline and remote:
                continue
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent)
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as target:
                if source.startswith(("https://", "file://")):
                    with urllib.request.urlopen(source, timeout=60) as response:
                        _stream_archive(response, target, row)
                else:
                    source_path = Path(source).expanduser()
                    if not source_path.is_absolute():
                        source_path = ROOT / source_path
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(source_path, flags)
                    with os.fdopen(descriptor, "rb") as handle:
                        _stream_archive(handle, target, row)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
            return {"changed": True, "archive": str(destination), "mirror": source}
        except Exception as exc:  # mirrors are an ordered fallback list
            errors.append(f"{mirror}: {type(exc).__name__}: {exc}")
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError(f"{row['id']}: every mirror failed: {'; '.join(errors)}")

def _same_materialization(source: Path, destination: Path, suffixes: set[str]) -> bool:
    if source.is_file() and not source.is_symlink():
        return (
            destination.is_file()
            and not destination.is_symlink()
            and source.read_bytes() == destination.read_bytes()
            and (source.stat().st_mode & 0o777) == (destination.stat().st_mode & 0o777)
        )
    if not source.is_dir() or source.is_symlink() \
            or not destination.is_dir() or destination.is_symlink():
        return False
    return source_inventory(source, suffixes)[0] == source_inventory(destination, suffixes)[0]

def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _recover_transaction(transaction: Path, row: dict[str, object]) -> None:
    journal = transaction / "journal.json"
    if journal.is_symlink():
        raise ValueError(f"symlinked transaction journal is forbidden: {journal}")
    if journal.exists() and not journal.is_file():
        raise ValueError(f"transaction journal is not regular: {journal}")
    if journal.is_file():
        document = json.loads(journal.read_text(encoding="utf-8"))
        records = document.get("records") if isinstance(document, dict) else None
        mappings = row["materialize"]
        if not isinstance(document, dict) or set(document) != {"records"} or not isinstance(records, list):
            raise ValueError("invalid materialization recovery journal")
        previous = -1
        for record in records:
            index = record.get("index") if isinstance(record, dict) else None
            if not isinstance(record, dict) \
                    or set(record) != {"index", "destination", "had_destination"} \
                    or type(index) is not int or not previous < index < len(mappings) \
                    or type(record["had_destination"]) is not bool \
                    or record["destination"] != mappings[index]["destination"]:
                raise ValueError("invalid materialization recovery record")
            previous = index
        for record in reversed(records):
            destination = _repo_path(
                safe_relative(record["destination"], label="recovery destination"),
                allowed=ALLOWED_DESTINATION_ROOTS,
            )
            backup = transaction / "backup" / str(record["index"])
            if backup.is_symlink():
                raise ValueError(f"symlinked transaction backup is forbidden: {backup}")
            if backup.exists():
                _remove(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
            elif not record["had_destination"]:
                _remove(destination)
    shutil.rmtree(transaction, ignore_errors=True)

def materialize_pack(hydrated: Path, row: dict[str, object],
                     suffixes: set[str], *, force: bool) -> dict[str, object]:
    validate_lock_row(row)
    mappings = row["materialize"]
    results: list[dict[str, object]] = []
    transaction = _repo_path(PurePosixPath(
        ".hawking", "transactions", str(row["id"])))
    if transaction.exists():
        _recover_transaction(transaction, row)
    records: list[dict[str, object]] = []
    for index, mapping in enumerate(mappings):  # type: ignore[assignment]
        source_rel = safe_relative(mapping.get("source"), label="materialization source")
        destination_rel = safe_relative(mapping.get("destination"),
                                        label="materialization destination")
        source = _child_path(hydrated, source_rel)
        destination = _repo_path(destination_rel, allowed=ALLOWED_DESTINATION_ROOTS)
        changed = not _same_materialization(source, destination, suffixes)
        if changed and destination.exists() and not force:
            raise ValueError(f"{destination}: existing materialization differs")
        if changed:
            stage = transaction / "stage" / str(index)
            stage.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir() and not source.is_symlink():
                shutil.copytree(source, stage, copy_function=shutil.copy2)
            elif source.is_file() and not source.is_symlink():
                shutil.copy2(source, stage)
            else:
                raise ValueError(f"materialization source is not regular: {source}")
            if not _same_materialization(source, stage, suffixes):
                raise ValueError(f"materialization copy differs: {destination}")
            records.append({"index": index, "destination": destination_rel.as_posix(),
                            "had_destination": destination.exists()})
        results.append({"source": source_rel.as_posix(),
                        "destination": destination_rel.as_posix(),
                        "changed": changed, "valid": not changed})
    if records:
        _atomic_json(transaction / "journal.json", {"records": records})
        try:
            for record in records:
                index = str(record["index"])
                destination = _repo_path(
                    safe_relative(record["destination"], label="destination"),
                    allowed=ALLOWED_DESTINATION_ROOTS,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if record["had_destination"]:
                    backup = transaction / "backup" / index
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, backup)
                os.replace(transaction / "stage" / index, destination)
            for result, mapping in zip(results, mappings):
                source = _child_path(hydrated, safe_relative(
                    mapping["source"], label="source"))
                destination = _repo_path(safe_relative(
                    mapping["destination"], label="destination"),
                    allowed=ALLOWED_DESTINATION_ROOTS)
                result["valid"] = _same_materialization(source, destination, suffixes)
            if not all(result["valid"] for result in results):
                raise ValueError("materialization transaction verification failed")
        except Exception:
            _recover_transaction(transaction, row)
            raise
        shutil.rmtree(transaction)
    receipt = {
        "schema": "hawking.pack_activation.v1",
        "pack_id": row["id"],
        "version": row["version"],
        "archive_sha256": row["archive_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "tree_sha256": row["tree_sha256"],
        "source_commit": row["source_commit"],
        "source_tree": row["source_tree"],
        "control_inventory_sha256": row["control_inventory_sha256"],
        "materializations": results,
    }
    receipt_path = _repo_path(PurePosixPath(
        ".hawking", "activations", f"{row['id']}.json"))
    _atomic_json(receipt_path, receipt)
    return receipt

def verify_materializations(hydrated: Path, row: dict[str, object],
                            suffixes: set[str]) -> dict[str, object]:
    results = []
    for mapping in row.get("materialize", []):
        source_rel = safe_relative(mapping["source"], label="materialization source")
        destination_rel = safe_relative(mapping["destination"],
                                        label="materialization destination")
        valid = _same_materialization(
            _child_path(hydrated, source_rel),
            _repo_path(destination_rel, allowed=ALLOWED_DESTINATION_ROOTS),
            suffixes,
        )
        results.append({"destination": destination_rel.as_posix(), "valid": valid})
    return {"valid": all(result["valid"] for result in results), "paths": results}

def lock_suffixes() -> set[str]:
    contract = json.loads((ROOT / "loc_floor.json").read_text(encoding="utf-8"))
    policy = contract.get("policy")
    if isinstance(policy, dict) and isinstance(policy.get("source_suffixes"), list):
        return {str(value) for value in policy["source_suffixes"]}
    return {str(value) for value in contract["accounting"]["primary_suffixes"]}

def select_rows(lock: dict[str, object], requested: list[str]) -> list[dict[str, object]]:
    rows = [row for row in lock["packs"] if isinstance(row, dict)]  # type: ignore[index]
    if not requested:
        return rows
    by_id = {str(row["id"]): row for row in rows}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"unknown packs: {', '.join(missing)}")
    return [by_id[pack_id] for pack_id in requested]

def verify_locked_hydration(lock: dict[str, object], row: dict[str, object],
                            suffixes: set[str]) -> dict[str, object]:
    destination = pack_destination(lock, row)
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"{row['id']}: hydration manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{row['id']}: hydration manifest is not an object")
    locked_manifest(row, manifest)
    tree = verify_tree(destination, manifest, suffixes)
    materializations = verify_materializations(destination, row, suffixes)
    return {
        **tree,
        "lock_bound": True,
        "materializations": materializations,
        "valid": bool(tree["valid"]) and bool(materializations["valid"]),
    }

def status(lock: dict[str, object], suffixes: set[str]) -> dict[str, object]:
    rows = []
    for row in select_rows(lock, []):
        destination = pack_destination(lock, row)
        archive = cached_archive(lock, row)
        record: dict[str, object] = {
            "id": row["id"],
            "destination": str(destination),
            "archive": str(archive),
            "archive_cached": archive.is_file(),
            "hydrated": destination.is_dir(),
        }
        if archive.exists() or archive.is_symlink():
            record["archive_sha256_valid"] = _locked_archive(archive, row)
        if destination.is_dir():
            try:
                record["verification"] = verify_locked_hydration(lock, row, suffixes)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                record["verification"] = {
                    "valid": False,
                    "lock_bound": False,
                    "errors": [str(exc)],
                }
        rows.append(record)
    return {"schema": "hawking.pack_status.v1", "packs": rows}

def selftest() -> None:
    suffixes = {".py", ".rs"}
    with tempfile.TemporaryDirectory(prefix="hawking-pack-selftest-") as raw:
        root = Path(raw)
        source = root / "source"
        source.mkdir()
        (source / "a.py").write_text("print('a')\n", encoding="utf-8")
        nested = source / "nested"
        nested.mkdir()
        (nested / "b.rs").write_text("fn b() {}\n", encoding="utf-8")
        archive = root / "sample.tar"
        files, lines = source_inventory(source, suffixes)
        manifest = {
            "schema": PACK_SCHEMA,
            "pack_id": "sample",
            "version": "1",
            "tree_sha256": sha256_bytes(canonical_json(files)),
            "primary_lines": lines,
            "files": files,
            "provenance": {},
        }
        manifest_path = root / MANIFEST_NAME
        manifest_path.write_bytes(canonical_json(manifest))
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
            bundle.add(manifest_path, arcname=MANIFEST_NAME, recursive=False)
            for file_row in files:
                bundle.add(source / str(file_row["path"]),
                           arcname=f"{PAYLOAD}/{file_row['path']}",
                           recursive=False)
        row = {
            "id": "sample",
            "version": "1",
            "archive_name": "sample.tar",
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "manifest_sha256": sha256_bytes(canonical_json(manifest)),
            "tree_sha256": manifest["tree_sha256"],
            "primary_lines": manifest["primary_lines"],
            "hydrate_name": "sample",
            "materialize": [],
        }
        row.update(
            control_primary_lines=0,
            control_inventory_sha256=sha256_bytes(canonical_json([])),
            control_files=[],
            source_commit="0" * 40, source_tree="0" * 40,
            source_repository="https://example.invalid", source_tag="sample",
            mirrors=["sample.tar"],
        )
        destination = root / "hydrated"
        hydrated = hydrate_archive(archive, destination, row=row,
                                   suffixes=suffixes)
        assert hydrated["valid"] and hydrated["changed"]
        repeated = hydrate_archive(archive, destination, row=row,
                                   suffixes=suffixes)
        assert repeated["valid"] and not repeated["changed"]
        (destination / "a.py").write_text("tampered\n", encoding="utf-8")
        assert not verify_tree(destination, manifest, suffixes)["valid"]
        rewritten = json.loads(json.dumps(manifest))
        rewritten["provenance"] = {"rewritten": True}
        try:
            locked_manifest(row, rewritten)
        except ValueError:
            pass
        else:
            raise AssertionError("self-consistent manifest rewrite escaped the lock")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "hydrate", "verify"):
        command = sub.add_parser(name)
        command.add_argument("packs", nargs="*")
        if name == "fetch":
            command.add_argument("--offline", action="store_true")
        if name == "hydrate":
            command.add_argument("--force", action="store_true")
    sub.add_parser("status")
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    suffixes = lock_suffixes()
    if args.command == "selftest":
        selftest()
        result = {"selftest": "ok"}
    else:
        lock = load_lock()
        if args.command == "status":
            result = status(lock, suffixes)
        else:
            results = []
            for row in select_rows(lock, args.packs):
                if args.command == "fetch":
                    results.append({"id": row["id"],
                                    **fetch_pack(lock, row, offline=args.offline)})
                elif args.command == "hydrate":
                    archive = cached_archive(lock, row)
                    if not archive.is_file():
                        raise ValueError(
                            f"{row['id']}: offline archive missing; run fetch explicitly"
                        )
                    lock_path = _repo_path(PurePosixPath(
                        ".hawking", "locks", f"{row['id']}.lock"))
                    with exclusive_lock(lock_path):
                        hydrated = hydrate_archive(
                            archive, pack_destination(lock, row), row=row,
                            suffixes=suffixes, force=args.force)
                        activation = materialize_pack(
                            pack_destination(lock, row), row, suffixes,
                            force=args.force)
                    results.append({"id": row["id"], **hydrated, "activation": activation})
                else:
                    results.append({"id": row["id"],
                                    **verify_locked_hydration(lock, row, suffixes)})
            result = {"packs": results}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "verify":
        return 1 if any(not row["valid"] for row in result["packs"]) else 0
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"hawking_packs: {exc}", file=sys.stderr)
        sys.exit(2)
