"""Recomposed science module glm52_common (C-SCI-R1)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HAWKING_ARTIFACT_ROOT_ENV = 'HAWKING_ARTIFACT_ROOT'
DEFAULT_HAWKING_ARTIFACT_ROOT = Path.home() / 'Downloads' / 'hawking-evidence'

class Glm52Error(RuntimeError):
    """Raised when a campaign invariant fails."""

def hawking_artifact_root() -> Path:
    """Return `$HAWKING_ARTIFACT_ROOT`, defaulting to `~/Downloads/hawking-evidence`."""
    raw = os.environ.get(HAWKING_ARTIFACT_ROOT_ENV)
    if raw is None or raw == '':
        return DEFAULT_HAWKING_ARTIFACT_ROOT
    return Path(raw).expanduser()

def resolve_artifact(name: str) -> Path:
    """Locate a campaign artifact by basename.

    Search order:

    1. Repository root (preserves today's in-tree behaviour)
    2. ``$HAWKING_ARTIFACT_ROOT`` (default ``~/Downloads/hawking-evidence``)

    Raises ``Glm52Error`` naming the artifact and the exact ``git checkout``
    command that restores it when neither location has a regular file.
    """
    if not isinstance(name, str) or not name or name != Path(name).name:
        raise Glm52Error(f'resolve_artifact expects a basename, got {name!r}')
    external = hawking_artifact_root()
    candidates = (REPO_ROOT / name, external / name)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    restore = f'git checkout HEAD -- {name}'
    searched = ', '.join((str(path) for path in candidates))
    raise Glm52Error(f'campaign artifact not found: {name}\nsearched: {searched}\nrestore with: {restore}')

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != 'seal_sha256'}
    return {**unsigned, 'seal_sha256': hashlib.sha256(canonical(unsigned)).hexdigest()}

def verify_sealed(value: dict[str, Any], *, label: str='artifact') -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Glm52Error(f'{label} is not a JSON object')
    recorded = value.get('seal_sha256')
    expected = seal(value)['seal_sha256']
    if recorded != expected:
        raise Glm52Error(f'{label} seal mismatch: recorded={recorded!r} expected={expected}')
    return value

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise Glm52Error(f'cannot read JSON {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise Glm52Error(f'JSON root is not an object: {path}')
    return value

def read_sealed_json(path: Path) -> dict[str, Any]:
    return verify_sealed(read_json(path), label=str(path))

def sha256_file(path: Path, *, chunk_bytes: int=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b''):
            digest.update(chunk)
    return digest.hexdigest()

def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f'blob {len(data)}\x00'.encode('ascii')
    return hashlib.sha1(header + data).hexdigest()

def atomic_bytes(path: Path, value: bytes, *, mode: int=420) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode('utf-8'))

def atomic_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    atomic_text(path, rendered)

def allocated_bytes(path: Path) -> int:
    """Return local allocated bytes without following a symlink target twice."""
    st = path.lstat()
    return int(st.st_blocks) * 512

def selfcheck() -> dict[str, Any]:
    sample = seal({'schema': 'selfcheck', 'status': 'PASS', 'number': 7})
    verify_sealed(sample)
    corrupted = dict(sample)
    corrupted['number'] = 8
    rejected = False
    try:
        verify_sealed(corrupted)
    except Glm52Error:
        rejected = True
    if not rejected:
        raise AssertionError('seal verifier accepted a modified object')
    return {'status': 'PASS', 'modified_object_rejected': rejected}
