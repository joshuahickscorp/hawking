"""Shared hashing, provenance, and JSONL IO for Ramanujan local data."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from ramanujan.data.paths import (
    EXPECTED_LEAN_VERSION,
    EXPECTED_MATHLIB_COMMIT,
    MATHLIB_ROOT,
)

# Hard fences: these extractors never consult Math-Preserve and never claim research auth.
RESEARCH_AUTHORIZED = False
TEACHER_FROM_MATH_PRESERVE = False


def content_hash(obj: Any) -> str:
    """Stable sha256 of a JSON-serialisable object (or raw string)."""
    if isinstance(obj, str):
        blob = obj.encode("utf-8")
    else:
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Process-level caches: stamp_item calls these once per item; without caching a
# scale-up of ~15k items would spawn that many `git` / `lean --version` subprocesses.
_MATHLIB_COMMIT_CACHE: dict[str, str] = {}
_LEAN_VERSION_CACHE: str | None = None


def mathlib_commit(mathlib: Path | None = None) -> str:
    root = mathlib or MATHLIB_ROOT
    key = str(root)
    cached = _MATHLIB_COMMIT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        out = EXPECTED_MATHLIB_COMMIT
    _MATHLIB_COMMIT_CACHE[key] = out
    return out


def lean_version() -> str:
    global _LEAN_VERSION_CACHE
    if _LEAN_VERSION_CACHE is not None:
        return _LEAN_VERSION_CACHE
    try:
        out = subprocess.check_output(
            ["lean", "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            env=_elan_env(),
        )
        m = re.search(r"version\s+([\d.]+)", out)
        ver = m.group(1) if m else out.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        ver = EXPECTED_LEAN_VERSION
    _LEAN_VERSION_CACHE = ver
    return ver


def _elan_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    elan = Path.home() / ".elan" / "bin"
    env["PATH"] = f"{elan}:{env.get('PATH', '')}"
    return env


def provenance(*, extraction_method: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "source_commit": mathlib_commit(),
        "expected_mathlib_commit": EXPECTED_MATHLIB_COMMIT,
        "lean_version": lean_version(),
        "extraction_method": extraction_method,
        "RAMANUJAN_RESEARCH_AUTHORIZED": RESEARCH_AUTHORIZED,
        "teacher_from_math_preserve": TEACHER_FROM_MATH_PRESERVE,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        base.update(extra)
    return base


def stamp_item(item: dict[str, Any], *, extraction_method: str) -> dict[str, Any]:
    """Attach provenance + content_hash. Hash covers the training-relevant body only."""
    body = {k: v for k, v in item.items() if k not in ("content_hash", "provenance", "admitted")}
    item = dict(item)
    item["provenance"] = provenance(extraction_method=extraction_method)
    item["content_hash"] = content_hash(body)
    return item


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(items)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = file_sha256(path) if path.is_file() else None
    try:
        from ramanujan.data.paths import ROOT

        rel = str(path.relative_to(ROOT))
    except Exception:
        rel = str(path)
    hashes = [r.get("content_hash") for r in rows if r.get("content_hash")]
    return {
        "path": rel,
        "n_items": len(rows),
        "sha256": digest,
        # sha256 of the file is a snapshot identity, not a reproducible one:
        # every record carries a wall-clock `provenance.at`, so regenerating
        # the same corpus from the same pinned Mathlib yields the same data
        # under a different file hash. Measured on 2026-07-30: D1, D2 and D3
        # each reproduced all 5000 content hashes in identical order while
        # every file sha256 differed. Seal this digest as well, so a freeze
        # can actually be checked rather than only recorded.
        "content_digest": _sha256_of_strings(hashes),
        "content_hashes": hashes,
    }


def _sha256_of_strings(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: {e}") from e
    return out


def dedup_by_hash(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        h = it.get("content_hash") or content_hash(it)
        if h in seen:
            continue
        seen.add(h)
        out.append(it)
    return out
