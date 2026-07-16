#!/usr/bin/env python3.12
"""Post-run immutable corpus indexer with an active-heavy-owner interlock."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
from typing import Any

import appendix_contract
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "reports" / "condense" / "doctor_v5_ultra" / "results"
SCHEMA = "hawking.appendix_corpus_index.v3"
SEMANTIC_CLASSES = ("negative_outcome", "failure", "partial")
SEMANTIC_KEYS = {"status", "outcome", "disposition", "state", "result_status"}
NEGATIVE_VALUES = {"negative", "unsupported", "no_effect", "no-effect", "not_supported"}
FAILURE_VALUES = {"failed", "failure", "error", "errored", "crashed", "blocked-execution"}
PARTIAL_VALUES = {"partial", "incomplete", "interrupted", "aborted"}
MAX_SEMANTIC_JSON_BYTES = 16 * 1024 * 1024


def _source_base_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _kind(path: pathlib.Path) -> str:
    name = path.name
    if name == "request.json":
        return "request_receipt"
    if name in {"result.json", "receipt.json"}:
        return "result_receipt"
    if name.endswith(".tq"):
        return "tq_artifact"
    if ".partial" in name:
        return "partial_artifact"
    if path.suffix == ".json":
        return "json_evidence"
    if path.suffix in {".log", ".jsonl"}:
        return "event_log"
    return "other_evidence"


def preview(root: pathlib.Path = DEFAULT_CORPUS) -> dict:
    """Name-only census. Does not open or hash a corpus file."""
    counts: dict[str, int] = {}
    files = 0
    if root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            directory = pathlib.Path(dirpath)
            for name in filenames:
                files += 1
                kind = _kind(directory / name)
                counts[kind] = counts.get(kind, 0) + 1
    return {
        "schema": "hawking.appendix_corpus_preview.v1",
        "root": str(root),
        "exists": root.is_dir(),
        "opens_or_hashes_files": False,
        "file_count": files,
        "kind_counts": dict(sorted(counts.items())),
    }


def _stable_hash(path: pathlib.Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"corpus file changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _explicit_semantics(path: pathlib.Path, size: int) -> list[str]:
    """Conservatively census only explicit filename or structured outcome semantics."""
    semantics: set[str] = set()
    low_name = path.name.lower()
    if ".partial" in low_name or low_name.startswith("partial"):
        semantics.add("partial")
    if low_name.startswith(("failed", "failure", "error")) or ".failed." in low_name:
        semantics.add("failure")
    if path.suffix != ".json" or size > MAX_SEMANTIC_JSON_BYTES:
        return sorted(semantics)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return sorted(semantics)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str) and key.lower() in SEMANTIC_KEYS \
                        and isinstance(child, str):
                    normalized = child.strip().lower()
                    if normalized in NEGATIVE_VALUES:
                        semantics.add("negative_outcome")
                    if normalized in FAILURE_VALUES:
                        semantics.add("failure")
                    if normalized in PARTIAL_VALUES:
                        semantics.add("partial")
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return sorted(semantics)


def _stamp(index: dict) -> dict:
    stamped = copy.deepcopy(index)
    stamped.pop("index_sha256", None)
    stamped["index_sha256"] = appendix_contract.canonical_sha256(stamped)
    return stamped


def build_index(
    root: pathlib.Path = DEFAULT_CORPUS,
    *,
    active_owners: list[dict] | None = None,
    source_base_commit: str | None = None,
) -> dict:
    owners = spec_reentry_scaffold.active_heavy_owners() if active_owners is None else active_owners
    if owners:
        raise RuntimeError("refusing corpus hashing while a heavy owner is active")
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        directory = pathlib.Path(dirpath)
        for name in sorted(filenames):
            path = directory / name
            if path.is_symlink():
                raise RuntimeError(f"corpus index refuses symlink: {path}")
            size, sha256 = _stable_hash(path)
            semantics = _explicit_semantics(path, size)
            if _stable_hash(path) != (size, sha256):
                raise RuntimeError(f"corpus file changed during semantic census: {path}")
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "kind": _kind(path),
                "size": size,
                "sha256": sha256,
                "semantics": semantics,
            })
    entries.sort(key=lambda item: item["path"])
    kind_counts: dict[str, int] = {}
    for entry in entries:
        kind_counts[entry["kind"]] = kind_counts.get(entry["kind"], 0) + 1
    semantic_counts = {
        semantic: sum(semantic in entry["semantics"] for entry in entries)
        for semantic in SEMANTIC_CLASSES
    }
    return _stamp({
        "schema": SCHEMA,
        "source_base_commit": source_base_commit or _source_base_commit(),
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "root": str(root.resolve()),
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "kind_counts": dict(sorted(kind_counts.items())),
        "semantic_counts": semantic_counts,
        "contains_explicit_negative_failure_or_partial_evidence": any(semantic_counts.values()),
        "semantic_census_policy": "explicit-filename-or-structured-status-v1",
        "entries": entries,
    })


def verify_index(index: Any, *, active_owners: list[dict] | None = None) -> list[str]:
    if not isinstance(index, dict):
        return ["corpus index must be an object"]
    errors: list[str] = []
    if index.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    base = index.get("source_base_commit")
    if (
        not isinstance(base, str) or not 7 <= len(base) <= 64
        or any(character not in "0123456789abcdef" for character in base)
    ):
        errors.append("source_base_commit is invalid")
    if index.get("source_base_commit_role") != "repository-base-only-not-byte-authority":
        errors.append("source base commit role overclaims byte authority")
    unstamped = copy.deepcopy(index)
    claimed = unstamped.pop("index_sha256", None)
    if claimed != appendix_contract.canonical_sha256(unstamped):
        errors.append("index_sha256 mismatch")
    owners = spec_reentry_scaffold.active_heavy_owners() if active_owners is None else active_owners
    if owners:
        errors.append("refusing corpus verification while a heavy owner is active")
        return errors
    root_value = index.get("root")
    root = pathlib.Path(root_value) if isinstance(root_value, str) else None
    entries = index.get("entries")
    if root is None or not root.is_dir():
        errors.append("indexed root is missing")
        return errors
    if not isinstance(entries, list):
        errors.append("entries must be a list")
        return errors
    seen: set[str] = set()
    total = 0
    verified_kind_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append("malformed corpus entry")
            continue
        relative = entry["path"]
        relative_path = pathlib.PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in {"", "."}:
            errors.append(f"unsafe corpus path: {relative}")
            continue
        if relative in seen:
            errors.append(f"duplicate corpus path: {relative}")
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"corpus file missing or non-regular: {relative}")
            continue
        try:
            size, sha256 = _stable_hash(path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if size != entry.get("size") or sha256 != entry.get("sha256"):
            errors.append(f"corpus file fingerprint mismatch: {relative}")
        semantics = _explicit_semantics(path, size)
        if entry.get("semantics") != semantics:
            errors.append(f"corpus explicit semantic census mismatch: {relative}")
        expected_kind = _kind(path)
        if entry.get("kind") != expected_kind:
            errors.append(f"corpus kind mismatch: {relative}")
        verified_kind_counts[expected_kind] = verified_kind_counts.get(expected_kind, 0) + 1
        total += size
    current: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        directory = pathlib.Path(dirpath)
        for name in sorted(filenames):
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                errors.append(f"corpus verification refuses symlink: {relative}")
            else:
                current.add(relative)
    for relative in sorted(current - seen):
        errors.append(f"unindexed corpus file: {relative}")
    for relative in sorted(seen - current):
        if not any(relative in error for error in errors):
            errors.append(f"indexed corpus file is absent: {relative}")
    if index.get("file_count") != len(entries):
        errors.append("file_count does not match entries")
    if index.get("total_bytes") != total:
        errors.append("total_bytes does not match entries")
    if index.get("kind_counts") != dict(sorted(verified_kind_counts.items())):
        errors.append("kind_counts do not match entries")
    semantic_counts = {
        semantic: sum(
            isinstance(entry, dict) and semantic in entry.get("semantics", [])
            for entry in entries
        )
        for semantic in SEMANTIC_CLASSES
    }
    if index.get("semantic_counts") != semantic_counts:
        errors.append("semantic_counts do not match entries")
    if index.get("contains_explicit_negative_failure_or_partial_evidence") \
            is not any(semantic_counts.values()):
        errors.append("explicit negative/failure/partial summary is not truthful")
    if index.get("semantic_census_policy") != "explicit-filename-or-structured-status-v1":
        errors.append("semantic census policy is invalid")
    return errors


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "cell").mkdir()
        (root / "cell" / "request.json").write_text("{}\n", encoding="utf-8")
        (root / "cell" / "failed.partial").write_bytes(b"negative evidence")
        assert preview(root)["file_count"] == 2
        index = build_index(root, active_owners=[], source_base_commit="0123456789abcdef")
        assert verify_index(index, active_owners=[]) == []
        assert index["kind_counts"]["partial_artifact"] == 1
        (root / "cell" / "late.json").write_text("{}\n", encoding="utf-8")
        assert any(
            "unindexed corpus file" in error
            for error in verify_index(index, active_owners=[])
        )
        try:
            build_index(root, active_owners=[{"pid": 1}], source_base_commit="0123456789abcdef")
        except RuntimeError:
            pass
        else:
            raise AssertionError("active owner must block corpus hashing")
    print("appendix_corpus.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_CORPUS)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--build", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.preview:
        print(json.dumps(preview(args.root), indent=2, sort_keys=True))
        return 0
    if args.build is not None:
        try:
            index = build_index(args.root)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 75
        _atomic_json(args.build, index)
        return 0
    if args.verify is not None:
        errors = verify_index(_load(args.verify))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    parser.error("choose --preview, --build, --verify, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
