from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

MAX_MUTATION_OPERATIONS = 20


class MutationError(Exception):
    pass


def _snapshot_file(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return {"path": path, "content": p.read_text(encoding="utf-8")}


def _restore_file(
    snapshot: Optional[Dict[str, Any]],
    path: Optional[str] = None,
) -> None:
    if snapshot is None:
        target = path
        if not target:
            return
        p = Path(target)
        if p.is_file():
            p.unlink()
        return
    p = Path(snapshot["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(snapshot["content"], encoding="utf-8")


def _apply_create(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _apply_replace(content: str, old: str, new: str) -> str:
    if old not in content:
        raise MutationError("old_text not found")
    if content.count(old) > 1:
        raise MutationError("old_text not unique")
    if old == new:
        raise MutationError("NO_OP_MUTATION: replace old_text equals new_text")
    return content.replace(old, new, 1)


def _apply_insert(content: str, anchor: str, text: str, mode: str) -> str:
    if anchor not in content:
        raise MutationError("anchor not found")
    if content.count(anchor) > 1:
        raise MutationError("anchor not unique")
    if not text:
        raise MutationError("NO_OP_MUTATION: insert with empty text")
    if mode == "insert_before":
        return content.replace(anchor, text + anchor, 1)
    return content.replace(anchor, anchor + text, 1)


def apply_mutation_operations(guard: Any, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not operations:
        raise MutationError("NO_OP_MUTATION")
    if len(operations) > MAX_MUTATION_OPERATIONS:
        raise MutationError(f"too many operations: {len(operations)}")
    snapshots: Dict[str, Optional[Dict[str, Any]]] = {}
    changed: List[str] = []
    created: List[str] = []
    for op in operations:
        op_type = op.get("op")
        path = op.get("path")
        if not path:
            raise MutationError("missing path")
        full = guard.resolve(path)
        if op_type == "create":
            if full in snapshots:
                raise MutationError("duplicate create")
            snapshots[full] = _snapshot_file(full)
            _apply_create(full, op.get("content", ""))
            created.append(path)
        elif op_type in ("replace", "insert_before", "insert_after"):
            if full not in snapshots:
                snapshots[full] = _snapshot_file(full)
            p = Path(full)
            if not p.exists():
                raise MutationError("file not found")
            content = p.read_text(encoding="utf-8")
            if op_type == "replace":
                new_content = _apply_replace(content, op.get("old_text", ""), op.get("new_text", ""))
            else:
                new_content = _apply_insert(content, op.get("anchor", ""), op.get("text", ""), op_type)
            p.write_text(new_content, encoding="utf-8")
            if path not in changed:
                changed.append(path)
        else:
            raise MutationError(f"unknown op: {op_type}")
    files = []
    any_change = False
    for full, snap in snapshots.items():
        before = None if snap is None else snap["content"].encode("utf-8")
        p = Path(full)
        after = p.read_bytes() if p.exists() and p.is_file() else None
        files.append(
            {
                "path": full,
                "sha256_before": hashlib.sha256(before).hexdigest() if before is not None else None,
                "sha256_after": hashlib.sha256(after).hexdigest() if after is not None else None,
            }
        )
        if before != after:
            any_change = True
    if snapshots and not any_change:
        raise MutationError("NO_OP_MUTATION")
    test_paths = discover_tests(changed + created)
    result = {
        "operation_count": len(operations),
        "paths": changed + created,
        "changed": changed,
        "created": created,
        "snapshots": snapshots,
        "files": files,
        "rewrites_tests": bool(test_paths),
        "test_paths": test_paths,
    }
    result["content_hash"] = mutation_content_hash(result)
    return result


def mutation_content_hash(result: Dict[str, Any]) -> str:
    """Stable hash of before/after bytes for every file the mutation touched.

    Scheduler.complete can use this as the fingerprint so a no-op is not
    recorded as progress.
    """
    payload = [
        {
            "path": item.get("path"),
            "sha256_before": item.get("sha256_before"),
            "sha256_after": item.get("sha256_after"),
        }
        for item in (result.get("files") or [])
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_fingerprint(paths: List[str], root: Optional[str] = None) -> str:
    """Hash of current file contents. Identical trees share a fingerprint."""
    digest = hashlib.sha256()
    for raw in sorted(paths):
        digest.update(raw.encode("utf-8"))
        digest.update(b"\0")
        path = Path(raw)
        if root is not None and not path.is_absolute():
            path = Path(root) / raw
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"-")
        digest.update(b"\0")
    return digest.hexdigest()


def rollback_mutation(mutation_result: Dict[str, Any]) -> None:
    for full_path, snap in mutation_result.get("snapshots", {}).items():
        _restore_file(snap, full_path)


def compile_python_file(path: Union[str, Path]) -> Dict[str, Any]:
    """In-process syntax check. Same parser as ``python -m py_compile``.

    Does not write ``.pyc`` and does not spawn an interpreter. Isolation is
    not required: the source is already on disk in this process's workspace.
    """
    target = Path(path)
    try:
        source = target.read_bytes()
    except OSError as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{exc}\n",
        }
    try:
        compile(source, str(target), "exec")
    except SyntaxError as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "".join(traceback.format_exception_only(type(exc), exc)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}\n",
        }
    return {"ok": True, "exit_code": 0, "stdout": "", "stderr": ""}


def validate_python_syntax(path: str) -> bool:
    return bool(compile_python_file(path)["ok"])


def discover_tests(paths: List[str]) -> List[str]:
    patterns = [r"^test_.*\.py$", r"^.*_test\.py$", r"^.*\.test\.js$", r"^.*\.spec\.js$", r"^.*\.test\.ts$", r"^.*\.spec\.ts$"]
    return [p for p in paths if any(re.match(pat, os.path.basename(p)) for pat in patterns)]