#!/usr/bin/env python3.12
"""Frozen-base loader used by pending-only Doctor V5 accelerators.

The live campaign's completed requests bind the original adapter and worker
sources.  Acceleration wrappers must therefore *not* edit those files.  This
module loads an exact, hash-pinned base implementation and lets a wrapper
replace only explicit execution bindings (worker path, quantizer path, and
opt-in arguments).  Runtime specs also include this loader and every frozen
base source as ordinary hash-bound inputs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class AccelerationBindingError(RuntimeError):
    pass


def hash_file(path: Path) -> tuple[str, int]:
    path = path.resolve(strict=True)
    path.relative_to(ROOT.resolve())
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AccelerationBindingError(f"bound source is not regular: {path}")
        digest, size = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if identity(before) != identity(after) or size != after.st_size:
            raise AccelerationBindingError(f"bound source changed while hashing: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def load_frozen(name: str, path: Path, expected_sha256: str) -> ModuleType:
    observed, _ = hash_file(path)
    if observed != expected_sha256:
        raise AccelerationBindingError(
            f"frozen base source drifted: {path} expected={expected_sha256} observed={observed}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AccelerationBindingError(f"cannot load frozen base source: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses and a few stdlib introspection helpers resolve annotations
    # through sys.modules while the module body is executing.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def input_row(role: str, path: Path) -> dict[str, Any]:
    digest, size = hash_file(path)
    return {"role": role, "path": str(path.resolve()), "sha256": digest, "bytes": size}


def bind_extra_inputs(document: dict[str, Any], rows: Iterable[dict[str, Any]]) \
        -> dict[str, Any]:
    inputs = document.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise AccelerationBindingError("accelerated runtime spec has no base inputs")
    result = [dict(row) for row in inputs]
    roles = {row.get("role") for row in result if isinstance(row, dict)}
    paths = {row.get("path") for row in result if isinstance(row, dict)}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
            raise AccelerationBindingError("accelerator input row is invalid")
        if row["role"] in roles or row["path"] in paths:
            raise AccelerationBindingError("accelerator input role/path is duplicated")
        roles.add(row["role"])
        paths.add(row["path"])
        result.append(dict(row))
    document["inputs"] = sorted(result, key=lambda row: row["role"])
    return document


def export_module(module: ModuleType, namespace: dict[str, Any], *, keep: set[str]) -> None:
    """Expose a configured base module through a thin source-bound wrapper."""
    for name, value in vars(module).items():
        if name.startswith("__") or name in keep:
            continue
        namespace[name] = value
