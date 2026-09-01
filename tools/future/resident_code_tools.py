#!/usr/bin/env python3
"""Resident code tools — wrap what is already here; do not choose the science.

The HCLI resident is a live sovereign scientist. It owns hypothesis selection
on the SUB2_EBPW frontier. This module is machinery: a bounded, resident-callable
API over capabilities the repo already has. It does not pick a representation
or a component to test next, and it is not a new agent framework.

    SEARCH_CODE         hcli.tool_registry fs.search (+ git grep for sparse trees)
    READ_CODE           hcli.tool_registry fs.read  (+ git show for sparse-absent)
    CREATE_WORKTREE     git worktree add under <repo>/.worktrees/<name>
    PATCH               hcli.mutation.apply_mutation_operations, worktree-bound
    BUILD               cargo build --target-dir workspace/ops/build/rust
                        or hcli.mutation.compile_python_file
    TEST                hcli.tool_registry tests.run
    RUN                 argv in the worktree (no shell); paths may not escape
    DIFF                hcli.tool_registry git.diff
    LAND_THROUGH_GATE   tools.future.integration_gate; never lands red
    ROLLBACK            hcli.mutation.rollback_mutation; git worktree remove
                        only when the worktree is clean

Every operation returns a uniform result dict. Ordinary failure (a red build,
a failing test, a miss) is a result, not an exception. Missing required inputs
RAISE. PATCH and RUN RAISE when a path leaves the worktree. Nothing deletes
uncommitted work: a dirty worktree is not removed.

    python3 tools/future/resident_code_tools.py --build
    python3 -m pytest tools/future/test_resident_code_tools.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from hcli.mutation import (
    MutationError,
    apply_mutation_operations,
    compile_python_file,
    rollback_mutation,
)
from hcli.tool_registry import default_tool_registry

from tools.future import integration_gate
from tools.future import sandbox as _sandbox
from tools.future._common import GIT_TIMEOUT_S, REPO, git, write_receipt

RECORDED_BY = "tools/future/resident_code_tools.py"
RECEIPT = "RESIDENT_CODE_TOOLS.json"
SCHEMA = "hawking.future.resident_code_tools.v1"
VERSION = 1

WORKTREES_DIRNAME = ".worktrees"
CARGO_TARGET_DIR = "workspace/ops/build/rust"
OUTPUT_CAP = 8000
RUN_TIMEOUT_S = 60
BUILD_TIMEOUT_S = 300
TEST_TIMEOUT_S = 120

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHELL_META = re.compile(r"[;&|<>`$()]|\n|\r")

OPERATIONS: tuple[str, ...] = (
    "SEARCH_CODE",
    "READ_CODE",
    "CREATE_WORKTREE",
    "PATCH",
    "BUILD",
    "TEST",
    "RUN",
    "DIFF",
    "LAND_THROUGH_GATE",
    "ROLLBACK",
)

# Recovered wrappers. Names are the existing callables, not a new stack.
WRAPS: dict[str, tuple[str, ...]] = {
    "SEARCH_CODE": (
        "hcli.tool_registry.default_tool_registry / fs.search",
        "git grep --no-optional-locks (sparse-absent files)",
    ),
    "READ_CODE": (
        "hcli.tool_registry.default_tool_registry / fs.read",
        "git show HEAD:<path> (sparse-absent files)",
    ),
    "CREATE_WORKTREE": (
        "git worktree add --detach under <repo>/.worktrees/<name>",
        "tools.future.sandbox WORKTREES_DIRNAME placement rule",
    ),
    "PATCH": ("hcli.mutation.apply_mutation_operations",),
    "BUILD": (
        "cargo build --target-dir workspace/ops/build/rust",
        "hcli.mutation.compile_python_file",
    ),
    "TEST": ("hcli.tool_registry.default_tool_registry / tests.run",),
    "RUN": ("subprocess argv, cwd=worktree; no shell (hcli.tool_registry._run_readonly shape)",),
    "DIFF": ("hcli.tool_registry.default_tool_registry / git.diff",),
    "LAND_THROUGH_GATE": ("tools.future.integration_gate.check / land",),
    "ROLLBACK": (
        "hcli.mutation.rollback_mutation",
        "git worktree remove (refused when porcelain is dirty)",
    ),
}

# Sessions keyed by realpath(worktree). PATCH stores the mutation so ROLLBACK
# can restore without the caller threading snapshots. Not a planner.
_SESSIONS: dict[str, dict[str, Any]] = {}


class CodeToolsRefused(RuntimeError):
    """Missing input, path escape, or a policy the resident may not override."""


def _contained(path: str, root: str) -> bool:
    path_n = os.path.normpath(path)
    root_n = os.path.normpath(root)
    if path_n == root_n:
        return True
    prefix = root_n if root_n.endswith(os.sep) else root_n + os.sep
    return path_n.startswith(prefix)


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(str(path)))


def _cap(text: str | None, limit: int = OUTPUT_CAP) -> str:
    raw = text or ""
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1] + "…"


def _result(op: str, ok: bool, **extra: Any) -> dict[str, Any]:
    error = extra.pop("error", None)
    out: dict[str, Any] = {
        "ok": bool(ok),
        "op": op,
        "status": "OK" if ok else "FAILED",
        "error": None if ok else (str(error) if error is not None else "failed"),
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
    }
    out.update(extra)
    return out


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise CodeToolsRefused(f"{name} is required")
    if isinstance(value, str) and not value.strip():
        raise CodeToolsRefused(f"{name} is required")
    if isinstance(value, (list, tuple)) and not value:
        raise CodeToolsRefused(f"{name} is required")
    return value


def _bound_worktree(worktree: str | Path) -> Path:
    _require(str(worktree) if worktree is not None else None, "worktree")
    wt = _real(worktree)
    if not wt.is_dir():
        raise CodeToolsRefused(f"worktree is not a directory: {wt}")
    marker = wt / ".git"
    if not marker.exists():
        raise CodeToolsRefused(f"worktree is not a git worktree: {wt}")
    return wt


class WorktreeGuard:
    """Resolve paths inside one worktree. Escape raises, it does not default."""

    def __init__(self, worktree: str | Path) -> None:
        self.root = _real(worktree)

    def resolve(self, path: Any) -> str:
        if path is None or str(path).strip() == "":
            raise CodeToolsRefused("path is required")
        text = str(path)
        if "\x00" in text:
            raise CodeToolsRefused("NUL byte in path")
        joined = text if os.path.isabs(text) else os.path.join(str(self.root), text)
        full = os.path.realpath(joined)
        if not _contained(full, str(self.root)):
            raise CodeToolsRefused(f"path escapes worktree: {text}")
        return full


def _registry(root: Path):
    return default_tool_registry(str(root), repo_root=str(root))


def _git_root(path: Path) -> Path | None:
    proc = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    return _real(proc.stdout.strip())


def _porcelain(worktree: Path) -> str:
    proc = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(worktree), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        return f"__STATUS_FAILED__ {(proc.stderr or proc.stdout or '').strip()}"
    return proc.stdout or ""


def _is_clean(worktree: Path) -> tuple[bool, str]:
    text = _porcelain(worktree)
    if text.startswith("__STATUS_FAILED__"):
        return False, text
    if text.strip():
        return False, "uncommitted or untracked changes present"
    return True, "clean: empty porcelain"


def search_code(
    *,
    pattern: str | None = None,
    root: str | None = None,
    worktree: str | None = None,
    glob: str | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    _require(pattern, "pattern")
    if root is None and worktree is None:
        raise CodeToolsRefused("root is required (or worktree)")
    base = _real(worktree if worktree is not None else root)  # type: ignore[arg-type]
    if worktree is not None:
        base = _bound_worktree(worktree)
        if root is not None:
            resolved = WorktreeGuard(base).resolve(root)
            base = Path(resolved)
    elif not base.exists():
        return _result("SEARCH_CODE", False, error=f"root is not on disk: {base}", matches=[])

    limit = max(1, min(1000, int(max_results or 100)))
    args: dict[str, Any] = {"pattern": pattern, "root": str(base), "max_results": limit}
    if glob:
        args["glob"] = glob
    wrapped = _registry(base).invoke("fs.search", args)
    matches: list[dict[str, Any]] = []
    if wrapped.ok and isinstance(wrapped.value, dict):
        matches.extend(list(wrapped.value.get("matches") or []))

    git_root = _git_root(base)
    git_matches = 0
    if git_root is not None and len(matches) < limit:
        try:
            rel = os.path.relpath(str(base), str(git_root))
        except ValueError:
            rel = "."
        grep_argv = [
            "git",
            "--no-optional-locks",
            "-C",
            str(git_root),
            "grep",
            "-n",
            "-I",
            "-F",
            "-e",
            pattern,
            "--",
            rel if rel != "." else ".",
        ]
        try:
            proc = subprocess.run(
                grep_argv,
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return _result(
                "SEARCH_CODE",
                True,
                matches=matches[:limit],
                truncated=len(matches) > limit,
                wrapped="fs.search",
                git_grep_error=str(exc),
            )
        seen = {(m.get("path"), m.get("line")) for m in matches}
        for line in (proc.stdout or "").splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path, lineno, text = parts[0], parts[1], parts[2]
            full = str((_real(git_root) / path) if not os.path.isabs(path) else Path(path))
            key = (full, int(lineno) if lineno.isdigit() else lineno)
            if key in seen:
                continue
            matches.append({"path": full, "line": int(lineno) if lineno.isdigit() else lineno, "text": text[:1000]})
            seen.add(key)
            git_matches += 1
            if len(matches) >= limit:
                break

    return _result(
        "SEARCH_CODE",
        True,
        matches=matches[:limit],
        n_matches=min(len(matches), limit),
        truncated=len(matches) > limit,
        wrapped="hcli.tool_registry fs.search",
        git_grep_added=git_matches,
        root=str(base),
        pattern=pattern,
    )


def read_code(
    *,
    path: str | None = None,
    worktree: str | None = None,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    _require(path, "path")
    if worktree is not None:
        wt = _bound_worktree(worktree)
        full = Path(WorktreeGuard(wt).resolve(path))
        root = wt
    else:
        text = str(path)
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = REPO / candidate
        full = Path(os.path.realpath(str(candidate)))
        if not _contained(str(full), str(_real(REPO))):
            raise CodeToolsRefused(f"path escapes repository: {path}")
        root = _real(REPO)

    if full.is_file():
        wrapped = _registry(root).invoke(
            "fs.read",
            {"path": str(full), "max_bytes": int(max_bytes or 64 * 1024)},
        )
        if wrapped.ok and isinstance(wrapped.value, dict):
            payload = dict(wrapped.value)
            return _result(
                "READ_CODE",
                True,
                path=payload.get("path", str(full)),
                content=payload.get("content"),
                bytes=payload.get("bytes"),
                truncated=payload.get("truncated"),
                sha256=payload.get("sha256"),
                source="disk",
                wrapped="hcli.tool_registry fs.read",
            )
        return _result("READ_CODE", False, error=wrapped.error, path=str(full), source="disk")

    rel = path
    git_root = _git_root(root)
    if git_root is not None:
        try:
            rel = os.path.relpath(str(full), str(git_root)).replace(os.sep, "/")
        except ValueError:
            rel = str(path).replace(os.sep, "/")
        if rel.startswith(".."):
            rel = str(path).lstrip("./")
        shown = git("show", f"HEAD:{rel}") if git_root == _real(REPO) else ""
        if git_root != _real(REPO):
            proc = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(git_root), "show", f"HEAD:{rel}"],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_S,
                check=False,
            )
            shown = proc.stdout if proc.returncode == 0 else ""
        if shown:
            blob = shown.encode()
            limit = int(max_bytes or 64 * 1024)
            clipped = blob[:limit]
            return _result(
                "READ_CODE",
                True,
                path=rel,
                content=clipped.decode("utf-8", errors="replace"),
                bytes=len(blob),
                truncated=len(blob) > limit,
                source="git-show",
                wrapped="git show HEAD:<path>",
            )
    return _result("READ_CODE", False, error=f"not found: {path}", path=str(path), source="missing")


def create_worktree(
    *,
    name: str | None = None,
    repo: str | None = None,
    start_point: str | None = None,
) -> dict[str, Any]:
    _require(name, "name")
    if not _SAFE_NAME.match(str(name)):
        raise CodeToolsRefused(
            f"name {name!r} is not a safe path component (allowed: A-Za-z0-9._-)"
        )
    canonical = _real(repo) if repo is not None else _real(REPO)
    if not (canonical / ".git").exists():
        # Linked worktree: .git may be a file. rev-parse is the authority.
        proc = subprocess.run(
            ["git", "-C", str(canonical), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0 or proc.stdout.strip() != "true":
            raise CodeToolsRefused(f"{canonical} is not a git work tree")

    dest = canonical / WORKTREES_DIRNAME / str(name)
    sibling = canonical.parent / str(name)
    if _real(dest) == _real(sibling) and dest.parent.name != WORKTREES_DIRNAME:
        raise CodeToolsRefused(
            "CREATE_WORKTREE must place worktrees under <repo>/.worktrees/<name>, "
            "never as a sibling of the repo"
        )
    if dest.parent.name != WORKTREES_DIRNAME:
        raise CodeToolsRefused(
            "CREATE_WORKTREE must place worktrees under <repo>/.worktrees/<name>"
        )
    if os.path.realpath(str(dest.parent.parent)) != os.path.realpath(str(canonical)):
        raise CodeToolsRefused(
            "CREATE_WORKTREE must place worktrees under <repo>/.worktrees/<name>, "
            "never as a sibling of the repo"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        marker = dest / ".git"
        if marker.exists():
            return _result(
                "CREATE_WORKTREE",
                True,
                worktree=str(_real(dest)),
                name=str(name),
                newly_created=False,
                under_dot_worktrees=True,
                not_sibling=_real(dest) != _real(sibling),
                placement=f"{WORKTREES_DIRNAME}/{name}",
            )
        return _result(
            "CREATE_WORKTREE",
            False,
            error=f"path exists and is not a worktree: {dest}",
            worktree=str(dest),
        )

    point = start_point if start_point else "HEAD"
    proc = subprocess.run(
        ["git", "-C", str(canonical), "worktree", "add", "--detach", str(dest), point],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        return _result(
            "CREATE_WORKTREE",
            False,
            error=_cap((proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"),
            worktree=str(dest),
            returncode=proc.returncode,
        )
    real_dest = _real(dest)
    return _result(
        "CREATE_WORKTREE",
        True,
        worktree=str(real_dest),
        name=str(name),
        newly_created=True,
        under_dot_worktrees=real_dest.parent.name == WORKTREES_DIRNAME,
        not_sibling=real_dest != _real(sibling),
        placement=f"{WORKTREES_DIRNAME}/{name}",
        sibling_path=str(_real(sibling)),
        stdout=_cap(proc.stdout),
        wrapped="git worktree add --detach",
    )


def _operations_from_kwargs(
    *,
    operations: list[dict[str, Any]] | None,
    path: str | None,
    old_text: str | None,
    new_text: str | None,
    content: str | None,
) -> list[dict[str, Any]]:
    if operations is not None:
        if not isinstance(operations, list) or not operations:
            raise CodeToolsRefused("operations is required")
        return operations
    _require(path, "path")
    if content is not None and old_text is None:
        return [{"op": "create", "path": path, "content": content}]
    if old_text is not None:
        if new_text is None:
            raise CodeToolsRefused("new_text is required")
        return [{"op": "replace", "path": path, "old_text": old_text, "new_text": new_text}]
    raise CodeToolsRefused("operations is required (or path+old_text+new_text / path+content)")


def patch(
    *,
    worktree: str | None = None,
    operations: list[dict[str, Any]] | None = None,
    path: str | None = None,
    old_text: str | None = None,
    new_text: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    wt = _bound_worktree(_require(worktree, "worktree"))
    ops = _operations_from_kwargs(
        operations=operations,
        path=path,
        old_text=old_text,
        new_text=new_text,
        content=content,
    )
    guard = WorktreeGuard(wt)
    # Resolve every path first so a later op cannot sneak outside after a write.
    for op in ops:
        guard.resolve(op.get("path"))
    try:
        result = apply_mutation_operations(guard, ops)
    except MutationError as exc:
        return _result("PATCH", False, error=str(exc), worktree=str(wt), wrapped="hcli.mutation")
    _SESSIONS[_real(wt).as_posix()] = {"mutation": result, "worktree": str(wt)}
    return _result(
        "PATCH",
        True,
        worktree=str(wt),
        paths=result.get("paths"),
        changed=result.get("changed"),
        created=result.get("created"),
        content_hash=result.get("content_hash"),
        wrapped="hcli.mutation.apply_mutation_operations",
    )


def build_op(
    *,
    worktree: str | None = None,
    paths: list[str] | None = None,
    cargo: bool | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    wt = _bound_worktree(_require(worktree, "worktree"))
    guard = WorktreeGuard(wt)
    want_cargo = bool(cargo) or ((wt / "Cargo.toml").is_file() and not paths)
    if not want_cargo and not paths:
        raise CodeToolsRefused("paths is required (or cargo=True)")
    timeout = min(BUILD_TIMEOUT_S, max(1.0, float(timeout_s or BUILD_TIMEOUT_S)))
    if want_cargo:
        target = wt / CARGO_TARGET_DIR
        argv = ["cargo", "build", "--target-dir", str(target)]
        try:
            proc = subprocess.run(
                argv,
                cwd=str(wt),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return _result("BUILD", False, error=str(exc), worktree=str(wt), argv=argv)
        return _result(
            "BUILD",
            proc.returncode == 0,
            error=None if proc.returncode == 0 else _cap(proc.stderr or proc.stdout),
            worktree=str(wt),
            returncode=proc.returncode,
            stdout=_cap(proc.stdout),
            stderr=_cap(proc.stderr),
            argv=argv,
            wrapped="cargo build --target-dir workspace/ops/build/rust",
        )

    rows = []
    ok = True
    for raw in paths or []:
        full = guard.resolve(raw)
        row = compile_python_file(full)
        row = dict(row)
        row["path"] = raw
        rows.append(row)
        if not row.get("ok"):
            ok = False
    return _result(
        "BUILD",
        ok,
        error=None if ok else "compile failed",
        worktree=str(wt),
        files=rows,
        wrapped="hcli.mutation.compile_python_file",
    )


def test_op(
    *,
    worktree: str | None = None,
    paths: list[str] | None = None,
    runner: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    wt = _bound_worktree(_require(worktree, "worktree"))
    _require(paths, "paths")
    guard = WorktreeGuard(wt)
    rels = []
    for raw in paths or []:
        full = Path(guard.resolve(raw))
        rels.append(os.path.relpath(str(full), str(wt)))
    which = (runner or "pytest").lower()
    if which != "pytest":
        # cargo/unittest go through the existing registry handler unchanged.
        args: dict[str, Any] = {
            "runner": which,
            "root": str(wt),
            "paths": rels,
            "timeout_s": float(timeout_s or TEST_TIMEOUT_S),
        }
        wrapped = _registry(wt).invoke("tests.run", args)
        value = wrapped.value if isinstance(wrapped.value, dict) else {}
        rc = value.get("returncode")
        verified = bool(value.get("verified")) if wrapped.ok else False
        ok = bool(wrapped.ok) and verified and rc == 0
        return _result(
            "TEST",
            ok,
            error=None if ok else (wrapped.error or value.get("stderr") or "tests failed"),
            worktree=str(wt),
            returncode=rc,
            stdout=_cap(value.get("stdout")),
            stderr=_cap(value.get("stderr")),
            verified=verified,
            wrapped="hcli.tool_registry tests.run",
            runner=which,
            paths=rels,
        )
    # Same argv as hcli.tool_registry._tests_run, plus the cache-disable flags
    # integration_gate already uses, so a TEST does not leave untracked bytecode
    # that would then look like uncommitted work at ROLLBACK.
    argv = [_sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *rels]
    timeout = min(900.0, max(0.1, float(timeout_s or TEST_TIMEOUT_S)))
    try:
        proc = subprocess.run(
            argv,
            cwd=str(wt),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return _result("TEST", False, error=str(exc), worktree=str(wt), argv=argv)
    ok = proc.returncode == 0
    return _result(
        "TEST",
        ok,
        error=None if ok else _cap(proc.stderr or proc.stdout or "tests failed"),
        worktree=str(wt),
        returncode=proc.returncode,
        stdout=_cap(proc.stdout),
        stderr=_cap(proc.stderr),
        verified=ok,
        wrapped="hcli.tool_registry tests.run argv (pytest -q)",
        runner="pytest",
        paths=rels,
        argv=argv,
    )


def run_op(
    *,
    worktree: str | None = None,
    argv: list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    wt = _bound_worktree(_require(worktree, "worktree"))
    _require(argv, "argv")
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise CodeToolsRefused("argv must be a non-empty string array")
    if any(_SHELL_META.search(item) for item in argv):
        raise CodeToolsRefused("shell metacharacters are not allowed")
    if "-c" in argv or argv[-1:] == ["-"]:
        raise CodeToolsRefused("inline command stdin/-c is not allowed through RUN")
    guard = WorktreeGuard(wt)
    run_cwd = Path(guard.resolve(cwd)) if cwd else wt
    if not run_cwd.is_dir():
        raise CodeToolsRefused(f"cwd is not a directory: {run_cwd}")
    for token in argv[1:]:
        if token.startswith("-"):
            continue
        if os.path.isabs(token):
            guard.resolve(token)
            continue
        joined = os.path.realpath(os.path.join(str(run_cwd), token))
        if not _contained(joined, str(wt)):
            raise CodeToolsRefused(f"path escapes worktree: {token}")
    timeout = min(600.0, max(0.1, float(timeout_s or RUN_TIMEOUT_S)))
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(run_cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return _result("RUN", False, error=str(exc), worktree=str(wt), argv=list(argv))
    return _result(
        "RUN",
        proc.returncode == 0,
        error=None if proc.returncode == 0 else _cap(proc.stderr or f"exit {proc.returncode}"),
        worktree=str(wt),
        returncode=proc.returncode,
        stdout=_cap(proc.stdout),
        stderr=_cap(proc.stderr),
        argv=list(argv),
        cwd=str(run_cwd),
        wrapped="subprocess argv (no shell)",
    )


def diff_op(
    *,
    worktree: str | None = None,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    wt = _bound_worktree(_require(worktree, "worktree"))
    guard = WorktreeGuard(wt)
    rels: list[str] = []
    for raw in paths or []:
        full = Path(guard.resolve(raw))
        rels.append(os.path.relpath(str(full), str(wt)))
    args: dict[str, Any] = {"path": str(wt)}
    if rels:
        args["paths"] = [str(wt / r) for r in rels]
    wrapped = _registry(wt).invoke("git.diff", args)
    value = wrapped.value if isinstance(wrapped.value, dict) else {}
    stdout = value.get("stdout") or ""
    ok = bool(wrapped.ok) and value.get("returncode", 1) == 0
    return _result(
        "DIFF",
        ok,
        error=None if ok else wrapped.error,
        worktree=str(wt),
        stdout=_cap(stdout),
        stderr=_cap(value.get("stderr")),
        returncode=value.get("returncode"),
        nonempty=bool(str(stdout).strip()),
        wrapped="hcli.tool_registry git.diff",
        paths=rels,
    )


def land_through_gate(
    *,
    paths: list[str] | None = None,
    message_file: str | None = None,
    known_red: str | None = None,
) -> dict[str, Any]:
    _require(paths, "paths")
    _require(message_file, "message_file")
    if known_red:
        raise CodeToolsRefused(
            "LAND_THROUGH_GATE must never land red; known_red is refused"
        )
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise CodeToolsRefused("paths must be a non-empty string array")
    checked = integration_gate.check(list(paths))
    if not checked.get("green"):
        return _result(
            "LAND_THROUGH_GATE",
            False,
            error="gate is RED; not landed",
            committed=False,
            green=False,
            verdict=checked.get("verdict") or "RED",
            gate=checked,
            wrapped="tools.future.integration_gate.check",
        )
    try:
        landed = integration_gate.land(list(paths), str(message_file), None)
    except integration_gate.GateRed as exc:
        return _result(
            "LAND_THROUGH_GATE",
            False,
            error=str(exc),
            committed=False,
            green=False,
            wrapped="tools.future.integration_gate.land",
        )
    committed = bool(landed.get("committed")) and bool(landed.get("green"))
    if landed.get("known_red"):
        # The wrapped land() can take an escape hatch. We never pass it, and if
        # a result still claims it, that is a failed land, not a quiet red.
        return _result(
            "LAND_THROUGH_GATE",
            False,
            error="refused to accept a known-red land",
            committed=False,
            green=False,
            gate=landed,
        )
    return _result(
        "LAND_THROUGH_GATE",
        committed,
        error=None if committed else (landed.get("git_stderr") or "commit failed"),
        committed=committed,
        green=bool(landed.get("green")),
        gate=landed,
        wrapped="tools.future.integration_gate.land",
    )


def rollback(
    *,
    worktree: str | None = None,
    remove_worktree: bool = False,
    mutation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    wt = _bound_worktree(_require(worktree, "worktree"))
    key = _real(wt).as_posix()
    stored = mutation if mutation is not None else (_SESSIONS.get(key) or {}).get("mutation")
    if stored is None and not remove_worktree:
        raise CodeToolsRefused("mutation is required (no PATCH session for this worktree)")

    restored = False
    if stored is not None:
        try:
            rollback_mutation(dict(stored))
            restored = True
            _SESSIONS.pop(key, None)
        except (OSError, TypeError, KeyError) as exc:
            return _result(
                "ROLLBACK",
                False,
                error=str(exc),
                worktree=str(wt),
                restored=False,
                wrapped="hcli.mutation.rollback_mutation",
            )

    removed = False
    remove_error = None
    if remove_worktree:
        clean, reason = _is_clean(wt)
        if not clean:
            raise CodeToolsRefused(
                f"refusing to delete uncommitted work: {reason}; porcelain={_porcelain(wt)!r}"
            )
        canonical = _git_root(wt)
        if canonical is None:
            raise CodeToolsRefused("cannot resolve owner repo; worktree not removed")
        proc = subprocess.run(
            ["git", "-C", str(canonical), "worktree", "remove", str(wt)],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            remove_error = _cap((proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}")
            return _result(
                "ROLLBACK",
                False,
                error=remove_error,
                worktree=str(wt),
                restored=restored,
                removed=False,
                wrapped="git worktree remove",
            )
        removed = True

    return _result(
        "ROLLBACK",
        True,
        worktree=str(wt),
        restored=restored,
        removed=removed,
        wrapped="hcli.mutation.rollback_mutation"
        + (" + git worktree remove" if remove_worktree else ""),
    )


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "SEARCH_CODE": search_code,
    "READ_CODE": read_code,
    "CREATE_WORKTREE": create_worktree,
    "PATCH": patch,
    "BUILD": build_op,
    "TEST": test_op,
    "RUN": run_op,
    "DIFF": diff_op,
    "LAND_THROUGH_GATE": land_through_gate,
    "ROLLBACK": rollback,
}


def invoke(op: str, **kwargs: Any) -> dict[str, Any]:
    """Resident-callable surface. Unknown ops RAISE. Ordinary failure is a dict."""
    _require(op, "op")
    key = str(op).strip().upper()
    fn = _DISPATCH.get(key)
    if fn is None:
        raise CodeToolsRefused(f"unknown operation {op!r}")
    return fn(**kwargs)


def _seed_fixture(root: Path) -> Path:
    (root / "scratch.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_scratch.py").write_text(
        "from scratch import VALUE\n\n"
        "def test_value():\n"
        "    assert VALUE == 1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "scratch.py", "test_scratch.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed scratch"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return root


def _hermetic_proof() -> dict[str, Any]:
    """Real worktree create / patch / test / diff / rollback. Never the live tree."""
    tmp = Path(tempfile.mkdtemp(prefix="rct-hermetic-"))
    name = "rct-proof"
    wt_path: Path | None = None
    try:
        canonical = _sandbox.init_fixture_repo(tmp / "canonical")
        _seed_fixture(canonical)
        created = create_worktree(name=name, repo=str(canonical))
        if not created.get("ok"):
            raise CodeToolsRefused(f"hermetic CREATE_WORKTREE failed: {created.get('error')}")
        wt_path = Path(str(created["worktree"]))
        sibling = Path(str(created["sibling_path"]))
        placement_ok = (
            bool(created.get("under_dot_worktrees"))
            and bool(created.get("not_sibling"))
            and wt_path.parent.name == WORKTREES_DIRNAME
            and not sibling.exists()
        )
        if not placement_ok:
            raise CodeToolsRefused("hermetic worktree was not placed under .worktrees/")

        outside = tmp / "outside.txt"
        outside.write_text("safe\n", encoding="utf-8")
        patch_refused = False
        try:
            patch(worktree=str(wt_path), path=str(outside), content="pwned\n")
        except CodeToolsRefused:
            patch_refused = True
        if outside.read_text(encoding="utf-8") != "safe\n":
            raise CodeToolsRefused("PATCH mutated a path outside the worktree")

        run_refused = False
        try:
            run_op(worktree=str(wt_path), argv=["cat", str(outside)])
        except CodeToolsRefused:
            run_refused = True

        patched = patch(
            worktree=str(wt_path),
            path="scratch.py",
            old_text="VALUE = 1\n",
            new_text="VALUE = 1  # resident-code-tools-probe\n",
        )
        if not patched.get("ok"):
            raise CodeToolsRefused(f"hermetic PATCH failed: {patched.get('error')}")
        tested = test_op(worktree=str(wt_path), paths=["test_scratch.py"])
        if not tested.get("ok"):
            raise CodeToolsRefused(
                f"hermetic TEST failed: {tested.get('error')} stdout={tested.get('stdout')!r}"
            )
        differed = diff_op(worktree=str(wt_path), paths=["scratch.py"])
        if not differed.get("ok") or not differed.get("nonempty"):
            raise CodeToolsRefused(f"hermetic DIFF was empty: {differed}")
        rolled = rollback(worktree=str(wt_path), remove_worktree=True)
        if not rolled.get("ok") or not rolled.get("removed"):
            raise CodeToolsRefused(f"hermetic ROLLBACK failed: {rolled}")
        wt_path = None
        return {
            "create": {
                "ok": True,
                "under_dot_worktrees": True,
                "not_sibling": True,
                "placement": created.get("placement"),
            },
            "patch_outside_refused": patch_refused,
            "run_outside_refused": run_refused,
            "patch": {"ok": True, "paths": patched.get("paths")},
            "test": {"ok": True, "returncode": tested.get("returncode")},
            "diff": {"ok": True, "nonempty": True},
            "rollback": {"ok": True, "removed": True, "restored": rolled.get("restored")},
            "real_not_mock": True,
        }
    finally:
        if wt_path is not None and wt_path.exists():
            owner = _git_root(wt_path)
            if owner is not None:
                subprocess.run(
                    ["git", "-C", str(owner), "worktree", "remove", str(wt_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
        shutil.rmtree(tmp, ignore_errors=True)


def _prove_land_never_red() -> dict[str, Any]:
    original_check = integration_gate.check
    original_land = integration_gate.land
    land_called = []

    def fake_check(paths: list[str]) -> dict[str, Any]:
        return {
            "green": False,
            "verdict": "RED",
            "tests": {"why": "forced-red"},
            "receipts": {"malformed": []},
        }

    def fake_land(*_a: Any, **_k: Any) -> dict[str, Any]:
        land_called.append(True)
        raise AssertionError("land must not be called when the gate is red")

    integration_gate.check = fake_check  # type: ignore[assignment]
    integration_gate.land = fake_land  # type: ignore[assignment]
    try:
        result = land_through_gate(
            paths=["tools/future/resident_code_tools.py"],
            message_file="/dev/null",
        )
        known_red_refused = False
        try:
            land_through_gate(
                paths=["tools/future/resident_code_tools.py"],
                message_file="/dev/null",
                known_red="please land anyway",
            )
        except CodeToolsRefused:
            known_red_refused = True
    finally:
        integration_gate.check = original_check  # type: ignore[assignment]
        integration_gate.land = original_land  # type: ignore[assignment]
    if result.get("ok") or result.get("committed") or land_called:
        raise CodeToolsRefused("LAND_THROUGH_GATE landed red or called land() on red")
    if not known_red_refused:
        raise CodeToolsRefused("LAND_THROUGH_GATE did not refuse known_red")
    return {
        "ok": False,
        "committed": False,
        "land_called": False,
        "known_red_refused": True,
        "error": result.get("error"),
    }


def recovered_implementation() -> list[dict[str, Any]]:
    rows = (
        ("hcli/tool_registry.py", "fs.search / fs.read / git.diff / tests.run"),
        ("hcli/mutation.py", "apply_mutation_operations, rollback_mutation, compile_python_file"),
        ("tools/future/sandbox.py", ".worktrees/ placement; init_fixture_repo for hermetic proofs"),
        ("tools/future/integration_gate.py", "check + land; this wrapper never passes known_red"),
        ("tools/grok_worktree_reaper.py", "cleanliness: empty porcelain; never force-delete"),
        ("tools/future/_common.py", "write_receipt, GIT_TIMEOUT_S, git --no-optional-locks"),
    )
    out = []
    for rel, what in rows:
        present = (REPO / rel).is_file()
        out.append({"path": rel, "what": what, "present": present})
    return out


def build() -> Path:
    proofs = _hermetic_proof()
    land_proof = _prove_land_never_red()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "purpose": (
            "Bounded resident-callable API over existing search/read/worktree/"
            "patch/build/test/run/diff/gate/rollback. Machinery, not science."
        ),
        "does_not_choose_science": True,
        "not_a_new_agent_framework": True,
        "operations": list(OPERATIONS),
        "n_operations": len(OPERATIONS),
        "wraps": {k: list(v) for k, v in WRAPS.items()},
        "placement": "<repo>/.worktrees/<name>",
        "never_sibling": True,
        "land_never_red": True,
        "never_deletes_uncommitted_work": True,
        "uniform_result": (
            "every operation returns {ok, op, status, error, ...}; a failed "
            "build or test is a result, not an exception"
        ),
        "refusals": [
            "missing required inputs RAISE",
            "PATCH and RUN RAISE on a path outside the worktree",
            "LAND_THROUGH_GATE refuses known_red and does not call land() when check is red",
            "ROLLBACK RAISE rather than git worktree remove a dirty tree",
        ],
        "hermetic_proof": proofs,
        "land_proof": land_proof,
        "recovered_implementation": recovered_implementation(),
        "cargo_target_dir": CARGO_TARGET_DIR,
        "head": git("rev-parse", "HEAD"),
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. The hermetic "
            "proof is a real git worktree of a fixture repo, not of live Hawking "
            "and not a mock of git."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.build or args.selftest:
        print(build())
        return 0
    print(json.dumps({"operations": list(OPERATIONS), "wraps": {k: list(v) for k, v in WRAPS.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
