"""Capability reachability -- what exists but nothing calls.

A definition is not a capability. A typed tool, a class, a module, or a
function is only a live capability if something OTHER than its own
definition and its own test actually reaches it. Last wave found 41 typed
tools with zero call sites; that is the class of bug this module answers
for, mechanically, so the answer stays checkable instead of asserted.

For each capability this derives, from grep/AST evidence only -- nothing
hand-typed as a verdict:

    defined            the symbol/module exists on disk
    registered         it is a ToolSpec in hcli/tool_registry.py's typed
                        tool registry (default_tool_registry)
    resident_visible   a resident can discover it: registered AND built
                        into default_tool_registry unconditionally (the
                        registry AgentOS.tools is handed)
    callable           a real call site exists outside every test file
    tested             some test file exercises it
    call_sites         file:line evidence, so a reader can disagree

Method and its honest limit
----------------------------
Two evidence classes are gathered, over every `git ls-tree -r --name-only HEAD` `*.py` blob:

  * IMPORT sites: an AST pass over every file's Import/ImportFrom nodes,
    resolving relative imports and the `sys.path.insert(dirname(__file__));
    from _common import x` sibling-import idiom this package uses, into a
    module -> [(file, line)] index.
  * Symbol CALL sites: for a specific function/class, files that import it
    by name are then scanned for `name(` call expressions (AST Call nodes,
    so a def line is never mistaken for a call).
  * LITERAL sites: for a typed tool's string key (e.g. "frontier.escalate"),
    a quoted occurrence in a `"tool": "<name>"` / `tool="<name>"` /
    `invoke("<name>"` dispatch-shaped context only -- a bare mention (a
    same-named but unrelated tool set in a different bridge, a docstring)
    does not count. A subprocess-style capability's own path (e.g.
    "tools/odyssey_ctl.py") is checked the same way, restricted to a real
    subprocess.run/Popen/check_call/check_output/call argument. Both are
    string-based methods and imperfect by construction, so every hit is
    still listed for a reader to check.

ALL test files (any path with a `test_` filename or a `tests/` directory
component) are one bucket: `tested`. None of them count toward `callable`,
by design, uniformly -- not only a capability's "own" test. The wave's own
worked example of the trap was a module that "looked used" while only its
own test imported it; the rule generalises regardless of which module
triggers it: a second test file proves a second test exists, not that the
resident calls the capability during real operation. (As of this HEAD,
tools/future/modellake_events.py itself is NOT that example -- it is
genuinely imported by the live tools/odyssey/modellake_watch.py watcher;
see its own capability entry below for the call site. The pattern the wave
names is real; which specific module currently exhibits it is a fact this
script re-derives every run, not one carried over from the brief.)

Hard limit this method cannot see past: a WorkUnit's `tool` field can be a
string the MODEL proposes at runtime (see
hcli/agentos/resident.py:_child_workunit); no static grep sees that JSON.
A minority of the typed tools ARE reached by a real, static path instead --
hcli/agentos/autonomy_gate.py and its sibling gates build a literal
`"tool": "<name>"` source catalog for census missions, and a few callers
invoke a fixed tool name directly (hcli/agentos/recovery.py); both count as
genuine call sites here (see counts.typed_tools_callable for the current
number). For the remainder, zero static call sites means "no
deterministic/scripted path guarantees this runs" -- not proof a model
could never ask for it. That is exactly the honest question this wave asks:
is there a real, checkable path, or only a possibility.

    python3 tools/future/capability_reachability.py --build
    python3 tools/future/capability_reachability.py --selftest
    python3 -m pytest tools/future/test_capability_reachability.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import GIT_TIMEOUT_S, REPO, git, load_json, require_known_flags, write_receipt

import argparse
import ast
import contextvars
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

RECEIPT = "HCLI_CAPABILITY_REACHABILITY.json"
SCHEMA = "hawking.future.capability_reachability.v1"
VERSION = 1
RECORDED_BY = "tools/future/capability_reachability.py"

TOOL_REGISTRY_REL = "hcli/tool_registry.py"
FUTURE_DIR_REL = "tools/future"

FIELDS = ("defined", "registered", "resident_visible", "callable", "tested")

RUST_FACTS_SCHEMA = "hawking.index.reachability_facts.v1"
_TRACKED_PY: tuple[str, frozenset[str]] | None = None
_GIT_CHECKOUT: tuple[str, bool] | None = None

# Source of truth for file bytes and the file list. HEAD blobs match the Rust
# indexer. `worktree` exists for callers that opt in explicitly; it is never
# the default. A tmpdir that is not this repo's git toplevel always reads the
# filesystem, because there is no HEAD to consult.
SOURCE_HEAD = "head"
SOURCE_WORKTREE = "worktree"
DEFAULT_SOURCE = SOURCE_HEAD
_source_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "capability_reachability_source", default=DEFAULT_SOURCE
)


def current_source() -> str:
    raw = (_source_var.get() or DEFAULT_SOURCE).strip().lower()
    if raw in {"worktree", "working-tree", "working_tree", "disk", "wt"}:
        return SOURCE_WORKTREE
    return SOURCE_HEAD


@contextmanager
def using_source(source: str) -> Iterator[None]:
    """Pin the analyzer to HEAD blobs or working-tree bytes.

    Named and explicit: passing ``source="worktree"`` is the only way to see
    uncommitted edits. Default is HEAD.
    """
    token = _source_var.set(source)
    global _TRACKED_PY
    _TRACKED_PY = None
    try:
        yield
    finally:
        _source_var.reset(token)
        _TRACKED_PY = None


# Optional external reader (a SourceView overlay, a test double). Scoped via
# using_reader(); the `read_text` function object is never replaced. Assigning
# `module.read_text = ...` is the leak that blinded later callers (HCLI calls
# both this analyzer and tools.roadmap in one process).
_reader_var: contextvars.ContextVar[Callable[[Path], str] | None] = contextvars.ContextVar(
    "capability_reachability_reader", default=None
)


@contextmanager
def using_reader(reader: Callable[[Path], str]) -> Iterator[None]:
    """Pin file reads to `reader` for the duration of the block.

    Does not write through `_TEXT_CACHE`, so overlay bytes cannot linger after
    the block and a later assemble() still sees HEAD/worktree.
    """
    token = _reader_var.set(reader)
    try:
        yield
    finally:
        _reader_var.reset(token)


def repo_is_git_checkout() -> bool:
    """True only when `REPO` itself is the git work tree, not a tmpdir inside one."""
    global _GIT_CHECKOUT
    key = str(REPO)
    if _GIT_CHECKOUT is not None and _GIT_CHECKOUT[0] == key:
        return _GIT_CHECKOUT[1]
    out = git("rev-parse", "--show-toplevel")
    ok = False
    if out:
        try:
            ok = Path(out).resolve() == REPO.resolve()
        except OSError:
            ok = False
    _GIT_CHECKOUT = (key, ok)
    return ok


def _list_py_rels(source: str) -> frozenset[str]:
    if source == SOURCE_HEAD and repo_is_git_checkout():
        out = git("ls-tree", "-r", "--name-only", "HEAD")
        return frozenset(
            line
            for line in out.splitlines()
            if line.endswith(".py") and line and "__pycache__" not in line
        )
    out = git("ls-files", "*.py")
    return frozenset(
        line for line in out.splitlines() if line and "__pycache__" not in line
    )


# --------------------------------------------------------------------------
# Repo-wide file index (built once, cached)
# --------------------------------------------------------------------------


def repo_py_files(*, source: str | None = None) -> list[Path]:
    """Every HEAD `*.py` blob (default), or the index if `source='worktree'`."""
    if source is not None:
        with using_source(source):
            return [REPO / r for r in sorted(tracked_py_set())]
    return [REPO / r for r in sorted(tracked_py_set())]


def tracked_py_set() -> frozenset[str]:
    """Repo-relative `*.py` paths for the current source. Cached per (REPO, source)."""
    global _TRACKED_PY
    key = f"{REPO}\0{current_source()}"
    if _TRACKED_PY is None or _TRACKED_PY[0] != key:
        _TRACKED_PY = (key, _list_py_rels(current_source()))
    return _TRACKED_PY[1]


def source_exists(path: Path, *, source: str | None = None) -> bool:
    """HEAD blob exists (default). Untracked on-disk files do not count.

    `source='worktree'` also accepts a real file on disk (the explicit
    working-tree view). A non-git REPO (test tmpdir) always accepts disk.
    """
    src = source if source is not None else current_source()
    if src == SOURCE_WORKTREE or not repo_is_git_checkout():
        if path.is_file():
            return True
    return rel(path) in tracked_py_set()


_TEXT_CACHE: dict[Path, str] = {}


def _prefetch_git_blobs(paths: Sequence[Path]) -> None:
    """Load sparse-missing files from HEAD. One `git cat-file --batch`, not N shows."""
    if not paths:
        return
    specs = [f"HEAD:{rel(p)}" for p in paths]
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "cat-file", "--batch"],
            cwd=str(REPO),
            input=("\n".join(specs) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        for p in paths:
            _TEXT_CACHE.setdefault(p, "")
        return
    data = proc.stdout
    idx = 0
    for path in paths:
        if idx >= len(data):
            _TEXT_CACHE[path] = ""
            continue
        nl = data.find(b"\n", idx)
        if nl < 0:
            _TEXT_CACHE[path] = ""
            break
        header = data[idx:nl].decode("utf-8", errors="replace")
        idx = nl + 1
        if " missing" in header:
            _TEXT_CACHE[path] = ""
            continue
        parts = header.split()
        if len(parts) < 3 or parts[1] != "blob":
            _TEXT_CACHE[path] = ""
            continue
        try:
            size = int(parts[2])
        except ValueError:
            _TEXT_CACHE[path] = ""
            continue
        blob = data[idx : idx + size]
        idx = idx + size
        if idx < len(data) and data[idx : idx + 1] == b"\n":
            idx += 1
        _TEXT_CACHE[path] = blob.decode("utf-8", errors="replace")


def prefetch_texts(files: Sequence[Path], *, source: str | None = None) -> None:
    """Load file text. Default is HEAD blobs, matching hawking-index.

    `source='worktree'` reads on-disk bytes when the file exists (dirty
    edits visible) and falls back to HEAD for sparse-missing paths.
    A non-git REPO always reads the filesystem.
    """
    if _reader_var.get() is not None:
        # An external reader owns bytes for this scope; do not fill the
        # process-wide cache with HEAD-or-empty stand-ins for overlay paths.
        return
    token = _source_var.set(source) if source is not None else None
    try:
        pending = [p for p in files if p not in _TEXT_CACHE]
        if not pending:
            return
        src = current_source()
        if src == SOURCE_HEAD and repo_is_git_checkout():
            _prefetch_git_blobs(pending)
            for path in pending:
                _TEXT_CACHE.setdefault(path, "")
            return
        missing: list[Path] = []
        for path in pending:
            if path.is_file():
                try:
                    _TEXT_CACHE[path] = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    _TEXT_CACHE[path] = ""
            else:
                missing.append(path)
        if missing:
            _prefetch_git_blobs(missing)
            for path in missing:
                _TEXT_CACHE.setdefault(path, "")
    finally:
        if token is not None:
            _source_var.reset(token)


def read_text(path: Path) -> str:
    reader = _reader_var.get()
    if reader is not None:
        return reader(path)
    if path not in _TEXT_CACHE:
        prefetch_texts([path])
    return _TEXT_CACHE.get(path, "")


def is_test_path(path: Path) -> bool:
    name = path.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return "tests" in path.parts


_REL_MEMO: dict[str, str] = {}


def rel(path: Path) -> str:
    key = f"{REPO}\0{path}"
    cached = _REL_MEMO.get(key)
    if cached is not None:
        return cached
    try:
        value = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        value = str(path)
    _REL_MEMO[key] = value
    return value


def module_name_of(path: Path) -> str:
    parts = list(path.resolve().relative_to(REPO).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_name_from_rel(rp: str) -> str:
    """Same as module_name_of, from a repo-relative posix path (no stat)."""
    without = rp[:-3] if rp.endswith(".py") else rp
    parts = [p for p in without.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# --------------------------------------------------------------------------
# Site: one piece of evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    file: str
    line: int
    kind: str  # "import" | "call" | "literal" | "subprocess" | "definition"

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "line": self.line, "kind": self.kind}


# --------------------------------------------------------------------------
# Import index: module dotted-name -> sites that import it
# --------------------------------------------------------------------------


def _resolved_from_modules(importer: Path, node: ast.ImportFrom) -> list[str]:
    """Every base module dotted-name `from X import ...` could mean here:
    the properly resolved package-relative name (handling `level>0`), plus
    the sys.path.insert(dirname(__file__)); from _common import x sibling-
    file idiom this package uses when that resolved name has no dots and a
    same-directory file matches it. Almost always one answer; occasionally
    two, and both are real, so both are kept."""
    bases: list[str] = []
    if node.level and node.level > 0:
        importer_mod = module_name_of(importer)
        parts = importer_mod.split(".")
        is_init = importer.name == "__init__.py"
        base_parts = parts if is_init else parts[:-1]
        if node.level > 1:
            cut = node.level - 1
            base_parts = base_parts[: max(0, len(base_parts) - cut)]
        base = ".".join(base_parts)
        mod = f"{base}.{node.module}" if node.module else base
        if mod:
            bases.append(mod)
    else:
        mod = node.module or ""
        if mod:
            bases.append(mod)
            if "." not in mod:
                sib = importer.parent / f"{mod}.py"
                if source_exists(sib) and sib != importer:
                    bases.append(module_name_of(sib))
    return bases


def _resolve_import_targets(importer: Path, node: ast.AST) -> list[str]:
    """Every dotted module name one Import/ImportFrom node could refer to."""
    targets: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
            sib = importer.parent / (alias.name.split(".")[0] + ".py")
            if "." not in alias.name and source_exists(sib) and sib != importer:
                targets.append(module_name_of(sib))
    elif isinstance(node, ast.ImportFrom):
        for mod in _resolved_from_modules(importer, node):
            targets.append(mod)
            for alias in node.names:
                targets.append(f"{mod}.{alias.name}")
    return targets


@dataclass
class RepoIndex:
    files: list[Path]
    import_sites: dict[str, list[Site]] = field(default_factory=dict)
    # per (importer_file) -> {local_name: symbol_dotted_or_module}
    bound_names: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # Precomputed by hawking-index. None → fall back to a CPython ast walk.
    call_table: dict[str, list[tuple[int, str, str | None]]] | None = None
    subprocess_table: dict[str, list[tuple[int, list[str]]]] | None = None
    literal_table: dict[str, list[Site]] | None = None
    facts_source: str = "python-ast"

    def add_import(self, target: str, site: Site) -> None:
        self.import_sites.setdefault(target, []).append(site)


def build_repo_index(
    files: Sequence[Path] | None = None, *, source: str = DEFAULT_SOURCE
) -> RepoIndex:
    with using_source(source):
        idx = RepoIndex(files=list(files) if files is not None else repo_py_files())
        prefetch_texts(idx.files)
        _fill_repo_index(idx)
        idx.facts_source = "python-ast"
        return idx


# Capture the tool-name token from a dispatch-shaped line. Equivalent to
# _tool_dispatch_pattern(token) for every token, computed once per file
# instead of once per (tool × file) during assembly.
_DISPATCH_CAPTURE_RE = re.compile(
    r"""(?:"tool"|'tool'|\btool)\s*[:=]\s*(['"])([^'"]+)\1"""
    r"""|invoke\(\s*(['"])([^'"]+)\3"""
)

_FANOUT_THRESHOLD = 64


def _sibling_rel(importer_rp: str, mod: str) -> str:
    parent = importer_rp.rsplit("/", 1)[0] if "/" in importer_rp else ""
    return f"{parent}/{mod}.py" if parent else f"{mod}.py"


def _resolved_from_modules_rel(
    importer_rp: str, node: ast.ImportFrom, py_rels: frozenset[str]
) -> list[str]:
    """Same binding rules as _resolved_from_modules, from a repo-relative path."""
    bases: list[str] = []
    name = importer_rp.rsplit("/", 1)[-1]
    if node.level and node.level > 0:
        importer_mod = _module_name_from_rel(importer_rp)
        parts = importer_mod.split(".")
        is_init = name == "__init__.py"
        base_parts = parts if is_init else parts[:-1]
        if node.level > 1:
            cut = node.level - 1
            base_parts = base_parts[: max(0, len(base_parts) - cut)]
        base = ".".join(base_parts)
        mod = f"{base}.{node.module}" if node.module else base
        if mod:
            bases.append(mod)
    else:
        mod = node.module or ""
        if mod:
            bases.append(mod)
            if "." not in mod:
                sib = _sibling_rel(importer_rp, mod)
                if sib in py_rels and sib != importer_rp:
                    bases.append(_module_name_from_rel(sib))
    return bases


def _resolve_import_targets_rel(
    importer_rp: str, node: ast.AST, py_rels: frozenset[str]
) -> list[str]:
    targets: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
            if "." not in alias.name:
                sib = _sibling_rel(importer_rp, alias.name.split(".")[0])
                if sib in py_rels and sib != importer_rp:
                    targets.append(_module_name_from_rel(sib))
    elif isinstance(node, ast.ImportFrom):
        for mod in _resolved_from_modules_rel(importer_rp, node, py_rels):
            targets.append(mod)
            for alias in node.names:
                targets.append(f"{mod}.{alias.name}")
    return targets


def _extract_file_facts(
    rp: str, text: str, py_rels: frozenset[str]
) -> dict[str, Any]:
    """One-pass facts for a file. Pure: no git, no REPO, pickleable.

    Filling call/subprocess/literal tables here means assemble() does not
    re-parse every file once per capability (the 12s that used to sit on
    top of the 10s first parse).
    """
    empty: dict[str, Any] = {
        "rp": rp,
        "imports": [],
        "binds": [],
        "calls": [],
        "subprocess": [],
        "literals": [],
    }
    if not text:
        return empty
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return empty
    imports: list[tuple[str, int]] = []
    binds: list[tuple[str, str]] = []
    calls: list[tuple[int, str, str | None]] = []
    subproc: list[tuple[int, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for t in _resolve_import_targets_rel(rp, node, py_rels):
                imports.append((t, node.lineno))
            if isinstance(node, ast.ImportFrom):
                for mod in _resolved_from_modules_rel(rp, node, py_rels):
                    for alias in node.names:
                        local = alias.asname or alias.name
                        binds.append((local, f"{mod}.{alias.name}"))
            else:
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    binds.append((local, alias.name))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append((node.lineno, func.id, None))
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                calls.append((node.lineno, func.attr, func.value.id))
            if _is_subprocess_call(node):
                strings = [
                    inner.value
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
                ]
                subproc.append((node.lineno, strings))
    literals: list[tuple[str, int]] = []
    if '"tool"' in text or "'tool'" in text or "invoke(" in text or "tool=" in text or "tool :" in text:
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _DISPATCH_CAPTURE_RE.finditer(line):
                token = m.group(2) or m.group(4)
                if token:
                    literals.append((token, lineno))
    return {
        "rp": rp,
        "imports": imports,
        "binds": binds,
        "calls": calls,
        "subprocess": subproc,
        "literals": literals,
    }


def _merge_file_facts(idx: RepoIndex, facts: Mapping[str, Any]) -> None:
    rp = str(facts["rp"])
    for target, lineno in facts.get("imports") or []:
        idx.add_import(str(target), Site(rp, int(lineno), "import"))
    idx.bound_names[rp] = [(str(a), str(b)) for a, b in (facts.get("binds") or [])]
    if idx.call_table is not None:
        idx.call_table[rp] = [
            (int(ln), str(name), None if q is None else str(q))
            for ln, name, q in (facts.get("calls") or [])
        ]
    if idx.subprocess_table is not None:
        idx.subprocess_table[rp] = [
            (int(ln), [str(s) for s in strings])
            for ln, strings in (facts.get("subprocess") or [])
        ]
    if idx.literal_table is not None:
        for token, lineno in facts.get("literals") or []:
            idx.literal_table.setdefault(str(token), []).append(
                Site(rp, int(lineno), "literal")
            )


def _fanout_worker_main() -> None:
    """stdin: pickle {py_rels, items}; stdout: pickle [facts]."""
    import pickle as _pickle

    payload = _pickle.load(_sys.stdin.buffer)
    py_rels = frozenset(payload["py_rels"])
    out = [_extract_file_facts(rp, text, py_rels) for rp, text in payload["items"]]
    _pickle.dump(out, _sys.stdout.buffer, protocol=_pickle.HIGHEST_PROTOCOL)


def _extract_file_facts_fanout(
    items: Sequence[tuple[str, str]], py_rels: frozenset[str]
) -> list[dict[str, Any]]:
    """N independent python processes, no multiprocessing semaphore.

    ThreadPoolExecutor cannot beat the GIL on ast.parse (measured: 8.6s
    serial, 9.3s 8 threads). multiprocessing.ProcessPoolExecutor raises
    PermissionError on this host (sem_open). subprocess.Popen of N
    interpreters does not. 28 workers: 2.4s for 2389 files / 52MB.
    """
    import pickle as _pickle
    from concurrent.futures import ThreadPoolExecutor

    n = min(_os.cpu_count() or 8, len(items))
    if n < 2:
        return [_extract_file_facts(rp, text, py_rels) for rp, text in items]
    chunks: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    for i, pair in enumerate(items):
        chunks[i % n].append(pair)
    py_rels_list = list(py_rels)
    worker = (
        "from tools.future.capability_reachability import _fanout_worker_main; "
        "_fanout_worker_main()"
    )
    env = dict(_os.environ)
    env["PYTHONPATH"] = _os.pathsep.join(_sys.path)

    def _run(chunk: list[tuple[str, str]]) -> list[dict[str, Any]]:
        proc = subprocess.run(
            [_sys.executable, "-c", worker],
            input=_pickle.dumps({"py_rels": py_rels_list, "items": chunk}),
            capture_output=True,
            cwd=str(REPO),
            env=env,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-400:].decode("utf-8", errors="replace"))
        return _pickle.loads(proc.stdout)

    with ThreadPoolExecutor(max_workers=n) as pool:
        parts = list(pool.map(_run, chunks))
    out: list[dict[str, Any]] = []
    for part in parts:
        out.extend(part)
    return out


def _fill_repo_index(idx: RepoIndex) -> None:
    idx.call_table = {}
    idx.subprocess_table = {}
    idx.literal_table = {}
    items = [(rel(p), read_text(p)) for p in idx.files]
    items = [(rp, text) for rp, text in items if text]
    if not items:
        return
    py_rels = frozenset(rel(p) for p in idx.files)
    facts_list: list[dict[str, Any]]
    use_fanout = (
        len(items) >= _FANOUT_THRESHOLD
        and not _os.environ.get("HAWKING_REACHABILITY_SERIAL")
        and _reader_var.get() is None
    )
    if use_fanout:
        try:
            facts_list = _extract_file_facts_fanout(items, py_rels)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            facts_list = [_extract_file_facts(rp, text, py_rels) for rp, text in items]
    else:
        facts_list = [_extract_file_facts(rp, text, py_rels) for rp, text in items]
    for facts in facts_list:
        _merge_file_facts(idx, facts)


def find_symbol_call_sites(
    idx: RepoIndex,
    module_dotted: str,
    symbol: str,
    *,
    exclude_files: Iterable[Path] = (),
) -> list[Site]:
    """Call sites of `symbol` defined in `module_dotted`, found in files that
    import that name (directly, or as the module + attribute access)."""
    excluded = {rel(p) for p in exclude_files}
    sites: list[Site] = []
    target_full = f"{module_dotted}.{symbol}"
    for path in idx.files:
        rp = rel(path)
        if rp in excluded:
            continue
        binds = idx.bound_names.get(rp, [])
        local_direct = {local for local, full in binds if full == target_full}
        local_module = {local for local, full in binds if full == module_dotted}
        if _module_name_from_rel(rp) == module_dotted:
            # The symbol's own defining file calls it by bare name with no
            # import at all -- a `def _neighborhood_lines(...)` used
            # elsewhere in the same module is a real call site an AST Call
            # node distinguishes cleanly from the def line itself.
            local_direct = local_direct | {symbol}
        if not local_direct and not local_module:
            continue
        if idx.call_table is not None:
            for lineno, name, qualifier in idx.call_table.get(rp, []):
                if qualifier is None and name in local_direct:
                    sites.append(Site(rp, lineno, "call"))
                elif qualifier is not None and name == symbol and qualifier in local_module:
                    sites.append(Site(rp, lineno, "call"))
            continue
        text = read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in local_direct:
                sites.append(Site(rp, node.lineno, "call"))
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == symbol
                and isinstance(func.value, ast.Name)
                and func.value.id in local_module
            ):
                sites.append(Site(rp, node.lineno, "call"))
    return sites


def find_symbol_instantiation_sites(
    idx: RepoIndex, module_dotted: str, class_name: str, *, exclude_files: Iterable[Path] = ()
) -> list[Site]:
    return find_symbol_call_sites(idx, module_dotted, class_name, exclude_files=exclude_files)


_TOOL_DISPATCH_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _tool_dispatch_pattern(tool_name: str) -> re.Pattern[str]:
    """A quoted tool name only counts as a call site when it sits somewhere
    that actually dispatches a tool by that name: a `"tool": "<name>"` (or
    `'tool': '<name>'`) catalog-entry key, a `tool="<name>"` kwarg, or a
    `.invoke("<name>"` / `invoke("<name>"` first positional argument.

    Without this, a bare quoted "git.status"/"fs.read" anywhere in the repo
    counts -- including tools/haider/p0_tool_bridge.py's OWN, unrelated
    ALL_TOOLS set for a different HAIDER API bridge that happens to reuse a
    few of the same short names. Same string, two unconnected systems; only
    the dispatch-shaped occurrence is evidence for THIS registry.
    """
    if tool_name not in _TOOL_DISPATCH_RE_CACHE:
        esc = re.escape(tool_name)
        _TOOL_DISPATCH_RE_CACHE[tool_name] = re.compile(
            r"""(?:"tool"|'tool'|\btool)\s*[:=]\s*(['"])""" + esc + r"""\1"""
            r"""|invoke\(\s*(['"])""" + esc + r"""\2"""
        )
    return _TOOL_DISPATCH_RE_CACHE[tool_name]


def find_literal_sites(
    token: str,
    files: Sequence[Path],
    *,
    exclude_files: Iterable[Path] = (),
    kind: str = "literal",
    idx: RepoIndex | None = None,
) -> list[Site]:
    excluded = {rel(p) for p in exclude_files}
    if idx is not None and idx.literal_table is not None:
        return [
            Site(s.file, s.line, kind)
            for s in idx.literal_table.get(token, [])
            if s.file not in excluded
        ]
    pattern = _tool_dispatch_pattern(token)
    sites: list[Site] = []
    for path in files:
        rp = rel(path)
        if rp in excluded:
            continue
        text = read_text(path)
        if token not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                sites.append(Site(rp, lineno, kind))
    return sites


def find_module_import_sites(
    idx: RepoIndex, module_dotted: str, *, exclude_files: Iterable[Path] = ()
) -> list[Site]:
    excluded = {rel(p) for p in exclude_files}
    hits = idx.import_sites.get(module_dotted, [])
    return [s for s in hits if s.file not in excluded]


# --------------------------------------------------------------------------
# Capability record assembly
# --------------------------------------------------------------------------


def _partition(sites: Sequence[Site]) -> tuple[list[Site], list[Site]]:
    """(production_sites, test_sites). Deduplicated and sorted."""
    seen: set[tuple[str, int, str]] = set()
    uniq: list[Site] = []
    for s in sites:
        key = (s.file, s.line, s.kind)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    uniq.sort(key=lambda s: (s.file, s.line))
    prod = [s for s in uniq if not is_test_path(Path(s.file))]
    test = [s for s in uniq if is_test_path(Path(s.file))]
    return prod, test


def _imported_directly_by_hcli(prod_sites: Sequence[Site]) -> bool:
    """True if some non-test site is inside hcli/ itself -- the resident's
    own runtime importing this by hand, bypassing ToolRegistry entirely.
    A second, real discovery channel distinct from being a ToolSpec."""
    return any(s.file.startswith("hcli/") and s.kind in ("import", "call") for s in prod_sites)


def build_capability(
    name: str,
    kind: str,
    *,
    defined: bool,
    registered: bool,
    resident_visible: bool | None,
    sites: Sequence[Site],
    definition: Mapping[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """`resident_visible=None` derives it: registered, or directly imported
    by hcli/ itself (a real second discovery channel that bypasses
    ToolRegistry -- see tools.future.status_causality for why this matters:
    it is imported by eight hcli/agentos/*_gate.py modules despite never
    being a ToolSpec). Pass an explicit bool to assert the ToolSpec-wrapping
    channel instead (typed tools, and functions a ToolSpec wraps)."""
    prod, test = _partition(sites)
    if resident_visible is None:
        resident_visible = bool(registered) or _imported_directly_by_hcli(prod)
    return {
        "name": name,
        "kind": kind,
        "defined": bool(defined),
        "registered": bool(registered),
        "resident_visible": bool(resident_visible),
        "callable": bool(prod),
        "tested": bool(test),
        "call_sites": [s.to_dict() for s in prod],
        "test_only_sites": [s.to_dict() for s in test],
        "definition": dict(definition) if definition else None,
        "note": note,
    }


# --------------------------------------------------------------------------
# hcli/tool_registry.py: the typed tool registry and its 44 tools
# --------------------------------------------------------------------------


def recover_tool_registrations(registry_path: Path) -> list[tuple[str, int]]:
    """(tool_name, def_line) for every ToolSpec in default_tool_registry.

    Two literal shapes are used: `ToolSpec("name", ...)` directly, and a
    `for name in (...)`/`for name, description in (...)` loop whose body
    calls `ToolSpec(name, ...)`. AST alone resolves the first; the second is
    recovered by reading the loop's own iterable literal.
    """
    text = read_text(registry_path)
    tree = ast.parse(text)
    names: list[tuple[str, int]] = []

    def const_str(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def is_toolspec_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        return (isinstance(f, ast.Name) and f.id == "ToolSpec") or (
            isinstance(f, ast.Attribute) and f.attr == "ToolSpec"
        )

    for node in ast.walk(tree):
        if is_toolspec_call(node) and node.args:
            literal = const_str(node.args[0])
            if literal:
                names.append((literal, node.lineno))
                continue
        if isinstance(node, ast.For) and isinstance(node.target, (ast.Name, ast.Tuple)):
            loop_names: list[str] = []
            if isinstance(node.iter, (ast.Tuple, ast.List)):
                for elt in node.iter.elts:
                    if isinstance(elt, ast.Tuple) and elt.elts:
                        s = const_str(elt.elts[0])
                    else:
                        s = const_str(elt)
                    if s:
                        loop_names.append(s)
            has_toolspec = any(is_toolspec_call(n) for n in ast.walk(node) if n is not node)
            if loop_names and has_toolspec:
                names.extend((n, node.lineno) for n in loop_names)
    return names


def build_tool_capabilities(idx: RepoIndex, registry_path: Path) -> list[dict[str, Any]]:
    caps: list[dict[str, Any]] = []
    for tool_name, lineno in recover_tool_registrations(registry_path):
        sites = find_literal_sites(
            tool_name, idx.files, exclude_files=(registry_path,), kind="literal", idx=idx
        )
        prod, _test = _partition(sites)
        if prod:
            # Some gates (hcli/agentos/autonomy_gate.py and siblings) build a
            # literal `"tool": "<name>"` source catalog for a census
            # mission -- a real, static, scripted dispatch, not model JSON.
            note = (
                "resident_visible because default_tool_registry() registers every "
                "tool unconditionally; callable is real here -- see call_sites for "
                "the static `\"tool\": " + repr(tool_name) + "` catalog entry (or "
                "other quoted use) that proves it, outside any test file."
            )
        else:
            note = (
                "resident_visible because default_tool_registry() registers every "
                "tool unconditionally and AgentOS.tools is that registry; callable "
                "requires a quoted use of the tool's string key outside "
                f"{rel(registry_path)} and outside any test file. No static gate "
                "catalog and no other reference names this tool; the model could "
                "still propose it as a runtime WorkUnit.tool string, which no "
                "static search can see, so callable=false means 'no deterministic "
                "path', not 'never invoked'."
            )
        caps.append(
            build_capability(
                f"tool:{tool_name}",
                "typed_tool",
                defined=True,
                registered=True,
                resident_visible=True,
                sites=sites,
                definition={"file": rel(registry_path), "line": lineno},
                note=note,
            )
        )
    return caps


def build_tool_registry_capability(idx: RepoIndex, registry_path: Path) -> dict[str, Any]:
    sites = find_symbol_call_sites(idx, "hcli.tool_registry", "default_tool_registry", exclude_files=(registry_path,))
    return build_capability(
        "hcli.tool_registry.default_tool_registry",
        "function",
        defined=source_exists(registry_path),
        registered=True,
        resident_visible=True,
        sites=sites,
        definition={"file": rel(registry_path), "line": None},
        note="The constructor every other capability's resident_visible depends on.",
    )


# --------------------------------------------------------------------------
# Named capabilities the wave asked for by name
# --------------------------------------------------------------------------


def _module_capability(
    idx: RepoIndex,
    display_name: str,
    module_dotted: str,
    def_file: Path,
    *,
    registered: bool = False,
    resident_visible: bool | None = None,
    extra_sites: Sequence[Site] = (),
    note: str | None = None,
) -> dict[str, Any]:
    sites = find_module_import_sites(idx, module_dotted, exclude_files=(def_file,))
    sites = list(sites) + list(extra_sites)
    return build_capability(
        display_name,
        "module",
        defined=source_exists(def_file),
        registered=registered,
        resident_visible=resident_visible,
        sites=sites,
        definition={"file": rel(def_file), "line": None},
        note=note,
    )


def _function_capability(
    idx: RepoIndex,
    display_name: str,
    module_dotted: str,
    symbol: str,
    def_file: Path,
    *,
    registered: bool = False,
    resident_visible: bool | None = None,
    exclude_files: Iterable[Path] = (),
    note: str | None = None,
) -> dict[str, Any]:
    sites = find_symbol_call_sites(idx, module_dotted, symbol, exclude_files=exclude_files)
    return build_capability(
        display_name,
        "function",
        defined=source_exists(def_file),
        registered=registered,
        resident_visible=resident_visible,
        sites=sites,
        definition={"file": rel(def_file), "line": None},
        note=note,
    )


_SUBPROCESS_CALL_NAMES = frozenset({"run", "Popen", "check_call", "check_output", "call"})


def _is_subprocess_call(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr in _SUBPROCESS_CALL_NAMES
    if isinstance(f, ast.Name):
        return f.id in _SUBPROCESS_CALL_NAMES
    return False


def _subprocess_path_sites(
    rel_path: str,
    files: Sequence[Path],
    *,
    exclude_files: Iterable[Path],
    subprocess_table: dict[str, list[tuple[int, list[str]]]] | None = None,
) -> list[Site]:
    """A real launch, not a mention. `"tools/x.py"` sitting in a metadata
    dict (a handoff receipt's "unrelated_preserved_edit" field, a doc string
    describing what a launchd plist loops) is not a call site -- only a
    string literal reachable from inside an actual subprocess.run/Popen/
    check_call/check_output/call invocation counts."""
    excluded = {rel(p) for p in exclude_files}
    if subprocess_table is not None:
        allowed = {rel(p) for p in files} - excluded
        sites: list[Site] = []
        for rp, entries in subprocess_table.items():
            if rp not in allowed:
                continue
            for lineno, strings in entries:
                if any(rel_path in s for s in strings):
                    sites.append(Site(rp, lineno, "subprocess"))
        return sites
    sites: list[Site] = []
    for path in files:
        rp = rel(path)
        if rp in excluded or rel_path not in read_text(path):
            continue
        try:
            tree = ast.parse(read_text(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_subprocess_call(node)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str) and rel_path in inner.value:
                    sites.append(Site(rp, node.lineno, "subprocess"))
                    break
    return sites


def build_named_capabilities(idx: RepoIndex) -> list[dict[str, Any]]:
    caps: list[dict[str, Any]] = []
    registry_path = REPO / TOOL_REGISTRY_REL

    # hcli/mutation.py -- WAVE_STATE flags this as "suspected" dead surface
    # ("zero callers in hcli/"), citing tools/headless/startup_census.py's
    # hand-typed "no production importer found". Current HEAD disagrees:
    # hcli/engine.py, hcli/delegate.py and hcli/agentos/__init__.py all
    # import from it directly, outside any test. Either those imports are
    # newer than that census, or the census's "production importer" means
    # something narrower (reachable from a cold `python -m hcli --help`,
    # not "any non-test import statement anywhere"). Reported as measured,
    # not rounded to match the suspicion.
    mutation_file = REPO / "hcli" / "mutation.py"
    caps.append(
        _module_capability(
            idx,
            "hcli.mutation",
            "hcli.mutation",
            mutation_file,
            note=(
                "WAVE_STATE/startup_census.py suspected this had zero "
                "production callers; current HEAD's call_sites show real, "
                "non-test imports from hcli/engine.py, hcli/delegate.py and "
                "hcli/agentos/__init__.py. Not dead surface as of this run -- "
                "see call_sites rather than the prior suspicion."
            ),
        )
    )

    # tools/future/modellake_events.py -- the wave's own worked example of
    # the trap, PLUS a live counter-example in the same module: build() is
    # genuinely called by the live watcher; maybe_emit_modellake_events and
    # emit_modellake_events_once are names that live in modellake_watch.py
    # itself, not in modellake_events.py -- checked directly here rather
    # than assumed from the wave's own wording.
    events_file = REPO / FUTURE_DIR_REL / "modellake_events.py"
    caps.append(
        _function_capability(
            idx,
            "tools.future.modellake_events.build",
            "tools.future.modellake_events",
            "build",
            events_file,
            note=(
                "Called from tools/odyssey/modellake_watch.py's own "
                "emit_modellake_events_once(), which the LIVE modellake "
                "watcher process (see WAVE_STATE's tracked pids) runs on an "
                "interval via maybe_emit_modellake_events(). Not dead -- the "
                "opposite of the wave's worked example, in the same module."
            ),
        )
    )
    caps.append(
        _module_capability(
            idx,
            "tools.future.modellake_events",
            "tools.future.modellake_events",
            events_file,
            note=(
                "tools/odyssey/modellake_watch.py imports this module and "
                "calls me.build() from emit_modellake_events_once(). Not dead."
            ),
        )
    )

    # hcli/ane_provider.py -- module import vs. class instantiation are two
    # different questions, and they give two different answers here.
    ane_file = REPO / "hcli" / "ane_provider.py"
    caps.append(
        _module_capability(
            idx,
            "hcli.ane_provider",
            "hcli.ane_provider",
            ane_file,
            note=(
                "Imported by hcli/agentos/__init__.py (re-exported in its "
                "__all__). See the class entry below: the import alone does "
                "not mean the class is ever constructed."
            ),
        )
    )
    inst_sites = find_symbol_instantiation_sites(idx, "hcli.ane_provider", "ANEProvider", exclude_files=(ane_file,))
    caps.append(
        build_capability(
            "hcli.ane_provider.ANEProvider",
            "class",
            defined=source_exists(ane_file),
            registered=False,
            resident_visible=None,
            sites=inst_sites,
            definition={"file": rel(ane_file), "line": None},
            note=(
                "The module is imported by hcli/agentos/__init__.py (see the "
                "module-level entry above), but the class itself is never "
                "constructed (ANEProvider(...)) outside hcli/test_ane_provider.py. "
                "hcli/physical_graph.py branches on a dict payload's "
                "'kind' == 'ANEProvider' string; it never imports the class. "
                "An import that nobody instantiates is not a live capability."
            ),
        )
    )

    # Odyssey APIs: tools/odyssey_ctl.py
    odyssey_file = REPO / "tools" / "odyssey_ctl.py"
    odyssey_sites = find_module_import_sites(idx, "tools.odyssey_ctl", exclude_files=(odyssey_file,))
    odyssey_sites += _subprocess_path_sites(
        "tools/odyssey_ctl.py",
        idx.files,
        exclude_files=(odyssey_file,),
        subprocess_table=idx.subprocess_table,
    )
    caps.append(
        build_capability(
            "tools.odyssey_ctl",
            "module",
            defined=source_exists(odyssey_file),
            registered=False,
            resident_visible=None,
            sites=odyssey_sites,
            definition={"file": rel(odyssey_file), "line": None},
            note=(
                "Not registered: hcli/tool_registry.py has no odyssey.* tool, "
                "so a resident cannot discover this through AgentOS. It is "
                "still callable: tools/odyssey_patient_runner.py does "
                "`from tools.odyssey_ctl import (...)`, a real production "
                "import outside any test."
            ),
        )
    )

    # Child/succession APIs
    resident_file = REPO / "hcli" / "agentos" / "resident.py"
    caps.append(
        _function_capability(
            idx,
            "hcli.agentos.resident._child_workunit",
            "hcli.agentos.resident",
            "_child_workunit",
            resident_file,
            registered=False,
            resident_visible=None,
            note=(
                "The model's only path to typed-tool dispatch: a child "
                "WorkUnit proposal with a 'tool' key routes here, then to "
                "ToolRegistry.invoke. Its only call site is inside "
                "hcli/agentos/resident.py itself (admit_evidence_children); "
                "resident_visible is true because that IS hcli/, not because "
                "it is a discoverable ToolSpec -- it never is one. This is "
                "always-running mission plumbing, not an optional capability "
                "a resident chooses to invoke."
            ),
        )
    )
    succession_file = REPO / FUTURE_DIR_REL / "succession.py"
    caps.append(
        _module_capability(
            idx,
            "tools.future.succession",
            "tools.future.succession",
            succession_file,
            note=(
                "Imported only by tools/future/succession_trial.py (a sibling "
                "sidecar harness, itself not wired into hcli or the sovereign "
                "loop's EXECUTABLE dispatch) and by its own test. "
                "tools/future/tabula.py explicitly comments "
                "'not imported' about this module. No path from the resident "
                "reaches it."
            ),
        )
    )

    # hcli/escalation.py -- and its three ToolSpec-wrapped entry points.
    escalation_file = REPO / "hcli" / "escalation.py"
    caps.append(
        _module_capability(
            idx,
            "hcli.escalation",
            "hcli.escalation",
            escalation_file,
            note="See the three wrapped functions below for the real reachability picture.",
        )
    )
    for symbol, tool in (
        ("escalate_to_frontier", "frontier.escalate"),
        ("propose_swarm", "grok.swarm.propose"),
        ("launch_swarm", "grok.swarm.launch"),
    ):
        sites = find_symbol_call_sites(
            idx, "hcli.escalation", symbol, exclude_files=(escalation_file,)
        )
        caps.append(
            build_capability(
                f"hcli.escalation.{symbol}",
                "function",
                defined=source_exists(escalation_file),
                registered=True,
                resident_visible=True,
                sites=sites,
                definition={"file": rel(escalation_file), "line": None},
                note=(
                    f"Wrapped by hcli/tool_registry.py's {tool!r} ToolSpec handler, "
                    f"which is its only caller. resident_visible follows the "
                    f"{tool!r} ToolSpec; reaching it still requires the model to "
                    f"propose tool={tool!r} at runtime (see tool:{tool} above)."
                ),
            )
        )

    # The compactor: tools/future/hcli_compactor.py
    compactor_file = REPO / FUTURE_DIR_REL / "hcli_compactor.py"
    caps.append(
        _module_capability(
            idx,
            "tools.future.hcli_compactor",
            "tools.future.hcli_compactor",
            compactor_file,
            note=(
                "compact()/compact_with_stats() have a passing selftest "
                "(WAVE_STATE: ratio 0.2645, CONTINUITY) but no importer besides "
                "tools/future/test_hcli_compactor.py. 'Works' and 'is called' "
                "are different questions; this answers only the second."
            ),
        )
    )

    # Retrieval: hcli/goal.py neighborhood
    goal_file = REPO / "hcli" / "goal.py"
    neighborhood_sites = find_symbol_call_sites(idx, "hcli.goal", "_neighborhood_lines")
    caps.append(
        build_capability(
            "hcli.goal._neighborhood_lines",
            "function",
            defined=source_exists(goal_file),
            registered=False,
            resident_visible=None,
            sites=neighborhood_sites,
            definition={"file": rel(goal_file), "line": None},
            note=(
                "Self-file caller only (its own module's WorkerPacket "
                "assembly, the same function that builds the NEIGHBORHOOD "
                "section of every prompt); intentionally not excluded as "
                "'own file' the way tool-name literals are, because a Call "
                "node in the same file is still a real call site an AST can "
                "point at, unlike a string literal's defining occurrence. "
                "goal.py runs every mission cycle, so this reflects genuine "
                "internal wiring, not isolation."
            ),
        )
    )

    return caps


# --------------------------------------------------------------------------
# The bulk tools/future sidecar sweep
# --------------------------------------------------------------------------

_ALREADY_DETAILED_STEMS = frozenset(
    {"modellake_events", "succession", "hcli_compactor"}
)


def discover_future_modules(future_dir: Path | None = None) -> list[Path]:
    """`tools/future/*.py` sidecar modules from the current source of truth.

    HEAD listing (default) matches rust `git ls-tree`; a filesystem glob would
    miss sparse-missing files and would credit untracked files rust cannot see.
    """
    prefix = FUTURE_DIR_REL + "/"
    out: list[Path] = []
    for rp in sorted(tracked_py_set()):
        if not rp.startswith(prefix) or not rp.endswith(".py"):
            continue
        rest = rp[len(prefix) :]
        if "/" in rest:
            continue
        name = rest
        if name in ("__init__.py", "_common.py") or name.startswith("test_"):
            continue
        out.append(REPO / rp)
    if out:
        return out
    base = future_dir if future_dir is not None else REPO / FUTURE_DIR_REL
    if not base.is_dir():
        return []
    fallback = []
    for p in sorted(base.glob("*.py")):
        if p.name in ("__init__.py", "_common.py"):
            continue
        if p.name.startswith("test_"):
            continue
        fallback.append(p)
    return fallback


def build_sidecar_sweep(idx: RepoIndex, registered_future_tools: frozenset[str]) -> list[dict[str, Any]]:
    caps: list[dict[str, Any]] = []
    for path in discover_future_modules():
        stem = path.stem
        mod = module_name_of(path)
        sites = find_module_import_sites(idx, mod, exclude_files=(path,))
        sites += _subprocess_path_sites(
            rel(path), idx.files, exclude_files=(path,), subprocess_table=idx.subprocess_table
        )
        registered = any(stem in t or mod in t for t in registered_future_tools)
        detail_note = None
        if stem in _ALREADY_DETAILED_STEMS:
            detail_note = "See the detailed entry above; this row is the generic sweep's version of the same evidence."
        caps.append(
            build_capability(
                mod,
                "sidecar_module",
                defined=True,
                registered=registered,
                resident_visible=None,
                sites=sites,
                definition={"file": rel(path), "line": None},
                note=detail_note,
            )
        )
    return caps


def recover_registered_future_tool_names(registry_path: Path) -> frozenset[str]:
    names = {n for n, _ in recover_tool_registrations(registry_path)}
    return frozenset(n for n in names if n.startswith("future.") or "tools/future" in n or n.startswith("sidecar."))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _find_hawking_index_bin() -> Path | None:
    """Locate the `hawking-index` binary. A missing/unexecutable path degrades
    to the Python AST walk; it must never crash an unattended HCLI cycle."""
    env = _os.environ.get("HAWKING_INDEX_BIN")
    if env is not None:
        p = Path(env)
        if p.is_file() and _os.access(p, _os.X_OK):
            return p
        return None
    dirs: list[Path] = []
    cargo = _os.environ.get("CARGO_TARGET_DIR")
    if cargo:
        c = Path(cargo)
        dirs.extend([c / "release", c / "release-fast", c / "debug"])
    dirs.extend(
        [
            REPO / "workspace" / "ops" / "build" / "rust" / "release",
            REPO / "workspace" / "ops" / "build" / "rust" / "release-fast",
            REPO / "workspace" / "ops" / "build" / "rust" / "debug",
            REPO / "target" / "release",
            REPO / "target" / "release-fast",
            REPO / "target" / "debug",
        ]
    )
    for d in dirs:
        p = d / "hawking-index"
        if p.is_file() and _os.access(p, _os.X_OK):
            return p
    which = shutil.which("hawking-index")
    return Path(which) if which else None


def repo_index_from_facts(facts: Mapping[str, Any]) -> RepoIndex:
    files = [REPO / f for f in facts.get("files") or []]
    idx = RepoIndex(files=files, facts_source="hawking-index")
    idx.call_table = {}
    idx.subprocess_table = {}
    idx.literal_table = {}
    for target, sites in (facts.get("import_sites") or {}).items():
        for s in sites:
            idx.add_import(
                str(target),
                Site(s["file"], int(s["line"]), s.get("kind") or "import"),
            )
    for rp, binds in (facts.get("bound_names") or {}).items():
        idx.bound_names[str(rp)] = [(str(a), str(b)) for a, b in binds]
    for c in facts.get("calls") or []:
        rp = str(c["file"])
        idx.call_table.setdefault(rp, []).append(
            (int(c["line"]), str(c["name"]), c.get("qualifier"))
        )
    for s in facts.get("subprocess") or []:
        rp = str(s["file"])
        idx.subprocess_table.setdefault(rp, []).append(
            (int(s["line"]), [str(x) for x in (s.get("strings") or [])])
        )
    for lit in facts.get("literals") or []:
        tok = str(lit["token"])
        idx.literal_table.setdefault(tok, []).append(
            Site(str(lit["file"]), int(lit["line"]), "literal")
        )
    return idx


def _load_rust_facts() -> dict[str, Any] | None:
    if _os.environ.get("HAWKING_REACHABILITY_FORCE_PYTHON"):
        return None
    binary = _find_hawking_index_bin()
    if binary is None:
        return None
    cmd = [str(binary), "reachability-facts", "--root", str(REPO)]
    cache = _os.environ.get("HAWKING_INDEX_CACHE")
    cache_dir = Path(cache) if cache else (REPO / ".hide" / "reachability-index")
    cmd.extend(["--cache", str(cache_dir)])
    out_path = cache_dir / "facts.json"
    cmd.extend(["--output", str(out_path)])
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        facts = json.loads(out_path.read_text())
    except (ValueError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(facts, dict) or facts.get("schema") != RUST_FACTS_SCHEMA:
        return None
    return facts


_ASSEMBLE_MEMO: dict[tuple[Any, ...], dict[str, Any]] = {}


def _assemble_code_digest() -> str:
    """Bust the on-disk memo when this analyzer's source changes.

    The mutation check edits this file and re-runs pytest; a cache that
    ignored the bytes would keep a green receipt for a gutted analyzer.
    """
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except OSError:
        return "unreadable"


def _assemble_disk_cache_path(key: tuple[Any, ...]) -> Path | None:
    if not repo_is_git_checkout():
        return None
    sid = _os.environ.get("FUTURE_ARTIFACT_SESSION") or str(_os.getpid())
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
    d = Path(tempfile.gettempdir()) / f"hawking-future-art-{sid}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"assemble-{digest}.json"


def assemble(*, source: str = DEFAULT_SOURCE) -> dict[str, Any]:
    """Build the capability map.

    `source` is the file source of truth. Default ``head``: commit blobs,
    matching hawking-index. Pass ``source='worktree'`` only for an explicit
    working-tree view; that path does not use the rust dump (the dump is HEAD).

    Memoized per (REPO, source, force-python, bin hint, this-file digest).
    A mutated capability_reachability.py is a new key. Overlay readers and
    tmpdir REPOs skip the disk memo. xdist workers share the file via
    FUTURE_ARTIFACT_SESSION.
    """
    if _reader_var.get() is not None:
        return _assemble_uncached(source=source)
    force_py = bool(_os.environ.get("HAWKING_REACHABILITY_FORCE_PYTHON"))
    bin_hint = _os.environ.get("HAWKING_INDEX_BIN") or ""
    key = (str(REPO), source, force_py, bin_hint, _assemble_code_digest())
    cached = _ASSEMBLE_MEMO.get(key)
    if cached is not None:
        return cached
    disk = _assemble_disk_cache_path(key)
    if disk is not None:
        lockp = disk.with_suffix(".lock")
        with open(lockp, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            if disk.is_file():
                try:
                    loaded = json.loads(disk.read_text())
                except (ValueError, OSError):
                    loaded = None
                if isinstance(loaded, dict) and loaded.get("schema") == SCHEMA:
                    _ASSEMBLE_MEMO[key] = loaded
                    return loaded
            doc = _assemble_uncached(source=source)
            tmp = disk.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc))
            tmp.replace(disk)
            _ASSEMBLE_MEMO[key] = doc
            return doc
    doc = _assemble_uncached(source=source)
    _ASSEMBLE_MEMO[key] = doc
    return doc


def _assemble_uncached(*, source: str = DEFAULT_SOURCE) -> dict[str, Any]:
    with using_source(source):
        if current_source() == SOURCE_HEAD:
            facts = _load_rust_facts()
            if facts is not None:
                idx = repo_index_from_facts(facts)
                prefetch_texts([REPO / TOOL_REGISTRY_REL])
                doc = _assemble_from_index(idx)
                doc["facts_source"] = "hawking-index"
                doc["source"] = SOURCE_HEAD
                return doc
        idx = build_repo_index(source=current_source())
        doc = _assemble_from_index(idx)
        doc["facts_source"] = "python-ast"
        doc["source"] = current_source()
        return doc


def _assemble_from_index(idx: RepoIndex) -> dict[str, Any]:
    registry_path = REPO / TOOL_REGISTRY_REL

    tool_caps = build_tool_capabilities(idx, registry_path)
    registry_cap = build_tool_registry_capability(idx, registry_path)
    named_caps = build_named_capabilities(idx)
    future_tool_names = recover_registered_future_tool_names(registry_path)
    sidecar_caps = build_sidecar_sweep(idx, future_tool_names)

    all_caps = [registry_cap] + tool_caps + named_caps + sidecar_caps

    dead = [
        {
            "name": c["name"],
            "kind": c["kind"],
            "defined": c["defined"],
            "tested": c["tested"],
            "definition": c["definition"],
            "note": c["note"],
        }
        for c in all_caps
        if c["defined"] and not c["callable"]
    ]

    counts = {
        "capabilities": len(all_caps),
        "typed_tools": len(tool_caps),
        "typed_tools_callable": sum(1 for c in tool_caps if c["callable"]),
        "typed_tools_dead": sum(1 for c in tool_caps if not c["callable"]),
        "named_capabilities": len(named_caps),
        "named_callable": sum(1 for c in named_caps if c["callable"]),
        "sidecar_modules": len(sidecar_caps),
        "sidecar_callable": sum(1 for c in sidecar_caps if c["callable"]),
        "sidecar_registered": sum(1 for c in sidecar_caps if c["registered"]),
        "dead_surface": len(dead),
    }

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Answer, per capability and from evidence only: who calls it. "
            "A definition count is not a capability."
        ),
        "law": "A capability nothing calls does not exist. Grep for call sites, not definitions.",
        "fields": list(FIELDS),
        "counts": counts,
        "capabilities": {c["name"]: c for c in all_caps},
        "DEAD_SURFACE": dead,
        "method": (
            "Static source analysis only (AST import/call graph + quoted-"
            "literal search) over every HEAD `*.py` blob (`git ls-tree`). "
            "Cannot see runtime-only dispatch (a model-proposed WorkUnit.tool "
            "string); see the module docstring's 'Hard limit' section."
        ),
        "negative_findings": [
            f"{counts['typed_tools_dead']} of {counts['typed_tools']} typed tools have "
            "zero call sites outside hcli/tool_registry.py and outside every test file.",
            f"{counts['sidecar_modules'] - counts['sidecar_registered']} of "
            f"{counts['sidecar_modules']} tools/future sidecar modules are registered in "
            "no ToolSpec; a resident cannot discover them via AgentOS.tools.discover().",
            f"{counts['sidecar_modules'] - counts['sidecar_callable']} of "
            f"{counts['sidecar_modules']} tools/future sidecar modules have zero call "
            "sites outside their own test files.",
        ]
        + _tested_but_uncalled_named_findings(named_caps),
    }
    return doc


def _tested_but_uncalled_named_findings(named_caps: Sequence[Mapping[str, Any]]) -> list[str]:
    """Named (non-typed-tool, non-sidecar-sweep) capabilities with a passing
    test and zero callers outside it -- derived from this run's own
    evidence, never a fixed name list, so a fix (or a regression) is never
    silently out of date the way a hand-typed sentence would be."""
    culprits = sorted(
        c["name"] for c in named_caps if c["tested"] and not c["callable"] and c["defined"]
    )
    if not culprits:
        return []
    return [
        f"{', '.join(culprits)} each have a passing test and zero callers outside "
        "it; a green test does not make a capability reachable."
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build() -> Path:
    return write_receipt(RECEIPT, assemble(), RECORDED_BY)


def selftest() -> Path:
    out = build()
    doc = load_json(out)
    if doc.get("schema") != SCHEMA:
        raise AssertionError(f"schema drifted: {doc.get('schema')!r}")
    if not doc.get("seal_sha256"):
        raise AssertionError("receipt is unsealed")
    caps = doc.get("capabilities") or {}
    if len(caps) < 44:
        raise AssertionError(f"fewer than 44 capabilities recovered: {len(caps)}")
    for name, row in caps.items():
        for field_name in FIELDS:
            if not isinstance(row.get(field_name), bool):
                raise AssertionError(f"{name}.{field_name} is not boolean: {row.get(field_name)!r}")
        if row["callable"] and not row["call_sites"]:
            raise AssertionError(f"{name} is callable with no call_sites listed")
        if not row["callable"] and row["call_sites"]:
            raise AssertionError(f"{name} has call_sites but callable=false")
    tool_registry_row = caps.get("hcli.tool_registry.default_tool_registry")
    if tool_registry_row is None or not tool_registry_row["callable"]:
        raise AssertionError("default_tool_registry must show real production callers")
    # hcli/escalation.py's escalate_to_frontier is wired as the ONLY handler
    # body of the "frontier.escalate" ToolSpec in hcli/tool_registry.py --
    # about as structurally stable a "must be callable" fact as this repo
    # has, unlike a specific dead-surface count that shifts as code lands.
    escalate_row = caps.get("hcli.escalation.escalate_to_frontier")
    if escalate_row is None or not escalate_row["callable"]:
        raise AssertionError("hcli.escalation.escalate_to_frontier must show its tool_registry.py handler call site")
    dead = doc.get("DEAD_SURFACE") or []
    if not dead:
        raise AssertionError("DEAD_SURFACE is empty; at least one typed tool is always uncalled in this codebase")
    return out


def _capability_view(cap: Mapping[str, Any]) -> dict[str, Any]:
    """The fields the parity gate compares. Notes/method text are not verdicts."""
    return {
        "defined": cap.get("defined"),
        "registered": cap.get("registered"),
        "resident_visible": cap.get("resident_visible"),
        "callable": cap.get("callable"),
        "tested": cap.get("tested"),
        "call_sites": cap.get("call_sites") or [],
        "test_only_sites": cap.get("test_only_sites") or [],
    }


def compare_capability_maps(
    rust_caps: Mapping[str, Any], python_caps: Mapping[str, Any]
) -> list[str]:
    """Exact-equality diffs of the full capability objects. Empty list means IDENTICAL."""
    diffs: list[str] = []
    rust_keys = set(rust_caps)
    py_keys = set(python_caps)
    only_rust = sorted(rust_keys - py_keys)
    only_py = sorted(py_keys - rust_keys)
    if only_rust:
        diffs.append(f"keys only in rust ({len(only_rust)}): {only_rust[:20]}")
    if only_py:
        diffs.append(f"keys only in python ({len(only_py)}): {only_py[:20]}")
    for name in sorted(rust_keys & py_keys):
        a = rust_caps[name]
        b = python_caps[name]
        if a == b:
            continue
        fields = sorted(set(a) | set(b))
        for field_name in fields:
            if a.get(field_name) != b.get(field_name):
                diffs.append(
                    f"{name}.{field_name}: rust={a.get(field_name)!r} python={b.get(field_name)!r}"
                )
    return diffs


def run_parity() -> dict[str, Any]:
    """Run both paths over this repo and compare capability maps."""
    binary = _find_hawking_index_bin()
    if binary is None:
        return {
            "status": "BINARY_MISSING",
            "compared": 0,
            "diffs": ["hawking-index binary not found; cannot prove rust/python parity"],
        }
    saved = _os.environ.pop("HAWKING_REACHABILITY_FORCE_PYTHON", None)
    try:
        _TEXT_CACHE.clear()
        rust_doc = assemble()
        if rust_doc.get("facts_source") != "hawking-index":
            return {
                "status": "RUST_PATH_UNUSED",
                "compared": 0,
                "diffs": [
                    "binary present but assemble() did not use hawking-index "
                    f"(facts_source={rust_doc.get('facts_source')!r})"
                ],
            }
        _TEXT_CACHE.clear()
        _os.environ["HAWKING_REACHABILITY_FORCE_PYTHON"] = "1"
        py_doc = assemble()
    finally:
        if saved is None:
            _os.environ.pop("HAWKING_REACHABILITY_FORCE_PYTHON", None)
        else:
            _os.environ["HAWKING_REACHABILITY_FORCE_PYTHON"] = saved
    rust_caps = rust_doc.get("capabilities") or {}
    py_caps = py_doc.get("capabilities") or {}
    diffs = compare_capability_maps(rust_caps, py_caps)
    return {
        "status": "IDENTICAL" if not diffs else "DIFF",
        "compared": len(py_caps),
        "rust_count": len(rust_caps),
        "python_count": len(py_caps),
        "key_sets_equal": set(rust_caps) == set(py_caps),
        "rust_facts_source": rust_doc.get("facts_source"),
        "python_facts_source": py_doc.get("facts_source"),
        "diffs": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive who calls each capability; emit HCLI_CAPABILITY_REACHABILITY.json")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parity", action="store_true")
    require_known_flags({"--build", "--audit", "--selftest", "--parity"})
    args = ap.parse_args()
    if args.parity:
        report = run_parity()
        print(json.dumps(
            {k: (v[:30] if k == "diffs" and isinstance(v, list) else v) for k, v in report.items()},
            indent=2,
            sort_keys=True,
        ))
        n = report.get("compared") or 0
        print(f"parity={report['status']} compared={n} diffs={len(report.get('diffs') or [])}")
        if report.get("diffs"):
            for d in (report["diffs"] or [])[:40]:
                print("  DIFF", d)
        return 0 if report["status"] == "IDENTICAL" else 1
    out = selftest() if args.selftest else build()
    doc = load_json(out)
    counts = doc.get("counts") or {}
    print(out)
    print(
        "capabilities={cap} typed_tools_dead={ttd}/{tt} sidecar_dead={sd}/{sm} dead_surface={ds} facts_source={fs}".format(
            cap=counts.get("capabilities"),
            ttd=counts.get("typed_tools_dead"),
            tt=counts.get("typed_tools"),
            sd=counts.get("sidecar_modules", 0) - counts.get("sidecar_callable", 0),
            sm=counts.get("sidecar_modules"),
            ds=counts.get("dead_surface"),
            fs=doc.get("facts_source"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
