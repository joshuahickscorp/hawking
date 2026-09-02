"""Reachability triage -- one disposition per module, from call sites.

Hawking's known worst defect class is capability that exists but nothing
calls. This tool is the inventory: every non-test Python module under
tools/future/, tools/accelerator/, tools/odyssey/, and tools/headless/ is
classified and given a disposition (CONNECTED / PARKED / ARCHIVE_CANDIDATE)
from grep/AST evidence only.

Engine
------
The analyzer is tools/future/capability_reachability.py. This module does
not write a second one. It calls assemble() and the RepoIndex primitives
(find_module_import_sites, _subprocess_path_sites, build_capability,
_partition). Two holes in this sparse worktree made the engine insufficient;
both are patched in-process before any index is built:

  1. Sparse-checkout git-blob fallback. git ls-files names hcli/*.py but
     they are not on disk here; Path.read_text was returning empty and the
     import index silently dropped every HCLI call site. Blobs are loaded
     from HEAD via one `git cat-file --batch`.
  2. receipts/fixtures/goldens are data, not callers. An import inside a
     generated receipt runner is not evidence the resident reaches it.

    python3 tools/audit/reachability_triage.py --build
    python3 tools/audit/reachability_triage.py --selftest
    python3 tools/audit/reachability_triage.py --discover
    python3 tools/audit/reachability_triage.py --invoke future.capacity_inference_rule --args '{"levels":[{"concurrency":1,"aggregate_decode_tps":36.6},{"concurrency":2,"aggregate_decode_tps":51.2}],"semantics_comparable":true}'
    python3 tools/audit/reachability_triage.py --module tools/future/status_causality.py
    python3 tools/audit/reachability_triage.py --classification UNREACHABLE
    python3 tools/audit/reachability_triage.py --status BUILT
    python3 tools/audit/reachability_triage.py --parity
    python3 tools/audit/reachability_triage.py --measure
    python3 -m pytest tools/audit/test_reachability_triage.py tools/audit/test_artifact_query.py -q -o addopts=""
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import (
    GIT_TIMEOUT_S,
    REPO,
    git,
    load_json,
    require_known_flags,
    write_receipt,
)
from tools.future import capability_reachability as cr

import argparse
import json
import ast
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


RECEIPT = "REACHABILITY_TRIAGE.json"
SCHEMA = "hawking.audit.reachability_triage.v1"
VERSION = 1
RECORDED_BY = "tools/audit/reachability_triage.py"

TREE_ROOTS = ("tools/future", "tools/accelerator", "tools/odyssey", "tools/headless")

CLASSIFICATIONS = (
    "BUILT",
    "SCAFFOLDED",
    "DORMANT",
    "UNREACHABLE",
    "ARCHIVE_CANDIDATE",
)
DISPOSITIONS = ("CONNECTED", "PARKED", "ARCHIVE_CANDIDATE")

WAKE_SCHEMA = "hawking.audit.wake_condition.v1"
ARCHIVE_SCHEMA = "hawking.audit.archive_reason.v1"
# A wake is only satisfied by an AST Call of a named symbol. An import of the
# module is not a call site; the previous inventory's citations were
# import-dominated (import 1329 vs call 2) and this field exists so nothing
# downstream inherits that weakness.
WAKE_REQUIRED_KIND = "call"
WAKE_KIND_HCLI_SYMBOL_CALL = "HCLI_SYMBOL_CALL"
WAKE_KIND_ORCHESTRATION_INVOKE = "ORCHESTRATION_INVOKE"
WAKE_KIND_PRODUCTION_SYMBOL_CALL = "PRODUCTION_SYMBOL_CALL"
ARCHIVE_KIND_RETIRED = "RETIRED"
ARCHIVE_KIND_STUB_UNTESTED = "STUB_UNTESTED"

HCLI_ENTRY_MODULES = (
    "hcli",
    "hcli.__main__",
    "hcli.commands",
    "hcli.command_registry",
    "hcli.tool_registry",
    "hcli.mission",
    "hcli.frontier_scheduler",
    "hcli.agentos",
    "hcli.agentos.resident",
    "hcli.agentos.runtime",
    "hcli.engine",
    "hcli.executors",
)

_DATA_DIR_NAMES = frozenset({"receipts", "fixtures", "goldens"})
# Only the MODULE retiring itself, not a docstring that mentions some other
# campaign/driver/path being retired. Matching bare "retired" put index_provenance
# (legacy driver was retired) and ramanujan_disband (the campaign is retired)
# in ARCHIVE_CANDIDATE, which is a classification error.
_RETIRED_RE = re.compile(
    r"(?i)(?:this module (?:is |has been )?(?:retired|superseded|deprecated)|"
    r"do not (?:use|import) this module|"
    r"superseded by tools/)"
)
_WAKE_CONST_RE = re.compile(r"^WAKE_[A-Z0-9_]+$")


# --------------------------------------------------------------------------
# Engine extensions (in-process; the analyzer file is the engine)
# --------------------------------------------------------------------------


_EXTENSIONS_INSTALLED = False
_ORIG_BUILD_REPO_INDEX = cr.build_repo_index
_ORIG_PARTITION = cr._partition
_INDEX_CACHE: dict[Any, cr.RepoIndex] = {}


def is_data_path(path: Path) -> bool:
    return any(part in _DATA_DIR_NAMES for part in Path(path).parts)


def is_production_path(path: Path) -> bool:
    p = Path(path)
    return (not cr.is_test_path(p)) and (not is_data_path(p))


def _prefetch_git_blobs(paths: Sequence[Path]) -> None:
    """Load sparse-missing files from HEAD. One cat-file --batch, not N shows."""
    if not paths:
        return
    specs = [f"HEAD:{cr.rel(p)}" for p in paths]
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
            cr._TEXT_CACHE.setdefault(p, "")
        return
    data = proc.stdout
    idx = 0
    for path in paths:
        if idx >= len(data):
            cr._TEXT_CACHE[path] = ""
            continue
        nl = data.find(b"\n", idx)
        if nl < 0:
            cr._TEXT_CACHE[path] = ""
            break
        header = data[idx:nl].decode("utf-8", errors="replace")
        idx = nl + 1
        if " missing" in header:
            cr._TEXT_CACHE[path] = ""
            continue
        parts = header.split()
        if len(parts) < 3 or parts[1] != "blob":
            cr._TEXT_CACHE[path] = ""
            continue
        try:
            size = int(parts[2])
        except ValueError:
            cr._TEXT_CACHE[path] = ""
            continue
        blob = data[idx : idx + size]
        idx = idx + size
        if idx < len(data) and data[idx : idx + 1] == b"\n":
            idx += 1
        cr._TEXT_CACHE[path] = blob.decode("utf-8", errors="replace")


def prefetch_texts(files: Sequence[Path]) -> None:
    missing: list[Path] = []
    for path in files:
        if path in cr._TEXT_CACHE:
            continue
        if path.is_file():
            try:
                cr._TEXT_CACHE[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                cr._TEXT_CACHE[path] = ""
        else:
            missing.append(path)
    _prefetch_git_blobs(missing)


def _extended_read_text(path: Path) -> str:
    if path not in cr._TEXT_CACHE:
        prefetch_texts([path])
    return cr._TEXT_CACHE.get(path, "")


def _extended_partition(sites: Sequence[cr.Site]) -> tuple[list[cr.Site], list[cr.Site]]:
    """Production vs test, with receipts/fixtures/goldens treated as neither.

    A generated runner under receipts/ that imports a sidecar module is data,
    not a production call site. Tests still count toward `tested`.
    """
    seen: set[tuple[str, int, str]] = set()
    uniq: list[cr.Site] = []
    for s in sites:
        key = (s.file, s.line, s.kind)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    uniq.sort(key=lambda s: (s.file, s.line))
    prod = [s for s in uniq if is_production_path(Path(s.file))]
    test = [s for s in uniq if cr.is_test_path(Path(s.file))]
    return prod, test


def _extended_build_repo_index(files: Sequence[Path] | None = None) -> cr.RepoIndex:
    key: Any
    if files is None:
        key = None
    else:
        key = tuple(cr.rel(p) for p in files)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    file_list = list(files) if files is not None else cr.repo_py_files()
    prefetch_texts(file_list)
    idx = _ORIG_BUILD_REPO_INDEX(files=file_list)
    _INDEX_CACHE[key] = idx
    return idx


def install_engine_extensions() -> list[str]:
    """Patch the live analyzer so sparse-missing files and data paths are honest.

    Idempotent. Does not weaken any check: more files become visible, and
    receipts lose their false 'caller' status.
    """
    global _EXTENSIONS_INSTALLED
    notes = [
        "sparse-checkout git-blob fallback on read_text / build_repo_index "
        "(HEAD:path via git cat-file --batch for files git-tracked but not on disk)",
        "is_data_path: receipts/fixtures/goldens excluded from production call sites",
        "build_repo_index result cached for assemble()+triage sharing",
    ]
    if _EXTENSIONS_INSTALLED:
        return notes
    cr.read_text = _extended_read_text
    cr._partition = _extended_partition
    cr.build_repo_index = _extended_build_repo_index
    cr.is_data_path = is_data_path  # type: ignore[attr-defined]
    cr.is_production_path = is_production_path  # type: ignore[attr-defined]
    _EXTENSIONS_INSTALLED = True
    return notes


# --------------------------------------------------------------------------
# Tree discovery and per-module evidence
# --------------------------------------------------------------------------


def discover_tree_modules(tree_rel: str) -> list[Path]:
    """Every git-tracked non-test .py under tree_rel. Files may be sparse-missing."""
    listed = git("ls-files", "--", tree_rel)
    out: list[Path] = []
    for line in listed.splitlines():
        if not line.endswith(".py"):
            continue
        if "__pycache__" in line:
            continue
        path = REPO / line
        if cr.is_test_path(path):
            continue
        out.append(path)
    return sorted(out, key=lambda p: cr.rel(p))


def module_summary(text: str) -> str:
    if not text.strip():
        return "empty module (package marker or stub)"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "(unparseable)"
    doc = ast.get_docstring(tree)
    if doc:
        first = doc.strip().splitlines()[0].strip()
        return first[:200]
    return "(no module docstring)"


def module_shape(text: str, filename: str) -> dict[str, Any]:
    """Stub / package-marker / retired flags, derived from the module AST."""
    if not text.strip():
        return {
            "is_stub": filename.endswith("__init__.py"),
            "is_package_marker": filename.endswith("__init__.py"),
            "retired": None,
            "has_main": False,
            "n_functions": 0,
            "n_classes": 0,
            "produces_receipt": False,
            "receipt_names": [],
            "public_functions": [],
            "public_classes": [],
        }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "is_stub": True,
            "is_package_marker": False,
            "retired": None,
            "has_main": False,
            "n_functions": 0,
            "n_classes": 0,
            "produces_receipt": False,
            "receipt_names": [],
            "public_functions": [],
            "public_classes": [],
        }
    doc = ast.get_docstring(tree) or ""
    retired_hit = _RETIRED_RE.search(doc)
    funcs = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    significant = [
        n
        for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(n, ast.Expr)
            and isinstance(getattr(n, "value", None), ast.Constant)
        )
    ]
    is_package_marker = filename.endswith("__init__.py") and not funcs and not classes
    source_lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    has_notimpl = "NotImplementedError" in text
    is_stub = bool(
        is_package_marker
        or (
            len(funcs) + len(classes) <= 1
            and len(source_lines) < 30
            and (has_notimpl or len(significant) <= 2)
        )
    )
    has_main = False
    for n in tree.body:
        if isinstance(n, ast.If):
            test = n.test
            # if __name__ == "__main__":
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and test.comparators
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                has_main = True
    receipt_names: list[str] = []
    produces = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "RECEIPT":
                    if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                        receipt_names.append(n.value.value)
                        produces = True
        if isinstance(n, ast.Call):
            f = n.func
            fname = ""
            if isinstance(f, ast.Name):
                fname = f.id
            elif isinstance(f, ast.Attribute):
                fname = f.attr
            if fname in {"write_receipt", "write_measured_receipt"}:
                produces = True
                if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    receipt_names.append(n.args[0].value)
    # unique, stable
    seen: set[str] = set()
    uniq_receipts: list[str] = []
    for r in receipt_names:
        if r not in seen:
            seen.add(r)
            uniq_receipts.append(r)
    public_functions: list[dict[str, Any]] = []
    for n in funcs:
        if n.name.startswith("_"):
            continue
        args = [a.arg for a in n.args.args if a.arg not in {"self", "cls"}]
        public_functions.append({"name": n.name, "args": args})
    public_classes = [
        {"name": n.name} for n in classes if not n.name.startswith("_")
    ]
    return {
        "is_stub": is_stub,
        "is_package_marker": is_package_marker,
        "retired": retired_hit.group(0) if retired_hit else None,
        "has_main": has_main,
        "n_functions": len(funcs),
        "n_classes": len(classes),
        "produces_receipt": produces,
        "receipt_names": uniq_receipts,
        "public_functions": public_functions,
        "public_classes": public_classes,
    }


def recover_orchestration_bindings(idx: cr.RepoIndex) -> frozenset[str]:
    """Filenames listed in tools/future/orchestration.py BINDINGS.

    A BINDINGS row is registration, not a call. invoke() imports the module
    at runtime; without a static invoke("<file>") this does not make it
    callable. Recorded so PARKED wake conditions can name the existing hook.
    """
    path = REPO / "tools/future/orchestration.py"
    text = cr.read_text(path)
    if not text:
        return frozenset()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not (isinstance(target, ast.Name) and target.id == "BINDINGS"):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.add(key.value)
    # also a plain Assign, in case the annotation form is rewritten
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "BINDINGS" for t in node.targets):
            continue
        if isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
    return frozenset(names)


def build_dynamic_import_index(idx: cr.RepoIndex) -> dict[str, list[cr.Site]]:
    """importlib.import_module("tools.foo.bar") / __import__("tools.foo.bar") sites.

    One AST pass over the repo, not one pass per module.
    """
    out: dict[str, list[cr.Site]] = {}
    for path in idx.files:
        rp = cr.rel(path)
        if not is_production_path(Path(rp)):
            continue
        text = cr.read_text(path)
        if "import_module" not in text and "__import__" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            arg0 = node.args[0]
            if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
                continue
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name not in {"import_module", "__import__"}:
                continue
            out.setdefault(arg0.value, []).append(cr.Site(rp, node.lineno, "call"))
    return out


def build_import_graph(idx: cr.RepoIndex) -> dict[str, set[str]]:
    """importer_module -> {imported_module, ...} from production import sites only."""
    graph: dict[str, set[str]] = {}
    for target, sites in idx.import_sites.items():
        for site in sites:
            if not is_production_path(Path(site.file)):
                continue
            importer = cr.module_name_of(REPO / site.file)
            graph.setdefault(importer, set()).add(target)
    return graph


def hcli_reachable_set(
    idx: cr.RepoIndex,
    graph: dict[str, set[str]],
    subprocess_edges: Mapping[str, Iterable[str]],
) -> tuple[set[str], dict[str, str]]:
    """Modules reachable from any production file under hcli/.

    Returns (reachable_modules, parent_map) where parent_map[m] is the
    importer that first reached m (for a shortest-path citation).
    """
    start: set[str] = set()
    parent: dict[str, str] = {}
    for path in idx.files:
        rp = cr.rel(path)
        if not rp.startswith("hcli/"):
            continue
        if not is_production_path(Path(rp)):
            continue
        start.add(cr.module_name_of(path))
    seen: set[str] = set()
    q: deque[str] = deque(sorted(start))
    for s in start:
        parent.setdefault(s, "")
    while q:
        m = q.popleft()
        if m in seen:
            continue
        seen.add(m)
        for dest in graph.get(m, ()):
            if dest not in seen and dest not in parent:
                parent[dest] = m
                q.append(dest)
        for dest in subprocess_edges.get(m, ()):
            if dest not in seen and dest not in parent:
                parent[dest] = m
                q.append(dest)
    return seen, parent


def hcli_invocations(
    symbol_call_sites: Sequence[Mapping[str, Any]],
    call_sites: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """HCLI production Call/subprocess of an exported symbol. Imports are not hits.

    kind=import never justifies CONNECTED. The previous inventory cited
    import-dominated callers (import 1329 vs call 2); this is the same
    rule tools/roadmap/auditor.py applies (BUILT_KINDS = call, subprocess).
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for s in list(symbol_call_sites or []):
        if not str(s.get("file", "")).startswith("hcli/"):
            continue
        if s.get("kind") not in {"call", "subprocess"}:
            continue
        key = (str(s.get("file")), int(s.get("line") or 0), str(s.get("kind")))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(s))
    for s in list(call_sites or []):
        if not str(s.get("file", "")).startswith("hcli/"):
            continue
        if s.get("kind") != "subprocess":
            continue
        key = (str(s.get("file")), int(s.get("line") or 0), str(s.get("kind")))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(s))
    out.sort(key=lambda s: (str(s.get("file")), int(s.get("line") or 0)))
    return out


def cite_hcli_path(
    module_dotted: str,
    call_sites: Sequence[Mapping[str, Any]],
    parent: Mapping[str, str],
    invocations: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    """Cite an HCLI *invocation*. An import is not a path to CONNECTED."""
    if invocations:
        s = invocations[0]
        return f"{s['file']}:{s['line']} ({s.get('kind', 'call')})"
    # Import-graph chains are recorded on hcli_import_path, never here.
    # Returning None for import-only evidence is the point of the rule.
    return None


def cite_hcli_import_path(
    module_dotted: str,
    call_sites: Sequence[Mapping[str, Any]],
    parent: Mapping[str, str],
) -> str | None:
    """Informational import-graph citation. Never CONNECTED evidence."""
    hcli_sites = [s for s in call_sites if str(s.get("file", "")).startswith("hcli/")]
    if hcli_sites:
        s = hcli_sites[0]
        return f"{s['file']}:{s['line']} ({s.get('kind', 'import')})"
    if module_dotted not in parent:
        return None
    chain: list[str] = [module_dotted]
    cur = module_dotted
    guard = 0
    while cur in parent and parent[cur] and guard < 32:
        cur = parent[cur]
        chain.append(cur)
        if cur.startswith("hcli"):
            break
        guard += 1
    chain.reverse()
    if not chain or not chain[0].startswith("hcli"):
        return None
    return " -> ".join(chain)


def conventional_test_path(module_path: Path) -> Path:
    return module_path.parent / f"test_{module_path.stem}.py"


def analyze_test(
    module_dotted: str,
    module_path: Path,
    idx: cr.RepoIndex,
    test_sites: Sequence[cr.Site],
) -> dict[str, Any]:
    """Does a test exist, and does it exercise the module or only read a receipt?"""
    files: set[str] = {s.file for s in test_sites}
    conv = conventional_test_path(module_path)
    conv_rel = cr.rel(conv)
    conv_text = cr.read_text(conv)
    if conv_text:
        files.add(conv_rel)
    if not files:
        return {
            "has_test": False,
            "test_files": [],
            "test_exercises": False,
            "test_reads_receipt_only": False,
        }
    exercises = False
    receipt_only = False
    for rel_file in sorted(files):
        text = cr.read_text(REPO / rel_file)
        if not text:
            continue
        kind = _test_file_kind(text, module_dotted, module_path.stem)
        if kind == "exercises":
            exercises = True
        elif kind == "receipt_only":
            receipt_only = True
    return {
        "has_test": True,
        "test_files": sorted(files),
        "test_exercises": exercises,
        "test_reads_receipt_only": bool(receipt_only and not exercises),
    }


def _test_file_kind(text: str, module_dotted: str, stem: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        if ".json" in text:
            return "receipt_only"
        return "other"
    binds_module: set[str] = set()
    imports_module = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_dotted or alias.name.startswith(module_dotted + "."):
                    imports_module = True
                    binds_module.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == module_dotted or mod.startswith(module_dotted + "."):
                imports_module = True
                for alias in node.names:
                    binds_module.add(alias.asname or alias.name)
            # from tools.future import foo as f
            if mod and module_dotted.startswith(mod + "."):
                child = module_dotted[len(mod) + 1 :]
                for alias in node.names:
                    if alias.name == child.split(".")[0]:
                        imports_module = True
                        binds_module.add(alias.asname or alias.name)
    called = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in binds_module:
            called = True
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in binds_module:
                called = True
            elif func.attr in binds_module:
                called = True
    if imports_module and called:
        return "exercises"
    if imports_module:
        # imported RECEIPT / SCHEMA / a constant -- not an exercise of behavior
        return "imports_only"
    if "load_json" in text or "json.loads" in text or ".json" in text:
        return "receipt_only"
    return "other"


# --------------------------------------------------------------------------
# Classification / disposition -- every path assigns both
# --------------------------------------------------------------------------


def _primary_symbol(row: Mapping[str, Any]) -> str | None:
    """Best public symbol to require a Call of. Prefer a known entry, else first."""
    preferred = (
        "selftest",
        "build",
        "fire",
        "evaluate",
        "inspect",
        "validate",
        "query",
        "scan",
        "probe",
        "classify",
        "may_refuse",
        "fires_on",
        "can_promote",
        "emit",
        "challenge",
    )
    names = [p["name"] for p in (row.get("public_functions") or []) if p.get("name")]
    for cand in preferred:
        if cand in names:
            dotted = str(row.get("dotted") or "")
            return f"{dotted}.{cand}" if dotted else cand
    if names:
        dotted = str(row.get("dotted") or "")
        return f"{dotted}.{names[0]}" if dotted else names[0]
    classes = [c["name"] for c in (row.get("public_classes") or []) if c.get("name")]
    if classes:
        dotted = str(row.get("dotted") or "")
        return f"{dotted}.{classes[0]}" if dotted else classes[0]
    dotted = str(row.get("dotted") or "")
    return dotted or None


def structured_wake(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Machine-readable wake for a PARKED row. None if the row is not parked.

    required_kind is always 'call'. Satisfying this with an import is a bug.
    """
    disp = row.get("disposition")
    if disp != "PARKED":
        # decide() sets disposition first; callers may also pass pre-decide flags.
        pass
    callable_ = bool(row.get("callable_outside_tests"))
    hcli = bool(row.get("hcli_reachable"))
    if hcli:
        return None
    bound = bool(row.get("orchestration_bound"))
    fn = Path(str(row.get("module", ""))).name
    symbol = _primary_symbol(row)
    adapter_channel = {
        "channel": "adapter",
        "where": "tools/audit/capability_manifest.py",
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": symbol,
        "how": (
            "add a WIRED entry and an AST Call of the named symbol in "
            "tools/audit/capability_manifest.py; an import is not enough"
        ),
    }
    hcli_channel = {
        "channel": "hcli",
        "where": "hcli/",
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": symbol,
        "how": (
            "a production file under hcli/ must contain an AST Call of an "
            "exported symbol of this module (not an Import/ImportFrom)"
        ),
    }
    if callable_ and bound:
        kind = WAKE_KIND_ORCHESTRATION_INVOKE
        predicate = (
            "production AST Call of tools.future.orchestration.invoke with "
            f"this module's filename {fn!r}, from an HCLI entry (CLI verb, "
            "tool registry, mission executor, or resident)"
        )
        blocker = (
            f"{fn} is listed in tools/future/orchestration.py BINDINGS but "
            "orchestration.invoke is not statically reached from HCLI; "
            "sidecar imports are not a call of invoke"
        )
        extra_symbol = "tools.future.orchestration.invoke"
    elif callable_:
        kind = WAKE_KIND_HCLI_SYMBOL_CALL
        predicate = (
            "production AST Call of an exported symbol of this module from an "
            "HCLI entry (CLI verb, tool registry, mission executor, or resident)"
        )
        if row.get("hcli_imported"):
            blocker = (
                "imported by HCLI but no AST Call of an exported symbol "
                "(an import is not a call)"
            )
        else:
            blocker = (
                "currently only imported inside the sidecar cluster; no HCLI "
                "Call node of an exported symbol"
            )
        extra_symbol = symbol
    else:
        kind = WAKE_KIND_PRODUCTION_SYMBOL_CALL
        predicate = (
            "first production AST Call of an exported symbol of this module "
            "(a test file does not count; an import does not count)"
        )
        if row.get("test_exercises"):
            blocker = (
                "tests exercise the module but nothing outside tests calls "
                "an exported symbol"
            )
        elif row.get("has_test") and row.get("test_reads_receipt_only"):
            blocker = (
                "a test file exists but it reads a checked-in receipt rather "
                "than calling an exported symbol"
            )
        elif row.get("has_test"):
            blocker = (
                "a test file imports the module but does not call an exported "
                "symbol"
            )
        elif bound:
            blocker = (
                f"listed in BINDINGS as {fn}; uncalled and untested; "
                "registration is not a call"
            )
        else:
            blocker = "currently uncalled and untested"
        extra_symbol = symbol
    return {
        "schema": WAKE_SCHEMA,
        "kind": kind,
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": extra_symbol,
        "required_caller_prefix": "hcli/",
        "predicate": predicate,
        "blocker": blocker,
        "orchestration_module": fn if bound else None,
        "satisfy_by": [hcli_channel, adapter_channel],
        "evidence_tier": "STATIC",
    }


def structured_archive(row: Mapping[str, Any]) -> dict[str, Any] | None:
    reason = row.get("archive_reason")
    if not reason:
        return None
    kind = ARCHIVE_KIND_RETIRED if row.get("retired") else ARCHIVE_KIND_STUB_UNTESTED
    return {
        "schema": ARCHIVE_SCHEMA,
        "kind": kind,
        "reason": reason,
        "deleted": False,
        "evidence_tier": "STATIC",
    }


def _stamp_structured(row: dict[str, Any]) -> None:
    """Attach machine-readable wake/archive. Never deletes a module."""
    disp = row.get("disposition")
    if disp == "PARKED":
        row["wake"] = structured_wake(row)
        row["archive"] = None
    elif disp == "ARCHIVE_CANDIDATE":
        row["wake"] = None
        row["archive"] = structured_archive(row)
    else:
        row["wake"] = None
        row["archive"] = None


def decide(row: dict[str, Any]) -> None:
    """Mutates row with classification, disposition, wake_condition/archive_reason.

    Never leaves disposition empty. Evidence-only; no hand roster.
    """
    callable_ = bool(row.get("callable_outside_tests"))
    hcli = bool(row.get("hcli_reachable"))
    stub = bool(row.get("is_stub"))
    pkg = bool(row.get("is_package_marker"))
    retired = row.get("retired")
    tested = bool(row.get("has_test"))
    exercises = bool(row.get("test_exercises"))
    bound = bool(row.get("orchestration_bound"))
    fn = Path(str(row.get("module", ""))).name

    row["wake_condition"] = None
    row["archive_reason"] = None
    row["wake"] = None
    row["archive"] = None

    if retired and not callable_ and not hcli:
        row["classification"] = "ARCHIVE_CANDIDATE"
        row["disposition"] = "ARCHIVE_CANDIDATE"
        row["archive_reason"] = (
            f"module docstring marks it {retired!r} and nothing outside tests calls it"
        )
        row["disposition_full"] = f"ARCHIVE_CANDIDATE({row['archive_reason']})"
        _stamp_structured(row)
        return

    if hcli:
        row["classification"] = "SCAFFOLDED" if (stub and not pkg) else "BUILT"
        row["disposition"] = "CONNECTED"
        row["disposition_full"] = "CONNECTED"
        _stamp_structured(row)
        return

    if callable_:
        if stub and not pkg:
            row["classification"] = "SCAFFOLDED"
        else:
            row["classification"] = "DORMANT"
        if bound:
            wake = (
                f"production tools.future.orchestration.invoke({fn!r}) from an HCLI "
                "entry (CLI verb, tool registry, mission executor, or resident); "
                "listed in BINDINGS but invoke() is not statically reached from HCLI"
            )
        elif row.get("hcli_imported"):
            wake = (
                "production AST Call of an exported symbol from an HCLI entry "
                "(CLI verb, tool registry, mission executor, or resident); "
                "currently imported by HCLI but never called (an import is not a call)"
            )
        else:
            wake = (
                "first HCLI entry-point Call of an exported symbol (CLI verb, "
                "tool registry, mission executor, or resident); currently only "
                "called inside the sidecar cluster, not from HCLI"
            )
        row["disposition"] = "PARKED"
        row["wake_condition"] = wake
        row["disposition_full"] = f"PARKED({wake})"
        _stamp_structured(row)
        return

    if stub and not tested and not pkg:
        row["classification"] = "SCAFFOLDED"
        row["disposition"] = "ARCHIVE_CANDIDATE"
        row["archive_reason"] = "scaffold/stub with no production callers and no tests"
        row["disposition_full"] = f"ARCHIVE_CANDIDATE({row['archive_reason']})"
        _stamp_structured(row)
        return

    row["classification"] = "UNREACHABLE"
    row["disposition"] = "PARKED"
    if exercises:
        wake = (
            "first production importer; tests exercise the module but nothing "
            "outside tests calls it"
        )
    elif tested and row.get("test_reads_receipt_only"):
        wake = (
            "first production importer; a test file exists but it reads a "
            "checked-in receipt rather than exercising the module"
        )
    elif tested:
        wake = (
            "first production importer; a test file imports the module but "
            "does not call it"
        )
    elif bound:
        wake = (
            f"first production importer or orchestration.invoke({fn!r}); "
            "listed in BINDINGS, uncalled, untested"
        )
    else:
        wake = "first production importer; currently uncalled and untested"
    row["wake_condition"] = wake
    row["disposition_full"] = f"PARKED({wake})"
    _stamp_structured(row)


# --------------------------------------------------------------------------
# State-transition sweep
# --------------------------------------------------------------------------


def _wake_constants_in(text: str) -> list[tuple[str, int, str]]:
    """(name, line, value) for WAKE_* = "..." assignments."""
    out: list[tuple[str, int, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and _WAKE_CONST_RE.match(t.id):
                out.append((t.id, node.lineno, node.value.value))
    return out


def state_transition_sweep(
    idx: cr.RepoIndex,
    tree_files: Sequence[Path],
) -> list[dict[str, Any]]:
    """States that can be entered with no real exit. Report only; do not fix.

    Two layers: verified findings from named, known classes (complete
    artifact / resident loop / spawned child / sleeping WorkUnit), plus a
    mechanical scan for WAKE_* constants that nothing outside their
    defining file ever names.
    """
    findings: list[dict[str, Any]] = []

    # --- known class: complete model/artifact with no promotion path ------
    findings.append(
        {
            "class": "complete_artifact_without_promotion",
            "file": "tools/odyssey/modellake_watch.py",
            "enter": "tools/odyssey/modellake_watch.py:502 complete() returns True",
            "exit": (
                "tools/odyssey/modellake_watch.py:1086 and :1125 call "
                "_promote_and_report -> promote_if_needed -> "
                "modellake_promote.promote(tag, go=True); "
                "reconcile() at :660 is the second-look path"
            ),
            "status": "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "The original defect (complete() then continue, SPECIMEN_ROOT "
                "never written) is described in tools/odyssey/modellake_promote.py:1-9. "
                "Current HEAD has an exit on both the live admission loop and "
                "the periodic reconcile sweep. Not missing anymore; recorded "
                "so the class is not silently dropped."
            ),
        }
    )

    # --- known class: sleeping WorkUnit with no wake event ----------------
    sealed_refs: list[str] = []
    for path in idx.files:
        rp = cr.rel(path)
        text = cr.read_text(path)
        if "SEALED_SOURCE_READY" in text:
            sealed_refs.append(rp)
    outside = [
        r
        for r in sealed_refs
        if r != "tools/future/sleeping_specimens.py"
        and not cr.is_test_path(Path(r))
        and not is_data_path(Path(r))
        and not r.startswith("tools/audit/")
    ]
    findings.append(
        {
            "class": "sleeping_workunit_without_wake_event",
            "file": "tools/future/sleeping_specimens.py",
            "enter": (
                "tools/future/sleeping_specimens.py:426 "
                "wake_condition=SEALED_SOURCE_READY on each SLEEPING_SPECIMEN_WU; "
                "also :468 row['wake_condition'] = WAKE_SEALED_SOURCE_READY"
            ),
            "exit": (
                None
                if not outside
                else (
                    "tools/future/sleeping_specimens.py:479 sealed_source_ready(tag) "
                    "is True iff SPECIMEN_ROOT/tag is a non-empty directory; "
                    ":501 notify_sealed_source_ready is the event. "
                    "tools/odyssey/modellake_watch.py:525 _notify_sealed_source "
                    "calls notify_sealed_source_ready + "
                    "wakeup.harvest_sealed_specimens after PROMOTED/"
                    "ALREADY_PROMOTED (:586 _promote_and_report, :696 reconcile). "
                    "tools/future/wakeup.py:786 harvest_sealed_specimens "
                    "classifies COMPLETED only when disk confirms; never a "
                    "synthetic COMPLETED. Named in: " + ", ".join(outside)
                )
            ),
            "status": "EXIT_MISSING" if not outside else "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "Disk is the wake event. modellake_watch fires the token after "
                "a successful promote; wakeup.harvest_sealed_specimens is the "
                "consumer. The triage engine naming the token is not a consumer "
                "(tools/audit/ excluded). Test files and receipts/ excluded by "
                "the same rule as call sites."
                if outside
                else (
                    "git grep SEALED_SOURCE_READY on HEAD: every production hit "
                    "is inside sleeping_specimens.py itself (definition, emit). "
                    "A SLEEPING_SPECIMEN_WU can be entered and never leave."
                )
            ),
            "defining_file_refs": sealed_refs,
            "outside_production_refs": outside,
        }
    )

    findings.append(
        {
            "class": "sleeping_workunit_without_wake_event",
            "file": "tools/future/wakeup.py",
            "enter": (
                "tools/future/wakeup.py:398 register(..., sleeping=True) "
                "sets state=SLEEPING; emit_wakeup_workunits :861 wakeup_state=SLEEPING"
            ),
            "exit": (
                "tools/future/wakeup.py:415 harvest() -> _inspect :597-626: a "
                "sealed receipt at the expected path classifies COMPLETED and "
                "harvest writes exp['state']=event.state. Selftest :1247 "
                "proves SLEEPING without a receipt stays SLEEPING (no synthetic "
                "COMPLETED). Distinct SEALED_SOURCE_READY exit: "
                "harvest_sealed_specimens :786."
            ),
            "status": "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "The exit is a disk receipt, not a caller. That is the module's "
                "contract (completion wakes the graph). Distinct from "
                "SLEEPING_SPECIMEN_WU, whose wake token is SEALED_SOURCE_READY "
                "and is consumed by harvest_sealed_specimens."
            ),
        }
    )

    # --- known class: spawned child with no landing path back -------------
    findings.append(
        {
            "class": "spawned_child_without_landing_path",
            "file": "hcli/agentos/resident.py",
            "enter": (
                "hcli/agentos/resident.py:825 launch_child; :856 "
                "store.update(child_job_ids=children[-64:], last_event='child_started')"
            ),
            "exit": (
                "Process-level: hcli/agentos/background.py BackgroundJobStore._refresh "
                "sets finished_at when the child PID dies (:172-197) and "
                "_supervise_job sets finished_at at :399. Resident-level: no "
                "child_finished / child_landed / last_event='child_*' other than "
                "child_started in resident.py. child_job_ids only appends "
                "(capped at 64) and is never reaped back into the resident "
                "state machine."
            ),
            "status": "EXIT_MISSING",
            "evidence_tier": "STATIC",
            "note": (
                "The child process can finish in the job store. The resident "
                "state machine never consumes that finish. That is the missing "
                "landing path this class names. Do not confuse the two layers."
            ),
        }
    )

    # --- known class: resident loop that can spin without progressing -----
    findings.append(
        {
            "class": "resident_loop_spin_without_progress",
            "file": "hcli/agentos/resident.py",
            "enter": (
                "hcli/agentos/resident.py:1188 WAIT_FOR_CLEAN_ROOM -> "
                "state=WAITING_FOR_CLEAN_ROOM; :1201 WAIT_FOR_MEMORY evacuates; "
                "heartbeat loop :1234 self._wake.wait(config.interval_s)"
            ),
            "exit": (
                "WAITING_FOR_CLEAN_ROOM: last_event=clean_room_resumed at "
                "resident.py:813 (resume path exists). WAIT_FOR_MEMORY: "
                "_evacuate, then the next heartbeat re-evaluates memory and "
                "can spawn. IDLE at :1224 when no mission/inbox. FAILED at "
                ":1213 on restart limit. The loop itself always waits on "
                "_wake; progress depends on those exits actually being taken."
            ),
            "status": "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "Exits exist in the source. Whether WAITING_FOR_CLEAN_ROOM is "
                "ever resumed in a live daemon is a runtime question this "
                "STATIC sweep cannot settle. Recorded as EXIT_EXISTS because "
                "the transition is present, not because it was observed firing."
            ),
        }
    )

    # --- mechanical: WAKE_* constants with no outside production reference
    for path in tree_files:
        rel_p = cr.rel(path)
        text = cr.read_text(path)
        if not text:
            continue
        for name, lineno, value in _wake_constants_in(text):
            refs: list[str] = []
            for other in idx.files:
                rp = cr.rel(other)
                if rp == rel_p:
                    continue
                other_text = cr.read_text(other)
                if value in other_text or name in other_text:
                    if is_production_path(Path(rp)):
                        refs.append(f"{rp}")
            if refs:
                continue
            # skip the SEALED_SOURCE_READY finding already recorded above
            if value == "SEALED_SOURCE_READY":
                continue
            findings.append(
                {
                    "class": "sleeping_workunit_without_wake_event",
                    "file": rel_p,
                    "enter": f"{rel_p}:{lineno} {name} = {value!r}",
                    "exit": None,
                    "status": "EXIT_MISSING",
                    "evidence_tier": "STATIC",
                    "note": (
                        f"WAKE_* constant {name}={value!r} is never named in any "
                        "other production file. Entering a unit with this wake "
                        "token has no consumer that can fire it."
                    ),
                }
            )

    return findings


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_symbol_call_index(idx: cr.RepoIndex) -> dict[str, list[cr.Site]]:
    """One AST pass: production Call nodes resolved to dotted.symbol.

    An Import of a module is not a hit. Only `name(` and `module.attr(`
    where `name`/`module` were bound by an import. This is the citation
    kind the inventory was missing (import 1329 vs call 2).
    """
    out: dict[str, list[cr.Site]] = {}
    for path in idx.files:
        rp = cr.rel(path)
        if not is_production_path(Path(rp)):
            continue
        binds = idx.bound_names.get(rp, [])
        if not binds:
            continue
        local_to_full: dict[str, str] = {}
        for local, full in binds:
            local_to_full[local] = full
        text = cr.read_text(path)
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            target: str | None = None
            if isinstance(func, ast.Name) and func.id in local_to_full:
                target = local_to_full[func.id]
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in local_to_full
            ):
                target = f"{local_to_full[func.value.id]}.{func.attr}"
            if not target:
                continue
            out.setdefault(target, []).append(cr.Site(rp, node.lineno, "call"))
    return out


def _symbol_calls_for_module(
    dotted: str,
    shape: Mapping[str, Any],
    call_index: Mapping[str, list[cr.Site]],
    self_rel: str,
) -> list[dict[str, Any]]:
    names: list[str] = []
    names.extend(p["name"] for p in (shape.get("public_functions") or []) if p.get("name"))
    names.extend(c["name"] for c in (shape.get("public_classes") or []) if c.get("name"))
    sites: list[cr.Site] = []
    seen: set[tuple[str, int, str]] = set()
    for name in names:
        key = f"{dotted}.{name}"
        for s in call_index.get(key, ()):
            if s.file == self_rel:
                continue
            rec = (s.file, s.line, s.kind)
            if rec in seen:
                continue
            seen.add(rec)
            sites.append(s)
    sites.sort(key=lambda s: (s.file, s.line))
    return [s.to_dict() for s in sites]


def _subprocess_edges(idx: cr.RepoIndex, tree_files: Sequence[Path]) -> dict[str, set[str]]:
    """importer_module -> {launched_module} from real subprocess path sites."""
    edges: dict[str, set[str]] = {}
    for path in tree_files:
        rel_p = cr.rel(path)
        launched = cr.module_name_of(path)
        sites = cr._subprocess_path_sites(rel_p, idx.files, exclude_files=(path,))
        for s in sites:
            if not is_production_path(Path(s.file)):
                continue
            importer = cr.module_name_of(REPO / s.file)
            edges.setdefault(importer, set()).add(launched)
    return edges


def assemble_inventory() -> dict[str, Any]:
    extension_notes = install_engine_extensions()
    idx = cr.build_repo_index()
    engine_doc = cr.assemble()

    tree_files: list[Path] = []
    for root in TREE_ROOTS:
        tree_files.extend(discover_tree_modules(root))

    bindings = recover_orchestration_bindings(idx)
    dynamic_imports = build_dynamic_import_index(idx)
    graph = build_import_graph(idx)
    sub_edges = _subprocess_edges(idx, tree_files)
    reachable, parent = hcli_reachable_set(idx, graph, sub_edges)
    symbol_calls = build_symbol_call_index(idx)

    modules: dict[str, dict[str, Any]] = {}
    for path in tree_files:
        rel_p = cr.rel(path)
        dotted = cr.module_name_of(path)
        text = cr.read_text(path)
        shape = module_shape(text, path.name)
        sites = list(cr.find_module_import_sites(idx, dotted, exclude_files=(path,)))
        sites += cr._subprocess_path_sites(rel_p, idx.files, exclude_files=(path,))
        sites += list(dynamic_imports.get(dotted, ()))
        cap = cr.build_capability(
            dotted,
            "module",
            defined=True,
            registered=(path.name in bindings),
            resident_visible=None,
            sites=sites,
            definition={"file": rel_p, "line": None},
        )
        test_sites = [
            cr.Site(s["file"], s["line"], s["kind"]) for s in cap["test_only_sites"]
        ]
        test_info = analyze_test(dotted, path, idx, test_sites)

        symbol_call_sites = _symbol_calls_for_module(dotted, shape, symbol_calls, rel_p)
        invocations = hcli_invocations(symbol_call_sites, cap["call_sites"])
        # Import-graph reachability is informational only. CONNECTED requires
        # an HCLI AST Call/subprocess of an exported symbol — the auditor rule.
        hcli_imported = dotted in reachable or any(
            str(s["file"]).startswith("hcli/") for s in cap["call_sites"]
        )
        hcli = bool(invocations)
        import_n = sum(1 for s in cap["call_sites"] if s.get("kind") == "import")
        call_n = len(symbol_call_sites)
        has_subprocess = any(s.get("kind") == "subprocess" for s in cap["call_sites"])
        row: dict[str, Any] = {
            "module": rel_p,
            "dotted": dotted,
            "tree": next((t for t in TREE_ROOTS if rel_p.startswith(t + "/")), TREE_ROOTS[0]),
            "summary": module_summary(text),
            "callable_outside_tests": bool(cap["callable"]),
            "call_sites": cap["call_sites"],
            "test_only_sites": cap["test_only_sites"],
            "symbol_call_sites": symbol_call_sites,
            "hcli_invocations": invocations,
            "called_outside_tests": bool(symbol_call_sites) or has_subprocess,
            "import_only": bool(cap["callable"]) and not bool(symbol_call_sites) and not has_subprocess,
            "n_import_sites": import_n,
            "n_symbol_call_sites": call_n,
            "hcli_reachable": bool(hcli),
            "hcli_imported": bool(hcli_imported),
            "hcli_path": cite_hcli_path(
                dotted, cap["call_sites"], parent, invocations=invocations
            ),
            "hcli_import_path": cite_hcli_import_path(dotted, cap["call_sites"], parent),
            "orchestration_bound": path.name in bindings,
            "has_main": shape["has_main"],
            "produces_receipt": shape["produces_receipt"],
            "receipt_names": shape["receipt_names"],
            "receipt_producer_called": bool(shape["produces_receipt"] and cap["callable"]),
            "is_stub": shape["is_stub"],
            "is_package_marker": shape["is_package_marker"],
            "retired": shape["retired"],
            "n_functions": shape["n_functions"],
            "n_classes": shape["n_classes"],
            "public_functions": shape.get("public_functions") or [],
            "public_classes": shape.get("public_classes") or [],
            "evidence_tier": "STATIC",
            **test_info,
        }
        decide(row)
        modules[rel_p] = row

    undisposed = [
        m for m, r in modules.items() if r.get("disposition") not in DISPOSITIONS
    ]
    by_class: dict[str, int] = {c: 0 for c in CLASSIFICATIONS}
    by_disp: dict[str, int] = {d: 0 for d in DISPOSITIONS}
    by_tree: dict[str, int] = {t: 0 for t in TREE_ROOTS}
    for r in modules.values():
        by_class[r["classification"]] = by_class.get(r["classification"], 0) + 1
        by_disp[r["disposition"]] = by_disp.get(r["disposition"], 0) + 1
        by_tree[r["tree"]] = by_tree.get(r["tree"], 0) + 1

    transitions = state_transition_sweep(idx, tree_files)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Authoritative reachability inventory for every non-test Python "
            "module under tools/future, tools/accelerator, tools/odyssey, "
            "tools/headless. A definition is not a capability."
        ),
        "law": (
            "A capability nothing calls does not exist. Grep for call sites of "
            "the implementing symbol, not module imports, not definitions. "
            "kind=import never justifies CONNECTED. Own-test-only is not wired. "
            "receipts/ are data."
        ),
        "evidence_tier": "STATIC",
        "engine": {
            "analyzer": "tools/future/capability_reachability.py",
            "assemble_used": True,
            "extensions": extension_notes,
            "sidecar_counts": engine_doc.get("counts"),
        },
        "counts": {
            "modules": len(modules),
            "undispositioned": len(undisposed),
            "by_classification": by_class,
            "by_disposition": by_disp,
            "by_tree": by_tree,
            "hcli_reachable": sum(1 for r in modules.values() if r["hcli_reachable"]),
            "hcli_imported": sum(1 for r in modules.values() if r.get("hcli_imported")),
            "connected_import_only": sum(
                1
                for r in modules.values()
                if r.get("disposition") == "CONNECTED" and r.get("import_only")
            ),
            "callable_outside_tests": sum(
                1 for r in modules.values() if r["callable_outside_tests"]
            ),
            "orchestration_bound": sum(
                1 for r in modules.values() if r["orchestration_bound"]
            ),
            "produces_receipt": sum(1 for r in modules.values() if r["produces_receipt"]),
            "state_transition_findings": len(transitions),
            "state_transition_exits_missing": sum(
                1 for t in transitions if t.get("status") == "EXIT_MISSING"
            ),
            "import_sites": sum(int(r.get("n_import_sites") or 0) for r in modules.values()),
            "symbol_call_sites": sum(
                int(r.get("n_symbol_call_sites") or 0) for r in modules.values()
            ),
            "import_only_callable": sum(1 for r in modules.values() if r.get("import_only")),
            "called_outside_tests": sum(
                1 for r in modules.values() if r.get("called_outside_tests")
            ),
            "parked_missing_wake": sum(
                1
                for r in modules.values()
                if r.get("disposition") == "PARKED"
                and not (isinstance(r.get("wake"), dict) and r["wake"].get("predicate"))
            ),
            "archive_missing_reason": sum(
                1
                for r in modules.values()
                if r.get("disposition") == "ARCHIVE_CANDIDATE" and not r.get("archive_reason")
            ),
        },
        "undispositioned": undisposed,
        "modules": modules,
        "state_transitions": transitions,
        "method": (
            "STATIC source analysis. AST Call/subprocess of an exported symbol "
            "justifies CONNECTED (same rule as tools/roadmap/auditor.py: "
            "kind=import never justifies BUILT). Import graph is recorded as "
            "hcli_imported / hcli_import_path and cannot CONNECT. Subprocess "
            "path sites + importlib.import_module literals are gathered over "
            "every git-tracked *.py file (sparse-missing files loaded from "
            "HEAD). Cannot see a model-proposed WorkUnit.tool string; "
            "callable=false means no deterministic path, not 'never invoked'."
        ),
    }
    return doc


def build() -> Path:
    return write_receipt(RECEIPT, assemble_inventory(), RECORDED_BY)


def _git_grep(pattern: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "grep",
            "-nE",
            pattern,
            "HEAD",
            "--",
            "*.py",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
    )
    return proc.stdout


_IMPORT_OR_LAUNCH = re.compile(
    r"""(?:^|[\s\(\[])(?:from\s+\S+\s+import|import\s+\S+|subprocess\.(?:run|Popen|check_call|check_output|call)\()"""
)


def _spot_production_hits(rg_out: str, self_path: str, *, import_shaped: bool) -> list[str]:
    hits: list[str] = []
    for line in rg_out.splitlines():
        if not line.startswith("HEAD:"):
            continue
        rest = line[5:]
        path, sep, tail = rest.partition(":")
        if not sep:
            continue
        _, _, text = tail.partition(":")
        if path == self_path:
            continue
        if cr.is_test_path(Path(path)) or is_data_path(Path(path)):
            continue
        if import_shaped and not _IMPORT_OR_LAUNCH.search(text):
            continue
        hits.append(line)
    return hits


def _pick_rows(
    modules: Mapping[str, Mapping[str, Any]],
    classification: str,
    disposition: str,
    n: int,
) -> list[str]:
    rows = [
        (name, row)
        for name, row in modules.items()
        if row.get("classification") == classification and row.get("disposition") == disposition
    ]
    rows.sort(key=lambda kv: kv[0])
    picked: list[str] = []
    seen_trees: set[str] = set()
    for name, row in rows:
        tree = str(row.get("tree") or "")
        if tree not in seen_trees:
            picked.append(name)
            seen_trees.add(tree)
        if len(picked) >= n:
            return picked
    for name, _row in rows:
        if name not in picked:
            picked.append(name)
        if len(picked) >= n:
            break
    return picked


def run_spotchecks(doc: Mapping[str, Any]) -> dict[str, Any]:
    """5 UNREACHABLE + 5 CONNECTED, each verified with git grep -nE against HEAD.

    Evidence tier STATIC. A production import-shaped hit on an UNREACHABLE
    row is a tool bug. A CONNECTED row with no production reference is too.
    """
    modules = doc.get("modules") or {}
    unreachable = _pick_rows(modules, "UNREACHABLE", "PARKED", 5)
    connected = _pick_rows(modules, "BUILT", "CONNECTED", 5)
    if len(unreachable) < 5:
        raise AssertionError(f"not enough UNREACHABLE rows to spot-check: {unreachable}")
    if len(connected) < 5:
        raise AssertionError(f"not enough CONNECTED rows to spot-check: {connected}")
    checks: list[dict[str, Any]] = []
    for name in unreachable:
        dotted = str(modules[name]["dotted"])
        stem = Path(name).stem
        pkg = dotted.rsplit(".", 1)[0]
        pattern = (
            rf"{re.escape(dotted)}"
            rf"|from {re.escape(pkg)} import {re.escape(stem)}"
            rf"|{re.escape(name)}"
        )
        cmd = f"git --no-optional-locks grep -nE {pattern!r} HEAD -- '*.py'"
        out = _git_grep(pattern)
        hits = _spot_production_hits(out, name, import_shaped=True)
        if hits:
            raise AssertionError(
                f"{name} classified UNREACHABLE but git grep -nE found production "
                f"import/launch sites:\n" + "\n".join(hits[:12])
            )
        if modules[name].get("call_sites"):
            raise AssertionError(f"{name} UNREACHABLE but inventory lists call_sites")
        checks.append(
            {
                "verdict": "UNREACHABLE",
                "module": name,
                "command": cmd,
                "n_matches": len(out.splitlines()) if out else 0,
                "production_import_hits": hits,
                "output_head": "\n".join(out.splitlines()[:40]),
            }
        )
    for name in connected:
        dotted = str(modules[name]["dotted"])
        pattern = re.escape(dotted)
        cmd = f"git --no-optional-locks grep -nE {pattern!r} HEAD -- '*.py'"
        out = _git_grep(pattern)
        hits = _spot_production_hits(out, name, import_shaped=False)
        hcli_hits = [h for h in (out.splitlines() if out else []) if h.startswith("HEAD:hcli/")]
        if not modules[name].get("hcli_reachable"):
            raise AssertionError(f"{name} CONNECTED but hcli_reachable is false")
        inv = list(modules[name].get("hcli_invocations") or [])
        if not inv:
            raise AssertionError(
                f"{name} CONNECTED with no HCLI symbol call/subprocess "
                "(an import is not a call)"
            )
        if any(s.get("kind") == "import" for s in inv):
            raise AssertionError(f"{name} CONNECTED invocation list includes kind=import")
        if not hits and not modules[name].get("hcli_path"):
            raise AssertionError(
                f"{name} CONNECTED but git grep -nE found no production reference to {dotted}"
            )
        checks.append(
            {
                "verdict": "CONNECTED",
                "module": name,
                "command": cmd,
                "n_matches": len(out.splitlines()) if out else 0,
                "hcli_path": modules[name].get("hcli_path"),
                "hcli_invocations": inv[:8],
                "hcli_hits": hcli_hits[:20],
                "output_head": "\n".join((out.splitlines() if out else [])[:40]),
            }
        )
    return {
        "schema": "hawking.audit.reachability_triage_spotchecks.v1",
        "version": 1,
        "evidence_tier": "STATIC",
        "purpose": "Mandatory 5 UNREACHABLE + 5 CONNECTED hand verification via git grep -nE",
        "unreachable": unreachable,
        "connected": connected,
        "checks": checks,
    }


def selftest() -> Path:
    out = build()
    doc = load_json(out)
    if doc.get("schema") != SCHEMA:
        raise AssertionError(f"schema drifted: {doc.get('schema')!r}")
    if not doc.get("seal_sha256"):
        raise AssertionError("receipt is unsealed")
    modules = doc.get("modules") or {}
    if not modules:
        raise AssertionError("inventory is empty")
    undisposed = [
        name
        for name, row in modules.items()
        if row.get("disposition") not in DISPOSITIONS
    ]
    if undisposed:
        raise AssertionError(
            f"count(modules with no disposition) == {len(undisposed)} (must be 0): "
            + ", ".join(undisposed[:20])
        )
    if int((doc.get("counts") or {}).get("undispositioned") or 0) != 0:
        raise AssertionError("counts.undispositioned is not 0")
    if int((doc.get("counts") or {}).get("connected_import_only") or 0) != 0:
        raise AssertionError(
            "CONNECTED rows exist on import-only evidence "
            "(kind=import never justifies CONNECTED)"
        )
    for name, row in modules.items():
        if row.get("classification") not in CLASSIFICATIONS:
            raise AssertionError(f"{name} classification {row.get('classification')!r}")
        if row.get("evidence_tier") != "STATIC":
            raise AssertionError(f"{name} evidence_tier is not STATIC")
        if row["disposition"] == "CONNECTED" and not row.get("hcli_reachable"):
            raise AssertionError(f"{name} CONNECTED but not hcli_reachable")
        if row["disposition"] == "CONNECTED":
            inv = row.get("hcli_invocations") or []
            if not inv:
                raise AssertionError(
                    f"{name} CONNECTED on import-only evidence "
                    "(no HCLI AST Call/subprocess of an exported symbol)"
                )
            if any(s.get("kind") == "import" for s in inv):
                raise AssertionError(
                    f"{name} CONNECTED citation includes kind=import: {inv[:4]!r}"
                )
        if row.get("import_only") and row.get("disposition") == "CONNECTED":
            raise AssertionError(f"{name} CONNECTED but import_only=true")
        if row["disposition"] == "PARKED" and not row.get("wake_condition"):
            raise AssertionError(f"{name} PARKED with no wake_condition")
        if row["disposition"] == "PARKED":
            wake = row.get("wake")
            if not isinstance(wake, dict) or not wake.get("kind") or not wake.get("predicate"):
                raise AssertionError(f"{name} PARKED with no machine-readable wake")
            if wake.get("required_kind") != WAKE_REQUIRED_KIND:
                raise AssertionError(
                    f"{name} wake.required_kind is {wake.get('required_kind')!r}, "
                    f"must be {WAKE_REQUIRED_KIND!r} (an import is not a call)"
                )
        if row["disposition"] == "ARCHIVE_CANDIDATE" and not row.get("archive_reason"):
            raise AssertionError(f"{name} ARCHIVE_CANDIDATE with no reason")
        if row["disposition"] == "ARCHIVE_CANDIDATE":
            arch = row.get("archive")
            if not isinstance(arch, dict) or not arch.get("reason"):
                raise AssertionError(f"{name} ARCHIVE_CANDIDATE with no structured archive reason")
            if arch.get("deleted") is not False:
                raise AssertionError(f"{name} archive.deleted must be False; nothing is deleted")
        if row.get("callable_outside_tests") and not row.get("call_sites"):
            raise AssertionError(f"{name} callable with no call_sites")
        if (not row.get("callable_outside_tests")) and row.get("call_sites"):
            raise AssertionError(f"{name} has call_sites but callable_outside_tests=false")
    transitions = doc.get("state_transitions") or []
    if not transitions:
        raise AssertionError("state_transitions is empty")
    for t in transitions:
        if not t.get("file") or not t.get("enter"):
            raise AssertionError(f"state-transition missing file/enter: {t!r}")
        if t.get("status") not in {"EXIT_EXISTS", "EXIT_MISSING"}:
            raise AssertionError(f"state-transition status {t.get('status')!r}")
    spot = run_spotchecks(doc)
    write_receipt("REACHABILITY_TRIAGE_SPOTCHECKS.json", spot, RECORDED_BY)
    if len(spot["checks"]) != 10:
        raise AssertionError(f"expected 10 spot-checks, got {len(spot['checks'])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-module reachability inventory with a disposition on every row"
    )
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--inspect", metavar="ID")
    ap.add_argument("--invoke", metavar="ID")
    ap.add_argument("--args", default="{}", help="JSON object for --invoke arguments")
    ap.add_argument("--family")
    ap.add_argument("--disposition")
    ap.add_argument("--module", metavar="PATH", help="targeted query: one module row")
    ap.add_argument("--classification", help="targeted query: modules with this classification")
    ap.add_argument("--gate", metavar="ID", help="targeted query: one CAPABILITY_GRAPH gate")
    ap.add_argument("--status", help="targeted query: gates with this status (e.g. BUILT)")
    ap.add_argument("--parity", action="store_true", help="all-entity equivalence vs full parse")
    ap.add_argument("--measure", action="store_true", help="time + bytes-read, both paths")
    require_known_flags({
        "--build", "--selftest", "--discover", "--inspect", "--invoke",
        "--args", "--family", "--disposition",
        "--module", "--classification", "--gate", "--status", "--parity", "--measure",
    })
    args = ap.parse_args()
    if args.inspect or args.invoke or args.discover:
        return manifest_main(
            (["--inspect", args.inspect] if args.inspect else [])
            + (["--invoke", args.invoke] if args.invoke else [])
            + (["--discover"] if args.discover else [])
            + (["--args", args.args] if args.invoke else [])
            + (["--family", args.family] if args.family else [])
            + (["--disposition", args.disposition] if args.disposition else [])
        )
    if (
        args.parity
        or args.measure
        or args.module
        or args.classification
        or args.gate
        or args.status
        or args.disposition
    ):
        return _query_main(args)
    out = selftest() if args.selftest else build()
    doc = load_json(out)
    counts = doc.get("counts") or {}
    print(out)
    print(
        "modules={n} undisposed={u} CONNECTED={c} PARKED={p} ARCHIVE={a} "
        "BUILT={b} DORMANT={d} UNREACHABLE={un} SCAFFOLDED={s} "
        "hcli_reachable={h} missing_exits={me}".format(
            n=counts.get("modules"),
            u=counts.get("undispositioned"),
            c=(counts.get("by_disposition") or {}).get("CONNECTED"),
            p=(counts.get("by_disposition") or {}).get("PARKED"),
            a=(counts.get("by_disposition") or {}).get("ARCHIVE_CANDIDATE"),
            b=(counts.get("by_classification") or {}).get("BUILT"),
            d=(counts.get("by_classification") or {}).get("DORMANT"),
            un=(counts.get("by_classification") or {}).get("UNREACHABLE"),
            s=(counts.get("by_classification") or {}).get("SCAFFOLDED"),
            h=counts.get("hcli_reachable"),
            me=counts.get("state_transition_exits_missing"),
        )
    )
    return 0


def _query_main(args: argparse.Namespace) -> int:
    """Targeted access path. Never writes receipts. Full parse is the oracle."""
    from tools.audit import artifact_index as _ai

    if args.parity or args.measure:
        triage = _ai.materialize(TRIAGE_REL)
        graph = _ai.materialize("civilization/CAPABILITY_GRAPH.json")
        report: dict[str, Any] = {"schema": "hawking.audit.artifact_query_parity.v1"}
        if args.parity:
            report["triage_modules"] = _ai.parity_map(triage, "modules")
            report["graph_gates"] = _ai.parity_map(graph, "gates")
            unreachable = _ai.measure_filter(triage, "modules", classification="UNREACHABLE")
            built = _ai.measure_filter(graph, "gates", status="BUILT")
            report["unreachable"] = unreachable
            report["built_gates"] = built
            ok = (
                report["triage_modules"]["ok"]
                and report["graph_gates"]["ok"]
                and unreachable["equal"]
                and built["equal"]
            )
            report["ok"] = ok
        if args.measure:
            # One-module question on a real CONNECTED row, plus the two filters.
            sample = "tools/future/status_causality.py"
            report["measure_module"] = _ai.measure_one(triage, "modules", sample)
            report["measure_unreachable"] = _ai.measure_filter(
                triage, "modules", classification="UNREACHABLE"
            )
            report["measure_built_gates"] = _ai.measure_filter(
                graph, "gates", status="BUILT"
            )
        print(json.dumps(report, indent=1, sort_keys=True))
        if args.parity and not report.get("ok", True):
            return 1
        return 0
    if args.module:
        print(json.dumps(query_module(args.module), indent=1, sort_keys=True))
        return 0
    if args.classification or args.disposition:
        print(
            json.dumps(
                query_modules(
                    classification=args.classification,
                    disposition=args.disposition,
                ),
                indent=1,
            )
        )
        return 0
    if args.gate:
        print(json.dumps(query_gates(gate=args.gate), indent=1, sort_keys=True))
        return 0
    if args.status:
        print(json.dumps(query_gates(status=args.status), indent=1))
        return 0
    return 2


# ==========================================================================
# Capability manifest adapter
# Compact HCLI-consumable surface over the dormant pile. Lives in this file
# because the sandbox cannot git-add a new untracked sibling; a capability
# does not exist until something CALLS it, and this is that call path.
# ==========================================================================

MANIFEST_DOC = """Capability manifest: compact HCLI-consumable surface over the dormant pile.

A definition is not a capability. A module import is not a call site. The
reachability inventory classified 516 modules and found hundreds PARKED /
UNREACHABLE against a handful CONNECTED; this adapter is the missing
discovery + call layer, built outside hcli/ so the live daemon is not edited.

What already existed (reused, not rewritten)
--------------------------------------------
- tools/future/orchestration.py BINDINGS + invoke(): registration, and
  invoke() is not statically reached from HCLI. A BINDINGS row is not a call.
- tools/future/capability_reachability.py: the analyzer. Its caller citations
  are import-dominated; this adapter requires an AST Call of a named symbol.
- tools/audit/reachability_triage.py: per-module inventory, wake strings.
  This module consumes that inventory and adds a typed invoke surface.
- hcli.tool_registry.ToolSpec schema ``hcli.agentos.tool.v1``: the shape HCLI
  already knows how to register. Mirrored here so the daemon can json-load
  these three specs and hand them a handler. This file does not import hcli.
- tools/verify/capability_manifest.py: entrypoint-replacement accounting.
  A different problem; not this surface.

Model-facing surface: three verbs (not one per module).
The registry underneath is one row per inventoried module.

    python3 tools/audit/reachability_triage.py --discover
    python3 tools/audit/reachability_triage.py --inspect future.capacity_inference_rule
    python3 tools/audit/reachability_triage.py --invoke future.capacity_inference_rule --args '{"levels":[...],"semantics_comparable":true}'
"""

MANIFEST_SCHEMA = "hawking.audit.capability_manifest.v1"
TOOL_SCHEMA = "hcli.agentos.tool.v1"
RESULT_SCHEMA = "hcli.agentos.tool.result.v1"
MANIFEST_VERSION = 1
MANIFEST_RECORDED_BY = "tools/audit/reachability_triage.py"
EVIDENCE_TIER_STATIC = "STATIC"
EVIDENCE_TIER_INVOKE = "FUNCTIONAL_SIM"

# Compact model-facing surface. Bounded. Not one verb per module.
SURFACE_VERBS: tuple[str, ...] = (
    "capability.discover",
    "capability.inspect",
    "capability.invoke",
)
SURFACE_VERB_COUNT = len(SURFACE_VERBS)

ADAPTER_REL = "tools/audit/reachability_triage.py"
TRIAGE_REL = "receipts/future/REACHABILITY_TRIAGE.json"

READ_ONLY = "read_only"


# --------------------------------------------------------------------------
# AST Call-site scan of THIS adapter. Import is not a call.
# --------------------------------------------------------------------------


def adapter_called_symbols(source: str | None = None) -> set[tuple[str, str]]:
    """(module_dotted, symbol) pairs that this file actually CALLS.

    Built from AST Call nodes plus the ImportFrom bindings that name them.
    A `from m import foo` with no `foo(` is not a hit.
    """
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    binds: dict[str, tuple[str, str]] = {}
    module_binds: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                binds[local] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                module_binds[local] = alias.name
    called: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in binds:
            called.add(binds[func.id])
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
        ):
            if func.value.id in binds:
                mod, name = binds[func.value.id]
                called.add((f"{mod}.{name}", func.attr))
            elif func.value.id in module_binds:
                called.add((module_binds[func.value.id], func.attr))
    return called


def _status_from_calls(dotted: str, symbol: str, source: str | None = None) -> str:
    called = adapter_called_symbols(source)
    if (dotted, symbol) in called:
        return "CALLABLE"
    return "UNREACHABLE"


# --------------------------------------------------------------------------
# Wired invokers. Each handler MUST contain an AST Call of its symbol.
# --------------------------------------------------------------------------


def _invoke_capacity_inference(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.capacity_inference_rule import fires_on

    levels = arguments.get("levels")
    if not isinstance(levels, list):
        raise ValueError("levels must be a list of {concurrency, aggregate_decode_tps}")
    comparable = bool(arguments.get("semantics_comparable", True))
    # WIRED_CALL future.capacity_inference_rule.fires_on
    result = fires_on(levels, semantics_comparable=comparable)
    return {
        "ok": True,
        "value": result,
        "symbol": "fires_on",
        "evidence_tier": EVIDENCE_TIER_INVOKE,
    }


def _invoke_fidelity_hierarchy(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.fidelity_hierarchy import may_refuse

    claim_level = str(arguments.get("claim_level") or "")
    measured_level = str(arguments.get("measured_level") or "")
    if not claim_level or not measured_level:
        raise ValueError("claim_level and measured_level are required")
    # WIRED_CALL future.fidelity_hierarchy.may_refuse
    result = may_refuse(claim_level=claim_level, measured_level=measured_level)
    return {
        "ok": True,
        "value": result,
        "symbol": "may_refuse",
        "evidence_tier": EVIDENCE_TIER_INVOKE,
    }


def _invoke_ebpw_promote(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.ebpw_categories import PromotionLedger, can_promote

    raw = arguments.get("ledger")
    if raw is None or raw == {}:
        ledger: Any = PromotionLedger()
    else:
        ledger = raw
    # WIRED_CALL future.ebpw_categories.can_promote
    ok, reason = can_promote(ledger)
    return {
        "ok": True,
        "value": {"can_promote": ok, "reason": reason},
        "symbol": "can_promote",
        "evidence_tier": EVIDENCE_TIER_INVOKE,
    }


def _invoke_tabula(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.tabula import ScoreVector, evaluate

    if arguments.get("disposition"):
        from tools.future.tabula import disposition as tabula_disposition

        return {
            "ok": True,
            "value": tabula_disposition(),
            "symbol": "evaluate",
            "evidence_tier": EVIDENCE_TIER_STATIC,
        }
    raw = arguments.get("scores")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "scores must be an object with behavioral, capability, tool_use, "
            "reasoning, instruction_following"
        )
    vec = ScoreVector.from_mapping(raw)
    # WIRED_CALL future.tabula.evaluate
    result = evaluate(vec)
    return {
        "ok": True,
        "value": result.to_dict(),
        "symbol": "evaluate",
        "evidence_tier": EVIDENCE_TIER_INVOKE,
    }


def _invoke_vmcp(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.vmcp import compact_surface

    act = str(arguments.get("act") or "disposition")
    # WIRED_CALL future.vmcp.compact_surface
    result = compact_surface(act, arguments)
    tier = str(result.get("evidence_tier") or EVIDENCE_TIER_INVOKE)
    return {
        "ok": True,
        "value": result,
        "symbol": "compact_surface",
        "evidence_tier": tier,
    }


# Stable ids. Typed signatures. The original three plus Tabula / VMCP
# dispositions: a dormant module can be discovered AND called through this adapter.
WIRED: dict[str, dict[str, Any]] = {
    "future.capacity_inference_rule": {
        "purpose": (
            "G090: infer SINGLE_WORKLOAD_UNDERUTILIZATION from N-stream "
            "throughput at comparable semantics, and generate competing questions"
        ),
        "module": "tools/future/capacity_inference_rule.py",
        "dotted": "tools.future.capacity_inference_rule",
        "symbol": "fires_on",
        "family": "science.rule",
        "handler": _invoke_capacity_inference,
        "input_schema": {
            "type": "object",
            "required": ["levels"],
            "additionalProperties": False,
            "properties": {
                "levels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "concurrency": {"type": "integer"},
                            "aggregate_decode_tps": {"type": "number"},
                        },
                    },
                },
                "semantics_comparable": {"type": "boolean"},
            },
        },
        "output_schema": {"type": "object"},
        "triage_classification": "UNREACHABLE",
    },
    "future.fidelity_hierarchy": {
        "purpose": (
            "G108: a candidate may not be refused at a bar stricter than its "
            "claim (the 64x CAPABILITY_INFORMATION_MAP error)"
        ),
        "module": "tools/future/fidelity_hierarchy.py",
        "dotted": "tools.future.fidelity_hierarchy",
        "symbol": "may_refuse",
        "family": "science.rule",
        "handler": _invoke_fidelity_hierarchy,
        "input_schema": {
            "type": "object",
            "required": ["claim_level", "measured_level"],
            "additionalProperties": False,
            "properties": {
                "claim_level": {"type": "string"},
                "measured_level": {"type": "string"},
            },
        },
        "output_schema": {"type": "object"},
        "triage_classification": "UNREACHABLE",
    },
    "future.ebpw_categories": {
        "purpose": (
            "EBPW category validator: prospective_meta_bpw < 1 never promotes; "
            "cross-category arithmetic is a type error"
        ),
        "module": "tools/future/ebpw_categories.py",
        "dotted": "tools.future.ebpw_categories",
        "symbol": "can_promote",
        "family": "science.ebpw",
        "handler": _invoke_ebpw_promote,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"ledger": {"type": "object"}},
        },
        "output_schema": {"type": "object"},
        "triage_classification": "DORMANT",
    },
    "future.tabula": {
        "purpose": (
            "Tabula independent-evaluation floor: score a behavioral-surgery "
            "child on the five-axis vector. Zero refusal is never the only score."
        ),
        "module": "tools/future/tabula.py",
        "dotted": "tools.future.tabula",
        "symbol": "evaluate",
        "family": "science.tabula",
        "handler": _invoke_tabula,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores": {
                    "type": "object",
                    "required": [
                        "behavioral",
                        "capability",
                        "tool_use",
                        "reasoning",
                        "instruction_following",
                    ],
                    "properties": {
                        "behavioral": {"type": "number"},
                        "capability": {"type": "number"},
                        "tool_use": {"type": "number"},
                        "reasoning": {"type": "number"},
                        "instruction_following": {"type": "number"},
                    },
                },
                "disposition": {"type": "boolean"},
            },
        },
        "output_schema": {"type": "object"},
        "triage_classification": "DORMANT",
    },
    "future.vmcp": {
        "purpose": (
            "VMCP compact E.14 surface: see/hold/know/check/prove of a local "
            "file, plus disposition of every named organ. PARKED acts return a "
            "wake, never an empty success."
        ),
        "module": "tools/future/vmcp.py",
        "dotted": "tools.future.vmcp",
        "symbol": "compact_surface",
        "family": "perception.vmcp",
        "handler": _invoke_vmcp,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "act": {"type": "string"},
                "path": {"type": "string"},
                "other_path": {"type": "string"},
                "expected_sha256": {"type": "string"},
                "sha256": {"type": "string"},
                "max_bytes": {"type": "integer"},
            },
        },
        "output_schema": {"type": "object"},
        "triage_classification": "UNREACHABLE",
    },
}


def wired_status(cap_id: str, source: str | None = None) -> str:
    spec = WIRED.get(cap_id)
    if spec is None:
        return "UNREACHABLE"
    return _status_from_calls(spec["dotted"], spec["symbol"], source)


# --------------------------------------------------------------------------
# Inventory load (no second analyzer; no receipts write)
# --------------------------------------------------------------------------


def capability_id(module_rel: str) -> str:
    parts = Path(module_rel).with_suffix("").parts
    if parts and parts[0] == "tools":
        parts = parts[1:]
    return ".".join(parts)


def family_of(module_rel: str, summary: str) -> str:
    rel = module_rel.replace("\\", "/")
    if rel.startswith("tools/accelerator/"):
        return "accelerator"
    if rel.startswith("tools/odyssey/"):
        return "odyssey"
    if rel.startswith("tools/headless/"):
        return "headless"
    blob = f"{rel} {(summary or '')}".lower()
    rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("ebpw", "bpw", "byte ledger"), "science.ebpw"),
        (("tabula", "abliterat"), "science.tabula"),
        (("vmcp", "visionmcp", "all-seeing"), "perception.vmcp"),
        (("fidelity", "claim_scope", "inference", "scar", "negative"), "science.rule"),
        (("resident", "wakeup", "workunit", "orchestration"), "resident"),
        (("fpga", "ane", "metal", "kernel", "hardware", "hwir"), "hardware.static"),
        (("deltanet", "mlp", "organ", "flash", "ngram"), "representation"),
        (("odyssey", "specimen", "tournament"), "odyssey"),
    )
    for keys, fam in rules:
        if any(k in blob for k in keys):
            return fam
    return "future.sidecar"


def load_triage() -> dict[str, Any]:
    """Load the reachability inventory. Prefer a live file; else HEAD blob.

    Does not write receipts. Does not assemble unless the blob is missing.

    This is the FULL PARSE path — the parity oracle. Targeted questions
    (one module, UNREACHABLE set, BUILT gates) go through query_module /
    query_modules / query_gates so they do not json.loads the 2.7MB document.
    """
    path = REPO / TRIAGE_REL
    if path.is_file():
        return load_json(path)
    blob = git("show", f"HEAD:{TRIAGE_REL}")
    if blob:
        return json.loads(blob)
    return assemble_inventory()


def _triage_json_path() -> Path:
    from tools.audit import artifact_index as _ai

    return _ai.materialize(TRIAGE_REL)


def _graph_json_path() -> Path:
    from tools.audit import artifact_index as _ai

    return _ai.materialize("civilization/CAPABILITY_GRAPH.json")


def query_module(module: str) -> dict[str, Any]:
    """Disposition row for one module. Identical to load_triage()['modules'][module].

    Fast path: sidecar pread. On any error, the full parse is the verdict.
    """
    if _os.environ.get("HCLI_TRIAGE_FULLPARSE") == "1":
        return load_triage()["modules"][module]
    try:
        from tools.audit import artifact_index as _ai

        return _ai.get("modules", module, json_path=_triage_json_path()).value
    except Exception:
        return load_triage()["modules"][module]


def query_modules(
    *,
    classification: str | None = None,
    disposition: str | None = None,
) -> list[str]:
    """Module paths matching a column filter. Identical to the full-parse key set."""
    if _os.environ.get("HCLI_TRIAGE_FULLPARSE") == "1":
        return _full_module_keys(classification=classification, disposition=disposition)
    try:
        from tools.audit import artifact_index as _ai

        keys, _n, _s = _ai.list_keys(
            "modules",
            json_path=_triage_json_path(),
            classification=classification,
            disposition=disposition,
        )
        return keys
    except Exception:
        return _full_module_keys(classification=classification, disposition=disposition)


def query_gates(*, status: str | None = None, gate: str | None = None) -> Any:
    """CAPABILITY_GRAPH gates. `gate` returns one row; `status` returns matching ids."""
    if gate is not None:
        if _os.environ.get("HCLI_TRIAGE_FULLPARSE") == "1":
            return _full_graph()["gates"][gate]
        try:
            from tools.audit import artifact_index as _ai

            return _ai.get("gates", gate, json_path=_graph_json_path()).value
        except Exception:
            return _full_graph()["gates"][gate]
    if _os.environ.get("HCLI_TRIAGE_FULLPARSE") == "1":
        return _full_gate_keys(status=status)
    try:
        from tools.audit import artifact_index as _ai

        keys, _n, _s = _ai.list_keys(
            "gates", json_path=_graph_json_path(), status=status
        )
        return keys
    except Exception:
        return _full_gate_keys(status=status)


def _full_graph() -> dict[str, Any]:
    from tools.audit import artifact_index as _ai

    path = REPO / "civilization" / "CAPABILITY_GRAPH.json"
    if path.is_file():
        return load_json(path)
    blob = git("show", "HEAD:civilization/CAPABILITY_GRAPH.json")
    if blob:
        return json.loads(blob)
    doc, _n, _s = _ai.full_parse(_graph_json_path())
    return doc


def _full_module_keys(
    *,
    classification: str | None = None,
    disposition: str | None = None,
) -> list[str]:
    modules = load_triage().get("modules") or {}
    out = []
    for name, row in modules.items():
        if classification is not None and row.get("classification") != classification:
            continue
        if disposition is not None and row.get("disposition") != disposition:
            continue
        out.append(name)
    out.sort()
    return out


def _full_gate_keys(*, status: str | None = None) -> list[str]:
    gates = (_full_graph().get("gates") or {})
    out = []
    for name, row in gates.items():
        if status is not None and row.get("status") != status:
            continue
        out.append(name)
    out.sort()
    return out


def _lookup_module_row(cap_id: str) -> dict[str, Any] | None:
    """One inventory row by capability id. Fast path first; full parse fallback."""
    if _os.environ.get("HCLI_TRIAGE_FULLPARSE") == "1":
        return _modules_by_id().get(cap_id)
    try:
        from tools.audit import artifact_index as _ai

        hit = _ai.get_by_cap_id(cap_id, json_path=_triage_json_path())
        if hit is not None:
            return hit.value
    except Exception:
        pass
    return _modules_by_id().get(cap_id)


def _ensure_structured(row: dict[str, Any]) -> dict[str, Any]:
    disp = row.get("disposition")
    if disp == "PARKED":
        wake = row.get("wake")
        if not isinstance(wake, dict) or not wake.get("predicate"):
            row["wake"] = structured_wake(row)
    elif disp == "ARCHIVE_CANDIDATE":
        arch = row.get("archive")
        if not isinstance(arch, dict) or not arch.get("reason"):
            row["archive"] = structured_archive(row)
    return row


def _signature_for(row: Mapping[str, Any], wired: Mapping[str, Any] | None) -> dict[str, Any]:
    if wired is not None:
        return {
            "symbol": wired["symbol"],
            "kind": "function",
            "arguments": wired["input_schema"],
            "returns": wired.get("output_schema") or {"type": "object"},
            "wired": True,
        }
    funcs = list(row.get("public_functions") or [])
    primary = None
    if funcs:
        preferred = {
            "selftest",
            "build",
            "fire",
            "evaluate",
            "inspect",
            "validate",
            "query",
            "scan",
            "probe",
            "classify",
        }
        for fn in funcs:
            if fn.get("name") in preferred:
                primary = fn
                break
        if primary is None:
            primary = funcs[0]
    symbol = (primary or {}).get("name")
    args = (primary or {}).get("args") or []
    return {
        "symbol": symbol,
        "kind": "function" if symbol else "none",
        "arguments": {
            "type": "object",
            "properties": {a: {"type": "string"} for a in args},
        },
        "returns": {"type": "object"},
        "wired": False,
        "note": (
            "invoke refuses until tools/audit/capability_manifest.py contains "
            f"an AST Call of {(row.get('dotted') or '')}.{symbol}"
            if symbol
            else "no public function to call"
        ),
    }


def _useful(row: Mapping[str, Any]) -> bool:
    if row.get("is_package_marker"):
        return False
    if int(row.get("n_functions") or 0) + int(row.get("n_classes") or 0) > 0:
        return True
    if row.get("has_main"):
        return True
    return False


def capability_entry(
    row: Mapping[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    row_d = _ensure_structured(dict(row))
    cid = capability_id(str(row_d.get("module") or ""))
    wired = WIRED.get(cid)
    disp = row_d.get("disposition")
    if wired is not None:
        status = wired_status(cid, source)
    elif disp == "CONNECTED":
        status = "CONNECTED"
    elif disp == "ARCHIVE_CANDIDATE":
        status = "ARCHIVE_CANDIDATE"
    else:
        status = "UNREACHABLE" if not row_d.get("called_outside_tests") else "PARKED"
    purpose = wired["purpose"] if wired else (row_d.get("summary") or "(no module docstring)")
    sig = _signature_for(row_d, wired)
    blocker = None
    if isinstance(row_d.get("wake"), dict):
        blocker = row_d["wake"].get("blocker")
    elif isinstance(row_d.get("archive"), dict):
        blocker = row_d["archive"].get("reason")
    return {
        "id": cid,
        "purpose": purpose,
        "module": row_d.get("module"),
        "dotted": row_d.get("dotted"),
        "family": family_of(str(row_d.get("module") or ""), str(purpose)),
        "disposition": disp,
        "classification": row_d.get("classification"),
        "status": status,
        "signature": sig,
        "wake": row_d.get("wake"),
        "archive": row_d.get("archive"),
        "wake_condition": row_d.get("wake_condition"),
        "archive_reason": row_d.get("archive_reason"),
        "evidence": {
            "tier": EVIDENCE_TIER_STATIC,
            "called_outside_tests": bool(row_d.get("called_outside_tests")),
            "import_only": bool(row_d.get("import_only")),
            "n_import_sites": int(row_d.get("n_import_sites") or 0),
            "n_symbol_call_sites": int(row_d.get("n_symbol_call_sites") or 0),
            "symbol_call_sites": row_d.get("symbol_call_sites") or [],
            "hcli_reachable": bool(row_d.get("hcli_reachable")),
            "orchestration_bound": bool(row_d.get("orchestration_bound")),
            "blocker": blocker,
        },
        "invoker": "wired" if wired is not None else None,
    }


def list_capabilities(
    *,
    inventory: Mapping[str, Any] | None = None,
    source: str | None = None,
    useful_only: bool = True,
) -> list[dict[str, Any]]:
    doc = inventory if inventory is not None else load_triage()
    modules = doc.get("modules") or {}
    out: list[dict[str, Any]] = []
    for name, row in modules.items():
        if useful_only and not _useful(row) and capability_id(name) not in WIRED:
            continue
        out.append(capability_entry(row, source=source))
    out.sort(key=lambda e: str(e["id"]))
    return out


# --------------------------------------------------------------------------
# Compact HCLI-consumable surface (3 verbs)
# --------------------------------------------------------------------------


def _tool_spec(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": TOOL_SCHEMA,
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "output_schema": output_schema or {"type": "object"},
        "mutation": READ_ONLY,
        "deterministic": True,
        "timeout_s": 30.0,
        "roles": ["resident", "mission"],
        "resources": [],
        "verifier_expectations": [],
        "provenance": MANIFEST_RECORDED_BY,
    }


def hcli_tool_specs() -> list[dict[str, Any]]:
    """Three ToolSpec-shaped dicts HCLI can register without a code change.

    Matching keys of hcli.tool_registry.ToolSpec.to_dict(). Handler is not
    serialized (it is not JSON); bind via handle(name, arguments).
    """
    return [
        _tool_spec(
            "capability.discover",
            "List dormant/connected capabilities. Compact rows, not one verb per module.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "family": {"type": "string"},
                    "disposition": {"type": "string"},
                    "status": {"type": "string"},
                    "useful_only": {"type": "boolean"},
                },
            },
        ),
        _tool_spec(
            "capability.inspect",
            "Inspect one capability: purpose, typed signature, evidence, wake/archive.",
            {
                "type": "object",
                "required": ["id"],
                "additionalProperties": False,
                "properties": {"id": {"type": "string"}},
            },
        ),
        _tool_spec(
            "capability.invoke",
            "Invoke one wired dormant capability by id. Unwired ids refuse with their wake condition.",
            {
                "type": "object",
                "required": ["id"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "arguments": {"type": "object"},
                },
            },
        ),
    ]


def _result(
    tool: str,
    *,
    ok: bool,
    value: Any = None,
    error: str | None = None,
    failure_class: str | None = None,
    evidence_tier: str = EVIDENCE_TIER_STATIC,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "tool": tool,
        "ok": ok,
        "value": value,
        "error": error,
        "failure_class": failure_class,
        "mutation": READ_ONLY,
        "deterministic": True,
        "evidence_tier": evidence_tier,
        "provenance": {"adapter": ADAPTER_REL, "recorded_by": MANIFEST_RECORDED_BY},
    }


def _modules_by_id(inventory: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    doc = inventory if inventory is not None else load_triage()
    return {capability_id(name): row for name, row in (doc.get("modules") or {}).items()}


def discover(arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    family = arguments.get("family")
    disposition = arguments.get("disposition")
    status = arguments.get("status")
    useful_only = arguments.get("useful_only")
    if useful_only is None:
        useful_only = True
    entries = list_capabilities(useful_only=bool(useful_only))
    if family:
        entries = [e for e in entries if e.get("family") == family]
    if disposition:
        entries = [e for e in entries if e.get("disposition") == disposition]
    if status:
        entries = [e for e in entries if e.get("status") == status]
    compact = [
        {
            "id": e["id"],
            "purpose": e["purpose"],
            "family": e["family"],
            "status": e["status"],
            "disposition": e["disposition"],
            "classification": e["classification"],
            "module": e["module"],
            "wired": e["invoker"] == "wired",
        }
        for e in entries
    ]
    parked = [e for e in entries if e.get("disposition") == "PARKED"]
    missing_wake = [
        e["id"] for e in parked if not (isinstance(e.get("wake"), dict) and e["wake"].get("predicate"))
    ]
    archive = [e for e in entries if e.get("disposition") == "ARCHIVE_CANDIDATE"]
    missing_archive = [e["id"] for e in archive if not e.get("archive_reason")]
    return _result(
        "capability.discover",
        ok=True,
        value={
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "evidence_tier": EVIDENCE_TIER_STATIC,
            "surface_verbs": list(SURFACE_VERBS),
            "surface_verb_count": SURFACE_VERB_COUNT,
            "n": len(compact),
            "n_parked": len(parked),
            "n_parked_missing_wake": len(missing_wake),
            "n_archive_missing_reason": len(missing_archive),
            "n_undispositioned": 0,
            "wired_ids": sorted(WIRED),
            "capabilities": compact,
        },
        evidence_tier=EVIDENCE_TIER_STATIC,
    )


def inspect(arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    cap_id = str(arguments.get("id") or "").strip()
    if not cap_id:
        return _result(
            "capability.inspect",
            ok=False,
            error="id is required",
            failure_class="invalid_arguments",
        )
    row = _lookup_module_row(cap_id)
    if row is None and cap_id in WIRED:
        spec = WIRED[cap_id]
        row = {
            "module": spec["module"],
            "dotted": spec["dotted"],
            "summary": spec["purpose"],
            "disposition": "PARKED",
            "classification": spec["triage_classification"],
            "callable_outside_tests": spec["triage_classification"] == "DORMANT",
            "hcli_reachable": False,
            "public_functions": [{"name": spec["symbol"], "args": []}],
        }
        decide(row)
    if row is None:
        return _result(
            "capability.inspect",
            ok=False,
            error=f"unknown capability {cap_id!r}",
            failure_class="unknown_capability",
        )
    entry = capability_entry(row)
    return _result("capability.inspect", ok=True, value=entry, evidence_tier=EVIDENCE_TIER_STATIC)


def invoke(arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    cap_id = str(arguments.get("id") or "").strip()
    inner = arguments.get("arguments") or {}
    if not isinstance(inner, Mapping):
        return _result(
            "capability.invoke",
            ok=False,
            error="arguments must be an object",
            failure_class="invalid_arguments",
        )
    spec = WIRED.get(cap_id)
    if spec is None:
        inspected = inspect({"id": cap_id})
        wake = None
        if inspected.get("ok"):
            wake = (inspected.get("value") or {}).get("wake")
        return _result(
            "capability.invoke",
            ok=False,
            error=f"{cap_id} has no wired call path in {ADAPTER_REL}",
            failure_class="UNREACHABLE",
            value={"wake": wake, "status": "UNREACHABLE"},
        )
    status = wired_status(cap_id)
    if status != "CALLABLE":
        return _result(
            "capability.invoke",
            ok=False,
            error=(
                f"{cap_id} is {status}: {ADAPTER_REL} does not contain an AST "
                f"Call of {spec['dotted']}.{spec['symbol']}"
            ),
            failure_class="UNREACHABLE",
            value={"status": status, "required_symbol": f"{spec['dotted']}.{spec['symbol']}"},
        )
    handler: Callable[[Mapping[str, Any]], dict[str, Any]] = spec["handler"]
    try:
        payload = handler(inner)
    except Exception as exc:
        return _result(
            "capability.invoke",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            failure_class="invoke_error",
            evidence_tier=EVIDENCE_TIER_INVOKE,
        )
    return _result(
        "capability.invoke",
        ok=bool(payload.get("ok", True)),
        value={
            "id": cap_id,
            "symbol": spec["symbol"],
            "module": spec["module"],
            "result": payload.get("value"),
            "evidence_tier": payload.get("evidence_tier") or EVIDENCE_TIER_INVOKE,
        },
        evidence_tier=EVIDENCE_TIER_INVOKE,
    )


HANDLERS: dict[str, Callable[[Mapping[str, Any] | None], dict[str, Any]]] = {
    "capability.discover": discover,
    "capability.inspect": inspect,
    "capability.invoke": invoke,
}


def handle(name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The bind point HCLI would pass as ToolSpec.handler."""
    fn = HANDLERS.get(name)
    if fn is None:
        return _result(
            name,
            ok=False,
            error=f"unknown verb {name!r}; surface is {list(SURFACE_VERBS)}",
            failure_class="unknown_tool",
        )
    return fn(arguments)


def parked_wake_gaps(inventory: Mapping[str, Any] | None = None) -> list[str]:
    doc = inventory if inventory is not None else load_triage()
    missing: list[str] = []
    for name, row in (doc.get("modules") or {}).items():
        stamped = _ensure_structured(dict(row))
        if stamped.get("disposition") != "PARKED":
            continue
        wake = stamped.get("wake")
        if not isinstance(wake, dict) or not wake.get("kind") or not wake.get("predicate"):
            missing.append(name)
        elif wake.get("required_kind") != WAKE_REQUIRED_KIND:
            missing.append(name)
    return missing


def undispositioned(inventory: Mapping[str, Any] | None = None) -> list[str]:
    doc = inventory if inventory is not None else load_triage()
    bad: list[str] = []
    for name, row in (doc.get("modules") or {}).items():
        if row.get("disposition") not in DISPOSITIONS:
            bad.append(name)
    counts = doc.get("counts") or {}
    if int(counts.get("undispositioned") or 0) != 0 and not bad:
        bad = list(doc.get("undispositioned") or ["counts.undispositioned!=0"])
    return bad


def manifest_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--inspect", metavar="ID")
    ap.add_argument("--invoke", metavar="ID")
    ap.add_argument("--args", default="{}", help="JSON object for --invoke arguments")
    ap.add_argument("--family")
    ap.add_argument("--disposition")
    require_known_flags({"--discover", "--inspect", "--invoke", "--args", "--family", "--disposition"})
    args = ap.parse_args(argv)
    if args.inspect:
        print(json.dumps(inspect({"id": args.inspect}), indent=1, sort_keys=True))
        return 0
    if args.invoke:
        inner = json.loads(args.args)
        print(json.dumps(invoke({"id": args.invoke, "arguments": inner}), indent=1, sort_keys=True))
        return 0
    payload = discover({"family": args.family, "disposition": args.disposition})
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
