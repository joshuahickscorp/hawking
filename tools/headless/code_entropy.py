#!/usr/bin/env python3
"""CODE_ENTROPY — classified census on top of CODE_GRAPH. Deletes nothing.

Reads receipts/headless/CODE_GRAPH.json and classifies every census module
and every top-level symbol as exactly one of DELETE / ARCHIVE / KEEP /
UNKNOWN. A missing sparse-checkout path is not evidence of absence: blobs
are read with `git show HEAD:<path>` and searches use `git grep HEAD`.

This lane is a plan for a later deletion pass. It never unlinks, git-rms,
or rewrites any source other than this file and the receipt it writes.

    python3 tools/headless/code_entropy.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCHEMA = "hawking.headless.code_entropy.v1"
REPO = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO / "receipts" / "headless" / "CODE_GRAPH.json"
RECEIPT = REPO / "receipts" / "headless" / "CODE_ENTROPY.json"

LABELS = ("DELETE", "ARCHIVE", "KEEP", "UNKNOWN")

# Mentions inside these paths are the census talking about a symbol, not a
# caller of it. A later deletion pass must re-run the searches, not trust us.
CENSUS_NOISE = (
    "tools/headless/code_entropy.py",
    "tools/headless/code_graph.py",
    "tools/headless/dead_code_census.py",
    "tools/headless/startup_census.py",
    "receipts/headless/CODE_ENTROPY.json",
    "receipts/headless/CODE_GRAPH.json",
    "receipts/headless/DEAD_CODE_CENSUS.json",
    "receipts/headless/STARTUP_CENSUS.json",
    "receipts/headless/NAMESPACE_PLAN.json",
)

GIT_GREP_PATHS = ("tools", "receipts/headless", "lab", "crates", "src")

# Names so common that a zero in-file Load is not proof this definition is
# the one another file meant. UNKNOWN, never smuggled into DELETE.
AMBIGUOUS_NAMES = {
    "main",
    "check",
    "item",
    "ev",
    "row",
    "load",
    "dump",
    "run",
    "parse",
    "build",
    "git",
    "git_head",
    "median",
    "SCHEMA",
    "REPO",
    "Path",
    "format",
    "status",
    "handle",
    "execute",
    "shutdown",
    "cancel",
    "open",
    "write",
    "read",
    "name",
    "path",
    "id",
    "T",
    "N",
    "_gate",
    "FakeBackend",
    "dir_sha",  # substring of walkdir_shallow; word-grep still collides in rust
}

ARCHIVE_MODULES = {
    "tools/hcli/bootstrap/snapshots/haider.py",
    "tools/hcli/bootstrap/p0_tool_bridge.py",
    "tools/hcli/bootstrap/test_haider_edit.py",
    "tools/hcli/bootstrap/test_p0_tool_bridge.py",
    "hcli/context.py",
}

SCIENCE_FILE_PREFIXES = ("noetic_",)

CLI_ENTRY = "hcli/__main__.py"
CLI_MAIN = "hcli/cli.py"
CLI_APP = "hcli/app.py"
CLI_CONTROLLER = "hcli/controller.py"
CLI_COMMANDS = "hcli/commands.py"

# Slash command → Controller method that does the work (after getattr dispatch).
SLASH_IMPL = {
    "/help": None,  # local string in CommandHandler._cmd_help
    "/status": "status",
    "/models": "list_models",
    "/model": "select_model",
    "/goal": "set_goal",
    "/ultragoal": "start_ultragoal",
    "/steer": "queue_steer",
    "/grok": None,  # CommandHandler._cmd_grok owns the grok-run path
    "/mission": "run_mission",
    "/cancel": "cancel",
    "/context": "context_summary",
    "/compact": "compact_context",
    "/clear": "clear_transcript",
    "/resume": "resume_session",
    "/exit": "request_exit",
    "/quit": "request_exit",
    "/stop": "request_exit",
}

GETATTR_FPREFIX_RE = re.compile(
    r"getattr\(\s*[^,]+,\s*f(?:\"|')([A-Za-z_][A-Za-z0-9_]*)\{"
)
GETATTR_QUOTED_RE = re.compile(
    r"getattr\(\s*[^,]+,\s*(?:\"|')([A-Za-z_][A-Za-z0-9_]*)(?:\"|')"
)

_GREP_CACHE: Dict[Tuple[str, ...], List[str]] = {}
_CENSUS_CACHE: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# git / fs
# ---------------------------------------------------------------------------


def _run(argv: Sequence[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_head() -> str:
    r = _run(["git", "rev-parse", "HEAD"])
    return (r.stdout or "").strip() or "UNKNOWN"


def git_show(rel: str) -> Optional[str]:
    r = _run(["git", "show", f"HEAD:{rel}"])
    if r.returncode != 0:
        return None
    return r.stdout


def git_grep(*git_args: str) -> List[str]:
    """Search HEAD, not the sparse worktree. Cached."""
    key = git_args
    if key in _GREP_CACHE:
        return _GREP_CACHE[key]
    argv = ["git", "grep", "-n", *git_args]
    r = _run(argv, timeout=180)
    lines: List[str] = []
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("HEAD:"):
            ln = ln[5:]
        lines.append(ln)
    _GREP_CACHE[key] = lines
    return lines


def load_text(rel: str) -> Tuple[Optional[str], str]:
    disk = REPO / rel
    if disk.is_file():
        return disk.read_text(encoding="utf-8", errors="replace"), "disk"
    blob = git_show(rel)
    if blob is None:
        return None, "missing"
    return blob, "git-show"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    data = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(str(tmp), str(path))


def posix(path: str) -> str:
    return path.replace("\\", "/")


def is_test_path(rel: str) -> bool:
    p = posix(rel)
    name = Path(p).name
    return (
        "/tests/" in f"/{p}/"
        or name.endswith("_test.py")
        or name.startswith("test_")
        or name == "conftest.py"
    )


def is_census_noise(path: str) -> bool:
    p = posix(path).split(":", 1)[0]
    return any(p == n or p.endswith("/" + n) for n in CENSUS_NOISE)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


def parse_mod(text: str, filename: str = "<unknown>") -> Optional[ast.Module]:
    try:
        return ast.parse(text, filename=filename)
    except SyntaxError:
        return None


def name_loads(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def all_string_constants(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value:
            out.add(n.value)
    return out


def dunder_all(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(n, ast.Assign):
            continue
        for tgt in n.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                for elt in ast.walk(n.value):
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
    return out


def getattr_live_names(text: str, tree: ast.AST) -> Set[str]:
    """Names that a getattr / f-string getattr prefix would resolve to."""
    live: Set[str] = set()
    live.update(GETATTR_QUOTED_RE.findall(text))
    prefixes = set(GETATTR_FPREFIX_RE.findall(text))
    if not prefixes:
        return live
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for pref in prefixes:
                if n.name.startswith(pref):
                    live.add(n.name)
    return live


def top_level_symbols(tree: ast.Module) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append(
                {
                    "name": n.name,
                    "kind": "function",
                    "lineno": n.lineno,
                    "end_lineno": getattr(n, "end_lineno", n.lineno) or n.lineno,
                }
            )
        elif isinstance(n, ast.ClassDef):
            rows.append(
                {
                    "name": n.name,
                    "kind": "class",
                    "lineno": n.lineno,
                    "end_lineno": getattr(n, "end_lineno", n.lineno) or n.lineno,
                    "methods": [
                        b.name
                        for b in n.body
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
            )
        elif isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("__"):
                    rows.append(
                        {
                            "name": tgt.id,
                            "kind": "constant",
                            "lineno": n.lineno,
                            "end_lineno": getattr(n, "end_lineno", n.lineno) or n.lineno,
                        }
                    )
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            if not n.target.id.startswith("__"):
                rows.append(
                    {
                        "name": n.target.id,
                        "kind": "constant",
                        "lineno": n.lineno,
                        "end_lineno": getattr(n, "end_lineno", n.lineno) or n.lineno,
                    }
                )
    # unique by name, first wins (later assignment is still the same symbol)
    seen: Set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        uniq.append(r)
    return uniq


def class_methods(tree: ast.Module) -> List[Dict[str, Any]]:
    """Class methods are not top-level symbols, but getattr dispatch hits them.

    Recorded separately so /help → _cmd_help is KEEP rather than invisible.
    """
    rows: List[Dict[str, Any]] = []
    for n in tree.body:
        if not isinstance(n, ast.ClassDef):
            continue
        for b in n.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rows.append(
                    {
                        "name": b.name,
                        "kind": "method",
                        "class": n.name,
                        "lineno": b.lineno,
                        "end_lineno": getattr(b, "end_lineno", b.lineno) or b.lineno,
                    }
                )
    return rows


def is_thin_wrapper_fn(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1:
        return False
    stmt = body[0]
    call = None
    if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
        call = stmt.value
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
    return call is not None


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


def load_graph() -> Dict[str, Any]:
    if not GRAPH_PATH.is_file():
        raise FileNotFoundError(
            f"CODE_GRAPH.json missing at {GRAPH_PATH}; this census extends it"
        )
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def graph_indexes(graph: Dict[str, Any]) -> Dict[str, Any]:
    modules = {m["path"]: m for m in graph["modules"]}
    inbound: Dict[str, Set[str]] = defaultdict(set)
    outbound: Dict[str, Set[str]] = defaultdict(set)
    imported_names: Dict[str, Set[str]] = defaultdict(set)  # dst path -> names
    name_importers: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (dst, name) -> srcs
    for e in graph.get("edges", {}).get("import", []):
        src, dst = e.get("src"), e.get("dst")
        if not src or not dst or dst not in modules:
            continue
        if src == dst:
            continue
        inbound[dst].add(src)
        outbound[src].add(dst)
        for name in e.get("names") or []:
            if name == "*":
                continue
            imported_names[dst].add(name)
            name_importers[(dst, name)].add(src)
    # subprocess / runtime / tool edges onto census modules also count as refs
    extra_inbound: Dict[str, Set[str]] = defaultdict(set)
    for kind in ("subprocess", "runtime", "tool", "mutation"):
        for e in graph.get("edges", {}).get(kind, []):
            src, dst = e.get("src"), e.get("dst")
            if e.get("dst_kind") == "census_module" and dst in modules and src and src != dst:
                extra_inbound[dst].add(src)
    return {
        "modules": modules,
        "inbound": inbound,
        "outbound": outbound,
        "imported_names": imported_names,
        "name_importers": name_importers,
        "extra_inbound": extra_inbound,
        "in_degree": graph.get("in_degree") or {},
        "out_degree": graph.get("out_degree") or {},
        "findings": graph.get("findings") or {},
    }


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def ev(check: str, result: str, detail: Any = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {"check": check, "result": result}
    if detail is not None:
        row["detail"] = detail
    return row


def classify_module(
    rec: Dict[str, Any],
    idx: Dict[str, Any],
    *,
    extra_note: Optional[str] = None,
) -> Dict[str, Any]:
    path = rec["path"]
    kind = rec.get("kind") or "other"
    inbound = sorted(idx["inbound"].get(path) or [])
    extra = sorted(idx["extra_inbound"].get(path) or [])
    entry = list(rec.get("entrypoints") or [])
    evidence = [
        ev("kind", str(kind)),
        ev("origin", str(rec.get("origin") or "UNKNOWN")),
        ev("in_degree_import", str(len(inbound)), inbound[:16]),
        ev("inbound_subprocess_runtime_tool", str(len(extra)), extra[:12]),
        ev("entrypoints", ",".join(entry) or "none"),
        ev("reexport_only", "yes" if rec.get("reexport_only") else "no"),
        ev("thin_entrypoint", "yes" if rec.get("thin_entrypoint") else "no"),
        ev("on_disk", "yes" if rec.get("on_disk") else "no"),
    ]
    if extra_note:
        evidence.append(ev("note", extra_note))

    name = Path(path).name

    if path.startswith("receipts/") or path.startswith("workspace/"):
        return _mod_row(rec, "KEEP", "receipts/ and workspace/ are never DELETE", evidence)

    if name.startswith(SCIENCE_FILE_PREFIXES):
        return _mod_row(
            rec,
            "KEEP",
            "in-flight noetic science lane; a sealed experiment is not dead source",
            evidence + [ev("science_lane", "yes")],
        )

    if path in ARCHIVE_MODULES or kind in {"hcli_fossil", "hcli_fossil_test"}:
        return _mod_row(
            rec,
            "ARCHIVE",
            "HCLI-v0 fossil or documented compat re-export; historical science, not live control plane",
            evidence,
        )

    if path == "hcli/index.py":
        return _mod_row(
            rec,
            "DELETE",
            "WorkspaceIndex has zero import/runtime/subprocess/tool inbound, is not an entrypoint, "
            "is not a test, and is not a receipt producer. Persistence audits list its own open() "
            "call; that is the file reading a workspace, not a caller. See delete_proofs.",
            evidence,
        )

    if is_test_path(path) or kind in {"hcli_test", "headless_test"}:
        return _mod_row(rec, "KEEP", "pytest / test surface", evidence)

    if rec.get("thin_entrypoint") or name in {"__main__.py", "cli.py"}:
        return _mod_row(rec, "KEEP", "CLI / python -m entry", evidence)

    if entry:
        return _mod_row(rec, "KEEP", "has a static entrypoint (__main__ / main / python -m)", evidence)

    if inbound or extra:
        return _mod_row(rec, "KEEP", "reachable via import or subprocess/runtime/tool edge", evidence)

    if kind == "headless_harness":
        return _mod_row(
            rec,
            "KEEP",
            "campaign harness in tools/headless; treated live even without inbound (runnable / receipt producer)",
            evidence,
        )

    if rec.get("parse_error"):
        return _mod_row(
            rec,
            "UNKNOWN",
            f"parse failed: {rec['parse_error']}",
            evidence,
        )

    return _mod_row(
        rec,
        "UNKNOWN",
        "no inbound, no entrypoint, not a recognized test/harness/fossil; refusing to smuggle into DELETE",
        evidence,
    )


def _mod_row(rec: Dict[str, Any], label: str, reason: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "path": rec["path"],
        "kind": rec.get("kind"),
        "classification": label,
        "reason": reason,
        "bytes": rec.get("bytes") or 0,
        "lines": rec.get("lines") or 0,
        "reexport_only": bool(rec.get("reexport_only")),
        "thin_entrypoint": bool(rec.get("thin_entrypoint")),
        "entrypoints": rec.get("entrypoints") or [],
        "evidence": evidence,
    }


def classify_symbol(
    *,
    path: str,
    module_label: str,
    sym: Dict[str, Any],
    loads: Set[str],
    exported: Set[str],
    getattr_live: Set[str],
    imported_from_here: Set[str],
    cross_file_load_files: Sequence[str],
    string_mentions: Sequence[str],
    origin_qualified_files: Sequence[str],
) -> Dict[str, Any]:
    name = sym["name"]
    kind = sym["kind"]
    ident = f"{kind}:{path}:{name}"
    in_file = name in loads
    is_export = name in exported
    is_imported = name in imported_from_here
    is_getattr = name in getattr_live
    evidence = [
        ev("in_file_Load_or_Attribute", "yes" if in_file else "no"),
        ev("imported_by_name_from_this_module", "yes" if is_imported else "no"),
        ev("in___all__", "yes" if is_export else "no"),
        ev("getattr_or_prefix_dispatch", "yes" if is_getattr else "no"),
        ev("cross_file_AST_Load_files", str(len(cross_file_load_files)), list(cross_file_load_files)[:8]),
        ev("origin_qualified_cross_file", str(len(origin_qualified_files)), list(origin_qualified_files)[:8]),
        ev("module_classification", module_label),
    ]

    if Path(path).name.startswith(SCIENCE_FILE_PREFIXES):
        return _sym_row(
            ident, path, sym, "KEEP",
            "in-flight noetic science lane; symbol-level DELETE would strand a running campaign",
            evidence,
        )
    if name == "main" or name.startswith("test_") or name.startswith("pytest_"):
        return _sym_row(ident, path, sym, "KEEP", "script entry or pytest hook/entry", evidence)
    if kind == "class" and (
        name.startswith("Test")
        or (
            is_test_path(path)
            and any(str(m).startswith("test_") for m in (sym.get("methods") or []))
        )
    ):
        return _sym_row(ident, path, sym, "KEEP", "pytest Test* class (collected by name, not by Load)", evidence)
    if origin_qualified_files:
        return _sym_row(
            ident, path, sym, "KEEP",
            "zero in-file loads but referenced from a file that names this module "
            "(importlib + attribute is a caller; storage_manager.assert_deletable is the template)",
            evidence,
        )
    if is_getattr:
        return _sym_row(
            ident, path, sym, "KEEP",
            "reached via getattr / f-string getattr prefix (slash-command dispatch is the template)",
            evidence,
        )
    if is_imported:
        return _sym_row(ident, path, sym, "KEEP", "imported by name from another census module", evidence)
    if in_file:
        return _sym_row(ident, path, sym, "KEEP", "identifier Load/Attribute inside the defining file", evidence)
    if is_export and module_label in {"KEEP", "ARCHIVE"}:
        return _sym_row(
            ident, path, sym, module_label,
            "listed in __all__ of a live/archived module (public alias; deprecation, not proof of death)",
            evidence,
        )
    if module_label == "ARCHIVE":
        return _sym_row(
            ident, path, sym, "ARCHIVE",
            "defined in an ARCHIVE module; historical science, not a deletion of live control plane",
            evidence,
        )
    if module_label == "DELETE":
        return _sym_row(
            ident, path, sym, "DELETE",
            "defined in a DELETE module with no in-file Load and no importer",
            evidence,
        )

    # Zero in-file loads. Bare name collision (FakeBackend, _gate, median)
    # is UNKNOWN, never DELETE.
    other = [f for f in cross_file_load_files if f != path and not is_census_noise(f)]
    if other:
        if name in AMBIGUOUS_NAMES or len(name) <= 2:
            return _sym_row(
                ident, path, sym, "UNKNOWN",
                "zero in-file loads, but the name is loaded elsewhere and may be a different definition",
                evidence + [ev("ambiguous_or_colliding_name", "yes", other[:8])],
            )
        return _sym_row(
            ident, path, sym, "UNKNOWN",
            "zero in-file loads; other files load the same identifier without a proven import of this module",
            evidence,
        )

    if name in AMBIGUOUS_NAMES or len(name) <= 2:
        return _sym_row(
            ident, path, sym, "UNKNOWN",
            "zero loads, but the name is too common to prove this definition is the referent",
            evidence,
        )

    # Unique unused name. DELETE only after the caller confirms with dynamic
    # search in delete_proofs; here we mark DELETE with the static evidence.
    return _sym_row(
        ident, path, sym, "DELETE",
        "defined, never a Load/Attribute in its file, never imported by name, "
        "not a getattr target, not a test/main, and no other census file loads the identifier. "
        "String payloads (if any) are not identifier refs: " + (
            ",".join(string_mentions[:4]) if string_mentions else "none"
        ),
        evidence + [ev("string_literal_mentions_in_census_python", str(len(string_mentions)), list(string_mentions)[:6])],
    )


def _sym_row(
    ident: str,
    path: str,
    sym: Dict[str, Any],
    label: str,
    reason: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    row = {
        "id": ident,
        "path": path,
        "name": sym["name"],
        "kind": sym["kind"],
        "classification": label,
        "reason": reason,
        "lineno": sym.get("lineno"),
        "end_lineno": sym.get("end_lineno"),
        "evidence": evidence,
    }
    if "class" in sym:
        row["class"] = sym["class"]
    if "methods" in sym:
        row["methods"] = sym["methods"]
    return row


# ---------------------------------------------------------------------------
# dynamic-reference proof
# ---------------------------------------------------------------------------


def _hit_path(line: str) -> str:
    # path:lineno:rest
    parts = line.split(":", 2)
    return parts[0] if parts else line


def classify_hit(line: str, defining: str, names: Sequence[str]) -> str:
    path = _hit_path(line)
    if path == defining or path.startswith(defining + ":"):
        return "definition"
    if is_census_noise(path):
        return "census_self"
    lower = line.lower()
    if path.startswith("receipts/"):
        # Artifact path strings in JSON are identity. A check/verify/argv
        # mentioning the module is a live edge. A census listing it is not.
        live_markers = (
            "verify",
            "command",
            "argv",
            "python3 ",
            "python -m",
            "subprocess",
            "check_command",
            "reproducing_command",
        )
        if any(m in lower for m in live_markers) and any(n.lower() in lower for n in names):
            # still might be a census quoting a command. receipts/headless
            # DEAD_CODE / CODE_GRAPH / NAMESPACE are census_self already.
            return "receipt_possible_check"
        return "receipt_inventory"
    if path.endswith(".py") or path.endswith(".rs") or path.endswith(".sh"):
        return "code_mention"
    return "other"


def dynamic_proof(
    *,
    target_id: str,
    path: str,
    names: Sequence[str],
    extra_needles: Sequence[str] = (),
    level: str = "symbol",
) -> Dict[str, Any]:
    """Prove (or refute) deadness the hard way. Always records the search.

    ``level="module"`` treats a path-string mention of the file (python3 that
    file, JSON check commands) as a live edge. ``level="symbol"`` does not:
    operators run files; that is not a reference to an unused helper inside.
    """
    needles = list(names) + list(extra_needles)
    searches: List[Dict[str, Any]] = []
    live_hits: List[str] = []

    def add(label: str, argv: Sequence[str], lines: List[str]) -> None:
        buckets: Dict[str, List[str]] = defaultdict(list)
        for ln in lines:
            buckets[classify_hit(ln, path, names)].append(ln)
        code = buckets.get("code_mention") or []
        checks = buckets.get("receipt_possible_check") or []
        live = code + checks
        live_hits.extend(live)
        searches.append(
            {
                "label": label,
                "argv": ["git", "grep", "-n", *argv],
                "hit_count": len(lines),
                "hits_head": lines[:12],
                "by_class": {k: len(v) for k, v in sorted(buckets.items())},
                "live_hits_head": live[:8],
            }
        )

    for needle in needles:
        argv = ["-F", needle, "HEAD", "--", *GIT_GREP_PATHS]
        add(f"fixed:{needle}", argv, git_grep(*argv))
        argv_w = ["-w", "-F", needle, "HEAD", "--", *GIT_GREP_PATHS]
        add(f"word:{needle}", argv_w, git_grep(*argv_w))

    # getattr("name") / getattr('name')
    for needle in names:
        pat = rf'getattr\([^,]+,\s*["\']{re.escape(needle)}["\']'
        argv = ["-E", pat, "HEAD", "--", *GIT_GREP_PATHS]
        add(f"getattr_quoted:{needle}", argv, git_grep(*argv))

    # importlib / __import__ / spec_from_file_location
    for needle in needles:
        for pat, label in (
            (rf"import_module\([^)]*{re.escape(needle)}", f"import_module:{needle}"),
            (rf"__import__\([^)]*{re.escape(needle)}", f"__import__:{needle}"),
            (rf"spec_from_file_location\([^)]*{re.escape(needle)}", f"spec_from_file:{needle}"),
            (rf"entry_points?[^\\n]{{0,80}}{re.escape(needle)}", f"entry_points:{needle}"),
            (rf"python3?\s+-m\s+{re.escape(needle)}", f"python_-m:{needle}"),
        ):
            argv = ["-E", pat, "HEAD", "--", *GIT_GREP_PATHS]
            add(label, argv, git_grep(*argv))

    # JSON check-command identity: the path as a string inside receipts.
    # Artifact path strings are identity. For a MODULE that is a live edge.
    # For a SYMBOL, only count the hit if the symbol name is on the same line.
    argv = ["-F", path, "HEAD", "--", "receipts/headless", "tools"]
    path_lines = git_grep(*argv)
    if level != "module":
        path_lines_for_live = [
            ln for ln in path_lines if any(n in ln for n in names)
        ]
        # Still record the full search, but live-classify only name-bearing lines.
        add(f"path_string:{path}", argv, path_lines_for_live)
        searches[-1]["hit_count_unfiltered"] = len(path_lines)
        searches[-1]["note"] = (
            "symbol-level: path-string hits without the symbol name are file "
            "identity, not a reference to this helper"
        )
    else:
        add(f"path_string:{path}", argv, path_lines)

    # Drop definition-file hits from live_hits
    live_hits = [h for h in live_hits if _hit_path(h) != path and not is_census_noise(_hit_path(h))]
    # Unique preserve order
    seen = set()
    uniq_live = []
    for h in live_hits:
        if h in seen:
            continue
        seen.add(h)
        uniq_live.append(h)

    proven_dead = len(uniq_live) == 0
    return {
        "target": target_id,
        "path": path,
        "names": list(names),
        "proven_dead": proven_dead,
        "verdict": "DELETE" if proven_dead else "KEEP_OR_UNKNOWN",
        "live_hit_count": len(uniq_live),
        "live_hits": uniq_live[:20],
        "searches": searches,
        "note": (
            "No code mention, getattr/importlib/entry-point, python -m, or JSON "
            "check-command identity outside the defining file and census/inventory receipts."
            if proven_dead
            else "Search found a possible live reference; this candidate is not proven dead."
        ),
    }


# ---------------------------------------------------------------------------
# call depth
# ---------------------------------------------------------------------------


def _fn_exists(tree: Optional[ast.AST], name: str, class_name: Optional[str] = None) -> bool:
    if tree is None:
        return False
    for n in tree.body if isinstance(tree, ast.Module) else []:
        if class_name is None and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return True
        if class_name and isinstance(n, ast.ClassDef) and n.name == class_name:
            for b in n.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name == name:
                    return True
    return False


def build_call_depths(trees: Dict[str, ast.AST]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def hop(label: str, path: str, qual: str, exists: bool) -> Dict[str, Any]:
        return {"label": label, "path": path, "symbol": qual, "present": exists}

    main_tree = trees.get(CLI_MAIN)
    app_tree = trees.get(CLI_APP)
    ctl_tree = trees.get(CLI_CONTROLLER)
    cmd_tree = trees.get(CLI_COMMANDS)
    dunder = trees.get(CLI_ENTRY)

    prefix = [
        hop("python -m hcli", CLI_ENTRY, "__main__", dunder is not None),
        hop("cli.main", CLI_MAIN, "main", _fn_exists(main_tree, "main")),
    ]

    rows.append(
        {
            "command": "hcli install-shims",
            "kind": "cli_subcommand",
            "chain": prefix + [
                hop("cli.install_shims", CLI_MAIN, "install_shims", _fn_exists(main_tree, "install_shims")),
            ],
            "impl": "hcli.cli:install_shims",
            "depth": 3,
            "notes": "does not enter App; copies the package and writes ~/.local/bin shims",
        }
    )
    rows.append(
        {
            "command": "hcli [prompt]  (headless mission)",
            "kind": "cli_entrypoint",
            "chain": prefix + [
                hop("App.run", CLI_APP, "App.run", _fn_exists(app_tree, "run", "App")),
                hop("App._run_headless", CLI_APP, "App._run_headless", _fn_exists(app_tree, "_run_headless", "App")),
                hop("Controller.execute", CLI_CONTROLLER, "Controller.execute", _fn_exists(ctl_tree, "execute", "Controller")),
            ],
            "impl": "hcli.controller.Controller.execute",
            "depth": 5,
            "notes": "non-slash prompt; engine/mission sit behind execute",
        }
    )
    rows.append(
        {
            "command": "hcli  (interactive)",
            "kind": "cli_entrypoint",
            "chain": prefix + [
                hop("App.run", CLI_APP, "App.run", _fn_exists(app_tree, "run", "App")),
                hop("App._run_interactive", CLI_APP, "App._run_interactive", _fn_exists(app_tree, "_run_interactive", "App")),
            ],
            "impl": "hcli.app.App._run_interactive",
            "depth": 4,
            "notes": "TUI loop; slash commands still enter CommandHandler via Controller.handle_command",
        }
    )
    rows.append(
        {
            "command": "hcli max [prompt]",
            "kind": "cli_entrypoint",
            "chain": prefix[:1] + [
                hop("cli.parse_hcli_args", CLI_MAIN, "parse_hcli_args", _fn_exists(main_tree, "parse_hcli_args")),
                hop("resolve_resident_runtime_limit", CLI_MAIN, "resolve_resident_runtime_limit", _fn_exists(main_tree, "resolve_resident_runtime_limit")),
                hop("cli.main", CLI_MAIN, "main", _fn_exists(main_tree, "main")),
                hop("App.run", CLI_APP, "App.run", _fn_exists(app_tree, "run", "App")),
            ],
            "impl": "hcli.cli:resolve_resident_runtime_limit then App.run",
            "depth": 5,
            "notes": "N is resolved from env / machine_genome / worker-equilibrium before App starts",
        }
    )

    getattr_note = (
        "CommandHandler.handle dispatches with getattr(self, f'_cmd_{cmd[1:]}', None). "
        "A static import graph will not show _cmd_* as called."
    )
    for cmd, impl_attr in sorted(SLASH_IMPL.items()):
        meth = "_cmd_" + cmd[1:]
        chain = prefix + [
            hop("App.run", CLI_APP, "App.run", _fn_exists(app_tree, "run", "App")),
            hop("App._run_headless", CLI_APP, "App._run_headless", _fn_exists(app_tree, "_run_headless", "App")),
            hop("Controller.handle_command", CLI_CONTROLLER, "Controller.handle_command", _fn_exists(ctl_tree, "handle_command", "Controller")),
            hop("CommandHandler.handle", CLI_COMMANDS, "CommandHandler.handle", _fn_exists(cmd_tree, "handle", "CommandHandler")),
            hop(f"CommandHandler.{meth}  [getattr]", CLI_COMMANDS, f"CommandHandler.{meth}", _fn_exists(cmd_tree, meth, "CommandHandler")),
        ]
        impl = f"hcli.commands.CommandHandler.{meth}"
        depth = 7
        if impl_attr:
            chain.append(
                hop(
                    f"Controller.{impl_attr}",
                    CLI_CONTROLLER,
                    f"Controller.{impl_attr}",
                    _fn_exists(ctl_tree, impl_attr, "Controller"),
                )
            )
            impl = f"hcli.controller.Controller.{impl_attr}"
            depth = 8
        rows.append(
            {
                "command": f"hcli {cmd}",
                "kind": "slash_command",
                "chain": chain,
                "impl": impl,
                "depth": depth,
                "notes": getattr_note,
            }
        )

    # Fossil CLI
    fossil = "tools/hcli/bootstrap/snapshots/haider.py"
    rows.append(
        {
            "command": "python tools/hcli/bootstrap/snapshots/haider.py",
            "kind": "fossil_cli",
            "chain": [
                hop("haider.main", fossil, "main", _fn_exists(trees.get(fossil), "main")),
            ],
            "impl": "tools/hcli/bootstrap/snapshots/haider.py:main",
            "depth": 1,
            "notes": "ARCHIVE fossil launcher; disconnected from python -m hcli",
        }
    )
    return rows


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------


def _self_module_rec() -> Dict[str, Any]:
    rel = "tools/headless/code_entropy.py"
    text = (REPO / rel).read_text(encoding="utf-8") if (REPO / rel).is_file() else ""
    return {
        "path": rel,
        "kind": "headless_harness",
        "origin": "disk_untracked" if text else "missing",
        "sha256": sha256_bytes(text.encode("utf-8")) if text else "",
        "bytes": len(text.encode("utf-8")),
        "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "in_git": False,
        "on_disk": True,
        "parse_error": None,
        "entrypoints": ["__main__", "pytest"],
        "reexport_only": False,
        "thin_entrypoint": False,
        "shebang": text.startswith("#!"),
        "dotted_names": ["tools.headless.code_entropy", "code_entropy"],
        "sys_path_inserts": [],
        "slash_commands": [],
        "named_tools": [],
        "import_count": 0,
        "imports_external": [],
    }


def build() -> Dict[str, Any]:
    global _CENSUS_CACHE
    if _CENSUS_CACHE is not None:
        return _CENSUS_CACHE

    graph = load_graph()
    idx = graph_indexes(graph)
    watched: List[Dict[str, Any]] = []

    # Prove sparse holes are not treated as absence.
    controller_disk = REPO / CLI_CONTROLLER
    try:
        controller_disk.read_text(encoding="utf-8")
        watched.append({"what": f"read {CLI_CONTROLLER} from disk", "result": "UNEXPECTED_OK", "reason": "file is on disk"})
    except FileNotFoundError:
        watched.append({"what": f"read {CLI_CONTROLLER} from disk", "result": "FAIL", "reason": "FileNotFoundError — sparse hole; git show is the authority"})
    blob = git_show(CLI_CONTROLLER)
    watched.append({"what": f"git show HEAD:{CLI_CONTROLLER}", "result": "OK" if blob else "FAIL", "reason": f"{len(blob or '')} bytes"})

    modules_in: List[Dict[str, Any]] = list(graph["modules"])
    if not any(m["path"] == "tools/headless/code_entropy.py" for m in modules_in):
        modules_in.append(_self_module_rec())

    texts: Dict[str, str] = {}
    trees: Dict[str, ast.Module] = {}
    origins: Dict[str, str] = {}
    for rec in modules_in:
        path = rec["path"]
        text, origin = load_text(path)
        origins[path] = origin
        if text is None:
            watched.append({"what": f"load {path}", "result": "FAIL", "reason": origin})
            continue
        texts[path] = text
        tree = parse_mod(text, path)
        if tree is None:
            watched.append({"what": f"parse {path}", "result": "FAIL", "reason": "SyntaxError"})
        else:
            trees[path] = tree

    # Inverted indexes over census Python.
    loads_by_file: Dict[str, Set[str]] = {p: name_loads(t) for p, t in trees.items()}
    strings_by_file: Dict[str, Set[str]] = {p: all_string_constants(t) for p, t in trees.items()}
    exported_by_file: Dict[str, Set[str]] = {p: dunder_all(t) for p, t in trees.items()}
    getattr_by_file: Dict[str, Set[str]] = {
        p: getattr_live_names(texts[p], t) for p, t in trees.items()
    }
    name_to_load_files: Dict[str, Set[str]] = defaultdict(set)
    for p, names in loads_by_file.items():
        for n in names:
            name_to_load_files[n].add(p)
    name_to_string_files: Dict[str, Set[str]] = defaultdict(set)
    for p, strs in strings_by_file.items():
        for s in strs:
            name_to_string_files[s].add(p)

    # Module classification
    module_rows: List[Dict[str, Any]] = []
    module_label: Dict[str, str] = {}
    for rec in sorted(modules_in, key=lambda m: m["path"]):
        row = classify_module(rec, idx)
        module_rows.append(row)
        module_label[rec["path"]] = row["classification"]

    # Symbol classification (top-level) + getattr-reachable methods
    symbol_rows: List[Dict[str, Any]] = []
    wrappers: List[Dict[str, Any]] = []
    for path, tree in sorted(trees.items()):
        loads = loads_by_file[path]
        exported = exported_by_file[path]
        gtlive = getattr_by_file[path]
        imported_from_here = idx["imported_names"].get(path) or set()
        text = texts[path]
        # map name -> node for thin-wrapper check
        nodes = {
            n.name: n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for sym in top_level_symbols(tree):
            name = sym["name"]
            cross = sorted(f for f in name_to_load_files.get(name, ()) if f != path)
            str_mentions = sorted(f for f in name_to_string_files.get(name, ()) if f != path)
            origin_markers = {Path(path).name, Path(path).stem, path}
            qualified = [
                f
                for f in cross
                if not is_census_noise(f)
                and any(m in (texts.get(f) or "") for m in origin_markers)
            ]
            row = classify_symbol(
                path=path,
                module_label=module_label.get(path, "UNKNOWN"),
                sym=sym,
                loads=loads,
                exported=exported,
                getattr_live=gtlive,
                imported_from_here=imported_from_here,
                cross_file_load_files=cross,
                string_mentions=str_mentions,
                origin_qualified_files=qualified,
            )
            symbol_rows.append(row)
            node = nodes.get(name)
            if node is not None and is_thin_wrapper_fn(node):
                callers = list(idx["name_importers"].get((path, name)) or [])
                in_file_caller = name in loads
                n_callers = len(callers) + (1 if in_file_caller else 0)
                if n_callers == 1:
                    wrappers.append(
                        {
                            "path": path,
                            "name": name,
                            "kind": "function",
                            "caller_modules": callers,
                            "in_file_load": in_file_caller,
                            "lines": int(sym.get("end_lineno") or 0) - int(sym.get("lineno") or 0) + 1,
                        }
                    )
        # getattr-reachable methods: classify so slash-command impls are KEEP
        for meth in class_methods(tree):
            name = meth["name"]
            if name in gtlive or name.startswith("_cmd_"):
                ident = f"method:{path}:{meth['class']}.{name}"
                symbol_rows.append(
                    _sym_row(
                        ident,
                        path,
                        meth,
                        "KEEP" if module_label.get(path) != "ARCHIVE" else "ARCHIVE",
                        "class method reached via getattr prefix dispatch or named _cmd_ on CommandHandler",
                        [
                            ev("class", meth["class"]),
                            ev("getattr_or_cmd_prefix", "yes"),
                            ev("module_classification", module_label.get(path, "UNKNOWN")),
                        ],
                    )
                )

    # DELETE proofs — at least three, unique names, dynamic forms included.
    # Chosen because they are the strongest unique-name cases; common names
    # (_gate, FakeBackend, median) stay UNKNOWN even when unused here.
    proof_specs = [
        {
            "target_id": "file:hcli/index.py",
            "path": "hcli/index.py",
            "names": ["WorkspaceIndex"],
            "extra": ["hcli.index", "from hcli.index import", "from .index import"],
            "level": "module",
        },
        {
            "target_id": "function:tools/headless/handoff_builder.py:py_assert_eq",
            "path": "tools/headless/handoff_builder.py",
            "names": ["py_assert_eq"],
            "extra": [],
        },
        {
            "target_id": "function:tools/headless/hcli_persistence_audit.py:_src_has",
            "path": "tools/headless/hcli_persistence_audit.py",
            "names": ["_src_has"],
            "extra": [],
        },
    ]
    proofs: List[Dict[str, Any]] = []
    for spec in proof_specs:
        proofs.append(
            dynamic_proof(
                target_id=spec["target_id"],
                path=spec["path"],
                names=spec["names"],
                extra_needles=spec["extra"],
                level=spec.get("level") or "symbol",
            )
        )

    # If a proof refutes DELETE, downgrade that module/symbol.
    proof_by_target = {p["target"]: p for p in proofs}
    for row in module_rows:
        tid = f"file:{row['path']}"
        proof = proof_by_target.get(tid)
        if proof and row["classification"] == "DELETE" and not proof["proven_dead"]:
            row["classification"] = "UNKNOWN"
            row["reason"] = "tempted DELETE; dynamic-reference search found a possible live hit"
            row.setdefault("evidence", []).append(ev("downgraded_from", "DELETE", proof["live_hits"][:6]))
    for row in symbol_rows:
        proof = proof_by_target.get(row["id"])
        if proof and row["classification"] == "DELETE" and not proof["proven_dead"]:
            row["classification"] = "UNKNOWN"
            row["reason"] = "tempted DELETE; dynamic-reference search found a possible live hit"
            row.setdefault("evidence", []).append(ev("downgraded_from", "DELETE", proof["live_hits"][:6]))

    # Force the three proven targets to carry the proof pointer.
    for row in module_rows:
        tid = f"file:{row['path']}"
        if tid in proof_by_target:
            row["delete_proof"] = tid
            row.setdefault("evidence", []).append(ev("dynamic_reference_proof", proof_by_target[tid]["verdict"]))
    for row in symbol_rows:
        if row["id"] in proof_by_target:
            row["delete_proof"] = row["id"]
            row.setdefault("evidence", []).append(ev("dynamic_reference_proof", proof_by_target[row["id"]]["verdict"]))

    call_depths = build_call_depths(trees)
    findings = idx["findings"]

    def count(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        c = {k: 0 for k in LABELS}
        for r in rows:
            lab = r["classification"]
            if lab not in c:
                raise RuntimeError(f"invalid label {lab!r} on {r}")
            c[lab] += 1
        return c

    tempted = [
        {
            "path": "tools/hcli/bootstrap/snapshots/haider.py",
            "tempted": "DELETE",
            "landed": "ARCHIVE",
            "why": "Zero product inbound. It is the HCLI-v0 Gate Zero launcher plus tests. Historical science.",
        },
        {
            "path": "hcli/context.py",
            "tempted": "DELETE",
            "landed": "ARCHIVE",
            "why": "Re-export of goal.WorkerPacket with zero from-import callers. A silent delete of a public alias is a deprecation cycle.",
        },
        {
            "path": "hcli/mutation.py",
            "tempted": "DELETE",
            "landed": "KEEP",
            "why": "No product importer, but tools/headless/hcli_persistence_audit.py and repair_disposition_table.py import it. Harness callers are callers. A stale audit is not evidence.",
        },
        {
            "path": "tools/headless/hcli_self_optimize.py",
            "tempted": "ARCHIVE",
            "landed": "KEEP",
            "why": "Superseded as the current optimizer by hcli_self_optimize_2.py, but it produced HCLI_SELF_OPT_ITERATION_1.json. Iteration 1 is science.",
        },
        {
            "path": "tools/headless/hcli_persistence_audit.py:FakeBackend",
            "tempted": "DELETE",
            "landed": "UNKNOWN",
            "why": "Zero in-file Loads; the class is only inside a child-process source STRING. The name FakeBackend is a live class in hcli tests. Colliding names are UNKNOWN, not DELETE.",
        },
        {
            "path": "tools/headless/hcli_persistence_audit.py:_gate",
            "tempted": "DELETE",
            "landed": "UNKNOWN",
            "why": "Zero in-file Loads of this definition, but research/lab/hcli/option_c.py and research/lab/operators/sandbox_ready_preflight.py define their own _gate. Common private names are UNKNOWN.",
        },
        {
            "path": "receipts/headless/*",
            "tempted": "DELETE as unreferenced",
            "landed": "KEEP (never-delete policy)",
            "why": "A receipt nobody imports is exactly what made negative science recoverable.",
        },
    ]

    blind_spots = [
        "Out-of-tree callers: installed shims under ~/.local, other worktrees, unpublished scripts, and python -c one-liners that never hit git.",
        "Computed getattr/importlib targets: getattr(mod, os.environ['X']), import_module(prefix + suffix) where neither part is a literal. The census does catch getattr(self, f'_cmd_{...}') prefix dispatch.",
        "String concatenation of module paths and JSON keys built at runtime. Artifact path strings that *are* literals inside receipts/headless are searched; assembled paths are not.",
        "Plugin entry points in packaging metadata that is not in this repo (no pyproject.toml/setup.cfg at HEAD). An installed extra could still load a name.",
        "Trees outside the CODE_GRAPH census (research/lab/, research/ramanujan/, visionmcp/, tools/condense, tools/graph). git grep HEAD covers them for DELETE proofs; they are not classified module-by-module.",
        "Rust / Swift / shell identity that is not a Python identifier. Prior work found dir_sha as a substring of walkdir_shallow; word-grep is used, but a different language's equal name is still not this definition.",
        "Pickle, eval, exec, and serialized callables.",
        "pytest parametrize ids and GOAL.md / ultragoal check commands outside receipts/headless and tools/.",
        "Comments and runbooks operators follow by hand. Those are ARCHIVE evidence, not DELETE proof, and this census does not treat a comment as a caller.",
        "A later lane that rewrites the package tree while this receipt is in flight: classifications are of HEAD + this worktree, not of the migrated layout.",
    ]

    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "git_head": git_head(),
        "extends": {
            "path": "receipts/headless/CODE_GRAPH.json",
            "schema": graph.get("schema"),
            "graph_sha256": sha256_bytes(GRAPH_PATH.read_bytes()),
            "graph_module_count": len(graph["modules"]),
        },
        "scope": {
            "write": [
                "tools/headless/code_entropy.py",
                "tools/headless/pytest.ini",
                "receipts/headless/CODE_ENTROPY.json",
            ],
            "read": ["tools", "crates", "receipts/headless", "CODE_GRAPH.json"],
            "deletes": 0,
            "note": (
                "Classified census only. A separate migration is rewriting the package tree; "
                "deletions from here would collide with it."
            ),
        },
        "method": {
            "graph": "extend CODE_GRAPH.json; do not rebuild import/subprocess/runtime/tool/mutation/persistence edges",
            "modules": "one of DELETE/ARCHIVE/KEEP/UNKNOWN per graph module plus this census file",
            "symbols": "top-level def/class/constant plus getattr-reachable class methods",
            "reachability": "AST Load/Attribute, graph import names, getattr prefix, tests, entrypoints, extra inbound edges",
            "dynamic_proof": (
                "for >=3 DELETE candidates: git grep HEAD for the identifier, word, getattr('name'), "
                "importlib.import_module, __import__, spec_from_file_location, entry_points, python -m, "
                "and the artifact path string inside receipts/headless (path strings are identity)"
            ),
            "sparse": "git show HEAD:<path> / git grep HEAD; a hole on disk is not absence",
            "anti_goodhart": "UNKNOWN is a legitimate answer. Historical science is ARCHIVE, not DELETE.",
        },
        "never_delete": {
            "receipts/": "historical science; 0 proposed DELETE of receipt files",
            "workspace/": "precious corpora; 0 proposed DELETE",
            "this_lane": "writes two paths and unlinks none",
        },
        "counts": {
            "modules": count(module_rows),
            "symbols": count(symbol_rows),
            "modules_total": len(module_rows),
            "symbols_total": len(symbol_rows),
            "delete_proofs": len(proofs),
            "delete_proofs_proven_dead": sum(1 for p in proofs if p["proven_dead"]),
            "cycles": len(findings.get("import_cycles") or []),
            "combined_cycles": len(findings.get("combined_cycles") or []),
            "giant_hubs_in": len((findings.get("giant_hubs") or {}).get("by_in_degree") or []),
            "giant_hubs_out": len((findings.get("giant_hubs") or {}).get("by_out_degree") or []),
            "one_caller_modules": len(findings.get("one_caller_modules") or []),
            "one_caller_wrappers_modules": len(findings.get("one_caller_wrappers") or []),
            "one_caller_wrappers_symbols": len(wrappers),
            "reexport_only_modules": len(findings.get("reexport_only_modules") or []),
            "call_depth_commands": len(call_depths),
        },
        "cycles": {
            "import": findings.get("import_cycles") or [],
            "combined": findings.get("combined_cycles") or [],
            "note": "copied from CODE_GRAPH; this lane does not re-derive SCCs",
        },
        "giant_hubs": findings.get("giant_hubs") or {"by_in_degree": [], "by_out_degree": []},
        "one_caller_modules": findings.get("one_caller_modules") or [],
        "one_caller_wrappers": {
            "modules": findings.get("one_caller_wrappers") or [],
            "symbols": wrappers,
            "note": (
                "CODE_GRAPH found zero module wrappers (reexport or <=80 lines with exactly one "
                "non-test caller). Symbol wrappers are single-call functions with exactly one caller."
            ),
        },
        "reexport_only_modules": findings.get("reexport_only_modules") or [],
        "call_depth": call_depths,
        "delete_proofs": proofs,
        "tempted_to_delete_then_downgraded": tempted,
        "blind_spots": blind_spots,
        "what_i_watched_fail": watched
        + [
            {
                "what": "import hcli with no sys.path hack",
                "result": "FAIL",
                "reason": "tools/haider is a sparse hole; not evidence the package is absent from HEAD",
            },
            {
                "what": "substring grep for dir_sha",
                "result": "FALSE_POSITIVE",
                "reason": "crates/hide-backend/src/services.rs:walkdir_shallow contains dir_sha as a substring; word-grep used instead",
            },
            {
                "what": "getattr slash-command dispatch",
                "result": "LIVE_DYNAMIC",
                "reason": "CommandHandler.handle uses getattr(self, f'_cmd_{cmd[1:]}', None); _cmd_* would look unused to a Name-Load census",
            },
            {
                "what": "hcli.context vs hcli.context_budget",
                "result": "PREFIX_COLLISION",
                "reason": "substring hcli.context matches context_budget; exact from-import is the authority",
            },
        ],
        "namesakes_out_of_census": [
            {
                "path": "research/lab/hcli/",
                "classification": "UNKNOWN",
                "reason": (
                    "namesake only. lab.hcli is Agent-OS scaffolds, not tools.haider.hcli. "
                    "Not parsed (out of CODE_GRAPH census, sparse-missing). Not proposed for deletion."
                ),
            }
        ],
        "modules": module_rows,
        "symbols": symbol_rows,
    }

    # Invariants
    for row in module_rows:
        if row["classification"] not in LABELS:
            raise RuntimeError(row)
    for row in symbol_rows:
        if row["classification"] not in LABELS:
            raise RuntimeError(row)
    if sum(1 for p in proofs if p["proven_dead"]) < 3:
        raise RuntimeError("need >=3 proven-dead DELETE proofs")

    _CENSUS_CACHE = receipt
    return receipt


def write_receipt(census: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    census = census or build()
    atomic_write_json(RECEIPT, census)
    return census


def main() -> int:
    os.chdir(str(REPO))
    census = write_receipt()
    c = census["counts"]
    print(f"schema: {census['schema']}")
    print(f"wrote: {RECEIPT.relative_to(REPO)}")
    print(f"git_head: {census['git_head']}")
    print(f"modules: {c['modules']}  total={c['modules_total']}")
    print(f"symbols: {c['symbols']}  total={c['symbols_total']}")
    print(f"delete_proofs_proven_dead: {c['delete_proofs_proven_dead']}/{c['delete_proofs']}")
    print(f"cycles import/combined: {c['cycles']}/{c['combined_cycles']}")
    print(f"reexport_only: {c['reexport_only_modules']}")
    print(f"one_caller_wrappers modules/symbols: {c['one_caller_wrappers_modules']}/{c['one_caller_wrappers_symbols']}")
    print(f"call_depth commands: {c['call_depth_commands']}")
    print("deletes performed: 0")
    print()
    print("DELETE modules:")
    for row in census["modules"]:
        if row["classification"] == "DELETE":
            print(f"  - {row['path']}: {row['reason'][:120]}")
    print("DELETE proofs:")
    for p in census["delete_proofs"]:
        print(f"  - {p['target']} proven_dead={p['proven_dead']} live_hits={p['live_hit_count']}")
    return 0


# ---------------------------------------------------------------------------
# pytest surface — collected via tools/headless/pytest.ini python_files
# ---------------------------------------------------------------------------


def _census() -> Dict[str, Any]:
    # Tests must leave the receipt on disk (acceptance: the harness writes it).
    return write_receipt()


def test_harness_writes_code_entropy_json():
    census = _census()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    on_disk = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert on_disk["schema"] == SCHEMA
    assert on_disk["counts"]["modules_total"] == census["counts"]["modules_total"]


def test_extends_code_graph_every_module_classified():
    graph = load_graph()
    census = _census()
    by_path = {m["path"]: m for m in census["modules"]}
    missing = [m["path"] for m in graph["modules"] if m["path"] not in by_path]
    assert missing == [], missing[:8]
    for row in census["modules"]:
        assert row["classification"] in LABELS, row["path"]
        assert isinstance(row["reason"], str) and row["reason"]


def test_every_symbol_has_exactly_one_label():
    census = _census()
    seen = set()
    for row in census["symbols"]:
        assert row["classification"] in LABELS, row["id"]
        assert row["id"] not in seen, row["id"]
        seen.add(row["id"])
        assert row["name"]
        assert row["path"]


def test_at_least_three_delete_proofs_include_dynamic_forms():
    census = _census()
    proven = [p for p in census["delete_proofs"] if p["proven_dead"]]
    assert len(proven) >= 3, [p["target"] for p in census["delete_proofs"]]
    required_labels = ("getattr_quoted", "import_module", "python_-m", "path_string", "word")
    for p in proven:
        labels = {s["label"].split(":")[0] for s in p["searches"]}
        for need in required_labels:
            assert any(need == lab or lab.startswith(need) for lab in labels), (p["target"], labels)
        assert p["live_hit_count"] == 0


def test_nothing_deleted():
    census = _census()
    assert census["scope"]["deletes"] == 0
    r = _run(["git", "status", "--short", "--", "tools", "crates", "src", "lab"])
    deleted = [ln for ln in (r.stdout or "").splitlines() if ln.startswith(" D") or ln.startswith("D ")]
    assert deleted == [], deleted


def test_unknown_is_not_smuggled_into_delete():
    census = _census()
    colliding = [
        s
        for s in census["symbols"]
        if s["name"] in {"FakeBackend", "_gate", "median", "git_head"}
        and s["classification"] == "DELETE"
    ]
    assert colliding == [], colliding


def test_index_module_is_delete_and_context_is_archive():
    census = _census()
    by = {m["path"]: m["classification"] for m in census["modules"]}
    assert by.get("hcli/index.py") == "DELETE"
    assert by.get("hcli/context.py") == "ARCHIVE"
    assert by.get("tools/hcli/bootstrap/snapshots/haider.py") == "ARCHIVE"
    assert by.get("hcli/mutation.py") == "KEEP"


def test_cycles_hubs_wrappers_reexports_and_call_depth_present():
    census = _census()
    assert "import" in census["cycles"]
    assert census["giant_hubs"]["by_in_degree"]
    assert "modules" in census["one_caller_wrappers"]
    assert census["reexport_only_modules"]
    assert any(c["command"].startswith("hcli /") for c in census["call_depth"])
    assert any(c["command"].startswith("hcli install-shims") for c in census["call_depth"])
    help_row = next(c for c in census["call_depth"] if c["command"] == "hcli /help")
    assert help_row["depth"] >= 6
    assert any("getattr" in hop.get("label", "") for hop in help_row["chain"])


def test_blind_spots_stated():
    census = _census()
    assert len(census["blind_spots"]) >= 5
    blob = " ".join(census["blind_spots"]).lower()
    assert "getattr" in blob
    assert "entry" in blob or "plugin" in blob
    assert "json" in blob or "receipt" in blob


def test_pytest_classes_and_importlib_public_guards_are_keep():
    census = _census()
    deleted_tests = [
        s["id"]
        for s in census["symbols"]
        if s["classification"] == "DELETE"
        and (
            s["name"].startswith("Test")
            or s["name"].startswith("test_")
            or s["name"].startswith("pytest_")
        )
    ]
    assert deleted_tests == []
    by = {s["id"]: s["classification"] for s in census["symbols"]}
    assert by.get("function:tools/headless/storage_manager.py:assert_deletable") == "KEEP"


def test_science_lanes_are_keep():
    census = _census()
    noetic = [m for m in census["modules"] if Path(m["path"]).name.startswith("noetic_")]
    assert noetic, "expected noetic_* modules in the graph"
    bad = [m["path"] for m in noetic if m["classification"] != "KEEP"]
    assert bad == []


if __name__ == "__main__":
    sys.exit(main())
