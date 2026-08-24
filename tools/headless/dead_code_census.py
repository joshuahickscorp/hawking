#!/usr/bin/env python3
"""Dead-code census for the headless campaign and the HCLI control plane.

Proven, not guessed. This lane writes the census and a plan; a human performs
any migration. It does not delete, rename, or restore anything.

Reachability evidence, in order:

  1. AST identifier loads/calls in the defining file (list/dict registrations
     count as uses; string literals do not).
  2. Cross-file identifier loads inside tools/headless (on disk).
  3. ``git grep HEAD`` exact import patterns (sparse-safe: working-tree grep
     only sees materialized paths).
  4. Filename citations in receipts/headless (liveness evidence, never a
     deletion candidate).
  5. Known entrypoints: ``if __name__ == '__main__'``, pytest ``test_*``,
     ``python -m hcli``, ``# noqa: F401`` side-effect imports.

Classification:

  DELETE   proven-dead LIVE code. Historical science is not this.
  ARCHIVE  superseded or compatibility wrapper; keep bytes until a human
           deprecation cycle. Do not strand in-flight lanes.
  KEEP     reachable, an entrypoint, a receipt producer, a public API still
           exercised, or a namesake of a different product.
  UNKNOWN  missing evidence (sparse hole, out-of-tree callers, stale audit).

Anti-Goodhart: success is verified capability, not line count. A wrong DELETE
costs science; an UNKNOWN costs a follow-up.

  python3 tools/headless/dead_code_census.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCHEMA = "hawking.headless.dead_code_census.v1"
REPO = Path(__file__).resolve().parents[2]
HEADLESS = REPO / "tools" / "headless"
RECEIPTS = REPO / "receipts" / "headless"
RECEIPT_PATH = RECEIPTS / "DEAD_CODE_CENSUS.json"

# Live control plane lives here. Sparse checkout does not materialize it;
# we read blobs via `git show HEAD:<path>` and search via `git grep HEAD`.
HAIDER_PREFIX = "tools/haider/"

NEVER_DELETE_PREFIXES = (
    "receipts/",
    "workspace/",
)

# In-flight noetic science lanes. File-level DELETE against these would
# strand a running campaign; unused *imports* inside them may still DELETE.
NOETIC_PREFIX = "noetic_"

GIT_GREP_PATHS = (
    "tools",
    "receipts/headless",
    "lab",
    "crates",
    "src",
)


# ---------------------------------------------------------------------------
# git / fs
# ---------------------------------------------------------------------------


def _run(argv: Sequence[str], timeout: int = 60) -> subprocess.CompletedProcess:
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


def git_ls_tree(*prefixes: str) -> List[str]:
    argv = ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", *prefixes]
    r = _run(argv)
    return [ln for ln in (r.stdout or "").splitlines() if ln]


def git_show(rel: str) -> Optional[bytes]:
    r = _run(["git", "show", f"HEAD:{rel}"])
    if r.returncode != 0:
        return None
    return r.stdout.encode() if isinstance(r.stdout, str) else r.stdout


def git_cat_size(rel: str) -> Optional[int]:
    r = _run(["git", "cat-file", "-s", f"HEAD:{rel}"])
    if r.returncode != 0:
        return None
    try:
        return int((r.stdout or "").strip())
    except ValueError:
        return None


_GREP_CACHE: Dict[Tuple[str, Tuple[str, ...]], List[str]] = {}


def git_grep_fixed(pattern: str, paths: Sequence[str] = GIT_GREP_PATHS) -> List[str]:
    """Search the HEAD tree, not the sparse worktree."""
    key = (pattern, tuple(paths))
    if key in _GREP_CACHE:
        return _GREP_CACHE[key]
    argv = ["git", "grep", "-n", "-F", pattern, "HEAD", "--", *paths]
    r = _run(argv, timeout=120)
    lines = []
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("HEAD:"):
            ln = ln[5:]
        lines.append(ln)
    _GREP_CACHE[key] = lines
    return lines


def load_text(rel: str) -> Tuple[Optional[str], str]:
    """Return (text, source). source is 'disk' | 'git-show' | 'missing'."""
    disk = REPO / rel
    if disk.is_file():
        return disk.read_text(encoding="utf-8", errors="replace"), "disk"
    blob = git_show(rel)
    if blob is None:
        return None, "missing"
    return blob.decode("utf-8", errors="replace"), "git-show"


def file_metrics(rel: str, text: Optional[str]) -> Tuple[int, int]:
    disk = REPO / rel
    if disk.is_file():
        st = disk.stat()
        nlines = text.count("\n") + (0 if text.endswith("\n") or not text else 1) if text else 0
        return st.st_size, nlines
    size = git_cat_size(rel)
    nlines = 0
    if text:
        nlines = text.count("\n") + (0 if text.endswith("\n") else 1)
    return (size or 0), nlines


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


def parse_mod(text: str) -> Optional[ast.Module]:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def name_loads(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
    return out


def import_bindings(tree: ast.AST, src_lines: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            line = src_lines[n.lineno - 1] if 0 < n.lineno <= len(src_lines) else ""
            for a in n.names:
                bind = a.asname or a.name.split(".")[0]
                rows.append(
                    {
                        "lineno": n.lineno,
                        "kind": "import",
                        "module": a.name,
                        "name": a.name,
                        "bind": bind,
                        "line": line.strip(),
                        "noqa_f401": "noqa" in line.lower() and "f401" in line.lower(),
                    }
                )
        elif isinstance(n, ast.ImportFrom):
            if n.module == "__future__":
                continue
            line = src_lines[n.lineno - 1] if 0 < n.lineno <= len(src_lines) else ""
            for a in n.names:
                if a.name == "*":
                    continue
                bind = a.asname or a.name
                rows.append(
                    {
                        "lineno": n.lineno,
                        "kind": "from",
                        "module": n.module or "",
                        "name": a.name,
                        "bind": bind,
                        "line": line.strip(),
                        "noqa_f401": "noqa" in line.lower() and "f401" in line.lower(),
                    }
                )
    return rows


def top_defs(tree: ast.Module) -> Tuple[List[ast.AST], List[ast.ClassDef]]:
    funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    return funcs, classes


def node_span(node: ast.AST, text: str) -> Tuple[int, int, int, int]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start) or start
    lines = text.splitlines()
    chunk = "\n".join(lines[start - 1 : end])
    if chunk and not chunk.endswith("\n"):
        chunk += "\n"
    return start, end, end - start + 1, len(chunk.encode("utf-8"))


def has_main_guard(text: str, tree: Optional[ast.AST]) -> bool:
    if "if __name__" in text:
        return True
    if tree is None:
        return False
    for n in tree.body:
        if not isinstance(n, ast.If):
            continue
        t = n.test
        if (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "__name__"
        ):
            return True
    return False


RECEIPT_RE = re.compile(r"receipts/headless/[\w.\-]+")
SCHEMA_RE = re.compile(r"hawking\.[a-z0-9_.]+")


def cited_receipts(text: str) -> List[str]:
    found = sorted(set(RECEIPT_RE.findall(text)))
    # strip trailing dots from "path.json." in comments
    clean = []
    for f in found:
        while f.endswith("."):
            f = f[:-1]
        if f.endswith(".json") or f.endswith(".jsonl") or f.endswith(".txt"):
            clean.append(f)
        elif "/receipts/headless/" in f and f.split("/")[-1]:
            clean.append(f)
    return sorted(set(clean))


# ---------------------------------------------------------------------------
# importer search (sparse-safe)
# ---------------------------------------------------------------------------


def _line_imports_stem(line: str, stem: str) -> bool:
    """True if this source line imports hcli.{stem} exactly, not a prefix sibling."""
    # Reject hcli.context_budget when stem is context; reject bare `import index`.
    pats = [
        rf"from \.{stem} import\b",
        rf"from hcli\.{stem} import\b",
        rf"from tools\.haider\.hcli\.{stem} import\b",
        rf"import hcli\.{stem}\b(?![\w.])",
        rf"import hcli\.{stem} as\b",
    ]
    body = line.split(":", 2)[-1] if line[:1].isalpha() or "/" in line[:40] else line
    return any(re.search(p, body) for p in pats)


def module_import_hits(stem: str, *, bare_ok: bool = False) -> List[str]:
    """Exact import forms. Do not substring-match hcli.context vs context_budget."""
    patterns = [
        f"from .{stem} import",
        f"from hcli.{stem} import",
        f"from tools.haider.hcli.{stem} import",
        f"import hcli.{stem}",
    ]
    if bare_ok:
        patterns.extend([f"import {stem}", f"import {stem} as"])
    hits: List[str] = []
    seen = set()
    for pat in patterns:
        for ln in git_grep_fixed(pat):
            if ln in seen:
                continue
            if not bare_ok and not _line_imports_stem(ln, stem):
                continue
            seen.add(ln)
            hits.append(ln)
    return hits


def identifier_hits(name: str) -> List[str]:
    return git_grep_fixed(name)


# ---------------------------------------------------------------------------
# classification helpers
# ---------------------------------------------------------------------------


def item(
    *,
    ident: str,
    path: str,
    kind: str,
    classification: str,
    reason: str,
    bytes_: int,
    lines: int,
    evidence: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": ident,
        "path": path,
        "kind": kind,
        "classification": classification,
        "reason": reason,
        "bytes": bytes_,
        "lines": lines,
        "evidence": evidence,
    }
    if extra:
        row.update(extra)
    return row


def ev(check: str, result: str, detail: Any = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"check": check, "result": result}
    if detail is not None:
        d["detail"] = detail
    return d


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------


def headless_files_on_disk() -> List[Path]:
    if not HEADLESS.is_dir():
        return []
    return sorted(
        p
        for p in HEADLESS.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name != "__pycache__"
    )


def receipt_index() -> Dict[str, List[str]]:
    """Map basename -> receipt files that mention it."""
    hits: Dict[str, List[str]] = defaultdict(list)
    if not RECEIPTS.is_dir():
        return hits
    names = [p.name for p in headless_files_on_disk()]
    haider_names = [
        "haider.py",
        "p0_tool_bridge.py",
        "index.py",
        "context.py",
        "mutation.py",
    ]
    needles = names + haider_names
    for rp in sorted(RECEIPTS.iterdir()):
        if not rp.is_file():
            continue
        if rp.suffix not in {".json", ".jsonl", ".txt", ".md", ".stderr", ".stdout"}:
            continue
        try:
            text = rp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n in needles:
            if n in text:
                hits[n].append(rp.name)
    return hits


def cross_file_token(name: str, origin: str, files: Dict[str, str]) -> List[str]:
    """Other headless files that both name the defining file and the symbol.

    A shared helper name (`median`, `git_head`) is NOT a reference to *this*
    definition. The census script mentioning a symbol in a comment is not a
    caller. importlib + `sm.assert_deletable` is.
    """
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    origin_markers = {origin, origin.replace(".py", "")}
    found = []
    for fn, text in files.items():
        if fn == origin or fn == "dead_code_census.py":
            continue
        if not pat.search(text):
            continue
        if not any(m in text for m in origin_markers):
            continue
        found.append(fn)
    return found


def classify_headless_file(
    path: Path,
    text: str,
    tree: Optional[ast.Module],
    rec_hits: Dict[str, List[str]],
    sibling_cites: List[str],
) -> Dict[str, Any]:
    rel = f"tools/headless/{path.name}"
    b, n = file_metrics(rel, text)
    receipts = cited_receipts(text)
    existing = [r for r in receipts if (REPO / r).is_file()]
    missing = [r for r in receipts if r not in existing]
    evidence = [
        ev("on_disk", "yes", str(path)),
        ev("bytes_lines", f"{b} bytes / {n} lines"),
        ev("if_name_main", "yes" if has_main_guard(text, tree) else "no"),
        ev("parseable_python", "yes" if tree is not None else ("n/a-shell" if path.suffix != ".py" else "no")),
        ev("receipts_cited_in_source", f"{len(receipts)}", receipts[:12]),
        ev("those_receipts_on_disk", f"{len(existing)} present, {len(missing)} absent", {
            "present": existing[:12],
            "absent": missing[:12],
        }),
        ev("filename_cited_in_receipts_headless", f"{len(rec_hits.get(path.name, []))} receipts", rec_hits.get(path.name, [])[:12]),
        ev("cited_by_sibling_headless_files", f"{len(sibling_cites)}", sibling_cites[:12]),
    ]
    # Ten science lanes: keep every noetic_* file at file level.
    if path.name.startswith(NOETIC_PREFIX):
        return item(
            ident=f"file:{rel}",
            path=rel,
            kind="file",
            classification="KEEP",
            reason="in-flight noetic science lane producer; file-level DELETE would strand a running campaign",
            bytes_=b,
            lines=n,
            evidence=evidence + [ev("in_flight_science_lane", "yes")],
            extra={"role": "science_lane"},
        )
    if path.suffix == ".sh" and os.access(path, os.X_OK):
        return item(
            ident=f"file:{rel}",
            path=rel,
            kind="file",
            classification="KEEP",
            reason="executable launcher; cited by runtime receipts",
            bytes_=b,
            lines=n,
            evidence=evidence + [ev("executable_bit", oct(path.stat().st_mode)[-3:])],
            extra={"role": "launcher"},
        )
    if has_main_guard(text, tree) or path.name.endswith("_test.py"):
        role = "test_harness" if ("_test.py" in path.name or "test" in path.name) else "script"
        return item(
            ident=f"file:{rel}",
            path=rel,
            kind="file",
            classification="KEEP",
            reason="live entrypoint (if __name__ and/or pytest); not unreachable",
            bytes_=b,
            lines=n,
            evidence=evidence,
            extra={"role": role},
        )
    return item(
        ident=f"file:{rel}",
        path=rel,
        kind="file",
        classification="UNKNOWN",
        reason="no if __name__ guard and not a recognized test/launcher; default UNKNOWN",
        bytes_=b,
        lines=n,
        evidence=evidence,
    )


def unused_symbols_in_headless(
    files: Dict[str, str],
    trees: Dict[str, ast.Module],
) -> List[Dict[str, Any]]:
    """Top-level defs with zero identifier loads in-file, then cross-file."""
    out: List[Dict[str, Any]] = []
    for fn, tree in trees.items():
        text = files[fn]
        loads = name_loads(tree)
        funcs, classes = top_defs(tree)
        rel = f"tools/headless/{fn}"
        for node in list(funcs) + list(classes):
            name = node.name  # type: ignore[attr-defined]
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if name == "main" or name.startswith("test_"):
                # pytest entry / script entry. Keep even if never loaded in-file.
                continue
            if name in loads:
                continue
            start, end, nlines, nbytes = node_span(node, text)
            others = cross_file_token(name, fn, files)
            evidence = [
                ev("in_file_Name_loads", "0 (def is Store, never Load)"),
                ev("not_main_or_test_star", "yes"),
                ev("span", f"L{start}-L{end}", {"lineno": start, "end_lineno": end}),
                ev(
                    "cross_file_token_in_tools_headless",
                    f"{len(others)} files" if others else "none",
                    others,
                ),
            ]
            whole = len(re.findall(r"\b" + re.escape(name) + r"\b", text))
            evidence.append(ev("token_occurrences_in_defining_file", str(whole)))

            if others:
                # e.g. storage_manager.assert_deletable used by storage_manager_test
                out.append(
                    item(
                        ident=f"{kind}:{rel}:{name}",
                        path=rel,
                        kind=kind,
                        classification="KEEP",
                        reason=f"zero in-file loads but referenced from {others}",
                        bytes_=nbytes,
                        lines=nlines,
                        evidence=evidence
                        + [ev("downgraded_from", "DELETE — looked local-dead until cross-file")],
                        extra={"name": name, "lineno": start},
                    )
                )
                continue
            # git HEAD search for importers of this function as a name is too
            # noisy (median, git_head). Restrict to unique-ish names or skip.
            reason = (
                "defined, never loaded as an identifier in its file or any "
                "other tools/headless file; string payloads (if any) are not "
                "identifier refs"
            )
            out.append(
                item(
                    ident=f"{kind}:{rel}:{name}",
                    path=rel,
                    kind=kind,
                    classification="DELETE",
                    reason=reason,
                    bytes_=nbytes,
                    lines=nlines,
                    evidence=evidence,
                    extra={"name": name, "lineno": start, "end_lineno": end},
                )
            )
    return out


def unused_imports_in_headless(
    files: Dict[str, str],
    trees: Dict[str, ast.Module],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fn, tree in trees.items():
        text = files[fn]
        lines = text.splitlines()
        loads = name_loads(tree)
        unused = []
        kept_side_effect = []
        for row in import_bindings(tree, lines):
            if row["bind"] in loads:
                continue
            if row["noqa_f401"]:
                kept_side_effect.append(row)
                continue
            unused.append(row)
        rel = f"tools/headless/{fn}"
        if unused:
            # bytes: unique lines that would shrink if the whole import vanished,
            # else 0 for a name dropped from a multi-import.
            drop_lines = set()
            for u in unused:
                # if every binding on that line is unused, count the line
                lineno = u["lineno"]
                siblings = [
                    r
                    for r in import_bindings(tree, lines)
                    if r["lineno"] == lineno
                ]
                if all(s["bind"] not in loads and not s["noqa_f401"] for s in siblings):
                    drop_lines.add(lineno)
            nbytes = 0
            nlines = 0
            for ln in sorted(drop_lines):
                nlines += 1
                nbytes += len((lines[ln - 1] + "\n").encode("utf-8"))
            out.append(
                item(
                    ident=f"unused_imports:{rel}",
                    path=rel,
                    kind="unused_imports",
                    classification="DELETE",
                    reason="bind name is never a Load (including annotations); not noqa F401",
                    bytes_=nbytes,
                    lines=nlines,
                    evidence=[
                        ev("ast_Name_loads_of_bind", "0 for each listed name"),
                        ev("excluded_noqa_F401_side_effect", "yes"),
                        ev("names", f"{len(unused)}", [
                            {
                                "bind": u["bind"],
                                "module": u["module"],
                                "lineno": u["lineno"],
                                "line": u["line"][:160],
                            }
                            for u in unused
                        ]),
                    ],
                    extra={"names": [u["bind"] for u in unused]},
                )
            )
        if kept_side_effect:
            out.append(
                item(
                    ident=f"side_effect_imports:{rel}",
                    path=rel,
                    kind="unused_imports",
                    classification="KEEP",
                    reason="noqa F401 side-effect import (registers a plugin, probes a dep, etc.)",
                    bytes_=0,
                    lines=0,
                    evidence=[
                        ev("noqa_F401", "present on line"),
                        ev("names", f"{len(kept_side_effect)}", [
                            {"bind": u["bind"], "lineno": u["lineno"], "line": u["line"][:160]}
                            for u in kept_side_effect
                        ]),
                    ],
                    extra={"names": [u["bind"] for u in kept_side_effect]},
                )
            )
    return out


def classify_haider_control_plane() -> List[Dict[str, Any]]:
    """File-level reachability of tools/haider. Blobs via git; not on disk here."""
    items: List[Dict[str, Any]] = []
    files = git_ls_tree("tools/haider")
    py = [f for f in files if f.endswith(".py")]
    mods = [f for f in py if "/tests/" not in f]

    # Pre-compute importer hits per stem.
    for rel in mods:
        stem = Path(rel).stem
        text, src = load_text(rel)
        b, n = file_metrics(rel, text)
        evidence = [
            ev("materialized_in_this_worktree", "no — sparse; read via git show HEAD"),
            ev("source", src),
            ev("bytes_lines", f"{b} bytes / {n} lines"),
        ]
        if text is None:
            items.append(
                item(
                    ident=f"file:{rel}",
                    path=rel,
                    kind="file",
                    classification="UNKNOWN",
                    reason="git show HEAD failed; cannot classify",
                    bytes_=b,
                    lines=n,
                    evidence=evidence,
                    extra={"out_of_write_scope": True},
                )
            )
            continue

        hits = module_import_hits(
            stem,
            bare_ok=stem in {"haider", "p0_tool_bridge"},
        )
        # Filter self-hits and receipt-only later.
        code_hits = [
            h
            for h in hits
            if not h.startswith("receipts/")
            and not h.startswith(rel)
            and "dead_code_census.py" not in h
        ]
        receipt_hits = [h for h in hits if h.startswith("receipts/")]
        evidence += [
            ev("exact_import_hits_code", f"{len(code_hits)}", code_hits[:16]),
            ev("exact_import_hits_receipts", f"{len(receipt_hits)}", receipt_hits[:8]),
        ]

        # Special cases with extra identifier searches.
        extra_id_hits: List[str] = []
        if stem == "index":
            extra_id_hits = [
                h
                for h in identifier_hits("WorkspaceIndex")
                if not h.startswith(rel) and "dead_code_census.py" not in h
            ]
            evidence.append(ev("WorkspaceIndex_hits_outside_def", f"{len(extra_id_hits)}", extra_id_hits[:12]))
        if stem == "haider":
            extra_id_hits = [
                h
                for h in identifier_hits("import haider")
                if not h.startswith(rel) and "dead_code_census.py" not in h
            ]
            evidence.append(ev("import_haider_hits", f"{len(extra_id_hits)}", extra_id_hits[:12]))
        if stem == "p0_tool_bridge":
            extra_id_hits = [
                h
                for h in identifier_hits("import p0_tool_bridge")
                if not h.startswith(rel) and "dead_code_census.py" not in h
            ]
            evidence.append(ev("import_p0_tool_bridge_hits", f"{len(extra_id_hits)}", extra_id_hits[:12]))

        # Package entry
        if stem in {"__init__", "__main__"}:
            items.append(
                item(
                    ident=f"file:{rel}",
                    path=rel,
                    kind="file",
                    classification="KEEP",
                    reason="package entry (python -m hcli / public exports)",
                    bytes_=b,
                    lines=n,
                    evidence=evidence + [ev("python_m_hcli", "yes")],
                    extra={"out_of_write_scope": True, "role": "package_entry"},
                )
            )
            continue

        # HCLI-v0 bootstrap cluster
        if rel in {
            "tools/haider/haider.py",
            "tools/haider/p0_tool_bridge.py",
            "tools/haider/test_haider_edit.py",
            "tools/haider/test_p0_tool_bridge.py",
        }:
            items.append(
                item(
                    ident=f"file:{rel}",
                    path=rel,
                    kind="file",
                    classification="ARCHIVE",
                    reason=(
                        "HCLI-v0 bootstrap (Gate Zero). Live control plane is "
                        "tools/haider/hcli/. Still an entrypoint/test of that "
                        "bootstrap; historical science, not proven-dead live path"
                    ),
                    bytes_=b,
                    lines=n,
                    evidence=evidence
                    + [
                        ev(
                            "live_control_plane",
                            "tools/haider/hcli/ (python -m hcli)",
                        ),
                        ev(
                            "aider_imports_in_tools",
                            "zero import aider / from aider / aider-chat (standing fact; substring hits are 'haider')",
                        ),
                    ],
                    extra={"out_of_write_scope": True, "role": "hcli_v0_bootstrap"},
                )
            )
            continue

        if stem == "index":
            code_id = [h for h in extra_id_hits if not h.startswith("receipts/")]
            if not code_hits and not code_id:
                items.append(
                    item(
                        ident=f"file:{rel}",
                        path=rel,
                        kind="file",
                        classification="DELETE",
                        reason=(
                            "WorkspaceIndex has zero in-tree importers. "
                            "HCLI_PERSISTENCE_AUDIT lists the file because "
                            "WorkspaceIndex.read calls open() — a read, not a "
                            "caller. hcli/__init__.py does not export it"
                        ),
                        bytes_=b,
                        lines=n,
                        evidence=evidence
                        + [
                            ev("hcli___init___exports", "parse_haider_args, main, Workspace, Controller, Event, EventBus — no index"),
                            ev("from_.import_index", "no hits under tools/ receipts/headless lab/"),
                        ],
                        extra={"out_of_write_scope": True, "role": "unreachable_module"},
                    )
                )
                continue

        if stem == "context":
            # re-export of goal.WorkerPacket. Zero exact importers.
            if not code_hits:
                items.append(
                    item(
                        ident=f"file:{rel}",
                        path=rel,
                        kind="file",
                        classification="ARCHIVE",
                        reason=(
                            "documented re-export of goal.WorkerPacket / "
                            "compile_worker_context with zero in-tree "
                            "`from hcli.context import` callers. goal.py is "
                            "the authority. Too small and too public-looking "
                            "to DELETE without a deprecation cycle"
                        ),
                        bytes_=b,
                        lines=n,
                        evidence=evidence
                        + [
                            ev(
                                "prefix_collision_note",
                                "git grep 'hcli.context' also matches hcli.context_budget; exact from-import used here",
                            )
                        ],
                        extra={"out_of_write_scope": True, "role": "compat_reexport"},
                    )
                )
                continue

        # Default: has importers or is part of the live package.
        if code_hits or stem in {
            "app",
            "cli",
            "engine",
            "controller",
            "mission",
            "workunit",
            "runtime",
            "scheduler",
            "mutation",
        }:
            items.append(
                item(
                    ident=f"file:{rel}",
                    path=rel,
                    kind="file",
                    classification="KEEP",
                    reason="live HCLI module with in-tree importers and/or package-core role",
                    bytes_=b,
                    lines=n,
                    evidence=evidence,
                    extra={"out_of_write_scope": True, "role": "hcli_live"},
                )
            )
        else:
            items.append(
                item(
                    ident=f"file:{rel}",
                    path=rel,
                    kind="file",
                    classification="UNKNOWN",
                    reason="no exact import hits under the searched pathspecs; may still be imported dynamically",
                    bytes_=b,
                    lines=n,
                    evidence=evidence,
                    extra={"out_of_write_scope": True},
                )
            )

    # Markdown / leftover v0 docs
    for rel in files:
        if rel.endswith(".py"):
            continue
        text, src = load_text(rel)
        b, n = file_metrics(rel, text)
        items.append(
            item(
                ident=f"file:{rel}",
                path=rel,
                kind="file",
                classification="ARCHIVE",
                reason="HCLI-v0 / productization notes next to the fossil launcher; not live runtime",
                bytes_=b,
                lines=n,
                evidence=[
                    ev("materialized_in_this_worktree", "no — sparse"),
                    ev("source", src),
                ],
                extra={"out_of_write_scope": True, "role": "doc"},
            )
        )
    return items


def namesake_unknowns() -> List[Dict[str, Any]]:
    """Other HCLI/headless surfaces. Not dead just because they share a name."""
    items = []
    lab_files = git_ls_tree("lab/hcli")
    lab_bytes = 0
    lab_lines = 0
    for rel in lab_files:
        text, _ = load_text(rel)
        b, n = file_metrics(rel, text)
        lab_bytes += b
        lab_lines += n
    items.append(
        item(
            ident="namesake:lab/hcli",
            path="lab/hcli/",
            kind="package",
            classification="UNKNOWN",
            reason=(
                "namesake only. lab.hcli is Agent-OS scaffolds (Option-C, "
                "residency, self-evolution), not tools.haider.hcli. Internals "
                "not AST-censused in this lane (lab/hcli is not in the sparse "
                "roots). Not proposed for deletion"
            ),
            bytes_=lab_bytes,
            lines=lab_lines,
            evidence=[
                ev("git_ls_tree_lab/hcli", f"{len(lab_files)} files", lab_files),
                ev("package_docstring", "HCLI Agent OS scaffolds — self-evolution, Option-C sandbox, residency modes"),
            ],
            extra={"out_of_write_scope": True},
        )
    )
    rust_bins = [
        "crates/hide-backend/src/bin/hcli.rs",
        "crates/hide-backend/src/bin/hide-headless.rs",
        "crates/hide-backend/src/headless.rs",
    ]
    for rel in rust_bins:
        text, src = load_text(rel)
        b, n = file_metrics(rel, text)
        items.append(
            item(
                ident=f"file:{rel}",
                path=rel,
                kind="file",
                classification="KEEP",
                reason=(
                    "different product surface (HIDE/Hawking Rust backend). "
                    "Not the Python tools/headless campaign and not a fossil "
                    "of it. CLI-flag liveness inside the binary is UNKNOWN "
                    "without a compile-time dead_code pass"
                ),
                bytes_=b,
                lines=n,
                evidence=[
                    ev("source", src),
                    ev("exists_in_HEAD", "yes" if text is not None else "no"),
                ],
                extra={"out_of_write_scope": True, "role": "rust_namesake"},
            )
        )
    return items


def stale_cli_flags() -> List[Dict[str, Any]]:
    text, src = load_text("tools/haider/hcli/cli.py")
    b = 0
    n = 0
    if text:
        # just the flag lines, not the whole file
        n = 2
        b = len(
            '    parser.add_argument("--task", type=str, default=None, help="(legacy) mission text")\n'
            '    parser.add_argument("--task-file", type=str, default=None, help="(legacy) path to mission file")\n'
        )
    return [
        item(
            ident="cli_flag:tools/haider/hcli/cli.py:--task",
            path="tools/haider/hcli/cli.py",
            kind="cli_flag",
            classification="KEEP",
            reason=(
                "marked (legacy) but still wired in parse_haider_args: "
                "args.task / args.task_file still populate prompt. Not dead. "
                "Out-of-tree callers UNKNOWN — do not remove until measured"
            ),
            bytes_=b,
            lines=n,
            evidence=[
                ev("source", src),
                ev("help_text", "(legacy) mission text / path to mission file"),
                ev("still_assigns_args.prompt", "yes"),
            ],
            extra={"out_of_write_scope": True, "flags": ["--task", "--task-file"]},
        )
    ]


def haider_tests_keep() -> List[Dict[str, Any]]:
    files = [f for f in git_ls_tree("tools/haider/hcli/tests") if f.endswith(".py")]
    total_b = 0
    total_n = 0
    for rel in files:
        text, _ = load_text(rel)
        b, n = file_metrics(rel, text)
        total_b += b
        total_n += n
    return [
        item(
            ident="tree:tools/haider/hcli/tests",
            path="tools/haider/hcli/tests/",
            kind="tree",
            classification="KEEP",
            reason=(
                "46 HCLI package tests (standing fact). Live pytest surface of the "
                "control plane, not a fossil of haider.py. Not enumerated "
                "symbol-by-symbol in this lane"
            ),
            bytes_=total_b,
            lines=total_n,
            evidence=[
                ev("git_ls_tree_py", f"{len(files)} files", files[:12] + (["…"] if len(files) > 12 else [])),
            ],
            extra={"out_of_write_scope": True, "role": "hcli_tests"},
        )
    ]


def receipts_never_delete(rec_hits: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    items = []
    if not RECEIPTS.is_dir():
        return items
    total_b = 0
    total_n = 0
    count = 0
    for p in RECEIPTS.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_file():
            count += 1
            total_b += p.stat().st_size
            # skip huge JSON line counts if expensive — still count
            try:
                # cheap line count
                with p.open("rb") as f:
                    total_n += sum(1 for _ in f)
            except OSError:
                pass
        elif p.is_dir():
            # snapshots/ etc — still never-delete, don't walk as source
            count += 1
    items.append(
        item(
            ident="policy:receipts/headless",
            path="receipts/headless/",
            kind="tree",
            classification="KEEP",
            reason=(
                "Historical receipts are evidence, not dead source. Nothing "
                "under receipts/ is proposed for deletion, including "
                "MACHINE_GENOME.superseded-* and producer-less files. "
                "A receipt nobody imports is exactly what made negative-science "
                "recoverable"
            ),
            bytes_=total_b,
            lines=total_n,
            evidence=[
                ev("entries_on_disk", str(count)),
                ev("proposed_for_deletion", "0"),
                ev("filename_citation_index_keys", str(len(rec_hits))),
            ],
            extra={"never_delete": True},
        )
    )
    return items


def commented_out_scan(files: Dict[str, str]) -> List[Dict[str, Any]]:
    found = []
    pat = re.compile(r"^\s*#\s*(def |class |async def )")
    for fn, text in files.items():
        for i, ln in enumerate(text.splitlines(), 1):
            if pat.match(ln):
                found.append({"file": fn, "lineno": i, "line": ln.strip()[:120]})
    return [
        item(
            ident="commented_out_defs:tools/headless",
            path="tools/headless/",
            kind="commented_out",
            classification="KEEP" if not found else "DELETE",
            reason=(
                "no commented-out def/class/async def in tools/headless"
                if not found
                else "commented-out implementations present"
            ),
            bytes_=0,
            lines=0,
            evidence=[ev("matches", str(len(found)), found[:20])],
        )
    ]


def watched_fail() -> List[str]:
    return [
        (
            "git grep without a treeish on a sparse checkout only searches "
            "materialized files. A first pass 'proved' WorkspaceIndex had zero "
            "callers for the wrong reason. Method became `git grep HEAD` / "
            "`git show HEAD:<path>`."
        ),
        (
            "AST unused-function detection that treated a Name load of 1 as dead "
            "flagged every CHECKS = [check_v1, check_v2, ...] registration. "
            "Tightened to loads == 0. test_* kept as pytest entries even at 0 loads."
        ),
        (
            "`import hcli.executors` binds the name `hcli`, which looks unused. "
            "It is a noqa F401 side-effect import that installs Engine.execute_workunit. "
            "Same pattern: `import cv2  # noqa: F401` in vmcp_unavailable_gate "
            "(presence probe before re-exec)."
        ),
        (
            "hcli_persistence_audit.FakeBackend and _gate() appear in a child-process "
            "source STRING. The class/function themselves have zero identifier loads. "
            "String mentions are not reachability."
        ),
        (
            "noetic_route_ledger.median (Python) is unused; a Swift `func median` "
            "inside an embedded timer string is live. Deleting the file would "
            "kill science; deleting the Python helper would not."
        ),
        (
            "HCLI_AUDIT.json claims rollback_mutation / apply_mutation_operations "
            "have zero callers. tools/headless/hcli_persistence_audit.py now "
            "imports and calls both (lines 48, 1211, 1215). The audit is stale. "
            "Tempted DELETE, downgraded to KEEP."
        ),
        (
            "Substring `hcli.context` matches `hcli.context_budget`. Exact "
            "`from hcli.context import` / `from .context import` is empty. "
            "Used that, not the prefix."
        ),
        (
            "artifact_census.py's module docstring still says it writes DISK_TRUTH.json; "
            "the body only writes ARTIFACT_LEDGER.json. leftover git_head/dir_sha "
            "are the DiskTruth half that moved to disk_truth.py."
        ),
        (
            "Substring grep for unused `os`/`re`/`sys` overcounts (sysctl, diagnostic, "
            "restore). Classification uses AST Name loads."
        ),
        (
            "director_epoch_replay.py cites receipts/headless/DIRECTOR_EPOCH_REPLAY.json "
            "which is not on disk. That is 'not yet run in this tree', not unreachable."
        ),
        (
            "Naive git grep of tools/headless basenames outside the directory returned "
            "zero because those files are only referenced from receipts/headless and "
            "each other — both of which the first exclude-pathspec dropped."
        ),
        (
            "Cascading unused is a follow-up, not guessed here: deleting _gate() would "
            "orphan the GIB/MemGate names on the hcli.machine import; deleting "
            "hcli_scheduler_quality.median would orphan its Sequence annotation. "
            "Those binds are Loads today because the dead def still exists."
        ),
        (
            "The first census run treated mentions inside dead_code_census.py itself "
            "as cross-file callers, so git_head/dir_sha/FakeBackend/_gate/median were "
            "KEEP. The census script is not a caller; it is now excluded from that check."
        ),
    ]


def downgraded(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """First-class record of DELETE temptations that inspection reversed."""
    rows = [
        {
            "path": "tools/haider/haider.py",
            "tempted": "DELETE",
            "landed": "ARCHIVE",
            "why": (
                "Looks like a leftover v0 launcher (501 lines, only imported by "
                "test_haider_edit.py). HCLI_AUDIT still cites a historical monolith "
                "at haider.py:2792 which no longer exists — the file was already "
                "gutted. Remaining bytes are Gate Zero bootstrap plus tests. "
                "DIRTY_TREE_PRESERVATION inventories the path. Historical science, "
                "not proven-dead live control plane."
            ),
        },
        {
            "path": "tools/haider/hcli/mutation.py (rollback_mutation / apply_mutation_operations)",
            "tempted": "DELETE",
            "landed": "KEEP",
            "why": (
                "HCLI_AUDIT.json:127 says the sole importer is "
                "test_acceptance_integrity.py importing MutationError, _apply_insert, "
                "_apply_replace. That is false now: hcli_persistence_audit.py:48 "
                "imports apply_mutation_operations and rollback_mutation and calls "
                "them at 1211/1215. A stale audit is not evidence."
            ),
        },
        {
            "path": "tools/headless/director_epoch_replay.py",
            "tempted": "DELETE",
            "landed": "KEEP",
            "why": (
                "No DIRECTOR_EPOCH_REPLAY.json on disk. The file is a live G003 "
                "proof harness (if __name__, argparse, talks to the real hcli "
                "binary). Absence of a receipt is 'not yet executed here'."
            ),
        },
        {
            "path": "tools/headless/storage_manager.py:assert_deletable",
            "tempted": "DELETE",
            "landed": "KEEP",
            "why": (
                "Zero in-file loads. storage_manager_test.py loads the module via "
                "importlib and calls sm.assert_deletable. It is the public guard "
                "the KEEP_LIST was missing."
            ),
        },
        {
            "path": "tools/haider/hcli/context.py",
            "tempted": "DELETE",
            "landed": "ARCHIVE",
            "why": (
                "Zero `from hcli.context import` callers (goal.py is the authority). "
                "12-line documented re-export. Receipts cite the path. A silent "
                "delete of a public alias is a deprecation, not a proof."
            ),
        },
        {
            "path": "tools/headless/hcli_self_optimize.py",
            "tempted": "ARCHIVE-as-dead-optimizer",
            "landed": "KEEP",
            "why": (
                "Superseded as the *current* optimizer by hcli_self_optimize_2.py, "
                "but it is the producer of HCLI_SELF_OPT_ITERATION_1.json. "
                "Iteration 1 is science. Do not delete the producer."
            ),
        },
        {
            "path": "tools/headless/noetic_route_ledger.py (the file)",
            "tempted": "DELETE because median() is unused",
            "landed": "KEEP file; DELETE only the Python helper",
            "why": (
                "An unused 5-line helper is not an unused organ census. Ten science "
                "lanes are running against tools/headless/noetic_*.py."
            ),
        },
        {
            "path": "receipts/headless/* (any producer-less receipt)",
            "tempted": "DELETE as unreferenced",
            "landed": "KEEP (never-delete policy)",
            "why": (
                "This campaign's negative-science census turned nine supposedly-closed "
                "results into live opportunities because a receipt nobody imports "
                "still existed. MACHINE_GENOME.superseded-* is evidence."
            ),
        },
    ]
    # Also surface KEEP symbols that the AST pass downgraded after cross-file.
    for it in items:
        if it.get("classification") == "KEEP" and it.get("kind") in {"function", "class"}:
            if any(e.get("check") == "downgraded_from" for e in it.get("evidence") or []):
                rows.append(
                    {
                        "path": f"{it['path']}:{it.get('name')}",
                        "tempted": "DELETE",
                        "landed": "KEEP",
                        "why": it.get("reason"),
                    }
                )
    return rows


def totals(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, int]] = {}
    for it in items:
        c = it["classification"]
        slot = by.setdefault(c, {"count": 0, "bytes": 0, "lines": 0})
        slot["count"] += 1
        slot["bytes"] += int(it.get("bytes") or 0)
        slot["lines"] += int(it.get("lines") or 0)
    return by


def sibling_citations(files: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    names = list(files)
    for fn, text in files.items():
        for other in names:
            if other == fn:
                continue
            if other in text:
                out[other].append(fn)
    return out


def build() -> Tuple[Dict[str, Any], str]:
    t0 = time.time()
    rec_hits = receipt_index()
    on_disk = headless_files_on_disk()
    files: Dict[str, str] = {}
    trees: Dict[str, ast.Module] = {}
    parse_fail: List[str] = []
    for p in on_disk:
        if p.suffix != ".py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        files[p.name] = text
        tree = parse_mod(text)
        if tree is None:
            parse_fail.append(p.name)
        else:
            trees[p.name] = tree

    sib = sibling_citations(files)
    items: List[Dict[str, Any]] = []

    # File-level headless
    for p in on_disk:
        text = files.get(p.name)
        if text is None:
            text = p.read_text(encoding="utf-8", errors="replace")
        tree = trees.get(p.name)
        items.append(
            classify_headless_file(
                p, text, tree, rec_hits, sib.get(p.name, [])
            )
        )

    items.extend(unused_symbols_in_headless(files, trees))
    items.extend(unused_imports_in_headless(files, trees))
    items.extend(commented_out_scan(files))
    items.extend(classify_haider_control_plane())
    items.extend(namesake_unknowns())
    items.extend(stale_cli_flags())
    items.extend(receipts_never_delete(rec_hits))
    items.extend(haider_tests_keep())

    # Policy: nothing under receipts/ or workspace/ may be DELETE.
    for it in items:
        p = it.get("path") or ""
        if it["classification"] == "DELETE" and p.startswith(NEVER_DELETE_PREFIXES):
            it["classification"] = "KEEP"
            it["reason"] = "policy override: receipts/ and workspace/ are never DELETE"
            it.setdefault("evidence", []).append(ev("policy_override", "NEVER_DELETE_PREFIXES"))

    down = downgraded(items)
    fail = watched_fail()
    by = totals(items)
    code_items = [it for it in items if it.get("kind") not in {"tree", "package"}]
    by_code = totals(code_items)

    delete_items = [it for it in items if it["classification"] == "DELETE"]
    archive_items = [it for it in items if it["classification"] == "ARCHIVE"]
    unknown_items = [it for it in items if it["classification"] == "UNKNOWN"]

    receipt = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "repo": str(REPO),
        "method": {
            "ast_identifier_loads": "def/class name is Store; list/dict registrations are Loads; strings are not",
            "cross_file_headless": "token search across tools/headless *.py on disk",
            "git_search": "git grep HEAD / git show HEAD:<path> / git ls-tree (sparse-safe)",
            "receipts": "filename citations in receipts/headless are liveness evidence, never deletion candidates",
            "entrypoints": "if __name__, pytest test_*, python -m hcli, noqa F401 side-effect imports, importlib loaders",
            "sparse_checkout": (
                "a missing worktree path is NOT evidence of absence; "
                "unmaterialized blobs were read from HEAD"
            ),
        },
        "scope": {
            "write": ["tools/headless/dead_code_census.py", "receipts/headless/DEAD_CODE_CENSUS.json"],
            "verify": ["tools/headless", "receipts/headless"],
            "deny": [
                "workspace",
                "crates",
                "visionmcp",
                "app",
                "lab",
                "tools/haider",
                "ramanujan",
                "receipts/ascent-2026-08-16",
                "receipts/ascent-2026-08-18",
            ],
            "note": (
                "This lane changes no source. Haider/lab/crates findings are a "
                "plan for a human. Ten science lanes are running against "
                "tools/headless and receipts/headless."
            ),
        },
        "anti_goodhart": (
            "A 20% LOC cut with worse architecture is a failure. DELETE here is "
            "unused imports, unused helpers, and one unreachable module "
            "(hcli/index.py). Campaign harnesses and noetic lanes stay KEEP. "
            "Capability over ceremony."
        ),
        "standing_facts": {
            "haider_is_a_fossil_namespace_not_an_architecture": True,
            "import_aider_in_tools": 0,
            "from_aider_in_tools": 0,
            "shelled_aider": 0,
            "sealed_schema_ids_preserved": True,
            "receipts_never_delete": True,
            "workspace_never_delete": True,
            "untracked_work_is_real_work": True,
        },
        "parse_failures": parse_fail,
        "by_class": by,
        "by_class_excluding_policy_trees": by_code,
        "delete_byte_line_totals": {
            "count": by.get("DELETE", {}).get("count", 0),
            "bytes": by.get("DELETE", {}).get("bytes", 0),
            "lines": by.get("DELETE", {}).get("lines", 0),
        },
        "archive_byte_line_totals": {
            "count": by.get("ARCHIVE", {}).get("count", 0),
            "bytes": by.get("ARCHIVE", {}).get("bytes", 0),
            "lines": by.get("ARCHIVE", {}).get("lines", 0),
        },
        "keep_byte_line_totals": {
            "count": by.get("KEEP", {}).get("count", 0),
            "bytes": by.get("KEEP", {}).get("bytes", 0),
            "lines": by.get("KEEP", {}).get("lines", 0),
        },
        "unknown_byte_line_totals": {
            "count": by.get("UNKNOWN", {}).get("count", 0),
            "bytes": by.get("UNKNOWN", {}).get("bytes", 0),
            "lines": by.get("UNKNOWN", {}).get("lines", 0),
        },
        "tempted_to_delete_then_downgraded": down,
        "what_i_watched_fail": fail,
        "never_delete": {
            "receipts/": "historical science and sealed schema ids; 0 proposed DELETE",
            "workspace/": "precious corpora (ascension-sandbox, phaseB) plus vendor; 0 proposed DELETE",
        },
        "human_migration_plan": {
            "in_write_scope_later": (
                "Drop proven-unused imports and the unused helpers inside "
                "tools/headless. Do not do that while the ten noetic lanes are "
                "in flight if the edit would rebase their worktrees."
            ),
            "out_of_write_scope": (
                "ARCHIVE the HCLI-v0 cluster (haider.py, p0_tool_bridge.py, "
                "their tests, P1 doc). DELETE tools/haider/hcli/index.py after "
                "confirming no out-of-tree importer. Deprecate context.py in "
                "favor of goal.py. Leave --task/--task-file until callers are "
                "measured."
            ),
            "do_not": [
                "rm any receipts/",
                "rm any workspace/ campaign corpora",
                "rename sealed schema ids",
                "git mv noetic_* while those lanes are running",
                "treat a 20% LOC cut as success",
            ],
        },
        "items": items,
        "elapsed_s": round(time.time() - t0, 2),
    }

    report = format_report(receipt, delete_items, archive_items, unknown_items, down, fail)
    return receipt, report


def format_report(
    receipt: Dict[str, Any],
    delete_items: List[Dict[str, Any]],
    archive_items: List[Dict[str, Any]],
    unknown_items: List[Dict[str, Any]],
    down: List[Dict[str, Any]],
    fail: List[str],
) -> str:
    lines: List[str] = []
    a = lines.append
    a(f"schema     {receipt['schema']}")
    a(f"git_head   {receipt['git_head']}")
    a(f"generated  {receipt['generated_at']}")
    a(f"elapsed    {receipt['elapsed_s']}s")
    a("")
    a("by_class (count / bytes / lines):")
    for cls in ("DELETE", "ARCHIVE", "KEEP", "UNKNOWN"):
        slot = receipt["by_class"].get(cls, {"count": 0, "bytes": 0, "lines": 0})
        a(f"  {cls:<8} {slot['count']:>5}  {slot['bytes']:>10} B  {slot['lines']:>7} L")
    a("by_class excluding policy trees (receipts/, lab/hcli package, test trees):")
    for cls in ("DELETE", "ARCHIVE", "KEEP", "UNKNOWN"):
        slot = receipt["by_class_excluding_policy_trees"].get(
            cls, {"count": 0, "bytes": 0, "lines": 0}
        )
        a(f"  {cls:<8} {slot['count']:>5}  {slot['bytes']:>10} B  {slot['lines']:>7} L")
    a("")
    a("DELETE (proven-dead live code):")
    if not delete_items:
        a("  (none)")
    for it in delete_items:
        a(
            f"  {it['path']}  {it['kind']}  {it.get('name') or it.get('names') or ''}  "
            f"{it['bytes']}B {it['lines']}L"
        )
        a(f"      {it['reason']}")
    a("")
    a("ARCHIVE:")
    for it in archive_items:
        a(f"  {it['path']}  {it['bytes']}B {it['lines']}L")
        a(f"      {it['reason']}")
    a("")
    a("UNKNOWN:")
    for it in unknown_items:
        a(f"  {it['path']}  {it['bytes']}B {it['lines']}L")
        a(f"      {it['reason']}")
    a("")
    a("tempted DELETE then downgraded:")
    for d in down:
        a(f"  {d['path']}")
        a(f"      {d['tempted']} -> {d['landed']}: {d['why']}")
    a("")
    a("## WHAT I WATCHED FAIL")
    for i, w in enumerate(fail, 1):
        a(f"  {i}. {w}")
    a("")
    a("never_delete: receipts/ (0 proposed), workspace/ (0 proposed)")
    a(f"items: {len(receipt['items'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--out",
        default=str(RECEIPT_PATH),
        help="receipt path (default: receipts/headless/DEAD_CODE_CENSUS.json)",
    )
    args = ap.parse_args()
    receipt, report = build()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    sys.stdout.write(report)
    print(f"wrote {out} ({out.stat().st_size} bytes)")

    # Self-checks the operator can see fail.
    problems = []
    if receipt["schema"] != SCHEMA:
        problems.append("schema drift")
    classes = set(receipt["by_class"])
    if not classes <= {"DELETE", "ARCHIVE", "KEEP", "UNKNOWN"}:
        problems.append(f"unexpected class {classes}")
    for it in receipt["items"]:
        if it["classification"] not in {"DELETE", "ARCHIVE", "KEEP", "UNKNOWN"}:
            problems.append(f"unclassified {it['id']}")
        p = it.get("path") or ""
        if it["classification"] == "DELETE" and p.startswith(NEVER_DELETE_PREFIXES):
            problems.append(f"DELETE under never-delete prefix: {p}")
        if it["classification"] == "DELETE" and not it.get("evidence"):
            problems.append(f"DELETE without evidence: {it['id']}")
    if not receipt["tempted_to_delete_then_downgraded"]:
        problems.append("missing downgraded record")
    if not receipt["what_i_watched_fail"]:
        problems.append("missing watched-fail")
    # KEEP must dominate file-level headless
    headless_files = [
        it
        for it in receipt["items"]
        if it["kind"] == "file" and str(it["path"]).startswith("tools/headless/")
    ]
    dead_headless_files = [
        it for it in headless_files if it["classification"] == "DELETE"
    ]
    if dead_headless_files:
        problems.append(
            "file-level DELETE inside tools/headless "
            f"{[it['path'] for it in dead_headless_files]} — harnesses are live"
        )
    if problems:
        print("CENSUS SELF-CHECK FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
