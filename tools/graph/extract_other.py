"""TypeScript, Metal, git co-change, registries, runtime-hot, tests coverage."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph_model import (
    Graph,
    complexity_of,
    detect_security,
    detect_side_effects,
    lang_for,
    make_node,
    subsystem_for,
)


_CONTROL_ROOTS = ("control/", "workspace/campaign/governance/control/")
_VENDOR_ROOTS = ("vendor/", "workspace/vendor/")
_ENV_FLAGS_REL = "workspace/docs/reference/env_flags.md"
_RUNTIME_HOT_DOCS = (
    "workspace/docs/reference/BENCHMARKS.md",
    "workspace/docs/reference/kernels.md",
)


def lang_for_path(rel: str) -> str:
    return lang_for(rel)
from scanner import CodeScanner

# --- TypeScript (scanner, lower confidence) ---

RE_TS_FN = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|"
    r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|"
    r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function\b",
)
RE_TS_CLASS = re.compile(r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)")
RE_TS_IFACE = re.compile(r"(?:export\s+)?interface\s+(\w+)")
RE_TS_TYPE = re.compile(r"(?:export\s+)?type\s+(\w+)\s*=")
RE_TS_IMPORT = re.compile(
    r"""import\s+(?:type\s+)?(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]"""
)
RE_TS_EXPORT = re.compile(r"export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|type|interface)\s+(\w+)")


def extract_typescript_file(
    repo: Path, rel: str, g: Graph, indexes: dict[str, Any]
) -> None:
    path = repo / rel
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    sc = CodeScanner(text, lang="typescript")
    code = sc.slice_code(0, len(text))
    file_id = f"file:{rel}"
    conf = 0.55  # lower confidence expected

    for m in RE_TS_CLASS.finditer(code):
        name = m.group(1)
        tid = f"type:{rel}#{name}"
        line = text.count("\n", 0, m.start()) + 1
        g.add_node(make_node(
            "type", tid, name, path=rel, lang="typescript",
            span=[line, line], public=bool(re.search(r"export", code[max(0, m.start()-10):m.start()+1])),
            complexity=1,
        ))
        g.ensure_contains(file_id, tid, evidence="regex")
        indexes["types_by_name"][name].append(tid)

    for m in RE_TS_IFACE.finditer(code):
        name = m.group(1)
        tid = f"type:{rel}#{name}"
        line = text.count("\n", 0, m.start()) + 1
        g.add_node(make_node(
            "type", tid, name, path=rel, lang="typescript",
            span=[line, line], public=True, complexity=1,
        ))
        g.ensure_contains(file_id, tid, evidence="regex")
        indexes["types_by_name"][name].append(tid)

    for m in RE_TS_TYPE.finditer(code):
        name = m.group(1)
        tid = f"type:{rel}#{name}"
        line = text.count("\n", 0, m.start()) + 1
        if tid not in g.nodes:
            g.add_node(make_node(
                "type", tid, name, path=rel, lang="typescript",
                span=[line, line], public=True, complexity=1,
            ))
            g.ensure_contains(file_id, tid, evidence="regex")
            indexes["types_by_name"][name].append(tid)

    for m in RE_TS_FN.finditer(code):
        name = m.group(1) or m.group(2) or m.group(3)
        if not name:
            continue
        fid = f"fn:{rel}#{name}"
        line = text.count("\n", 0, m.start()) + 1
        brace = sc.find_next_brace(m.end())
        end_line = line
        body = ""
        if brace is not None and brace - m.end() < 200:
            matched = sc.match_braces(brace)
            if matched:
                end_line = text.count("\n", 0, matched - 1) + 1
                body = text[brace:matched]
        # public only when export appears on the same line / nearby
        line_txt = text.splitlines()[line - 1] if line <= len(text.splitlines()) else ""
        is_pub = bool(re.search(r"\bexport\b", line_txt)) or bool(
            re.search(r"\bexport\b", text[max(0, m.start() - 20):m.start() + 1])
        )
        g.add_node(make_node(
            "function", fid, name, path=rel, lang="typescript",
            span=[line, end_line],
            public=is_pub,
            complexity=complexity_of(body),
            side_effects=detect_side_effects(body),
            security_sensitive=detect_security(body),
        ))
        g.ensure_contains(file_id, fid, evidence="regex")
        indexes["fns_by_name"][name].append(fid)
        indexes["fn_meta"][fid] = {
            "crate": None, "file": rel, "name": name, "qname": name,
            "body": body, "is_test": False, "lang": "typescript",
        }

    for m in RE_TS_IMPORT.finditer(code):
        spec = m.group(1)
        if spec.startswith("."):
            # resolve relative
            base = Path(rel).parent
            cand = (base / spec).as_posix()
            for ext in ("", ".ts", ".tsx", ".js", "/index.ts"):
                p = cand + ext if ext != "/index.ts" else cand.rstrip("/") + "/index.ts"
                p = str(Path(p))  # normalize
                # try normalized without leading
                for trial in (p, p.lstrip("./")):
                    tf = f"file:{trial}"
                    if trial in indexes.get("all_files", set()) or tf in g.nodes:
                        g.add_edge(
                            file_id, "imports", tf if tf in g.nodes else f"file:{trial}",
                            evidence="regex", confidence=conf,
                        )
                        break


def extract_all_typescript(
    repo: Path, files: list[str], g: Graph, indexes: dict[str, Any]
) -> None:
    indexes.setdefault("all_files", set()).update(files)
    for rel in files:
        extract_typescript_file(repo, rel, g, indexes)


# --- Metal ---

RE_METAL_ENTRY = re.compile(
    r"\b(kernel|vertex|fragment)\s+\w[\w\s*<>,:]*?\b([A-Za-z_]\w*)\s*\("
)


def extract_metal_file(repo: Path, rel: str, g: Graph, indexes: dict[str, Any]) -> None:
    path = repo / rel
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    file_id = f"file:{rel}"
    for m in RE_METAL_ENTRY.finditer(text):
        kind, name = m.group(1), m.group(2)
        if name in ("void", "float", "half", "int", "uint", "bool", "device", "constant"):
            continue
        fid = f"fn:{rel}#{name}"
        line = text.count("\n", 0, m.start()) + 1
        g.add_node(make_node(
            "function", fid, name, path=rel, lang="metal",
            span=[line, line],
            # entry points are exported to the host, but topology public_symbols
            # does not count .metal — leave public=true for graph usefulness
            public=True,
            runtime_hot=None,
            complexity=1,
        ))
        g.ensure_contains(file_id, fid, evidence="regex")
        indexes["metal_fns"][name] = fid
        indexes["fns_by_name"][name].append(fid)


def link_metal_from_rust(repo: Path, g: Graph, indexes: dict[str, Any]) -> None:
    """Match pipeline/function-name string literals in metal glue to shader entries."""
    metal_dir = repo / "crates/hawking-core/src/metal"
    if not metal_dir.exists():
        return
    names = indexes.get("metal_fns", {})
    if not names:
        return
    for path in sorted(metal_dir.rglob("*.rs")):
        try:
            rel = str(path.relative_to(repo))
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        # string literals that equal a kernel name
        for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"', text):
            name = m.group(1)
            if name in names:
                # find enclosing function if possible — use file-level fn nodes
                # link all rust fns in this file that contain the string
                for nid, n in list(g.nodes.items()):
                    if n.type == "function" and n.path == rel and n.lang == "rust":
                        meta = indexes.get("fn_meta", {}).get(nid, {})
                        body = meta.get("body") or ""
                        if name in body or f'"{name}"' in text:
                            # only if string appears near this function's span
                            if n.span:
                                lines = text.splitlines()
                                chunk = "\n".join(
                                    lines[max(0, n.span[0] - 1):n.span[1]]
                                )
                                if f'"{name}"' in chunk or name in chunk:
                                    g.add_edge(
                                        nid, "calls", names[name],
                                        evidence="regex", confidence=0.7,
                                    )


# --- Git co-change ---

SOURCE_EXT = {
    ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".metal", ".sh", ".lean",
    ".md", ".toml",
}


def extract_git_cochange(repo: Path, g: Graph, file_set: set[str]) -> None:
    """co_changes edges + change_freq_90d / change_freq_all on file nodes."""
    r = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H", "--name-only", "--all"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        return

    # 90-day cutoff
    r90 = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H", "--name-only",
         "--since=90 days ago", "--all"],
        capture_output=True, text=True, check=False,
    )

    def parse_commits(text: str) -> list[list[str]]:
        commits: list[list[str]] = []
        cur: list[str] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if re.fullmatch(r"[0-9a-f]{7,40}", line.strip()):
                if cur:
                    commits.append(cur)
                cur = []
            else:
                p = line.strip()
                if Path(p).suffix in SOURCE_EXT or p in file_set:
                    cur.append(p)
        if cur:
            commits.append(cur)
        return commits

    all_commits = parse_commits(r.stdout)
    commits_90 = parse_commits(r90.stdout) if r90.returncode == 0 else []

    freq_all: dict[str, int] = defaultdict(int)
    freq_90: dict[str, int] = defaultdict(int)
    pair_count: dict[tuple[str, str], int] = defaultdict(int)

    for files in all_commits:
        files = sorted(set(f for f in files if f in file_set or Path(f).suffix in SOURCE_EXT))
        for f in files:
            freq_all[f] += 1
        if 2 <= len(files) <= 40:
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    a, b = files[i], files[j]
                    if a > b:
                        a, b = b, a
                    pair_count[(a, b)] += 1

    for files in commits_90:
        files = set(f for f in files if f in file_set or Path(f).suffix in SOURCE_EXT)
        for f in files:
            freq_90[f] += 1

    for f, n in freq_all.items():
        fid = f"file:{f}"
        if fid in g.nodes:
            g.nodes[fid].attrs["change_freq_all"] = n
            g.nodes[fid].attrs["change_freq_90d"] = freq_90.get(f, 0)

    for (a, b), count in pair_count.items():
        fa, fb = f"file:{a}", f"file:{b}"
        if fa not in g.nodes or fb not in g.nodes:
            continue
        min_c = min(freq_all[a], freq_all[b])
        weight = (count / min_c) if min_c else 0.0
        g.add_edge(
            fa, "co_changes", fb,
            count=count, weight=weight, evidence="git", confidence=1.0,
        )
        g.add_edge(
            fb, "co_changes", fa,
            count=count, weight=weight, evidence="git", confidence=1.0,
        )


# --- Tests coverage via call graph depth <= 3 ---

def mark_test_coverage(g: Graph, indexes: dict[str, Any]) -> None:
    # build adjacency for calls
    adj: dict[str, list[str]] = defaultdict(list)
    for e in g.edges.values():
        if e.type == "calls":
            adj[e.src].append(e.dst)

    def closure(start: str, depth: int = 3) -> set[str]:
        seen: set[str] = set()
        frontier = {start}
        for _ in range(depth):
            nxt: set[str] = set()
            for n in frontier:
                for d in adj.get(n, []):
                    if d not in seen:
                        seen.add(d)
                        nxt.add(d)
            frontier = nxt
            if not frontier:
                break
        return seen

    for tid, fid in indexes.get("test_nodes", {}).items():
        if tid not in g.nodes:
            continue
        reached = closure(fid, 3)
        reached.add(fid)
        for nid in reached:
            if nid in g.nodes:
                g.nodes[nid].attrs["test_covered"] = True
                if nid != fid and g.nodes[nid].type in ("function", "type"):
                    g.add_edge(
                        tid, "tests", nid,
                        evidence="test", confidence=0.7, weight=1.0,
                    )
        # Types named in the test body: only candidates from the same source file
        meta = indexes.get("fn_meta", {}).get(fid, {})
        body = meta.get("body") or ""
        test_file = meta.get("file") or (g.nodes[tid].path if tid in g.nodes else None)
        if not body or not test_file or len(body) > 50_000:
            continue
        for tname, tids in indexes.get("types_by_name", {}).items():
            if not (3 <= len(tname) <= 48 and tname[0].isupper()):
                continue
            same_file = [t for t in tids if t.startswith(f"type:{test_file}#")]
            if not same_file:
                continue
            if tname in body and re.search(rf"\b{re.escape(tname)}\b", body):
                for t in same_file[:2]:
                    if t in g.nodes:
                        g.nodes[t].attrs["test_covered"] = True
                        g.add_edge(
                            tid, "tests", t,
                            evidence="test", confidence=0.5, weight=1.0,
                        )


# --- Registries ---

def _extract_cmd_enum_variants(text: str) -> list[str]:
    """Return clap ``enum Cmd`` variant names from main.rs source."""
    m = re.search(r"\benum\s+Cmd\s*\{", text)
    if not m:
        return []
    i = text.find("{", m.start())
    if i < 0:
        return []
    depth = 0
    end = -1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end < 0:
        return []
    block = text[i + 1:end]
    variants: list[str] = []
    for vm in re.finditer(r"^\s+([A-Z][A-Za-z0-9]*)\s*[{(]", block, re.M):
        name = vm.group(1)
        if name in ("Ok", "Err", "Some", "None", "Cmd", "PathBuf", "String"):
            continue
        variants.append(name)
    return variants


def _match_arm_bodies(text: str) -> dict[str, str]:
    """Extract ``Cmd::Variant => body`` arm bodies from ``match cli.cmd``."""
    idx = text.find("match cli.cmd")
    if idx < 0:
        # fall back to any match on .cmd
        m = re.search(r"match\s+\w+\.cmd\b", text)
        if not m:
            return {}
        idx = m.start()
    arms: dict[str, str] = {}
    pos = idx
    # scan only within a reasonable window of the match (whole main is fine)
    limit = min(len(text), idx + 250_000)
    while pos < limit:
        m = re.search(r"Cmd::([A-Z][A-Za-z0-9]*)", text[pos:limit])
        if not m:
            break
        name = m.group(1)
        abs_start = pos + m.start()
        arrow = text.find("=>", abs_start)
        if arrow < 0 or arrow - abs_start > 3000:
            pos = abs_start + 1
            continue
        j = arrow + 2
        while j < len(text) and text[j] in " \t\n":
            j += 1
        if j >= len(text):
            break
        if text[j] == "{":
            depth = 0
            k_end = j
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        k_end = k
                        break
            body = text[j:k_end + 1]
            pos = k_end + 1
        else:
            k = j
            depth = 0
            while k < len(text):
                c = text[k]
                if c in "({[":
                    depth += 1
                elif c in ")}]":
                    depth -= 1
                elif c == "," and depth == 0:
                    break
                k += 1
            body = text[j:k]
            pos = k + 1
        # keep first body per variant (match arms, not matches! macros earlier)
        if name not in arms:
            arms[name] = body
    return arms


_CLI_CALL_SKIP = frozenset({
    "if", "for", "while", "loop", "match", "return", "break", "continue",
    "move", "async", "await", "unsafe", "let", "const", "static", "fn",
    "impl", "struct", "enum", "trait", "type", "use", "mod", "crate",
    "super", "self", "Self", "true", "false", "Some", "None", "Ok", "Err",
    "vec", "format", "println", "eprintln", "print", "write", "writeln",
    "panic", "assert", "assert_eq", "assert_ne", "todo", "unimplemented",
    "or_else", "and_then", "map_err", "ok_or_else", "unwrap_or",
    "unwrap_or_else", "as_deref", "into", "from", "clone", "to_string",
    "to_owned", "default", "new", "with", "map", "filter", "collect",
    "iter", "push", "insert", "get", "len", "is_empty", "expect", "unwrap",
    "parse", "ok", "err", "then", "cfg", "display", "is_none", "from_str",
    "var_os", "var", "not", "as_ref", "as_mut", "to_path_buf", "PathBuf",
    "String", "bool", "usize", "u64", "i64", "f64", "Box", "Arc", "Vec",
    "HashMap", "env", "std", "path", "Default", "Option", "Result",
    "drop", "into_iter", "copied", "cloned", "take", "as_str", "to_vec",
    "is_some", "is_ok", "is_err", "unwrap_or_default", "or", "and",
    "format_args", "write_fmt", "as_os_str", "to_string_lossy",
})


def _handler_calls_from_body(body: str) -> list[str]:
    """Return candidate handler function names / short ids from a match arm body."""
    # Prefer path::func and Type::method last segments, plus free functions.
    names: list[str] = []
    for m in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\s*\(",
        body,
    ):
        path = m.group(1)
        # skip std / tokio constructors etc.
        if path.startswith(("std::", "core::", "alloc::", "tokio::", "anyhow::")):
            # keep leaf if it looks like a domain handler
            leaf = path.split("::")[-1]
            if leaf not in _CLI_CALL_SKIP and not leaf[0].isupper():
                names.append(leaf)
            # also record full path leaf for crate-local resolution
            names.append(path.split("::")[-1])
            # keep middle segment for hawking_serve::run → run
            continue
        leaf = path.split("::")[-1]
        if leaf not in _CLI_CALL_SKIP:
            names.append(leaf)
    for m in re.finditer(r"(?<![:\w])([a-z_][A-Za-z0-9_]*)\s*\(", body):
        name = m.group(1)
        if name in _CLI_CALL_SKIP:
            continue
        names.append(name)
    # preserve order, unique
    return list(dict.fromkeys(names))


def _link_cli_to_handlers(
    g: Graph,
    indexes: dict[str, Any],
    cid: str,
    handler_names: list[str],
    *,
    prefer_file: str | None = None,
    confidence: float = 0.95,
) -> int:
    """Add calls edges from cli/adapter node to resolved function handlers. Returns count."""
    linked = 0
    for name in handler_names:
        cands = list(dict.fromkeys(indexes.get("fns_by_name", {}).get(name, [])))
        if prefer_file:
            same = [c for c in cands if c.startswith(f"fn:{prefer_file}#")]
            if same:
                cands = same
        # also try qualified last-segment already
        if not cands:
            cands = list(dict.fromkeys(indexes.get("fns_by_qual", {}).get(name, [])))
        # unique global only when not preferred-file filtered
        if len(cands) > 5:
            # keep same-crate-ish: prefer paths under crates/hawking
            narrowed = [
                c for c in cands
                if "/hawking" in c or (prefer_file and prefer_file in c)
            ]
            if narrowed:
                cands = narrowed
        for dst in cands[:4]:
            if dst not in g.nodes:
                continue
            g.add_edge(
                cid, "calls", dst,
                evidence="regex", confidence=confidence, weight=1.0,
            )
            linked += 1
    return linked


def extract_registries(repo: Path, g: Graph, indexes: dict[str, Any]) -> None:
    # CLI from clap in hawking main — only real Cmd enum variants, linked to handlers
    main_rel = "crates/hawking/src/main.rs"
    main_rs = repo / main_rel
    if main_rs.exists():
        text = main_rs.read_text(errors="ignore")
        variants = _extract_cmd_enum_variants(text)
        arms = _match_arm_bodies(text)
        # ShaderHash appears in match but may be missing from enum in some revisions
        for name in list(dict.fromkeys([*variants, *arms.keys()])):
            cid = f"cli:hawking {name.lower()}"
            g.add_node(make_node(
                "cli_command", cid, f"hawking {name.lower()}",
                path=main_rel, lang="rust", public=True,
            ))
            g.ensure_contains(f"file:{main_rel}", cid, evidence="registry")
            body = arms.get(name, "")
            handlers = _handler_calls_from_body(body) if body else []
            # Prefer *_main style and domain calls; still try all
            handlers_sorted = sorted(
                handlers,
                key=lambda h: (
                    0 if h.endswith("_main") or h in ("run", "main") else 1,
                    0 if "main" in h else 1,
                    h,
                ),
            )
            _link_cli_to_handlers(
                g, indexes, cid, handlers_sorted,
                prefer_file=main_rel, confidence=0.95,
            )

    # HTTP routes (axum) → adapter nodes with calls to handlers
    _extract_http_routes(repo, g, indexes)

    # Events from categories + you_events
    for rel in (
        "crates/hawking-events/src/categories.rs",
        "crates/hawking-events/src/you_events.rs",
        "crates/hide-core/src/event.rs",
    ):
        p = repo / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        for m in re.finditer(r'"([a-z][a-z0-9]+(?:\.[a-z0-9_]+)+)"', text):
            ev = m.group(1)
            if ev.count(".") < 1:
                continue
            eid = f"event:{ev}"
            if eid not in g.nodes:
                g.add_node(make_node(
                    "event", eid, ev, path=rel, lang="rust", public=True,
                ))

    # Schemas: hide-protocol + "schema": "..." in tracked JSON under root and control.
    for rel in _git_ls(repo, ["*.json"]):
        if not (rel.startswith(_CONTROL_ROOTS) or "/" not in rel or rel.startswith("crates/hide-protocol/")
                or rel.startswith("crates/hawking-adapters/generated/")):
            # root-level json + control + protocol
            if "/" in rel and not rel.startswith(_CONTROL_ROOTS) and not rel.startswith("crates/"):
                continue
        if rel.startswith(_VENDOR_ROOTS) or "/node_modules/" in rel:
            continue
        p = repo / rel
        if not p.exists() or p.stat().st_size > 5_000_000:
            continue
        try:
            # stream first 200KB
            chunk = p.read_text(errors="ignore")[:200_000]
        except OSError:
            continue
        for m in re.finditer(r'"schema"\s*:\s*"([^"]+)"', chunk):
            sid = m.group(1)
            nid = f"schema:{sid}"
            if nid not in g.nodes:
                g.add_node(make_node(
                    "schema", nid, sid, path=rel, lang="none", public=True,
                ))

    # Adapters from registry
    reg = repo / "crates/hawking-adapters/generated/HAWKING_ADAPTER_REGISTRY.json"
    if reg.exists():
        try:
            data = json.loads(reg.read_text(errors="ignore"))
            for fam in data.get("families") or []:
                fid = fam.get("id") or fam.get("name")
                if not fid:
                    continue
                aid = f"adapter:family/{fid}"
                g.add_node(make_node(
                    "adapter", aid, fid,
                    path="crates/hawking-adapters/generated/HAWKING_ADAPTER_REGISTRY.json",
                    lang="none", public=True,
                ))
                g.ensure_contains("crate:hawking-adapters", aid, evidence="registry")
        except json.JSONDecodeError:
            pass

    # Tools from hide-core tool.rs / hide-kernel
    for rel, patterns in (
        ("crates/hide-core/src/tool.rs", [r'name:\s*"([^"]+)"', r'Tool(?:Name|Id)::([A-Za-z0-9_]+)']),
        ("crates/hide-backend/src/tools.rs", [r'"([a-z][a-z0-9_./-]+)"']),
        ("crates/hide-kernel/src/extension_registry.rs", [r'"([a-z][a-z0-9_.-]+)"']),
    ):
        p = repo / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        for pat in patterns:
            for m in re.finditer(pat, text):
                name = m.group(1)
                if len(name) < 2 or len(name) > 64:
                    continue
                if name in ("true", "false", "null", "string", "object"):
                    continue
                tid = f"tool:{name}"
                if tid not in g.nodes:
                    g.add_node(make_node(
                        "tool", tid, name, path=rel, lang="rust", public=True,
                        security_sensitive=True,
                    ))

    # Operators — tools/condense/engine/operators.py is the lab registry
    op_py = repo / "tools/condense/engine/operators.py"
    if op_py.exists():
        text = op_py.read_text(errors="ignore")
        # IRREDUCIBLE_MODULES / records: first positional string is the module path
        for m in re.finditer(
            r"""(?:OperatorRecord|_record|_mod|register)\s*\(\s*["']([^"']+)["']"""
            r"""|"path"\s*:\s*["']([^"']+)["']"""
            r"""|path\s*=\s*["']([^"']+\.py)["']""",
            text,
        ):
            name = m.group(1) or m.group(2) or m.group(3)
            if not name:
                continue
            short = Path(name).stem if "/" in name or name.endswith(".py") else name
            oid = f"op:{short}"
            if oid not in g.nodes:
                g.add_node(make_node(
                    "operator", oid, short,
                    path="tools/condense/engine/operators.py",
                    lang="python", public=True,
                ))
        # string literals that look like module stems listed in the registry
        for m in re.finditer(r'"(glm52_[a-z0-9_]+|gravity_[a-z0-9_]+|gptoss_[a-z0-9_]+)"', text):
            name = m.group(1)
            oid = f"op:{name}"
            if oid not in g.nodes:
                g.add_node(make_node(
                    "operator", oid, name,
                    path="tools/condense/engine/operators.py",
                    lang="python", public=True,
                ))
        for m in re.finditer(r"^def\s+(op_\w+|\w+_operator)\b", text, re.M):
            name = m.group(1)
            oid = f"op:{name}"
            if oid not in g.nodes:
                g.add_node(make_node(
                    "operator", oid, name,
                    path="tools/condense/engine/operators.py",
                    lang="python", public=True,
                ))

    # Runtime/lab operators from hawking-core kernels / model modules
    for rel in (
        "crates/hawking-core/src/kernels/mod.rs",
        "crates/hawking-core/src/model/mod.rs",
        "tools/foundry/gravity_potency.py",
    ):
        p = repo / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        for m in re.finditer(r"""(?:operator|Operator|op_name|method)\s*[:=]\s*["']([A-Za-z0-9_./-]+)["']""", text):
            name = m.group(1)
            oid = f"op:{name}"
            if oid not in g.nodes:
                g.add_node(make_node(
                    "operator", oid, name, path=rel, lang=lang_for_path(rel), public=True,
                ))

    # Also GRAVITY_METHOD_REGISTRY
    gmr = repo / "tools/foundry/GRAVITY_METHOD_REGISTRY.json"
    if gmr.exists():
        try:
            data = json.loads(gmr.read_text(errors="ignore"))
            items = data if isinstance(data, list) else data.get("methods") or data.get("operators") or []
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in ("schema", "note", "version"):
                        continue
                    if isinstance(v, (list, dict)):
                        seq = v if isinstance(v, list) else [v]
                        for item in seq:
                            if isinstance(item, dict):
                                name = item.get("name") or item.get("id") or item.get("method")
                            elif isinstance(item, str):
                                name = item
                            else:
                                continue
                            if name:
                                oid = f"op:{name}"
                                if oid not in g.nodes:
                                    g.add_node(make_node(
                                        "operator", oid, str(name),
                                        path="tools/foundry/GRAVITY_METHOD_REGISTRY.json",
                                        lang="none", public=True,
                                    ))
                    elif isinstance(v, str) and k not in ("schema",):
                        oid = f"op:{k}"
                        if oid not in g.nodes:
                            g.add_node(make_node(
                                "operator", oid, k,
                                path="tools/foundry/GRAVITY_METHOD_REGISTRY.json",
                                lang="none", public=True,
                            ))
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("id")
                    if name:
                        oid = f"op:{name}"
                        if oid not in g.nodes:
                            g.add_node(make_node(
                                "operator", oid, str(name),
                                path="tools/foundry/GRAVITY_METHOD_REGISTRY.json",
                                lang="none", public=True,
                            ))
        except json.JSONDecodeError:
            pass

    # Feature flags from the workspace docs + env::var / os.environ.
    env_doc = repo / _ENV_FLAGS_REL
    if env_doc.exists():
        text = env_doc.read_text(errors="ignore")
        for m in re.finditer(r"\b(HAWKING_[A-Z0-9_]+)\b", text):
            name = m.group(1)
            fid = f"flag:{name}"
            if fid not in g.nodes:
                g.add_node(make_node(
                    "feature_flag", fid, name,
                    path=_ENV_FLAGS_REL, lang="markdown", public=True,
                ))

    # Scan source for env reads (sample via git ls-files limited)
    for rel in _git_ls(repo, ["*.rs", "*.py"]):
        if rel.startswith(_VENDOR_ROOTS):
            continue
        p = repo / rel
        if not p.exists() or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in re.finditer(
            r"""(?:std::env::var(?:_os)?|env::var(?:_os)?|env_on|os\.environ\.get|os\.getenv)\(\s*["']([A-Z][A-Z0-9_]+)["']""",
            text,
        ):
            name = m.group(1)
            fid = f"flag:{name}"
            if fid not in g.nodes:
                g.add_node(make_node(
                    "feature_flag", fid, name, path=rel,
                    lang="rust" if rel.endswith(".rs") else "python",
                    public=True,
                ))

    # Artifacts: .gravity / .tq / receipt kinds
    for kind in (
        "gravity", "tq", "gguf", "receipt", "campaign_ledger",
        "kernel_profile", "shard", "adapter_registry",
    ):
        aid = f"artifact:{kind}"
        g.add_node(make_node(
            "artifact", aid, kind, path=None, lang="none", public=True,
        ))

    # State stores
    state_specs = [
        ("sqlite", "sessions"),
        ("sqlite", "events"),
        ("jsonl", "campaign_ledger"),
        ("jsonl", "event_log"),
        ("fs", "prefix_cache"),
        ("fs", "prefill_cache"),
        ("fs", "gravity_artifact"),
        ("memory", "kv_cache"),
        ("memory", "session_graph"),
    ]
    # discover from hide-core state.rs
    for rel in (
        "crates/hide-core/src/state.rs",
        "crates/hide-core/src/persistence.rs",
        "crates/hawking-seed-c/src/state.rs",
        "crates/hawking-core/src/cache/mod.rs",
    ):
        p = repo / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        for m in re.finditer(r"(?:table|TABLE|store|Store|ledger|Ledger)\s*[:=]\s*\"([^\"]+)\"", text):
            state_specs.append(("sqlite" if "sql" in rel.lower() else "fs", m.group(1)))
        for m in re.finditer(r'create\s+table\s+(?:if\s+not\s+exists\s+)?[`"]?(\w+)', text, re.I):
            state_specs.append(("sqlite", m.group(1)))

    seen_state: set[str] = set()
    for store, key in state_specs:
        sid = f"state:{store}/{key}"
        if sid in seen_state:
            continue
        seen_state.add(sid)
        g.add_node(make_node(
            "state", sid, f"{store}/{key}", path=None, lang="none", public=True,
        ))

    # Wire reads_state / writes_state from fn names / bodies mentioning stores
    state_ids = {n.id: n for n in g.nodes.values() if n.type == "state"}
    if state_ids:
        for fid, meta in indexes.get("fn_meta", {}).items():
            body = (meta.get("body") or "") + " " + (meta.get("name") or "")
            low = body.lower()
            for sid, sn in state_ids.items():
                key = sn.name.split("/")[-1].lower()
                if len(key) < 3:
                    continue
                if key in low:
                    if any(w in low for w in ("write", "insert", "append", "save", "put", "store", "update")):
                        g.add_edge(fid, "writes_state", sid, evidence="regex", confidence=0.45)
                    if any(w in low for w in ("read", "get", "load", "fetch", "query", "select", "open")):
                        g.add_edge(fid, "reads_state", sid, evidence="regex", confidence=0.45)


def _extract_http_routes(repo: Path, g: Graph, indexes: dict[str, Any]) -> None:
    """Create adapter:http/... nodes for axum ``.route(path, method(handler))`` sites."""
    # Scan common serve / backend HTTP entry files
    candidates = [
        "crates/hawking-serve/src/http.rs",
        "crates/hawking-serve/src/lib.rs",
        "crates/hide-backend/src/http.rs",
        "crates/hide-backend/src/server.rs",
        "crates/hide-kernel/src/tooling_mcp.rs",
    ]
    # Also any .rs under hawking-serve and hide-backend mentioning .route(
    extra: list[str] = []
    for root in ("crates/hawking-serve", "crates/hide-backend", "crates/hide-kernel"):
        p = repo / root
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*.rs")):
            try:
                rel = str(f.relative_to(repo))
            except ValueError:
                continue
            if rel not in candidates:
                extra.append(rel)
    seen_files: list[str] = []
    for rel in [*candidates, *extra]:
        if rel in seen_files:
            continue
        seen_files.append(rel)
        path = repo / rel
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if ".route(" not in text and "Router::new" not in text:
            continue
        # .route("/path", get(handler)) or post(handler) / put / delete / patch
        for m in re.finditer(
            r"""\.route\(\s*["']([^"']+)["']\s*,\s*(get|post|put|delete|patch|head|options)\s*\(\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\)""",
            text,
        ):
            route_path, method, handler = m.group(1), m.group(2), m.group(3)
            leaf = handler.split("::")[-1]
            # id: adapter:http/<METHOD> <path>
            safe_path = route_path if len(route_path) < 80 else route_path[:80]
            aid = f"adapter:http/{method.upper()} {safe_path}"
            if aid not in g.nodes:
                g.add_node(make_node(
                    "adapter", aid, f"{method.upper()} {safe_path}",
                    path=rel, lang="rust", public=True,
                ))
                # contain under file when present
                fid = f"file:{rel}"
                if fid in g.nodes:
                    g.ensure_contains(fid, aid, evidence="regex")
            _link_cli_to_handlers(
                g, indexes, aid, [leaf],
                prefer_file=rel, confidence=0.95,
            )


def _git_ls(repo: Path, globs: list[str] | None = None) -> list[str]:
    args = ["git", "-C", str(repo), "ls-files"]
    if globs:
        args.extend(globs)
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return []
    return [ln for ln in r.stdout.splitlines() if ln]


# --- Runtime hot (static only) ---

def mark_runtime_hot(repo: Path, g: Graph, indexes: dict[str, Any]) -> None:
    """Static evidence only — NOT traced. Marks runtime_hot=true from docs + bench + metal proximity."""
    # Prefer longer, more specific identifiers from docs (avoid stop-words).
    STOP = frozenset({
        "with", "from", "this", "that", "when", "then", "than", "have", "been",
        "will", "into", "over", "under", "after", "before", "about", "above",
        "true", "false", "none", "null", "type", "data", "name", "path", "file",
        "test", "main", "self", "size", "byte", "time", "mode", "case", "default",
        "value", "error", "result", "option", "model", "layer", "token", "batch",
        "input", "output", "config", "state", "cache", "load", "save", "init",
        "start", "stop", "open", "close", "read", "write", "call", "run",
    })
    hot_names: set[str] = set()
    for rel in _RUNTIME_HOT_DOCS:
        p = repo / rel
        if p.exists():
            text = p.read_text(errors="ignore")
            # backtick-quoted identifiers and long snake_case tokens
            for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]{4,})`", text):
                hot_names.add(m.group(1))
            for m in re.finditer(r"\b([a-z][a-z0-9_]{5,})\b", text):
                if m.group(1) not in STOP:
                    hot_names.add(m.group(1))

    seeds: set[str] = set()

    # functions in hawking-bench / kernel_bench
    for nid, n in g.nodes.items():
        if n.type != "function":
            continue
        if n.path and (
            n.path.startswith("crates/hawking-bench/")
            or "kernel_bench" in (n.path or "")
        ):
            n.attrs["runtime_hot"] = True
            seeds.add(nid)
            hot_names.add(n.name.split("::")[-1])

    # metal kernels are hot by nature
    for name, fid in indexes.get("metal_fns", {}).items():
        if fid in g.nodes:
            g.nodes[fid].attrs["runtime_hot"] = True
            seeds.add(fid)
            hot_names.add(name)

    # exact name match against curated hot_names (length >= 6)
    for nid, n in g.nodes.items():
        if n.type != "function":
            continue
        short = n.name.split("::")[-1]
        if short in hot_names and len(short) >= 6:
            n.attrs["runtime_hot"] = True
            seeds.add(nid)

    # seed: metal dispatch glue and decode-loop names (narrow patterns)
    for nid, n in g.nodes.items():
        if n.type != "function" or not n.name:
            continue
        low = n.name.lower()
        if any(
            k in low
            for k in (
                "decode_token", "decode_step", "mla_decode", "flash_attn",
                "dispatch_threads", "encode_compute", "generate_tokens",
                "forward_layer", "prefill_", "gemm_q", "sample_argmax",
            )
        ):
            n.attrs["runtime_hot"] = True
            seeds.add(nid)
        if n.path and n.path.startswith("crates/hawking-core/src/metal/"):
            n.attrs["runtime_hot"] = True
            seeds.add(nid)

    # 2-hop from seeds only (not from every name containing "forward")
    adj: dict[str, list[str]] = defaultdict(list)
    radj: dict[str, list[str]] = defaultdict(list)
    for e in g.edges.values():
        if e.type == "calls":
            adj[e.src].append(e.dst)
            radj[e.dst].append(e.src)

    frontier = set(seeds)
    reached = set(seeds)
    for _ in range(2):
        nxt: set[str] = set()
        for n in frontier:
            for c in radj.get(n, []):
                if c not in reached:
                    reached.add(c)
                    nxt.add(c)
            for c in adj.get(n, []):
                if c not in reached:
                    reached.add(c)
                    nxt.add(c)
        frontier = nxt
    for nid in reached:
        if nid in g.nodes and g.nodes[nid].type == "function":
            g.nodes[nid].attrs["runtime_hot"] = True
