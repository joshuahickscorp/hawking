"""Bounded structural source views for the Hawking tool registry.

The repository already has a durable Rust/tree-sitter index.  This module is
the small, synchronous registry adapter used by the live Python daemon while
that index is reached through its typed IPC boundary.  It uses only a bounded,
in-process rebuildable outline cache; it does not create a second durable
index. Every result is scoped to the caller's authorized root and carries a
content/scope revision so a later owner can replace the local parser without
changing the tool contract.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


STRUCTURAL_SCHEMA = "hawking.source.structural.v1"
PARSER_REVISION = "hawking-source-structure-v1"
_MAX_FILES = 800
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_RESULTS = 100
_SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi", ".rs", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".go", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".java", ".swift",
})
_SKIP_DIRS = frozenset({
    ".git", ".hawking", ".venv", "venv", "node_modules", "target", "build",
    "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".worktrees",
})
_DEFINITION_RE = re.compile(
    r"^\s*(?:(?:pub(?:\([^)]*\))?|async|unsafe|export|default|static|const|let|var)\s+)*"
    r"(fn|struct|enum|trait|impl|type|class|interface|function)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_OUTLINE_CACHE: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
_OUTLINE_CACHE_LOCK = threading.Lock()
_OUTLINE_CACHE_MAX = 256


def _is_test_path(path: Path) -> bool:
    name = path.name.casefold()
    return (
        name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts"))
        or any(part.casefold() in {"test", "tests", "__tests__"} for part in path.parts)
    )


def _limit(value: Any, default: int = 100) -> int:
    try:
        return max(1, min(_MAX_RESULTS, int(value if value is not None else default)))
    except (TypeError, ValueError):
        return default


def _language(path: Path) -> str:
    return {
        ".py": "python", ".pyi": "python", ".rs": "rust", ".js": "javascript",
        ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".c": "c",
        ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
        ".java": "java", ".swift": "swift",
    }.get(path.suffix.lower(), "unknown")


def _relative(context: Any, path: Path) -> str:
    repo = Path(getattr(context, "repo_root", path)).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(repo)).replace(os.sep, "/")
    except ValueError:
        return str(path.resolve())


def _read(path: Path) -> Optional[str]:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _files(
    context: Any,
    scope: Any = ".",
    *,
    limit: Any = None,
    include_tests: bool = False,
) -> Tuple[List[Path], bool, str]:
    """Return source files inside the existing ToolContext read boundary."""
    raw = str(scope or ".")
    root = context.resolve_read_path(raw)
    if root.is_file():
        return ([root] if root.suffix.lower() in _SOURCE_SUFFIXES else []), False, raw
    if not root.is_dir():
        raise NotADirectoryError(root)
    try:
        cap = max(1, min(_MAX_FILES, int(limit if limit is not None else _MAX_FILES)))
    except (TypeError, ValueError):
        cap = _MAX_FILES
    found: List[Path] = []
    truncated = False
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
        for name in sorted(filenames):
            candidate = Path(directory) / name
            if candidate.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            if not include_tests and _is_test_path(candidate):
                continue
            try:
                resolved = context.resolve_read_path(candidate)
                if not resolved.is_file() or resolved.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except (OSError, PermissionError, ValueError):
                continue
            found.append(resolved)
            if len(found) >= cap:
                truncated = True
                return found, truncated, raw
    return found, truncated, raw


def _build_configuration(context: Any) -> str:
    value = getattr(context, "source_build_configuration", None)
    if value is None:
        value = os.environ.get("HAWKING_SOURCE_BUILD_CONFIGURATION")
    return str(value or "default")[:160]


def _worktree_revision(context: Any) -> str:
    """Return a stable worktree identity without making it a new store.

    Content hashes remain the correctness boundary for derived outlines.  The
    worktree/HEAD component prevents two independent checkouts with otherwise
    similar paths from sharing a cached result when a caller's context moves.
    """
    repo = Path(getattr(context, "repo_root", ".")).expanduser().resolve()
    head = "nogit"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            head = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{repo}:{head}"


def _scope_revision(context: Any, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(PARSER_REVISION.encode("ascii"))
    digest.update(b"\0")
    digest.update(_build_configuration(context).encode("utf-8", "replace"))
    digest.update(b"\0")
    digest.update(_worktree_revision(context).encode("utf-8", "surrogateescape"))
    digest.update(b"\0")
    for path in sorted(files, key=str):
        try:
            stat = path.stat()
            content = path.read_bytes()
        except OSError:
            continue
        digest.update(str(path).encode("utf-8", "surrogateescape"))
        digest.update(hashlib.sha256(content).digest())
        digest.update(f"\0{stat.st_size}".encode("ascii"))
    return f"source-scope-v1:{digest.hexdigest()}"


def _node_end(node: ast.AST, fallback: int) -> int:
    return int(getattr(node, "end_lineno", None) or fallback)


def _python_outline(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [], f"SyntaxError:{exc.lineno or 0}:{exc.msg}"
    symbols: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []

    def visit(node: ast.AST, parents: Tuple[str, ...] = ()) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            name = str(node.name)
            symbols.append({
                "kind": kind,
                "name": name,
                "qualified_name": ".".join((*parents, name)),
                "line": int(getattr(node, "lineno", 1)),
                "end_line": _node_end(node, int(getattr(node, "lineno", 1))),
            })
            next_parents = (*parents, name)
        else:
            next_parents = parents
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None)
            if name is None and isinstance(func, ast.Attribute):
                name = func.attr
            if name:
                calls.append({
                    "name": str(name),
                    "line": int(getattr(node, "lineno", 1)),
                })
        for child in ast.iter_child_nodes(node):
            visit(child, next_parents)

    visit(tree)
    return symbols, calls, None


def _text_outline(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    symbols: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _DEFINITION_RE.match(line)
        if match:
            keyword = match.group(1)
            name = match.group(2)
            kind = "function" if keyword in {"fn", "function"} else keyword
            symbols.append({
                "kind": kind,
                "name": name,
                "qualified_name": name,
                "line": line_number,
                "end_line": line_number,
            })
        for call in _CALL_RE.finditer(line):
            calls.append({"name": call.group(1), "line": line_number})
    return symbols, calls, None


def _outline(context: Any, path: Path, text: Optional[str] = None) -> Dict[str, Any]:
    body = text if text is not None else _read(path)
    if body is None:
        raise ValueError(f"source file is missing or above the bounded read limit: {path}")
    content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    build_configuration = _build_configuration(context)
    worktree_revision = _worktree_revision(context)
    cache_key = (
        str(path.resolve()),
        content_sha256,
        _language(path),
        PARSER_REVISION,
        build_configuration,
        worktree_revision,
    )
    with _OUTLINE_CACHE_LOCK:
        cached = _OUTLINE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    if _language(path) == "python":
        symbols, calls, parse_error = _python_outline(body)
        backend = "python-ast.v1"
    else:
        symbols, calls, parse_error = _text_outline(body)
        backend = "bounded-text-structure.v1"
    result = {
        "path": _relative(context, path),
        "language": _language(path),
        "content_sha256": content_sha256,
        "symbols": symbols[:_MAX_RESULTS],
        "calls": calls[:_MAX_RESULTS],
        "parse_error": parse_error,
        "backend": backend,
        "schema": STRUCTURAL_SCHEMA,
        "parser_revision": PARSER_REVISION,
        "build_configuration": build_configuration,
        "worktree_revision": worktree_revision,
    }
    with _OUTLINE_CACHE_LOCK:
        if len(_OUTLINE_CACHE) >= _OUTLINE_CACHE_MAX:
            _OUTLINE_CACHE.pop(next(iter(_OUTLINE_CACHE)))
        _OUTLINE_CACHE[cache_key] = dict(result)
    return result


def source_outline(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    path = context.resolve_read_path(args.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    result = _outline(context, path)
    result["truncated"] = len(result["symbols"]) >= _MAX_RESULTS or len(result["calls"]) >= _MAX_RESULTS
    return result


def source_owner(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    files, truncated, scope = _files(context, args.get("scope") or ".", limit=args.get("max_files"))
    needle = query.casefold()
    candidates: List[Dict[str, Any]] = []
    for path in files:
        body = _read(path)
        if body is None:
            continue
        outline = _outline(context, path, body)
        score = 0
        reasons: List[str] = []
        matching_symbols = [item for item in outline["symbols"] if needle in str(item["name"]).casefold() or needle in str(item["qualified_name"]).casefold()]
        if matching_symbols:
            score += 100
            reasons.append("structural symbol")
        line_hits = []
        for line_number, line in enumerate(body.splitlines(), 1):
            if needle in line.casefold():
                line_hits.append(line_number)
        if line_hits:
            score += min(60, len(line_hits) * 5)
            reasons.append("content match")
        path_score = sum(10 for part in path.parts if needle in part.casefold())
        score += path_score
        if path_score:
            reasons.append("path match")
        if score:
            candidates.append({
                "path": _relative(context, path),
                "score": score,
                "reason": ", ".join(reasons),
                "lines": line_hits[:8],
                "symbols": [item["qualified_name"] for item in matching_symbols[:8]],
                "content_sha256": outline["content_sha256"],
            })
    candidates.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
    candidates = candidates[:_limit(args.get("max_results"), 12)]
    return {
        "schema": STRUCTURAL_SCHEMA,
        "query": query,
        "scope": scope,
        "owner": candidates[0] if candidates else None,
        "candidates": candidates,
        "files_scanned": len(files),
        "truncated": truncated,
        "index_revision": _scope_revision(context, files),
        "provenance": "hawking.source_tools.owner.v1",
    }


def source_symbol(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    files, truncated, scope = _files(context, args.get("scope") or ".", limit=args.get("max_files"))
    needle = name.casefold()
    definitions: List[Dict[str, Any]] = []
    for path in files:
        result = _outline(context, path)
        for symbol in result["symbols"]:
            if needle in str(symbol["name"]).casefold() or needle in str(symbol["qualified_name"]).casefold():
                definitions.append({"path": result["path"], **symbol, "content_sha256": result["content_sha256"]})
    definitions.sort(key=lambda item: (0 if str(item["name"]).casefold() == needle else 1, str(item["path"]), int(item["line"])))
    definitions = definitions[:_limit(args.get("max_results"), 24)]
    return {
        "schema": STRUCTURAL_SCHEMA,
        "name": name,
        "scope": scope,
        "definitions": definitions,
        "files_scanned": len(files),
        "truncated": truncated,
        "index_revision": _scope_revision(context, files),
        "provenance": "hawking.source_tools.symbol.v1",
    }


def source_references(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    symbol = str(args.get("symbol") or "").strip()
    if not symbol or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ValueError("symbol must be a simple identifier")
    files, truncated, scope = _files(context, args.get("scope") or ".", limit=args.get("max_files"))
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    rows: List[Dict[str, Any]] = []
    for path in files:
        body = _read(path)
        if body is None:
            continue
        outline = _outline(context, path, body)
        definition_lines = {int(item["line"]) for item in outline["symbols"] if item["name"] == symbol}
        for line_number, line in enumerate(body.splitlines(), 1):
            if pattern.search(line):
                rows.append({
                    "path": _relative(context, path),
                    "line": line_number,
                    "kind": "definition" if line_number in definition_lines else "reference",
                    "text": line[:1000],
                })
                if len(rows) >= _limit(args.get("max_results"), 100):
                    truncated = True
                    break
        if len(rows) >= _limit(args.get("max_results"), 100):
            break
    return {
        "schema": STRUCTURAL_SCHEMA,
        "symbol": symbol,
        "scope": scope,
        "references": rows,
        "count": len(rows),
        "truncated": truncated,
        "index_revision": _scope_revision(context, files),
        "provenance": "hawking.source_tools.references.v1",
    }


def source_ast_query(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    query = str(args.get("pattern") or "").strip()
    if not query:
        raise ValueError("pattern is required")
    if ":" in query:
        kind, name = query.split(":", 1)
        kind = kind.casefold().strip()
        name = name.strip()
    else:
        kind, name = "symbol", query
    if kind not in {"symbol", "function", "class", "call"} or not name:
        raise ValueError("pattern must be symbol:name, function:name, class:name, or call:name")
    files, truncated, scope = _files(context, args.get("scope") or ".", limit=args.get("max_files"))
    needle = name.casefold()
    matches: List[Dict[str, Any]] = []
    for path in files:
        body = _read(path)
        if body is None:
            continue
        outline = _outline(context, path, body)
        if kind == "call":
            items = [{"kind": "call", **item} for item in outline["calls"] if needle in item["name"].casefold()]
        else:
            items = [item for item in outline["symbols"] if needle in item["name"].casefold() and (kind == "symbol" or item["kind"] == kind)]
        for item in items:
            matches.append({"path": outline["path"], **item})
            if len(matches) >= _limit(args.get("max_results"), 100):
                truncated = True
                break
        if len(matches) >= _limit(args.get("max_results"), 100):
            break
    return {
        "schema": STRUCTURAL_SCHEMA,
        "pattern": query,
        "supported_patterns": ["symbol:name", "function:name", "class:name", "call:name"],
        "scope": scope,
        "matches": matches,
        "count": len(matches),
        "truncated": truncated,
        "index_revision": _scope_revision(context, files),
        "provenance": "hawking.source_tools.ast_query.v1",
    }


def source_affected_tests(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    raw_paths = args.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("paths must be a non-empty array")
    tokens = {Path(str(item)).stem.casefold() for item in raw_paths if str(item).strip()}
    tokens.update({part.casefold() for item in raw_paths for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", str(item))})
    files, truncated, scope = _files(context, args.get("scope") or ".", limit=args.get("max_files"), include_tests=True)
    tests: List[Dict[str, Any]] = []
    for path in files:
        if not (path.name.casefold().startswith("test_") or path.name.casefold().endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts")) or any(part.casefold() in {"test", "tests", "__tests__"} for part in path.parts)):
            continue
        body = _read(path)
        if body is None:
            continue
        hits = [line for line, text in enumerate(body.splitlines(), 1) if any(token and token in text.casefold() for token in tokens)]
        if hits:
            tests.append({"path": _relative(context, path), "lines": hits[:12], "matched_tokens": sorted(token for token in tokens if token in body.casefold())[:12]})
    tests = tests[:_limit(args.get("max_results"), 50)]
    return {
        "schema": STRUCTURAL_SCHEMA,
        "paths": [str(item) for item in raw_paths],
        "scope": scope,
        "tests": tests,
        "count": len(tests),
        "truncated": truncated,
        "index_revision": _scope_revision(context, files),
        "provenance": "hawking.source_tools.affected_tests.v1",
    }


def source_rewrite_preview(context: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    pattern = str(args.get("pattern") or "")
    if not pattern:
        raise ValueError("pattern is required")
    replacement = str(args.get("replacement") or "")
    files, truncated, scope = _files(context, args.get("scope") or ".", limit=args.get("max_files"))
    changes: List[Dict[str, Any]] = []
    for path in files:
        before = _read(path)
        if before is None or pattern not in before:
            continue
        count = before.count(pattern)
        after = before.replace(pattern, replacement, 1)
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=_relative(context, path), tofile=_relative(context, path), n=2,
        ))
        changes.append({
            "path": _relative(context, path),
            "matches": count,
            "anchor_unique": count == 1,
            "content_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
            "diff": diff[:4000],
        })
        if len(changes) >= _limit(args.get("max_results"), 20):
            truncated = True
            break
    return {
        "schema": STRUCTURAL_SCHEMA,
        "pattern": pattern,
        "replacement": replacement,
        "scope": scope,
        "changes": changes,
        "count": len(changes),
        "would_write": False,
        "truncated": truncated,
        "index_revision": _scope_revision(context, files),
        "provenance": "hawking.source_tools.rewrite_preview.v1",
    }


__all__ = [
    "STRUCTURAL_SCHEMA",
    "source_ast_query",
    "source_affected_tests",
    "source_outline",
    "source_owner",
    "source_references",
    "source_rewrite_preview",
    "source_symbol",
]
