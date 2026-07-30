#!/usr/bin/env python3.12
"""Deterministic assertion-case extractor for Core F rung F1.

Enumerates stable CASE.<kind>.<slug> identities from a git revision without
executing repository tests, importing product code, or depending on wall-clock
time. Stdlib only.

Reads exact git tree/blob objects via `git ls-tree` / `git show` (no checkout).
Ledger generation defaults to HEAD (or --rev). --check re-extracts at the
ledger's sealed_at_commit (or an explicit identical --rev), not the worktree.

    python3.12 tools/verify/case_extract.py --json
    python3.12 tools/verify/case_extract.py --write control/ASSERTION_LEDGER.json
    python3.12 tools/verify/case_extract.py --check control/ASSERTION_LEDGER.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXTRACTOR_VERSION = "hawking.case_extract.v2"
SCHEMA = "hawking.assertion_ledger.v1"

# Inventory-compatible attr form, plus parenthesised tokio::test(...).
RS_TEST_ATTR = re.compile(
    r"#\[(?:\s*)(?:tokio::)?test(?:\s*\([^]]*\))?\s*\]"
)
RS_FN_AFTER = re.compile(r"(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
TS_CALL = re.compile(
    r"""(?P<kw>\b(?:it|test))\s*\(\s*(?P<q>['"`])(?P<title>(?:\\.|(?!(?P=q)).)*?)(?P=q)"""
)
TS_DESCRIBE = re.compile(
    r"""\bdescribe\s*\(\s*(?P<q>['"`])(?P<title>(?:\\.|(?!(?P=q)).)*?)(?P=q)"""
)


def get_root() -> Path:
    env = os.environ.get("HAWKING_CASE_EXTRACT_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(get_root()), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def resolve_rev(rev: str | None = None) -> str:
    return git("rev-parse", rev or "HEAD").strip()


def tracked(rev: str, *suffixes: str) -> list[str]:
    files = [
        p
        for p in git("ls-tree", "-r", "--name-only", rev).splitlines()
        if p and not p.startswith("vendor/")
    ]
    if not suffixes:
        return files
    return [p for p in files if any(p.endswith(s) for s in suffixes)]


def read_text(rev: str, rel: str) -> str:
    """Read a blob at rev:rel. Does not touch the worktree."""
    try:
        return git("show", f"{rev}:{rel}")
    except subprocess.CalledProcessError:
        return ""


def read_artifact(rev: str, name: str) -> str:
    """Read a REBUILD_* artifact at `rev`, across the evidence/ move.

    The artifacts live under ``evidence/rebuild/`` as of this revision but sat
    at the repository root before it, and `read_text` returns "" for a missing
    blob rather than raising. Without trying both, comparing against any
    pre-move revision silently parses an empty document and reports every
    behaviour as missing.
    """
    return read_text(rev, f"evidence/rebuild/{name}") or read_text(rev, name)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def path_slug(rel: str) -> str:
    p = rel.replace("\\", "/")
    for ext in (".rs", ".py", ".ts", ".tsx", ".js", ".jsx"):
        if p.endswith(ext):
            p = p[: -len(ext)]
            break
    p = p.replace("-", "_").replace("/", "_").replace(".", "_")
    p = re.sub(r"_+", "_", p).strip("_")
    return p


def title_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "untitled"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def param_suffix(param: dict[str, Any] | None) -> str:
    if not param:
        return ""
    parts = []
    for k in sorted(param):
        parts.append(f"{k}={canonical_json(param[k])}")
    return "#" + "#".join(parts)


def _decode_js_string(raw: str) -> str:
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# Shared code-region / span helpers (Rust + TypeScript)
# ---------------------------------------------------------------------------


def _js_regex_start(src: str, i: int) -> bool:
    """Heuristic: `/` begins a regex literal, not division or a comment."""
    if i + 1 < len(src) and src[i + 1] in "/*":
        return False
    j = i - 1
    while j >= 0 and src[j] in " \t":
        j -= 1
    if j < 0:
        return True
    prev = src[j]
    # After operators/punct where a regex is legal (not binary division).
    if prev in "=([,!&|?{:;~^%*+>\n\r":
        return True
    # after `return /re/` or `case /re/:`
    k = j
    while k >= 0 and (src[k].isalnum() or src[k] == "_"):
        k -= 1
    word = src[k + 1 : j + 1]
    return word in {"return", "case", "throw", "in", "of", "typeof", "void", "delete", "else"}


def _scan_js_regex(src: str, i: int) -> int:
    """i points at opening `/`. Return index after optional flags."""
    j = i + 1
    n = len(src)
    in_class = False
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "[" and not in_class:
            in_class = True
            j += 1
            continue
        if c == "]" and in_class:
            in_class = False
            j += 1
            continue
        if c == "/" and not in_class:
            j += 1
            while j < n and src[j] in "gimsuybdv":
                j += 1
            return j
        if c == "\n":
            return j
        j += 1
    return j


def _code_mask_c_like(src: str, *, line_comments: bool = True) -> bytearray:
    """Mark real code as 1; comments, strings/templates, and JS regex literals as 0."""
    n = len(src)
    mask = bytearray(b"\x01" * n)
    i = 0
    while i < n:
        c = src[i]
        if line_comments and c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            mask[i:j] = b"\x00" * (j - i)
            i = j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            mask[i:j] = b"\x00" * (j - i)
            i = j
            continue
        if c == "/" and _js_regex_start(src, i):
            j = _scan_js_regex(src, i)
            mask[i:j] = b"\x00" * (j - i)
            i = j
            continue
        if c in ("'", '"', "`"):
            j = _scan_js_string(src, i)
            mask[i:j] = b"\x00" * (j - i)
            i = j
            continue
        i += 1
    return mask


def _scan_js_string(src: str, i: int) -> int:
    """i points at opening quote. Return index after the string/template."""
    quote = src[i]
    n = len(src)
    j = i + 1
    while j < n:
        if src[j] == "\\":
            j += 2
            continue
        if quote == "`" and src[j] == "$" and j + 1 < n and src[j + 1] == "{":
            # ${ expression }: full sub-lex (strings, regex, comments, braces).
            j += 2
            depth = 1
            while j < n and depth:
                ch = src[j]
                if ch == "/" and j + 1 < n and src[j + 1] == "/":
                    nl = src.find("\n", j)
                    j = n if nl < 0 else nl
                    continue
                if ch == "/" and j + 1 < n and src[j + 1] == "*":
                    end = src.find("*/", j + 2)
                    j = n if end < 0 else end + 2
                    continue
                if ch == "/" and _js_regex_start(src, j):
                    j = _scan_js_regex(src, j)
                    continue
                if ch in ("'", '"', "`"):
                    j = _scan_js_string(src, j)
                    continue
                if ch == "{":
                    depth += 1
                    j += 1
                    continue
                if ch == "}":
                    depth -= 1
                    j += 1
                    continue
                j += 1
            continue
        if src[j] == quote:
            return j + 1
        j += 1
    return j


def _match_delimited(
    src: str, start: int, open_ch: str, close_ch: str, mask: bytearray | None = None
) -> int:
    """Return index after matching close_ch starting at open_ch position start."""
    if start >= len(src) or src[start] != open_ch:
        return start
    depth = 0
    i = start
    n = len(src)
    while i < n:
        if mask is not None and not mask[i]:
            i += 1
            continue
        c = src[i]
        if c == open_ch:
            depth += 1
            i += 1
            continue
        if c == close_ch:
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        # Strings / comments when no mask supplied.
        if mask is None:
            if c == "/" and i + 1 < n and src[i + 1] == "/":
                j = src.find("\n", i)
                i = n if j < 0 else j + 1
                continue
            if c == "/" and i + 1 < n and src[i + 1] == "*":
                j = src.find("*/", i + 2)
                i = n if j < 0 else j + 2
                continue
            if c in ("'", '"', "`"):
                quote = c
                i += 1
                while i < n:
                    if src[i] == "\\":
                        i += 2
                        continue
                    if src[i] == quote:
                        i += 1
                        break
                    i += 1
                continue
        i += 1
    return n


def _rust_item_end(src: str, fn_name_end: int) -> int:
    """After fn name (and signature), find end of function body or trailing `;`."""
    i = fn_name_end
    n = len(src)
    while i < n and src[i] not in "{;":
        # Skip strings in where-clauses / types rarely present on test fns.
        if src[i] in ("'", '"'):
            q = src[i]
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == q:
                    i += 1
                    break
                i += 1
            continue
        i += 1
    if i >= n:
        return fn_name_end
    if src[i] == ";":
        return i + 1
    return _match_delimited(src, i, "{", "}")


def _ts_call_end(src: str, open_paren: int, mask: bytearray) -> int:
    """open_paren points at '(' of it(/test(/describe(. Return index after ')'."""
    return _match_delimited(src, open_paren, "(", ")", mask)


def make_entry(
    *,
    kind: str,
    source_path: str,
    symbol: str,
    param: dict[str, Any] | None = None,
    seed: str | None = None,
    bc: str | None = None,
    fingerprint_material: str,
    status: str | None = None,
    notes: str | None = None,
    describe_chain: list[str] | None = None,
) -> dict[str, Any]:
    slug_base = path_slug(source_path) if source_path else symbol
    if kind in ("bb", "mig", "perf"):
        case_id = f"CASE.{kind}.{symbol}"
    elif kind == "ts_it":
        # path + lexical describe chain + literal title (+ content digest if needed)
        parts = [f"CASE.{kind}.{slug_base}"]
        for d in describe_chain or []:
            parts.append(title_slug(d))
        parts.append(title_slug(symbol))
        case_id = "::".join(parts)
        if seed:
            # Content/structure digest only — never line numbers or discovery order.
            case_id += f"#d={seed}" if not str(seed).startswith("d=") else f"#{seed}"
    else:
        case_id = f"CASE.{kind}.{slug_base}::{symbol}{param_suffix(param)}"
        if seed:
            case_id += f"#seed={seed}"
    return {
        "case_id": case_id,
        "kind": kind,
        "source_path": source_path.replace("\\", "/") if source_path else "",
        "symbol": symbol,
        "param": param,
        "seed": seed,
        "bc": bc,
        "content_fingerprint": sha256_hex(fingerprint_material),
        "status": status,
        "notes": notes,
    }


def extract_rust(rev: str, warnings: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in tracked(rev, ".rs"):
        src = read_text(rev, rel)
        if "#[test]" not in src and "#[tokio::test" not in src and "#[ tokio::test" not in src:
            if not RS_TEST_ATTR.search(src):
                continue
        kind = "rust_int" if "/tests/" in rel or rel.startswith("tests/") else "rust_unit"
        for m in RS_TEST_ATTR.finditer(src):
            window = src[m.end() : m.end() + 400]
            fm = RS_FN_AFTER.search(window)
            if not fm:
                warnings.append(
                    f"rust_attr_without_fn:{rel}:{src.count(chr(10), 0, m.start()) + 1}"
                )
                continue
            fn = fm.group(1)
            fn_name_end = m.end() + fm.end()
            item_end = _rust_item_end(src, fn_name_end)
            # Whole deterministic test item: attributes + signature + body.
            span = src[m.start() : item_end]
            norm = re.sub(r"\s+", " ", span.strip())
            out.append(
                make_entry(
                    kind=kind,
                    source_path=rel,
                    symbol=fn,
                    fingerprint_material=f"{rel}\n{norm}",
                )
            )
    return out


def _const_eval(node: ast.AST) -> Any:
    """Evaluate a constant AST fragment without importing or executing repo code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_const_eval(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _const_eval(k): _const_eval(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        v = _const_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        return not v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        l, r = _const_eval(node.left), _const_eval(node.right)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        return l * r
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[node.id]
    raise ValueError("non-literal")


def _parametrize_from_decorator(dec: ast.AST) -> tuple[list[str], list[Any]] | None:
    """Return (argnames, rows) for a pytest.mark.parametrize decorator, or None if dynamic."""
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not (isinstance(func, ast.Attribute) and func.attr == "parametrize"):
        return None
    if not dec.args:
        return None
    try:
        names_raw = _const_eval(dec.args[0])
    except (ValueError, TypeError):
        return None
    if not isinstance(names_raw, str):
        return None
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    if len(dec.args) < 2:
        return None
    try:
        values = _const_eval(dec.args[1])
    except (ValueError, TypeError):
        return None
    if not isinstance(values, list):
        return None
    return names, list(values)


def _expand_param_rows(
    decorators: list[ast.AST], rel: str, fn: str, warnings: list[str]
) -> list[dict[str, Any]] | None:
    """Return list of param dicts (cartesian product), [] if no parametrize, None if dynamic."""
    layers: list[tuple[list[str], list[Any]]] = []
    saw_param = False
    for dec in decorators:
        parsed = _parametrize_from_decorator(dec)
        if parsed is None:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if dec.func.attr == "parametrize":
                    saw_param = True
                    warnings.append(f"py_parametrize_unparsed:{rel}::{fn}")
                    return None
            continue
        saw_param = True
        layers.append(parsed)
    if not saw_param:
        return []
    rows: list[dict[str, Any]] = [{}]
    for names, values in layers:
        next_rows: list[dict[str, Any]] = []
        for base in rows:
            for val in values:
                d = dict(base)
                if len(names) == 1:
                    d[names[0]] = val
                else:
                    if not isinstance(val, (list, tuple)) or len(val) != len(names):
                        warnings.append(f"py_parametrize_arity:{rel}::{fn}")
                        return None
                    for n, v in zip(names, val):
                        d[n] = v
                next_rows.append(d)
        rows = next_rows
    return rows


def extract_python(rev: str, warnings: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in tracked(rev, ".py"):
        name = Path(rel).name
        if not (name.startswith("test_") or "/tests/" in rel or rel.startswith("tests/")):
            continue
        src = read_text(rev, rel)
        if "def test_" not in src and "async def test_" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            warnings.append(f"py_syntax_error:{rel}:{e.lineno}")
            continue

        def handle_fn(item: ast.AST, class_name: str | None) -> None:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            if not item.name.startswith("test_"):
                return
            fn = f"{class_name}.{item.name}" if class_name else item.name
            try:
                segment = ast.get_source_segment(src, item) or fn
            except Exception:
                segment = fn
            norm = re.sub(r"\s+", " ", segment.strip())
            expanded = _expand_param_rows(list(item.decorator_list), rel, fn, warnings)
            if expanded is None:
                out.append(
                    make_entry(
                        kind="py_fn",
                        source_path=rel,
                        symbol=fn,
                        fingerprint_material=f"{rel}\n{fn}\n{norm}",
                        notes="unparsed_parametrize",
                    )
                )
            elif not expanded:
                out.append(
                    make_entry(
                        kind="py_fn",
                        source_path=rel,
                        symbol=fn,
                        fingerprint_material=f"{rel}\n{fn}\n{norm}",
                    )
                )
            else:
                for row in expanded:
                    out.append(
                        make_entry(
                            kind="py_param",
                            source_path=rel,
                            symbol=fn,
                            param=row,
                            fingerprint_material=(
                                f"{rel}\n{fn}\n{canonical_json(row)}\n{norm}"
                            ),
                        )
                    )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    handle_fn(item, node.name)
            else:
                handle_fn(node, None)
    return out


def extract_typescript(rev: str, warnings: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = [
        p
        for p in tracked(rev, ".ts", ".tsx")
        if p.startswith("app/") and (p.endswith(".test.ts") or p.endswith(".test.tsx"))
    ]
    for rel in files:
        src = read_text(rev, rel)
        mask = _code_mask_c_like(src)

        # Lexical describe ranges for durable identity (never line numbers).
        describes: list[tuple[int, int, str]] = []  # (start, end, title)
        for m in TS_DESCRIBE.finditer(src):
            if not mask[m.start()]:
                continue
            # Find '(' after describe
            paren = src.find("(", m.start())
            if paren < 0:
                continue
            end = _ts_call_end(src, paren, mask)
            title = _decode_js_string(m.group("title"))
            describes.append((m.start(), end, title))

        # Literal it/test titles in real code only.
        hits: list[tuple[int, int, str, str, tuple[str, ...]]] = []
        # (start, end, title, call_norm, describe_chain)
        for m in TS_CALL.finditer(src):
            if not mask[m.start()]:
                continue
            # Skip method calls like foo.it( — require word-boundary already in regex.
            if m.start() > 0 and (src[m.start() - 1].isalnum() or src[m.start() - 1] == "_"):
                continue
            paren = src.find("(", m.start())
            if paren < 0:
                continue
            end = _ts_call_end(src, paren, mask)
            title = _decode_js_string(m.group("title"))
            call_span = src[m.start() : end]
            norm = re.sub(r"\s+", " ", call_span.strip())
            chain = tuple(
                d_title
                for s, e, d_title in sorted(describes, key=lambda x: x[0])
                if s < m.start() < e
            )
            hits.append((m.start(), end, title, norm, chain))

        # Group by (describe_chain, title_slug). Same chain+title + different content
        # → stable content digest. Identical content → same id → collision fail later.
        groups: dict[tuple[tuple[str, ...], str], list[int]] = {}
        for i, (_s, _e, title, _norm, chain) in enumerate(hits):
            key = (chain, title_slug(title))
            groups.setdefault(key, []).append(i)

        digest_for: dict[int, str | None] = {}
        for _key, idxs in groups.items():
            if len(idxs) == 1:
                digest_for[idxs[0]] = None
                continue
            # Multiple hits share chain+title: disambiguate by content digest.
            for i in idxs:
                digest_for[i] = sha256_hex(hits[i][3])[:12]

        for i, (_s, _e, title, norm, chain) in enumerate(hits):
            out.append(
                make_entry(
                    kind="ts_it",
                    source_path=rel,
                    symbol=title,
                    seed=digest_for[i],
                    describe_chain=list(chain),
                    fingerprint_material=f"{rel}\n{'/'.join(chain)}\n{norm}",
                    notes=(
                        f"describe_chain:{'/'.join(title_slug(c) for c in chain)}"
                        if chain
                        else None
                    ),
                )
            )

        # Non-literal it/test factories in code (warn; do not fabricate).
        literal_starts = {h[0] for h in hits}
        for m in re.finditer(r"\b(?:it|test)\s*\(", src):
            if not mask[m.start()]:
                continue
            if m.start() in literal_starts:
                continue
            if m.start() > 0 and src[m.start() - 1] == ".":
                continue
            tail = src[m.end() : m.end() + 80].lstrip()
            if tail[:1] in ("'", '"', "`"):
                continue
            warnings.append(
                f"ts_nonliteral_title:{rel}:{src.count(chr(10), 0, m.start()) + 1}"
            )
    return out


def extract_bb(rev: str, warnings: list[str]) -> list[dict[str, Any]]:
    bc_path = "REBUILD_BEHAVIOUR_CONSTITUTION.json"
    mx_path = "REBUILD_BLACKBOX_TEST_MATRIX.json"
    bc = json.loads(read_artifact(rev, bc_path) or "{}")
    mx = json.loads(read_artifact(rev, mx_path) or "{}")
    behaviours = {b["id"]: b for b in bc.get("behaviours", []) if "id" in b}
    checks = {c["behaviour_id"]: c for c in mx.get("checks", []) if "behaviour_id" in c}
    bc_ids = set(behaviours)
    mx_ids = set(checks)
    only_bc = sorted(bc_ids - mx_ids)
    only_mx = sorted(mx_ids - bc_ids)
    if only_bc:
        warnings.append(f"bc_not_in_matrix:{len(only_bc)}:{','.join(only_bc)}")
    if only_mx:
        warnings.append(f"matrix_not_in_bc:{len(only_mx)}:{','.join(only_mx)}")
    out: list[dict[str, Any]] = []
    for bid in sorted(bc_ids | mx_ids):
        b = behaviours.get(bid, {})
        c = checks.get(bid, {})
        if c:
            status = "runnable" if c.get("runnable_now") else "not_runnable"
            if c.get("blocker"):
                status = f"{status}:{c['blocker']}"
            source = mx_path
            material = canonical_json(
                {
                    "behaviour_id": bid,
                    "runnable_now": c.get("runnable_now"),
                    "blocker": c.get("blocker"),
                    "command": c.get("command"),
                    "assertion": c.get("assertion"),
                }
            )
        else:
            status = f"matrix_missing:{b.get('verification_status', 'unknown')}"
            source = bc_path
            material = canonical_json(
                {
                    "id": bid,
                    "verification_status": b.get("verification_status"),
                    "criticality": b.get("criticality"),
                    "domain": b.get("domain"),
                }
            )
        out.append(
            make_entry(
                kind="bb",
                source_path=source,
                symbol=bid,
                bc=bid,
                fingerprint_material=material,
                status=status,
            )
        )
    return out


def extract_mig(rev: str, warnings: list[str]) -> list[dict[str, Any]]:
    path = "REBUILD_DATA_MIGRATION_CONTRACT.json"
    doc = json.loads(read_artifact(rev, path) or "{}")
    out: list[dict[str, Any]] = []
    for item in doc.get("items", []):
        mid = item.get("id")
        if not mid:
            warnings.append("mig_item_missing_id")
            continue
        sample_exists = bool(item.get("sample_exists"))
        status = "available" if sample_exists else "blocked_fixture"
        material = canonical_json(
            {
                "id": mid,
                "name": item.get("name"),
                "location": item.get("location"),
                "format": item.get("format"),
                "sample_path": item.get("sample_path"),
                "sample_exists": sample_exists,
                "migration_policy": item.get("migration_policy"),
            }
        )
        out.append(
            make_entry(
                kind="mig",
                source_path=path,
                symbol=mid,
                fingerprint_material=material,
                status=status,
            )
        )
    return out


def extract_perf(rev: str, warnings: list[str]) -> list[dict[str, Any]]:
    path = "REBUILD_PERFORMANCE_BASELINE_MEASURED.json"
    doc = json.loads(read_artifact(rev, path) or "{}")
    out: list[dict[str, Any]] = []
    for metric in doc.get("metrics", []):
        name = metric.get("name")
        if not name:
            warnings.append("perf_metric_missing_name")
            continue
        status = metric.get("status") or "unknown"
        material = canonical_json(
            {
                "name": name,
                "family": metric.get("family"),
                "status": status,
                "reason": metric.get("reason"),
                "unit": metric.get("unit"),
                "higher_is_better": metric.get("higher_is_better"),
            }
        )
        out.append(
            make_entry(
                kind="perf",
                source_path=path,
                symbol=name,
                fingerprint_material=material,
                status=status,
            )
        )
    return out


def build_ledger(rev: str | None = None) -> dict[str, Any]:
    commit = resolve_rev(rev)
    warnings: list[str] = []
    entries: list[dict[str, Any]] = []
    entries.extend(extract_rust(commit, warnings))
    entries.extend(extract_python(commit, warnings))
    entries.extend(extract_typescript(commit, warnings))
    entries.extend(extract_bb(commit, warnings))
    entries.extend(extract_mig(commit, warnings))
    entries.extend(extract_perf(commit, warnings))

    entries.sort(key=lambda e: e["case_id"])
    warnings = sorted(set(warnings))

    ids = [e["case_id"] for e in entries]
    counts = Counter(ids)
    collisions = sorted(cid for cid, n in counts.items() if n > 1)
    if collisions:
        sample = collisions[:10]
        raise SystemExit(
            f"case_id collision ({len(collisions)}): {sample}"
            + (" …" if len(collisions) > 10 else "")
        )

    by_kind = dict(sorted(Counter(e["kind"] for e in entries).items()))
    source_kinds = ("rust_unit", "rust_int", "py_fn", "py_param", "ts_it")
    source_total = sum(by_kind.get(k, 0) for k in source_kinds)
    obligation_kinds = ("bb", "mig", "perf")
    obligation_total = sum(by_kind.get(k, 0) for k in obligation_kinds)

    overlap_policy = (
        "Categories are disjoint by kind. Source cases (rust_unit, rust_int, py_fn, "
        "py_param, ts_it) are enumerated from source attributes/functions/titles. "
        "Constitution obligations (bb, mig, perf) use public ids and never share a "
        "case_id with a source case. Totals are a simple sum of by_kind — never add "
        "category totals that re-represent the same case_id. Inventory v1 counts one "
        "case per Rust test attr and one per Python test function (no AST param "
        "expansion, no Vitest); this ledger expands literal pytest.mark.parametrize "
        "rows via AST and adds ts_it/bb/mig/perf. Opaque Rust loops remain one case. "
        "Vitest identities use path + lexical describe chain + literal title; "
        "content digest only for same-chain duplicate titles; identical duplicates "
        "collision-fail. Fingerprints bind full Rust test items and full Vitest calls."
    )

    return {
        "schema": SCHEMA,
        "extractor_version": EXTRACTOR_VERSION,
        "sealed_at_commit": commit,
        "authority": "controller",
        "source_tools": {
            "inventory": "tools/loc/hawking_inventory.py --tests",
            "extractor": "tools/verify/case_extract.py",
        },
        "counts": {
            "total": len(entries),
            "by_kind": by_kind,
            "source_cases": source_total,
            "obligation_cases": obligation_total,
            "inventory_logical_cases_v1_reference": 3962,
            "overlap_policy": overlap_policy,
        },
        "warnings": warnings,
        "entries": entries,
    }


def dumps_ledger(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def semantic_payload(doc: dict[str, Any]) -> str:
    """Byte-stable semantic view for --check (no wall-clock fields exist)."""
    return dumps_ledger(doc)


def check_ledger(path: Path, rev_arg: str | None = None) -> tuple[int, list[str]]:
    """Re-extract at sealed_at_commit (or identical --rev) and compare.

    Uses git object reads only — never the current worktree or HEAD unless the
    sealed pin resolves to HEAD.
    """
    msgs: list[str] = []
    if not path.exists():
        return 2, [f"missing ledger: {path}"]
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return 2, [f"invalid JSON in {path}: {e}"]

    sealed = old.get("sealed_at_commit")
    if not sealed:
        return 1, ["ledger missing sealed_at_commit — cannot immutable-check"]

    try:
        sealed_full = resolve_rev(sealed)
    except subprocess.CalledProcessError:
        return 1, [f"sealed_at_commit {sealed!r} not resolvable in this repository"]

    if rev_arg:
        try:
            rev_full = resolve_rev(rev_arg)
        except subprocess.CalledProcessError:
            return 1, [f"--rev {rev_arg!r} not resolvable"]
        if rev_full != sealed_full:
            return 1, [
                f"--rev {rev_full} is not identical to sealed_at_commit {sealed_full}"
            ]
        pin = rev_full
    else:
        pin = sealed_full

    try:
        doc = build_ledger(pin)
    except SystemExit as e:
        return 2, [str(e)]

    payload = semantic_payload(doc)
    old_payload = semantic_payload(old)
    if old_payload == payload:
        return 0, [
            f"ASSERTION LEDGER CHECK PASS  total={doc['counts']['total']}  "
            f"sealed_at_commit={doc['sealed_at_commit'][:12]}"
        ]

    reasons: list[str] = []
    if old.get("sealed_at_commit") != doc.get("sealed_at_commit"):
        # After resolve, pins should match; difference means ledger pin was not
        # a full/equivalent hash of the extract rev.
        reasons.append(
            f"sealed_at_commit {old.get('sealed_at_commit')} != {doc.get('sealed_at_commit')}"
        )
    if old.get("counts", {}).get("total") != doc["counts"]["total"]:
        reasons.append(
            f"total {old.get('counts', {}).get('total')} != {doc['counts']['total']}"
        )
    old_ids = {e["case_id"]: e for e in old.get("entries", [])}
    new_ids = {e["case_id"]: e for e in doc["entries"]}
    missing = sorted(set(old_ids) - set(new_ids))
    added = sorted(set(new_ids) - set(old_ids))
    if missing:
        reasons.append(f"missing_ids:{len(missing)}:{missing[:5]}")
    if added:
        reasons.append(f"added_ids:{len(added)}:{added[:5]}")
    fp_drift = [
        cid
        for cid in sorted(set(old_ids) & set(new_ids))
        if old_ids[cid].get("content_fingerprint")
        != new_ids[cid].get("content_fingerprint")
    ]
    if fp_drift:
        reasons.append(f"fingerprint_drift:{len(fp_drift)}:{fp_drift[:5]}")
    if old.get("extractor_version") != doc.get("extractor_version"):
        reasons.append(
            f"extractor_version {old.get('extractor_version')} != {doc.get('extractor_version')}"
        )
    if not reasons:
        reasons.append("byte_payload_diff")
    msgs = ["ASSERTION LEDGER CHECK FAIL"] + [f"  ! {r}" for r in reasons]
    return 1, msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print ledger JSON to stdout")
    ap.add_argument("--write", metavar="PATH", help="write ledger JSON to PATH")
    ap.add_argument(
        "--check",
        metavar="PATH",
        help="re-extract at ledger sealed_at_commit and fail on semantic drift",
    )
    ap.add_argument(
        "--rev",
        metavar="REV",
        help="git revision for --write/--json (default HEAD); for --check must be identical to sealed_at_commit",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="print counts only",
    )
    args = ap.parse_args(argv)

    if not (args.json or args.write or args.check or args.summary):
        args.summary = True

    if args.check:
        path = Path(args.check)
        if not path.is_absolute():
            path = get_root() / path
        rc, msgs = check_ledger(path, args.rev)
        stream = sys.stdout if rc == 0 else sys.stderr
        for m in msgs:
            print(m, file=stream)
        return rc

    try:
        doc = build_ledger(args.rev)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"git error: {e.stderr or e}", file=sys.stderr)
        return 2

    payload = semantic_payload(doc)

    if args.write:
        path = Path(args.write)
        if not path.is_absolute():
            path = get_root() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        print(
            f"wrote {path}  total={doc['counts']['total']}  "
            f"by_kind={doc['counts']['by_kind']}  "
            f"sealed_at_commit={doc['sealed_at_commit'][:12]}"
        )

    if args.json:
        sys.stdout.write(payload)
    elif args.summary and not args.write:
        print(f"commit={doc['sealed_at_commit']}")
        print(f"total={doc['counts']['total']}")
        for k, v in doc["counts"]["by_kind"].items():
            print(f"  {k}: {v}")
        print(f"warnings={len(doc['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
