"""Rust source extraction via brace-aware line/token scanning (no rust-analyzer/SCIP)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from extract_cargo import crate_for_path
from graph_model import (
    Graph,
    complexity_of,
    detect_security,
    detect_side_effects,
    make_node,
)
from scanner import CodeScanner

# Match topology authority patterns for comparable function/public counts.
# Use [ \t] not \s so matches cannot span blanked comment lines (newlines).
RE_FN = re.compile(
    r"^([ \t]*)(?:pub(?:\([^)]*\))?[ \t]+)?(?:async[ \t]+)?(?:const[ \t]+)?(?:unsafe[ \t]+)?"
    r"(?:extern[ \t]+\"[^\"]+\"[ \t]+)?fn[ \t]+(\w+)",
    re.M,
)
RE_PUB_ITEM = re.compile(
    r"^[ \t]*pub(?:\(crate\))?[ \t]+(?:async[ \t]+)?(?:fn|struct|enum|trait|type|const|static)[ \t]+(\w+)",
    re.M,
)
RE_TYPE = re.compile(
    r"^([ \t]*)(?:pub(?:\([^)]*\))?[ \t]+)?(?:struct|enum|trait|type)[ \t]+(\w+)",
    re.M,
)
RE_IMPL = re.compile(
    r"^([ \t]*)(?:pub(?:\([^)]*\))?[ \t]+)?(?:unsafe[ \t]+)?impl(?:[ \t]*<[^;{\n]*>)?[ \t]+"
    r"(?:((?:[\w:]+(?:[ \t]*<[^;{\n]*>)?)(?:[ \t]*\+[ \t]*[\w:]+)*)[ \t]+for[ \t]+)?"
    r"([\w:]+)(?:[ \t]*<[^;{\n]*>)?[ \t]*(?:where[ \t]+[^{\n]*)?\{",
    re.M,
)
RE_USE = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?use[ \t]+([^;]+);",
    re.M,
)
RE_CALL = re.compile(r"\b([A-Z][A-Za-z0-9_]*)?::([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RE_CALL_SIMPLE = re.compile(r"(?<![:\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RE_TEST_ATTR = re.compile(r"#\[(?:\w+::)*test(?:\([^\)]*\))?\]")
RE_DERIVE = re.compile(r"#\[derive\s*\(([^)]*)\)\]")
RE_SERDE_TO = re.compile(r"\bserde_json::to_(?:string|vec|writer|value)\b")
RE_SERDE_FROM = re.compile(r"\bserde_json::from_(?:str|slice|reader|value)\b")
RE_EMIT = re.compile(
    r"\b(\w*(?:emit|publish|send_event|subscribe|on_event)\w*)\s*\(([^;]{0,200})\)",
    re.I,
)
RE_STRING = re.compile(r'"([^"\\]|\\.)*"')

# Control-flow keywords that are not functions
NOT_CALLS = frozenset({
    "if", "for", "while", "loop", "match", "return", "break", "continue",
    "move", "async", "await", "unsafe", "typeof", "sizeof", "box", "in",
    "where", "as", "ref", "mut", "let", "const", "static", "fn", "impl",
    "struct", "enum", "trait", "type", "use", "mod", "crate", "super",
    "self", "Self", "true", "false", "Some", "None", "Ok", "Err", "vec",
    "format", "println", "eprintln", "print", "write", "writeln", "panic",
    "assert", "assert_eq", "assert_ne", "debug_assert", "todo", "unimplemented",
    "unreachable", "cfg", "include", "include_str", "include_bytes", "env",
    "option_env", "concat", "stringify", "module_path", "file", "line",
    "column", "cfg_if", "matches", "matches_eq",
})


def _strip_generics(name: str) -> str:
    return name.split("<", 1)[0].split("::")[-1].strip()


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _mod_path_for_file(rel: str, crate_root: str) -> tuple[str, ...] | None:
    """Derive crate-local module path segments from a .rs file layout.

    ``src/lib.rs`` / ``src/main.rs`` → () (crate root)
    ``src/foo.rs`` / ``src/foo/mod.rs`` → ("foo",)
    ``src/foo/bar.rs`` → ("foo", "bar")
    """
    if not crate_root:
        return None
    prefix = crate_root if crate_root.endswith("/") else crate_root + "/"
    if not rel.startswith(prefix):
        return None
    rest = rel[len(prefix):]
    # Prefer src/ layout; also accept root-level for binary-only crates
    if rest.startswith("src/"):
        rest = rest[4:]
    elif "/" in rest and not rest.endswith(".rs"):
        return None
    if not rest.endswith(".rs"):
        return None
    if rest in ("lib.rs", "main.rs", "build.rs"):
        return ()
    if rest.endswith("/mod.rs"):
        rest = rest[: -len("/mod.rs")]
    elif rest.endswith(".rs"):
        rest = rest[: -len(".rs")]
    else:
        return None
    if not rest:
        return ()
    parts = tuple(p for p in rest.split("/") if p and p != "mod")
    return parts


def build_rust_module_index(
    rust_files: list[str],
    cargo_ctx: dict[str, Any],
) -> dict[str, Any]:
    """Map each crate's module tree from file layout.

    Returns {
      module_file: dict[crate_name, dict[tuple[str,...], file_rel]],
      file_module: dict[file_rel, tuple[crate_name, tuple[str,...]]],
    }
    """
    module_file: dict[str, dict[tuple[str, ...], str]] = defaultdict(dict)
    file_module: dict[str, tuple[str, tuple[str, ...]]] = {}
    for rel in rust_files:
        cname = crate_for_path(rel, cargo_ctx)
        if not cname:
            continue
        root = cargo_ctx["package_root"].get(cname) or ""
        mpath = _mod_path_for_file(rel, root)
        if mpath is None:
            continue
        # Prefer mod.rs over foo.rs when both map to the same module (rare)
        existing = module_file[cname].get(mpath)
        if existing is None or rel.endswith("/mod.rs") or rel.endswith("mod.rs"):
            module_file[cname][mpath] = rel
        file_module[rel] = (cname, mpath)
    return {"module_file": dict(module_file), "file_module": file_module}


def _split_use_segments(path_str: str) -> list[str]:
    """Split a use path on '::' ignoring content inside <...> and (...)."""
    segs: list[str] = []
    cur: list[str] = []
    depth_angle = 0
    depth_paren = 0
    i = 0
    s = path_str.strip()
    while i < len(s):
        if s[i] == "<":
            depth_angle += 1
            cur.append(s[i])
            i += 1
        elif s[i] == ">":
            depth_angle = max(0, depth_angle - 1)
            cur.append(s[i])
            i += 1
        elif s[i] == "(":
            depth_paren += 1
            cur.append(s[i])
            i += 1
        elif s[i] == ")":
            depth_paren = max(0, depth_paren - 1)
            cur.append(s[i])
            i += 1
        elif (
            s[i:i + 2] == "::"
            and depth_angle == 0
            and depth_paren == 0
        ):
            segs.append("".join(cur).strip())
            cur = []
            i += 2
        else:
            cur.append(s[i])
            i += 1
    if cur:
        segs.append("".join(cur).strip())
    return [x for x in segs if x]


def _expand_use_tree(path_str: str) -> list[str]:
    """Expand ``a::{b, c::d}`` / ``a::b as c`` into flat use paths (no aliases kept)."""
    s = path_str.strip().rstrip(";").strip()
    if not s:
        return []
    # Drop leading visibility already stripped by RE_USE
    # Handle `self::` prefixes etc. as-is

    def expand(prefix: str, body: str) -> list[str]:
        body = body.strip()
        if not body:
            return [prefix] if prefix else []
        # Find outermost brace group
        if "{" not in body:
            # strip `as Alias` / trailing rename
            part = re.sub(r"\s+as\s+\w+\s*$", "", body).strip()
            if not part or part == "*":
                return [prefix] if prefix else []
            if prefix:
                return [f"{prefix}::{part}"]
            return [part]
        # split prefix before first {
        brace = body.find("{")
        head = body[:brace].rstrip().rstrip(":").rstrip()
        # find matching }
        depth = 0
        end = -1
        for i, ch in enumerate(body[brace:], start=brace):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            part = re.sub(r"\s+as\s+\w+\s*$", "", body).strip()
            return [f"{prefix}::{part}" if prefix else part]
        inner = body[brace + 1:end]
        tail = body[end + 1:].strip()  # rarely used
        new_prefix = (
            f"{prefix}::{head}" if prefix and head else (prefix or head)
        )
        # split inner on commas at depth 0
        items: list[str] = []
        cur: list[str] = []
        d = 0
        for ch in inner:
            if ch == "{":
                d += 1
                cur.append(ch)
            elif ch == "}":
                d -= 1
                cur.append(ch)
            elif ch == "," and d == 0:
                items.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            items.append("".join(cur).strip())
        out: list[str] = []
        for item in items:
            if not item:
                continue
            if item == "self":
                if new_prefix:
                    out.append(new_prefix)
                continue
            if item == "*":
                if new_prefix:
                    out.append(new_prefix)
                continue
            out.extend(expand(new_prefix, item))
        if tail:
            # ignore unexpected tail
            pass
        return out

    return expand("", s)


def _lookup_crate_id(name: str, cargo_ctx: dict[str, Any]) -> str | None:
    return (
        cargo_ctx["name_to_id"].get(name)
        or cargo_ctx["name_to_id"].get(name.replace("_", "-"))
        or cargo_ctx["name_to_id"].get(name.replace("-", "_"))
    )


def _resolve_use_path(
    path_str: str,
    *,
    file_rel: str,
    crate_name: str | None,
    cargo_ctx: dict[str, Any],
    mod_index: dict[str, Any],
    g: Graph,
    indexes: dict[str, Any],
) -> tuple[str | None, float]:
    """Resolve one expanded use path to an in-repo node id + confidence.

    Returns (node_id_or_None, confidence). Drop unresolved paths (no guessing).
    """
    segs = _split_use_segments(path_str)
    if not segs:
        return None, 0.0
    # strip generics from each segment for resolution
    segs = [_strip_generics(s).strip().strip('"') for s in segs]
    segs = [s for s in segs if s]
    if not segs:
        return None, 0.0

    file_module = mod_index.get("file_module") or {}
    module_file = mod_index.get("module_file") or {}

    cur_info = file_module.get(file_rel)
    cur_crate = crate_name or (cur_info[0] if cur_info else None)
    cur_mod: tuple[str, ...] = cur_info[1] if cur_info else ()

    target_crate = cur_crate
    mod_segs: list[str] = []
    first = segs[0]

    if first == "crate":
        if not cur_crate:
            return None, 0.0
        target_crate = cur_crate
        mod_segs = list(segs[1:])
    elif first == "super":
        if not cur_crate:
            return None, 0.0
        # count leading supers
        up = 0
        while up < len(segs) and segs[up] == "super":
            up += 1
        if up > len(cur_mod):
            return None, 0.0
        base = cur_mod[: len(cur_mod) - up]
        mod_segs = list(base) + list(segs[up:])
        target_crate = cur_crate
    elif first == "self":
        if not cur_crate:
            return None, 0.0
        mod_segs = list(cur_mod) + list(segs[1:])
        target_crate = cur_crate
    else:
        # external or absolute crate path (workspace member)
        cid = _lookup_crate_id(first, cargo_ctx)
        if cid:
            # map to the canonical package name used by module_file keys
            pkg = None
            for n in cargo_ctx.get("package_names") or []:
                if n == first or n.replace("-", "_") == first.replace("-", "_"):
                    pkg = n
                    break
            if pkg is None:
                # cid is "crate:<name>"
                pkg = cid.removeprefix("crate:")
            target_crate = pkg
            mod_segs = list(segs[1:])
            if not mod_segs:
                # bare `use other_crate;` → crate node, first-segment only
                return cid, 0.6
        else:
            # could be a prelude / extern crate we don't know — drop
            return None, 0.0

    if not target_crate:
        return None, 0.0

    crate_mods: dict[tuple[str, ...], str] = module_file.get(target_crate) or {}
    # Also try underscore/hyphen crate name variants
    if not crate_mods:
        for alt in (
            target_crate.replace("-", "_"),
            target_crate.replace("_", "-"),
        ):
            if alt in module_file:
                crate_mods = module_file[alt]
                target_crate = alt
                break

    # Walk longest module prefix that exists as a file; remaining may be type/fn/item
    best_file: str | None = None
    best_len = -1
    item_tail: list[str] = []
    # try full path as module, then peel last segments as items
    for peel in range(len(mod_segs) + 1):
        keep = len(mod_segs) - peel
        mpath = tuple(mod_segs[:keep])
        if mpath in crate_mods:
            best_file = crate_mods[mpath]
            best_len = keep
            item_tail = mod_segs[keep:]
            break
        # also try if last segment is `self`
    if best_file is None:
        # first-segment crate match only (no module path resolved)
        cid = _lookup_crate_id(target_crate, cargo_ctx)
        if cid and not mod_segs:
            return cid, 0.6
        if cid and first not in ("crate", "super", "self"):
            # external workspace crate with unresolved subpath — crate-level edge
            return cid, 0.6
        return None, 0.0

    conf = 0.9  # resolved through module tree
    # Try to resolve item_tail to a type or function in that file
    if item_tail:
        item = item_tail[0]
        # type in this file
        tid = f"type:{best_file}#{item}"
        if tid in g.nodes:
            return tid, conf
        # function free or Type::method — try free fn and any qname ending
        fid = f"fn:{best_file}#{item}"
        if fid in g.nodes:
            return fid, conf
        # methods / qualified names in this file
        cands = [
            nid for nid in indexes.get("fns_by_name", {}).get(item, [])
            if nid.startswith(f"fn:{best_file}#")
        ]
        if len(cands) == 1:
            return cands[0], conf
        type_cands = [
            nid for nid in indexes.get("types_by_name", {}).get(item, [])
            if nid.startswith(f"type:{best_file}#")
        ]
        if len(type_cands) == 1:
            return type_cands[0], conf
        # unresolved item but module file known — point at file
        return f"file:{best_file}", conf

    return f"file:{best_file}", conf


def emit_rust_imports(
    repo: Path,
    rel: str,
    text: str,
    code: str,
    g: Graph,
    cargo_ctx: dict[str, Any],
    indexes: dict[str, Any],
    mod_index: dict[str, Any],
) -> None:
    """Emit file-level ``imports`` edges from ``use`` statements."""
    crate_name = crate_for_path(rel, cargo_ctx)
    file_id = f"file:{rel}"
    for m in RE_USE.finditer(code):
        path_str = m.group(1).strip()
        for expanded in _expand_use_tree(path_str):
            dst, conf = _resolve_use_path(
                expanded,
                file_rel=rel,
                crate_name=crate_name,
                cargo_ctx=cargo_ctx,
                mod_index=mod_index,
                g=g,
                indexes=indexes,
            )
            if not dst or dst == file_id:
                continue
            if dst not in g.nodes and not dst.startswith("crate:"):
                continue
            if dst not in g.nodes:
                continue
            g.add_edge(
                file_id, "imports", dst,
                evidence="regex", confidence=conf, weight=1.0,
            )


def extract_rust_file(
    repo: Path,
    rel: str,
    g: Graph,
    cargo_ctx: dict[str, Any],
    indexes: dict[str, Any],
    mod_index: dict[str, Any] | None = None,
) -> None:
    """Extract nodes/edges from one .rs file into g; update indexes for call resolution."""
    path = repo / rel
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    if not text:
        return

    sc = CodeScanner(text, lang="rust")
    code = sc.slice_code(0, len(text))
    crate_name = crate_for_path(rel, cargo_ctx)
    crate_id = cargo_ctx["name_to_id"].get(crate_name) if crate_name else None
    file_id = f"file:{rel}"
    loc = len(text.split("\n")) - (0 if text.endswith("\n") else 0)
    # physical lines matching hawking_loc: split on \n minus 1 if no trailing? loc tool uses
    # len(bytes.split(b"\n")) - 1
    loc = len(text.encode("utf-8", errors="ignore").split(b"\n")) - 1

    # File node is created by the orchestrator; still safe to ensure
    if file_id not in g.nodes:
        g.add_node(make_node("file", file_id, Path(rel).name, path=rel, lang="rust", loc=loc))

    # use imports are emitted in a second pass once the module tree + symbols exist
    # (see extract_all_rust); stash blanked code for that pass.
    indexes.setdefault("rust_file_code", {})[rel] = (text, code)

    # --- types ---
    type_nodes: dict[str, str] = {}  # short name -> id
    for m in RE_TYPE.finditer(code):
        indent, tname = m.group(1), m.group(2)
        # skip if inside string was already blanked
        if not tname:
            continue
        # determine kind keyword
        line_start = code.rfind("\n", 0, m.start()) + 1
        line = code[line_start: code.find("\n", m.start())]
        is_pub = bool(re.match(r"[ \t]*pub\b", line))
        start_line = _line_of(text, m.start())
        # RE_TYPE ends at name; find brace after (same line / nearby)
        brace = sc.find_next_brace(m.end())
        end_line = start_line
        end_off = m.end()
        if brace is not None and brace - m.end() < 300:
            between = code[m.end():brace]
            if all(c in " \t\n<>:_,'\"()[]&*?!=," or c.isalnum() for c in between[:120]):
                matched = sc.match_braces(brace)
                if matched:
                    end_off = matched
                    end_line = _line_of(text, matched - 1)
        if "type " in line and "=" in line and "{" not in line:
            semi = code.find(";", m.end())
            if semi != -1:
                end_off = semi + 1
                end_line = _line_of(text, semi)

        tid = f"type:{rel}#{tname}"
        body = text[m.start():end_off]
        se = detect_side_effects(body)
        sec = detect_security(body)
        g.add_node(make_node(
            "type", tid, tname, path=rel, lang="rust",
            span=[start_line, end_line],
            loc=max(1, end_line - start_line + 1),
            public=is_pub,
            complexity=complexity_of(body),
            side_effects=se,
            security_sensitive=sec,
        ))
        g.ensure_contains(file_id, tid, evidence="regex")
        type_nodes[tname] = tid
        indexes["types_by_name"][tname].append(tid)
        if crate_name:
            indexes["types_by_crate"][crate_name][tname] = tid
        indexes["type_id_by_qual"][f"{crate_name}::{tname}" if crate_name else tname] = tid

        # derives Serialize / Deserialize
        # look back a few lines for #[derive(...)]
        pre = code[max(0, m.start() - 400):m.start()]
        for dm in RE_DERIVE.finditer(pre):
            # only the last derive block before the type
            pass
        last_derive = None
        for dm in RE_DERIVE.finditer(pre):
            last_derive = dm
        if last_derive:
            dlist = last_derive.group(1)
            if re.search(r"\bSerialize\b", dlist):
                indexes["serializable"].add(tid)
            if re.search(r"\bDeserialize\b", dlist):
                indexes["deserializable"].add(tid)

    # --- impl blocks: implements edges + track impl type for methods ---
    impl_regions: list[tuple[int, int, str | None, str | None]] = []
    # (start_off, end_off, trait_name_or_None, type_name)
    for m in RE_IMPL.finditer(code):
        trait_part = m.group(2)
        type_part = m.group(3)
        tname = _strip_generics(type_part or "")
        trait_name = _strip_generics(trait_part) if trait_part else None
        brace = sc.find_next_brace(m.start())
        if brace is None:
            continue
        end = sc.match_braces(brace)
        if end is None:
            continue
        impl_regions.append((m.start(), end, trait_name, tname))
        if trait_name and tname:
            # implements: type implements trait
            type_id = type_nodes.get(tname) or (
                indexes["types_by_crate"].get(crate_name or "", {}).get(tname)
            )
            trait_id = type_nodes.get(trait_name) or (
                indexes["types_by_crate"].get(crate_name or "", {}).get(trait_name)
            )
            # may resolve later; store pending
            if type_id and trait_id:
                g.add_edge(
                    type_id, "implements", trait_id,
                    evidence="regex", confidence=0.85,
                )
            else:
                indexes["pending_impls"].append(
                    (crate_name, tname, trait_name, rel)
                )

    def impl_for_offset(off: int) -> tuple[str | None, str | None]:
        for s, e, trait, ty in impl_regions:
            if s <= off < e:
                return trait, ty
        return None, None

    # Precompute test attribute line numbers
    test_lines: set[int] = set()
    for m in RE_TEST_ATTR.finditer(code):
        test_lines.add(_line_of(text, m.start()))

    # --- functions ---
    for m in RE_FN.finditer(code):
        fname = m.group(2)
        line_start = code.rfind("\n", 0, m.start()) + 1
        line = code[line_start: code.find("\n", m.start()) if code.find("\n", m.start()) != -1 else len(code)]
        is_pub = bool(re.match(r"[ \t]*pub\b", line))
        # Also treat pub captured by RE_FN group region
        if not is_pub and re.search(r"[ \t]pub(?:\(|[ \t])", m.group(0)):
            is_pub = True
        start_line = _line_of(text, m.start())

        # check attributes on previous non-empty lines for #[test]
        is_test = False
        for prev in range(start_line - 1, max(0, start_line - 8), -1):
            if prev in test_lines:
                is_test = True
                break
        pre_raw = text[max(0, m.start() - 200):m.start()]
        if RE_TEST_ATTR.search(pre_raw):
            is_test = True

        trait, impl_ty = impl_for_offset(m.start())
        if impl_ty:
            qname = f"{impl_ty}::{fname}"
        else:
            qname = fname

        brace = sc.find_next_brace(m.end())
        end_off = m.end()
        end_line = start_line
        body = ""
        if brace is not None and brace - m.end() < 800:
            between = code[m.end():brace]
            if (
                all(c in " \t\n" or c.isalnum() or c in "<>:_,'\"()[]&*?!=,+'" for c in between[:300])
                or between.strip() == ""
                or "where" in between
                or "->" in between
            ):
                matched = sc.match_braces(brace)
                if matched:
                    end_off = matched
                    end_line = _line_of(text, matched - 1)
                    body = text[brace:matched]

        fid = f"fn:{rel}#{qname}"
        se = detect_side_effects(body or line)
        sec = detect_security(body or line) or ("unsafe" in line)
        g.add_node(make_node(
            "function", fid, qname, path=rel, lang="rust",
            span=[start_line, end_line],
            loc=max(1, end_line - start_line + 1),
            public=is_pub,
            test=is_test or is_test_path_name(rel, fname),
            complexity=complexity_of(body),
            side_effects=se,
            security_sensitive=sec,
        ))
        g.ensure_contains(file_id, fid, evidence="regex")

        indexes["fns_by_name"][fname].append(fid)
        indexes["fns_by_qual"][qname].append(fid)
        if crate_name:
            indexes["fns_by_crate"][crate_name][fname].append(fid)
            indexes["fns_by_crate"][crate_name][qname].append(fid)
        indexes["fn_meta"][fid] = {
            "crate": crate_name,
            "file": rel,
            "name": fname,
            "qname": qname,
            "body": body,
            "is_test": is_test,
        }

        if is_test:
            tid = f"test:{rel}#{qname}"
            g.add_node(make_node(
                "test", tid, qname, path=rel, lang="rust",
                span=[start_line, end_line],
                loc=max(1, end_line - start_line + 1),
                public=False,
                test=True,
                complexity=complexity_of(body),
                side_effects=se,
                security_sensitive=sec,
            ))
            g.ensure_contains(file_id, tid, evidence="regex")
            # link test node to function
            g.add_edge(tid, "tests", fid, evidence="test", confidence=1.0, weight=1.0)
            indexes["test_nodes"][tid] = fid

        # serde call sites inside body
        if body:
            if RE_SERDE_TO.search(body):
                indexes["serde_to_fns"].append(fid)
            if RE_SERDE_FROM.search(body):
                indexes["serde_from_fns"].append(fid)
            # emit/consume heuristics
            for em in RE_EMIT.finditer(body):
                call_name = em.group(1).lower()
                args = em.group(2)
                # string or ident args
                for sm in RE_STRING.finditer(args):
                    ev = sm.group(0).strip('"')
                    if not ev or len(ev) > 80:
                        continue
                    eid = f"event:{ev}"
                    if eid not in g.nodes:
                        g.add_node(make_node(
                            "event", eid, ev, path=rel, lang="rust", public=True,
                        ))
                    etype = "emits" if any(x in call_name for x in ("emit", "publish", "send")) else "consumes"
                    g.add_edge(fid, etype, eid, evidence="regex", confidence=0.5)


def is_test_path_name(rel: str, fname: str) -> bool:
    return (
        "/tests/" in rel
        or rel.startswith("workspace/quality/tests/")
        or rel.endswith("_test.rs")
        or fname.startswith("test_")
        or fname.endswith("_test")
    )


def resolve_rust_calls(g: Graph, indexes: dict[str, Any], cargo_ctx: dict[str, Any]) -> None:
    """Second pass: calls edges with crate-local then unique-global resolution."""
    for fid, meta in indexes["fn_meta"].items():
        body = meta.get("body") or ""
        if not body:
            continue
        crate = meta.get("crate")
        sc = CodeScanner(body, lang="rust")
        code = sc.slice_code(0, len(body))
        seen: dict[str, int] = defaultdict(int)

        # Type::method(
        for m in RE_CALL.finditer(code):
            ty, name = m.group(1), m.group(2)
            if name in NOT_CALLS:
                continue
            key = f"{ty}::{name}" if ty else name
            seen[key] += 1

        # simple ident(
        for m in RE_CALL_SIMPLE.finditer(code):
            name = m.group(1)
            if name in NOT_CALLS:
                continue
            # skip if already counted as part of Type::name
            seen[name] += 1

        for key, count in seen.items():
            targets = _resolve_fn(key, crate, indexes)
            if not targets:
                continue
            if len(targets) == 1:
                conf = 0.8
                dst = targets[0]
                g.add_edge(
                    fid, "calls", dst,
                    count=count, evidence="regex", confidence=conf, weight=float(count),
                )
            else:
                # ambiguous: still record each with low confidence (schema says 0.4)
                conf = 0.4
                for dst in targets[:5]:  # cap fanout
                    g.add_edge(
                        fid, "calls", dst,
                        count=count, evidence="regex", confidence=conf, weight=float(count),
                    )

        # serializes / deserializes: only types named in the same body
        crate_types = indexes["types_by_crate"].get(crate or "", {})
        if RE_SERDE_TO.search(body):
            for tname, tid in crate_types.items():
                if tid in indexes.get("serializable", set()) and re.search(
                    rf"\b{re.escape(tname)}\b", body
                ):
                    g.add_edge(fid, "serializes", tid, evidence="regex", confidence=0.7)
        if RE_SERDE_FROM.search(body):
            for tname, tid in crate_types.items():
                if tid in indexes.get("deserializable", set()) and re.search(
                    rf"\b{re.escape(tname)}\b", body
                ):
                    g.add_edge(fid, "deserializes", tid, evidence="regex", confidence=0.7)
        # Derive-level: type with Serialize is serializable via its inherent methods — soft
        # edge type->self skipped; instead file-level evidence already in serializable set.

    # Derive-based serializes edges from type to a synthetic note: use type self-edge skip
    # Instead: from file containing type, soft edges not required; skip

    # Resolve pending impls
    for crate_name, tname, trait_name, rel in indexes.get("pending_impls", []):
        type_id = indexes["types_by_crate"].get(crate_name or "", {}).get(tname)
        trait_id = indexes["types_by_crate"].get(crate_name or "", {}).get(trait_name)
        if not trait_id:
            # global unique trait
            cands = indexes["types_by_name"].get(trait_name, [])
            if len(cands) == 1:
                trait_id = cands[0]
        if not type_id:
            cands = indexes["types_by_name"].get(tname, [])
            if len(cands) == 1:
                type_id = cands[0]
        if type_id and trait_id:
            g.add_edge(type_id, "implements", trait_id, evidence="regex", confidence=0.85)


def _resolve_fn(key: str, crate: str | None, indexes: dict[str, Any]) -> list[str]:
    """Resolve call key to function node ids: same crate first, then unique global."""
    local = indexes["fns_by_crate"].get(crate or "", {})
    if key in local and local[key]:
        return list(dict.fromkeys(local[key]))  # unique preserve order
    # try short name if qualified
    short = key.split("::")[-1]
    if short != key and short in local and local[short]:
        return list(dict.fromkeys(local[short]))
    # global unique
    if key in indexes["fns_by_qual"] and len(set(indexes["fns_by_qual"][key])) == 1:
        return [indexes["fns_by_qual"][key][0]]
    if key in indexes["fns_by_name"]:
        uniq = list(dict.fromkeys(indexes["fns_by_name"][key]))
        if len(uniq) == 1:
            return uniq
        if len(uniq) > 1:
            return uniq  # ambiguous
    if short != key and short in indexes["fns_by_name"]:
        uniq = list(dict.fromkeys(indexes["fns_by_name"][short]))
        if len(uniq) == 1:
            return uniq
        if len(uniq) > 1:
            return uniq
    return []


def empty_rust_indexes() -> dict[str, Any]:
    return {
        "fns_by_name": defaultdict(list),
        "fns_by_qual": defaultdict(list),
        "fns_by_crate": defaultdict(lambda: defaultdict(list)),
        "types_by_name": defaultdict(list),
        "types_by_crate": defaultdict(dict),
        "type_id_by_qual": {},
        "fn_meta": {},
        "test_nodes": {},
        "serializable": set(),
        "deserializable": set(),
        "serde_to_fns": [],
        "serde_from_fns": [],
        "pending_impls": [],
    }


def extract_all_rust(
    repo: Path,
    files: list[str],
    g: Graph,
    cargo_ctx: dict[str, Any],
) -> dict[str, Any]:
    indexes = empty_rust_indexes()
    mod_index = build_rust_module_index(files, cargo_ctx)
    indexes["rust_module_index"] = mod_index
    for rel in files:
        extract_rust_file(repo, rel, g, cargo_ctx, indexes, mod_index=mod_index)
    # File-level imports after symbols exist so type/fn targets resolve.
    for rel, (text, code) in indexes.get("rust_file_code", {}).items():
        emit_rust_imports(
            repo, rel, text, code, g, cargo_ctx, indexes, mod_index,
        )
    indexes.pop("rust_file_code", None)
    resolve_rust_calls(g, indexes, cargo_ctx)
    # serializes edges for Serialize derives (type -> mark via edge from crate file)
    for tid in indexes["serializable"]:
        # self-describing: edge from type's containing file function not required;
        # add constructs? skip — schema has serializes as edge type for call/type relation
        pass
    return indexes
