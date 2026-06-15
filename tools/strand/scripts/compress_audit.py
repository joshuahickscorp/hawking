#!/usr/bin/env python3
"""
compress_audit.py — structural compression opportunities in the STRAND codebase.
Finds: trivial wrappers, re-export-only files, single-call functions,
thin modules (candidates to merge), duplicate struct shapes, dead pub items.
"""
import os, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
CRATES = ROOT / "crates"

# ── helpers ──────────────────────────────────────────────────────────────────

def rs_files():
    for p in CRATES.rglob("*.rs"):
        if "target" not in p.parts:
            yield p

def read(p):
    return p.read_text(errors="replace")

def strip_comments(src):
    """Very rough comment stripper for analysis purposes."""
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)

def code_lines(src):
    """Count non-blank, non-comment lines."""
    return sum(1 for l in src.splitlines()
               if l.strip() and not l.strip().startswith("//"))

# ── 1. trivial wrapper functions ─────────────────────────────────────────────
# Pattern: fn foo(...) -> T { bar(...) }  (single expression body)

WRAPPER_RE = re.compile(
    r'(?:pub(?:\([^)]*\))?\s+)?(?:fn\s+(\w+)\s*'
    r'(?:<[^>]*>)?\s*\([^)]*\)[^{]*\{)\s*\n\s*(\w[\w:!()[\], &<>*_\'"\.]*)\s*\n\s*\}',
    re.MULTILINE
)

SIMPLE_BODY_RE = re.compile(
    r'pub(?:\([^)]*\))?\s+fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)[^{]*\{([^}]*)\}',
    re.DOTALL
)

def find_trivial_wrappers(path, src):
    results = []
    # Find functions whose entire body is one statement
    fn_re = re.compile(
        r'(pub(?:\([^)]*\))?\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)[^{]*\{([^{}]*)\}',
        re.DOTALL
    )
    for m in fn_re.finditer(src):
        pub, name, body = m.group(1), m.group(2), m.group(3)
        body_stripped = strip_comments(body).strip()
        lines = [l for l in body_stripped.splitlines() if l.strip()]
        if len(lines) == 1:
            stmt = lines[0].strip()
            # Single-statement body that just calls another function
            if re.match(r'[\w:]+\s*[(!]', stmt) or stmt.startswith("self."):
                results.append((name, stmt[:80]))
    return results

# ── 2. re-export / pass-through files ────────────────────────────────────────

def is_reexport_only(src):
    """File that only re-exports (pub use) and has no own definitions."""
    s = strip_comments(src)
    defs = re.findall(r'\b(fn|struct|enum|impl|trait|mod|const|static|type)\b', s)
    pub_uses = re.findall(r'pub use ', s)
    uses = re.findall(r'^use ', s, re.MULTILINE)
    return len(defs) == 0 and (len(pub_uses) > 0 or len(uses) > 2)

# ── 3. thin modules (few items) ───────────────────────────────────────────────

def count_items(src):
    s = strip_comments(src)
    fns = len(re.findall(r'\bfn\s+\w+', s))
    structs = len(re.findall(r'\bstruct\s+\w+', s))
    enums = len(re.findall(r'\benum\s+\w+', s))
    impls = len(re.findall(r'\bimpl\b', s))
    traits = len(re.findall(r'\btrait\s+\w+', s))
    return {"fns": fns, "structs": structs, "enums": enums,
            "impls": impls, "traits": traits,
            "total": fns + structs + enums + traits}

# ── 4. single-call functions (inlining candidates) ────────────────────────────

def find_single_call_functions(path, src, all_srcs):
    """Functions that are only called once across the whole codebase."""
    s = strip_comments(src)
    # Find all pub fn names in this file
    fn_names = re.findall(r'pub\s+fn\s+(\w+)', s)
    results = []
    for name in fn_names:
        if name in ("new", "default", "main", "from", "into", "clone",
                    "fmt", "drop", "next", "poll"):
            continue
        # Count calls across all sources
        total_calls = sum(
            len(re.findall(r'\b' + re.escape(name) + r'\s*[(<]', src2))
            for src2 in all_srcs
        )
        # subtract the definition itself
        defs = len(re.findall(r'\bfn\s+' + re.escape(name) + r'\b', src))
        calls = total_calls - defs
        if calls == 1:
            results.append(name)
    return results

# ── 5. duplicate / near-identical struct shapes ───────────────────────────────

def extract_structs(src):
    """Extract struct names and their field names."""
    s = strip_comments(src)
    results = {}
    for m in re.finditer(r'struct\s+(\w+)\s*(?:<[^>]*>)?\s*\{([^}]*)\}', s, re.DOTALL):
        name = m.group(1)
        fields = re.findall(r'(\w+)\s*:', m.group(2))
        if fields:
            results[name] = tuple(sorted(fields))
    return results

# ── 6. pub fn delegation chains ───────────────────────────────────────────────
# lib.rs that just delegates to a sub-module

def find_delegation_wrappers(src):
    """pub fn X(...) that just calls self.Y.X(...) or module::X(...)"""
    results = []
    pat = re.compile(
        r'pub\s+fn\s+(\w+)\s*\(([^)]*)\)[^{]*\{\s*([^}]{0,120})\s*\}',
        re.DOTALL
    )
    for m in pat.finditer(src):
        name, args, body = m.group(1), m.group(2), m.group(3).strip()
        body_lines = [l.strip() for l in body.splitlines() if l.strip()
                      and not l.strip().startswith("//")]
        if len(body_lines) == 1:
            b = body_lines[0]
            # Delegates to self.field.method or crate::mod::fn
            if (re.search(r'self\.\w+\.' + re.escape(name), b) or
                re.search(r'\w+::' + re.escape(name) + r'\s*\(', b)):
                results.append((name, b[:80]))
    return results

# ── 7. crate-level thinness ───────────────────────────────────────────────────

def crate_stats():
    crates = {}
    for crate_dir in CRATES.iterdir():
        if not crate_dir.is_dir():
            continue
        src_dir = crate_dir / "src"
        if not src_dir.exists():
            continue
        rs = list(src_dir.rglob("*.rs"))
        total_lines = sum(p.read_text(errors="replace").count("\n") for p in rs)
        code = sum(code_lines(p.read_text(errors="replace")) for p in rs)
        items = sum(count_items(p.read_text(errors="replace"))["total"] for p in rs)
        crates[crate_dir.name] = {
            "files": len(rs),
            "total_lines": total_lines,
            "code_lines": code,
            "items": items,
        }
    return crates

# ── 8. large inline test modules ─────────────────────────────────────────────

def find_large_test_mods(path, src):
    """#[cfg(test)] mod tests { ... } that are large."""
    m = re.search(r'#\[cfg\(test\)\]\s*(?:pub\s+)?mod\s+\w+\s*\{', src)
    if m:
        start = m.end()
        depth = 1
        pos = start
        while pos < len(src) and depth > 0:
            if src[pos] == '{':
                depth += 1
            elif src[pos] == '}':
                depth -= 1
            pos += 1
        test_src = src[start:pos]
        return test_src.count("\n")
    return 0

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("STRAND STRUCTURAL COMPRESSION AUDIT")
    print("=" * 72)

    all_files = list(rs_files())
    all_srcs = [read(p) for p in all_files]
    src_map = dict(zip(all_files, all_srcs))

    # ── Crate-level stats ────────────────────────────────────────────────────
    print("\n## CRATE SIZES")
    print(f"{'crate':<30} {'files':>5} {'total':>7} {'code':>6} {'items':>6}")
    print("-" * 60)
    stats = crate_stats()
    total_code = 0
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["total_lines"]):
        print(f"{name:<30} {s['files']:>5} {s['total_lines']:>7} "
              f"{s['code_lines']:>6} {s['items']:>6}")
        total_code += s["code_lines"]
    print(f"{'TOTAL':<30} {'':>5} {'':>7} {total_code:>6}")

    # ── Re-export-only files ─────────────────────────────────────────────────
    print("\n## RE-EXPORT / PASS-THROUGH FILES (merge into parent)")
    for path, src in src_map.items():
        if is_reexport_only(src):
            cl = code_lines(src)
            if cl < 30:
                rel = path.relative_to(ROOT)
                print(f"  {rel}  ({cl} code lines)")

    # ── Thin modules (candidates to merge up) ────────────────────────────────
    print("\n## THIN MODULES (≤8 items, ≤80 code lines — merge into parent?)")
    for path, src in sorted(src_map.items(), key=lambda x: code_lines(x[1])):
        if path.name in ("lib.rs", "mod.rs", "main.rs"):
            continue
        items = count_items(src)
        cl = code_lines(src)
        if items["total"] <= 8 and cl <= 120 and cl > 0:
            rel = path.relative_to(ROOT)
            print(f"  {rel}  ({cl} code lines, {items['total']} items: "
                  f"fns={items['fns']} structs={items['structs']} "
                  f"enums={items['enums']})")

    # ── Trivial delegation wrappers ──────────────────────────────────────────
    print("\n## TRIVIAL DELEGATION WRAPPERS (single-line body, inlining candidates)")
    wrapper_count = 0
    for path, src in src_map.items():
        wrappers = find_delegation_wrappers(src)
        if wrappers:
            rel = path.relative_to(ROOT)
            for name, body in wrappers[:5]:
                print(f"  {rel}: fn {name}() → {body}")
                wrapper_count += 1
    print(f"  Total: {wrapper_count} delegation wrappers")

    # ── Struct shape duplicates ───────────────────────────────────────────────
    print("\n## STRUCT FIELD DUPLICATES (same field names → possible consolidation)")
    all_structs = {}  # fields_tuple → [(file, name)]
    for path, src in src_map.items():
        for name, fields in extract_structs(src).items():
            if len(fields) >= 3:
                if fields not in all_structs:
                    all_structs[fields] = []
                all_structs[fields].append((path.relative_to(ROOT), name))
    for fields, locations in all_structs.items():
        if len(locations) >= 2:
            print(f"  fields={fields}")
            for loc, name in locations:
                print(f"    {loc}: {name}")

    # ── Large inline test modules ─────────────────────────────────────────────
    print("\n## LARGE INLINE TEST MODULES (consider moving to tests/ dir)")
    for path, src in src_map.items():
        n = find_large_test_mods(path, src)
        if n > 100:
            rel = path.relative_to(ROOT)
            print(f"  {rel}: {n} lines of inline tests")

    # ── Files with many single-line functions ─────────────────────────────────
    print("\n## FILES WITH HIGH TRIVIAL-FN RATIO")
    for path, src in src_map.items():
        wrappers = find_trivial_wrappers(path, src)
        cl = code_lines(src)
        if cl > 0 and len(wrappers) * 3 > cl * 0.15 and len(wrappers) >= 3:
            rel = path.relative_to(ROOT)
            print(f"  {rel}: {len(wrappers)} trivial fns / {cl} code lines "
                  f"({100*len(wrappers)*3//cl}% trivial)")
            for name, body in wrappers[:4]:
                print(f"    fn {name}() {{ {body} }}")

    # ── pub fn that appear only once in the codebase ─────────────────────────
    print("\n## PUB FNS CALLED EXACTLY ONCE (inline or make private)")
    concat = "\n".join(all_srcs)
    once_total = 0
    for path, src in src_map.items():
        s = strip_comments(src)
        fns = re.findall(r'pub\s+fn\s+(\w+)', s)
        once = []
        for name in set(fns):
            if name in ("new", "default", "main", "from", "into", "clone",
                        "fmt", "drop", "next", "update", "predict", "encode",
                        "decode", "compress", "decompress", "check"):
                continue
            # count all occurrences across full codebase
            count = len(re.findall(r'\b' + re.escape(name) + r'\b', concat))
            # subtract definition (fn name, impl blocks, test calls, doc refs)
            if count <= 3:
                once.append((name, count))
        if once:
            rel = path.relative_to(ROOT)
            print(f"  {rel}: {[f'{n}(×{c})' for n,c in once[:6]]}")
            once_total += len(once)
    print(f"  Total: {once_total} rarely-used pub fns")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n## SUMMARY")
    total_lines = sum(src.count("\n") for src in all_srcs)
    total_code_lines = sum(code_lines(src) for src in all_srcs)
    print(f"  Total lines: {total_lines}")
    print(f"  Total code lines: {total_code_lines}")
    print(f"  Files: {len(all_files)}")

if __name__ == "__main__":
    main()
