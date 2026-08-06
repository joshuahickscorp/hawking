#!/usr/bin/env python3.12
"""Hawking multi-language semantic graph extractor (G1 instrument).

Obeys workspace/campaign/governance/control/catalog/manifests/
SEMANTIC_GRAPH_SCHEMA.json exactly — no added/renamed/dropped node types,
edge types, attribute names, or id formats.

Usage:
    python3.12 tools/graph/hawking_graph.py --emit all
    python3.12 tools/graph/hawking_graph.py --emit jsonl
    python3.12 tools/graph/hawking_graph.py --verify

Output defaults to workspace/ops/build/graph/ (gitignored): every artifact here is
deterministic and rebuilt by --emit all in about 40s.

Dependencies: stdlib + networkx (optional for future G2; not required to emit).
No tree_sitter, no rust-analyzer, no SCIP, no new pip/cargo deps.

Side-effect allowlist (node attrs.side_effects):
    fs    — File/std::fs/open/pathlib/...
    net   — reqwest/hyper/socket/urllib/...
    proc  — Command/subprocess/...
    env   — env::var/os.environ/...
    clock — Instant/SystemTime/datetime/...
    rand  — rand::/random/...
    gpu   — metal/wgpu/cuda/dispatch...
    none  — default when nothing matches
See graph_model.SIDE_EFFECT_PATTERNS.

Runtime hotness is STATIC only (workspace/docs/reference/BENCHMARKS.md,
workspace/docs/reference/kernels.md,
hawking-bench, Metal kernels, 2-hop from decode/dispatch) — not traced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_cargo import extract_cargo, crate_for_path
from extract_other import (
    extract_all_typescript,
    extract_git_cochange,
    extract_metal_file,
    extract_registries,
    link_metal_from_rust,
    mark_runtime_hot,
    mark_test_coverage,
)
from extract_python import extract_all_python
from extract_rust import extract_all_rust
from graph_model import (
    Graph,
    classify_path,
    is_test_path,
    lang_for,
    make_node,
    subsystem_for,
)

REPO = Path(__file__).resolve().parents[2]

# Same language extensions as tools/loc/hawking_loc.py
LOC_LANGS = {
    ".rs": "rust",
    ".py": "python",
    ".md": "markdown",
    ".sh": "shell",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".metal": "metal",
    ".lean": "lean",
}

CODE_EXT = {".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".metal", ".sh", ".lean"}


def git_ls_files(repo: Path) -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(repo), "ls-files"],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.splitlines()


def physical_loc(repo: Path, rel: str) -> int:
    f = repo / rel
    try:
        return len(f.read_bytes().split(b"\n")) - 1 if f.exists() else 0
    except OSError:
        return 0


def ensure_dirs_and_files(repo: Path, g: Graph, files: list[str]) -> set[str]:
    """Create repository, directory, and file nodes; return active-ish file set."""
    g.add_node(make_node(
        "repository", "repo", "hawking",
        path=".", lang="none", public=True, subsystem="shared",
    ))

    file_set: set[str] = set()
    dirs: set[str] = set()

    for rel in files:
        ext = Path(rel).suffix
        if ext not in LOC_LANGS and ext not in {".toml"}:
            # still track Cargo.toml etc. lightly
            if not rel.endswith("Cargo.toml") and not rel.endswith("package.json"):
                continue
        file_set.add(rel)
        parent = str(Path(rel).parent)
        if parent != ".":
            parts = parent.split("/")
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]))

    for d in sorted(dirs):
        did = f"dir:{d}"
        g.add_node(make_node(
            "directory", did, d, path=d, lang="none",
            subsystem=subsystem_for(d + "/"),
        ))
        # parent contains
        parent = str(Path(d).parent)
        if parent == ".":
            g.ensure_contains("repo", did, evidence="ast")
        else:
            g.ensure_contains(f"dir:{parent}", did, evidence="ast")

    for rel in sorted(file_set):
        loc = physical_loc(repo, rel)
        if loc == 0 and not (repo / rel).exists():
            continue
        vendored, generated = classify_path(rel)
        fid = f"file:{rel}"
        lang = lang_for(rel)
        if rel.endswith("Cargo.toml"):
            lang = "toml"
        g.add_node(make_node(
            "file", fid, Path(rel).name, path=rel, lang=lang,
            loc=loc,
            vendored=vendored,
            generated=generated,
            test=is_test_path(rel),
            subsystem=subsystem_for(rel),
            public=False,
            complexity=1,
            side_effects=["none"],
            security_sensitive=False,
        ))
        parent = str(Path(rel).parent)
        if parent == ".":
            g.ensure_contains("repo", fid, evidence="ast")
        else:
            g.ensure_contains(f"dir:{parent}", fid, evidence="ast")

    return file_set


def build_graph(repo: Path) -> tuple[Graph, dict[str, Any]]:
    t0 = time.perf_counter()
    stats: dict[str, Any] = {}
    g = Graph()
    all_files = git_ls_files(repo)
    file_set = ensure_dirs_and_files(repo, g, all_files)

    # Cargo
    print("[graph] cargo metadata…", flush=True)
    cargo_ctx = extract_cargo(repo, g)
    # attach crate contains for source files
    for rel in file_set:
        if rel.endswith(".rs"):
            cname = crate_for_path(rel, cargo_ctx)
            if cname:
                cid = cargo_ctx["name_to_id"].get(cname)
                if cid:
                    g.ensure_contains(cid, f"file:{rel}", evidence="cargo")

    # Partition sources
    rust_files = sorted(
        f for f in file_set if f.endswith(".rs") and not classify_path(f)[0]
    )
    py_files = sorted(
        f for f in file_set if f.endswith(".py") and not classify_path(f)[0]
    )
    ts_files = sorted(
        f for f in file_set
        if Path(f).suffix in {".ts", ".tsx", ".js", ".jsx"}
        and not classify_path(f)[0]
    )
    metal_files = sorted(f for f in file_set if f.endswith(".metal"))

    print(f"[graph] rust files: {len(rust_files)}…", flush=True)
    rust_idx = extract_all_rust(repo, rust_files, g, cargo_ctx)

    print(f"[graph] python files: {len(py_files)}…", flush=True)
    py_idx = extract_all_python(repo, py_files, g)

    # merge indexes for cross-cutting passes
    indexes: dict[str, Any] = {
        "fns_by_name": defaultdict(list),
        "fns_by_qual": defaultdict(list),
        "types_by_name": defaultdict(list),
        "fn_meta": {},
        "test_nodes": {},
        "metal_fns": {},
        "all_files": file_set,
    }
    for k in ("fns_by_name", "fns_by_qual", "types_by_name"):
        for name, lst in rust_idx.get(k, {}).items():
            indexes[k][name].extend(lst)
        for name, lst in py_idx.get(k, {}).items():
            indexes[k][name].extend(lst)
    indexes["fn_meta"].update(rust_idx.get("fn_meta", {}))
    indexes["fn_meta"].update(py_idx.get("fn_meta", {}))
    indexes["test_nodes"].update(rust_idx.get("test_nodes", {}))
    indexes["test_nodes"].update(py_idx.get("test_nodes", {}))

    print(f"[graph] typescript files: {len(ts_files)}…", flush=True)
    extract_all_typescript(repo, ts_files, g, indexes)

    print(f"[graph] metal files: {len(metal_files)}…", flush=True)
    for rel in metal_files:
        extract_metal_file(repo, rel, g, indexes)
    link_metal_from_rust(repo, g, indexes)

    print("[graph] registries…", flush=True)
    extract_registries(repo, g, indexes)

    print("[graph] git co-change…", flush=True)
    extract_git_cochange(repo, g, file_set)

    print("[graph] test coverage…", flush=True)
    mark_test_coverage(g, indexes)

    print("[graph] runtime hot (static)…", flush=True)
    mark_runtime_hot(repo, g, indexes)

    g.compute_fan()

    stats["wall_s"] = time.perf_counter() - t0
    stats["n_nodes"] = len(g.nodes)
    stats["n_edges"] = len(g.edges)
    stats["rust_files"] = len(rust_files)
    stats["py_files"] = len(py_files)
    stats["ts_files"] = len(ts_files)
    stats["metal_files"] = len(metal_files)
    return g, stats


def emit(g: Graph, out_dir: Path, what: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if what in ("all", "jsonl"):
        p = out_dir / "HAWKING_SEMANTIC_GRAPH.jsonl"
        g.emit_jsonl(p)
        written.append(p)
        print(f"[graph] wrote {p}", flush=True)
    if what in ("all", "gexf"):
        p = out_dir / "HAWKING_SEMANTIC_GRAPH.gexf"
        g.emit_gexf(p)
        written.append(p)
        print(f"[graph] wrote {p}", flush=True)
    if what in ("all", "dot"):
        p = out_dir / "HAWKING_SEMANTIC_GRAPH.dot"
        g.emit_dot(p)
        written.append(p)
        print(f"[graph] wrote {p}", flush=True)
    if what in ("all", "behaviour"):
        p = out_dir / "HAWKING_BEHAVIOUR_TO_CODE_MAP.json"
        g.emit_behaviour_stub(p)
        written.append(p)
        print(f"[graph] wrote {p}", flush=True)
    return written


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(repo: Path, out_dir: Path) -> int:
    """Self-checks; exit non-zero on failure. Includes a full re-run for byte identity."""
    errors: list[str] = []
    jsonl = out_dir / "HAWKING_SEMANTIC_GRAPH.jsonl"
    if not jsonl.exists():
        print("[verify] building graph for verify…", flush=True)
        g, _ = build_graph(repo)
        emit(g, out_dir, "all")

    # Parse jsonl
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    prev_key = None
    line_no = 0
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line_no += 1
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid json: {e}")
                continue
            kind = rec.get("kind")
            if kind == "node":
                nid = rec["id"]
                if nid in nodes:
                    errors.append(f"duplicate node id: {nid}")
                nodes[nid] = rec
                key = ("node", rec.get("type"), nid)
            elif kind == "edge":
                edges.append(rec)
                key = ("edge", rec.get("type"), rec.get("id"))
            else:
                errors.append(f"line {line_no}: bad kind {kind!r}")
                continue
            if prev_key is not None:
                # kind desc: node before edge; then type, id
                def sort_tuple(k):
                    kind_rank = 0 if k[0] == "node" else 1
                    return (kind_rank, k[1] or "", k[2] or "")
                if sort_tuple(key) < sort_tuple(prev_key):
                    errors.append(
                        f"non-sorted output at line {line_no}: {key} after {prev_key}"
                    )
            prev_key = key

    # edges reference existing nodes
    for e in edges:
        if e["src"] not in nodes:
            errors.append(f"edge {e.get('id')} missing src {e['src']}")
        if e["dst"] not in nodes:
            errors.append(f"edge {e.get('id')} missing dst {e['dst']}")

    # required attr keys on a sample of nodes
    required = {
        "loc", "fan_in", "fan_out", "betweenness", "runtime_hot",
        "compile_cost_ms", "binary_bytes", "test_covered",
        "change_freq_90d", "change_freq_all", "complexity",
        "side_effects", "security_sensitive", "public", "generated",
        "vendored", "test", "subsystem",
    }
    for nid, n in list(nodes.items())[:50]:
        missing = required - set((n.get("attrs") or {}).keys())
        if missing:
            errors.append(f"node {nid} missing attrs: {sorted(missing)}")
            break

    # Determinism: re-run and compare jsonl bytes
    print("[verify] re-run for byte-identity…", flush=True)
    g2, _ = build_graph(repo)
    tmp = out_dir / ".HAWKING_SEMANTIC_GRAPH.verify.jsonl"
    g2.emit_jsonl(tmp)
    h1 = file_sha256(jsonl)
    h2 = file_sha256(tmp)
    if h1 != h2:
        errors.append(f"non-deterministic jsonl: {h1} != {h2}")
    else:
        print(f"[verify] jsonl sha256 match: {h1[:16]}…", flush=True)
    try:
        tmp.unlink()
    except OSError:
        pass

    # Also check gexf/dot if present
    for name in ("HAWKING_SEMANTIC_GRAPH.gexf", "HAWKING_SEMANTIC_GRAPH.dot",
                 "HAWKING_BEHAVIOUR_TO_CODE_MAP.json"):
        p = out_dir / name
        if not p.exists():
            # emit missing from g2
            if name.endswith(".gexf"):
                g2.emit_gexf(p)
            elif name.endswith(".dot"):
                g2.emit_dot(p)
            else:
                g2.emit_behaviour_stub(p)

    if errors:
        print(f"[verify] FAIL ({len(errors)} errors)", flush=True)
        for e in errors[:40]:
            print(f"  - {e}", flush=True)
        if len(errors) > 40:
            print(f"  … and {len(errors) - 40} more", flush=True)
        return 1
    print(f"[verify] OK  nodes={len(nodes)} edges={len(edges)}", flush=True)
    return 0


def count_report(g: Graph) -> dict[str, Any]:
    nc = Counter(n.type for n in g.nodes.values())
    ec = Counter(e.type for e in g.edges.values())
    # LOC agreement
    file_loc = 0
    for n in g.nodes.values():
        if n.type != "file":
            continue
        if n.attrs.get("vendored") or n.attrs.get("generated"):
            continue
        # only code langs matching hawking_loc
        if n.lang in ("rust", "python", "typescript", "metal", "shell", "lean", "markdown"):
            file_loc += int(n.attrs.get("loc") or 0)
    fn_count = sum(1 for n in g.nodes.values() if n.type == "function")
    pub_count = sum(
        1 for n in g.nodes.values()
        if n.attrs.get("public") and n.type in ("function", "type")
    )
    # topology-style public is broader; report function+type public
    return {
        "nodes_by_type": dict(sorted(nc.items())),
        "edges_by_type": dict(sorted(ec.items())),
        "file_loc_active": file_loc,
        "function_nodes": fn_count,
        "public_fn_type": pub_count,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hawking semantic graph extractor (G1)")
    ap.add_argument("--repo", default=str(REPO), help="repository root")
    ap.add_argument(
        "--out",
        default=str(REPO / "workspace" / "ops" / "build" / "graph"),
        help="output directory (default: workspace/ops/build/graph, gitignored — regenerable)",
    )
    ap.add_argument(
        "--emit",
        choices=["all", "jsonl", "gexf", "dot", "behaviour", "none"],
        default="none",
        help="what to emit",
    )
    ap.add_argument("--verify", action="store_true", help="self-checks; non-zero on failure")
    ap.add_argument("--stats", action="store_true", help="print counts after build")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    if args.emit != "none":
        g, stats = build_graph(repo)
        emit(g, out_dir, args.emit)
        rep = count_report(g)
        print("[graph] node counts:", json.dumps(rep["nodes_by_type"], sort_keys=True))
        print("[graph] edge counts:", json.dumps(rep["edges_by_type"], sort_keys=True))
        print(f"[graph] file_loc_active={rep['file_loc_active']} functions={rep['function_nodes']} "
              f"wall={stats['wall_s']:.1f}s")
        if args.stats:
            print(json.dumps({**stats, **rep}, indent=2, sort_keys=True))

    if args.verify:
        return verify(repo, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
