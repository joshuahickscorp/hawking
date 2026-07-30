#!/usr/bin/env python3.12
"""Emit deterministic Graphviz DOT checkpoint diagrams for the crate level.

Usage:
    python3.12 tools/graph/render_dot.py --graph build/graph/HAWKING_SEMANTIC_GRAPH.jsonl --out tools/graph/HAWKING_CRATE_GRAPH.dot
    python3.12 tools/graph/render_dot.py --graph fixture.jsonl --out /tmp/crates.dot --cluster-map build/graph/HAWKING_CLUSTER_MAP.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_GRAPH_DIR = Path(__file__).resolve().parent
if str(_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_DIR))

from graph_io import SemanticGraph  # noqa: E402


def crate_of_path(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("crates/"):
        parts = path.split("/")
        if len(parts) >= 2:
            return f"crate:{parts[1]}"
    return None


def build_crate_dot(
    g: SemanticGraph,
    cluster_map: dict | None = None,
    *,
    title: str = "Hawking crate-level checkpoint",
) -> str:
    # Aggregate imports/calls to crate-crate edges
    weights: dict[tuple[str, str], float] = defaultdict(float)
    file_to_crate: dict[str, str] = {}
    for fid in g.by_type.get("file", []):
        c = crate_of_path(g.path_of(fid))
        if c:
            file_to_crate[fid] = c
    for nid, n in g.nodes.items():
        if n["type"] == "function" and n.get("path"):
            c = crate_of_path(n["path"])
            if c:
                file_to_crate[nid] = c

    crates = sorted(g.by_type.get("crate", []))
    crate_loc: dict[str, int] = {c: 0 for c in crates}
    for fid, c in file_to_crate.items():
        if c in crate_loc and g.nodes.get(fid, {}).get("type") == "file":
            crate_loc[c] = crate_loc.get(c, 0) + g.loc_of(fid)

    for e in g.edges:
        if e["type"] not in ("imports", "calls", "runtime_calls"):
            continue
        sc = file_to_crate.get(e["src"])
        if e["src"].startswith("crate:"):
            sc = e["src"]
        dc = file_to_crate.get(e["dst"])
        if e["dst"].startswith("crate:"):
            dc = e["dst"]
        if sc and dc and sc != dc and sc in crate_loc and dc in crate_loc:
            weights[(sc, dc)] += float(e.get("attrs", {}).get("weight", 1.0))

    # Community colouring if available — map crate to dominant community
    crate_comm: dict[str, str] = {}
    if cluster_map:
        for comm in (
            cluster_map.get("analyses", {})
            .get("communities", {})
            .get("machine", {})
            .get("communities", [])
        ):
            cid = comm.get("id", "?")
            for m in comm.get("members") or []:
                c = file_to_crate.get(m) or crate_of_path(
                    m[5:] if m.startswith("file:") else None
                )
                if c:
                    crate_comm.setdefault(c, cid)

    # SCC highlight
    scc_crates: set[str] = set()
    if cluster_map:
        for scc in (
            cluster_map.get("analyses", {})
            .get("scc", {})
            .get("machine", {})
            .get("crate_sccs", [])
        ):
            scc_crates.update(scc.get("members") or [])

    # Deterministic palette
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ac",
    ]
    comm_ids = sorted(set(crate_comm.values()))
    comm_color = {c: colors[i % len(colors)] for i, c in enumerate(comm_ids)}

    lines = [
        "digraph hawking_crates {",
        "  graph [rankdir=LR, fontname=Helvetica, fontsize=12, label="
        + json.dumps(title)
        + ", labelloc=t];",
        "  node [shape=box, style=filled, fontname=Helvetica, fontsize=10];",
        "  edge [fontname=Helvetica, fontsize=8];",
    ]

    for cid in crates:
        name = cid.replace("crate:", "")
        loc = crate_loc.get(cid, 0)
        label = f"{name}\\n{loc} LOC"
        fill = comm_color.get(crate_comm.get(cid, ""), "#dddddd")
        penwidth = "3" if cid in scc_crates else "1"
        color = "#c0392b" if cid in scc_crates else "#333333"
        lines.append(
            f'  "{cid}" [label={json.dumps(label)}, fillcolor="{fill}", '
            f'color="{color}", penwidth={penwidth}];'
        )

    for (s, d), w in sorted(weights.items(), key=lambda x: (x[0][0], x[0][1])):
        # skip tiny noise
        if w < 1.0:
            continue
        pen = min(6.0, 1.0 + w / 20.0)
        lines.append(f'  "{s}" -> "{d}" [penwidth={pen:.2f}, label="{w:.0f}"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cluster-map", default=None)
    p.add_argument("--title", default="Hawking crate-level checkpoint")
    args = p.parse_args(argv)

    g = SemanticGraph.load(args.graph)
    cluster = None
    if args.cluster_map and Path(args.cluster_map).is_file():
        cluster = json.loads(Path(args.cluster_map).read_text(encoding="utf-8"))

    dot = build_crate_dot(g, cluster, title=args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dot, encoding="utf-8")
    print(f"wrote {out} ({len(dot)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
