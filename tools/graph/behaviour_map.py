#!/usr/bin/env python3.12
"""Bind the Behaviour Constitution to the semantic graph.

`REBUILD_BEHAVIOUR_CONSTITUTION.json` says what the system must observably do and names the
paths that implement each behaviour today. The semantic graph says what calls what. Joining
them answers the question the rebuild actually turns on: **which code is reachable from no
observable contract at all**, and is therefore a deletion or replacement candidate rather
than something that has to be re-expressed.

    python3.12 tools/graph/behaviour_map.py    # writes workspace/ops/build/graph/HAWKING_BEHAVIOUR_TO_CODE_MAP.json

Reachability is deliberately generous. A behaviour's seed set is every graph node whose path
is named in its `current_code`, and the closure follows `contains` (a file owns its
functions) and `calls` (a named entry drags in what it needs) to a bounded depth. Being
generous is the conservative choice here: it *shrinks* the uncovered bucket, so anything
still uncovered is uncovered under an assumption that favours keeping code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPTH = 6


def load_graph(path: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    nodes: dict[str, dict] = {}
    out: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["kind"] == "node":
                nodes[d["id"]] = d
            elif d["type"] in ("contains", "calls"):
                out[d["src"]].append(d["dst"])
    return nodes, out


def seeds_for(paths: list[str], by_path: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    for p in paths:
        p = p.strip()
        if not p:
            continue
        # A behaviour may name a file, or a directory standing for everything under it.
        for known, ids in by_path.items():
            if known == p or known.startswith(p.rstrip("/") + "/"):
                seen.update(ids)
    return seen


def closure(seeds: set[str], out: dict[str, list[str]], depth: int) -> set[str]:
    seen = set(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier:
        nid, d = frontier.popleft()
        if d >= depth:
            continue
        for dst in out.get(nid, ()):
            if dst not in seen:
                seen.add(dst)
                frontier.append((dst, d + 1))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--graph",
        default=str(ROOT / "workspace" / "ops" / "build" / "graph" / "HAWKING_SEMANTIC_GRAPH.jsonl"),
    )
    ap.add_argument(
        "--constitution",
        default="workspace/campaign/evidence/runtime/rebuild/REBUILD_BEHAVIOUR_CONSTITUTION.json",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "workspace" / "ops" / "build" / "graph" / "HAWKING_BEHAVIOUR_TO_CODE_MAP.json"),
    )
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = ap.parse_args()

    gpath = ROOT / args.graph
    if not gpath.exists():
        print(f"graph not found: {gpath}\nrun tools/graph/hawking_graph.py --emit all first",
              file=sys.stderr)
        return 2

    nodes, out = load_graph(gpath)
    by_path: dict[str, list[str]] = defaultdict(list)
    for nid, n in nodes.items():
        if n.get("path"):
            by_path[n["path"]].append(nid)

    const = json.loads((ROOT / args.constitution).read_text(encoding="utf-8"))
    behaviours: dict[str, dict] = {}
    unbound: list[str] = []

    for b in const["behaviours"]:
        seeds = seeds_for(b.get("current_code") or [], by_path)
        if not seeds:
            unbound.append(b["id"])
        reach = closure(seeds, out, args.depth)
        behaviours[b["id"]] = {
            "domain": b["domain"],
            "criticality": b["criticality"],
            "seed_paths": b.get("current_code") or [],
            "seed_nodes": len(seeds),
            "reachable_nodes": sorted(reach),
        }

    covered = set()
    for body in behaviours.values():
        covered.update(body["reachable_nodes"])

    uncovered_loc = 0
    uncovered_by_sub: dict[str, int] = defaultdict(int)
    for nid, n in nodes.items():
        if n["type"] != "file" or nid in covered:
            continue
        a = n["attrs"]
        if a.get("vendored") or a.get("generated"):
            continue
        uncovered_loc += a.get("loc", 0)
        uncovered_by_sub[a.get("subsystem", "?")] += a.get("loc", 0)

    doc = {
        "schema": "hawking.behaviour_to_code_map.v1",
        "constitution_commit": const.get("commit"),
        "graph": args.graph,
        "closure_depth": args.depth,
        "closure_edges": ["contains", "calls"],
        "behaviours": behaviours,
        "summary": {
            "n_behaviours": len(behaviours),
            "behaviours_with_no_graph_binding": unbound,
            "covered_nodes": len(covered),
            "uncovered_file_LOC": uncovered_loc,
            "uncovered_LOC_by_subsystem": dict(sorted(uncovered_by_sub.items(),
                                                      key=lambda kv: -kv[1])),
        },
        "reading": (
            "Uncovered LOC is reachable from no behaviour in the constitution under a "
            "deliberately generous closure. It is a deletion/replacement CANDIDATE set, not "
            "a verdict: the constitution may be incomplete, and every removal still has to "
            "pass the black-box gate."
        ),
    }
    (ROOT / args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    s = doc["summary"]
    print(f"behaviours bound: {s['n_behaviours'] - len(unbound)}/{s['n_behaviours']}"
          f"  ({len(unbound)} with no graph binding)")
    print(f"covered nodes: {s['covered_nodes']:,}")
    print(f"uncovered file LOC: {s['uncovered_file_LOC']:,}")
    for k, v in s["uncovered_LOC_by_subsystem"].items():
        print(f"  {k:12s} {v:,}")
    return 0


def _selfcheck() -> None:
    out = {"a": ["b"], "b": ["c"], "c": ["d"]}
    assert closure({"a"}, out, 6) == {"a", "b", "c", "d"}
    assert closure({"a"}, out, 1) == {"a", "b"}
    by_path = {"x/y.py": ["file:x/y.py"], "x/z.py": ["file:x/z.py"], "w/q.py": ["file:w/q.py"]}
    assert seeds_for(["x/"], by_path) == {"file:x/y.py", "file:x/z.py"}
    assert seeds_for(["x/y.py"], by_path) == {"file:x/y.py"}
    assert seeds_for(["nope/"], by_path) == set()
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
