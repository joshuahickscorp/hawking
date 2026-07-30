#!/usr/bin/env python3.12
"""Run the eight G2 topology analyses and emit cluster map + recomposition queue.

Usage:
    python3.12 tools/graph/hawking_analyze.py --graph build/graph/HAWKING_SEMANTIC_GRAPH.jsonl
    python3.12 tools/graph/hawking_analyze.py --graph /tmp/fixture.jsonl --out /tmp/out \\
        --behaviour-map /tmp/beh.json --betweenness-k 64

Outputs (under --out):
    HAWKING_CLUSTER_MAP.json
    HAWKING_RECOMPOSITION_CANDIDATES.json
    HAWKING_ANALYSIS_REPORT.json   (timings + planted verification if requested)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_GRAPH_DIR = Path(__file__).resolve().parent
_BUILD_GRAPH = _GRAPH_DIR.parents[1] / "build" / "graph"
if str(_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_DIR))

from analyses import run_all  # noqa: E402
from graph_io import SemanticGraph  # noqa: E402
from recompose import build_candidates  # noqa: E402


def verify_planted(cluster: dict[str, Any], planted_path: Path) -> dict[str, Any]:
    """Check which planted structures the analyses recovered. Misses are findings."""
    planted = json.loads(planted_path.read_text(encoding="utf-8"))
    analyses = cluster.get("analyses", {})
    findings: dict[str, Any] = {"planted_path": str(planted_path), "checks": []}

    def add(name: str, found: bool, detail: str) -> None:
        findings["checks"].append({"name": name, "found": found, "detail": detail})

    # File SCCs
    file_sccs = analyses.get("scc", {}).get("machine", {}).get("file_sccs", [])
    scc_member_sets = [set(s.get("members") or []) for s in file_sccs]
    for p in planted.get("sccs_file", []):
        want = set(p.get("members") or [])
        hit = any(want <= s or len(want & s) >= max(2, len(want) - 1) for s in scc_member_sets)
        add(
            p.get("id", "scc_file"),
            hit,
            f"wanted {sorted(want)}; matched={hit}; n_file_sccs={len(file_sccs)}",
        )

    crate_sccs = analyses.get("scc", {}).get("machine", {}).get("crate_sccs", [])
    crate_sets = [set(s.get("members") or []) for s in crate_sccs]
    for p in planted.get("sccs_crate", []):
        want = set(p.get("members") or [])
        hit = any(want <= s or want == s for s in crate_sets)
        add(
            p.get("id", "scc_crate"),
            hit,
            f"wanted {sorted(want)}; matched={hit}; crate_sccs={crate_sets}",
        )

    # Communities: alpha/beta members should concentrate in some community
    comms = analyses.get("communities", {}).get("machine", {}).get("communities", [])
    for key, meta in (planted.get("communities") or {}).items():
        want = set(meta.get("members") or [])
        best = 0
        best_id = None
        best_dirs = None
        for c in comms:
            members = set(c.get("members") or [])
            # members may be truncated — also use that community fully if small
            ov = len(want & members)
            # If truncated, approximate via overlap ratio on reported members
            if ov > best:
                best = ov
                best_id = c.get("id")
                best_dirs = c.get("n_directories")
        # success if >= 50% of planted members appear together (or >= 10)
        # Require majority of planted members co-located in one community
        threshold = max(8, (len(want) * 2 + 2) // 3)  # >= ~2/3
        hit = best >= min(threshold, len(want))
        add(
            meta.get("id", f"community_{key}"),
            hit,
            f"overlap={best}/{len(want)} in {best_id} n_dirs={best_dirs} "
            f"threshold={threshold} "
            f"(scatter expected={meta.get('scatter_dirs') or meta.get('n_dirs')})",
        )

    # Broker — accept top-betweenness, broker flag, or articulation point
    broker = planted.get("broker") or {}
    if broker:
        brokers = analyses.get("betweenness", {}).get("machine", {}).get("brokers", [])
        top = analyses.get("betweenness", {}).get("machine", {}).get("top_betweenness", [])
        arts = analyses.get("betweenness", {}).get("machine", {}).get(
            "articulation_points", []
        )
        bid = broker.get("file")
        in_brokers = any(b.get("id") == bid for b in brokers)
        in_top = any(b.get("id") == bid for b in top[:100])
        in_art = any(b.get("id") == bid for b in arts)
        # rank among all scored nodes
        rank = next((i for i, b in enumerate(top) if b.get("id") == bid), None)
        hit = in_brokers or in_top or in_art or (rank is not None and rank < 200)
        add(
            broker.get("id", "broker"),
            hit,
            f"file={bid} in_brokers={in_brokers} in_top100={in_top} "
            f"articulation={in_art} rank={rank}",
        )

    # Dominator shared path
    dom = planted.get("dominator_chain") or {}
    if dom:
        shared_nodes = set()
        for s in analyses.get("dominators", {}).get("machine", {}).get("shared_control_nodes", []):
            shared_nodes.add(s.get("id"))
        for ch in analyses.get("dominators", {}).get("machine", {}).get("shared_chains", []):
            shared_nodes.update(ch.get("shared_nodes") or [])
        want = set(dom.get("shared_path") or [])
        ov = want & shared_nodes
        hit = len(ov) >= max(1, len(want) // 2)
        add(
            dom.get("id", "dominator"),
            hit,
            f"shared_path overlap {sorted(ov)} / wanted {sorted(want)}",
        )

    # Clone families
    fams = analyses.get("clones", {}).get("machine", {}).get("families", [])
    sig_to_fam = {f.get("signature"): f for f in fams}
    for p in planted.get("clone_families", []):
        sig = p.get("signature")
        fam = sig_to_fam.get(sig)
        want = set(p.get("members") or [])
        if fam:
            got = set(fam.get("members") or [])
            hit = want <= got or len(want & got) >= len(want) - 1
            add(
                p.get("id", "clone"),
                hit,
                f"signature hit; members {len(want & got)}/{len(want)}; "
                f"match_kind={fam.get('match_kind')}",
            )
        else:
            add(p.get("id", "clone"), False, f"signature {sig!r} not found in families")

    # Wrapper ring
    wrap = planted.get("wrapper_chain") or {}
    if wrap:
        rings = analyses.get("fanin", {}).get("machine", {}).get("rings", [])
        auth = wrap.get("authority")
        hit = any(r.get("authority") == auth for r in rings)
        matched = next((r for r in rings if r.get("authority") == auth), None)
        add(
            wrap.get("id", "wrapper"),
            hit,
            f"authority={auth} found={hit} "
            f"adapter_count={matched.get('adapter_count') if matched else None} "
            f"planted={wrap.get('wrapper_count')}",
        )

    # Co-change split
    for p in planted.get("cochange_split") or []:
        pairs = analyses.get("cochange", {}).get("machine", {}).get("pairs", [])
        a, b = p.get("a"), p.get("b")
        hit = any(
            {x.get("a"), x.get("b")} == {a, b}
            or {x.get("a_path"), x.get("b_path")}
            == {gpath for gpath in (a, b)}
            for x in pairs
        )
        # path-based
        if not hit:
            for x in pairs:
                ids = {x.get("a"), x.get("b")}
                if a in ids and b in ids:
                    hit = True
                    break
        add(p.get("id", "cochange"), hit, f"pair {a} — {b} found={hit}")

    findings["n_found"] = sum(1 for c in findings["checks"] if c["found"])
    findings["n_total"] = len(findings["checks"])
    findings["n_missed"] = findings["n_total"] - findings["n_found"]
    findings["missed"] = [c["name"] for c in findings["checks"] if not c["found"]]
    return findings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="HAWKING_SEMANTIC_GRAPH.jsonl path")
    p.add_argument("--out", default=str(_BUILD_GRAPH),
                   help="Output directory (default: build/graph, gitignored — regenerable)")
    p.add_argument(
        "--behaviour-map",
        default=None,
        help="HAWKING_BEHAVIOUR_TO_CODE_MAP.json (optional)",
    )
    p.add_argument("--betweenness-k", type=int, default=64)
    p.add_argument(
        "--planted-manifest",
        default=None,
        help="If set, verify planted structures and include in report",
    )
    p.add_argument(
        "--also-report",
        default=None,
        help="Extra path for HAWKING_ANALYSIS_REPORT.json (default: under --out)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = Path(args.graph)
    if not graph_path.is_file():
        print(f"error: graph not found: {graph_path}", file=sys.stderr)
        return 2

    # Default behaviour map next to graph or at out/repo root
    beh = args.behaviour_map
    if beh is None:
        candidates = [
            graph_path.parent / "HAWKING_BEHAVIOUR_TO_CODE_MAP.json",
            out_dir / "HAWKING_BEHAVIOUR_TO_CODE_MAP.json",
            _BUILD_GRAPH / "HAWKING_BEHAVIOUR_TO_CODE_MAP.json",
        ]
        for c in candidates:
            if c.is_file():
                beh = str(c)
                break

    print(f"loading {graph_path} …", flush=True)
    t_load = time.perf_counter()
    g = SemanticGraph.load(graph_path)
    load_s = time.perf_counter() - t_load
    print(f"loaded {len(g.nodes)} nodes, {len(g.edges)} edges in {load_s:.2f}s", flush=True)

    print("running analyses …", flush=True)
    t0 = time.perf_counter()
    cluster = run_all(g, behaviour_map=beh, betweenness_k=args.betweenness_k)
    cluster["load_seconds"] = load_s
    cluster["wall_seconds"] = time.perf_counter() - t0
    cluster["graph_path"] = str(graph_path)
    cluster["betweenness_k"] = args.betweenness_k

    cluster_path = out_dir / "HAWKING_CLUSTER_MAP.json"
    cluster_path.write_text(json.dumps(cluster, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {cluster_path}", flush=True)

    print("building recomposition candidates …", flush=True)
    candidates = build_candidates(g, cluster)
    cand_path = out_dir / "HAWKING_RECOMPOSITION_CANDIDATES.json"
    cand_path.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {cand_path} ({candidates['n_candidates']} candidates)", flush=True)

    planted_findings = None
    if args.planted_manifest:
        planted_findings = verify_planted(cluster, Path(args.planted_manifest))
        print(
            f"planted verification: {planted_findings['n_found']}/{planted_findings['n_total']} found; "
            f"missed={planted_findings['missed']}",
            flush=True,
        )

    report = {
        "schema": "hawking.analysis_report.v1",
        "graph": str(graph_path),
        "graph_summary": cluster.get("graph_summary"),
        "timings_seconds": {
            "load": load_s,
            **cluster.get("timings_seconds", {}),
            "total_analyses": cluster.get("total_seconds"),
            "wall": cluster.get("wall_seconds"),
        },
        "budget_seconds": 20 * 60,
        "within_budget": (cluster.get("wall_seconds") or 0) < 20 * 60,
        "analysis_summaries": {
            name: body.get("summary")
            for name, body in cluster.get("analyses", {}).items()
        },
        "n_candidates": candidates.get("n_candidates"),
        "planted_verification": planted_findings,
        "outputs": {
            "cluster_map": str(cluster_path),
            "candidates": str(cand_path),
        },
    }
    report_path = Path(args.also_report) if args.also_report else out_dir / "HAWKING_ANALYSIS_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}", flush=True)

    # Timing table to stdout
    print("\n=== Timing table (seconds) ===")
    print(f"{'phase':<24} {'seconds':>10}")
    print("-" * 36)
    print(f"{'load':<24} {load_s:>10.2f}")
    for name, sec in (cluster.get("timings_seconds") or {}).items():
        print(f"{name:<24} {sec:>10.2f}")
    print(f"{'TOTAL analyses':<24} {cluster.get('total_seconds', 0):>10.2f}")
    print(f"{'WALL':<24} {cluster.get('wall_seconds', 0):>10.2f}")
    print(f"within 20min budget: {report['within_budget']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
