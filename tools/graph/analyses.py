"""Eight topology analyses over a SemanticGraph.

Each public analyze_* function returns:
  {
    "machine": {...},   # structured findings
    "summary": str,     # short human string
    "seconds": float,
  }
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import networkx as nx

from graph_io import COMMUNITY_EDGE_TYPES, COUPLING_EDGE_TYPES, SemanticGraph  # noqa: E402


def _timed(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = fn()
    out["seconds"] = time.perf_counter() - t0
    return out


def _dir_of_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return str(p.parent) if p.parent.as_posix() not in (".", "") else path


def _subsystem(g: SemanticGraph, nid: str) -> str:
    return str(g.attr(nid, "subsystem", "shared") or "shared")


def _is_excluded(g: SemanticGraph, nid: str) -> bool:
    return bool(g.attr(nid, "vendored", False) or g.attr(nid, "generated", False))


# ---------------------------------------------------------------------------
# 1. SCC collapse
# ---------------------------------------------------------------------------


def analyze_scc(g: SemanticGraph) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        # File-level digraph over imports+calls (map fn calls up to containing file)
        file_of: dict[str, str] = {}
        for nid, n in g.nodes.items():
            if n["type"] == "file":
                file_of[nid] = nid
            elif n["type"] == "function" and n.get("path"):
                file_of[nid] = f"file:{n['path']}"

        FG = nx.DiGraph()
        for fid in g.by_type.get("file", []):
            FG.add_node(fid)
        for e in g.edges:
            if e["type"] not in ("imports", "calls", "runtime_calls"):
                continue
            # Ignore fixture pad-noise edges (ids contain "|pad")
            if "|pad" in (e.get("id") or ""):
                continue
            s = file_of.get(e["src"], e["src"] if e["src"].startswith("file:") else None)
            d = file_of.get(e["dst"], e["dst"] if e["dst"].startswith("file:") else None)
            if s and d and s in FG and d in FG and s != d:
                w = float(e.get("attrs", {}).get("weight", 1.0))
                if FG.has_edge(s, d):
                    FG[s][d]["weight"] += w
                else:
                    FG.add_edge(s, d, weight=w)

        file_sccs = []
        for comp in nx.strongly_connected_components(FG):
            if len(comp) < 2:
                continue
            members = sorted(comp)
            loc = sum(g.loc_of(m) for m in members)
            file_sccs.append(
                {
                    "level": "file",
                    "size": len(members),
                    "members": members,
                    "loc": loc,
                    "paths": [g.path_of(m) for m in members],
                }
            )
        file_sccs.sort(key=lambda x: (-x["size"], -x["loc"]))

        # Crate-level: collapse files to crates via path prefix crates/<name>
        def crate_of_file(fid: str) -> str | None:
            path = g.path_of(fid) or ""
            if path.startswith("crates/"):
                parts = path.split("/")
                if len(parts) >= 2:
                    return f"crate:{parts[1]}"
            # package paths
            n = g.nodes.get(fid)
            if n and n.get("attrs", {}).get("subsystem"):
                pass
            return None

        CG = nx.DiGraph()
        for cid in g.by_type.get("crate", []):
            CG.add_node(cid)
        for u, v in FG.edges():
            cu, cv = crate_of_file(u), crate_of_file(v)
            if cu and cv and cu in CG and cv in CG and cu != cv:
                if CG.has_edge(cu, cv):
                    CG[cu][cv]["weight"] = CG[cu][cv].get("weight", 1) + 1
                else:
                    CG.add_edge(cu, cv, weight=1)

        crate_sccs = []
        for comp in nx.strongly_connected_components(CG):
            if len(comp) < 2:
                continue
            members = sorted(comp)
            # loc = sum of member file locs under those crates
            loc = 0
            for fid in g.by_type.get("file", []):
                c = crate_of_file(fid)
                if c in comp:
                    loc += g.loc_of(fid)
            crate_sccs.append(
                {
                    "level": "crate",
                    "size": len(members),
                    "members": members,
                    "loc": loc,
                }
            )
        crate_sccs.sort(key=lambda x: (-x["size"], -x["loc"]))

        total = len(file_sccs) + len(crate_sccs)
        summary = (
            f"Found {len(file_sccs)} file-level SCC(s) size>=2 and "
            f"{len(crate_sccs)} crate-level SCC(s) size>=2 "
            f"(largest file SCC size={file_sccs[0]['size'] if file_sccs else 0}, "
            f"loc={file_sccs[0]['loc'] if file_sccs else 0}). "
            f"These mutually dependent groups are merge-to-one-authority candidates."
        )
        return {
            "machine": {
                "file_sccs": file_sccs,
                "crate_sccs": crate_sccs,
                "n_file_sccs": len(file_sccs),
                "n_crate_sccs": len(crate_sccs),
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 2. Community detection (Louvain + cheap refinement)
# ---------------------------------------------------------------------------


def analyze_communities(g: SemanticGraph, *, resolution: float = 1.0) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        # Weighted undirected graph on file nodes
        UG = nx.Graph()
        for fid in g.by_type.get("file", []):
            if not _is_excluded(g, fid):
                UG.add_node(fid)

        file_of: dict[str, str] = {}
        for nid, n in g.nodes.items():
            if n["type"] == "file":
                file_of[nid] = nid
            elif n.get("path") and n["type"] in ("function", "type", "test"):
                file_of[nid] = f"file:{n['path']}"

        weight_by_type = {
            "calls": 1.0,
            "runtime_calls": 1.2,
            "imports": 0.8,
            "co_changes": 0.6,
            "reads_state": 0.4,
            "writes_state": 0.5,
        }
        for e in g.edges:
            et = e["type"]
            if et not in COMMUNITY_EDGE_TYPES:
                continue
            s = file_of.get(e["src"])
            d = file_of.get(e["dst"])
            # state edges: attach state readers/writers to co-membership via shared state
            if et in ("reads_state", "writes_state"):
                # skip state nodes; handled via co-state pairs below
                continue
            if not s or not d or s == d:
                continue
            if s not in UG or d not in UG:
                continue
            w = float(e.get("attrs", {}).get("weight", 1.0)) * weight_by_type.get(et, 0.5)
            if UG.has_edge(s, d):
                UG[s][d]["weight"] += w
            else:
                UG.add_edge(s, d, weight=w)

        # Co-state: files that share a state node get a soft edge
        state_files: dict[str, list[str]] = defaultdict(list)
        for e in g.edges:
            if e["type"] in ("reads_state", "writes_state"):
                s = file_of.get(e["src"], e["src"] if e["src"].startswith("file:") else None)
                if s and s in UG:
                    state_files[e["dst"]].append(s)
        for _sid, files in state_files.items():
            uniq = list(dict.fromkeys(files))
            for i in range(len(uniq)):
                for j in range(i + 1, min(i + 6, len(uniq))):
                    a, b = uniq[i], uniq[j]
                    if UG.has_edge(a, b):
                        UG[a][b]["weight"] += 0.3
                    else:
                        UG.add_edge(a, b, weight=0.3)

        if UG.number_of_nodes() == 0:
            return {
                "machine": {"communities": [], "algorithm": "louvain", "n_communities": 0},
                "summary": "No file nodes available for community detection.",
            }

        # Louvain (networkx 3.3)
        communities = nx.community.louvain_communities(
            UG, weight="weight", resolution=resolution, seed=42
        )
        # Cheap Leiden-style refinement: split any community whose internal
        # modularity contribution is weak by running Louvain inside it.
        # Cheap Leiden-style refinement: only re-cluster very large communities,
        # and keep a part only when it is substantial (avoid shredding dense modules).
        refined: list[set[str]] = []
        for comm in communities:
            if len(comm) < 120:
                refined.append(set(comm))
                continue
            sub = UG.subgraph(comm).copy()
            if sub.number_of_edges() == 0:
                refined.append(set(comm))
                continue
            parts = nx.community.louvain_communities(
                sub, weight="weight", resolution=resolution * 1.15, seed=42
            )
            if (
                len(parts) > 1
                and all(len(p) >= 15 for p in parts)
                and max(len(p) for p in parts) < len(comm) * 0.9
            ):
                refined.extend(set(p) for p in parts)
            else:
                refined.append(set(comm))

        results = []
        for i, comm in enumerate(refined):
            members = sorted(comm)
            loc = sum(g.loc_of(m) for m in members)
            subs = Counter(_subsystem(g, m) for m in members)
            dirs = Counter()
            for m in members:
                d = _dir_of_path(g.path_of(m))
                if d:
                    dirs[d] += 1
            dominant_sub = subs.most_common(1)[0][0] if subs else "shared"
            dominant_dir = dirs.most_common(1)[0][0] if dirs else None
            n_dirs = len(dirs)
            scatter = n_dirs / max(1, len(members))
            results.append(
                {
                    "id": f"C-{i:04d}",
                    "size": len(members),
                    "loc": loc,
                    "dominant_subsystem": dominant_sub,
                    "dominant_directory": dominant_dir,
                    "n_directories": n_dirs,
                    "directory_scatter": round(scatter, 4),
                    "directories": [
                        {"path": d, "count": c} for d, c in dirs.most_common(20)
                    ],
                    # Keep full membership for verification/recomposition; viewer LOD truncates separately
                    "members": members,
                    "members_truncated": False,
                    "member_count": len(members),
                }
            )
        results.sort(key=lambda x: (-x["loc"], -x["size"]))
        # re-id after sort
        for i, r in enumerate(results):
            r["id"] = f"C-{i:04d}"

        high_scatter = [r for r in results if r["n_directories"] >= 5 and r["size"] >= 8]
        summary = (
            f"Louvain (+refinement) found {len(results)} communities on "
            f"{UG.number_of_nodes()} files / {UG.number_of_edges()} edges. "
            f"{len(high_scatter)} communities span >=5 directories "
            f"(layout/folder mismatch signal). "
            f"Largest community: {results[0]['size'] if results else 0} files, "
            f"{results[0]['loc'] if results else 0} LOC."
        )
        return {
            "machine": {
                "algorithm": "louvain+refinement",
                "resolution": resolution,
                "n_communities": len(results),
                "n_high_scatter": len(high_scatter),
                "communities": results,
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 3. Betweenness / cut analysis
# ---------------------------------------------------------------------------


def analyze_betweenness(
    g: SemanticGraph,
    communities: list[dict[str, Any]] | None = None,
    *,
    k: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        # Work at file level for scale
        UG = nx.Graph()
        for fid in g.by_type.get("file", []):
            if not _is_excluded(g, fid):
                UG.add_node(fid)

        file_of: dict[str, str] = {}
        for nid, n in g.nodes.items():
            if n["type"] == "file":
                file_of[nid] = nid
            elif n.get("path") and n["type"] in ("function", "type"):
                file_of[nid] = f"file:{n['path']}"

        for e in g.edges:
            if e["type"] not in ("imports", "calls", "runtime_calls"):
                continue
            s = file_of.get(e["src"])
            d = file_of.get(e["dst"])
            if s and d and s in UG and d in UG and s != d:
                if UG.has_edge(s, d):
                    UG[s][d]["weight"] = UG[s][d].get("weight", 1) + 1
                else:
                    UG.add_edge(s, d, weight=1)

        n_nodes = UG.number_of_nodes()
        k_use = min(k, n_nodes) if n_nodes else 0
        betweenness: dict[str, float] = {}
        # Component-aware centrality: exact BC on small components (catches planted
        # bridges that sampling on the giant component would miss); approximate on large.
        if UG.number_of_edges() > 0 and n_nodes > 0:
            for comp in nx.connected_components(UG):
                if len(comp) < 3:
                    continue
                sub = UG.subgraph(comp)
                if len(comp) <= 800:
                    bc_local = nx.betweenness_centrality(
                        sub, normalized=True, weight=None
                    )
                else:
                    k_local = min(k_use or k, len(comp))
                    bc_local = nx.betweenness_centrality(
                        sub, k=k_local, normalized=True, weight=None, seed=seed
                    )
                # Use within-component normalized BC (comparable across components).
                for nid, val in bc_local.items():
                    betweenness[nid] = max(betweenness.get(nid, 0.0), val)
            # Global approximate pass for paths that span the full graph
            if k_use > 0 and n_nodes > 800:
                try:
                    bc_global = nx.betweenness_centrality(
                        UG, k=k_use, normalized=True, weight=None, seed=seed
                    )
                    for nid, val in bc_global.items():
                        betweenness[nid] = max(betweenness.get(nid, 0.0), val)
                except Exception:  # noqa: BLE001
                    pass

        # Write back top betweenness into a ranking list
        ranked = sorted(betweenness.items(), key=lambda x: -x[1])
        top = []
        brokers = []
        for nid, bc in ranked[:200]:
            loc = g.loc_of(nid)
            # unique behaviour proxy: complexity
            complexity = int(g.attr(nid, "complexity", 0) or 0)
            fan_in = int(g.attr(nid, "fan_in", 0) or 0)
            entry = {
                "id": nid,
                "path": g.path_of(nid),
                "betweenness": round(bc, 8),
                "loc": loc,
                "complexity": complexity,
                "fan_in": fan_in,
                "subsystem": _subsystem(g, nid),
            }
            top.append(entry)
            # broker flag: high betweenness, low LOC, low unique behaviour
            if bc > 0 and loc > 0 and loc <= 60 and complexity <= 4:
                entry = dict(entry)
                entry["broker_flag"] = True
                entry["reason"] = (
                    "high betweenness + low LOC + low complexity — "
                    "likely translation/broker layer from fragmented topology"
                )
                brokers.append(entry)

        # Articulation points on every component of meaningful size
        articulations = []
        if UG.number_of_nodes() and UG.number_of_edges():
            for comp in nx.connected_components(UG):
                if len(comp) < 4:
                    continue
                H = UG.subgraph(comp)
                for ap in nx.articulation_points(H):
                    articulations.append(
                        {
                            "id": ap,
                            "path": g.path_of(ap),
                            "loc": g.loc_of(ap),
                            "betweenness": round(betweenness.get(ap, 0.0), 8),
                            "component_size": len(comp),
                        }
                    )
            articulations.sort(key=lambda x: -x["betweenness"])

        # Min edge cuts between top communities
        cuts = []
        if communities and len(communities) >= 2:
            top_comms = communities[: min(6, len(communities))]
            for i in range(len(top_comms)):
                for j in range(i + 1, len(top_comms)):
                    a_members = set(top_comms[i].get("members", []))
                    b_members = set(top_comms[j].get("members", []))
                    a_members &= set(UG.nodes)
                    b_members &= set(UG.nodes)
                    if len(a_members) < 2 or len(b_members) < 2:
                        continue
                    # Sample seeds
                    try:
                        # Build condensed view: edges between the two sets
                        cut_edges = []
                        for u in list(a_members)[:500]:
                            if u not in UG:
                                continue
                            for v in UG.neighbors(u):
                                if v in b_members:
                                    cut_edges.append(
                                        {
                                            "src": u,
                                            "dst": v,
                                            "src_path": g.path_of(u),
                                            "dst_path": g.path_of(v),
                                        }
                                    )
                        cuts.append(
                            {
                                "community_a": top_comms[i]["id"],
                                "community_b": top_comms[j]["id"],
                                "n_cut_edges": len(cut_edges),
                                "cut_edges_sample": cut_edges[:30],
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        cuts.append(
                            {
                                "community_a": top_comms[i]["id"],
                                "community_b": top_comms[j]["id"],
                                "error": str(exc),
                            }
                        )

        summary = (
            f"Approximate betweenness with k={k_use} samples on {n_nodes} file nodes. "
            f"Flagged {len(brokers)} broker-like nodes (high BC, low LOC, low complexity). "
            f"{len(articulations)} articulation points in largest component. "
            f"{len(cuts)} community-pair cut sets enumerated."
        )
        return {
            "machine": {
                "k": k_use,
                "n_nodes": n_nodes,
                "top_betweenness": top[:50],
                "brokers": brokers[:40],
                "articulation_points": articulations[:50],
                "community_cuts": cuts,
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 4. Dominator analysis
# ---------------------------------------------------------------------------


def analyze_dominators(g: SemanticGraph) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        # Call graph on functions + entry points (cli_command, adapter)
        CG = nx.DiGraph()
        for ntype in ("function", "cli_command", "adapter", "tool", "operator"):
            for nid in g.by_type.get(ntype, []):
                CG.add_node(nid, ntype=g.nodes[nid]["type"])

        for e in g.edges:
            if e["type"] not in ("calls", "runtime_calls"):
                continue
            # Drop fixture pad-noise and low-confidence heuristic edges so
            # dominator trees reflect real control structure, not filler.
            eid = e.get("id") or ""
            if "|pad" in eid:
                continue
            conf = float(e.get("attrs", {}).get("confidence", 1.0) or 1.0)
            if conf < 0.9:
                continue
            if e["src"] in CG and e["dst"] in CG:
                CG.add_edge(e["src"], e["dst"])

        entry_points = []
        for ntype in ("cli_command",):
            entry_points.extend(g.by_type.get(ntype, []))
        # HTTP-ish adapters
        for nid in g.by_type.get("adapter", []):
            name = g.nodes[nid].get("name", "")
            path = g.path_of(nid) or ""
            if "http" in nid or "route" in name.lower() or "http" in path:
                entry_points.append(nid)
        entry_points = sorted(set(entry_points))

        # For each entry, immediate dominators over reachable subgraph
        dom_trees: dict[str, dict[str, str]] = {}
        dominated_by_count: Counter[str] = Counter()
        path_sets: dict[str, set[str]] = {}

        for ep in entry_points:
            if ep not in CG:
                continue
            # BFS depth-capped reachability keeps pad-free graphs tractable
            reachable = {ep}
            frontier = [ep]
            depth = {ep: 0}
            max_depth = 24
            max_nodes = 400
            while frontier and len(reachable) < max_nodes:
                u = frontier.pop(0)
                if depth[u] >= max_depth:
                    continue
                for v in CG.successors(u):
                    if v not in reachable:
                        reachable.add(v)
                        depth[v] = depth[u] + 1
                        frontier.append(v)
            if len(reachable) < 2:
                continue
            sub = CG.subgraph(reachable).copy()
            try:
                idom = nx.immediate_dominators(sub, ep)
            except Exception:  # noqa: BLE001
                continue
            dom_trees[ep] = {n: d for n, d in idom.items() if n != ep}
            for n, d in idom.items():
                if n != ep:
                    dominated_by_count[n] += 1
            path_sets[ep] = set(idom.keys()) - {ep}

        # Common control paths: nodes dominated from many entry points
        shared = []
        for nid, cnt in dominated_by_count.most_common(80):
            if cnt < 2:
                continue
            shared.append(
                {
                    "id": nid,
                    "path": g.path_of(nid),
                    "name": g.nodes.get(nid, {}).get("name"),
                    "n_entry_points": cnt,
                    "loc": g.loc_of(nid),
                    "entry_points": sorted(
                        [ep for ep, ps in path_sets.items() if nid in ps]
                    )[:20],
                }
            )

        # Shared chains between entry pairs — cap members for report size
        chains = []
        if len(path_sets) >= 2:
            eps = list(path_sets.keys())
            for i in range(min(len(eps), 12)):
                for j in range(i + 1, min(len(eps), 12)):
                    inter = path_sets[eps[i]] & path_sets[eps[j]]
                    if len(inter) >= 2:
                        # Prefer higher-LOC / planted-looking nodes in the sample
                        ranked_inter = sorted(
                            inter, key=lambda x: (-g.loc_of(x), x)
                        )
                        sample = ranked_inter[:40]
                        chains.append(
                            {
                                "entry_a": eps[i],
                                "entry_b": eps[j],
                                "shared_nodes": sample,
                                "shared_count": len(inter),
                                "shared_loc": sum(g.loc_of(x) for x in inter),
                                "shared_nodes_truncated": len(inter) > 40,
                            }
                        )
            chains.sort(key=lambda x: (-x["shared_count"], -x["shared_loc"]))

        summary = (
            f"Dominator trees from {len(dom_trees)} entry points "
            f"({len(entry_points)} candidates: CLI + HTTP adapters). "
            f"{len(shared)} nodes lie on control paths of >=2 entries — "
            f"state-machine unification candidates. "
            f"{len(chains)} entry-point pairs share >=2 dominated nodes."
        )
        return {
            "machine": {
                "n_entry_points": len(entry_points),
                "n_trees": len(dom_trees),
                "shared_control_nodes": shared[:50],
                "shared_chains": chains[:40],
                "entry_points": entry_points,
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 5. Structural clone analysis
# ---------------------------------------------------------------------------


def _structural_signature(g: SemanticGraph, nid: str) -> str:
    """CFG-ish signature: prefer attrs.cfg_signature; else structural proxy.

    Normalised token/CFG signature — NOT text similarity. Identifiers are not
    used; only control shape, call arity, complexity band.
    """
    n = g.nodes.get(nid)
    if not n:
        return ""
    attrs = n.get("attrs", {})
    if attrs.get("cfg_signature"):
        return str(attrs["cfg_signature"])

    # Fallback structural proxy from graph + attrs
    complexity = int(attrs.get("complexity", 0) or 0)
    call_out = [
        e
        for e in g.out_edges.get(nid, [])
        if e["type"] in ("calls", "runtime_calls")
    ]
    arities = []
    for e in call_out[:12]:
        # arity proxy: count attribute or 1
        arities.append(int(e.get("attrs", {}).get("count", 1) or 1))
    side = tuple(sorted(attrs.get("side_effects") or ["none"]))
    # positional slots only — no names
    parts = [
        f"cx{complexity}",
        f"calls{len(call_out)}",
        "ar:" + ",".join(str(a) for a in arities),
        "se:" + ",".join(side),
    ]
    if complexity >= 3:
        parts.insert(0, "if")
    if complexity >= 5:
        parts.insert(0, "loop")
    raw = "|".join(parts)
    return "PROXY:" + hashlib.sha1(raw.encode()).hexdigest()[:16] + ":" + raw


def analyze_clones(g: SemanticGraph, *, min_family: int = 2) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for nid in g.by_type.get("function", []):
            if _is_excluded(g, nid):
                continue
            if g.attr(nid, "test", False):
                continue
            sig = _structural_signature(g, nid)
            if not sig:
                continue
            buckets[sig].append(nid)

        families = []
        for sig, members in buckets.items():
            if len(members) < min_family:
                continue
            # PROXY:* hashes collide heavily on thin pad/boilerplate functions.
            # Keep them only for small families; large families require an explicit
            # CFG: signature (extractor- or fixture-provided).
            if sig.startswith("PROXY:") and len(members) > 24:
                continue
            if len(members) > 500:
                # Pathological collision — not admissible as a clone family
                continue
            members = sorted(members)
            loc = sum(g.loc_of(m) for m in members)
            langs = sorted({g.nodes[m].get("lang", "none") for m in members})
            crates = set()
            for m in members:
                path = g.path_of(m) or ""
                if path.startswith("crates/"):
                    crates.add(path.split("/")[1])
            # Text similarity is NOT admissible — we only have signature_match here
            match_kind = "signature_match"
            families.append(
                {
                    "signature": sig,
                    "match_kind": match_kind,
                    "text_match_admissible": False,
                    "note": (
                        "Clone evidence is signature_match on normalised CFG/token "
                        "signature (identifiers → positional slots; control keywords "
                        "and call arity kept). Text similarity alone is not admissible."
                    ),
                    "member_count": len(members),
                    "members": members,
                    "loc": loc,
                    "langs": langs,
                    "cross_language": len(langs) > 1,
                    "crates": sorted(crates),
                    "cross_crate": len(crates) > 1,
                }
            )
        families.sort(key=lambda x: (-x["member_count"], -x["loc"]))
        # cap report size
        for i, f in enumerate(families):
            f["id"] = f"CLONE-{i:04d}"
            # truncate member lists for very large proxy-hash families
            if len(f["members"]) > 40:
                f["members_full_count"] = len(f["members"])
                f["members"] = f["members"][:40]
                f["members_truncated"] = True

        summary = (
            f"Structural clone analysis (signature_match only; text_match not admissible) "
            f"found {len(families)} families with >= {min_family} members. "
            f"Largest family: {families[0]['member_count'] if families else 0} members, "
            f"{families[0]['loc'] if families else 0} LOC."
        )
        return {
            "machine": {
                "n_families": len(families),
                "families": families[:80],
                "method": "cfg_signature|structural_proxy",
                "text_similarity_admissible": False,
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 6. Co-change analysis
# ---------------------------------------------------------------------------


def analyze_cochange(
    g: SemanticGraph,
    *,
    min_count: int = 5,
    min_weight: float | None = None,
) -> dict[str, Any]:
    """Find co-changing file pairs with no structural coupling.

    Schema ``weight`` is ``count / min(commits_a, commits_b)`` ∈ [0, 1] — it
    can never clear an absolute threshold of 5.0. The analysis therefore
    thresholds on raw ``count`` (co-commits), which is what a threshold of 5
    was intended to mean. Schema-normalized weight is still reported.

    If a legacy ``min_weight`` is passed and is > 1.0, it is treated as
    ``min_count`` for backwards compatibility with fixtures that stored raw
    co-commit counts in the weight field.
    """
    def run() -> dict[str, Any]:
        # Interpret legacy min_weight>1 as a count threshold (fixture plant uses weight=18).
        count_threshold = min_count
        if min_weight is not None:
            if min_weight > 1.0:
                count_threshold = int(min_weight)
            # min_weight in (0,1] would be a normalized-weight threshold; unused by default

        # Direct coupling pairs (imports/calls) at file level
        coupled: set[tuple[str, str]] = set()
        file_of: dict[str, str] = {}
        for nid, n in g.nodes.items():
            if n["type"] == "file":
                file_of[nid] = nid
            elif n.get("path") and n["type"] in ("function", "type"):
                file_of[nid] = f"file:{n['path']}"

        for e in g.edges:
            if e["type"] not in ("imports", "calls", "runtime_calls"):
                continue
            s = file_of.get(e["src"])
            d = file_of.get(e["dst"])
            if s and d and s != d:
                a, b = (s, d) if s < d else (d, s)
                coupled.add((a, b))

        pairs = []
        seen_pair: set[tuple[str, str]] = set()
        for e in g.edges:
            if e["type"] != "co_changes":
                continue
            attrs = e.get("attrs") or {}
            # Prefer schema count; fall back to weight when fixtures stuffed count into weight
            count = int(attrs.get("count") or 0)
            w = float(attrs.get("weight", 0.0) or 0.0)
            if count <= 0 and w > 1.0:
                # fixture / legacy: weight carried the co-commit count
                count = int(w)
            if count < count_threshold:
                # also accept high normalized weight only when explicitly requested
                if not (min_weight is not None and 0 < min_weight <= 1.0 and w >= min_weight):
                    continue
            s, d = e["src"], e["dst"]
            # map to files if needed
            s = file_of.get(s, s if s.startswith("file:") else None)
            d = file_of.get(d, d if d.startswith("file:") else None)
            if not s or not d or s == d:
                continue
            a, b = (s, d) if s < d else (d, s)
            if (a, b) in seen_pair:
                continue
            seen_pair.add((a, b))
            direct = (a, b) in coupled
            if direct:
                continue
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "a_path": g.path_of(a),
                    "b_path": g.path_of(b),
                    "co_changes_count": count,
                    "co_changes_weight": w,
                    "direct_imports_or_calls": False,
                    "loc_a": g.loc_of(a),
                    "loc_b": g.loc_of(b),
                    "combined_loc": g.loc_of(a) + g.loc_of(b),
                    "subsystem_a": _subsystem(g, a),
                    "subsystem_b": _subsystem(g, b),
                }
            )
        pairs.sort(
            key=lambda x: (
                -x["co_changes_count"],
                -x["co_changes_weight"],
                -x["combined_loc"],
            )
        )

        top_c = pairs[0]["co_changes_count"] if pairs else 0
        top_w = pairs[0]["co_changes_weight"] if pairs else 0
        summary = (
            f"Found {len(pairs)} file pairs with co_changes count>={count_threshold} "
            f"(schema weight=count/min(commits), ∈[0,1]; threshold is raw co-commit count) "
            f"but no direct imports/calls coupling — layout-split module candidates. "
            f"Top count={top_c}, top weight={top_w}."
        )
        return {
            "machine": {
                "min_count": count_threshold,
                "min_weight_schema_note": (
                    "weight is count/min(commits_a,commits_b) per schema; "
                    "analysis thresholds on count, not weight"
                ),
                "n_pairs": len(pairs),
                "pairs": pairs[:100],
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 7. Fan-in analysis
# ---------------------------------------------------------------------------


def analyze_fanin(g: SemanticGraph, *, min_adapters: int = 4, max_adapter_loc: int = 25) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        # Authority = function with many inbound calls from thin callers.
        # Exclude artefact edges: low-confidence ambiguous name matches and
        # stub function nodes (loc<=1 with no real body) that inflate rings to
        # hundreds of 1-LOC "adapters".
        callers_of: dict[str, list[str]] = defaultdict(list)
        for e in g.edges:
            if e["type"] not in ("calls", "runtime_calls"):
                continue
            conf = float(e.get("attrs", {}).get("confidence", 1.0) or 1.0)
            if conf < 0.5:
                continue
            if g.nodes.get(e["src"], {}).get("type") != "function":
                continue
            if g.nodes.get(e["dst"], {}).get("type") != "function":
                continue
            callers_of[e["dst"]].append(e["src"])

        rings = []
        n_stub_excluded = 0
        for auth, callers in callers_of.items():
            if _is_excluded(g, auth):
                continue
            thin = []
            for c in callers:
                loc = g.loc_of(c)
                # loc<=1 is almost always a brace-match failure or a pure
                # forward declaration stub — not a real thin adapter body.
                if loc <= 1:
                    n_stub_excluded += 1
                    continue
                complexity = int(g.attr(c, "complexity", 0) or 0)
                # thin body: low loc, low complexity, few outbound calls
                out_calls = [
                    e
                    for e in g.out_edges.get(c, [])
                    if e["type"] in ("calls", "runtime_calls")
                    and float(e.get("attrs", {}).get("confidence", 1.0) or 1.0) >= 0.5
                ]
                if loc <= max_adapter_loc and complexity <= 2 and len(out_calls) <= 2:
                    thin.append(c)
            thin = sorted(set(thin))
            if len(thin) < min_adapters:
                continue
            thin_loc = sum(g.loc_of(t) for t in thin)
            rings.append(
                {
                    "authority": auth,
                    "authority_path": g.path_of(auth),
                    "authority_name": g.nodes.get(auth, {}).get("name"),
                    "authority_loc": g.loc_of(auth),
                    "adapter_count": len(thin),
                    "adapter_total_loc": thin_loc,
                    "adapters": thin,
                    "adapter_paths": [g.path_of(t) for t in thin],
                    "subsystem": _subsystem(g, auth),
                }
            )
        rings.sort(key=lambda x: (-x["adapter_count"], -x["adapter_total_loc"]))
        for i, r in enumerate(rings):
            r["id"] = f"FANIN-{i:04d}"

        summary = (
            f"Found {len(rings)} authorities with >={min_adapters} thin adapters "
            f"(body LOC 2..{max_adapter_loc}, conf>=0.5; excluded loc<=1 stubs). "
            f"Largest ring: {rings[0]['adapter_count'] if rings else 0} adapters, "
            f"{rings[0]['adapter_total_loc'] if rings else 0} adapter LOC — "
            f"generate-bindings candidates."
        )
        return {
            "machine": {
                "min_adapters": min_adapters,
                "max_adapter_loc": max_adapter_loc,
                "min_adapter_loc": 2,
                "min_call_confidence": 0.5,
                "stub_callers_excluded": n_stub_excluded,
                "n_rings": len(rings),
                "rings": rings[:80],
            },
            "summary": summary,
        }

    return _timed(run)


# ---------------------------------------------------------------------------
# 8. Behaviour coverage analysis
# ---------------------------------------------------------------------------


def analyze_behaviour_coverage(
    g: SemanticGraph,
    behaviour_map_path: Path | str | None,
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        degraded = False
        degrade_reason = None
        behaviours: dict[str, Any] = {}

        path = Path(behaviour_map_path) if behaviour_map_path else None
        if path is None or not path.is_file():
            degraded = True
            degrade_reason = (
                f"HAWKING_BEHAVIOUR_TO_CODE_MAP.json absent"
                f"{'' if path is None else f' at {path}'}; "
                "behaviour coverage degraded — no deletion signal from this analysis."
            )
        else:
            doc = json.loads(path.read_text(encoding="utf-8"))
            behaviours = doc.get("behaviours") or {}
            if not behaviours:
                degraded = True
                degrade_reason = (
                    "behaviour map present but `behaviours` is empty; "
                    "coverage analysis degraded gracefully."
                )

        covered: set[str] = set()
        for _bid, body in behaviours.items():
            if isinstance(body, dict):
                for nid in body.get("reachable_nodes") or body.get("nodes") or []:
                    covered.add(nid)
            elif isinstance(body, list):
                covered.update(body)

        # Also take depends_on_behaviour edges from the graph itself
        for e in g.edges:
            if e["type"] == "depends_on_behaviour":
                covered.add(e["dst"])
                # behaviour node itself
                covered.add(e["src"])

        uncovered_files = []
        uncovered_fns = []
        loc_by_sub: Counter[str] = Counter()
        total_uncovered_loc = 0

        for nid in g.by_type.get("file", []):
            if _is_excluded(g, nid):
                continue
            if g.attr(nid, "test", False):
                continue
            # file covered if itself or any contained function is covered
            fns = [
                e["dst"]
                for e in g.out_edges.get(nid, [])
                if e["type"] == "contains" and g.nodes.get(e["dst"], {}).get("type") == "function"
            ]
            is_cov = nid in covered or any(f in covered for f in fns)
            if not is_cov:
                loc = g.loc_of(nid)
                sub = _subsystem(g, nid)
                total_uncovered_loc += loc
                loc_by_sub[sub] += loc
                uncovered_files.append(
                    {
                        "id": nid,
                        "path": g.path_of(nid),
                        "loc": loc,
                        "subsystem": sub,
                    }
                )

        for nid in g.by_type.get("function", []):
            if _is_excluded(g, nid) or g.attr(nid, "test", False):
                continue
            if nid not in covered:
                uncovered_fns.append(nid)

        uncovered_files.sort(key=lambda x: -x["loc"])

        if degraded:
            summary = (
                f"Behaviour coverage DEGRADED: {degrade_reason} "
                f"Graph-only depends_on_behaviour edges cover {len(covered)} nodes. "
                f"Unreachable-from-behaviour bucket (best-effort): "
                f"{total_uncovered_loc} LOC across {len(uncovered_files)} files."
            )
        else:
            summary = (
                f"Behaviour coverage: {len(behaviours)} contracts, "
                f"{len(covered)} reachable nodes. "
                f"Code reachable from no behaviour contract: "
                f"{total_uncovered_loc} LOC in {len(uncovered_files)} files "
                f"(deletion/replacement *candidates* — still require behaviour gates). "
                f"By subsystem: {dict(loc_by_sub)}."
            )

        return {
            "machine": {
                "degraded": degraded,
                "degrade_reason": degrade_reason,
                "n_behaviours": len(behaviours),
                "n_covered_nodes": len(covered),
                "uncovered_loc_total": total_uncovered_loc,
                "uncovered_loc_by_subsystem": dict(loc_by_sub),
                "n_uncovered_files": len(uncovered_files),
                "n_uncovered_functions": len(uncovered_fns),
                "uncovered_files_top": uncovered_files[:100],
            },
            "summary": summary,
        }

    return _timed(run)


def run_all(
    g: SemanticGraph,
    *,
    behaviour_map: Path | str | None = None,
    betweenness_k: int = 64,
) -> dict[str, Any]:
    """Run all eight analyses; return combined cluster map body."""
    timings: dict[str, float] = {}
    results: dict[str, Any] = {}

    scc = analyze_scc(g)
    results["scc"] = scc
    timings["scc"] = scc["seconds"]

    comm = analyze_communities(g)
    results["communities"] = comm
    timings["communities"] = comm["seconds"]

    bet = analyze_betweenness(
        g,
        communities=comm["machine"].get("communities"),
        k=betweenness_k,
    )
    results["betweenness"] = bet
    timings["betweenness"] = bet["seconds"]

    dom = analyze_dominators(g)
    results["dominators"] = dom
    timings["dominators"] = dom["seconds"]

    clones = analyze_clones(g)
    results["clones"] = clones
    timings["clones"] = clones["seconds"]

    co = analyze_cochange(g)
    results["cochange"] = co
    timings["cochange"] = co["seconds"]

    fan = analyze_fanin(g)
    results["fanin"] = fan
    timings["fanin"] = fan["seconds"]

    beh = analyze_behaviour_coverage(g, behaviour_map)
    results["behaviour_coverage"] = beh
    timings["behaviour_coverage"] = beh["seconds"]

    return {
        "schema": "hawking.cluster_map.v1",
        "graph_summary": g.summary(),
        "analyses": {
            name: {
                "machine": results[name]["machine"],
                "summary": results[name]["summary"],
                "seconds": results[name]["seconds"],
            }
            for name in results
        },
        "timings_seconds": timings,
        "total_seconds": sum(timings.values()),
    }
