"""Load and save HAWKING_SEMANTIC_GRAPH.jsonl under the frozen schema contract."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

NODE_TYPES = frozenset(
    {
        "repository",
        "crate",
        "directory",
        "file",
        "type",
        "function",
        "cli_command",
        "event",
        "schema",
        "tool",
        "operator",
        "adapter",
        "test",
        "artifact",
        "state",
        "feature_flag",
        "behaviour",
    }
)

EDGE_TYPES = frozenset(
    {
        "contains",
        "imports",
        "calls",
        "constructs",
        "implements",
        "reads_state",
        "writes_state",
        "serializes",
        "deserializes",
        "emits",
        "consumes",
        "tests",
        "generates",
        "duplicates",
        "co_changes",
        "runtime_calls",
        "feature_gates",
        "provides_capability",
        "depends_on_behaviour",
    }
)

COUPLING_EDGE_TYPES = frozenset({"imports", "calls", "runtime_calls"})
COMMUNITY_EDGE_TYPES = frozenset(
    {"calls", "imports", "co_changes", "reads_state", "writes_state", "runtime_calls"}
)


def default_node_attrs() -> dict[str, Any]:
    return {
        "loc": 0,
        "fan_in": 0,
        "fan_out": 0,
        "betweenness": 0.0,
        "runtime_hot": None,
        "compile_cost_ms": None,
        "binary_bytes": None,
        "test_covered": False,
        "change_freq_90d": 0,
        "change_freq_all": 0,
        "complexity": 0,
        "side_effects": ["none"],
        "security_sensitive": False,
        "public": False,
        "generated": False,
        "vendored": False,
        "test": False,
        "subsystem": "shared",
    }


def make_node(
    nid: str,
    ntype: str,
    name: str,
    *,
    path: str | None = None,
    lang: str = "none",
    span: list[int] | None = None,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    a = default_node_attrs()
    if attrs:
        a.update(attrs)
    return {
        "kind": "node",
        "id": nid,
        "type": ntype,
        "name": name,
        "path": path,
        "lang": lang,
        "span": span or [0, 0],
        "attrs": a,
    }


def make_edge(
    src: str,
    etype: str,
    dst: str,
    *,
    weight: float = 1.0,
    count: int = 1,
    evidence: str = "ast",
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "kind": "edge",
        "id": f"{src}|{etype}|{dst}",
        "src": src,
        "dst": dst,
        "type": etype,
        "attrs": {
            "weight": float(weight),
            "count": int(count),
            "evidence": evidence,
            "confidence": float(confidence),
        },
    }


def edge_sort_key(e: dict[str, Any]) -> tuple:
    return (e.get("type", ""), e.get("id", ""))


def node_sort_key(n: dict[str, Any]) -> tuple:
    return (n.get("type", ""), n.get("id", ""))


def sort_graph_records(nodes: list[dict], edges: list[dict]) -> list[dict]:
    """Deterministic ordering: all nodes then all edges, each by (type, id)."""
    nodes_sorted = sorted(nodes, key=node_sort_key)
    edges_sorted = sorted(edges, key=edge_sort_key)
    return nodes_sorted + edges_sorted


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class SemanticGraph:
    """In-memory semantic graph with indexes for analysis."""

    __slots__ = ("nodes", "edges", "out_edges", "in_edges", "by_type", "path")

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.in_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_type: dict[str, list[str]] = defaultdict(list)
        self.path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> "SemanticGraph":
        g = cls()
        g.path = Path(path)
        for rec in iter_jsonl(path):
            kind = rec.get("kind")
            if kind == "node":
                nid = rec["id"]
                g.nodes[nid] = rec
                g.by_type[rec.get("type", "")].append(nid)
            elif kind == "edge":
                g.edges.append(rec)
                g.out_edges[rec["src"]].append(rec)
                g.in_edges[rec["dst"]].append(rec)
        return g

    def loc_of(self, nid: str) -> int:
        n = self.nodes.get(nid)
        if not n:
            return 0
        return int(n.get("attrs", {}).get("loc", 0) or 0)

    def attr(self, nid: str, key: str, default: Any = None) -> Any:
        n = self.nodes.get(nid)
        if not n:
            return default
        return n.get("attrs", {}).get(key, default)

    def path_of(self, nid: str) -> str | None:
        n = self.nodes.get(nid)
        return n.get("path") if n else None

    def edges_of_type(self, *types: str) -> Iterator[dict[str, Any]]:
        wanted = set(types)
        for e in self.edges:
            if e.get("type") in wanted:
                yield e

    def summary(self) -> dict[str, Any]:
        type_counts = {t: len(ids) for t, ids in sorted(self.by_type.items())}
        edge_counts: dict[str, int] = defaultdict(int)
        for e in self.edges:
            edge_counts[e.get("type", "?")] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": type_counts,
            "edge_types": dict(sorted(edge_counts.items())),
        }
