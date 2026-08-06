"""Semantic graph model matching
workspace/campaign/governance/control/catalog/manifests/SEMANTIC_GRAPH_SCHEMA.json exactly.

No node types, edge types, attribute names, or id formats may be added,
renamed, or dropped relative to that contract.
"""

from __future__ import annotations

import json
import re
import xml.sax.saxutils
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


_VENDOR_PATH_PREFIXES = ("vendor/", "workspace/vendor/")
_TEST_PATH_PREFIXES = ("tests/", "workspace/quality/tests/")

NODE_TYPES = frozenset({
    "repository", "crate", "directory", "file", "type", "function",
    "cli_command", "event", "schema", "tool", "operator", "adapter",
    "test", "artifact", "state", "feature_flag", "behaviour",
})

EDGE_TYPES = frozenset({
    "contains", "imports", "calls", "constructs", "implements",
    "reads_state", "writes_state", "serializes", "deserializes",
    "emits", "consumes", "tests", "generates", "duplicates",
    "co_changes", "runtime_calls", "feature_gates",
    "provides_capability", "depends_on_behaviour",
})

LANGS = frozenset({
    "rust", "python", "typescript", "metal", "shell", "lean",
    "markdown", "toml", "none",
})

# Side-effect classification (documented for the extractor module).
# A node gets a side_effects entry when its body text / imports match:
#   fs:    File, std::fs, open(, pathlib, os.path, read_to_string, write!, create_dir
#   net:   reqwest, hyper, TcpStream, UdpSocket, urllib, http.client, socket, axum, warp
#   proc:  Command::, std::process, subprocess, os.system, Popen
#   env:   env::var, std::env, os.environ, os.getenv, env_on(
#   clock: Instant::, SystemTime, chrono::, time::, datetime, time.time
#   rand:  rand::, thread_rng, random., secrets.
#   gpu:   metal::, mtl, wgpu, cuda, dispatch_threads, encode_compute
#   none:  default when nothing matches
SIDE_EFFECT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("fs", re.compile(
        r"\b(?:std::fs|File::|OpenOptions|read_to_string|write_all|create_dir|"
        r"remove_file|rename\(|pathlib|os\.path|open\(|Path\(|aiofiles)\b",
        re.I,
    )),
    ("net", re.compile(
        r"\b(?:reqwest|hyper|TcpStream|UdpSocket|tokio::net|axum|warp|"
        r"urllib|http\.client|aiohttp|socket\.|fetch\()\b",
        re.I,
    )),
    ("proc", re.compile(
        r"\b(?:std::process|Command::|subprocess|Popen|os\.system|os\.exec)\b",
        re.I,
    )),
    ("env", re.compile(
        r"\b(?:std::env|env::var|env::var_os|env_on\(|os\.environ|os\.getenv|"
        r"os\.putenv)\b",
        re.I,
    )),
    ("clock", re.compile(
        r"\b(?:Instant::|SystemTime|chrono::|time::|datetime\.|time\.time|"
        r"time\.sleep|Duration::)\b",
        re.I,
    )),
    ("rand", re.compile(
        r"\b(?:rand::|thread_rng|random\.|secrets\.|StdRng|getrandom)\b",
        re.I,
    )),
    ("gpu", re.compile(
        r"\b(?:metal::|mtl|wgpu|cuda|dispatch_threads|encode_compute|"
        r"new_compute_pipeline|set_compute_pipeline)\b",
        re.I,
    )),
]

SECURITY_RE = re.compile(
    r"(?i)\b(?:permission|capability|allowlist|credential|secret|token|"
    r"sha256|blake3|signature|sandbox|path\s*traversal|unsafe)\b|"
    r"hash\s*verif",
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
        "complexity": 1,
        "side_effects": ["none"],
        "security_sensitive": False,
        "public": False,
        "generated": False,
        "vendored": False,
        "test": False,
        "subsystem": "shared",
    }


def default_edge_attrs() -> dict[str, Any]:
    return {
        "weight": 1.0,
        "count": 1,
        "evidence": "regex",
        "confidence": 0.5,
    }


def classify_path(path: str) -> tuple[bool, bool]:
    """Return (vendored, generated) flags."""
    vendored = path.startswith(_VENDOR_PATH_PREFIXES)
    generated = (
        "/generated/" in path
        or path.endswith(".generated.rs")
        or path.endswith(".generated.ts")
    )
    return vendored, generated


def subsystem_for(path: str | None) -> str:
    if not path:
        return "shared"
    if path.startswith("crates/hide-") or path.startswith("app/"):
        return "hide"
    if path.startswith("crates/"):
        return "hawking"
    if path.startswith("tools/") or path.startswith("ramanujan/"):
        return "laboratory"
    if path.startswith(_VENDOR_PATH_PREFIXES):
        return "vendor"
    return "shared"


def is_test_path(path: str | None) -> bool:
    if not path:
        return False
    return (
        "/tests/" in path
        or path.startswith(_TEST_PATH_PREFIXES)
        or Path(path).name.startswith("test_")
        or Path(path).name.endswith("_test.rs")
        or "/benches/" in path
    )


def lang_for(path: str | None) -> str:
    if not path:
        return "none"
    ext = Path(path).suffix.lower()
    return {
        ".rs": "rust",
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "typescript",
        ".jsx": "typescript",
        ".metal": "metal",
        ".sh": "shell",
        ".lean": "lean",
        ".md": "markdown",
        ".toml": "toml",
    }.get(ext, "none")


def detect_side_effects(text: str) -> list[str]:
    found: list[str] = []
    for name, pat in SIDE_EFFECT_PATTERNS:
        if pat.search(text):
            found.append(name)
    return found or ["none"]


def detect_security(text: str) -> bool:
    return bool(SECURITY_RE.search(text))


def complexity_of(body: str) -> int:
    """1 + branch/loop/match-arm/&&/|| tokens (approx cyclomatic)."""
    if not body:
        return 1
    n = 0
    # Keywords that introduce branches/loops
    n += len(re.findall(
        r"\b(?:if|else\s+if|elif|for|while|loop|match|case|catch|except|"
        r"&&|\|\||\?\s|:|\band\b|\bor\b)\b",
        body,
    ))
    # Explicit tokens
    n += body.count("&&") + body.count("||")
    # Match arms roughly: "=>" in rust
    n += body.count("=>")
    return 1 + n


@dataclass
class Node:
    id: str
    type: str
    name: str
    path: str | None = None
    lang: str = "none"
    span: list[int] | None = None
    attrs: dict[str, Any] = field(default_factory=default_node_attrs)

    def to_dict(self) -> dict[str, Any]:
        attrs = default_node_attrs()
        attrs.update(self.attrs)
        # Ensure required keys always present with correct nulls
        for k in default_node_attrs():
            if k not in attrs:
                attrs[k] = default_node_attrs()[k]
        return {
            "kind": "node",
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "lang": self.lang,
            "span": self.span if self.span is not None else None,
            "attrs": attrs,
        }


@dataclass
class Edge:
    src: str
    type: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=default_edge_attrs)

    @property
    def id(self) -> str:
        return f"{self.src}|{self.type}|{self.dst}"

    def to_dict(self) -> dict[str, Any]:
        attrs = default_edge_attrs()
        attrs.update(self.attrs)
        for k in default_edge_attrs():
            if k not in attrs:
                attrs[k] = default_edge_attrs()[k]
        return {
            "kind": "edge",
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "attrs": attrs,
        }


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}  # keyed by edge.id

    def add_node(self, node: Node) -> Node:
        if node.type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {node.type}")
        existing = self.nodes.get(node.id)
        if existing is not None:
            # Merge attrs conservatively (prefer true/higher counts)
            for k, v in node.attrs.items():
                if k in ("loc", "complexity", "change_freq_90d", "change_freq_all"):
                    existing.attrs[k] = max(int(existing.attrs.get(k) or 0), int(v or 0))
                elif k in ("public", "generated", "vendored", "test",
                           "test_covered", "security_sensitive"):
                    existing.attrs[k] = bool(existing.attrs.get(k)) or bool(v)
                elif k == "runtime_hot":
                    if v is True or existing.attrs.get(k) is True:
                        existing.attrs[k] = True
                    elif existing.attrs.get(k) is None:
                        existing.attrs[k] = v
                elif k == "side_effects":
                    cur = set(existing.attrs.get("side_effects") or ["none"])
                    new = set(v or ["none"])
                    merged = (cur | new) - ({"none"} if (cur | new) - {"none"} else set())
                    existing.attrs["side_effects"] = sorted(merged) if merged else ["none"]
                elif existing.attrs.get(k) in (None, 0, False, [], ["none"]):
                    existing.attrs[k] = v
            if existing.span is None and node.span is not None:
                existing.span = node.span
            if existing.path is None and node.path is not None:
                existing.path = node.path
            return existing
        self.nodes[node.id] = node
        return node

    def add_edge(
        self,
        src: str,
        etype: str,
        dst: str,
        *,
        weight: float = 1.0,
        count: int = 1,
        evidence: str = "regex",
        confidence: float = 0.5,
    ) -> Edge | None:
        if etype not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {etype}")
        if src == dst and etype not in ("co_changes", "duplicates"):
            return None
        eid = f"{src}|{etype}|{dst}"
        existing = self.edges.get(eid)
        if existing is not None:
            existing.attrs["count"] = int(existing.attrs.get("count", 1)) + count
            # keep max confidence / weight
            existing.attrs["weight"] = max(float(existing.attrs.get("weight", 1.0)), weight)
            existing.attrs["confidence"] = max(
                float(existing.attrs.get("confidence", 0.5)), confidence
            )
            return existing
        e = Edge(
            src=src,
            type=etype,
            dst=dst,
            attrs={
                "weight": float(weight),
                "count": int(count),
                "evidence": evidence,
                "confidence": float(confidence),
            },
        )
        self.edges[eid] = e
        return e

    def ensure_contains(self, parent: str, child: str, evidence: str = "ast") -> None:
        self.add_edge(parent, "contains", child, evidence=evidence, confidence=1.0)

    def compute_fan(self) -> None:
        fin: dict[str, int] = defaultdict(int)
        fout: dict[str, int] = defaultdict(int)
        for e in self.edges.values():
            fout[e.src] += 1
            fin[e.dst] += 1
        for nid, n in self.nodes.items():
            n.attrs["fan_in"] = fin.get(nid, 0)
            n.attrs["fan_out"] = fout.get(nid, 0)

    def sorted_records(self) -> list[dict[str, Any]]:
        """JSONL ordering: kind desc (nodes then edges), then type, then id."""
        nodes = sorted(
            (n.to_dict() for n in self.nodes.values()),
            key=lambda r: (r["type"], r["id"]),
        )
        edges = sorted(
            (e.to_dict() for e in self.edges.values()),
            key=lambda r: (r["type"], r["id"]),
        )
        # kind desc: "node" > "edge"
        return nodes + edges

    def emit_jsonl(self, path: Path) -> None:
        records = self.sorted_records()
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in records:
                # compact, stable separators; no spaces after :/, no timestamps
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"),
                                   sort_keys=False))
                f.write("\n")

    def emit_gexf(self, path: Path) -> None:
        """File-level projection: file nodes + edges between files (aggregated)."""
        file_nodes = {
            nid: n for nid, n in self.nodes.items() if n.type == "file"
        }
        # Project non-file edges onto owning file nodes when possible
        file_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        node_file: dict[str, str] = {}
        for nid, n in self.nodes.items():
            if n.type == "file":
                node_file[nid] = nid
            elif n.path:
                node_file[nid] = f"file:{n.path}"

        for e in self.edges.values():
            sf = node_file.get(e.src)
            df = node_file.get(e.dst)
            if not sf or not df or sf not in file_nodes or df not in file_nodes:
                continue
            if sf == df and e.type != "co_changes":
                continue
            key = (sf, e.type, df)
            if key not in file_edges:
                file_edges[key] = {
                    "weight": float(e.attrs.get("weight", 1.0)),
                    "count": int(e.attrs.get("count", 1)),
                }
            else:
                file_edges[key]["count"] += int(e.attrs.get("count", 1))
                file_edges[key]["weight"] = max(
                    file_edges[key]["weight"], float(e.attrs.get("weight", 1.0))
                )

        # Also include co_changes / contains that already link files
        for e in self.edges.values():
            if e.src in file_nodes and e.dst in file_nodes:
                key = (e.src, e.type, e.dst)
                if key not in file_edges:
                    file_edges[key] = {
                        "weight": float(e.attrs.get("weight", 1.0)),
                        "count": int(e.attrs.get("count", 1)),
                    }

        lines: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://www.gexf.net/1.3" version="1.3">',
            '  <meta>',
            '    <creator>hawking_graph</creator>',
            '    <description>Hawking semantic graph file-level projection</description>',
            '  </meta>',
            '  <graph defaultedgetype="directed" mode="static">',
            '    <attributes class="node">',
            '      <attribute id="0" title="loc" type="integer"/>',
            '      <attribute id="1" title="lang" type="string"/>',
            '      <attribute id="2" title="subsystem" type="string"/>',
            '      <attribute id="3" title="fan_in" type="integer"/>',
            '      <attribute id="4" title="fan_out" type="integer"/>',
            '      <attribute id="5" title="change_freq_all" type="integer"/>',
            '      <attribute id="6" title="public" type="boolean"/>',
            '      <attribute id="7" title="test" type="boolean"/>',
            '      <attribute id="8" title="security_sensitive" type="boolean"/>',
            '      <attribute id="9" title="runtime_hot" type="string"/>',
            '    </attributes>',
            '    <attributes class="edge">',
            '      <attribute id="0" title="type" type="string"/>',
            '      <attribute id="1" title="count" type="integer"/>',
            '      <attribute id="2" title="weight" type="float"/>',
            '    </attributes>',
            '    <nodes>',
        ]
        for nid in sorted(file_nodes):
            n = file_nodes[nid]
            a = n.attrs
            rh = a.get("runtime_hot")
            rh_s = "" if rh is None else ("true" if rh else "false")
            name = xml.sax.saxutils.escape(n.name or nid)
            nid_e = xml.sax.saxutils.escape(nid)
            lines.append(f'      <node id="{nid_e}" label="{name}">')
            lines.append("        <attvalues>")
            lines.append(f'          <attvalue for="0" value="{int(a.get("loc", 0))}"/>')
            lines.append(f'          <attvalue for="1" value="{xml.sax.saxutils.escape(n.lang)}"/>')
            lines.append(
                f'          <attvalue for="2" value="{xml.sax.saxutils.escape(str(a.get("subsystem", "shared")))}"/>'
            )
            lines.append(f'          <attvalue for="3" value="{int(a.get("fan_in", 0))}"/>')
            lines.append(f'          <attvalue for="4" value="{int(a.get("fan_out", 0))}"/>')
            lines.append(f'          <attvalue for="5" value="{int(a.get("change_freq_all", 0))}"/>')
            lines.append(f'          <attvalue for="6" value="{str(bool(a.get("public"))).lower()}"/>')
            lines.append(f'          <attvalue for="7" value="{str(bool(a.get("test"))).lower()}"/>')
            lines.append(
                f'          <attvalue for="8" value="{str(bool(a.get("security_sensitive"))).lower()}"/>'
            )
            lines.append(f'          <attvalue for="9" value="{rh_s}"/>')
            lines.append("        </attvalues>")
            lines.append("      </node>")
        lines.append("    </nodes>")
        lines.append("    <edges>")
        for i, ((src, etype, dst), meta) in enumerate(sorted(file_edges.items())):
            src_e = xml.sax.saxutils.escape(src)
            dst_e = xml.sax.saxutils.escape(dst)
            et_e = xml.sax.saxutils.escape(etype)
            lines.append(
                f'      <edge id="{i}" source="{src_e}" target="{dst_e}" weight="{meta["weight"]:.6f}">'
            )
            lines.append("        <attvalues>")
            lines.append(f'          <attvalue for="0" value="{et_e}"/>')
            lines.append(f'          <attvalue for="1" value="{meta["count"]}"/>')
            lines.append(f'          <attvalue for="2" value="{meta["weight"]:.6f}"/>')
            lines.append("        </attvalues>")
            lines.append("      </edge>")
        lines.append("    </edges>")
        lines.append("  </graph>")
        lines.append("</gexf>")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def emit_dot(self, path: Path) -> None:
        """Crate/package-level projection."""
        crates = {
            nid: n for nid, n in self.nodes.items()
            if n.type == "crate"
        }
        # Edges between crates: imports + feature_gates
        crate_edges: list[tuple[str, str, str, float]] = []
        for e in self.edges.values():
            if e.src in crates and e.dst in crates and e.type in (
                "imports", "feature_gates", "depends_on_behaviour"
            ):
                crate_edges.append((
                    e.src, e.type, e.dst, float(e.attrs.get("weight", 1.0))
                ))
        crate_edges.sort(key=lambda t: (t[0], t[1], t[2]))

        lines = [
            "digraph hawking_crates {",
            "  rankdir=LR;",
            '  node [shape=box, fontname="Helvetica"];',
            '  edge [fontname="Helvetica", fontsize=9];',
        ]
        for nid in sorted(crates):
            n = crates[nid]
            label = n.name.replace('"', '\\"')
            sub = n.attrs.get("subsystem", "shared")
            color = {
                "hawking": "lightblue",
                "hide": "lightyellow",
                "laboratory": "lightgreen",
                "vendor": "lightgrey",
            }.get(sub, "white")
            nid_safe = nid.replace('"', '\\"')
            lines.append(
                f'  "{nid_safe}" [label="{label}", style=filled, fillcolor="{color}"];'
            )
        for src, etype, dst, w in crate_edges:
            style = "dashed" if etype == "feature_gates" else "solid"
            src_s = src.replace('"', '\\"')
            dst_s = dst.replace('"', '\\"')
            lines.append(
                f'  "{src_s}" -> "{dst_s}" [label="{etype}", style={style}, weight={w:.2f}];'
            )
        lines.append("}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def emit_behaviour_stub(self, path: Path) -> None:
        stub = {
            "schema": "hawking.behaviour_to_code_map.v1",
            "note": (
                "Stub emitted by G1 extractor (tools/graph). "
                "A separate lane populates behaviours from the behaviour constitution."
            ),
            "behaviours": [],
        }
        path.write_text(
            json.dumps(stub, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def make_node(
    ntype: str,
    nid: str,
    name: str,
    *,
    path: str | None = None,
    lang: str | None = None,
    span: list[int] | None = None,
    **attr_overrides: Any,
) -> Node:
    attrs = default_node_attrs()
    if path:
        vendored, generated = classify_path(path)
        attrs["vendored"] = vendored
        attrs["generated"] = generated
        attrs["subsystem"] = subsystem_for(path)
        attrs["test"] = is_test_path(path)
        if lang is None:
            lang = lang_for(path)
    else:
        if lang is None:
            lang = "none"
    attrs.update(attr_overrides)
    return Node(
        id=nid,
        type=ntype,
        name=name,
        path=path,
        lang=lang or "none",
        span=span,
        attrs=attrs,
    )
