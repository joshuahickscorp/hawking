#!/usr/bin/env python3.12
"""Build a single self-contained HAWKING_GRAPH_VIEWER.html (offline, no CDN).

Inlines vendored Cytoscape.js and a level-of-detail graph payload:
  - crate + directory levels in full
  - top-N files by LOC
  - function level on demand from sibling HAWKING_GRAPH_VIEWER_FUNCTIONS.json

Usage:
    python3.12 tools/graph/viewer/build_viewer.py \\
        --graph build/graph/HAWKING_SEMANTIC_GRAPH.jsonl \\
        --cluster-map build/graph/HAWKING_CLUSTER_MAP.json \\
        --candidates build/graph/HAWKING_RECOMPOSITION_CANDIDATES.json

Writes to build/graph/ (gitignored) by default. The .html fetches its sibling
HAWKING_GRAPH_VIEWER_FUNCTIONS.json at runtime, so keep the three files together.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_GRAPH_DIR = Path(__file__).resolve().parents[1]
if str(_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_DIR))

from graph_io import SemanticGraph  # noqa: E402

VIEWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_DIR.parents[2]


def _crate_of(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("crates/"):
        parts = path.split("/")
        if len(parts) >= 2:
            return f"crate:{parts[1]}"
    return None


def build_lod_payload(
    g: SemanticGraph,
    cluster: dict[str, Any] | None,
    candidates: dict[str, Any] | None,
    *,
    top_files: int = 400,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (inline_payload, functions_by_file)."""

    # Community membership for files
    file_comm: dict[str, str] = {}
    if cluster:
        for comm in (
            cluster.get("analyses", {})
            .get("communities", {})
            .get("machine", {})
            .get("communities", [])
        ):
            cid = comm.get("id", "?")
            for m in comm.get("members") or []:
                file_comm[m] = cid

    scc_files: set[str] = set()
    if cluster:
        for scc in (
            cluster.get("analyses", {})
            .get("scc", {})
            .get("machine", {})
            .get("file_sccs", [])
        ):
            scc_files.update(scc.get("members") or [])

    scc_crates: set[str] = set()
    if cluster:
        for scc in (
            cluster.get("analyses", {})
            .get("scc", {})
            .get("machine", {})
            .get("crate_sccs", [])
        ):
            scc_crates.update(scc.get("members") or [])

    brokers = set()
    if cluster:
        for b in (
            cluster.get("analyses", {})
            .get("betweenness", {})
            .get("machine", {})
            .get("brokers", [])
        ):
            brokers.add(b.get("id"))

    # Dominator edges (parent -> child from shared chains / entry trees)
    dom_edges: list[dict[str, str]] = []
    if cluster:
        for ch in (
            cluster.get("analyses", {})
            .get("dominators", {})
            .get("machine", {})
            .get("shared_chains", [])
        )[:20]:
            nodes = ch.get("shared_nodes") or []
            for i in range(len(nodes) - 1):
                dom_edges.append({"src": nodes[i], "dst": nodes[i + 1]})

    cut_edges: list[dict[str, str]] = []
    if cluster:
        for cut in (
            cluster.get("analyses", {})
            .get("betweenness", {})
            .get("machine", {})
            .get("community_cuts", [])
        )[:10]:
            for e in cut.get("cut_edges_sample") or []:
                cut_edges.append({"src": e["src"], "dst": e["dst"]})

    # Levels
    crates = []
    for cid in sorted(g.by_type.get("crate", [])):
        n = g.nodes[cid]
        crates.append(
            {
                "id": cid,
                "label": n.get("name") or cid,
                "type": "crate",
                "path": n.get("path"),
                "loc": g.loc_of(cid),
                "subsystem": g.attr(cid, "subsystem", "shared"),
                "scc": cid in scc_crates,
            }
        )

    directories = []
    for did in sorted(g.by_type.get("directory", [])):
        n = g.nodes[did]
        directories.append(
            {
                "id": did,
                "label": n.get("name") or did,
                "type": "directory",
                "path": n.get("path"),
                "loc": g.loc_of(did),
                "subsystem": g.attr(did, "subsystem", "shared"),
                "parent": _crate_of(n.get("path")),
            }
        )

    # Files ranked by LOC
    file_nodes = []
    for fid in g.by_type.get("file", []):
        file_nodes.append((g.loc_of(fid), fid))
    file_nodes.sort(reverse=True)
    top = file_nodes[:top_files]
    # Always include planted / SCC / broker files
    must = set(scc_files) | brokers
    for _, fid in file_nodes:
        if fid in must and fid not in {f for _, f in top}:
            top.append((g.loc_of(fid), fid))

    files = []
    for loc, fid in top:
        n = g.nodes[fid]
        files.append(
            {
                "id": fid,
                "label": n.get("name") or fid,
                "type": "file",
                "path": n.get("path"),
                "loc": loc,
                "subsystem": g.attr(fid, "subsystem", "shared"),
                "community": file_comm.get(fid),
                "scc": fid in scc_files,
                "broker": fid in brokers,
                "runtime_hot": bool(g.attr(fid, "runtime_hot", False)),
                "test_covered": bool(g.attr(fid, "test_covered", False)),
                "security_sensitive": bool(g.attr(fid, "security_sensitive", False)),
                "parent_dir": f"dir:{Path(n['path']).parent.as_posix()}" if n.get("path") else None,
                "parent_crate": _crate_of(n.get("path")),
            }
        )
    top_file_ids = {f["id"] for f in files}

    # Crate/dir edges from contains + aggregated coupling
    edges_crate: list[dict] = []
    crate_w: dict[tuple[str, str], float] = defaultdict(float)
    dir_w: dict[tuple[str, str], float] = defaultdict(float)
    file_w: dict[tuple[str, str], float] = defaultdict(float)

    file_to_crate: dict[str, str] = {}
    file_to_dir: dict[str, str] = {}
    for fid in g.by_type.get("file", []):
        path = g.path_of(fid)
        c = _crate_of(path)
        if c:
            file_to_crate[fid] = c
        if path:
            file_to_dir[fid] = f"dir:{Path(path).parent.as_posix()}"

    for e in g.edges:
        et = e["type"]
        if et == "contains":
            s, d = e["src"], e["dst"]
            if s.startswith("crate:") and d.startswith("dir:"):
                edges_crate.append(
                    {"id": e["id"], "src": s, "dst": d, "type": "contains", "level": "crate-dir"}
                )
            continue
        if et not in ("imports", "calls", "runtime_calls", "co_changes"):
            continue
        s, d = e["src"], e["dst"]
        # map to files
        def as_file(x: str) -> str | None:
            if x.startswith("file:"):
                return x
            n = g.nodes.get(x)
            if n and n.get("path") and n["type"] in ("function", "type"):
                return f"file:{n['path']}"
            return None

        sf, df = as_file(s), as_file(d)
        w = float(e.get("attrs", {}).get("weight", 1.0))
        if sf and df and sf != df:
            if sf in top_file_ids and df in top_file_ids:
                a, b = (sf, df) if et != "co_changes" else tuple(sorted((sf, df)))
                file_w[(sf, df)] += w
            cs, cd = file_to_crate.get(sf), file_to_crate.get(df)
            if cs and cd and cs != cd:
                crate_w[(cs, cd)] += w
            ds, dd = file_to_dir.get(sf), file_to_dir.get(df)
            if ds and dd and ds != dd:
                dir_w[(ds, dd)] += w

    crate_edges = [
        {
            "id": f"{s}|agg|{d}",
            "src": s,
            "dst": d,
            "type": "depends",
            "weight": w,
            "level": "crate",
        }
        for (s, d), w in sorted(crate_w.items())
        if w >= 1
    ]
    dir_edges = [
        {
            "id": f"{s}|agg|{d}",
            "src": s,
            "dst": d,
            "type": "depends",
            "weight": w,
            "level": "directory",
        }
        for (s, d), w in sorted(dir_w.items(), key=lambda x: -x[1])
        if w >= 2
    ][:2000]
    cut_set = {(e["src"], e["dst"]) for e in cut_edges}
    file_edges = [
        {
            "id": f"{s}|agg|{d}",
            "src": s,
            "dst": d,
            "type": "depends",
            "weight": w,
            "level": "file",
            "cut": (s, d) in cut_set,
        }
        for (s, d), w in sorted(file_w.items(), key=lambda x: -x[1])
        if w >= 1
    ][:3000]

    # Functions on demand — only for inlined top files (keeps sibling JSON tractable)
    functions_by_file: dict[str, list[dict]] = {}
    for fid in top_file_ids:
        fns = []
        for e in g.out_edges.get(fid, []):
            if e["type"] == "contains":
                dst = e["dst"]
                n = g.nodes.get(dst)
                if n and n["type"] == "function":
                    fns.append(
                        {
                            "id": dst,
                            "label": n.get("name") or dst,
                            "type": "function",
                            "path": n.get("path"),
                            "loc": g.loc_of(dst),
                            "runtime_hot": bool(g.attr(dst, "runtime_hot", False)),
                            "test_covered": bool(g.attr(dst, "test_covered", False)),
                            "security_sensitive": bool(g.attr(dst, "security_sensitive", False)),
                            "cfg_signature": g.attr(dst, "cfg_signature"),
                            "community": file_comm.get(fid),
                        }
                    )
        if fns:
            functions_by_file[fid] = fns

    # Candidate index by member
    cand_list = (candidates or {}).get("candidates") or []
    member_to_rc: dict[str, list[str]] = defaultdict(list)
    for c in cand_list:
        for m in c.get("members") or []:
            member_to_rc[m].append(c["id"])

    # Behaviour coverage uncovered set
    uncovered = set()
    if cluster:
        for f in (
            cluster.get("analyses", {})
            .get("behaviour_coverage", {})
            .get("machine", {})
            .get("uncovered_files_top", [])
        ):
            uncovered.add(f.get("id"))

    for f in files:
        f["uncovered"] = f["id"] in uncovered
        f["rc_ids"] = member_to_rc.get(f["id"], [])

    inline = {
        "schema": "hawking.graph_viewer_payload.v1",
        "levels": {
            "crate": {"nodes": crates, "edges": crate_edges},
            "directory": {"nodes": directories, "edges": dir_edges},
            "file": {"nodes": files, "edges": file_edges},
        },
        "overlays": {
            "dominator_edges": dom_edges,
            "cut_edges": cut_edges,
            "scc_files": sorted(scc_files),
            "scc_crates": sorted(scc_crates),
            "brokers": sorted(x for x in brokers if x),
        },
        "candidates": [
            {
                "id": c["id"],
                "kind": c["kind"],
                "title": c["title"],
                "members": c.get("members"),
                "paths": c.get("paths"),
                "expected_loc_removed": c.get("expected_loc_removed"),
                "risk": c.get("risk"),
                "rank": c.get("rank"),
                "score": c.get("score"),
            }
            for c in cand_list[:200]
        ],
        "meta": {
            "n_nodes_graph": len(g.nodes),
            "n_edges_graph": len(g.edges),
            "top_files": len(files),
            "note": "Function level loads from sibling HAWKING_GRAPH_VIEWER_FUNCTIONS.json",
        },
    }
    return inline, functions_by_file


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Hawking Semantic Graph Viewer</title>
<style>
:root {
  --bg: #f6f7f9;
  --fg: #1a1d23;
  --panel: #ffffff;
  --border: #d0d5dd;
  --accent: #3b6ef5;
  --muted: #667085;
  --danger: #c0392b;
  --ok: #1f7a4c;
  --sel: #f5a524;
}
[data-theme="dark"] {
  --bg: #0f1218;
  --fg: #e8eaed;
  --panel: #1a1f2b;
  --border: #2e3648;
  --accent: #6b8cff;
  --muted: #9aa3b5;
  --danger: #ff6b6b;
  --ok: #3dd68c;
  --sel: #ffb020;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; height: 100%;
  background: var(--bg); color: var(--fg);
  font: 13px/1.4 system-ui, -apple-system, Segoe UI, sans-serif;
  overflow-x: hidden;
}
#app {
  display: grid;
  grid-template-columns: 280px 1fr 300px;
  grid-template-rows: 48px 1fr;
  height: 100vh;
  max-width: 100vw;
}
header {
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 12px;
  padding: 0 14px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
header h1 { font-size: 14px; margin: 0; font-weight: 600; }
header .spacer { flex: 1; }
#search {
  width: min(360px, 40vw);
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--fg);
}
#cy {
  background: var(--bg);
  border-right: 1px solid var(--border);
  min-width: 0;
  overflow: hidden;
}
/* Below ~900px the two fixed panels (280 + 300) leave the 1fr graph column at zero
   width and the page scrolls sideways. Stack instead so the canvas always has room. */
@media (max-width: 900px) {
  #app {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: 48px minmax(240px, 50vh) auto auto;
  }
  /* DOM order is left panel, #cy, right panel -- place explicitly so the canvas takes
     the tall row rather than whichever element happens to come second. */
  #cy { grid-row: 2; border-right: none; border-bottom: 1px solid var(--border); }
  .panel.left { grid-row: 3; }
  .panel.right { grid-row: 4; border-left: none; }
  .panel { max-height: 45vh; border-right: none; }
  .panel * { max-width: 100%; box-sizing: border-box; }
}
.panel {
  background: var(--panel);
  border-right: 1px solid var(--border);
  overflow: auto;
  padding: 12px;
  min-width: 0;
}
.panel.right { border-right: none; border-left: 1px solid var(--border); }
.panel h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin: 12px 0 6px;
}
.panel h2:first-child { margin-top: 0; }
label.row {
  display: flex; align-items: center; gap: 8px;
  margin: 4px 0; cursor: pointer;
}
select, button, input[type="file"] {
  font: inherit; color: var(--fg);
}
select, button {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  cursor: pointer;
}
button.primary { background: var(--accent); color: #fff; border-color: transparent; }
button:hover { filter: brightness(1.05); }
#meta, #sel-info, #node-info {
  font-size: 12px; color: var(--muted);
  word-break: break-word;
}
#sel-list { list-style: none; padding: 0; margin: 0; }
#sel-list li {
  padding: 4px 6px; border-radius: 4px;
  background: var(--bg); margin: 3px 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}
.badge {
  display: inline-block; padding: 1px 6px; border-radius: 999px;
  background: var(--bg); border: 1px solid var(--border); font-size: 11px;
  margin-right: 4px;
}
.kbd {
  font-family: ui-monospace, monospace; font-size: 11px;
  border: 1px solid var(--border); border-radius: 3px; padding: 0 4px;
  background: var(--bg);
}
textarea {
  width: 100%; min-height: 120px; font-family: ui-monospace, monospace;
  font-size: 11px; background: var(--bg); color: var(--fg);
  border: 1px solid var(--border); border-radius: 6px; padding: 8px;
}
.diff-added { color: var(--ok); }
.diff-removed { color: var(--danger); }
.diff-moved { color: var(--sel); }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Hawking Graph Viewer</h1>
    <span class="badge" id="level-badge">crate</span>
    <span class="spacer"></span>
    <input id="search" type="search" placeholder="Search node id  (/)" autocomplete="off"/>
    <button id="theme-btn" type="button" title="Toggle theme">Theme</button>
  </header>

  <aside class="panel left">
    <h2>Level</h2>
    <select id="level-select">
      <option value="crate">crate</option>
      <option value="directory">directory</option>
      <option value="file">file</option>
      <option value="function">function (open a file)</option>
    </select>

    <h2>Overlays</h2>
    <label class="row"><input type="checkbox" id="ov-community" checked/> Community colouring</label>
    <label class="row"><input type="checkbox" id="ov-scc" checked/> SCC highlight</label>
    <label class="row"><input type="checkbox" id="ov-dom"/> Dominator tree</label>
    <label class="row"><input type="checkbox" id="ov-cut"/> Cut-set edges</label>
    <label class="row"><input type="checkbox" id="ov-hot"/> Hot-path (runtime_hot)</label>
    <label class="row"><input type="checkbox" id="ov-cov"/> Test / behaviour coverage</label>
    <label class="row"><input type="checkbox" id="ov-sec"/> Security-sensitive</label>

    <h2>Before / after diff</h2>
    <p style="color:var(--muted);margin:0 0 6px">Load a second payload JSON to diff communities &amp; nodes.</p>
    <input type="file" id="diff-file" accept="application/json,.json"/>
    <div id="diff-summary"></div>

    <h2>Meta</h2>
    <div id="meta"></div>
  </aside>

  <div id="cy"></div>

  <aside class="panel right">
    <h2>Selection (merge candidates)</h2>
    <div id="sel-info">Click nodes to select. Shift-click multi-select.</div>
    <ul id="sel-list"></ul>
    <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap">
      <button type="button" id="sel-clear">Clear</button>
      <button type="button" id="sel-export" class="primary">Export RC entry</button>
    </div>
    <textarea id="rc-out" readonly placeholder="Exported RC JSON appears here"></textarea>

    <h2>Node</h2>
    <div id="node-info">Hover or click a node.</div>

    <h2>Keyboard</h2>
    <div style="color:var(--muted)">
      <span class="kbd">/</span> focus search ·
      <span class="kbd">1</span>–<span class="kbd">4</span> levels ·
      <span class="kbd">Esc</span> clear selection
    </div>
  </aside>
</div>

<script>
/* CYTOSCAPE_JS_INLINE */
</script>
<script>
/* === embedded payload === */
const PAYLOAD = /* PAYLOAD_JSON */;
const FUNCTIONS_BY_FILE = /* FUNCTIONS_JSON */;

const COMM_COLORS = [
  "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
  "#86bcb6","#8cd17d","#b6992d","#499894","#d37295"
];

function hashColor(key) {
  if (!key) return "#888";
  let h = 0;
  for (let i = 0; i < key.length; i++) h = ((h << 5) - h + key.charCodeAt(i)) | 0;
  return COMM_COLORS[Math.abs(h) % COMM_COLORS.length];
}

const state = {
  level: "crate",
  selection: new Map(),
  functionsCache: null,
  openFile: null,
  diff: null,
  theme: (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light",
};

document.documentElement.setAttribute("data-theme", state.theme);

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements: [],
  style: [
    {
      selector: "node",
      style: {
        "label": "data(label)",
        "font-size": 9,
        "color": "data(fg)",
        "text-valign": "center",
        "text-halign": "center",
        "background-color": "data(bg)",
        "border-width": 1,
        "border-color": "data(border)",
        "width": "data(size)",
        "height": "data(size)",
        "text-wrap": "ellipsis",
        "text-max-width": 80,
      }
    },
    {
      selector: "node:selected",
      style: {
        "border-width": 3,
        "border-color": "#f5a524",
        "background-color": "#f5a524",
      }
    },
    {
      selector: "edge",
      style: {
        "width": "data(width)",
        "line-color": "data(line)",
        "target-arrow-color": "data(line)",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        "opacity": 0.7,
        "arrow-scale": 0.7,
      }
    },
    {
      selector: "edge.cut",
      style: { "line-color": "#c0392b", "target-arrow-color": "#c0392b", "width": 3, "opacity": 1 }
    },
    {
      selector: "edge.dom",
      style: { "line-color": "#8e44ad", "target-arrow-color": "#8e44ad", "width": 2, "line-style": "dashed" }
    },
    {
      selector: "node.scc",
      style: { "border-width": 3, "border-color": "#c0392b" }
    },
    {
      selector: "node.hot",
      style: { "border-width": 3, "border-color": "#e67e22" }
    },
    {
      selector: "node.uncovered",
      style: { "opacity": 0.45 }
    },
    {
      selector: "node.secure",
      style: { "border-width": 3, "border-color": "#16a085" }
    },
    {
      selector: "node.diff-added",
      style: { "border-width": 3, "border-color": "#1f7a4c" }
    },
    {
      selector: "node.diff-removed",
      style: { "border-width": 3, "border-color": "#c0392b", "opacity": 0.5 }
    },
    {
      selector: "node.diff-moved",
      style: { "border-width": 3, "border-color": "#f5a524" }
    },
  ],
  layout: { name: "grid" },
  wheelSensitivity: 0.2,
  minZoom: 0.05,
  maxZoom: 4,
});

function themeFg() {
  return state.theme === "dark" ? "#e8eaed" : "#1a1d23";
}

function nodeStyle(n, overlays) {
  const bg = overlays.community && n.community ? hashColor(n.community) : "#7f8c9b";
  const size = Math.max(18, Math.min(64, 14 + Math.sqrt(n.loc || 1) * 1.8));
  return {
    data: {
      id: n.id,
      label: n.label || n.id,
      loc: n.loc || 0,
      type: n.type,
      path: n.path || "",
      community: n.community || "",
      subsystem: n.subsystem || "",
      bg,
      border: "#333",
      fg: themeFg(),
      size,
      raw: n,
    },
    classes: [
      n.scc && overlays.scc ? "scc" : "",
      n.runtime_hot && overlays.hot ? "hot" : "",
      n.uncovered && overlays.cov ? "uncovered" : "",
      n.security_sensitive && overlays.sec ? "secure" : "",
      n.broker ? "scc" : "",
    ].filter(Boolean).join(" "),
  };
}

function currentOverlays() {
  return {
    community: document.getElementById("ov-community").checked,
    scc: document.getElementById("ov-scc").checked,
    dom: document.getElementById("ov-dom").checked,
    cut: document.getElementById("ov-cut").checked,
    hot: document.getElementById("ov-hot").checked,
    cov: document.getElementById("ov-cov").checked,
    sec: document.getElementById("ov-sec").checked,
  };
}

function loadLevel(level) {
  state.level = level;
  document.getElementById("level-badge").textContent = level;
  document.getElementById("level-select").value = level === "function" ? "function" : level;
  const overlays = currentOverlays();
  const elements = [];

  if (level === "function") {
    const fileId = state.openFile;
    const fns = (state.functionsCache && state.functionsCache[fileId]) || [];
    for (const n of fns) elements.push(nodeStyle({ ...n, community: n.community }, overlays));
    // call edges among loaded functions if present in payload overlays — skip heavy
  } else {
    const pack = PAYLOAD.levels[level];
    if (!pack) return;
    for (const n of pack.nodes) elements.push(nodeStyle(n, overlays));
    for (const e of pack.edges) {
      const isCut = overlays.cut && e.cut;
      elements.push({
        data: {
          id: e.id,
          source: e.src,
          target: e.dst,
          width: Math.max(1, Math.min(6, (e.weight || 1) / 5)),
          line: isCut ? "#c0392b" : (state.theme === "dark" ? "#4a5568" : "#b0b7c3"),
        },
        classes: isCut ? "cut" : "",
      });
    }
  }

  if (overlays.dom && PAYLOAD.overlays && PAYLOAD.overlays.dominator_edges) {
    for (const e of PAYLOAD.overlays.dominator_edges) {
      if (cy.getElementById(e.src).nonempty?.() || elements.some(x => x.data && x.data.id === e.src)) {
        elements.push({
          data: {
            id: "dom|" + e.src + "|" + e.dst,
            source: e.src,
            target: e.dst,
            width: 2,
            line: "#8e44ad",
          },
          classes: "dom",
        });
      }
    }
  }

  // Diff classes
  if (state.diff) {
    for (const el of elements) {
      if (!el.data || !el.data.id) continue;
      if (state.diff.added.has(el.data.id)) el.classes = (el.classes || "") + " diff-added";
      if (state.diff.removed.has(el.data.id)) el.classes = (el.classes || "") + " diff-removed";
      if (state.diff.moved.has(el.data.id)) el.classes = (el.classes || "") + " diff-moved";
    }
    // show removed as ghost nodes
    for (const id of state.diff.removed) {
      if (!elements.some(e => e.data && e.data.id === id)) {
        elements.push({
          data: {
            id, label: id, bg: "#555", border: "#c0392b", fg: themeFg(), size: 20, loc: 0,
          },
          classes: "diff-removed",
        });
      }
    }
  }

  cy.elements().remove();
  cy.add(elements.filter(e => {
    // drop edges whose endpoints missing
    if (e.data && e.data.source) {
      const ids = new Set(elements.filter(x => x.data && !x.data.source).map(x => x.data.id));
      return ids.has(e.data.source) && ids.has(e.data.target);
    }
    return true;
  }));

  const layoutName = level === "crate" ? "circle" : (level === "directory" ? "concentric" : "cose");
  cy.layout({
    name: layoutName,
    animate: false,
    padding: 24,
    nodeOverlap: 12,
    componentSpacing: 40,
    idealEdgeLength: 60,
    nestingFactor: 1.2,
  }).run();

  document.getElementById("meta").innerHTML =
    `Graph: ${PAYLOAD.meta.n_nodes_graph} nodes / ${PAYLOAD.meta.n_edges_graph} edges<br/>` +
    `View: ${cy.nodes().length} nodes / ${cy.edges().length} edges<br/>` +
    `Top files inlined: ${PAYLOAD.meta.top_files}<br/>` +
    `<span style="opacity:.8">${PAYLOAD.meta.note}</span>`;
}

function updateSelectionUI() {
  const list = document.getElementById("sel-list");
  list.innerHTML = "";
  let loc = 0;
  const ids = [];
  for (const [id, n] of state.selection) {
    loc += n.loc || 0;
    ids.push(id);
    const li = document.createElement("li");
    li.textContent = `${id} (${n.loc || 0} LOC)`;
    list.appendChild(li);
  }
  // Match candidate
  let match = null;
  for (const c of PAYLOAD.candidates || []) {
    const ms = new Set(c.members || []);
    if (ids.length && ids.every(i => ms.has(i)) && Math.abs(ms.size - ids.length) <= 2) {
      match = c;
      break;
    }
  }
  document.getElementById("sel-info").innerHTML =
    `${state.selection.size} nodes · <strong>${loc}</strong> LOC` +
    (match ? ` · matches <strong>${match.id}</strong> rank ${match.rank} (${match.kind})` : " · no exact RC match");
}

cy.on("tap", "node", (evt) => {
  const n = evt.target;
  const d = n.data();
  const info = document.getElementById("node-info");
  info.innerHTML =
    `<div><strong>${d.id}</strong></div>` +
    `<div>type=${d.type} loc=${d.loc}</div>` +
    `<div>path=${d.path || "—"}</div>` +
    `<div>community=${d.community || "—"} subsystem=${d.subsystem || "—"}</div>`;

  if (evt.originalEvent && evt.originalEvent.shiftKey) {
    if (state.selection.has(d.id)) state.selection.delete(d.id);
    else state.selection.set(d.id, d);
  } else {
    state.selection.clear();
    state.selection.set(d.id, d);
  }
  updateSelectionUI();

  // double purpose: opening file for function level
  if (d.type === "file") {
    state.openFile = d.id;
  }
});

document.getElementById("level-select").addEventListener("change", async (e) => {
  const v = e.target.value;
  if (v === "function") {
    if (!state.openFile) {
      alert("Select a file node first, then switch to function level.");
      e.target.value = state.level;
      return;
    }
    if (!state.functionsCache) {
      // Prefer inlined offline map; fall back to sibling JSON (http servers).
      if (FUNCTIONS_BY_FILE && Object.keys(FUNCTIONS_BY_FILE).length) {
        state.functionsCache = FUNCTIONS_BY_FILE;
      } else {
        try {
          const url = new URL("HAWKING_GRAPH_VIEWER_FUNCTIONS.json", window.location.href);
          const resp = await fetch(url);
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          state.functionsCache = await resp.json();
        } catch (err) {
          console.warn(err);
          alert("Function-level data unavailable offline. Re-run build_viewer.py.");
          state.functionsCache = {};
        }
      }
    }
  }
  loadLevel(v);
});

["ov-community","ov-scc","ov-dom","ov-cut","ov-hot","ov-cov","ov-sec"].forEach(id => {
  document.getElementById(id).addEventListener("change", () => loadLevel(state.level));
});

document.getElementById("sel-clear").addEventListener("click", () => {
  state.selection.clear();
  cy.nodes().unselect();
  updateSelectionUI();
});

document.getElementById("sel-export").addEventListener("click", () => {
  const members = [...state.selection.keys()];
  const paths = [...state.selection.values()].map(n => n.path).filter(Boolean);
  const loc = [...state.selection.values()].reduce((a, n) => a + (n.loc || 0), 0);
  const rc = {
    id: "RC-export-local",
    kind: "merge",
    title: "Viewer selection export (" + members.length + " nodes)",
    members,
    paths,
    evidence: ["viewer manual selection"],
    expected_loc_removed: null,
    expected_dirs_removed: 0,
    expected_files_removed: 0,
    expected_functions_removed: 0,
    behaviour_contracts_touched: [],
    risk: "medium",
    risk_reason: "Manual selection — estimate unset; fill after analysis review",
    test_plan: "Define behaviour gates for selected members before applying",
    rollback: "checkpoint tag before change",
    blocked_by: [],
    selection_loc_sum: loc,
    note: "Proposal only. expected_loc_removed left null until retention fraction is justified.",
  };
  document.getElementById("rc-out").value = JSON.stringify(rc, null, 2);
});

document.getElementById("theme-btn").addEventListener("click", () => {
  state.theme = state.theme === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", state.theme);
  loadLevel(state.level);
});

document.getElementById("search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const q = e.target.value.trim();
    if (!q) return;
    const n = cy.getElementById(q);
    if (n.nonempty()) {
      cy.animate({ center: { eles: n }, zoom: 1.5 }, { duration: 250 });
      n.select();
      state.selection.clear();
      state.selection.set(q, n.data());
      updateSelectionUI();
    } else {
      // substring search
      const hit = cy.nodes().filter(x => x.id().includes(q) || (x.data("label") || "").includes(q));
      if (hit.nonempty()) {
        cy.animate({ fit: { eles: hit, padding: 40 } }, { duration: 250 });
        hit.select();
      }
    }
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
    e.preventDefault();
    document.getElementById("search").focus();
  }
  if (e.key === "Escape") {
    state.selection.clear();
    cy.nodes().unselect();
    updateSelectionUI();
  }
  if (e.key >= "1" && e.key <= "4" && document.activeElement.tagName !== "INPUT") {
    const levels = ["crate", "directory", "file", "function"];
    const lv = levels[+e.key - 1];
    document.getElementById("level-select").value = lv;
    document.getElementById("level-select").dispatchEvent(new Event("change"));
  }
});

document.getElementById("diff-file").addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const text = await file.text();
  const other = JSON.parse(text);
  // Build id sets at current level
  const level = state.level === "function" ? "file" : state.level;
  const aIds = new Set((PAYLOAD.levels[level] && PAYLOAD.levels[level].nodes || []).map(n => n.id));
  const bIds = new Set((other.levels && other.levels[level] && other.levels[level].nodes || []).map(n => n.id));
  const added = new Set([...bIds].filter(x => !aIds.has(x)));
  const removed = new Set([...aIds].filter(x => !bIds.has(x)));
  // community moves
  const aComm = new Map((PAYLOAD.levels[level] && PAYLOAD.levels[level].nodes || []).map(n => [n.id, n.community]));
  const bComm = new Map((other.levels && other.levels[level] && other.levels[level].nodes || []).map(n => [n.id, n.community]));
  const moved = new Set();
  for (const id of aIds) {
    if (bIds.has(id) && aComm.get(id) && bComm.get(id) && aComm.get(id) !== bComm.get(id)) {
      moved.add(id);
    }
  }
  state.diff = { added, removed, moved };
  document.getElementById("diff-summary").innerHTML =
    `<div class="diff-added">+ added: ${added.size}</div>` +
    `<div class="diff-removed">− removed: ${removed.size}</div>` +
    `<div class="diff-moved">↔ moved communities: ${moved.size}</div>`;
  loadLevel(state.level);
});

// init
loadLevel("crate");
updateSelectionUI();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cluster-map", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[3] / "build" / "graph" / "HAWKING_GRAPH_VIEWER.html"),
        help="Default: build/graph/ (gitignored). The functions and payload files are "
             "written beside it and the page fetches them at runtime, so they must stay siblings.",
    )
    p.add_argument("--functions-out", default=None, help="Default: sibling of --out")
    p.add_argument("--top-files", type=int, default=400)
    p.add_argument(
        "--cytoscape",
        default=None,
        help="Path to cytoscape.min.js (default: search common locations)",
    )
    args = p.parse_args(argv)

    cy_path = None
    search = [
        Path(args.cytoscape) if args.cytoscape else None,
        VIEWER_DIR / "cytoscape.min.js",
        Path("/tmp/cytoscape.min.js"),
        REPO_ROOT / "tools/graph/viewer/cytoscape.min.js",
    ]
    for c in search:
        if c and c.is_file():
            cy_path = c
            break
    if cy_path is None:
        print(
            "error: cytoscape.min.js not found. Fetch once and pass --cytoscape, "
            "or place it in tools/graph/viewer/",
            file=sys.stderr,
        )
        print(
            "Falling back to a minimal canvas renderer is not implemented in this build; "
            "refusing to ship a network-dependent page.",
            file=sys.stderr,
        )
        return 2

    cy_js = cy_path.read_text(encoding="utf-8", errors="replace")
    # vendor copy into viewer dir for reproducibility
    vendor_dest = VIEWER_DIR / "cytoscape.min.js"
    if not vendor_dest.is_file():
        vendor_dest.write_text(cy_js, encoding="utf-8")

    print(f"loading graph {args.graph} …", flush=True)
    g = SemanticGraph.load(args.graph)
    cluster = None
    if args.cluster_map and Path(args.cluster_map).is_file():
        cluster = json.loads(Path(args.cluster_map).read_text(encoding="utf-8"))
    candidates = None
    if args.candidates and Path(args.candidates).is_file():
        candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))

    print("building LOD payload …", flush=True)
    inline, functions_by_file = build_lod_payload(
        g, cluster, candidates, top_files=args.top_files
    )

    payload_json = json.dumps(inline, separators=(",", ":"))
    functions_json = json.dumps(functions_by_file, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("/* CYTOSCAPE_JS_INLINE */", cy_js)
    html = html.replace("/* PAYLOAD_JSON */", payload_json)
    html = html.replace("/* FUNCTIONS_JSON */", functions_json)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)", flush=True)

    fn_out = Path(args.functions_out) if args.functions_out else out.with_name(
        "HAWKING_GRAPH_VIEWER_FUNCTIONS.json"
    )
    fn_out.write_text(functions_json + "\n", encoding="utf-8")
    print(f"wrote {fn_out} ({fn_out.stat().st_size} bytes)", flush=True)

    # The before/after diff control reads a payload with the same `levels` shape as the one
    # inlined above. Without emitting it standalone there is nothing for a later rung's
    # viewer to load, so the comparison the campaign grades rungs by could not be run.
    payload_out = out.with_name("HAWKING_GRAPH_PAYLOAD.json")
    payload_out.write_text(payload_json + "\n", encoding="utf-8")
    print(f"wrote {payload_out} ({payload_out.stat().st_size} bytes)"
          f" -- load this in a later viewer's before/after diff", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
