# G2 — Semantic graph analysis, recomposition ranking, viewer

Consumes the frozen contract `control/SEMANTIC_GRAPH_SCHEMA.json`.
Does **not** own extraction (`hawking_graph.py` is G1).

## Pipeline

Every output below lands in `build/graph/` by default, which is gitignored:
all of it is deterministic and rebuilt from source in about 40s. Nothing here
is evidence, so nothing here belongs in `evidence/`.

```bash
# 1) Extract the real graph (G1), then the behaviour map — in that order.
#    `--emit all` writes a *stub* behaviour map, so running it afterwards
#    would overwrite the real one.
python3.12 tools/graph/hawking_graph.py --emit all
python3.12 tools/graph/behaviour_map.py

# 2) Eight analyses + ranked candidates
python3.12 tools/graph/hawking_analyze.py \
  --graph build/graph/HAWKING_SEMANTIC_GRAPH.jsonl \
  --betweenness-k 64

# 3) Offline interactive viewer (single HTML, no network)
python3.12 tools/graph/viewer/build_viewer.py \
  --graph build/graph/HAWKING_SEMANTIC_GRAPH.jsonl \
  --cluster-map build/graph/HAWKING_CLUSTER_MAP.json \
  --candidates build/graph/HAWKING_RECOMPOSITION_CANDIDATES.json

# 4) Deterministic crate-level Graphviz checkpoint
python3.12 tools/graph/render_dot.py \
  --graph build/graph/HAWKING_SEMANTIC_GRAPH.jsonl \
  --cluster-map build/graph/HAWKING_CLUSTER_MAP.json \
  --out tools/graph/HAWKING_CRATE_GRAPH.dot
```

Against a synthetic fixture instead of the real repo, substitute step 1:

```bash
python3.12 tools/graph/fixture.py \
  --scale full --seed 42 \
  --out /tmp/HAWKING_SEMANTIC_GRAPH.jsonl \
  --planted-manifest /tmp/planted.json \
  --behaviour-map /tmp/HAWKING_BEHAVIOUR_TO_CODE_MAP.json
```

## Outputs

All under `build/graph/` unless noted.

| File | Role |
|------|------|
| `HAWKING_SEMANTIC_GRAPH.{jsonl,gexf,dot}` | The extracted graph, three serializations |
| `HAWKING_BEHAVIOUR_TO_CODE_MAP.json` | Behaviour constitution mapped onto code |
| `HAWKING_CLUSTER_MAP.json` | Eight analyses (machine + summary + timings) |
| `HAWKING_RECOMPOSITION_CANDIDATES.json` | Ranked reduction **proposals** |
| `HAWKING_ANALYSIS_REPORT.json` | Timing table + planted verification |
| `HAWKING_GRAPH_VIEWER.html` | Self-contained offline viewer |
| `HAWKING_GRAPH_VIEWER_FUNCTIONS.json` | Function-level LOD sibling; the viewer `fetch`es this at runtime, so keep the two together |
| `HAWKING_GRAPH_PAYLOAD.json` | Viewer payload, for before/after diffs |
| `tools/graph/HAWKING_CRATE_GRAPH.dot` | Crate checkpoint diagram (tracked, not under build/) |

## Analyses

1. SCC collapse (file + crate) over `imports`/`calls`
2. Louvain communities (+ light refinement) with directory scatter
3. Approximate betweenness (`k` samples) + articulation points + community cuts
4. Dominator trees from CLI / HTTP entry points
5. Structural clones via CFG signature (`signature_match` only; text not admissible)
6. Co-change without direct coupling
7. Fan-in adapter rings around thin-call authorities
8. Behaviour coverage (degrades if map empty/absent)

## Rules

- No new dependencies (networkx 3.3 only).
- Candidates are proposals, not authorised deletions.
- `expected_loc_removed` is honest: member LOC × (1 − retention); null sorts last.
- Merge requires ≥2 semantic criteria (state, lifecycle, error policy, tests, change history, callers).
