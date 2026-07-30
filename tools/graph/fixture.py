#!/usr/bin/env python3.12
"""Synthesise a schema-valid semantic graph with planted structure for G2 tests.

Produces a deterministic graph from --seed. Default scale is realistic:
~60_000 nodes and ~600_000 edges, with known SCCs, communities, a dominator
chain, a clone family, a pass-through wrapper ring, and a high-betweenness broker.

Usage:
    python3.12 tools/graph/fixture.py --out /tmp/fixture.jsonl
    python3.12 tools/graph/fixture.py --scale small --out /tmp/small.jsonl
    python3.12 tools/graph/fixture.py --out fixture.jsonl --planted-manifest /tmp/planted.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

# Allow running as script: python3.12 tools/graph/fixture.py
_GRAPH_DIR = Path(__file__).resolve().parent
if str(_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_DIR))
from graph_io import make_edge, make_node, sort_graph_records, write_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

SCALES = {
    # Fast unit-test scale
    "tiny": {
        "crates": 6,
        "dirs_per_crate": 3,
        "files_per_dir": 4,
        "fns_per_file": 5,
        "extra_call_mult": 2,
        "cochange_pairs": 40,
    },
    # Smoke scale (~3k nodes)
    "small": {
        "crates": 12,
        "dirs_per_crate": 4,
        "files_per_dir": 8,
        "fns_per_file": 6,
        "extra_call_mult": 4,
        "cochange_pairs": 200,
    },
    # Definition-of-done scale (~60k nodes, ~600k edges)
    "full": {
        "crates": 24,
        "dirs_per_crate": 8,
        "files_per_dir": 12,
        "fns_per_file": 18,
        "extra_call_mult": 8,
        "cochange_pairs": 8000,
    },
}


PLANTED_CLONE_SIG = "CFG:if>call(2)>ret|if>call(1)>ret|seq(3)"
PLANTED_CLONE_SIG_B = "CFG:loop>call(3)>if>ret|call(1)"


def _cfg_sig_for(rng: random.Random, complexity: int, fan_out: int) -> str:
    """Synthetic CFG signature from structural features (not text)."""
    parts = []
    if complexity >= 3:
        parts.append(f"if>call({1 + fan_out % 3})")
    if complexity >= 5:
        parts.append("loop")
    if complexity >= 7:
        parts.append(f"match({2 + complexity % 4})")
    parts.append(f"call({max(0, fan_out)})")
    parts.append("ret")
    return "CFG:" + ">".join(parts)


def generate(
    scale: str = "full",
    seed: int = 42,
    target_nodes: int | None = None,
    target_edges: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (nodes, edges, planted_manifest). Deterministic for (scale, seed)."""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose from {list(SCALES)}")
    cfg = dict(SCALES[scale])
    rng = random.Random(seed)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()

    def add_node(n: dict[str, Any]) -> str:
        nid = n["id"]
        if nid in node_ids:
            return nid
        node_ids.add(nid)
        nodes.append(n)
        return nid

    def add_edge(e: dict[str, Any]) -> None:
        edges.append(e)

    planted: dict[str, Any] = {
        "seed": seed,
        "scale": scale,
        "sccs_file": [],
        "sccs_crate": [],
        "communities": {},
        "dominator_chain": {},
        "clone_families": [],
        "wrapper_chain": {},
        "broker": {},
        "cochange_split": [],
    }

    # ---- repository ----
    add_node(
        make_node(
            "repo",
            "repository",
            "hawking",
            path=".",
            lang="none",
            attrs={"loc": 0, "subsystem": "shared"},
        )
    )

    subsystems = ["hawking", "hide", "laboratory", "shared"]
    langs = ["rust", "python", "typescript"]

    # ---- baseline crates / dirs / files / functions ----
    crate_names = [f"crate-{i:02d}" for i in range(cfg["crates"])]
    # Reserve names for planted structure
    plant_crates = [
        "plant-scc-a",
        "plant-scc-b",
        "plant-scc-c",
        "plant-comm-alpha",
        "plant-comm-beta",
        "plant-dom",
        "plant-clone-rs",
        "plant-clone-py",
        "plant-wrapper",
        "plant-broker",
    ]
    all_crate_names = plant_crates + crate_names

    file_ids: list[str] = []
    fn_ids: list[str] = []
    crate_file_map: dict[str, list[str]] = {c: [] for c in all_crate_names}
    file_fn_map: dict[str, list[str]] = {}

    for ci, cname in enumerate(all_crate_names):
        crate_id = f"crate:{cname}"
        sub = subsystems[ci % len(subsystems)]
        lang = "rust" if not cname.endswith("-py") else "python"
        if "clone-py" in cname:
            lang = "python"
        elif "clone-rs" in cname or cname.startswith("crate-") or cname.startswith("plant-"):
            lang = "rust" if "py" not in cname else "python"
        crate_path = f"crates/{cname}"
        add_node(
            make_node(
                crate_id,
                "crate",
                cname,
                path=crate_path,
                lang=lang,
                attrs={"loc": 0, "subsystem": sub, "public": True},
            )
        )
        add_edge(make_edge("repo", "contains", crate_id, evidence="cargo"))

        # Planted crates get a thin directory skeleton only; structure is added below.
        # This avoids thousands of isolated bulk files diluting planted communities.
        n_dirs = cfg["dirs_per_crate"]
        if cname.startswith("plant-"):
            n_dirs = 2
        for di in range(n_dirs):
            dpath = f"crates/{cname}/src/mod_{di:02d}"
            dir_id = f"dir:{dpath}"
            add_node(
                make_node(
                    dir_id,
                    "directory",
                    f"mod_{di:02d}",
                    path=dpath,
                    lang=lang,
                    attrs={"loc": 0, "subsystem": sub},
                )
            )
            add_edge(make_edge(crate_id, "contains", dir_id, evidence="ast"))

            # Skeleton only for plant crates (one placeholder file per dir)
            n_files = 1 if cname.startswith("plant-") else cfg["files_per_dir"]
            for fi in range(n_files):
                fpath = f"{dpath}/file_{fi:02d}.{'py' if lang == 'python' else 'rs'}"
                fid = f"file:{fpath}"
                floc = rng.randint(20, 180) if not cname.startswith("plant-") else 10
                add_node(
                    make_node(
                        fid,
                        "file",
                        Path(fpath).name,
                        path=fpath,
                        lang=lang,
                        span=[1, floc],
                        attrs={
                            "loc": floc,
                            "subsystem": sub,
                            "test_covered": rng.random() < 0.55,
                            "change_freq_90d": rng.randint(0, 12),
                            "change_freq_all": rng.randint(0, 80),
                            "complexity": rng.randint(1, 20),
                            "runtime_hot": rng.random() < 0.05,
                            "security_sensitive": rng.random() < 0.03,
                            "public": rng.random() < 0.4,
                        },
                    )
                )
                add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
                file_ids.append(fid)
                crate_file_map[cname].append(fid)
                file_fn_map[fid] = []

                n_fns = 1 if cname.startswith("plant-") else cfg["fns_per_file"]
                for fni in range(n_fns):
                    qname = f"mod_{di:02d}::fn_{fi:02d}_{fni:02d}"
                    fnid = f"fn:{fpath}#{qname}"
                    fl = max(3, floc // n_fns + rng.randint(-2, 4))
                    complexity = rng.randint(1, 12)
                    fo = rng.randint(0, 5)
                    sig = _cfg_sig_for(rng, complexity, fo)
                    add_node(
                        make_node(
                            fnid,
                            "function",
                            qname.split("::")[-1],
                            path=fpath,
                            lang=lang,
                            span=[1 + fni * 5, 1 + fni * 5 + fl],
                            attrs={
                                "loc": fl,
                                "subsystem": sub,
                                "complexity": complexity,
                                "fan_out": fo,
                                "fan_in": 0,
                                "public": fni == 0,
                                "test_covered": rng.random() < 0.5,
                                "cfg_signature": sig,
                                "runtime_hot": rng.random() < 0.03,
                                "security_sensitive": rng.random() < 0.02,
                                "change_freq_90d": rng.randint(0, 8),
                            },
                        )
                    )
                    add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
                    fn_ids.append(fnid)
                    file_fn_map[fid].append(fnid)

    # ---- bulk coupling edges (imports + calls) ----
    # Keep bulk graph DAG-ish so only planted SCCs form non-trivial components.
    # Forward-only imports/calls (higher index); no modular wrap-around.
    for cname, files in crate_file_map.items():
        if cname.startswith("plant-"):
            continue  # planted crates get bespoke edges
        for i, fid in enumerate(files):
            for k in range(min(3, len(files) - i - 1)):
                j = i + 1 + k * 2
                if j >= len(files):
                    break
                add_edge(
                    make_edge(
                        fid,
                        "imports",
                        files[j],
                        weight=1.0,
                        evidence="ast",
                        confidence=1.0,
                    )
                )
            fns = file_fn_map[fid]
            for fn in fns:
                targets = []
                for off in range(1, 1 + cfg["extra_call_mult"]):
                    j = i + off
                    if j >= len(files):
                        break
                    tfns = file_fn_map[files[j]]
                    if tfns:
                        targets.append(rng.choice(tfns))
                # same-file: only call higher-index siblings (DAG)
                if len(fns) > 1:
                    idx = fns.index(fn) if fn in fns else 0
                    later = fns[idx + 1 :]
                    if later:
                        targets.append(rng.choice(later))
                for t in targets:
                    add_edge(
                        make_edge(
                            fn,
                            "calls",
                            t,
                            weight=float(rng.randint(1, 5)),
                            count=rng.randint(1, 8),
                            evidence="ast",
                        )
                    )

    # Cross-crate imports: forward-only along crate_names to avoid a giant crate SCC
    for i, cname in enumerate(crate_names):
        src_files = crate_file_map[cname]
        if not src_files:
            continue
        for step in range(1, min(4, len(crate_names) - i)):
            other = crate_names[i + step]
            dst_files = crate_file_map[other]
            if not dst_files:
                continue
            for _ in range(min(4, len(src_files), len(dst_files))):
                sf = rng.choice(src_files)
                df = rng.choice(dst_files)
                add_edge(
                    make_edge(sf, "imports", df, weight=1.0, evidence="cargo", confidence=0.95)
                )

    # =====================================================================
    # PLANTED STRUCTURE 1 — file-level SCCs (cycle of 4 files)
    # =====================================================================
    scc_files = []
    scc_crate = "plant-scc-a"
    scc_paths = [
        f"crates/{scc_crate}/src/mod_00/scc_cycle_{i}.rs" for i in range(4)
    ]
    for i, fpath in enumerate(scc_paths):
        fid = f"file:{fpath}"
        # may already exist from bulk generation under mod_00 — use dedicated names
        if fid not in node_ids:
            add_node(
                make_node(
                    fid,
                    "file",
                    Path(fpath).name,
                    path=fpath,
                    lang="rust",
                    span=[1, 90],
                    attrs={
                        "loc": 80 + i * 5,
                        "subsystem": "hawking",
                        "complexity": 8,
                        "test_covered": True,
                        "public": True,
                        "planted": "scc_file",
                    },
                )
            )
            dir_id = f"dir:crates/{scc_crate}/src/mod_00"
            if dir_id in node_ids:
                add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
            file_ids.append(fid)
            crate_file_map[scc_crate].append(fid)
            file_fn_map[fid] = []
            # one public fn per scc file
            fnid = f"fn:{fpath}#cycle_entry_{i}"
            add_node(
                make_node(
                    fnid,
                    "function",
                    f"cycle_entry_{i}",
                    path=fpath,
                    lang="rust",
                    span=[1, 40],
                    attrs={
                        "loc": 35,
                        "subsystem": "hawking",
                        "complexity": 5,
                        "public": True,
                        "cfg_signature": f"CFG:call(1)>ret#scc{i}",
                        "planted": "scc_file",
                    },
                )
            )
            add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
            fn_ids.append(fnid)
            file_fn_map[fid].append(fnid)
        scc_files.append(fid)

    # Cycle: 0->1->2->3->0 via imports + calls
    for i in range(4):
        a, b = scc_files[i], scc_files[(i + 1) % 4]
        add_edge(make_edge(a, "imports", b, weight=2.0, evidence="ast"))
        fa = file_fn_map[a][0]
        fb = file_fn_map[b][0]
        add_edge(make_edge(fa, "calls", fb, weight=3.0, count=4, evidence="ast"))

    planted["sccs_file"].append(
        {
            "id": "planted-scc-file-cycle4",
            "members": list(scc_files),
            "size": 4,
            "level": "file",
        }
    )

    # Second smaller file SCC (3-cycle) in plant-scc-b
    scc2_files = []
    for i in range(3):
        fpath = f"crates/plant-scc-b/src/mod_00/scc2_{i}.rs"
        fid = f"file:{fpath}"
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang="rust",
                span=[1, 60],
                attrs={
                    "loc": 55,
                    "subsystem": "hawking",
                    "planted": "scc_file",
                    "complexity": 4,
                },
            )
        )
        dir_id = "dir:crates/plant-scc-b/src/mod_00"
        if dir_id in node_ids:
            add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
        scc2_files.append(fid)
        file_fn_map[fid] = []
        fnid = f"fn:{fpath}#s2_{i}"
        add_node(
            make_node(
                fnid,
                "function",
                f"s2_{i}",
                path=fpath,
                lang="rust",
                span=[1, 30],
                attrs={"loc": 25, "subsystem": "hawking", "cfg_signature": f"CFG:scc2:{i}"},
            )
        )
        add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
        file_fn_map[fid].append(fnid)
        fn_ids.append(fnid)
        file_ids.append(fid)

    for i in range(3):
        a, b = scc2_files[i], scc2_files[(i + 1) % 3]
        add_edge(make_edge(a, "imports", b, weight=1.0, evidence="ast"))
        add_edge(
            make_edge(
                file_fn_map[a][0],
                "calls",
                file_fn_map[b][0],
                weight=2.0,
                evidence="ast",
            )
        )
    planted["sccs_file"].append(
        {
            "id": "planted-scc-file-cycle3",
            "members": list(scc2_files),
            "size": 3,
            "level": "file",
        }
    )

    # =====================================================================
    # PLANTED STRUCTURE 1b — crate-level SCC (plant-scc-a/b/c mutual imports)
    # =====================================================================
    # Pick representative files in each plant-scc crate that import each other
    crate_scc_reps = {}
    for cname in ("plant-scc-a", "plant-scc-b", "plant-scc-c"):
        files = crate_file_map[cname]
        if not files:
            # ensure at least one file
            fpath = f"crates/{cname}/src/mod_00/bridge.rs"
            fid = f"file:{fpath}"
            add_node(
                make_node(
                    fid,
                    "file",
                    "bridge.rs",
                    path=fpath,
                    lang="rust",
                    span=[1, 40],
                    attrs={"loc": 40, "subsystem": "hawking", "planted": "scc_crate"},
                )
            )
            files = [fid]
            crate_file_map[cname] = files
            file_fn_map[fid] = []
        crate_scc_reps[cname] = files[0]

    # Mutual imports a->b->c->a at file level induces crate-level SCC
    order = ["plant-scc-a", "plant-scc-b", "plant-scc-c"]
    for i, cname in enumerate(order):
        nxt = order[(i + 1) % 3]
        add_edge(
            make_edge(
                crate_scc_reps[cname],
                "imports",
                crate_scc_reps[nxt],
                weight=5.0,
                evidence="cargo",
                confidence=1.0,
            )
        )
    planted["sccs_crate"].append(
        {
            "id": "planted-scc-crate-abc",
            "members": [f"crate:{c}" for c in order],
            "size": 3,
            "level": "crate",
            "via_files": [crate_scc_reps[c] for c in order],
        }
    )

    # =====================================================================
    # PLANTED STRUCTURE 2 — communities (alpha dense; beta scattered across 9 dirs)
    # Scale with fixture size so communities remain detectable at full scale.
    # =====================================================================
    alpha_n = 24 if scale == "tiny" else (36 if scale == "small" else 80)
    alpha_files: list[str] = []
    # Dense clique-ish community in plant-comm-alpha
    for i in range(alpha_n):
        di = i % 3
        fi = i
        fpath = f"crates/plant-comm-alpha/src/mod_{di:02d}/alpha_{fi:02d}.rs"
        fid = f"file:{fpath}"
        if fid not in node_ids:
            add_node(
                make_node(
                    fid,
                    "file",
                    Path(fpath).name,
                    path=fpath,
                    lang="rust",
                    span=[1, 70],
                    attrs={
                        "loc": 65,
                        "subsystem": "hawking",
                        "planted": "community_alpha",
                        "complexity": 6,
                    },
                )
            )
            dir_id = f"dir:crates/plant-comm-alpha/src/mod_{di:02d}"
            if dir_id in node_ids:
                add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
            file_fn_map[fid] = []
            fnid = f"fn:{fpath}#alpha_work"
            add_node(
                make_node(
                    fnid,
                    "function",
                    "alpha_work",
                    path=fpath,
                    lang="rust",
                    span=[1, 40],
                    attrs={
                        "loc": 40,
                        "subsystem": "hawking",
                        "cfg_signature": "CFG:alpha_work",
                        "planted": "community_alpha",
                    },
                )
            )
            add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
            file_fn_map[fid].append(fnid)
            fn_ids.append(fnid)
            file_ids.append(fid)
            crate_file_map["plant-comm-alpha"].append(fid)
        alpha_files.append(fid)

    # Complete undirected clique among alpha (each unordered pair once, high weight)
    for i, fid in enumerate(alpha_files):
        for other in alpha_files[i + 1 :]:
            add_edge(make_edge(fid, "imports", other, weight=8.0, evidence="ast"))
            add_edge(make_edge(other, "imports", fid, weight=8.0, evidence="ast"))
            fa = file_fn_map[fid][0] if file_fn_map[fid] else None
            fb = file_fn_map[other][0] if file_fn_map[other] else None
            if fa and fb:
                add_edge(make_edge(fa, "calls", fb, weight=8.0, count=4, evidence="ast"))
                add_edge(make_edge(fb, "calls", fa, weight=8.0, count=4, evidence="ast"))

    planted["communities"]["alpha"] = {
        "id": "planted-community-alpha",
        "members": list(alpha_files),
        "n_dirs": 3,
        "expected_coherent": True,
    }

    # Beta: same tight coupling but members deliberately scattered across 9 directories
    beta_n = 24 if scale == "tiny" else (36 if scale == "small" else 80)
    beta_files: list[str] = []
    beta_dirs = [f"scatter_{i}" for i in range(9)]
    for i in range(beta_n):
        dname = beta_dirs[i % 9]
        fpath = f"crates/plant-comm-beta/src/{dname}/beta_{i:02d}.rs"
        fid = f"file:{fpath}"
        dir_id = f"dir:crates/plant-comm-beta/src/{dname}"
        if dir_id not in node_ids:
            add_node(
                make_node(
                    dir_id,
                    "directory",
                    dname,
                    path=f"crates/plant-comm-beta/src/{dname}",
                    lang="rust",
                    attrs={"loc": 0, "subsystem": "hide", "planted": "community_beta"},
                )
            )
            add_edge(
                make_edge(
                    "crate:plant-comm-beta",
                    "contains",
                    dir_id,
                    evidence="ast",
                )
            )
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang="rust",
                span=[1, 55],
                attrs={
                    "loc": 50,
                    "subsystem": "hide",
                    "planted": "community_beta",
                    "complexity": 5,
                },
            )
        )
        add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
        file_fn_map[fid] = []
        fnid = f"fn:{fpath}#beta_work"
        add_node(
            make_node(
                fnid,
                "function",
                "beta_work",
                path=fpath,
                lang="rust",
                span=[1, 30],
                attrs={
                    "loc": 30,
                    "subsystem": "hide",
                    "cfg_signature": "CFG:beta_work",
                    "planted": "community_beta",
                },
            )
        )
        add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
        file_fn_map[fid].append(fnid)
        beta_files.append(fid)
        file_ids.append(fid)
        fn_ids.append(fnid)
        crate_file_map["plant-comm-beta"].append(fid)

    for i, fid in enumerate(beta_files):
        for k in range(5):
            other = beta_files[(i + 2 + k * 3) % len(beta_files)]
            if other == fid:
                continue
            add_edge(make_edge(fid, "imports", other, weight=2.5, evidence="ast"))
            add_edge(
                make_edge(
                    file_fn_map[fid][0],
                    "calls",
                    file_fn_map[other][0],
                    weight=4.0,
                    count=4,
                    evidence="ast",
                )
            )

    planted["communities"]["beta"] = {
        "id": "planted-community-beta",
        "members": list(beta_files),
        "n_dirs": 9,
        "expected_coherent": True,
        "scatter_dirs": 9,
        "note": "members tightly coupled but live in 9 directories — folders are wrong",
    }

    # =====================================================================
    # PLANTED STRUCTURE 6 — high-betweenness broker between alpha and beta
    # =====================================================================
    broker_path = "crates/plant-broker/src/mod_00/broker.rs"
    broker_fid = f"file:{broker_path}"
    broker_fn = f"fn:{broker_path}#translate"
    add_node(
        make_node(
            broker_fid,
            "file",
            "broker.rs",
            path=broker_path,
            lang="rust",
            span=[1, 28],
            attrs={
                "loc": 28,
                "subsystem": "shared",
                "planted": "broker",
                "complexity": 2,
                "public": True,
            },
        )
    )
    dir_id = "dir:crates/plant-broker/src/mod_00"
    if dir_id in node_ids:
        add_edge(make_edge(dir_id, "contains", broker_fid, evidence="ast"))
    add_node(
        make_node(
            broker_fn,
            "function",
            "translate",
            path=broker_path,
            lang="rust",
            span=[1, 20],
            attrs={
                "loc": 18,
                "subsystem": "shared",
                "planted": "broker",
                "complexity": 2,
                "cfg_signature": "CFG:call(2)>ret",
                "public": True,
            },
        )
    )
    add_edge(make_edge(broker_fid, "contains", broker_fn, evidence="ast"))
    file_fn_map[broker_fid] = [broker_fn]
    file_ids.append(broker_fid)
    fn_ids.append(broker_fn)

    # Broker is the *only* bridge between alpha and beta file-coupling graphs.
    # Connect many members so shortest alpha↔beta paths concentrate on the broker
    # (detectable even with approximate betweenness sampling).
    for fid in alpha_files:
        add_edge(make_edge(fid, "imports", broker_fid, weight=1.0, evidence="ast"))
        if file_fn_map[fid]:
            add_edge(
                make_edge(
                    file_fn_map[fid][0],
                    "calls",
                    broker_fn,
                    weight=1.0,
                    count=1,
                    evidence="ast",
                )
            )
    for fid in beta_files:
        add_edge(make_edge(broker_fid, "imports", fid, weight=1.0, evidence="ast"))
        if file_fn_map[fid]:
            add_edge(
                make_edge(
                    broker_fn,
                    "calls",
                    file_fn_map[fid][0],
                    weight=1.0,
                    count=1,
                    evidence="ast",
                )
            )

    planted["broker"] = {
        "id": "planted-broker",
        "file": broker_fid,
        "function": broker_fn,
        "loc": 28,
        "note": "low-LOC translation layer between alpha and beta communities",
    }

    # =====================================================================
    # PLANTED STRUCTURE 3 — dominator chain shared by CLI + HTTP entry points
    # =====================================================================
    dom_files = {
        "entry": "crates/plant-dom/src/mod_00/entry.rs",
        "prepare": "crates/plant-dom/src/mod_00/prepare.rs",
        "execute": "crates/plant-dom/src/mod_00/execute.rs",
        "finalize": "crates/plant-dom/src/mod_00/finalize.rs",
    }
    dom_fns = {}
    for key, fpath in dom_files.items():
        fid = f"file:{fpath}"
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang="rust",
                span=[1, 50],
                attrs={"loc": 45, "subsystem": "hawking", "planted": "dominator"},
            )
        )
        dir_id = "dir:crates/plant-dom/src/mod_00"
        if dir_id in node_ids:
            add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
        fnid = f"fn:{fpath}#{key}"
        add_node(
            make_node(
                fnid,
                "function",
                key,
                path=fpath,
                lang="rust",
                span=[1, 40],
                attrs={
                    "loc": 38,
                    "subsystem": "hawking",
                    "planted": "dominator",
                    "cfg_signature": f"CFG:dom:{key}",
                    "public": True,
                },
            )
        )
        add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
        file_fn_map[fid] = [fnid]
        dom_fns[key] = fnid
        fn_ids.append(fnid)
        file_ids.append(fid)

    # Call chain: entry -> prepare -> execute -> finalize
    add_edge(make_edge(dom_fns["entry"], "calls", dom_fns["prepare"], weight=1.0, evidence="ast"))
    add_edge(
        make_edge(dom_fns["prepare"], "calls", dom_fns["execute"], weight=1.0, evidence="ast")
    )
    add_edge(
        make_edge(dom_fns["execute"], "calls", dom_fns["finalize"], weight=1.0, evidence="ast")
    )

    # CLI commands
    cli_run = "cli:plant-tool run"
    cli_check = "cli:plant-tool check"
    for cli_id, name in ((cli_run, "run"), (cli_check, "check")):
        add_node(
            make_node(
                cli_id,
                "cli_command",
                name,
                path="crates/plant-dom",
                lang="rust",
                attrs={"loc": 0, "subsystem": "hawking", "planted": "dominator", "public": True},
            )
        )
        add_edge(make_edge(cli_id, "calls", dom_fns["entry"], weight=1.0, evidence="ast"))

    # HTTP route as adapter entry
    route = "adapter:http/plant_route"
    add_node(
        make_node(
            route,
            "adapter",
            "plant_route",
            path="crates/plant-dom/src/http.rs",
            lang="rust",
            attrs={"loc": 12, "subsystem": "hawking", "planted": "dominator", "public": True},
        )
    )
    add_edge(make_edge(route, "calls", dom_fns["entry"], weight=1.0, evidence="ast"))
    # Also a second route that joins at prepare (shared control path)
    route2 = "adapter:http/plant_route_status"
    add_node(
        make_node(
            route2,
            "adapter",
            "plant_route_status",
            path="crates/plant-dom/src/http_status.rs",
            lang="rust",
            attrs={"loc": 10, "subsystem": "hawking", "planted": "dominator"},
        )
    )
    add_edge(make_edge(route2, "calls", dom_fns["prepare"], weight=1.0, evidence="ast"))

    planted["dominator_chain"] = {
        "id": "planted-dominator-chain",
        "entry_points": [cli_run, cli_check, route, route2],
        "shared_path": [dom_fns["prepare"], dom_fns["execute"], dom_fns["finalize"]],
        "full_chain": [
            dom_fns["entry"],
            dom_fns["prepare"],
            dom_fns["execute"],
            dom_fns["finalize"],
        ],
        "note": "prepare/execute/finalize dominated from multiple entry points",
    }

    # =====================================================================
    # PLANTED STRUCTURE 4 — structural clone family (signature_match)
    # =====================================================================
    clone_members = []
    for i in range(8):
        lang = "python" if i >= 5 else "rust"
        crate = "plant-clone-py" if lang == "python" else "plant-clone-rs"
        ext = "py" if lang == "python" else "rs"
        fpath = f"crates/{crate}/src/mod_00/clone_impl_{i:02d}.{ext}"
        fid = f"file:{fpath}"
        fname = f"do_thing_variant_{i}" if i % 2 == 0 else f"handle_payload_{i}"
        fnid = f"fn:{fpath}#{fname}"
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang=lang,
                span=[1, 44],
                attrs={
                    "loc": 44,
                    "subsystem": "laboratory" if lang == "python" else "hawking",
                    "planted": "clone_family",
                    "complexity": 6,
                },
            )
        )
        dir_id = f"dir:crates/{crate}/src/mod_00"
        if dir_id in node_ids:
            add_edge(make_edge(dir_id, "contains", fid, evidence="ast"))
        add_node(
            make_node(
                fnid,
                "function",
                fname,
                path=fpath,
                lang=lang,
                span=[1, 40],
                attrs={
                    "loc": 40,
                    "subsystem": "laboratory" if lang == "python" else "hawking",
                    "planted": "clone_family",
                    "complexity": 6,
                    "cfg_signature": PLANTED_CLONE_SIG,
                    "public": True,
                },
            )
        )
        add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
        clone_members.append(fnid)
        file_fn_map[fid] = [fnid]
        fn_ids.append(fnid)
        file_ids.append(fid)

    # Secondary smaller clone family
    clone_b = []
    for i in range(3):
        fpath = f"crates/plant-clone-rs/src/mod_01/clone_b_{i}.rs"
        fid = f"file:{fpath}"
        fnid = f"fn:{fpath}#other_name_{i}"
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang="rust",
                span=[1, 30],
                attrs={"loc": 30, "subsystem": "hawking", "planted": "clone_family_b"},
            )
        )
        add_node(
            make_node(
                fnid,
                "function",
                f"other_name_{i}",
                path=fpath,
                lang="rust",
                span=[1, 28],
                attrs={
                    "loc": 28,
                    "subsystem": "hawking",
                    "cfg_signature": PLANTED_CLONE_SIG_B,
                    "planted": "clone_family_b",
                },
            )
        )
        add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
        clone_b.append(fnid)
        fn_ids.append(fnid)

    planted["clone_families"] = [
        {
            "id": "planted-clone-family-a",
            "signature": PLANTED_CLONE_SIG,
            "members": clone_members,
            "cross_lang": True,
            "cross_crate": True,
            "match_kind": "signature_match",
        },
        {
            "id": "planted-clone-family-b",
            "signature": PLANTED_CLONE_SIG_B,
            "members": clone_b,
            "cross_lang": False,
            "cross_crate": False,
            "match_kind": "signature_match",
        },
    ]

    # =====================================================================
    # PLANTED STRUCTURE 5 — pass-through wrapper chain (fan-in adapters)
    # =====================================================================
    auth_path = "crates/plant-wrapper/src/mod_00/authority.rs"
    auth_fid = f"file:{auth_path}"
    auth_fn = f"fn:{auth_path}#core_authority"
    add_node(
        make_node(
            auth_fid,
            "file",
            "authority.rs",
            path=auth_path,
            lang="rust",
            span=[1, 120],
            attrs={
                "loc": 120,
                "subsystem": "hawking",
                "planted": "wrapper_authority",
                "complexity": 15,
                "public": True,
            },
        )
    )
    dir_id = "dir:crates/plant-wrapper/src/mod_00"
    if dir_id in node_ids:
        add_edge(make_edge(dir_id, "contains", auth_fid, evidence="ast"))
    add_node(
        make_node(
            auth_fn,
            "function",
            "core_authority",
            path=auth_path,
            lang="rust",
            span=[1, 100],
            attrs={
                "loc": 100,
                "subsystem": "hawking",
                "planted": "wrapper_authority",
                "complexity": 15,
                "cfg_signature": "CFG:auth:core",
                "public": True,
            },
        )
    )
    add_edge(make_edge(auth_fid, "contains", auth_fn, evidence="ast"))

    wrappers = []
    for i in range(12):
        fpath = f"crates/plant-wrapper/src/mod_01/wrap_{i:02d}.rs"
        fid = f"file:{fpath}"
        fnid = f"fn:{fpath}#thin_wrap_{i}"
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang="rust",
                span=[1, 12],
                attrs={
                    "loc": 12,
                    "subsystem": "hawking",
                    "planted": "wrapper",
                    "complexity": 1,
                },
            )
        )
        dir_w = "dir:crates/plant-wrapper/src/mod_01"
        if dir_w not in node_ids:
            add_node(
                make_node(
                    dir_w,
                    "directory",
                    "mod_01",
                    path="crates/plant-wrapper/src/mod_01",
                    lang="rust",
                    attrs={"loc": 0, "subsystem": "hawking"},
                )
            )
            add_edge(
                make_edge("crate:plant-wrapper", "contains", dir_w, evidence="ast")
            )
        add_edge(make_edge(dir_w, "contains", fid, evidence="ast"))
        add_node(
            make_node(
                fnid,
                "function",
                f"thin_wrap_{i}",
                path=fpath,
                lang="rust",
                span=[1, 10],
                attrs={
                    "loc": 8,
                    "subsystem": "hawking",
                    "planted": "wrapper",
                    "complexity": 1,
                    "cfg_signature": "CFG:call(1)>ret",
                    "public": True,
                },
            )
        )
        add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
        # thin body: only calls authority
        add_edge(make_edge(fnid, "calls", auth_fn, weight=1.0, count=1, evidence="ast"))
        add_edge(make_edge(fid, "imports", auth_fid, weight=1.0, evidence="ast"))
        wrappers.append({"file": fid, "function": fnid, "loc": 8})
        fn_ids.append(fnid)
        file_ids.append(fid)

    planted["wrapper_chain"] = {
        "id": "planted-wrapper-ring",
        "authority": auth_fn,
        "authority_file": auth_fid,
        "wrappers": wrappers,
        "wrapper_count": len(wrappers),
        "wrapper_total_loc": sum(w["loc"] for w in wrappers),
        "note": "12 thin adapters around one authority — generate bindings candidate",
    }

    # =====================================================================
    # PLANTED STRUCTURE 7 — co-change without direct coupling (split module)
    # =====================================================================
    co_a = "file:crates/crate-00/src/mod_00/split_left.rs"
    co_b = "file:crates/crate-01/src/mod_00/split_right.rs"
    # ensure nodes
    for fpath, sub in (
        ("crates/crate-00/src/mod_00/split_left.rs", "hawking"),
        ("crates/crate-01/src/mod_00/split_right.rs", "hawking"),
    ):
        fid = f"file:{fpath}"
        if fid not in node_ids:
            add_node(
                make_node(
                    fid,
                    "file",
                    Path(fpath).name,
                    path=fpath,
                    lang="rust",
                    span=[1, 80],
                    attrs={
                        "loc": 80,
                        "subsystem": sub,
                        "planted": "cochange_split",
                        "change_freq_90d": 20,
                    },
                )
            )
            file_ids.append(fid)
    add_edge(
        make_edge(
            "file:crates/crate-00/src/mod_00/split_left.rs",
            "co_changes",
            "file:crates/crate-01/src/mod_00/split_right.rs",
            weight=18.0,
            count=18,
            evidence="git",
            confidence=0.9,
        )
    )
    # deliberately NO imports/calls between them
    planted["cochange_split"].append(
        {
            "id": "planted-cochange-split",
            "a": "file:crates/crate-00/src/mod_00/split_left.rs",
            "b": "file:crates/crate-01/src/mod_00/split_right.rs",
            "co_changes_weight": 18.0,
            "direct_coupling": False,
        }
    )

    # Bulk co_changes among random *non-planted* file pairs (do not dissolve planted communities)
    planted_files = {
        n["id"]
        for n in nodes
        if n["type"] == "file" and n.get("attrs", {}).get("planted")
    }
    bulk_files = [f for f in file_ids if f not in planted_files]
    for _ in range(cfg["cochange_pairs"]):
        if len(bulk_files) < 2:
            break
        a, b = rng.sample(bulk_files, 2)
        w = float(rng.randint(1, 6))
        add_edge(
            make_edge(a, "co_changes", b, weight=w, count=int(w), evidence="git", confidence=0.7)
        )

    # reads_state / writes_state for community weights
    state_ids = []
    for i in range(min(40, max(5, len(all_crate_names)))):
        sid = f"state:store/table_{i}"
        add_node(
            make_node(
                sid,
                "state",
                f"table_{i}",
                path=None,
                lang="none",
                attrs={"loc": 0, "subsystem": "shared"},
            )
        )
        state_ids.append(sid)
    for fid in rng.sample(file_ids, min(len(file_ids), 200)):
        s = rng.choice(state_ids)
        et = rng.choice(["reads_state", "writes_state"])
        add_edge(make_edge(fid, et, s, weight=1.0, evidence="ast", confidence=0.8))

    # CLI bulk + a few more entry points for dominator analysis
    for i in range(8):
        cli = f"cli:tool-{i} main"
        add_node(
            make_node(
                cli,
                "cli_command",
                "main",
                path=f"crates/crate-{i:02d}",
                lang="rust",
                attrs={"loc": 0, "subsystem": "hawking", "public": True},
            )
        )
        # call into some function of that crate
        cname = f"crate-{i:02d}"
        files = crate_file_map.get(cname, [])
        if files and file_fn_map.get(files[0]):
            add_edge(
                make_edge(cli, "calls", file_fn_map[files[0]][0], weight=1.0, evidence="ast")
            )

    # Behaviours (optional map companions)
    behaviours = []
    for i in range(6):
        bid = f"behaviour:BC-{i:03d}"
        add_node(
            make_node(
                bid,
                "behaviour",
                f"BC-{i:03d}",
                path=None,
                lang="none",
                attrs={"loc": 0, "subsystem": "shared", "public": True},
            )
        )
        behaviours.append(bid)
        # depend on some functions
        for fn in rng.sample(fn_ids, min(20, len(fn_ids))):
            add_edge(
                make_edge(
                    bid,
                    "depends_on_behaviour",
                    fn,
                    weight=1.0,
                    evidence="registry",
                    confidence=0.9,
                )
            )

    # ---- pad edges if below target ----
    # Cross-link more calls to reach ~10 edges/node average
    if target_edges is None:
        # full scale aims ~600k
        target_edges = 600_000 if scale == "full" else (50_000 if scale == "small" else 5_000)

    # Also pad nodes if needed
    if target_nodes is None:
        target_nodes = 60_000 if scale == "full" else (5_000 if scale == "small" else 800)

    pad_i = 0
    while len(nodes) < target_nodes:
        fpath = f"crates/crate-pad/src/mod_00/pad_{pad_i:05d}.rs"
        # ensure pad crate
        if "crate:crate-pad" not in node_ids:
            add_node(
                make_node(
                    "crate:crate-pad",
                    "crate",
                    "crate-pad",
                    path="crates/crate-pad",
                    lang="rust",
                    attrs={"loc": 0, "subsystem": "shared"},
                )
            )
            add_edge(make_edge("repo", "contains", "crate:crate-pad", evidence="cargo"))
            add_node(
                make_node(
                    "dir:crates/crate-pad/src/mod_00",
                    "directory",
                    "mod_00",
                    path="crates/crate-pad/src/mod_00",
                    lang="rust",
                    attrs={"loc": 0, "subsystem": "shared"},
                )
            )
            add_edge(
                make_edge(
                    "crate:crate-pad",
                    "contains",
                    "dir:crates/crate-pad/src/mod_00",
                    evidence="ast",
                )
            )
            crate_file_map["crate-pad"] = []
        fid = f"file:{fpath}"
        add_node(
            make_node(
                fid,
                "file",
                Path(fpath).name,
                path=fpath,
                lang="rust",
                span=[1, 30],
                attrs={"loc": 30, "subsystem": "shared", "complexity": 2},
            )
        )
        add_edge(
            make_edge(
                "dir:crates/crate-pad/src/mod_00",
                "contains",
                fid,
                evidence="ast",
            )
        )
        file_ids.append(fid)
        file_fn_map[fid] = []
        for j in range(3):
            fnid = f"fn:{fpath}#pad_fn_{j}"
            add_node(
                make_node(
                    fnid,
                    "function",
                    f"pad_fn_{j}",
                    path=fpath,
                    lang="rust",
                    span=[1, 10],
                    attrs={
                        "loc": 8,
                        "subsystem": "shared",
                        "cfg_signature": _cfg_sig_for(rng, 2, 1),
                        "complexity": 2,
                    },
                )
            )
            add_edge(make_edge(fid, "contains", fnid, evidence="ast"))
            file_fn_map[fid].append(fnid)
            fn_ids.append(fnid)
        pad_i += 1
        if pad_i > target_nodes:  # safety
            break

    # Pad edges with additional call noise — only among non-planted bulk functions,
    # and only forward in list order so we do not dissolve planted SCCs into a giant component.
    planted_fn_ids = {
        n["id"]
        for n in nodes
        if n["type"] == "function" and n.get("attrs", {}).get("planted")
    }
    bulk_fns = [f for f in fn_ids if f not in planted_fn_ids]
    # stable index for DAG direction
    bulk_index = {f: i for i, f in enumerate(bulk_fns)}
    edge_budget = target_edges - len(edges)
    if edge_budget > 0 and len(bulk_fns) > 10:
        # batch-append for speed at full scale
        for _ in range(edge_budget):
            i = rng.randrange(0, len(bulk_fns) - 1)
            j = rng.randrange(i + 1, min(len(bulk_fns), i + 1 + max(50, len(bulk_fns) // 20)))
            a = bulk_fns[i]
            b = bulk_fns[j]
            edges.append(
                {
                    "kind": "edge",
                    "id": f"{a}|calls|{b}|pad{_}",
                    "src": a,
                    "dst": b,
                    "type": "calls",
                    "attrs": {
                        "weight": float(rng.randint(1, 3)),
                        "count": 1,
                        "evidence": "ast",
                        "confidence": 0.85,
                    },
                }
            )

    # Recompute fan_in/fan_out roughly for functions
    fan_in: dict[str, int] = {}
    fan_out: dict[str, int] = {}
    for e in edges:
        if e["type"] in ("calls", "imports"):
            fan_out[e["src"]] = fan_out.get(e["src"], 0) + 1
            fan_in[e["dst"]] = fan_in.get(e["dst"], 0) + 1
    id_to_node = {n["id"]: n for n in nodes}
    for nid, n in id_to_node.items():
        if nid in fan_in:
            n["attrs"]["fan_in"] = fan_in[nid]
        if nid in fan_out:
            n["attrs"]["fan_out"] = fan_out[nid]

    # Sum loc up directories/crates (approx)
    for n in nodes:
        if n["type"] == "file":
            continue
        # leave as-is; analysis sums children

    planted["stats"] = {
        "nodes": len(nodes),
        "edges": len(edges),
        "seed": seed,
        "scale": scale,
    }
    planted["behaviour_ids"] = behaviours

    return nodes, edges, planted


def write_behaviour_map(
    path: Path,
    planted: dict[str, Any],
    nodes: list[dict],
    edges: list[dict],
) -> None:
    """Companion HAWKING_BEHAVIOUR_TO_CODE_MAP.json with partial coverage."""
    # behaviours depend_on some fns; everything else is uncovered
    covered: dict[str, list[str]] = {b: [] for b in planted.get("behaviour_ids", [])}
    for e in edges:
        if e["type"] == "depends_on_behaviour" and e["src"] in covered:
            covered[e["src"]].append(e["dst"])
    doc = {
        "schema": "hawking.behaviour_to_code.v1",
        "note": "Fixture behaviour map — partial coverage for behaviour analysis tests",
        "behaviours": {
            bid: {
                "id": bid,
                "reachable_nodes": sorted(set(nids)),
                "reachable_loc": 0,
            }
            for bid, nids in covered.items()
        },
    }
    # compute loc
    loc = {n["id"]: int(n["attrs"].get("loc", 0)) for n in nodes}
    for bid, body in doc["behaviours"].items():
        body["reachable_loc"] = sum(loc.get(x, 0) for x in body["reachable_nodes"])
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--scale", choices=list(SCALES), default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-nodes", type=int, default=None)
    p.add_argument("--target-edges", type=int, default=None)
    p.add_argument(
        "--planted-manifest",
        default=None,
        help="Write planted-structure ground truth JSON here",
    )
    p.add_argument(
        "--behaviour-map",
        default=None,
        help="Write HAWKING_BEHAVIOUR_TO_CODE_MAP.json companion",
    )
    args = p.parse_args(argv)

    nodes, edges, planted = generate(
        scale=args.scale,
        seed=args.seed,
        target_nodes=args.target_nodes,
        target_edges=args.target_edges,
    )
    records = sort_graph_records(nodes, edges)
    n = write_jsonl(args.out, records)
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "records": n,
                "nodes": len(nodes),
                "edges": len(edges),
                "seed": args.seed,
                "scale": args.scale,
            },
            indent=2,
        )
    )

    if args.planted_manifest:
        Path(args.planted_manifest).write_text(
            json.dumps(planted, indent=2) + "\n", encoding="utf-8"
        )
        print(f"planted_manifest: {args.planted_manifest}")

    if args.behaviour_map:
        write_behaviour_map(Path(args.behaviour_map), planted, nodes, edges)
        print(f"behaviour_map: {args.behaviour_map}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
