#!/usr/bin/env python3.12
"""The graph instrument's own correctness oracle.

`tools/graph/fixture.py` synthesises a schema-valid graph with *known* planted structure --
strongly connected components, communities scattered across directories, a dominator chain,
clone families, a pass-through wrapper ring, a high-betweenness broker, a co-change split.
Running the analyses over it is the only way to know that "Louvain found 157 communities" on
the real tree means anything.

That fixture was written to develop the analyses and was then left dormant: 1,510 lines that
nothing invoked. Dormant validation is not validation, so this wires it into the suite.

    python3.12 -m pytest tools/graph/test_graph_instrument.py

Deliberately runs at `tiny` scale. The full 60k-node/600k-edge run takes ~23s and belongs in
a manual sweep, not in every suite run; what has to be checked continuously is that the
analyses still *find planted truth*, and that is scale-independent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable or "python3.12"


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    d = tmp_path_factory.mktemp("graph_instrument")
    graph, manifest, bmap = d / "g.jsonl", d / "planted.json", d / "bmap.json"
    r = subprocess.run(
        [PY, str(ROOT / "tools/graph/fixture.py"), "--scale", "tiny", "--seed", "42",
         "--out", str(graph), "--planted-manifest", str(manifest),
         "--behaviour-map", str(bmap)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-800:]
    r = subprocess.run(
        [PY, str(ROOT / "tools/graph/hawking_analyze.py"), "--graph", str(graph),
         "--out", str(d), "--behaviour-map", str(bmap),
         "--planted-manifest", str(manifest)],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert r.returncode == 0, r.stderr[-800:]
    return {
        "manifest": json.loads(manifest.read_text(encoding="utf-8")),
        "cluster_map": json.loads((d / "HAWKING_CLUSTER_MAP.json").read_text(encoding="utf-8")),
        "candidates": json.loads(
            (d / "HAWKING_RECOMPOSITION_CANDIDATES.json").read_text(encoding="utf-8")),
        "report": json.loads((d / "HAWKING_ANALYSIS_REPORT.json").read_text(encoding="utf-8")),
    }


def test_fixture_is_deterministic(tmp_path):
    """Same seed, same bytes. Without this the analyses cannot be compared across runs."""
    out = []
    for i in range(2):
        p = tmp_path / f"g{i}.jsonl"
        r = subprocess.run(
            [PY, str(ROOT / "tools/graph/fixture.py"), "--scale", "tiny", "--seed", "7",
             "--out", str(p)],
            cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-500:]
        out.append(p.read_bytes())
    assert out[0] == out[1], "fixture is not deterministic from its seed"


def test_every_analysis_produces_output(planted):
    """A silent zero reads as 'no signal here' when it means 'the instrument is unplugged'.
    Three analyses shipped in exactly that state and had to be repaired."""
    analyses = planted["cluster_map"]["analyses"]
    expected = {"scc", "communities", "betweenness", "dominators",
                "clones", "cochange", "fanin", "behaviour_coverage"}
    assert expected <= set(analyses), sorted(expected - set(analyses))
    for name, body in analyses.items():
        assert body.get("summary"), f"{name} produced no summary"
        assert body.get("machine") is not None, f"{name} produced no machine section"


def test_planted_structures_are_found(planted):
    """The point of the whole harness: structure known to be there is detected.

    `hawking_analyze.py --planted-manifest` writes its verdict into
    HAWKING_ANALYSIS_REPORT.json. A miss is a finding about the instrument, not a reason to
    skip -- so this asserts rather than degrading when the section is absent."""
    rep = planted["report"]
    checks = (rep.get("planted_verification") or {}).get("checks")
    assert checks, ("HAWKING_ANALYSIS_REPORT.json has no planted verification section; "
                    "--planted-manifest was passed, so its absence is a defect")
    missed = [c for c in checks if not c.get("found", c.get("ok"))]
    assert not missed, ("planted structures the analyses failed to find: "
                        + ", ".join(str(c.get("what") or c.get("name") or c) for c in missed))


def test_scc_reports_the_planted_component(planted):
    scc = planted["cluster_map"]["analyses"]["scc"]["machine"]
    sizes = [c["size"] for c in scc.get("file_sccs", [])]
    assert sizes, "no file-level SCC found in a fixture that plants one"
    assert max(sizes) >= 2


def test_communities_flag_directory_scatter(planted):
    """The finding that matters on the real tree is 'this community lives in N directories'.
    If the fixture's scattered community does not register, that finding is not trustworthy."""
    com = planted["cluster_map"]["analyses"]["communities"]["machine"]
    assert com["n_communities"] >= 2
    assert max(c["n_directories"] for c in com["communities"]) >= 3


def test_clone_families_are_signature_matched_not_text_matched(planted):
    """Campaign rule: text similarity is not admissible evidence for a merge."""
    clones = planted["cluster_map"]["analyses"]["clones"]["machine"]
    fams = clones.get("families", [])
    for f in fams:
        assert f.get("match_kind", "signature_match") != "text_match", \
            f"clone family {f.get('id')} claims a text match as evidence"


def test_candidates_never_invent_a_saving(planted):
    """A candidate whose saving cannot be estimated must sort last with null, not carry a
    made-up number."""
    cands = planted["candidates"]
    cands = cands["candidates"] if isinstance(cands, dict) else cands
    for c in cands:
        loc = c.get("expected_loc_removed")
        assert loc is None or (isinstance(loc, int) and loc >= 0), \
            f"{c.get('id')} has a nonsense expected_loc_removed: {loc!r}"
        assert c.get("evidence"), f"{c.get('id')} carries no evidence"
