"""The canonical roadmap vanished and twelve modules kept pointing at it.

These tests exist because the failure was silent in three different ways at once:
`parse_roadmap` raised (loud, but it took 17 graph-invariant tests down with it as
collection errors nobody read), `recompile.render` substituted an empty file (so
all 83 gates printed no defining property), and four acceptance harnesses returned
a placeholder string that a receipt would store as the criterion it swears it did
not alter.

Every check below has a negative control: if the load-bearing line is reverted the
test must FAIL, or it is not evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.roadmap import lineage
from tools.roadmap.auditor import _criterion_is_real
from tools.roadmap.parse import parse_roadmap

REPO = lineage.REPO


def test_the_recorded_digest_is_the_one_preservation_md_documents():
    """The historical identity remains recorded after the duplicate is removed."""
    text = (REPO / "docs" / "roadmap-lineage" / "PRESERVATION.md").read_text()
    assert "d43a6b07ab9590bc11c265bfe8a1466131cce291b0622c076370a01d811328e4" in text


def test_the_external_roadmap_still_wins_when_it_is_present(monkeypatch, tmp_path):
    """The operator's copy is the authority; history is not a fallback."""
    external = tmp_path / "H-ROADMAP.md"
    external.write_text("# whatever the operator put there\n")
    monkeypatch.setattr(lineage, "EXTERNAL", external)
    monkeypatch.delenv("H_ROADMAP", raising=False)
    assert lineage.roadmap_path() == external


def test_the_env_override_wins_over_both(monkeypatch, tmp_path):
    override = tmp_path / "override.md"
    override.write_text("x\n")
    monkeypatch.setenv("H_ROADMAP", str(override))
    assert lineage.roadmap_path() == override


def test_roadmap_lines_raises_rather_than_returning_an_empty_file(monkeypatch, tmp_path):
    """recompile.render used `[] if not is_file()`, so a missing roadmap read as
    a roadmap with no content and every defining_property came out empty."""
    monkeypatch.setattr(lineage, "EXTERNAL", tmp_path / "absent.md")
    monkeypatch.delenv("H_ROADMAP", raising=False)
    with pytest.raises(FileNotFoundError):
        lineage.roadmap_lines()


def test_quote_span_returns_the_real_text_not_a_placeholder():
    quoted = lineage.quote_span(1, 3)
    assert quoted.strip(), "quote_span produced nothing"
    assert "not readable" not in quoted.lower()


def test_a_placeholder_criterion_is_not_accepted():
    """criterion_altered=false says nothing when the criterion is a placeholder."""
    assert _criterion_is_real({"criterion_quoted": "Cancellation writes a durable state."}) == ""
    assert _criterion_is_real({"quote": "Repair depth is bounded structurally."}) == ""
    assert _criterion_is_real({"criterion": {"quoted": "Orphan jobs are adopted."}}) == ""
    assert _criterion_is_real({}) != ""
    assert _criterion_is_real({"criterion_quoted": "   "}) != ""
    assert _criterion_is_real(
        {"criterion_quoted": "<H-ROADMAP.md not readable at /Users/x/H-ROADMAP.md>"}
    ) != ""
    assert _criterion_is_real(
        {"criterion_quoted": "(roadmap missing at /Users/x/H-ROADMAP.md; span 10-20)"}
    ) != ""


def test_every_claiming_acceptance_receipt_quotes_a_real_criterion():
    """The corpus must not already contain a placeholder-criterion acceptance."""
    bad = []
    for path in sorted((REPO / "receipts" / "acceptance").glob("*.json")):
        if "." in path.stem:          # .gate/.run/.cycle sidecars are not verdicts
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or str(doc.get("verdict", "")).upper() != "ACCEPTED":
            continue
        why = _criterion_is_real(doc)
        if why:
            bad.append(f"{path.stem}: {why}")
    assert not bad, "ACCEPTED receipts with no real criterion: " + "; ".join(bad)


def test_build_state_refuses_to_clobber_a_foreign_schema():
    """Two generators write civilization/ROADMAP_STATE.json with incompatible
    schemas and build_state.py parses no arguments, so any invocation used to
    destroy whichever ledger was on disk. This is that guard, run for real."""
    state = REPO / "civilization" / "ROADMAP_STATE.json"
    before = state.read_bytes()
    proc = subprocess.run(
        [sys.executable, "civilization/build_state.py"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert state.read_bytes() == before, "build_state.py overwrote a v3 ledger"
    assert proc.returncode != 0
    assert "refusing to overwrite" in (proc.stderr + proc.stdout)


def test_there_is_exactly_one_blocker_classifier():
    """PART II and ROADMAP_STATE.json disagreed about 12 of 83 gates.

    recompile.py defined its own five-class blocker_class while importing the
    eight-class one from blockers.py for the machine-readable state. The two
    authorities disagreed in exactly the ways blockers.py's docstring says the
    old classes caused: seven THEIA programs nobody has started filed as
    "gather long-run evidence", three VMCP gates waiting on a browser install
    filed the same way, and unwritten code filed as UNKNOWN_RESEARCH.
    """
    from tools.roadmap import recompile
    from tools.roadmap.blockers import CLASSES, classify
    graph = json.loads((REPO / "civilization" / "CAPABILITY_GRAPH.json").read_text())
    disagree = [
        gid for gid, gate in graph["gates"].items()
        if recompile.blocker_class(gate)[0] != classify(gate)[0]
    ]
    assert not disagree, f"two classifiers disagree about {disagree}"
    assert tuple(recompile.BLOCKER_CLASSES) == tuple(CLASSES), (
        "PART II renders a different class vocabulary than the state file"
    )


def test_a_missing_verifier_is_not_filed_as_a_missing_caller():
    """VERIFIER_MISSING exists because those are different repairs.

    VMCP_COMPACT_SURFACE has three real non-test callers and a passed acceptance,
    and was still listed under "no non-test call site reaches this capability" --
    sending an operator to hunt for a caller that already exists three times.
    """
    from tools.roadmap.blockers import classify
    wired_unverified = {
        "id": "X", "status": "BUILT", "code_refs": [{"file": "a.py"}], "tests": [],
        "wired": {"value": True}, "accepted": {"value": True},
    }
    cls, missing = classify(wired_unverified)
    assert cls == "VERIFIER_MISSING", cls
    assert "verifies" in missing

    unwired = dict(wired_unverified, wired={"value": False})
    assert classify(unwired)[0] == "SOFTWARE_CONNECTION_REMAINING"
