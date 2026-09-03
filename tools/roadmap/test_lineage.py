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


def test_the_preserved_copy_is_still_the_document_preservation_md_recorded():
    """PRESERVATION.md records the digest; the copy must still match it."""
    assert lineage.PRESERVED.is_file(), f"lineage copy missing: {lineage.PRESERVED}"
    assert lineage.preserved_is_intact(), (
        "the preserved roadmap no longer matches its recorded sha256; the lineage "
        "record and the file have diverged and one of them is wrong"
    )


def test_the_recorded_digest_is_the_one_preservation_md_documents():
    """The constant is not free to drift away from the preservation record."""
    text = (REPO / "docs" / "roadmap-lineage" / "PRESERVATION.md").read_text()
    assert lineage.PRESERVED_SHA256 in text, (
        "lineage.PRESERVED_SHA256 does not appear in PRESERVATION.md, so the "
        "resolver is verifying against a digest no record vouches for"
    )


def test_a_digest_mismatch_refuses_the_preserved_copy(monkeypatch, tmp_path):
    """NEGATIVE CONTROL for the digest check.

    An EARLIER 9028-line roadmap exists on /Volumes/corpdrive. Acceptance spans
    are LINE RANGES, so parsing the wrong-length document quotes the wrong text
    while looking perfectly well formed. Break the digest and the resolver must
    refuse rather than fall through.
    """
    monkeypatch.setattr(lineage, "EXTERNAL", tmp_path / "absent-H-ROADMAP.md")
    monkeypatch.delenv("H_ROADMAP", raising=False)
    monkeypatch.setattr(lineage, "PRESERVED_SHA256", "0" * 64)
    with pytest.raises(FileNotFoundError) as err:
        lineage.roadmap_path()
    assert "digest" in str(err.value)


def test_the_resolver_falls_back_when_the_external_roadmap_is_absent(monkeypatch, tmp_path):
    """The regression itself: no ~/Downloads copy must not mean no roadmap."""
    monkeypatch.setattr(lineage, "EXTERNAL", tmp_path / "absent-H-ROADMAP.md")
    monkeypatch.delenv("H_ROADMAP", raising=False)
    assert lineage.roadmap_path() == lineage.PRESERVED


def test_the_external_roadmap_still_wins_when_it_is_present(monkeypatch, tmp_path):
    """The operator's copy is the authority; lineage is only the fallback."""
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
    monkeypatch.setattr(lineage, "PRESERVED", tmp_path / "also-absent.md")
    monkeypatch.delenv("H_ROADMAP", raising=False)
    with pytest.raises(FileNotFoundError):
        lineage.roadmap_lines()


def test_quote_span_returns_the_real_text_not_a_placeholder():
    quoted = lineage.quote_span(1, 3)
    assert quoted.strip(), "quote_span produced nothing"
    assert "not readable" not in quoted.lower()


def test_parse_roadmap_succeeds_with_no_external_roadmap(monkeypatch, tmp_path):
    """End to end: the whole capability graph parses off the lineage copy.

    83 gates and 25 genes is the census the frozen graph recorded, so this also
    proves the preserved copy is the same authority the catalog was built from.
    """
    monkeypatch.setattr(lineage, "EXTERNAL", tmp_path / "absent-H-ROADMAP.md")
    monkeypatch.delenv("H_ROADMAP", raising=False)
    parsed = parse_roadmap()
    assert len(parsed["gates"]) == 83
    assert len(parsed["genes"]) == 25


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


def test_the_lineage_copy_reproduces_every_stored_criterion():
    """The strongest available proof that the substitution is sound.

    A matching sha256 says the bytes are the same file. This says something
    harder: re-quoting each receipt's OWN line span out of the resolved roadmap
    reproduces the criterion text that receipt stored when it could still read
    ~/Downloads/H-ROADMAP.md. Line numbering, span semantics and content all
    have to agree, through the real quoting path, at the real spans.

    Receipts prepend a one-line summary header to the span, so the span must be
    CONTAINED in the stored quote rather than equal to it.
    """
    lines = lineage.roadmap_lines()
    checked = 0
    for path in sorted((REPO / "receipts" / "acceptance").glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        src = doc.get("criterion_source")
        quoted = doc.get("criterion_quoted")
        if not (isinstance(src, dict) and isinstance(quoted, str) and quoted.strip()):
            continue
        start, end = src.get("start_line"), src.get("end_line")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        span = "\n".join(lines[start - 1:end]).strip()
        assert span, f"{path.stem}: span {start}-{end} is empty in the resolved roadmap"
        assert span in quoted, (
            f"{path.stem}: re-quoting {start}-{end} from {lineage.roadmap_path().name} "
            "does not reproduce the criterion this receipt stored"
        )
        checked += 1
    assert checked >= 8, f"only {checked} receipts carry a re-checkable criterion span"


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
