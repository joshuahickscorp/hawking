"""The compile-economics ledger recorded 9,573 events and zero seconds.

Every call site in tools/odyssey_ctl.py records a LAUNCH marker and passes
wall_s=0.0, which is the correct value -- a start has no duration yet. What was
wrong was the label: record() stamped "_evidence": "MEASURED" on all of them, so
a ledger spanning 72 hours of timestamps asserted 9,573 measurements of zero
seconds, and any cost model fitted on it would fit nothing while looking
well-populated.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from tools import odyssey_costmodel as C

REPO = pathlib.Path(__file__).resolve().parents[1]
ECONOMICS = REPO / "workspace" / "campaign" / "odyssey" / "COMPILE_ECONOMICS.jsonl"


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_zero_wall_is_not_stamped_measured():
    """The load-bearing guard."""
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "e.jsonl"
        assert C.record("O001", "cpu", 0.0, path=f, ts=1.0)["_evidence"] == "UNRECORDED"
        assert C.record("O001", "cpu", 12.5, path=f, ts=2.0)["_evidence"] == "MEASURED"


def test_an_explicit_evidence_label_still_wins():
    """The guard must not overwrite a caller that knows what it recorded."""
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "e.jsonl"
        rec = C.record("O001", "cpu", 0.0, path=f, ts=1.0, extra={"_evidence": "DERIVED"})
        assert rec["_evidence"] == "DERIVED"


def test_a_non_numeric_wall_is_refused_outright():
    """Recording an unparseable duration as if it were a number is worse than
    recording nothing, so record() already refuses. Keep it refusing."""
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "e.jsonl"
        with pytest.raises(ValueError):
            C.record("O001", "cpu", "quickly", path=f, ts=1.0)
        with pytest.raises(ValueError):
            C.record("O001", "cpu", None, path=f, ts=1.0)
        assert not f.exists(), "a refused event still touched the ledger"


def test_the_committed_ledger_contains_no_measured_duration():
    """Documents the state this guard was written for, and fails when it changes.

    This is not an assertion that zero wall is acceptable -- it is the record
    that no Odyssey wall time has ever been measured, so nothing may claim a
    projected campaign wall derived from this ledger. When instrumentation
    starts producing real durations this test fails, and that failure is the
    signal to rewrite it against real data.
    """
    if not ECONOMICS.is_file():
        pytest.skip("no compile-economics ledger on this machine")
    rows = _rows(ECONOMICS)
    assert rows, "ledger exists but is empty"
    measured = [r for r in rows if float(r.get("wall_s") or 0.0) > 0]
    assert not measured, (
        f"{len(measured)} of {len(rows)} events now carry a real duration -- "
        "the ledger has become measurable and this test must be rewritten to "
        "check the durations instead of their absence"
    )


def test_no_event_in_the_committed_ledger_claims_measured_against_zero():
    """After a regeneration, the corpus must not re-acquire the false stamp.

    The existing rows are left as they are: they are another campaign's history
    and rewriting them would be correcting the artifact instead of the producer.
    This checks the shape a NEW row must have.
    """
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "e.jsonl"
        for event in ("cpu", "grok", "acquisition", "retirement"):
            C.record("O006", event, 0.0, path=f, ts=1.0)
        for row in _rows(f):
            assert row["_evidence"] != "MEASURED", row
