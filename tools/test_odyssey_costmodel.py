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


# ---------------------------------------------------------------------------
# NX / NR per-specimen accounting (operator directive 2026-09-05)
#
# NX (tools/flash_nx_genome.py) is a small machine-bound seal: source/shader
# hashes + machine genome + a pointer (sha256) at the NR it lowers. NR
# (tools/flash_complete_nr.py, tools/nr_container.py) is the portable
# representation; its JSON descriptor is small too, but a *materialized*
# NR candidate (a catalog + segments/ tree of quantized tensors) is the real,
# multi-gigabyte payload it describes -- that materialization is what "costs
# storage" and is meant to be transient scratch, disposable once superseded
# or sealed by hash into an NX.
# ---------------------------------------------------------------------------

def test_specimen_storage_measures_real_bytes_not_estimates():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        nx = root / "seal.nx.json"
        nx.write_text("{}")  # 2 bytes
        nr_desc = root / "candidate.nr.json"
        nr_desc.write_text("{}")  # 2 bytes
        payload = root / "segments"
        payload.mkdir()
        (payload / "a.hq30uq4").write_bytes(b"0" * 100)
        (payload / "b.hq30uq4").write_bytes(b"0" * 300)

        got = C.specimen_storage(nx_path=nx, nr_descriptor_path=nr_desc, nr_payload_path=payload)
        assert got["nx_bytes"] == 2
        assert got["nr_descriptor_bytes"] == 2
        assert got["nr_payload_bytes"] == 400
        assert got["nr_payload_present"] is True


def test_record_nx_nr_defaults_released_to_whether_payload_is_still_on_disk():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        ledger = root / "e.jsonl"
        nx = root / "seal.nx.json"
        nx.write_text("{}")
        payload = root / "segments"
        payload.mkdir()
        (payload / "a.hq30uq4").write_bytes(b"0" * 1000)

        rec = C.record_nx_nr_accounting(
            "O009", nx_path=nx, nr_payload_path=payload, path=ledger, ts=1.0,
            gpu={"gpu_working_set_bytes": 100, "gpu_allocated_bytes": 50,
                 "gpu_share": 0.5, "gpu_source": "test"},
        )
        assert rec["nx_bytes"] == 2
        assert rec["nr_payload_bytes"] == 1000
        assert rec["nr_released"] is False  # payload still exists; nobody said it was cleaned up
        assert rec["gpu_share"] == 0.5

        # Now the payload is gone (the specimen's search moved on).
        for f in payload.iterdir():
            f.unlink()
        payload.rmdir()
        rec2 = C.record_nx_nr_accounting("O009", nx_path=nx, nr_payload_path=payload,
                                          path=ledger, ts=2.0)
        assert rec2["nr_released"] is True
        assert rec2["nr_payload_bytes"] == 0


def test_nr_retention_violation_flagged_when_payload_outlives_the_nx_seal():
    """Transient means transient: an NX seal (nx_bytes > 0) with the NR payload
    still on disk and un-released is exactly the failure that fills the volume
    across many specimens."""
    with tempfile.TemporaryDirectory() as d:
        ledger = pathlib.Path(d) / "e.jsonl"
        C.record_nx_nr_accounting(
            "O010", path=ledger, ts=1.0, gpu={},
            nx_bytes_override=4096, nr_payload_bytes_override=2_000_000_000,
            nr_released=False,
        )
        violations = C.nr_retention_violations(path=ledger)
        assert len(violations) == 1
        assert violations[0]["patient"] == "O010"
        assert violations[0]["nr_payload_bytes"] == 2_000_000_000


def test_nr_retention_is_clean_once_released():
    with tempfile.TemporaryDirectory() as d:
        ledger = pathlib.Path(d) / "e.jsonl"
        C.record_nx_nr_accounting(
            "O011", path=ledger, ts=1.0, gpu={},
            nx_bytes_override=4096, nr_payload_bytes_override=2_000_000_000,
            nr_released=True,
        )
        assert C.nr_retention_violations(path=ledger) == []
