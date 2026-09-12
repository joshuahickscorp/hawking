"""Protected G011 verifier for the Odyssey three-stream receipt.

This module is intentionally a receipt gate, not a producer. It accepts only
the receipt emitted by ``tools/sovereign/g011_streaming.py`` and keeps the
claim narrow: HCLI-owned stream entries overlapped in recorded wall time while
an incomplete specimen was present, and the ledger contains scientific
progress rather than bookkeeping alone.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "sovereign" / "G011_odyssey_streaming.json"
PRODUCER = "tools/sovereign/g011_streaming.py"
REQUIRED_STREAMS = ("I", "II", "III")


def _load() -> dict:
    if not RECEIPT.is_file():
        raise AssertionError(f"G011 receipt is absent: {RECEIPT}")
    try:
        value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssertionError(f"G011 receipt is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise AssertionError("G011 receipt must be a JSON object")
    return value


def _assert_common(doc: dict) -> None:
    assert doc.get("schema") == "hawking.sovereign.g011_odyssey_streaming.v1"
    assert doc.get("gate") == "G011"
    assert doc.get("status") == "completed"
    assert doc.get("produced_by") == PRODUCER
    assert doc.get("command") == "python3 tools/sovereign/g011_streaming.py"


def test_producer_is_named_and_actually_ran():
    doc = _load()
    _assert_common(doc)
    assert doc.get("produced_at")
    assert doc.get("hcli_owned") is True
    ownership = (doc.get("evidence") or {}).get("ownership") or {}
    assert ownership.get("worker_pid")
    assert not ownership.get("reason")


def test_claimed_completion_is_not_accepted_alone():
    doc = _load()
    _assert_common(doc)
    streams = doc.get("streams")
    assert isinstance(streams, dict)
    assert set(REQUIRED_STREAMS).issubset(streams)
    evidence = doc.get("evidence") or {}
    ownership = evidence.get("ownership") or {}
    assert evidence.get("ledger_sha256")
    assert ownership.get("basis") or ownership.get("worker_pid")
    assert ownership.get("producer_sid") == ownership.get("worker_sid")
    for name in REQUIRED_STREAMS:
        assert all(
            isinstance(entry, dict) and entry.get("entry_count", 0) > 0
            for entry in [streams[name]]
        )


def test_the_three_streams_overlapped_in_wall_time():
    doc = _load()
    _assert_common(doc)
    streams = doc["streams"]
    pairs = {tuple(pair) for pair in doc.get("overlapping_pairs") or []}
    assert len(streams) == 3
    assert pairs == {
        ("I", "II"),
        ("I", "III"),
        ("II", "III"),
    }
    for name in REQUIRED_STREAMS:
        row = streams[name]
        assert row["start_s"] < row["end_s"]
        assert row.get("span_s") == row["end_s"] - row["start_s"]


def test_an_incomplete_specimen_did_not_block_independent_science():
    doc = _load()
    _assert_common(doc)
    incomplete = (doc.get("evidence") or {}).get("incomplete_specimens") or []
    assert incomplete, "without an incomplete specimen the claim is vacuous"
    assert doc.get("blocked_on_incomplete_specimen") is False
    assert doc.get("overlapping_pairs")


def test_progress_is_scientific_not_bookkeeping():
    doc = _load()
    _assert_common(doc)
    evidence = doc.get("evidence") or {}
    laws = int(evidence.get("law_count") or 0)
    scars = int(evidence.get("scar_count") or 0)
    assert laws + scars > 0
    assert int(doc.get("laws_or_scars_added") or 0) == laws + scars
