"""The G011 producer must refuse far more often than it writes.

The interesting assertions here are the refusals: every one of them is a way a
receipt could have been a lie. The single happy-path test feeds the producer's
own output straight into the protected gate's assertions, so "it passes" is
checked against the real verifier rather than against my reading of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sovereign import g011_streaming as g011

T0 = 1_788_000_000.0


def _iso(offset: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def _entry(offset: float, **extra) -> dict:
    return {"recorded_at": _iso(offset), "hcli_owned": True, **extra}


def _overlapping_ledger() -> dict:
    """I runs 0-300s, II starts at 100s, III at 150s: no barrier between them."""
    return {
        "laws": [_entry(0, id="LAW001"), _entry(300, id="LAW002")],
        "transfer_probes": [_entry(100, id="TP001"), _entry(400, id="TP002")],
        "adversarial_probes": [_entry(150, id="AP001"), _entry(500, id="AP002")],
        "scars": [_entry(450, law_id="LAW001")],
    }


def _serial_ledger() -> dict:
    """The forbidden shape: each stream finishes before the next one starts."""
    return {
        "laws": [_entry(0), _entry(100)],
        "transfer_probes": [_entry(200), _entry(300)],
        "adversarial_probes": [_entry(400), _entry(500)],
        "scars": [],
    }


STATE = {"patients": [{"oxx": "O003", "on_disk": True}, {"oxx": "O010", "on_disk": False}]}
OWNED = {"reason": None, "worker_pid": 5953}


def _assess(ledger, state=STATE, owned=True):
    return g011.assess(ledger, state, owned, OWNED, "sha-of-the-fixture-ledger")


def test_overlap_is_computed_from_the_entries_own_timestamps():
    spans, problems = g011.stream_spans(_overlapping_ledger())
    assert not problems
    assert spans["I"]["start_s"] == T0 and spans["I"]["end_s"] == T0 + 300
    assert g011.overlapping_pairs(spans) == [["I", "II"], ["I", "III"], ["II", "III"]]


def test_strictly_serial_streams_are_refused_as_a_global_barrier():
    doc, reasons = _assess(_serial_ledger())
    assert doc is None
    assert any("global barrier" in r for r in reasons)


def test_a_ledger_written_by_hand_is_refused():
    """Entries with no hcli_owned stamp were not written by the resident."""
    ledger = _overlapping_ledger()
    for entry in ledger["laws"]:
        entry.pop("hcli_owned")
    doc, reasons = _assess(ledger)
    assert doc is None
    assert any("not stamped" in r for r in reasons)


def test_a_shell_run_can_never_mint_hcli_owned():
    doc, reasons = _assess(_overlapping_ledger(), owned=False)
    assert doc is None
    assert any("hcli_owned cannot be earned" in r for r in reasons)


def test_bookkeeping_without_laws_or_scars_is_refused():
    ledger = _overlapping_ledger()
    ledger["laws"], ledger["scars"] = [], []
    doc, reasons = _assess(ledger)
    assert doc is None
    assert any("not scientific movement" in r for r in reasons)


def test_a_missing_stream_is_refused():
    ledger = _overlapping_ledger()
    ledger["adversarial_probes"] = []
    doc, reasons = _assess(ledger)
    assert doc is None
    assert any("stream III" in r for r in reasons)


def test_all_specimens_complete_makes_the_claim_vacuous_not_true():
    doc, reasons = _assess(_overlapping_ledger(), state={"patients": [{"oxx": "O003", "on_disk": True}]})
    assert doc is None
    assert any("vacuous" in r for r in reasons)


def test_an_absent_ledger_writes_nothing():
    doc, reasons = _assess(None)
    assert doc is None and reasons


def test_the_producers_output_satisfies_the_protected_gate(tmp_path, monkeypatch):
    doc, reasons = _assess(_overlapping_ledger())
    assert reasons == [] and doc is not None
    receipt = tmp_path / "G011_odyssey_streaming.json"
    receipt.write_text(json.dumps(doc), encoding="utf-8")

    from tools.odyssey import test_odyssey_streaming_runtime as gate

    monkeypatch.setattr(gate, "RECEIPT", Path(receipt))
    gate.test_producer_is_named_and_actually_ran()
    gate.test_claimed_completion_is_not_accepted_alone()
    gate.test_the_three_streams_overlapped_in_wall_time()
    gate.test_an_incomplete_specimen_did_not_block_independent_science()
    gate.test_progress_is_scientific_not_bookkeeping()


def test_the_gate_still_fails_on_the_serial_shape(tmp_path, monkeypatch):
    """Negative control: the gate is not vacuously green on any receipt."""
    doc, _ = _assess(_overlapping_ledger())
    doc["streams"]["II"] = {"start_s": T0 + 1000, "end_s": T0 + 1100, "ledger_key": "transfer_probes"}
    doc["streams"]["III"] = {"start_s": T0 + 2000, "end_s": T0 + 2100, "ledger_key": "adversarial_probes"}
    doc["streams"]["I"] = {"start_s": T0, "end_s": T0 + 100, "ledger_key": "laws"}
    receipt = tmp_path / "G011_odyssey_streaming.json"
    receipt.write_text(json.dumps(doc), encoding="utf-8")

    from tools.odyssey import test_odyssey_streaming_runtime as gate

    monkeypatch.setattr(gate, "RECEIPT", Path(receipt))
    with pytest.raises(AssertionError):
        gate.test_the_three_streams_overlapped_in_wall_time()
