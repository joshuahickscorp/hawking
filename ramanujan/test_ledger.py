"""The Ledger's law is only worth having if breaking it fails loudly."""
from __future__ import annotations

import json

import pytest

from ramanujan.ledger import GENESIS, KINDS, Ledger, LedgerError


@pytest.fixture()
def led(tmp_path):
    return Ledger(tmp_path / "L.jsonl")


def test_empty_ledger_verifies(led):
    assert led.rows() == []
    assert led.verify() == 0


def test_append_chains_from_genesis(led):
    r = led.append(kind="claim", role="researcher", payload={"x": 1}, id="c1")
    assert r.seq == 0 and r.prev_hash == GENESIS
    r2 = led.append(kind="objection", role="critic", payload={}, id="o1", parents=["c1"])
    assert r2.prev_hash == r.chain_sha256, "row 1 must chain to row 0"
    assert led.verify() == 2


def test_supersession_keeps_history_and_changes_the_current_view(led):
    led.append(kind="claim", role="researcher", payload={"x": 1}, id="c1")
    led.append(kind="claim", role="researcher", payload={"x": 2}, id="c2", supersedes="c1")
    assert len(led.rows()) == 2, "the superseded row must remain recorded"
    assert {r["id"] for r in led.current()} == {"c2"}


def test_superseding_an_unknown_row_is_refused(led):
    with pytest.raises(LedgerError, match="no such row"):
        led.append(kind="claim", role="r", payload={}, id="c1", supersedes="ghost")


def test_unknown_kind_is_refused(led):
    with pytest.raises(LedgerError, match="unknown event kind"):
        led.append(kind="vibes", role="r", payload={}, id="x")


def test_every_contract_kind_is_accepted(led):
    for i, kind in enumerate(sorted(KINDS)):
        led.append(kind=kind, role="r", payload={"i": i}, id=f"k{i}")
    assert led.verify() == len(KINDS)


def test_duplicate_id_is_refused(led):
    led.append(kind="claim", role="r", payload={}, id="c1")
    with pytest.raises(LedgerError, match="already recorded"):
        led.append(kind="claim", role="r", payload={}, id="c1")


def test_editing_a_row_breaks_verification(led):
    led.append(kind="claim", role="r", payload={}, id="c1")
    led.append(kind="claim", role="r", payload={}, id="c2")
    lines = led.path.read_text().splitlines()
    row = json.loads(lines[0])
    row["role"] = "tamperer"
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    led.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError, match="hash does not match"):
        led.verify()


def test_deleting_a_row_breaks_verification(led):
    """The case an edit-only check would miss: removing a row entirely."""
    for i in range(3):
        led.append(kind="claim", role="r", payload={"i": i}, id=f"c{i}")
    lines = led.path.read_text().splitlines()
    del lines[1]
    led.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError, match="sequence break|chain break"):
        led.verify()


def test_truncated_write_is_detected(led):
    led.append(kind="claim", role="r", payload={}, id="c1")
    led.path.write_text(led.path.read_text().rstrip("\n"))
    with pytest.raises(LedgerError, match="torn tail"):
        led.verify()
