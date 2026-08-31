"""Tests for tools/future/wakeup.py.

Negative controls nobody has watched fail:
  - a truncated receipt does not unblock dependents (PARTIAL_TRUNCATED)
  - a seal-mismatched receipt does not unblock dependents (SEAL_MISMATCH)
  - the same completion event dispatched twice performs the downstream action once
  - a missing receipt past its deadline is a third distinct terminal
  - a forged COMPLETED event without matching disk bytes is refused
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.future import wakeup as wu
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims, write_receipt
from hcli.persist import atomic_write_json


def _hits() -> dict[str, list[str]]:
    box: dict[str, list[str]] = {k: [] for k in wu.CONSUMER_KINDS}

    def _mk(kind: str):
        def _fn(event: wu.WakeEvent) -> None:
            box[kind].append(event.state)

        return _fn

    return box, {"v": _mk("verifier"), "g": _mk("graph"), "f": _mk("frontier")}


def _bind(watcher: wu.Watcher) -> dict[str, list[str]]:
    box, consumers = _hits()
    for name, fn in consumers.items():
        watcher.register_consumer(name, fn)
    return box


def _expect(watcher: wu.Watcher, uid: str, path: Path, *dependents: str) -> str:
    return watcher.register_expectation(
        unit_id=uid,
        path=path,
        dependents=list(dependents),
        verifier="v",
        graph="g",
        frontier="f",
    )


# ---------------------------------------------------------------------------
# Entry point / receipt
# ---------------------------------------------------------------------------


def test_entry_point_runs_and_seals_receipt():
    out = wu.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RECEIPT_WAKEUP.json"
    assert doc["schema"] == "hawking.future.wakeup.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert wu.seal_is_valid(doc)
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["watch_receipts_not_processes"] is True
    assert doc["no_model_poll"] is True
    assert doc["idempotent"] is True
    assert list(doc["dispatch_wakes"]) == list(wu.CONSUMER_KINDS)
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    rc = doc["resident_callable"]
    assert rc["can_hcli_invoke"] is True
    assert rc["receipt"] == "receipts/future/RECEIPT_WAKEUP.json"
    assert rc["frontier_fed"] == wu.DEFAULT_FRONTIER
    assert rc["workunit_emitted"]
    assert doc["vocabulary"]["no_era_vi"] is True
    assert doc["vocabulary"]["no_odyssey_iv"] is True
    assert len(doc["vocabulary"]["eras"]) == 5
    assert len(doc["vocabulary"]["odysseys"]) == 3
    assert "VI" not in "".join(doc["vocabulary"]["eras"])
    assert not any(s.startswith("IV") for s in doc["vocabulary"]["odysseys"])
    proofs = doc["proofs"]
    assert proofs["truncated_does_not_unblock"] is True
    assert proofs["seal_mismatch_does_not_unblock"] is True
    assert proofs["dispatch_twice_is_once"] is True
    assert proofs["cross_process"]["fresh_watcher_dispatched"] is True
    assert proofs["fail_closed_terminals_distinct"] is True
    _assert_no_hardware_claims(doc)


def test_selftest_is_the_cli_entry(tmp_path, monkeypatch):
    # Drive main() against a throwaway proof dir so this test does not
    # depend on a second full write of RECEIPT_WAKEUP.json succeeding.
    monkeypatch.chdir(tmp_path)
    rc = wu.main(["--selftest"])
    assert rc == 0
    path = RECEIPTS / "RECEIPT_WAKEUP.json"
    assert path.is_file()
    doc = json.loads(path.read_text())
    assert doc["schema"] == wu.SCHEMA


# ---------------------------------------------------------------------------
# Classify (pure)
# ---------------------------------------------------------------------------


def test_classify_truncated_is_partial():
    state, reason = wu.classify_receipt_bytes(b'{"schema":')
    assert state == wu.PARTIAL_TRUNCATED
    assert "JSON" in reason or "truncated" in reason.lower() or "invalid" in reason.lower()


def test_classify_empty_is_partial():
    state, _ = wu.classify_receipt_bytes(b"")
    assert state == wu.PARTIAL_TRUNCATED
    state, _ = wu.classify_receipt_bytes(b"   \n")
    assert state == wu.PARTIAL_TRUNCATED


def test_classify_complete_false_is_partial_even_when_sealed():
    sealed = wu.seal_document(
        {"schema": wu.FIXTURE_SCHEMA, "complete": False, "unit": "x"}
    )
    raw = json.dumps(sealed).encode()
    state, reason = wu.classify_receipt_bytes(raw)
    assert state == wu.PARTIAL_TRUNCATED
    assert "complete" in reason


def test_classify_wrong_seal_is_mismatch():
    doc = {"schema": wu.FIXTURE_SCHEMA, "complete": True, "seal_sha256": "0" * 64}
    state, _ = wu.classify_receipt_bytes(json.dumps(doc).encode())
    assert state == wu.SEAL_MISMATCH


def test_classify_missing_seal_is_mismatch():
    doc = {"schema": wu.FIXTURE_SCHEMA, "complete": True}
    state, _ = wu.classify_receipt_bytes(json.dumps(doc).encode())
    assert state == wu.SEAL_MISMATCH


def test_classify_valid_sealed_is_completed():
    sealed = wu.seal_document({"schema": wu.FIXTURE_SCHEMA, "complete": True, "unit": "x"})
    state, _ = wu.classify_receipt_bytes(json.dumps(sealed, sort_keys=True).encode())
    assert state == wu.COMPLETED


def test_classify_is_pure_function_of_bytes():
    raw = json.dumps(wu.seal_document({"schema": wu.FIXTURE_SCHEMA, "unit": "p"})).encode()
    assert wu.classify_receipt_bytes(raw) == wu.classify_receipt_bytes(raw)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_receipt_wakes_three_consumers_and_unblocks(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "u1", path, "child")
    wu.write_sealed(path, wu._fixture("u1"))
    results = wu.run_once(w)
    assert len(results) == 1
    assert results[0].state == wu.COMPLETED
    assert not results[0].duplicate
    assert set(results[0].consumers_invoked) == set(wu.CONSUMER_KINDS)
    assert set(results[0].unblocked) == {"child"}
    assert w.graph_record("child")["state"] == wu.GRAPH_UNBLOCKED
    assert w.graph_record("child")["blocked_by"] == []
    assert all(hits[k] == [wu.COMPLETED] for k in wu.CONSUMER_KINDS)
    assert w.record("u1")["state"] == wu.COMPLETED


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: truncated / seal-mismatch / deadline do not unblock
# ---------------------------------------------------------------------------


def test_truncated_receipt_does_not_unblock_dependents(tmp_path):
    """NEGATIVE CONTROL: torn JSON is PARTIAL_TRUNCATED and must not unblock."""
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "trunc.u", path, "trunc.child")
    path.write_bytes(b'{"schema": "hawking.future.wakeup.fixture.v1", "complete":')
    results = wu.run_once(w)
    assert len(results) == 1
    assert results[0].state == wu.PARTIAL_TRUNCATED
    assert results[0].state != wu.COMPLETED
    child = w.graph_record("trunc.child")
    assert child["state"] == wu.GRAPH_BLOCKED
    assert child["state"] != wu.GRAPH_UNBLOCKED
    assert "trunc.u" in child["blocked_by"]
    assert "PARTIAL_TRUNCATED" in (child["reason"] or "")
    assert all(hits[k] == [wu.PARTIAL_TRUNCATED] for k in wu.CONSUMER_KINDS)


def test_seal_mismatch_does_not_unblock_dependents(tmp_path):
    """NEGATIVE CONTROL: a broken seal is SEAL_MISMATCH and must not unblock."""
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "seal.u", path, "seal.child")
    body = wu._fixture("seal.u")
    body["seal_sha256"] = "0" * 64
    atomic_write_json(path, body)
    results = wu.run_once(w)
    assert len(results) == 1
    assert results[0].state == wu.SEAL_MISMATCH
    assert results[0].state != wu.COMPLETED
    child = w.graph_record("seal.child")
    assert child["state"] == wu.GRAPH_BLOCKED
    assert child["state"] != wu.GRAPH_UNBLOCKED
    assert "seal.u" in child["blocked_by"]
    assert all(hits[k] == [wu.SEAL_MISMATCH] for k in wu.CONSUMER_KINDS)


def test_missing_past_deadline_does_not_unblock(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "never.json"
    w.register_expectation(
        unit_id="miss.u",
        path=path,
        dependents=["miss.child"],
        verifier="v",
        graph="g",
        frontier="f",
        deadline_tick=0,
    )
    # still waiting at the deadline
    assert wu.run_once(w, now_tick=0) == []
    results = wu.run_once(w, now_tick=1)
    assert len(results) == 1
    assert results[0].state == wu.MISSING_PAST_DEADLINE
    child = w.graph_record("miss.child")
    assert child["state"] == wu.GRAPH_BLOCKED
    assert child["state"] != wu.GRAPH_UNBLOCKED
    assert all(hits[k] == [wu.MISSING_PAST_DEADLINE] for k in wu.CONSUMER_KINDS)


def test_fail_closed_terminals_are_distinct(tmp_path):
    seen = set()
    # truncated
    w = wu.Watcher(tmp_path / "a.json")
    _bind(w)
    p = tmp_path / "a-done.json"
    _expect(w, "a", p, "ac")
    p.write_bytes(b"{")
    seen.add(wu.run_once(w)[0].state)
    # seal
    w = wu.Watcher(tmp_path / "b.json")
    _bind(w)
    p = tmp_path / "b-done.json"
    _expect(w, "b", p, "bc")
    atomic_write_json(p, {"schema": "x", "seal_sha256": "0" * 64})
    seen.add(wu.run_once(w)[0].state)
    # missing
    w = wu.Watcher(tmp_path / "c.json")
    _bind(w)
    w.register_expectation(
        unit_id="c",
        path=tmp_path / "c-done.json",
        dependents=["cc"],
        verifier="v",
        graph="g",
        frontier="f",
        deadline_tick=0,
    )
    seen.add(wu.run_once(w, now_tick=2)[0].state)
    assert seen == {wu.PARTIAL_TRUNCATED, wu.SEAL_MISMATCH, wu.MISSING_PAST_DEADLINE}


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: dispatch twice → downstream action once
# ---------------------------------------------------------------------------


def test_same_event_dispatched_twice_runs_downstream_once(tmp_path):
    """NEGATIVE CONTROL: replay must not re-fire consumers or re-unblock."""
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "id.u", path, "id.child")
    wu.write_sealed(path, wu._fixture("id.u"))
    events = w.harvest()
    assert len(events) == 1
    first = wu.dispatch(events[0], w)
    second = wu.dispatch(events[0], w)
    assert first.duplicate is False
    assert second.duplicate is True
    assert set(first.consumers_invoked) == set(wu.CONSUMER_KINDS)
    assert second.consumers_invoked == ()
    assert list(first.unblocked) == ["id.child"]
    assert second.unblocked == ()
    assert all(len(hits[k]) == 1 for k in wu.CONSUMER_KINDS)
    assert all(hits[k] == [wu.COMPLETED] for k in wu.CONSUMER_KINDS)
    # harvest + dispatch again after reconstruction
    w2 = wu.Watcher(tmp_path / "ledger.json")
    hits2 = _bind(w2)
    replay = wu.run_once(w2)
    assert replay == []
    assert all(hits2[k] == [] for k in wu.CONSUMER_KINDS)
    assert w2.graph_record("id.child")["state"] == wu.GRAPH_UNBLOCKED


# ---------------------------------------------------------------------------
# Cross-process: registrar dies, writer is a child, detector is fresh
# ---------------------------------------------------------------------------


def test_cross_process_wakeup_from_fresh_watcher(tmp_path):
    """A different process writes the receipt; a new Watcher dispatches it."""
    ledger = tmp_path / "ledger.json"
    receipt = tmp_path / "done.json"
    register_src = (
        "import sys\n"
        f"sys.path.insert(0, {str(wu.REPO)!r})\n"
        "from pathlib import Path\n"
        "from tools.future.wakeup import Watcher\n"
        f"w = Watcher(Path({str(ledger)!r}))\n"
        "w.register_expectation(\n"
        "    unit_id='cross.u',\n"
        f"    path=Path({str(receipt)!r}),\n"
        "    dependents=['cross.child'],\n"
        "    verifier='v', graph='g', frontier='f',\n"
        ")\n"
        "print('registered')\n"
    )
    write_src = (
        "import sys\n"
        f"sys.path.insert(0, {str(wu.REPO)!r})\n"
        "from pathlib import Path\n"
        "from tools.future.wakeup import write_sealed\n"
        f"write_sealed(Path({str(receipt)!r}), "
        "{'schema': 'hawking.future.wakeup.fixture.v1', 'complete': True, "
        "'unit': 'cross.u', 'evidence_class': 'STATIC_ONLY', "
        "'bench_state': 'UNKNOWN', 'gpu_authority': False})\n"
        "print('wrote')\n"
    )
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(wu.REPO) + (os.pathsep + pp if pp else "")

    r1 = subprocess.run(
        [sys.executable, "-c", register_src],
        cwd=str(wu.REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(
        [sys.executable, "-c", write_src],
        cwd=str(wu.REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r2.returncode == 0, r2.stderr

    detector = wu.Watcher(ledger)
    hits = _bind(detector)
    results = wu.run_once(detector)
    assert len(results) == 1
    assert results[0].state == wu.COMPLETED
    assert not results[0].duplicate
    assert detector.graph_record("cross.child")["state"] == wu.GRAPH_UNBLOCKED
    assert all(hits[k] == [wu.COMPLETED] for k in wu.CONSUMER_KINDS)
    # no in-memory state was shared: the detector was constructed after both children exited


# ---------------------------------------------------------------------------
# SLEEPING / synthetic / mtime / ledger / poll / readonly
# ---------------------------------------------------------------------------


def test_sleeping_unit_does_not_complete_without_a_receipt(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    _bind(w)
    w.register_expectation(
        unit_id="sleep.u",
        path=tmp_path / "qual.json",
        dependents=["sleep.child"],
        verifier="v",
        graph="g",
        frontier="f",
        sleeping=True,
    )
    assert wu.run_once(w, now_tick=10**6) == []
    assert w.record("sleep.u")["state"] == wu.SLEEPING
    assert w.graph_record("sleep.child")["state"] == wu.GRAPH_BLOCKED


def test_synthetic_completed_without_disk_bytes_is_refused(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    _bind(w)
    path = tmp_path / "qual.json"
    w.register_expectation(
        unit_id="sleep.u",
        path=path,
        dependents=["sleep.child"],
        verifier="v",
        graph="g",
        frontier="f",
        sleeping=True,
    )
    forged = wu.WakeEvent(
        event_id=wu.event_id_for(
            unit_id="sleep.u",
            path=str(path),
            state=wu.COMPLETED,
            content_sha256="ab" * 32,
        ),
        unit_id="sleep.u",
        path=str(path),
        state=wu.COMPLETED,
        content_sha256="ab" * 32,
        verifier="v",
        graph="g",
        frontier="f",
        dependents=("sleep.child",),
        reason="forged",
    )
    with pytest.raises(wu.FailClosed) as ei:
        w.dispatch(forged)
    assert ei.value.fault == "synthetic_completion"
    assert w.graph_record("sleep.child")["state"] == wu.GRAPH_BLOCKED


def test_mtime_alone_is_not_a_completion(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "mtime.u", path)
    wu.write_sealed(path, wu._fixture("mtime.u"))
    first = wu.run_once(w)
    assert len(first) == 1
    assert first[0].state == wu.COMPLETED
    os.utime(path, (0, 0))
    assert wu.run_once(w) == []


def test_content_hash_change_is_a_new_event(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "flip.u", path, "flip.child")
    wu.write_sealed(path, wu._fixture("flip.u") | {"n": 1})
    first = wu.run_once(w)
    assert first[0].state == wu.COMPLETED
    wu.write_sealed(path, wu._fixture("flip.u") | {"n": 2})
    second = wu.run_once(w)
    assert len(second) == 1
    assert second[0].state == wu.COMPLETED
    assert second[0].event_id != first[0].event_id
    assert hits["verifier"] == [wu.COMPLETED, wu.COMPLETED]


def test_corrupt_ledger_fails_closed(tmp_path):
    ledger = tmp_path / "ledger.json"
    w = wu.Watcher(ledger)
    _expect(w, "lg.u", tmp_path / "x.json")
    ledger.write_bytes(b'{"schema": "hawking.future.wakeup.ledger.v1"')
    with pytest.raises(wu.FailClosed) as ei:
        wu.Watcher(ledger)
    assert ei.value.fault == "corrupt_ledger"


def test_checksum_mismatch_fails_closed(tmp_path):
    ledger = tmp_path / "ledger.json"
    w = wu.Watcher(ledger)
    _expect(w, "lg.u", tmp_path / "x.json")
    doc = json.loads(ledger.read_text())
    doc["expectations"]["lg.u"]["unit_id"] = "tampered"
    # leave the old checksum
    ledger.write_text(json.dumps(doc))
    with pytest.raises(wu.FailClosed) as ei:
        wu.Watcher(ledger)
    assert ei.value.fault == "corrupt_ledger"


def test_no_poll_api_on_module_or_watcher(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    for name in wu.POLL_ALIASES:
        with pytest.raises(AttributeError) as ei:
            getattr(wu, name)
        assert "poll" in str(ei.value).lower() or "dispatch" in str(ei.value).lower()
        with pytest.raises(AttributeError):
            getattr(w, name)


def test_harvest_opens_receipts_read_only(tmp_path, monkeypatch):
    w = wu.Watcher(tmp_path / "ledger.json")
    path = tmp_path / "done.json"
    _expect(w, "ro.u", path)
    wu.write_sealed(path, wu._fixture("ro.u"))
    opens: list[int] = []
    real_open = os.open

    def spy_open(p, flags, *args, **kwargs):
        if str(p) == str(path):
            opens.append(flags)
            assert (flags & os.O_ACCMODE) == os.O_RDONLY
            assert not (flags & os.O_WRONLY)
            assert not (flags & os.O_RDWR)
            assert not (flags & os.O_CREAT)
            assert not (flags & os.O_TRUNC)
        return real_open(p, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)
    wu.run_once(w)
    assert opens, "expected the completion receipt to be opened"


def test_notify_harvests_only_the_named_path(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    _bind(w)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _expect(w, "a", a, "ac")
    _expect(w, "b", b, "bc")
    wu.write_sealed(a, wu._fixture("a"))
    wu.write_sealed(b, wu._fixture("b"))
    events = w.notify(a)
    assert [e.unit_id for e in events] == ["a"]
    wu.dispatch(events[0], w)
    assert w.graph_record("ac")["state"] == wu.GRAPH_UNBLOCKED
    assert w.graph_record("bc")["state"] == wu.GRAPH_BLOCKED


def test_later_valid_receipt_after_truncated_is_a_new_event(tmp_path):
    w = wu.Watcher(tmp_path / "ledger.json")
    hits = _bind(w)
    path = tmp_path / "done.json"
    _expect(w, "rec.u", path, "rec.child")
    path.write_bytes(b'{"schema":')
    bad = wu.run_once(w)
    assert bad[0].state == wu.PARTIAL_TRUNCATED
    assert w.graph_record("rec.child")["state"] == wu.GRAPH_BLOCKED
    wu.write_sealed(path, wu._fixture("rec.u"))
    good = wu.run_once(w)
    assert good[0].state == wu.COMPLETED
    assert good[0].event_id != bad[0].event_id
    assert w.graph_record("rec.child")["state"] == wu.GRAPH_UNBLOCKED
    assert hits["verifier"] == [wu.PARTIAL_TRUNCATED, wu.COMPLETED]


def test_module_receipt_refuses_hardware_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.future._common.RECEIPTS", tmp_path)
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "WAKEUP_HW_GUARD.json",
            {"schema": "test", "version": 1, "tps": 12.5},
            "test",
        )


def test_emitted_workunits_round_trip_hcli():
    units = wu.emit_wakeup_workunits()
    assert units
    ids = [u["id"] for u in units]
    assert "future.wakeup.dispatch-on-receipt" in ids
    sleeping = [u for u in units if u.get("wakeup_state") == wu.SLEEPING]
    assert sleeping
    assert all(u["status"] == "blocked" for u in sleeping)
    pending = [u for u in units if u.get("wakeup_state") == wu.WAITING]
    assert pending
    assert all(u["classification"] == "STATIC_ONLY" for u in pending)
    assert all(u.get("may_promote") in (None, False) for u in units)
