#!/usr/bin/env python3.12
"""Controller-spine tests: event log, state machine, queue, engine, audit (succ_*)."""
import json
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_events as ev  # noqa: E402
import succ_state as st  # noqa: E402
import succ_queue as q  # noqa: E402
import succ_engine as eng  # noqa: E402


# -- selftests (each module's own battery) --------------------------------------------
def test_events_selftest():
    assert ev.selftest()["ok"] is True


def test_state_selftest():
    assert st.selftest()["ok"] is True


def test_queue_selftest():
    assert q.selftest()["ok"] is True


def test_engine_selftest():
    assert eng.selftest()["ok"] is True


# -- event log ------------------------------------------------------------------------
def test_event_chain_and_resume(tmp_path):
    log = ev.EventLog(tmp_path / "e.jsonl")
    log.append("boot", {"g": 1})
    log.append("audit", {"n": 187})
    ok, why = log.verify_chain()
    assert ok, why
    assert ev.EventLog(tmp_path / "e.jsonl").next_seq() == 2


def test_event_tamper_detected(tmp_path):
    log = ev.EventLog(tmp_path / "e.jsonl")
    log.append("a", {"x": 1})
    log.append("b", {"x": 2})
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    doc = json.loads(lines[0]); doc["payload"] = {"x": 999}
    lines[0] = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    (tmp_path / "e.jsonl").write_text("\n".join(lines) + "\n")
    ok, _ = ev.EventLog(tmp_path / "e.jsonl").verify_chain()
    assert ok is False


# -- state machine --------------------------------------------------------------------
def test_illegal_transition_refused(tmp_path):
    c = st.Controller(tmp_path / "g", generation="g")
    c.boot()
    c.transition("AUDIT")
    with pytest.raises(st.StateError, match="illegal transition"):
        c.transition("LAUNCH")


def test_checkpoint_tamper_refused(tmp_path):
    # tampering any checkpoint field breaks its self-seal (caught before resume proceeds)
    c = st.Controller(tmp_path / "g", generation="g")
    c.boot(); c.transition("AUDIT"); c.transition("WAIT_OLD_RELEASE")
    cp = json.loads((tmp_path / "g" / "checkpoint.json").read_text())
    cp["event_head_hash"] = "b" * 64
    (tmp_path / "g" / "checkpoint.json").write_text(json.dumps(cp))
    with pytest.raises(st.StateError, match="self-seal invalid"):
        st.Controller(tmp_path / "g", generation="g").resume()


def test_split_brain_resume_refused(tmp_path):
    # genuine split-brain: checkpoint stays valid but the event log head advances past it
    c = st.Controller(tmp_path / "g", generation="g")
    c.boot(); c.transition("AUDIT"); c.transition("WAIT_OLD_RELEASE")
    # append a raw event directly to the log WITHOUT re-checkpointing -> head diverges
    c.log.append("state", {"state": "IMPORT_LEGACY", "from": "WAIT_OLD_RELEASE"})
    with pytest.raises(st.StateError, match="ambiguous resume"):
        st.Controller(tmp_path / "g", generation="g").resume()


def test_full_lifecycle_waits_then_imports(tmp_path):
    c = st.Controller(tmp_path / "g", generation="g")
    c.boot()
    c.transition("AUDIT")
    c.transition("WAIT_OLD_RELEASE", {"reason": "legacy running"})
    for _ in range(3):
        c.transition("WAIT_OLD_RELEASE", {"heartbeat": 1})  # heartbeat loop
    c.transition("IMPORT_LEGACY")
    assert c.current_state() == "IMPORT_LEGACY"
    assert st.Controller(tmp_path / "g", generation="g").resume()["resumed_state"] == "IMPORT_LEGACY"


# -- queue ----------------------------------------------------------------------------
def test_default_rows_are_honestly_blocked(tmp_path):
    queue = q.Queue(tmp_path / "queue")
    for row in q.build_default_rows():
        queue.upsert(row)
    summ = q.Queue(tmp_path / "queue").summary()
    assert summ["by_status"] == {"72B": "waiting_old_release", "120B": "waiting_adapter",
                                 "671B": "waiting_source_authority"}
    # every blocked row carries concrete blockers + exit criteria
    for row in q.Queue(tmp_path / "queue").rows():
        assert row["blockers"], f"{row['parent_label']} has no blockers"
        assert row["exit_criteria"], f"{row['parent_label']} has no exit criteria"
        assert row["prior_is_evidence"] is False


def test_queue_row_tamper_detected_on_reload(tmp_path):
    queue = q.Queue(tmp_path / "queue")
    for row in q.build_default_rows():
        queue.upsert(row)
    doc = json.loads((tmp_path / "queue" / "queue.json").read_text())
    doc["rows"]["671B"]["source_bytes"] = 1
    (tmp_path / "queue" / "queue.json").write_text(json.dumps(doc))
    with pytest.raises(q.QueueError, match="self-seal invalid"):
        q.Queue(tmp_path / "queue").load()


def test_invalid_status_refused():
    with pytest.raises(q.QueueError, match="invalid"):
        q.make_row(parent_label="x", current_status="done_lol")


# -- engine ---------------------------------------------------------------------------
def test_unbound_program_fails_validation():
    admission = {"adapter_id": "a", "adapter_source_sha256": "a" * 64,
                 "ready_for_execution": True, "blockers": []}
    exp = {"selected": {"model_label": "7B", "rate_bpw": 2.0, "feasibility_tier": "F3",
                        "doctor_program": {}}}
    prog = eng.materialize_program(exp, admission, source_manifest_sha256=None)
    ok, reasons = eng.validate_program(prog, admission)
    assert ok is False
    assert any("source-bound" in r for r in reasons)


def test_heavy_dispatch_is_gated():
    prog = eng.seal_field({"schema": eng.PROGRAM_SCHEMA, "adapter_id": "a"}, "program_sha256")
    with pytest.raises(eng.EngineError, match="non-lightweight"):
        eng.dispatch_lightweight(prog, "adapter.py", "run", runner=lambda a: {})
