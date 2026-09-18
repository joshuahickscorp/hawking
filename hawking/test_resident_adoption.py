"""Every negative control S010 §3 names, plus the positive case."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
import resident_adoption as A

NOW = 1_000_000.0
def rec(**kw):
    base = dict(schema=A.SCHEMA, pid=4242, proc_start="Sat Sep  6 17:00:00 2026",
                model_hash="m-abc", config_hash="c-def", owner_generation=7,
                owner_pid=999, heartbeat_epoch=NOW - 5.0,
                rss_bytes=int(9.9 * (1 << 30)), endpoint="/tmp/r.sock", ready=True)
    base.update(kw); return A.Record(**base)

def obs(**kw):
    base = dict(pid_alive=True, proc_start="Sat Sep  6 17:00:00 2026",
                model_hash="m-abc", config_hash="c-def", responsive=True,
                guard_state="OK", owned_by_live_other=False)
    base.update(kw); return base

def test_positive_healthy_resident_is_adopted():
    d, why = A.decide(rec(), obs(), NOW)
    assert d == A.ADOPT, why

def test_stale_pid():
    assert A.decide(rec(), obs(pid_alive=False), NOW)[0] == A.REAP

def test_pid_reuse_same_pid_different_start():
    d, why = A.decide(rec(), obs(proc_start="Sat Sep  6 17:45:00 2026"), NOW)
    assert d == A.REAP and "PID REUSE" in why, why

def test_wrong_model():
    assert A.decide(rec(), obs(model_hash="m-other"), NOW)[0] == A.REAP

def test_wrong_config():
    assert A.decide(rec(), obs(config_hash="c-other"), NOW)[0] == A.REAP

def test_dead_resident_alive_but_unresponsive():
    assert A.decide(rec(), obs(responsive=False), NOW)[0] == A.REAP

def test_half_initialised():
    assert A.decide(rec(ready=False), obs(), NOW)[0] == A.REAP

def test_corrupted_record(tmp_path):
    p = tmp_path / "own.json"; p.write_text("{not json")
    assert A.load(p) is None
    assert A.decide(A.load(p), obs(), NOW)[0] == A.REAP

def test_truncated_record_missing_fields(tmp_path):
    p = tmp_path / "own.json"; p.write_text(json.dumps({"pid": 1}))
    assert A.load(p) is None

def test_stale_heartbeat():
    assert A.decide(rec(heartbeat_epoch=NOW - 600), obs(), NOW)[0] == A.REAP

def test_over_resource_ceiling():
    assert A.decide(rec(rss_bytes=int(40 * (1 << 30))), obs(), NOW)[0] == A.REAP

def test_guard_stop_refuses_adoption():
    assert A.decide(rec(), obs(guard_state="STOP"), NOW)[0] == A.REAP

def test_already_owned_by_another_live_pool():
    assert A.decide(rec(), obs(owned_by_live_other=True), NOW)[0] == A.REAP

def test_unknown_schema():
    assert A.decide(rec(schema="something.else.v9"), obs(), NOW)[0] == A.REAP

def test_two_pools_racing_only_one_wins(tmp_path):
    p = tmp_path / "own.json"
    assert A.claim(p, 1) is True
    assert A.claim(p, 2) is False, "TWO owners claimed the same resident"
    A.release(p)
    assert A.claim(p, 3) is True

def test_owner_crash_mid_transfer_leaves_claim_and_blocks(tmp_path):
    p = tmp_path / "own.json"
    assert A.claim(p, 1) is True          # owner dies here, never releases
    assert A.claim(p, 2) is False         # a later pool must NOT silently take it

def test_never_adopts_on_a_bare_pid():
    # a record with nothing but a live pid must still be refused
    d, _ = A.decide(rec(model_hash="", config_hash="", ready=False), obs(), NOW)
    assert d == A.REAP
