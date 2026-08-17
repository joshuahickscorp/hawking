"""Genesis child pool: spawn / poll / kill / admission, against real OS processes.

The live Qwen3.8 binary is not required. Tests drive the stub child (a real
process that speaks the same stdout contract) so liveness, kill-reap, disk
capture and the admission gate can fail loudly without swapping the box.

A live four-child generate is a separate measurement (tools/genesis_pool.py e2e)
and is recorded in receipts/ascent-2026-08-16/. GENESIS_POOL_LIVE=1 opts this
file into that run when the artifact and binary are present.
"""
from __future__ import annotations

import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

import pytest

from lab.genesis_pool import (
    MEASURED_SAFE_N,
    AdmissionRefused,
    ChildBudget,
    GenesisPool,
    PoolConfig,
    UnknownChild,
    _normalize_footprint_json,
    discover_artifact,
    discover_binary,
    extract_generated_text,
    kv_bytes_estimate,
    live_ready,
    parse_child_metrics,
    pgrep_binary_pids,
    process_liveness,
    process_rss_bytes,
    recommended_safe_n,
    sample_machine,
)

REPO = Path(__file__).resolve().parents[2]
STUB = REPO / "lab" / "genesis_pool.py"
LOCK = REPO / "tools" / "gpu_lane_lock.sh"


def _cfg(tmp_path: Path, *, safe_n: int = 4, extra_args: list[str] | None = None) -> PoolConfig:
    return PoolConfig(
        binary=STUB,
        artifact_root=tmp_path / "artifact",
        tokenizer=tmp_path / "tok.json",
        output_root=tmp_path / "pool",
        safe_n=safe_n,
        lock_script=LOCK,
        max_seq_len=128,
        extra_args=list(extra_args or []),
    )


def _pool(tmp_path: Path, **kwargs: object) -> GenesisPool:
    return GenesisPool(_cfg(tmp_path, **kwargs))


def test_spawn_poll_done_writes_stdout(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cid = pool.spawn("hello-world", 4, extra_args=["--sleep", "0.15"])
    running = pool.poll(cid)
    assert running.state == "running"
    assert process_liveness(running.pid or 0) == "running"
    done = pool.wait(cid, timeout_s=5)
    assert done.state == "done"
    assert done.text is not None and "hello-world" in done.text
    assert done.wall_ns is not None and done.wall_ns > 0
    stdout = Path(done.output_dir or "") / "stdout.txt"
    assert stdout.is_file()
    assert "GENERATED_TEXT_VERBATIM:" in stdout.read_text()
    result = Path(done.output_dir or "") / "result.json"
    assert result.is_file()
    body = json.loads(result.read_text())
    assert body["observed_from"] == "process_state"
    assert body["state"] == "done"


def test_poll_ignores_stale_status_file(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cid = pool.spawn("stale", 4, extra_args=["--sleep", "0.8"])
    rec = pool._children[cid]
    out = Path(rec.output_dir)
    (out / "status").write_text("done\n")
    (out / "result.json").write_text(json.dumps({"state": "done", "text": "lie"}))
    st = pool.poll(cid)
    assert st.state == "running", "a status file must not impersonate a live child"
    pool.kill(cid)


def test_externally_killed_child_is_failed_not_running(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cid = pool.spawn("die-please", 4, extra_args=["--sleep", "8"])
    rec = pool._children[cid]
    os.kill(rec.pid, signal.SIGKILL)
    deadline = time.monotonic() + 3
    st = pool.poll(cid)
    while st.state == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        st = pool.poll(cid)
    assert st.state == "failed"
    assert st.reason is not None
    assert process_liveness(rec.pid) == "dead"
    # A second poll stays failed — it must not flip back to running.
    assert pool.poll(cid).state == "failed"


def test_stub_die_flag_is_failed(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cid = pool.spawn("boom", 4, extra_args=["--die"])
    st = pool.wait(cid, timeout_s=3)
    assert st.state == "failed"
    assert process_liveness(pool._children[cid].pid) == "dead"


def test_kill_reaps_process_group_and_frees_slot(tmp_path: Path) -> None:
    pool = _pool(tmp_path, safe_n=1)
    cid = pool.spawn("hold", 4, extra_args=["--sleep", "20", "--alloc-mb", "48"])
    rec = pool._children[cid]
    deadline = time.monotonic() + 3
    rss = process_rss_bytes(rec.pid)
    while (rss is None or rss < 20 * 1024 * 1024) and time.monotonic() < deadline:
        time.sleep(0.05)
        rss = process_rss_bytes(rec.pid)
    assert rss is not None and rss >= 20 * 1024 * 1024
    pid = rec.pid
    killed = pool.kill(cid, grace_s=2.0)
    assert killed.state == "failed"
    assert killed.reason == "killed"
    assert process_liveness(pid) == "dead"
    assert process_rss_bytes(pid) is None
    # Slot is free: a second spawn must be admitted.
    cid2 = pool.spawn("after-kill", 2, extra_args=["--sleep", "0.05"])
    assert cid2 != cid
    assert pool.wait(cid2, timeout_s=3).state == "done"
    pool.shutdown(kill=True)


def test_admission_refuses_oversubscription(tmp_path: Path) -> None:
    pool = _pool(tmp_path, safe_n=2)
    a = pool.spawn("a", 4, extra_args=["--sleep", "3"])
    b = pool.spawn("b", 4, extra_args=["--sleep", "3"])
    assert pool.alive_count() == 2
    with pytest.raises(AdmissionRefused) as ei:
        pool.spawn("c", 4, extra_args=["--sleep", "1"])
    assert ei.value.safe_n == 2
    assert ei.value.alive == 2
    # Prove no third child directory with a live pid was created after the refusal.
    live = pool.running_ids()
    assert sorted(live) == sorted([a, b])
    pool.shutdown(kill=True)
    assert pool.alive_count() == 0


def test_output_survives_dropping_the_pool_object(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cid = pool.spawn("persist", 4, extra_args=["--sleep", "0.05"])
    rec = pool._children[cid]
    out_dir = Path(rec.output_dir)
    st = pool.wait(cid, timeout_s=3)
    assert st.state == "done"
    del pool
    assert (out_dir / "stdout.txt").is_file()
    assert "GENERATED_TEXT_VERBATIM:" in (out_dir / "stdout.txt").read_text()
    assert (out_dir / "meta.json").is_file()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["child_id"] == cid
    assert "state" not in meta


def test_four_children_complete_concurrently_faster_than_serial(tmp_path: Path) -> None:
    sleep_s = 0.45
    pool = _pool(tmp_path, safe_n=4)
    t0 = time.monotonic()
    ids = [
        pool.spawn(f"p{i}", ChildBudget(max_new_tokens=4), extra_args=["--sleep", str(sleep_s)])
        for i in range(4)
    ]
    assert pool.alive_count() == 4
    results = [pool.wait(cid, timeout_s=5) for cid in ids]
    parallel = time.monotonic() - t0
    assert all(r.state == "done" for r in results)
    assert all(r.wall_ns and r.wall_ns > 0 for r in results)
    # Serial would be ~1.8s. Overlap must beat 1.3s with slack for scheduling.
    assert parallel < 1.3
    serial_sum = sum(r.wall_ns or 0 for r in results) / 1e9
    assert serial_sum > parallel
    pool.shutdown(kill=True)


def test_text_does_not_take_the_lock_timing_does(tmp_path: Path) -> None:
    lock_dir = tmp_path / "lane.lock"
    script = tmp_path / "lock.sh"
    script.write_text(
        "#!/bin/bash\n"
        f'LOCK="{lock_dir}"\n'
        'NAME="$1"; shift\n'
        'until mkdir "$LOCK" 2>/dev/null; do sleep 0.05; done\n'
        'echo $$ > "$LOCK/pid"\n'
        'echo "$NAME" > "$LOCK/owner"\n'
        'trap \'rm -rf "$LOCK"\' EXIT INT TERM\n'
        '"$@"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    cfg = _cfg(tmp_path, safe_n=4)
    cfg.lock_script = script
    pool = GenesisPool(cfg)
    text_id = pool.spawn("text", 4, hold_gpu_lock=False, extra_args=["--sleep", "0.6"])
    time.sleep(0.1)
    assert not lock_dir.exists()
    assert "lock.sh" not in " ".join(pool._children[text_id].argv)
    timing_id = pool.spawn("timing", 4, hold_gpu_lock=True, extra_args=["--sleep", "0.6"])
    deadline = time.monotonic() + 2
    while not lock_dir.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert lock_dir.exists()
    assert str(script) in pool._children[timing_id].argv
    assert pool._children[timing_id].hold_gpu_lock is True
    assert pool._children[text_id].hold_gpu_lock is False
    assert pool.poll(text_id).state == "running"
    pool.shutdown(kill=True)
    assert not lock_dir.exists() or process_liveness(pool._children[timing_id].pid) == "dead"


def test_two_timing_children_serialize_on_the_lock(tmp_path: Path) -> None:
    lock_dir = tmp_path / "lane.lock"
    script = tmp_path / "lock.sh"
    script.write_text(
        "#!/bin/bash\n"
        f'LOCK="{lock_dir}"\n'
        'NAME="$1"; shift\n'
        'until mkdir "$LOCK" 2>/dev/null; do sleep 0.05; done\n'
        'echo $$ > "$LOCK/pid"\n'
        'echo "$NAME" > "$LOCK/owner"\n'
        'trap \'rm -rf "$LOCK"\' EXIT INT TERM\n'
        '"$@"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    cfg = _cfg(tmp_path, safe_n=4)
    cfg.lock_script = script
    pool = GenesisPool(cfg)
    sleep_s = 0.5
    t0 = time.monotonic()
    a = pool.spawn("t0", 4, hold_gpu_lock=True, extra_args=["--sleep", str(sleep_s)])
    b = pool.spawn("t1", 4, hold_gpu_lock=True, extra_args=["--sleep", str(sleep_s)])
    ra = pool.wait(a, timeout_s=8)
    rb = pool.wait(b, timeout_s=8)
    elapsed = time.monotonic() - t0
    if ra.state != "done" or rb.state != "done":
        for label, cid, st in (("a", a, ra), ("b", b, rb)):
            rec = pool._children[cid]
            err = Path(rec.output_dir) / "stderr.txt"
            out = Path(rec.output_dir) / "stdout.txt"
            raise AssertionError(
                f"{label} state={st.state} reason={st.reason} exit={st.exit_code} "
                f"stderr={err.read_text(errors='replace')[-400]!r} "
                f"stdout={out.read_text(errors='replace')[-200]!r}"
            )
    # Serialized ~1.0s; concurrent would be ~0.5s. Require > 0.85s.
    assert elapsed >= 0.85
    pool.shutdown(kill=True)


def test_unknown_child_poll_raises(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    with pytest.raises(UnknownChild):
        pool.poll("does-not-exist")


def test_parse_generated_text_and_complete_token() -> None:
    stdout = (
        "GENERATED_TEXT_VERBATIM: hello\nthere\n"
        "FALLBACKS: 0\n"
        "NEW_TOKENS: [1, 2, 3]\n"
        "WALL_NS: 17349168959\n"
        "STEADY_DECODE_WALL_NS_PER_TOKEN: Some(79299091)\n"
    )
    assert extract_generated_text(stdout) == "hello\nthere"
    metrics = parse_child_metrics(stdout)
    assert metrics["child_wall_ns"] == 17_349_168_959
    assert metrics["complete_token_ns"] == 79_299_091
    assert metrics["new_tokens"] == [1, 2, 3]


def test_kv_estimate_is_labeled_and_does_not_double_n() -> None:
    # ESTIMATE from source constants, not a measurement.
    assert kv_bytes_estimate(128) == 131_072 * 128
    delta_8192_vs_128 = kv_bytes_estimate(8192) - kv_bytes_estimate(128)
    # ~1.05 GB. Cannot double child count: Metal weights are ~14.73 GB.
    assert 1_000_000_000 < delta_8192_vs_128 < 1_200_000_000
    assert recommended_safe_n(128) == MEASURED_SAFE_N
    assert recommended_safe_n(8192) == MEASURED_SAFE_N


def test_sample_machine_reports_trusted_figure() -> None:
    m = sample_machine()
    assert m["page_size"] == 16384
    assert "trusted_pressure_bytes" in m
    assert "wired" in m["trusted_figure"]


def test_rehydrated_pool_still_sees_process_state(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cid = pool.spawn("rehydrate", 4, extra_args=["--sleep", "2"])
    rec = pool._children[cid]
    other = GenesisPool(_cfg(tmp_path))
    st = other.poll(cid)
    assert st.state == "running"
    assert st.pid == rec.pid
    other.kill(cid)
    assert process_liveness(rec.pid) == "dead"


@pytest.mark.skipif(
    os.environ.get("GENESIS_POOL_LIVE") != "1" or live_ready() is None,
    reason="live Qwen3.8 e2e is opt-in (GENESIS_POOL_LIVE=1) and needs the artifact",
)
def test_live_four_text_children() -> None:
    ready = live_ready()
    assert ready is not None
    binary, artifact, tokenizer = ready
    out = REPO / "workspace/ops/local/genesis-pool-live-test"
    pool = GenesisPool(
        PoolConfig(
            binary=binary,
            artifact_root=artifact,
            tokenizer=tokenizer,
            output_root=out,
            safe_n=MEASURED_SAFE_N,
            max_seq_len=128,
        )
    )
    n = min(4, MEASURED_SAFE_N)
    ids = [
        pool.spawn(f"Name one noun. #{i}", ChildBudget(max_new_tokens=4, max_seq_len=128))
        for i in range(n)
    ]
    results = [pool.wait(cid, timeout_s=300) for cid in ids]
    assert all(r.state == "done" for r in results)
    assert all(r.text for r in results)
    pool.shutdown(kill=True)


def test_discover_helpers_do_not_raise() -> None:
    discover_binary()
    discover_artifact()
    pgrep_binary_pids()


def test_footprint_json_uses_macos_category_map() -> None:
    payload = {
        "processes": [
            {
                "pid": 1,
                "footprint": 15980048096,
                "categories": {
                    "IOAccelerator (graphics)": {"dirty": 14727856128, "clean": 0},
                    "mapped file": {"dirty": 0, "clean": 4653056},
                    "Malloc Large": {"dirty": 861405184, "clean": 0},
                },
            }
        ]
    }
    parsed = _normalize_footprint_json(payload, 1)
    assert parsed is not None
    assert parsed["phys_footprint_bytes"] == 15_980_048_096
    assert parsed["ioaccelerator_dirty_bytes"] == 14_727_856_128
    assert parsed["mapped_file_bytes"] == 4_653_056
    assert parsed["malloc_large_bytes"] == 861_405_184
