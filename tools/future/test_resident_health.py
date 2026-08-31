"""Negative controls for tools/future/resident_health.py.

A health system nobody has watched reject is decoration. These tests drive
PATHOLOGICAL, refuse it on a spike, refuse a one-sample verdict, and prove a
dead pid is ABSENT with rss_bytes null — never healthy-with-zero-RSS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hcli.resources import process_start_token
from tools.future import resident_health as rh
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _present(rss: int, *, t: float = 0.0, **extra: Any) -> dict:
    body: dict[str, Any] = dict(
        presence="PRESENT",
        pid=4242,
        rss_bytes=rss,
        sampled_at_unix=1_700_000_000.0 + t,
        children_status="OK",
        children_n=0,
        memory_status="OK",
        available_bytes=8 * 1024 ** 3,
        swap_ins=0,
        uma_pressure_level=0,
        uma_pressure_name="normal",
        queue_status="UNOBSERVABLE",
        detached_status="UNOBSERVABLE",
        context_status="UNOBSERVABLE",
    )
    body.update(extra)
    return rh.make_sample(**body)


def test_build_emits_sealed_receipt():
    out = rh.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_HEALTH.json"
    assert doc["schema"] == rh.SCHEMA
    assert doc["schema"] == "hawking.future.resident_health.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["telemetry_evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["gpu_authority"] is False
    assert doc["authorizes"] == "restart_decision_only"
    assert doc["is_a_measurement"] is False
    assert doc["is_hardware_performance_claim"] is False
    assert doc["executed"]["sample_undeclared"] is True
    assert doc["executed"]["trend"] is True
    assert doc["executed"]["verdict"] is True
    assert doc["proofs"]["synthetic"]["all_passed"] is True
    assert "PATHOLOGICAL" in doc["proofs"]["synthetic"]["verdicts_reached"]
    assert "HEALTHY" in doc["proofs"]["synthetic"]["verdicts_reached"]
    assert "DEGRADED" in doc["proofs"]["synthetic"]["verdicts_reached"]
    assert doc["live_undeclared"]["presence"] == "UNDECLARED"
    assert doc["live_undeclared"]["rss_bytes"] is None
    # The queue exists whether or not a resident is bound to it, so its depth is
    # a fact about the frontier and not about the process. This asserted None
    # because a sparse lane worktree had no queue at all -- the same shape of
    # mistake as a test asserting a file is absent, which is true only where it
    # was written. What must hold is that an UNDECLARED resident reports no
    # PROCESS telemetry, and that a queue depth, when present, is a real count.
    queue_depth = doc["live_undeclared"]["queue_depth"]
    assert queue_depth is None or (isinstance(queue_depth, int) and queue_depth >= 0)
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["resident_callable"]["frontier"] == "FT.HCLI_SELF.emit-workunits"
    assert "HealthRefuse" in doc["resident_callable"]["fails_closed"]
    assert "VI" not in "".join(doc["vocabulary"]["eras"])
    assert len(doc["vocabulary"]["eras"]) == 5
    assert len(doc["vocabulary"]["odysseys"]) == 3
    _assert_no_hardware_claims(doc)


def test_selftest_aliases_build():
    assert rh.selftest().name == "RESIDENT_HEALTH.json"


def test_receipt_contains_no_hardware_measurement_fields():
    doc = json.loads(rh.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    raise AssertionError(f"{here} = {v!r} is a hardware field")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_one_sample_refuses_a_trend_verdict():
    one = [_present(100)]
    with pytest.raises(rh.HealthRefuse, match="one sample cannot produce a trend"):
        rh.trend(one)
    with pytest.raises(rh.HealthRefuse, match="one sample cannot produce a trend"):
        rh.verdict(one)


def test_empty_window_refuses():
    with pytest.raises(rh.HealthRefuse):
        rh.trend([])
    with pytest.raises(rh.HealthRefuse):
        rh.verdict([])


def test_single_spike_in_flat_series_is_not_pathological():
    spike = [_present(b, t=float(i)) for i, b in enumerate((100, 100, 800, 100, 100))]
    tr = rh.trend(spike)
    assert tr["signals"]["rss_bytes"]["direction"] == "SPIKE"
    got = rh.verdict(tr)
    assert got["verdict"] != "PATHOLOGICAL"
    assert got["verdict"] in rh.VERDICTS
    assert rh.authorize_restart(got) is False


def test_spike_at_end_of_flat_series_is_not_pathological():
    """The reading that would restart-thrash: one high value after a flat run."""
    spike = [_present(b, t=float(i)) for i, b in enumerate((100, 100, 100, 900))]
    got = rh.verdict(spike)
    assert got["verdict"] != "PATHOLOGICAL"
    assert rh.authorize_restart(got) is False


def test_monotonic_climb_across_the_window_is_pathological():
    climb = [_present(b, t=float(i)) for i, b in enumerate((100, 200, 400, 800, 1600))]
    tr = rh.trend(climb)
    assert tr["signals"]["rss_bytes"]["direction"] == "CLIMB_NO_PLATEAU"
    got = rh.verdict(tr)
    assert got["verdict"] == "PATHOLOGICAL"
    assert got["signal"] == "rss_bytes"
    assert rh.authorize_restart(got) is True


def test_plateau_after_climb_is_not_pathological():
    plateau = [_present(b, t=float(i)) for i, b in enumerate((100, 200, 400, 400, 400))]
    tr = rh.trend(plateau)
    assert tr["signals"]["rss_bytes"]["direction"] != "CLIMB_NO_PLATEAU"
    got = rh.verdict(tr)
    assert got["verdict"] != "PATHOLOGICAL"
    assert rh.authorize_restart(got) is False


def test_flat_series_is_healthy():
    flat = [_present(250, t=float(i)) for i in range(4)]
    got = rh.verdict(flat)
    assert got["verdict"] == "HEALTHY"
    assert got["signal"] is None
    assert rh.authorize_restart(got) is False


def test_two_point_increase_is_not_pathological():
    """Two points are a direction, not a window long enough to distinguish spike from climb."""
    pair = [_present(100, t=0), _present(800, t=1)]
    tr = rh.trend(pair)
    assert tr["signals"]["rss_bytes"]["direction"] != "CLIMB_NO_PLATEAU"
    got = rh.verdict(tr)
    assert got["verdict"] != "PATHOLOGICAL"
    assert rh.authorize_restart(got) is False


def test_make_sample_rejects_absent_with_zero_rss():
    with pytest.raises(rh.HealthRefuse, match="healthy-with-zero-RSS"):
        rh.make_sample(presence="ABSENT", rss_bytes=0)
    with pytest.raises(rh.HealthRefuse, match="healthy-with-zero-RSS"):
        rh.make_sample(presence="UNDECLARED", rss_bytes=0)


def test_dead_resident_is_absent_not_healthy_with_zero_rss():
    try:
        proc = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        # Spawn failed: the module must still refuse to call a missing pid healthy.
        dead = rh.sample(pid=2_147_000_000)
        assert dead["resident"]["presence"] == "ABSENT"
        assert dead["resident"]["rss_bytes"] is None
        return
    pid = proc.pid
    proc.kill()
    proc.wait(timeout=5)
    dead = rh.sample(pid=pid)
    assert dead["resident"]["presence"] == "ABSENT"
    assert dead["resident"]["rss_bytes"] is None
    assert dead["resident"]["rss_bytes"] != 0
    assert dead["evidence_class"] == "SELF_MEASURED_DIRTY"
    # A window of absence is PATHOLOGICAL, not HEALTHY-with-zero.
    window = [
        rh.make_sample(presence="ABSENT", pid=pid, sampled_at_unix=1_700_000_000.0 + i)
        for i in range(3)
    ]
    got = rh.verdict(window)
    assert got["verdict"] == "PATHOLOGICAL"
    assert got["signal"] == "resident_absent"
    assert rh.authorize_restart(got) is True


def test_live_self_is_present_and_rss_is_not_a_fake_zero():
    live = rh.sample(pid=os.getpid())
    assert live["resident"]["presence"] == "PRESENT"
    rss = live["resident"]["rss_bytes"]
    if rss is None:
        assert live["resident"]["reason"] and "rss unreadable" in str(live["resident"]["reason"])
    else:
        assert isinstance(rss, int)
        assert rss > 0


def test_undeclared_pid_is_not_invented():
    got = rh.sample()
    assert got["resident"]["presence"] == "UNDECLARED"
    assert got["resident"]["pid"] is None
    assert got["resident"]["rss_bytes"] is None
    assert "largest RSS neighbour" in str(got["resident"]["reason"])


def test_missing_queue_and_detached_are_unobservable_not_empty_healthy(tmp_path):
    got = rh.sample(pid=os.getpid())
    assert got["queue"]["status"] == "UNOBSERVABLE" or got["queue"]["status"] in {"OK", "PARTIAL", "FAILED"}
    if got["queue"]["status"] == "UNOBSERVABLE":
        assert got["queue"]["depth"] is None
    assert got["detached"]["status"] == "UNOBSERVABLE"
    assert got["detached"]["n_jobs"] is None
    assert got["context"]["status"] == "UNOBSERVABLE"
    assert got["context"]["bytes"] is None
    # An empty jobs directory that actually exists is observable zero, which is not the missing case.
    jobs = tmp_path / "detached" / "jobs"
    jobs.mkdir(parents=True)
    observed = rh.sample(pid=os.getpid(), workspace=tmp_path)
    assert observed["detached"]["status"] == "OK"
    assert observed["detached"]["n_jobs"] == 0
    assert observed["detached"]["n_dead"] == 0


def test_reused_pid_is_absent_not_the_stranger_rss():
    try:
        first = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        second = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        # Cannot spawn: identity reuse still has to refuse a dead pid as ABSENT.
        dead = rh.sample(pid=2_147_000_001, start_token="not-this-process")
        assert dead["resident"]["presence"] == "ABSENT"
        assert dead["resident"]["rss_bytes"] is None
        return
    try:
        token = process_start_token(first.pid)
        if not token:
            # Token unreadable: still must not report the stranger's RSS as the resident.
            live = rh.sample(pid=first.pid)
            assert live["resident"]["presence"] == "PRESENT"
            return
        forged = rh.sample(pid=second.pid, start_token=token)
        assert forged["resident"]["presence"] == "ABSENT"
        assert forged["resident"]["rss_bytes"] is None
        assert "not the resident" in str(forged["resident"]["reason"])
    finally:
        for proc in (first, second):
            try:
                proc.kill()
                proc.wait(timeout=5)
            except OSError:
                proc.poll()


def test_shrinking_available_memory_without_plateau_is_pathological():
    rows = [
        _present(200, t=float(i), available_bytes=8_000 - i * 1_000)
        for i in range(4)
    ]
    tr = rh.trend(rows)
    assert tr["signals"]["available_bytes"]["direction"] == "CLIMB_NO_PLATEAU"
    got = rh.verdict(tr)
    assert got["verdict"] == "PATHOLOGICAL"
    assert got["signal"] in {"available_bytes", "rss_bytes"}


def test_children_tree_climb_is_pathological():
    rows = [_present(200, t=float(i), children_n=i + 1) for i in range(4)]
    tr = rh.trend(rows)
    assert tr["signals"]["children_n"]["direction"] == "CLIMB_NO_PLATEAU"
    got = rh.verdict(tr)
    assert got["verdict"] == "PATHOLOGICAL"
    assert "children_n" in got["triggers"]["pathological"]


def test_reversed_timestamps_are_refused():
    rows = [_present(100 + i * 50, t=float(3 - i)) for i in range(4)]
    with pytest.raises(rh.HealthRefuse, match="not in time order"):
        rh.trend(rows)


def test_silence_is_not_healthy():
    silent = [
        rh.make_sample(presence="UNKNOWN", sampled_at_unix=1_700_000_000.0 + i)
        for i in range(3)
    ]
    with pytest.raises(rh.HealthRefuse, match="refusing HEALTHY on silence"):
        rh.verdict(silent)
    undeclared = [
        rh.make_sample(presence="UNDECLARED", sampled_at_unix=1_700_000_000.0 + i)
        for i in range(3)
    ]
    with pytest.raises(rh.HealthRefuse, match="refusing HEALTHY on silence"):
        rh.verdict(undeclared)


def test_restart_supervisor_absence_is_coped_not_skipped():
    landed = rh.restart_supervisor_presence()
    assert landed["path"] == "tools/future/restart_supervisor.py"
    assert "landed" in landed
    # Either way the module still samples and still refuses a one-sample verdict.
    with pytest.raises(rh.HealthRefuse):
        rh.verdict([_present(1)])
    doc = json.loads(rh.build().read_text())
    assert any("restart_supervisor" in row for row in doc["recovered_implementation"])


def test_module_has_no_placeholder_tokens():
    src = Path(rh.__file__).read_text()
    for banned in ("raise NotImplementedError", "pytest.skip", "TODO", "FIXME"):
        assert banned not in src
    tree = __import__("ast").parse(src)
    for node in __import__("ast").walk(tree):
        if isinstance(node, __import__("ast").Pass):
            raise AssertionError(f"pass statement at line {node.lineno}")


def test_cli_build_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(Path(rh.__file__)), "--build"],
        cwd=str(rh.REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESIDENT_HEALTH.json" in proc.stdout
