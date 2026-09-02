"""The supervision loop must not block on a lake sweep.

MEASURED before this: 108 loop blackouts totalling 240 minutes, the longest
1607s, each one a window in which a refreshed transfer could not be relaunched.
"""
import sys, time, threading
sys.path.insert(0, "tools/odyssey")
import modellake_watch as w


def test_run_detached_returns_immediately_and_runs_the_sweep():
    done = threading.Event()

    def slow():
        time.sleep(1.5)
        done.set()

    t0 = time.monotonic()
    assert w.run_detached("probe_sweep", slow) is True
    elapsed = time.monotonic() - t0
    assert elapsed < 0.25, f"run_detached blocked the loop for {elapsed:.2f}s"
    assert done.wait(5), "the sweep never ran"


def test_a_sweep_still_in_flight_is_never_started_twice():
    started = []
    gate = threading.Event()

    def slow():
        started.append(1)
        gate.wait(3)

    assert w.run_detached("probe_single", slow) is True
    time.sleep(0.1)
    assert w.run_detached("probe_single", slow) is True
    assert w.run_detached("probe_single", slow) is True
    gate.set()
    time.sleep(0.3)
    assert started == [1], f"single-flight broken: {len(started)} runs"


def test_a_raising_sweep_does_not_kill_the_watcher():
    def boom():
        raise RuntimeError("sweep exploded")

    assert w.run_detached("probe_boom", boom) is True
    time.sleep(0.4)   # if it propagated, the thread dies silently; loop lives


def test_emit_is_serialised_across_threads(tmp_path, monkeypatch):
    # Never write probe rows into the live operational log.
    monkeypatch.setattr(w, "LOG", tmp_path / "probe.jsonl")
    monkeypatch.setattr(w, "DOWNLOAD_DIR", tmp_path)
    rows = 200
    errs = []

    def spam(n):
        try:
            for i in range(rows):
                w.emit("probe_row", worker=n, i=i, pad="x" * 4000)
        except Exception as exc:
            errs.append(exc)

    ts = [threading.Thread(target=spam, args=(n,)) for n in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errs, errs
    import json
    bad = 0
    with open(w.LOG, errors="replace") as f:
        for line in f:
            if '"probe_row"' not in line:
                continue
            try:
                json.loads(line)
            except Exception:
                bad += 1
    assert bad == 0, f"{bad} interleaved/corrupt rows"
