"""G101: a sealed specimen must reach a WorkUnit without a human running
`modellake_events.py --build` by hand.

The watcher (tools/odyssey/modellake_watch.py) is the one process already
running unattended (launchd com.hawking.modellake.watch). Before this wiring,
nothing ever called the seal -> registry -> fingerprint -> role -> WorkUnit
consumer except a human typing `--build`. These tests exercise the two new
functions directly and never call main()/acquire_lock() -- the live watcher
holds that flock, and colliding with it (or its two live downloads) is
exactly what this file must not do.

G168 adds a second, unrelated gap of the same shape: complete(item, ...)
returning True used to just mean "skip"; nothing promoted the finished
payload out of partial/. The tests below exercise that wiring (promotion,
reconciliation) the same way -- direct function calls against a scratch
tree, never main()/acquire_lock().
"""
from __future__ import annotations

import importlib
import json

import pytest

from tools.future._common import RECEIPTS
from tools.odyssey import modellake_promote as mp
from tools.odyssey import modellake_watch as mw


def test_the_watcher_loop_actually_calls_the_gate_function():
    """Compiled-bytecode check, not a comment/docstring grep: main() must
    load `maybe_emit_modellake_events` as a real name in its body, and must
    thread a `last_events_emit` variable across loop iterations. A revert
    that deletes the call site (but leaves the helper functions standing)
    must fail this test."""
    names = mw.main.__code__.co_names
    varnames = mw.main.__code__.co_varnames
    assert "maybe_emit_modellake_events" in names
    assert "last_events_emit" in varnames


def test_emit_modellake_events_once_runs_the_real_consumer_and_writes_the_receipt():
    from tools.future import modellake_events as me

    receipt_path = RECEIPTS / me.RECEIPT_NAME
    before = receipt_path.read_bytes() if receipt_path.is_file() else None
    try:
        n = mw.emit_modellake_events_once()
        assert isinstance(n, int)
        assert n == me.build()["n_emitted_specimens"]
        assert receipt_path.is_file()
        doc = json.loads(receipt_path.read_text())
        assert doc["schema"] == me.SCHEMA
        assert doc["is_this_wired"] is True
    finally:
        if before is not None:
            receipt_path.write_bytes(before)


def test_maybe_emit_modellake_events_respects_the_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(mw, "emit_modellake_events_once", lambda: calls.append(1) or 3)
    monkeypatch.setattr(mw, "emit", lambda *a, **k: None)

    # last_events_emit == 0.0 is "never run yet" -> fires immediately.
    stamp = mw.maybe_emit_modellake_events(1000.0, 0.0)
    assert stamp == 1000.0
    assert len(calls) == 1

    # Well inside the interval -> does not fire again.
    stamp = mw.maybe_emit_modellake_events(1000.0 + 1.0, stamp)
    assert stamp == 1000.0
    assert len(calls) == 1

    # Past the interval -> fires again.
    stamp = mw.maybe_emit_modellake_events(
        1000.0 + mw.MODELLAKE_EVENTS_INTERVAL_SECONDS + 1.0, stamp
    )
    assert len(calls) == 2
    assert stamp == 1000.0 + mw.MODELLAKE_EVENTS_INTERVAL_SECONDS + 1.0


def test_maybe_emit_modellake_events_never_raises_on_a_broken_consumer(monkeypatch):
    def boom():
        raise RuntimeError("sidecar is missing a dependency")

    events = []
    monkeypatch.setattr(mw, "emit_modellake_events_once", boom)
    monkeypatch.setattr(mw, "emit", lambda event, **fields: events.append((event, fields)))

    # Must not raise -- a broken consumer must never crash the watcher that
    # is admitting the two live downloads.
    stamp = mw.maybe_emit_modellake_events(500.0, 0.0)
    assert stamp == 500.0
    assert events
    event, fields = events[0]
    assert event == "modellake_events_error"
    assert "sidecar is missing a dependency" in fields["error"]


def test_emit_modellake_events_once_does_no_network_or_download_side_effects():
    """The consumer must not import or touch anything download-shaped; it is
    a CPU-only reader of manifests already on disk and this watcher's own
    JSONL tail. Guard against a future edit accidentally wiring it to
    `launch()`/`manifest_for()` (both do real network I/O)."""
    import inspect

    src = inspect.getsource(mw.emit_modellake_events_once)
    for forbidden in ("launch(", "manifest_for(", "HF_BIN", "subprocess"):
        assert forbidden not in src


# --- G168: complete() -> promotion wiring, and the reconciliation pass -----


@pytest.fixture
def wlake(tmp_path, monkeypatch):
    """A scratch ModelLake: partial/, specimens/, watch-manifests/ and a
    fresh JSONL log, with both modellake_watch and modellake_promote pointed
    at it. Never the real /Volumes/corpdrive tree, never a download."""
    model_root = tmp_path / "model_root"
    partial = model_root / "partial"
    specimens = model_root / "specimens"
    manifests = tmp_path / "watch-manifests"
    partial.mkdir(parents=True)
    specimens.mkdir(parents=True)
    manifests.mkdir(parents=True)
    for mod in (mp, mw):
        monkeypatch.setattr(mod, "MODEL_ROOT", model_root, raising=False)
        monkeypatch.setattr(mod, "SPECIMEN_ROOT", specimens, raising=False)
        monkeypatch.setattr(mod, "MANIFEST_DIR", manifests, raising=False)
    monkeypatch.setattr(mp, "PARTIAL_ROOT", partial, raising=False)
    monkeypatch.setattr(mw, "P0", [], raising=False)
    monkeypatch.setattr(mw, "QUEUE", [], raising=False)
    monkeypatch.setattr(mw, "LOG", tmp_path / "watch.jsonl", raising=False)
    monkeypatch.setattr(mw, "notify", lambda *a, **k: None, raising=False)
    return {"partial": partial, "specimens": specimens, "manifests": manifests}


def _write_manifest(wlake, tag, files):
    sizes = {name: len(content) for name, content in files.items()}
    (wlake["manifests"] / f"{tag}.json").write_text(json.dumps({
        "repo": "acme/x", "revision": "r", "mode": "safe",
        "expected": sum(sizes.values()), "files": list(files),
        "sizes": sizes, "resolved_sha": "r",
    }))
    return sizes


def _write_partial(wlake, tag, files):
    d = wlake["partial"] / tag
    d.mkdir(parents=True)
    for name, content in files.items():
        (d / name).write_bytes(content)
    return d


TAG = "acme--x@deadbeefcafe"
FILES = {"config.json": b"{}", "weights.bin": b"weights"}


def test_the_watcher_loop_actually_calls_promotion_on_completion():
    """Bytecode check, not a comment grep: main() must call
    `_promote_and_report` from within its body. A revert that restores the
    old bare `emit('already_complete', ...); continue` (dropping the call
    but leaving the helper standing) must fail this test."""
    assert "_promote_and_report" in mw.main.__code__.co_names


def test_the_watcher_loop_actually_calls_reconciliation():
    """Same shape: main() must load `maybe_reconcile` as a real name and
    thread a `last_reconcile` variable across loop iterations."""
    assert "maybe_reconcile" in mw.main.__code__.co_names
    assert "last_reconcile" in mw.main.__code__.co_varnames


def test_promote_if_needed_is_a_noop_once_the_partial_is_already_gone(wlake):
    assert mw.promote_if_needed("nonexistent-tag", str(wlake["partial"] / "gone")) is None


def test_promote_if_needed_promotes_a_complete_partial(wlake):
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)

    outcome = mw.promote_if_needed(TAG, str(d))

    assert outcome["action"] == "PROMOTED"
    assert not d.is_dir()
    assert (wlake["specimens"] / TAG).is_dir()


def test_promote_and_report_notifies_on_success(wlake, monkeypatch):
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    calls = []
    monkeypatch.setattr(mw, "notify", lambda msg, kind="warning": calls.append((msg, kind)))

    mw._promote_and_report(TAG, str(d), sum(len(c) for c in FILES.values()))

    assert (wlake["specimens"] / TAG).is_dir()
    assert calls and "Promoted" in calls[0][0]


def test_promote_and_report_fires_sealed_source_ready_transition(wlake, monkeypatch):
    """Promotion is the enter->exit for SLEEPING_SPECIMEN_WU."""
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    fired = []
    monkeypatch.setattr(
        mw,
        "_notify_sealed_source",
        lambda tag, action, source="": fired.append((tag, action, source)),
    )

    mw._promote_and_report(TAG, str(d), sum(len(c) for c in FILES.values()))

    assert fired and fired[0][0] == TAG
    assert fired[0][1] == "PROMOTED"
    assert "_notify_sealed_source" in mw._promote_and_report.__code__.co_names
    assert "_notify_sealed_source" in mw.reconcile.__code__.co_names


def test_promote_and_report_notifies_on_conflicting_destination(wlake, monkeypatch):
    """A destination conflict must be surfaced (escalated), not silently
    dropped the way the pre-wiring `continue` did."""
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    (wlake["specimens"] / TAG).mkdir()
    calls = []
    monkeypatch.setattr(mw, "notify", lambda msg, kind="warning": calls.append((msg, kind)))

    mw._promote_and_report(TAG, str(d), sum(len(c) for c in FILES.values()))

    assert d.is_dir()  # never touched -- preserved alongside the conflict
    assert calls and "attention" in calls[0][0]


def test_promote_and_report_replay_is_silent(wlake, monkeypatch):
    """A replayed already_complete event over an already-promoted tag must
    not fire a notification every time -- that would be exactly the kind of
    noise that trains an operator to ignore the channel."""
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    mw._promote_and_report(TAG, str(d), 0)  # first call actually promotes
    calls = []
    monkeypatch.setattr(mw, "notify", lambda msg, kind="warning": calls.append((msg, kind)))

    mw._promote_and_report(TAG, str(d), 0)  # replay: partial already gone

    assert calls == []


def test_reconcile_promotes_complete_but_unpromoted(wlake):
    """This is the seven-day Qwen2.5-72B shape: a payload finished and
    nothing ever looked again. reconcile() is the second look."""
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)

    result = mw.reconcile()

    assert TAG in result["promoted"]
    assert not d.is_dir()
    assert (wlake["specimens"] / TAG).is_dir()


def test_reconcile_flags_duplicate_source_and_touches_neither(wlake):
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    dest = wlake["specimens"] / TAG
    dest.mkdir()

    result = mw.reconcile()

    assert {"kind": "duplicate_source", "tag": TAG} in result["anomalies"]
    assert TAG not in result["promoted"]
    assert d.is_dir() and dest.is_dir()


def test_reconcile_flags_registered_but_missing(wlake):
    _write_manifest(wlake, TAG, FILES)  # manifest exists, nothing on disk
    with mw.LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": "t", "event": "download_started", "job": TAG}) + "\n")

    result = mw.reconcile()

    assert {"kind": "registered_but_missing", "tag": TAG} in result["anomalies"]


def test_reconcile_does_not_flag_a_never_started_queue_entry_as_missing(wlake):
    """A manifest with no `download_started` history is the normal state for
    most of QUEUE at any given moment -- not admitted yet, not an anomaly."""
    _write_manifest(wlake, TAG, FILES)

    result = mw.reconcile()

    kinds = {a["kind"] for a in result["anomalies"]}
    assert "registered_but_missing" not in kinds


def test_reconcile_flags_stale_downloader_state_but_still_trusts_the_manifest(wlake):
    """Qwen2.5-72B recorded exit_code=1 while 47/47 files were present and
    correct. The exit code must never block promotion -- only get reported
    as a diagnostic disagreement."""
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    with mw.LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": "t", "event": "download_exit",
                                 "job": TAG, "returncode": 1}) + "\n")

    result = mw.reconcile()

    assert TAG in result["promoted"]  # exit code did not block it
    assert not d.is_dir()
    assert {"kind": "stale_downloader_state", "tag": TAG,
            "recorded_exit_code": 1} in result["anomalies"]


def test_reconcile_never_promotes_while_a_live_writer_is_present(wlake, monkeypatch):
    d = _write_partial(wlake, TAG, FILES)
    _write_manifest(wlake, TAG, FILES)
    monkeypatch.setattr(mw, "process_rows",
                        lambda: [(999, f"hf download acme/x --local-dir {d}")])

    result = mw.reconcile()

    assert result["promoted"] == []
    assert d.is_dir()
    assert not (wlake["specimens"] / TAG).is_dir()


def test_maybe_reconcile_respects_the_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(mw, "reconcile", lambda: calls.append(1))

    stamp = mw.maybe_reconcile(1000.0, 0.0)  # never run yet -> fires immediately
    assert stamp == 1000.0 and len(calls) == 1

    stamp = mw.maybe_reconcile(1000.0 + 1.0, stamp)  # inside interval -> no fire
    assert stamp == 1000.0 and len(calls) == 1

    stamp = mw.maybe_reconcile(1000.0 + mw.RECONCILE_INTERVAL_SECONDS + 1.0, stamp)
    assert len(calls) == 2 and stamp == 1000.0 + mw.RECONCILE_INTERVAL_SECONDS + 1.0


def test_maybe_reconcile_never_raises_on_a_broken_reconcile(monkeypatch):
    def boom():
        raise RuntimeError("manifest dir vanished mid-sweep")

    events = []
    monkeypatch.setattr(mw, "reconcile", boom)
    monkeypatch.setattr(mw, "emit", lambda event, **fields: events.append((event, fields)))

    stamp = mw.maybe_reconcile(500.0, 0.0)

    assert stamp == 500.0
    assert events and events[0][0] == "reconciliation_error"
    assert "manifest dir vanished mid-sweep" in events[0][1]["error"]


def test_modellake_watch_and_modellake_promote_share_one_module_identity():
    """A bare sibling `import modellake_promote` would create a second
    module object under a different sys.modules key than
    tools.odyssey.modellake_promote -- a test (or future caller) patching
    the package-qualified module would silently miss the copy this file
    actually calls. Guards the fix for that exact regression."""
    assert mw.modellake_promote is mp
