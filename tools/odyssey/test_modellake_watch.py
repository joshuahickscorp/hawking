"""G101: a sealed specimen must reach a WorkUnit without a human running
`modellake_events.py --build` by hand.

The watcher (tools/odyssey/modellake_watch.py) is the one process already
running unattended (launchd com.hawking.modellake.watch). Before this wiring,
nothing ever called the seal -> registry -> fingerprint -> role -> WorkUnit
consumer except a human typing `--build`. These tests exercise the two new
functions directly and never call main()/acquire_lock() -- the live watcher
holds that flock, and colliding with it (or its two live downloads) is
exactly what this file must not do.
"""
from __future__ import annotations

import importlib
import json

import pytest

from tools.future._common import RECEIPTS
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
