"""Focused regression for the native boundary-trace permission predicate.

Scope: hawking/hawking_native.py only.  This exercises the observable
behavior of ``native_boundary_trace_allowed`` under the constrained-worker
fence and under an explicit opt-in trace destination.  No hardware, no
native child process, and no release qualification is claimed here.
"""
from __future__ import annotations

import json

import pytest

from hawking import hawking_native


def test_boundary_trace_allowed_defaults_true():
    assert hawking_native.native_boundary_trace_allowed() is True


def test_suppress_fence_disables_boundary_trace_permission():
    assert hawking_native.native_boundary_trace_allowed() is True
    with hawking_native.suppress_native_runtime_spawning():
        assert hawking_native.native_boundary_trace_allowed() is False
        assert hawking_native.native_runtime_spawning_allowed() is False
    assert hawking_native.native_boundary_trace_allowed() is True
    assert hawking_native.native_runtime_spawning_allowed() is True


def test_boundary_trace_predicate_gates_emission(tmp_path, monkeypatch):
    destination = tmp_path / "boundary.jsonl"
    monkeypatch.setenv("HAWKING_BOUNDARY_TRACE", str(destination))

    hawking_native._boundary_trace("unit.allowed", detail="visible")
    with hawking_native.suppress_native_runtime_spawning():
        assert hawking_native.native_boundary_trace_allowed() is False
        hawking_native._boundary_trace("unit.suppressed", detail="hidden")

    records = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [record["event"] for record in records]
    assert "unit.allowed" in events
    assert "unit.suppressed" not in events


def test_boundary_trace_predicate_is_context_local():
    import contextvars

    def probe():
        return hawking_native.native_boundary_trace_allowed()

    ctx = contextvars.copy_context()
    with hawking_native.suppress_native_runtime_spawning():
        assert ctx.run(probe) is True
        assert probe() is False
    assert ctx.run(probe) is True