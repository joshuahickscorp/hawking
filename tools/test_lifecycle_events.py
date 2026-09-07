"""Lifecycle event audit: existing buses, real call sites, cheap connections."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.lifecycle_events import (
    CHILD_QUALIFIED,
    EXPERIMENT_COMPLETED,
    HARDWARE_PROFILE_CHANGED,
    LAW_UPDATED,
    LIFECYCLE_KINDS,
    RESOURCE_AVAILABLE,
    SPECIMEN_SEALED,
    on_child_qualified,
    on_hardware_profile_changed,
    route,
    subscriber_table,
)
from tools.successor_select import Refused

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text()


def test_six_named_lifecycle_events_are_audited():
    rows = {r["event"]: r for r in subscriber_table()}
    assert set(rows) == set(LIFECYCLE_KINDS)
    assert set(LIFECYCLE_KINDS) == {
        SPECIMEN_SEALED,
        EXPERIMENT_COMPLETED,
        LAW_UPDATED,
        CHILD_QUALIFIED,
        RESOURCE_AVAILABLE,
        HARDWARE_PROFILE_CHANGED,
    }
    for event, row in rows.items():
        assert row["status"] in {"live_consumer", "no consumer"}, event
        assert row["consumer"], event
        assert row["call_site"], event
        # Import-only rows are forbidden: live rows must name an invocation.
        # Receipt-only rows must say explicitly that there is no call site.
        if row["status"] == "live_consumer":
            assert "(" in row["call_site"] or "->" in row["call_site"], event
        else:
            assert "no production call site" in row["call_site"], event


def test_specimen_sealed_live_consumer_is_modellake_events_consume():
    watch = _read("tools/odyssey/modellake_watch.py")
    events = _read("tools/future/modellake_events.py")
    assert "def emit_modellake_events_once(" in watch
    assert "me.build()" in watch
    assert "maybe_emit_modellake_events" in watch
    assert "def consume(" in events
    assert "def build(" in events
    # build() invokes consume — the watcher's production call site.
    assert "emitted = consume(ledger)" in events
    row = next(r for r in subscriber_table() if r["event"] == SPECIMEN_SEALED)
    assert row["status"] == "live_consumer"
    assert "consume" in row["consumer"]


def test_experiment_completed_receipt_only_trial_has_no_product_consumer():
    row = next(r for r in subscriber_table() if r["event"] == EXPERIMENT_COMPLETED)
    assert row["status"] == "no consumer"
    assert "no production call site" in row["call_site"]


def test_law_updated_receipt_only_trigger_has_no_product_consumer():
    row = next(r for r in subscriber_table() if r["event"] == LAW_UPDATED)
    assert row["status"] == "no consumer"
    assert "no production call site" in row["call_site"]


def test_resource_available_uses_existing_profile_router():
    row = next(r for r in subscriber_table() if r["event"] == RESOURCE_AVAILABLE)
    assert row["status"] == "live_consumer"
    assert "on_hardware_profile_changed" in row["consumer"]
    assert "route(RESOURCE_AVAILABLE" in row["call_site"]


def test_child_qualified_consumer_calls_successor_gate():
    rec = {
        "state": "QUALIFIED",
        "identity": {"candidate_id": "child-a"},
        "metrics": {"effective_bpw": 3.0, "token_ns": 80},
        "flags": {
            "doctor_pass": True,
            "provenance_valid": True,
            "native_path": True,
            "no_hidden_fallback": True,
        },
    }
    out = on_child_qualified(rec)
    assert out["event"] == CHILD_QUALIFIED
    assert out["hard_gate_ran"] is True
    assert out["installed"] is False
    assert "successor_select.gate" in out["consumer"]
    broken = dict(rec)
    broken["flags"] = dict(rec["flags"], doctor_pass=False)
    with pytest.raises(Refused):
        on_child_qualified(broken)


def test_hardware_profile_changed_covers_all_blocked_gates():
    import json
    graph = json.loads((REPO / "civilization/CAPABILITY_GRAPH.json").read_text())
    woken: list[str] = []
    out = on_hardware_profile_changed(graph, wake_sleeping=lambda: woken.append("x") or woken)
    assert out["event"] == HARDWARE_PROFILE_CHANGED
    assert out["n_blocked_hardware"] == 13
    assert out["wake_ids"]
    assert all(w in {"U50_PRESENT", "HMF_PRESENT", "DGX_PRESENT",
                     "NEW_M_SERIES_PRESENT", "EGPU_PRESENT"} for w in out["wake_ids"])
    # This host has no U50/DGX/HMF/eGPU; activable stays empty and we must
    # not call wake_sleeping with a fabricated True.
    assert out["activable"] == []
    assert woken == []
    assert len(out["sleeping"]) == 13


def test_route_does_not_invent_a_second_bus_for_already_connected_events():
    out = route(SPECIMEN_SEALED, {})
    assert out["routed"] is False
    assert "consume" in out["consumer"]
    with pytest.raises(ValueError):
        route("not_an_event", {})


def test_roadmap_main_invokes_hardware_profile_changed():
    src = _read("tools/roadmap/__main__.py")
    assert "on_hardware_profile_changed" in src
    assert "on_hardware_profile_changed(doc)" in src
    row = next(r for r in subscriber_table() if r["event"] == HARDWARE_PROFILE_CHANGED)
    assert row["status"] == "live_consumer"
    assert "on_hardware_profile_changed" in row["consumer"]


def test_qualify_invokes_on_child_qualified():
    src = _read("tools/selection_contract.py")
    assert "on_child_qualified" in src
    assert "on_child_qualified(out)" in src
