"""Four model-ladder stages: BLOCKED_EXTERNAL with machine-readable wakes."""
from __future__ import annotations

from tools.theia.ladder import (
    BLOCKED_EXTERNAL,
    STAGES,
    WAKE_THEIA_LAB,
    WAKE_THEIA_MICRO,
    WAKE_THEIA_RESEARCH,
    WAKE_THEIA_WORKER,
    evaluate_wake,
)


def test_four_stages_blocked_external_never_absent_or_scaffolded():
    names = [s.name for s in STAGES]
    assert names == ["THEIA_MICRO", "THEIA_LAB", "THEIA_WORKER", "THEIA_RESEARCH"]
    for stage in STAGES:
        assert stage.status == BLOCKED_EXTERNAL
        assert stage.status not in {"ABSENT", "SCAFFOLDED"}
        ev = evaluate_wake(stage)
        assert ev.satisfied is False
        assert ev.status == BLOCKED_EXTERNAL
        assert ev.missing == tuple(stage.wake_condition["all_of"])


def test_wake_conditions_are_machine_readable():
    wakes = {
        "THEIA_MICRO": WAKE_THEIA_MICRO,
        "THEIA_LAB": WAKE_THEIA_LAB,
        "THEIA_WORKER": WAKE_THEIA_WORKER,
        "THEIA_RESEARCH": WAKE_THEIA_RESEARCH,
    }
    for stage in STAGES:
        wc = stage.wake_condition
        assert wc == wakes[stage.name]
        assert wc["id"] == f"WAKE.{stage.name}"
        assert wc["kind"] == "TRAINING_CAMPAIGN"
        assert wc["status_if_unmet"] == BLOCKED_EXTERNAL
        assert isinstance(wc["all_of"], list) and wc["all_of"]
        assert all(isinstance(p, str) for p in wc["all_of"])
        assert "not_sufficient" in wc
        assert "roadmap" in wc


def test_wake_does_not_fire_on_scaffold_evidence():
    fake = {
        "tools/theia directory exists": True,
        "import tools.theia succeeds": True,
        "empty weight file": True,
        "SCAFFOLDED status": True,
    }
    for stage in STAGES:
        ev = evaluate_wake(stage, fake)
        assert ev.satisfied is False
        assert ev.status == BLOCKED_EXTERNAL
