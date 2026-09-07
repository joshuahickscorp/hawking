"""Canonical producers are the implementations behind planning tools."""
from __future__ import annotations

from hcli.tool_registry import ToolContext, _doctor_query, _gravity_experiment


def test_doctor_query_calls_the_canonical_control_producer(monkeypatch, tmp_path):
    called = []

    def controls():
        called.append(True)
        return {"all_three_fail_on_broken": True}

    monkeypatch.setattr("tools.doctor.engine.zeros_controls", controls)
    result = _doctor_query(ToolContext(tmp_path, tmp_path), {})
    assert result["status"] == "CONTROLLED_PROPOSAL"
    assert result["producer"] == "tools.doctor.engine.zeros_controls"
    assert called == [True]


def test_doctor_query_dispatches_targeted_diagnosis(monkeypatch, tmp_path):
    target = tmp_path / "receipts" / "receipt.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    seen = []

    def diagnose(value):
        seen.append(value)
        return {"overall": "UNKNOWN"}

    monkeypatch.setattr("tools.doctor.engine.diagnose", diagnose)
    result = _doctor_query(
        ToolContext(tmp_path, tmp_path), {"receipt": str(target)}
    )
    assert result["status"] == "DIAGNOSED"
    assert result["producer"] == "tools.doctor.engine.diagnose"
    assert seen == [str(target)]


def test_gravity_execute_dispatches_one_bounded_representation_runner(monkeypatch, tmp_path):
    called = []

    def run(*, root=None):
        called.append(root)
        return {"status": "PASSED", "whole_model_capability": "NOT_TESTED"}

    monkeypatch.setattr(
        "hcli.agentos.flash_representation_experiment.run_flash_representation_experiment",
        run,
    )
    result = _gravity_experiment(
        ToolContext(tmp_path, tmp_path), {"model": str(tmp_path), "execute": True}
    )
    assert result["status"] == "EXECUTED"
    assert result["producer"].endswith("run_flash_representation_experiment")
    assert result["capability_claim"] == "none"
    assert called == [str(tmp_path)]
