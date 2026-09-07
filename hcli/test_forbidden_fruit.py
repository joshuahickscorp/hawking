"""Tests for the HCLI-callable Apple Neural Engine probe lab.

The load-bearing guard: ``observed_placement`` must report exactly what a
live MLComputePlan says and nothing else -- it must not invent a NEURAL
ENGINE preference when none was observed, and it must recognize one when it
is. Everything else here either refuses fast (no subprocess) or, in the one
end-to-end test, actually compiles and runs the real probes against the
real compiled fixture -- because a lab that only "passes" against fakes is
not evidence the shell-only lab is now HCLI-callable.
"""
from __future__ import annotations

import inspect
import json

import pytest

from hcli import forbidden_fruit as ff


def test_observed_placement_reports_cpu_without_inventing_ane():
    cpu_profile = {
        "mlcomputeplan": {
            "status": "PLANNED",
            "api": "MLComputePlan.load(contentsOf:configuration:)",
            "operations": [
                {"operator": "ios16.mul", "preferred": "CPU", "supported": ["CPU", "GPU", "NEURAL_ENGINE"]}
            ],
        }
    }
    result = ff.observed_placement(cpu_profile)
    assert result["preferred_devices_observed"] == ["CPU"]
    assert result["ane_preferred_this_run"] is False
    assert result["evidence_class"] == "PUBLIC_API_OBSERVED"


def test_observed_placement_recognizes_a_real_ane_preference():
    ane_profile = {
        "mlcomputeplan": {
            "status": "PLANNED",
            "operations": [
                {"operator": "ios16.mul", "preferred": "NEURAL_ENGINE", "supported": ["CPU", "NEURAL_ENGINE"]}
            ],
        }
    }
    result = ff.observed_placement(ane_profile)
    assert result["ane_preferred_this_run"] is True


def test_observed_placement_without_a_live_plan_is_not_measured():
    result = ff.observed_placement({})
    assert result["ane_preferred_this_run"] is False
    assert result["evidence_class"] == "NOT_MEASURED"


def test_time_predict_refuses_unknown_compute_units():
    with pytest.raises(ff.ForbiddenFruitRefused):
        ff.time_predict("/tmp/nonexistent.mlmodelc", compute_units="gpuOnly", repeats=1)


def test_time_predict_refuses_non_positive_repeats():
    with pytest.raises(ff.ForbiddenFruitRefused):
        ff.time_predict("/tmp/nonexistent.mlmodelc", compute_units="all", repeats=0)


def test_resolve_sdk_refuses_a_missing_path():
    with pytest.raises(ff.ForbiddenFruitRefused):
        ff.resolve_sdk("/nonexistent/sdk/path.sdk")


def test_run_forbidden_fruit_lab_refuses_a_missing_fixture(tmp_path):
    emit = tmp_path / "receipt.json"
    report = ff.run_forbidden_fruit_lab(
        compiled_model=str(tmp_path / "no_such.mlmodelc"), emit=str(emit)
    )
    assert report["status"] == "REFUSED"
    assert "not on disk" in report["errors"][0]["message"]
    assert json.loads(emit.read_text())["status"] == "REFUSED"


@pytest.mark.slow
@pytest.mark.skipif(
    not ff.DEFAULT_FIXTURE.is_dir(),
    reason="requires the checked-in Core ML compiled fixture",
)
def test_run_forbidden_fruit_lab_runs_the_real_fixture_and_reports_observed_placement(tmp_path):
    """End-to-end: compiles and runs the real probes against the real fixture.

    This is the HCLI-callable surface the mission asked for. It does not
    require ANE placement to succeed -- no run ever has -- only that the
    receipt is honest about whatever this run actually observed.
    """
    emit = tmp_path / "receipt.json"
    report = ff.run_forbidden_fruit_lab(repeats=2, emit=str(emit))
    assert report["status"] == "PASSED", report["errors"]
    assert report["neural_engine_present"] is True
    assert report["placement"]["preferred_devices_observed"]
    assert report["placement"]["ane_preferred_this_run"] == report["ane_placement_observed_this_run"]
    assert report["timing_status"] == "MEASURED"
    assert report["predict"]["warm_predict_ns"]
    assert report["concurrent_pair"]["instances"] == 2
    written = json.loads(emit.read_text())
    assert written["receipt_path"] == report["receipt_path"]


def test_forbidden_fruit_lab_is_reachable_through_the_typed_registry(tmp_path, monkeypatch):
    """The tool a resident actually calls, not the function underneath it.

    Every test above calls ``ff.run_forbidden_fruit_lab`` directly and stayed
    green while ``registry.invoke("forbidden_fruit.lab", ...)`` raised
    ``TypeError: unexpected keyword argument 'timeout_s'`` on every call --
    the handler forwarded a schema property the function never accepted, so
    the only route a model has to this lab was dead. Exercise the registry
    path, or a signature drift here is invisible again.
    """
    from hcli.tool_registry import default_tool_registry

    registry = default_tool_registry(tmp_path, repo_root=tmp_path)
    spec = registry.get("forbidden_fruit.lab")
    assert spec is not None, "forbidden_fruit.lab is not registered"

    # The schema may not advertise an argument the handler cannot deliver.
    declared = set((spec.input_schema.get("properties") or {}))
    accepted = set(
        inspect.signature(ff.run_forbidden_fruit_lab).parameters
    )
    assert declared <= accepted, (
        f"forbidden_fruit.lab advertises {sorted(declared - accepted)}, "
        f"which run_forbidden_fruit_lab() does not accept"
    )

    # The handler must actually reach the function, with arguments it accepts.
    # Stubbed on purpose: the real lab compiles Swift and runs the fixture, and
    # `test_run_forbidden_fruit_lab_runs_the_real_fixture...` already owns that
    # evidence. Re-running it here cost 26.26s of a 77.82s suite and proved
    # nothing the assertion above does not.
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"status": "PASSED", "stubbed": True}

    monkeypatch.setattr(ff, "run_forbidden_fruit_lab", fake)
    result = registry.invoke("forbidden_fruit.lab", {})
    assert result.ok, result.error
    assert result.value["status"] == "PASSED"
    assert set(seen) <= accepted, f"handler passed {sorted(set(seen) - accepted)}"
