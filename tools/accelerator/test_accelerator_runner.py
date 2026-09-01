from __future__ import annotations

import json
from pathlib import Path

import pytest

import accelerator_runner as runner
import architecture_atlas as atlas


def test_atlas_compiles_to_parameterized_specs_and_hcli_workunits(tmp_path: Path):
    queue = runner.build_compiled_queue(repo_root=tmp_path)
    result = runner.validate_compiled_queue(queue)

    assert result["passed"] is True
    assert queue["experiment_spec_schema"] == runner.SPEC_SCHEMA
    assert queue["counts"]["specs"] == 9
    assert queue["counts"]["ready"] == 6
    assert queue["counts"]["work_units"] == 6
    assert queue["funnel"]["promotion"] == []
    assert all("complete_useful_wall_ns" in row["metrics"] for row in queue["specs"])
    assert all(row["runner"]["shell"] is False for row in queue["specs"])
    assert all(row["status"] == "pending" for row in queue["work_units"])
    assert all(row["resource_class"] for row in queue["work_units"])
    assert all(row["effect_class"] == "REVERSIBLE" for row in queue["work_units"])


def test_spec_contains_all_requested_runner_inputs_and_round_trips():
    document = atlas.build_atlas()
    specs = runner.compile_experiment_specs(document, model="Qwen27", backend="metal")

    assert specs
    spec = specs[0]
    raw = spec.to_dict()
    for field in (
        "model_identity",
        "nx_identity",
        "nr_identity",
        "organ_range",
        "backend",
        "kernel_lowering",
        "verification_mode",
        "benchmark_mode",
        "state_session_inputs",
        "output_receipt_path",
    ):
        assert raw[field]
    assert runner.validate_experiment_spec(raw)["passed"] is True
    restored = runner.AcceleratorExperimentSpec.from_dict(raw)
    assert restored == spec
    assert "--repo-root" in spec.command
    assert "--emit" in spec.command


def test_blocked_specs_are_visible_but_not_admitted_as_workunits():
    queue = runner.build_compiled_queue()
    blocked = [row for row in queue["specs"] if row["status"] == "BLOCKED"]
    assert blocked
    assert all(row["state_session_inputs"]["blocked_reason"] for row in blocked)
    assert not any(
        row["experiment_id"] == blocked[0]["experiment_id"]
        for row in queue["work_units"]
    )

    with pytest.raises(runner.RunnerSpecError, match="non-ready"):
        tampered = json.loads(json.dumps(queue))
        tampered["work_units"].append(
            {
                "experiment_id": blocked[0]["experiment_id"],
                "status": "blocked",
            }
        )
        runner.validate_compiled_queue(tampered)


def test_runner_is_plan_only_by_default_and_respects_quiescence(tmp_path: Path):
    specs = runner.compile_experiment_specs(atlas.build_atlas())
    ready = next(spec for spec in specs if spec.status == "READY" and spec.requires_quiescence)

    blocked_runner = runner.AcceleratorRunner(
        tmp_path,
        repo_root=tmp_path,
        quiescence=lambda: {"quiet": False, "contenders": [{"pid": 7}]},
    )
    waiting = blocked_runner.start(ready, execute=True)
    assert waiting["status"] == "WAITING_FOR_QUIESCENCE"
    assert waiting["started"] is False

    planned = blocked_runner.start(ready)
    assert planned["status"] == "PLANNED"
    assert planned["started"] is False


def test_runner_detaches_only_after_explicit_execute_and_quiet_window(tmp_path: Path):
    specs = runner.compile_experiment_specs(atlas.build_atlas())
    ready = next(spec for spec in specs if spec.status == "READY" and spec.requires_quiescence)
    calls: list[dict[str, object]] = []

    class FakeStore:
        def __init__(self, workspace, **kwargs):
            calls.append({"workspace": workspace, **kwargs})

        def start(self, argv, **kwargs):
            calls.append({"argv": list(argv), **kwargs})
            return {"job_id": "job-test", "state": "RUNNING"}

    launched = runner.AcceleratorRunner(
        tmp_path,
        repo_root=tmp_path,
        quiescence=lambda: {"quiet": True, "contenders": []},
        background_store_factory=FakeStore,
    ).start(ready, execute=True)

    assert launched["status"] == "DETACHED"
    assert launched["started"] is True
    assert calls[1]["argv"] == list(ready.command)
    assert calls[1]["cwd"] == tmp_path.resolve()


def test_blocked_runner_never_starts_a_job(tmp_path: Path):
    specs = runner.compile_experiment_specs(atlas.build_atlas())
    blocked = next(spec for spec in specs if spec.status == "BLOCKED")
    result = runner.AcceleratorRunner(tmp_path, repo_root=tmp_path).start(blocked, execute=True)

    assert result["status"] == "BLOCKED"
    assert result["started"] is False
