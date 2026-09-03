from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_l0_l1_same_runtime_prefix_lifecycle as lifecycle
from lab.operators import ascension_qwen80_l0_l1_same_runtime_prefix_outer_runner as runner
from lab.receipts import verify


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLETE = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
OUTER_PREFLIGHT = COMPLETE / "QWEN80_L0_L1_STRICT_HOST_INTERFACE_OUTER_CPU_PREFLIGHT_20260809T114233Z/outer-preflight.json"
EXECUTION_BINDING = COMPLETE / "QWEN80_L0_L1_STRICT_HOST_INTERFACE_EXECUTION_BINDING_20260809T114245Z.json"
WATCHER_HOLD = COMPLETE / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220751Z.json"
HOST = REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device"


def _green_snapshot() -> dict[str, object]:
    return {
        "observed_at": "2026-08-09T11:00:00Z",
        "memory_free_percent": 91,
        "swap_used_bytes": 0,
        "q80_watcher_parent_pids": [22035],
        "q80_strict_joint_host_children": [],
        "q30_metal_or_capture_children": [],
    }


def test_host_process_detection_does_not_match_the_calling_shell() -> None:
    assert runner._is_strict_joint_host_process(
        {"command": f"{runner.Q80_HOST_EXECUTABLE} --mode metal"}
    )
    assert not runner._is_strict_joint_host_process(
        {"command": f"/bin/zsh -c shasum -a 256 {runner.Q80_HOST_EXECUTABLE}"}
    )


def test_resource_admission_is_sealed_and_refuses_pressure_or_competing_work(tmp_path: Path) -> None:
    green = runner.write_resource_admission(
        outer_preflight=OUTER_PREFLIGHT,
        execution_binding=EXECUTION_BINDING,
        watcher_hold=WATCHER_HOLD,
        out=tmp_path / "green-resource.json",
        snapshot_provider=_green_snapshot,
    )
    assert verify(green) == green
    assert green["status"] == runner.RESOURCE_STATUS
    assert green["prepared"] is True
    assert green["claim_boundary"]["metal_or_gpu_activity_performed"] is False

    blocked_snapshot = _green_snapshot()
    blocked_snapshot["memory_free_percent"] = 79
    blocked_snapshot["q30_metal_or_capture_children"] = [{"pid": 77, "command": "q30 capture"}]
    refused = runner.build_resource_admission(
        outer_preflight=OUTER_PREFLIGHT,
        execution_binding=EXECUTION_BINDING,
        watcher_hold=WATCHER_HOLD,
        snapshot=blocked_snapshot,
    )
    assert verify(refused) == refused
    assert refused["status"] == runner.RESOURCE_REFUSED_STATUS
    assert refused["prepared"] is False
    assert "memory free percentage is below 80" in refused["blockers"]
    assert "Q30 Metal/capture child is already active" in refused["blockers"]


def test_production_runner_reaps_fake_refusal_and_releases_without_second_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource_path = tmp_path / "resource.json"
    runner.write_resource_admission(
        outer_preflight=OUTER_PREFLIGHT,
        execution_binding=EXECUTION_BINDING,
        watcher_hold=WATCHER_HOLD,
        out=resource_path,
        snapshot_provider=_green_snapshot,
    )
    fake_child = tmp_path / "fake-refusal.py"
    fake_child.write_text("import sys\nprint('intentional CPU test refusal')\nsys.exit(23)\n", encoding="utf-8")
    capture = tmp_path / "capture"
    lease = tmp_path / "lease.json"
    release = tmp_path / "release.json"
    source_preflight_calls: list[tuple[Path, Path, int]] = []
    monkeypatch.setattr(
        runner,
        "_run_host_source_admission_preflight",
        lambda *, host, outer_preflight, workers: source_preflight_calls.append(
            (host, outer_preflight, workers)
        ),
    )
    result = runner.execute_one_shot(
        outer_preflight=OUTER_PREFLIGHT,
        execution_binding=EXECUTION_BINDING,
        watcher_hold=WATCHER_HOLD,
        resource_admission=resource_path,
        lease_out=lease,
        capture_dir=capture,
        release_out=release,
        host_binary=HOST,
        workers=1,
        timeout_seconds=5.0,
        snapshot_provider=_green_snapshot,
        child_command_for_test=(sys.executable, str(fake_child)),
    )
    assert verify(result) == result
    assert result["status"] == runner.RUNNER_STATUS
    assert source_preflight_calls == [(HOST.resolve(), OUTER_PREFLIGHT, 1)]
    terminal = json.loads((capture / lifecycle.OUTER_TERMINAL_FILENAME).read_text(encoding="utf-8"))
    terminal = verify(terminal)
    assert terminal["status"] == f"{lifecycle.OUTER_REFUSED_PREFIX}CHILD_NONZERO"
    assert terminal["child_terminal"]["reaped"] is True
    assert terminal["child"]["terminal"]["exit_code"] == 23
    released = verify(json.loads(release.read_text(encoding="utf-8")))
    assert released["capture_succeeded"] is False
    assert released["outer_terminal_status"] == terminal["status"]
    assert released["lease_id"] == verify(json.loads(lease.read_text(encoding="utf-8")))["lease_id"]

    with pytest.raises(runner.JointOuterRunnerError, match="capture directory must be new before lease issuance"):
        runner.execute_one_shot(
            outer_preflight=OUTER_PREFLIGHT,
            execution_binding=EXECUTION_BINDING,
            watcher_hold=WATCHER_HOLD,
            resource_admission=resource_path,
            lease_out=lease,
            capture_dir=capture,
            release_out=tmp_path / "second-release.json",
            host_binary=HOST,
            workers=1,
            timeout_seconds=5.0,
            snapshot_provider=_green_snapshot,
            child_command_for_test=(sys.executable, str(fake_child)),
        )
