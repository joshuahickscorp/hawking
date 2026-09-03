"""AgentOS worker turns persist real HCLI outcomes without lineage authority."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from lab.lineage.identity import make_qwen38_genesis
from lab.lineage.lifecycle import WorkerRegistry, generation_record
from lab.lineage.state import LineageState
from lab.receipts import verify


REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "tools" / "genesis_agentos.py"


def _load():
    spec = importlib.util.spec_from_file_location("genesis_agentos_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agentos = _load()


def _state_and_workers(tmp_path: Path) -> tuple[Path, WorkerRegistry]:
    parent = make_qwen38_genesis()
    parent.physical_bpw = 4.25
    lineage = LineageState()
    lineage.install(parent)
    state_path = tmp_path / "lineage.json"
    state_path.write_text(json.dumps(lineage.to_dict()))
    registry = WorkerRegistry(tmp_path / "workers.json")
    registry.bootstrap(generation=generation_record(parent, repo_head="0" * 40), repo=REPO)
    workers, previous_sha = registry.load()
    for worker in workers:
        home = tmp_path / "worker-worktrees" / worker["worker_id"]
        home.mkdir(parents=True)
        durable = worker["durable_task_state"]
        durable["worktree"] = str(home)
        durable["worktree_isolated"] = True
    registry.replace(workers, expected_previous_sha256=previous_sha)
    return state_path, registry


def test_agentos_turn_claims_worker_and_persists_actual_hcli_result(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, registry = _state_and_workers(tmp_path)
    monkeypatch.setattr(agentos, "GPU_LOCK", tmp_path / "no-gpu-lock")
    monkeypatch.setattr(agentos, "DEFAULT_STOPFILE", tmp_path / "no-stop")
    seen: dict[str, object] = {}

    class FakeAct:
        ok = True
        text = "<tool_call> executed a bounded source inspection"

        def to_dict(self):
            return {"ok": True, "results": [{"name": "read", "ok": True}]}

    class FakeUnit:
        def act(self, prompt, **kwargs):
            seen["prompt"] = prompt
            seen["kwargs"] = kwargs
            return FakeAct()

    result = agentos.run_once(
        repo=REPO,
        state_path=state_path,
        worker_registry_path=registry.path,
        checkpoint_root=tmp_path / "checkpoints",
        candidate_root=tmp_path / "candidates",
        session_root=tmp_path / "sessions",
        worker_id="gravity",
        unit_factory=lambda _worker, _worktree: FakeUnit(),
    )
    verify(result, label="agentos-turn")
    assert result["outcome"] == "TURN_COMPLETE"
    assert result["authority_moved"] is not True
    assert "submit_candidate" in str(seen["prompt"])
    assert seen["kwargs"]["max_rounds"] == 1
    assert "bash" not in seen["kwargs"]["known_tools"]
    workers, _ = registry.load()
    gravity = next(row for row in workers if row["worker_id"] == "gravity")
    assert gravity["state"] == "READY"
    assert gravity["durable_task_state"]["tool_results"][-1]["outcome"] == "TURN_COMPLETE"
    assert "actual AgentOS HCLI turn" in gravity["durable_task_state"]["NEXT_ACTION"]
    assert (tmp_path / "checkpoints" / "gravity" / "checkpoint.json").is_file()


def test_next_worker_prompt_carries_the_actual_prior_tool_observation(tmp_path: Path) -> None:
    _state_path, registry = _state_and_workers(tmp_path)
    workers, _sha = registry.load()
    worker = next(row for row in workers if row["worker_id"] == "gravity")
    worker["durable_task_state"]["tool_results"] = [
        {
            "outcome": "TURN_COMPLETE",
            "ok": False,
            "summary": "model chose a directory read",
            "act": {
                "results": [
                    {"name": "read", "ok": False, "output": "missing path worker-home"}
                ]
            },
        }
    ]
    prompt = agentos._worker_prompt(
        worker=worker,
        context={"generation": worker["bound_generation"], "worker": worker["worker_id"]},
        candidate_root=tmp_path / "candidates",
    )
    assert "LAST_OBSERVED_TURN" in prompt
    assert "missing path worker-home" in prompt


def test_agentos_defers_without_claiming_when_gpu_lane_is_busy(tmp_path: Path, monkeypatch) -> None:
    state_path, registry = _state_and_workers(tmp_path)
    lock = tmp_path / "gpu-lock"
    lock.mkdir()
    monkeypatch.setattr(agentos, "GPU_LOCK", lock)
    monkeypatch.setattr(agentos, "DEFAULT_STOPFILE", tmp_path / "no-stop")
    result = agentos.run_once(
        repo=REPO,
        state_path=state_path,
        worker_registry_path=registry.path,
        checkpoint_root=tmp_path / "checkpoints",
        candidate_root=tmp_path / "candidates",
        session_root=tmp_path / "sessions",
    )
    verify(result, label="agentos-deferred")
    assert result["outcome"] == "DEFERRED_GPU_LANE_BUSY"
    workers, _ = registry.load()
    assert {row["state"] for row in workers} == {"READY"}


def test_agentos_defers_without_claiming_while_resident_is_not_ready(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, registry = _state_and_workers(tmp_path)
    monkeypatch.setattr(agentos, "GPU_LOCK", tmp_path / "no-gpu-lock")
    monkeypatch.setattr(agentos, "DEFAULT_STOPFILE", tmp_path / "no-stop")
    monkeypatch.setattr(agentos, "_resident_is_ready", lambda _repo: False)
    result = agentos.run_once(
        repo=REPO,
        state_path=state_path,
        worker_registry_path=registry.path,
        checkpoint_root=tmp_path / "checkpoints",
        candidate_root=tmp_path / "candidates",
        session_root=tmp_path / "sessions",
    )
    assert result["outcome"] == "DEFERRED_RESIDENT_NOT_READY"
    workers, _ = registry.load()
    assert {row["state"] for row in workers} == {"READY"}


def test_agentos_allows_a_proven_stale_gpu_lock_to_reach_resident_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, registry = _state_and_workers(tmp_path)
    lock = tmp_path / "gpu-lock"
    lock.mkdir()
    (lock / "pid").write_text("4242\n")
    monkeypatch.setattr(agentos, "GPU_LOCK", lock)
    monkeypatch.setattr(agentos, "DEFAULT_STOPFILE", tmp_path / "no-stop")
    monkeypatch.setattr(agentos, "_process_alive", lambda _pid: False)

    class FakeAct:
        ok = True
        text = "<tool_call> recovered"

        def to_dict(self):
            return {"ok": True, "results": []}

    class FakeUnit:
        def act(self, _prompt, **_kwargs):
            return FakeAct()

    result = agentos.run_once(
        repo=REPO,
        state_path=state_path,
        worker_registry_path=registry.path,
        checkpoint_root=tmp_path / "checkpoints",
        candidate_root=tmp_path / "candidates",
        session_root=tmp_path / "sessions",
        worker_id="gravity",
        unit_factory=lambda _worker, _worktree: FakeUnit(),
    )
    assert result["outcome"] == "TURN_COMPLETE"
