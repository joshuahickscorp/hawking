"""CPU-side tests for the Qwen3.8 special-unit harness (G005 / G015 legs)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.hcli.special_unit import (
    NativeDecodeError,
    NativeDecodeRefused,
    NativeQwen38Backend,
    ResourceClass,
    ResourceGate,
    ScriptedBackend,
    SessionStore,
    SpecialUnit,
    SpecialUnitError,
    StepStatus,
    consume_grok_task,
    looks_gpu_command,
    main,
    project_context,
)
from lab.layout import REPO_ROOT
from lab.verification_authority import AuthorityPrincipal, SelfPromotionError


@pytest.fixture
def unit(tmp_path: Path) -> SpecialUnit:
    return SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=ScriptedBackend(["hello from scripted backend"]),
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )


def test_conversation_records_context_and_reply(unit: SpecialUnit) -> None:
    turn = unit.say("what is the native-leg status?")
    assert turn.role == "assistant"
    assert "scripted" in turn.text
    assert turn.meta.get("produced_by") == "scripted"
    assert len(unit.session.transcript) == 2
    assert unit.session.context_digest
    ctx = project_context(REPO_ROOT)
    assert ctx["model_identity"]["native_leg"] == "DONE"
    assert ctx["git"]["head"]


def test_session_survives_process_restart(tmp_path: Path) -> None:
    first = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=ScriptedBackend(["one"]),
    )
    first.say("remember this")
    sid = first.session.session_id
    second = SpecialUnit.open(
        sid,
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
    )
    assert second.session.transcript[0].text == "remember this"
    assert second.session.transcript[1].text == "one"


def test_interrupt_and_resume(unit: SpecialUnit) -> None:
    unit.say("go")
    unit.interrupt("user")
    assert unit.session.status == "interrupted"
    unit.resume()
    assert unit.session.status == "idle"
    assert unit.session.pending_interrupt is None


def test_read_and_grep_tools(unit: SpecialUnit) -> None:
    read = unit.tool("read", {"path": "receipts/ascent-2026-08-16/G015_NATIVE_LEG_VERIFY_ON_MAIN.json", "limit": 40})
    assert read.ok
    assert "legs_STILL_OPEN_for_G015" in read.output
    grep = unit.tool("grep", {"pattern": "SPECIAL_UNIT_READY", "path": "receipts/ascent-2026-08-16", "max_hits": 10})
    assert grep.ok
    assert grep.detail["hits"] >= 1


def test_write_stays_in_owned_worktree(unit: SpecialUnit, tmp_path: Path) -> None:
    target = tmp_path / "wt" / "note.txt"
    result = unit.tool("write", {"path": str(target), "content": "hi"})
    assert result.ok
    assert target.read_text() == "hi"
    leaked = unit.tool("write", {"path": str(REPO_ROOT / "README.md"), "content": "nope"})
    assert not leaked.ok
    assert leaked.detail.get("denied") is True


def test_gpu_commands_refused(unit: SpecialUnit) -> None:
    assert looks_gpu_command("./tools/gpu_lane_lock.sh q80-x true")
    result = unit.tool("bash", {"command": "gpu_lane_lock.sh dsv4f-host cargo bench"})
    assert not result.ok
    assert "refused" in result.output


def test_pytest_tool_runs_real_lab_test(unit: SpecialUnit) -> None:
    result = unit.tool("pytest", {"target": "lab/tests/test_option_c.py", "timeout": 90})
    assert result.ok, result.output


def test_cargo_test_dry_and_gpu_refuse(unit: SpecialUnit) -> None:
    dry = unit.tool("cargo_test", {"package": "hide-kernel", "extra": ["--lib"], "dry_run": True})
    assert dry.ok
    assert dry.detail["dry_run"] is True
    refused = unit.tool("cargo_test", {"package": "hawking-core", "extra": ["--example", "ascension_qwen38_hybrid_greedy"]})
    assert not refused.ok


def test_grok_delegate_does_not_consume(unit: SpecialUnit, tmp_path: Path) -> None:
    contract = tmp_path / "wt" / "c.md"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text("do x\n")
    handle = unit.grok_delegate(slug="demo", contract=contract)
    assert handle["dispatched"] is True
    assert handle["consumed"] is False
    assert handle["verified_complete"] is False
    assert unit.unconsumed_grok()


def test_grok_consume_real_or_fixture(tmp_path: Path) -> None:
    tdir = tmp_path / "tasks" / "demo-1"
    tdir.mkdir(parents=True)
    (tdir / "status").write_text("done\n")
    (tdir / "exit_code").write_text("0\n")
    (tdir / "metadata.json").write_text(json.dumps({"task_id": "demo-1", "mode": "delegate"}))
    (tdir / "grok-report.md").write_text("CLAIM\nI shipped it.\n")
    (tdir / "diff.patch").write_text("diff --git a/foo.rs b/foo.rs\n")
    doc = consume_grok_task("demo-1", tmp_path / "tasks")
    assert doc["consumed"] is True
    assert doc["finished"] is True
    assert doc["verified_complete"] is False
    assert doc["files_in_diff"] == ["foo.rs"]
    assert doc["seal_sha256"]


def test_proposed_complete_is_not_verified_complete(unit: SpecialUnit) -> None:
    unit.plan("x", steps=[{"id": "s", "title": "S", "dependencies": [], "oracle": {"kind": "predicate"}}])
    unit.propose("s")
    step = unit.session.plan["steps"][0]
    assert step["status"] == StepStatus.PROPOSED_COMPLETE.value
    assert step["status"] != StepStatus.VERIFIED_COMPLETE.value
    with pytest.raises(SelfPromotionError):
        unit.verify("s", principal=AuthorityPrincipal.SANDBOX_MODEL, certifier_id="model")
    unit.verify("s")
    assert unit.session.plan["steps"][0]["status"] == StepStatus.VERIFIED_COMPLETE.value


def test_verify_requires_verified_dependencies(unit: SpecialUnit) -> None:
    unit.plan(
        "dep",
        steps=[
            {"id": "a", "title": "A", "dependencies": [], "oracle": {"kind": "predicate"}},
            {"id": "b", "title": "B", "dependencies": ["a"], "oracle": {"kind": "predicate"}},
        ],
    )
    with pytest.raises(SpecialUnitError, match="dependency a"):
        unit.verify("b")
    unit.verify("a")
    unit.verify("b")
    assert unit.session.plan["steps"][1]["status"] == StepStatus.VERIFIED_COMPLETE.value


def test_resource_gate_pauses_for_protected_owner(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner").write_text("auto-dsv4f-host-exclusive\n")
    gate = ResourceGate(lock_path=lock, allow_gpu=True)
    live, why = gate.protected_bench_live()
    assert live
    assert "protected" in why
    assert gate.admit(ResourceClass.CPU)[0] is True
    assert gate.admit(ResourceClass.GPU_HEAVY)[0] is False
    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        gate=gate,
    )
    ok, _ = unit.pause_for_resources(ResourceClass.GPU_HEAVY)
    assert ok is False
    assert unit.session.status == "paused"
    unit.resume()
    assert unit.session.status == "paused"
    (lock / "owner").unlink()
    lock.rmdir()
    unit.resume()
    assert unit.session.status == "idle"


def test_resource_gate_never_creates_real_lock(tmp_path: Path) -> None:
    # Using a fake path; the production lock must not appear as a side effect.
    before = Path("/tmp/hawking-gpu-lane.lock").exists()
    gate = ResourceGate(lock_path=tmp_path / "absent-lock")
    gate.admit(ResourceClass.GPU_HEAVY)
    after = Path("/tmp/hawking-gpu-lane.lock").exists()
    assert after == before


def test_cli_new_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["new", "--repo", str(REPO_ROOT), "--session-root", str(tmp_path / "s")])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    sid = payload["session_id"]
    rc = main(["show", "--repo", str(REPO_ROOT), "--session-root", str(tmp_path / "s"), "--session", sid])
    assert rc == 0


def test_session_store_missing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "none")
    with pytest.raises(SpecialUnitError):
        store.load("ghost")


def test_native_refuse_task_is_pass_not_skip(tmp_path: Path) -> None:
    from lab.hcli.claude_offload_bench import _t_native_refuse_protected

    ok, detail = _t_native_refuse_protected(tmp_path)
    assert ok
    assert not detail.startswith("SKIP ")
    assert "refused protected lock" in detail


def test_offload_catalog_cites_real_repo_files() -> None:
    from lab.hcli.claude_offload_bench import G015, PRODUCER_HARNESS, PRODUCER_MODEL, TASKS

    assert G015.is_file()
    assert len(TASKS) >= 12
    assert all(t.source for t in TASKS)
    assert all(t.routine_claude for t in TASKS)
    producers = {t.producer for t in TASKS}
    assert PRODUCER_HARNESS in producers
    assert PRODUCER_MODEL in producers
    assert any(t.id == "native_qwen38_say" and t.producer == PRODUCER_MODEL for t in TASKS)


def test_native_backend_refuses_protected_lock_without_invoke(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner").write_text("q80-mixed-bench\n", encoding="utf-8")
    called: list[list[str]] = []

    def runner(cmd: list[str]) -> object:
        called.append(cmd)
        raise AssertionError("must not invoke generate")

    before = Path("/tmp/hawking-gpu-lane.lock").exists()
    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=ResourceGate(lock_path=lock, allow_gpu=True),
        runner=runner,  # type: ignore[arg-type]
    )
    with pytest.raises(NativeDecodeRefused, match="protected"):
        backend.complete("Say hi.", {})
    assert called == []
    assert Path("/tmp/hawking-gpu-lane.lock").exists() == before


def test_native_backend_refuses_nonprotected_holder(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner").write_text("some-other-lane\n", encoding="utf-8")
    called: list[list[str]] = []

    def runner(cmd: list[str]) -> object:
        called.append(cmd)
        raise AssertionError("must not contend")

    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=ResourceGate(lock_path=lock),
        runner=runner,  # type: ignore[arg-type]
    )
    with pytest.raises(NativeDecodeRefused, match="will not contend"):
        backend.complete("Say hi.", {})
    assert called == []


def test_native_backend_command_wraps_gpu_lane_lock(tmp_path: Path) -> None:
    import json
    import subprocess

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        assert Path(cmd[0]).name == "gpu_lane_lock.sh"
        assert cmd[1] == "qwen38-special-unit"
        assert "ascension_qwen38_hybrid_greedy" in cmd[2]
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(
            json.dumps(
                {
                    "generated_text": "hello from native stub",
                    "fallbacks": 0,
                    "new_token_ids": [1, 2, 3],
                    "median_gpu_ns_per_token": 33500000,
                    "wall_ns": 1000,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="GENERATED_TEXT_VERBATIM: hello from native stub\nFALLBACKS: 0\n", stderr="")

    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
        runner=runner,
    )
    text = backend.complete("Say hi.", {})
    assert text == "hello from native stub"
    assert backend.last_receipt is not None
    assert backend.last_receipt["fallbacks"] == 0
    assert backend.last_receipt["used_gpu_lane_lock"] is False


def test_native_backend_rejects_nonzero_fallbacks(tmp_path: Path) -> None:
    import json
    import subprocess

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"generated_text": "x", "fallbacks": 2}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="FALLBACKS: 2\n", stderr="")

    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
        runner=runner,
    )
    with pytest.raises(NativeDecodeError, match="fallbacks=2"):
        backend.complete("Say hi.", {})


def test_say_records_model_producer_from_native_stub(tmp_path: Path) -> None:
    import json
    import subprocess

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"generated_text": "hi from model", "fallbacks": 0}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
        runner=runner,
    )
    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=backend,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    turn = unit.say("Say hi.")
    assert turn.meta.get("produced_by") == "model"
    assert turn.meta.get("native", {}).get("fallbacks") == 0
    assert turn.text == "hi from model"


def test_proposed_complete_fence_holds_with_native_stub(tmp_path: Path) -> None:
    import json
    import subprocess

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"generated_text": "ok", "fallbacks": 0}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=NativeQwen38Backend(
            repo=REPO_ROOT,
            gate=ResourceGate(lock_path=tmp_path / "no-lock"),
            runner=runner,
        ),
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    unit.say("plan this")
    unit.plan("x", steps=[{"id": "s", "title": "S", "dependencies": [], "oracle": {"kind": "predicate"}}])
    unit.propose("s", author="qwen38")
    step = unit.session.plan["steps"][0]
    assert step["status"] == StepStatus.PROPOSED_COMPLETE.value
    assert step["status"] != StepStatus.VERIFIED_COMPLETE.value
    with pytest.raises(SelfPromotionError):
        unit.verify("s", principal=AuthorityPrincipal.SANDBOX_MODEL, certifier_id="qwen38")
