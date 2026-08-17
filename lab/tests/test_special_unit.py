"""CPU-side tests for the Qwen3.8 special-unit harness (G005 / G015 legs)."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from lab.hcli.special_unit import (
    DEFAULT_TOOL_NAMES,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_MAX_NEW_TOKENS,
    TOOL_MAX_SEQ_LEN,
    WORKER_SESSION_ROLES,
    GenesisResidentBackend,
    MalformedToolCall,
    NativeDecodeError,
    NativeDecodeRefused,
    NativeQwen38Backend,
    ResidentRefused,
    ResidentReplyRejected,
    ResourceClass,
    ResourceGate,
    ScriptedBackend,
    SessionStore,
    SpecialUnit,
    SpecialUnitError,
    StepStatus,
    consume_grok_task,
    looks_gpu_command,
    looks_lifecycle_authority_command,
    main,
    parse_tool_calls,
    project_context,
    render_tool_prompt,
    validate_resident_reply,
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


def test_worker_cannot_bypass_external_promotion_authority(unit: SpecialUnit) -> None:
    command = "python3 tools/genesis_lifecycle.py promote --request child.json"
    assert looks_lifecycle_authority_command(command)
    result = unit.tool("bash", {"command": command})
    assert not result.ok
    assert result.detail.get("authority_boundary") is True
    assert "submit_candidate" in result.output


def test_pytest_tool_runs_real_lab_test(unit: SpecialUnit) -> None:
    result = unit.tool("pytest", {"target": "lab/tests/test_option_c.py", "timeout": 90})
    assert result.ok, result.output


def test_cargo_test_dry_and_gpu_refuse(unit: SpecialUnit) -> None:
    dry = unit.tool("cargo_test", {"package": "hide-kernel", "extra": ["--lib"], "dry_run": True})
    assert dry.ok
    assert dry.detail["dry_run"] is True
    refused = unit.tool("cargo_test", {"package": "hawking-core", "extra": ["--example", "ascension_qwen38_hybrid_greedy"]})
    assert not refused.ok


def test_cargo_test_accepts_string_extra(unit: SpecialUnit) -> None:
    dry = unit.tool(
        "cargo_test",
        {"package": "hide-protocol", "extra": "ids_serialize_transparently_as_bare_strings", "dry_run": True},
    )
    assert dry.ok
    argv = dry.detail.get("argv") or []
    assert "hide-protocol" in argv
    assert "ids_serialize_transparently_as_bare_strings" in argv
    assert argv.count("i") == 0


def test_prepare_candidate_stages_only_an_owned_unqualified_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lab.lineage.lifecycle as lifecycle

    worktree = tmp_path / "worker-home"
    worktree.mkdir()
    unit = SpecialUnit(
        repo=worktree,
        session_root=tmp_path / "sessions",
        owned_worktree=worktree,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    seen: dict[str, object] = {}

    def fake_builder(**kwargs):
        seen.update(kwargs)
        return {
            "schema": "hawking.genesis.candidate_request.v1",
            "candidate": {"instance_id": "genesis-g1", "generation": 1},
        }

    monkeypatch.setattr(lifecycle, "build_candidate_request", fake_builder)
    result = unit.tool(
        "prepare_candidate",
        {
            "output": "candidate.json",
            "instance_id": "genesis-g1",
            "artifact_root": "artifact",
            "next_bottleneck": "representation bytes",
        },
    )
    assert result.ok
    assert result.detail["qualified"] is False
    assert (worktree / "candidate.json").is_file()
    assert seen["origin_repo"] == worktree
    assert seen["authority_repo"] == REPO_ROOT


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
    assert len(TASKS) >= 27
    assert all(t.source for t in TASKS)
    assert all(t.routine_claude for t in TASKS)
    assert all(t.producer in {PRODUCER_HARNESS, PRODUCER_MODEL} for t in TASKS)
    producers = {t.producer for t in TASKS}
    assert PRODUCER_HARNESS in producers
    assert PRODUCER_MODEL in producers
    assert any(t.id == "native_qwen38_say" and t.producer == PRODUCER_MODEL for t in TASKS)
    model_ids = {t.id for t in TASKS if t.producer == PRODUCER_MODEL}
    assert len(model_ids) >= 16
    assert "grep_proposed_complete" in model_ids
    assert "read_g015_open_legs" in model_ids
    assert "run_option_c_tests" in model_ids
    assert "write_and_pytest_small" in model_ids
    assert "extract_g003_remaining_factor" in model_ids
    assert "grep_nativedecode_file_line" in model_ids
    assert "cargo_test_hide_protocol_ids" in model_ids
    assert "extract_token_ns_grok_report" in model_ids
    assert "compare_g013_supersession" in model_ids
    assert "merge_guard_device_topk" in model_ids
    assert "superwave_names_g013_v2" in model_ids
    assert "classify_storage_vs_active_bpw" in model_ids
    assert "read_goal_md_obligations" in model_ids
    assert "ascent_state_q80_receipt" in model_ids
    harness_ids = {t.id for t in TASKS if t.producer == PRODUCER_HARNESS}
    assert "proposed_vs_verified" in harness_ids
    assert "native_refuses_protected_lock" in harness_ids
    assert "refuse_protected_gpu_command" in harness_ids
    assert "verification_authority_no_self_promote" in harness_ids
    assert "delegate_then_consume_roundtrip" in harness_ids


def test_goal_md_task_skips_naming_missing_path(tmp_path: Path) -> None:
    from lab.hcli.claude_offload_bench import GOAL_MD, _t_read_goal_md_obligations

    if GOAL_MD.is_file():
        pytest.skip("GOAL.md is present")
    ok, detail, produced = _t_read_goal_md_obligations(tmp_path)
    assert ok is False
    assert produced is None
    assert detail.startswith("SKIP ")
    assert "GOAL.md" in detail


def test_write_and_pytest_still_requires_pytest_call() -> None:
    import inspect

    from lab.hcli.claude_offload_bench import _t_write_and_pytest

    src = inspect.getsource(_t_write_and_pytest)
    assert "model did not call pytest" in src
    assert "need tiny.py and test_tiny.py" in src
    assert "MUST call the" in src or "pytest tool" in src


def test_displacement_metric_is_not_rescored_to_model_expected() -> None:
    import inspect

    from lab.hcli.claude_offload_bench import run_bench

    src = inspect.getsource(run_bench)
    assert "model_passed / attempted" in src
    assert "model_passed / model_expected" not in src
    assert "model_expected_catalog" in src


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


def _call(name: str, arguments: dict) -> str:
    return (
        f"{TOOL_CALL_OPEN}\n"
        + json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
        + f"\n{TOOL_CALL_CLOSE}\n"
    )


def test_parse_tool_calls_well_formed_single() -> None:
    text = _call("grep", {"pattern": "proposed_complete", "path": "lab/hcli"})
    calls = parse_tool_calls(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].name == "grep"
    assert calls[0].arguments["pattern"] == "proposed_complete"


def test_parse_tool_calls_well_formed_multiple() -> None:
    text = _call("write", {"path": "tiny.py", "content": "x"}) + _call(
        "pytest", {"target": "test_tiny.py"}
    )
    calls = parse_tool_calls(text)
    assert calls is not None
    assert [c.name for c in calls] == ["write", "pytest"]


def test_parse_tool_calls_plain_text_is_none() -> None:
    assert parse_tool_calls("no tools here, just an answer") is None


def test_parse_tool_calls_refuses_unclosed() -> None:
    with pytest.raises(MalformedToolCall, match="opening tag without closing tag"):
        parse_tool_calls(f'{TOOL_CALL_OPEN}\n{{"name":"grep","arguments":{{}}}}')


def test_parse_tool_calls_refuses_close_without_open() -> None:
    with pytest.raises(MalformedToolCall, match="closing tag without opening tag"):
        parse_tool_calls(f'{TOOL_CALL_CLOSE}\n')


def test_parse_tool_calls_refuses_invalid_json() -> None:
    with pytest.raises(MalformedToolCall, match="not JSON"):
        parse_tool_calls(f"{TOOL_CALL_OPEN}\nnot-json\n{TOOL_CALL_CLOSE}")


def test_parse_tool_calls_refuses_extra_keys() -> None:
    body = json.dumps({"name": "grep", "arguments": {"pattern": "x"}, "extra": 1})
    with pytest.raises(MalformedToolCall, match="exactly name and arguments"):
        parse_tool_calls(f"{TOOL_CALL_OPEN}\n{body}\n{TOOL_CALL_CLOSE}")


def test_parse_tool_calls_refuses_args_alias() -> None:
    body = json.dumps({"name": "grep", "args": {"pattern": "x"}})
    with pytest.raises(MalformedToolCall, match="exactly name and arguments"):
        parse_tool_calls(f"{TOOL_CALL_OPEN}\n{body}\n{TOOL_CALL_CLOSE}")


def test_parse_tool_calls_refuses_string_arguments() -> None:
    body = json.dumps({"name": "grep", "arguments": "{\"pattern\":\"x\"}"})
    with pytest.raises(MalformedToolCall, match="arguments must be a JSON object"):
        parse_tool_calls(f"{TOOL_CALL_OPEN}\n{body}\n{TOOL_CALL_CLOSE}")


def test_parse_tool_calls_refuses_unknown_tool() -> None:
    with pytest.raises(MalformedToolCall, match="unknown tool"):
        parse_tool_calls(_call("rm", {"path": "/"}))


def test_parse_tool_calls_refuses_untagged_json() -> None:
    # Untagged JSON is a plain reply, not a repaired call.
    text = json.dumps({"name": "grep", "arguments": {"pattern": "proposed_complete"}})
    assert parse_tool_calls(text) is None


def test_parse_tool_calls_refuses_empty_body() -> None:
    with pytest.raises(MalformedToolCall, match="empty tool_call body"):
        parse_tool_calls(f"{TOOL_CALL_OPEN}\n\n{TOOL_CALL_CLOSE}")


def test_act_executes_scripted_well_formed_call(unit: SpecialUnit) -> None:
    unit.backend = ScriptedBackend(
        [_call("grep", {"pattern": "proposed_complete", "path": "lab/hcli", "max_hits": 10})]
    )
    result = unit.act("Search lab/hcli for proposed_complete.", known_tools=["grep"])
    assert result.malformed is None
    assert result.produced_by == "scripted"
    assert result.produced_by != "model"
    assert result.calls and result.calls[0].name == "grep"
    assert result.ok
    assert result.results[0].ok
    assert result.results[0].detail.get("hits", 0) >= 1


def test_act_refuses_malformed_and_does_not_execute(tmp_path: Path) -> None:
    # Alias "args" would work if we repaired. We must not.
    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=ScriptedBackend(
            [f'{TOOL_CALL_OPEN}\n{{"name":"write","args":{{"path":"x","content":"nope"}}}}\n{TOOL_CALL_CLOSE}']
        ),
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    result = unit.act("write a file", known_tools=["write"])
    assert result.ok is False
    assert result.malformed
    assert result.results == []
    assert result.calls is None
    assert not (tmp_path / "wt" / "x").exists()
    assert not any(t.role == "tool" for t in unit.session.transcript)


def test_act_refuses_malformed_json_and_does_not_execute(tmp_path: Path) -> None:
    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=ScriptedBackend([f"{TOOL_CALL_OPEN}\n{{name: grep}}\n{TOOL_CALL_CLOSE}"]),
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    result = unit.act("search", known_tools=["grep"])
    assert result.ok is False
    assert "not JSON" in (result.malformed or "")
    assert result.results == []


def test_act_multi_round_accumulates_model_chosen_calls(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    unit = SpecialUnit(
        repo=wt,
        session_root=tmp_path / "sessions",
        owned_worktree=wt,
        backend=ScriptedBackend(
            [
                _call("write", {"path": "tiny.py", "content": "def add(a, b):\n    return a + b\n"}),
                _call(
                    "write",
                    {
                        "path": "test_tiny.py",
                        "content": "from tiny import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                    },
                ),
                _call("pytest", {"target": "test_tiny.py"}),
                "done",
            ]
        ),
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    result = unit.act(
        "write tiny.py and test_tiny.py then pytest",
        known_tools=["write", "pytest"],
        max_rounds=4,
    )
    assert result.malformed is None
    assert result.produced_by == "scripted"
    assert result.calls is not None
    assert [c.name for c in result.calls] == ["write", "write", "pytest"]
    assert (wt / "tiny.py").is_file()
    assert (wt / "test_tiny.py").is_file()
    assert result.ok


def test_scripted_act_is_not_counted_as_model(unit: SpecialUnit) -> None:
    unit.backend = ScriptedBackend([_call("read", {"path": str(
        REPO_ROOT / "receipts/ascent-2026-08-16/G015_NATIVE_LEG_VERIFY_ON_MAIN.json"
    ), "limit": 20})])
    result = unit.act("read G015", known_tools=["read"])
    assert result.ok
    assert result.produced_by == "scripted"
    from lab.hcli.claude_offload_bench import PRODUCER_MODEL

    assert result.produced_by != PRODUCER_MODEL


def test_act_does_not_let_sandbox_verify(unit: SpecialUnit) -> None:
    unit.backend = ScriptedBackend([_call("grep", {"pattern": "x", "path": "lab/hcli"})])
    unit.act("search", known_tools=["grep"])
    unit.plan("x", steps=[{"id": "s", "title": "S", "dependencies": [], "oracle": {"kind": "predicate"}}])
    unit.propose("s", author="qwen38")
    step = unit.session.plan["steps"][0]
    assert step["status"] == StepStatus.PROPOSED_COMPLETE.value
    assert step["status"] != StepStatus.VERIFIED_COMPLETE.value
    with pytest.raises(SelfPromotionError):
        unit.verify("s", principal=AuthorityPrincipal.SANDBOX_MODEL, certifier_id="qwen38")


def test_render_tool_prompt_is_raw_chat() -> None:
    prompt = render_tool_prompt("Search lab/hcli for proposed_complete.", DEFAULT_TOOL_NAMES)
    assert prompt.startswith("<|im_start|>system\n")
    assert "<tools>" in prompt
    assert TOOL_CALL_OPEN in prompt
    assert prompt.rstrip().endswith("</think>")
    assert "<|im_start|>assistant\n" in prompt


def test_render_tool_prompt_for_resident_leaves_system_role_to_socket_contract() -> None:
    prompt = render_tool_prompt("Search lab/hcli.", ["grep"], user_only=True)
    assert prompt.startswith("<|im_start|>user\n")
    assert "<|im_start|>system" not in prompt
    assert "<tools>" in prompt
    assert prompt.rstrip().endswith("</think>")


def test_native_backend_tool_budget_uses_raw_prompt(tmp_path: Path) -> None:
    import subprocess

    seen: list[list[str]] = []

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"generated_text": _call("grep", {"pattern": "x"}), "fallbacks": 0}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
        runner=runner,
    )
    text = backend.complete(
        render_tool_prompt("x"),
        {"raw_prompt": True, "max_new_tokens": TOOL_MAX_NEW_TOKENS, "max_seq_len": TOOL_MAX_SEQ_LEN},
    )
    assert TOOL_CALL_OPEN in text
    assert seen
    cmd = seen[0]
    assert "--raw-prompt" in cmd
    assert cmd[cmd.index("--max-new-tokens") + 1] == str(TOOL_MAX_NEW_TOKENS)
    assert cmd[cmd.index("--max-seq-len") + 1] == str(TOOL_MAX_SEQ_LEN)
    assert backend.last_receipt is not None
    assert backend.last_receipt["raw_prompt"] is True
    assert backend.last_receipt["max_new_tokens"] == TOOL_MAX_NEW_TOKENS


def test_act_native_stub_records_model_producer(tmp_path: Path) -> None:
    import subprocess

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        assert "--raw-prompt" in cmd
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(
            json.dumps(
                {
                    "generated_text": _call(
                        "grep", {"pattern": "proposed_complete", "path": "lab/hcli"}
                    ),
                    "fallbacks": 0,
                }
            )
        )
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
    result = unit.act("Search lab/hcli for proposed_complete.", known_tools=["grep"])
    assert result.produced_by == "model"
    assert result.ok
    assert result.calls and result.calls[0].name == "grep"
    assert result.native and result.native.get("raw_prompt") is True


def test_act_refuses_protected_lock_without_invoke(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner").write_text("q80-mixed-bench\n", encoding="utf-8")
    called: list[list[str]] = []

    def runner(cmd: list[str]) -> object:
        called.append(cmd)
        raise AssertionError("must not invoke generate")

    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=NativeQwen38Backend(
            repo=REPO_ROOT,
            gate=ResourceGate(lock_path=lock, allow_gpu=True),
            runner=runner,  # type: ignore[arg-type]
        ),
        gate=ResourceGate(lock_path=lock, allow_gpu=True),
    )
    with pytest.raises(NativeDecodeRefused, match="protected"):
        unit.act("Search lab/hcli for proposed_complete.")
    assert called == []


def test_bench_skip_is_not_a_pass() -> None:
    from lab.hcli.claude_offload_bench import _is_skip

    assert _is_skip("SKIP missing /tmp/nope")
    assert not _is_skip("malformed tool call refused: extra key")
    assert not _is_skip("model produced no tool call: 'hi'")


def test_bench_unpack_preserves_measured_producer() -> None:
    from lab.hcli.claude_offload_bench import _unpack_run

    ok, detail, produced = _unpack_run((True, "hits=3", "model"))
    assert ok and produced == "model"
    ok, detail, produced = _unpack_run((True, "hits=3", "scripted"))
    assert produced == "scripted"
    ok, detail, produced = _unpack_run((True, "hits=3"))
    assert produced is None


# ---------------------------------------------------------------------------
# Opt-in Genesis resident worker adapter (CPU-only, injected client)
# ---------------------------------------------------------------------------

_RESIDENT_CONTRACT = {
    "schema": "hawking.genesis.system_contract_set.v1",
    "model": "qwen3.8",
    "binding_sha256": "3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f",
    "integrity_verified": True,
    "contracts": [
        {
            "canonical_path": "contracts/genesis/QWEN38_GENESIS_SYSTEM_DIRECTIVE.md",
            "sha256": "881ae469e0287cf386467002d3fc7951524b47054ac6d7f753b94a8e4e3ceff7",
            "size_bytes": 16414,
            "source_provenance": "injected-test",
            "integrity_verified": True,
        },
        {
            "canonical_path": "contracts/genesis/GENESIS_CONTINUITY_DIRECTIVE.md",
            "sha256": "c4a58bc06575effb8f759dbb22c49abfc65e1957910b18917d45d02592d1fdbc",
            "size_bytes": 11912,
            "source_provenance": "injected-test",
            "integrity_verified": True,
        },
        {
            "canonical_path": "contracts/genesis/GENESIS_OUTPUT_LAW.md",
            "sha256": "9679490e8ae623a6fdb408fd906a15d676bc55926580f6d7ed60e9ea610c9ada",
            "size_bytes": 4871,
            "source_provenance": "injected-test",
            "integrity_verified": True,
        },
    ],
}


def _resident_reply(**overrides: object) -> dict:
    body = {
        "ok": True,
        "protocol": "hawking.genesis.resident.v1",
        "text": "resident worker reply",
        "fallbacks": 0,
        "wall_ns": 1,
        "new_tokens": [1],
        "session": "child_a",
        "body_resident": True,
        "pid": os.getpid(),
        "genesis_system_contract": dict(_RESIDENT_CONTRACT),
        "genesis_contract_mode": "runtime_capsule_injected",
        "stub": True,
    }
    body.update(overrides)
    return body


class _StubResident:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict]] = []

    def propose(self, prompt: str, **kwargs: object) -> object:
        self.calls.append((prompt, dict(kwargs)))
        if callable(self.reply):
            return self.reply(prompt, **kwargs)
        return self.reply


_REPLY_UNSET = object()


def _resident_backend(
    tmp_path: Path,
    *,
    session_role: str = "child_a",
    reply: object = _REPLY_UNSET,
    client: _StubResident | None = None,
    lock_owner: str | None = None,
) -> tuple[GenesisResidentBackend, _StubResident]:
    lock = tmp_path / "gpu-lock"
    if lock_owner is not None:
        lock.mkdir()
        (lock / "owner").write_text(lock_owner + "\n", encoding="utf-8")
    stub = client or _StubResident(_resident_reply() if reply is _REPLY_UNSET else reply)
    backend = GenesisResidentBackend(
        session_role=session_role,
        gate=ResourceGate(lock_path=lock if lock_owner is not None else tmp_path / "no-lock"),
        client=stub,
        expected_contract=_RESIDENT_CONTRACT,
    )
    return backend, stub


def test_resident_injected_child_a_succeeds_without_native_binary(tmp_path: Path) -> None:
    NativeQwen38Backend.reset_counters()
    before_lock = Path("/tmp/hawking-gpu-lane.lock").exists()
    backend, stub = _resident_backend(tmp_path, session_role="child_a")
    text = backend.complete("summarize the open worker task", {})
    assert text == "resident worker reply"
    assert stub.calls
    assert stub.calls[0][0] == "summarize the open worker task"
    assert stub.calls[0][1]["session"] == "child_a"
    assert backend.last_receipt is not None
    assert backend.last_receipt["transport"] == "genesis_resident"
    assert backend.last_receipt["fallbacks"] == 0
    assert backend.last_receipt["used_gpu_lane_lock"] is False
    assert backend.last_receipt["binary"] is None
    assert backend.last_receipt["session_role"] == "child_a"
    assert backend.last_receipt["body_resident"] is True
    assert NativeQwen38Backend.lock_acquisitions == 0
    assert NativeQwen38Backend.generates_completed == 0
    assert Path("/tmp/hawking-gpu-lane.lock").exists() == before_lock
    src = inspect.getsource(GenesisResidentBackend.complete)
    assert "gpu_lane_lock.sh" not in src
    assert "lock_script" not in src
    assert "hybrid_greedy" not in src
    assert backend.last_receipt["used_gpu_lane_lock"] is False


def test_resident_injected_child_b_is_an_explicit_worker_home(tmp_path: Path) -> None:
    backend, stub = _resident_backend(
        tmp_path,
        session_role="child_b",
        reply=_resident_reply(session="child_b", text="from child_b"),
    )
    assert backend.complete("continue the kernel front", {}) == "from child_b"
    assert stub.calls[0][1]["session"] == "child_b"
    assert backend.last_receipt is not None
    assert backend.last_receipt["session_role"] == "child_b"


def test_resident_backend_refuses_parent_and_protected_test() -> None:
    gate = ResourceGate(lock_path=Path("/tmp/no-such-special-unit-lock"))
    for role in ("parent", "protected_test", "worker_c", ""):
        with pytest.raises(ResidentRefused, match="not worker homes"):
            GenesisResidentBackend(session_role=role, gate=gate, client=_StubResident(_resident_reply()))


def test_resident_backend_refuses_protected_lock_without_client_call(tmp_path: Path) -> None:
    NativeQwen38Backend.reset_counters()
    backend, stub = _resident_backend(tmp_path, lock_owner="q80-mixed-bench")
    with pytest.raises(ResidentRefused, match="protected"):
        backend.complete("Say hi.", {})
    assert stub.calls == []
    assert backend.propose_calls == 0
    assert NativeQwen38Backend.lock_acquisitions == 0


def test_resident_backend_does_not_contend_for_nonprotected_lock(tmp_path: Path) -> None:
    backend, stub = _resident_backend(tmp_path, lock_owner="some-other-lane")
    assert backend.complete("Say hi.", {}) == "resident worker reply"
    assert stub.calls and stub.calls[0][1]["session"] == "child_a"
    assert backend.last_receipt is not None
    assert backend.last_receipt["used_gpu_lane_lock"] is False


def test_resident_none_reply_is_refusal_not_synthetic_text(tmp_path: Path) -> None:
    backend, _stub = _resident_backend(tmp_path, reply=None)
    with pytest.raises(ResidentRefused, match="no result"):
        backend.complete("Say hi.", {})
    assert backend.last_receipt is None


def test_resident_ok_false_is_not_converted_to_model_text(tmp_path: Path) -> None:
    backend, _stub = _resident_backend(
        tmp_path,
        reply=_resident_reply(ok=False, text="I refuse but here is text anyway"),
    )
    with pytest.raises(ResidentReplyRejected, match="not ok"):
        backend.complete("Say hi.", {})
    assert backend.last_receipt is None


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"genesis_system_contract": None}, "provenance"),
        (
            {
                "genesis_system_contract": {
                    **_RESIDENT_CONTRACT,
                    "integrity_verified": False,
                }
            },
            "provenance",
        ),
        (
            {
                "genesis_system_contract": {
                    **_RESIDENT_CONTRACT,
                    "schema": "not-the-contract-set",
                }
            },
            "provenance",
        ),
        ({"body_resident": False}, "live body"),
        ({"body_resident": None}, "live body"),
        ({"text": ""}, "usable text"),
        ({"text": "   "}, "usable text"),
        ({"text": None}, "usable text"),
        ({"fallbacks": 1}, "fallbacks"),
        ({"fallbacks": "0"}, "fallbacks"),
        ({"fallbacks": None}, "fallbacks"),
        ({"session": "parent"}, "session"),
        (
            {"genesis_contract_mode": "protected_capability_prompt_preserved"},
            "protected capability",
        ),
    ],
)
def test_resident_invalid_reply_is_rejected(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    backend, _stub = _resident_backend(tmp_path, reply=_resident_reply(**overrides))
    with pytest.raises((ResidentRefused, ResidentReplyRejected), match=match):
        backend.complete("Say hi.", {})
    assert backend.last_receipt is None


def test_resident_expected_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    other = dict(_RESIDENT_CONTRACT)
    other["binding_sha256"] = "0" * 64
    backend, _stub = _resident_backend(tmp_path, reply=_resident_reply(genesis_system_contract=other))
    with pytest.raises(ResidentReplyRejected, match="expected binding"):
        backend.complete("Say hi.", {})


def test_resident_say_records_transport_and_worker_session(tmp_path: Path) -> None:
    backend, _stub = _resident_backend(tmp_path, session_role="child_a")
    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=backend,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    turn = unit.say("what is the next worker action?")
    assert turn.meta.get("produced_by") == "model"
    assert turn.meta.get("native", {}).get("transport") == "genesis_resident"
    assert turn.meta.get("native", {}).get("session_role") == "child_a"
    assert turn.meta.get("native", {}).get("used_gpu_lane_lock") is False
    assert turn.text == "resident worker reply"


def test_resident_act_uses_injected_client_not_native_runner(tmp_path: Path) -> None:
    NativeQwen38Backend.reset_counters()
    call = (
        f"{TOOL_CALL_OPEN}\n"
        + json.dumps({"name": "grep", "arguments": {"pattern": "proposed_complete", "path": "lab/hcli"}}, separators=(",", ":"))
        + f"\n{TOOL_CALL_CLOSE}\n"
    )
    backend, stub = _resident_backend(tmp_path, reply=_resident_reply(text=call))
    unit = SpecialUnit(
        repo=REPO_ROOT,
        session_root=tmp_path / "sessions",
        owned_worktree=tmp_path / "wt",
        backend=backend,
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
    )
    result = unit.act("Search lab/hcli for proposed_complete.", known_tools=["grep"])
    assert result.ok
    assert result.produced_by == "model"
    assert result.native and result.native.get("transport") == "genesis_resident"
    assert stub.calls[0][1]["session"] == "child_a"
    assert stub.calls[0][1]["raw"] is True
    assert stub.calls[0][0].startswith("<|im_start|>user\n")
    assert "<|im_start|>system" not in stub.calls[0][0]
    assert NativeQwen38Backend.lock_acquisitions == 0


def test_resident_missing_client_does_not_fall_back_to_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lab.hcli.special_unit as su

    NativeQwen38Backend.reset_counters()
    monkeypatch.setattr(su, "REPO_ROOT", tmp_path)
    backend = GenesisResidentBackend(
        session_role="child_a",
        gate=ResourceGate(lock_path=tmp_path / "no-lock"),
        client=None,
        expected_contract=_RESIDENT_CONTRACT,
    )
    with pytest.raises(ResidentRefused, match="inject a client"):
        backend.complete("Say hi.", {})
    assert backend.last_receipt is None
    assert NativeQwen38Backend.lock_acquisitions == 0
    assert NativeQwen38Backend.generates_completed == 0
    loader_src = inspect.getsource(su._load_resident_propose)
    assert "hybrid_greedy" not in loader_src
    assert "gpu_lane_lock" not in loader_src
    assert "NativeQwen38Backend" not in loader_src


def test_validate_resident_reply_rejects_none_without_inventing_text() -> None:
    with pytest.raises(ResidentRefused):
        validate_resident_reply(None, session_role="child_a")


def test_worker_session_roles_are_exactly_continuity_homes() -> None:
    assert WORKER_SESSION_ROLES == frozenset({"child_a", "child_b"})
    assert "parent" not in WORKER_SESSION_ROLES
    assert "protected_test" not in WORKER_SESSION_ROLES
