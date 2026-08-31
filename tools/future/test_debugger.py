"""Tests for the provider-neutral debugger lab.

The load-bearing guard is the negative control: every operation on an
unavailable provider must RAISE with the missing dependency named, and no
code path may return a fabricated stack, variable, or transcript. A guard
nobody has watched fail is not a guard.

Sparse-checkout trap: never assert a tool is ABSENT as a property of the
code. Assert the module COPES with either state and records which path it
took. The portable negative control is StubProvider(available=False).
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from tools.future import debugger as dbg
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def test_build_emits_sealed_receipt():
    out = dbg.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DEBUGGER_LAB.json"
    assert doc["schema"] == "hawking.future.debugger.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["claim_class"] == "STATIC_ONLY"
    assert "DIAGNOSTIC_RELATIVE" in doc["does_not_produce"]
    assert "PROTECTED_ABSOLUTE" in doc["does_not_produce"]
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert dbg.ERAS[-1].startswith("V ")
    assert "VI" not in "".join(dbg.ERAS)
    assert len(dbg.ODYSSEYS) == 3
    assert doc["operations"] == list(dbg.OPERATIONS)
    assert doc["n_operations"] == len(dbg.OPERATIONS)
    assert doc["lab_capabilities"] == list(dbg.LAB_CAPABILITIES)
    assert doc["apple_lab"]["gpu_authority"] is False
    assert doc["apple_lab"]["evidence_class"] == "STATIC_ONLY"
    rc = doc["resident_callable"]
    assert rc["can_hcli_invoke"] is True
    assert rc["entry_point"].endswith("tools/future/debugger.py --probe")
    assert rc["receipt"] == "receipts/future/DEBUGGER_LAB.json"
    assert rc["workunit_emitted"]
    assert rc["frontier"]["feeds"] == "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
    assert rc["fail_closed"]["unavailable_provider"]


def test_selftest_emits_sealed_receipt():
    out = dbg.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["schema"] == dbg.SCHEMA
    assert doc["bench"]["state"] == "UNKNOWN"


def test_nineteen_operations_are_the_contract():
    expected = (
        "launch",
        "attach",
        "set_breakpoint",
        "conditional_breakpoint",
        "tracepoint",
        "pause",
        "resume",
        "step_in",
        "step_over",
        "step_out",
        "run_to_location",
        "stacks",
        "threads",
        "variables",
        "memory_inspect",
        "evaluate",
        "crash_capture",
        "sanitizer_result",
        "transcript_receipt",
    )
    assert dbg.OPERATIONS == expected
    for provider in dbg.providers().values():
        for op in dbg.OPERATIONS:
            assert callable(getattr(provider, op)), (provider.name, op)


def test_apple_lab_probe_covers_named_capabilities_and_ran():
    lab = dbg.probe_apple_lab()
    assert set(lab["probes"]) == set(dbg.LAB_CAPABILITIES)
    assert "did_not" in lab
    assert "install any package" in lab["did_not"]
    assert "run xcode-select to change the active developer directory" in lab["did_not"]
    assert lab["gpu_authority"] is False
    assert lab["metal_compiler_finding"] in {"ABSENT", "PRESENT"}
    metal = lab["probes"]["metal_compilation"]
    assert isinstance(metal["available"], bool)
    locator = metal["locators"]["metal"]
    # The probe actually ran xcrun. Absence or presence is a host fact.
    assert locator["xcrun_returncode"] is not None
    if not metal["available"]:
        blob = " ".join(metal["missing"]).lower()
        assert "metal" in blob
        assert lab["metal_compiler_finding"] == "ABSENT"
    else:
        assert lab["metal_compiler_finding"] == "PRESENT"
        assert metal["locators"]["metal"]["xcrun_path"]
    lldb = lab["probes"]["lldb"]
    assert isinstance(lldb["available"], bool)
    assert "launch_probe" in lldb
    assert lldb["launch_probe"]["argv"][0] == "lldb"
    for name in dbg.LAB_CAPABILITIES:
        row = lab["probes"][name]
        assert "available" in row
        assert "missing" in row
        assert "present" in row


def test_both_providers_satisfy_the_same_interface():
    lldb = dbg.LldbProvider()
    stub = dbg.StubProvider(available=False)
    for op in dbg.OPERATIONS:
        assert hasattr(lldb, op) and hasattr(stub, op)
        assert callable(getattr(lldb, op)) and callable(getattr(stub, op))


def _assert_raises_unavailable(provider: dbg.DebuggerProvider, operation: str, **kwargs: Any):
    with pytest.raises(dbg.DebuggerUnavailableError) as caught:
        dbg.invoke(provider, operation, **kwargs)
    err = caught.value
    assert err.provider == provider.name
    assert err.operation == operation
    assert err.missing, operation
    text = str(err)
    assert provider.name in text
    assert operation in text
    # The missing dependency is named in the error, not swallowed.
    assert any(str(item) for item in err.missing)
    return err


def test_unavailable_stub_every_operation_raises_with_named_dependency():
    """Negative control: the refusal must actually fire."""
    stub = dbg.StubProvider(available=False)
    assert stub.available() is False
    fired = []
    for name, thunk in dbg.debugger_execution_entry_points(stub):
        with pytest.raises(dbg.DebuggerUnavailableError) as caught:
            result = thunk()
            raise AssertionError(f"{name} returned {result!r} instead of raising")
        err = caught.value
        assert err.provider == "stub", name
        assert err.operation == name, name
        assert err.missing, name
        blob = " ".join(err.missing).lower() + str(err).lower()
        assert "stub" in blob or "unavailable" in blob or "cannot" in blob
        fired.append(name)
    assert fired == list(dbg.OPERATIONS)
    # Named calls with explicit arguments, so a refactor of the thunk table
    # cannot silently drop the guard.
    _assert_raises_unavailable(stub, "launch", program="/bin/echo")
    _assert_raises_unavailable(stub, "attach", pid=2)
    _assert_raises_unavailable(stub, "set_breakpoint", location="main")
    _assert_raises_unavailable(stub, "conditional_breakpoint", location="main", condition="x==1")
    _assert_raises_unavailable(stub, "tracepoint", location="main")
    _assert_raises_unavailable(stub, "pause")
    _assert_raises_unavailable(stub, "resume")
    _assert_raises_unavailable(stub, "step_in")
    _assert_raises_unavailable(stub, "step_over")
    _assert_raises_unavailable(stub, "step_out")
    _assert_raises_unavailable(stub, "run_to_location", location="main")
    _assert_raises_unavailable(stub, "stacks")
    _assert_raises_unavailable(stub, "threads")
    _assert_raises_unavailable(stub, "variables")
    _assert_raises_unavailable(stub, "memory_inspect", address="0x1")
    _assert_raises_unavailable(stub, "evaluate", expression="1")
    _assert_raises_unavailable(stub, "crash_capture")
    _assert_raises_unavailable(stub, "sanitizer_result")
    _assert_raises_unavailable(stub, "transcript_receipt")


def test_host_lldb_provider_copes_with_either_state():
    """Do not encode this checkout. If LLDB cannot operate, ops raise; if it can, they still refuse fabrication."""
    provider = dbg.LldbProvider()
    probe = provider.probe()
    assert isinstance(provider.available(), bool)
    assert "missing" in probe
    if not provider.available():
        fired = []
        for op in dbg.OPERATIONS:
            err = _assert_raises_unavailable(provider, op)
            blob = " ".join(err.missing).lower()
            assert blob, op
            fired.append(op)
        assert fired == list(dbg.OPERATIONS)
    else:
        with pytest.raises(dbg.AttachRefused):
            provider.launch(program=None)
        with pytest.raises(dbg.AttachRefused):
            provider.attach(pid=None)
        with pytest.raises(dbg.MemoryWriteForbidden):
            provider.memory_inspect(address="0x1", write=True)


def test_no_fabricated_stack_variable_or_transcript_from_unavailable_provider():
    stub = dbg.StubProvider(available=False)
    lldb = dbg.LldbProvider()
    targets = [stub]
    if not lldb.available():
        targets.append(lldb)
    for provider in targets:
        for op in ("stacks", "threads", "variables", "memory_inspect", "crash_capture", "sanitizer_result", "transcript_receipt"):
            with pytest.raises(dbg.DebuggerUnavailableError) as caught:
                result = dbg.invoke(provider, op)
                raise AssertionError(f"{provider.name}.{op} returned {result!r} instead of raising")
            result_like = getattr(caught.value, "probe", {})
            assert result_like.get("fabricated") is not True
            # The exception is not a stand-in stack.
            assert "frames" not in (caught.value.probe or {})
            assert not hasattr(caught.value, "frames")
            assert not hasattr(caught.value, "variables")


def test_enabled_stub_still_refuses_to_invent_debuggee_state():
    stub = dbg.StubProvider(available=True)
    assert stub.available() is True
    for op in (
        "launch",
        "attach",
        "stacks",
        "variables",
        "memory_inspect",
        "crash_capture",
        "sanitizer_result",
    ):
        with pytest.raises(dbg.DebuggerUnavailableError) as caught:
            dbg.invoke(
                stub,
                op,
                **(
                    {"program": "/bin/echo"}
                    if op == "launch"
                    else {"pid": 2, "command": "/bin/echo debugger-lab"}
                    if op == "attach"
                    else {}
                ),
            )
        blob = " ".join(caught.value.missing).lower()
        assert "invent" in blob or "fabricat" in blob or "stub" in blob


def test_attach_safety_predicates():
    with pytest.raises(dbg.AttachRefused, match="explicit"):
        dbg.guard_attach(None, explicit=False)
    with pytest.raises(dbg.AttachRefused, match="explicit"):
        dbg.guard_attach(None, explicit=True)
    with pytest.raises(dbg.AttachRefused, match="kernel/launchd"):
        dbg.guard_attach(1)
    with pytest.raises(dbg.AttachRefused, match="itself"):
        dbg.guard_attach(os.getpid())
    with pytest.raises(dbg.AttachRefused, match="Codex"):
        dbg.guard_attach(4242, command="python3 tools/odyssey_ctl.py cycle --go")
    with pytest.raises(dbg.AttachRefused, match="Codex"):
        dbg.guard_attach(99, command="/usr/bin/python tools/odyssey/modellake_watch.py --poll-secs 0.10")
    allowed = dbg.guard_attach(12345, command="/bin/echo debugger-lab")
    assert allowed == 12345
    assert dbg.command_looks_like_codex("python3 tools/odyssey_ctl.py cycle")
    assert not dbg.command_looks_like_codex("/bin/echo hello")


def test_memory_inspection_is_read_only():
    with pytest.raises(dbg.MemoryWriteForbidden):
        dbg.guard_memory(write=True)
    with pytest.raises(dbg.MemoryWriteForbidden):
        dbg.guard_memory(command="memory write 0x1000 1")
    with pytest.raises(dbg.MemoryWriteForbidden):
        dbg.guard_memory(command="register write sp 0")
    dbg.guard_memory(write=False, command="memory read 0x1000")
    stub = dbg.StubProvider(available=False)
    # Unavailable still raises unavailable first, not a successful write.
    with pytest.raises(dbg.DebuggerUnavailableError):
        stub.memory_inspect(address="0x1", write=True)


def test_lab_execution_raises_when_capability_unavailable():
    lab = dbg.probe_apple_lab()
    mapping = {
        "xcodebuild": dbg.xcodebuild_build,
        "metal_compilation": dbg.metal_compile,
        "shader_diagnostics": dbg.shader_diagnostics,
        "coreml_compilation": dbg.coreml_compile,
        "mlcomputeplan": dbg.mlcomputeplan_live,
        "mlstate": dbg.mlstate_live,
        "instruments": dbg.instruments_record,
        "simulator": dbg.simulator_boot,
    }
    watched = []
    for name, fn in mapping.items():
        row = lab["probes"][name]
        if row["available"]:
            continue
        with pytest.raises(dbg.LabUnavailableError) as caught:
            result = fn()
            raise AssertionError(f"{name} returned {result!r} instead of raising")
        err = caught.value
        assert err.capability == name
        assert err.missing
        blob = (str(err) + " " + " ".join(err.missing)).lower()
        assert name.split("_")[0] in blob or "unavailable" in blob or "absent" in blob or "require" in blob
        watched.append(name)
    # Metal is the finding this lane was asked to watch fail.
    if not lab["probes"]["metal_compilation"]["available"]:
        assert "metal_compilation" in watched
        with pytest.raises(dbg.LabUnavailableError) as caught:
            dbg.metal_compile(source="kernel void k() {}")
        assert "metal" in str(caught.value).lower() or "metal" in " ".join(caught.value.missing).lower()


def test_sanitizer_capability_is_recorded_not_invented():
    row = dbg.probe_apple_lab()["probes"]["sanitizers"]
    assert isinstance(row["available"], bool)
    compiled = row.get("compiled") or {}
    if row["available"]:
        assert any(v.get("ok") for v in compiled.values())
        result = dbg.sanitizer_compile("address") if compiled.get("address", {}).get("ok") else None
        if result is not None:
            assert result["fabricated"] is False
            assert result["ok"] is True
    else:
        with pytest.raises(dbg.LabUnavailableError):
            dbg.sanitizer_compile("address")
    # sanitizer_result is a debugger op, not a compile probe, and still
    # refuses to invent a debuggee report.
    with pytest.raises(dbg.DebuggerUnavailableError):
        dbg.StubProvider(available=False).sanitizer_result()


def test_transcript_format_is_honest_and_not_a_fake_debuggee():
    session = {
        "session_id": "stub-session-001",
        "provider": "stub",
        "program": None,
        "pid": None,
        "live": False,
        "commands": [
            {
                "command": "xcrun -f metal",
                "ok": False,
                "output": 'xcrun: error: unable to find utility "metal"',
            }
        ],
    }
    doc = dbg.format_transcript(session)
    assert doc["schema"] == dbg.TRANSCRIPT_SCHEMA
    assert doc["fabricated"] is False
    assert doc["observed"] is True
    assert doc["live"] is False
    assert doc["n_commands"] == 1
    assert doc["gpu_authority"] is False
    assert doc["bench_state"] == "UNKNOWN"
    dbg.refuse_fabricated("transcript", doc)
    with pytest.raises(dbg.FabricationForbidden):
        dbg.refuse_fabricated("stack", {"fabricated": True, "frames": [{"fn": "main"}]})


def test_workunits_sleep_when_lab_is_closed_and_round_trip_hcli():
    lab = dbg.probe_apple_lab()
    units = dbg.emit_workunits(lab)
    ids = [row["id"] for row in units]
    assert ids == sorted(ids)
    assert "future.debugger.probe-apple-lab" in ids
    probe = next(row for row in units if row["id"] == "future.debugger.probe-apple-lab")
    assert probe["status"] == "pending"
    assert probe["classification"] == "STATIC_ONLY"
    from hcli.workunit import WorkUnit

    for row in units:
        WorkUnit.from_dict(row)
        assert row["verifier"]
        assert row["claim_boundary"]
        assert row["provider"] == "future.debugger"
        if row["status"] == "SLEEPING":
            assert row["classification"] == "SLEEPING"
            assert row.get("blocked_reason")
            assert "synthetic" in row["description"].lower() or "SLEEPING" in row["description"]
    closed = [name for name, row in lab["probes"].items() if not row["available"]]
    sleeping_ids = {row["id"] for row in units if row["status"] == "SLEEPING"}
    for name in closed:
        assert f"future.debugger.sleep.{name}" in sleeping_ids
    # Derive from data: number of sleeping units equals unavailable capabilities.
    assert len(sleeping_ids) == len(closed)


def test_hcli_invoke_probe_and_unknown_action():
    out = dbg.hcli_invoke({"action": "probe"})
    assert "probes" in out
    assert set(out["probes"]) == set(dbg.LAB_CAPABILITIES)
    with pytest.raises(ValueError, match="unknown"):
        dbg.hcli_invoke({"action": "steal-gpu"})
    with pytest.raises(dbg.DebuggerUnavailableError):
        dbg.hcli_invoke({"action": "invoke", "provider": "stub", "operation": "stacks"})


def test_recovered_implementation_accounts_for_named_seams():
    recovered = {row["path"]: row for row in dbg.recovered_implementation()}
    for path in (
        "hcli/tool_registry.py",
        "hcli/providers.py",
        "hcli/vmcp_adapter.py",
        "tools/future/ane_preboard.py",
        "receipts/future/ANE_PREBOARD.json",
        "CODEX_ACCELERATOR_HANDOFF.json",
    ):
        assert path in recovered
        assert recovered[path]["source"] in {"ABSENT", "ON_DISK", "GIT_HEAD"}
    concepts = dbg.recovered_concepts()
    assert concepts["XDebugger"]["status"] == "ABSENT"
    findings = "\n".join(dbg.negative_findings())
    if recovered["CODEX_ACCELERATOR_HANDOFF.json"]["source"] == "ABSENT":
        assert "CODEX_ACCELERATOR_HANDOFF.json" in findings
    assert "XDebugger" in findings


def test_receipt_has_no_hardware_numbers():
    doc = json.loads(dbg.build().read_text())

    def walk(node: Any, path: str = "") -> list[str]:
        hits: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                    hits.append(f"{here}={value!r}")
                hits.extend(walk(value, here))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                hits.extend(walk(value, f"{path}[{i}]"))
        return hits

    assert walk(doc) == []
    assert doc["bench"]["gpu_authority"] is False
    assert doc["metal_compiler_finding"] in {"ABSENT", "PRESENT"}


def test_eras_are_five_odysseys_are_three():
    assert len(dbg.ERAS) == 5
    assert len(dbg.ODYSSEYS) == 3
    assert "IV" not in "".join(dbg.ODYSSEYS)


def test_launch_requires_explicit_program_even_in_the_guard():
    with pytest.raises(dbg.AttachRefused, match="explicit program"):
        dbg.guard_launch(None)
    with pytest.raises(dbg.AttachRefused, match="explicit program"):
        dbg.guard_launch("  ")
    assert dbg.guard_launch("/bin/echo") == "/bin/echo"
