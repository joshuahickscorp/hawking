from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hawking.providers import GenerationRequest, GenerationResponse
from hawking.remote_cognition import (
    OpenRouterProvider,
    _next_bounded_mutation_tool,
    _tool_trace_completion,
    _keychain_resolver_from_environment,
    _normalized_research_result,
    _mutation_proposal_serialization_instruction,
    _mutation_proposal_from_text,
    _response_tool_calls,
    _structured_result_response_format,
    _workunit_tool_name_map,
)
from hawking.workunit_owner import (
    ProviderSemanticResponseError,
    RESEARCH_MAX_CONTINUATION_TURNS,
    WorkunitOwner,
    WorkunitRecord,
    _validate_owner_chat_payload,
    _clear_recovered_provider_classification,
    _discrete_goal_prompt,
    _research_worker_prompt,
    _write_research_worker_partial_packet,
    _write_research_worker_result,
    research_continuation_exhausted,
    resolve_live_worker,
)
from hawking.goal_surface import reclassify_blocked_child_workunit
from hawking.persist import atomic_write_json


class _Registry:
    def __init__(self):
        self.calls = []

    def invoke(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "filesystem.write":
            value = {"path": arguments.get("path"), "changed": True}
        elif name == "tests.run":
            value = {"verified": True, "returncode": 0}
        else:
            value = {"stdout": "bounded evidence"}
        return SimpleNamespace(ok=True, value=value, error=None)


def _response(text, calls=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return GenerationResponse(
        text=text,
        raw={"choices": [{"message": message, "finish_reason": "stop"}]},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _call(name, arguments=None):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {
            "name": _workunit_tool_name_map([name])[name],
            "arguments": json.dumps(arguments or {}),
        },
    }


def test_response_tool_calls_normalizes_content_serialized_tool_envelope():
    response = _response(json.dumps({"tool_calls": [{
        "name": "fs.read", "arguments": {"path": "hawking/mutation.py"},
    }]}))
    calls = _response_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "fs.read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "hawking/mutation.py"}


def test_response_tool_calls_normalizes_bare_arguments_only_for_forced_read():
    response = _response('{"path":"hawking/mutation.py","start_line":1}')
    assert _response_tool_calls(response) == []
    calls = _response_tool_calls(response, fallback_read_tool="fs.read")
    assert calls[0]["function"]["name"] == "fs.read"
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "hawking/mutation.py"


def test_paired_mutation_synthesis_repeats_the_required_test_operation():
    instruction = _mutation_proposal_serialization_instruction({
        "require_regression_test_mutation": True,
        "required_regression_test_paths": ["hawking/tests/test_canary.py"],
    })
    assert "exactly two" in instruction
    assert "hawking/tests/test_canary.py" in instruction
    assert "source-only" in instruction


def test_mutation_synthesis_prefers_anchor_intent_without_old_text():
    instruction = _mutation_proposal_serialization_instruction({})
    assert "s:MUTATE" in instruction
    assert "anchor" in instruction
    assert "Do not return old_text" in instruction
    assert "ONLY valid mutation shape" in instruction
    assert "MISSING_SOURCE_ANCHOR" in instruction


def test_compact_mutation_intent_is_recognized_as_provider_proposal():
    proposal = _mutation_proposal_from_text(
        '{"s":"MUTATE","anchor":"SRC-A91F","op":"replace","body":"new\\n","tests":["T-1"]}'
    )
    assert proposal["s"] == "MUTATE"
    assert proposal["anchor"] == "SRC-A91F"
    assert "old_text" not in proposal
    assert _mutation_proposal_from_text(
        '{"s":"MUTATE","path":"owner.py","body":"new\\n","old_text":"old\\n"}'
    )["s"] == "MUTATE"


def test_owner_rejects_http_200_empty_length_before_completion_acceptance():
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": None},
            "finish_reason": "length",
        }],
    }
    try:
        _validate_owner_chat_payload(payload, completion_contract={"require_structured_result": True})
    except ProviderSemanticResponseError as exc:
        assert exc.reason == "empty_assistant_message"
        assert exc.finish_reason == "length"
    else:  # pragma: no cover - makes the fail-closed assertion explicit
        raise AssertionError("HTTP 200 empty provider output was accepted")


def test_owner_rejects_http_200_tool_finish_without_callable_payload():
    payload = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{}],
            },
            "finish_reason": "tool_calls",
        }],
    }
    try:
        _validate_owner_chat_payload(payload)
    except ProviderSemanticResponseError as exc:
        assert exc.reason == "empty_assistant_message"
        assert exc.finish_reason == "tool_calls"
    else:  # pragma: no cover - makes the fail-closed assertion explicit
        raise AssertionError("malformed HTTP 200 tool output was accepted")


def _provider(responses):
    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.model_id = "deepseek/deepseek-v4.1-flash"
    queue = list(responses)

    def generate(_request, *, timeout=None):
        del timeout
        return queue.pop(0)

    provider.generate = generate
    return provider, queue


def test_completion_contract_continues_after_prose_and_seals_real_evidence():
    names = ["filesystem.write", "tests.run", "git.status", "git.diff"]
    responses = [
        _response("I inspected the repository and the task is complete."),
        _response("", [_call("filesystem.write", {"path": "proof.py"})]),
        _response("", [_call("tests.run", {"paths": ["."]})]),
        _response("", [_call("git.status")]),
        _response("", [_call("git.diff")]),
        _response(json.dumps({"observations": [], "operations": [], "tests": [], "questions": [], "status": "complete", "content": "evidence"})),
    ]
    provider, queue = _provider(responses)
    registry = _Registry()
    request = GenerationRequest(
        messages=[{"role": "user", "content": "do the bounded unit"}],
        model=provider.model_id,
        request_id="contract-test",
    )
    result, trace = provider._generate_with_workunit_tools(
        request,
        {
            "tool_registry": registry,
            "allowed_tool_names": names,
            "max_tool_calls": 8,
            "completion_contract": {
                "required_successful_tools": names,
                "require_final_tests_pass": True,
                "require_structured_result": True,
                "max_continuation_turns": 2,
            },
        },
        tuple({"type": "function", "function": {"name": _workunit_tool_name_map([name])[name], "parameters": {}}} for name in names),
        names,
    )
    completion = result.raw["hawking_completion"]
    assert completion["complete"] is True
    assert completion["unmet"] == []
    assert completion["continuation_turns"] == 1
    assert not queue
    assert [row["tool"] for row in trace if row.get("dispatched")] == names


def test_completion_contract_returns_incomplete_when_bounded_reminders_exhaust():
    provider, queue = _provider([_response("still thinking") for _ in range(4)])
    request = GenerationRequest(
        messages=[{"role": "user", "content": "do the bounded unit"}],
        model=provider.model_id,
        request_id="incomplete-contract-test",
    )
    result, _trace = provider._generate_with_workunit_tools(
        request,
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["tests.run"],
            "max_tool_calls": 2,
            "completion_contract": {
                "required_successful_tools": ["tests.run"],
                "max_continuation_turns": 2,
            },
        },
        ({"type": "function", "function": {"name": "tests_run", "parameters": {}}},),
        ["tests.run"],
    )
    completion = result.raw["hawking_completion"]
    assert completion["complete"] is False
    assert "tests.run" in completion["unmet"]
    assert completion["continuation_turns"] == 2
    assert len(queue) == 1


def test_mutation_proposal_closes_after_hawking_verification_without_followup_decode():
    proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "schema": "HAWKING_PATCH_V1",
        "operations": [{"op": "create", "path": "proof.py", "new_lines": ["VALUE = 1"]}],
        "required_tests": ["test_proof.py"],
    })
    provider, queue = _provider([_response(proposal), _response("prose follow-up that must not be needed")])
    executions = []

    def execute(value):
        executions.append(value)
        return {
            "status": "accepted",
            "applied": True,
            "post_mutation_trace": [
                {"tool": "repo.edit", "ok": True, "result": {"value": {
                    "status": "accepted", "applied": True, "rolled_back": False,
                }}},
                {"tool": "tests.run", "ok": True, "result": {"value": {
                    "verified": True, "returncode": 0,
                }}},
                {"tool": "git.status", "ok": True, "result": {"value": {}}},
                {"tool": "git.diff", "ok": True, "result": {"value": {}}},
            ],
        }

    result, trace = provider._generate_with_workunit_tools(
        GenerationRequest(
            messages=[{"role": "user", "content": "return the bounded patch"}],
            model=provider.model_id,
            request_id="proposal-close-test",
        ),
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["fs.read"],
            "max_tool_calls": 1,
            "mutation_proposal_executor": execute,
            "completion_contract": {
                "mutation_proposal_mode": True,
                "required_successful_tools": ["repo.edit", "tests.run", "git.status", "git.diff"],
                "require_final_tests_pass": True,
                "max_continuation_turns": 4,
            },
        },
        ({"type": "function", "function": {"name": "fs_read", "parameters": {}}},),
        ["fs.read"],
    )
    assert len(executions) == 1
    assert len(queue) == 1
    assert result.raw["hawking_completion"]["complete"] is True
    assert result.raw["hawking_completion"]["reason"] == "evidence_contract_satisfied"
    assert {row["tool"] for row in trace if row.get("ok") is True} >= {
        "repo.edit", "tests.run", "git.status", "git.diff",
    }
    assert result.raw["hawking_proposal_bridge"] == [{
        "turn": 0,
        "typed_proposal": True,
        "response_chars": len(proposal),
        "has_tool_calls": False,
        "executor_available": True,
        "executor_invoked": True,
        "execution_status": "accepted",
    }]


def test_mutation_contract_closes_tool_menu_after_one_forced_orientation_read():
    proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{"op": "create", "path": "proof.py", "new_lines": ["VALUE = 1"]}],
    })
    provider, queue = _provider([
        _response("", [_call("fs.read", {"path": "proof.py"})]),
        _response(proposal),
    ])
    seen = []

    def generate(request, *, timeout=None):
        del timeout
        seen.append(request.to_dict())
        return queue.pop(0)

    provider.generate = generate

    def execute(_value):
        return {
            "status": "accepted",
            "applied": True,
            "post_mutation_trace": [
                {"tool": "repo.edit", "ok": True, "result": {"value": {"status": "accepted", "applied": True}}},
                {"tool": "tests.run", "ok": True, "result": {"value": {"verified": True, "returncode": 0}}},
                {"tool": "git.status", "ok": True, "result": {"value": {}}},
                {"tool": "git.diff", "ok": True, "result": {"value": {}}},
            ],
        }

    result, trace = provider._generate_with_workunit_tools(
        GenerationRequest(
            messages=[{"role": "user", "content": "read the focus then patch it"}],
            model=provider.model_id,
            request_id="one-read-proposal-test",
        ),
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["fs.read", "fs.search"],
            "max_tool_calls": 2,
            "mutation_proposal_executor": execute,
            "require_initial_read": True,
            "completion_contract": {
                "mutation_proposal_mode": True,
                "required_successful_tools": ["repo.edit", "tests.run", "git.status", "git.diff"],
                "require_final_tests_pass": True,
                "read_only_tool_call_limit": 1,
                "max_continuation_turns": 1,
            },
        },
        tuple(
            {"type": "function", "function": {"name": _workunit_tool_name_map([name])[name], "parameters": {}}}
            for name in ("fs.read", "fs.search")
        ),
        ["fs.read", "fs.search"],
    )

    assert [item["function"]["name"] for item in seen[0]["tools"]] == ["fs_read"]
    assert seen[1]["tools"] == []
    assert "tool_choice" not in seen[1]["transport_options"]
    assert seen[1]["transport_options"]["response_format"] == {"type": "json_object"}
    synthesis = seen[1]["messages"][-1]["content"]
    assert "exact old_text" in synthesis
    assert "complete new_text" in synthesis
    assert "source_edit" in synthesis
    assert result.raw["hawking_completion"]["complete"] is True
    assert [row["tool"] for row in trace if row.get("dispatched")] == ["fs.read", "mutation.proposal"]
    assert {row["tool"] for row in trace if row.get("ok") is True} >= {
        "repo.edit", "tests.run", "git.status", "git.diff",
    }


def test_mutation_contract_executes_read_only_proposal_before_tool_free_patch():
    read_proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{"kind": "fs.search", "path": "proof.py", "pattern": "VALUE"}],
    })
    patch_proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{"op": "create", "path": "proof.py", "new_lines": ["VALUE = 1"]}],
    })
    provider, queue = _provider([
        _response("", [_call("fs.read", {"path": "proof.py"})]),
        _response(read_proposal),
        _response(patch_proposal),
    ])
    seen = []

    def generate(request, *, timeout=None):
        del timeout
        seen.append(request.to_dict())
        return queue.pop(0)

    provider.generate = generate

    def execute(_value):
        return {
            "status": "accepted",
            "applied": True,
            "post_mutation_trace": [
                {"tool": "repo.edit", "ok": True, "result": {"value": {"status": "accepted", "applied": True}}},
                {"tool": "tests.run", "ok": True, "result": {"value": {"verified": True, "returncode": 0}}},
                {"tool": "git.status", "ok": True, "result": {"value": {}}},
                {"tool": "git.diff", "ok": True, "result": {"value": {}}},
            ],
        }

    result, trace = provider._generate_with_workunit_tools(
        GenerationRequest(
            messages=[{"role": "user", "content": "read focus, inspect test, then patch"}],
            model=provider.model_id,
            request_id="proposal-read-recovery-test",
        ),
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["fs.read", "fs.search"],
            "max_tool_calls": 2,
            "mutation_proposal_executor": execute,
            "require_initial_read": True,
            "completion_contract": {
                "mutation_proposal_mode": True,
                "required_successful_tools": ["repo.edit", "tests.run", "git.status", "git.diff"],
                "require_final_tests_pass": True,
                "read_only_tool_call_limit": 2,
                "max_continuation_turns": 1,
            },
        },
        tuple(
            {"type": "function", "function": {"name": _workunit_tool_name_map([name])[name], "parameters": {}}}
            for name in ("fs.read", "fs.search")
        ),
        ["fs.read", "fs.search"],
    )

    assert [row["tool"] for row in trace if row.get("dispatched")] == [
        "fs.read", "fs.search", "mutation.proposal",
    ]
    assert seen[2]["tools"] == []
    assert result.raw["hawking_completion"]["complete"] is True


def test_mutation_contract_does_not_disable_required_glm_reasoning():
    proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{"op": "create", "path": "proof.py", "new_lines": ["VALUE = 1"]}],
    })
    provider, queue = _provider([_response(proposal)])
    provider.model_id = "z-ai/glm-5.3"
    seen = []

    def generate(request, *, timeout=None):
        del timeout
        seen.append(request.to_dict())
        return queue.pop(0)

    provider.generate = generate
    result, _trace = provider._generate_with_workunit_tools(
        GenerationRequest(
            messages=[{"role": "user", "content": "return one patch"}],
            model=provider.model_id,
            request_id="glm-required-reasoning-test",
        ),
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["fs.read"],
            "max_tool_calls": 1,
            "mutation_proposal_executor": lambda _value: {
                "status": "accepted", "applied": True,
                "post_mutation_trace": [{"tool": "repo.edit", "ok": True, "result": {"value": {"status": "accepted", "applied": True}}}],
            },
            "completion_contract": {
                "mutation_proposal_mode": True,
                "required_successful_tools": ["repo.edit"],
                "max_continuation_turns": 0,
            },
        },
        ({"type": "function", "function": {"name": "fs_read", "parameters": {}}},),
        ["fs.read"],
    )

    assert "reasoning" not in seen[0]
    assert result.raw["hawking_completion"]["complete"] is True


def test_applied_legacy_execution_packet_projects_repo_edit_receipt():
    proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{"op": "create", "path": "proof.py", "new_lines": ["VALUE = 1"]}],
    })
    provider, queue = _provider([_response(proposal), _response("must not be requested")])

    def execute(_value):
        # Compatibility executor: the engine proved that the write landed,
        # but the adapter omitted its canonical accepted/unproven label and
        # post trace. The owner must still see a machine-owned repo.edit edge;
        # focused tests remain a separate obligation when required.
        return {"status": "legacy_applied", "applied": True, "rolled_back": False}

    result, trace = provider._generate_with_workunit_tools(
        GenerationRequest(
            messages=[{"role": "user", "content": "return the bounded patch"}],
            model=provider.model_id,
            request_id="legacy-applied-projection-test",
        ),
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["fs.read"],
            "max_tool_calls": 1,
            "mutation_proposal_executor": execute,
            "completion_contract": {
                "mutation_proposal_mode": True,
                "required_successful_tools": ["repo.edit"],
                "max_continuation_turns": 0,
            },
        },
        ({"type": "function", "function": {"name": "fs_read", "parameters": {}}},),
        ["fs.read"],
    )
    assert len(queue) == 1
    assert result.raw["hawking_completion"]["complete"] is True
    assert any(row.get("tool") == "repo.edit" and row.get("ok") is True for row in trace)


def test_research_contract_forces_named_first_read_tool():
    provider, queue = _provider([
        _response("", [_call("fs.read", {"path": "hawking/goal_surface.py"})]),
        _response(json.dumps({
            "observations": [], "sources": [], "recommendations": [],
            "unresolved_questions": [], "child_worker_ids": [], "status": "complete",
        })),
    ])
    seen = []

    def generate(request, *, timeout=None):
        del timeout
        seen.append(request.to_dict())
        return queue.pop(0)

    provider.generate = generate
    result, trace = provider._generate_with_workunit_tools(
        GenerationRequest(
            messages=[{"role": "user", "content": "inspect the focus file"}],
            model=provider.model_id,
            request_id="first-read-contract-test",
        ),
        {
            "tool_registry": _Registry(),
            "allowed_tool_names": ["fs.read", "fs.search"],
            "max_tool_calls": 1,
            "completion_contract": {
                "required_any_successful_tools": ["fs.read"],
                "required_first_tool": "fs.read",
                "require_structured_result": True,
                "read_only_tool_call_limit": 1,
            },
        },
        tuple(
            {"type": "function", "function": {"name": _workunit_tool_name_map([name])[name], "parameters": {}}}
            for name in ("fs.read", "fs.search")
        ),
        ["fs.read", "fs.search"],
    )
    assert seen[0]["transport_options"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "fs_read"},
    }
    assert seen[0]["transport_options"]["response_format"] == {
        "type": "json_object",
    }
    assert [item["function"]["name"] for item in seen[0]["tools"]] == ["fs_read"]
    assert seen[1]["tools"] == []
    assert seen[1]["transport_options"]["response_format"] == {
        "type": "json_object",
    }
    assert [row["tool"] for row in trace if row.get("dispatched")] == ["fs.read"]
    assert result.raw["hawking_completion"]["complete"] is True


def test_keychain_defaults_to_openrouter_current_user_without_exposing_value(monkeypatch):
    monkeypatch.delenv("HAWKING_OPENROUTER_KEYCHAIN_SERVICE", raising=False)
    monkeypatch.delenv("HAWKING_OPENROUTER_KEYCHAIN_ACCOUNT", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "current-user")
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="secure-value\n")

    monkeypatch.setattr("hawking.remote_cognition.subprocess.run", fake_run)
    resolver = _keychain_resolver_from_environment()
    assert resolver is not None
    assert resolver() == "secure-value"
    assert observed["argv"] == [
        "security", "find-generic-password", "-s", "OpenRouter",
        "-a", "current-user", "-w",
    ]


def test_remote_worker_resolution_uses_hawkingd_lease_without_fake_pid():
    resolved = resolve_live_worker(
        "deepseek/deepseek-v4.1-flash",
        lease_reader=lambda: {"host": "127.0.0.1", "port": 8014, "model": "native-waiting"},
        health_get=lambda _url, _timeout: 200,
    )
    assert resolved["logical_role"] == "remote_openrouter"
    assert resolved["resolved_via"] == "daemon_lease_metadata+hawking_remote_gateway"
    assert resolved["artifact"]["kind"] == "remote_openrouter"
    assert resolved["artifact"]["model_id"] == "deepseek/deepseek-v4.1-flash"
    assert resolved["lease_pid"] is None


def test_research_worker_contract_accepts_read_or_delegate_evidence():
    final = _response(json.dumps({"observations": ["bounded"], "status": "complete"}))
    completion = _tool_trace_completion(
        [{
            "tool": "hawking.worker.spawn",
            "ok": True,
            "result": {"value": {"worker_id": "hawking-worker-child"}},
        }],
        final,
        {
            "required_any_successful_tools": ["fs.read", "hawking.worker.spawn"],
            "require_structured_result": True,
        },
    )
    assert completion["complete"] is True
    assert completion["unmet"] == []


def test_research_json_mode_waits_for_real_read_evidence():
    contract = {
        "required_any_successful_tools": ["fs.read"],
        "require_structured_result": True,
    }
    assert _structured_result_response_format([], contract) is None
    trace = [{"tool": "fs.read", "ok": True, "result": {"value": {"text": "evidence"}}}]
    assert _structured_result_response_format(trace, contract) == {"type": "json_object"}
    assert _structured_result_response_format([], contract, {
        "successful_tools": ["fs.read"],
    }) == {"type": "json_object"}


def test_research_contract_requires_every_admitted_focus_path_before_terminal_packet():
    contract = {
        "required_any_successful_tools": ["fs.read"],
        "required_focus_paths": [
            "hawking/auto_orchestration.py",
            "hawking/tests/test_auto_orchestration.py",
        ],
        "require_structured_result": True,
        "structured_required_keys": ["observations", "sources", "status"],
        "structured_status_values": ["COMPLETE"],
    }
    final = _response(json.dumps({
        "observations": ["bounded"], "sources": [], "status": "COMPLETE",
    }))
    first_only = _tool_trace_completion([{
        "tool": "fs.read", "ok": True,
        "arguments": {"path": "hawking/auto_orchestration.py"},
        "result": {"value": {"text": "evidence"}},
    }], final, contract)
    assert first_only["complete"] is False
    assert "focus_paths:hawking/tests/test_auto_orchestration.py" in first_only["unmet"]
    complete = _tool_trace_completion([
        {
            "tool": "fs.read", "ok": True,
            "arguments": {"path": "/workspace/hawking/auto_orchestration.py"},
            "result": {"value": {"text": "evidence"}},
        },
        {
            "tool": "fs.read", "ok": True,
            "arguments": {"path": "hawking/tests/test_auto_orchestration.py"},
            "result": {"value": {"text": "evidence"}},
        },
    ], final, contract)
    assert complete["complete"] is True
    assert complete["unmet"] == []


def test_research_json_mode_waits_for_all_admitted_focus_paths():
    contract = {
        "required_any_successful_tools": ["fs.read"],
        "required_focus_paths": ["hawking/one.py", "hawking/two.py"],
        "require_structured_result": True,
    }
    one = [{
        "tool": "fs.read", "ok": True,
        "arguments": {"path": "hawking/one.py"},
        "result": {"value": {"text": "evidence"}},
    }]
    assert _structured_result_response_format(one, contract) is None
    both = one + [{
        "tool": "fs.read", "ok": True,
        "arguments": {"path": "hawking/two.py"},
        "result": {"value": {"text": "evidence"}},
    }]
    assert _structured_result_response_format(both, contract) == {"type": "json_object"}


def test_research_contract_rejects_tool_shaped_json_without_result_fields():
    final = _response('{"tool_calls":[{"name":"fs_read"}]}')
    completion = _tool_trace_completion(
        [{"tool": "fs.read", "ok": True, "result": {"value": {"text": "evidence"}}}],
        final,
        {
            "required_any_successful_tools": ["fs.read"],
            "require_structured_result": True,
            "structured_required_keys": ["observations", "sources", "status"],
        },
    )
    assert completion["complete"] is False
    assert completion["unmet"] == ["structured_result:observations,sources,status"]


def test_research_contract_rejects_in_progress_packet_after_real_read():
    final = _response('{"observations":[],"sources":[],"status":"IN_PROGRESS"}')
    completion = _tool_trace_completion(
        [{"tool": "fs.read", "ok": True, "result": {"value": {"text": "evidence"}}}],
        final,
        {
            "required_any_successful_tools": ["fs.read"],
            "require_structured_result": True,
            "structured_required_keys": ["observations", "sources", "status"],
            "structured_status_values": ["COMPLETE"],
        },
    )
    assert completion["complete"] is False
    assert completion["unmet"] == ["structured_result:terminal_status"]


def test_research_result_normalizes_provider_omitted_empty_fields():
    contract = {
        "require_structured_result": True,
        "structured_required_keys": [
            "observations", "sources", "recommendations",
            "unresolved_questions", "child_worker_ids", "status",
        ],
        "structured_status_values": ["COMPLETE", "STRUCTURED_RESULT_READY"],
    }
    text = json.dumps({"observations": ["bounded source finding"], "status": "complete"})
    normalized = _normalized_research_result(text, contract)
    assert normalized["status"] == "COMPLETE"
    assert normalized["observations"] == ["bounded source finding"]
    assert normalized["sources"] == []
    assert normalized["recommendations"] == []
    assert normalized["unresolved_questions"] == []
    assert normalized["child_worker_ids"] == []
    completion = _tool_trace_completion(
        [{"tool": "fs.read", "ok": True, "result": {"value": {"text": "evidence"}}}],
        _response(text),
        contract,
    )
    assert completion["complete"] is True
    assert completion["unmet"] == []


def test_research_result_normalizer_never_turns_prose_into_completion():
    contract = {
        "structured_required_keys": ["observations", "status"],
        "structured_status_values": ["COMPLETE"],
    }
    assert _normalized_research_result(
        "I inspected the repository and everything is complete.", contract,
    ) == {}


def test_research_contract_compiles_prose_after_real_read_only_evidence():
    contract = {
        "required_any_successful_tools": ["fs.read"],
        "require_structured_result": True,
        "structured_required_keys": ["observations", "sources", "status"],
        "structured_status_values": ["COMPLETE"],
        "allow_read_only_prose_result": True,
    }
    completion = _tool_trace_completion(
        [{"tool": "fs.read", "ok": True, "result": {"value": {"text": "evidence"}}}],
        _response("The canonical owner is hawking/web_launcher.py; it reuses the daemon."),
        contract,
    )
    assert completion["complete"] is True
    assert completion["unmet"] == []


def test_research_prose_without_read_evidence_remains_unusable():
    contract = {
        "required_any_successful_tools": ["fs.read"],
        "require_structured_result": True,
        "structured_required_keys": ["observations", "sources", "status"],
        "structured_status_values": ["COMPLETE"],
        "allow_read_only_prose_result": True,
    }
    completion = _tool_trace_completion([], _response("I inspected the repository."), contract)
    assert completion["complete"] is False
    assert "one_of:fs.read" in completion["unmet"]
    assert "structured_result" in completion["unmet"]


def test_research_outer_continuation_cap_preserves_one_opening_turn():
    assert research_continuation_exhausted(RESEARCH_MAX_CONTINUATION_TURNS + 1) is False
    assert research_continuation_exhausted(RESEARCH_MAX_CONTINUATION_TURNS + 2) is True


def test_research_worker_roundtrip_preserves_parent_and_child_identity():
    record = WorkunitRecord(
        workunit_id="GOAL-CHILD",
        goal_id="GOAL-CHILD",
        goal_mode="worker_research",
        worker_model="moonshotai/kimi-k3",
        worker_id="hawking-worker-child",
        parent_worker_id="hawking-worker-parent",
        parent_workunit_id="GOAL-PARENT",
        child_worker_ids=["hawking-worker-grandchild"],
        objective="inspect one bounded surface",
    )
    restored = WorkunitRecord.from_dict(record.to_dict())
    assert restored.parent_worker_id == "hawking-worker-parent"
    assert restored.parent_workunit_id == "GOAL-PARENT"
    assert restored.child_worker_ids == ["hawking-worker-grandchild"]
    prompt = _research_worker_prompt(
        {"acceptance": ["return evidence"]}, restored, 1, opening=True
    )
    assert "WORKER_MODE=read_only_research" in prompt
    assert "hawking.worker.spawn" in prompt
    assert "may not edit files" in prompt
    assert "exact JSON object" in prompt


def test_research_worker_prompt_projects_declared_focus_files_and_first_action():
    record = WorkunitRecord(
        workunit_id="GOAL-FOCUS",
        goal_id="GOAL-FOCUS",
        goal_mode="worker_research",
        worker_model="deepseek/deepseek-v4.1-flash",
        objective="Map one bounded owner.",
    )
    prompt = _research_worker_prompt(
        {
            "focus_files": ["hawking/auto_orchestration.py"],
            "focus_tests": ["hawking/tests/test_auto_orchestration.py"],
            "acceptance": ["return source-grounded evidence"],
        },
        record,
        1,
        opening=True,
    )
    assert "ADMITTED_FOCUS_FILES: hawking/auto_orchestration.py" in prompt
    assert "ADMITTED_FOCUS_TESTS: hawking/tests/test_auto_orchestration.py" in prompt
    assert "first read-only observation MUST target one ADMITTED_FOCUS_FILE" in prompt
    assert "do not substitute README.md" in prompt


def test_discrete_prompt_requires_paired_dedicated_red_green_bundle(tmp_path):
    source = tmp_path / "owner.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    record = WorkunitRecord(
        workunit_id="WU-PAIR", goal_id="GOAL-PAIR", goal_mode="discrete_goal",
        worker_model="deepseek/deepseek-v4.1-flash", objective="Change one owner.",
        staging_workspace=str(tmp_path),
    )
    prompt = _discrete_goal_prompt({
        "focus_files": ["owner.py"],
        "focus_tests": ["test_owner_regression.py"],
        "acceptance": ["change production behavior"],
        "mutation_bundle": {"require_regression_test_mutation": True},
    }, record, 1, opening=True)
    assert "PAIRED RED-GREEN BUNDLE" in prompt
    assert "source-only payload is incomplete" in prompt
    assert "test_owner_regression.py" in prompt


def test_discrete_prompt_uses_compact_anchor_intent_for_existing_test_route(tmp_path):
    source = tmp_path / "owner.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    record = WorkunitRecord(
        workunit_id="WU-COMPACT", goal_id="GOAL-COMPACT", goal_mode="discrete_goal",
        worker_model="deepseek/deepseek-v4.1-flash", objective="Change one owner.",
        staging_workspace=str(tmp_path),
    )
    prompt = _discrete_goal_prompt({
        "focus_files": ["owner.py"],
        "focus_tests": ["test_owner.py"],
        "acceptance": ["change production behavior"],
        "mutation_bundle": {"require_regression_test_mutation": False},
    }, record, 1, opening=True)
    assert "COMPACT_MUTATION_PROTOCOL=hawking.compact_worker.v1" in prompt
    assert '"s":"MUTATE"' in prompt
    assert "Do not emit old_text" in prompt
    assert "MISSING_SOURCE_ANCHOR" in prompt


def test_blocked_child_can_be_reclassified_to_read_only_without_losing_evidence(tmp_path):
    contract_path = tmp_path / ".hawking" / "goals" / "WU-READ.contract.json"
    atomic_write_json(contract_path, {
        "goal_id": "GOAL-PARENT",
        "workunit_id": "WU-READ",
        "goal_mode": "discrete_goal",
        "goal_terminal_on_workunit_complete": False,
        "mutation_bundle": {
            "require_source_mutation": True,
            "require_regression_test_mutation": True,
            "require_focused_test_invocation": True,
        },
    })
    owner = WorkunitOwner(tmp_path)
    owner.save(WorkunitRecord(
        workunit_id="WU-READ",
        goal_id="GOAL-PARENT",
        goal_mode="discrete_goal",
        state="BLOCKED",
        contract_path=str(contract_path),
        checkpoint_path=str(tmp_path / "receipts" / "future" / "workunits" / "WU-READ_CHECKPOINT.json"),
        staging_workspace=str(tmp_path),
        evidence=[{"kind": "worker_completion_contract", "unmet": ["repo.edit"]}],
    ))

    result = reclassify_blocked_child_workunit(
        tmp_path,
        "WU-READ",
        goal_mode="worker_research",
        reason="bounded reconnaissance must not inherit mutation obligations",
    )

    record = owner.load("WU-READ")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert result["changed"] is True
    assert record.goal_mode == "worker_research"
    assert record.state == "BLOCKED"
    assert any(item.get("kind") == "worker_completion_contract" for item in record.evidence)
    assert any(item.get("kind") == "workunit_mode_reclassified" for item in record.evidence)
    assert contract["goal_mode"] == "worker_research"
    assert contract["mutation_bundle"]["require_source_mutation"] is False
    assert Path(result["receipt"]).is_file()


def test_nonterminal_research_child_writes_its_own_result_not_parent_goal(tmp_path):
    contract_path = tmp_path / ".hawking" / "goals" / "WU-CHILD.contract.json"
    atomic_write_json(contract_path, {
        "goal_id": "GOAL-PARENT",
        "workunit_id": "WU-CHILD",
        "goal_mode": "worker_research",
        "goal_terminal_on_workunit_complete": False,
    })
    record = WorkunitRecord(
        workunit_id="WU-CHILD",
        goal_id="GOAL-PARENT",
        goal_mode="worker_research",
        worker_id="hawking-worker-child",
        worker_model="deepseek/deepseek-v4.1-flash",
        contract_path=str(contract_path),
        staging_workspace=str(tmp_path),
    )
    assert _write_research_worker_result(
        record,
        text='{"observations":["bounded"]}',
        payload={"hawking": {"completion": {"complete": True}}},
    ) is True
    child_result = tmp_path / "receipts" / "future" / "workunits" / "WU-CHILD_RESULT_PACKET.json"
    assert child_result.is_file()
    assert not (tmp_path / ".hawking" / "goals" / "GOAL-PARENT.result.json").exists()
    assert any(item.get("kind") == "research_worker_result" for item in record.evidence)


def test_successful_fresh_result_clears_stale_provider_failure_label():
    record = WorkunitRecord(
        workunit_id="WU-RECOVERED",
        goal_id="GOAL-PARENT",
        goal_mode="worker_research",
        classification="PROVIDER_TRANSPORT_ERROR",
        worker_id="hawking-worker-recovered",
        worker_model="moonshotai/kimi-k3",
    )

    _clear_recovered_provider_classification(record)

    assert record.classification == ""
    assert any(item.get("kind") == "provider_failure_recovered" for item in record.evidence)


def test_research_partial_packet_preserves_read_evidence_without_completion(tmp_path):
    record = WorkunitRecord(
        workunit_id="WU-PARTIAL",
        goal_id="GOAL-PARENT",
        goal_mode="worker_research",
        worker_id="hawking-worker-partial",
        worker_model="moonshotai/kimi-k3",
        staging_workspace=str(tmp_path),
    )
    contract = {
        "required_any_successful_tools": ["fs.read"],
        "structured_required_keys": [
            "observations", "sources", "recommendations",
            "unresolved_questions", "child_worker_ids", "status",
        ],
    }
    path = _write_research_worker_partial_packet(
        record,
        text=json.dumps({
            "observations": [{"claim": "native path is source-only"}],
            "sources": ["crates/hawking-core/src/lib.rs"],
            "status": "BLOCKED",
        }),
        payload={"hawking": {"completion": {
            "complete": False,
            "successful_tools": ["fs.read", "fs.search"],
        }}},
        contract=contract,
        reason="provider_blocked_after_read_evidence",
    )

    assert path is not None
    packet = json.loads(Path(path).read_text(encoding="utf-8"))
    assert packet["status"] == "PARTIAL"
    assert packet["provider_status"] == "BLOCKED"
    assert packet["successful_read_tools"] == ["fs.read", "fs.search"]
    assert packet["observations"][0]["claim"] == "native path is source-only"
    assert any(item.get("kind") == "research_worker_partial_packet" for item in record.evidence)


def test_research_partial_packet_requires_real_read_evidence(tmp_path):
    record = WorkunitRecord(
        workunit_id="WU-NO-READ",
        goal_id="GOAL-PARENT",
        goal_mode="worker_research",
        worker_id="hawking-worker-no-read",
        worker_model="deepseek/deepseek-v4.1-flash",
        staging_workspace=str(tmp_path),
    )
    assert _write_research_worker_partial_packet(
        record,
        text="Useful but ungrounded prose.",
        payload={"hawking": {"completion": {
            "complete": False,
            "successful_tools": ["campaign.state"],
        }}},
        contract={"required_any_successful_tools": ["fs.read"]},
        reason="provider_nonterminal_after_read_evidence",
    ) is None


def test_worker_cancel_persists_terminal_state_before_driver_signal(tmp_path):
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="GOAL-CANCEL",
        goal_id="GOAL-CANCEL",
        goal_mode="worker_research",
        worker_id="hawking-worker-cancel",
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        contract_path=str(tmp_path / "contract.json"),
        staging_workspace=str(tmp_path),
    )
    owner.save(record)
    cancelled = owner.cancel(record.workunit_id)
    assert cancelled["state"] == "CANCELLED"
    assert cancelled["worker_status"] == "KILLED"
    assert owner.load(record.workunit_id).state == "CANCELLED"


def test_completion_contract_requires_tests_run_when_final_tests_must_pass():
    from hawking.remote_cognition import _workunit_completion_contract

    implied = _workunit_completion_contract(
        {
            "completion_contract": {
                "required_successful_tools": ["repo.edit"],
                "require_final_tests_pass": True,
            }
        }
    )
    assert implied["required_successful_tools"] == ["repo.edit", "tests.run"]

    plain = _workunit_completion_contract(
        {"completion_contract": {"required_successful_tools": ["repo.edit"]}}
    )
    assert plain["required_successful_tools"] == ["repo.edit"]


def test_completion_contract_ignores_blank_required_tool_names():
    completion = _tool_trace_completion(
        [
            {
                "tool": "tests.run",
                "ok": True,
                "result": {"value": {"verified": True, "returncode": 0}},
            }
        ],
        _response(json.dumps({"status": "complete"})),
        {
            "required_successful_tools": ["tests.run", "", "   "],
            "require_final_tests_pass": True,
            "require_structured_result": True,
        },
    )
    assert completion["complete"] is True
    assert completion["unmet"] == []


def test_no_focused_test_contract_accepts_engine_unproven_edit_with_diff_status():
    completion = _tool_trace_completion(
        [
            {"tool": "repo.edit", "ok": True,
             "result": {"value": {"status": "unproven", "applied": True,
                                    "rolled_back": False}}},
            {"tool": "git.status", "ok": True, "result": {"value": {}}},
            {"tool": "git.diff", "ok": True, "result": {"value": {}}},
        ],
        _response(""),
        {
            "required_successful_tools": ["repo.edit", "git.status", "git.diff"],
            "verification_mode": "engine_validation_and_status_diff_no_focused_tests",
        },
    )
    assert completion["complete"] is True
    assert completion["unmet"] == []


def test_build_contract_forces_next_typed_door_after_orientation_budget():
    contract = {
        "required_successful_tools": [
            "repo.edit", "tests.run", "git.status", "git.diff",
        ],
        "read_only_tool_call_limit": 8,
    }
    context = {
        "tool_write_authority": True,
        "tool_budget": {"scope": "interactive_build"},
    }
    assert _next_bounded_mutation_tool([], contract, {}, context, calls_used=7) is None
    assert _next_bounded_mutation_tool([], contract, {}, context, calls_used=8) == "repo.edit"
    assert _next_bounded_mutation_tool(
        [{"tool": "fs.read", "ok": True}], contract, {}, context, calls_used=1
    ) == "repo.edit"

    trace = [{
        "tool": "repo.edit",
        "ok": True,
        "result": {"value": {"status": "accepted", "applied": True}},
    }]
    assert _next_bounded_mutation_tool(trace, contract, {}, context, calls_used=9) == "tests.run"
    trace.append({
        "tool": "tests.run",
        "ok": True,
        "result": {"value": {"verified": True, "returncode": 0}},
    })
    assert _next_bounded_mutation_tool(trace, contract, {}, context, calls_used=10) == "git.status"
    trace.append({"tool": "git.status", "ok": True, "result": {"value": {}}})
    assert _next_bounded_mutation_tool(trace, contract, {}, context, calls_used=11) == "git.diff"


def test_build_contract_does_not_require_legacy_mutation_plan_kind():
    contract = {
        "required_successful_tools": [
            "repo.edit", "tests.run", "git.status", "git.diff",
        ],
        "read_only_tool_call_limit": 2,
    }
    context = {"tool_write_authority": True, "allowed_tool_names": ["fs.search"]}
    assert _next_bounded_mutation_tool(
        [{"tool": "fs.read", "ok": True}],
        contract,
        {},
        context,
        calls_used=1,
    ) == "repo.edit"


def test_build_contract_forces_initial_scoped_search_before_mutation():
    contract = {
        "required_successful_tools": [
            "repo.edit", "tests.run", "git.status", "git.diff",
        ],
        "read_only_tool_call_limit": 2,
    }
    context = {"tool_write_authority": True, "allowed_tool_names": ["fs.search"]}
    assert _next_bounded_mutation_tool(
        [], contract, {}, context, calls_used=0,
    ) == "fs.search"
def test_http200_semantic_gate_rejects_empty_and_non200():
    from hawking.backends import _http200_semantic_gate

    class _Resp:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self.body = body

    assert _http200_semantic_gate(_Resp(200, "ok")) is True
    assert _http200_semantic_gate(_Resp(200, b"ok")) is True
    assert _http200_semantic_gate(_Resp(200, "   ")) is False
    assert _http200_semantic_gate(_Resp(200, "")) is False
    assert _http200_semantic_gate(_Resp(200, None)) is False
    assert _http200_semantic_gate(_Resp(500, "ok")) is False
    assert _http200_semantic_gate(None) is False
