"""The agentic loop: a model that needs to LOOK at something can now say so.

Before this, HCLI_RESULT_SCHEMA was edit-only -- `operations` are file edits, so a
model with a question about the repo had no field to put a tool call in. 61
registered tools were unreachable from the natural-language path BY SCHEMA
CONSTRUCTION, and the measured symptom was HCLI answering "no deterministic
evidence provided" to "list the python files in the hcli directory".

These tests use a stub model so the loop's control flow is checked deterministically,
without a resident. The live end-to-end run is separate and slow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.engine import Engine, HCLI_RESULT_SCHEMA
from hcli.backends import StructuredOutputContract, schema_instruction
from hcli.tool_registry import default_tool_registry
from hcli.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]


def _registry():
    return default_tool_registry(REPO, repo_root=REPO)


def test_the_schema_can_express_a_tool_call_at_all():
    """The defect was structural: there was nowhere to put one."""
    props = HCLI_RESULT_SCHEMA["properties"]
    assert "tool_calls" in props, "no field a tool call could occupy"
    assert "tool_use" in props["kind"]["enum"]
    # Only the universal envelope fields are required. Mode-specific arrays
    # are defaulted by Engine._sanitize_result, which keeps short answers
    # short without weakening the mutation evidence gate.
    assert set(HCLI_RESULT_SCHEMA["required"]) == {"kind", "content"}
    assert {"operations", "tests", "tool_calls"}.issubset(props)
    item = props["tool_calls"]["items"]["properties"]
    assert set(item) == {"tool", "arguments"}
    # Arguments must NOT be a JSON-encoded string. That version was tried and
    # measurably failed: the model could not escape quotes inside quotes and
    # blew all 3 structured-output attempts on "Expecting ',' delimiter".
    assert item["arguments"]["type"] == "array"
    assert item["arguments"]["items"]["properties"]["value"]["type"] == "string"


def test_string_pairs_become_the_types_each_tool_declares():
    """The model can only emit strings; tools want ints and bools."""
    registry = _registry()
    args = Engine._typed_arguments(registry, "fs.list", [
        {"name": "path", "value": "hcli"},
        {"name": "max_results", "value": "5"},
        {"name": "recursive", "value": "false"},
    ])
    assert args == {"path": "hcli", "max_results": 5, "recursive": False}
    # and the coerced call must actually satisfy the registry
    result = registry.invoke("fs.list", args)
    assert result.ok, result.error
    assert len(result.value["files"]) == 5


def test_an_uncoercible_value_reaches_the_registry_as_a_readable_error():
    """Never guess. A bad value must surface as text the model can react to."""
    registry = _registry()
    args = Engine._typed_arguments(registry, "fs.list", [
        {"name": "path", "value": "hcli"},
        {"name": "max_results", "value": "not-a-number"},
    ])
    assert args["max_results"] == "not-a-number"  # left alone, not invented
    assert registry.invoke("fs.list", args).ok is False


def test_a_failing_tool_is_an_observation_not_an_exception():
    """A daemon that dies on a bad argument is not unattended.

    Every failure mode here -- unknown tool, missing required argument, a handler
    that raises -- must come back as readable text.
    """
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    engine.MAX_EVIDENCE_CHARS_PER_FILE = 4000
    engine._emit = lambda *a, **k: None

    out = engine._run_tool_calls([
        {"tool": "no.such.tool", "arguments": []},
        {"tool": "fs.search", "arguments": [{"name": "root", "value": "hcli"}]},
        {"tool": "fs.read", "arguments": [{"name": "path", "value": "does/not/exist.py"}]},
    ], goal_id="g")

    assert len(out) == 3
    assert all(o["ok"] is False for o in out)
    assert "unknown tool" in out[0]["text"]
    # THE FIX THAT MATTERS: the signature travels WITH the error, so the retry
    # does not have to guess. Without it the model burned a whole tool budget
    # calling fs.search without `pattern` and guessing again each round.
    assert "pattern" in out[1]["text"] and "signature:" in out[1]["text"]
    assert "pattern*" in out[1]["text"], "required args must be marked"


def test_the_catalog_tells_the_model_the_arguments_not_just_the_names():
    catalog = Engine._tool_catalog(_registry())
    assert "fs.list(" in catalog
    line = next(l for l in catalog.splitlines() if l.startswith("fs.search("))
    assert "pattern*:string" in line, "required marker missing"
    assert "root:string" in line and "root*" not in line, "optional marked required"


def test_observation_round_catalog_is_alias_aware_and_focused():
    registry = _registry()
    full = Engine._tool_catalog(registry)
    focused = Engine._compact_tool_catalog(
        registry,
        focus="list the python files in the hcli directory",
    )
    assert len(focused) < len(full) // 2, (len(full), len(focused))
    assert "fs.list|filesystem.list" in focused
    assert "git.checkout-safe" not in focused  # destructive capability is opt-in
    assert "git.checkout/revert-safe" not in focused  # slash alias stays callable but not model-facing
    assert "odyssey:" not in focused

    rollback = Engine._compact_tool_catalog(registry, focus="inspect git revert safety")
    assert "git.checkout-safe|git.revert-safe" in rollback
    assert "git.checkout/revert-safe" not in rollback


def test_compact_catalog_expands_the_domain_selected_by_the_goal():
    catalog = Engine._compact_tool_catalog(
        _registry(),
        focus="audit Odyssey and the Gravity compression experiment",
    )
    assert "odyssey:" in catalog
    assert "gravity:" in catalog
    assert "odyssey: " in catalog and "status()" in catalog


def test_tools_catalog_reveals_a_focused_signature_without_running_it():
    registry = _registry()
    result = registry.invoke("tools.catalog", {"focus": "directory listing"})

    assert result.ok, result.error
    names = {item["name"] for item in result.value["matches"]}
    assert "fs.list" in names
    assert result.value["provenance"] == "hcli.tool_registry.ToolRegistry.describe"


def test_context_recall_is_a_bounded_typed_tool(tmp_path):
    from hcli.knowledge import KnowledgeStore

    store = KnowledgeStore(tmp_path)
    store.record_note("remember the overnight production gate", source="test")
    registry = default_tool_registry(tmp_path, repo_root=tmp_path)

    result = registry.invoke("context.recall", {"focus": "overnight production"})

    assert result.ok, result.error
    assert result.value["retrieval"]["mode"] == "cold_recall"
    assert any(
        "overnight production gate" in json.dumps(item)
        for item in result.value["records"]
    )


def test_a_directory_listing_verb_exists():
    """It did not, and that cost an entire tool budget on the first real query.

    61 tools and none could answer "what files are in this directory": fs.search
    requires a content `pattern`, so listing was inexpressible. The model was not
    confused, the capability was absent.
    """
    registry = _registry()
    assert registry.get("fs.list") is not None
    result = registry.invoke("fs.list", {"path": "hcli", "glob": "*.py", "recursive": False})
    assert result.ok, result.error
    names = {row["path"] for row in result.value["files"]}
    assert "engine.py" in names and "tool_registry.py" in names
    assert all(row["path"].endswith(".py") for row in result.value["files"])


def test_directory_listing_includes_immediate_directories(tmp_path):
    (tmp_path / "child").mkdir()
    (tmp_path / "note.txt").write_text("observed\n", encoding="utf-8")
    registry = default_tool_registry(tmp_path, repo_root=tmp_path)

    result = registry.invoke(
        "fs.list",
        {"path": ".", "recursive": False, "max_results": 10},
    )

    assert result.ok, result.error
    assert result.value["directories"] == [{"path": "child", "kind": "directory"}]
    assert result.value["files"] == [{"path": "note.txt", "bytes": 9}]


def test_obvious_directory_question_uses_the_typed_tool_without_model_startup(tmp_path):
    (tmp_path / "child").mkdir()
    (tmp_path / "note.txt").write_text("observed\n", encoding="utf-8")

    class ModelMustNotStart:
        def complete(self, **_kwargs):
            raise AssertionError("simple directory listing should not cold-start the model")

    engine = Engine(Workspace(str(tmp_path)), model_client=ModelMustNotStart())
    result = engine.execute(
        "What is in this directory? Inspect it with the available tools and answer from observed files."
    )

    assert result["status"] == "completed"
    assert "[directory] child/" in result["content"]
    assert "[file] note.txt (9 bytes)" in result["content"]
    assert result["receipt"].endswith(".json")
    assert engine._model_calls == []


def test_tool_output_never_enters_the_evidence_list():
    """Evidence items are hashed, size/mtime-stamped file snapshots that get
    re-read when they change. Tool output is none of those. Letting it into the
    same list would let unhashed, model-directed content pose as deterministic
    evidence and quietly weaken the freshness gate."""
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    text = engine._prompt_with_observations(
        "the goal", [{"tool": "fs.list", "ok": True, "text": "engine.py"}]
    )
    assert "OBSERVATIONS" in text and "engine.py" in text
    assert "AVAILABLE TOOLS" in text
    assert "TOOL BUDGET EXHAUSTED" not in text
    final = engine._prompt_with_observations("the goal", [], final=True)
    assert "TOOL BUDGET EXHAUSTED" in final


def test_observation_round_keeps_the_tool_catalog_prefix_stable():
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    before = engine._prompt_with_observations("the goal", [])
    after = engine._prompt_with_observations(
        "the goal",
        [{"tool": "fs.read", "ok": True, "text": "new observation"}],
    )
    before_catalog = before.split("OBSERVATIONS", 1)[0].rstrip()
    after_catalog = after.split("OBSERVATIONS", 1)[0].rstrip()
    assert before_catalog == after_catalog


def test_tool_round_history_keeps_schema_in_the_stable_user_turn(tmp_path):
    engine = Engine(Workspace(str(tmp_path)))
    payload = engine._build_model_payload(
        "the goal",
        [],
        None,
        history=[
            {"role": "assistant", "content": '{"kind":"tool_use"}'},
            {"role": "user", "content": "OBSERVATIONS (tool results):\nresult"},
        ],
    )
    prepared = StructuredOutputContract(
        HCLI_RESULT_SCHEMA, schema_instruction(HCLI_RESULT_SCHEMA)
    ).apply(payload)
    assert prepared["messages"][1]["role"] == "user"
    assert "MUST satisfy this JSON Schema" in prepared["messages"][1]["content"]
    assert "MUST satisfy this JSON Schema" not in prepared["messages"][-1]["content"]
def test_failed_tool_loop_can_close_catalog_for_one_final_model_call():
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    closed = engine._prompt_with_observations(
        "finish the goal",
        [{"tool": "fs.read", "ok": False, "repeat": True, "text": "missing"}],
        tools_allowed=False,
    )
    assert "TOOL ACCESS IS CLOSED" in closed
    assert "AVAILABLE TOOLS" not in closed
    assert "missing" in closed


def test_implementation_close_requires_a_real_existing_proof_path():
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    closed = engine._prompt_with_observations(
        "ROLE: implementation\nOBJECTIVE: repair the engine",
        [],
        tools_allowed=False,
    )
    assert "hcli/test_engine_tool_loop.py" in closed
    assert "hcli/tests/test_engine.py" not in closed
    assert "comment-only" in closed
    assert "exactly one operation" in closed
    assert "under 900 UTF-8 bytes" in closed
    assert "never use `create`" in closed
    assert "never trailing spaces" in closed


def test_no_tools_enters_the_closed_budget_path_on_the_first_call(monkeypatch):
    engine = Engine.__new__(Engine)
    engine._cancelled = False
    engine.MAX_TOOL_ROUNDS = 4
    prompts = []
    engine._prompt_with_observations = (
        lambda *args, **kwargs: prompts.append(kwargs) or "closed"
    )
    engine._call_model = lambda *args, **kwargs: {
        "kind": "answer",
        "content": "bounded",
    }
    engine._sanitize_result = lambda value: value
    engine._emit = lambda *args, **kwargs: None
    engine._write_receipt = lambda **kwargs: "receipt"
    monkeypatch.setenv("HCLI_NO_TOOLS", "1")

    result = engine.execute("ROLE: implementation\nOBJECTIVE: repair", evidence=[], compiled={})

    assert result["kind"] == "answer"
    assert len(prompts) == 1
    assert prompts[0]["tools_allowed"] is False


def test_sanitizer_accepts_tool_use_and_drops_nameless_calls():
    engine = Engine.__new__(Engine)
    out = engine._sanitize_result({
        "kind": "tool_use", "content": "looking",
        "operations": [], "tests": [],
        "tool_calls": [
            {"tool": "fs.list", "arguments": [{"name": "path", "value": "hcli"}]},
            {"tool": "   ", "arguments": []},
            "not-a-dict",
        ],
    })
    assert out["kind"] == "tool_use"
    assert [c["tool"] for c in out["tool_calls"]] == ["fs.list"]


def test_an_answer_still_carries_no_tool_calls():
    engine = Engine.__new__(Engine)
    out = engine._sanitize_result(
        {"kind": "answer", "content": "hi", "operations": [], "tests": []}
    )
    assert out["kind"] == "answer" and out["tool_calls"] == []
