"""The chat's tool loop: what it must never do.

The failures guarded here are the ones that produce a confident wrong answer
rather than an error:

  * claiming a tool ran when the body never emitted a usable action
  * offering a menu whose argument shapes disagree with the tools' schemas
  * reporting "no matches" from a search that stopped before it reached the code
  * losing the source URLs behind a web answer
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hcli.chat_tools import (
    CHAT_TOOLS,
    argument_shape,
    build_registry,
    parse_call,
    parse_calls,
    provenance_of,
    qualify,
    run_with_tools,
    session_menu,
    _search_needs_escalation,
)


class _Result:
    def __init__(self, ok=True, value=None, error=None, provenance=None, failure_class=None):
        self.ok, self.value, self.error = ok, value, error
        self.provenance = provenance or {}
        self.failure_class = failure_class


class _Registry:
    def __init__(self, results=None):
        self.calls = []
        self._results = results or {}

    def invoke(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return self._results.get(name, _Result(value={"ok": True}))

    def get(self, name):
        return None


class _OfferingRegistry(_Registry):
    """Small fake whose named results are also discoverable session doors."""

    context = type("Context", (), {
        "permissions": ("READ_ONLY", "RESEARCH"),
    })()

    def get(self, name):
        return object() if name in self._results else None

    def discover(self, include_aliases=True):
        del include_aliases
        return [
            {"name": name, "description": name, "mutation": "READ_ONLY"}
            for name in self._results
        ]


class _AcceptingEngine:
    def apply_typed_mutation(self, operations, tests=None):
        del tests
        return {
            "status": "accepted",
            "applied": True,
            "rolled_back": False,
            "reason": None,
            "diff": "+ accepted",
            "paths": [str(operations[0]["path"])],
        }


def _scripted(replies):
    it = iter(replies)

    def complete(_conversation):
        return next(it)
    return complete


class TestTheMenuMatchesTheTools(unittest.TestCase):
    """A menu that disagrees with a schema sends the model into a loop.

    Measured: this table documented fs.search as taking {"query": ...} while the
    tool requires `pattern`, so every search failed and a body burned its whole
    budget correcting a shape the menu had wrong.
    """

    def setUp(self):
        self.registry = build_registry(".", ".")

    def test_every_offered_tool_is_served_by_exactly_one_owner(self):
        # Two classes of door: registry tools, and retrieval doors HCLI serves
        # itself. Every entry must have exactly one owner -- a name in neither
        # is a menu entry pointing at nothing, and a name in both is ambiguous.
        from hcli.chat_tools import LOCAL_TOOLS
        for name in CHAT_TOOLS:
            in_registry = self.registry.get(name) is not None
            is_local = name in LOCAL_TOOLS
            self.assertTrue(
                in_registry != is_local,
                f"{name}: registry={in_registry} local={is_local} -- a door must "
                f"have exactly one owner, or it points at nothing / is ambiguous")

    def test_every_offered_tool_advertises_its_real_required_arguments(self):
        from hcli.chat_tools import LOCAL_SHAPES, LOCAL_TOOLS
        for name in CHAT_TOOLS:
            shape = argument_shape(self.registry, name)
            self.assertTrue(shape, f"{name} advertises no argument shape")
            if name in LOCAL_TOOLS:
                self.assertIn(name, LOCAL_SHAPES,
                              f"{name} is served locally with no declared shape")
                continue
            spec = self.registry.get(name)
            schema = getattr(spec, "input_schema", None) or {}
            for field in (schema.get("required") or []):
                self.assertIn(f'"{field}"', shape,
                              f"{name} requires {field!r} and the menu never says so")

    def test_no_offered_tool_can_mutate(self):
        # The registry is built read/research; assert the property rather than
        # trusting the construction call stayed that way. Local doors are
        # retrieval of material already on disk and write nothing.
        from hcli.chat_tools import LOCAL_TOOLS
        for name in CHAT_TOOLS:
            if name in LOCAL_TOOLS:
                continue
            spec = self.registry.get(name)
            self.assertIn(getattr(spec, "mutation", None), ("read_only", "research"),
                          f"{name} is not read-only/research")

    def test_write_session_reaches_every_non_destructive_registry_name(self):
        registry = build_registry(".", ".", write=True)
        menu = session_menu(registry, write=True)
        permissions = set(registry.context.permissions)
        for item in registry.discover(include_aliases=True):
            if item["mutation"] in permissions:
                self.assertIn(item["name"], menu)
        self.assertNotIn("git.checkout-safe", menu)

    def test_prompt_surface_is_compact_but_core_tools_remain_visible(self):
        from hcli.chat_tools import prompt_menu_names
        registry = build_registry(".", ".", write=True)
        offered = session_menu(registry, write=True)
        selected = prompt_menu_names(registry, write=True)
        self.assertLess(len(selected), len(offered))
        self.assertTrue({
            "fs.read", "fs.search", "receipt.read", "tools.catalog",
            "campaign.state", "repo.edit", "tests.run",
        } <= set(selected))
        self.assertTrue(set(selected) <= set(offered))


class TestParsing(unittest.TestCase):
    def test_a_bare_object(self):
        self.assertEqual(parse_call('{"tool": "fs.list", "arguments": {"path": "."}}'),
                         ("fs.list", {"path": "."}))

    def test_json_illegal_regex_escape_does_not_drop_a_real_tool_decision(self):
        got = parse_call(
            '{"tool":"fs.search","arguments":{"pattern":".*\\.json"}}')
        self.assertEqual(got, ("fs.search", {"pattern": ".*\\.json"}))

    def test_a_fenced_object(self):
        got = parse_call('```json\n{"tool": "fs.read", "arguments": {"path": "a.py"}}\n```')
        self.assertEqual(got, ("fs.read", {"path": "a.py"}))

    def test_an_object_buried_in_prose(self):
        got = parse_call('Let me look.\n{"tool": "fs.list", "arguments": {"path": "hcli"}}\nOK?')
        self.assertEqual(got[0], "fs.list")

    def test_plain_prose_is_not_a_call(self):
        self.assertIsNone(parse_call("I will search the web for that."))

    def test_an_answer_mentioning_tools_is_not_a_call(self):
        self.assertIsNone(parse_call("You could use a web search tool for this."))

    def test_an_explicit_zero_argument_narrative_action_is_adapted(self):
        self.assertEqual(
            parse_call("I will use the `fs.list` tool to inspect the repository."),
            ("fs.list", {}),
        )

    def test_zero_argument_narrative_function_dialect_is_adapted(self):
        self.assertEqual(
            parse_call("I will use the `fs.list` function to inspect the repository."),
            ("fs.list", {}),
        )

    def test_narrative_action_with_required_arguments_is_not_guessed(self):
        self.assertIsNone(
            parse_call("I will use the `fs.read` tool to inspect the repository."))

    def test_negative_narrative_tool_mention_is_not_adapted(self):
        self.assertIsNone(parse_call("Do not use the `fs.list` tool here."))

    def test_batched_json_actions_are_consumed_in_order(self):
        text = (
            '```json\n{"tool":"fs.search","arguments":{"pattern":"x"}}\n```\n'
            '```json\n{"tool":"fs.list","arguments":{"path":"hcli"}}\n```'
        )
        self.assertEqual(parse_calls(text), [
            ("fs.search", {"pattern": "x"}),
            ("fs.list", {"path": "hcli"}),
        ])

    def test_compact_op_dialect_maps_only_bounded_read_tools(self):
        self.assertEqual(
            parse_call('```json\n{"op":"read","path":"receipt.json"}\n```'),
            ("fs.read", {"path": "receipt.json"}),
        )
        self.assertIsNone(parse_call('{"op":"shell","command":"ls"}'))
        self.assertEqual(
            parse_call('{"op":"receipt.read","path":"receipts/x.json"}'),
            ("receipt.read", {"path": "receipts/x.json"}),
        )

    def test_compact_builder_dialect_maps_to_authority_checked_doors(self):
        self.assertEqual(
            parse_call('{"op":"edit","operations":[]}'),
            ("repo.edit", {"operations": []}),
        )
        self.assertEqual(
            parse_call('{"op":"test","paths":["hcli/tests/test_x.py"]}'),
            ("tests.run", {"paths": ["hcli/tests/test_x.py"]}),
        )

    def test_compact_multi_edit_becomes_one_typed_transaction(self):
        from hcli.chat_tools import coerce_arguments

        call = parse_call(
            '{"op":"edit","edits":['
            '["hcli/cli.py","old source","new source"],'
            '["hcli/test_cli.py","old test","new test"]],'
            '"test":"hcli/test_cli.py"}'
        )
        self.assertIsNotNone(call)
        name, arguments = call
        self.assertEqual(name, "repo.edit")
        self.assertEqual(coerce_arguments(None, name, arguments), {
            "operations": [
                {"op": "replace", "path": "hcli/cli.py",
                 "old_lines": ["old source"], "new_lines": ["new source"]},
                {"op": "replace", "path": "hcli/test_cli.py",
                 "old_lines": ["old test"], "new_lines": ["new test"]},
            ],
            "tests": ["hcli/test_cli.py"],
        })

    def test_compact_edit_survives_template_marker_and_fstring_anchor(self):
        from hcli.chat_tools import coerce_arguments

        call = parse_call(
            '{"op":"edit","edits":[{"path":"hcli/test_cli.py",'
            '"old":"assert argv, f\\"bad: {argv}\\"",'
            '"new":"assert argv[0], f\\"bad: {argv}\\""}],'
            '"test":"hcli/test_cli.py"}<|im_end|>'
        )
        self.assertIsNotNone(call)
        name, arguments = call
        self.assertEqual(name, "repo.edit")
        self.assertEqual(coerce_arguments(None, name, arguments), {
            "operations": [{
                "op": "replace",
                "path": "hcli/test_cli.py",
                "old_lines": ['assert argv, f"bad: {argv}"'],
                "new_lines": ['assert argv[0], f"bad: {argv}"'],
            }],
            "tests": ["hcli/test_cli.py"],
        })

    def test_selected_lines_survive_a_truncated_structural_observation(self):
        from hcli.chat_tools import _selected_observation_lines

        row = {"observation": (
            'fs.read returned:\n{"content_selection":{"selected_lines":['
            '{"line":357,\n"text":"exec -m hcli"},'
            '{"line":93,\n"text":"assert \\\"current\\\" in dtext"}]},'
            '"structure":\n... [structured summary truncated]')}
        self.assertEqual(_selected_observation_lines(row), [
            (357, "exec -m hcli"),
            (93, 'assert "current" in dtext'),
        ])

    def test_action_args_dialect_maps_to_the_canonical_tool(self):
        self.assertEqual(
            parse_call('{"action":"campaign.state","args":{}}'),
            ("campaign.state", {}),
        )

    def test_operation_dialect_maps_to_the_canonical_tool(self):
        self.assertEqual(
            parse_call('{"operation":"git.status","arguments":{}}'),
            ("git.status", {}),
        )

    def test_nested_compact_operation_tag_is_removed_only_for_its_tool(self):
        from hcli.chat_tools import coerce_arguments
        self.assertEqual(
            coerce_arguments(None, "fs.search",
                             {"op": "search", "pattern": "receipt"}),
            {"pattern": "receipt"},
        )
        self.assertEqual(
            coerce_arguments(None, "fs.search",
                             {"op": "read", "pattern": "receipt"}),
            {"op": "read", "pattern": "receipt"},
        )
        self.assertEqual(
            coerce_arguments(None, "receipt.read",
                             {"op": "read", "path": "receipts/x.json"}),
            {"path": "receipts/x.json"},
        )
        self.assertEqual(
            coerce_arguments(None, "fs.read",
                             {"path": "hcli/serve.py", "line": "728"}),
            {"path": "hcli/serve.py", "start_line": 728},
        )
        self.assertEqual(
            coerce_arguments(None, "fs.search", {"query": "session state"}),
            {"pattern": "session state"},
        )


class TestTheLoop(unittest.TestCase):
    def test_an_answer_with_no_action_returns_immediately(self):
        registry = _Registry()
        text, trace = run_with_tools(_scripted(["Paris."]), [], registry)
        self.assertEqual(text, "Paris.")
        self.assertEqual(trace, [])
        self.assertEqual(registry.calls, [], "a tool ran for a question that needed none")

    def test_explicit_tool_obligation_retries_unsupported_prose_then_dispatches(self):
        registry = _Registry({"fs.read": _Result(value={"text": "status: PASS"})})
        text, trace = run_with_tools(
            _scripted([
                "I used the tool and the status is DISPROVEN.",
                '{"tool": "fs.read", "arguments": {"path": "receipts/x.json"}}',
                "The observed status is PASS.",
            ]),
            [{"role": "user", "content": (
                "Use the repository file-reading tool to read receipts/x.json. "
                "Do not guess; use the returned observation."
            )}],
            registry,
        )
        self.assertEqual(text, "The observed status is PASS.")
        self.assertEqual(registry.calls, [("fs.read", {"path": "receipts/x.json"})])
        self.assertFalse(trace[0]["dispatched"])
        self.assertTrue(trace[1]["dispatched"])

    def test_explicit_tool_obligation_fails_closed_after_one_retry(self):
        registry = _Registry()
        text, trace = run_with_tools(
            _scripted(["I checked it.", "I definitely checked it."]),
            [{"role": "user", "content": "Please use a tool to inspect the repository."}],
            registry,
        )
        self.assertIn("could not satisfy the required tool observation", text)
        self.assertEqual(registry.calls, [])
        self.assertEqual(len(trace), 2)
        self.assertTrue(all(not row["dispatched"] for row in trace))

    def test_plural_hcli_tools_is_an_explicit_observation_obligation(self):
        registry = _Registry()
        text, trace = run_with_tools(
            _scripted(["--- fs.search\n" * 1000, "I still only planned it."]),
            [{"role": "user", "content": (
                "Use HCLI tools to inspect the repository, then report exact "
                "files and results. Do not guess without returned observations."
            )}],
            registry,
        )
        self.assertIn("could not satisfy the required tool observation", text)
        self.assertEqual(registry.calls, [])
        self.assertEqual(len(trace), 2)
        self.assertLess(len(trace[0]["reply_excerpt"]), 300)

    def test_explicit_no_tool_request_does_not_trigger_observation_retry(self):
        registry = _Registry()
        text, trace = run_with_tools(
            _scripted(["DIRECT"]),
            [{"role": "user", "content": "Do not use a tool; reply DIRECT."}],
            registry,
        )
        self.assertEqual(text, "DIRECT")
        self.assertEqual(trace, [])

    def test_capability_surface_curveball_requires_a_real_observation(self):
        from hcli.chat_tools import explicit_tool_observation_required
        self.assertTrue(explicit_tool_observation_required([{
            "role": "user",
            "content": (
                "Discover the complete admitted capability surface.\n"
                "Use every safe and materially relevant capability.\n"
                "Do not use irrelevant network tools."),
        }]))
        self.assertFalse(explicit_tool_observation_required([{
            "role": "user",
            "content": "Do not use a tool; reply DIRECT.",
        }]))

    def test_blind_write_mission_cannot_finish_before_mutation_and_test(self):
        prompt = (
            "Discover the complete admitted capability surface. Use every safe "
            "and materially relevant capability. Produce an accepted mutation "
            "and a focused proving test. Return a JSON object in exactly this "
            "structure:\n"
            '{"mission":"PASS | FAIL","tools":{"successful_invocations":[]},'
            '"mutation":{"accepted":false,"paths":[]},'
            '"verification":{"passed":false,"returncode":null}}'
        )
        registry = _OfferingRegistry({
            "tools.catalog": _Result(value={
                "names": ["repo.edit", "tests.run", "tools.catalog"],
                "shown": 3,
                "match_count": 3,
                "truncated": False,
                "focus": "HCLI",
            }),
            "tests.run": _Result(value={"verified": True, "returncode": 0}),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"tools.catalog","arguments":{"focus":"HCLI"}}',
                '{"mission":"FAIL","tools":{"successful_invocations":'
                '["tools.catalog"]},"mutation":{"accepted":false,"paths":[]},'
                '"verification":{"passed":false,"returncode":null}}',
                '{"tool":"repo.edit","arguments":{"operations":[{"op":'
                '"append","path":"hcli/example.py","new_lines":["x = 1"]}],'
                '"tests":["hcli/tests/test_chat_tools.py"]}}',
                '{"mission":"FAIL","tools":{"successful_invocations":'
                '["repo.edit","tools.catalog"]},"mutation":{"accepted":true,'
                '"paths":["hcli/example.py"]},"verification":{"passed":false,'
                '"returncode":null}}',
                '{"tool":"tests.run","arguments":{"paths":'
                '["hcli/tests/test_chat_tools.py"]}}',
                '{"mission":"PASS","tools":{"successful_invocations":'
                '["repo.edit","tests.run","tools.catalog"]},"mutation":'
                '{"accepted":true,"paths":["hcli/example.py"]},'
                '"verification":{"passed":true,"returncode":0}}',
            ]),
            [{"role": "user", "content": prompt}],
            registry,
            engine=_AcceptingEngine(),
        )
        document = json.loads(text)
        self.assertEqual(document["mission"], "PASS")
        self.assertTrue(document["mutation"]["accepted"])
        self.assertTrue(document["verification"]["passed"])
        self.assertTrue(any("accepted mutation" in (row.get("pending_mission") or [])
                            for row in trace))
        self.assertTrue(any(row.get("pending_mission") == ["focused passing test"]
                            for row in trace))
        self.assertIsNotNone(next(row for row in trace
                                  if row.get("tool") == "repo.edit"
                                  and row.get("applied") is True
                                  and row.get("verdict") == "accepted"))
        self.assertIsNotNone(next(row for row in trace
                                  if row.get("tool") == "tests.run"
                                  and row.get("verified") is True
                                  and row.get("returncode") == 0))

    def test_explicit_accepted_mission_does_not_leave_unproven_edit(self):
        prompt = (
            "Implement the defect repair. Produce an accepted mutation and a "
            "focused passing test. Return a JSON object in exactly this "
            "structure:\n"
            '{"mission":"PASS | FAIL","mutation":{"accepted":false},'
            '"verification":{"passed":false}}'
        )
        registry = _OfferingRegistry({
            "tests.run": _Result(value={"verified": True, "returncode": 0}),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"repo.edit","arguments":{"operations":['
                '{"op":"append","path":"hcli/example.py",'
                '"new_lines":["x = 1"]}]}}',
                '{"tool":"repo.edit","arguments":{"operations":['
                '{"op":"append","path":"hcli/example.py",'
                '"new_lines":["x = 1"]}],"tests":['
                '"hcli/tests/test_example.py"]}}',
                '{"tool":"tests.run","arguments":{"paths":['
                '"hcli/tests/test_example.py"]}}',
                '{"mission":"PASS","mutation":{"accepted":true},'
                '"verification":{"passed":true}}',
            ]),
            [{"role": "user", "content": prompt}], registry,
            engine=_AcceptingEngine(), max_calls=5,
        )
        self.assertEqual(json.loads(text)["mission"], "PASS")
        first = next(row for row in trace if row.get("tool") == "repo.edit")
        self.assertFalse(first["ok"])
        self.assertIn("non-empty `tests` list", first["error"])
        self.assertFalse(first.get("applied", False))
        accepted = next(row for row in trace
                        if row.get("tool") == "repo.edit" and row.get("applied"))
        self.assertEqual(accepted["verdict"], "accepted")

    def test_blind_write_pass_claim_without_trace_is_rejected(self):
        from hcli.chat_tools import _trace_grounding_issues
        claimed = json.dumps({
            "mission": "PASS",
            "mutation": {"accepted": True},
            "verification": {"passed": True},
        })
        issues = "\n".join(_trace_grounding_issues(
            claimed, [], {},
            frozenset({"accepted mutation", "focused passing test"})))
        self.assertIn("mutation.accepted", issues)
        self.assertIn("verification.passed", issues)
        self.assertIn("mission is PASS", issues)

    def test_runtime_ownership_requires_one_hawkingd_ancestry_tree(self):
        from hcli.chat_tools import _runtime_tree_preserved

        owned = [
            {"pid": 100, "ppid": 1, "role": "hawkingd",
             "class": "ESSENTIAL_PERSISTENT"},
            {"pid": 101, "ppid": 100, "role": "resident-provider",
             "class": "ESSENTIAL_PERSISTENT"},
            {"pid": 102, "ppid": 100, "role": "web-ui",
             "class": "ESSENTIAL_EPHEMERAL"},
            {"pid": 103, "ppid": 101, "role": "resident-worker",
             "class": "ESSENTIAL_EPHEMERAL"},
        ]
        self.assertTrue(_runtime_tree_preserved(owned))
        unowned = [dict(row) for row in owned]
        unowned[-1]["ppid"] = 1
        self.assertFalse(_runtime_tree_preserved(unowned))
        duplicated = owned + [{
            "pid": 200, "ppid": 1, "role": "hawkingd",
            "class": "ESSENTIAL_PERSISTENT",
        }]
        self.assertFalse(_runtime_tree_preserved(duplicated))

    def test_a_tool_result_comes_back_and_the_answer_follows(self):
        registry = _Registry({"fs.list": _Result(value={"files": ["a.py"]})})
        seen = []

        def complete(conversation):
            seen.append([dict(row) for row in conversation])
            return (
                '{"tool": "fs.list", "arguments": {"path": "."}}'
                if len(seen) == 1 else "There is one file: a.py"
            )

        text, trace = run_with_tools(
            complete, [], registry)
        self.assertEqual(text, "There is one file: a.py")
        self.assertEqual([t["tool"] for t in trace], ["fs.list"])
        self.assertTrue(trace[0]["ok"])
        self.assertTrue(trace[0]["dispatched"])
        self.assertIn("fs.list returned:", trace[0]["observation_excerpt"])
        response = next(
            row["content"] for row in seen[1]
            if str(row.get("content") or "").startswith("<tool_response>")
        )
        self.assertIn("fs.list returned:", response)
        self.assertTrue(response.endswith("</tool_response>"))

    def test_compaction_safe_ledger_replays_prior_success_before_synthesis(self):
        registry = _Registry({"fs.list": _Result(value={"files": ["a.py"]})})
        seen = []

        def complete(conversation):
            seen.append([str(row.get("content") or "") for row in conversation])
            if len(seen) == 1:
                return '{"tool":"fs.list","arguments":{"path":"."}}'
            return "Observed a.py."

        text, _ = run_with_tools(complete, [], registry)
        self.assertEqual(text, "Observed a.py.")
        ledger = "\n".join(seen[1])
        self.assertIn("HCLI OBSERVATION LEDGER", ledger)
        self.assertIn("SUCCESS fs.list", ledger)
        self.assertIn('"path": "."', ledger)

    def test_textual_batch_is_not_replayed_as_self_priming_context(self):
        registry = _OfferingRegistry({
            "fs.list": _Result(value={"files": ["a.py"]}),
        })
        batch = (
            '{"tool":"fs.list","arguments":{"path":"."}}\n'
            '{"tool":"fs.list","arguments":{"path":"hcli"}}'
        )
        seen = []

        def complete(conversation):
            seen.append([dict(row) for row in conversation])
            return batch if len(seen) == 1 else "Observed both results."

        text, trace = run_with_tools(complete, [], registry)
        self.assertEqual(text, "Observed both results.")
        assistant_turns = [
            row["content"] for row in seen[1]
            if row.get("role") == "assistant"
        ]
        self.assertEqual(assistant_turns, [])
        self.assertEqual(sum(bool(row.get("dispatched")) for row in trace), 2)

    def test_explicit_evidence_categories_remain_open_until_observed(self):
        from hcli.chat_tools import _evidence_obligations, _pending_evidence
        messages = [{"role": "user", "content": (
            "Use actual tools. Discover the repository and HCLI capabilities, "
            "find the current sovereign objective, read one receipt, and "
            "inspect live hawkingd processes."
        )}]
        obligations = _evidence_obligations(messages)
        self.assertEqual(set(obligations), {
            "repository", "tool discovery", "current objective",
            "receipt/evidence artifact", "live runtime",
        })
        trace = [
            {"tool": "git.status", "arguments": {},
             "ok": True, "dispatched": True},
            {"tool": "tools.catalog", "arguments": {"focus": "HCLI"},
             "ok": True, "dispatched": True},
            {"tool": "campaign.state", "arguments": {},
             "ok": True, "dispatched": True},
            {"tool": "receipt.read", "arguments": {"path": "receipts/x.json"},
             "ok": True, "dispatched": True},
            {"tool": "processes.summary", "arguments": {},
             "ok": True, "dispatched": True},
        ]
        self.assertEqual(_pending_evidence(obligations, trace), {})

    def test_readonly_controller_routes_one_explicit_category_after_retry(self):
        registry = _OfferingRegistry({
            "fs.list": _Result(value={"files": ["README.md"]}),
            "campaign.state": _Result(value={
                "continuation": {}, "recent_evidence": [],
                "authoritative_parent": {"path": "civilization/sovereign-goal.txt"},
            }),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"fs.list","arguments":{"path":"."}}',
                "I have not checked the current objective.",
                "I still have not checked it.",
                "The current objective was observed.",
            ]),
            [{"role": "user", "content": (
                "Please use fs.list to inspect the repository, then determine "
                "the current sovereign objective."
            )}], registry, max_calls=4,
        )
        self.assertEqual(text, "The current objective was observed.")
        self.assertIn(("campaign.state", {}), registry.calls)
        routed = next(row for row in trace
                      if row.get("tool") == "campaign.state")
        self.assertEqual(routed["controller_routed_from"], "current objective")
        self.assertEqual(routed["evidence_class"], "HCLI_SERVING_CONTRACT")

    def test_receipt_controller_only_follows_campaign_returned_path(self):
        path = "receipts/future/current.json"
        registry = _OfferingRegistry({
            "campaign.state": _Result(value={
                "continuation": {},
                "recent_evidence": [{"receipt": path, "mtime": 1}],
            }),
            "receipt.read": _Result(value={"document": {"status": "PASS"}}),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"campaign.state","arguments":{}}',
                "I have not read a receipt.",
                "I still have not read it.",
                "The returned receipt status is PASS.",
            ]),
            [{"role": "user", "content": (
                "Use actual tools to determine the current objective and find "
                "and read one receipt."
            )}], registry, max_calls=4,
        )
        self.assertEqual(text, "The returned receipt status is PASS.")
        self.assertIn(("receipt.read", {"path": path}), registry.calls)
        routed = next(row for row in trace if row.get("tool") == "receipt.read")
        self.assertEqual(routed["controller_routed_from"],
                         "receipt/evidence artifact")
        self.assertNotIn("controller_routed_from", trace[0])

    def test_source_controller_reads_the_exact_search_hit_window(self):
        path = "/repo/hcli/cli.py"
        registry = _OfferingRegistry({
            "fs.search": _Result(value={
                "root": "/repo",
                "pattern": "installed",
                "matches": [{
                    "path": path,
                    "line": 282,
                    "text": "running the installed snapshot",
                }],
                "truncated": False,
            }),
            "fs.read": _Result(value={
                "path": path,
                "content": "def install_shims():\n    pass\n",
                "bytes": 31,
                "shown_bytes": 31,
                "truncated": False,
                "start_line": 262,
            }),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"fs.search","arguments":'
                '{"path":".","pattern":"installed"}}',
                "I found a source location but have not read it.",
                "I still have not read it.",
                "The matching implementation was read.",
            ]),
            [{"role": "user", "content": (
                "Use actual tools to reproduce the installed HCLI shadowing "
                "defect and implement the smallest durable repair."
            )}], registry, max_calls=4,
        )
        self.assertEqual(text, "The matching implementation was read.")
        self.assertIn(("fs.read", {
            "path": path,
            "start_line": 262,
            "end_line": 402,
        }), registry.calls)
        routed = next(row for row in trace if row.get("tool") == "fs.read")
        self.assertEqual(routed["controller_routed_from"], "source inspection")
        self.assertEqual(routed["evidence_class"], "HCLI_SERVING_CONTRACT")

    def test_discriminating_mission_reads_a_test_before_running_it(self):
        path = "/repo/hcli/test_install_shims.py"
        registry = _OfferingRegistry({
            "fs.search": _Result(value={
                "root": "/repo", "pattern": "installed",
                "matches": [{"path": path, "line": 31,
                             "text": "def test_installed_shim():"}],
                "truncated": False,
            }),
            "fs.read": _Result(value={
                "path": path, "content": "def test_installed_shim():\n    pass\n",
                "bytes": 40, "shown_bytes": 40, "truncated": False,
                "start_line": 1,
            }),
            "tests.run": _Result(value={"verified": True, "returncode": 0}),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"fs.search","arguments":'
                '{"path":".","pattern":"installed","glob":"test*.py"}}',
                "I found a test but have not read it.",
                "I still have not read it.",
                '{"tool":"tests.run","arguments":{"paths":['
                '"hcli/test_install_shims.py"]}}',
                "The inspected focused test passed.",
            ]),
            [{"role": "user", "content": (
                "Use actual tools. Inspect a focused test that failed before "
                "the repair, then run that test."
            )}], registry, max_calls=5,
        )
        self.assertEqual(text, "The inspected focused test passed.")
        self.assertIn(("fs.read", {
            "path": path, "start_line": 1, "end_line": 181,
        }), registry.calls)
        read = next(row for row in trace if row.get("tool") == "fs.read")
        self.assertEqual(read["controller_routed_from"],
                         "focused test inspection")

    def test_source_window_projection_keeps_absolute_behavior_lines(self):
        from hcli.chat_tools import _content_selection

        content = "\n".join([
            '"""installation details"""',
            "# background one",
            "# background two",
            "def install_shims():",
            "    current.symlink_to(destination)",
            "    def _script(module):",
            '        return f\'exec "{python}" -P -m {module} "$@"\' ',
            '    assert "-P -m hcli" in script',
        ])
        selected = _content_selection({
            "path": "/repo/hcli/cli.py",
            "content": content,
            "start_line": 340,
            "bytes": len(content),
            "shown_bytes": len(content),
            "truncated": False,
        })
        rows = selected["selected_lines"]
        self.assertIn({"line": 343, "text": "def install_shims():"}, rows)
        self.assertTrue(any(row["line"] == 346 and "exec" in row["text"]
                            for row in rows))
        self.assertTrue(any("assert" in row["text"] for row in rows))

    def test_an_unoffered_tool_is_told_what_IS_available(self):
        registry = _Registry()
        text, trace = run_with_tools(
            _scripted(['{"tool": "shell.exec", "arguments": {"cmd": "rm -rf /"}}',
                       "I cannot do that."]), [], registry)
        self.assertEqual(registry.calls, [], "an unoffered tool was invoked")
        self.assertFalse(trace[0]["ok"])
        self.assertFalse(trace[0]["dispatched"])
        self.assertEqual(text, "I cannot do that.")

    def test_a_failed_observation_gets_one_bounded_recovery_turn(self):
        registry = _Registry({
            "fs.read": _Result(ok=False, error="missing file"),
            "fs.list": _Result(value={"files": ["receipts/x.json"]}),
        })
        text, trace = run_with_tools(
            _scripted([
                '{"tool": "fs.read", "arguments": {"path": "latest"}}',
                '{"tool": "fs.list", "arguments": {"path": "receipts", "glob": "*.json"}}',
                "I recovered the receipt directory.",
            ]),
            [], registry, max_calls=3)
        self.assertEqual(text, "I recovered the receipt directory.")
        self.assertEqual([row["tool"] for row in trace], ["fs.read", "fs.list"])
        self.assertFalse(trace[0]["ok"])
        self.assertTrue(trace[1]["ok"])

    def test_broad_interactive_test_run_is_redirected_to_a_focused_file(self):
        registry = _Registry({"tests.run": _Result(value={
            "verified": True, "returncode": 0,
        })})
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"tests.run","arguments":{"paths":["hcli/tests"]}}',
                '{"tool":"tests.run","arguments":{"paths":["hcli/tests/test_chat_state.py"]}}',
                "The focused test passed.",
            ]),
            [], registry, engine=object(), max_calls=3,
        )
        self.assertEqual(text, "The focused test passed.")
        self.assertFalse(trace[0]["dispatched"])
        self.assertFalse(trace[0]["ok"])
        self.assertIn("broad test directories", trace[0]["error"])
        self.assertEqual(registry.calls, [
            ("tests.run", {"paths": ["hcli/tests/test_chat_state.py"]}),
        ])
        self.assertTrue(trace[1]["verified"])

    def test_the_budget_is_bounded_and_ends_with_an_answer(self):
        registry = _Registry({"fs.list": _Result(value={})})
        # DISTINCT calls, so budget exhaustion (not the repeat guard) is what ends it.
        calls = ['{"tool": "fs.list", "arguments": {"path": "%s"}}' % p
                 for p in ("a", "b", "c")]
        text, trace = run_with_tools(
            _scripted(calls + ["I ran out of budget."]), [], registry, max_calls=3)
        self.assertEqual(len(trace), 3)
        self.assertEqual(text, "I ran out of budget.")

    def test_a_tool_call_after_budget_is_not_rendered_as_if_it_ran(self):
        registry = _Registry({"fs.list": _Result(value={})})
        call = '{"tool":"fs.list","arguments":{"path":"a"}}'
        text, trace = run_with_tools(
            _scripted([call, '{"tool":"fs.read","arguments":{"path":"x"}}']),
            [], registry, max_calls=1,
        )
        self.assertIn("bounded tool budget", text)
        self.assertEqual(registry.calls, [("fs.list", {"path": "a"})])
        self.assertEqual(trace[-1]["error"],
                         "tool call emitted after bounded tool budget")
        self.assertFalse(trace[-1]["dispatched"])

    def test_a_repeated_tool_call_breaks_the_loop_instead_of_churning(self):
        # A greedy body degenerates into re-calling the SAME tool on the same
        # args forever (measured: sealed-3.14 looped observation.expand on a
        # hallucinated id through the whole budget). The loop guard must stop it
        # well before max_calls and never re-execute the identical call.
        registry = _Registry({"fs.read": _Result(ok=False, error="no such file")})
        call = '{"tool": "fs.read", "arguments": {"path": "ghost.py"}}'
        text, trace = run_with_tools(
            _scripted([call] * 12 + ["Answering with what I have."]), [], registry,
            max_calls=12)
        self.assertLessEqual(len(registry.calls), 1,
                             "the identical failing call was re-executed")
        self.assertTrue(any(t.get("error") == "repeated call (loop guard)"
                            for t in trace), "the repeat was not caught")
        self.assertLess(len(trace), 12, "the loop churned to the budget")

    def test_one_repeat_redirect_does_not_stop_a_different_recovery_call(self):
        registry = _Registry({
            "fs.read": _Result(value={"content": "x"}),
            "fs.list": _Result(value={"files": []}),
        })
        repeated = '{"tool":"fs.read","arguments":{"path":"a.py"}}'
        text, trace = run_with_tools(
            _scripted([
                repeated, repeated,
                '{"tool":"fs.list","arguments":{"path":"."}}',
                "Recovered with a different observation.",
            ]), [], registry, max_calls=4)
        self.assertEqual(text, "Recovered with a different observation.")
        self.assertEqual(registry.calls, [
            ("fs.read", {"path": "a.py"}), ("fs.list", {"path": "."}),
        ])
        self.assertEqual(trace[1]["error"], "repeated call (loop guard)")

    def test_third_repeat_in_a_batch_does_not_drop_a_new_later_call(self):
        registry = _Registry({
            "fs.read": _Result(value={"content": "x"}),
            "fs.list": _Result(value={"files": ["b.py"]}),
        })
        old = '{"tool":"fs.read","arguments":{"path":"a.py"}}'
        batch = old + '\n' + '{"tool":"fs.list","arguments":{"path":"."}}'
        text, trace = run_with_tools(
            _scripted([old, old, batch, "Recovered b.py."]),
            [], registry, max_calls=4)
        self.assertEqual(text, "Recovered b.py.")
        self.assertIn(("fs.list", {"path": "."}), registry.calls)
        self.assertTrue(any(row.get("tool") == "fs.list" and row.get("ok")
                            for row in trace))

    def test_a_pure_extension_search_routes_to_file_listing(self):
        registry = _Registry({"fs.list": _Result(value={"files": [
            {"path": "receipts/x.json"},
        ]})})
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"fs.search","arguments":'
                '{"path":"receipts","pattern":".*\\.json"}}',
                "Found receipts/x.json.",
            ]), [], registry)
        self.assertEqual(text, "Found receipts/x.json.")
        self.assertEqual(registry.calls, [("fs.list", {
            "path": "receipts", "glob": "*.json", "recursive": False,
        })])
        self.assertEqual(trace[0]["tool"], "fs.list")
        self.assertEqual(trace[0]["requested_tool"], "fs.search")

    def test_the_conversation_never_exceeds_a_given_prompt_budget(self):
        # SCAR: serve.py compacts the INCOMING messages once, before the tool
        # loop starts -- but every completion INSIDE the loop is unchecked.
        # Measured live: after the batching fix started executing several
        # reads per turn, a real cycle grew the conversation to 10203 tokens
        # against the resident's native max_seq_len=8192, and the backend
        # raised (HawkingNativeError, surfaced to the caller as a bare 502
        # with no information HCLI could act on). A body that never gets a
        # response back cannot repair anything, including this.
        registry = _Registry({"fs.read": _Result(value={"text": "x" * 4000})})
        calls = ['{"tool": "fs.read", "arguments": {"path": "%s"}}' % p
                 for p in ("a", "b", "c", "d", "e")]
        seen_sizes = []

        def complete(convo):
            seen_sizes.append(sum(len(str(m.get("content") or "")) for m in convo))
            i = len(seen_sizes) - 1
            return calls[i] if i < len(calls) else "Answering with what fits."

        # Room for about two turns (~3.1KB each, measured below) -- enough to
        # prove eviction caps growth without asking for less than one turn's
        # worth of grounding, which no admission guard can honor.
        text, trace = run_with_tools(
            complete, [{"role": "system", "content": "objective"}], registry,
            max_calls=6, max_prompt_chars=7000)
        self.assertEqual(text, "Answering with what fits.")
        self.assertEqual(len(trace), 5, "a budget must not silently drop calls")
        # The budget is admission control on what's SENT, not a cap on work
        # done -- assert what the backend actually saw each turn, not what
        # accumulated afterward.
        self.assertTrue(all(s <= 7000 for s in seen_sizes),
                        f"a turn was sent over budget: {seen_sizes}")
        # Structural observation compaction can make all five bounded turns
        # fit without eviction. Prove each added observation stays small and
        # the complete prompt remains under the hard admission ceiling.
        increments = [right - left for left, right in zip(seen_sizes, seen_sizes[1:])]
        self.assertTrue(all(delta < 2200 for delta in increments),
                        f"one observation+ledger caused runaway growth: {seen_sizes}")

    def test_with_no_budget_given_behaviour_is_unchanged(self):
        # The default (no budget) must stay exactly what every other test in
        # this file already assumes -- opt-in, not a silent behavior change.
        registry = _Registry({"fs.read": _Result(value={"text": "x" * 4000})})
        text, trace = run_with_tools(
            _scripted(['{"tool": "fs.read", "arguments": {"path": "a"}}',
                       "Done."]), [], registry)
        self.assertEqual(text, "Done.")
        self.assertEqual(len(trace), 1)

    def test_exact_user_json_shape_gets_one_bounded_correction(self):
        prompt = (
            "Return a compact JSON object in exactly this structure:\n"
            '{"liveness":"PASS | FAIL","hawking":{"found":false},'
            '"tools":[]}')
        text, trace = run_with_tools(
            _scripted([
                "I found Hawking and the tools work.",
                '{"liveness":"PASS","hawking":{"found":true},"tools":["fs.list"]}'
                "<|im_end|>",
            ]), [{"role": "user", "content": prompt}], _Registry())
        self.assertEqual(json.loads(text)["hawking"]["found"], True)
        self.assertNotIn("im_end", text)
        self.assertEqual(trace[-1]["event"],
                         "requested final JSON contract repaired")

    def test_a_single_outer_json_fence_is_transport_format_not_content(self):
        prompt = (
            "Return a JSON object in exactly this structure:\n"
            '{"status":null,"evidence":[]}')
        text, trace = run_with_tools(
            _scripted(['```json\n{"status":"PASS","evidence":[]}\n```']),
            [{"role": "user", "content": prompt}], _Registry())
        self.assertEqual(json.loads(text)["status"], "PASS")
        self.assertNotIn("```", text)
        self.assertEqual(trace, [])

    def test_final_receipt_cannot_erase_successful_dispatches(self):
        prompt = (
            "Use a tool to inspect the repository. Return a JSON object in "
            "exactly this structure:\n"
            '{"liveness":"PASS | FAIL","tools":'
            '{"successful_invocations":[]}}')
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"fs.list","arguments":{"path":"."}}',
                '{"liveness":"PASS","tools":{"successful_invocations":[]}}',
                '{"liveness":"PASS","tools":'
                '{"successful_invocations":["fs.list"]}}',
            ]), [{"role": "user", "content": prompt}],
            _Registry({"fs.list": _Result(value={"files": []})}))
        self.assertEqual(json.loads(text)["tools"]["successful_invocations"],
                         ["fs.list"])
        self.assertIn("recorded successful dispatches",
                      " ".join(trace[-2]["contract_issues"]))
        self.assertEqual(trace[-1]["event"],
                         "requested final JSON contract repaired")

    def test_trace_repairs_only_mechanical_catalog_fields_after_model_retry(self):
        prompt = (
            "Use a tool to discover capabilities. Return a JSON object in "
            "exactly this structure:\n"
            '{"liveness":"PASS | FAIL","tools":'
            '{"discovered_count":0,"broad_categories":[],'
            '"successful_invocations":[]}}')
        catalog = {
            "names": ["campaign.state", "processes.summary", "tools.catalog"],
            "shown": 3,
            "match_count": 3,
            "truncated": False,
            "focus": "HCLI",
        }
        text, trace = run_with_tools(
            _scripted([
                '{"tool":"tools.catalog","arguments":{"focus":"HCLI"}}',
                '{"liveness":"PASS","tools":{"discovered_count":0,'
                '"broad_categories":[],"successful_invocations":[]}}',
                '{"liveness":"PASS","tools":{"discovered_count":0,'
                '"broad_categories":[],"successful_invocations":[]}}',
            ]), [{"role": "user", "content": prompt}],
            _OfferingRegistry({"tools.catalog": _Result(value=catalog)}),
        )
        document = json.loads(text)
        self.assertEqual(document["tools"], {
            "discovered_count": 3,
            "broad_categories": ["campaign", "processes", "tools"],
            "successful_invocations": ["tools.catalog"],
        })
        self.assertEqual(trace[-1]["evidence_class"], "HCLI_SERVING_CONTRACT")
        self.assertEqual(set(trace[-1]["mechanical_fields"]), {
            "tools.discovered_count", "tools.broad_categories",
            "tools.successful_invocations",
        })

    def test_final_grounding_cannot_drop_current_authority_or_read_receipt(self):
        from hcli.chat_tools import _trace_grounding_issues
        trace = [
            {"tool": "git.status", "ok": True, "dispatched": True,
             "evidence_projection": {"cwd": "/repo", "returncode": 0}},
            {"tool": "tools.catalog", "ok": True, "dispatched": True,
             "evidence_projection": {"match_count": 3}},
            {"tool": "campaign.state", "ok": True, "dispatched": True,
             "evidence_projection": {"authoritative_parent": {
                 "path": "civilization/sovereign-goal.txt",
                 "latest_section": "KIMI P0 final operational closeout — now",
             }}},
            {"tool": "receipt.read", "arguments": {"path": "receipts/current.json"},
             "ok": True, "dispatched": True,
             "evidence_projection": {"authoritative_fields": {"status": "RED"}}},
            {"tool": "processes.summary", "ok": True, "dispatched": True,
             "evidence_projection": {"processes": [{
                 "role": "hawkingd",
                 "command": "hawkingd serve :8011 [KIMI_P0_OPERATIONAL]",
             }, {"role": "resident-provider", "command": "provider"}]}},
        ]
        obligations = {
            "repository": frozenset({"git.status"}),
            "tool discovery": frozenset({"tools.catalog"}),
            "current objective": frozenset({"campaign.state"}),
            "receipt/evidence artifact": frozenset({"receipt.read"}),
            "live runtime": frozenset({"processes.summary"}),
        }
        bad = json.dumps({
            "liveness": "PARTIAL",
            "hawking": {"repository_found": True},
            "model_identity": {"independently_verified": False, "evidence": None},
            "tools": {"successful_invocations": ["git.status"]},
            "objective": {"found": True, "source": ".aider.chat.history.md",
                          "summary": "This is Hawking."},
            "evidence_check": {"receipt_or_artifact": None,
                               "measured_fact": None, "provenance": None},
            "daemon": {"reachable": False, "served_model_or_artifact": None,
                       "unexpected_duplicate_runtime": "NOT_ESTABLISHED"},
            "real_action_performed": None,
            "not_established": ["model_identity.independently_verified",
                                "daemon.unexpected_duplicate_runtime"],
        })
        issues = "\n".join(_trace_grounding_issues(bad, trace, obligations))
        self.assertIn("objective.source", issues)
        self.assertIn("objective.summary", issues)
        self.assertIn("receipt_or_artifact", issues)
        self.assertIn("measured_fact", issues)
        self.assertIn("independently_verified", issues)
        self.assertIn("real_action_performed", issues)

        from hcli.chat_tools import _repair_trace_redundant_fields
        repaired_text, fields = _repair_trace_redundant_fields(bad, trace)
        repaired = json.loads(repaired_text)
        self.assertEqual(repaired["liveness"], "PASS")
        self.assertEqual(repaired["objective"], {
            "found": True,
            "source": "civilization/sovereign-goal.txt",
            "summary": "KIMI P0 final operational closeout — now",
        })
        self.assertEqual(repaired["evidence_check"]["receipt_or_artifact"],
                         "receipts/current.json")
        self.assertIn("status=RED", repaired["evidence_check"]["measured_fact"])
        self.assertTrue(repaired["model_identity"]["independently_verified"])
        self.assertEqual(repaired["daemon"]["unexpected_duplicate_runtime"], "NO")
        self.assertEqual(repaired["not_established"], [])
        self.assertIn("liveness", fields)

    def test_exact_json_correction_does_not_inject_missing_answers(self):
        prompt = (
            "Reply with a JSON object in exactly this structure:\n"
            '{"answer":null,"evidence":[]}')
        seen = []

        def complete(conversation):
            seen.append(conversation[-1]["content"])
            return "not json"

        text, trace = run_with_tools(
            complete, [{"role": "user", "content": prompt}], _Registry())
        self.assertIn("could not satisfy", text)
        self.assertIn("exactly the requested keys", seen[-1])
        self.assertNotIn('"answer":', seen[-1],
                         "the correction must enforce shape, not supply values")
        self.assertEqual(trace[-1]["error"],
                         "requested final JSON contract remained incomplete")

    def test_a_batch_of_calls_in_one_reply_all_execute(self):
        # SCAR: sealed-3.14 routinely batches 2-3 reads in ONE reply (measured,
        # .hcli/selfdev/evidence/cycle-0008.txt: three <tool_call> blocks in a
        # single completion). The loop parsed only the FIRST and silently
        # dropped the rest, so a body that batches its whole plan into one
        # turn burned its turn budget re-emitting reads it thought had already
        # run, and never reached a repair. All calls in one reply must execute
        # -- and it must cost only the ONE completion that named them, not one
        # completion per call.
        registry = _Registry({"fs.read": _Result(value={"text": "ok"})})
        batch = ("<tool_call>\n<function=fs.read>\n<parameter=path>\na.py\n"
                "</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=fs.read>\n<parameter=path>\nb.py\n"
                "</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=fs.read>\n<parameter=path>\nc.py\n"
                "</parameter>\n</function>\n</tool_call>")
        text, trace = run_with_tools(
            _scripted([batch, "Read all three."]), [], registry, max_calls=3)
        self.assertEqual(text, "Read all three.")
        self.assertEqual(sorted(a["path"] for _, a in registry.calls),
                         ["a.py", "b.py", "c.py"],
                         "the batch was not fully executed")
        self.assertEqual(len(trace), 3)

    def test_a_failing_tool_is_reported_not_invented(self):
        registry = _Registry({"fs.read": _Result(ok=False, error="no such file")})
        _, trace = run_with_tools(
            _scripted(['{"tool": "fs.read", "arguments": {"path": "nope.py"}}',
                       "That file does not exist."]), [], registry)
        self.assertFalse(trace[0]["ok"])
        self.assertIn("no such file", trace[0]["error"])


class TestQualification(unittest.TestCase):
    """A body that cannot emit an action must not be handed a tool contract."""

    def test_a_body_that_emits_an_action_qualifies(self):
        got = qualify(_scripted(['{"tool": "fs.list", "arguments": {"path": "."}}']))
        self.assertTrue(got["qualified"], got)
        self.assertEqual(got["parsed_tool"], "fs.list")
        self.assertGreater(got["offered_count"], 0)

    def test_a_body_that_answers_in_prose_does_not_qualify(self):
        got = qualify(_scripted(["Sure! I would list the directory for you."]))
        self.assertFalse(got["qualified"])
        self.assertIn("did not emit", got["reason"])
        self.assertIn("reply_excerpt", got)

    def test_any_OFFERED_tool_counts_because_the_body_is_deciding(self):
        # The probe asks a question, not for a transcription, so choosing
        # fs.list or fs.search or even web.search is a real decision made in a
        # usable shape -- which is the capability being measured.
        got = qualify(_scripted(['{"tool": "fs.search", "arguments": {"pattern": "x"}}']))
        self.assertTrue(got["qualified"], got)

    def test_a_tool_that_was_never_offered_does_not_qualify(self):
        got = qualify(_scripted(['{"tool": "shell.exec", "arguments": {"cmd": "ls"}}']))
        self.assertFalse(got["qualified"])
        self.assertIn("shell.exec", got["reason"])

    def test_the_bodys_native_dialect_qualifies(self):
        got = qualify(_scripted(['<tool_call>\n{"function": "fs.list", '
                                 '"arguments": {"path": "."}}}\n</tool_call>']))
        self.assertTrue(got["qualified"], got)

    def test_a_body_that_raises_does_not_qualify(self):
        def boom(_c):
            raise RuntimeError("resident died")
        got = qualify(boom)
        self.assertFalse(got["qualified"])
        self.assertIn("resident died", got["reason"])


class TestTruncatedSearchIsNotEvidenceOfAbsence(unittest.TestCase):
    def test_a_truncated_thin_search_escalates(self):
        # Measured: fs.search for a real symbol returned 2 hits, both in
        # .hcli/receipts/*.json, files_seen 5001, truncated True -- the budget
        # was spent on generated data before reaching hcli/*.py, and the model
        # reported that nothing in the project called it.
        thin = _Result(value={"matches": [{"path": "x"}], "truncated": True,
                              "files_seen": 5001})
        self.assertTrue(_search_needs_escalation(thin))

    def test_a_complete_search_does_not_escalate(self):
        full = _Result(value={"matches": [{"path": "x"}], "truncated": False})
        self.assertFalse(_search_needs_escalation(full))

    def test_a_truncated_but_rich_search_does_not_escalate(self):
        rich = _Result(value={"matches": [{"path": str(i)} for i in range(40)],
                              "truncated": True})
        self.assertFalse(_search_needs_escalation(rich))

    def test_filename_search_keeps_receipts_in_the_fallback_scope(self):
        from hcli.chat_tools import _search_fallback_glob
        self.assertEqual(
            _search_fallback_glob({"pattern": "KIMI_P0_OPERATIONAL.json"}),
            "*.json")
        self.assertEqual(
            _search_fallback_glob({"path": "docs/HCLI_WEB.md"}),
            "*.md")
        self.assertEqual(_search_fallback_glob({"pattern": "daemon_lease"}),
                         "*.py")

    def test_zero_result_prose_phrase_falls_back_to_source_tokens(self):
        from hcli.chat_tools import escalate_search

        class PhraseRegistry:
            def __init__(self):
                self.calls = []

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                matches = ([{"path": "hcli/serve.py", "line": 1}]
                           if arguments["pattern"] == "session" else [])
                return _Result(value={
                    "pattern": arguments["pattern"],
                    "matches": matches,
                    "truncated": False,
                })

        registry = PhraseRegistry()
        initial = _Result(value={
            "pattern": "session state preservation",
            "matches": [],
            "truncated": False,
        })
        result = escalate_search(
            registry, {"pattern": "session state preservation"}, initial
        )
        self.assertEqual(result.value["pattern"], "session")
        self.assertEqual(registry.calls[0][1]["glob"], "*.py")
        self.assertEqual(registry.calls[0][1]["max_per_file"], 2)

    def test_phrase_fallback_prefers_active_source_over_archaeology(self):
        from hcli.chat_tools import escalate_search

        class RankedRegistry:
            def invoke(self, _name, arguments):
                token = arguments["pattern"]
                paths = {
                    "globally": [
                        "research/hawking-experiments/old/fixture/config.json"],
                    "installed": ["hcli/cli.py"],
                }.get(token, [])
                return _Result(value={
                    "pattern": token,
                    "matches": [{"path": path, "line": 1} for path in paths],
                    "truncated": False,
                })

        initial = _Result(value={"matches": [], "truncated": False})
        result = escalate_search(
            RankedRegistry(),
            {"pattern": "globally installed HCLI must"},
            initial,
        )
        self.assertEqual(result.value["pattern"], "installed")
        self.assertEqual(result.value["matches"][0]["path"], "hcli/cli.py")

    def test_search_can_preserve_file_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "a.py").write_text("needle\n" * 10, encoding="utf-8")
            (root / "b.py").write_text("needle\n" * 10, encoding="utf-8")
            result = build_registry(tmp, tmp).invoke("fs.search", {
                "pattern": "needle",
                "glob": "*.py",
                "max_results": 20,
                "max_per_file": 2,
            })
            self.assertTrue(result.ok, result.error)
            paths = [row["path"] for row in result.value["matches"]]
            self.assertEqual(paths.count(str(root / "a.py")), 2)
            self.assertEqual(paths.count(str(root / "b.py")), 2)


class TestBoundedPathFeedback(unittest.TestCase):
    def test_root_escape_explains_the_repository_relative_retry(self):
        from hcli.chat_tools import _render
        result = _Result(error="PermissionError: path is outside the AgentOS read roots: /",
                         value=None)
        result.ok = False
        text = _render(result, "fs.list")
        self.assertIn('path "."', text)
        self.assertIn("repository-relative", text)

    def test_receipt_escape_explains_the_receipt_namespace(self):
        from hcli.chat_tools import _render
        result = _Result(error="PermissionError: receipt path must be under receipts or .hcli state",
                         value=None)
        result.ok = False
        text = _render(result, "receipt.read")
        self.assertIn("receipts/", text)
        self.assertIn("fs.search", text)

    def test_missing_latest_receipt_explains_discovery(self):
        from hcli.chat_tools import _render
        result = _Result(error="FileNotFoundError: /repo/receipts/latest",
                         value=None)
        result.ok = False
        text = _render(result, "receipt.read")
        self.assertIn("no magic latest", text)
        self.assertIn("glob *.json", text)

    def test_long_receipt_projects_exact_decision_fields_before_the_body(self):
        from hcli.chat_tools import _render
        result = _Result(value={
            "path": "/repo/receipts/example.json",
            "document": {
                "detail": "x" * 4000,
                "negative_science_preflight": {
                    "status": "NEGATIVE_SCIENCE_REFUSED",
                    "scar": {
                        "claim_refuted": "the tested causal claim",
                        "failure_mechanism": "matched controls were not beaten",
                    },
                },
                "status": "NEGATIVE_SCIENCE_REFUSED",
            },
        })
        text = _render(result, "receipt.read")
        self.assertIn("receipt decision fields (exact projection)", text)
        self.assertIn('"status": "NEGATIVE_SCIENCE_REFUSED"', text[:2400])
        self.assertIn(
            '"negative_science_preflight.scar.failure_mechanism": '
            '"matched controls were not beaten"',
            text[:2400],
        )
        self.assertIn("complete receipt follows", text)

    def test_pathological_receipt_is_structurally_compacted_not_prefix_cut(self):
        from hcli.chat_tools import _render
        from hcli.paste_cache import PasteCache

        with tempfile.TemporaryDirectory() as tmp:
            cache = PasteCache(tmp)
            result = _Result(value={
                "path": "/repo/receipts/pathological.json",
                "sha256": "a" * 64,
                "bytes": 100_000,
                "document": {
                    "payload": "HAWKING_RECIPIENT" * 5000,
                    "measurement": {"steady_decode_tps": 99.4},
                    "status": "PASS",
                    "accepted": True,
                    "written_at": "2026-09-10T23:59:00Z",
                },
            })
            text = _render(result, "receipt.read", cache=cache)

        self.assertIn("observation structural summary", text)
        self.assertIn('"status": "PASS"', text)
        self.assertIn('"accepted": true', text)
        self.assertIn('"measurement.steady_decode_tps": 99.4', text)
        self.assertIn("largest_line_chars", text)
        self.assertIn("complete result is kept at paste_", text)
        self.assertLess(text.count("HAWKING_RECIPIENT"), 2)

    def test_large_unknown_observation_exposes_shape_hash_and_retrieval(self):
        from hcli.chat_tools import _render
        from hcli.paste_cache import PasteCache

        with tempfile.TemporaryDirectory() as tmp:
            cache = PasteCache(tmp)
            result = _Result(value={
                "unknown": [{"blob": "z" * 5000} for _ in range(100)],
            })
            text = _render(result, "fs.read", cache=cache)

        self.assertIn('"compacted": true', text)
        self.assertIn('"largest_array_items": 100', text)
        self.assertIn('"sha256":', text)
        self.assertIn("observation.expand", text)

    def test_large_file_selects_headings_and_signal_lines_across_content(self):
        from hcli.chat_tools import _render
        from hcli.paste_cache import PasteCache

        content = "# Hawking\nidentity line\n" + ("ordinary\n" * 800)
        content += "## Runtime\nstatus: READY\n"
        with tempfile.TemporaryDirectory() as tmp:
            text = _render(_Result(value={
                "path": "/repo/README.md", "bytes": len(content),
                "shown_bytes": len(content), "truncated": False,
                "sha256": "b" * 64, "content": content,
            }), "fs.read", cache=PasteCache(tmp))
        self.assertIn('"text": "# Hawking"', text)
        self.assertIn('"text": "## Runtime"', text)
        self.assertIn('"text": "status: READY"', text)
        self.assertIn("complete result is kept at paste_", text)

    def test_large_listing_keeps_directories_and_bounded_file_names(self):
        from hcli.chat_tools import _render
        from hcli.paste_cache import PasteCache

        files = [{"path": f"f{i:03}.json", "filename": f"f{i:03}.json",
                  "type": "file", "kind": "file", "size": i, "bytes": i}
                 for i in range(100)]
        value = {"root": "/repo/receipts", "glob": "*.json", "files": files,
                 "directories": [{"path": "future", "filename": "future",
                                    "type": "directory", "kind": "directory",
                                    "size": None}],
                 "truncated": False, "directories_seen": 1}
        with tempfile.TemporaryDirectory() as tmp:
            text = _render(_Result(value=value), "fs.list", cache=PasteCache(tmp))
        self.assertIn('"path": "future"', text)
        self.assertIn('"path": "f000.json"', text)
        self.assertIn('"path": "f099.json"', text)
        self.assertNotIn('"path": "f050.json"', text)

    def test_observation_expansion_never_creates_a_pointer_to_a_pointer(self):
        from hcli.chat_tools import _render

        class Cache:
            def __init__(self):
                self.stored = []

            def store(self, value):
                self.stored.append(value)
                raise AssertionError("an expansion wrapper must not be cached")

        cache = Cache()
        result = _Result(value={
            "id": "paste_20260910_000000_deadbeef",
            "start": 1,
            "end": 20,
            "text": "x" * 5000,
        })
        rendered = _render(result, "observation.expand", cache=cache)
        self.assertEqual(cache.stored, [])
        self.assertIn("original observation", rendered)
        self.assertIn('"start": 21', rendered)


class TestProvenanceSurvives(unittest.TestCase):
    def test_source_urls_and_time_are_lifted_from_the_value(self):
        # web.search reports these inside its value, not on the envelope, so
        # lifting only result.provenance discarded exactly what a web answer
        # must be able to cite.
        result = _Result(value={
            "provider": "bing-html", "retrieved_at": 123, "confidence": "x",
            "results": [{"url": "https://a.example"}, {"url": "https://b.example"}]})
        got = provenance_of(result)
        self.assertEqual(got["provider"], "bing-html")
        self.assertEqual(got["retrieved_at"], 123)
        self.assertEqual(got["sources"], ["https://a.example", "https://b.example"])


if __name__ == "__main__":
    unittest.main()


class TestAnEmptyFirstReplyIsWarmUpNotIncapacity(unittest.TestCase):
    """backend.ready() said READY and the first completion returned "".

    Measured on sealed-3.14: the probe was the first real call after spawn, came
    back empty, and the flagship resident was marked tool-INCAPABLE while
    emitting the exact action correctly seconds later. A capability gate is the
    worst possible place for this repo's readiness scar to land, because the
    failure is SILENT -- a body simply never gets offered what it can do.
    """

    def test_an_empty_first_reply_is_retried(self):
        got = qualify(_scripted(["", '{"tool": "fs.list", "arguments": {"path": "."}}']))
        self.assertTrue(got["qualified"], got)
        self.assertEqual(got["attempts"], 2)

    def test_two_empty_replies_are_undetermined_not_refused(self):
        got = qualify(_scripted(["", "   "]))
        self.assertFalse(got["qualified"])
        self.assertIn("undetermined", got["reason"])
        self.assertIn("not generating", got["reason"])

    def test_a_first_reply_that_works_costs_one_attempt(self):
        got = qualify(_scripted(['{"tool": "fs.list", "arguments": {"path": "."}}']))
        self.assertEqual(got["attempts"], 1)

    def test_real_prose_is_still_a_refusal_not_a_retry(self):
        got = qualify(_scripted(["I would list the directory."]))
        self.assertFalse(got["qualified"])
        self.assertIn("did not emit", got["reason"])
        self.assertEqual(got["attempts"], 1, "prose was retried as though it were empty")


class TestOneSystemMessage(unittest.TestCase):
    """The native template refuses a second system turn, and refuses a late one.

    Measured: "artifact chat-template rendering failed: System message must be
    at the beginning." Prepending the tool contract as its own system message on
    top of the repo context 502'd sealed-3.14 -- while the MLX backend tolerated
    it, so every browser acceptance test passed against a body that was not the
    one this campaign ships.
    """

    def test_the_contract_merges_into_an_existing_system_message(self):
        from hcli.chat_tools import prepend_system
        out = prepend_system([{"role": "system", "content": "REPO"},
                              {"role": "user", "content": "hi"}], "TOOLS")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "system")
        self.assertTrue(out[0]["content"].startswith("TOOLS"))
        self.assertIn("REPO", out[0]["content"])
        self.assertEqual([m["role"] for m in out], ["system", "user"])

    def test_only_one_system_message_ever_results(self):
        from hcli.chat_tools import prepend_system
        for messages in ([{"role": "user", "content": "hi"}],
                         [{"role": "system", "content": "a"},
                          {"role": "user", "content": "b"}]):
            out = prepend_system(messages, "TOOLS")
            self.assertEqual(sum(1 for m in out if m["role"] == "system"), 1)
            self.assertEqual(out[0]["role"], "system")

    def test_the_tool_contract_leads_so_the_prefix_stays_invariant(self):
        from hcli.chat_tools import prepend_system
        first = prepend_system([{"role": "system", "content": "REPO\nfile A"}], "TOOLS")
        second = prepend_system([{"role": "system", "content": "REPO\nfile B"}], "TOOLS")
        shared = first[0]["content"][:len("TOOLS\n\nREPO")]
        self.assertTrue(second[0]["content"].startswith(shared))


class TestTheBodysOwnDialect(unittest.TestCase):
    """A protocol that only accepts its own spelling is provincial, not strict.

    Captured verbatim from sealed-3.14 under real use, after it PASSED a
    qualification probe that asked it to copy an object:

        <tool_call>
        {"function": "fs.list", "arguments": {"path": "hcli"}}}
        </tool_call>

    A correct decision, in Qwen's native convention, with a stray brace. The
    first parser wanted `"tool"` and read this as prose -- so the raw markup was
    handed to the user as the answer, and a real tool decision was thrown away.
    """

    OBSERVED = ('<tool_call>\n{"function": "fs.list", "arguments": '
                '{"path": "hcli"}}}\n</tool_call>')

    def test_the_exact_reply_sealed_3_14_produced_is_understood(self):
        self.assertEqual(parse_call(self.OBSERVED), ("fs.list", {"path": "hcli"}))

    def test_a_stray_closing_brace_is_tolerated(self):
        self.assertEqual(parse_call('{"tool": "fs.list", "arguments": {"path": "."}}}'),
                         ("fs.list", {"path": "."}))

    def test_the_nested_qwen_function_form_is_understood(self):
        got = parse_call('{"function": {"name": "web.search", '
                         '"arguments": {"query": "x"}}}')
        self.assertEqual(got, ("web.search", {"query": "x"}))

    def test_name_is_accepted_as_well_as_tool(self):
        self.assertEqual(parse_call('{"name": "fs.read", "arguments": {"path": "a"}}'),
                         ("fs.read", {"path": "a"}))

    def test_prose_is_still_not_a_call(self):
        self.assertIsNone(parse_call("I will list the directory for you."))
        self.assertIsNone(parse_call("The function fs.list would help here."))


class TestTheXmlDialect(unittest.TestCase):
    """sealed-3.14's decision form, captured verbatim.

    Asked a QUESTION (not asked to copy an object) it replied:

        I'll list the files in the current directory using a tool.
        <tool_call>
        <function=shell>
        <parameter=command>
        ls -a
        </parameter>
        </function>
        </tool_call>

    Parsing this is not endorsement -- `shell` is not offered and is a mutation
    surface, so the action is still refused. It is how the refusal gets to name
    WHICH tool the body reached for, instead of the useless "did not emit a
    parseable action".
    """

    OBSERVED = ("I'll list the files in the current directory using a tool.\n\n"
                "<tool_call>\n<function=shell>\n<parameter=command>\nls -a\n"
                "</parameter>\n</function>\n</tool_call>")

    def test_the_xml_form_is_parsed(self):
        self.assertEqual(parse_call(self.OBSERVED), ("shell", {"command": "ls -a"}))

    def test_the_opener_missing_its_leading_bracket_still_parses(self):
        # SCAR: under greedy-argmax the body reliably drops the `<` of the
        # opener, emitting `=function=fs.read>` while every other tag stays
        # intact (od -c confirmed in .hcli/selfdev evidence, 3x in one reply).
        # That one byte made every tool call unparseable and stalled the
        # self-development loop at zero executed actions.
        observed = ("<tool_call>\n=function=fs.read>\n<parameter=path>\n"
                    "H-MANIFESTO.md\n</parameter>\n<parameter=start_line>\n1\n"
                    "</parameter>\n<parameter=end_line>\n200\n</parameter>\n"
                    "</function>\n</tool_call>")
        self.assertEqual(
            parse_call(observed),
            ("fs.read", {"path": "H-MANIFESTO.md",
                         "start_line": "1", "end_line": "200"}))

    def test_the_opener_missing_the_whole_function_token_still_parses(self):
        # SCAR: the drift did not stop at the leading `<` -- three cycles
        # after that fix deployed (.hcli/selfdev/evidence/cycle-0012.txt,
        # cycle-0013.txt), EVERY opener in EVERY call had dropped `function`
        # entirely too, down to bare `=fs.read>`. <tool_call> and
        # <parameter=...> stayed intact throughout; only the opener kept
        # eroding. Cycle 13's entire reply was exactly this one call, so
        # parse_calls returning [] wasted the whole turn -- zero tool
        # executions, same as the original leading-bracket drop.
        observed = ("<tool_call>\n=fs.read>\n<parameter=path>\n"
                    "hcli/engine.py\n</parameter>\n<parameter=start_line>\n7560\n"
                    "</parameter>\n<parameter=end_line>\n7620\n</parameter>\n"
                    "</function>\n</tool_call>")
        self.assertEqual(
            parse_call(observed),
            ("fs.read", {"path": "hcli/engine.py",
                         "start_line": "7560", "end_line": "7620"}))

    def test_a_doubled_equals_opener_still_parses(self):
        # SCAR: cycle-0014.txt's third batched call drifted one step further
        # still: `==fs.read>` (two `=`, no `function`).
        observed = ("<tool_call>\n==fs.read>\n<parameter=path>\n"
                    "hcli/tests/test_splicing_ops_are_not_falsely_rejected.py\n"
                    "</parameter>\n</function>\n</tool_call>")
        self.assertEqual(
            parse_call(observed),
            ("fs.read", {"path": "hcli/tests/test_splicing_ops_are_not_falsely_rejected.py"}))

    def test_an_unoffered_tool_from_it_is_still_refused(self):
        got = qualify(_scripted([self.OBSERVED]))
        self.assertFalse(got["qualified"])
        self.assertIn("shell", got["reason"],
                      "the verdict does not say which tool it reached for")

    def test_an_offered_tool_in_the_xml_form_qualifies(self):
        reply = "<tool_call><function=fs.list><parameter=path>.</parameter></function></tool_call>"
        self.assertTrue(qualify(_scripted([reply]))["qualified"])


class TestTheXmlDialectCannotExpressNumbers(unittest.TestCase):
    """<parameter=max_results>200</parameter> is text, and the schema wants an int.

    Measured on sealed-3.14: it called fs.list with max_results "200" twice and
    got "$.max_results: expected integer, got string" both times -- punished for
    following the call format its own artifact template specifies. Converting
    "200" to 200 is mechanics, not judgement, so the harness owns it.
    """

    def setUp(self):
        from hcli.chat_tools import build_registry
        self.registry = build_registry(".", ".")

    def test_a_string_integer_becomes_an_integer(self):
        from hcli.chat_tools import coerce_arguments
        got = coerce_arguments(self.registry, "fs.list",
                               {"path": "hcli", "max_results": "200"})
        self.assertEqual(got["max_results"], 200)
        self.assertEqual(got["path"], "hcli", "a string field was mangled")

    def test_a_value_that_cannot_convert_is_left_for_the_tool_to_refuse(self):
        from hcli.chat_tools import coerce_arguments
        got = coerce_arguments(self.registry, "fs.list",
                               {"path": "hcli", "max_results": "many"})
        self.assertEqual(got["max_results"], "many")

    def test_an_unknown_tool_passes_arguments_through(self):
        from hcli.chat_tools import coerce_arguments
        args = {"a": "1"}
        self.assertEqual(coerce_arguments(self.registry, "nope.nope", args), args)

    def test_the_real_call_sealed_made_now_succeeds(self):
        from hcli.chat_tools import coerce_arguments
        args = coerce_arguments(self.registry, "fs.list",
                                {"path": "hcli", "max_results": "200"})
        result = self.registry.invoke("fs.list", args)
        self.assertTrue(result.ok, getattr(result, "error", None))


class TestObservationsAreDiskFirst(unittest.TestCase):
    """A big result must be EVICTED from context, never destroyed.

    The first version truncated at 3000 characters and dropped the rest, so a
    test log or page the model needed ten lines further down was simply gone --
    and it could not ask, because nothing had kept it. PasteCache is a
    content-addressed store built for exactly this whose store() had no caller
    outside its own tests.

    S035: "HISTORY LIVES ON DISK. CONTEXT IS THE CURRENT WORKING SET." A summary
    that cannot be expanded is forgetting.
    """

    def setUp(self):
        import tempfile
        from hcli.paste_cache import PasteCache
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = PasteCache(self._tmp.name)

    def _big(self):
        body = "\n".join(
            f"line {i}: something about deltanet recurrent state and prefill"
            for i in range(1, 900))
        return body + "\nDECISIVE: chunk 16 rejected, dispatches tripled\n"

    def _observe(self, text):
        from hcli.chat_tools import _render

        class _R:
            ok, error, provenance, failure_class = True, None, {}, None
            def __init__(self, v): self.value = v
        return _render(_R(text), "fs.read", None, self.cache)

    def test_the_context_copy_is_small_and_names_a_handle(self):
        from hcli.chat_tools import RESULT_CHARS
        big = self._big()
        obs = self._observe(big)
        self.assertLess(len(obs), RESULT_CHARS + 600)
        self.assertLess(len(obs), len(big) / 5,
                        f"{len(big)} chars became {len(obs)} -- not much eviction")
        self.assertIn("observation.expand", obs,
                      "the model was given no way to ask for the rest")

    def test_the_line_truncation_would_have_destroyed_is_retrievable(self):
        from hcli.chat_tools import RESULT_CHARS, run_local_tool
        big = self._big()
        obs = self._observe(big)
        self.assertNotIn("DECISIVE", obs[:RESULT_CHARS],
                         "fixture is wrong: the decisive line must fall past the cut")
        paste_id = obs.split("kept at ")[1].split(" ")[0]
        got = run_local_tool("observation.expand",
                             {"id": paste_id, "query": "DECISIVE"}, cache=self.cache)
        self.assertTrue(got.ok, got.error)
        self.assertTrue(got.value["matches"], "the evicted evidence is unrecoverable")
        self.assertIn("chunk 16", got.value["matches"][0]["text"])

    def test_a_range_can_be_expanded(self):
        from hcli.chat_tools import run_local_tool
        obs = self._observe(self._big())
        paste_id = obs.split("kept at ")[1].split(" ")[0]
        got = run_local_tool("observation.expand",
                             {"id": paste_id, "start": 1, "end": 3}, cache=self.cache)
        self.assertTrue(got.ok, got.error)
        self.assertIn("line 1:", got.value["text"])
        self.assertGreater(got.value["total_lines"], 3)

    def test_an_out_of_range_expansion_is_not_a_successful_empty_observation(self):
        from hcli.chat_tools import run_local_tool
        obs = self._observe(self._big())
        paste_id = obs.split("kept at ")[1].split(" ")[0]
        got = run_local_tool(
            "observation.expand",
            {"id": paste_id, "start": 3000, "end": 4000},
            cache=self.cache,
        )
        self.assertFalse(got.ok)
        self.assertIn("out of range", got.error)
        self.assertIn("1..900", got.error)

    def test_a_small_result_is_not_sent_to_disk_at_all(self):
        obs = self._observe("just a short answer")
        self.assertIn("just a short answer", obs)
        self.assertNotIn("observation.expand", obs)

    def test_an_unknown_handle_refuses_clearly(self):
        from hcli.chat_tools import run_local_tool
        got = run_local_tool("observation.expand", {"id": "nope"}, cache=self.cache)
        self.assertFalse(got.ok)
        self.assertIn("paste id", got.error)

    def test_expand_without_an_id_says_what_it_needs(self):
        from hcli.chat_tools import run_local_tool
        got = run_local_tool("observation.expand", {}, cache=self.cache)
        self.assertFalse(got.ok)
        self.assertIn('"id"', got.error)

    def test_with_no_store_the_old_behaviour_still_works(self):
        # A session without a cache must degrade to truncation, not crash.
        from hcli.chat_tools import _render

        class _R:
            ok, error, provenance, failure_class = True, None, {}, None
            def __init__(self, v): self.value = v
        obs = _render(_R(self._big()), "fs.read", None, None)
        self.assertIn("truncated", obs)


class TestRecallIsADoor(unittest.TestCase):
    """context.recall was a registered tool the chat could not reach.

    It is now offered, and it is served by the REGISTRY -- not by a second
    implementation here. A door with two owners is how a hand-written shape
    ('limit') came to override the real schema (max_results/max_chars).
    """

    def test_the_registry_owns_recall_and_answers_it(self):
        registry = build_registry(".", ".")
        spec = registry.get("context.recall")
        self.assertIsNotNone(spec, "the registry no longer offers context.recall")
        result = registry.invoke("context.recall", {"focus": "prefill"})
        self.assertTrue(result.ok, getattr(result, "error", None))

    def test_recall_is_not_also_served_locally(self):
        from hcli.chat_tools import LOCAL_SHAPES, LOCAL_TOOLS
        self.assertNotIn("context.recall", LOCAL_TOOLS)
        self.assertNotIn("context.recall", LOCAL_SHAPES)

    def test_the_advertised_shape_comes_from_the_real_schema(self):
        # The defect: the menu said "limit"; the tool takes max_results/max_chars.
        registry = build_registry(".", ".")
        shape = argument_shape(registry, "context.recall")
        self.assertIn('"focus"', shape)
        self.assertNotIn("limit", shape,
                         "the menu advertises a field the tool rejects")


class TestDebugIsAChatDoor(unittest.TestCase):
    """The model can turn a stored failure into a localized fact.

    Recon confirmed no reproduce/localize tool existed on any surface. This is
    the disk-first loop closed: a large failure goes to a handle, and
    debug.diagnose reads the handle and returns the cause -- the deepest project
    frame and the assertion -- not the dump.
    """

    def setUp(self):
        import tempfile
        from hcli.paste_cache import PasteCache
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = PasteCache(self._tmp.name)

    def test_a_stored_failure_is_localized(self):
        from hcli.chat_tools import run_local_tool
        fail = ("test_x.py:6: in test_value\n    assert compute() == 5\n"
                "E   AssertionError: assert 3 == 5\n") + "noise\n" * 200
        ref = self.cache.store(fail)
        got = run_local_tool("debug.diagnose",
                             {"id": ref.id, "command": "pytest", "exit_code": 1},
                             cache=self.cache)
        self.assertTrue(got.ok, got.error)
        self.assertTrue(got.value["reproduced"])
        self.assertEqual(got.value["kind"], "AssertionError")

    def test_a_missing_id_says_what_it_needs(self):
        from hcli.chat_tools import run_local_tool
        got = run_local_tool("debug.diagnose", {"command": "x"}, cache=self.cache)
        self.assertFalse(got.ok)
        self.assertIn('"id"', got.error)

    def test_an_unknown_handle_refuses(self):
        from hcli.chat_tools import run_local_tool
        got = run_local_tool("debug.diagnose", {"id": "nope", "command": "x"},
                             cache=self.cache)
        self.assertFalse(got.ok)


class TestCapabilityDoorsAreServedLocally(unittest.TestCase):
    """reverse.identify and forensics.snapshot are HCLI-served, one owner each."""

    def setUp(self):
        self.registry = build_registry(".", ".")

    def test_they_are_local_not_registry(self):
        from hcli.chat_tools import LOCAL_TOOLS
        for name in ("reverse.identify", "forensics.snapshot", "debug.diagnose"):
            self.assertIn(name, LOCAL_TOOLS)
            self.assertIsNone(self.registry.get(name),
                              f"{name} is claimed by both owners")

    def test_reverse_identify_reads_a_file(self):
        import tempfile
        from pathlib import Path
        from hcli.chat_tools import run_local_tool
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "b").write_bytes(b"\x7fELF\x02")
            got = run_local_tool("reverse.identify", {"path": str(Path(tmp) / "b")})
            self.assertTrue(got.ok)
            self.assertEqual(got.value["kind"], "ELF executable")

    def test_reverse_identify_needs_a_path(self):
        from hcli.chat_tools import run_local_tool
        got = run_local_tool("reverse.identify", {})
        self.assertFalse(got.ok)
        self.assertIn("path", got.error)
