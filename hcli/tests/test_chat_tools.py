"""The chat's tool loop: what it must never do.

The failures guarded here are the ones that produce a confident wrong answer
rather than an error:

  * claiming a tool ran when the body never emitted a usable action
  * offering a menu whose argument shapes disagree with the tools' schemas
  * reporting "no matches" from a search that stopped before it reached the code
  * losing the source URLs behind a web answer
"""
from __future__ import annotations

import unittest

from hcli.chat_tools import (
    CHAT_TOOLS,
    argument_shape,
    build_registry,
    parse_call,
    provenance_of,
    qualify,
    run_with_tools,
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


class TestParsing(unittest.TestCase):
    def test_a_bare_object(self):
        self.assertEqual(parse_call('{"tool": "fs.list", "arguments": {"path": "."}}'),
                         ("fs.list", {"path": "."}))

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


class TestTheLoop(unittest.TestCase):
    def test_an_answer_with_no_action_returns_immediately(self):
        registry = _Registry()
        text, trace = run_with_tools(_scripted(["Paris."]), [], registry)
        self.assertEqual(text, "Paris.")
        self.assertEqual(trace, [])
        self.assertEqual(registry.calls, [], "a tool ran for a question that needed none")

    def test_a_tool_result_comes_back_and_the_answer_follows(self):
        registry = _Registry({"fs.list": _Result(value={"files": ["a.py"]})})
        text, trace = run_with_tools(
            _scripted(['{"tool": "fs.list", "arguments": {"path": "."}}',
                       "There is one file: a.py"]), [], registry)
        self.assertEqual(text, "There is one file: a.py")
        self.assertEqual([t["tool"] for t in trace], ["fs.list"])
        self.assertTrue(trace[0]["ok"])

    def test_an_unoffered_tool_is_told_what_IS_available(self):
        registry = _Registry()
        text, trace = run_with_tools(
            _scripted(['{"tool": "shell.exec", "arguments": {"cmd": "rm -rf /"}}',
                       "I cannot do that."]), [], registry)
        self.assertEqual(registry.calls, [], "an unoffered tool was invoked")
        self.assertFalse(trace[0]["ok"])
        self.assertEqual(text, "I cannot do that.")

    def test_the_budget_is_bounded_and_ends_with_an_answer(self):
        registry = _Registry({"fs.list": _Result(value={})})
        # DISTINCT calls, so budget exhaustion (not the repeat guard) is what ends it.
        calls = ['{"tool": "fs.list", "arguments": {"path": "%s"}}' % p
                 for p in ("a", "b", "c")]
        text, trace = run_with_tools(
            _scripted(calls + ["I ran out of budget."]), [], registry, max_calls=3)
        self.assertEqual(len(trace), 3)
        self.assertEqual(text, "I ran out of budget.")

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
        # The real bug was UNBOUNDED growth turn over turn -- prove it
        # plateaus rather than merely staying under one arbitrary ceiling.
        self.assertEqual(seen_sizes[-1], seen_sizes[-2],
                         f"size kept growing instead of reaching steady state: {seen_sizes}")

    def test_with_no_budget_given_behaviour_is_unchanged(self):
        # The default (no budget) must stay exactly what every other test in
        # this file already assumes -- opt-in, not a silent behavior change.
        registry = _Registry({"fs.read": _Result(value={"text": "x" * 4000})})
        text, trace = run_with_tools(
            _scripted(['{"tool": "fs.read", "arguments": {"path": "a"}}',
                       "Done."]), [], registry)
        self.assertEqual(text, "Done.")
        self.assertEqual(len(trace), 1)

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
