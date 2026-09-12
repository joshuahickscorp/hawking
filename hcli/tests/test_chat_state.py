"""Session identity, durable plans, and reference resolution.

The browser path was intellectually stateless: Open WebUI held the conversation
and resent it whole, so HCLI could not remember a plan or resolve "apply it".
These pin the three properties that make a long conversation workable:

  * the same conversation maps to the same session without client cooperation
  * a plan outlives the context window and carries its step, not its prose
  * "it" resolves when there is one answer and ASKS when there are several
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.chat_state import (ChatSession, Plan, record_tool_trace, session_key,
                             working_set)


def _msgs(first="Write me a plan for context gravity"):
    return [{"role": "user", "content": first},
            {"role": "assistant", "content": "here it is"},
            {"role": "user", "content": "thanks"}]


class TestIdentity(unittest.TestCase):
    def test_the_same_conversation_gets_the_same_key(self):
        with TemporaryDirectory() as tmp:
            a = session_key(_msgs(), tmp)
            b = session_key(_msgs() + [{"role": "user", "content": "more"}], tmp)
            self.assertEqual(a, b, "the key changed as the conversation grew")

    def test_a_different_conversation_gets_a_different_key(self):
        with TemporaryDirectory() as tmp:
            self.assertNotEqual(session_key(_msgs("A"), tmp),
                                session_key(_msgs("B"), tmp))

    def test_the_same_question_in_two_repos_is_two_sessions(self):
        with TemporaryDirectory() as one, TemporaryDirectory() as two:
            self.assertNotEqual(session_key(_msgs(), one), session_key(_msgs(), two))

    def test_an_explicit_id_wins(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(session_key(_msgs(), tmp, explicit="abc"), "abc")


class TestDurability(unittest.TestCase):
    def test_a_plan_survives_a_new_process(self):
        with TemporaryDirectory() as tmp:
            first = ChatSession.load(tmp, "s1")
            plan = first.add_plan("Context gravity", "long prose here",
                                  steps=["evidence store", "compaction", "resume"])
            first.objective = "make long runs survive"
            first.save()

            # a new object, as a new process would build it
            second = ChatSession.load(tmp, "s1")
            self.assertEqual(second.active_plan, plan.id)
            got = second.plan()
            self.assertIsNotNone(got)
            self.assertEqual(got.title, "Context gravity")
            self.assertEqual(got.steps[1], "compaction")
            self.assertEqual(second.objective, "make long runs survive")

    def test_a_corrupt_record_does_not_end_the_conversation(self):
        with TemporaryDirectory() as tmp:
            path = ChatSession.path_for(tmp, "s1")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json", encoding="utf-8")
            session = ChatSession.load(tmp, "s1")
            self.assertEqual(session.id, "s1")
            self.assertEqual(session.plans, {})

    def test_the_compiled_plan_carries_position_not_prose(self):
        plan = Plan(id="P1", title="t", body="x" * 5000,
                    steps=["a", "b", "c"], current_step=1)
        compiled = plan.compiled()
        self.assertLess(len(compiled), 400, "the working set carried the prose")
        self.assertIn("STEP 2/3", compiled)
        self.assertIn("[x] a", compiled)
        self.assertIn("[>] b", compiled)


class TestReferenceResolution(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.session = ChatSession.load(self._tmp.name, "s1")

    def test_it_resolves_to_the_only_plan(self):
        self.session.add_plan("Context gravity", "body")
        got = self.session.resolve("okay, apply it")
        self.assertTrue(got.ok)
        self.assertEqual(got.plan.title, "Context gravity")

    def test_it_resolves_to_the_active_plan_when_several_exist(self):
        self.session.add_plan("First", "b1")
        second = self.session.add_plan("Second", "b2")
        got = self.session.resolve("apply it")
        self.assertEqual(got.plan_id, second.id, "resolved to a stale plan")

    def test_an_explicit_id_selects_that_plan(self):
        first = self.session.add_plan("First", "b1")
        self.session.add_plan("Second", "b2")
        got = self.session.resolve(f"apply {first.id.lower()} please")
        self.assertEqual(got.plan_id, first.id)

    def test_a_title_reference_selects_that_plan(self):
        first = self.session.add_plan("context gravity", "b1")
        self.session.add_plan("builder qualification", "b2")
        got = self.session.resolve("go ahead with the context gravity one")
        self.assertEqual(got.plan_id, first.id)

    def test_two_explicit_ids_are_asked_about_not_guessed(self):
        a = self.session.add_plan("First", "b1")
        b = self.session.add_plan("Second", "b2")
        got = self.session.resolve(f"merge {a.id.lower()} and {b.id.lower()}")
        self.assertFalse(got.ok)
        self.assertIn(a.id, got.question())
        self.assertIn(b.id, got.question())

    def test_no_plans_at_all_resolves_to_nothing(self):
        got = self.session.resolve("apply it")
        self.assertFalse(got.ok)


class TestWorkingSet(unittest.TestCase):
    def test_an_empty_session_contributes_nothing(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(working_set(ChatSession.load(tmp, "s1")), "")

    def test_the_working_set_states_objective_step_and_authority(self):
        with TemporaryDirectory() as tmp:
            session = ChatSession.load(tmp, "s1")
            session.objective = "survive long runs"
            session.authority = "write"
            session.add_plan("CG", "prose", steps=["one", "two"])
            block = working_set(session)
            self.assertIn("survive long runs", block)
            self.assertIn("STEP 1/2", block)
            self.assertIn("write", block)
            self.assertNotIn("prose", block, "the working set carried the plan body")


if __name__ == "__main__":
    unittest.main()


class TestPromotionIsNarrow(unittest.TestCase):
    """Not every sentence deserves permanent state (S035 s32)."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.session = ChatSession.load(self._tmp.name, "s1")

    def test_asking_for_a_plan_creates_one(self):
        from hcli.chat_state import remember_turn
        answer = ("Here is the approach.\n1. Wire the store\n2. Compact\n"
                  "3. Measure\n") + "detail " * 60
        got = remember_turn(
            self.session,
            [{"role": "user", "content": "Write me a plan for context gravity"}],
            answer)
        self.assertIn("plan_created", got)
        plan = self.session.plan()
        self.assertEqual(plan.steps, ["Wire the store", "Compact", "Measure"])
        self.assertEqual(plan.status, "draft")

    def test_ordinary_conversation_creates_nothing(self):
        from hcli.chat_state import remember_turn
        got = remember_turn(self.session,
                            [{"role": "user", "content": "what is a GPU?"}],
                            "A GPU is a processor." * 30)
        self.assertNotIn("plan_created", got)
        self.assertEqual(self.session.plans, {})

    def test_a_short_answer_is_not_promoted_to_a_plan(self):
        from hcli.chat_state import remember_turn
        got = remember_turn(
            self.session,
            [{"role": "user", "content": "write a plan for X"}], "soon.")
        self.assertNotIn("plan_created", got)

    def test_apply_moves_the_plan_to_executing(self):
        from hcli.chat_state import remember_turn
        self.session.add_plan("CG", "body", steps=["a", "b"])
        got = remember_turn(self.session,
                            [{"role": "user", "content": "Okay, apply it."}], "ok")
        self.assertEqual(got.get("plan_executing"), "P1")
        self.assertEqual(self.session.plan().status, "executing")
        self.assertEqual(self.session.objective, "CG")

    def test_apply_with_two_explicit_plans_reports_ambiguity(self):
        from hcli.chat_state import remember_turn
        a = self.session.add_plan("First", "b")
        b = self.session.add_plan("Second", "b")
        got = remember_turn(
            self.session,
            [{"role": "user", "content": f"apply {a.id.lower()} and {b.id.lower()}"}],
            "ok")
        self.assertIn("ambiguous", got)

    def test_a_plan_records_the_commit_it_was_written_against(self):
        from hcli.chat_state import remember_turn
        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=self._tmp.name, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "x"],
                       cwd=self._tmp.name, check=True)
        session = ChatSession.load(self._tmp.name, "s2")
        remember_turn(session,
                      [{"role": "user", "content": "draft a plan for it"}],
                      "1. one\n2. two\n" + "detail " * 60)
        self.assertTrue(session.plan().repo_commit,
                        "a plan with no basis cannot be checked against drift")


class TestApplyIntentIsImperative(unittest.TestCase):
    """This flips a plan to executing, so a mention must not count as an order.

    "tell me about apply" contains the verb and asks for nothing to happen.
    "what would applying it involve?" is a question about the work, not the
    instruction to do it.
    """

    CASES = [
        ("Okay, apply it.", True),
        ("apply p1 and p2", True),
        ("let's apply the plan", True),
        ("go ahead and apply it", True),
        ("implement the second phase", True),
        ("what would applying it involve?", False),
        ("should we apply this?", False),
        ("tell me about apply", False),
        ("I was going to implement something else", False),
        ("", False),
    ]

    def test_every_case(self):
        from hcli.chat_state import asked_to_apply
        wrong = [(text, want, asked_to_apply(text))
                 for text, want in self.CASES if asked_to_apply(text) != want]
        self.assertEqual(wrong, [], f"apply-intent misread: {wrong}")


class TestBrowserToolTrace(unittest.TestCase):
    def test_streaming_trace_and_final_are_durable(self):
        with TemporaryDirectory() as tmp:
            session = ChatSession.load(tmp, "browser-a")
            trace = [{
                "tool": "fs.list",
                "arguments": {"path": "."},
                "ok": True,
                "dispatched": True,
                "observation": "fs.list returned: one.py",
            }]
            path = record_tool_trace(
                session, trace,
                tool_contract={"selected_count": 21, "actual_invocations": 1},
                answer='{"liveness":"PASS"}', resident="KIMI_P0_OPERATIONAL",
            )
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(value["schema"], "hcli.chat.tool-trace.v1")
            self.assertEqual(value["trace"], trace)
            self.assertEqual(value["final"]["text"], '{"liveness":"PASS"}')
            self.assertIn(path, session.evidence)
