"""Write authority is structural, not a sentence in a prompt.

S035 s26: PROPOSED, APPLIED, VALIDATED and ACCEPTED must stay distinguishable,
and a rejected mutation must never be reported as "fixed". S035 s24/s26: the
default browser session is read/research, and builder doors appear only when a
caller grants authority -- so a model that asks to edit in a read-only session
is refused by the MENU, not by persuasion.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.chat_tools import (
    BUILDER_TOOLS,
    builder_menu,
    run_builder_tool,
    run_with_tools,
)
from hcli.engine import Engine
from hcli.workspace import Workspace


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


class _Registry:
    def __init__(self):
        self.calls = []

    def invoke(self, name, arguments=None):
        self.calls.append(name)
        class _R:
            ok, value, error, provenance, failure_class = True, {}, None, {}, None
        return _R()

    def get(self, name):
        return None


def _scripted(replies):
    it = iter(replies)
    return lambda _c: next(it)


class TestAuthorityIsStructural(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        (self.root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.engine = Engine(Workspace(str(self.root)),
                             runtime_provider=lambda: _Pool())

    def test_a_read_only_session_does_not_offer_the_builder_door(self):
        self.assertNotIn("repo.edit", builder_menu(False))
        self.assertIn("repo.edit", builder_menu(True))

    def test_a_read_only_session_refuses_the_edit_at_the_MENU(self):
        # TWO gates exist: the menu, and run_builder_tool refusing a None
        # engine. A mutation test that cannot tell them apart proves neither --
        # granting authority unconditionally still passed this until the
        # refusal reason was asserted, because the second gate caught it.
        registry = _Registry()
        edit = ('{"tool": "repo.edit", "arguments": {"operations": '
                '[{"op": "replace", "path": "mod.py", "old_lines": ["VALUE = 1"], '
                '"new_lines": ["VALUE = 99"]}]}}')
        text, trace = run_with_tools(
            _scripted([edit, "I do not have write access here."]), [], registry)
        self.assertFalse(trace[0]["ok"])
        self.assertEqual(trace[0].get("error"), "not offered to chat",
                         "repo.edit reached the transaction in a read-only "
                         "session -- the MENU gate did not hold")
        self.assertEqual((self.root / "mod.py").read_text(), "VALUE = 1\n",
                         "a read-only session mutated the repository")

    def test_the_engine_gate_is_a_SECOND_independent_refusal(self):
        # Belt and braces, asserted separately so each gate has its own test.
        got = run_builder_tool("repo.edit", {"operations": [
            {"op": "create", "path": "x.py", "new_lines": ["X = 1"]}]},
            engine=None)
        self.assertFalse(got.ok)
        self.assertIn("write authority", got.error)
        self.assertFalse((self.root / "x.py").exists())

    def test_with_authority_the_edit_reaches_the_transaction(self):
        got = run_builder_tool("repo.edit", {"operations": [
            {"op": "replace", "path": "mod.py",
             "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]}]},
            engine=self.engine)
        self.assertTrue(got.ok, got.error)
        self.assertIn(got.value["status"], ("accepted", "unproven"))
        self.assertEqual((self.root / "mod.py").read_text(), "VALUE = 2\n")
        self.assertIn("VALUE = 2", got.value["diff"])

    def test_a_rejected_mutation_is_reported_as_rejected_not_fixed(self):
        proving = self.root / "test_mod.py"
        proving.write_text(
            "from mod import VALUE\n\n\ndef test_v():\n    assert VALUE == 1\n",
            encoding="utf-8")
        got = run_builder_tool("repo.edit", {
            "operations": [{"op": "replace", "path": "mod.py",
                            "old_lines": ["VALUE = 1"],
                            "new_lines": ["VALUE = 999"]}],
            "tests": ["test_mod.py"]}, engine=self.engine)
        self.assertTrue(got.ok, "a rejection is a result to report, not an error")
        self.assertEqual(got.value["status"], "rejected")
        self.assertTrue(got.value["rolled_back"])
        self.assertEqual((self.root / "mod.py").read_text(), "VALUE = 1\n")

    def test_an_edit_with_no_test_is_unproven_not_verified(self):
        got = run_builder_tool("repo.edit", {"operations": [
            {"op": "create", "path": "new.py", "new_lines": ["X = 1"]}]},
            engine=self.engine)
        self.assertEqual(got.value["status"], "unproven")

    def test_a_malformed_edit_gets_the_shape_back(self):
        got = run_builder_tool("repo.edit", {"operations": "not a list"},
                               engine=self.engine)
        self.assertFalse(got.ok)
        self.assertIn("operations", got.error)
        self.assertIn("insert_after", got.error, "the refusal omits the accepted ops")

    def test_there_is_no_shell_door_at_any_authority(self):
        # A typed operation can be refused with a reason; an arbitrary command
        # string cannot. No authority level introduces one.
        for write in (False, True):
            names = " ".join(builder_menu(write))
            self.assertNotIn("shell", names)
            self.assertNotIn("exec", names)

    def test_the_builder_door_count_is_deliberately_one(self):
        self.assertEqual(list(BUILDER_TOOLS), ["repo.edit"])


if __name__ == "__main__":
    unittest.main()


class TestTheContractItselfIsExercised(unittest.TestCase):
    """system_block had no test and was broken for real.

    Two edits landed in the wrong scope and left it referencing `engine` and
    `menu`, neither defined there -- a guaranteed NameError in the function the
    browser calls on every tool turn. Sixty tests passed anyway, because not one
    of them called it. A load-bearing function with no caller in the suite is
    untested by definition.
    """

    def setUp(self):
        from hcli.chat_tools import build_registry
        self.registry = build_registry(".", ".")

    def test_the_read_only_contract_renders(self):
        from hcli.chat_tools import system_block
        block = system_block(registry=self.registry)
        self.assertIn("web.search", block)
        self.assertIn("fs.search", block)
        self.assertNotIn("repo.edit", block,
                         "a read-only contract advertised the builder door")

    def test_the_write_contract_renders_and_adds_the_builder_door(self):
        from hcli.chat_tools import system_block
        block = system_block(registry=self.registry, write=True)
        self.assertIn("repo.edit", block)
        self.assertIn("insert_after", block, "the edit shape is not stated")

    def test_every_offered_door_appears_with_its_shape(self):
        from hcli.chat_tools import argument_shape, builder_menu, system_block
        for write in (False, True):
            block = system_block(registry=self.registry, write=write)
            for name in builder_menu(write):
                self.assertIn(name, block, f"{name} missing from the contract")
                shape = argument_shape(self.registry, name)
                if shape:
                    self.assertIn(shape.split("}")[0], block,
                                  f"{name} advertised without its shape")

    def test_the_contract_is_byte_identical_between_calls(self):
        # It rides the stable prefix; regenerating different text each turn
        # would cost the 30x native prefix reuse.
        from hcli.chat_tools import system_block
        first = system_block(registry=self.registry)
        second = system_block(registry=self.registry)
        self.assertEqual(first, second)


class TestToolsAreNeverSilentlyDropped(unittest.TestCase):
    """A renderer that cannot declare tools must REFUSE, not omit them.

    Measured: leaving the artifact's `tools` slot empty while describing tools
    in prose made sealed-3.14 reach for a `shell` function nobody offered -- it
    did what its training says to do when no tools are declared. So a renderer
    that silently drops the declarations recreates that failure and blames the
    body for it. Forcing the keyword instead broke every renderer that does not
    take it, which is how three native-connector tests went red.
    """

    class _NoToolsRenderer:
        def render(self, messages, *, thinking_requested):
            class _P:
                text = "rendered"
                prompt_tokens = 1
                qualified = True
            return _P()

    def _connector(self):
        from pathlib import Path
        from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector
        profile = (Path(__file__).resolve().parents[1]
                   / "hawking-native.sealed-3.14.json")
        config = HawkingNativeConfig.from_file(str(profile))
        connector = HawkingNativeConnector(config)
        connector.renderer = self._NoToolsRenderer()
        return connector

    def test_no_tools_still_renders_through_a_plain_renderer(self):
        connector = self._connector()
        prompt = connector._render({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(getattr(prompt, "text", None), "rendered")

    def test_tools_requested_but_undeclarable_raises_rather_than_omitting(self):
        from hcli.hawking_native import HawkingNativeError
        connector = self._connector()
        with self.assertRaises(HawkingNativeError) as caught:
            connector._render({
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "fs.list"}}]})
        message = str(caught.exception)
        self.assertIn("cannot declare tools", message)
        self.assertIn("invent", message, "the refusal does not say why it matters")
