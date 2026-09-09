"""The body is told it can TEST. It must have a door that actually tests.

`/health` advertises a capability map including TEST: "run admitted tests", and
the registry really does carry `tests.run` -- a bounded pytest/unittest/cargo
runner with a typed schema. But the chat surface offered 11 doors out of 109
registry tools, and none of them ran anything. `debug.diagnose` localizes a
failure *from a test/command output handle* it cannot itself produce, and
`repo.edit` runs tests only as a rollback gate on a mutation that has already
been written.

So a body asked to "write a RED discriminator" could not run one. It could
form a hypothesis and it could edit source, but the DISCRIMINATE step between
them was structurally missing -- the step that decides whether the repair is
needed at all. A 66-cycle autonomous run produced zero edits, and when the
repaired loop finally produced a falsifiable hypothesis naming an exact test,
the body still had no way to execute it and find out it was wrong.

This is the same disease this project has recorded before: a capability that
nothing can call does not exist. Registration is not reachability.
"""
from __future__ import annotations

import unittest

from hcli.chat_tools import BUILDER_TOOLS, CHAT_TOOLS, builder_menu, openai_schemas


class TestTheBodyCanRunItsOwnDiscriminator(unittest.TestCase):
    def test_a_write_session_is_offered_a_way_to_run_tests(self):
        menu = builder_menu(True)
        self.assertIn("tests.run", menu,
                      "a body asked for RED before GREEN must be able to run something")

    def test_the_runner_reaches_the_model_as_a_callable_schema(self):
        """In the menu is not enough -- it has to survive into the tool schemas.

        serve.py hands the body openai_schemas(registry, builder_menu(write)).
        A name that is dropped there is a door the body never sees.
        """
        emitted = [s["function"]["name"]
                   for s in openai_schemas(None, list(builder_menu(True)))]
        self.assertIn("tests.run", emitted)

    def test_a_read_only_session_is_NOT_given_execution(self):
        """Negative control, and the reason this door is a builder door.

        tests.run executes code on the host. A session that was not granted
        write authority has no business running anything, so the gate that
        withholds repo.edit must withhold this too.
        """
        self.assertNotIn("tests.run", builder_menu(False))
        self.assertNotIn("tests.run", CHAT_TOOLS)

    def test_the_door_is_described_well_enough_to_call(self):
        description = BUILDER_TOOLS.get("tests.run", "")
        self.assertTrue(description, "an undescribed tool is one the body will not choose")
        self.assertIn("test", description.lower())


if __name__ == "__main__":
    unittest.main()
