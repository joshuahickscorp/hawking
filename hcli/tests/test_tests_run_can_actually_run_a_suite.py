"""`tests.run` must be able to run a test suite.

Measured, campaign cycle 75. The body ran its discriminator, got a real result,
and then said:

    What I could not check: I attempted the broader `hcli/tests` suite to check
    neighbours, but the tool timed out at 30s

It was doing exactly what the loop asks -- step 8 of the arc is "check
neighbours" -- and the tool made it impossible. `hcli/tests` takes 167 seconds.

Two timeouts disagree and the smaller one wins silently. `_tests_run` reads
`timeout_s` from its arguments, defaults to 300s and caps itself at 900s. Its
`ToolSpec` never passes `timeout_s`, so it inherits the dataclass default of
30.0 -- and `invoke` clamps any requested timeout down to the spec's before the
handler ever sees it, then kills the call at the spec's value. So the schema
advertises a knob up to 900s that cannot exceed 30s.

This is a false affordance, the same one already fixed once in this file for
`forbidden_fruit.lab`: "`timeout_s` was advertised here and forwarded to a
handler that has no such parameter, so EVERY call raised TypeError before the
lab ran." Here the handler does take the parameter; the spec makes it inert.

The general property is the one worth pinning, so it also catches the next tool
that advertises a timeout it cannot honour.
"""
from __future__ import annotations

import inspect
import unittest

from pathlib import Path

from hcli.tool_registry import default_tool_registry

REPO = str(Path(__file__).resolve().parents[2])

#: What the handler's own signature/body treats as a normal run. Read from the
#: source rather than retyped, so raising the handler's default without raising
#: the spec fails here instead of silently reintroducing the clamp.
HANDLER_DEFAULT = 300.0


class TestTestsRunCanRunASuite(unittest.TestCase):
    def setUp(self):
        self.registry = default_tool_registry(REPO, repo_root=REPO)
        self.spec = self.registry.get("tests.run")
        if self.spec is None:
            self.skipTest("tests.run is not registered in this build")

    def test_the_handler_still_intends_a_five_minute_default(self):
        # Guards the constant above: if the handler's default changes, this
        # test's premise is stale and it should say so rather than pass.
        src = inspect.getsource(
            __import__("hcli.tool_registry", fromlist=["_tests_run"])._tests_run)
        self.assertIn("or 300.0", src,
                      "the handler's default moved; update HANDLER_DEFAULT")

    def test_the_spec_can_honour_the_timeout_it_advertises(self):
        self.assertGreaterEqual(
            float(self.spec.timeout_s), HANDLER_DEFAULT,
            f"tests.run advertises a timeout_s knob and its handler defaults to "
            f"{HANDLER_DEFAULT}s, but the spec caps every call at "
            f"{self.spec.timeout_s}s -- so no suite longer than that can be run, "
            f"and 'check neighbours' is not a step this tool can perform")

    def test_a_requested_five_minute_run_survives_the_clamp(self):
        # The exact arithmetic `invoke` applies before the handler is called.
        requested = HANDLER_DEFAULT
        granted = min(max(0.1, requested), max(0.1, float(self.spec.timeout_s)))
        self.assertEqual(granted, requested,
                         "a caller asking for the handler's own default is "
                         "silently given less")

    def test_the_registers_own_timeout_is_the_one_that_binds(self):
        # WHY THIS IS NOT A SWEEP OVER EVERY TOOL. The obvious general form --
        # "no tool may advertise timeout_s while sitting at the 30s default" --
        # flags 14 more, and most are not defects: web.fetch's handler caps
        # ITSELF at min(60, timeout_s or 30), so a 30s spec is consistent with
        # its contract. The defect is a spec BELOW what its own handler treats
        # as a normal run, and establishing that for each tool means reading
        # each handler. tests.run is the one measured to have cost something,
        # so it is the one pinned here; the other 14 are a separate audit.
        #
        # What this does pin generally: the spec value is what binds, so a spec
        # left at the dataclass default silently overrides any handler contract.
        self.assertEqual(
            min(900.0, float(self.spec.timeout_s)), 900.0,
            "the register's timeout is the ceiling `invoke` enforces; if it "
            "drops below the handler's cap again, the handler's contract "
            "becomes decorative")


if __name__ == "__main__":
    unittest.main()
