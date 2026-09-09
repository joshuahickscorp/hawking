"""Room for the answer must be reserved ONCE, not twice.

`context_budget.resolve` is the repo's authority on window arithmetic and it
already does the subtraction:

    model_ceiling 8192 - generation_reserve 2048 - framing_reserve 512
        = usable_input_tokens 5632

`context_window` in the health payload reports that 5632. The chat path then
multiplies it by CONTEXT_SHARE (0.55) to "leave room for the answer" -- room
that was subtracted two lines earlier in another module. `compact`'s own
docstring says what the share is measured against: "The body's own window is the
ceiling; this is the share of it the conversation may occupy". The ceiling is
8192. It was being handed 5632.

Cost: the conversation budget is 3097 tokens where the design intends 4505. The
self-development campaign ran 74 cycles against a body given 31% less context
than the arithmetic allows, on a path whose failures were all shaped like
running out of room.

This does not raise any risk of overrun: 4505 prompt + 2048 generation + 512
framing = 7065 against a 8192 ceiling, 1127 to spare. The bug was never that
the reserve was too small.
"""
from __future__ import annotations

import unittest

from hcli.chat_state import CONTEXT_SHARE
from hcli.context_budget import resolve
from hcli.serve import _context_window, _prompt_window

MODEL = "hcli/hawking-native.sealed-3.14.json"


class TestTheReserveIsTakenOnce(unittest.TestCase):
    def setUp(self):
        self.budget = resolve(model_path=MODEL)
        if not isinstance(getattr(self.budget, "model_ceiling", None), int):
            self.skipTest("this profile is not on this machine")

    def test_the_health_window_is_still_the_usable_input(self):
        # Unchanged on purpose: `context_window` is a public field and callers
        # read it as "how much input may I send", which is what it means.
        self.assertEqual(_context_window(MODEL), self.budget.usable_input_tokens)

    def test_the_share_is_measured_against_the_ceiling_not_the_net_window(self):
        self.assertEqual(_prompt_window(MODEL), self.budget.model_ceiling,
                         "CONTEXT_SHARE is a share OF THE CEILING; measuring it "
                         "against a window that already subtracted the answer "
                         "reserves the same room twice")

    def test_the_double_reserve_actually_cost_context(self):
        # Guards the fix against being cosmetic: the two windows must differ,
        # or there was nothing to fix and this whole file is noise.
        self.assertGreater(_prompt_window(MODEL), _context_window(MODEL))
        was = int(_context_window(MODEL) * CONTEXT_SHARE)
        now = int(_prompt_window(MODEL) * CONTEXT_SHARE)
        self.assertGreater(now, was)

    def test_the_budget_still_leaves_the_answer_its_room(self):
        # The reserve exists for a reason. Taking it once must not mean taking
        # it zero times.
        prompt = int(_prompt_window(MODEL) * CONTEXT_SHARE)
        need = (prompt + self.budget.generation_reserve
                + self.budget.framing_reserve)
        self.assertLessEqual(need, self.budget.model_ceiling,
                             f"prompt {prompt} + generation "
                             f"{self.budget.generation_reserve} + framing "
                             f"{self.budget.framing_reserve} overruns the "
                             f"{self.budget.model_ceiling} ceiling")


if __name__ == "__main__":
    unittest.main()
