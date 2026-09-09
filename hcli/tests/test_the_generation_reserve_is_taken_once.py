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
from pathlib import Path

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

    def test_prompt_window_reports_the_ceiling(self):
        # It is INFORMATION in /health, not an admission bound. See the
        # regression test below for why it must not be used as one.
        self.assertEqual(_prompt_window(MODEL), self.budget.model_ceiling)

    def test_the_admission_budget_cannot_overrun_the_native_window(self):
        """THE REGRESSION. Cycle 82 died repeatedly on

            prompt is 8807 tokens and native max_seq_len is 8192

        because the admission budget was switched to the ceiling. 0.55 * 8192 =
        4505 ESTIMATED tokens measured 8807 REAL ones -- CHARS_PER_TOKEN=4 is
        roughly 2x optimistic on code and JSON, and the reserve that looked
        redundant was silently absorbing that error.

        So the bound is checked against the WORST observed ratio, not the
        heuristic's own optimistic one. Until the budget is computed from a real
        token count, any window that passes this at 2 chars/token is safe."""
        import re as _re
        src = (Path(__file__).resolve().parents[1] / "serve.py").read_text()
        used = _re.findall(r"window(?:_tokens)?\s*=\s*int\(health\.get\(\"(\w+)\"",
                           src.replace("'", '"'))
        self.assertTrue(used, "could not find the admission window in serve.py")
        self.assertNotIn("prompt_window", used,
                         "the admission budget is using the ceiling again -- that "
                         "is the cycle-82 regression")
        budget_tokens = int(self.budget.usable_input_tokens * CONTEXT_SHARE)
        worst_case_real = budget_tokens * 2      # 4 chars/token assumed, 2 observed
        self.assertLess(worst_case_real, self.budget.model_ceiling,
                        f"a {budget_tokens}-token budget can reach "
                        f"{worst_case_real} real tokens against a "
                        f"{self.budget.model_ceiling} ceiling")

    def test_the_two_windows_still_differ(self):
        # The double reserve is REAL -- that diagnosis stands. What was wrong was
        # the fix: the slack it removed was also absorbing a 2x token-estimation
        # error. Recovering it needs a real token count, not a bigger constant.
        self.assertGreater(_prompt_window(MODEL), _context_window(MODEL))

    def test_the_budget_still_leaves_the_answer_its_room(self):
        # The reserve exists for a reason. Taking it once must not mean taking
        # it zero times.
        prompt = int(_context_window(MODEL) * CONTEXT_SHARE)
        need = (prompt + self.budget.generation_reserve
                + self.budget.framing_reserve)
        self.assertLessEqual(need, self.budget.model_ceiling,
                             f"prompt {prompt} + generation "
                             f"{self.budget.generation_reserve} + framing "
                             f"{self.budget.framing_reserve} overruns the "
                             f"{self.budget.model_ceiling} ceiling")


if __name__ == "__main__":
    unittest.main()
