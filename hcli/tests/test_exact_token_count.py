"""Count the prompt with the resident's own tokenizer instead of guessing.

Estimating characters-per-token produced two OPPOSITE failures against the same
8192-token window, and no single ratio satisfies both:

  - at 3 chars/token the estimate under-counted Python source by 25%, the
    prompt overflowed, and the call was refused outright.
  - the 30% reserve that fixed the overflow then left 1107 completion tokens
    after a 5422-token prompt, and the reply was truncated mid-object:
    "produced 1107 tokens against a 1107-token completion budget ... and never
    closed the JSON object".

The tokenizer is on disk and `tokenizers` is installed in the runtime venv, so
the count can simply be exact. Measured on the truncating prompt, the
completion budget goes 1107 -> 2554.

The fallback stays: a checkout without `tokenizers` keeps the heuristic and the
wider reserve, which is why the exact tests skip rather than fail there.
"""
from __future__ import annotations

import unittest

from hcli.engine import (
    _CTX_ESTIMATE_ERROR,
    _CTX_EXACT_MARGIN,
    Engine,
    _exact_tokenizer,
)

HAVE_TOKENIZER = _exact_tokenizer() is not None


def _engine():
    eng = Engine.__new__(Engine)
    eng.config = type("C", (), {"model_tokens": lambda self: (None, None)})()
    eng._context_budget = lambda: type("B", (), {"per_request_ctx": 8192})()
    return eng


class TestExactCount(unittest.TestCase):
    @unittest.skipUnless(HAVE_TOKENIZER, "tokenizers not installed in this interpreter")
    def test_the_count_is_marked_exact(self):
        eng = _engine()
        eng._estimate_prompt_tokens([{"role": "user", "content": "def f():\n    return 1\n"}])
        self.assertTrue(eng._last_estimate_exact)

    @unittest.skipUnless(HAVE_TOKENIZER, "tokenizers not installed in this interpreter")
    def test_an_exact_count_buys_back_the_completion_budget(self):
        """The truncation: 5422 prompt tokens left only 1107 to answer in."""
        eng = _engine()
        eng._last_estimate_exact = True
        exact, _ = eng._resolve_max_tokens(5422)
        eng._last_estimate_exact = False
        estimated, _ = eng._resolve_max_tokens(5422)
        self.assertGreater(exact, estimated)
        self.assertGreater(exact, 2000, "an exact count should leave room to answer")

    @unittest.skipUnless(HAVE_TOKENIZER, "tokenizers not installed in this interpreter")
    def test_it_still_fits_the_window(self):
        eng = _engine()
        for prompt in (1, 1000, 4000, 5422, 7000, 8100):
            eng._last_estimate_exact = True
            max_new, _ = eng._resolve_max_tokens(prompt)
            self.assertLessEqual(prompt + max_new, 8193, f"overflow at {prompt}")

    def test_a_missing_tokenizer_falls_back_to_the_heuristic(self):
        """Negative control: a checkout without `tokenizers` must still run.

        The wider reserve is what keeps the estimated path safe, so it must not
        be narrowed to the exact one.
        """
        self.assertGreater(_CTX_ESTIMATE_ERROR, _CTX_EXACT_MARGIN)
        eng = _engine()
        eng._last_estimate_exact = False
        max_new, _ = eng._resolve_max_tokens(5422)
        self.assertLessEqual(5422 + max_new, 8193)

    def test_the_estimator_never_returns_zero(self):
        eng = _engine()
        self.assertGreaterEqual(eng._estimate_prompt_tokens([]), 1)
        self.assertGreaterEqual(eng._estimate_prompt_tokens([{"content": ""}]), 1)


if __name__ == "__main__":
    unittest.main()
