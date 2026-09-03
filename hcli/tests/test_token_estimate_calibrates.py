"""The chars-per-token ratio is measured, not assumed.

A goal that read one 40 KB Python file was refused outright:

    prompt has 6605 tokens and max_new_tokens is 2148; max_seq_len is 8192

The estimator divides characters by a single constant of 3. That is close for
markdown prose and wrong for source code, where indentation and punctuation are
their own tokens -- about 2.4. It estimated ~5300 against a real 6605, a 25%
error against a 12% reserve, and the whole call died.

One constant cannot describe both. The runtime reports a real prompt_tokens on
every reply, so the ratio is learned from it. The MINIMUM ratio ever seen wins,
because it yields the largest token estimate and so the safest budget: being
over by one token costs the entire call, while being under only shortens a reply.
"""
from __future__ import annotations

import unittest

from hcli.engine import (
    _CHARS_PER_TOKEN,
    _CHARS_PER_TOKEN_CEILING,
    _CHARS_PER_TOKEN_FLOOR,
    Engine,
)


def _engine():
    eng = Engine.__new__(Engine)
    eng.config = type("C", (), {"model_tokens": lambda self: (None, None)})()
    eng._context_budget = lambda: type("B", (), {"per_request_ctx": 8192})()
    return eng


class TestCalibration(unittest.TestCase):
    def test_the_default_is_used_before_anything_is_measured(self):
        eng = _engine()
        msgs = [{"role": "user", "content": "x" * 300}]
        self.assertEqual(eng._estimate_prompt_tokens(msgs), 300 // _CHARS_PER_TOKEN)

    def test_a_denser_real_count_lowers_the_ratio(self):
        """The live failure: 15,900 chars really tokenized to 6,605."""
        eng = _engine()
        eng._last_rendered_prompt = "x" * 15900
        eng._calibrate_chars_per_token(6605)
        self.assertLess(eng._chars_per_token, _CHARS_PER_TOKEN)
        self.assertAlmostEqual(eng._chars_per_token, 15900 / 6605, places=3)

    def test_the_estimate_then_covers_the_real_count(self):
        """The property that matters: no second overflow on the same payload."""
        eng = _engine()
        eng._last_rendered_prompt = "x" * 15900
        eng._calibrate_chars_per_token(6605)
        estimate = eng._estimate_prompt_tokens([{"role": "user", "content": "x" * 15900}])
        self.assertGreaterEqual(
            estimate, 6605, "the calibrated estimate still under-counts the real prompt"
        )

    def test_a_sparser_measurement_never_raises_the_ratio(self):
        """Negative control: one prose-heavy call must not undo a code-heavy one."""
        eng = _engine()
        eng._last_rendered_prompt = "x" * 15900
        eng._calibrate_chars_per_token(6605)
        tight = eng._chars_per_token
        eng._last_rendered_prompt = "y" * 16000
        eng._calibrate_chars_per_token(4000)  # 4.0 chars/token, sparser
        self.assertEqual(eng._chars_per_token, tight)

    def test_the_ratio_stays_inside_its_band(self):
        eng = _engine()
        eng._last_rendered_prompt = "x" * 100
        eng._calibrate_chars_per_token(1000)  # absurdly dense
        self.assertGreaterEqual(eng._chars_per_token, _CHARS_PER_TOKEN_FLOOR)
        self.assertLessEqual(eng._chars_per_token, _CHARS_PER_TOKEN_CEILING)

    def test_a_missing_or_useless_count_changes_nothing(self):
        eng = _engine()
        eng._last_rendered_prompt = ""
        eng._calibrate_chars_per_token(500)
        self.assertIsNone(getattr(eng, "_chars_per_token", None))
        eng._last_rendered_prompt = "x" * 100
        eng._calibrate_chars_per_token(0)
        self.assertIsNone(getattr(eng, "_chars_per_token", None))


if __name__ == "__main__":
    unittest.main()
