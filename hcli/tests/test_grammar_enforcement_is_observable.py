"""The receipt must say whether the JSON mask ran.

"the reply is NOT valid JSON" has two opposite causes -- the mask was off, or
the mask was wrong -- and they need opposite fixes. The resident has always
returned `grammar_enforced`; `_record_model_call` dropped it, so every receipt
in the campaign reported None and neither cause could be ruled out.

Same shape as the comment already sitting above this allowlist: the resident
reported prefix reuse fields and the recorder dropped them, leaving a wall
clock as the only evidence of KV reuse.
"""
from __future__ import annotations

import unittest

from hcli.engine import Engine


def _engine():
    eng = Engine.__new__(Engine)
    eng._model_calls = []
    return eng


def _record(eng, native):
    Engine._record_model_call(
        eng, endpoint="native://resident", finish_reason="stop",
        prompt_tokens=10, completion_tokens=5, wall_s=0.1, native=native,
    )
    return eng._model_calls[-1]


class TestGrammarObservable(unittest.TestCase):
    def test_enforcement_is_recorded_when_the_mask_ran(self):
        entry = _record(_engine(), {"grammar_enforced": True})
        self.assertIs(entry.get("grammar_enforced"), True)

    def test_it_is_recorded_when_the_mask_did_NOT_run(self):
        """False is the diagnostically important value, not the boring one."""
        entry = _record(_engine(), {"grammar_enforced": False})
        self.assertIs(entry.get("grammar_enforced"), False)

    def test_a_reply_without_the_field_records_nothing(self):
        """Negative control: absent is not the same as False."""
        entry = _record(_engine(), {"prefix_reused_tokens": 4})
        self.assertNotIn("grammar_enforced", entry)
        self.assertEqual(entry.get("prefix_reused_tokens"), 4)

    def test_the_existing_native_fields_still_land(self):
        entry = _record(_engine(), {
            "grammar_enforced": True, "prefix_reused_tokens": 7,
            "prefix_source": "checkpoint",
        })
        self.assertEqual(entry.get("prefix_reused_tokens"), 7)
        self.assertEqual(entry.get("prefix_source"), "checkpoint")


if __name__ == "__main__":
    unittest.main()
