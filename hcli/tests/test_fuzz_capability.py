"""FUZZ finds a real crash, clears a real safe target, and returns a reproducer.

S036 s47: HCLI reasons about boundary/harness/corpus/invariant; the machine
handles iterations, mutation, crash capture, dedup, minimization; raw noise
never enters context -- only a minimized reproducer does. A crash you cannot
reproduce is a story; a minimized input is a fact.
"""
from __future__ import annotations

import unittest

from hcli.capabilities import fuzz


def _buggy(s):
    if not isinstance(s, str):
        raise ValueError("need a string")
    return s[0]  # IndexError on ""


def _safe(s):
    if not isinstance(s, str) or not s:
        raise ValueError("bad input")
    return s.strip()


class TestFuzz(unittest.TestCase):
    def test_a_real_crash_is_found(self):
        r = fuzz(_buggy, name="buggy", allowed=(ValueError,))
        self.assertFalse(r.clean)
        kinds = {c.exception for c in r.crashes}
        self.assertIn("IndexError", kinds)

    def test_the_crash_is_minimized_to_a_reproducer(self):
        r = fuzz(_buggy, allowed=(ValueError,))
        crash = next(c for c in r.crashes if c.exception == "IndexError")
        self.assertEqual(crash.minimized_repr, "''",
                         "the reproducer was not minimized to the empty string")

    def test_a_safe_target_is_clean(self):
        r = fuzz(_safe, name="safe", allowed=(ValueError,))
        self.assertTrue(r.clean, r.summary())
        self.assertIn("held the invariant", r.note)

    def test_crashes_are_deduplicated(self):
        # A function that always IndexErrors must report ONE crash, not hundreds.
        def always(s):
            raise IndexError("boom")
        r = fuzz(always, allowed=(ValueError,))
        self.assertEqual(len(r.crashes), 1, "one bug reported many times")

    def test_the_allowed_exception_is_not_a_crash(self):
        # The invariant IS that only ValueError is raised; raising it is contract.
        def only_value(s):
            raise ValueError("expected")
        r = fuzz(only_value, allowed=(ValueError,))
        self.assertTrue(r.clean)

    def test_a_seed_is_mutated_into_the_corpus(self):
        seen = []
        def record(s):
            seen.append(s)
        fuzz(record, seeds=["hello"], allowed=(Exception,))
        self.assertIn("hello", seen)
        self.assertTrue(any(s == "hellohello" for s in seen),
                        "the seed was not mutated")

    def test_the_summary_is_actionable_not_noise(self):
        r = fuzz(_buggy, allowed=(ValueError,))
        summary = r.summary()
        self.assertIn("reproduce with", summary)
        self.assertLess(len(summary), 800)


if __name__ == "__main__":
    unittest.main()
