#!/usr/bin/env python3.12
"""Tests for the executable cognition mechanisms.

The Math-Preserve test is the one worth reading: it replays the actual failure through the
Cheapest Falsifier and shows the mechanism would have caught it for one forward pass.

    python3.12 -m ramanujan.test_cognition
"""
from __future__ import annotations

import unittest

from ramanujan.cognition import (
    Calibration,
    CheapestFalsifier,
    Falsifier,
    ResearchObjectGraph,
    compile_capsule,
    verify_capsule,
)
from ramanujan.evidence import Tier


class TestResearchObjectGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.g = ResearchObjectGraph()
        for n in ("substrate", "training_plan", "checkpoint", "paper"):
            self.g.add(n, kind="claim")
        self.g.depend("training_plan", "substrate")
        self.g.depend("checkpoint", "training_plan")
        self.g.depend("paper", "checkpoint")

    def test_refutation_propagates_transitively(self) -> None:
        """The property that makes this a graph rather than a diagram."""
        undermined = self.g.refute("substrate", "semantic collapse")
        self.assertEqual(undermined, {"training_plan", "checkpoint", "paper"})

    def test_standing_separates_refuted_from_undermined(self) -> None:
        self.g.refute("substrate", "semantic collapse")
        st = self.g.standing()
        self.assertEqual(st["refuted"], ["substrate"])
        self.assertCountEqual(st["undermined"], ["training_plan", "checkpoint", "paper"])
        self.assertEqual(st["live"], [])

    def test_dependency_requires_both_nodes(self) -> None:
        with self.assertRaises(KeyError):
            self.g.depend("paper", "nonexistent")


class TestCheapestFalsifier(unittest.TestCase):
    def test_the_math_preserve_case(self) -> None:
        """Replay of the real failure.

        Math-Preserve was sealed with 282/282 shards, 59,585 frozen decisions and six green
        gates. A single forward pass on '2 + 2 =' refuted the whole substrate. This asserts
        the mechanism reaches for that first rather than for the expensive checks.
        """
        cf = CheapestFalsifier()
        cf.register(Falsifier("full_support_halo", 26_000.0, lambda: True,
                              "26 tasks of live generation"))
        cf.register(Falsifier("integrity_reverify", 900.0, lambda: True,
                              "re-hash 282 shards; passes, and proves nothing about capability"))
        cf.register(Falsifier("two_plus_two", 1.0, lambda: False,
                              "one forward pass: does the model complete '2 + 2 ='"))

        r = cf.run(proof_attempt_cost=100_000.0)
        self.assertEqual(r["verdict"], "REFUTED")
        self.assertEqual(r["by"], "two_plus_two")
        self.assertEqual(r["attempted"], 1, "the cheapest check must run first and alone")
        self.assertGreater(r["saved"], 99_000.0)

    def test_integrity_checks_do_not_refute_capability(self) -> None:
        """The trap the campaign actually fell into: integrity passing reads like capability."""
        cf = CheapestFalsifier()
        cf.register(Falsifier("integrity", 1.0, lambda: True, "shards hash correctly"))
        r = cf.run(proof_attempt_cost=1000.0)
        self.assertEqual(r["verdict"], "SURVIVED_ALL",
                         "an integrity check cannot refute a capability claim, so the claim survives it")

    def test_inverted_mechanism_is_killed(self) -> None:
        cf = CheapestFalsifier()
        cf.register(Falsifier("expensive", 500.0, lambda: True, "x"))
        r = cf.run(proof_attempt_cost=100.0)
        self.assertEqual(r["verdict"], "MECHANISM_INVERTED")

    def test_refutation_rate_is_the_self_deception_test(self) -> None:
        """A suite that never refutes anything is empty, not cheap."""
        cf = CheapestFalsifier()
        cf.register(Falsifier("vacuous_a", 1.0, lambda: True, "always passes"))
        cf.register(Falsifier("vacuous_b", 2.0, lambda: True, "always passes"))
        cf.run(proof_attempt_cost=1000.0)
        self.assertEqual(cf.refutation_rate(), 0.0)


class TestCalibration(unittest.TestCase):
    def test_only_verifier_outcomes_are_admissible(self) -> None:
        c = Calibration()
        with self.assertRaises(ValueError):
            c.record(0.9, True, source="self_assessment")
        c.record(0.9, True, source="verifier")
        self.assertEqual(len(c.predictions), 1)

    def test_overconfidence_is_detected(self) -> None:
        c = Calibration()
        for _ in range(10):
            c.record(0.95, False, source="verifier")
        self.assertTrue(c.overconfident())

    def test_a_calibrated_predictor_beats_the_base_rate(self) -> None:
        c = Calibration()
        for _ in range(8):
            c.record(0.95, True, source="verifier")
        for _ in range(2):
            c.record(0.05, False, source="verifier")
        self.assertTrue(c.beats_base_rate())
        self.assertFalse(c.overconfident())


class TestCapsules(unittest.TestCase):
    def test_capsule_reproduces_and_detects_change(self) -> None:
        arts = {"result.json": b'{"x":1}', "input.txt": b"hello"}
        cap = compile_capsule({"question": "toy", "fixture": True}, arts)
        ok, msg = verify_capsule(cap, arts)
        self.assertTrue(ok, msg)
        ok, msg = verify_capsule(cap, {"result.json": b'{"x":2}', "input.txt": b"hello"})
        self.assertFalse(ok)
        self.assertIn("changed", msg)

    def test_fixture_capsules_are_marked_non_production(self) -> None:
        cap = compile_capsule({"fixture": True}, {})
        self.assertEqual(cap["authority"], "NON_PRODUCTION_AUTHORITY")

    def test_non_fixture_content_cannot_self_mint_research_authority(self) -> None:
        cap = compile_capsule({"fixture": False, "claim": "caller controlled"}, {})
        self.assertEqual(cap["authority"], "NON_PRODUCTION_AUTHORITY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
