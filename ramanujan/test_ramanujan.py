#!/usr/bin/env python3.12
"""Tests for the Ledger, evidence lattice and stores.

Each test pins a law that is stated in a contract JSON. A law without a test is a
sentence, and the failure mode these guard against is drift rather than decision --
nobody chooses to let Tier 1 become proof by accumulation; it happens because a hundred
supporting computations feel like enough.

    python3.12 -m ramanujan.test_ramanujan
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ramanujan.evidence import (
    PromotionRefused,
    Tier,
    VerifierEvent,
    promote,
    promote_many,
)
from ramanujan.ledger import Ledger, LedgerViolation
from ramanujan.stores import Stores, TribunalRefused


def _ledger(tmp: Path) -> Ledger:
    n = [0]

    def clock() -> str:
        n[0] += 1
        return f"2026-01-01T00:00:{n[0]:02d}Z"

    return Ledger(tmp / "ledger.jsonl", clock=clock)


class TestLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.led = _ledger(self.tmp)

    def test_append_only_no_edit_or_delete_api(self) -> None:
        """The contract forbids editing and removing rows, so the API offers neither."""
        self.assertFalse(hasattr(self.led, "edit"))
        self.assertFalse(hasattr(self.led, "delete"))
        self.assertFalse(hasattr(self.led, "update"))

    def test_unknown_event_kind_is_refused(self) -> None:
        with self.assertRaises(LedgerViolation):
            self.led.append("quietly_fix_things", {}, actor="conjecturer")

    def test_supersession_keeps_both_rows(self) -> None:
        r0 = self.led.append("claim", {"id": "c1", "statement": "wrong"}, actor="conjecturer")
        self.led.supersede(r0.seq, "the statement was wrong", "skeptic", {"statement": "right"})
        self.assertEqual(len(self.led.rows()), 2, "the mistake must remain readable")
        live = [r.seq for r in self.led.live_rows()]
        self.assertNotIn(r0.seq, live, "superseded rows leave the working set")
        self.assertEqual(self.led.rows()[0].payload["statement"], "wrong", "and are unchanged")

    def test_chain_detects_in_place_edit(self) -> None:
        self.led.append("claim", {"id": "c1"}, actor="conjecturer")
        self.led.append("claim", {"id": "c2"}, actor="conjecturer")
        ok, msg = self.led.verify_chain()
        self.assertTrue(ok, msg)
        # Tamper with the file the way a helpful agent might.
        lines = (self.tmp / "ledger.jsonl").read_text().splitlines()
        lines[0] = lines[0].replace('"c1"', '"c1_tampered"')
        (self.tmp / "ledger.jsonl").write_text("\n".join(lines) + "\n")
        ok, msg = Ledger(self.tmp / "ledger.jsonl").verify_chain()
        self.assertFalse(ok)
        self.assertIn("edited in place", msg)


class TestEvidenceLattice(unittest.TestCase):
    def test_tier1_never_becomes_proof_by_accumulation(self) -> None:
        """The invariant most likely to be lost to drift rather than to a decision."""
        thousand = [
            VerifierEvent("computation", "computationalist", None, True, {"n": i})
            for i in range(1000)
        ]
        self.assertEqual(
            promote_many(Tier.ASSERTED, thousand, author="conjecturer"),
            Tier.EMPIRICALLY_SUPPORTED,
            "a thousand computations are still Tier 1",
        )

    def test_tier2_refuses_the_authors_own_fidelity_assessment(self) -> None:
        ev = VerifierEvent("fidelity_assessment", "formalizer", None, True, {})
        with self.assertRaises(PromotionRefused):
            promote(Tier.EMPIRICALLY_SUPPORTED, ev, author="formalizer")

    def test_tier2_refuses_a_non_independent_assessment(self) -> None:
        ev = VerifierEvent("fidelity_assessment", "reviewer", None, False, {})
        with self.assertRaises(PromotionRefused):
            promote(Tier.EMPIRICALLY_SUPPORTED, ev, author="formalizer")

    def test_tier3_requires_a_container_hash(self) -> None:
        ev = VerifierEvent("machine_check", "prover", None, True, {})
        with self.assertRaises(PromotionRefused):
            promote(Tier.FORMALIZED, ev, author="conjecturer")
        ok = VerifierEvent("machine_check", "prover", "deadbeef" * 8, True, {})
        self.assertEqual(promote(Tier.FORMALIZED, ok, author="conjecturer"), Tier.PROVEN)

    def test_model_assertion_carries_no_weight(self) -> None:
        """Tier 0 is where a confident wrong model lives. Math-Preserve predicts
        ' combust' for the capital of France at logit 8.03 -- confidence is not knowledge."""
        self.assertEqual(Tier.ASSERTED, 0)
        with self.assertRaises(PromotionRefused):
            promote(Tier.ASSERTED, VerifierEvent("assertion", "director", None, True, {}), "director")


class TestStores(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.s = Stores(ledger=_ledger(self.tmp))
        self.s.add_claim("c1", "every even n > 2 is a sum of two primes", author="conjecturer")

    def _to_tier2(self) -> None:
        self.s.record_evidence(
            "c1", VerifierEvent("computation", "computationalist", None, True, {})
        )
        self.s.record_evidence(
            "c1", VerifierEvent("fidelity_assessment", "skeptic", None, True, {})
        )

    def test_author_cannot_admit_their_own_claim(self) -> None:
        self._to_tier2()
        with self.assertRaises(TribunalRefused):
            self.s.tribunal_admit("c1", admitting_actor="conjecturer", human_expert_gate=True)

    def test_admission_requires_the_human_gate(self) -> None:
        self._to_tier2()
        with self.assertRaises(TribunalRefused):
            self.s.tribunal_admit("c1", admitting_actor="tribunal", human_expert_gate=False)

    def test_admission_requires_at_least_tier2(self) -> None:
        with self.assertRaises(TribunalRefused):
            self.s.tribunal_admit("c1", admitting_actor="tribunal", human_expert_gate=True)

    def test_admission_succeeds_when_all_three_hold(self) -> None:
        self._to_tier2()
        self.s.tribunal_admit("c1", admitting_actor="tribunal", human_expert_gate=True)
        self.assertTrue(self.s.claims["c1"].admitted)

    def test_burial_is_not_deletion(self) -> None:
        self.s.bury("c1", "counterexample at n=4", actor="adversary")
        self.assertNotIn("c1", [c.id for c in self.s.live_claims()])
        self.assertIn("c1", self.s.claims, "nothing is deleted")
        self.assertIn("c1", [c.id for c in self.s.graveyard()])
        self.assertEqual(self.s.claims["c1"].graveyard_reason, "counterexample at n=4")

    def test_revival_is_a_ledger_event(self) -> None:
        before = len(self.s.ledger.rows())
        self.s.bury("c1", "refuted", actor="adversary")
        # Premises must change after burial: a literature result that withdraws the refutation.
        lit = self.s.ledger.append(
            "literature_query",
            {"claim": "c1", "result": "refuting lemma withdrawn"},
            actor="librarian",
        )
        self.s.revive(
            "c1",
            "the refuting lemma was itself withdrawn",
            actor="librarian",
            premise_change_seq=lit.seq,
        )
        self.assertGreater(len(self.s.ledger.rows()), before + 1)
        self.assertFalse(self.s.claims["c1"].in_graveyard)

    def test_every_state_change_wrote_a_row(self) -> None:
        n = len(self.s.ledger.rows())
        self.s.bury("c1", "x", actor="adversary")
        self.assertEqual(len(self.s.ledger.rows()), n + 1)
        ok, msg = self.s.ledger.verify_chain()
        self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
