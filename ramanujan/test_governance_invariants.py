#!/usr/bin/env python3.12
"""Adversarial invariant tests for Ramanujan governance.

Pins every law the deliverable requires, including attempts that must fail:

  - edit the ledger
  - self-admit a claim
  - promote as a generator
  - free-resurrect a buried claim
  - flip RAMANUJAN_RESEARCH_AUTHORIZED

    python3 -m ramanujan.test_governance_invariants
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ramanujan.economics import (
    BranchAccount,
    BranchHalted,
    SessionEconomics,
    SessionHalted,
)
from ramanujan.evidence import PromotionRefused, Tier, VerifierEvent, promote
from ramanujan.forge import f0_diagnose
from ramanujan.ledger import Ledger, LedgerViolation
from ramanujan.limits import LimitRegistry
from ramanujan.roles import (
    GENERATOR_IDS,
    PROMOTER_IDS,
    ROLE_CATALOG,
    CapabilityRefused,
    RoleCapability,
    RoleSession,
    capability_for,
    generators_may_not_promote,
)
from ramanujan.search import SearchEconomics
from ramanujan.sovereignty import SovereigntyHooks, SovereigntyRefused
from ramanujan.stores import (
    STORE_NAMES,
    GraveyardRefused,
    StoreRefused,
    Stores,
    TribunalRefused,
)


def _ledger(tmp: Path) -> Ledger:
    n = [0]

    def clock() -> str:
        n[0] += 1
        return f"2026-01-01T00:00:{n[0]:02d}Z"

    return Ledger(tmp / "ledger.jsonl", clock=clock)


def _stores(tmp: Path) -> Stores:
    return Stores(ledger=_ledger(tmp))


class TestLedgerAppendOnly(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.led = _ledger(self.tmp)

    def test_no_edit_delete_update_api(self) -> None:
        for name in ("edit", "delete", "update", "remove"):
            self.assertFalse(hasattr(self.led, name), name)

    def test_adversarial_in_place_edit_detected(self) -> None:
        self.led.append("claim", {"id": "c1"}, actor="conjecturer")
        self.led.append("claim", {"id": "c2"}, actor="conjecturer")
        ok, _ = self.led.verify_chain()
        self.assertTrue(ok)
        path = self.tmp / "ledger.jsonl"
        lines = path.read_text().splitlines()
        lines[0] = lines[0].replace('"c1"', '"c1_tampered"')
        path.write_text("\n".join(lines) + "\n")
        ok, msg = Ledger(path).verify_chain()
        self.assertFalse(ok)
        self.assertIn("edited in place", msg)

    def test_adversarial_row_deletion_breaks_chain(self) -> None:
        self.led.append("claim", {"id": "c1"}, actor="conjecturer")
        self.led.append("claim", {"id": "c2"}, actor="conjecturer")
        self.led.append("claim", {"id": "c3"}, actor="conjecturer")
        path = self.tmp / "ledger.jsonl"
        lines = path.read_text().splitlines()
        # Drop the middle row -- a "helpful" rewrite.
        path.write_text(lines[0] + "\n" + lines[2] + "\n")
        ok, msg = Ledger(path).verify_chain()
        self.assertFalse(ok)
        self.assertTrue("prev_hash" in msg or "edited" in msg or "chain" in msg)

    def test_supersession_is_not_edit(self) -> None:
        r = self.led.append("claim", {"id": "c1", "v": 1}, actor="conjecturer")
        self.led.supersede(r.seq, "fix", "skeptic", {"v": 2})
        self.assertEqual(len(self.led.rows()), 2)
        self.assertEqual(self.led.rows()[0].payload["v"], 1)


class TestPromotionRefuses(unittest.TestCase):
    def test_wrong_kind_refuses_not_ignores(self) -> None:
        with self.assertRaises(PromotionRefused):
            promote(
                Tier.ASSERTED,
                VerifierEvent("assertion", "director", None, True, {}),
                author="conjecturer",
            )

    def test_attempt_promotion_on_store_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = _stores(Path(td))
            s.add_claim("c1", "p", author="conjecturer")
            with self.assertRaises(PromotionRefused):
                s.attempt_promotion(
                    "c1",
                    VerifierEvent("assertion", "director", None, True, {}),
                )
            self.assertEqual(s.claims["c1"].tier, Tier.ASSERTED)


class TestSevenStores(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.s = _stores(self.tmp)

    def test_all_seven_named_and_writable(self) -> None:
        self.assertEqual(len(STORE_NAMES), 7)
        self.s.add_problem("p1", "is every even > 2 sum of two primes?", actor="director")
        self.s.add_claim("c1", "goldbach", author="conjecturer")
        self.s.add_proof_state("ps1", "c1", "theorem t : True := trivial", actor="formalizer")
        self.s.add_counterexample("ce1", "c1", {"n": 4}, actor="skeptic")
        self.s.add_prior_art("pa1", {"title": "Euler"}, actor="librarian")
        self.s.add_strategy("st1", {"plan": "try small n"}, actor="cartographer")
        self.s.bury("c1", "counterexample", actor="adversary")
        inv = self.s.store_inventory()
        for name in STORE_NAMES:
            self.assertIn(name, inv)
        self.assertEqual(inv["problem"], 1)
        self.assertEqual(inv["claim"], 1)
        self.assertEqual(inv["proof_state"], 1)
        self.assertEqual(inv["counterexample"], 1)
        self.assertEqual(inv["prior_art"], 1)
        self.assertEqual(inv["strategy"], 1)
        self.assertEqual(inv["graveyard"], 1)

    def test_no_delete_api_on_stores(self) -> None:
        for name in ("delete", "remove", "edit", "update"):
            meth = getattr(Stores, name, None)
            if meth is not None and callable(meth):
                self.assertFalse(
                    getattr(meth, "__qualname__", "").startswith("Stores."),
                    f"Stores must not define {name}",
                )

    def test_overwrite_refused(self) -> None:
        self.s.add_claim("c1", "p", author="conjecturer")
        with self.assertRaises(LedgerViolation):
            self.s.add_claim("c1", "q", author="conjecturer")
        self.s.add_problem("p1", "q?", actor="director")
        with self.assertRaises(StoreRefused):
            self.s.add_problem("p1", "other?", actor="director")


class TestTribunalSeparation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.s = _stores(self.tmp)
        self.s.add_claim("c1", "goldbach", author="conjecturer")
        self.s.record_evidence(
            "c1", VerifierEvent("computation", "computationalist", None, True, {})
        )
        self.s.record_evidence(
            "c1", VerifierEvent("fidelity_assessment", "skeptic", None, True, {})
        )

    def test_adversarial_self_admit_refused(self) -> None:
        with self.assertRaises(TribunalRefused) as ctx:
            self.s.tribunal_admit("c1", admitting_actor="conjecturer", human_expert_gate=True)
        self.assertIn("never the system that admits", str(ctx.exception))
        self.assertFalse(self.s.claims["c1"].admitted)

    def test_foreign_admitter_with_gate_succeeds(self) -> None:
        self.s.tribunal_admit("c1", admitting_actor="tribunal", human_expert_gate=True)
        self.assertTrue(self.s.claims["c1"].admitted)


class TestGeneratorsCannotPromote(unittest.TestCase):
    def test_construction_forbids_promote_on_generators(self) -> None:
        self.assertTrue(generators_may_not_promote())
        for rid in GENERATOR_IDS:
            self.assertNotIn("promote_claim", ROLE_CATALOG[rid].capabilities, rid)
            self.assertFalse(ROLE_CATALOG[rid].may_promote, rid)

    def test_adversarial_generator_promote_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            RoleSession("conjecturer", stores).write_claim("c1", "p")
            stores.record_evidence(
                "c1", VerifierEvent("computation", "computationalist", None, True, {})
            )
            stores.record_evidence(
                "c1", VerifierEvent("fidelity_assessment", "skeptic", None, True, {})
            )
            for rid in sorted(GENERATOR_IDS):
                sess = RoleSession(rid, stores)
                with self.assertRaises(CapabilityRefused, msg=rid):
                    sess.tribunal_admit("c1", human_expert_gate=True)
                with self.assertRaises(CapabilityRefused, msg=rid):
                    sess.attempt_promote("c1")
            self.assertFalse(stores.claims["c1"].admitted)

    def test_cannot_derive_promote_for_generator(self) -> None:
        with self.assertRaises(CapabilityRefused):
            RoleCapability.derive_subset(
                ROLE_CATALOG["conjecturer"],
                frozenset({"write_claim", "promote_claim"}),
            )

    def test_only_two_promoters(self) -> None:
        self.assertEqual(PROMOTER_IDS, frozenset({"tribunal", "verifier"}))


class TestGraveyardNotDeletion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.s = _stores(self.tmp)
        self.s.add_claim("c1", "2+2=5", author="conjecturer")

    def test_burial_keeps_claim_auditable_and_dead(self) -> None:
        self.s.bury("c1", "counterexample: 2+2=4", actor="adversary")
        self.assertTrue(self.s.is_auditable("c1"))
        self.assertIn("c1", self.s.claims)
        self.assertTrue(self.s.claims["c1"].in_graveyard)
        self.assertNotIn("c1", [c.id for c in self.s.live_claims()])
        self.assertIn("c1", [c.id for c in self.s.graveyard()])
        # Statement still readable.
        self.assertEqual(self.s.claims["c1"].statement, "2+2=5")

    def test_adversarial_free_resurrection_refused(self) -> None:
        self.s.bury("c1", "refuted", actor="adversary")
        with self.assertRaises(GraveyardRefused) as ctx:
            self.s.revive("c1", "I changed my mind", actor="conjecturer")
        self.assertIn("without a premise-changing", str(ctx.exception))
        self.assertTrue(self.s.claims["c1"].in_graveyard)
        self.assertTrue(self.s.is_auditable("c1"))

    def test_adversarial_pre_burial_evidence_cannot_resurrect(self) -> None:
        # Evidence before burial does not license revival.
        pre = self.s.ledger.append(
            "literature_query",
            {"note": "before burial"},
            actor="librarian",
        )
        self.s.bury("c1", "refuted", actor="adversary")
        with self.assertRaises(GraveyardRefused):
            self.s.revive(
                "c1",
                "using old evidence",
                actor="librarian",
                premise_change_seq=pre.seq,
            )
        self.assertTrue(self.s.claims["c1"].in_graveyard)

    def test_adversarial_wrong_kind_cannot_resurrect(self) -> None:
        self.s.bury("c1", "refuted", actor="adversary")
        row = self.s.ledger.append(
            "budget_grant",
            {"branch": "b1", "max": 1},
            actor="economist",
        )
        with self.assertRaises(GraveyardRefused):
            self.s.revive(
                "c1",
                "budget is not a premise",
                actor="economist",
                premise_change_seq=row.seq,
            )

    def test_valid_revival_with_post_burial_literature(self) -> None:
        self.s.bury("c1", "refuted", actor="adversary")
        lit = self.s.ledger.append(
            "literature_query",
            {"result": "refutation withdrawn"},
            actor="librarian",
        )
        self.s.revive(
            "c1",
            "premises changed",
            actor="librarian",
            premise_change_seq=lit.seq,
        )
        self.assertFalse(self.s.claims["c1"].in_graveyard)

    def test_buried_claim_cannot_be_admitted_or_promoted(self) -> None:
        self.s.record_evidence(
            "c1", VerifierEvent("computation", "computationalist", None, True, {})
        )
        self.s.record_evidence(
            "c1", VerifierEvent("fidelity_assessment", "skeptic", None, True, {})
        )
        self.s.bury("c1", "later refuted", actor="adversary")
        with self.assertRaises(TribunalRefused):
            self.s.tribunal_admit("c1", admitting_actor="tribunal", human_expert_gate=True)
        with self.assertRaises(GraveyardRefused):
            self.s.attempt_promotion(
                "c1",
                VerifierEvent("machine_check", "prover", "abc", True, {}),
            )

    def test_proof_state_refused_on_buried_claim(self) -> None:
        self.s.bury("c1", "dead", actor="adversary")
        with self.assertRaises(GraveyardRefused):
            self.s.add_proof_state("ps1", "c1", "sorry", actor="formalizer")


class TestEconomicsStopRule(unittest.TestCase):
    def test_branch_cannot_talk_past_budget(self) -> None:
        b = BranchAccount("b", economics=SearchEconomics(max_expansions=2))
        b.charge()
        b.charge()
        with self.assertRaises(BranchHalted):
            b.charge()
        self.assertTrue(b.is_halted())

    def test_session_halts_all_branches(self) -> None:
        sess = SessionEconomics("s1", max_total_expansions=3)
        b1 = sess.open_branch("b1", max_expansions=10)
        b2 = sess.open_branch("b2", max_expansions=10)
        sess.charge("b1")
        sess.charge("b2")
        sess.charge("b1")
        self.assertTrue(sess.is_halted())
        self.assertTrue(b1.is_halted())
        self.assertTrue(b2.is_halted())
        with self.assertRaises(SessionHalted):
            sess.charge("b2")
        with self.assertRaises(SessionHalted):
            sess.open_branch("b3")


class TestLimitRegistryAndResearchFence(unittest.TestCase):
    def test_research_authorized_stays_false_no_flip_path(self) -> None:
        reg = LimitRegistry()
        self.assertFalse(reg.research_authorized())
        for name in (
            "set_research_authorized",
            "authorize_research",
            "flip_research",
            "enable_research",
        ):
            self.assertFalse(hasattr(reg, name), name)
        pub = reg.as_public_dict()
        self.assertFalse(pub["RAMANUJAN_RESEARCH_AUTHORIZED"])

    def test_run_research_blocked(self) -> None:
        reg = LimitRegistry()
        v = reg.consult("run_research")
        self.assertFalse(v.allowed)

    def test_sovereignty_refuse_research(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            led = _ledger(Path(td))
            hooks = SovereigntyHooks(limits=LimitRegistry(), ledger=led)
            with self.assertRaises(SovereigntyRefused):
                hooks.refuse_research(role_id="director")
            kinds = [r.kind for r in led.rows()]
            self.assertIn("sovereignty_event", kinds)

    def test_forge_f2_gated_f0_allowed(self) -> None:
        hooks = SovereigntyHooks(limits=LimitRegistry())
        self.assertTrue(hooks.forge_gate("F0")["allowed"])
        self.assertTrue(hooks.forge_gate("F1")["allowed"])
        self.assertFalse(hooks.forge_gate("F2")["allowed"])
        with self.assertRaises(SovereigntyRefused):
            hooks.require_forge_stage("F9")


class TestF0SeesCompletedSurface(unittest.TestCase):
    def test_f0_lists_seven_stores_and_sovereignty(self) -> None:
        receipt = f0_diagnose()
        self.assertFalse(receipt["RAMANUJAN_RESEARCH_AUTHORIZED"])
        self.assertEqual(receipt["components"]["stores"]["count"], 7)
        self.assertEqual(
            set(receipt["components"]["stores"]["names"]), set(STORE_NAMES)
        )
        self.assertEqual(
            receipt["components"]["math_preserve_teacher_traces"]["status"], "REFUSED"
        )
        self.assertIn("sovereignty", receipt["components"])
        self.assertIn("graveyard", receipt["components"])
        self.assertTrue(receipt["forge_gate"]["allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
