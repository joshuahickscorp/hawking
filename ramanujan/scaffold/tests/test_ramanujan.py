#!/usr/bin/env python3.12
"""Tests for the Ledger, evidence lattice and stores.

Each test pins a law that is stated in a contract JSON. A law without a test is a
sentence, and the failure mode these guard against is drift rather than decision --
nobody chooses to let Tier 1 become proof by accumulation; it happens because a hundred
supporting computations feel like enough.

    python3.12 -m ramanujan.test_ramanujan
"""
from __future__ import annotations

import json
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
from ramanujan.controllers import (
    ControllerRefused,
    FormalizerController,
    RepairController,
    qualification_contracts,
    verify_q0_evidence_bundle,
)
from ramanujan.layout import AUDITS_ROOT, CONTAINER_ROOT, CONTRACTS_ROOT, RUNTIME_RECORDS_ROOT


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


class TestFormalControllers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.stores = Stores(ledger=_ledger(self.tmp))
        self.stores.add_claim("c1", "two plus two equals four", author="conjecturer")

    def test_q_contracts_are_complete_and_non_authorizing(self) -> None:
        contracts = qualification_contracts()
        self.assertEqual(tuple(contracts), ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
        self.assertEqual(contracts["Q1"]["status"], "PROVEN_OFFLINE")
        self.assertIn("PENDING", contracts["Q2"]["status"])
        self.assertIn("independent", contracts["Q4"]["admission"].lower())

    def test_q_contract_table_and_receipts_are_hash_bound(self) -> None:
        source = CONTRACTS_ROOT / "RAMANUJAN_Q0_Q6_CONTRACTS.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["seal_sha256"]), 64)
        self.assertIn("ramanujan/RAMANUJAN_Q0_CLOSURE.json", raw["receipt_bindings"])
        tampered = dict(raw)
        tampered["authority"] = "silently authorized"
        path = self.tmp / "tampered-contracts.json"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaises(ControllerRefused, msg="mutable contract tables must fail closed"):
            qualification_contracts(path)

    def test_q0_closure_environment_and_replay_bind_the_same_image(self) -> None:
        closure = json.loads((AUDITS_ROOT / "RAMANUJAN_Q0_CLOSURE.json").read_text(encoding="utf-8"))
        lock = json.loads((RUNTIME_RECORDS_ROOT / "RAMANUJAN_ENVIRONMENT_LOCK.json").read_text(encoding="utf-8"))
        build = json.loads((CONTAINER_ROOT / "BUILD_RECEIPT.json").read_text(encoding="utf-8"))
        replay = json.loads((CONTAINER_ROOT / "REPLAY_RECEIPT.json").read_text(encoding="utf-8"))
        image_ids = {
            closure["image"]["id"],
            lock["clean_proof_replay_contract"]["image_id"],
            build["image_id"],
            replay["image_id"],
        }
        self.assertEqual(len(image_ids), 1, "Q0 may not cite mutually different immutable images")
        self.assertEqual(replay["network"], "none")
        self.assertEqual(replay["exit_code"], 0)

    def test_q0_leaf_tamper_is_refused(self) -> None:
        leaf = self.tmp / "leaf.txt"
        leaf.write_bytes(b"original")
        body = {
            "schema": "hawking.ramanujan.q0_evidence_bundle.v1",
            "status": "PROVEN_OFFLINE_LEAF_CHAIN_HASH_BOUND",
            "production_authority": False,
            "research_authority": False,
            "image": {
                "id": "sha256:21114fb4b7066b5a7c535d36685211147a920233fc7544a922846056c8ec03ad",
                "size_bytes": 4_644_372_611,
            },
            "leaf_sha256": {
                "leaf.txt": __import__("hashlib").sha256(leaf.read_bytes()).hexdigest(),
            },
        }
        body["seal_sha256"] = __import__("hashlib").sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        bundle = self.tmp / "bundle.json"
        bundle.write_text(json.dumps(body), encoding="utf-8")
        verify_q0_evidence_bundle(bundle, self.tmp)
        leaf.write_bytes(b"tampered")
        with self.assertRaisesRegex(ControllerRefused, "leaf changed"):
            verify_q0_evidence_bundle(bundle, self.tmp)

    def test_formalizer_writes_a_proposal_without_promoting(self) -> None:
        result = FormalizerController().propose(
            self.stores,
            claim_id="c1",
            proof_state_id="ps1",
            lean="theorem two_plus_two : (2 : Nat) + 2 = 4 := by norm_num",
            informal_binding="two plus two equals four",
        )
        self.assertEqual(result["status"], "PROPOSED_PENDING_INDEPENDENT_FIDELITY")
        self.assertEqual(self.stores.claims["c1"].tier, Tier.ASSERTED)
        self.assertIn("ps1", self.stores.proof_states)
        self.assertFalse(hasattr(FormalizerController(), "promote"))

    def test_formalizer_refuses_statement_drift_and_proof_state_overwrite(self) -> None:
        ctl = FormalizerController()
        with self.assertRaises(ControllerRefused, msg="claim binding cannot silently drift"):
            ctl.propose(
                self.stores, claim_id="c1", proof_state_id="ps1", lean="by trivial",
                informal_binding="a different statement",
            )
        ctl.propose(
            self.stores, claim_id="c1", proof_state_id="ps1", lean="by trivial",
            informal_binding="two plus two equals four",
        )
        with self.assertRaises(ControllerRefused, msg="repair is a new row, never an overwrite"):
            ctl.propose(
                self.stores, claim_id="c1", proof_state_id="ps1", lean="by rfl",
                informal_binding="two plus two equals four",
            )

    def test_repair_records_a_candidate_but_keeps_the_source_and_tier(self) -> None:
        FormalizerController().propose(
            self.stores, claim_id="c1", proof_state_id="ps1", lean="exact missing",
            informal_binding="two plus two equals four",
        )
        before = self.stores.proof_states["ps1"].lean
        result = RepairController().propose(
            self.stores, proof_state_id="ps1", compiler_error="unknown identifier 'missing'",
        )
        self.assertEqual(result["status"], "PROPOSED_PENDING_COMPILER")
        self.assertIn("have missing", result["candidate"])
        self.assertEqual(self.stores.proof_states["ps1"].lean, before)
        self.assertEqual(self.stores.claims["c1"].tier, Tier.ASSERTED)
        self.assertEqual(self.stores.ledger.rows()[-1].kind, "proof_attempt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
