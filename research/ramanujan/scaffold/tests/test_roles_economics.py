#!/usr/bin/env python3.12
"""Tests for roles, economics, the Limit Registry, and the F0/F1 forge stubs.

Pins the four properties the deliverable requires:

  1. a role cannot exceed its capability
  2. a generator cannot promote
  3. a branch that exhausts its budget halts and records why
  4. the Limit Registry is consulted rather than decorative

    python3.12 -m ramanujan.test_roles_economics
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ramanujan.economics import (
    REWARDS,
    BranchAccount,
    BranchHalted,
    run_branch_search,
)
from ramanujan.evidence import Tier, VerifierEvent
from ramanujan.forge import f0_diagnose, f1_train_premise_retrieval, run_f0_f1
from ramanujan.ledger import Ledger
from ramanujan.limits import LimitBlocked, LimitRegistry
from ramanujan.roles import (
    GENERATOR_IDS,
    ROLE_CATALOG,
    CapabilityRefused,
    RoleCapability,
    RoleSession,
    capability_for,
    generators_may_not_promote,
)
from ramanujan.search import AUTHORITY, PremiseRetrieval, ProofState, SearchEconomics
from ramanujan.stores import Stores


def _ledger(tmp: Path) -> Ledger:
    n = [0]

    def clock() -> str:
        n[0] += 1
        return f"2026-01-01T00:00:{n[0]:02d}Z"

    return Ledger(tmp / "ledger.jsonl", clock=clock)


def _stores(tmp: Path) -> Stores:
    return Stores(ledger=_ledger(tmp))


class TestRoleCapabilities(unittest.TestCase):
    def test_every_generator_has_may_promote_false(self) -> None:
        self.assertTrue(generators_may_not_promote())
        for rid in GENERATOR_IDS:
            self.assertFalse(ROLE_CATALOG[rid].may_promote, rid)
            self.assertFalse(capability_for(rid).may_promote, rid)

    def test_capability_is_non_widening(self) -> None:
        """Same shape as JobCapability: derive only, no grant/add surface."""
        cap = capability_for("conjecturer")
        self.assertFalse(hasattr(cap, "grant"))
        self.assertFalse(hasattr(cap, "add"))
        self.assertFalse(hasattr(cap, "grant_tool"))
        with self.assertRaises(CapabilityRefused):
            RoleCapability.derive_subset(
                ROLE_CATALOG["conjecturer"],
                frozenset({"write_claim", "promote_claim"}),
            )

    def test_role_cannot_exceed_its_capability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stores = _stores(tmp)
            conj = RoleSession("conjecturer", stores)
            # Conjecturer may write claims.
            conj.write_claim("c1", "2 + 2 = 4")
            self.assertIn("c1", stores.claims)
            # Conjecturer may not write literature, budgets, or lean.
            with self.assertRaises(CapabilityRefused):
                conj.write_literature("p1", {"title": "nope"})
            with self.assertRaises(CapabilityRefused):
                conj.write_budget("b1", {"max_expansions": 10})
            with self.assertRaises(CapabilityRefused):
                conj.write_lean("c1", "theorem t : true := trivial")

    def test_skeptic_cannot_write_claims(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            RoleSession("conjecturer", stores).write_claim("c1", "p")
            sk = RoleSession("skeptic", stores)
            with self.assertRaises(CapabilityRefused):
                sk.write_claim("c2", "q")
            sk.write_objection("c1", "counterexample exists")
            self.assertTrue(stores.claims["c1"].in_graveyard)

    def test_economist_never_sees_claim_content(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            eco = RoleSession("economist", stores)
            with self.assertRaises(CapabilityRefused) as ctx:
                eco.write_budget("b1", {"max_expansions": 5, "statement": "secret claim"})
            self.assertIn("claim content", str(ctx.exception))
            # Clean grant is fine.
            eco.write_budget("b1", {"max_expansions": 5})


class TestGeneratorCannotPromote(unittest.TestCase):
    def test_each_generator_require_promote_refuses(self) -> None:
        for rid in sorted(GENERATOR_IDS):
            cap = capability_for(rid)
            with self.assertRaises(CapabilityRefused, msg=rid):
                cap.require_promote()

    def test_conjecturer_session_cannot_tribunal_admit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            RoleSession("conjecturer", stores).write_claim("c1", "goldbach")
            # Lift the claim to tier 2 so the *stores* gate would pass if the role could.
            stores.record_evidence(
                "c1", VerifierEvent("computation", "computationalist", None, True, {})
            )
            stores.record_evidence(
                "c1", VerifierEvent("fidelity_assessment", "skeptic", None, True, {})
            )
            conj = RoleSession("conjecturer", stores)
            with self.assertRaises(CapabilityRefused):
                conj.tribunal_admit("c1", human_expert_gate=True)
            self.assertFalse(stores.claims["c1"].admitted)

    def test_tribunal_can_admit_when_gates_hold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            RoleSession("conjecturer", stores).write_claim("c1", "goldbach")
            stores.record_evidence(
                "c1", VerifierEvent("computation", "computationalist", None, True, {})
            )
            stores.record_evidence(
                "c1", VerifierEvent("fidelity_assessment", "skeptic", None, True, {})
            )
            RoleSession("tribunal", stores).tribunal_admit("c1", human_expert_gate=True)
            self.assertTrue(stores.claims["c1"].admitted)

    def test_computationalist_writes_evidence_but_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            RoleSession("conjecturer", stores).write_claim("c1", "sum")
            comp = RoleSession("computationalist", stores)
            tier = comp.write_evidence(
                "c1",
                VerifierEvent("computation", "computationalist", None, True, {"n": 1}),
            )
            self.assertEqual(tier, Tier.EMPIRICALLY_SUPPORTED)
            with self.assertRaises(CapabilityRefused):
                comp.tribunal_admit("c1", human_expert_gate=True)


class TestBranchBudgetStopRule(unittest.TestCase):
    def test_exhausted_branch_halts_and_records_why(self) -> None:
        def tactics(s: ProofState):
            if s.goal.startswith("g"):
                n = int(s.goal[1:])
                if n > 0:
                    yield "peel", ProofState(f"g{n - 1}", s.hyps)
                else:
                    yield "close", ProofState("True", s.hyps)

        branch = BranchAccount(
            "b_exhaust",
            economics=SearchEconomics(max_expansions=2, max_depth=100),
        )
        result, branch = run_branch_search(
            branch,
            ProofState("g50"),
            tactics,
            heuristic=lambda s: float(int(s.goal[1:])) if s.goal.startswith("g") else 0.0,
        )
        self.assertFalse(result.found)
        self.assertIn("max_expansions", result.stopped_by)
        self.assertTrue(branch.is_halted())
        self.assertIsNotNone(branch.halt)
        assert branch.halt is not None
        self.assertIn("max_expansions", branch.halt.reason)
        self.assertEqual(branch.halt.spent, branch.economics.spent)
        self.assertLessEqual(branch.economics.spent, 2)
        # Clean exhaustion is rewarded under the schedule.
        self.assertGreaterEqual(branch.value_earned, REWARDS["clean_branch_exhaustion"])
        score = branch.score()
        self.assertEqual(score["halt_reason"], branch.halt.reason)

    def test_charge_after_halt_raises(self) -> None:
        branch = BranchAccount("b", economics=SearchEconomics(max_expansions=1))
        branch.charge()
        with self.assertRaises(BranchHalted):
            branch.charge()
        self.assertTrue(branch.is_halted())
        self.assertIn("max_expansions", branch.why_cannot_spend())

    def test_manual_stop_records_reason(self) -> None:
        branch = BranchAccount("b")
        rec = branch.stop("operator: abandoned", detail={"by": "adversary"})
        self.assertEqual(rec.reason, "operator: abandoned")
        self.assertEqual(branch.halt.detail["by"], "adversary")


class TestLimitRegistryConsulted(unittest.TestCase):
    def test_research_authorized_stays_false(self) -> None:
        reg = LimitRegistry()
        self.assertFalse(reg.research_authorized())
        self.assertFalse(hasattr(reg, "set_research_authorized"))
        self.assertFalse(hasattr(reg, "authorize_research"))

    def test_consult_blocks_run_research_and_is_logged(self) -> None:
        reg = LimitRegistry()
        v = reg.consult("run_research", role_id="director")
        self.assertFalse(v.allowed)
        # Either launch or research fence is sufficient; both list run_research.
        self.assertIn(v.blocking_limit, {"L-LAUNCH-01", "L-RESEARCH-01"})
        self.assertTrue(reg.consult_log, "consult must leave a log so it is not decorative")
        self.assertEqual(reg.consult_log[-1]["action"], "run_research")
        self.assertFalse(reg.consult_log[-1]["allowed"])
        # And the research-authorized flag itself stays false regardless of which fence hit first.
        self.assertFalse(reg.research_authorized())

    def test_math_preserve_teacher_traces_blocked(self) -> None:
        reg = LimitRegistry()
        v = reg.consult("teacher_trace_from_math_preserve", role_id="librarian")
        self.assertFalse(v.allowed)
        self.assertEqual(v.blocking_limit, "L-TEACHER-01")

    def test_role_write_consults_registry(self) -> None:
        """When a registry is bound, a blocked action never reaches the store."""
        with tempfile.TemporaryDirectory() as td:
            stores = _stores(Path(td))
            reg = LimitRegistry()
            # write_claim is not blocked by default limits -- consult still happens.
            conj = RoleSession("conjecturer", stores, limits=reg)
            n_before = len(reg.consult_log)
            conj.write_claim("c1", "x")
            self.assertGreater(len(reg.consult_log), n_before)
            self.assertEqual(reg.consult_log[-1]["action"], "write_claim")
            self.assertTrue(reg.consult_log[-1]["allowed"])

    def test_require_raises_on_blocked_action(self) -> None:
        reg = LimitRegistry()
        with self.assertRaises(LimitBlocked):
            reg.require("network", role_id="librarian")


class TestF0F1Forge(unittest.TestCase):
    def test_f0_instrument_inventory(self) -> None:
        receipt = f0_diagnose()
        self.assertEqual(receipt["stage"], "F0")
        self.assertEqual(receipt["authority"], AUTHORITY)
        self.assertFalse(receipt["RAMANUJAN_RESEARCH_AUTHORIZED"])
        self.assertTrue(receipt["ready_for_f1"])
        self.assertTrue(
            receipt["components"]["roles"]["generators_may_not_promote"]
        )
        self.assertEqual(
            receipt["components"]["math_preserve_teacher_traces"]["status"], "REFUSED"
        )
        self.assertEqual(
            receipt["components"]["premise_retrieval"]["label"], "crude_token_overlap"
        )

    def test_f1_trains_and_relabels_honestly(self) -> None:
        receipt = f1_train_premise_retrieval(steps=30)
        self.assertEqual(receipt["stage"], "F1")
        self.assertEqual(receipt["authority"], AUTHORITY)
        self.assertFalse(receipt["RAMANUJAN_RESEARCH_AUTHORIZED"])
        self.assertTrue(receipt["teacher_trace_blocked"])
        self.assertTrue(receipt["retriever"]["trained"])
        self.assertEqual(receipt["retriever"]["label"], "trainable_trained_on_fixtures")
        # Registry was consulted (not decorative).
        actions = {c["action"] for c in receipt["limit_consults"]}
        self.assertIn("run_research", actions)
        self.assertIn("teacher_trace_from_math_preserve", actions)
        # Quality is measured; training should not make MRR worse on its own train set.
        self.assertGreaterEqual(
            receipt["quality_after"]["mean_reciprocal_rank"],
            receipt["quality_before"]["mean_reciprocal_rank"] - 1e-9,
        )
        # Live object ranks a known goal to its relevant premise.
        ret: PremiseRetrieval = receipt["retriever_object"]
        top = ret.retrieve("prove a + b = b + a on naturals", k=1)
        self.assertEqual(top[0][0], "add_comm")

    def test_untrained_retriever_is_honestly_labelled(self) -> None:
        r = PremiseRetrieval(corpus={"a": "addition is commutative"})
        self.assertFalse(r.trained)
        self.assertEqual(r.label, "crude_token_overlap")
        self.assertEqual(r.authority, AUTHORITY)

    def test_run_f0_f1_combined(self) -> None:
        receipt = run_f0_f1(steps=20)
        self.assertEqual(receipt["status"], "F0_F1_COMPLETE_ON_FIXTURES")
        self.assertFalse(receipt["RAMANUJAN_RESEARCH_AUTHORIZED"])
        self.assertNotIn("retriever_object", receipt["f1"])


class TestSearchPremiseStillDeterministic(unittest.TestCase):
    """Existing PremiseRetrieval contract: deterministic ranking before/after train API."""

    def test_crude_ranking_stable(self) -> None:
        r = PremiseRetrieval(
            {
                "add_comm": "addition is commutative",
                "mul_comm": "multiplication is commutative",
                "nat_lt": "less than on naturals",
            }
        )
        a = r.retrieve("addition is commutative on naturals", k=2)
        b = r.retrieve("addition is commutative on naturals", k=2)
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], "add_comm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
