#!/usr/bin/env python3.12
"""Tests for the tiny-fixture search stack.

Each pins a property that is invisible until it is expensive: silent re-exploration,
non-deterministic ordering that makes resume impossible, a budget the search argues past.

    python3.12 -m ramanujan.test_search
"""
from __future__ import annotations

import unittest

from ramanujan.search import (
    AUTHORITY,
    CounterexampleQueue,
    PremiseRetrieval,
    ProofState,
    ProofStateDAG,
    SearchEconomics,
    best_first,
    repair_from_error,
    search_checkpoint,
    state_id,
)


def toy_tactics(s: ProofState):
    """A deterministic toy tactic set over a chain goal `a -> b -> ... -> True`.

    `peel` advances. `noop` produces a state reachable another way, so deduplication has
    something real to catch rather than a contrived duplicate.
    """
    if s.goal.startswith("g"):
        n = int(s.goal[1:])
        if n > 0:
            yield "peel", ProofState(f"g{n-1}", s.hyps)
        else:
            yield "close", ProofState("True", s.hyps)
    if len(s.hyps) < 2:
        yield "intro", ProofState(s.goal, tuple(sorted(s.hyps + ("h",))))


def h(s: ProofState) -> float:
    return float(int(s.goal[1:])) if s.goal.startswith("g") else 0.0


class TestIdentityAndDedup(unittest.TestCase):
    def test_state_id_is_order_insensitive_on_hypotheses(self) -> None:
        """Two states differing only in hypothesis ORDER are the same state. Missing this
        is how a search re-explores the same node under a different spelling."""
        self.assertEqual(state_id("g", ("a", "b")), state_id("g", ("b", "a")))

    def test_dag_deduplicates(self) -> None:
        d = ProofStateDAG()
        s = ProofState("g1")
        self.assertTrue(d.add(s))
        self.assertFalse(d.add(ProofState("g1")), "the same state must not be added twice")

    def test_search_deduplicates_a_real_diamond(self) -> None:
        """`peel` then `intro` and `intro` then `peel` both reach g3 with hypothesis h.

        The first version of this test asserted dedup > 0 on a plain descent and failed,
        because best-first drives straight to the closing goal and never explores the
        second arm of the diamond. That was the test being wrong about exploration order,
        not the DAG failing to deduplicate. Forcing exhaustive exploration tests the
        property itself.
        """
        econ = SearchEconomics(max_expansions=50, max_depth=6)
        # A constant heuristic makes the frontier explore broadly instead of diving.
        r, dag = best_first(ProofState("g2"), toy_tactics, lambda s: 0.0, econ)
        self.assertGreater(r.deduplicated, 0, "the diamond must be caught by the DAG")
        self.assertEqual(len(dag.nodes), len(set(dag.nodes)), "node ids are unique by construction")

    def test_descent_finds_the_goal(self) -> None:
        r, _ = best_first(ProofState("g4"), toy_tactics, h)
        self.assertTrue(r.found)
        self.assertEqual(r.stopped_by, "closed")


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_same_path(self) -> None:
        """A search whose result depends on dict ordering cannot be resumed or reproduced."""
        a, _ = best_first(ProofState("g5"), toy_tactics, h)
        b, _ = best_first(ProofState("g5"), toy_tactics, h)
        self.assertEqual(a.path, b.path)
        self.assertEqual(a.expansions, b.expansions)

    def test_checkpoint_is_content_addressed(self) -> None:
        _, d1 = best_first(ProofState("g3"), toy_tactics, h, SearchEconomics())
        _, d2 = best_first(ProofState("g3"), toy_tactics, h, SearchEconomics())
        c1 = search_checkpoint(d1, SearchEconomics(spent=7))
        c2 = search_checkpoint(d2, SearchEconomics(spent=7))
        self.assertEqual(c1["id"], c2["id"], "identical searches must checkpoint identically")


class TestEconomics(unittest.TestCase):
    def test_budget_stops_the_search(self) -> None:
        """The budget the search cannot talk its way past."""
        econ = SearchEconomics(max_expansions=2)
        r, _ = best_first(ProofState("g500"), toy_tactics, h, econ)
        self.assertFalse(r.found)
        self.assertIn("max_expansions", r.stopped_by)
        self.assertLessEqual(r.expansions, 2)

    def test_depth_limit_is_enforced(self) -> None:
        econ = SearchEconomics(max_expansions=10_000, max_depth=3)
        r, _ = best_first(ProofState("g50"), toy_tactics, h, econ)
        self.assertFalse(r.found, "a goal deeper than the depth limit must not be reported found")


class TestCounterexampleQueue(unittest.TestCase):
    def test_cheapest_first(self) -> None:
        """Refutation ordering is the whole point: a claim killable for one unit should
        never consume a thousand being proved."""
        q = CounterexampleQueue()
        q.push(10.0, "c1", {"n": 1})
        q.push(0.5, "c2", {"n": 2})
        q.push(3.0, "c3", {"n": 3})
        cost, cid, _ = q.pop_cheapest()
        self.assertEqual(cid, "c2")
        self.assertEqual(cost, 0.5)

    def test_empty_queue_returns_none(self) -> None:
        self.assertIsNone(CounterexampleQueue().pop_cheapest())


class TestPremiseAndRepair(unittest.TestCase):
    def test_retrieval_ranks_and_is_deterministic(self) -> None:
        r = PremiseRetrieval({"add_comm": "addition is commutative",
                              "mul_comm": "multiplication is commutative",
                              "nat_lt": "less than on naturals"})
        a = r.retrieve("addition is commutative on naturals", k=2)
        b = r.retrieve("addition is commutative on naturals", k=2)
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], "add_comm")

    def test_repair_handles_known_shapes_and_refuses_others(self) -> None:
        self.assertIsNotNone(repair_from_error("p", "unknown identifier 'foo'"))
        self.assertIsNotNone(repair_from_error("p", "unsolved goals"))
        self.assertIsNone(repair_from_error("p", "something nobody mapped"),
                          "an unmapped error must return None rather than a guess")


class TestAuthority(unittest.TestCase):
    def test_module_declares_non_production_authority(self) -> None:
        """Fixtures must never be citable as evidence about mathematics."""
        self.assertEqual(AUTHORITY, "NON_PRODUCTION_AUTHORITY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
