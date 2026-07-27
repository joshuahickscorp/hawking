"""Tiny-fixture proof/program/counterexample search: the orchestration, not the mathematics.

Everything here runs against deterministic toy theorems with a stub evaluator, and is
explicitly `NON_PRODUCTION_AUTHORITY`.  It exists to test the parts that are wrong in
subtle ways when a real prover finally arrives: whether the state DAG deduplicates, whether
best-first actually orders, whether search economics stop a runaway branch, whether a
resumed search reaches the same place as an uninterrupted one.

Those are the failures that are invisible until they are expensive.  A real Lean is not
installed (`RAMANUJAN_TOOLCHAIN_SELFTEST.json`: 10 of 12 components missing), so the
alternative to a fixture is nothing at all.

Nothing in this module may be cited as evidence about any mathematical claim.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

AUTHORITY = "NON_PRODUCTION_AUTHORITY"


def state_id(goal: str, hyps: tuple[str, ...]) -> str:
    """Content address of a proof state. Two states that are the same ARE the same.

    Deduplication is keyed on this. Re-exploring an identical state is the cheapest large
    waste a proof search commits, and it is invisible without an explicit identity.
    """
    body = json.dumps({"goal": goal, "hyps": sorted(hyps)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProofState:
    goal: str
    hyps: tuple[str, ...] = ()

    @property
    def sid(self) -> str:
        return state_id(self.goal, self.hyps)

    def closed(self) -> bool:
        return self.goal == "True" or self.goal in self.hyps


@dataclass
class SearchEconomics:
    """A budget the search cannot talk its way past.

    `max_expansions` is the one that matters: a search that always finds a reason to
    continue is how a research system spends a night on one branch.
    """

    max_expansions: int = 200
    max_depth: int = 12
    spent: int = 0

    def may_expand(self) -> bool:
        return self.spent < self.max_expansions

    def charge(self) -> None:
        self.spent += 1


@dataclass
class ProofStateDAG:
    """States plus edges. A DAG rather than a tree, because the same state is reachable
    by different tactic sequences and must not be explored twice."""

    nodes: dict[str, ProofState] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (from, tactic, to)
    _seen: set[str] = field(default_factory=set)

    def add(self, s: ProofState) -> bool:
        """Returns True when the state is new. False means it was deduplicated."""
        if s.sid in self._seen:
            return False
        self._seen.add(s.sid)
        self.nodes[s.sid] = s
        return True

    def link(self, a: ProofState, tactic: str, b: ProofState) -> None:
        self.edges.append((a.sid, tactic, b.sid))

    @property
    def dedup_hits(self) -> int:
        return len(self.edges) - (len(self.nodes) - 1) if self.nodes else 0


Tactic = Callable[[ProofState], Iterable[tuple[str, ProofState]]]


@dataclass
class SearchResult:
    found: bool
    path: list[str]
    expansions: int
    deduplicated: int
    stopped_by: str


def best_first(
    start: ProofState,
    tactics: Tactic,
    heuristic: Callable[[ProofState], float],
    economics: SearchEconomics | None = None,
    dag: ProofStateDAG | None = None,
) -> tuple[SearchResult, ProofStateDAG]:
    """Best-first proof search with content-addressed deduplication.

    Deterministic: ties break on the state id, so the same inputs give the same path on
    every run and on every machine. A search whose result depends on dict ordering cannot
    be resumed or reproduced, and both are requirements here.
    """
    econ = economics or SearchEconomics()
    dag = dag or ProofStateDAG()
    dag.add(start)
    heap: list[tuple[float, str, ProofState, list[str]]] = [
        (heuristic(start), start.sid, start, [])
    ]
    dedup = 0

    while heap:
        if not econ.may_expand():
            return SearchResult(False, [], econ.spent, dedup, "economics: max_expansions"), dag
        _, _, cur, path = heapq.heappop(heap)
        if cur.closed():
            return SearchResult(True, path, econ.spent, dedup, "closed"), dag
        if len(path) >= econ.max_depth:
            continue
        econ.charge()
        for tactic, nxt in tactics(cur):
            if not dag.add(nxt):
                dedup += 1
                continue
            dag.link(cur, tactic, nxt)
            heapq.heappush(heap, (heuristic(nxt), nxt.sid, nxt, path + [tactic]))

    return SearchResult(False, [], econ.spent, dedup, "exhausted"), dag


@dataclass
class CounterexampleQueue:
    """Candidate refutations, cheapest first.

    Refutation is prioritized over proof deliberately: a claim that can be killed for one
    unit of compute should never consume a thousand trying to prove it. This is the
    Cheapest Falsifier's queue, and the reason the mechanism exists at all -- '2 + 2 =' was
    one forward pass and refuted an entire substrate.
    """

    items: list[tuple[float, str, dict]] = field(default_factory=list)

    def push(self, cost: float, claim_id: str, witness: dict) -> None:
        heapq.heappush(self.items, (cost, claim_id, witness))

    def pop_cheapest(self) -> tuple[float, str, dict] | None:
        return heapq.heappop(self.items) if self.items else None


@dataclass
class PremiseRetrieval:
    """Premise selection over a tiny fixture corpus. Interface first, ranking later."""

    corpus: dict[str, str] = field(default_factory=dict)

    def retrieve(self, goal: str, k: int = 3) -> list[tuple[str, float]]:
        """Token-overlap scoring. Deliberately crude and honestly labelled: this is a
        placeholder for a trained retriever and must never be reported as one."""
        gt = set(goal.lower().split())
        scored = [
            (name, len(gt & set(text.lower().split())) / max(1, len(gt)))
            for name, text in self.corpus.items()
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def repair_from_error(proof: str, error: str) -> str | None:
    """Compiler-feedback repair interface, fixture-level.

    Returns a repaired proof or None. Real repair needs a real compiler's real errors;
    this maps two synthetic error shapes so the loop can be tested end to end.
    """
    if "unknown identifier" in error:
        missing = error.split("unknown identifier")[-1].strip().strip("'\"")
        return f"have {missing} := by simp\n{proof}" if missing else None
    if "unsolved goals" in error:
        return f"{proof}\n  simp_all"
    return None


def search_checkpoint(dag: ProofStateDAG, econ: SearchEconomics) -> dict:
    """Content-addressed search checkpoint, so a resume is verifiable rather than hopeful."""
    body = {
        "nodes": sorted(dag.nodes),
        "edges": sorted(dag.edges),
        "spent": econ.spent,
    }
    blob = json.dumps(body, sort_keys=True)
    return {"id": hashlib.sha256(blob.encode()).hexdigest()[:16], "body": body}
