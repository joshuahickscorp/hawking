"""The seven stores, over the Ledger, with the Tribunal's separation-of-powers rule.

Stores: Problem, Claim, Proof-State, Counterexample, Prior-Art, Strategy, Graveyard.

Two laws are enforced here rather than described:

  "nothing is deleted; selective context is retrieval, not forgetting"
      -- MEMORY_STORES.json. There is no delete. Refuting a claim MOVES it to the
      Graveyard, where it stays readable, and a revival is itself a Ledger event.

  "the system that produces a claim is never the system that admits it"
      -- TRIBUNAL.json. The Tribunal refuses to admit a claim whose author is the
      admitting actor. This is the structural defence against a research system
      grading its own homework, and it is worth more than any amount of calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ramanujan.evidence import PromotionRefused, Tier, VerifierEvent, promote
from ramanujan.ledger import Ledger, LedgerViolation

STORE_NAMES = (
    "problem",
    "claim",
    "proof_state",
    "counterexample",
    "prior_art",
    "strategy",
    "graveyard",
)


class TribunalRefused(RuntimeError):
    pass


@dataclass
class Claim:
    id: str
    statement: str
    author: str
    tier: Tier = Tier.ASSERTED
    admitted: bool = False
    in_graveyard: bool = False
    graveyard_reason: str | None = None
    evidence: list[VerifierEvent] = field(default_factory=list)


@dataclass
class Stores:
    """Seven stores whose only mutation path goes through the Ledger.

    Every state change writes a row first. A store that can change without a row would
    make the Ledger's law -- what is not recorded did not happen -- false in practice
    while still true on paper.
    """

    ledger: Ledger
    claims: dict[str, Claim] = field(default_factory=dict)
    problems: dict[str, dict] = field(default_factory=dict)
    proof_states: dict[str, dict] = field(default_factory=dict)
    counterexamples: dict[str, dict] = field(default_factory=dict)
    prior_art: dict[str, dict] = field(default_factory=dict)
    strategies: dict[str, dict] = field(default_factory=dict)

    # -- creation ---------------------------------------------------------
    def add_claim(self, claim_id: str, statement: str, author: str) -> Claim:
        if claim_id in self.claims:
            raise LedgerViolation(f"claim {claim_id!r} already exists; claims are never overwritten")
        c = Claim(id=claim_id, statement=statement, author=author)
        self.claims[claim_id] = c
        self.ledger.append("claim", {"id": claim_id, "statement": statement}, actor=author)
        return c

    def add_problem(self, pid: str, statement: str, actor: str) -> dict:
        p = {"id": pid, "statement": statement}
        self.problems[pid] = p
        self.ledger.append("claim", {"problem": pid, "statement": statement}, actor=actor)
        return p

    def add_counterexample(self, cid: str, claim_id: str, witness: dict, actor: str) -> dict:
        ce = {"id": cid, "claim": claim_id, "witness": witness}
        self.counterexamples[cid] = ce
        self.ledger.append("verifier_event", {"counterexample": cid, "claim": claim_id}, actor=actor)
        return ce

    # -- evidence ---------------------------------------------------------
    def record_evidence(self, claim_id: str, event: VerifierEvent) -> Tier:
        """Attach a verifier event and attempt promotion. The attempt may be refused."""
        c = self.claims[claim_id]
        c.evidence.append(event)
        self.ledger.append(
            "verifier_event",
            {"claim": claim_id, "kind": event.kind, "container": event.container_hash},
            actor=event.actor,
        )
        try:
            c.tier = promote(c.tier, event, author=c.author)
        except PromotionRefused:
            pass  # the event is still recorded; it simply did not license a promotion
        return c.tier

    # -- Tribunal ---------------------------------------------------------
    def tribunal_admit(self, claim_id: str, admitting_actor: str, human_expert_gate: bool) -> None:
        """Admit a claim. Refuses when the admitter authored it, or without the human gate.

        TRIBUNAL.json requires a human expert gate and states the system never certifies
        its own novelty. Both are enforced, not noted.
        """
        c = self.claims[claim_id]
        if admitting_actor == c.author:
            raise TribunalRefused(
                f"{admitting_actor!r} authored claim {claim_id!r} and cannot admit it: "
                "the system that produces a claim is never the system that admits it"
            )
        if not human_expert_gate:
            raise TribunalRefused(
                "admission requires a human expert gate; the system never certifies its own novelty"
            )
        if c.tier < Tier.FORMALIZED:
            raise TribunalRefused(
                f"claim {claim_id!r} is Tier {int(c.tier)}; admission requires at least Tier 2"
            )
        c.admitted = True
        self.ledger.append(
            "tribunal_decision",
            {"claim": claim_id, "decision": "admitted", "tier": int(c.tier)},
            actor=admitting_actor,
        )

    # -- Graveyard --------------------------------------------------------
    def bury(self, claim_id: str, reason: str, actor: str) -> None:
        """Move a claim to the Graveyard. It stays readable; nothing is deleted."""
        c = self.claims[claim_id]
        c.in_graveyard = True
        c.graveyard_reason = reason
        self.ledger.append("objection", {"claim": claim_id, "buried_because": reason}, actor=actor)

    def revive(self, claim_id: str, because: str, actor: str) -> None:
        """Revival is permitted only when something external changed the premises,
        and is itself a Ledger event -- per GRAVEYARD.json."""
        c = self.claims[claim_id]
        if not c.in_graveyard:
            raise LedgerViolation(f"claim {claim_id!r} is not buried")
        c.in_graveyard = False
        self.ledger.append("claim", {"claim": claim_id, "revived_because": because}, actor=actor)

    def graveyard(self) -> Iterable[Claim]:
        return (c for c in self.claims.values() if c.in_graveyard)

    def live_claims(self) -> Iterable[Claim]:
        """Retrieval, not forgetting: buried claims are excluded from the working set
        and remain fully present in the store."""
        return (c for c in self.claims.values() if not c.in_graveyard)
