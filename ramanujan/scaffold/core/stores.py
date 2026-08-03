"""The seven stores, over the Ledger, with the Tribunal's separation-of-powers rule.

Stores (handoff contract): Problem, Claim, Proof-State, Counterexample, Prior-Art,
Strategy, Graveyard.

Two laws are enforced here rather than described:

  "nothing is deleted; selective context is retrieval, not forgetting"
      -- MEMORY_STORES.json / GRAVEYARD.json. There is no delete. Refuting a claim
      MOVES it to the Graveyard, where it stays readable. A free resurrection is
      refused; revival is allowed only when a post-burial ledger event changed the
      premises (verifier_event or literature_query), and the revival is itself a
      Ledger event.

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

# Ledger kinds that can change a buried claim's premises and license revival.
PREMISE_CHANGE_KINDS = frozenset({"verifier_event", "literature_query"})


class TribunalRefused(RuntimeError):
    pass


class GraveyardRefused(RuntimeError):
    """Raised when a caller tries to delete, overwrite, or free-resurrect a burial."""


class StoreRefused(RuntimeError):
    """Raised for store-level invariant violations (overwrite, missing parent, etc.)."""


@dataclass
class Problem:
    id: str
    statement: str
    actor: str
    meta: dict = field(default_factory=dict)


@dataclass
class Claim:
    id: str
    statement: str
    author: str
    tier: Tier = Tier.ASSERTED
    admitted: bool = False
    in_graveyard: bool = False
    graveyard_reason: str | None = None
    buried_at_seq: int | None = None
    evidence: list[VerifierEvent] = field(default_factory=list)


@dataclass
class ProofStateEntry:
    id: str
    claim_id: str
    lean: str
    actor: str
    meta: dict = field(default_factory=dict)


@dataclass
class Counterexample:
    id: str
    claim_id: str
    witness: dict
    actor: str


@dataclass
class PriorArt:
    id: str
    body: dict
    actor: str


@dataclass
class Strategy:
    id: str
    body: dict
    actor: str


@dataclass
class GraveyardRecord:
    """A burial. The claim object remains in claims[]; this is the audit trail entry."""

    claim_id: str
    reason: str
    actor: str
    buried_at_seq: int
    revived_at_seq: int | None = None
    revival_premise_seq: int | None = None
    revival_because: str | None = None

    @property
    def is_dead(self) -> bool:
        return self.revived_at_seq is None


@dataclass
class Stores:
    """Seven stores whose only mutation path goes through the Ledger.

    Every state change writes a row first. A store that can change without a row would
    make the Ledger's law -- what is not recorded did not happen -- false in practice
    while still true on paper.

    There is deliberately no `delete`, `remove`, `edit`, or `update` method on this
    class. Corrections are supersessions on the Ledger; refutations are burials.
    """

    ledger: Ledger
    claims: dict[str, Claim] = field(default_factory=dict)
    problems: dict[str, Problem] = field(default_factory=dict)
    proof_states: dict[str, ProofStateEntry] = field(default_factory=dict)
    counterexamples: dict[str, Counterexample] = field(default_factory=dict)
    prior_art: dict[str, PriorArt] = field(default_factory=dict)
    strategies: dict[str, Strategy] = field(default_factory=dict)
    # Graveyard is not a shadow copy: keys are claim ids currently or historically buried.
    graveyard_records: dict[str, list[GraveyardRecord]] = field(default_factory=dict)

    # -- construction surface: no delete / edit ---------------------------------
    def __post_init__(self) -> None:
        # Fail loud if a future patch adds a destructive surface.
        for forbidden in ("delete", "remove", "edit", "update", "pop", "clear"):
            if hasattr(type(self), forbidden) and callable(getattr(type(self), forbidden)):
                # Only flag methods defined on Stores itself, not dict helpers.
                owner = getattr(type(self), forbidden)
                if getattr(owner, "__qualname__", "").startswith("Stores."):
                    raise RuntimeError(
                        f"Stores must not expose {forbidden!r}; burial is not deletion"
                    )

    # -- Problem --------------------------------------------------------------
    def add_problem(self, pid: str, statement: str, actor: str, **meta) -> Problem:
        if pid in self.problems:
            raise StoreRefused(f"problem {pid!r} already exists; stores never overwrite")
        p = Problem(id=pid, statement=statement, actor=actor, meta=dict(meta))
        self.problems[pid] = p
        self.ledger.append(
            "claim",
            {"store": "problem", "id": pid, "statement": statement},
            actor=actor,
        )
        return p

    # -- Claim ----------------------------------------------------------------
    def add_claim(self, claim_id: str, statement: str, author: str) -> Claim:
        if claim_id in self.claims:
            raise LedgerViolation(
                f"claim {claim_id!r} already exists; claims are never overwritten"
            )
        c = Claim(id=claim_id, statement=statement, author=author)
        self.claims[claim_id] = c
        self.ledger.append(
            "claim",
            {"store": "claim", "id": claim_id, "statement": statement},
            actor=author,
        )
        return c

    # -- Proof-State ----------------------------------------------------------
    def add_proof_state(
        self,
        ps_id: str,
        claim_id: str,
        lean: str,
        actor: str,
        **meta,
    ) -> ProofStateEntry:
        if claim_id not in self.claims:
            raise StoreRefused(f"proof state needs an existing claim; {claim_id!r} missing")
        if self.claims[claim_id].in_graveyard:
            raise GraveyardRefused(
                f"claim {claim_id!r} is buried; proof-state writes go to a live claim only. "
                "Burial is not deletion -- revive with a premise change first."
            )
        if ps_id in self.proof_states:
            raise StoreRefused(f"proof_state {ps_id!r} already exists; stores never overwrite")
        ps = ProofStateEntry(
            id=ps_id, claim_id=claim_id, lean=lean, actor=actor, meta=dict(meta)
        )
        self.proof_states[ps_id] = ps
        self.ledger.append(
            "formalization",
            {"store": "proof_state", "id": ps_id, "claim": claim_id, "bytes": len(lean)},
            actor=actor,
        )
        return ps

    # -- Counterexample -------------------------------------------------------
    def add_counterexample(
        self, cid: str, claim_id: str, witness: dict, actor: str
    ) -> Counterexample:
        if claim_id not in self.claims:
            raise StoreRefused(f"counterexample needs an existing claim; {claim_id!r} missing")
        if cid in self.counterexamples:
            raise StoreRefused(f"counterexample {cid!r} already exists; stores never overwrite")
        ce = Counterexample(id=cid, claim_id=claim_id, witness=dict(witness), actor=actor)
        self.counterexamples[cid] = ce
        self.ledger.append(
            "verifier_event",
            {"store": "counterexample", "id": cid, "claim": claim_id, "witness": witness},
            actor=actor,
        )
        return ce

    # -- Prior-Art ------------------------------------------------------------
    def add_prior_art(self, entry_id: str, body: dict, actor: str) -> PriorArt:
        if entry_id in self.prior_art:
            raise StoreRefused(f"prior_art {entry_id!r} already exists; stores never overwrite")
        entry = PriorArt(id=entry_id, body=dict(body), actor=actor)
        self.prior_art[entry_id] = entry
        self.ledger.append(
            "literature_query",
            {"store": "prior_art", "id": entry_id, "keys": sorted(body)},
            actor=actor,
        )
        return entry

    # -- Strategy -------------------------------------------------------------
    def add_strategy(self, sid: str, body: dict, actor: str) -> Strategy:
        if sid in self.strategies:
            raise StoreRefused(f"strategy {sid!r} already exists; stores never overwrite")
        s = Strategy(id=sid, body=dict(body), actor=actor)
        self.strategies[sid] = s
        self.ledger.append(
            "checkpoint",
            {"store": "strategy", "id": sid, "keys": sorted(body)},
            actor=actor,
        )
        return s

    # -- inventory ------------------------------------------------------------
    def store_inventory(self) -> dict:
        """Counts per store. Used by Forge F0 so the inventory is data, not prose."""
        return {
            "problem": len(self.problems),
            "claim": len(self.claims),
            "proof_state": len(self.proof_states),
            "counterexample": len(self.counterexamples),
            "prior_art": len(self.prior_art),
            "strategy": len(self.strategies),
            "graveyard": sum(1 for c in self.claims.values() if c.in_graveyard),
            "names": list(STORE_NAMES),
        }

    # -- evidence -------------------------------------------------------------
    def record_evidence(self, claim_id: str, event: VerifierEvent) -> Tier:
        """Attach a verifier event and attempt promotion. The attempt may be refused.

        The event is always recorded. Promotion uses `promote`, which raises rather
        than silently licensing a tier the evidence does not earn; the refusal is
        swallowed here so a mismatched event does not discard the audit row. Callers
        that need the refusal as an error use `attempt_promotion`.
        """
        c = self.claims[claim_id]
        if c.in_graveyard:
            raise GraveyardRefused(
                f"claim {claim_id!r} is buried; evidence attaches only after revival. "
                "A buried claim stays auditable and stays dead."
            )
        c.evidence.append(event)
        self.ledger.append(
            "verifier_event",
            {
                "store": "claim",
                "claim": claim_id,
                "kind": event.kind,
                "container": event.container_hash,
            },
            actor=event.actor,
        )
        try:
            c.tier = promote(c.tier, event, author=c.author)
        except PromotionRefused:
            pass  # the event is still recorded; it simply did not license a promotion
        return c.tier

    def attempt_promotion(self, claim_id: str, event: VerifierEvent) -> Tier:
        """Explicit promotion attempt. Refuses rather than ignoring insufficient evidence.

        Unlike `record_evidence`, a refused promotion raises. Use this when a caller
        is claiming a tier move, not merely attaching an event.
        """
        c = self.claims[claim_id]
        if c.in_graveyard:
            raise GraveyardRefused(
                f"claim {claim_id!r} is buried; promotion is refused while dead"
            )
        c.evidence.append(event)
        self.ledger.append(
            "verifier_event",
            {
                "store": "claim",
                "claim": claim_id,
                "kind": event.kind,
                "container": event.container_hash,
                "explicit_promotion": True,
            },
            actor=event.actor,
        )
        # Raises PromotionRefused on insufficient evidence -- does not ignore.
        c.tier = promote(c.tier, event, author=c.author)
        return c.tier

    # -- Tribunal -------------------------------------------------------------
    def tribunal_admit(
        self, claim_id: str, admitting_actor: str, human_expert_gate: bool
    ) -> None:
        """Admit a claim. Refuses when the admitter authored it, or without the human gate.

        TRIBUNAL.json requires a human expert gate and states the system never certifies
        its own novelty. Both are enforced, not noted. Buried claims cannot be admitted.
        """
        c = self.claims[claim_id]
        if c.in_graveyard:
            raise TribunalRefused(
                f"claim {claim_id!r} is in the Graveyard; the dead are not admitted. "
                "Burial is not deletion -- revive only with a premise-changing event."
            )
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

    # -- Graveyard ------------------------------------------------------------
    def bury(self, claim_id: str, reason: str, actor: str) -> GraveyardRecord:
        """Move a claim to the Graveyard. It stays readable; nothing is deleted."""
        c = self.claims[claim_id]
        if c.in_graveyard:
            raise GraveyardRefused(f"claim {claim_id!r} is already buried")
        row = self.ledger.append(
            "objection",
            {"store": "graveyard", "claim": claim_id, "buried_because": reason},
            actor=actor,
        )
        c.in_graveyard = True
        c.graveyard_reason = reason
        c.buried_at_seq = row.seq
        c.admitted = False  # admission does not survive burial
        rec = GraveyardRecord(
            claim_id=claim_id,
            reason=reason,
            actor=actor,
            buried_at_seq=row.seq,
        )
        self.graveyard_records.setdefault(claim_id, []).append(rec)
        return rec

    def revive(
        self,
        claim_id: str,
        because: str,
        actor: str,
        premise_change_seq: int | None = None,
    ) -> GraveyardRecord:
        """Revive only when a post-burial premise-changing ledger event exists.

        GRAVEYARD.json: revival when a new verifier event or literature result changes
        premises; revival is itself a Ledger event. A free resurrection (no premise
        change) is refused -- buried claims stay dead.
        """
        c = self.claims[claim_id]
        if not c.in_graveyard:
            raise LedgerViolation(f"claim {claim_id!r} is not buried")
        if premise_change_seq is None:
            raise GraveyardRefused(
                f"cannot resurrect claim {claim_id!r} without a premise-changing ledger "
                f"event (verifier_event or literature_query after burial). "
                "Burial is not deletion, and free resurrection is not revival."
            )
        self._require_valid_premise_change(claim_id, premise_change_seq)
        row = self.ledger.append(
            "claim",
            {
                "store": "graveyard",
                "claim": claim_id,
                "revived_because": because,
                "premise_change_seq": premise_change_seq,
            },
            actor=actor,
        )
        c.in_graveyard = False
        recs = self.graveyard_records.get(claim_id) or []
        if not recs:
            raise LedgerViolation(f"claim {claim_id!r} has no graveyard record")
        active = recs[-1]
        active.revived_at_seq = row.seq
        active.revival_premise_seq = premise_change_seq
        active.revival_because = because
        return active

    def _require_valid_premise_change(self, claim_id: str, seq: int) -> None:
        rows = self.ledger.rows()
        if not 0 <= seq < len(rows):
            raise GraveyardRefused(
                f"premise_change_seq {seq} does not refer to a ledger row"
            )
        row = rows[seq]
        if row.kind not in PREMISE_CHANGE_KINDS:
            raise GraveyardRefused(
                f"row {seq} kind={row.kind!r} cannot change premises; "
                f"need one of {sorted(PREMISE_CHANGE_KINDS)}"
            )
        c = self.claims[claim_id]
        if c.buried_at_seq is None or seq <= c.buried_at_seq:
            raise GraveyardRefused(
                f"premise_change_seq {seq} must be strictly after burial seq "
                f"{c.buried_at_seq}; earlier evidence does not un-bury a claim"
            )
        # The premise-change row must still be live (not superseded).
        if seq in self.ledger.superseded_seqs():
            raise GraveyardRefused(
                f"premise_change_seq {seq} was superseded; a dead premise change "
                "cannot license revival"
            )

    def graveyard(self) -> Iterable[Claim]:
        """Currently dead claims. Fully present in claims[]; excluded from live set."""
        return (c for c in self.claims.values() if c.in_graveyard)

    def live_claims(self) -> Iterable[Claim]:
        """Retrieval, not forgetting: buried claims leave the working set and stay present."""
        return (c for c in self.claims.values() if not c.in_graveyard)

    def is_auditable(self, claim_id: str) -> bool:
        """Every claim ever added remains auditable, buried or not."""
        return claim_id in self.claims
