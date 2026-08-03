"""Branch budgets, value scoring, and the stop rule that cannot be argued past.

Reuses `SearchEconomics` from `ramanujan.search` for the expansion counter. The layer
above it is the *branch* account: a named investigation line with a budget, a value
score, a halt reason, and a permanent record of why it stopped.

`odyssey/economics/ECONOMICS.json` sets the currency (verified evidence) and the reward
schedule. Refutation and clean exhaustion are paid deliberately: a system that only
pays for promotion learns to avoid falsification.

The economist never sees claim content -- that constraint is enforced in
`roles.RoleSession.write_budget`, not here. This module scores and stops; it does not
read statements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ramanujan.search import SearchEconomics, SearchResult, best_first, ProofState, Tactic


# From odyssey/economics/ECONOMICS.json. Kept as data, not magic numbers at call sites.
REWARDS: dict[str, float] = {
    "clean_branch_exhaustion": 3.0,
    "refutation": 5.0,
    "tier0_to_tier1": 1.0,
    "tier1_to_tier2": 3.0,
    "tier2_to_tier3": 10.0,
    "tribunal_admission": 25.0,
}

CURRENCY = "verified evidence"
ALLOCATOR = "economist"


class BranchHalted(RuntimeError):
    """Raised when a caller tries to spend on a branch that has already stopped."""


@dataclass
class HaltRecord:
    """Why a branch stopped. The reason is the whole point of recording it."""

    reason: str
    spent: int
    value_earned: float
    detail: dict = field(default_factory=dict)


@dataclass
class BranchAccount:
    """A named branch with a budget the search cannot talk its way past.

    When the budget is exhausted the branch HALTS and records why. A halted branch
    refuses further charges. That is the stop rule: not a suggestion, not a metric
    that a later stage can ignore.
    """

    branch_id: str
    economics: SearchEconomics = field(default_factory=SearchEconomics)
    value_earned: float = 0.0
    halt: HaltRecord | None = None
    charges: list[dict] = field(default_factory=list)
    value_events: list[dict] = field(default_factory=list)

    def is_halted(self) -> bool:
        return self.halt is not None

    def may_spend(self) -> bool:
        if self.is_halted():
            return False
        return self.economics.may_expand()

    def why_cannot_spend(self) -> str:
        if self.halt is not None:
            return f"already_halted: {self.halt.reason}"
        if not self.economics.may_expand():
            return "economics: max_expansions"
        return "ok"

    def charge(self, units: int = 1, what: str = "expand") -> None:
        """Spend budget. Halts and records when the budget is exhausted mid-charge."""
        if self.is_halted():
            raise BranchHalted(
                f"branch {self.branch_id!r} is halted ({self.halt.reason}); no further charges"
            )
        for _ in range(units):
            if not self.economics.may_expand():
                self.stop("economics: max_expansions", detail={"last_attempt": what})
                raise BranchHalted(
                    f"branch {self.branch_id!r} exhausted budget after {self.economics.spent} expansions"
                )
            self.economics.charge()
            self.charges.append({"what": what, "spent_after": self.economics.spent})

    def stop(self, reason: str, detail: dict | None = None) -> HaltRecord:
        """Halt the branch and record why. Idempotent on reason if already halted."""
        if self.halt is not None:
            return self.halt
        self.halt = HaltRecord(
            reason=reason,
            spent=self.economics.spent,
            value_earned=self.value_earned,
            detail=dict(detail or {}),
        )
        return self.halt

    def award(self, kind: str, detail: dict | None = None) -> float:
        """Credit value for a verified outcome. Unknown kinds earn nothing (fail closed)."""
        if kind not in REWARDS:
            raise ValueError(
                f"unknown value kind {kind!r}; currency is {CURRENCY!r} and the "
                f"schedule is fixed: {sorted(REWARDS)}"
            )
        amount = REWARDS[kind]
        self.value_earned += amount
        self.value_events.append({"kind": kind, "amount": amount, "detail": detail or {}})
        return amount

    def score(self) -> dict:
        """Value-per-spend summary. Clean exhaustion is itself a reward-eligible outcome."""
        spent = max(1, self.economics.spent)
        return {
            "branch_id": self.branch_id,
            "spent": self.economics.spent,
            "value_earned": self.value_earned,
            "value_per_spend": self.value_earned / spent,
            "halted": self.is_halted(),
            "halt_reason": None if self.halt is None else self.halt.reason,
            "currency": CURRENCY,
        }


def run_branch_search(
    branch: BranchAccount,
    start: ProofState,
    tactics: Tactic,
    heuristic: Callable[[ProofState], float],
) -> tuple[SearchResult, BranchAccount]:
    """Best-first search bound to a branch budget.

    On budget exhaustion the SearchResult already carries `stopped_by`; this also
    writes a HaltRecord onto the branch so the stop is visible outside the search
    stack (Ledger-adjacent callers, the economist, the Cartographer).
    """
    result, _dag = best_first(start, tactics, heuristic, economics=branch.economics)
    if result.stopped_by.startswith("economics:"):
        branch.stop(result.stopped_by, detail={"found": result.found, "path_len": len(result.path)})
        # Clean exhaustion is a positive outcome under the reward schedule when the
        # branch did not find a proof -- it stopped for a real reason rather than
        # wandering. Award only on genuine budget halt, not on depth cutoffs.
        if not result.found and result.stopped_by == "economics: max_expansions":
            branch.award("clean_branch_exhaustion", detail={"expansions": result.expansions})
    elif result.stopped_by == "exhausted":
        branch.stop("search: exhausted", detail={"expansions": result.expansions})
        if not result.found:
            branch.award("clean_branch_exhaustion", detail={"expansions": result.expansions})
    elif result.stopped_by == "closed":
        # Finding a fixture goal is not Tier-3; do not award promotion value here.
        branch.stop("search: closed", detail={"path": list(result.path)})
    return result, branch


def value_for_tier_step(from_tier: int, to_tier: int) -> str | None:
    """Map a tier transition to a reward kind, or None if the step is not paid."""
    key = f"tier{from_tier}_to_tier{to_tier}"
    return key if key in REWARDS else None


class SessionHalted(RuntimeError):
    """Raised when the session-level budget is exhausted; no branch may spend further."""


@dataclass
class SessionEconomics:
    """System-level cost model: a hard cap across every open branch.

    `SearchEconomics.max_expansions` stops one search. This stops the *session*: the
    sum of branch spends cannot talk past `max_total_expansions`. When the session
    halts, every open branch is halted with the same reason so nothing sneaks a
    charge through a still-open branch account.
    """

    session_id: str
    max_total_expansions: int = 500
    spent: int = 0
    branches: dict[str, BranchAccount] = field(default_factory=dict)
    halt: HaltRecord | None = None
    grants: list[dict] = field(default_factory=list)

    def is_halted(self) -> bool:
        return self.halt is not None

    def remaining(self) -> int:
        if self.is_halted():
            return 0
        return max(0, self.max_total_expansions - self.spent)

    def may_spend(self, units: int = 1) -> bool:
        if self.is_halted():
            return False
        return self.spent + units <= self.max_total_expansions

    def open_branch(
        self,
        branch_id: str,
        max_expansions: int | None = None,
        max_depth: int = 12,
    ) -> BranchAccount:
        if branch_id in self.branches:
            raise ValueError(f"branch {branch_id!r} already open in session {self.session_id!r}")
        if self.is_halted():
            raise SessionHalted(
                f"session {self.session_id!r} is halted ({self.halt.reason}); no new branches"
            )
        # Branch budget cannot exceed what the session has left.
        cap = self.remaining() if max_expansions is None else min(max_expansions, self.remaining())
        if cap <= 0:
            raise SessionHalted(
                f"session {self.session_id!r} has no remaining expansions"
            )
        branch = BranchAccount(
            branch_id=branch_id,
            economics=SearchEconomics(max_expansions=cap, max_depth=max_depth),
        )
        self.branches[branch_id] = branch
        self.grants.append(
            {"branch": branch_id, "max_expansions": cap, "session_remaining_after": self.remaining()}
        )
        return branch

    def charge(self, branch_id: str, units: int = 1, what: str = "expand") -> None:
        """Charge a branch and the session together. Either both spend or neither does."""
        if self.is_halted():
            raise SessionHalted(
                f"session {self.session_id!r} is halted ({self.halt.reason}); no further charges"
            )
        if branch_id not in self.branches:
            raise KeyError(f"unknown branch {branch_id!r}")
        branch = self.branches[branch_id]
        if not self.may_spend(units):
            self.stop("economics: max_total_expansions", detail={"last_branch": branch_id})
            raise SessionHalted(
                f"session {self.session_id!r} exhausted total budget after {self.spent} expansions"
            )
        # Branch.charge may itself halt the branch; session spent still advances on success.
        before = branch.economics.spent
        branch.charge(units=units, what=what)
        advanced = branch.economics.spent - before
        self.spent += advanced
        if self.spent >= self.max_total_expansions:
            self.stop("economics: max_total_expansions", detail={"last_branch": branch_id})

    def stop(self, reason: str, detail: dict | None = None) -> HaltRecord:
        if self.halt is not None:
            return self.halt
        self.halt = HaltRecord(
            reason=reason,
            spent=self.spent,
            value_earned=sum(b.value_earned for b in self.branches.values()),
            detail=dict(detail or {}),
        )
        # Propagate: a halted session freezes every open branch.
        for b in self.branches.values():
            if not b.is_halted():
                b.stop(f"session_halted: {reason}", detail={"session": self.session_id})
        return self.halt

    def score(self) -> dict:
        return {
            "session_id": self.session_id,
            "spent": self.spent,
            "max_total_expansions": self.max_total_expansions,
            "remaining": self.remaining(),
            "halted": self.is_halted(),
            "halt_reason": None if self.halt is None else self.halt.reason,
            "n_branches": len(self.branches),
            "value_earned": sum(b.value_earned for b in self.branches.values()),
            "currency": CURRENCY,
            "authority": "NON_PRODUCTION_AUTHORITY",
        }
