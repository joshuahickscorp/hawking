"""Limit Registry: the things the system knows it cannot do, as data rather than prose.

`odyssey/sovereignty/LIMIT_REGISTRY.json` states the law -- "Hawking is not literally
unlimited; it is free of invisible limits" -- and lists a few. This module makes the
registry consultable on the write/spend path so a role can ask before spending.

A decorative registry is one that is only read by humans. A working registry is one a
role *must* consult: the consult returns allow/deny with a named limit, and a deny
stops the action. The tests pin that behaviour.

`RAMANUJAN_RESEARCH_AUTHORIZED` lives here as limit L-RESEARCH-01 with value false.
There is no method that sets it to true. That is intentional and permanent for this
lane: reaching a sandbox is not authorization to run research in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Limit:
    """One known bound. Immutable after registration so a role cannot renegotiate it."""

    id: str
    type: str  # policy | permission | resource | capability
    current_value: Any
    threshold: Any
    scope: str
    reason: str
    owner: str
    user_changeable: bool = False
    # When True, any action that matches `blocks_actions` is denied while the limit holds.
    blocking: bool = True
    blocks_actions: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ConsultVerdict:
    allowed: bool
    reason: str
    blocking_limit: str | None
    consulted: tuple[str, ...]


# Actions a role may name when consulting. Closed vocabulary.
KNOWN_ACTIONS = frozenset(
    {
        "write_claim",
        "write_objection",
        "write_lean",
        "write_evidence",
        "write_literature",
        "write_topology",
        "write_budget",
        "write_audit",
        "write_strategy",
        "promote_claim",
        "branch_expand",
        "local_compute",
        "network",
        "run_research",
        "teacher_trace_from_math_preserve",
        "run_lean",
        "run_sandbox",
        "run_retrieval",
    }
)


def default_limits() -> list[Limit]:
    """The working set. Includes Odyssey limits plus Ramanujan-specific fences."""
    return [
        Limit(
            id="L-LAUNCH-01",
            type="policy",
            current_value=False,
            threshold="explicit authorization",
            scope="the entire Odyssey/Ramanujan research run",
            reason="Research must be started deliberately, in a session that intends it",
            owner="user",
            user_changeable=True,
            blocking=True,
            blocks_actions=frozenset({"run_research"}),
        ),
        Limit(
            id="L-RESEARCH-01",
            type="policy",
            current_value=False,  # RAMANUJAN_RESEARCH_AUTHORIZED stays false
            threshold=True,
            scope="all Ramanujan research sessions",
            reason=(
                "RAMANUJAN_RESEARCH_AUTHORIZED is false at sandbox-ready and stays false. "
                "Reaching the sandbox is not authorization to run research in it."
            ),
            owner="controller",
            user_changeable=False,
            blocking=True,
            blocks_actions=frozenset({"run_research"}),
        ),
        Limit(
            id="L-NET-01",
            type="permission",
            current_value="deny",
            threshold="deny by default",
            scope="all sandboxed processes",
            reason="a research loop with live network access is not reproducible",
            owner="sandbox",
            user_changeable=True,
            blocking=True,
            blocks_actions=frozenset({"network"}),
        ),
        Limit(
            id="L-HEAVY-01",
            type="resource",
            current_value=1,
            threshold=1,
            scope="local compute",
            reason="one heavy local lane keeps measurements comparable",
            owner="operator",
            user_changeable=True,
            blocking=False,  # advisory on count; not a hard deny of branch_expand
            blocks_actions=frozenset(),
        ),
        Limit(
            id="L-TEACHER-01",
            type="capability",
            current_value="REFUSED",
            threshold="capability_verdict != REFUSED",
            scope="teacher-trace generation",
            reason=(
                "Math-Preserve is semantically collapsed and hash-REFUSED in "
                "odyssey/launch/SUBSTRATE_CAPABILITY.json. Traces from it would teach a "
                "student to reproduce the collapse and would look like data."
            ),
            owner="controller",
            user_changeable=False,
            blocking=True,
            blocks_actions=frozenset({"teacher_trace_from_math_preserve"}),
        ),
        Limit(
            id="L-TOOLCHAIN-01",
            type="resource",
            current_value="10_of_12_missing",
            threshold="lean_and_mathlib_pinned",
            scope="formal proof attempts against real Lean",
            reason=(
                "RAMANUJAN_TOOLCHAIN_SELFTEST.json records 10 of 12 components missing. "
                "Build against fixtures labelled NON_PRODUCTION_AUTHORITY until install "
                "is a separate gated decision."
            ),
            owner="operator",
            user_changeable=True,
            blocking=True,
            # Fixture Lean is fine; this blocks claims of production Lean authority.
            blocks_actions=frozenset(),
        ),
    ]


class LimitRegistry:
    """Consultable limits. The consult path is what makes this data rather than prose."""

    def __init__(self, limits: list[Limit] | None = None) -> None:
        items = limits if limits is not None else default_limits()
        self._by_id: dict[str, Limit] = {}
        for lim in items:
            if lim.id in self._by_id:
                raise ValueError(f"duplicate limit id {lim.id!r}")
            self._by_id[lim.id] = lim
        # Audit of every consult so tests can prove the registry was not decorative.
        self.consult_log: list[dict] = []

    def get(self, limit_id: str) -> Limit:
        try:
            return self._by_id[limit_id]
        except KeyError as e:
            raise KeyError(f"unknown limit {limit_id!r}") from e

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def research_authorized(self) -> bool:
        """Always false until a human changes L-RESEARCH-01 outside this codebase path."""
        return bool(self.get("L-RESEARCH-01").current_value)

    def consult(self, action: str, role_id: str | None = None) -> ConsultVerdict:
        """Ask whether `action` is currently allowed.

        Unknown actions are refused (fail closed): an open action vocabulary would let a
        caller invent a verb that bypasses every named limit.
        """
        if action not in KNOWN_ACTIONS:
            verdict = ConsultVerdict(
                allowed=False,
                reason=f"unknown action {action!r}; not in the closed action vocabulary",
                blocking_limit=None,
                consulted=(),
            )
            self._log(action, role_id, verdict)
            return verdict

        consulted: list[str] = []
        for lim in self._by_id.values():
            consulted.append(lim.id)
            if not lim.blocking:
                continue
            if action not in lim.blocks_actions:
                continue
            # A blocking limit that lists this action denies it while the limit holds.
            # For boolean policy limits, current_value False means "not authorized".
            # For permission limits with value "deny", same.
            if self._is_active_block(lim):
                verdict = ConsultVerdict(
                    allowed=False,
                    reason=lim.reason,
                    blocking_limit=lim.id,
                    consulted=tuple(consulted),
                )
                self._log(action, role_id, verdict)
                return verdict

        verdict = ConsultVerdict(
            allowed=True,
            reason="no active blocking limit",
            blocking_limit=None,
            consulted=tuple(consulted),
        )
        self._log(action, role_id, verdict)
        return verdict

    def require(self, action: str, role_id: str | None = None) -> None:
        """Consult and raise on deny. The spend path uses this so a skip is impossible."""
        v = self.consult(action, role_id=role_id)
        if not v.allowed:
            raise LimitBlocked(
                f"action {action!r} blocked by {v.blocking_limit}: {v.reason}"
            )

    def _is_active_block(self, lim: Limit) -> bool:
        if lim.type == "policy":
            # False / missing authorization keeps the block active.
            return lim.current_value is False or lim.current_value in (None, "deny", "REFUSED")
        if lim.type == "permission":
            return lim.current_value in ("deny", False, "REFUSED")
        if lim.type == "capability":
            return lim.current_value in ("REFUSED", "UNVERIFIED", False)
        if lim.type == "resource":
            # Resource limits block only when explicitly over threshold; default set
            # uses blocking=False for L-HEAVY-01.
            return False
        return bool(lim.blocking)

    def _log(self, action: str, role_id: str | None, verdict: ConsultVerdict) -> None:
        self.consult_log.append(
            {
                "action": action,
                "role_id": role_id,
                "allowed": verdict.allowed,
                "blocking_limit": verdict.blocking_limit,
                "reason": verdict.reason,
            }
        )

    def as_public_dict(self) -> dict:
        """Receipt shape: values only, no mutation surface."""
        return {
            "schema": "hawking.ramanujan.limit_registry.v1",
            "law": "not literally unlimited; free of invisible limits",
            "RAMANUJAN_RESEARCH_AUTHORIZED": self.research_authorized(),
            "limits": [
                {
                    "id": lim.id,
                    "type": lim.type,
                    "current_value": lim.current_value,
                    "threshold": lim.threshold,
                    "scope": lim.scope,
                    "reason": lim.reason,
                    "owner": lim.owner,
                    "user_changeable": lim.user_changeable,
                    "blocking": lim.blocking,
                    "blocks_actions": sorted(lim.blocks_actions),
                }
                for lim in self._by_id.values()
            ],
        }


class LimitBlocked(RuntimeError):
    pass


# No set_research_authorized. No flip path. If you are looking for one, stop.
