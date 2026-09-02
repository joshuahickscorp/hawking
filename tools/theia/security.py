"""H.3 authorized-security state machine.

States are modeled. The ACTIVE_TEST transition is a hard stop: it is not
an executable action in this scaffold and cannot be forced. Nothing here
scans, generates a payload, handles a credential, or opens a network.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

from tools.theia.authority import (
    BlockedRightsError,
    PinnedScope,
    load_scope,
    pin_scope,
)


class SecurityState(Enum):
    DISCOVERED_PROGRAM = "DISCOVERED_PROGRAM"
    RULES_PINNED = "RULES_PINNED"
    SCOPE_PINNED = "SCOPE_PINNED"
    TARGET_IN_SCOPE = "TARGET_IN_SCOPE"
    SAFE_TEST_PLAN = "SAFE_TEST_PLAN"
    RATE_IMPACT_POLICY_PASS = "RATE/IMPACT_POLICY_PASS"
    ACTIVE_TEST = "ACTIVE_TEST"
    MINIMAL_REPRODUCTION = "MINIMAL_REPRODUCTION"
    ROOT_CAUSE = "ROOT_CAUSE"
    IMPACT_BOUNDED = "IMPACT_BOUNDED"
    REPORT_DRAFT = "REPORT_DRAFT"
    PRIVATE_VERIFICATION = "PRIVATE_VERIFICATION"
    SUBMIT_ACCORDING_TO_PROGRAM = "SUBMIT_ACCORDING_TO_PROGRAM"
    DISCLOSURE_STATE_TRACKED = "DISCLOSURE_STATE_TRACKED"


LAST_LEGAL_STATE = SecurityState.RATE_IMPACT_POLICY_PASS

POST_ACTIVE_STATES: frozenset[SecurityState] = frozenset(
    {
        SecurityState.ACTIVE_TEST,
        SecurityState.MINIMAL_REPRODUCTION,
        SecurityState.ROOT_CAUSE,
        SecurityState.IMPACT_BOUNDED,
        SecurityState.REPORT_DRAFT,
        SecurityState.PRIVATE_VERIFICATION,
        SecurityState.SUBMIT_ACCORDING_TO_PROGRAM,
        SecurityState.DISCLOSURE_STATE_TRACKED,
    }
)

LEGAL_TRANSITIONS: dict[SecurityState, tuple[SecurityState, ...]] = {
    SecurityState.DISCOVERED_PROGRAM: (SecurityState.RULES_PINNED,),
    SecurityState.RULES_PINNED: (SecurityState.SCOPE_PINNED,),
    SecurityState.SCOPE_PINNED: (SecurityState.TARGET_IN_SCOPE,),
    SecurityState.TARGET_IN_SCOPE: (SecurityState.SAFE_TEST_PLAN,),
    SecurityState.SAFE_TEST_PLAN: (SecurityState.RATE_IMPACT_POLICY_PASS,),
    SecurityState.RATE_IMPACT_POLICY_PASS: (),
}


class ActiveTestRefused(RuntimeError):
    """ACTIVE_TEST is modeled and refused. There is no executable action."""


class InvalidSecurityTransition(ValueError):
    pass


class ScopeAlreadyPinned(RuntimeError):
    pass


class SecurityMachine:
    def __init__(self, state: SecurityState = SecurityState.DISCOVERED_PROGRAM) -> None:
        if state in POST_ACTIVE_STATES:
            raise ActiveTestRefused(
                f"cannot occupy {state.value}: ACTIVE_TEST and later states "
                "are not executable in this scaffold"
            )
        self._state = state
        self._pinned: PinnedScope | None = None

    @property
    def state(self) -> SecurityState:
        return self._state

    @property
    def pinned(self) -> PinnedScope | None:
        return self._pinned

    def pin_from_file(self, authority_file: Path) -> PinnedScope:
        if self._pinned is not None:
            raise ScopeAlreadyPinned("scope object is immutable once pinned")
        self._pinned = pin_scope(load_scope(authority_file))
        return self._pinned

    def advance(self, target: SecurityState) -> SecurityState:
        if target in POST_ACTIVE_STATES:
            raise ActiveTestRefused(
                f"refused {self._state.value} -> {target.value}: ACTIVE_TEST "
                "is a hard stop in this scaffold and is not an executable action"
            )
        allowed = LEGAL_TRANSITIONS.get(self._state, ())
        if target not in allowed:
            raise InvalidSecurityTransition(
                f"{self._state.value} -> {target.value} is not a legal H.3 step"
            )
        if target == SecurityState.SCOPE_PINNED and self._pinned is None:
            raise BlockedRightsError("unpinned", "SCOPE_PINNED requires a pinned scope")
        self._state = target
        return self._state

    def force(self, target: SecurityState) -> None:
        """There is no force path. The name exists so tests can prove it refuses."""
        raise ActiveTestRefused(
            f"ACTIVE_TEST cannot be forced; refused {self._state.value} -> {target.value}"
        )

    def execute(self) -> None:
        if self._state in POST_ACTIVE_STATES or self._state == SecurityState.ACTIVE_TEST:
            raise ActiveTestRefused(
                f"{self._state.value} is not an executable action in this scaffold"
            )

    def walk_to_last_legal(self) -> SecurityState:
        order = (
            SecurityState.RULES_PINNED,
            SecurityState.SCOPE_PINNED,
            SecurityState.TARGET_IN_SCOPE,
            SecurityState.SAFE_TEST_PLAN,
            SecurityState.RATE_IMPACT_POLICY_PASS,
        )
        for target in order:
            self.advance(target)
        return self._state
