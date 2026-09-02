"""Authorization boundary (§19.3, H.3).

The scope object is loaded only from an explicit operator-supplied authority
file. It is never inferred, never guessed, never derived from bounty text.
Once pinned it is immutable. Ambiguous / unpinned / out-of-scope all return
BLOCKED_RIGHTS. Fail closed: there is no default-allow path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping


WILDCARDS = frozenset({"*", "ANY", "ALL", "EVERYWHERE"})
AUTHORITY_SCHEMA = "hawking.theia.authority.v1"
REQUIRED_AUTHORITY_KEYS = (
    "schema",
    "program_id",
    "allowed_targets",
    "forbidden_test_classes",
    "operator",
)


class BlockedRightsError(Exception):
    status = "BLOCKED_RIGHTS"

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class BlockedRights:
    reason: str
    detail: str = ""
    status: Literal["BLOCKED_RIGHTS"] = "BLOCKED_RIGHTS"


@dataclass(frozen=True)
class ScopeObject:
    program_id: str
    allowed_targets: frozenset[str]
    forbidden_test_classes: frozenset[str]
    rate_rules: tuple[str, ...]
    disclosure_policy: str
    operator: str
    authority_file: str
    schema: str


@dataclass(frozen=True)
class PinnedScope:
    scope: ScopeObject
    pinned: Literal[True] = True


@dataclass(frozen=True)
class AuthorizationDecision:
    status: Literal["IN_SCOPE", "BLOCKED_RIGHTS"]
    reason: str
    pinned: bool
    authority_file: str | None
    program_id: str | None = None
    declared_target: str | None = None
    detail: str = ""


def fail_closed() -> bool:
    """Fail closed. Returning False default-allows unresolved scope.

    Mutation-check target: flip this to False and the authorization tests
    must FAIL. Restored value is True.
    """
    return True


def pin_scope(scope: ScopeObject) -> PinnedScope:
    return PinnedScope(scope=scope)


def load_scope(authority_file: Path) -> ScopeObject:
    """Load a scope from an operator-supplied file. Any defect raises BLOCKED_RIGHTS."""
    if authority_file is None:
        raise BlockedRightsError("unpinned", "no authority file")
    path = Path(authority_file)
    if not path.is_file():
        raise BlockedRightsError(
            "unpinned", f"authority file is not a local file: {path}"
        )
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise BlockedRightsError("ambiguous", f"authority file unreadable: {e}") from e
    if not isinstance(doc, Mapping):
        raise BlockedRightsError("ambiguous", "authority file is not a JSON object")
    missing = [k for k in REQUIRED_AUTHORITY_KEYS if k not in doc]
    if missing:
        raise BlockedRightsError("ambiguous", f"authority file missing {missing}")
    if doc.get("schema") != AUTHORITY_SCHEMA:
        raise BlockedRightsError(
            "ambiguous",
            f"authority schema {doc.get('schema')!r} != {AUTHORITY_SCHEMA!r}",
        )
    program_id = doc.get("program_id")
    if not isinstance(program_id, str) or not program_id.strip():
        raise BlockedRightsError("ambiguous", "program_id is empty")
    operator = doc.get("operator")
    if not isinstance(operator, str) or not operator.strip():
        raise BlockedRightsError(
            "ambiguous", "operator must be an explicit non-empty string"
        )
    targets = doc.get("allowed_targets")
    if not isinstance(targets, list) or not targets:
        raise BlockedRightsError("ambiguous", "allowed_targets must be a non-empty list")
    if any(not isinstance(t, str) or not t.strip() for t in targets):
        raise BlockedRightsError("ambiguous", "allowed_targets entries must be strings")
    if any(t.strip().upper() in WILDCARDS for t in targets):
        raise BlockedRightsError(
            "ambiguous", "wildcard allowed_targets are a guess; refuse"
        )
    forbidden = doc.get("forbidden_test_classes")
    if not isinstance(forbidden, list):
        raise BlockedRightsError("ambiguous", "forbidden_test_classes must be a list")
    rate = doc.get("rate_rules") or []
    if not isinstance(rate, list):
        raise BlockedRightsError("ambiguous", "rate_rules must be a list if present")
    disclosure = doc.get("disclosure_policy") or "private"
    if not isinstance(disclosure, str):
        raise BlockedRightsError("ambiguous", "disclosure_policy must be a string")
    return ScopeObject(
        program_id=program_id.strip(),
        allowed_targets=frozenset(t.strip() for t in targets),
        forbidden_test_classes=frozenset(str(x) for x in forbidden),
        rate_rules=tuple(str(x) for x in rate),
        disclosure_policy=disclosure,
        operator=operator.strip(),
        authority_file=str(path),
        schema=AUTHORITY_SCHEMA,
    )


def _resolve_strict(
    *,
    authority_file: Path | None,
    declared_target: str | None,
    pinned: PinnedScope | None,
    bounty_text: str | None,
) -> AuthorizationDecision:
    del bounty_text  # never used; scope is never derived from bounty text
    if pinned is None:
        if authority_file is None:
            raise BlockedRightsError("unpinned", "no pinned scope and no authority file")
        pinned = pin_scope(load_scope(authority_file))
    if declared_target is None or not str(declared_target).strip():
        raise BlockedRightsError(
            "ambiguous",
            "declared_target is required and is operator-supplied, not bounty text",
        )
    target = str(declared_target).strip()
    if target.upper() in WILDCARDS:
        raise BlockedRightsError("ambiguous", "wildcard declared_target is a guess")
    allowed = pinned.scope.allowed_targets
    if target not in allowed:
        raise BlockedRightsError(
            "out_of_scope",
            f"declared_target {target!r} not in allowed_targets",
        )
    return AuthorizationDecision(
        status="IN_SCOPE",
        reason="declared_target is in the pinned operator-supplied allowed_targets",
        pinned=True,
        authority_file=pinned.scope.authority_file,
        program_id=pinned.scope.program_id,
        declared_target=target,
    )


def resolve(
    *,
    authority_file: Path | None = None,
    declared_target: str | None = None,
    pinned: PinnedScope | None = None,
    bounty_text: str | None = None,
) -> AuthorizationDecision:
    """Single chokepoint. Fail closed: defects become BLOCKED_RIGHTS."""
    try:
        decision = _resolve_strict(
            authority_file=authority_file,
            declared_target=declared_target,
            pinned=pinned,
            bounty_text=bounty_text,
        )
    except BlockedRightsError as e:
        decision = AuthorizationDecision(
            status="BLOCKED_RIGHTS",
            reason=e.reason,
            pinned=pinned is not None,
            authority_file=str(authority_file) if authority_file is not None else None,
            declared_target=declared_target,
            detail=e.detail,
        )
    if decision is None or decision.status not in {"IN_SCOPE", "BLOCKED_RIGHTS"}:
        decision = AuthorizationDecision(
            status="BLOCKED_RIGHTS",
            reason="ambiguous",
            pinned=False,
            authority_file=None,
            detail="resolver produced no decision",
        )
    if decision.status == "BLOCKED_RIGHTS" and not fail_closed():
        return replace(
            decision,
            status="IN_SCOPE",
            reason="default-allow",
            detail="fail_closed() was False; unresolved scope was allowed",
        )
    return decision
