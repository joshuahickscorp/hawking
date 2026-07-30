"""Sovereignty hooks for the Ramanujan governance layer.

Binds three things that must not drift apart:

  1. the Limit Registry (what we know we cannot do)
  2. the Ledger (sovereignty_event rows so refusals are auditable)
  3. the Forge gate (F0/F1 fixture stages vs gated F2+)

This is not the full Odyssey sovereignty spine (`tools/sovereignty/sovereignty.py`),
which needs a served model for boundary/false-refusal metrics. Those metrics stay
GATED. What this module provides is the *hook* surface the governance layer needs:
consult before spend, attribute a refusal, refuse research while unauthorized, and
expose Forge readiness without inventing capability.

`RAMANUJAN_RESEARCH_AUTHORIZED` has no flip path here either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ramanujan.ledger import Ledger
from ramanujan.limits import LimitBlocked, LimitRegistry

# From odyssey/sovereignty/SOVEREIGNTY.json. Planes may not impersonate one another.
PLANES = frozenset({"capability", "policy", "permission", "evidence", "resource"})

REFUSAL_CODES = {
    "R-CAPABILITY": "model attempted and failed",
    "R-LEARNED": "inherited blanket refusal",
    "R-POLICY": "declared profile excludes the request",
    "R-PERMISSION": "reasoning is allowed, external action is not",
    "R-VERIFICATION": "generated claim failed evidence requirements",
    "R-RESOURCE": "memory, context, time, or compute exhausted",
}

# Forge stages that this layer may run. F2+ remains GATED (needs served model).
LOCAL_FORGE_STAGES = frozenset({"F0", "F1"})
GATED_FORGE_STAGES = frozenset({"F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"})

AUTHORITY = "NON_PRODUCTION_AUTHORITY"


class SovereigntyRefused(RuntimeError):
    pass


@dataclass
class SovereigntyHooks:
    """Consultable, ledger-backed sovereignty surface for Ramanujan governance.

    Bind a Ledger when refusals must be permanent. Bind a LimitRegistry always --
    a decorative registry is one that is never consulted, and this class exists so
    that cannot happen on the write/spend path.
    """

    limits: LimitRegistry
    ledger: Ledger | None = None
    actor: str = "sovereignty"
    events: list[dict] = field(default_factory=list)

    def research_authorized(self) -> bool:
        return self.limits.research_authorized()

    def require_action(self, action: str, role_id: str | None = None) -> None:
        """Fail closed on a blocked action. Records a sovereignty_event on deny."""
        try:
            self.limits.require(action, role_id=role_id)
        except LimitBlocked as e:
            self._record(
                kind="refusal",
                plane="policy" if action == "run_research" else "permission",
                code="R-POLICY" if action in {"run_research", "teacher_trace_from_math_preserve"} else "R-PERMISSION",
                action=action,
                role_id=role_id,
                detail={"error": str(e)},
            )
            raise SovereigntyRefused(str(e)) from e

    def refuse_research(self, role_id: str | None = None) -> None:
        """Always raises while RAMANUJAN_RESEARCH_AUTHORIZED is false."""
        if self.research_authorized():
            # There is no path in this codebase that sets it true; if a test injects
            # a Limit with current_value True, still refuse to invent a run path.
            raise SovereigntyRefused(
                "research authorization flipped outside the no-flip contract; "
                "refusing to open a research path from the governance layer"
            )
        self.require_action("run_research", role_id=role_id)

    def attribute_refusal(
        self,
        code: str,
        plane: str,
        action: str,
        detail: dict | None = None,
        role_id: str | None = None,
    ) -> dict:
        """Record a classified refusal. Unknown codes fail closed."""
        if code not in REFUSAL_CODES:
            raise SovereigntyRefused(
                f"unknown refusal code {code!r}; known={sorted(REFUSAL_CODES)}"
            )
        if plane not in PLANES:
            raise SovereigntyRefused(
                f"unknown plane {plane!r}; planes may not impersonate one another; "
                f"known={sorted(PLANES)}"
            )
        return self._record(
            kind="refusal",
            plane=plane,
            code=code,
            action=action,
            role_id=role_id,
            detail=detail or {},
        )

    def forge_gate(self, stage: str) -> dict[str, Any]:
        """Allow F0/F1 fixture stages; gate the rest. Never fabricates readiness.

        F0/F1 in ramanujan.forge are NON_PRODUCTION_AUTHORITY instrument loops. They
        do not require research authorization. F2+ stays GATED until a served model
        and evaluation set exist (see odyssey/forge/FORGE.json).
        """
        stage = stage.upper()
        research = self.research_authorized()
        if stage in LOCAL_FORGE_STAGES:
            verdict = {
                "stage": stage,
                "allowed": True,
                "authority": AUTHORITY,
                "RAMANUJAN_RESEARCH_AUTHORIZED": research,
                "reason": f"{stage} is a local fixture instrument; not research authorization",
            }
            self._record(
                kind="forge_gate",
                plane="capability",
                code=None,
                action=f"forge_{stage.lower()}",
                role_id=None,
                detail=verdict,
            )
            return verdict
        if stage in GATED_FORGE_STAGES:
            verdict = {
                "stage": stage,
                "allowed": False,
                "authority": AUTHORITY,
                "RAMANUJAN_RESEARCH_AUTHORIZED": research,
                "reason": (
                    f"{stage} is GATED: needs a served model and an evaluated prompt set "
                    "(odyssey/forge/FORGE.json). The governance layer will not invent readiness."
                ),
            }
            self._record(
                kind="forge_gate",
                plane="capability",
                code="R-CAPABILITY",
                action=f"forge_{stage.lower()}",
                role_id=None,
                detail=verdict,
            )
            return verdict
        raise SovereigntyRefused(f"unknown forge stage {stage!r}")

    def require_forge_stage(self, stage: str) -> None:
        v = self.forge_gate(stage)
        if not v["allowed"]:
            raise SovereigntyRefused(v["reason"])

    def as_public_dict(self) -> dict:
        return {
            "schema": "hawking.ramanujan.sovereignty_hooks.v1",
            "authority": AUTHORITY,
            "RAMANUJAN_RESEARCH_AUTHORIZED": self.research_authorized(),
            "planes": sorted(PLANES),
            "refusal_codes": dict(REFUSAL_CODES),
            "local_forge_stages": sorted(LOCAL_FORGE_STAGES),
            "gated_forge_stages": sorted(GATED_FORGE_STAGES),
            "n_events": len(self.events),
            "limit_ids": list(self.limits.ids()),
            "note": (
                "Hook surface only. Full sovereignty metrics "
                "(false_refusal_rate, boundary_error_rate) remain GATED in "
                "tools/sovereignty/sovereignty.py."
            ),
        }

    def _record(
        self,
        kind: str,
        plane: str,
        code: str | None,
        action: str,
        role_id: str | None,
        detail: dict,
    ) -> dict:
        payload = {
            "kind": kind,
            "plane": plane,
            "code": code,
            "action": action,
            "role_id": role_id,
            "detail": detail,
            "RAMANUJAN_RESEARCH_AUTHORIZED": self.research_authorized(),
            "authority": AUTHORITY,
        }
        self.events.append(payload)
        if self.ledger is not None:
            self.ledger.append("sovereignty_event", payload, actor=self.actor)
        return payload
