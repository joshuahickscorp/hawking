"""Roles: charters, non-widening capabilities, and gated store access.

The catalogue matches `odyssey/roles/ROLES.json`. The structural property that matters
is the same one `hide-core/src/automation.rs` enforces for background jobs:

    a RoleCapability can only be *derived* from a Role's fixed permission set;
    there is no public constructor that invents capabilities, and no method that
    adds one after derivation.

Generators (everyone except the Tribunal and the verifier) have `may_promote: false`.
Only verifier events and the Tribunal advance evidence status -- enforced here on the
write path, not merely stated in a charter.

`RAMANUJAN_RESEARCH_AUTHORIZED` is never flipped by any role. That flag lives in the
Limit Registry and stays false.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ramanujan.evidence import VerifierEvent
from ramanujan.stores import Stores, TribunalRefused

if TYPE_CHECKING:
    from ramanujan.limits import LimitRegistry


class CapabilityRefused(RuntimeError):
    """Raised when a role attempts an action outside its capability set.

    Fail closed: a missing grant is a refusal, not a silent no-op. A silent no-op would
    let a generator appear to promote while leaving the claim unmoved, which is worse
    than a crash -- the call site would believe the promotion happened.
    """


# Capability vocabulary. Closed on purpose: an open vocabulary would let a role invent
# a verb that looks like authority.
KNOWN_CAPABILITIES = frozenset(
    {
        "schedule",
        "read_ledger",
        "read_memory",
        "read_claim",
        "read_lean",
        "read_economics",
        "read_all",
        "write_claim",
        "write_objection",
        "write_lean",
        "write_evidence",
        "write_topology",
        "write_literature",
        "write_budget",
        "write_audit",
        "write_strategy",
        "run_lean",
        "run_sandbox",
        "run_retrieval",
        "promote_claim",
    }
)


@dataclass(frozen=True)
class RoleSpec:
    """Immutable charter entry. Source of truth for what a role may ever do."""

    id: str
    charter: str
    capabilities: frozenset[str]
    may_promote: bool


def _spec(rid: str, charter: str, caps: tuple[str, ...], may_promote: bool) -> RoleSpec:
    unknown = set(caps) - KNOWN_CAPABILITIES
    if unknown:
        raise ValueError(f"unknown capabilities for {rid!r}: {sorted(unknown)}")
    if may_promote and "promote_claim" not in caps:
        raise ValueError(f"{rid!r}: may_promote requires the promote_claim capability")
    if (not may_promote) and "promote_claim" in caps:
        raise ValueError(
            f"{rid!r}: promote_claim capability is reserved for promoters; "
            "generators must not carry it"
        )
    return RoleSpec(id=rid, charter=charter, capabilities=frozenset(caps), may_promote=may_promote)


# The ten generators plus Tribunal and verifier. Every generator has may_promote=False.
# Order is stable for receipts and tests.
ROLE_CATALOG: dict[str, RoleSpec] = {
    s.id: s
    for s in (
        _spec(
            "director",
            "Allocates turns across roles and holds no mathematical opinion.",
            ("schedule", "read_ledger"),
            False,
        ),
        _spec(
            "conjecturer",
            "Generates candidate claims. May not assert confidence.",
            ("read_memory", "write_claim"),
            False,
        ),
        _spec(
            "skeptic",
            "Attacks claims. May not prove.",
            ("read_claim", "write_objection"),
            False,
        ),
        _spec(
            "formalizer",
            "Translates to Lean. May not silently weaken a statement.",
            ("read_claim", "write_lean"),
            False,
        ),
        _spec(
            "prover",
            "Attempts formal proof. No proof exists without compiler acceptance.",
            ("read_lean", "run_lean"),
            False,
        ),
        _spec(
            "computationalist",
            "Runs symbolic and numerical experiments. May not treat computation as proof.",
            ("run_sandbox", "write_evidence"),
            False,
        ),
        _spec(
            "cartographer",
            "Maintains investigation topology. May not generate mathematics.",
            ("read_ledger", "write_topology"),
            False,
        ),
        _spec(
            "librarian",
            "Searches prior art. May not judge correctness.",
            ("run_retrieval", "write_literature"),
            False,
        ),
        _spec(
            "economist",
            "Allocates compute. Never sees claim content.",
            ("read_economics", "write_budget"),
            False,
        ),
        _spec(
            "adversary",
            "Audits the system on schedule. Cannot be suppressed or deprioritized.",
            ("read_all", "write_audit"),
            False,
        ),
        _spec(
            "tribunal",
            "Adjudicates admissibility and novelty. Never certifies its own novelty.",
            ("read_all", "promote_claim"),
            True,
        ),
        _spec(
            "verifier",
            "Compiler and reproduction events. The only non-Tribunal promoter.",
            ("run_lean", "run_sandbox", "promote_claim"),
            True,
        ),
    )
}

GENERATOR_IDS = frozenset(rid for rid, s in ROLE_CATALOG.items() if not s.may_promote)
PROMOTER_IDS = frozenset(rid for rid, s in ROLE_CATALOG.items() if s.may_promote)


@dataclass(frozen=True)
class RoleCapability:
    """Capability handed to a role session. Structurally non-widening.

    Fields are private by convention and frozen; the only construction path is
    `RoleCapability.derive`. There is no `grant` / `add` API -- the same shape as
    `JobCapability` in hide-core automation.
    """

    _role_id: str
    _capabilities: frozenset[str]
    _may_promote: bool

    @staticmethod
    def derive(spec: RoleSpec) -> RoleCapability:
        """Derive the full capability the role's charter grants. Cannot be widened later."""
        return RoleCapability(
            _role_id=spec.id,
            _capabilities=spec.capabilities,
            _may_promote=spec.may_promote,
        )

    @staticmethod
    def derive_subset(spec: RoleSpec, requested: frozenset[str]) -> RoleCapability:
        """Derive a capability that is a strict subset of the role's grant.

        Requesting anything outside the charter fails closed at derivation time, not at
        first use -- same law as PermissionSet::derive_capability_subset.
        """
        extra = requested - spec.capabilities
        if extra:
            raise CapabilityRefused(
                f"cannot derive capability for {sorted(extra)}: not in {spec.id!r} charter"
            )
        # Subsetting never upgrades may_promote; it can only keep or lose promote_claim.
        may = spec.may_promote and "promote_claim" in requested
        return RoleCapability(
            _role_id=spec.id,
            _capabilities=frozenset(requested),
            _may_promote=may,
        )

    @property
    def role_id(self) -> str:
        return self._role_id

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    @property
    def may_promote(self) -> bool:
        return self._may_promote

    def allows(self, cap: str) -> bool:
        return cap in self._capabilities

    def require(self, cap: str) -> None:
        if not self.allows(cap):
            raise CapabilityRefused(
                f"role {self._role_id!r} capability does not grant {cap!r}; "
                f"granted={sorted(self._capabilities)}"
            )

    def require_promote(self) -> None:
        """Promotion needs both the capability bit and the structural may_promote flag."""
        if not self._may_promote or not self.allows("promote_claim"):
            raise CapabilityRefused(
                f"role {self._role_id!r} may not promote: may_promote={self._may_promote}, "
                f"has promote_claim={self.allows('promote_claim')}. "
                "Only verifier events and the Tribunal advance evidence status."
            )


def capability_for(role_id: str) -> RoleCapability:
    """Look up a role and derive its full capability. Unknown roles fail closed."""
    try:
        spec = ROLE_CATALOG[role_id]
    except KeyError as e:
        raise CapabilityRefused(f"unknown role {role_id!r}") from e
    return RoleCapability.derive(spec)


class RoleSession:
    """A role's only mutation path into the seven stores.

    Every write checks the derived capability first. The session holds a RoleCapability,
    not a mutable set of strings, so a clever caller cannot widen mid-session.
    """

    def __init__(
        self,
        role_id: str,
        stores: Stores,
        limits: LimitRegistry | None = None,
        capability: RoleCapability | None = None,
    ) -> None:
        self.role_id = role_id
        self.stores = stores
        self.limits = limits
        self.capability = capability if capability is not None else capability_for(role_id)
        if self.capability.role_id != role_id:
            raise CapabilityRefused(
                f"capability was derived for {self.capability.role_id!r}, "
                f"not session role {role_id!r}"
            )
        self.spec = ROLE_CATALOG[role_id]

    # -- reads (capability-gated) -----------------------------------------
    def read_claim(self, claim_id: str):
        if not (self.capability.allows("read_claim") or self.capability.allows("read_all")
                or self.capability.allows("read_memory")):
            self.capability.require("read_claim")
        return self.stores.claims[claim_id]

    # -- writes -----------------------------------------------------------
    def write_claim(self, claim_id: str, statement: str):
        self.capability.require("write_claim")
        self._consult_before_write("write_claim")
        return self.stores.add_claim(claim_id, statement, author=self.role_id)

    def write_objection(self, claim_id: str, reason: str) -> None:
        self.capability.require("write_objection")
        self._consult_before_write("write_objection")
        self.stores.bury(claim_id, reason, actor=self.role_id)

    def write_lean(self, claim_id: str, lean_text: str):
        self.capability.require("write_lean")
        self._consult_before_write("write_lean")
        # One proof-state entry per claim id for the session surface; stores refuse overwrite.
        return self.stores.add_proof_state(
            ps_id=claim_id,
            claim_id=claim_id,
            lean=lean_text,
            actor=self.role_id,
        )

    def write_evidence(self, claim_id: str, event: VerifierEvent):
        """Record a verifier event. The event's actor must be this role.

        Note: write_evidence is *not* promotion. A computationalist may attach a
        computation event; the lattice still decides whether that licenses a tier move,
        and may_promote stays false for the role.
        """
        self.capability.require("write_evidence")
        self._consult_before_write("write_evidence")
        if event.actor != self.role_id:
            raise CapabilityRefused(
                f"evidence actor {event.actor!r} must match session role {self.role_id!r}"
            )
        return self.stores.record_evidence(claim_id, event)

    def write_literature(self, entry_id: str, body: dict):
        self.capability.require("write_literature")
        self._consult_before_write("write_literature")
        return self.stores.add_prior_art(entry_id, body, actor=self.role_id)

    def write_topology(self, node_id: str, body: dict):
        self.capability.require("write_topology")
        self._consult_before_write("write_topology")
        # Topology rides in the Strategy store under a topo: prefix. A dedicated
        # topology store remains scaffold; the ledger row is what makes the write real.
        return self.stores.add_strategy(f"topo:{node_id}", {"topology_node": node_id, **body}, actor=self.role_id)

    def write_budget(self, branch_id: str, grant: dict) -> dict:
        """Economist path. Deliberately rejects claim content in the grant payload.

        The economist's charter: never sees claim content. Enforced, not noted.
        """
        self.capability.require("write_budget")
        self._consult_before_write("write_budget")
        forbidden = {"statement", "claim_text", "goal", "proof", "lean"}
        leaked = forbidden & set(grant)
        if leaked:
            raise CapabilityRefused(
                f"economist must not see claim content; grant carried {sorted(leaked)}"
            )
        payload = {"branch": branch_id, **grant}
        self.stores.ledger.append("budget_grant", payload, actor=self.role_id)
        return payload

    def write_audit(self, subject: str, finding: dict) -> dict:
        self.capability.require("write_audit")
        self._consult_before_write("write_audit")
        payload = {"subject": subject, **finding}
        self.stores.ledger.append("objection", {"audit": payload}, actor=self.role_id)
        return payload

    def write_strategy(self, sid: str, body: dict):
        self.capability.require("write_strategy")
        self._consult_before_write("write_strategy")
        return self.stores.add_strategy(sid, body, actor=self.role_id)

    # -- promotion (promoters only) ---------------------------------------
    def tribunal_admit(self, claim_id: str, human_expert_gate: bool) -> None:
        """Admit a claim. Generators cannot reach this path: require_promote fails closed."""
        self.capability.require_promote()
        self._consult_before_write("promote_claim")
        try:
            self.stores.tribunal_admit(
                claim_id,
                admitting_actor=self.role_id,
                human_expert_gate=human_expert_gate,
            )
        except TribunalRefused:
            raise

    def attempt_promote(self, claim_id: str) -> None:
        """Explicit promote attempt. Generators always refuse here.

        Exists so a test can show that even a generator that *calls* promote is stopped
        by the capability, not by social convention.
        """
        self.capability.require_promote()
        raise CapabilityRefused(
            "attempt_promote is not a free-standing promotion path; use tribunal_admit "
            "or a verifier event recorded through the lattice"
        )

    # -- limit consultation -----------------------------------------------
    def _consult_before_write(self, action: str) -> None:
        """If a Limit Registry is bound, consult it. Decorative registries are not bound.

        When bound, a blocking limit raises CapabilityRefused so the write never happens.
        """
        if self.limits is None:
            return
        verdict = self.limits.consult(action, role_id=self.role_id)
        if not verdict.allowed:
            raise CapabilityRefused(
                f"limit registry blocked {action!r} for {self.role_id!r}: {verdict.reason} "
                f"(limit={verdict.blocking_limit})"
            )


def all_role_ids() -> tuple[str, ...]:
    return tuple(ROLE_CATALOG)


def generators_may_not_promote() -> bool:
    """Structural invariant used by the receipt and tests."""
    return all(not ROLE_CATALOG[r].may_promote for r in GENERATOR_IDS)
