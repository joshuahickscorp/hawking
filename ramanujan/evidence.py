"""Evidence tiers, and the rule that only verifier events and the Tribunal promote.

`odyssey/verifiers/VERIFICATION_LATTICE.json` names three invariants:

    Tier 1 never becomes proof by accumulation
    Tier 2 requires an independent fidelity assessment, not the formalizer's own
    Tier 3 requires a clean-container reproduction against the pinned Mathlib

The first is the one a research system violates by drift rather than by decision: a
hundred supporting computations feel like a proof, and are not one.  So promotion here
is not a function of how much evidence exists.  It is a function of what KIND arrived,
and each tier names a kind that cannot be manufactured by repetition.

The model's own assertion is Tier 0 and carries no weight at all, including when the
model is confident.  Math-Preserve is the standing reminder: it predicts ' combust' for
the capital of France with a logit of 8.03, which is confidence without knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tier(IntEnum):
    ASSERTED = 0
    EMPIRICALLY_SUPPORTED = 1
    FORMALIZED = 2
    PROVEN = 3


TIER_MEANING = {
    Tier.ASSERTED: "The model said it. No evidentiary weight.",
    Tier.EMPIRICALLY_SUPPORTED: "Reproducible computation over a stated domain. Not a proof.",
    Tier.FORMALIZED: "A formal statement whose fidelity to the informal claim was independently assessed.",
    Tier.PROVEN: "Machine-checked in a clean container against the pinned Mathlib.",
}


class PromotionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifierEvent:
    """The only thing that moves a claim up. Produced by a verifier, never by a generator."""

    kind: str  # "computation" | "fidelity_assessment" | "machine_check"
    actor: str
    container_hash: str | None
    independent_of_author: bool
    detail: dict


# Which verifier event kind licenses which tier, and nothing else does.
_LICENSES = {
    Tier.EMPIRICALLY_SUPPORTED: "computation",
    Tier.FORMALIZED: "fidelity_assessment",
    Tier.PROVEN: "machine_check",
}


def promote(current: Tier, event: VerifierEvent, author: str) -> Tier:
    """Return the new tier, or raise. One step at a time; no skipping.

    Refuses rather than returns the unchanged tier, because a silently-ignored promotion
    attempt is how a claim ends up cited at a tier nobody granted it.
    """
    target = Tier(min(int(current) + 1, int(Tier.PROVEN)))
    if current == Tier.PROVEN:
        raise PromotionRefused("already Tier 3; there is nothing above proven")

    required = _LICENSES[target]
    if event.kind != required:
        raise PromotionRefused(
            f"tier {int(current)} -> {int(target)} requires a {required!r} verifier event, "
            f"got {event.kind!r}. Accumulating more {event.kind!r} events does not substitute: "
            f"'Tier 1 never becomes proof by accumulation'."
        )

    if target == Tier.FORMALIZED and not event.independent_of_author:
        raise PromotionRefused(
            "Tier 2 requires an independent fidelity assessment, not the formalizer's own. "
            f"author={author!r} assessed their own formalization."
        )
    if target == Tier.FORMALIZED and event.actor == author:
        raise PromotionRefused(
            f"Tier 2 fidelity assessment by the author ({author!r}) is not independent"
        )

    if target == Tier.PROVEN and not event.container_hash:
        raise PromotionRefused(
            "Tier 3 requires a clean-container reproduction against the pinned Mathlib; "
            "no container hash was recorded, so the check is not reproducible"
        )

    return target


def promote_many(current: Tier, events: list[VerifierEvent], author: str) -> Tier:
    """Apply events in order. Exists mainly to make the accumulation invariant testable:
    a thousand computation events still land on Tier 1."""
    tier = current
    for e in events:
        try:
            tier = promote(tier, e, author)
        except PromotionRefused:
            continue
    return tier
