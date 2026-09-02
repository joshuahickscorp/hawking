"""How much exact prompt prefix HCLI preserves between successive turns.

The resident can only reuse KV for a prompt that begins with exactly the tokens
it already holds. Whether that ever happens is a property of the PROMPT BUILDER,
not of the cache, and it was never measured -- the only evidence was a wall
clock, which cannot separate "reuse worked" from "the prompt was shorter".

Two different questions, deliberately measured in two different places:

* **Reusable** -- how much prefix the builder PRESERVED. Measured here, on the
  rendered prompt text, because that is the artifact the builder produces.
* **Realized** -- how much the resident actually skipped. Measured in the
  resident, which is the only thing that knows token boundaries, and reported
  back as ``prefix_reused_tokens``.

A gap between them is a resident problem. A low *reusable* fraction is an
architecture problem, and no amount of cache work fixes it.

Character prefix is the honest unit for the builder question: tokenization is
the resident's business, and a builder that preserves the leading characters
preserves the leading tokens up to at most one boundary token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def longest_common_prefix(a: str, b: str) -> int:
    """Characters, not a hash. The whole point is that it must be exact."""
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def divergence_reason(previous: str, current: str, common: int) -> str:
    """Why the prefix stopped matching, in terms an architect can act on."""
    if not previous:
        return "first_turn_no_previous_prompt"
    if common == len(previous) and len(current) > len(previous):
        return "pure_append"
    if common == len(current) == len(previous):
        return "identical_prompt"
    if len(current) < len(previous):
        return "prompt_shrank_content_dropped_or_compacted"
    if common == 0:
        return "diverges_at_first_character"
    fraction = common / max(1, len(previous))
    if fraction < 0.10:
        return "diverges_in_the_leading_10_percent_system_or_header_rewritten"
    if fraction < 0.90:
        return "diverges_mid_prompt_content_reordered_or_rewritten"
    return "diverges_in_the_trailing_10_percent_tail_rebuilt"


@dataclass
class PrefixTransition:
    """One same-conversation turn boundary."""

    goal_id: str
    turn: int
    previous_prompt_chars: int
    current_prompt_chars: int
    longest_common_prefix_chars: int
    reason_for_prefix_divergence: str
    previous_prompt_tokens: Optional[int] = None
    current_prompt_tokens: Optional[int] = None
    longest_common_prefix_tokens: Optional[int] = None
    prefix_reused_tokens: Optional[int] = None
    prefill_tokens_stepped: Optional[int] = None
    active_context_tokens: Optional[int] = None

    @property
    def reusable_fraction(self) -> Optional[float]:
        """What the BUILDER preserved, as a share of the current prompt."""
        if not self.current_prompt_chars:
            return None
        return self.longest_common_prefix_chars / self.current_prompt_chars

    @property
    def realized_reuse_fraction(self) -> Optional[float]:
        """What the RESIDENT actually skipped, as a share of the current prompt."""
        if not self.current_prompt_tokens or self.prefix_reused_tokens is None:
            return None
        return self.prefix_reused_tokens / self.current_prompt_tokens

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "goal_id": self.goal_id,
            "turn": self.turn,
            "previous_prompt_chars": self.previous_prompt_chars,
            "current_prompt_chars": self.current_prompt_chars,
            "longest_common_prefix_chars": self.longest_common_prefix_chars,
            "previous_prompt_tokens": self.previous_prompt_tokens,
            "current_prompt_tokens": self.current_prompt_tokens,
            "longest_common_prefix_tokens": self.longest_common_prefix_tokens,
            "prefix_reused_tokens": self.prefix_reused_tokens,
            "prefill_tokens_stepped": self.prefill_tokens_stepped,
            "active_context_tokens": self.active_context_tokens,
            "reason_for_prefix_divergence": self.reason_for_prefix_divergence,
        }
        reusable = self.reusable_fraction
        realized = self.realized_reuse_fraction
        if reusable is not None:
            out["reusable_fraction"] = round(reusable, 6)
        if realized is not None:
            out["realized_reuse_fraction"] = round(realized, 6)
        return out


class PrefixProbe:
    """Per-conversation memory of the last rendered prompt.

    Bounded by construction: one string per live goal, dropped when the goal
    ends. It never stores a transcript.
    """

    def __init__(self) -> None:
        self._last: Dict[str, str] = {}
        self._turn: Dict[str, int] = {}
        self.transitions: List[PrefixTransition] = []

    def observe(
        self,
        goal_id: str,
        prompt: str,
        *,
        prompt_tokens: Optional[int] = None,
        prefix_reused_tokens: Optional[int] = None,
        prefill_tokens_stepped: Optional[int] = None,
        active_context_tokens: Optional[int] = None,
        previous_prompt_tokens: Optional[int] = None,
    ) -> Optional[PrefixTransition]:
        """Record one turn. Returns the transition, or None on the first turn."""
        key = str(goal_id or "")
        prompt = prompt or ""
        turn = self._turn.get(key, 0)
        previous = self._last.get(key)
        self._last[key] = prompt
        self._turn[key] = turn + 1
        if previous is None:
            return None
        common = longest_common_prefix(previous, prompt)
        # The token-level common prefix is bounded ABOVE by what the resident
        # could reuse; it is not derivable from characters, so it is only
        # populated when the resident reports it.
        transition = PrefixTransition(
            goal_id=key,
            turn=turn,
            previous_prompt_chars=len(previous),
            current_prompt_chars=len(prompt),
            longest_common_prefix_chars=common,
            reason_for_prefix_divergence=divergence_reason(previous, prompt, common),
            previous_prompt_tokens=previous_prompt_tokens,
            current_prompt_tokens=prompt_tokens,
            longest_common_prefix_tokens=None,
            prefix_reused_tokens=prefix_reused_tokens,
            prefill_tokens_stepped=prefill_tokens_stepped,
            active_context_tokens=active_context_tokens,
        )
        self.transitions.append(transition)
        return transition

    def forget(self, goal_id: str) -> None:
        self._last.pop(str(goal_id or ""), None)
        self._turn.pop(str(goal_id or ""), None)

    def summary(self) -> Dict[str, Any]:
        """Aggregate over the transitions seen so far. Empty is not zero."""
        rows = [t for t in self.transitions if t.current_prompt_chars]
        if not rows:
            return {"transitions": 0}
        reusable = [t.reusable_fraction for t in rows if t.reusable_fraction is not None]
        realized = [
            t.realized_reuse_fraction
            for t in rows
            if t.realized_reuse_fraction is not None
        ]
        reasons: Dict[str, int] = {}
        for t in rows:
            reasons[t.reason_for_prefix_divergence] = (
                reasons.get(t.reason_for_prefix_divergence, 0) + 1
            )
        out: Dict[str, Any] = {
            "transitions": len(rows),
            "reason_counts": reasons,
        }
        if reusable:
            out["reusable_fraction_mean"] = round(sum(reusable) / len(reusable), 6)
            out["reusable_fraction_min"] = round(min(reusable), 6)
        if realized:
            out["realized_reuse_fraction_mean"] = round(sum(realized) / len(realized), 6)
        else:
            out["realized_reuse_fraction_mean"] = None
            out["realized_note"] = (
                "the resident reported no prefix_reused_tokens; realized reuse is "
                "unmeasured, NOT zero"
            )
        return out
