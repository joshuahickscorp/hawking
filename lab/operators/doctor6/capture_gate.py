"""Capture quality gates: refuse starved or under-censused inputs.

Retention is now deterministic per-expert first-N (default 64). Historical
pathology: 1024 rows per LAYER shared across 128 experts → ~8 rows/expert,
underdetermined fits. v6 refuses to prescribe from a capture whose per-organ
row count is below the stated floor.

Never-routed census (M05): needs >= 1000 tokens; 88 is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_MIN_ROWS_PER_ORGAN = 64
NEVER_ROUTED_MIN_TOKENS = 1000


@dataclass(frozen=True)
class CaptureGateResult:
    ok: bool
    min_rows: int
    floor: int
    n_organs_checked: int
    n_below_floor: int
    starved: list[dict[str, Any]]
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "min_rows_observed": self.min_rows,
            "floor": self.floor,
            "n_organs_checked": self.n_organs_checked,
            "n_below_floor": self.n_below_floor,
            "starved": self.starved[:32],  # cap report size
            "reason": self.reason,
        }


def check_per_organ_rows(
    row_counts: Mapping[str, int] | list[tuple[str, int]],
    *,
    floor: int = DEFAULT_MIN_ROWS_PER_ORGAN,
) -> CaptureGateResult:
    if isinstance(row_counts, Mapping):
        items = list(row_counts.items())
    else:
        items = list(row_counts)
    if not items:
        return CaptureGateResult(
            ok=False,
            min_rows=0,
            floor=floor,
            n_organs_checked=0,
            n_below_floor=0,
            starved=[],
            reason="capture has zero organ rows; refuse prescribe",
        )
    starved = [{"organ": k, "n_rows": int(v)} for k, v in items if int(v) < floor]
    mins = min(int(v) for _, v in items)
    ok = len(starved) == 0
    reason = None
    if not ok:
        reason = (
            f"capture starvation: {len(starved)}/{len(items)} organs below "
            f"floor={floor} (min_rows={mins}); refuse to fit noise. "
            f"Retention must be per-expert first-N >= {floor}."
        )
    return CaptureGateResult(
        ok=ok,
        min_rows=mins,
        floor=floor,
        n_organs_checked=len(items),
        n_below_floor=len(starved),
        starved=starved,
        reason=reason,
    )


def check_never_routed_census(n_tokens: int) -> dict[str, Any]:
    """M05 structured omission requires >= 1000 tokens; 88 is refused."""
    ok = int(n_tokens) >= NEVER_ROUTED_MIN_TOKENS
    return {
        "ok": ok,
        "n_tokens": int(n_tokens),
        "floor": NEVER_ROUTED_MIN_TOKENS,
        "method": "M05_structured_expert_omission",
        "reason": None
        if ok
        else (
            f"never-routed census refused: {n_tokens} tokens < {NEVER_ROUTED_MIN_TOKENS}; "
            "88-token calibration is not evidence"
        ),
    }
