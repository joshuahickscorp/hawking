"""Shared benchmark qualification and contamination boundaries.

The machine-wide sampler is intentionally stricter than a process-name check.
This module gives every experiment the same two-way classification:

* ``QUALIFIED_PROTECTED`` means both ends of the measured window were quiet;
* ``DIAGNOSTIC_CONTAMINATED`` means the run is still useful for structure or
  execution questions, but it cannot support a protected performance claim.

The result is deliberately model-neutral.  A provider may be Qwen today, but
the boundary belongs to the benchmark, not to the resident model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


QUALIFIED_PROTECTED = "QUALIFIED_PROTECTED"
DIAGNOSTIC_CONTAMINATED = "DIAGNOSTIC_CONTAMINATED"


def _rows(sample: Optional[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    if not isinstance(sample, Mapping):
        return ()
    contenders = sample.get("contenders")
    if not isinstance(contenders, list):
        return ()
    return (row for row in contenders if isinstance(row, Mapping))


def _contamination(before: Optional[Mapping[str, Any]], after: Optional[Mapping[str, Any]]) -> list[str]:
    names = {
        str(row.get("comm"))
        for sample in (before, after)
        for row in _rows(sample)
        if row.get("comm")
    }
    if not names and any(
        not isinstance(sample, Mapping) or sample.get("quiet") is not True
        for sample in (before, after)
    ):
        names.add("QUIESCENCE_UNKNOWN")
    return sorted(names)


def classify_window(
    before: Optional[Mapping[str, Any]],
    after: Optional[Mapping[str, Any]],
    bench: Optional[Mapping[str, Any]],
    *,
    qualification: Optional[bool] = None,
    not_for_promotion: bool = True,
) -> Dict[str, Any]:
    """Return a durable classification for one measured window.

    ``bench`` is included because it is the receipt-level derivation of the
    samples.  The samples are checked again here so a stale or hand-authored
    ``QUIESCED`` field cannot silently qualify a run.
    """
    protected = (
        isinstance(bench, Mapping)
        and bench.get("state") == "QUIESCED"
        and isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and before.get("quiet") is True
        and after.get("quiet") is True
    )
    benchmark_class = QUALIFIED_PROTECTED if protected else DIAGNOSTIC_CONTAMINATED
    contamination = [] if protected else _contamination(before, after)
    # A caller may provide an experiment-specific qualification predicate,
    # but it can never override the machine-wide protected boundary.
    qualified = protected if qualification is None else (protected and bool(qualification))
    return {
        "benchmark_class": benchmark_class,
        "qualification": qualified,
        "NOT_FOR_PROMOTION": bool(not_for_promotion or not qualified),
        "contamination": contamination,
        "machine_snapshot": {
            "before": dict(before) if isinstance(before, Mapping) else None,
            "after": dict(after) if isinstance(after, Mapping) else None,
        },
        "protected_window": protected,
        "claim_boundary": (
            "QUALIFIED_PROTECTED is valid only when machine_quiescence reports "
            "quiet=true before and after and bench_block derives QUIESCED. "
            "DIAGNOSTIC_CONTAMINATED observations may answer structural or "
            "execution questions but are not promotion evidence."
        ),
    }


__all__ = [
    "DIAGNOSTIC_CONTAMINATED",
    "QUALIFIED_PROTECTED",
    "classify_window",
]
