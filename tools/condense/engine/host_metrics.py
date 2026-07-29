#!/usr/bin/env python3.12
"""Host metric parsers shared by retired campaign controllers.

Controllers that only needed these helpers are deleted; the helpers live here so
specs and remaining live readers share one implementation.
"""
from __future__ import annotations

import re
from decimal import ROUND_CEILING, Decimal, InvalidOperation

_SWAP_USED = re.compile(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([BKMGTPE])", re.I)


def parse_swap_used(text: str) -> int:
    """Parse macOS ``vm.swapusage`` used-bytes, rounding up to a whole byte."""
    match = _SWAP_USED.search(text)
    if match is None:
        raise ValueError("vm.swapusage omitted used bytes")
    multipliers = {
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
        "E": 1024**6,
    }
    try:
        value = Decimal(match.group(1)) * multipliers[match.group(2).upper()]
    except (InvalidOperation, KeyError) as exc:
        raise ValueError("vm.swapusage used bytes are malformed") from exc
    if value < 0:
        raise ValueError("vm.swapusage used bytes are negative")
    return int(value.to_integral_value(rounding=ROUND_CEILING))
