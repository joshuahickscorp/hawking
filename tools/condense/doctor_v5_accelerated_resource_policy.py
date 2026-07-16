#!/usr/bin/env python3.12
"""Pure resource-stop classification for opt-in Doctor V5 supervisors.

The sealed base queue historically used ``sole_live`` as a proxy for a cell
whose own process-tree RSS reached the budget.  That proxy is false when an
unrelated host owner causes global memory pressure or swap.  This module keeps
the safety action (shed the lane) separate from the evidence action (whether
the cell itself may be escalated to blocked-execution).

It imports no queue implementation and mutates no campaign state.  Wrappers
provide the exact measured tree RSS and apply the returned transition.
"""
from __future__ import annotations

import math
from typing import Any


SCHEMA = "hawking.doctor_v5_accelerated_resource_stop_decision.v1"
POOL_BUDGET_REASON = "pool_tree_rss_at_or_over_process_budget"
GLOBAL_PRESSURE_REASON = "system_memory_pressure_or_swap"
VALID_REASONS = frozenset({POOL_BUDGET_REASON, GLOBAL_PRESSURE_REASON})


class ResourcePolicyError(ValueError):
    pass


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourcePolicyError(f"{label} must be a nonnegative integer")
    return value


def resource_stop_decision(*, reason: str, measured_cell_rss_bytes: int,
                           process_budget_bytes: int,
                           previous_consecutive_stops: int,
                           max_resource_stops: int) -> dict[str, Any]:
    """Classify one already-authorized safety shed.

    The caller has already decided to terminate a lane for aggregate or global
    host safety.  This decision answers only whether that event is truthful
    per-cell evidence for blocked-execution.
    """
    if reason not in VALID_REASONS:
        raise ResourcePolicyError("resource stop reason is not classified")
    measured = _nonnegative_int(measured_cell_rss_bytes,
                                "measured_cell_rss_bytes")
    budget = _nonnegative_int(process_budget_bytes, "process_budget_bytes")
    previous = _nonnegative_int(previous_consecutive_stops,
                                "previous_consecutive_stops")
    maximum = _nonnegative_int(max_resource_stops, "max_resource_stops")
    if budget <= 0 or maximum <= 0:
        raise ResourcePolicyError("resource budget and stop bound must be positive")

    cell_at_budget = measured >= budget
    if cell_at_budget:
        count = previous + 1
        classification = "measured-cell-budget"
        escalate = True
        next_count = 0
        detail = (f"measured cell RSS {measured} reaches the process RAM budget "
                  f"{budget}")
    elif reason == GLOBAL_PRESSURE_REASON:
        # An unrelated process or the OS can cause this shed. It breaks, rather
        # than advances, a consecutive per-cell aggregate-contention streak.
        count = 0
        classification = "global-pressure-or-swap"
        escalate = False
        next_count = 0
        detail = "global pressure/swap is not per-cell residency evidence"
    else:
        count = previous + 1
        classification = "aggregate-pool-contention"
        escalate = count >= maximum
        next_count = 0 if escalate else count
        detail = (f"{count} consecutive aggregate resource-stops reached the bound "
                  f"({maximum})" if escalate else
                  f"aggregate contention retry {count} of {maximum}")

    return {
        "schema": SCHEMA,
        "reason": reason,
        "classification": classification,
        "measured_cell_rss_bytes": measured,
        "process_budget_bytes": budget,
        "cell_at_or_over_budget": cell_at_budget,
        "previous_consecutive_stops": previous,
        "consecutive_stops": count,
        "next_consecutive_stops": next_count,
        "max_resource_stops": maximum,
        "escalate": escalate,
        "detail": detail,
    }


def fixed_thread_cpu_launch_decision(
        samples: list[float], *, logical_cores: int, guard_cores: float,
        launch_threads: float) -> dict[str, Any]:
    """Gate one fixed-thread launch against total-host CPU occupancy.

    ``samples`` is a trailing window maintained by the supervisor. Charging the
    maximum makes a saturated observation persist until it ages out, providing
    recovery hysteresis while still blocking immediately on the first saturated
    sample. The function only controls new launches; it has no stop authority.
    """
    if isinstance(logical_cores, bool) or not isinstance(logical_cores, int) \
            or logical_cores <= 0:
        raise ResourcePolicyError("logical_cores must be a positive integer")
    if isinstance(guard_cores, bool) or not isinstance(guard_cores, (int, float)) \
            or not math.isfinite(float(guard_cores)) \
            or not 0 <= float(guard_cores) < logical_cores:
        raise ResourcePolicyError("guard_cores is outside the CPU envelope")
    if isinstance(launch_threads, bool) \
            or not isinstance(launch_threads, (int, float)) \
            or not math.isfinite(float(launch_threads)) \
            or float(launch_threads) <= 0:
        raise ResourcePolicyError("launch_threads must be finite and positive")
    if not isinstance(samples, list) or not samples:
        raise ResourcePolicyError("at least one global CPU sample is required")
    values: list[float] = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0:
            raise ResourcePolicyError("global CPU samples are invalid")
        values.append(float(value))
    budget = float(logical_cores) - float(guard_cores)
    charged = max(values)
    available = max(0.0, budget - charged)
    tokens = int(available // float(launch_threads))
    return {
        "samples": values,
        "logical_cores": logical_cores,
        "guard_cores": float(guard_cores),
        "budget_cores": budget,
        "charged_global_cpu_cores": charged,
        "available_cpu_cores": available,
        "launch_threads": float(launch_threads),
        "launch_tokens": tokens,
        "ok": tokens >= 1,
        "blockers": ([] if tokens >= 1 else [
            "total-host CPU leaves no full fixed-thread launch token"
        ]),
        "recovery_hysteresis_samples": len(values),
    }
