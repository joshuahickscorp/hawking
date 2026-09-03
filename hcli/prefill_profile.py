"""Decompose native prefill cost by position, from the resident's step trace.

Prefill on this runtime is superlinear: 2500 prompt tokens took 170.7 s and 3099
took 385.4 s on the same body. That is a fact. "Quadratic attention" is a GUESS,
and an unsafe one here -- this body is HYBRID. It interleaves full-attention
(GQA) layers with recurrent DeltaNet layers, and the prefill implementation is
itself incomplete, so several mechanisms could produce a rising curve:

* full-attention scaling      -- per-step cost RISES with position (KV grows)
* projection / GEMM work      -- per-step cost FLAT, total linear
* matvec / f32 fallback       -- flat but high, and visible as fallback counts
* DeltaNet scan               -- flat per step; recurrent state is fixed size
* host / control overhead     -- encode_ns + submit_ns, not gpu_ns

Those have DIFFERENT SHAPES against position, which is why this module buckets
by position instead of reporting one mean. A rising gpu_ns curve implicates
attention; a flat curve with a large wall-minus-gpu gap implicates the host; a
flat curve with high gpu_ns implicates per-layer GEMM and says the total is
linear and the superlinearity is somewhere else entirely.

This reads the trace the resident ALREADY emits. It runs no model and competes
with no running mission.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _nums(values: Any) -> List[float]:
    if not isinstance(values, (list, tuple)):
        return []
    out: List[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def bucket_profile(
    step_trace: Dict[str, Any],
    *,
    prefill_steps: int,
    buckets: int = 8,
) -> Dict[str, Any]:
    """Per-position means over the PREFILL steps only.

    ``prefill_steps`` must be the number of steps actually stepped, which is
    ``prompt_tokens - prefix_reused_tokens`` when the KV prefix was reused. It
    is not ``prompt_tokens``: with reuse the trace is shorter than the prompt,
    and treating the whole trace as prefill silently swallows the decode steps.
    """
    wall = _nums(step_trace.get("wall_ns"))
    gpu = _nums(step_trace.get("gpu_ns"))
    encode = _nums(step_trace.get("encode_ns"))
    submit = _nums(step_trace.get("submit_ns"))
    wait = _nums(step_trace.get("wait_ns"))
    dispatches = _nums(step_trace.get("dispatches"))

    steps = max(0, min(int(prefill_steps or 0), len(wall)))
    if steps < 2:
        return {"buckets": [], "prefill_steps": steps, "note": "too few steps to shape"}

    width = max(1, steps // max(1, buckets))
    rows: List[Dict[str, Any]] = []
    start = 0
    while start < steps:
        stop = min(steps, start + width)
        row = {
            "from_step": start,
            "to_step": stop - 1,
            "steps": stop - start,
            "wall_ns_mean": _mean(wall[start:stop]),
            "gpu_ns_mean": _mean(gpu[start:stop]),
            "encode_ns_mean": _mean(encode[start:stop]),
            "submit_ns_mean": _mean(submit[start:stop]),
            "wait_ns_mean": _mean(wait[start:stop]),
            "dispatches_mean": _mean(dispatches[start:stop]),
        }
        rows.append(row)
        start = stop

    return {
        "prefill_steps": steps,
        "buckets": rows,
        "totals": {
            "wall_ns": sum(wall[:steps]),
            "gpu_ns": sum(gpu[:steps]),
            "encode_ns": sum(encode[:steps]),
            "submit_ns": sum(submit[:steps]),
            "wait_ns": sum(wait[:steps]),
            "dispatches": sum(dispatches[:steps]),
        },
    }


def attribute(profile: Dict[str, Any], layers: Optional[int] = None) -> Dict[str, Any]:
    """Name what the SHAPE implicates, and say what it cannot settle.

    This deliberately returns a hypothesis plus the discriminator that would
    confirm it, not a cause. The shape narrows the field; it does not identify
    a kernel. Only a per-kernel trace does that.
    """
    rows = profile.get("buckets") or []
    if len(rows) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "buckets": len(rows)}

    def series(key: str) -> List[float]:
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

    walls = series("wall_ns_mean")
    gpus = series("gpu_ns_mean")
    if len(walls) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "buckets": len(rows)}

    first_wall, last_wall = walls[0], walls[-1]
    wall_growth = (last_wall / first_wall) if first_wall else None
    gpu_growth = (gpus[-1] / gpus[0]) if len(gpus) >= 2 and gpus[0] else None

    totals = profile.get("totals") or {}
    wall_total = float(totals.get("wall_ns") or 0.0)
    gpu_total = float(totals.get("gpu_ns") or 0.0)
    host_total = float(totals.get("encode_ns") or 0.0) + float(
        totals.get("submit_ns") or 0.0
    )
    host_share = (host_total / wall_total) if wall_total else None
    gpu_share = (gpu_total / wall_total) if wall_total else None

    # A per-step cost that RISES with position is the signature of work that
    # grows with the sequence -- on this body, full attention over a growing KV.
    # A FLAT per-step cost means the total is linear in prompt length, and any
    # superlinearity measured end-to-end lives outside the step loop.
    if wall_growth is not None and wall_growth >= 1.25:
        shape = "RISING_WITH_POSITION"
        implicates = "full-attention scaling over a growing KV"
        discriminator = (
            "re-run one prompt with HAWKING_TRACE_DISPATCH=1 and compare the GQA "
            "attention kernel's share in the first and last position bucket; "
            "DeltaNet and the MLP GEMMs must stay flat if this is attention"
        )
    elif wall_growth is not None and wall_growth <= 0.8:
        shape = "FALLING_WITH_POSITION"
        implicates = "warmup, clock ramp, or first-token allocation, not sequence work"
        discriminator = "discard the first bucket and re-shape; the tail is the steady state"
    else:
        shape = "FLAT_WITH_POSITION"
        implicates = (
            "per-layer GEMM / projection and the DeltaNet scan, both of which are "
            "constant per step; total prefill is LINEAR in prompt length"
        )
        discriminator = (
            "if end-to-end prefill is nonetheless superlinear, the cost is NOT in "
            "the step loop -- measure prompt construction and tokenization"
        )

    out: Dict[str, Any] = {
        "shape": shape,
        "implicates": implicates,
        "next_discriminator": discriminator,
        "wall_ns_first_bucket": first_wall,
        "wall_ns_last_bucket": last_wall,
        "wall_growth_last_over_first": round(wall_growth, 4) if wall_growth else None,
        "gpu_growth_last_over_first": round(gpu_growth, 4) if gpu_growth else None,
    }
    if host_share is not None:
        out["host_control_share_of_wall"] = round(host_share, 4)
        out["gpu_share_of_wall"] = round(gpu_share, 4) if gpu_share is not None else None
        # encode+submit is host-side command construction. When it dominates,
        # no kernel change helps and the lever is dispatch count per step.
        if host_share >= 0.35:
            out["host_bound"] = True
            out["host_note"] = (
                "encode+submit is over a third of prefill wall: this is host "
                "command construction, not GPU work. The lever is dispatches "
                "per step, not a faster kernel."
            )
        else:
            out["host_bound"] = False
    # The structural reading, which the shape alone does not give. Prefill on
    # this body steps ONE PROMPT TOKEN AT A TIME through the same function
    # decode uses, so every prompt token costs a full decode step. That is a
    # property of the loop, not of any kernel, and no kernel change touches it.
    steps = float(profile.get("prefill_steps") or 0.0)
    dispatches = float(totals.get("dispatches") or 0.0)
    if steps > 0 and dispatches > 0:
        per_step = dispatches / steps
        out["dispatches_per_step"] = round(per_step, 1)
        out["total_dispatches"] = int(dispatches)
        if layers:
            out["dispatches_per_layer_per_step"] = round(per_step / layers, 2)
        out["structural"] = (
            f"prefill stepped {int(steps)} tokens one at a time for "
            f"{int(dispatches):,} dispatches. Prompt tokens are being paid for "
            f"at decode prices."
        )
        # Whether the per-layer kernel count is itself the problem, or whether
        # the loop shape is. Under ~12 per layer there is no fat to cut: the
        # kernels are the ones the architecture needs, and the only lever left
        # is running many tokens through them at once.
        if layers and per_step / layers <= 12.0:
            out["lever"] = (
                "BATCH THE PREFILL. Per-layer dispatch count is already tight, "
                "so cutting kernels buys nothing; stepping N tokens together "
                "turns per-layer GEMV into one GEMM and divides the dispatch "
                "count by N. Cost: the recurrent state must be advanced "
                "chunk-wise rather than per position."
            )
        elif layers:
            out["lever"] = (
                "CUT DISPATCHES PER LAYER first: the per-layer kernel count is "
                "high enough that fusing adjacent kernels pays before batching."
            )
    out["cannot_settle"] = (
        "which kernel. Shape narrows the field; only a per-kernel trace "
        "(HAWKING_TRACE_DISPATCH=1) attributes cost to attention vs DeltaNet vs MLP."
    )
    return out
