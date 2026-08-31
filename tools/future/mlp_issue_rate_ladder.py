#!/usr/bin/env python3
"""IF NOT FMA COUNT, WHAT? A monotone total-ops/byte ladder.

fold_addqx cut total FMA/byte eight-fold (2.6667 -> 0.3333) and bought 12.7%.
ARM A deleted almost everything, including the eight MACs, and bought 50.9%.
FMA count is therefore not the ceiling. This sidecar discriminates the four
still-open branches of mlp.why_330 with one back-to-back experiment:

    B2  conversion
    B3  accumulation
    C1  instruction dependency chain
    C2  register pressure

The ladder decreases TOTAL instruction count per byte (every op, not only
FMA) monotonically from production down to ARM A. The two discriminators
hold op count fixed and vary ILP (independent accumulators 2/4/8) or the
per-thread working set / threads-per-threadgroup.

PRE-REGISTERED INTERPRETATION (locked before the GPU run, not after):

    GB/s tracks TOTAL ops/byte roughly linearly
        (Pearson r^2 of (ops/byte, gpu_ns) >= 0.80 with positive slope,
         and issue-rate (ops/byte * GB/s) max/min <= 1.25)
        -> ISSUE_RATE_BOUND. The target is total instructions per byte.
           Then name the cheapest remaining op and the rate it implies.

    GB/s plateaus (max/min of ladder GB/s <= 1.12) no matter how few ops
        -> NOT issue rate. Then the same-run discriminators:
           ILP gb_s(8)/gb_s(1) >= 1.12 at constant op count
               -> DEPENDENCY_BOUND (C1)
           working-set gb_s(ws32)/gb_s(ws0) <= 0.88 at constant op count
           AND occupancy max/min <= 1.05
               -> REGISTER_PRESSURE_BOUND (C2)
           both or neither -> MIXED

    Neither shape
        -> MIXED with the curve. Do not force a verdict.

    python3 tools/future/mlp_issue_rate_ladder.py --record
    python3 tools/future/mlp_issue_rate_ladder.py --from receipts/future/_MLP_ISSUE_RATE_LADDER_raw.json --record
    python3 tools/future/mlp_issue_rate_ladder.py --measure --record
    python3 -m pytest tools/future/test_mlp_issue_rate_ladder.py -q

evidence_class SELF_MEASURED_DIRTY. Absolute GB/s is measured-under-load.
The RATIO to production measured back to back in the same process is the
robust number. Does not change the production decode path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "MLP_ISSUE_RATE_LADDER.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_MLP_ISSUE_RATE_LADDER_raw.json"
SCHEMA = "hawking.future.mlp_issue_rate_ladder.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_issue_rate_ladder.py"

LM_HEAD_GB_S = 497.4
ARM_A_GB_S_CITED = 497.4
FOLD_ADDQX_GB_S_CITED = 370.9
PRODUCTION_GB_S_CITED = 329.2
MLP_MS = 15.541
TOKEN_MS = 28.722
TOKEN_TPS = 34.82
WEIGHT_BYTES_PER_TILE = 6

# Pre-registered bars. Not fitted after the fact.
LINEAR_R2_BAR = 0.80
RATE_SPAN_BAR = 1.25
STAY_RATIO = 1.12          # plateau: max GB/s / min GB/s on the ladder
ILP_JUMP_BAR = 1.12        # gb_s(8 acc) / gb_s(1 acc)
WS_DROP_BAR = 0.88         # gb_s(ws32) / gb_s(ws0)
OCC_SPAN_BAR = 1.05        # occupancy considered constant
MIN_RUNGS = 4

VERDICT_ISSUE = "ISSUE_RATE_BOUND"
VERDICT_DEP = "DEPENDENCY_BOUND"
VERDICT_REG = "REGISTER_PRESSURE_BOUND"
VERDICT_MIXED = "MIXED"

SHAPE_LINEAR = "LINEAR"
SHAPE_PLATEAU = "PLATEAU"
SHAPE_NEITHER = "NEITHER"

LADDER_IDS = ("production", "k6", "k4", "k2", "arm_a")
ILP_IDS = ("ilp2", "ilp4", "ilp8")
WS_IDS = ("ws8", "ws16", "ws32")

PRE_REGISTERED = {
    "locked_before_measurement": True,
    "linear": {
        "shape": SHAPE_LINEAR,
        "verdict": VERDICT_ISSUE,
        "pearson_r2_gpu_ns_vs_ops_per_byte_ge": LINEAR_R2_BAR,
        "slope_of_gpu_ns_vs_ops_must_be_positive": True,
        "issue_rate_ops_per_byte_times_gb_s_max_over_min_le": RATE_SPAN_BAR,
        "then": (
            "The target is total instructions per byte, not FMAs. "
            "Name the cheapest remaining op on the lowest-ops rung and the "
            "issue rate (ops/byte * GB/s) it implies."
        ),
    },
    "plateau": {
        "shape": SHAPE_PLATEAU,
        "ladder_gb_s_max_over_min_le": STAY_RATIO,
        "then_not": "ISSUE_RATE_BOUND",
        "c1": {
            "verdict": VERDICT_DEP,
            "ilp_gb_s_8_over_1_ge": ILP_JUMP_BAR,
            "at_constant_op_count": True,
            "fix": "unrolling and more accumulators rather than fewer operations",
        },
        "c2": {
            "verdict": VERDICT_REG,
            "ws32_over_ws0_le": WS_DROP_BAR,
            "occupancy_max_over_min_le": OCC_SPAN_BAR,
            "at_constant_op_count": True,
        },
        "both_or_neither": VERDICT_MIXED,
    },
    "neither": {
        "shape": SHAPE_NEITHER,
        "verdict": VERDICT_MIXED,
        "then": "report MIXED with the curve; do not force a verdict",
    },
}

CLAIM_BOUNDARY = (
    "One representative MLP layer (gate+up+down) on sealed-3.14, "
    "SELF_MEASURED_DIRTY. GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime "
    "for an isolated command buffer of three geo_tpr64 dispatches. Bytes are "
    "GPU-resident codes+scales+biases of the launched tensors. Bandwidth is "
    "those bytes divided by GPU ns (perfect-locality). Absolute GB/s is "
    "measured-under-load; the robust number is the back-to-back ratio to the "
    "production kernel in the same process. Bit-identity is a byte comparison "
    "of output buffers against that production kernel. Op counts are a static "
    "inner-loop tax counted from issue_ladder_mlp.metal the same way "
    "MLP_ALU_ROOFLINE counted the production body — not a hardware counter. "
    "The verdict is forced by the pre-registered curve shape plus the two "
    "constant-op-count discriminators; it is one of ISSUE_RATE_BOUND / "
    "DEPENDENCY_BOUND / REGISTER_PRESSURE_BOUND / MIXED. Does not change the "
    "production decode path. Does not re-run region granularity, catalog "
    "addressing, dispatch count, decode fusion, LUT-composed-with-vec4, or "
    "the bare fold. half_mac is not a candidate."
)

# Static inner-loop tax per 8-weight tile / 6 B. See issue_ladder_mlp.metal.
# Tile overhead (every rung that keeps the access pattern):
#   integer 6, conversion 2, memory 3, control 1, plus 8 x-loads.
# Per production weight slot: integer 2 + conv 1 + fma 2 + mem 1.
# Per xor-sink weight slot:   integer 1 + mem 1.
# ARM A: fma=2 (scale+bias FADD), integer 6+1+8, conv 2, mem 11, control 1.


def _tax(
    *,
    fma: int,
    integer: int,
    conversion: int,
    memory: int,
    control: int,
    note: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    total = fma + integer + conversion + memory + control
    nbytes = WEIGHT_BYTES_PER_TILE
    out = {
        "fma": fma,
        "integer": integer,
        "conversion": conversion,
        "memory": memory,
        "control": control,
        "total": total,
        "weight_bytes_per_iteration": nbytes,
        "ops_per_byte_by_class": {
            "fma": round(fma / nbytes, 4),
            "integer": round(integer / nbytes, 4),
            "conversion": round(conversion / nbytes, 4),
            "memory": round(memory / nbytes, 4),
            "control": round(control / nbytes, 4),
        },
        "total_ops_per_byte": round(total / nbytes, 4),
        "note": note,
    }
    if extra:
        out.update(extra)
    return out


INNER_LOOP_TAX: dict[str, dict[str, Any]] = {
    "production": _tax(
        fma=16, integer=22, conversion=10, memory=11, control=1,
        note="8 dequant FMA + 8 MAC FMA + 16 extract + tile overhead",
        extra={"keep_k": 8, "n_accumulators": 1, "n_live_floats": 1},
    ),
    "k6": _tax(
        fma=12, integer=20, conversion=8, memory=11, control=1,
        note="6 production slots + 2 xor-sink x loads",
        extra={"keep_k": 6, "n_accumulators": 1, "n_live_floats": 1},
    ),
    "k4": _tax(
        fma=8, integer=18, conversion=6, memory=11, control=1,
        note="4 production slots + 4 xor-sink x loads",
        extra={"keep_k": 4, "n_accumulators": 1, "n_live_floats": 1},
    ),
    "k2": _tax(
        fma=4, integer=16, conversion=4, memory=11, control=1,
        note="2 production slots + 6 xor-sink x loads",
        extra={"keep_k": 2, "n_accumulators": 1, "n_live_floats": 1},
    ),
    "arm_a": _tax(
        fma=2, integer=15, conversion=2, memory=11, control=1,
        note="ARM A XOR/add sink; fma class is 2 FADD of scale+bias",
        extra={"keep_k": 0, "n_accumulators": 1, "n_live_floats": 1},
    ),
    "ilp2": _tax(
        fma=17, integer=22, conversion=10, memory=11, control=1,
        note="16 production FMA + 1 combine FADD; 2 acc chains",
        extra={"keep_k": 8, "n_accumulators": 2, "n_live_floats": 2, "combine_adds": 1},
    ),
    "ilp4": _tax(
        fma=19, integer=22, conversion=10, memory=11, control=1,
        note="16 production FMA + 3 combine FADD; 4 acc chains",
        extra={"keep_k": 8, "n_accumulators": 4, "n_live_floats": 4, "combine_adds": 3},
    ),
    "ilp8": _tax(
        fma=23, integer=22, conversion=10, memory=11, control=1,
        note="16 production FMA + 7 combine FADD; 8 acc chains",
        extra={"keep_k": 8, "n_accumulators": 8, "n_live_floats": 8, "combine_adds": 7},
    ),
    "ws8": _tax(
        fma=16, integer=25, conversion=10, memory=11, control=1,
        note="production inner loop + 3 integer for rotating 8 live floats",
        extra={"keep_k": 8, "n_accumulators": 1, "n_live_floats": 8},
    ),
    "ws16": _tax(
        fma=16, integer=25, conversion=10, memory=11, control=1,
        note="production inner loop + 3 integer for rotating 16 live floats",
        extra={"keep_k": 8, "n_accumulators": 1, "n_live_floats": 16},
    ),
    "ws32": _tax(
        fma=16, integer=25, conversion=10, memory=11, control=1,
        note="production inner loop + 3 integer for rotating 32 live floats",
        extra={"keep_k": 8, "n_accumulators": 1, "n_live_floats": 32},
    ),
}


class MissingRung(Exception):
    """Raised rather than emit a verdict over an incomplete ladder."""


class EmptyGpuSample(Exception):
    """Raised rather than divide by a missing GPU timestamp."""


class NoByteComparison(Exception):
    """Raised rather than stamp bit-identical without comparing output bytes."""


class BitIdentityClaimWithoutCompare(Exception):
    """Raised when a rung claims bit-identical but has no byte comparison."""


class LadderNotMonotone(Exception):
    """Raised rather than report a verdict over a non-monotone ops/byte ladder."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Return (r^2, r). r is 0 if either series is constant."""
    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("pearson_r needs two series of equal length >= 2")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0, 0.0
    r = num / (dx ** 0.5 * dy ** 0.5)
    return r * r, r


def require_byte_compare(variant: Mapping[str, Any]) -> Mapping[str, Any]:
    cmp_ = variant.get("byte_compare")
    ident = str(variant.get("id") or variant.get("kernel") or "?")
    if not isinstance(cmp_, Mapping):
        raise NoByteComparison(
            f"refusing bit-identical for {ident}: no byte_compare against "
            "the production kernel output"
        )
    n = int(cmp_.get("n_bytes_compared") or 0)
    if n <= 0:
        raise NoByteComparison(
            f"refusing bit-identical for {ident}: n_bytes_compared={n} "
            "(no bytes were compared against the production kernel)"
        )
    if "n_mismatch_bytes" not in cmp_ and "n_float_mismatch" not in cmp_:
        raise NoByteComparison(
            f"refusing bit-identical for {ident}: byte_compare has no mismatch count"
        )
    return cmp_


def bit_identical_from_compare(variant: Mapping[str, Any]) -> bool:
    """True only after a real byte comparison against production output.

    A missing comparison is an error, not a False, when bit-identical is claimed.
    """
    if variant.get("bit_identical") is True and not isinstance(
        variant.get("byte_compare"), Mapping
    ):
        raise BitIdentityClaimWithoutCompare(
            f"refusing to report {variant.get('id')} as bit-identical "
            "without a byte comparison against the production kernel's output"
        )
    cmp_ = require_byte_compare(variant)
    n_bytes = int(cmp_["n_bytes_compared"])
    n_mis_b = int(cmp_.get("n_mismatch_bytes") or 0)
    n_mis_f = int(cmp_.get("n_float_mismatch") or 0)
    return n_bytes > 0 and n_mis_b == 0 and n_mis_f == 0


def require_monotone(rungs: Sequence[Mapping[str, Any]]) -> list[float]:
    """Raise unless total_ops_per_byte strictly decreases along the ladder."""
    if len(rungs) < MIN_RUNGS:
        raise MissingRung(
            f"refusing a verdict: ladder has {len(rungs)} rungs, need >= {MIN_RUNGS}"
        )
    ops = []
    ids = []
    for r in rungs:
        if "total_ops_per_byte" not in r:
            raise MissingRung(
                f"refusing a verdict: rung {r.get('id')} is missing total_ops_per_byte"
            )
        ops.append(float(r["total_ops_per_byte"]))
        ids.append(str(r.get("id")))
    for i in range(1, len(ops)):
        if not ops[i] < ops[i - 1]:
            raise LadderNotMonotone(
                "refusing a verdict: ladder is not monotone in total ops/byte: "
                + " -> ".join(f"{i_}={o}" for i_, o in zip(ids, ops))
            )
    return ops


def _require_raw_rung(raw_v: Mapping[str, Any], where: str) -> None:
    for field in ("id", "kernel", "weight_bytes", "gpu_ns_median", "byte_compare"):
        if field not in raw_v:
            raise MissingRung(f"{where} is missing {field}")
    if int(raw_v["gpu_ns_median"]) <= 0:
        raise EmptyGpuSample(
            f"{where} {raw_v.get('id')} gpu_ns_median must be positive"
        )
    if int(raw_v["weight_bytes"]) <= 0:
        raise ValueError(f"{where} weight_bytes must be positive")


def attach_rung(
    raw_v: Mapping[str, Any],
    *,
    tax_id: str | None = None,
    production_gb_s: float,
    production_ns: int,
) -> dict[str, Any]:
    ident = str(raw_v["id"])
    tax_key = tax_id or ident
    if ident.startswith("tg"):
        tax = dict(INNER_LOOP_TAX["production"])
        tax = {
            **tax,
            "note": "production inner-loop tax at varying threads per threadgroup",
            "n_accumulators": 1,
            "n_live_floats": 1,
            "keep_k": 8,
        }
    elif tax_key not in INNER_LOOP_TAX:
        raise MissingRung(
            f"unknown rung id {ident}; tax table has {list(INNER_LOOP_TAX)}"
        )
    else:
        tax = INNER_LOOP_TAX[tax_key]
    weight_bytes = int(raw_v["weight_bytes"])
    gpu_ns = int(raw_v["gpu_ns_median"])
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    claimed = raw_v.get("bit_identical")
    identical = bit_identical_from_compare(raw_v)
    if claimed is True and not identical:
        raise BitIdentityClaimWithoutCompare(
            f"refusing to report {ident} as bit-identical: byte_compare shows a mismatch"
        )
    cmp_ = require_byte_compare(raw_v)
    occ = raw_v.get("occupancy") or {}
    out: dict[str, Any] = {
        "id": ident,
        "kernel": raw_v.get("kernel"),
        "family": raw_v.get("family"),
        "note": raw_v.get("note") or tax.get("note"),
        "keep_k": raw_v.get("keep_k", tax.get("keep_k")),
        "n_accumulators": int(raw_v.get("n_accumulators") or tax.get("n_accumulators") or 1),
        "n_live_floats": int(raw_v.get("n_live_floats") or tax.get("n_live_floats") or 1),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw_v.get("gpu_ns_reps", [])],
        "gpu_us_median": round(gpu_ns / 1e3, 1),
        "effective_gb_s": round(gb_s, 1),
        "ratio_to_this_run_production": (
            round(gb_s / production_gb_s, 4) if production_gb_s > 0 else None
        ),
        "inner_loop": tax,
        "ops_per_byte_by_class": tax["ops_per_byte_by_class"],
        "total_ops_per_byte": tax["total_ops_per_byte"],
        "fma_per_weight_byte": tax["ops_per_byte_by_class"]["fma"],
        "issue_rate_ops_per_ns": round(
            tax["total_ops_per_byte"] * gb_s, 4
        ),  # (ops/byte)*(bytes/ns) in GB-ops units
        "byte_compare": dict(cmp_),
        "bit_identical": identical,
        "dispatches": int(raw_v.get("dispatches", 3)),
        "encoders": int(raw_v.get("encoders", 1)),
        "command_buffers": int(raw_v.get("command_buffers", 1)),
        "threads_per_threadgroup": int(raw_v.get("threads_per_threadgroup", 128)),
        "occupancy": occ,
        "projection": project_from_probe(production_ns, gpu_ns),
    }
    return out


def project_from_probe(production_gpu_ns: int, variant_gpu_ns: int) -> dict[str, Any]:
    if production_gpu_ns <= 0 or variant_gpu_ns <= 0:
        raise EmptyGpuSample("projection requires positive GPU ns on both arms")
    speedup = production_gpu_ns / variant_gpu_ns
    mlp_ms = MLP_MS / speedup
    saved_ms = MLP_MS - mlp_ms
    token_ms = TOKEN_MS - saved_ms
    tps = 1000.0 / token_ms if token_ms > 0 else None
    return {
        "kind": "projection",
        "label": "projection from a single-layer probe, not a resident measurement",
        "probe_speedup": round(speedup, 4),
        "baseline_mlp_ms": MLP_MS,
        "baseline_token_ms": TOKEN_MS,
        "baseline_tps": TOKEN_TPS,
        "mlp_ms_projected": round(mlp_ms, 3),
        "token_ms_projected": round(token_ms, 3),
        "tps_projected": None if tps is None else round(tps, 2),
        "delta_mlp_ms": round(-saved_ms, 3),
        "delta_token_ms": round(-saved_ms, 3),
        "delta_tps": None if tps is None else round(tps - TOKEN_TPS, 2),
        "note": (
            "PROJECTION from a single-layer probe. Not a resident measurement."
        ),
    }


def _list_from_raw(raw: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    v = raw.get(key)
    if not isinstance(v, list) or not v:
        raise MissingRung(f"refusing a verdict: raw measurement has no {key}")
    out: list[Mapping[str, Any]] = []
    for i, item in enumerate(v):
        if not isinstance(item, Mapping):
            raise MissingRung(f"{key}[{i}] is not an object")
        out.append(item)
    return out


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    ladder_raw = _list_from_raw(raw, "ladder")
    ilp_raw = _list_from_raw(raw, "ilp")
    ws_raw = _list_from_raw(raw, "register_pressure")
    tg_raw = raw.get("threadgroup") or []
    if not isinstance(tg_raw, list):
        raise MissingRung("threadgroup must be a list")

    ids = [str(v.get("id")) for v in ladder_raw]
    if "production" not in ids:
        raise MissingRung("refusing a verdict: production rung is missing")
    missing = [v for v in LADDER_IDS if v not in ids]
    if missing:
        raise MissingRung(f"refusing a verdict: missing ladder rungs {missing}")
    ilp_ids = [str(v.get("id")) for v in ilp_raw]
    missing_ilp = [v for v in ILP_IDS if v not in ilp_ids]
    if missing_ilp:
        raise MissingRung(f"refusing a verdict: missing ILP rungs {missing_ilp}")
    ws_ids = [str(v.get("id")) for v in ws_raw]
    missing_ws = [v for v in WS_IDS if v not in ws_ids]
    if missing_ws:
        raise MissingRung(
            f"refusing a verdict: missing register-pressure rungs {missing_ws}"
        )

    prod_raw = next(v for v in ladder_raw if v.get("id") == "production")
    _require_raw_rung(prod_raw, "ladder.production")
    prod_bytes = int(prod_raw["weight_bytes"])
    prod_ns = int(prod_raw["gpu_ns_median"])
    prod_gb = effective_gb_s(prod_bytes, prod_ns)

    def attach_all(items: Sequence[Mapping[str, Any]], where: str) -> list[dict[str, Any]]:
        attached: list[dict[str, Any]] = []
        for i, v in enumerate(items):
            _require_raw_rung(v, f"{where}[{i}]")
            if int(v["weight_bytes"]) != prod_bytes:
                raise ValueError(
                    f"{where} {v.get('id')} weight_bytes {v['weight_bytes']} != "
                    f"production {prod_bytes}"
                )
            attached.append(
                attach_rung(v, production_gb_s=prod_gb, production_ns=prod_ns)
            )
        return attached

    ladder = attach_all(ladder_raw, "ladder")
    # Force the declared ladder order so monotone is a property of the experiment,
    # not of however the JSON happened to be written.
    by_id = {r["id"]: r for r in ladder}
    ladder_ordered = [by_id[i] for i in LADDER_IDS if i in by_id]
    require_monotone(ladder_ordered)

    ilp = attach_all(ilp_raw, "ilp")
    ws = attach_all(ws_raw, "register_pressure")
    tg = attach_all(tg_raw, "threadgroup") if tg_raw else []

    return {
        "layer": int(raw.get("layer", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "concurrent_load": raw.get("concurrent_load") or {},
        "concurrent_load_end": raw.get("concurrent_load_end") or {},
        "absolute_gb_s_are_measured_under_load": True,
        "organ": raw.get("organ", "mlp"),
        "codec": raw.get("codec"),
        "geometry": raw.get("geometry", "geo_tpr64_tg128"),
        "production_kernel": raw.get("production_kernel"),
        "projections": raw.get("projections"),
        "weight_bytes": prod_bytes,
        "inner_loop_trips_gate_up": raw.get("inner_loop_trips_gate_up"),
        "bytes_per_thread_iteration": raw.get("bytes_per_thread_iteration"),
        "production_gb_s": round(prod_gb, 1),
        "production_gpu_ns_median": prod_ns,
        "ladder": ladder_ordered,
        "ilp": ilp,
        "register_pressure": ws,
        "threadgroup": tg,
    }


def cheapest_remaining_op(rung: Mapping[str, Any]) -> dict[str, Any]:
    by = rung["ops_per_byte_by_class"]
    # "Cheapest remaining" = the class that still dominates the lowest-ops rung.
    ranked = sorted(by.items(), key=lambda kv: kv[1], reverse=True)
    top_cls, top_ops = ranked[0]
    gb = float(rung["effective_gb_s"])
    return {
        "rung": rung["id"],
        "dominant_class": top_cls,
        "ops_per_byte": top_ops,
        "class_share_of_total": round(top_ops / float(rung["total_ops_per_byte"]), 4)
        if float(rung["total_ops_per_byte"])
        else None,
        "rung_gb_s": gb,
        "implied_issue_rate_class_ops_times_gb_s": round(top_ops * gb, 4),
        "ranked": [{"class": c, "ops_per_byte": o} for c, o in ranked],
        "note": (
            "Static tax, not a hardware counter. The dominant class on the "
            "lowest-ops rung is the cheapest remaining work that is still issued."
        ),
    }


def ladder_shape(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ops = require_monotone(rungs)
    ns = [float(r["gpu_ns_median"]) for r in rungs]
    gbs = [float(r["effective_gb_s"]) for r in rungs]
    r2, r = pearson_r(ops, ns)
    slope_positive = r > 0.0
    rates = [float(rung["total_ops_per_byte"]) * float(rung["effective_gb_s"]) for rung in rungs]
    rate_min = min(rates)
    rate_span = (max(rates) / rate_min) if rate_min > 0 else float("inf")
    gb_min = min(gbs)
    gb_span = (max(gbs) / gb_min) if gb_min > 0 else float("inf")
    linear = bool(r2 >= LINEAR_R2_BAR and slope_positive and rate_span <= RATE_SPAN_BAR)
    plateau = bool(gb_span <= STAY_RATIO)
    if linear and not plateau:
        shape = SHAPE_LINEAR
    elif plateau:
        shape = SHAPE_PLATEAU
    else:
        shape = SHAPE_NEITHER
    return {
        "shape": shape,
        "pearson_r2_gpu_ns_vs_ops_per_byte": round(r2, 4),
        "pearson_r": round(r, 4),
        "slope_positive": slope_positive,
        "issue_rate_span": None if rate_span == float("inf") else round(rate_span, 4),
        "gb_s_span": None if gb_span == float("inf") else round(gb_span, 4),
        "linear": linear,
        "plateau": plateau,
        "linear_r2_bar": LINEAR_R2_BAR,
        "rate_span_bar": RATE_SPAN_BAR,
        "stay_ratio_bar": STAY_RATIO,
        "ops_per_byte": [round(o, 4) for o in ops],
        "gpu_ns": [int(n) for n in ns],
        "effective_gb_s": [round(g, 1) for g in gbs],
        "issue_rate_ops_times_gb_s": [round(x, 4) for x in rates],
    }


def ilp_judgement(
    production: Mapping[str, Any], ilp: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_n = {1: production}
    for r in ilp:
        by_n[int(r["n_accumulators"])] = r
    missing = [n for n in (1, 2, 4, 8) if n not in by_n]
    if missing:
        raise MissingRung(
            f"refusing a verdict: ILP discriminator missing n_accumulators={missing}"
        )
    g1 = float(by_n[1]["effective_gb_s"])
    g8 = float(by_n[8]["effective_gb_s"])
    ratio = g8 / g1 if g1 > 0 else float("inf")
    jumped = bool(ratio >= ILP_JUMP_BAR)
    # Constant op count of the *inner FMA*: 16 on every ILP arm. Combine adds
    # grow; they are reported and would bias AGAINST a jump.
    return {
        "jumped": jumped,
        "gb_s_1": round(g1, 1),
        "gb_s_2": round(float(by_n[2]["effective_gb_s"]), 1),
        "gb_s_4": round(float(by_n[4]["effective_gb_s"]), 1),
        "gb_s_8": round(g8, 1),
        "ratio_8_over_1": None if ratio == float("inf") else round(ratio, 4),
        "jump_bar": ILP_JUMP_BAR,
        "inner_fma_count_held_at": 16,
        "points": [
            {
                "n_accumulators": n,
                "id": by_n[n]["id"],
                "effective_gb_s": round(float(by_n[n]["effective_gb_s"]), 1),
                "total_ops_per_byte": by_n[n]["total_ops_per_byte"],
            }
            for n in (1, 2, 4, 8)
        ],
    }


def occupancy_of(rung: Mapping[str, Any]) -> float | None:
    occ = rung.get("occupancy") or {}
    v = occ.get("occupancy_of_max_threads")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def register_judgement(
    production: Mapping[str, Any], ws: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_n = {1: production}  # n_live_floats; production is the 1-live-acc control
    # Map by n_live_floats. production reports 1; ws rungs report 8/16/32.
    by_n[int(production.get("n_live_floats") or 1)] = production
    for r in ws:
        by_n[int(r["n_live_floats"])] = r
    if 32 not in by_n:
        raise MissingRung("refusing a verdict: register-pressure missing ws32")
    g0 = float(production["effective_gb_s"])
    g32 = float(by_n[32]["effective_gb_s"])
    ratio = g32 / g0 if g0 > 0 else float("inf")
    occs = []
    for r in [production, *ws]:
        o = occupancy_of(r)
        if o is not None and o > 0:
            occs.append(o)
    if occs:
        occ_span = max(occs) / min(occs)
        constant_occ = occ_span <= OCC_SPAN_BAR
    else:
        occ_span = None
        constant_occ = False
    dropped = bool(ratio <= WS_DROP_BAR)
    pressed = bool(dropped and constant_occ)
    return {
        "pressed": pressed,
        "dropped": dropped,
        "constant_occupancy": constant_occ,
        "gb_s_ws0": round(g0, 1),
        "gb_s_ws32": round(g32, 1),
        "ratio_ws32_over_ws0": None if ratio == float("inf") else round(ratio, 4),
        "drop_bar": WS_DROP_BAR,
        "occupancy_span": None if occ_span is None else round(occ_span, 4),
        "occupancy_span_bar": OCC_SPAN_BAR,
        "points": [
            {
                "n_live_floats": int(r["n_live_floats"]),
                "id": r["id"],
                "effective_gb_s": round(float(r["effective_gb_s"]), 1),
                "total_ops_per_byte": r["total_ops_per_byte"],
                "occupancy_of_max_threads": occupancy_of(r),
                "threads_per_threadgroup": r.get("threads_per_threadgroup"),
            }
            for r in [production, *ws]
        ],
    }


def threadgroup_judgement(tg: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not tg:
        return {"ran": False, "points": []}
    gbs = [float(r["effective_gb_s"]) for r in tg]
    gmin = min(gbs) if gbs else 0.0
    span = (max(gbs) / gmin) if gmin > 0 else None
    return {
        "ran": True,
        "gb_s_span": None if span is None else round(span, 4),
        "moved": bool(span is not None and span >= STAY_RATIO),
        "stay_ratio_bar": STAY_RATIO,
        "points": [
            {
                "id": r["id"],
                "threads_per_threadgroup": r.get("threads_per_threadgroup"),
                "effective_gb_s": round(float(r["effective_gb_s"]), 1),
                "occupancy_of_max_threads": occupancy_of(r),
                "bit_identical": r.get("bit_identical"),
            }
            for r in tg
        ],
        "note": (
            "TG size changes occupancy_of_max_threads. C2 is forced by the "
            "working-set sweep at constant TG=128, not by this sweep."
        ),
    }


def judge(measurement: Mapping[str, Any]) -> dict[str, Any]:
    ladder = measurement["ladder"]
    production = next(r for r in ladder if r["id"] == "production")
    shape = ladder_shape(ladder)
    ilp = ilp_judgement(production, measurement["ilp"])
    ws = register_judgement(production, measurement["register_pressure"])
    tg = threadgroup_judgement(measurement.get("threadgroup") or [])
    if shape["shape"] == SHAPE_LINEAR:
        verdict = VERDICT_ISSUE
        why = (
            f"ladder gpu_ns tracks total ops/byte (r^2={shape['pearson_r2_gpu_ns_vs_ops_per_byte']}, "
            f"issue-rate span {shape['issue_rate_span']} <= {RATE_SPAN_BAR})"
        )
    elif shape["shape"] == SHAPE_PLATEAU:
        if ilp["jumped"] and not ws["pressed"]:
            verdict = VERDICT_DEP
            why = (
                f"GB/s plateau (span {shape['gb_s_span']} <= {STAY_RATIO}) and "
                f"ILP 8/1 = {ilp['ratio_8_over_1']} >= {ILP_JUMP_BAR} at constant FMA count"
            )
        elif ws["pressed"] and not ilp["jumped"]:
            verdict = VERDICT_REG
            why = (
                f"GB/s plateau (span {shape['gb_s_span']} <= {STAY_RATIO}) and "
                f"ws32/ws0 = {ws['ratio_ws32_over_ws0']} <= {WS_DROP_BAR} "
                f"at constant occupancy (span {ws['occupancy_span']})"
            )
        else:
            verdict = VERDICT_MIXED
            why = (
                f"GB/s plateau (span {shape['gb_s_span']}) but discriminators "
                f"do not isolate C1 vs C2 (ILP jumped={ilp['jumped']}, "
                f"WS pressed={ws['pressed']})"
            )
    else:
        verdict = VERDICT_MIXED
        why = (
            f"ladder shape is neither linear nor plateau "
            f"(r^2={shape['pearson_r2_gpu_ns_vs_ops_per_byte']}, "
            f"issue-rate span {shape['issue_rate_span']}, "
            f"GB/s span {shape['gb_s_span']})"
        )
    return {
        "verdict": verdict,
        "why": why,
        "shape": shape,
        "ilp": ilp,
        "register_pressure": ws,
        "threadgroup": tg,
        "cheapest_remaining": cheapest_remaining_op(ladder[-1]),
    }


def _finding(measurement: Mapping[str, Any], judged: Mapping[str, Any]) -> str:
    prod = measurement["production_gb_s"]
    ladder = measurement["ladder"]
    shape = judged["shape"]
    lines = [
        f"This-run production {prod} GB/s ({measurement['production_gpu_ns_median']} ns). "
        f"Cited fold_addqx {FOLD_ADDQX_GB_S_CITED} GB/s / ARM A {ARM_A_GB_S_CITED} GB/s. "
        f"Ladder shape {shape['shape']} (r^2 gpu_ns~ops/byte "
        f"{shape['pearson_r2_gpu_ns_vs_ops_per_byte']}, issue-rate span "
        f"{shape['issue_rate_span']}, GB/s span {shape['gb_s_span']}). "
        f"Verdict {judged['verdict']}: {judged['why']}."
    ]
    bits = []
    for r in ladder:
        bits.append(
            f"{r['id']} {r['total_ops_per_byte']} ops/B "
            f"(FMA {r['ops_per_byte_by_class']['fma']}/int "
            f"{r['ops_per_byte_by_class']['integer']}/cv "
            f"{r['ops_per_byte_by_class']['conversion']}/mem "
            f"{r['ops_per_byte_by_class']['memory']}/ctrl "
            f"{r['ops_per_byte_by_class']['control']}) "
            f"{r['effective_gb_s']} GB/s {r['gpu_us_median']} us "
            f"{'IDENTICAL' if r['bit_identical'] else 'not-identical'}"
        )
    lines.append("Ladder: " + "; ".join(bits) + ".")
    ilp = judged["ilp"]
    lines.append(
        f"ILP (constant inner FMA=16): 1/2/4/8 acc -> "
        f"{ilp['gb_s_1']}/{ilp['gb_s_2']}/{ilp['gb_s_4']}/{ilp['gb_s_8']} GB/s "
        f"(8/1={ilp['ratio_8_over_1']}, jumped={ilp['jumped']})."
    )
    ws = judged["register_pressure"]
    lines.append(
        f"Register pressure (TG=128): ws0/ws32 {ws['gb_s_ws0']}/{ws['gb_s_ws32']} GB/s "
        f"(ratio {ws['ratio_ws32_over_ws0']}, occupancy span {ws['occupancy_span']}, "
        f"pressed={ws['pressed']})."
    )
    if judged["verdict"] == VERDICT_ISSUE:
        cheap = judged["cheapest_remaining"]
        lines.append(
            f"Cheapest remaining op class on {cheap['rung']}: {cheap['dominant_class']} "
            f"at {cheap['ops_per_byte']} ops/byte, implied class-rate "
            f"{cheap['implied_issue_rate_class_ops_times_gb_s']} "
            f"(ops/byte * GB/s). Production FMA/byte is not the target."
        )
    arm = next(r for r in ladder if r["id"] == "arm_a")
    p = arm["projection"]
    gap = ARM_A_GB_S_CITED - prod
    closed = arm["effective_gb_s"] - prod
    frac = closed / gap if gap else None
    lines.append(
        f"ARM A this-run {arm['effective_gb_s']} GB/s "
        f"({arm['ratio_to_this_run_production']}x production). "
        f"PROJECTION at ARM A: MLP {p['mlp_ms_projected']} ms, token "
        f"{p['token_ms_projected']} ms. Gap to cited 497.4: closed "
        f"{None if frac is None else round(frac, 4)} of {round(gap, 1)} GB/s "
        f"(this-run numbers; cited 329.2/370.9/497.4 are not this-run)."
    )
    return " ".join(lines)


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    if "ladder" not in measurement:
        raise MissingRung("refusing a receipt without a ladder")
    require_monotone(measurement["ladder"])
    for r in measurement["ladder"]:
        bit_identical_from_compare(r)
        if "effective_gb_s" not in r or float(r["effective_gb_s"]) <= 0:
            raise EmptyGpuSample(f"rung {r.get('id')} is missing measured GB/s")
        if "ops_per_byte_by_class" not in r:
            raise MissingRung(f"rung {r.get('id')} is missing ops_per_byte_by_class")
    judged = judge(measurement)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/issue_ladder_mlp.rs; "
            "crates/hawking-core/examples/issue_ladder_mlp.metal; "
            "geometry and buffers reused from decode_cheapen_mlp.rs / "
            "alu_roofline_organs.rs; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime); "
            "one representative layer of sealed-3.14, production MLP affine2 geo_tpr64"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "pre_registered": PRE_REGISTERED,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "arm_a_gb_s_cited": ARM_A_GB_S_CITED,
        "fold_addqx_gb_s_cited": FOLD_ADDQX_GB_S_CITED,
        "production_gb_s_cited": PRODUCTION_GB_S_CITED,
        "mlp_ms_baseline": MLP_MS,
        "token_ms_baseline": TOKEN_MS,
        "token_tps_baseline": TOKEN_TPS,
        "absolute_gb_s_are_measured_under_load": True,
        "layer": measurement.get("layer"),
        "warmup": measurement.get("warmup"),
        "reps": measurement.get("reps"),
        "organ": measurement.get("organ"),
        "codec": measurement.get("codec"),
        "geometry": measurement.get("geometry"),
        "production_kernel": measurement.get("production_kernel"),
        "projections": measurement.get("projections"),
        "weight_bytes": measurement.get("weight_bytes"),
        "production_gb_s": measurement.get("production_gb_s"),
        "production_gpu_ns_median": measurement.get("production_gpu_ns_median"),
        "ladder": measurement["ladder"],
        "ilp": measurement["ilp"],
        "register_pressure": measurement["register_pressure"],
        "threadgroup": measurement.get("threadgroup") or [],
        "judgement": judged,
        "verdict": judged["verdict"],
        "finding": _finding(measurement, judged),
        "timing": measurement.get("timing"),
        "concurrent_load": measurement.get("concurrent_load"),
        "concurrent_load_end": measurement.get("concurrent_load_end"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "does_not_edit_production_shaders": True,
        "refuted_elsewhere": [
            "region_granularity",
            "catalog_addressing",
            "raw_dispatch_count",
            "fuse_representation_decode",
            "lut4_vec4",
            "bare_fold",
            "half_mac",
        ],
        "open_branches_this_lane": ["B2", "B3", "C1", "C2"],
        "demoted_elsewhere": ["A", "A1", "A2", "A3"],
        "promoted_elsewhere": ["B1"],
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise MissingRung("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def example_binaries() -> list[Path]:
    names = ("issue_ladder_mlp",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "target",
            REPO / "workspace" / "ops" / "build" / "rust",
        ]
    )
    out: list[Path] = []
    for root in roots:
        for profile in ("release-fast", "release"):
            for name in names:
                p = root / profile / "examples" / name
                if p.is_file():
                    out.append(p)
    return out


def run_example(
    artifact_root: Path,
    *,
    layer: int = 0,
    warmup: int = 5,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "issue_ladder_mlp binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example issue_ladder_mlp`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--layer",
        str(layer),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{exe} exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/Users/scammermike/noetic/NOETIC_PARENT_A"),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
            layer=args.layer,
            warmup=args.warmup,
            reps=args.reps,
            out=RAW_DEFAULT,
        )
    elif args.raw_path:
        raw = load_raw(Path(args.raw_path))
    elif RAW_DEFAULT.is_file():
        raw = load_raw(RAW_DEFAULT)

    if raw is None:
        print(
            "no measurement: pass --from RAW.json, --measure, or write "
            f"{RAW_DEFAULT}",
            file=sys.stderr,
        )
        return 2

    measured = measurement_from_raw(raw)
    judged = judge(measured)
    if args.record:
        path = record(measured, path=args.out)
        print(f"wrote {path} verdict={judged['verdict']}")
        print(_finding(measured, judged))
    else:
        print(_finding(measured, judged))
        for v in measured["ladder"]:
            flag = "IDENTICAL" if v["bit_identical"] else "approx"
            print(
                f"  {v['id']:12s} {v['total_ops_per_byte']:7.4f} ops/B  "
                f"{v['effective_gb_s']:7.1f} GB/s  "
                f"{v['gpu_us_median']:7.1f} us  {flag}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
