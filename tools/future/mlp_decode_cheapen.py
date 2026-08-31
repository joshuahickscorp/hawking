#!/usr/bin/env python3
"""Cheapen the MLP affine2 q2 decode arithmetic. Measured target, not a search.

The ALU-roofline matched pair (receipts/future/MLP_ALU_ROOFLINE.json) put
production MLP gate+up+down at 329.6 GB/s and ARM A (same bytes, ALU stripped)
at 497.4 GB/s — 1.51x — on geo_tpr64, 128 threads/threadgroup, one layer.
The organ is not occupancy-limited. The cost is the arithmetic:

    production decode          1.3333 dequant-FMA per weight-byte
                               (8 dequant FMA + 8 MAC FMA per 6 B)
    to sit at 497.4 GB/s       0.8835 decode-FMA per weight-byte
    required cheapening        1.509x

This sidecar runs variants that attack the 8 dequant FMAs (not the 8 MACs)
without changing what the organ computes. Exact-class variants must prove
bit-identity by a byte comparison against the production kernel's output.
Approx-class variants report error and are scored against
receipts/future/CAPABILITY_INFORMATION_MAP.json; the two classes are never
blended into one verdict.

    python3 tools/future/mlp_decode_cheapen.py --record
    python3 tools/future/mlp_decode_cheapen.py --from receipts/future/_MLP_DECODE_CHEAPEN_raw.json --record
    python3 tools/future/mlp_decode_cheapen.py --measure --record
    python3 -m pytest tools/future/test_mlp_decode_cheapen.py -q

evidence_class SELF_MEASURED_DIRTY. Absolute GB/s is measured-under-load.
The RATIO to production measured back to back in the same process is the
robust number. Projection to token ms is arithmetic over the measured
single-layer probe, not a resident measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "MLP_DECODE_CHEAPEN.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_MLP_DECODE_CHEAPEN_raw.json"
ALU_RECEIPT = REPO / "receipts" / "future" / "MLP_ALU_ROOFLINE.json"
CAP_MAP = REPO / "receipts" / "future" / "CAPABILITY_INFORMATION_MAP.json"
ORGAN_BW = REPO / "receipts" / "future" / "ORGAN_BANDWIDTH.json"
PATH_TO_71 = REPO / "receipts" / "future" / "PATH_TO_71.json"
SCHEMA = "hawking.future.mlp_decode_cheapen.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_decode_cheapen.py"

# Cited from the ALU-roofline receipt / organ bandwidth / PATH_TO_71.
# This-run production GB/s is measured; these are the named targets.
LM_HEAD_GB_S = 497.4
ARM_A_GB_S = 497.4
ARM_A_US = 168.0
U1ALU_PRODUCTION_GB_S = 329.6
U1ALU_PRODUCTION_US = 253.5
MLP_MS = 15.541
TOKEN_MS = 28.722
TOKEN_TPS = 34.82
WEIGHT_BYTES_PER_TILE = 6
WEIGHTS_PER_TILE = 8
PRODUCTION_DEQUANT_FMA = 8
PRODUCTION_MAC_FMA = 8
PRODUCTION_DECODE_FMA_PER_BYTE = PRODUCTION_DEQUANT_FMA / WEIGHT_BYTES_PER_TILE
PRODUCTION_FMA_PER_BYTE = (
    PRODUCTION_DEQUANT_FMA + PRODUCTION_MAC_FMA
) / WEIGHT_BYTES_PER_TILE
TARGET_DECODE_FMA_PER_BYTE = PRODUCTION_DECODE_FMA_PER_BYTE * (
    U1ALU_PRODUCTION_GB_S / LM_HEAD_GB_S
)
REQUIRED_CHEAPENING = LM_HEAD_GB_S / U1ALU_PRODUCTION_GB_S

# Capability map bars. Used only to score approx-class matvec error; not a
# generate gate and not a layer-output measurement.
COSINE_BAR = 0.99
HIDDEN_BAR = 0.99
GATE_BAR = 0.99

CLAIM_BOUNDARY = (
    "One representative MLP layer (gate+up+down) on sealed-3.14, "
    "SELF_MEASURED_DIRTY. GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime "
    "for an isolated command buffer of three geo_tpr64_tg128 dispatches. Bytes "
    "are GPU-resident codes+scales+biases of the launched tensors. Bandwidth is "
    "those bytes divided by GPU ns (perfect-locality). Absolute GB/s is "
    "measured-under-load; the robust number is the back-to-back ratio to the "
    "production kernel in the same process. Exact-class bit-identity is a byte "
    "comparison of output buffers against that production kernel, not a hash of "
    "the shader source. Approx-class error is the same-x matvec cosine / "
    "rel-fro / max-abs against production, scored against "
    "CAPABILITY_INFORMATION_MAP cosine_bar=0.99 as a proxy only — not a layer "
    "output, not a generate identity. Token-ms numbers tagged projection are "
    "arithmetic over the measured probe (MLP 15.541 ms of a 28.722 ms token) "
    "and are not a resident measurement. Does not change the production decode "
    "path."
)

# Static inner-loop tax. Counted from decode_cheapen_mlp.metal the same way
# MLP_ALU_ROOFLINE counted the production body. Not a hardware counter.
INNER_LOOP_TAX: dict[str, dict[str, Any]] = {
    "production": {
        "dequant_fma": 8,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "w=q*s+b then w*x, eight times",
    },
    "lut4_select": {
        "dequant_fma": 4,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "four production-formula w values, select, then 8 MAC",
    },
    "lut4_index": {
        "dequant_fma": 4,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "four production-formula w values, array index, then 8 MAC",
    },
    "vec4": {
        "dequant_fma": 8,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "same FMAs as production; float4 x loads",
    },
    "lut4_vec4": {
        "dequant_fma": 4,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "lut4_select composed with float4 x loads",
    },
    "unrolled": {
        "dequant_fma": 8,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "production association, fully unrolled 8-weight tile",
    },
    "fold": {
        "dequant_fma": 2,
        "mac_fma": 8,
        "extra_add": 8,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "8 FMA of q*x + 8 ADD of x + 2 FMA applying s,b once",
    },
    "fold_addqx": {
        "dequant_fma": 2,
        "mac_fma": 0,
        "extra_add": 16,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "note": "q*x as adds; 2 FMA applying s,b once; extra_add is an upper bound",
    },
    "half_mac": {
        "dequant_fma": 8,
        "mac_fma": 8,
        "extra_add": 0,
        "weight_bytes_per_iteration": WEIGHT_BYTES_PER_TILE,
        "weights_per_iteration": WEIGHTS_PER_TILE,
        "mac_dtype": "half",
        "note": "dequant still 8 f32 FMA; MAC accumulated in half",
    },
}

VARIANT_IDS = tuple(INNER_LOOP_TAX.keys())
EXACT_CANDIDATE_IDS = (
    "lut4_select",
    "lut4_index",
    "vec4",
    "lut4_vec4",
    "unrolled",
)
APPROX_CANDIDATE_IDS = ("fold", "fold_addqx", "half_mac")


class MissingVariant(Exception):
    """Raised rather than emit a receipt over an incomplete variant set."""


class EmptyGpuSample(Exception):
    """Raised rather than divide by a missing GPU timestamp."""


class NoByteComparison(Exception):
    """Raised rather than stamp bit-identical without comparing output bytes."""


class BitIdentityClaimWithoutCompare(Exception):
    """Raised when a variant claims bit-identical but has no byte comparison."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def f32(x: float) -> float:
    """Round a Python float to IEEE-754 binary32."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def tax_view(tax: Mapping[str, Any]) -> dict[str, Any]:
    deq = int(tax["dequant_fma"])
    mac = int(tax["mac_fma"])
    extra = int(tax.get("extra_add") or 0)
    nbytes = int(tax["weight_bytes_per_iteration"])
    out = {
        "dequant_fma": deq,
        "mac_fma": mac,
        "extra_add": extra,
        "alu_ops": deq + mac + extra,
        "weight_bytes_per_iteration": nbytes,
        "weights_per_iteration": int(tax["weights_per_iteration"]),
        "fma_per_weight_byte": round((deq + mac) / nbytes, 4),
        "decode_fma_per_weight_byte": round(deq / nbytes, 4),
        "alu_ops_per_weight_byte": round((deq + mac + extra) / nbytes, 4),
        "note": tax.get("note"),
    }
    if "mac_dtype" in tax:
        out["mac_dtype"] = tax["mac_dtype"]
    return out


def production_unpack8(
    packed16: int, scale: float, bias: float, x: list[float]
) -> float:
    """Production association in f32: sum_i (float(q)*scale+bias)*x_i."""
    acc = f32(0.0)
    s = f32(scale)
    b = f32(bias)
    for i in range(8):
        q = (int(packed16) >> (2 * i)) & 3
        w = f32(f32(float(q) * s) + b)
        acc = f32(acc + f32(w * f32(x[i])))
    return acc


def lut4_unpack8(packed16: int, scale: float, bias: float, x: list[float]) -> float:
    s = f32(scale)
    b = f32(bias)
    wtab = [f32(f32(float(q) * s) + b) for q in range(4)]
    acc = f32(0.0)
    for i in range(8):
        q = (int(packed16) >> (2 * i)) & 3
        acc = f32(acc + f32(wtab[q] * f32(x[i])))
    return acc


def fold_unpack8_f32(
    packed16: int, scale: float, bias: float, x: list[float]
) -> float:
    s = f32(scale)
    b = f32(bias)
    acc_qx = f32(0.0)
    acc_x = f32(0.0)
    for i in range(8):
        q = (int(packed16) >> (2 * i)) & 3
        xi = f32(x[i])
        acc_qx = f32(acc_qx + f32(float(q) * xi))
        acc_x = f32(acc_x + xi)
    return f32(f32(s * acc_qx) + f32(b * acc_x))


def fold_unpack8_reals(
    packed16: int, scale: float, bias: float, x: list[float]
) -> float:
    acc_qx = 0.0
    acc_x = 0.0
    lhs = 0.0
    for i in range(8):
        q = (int(packed16) >> (2 * i)) & 3
        xi = float(x[i])
        lhs += (float(q) * float(scale) + float(bias)) * xi
        acc_qx += float(q) * xi
        acc_x += xi
    rhs = float(scale) * acc_qx + float(bias) * acc_x
    return lhs, rhs


def affine_fold_identity_over_reals(
    packed16: int = 0b11_10_01_00_11_10_01_00,
    scale: float = 0.37,
    bias: float = -0.11,
    x: list[float] | None = None,
) -> dict[str, Any]:
    if x is None:
        x = [0.125 * (i % 17) - 1.0 for i in range(8)]
    lhs, rhs = fold_unpack8_reals(packed16, scale, bias, x)
    return {
        "identity": "sum((s*q_i+b)*x_i) = s*sum(q_i*x_i) + b*sum(x_i)",
        "lhs": lhs,
        "rhs": rhs,
        "abs_err": abs(lhs - rhs),
        "exact_over_reals": abs(lhs - rhs) <= 1e-12 * max(1.0, abs(lhs), abs(rhs)),
    }


def f32_counterexample_for_fold() -> dict[str, Any]:
    """A tile where production f32 and the fold disagree. Not a GPU result."""
    packed16 = 0xFFFF
    scale = 0.3
    bias = 0.1
    x = [0.7] * 8
    prod = production_unpack8(packed16, scale, bias, x)
    folded = fold_unpack8_f32(packed16, scale, bias, x)
    lut = lut4_unpack8(packed16, scale, bias, x)
    return {
        "packed16": packed16,
        "scale": scale,
        "bias": bias,
        "x": x,
        "production_f32": prod,
        "fold_f32": folded,
        "lut4_f32": lut,
        "fold_matches_production_f32": prod == folded,
        "lut4_matches_production_f32": prod == lut,
        "fold_abs_err": abs(prod - folded),
    }


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

    A missing comparison is an error, not a False. Callers that want to
    *claim* bit-identical must go through this.
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


def score_against_capability_map(cmp_: Mapping[str, Any]) -> dict[str, Any]:
    cosine = float(cmp_.get("cosine") or 0.0)
    rel_fro = float(cmp_.get("rel_fro") or 0.0)
    max_abs = float(cmp_.get("max_abs_err") or 0.0)
    return {
        "source": "receipts/future/CAPABILITY_INFORMATION_MAP.json",
        "cosine_bar": COSINE_BAR,
        "hidden_bar": HIDDEN_BAR,
        "gate_bar": GATE_BAR,
        "matvec_cosine_vs_production": cosine,
        "matvec_rel_fro_vs_production": rel_fro,
        "matvec_max_abs_vs_production": max_abs,
        "above_cosine_bar": cosine >= COSINE_BAR,
        "note": (
            "Same-x matvec cosine against the production kernel, scored on "
            "the capability map's cosine_bar=0.99. This is not a layer-output "
            "cosine, not a generate identity, and not a licence to ship. A "
            "bit drop in the capability map perturbs W; this scores a "
            "different rounding of the same W."
        ),
    }


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
        "from": (
            "arithmetic over measured GPU ns of one layer's gate+up+down "
            f"applied to MLP {MLP_MS} ms of a {TOKEN_MS} ms token "
            f"(PATH_TO_71 baseline / ORGAN_BANDWIDTH mlp.gpu_ms)"
        ),
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
            "PROJECTION from a single-layer probe. Not a resident measurement. "
            "Applies the measured probe time ratio to the recorded MLP organ "
            "ms and subtracts the saved ms from the recorded token ms."
        ),
    }


def gap_to_arm_a(production_gb_s: float, variant_gb_s: float) -> dict[str, Any]:
    gap = ARM_A_GB_S - production_gb_s
    closed = variant_gb_s - production_gb_s
    frac = None if gap == 0 else closed / gap
    return {
        "production_gb_s": round(production_gb_s, 1),
        "variant_gb_s": round(variant_gb_s, 1),
        "arm_a_target_gb_s": ARM_A_GB_S,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "u1alu_production_gb_s_cited": U1ALU_PRODUCTION_GB_S,
        "gap_gb_s": round(gap, 1),
        "closed_gb_s": round(closed, 1),
        "fraction_of_gap_closed": None if frac is None else round(frac, 4),
        "ratio_to_this_run_production": round(variant_gb_s / production_gb_s, 4)
        if production_gb_s > 0
        else None,
        "required_ratio": round(REQUIRED_CHEAPENING, 3),
        "note": (
            "Gap is ARM A 497.4 GB/s minus this-run production GB/s. "
            "Fraction closed uses this-run numbers; 329.6 is cited from "
            "u1alu and is not this-run production."
        ),
    }


def _require_variant_raw(raw_v: Mapping[str, Any], idx: int) -> None:
    for field in ("id", "kernel", "weight_bytes", "gpu_ns_median", "byte_compare"):
        if field not in raw_v:
            raise MissingVariant(f"variants[{idx}] is missing {field}")
    if int(raw_v["gpu_ns_median"]) <= 0:
        raise EmptyGpuSample(f"variants[{idx}] {raw_v.get('id')} gpu_ns_median must be positive")
    if int(raw_v["weight_bytes"]) <= 0:
        raise ValueError(f"variants[{idx}] weight_bytes must be positive")


def attach_variant(raw_v: Mapping[str, Any], production_gb_s: float, production_ns: int) -> dict[str, Any]:
    ident = str(raw_v["id"])
    if ident not in INNER_LOOP_TAX:
        raise MissingVariant(f"unknown variant id {ident}; tax table has {list(INNER_LOOP_TAX)}")
    tax = tax_view(INNER_LOOP_TAX[ident])
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
    declared = str(raw_v.get("class") or ("control" if ident == "production" else "unknown"))
    if ident == "production":
        klass = "control"
    elif identical:
        klass = "exact_candidate"
    else:
        klass = "approx_candidate"
    out: dict[str, Any] = {
        "id": ident,
        "kernel": raw_v.get("kernel"),
        "class": klass,
        "declared_class": declared,
        "mechanisms": list(raw_v.get("mechanisms") or []),
        "note": raw_v.get("note"),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw_v.get("gpu_ns_reps", [])],
        "gpu_us_median": round(gpu_ns / 1e3, 1),
        "effective_gb_s": round(gb_s, 1),
        "inner_loop": tax,
        "fma_per_weight_byte": tax["fma_per_weight_byte"],
        "decode_fma_per_weight_byte": tax["decode_fma_per_weight_byte"],
        "byte_compare": dict(cmp_),
        "bit_identical": identical,
        "dispatches": int(raw_v.get("dispatches", 3)),
        "encoders": int(raw_v.get("encoders", 1)),
        "command_buffers": int(raw_v.get("command_buffers", 1)),
        "threads_per_threadgroup": int(raw_v.get("threads_per_threadgroup", 128)),
        "occupancy": raw_v.get("occupancy"),
        "gap_to_arm_a": gap_to_arm_a(production_gb_s, gb_s),
        "projection": project_from_probe(production_ns, gpu_ns),
    }
    if not identical:
        out["capability_score"] = score_against_capability_map(cmp_)
        out["error_class"] = "approx_candidate"
    else:
        out["error_class"] = "bit_identical"
    return out


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise MissingVariant("refusing a receipt: raw measurement has no variants")
    ids = [str(v.get("id")) for v in variants_raw if isinstance(v, Mapping)]
    if "production" not in ids:
        raise MissingVariant("refusing a receipt: production variant is missing")
    missing = [v for v in VARIANT_IDS if v not in ids]
    if missing:
        raise MissingVariant(f"refusing a receipt: missing variants {missing}")

    prod_raw = next(v for v in variants_raw if v.get("id") == "production")
    _require_variant_raw(prod_raw, ids.index("production"))
    prod_bytes = int(prod_raw["weight_bytes"])
    prod_ns = int(prod_raw["gpu_ns_median"])
    prod_gb = effective_gb_s(prod_bytes, prod_ns)

    attached: list[dict[str, Any]] = []
    for i, v in enumerate(variants_raw):
        if not isinstance(v, Mapping):
            raise MissingVariant(f"variants[{i}] is not an object")
        _require_variant_raw(v, i)
        if int(v["weight_bytes"]) != prod_bytes:
            raise ValueError(
                f"variant {v.get('id')} weight_bytes {v['weight_bytes']} != "
                f"production {prod_bytes}"
            )
        attached.append(attach_variant(v, prod_gb, prod_ns))

    exact = [v for v in attached if v["id"] != "production" and v["bit_identical"]]
    approx = [v for v in attached if v["id"] != "production" and not v["bit_identical"]]
    best_exact = max(exact, key=lambda v: v["effective_gb_s"]) if exact else None
    best_approx = max(approx, key=lambda v: v["effective_gb_s"]) if approx else None
    return {
        "layer": int(raw.get("layer", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "concurrent_load": raw.get("concurrent_load") or {},
        "absolute_gb_s_are_measured_under_load": True,
        "fast_math": raw.get("fast_math", False),
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
        "variants": attached,
        "best_exact": best_exact,
        "best_approx": best_approx,
    }


def _finding(measurement: Mapping[str, Any]) -> str:
    prod = measurement["production_gb_s"]
    lines = [
        f"This-run production {prod} GB/s on the matched geo_tpr64 layer "
        f"({measurement['production_gpu_ns_median']} ns). "
        f"u1alu cited 329.6 GB/s / ARM A 497.4 GB/s."
    ]
    exact = measurement.get("best_exact")
    approx = measurement.get("best_approx")
    if exact:
        g = exact["gap_to_arm_a"]
        p = exact["projection"]
        lines.append(
            f"Best bit-identical variant {exact['id']}: {exact['effective_gb_s']} GB/s "
            f"({exact['gpu_us_median']} us), decode FMA/byte {exact['decode_fma_per_weight_byte']} "
            f"(production {round(PRODUCTION_DECODE_FMA_PER_BYTE, 4)}), "
            f"closes {g['fraction_of_gap_closed']} of the gap to 497.4 GB/s. "
            f"PROJECTION: MLP {p['mlp_ms_projected']} ms, token {p['token_ms_projected']} ms "
            f"({p['tps_projected']} TPS, {p['delta_tps']:+} vs {TOKEN_TPS})."
        )
        if (
            exact["decode_fma_per_weight_byte"] <= round(TARGET_DECODE_FMA_PER_BYTE, 4)
            and exact["effective_gb_s"] < ARM_A_GB_S
        ):
            lines.append(
                f"Decode FMA/byte is at or below the 0.88 target "
                f"({exact['decode_fma_per_weight_byte']} <= {round(TARGET_DECODE_FMA_PER_BYTE, 4)}) "
                f"but GB/s is not at ARM A's 497.4: ARM A also stripped the 8 MACs, "
                f"and this sidecar was forbidden to."
            )
    else:
        lines.append("No bit-identical variant beat the comparison (or none ran).")
    if approx:
        g = approx["gap_to_arm_a"]
        p = approx["projection"]
        score = approx.get("capability_score") or {}
        lines.append(
            f"Best approx-class variant {approx['id']}: {approx['effective_gb_s']} GB/s "
            f"({approx['gpu_us_median']} us), decode FMA/byte {approx['decode_fma_per_weight_byte']}, "
            f"NOT bit-identical (max_abs {approx['byte_compare'].get('max_abs_err')}, "
            f"cosine {approx['byte_compare'].get('cosine')}). "
            f"Capability-map proxy above_cosine_bar={score.get('above_cosine_bar')}. "
            f"PROJECTION: MLP {p['mlp_ms_projected']} ms, token {p['token_ms_projected']} ms. "
            f"Do not blend this with the exact-class verdict."
        )
    lines.append(
        "Affine fold is exact over reals (sum((s*q+b)*x)=s*sum(q*x)+b*sum(x)). "
        "A synthetic f32 tile is not bit-identical to production (q*s+b)*x; "
        "GPU byte comparison on this layer is the authority for bit-identity."
    )
    return " ".join(lines)


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    if "variants" not in measurement:
        raise MissingVariant("refusing a receipt without variants")
    for v in measurement["variants"]:
        bit_identical_from_compare(v)
        if "effective_gb_s" not in v or float(v["effective_gb_s"]) <= 0:
            raise EmptyGpuSample(f"variant {v.get('id')} is missing measured GB/s")
        if "fma_per_weight_byte" not in v:
            raise MissingVariant(f"variant {v.get('id')} is missing fma_per_weight_byte")
    algebra = {
        "over_reals": affine_fold_identity_over_reals(),
        "f32_counterexample": f32_counterexample_for_fold(),
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/decode_cheapen_mlp.rs; "
            "crates/hawking-core/examples/decode_cheapen_mlp.metal; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime); "
            "one representative layer of sealed-3.14, production MLP affine2 geo_tpr64"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "arm_a_gb_s": ARM_A_GB_S,
        "arm_a_us": ARM_A_US,
        "u1alu_production_gb_s": U1ALU_PRODUCTION_GB_S,
        "u1alu_production_us": U1ALU_PRODUCTION_US,
        "required_cheapening": round(REQUIRED_CHEAPENING, 3),
        "production_decode_fma_per_weight_byte": round(PRODUCTION_DECODE_FMA_PER_BYTE, 4),
        "target_decode_fma_per_weight_byte": round(TARGET_DECODE_FMA_PER_BYTE, 4),
        "mlp_ms_baseline": MLP_MS,
        "token_ms_baseline": TOKEN_MS,
        "token_tps_baseline": TOKEN_TPS,
        "capability_bars": {
            "cosine_bar": COSINE_BAR,
            "hidden_bar": HIDDEN_BAR,
            "gate_bar": GATE_BAR,
            "source": "receipts/future/CAPABILITY_INFORMATION_MAP.json",
        },
        "algebra": algebra,
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
        "variants": measurement["variants"],
        "best_exact": measurement.get("best_exact"),
        "best_approx": measurement.get("best_approx"),
        "finding": _finding(measurement),
        "timing": measurement.get("timing"),
        "concurrent_load": measurement.get("concurrent_load"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "does_not_edit_production_shaders": True,
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise MissingVariant("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    # A hardware number must be placeable in time. This module used to write its
    # own json.dumps with no timestamp, so when /tmp/hawking-gpu-lane.lock was
    # found wedged, placing this receipt against that window needed git landing
    # time - a proxy for when the measurement actually ran.
    doc.setdefault(
        "measurement_provenance",
        measurement_provenance(
            lock_held=bool(os.environ.get("HAWKING_GPU_LANE_LOCK_HELD")),
            lane="mlp_decode_cheapen",
            # A receipt rebuilt from a stored raw capture must not stamp the
            # rebuild time as the measurement time. The raw files carry no
            # timestamp, so a retrofit records the measurement time as UNKNOWN.
            retrofit=not os.environ.get("HAWKING_MEASURED_NOW"),
        ),
    )
    write_measured_receipt(out, doc, "tools/future/mlp_decode_cheapen.py")
    return out


def example_binaries() -> list[Path]:
    names = ("decode_cheapen_mlp",)
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
            "decode_cheapen_mlp binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example decode_cheapen_mlp`"
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
    if args.record:
        path = record(measured, path=args.out)
        print(f"wrote {path}")
        print(measured and "")
        print(_finding(measured))
    else:
        print(_finding(measured))
        for v in measured["variants"]:
            flag = "IDENTICAL" if v["bit_identical"] else "approx"
            print(
                f"  {v['id']:14s} {v['effective_gb_s']:7.1f} GB/s  "
                f"{v['gpu_us_median']:7.1f} us  "
                f"decode-FMA/B {v['decode_fma_per_weight_byte']:.4f}  {flag}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
