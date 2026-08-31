#!/usr/bin/env python3
"""LUT-decode consumer of the screened u8 aux. Beside aux_u8_native, not instead.

A u8 has 256 values. The log-scale exp and the linear bias are functions of
one byte: two 256-entry float tables, 2 KB, not a transcendental. This
module is that consumer.

    The runtime reads u8 scale/bias and indexes two 256-entry tables
    filled once with the same Metal exp / linear map the exp-variant
    computes in-register. 2-bit codes stay. The aux is never expanded
    back to f16.

    Table placement (constant / threadgroup / device) is measured; a
    table in the wrong address space can cost more than the exp it
    replaced.

    python3 tools/future/aux_u8_lut.py --measure --record
    python3 -m pytest tools/future/test_aux_u8_lut.py -q

Does not edit tools/future/aux_u8_native.py or aux_u8_ab.*. Does not
change production shaders. GPU A/B is SELF_MEASURED_DIRTY; absolute
GB/s is measured-under-load with loadavg recorded.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import os
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.future import aux_capability_screen as acs
from tools.future import aux_u8_native as n8
from tools.future import executable_economics as ee
from tools.future._common import (
    REPO,
    RECEIPTS,
    measurement_provenance,
    write_measured_receipt,
)
from tools.future.mlp_auxiliary_information import AUXILIARY_BYTES_TARGET


RECEIPT = RECEIPTS / "AUX_U8_LUT.json"
RAW_DEFAULT = RECEIPTS / "_AUX_U8_LUT_raw.json"
SCHEMA = "hawking.future.aux_u8_lut.v1"
VERSION = 1
RECORDED_BY = "tools/future/aux_u8_lut.py"
EXAMPLE_NAME = "aux_u8_lut_ab"
SHADER_REL = "crates/hawking-core/examples/aux_u8_lut_ab.metal"
EXAMPLE_REL = "crates/hawking-core/examples/aux_u8_lut_ab.rs"
NATIVE_RECEIPT_REL = "receipts/future/AUX_U8_NATIVE.json"
ERROR_BUDGET_REL = "receipts/future/MLP_ERROR_BUDGET.json"

DIRECT_CONSUME = n8.DIRECT_CONSUME
MATERIALIZE_F16_AUX = n8.MATERIALIZE_F16_AUX
LEVER_ID = n8.LEVER_ID
N_LAYERS = n8.N_LAYERS
N_TENSORS = n8.N_TENSORS
GROUP = n8.GROUP
LUT_N = 256
LUT_BYTES_PER_TENSOR = LUT_N * 4 * 2  # scale + bias, f32
ARTIFACT_DEFAULT = n8.ARTIFACT_DEFAULT

# Native A/B down_proj relfro vs incumbent (AUX_U8_NATIVE). LUT is a
# reindexing of the same values and must not move this.
NATIVE_DOWN_PROJ_RELFRO = 0.02629999837482214
ALL_LAYERS_STRUCTURED_BAR = 0.03

PLACEMENTS = ("constant", "threadgroup", "device")

LUT_CONSUMER_KERNELS = (
    "aux_u8_lut_constant_affine_q2_geo_tpr64_tg128",
    "aux_u8_lut_threadgroup_affine_q2_geo_tpr64_tg128",
    "aux_u8_lut_device_affine_q2_geo_tpr64_tg128",
)

CLAIM_BOUNDARY = (
    "LUT consumer of the screened u8 aux: two 256-entry tables (log-scale, "
    "linear bias) resolved once and indexed by the u8 value. 2-bit codes "
    "kept. GPU A/B is one representative MLP layer of sealed-3.14, three-way "
    "in one process (incumbent f16-aux, exp-variant u8, LUT-variant u8), "
    "MTLCommandBuffer GPUStartTime/GPUEndTime, 11 reps 5 warmup. Table "
    "placement (constant / threadgroup / device) is measured; the LUT arm "
    "is the fastest placement. Absolute GB/s is measured-under-load "
    "(loadavg recorded); it is not a clean roof. Token-level ms from the "
    "probe is ARITHMETIC over the measured layer (x64), labelled a "
    "projection. A path that expands u8 aux to an f16 aux buffer is "
    "refused. A speedup is refused unless a byte comparison proves LUT "
    "output is unchanged from the exp-variant. Production shaders are not "
    "edited. FMA/byte by class (FMA, integer, conversion, memory) is the "
    "inner-loop tax for all three arms; extra decode arithmetic is billed "
    "through executable_economics.py both ways (bytes only, and with decode "
    "FLOPs)."
)


class LutRefuse(n8.NativeRefuse):
    """The LUT consumer refused rather than guessing."""


class LutSpeedupWithoutByteMatchRefuse(LutRefuse):
    """Speedup is not reportable without LUT-vs-exp byte identity."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: a LUT speedup is not reportable without a byte "
            "comparison proving the output is unchanged from the "
            f"exp-variant{extra}"
        )


class LutOutputMovedRefuse(LutRefuse):
    """LUT is a reindexing; a moved output means the table is wrong."""


ArgmaxAloneParityRefuse = n8.ArgmaxAloneParityRefuse
MaterializeF16AuxRefuse = n8.MaterializeF16AuxRefuse
IncompleteAB = n8.IncompleteAB
EmptyGpuSample = n8.EmptyGpuSample


# ---------------------------------------------------------------------------
# Inner-loop tax by class. Same 8-weight geo_tpr64 iteration as native.
# ---------------------------------------------------------------------------

INCUMBENT_INNER = {
    "label": "incumbent_f16_aux",
    "weights_per_iteration": 8,
    "weight_bytes_per_iteration": 6,  # 2 code + 2 scale + 2 bias
    "x_bytes_per_iteration": 32,
    "dequant_fma": 8,
    "mac_fma": 8,
    "aux_decode_fma": 0,
    "exp": 0,
    "bitops": 16,
    "int_to_float": 8,  # 8 q
    "half_to_float": 2,  # f16 scale, f16 bias
    "lut_loads": 0,
    "lut_load_bytes": 0,
}

EXP_INNER = {
    "label": "exp_variant_u8_aux",
    "weights_per_iteration": 8,
    "weight_bytes_per_iteration": 4,  # 2 code + 1 scale_u8 + 1 bias_u8
    "x_bytes_per_iteration": 32,
    "dequant_fma": 8,
    "mac_fma": 8,
    "aux_decode_fma": 2,  # lmin + u8*span  and  bmin + u8*span
    "exp": 1,
    "bitops": 16,
    "int_to_float": 10,  # 8 q + 2 u8
    "half_to_float": 0,
    "lut_loads": 0,
    "lut_load_bytes": 0,
}

LUT_INNER = {
    "label": "lut_variant_u8_aux",
    "weights_per_iteration": 8,
    "weight_bytes_per_iteration": 4,  # same streamed aux as exp-variant
    "x_bytes_per_iteration": 32,
    "dequant_fma": 8,
    "mac_fma": 8,
    "aux_decode_fma": 0,  # replaced by table lookup
    "exp": 0,
    "bitops": 16,
    "int_to_float": 8,  # 8 q only; u8 is an index, not a conversion
    "half_to_float": 0,
    "lut_loads": 2,  # scale_lut[s], bias_lut[b]
    "lut_load_bytes": 8,  # 2 f32
}


def _class_row(inner: Mapping[str, Any], *, count_exp_as: float = 0.0) -> dict[str, Any]:
    wb = int(inner["weight_bytes_per_iteration"])
    dequant = int(inner["dequant_fma"])
    mac = int(inner["mac_fma"])
    aux = int(inner["aux_decode_fma"])
    exp_n = float(inner["exp"]) * float(count_exp_as)
    fma = dequant + mac + aux + exp_n
    integer = int(inner["bitops"])
    conversion = int(inner["int_to_float"]) + int(inner["half_to_float"])
    decode_fma = dequant + aux + exp_n
    lut_loads = int(inner["lut_loads"])
    return {
        **dict(inner),
        "counted_exp_as_fma": float(count_exp_as),
        "fma_count": fma,
        "integer_count": integer,
        "conversion_count": conversion,
        "memory_streamed_weight_bytes": wb,
        "class": {
            "fma": round(fma / wb, 4),
            "integer": round(integer / wb, 4),
            "conversion": round(conversion / wb, 4),
            "memory": {
                "streamed_weight_bytes_per_weight_byte": 1.0,
                "lut_loads_per_weight_byte": round(lut_loads / wb, 4),
                "lut_load_bytes_per_weight_byte": round(int(inner["lut_load_bytes"]) / wb, 4),
            },
        },
        "fma_per_weight_byte": round(fma / wb, 4),
        "decode_fma_per_weight_byte": round(decode_fma / wb, 4),
        "dequant_fma_per_weight_byte": round(dequant / wb, 4),
        "aux_decode_fma_per_weight_byte": round((aux + exp_n) / wb, 4),
        "mac_fma_per_weight_byte": round(mac / wb, 4),
    }


def fma_per_byte_by_class(*, count_exp_as: float = 0.0) -> dict[str, Any]:
    """FMA / integer / conversion / memory per weight-byte, three arms."""
    inc = _class_row(INCUMBENT_INNER, count_exp_as=0.0)
    exp = _class_row(EXP_INNER, count_exp_as=count_exp_as)
    lut = _class_row(LUT_INNER, count_exp_as=0.0)
    return {
        "convention": (
            "One geo_tpr64 inner iteration = 8 weights. Classes are ops per "
            "streamed weight-byte. Incumbent: 2 B codes + 2 B f16 scale + "
            "2 B f16 bias; 8 dequant-FMA + 8 MAC; 8 q int_to_float + 2 "
            "half_to_float; 16 bitops. Exp-variant: 2 B codes + 1 B u8 "
            "scale + 1 B u8 bias; same dequant+MAC plus 2 aux-decode FMA "
            "and 1 exp, 10 int_to_float. LUT-variant: same streamed 4 B, "
            "same dequant+MAC, 0 aux-decode FMA, 0 exp, 8 int_to_float "
            "(u8 is an index), 2 indexed loads from a resident 256-entry "
            "table (2 KB, not streamed). exp is counted as FMA only when "
            "count_exp_as > 0."
        ),
        "incumbent": inc,
        "exp_variant": exp,
        "lut_variant": lut,
        "claim": (
            "LUT removes the aux-decode arithmetic tax: exp-variant decode "
            f"{exp['decode_fma_per_weight_byte']} FMA/weight-byte at 4 B/iter "
            f"vs LUT {lut['decode_fma_per_weight_byte']} (dequant only) vs "
            f"incumbent {inc['decode_fma_per_weight_byte']} at 6 B/iter. "
            "Remaining LUT-vs-incumbent intensity is the same 8+8 FMA over "
            "fewer bytes (4.0 vs 2.6667) plus two indexed loads."
        ),
        "delta_lut_minus_exp": {
            "aux_decode_fma": -2,
            "exp": -1,
            "int_to_float": -2,
            "lut_loads": 2,
            "decode_fma_per_weight_byte": round(
                lut["decode_fma_per_weight_byte"] - exp["decode_fma_per_weight_byte"], 4
            ),
            "fma_per_weight_byte": round(
                lut["fma_per_weight_byte"] - exp["fma_per_weight_byte"], 4
            ),
        },
        "delta_lut_minus_incumbent": {
            "weight_bytes_per_iteration": -2,
            "aux_decode_fma": 0,
            "exp": 0,
            "lut_loads": 2,
            "decode_fma_per_weight_byte": round(
                lut["decode_fma_per_weight_byte"] - inc["decode_fma_per_weight_byte"], 4
            ),
            "fma_per_weight_byte": round(
                lut["fma_per_weight_byte"] - inc["fma_per_weight_byte"], 4
            ),
        },
    }


def extra_flops_per_output_element_exp(*, count_exp_as: float = 0.0) -> float:
    return n8.extra_flops_per_output_element(count_exp_as=count_exp_as)


def extra_flops_per_output_element_lut() -> float:
    """LUT replaces the 2 aux-decode FMA + exp with indexed loads. Extra FMA = 0."""
    return 0.0


# ---------------------------------------------------------------------------
# Pack / consume. LUT never holds a decoded aux of n_groups.
# ---------------------------------------------------------------------------

AuxAllocLedger = n8.AuxAllocLedger
PackedU8Aux = n8.PackedU8Aux
pack_u8_aux = n8.pack_u8_aux
classify_consumer = n8.classify_consumer
cosine = n8.cosine
relfro = n8.relfro
report_equivalence = n8.report_equivalence


def build_luts(packed: PackedU8Aux) -> tuple[np.ndarray, np.ndarray]:
    """Two 256-entry tables. Exact reindexing of the exp-variant decode."""
    ep = packed.endpoints()
    idx = np.arange(LUT_N, dtype=np.float64)
    if ep["scale_span"] == 0.0:
        scale_lut = np.full(LUT_N, math.exp(ep["scale_lmin"]), dtype=np.float32)
    else:
        scale_lut = np.exp(ep["scale_lmin"] + idx * ep["scale_span"]).astype(np.float32)
    if ep["bias_span"] == 0.0:
        bias_lut = np.full(LUT_N, ep["bias_min"], dtype=np.float32)
    else:
        bias_lut = (ep["bias_min"] + idx * ep["bias_span"]).astype(np.float32)
    return scale_lut, bias_lut


def decode_u8_scale_column_lut(
    packed: PackedU8Aux, gi: int, scale_lut: np.ndarray
) -> np.ndarray:
    return scale_lut[packed.scale_u8[:, int(gi)]]


def decode_u8_bias_column_lut(
    packed: PackedU8Aux, gi: int, bias_lut: np.ndarray
) -> np.ndarray:
    return bias_lut[packed.bias_u8[:, int(gi)]]


def lut_matvec_u8(
    x: np.ndarray,
    packed: PackedU8Aux,
    *,
    ledger: AuxAllocLedger | None = None,
    scale_lut: np.ndarray | None = None,
    bias_lut: np.ndarray | None = None,
) -> np.ndarray:
    """y = x @ W.T with W decoded group-by-group via 256-entry tables.

    Never allocates a decoded scale/bias array of n_groups. Peak decoded
    aux is 2*rows (one group-column). The tables themselves are 512 floats,
    not an f16 aux of n_groups.
    """
    if scale_lut is None or bias_lut is None:
        scale_lut, bias_lut = build_luts(packed)
    xc = np.ascontiguousarray(x, dtype=np.float32)
    if xc.ndim == 1:
        xc = xc[None, :]
    if int(xc.shape[1]) != int(packed.cols):
        raise LutRefuse(f"x cols {xc.shape[1]} != packed.cols {packed.cols}")
    n = int(xc.shape[0])
    rows = int(packed.rows)
    g = int(packed.group_size)
    gpr = packed.gpr
    y = np.zeros((n, rows), dtype=np.float32)
    q = packed.q.astype(np.float32, copy=False)
    for gi in range(gpr):
        scale = decode_u8_scale_column_lut(packed, gi, scale_lut)
        bias = decode_u8_bias_column_lut(packed, gi, bias_lut)
        if ledger is not None:
            ledger.note_decoded_aux(int(scale.size) + int(bias.size))
        w_slice = q[:, gi, :] * scale[:, None] + bias[:, None]
        x_slice = xc[:, gi * g : (gi + 1) * g]
        y += x_slice @ w_slice.T
        del scale, bias, w_slice
    if ledger is not None:
        ledger.wrote_f16_aux_buffer = False
        ledger.notes.append("lut_matvec_u8: group-column table index, no f16 aux buffer")
    return y[0] if np.asarray(x).ndim == 1 else y


def _as_bytes(x: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    return np.ascontiguousarray(x).tobytes()


def prove_output_unchanged_from_exp(
    lut_output: bytes | bytearray | memoryview | np.ndarray | None = None,
    exp_output: bytes | bytearray | memoryview | np.ndarray | None = None,
    *,
    bytes_equal: bool | None = None,
    n_mismatch: int | None = None,
) -> dict[str, Any]:
    """Byte comparison of LUT output vs the exp-variant. Not a speedup."""
    if bytes_equal is None:
        if lut_output is None or exp_output is None:
            raise LutSpeedupWithoutByteMatchRefuse(
                "lut_output and exp_output are required when bytes_equal is not supplied"
            )
        lb = _as_bytes(lut_output)
        eb = _as_bytes(exp_output)
        if len(lb) != len(eb):
            return {
                "unchanged": False,
                "bytes_equal": False,
                "len_lut": len(lb),
                "len_exp": len(eb),
                "n_mismatch": max(len(lb), len(eb)),
            }
        n_mismatch = 0 if lb == eb else sum(a != b for a, b in zip(lb, eb))
        bytes_equal = n_mismatch == 0
    return {
        "unchanged": bool(bytes_equal),
        "bytes_equal": bool(bytes_equal),
        "n_mismatch": 0 if bytes_equal else (0 if n_mismatch is None else int(n_mismatch)),
    }


def report_speedup(
    *,
    incumbent_gpu_ns: int,
    exp_gpu_ns: int,
    lut_gpu_ns: int,
    unchanged_proof: Mapping[str, Any] | None = None,
    lut_output: bytes | np.ndarray | None = None,
    exp_output: bytes | np.ndarray | None = None,
    bytes_equal: bool | None = None,
) -> dict[str, Any]:
    """May only emit a speedup after LUT output is byte-equal to the exp-variant.

    Times without that proof are still a measurement; they are not a speedup.
    """
    if unchanged_proof is None:
        unchanged_proof = prove_output_unchanged_from_exp(
            lut_output, exp_output, bytes_equal=bytes_equal
        )
    if not unchanged_proof.get("unchanged"):
        raise LutSpeedupWithoutByteMatchRefuse(
            f"unchanged_proof={dict(unchanged_proof)!r} "
            f"incumbent_ns={incumbent_gpu_ns} exp_ns={exp_gpu_ns} lut_ns={lut_gpu_ns}"
        )
    inc = int(incumbent_gpu_ns)
    exp = int(exp_gpu_ns)
    lut = int(lut_gpu_ns)
    if inc <= 0 or exp <= 0 or lut <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a speedup")
    return {
        "unchanged_from_exp": True,
        "lut_minus_incumbent_ns": lut - inc,
        "lut_minus_exp_ns": lut - exp,
        "exp_minus_incumbent_ns": exp - inc,
        "lut_faster_than_incumbent": lut < inc,
        "lut_faster_than_exp": lut < exp,
        "exp_faster_than_incumbent": exp < inc,
        "time_ratio_lut_over_incumbent": round(lut / inc, 6),
        "time_ratio_lut_over_exp": round(lut / exp, 6),
    }


# ---------------------------------------------------------------------------
# Shader invariants. LUT kernels must not exp() and must not materialize f16.
# ---------------------------------------------------------------------------


def lut_shader_source() -> str:
    path = REPO / SHADER_REL
    if not path.is_file():
        raise LutRefuse(f"REFUSED: LUT shader {SHADER_REL} is not on disk")
    return path.read_text(encoding="utf-8")


def _kernels(src: str) -> dict[str, str]:
    parts = src.split("kernel void ")
    out: dict[str, str] = {}
    for part in parts[1:]:
        name = part.split("(", 1)[0].strip()
        out[name] = "kernel void " + part
    return out


def _kernel_header(body: str) -> str:
    return body.split("{", 1)[0]


def lut_shader_invariants(src: str | None = None) -> dict[str, Any]:
    """Static proof the LUT Metal consumer does not materialize f16 aux."""
    text = lut_shader_source() if src is None else str(src)
    kernels = _kernels(text)
    required = (
        "device const uchar* scales_u8",
        "device const uchar* biases_u8",
        "aux_u8_incumbent_affine_q2_geo_tpr64_tg128",
        "aux_u8_native_affine_q2_geo_tpr64_tg128",
        "aux_u8_fill_lut256",
        *LUT_CONSUMER_KERNELS,
        "scale_lut[s]",
        "bias_lut[b]",
        "constant float* scale_lut",
        "threadgroup float tg_scale[256]",
        "device const float* scale_lut",
    )
    missing = [p for p in required if p not in text]
    forbidden_hits = []
    for needle in (
        "device half* decoded_scales",
        "device half* aux_f16",
        "device half* scales_f16_out",
        "scales_f16[rgb] =",
        "materialize_f16",
    ):
        if needle in text:
            forbidden_hits.append(needle)
    lut_exp_hits: list[str] = []
    lut_half_header: list[str] = []
    lut_binds_u8 = True
    for name in LUT_CONSUMER_KERNELS:
        body = kernels.get(name, "")
        if not body:
            missing.append(f"kernel {name}")
            continue
        header = _kernel_header(body)
        if "exp(" in body:
            lut_exp_hits.append(name)
        if "device const half*" in header or "device half*" in header:
            lut_half_header.append(name)
        if "uchar* scales_u8" not in header or "uchar* biases_u8" not in header:
            lut_binds_u8 = False
    fill_has_exp = "exp(" in kernels.get("aux_u8_fill_lut256", "")
    # The exp-variant decode lives in the static helper *before* the native
    # kernel (`affine_q2_geo_acc_g64_u8`), so a kernel-body split misses it.
    native_has_exp = "exp(ep.scale_lmin + float(s) * ep.scale_span)" in text
    ok = (
        not missing
        and not forbidden_hits
        and not lut_exp_hits
        and not lut_half_header
        and lut_binds_u8
        and fill_has_exp
        and native_has_exp
    )
    return {
        "ok": ok,
        "materializes_f16_aux": bool(forbidden_hits or lut_half_header),
        "missing_required": missing,
        "forbidden_hits": forbidden_hits,
        "lut_kernels_contain_exp": lut_exp_hits,
        "lut_kernels_bind_half_aux": lut_half_header,
        "binds_u8_aux": lut_binds_u8,
        "fill_kernel_uses_metal_exp": fill_has_exp,
        "exp_variant_uses_exp": native_has_exp,
        "in_register_exp": False,
        "indexed_lut": True,
        "placements": list(PLACEMENTS),
        "shader": SHADER_REL,
    }


# ---------------------------------------------------------------------------
# Economics. LUT extra decode FMA is 0; exp-variant still bills 2 FMA/8 w.
# ---------------------------------------------------------------------------


def byte_model() -> dict[str, int]:
    """Same u8 payload as the native lever; tables replace endpoints as metadata."""
    bm = dict(n8.byte_model())
    lut_bytes = int(N_TENSORS) * int(LUT_BYTES_PER_TENSOR)
    bm["lut_table_bytes"] = lut_bytes
    bm["bytes_added_metadata"] = lut_bytes
    bm["endpoint_bytes_baked_into_lut"] = int(bm.get("endpoint_bytes") or 0)
    return bm


def score_lut_aux(*, extra_flops_per_output_element: float) -> dict[str, Any]:
    bm = byte_model()
    scored = ee.score(
        bytes_removed=int(bm["bytes_removed"]),
        bytes_added={"metadata": int(bm["bytes_added_metadata"])},
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        consuming_primitive="FusedDecodeCompute",
        bandwidth_regime="affine_q2_family",
        organ="mlp",
        # The bytes this lever removes come from the BROADCAST AUX stream - a
        # per-group scale and bias that many threads read - not from the
        # per-thread-unique weight codes. That distinction is the whole reason
        # the lever billed +1.5530 ms and measured SLOWER: broadcast bytes are
        # cache-served and were never on the critical path, so removing them
        # removes no time. executable_economics now REFUSES an undeclared
        # stream_class rather than defaulting to the organ average, which is
        # exactly the defect this candidate exposed.
        stream_class="broadcast_aux",
        reusable_family=True,
        candidate_id=LEVER_ID,
        status="OPEN",
    )
    scored["byte_model"] = bm
    scored["extra_flops_note"] = (
        "LUT extra decode FMA is 0 (indexed load). Exp-variant billed 2 FMA "
        "per 8 weights."
    )
    return scored


def net_sign(predicted_ms_saved: float) -> str:
    if predicted_ms_saved > 0:
        return "POSITIVE"
    if predicted_ms_saved < 0:
        return "NEGATIVE"
    return "ZERO"


def economics_bundle() -> dict[str, Any]:
    """Bytes-only vs billed extra decode arithmetic, LUT extra FMA = 0."""
    bytes_only = score_lut_aux(extra_flops_per_output_element=0.0)
    with_fma = score_lut_aux(extra_flops_per_output_element=extra_flops_per_output_element_lut())
    exp_fma = extra_flops_per_output_element_exp(count_exp_as=0.0)
    exp_billed = n8.score_u8_aux(extra_flops_per_output_element=exp_fma, count_exp_as=0.0)
    return {
        "bytes_only": bytes_only,
        "with_aux_decode_fma": with_fma,
        "exp_variant_with_aux_decode_fma": exp_billed,
        "extra_flops_per_output_element_lut": extra_flops_per_output_element_lut(),
        "extra_flops_per_output_element_exp_fma_only": exp_fma,
        "net_sign_bytes_only": net_sign(float(bytes_only["predicted_ms_saved"])),
        "net_sign_with_aux_decode_fma": net_sign(float(with_fma["predicted_ms_saved"])),
        "note": (
            "The capability screen billed extra_flops_per_output_element=0 "
            "and got +1.554 ms. The exp-variant's 2 aux-decode FMA per 8 "
            "weights flipped the sign (flop_ms 1.9426, net -0.3885 ms). "
            "LUT extra FMA is 0, so both billings share the byte-lever sign. "
            "The GPU A/B is the physical number: billed-positive can still "
            "be wall-negative if the indexed load costs more than the exp."
        ),
    }


# ---------------------------------------------------------------------------
# GPU A/B. Real timestamps or a refuse — never a fabricated GB/s.
# ---------------------------------------------------------------------------


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    return n8.effective_gb_s(weight_bytes, gpu_ns)


def _arm_view(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise IncompleteAB(f"A/B missing {name}")
    for field in ("weight_bytes", "gpu_ns_median"):
        if field not in raw:
            raise IncompleteAB(f"A/B {name} missing {field}")
    weight_bytes = int(raw["weight_bytes"])
    gpu_ns = int(raw["gpu_ns_median"])
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    out = {
        "label": str(raw.get("label") or name),
        "kernel": raw.get("kernel"),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_us_median": round(gpu_ns / 1e3, 4),
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "dispatches": int(raw.get("dispatches", 3)),
        "encoders": int(raw.get("encoders", 1)),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "effective_gb_s": round(gb_s, 4),
        "aux_bytes": raw.get("aux_bytes"),
        "code_bytes": raw.get("code_bytes"),
        "aux_dtype": raw.get("aux_dtype"),
    }
    for k in (
        "endpoint_bytes",
        "materializes_f16_aux",
        "endpoint_note",
        "placement",
        "address_space",
        "lut_table_bytes",
        "output_bytes_equal_vs_exp",
        "vs_exp",
        "chosen_because",
    ):
        if k in raw:
            out[k] = raw[k]
    return out


def _placement_views(raw: Mapping[str, Any]) -> dict[str, Any]:
    placements = raw.get("placements") or {}
    if not isinstance(placements, Mapping):
        raise IncompleteAB("A/B missing placements mapping")
    out: dict[str, Any] = {}
    for name in PLACEMENTS:
        arm = placements.get(name)
        if not isinstance(arm, Mapping):
            raise IncompleteAB(f"A/B missing placement {name}")
        view = _arm_view(arm, f"placement.{name}")
        if view.get("materializes_f16_aux") is True:
            raise MaterializeF16AuxRefuse(f"LUT placement {name} materialized f16 aux")
        out[name] = view
    return out


def _output_proof_from_raw(raw: Mapping[str, Any], lut_arm: Mapping[str, Any]) -> dict[str, Any]:
    blob = raw.get("lut_vs_exp_output") or {}
    if isinstance(blob, Mapping) and "bytes_equal" in blob:
        chosen = blob.get("chosen") if isinstance(blob.get("chosen"), Mapping) else blob
        n_mismatch = None
        if isinstance(chosen, Mapping) and "n_mismatch" in chosen:
            n_mismatch = int(chosen["n_mismatch"])
        return prove_output_unchanged_from_exp(
            bytes_equal=bool(blob.get("bytes_equal")),
            n_mismatch=n_mismatch,
        )
    flag = lut_arm.get("output_bytes_equal_vs_exp")
    if flag is None:
        raise LutSpeedupWithoutByteMatchRefuse(
            "raw A/B has no lut_vs_exp_output.bytes_equal and no "
            "lut_variant.output_bytes_equal_vs_exp"
        )
    return prove_output_unchanged_from_exp(bytes_equal=bool(flag))


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse an A/B that has no loadavg, no three arms, or no GPU ns."""
    if not isinstance(raw, Mapping):
        raise IncompleteAB("raw A/B is not a mapping")
    load = raw.get("concurrent_load")
    if not isinstance(load, Mapping) or not load.get("loadavg"):
        raise IncompleteAB("A/B without loadavg is not a measurement")
    inc = _arm_view(raw.get("incumbent") or {}, "incumbent")
    exp_raw = raw.get("exp_variant") or raw.get("native_u8") or {}
    exp = _arm_view(exp_raw, "exp_variant")
    lut = _arm_view(raw.get("lut_variant") or {}, "lut_variant")
    if lut.get("materializes_f16_aux") is True or exp.get("materializes_f16_aux") is True:
        raise MaterializeF16AuxRefuse("u8 arm materialized f16 aux")
    placements = _placement_views(raw)
    proof = _output_proof_from_raw(raw, lut)
    inc_ns = int(inc["gpu_ns_median"])
    exp_ns = int(exp["gpu_ns_median"])
    lut_ns = int(lut["gpu_ns_median"])
    inc_b = int(inc["weight_bytes"])
    exp_b = int(exp["weight_bytes"])
    lut_b = int(lut["weight_bytes"])
    layer = int(raw.get("layer", -1))
    speedup: dict[str, Any] | None
    speedup_refused: dict[str, Any] | None
    try:
        speedup = report_speedup(
            incumbent_gpu_ns=inc_ns,
            exp_gpu_ns=exp_ns,
            lut_gpu_ns=lut_ns,
            unchanged_proof=proof,
        )
        speedup_refused = None
    except LutSpeedupWithoutByteMatchRefuse as exc:
        speedup = None
        speedup_refused = {"refused": True, "reason": str(exc), "claim": None}
    delta_lut_inc = lut_ns - inc_ns
    return {
        "layer": layer,
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "artifact_root": raw.get("artifact_root", ""),
        "git_head": raw.get("git_head", ""),
        "concurrent_load": load,
        "concurrent_load_end": raw.get("concurrent_load_end") or {},
        "absolute_gb_s_are_measured_under_load": True,
        "native_consumer": raw.get("native_consumer")
        or {
            "materializes_f16_aux": False,
            "scale_buffer": "u8",
            "bias_buffer": "u8",
        },
        "incumbent": inc,
        "exp_variant": exp,
        "lut_variant": lut,
        "placements": placements,
        "chosen_placement": raw.get("chosen_placement") or lut.get("placement"),
        "lut_vs_exp_output": raw.get("lut_vs_exp_output") or {},
        "output_unchanged_from_exp": proof,
        "speedup": speedup,
        "speedup_refused": speedup_refused,
        "delta": {
            "lut_minus_incumbent": {
                "weight_bytes": lut_b - inc_b,
                "gpu_ns": lut_ns - inc_ns,
                "gpu_us": round((lut_ns - inc_ns) / 1e3, 4),
                "effective_gb_s": round(float(lut["effective_gb_s"]) - float(inc["effective_gb_s"]), 4),
                "time_ratio": round(lut_ns / inc_ns, 6) if inc_ns else None,
            },
            "lut_minus_exp": {
                "weight_bytes": lut_b - exp_b,
                "gpu_ns": lut_ns - exp_ns,
                "gpu_us": round((lut_ns - exp_ns) / 1e3, 4),
                "effective_gb_s": round(float(lut["effective_gb_s"]) - float(exp["effective_gb_s"]), 4),
                "time_ratio": round(lut_ns / exp_ns, 6) if exp_ns else None,
            },
            "exp_minus_incumbent": {
                "weight_bytes": exp_b - inc_b,
                "gpu_ns": exp_ns - inc_ns,
                "gpu_us": round((exp_ns - inc_ns) / 1e3, 4),
                "effective_gb_s": round(float(exp["effective_gb_s"]) - float(inc["effective_gb_s"]), 4),
                "time_ratio": round(exp_ns / inc_ns, 6) if inc_ns else None,
            },
        },
        "paired_reps": raw.get("paired_reps") or [],
        "projections": raw.get("projections") or [],
        "output_cosine_mean": raw.get("output_cosine_mean"),
        "bytes_removed_this_layer": raw.get("bytes_removed_this_layer", inc_b - lut_b),
        "lut_table_bytes_this_layer": raw.get("lut_table_bytes_this_layer"),
        "token_projection": {
            "kind": "PROJECTION_ARITHMETIC_OVER_PROBE",
            "not_a_resident_measurement": True,
            "layers": N_LAYERS,
            "probe_layer": layer,
            "probe_delta_gpu_us": round(delta_lut_inc / 1e3, 4),
            "projected_mlp_delta_us": round(delta_lut_inc * N_LAYERS / 1e3, 4),
            "projected_mlp_delta_ms": round(delta_lut_inc * N_LAYERS / 1e6, 6),
            "formula": "measured_layer_delta_ns * 64",
            "note": (
                "64 identical MLP layers assumed. Not a protected generate "
                "gate and not a host+GPU complete-token measurement. Delta is "
                "LUT minus incumbent."
            ),
        },
        "occupancy": raw.get("occupancy") or {},
        "shader": raw.get("shader") or SHADER_REL,
    }


def error_did_not_move(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """LUT vs exp must not move; down_proj vs incumbent must stay near 0.0263."""
    proof = measurement.get("output_unchanged_from_exp") or {}
    projections = measurement.get("projections") or []
    down = None
    for p in projections:
        name = str(p.get("name") or "")
        if "down_proj" in name:
            down = p
            break
    down_relfro_lut = None
    down_relfro_exp = None
    if isinstance(down, Mapping):
        down_relfro_exp = down.get("output_relfro_exp_vs_incumbent")
        for key in (
            "output_relfro_lut_constant_vs_incumbent",
            "output_relfro_lut_threadgroup_vs_incumbent",
            "output_relfro_lut_device_vs_incumbent",
        ):
            if down.get(key) is not None:
                down_relfro_lut = down.get(key)
                # prefer the chosen placement if tagged
        chosen = measurement.get("chosen_placement")
        if chosen:
            key = f"output_relfro_lut_{chosen}_vs_incumbent"
            if down.get(key) is not None:
                down_relfro_lut = down.get(key)
    moved_vs_exp = not bool(proof.get("unchanged"))
    moved_vs_native_down = False
    down_note = "down_proj relfro not in raw projections"
    if down_relfro_lut is not None:
        # A LUT reindexing of the same u8 encode must match the exp-variant
        # down_proj error, which AUX_U8_NATIVE measured at 0.0263 against
        # the 0.03 all-layers structured bar.
        drift = abs(float(down_relfro_lut) - NATIVE_DOWN_PROJ_RELFRO)
        moved_vs_native_down = drift > 1e-4
        down_note = (
            f"down_proj output_relfro lut={down_relfro_lut} exp={down_relfro_exp} "
            f"native_receipt={NATIVE_DOWN_PROJ_RELFRO} all_layers_bar={ALL_LAYERS_STRUCTURED_BAR}"
        )
    ok = (not moved_vs_exp) and (not moved_vs_native_down)
    return {
        "ok": ok,
        "unchanged_from_exp": bool(proof.get("unchanged")),
        "bytes_equal_vs_exp": bool(proof.get("bytes_equal")),
        "down_proj_relfro_lut": down_relfro_lut,
        "down_proj_relfro_exp": down_relfro_exp,
        "native_receipt_down_proj_relfro": NATIVE_DOWN_PROJ_RELFRO,
        "all_layers_structured_bar": ALL_LAYERS_STRUCTURED_BAR,
        "moved_vs_exp": moved_vs_exp,
        "moved_vs_native_down_proj": moved_vs_native_down,
        "note": down_note,
    }


def example_binaries() -> list[Path]:
    roots: list[Path] = []
    env = _os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "workspace" / "ops" / "build" / "rust",
            REPO / "target",
        ]
    )
    out: list[Path] = []
    for root in roots:
        for profile in ("release-fast", "release"):
            p = root / profile / "examples" / EXAMPLE_NAME
            if p.is_file():
                out.append(p)
    return out


def run_example(
    artifact_root: Path,
    *,
    layer: int = 3,
    warmup: int = 5,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            f"{EXAMPLE_NAME} binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            f"--profile release-fast -p hawking-core --example {EXAMPLE_NAME}`"
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


def current_loadavg() -> dict[str, Any]:
    loadavg = subprocess.run(
        ["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True, check=False
    ).stdout.strip()
    uptime = subprocess.run(
        ["uptime"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "loadavg": loadavg,
        "uptime": uptime,
        "note": "host loadavg; not a GPU bandwidth",
    }


def gpu_blocked_ab(reason: str, *, binary: str | None = None) -> dict[str, Any]:
    return {
        "status": "NOT_MEASURED",
        "reason": str(reason),
        "concurrent_load": current_loadavg(),
        "absolute_gb_s_are_measured_under_load": False,
        "fabricated_hardware_numbers": False,
        "probe_binary": binary,
        "incumbent": None,
        "exp_variant": None,
        "lut_variant": None,
        "placements": None,
        "delta": None,
        "token_projection": None,
        "speedup": None,
        "needs": "unsandboxed Metal (gate profile): MTLCreateSystemDefaultDevice was nil",
    }


def _py(x: Any) -> Any:
    return acs._py(x)


def _honest_outcome(measurement: Mapping[str, Any], econ: Mapping[str, Any]) -> dict[str, Any]:
    inc_us = float(measurement["incumbent"]["gpu_us_median"])
    exp_us = float(measurement["exp_variant"]["gpu_us_median"])
    lut_us = float(measurement["lut_variant"]["gpu_us_median"])
    billed = econ["with_aux_decode_fma"]
    net = net_sign(float(billed["predicted_ms_saved"]))
    faster_than_inc = lut_us < inc_us
    faster_than_exp = lut_us < exp_us
    if faster_than_inc and net == "POSITIVE":
        kind = "LUT_BEATS_INCUMBENT_AND_BILLED_POSITIVE"
        prose = (
            "LUT variant is faster than incumbent AND net positive with "
            "FLOPs billed — the 1.554 ms byte lever is alive again."
        )
    elif faster_than_exp and not faster_than_inc:
        kind = "LUT_BEATS_EXP_STILL_SLOWER_THAN_INCUMBENT"
        prose = (
            "LUT variant is faster than the exp variant but still slower "
            "than incumbent — the arithmetic tax is real but not only the exp."
        )
    else:
        kind = "LUT_NO_BETTER"
        prose = (
            "LUT variant is no better — the tax is not the exp, and "
            "quantize_aux_u8 is MEASURED_NEGATIVE as a class rather than "
            "as an implementation."
        )
    return {
        "kind": kind,
        "prose": prose,
        "lut_faster_than_incumbent": faster_than_inc,
        "lut_faster_than_exp": faster_than_exp,
        "billed_net_sign": net,
        "incumbent_gpu_us": inc_us,
        "exp_gpu_us": exp_us,
        "lut_gpu_us": lut_us,
    }


def build(
    *,
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shader = lut_shader_invariants()
    if shader["materializes_f16_aux"] or not shader["ok"]:
        raise MaterializeF16AuxRefuse(f"shader invariants {shader}")
    classes = fma_per_byte_by_class(count_exp_as=0.0)
    classes_exp1 = fma_per_byte_by_class(count_exp_as=1.0)
    econ = economics_bundle()
    gpu = None if measurement is None else dict(measurement)
    finding_parts = [
        "LUT u8-aux consumer binds uchar scale/bias and indexes two "
        "256-entry tables (log-scale, linear bias) instead of calling exp. "
        "It does not expand the compact aux back to f16.",
        classes["claim"],
    ]
    bytes_only = econ["bytes_only"]
    with_fma = econ["with_aux_decode_fma"]
    finding_parts.append(
        f"executable_economics bytes-only: {bytes_only['predicted_ms_saved']:.4f} ms saved "
        f"(sign {econ['net_sign_bytes_only']}). With aux-decode FMA billed "
        f"(LUT extra FMA=0): {with_fma['predicted_ms_saved']:.4f} ms "
        f"(sign {econ['net_sign_with_aux_decode_fma']}, "
        f"flop_ms {with_fma['terms']['flop_ms_delta']:.4f}, "
        f"byte_ms {with_fma['terms']['byte_ms_delta']:.4f}). "
        f"Exp-variant billed {econ['exp_variant_with_aux_decode_fma']['predicted_ms_saved']:.4f} ms "
        f"(flop_ms {econ['exp_variant_with_aux_decode_fma']['terms']['flop_ms_delta']:.4f})."
    )
    error_block: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    if gpu is not None:
        d = gpu["delta"]
        finding_parts.append(
            f"GPU A/B layer {gpu['layer']}: incumbent {gpu['incumbent']['effective_gb_s']} GB/s "
            f"in {gpu['incumbent']['gpu_us_median']} us, exp-variant "
            f"{gpu['exp_variant']['effective_gb_s']} GB/s in "
            f"{gpu['exp_variant']['gpu_us_median']} us, LUT-{gpu.get('chosen_placement')} "
            f"{gpu['lut_variant']['effective_gb_s']} GB/s in "
            f"{gpu['lut_variant']['gpu_us_median']} us "
            f"(Δinc {d['lut_minus_incumbent']['gpu_us']} us, "
            f"Δexp {d['lut_minus_exp']['gpu_us']} us). "
            f"Token projection {gpu['token_projection']['projected_mlp_delta_ms']} ms "
            f"({gpu['token_projection']['kind']})."
        )
        place_bits = []
        for name, arm in (gpu.get("placements") or {}).items():
            place_bits.append(
                f"{name} {arm['effective_gb_s']} GB/s {arm['gpu_us_median']} us"
            )
        if place_bits:
            finding_parts.append("Placement rates: " + "; ".join(place_bits) + ".")
        error_block = error_did_not_move(gpu)
        if error_block["moved_vs_exp"]:
            finding_parts.append(
                "LUT output MOVED vs the exp-variant; the table is not an "
                "exact reindexing and a speedup is refused."
            )
        else:
            finding_parts.append(
                "LUT output is byte-unchanged from the exp-variant "
                f"(down_proj relfro vs incumbent {error_block.get('down_proj_relfro_lut')}, "
                f"native receipt {NATIVE_DOWN_PROJ_RELFRO}, bar {ALL_LAYERS_STRUCTURED_BAR})."
            )
        outcome = _honest_outcome(gpu, econ)
        finding_parts.append(outcome["prose"])
        if gpu.get("speedup_refused"):
            finding_parts.append(
                "Speedup claim REFUSED: " + str(gpu["speedup_refused"].get("reason"))
            )
        load = gpu.get("concurrent_load") or {}
        finding_parts.append(f"loadavg {load.get('loadavg')}.")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY" if gpu is not None else "STATIC_ONLY",
        "gpu_authority": False,
        "took_gpu_lease": gpu is not None,
        "claim_boundary": CLAIM_BOUNDARY,
        "lever_id": LEVER_ID,
        "dense_rematerialization": DIRECT_CONSUME,
        "forbidden_shape": MATERIALIZE_F16_AUX,
        "argmax_alone_is_not_parity": True,
        "shader_invariants": shader,
        "fma_per_byte_by_class": {
            "fma_only": classes,
            "exp_counted_as_1": classes_exp1,
        },
        "economics": _py(econ),
        "gpu_ab": None if gpu is None else _py(gpu),
        "error_did_not_move": None if error_block is None else _py(error_block),
        "honest_outcome": None if outcome is None else _py(outcome),
        "finding": " ".join(finding_parts),
        "auxiliary_bytes_target": AUXILIARY_BYTES_TARGET,
        "native_down_proj_relfro": NATIVE_DOWN_PROJ_RELFRO,
        "all_layers_structured_bar": ALL_LAYERS_STRUCTURED_BAR,
        "sources": [
            NATIVE_RECEIPT_REL,
            ERROR_BUDGET_REL,
            "receipts/future/AUX_CAPABILITY_SCREEN.json",
            SHADER_REL,
            EXAMPLE_REL,
            "tools/future/executable_economics.py",
        ],
    }


def record(
    *,
    measurement: Mapping[str, Any] | None = None,
    gpu_blocked_reason: str | None = None,
    path: Path | None = None,
) -> Path:
    if measurement is None and not gpu_blocked_reason:
        raise IncompleteAB(
            "refusing to record a LUT receipt without a GPU A/B or an explicit "
            "gpu_blocked_reason (never fabricate GB/s)"
        )
    doc = build(measurement=measurement)
    if measurement is None:
        binary = None
        bins = example_binaries()
        if bins:
            binary = str(bins[0])
        doc["gpu_ab"] = gpu_blocked_ab(gpu_blocked_reason or "unspecified", binary=binary)
        doc["evidence_class"] = "STATIC_ONLY"
        doc["took_gpu_lease"] = False
        doc["gpu_blocked"] = True
        doc["finding"] = (
            str(doc.get("finding") or "")
            + " GPU A/B was NOT measured: "
            + str(gpu_blocked_reason)
            + " No GB/s or gpu_ns is recorded."
        )
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
            lane="aux_u8_lut",
            # A receipt rebuilt from a stored raw capture must not stamp the
            # rebuild time as the measurement time. The raw files carry no
            # timestamp, so a retrofit records the measurement time as UNKNOWN.
            retrofit=not os.environ.get("HAWKING_MEASURED_NOW"),
        ),
    )
    write_measured_receipt(out, doc, "tools/future/aux_u8_lut.py")
    return out


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--from", dest="raw_path", default=None)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DEFAULT)
    parser.add_argument("--layer", type=int, default=3)
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

    measured = None if raw is None else measurement_from_raw(raw)
    if measured is not None:
        inc = measured["incumbent"]
        exp = measured["exp_variant"]
        lut = measured["lut_variant"]
        print(
            f"A/B layer {measured['layer']}: incumbent {inc['effective_gb_s']} GB/s "
            f"{inc['gpu_us_median']} us / exp {exp['effective_gb_s']} GB/s "
            f"{exp['gpu_us_median']} us / lut-{measured.get('chosen_placement')} "
            f"{lut['effective_gb_s']} GB/s {lut['gpu_us_median']} us  "
            f"loadavg={measured['concurrent_load'].get('loadavg')}",
            flush=True,
        )
        for name, arm in measured["placements"].items():
            print(
                f"  placement {name}: {arm['effective_gb_s']} GB/s {arm['gpu_us_median']} us",
                flush=True,
            )

    if args.record:
        path = record(measurement=measured, path=args.out)
        print(f"wrote {path}")
        return 0

    if measured is None:
        print(
            "nothing to do: pass --measure or --from RAW.json",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
