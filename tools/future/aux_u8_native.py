#!/usr/bin/env python3
"""NATIVE CONSUMER for the screened u8 aux. It does not sit in a notebook.

quantize_aux_u8 survived AUX_CAPABILITY_SCREEN as a real u8 encode of the
incumbent f16 aux (log scale + linear bias, 2-bit codes kept): organ cosine
0.998, logit KL 0.003, argmax 1.00, FITTED_HELDOUT. Campaign law: a byte
reduction does not remain a fit. This module is the consumer.

    The runtime reads u8 scale/bias and decodes them in-register
    (exp of the log-minmax code, linear minmax for bias) inside the
    same inner loop that does w = q*scale+bias; acc += w*x.

    Expanding the compact aux back to an f16 scale/bias array and
    feeding the ordinary kernel is REJECTED. That shape gives the
    bytes back at the point they matter.

    python3 tools/future/aux_u8_native.py --prove-layer --measure --record
    python3 -m pytest tools/future/test_aux_u8_native.py -q

evidence_class SELF_MEASURED_DIRTY for the GPU A/B (absolute GB/s is
measured-under-load; the ratio is back-to-back in one process). CPU
equivalence is DIAGNOSTIC_RELATIVE against the screened fit. Does not
change production shaders.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import aux_capability_screen as acs
from tools.future import executable_economics as ee
from tools.future._common import (
    REPO,
    RECEIPTS,
    load_json,
    measurement_provenance,
    write_measured_receipt,
)
from tools.future.mlp_auxiliary_information import (
    AUXILIARY_BYTES_TARGET,
    INCUMBENT_GROUP,
    _unpack_q,
)
from tools.future.mlp_teacher_corpus import HIDDEN, INTERMEDIATE, silu, _matmul


RECEIPT = RECEIPTS / "AUX_U8_NATIVE.json"
RAW_DEFAULT = RECEIPTS / "_AUX_U8_NATIVE_raw.json"
SCHEMA = "hawking.future.aux_u8_native.v1"
VERSION = 1
RECORDED_BY = "tools/future/aux_u8_native.py"
EXAMPLE_NAME = "aux_u8_ab"
SHADER_REL = "crates/hawking-core/examples/aux_u8_ab.metal"
EXAMPLE_REL = "crates/hawking-core/examples/aux_u8_ab.rs"
SCREEN_REL = "receipts/future/AUX_CAPABILITY_SCREEN.json"

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
MATERIALIZE_F16_AUX = "MATERIALIZE_F16_AUX"

LEVER_ID = "quantize_aux_u8"

# Task-stated bars (rounded). The screened fit is the operational match.
TASK_ORGAN_COSINE_BAR = 0.998
TASK_LOGIT_KL_BAR = 0.003
TASK_ARGMAX_BAR = 1.00

# Screened fit (AUX_CAPABILITY_SCREEN, quantize_aux_u8). A native path
# that lands here has matched the fit; worse is the finding.
SCREENED_ORGAN_COSINE = 0.998450457881851
SCREENED_LOGIT_KL = 0.0033654539143456084
SCREENED_ARGMAX = 1.0
SCREENED_WEIGHT_RELFRO = 0.03410476504048445

N_LAYERS = 64
N_TENSORS = 192
GROUP = INCUMBENT_GROUP  # 64

ARTIFACT_DEFAULT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")

CLAIM_BOUNDARY = (
    "Native consumer of the screened u8 aux: in-register log-scale exp + "
    "linear bias, 2-bit codes kept. GPU A/B is one representative MLP layer "
    "of sealed-3.14, incumbent f16-aux kernel vs native u8-aux kernel, "
    "back-to-back in one process, MTLCommandBuffer GPUStartTime/GPUEndTime. "
    "Absolute GB/s is measured-under-load (loadavg recorded); it is not a "
    "clean roof. Token-level ms/TPS from the probe is ARITHMETIC over the "
    "measured layer (x64), labelled a projection, not a resident measurement. "
    "CPU organ/logit numbers are native in-register consume vs the incumbent "
    "on the teacher-corpus hold split; the screened fit is cited beside them. "
    "A path that expands u8 aux to an f16 aux buffer and then calls the "
    "ordinary kernel is refused. Production shaders are not edited. "
    "FMA/byte is the inner-loop tax before and after; extra decode arithmetic "
    "is billed through executable_economics.py."
)


class NativeRefuse(ValueError):
    """The native consumer refused rather than guessing."""


class ArgmaxAloneParityRefuse(NativeRefuse):
    """Argmax agreement is not equivalence."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: argmax agreement is not parity; a logit-space number "
            f"(KL in nats) is required{extra}"
        )


class MaterializeF16AuxRefuse(NativeRefuse):
    """Compact aux was expanded back to the incumbent f16 buffer."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: expanding u8 aux back to an f16 aux buffer and feeding "
            f"the ordinary kernel is the forbidden shape{extra}"
        )


class IncompleteAB(NativeRefuse):
    """A/B is missing a measured arm, GB/s, or loadavg."""


class EmptyGpuSample(NativeRefuse):
    """Raised rather than divide by a missing GPU timestamp."""


# ---------------------------------------------------------------------------
# Inner-loop tax. Same counting convention as mlp_alu_roofline.DECODE_TAX.
# Affine2 g64: 8 weights / iter from 2 B codes + aux + 8 x-floats.
# ---------------------------------------------------------------------------

INCUMBENT_INNER = {
    "weights_per_iteration": 8,
    "weight_bytes_per_iteration": 6,  # 2 code + 2 scale + 2 bias
    "x_bytes_per_iteration": 32,
    "dequant_fma": 8,  # float(q)*scale+bias
    "mac_fma": 8,
    "aux_decode_fma": 0,  # half-to-float is a conversion, not an FMA
    "exp": 0,
    "bitops": 16,
    "int_to_float": 8,
}

NATIVE_INNER = {
    "weights_per_iteration": 8,
    "weight_bytes_per_iteration": 4,  # 2 code + 1 scale_u8 + 1 bias_u8
    "x_bytes_per_iteration": 32,
    "dequant_fma": 8,
    "mac_fma": 8,
    "aux_decode_fma": 2,  # lmin + u8*span  and  bmin + u8*span
    "exp": 1,  # log-scale
    "bitops": 16,
    "int_to_float": 10,  # 8 q + 2 u8
}


def _fma_per_weight_byte(inner: Mapping[str, int], *, count_exp_as: float = 0.0) -> dict[str, float]:
    wb = int(inner["weight_bytes_per_iteration"])
    dequant = int(inner["dequant_fma"])
    mac = int(inner["mac_fma"])
    aux = int(inner["aux_decode_fma"])
    exp_n = float(inner["exp"]) * float(count_exp_as)
    total_fma = dequant + mac + aux + exp_n
    decode_fma = dequant + aux + exp_n
    return {
        "weight_bytes_per_iteration": wb,
        "fma_per_weight_byte": round(total_fma / wb, 4),
        "decode_fma_per_weight_byte": round(decode_fma / wb, 4),
        "dequant_fma_per_weight_byte": round(dequant / wb, 4),
        "aux_decode_fma_per_weight_byte": round((aux + exp_n) / wb, 4),
        "mac_fma_per_weight_byte": round(mac / wb, 4),
        "counted_exp_as_fma": float(count_exp_as),
    }


def fma_per_byte_table(*, count_exp_as: float = 0.0) -> dict[str, Any]:
    """Before (incumbent f16 aux) and after (native u8 aux)."""
    before = {**INCUMBENT_INNER, **_fma_per_weight_byte(INCUMBENT_INNER, count_exp_as=0.0)}
    after = {**NATIVE_INNER, **_fma_per_weight_byte(NATIVE_INNER, count_exp_as=count_exp_as)}
    return {
        "convention": (
            "One geo_tpr64 inner iteration = 8 weights. Incumbent loads "
            "2 B codes + 2 B f16 scale + 2 B f16 bias. Native loads 2 B "
            "codes + 1 B u8 scale + 1 B u8 bias and decodes scale/bias in "
            "register (1 FMA + 1 exp for log-scale, 1 FMA for linear bias) "
            "before the same dequant-FMA. exp is counted only when "
            "count_exp_as > 0; the default table counts FMA only so the "
            "exp tax is visible as its own column."
        ),
        "before": before,
        "after": after,
        "delta": {
            "weight_bytes_per_iteration": after["weight_bytes_per_iteration"]
            - before["weight_bytes_per_iteration"],
            "fma_per_weight_byte": round(
                after["fma_per_weight_byte"] - before["fma_per_weight_byte"], 4
            ),
            "decode_fma_per_weight_byte": round(
                after["decode_fma_per_weight_byte"] - before["decode_fma_per_weight_byte"],
                4,
            ),
            "note": (
                "Fewer bytes, more FMA per remaining byte. u1alu put the MLP "
                "at 1.33 dequant-FMA per weight-byte against 0.88 needed to "
                "reach 497 GB/s; this lever moves the other way."
            ),
        },
    }


def extra_flops_per_output_element(*, count_exp_as: float = 0.0) -> float:
    """Mean extra decode ops per MLP output element (gate+up+down, 64 layers).

    Per inner iteration of 8 weights: 2 FMA (scale, bias) + count_exp_as * exp.
    Gate/up: cols=5120 → 640 inners/row. Down: cols=17408 → 2176 inners/row.
    """
    extra_per_inner = 2.0 + float(count_exp_as)
    gate_up = (HIDDEN / 8.0) * extra_per_inner  # 640 * extra
    down = (INTERMEDIATE / 8.0) * extra_per_inner  # 2176 * extra
    n_out = 2 * INTERMEDIATE + HIDDEN
    total = 2 * INTERMEDIATE * gate_up + HIDDEN * down
    return float(total / n_out)


# ---------------------------------------------------------------------------
# Equivalence. Argmax alone is a loud refuse.
# ---------------------------------------------------------------------------


def report_equivalence(
    *,
    organ_cosine: float | None,
    kl_nats: float | None,
    argmax_agreement: float | None,
    top_k_agreement: float | None = None,
    n_rows: int | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Organ cosine + logit KL are the bar. Argmax is a side report.

    Passing only argmax_agreement (kl_nats is None) raises
    ArgmaxAloneParityRefuse. A candidate that keeps argmax while drifting
    in KL has not been shown equivalent.
    """
    if kl_nats is None:
        raise ArgmaxAloneParityRefuse(
            f"kl_nats={kl_nats!r} organ_cosine={organ_cosine!r} "
            f"argmax_agreement={argmax_agreement!r}"
        )
    if organ_cosine is None:
        raise NativeRefuse(
            "REFUSED: organ cosine is required alongside logit KL; "
            f"kl_nats={kl_nats!r} argmax_agreement={argmax_agreement!r}"
        )
    return {
        "organ_cosine": float(organ_cosine),
        "kl_nats": float(kl_nats),
        "top_k": int(k),
        "top_k_agreement": None if top_k_agreement is None else float(top_k_agreement),
        "argmax_agreement": None if argmax_agreement is None else float(argmax_agreement),
        "argmax_is_not_parity": True,
        "n_rows": None if n_rows is None else int(n_rows),
        "parity_quantities": ["organ_cosine", "kl_nats"],
        "task_bar": {
            "organ_cosine": TASK_ORGAN_COSINE_BAR,
            "logit_kl_nats": TASK_LOGIT_KL_BAR,
            "argmax_agreement": TASK_ARGMAX_BAR,
        },
        "screened_fit": {
            "organ_cosine": SCREENED_ORGAN_COSINE,
            "logit_kl_nats": SCREENED_LOGIT_KL,
            "argmax_agreement": SCREENED_ARGMAX,
        },
        "clears_task_bar": bool(
            float(organ_cosine) >= TASK_ORGAN_COSINE_BAR
            and float(kl_nats) <= TASK_LOGIT_KL_BAR
            and (argmax_agreement is None or float(argmax_agreement) >= TASK_ARGMAX_BAR)
        ),
        "matches_or_beats_screened_fit": bool(
            float(organ_cosine) >= SCREENED_ORGAN_COSINE
            and float(kl_nats) <= SCREENED_LOGIT_KL
            and (argmax_agreement is None or float(argmax_agreement) >= SCREENED_ARGMAX)
        ),
    }


# ---------------------------------------------------------------------------
# Pack / consume. Native never holds a decoded aux of n_groups.
# ---------------------------------------------------------------------------


@dataclass
class AuxAllocLedger:
    """Peak decoded aux elements held at once. Native must stay << n_groups."""

    max_decoded_aux_elems: int = 0
    wrote_f16_aux_buffer: bool = False
    notes: list[str] = field(default_factory=list)

    def note_decoded_aux(self, n: int) -> None:
        n = int(n)
        if n > self.max_decoded_aux_elems:
            self.max_decoded_aux_elems = n


@dataclass
class PackedU8Aux:
    """u8 aux + incumbent 2-bit codes. No decoded f16 scale/bias array."""

    q: np.ndarray  # (rows, gpr, group) uint8 in {0,1,2,3}
    scale_u8: np.ndarray  # (rows, gpr) uint8
    bias_u8: np.ndarray
    scale_lmin: float
    scale_lmax: float
    bias_min: float
    bias_max: float
    rows: int
    cols: int
    group_size: int = GROUP
    path: str | None = None
    organ: str | None = None
    layer: int | None = None

    @property
    def n_groups(self) -> int:
        return int(self.scale_u8.size)

    @property
    def gpr(self) -> int:
        return int(self.cols) // int(self.group_size)

    @property
    def native_aux_bytes(self) -> int:
        return int(self.scale_u8.size) + int(self.bias_u8.size)

    @property
    def incumbent_aux_bytes(self) -> int:
        return int(self.n_groups) * 4  # f16 + f16

    @property
    def endpoint_bytes(self) -> int:
        return 16  # 4 f32

    def endpoints(self) -> dict[str, float]:
        slmin, slmax = float(self.scale_lmin), float(self.scale_lmax)
        bmin, bmax = float(self.bias_min), float(self.bias_max)
        return {
            "scale_lmin": slmin,
            "scale_lmax": slmax,
            "scale_span": 0.0 if slmax <= slmin else (slmax - slmin) / 255.0,
            "bias_min": bmin,
            "bias_max": bmax,
            "bias_span": 0.0 if bmax <= bmin else (bmax - bmin) / 255.0,
        }


def pack_u8_aux(
    q: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    *,
    rows: int,
    cols: int,
    group_size: int = GROUP,
    path: str | None = None,
    organ: str | None = None,
    layer: int | None = None,
) -> PackedU8Aux:
    """Keep 2-bit codes; replace f16 scale/bias with the screened u8 encode."""
    s_q, s_lo, s_hi = acs.u8_log_encode(np.asarray(scale).reshape(-1))
    b_q, b_lo, b_hi = acs.u8_linear_encode(np.asarray(bias).reshape(-1))
    gpr = int(cols) // int(group_size)
    q_arr = np.asarray(q)
    if q_arr.ndim == 2:
        q_arr = q_arr.reshape(int(rows), gpr, int(group_size))
    return PackedU8Aux(
        q=np.ascontiguousarray(q_arr, dtype=np.uint8),
        scale_u8=s_q.reshape(int(rows), gpr),
        bias_u8=b_q.reshape(int(rows), gpr),
        scale_lmin=float(s_lo),
        scale_lmax=float(s_hi),
        bias_min=float(b_lo),
        bias_max=float(b_hi),
        rows=int(rows),
        cols=int(cols),
        group_size=int(group_size),
        path=path,
        organ=organ,
        layer=layer,
    )


def pack_from_incumbent(packed: Mapping[str, Any], **kwargs: Any) -> PackedU8Aux:
    rows, cols = int(packed["shape"][0]), int(packed["shape"][1])
    return pack_u8_aux(
        packed["q"],
        packed["scale"],
        packed["bias"],
        rows=rows,
        cols=cols,
        group_size=int(packed.get("group_size") or GROUP),
        path=packed.get("path"),
        **kwargs,
    )


def decode_u8_scale_column(packed: PackedU8Aux, gi: int) -> np.ndarray:
    """One group-column of scales, in-register. Shape (rows,), not n_groups."""
    ep = packed.endpoints()
    su = packed.scale_u8[:, int(gi)].astype(np.float64, copy=False)
    if ep["scale_span"] == 0.0:
        return np.full(su.shape, math.exp(ep["scale_lmin"]), dtype=np.float32)
    return np.exp(ep["scale_lmin"] + su * ep["scale_span"]).astype(np.float32)


def decode_u8_bias_column(packed: PackedU8Aux, gi: int) -> np.ndarray:
    ep = packed.endpoints()
    bu = packed.bias_u8[:, int(gi)].astype(np.float64, copy=False)
    if ep["bias_span"] == 0.0:
        return np.full(bu.shape, ep["bias_min"], dtype=np.float32)
    return (ep["bias_min"] + bu * ep["bias_span"]).astype(np.float32)


def native_matvec_u8(
    x: np.ndarray,
    packed: PackedU8Aux,
    *,
    ledger: AuxAllocLedger | None = None,
) -> np.ndarray:
    """y = x @ W.T with W decoded group-by-group in-register from u8 aux.

    Never allocates a decoded scale/bias array of n_groups. Peak decoded
    aux is 2*rows (one group-column of scale and of bias).
    """
    xc = np.ascontiguousarray(x, dtype=np.float32)
    if xc.ndim == 1:
        xc = xc[None, :]
    if int(xc.shape[1]) != int(packed.cols):
        raise NativeRefuse(f"x cols {xc.shape[1]} != packed.cols {packed.cols}")
    n = int(xc.shape[0])
    rows = int(packed.rows)
    g = int(packed.group_size)
    gpr = packed.gpr
    y = np.zeros((n, rows), dtype=np.float32)
    q = packed.q.astype(np.float32, copy=False)
    for gi in range(gpr):
        scale = decode_u8_scale_column(packed, gi)
        bias = decode_u8_bias_column(packed, gi)
        if ledger is not None:
            ledger.note_decoded_aux(int(scale.size) + int(bias.size))
        w_slice = q[:, gi, :] * scale[:, None] + bias[:, None]
        x_slice = xc[:, gi * g : (gi + 1) * g]
        y += x_slice @ w_slice.T
        del scale, bias, w_slice
    if ledger is not None:
        ledger.wrote_f16_aux_buffer = False
        ledger.notes.append("native_matvec_u8: group-column decode, no f16 aux buffer")
    return y[0] if np.asarray(x).ndim == 1 else y


def forbidden_remat_f16_aux(
    packed: PackedU8Aux,
    *,
    ledger: AuxAllocLedger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The forbidden shape: expand u8 aux to a dense decoded scale/bias array.

    Exists so tests can refuse it. Not a consumer.
    """
    ep = packed.endpoints()
    su = packed.scale_u8.astype(np.float64, copy=False).reshape(-1)
    bu = packed.bias_u8.astype(np.float64, copy=False).reshape(-1)
    if ep["scale_span"] == 0.0:
        scale = np.full(su.shape, math.exp(ep["scale_lmin"]), dtype=np.float32)
    else:
        scale = np.exp(ep["scale_lmin"] + su * ep["scale_span"]).astype(np.float32)
    if ep["bias_span"] == 0.0:
        bias = np.full(bu.shape, ep["bias_min"], dtype=np.float32)
    else:
        bias = (ep["bias_min"] + bu * ep["bias_span"]).astype(np.float32)
    if ledger is not None:
        ledger.note_decoded_aux(int(scale.size) + int(bias.size))
        ledger.wrote_f16_aux_buffer = True
        ledger.notes.append("forbidden_remat_f16_aux: decoded n_groups scale+bias")
    return scale.reshape(packed.scale_u8.shape), bias.reshape(packed.bias_u8.shape)


def remat_then_ordinary_matvec(
    x: np.ndarray,
    packed: PackedU8Aux,
    *,
    ledger: AuxAllocLedger | None = None,
) -> np.ndarray:
    """Forbidden consumer: rematerialize decoded aux, then the ordinary affine."""
    scale, bias = forbidden_remat_f16_aux(packed, ledger=ledger)
    q = packed.q.astype(np.float32, copy=False)
    w = (q * scale[:, :, None] + bias[:, :, None]).reshape(packed.rows, packed.cols)
    xc = np.ascontiguousarray(x, dtype=np.float32)
    if xc.ndim == 1:
        return w @ xc
    return xc @ w.T


def classify_consumer(
    *,
    materializes_f16_aux: bool,
    binds_u8_aux: bool,
    peak_decoded_aux_elems: int | None = None,
    n_groups: int | None = None,
) -> str:
    """DIRECT_CONSUME or a loud refuse. No quiet pass for remat."""
    if materializes_f16_aux:
        raise MaterializeF16AuxRefuse("materializes_f16_aux=True")
    if n_groups is not None and peak_decoded_aux_elems is not None:
        # Remat holds decoded scale AND bias of every group at once (2*n_groups).
        # Native holds one group-column (2*rows). gpr==1 is the degenerate
        # case where a column is the whole aux; real MLP tensors have gpr>=80.
        if int(peak_decoded_aux_elems) >= 2 * int(n_groups):
            raise MaterializeF16AuxRefuse(
                f"peak decoded aux {peak_decoded_aux_elems} >= 2*n_groups "
                f"{2 * int(n_groups)}"
            )
    if not binds_u8_aux:
        raise NativeRefuse("REFUSED: consumer does not bind u8 aux")
    return DIRECT_CONSUME


def native_shader_source() -> str:
    path = REPO / SHADER_REL
    if not path.is_file():
        raise NativeRefuse(f"REFUSED: native shader {SHADER_REL} is not on disk")
    return path.read_text(encoding="utf-8")


def native_shader_invariants(src: str | None = None) -> dict[str, Any]:
    """Static proof the Metal consumer does not materialize f16 aux."""
    text = native_shader_source() if src is None else str(src)
    required = (
        "device const uchar* scales_u8",
        "device const uchar* biases_u8",
        "exp(",
        "aux_u8_native_affine_q2_geo_tpr64_tg128",
        "aux_u8_incumbent_affine_q2_geo_tpr64_tg128",
    )
    missing = [p for p in required if p not in text]
    # Forbidden: a device half aux produced from u8, or a store of decoded aux.
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
    native_fn = "kernel void aux_u8_native_affine_q2_geo_tpr64_tg128"
    native_binds_half_aux = False
    if native_fn in text:
        body = text.split(native_fn, 1)[1]
        # The native kernel's parameter list must not take device half* aux.
        header = body.split("{", 1)[0]
        native_binds_half_aux = "device const half*" in header or "device half*" in header
    ok = not missing and not forbidden_hits and not native_binds_half_aux
    return {
        "ok": ok,
        "materializes_f16_aux": bool(forbidden_hits or native_binds_half_aux),
        "missing_required": missing,
        "forbidden_hits": forbidden_hits,
        "native_binds_half_aux": native_binds_half_aux,
        "binds_u8_aux": "device const uchar* scales_u8" in text
        and "device const uchar* biases_u8" in text,
        "in_register_exp": "exp(" in text,
        "shader": SHADER_REL,
    }


# ---------------------------------------------------------------------------
# Organ consume (SwiGLU) from three packed tensors.
# ---------------------------------------------------------------------------


def swiglu_native(
    x: np.ndarray,
    gate: PackedU8Aux,
    up: PackedU8Aux,
    down: PackedU8Aux,
    *,
    ledger: AuxAllocLedger | None = None,
) -> np.ndarray:
    g = native_matvec_u8(x, gate, ledger=ledger)
    u = native_matvec_u8(x, up, ledger=ledger)
    h = silu(g) * u
    del g, u
    return native_matvec_u8(h, down, ledger=ledger)


def incumbent_matvec(x: np.ndarray, packed_inc: Mapping[str, Any]) -> np.ndarray:
    w = np.asarray(packed_inc["W"], dtype=np.float32)
    xc = np.ascontiguousarray(x, dtype=np.float32)
    if xc.ndim == 1:
        return w @ xc
    return xc @ w.T


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return acs.cosine(a, b)


def relfro(a: np.ndarray, b: np.ndarray) -> float:
    return acs.relfro(a, b)


# ---------------------------------------------------------------------------
# Economics. Bill extra decode arithmetic; the screen billed extra_flops=0.
# ---------------------------------------------------------------------------


def byte_model() -> dict[str, int]:
    return acs.u8_aux_bytes()


def score_u8_aux(*, extra_flops_per_output_element: float, count_exp_as: float | None = None) -> dict[str, Any]:
    bm = byte_model()
    scored = ee.score(
        bytes_removed=int(bm["bytes_removed"]),
        bytes_added={"metadata": int(bm["bytes_added_metadata"])},
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        consuming_primitive="FusedDecodeCompute",
        bandwidth_regime="affine_q2_family",
        organ="mlp",
        # BROADCAST AUX, not weight codes. The per-group scale and bias are read
        # by many threads, so they are cache-served and were never on the
        # critical path - which is why this lever billed +1.5541 ms and measured
        # 29.04 us SLOWER for 8,355,840 fewer bytes. executable_economics now
        # refuses an undeclared stream_class rather than defaulting to the organ
        # average, and this candidate is the reason that guard exists.
        stream_class="broadcast_aux",
        reusable_family=True,
        candidate_id=LEVER_ID,
        status="OPEN",
    )
    scored["count_exp_as"] = count_exp_as
    scored["byte_model"] = bm
    return scored


def economics_bundle() -> dict[str, Any]:
    """Bytes-only (what the screen billed) vs billed extra decode arithmetic."""
    fma_only = extra_flops_per_output_element(count_exp_as=0.0)
    fma_plus_exp1 = extra_flops_per_output_element(count_exp_as=1.0)
    bytes_only = score_u8_aux(extra_flops_per_output_element=0.0, count_exp_as=None)
    with_fma = score_u8_aux(extra_flops_per_output_element=fma_only, count_exp_as=0.0)
    with_exp = score_u8_aux(extra_flops_per_output_element=fma_plus_exp1, count_exp_as=1.0)
    return {
        "bytes_only_screen_style": bytes_only,
        "with_aux_decode_fma": with_fma,
        "with_aux_decode_fma_and_exp_as_1": with_exp,
        "extra_flops_per_output_element_fma_only": fma_only,
        "extra_flops_per_output_element_fma_plus_exp1": fma_plus_exp1,
        "note": (
            "The capability screen billed extra_flops_per_output_element=0. "
            "Native consume adds 2 FMA + 1 exp per 8 weights. u1alu showed "
            "the MLP is near its arithmetic ceiling (1.33 dequant-FMA per "
            "weight-byte vs 0.88 to reach 497 GB/s). If flop_ms_delta eats "
            "byte_removed_ms, the lever is worth less than 1.554 ms and "
            "possibly nothing. The GPU A/B is the physical number."
        ),
    }


# ---------------------------------------------------------------------------
# GPU A/B. Real timestamps or a refuse — never a fabricated GB/s.
# ---------------------------------------------------------------------------


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise NativeRefuse("weight_bytes must be positive to form a bandwidth")
    return float(weight_bytes) / float(gpu_ns)


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
    for k in ("endpoint_bytes", "materializes_f16_aux", "endpoint_note"):
        if k in raw:
            out[k] = raw[k]
    return out


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse an A/B that has no loadavg or no measured GPU ns."""
    if not isinstance(raw, Mapping):
        raise IncompleteAB("raw A/B is not a mapping")
    load = raw.get("concurrent_load")
    if not isinstance(load, Mapping) or not load.get("loadavg"):
        raise IncompleteAB("A/B without loadavg is not a measurement")
    inc = _arm_view(raw.get("incumbent") or {}, "incumbent")
    nat = _arm_view(raw.get("native_u8") or {}, "native_u8")
    if nat.get("materializes_f16_aux") is True:
        raise MaterializeF16AuxRefuse("native_u8 arm materialized f16 aux")
    inc_ns = int(inc["gpu_ns_median"])
    nat_ns = int(nat["gpu_ns_median"])
    inc_b = int(inc["weight_bytes"])
    nat_b = int(nat["weight_bytes"])
    delta_ns = nat_ns - inc_ns
    delta_bytes = nat_b - inc_b
    inc_gb = float(inc["effective_gb_s"])
    nat_gb = float(nat["effective_gb_s"])
    layer = int(raw.get("layer", -1))
    # Token-level projection: 64 layers, arithmetic over this probe.
    proj_ns = delta_ns * N_LAYERS
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
        "native_u8": nat,
        "delta": {
            "weight_bytes": delta_bytes,
            "gpu_ns": delta_ns,
            "gpu_us": round(delta_ns / 1e3, 4),
            "effective_gb_s": round(nat_gb - inc_gb, 4),
            "time_ratio_native_over_incumbent": round(nat_ns / inc_ns, 6) if inc_ns else None,
            "byte_ratio_native_over_incumbent": round(nat_b / inc_b, 6) if inc_b else None,
        },
        "paired_reps": raw.get("paired_reps") or [],
        "projections": raw.get("projections") or [],
        "output_cosine_mean": raw.get("output_cosine_mean"),
        "bytes_removed_this_layer": raw.get("bytes_removed_this_layer", inc_b - nat_b),
        "token_projection": {
            "kind": "PROJECTION_ARITHMETIC_OVER_PROBE",
            "not_a_resident_measurement": True,
            "layers": N_LAYERS,
            "probe_layer": layer,
            "probe_delta_gpu_us": round(delta_ns / 1e3, 4),
            "projected_mlp_delta_us": round(proj_ns / 1e3, 4),
            "projected_mlp_delta_ms": round(proj_ns / 1e6, 6),
            "formula": "measured_layer_delta_ns * 64",
            "note": (
                "64 identical MLP layers assumed. Not a protected generate "
                "gate and not a host+GPU complete-token measurement."
            ),
        },
        "occupancy": raw.get("occupancy") or {},
        "shader": raw.get("shader") or SHADER_REL,
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


# ---------------------------------------------------------------------------
# CPU prove on a real (or synthetic) layer.
# ---------------------------------------------------------------------------


def pack_layer(layer: int, *, root: Path | None = None) -> dict[str, PackedU8Aux]:
    index = acs.organ_index(root)
    out: dict[str, PackedU8Aux] = {}
    for organ in ("mlp.gate", "mlp.up", "mlp.down"):
        rec = index.get((int(layer), organ))
        if rec is None:
            raise NativeRefuse(f"missing {organ} layer {layer}")
        packed = acs.load_affine_q2(Path(rec["segment_path"]))
        out[organ] = pack_from_incumbent(packed, organ=organ, layer=int(layer))
        del packed
    return out


def _hold_matching_screen(
    layer: int,
    corpus_dir: Path,
    *,
    organ_hold_cap: int | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Replay the screen's shared-RNG hold draw (layers 3, 31, 38, then 63)."""
    for prior in (3, 31, 38, 63):
        hold = acs.load_hold_xy(
            prior, corpus_dir, max_rows=organ_hold_cap, rng=rng
        )
        if int(prior) == int(layer):
            return hold
    return hold


def remat_swiglu_oracle(
    x: np.ndarray,
    gate: PackedU8Aux,
    up: PackedU8Aux,
    down: PackedU8Aux,
) -> np.ndarray:
    """Dense-W SwiGLU of the u8 encode. Numerical oracle of the screened fit.

    Not a consumer: it materializes decoded aux and W.
    """
    def _w(p: PackedU8Aux) -> np.ndarray:
        scale, bias = forbidden_remat_f16_aux(p)
        q = p.q.astype(np.float32, copy=False)
        return (q * scale[:, :, None] + bias[:, :, None]).reshape(p.rows, p.cols)

    return acs.swiglu_from_weights(x, _w(gate), _w(up), _w(down))


def prove_layer_equivalence(
    layer: int,
    *,
    root: Path | None = None,
    organ_hold_cap: int | None = acs.ORGAN_HOLD_ROW_CAP,
    logit_hold_rows: int = acs.LOGIT_HOLD_ROWS,
    do_logits: bool = False,
    match_screen_hold: bool = True,
    vs_fit_oracle: bool = True,
) -> dict[str, Any]:
    """Native in-register consume vs incumbent on hold rows.

    Logits are optional (LM head is expensive) but report_equivalence
    will not accept argmax without KL: if do_logits is false the logit
    block is skipped=true, not an argmax-only pass.
    """
    rng = np.random.default_rng(acs.RNG_SEED)
    corpus_dir = acs.resolve_corpus_dir()
    if match_screen_hold:
        hold = _hold_matching_screen(
            layer, corpus_dir, organ_hold_cap=organ_hold_cap, rng=rng
        )
    else:
        hold = acs.load_hold_xy(layer, corpus_dir, max_rows=organ_hold_cap, rng=rng)
    x = hold["X"]
    y_inc = hold["Y"]
    packed = pack_layer(layer, root=root)
    ledger = AuxAllocLedger()
    y_hat = swiglu_native(
        x, packed["mlp.gate"], packed["mlp.up"], packed["mlp.down"], ledger=ledger
    )
    n_groups = packed["mlp.gate"].n_groups
    classify_consumer(
        materializes_f16_aux=ledger.wrote_f16_aux_buffer,
        binds_u8_aux=True,
        peak_decoded_aux_elems=ledger.max_decoded_aux_elems,
        n_groups=n_groups,
    )
    organ_cos = cosine(y_hat, y_inc)
    organ_rf = relfro(y_hat, y_inc)
    y_fit = None
    vs_fit: dict[str, Any] | None = None
    if vs_fit_oracle:
        y_fit = remat_swiglu_oracle(
            x, packed["mlp.gate"], packed["mlp.up"], packed["mlp.down"]
        )
        vs_fit = {
            "role": "numerical oracle of the screened u8 encode; FORBIDDEN as a consumer",
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "cosine_native_vs_fit": float(cosine(y_hat, y_fit)),
            "relfro_native_vs_fit": float(relfro(y_hat, y_fit)),
            "cosine_fit_vs_incumbent": float(cosine(y_fit, y_inc)),
            "note": (
                "Native in-register consume vs the same u8 encode materialized "
                "to W and fed to ordinary SwiGLU. Cosine ~1 means the consumer "
                "matches the fit; any remaining gap vs incumbent is the encode, "
                "not the lowering."
            ),
        }
    organ_block = {
        "cosine": float(organ_cos),
        "relfro": float(organ_rf),
        "failed": bool(organ_cos < TASK_ORGAN_COSINE_BAR),
        "layer": int(layer),
        "n_hold_used": int(hold["n_hold_used"]),
        "n_hold_available": int(hold["n_hold_available"]),
        "n_hold_prompts": int(hold["n_hold_prompts"]),
        "split": "hold",
        "split_unit": "prompt_id",
        "native_peak_decoded_aux_elems": int(ledger.max_decoded_aux_elems),
        "n_groups": int(n_groups),
        "materializes_f16_aux": False,
        "vs_screened_fit_cosine": SCREENED_ORGAN_COSINE,
        "gap_vs_screened_fit_cosine": float(organ_cos) - SCREENED_ORGAN_COSINE,
        "vs_fit_oracle": vs_fit,
    }
    logit_block: dict[str, Any]
    eq: dict[str, Any] | None
    if do_logits:
        if int(layer) != 63:
            raise NativeRefuse("logit-space uses last-layer organ output (layer 63)")
        n = min(int(logit_hold_rows), int(y_inc.shape[0]))
        inc_logits = acs.lm_head_batch(y_inc[:n])
        hat_logits = acs.lm_head_batch(y_hat[:n])
        parity = acs.mean_logit_parity(inc_logits, hat_logits, k=5)
        logit_block = {
            **parity,
            "layer": 63,
            "n_rows": n,
            "kind": "lm_head_on_last_layer_organ_output",
            "not_full_stack_generate": True,
            "skipped": False,
            "vs_screened_fit_kl": SCREENED_LOGIT_KL,
            "gap_vs_screened_fit_kl": float(parity["kl_nats"]) - SCREENED_LOGIT_KL,
        }
        eq = report_equivalence(
            organ_cosine=organ_cos,
            kl_nats=float(parity["kl_nats"]),
            argmax_agreement=float(parity["argmax_agreement"]),
            top_k_agreement=float(parity["top_k_agreement"]),
            n_rows=n,
        )
    else:
        logit_block = {
            "skipped": True,
            "reason": "logits not requested; organ-space only. Not an argmax-only pass.",
        }
        eq = None
    return {
        "layer": int(layer),
        "organ_space": organ_block,
        "logit_space": logit_block,
        "equivalence": eq,
        "native_consumer": {
            "materializes_f16_aux": False,
            "peak_decoded_aux_elems": int(ledger.max_decoded_aux_elems),
            "n_groups": int(n_groups),
            "classify": DIRECT_CONSUME,
        },
        "packed": {
            organ: {
                "rows": p.rows,
                "cols": p.cols,
                "n_groups": p.n_groups,
                "native_aux_bytes": p.native_aux_bytes,
                "incumbent_aux_bytes": p.incumbent_aux_bytes,
                "endpoint_bytes": p.endpoint_bytes,
                "endpoints": p.endpoints(),
            }
            for organ, p in packed.items()
        },
    }


def screened_u8() -> dict[str, Any]:
    doc = load_json(REPO / SCREEN_REL)
    for lev in doc.get("levers") or []:
        if lev.get("id") == LEVER_ID:
            return {
                "evidence_tier": lev.get("evidence_tier"),
                "weight_space": lev.get("weight_space"),
                "organ_space": {
                    "cosine_mean": (lev.get("organ_space") or {}).get("cosine_mean"),
                    "relfro_mean": (lev.get("organ_space") or {}).get("relfro_mean"),
                    "failed": (lev.get("organ_space") or {}).get("failed"),
                    "per_layer": (lev.get("organ_space") or {}).get("per_layer"),
                },
                "logit_space": lev.get("logit_space"),
                "bytes_removed": lev.get("bytes_removed"),
                "bytes_added": lev.get("bytes_added"),
            }
    raise NativeRefuse("quantize_aux_u8 missing from AUX_CAPABILITY_SCREEN")


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def _py(x: Any) -> Any:
    return acs._py(x)


def build(
    *,
    measurement: Mapping[str, Any] | None = None,
    prove: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shader = native_shader_invariants()
    if shader["materializes_f16_aux"] or not shader["ok"]:
        raise MaterializeF16AuxRefuse(f"shader invariants {shader}")
    fma = fma_per_byte_table(count_exp_as=0.0)
    fma_exp = fma_per_byte_table(count_exp_as=1.0)
    econ = economics_bundle()
    gpu = None if measurement is None else dict(measurement)
    finding_parts = [
        "Native u8-aux consumer binds uchar scale/bias and decodes them "
        "in-register (log-scale exp + linear bias). It does not expand "
        "the compact aux back to f16.",
        (
            f"Inner-loop tax: incumbent {fma['before']['decode_fma_per_weight_byte']} "
            f"decode-FMA/weight-byte at {fma['before']['weight_bytes_per_iteration']} B/iter; "
            f"native {fma['after']['decode_fma_per_weight_byte']} at "
            f"{fma['after']['weight_bytes_per_iteration']} B/iter"
            f" ({fma['after']['fma_per_weight_byte']} FMA/byte vs "
            f"{fma['before']['fma_per_weight_byte']} before)."
        ),
    ]
    bytes_only = econ["bytes_only_screen_style"]
    with_fma = econ["with_aux_decode_fma"]
    finding_parts.append(
        f"executable_economics bytes-only: {bytes_only['predicted_ms_saved']:.4f} ms saved "
        f"(the screen's +1.554 ms). With aux-decode FMA billed: "
        f"{with_fma['predicted_ms_saved']:.4f} ms "
        f"(flop_ms {with_fma['terms']['flop_ms_delta']:.4f}, "
        f"byte_ms {with_fma['terms']['byte_ms_delta']:.4f})."
    )
    if gpu is not None:
        d = gpu["delta"]
        finding_parts.append(
            f"GPU A/B layer {gpu['layer']}: incumbent {gpu['incumbent']['effective_gb_s']} GB/s "
            f"in {gpu['incumbent']['gpu_us_median']} us, native "
            f"{gpu['native_u8']['effective_gb_s']} GB/s in "
            f"{gpu['native_u8']['gpu_us_median']} us, delta "
            f"{d['gpu_us']} us / {d['weight_bytes']} bytes. "
            f"Token projection {gpu['token_projection']['projected_mlp_delta_ms']} ms "
            f"({gpu['token_projection']['kind']})."
        )
    if prove is not None and prove.get("equivalence"):
        eq = prove["equivalence"]
        finding_parts.append(
            f"Native CPU consume: organ cosine {eq['organ_cosine']:.6f} "
            f"(screened fit {SCREENED_ORGAN_COSINE:.6f}, bar {TASK_ORGAN_COSINE_BAR}), "
            f"KL {eq['kl_nats']:.6f} nats (screened {SCREENED_LOGIT_KL:.6f}, "
            f"task bar {TASK_LOGIT_KL_BAR}), argmax {eq['argmax_agreement']}."
        )
        if not eq.get("matches_or_beats_screened_fit"):
            finding_parts.append(
                "Native is worse than the screened fit on at least one bar; "
                "that gap is the finding."
            )
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
        "bars": {
            "task_organ_cosine": TASK_ORGAN_COSINE_BAR,
            "task_logit_kl_nats": TASK_LOGIT_KL_BAR,
            "task_argmax_agreement": TASK_ARGMAX_BAR,
            "screened_organ_cosine": SCREENED_ORGAN_COSINE,
            "screened_logit_kl_nats": SCREENED_LOGIT_KL,
            "screened_argmax_agreement": SCREENED_ARGMAX,
            "note": (
                "Task stated KL 0.003 / cosine 0.998 as the screened numbers. "
                "The receipt's exact screened fit is KL "
                f"{SCREENED_LOGIT_KL} / cosine {SCREENED_ORGAN_COSINE}. "
                "Native is judged against both; a gap vs the fit is reported."
            ),
        },
        "shader_invariants": shader,
        "fma_per_byte": {
            "fma_only": fma,
            "exp_counted_as_1": fma_exp,
        },
        "economics": _py(econ),
        "screened_fit": screened_u8(),
        "gpu_ab": None if gpu is None else _py(gpu),
        "native_prove": None if prove is None else _py(prove),
        "finding": " ".join(finding_parts),
        "auxiliary_bytes_target": AUXILIARY_BYTES_TARGET,
        "sources": [
            SCREEN_REL,
            "receipts/future/MLP_AUXILIARY_INFORMATION.json",
            "receipts/future/MLP_ALU_ROOFLINE.json",
            SHADER_REL,
            EXAMPLE_REL,
        ],
    }


def current_loadavg() -> dict[str, Any]:
    """Host loadavg. Not a GPU number."""
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
    """An A/B that did not run. No GB/s, no gpu_ns — never a fabricated number."""
    return {
        "status": "NOT_MEASURED",
        "reason": str(reason),
        "concurrent_load": current_loadavg(),
        "absolute_gb_s_are_measured_under_load": False,
        "fabricated_hardware_numbers": False,
        "probe_binary": binary,
        "incumbent": None,
        "native_u8": None,
        "delta": None,
        "token_projection": None,
        "needs": "unsandboxed Metal (gate profile): MTLCreateSystemDefaultDevice was nil",
    }


def record(
    *,
    measurement: Mapping[str, Any] | None = None,
    prove: Mapping[str, Any] | None = None,
    gpu_blocked_reason: str | None = None,
    path: Path | None = None,
) -> Path:
    if measurement is None and not gpu_blocked_reason:
        raise IncompleteAB(
            "refusing to record a native receipt without a GPU A/B or an explicit "
            "gpu_blocked_reason (never fabricate GB/s)"
        )
    doc = build(measurement=measurement, prove=prove)
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
            lane="aux_u8_native",
            # A receipt rebuilt from a stored raw capture must not stamp the
            # rebuild time as the measurement time. The raw files carry no
            # timestamp, so a retrofit records the measurement time as UNKNOWN.
            retrofit=not os.environ.get("HAWKING_MEASURED_NOW"),
        ),
    )
    write_measured_receipt(out, doc, "tools/future/aux_u8_native.py")
    return out


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--from", dest="raw_path", default=None)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--prove-layer", action="store_true")
    parser.add_argument("--logits", action="store_true", help="LM-head KL on layer 63 hold rows")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DEFAULT)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--logit-layer", type=int, default=63)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    prove: dict[str, Any] | None = None
    if args.prove_layer:
        layer = int(args.logit_layer if args.logits else args.layer)
        print(f"proving native consume on layer {layer} logits={args.logits}", flush=True)
        prove = prove_layer_equivalence(
            layer,
            root=args.artifact_root if args.artifact_root.is_dir() else None,
            do_logits=bool(args.logits),
        )
        org = prove["organ_space"]
        print(
            f"  organ cosine={org['cosine']:.6f} relfro={org['relfro']:.6f} "
            f"peak_aux={org['native_peak_decoded_aux_elems']} n_groups={org['n_groups']}",
            flush=True,
        )
        if prove.get("equivalence"):
            eq = prove["equivalence"]
            print(
                f"  KL={eq['kl_nats']:.6f} argmax={eq['argmax_agreement']} "
                f"clears_task_bar={eq['clears_task_bar']} "
                f"matches_fit={eq['matches_or_beats_screened_fit']}",
                flush=True,
            )

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
        nat = measured["native_u8"]
        print(
            f"A/B layer {measured['layer']}: incumbent {inc['effective_gb_s']} GB/s "
            f"{inc['gpu_us_median']} us / native {nat['effective_gb_s']} GB/s "
            f"{nat['gpu_us_median']} us  loadavg={measured['concurrent_load'].get('loadavg')}",
            flush=True,
        )

    if args.record:
        path = record(measurement=measured, prove=prove, path=args.out)
        print(f"wrote {path}")
        return 0

    if measured is None and prove is None:
        print(
            "nothing to do: pass --measure, --from RAW.json, and/or --prove-layer",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
