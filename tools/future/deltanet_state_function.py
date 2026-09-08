"""DELTANET STATE FUNCTION — what recurrent update the mixer actually needs.

DeltaNet is 2,961,659,904 catalog bytes per token (29.98%). Its q/k/v/z codes
are at their entropy floor (H(q) ≈ 3.47 of 4 bits, all four sub-blocks within
0.014 bits) and heterogeneous precision is MEASURED_NEGATIVE
(DELTANET_QKVZ_PRECISION). This module therefore stops treating the organ as
four matrices to compress and asks what recurrent state update the model
needs.

The incumbent update

    S_t = (I - beta k k^T) (decay * S_{t-1}) + beta k v^T
    h_t = S_t^T q

is a parameterisation, not a requirement. A different parameterisation that
preserves the state trajectory and downstream logits is a win even if its
weights look nothing like W_qkvz. Reconstructing those matrices is the wrong
bar. Unpacking to dense W then running ordinary GEMV is REJECTED_DENSE_REMAT.

    python3 tools/future/deltanet_state_function.py --build
    python3 -m pytest tools/future/test_deltanet_state_function.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future._common import write_receipt
from tools.future.ebpw_categories import PRODUCTION, judge_dense_rematerialization
from tools.future.physical_primitives import ATLAS_PRIMITIVES
from tools.future import deltanet_representation as dnr


RECEIPT = "DELTANET_STATE_FUNCTION.json"
SCHEMA = "hawking.future.deltanet_state_function.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_state_function.py"
PREDECESSOR_REPR = "receipts/future/DELTANET_REPRESENTATION.json"
PREDECESSOR_QKVZ = "receipts/future/DELTANET_QKVZ_PRECISION.json"
BA_DELTA_REL = "receipts/future/BA_DELTA_AB.json"
DISPATCH_MOTIFS_REL = "receipts/future/DISPATCH_MOTIFS.json"
QN_REL = "tools/headless/negative_science.py"

DELTANET_ACTIVE_TARGET = dnr.DELTANET_ACTIVE_TARGET
QKVZ_ACTIVE_TARGET = dnr.QKVZ_ACTIVE_TARGET
OUT_ACTIVE_TARGET = dnr.OUT_ACTIVE_TARGET
BA_ACTIVE_TARGET = dnr.BA_ACTIVE_TARGET
CONV_ACTIVE_TARGET = dnr.CONV_ACTIVE_TARGET
TOKEN_ACTIVE_TARGET = dnr.TOKEN_ACTIVE_TARGET

# Geometry-derived f32 S. Not HQ38M20. 48 layers * 48 heads * 128 * 128 * 4.
REC_STATE_RESIDENT = 150_994_944
REC_STATE_RW_PER_TOKEN = REC_STATE_RESIDENT * 2
CONV_STATE_RESIDENT = 5_898_240
CONV_STATE_RW_PER_TOKEN = CONV_STATE_RESIDENT * 2
A_LOG_BYTES = 9_600
DT_BIAS_BYTES = 9_600
LINEAR_ATTN_NORM_BYTES = 24_960

HIDDEN = 5120
KEY_HEADS = 16
VALUE_HEADS = 48
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
VALUES_PER_KEY = 3
CONV_KERNEL = 4
N_DN_LAYERS = 48
QKVZ_ROWS = 16384
F32_BYTES = 4
F16_BYTES = 2

# Named architecture points used for executable economics. Not a fit.
NAMED_NARROW_DIM = 64
NAMED_RANK = 16
NAMED_DSTATE = 16
NAMED_BANDWIDTH = 8
NAMED_DPLR_RANK = 8
SPECTRUM_TOKENS = 256
SPECTRUM_HEADS = 8
RNG_SEED = 38
ENERGY_BAR = 0.99

# Sealed-3.14 dispatch (DISPATCH_MOTIFS / BA_DELTA_AB). Launch counts, not ns.
SEALED_DISPATCHES = 628
DN_LAYER_LAUNCHES_SEALED = 337
GATED_DELTA_LAUNCHES = 48
BA_TO_DECAY_LAUNCHES = 48
FUSE_BA_DELTA_DISPATCHES = 580

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FALSIFIED = "ALREADY_FALSIFIED"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
OPEN = "OPEN"
EXISTING_LEVER = "EXISTING_LEVER"
UNMEASURED = "UNMEASURED"

REMOVED_PARTS: tuple[str, ...] = ("catalog_weights", "state", "other")
ADDED_PARTS: tuple[str, ...] = (
    "generator",
    "embeddings",
    "residuals",
    "metadata",
    "state",
    "other",
)

REQUIRED_CANDIDATE_IDS: tuple[str, ...] = (
    "smaller_state_machine",
    "learned_recurrence",
    "structured_transition",
    "generated_transition_coefficients",
    "conditional_recurrence",
    "shared_transforms_w_unmeasured",
    "fused_update_consume",
    "share_or_merge_state_across_depth",
    "emit_w_then_ordinary_gemv",
    "qkvz_bit_descent",
)

REQUIRED_CANDIDATE_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "mechanism",
    "byte_model",
    "bytes_removed",
    "bytes_added",
    "extra_flops",
    "dispatch_change",
    "physical_primitive",
    "cheapest_falsifier",
    "dense_rematerialization",
    "status",
    "evidence_class",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "The 2,961,659,904 figure is catalog stored bytes of the four DeltaNet "
    "weight organs. Recurrent state (150,994,944 f32 resident) is geometry, "
    "not HQ38M20, and is accounted separately. Candidates are judged on "
    "function preservation (state trajectory and downstream logits), not on "
    "reconstructing W_qkvz. A synthetic SVD of S under isotropic coefficients "
    "is not the trained trajectory. A candidate that unpacks to dense W then "
    "runs ordinary GEMV is REJECTED_DENSE_REMAT. A compression ratio with no "
    "bytes_removed and bytes_added is not a candidate."
)


class StateFunctionRefuse(ValueError):
    """The state-function census refused rather than guessing."""


class UnreconciledState(StateFunctionRefuse):
    """Geometry-derived recurrent-state bytes do not equal 150,994,944."""

    def __init__(self, got: int, want: int = REC_STATE_RESIDENT, *, detail: str = "") -> None:
        self.got = int(got)
        self.want = int(want)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: recurrent-state resident bytes {got} != recorded {want}{extra}"
        )


class MissingEconomics(StateFunctionRefuse):
    """A candidate has no executable byte economics."""

    def __init__(self, cand_id: str, *, missing: Sequence[str]) -> None:
        self.cand_id = cand_id
        self.missing = tuple(missing)
        super().__init__(
            f"REFUSED: candidate {cand_id!r} has no executable economics "
            f"(missing {list(self.missing)}; a compression ratio is not a candidate). "
            "Every candidate must carry both bytes_removed and bytes_added."
        )


# ---------------------------------------------------------------------------
# Geometry and incumbent economics.
# ---------------------------------------------------------------------------


def geometry() -> dict[str, Any]:
    """Sealed-3.14 DeltaNet layout. Constants are the recorded specimen, not a guess."""
    vpk = VALUE_HEADS // KEY_HEADS
    if vpk != VALUES_PER_KEY:
        raise StateFunctionRefuse(f"values_per_key {vpk} != {VALUES_PER_KEY}")
    q_rows = KEY_HEADS * KEY_HEAD_DIM
    k_rows = KEY_HEADS * KEY_HEAD_DIM
    v_rows = VALUE_HEADS * VALUE_HEAD_DIM
    z_rows = VALUE_HEADS * VALUE_HEAD_DIM
    rec = VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM
    conv_channels = q_rows + k_rows + v_rows
    conv_state = conv_channels * (CONV_KERNEL - 1)
    return {
        "hidden_size": HIDDEN,
        "key_heads": KEY_HEADS,
        "value_heads": VALUE_HEADS,
        "key_head_dim": KEY_HEAD_DIM,
        "value_head_dim": VALUE_HEAD_DIM,
        "values_per_key": vpk,
        "conv_kernel": CONV_KERNEL,
        "n_deltanet_layers": N_DN_LAYERS,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "v_rows": v_rows,
        "z_rows": z_rows,
        "qkvz_rows": q_rows + k_rows + v_rows + z_rows,
        "qkvz_cols": HIDDEN,
        "ba_rows": KEY_HEADS * vpk * 2,
        "out_rows": HIDDEN,
        "out_cols": v_rows,
        "conv_channels": conv_channels,
        "recurrent_state_elements_per_layer": rec,
        "conv_state_elements_per_layer": conv_state,
        "qkvz_layout": "per_key_head_Q128_K128_V384_Z384",
    }


def rec_state_resident_bytes(
    *,
    n_layers: int = N_DN_LAYERS,
    value_heads: int = VALUE_HEADS,
    key_dim: int = KEY_HEAD_DIM,
    value_dim: int = VALUE_HEAD_DIM,
) -> int:
    return int(n_layers) * int(value_heads) * int(key_dim) * int(value_dim) * F32_BYTES


def conv_state_resident_bytes(
    *,
    n_layers: int = N_DN_LAYERS,
    conv_channels: int,
    conv_kernel: int = CONV_KERNEL,
) -> int:
    return int(n_layers) * int(conv_channels) * (int(conv_kernel) - 1) * F32_BYTES


def reconcile_state(got: int, want: int = REC_STATE_RESIDENT, *, detail: str = "") -> int:
    if int(got) != int(want):
        raise UnreconciledState(int(got), int(want), detail=detail)
    return int(got)


def q4_stored(rows: int, cols: int, *, n_layers: int = 1) -> int:
    return int(dnr.q4_parts(int(rows), int(cols))["stored_bytes"]) * int(n_layers)


def f32v2_stored(n_elements: int, *, n_layers: int = 1) -> int:
    return int(dnr.f32v2_parts(int(n_elements))["stored_bytes"]) * int(n_layers)


def gated_delta_flops_per_token(
    *,
    n_layers: int = N_DN_LAYERS,
    value_heads: int = VALUE_HEADS,
    key_dim: int = KEY_HEAD_DIM,
    value_dim: int = VALUE_HEAD_DIM,
) -> int:
    """Exact flop count of one token of the incumbent vi_simd update + readout.

    Per head: decay*S (kd*vd muls), k^T S (2 kd vd), delta (2 vd),
    outer-product write (2 kd vd), add into S (kd vd), h = S^T q (2 kd vd).
    """
    kd = int(key_dim)
    vd = int(value_dim)
    per_head = (kd * vd) + (2 * kd * vd) + (2 * vd) + (2 * kd * vd) + (kd * vd) + (2 * kd * vd)
    return int(n_layers) * int(value_heads) * per_head


def gemv_flops(rows: int, cols: int, *, n_layers: int = 1) -> int:
    return 2 * int(rows) * int(cols) * int(n_layers)


def incumbent_flops() -> dict[str, Any]:
    delta = gated_delta_flops_per_token()
    qkvz = gemv_flops(QKVZ_ROWS, HIDDEN, n_layers=N_DN_LAYERS)
    ba = gemv_flops(KEY_HEADS * VALUES_PER_KEY * 2, HIDDEN, n_layers=N_DN_LAYERS)
    out = gemv_flops(HIDDEN, VALUE_HEADS * VALUE_HEAD_DIM, n_layers=N_DN_LAYERS)
    return {
        "gated_delta_per_token": delta,
        "qkvz_gemv_per_token": qkvz,
        "ba_gemv_per_token": ba,
        "out_gemv_per_token": out,
        "note": (
            "qkvz GEMV dwarfs gated-delta arithmetic. A smaller S without "
            "shrinking W_qkvz saves S traffic, not the 8e9-class projection."
        ),
    }


def incumbent_operator() -> dict[str, Any]:
    geo = geometry()
    rec = rec_state_resident_bytes()
    reconcile_state(rec, detail="geometry()")
    info = dnr.independent_information(geo)
    return {
        "operator": info["operator"],
        "source": info["source"],
        "parameterisation_not_requirement": True,
        "judged_on": (
            "state trajectory and downstream logits; not reconstruction of "
            "the stored q/k/v/z matrices"
        ),
        "update_rank_per_token": 1,
        "state_rank_capacity_per_head": KEY_HEAD_DIM,
        "state_not_closed_under_rank_r": (
            "A rank-r factorisation of S is not invariant under a rank-1 "
            "write. After T tokens rank(S) can reach min(T, 128) per head "
            "unless each step truncates (a different function)."
        ),
        "every_element_rw_every_token": True,
        "roles": info["roles"],
        "state_rank_growth": info["state_rank_growth"],
        "consumed_by": "qwen38_gated_delta_decode_vi_simd",
        "z_enters_gated_delta": False,
        "physical_primitive": _require_primitive("LocalStateMachine"),
    }


def incumbent_bytes() -> dict[str, Any]:
    geo = geometry()
    rec = rec_state_resident_bytes()
    reconcile_state(rec)
    conv = conv_state_resident_bytes(conv_channels=int(geo["conv_channels"]))
    if conv != CONV_STATE_RESIDENT:
        raise UnreconciledState(conv, CONV_STATE_RESIDENT, detail="conv_state")
    qkvz = q4_stored(int(geo["qkvz_rows"]), HIDDEN, n_layers=N_DN_LAYERS)
    if qkvz != QKVZ_ACTIVE_TARGET:
        raise StateFunctionRefuse(f"qkvz stored {qkvz} != {QKVZ_ACTIVE_TARGET}")
    out = q4_stored(int(geo["out_rows"]), int(geo["out_cols"]), n_layers=N_DN_LAYERS)
    if out != OUT_ACTIVE_TARGET:
        raise StateFunctionRefuse(f"out stored {out} != {OUT_ACTIVE_TARGET}")
    ba = q4_stored(int(geo["ba_rows"]), HIDDEN, n_layers=N_DN_LAYERS)
    if ba != BA_ACTIVE_TARGET:
        raise StateFunctionRefuse(f"ba stored {ba} != {BA_ACTIVE_TARGET}")
    conv_w = f32v2_stored(int(geo["conv_channels"]) * CONV_KERNEL, n_layers=N_DN_LAYERS)
    if conv_w != CONV_ACTIVE_TARGET:
        raise StateFunctionRefuse(f"conv1d stored {conv_w} != {CONV_ACTIVE_TARGET}")
    catalog = qkvz + out + ba + conv_w
    if catalog != DELTANET_ACTIVE_TARGET:
        raise StateFunctionRefuse(f"DeltaNet catalog {catalog} != {DELTANET_ACTIVE_TARGET}")
    return {
        "catalog_weights": {
            "attention.linear_qkvz": qkvz,
            "attention.linear_out": out,
            "attention.linear_ba": ba,
            "attention.linear_conv1d": conv_w,
            "total": catalog,
            "share_of_token": catalog / TOKEN_ACTIVE_TARGET,
        },
        "adjacent_catalog_not_in_2gb": {
            "state.A_log": A_LOG_BYTES,
            "state.dt_bias": DT_BIAS_BYTES,
            "norms.linear_attn": LINEAR_ATTN_NORM_BYTES,
            "total": A_LOG_BYTES + DT_BIAS_BYTES + LINEAR_ATTN_NORM_BYTES,
        },
        "state": {
            "recurrent_resident": rec,
            "recurrent_rw_per_token": rec * 2,
            "conv_resident": conv,
            "conv_rw_per_token": conv * 2,
            "resident": rec + conv,
            "rw_per_token": (rec + conv) * 2,
            "in_catalog": False,
            "in_2gb_bar": False,
            "dtype": "f32",
        },
        "identity": {
            "model_id": "qwen3.8-27b-sealed-3.14",
            "resident_identity": "sealed-3.14",
            "geometry": {
                "hidden_size": HIDDEN,
                "n_deltanet_layers": N_DN_LAYERS,
                "key_heads": KEY_HEADS,
                "value_heads": VALUE_HEADS,
                "key_head_dim": KEY_HEAD_DIM,
                "value_head_dim": VALUE_HEAD_DIM,
                "qkvz_rows": QKVZ_ROWS,
                "qkvz_layout": geo["qkvz_layout"],
            },
        },
        "token_active_bytes": TOKEN_ACTIVE_TARGET,
        "reconciled": True,
    }


# ---------------------------------------------------------------------------
# Economics. Load-bearing refusal.
# ---------------------------------------------------------------------------


def _nonneg_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateFunctionRefuse(f"{name} must be a non-negative int, got {type(value).__name__}")
    if value < 0:
        raise StateFunctionRefuse(f"{name} must be >= 0, got {value}")
    return value


def _as_breakdown(value: Any, *, added: bool, name: str) -> dict[str, int]:
    parts = ADDED_PARTS if added else REMOVED_PARTS
    if isinstance(value, int) and not isinstance(value, bool):
        total = _nonneg_int(value, name=name)
        return {"total": total}
    if not isinstance(value, Mapping):
        raise StateFunctionRefuse(
            f"{name} must be a non-negative int or a breakdown dict with total"
        )
    if "total" not in value:
        raise StateFunctionRefuse(f"{name} breakdown is missing total")
    total = _nonneg_int(value["total"], name=f"{name}.total")
    out: dict[str, int] = {"total": total}
    summed = 0
    seen = 0
    for key in parts:
        if key not in value:
            continue
        n = _nonneg_int(value[key], name=f"{name}.{key}")
        out[key] = n
        summed += n
        seen += 1
    if seen == len(parts) and summed != total:
        raise StateFunctionRefuse(
            f"{name}: parts {summed} != total {total}"
        )
    return out


def require_economics(cand: Mapping[str, Any] | None) -> tuple[int, int]:
    """Refuse unless both bytes_removed and bytes_added are present and integer.

    A compression ratio (or bytes_eliminated_if_true alone) is not a candidate.
    This is the load-bearing guard: executable economics, not a packing slogan.
    """
    if not isinstance(cand, Mapping):
        raise MissingEconomics("?", missing=("bytes_removed", "bytes_added"))
    cid = str(cand.get("id") or "?")
    missing: list[str] = []
    if "bytes_removed" not in cand:
        missing.append("bytes_removed")
    if "bytes_added" not in cand:
        missing.append("bytes_added")
    if missing:
        raise MissingEconomics(cid, missing=missing)
    removed = _as_breakdown(cand["bytes_removed"], added=False, name=f"{cid}.bytes_removed")
    added = _as_breakdown(cand["bytes_added"], added=True, name=f"{cid}.bytes_added")
    return int(removed["total"]), int(added["total"])


def removed(**parts: int) -> dict[str, int]:
    catalog = int(parts.get("catalog_weights", 0))
    state = int(parts.get("state", 0))
    other = int(parts.get("other", 0))
    total = catalog + state + other
    return {
        "catalog_weights": catalog,
        "state": state,
        "other": other,
        "total": total,
    }


def added(**parts: int) -> dict[str, int]:
    generator = int(parts.get("generator", 0))
    embeddings = int(parts.get("embeddings", 0))
    residuals = int(parts.get("residuals", 0))
    metadata = int(parts.get("metadata", 0))
    state = int(parts.get("state", 0))
    other = int(parts.get("other", 0))
    total = generator + embeddings + residuals + metadata + state + other
    return {
        "generator": generator,
        "embeddings": embeddings,
        "residuals": residuals,
        "metadata": metadata,
        "state": state,
        "other": other,
        "total": total,
    }


def extra_flops_row(per_token: int, *, formula: str, relative_to: str) -> dict[str, Any]:
    if isinstance(per_token, bool) or not isinstance(per_token, int):
        raise StateFunctionRefuse("extra_flops.per_token must be an int")
    return {
        "per_token": int(per_token),
        "formula": formula,
        "relative_to": relative_to,
    }


def dispatch_row(incumbent: int, candidate: int, *, surface: str) -> dict[str, Any]:
    inc = _nonneg_int(incumbent, name="dispatch.incumbent")
    cand = _nonneg_int(candidate, name="dispatch.candidate")
    return {
        "incumbent": inc,
        "candidate": cand,
        "delta": cand - inc,
        "surface": surface,
    }


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise StateFunctionRefuse(f"{name} is not an atlas primitive")
    return name


def _remat_tag(decompresses_w: bool, ordinary: bool) -> str:
    verdict = judge_dense_rematerialization(
        {
            "path_kind": PRODUCTION,
            "dense_rematerialization": decompresses_w,
            "decompresses_to_dense_weight_tensor": decompresses_w,
            "runs_ordinary_kernels": ordinary,
            "consumes_representation_directly": (not decompresses_w),
        }
    )
    if decompresses_w:
        if verdict.ok:
            raise StateFunctionRefuse("expected REJECTED_DENSE_REMAT for emit-W lowering")
        return REJECTED_DENSE_REMAT
    return DIRECT_CONSUME


def _py(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# Synthetic spectrum of the update. Not the trained trajectory.
# ---------------------------------------------------------------------------


def gated_delta_step(
    state: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q. Vectorized vi kernel."""
    decayed = state * decay[:, None, None]
    kv_mem = np.einsum("hkv,hk->hv", decayed, key)
    delta = (value - kv_mem) * beta[:, None]
    state2 = decayed + np.einsum("hk,hv->hkv", key, delta)
    h = np.einsum("hkv,hk->hv", state2, query)
    return state2, h


def _summ(xs: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([float(x) for x in xs], dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
    }


def synthetic_update_spectrum(
    *,
    n_tokens: int = SPECTRUM_TOKENS,
    n_heads: int = SPECTRUM_HEADS,
    dim: int = KEY_HEAD_DIM,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """SVD of S after a rank-1 rollout from zero. Isotropic coefficients, not real x.

    Answers whether the UPDATE is structurally low-rank regardless of W.
    Does not answer whether the trained function uses 128 directions.
    """
    rng = np.random.default_rng(int(seed))
    d = int(dim)
    h = int(n_heads)
    S = np.zeros((h, d, d), dtype=np.float32)
    numeric_rank_cap = min(int(n_tokens), d)
    for _ in range(int(n_tokens)):
        q = rng.normal(size=(h, d)).astype(np.float32)
        k = rng.normal(size=(h, d)).astype(np.float32)
        v = rng.normal(size=(h, d)).astype(np.float32)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / math.sqrt(d)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        decay = rng.uniform(0.85, 0.99, size=(h,)).astype(np.float32)
        beta = rng.uniform(0.2, 0.9, size=(h,)).astype(np.float32)
        S, _h = gated_delta_step(S, q, k, v, decay, beta)
    ranks: list[int] = []
    energy_at = {1: [], 4: [], 8: [], 16: [], 32: [], 64: [], 128: []}
    for i in range(h):
        svals = np.linalg.svd(S[i].astype(np.float64), compute_uv=False)
        energy = np.cumsum(svals * svals)
        tot = float(energy[-1]) if energy.size else 0.0
        if tot <= 0.0:
            frac = np.zeros_like(energy)
        else:
            frac = energy / tot
        thresh = 1e-4 * float(svals[0]) if svals.size else 0.0
        ranks.append(int((svals > thresh).sum()) if svals.size else 0)
        for r, bucket in energy_at.items():
            if frac.size == 0:
                bucket.append(0.0)
            else:
                bucket.append(float(frac[min(int(r), frac.size) - 1]))
    e16 = _summ(energy_at[16])
    structurally_low_rank = bool(e16["mean"] is not None and e16["mean"] >= ENERGY_BAR)
    return {
        "measured": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "n_tokens": int(n_tokens),
        "n_heads": int(n_heads),
        "dim": d,
        "seed": int(seed),
        "state_init": "S=0",
        "coefficients": (
            "isotropic Gaussian q,k,v then L2 like the shader; decay U(0.85,0.99); "
            "beta U(0.2,0.9). Not post-norm x and not a generate gate."
        ),
        "numeric_rank": _summ(ranks),
        "rank_capacity": numeric_rank_cap,
        "energy_at_rank": {str(r): _summ(v) for r, v in energy_at.items()},
        "energy_bar": ENERGY_BAR,
        "structurally_low_rank_at_r16": structurally_low_rank,
        "update_fills_rank": not structurally_low_rank,
        "trained_function_rank": UNMEASURED,
        "note": (
            "If energy at r=16 is below 0.99, the update fills rank under generic "
            "inputs and a rank-r store is function replacement, not a compression "
            "of this mixer. The trained trajectory is a different measurement."
        ),
    }


# ---------------------------------------------------------------------------
# Negative index. Query with AND without a model (GENERAL_PHYSICAL).
# ---------------------------------------------------------------------------


GENERAL_PHYSICAL_PROBE_FAMILIES: tuple[str, ...] = (
    "prefill_over_generated_token_denominator",
    "environment_mismatch_unfused_vs_sealed",
    "source_instrumented_runtime_binary_stale",
    "adjacency_is_not_overlap",
)


def probe_general_physical() -> list[dict[str, Any]]:
    """GENERAL_PHYSICAL scars must refuse with a named parent and with none.

    Campaign process scars are not DeltaNet proposals. They are queried here
    so a model-specific receipt cannot silently drop them — the defect the
    scars themselves record.
    """
    try:
        from tools.future.negative_index import refuse_if_dead
    except Exception as exc:  # pragma: no cover
        return [{"index_error": f"{type(exc).__name__}: {exc}"}]
    out: list[dict[str, Any]] = []
    for slug in GENERAL_PHYSICAL_PROBE_FAMILIES:
        for model in ("qwen3.8-27b", None):
            proposal: dict[str, Any] = {"hypothesis_family": slug}
            if model is not None:
                proposal["model"] = model
            refusal = refuse_if_dead(proposal)
            out.append(
                {
                    "queried_slug": slug,
                    "queried_model": model,
                    "query_mode": "with_model" if model else "without_model",
                    "refused": bool(refusal and refusal.get("refused")),
                    "level": None if not refusal else refusal.get("level"),
                    "scar_id": None if not refusal else refusal.get("scar_id"),
                    "source_path": None if not refusal else refusal.get("source_path"),
                }
            )
    return out


def _index_hits(family_slugs: Sequence[str]) -> list[dict[str, Any]]:
    try:
        from tools.future.negative_index import refuse_if_dead
    except Exception as exc:  # pragma: no cover
        return [{"index_error": f"{type(exc).__name__}: {exc}"}]
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    organs = ("deltanet", "attention")
    models: tuple[str | None, ...] = ("qwen3.8-27b", None)
    for slug in family_slugs:
        for organ in organs:
            for model in models:
                proposal: dict[str, Any] = {
                    "organ": organ,
                    "hypothesis_family": slug,
                }
                if model is not None:
                    proposal["model"] = model
                refusal = refuse_if_dead(proposal)
                if not refusal:
                    continue
                key = f"{refusal.get('scar_id')}|{model}|{organ}|{slug}"
                if key in seen:
                    continue
                seen.add(key)
                level = str(refusal.get("level") or "")
                general = level == "GENERAL_PHYSICAL"
                hits.append(
                    {
                        "scar_id": refusal.get("scar_id"),
                        "source_path": refusal.get("source_path"),
                        "hypothesis_family": refusal.get("hypothesis_family"),
                        "organ": refusal.get("organ"),
                        "verdict": refusal.get("verdict"),
                        "claim_refuted": refusal.get("claim_refuted"),
                        "reopen_condition": refusal.get("reopen_condition"),
                        "level": level,
                        "queried_slug": slug,
                        "queried_organ": organ,
                        "queried_model": model,
                        "query_mode": "with_model" if model else "without_model",
                        "general_physical": general,
                        "applies_regardless_of_model": general,
                    }
                )
    return hits


def _nns_cite(nns_id: str, *, this_object: str) -> dict[str, Any]:
    return dnr._nns_cite(nns_id, this_object=this_object)


def _qn_cite(qn_id: str, claim: str, reopen: str, *, organ: str) -> dict[str, Any]:
    return dnr._qn_cite(qn_id, claim, reopen, organ=organ)


# ---------------------------------------------------------------------------
# Named architecture points (executable numbers, not a fit).
# ---------------------------------------------------------------------------


def narrow_mixer_bytes(dim: int = NAMED_NARROW_DIM) -> dict[str, Any]:
    """Same head counts, key/value dim reduced. Closed under the incumbent update."""
    d = int(dim)
    if d <= 0 or d >= KEY_HEAD_DIM:
        raise StateFunctionRefuse(f"narrow dim {d} is not a reduction of {KEY_HEAD_DIM}")
    q_rows = KEY_HEADS * d
    k_rows = KEY_HEADS * d
    v_rows = VALUE_HEADS * d
    z_rows = VALUE_HEADS * d
    qkvz_rows = q_rows + k_rows + v_rows + z_rows
    conv_channels = q_rows + k_rows + v_rows
    qkvz = q4_stored(qkvz_rows, HIDDEN, n_layers=N_DN_LAYERS)
    out = q4_stored(HIDDEN, v_rows, n_layers=N_DN_LAYERS)
    conv_w = f32v2_stored(conv_channels * CONV_KERNEL, n_layers=N_DN_LAYERS)
    rec = rec_state_resident_bytes(key_dim=d, value_dim=d)
    conv_s = conv_state_resident_bytes(conv_channels=conv_channels)
    return {
        "dim": d,
        "qkvz_rows": qkvz_rows,
        "qkvz": qkvz,
        "out": out,
        "conv1d": conv_w,
        "rec_state": rec,
        "conv_state": conv_s,
        "ba_unchanged": BA_ACTIVE_TARGET,
    }


def learned_recurrence_bytes(d_state: int = NAMED_DSTATE) -> dict[str, Any]:
    """Named compact SSM: per-head diagonal A of size d_state, B/C/dt from a Q4 in_proj.

    Replaces q,k,v,ba,conv,A_log,dt_bias,S,conv_state. Keeps z rows of qkvz and out_proj
    (output gate and residual map are a different organ).
    """
    ds = int(d_state)
    in_rows = VALUE_HEADS * ds * 3  # B, C, dt-in
    in_proj = q4_stored(in_rows, HIDDEN, n_layers=N_DN_LAYERS)
    a_diag = N_DN_LAYERS * VALUE_HEADS * ds * F32_BYTES
    state = N_DN_LAYERS * VALUE_HEADS * ds * F32_BYTES
    meta = N_DN_LAYERS * 40
    q_code = int(dnr.qkvz_subblock_parts(geometry())["q"]["payload_bytes"])
    k_code = int(dnr.qkvz_subblock_parts(geometry())["k"]["payload_bytes"])
    v_code = int(dnr.qkvz_subblock_parts(geometry())["v"]["payload_bytes"])
    # Header stays with the fused tensor; z payload stays.
    qkv_payload = q_code + k_code + v_code
    return {
        "d_state": ds,
        "in_proj_rows": in_rows,
        "in_proj": in_proj,
        "a_diag": a_diag,
        "state": state,
        "metadata": meta,
        "removed_qkv_payload": qkv_payload,
        "removed_ba": BA_ACTIVE_TARGET,
        "removed_conv": CONV_ACTIVE_TARGET,
        "removed_a_log": A_LOG_BYTES,
        "removed_dt_bias": DT_BIAS_BYTES,
        "removed_rec_state": REC_STATE_RESIDENT,
        "removed_conv_state": CONV_STATE_RESIDENT,
        "kept_z_and_out": True,
    }


def structured_state_bytes(
    *,
    kind: str,
    rank: int = NAMED_DPLR_RANK,
    bandwidth: int = NAMED_BANDWIDTH,
) -> dict[str, Any]:
    h = VALUE_HEADS
    d = KEY_HEAD_DIM
    n = N_DN_LAYERS
    if kind == "diagonal":
        elems = h * d
        stored = n * elems * F32_BYTES
    elif kind == "dplr":
        r = int(rank)
        elems = h * (d + 2 * d * r)  # diag + U + V
        stored = n * elems * F32_BYTES
    elif kind == "banded":
        b = int(bandwidth)
        per = d * (2 * b + 1) - b * (b + 1)
        elems = h * per
        stored = n * elems * F32_BYTES
    elif kind == "orthogonal_householder":
        r = int(rank)
        elems = h * d * r
        stored = n * elems * F32_BYTES
    else:
        raise StateFunctionRefuse(f"unknown structured kind {kind}")
    return {
        "kind": kind,
        "elements_per_layer": elems,
        "resident_bytes": stored,
        "rank": None if kind in {"diagonal", "banded"} else int(rank),
        "bandwidth": bandwidth if kind == "banded" else None,
    }


def shared_w_bytes() -> dict[str, Any]:
    one = q4_stored(QKVZ_ROWS, HIDDEN, n_layers=1)
    diagonals = (N_DN_LAYERS - 1) * QKVZ_ROWS * F16_BYTES
    return {
        "one_copy": one,
        "n_copies_incumbent": N_DN_LAYERS,
        "per_layer_diagonal_f16": diagonals,
        "forty_seven_copies": QKVZ_ACTIVE_TARGET - one,
    }


# ---------------------------------------------------------------------------
# Candidates.
# ---------------------------------------------------------------------------


def _cand(
    *,
    cid: str,
    name: str,
    mechanism: str,
    byte_model: str,
    bytes_removed: Mapping[str, int],
    bytes_added: Mapping[str, int],
    extra_flops: Mapping[str, Any],
    dispatch_change: Mapping[str, Any],
    physical_primitive: str,
    cheapest_falsifier: str,
    dense: str,
    dense_reason: str,
    status: str,
    index_slugs: Sequence[str],
    citations: list[dict[str, Any]] | None = None,
    measured: Mapping[str, Any] | None = None,
    note: str | None = None,
    cousin: bool = False,
    consult_index: bool = True,
    rw_bytes: Mapping[str, Any] | None = None,
    licensed: bool | None = None,
) -> dict[str, Any]:
    hits = _index_hits(index_slugs) if consult_index else []
    general = [h for h in hits if h.get("general_physical")]
    row: dict[str, Any] = {
        "id": cid,
        "name": name,
        "mechanism": mechanism,
        "byte_model": byte_model,
        "bytes_removed": dict(bytes_removed),
        "bytes_added": dict(bytes_added),
        "extra_flops": dict(extra_flops),
        "dispatch_change": dict(dispatch_change),
        "physical_primitive": physical_primitive,
        "cheapest_falsifier": cheapest_falsifier,
        "dense_rematerialization": dense,
        "dense_rematerialization_reason": dense_reason,
        "status": status,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "index_slugs": list(index_slugs),
        "index_refusals": hits,
        "index_query_modes": ["with_model", "without_model"],
        "general_physical_refusals": general,
    }
    removed_t, added_t = require_economics(row)
    row["net_bytes"] = removed_t - added_t
    row["licensed"] = bool(licensed) if licensed is not None else status == OPEN
    if citations:
        row["citations"] = citations
    if measured is not None:
        row["measured"] = _py(measured)
    if note:
        row["note"] = note
    if cousin:
        row["cousin_not_this_object"] = True
        row["index_hits_are_cousins"] = True
    if rw_bytes is not None:
        row["rw_bytes"] = dict(rw_bytes)
    if dense == REJECTED_DENSE_REMAT:
        tag = _remat_tag(True, True)
        if tag != REJECTED_DENSE_REMAT:
            raise StateFunctionRefuse(f"{cid}: expected REJECTED_DENSE_REMAT")
        row["physical_primitive"] = physical_primitive
    elif dense == DIRECT_CONSUME:
        if physical_primitive not in ATLAS_PRIMITIVES:
            raise StateFunctionRefuse(f"{cid} missing atlas primitive")
    return row


def finalize_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Refuse a catalog that is missing economics or a required field."""
    out: list[dict[str, Any]] = []
    for raw in rows:
        cid = str(raw.get("id") or "?")
        missing_fields = [f for f in REQUIRED_CANDIDATE_FIELDS if f not in raw]
        if missing_fields:
            if "bytes_removed" in missing_fields or "bytes_added" in missing_fields:
                raise MissingEconomics(cid, missing=missing_fields)
            raise StateFunctionRefuse(f"REFUSED: candidate {cid!r} missing {missing_fields}")
        require_economics(raw)
        if raw.get("evidence_class") != "STATIC_ONLY":
            raise StateFunctionRefuse(f"{cid}: evidence_class must be STATIC_ONLY")
        out.append(dict(raw))
    have = [r["id"] for r in out]
    if have != list(REQUIRED_CANDIDATE_IDS):
        raise StateFunctionRefuse(f"candidate catalog {have} != required {list(REQUIRED_CANDIDATE_IDS)}")
    return out


def candidates(
    *,
    spectrum: Mapping[str, Any] | None = None,
    consult_index: bool = True,
) -> list[dict[str, Any]]:
    geo = geometry()
    flops = incumbent_flops()
    specimen = "qwen3.8-27b sealed-3.14 DeltaNet gated-delta S"
    nns019 = _nns_cite("NNS-019", this_object=specimen)
    nns029 = _nns_cite("NNS-029", this_object=specimen)
    nns016 = _nns_cite("NNS-016", this_object=specimen)
    nns015 = _nns_cite("NNS-015", this_object=specimen)
    qn_state = _qn_cite(
        "QN-STATE-MERGING",
        "depth-state and KV merging measured negative on this Qwen under the tested conditions",
        "a state topology (recurrent, latent-attention, or a longer-context regime) where merged state preserves capability",
        organ="kv_state+deltanet_state",
    )
    qn_shared = _qn_cite(
        "QN-SHARED-BASIS-DENSITY",
        "no K below ~2.25 bpw composes coherently for the MLP: the local functional probe dies at held-out activation",
        "a shared-basis point that is coherent at held-out activation AND beats q2f on both density and COMPLETE_TOKEN_NS",
        organ="mlp_gate_up+mlp_down",
    )
    qn_head = _qn_cite(
        "QN-HEAD-REDUNDANCY",
        "Q heads mean cosine 0.0438 and K/V/O similarly near-orthogonal, so there is no shared-head structure to exploit",
        "an organ or model where head cosine similarity is high enough that sharing costs less capability than the bits it saves",
        organ="gqa_attention",
    )
    local = _require_primitive("LocalStateMachine")
    tiled = _require_primitive("TiledProjection")
    fused = _require_primitive("FusedDecodeCompute")
    cond = _require_primitive("ConditionalPhysicalProgram")
    persist = _require_primitive("PersistentPhysicalRegion")

    spec = spectrum if spectrum is not None else synthetic_update_spectrum()
    e16 = ((spec.get("energy_at_rank") or {}).get("16") or {}).get("mean")
    fills = bool(spec.get("update_fills_rank"))

    narrow = narrow_mixer_bytes(NAMED_NARROW_DIM)
    rec_now = REC_STATE_RESIDENT
    conv_now = CONV_STATE_RESIDENT
    removed_narrow = removed(
        catalog_weights=(
            QKVZ_ACTIVE_TARGET
            + OUT_ACTIVE_TARGET
            + CONV_ACTIVE_TARGET
        ),
        state=rec_now + conv_now,
    )
    added_narrow = added(
        generator=int(narrow["qkvz"]) + int(narrow["out"]) + int(narrow["conv1d"]),
        state=int(narrow["rec_state"]) + int(narrow["conv_state"]),
        metadata=N_DN_LAYERS * 40,
    )
    delta_now = int(flops["gated_delta_per_token"])
    delta_narrow = gated_delta_flops_per_token(
        key_dim=NAMED_NARROW_DIM, value_dim=NAMED_NARROW_DIM
    )
    qkvz_now = int(flops["qkvz_gemv_per_token"])
    qkvz_narrow = gemv_flops(int(narrow["qkvz_rows"]), HIDDEN, n_layers=N_DN_LAYERS)
    out_now = int(flops["out_gemv_per_token"])
    out_narrow = gemv_flops(HIDDEN, VALUE_HEADS * NAMED_NARROW_DIM, n_layers=N_DN_LAYERS)

    learned = learned_recurrence_bytes(NAMED_DSTATE)
    removed_learned = removed(
        catalog_weights=(
            int(learned["removed_qkv_payload"])
            + int(learned["removed_ba"])
            + int(learned["removed_conv"])
            + int(learned["removed_a_log"])
            + int(learned["removed_dt_bias"])
        ),
        state=int(learned["removed_rec_state"]) + int(learned["removed_conv_state"]),
    )
    added_learned = added(
        generator=int(learned["in_proj"]) + int(learned["a_diag"]),
        state=int(learned["state"]),
        metadata=int(learned["metadata"]),
    )
    ssm_flops = N_DN_LAYERS * VALUE_HEADS * NAMED_DSTATE * 6
    in_proj_flops = gemv_flops(int(learned["in_proj_rows"]), HIDDEN, n_layers=N_DN_LAYERS)
    qkv_flops_dropped = gemv_flops(
        int(geo["q_rows"]) + int(geo["k_rows"]) + int(geo["v_rows"]),
        HIDDEN,
        n_layers=N_DN_LAYERS,
    )
    ba_flops = int(flops["ba_gemv_per_token"])

    dplr = structured_state_bytes(kind="dplr", rank=NAMED_DPLR_RANK)
    diag = structured_state_bytes(kind="diagonal")
    banded = structured_state_bytes(kind="banded", bandwidth=NAMED_BANDWIDTH)
    orth = structured_state_bytes(kind="orthogonal_householder", rank=NAMED_DPLR_RANK)
    # Listed economics: DPLR store of S (W unchanged). Generator/residuals/embeddings = 0.
    removed_struct = removed(state=rec_now)
    added_struct = added(state=int(dplr["resident_bytes"]), metadata=N_DN_LAYERS * 40)

    # Shared generator of coefficients: skinny T then per-layer diagonal. No W.
    t1 = q4_stored(256, HIDDEN, n_layers=1)
    t2 = q4_stored(QKVZ_ROWS, 256, n_layers=1)
    adapters = (N_DN_LAYERS) * QKVZ_ROWS * F16_BYTES
    removed_gen = removed(catalog_weights=QKVZ_ACTIVE_TARGET)
    added_gen = added(
        generator=t1 + t2,
        residuals=adapters,
        embeddings=N_DN_LAYERS * 256 * F32_BYTES,
        metadata=N_DN_LAYERS * 40,
    )
    skinny_flops = (
        gemv_flops(256, HIDDEN, n_layers=N_DN_LAYERS)
        + gemv_flops(QKVZ_ROWS, 256, n_layers=N_DN_LAYERS)
        + N_DN_LAYERS * QKVZ_ROWS  # diagonal
    )

    shared = shared_w_bytes()
    removed_share = removed(catalog_weights=QKVZ_ACTIVE_TARGET)
    added_share = added(
        generator=int(shared["one_copy"]),
        residuals=int(shared["per_layer_diagonal_f16"]),
        metadata=N_DN_LAYERS * 40,
    )

    q3_all = int(dnr.qkvz_subblock_parts(geo)["q"]["code_bytes"]) // 4
    q3_all += int(dnr.qkvz_subblock_parts(geo)["k"]["code_bytes"]) // 4
    q3_all += int(dnr.qkvz_subblock_parts(geo)["v"]["code_bytes"]) // 4
    q3_all += int(dnr.qkvz_subblock_parts(geo)["z"]["code_bytes"]) // 4

    rows: list[dict[str, Any]] = [
        _cand(
            cid="smaller_state_machine",
            name="narrower (d=64) state; rank-r store of 128x128 is not closed",
            mechanism=(
                "Incumbent S is 128x128 f32 per value head, updated by a rank-1 "
                "projector plus write. Rank-r factorisation of that S is not "
                "invariant under the write: rank grows with tokens unless each "
                "step truncates (a different function). A closed reduction is "
                "narrower key/value dim at the same head counts, which also "
                "shrinks qkvz, out, conv, and S. Named point d=64."
            ),
            byte_model=(
                f"Remove qkvz {QKVZ_ACTIVE_TARGET} + out {OUT_ACTIVE_TARGET} + "
                f"conv {CONV_ACTIVE_TARGET} + S {rec_now} + conv_state {conv_now}. "
                f"Add d=64 qkvz {narrow['qkvz']} + out {narrow['out']} + conv "
                f"{narrow['conv1d']} + S {narrow['rec_state']} + conv_state "
                f"{narrow['conv_state']} + {N_DN_LAYERS * 40} headers. ba unchanged. "
                f"A rank-{NAMED_RANK} factor of the present 128x128 (U,V resident "
                f"{2 * N_DN_LAYERS * VALUE_HEADS * KEY_HEAD_DIM * NAMED_RANK * F32_BYTES}) "
                "is listed only as a non-closed alternative: it requires per-step "
                "truncation."
            ),
            bytes_removed=removed_narrow,
            bytes_added=added_narrow,
            extra_flops=extra_flops_row(
                (delta_narrow - delta_now) + (qkvz_narrow - qkvz_now) + (out_narrow - out_now),
                formula=(
                    "gated_delta(d=64)-gated_delta(d=128) + qkvz_gemv(8192x5120)-"
                    "qkvz_gemv(16384x5120) + out_gemv(5120x3072)-out_gemv(5120x6144)"
                ),
                relative_to="incumbent_gated_delta_plus_qkvz_plus_out",
            ),
            dispatch_change=dispatch_row(
                GATED_DELTA_LAUNCHES,
                GATED_DELTA_LAUNCHES,
                surface="same LocalStateMachine, smaller tiles; 48 launches remain",
            ),
            physical_primitive=local,
            cheapest_falsifier=(
                "STATIC, run here: isotropic rank-1 rollout, "
                f"{SPECTRUM_TOKENS} tokens, {SPECTRUM_HEADS} heads. Energy at r=16 "
                f"mean={e16}; update_fills_rank={fills}. A rank-r store of this "
                "S is not a compression of the mixer. CHEAP next: SVD of S on "
                "real post-norm x after a long prompt; if energy at r=16 is still "
                "below 0.99, d=64 is function replacement and needs a generate "
                "gate, not a packing. Do not quote the MLP NNS-016 spectrum as "
                "this S. Do not rematerialize a dense 128x128 from factors."
            ),
            dense=DIRECT_CONSUME,
            dense_reason=(
                "A native smaller-d vi_simd consumes S in-register. Expanding a "
                "rank-r factor to dense 128x128 then running incumbent vi_simd "
                "is REJECTED_DENSE_REMAT and eliminates zero active bytes of S."
            ),
            status=OPEN,
            index_slugs=["state_merging", "low_rank"],
            citations=[nns016, qn_state],
            measured=spec,
            note=(
                "NNS-016 is the MLP Kronecker/low-rank cousin, not this S. "
                "QN-STATE-MERGING is depth-merge of the present S, a different "
                "candidate (share_or_merge_state_across_depth)."
            ),
            cousin=True,
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="learned_recurrence",
            name="diagonal SSM of size 16 per value head, Q4 in_proj emits B/C/dt",
            mechanism=(
                "Stop generating a rank-1 projector from W_qkvz. Store a per-head "
                "diagonal A (d_state=16) and emit B, C, dt from a skinny Q4 map "
                "of x. This is function replacement of gated-delta, not a codec "
                "of S or of W. z and out_proj stay (output gate / residual). "
                "Cousin of NNS-015 (distilled MLP operator), not a retry."
            ),
            byte_model=(
                f"Remove q/k/v payload {learned['removed_qkv_payload']} + ba "
                f"{BA_ACTIVE_TARGET} + conv {CONV_ACTIVE_TARGET} + A_log "
                f"{A_LOG_BYTES} + dt_bias {DT_BIAS_BYTES} + S {REC_STATE_RESIDENT} "
                f"+ conv_state {CONV_STATE_RESIDENT}. Add in_proj "
                f"{learned['in_proj']} + A_diag {learned['a_diag']} + state "
                f"{learned['state']} + headers {learned['metadata']}. z payload "
                "and out_proj are kept."
            ),
            bytes_removed=removed_learned,
            bytes_added=added_learned,
            extra_flops=extra_flops_row(
                in_proj_flops + ssm_flops - qkv_flops_dropped - ba_flops - delta_now,
                formula=(
                    "in_proj_gemv(2304x5120)+ssm(6*d_state*heads*layers)"
                    "-qkv_gemv-ba_gemv-gated_delta"
                ),
                relative_to="incumbent_qkv_ba_delta",
            ),
            dispatch_change=dispatch_row(
                GATED_DELTA_LAUNCHES + BA_TO_DECAY_LAUNCHES + 48,  # delta+ba+rearrange
                48,
                surface="one LocalStateMachine per DN layer for the SSM step",
            ),
            physical_primitive=local,
            cheapest_falsifier=(
                "This is an architecture change. CHEAP CPU: replace one layer's "
                "gated-delta with a fit diagonal SSM on real post-norm x, report "
                "h cosine vs incumbent and next-token S trajectory. If cosine "
                "fails 0.99, stop before a generate. Do not launder NNS-015 "
                "(MLP distillation, not run) as this mixer. Do not emit W_qkvz "
                "from the SSM and then Q4 GEMV."
            ),
            dense=DIRECT_CONSUME,
            dense_reason=(
                "A native diagonal SSM consumes A, B, C in-register. "
                "Materializing a dense 128x128 S from the SSM every token is "
                "REJECTED_DENSE_REMAT."
            ),
            status=OPEN,
            index_slugs=["low_rank", "qn_state_merging"],
            citations=[nns015, qn_state],
            cousin=True,
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="structured_transition",
            name="store S as DPLR / diagonal / banded / Householder generators",
            mechanism=(
                "The incumbent transition is already structured in time: scalar "
                "decay times S, then a rank-1 Householder (I - beta k k^T), then "
                "a rank-1 write k v^T. There is no stored A. Storing a structured "
                "A (diagonal, DPLR, banded, orthogonal generators) or storing S "
                "itself in that form is a different mixer, not a packing of the "
                "present 128x128. Listed economics store S as DPLR rank-8 "
                "(diag + U,V) and keep W_qkvz."
            ),
            byte_model=(
                f"Remove dense S {rec_now}. Add DPLR S {dplr['resident_bytes']} + "
                f"headers {N_DN_LAYERS * 40}. Alternatives: diagonal S "
                f"{diag['resident_bytes']}; banded b={NAMED_BANDWIDTH} "
                f"{banded['resident_bytes']}; {NAMED_DPLR_RANK} Householder "
                f"generators {orth['resident_bytes']}. W_qkvz unchanged. "
                "Incumbent already IS DPLR-in-time; a stored A drops the "
                "data-dependent k."
            ),
            bytes_removed=removed_struct,
            bytes_added=added_struct,
            extra_flops=extra_flops_row(
                N_DN_LAYERS * VALUE_HEADS * (
                    KEY_HEAD_DIM  # diag * S
                    + 2 * KEY_HEAD_DIM * NAMED_DPLR_RANK  # U,V products
                    + 2 * KEY_HEAD_DIM * VALUE_HEAD_DIM  # still a write/readout on factors
                ) - delta_now,
                formula="DPLR apply + readout vs incumbent dense vi_simd",
                relative_to="incumbent_gated_delta_per_token",
            ),
            dispatch_change=dispatch_row(
                GATED_DELTA_LAUNCHES,
                GATED_DELTA_LAUNCHES,
                surface="replacement LocalStateMachine, same 48 launches",
            ),
            physical_primitive=local,
            cheapest_falsifier=(
                "STATIC: the update is already rank-1 + decay; a stored structured "
                "A is a different operator. CHEAP CPU: replace S with its DPLR "
                "rank-8 SVD truncation after a real prompt and continue the "
                "incumbent update; if subsequent h cosine falls below 0.99, "
                "truncation is not a compression of this mixer. Do not quote "
                "Mamba's stored A as this generated Householder."
            ),
            dense=DIRECT_CONSUME,
            dense_reason=(
                "A DPLR/diagonal/banded kernel consumes the factors. Expanding "
                "them to dense 128x128 then running vi_simd is REJECTED_DENSE_REMAT."
            ),
            status=OPEN,
            index_slugs=["low_rank", "kronecker", "state_merging"],
            citations=[nns016],
            measured={
                "incumbent_already_rank1_plus_decay": True,
                "diagonal_resident": diag["resident_bytes"],
                "dplr_resident": dplr["resident_bytes"],
                "banded_resident": banded["resident_bytes"],
                "householder_resident": orth["resident_bytes"],
            },
            cousin=True,
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="generated_transition_coefficients",
            name="skinny program emits qkvz activations, never W",
            mechanism=(
                "Store a shared skinny map T2 T1 (5120→256→16384) plus per-layer "
                "f16 diagonals that emit the fused qkvz activation, then the "
                "incumbent rearrange + gated-delta. The generator IS the matvec. "
                "Emitting W and then running qwen_uniform_q4 is a different "
                "candidate (emit_w_then_ordinary_gemv) and is REJECTED_DENSE_REMAT. "
                "Scale-field sharing across layers is already MEASURED_NEGATIVE; "
                "W itself is UNMEASURED — this is that unmeasured W, as a program."
            ),
            byte_model=(
                f"Remove 48 qkvz {QKVZ_ACTIVE_TARGET}. Add T1 {t1} + T2 {t2} + "
                f"48 f16 diagonals {adapters} + 48 layer embeddings "
                f"{N_DN_LAYERS * 256 * F32_BYTES} + headers {N_DN_LAYERS * 40}."
            ),
            bytes_removed=removed_gen,
            bytes_added=added_gen,
            extra_flops=extra_flops_row(
                skinny_flops - qkvz_now,
                formula="48*(GEMV 256x5120 + GEMV 16384x256 + 16384 diagonal) - qkvz GEMV",
                relative_to="incumbent_qkvz_gemv_per_token",
            ),
            dispatch_change=dispatch_row(
                48,
                96,
                surface=(
                    "two skinny GEMVs per DN layer unless FusedDecodeCompute "
                    "folds them; +48 launches if split"
                ),
            ),
            physical_primitive=tiled,
            cheapest_falsifier=(
                "STATIC, already run on the scale field (DELTANET_REPRESENTATION "
                "shared_transforms_across_layers, consecutive-layer Pearson ~0). "
                "That kills the identity residual on scales, not this skinny "
                "program on W. CHEAP CPU: fit T1,T2 on one reconstructed qkvz "
                "and report rec_out cosine vs incumbent Q4; if it fails 0.99, "
                "the program is not this function. A lowering whose native "
                "execution is 'emit W, then qwen_uniform_q4' is refused before "
                "a fit."
            ),
            dense=DIRECT_CONSUME,
            dense_reason=(
                "y = diag_l T2 (T1 x) is two skinny GEMVs. Materializing "
                "W_l = diag_l T2 T1 then Q4 is REJECTED_DENSE_REMAT."
            ),
            status=OPEN,
            index_slugs=["generated_tied_params", "cross_expert_structure", "shared_basis"],
            citations=[qn_shared, qn_head],
            cousin=True,
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="conditional_recurrence",
            name="skip the S update when incumbent beta is near 0",
            mechanism=(
                "Incumbent vi_simd reads and writes every element of S every "
                "token. beta is already computed (sigmoid of ba). Predicating "
                "the rank-1 write (and optionally the decay) on beta > ε uses "
                "ConditionalPhysicalProgram on the existing coefficient. No new "
                "gate weights. Skip rate is UNMEASURED on real x. Skipping "
                "whole layers is share_or_merge_state_across_depth, already "
                "falsified (QN-STATE-MERGING)."
            ),
            byte_model=(
                "Catalog bytes unchanged (0 removed, 0 added). RW traffic of S "
                f"is {REC_STATE_RW_PER_TOKEN}/token; a skip fraction p saves "
                f"p*{REC_STATE_RW_PER_TOKEN} RW and p*{delta_now} flops. "
                "Threshold ε is a scalar, not a tensor. Layer-skip would remove "
                "47/48 of S and is QN-STATE-MERGING, not this candidate."
            ),
            bytes_removed=removed(),
            bytes_added=added(),
            extra_flops=extra_flops_row(
                N_DN_LAYERS * VALUE_HEADS,  # compares; savings unknown until p is measured
                formula="48*48 compares per token; flop savings = p * gated_delta, p UNMEASURED",
                relative_to="incumbent_gated_delta_per_token",
            ),
            dispatch_change=dispatch_row(
                GATED_DELTA_LAUNCHES,
                GATED_DELTA_LAUNCHES,
                surface="same 48 launches; predication inside vi_simd, not a skip kernel",
            ),
            physical_primitive=cond,
            cheapest_falsifier=(
                "CHEAP CPU: histogram of beta on a real prompt at several layers. "
                "If P(beta < 0.05) ≈ 0, the skip never fires and the candidate "
                "dies without a kernel. Do not retry QN-STATE-MERGING as a "
                "per-token skip. Do not add a learned skip head that writes a "
                "dense W."
            ),
            dense=DIRECT_CONSUME,
            dense_reason="Predicated vi_simd. No W is written.",
            status=OPEN,
            index_slugs=["state_merging", "qn_state_merging"],
            citations=[qn_state],
            rw_bytes={
                "incumbent_s_rw_per_token": REC_STATE_RW_PER_TOKEN,
                "saved_at_skip_fraction_p": "p * 301989888",
                "skip_fraction": UNMEASURED,
            },
            cousin=True,
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="shared_transforms_w_unmeasured",
            name="one shared W_qkvz plus per-layer f16 diagonals",
            mechanism=(
                "48 independent qkvz maps, 44.6 MB each. Share one W and a "
                "per-layer diagonal on the 16384 fused rows. The Q4 scale field "
                "is already MEASURED_NEGATIVE for unconditioned sharing "
                "(DELTANET_REPRESENTATION: consecutive-layer Pearson of per-row "
                "mean scales ~0). W itself is UNMEASURED on this organ. Cousin "
                "of QN-SHARED-BASIS-DENSITY (MLP) and QN-HEAD-REDUNDANCY (GQA), "
                "not a DN-head cosine."
            ),
            byte_model=(
                f"Remove 48 qkvz {QKVZ_ACTIVE_TARGET}. Add one copy "
                f"{shared['one_copy']} + 47 f16 diagonals "
                f"{shared['per_layer_diagonal_f16']} + headers {N_DN_LAYERS * 40}."
            ),
            bytes_removed=removed_share,
            bytes_added=added_share,
            extra_flops=extra_flops_row(
                N_DN_LAYERS * QKVZ_ROWS,
                formula="same qkvz GEMV plus 16384 diagonal muls per DN layer",
                relative_to="incumbent_qkvz_gemv_per_token",
            ),
            dispatch_change=dispatch_row(
                48,
                48,
                surface="same 48 fused GEMVs; diagonal folds into FusedDecodeCompute",
            ),
            physical_primitive=fused,
            cheapest_falsifier=(
                "STATIC, already run on scales: sharing is not in the scale "
                "field. CHEAP CPU: cosine of reconstructed W_qkvz between "
                "consecutive DN layers. If mean cosine is ~0 like GQA heads "
                "(QN-HEAD-REDUNDANCY 0.0438), shared W dies without a generate. "
                "Do not quote the scale Pearson as a W cosine. Do not materialize "
                "W_l = diag_l W_shared as production Q4."
            ),
            dense=DIRECT_CONSUME,
            dense_reason=(
                "y = diag_l (W_shared x) is one GEMV plus a diagonal. "
                "Materializing W_l then Q4 is REJECTED_DENSE_REMAT."
            ),
            status=OPEN,
            index_slugs=["shared_basis", "head_sharing", "cross_expert_structure"],
            citations=[qn_shared, qn_head],
            measured={
                "scale_field_sharing": MEASURED_NEGATIVE,
                "scale_field_source": PREDECESSOR_REPR,
                "W_sharing": UNMEASURED,
            },
            cousin=True,
            consult_index=consult_index,
            licensed=False,
            note=(
                "Status OPEN is the W question. The scale-field residual is "
                "already MEASURED_NEGATIVE and is not re-derived here."
            ),
        ),
        _cand(
            cid="fused_update_consume",
            name="update+consume of S is already one kernel; FUSE_BA_DELTA is in tree",
            mechanism=(
                "qwen38_gated_delta_decode_vi_simd already applies the rank-1 "
                "update and the readout h = S^T q in one LocalStateMachine; S "
                "must still be stored for the next token (O(1) recurrence). "
                "HAWKING_QWEN38_FUSE_BA_DELTA=1 folds ba_to_decay into that "
                "kernel (48 launches, token-identical, default Off). A further "
                "PersistentPhysicalRegion over the 7-launch DN sequence is "
                "DISPATCH_MOTIFS dn_layer_state_machine (337→48), zero catalog "
                "bytes. Not a 2.96 GB lever."
            ),
            byte_model=(
                "Zero catalog bytes removed or added. A_log and dt_bias (200 B "
                "each per layer) are still read. Decay/beta workspace ceases to "
                "be a stored round-trip under FUSE_BA_DELTA. Full DN region "
                "removes launches, not W or S."
            ),
            bytes_removed=removed(),
            bytes_added=added(),
            extra_flops=extra_flops_row(
                0,
                formula="same arithmetic; ba_to_decay folds into gated-delta",
                relative_to="incumbent_gated_delta_per_token",
            ),
            dispatch_change=dispatch_row(
                SEALED_DISPATCHES,
                FUSE_BA_DELTA_DISPATCHES,
                surface="HAWKING_QWEN38_FUSE_BA_DELTA=1; 628→580, 48 DN launches. Cited from BA_DELTA_AB, not re-measured.",
            ),
            physical_primitive=persist,
            cheapest_falsifier=(
                "Already run: receipts/future/BA_DELTA_AB.json. 628→580 "
                "dispatches, token ids identical, zero fallbacks. STATIC_ONLY "
                "here: cited, not re-measured. Enabling the flag in the sealed "
                "profile is an ops act. A 7-kernel DN region is DISPATCH_MOTIFS "
                "YES_STATE_MACHINE; its inner cut is this lever. Do not retarget "
                "628→500 as a byte plan."
            ),
            dense=DIRECT_CONSUME,
            dense_reason="Same arithmetic, one kernel, no W.",
            status=EXISTING_LEVER,
            index_slugs=["resident_state", "megakernel"],
            citations=[
                {
                    "scar_id": "BA_DELTA_AB",
                    "source_path": BA_DELTA_REL,
                    "claim_refuted": "",
                    "reopen_condition": "",
                    "surface": "HAWKING_QWEN38_FUSE_BA_DELTA=1 on sealed-3.14",
                    "kind": "EXISTING_LEVER",
                    "this_specimen": specimen,
                },
                {
                    "scar_id": "dn_layer_state_machine",
                    "source_path": DISPATCH_MOTIFS_REL,
                    "claim_refuted": "",
                    "reopen_condition": "",
                    "surface": "sealed-3.14 337 DN launches",
                    "kind": "EXISTING_JUDGMENT",
                    "this_specimen": specimen,
                },
            ],
            measured={
                "vi_simd_already_update_and_readout": True,
                "fuse_ba_delta_dispatches_after": FUSE_BA_DELTA_DISPATCHES,
                "token_ids_identical_cited": True,
                "catalog_bytes_eliminated": 0,
            },
            consult_index=consult_index,
            licensed=True,
        ),
        _cand(
            cid="share_or_merge_state_across_depth",
            name="one S (or tied S) across several of the 48 DN layers",
            mechanism=(
                "Share or merge rec_state across DeltaNet depth, or share KV "
                "across GQA depth. Cuts resident S, not W_qkvz."
            ),
            byte_model=(
                f"Remove 47/48 of S ({rec_now - rec_now // 48}). Add nothing if "
                "the remaining slot is reused. Not a packing of W."
            ),
            bytes_removed=removed(state=rec_now - rec_now // 48),
            bytes_added=added(),
            extra_flops=extra_flops_row(
                0,
                formula="same vi_simd, fewer resident slots",
                relative_to="incumbent_gated_delta_per_token",
            ),
            dispatch_change=dispatch_row(
                GATED_DELTA_LAUNCHES,
                GATED_DELTA_LAUNCHES,
                surface="same kernel, different state slot",
            ),
            physical_primitive=local,
            cheapest_falsifier=(
                "Already run: QN-STATE-MERGING on this parent "
                "(kv_state+deltanet_state). Reopen is a different state topology "
                "or a longer-context regime, not a retry of shared S across "
                "these 48 layers. Cited, not re-derived."
            ),
            dense=DIRECT_CONSUME,
            dense_reason="One S buffer, same kernel, different slot.",
            status=ALREADY_FALSIFIED,
            index_slugs=["state_merging", "qn_state_merging"],
            citations=[qn_state],
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="emit_w_then_ordinary_gemv",
            name="generate W_qkvz then run incumbent Q4 GEMV",
            mechanism=(
                "A compact program emits the 16384x5120 matrix, which is then "
                "quantized and consumed by qwen_uniform_q4. That is dense "
                "rematerialization of W. The no-W generator is "
                "generated_transition_coefficients."
            ),
            byte_model=(
                f"Would claim to remove {QKVZ_ACTIVE_TARGET} and add |θ|. The "
                "production path writes W (at least the incumbent 44.6 MB/layer) "
                "before the GEMV, so active bytes of W are not eliminated."
            ),
            bytes_removed=removed(catalog_weights=QKVZ_ACTIVE_TARGET),
            bytes_added=added(generator=QKVZ_ACTIVE_TARGET),
            extra_flops=extra_flops_row(
                gemv_flops(QKVZ_ROWS, HIDDEN, n_layers=N_DN_LAYERS),
                formula="generate W plus the incumbent qkvz GEMV; not a save",
                relative_to="incumbent_qkvz_gemv_per_token",
            ),
            dispatch_change=dispatch_row(
                48,
                96,
                surface="emit + GEMV is two operations per layer",
            ),
            physical_primitive=fused,
            cheapest_falsifier=(
                "STATIC: any plan whose native_execution_concept is 'emit W, "
                "then qwen_uniform_q4' is REJECTED_DENSE_REMAT before a fit. "
                "Predecessor generated_coefficients already recorded this. "
                "Not re-derived."
            ),
            dense=REJECTED_DENSE_REMAT,
            dense_reason=(
                "Production NX decompresses to a dense weight tensor and then "
                "runs ordinary kernels. Verification MAY reconstruct; production "
                "must not."
            ),
            status=ALREADY_FALSIFIED,
            index_slugs=["generated_tied_params"],
            citations=[
                {
                    "scar_id": "REJECTED_DENSE_REMAT",
                    "source_path": "tools/future/ebpw_categories.py",
                    "claim_refuted": (
                        "That generating W and running ordinary GEMV eliminates "
                        "active bytes of W"
                    ),
                    "reopen_condition": (
                        "A generator that IS the matvec (no W). That is "
                        "generated_transition_coefficients, a different candidate."
                    ),
                    "surface": "production NX",
                    "kind": "CATEGORY_ERROR",
                    "this_specimen": specimen,
                }
            ],
            consult_index=consult_index,
            licensed=False,
        ),
        _cand(
            cid="qkvz_bit_descent",
            name="heterogeneous or uniform bit-descent of q/k/v/z (not this question)",
            mechanism=(
                "Treat DeltaNet as four matrices and spend fewer bits. Codes "
                "are at their entropy floor (H(q)≈3.47 of 4, spread <0.014) and "
                "Q3 injures rec_state / gated cosine. Recorded so this module "
                "does not fall back to quantization."
            ),
            byte_model=(
                f"Uniform Q3 of qkvz codes would remove {q3_all} and add 0. "
                "DELTANET_QKVZ_PRECISION licensed 0 bytes (any_supported false)."
            ),
            bytes_removed=removed(catalog_weights=q3_all),
            bytes_added=added(),
            extra_flops=extra_flops_row(
                0,
                formula="same GEMV, narrower codes; not licensed",
                relative_to="incumbent_qkvz_gemv_per_token",
            ),
            dispatch_change=dispatch_row(
                48,
                48,
                surface="row-range Q4/Q3 GEMV, same 48 launches",
            ),
            physical_primitive=fused,
            cheapest_falsifier=(
                "Already run: DELTANET_QKVZ_PRECISION.json. H(q) 3.47 of 4 on "
                "every block; any_supported false; k/v rewrite S at ~0.2 rel-fro. "
                "NNS-019 (Gravity on DN in/out) and NNS-029 (uniform bit-descent) "
                "cited as cousins. Not re-derived."
            ),
            dense=DIRECT_CONSUME,
            dense_reason="Native row-range GEMV. Not this module's question.",
            status=MEASURED_NEGATIVE,
            index_slugs=["uniform_q3", "uniform_subbit_allocation"],
            citations=[nns019, nns029],
            measured={
                "predecessor": PREDECESSOR_QKVZ,
                "any_supported": False,
                "H_q_bits_floor": 3.47,
                "licensed_bytes_removed": 0,
            },
            consult_index=consult_index,
            licensed=False,
            cousin=True,
        ),
    ]
    return finalize_candidates(rows)


def answers(
    inc: Mapping[str, Any],
    spec: Mapping[str, Any],
    cands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cands}
    e16 = ((spec.get("energy_at_rank") or {}).get("16") or {}).get("mean")
    return {
        "what_update_does_the_model_need": {
            "answer": (
                "A recurrent map (q, k, v, decay, beta, S) → (S', h) that is "
                "O(1) in sequence length. The incumbent parameterises that map "
                "as a scalar decay, a rank-1 Householder along k, a rank-1 write "
                "k v^T, and a readout S^T q. That is a parameterisation, not a "
                "requirement. Function preservation is the bar, not reconstructing "
                "W_qkvz. z never enters the map. Codes of q/k/v/z are at their "
                "entropy floor; more quantization is the wrong lever."
            ),
            "operator": inc["operator"]["operator"],
            "parameterisation_not_requirement": True,
            "z_enters_gated_delta": False,
        },
        "is_state_larger_than_the_function": {
            "answer": (
                "The update can fill 128 dimensions: a rank-r store is not closed "
                "under the rank-1 write, and a synthetic isotropic rollout does "
                f"not concentrate 99% of energy at r=16 (mean={e16}). Whether the "
                "trained function uses those 128 directions is UNMEASURED on real "
                "x. A closed reduction is narrower d, which is a different mixer "
                "and also shrinks W. Status OPEN, not licensed."
            ),
            "status": by_id["smaller_state_machine"]["status"],
            "update_fills_rank": spec.get("update_fills_rank"),
            "trained_function_rank": UNMEASURED,
            "net_bytes_if_d64": by_id["smaller_state_machine"]["net_bytes"],
        },
        "can_recurrence_be_learned_with_fewer_params": {
            "answer": (
                "Yes as a named function replacement (diagonal SSM d_state=16 + "
                "skinny in_proj), not as a packing of S. Cousin of NNS-015 "
                "(MLP distillation, still open there), not a retry. Capability "
                "UNMEASURED. Emit-W lowering is dead."
            ),
            "status": by_id["learned_recurrence"]["status"],
            "net_bytes_if_true": by_id["learned_recurrence"]["net_bytes"],
        },
        "do_structured_transitions_fit": {
            "answer": (
                "The incumbent already IS structured in time (decay + rank-1 "
                "Householder + rank-1 write). A stored diagonal/DPLR/banded/"
                "orthogonal A is a different mixer. Listed DPLR store of S keeps "
                "W and is OPEN as function replacement, not compression."
            ),
            "status": by_id["structured_transition"]["status"],
            "incumbent_already_rank1_plus_decay": True,
            "net_bytes_if_dplr_s": by_id["structured_transition"]["net_bytes"],
        },
        "can_coefficients_be_generated": {
            "answer": (
                "A no-W skinny program is OPEN and has executable economics. "
                "Emit-W-then-Q4 is REJECTED_DENSE_REMAT (already recorded). "
                "Scale-field sharing is MEASURED_NEGATIVE; this program is the "
                "UNMEASURED W question, not a retry of the scale Pearson."
            ),
            "status": by_id["generated_transition_coefficients"]["status"],
            "emit_w_status": by_id["emit_w_then_ordinary_gemv"]["status"],
            "net_bytes_if_true": by_id["generated_transition_coefficients"]["net_bytes"],
        },
        "does_every_token_need_the_full_update": {
            "answer": (
                "The kernel always updates every element. Whether beta is ever "
                "near 0 on real x is UNMEASURED. Per-token predication adds 0 "
                "catalog bytes. Skipping whole layers is QN-STATE-MERGING, "
                "ALREADY_FALSIFIED."
            ),
            "status": by_id["conditional_recurrence"]["status"],
            "skip_fraction": UNMEASURED,
            "layer_merge_status": by_id["share_or_merge_state_across_depth"]["status"],
        },
        "can_W_be_shared_across_layers": {
            "answer": (
                "Not from the Q4 scale field (already MEASURED_NEGATIVE). W "
                "itself is UNMEASURED on this organ. Cousin of QN-SHARED-BASIS "
                "(MLP) and QN-HEAD-REDUNDANCY (GQA). A shared copy plus diagonals "
                "has executable economics; capability is the missing measurement."
            ),
            "status": by_id["shared_transforms_w_unmeasured"]["status"],
            "scale_field": MEASURED_NEGATIVE,
            "W": UNMEASURED,
            "net_bytes_if_true": by_id["shared_transforms_w_unmeasured"]["net_bytes"],
        },
        "is_fused_update_consume_already_one_op": {
            "answer": (
                "Yes for S: vi_simd updates S and consumes it as h = S^T q in "
                "one kernel. FUSE_BA_DELTA additionally folds ba_to_decay, "
                "token-identical, 628→580, zero catalog bytes (EXISTING_LEVER). "
                "A 7-launch DN PersistentPhysicalRegion is a dispatch judgment, "
                "not a 2.96 GB lever."
            ),
            "status": by_id["fused_update_consume"]["status"],
            "catalog_bytes_eliminated": 0,
        },
        "what_is_already_falsified": {
            "answer": (
                "Depth-merge of S (QN-STATE-MERGING). Emit-W-then-GEMV "
                "(REJECTED_DENSE_REMAT). q/k/v/z bit-descent "
                "(DELTANET_QKVZ_PRECISION, NNS-019, NNS-029). Scale-field "
                "sharing (DELTANET_REPRESENTATION). Gravity families on DN "
                "in/out (NNS-019)."
            ),
            "share_or_merge": ALREADY_FALSIFIED,
            "emit_w": ALREADY_FALSIFIED,
            "qkvz_bits": MEASURED_NEGATIVE,
        },
    }


# ---------------------------------------------------------------------------
# Snapshot / receipt.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _measured() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    geo = geometry()
    bytes_row = incumbent_bytes()
    op = incumbent_operator()
    spec = synthetic_update_spectrum()
    return bytes_row, {"geometry": geo, "operator": op, "flops": incumbent_flops()}, spec


def snapshot(consult_index: bool = True) -> dict[str, Any]:
    inc_bytes, inc, spec = _measured()
    cands = candidates(spectrum=spec, consult_index=consult_index)
    return {
        "accounting": inc_bytes,
        "incumbent": inc,
        "spectrum": spec,
        "candidates": cands,
        "answers": answers({**inc, "bytes": inc_bytes}, spec, cands),
    }


def build(*, consult_index: bool = True) -> Path:
    snap = snapshot(consult_index=consult_index)
    cands = snap["candidates"]
    n_open = sum(1 for c in cands if c["status"] == OPEN)
    n_meas = sum(1 for c in cands if c["status"] == MEASURED_NEGATIVE)
    n_dead = sum(1 for c in cands if c["status"] == ALREADY_FALSIFIED)
    n_exist = sum(1 for c in cands if c["status"] == EXISTING_LEVER)
    n_remat = sum(1 for c in cands if c["dense_rematerialization"] == REJECTED_DENSE_REMAT)
    general_hits = []
    seen_g: set[str] = set()
    for c in cands:
        for h in c.get("general_physical_refusals") or []:
            sid = str(h.get("scar_id") or "")
            if sid and sid not in seen_g:
                seen_g.add(sid)
                general_hits.append(h)
    gp_probe = probe_general_physical() if consult_index else []
    for h in gp_probe:
        if h.get("refused") and h.get("scar_id"):
            sid = str(h["scar_id"])
            if sid not in seen_g:
                seen_g.add(sid)
                general_hits.append(h)
    with_model = 0
    without_model = 0
    for c in cands:
        for h in c.get("index_refusals") or []:
            if h.get("query_mode") == "with_model":
                with_model += 1
            elif h.get("query_mode") == "without_model":
                without_model += 1
    for h in gp_probe:
        if not h.get("refused"):
            continue
        if h.get("query_mode") == "with_model":
            with_model += 1
        elif h.get("query_mode") == "without_model":
            without_model += 1
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Ask what recurrent state update sealed-3.14 DeltaNet actually "
            "needs, now that its q/k/v/z codes are at their entropy floor and "
            "heterogeneous precision is MEASURED_NEGATIVE. Function "
            "preservation, not matrix reconstruction. Executable economics "
            "(bytes_removed and bytes_added) on every candidate."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "predecessor": [PREDECESSOR_REPR, PREDECESSOR_QKVZ],
        "what_this_does_not_prove": [
            "capability or generate identity of a narrower d, a diagonal SSM, or a shared W",
            "that isotropic coefficients are the trained trajectory of S",
            "physical EBPW of a different mixer",
            "actual_read_bytes_per_token (cache, contention)",
            "that FUSE_BA_DELTA's TPS delta is a hardware claim of this receipt",
        ],
        "accounting": _py(snap["accounting"]),
        "incumbent": _py(snap["incumbent"]),
        "spectrum": _py(snap["spectrum"]),
        "candidates": _py(cands),
        "answers": _py(snap["answers"]),
        "candidate_counts": {
            "n": len(cands),
            "open": n_open,
            "measured_negative": n_meas,
            "already_falsified": n_dead,
            "existing_lever": n_exist,
            "rejected_dense_remat": n_remat,
        },
        "index_consultation": {
            "modes": ["with_model", "without_model"],
            "with_model_hits": with_model,
            "without_model_hits": without_model,
            "general_physical_scar_ids": sorted(seen_g),
            "general_physical_hits": _py(general_hits),
            "general_physical_probe": _py(gp_probe),
            "general_physical_probe_all_refused": bool(gp_probe) and all(
                h.get("refused") for h in gp_probe
            ),
            "why_both_modes": (
                "GENERAL_PHYSICAL scars now refuse regardless of the model "
                "named. Querying with and without a model is how a process "
                "scar stays reachable from a model-specific proposal."
            ),
        },
        "open_byte_levers": [
            {
                "id": c["id"],
                "bytes_removed": (c["bytes_removed"] or {}).get("total"),
                "bytes_added": (c["bytes_added"] or {}).get("total"),
                "net_bytes": c.get("net_bytes"),
                "status": c["status"],
                "licensed": c.get("licensed"),
            }
            for c in cands
            if c["status"] == OPEN
        ],
        "recovered_implementation": {
            "gated_delta": "S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q",
            "kernel": "qwen38_gated_delta_decode_vi_simd",
            "fuse_ba_delta": "qwen38_gated_delta_decode_vi_simd_ba",
            "z_binding": "gated RMSNorm only; not gated-delta; not conv",
            "state_layout": "value_heads x key_head_dim x value_head_dim",
            "catalog_format": "HQ38M20 + HQ30UQ4 (qkvz/ba/out) + f32v2 (conv1d)",
        },
        "gaps_closed": [
            "every candidate carries bytes_removed and bytes_added or the module refuses",
            "recurrent-state resident bytes reconcile to 150,994,944",
            "the rank-1 + decay update is recorded as a parameterisation, not a requirement",
            "a synthetic SVD of S under isotropic coefficients is reported so structural low-rank is not assumed",
            "QN-STATE-MERGING, REJECTED_DENSE_REMAT, and q/k/v/z bit-descent are cited rather than re-derived",
            "negative_index queried with and without a model so GENERAL_PHYSICAL scars refuse",
            "FUSE_BA_DELTA is EXISTING_LEVER (zero catalog bytes), not a new representation",
        ],
        "negative_findings": [
            "q/k/v/z bit-descent is not this question and is already MEASURED_NEGATIVE",
            "a rank-r store of 128x128 S is not closed under the incumbent write",
            "depth-merge of S is QN-STATE-MERGING, already falsified",
            "emit-W-then-Q4 is REJECTED_DENSE_REMAT",
            "scale-field sharing across DN layers is MEASURED_NEGATIVE; W itself remains UNMEASURED",
            "fused update+consume of S is already the vi_simd kernel; FUSE_BA_DELTA is in tree and eliminates launches, not the 2.96 GB",
        ],
        "nomenclature": {
            "already_falsified": ALREADY_FALSIFIED,
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "existing_lever": EXISTING_LEVER,
            "unmeasured": UNMEASURED,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "direct_consume": DIRECT_CONSUME,
            "depends_on_lowering": DEPENDS_ON_LOWERING,
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--accounting-only", action="store_true")
    args = parser.parse_args(argv_list)
    if args.accounting_only:
        snap = incumbent_bytes()
        json.dump(
            {
                "catalog_bytes": snap["catalog_weights"]["total"],
                "recurrent_resident": snap["state"]["recurrent_resident"],
                "reconciled": snap["reconciled"],
            },
            _sys.stdout,
            indent=2,
        )
        _sys.stdout.write("\n")
        return 0
    if args.build or args.selftest or not argv_list:
        out = build()
        print(out)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))
