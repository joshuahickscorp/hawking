"""CAPABILITY INFORMATION MAP — which bits the resident actually needs.

Both major organs sit at their entropy floor (MLP codes 1.87018 of 2 bits,
DeltaNet 3.47 of 4). Uniform compression is finished. This lane asks a
different question: does every layer, channel and block deserve the SAME
number of bits, measured on the resident's own function F, not on the
stored q-stream.

A bit reduction is NEVER reported as supported on an entropy or W-space
distortion argument alone. It needs a recorded effect on a downstream
quantity (layer output, hidden after N further layers, logits/argmax, or
a gate the consume path depends on). Missing that measurement is
DOWNSTREAM_UNMEASURED, not a quiet yes.

Activations come from a real CPU replay of the sealed-3.14 packed weights
on real vocabulary rows (embedding lookup of real token ids, then the
hybrid GQA/DeltaNet + SwiGLU stack). Isotropic Gaussian x is refused.

    python3 tools/future/capability_information_map.py --build
    python3 -m pytest tools/future/test_capability_information_map.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import causal_budget_71 as cb71
from tools.future import deltanet_qkvz_precision as dqp
from tools.future import deltanet_representation as dnr
from tools.future._common import REPO, write_receipt
from tools.future.mlp_auxiliary_information import (
    _unpack_q,
    parse_catalog_records,
    parse_hgrafv01_header,
)
from tools.future.mlp_byte_census import (
    CATALOG_NAME,
    CatalogAbsent,
    classify_tensor,
    load_geometry,
    load_sealed,
    resolve_artifact_root,
)
from tools.future.mlp_code_information import CODE_BYTES_TARGET
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "CAPABILITY_INFORMATION_MAP.json"
SCHEMA = "hawking.future.capability_information_map.v1"
VERSION = 1
RECORDED_BY = "tools/future/capability_information_map.py"

TOKEN_ACTIVE_TARGET = cb71.ACTIVE_BYTES
MLP_ACTIVE_TARGET = 5_347_795_776
MLP_CODE_BYTES = CODE_BYTES_TARGET
QKVZ_ACTIVE_TARGET = dnr.QKVZ_ACTIVE_TARGET
Q4_CODE_BITS = 4
Q2_CODE_BITS = 2
INCUMBENT_GROUP = 64
MLP_CANDIDATE_BITS = 1
Q4_CANDIDATE_BITS = 3
RMS_EPS = 1.0e-6
ROPE_THETA = 10_000_000.0
GQA_HEADS = 24
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256
GQA_ROTARY_DIM = 64
COSINE_BAR = 0.990
GATE_COSINE_BAR = 0.990
HIDDEN_COSINE_BAR = 0.990
LOGIT_COSINE_BAR = 0.990
UNIFORM_COSINE_STD_BAR = 0.02
N_TOKENS = 4
N_DOWNSTREAM = 2
SAMPLE_LAYERS: tuple[int, ...] = (0, 21, 42, 63)
CHANNEL_GROUPS = 4
# Real vocabulary rows, not Gaussian x. ĠThe / Ġwould / is / France.
PROMPT_TOKEN_IDS: tuple[int, ...] = (561, 1000, 284, 47358)
RNG_SEED = 38  # unused for activations; kept so a caller cannot silently swap in noise

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FALSIFIED = "ALREADY_FALSIFIED"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
OPEN = "OPEN"
UNMEASURED = "UNMEASURED"

DOWNSTREAM_UNMEASURED = "DOWNSTREAM_UNMEASURED"
ENTROPY_OR_WSPACE_ALONE_INSUFFICIENT = "ENTROPY_OR_WSPACE_ALONE_INSUFFICIENT"
SENSITIVITY_INCOMPLETE = "SENSITIVITY_INCOMPLETE"
LAYER_OUTPUT_BELOW_BAR = "LAYER_OUTPUT_BELOW_BAR"
HIDDEN_AFTER_N_BELOW_BAR = "HIDDEN_AFTER_N_BELOW_BAR"
ARGMAX_CHANGED = "ARGMAX_CHANGED"
LOGITS_BELOW_BAR = "LOGITS_BELOW_BAR"
GATE_BELOW_BAR = "GATE_BELOW_BAR"
SENSITIVITY_CLEARS_BAR = "SENSITIVITY_CLEARS_BAR"
SYNTHETIC_INPUT_REFUSED = "SYNTHETIC_INPUT_REFUSED"

FUSION_ENV = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}

RESIDENT_REL = "workspace/ops/build/rust/release/examples/ascension_qwen38_resident"
BUDGET_REL = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"
MLP_CODE_REL = "receipts/future/MLP_CODE_INFORMATION.json"
QKVZ_PREC_REL = "receipts/future/DELTANET_QKVZ_PRECISION.json"
CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"

REQUIRED_CANDIDATE_IDS: tuple[str, ...] = (
    "heterogeneous_layer_bits",
    "heterogeneous_block_bits",
    "heterogeneous_channel_bits",
    "mlp_q1_where_quiet",
    "deltanet_q3_where_quiet",
    "gqa_q3_where_quiet",
    "uniform_bit_drop",
    "entropy_or_wspace_alone",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Activations are a CPU replay of sealed-3.14 HQ38M20 packed weights on "
    "real embedding rows (token ids 561,1000,284,47358). A candidate bit drop "
    "is applied to every prompt token from zero mixer state, not only the last "
    "token. The consume path "
    "is the same operators the resident binds: RMSNorm as (1+δ) on "
    "input/post/final/q/k, DeltaNet rearrange+gated-delta+gated RMSNorm, "
    "GQA qk-norm + partial RoPE + MHA + sigmoid output gate, SwiGLU. "
    "A bit reduction is never supported on entropy or W-space distortion "
    "alone. GPU generate identity is UNMEASURED. Isotropic Gaussian x is "
    "refused. Quoted roof movement is arithmetic over "
    "RESIDENT_71TPS_CAUSAL_BUDGET.json, not a new hardware number."
)


class CapabilityMapRefuse(ValueError):
    """The capability map refused rather than guessing."""


class SyntheticActivationRefuse(CapabilityMapRefuse):
    """Gaussian / invented x is not a capability measurement."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: synthetic activations are not a capability map{extra}"
        )


class DownstreamRequired(CapabilityMapRefuse):
    """A supported drop was requested without a downstream measurement."""

    def __init__(self, region: str = "") -> None:
        who = f" on {region}" if region else ""
        super().__init__(
            f"REFUSED: bit reduction cannot be marked supported{who} without "
            "a recorded downstream measurement"
        )


# ---------------------------------------------------------------------------
# Tiny numeric helpers.
# ---------------------------------------------------------------------------


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _relfro(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.square(a - b).sum()))
    den = float(np.sqrt(np.square(b).sum()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    den = math.sqrt(float(aa @ aa) * float(bb @ bb))
    if den == 0.0:
        return float("nan")
    return float(aa @ bb / den)


def _summ(xs: Sequence[float]) -> dict[str, Any] | None:
    vals = [
        float(x)
        for x in xs
        if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    ]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


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


def is_gqa_layer(layer: int) -> bool:
    return (int(layer) + 1) % 4 == 0


def mixer_kind(layer: int) -> str:
    return "gqa" if is_gqa_layer(layer) else "delta_net"


def rmsnorm_delta(x: np.ndarray, w_delta: np.ndarray, *, eps: float = RMS_EPS) -> np.ndarray:
    """HF-δ packing: stored weight is (true_scale - 1)."""
    x = np.asarray(x, dtype=np.float32)
    inv = 1.0 / math.sqrt(float(np.mean(x * x)) + float(eps))
    return x * np.float32(inv) * (np.float32(1.0) + w_delta.astype(np.float32, copy=False))


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise CapabilityMapRefuse(f"{name} is not an atlas primitive")
    return name


# ---------------------------------------------------------------------------
# Catalog index + packed loaders. Header-only HQ30UQ4 so the 675 MB
# embedding table is never slurped for a four-token prompt.
# ---------------------------------------------------------------------------


def hq30uq4_meta(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        hdr = handle.read(32)
        if len(hdr) < 32 or hdr[:8] != dnr.HQ30UQ4_MAGIC:
            raise CapabilityMapRefuse(f"{path} is not HQ30UQ4")
        version, group_size = struct.unpack_from("<II", hdr, 8)
        rank, reserved = struct.unpack_from("<HH", hdr, 16)
        elements = struct.unpack_from("<Q", hdr, 20)[0]
        reserved_tail = struct.unpack_from("<I", hdr, 28)[0]
        if version != 1 or reserved != 0 or reserved_tail != 0 or rank <= 0:
            raise CapabilityMapRefuse(f"{path} HQ30UQ4 header refused")
        dim_bytes = rank * 4
        dims = handle.read(dim_bytes)
        if len(dims) != dim_bytes:
            raise CapabilityMapRefuse(f"{path} truncated HQ30UQ4 dims")
        shape = list(struct.unpack("<" + "I" * rank, dims))
    groups = (int(elements) + int(group_size) - 1) // int(group_size)
    header_bytes = 32 + dim_bytes
    scale_bytes = groups * 2
    code_bytes = groups * (int(group_size) // 2)
    size = path.stat().st_size
    if size != header_bytes + scale_bytes + code_bytes:
        raise CapabilityMapRefuse(
            f"{path} size {size} != header+scale+code "
            f"{header_bytes + scale_bytes + code_bytes}"
        )
    cols = int(shape[1]) if len(shape) >= 2 else int(group_size)
    return {
        "path": str(path),
        "shape": shape,
        "elements": int(elements),
        "group_size": int(group_size),
        "groups": int(groups),
        "header_bytes": int(header_bytes),
        "payload_off": int(header_bytes),
        "scale_bytes": int(scale_bytes),
        "code_bytes": int(code_bytes),
        "groups_per_row": cols // int(group_size),
    }


def unpack_q4_matrix(path: Path) -> np.ndarray:
    blob = Path(path).read_bytes()
    parsed = dnr.parse_hq30uq4_header(blob, name=str(path))
    rows_n, cols = parsed["shape"]
    gpr = cols // INCUMBENT_GROUP
    scales = np.frombuffer(
        blob[parsed["payload_off"] : parsed["payload_off"] + parsed["scale_bytes"]],
        dtype="<f2",
    ).reshape(rows_n, gpr)
    codes = np.frombuffer(
        blob[parsed["payload_off"] + parsed["scale_bytes"] :], dtype=np.uint8
    ).reshape(rows_n, gpr, INCUMBENT_GROUP // 2)
    W, _q = dqp._unpack_q4(codes, scales)
    return W


def unpack_q4_rows(meta: Mapping[str, Any], row0: int, row1: int) -> np.ndarray:
    path = Path(meta["path"])
    gpr = int(meta["groups_per_row"])
    group = int(meta["group_size"])
    n_rows = int(row1) - int(row0)
    scale_off = int(meta["payload_off"]) + int(row0) * gpr * 2
    code_off = (
        int(meta["payload_off"])
        + int(meta["scale_bytes"])
        + int(row0) * gpr * (group // 2)
    )
    with path.open("rb") as handle:
        handle.seek(scale_off)
        scale_raw = handle.read(n_rows * gpr * 2)
        handle.seek(code_off)
        code_raw = handle.read(n_rows * gpr * (group // 2))
    scales = np.frombuffer(scale_raw, dtype="<f2").reshape(n_rows, gpr)
    codes = np.frombuffer(code_raw, dtype=np.uint8).reshape(n_rows, gpr, group // 2)
    W, _q = dqp._unpack_q4(codes, scales, group=group)
    return W


def q4_and_codes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = Path(path).read_bytes()
    parsed = dnr.parse_hq30uq4_header(blob, name=str(path))
    rows_n, cols = parsed["shape"]
    gpr = cols // INCUMBENT_GROUP
    scales = np.frombuffer(
        blob[parsed["payload_off"] : parsed["payload_off"] + parsed["scale_bytes"]],
        dtype="<f2",
    ).reshape(rows_n, gpr)
    codes = np.frombuffer(
        blob[parsed["payload_off"] + parsed["scale_bytes"] :], dtype=np.uint8
    ).reshape(rows_n, gpr, INCUMBENT_GROUP // 2)
    W, qn = dqp._unpack_q4(codes, scales)
    return W, qn, scales


def load_f32(path: Path, shape: Sequence[int] | None = None) -> np.ndarray:
    blob = Path(path).read_bytes()
    parsed = dnr.parse_f32v2_header(blob, name=str(path), shape=shape)
    arr = np.frombuffer(blob[parsed["payload_off"] :], dtype="<f4").copy()
    if shape is not None:
        arr = arr.reshape(list(shape))
    return arr


def load_affine_q2(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parsed = parse_hgrafv01_header(Path(path))
    rows, cols = (int(parsed["shape"][0]), int(parsed["shape"][1]))
    gpr = cols // INCUMBENT_GROUP
    blob = Path(path).read_bytes()
    off = int(parsed["payload_off"])
    scale = np.frombuffer(blob[off : off + parsed["scale_bytes"]], dtype="<f2").astype(
        np.float32
    ).reshape(rows, gpr)
    off2 = off + int(parsed["scale_bytes"])
    bias = np.frombuffer(blob[off2 : off2 + parsed["bias_bytes"]], dtype="<f2").astype(
        np.float32
    ).reshape(rows, gpr)
    codes = np.frombuffer(blob[off2 + int(parsed["bias_bytes"]) :], dtype=np.uint8)
    q = _unpack_q(codes.reshape(rows * gpr, 16)).reshape(rows, gpr, INCUMBENT_GROUP).astype(
        np.float32
    )
    W = (q * scale[:, :, None] + bias[:, :, None]).reshape(rows, cols)
    return W, q, scale, bias


def requant_affine_q2(W: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    """Per-group two-level (bits=1) LS reconstruction of affine-Q2 W.

    Not a production kernel. The float W is a CPU probe object; the
    lowering that would ship is a native 1-bit GEMV, not unpack-to-dense.
    """
    if int(bits) >= Q2_CODE_BITS:
        return W
    if int(bits) != 1:
        raise CapabilityMapRefuse(f"MLP candidate bits {bits} is not 1")
    rows, cols = W.shape
    gpr = cols // INCUMBENT_GROUP
    group = INCUMBENT_GROUP
    q1 = (q >= 1.5).astype(np.float32)
    Wg = W.reshape(rows, gpr, group)
    n1 = q1.sum(-1)
    n0 = float(group) - n1
    sum1 = (q1 * Wg).sum(-1)
    sum0 = ((1.0 - q1) * Wg).sum(-1)
    mean1 = np.divide(sum1, n1, out=np.zeros_like(sum1), where=n1 > 0)
    mean0 = np.divide(sum0, n0, out=np.zeros_like(sum0), where=n0 > 0)
    a = mean1 - mean0
    b = mean0
    return (q1 * a[:, :, None] + b[:, :, None]).reshape(rows, cols)


def requant_q4_rows(W: np.ndarray, q: np.ndarray, bits: int, rows: np.ndarray | None = None) -> np.ndarray:
    gpr = W.shape[1] // INCUMBENT_GROUP
    if rows is None:
        return dqp._refit_subblock(W, q, int(bits), gpr=gpr, group=INCUMBENT_GROUP)
    Wp = W.copy()
    ri = np.asarray(rows, dtype=np.int64)
    Wp[ri] = dqp._refit_subblock(W[ri], q[ri], int(bits), gpr=gpr, group=INCUMBENT_GROUP)
    return Wp


def code_bytes_at_bits(code_bytes: int, incumbent_bits: int, bits: int) -> int:
    if bits <= 0 or incumbent_bits <= 0:
        raise CapabilityMapRefuse(f"illegal bits {bits}/{incumbent_bits}")
    n_codes = (int(code_bytes) * 8) // int(incumbent_bits)
    if n_codes * int(incumbent_bits) != int(code_bytes) * 8:
        raise CapabilityMapRefuse("code bytes do not divide the incumbent width")
    if (n_codes * bits) % 8:
        raise CapabilityMapRefuse(f"{n_codes} codes at {bits} bits is not a whole number of bytes")
    return (n_codes * bits) // 8


def bytes_eliminated_at_bits(code_bytes: int, incumbent_bits: int, bits: int) -> int:
    return int(code_bytes) - code_bytes_at_bits(code_bytes, incumbent_bits, bits)


# ---------------------------------------------------------------------------
# Real-activation source. Gaussian is a hard refuse.
# ---------------------------------------------------------------------------


def refuse_synthetic_activations(record: Mapping[str, Any] | None) -> None:
    if record is None:
        raise SyntheticActivationRefuse("no activation record")
    kind = str(record.get("kind") or record.get("source") or "")
    banned = ("gaussian", "isotropic", "synthetic", "randn", "random_normal")
    if any(tok in kind.lower() for tok in banned):
        raise SyntheticActivationRefuse(kind)
    if record.get("real_forward_pass") is not True:
        raise SyntheticActivationRefuse("real_forward_pass is not True")
    if record.get("from_embedding_table") is not True and record.get("from_prefix") is not True:
        raise SyntheticActivationRefuse("activations are not from the embedding table or its prefix")


def real_activation_source(token_ids: Sequence[int] = PROMPT_TOKEN_IDS) -> dict[str, Any]:
    ids = [int(t) for t in token_ids]
    if not ids:
        raise CapabilityMapRefuse("REFUSED: empty token id list")
    if any(t < 0 for t in ids):
        raise CapabilityMapRefuse("REFUSED: negative token id")
    return {
        "kind": "cpu_catalog_forward_from_real_embedding_rows",
        "real_forward_pass": True,
        "from_embedding_table": True,
        "from_prefix": True,
        "synthetic": False,
        "token_ids": ids,
        "n_tokens": len(ids),
        "note": (
            "Token ids are rows of language_model.model.embed_tokens.weight "
            "in the sealed HQ38M20 catalog. Not isotropic Gaussian x."
        ),
    }


# ---------------------------------------------------------------------------
# Consume operators. Same math the resident shaders run.
# ---------------------------------------------------------------------------


def _rope_pair(vec: np.ndarray, pos: int, rotary_dim: int = GQA_ROTARY_DIM, theta: float = ROPE_THETA) -> np.ndarray:
    out = vec.copy()
    half = rotary_dim // 2
    fi = np.arange(half, dtype=np.float32)
    inv = np.power(np.float32(theta), -2.0 * fi / np.float32(rotary_dim))
    angle = np.float32(pos) * inv
    c = np.cos(angle)
    s = np.sin(angle)
    a = vec[:half]
    b = vec[half:rotary_dim]
    out[:half] = a * c - b * s
    out[half:rotary_dim] = b * c + a * s
    return out


def gqa_mixer(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    pos: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One decode step of qwen38 GQA: qk-norm, partial RoPE, MHA, sigmoid gate."""
    q_proj = Wq @ x
    k_proj = Wk @ x
    v_proj = Wv @ x
    q_heads = np.empty((GQA_HEADS, GQA_HEAD_DIM), dtype=np.float32)
    gates = np.empty((GQA_HEADS, GQA_HEAD_DIM), dtype=np.float32)
    for h in range(GQA_HEADS):
        base = h * (2 * GQA_HEAD_DIM)
        q_raw = q_proj[base : base + GQA_HEAD_DIM]
        gates[h] = q_proj[base + GQA_HEAD_DIM : base + 2 * GQA_HEAD_DIM]
        inv = 1.0 / math.sqrt(float(np.mean(q_raw * q_raw)) + RMS_EPS)
        q_n = q_raw * np.float32(inv) * (np.float32(1.0) + q_norm)
        q_heads[h] = _rope_pair(q_n, pos)
    k_heads = np.empty((GQA_KV_HEADS, GQA_HEAD_DIM), dtype=np.float32)
    v_heads = np.empty((GQA_KV_HEADS, GQA_HEAD_DIM), dtype=np.float32)
    for h in range(GQA_KV_HEADS):
        k_raw = k_proj[h * GQA_HEAD_DIM : (h + 1) * GQA_HEAD_DIM]
        v_raw = v_proj[h * GQA_HEAD_DIM : (h + 1) * GQA_HEAD_DIM]
        inv = 1.0 / math.sqrt(float(np.mean(k_raw * k_raw)) + RMS_EPS)
        k_n = k_raw * np.float32(inv) * (np.float32(1.0) + k_norm)
        k_heads[h] = _rope_pair(k_n, pos)
        v_heads[h] = v_raw
    k_cache[pos] = k_heads
    v_cache[pos] = v_heads
    seq = pos + 1
    scale = 1.0 / math.sqrt(GQA_HEAD_DIM)
    attn = np.empty((GQA_HEADS, GQA_HEAD_DIM), dtype=np.float32)
    reps = GQA_HEADS // GQA_KV_HEADS
    for h in range(GQA_HEADS):
        kv = h // reps
        qh = q_heads[h]
        keys = k_cache[:seq, kv, :]
        vals = v_cache[:seq, kv, :]
        scores = (keys @ qh) * np.float32(scale)
        scores = scores - float(scores.max())
        w = np.exp(scores)
        w = w / np.float32(w.sum())
        attn[h] = w @ vals
    sigmoid_gate = _sigmoid(gates)
    gated = (attn * sigmoid_gate).reshape(-1)
    out = Wo @ gated
    return out, {"sigmoid_gate": sigmoid_gate, "attn": attn, "gated": gated}


def deltanet_mixer(
    x: np.ndarray,
    Wqkvz: np.ndarray,
    Wba: np.ndarray,
    Wout: np.ndarray,
    conv: np.ndarray,
    conv_state: np.ndarray,
    S: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    norm: np.ndarray,
    geo: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    y = Wqkvz @ x
    ba = Wba @ x
    q, k, v, z = dqp._rearrange_conv(y, conv, conv_state, geo)
    decay, beta = dqp._ba_decay_beta(ba, a_log, dt_bias, geo)
    S2, h = dqp._gated_delta(S, q, k, v, decay, beta)
    S[:, :, :] = S2
    gated = dqp._gated_rmsnorm(h, z, norm)
    out = Wout @ gated.reshape(-1)
    return out, {"z": z, "beta": beta, "decay": decay, "gated": gated, "h": h, "S": S2}


def swiglu(
    x: np.ndarray, Wg: np.ndarray, Wu: np.ndarray, Wd: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    g = Wg @ x
    u = Wu @ x
    gate = _silu(g)
    pre = gate * u
    down = Wd @ pre
    return down, {"silu_gate": gate, "up": u, "pre_down": pre}


# ---------------------------------------------------------------------------
# Catalogue of tensors the prefix actually consumes.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _catalog_index() -> tuple[dict[tuple[Any, str], dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sealed = load_sealed()
    root = resolve_artifact_root(sealed)
    geo = load_geometry(root)
    dn = dnr.load_dn_geometry(root)
    records = parse_catalog_records(root / CATALOG_NAME)
    by: dict[tuple[Any, str], dict[str, Any]] = {}
    for rec in records:
        layer, organ, _whole = classify_tensor(rec["name"])
        by[(layer, organ)] = {
            **dict(rec),
            "layer": layer,
            "organ": organ,
            "segment_path": str(root / "segments" / rec["filename"]),
        }
    return by, geo, dn


def _tensor(layer: int | None, organ: str) -> dict[str, Any]:
    by, _geo, _dn = _catalog_index()
    rec = by.get((layer, organ))
    if rec is None:
        raise CapabilityMapRefuse(f"catalog missing {organ} layer={layer}")
    return rec


def embed_row(token_id: int) -> np.ndarray:
    rec = _tensor(None, "embedding")
    meta = hq30uq4_meta(Path(rec["segment_path"]))
    vocab = int(meta["shape"][0])
    if int(token_id) >= vocab:
        raise CapabilityMapRefuse(f"token id {token_id} >= vocab {vocab}")
    return unpack_q4_rows(meta, int(token_id), int(token_id) + 1)[0]


# ---------------------------------------------------------------------------
# Bit-reduction licence. Entropy / W-space alone can never support a drop.
# ---------------------------------------------------------------------------


def _downstream_measured(downstream: Mapping[str, Any] | None) -> bool:
    return isinstance(downstream, Mapping) and downstream.get("measured") is True


def _has_recorded_downstream_quantity(downstream: Mapping[str, Any]) -> bool:
    """True iff at least one consume-path quantity was actually written down."""
    keys = (
        "layer_output_cosine",
        "hidden_after_n_cosine",
        "logits_cosine",
        "argmax_identical",
        "gate_cosine",
    )
    for k in keys:
        if k in downstream and downstream[k] is not None:
            return True
    return False


def decide_supported_bit_reduction(
    *,
    candidate_bits: int,
    incumbent_bits: int,
    H_q_bits: float | None = None,
    wspace_relfro: float | None = None,
    downstream: Mapping[str, Any] | None = None,
    cosine_bar: float = COSINE_BAR,
    hidden_bar: float = HIDDEN_COSINE_BAR,
    gate_bar: float = GATE_COSINE_BAR,
    logit_bar: float = LOGIT_COSINE_BAR,
) -> dict[str, Any]:
    """A bit reduction is never supported on entropy or W-space alone.

    `downstream` must be an object with measured=True and at least one of
    layer_output / hidden_after_n / logits / argmax / gate recorded.
    Anything short of that is DOWNSTREAM_UNMEASURED, even when H(q) would
    allow a lossless recode and even when W-space rel-fro is tiny.
    """
    lossless_possible = H_q_bits is not None and float(H_q_bits) <= float(candidate_bits)
    entropy_row = {
        "H_q_bits": None if H_q_bits is None else float(H_q_bits),
        "wspace_relfro": None if wspace_relfro is None else float(wspace_relfro),
        "incumbent_bits": int(incumbent_bits),
        "candidate_bits": int(candidate_bits),
        "lossless_possible": bool(lossless_possible),
        "entropy_is_not_a_licence": True,
        "wspace_is_not_a_licence": True,
    }
    if not _downstream_measured(downstream):
        return {
            "supported": False,
            "reason": DOWNSTREAM_UNMEASURED,
            "downstream_measured": False,
            **entropy_row,
        }
    assert downstream is not None
    if not _has_recorded_downstream_quantity(downstream):
        return {
            "supported": False,
            "reason": SENSITIVITY_INCOMPLETE,
            "downstream_measured": True,
            **entropy_row,
        }
    # Supporting a drop requires the organ output AND one quantity past it.
    layer_cos = downstream.get("layer_output_cosine")
    hidden_cos = downstream.get("hidden_after_n_cosine")
    logits_cos = downstream.get("logits_cosine")
    argmax_ident = downstream.get("argmax_identical")
    gate_cos = downstream.get("gate_cosine")
    past = hidden_cos is not None or logits_cos is not None or argmax_ident is not None
    if layer_cos is None or not past:
        return {
            "supported": False,
            "reason": SENSITIVITY_INCOMPLETE,
            "downstream_measured": True,
            "layer_output_cosine": None if layer_cos is None else float(layer_cos),
            **entropy_row,
        }
    if float(layer_cos) < float(cosine_bar):
        return {
            "supported": False,
            "reason": LAYER_OUTPUT_BELOW_BAR,
            "downstream_measured": True,
            "layer_output_cosine": float(layer_cos),
            "cosine_bar": float(cosine_bar),
            **entropy_row,
        }
    if hidden_cos is not None and float(hidden_cos) < float(hidden_bar):
        return {
            "supported": False,
            "reason": HIDDEN_AFTER_N_BELOW_BAR,
            "downstream_measured": True,
            "hidden_after_n_cosine": float(hidden_cos),
            **entropy_row,
        }
    if gate_cos is not None and float(gate_cos) < float(gate_bar):
        return {
            "supported": False,
            "reason": GATE_BELOW_BAR,
            "downstream_measured": True,
            "gate_cosine": float(gate_cos),
            **entropy_row,
        }
    if logits_cos is not None and float(logits_cos) < float(logit_bar):
        return {
            "supported": False,
            "reason": LOGITS_BELOW_BAR,
            "downstream_measured": True,
            "logits_cosine": float(logits_cos),
            **entropy_row,
        }
    if argmax_ident is False:
        return {
            "supported": False,
            "reason": ARGMAX_CHANGED,
            "downstream_measured": True,
            **entropy_row,
        }
    return {
        "supported": True,
        "reason": SENSITIVITY_CLEARS_BAR,
        "downstream_measured": True,
        "layer_output_cosine": float(layer_cos),
        "cosine_bar": float(cosine_bar),
        **entropy_row,
    }


def refuse_supported_without_downstream(region: Mapping[str, Any], *, name: str = "") -> None:
    """Load-bearing: a supported row without downstream raises, it does not pass."""
    if region.get("supported") and not region.get("downstream_measured"):
        raise DownstreamRequired(name or str(region.get("id") or region.get("region") or ""))


# ---------------------------------------------------------------------------
# Roof arithmetic. Quoted from the live 71-TPS budget, not a new measurement.
# Keys avoid HARDWARE_FIELDS (`tps`, `token_ns`, …).
# ---------------------------------------------------------------------------


def roof_after_bytes(
    bytes_eliminated: int,
    *,
    apply_to: str = "mlp",
) -> dict[str, Any]:
    """Clean-GEMV roof after removing `bytes_eliminated` from one organ family.

    Arithmetic over causal_budget_71.ORGANS. Not a hardware claim.
    """
    gb_saved = int(bytes_eliminated) / 1e9
    today_ms = 0.0
    after_ms = 0.0
    for organ in cb71.ORGANS:
        gb = float(organ["gb"])
        take = min(gb_saved, gb) if organ["organ"] == apply_to else 0.0
        today_ms += gb / cb71.CLEAN_GEMV_GB_S * 1000.0
        after_ms += (gb - take) / cb71.CLEAN_GEMV_GB_S * 1000.0
    # Third copy of the same reconstruction. Every fixed term the budget carries
    # belongs here too, or this function quotes a faster token than the one measured.
    today_ms += cb71.HOST_GAP_MS + cb71.UNATTRIBUTED_GPU_MS
    after_ms += cb71.HOST_GAP_MS + cb71.UNATTRIBUTED_GPU_MS
    today_rate = 1000.0 / today_ms
    after_rate = 1000.0 / after_ms
    need_ms = 1000.0 / 71.0
    return {
        "bytes_eliminated": int(bytes_eliminated),
        "gb_eliminated": gb_saved,
        "apply_to_organ": apply_to,
        "quoted_roof_on_todays_bytes": round(today_rate, 2),
        "quoted_roof_after_allocation": round(after_rate, 2),
        "quoted_delta": round(after_rate - today_rate, 2),
        "quoted_roof_ms_today": round(today_ms, 3),
        "quoted_roof_ms_after": round(after_ms, 3),
        "seventy_one_ms": round(need_ms, 3),
        "seventy_one_reachable_at_roof": bool(after_ms <= need_ms + 1e-9),
        "source": BUDGET_REL,
        "note": (
            "Quoted from receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json "
            "arithmetic (clean GEMV 703.5 GB/s + 0.989 ms host gap). Not a "
            "hardware measurement and not a key named tps."
        ),
    }


# ---------------------------------------------------------------------------
# Layer kits. Unpack on demand, drop after the prefix step unless sampled.
# ---------------------------------------------------------------------------


class LayerKit:
    def __init__(self, layer: int, geo: Mapping[str, Any], dn: Mapping[str, Any]) -> None:
        self.layer = int(layer)
        self.kind = mixer_kind(layer)
        self.geo = geo
        self.dn = dn
        self.input_ln = load_f32(Path(_tensor(layer, "norms.input")["segment_path"]), [int(geo["hidden_size"])])
        self.post_ln = load_f32(Path(_tensor(layer, "norms.post_attn")["segment_path"]), [int(geo["hidden_size"])])
        self._mlp: dict[str, Any] | None = None
        self._mix: dict[str, Any] | None = None

    def mlp(self) -> dict[str, Any]:
        if self._mlp is None:
            pack = {}
            for organ, short in (("mlp.gate", "gate"), ("mlp.up", "up"), ("mlp.down", "down")):
                rec = _tensor(self.layer, organ)
                path = Path(rec["segment_path"])
                W, q, scale, bias = load_affine_q2(path)
                header = parse_hgrafv01_header(path)
                pack[short] = {
                    "W": W,
                    "q": q,
                    "scale": scale,
                    "bias": bias,
                    "code_bytes": int(header["code_bytes"]),
                    "stored_bytes": int(rec["stored_bytes"]),
                    "shape": list(W.shape),
                    "organ": organ,
                }
            self._mlp = pack
        return self._mlp

    def mix(self) -> dict[str, Any]:
        if self._mix is None:
            if self.kind == "delta_net":
                Wqkvz, q_qkvz, _s = q4_and_codes(Path(_tensor(self.layer, "attention.linear_qkvz")["segment_path"]))
                Wba = unpack_q4_matrix(Path(_tensor(self.layer, "attention.linear_ba")["segment_path"]))
                Wout = unpack_q4_matrix(Path(_tensor(self.layer, "attention.linear_out")["segment_path"]))
                conv = load_f32(
                    Path(_tensor(self.layer, "attention.linear_conv1d")["segment_path"]),
                    [int(self.dn["conv_channels"]), int(self.dn["conv_kernel"]), 1],
                ).reshape(int(self.dn["conv_channels"]), int(self.dn["conv_kernel"]))
                a_log = load_f32(Path(_tensor(self.layer, "state.A_log")["segment_path"]))
                dt_bias = load_f32(Path(_tensor(self.layer, "state.dt_bias")["segment_path"]))
                norm = load_f32(Path(_tensor(self.layer, "norms.linear_attn")["segment_path"]))
                qkvz_rec = _tensor(self.layer, "attention.linear_qkvz")
                parsed = parse_hq30_code_bytes(Path(qkvz_rec["segment_path"]))
                self._mix = {
                    "Wqkvz": Wqkvz,
                    "q_qkvz": q_qkvz,
                    "Wba": Wba,
                    "Wout": Wout,
                    "conv": conv,
                    "a_log": a_log,
                    "dt_bias": dt_bias,
                    "norm": norm,
                    "qkvz_code_bytes": parsed,
                    "qkvz_stored": int(qkvz_rec["stored_bytes"]),
                    "out_stored": int(_tensor(self.layer, "attention.linear_out")["stored_bytes"]),
                    "ba_stored": int(_tensor(self.layer, "attention.linear_ba")["stored_bytes"]),
                }
            else:
                Wq, q_q, _ = q4_and_codes(Path(_tensor(self.layer, "attention.q")["segment_path"]))
                Wk, q_k, _ = q4_and_codes(Path(_tensor(self.layer, "attention.k")["segment_path"]))
                Wv, q_v, _ = q4_and_codes(Path(_tensor(self.layer, "attention.v")["segment_path"]))
                Wo, q_o, _ = q4_and_codes(Path(_tensor(self.layer, "attention.o")["segment_path"]))
                q_norm = load_f32(Path(_tensor(self.layer, "norms.q")["segment_path"]), [GQA_HEAD_DIM])
                k_norm = load_f32(Path(_tensor(self.layer, "norms.k")["segment_path"]), [GQA_HEAD_DIM])
                self._mix = {
                    "Wq": Wq,
                    "q_q": q_q,
                    "Wk": Wk,
                    "q_k": q_k,
                    "Wv": Wv,
                    "q_v": q_v,
                    "Wo": Wo,
                    "q_o": q_o,
                    "q_norm": q_norm,
                    "k_norm": k_norm,
                    "q_stored": int(_tensor(self.layer, "attention.q")["stored_bytes"]),
                    "k_stored": int(_tensor(self.layer, "attention.k")["stored_bytes"]),
                    "v_stored": int(_tensor(self.layer, "attention.v")["stored_bytes"]),
                    "o_stored": int(_tensor(self.layer, "attention.o")["stored_bytes"]),
                    "q_code_bytes": parse_hq30_code_bytes(Path(_tensor(self.layer, "attention.q")["segment_path"])),
                    "k_code_bytes": parse_hq30_code_bytes(Path(_tensor(self.layer, "attention.k")["segment_path"])),
                    "v_code_bytes": parse_hq30_code_bytes(Path(_tensor(self.layer, "attention.v")["segment_path"])),
                    "o_code_bytes": parse_hq30_code_bytes(Path(_tensor(self.layer, "attention.o")["segment_path"])),
                }
        return self._mix

    def drop_weights(self) -> None:
        self._mlp = None
        self._mix = None


def parse_hq30_code_bytes(path: Path) -> int:
    meta = hq30uq4_meta(path)
    return int(meta["code_bytes"])


def _new_dn_state(dn: Mapping[str, Any], n_tokens: int) -> dict[str, np.ndarray]:
    vh = int(dn["value_heads"])
    kd = int(dn["key_head_dim"])
    vd = int(dn["value_head_dim"])
    C = int(dn["conv_channels"])
    k = int(dn["conv_kernel"])
    return {
        "S": np.zeros((vh, kd, vd), dtype=np.float32),
        "conv": np.zeros((C, k - 1), dtype=np.float32),
    }


def _new_gqa_state(n_tokens: int) -> dict[str, np.ndarray]:
    return {
        "k": np.zeros((n_tokens, GQA_KV_HEADS, GQA_HEAD_DIM), dtype=np.float32),
        "v": np.zeros((n_tokens, GQA_KV_HEADS, GQA_HEAD_DIM), dtype=np.float32),
    }


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, dict):
            out[k] = clone_state(v)
        else:
            out[k] = v
    return out


def _run_layer(
    kit: LayerKit,
    hidden: np.ndarray,
    mix_state: dict[str, np.ndarray],
    pos: int,
    *,
    mix_override: Mapping[str, np.ndarray] | None = None,
    mlp_override: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mix = kit.mix()
    mlp = kit.mlp()
    x_in = rmsnorm_delta(hidden, kit.input_ln)
    if kit.kind == "delta_net":
        Wqkvz = (mix_override or {}).get("Wqkvz", mix["Wqkvz"])
        Wba = (mix_override or {}).get("Wba", mix["Wba"])
        Wout = (mix_override or {}).get("Wout", mix["Wout"])
        mixer_out, mix_aux = deltanet_mixer(
            x_in,
            Wqkvz,
            Wba,
            Wout,
            mix["conv"],
            mix_state["conv"],
            mix_state["S"],
            mix["a_log"],
            mix["dt_bias"],
            mix["norm"],
            kit.dn,
        )
    else:
        Wq = (mix_override or {}).get("Wq", mix["Wq"])
        Wk = (mix_override or {}).get("Wk", mix["Wk"])
        Wv = (mix_override or {}).get("Wv", mix["Wv"])
        Wo = (mix_override or {}).get("Wo", mix["Wo"])
        mixer_out, mix_aux = gqa_mixer(
            x_in,
            Wq,
            Wk,
            Wv,
            Wo,
            mix["q_norm"],
            mix["k_norm"],
            mix_state["k"],
            mix_state["v"],
            pos,
        )
    h2 = hidden + mixer_out
    x_mlp = rmsnorm_delta(h2, kit.post_ln)
    Wg = (mlp_override or {}).get("gate", mlp["gate"]["W"])
    Wu = (mlp_override or {}).get("up", mlp["up"]["W"])
    Wd = (mlp_override or {}).get("down", mlp["down"]["W"])
    mlp_out, mlp_aux = swiglu(x_mlp, Wg, Wu, Wd)
    h3 = h2 + mlp_out
    aux = {
        "hidden_in": hidden,
        "post_input_norm": x_in,
        "mixer_out": mixer_out,
        "post_attn_residual": h2,
        "post_attn_norm": x_mlp,
        "mlp_out": mlp_out,
        "hidden_out": h3,
        "mix_aux": mix_aux,
        "mlp_aux": mlp_aux,
        "kind": kit.kind,
    }
    return h3, aux


# ---------------------------------------------------------------------------
# Real prefix. Cached so pytest and --build share one pass.
# ---------------------------------------------------------------------------


def _keep_weight_layers(sample: Sequence[int], n_layers: int) -> set[int]:
    keep: set[int] = set()
    last = int(n_layers) - 1
    for s in sample:
        s = int(s)
        span = 0 if s == last else N_DOWNSTREAM
        for L in range(s, min(last, s + span) + 1):
            keep.add(int(L))
    return keep


@lru_cache(maxsize=1)
def capture_real_prefix(
    token_ids: tuple[int, ...] = PROMPT_TOKEN_IDS,
    sample_layers: tuple[int, ...] = SAMPLE_LAYERS,
) -> dict[str, Any]:
    """Layer-outer prefix: unpack each layer once, run every token through it.

    Equivalent to token-outer decode (causal mixer state is per-layer), but
    does not reload 27 MB codes four times.
    """
    src = real_activation_source(token_ids)
    refuse_synthetic_activations(src)
    _by, geo, dn = _catalog_index()
    n_layers = int(geo["num_hidden_layers"])
    hidden_size = int(geo["hidden_size"])
    ids = list(token_ids)
    n_tokens = len(ids)
    final_ln = load_f32(Path(_tensor(None, "norms.final")["segment_path"]), [hidden_size])
    sample = [int(x) for x in sample_layers]
    keep_w = _keep_weight_layers(sample, n_layers)
    sample_set = set(sample)

    hiddens = [embed_row(int(t)) for t in ids]
    kits: dict[int, LayerKit] = {}
    snaps: dict[int, dict[str, Any]] = {}
    token_hidden_in: dict[int, list[np.ndarray]] = {}
    last_aux_63 = None

    for layer in range(n_layers):
        kit = LayerKit(layer, geo, dn)
        st = _new_gqa_state(n_tokens) if kit.kind == "gqa" else _new_dn_state(dn, n_tokens)
        last_aux = None
        if layer in keep_w or layer in sample_set:
            token_hidden_in[layer] = []
        for pos in range(n_tokens):
            if layer in token_hidden_in:
                token_hidden_in[layer].append(hiddens[pos].copy())
            hiddens[pos], aux = _run_layer(kit, hiddens[pos], st, pos)
            last_aux = aux
        if last_aux is None:
            raise CapabilityMapRefuse(f"layer {layer} produced no last-token aux")
        if layer in sample_set or layer in keep_w:
            snaps[layer] = {
                "hidden_in": last_aux["hidden_in"].copy(),
                "hidden_out": last_aux["hidden_out"].copy(),
                "mixer_out": last_aux["mixer_out"].copy(),
                "mlp_out": last_aux["mlp_out"].copy(),
                "post_input_norm": last_aux["post_input_norm"].copy(),
                "post_attn_norm": last_aux["post_attn_norm"].copy(),
                "silu_gate": last_aux["mlp_aux"]["silu_gate"].copy(),
                "kind": kit.kind,
            }
        if layer == n_layers - 1:
            last_aux_63 = {
                "hidden_out": last_aux["hidden_out"].copy(),
                "silu_gate": last_aux["mlp_aux"]["silu_gate"].copy(),
                "mlp_out": last_aux["mlp_out"].copy(),
                "mixer_out": last_aux["mixer_out"].copy(),
            }
        if layer in keep_w:
            kits[layer] = kit
        else:
            kit.drop_weights()

    last_hidden = hiddens[-1].copy()
    final_h = rmsnorm_delta(last_hidden, final_ln)
    logits = None
    argmax = None
    logits_status = UNMEASURED
    try:
        logits = lm_head_gemv(final_h)
        argmax = int(np.argmax(logits))
        logits_status = "MEASURED"
    except (OSError, CapabilityMapRefuse, MemoryError) as exc:
        logits_status = f"UNMEASURED:{type(exc).__name__}"

    return {
        "source": src,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_tokens": n_tokens,
        "sample_layers": sample,
        "snaps": snaps,
        "token_hidden_in": token_hidden_in,
        "kits": kits,
        "geo": geo,
        "dn": dn,
        "final_ln": final_ln,
        "final_hidden": last_hidden,
        "final_normed": final_h,
        "logits": logits,
        "argmax": argmax,
        "logits_status": logits_status,
        "last_aux_63": last_aux_63,
    }


def lm_head_gemv(x: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
    rec = _tensor(None, "lm_head")
    meta = hq30uq4_meta(Path(rec["segment_path"]))
    rows = int(meta["shape"][0])
    y = np.empty(rows, dtype=np.float32)
    for r0 in range(0, rows, int(chunk_rows)):
        r1 = min(rows, r0 + int(chunk_rows))
        W = unpack_q4_rows(meta, r0, r1)
        y[r0:r1] = W @ x
    return y


# ---------------------------------------------------------------------------
# Perturbation probes on captured real X.
# ---------------------------------------------------------------------------


def _region_id(layer: int, organ: str, block: str, channel: str | None = None) -> str:
    if channel is None:
        return f"L{layer}.{organ}.{block}"
    return f"L{layer}.{organ}.{block}.{channel}"


def _mlp_channel_slices(n_rows: int, n_groups: int) -> list[tuple[str, slice]]:
    width = n_rows // n_groups
    out = []
    for i in range(n_groups):
        a = i * width
        b = n_rows if i == n_groups - 1 else (i + 1) * width
        out.append((f"rows_{a}_{b}", slice(a, b)))
    return out


def _measure_pair(
    base: np.ndarray,
    pert: np.ndarray,
) -> dict[str, Any]:
    return {
        "cosine": _cosine(pert, base),
        "relfro": _relfro(pert, base),
        "identical": bool(np.array_equal(pert, base)),
    }


def _apply_mlp_bits(kit: LayerKit, organ_short: str, bits: int, rows: slice | None = None) -> np.ndarray:
    slot = kit.mlp()[organ_short]
    W = slot["W"]
    q = slot["q"]
    Wp = requant_affine_q2(W, q, bits)
    if rows is None:
        return Wp
    out = W.copy()
    out[rows] = Wp[rows]
    return out


def _apply_q4_bits(W: np.ndarray, q: np.ndarray, bits: int, rows: np.ndarray | None = None) -> np.ndarray:
    return requant_q4_rows(W, q, bits, rows=rows)


def replay_prompt_from(
    cap: Mapping[str, Any],
    start_layer: int,
    n_more: int,
    *,
    mix_override: Mapping[str, np.ndarray] | None = None,
    mlp_override: Mapping[str, np.ndarray] | None = None,
    override_layer: int | None = None,
) -> dict[str, Any]:
    """Replay every prompt token from `start_layer` through start+n_more.

    Mixer state starts at zero. Incoming residuals at `start_layer` are the
    unperturbed prefix values (layers below are incumbent). A bit drop is
    applied to every token, not just the last — last-token-only would let
    k/v look quiet because earlier tokens had already written S.
    """
    kits: dict[int, LayerKit] = cap["kits"]
    n_layers = int(cap["n_layers"])
    n_tokens = int(cap["n_tokens"])
    layer = int(start_layer)
    end = min(n_layers - 1, layer + int(n_more))
    hin = cap["token_hidden_in"].get(layer)
    if hin is None or len(hin) != n_tokens:
        raise CapabilityMapRefuse(f"missing per-token hidden_in for layer {layer}")
    states: dict[int, dict[str, np.ndarray]] = {}
    for L in range(layer, end + 1):
        kit = kits[L]
        states[L] = (
            _new_gqa_state(n_tokens) if kit.kind == "gqa" else _new_dn_state(kit.dn, n_tokens)
        )
    aux_start = None
    hidden_after = None
    for pos in range(n_tokens):
        hidden = hin[pos].copy()
        for L in range(layer, end + 1):
            kit = kits[L]
            mo = mix_override if (override_layer is None or L == override_layer) else None
            po = mlp_override if (override_layer is None or L == override_layer) else None
            hidden, aux = _run_layer(
                kit, hidden, states[L], pos, mix_override=mo, mlp_override=po
            )
            if L == layer:
                aux_start = aux
        hidden_after = hidden.copy()
    return {"aux": aux_start, "hidden_after_n": hidden_after, "end_layer": end}


def _gate_from_aux(aux: Mapping[str, Any], kind: str) -> np.ndarray:
    if kind == "mlp":
        return aux["mlp_aux"]["silu_gate"]
    if aux["kind"] == "delta_net":
        return aux["mix_aux"]["z"].reshape(-1)
    return aux["mix_aux"]["sigmoid_gate"].reshape(-1)


def _organ_output(aux: Mapping[str, Any], family: str) -> np.ndarray:
    if family == "mlp":
        return aux["mlp_out"]
    return aux["mixer_out"]


def measure_region(
    cap: Mapping[str, Any],
    *,
    layer: int,
    organ: str,
    block: str,
    candidate_bits: int,
    incumbent_bits: int,
    rows: slice | np.ndarray | None = None,
    channel: str | None = None,
    H_q_bits: float | None = None,
    code_bytes: int,
    stored_bytes: int,
    apply_to_organ: str,
) -> dict[str, Any]:
    kits: dict[int, LayerKit] = cap["kits"]
    kit = kits[int(layer)]
    mix_ov: dict[str, np.ndarray] | None = None
    mlp_ov: dict[str, np.ndarray] | None = None
    family = "mlp" if organ.startswith("mlp") else "attention"
    w_rel = None

    if organ in ("mlp.gate", "mlp.up", "mlp.down"):
        short = organ.split(".")[1]
        Wp = _apply_mlp_bits(kit, short, candidate_bits, rows=rows if isinstance(rows, slice) else None)
        if isinstance(rows, slice):
            W = kit.mlp()[short]["W"]
            w_rel = _relfro(Wp[rows], W[rows])
        else:
            w_rel = _relfro(Wp, kit.mlp()[short]["W"])
        mlp_ov = {short: Wp}
        gate_kind = "mlp"
    elif organ == "attention.linear_qkvz":
        mix = kit.mix()
        idx = dnr.fused_qkvz_row_indices(kit.dn)
        ri = idx[block] if block in idx else None
        Wp = _apply_q4_bits(mix["Wqkvz"], mix["q_qkvz"], candidate_bits, rows=ri)
        w_rel = _relfro(Wp if ri is None else Wp[ri], mix["Wqkvz"] if ri is None else mix["Wqkvz"][ri])
        mix_ov = {"Wqkvz": Wp}
        gate_kind = "attention"
    elif organ in ("attention.q", "attention.k", "attention.v", "attention.o"):
        mix = kit.mix()
        key_W = {"attention.q": "Wq", "attention.k": "Wk", "attention.v": "Wv", "attention.o": "Wo"}[organ]
        key_q = {"attention.q": "q_q", "attention.k": "q_k", "attention.v": "q_v", "attention.o": "q_o"}[organ]
        Wp = _apply_q4_bits(mix[key_W], mix[key_q], candidate_bits, rows=None)
        w_rel = _relfro(Wp, mix[key_W])
        mix_ov = {key_W: Wp}
        gate_kind = "attention"
    else:
        raise CapabilityMapRefuse(f"unknown organ {organ}")

    n_more = 0 if int(layer) == int(cap["n_layers"]) - 1 else N_DOWNSTREAM
    base = replay_prompt_from(cap, int(layer), n_more, override_layer=int(layer))
    pert = replay_prompt_from(
        cap,
        int(layer),
        n_more,
        mix_override=mix_ov,
        mlp_override=mlp_ov,
        override_layer=int(layer),
    )
    layer_m = _measure_pair(_organ_output(base["aux"], family), _organ_output(pert["aux"], family))
    hidden_m = _measure_pair(base["hidden_after_n"], pert["hidden_after_n"])
    gate_m = _measure_pair(_gate_from_aux(base["aux"], gate_kind), _gate_from_aux(pert["aux"], gate_kind))

    logits_cos = None
    argmax_ident = None
    logits_status = UNMEASURED
    if int(layer) == int(cap["n_layers"]) - 1 and cap.get("logits") is not None:
        final_ln = cap["final_ln"]
        h_b = rmsnorm_delta(base["hidden_after_n"], final_ln)
        h_p = rmsnorm_delta(pert["hidden_after_n"], final_ln)
        # The last-layer replay already ends at hidden_out of L63.
        logits_b = cap["logits"]
        # Recompute perturbed logits only (baseline logits already captured).
        logits_p = lm_head_gemv(h_p)
        # Recompute baseline from this replay to keep the pair honest.
        logits_b = lm_head_gemv(h_b)
        logits_cos = _cosine(logits_p, logits_b)
        argmax_ident = bool(int(np.argmax(logits_p)) == int(np.argmax(logits_b)))
        logits_status = "MEASURED"

    downstream = {
        "measured": True,
        "layer_output_cosine": layer_m["cosine"],
        "layer_output_relfro": layer_m["relfro"],
        "hidden_after_n_cosine": hidden_m["cosine"],
        "hidden_after_n_relfro": hidden_m["relfro"],
        "hidden_after_n_layers": n_more,
        "gate_cosine": gate_m["cosine"],
        "gate_relfro": gate_m["relfro"],
        "logits_cosine": logits_cos,
        "argmax_identical": argmax_ident,
        "logits_status": logits_status,
        "real_forward_pass": True,
        "n_tokens": cap["n_tokens"],
        "token_ids": list(cap["source"]["token_ids"]),
    }
    decision = decide_supported_bit_reduction(
        candidate_bits=int(candidate_bits),
        incumbent_bits=int(incumbent_bits),
        H_q_bits=H_q_bits,
        wspace_relfro=w_rel,
        downstream=downstream,
    )
    save_if = bytes_eliminated_at_bits(int(code_bytes), int(incumbent_bits), int(candidate_bits))
    if rows is not None and isinstance(rows, slice):
        span = int(rows.stop) - int(rows.start)
        # Proportional code-byte model: row share of the tensor.
        n_rows = int(kit.mlp()[organ.split(".")[1]]["W"].shape[0]) if organ.startswith("mlp") else span
        save_if = int(round(save_if * (span / max(n_rows, 1))))
    save = save_if if decision["supported"] else 0
    row = {
        "id": _region_id(layer, organ, block, channel),
        "layer": int(layer),
        "organ": organ,
        "block": block,
        "channel": channel,
        "mixer_kind": kit.kind,
        "incumbent_bits": int(incumbent_bits),
        "candidate_bits": int(candidate_bits),
        "bits": int(candidate_bits) if decision["supported"] else int(incumbent_bits),
        "supported": bool(decision["supported"]),
        "reason": decision["reason"],
        "downstream_measured": bool(decision["downstream_measured"]),
        "bytes_eliminated": int(save),
        "bytes_eliminated_if_true": int(save_if),
        "stored_bytes": int(stored_bytes),
        "code_bytes": int(code_bytes),
        "apply_to_organ": apply_to_organ,
        "wspace_relfro": w_rel,
        "H_q_bits": H_q_bits,
        "decision": decision,
        "downstream": downstream,
        "physical_primitive": _require_primitive(
            "ConditionalPhysicalProgram" if channel is not None else "FusedDecodeCompute"
        ),
    }
    refuse_supported_without_downstream(row, name=row["id"])
    return row


def _qkvz_block_code_bytes(total_code: int, block: str, dn: Mapping[str, Any]) -> int:
    parts = dnr.qkvz_subblock_parts(dn)
    return int(parts[block]["code_bytes"]) // int(dn["n_deltanet_layers"])


def measure_sensitivity(cap: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cap = cap if cap is not None else capture_real_prefix()
    refuse_synthetic_activations(cap["source"])
    by, geo, dn = _catalog_index()
    mlp_H = 1.87017973205819
    qkvz_H = {
        "q": 3.471463118861117,
        "k": 3.4652345543760426,
        "v": 3.4700463156032484,
        "z": 3.479417630024027,
    }
    regions: list[dict[str, Any]] = []
    for layer in cap["sample_layers"]:
        kit = cap["kits"][int(layer)]
        kind = mixer_kind(int(layer))
        # Whole-organ / whole-block drops.
        for short, organ in (("gate", "mlp.gate"), ("up", "mlp.up"), ("down", "mlp.down")):
            code_b = int(kit.mlp()[short]["code_bytes"])
            regions.append(
                measure_region(
                    cap,
                    layer=int(layer),
                    organ=organ,
                    block="all",
                    candidate_bits=MLP_CANDIDATE_BITS,
                    incumbent_bits=Q2_CODE_BITS,
                    H_q_bits=mlp_H,
                    code_bytes=code_b,
                    stored_bytes=int(kit.mlp()[short]["stored_bytes"]),
                    apply_to_organ="mlp",
                )
            )
        if kind == "delta_net":
            total_code = int(kit.mix()["qkvz_code_bytes"])
            for block in dnr.SUBBLOCKS:
                regions.append(
                    measure_region(
                        cap,
                        layer=int(layer),
                        organ="attention.linear_qkvz",
                        block=block,
                        candidate_bits=Q4_CANDIDATE_BITS,
                        incumbent_bits=Q4_CODE_BITS,
                        H_q_bits=qkvz_H[block],
                        code_bytes=_qkvz_block_code_bytes(total_code, block, dn),
                        stored_bytes=int(kit.mix()["qkvz_stored"]),
                        apply_to_organ="deltanet",
                    )
                )
        else:
            for organ, key_store, key_code in (
                ("attention.q", "q_stored", "q_code_bytes"),
                ("attention.k", "k_stored", "k_code_bytes"),
                ("attention.v", "v_stored", "v_code_bytes"),
                ("attention.o", "o_stored", "o_code_bytes"),
            ):
                mix = kit.mix()
                regions.append(
                    measure_region(
                        cap,
                        layer=int(layer),
                        organ=organ,
                        block="all",
                        candidate_bits=Q4_CANDIDATE_BITS,
                        incumbent_bits=Q4_CODE_BITS,
                        H_q_bits=3.47,
                        code_bytes=int(mix[key_code]),
                        stored_bytes=int(mix[key_store]),
                        apply_to_organ="gqa",
                    )
                )
        # Channel groups: mlp.gate only, first and last sampled layer.
        if int(layer) in (int(cap["sample_layers"][0]), int(cap["sample_layers"][-1])):
            n_rows = int(kit.mlp()["gate"]["W"].shape[0])
            code_b = int(kit.mlp()["gate"]["code_bytes"])
            stored = int(kit.mlp()["gate"]["stored_bytes"])
            for name, sl in _mlp_channel_slices(n_rows, CHANNEL_GROUPS):
                regions.append(
                    measure_region(
                        cap,
                        layer=int(layer),
                        organ="mlp.gate",
                        block="channel",
                        channel=name,
                        rows=sl,
                        candidate_bits=MLP_CANDIDATE_BITS,
                        incumbent_bits=Q2_CODE_BITS,
                        H_q_bits=mlp_H,
                        code_bytes=code_b,
                        stored_bytes=stored,
                        apply_to_organ="mlp",
                    )
                )

    layer_cos = [r["downstream"]["layer_output_cosine"] for r in regions]
    hidden_cos = [r["downstream"]["hidden_after_n_cosine"] for r in regions]
    gate_cos = [r["downstream"]["gate_cosine"] for r in regions]
    w_rel = [r["wspace_relfro"] for r in regions if r["wspace_relfro"] is not None]
    by_layer: dict[str, Any] = {}
    for r in regions:
        by_layer.setdefault(str(r["layer"]), []).append(r["downstream"]["layer_output_cosine"])
    layer_means = {k: float(np.mean(v)) for k, v in by_layer.items()}
    spread = float(max(layer_cos) - min(layer_cos)) if layer_cos else 0.0
    std = float(np.std(np.asarray(layer_cos, dtype=np.float64))) if layer_cos else 0.0
    uniform = bool(std <= UNIFORM_COSINE_STD_BAR and spread <= 3 * UNIFORM_COSINE_STD_BAR)
    return {
        "measured": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "real_forward_pass": True,
        "synthetic": False,
        "source": cap["source"],
        "n_tokens": cap["n_tokens"],
        "sample_layers": list(cap["sample_layers"]),
        "n_downstream": N_DOWNSTREAM,
        "cosine_bar": COSINE_BAR,
        "hidden_bar": HIDDEN_COSINE_BAR,
        "gate_bar": GATE_COSINE_BAR,
        "uniform_cosine_std_bar": UNIFORM_COSINE_STD_BAR,
        "n_regions": len(regions),
        "regions": regions,
        "layer_output_cosine": _summ(layer_cos),
        "hidden_after_n_cosine": _summ(hidden_cos),
        "gate_cosine": _summ(gate_cos),
        "wspace_relfro": _summ(w_rel),
        "layer_output_cosine_mean_by_layer": layer_means,
        "sensitivity_spread": spread,
        "sensitivity_std": std,
        "sensitivity_uniform": uniform,
        "any_supported": any(r["supported"] for r in regions),
        "logits_status_last_layer": cap["logits_status"],
        "baseline_argmax": cap["argmax"],
        "perturb_every_prompt_token": True,
        "note": (
            "W-space rel-fro is recorded and is not a licence. Gaussian x was "
            "not used. A bit drop is applied to every prompt token from zero "
            "mixer state. Generate identity of a packed kernel is UNMEASURED. "
            "Unsampled layers are UNMEASURED."
        ),
    }


# ---------------------------------------------------------------------------
# Allocation.
# ---------------------------------------------------------------------------


def allocation_from_sensitivity(sens: Mapping[str, Any]) -> dict[str, Any]:
    regions = list(sens.get("regions") or [])
    if not regions or not sens.get("measured"):
        raise CapabilityMapRefuse("REFUSED: allocation without a sensitivity measurement")
    supported = [r for r in regions if r["supported"]]
    for r in regions:
        refuse_supported_without_downstream(r, name=r["id"])
        if r["supported"] and not r["downstream_measured"]:
            raise DownstreamRequired(r["id"])
        if r["supported"] and r["reason"] in {
            ENTROPY_OR_WSPACE_ALONE_INSUFFICIENT,
            DOWNSTREAM_UNMEASURED,
        }:
            raise DownstreamRequired(r["id"])
    # Whole-tensor rows count first. A licensed channel slice of an organ that
    # did NOT itself license is extra; a slice of an already-licensed organ
    # would double-count.
    total = 0
    by_apply: dict[str, int] = {}
    licensed_parent: set[tuple[int, str]] = set()
    for r in regions:
        if r.get("channel") is None and r["supported"]:
            total += int(r["bytes_eliminated"])
            by_apply[r["apply_to_organ"]] = by_apply.get(r["apply_to_organ"], 0) + int(
                r["bytes_eliminated"]
            )
            licensed_parent.add((int(r["layer"]), str(r["organ"])))
    for r in regions:
        if r.get("channel") is not None and r["supported"]:
            if (int(r["layer"]), str(r["organ"])) in licensed_parent:
                continue
            total += int(r["bytes_eliminated"])
            by_apply[r["apply_to_organ"]] = by_apply.get(r["apply_to_organ"], 0) + int(
                r["bytes_eliminated"]
            )
    keep = [r["id"] for r in regions if not r["supported"]]
    drop = [r["id"] for r in supported]
    uniform = bool(sens.get("sensitivity_uniform"))
    if uniform and not supported:
        verdict = (
            "Sensitivity is uniform: every sampled layer, block and channel "
            f"fails the {COSINE_BAR} downstream bar in the same way "
            f"(layer-output cosine std {sens.get('sensitivity_std')}). "
            "Heterogeneous allocation is closed as a school on this parent. "
            "Zero bytes eliminated. The 71-TPS roof does not move."
        )
        school = "CLOSED_UNIFORM_SENSITIVITY"
    elif supported:
        verdict = (
            f"{len(supported)} sampled region(s) cleared the downstream bars; "
            f"{total} bytes of those measured tensors would be eliminated. "
            "Unsampled layers are UNMEASURED and keep incumbent bits. "
            "This is not a generate identity gate."
        )
        school = "HETEROGENEOUS_LICENSED"
    else:
        verdict = (
            "Sensitivity is not perfectly uniform, but no region cleared the "
            "downstream bars. Nothing is licensed. Zero bytes eliminated."
        )
        school = "CLOSED_NO_REGION_CLEARS"
    # Roof: apply MLP bytes to mlp, leftover to the named organ. Quote one
    # combined movement by summing into mlp first (the dominant body).
    roof = roof_after_bytes(total, apply_to="mlp")
    if by_apply.get("deltanet") or by_apply.get("gqa"):
        # Recompute a combined roof by subtracting each organ's save.
        gb = {k: v / 1e9 for k, v in by_apply.items()}
        # Seed with EVERY fixed term the budget's own reconstruction carries, not
        # just the host gap. This loop is a second copy of cb71.token_ms, and it
        # went stale the moment the budget grew UNATTRIBUTED_GPU_MS: it kept
        # reporting a 66.54 roof while the budget said 65.15, and because it
        # DERIVES the figure rather than reading it, rebuilding the receipt did
        # not fix it. Two routes to one ceiling is the drift; naming both terms
        # here is the cheapest repair short of deleting the copy.
        _fixed_ms = cb71.HOST_GAP_MS + cb71.UNATTRIBUTED_GPU_MS
        today_ms = _fixed_ms
        after_ms = _fixed_ms
        for organ in cb71.ORGANS:
            g = float(organ["gb"])
            take = gb.get(organ["organ"], 0.0)
            today_ms += g / cb71.CLEAN_GEMV_GB_S * 1000.0
            after_ms += max(g - take, 0.0) / cb71.CLEAN_GEMV_GB_S * 1000.0
        roof = {
            **roof,
            "quoted_roof_on_todays_bytes": round(1000.0 / today_ms, 2),
            "quoted_roof_after_allocation": round(1000.0 / after_ms, 2),
            "quoted_delta": round(1000.0 / after_ms - 1000.0 / today_ms, 2),
            "quoted_roof_ms_today": round(today_ms, 3),
            "quoted_roof_ms_after": round(after_ms, 3),
            "seventy_one_reachable_at_roof": bool(after_ms <= (1000.0 / 71.0) + 1e-9),
            "gb_eliminated_by_organ": gb,
        }
    return {
        "any_supported": bool(supported),
        "n_regions": len(regions),
        "n_supported": len(supported),
        "total_bytes_eliminated": int(total),
        "bytes_eliminated_by_organ": by_apply,
        "token_bytes_after": TOKEN_ACTIVE_TARGET - int(total),
        "share_of_token_eliminated": int(total) / float(TOKEN_ACTIVE_TARGET),
        "could_take_fewer_bits": drop,
        "must_keep_or_gain": keep,
        "school": school,
        "sensitivity_uniform": uniform,
        "verdict": verdict,
        "roof_movement": roof,
        "cosine_bar": COSINE_BAR,
        "regions": [
            {
                "id": r["id"],
                "layer": r["layer"],
                "organ": r["organ"],
                "block": r["block"],
                "channel": r["channel"],
                "bits": r["bits"],
                "supported": r["supported"],
                "reason": r["reason"],
                "downstream_measured": r["downstream_measured"],
                "bytes_eliminated": r["bytes_eliminated"],
                "layer_output_cosine": r["downstream"]["layer_output_cosine"],
                "hidden_after_n_cosine": r["downstream"]["hidden_after_n_cosine"],
                "gate_cosine": r["downstream"]["gate_cosine"],
                "logits_cosine": r["downstream"]["logits_cosine"],
                "argmax_identical": r["downstream"]["argmax_identical"],
                "logits_status": r["downstream"]["logits_status"],
                "wspace_relfro": r["wspace_relfro"],
            }
            for r in regions
        ],
    }


# ---------------------------------------------------------------------------
# Candidates + answers.
# ---------------------------------------------------------------------------


def _index_hits(family_slugs: Sequence[str]) -> list[dict[str, Any]]:
    try:
        from tools.future.negative_index import refuse_if_dead
    except Exception as exc:  # pragma: no cover
        return [{"index_error": f"{type(exc).__name__}: {exc}"}]
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slug in family_slugs:
        for organ in ("mlp", "deltanet", "attention"):
            refusal = refuse_if_dead(
                {
                    "model": "qwen3.8-27b",
                    "organ": organ,
                    "hypothesis_family": slug,
                }
            )
            if not refusal:
                continue
            key = str(refusal.get("scar_id") or "")
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "scar_id": refusal.get("scar_id"),
                    "source_path": refusal.get("source_path"),
                    "hypothesis_family": refusal.get("hypothesis_family"),
                    "organ": refusal.get("organ"),
                    "verdict": refusal.get("verdict"),
                    "claim_refuted": refusal.get("claim_refuted"),
                    "reopen_condition": refusal.get("reopen_condition"),
                    "queried_slug": slug,
                    "queried_organ": organ,
                }
            )
    return hits


def candidates(alloc: Mapping[str, Any], sens: Mapping[str, Any], *, consult_index: bool = True) -> list[dict[str, Any]]:
    total = int(alloc["total_bytes_eliminated"])
    uniform = bool(alloc["sensitivity_uniform"])
    school = alloc["school"]
    mlp_q1_if = bytes_eliminated_at_bits(MLP_CODE_BYTES, Q2_CODE_BITS, MLP_CANDIDATE_BITS)
    qkvz_q3_if = bytes_eliminated_at_bits(2_013_265_920, Q4_CODE_BITS, Q4_CANDIDATE_BITS)
    drop = list(alloc["could_take_fewer_bits"])
    dn_ids = [i for i in drop if "linear_qkvz" in i]
    gqa_ids = [i for i in drop if "attention.q" in i or "attention.k" in i or "attention.v" in i or "attention.o" in i]
    mlp_ids = [i for i in drop if "mlp." in i]
    hetero_status = OPEN if alloc["any_supported"] else MEASURED_NEGATIVE
    rows = []

    def add(
        cid: str,
        name: str,
        mechanism: str,
        byte_model: str,
        bytes_if: int | None,
        status: str,
        falsifier: str,
        primitive: str,
        slugs: Sequence[str],
        measured: Mapping[str, Any] | None = None,
        support: str = "MEASURED",
        note: str | None = None,
    ) -> None:
        hits = _index_hits(slugs) if consult_index else []
        row = {
            "id": cid,
            "name": name,
            "mechanism": mechanism,
            "byte_model": byte_model,
            "bytes_eliminated_if_true": bytes_if,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A native mixed-width GEMV consumes packed codes in-register. "
                "Unpack-to-dense-W is REJECTED_DENSE_REMAT and is not this probe."
            ),
            "physical_primitive": _require_primitive(primitive),
            "status": status,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
            "cheapest_falsifier": falsifier,
            "index_slugs": list(slugs),
            "index_refusals": hits,
            "support": support,
        }
        if measured is not None:
            row["measured"] = _py(measured)
        if note:
            row["note"] = note
        rows.append(row)

    add(
        "heterogeneous_layer_bits",
        "spend bits by depth",
        "Give later (or earlier) layers fewer bits if F is less fragile there.",
        "sum_layers bits_l * n_params_l / 8. A win needs a proper subset of the 64 layers whose downstream injury clears the bar.",
        total if school == "HETEROGENEOUS_LICENSED" else 0,
        hetero_status,
        "STATIC, run here: per-layer layer-output cosine means "
        f"{sens.get('layer_output_cosine_mean_by_layer')}. Uniformity="
        f"{uniform}. Licensed ids={drop}. Unsampled layers stay UNMEASURED.",
        "ConditionalPhysicalProgram",
        ("uniform_subbit_allocation",),
        measured={"by_layer": sens.get("layer_output_cosine_mean_by_layer"), "uniform": uniform},
    )
    add(
        "heterogeneous_block_bits",
        "spend bits by block (gate/up/down, q/k/v/z, q/k/v/o)",
        "Crush the consume-quiet block; keep the state-writing block.",
        "DeltaNet Q3 of z is 188743680 if licensed; MLP 1-bit of one organ is 1/6 of 4.28 GB.",
        total if dn_ids or gqa_ids else 0,
        hetero_status if (dn_ids or gqa_ids) else MEASURED_NEGATIVE,
        "STATIC, run here: licensed block ids "
        f"{[i for i in drop if 'mlp.gate.channel' not in i]}. "
        "Unsampled layers are UNMEASURED, not licensed by resemblance.",
        "FusedDecodeCompute",
        ("uniform_subbit_allocation", "uniform_q3"),
    )
    add(
        "heterogeneous_channel_bits",
        "spend bits by row-channel inside an organ",
        "Keep a sensitivity-selected subset of rows at incumbent bits; crush the rest.",
        "row_frac * incumbent + (1-row_frac) * candidate, per tensor.",
        sum(int(r["bytes_eliminated"]) for r in alloc["regions"] if r.get("channel") and r["supported"]),
        hetero_status if any(r.get("channel") and r["supported"] for r in alloc["regions"]) else MEASURED_NEGATIVE,
        "STATIC, run here: mlp.gate channel groups measured at L0 and L63. "
        "A licensed slice is a measured F-space island, not an entropy island.",
        "ConditionalPhysicalProgram",
        ("protected_islands", "qn_binary_healing", "uniform_subbit_allocation"),
    )
    add(
        "mlp_q1_where_quiet",
        "1-bit MLP codes on quiet layers/channels only",
        "Replace affine-Q2 with a native 1-bit GEMV on the quiet subset.",
        f"Full-body 1-bit would save {mlp_q1_if} code bytes. Licensed subset saves {total} of those.",
        sum(int(r["bytes_eliminated"]) for r in alloc["regions"] if r["supported"] and str(r["organ"]).startswith("mlp.") and r.get("channel") is None),
        MEASURED_NEGATIVE if not mlp_ids else hetero_status,
        "STATIC, run here: whole-organ MLP 1-bit vs licensed channel slices "
        f"{mlp_ids}. NNS-029 killed uniform Q1; this is the heterogeneous retry.",
        "FusedDecodeCompute",
        ("uniform_q2", "binary_quantization"),
    )
    add(
        "deltanet_q3_where_quiet",
        "3-bit DeltaNet q/k/v/z on quiet blocks only",
        "Row-range Q4/Q3 fused decode on linear_qkvz.",
        f"All-four Q3 would save {qkvz_q3_if}. Licensed subset saves 0.",
        sum(int(r["bytes_eliminated"]) for r in alloc["regions"] if r["supported"] and r["organ"] == "attention.linear_qkvz"),
        MEASURED_NEGATIVE if not dn_ids else hetero_status,
        "STATIC, run here: sampled DeltaNet q/k/v/z Q3 on real X, every prompt "
        f"token, licensed={dn_ids}. Remaining DN layers UNMEASURED. Gaussian-x "
        "predecessor is not this measurement.",
        "FusedDecodeCompute",
        ("uniform_q3", "uniform_subbit_allocation"),
    )
    add(
        "gqa_q3_where_quiet",
        "3-bit GQA q/k/v/o on quiet layers only",
        "Same as DeltaNet, on the 16 full-attention layers.",
        "Q3 of q/k/v/o code bytes on 16 layers if licensed.",
        sum(int(r["bytes_eliminated"]) for r in alloc["regions"] if r["supported"] and r["organ"] in {"attention.q", "attention.k", "attention.v", "attention.o"}),
        MEASURED_NEGATIVE if not gqa_ids else hetero_status,
        f"STATIC, run here: sampled GQA Q3 licensed={gqa_ids}. Other GQA layers UNMEASURED.",
        "FusedDecodeCompute",
        ("uniform_q3",),
    )
    add(
        "uniform_bit_drop",
        "the same fewer bits on every layer and channel",
        "Uniform Q1 MLP and/or uniform Q3 attention. Entropy-floor cousin; capability retry.",
        f"MLP Q1 {mlp_q1_if}; qkvz Q3 {qkvz_q3_if}.",
        0,
        ALREADY_FALSIFIED,
        "Already run: NNS-029 uniform bit-descent below q3; QN-BINARY-INJURY 1-bit body. "
        "This map's uniform sensitivity is the F-space restatement, not a reopen.",
        "FusedDecodeCompute",
        ("uniform_q2", "uniform_q3", "binary_quantization"),
    )
    add(
        "entropy_or_wspace_alone",
        "licence a drop from H(q) or W-space rel-fro",
        "The thing this campaign has been bitten by. Explicitly refused.",
        "n/a — not a byte lever, a forbidden argument.",
        0,
        MEASURED_NEGATIVE,
        "The decide_supported_bit_reduction guard: downstream=None is "
        f"{DOWNSTREAM_UNMEASURED} even when H(q) is lossless and W-space rel-fro is 0.",
        "ConditionalPhysicalProgram",
        ("uniform_subbit_allocation",),
        support="REFUSED",
        note="A test in tools/future/test_capability_information_map.py proves the refusal.",
    )
    ids = [r["id"] for r in rows]
    if ids != list(REQUIRED_CANDIDATE_IDS):
        raise CapabilityMapRefuse(f"candidate ids {ids} != {list(REQUIRED_CANDIDATE_IDS)}")
    return rows


def answers(alloc: Mapping[str, Any], sens: Mapping[str, Any], cap: Mapping[str, Any]) -> dict[str, Any]:
    unmeasured = []
    if cap.get("logits_status") != "MEASURED":
        unmeasured.append("baseline_logits_argmax")
    unmeasured.append("gpu_generate_identity")
    unmeasured.append("every_layer_not_in_sample")
    unmeasured.append("every_row_not_in_channel_group")
    if any(r["downstream"]["logits_status"] == UNMEASURED for r in sens["regions"]):
        unmeasured.append("final_logits_for_early_layer_perturbations")
    return {
        "is_sensitivity_uniform": {
            "answer": (
                "YES at the scale that would change bits. Layer-output cosine "
                f"std={sens.get('sensitivity_std')} spread={sens.get('sensitivity_spread')} "
                f"vs bar {UNIFORM_COSINE_STD_BAR}. Heterogeneous allocation is "
                "closed as a school."
                if sens.get("sensitivity_uniform")
                else (
                    "NO. Spread "
                    f"{sens.get('sensitivity_spread')} exceeds the uniformity bar "
                    f"(std {sens.get('sensitivity_std')}). Licensed subset: "
                    f"{alloc['could_take_fewer_bits'] or 'none'}. Unsampled "
                    "layers remain UNMEASURED."
                )
            ),
            "status": MEASURED_NEGATIVE if (sens.get("sensitivity_uniform") or not alloc["any_supported"]) else OPEN,
            "uniform": bool(sens.get("sensitivity_uniform")),
            "std": sens.get("sensitivity_std"),
            "spread": sens.get("sensitivity_spread"),
            "by_layer": sens.get("layer_output_cosine_mean_by_layer"),
        },
        "which_regions_could_take_fewer_bits": {
            "answer": alloc["could_take_fewer_bits"] or "none",
            "ids": alloc["could_take_fewer_bits"],
            "status": MEASURED_NEGATIVE if not alloc["any_supported"] else OPEN,
        },
        "which_must_keep_or_gain": {
            "answer": (
                "Every sampled region keeps its incumbent width. Nothing is "
                "licensed to drop, and nothing was measured as wanting *more* "
                "bits than it already has."
            ),
            "ids": alloc["must_keep_or_gain"],
            "status": MEASURED_NEGATIVE,
        },
        "bytes_a_nonuniform_allocation_would_eliminate": {
            "answer": int(alloc["total_bytes_eliminated"]),
            "share_of_token": alloc["share_of_token_eliminated"],
            "token_bytes_after": alloc["token_bytes_after"],
            "by_organ": alloc["bytes_eliminated_by_organ"],
        },
        "roof_movement_on_the_71tps_ladder": alloc["roof_movement"],
        "what_is_unmeasured": {
            "answer": unmeasured,
            "status": UNMEASURED,
        },
        "did_entropy_or_wspace_licence_anything": {
            "answer": "NO. The guard refused. See entropy_or_wspace_alone.",
            "status": MEASURED_NEGATIVE,
        },
    }


# ---------------------------------------------------------------------------
# Snapshot / receipt.
# ---------------------------------------------------------------------------


def _predecessor_entropy() -> dict[str, Any]:
    mlp = json.loads((REPO / MLP_CODE_REL).read_text()) if (REPO / MLP_CODE_REL).is_file() else {}
    qkvz = json.loads((REPO / QKVZ_PREC_REL).read_text()) if (REPO / QKVZ_PREC_REL).is_file() else {}
    return {
        "mlp_H_q_bits": ((mlp.get("answers") or {}).get("how_much_of_the_code_body_is_independent") or {}).get(
            "H_q_bits"
        )
        or 1.87017973205819,
        "mlp_independent_fraction": ((mlp.get("answers") or {}).get("how_much_of_the_code_body_is_independent") or {}).get(
            "independent_fraction"
        ),
        "qkvz_H_q_bits": (qkvz.get("entropy") or {}).get("H_q_bits") or 3.473171395612939,
        "note": (
            "Cited from MLP_CODE_INFORMATION and DELTANET_QKVZ_PRECISION. "
            "Entropy is not a licence on this receipt."
        ),
        "licence": False,
    }


def _resident_binding() -> dict[str, Any]:
    sealed = load_sealed()
    parent = Path("/Users/scammermike/Downloads/hawking")
    binary = parent / RESIDENT_REL
    worktree = REPO / RESIDENT_REL
    path = str(binary if binary.is_file() else worktree if worktree.is_file() else binary)
    return {
        "resident_identity": sealed.get("resident_identity"),
        "model_id": sealed.get("model_id"),
        "require_fusion_env": sealed.get("require_fusion_env"),
        "fusion_env": dict(sealed.get("fusion_env") or FUSION_ENV),
        "artifact_root": sealed.get("artifact_root"),
        "tokenizer": sealed.get("tokenizer"),
        "resident_binary": path,
        "resident_binary_readable": Path(path).is_file(),
        "used_for_this_map": (
            "CPU consume of the same packed tensors the resident uploads. "
            "GPU generate was not required for the organ-level probe and is "
            "UNMEASURED as a generate identity gate."
        ),
    }


@lru_cache(maxsize=1)
def _measured_map() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    src = real_activation_source()
    refuse_synthetic_activations(src)
    cap = capture_real_prefix()
    sens = measure_sensitivity(cap)
    alloc = allocation_from_sensitivity(sens)
    return cap, sens, alloc


def snapshot(consult_index: bool = True) -> dict[str, Any]:
    cap, sens, alloc = _measured_map()
    cands = candidates(alloc, sens, consult_index=consult_index)
    return {
        "source": cap["source"],
        "capture": {
            "n_tokens": cap["n_tokens"],
            "token_ids": list(cap["source"]["token_ids"]),
            "sample_layers": list(cap["sample_layers"]),
            "n_layers": cap["n_layers"],
            "logits_status": cap["logits_status"],
            "baseline_argmax": cap["argmax"],
            "real_forward_pass": True,
        },
        "sensitivity": sens,
        "allocation": alloc,
        "candidates": cands,
        "answers": answers(alloc, sens, cap),
        "predecessor_entropy": _predecessor_entropy(),
        "resident": _resident_binding(),
    }


def accounting() -> dict[str, Any]:
    by, geo, dn = _catalog_index()
    mlp_code = 0
    mlp_stored = 0
    qkvz_stored = 0
    for (layer, organ), rec in by.items():
        if organ in ("mlp.gate", "mlp.up", "mlp.down"):
            mlp_stored += int(rec["stored_bytes"])
            mlp_code += int(parse_hgrafv01_header(Path(rec["segment_path"]))["code_bytes"])
        if organ == "attention.linear_qkvz":
            qkvz_stored += int(rec["stored_bytes"])
    if mlp_stored != MLP_ACTIVE_TARGET:
        raise CapabilityMapRefuse(
            f"REFUSED: MLP stored {mlp_stored} != {MLP_ACTIVE_TARGET}"
        )
    if mlp_code != MLP_CODE_BYTES:
        raise CapabilityMapRefuse(
            f"REFUSED: MLP code {mlp_code} != {MLP_CODE_BYTES}"
        )
    if qkvz_stored != QKVZ_ACTIVE_TARGET:
        raise CapabilityMapRefuse(
            f"REFUSED: qkvz stored {qkvz_stored} != {QKVZ_ACTIVE_TARGET}"
        )
    sealed = load_sealed()
    return {
        "token_active_bytes": TOKEN_ACTIVE_TARGET,
        "mlp_stored_bytes": mlp_stored,
        "mlp_code_bytes": mlp_code,
        "qkvz_stored_bytes": qkvz_stored,
        "n_layers": int(geo["num_hidden_layers"]),
        "hidden_size": int(geo["hidden_size"]),
        "reconciled": True,
        "identity": {
            "resident_identity": sealed.get("resident_identity"),
            "model_id": sealed.get("model_id"),
            "artifact_root": sealed.get("artifact_root"),
        },
    }


def build(*, consult_index: bool = True) -> Path:
    acc = accounting()
    snap = snapshot(consult_index=consult_index)
    cands = snap["candidates"]
    n_open = sum(1 for c in cands if c["status"] == OPEN)
    n_meas = sum(1 for c in cands if c["status"] == MEASURED_NEGATIVE)
    n_dead = sum(1 for c in cands if c["status"] == ALREADY_FALSIFIED)
    n_unm = sum(1 for c in cands if c["status"] == UNMEASURED)
    # Strip live numpy from sensitivity for the receipt.
    sens = dict(snap["sensitivity"])
    # Keep region downstream numbers; drop nothing else needed.
    alloc = snap["allocation"]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Sensitivity map of sealed-3.14 over layers, blocks and channels, "
            "measured on real activations from a real forward pass of the "
            "resident's packed weights. Allocates bits only where a downstream "
            "quantity was measured and cleared its bar."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "predecessors": [CENSUS_REL, MLP_CODE_REL, QKVZ_PREC_REL, BUDGET_REL],
        "what_this_does_not_prove": [
            "GPU generate identity of any mixed-width kernel",
            "physical EBPW of a different packing",
            "that every unscanned row is identical to its channel group",
            "that unsampled layers match the sampled ones (they are UNMEASURED)",
            "that Gaussian x would have been an acceptable substitute (it is refused)",
        ],
        "accounting": _py(acc),
        "resident": _py(snap["resident"]),
        "capture": _py(snap["capture"]),
        "predecessor_entropy": _py(snap["predecessor_entropy"]),
        "sensitivity": _py({k: v for k, v in sens.items() if k != "regions"}),
        "allocation": _py(alloc),
        "candidates": _py(cands),
        "answers": _py(snap["answers"]),
        "candidate_counts": {
            "n": len(cands),
            "open": n_open,
            "measured_negative": n_meas,
            "already_falsified": n_dead,
            "unmeasured": n_unm,
        },
        "unmeasured": snap["answers"]["what_is_unmeasured"]["answer"],
        "recovered_implementation": {
            "activations": "CPU embedding-row lookup + 64-layer hybrid consume on last-token residual",
            "token_ids": list(PROMPT_TOKEN_IDS),
            "rmsnorm": "HF-δ (1+w) on input/post/final/q/k; linear_attn.norm is absolute",
            "deltanet": "qwen38_qkvz_rearrange_conv_l2_f32 + gated-delta + gated RMSNorm",
            "gqa": "qk-norm + partial RoPE (dim 64, θ=1e7) + MHA + sigmoid output gate",
            "mlp": "down(silu(gate(x))*up(x)) on affine-Q2 W",
            "licence": "decide_supported_bit_reduction refuses entropy/W-space alone",
            "perturbation": "every prompt token, mixer state from zero, incoming residual from the unperturbed prefix",
        },
        "gaps_closed": [
            "capability sensitivity of F measured on real X, not on H(q)",
            "layers, blocks (gate/up/down, q/k/v/z, q/k/v/o) and mlp.gate channel groups probed",
            "downstream quantities recorded: organ output, hidden after N layers, SwiGLU/DeltaNet/GQA gates, last-layer logits/argmax when reached",
            "a bit reduction cannot be marked supported without that record",
            "bit drop applied to every prompt token from zero mixer state (not last-token-only)",
            "roof movement quoted against RESIDENT_71TPS_CAUSAL_BUDGET.json",
        ],
        "negative_findings": [
            alloc["verdict"],
            "entropy and W-space distortion do not licence a drop",
        ],
        "nomenclature": {
            "already_falsified": ALREADY_FALSIFIED,
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "unmeasured": UNMEASURED,
            "downstream_unmeasured": DOWNSTREAM_UNMEASURED,
            "entropy_or_wspace_alone_insufficient": ENTROPY_OR_WSPACE_ALONE_INSUFFICIENT,
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
        snap = accounting()
        json.dump(_py(snap), _sys.stdout, indent=2)
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
