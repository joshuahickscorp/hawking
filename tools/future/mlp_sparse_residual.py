"""MLP SPARSE RESIDUAL — keep only the exceptions.

Every bulk program so far has been judged alone. The structure that
actually ships is usually cheap predictable bulk + a tiny expensive
exception. A bulk approximation at ~0.9 relative error is useless; the
same bulk plus a residual on the coordinates that carry the error might
not be. Nobody has measured the residual after the best bulk
approximation already on record.

This module:

  1. Refits the best shared-program bulks (SHARED_INPUT / SHARED_OUTPUT /
     SHARED_BOTH at rank 64) and the oracle PCA-of-F control, on the
     sealed teacher corpus, held out by PROMPT.
  2. Forms R(x) = F(x) - F_bulk(x) on held-out prompts and reports the
     concentration curve on four axes: output coordinates, input
     directions, tokens, and blocks of W.
  3. Sweeps a residual budget as a fraction of the 5,347,795,776 MLP
     bytes. For each budget it reports held-out bulk+residual error under
     UNIFORM allocation and under CAPABILITY allocation (the sensitivity
     map allocates the budget; it is never itself a byte win).
  4. Scores every point with executable_economics. The residual's native
     consumer is gather-and-add (atlas DirectRoutedAccumulate). Index
     bytes and the extra dispatch are billed; a free index is a
     fabrication.

If the residual is dense — energy spread evenly, no concentration — no
sparse exception can rescue a bulk program for this organ, and that
closes residual-rescue for every bulk family at once.

    python3 tools/future/mlp_sparse_residual.py --build
    python3 -m pytest tools/future/test_mlp_sparse_residual.py -q

evidence_class STATIC_ONLY. No GPU lease. Does not touch crates/.
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
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import solve_triangular

from tools.future import executable_economics as ee
from tools.future import mlp_shared_program as msp
from tools.future import negative_index as ni
from tools.future._common import REPO, git, load_json, write_receipt
from tools.future.mlp_teacher_corpus import HIDDEN, INTERMEDIATE, N_LAYERS
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "MLP_SPARSE_RESIDUAL.json"
SCHEMA = "hawking.future.mlp_sparse_residual.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_sparse_residual.py"
EVIDENCE_CLASS = "STATIC_ONLY"

CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"
SHARED_REL = "receipts/future/MLP_SHARED_PROGRAM.json"
MAP_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"
BUDGET_REL = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"

ROUND1_LAYER = msp.ROUND1_LAYER  # 38, typical H(q)
BULK_RANK = 64
ELEMENT_BYTES = ee.F16_BYTES
INDEX_BYTES = 4  # uint32 gather index; billed, never waved
METADATA_BASE_BYTES = 256
RIDGE_LAM = 1e-3
N_BLOCKS = 4  # matches capability-map CHANNEL_GROUPS
W_GROUP = 64  # affine-Q2 incumbent group size
RNG_SEED = 38

GATHER_ADD_PRIMITIVE = "DirectRoutedAccumulate"
GATHER_ADD_ALSO: tuple[str, ...] = ("SparseSkip", "TiledProjection")

UNIFORM = "uniform"
CAPABILITY = "capability"
ENERGY_GREEDY = "energy_greedy"
ALLOCATIONS: tuple[str, ...] = (UNIFORM, CAPABILITY)

DIRECT_CONSUME = msp.DIRECT_CONSUME
REJECTED_DENSE_REMAT = msp.REJECTED_DENSE_REMAT
MEASURED_NEGATIVE = msp.MEASURED_NEGATIVE
OPEN = msp.OPEN
RESIDUAL_DENSE_CLOSED = "RESIDUAL_DENSE_CLOSED"
RESIDUAL_RESCUE_CLOSED = "RESIDUAL_RESCUE_CLOSED"

HELD_OUT_KILL_REL = msp.HELD_OUT_KILL_REL  # 0.25
# Residual is dense if 1% of units capture <= 3% of energy (3x uniform
# is still spread out) AND 10% of units capture <= 20%.
DENSE_ENERGY_AT_1PCT = 0.03
DENSE_ENERGY_AT_10PCT = 0.20
# MATERIAL (task): >= 1 ms of token time, or >= 5% of organ bytes, or a
# reusable family, or a decisive falsifier. Byte-bar used as a named sweep
# point, not as a hardware number.
MATERIAL_MS_BAR = 1.0
MATERIAL_BYTE_FRAC = 0.05
COSINE_BAR = 0.99
QUIET_WEIGHT = 0.05

CURVE_FRACS: tuple[float, ...] = (
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
)

# Named residual budgets as a fraction of MLP bytes. The sensitivity map
# licenses 2,785,280 MLP bytes (and 0.28% of the token); those are
# allocation priors, not a byte win. k is derived from the billed
# gather-and-add ledger (value row + uint32 index, 64 layers).
NAMED_BUDGETS: tuple[tuple[str, float], ...] = (
    ("bulk_only", 0.0),
    ("map_licensed_mlp_bytes", 2_785_280 / ee.MLP_ACTIVE_BYTES),
    ("mlp_frac_0.001", 0.001),
    ("map_token_share_0.0028", 0.0028028380503877935),
    ("mlp_frac_0.01", 0.01),
    ("mlp_frac_0.03", 0.03),
    ("mlp_frac_0.05", 0.05),
    ("mlp_frac_0.10", 0.10),
    ("mlp_frac_0.25", 0.25),
)

BILLED_BULKS: tuple[tuple[str, str, int], ...] = (
    ("shared_input_r64", msp.SHARED_INPUT, BULK_RANK),
    ("shared_output_r64", msp.SHARED_OUTPUT, BULK_RANK),
    ("shared_both_r64", msp.SHARED_BOTH, BULK_RANK),
)
ORACLE_PCA_ID = "oracle_pca_r64"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Held-out errors are CPU arithmetic on the sealed-3.14 MLP teacher corpus "
    "(real post_attn_norm X, exact affine-Q2 SwiGLU F(X), split by prompt_id). "
    "They are not capability and not a protected complete-token number. "
    "A train-set figure cannot be reported as held-out. Predicted ms/token is "
    "executable_economics arithmetic over cited organ times with a stated "
    "bandwidth-regime ASSUMPTION. Index bytes and gather-and-add dispatch "
    "are billed. gpu_authority is false. evidence_class is STATIC_ONLY. "
    "The capability map allocates a residual budget; it is not itself a "
    "byte win (licensed subset is 0.28% of the token)."
)

TrainReportedAsHeldOut = msp.TrainReportedAsHeldOut


class SparseResidualRefuse(ValueError):
    """The sparse-residual census refused rather than guessing."""


class UnbilledResidualIndex(SparseResidualRefuse):
    """A sparse residual whose gather indices are free in the receipt is a fabrication."""


class UnbilledDispatch(SparseResidualRefuse):
    """A gather-and-add residual with dispatch_delta 0 is waved, not billed."""


class CorpusUnavailable(msp.CorpusUnavailable):
    """Real (X, F(X)) is not readable; synthesizing X is NNS-001."""


class CapabilityMapUnavailable(SparseResidualRefuse):
    """The sensitivity map is missing; refusing to invent weights."""


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


def _r(value: float, n: int = 6) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise SparseResidualRefuse(f"{name} is not an atlas primitive")
    return name


function_error = msp.function_error
validate_error_authority = msp.validate_error_authority
mean_l2_ratio = msp.mean_l2_ratio


# ---------------------------------------------------------------------------
# Billing. Gather indices and extra dispatch are first-class costs.
# ---------------------------------------------------------------------------


def per_coord_bytes(
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    element_bytes: int = ELEMENT_BYTES,
    index_bytes: int = INDEX_BYTES,
) -> int:
    """Model-scope bytes of one residual output coordinate.

    value row: hidden * f16 * n_layers
    gather index: uint32 * n_layers
    """
    return int(n_layers) * (int(hidden) * int(element_bytes) + int(index_bytes))


def k_from_mlp_frac(
    frac: float,
    *,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    mlp_bytes: int = ee.MLP_ACTIVE_BYTES,
) -> int:
    budget = float(frac) * int(mlp_bytes)
    per = per_coord_bytes(n_layers=n_layers, hidden=hidden)
    if per <= 0 or budget <= 0:
        return 0
    return int(max(0, min(int(hidden), math.floor(budget / per))))


def mlp_frac_of_k(
    k: int,
    *,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    mlp_bytes: int = ee.MLP_ACTIVE_BYTES,
) -> float:
    return float(int(k) * per_coord_bytes(n_layers=n_layers, hidden=hidden)) / float(
        max(int(mlp_bytes), 1)
    )


def residual_byte_breakdown(
    *,
    k: int,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    element_bytes: int = ELEMENT_BYTES,
    index_bytes: int = INDEX_BYTES,
) -> dict[str, int]:
    kk = int(k)
    layers = int(n_layers)
    h = int(hidden)
    eb = int(element_bytes)
    ib = int(index_bytes)
    if kk < 0 or layers < 1 or h < 1 or eb < 1 or ib < 1:
        raise SparseResidualRefuse("k/layers/hidden/element_bytes/index_bytes illegal")
    value_bytes = kk * h * eb * layers
    idx_bytes = kk * ib * layers
    metadata_base = METADATA_BASE_BYTES * layers
    return {
        "k": kk,
        "n_layers": layers,
        "hidden": h,
        "element_bytes": eb,
        "index_dtype_bytes": ib,
        "value_bytes": int(value_bytes),
        "index_bytes": int(idx_bytes),
        "metadata_base_bytes": int(metadata_base),
        "metadata_bytes": int(metadata_base + idx_bytes),
        "per_coord_bytes": per_coord_bytes(
            n_layers=layers, hidden=h, element_bytes=eb, index_bytes=ib
        ),
    }


def bytes_added_from_breakdown(
    br: Mapping[str, int],
    *,
    bulk_added: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Canonical five-field ledger. Indices live in metadata, never omitted."""
    bulk = {k: 0 for k in ee.BYTES_ADDED_FIELDS}
    if bulk_added:
        for key in ee.BYTES_ADDED_FIELDS:
            bulk[key] = int(bulk_added.get(key, 0) or 0)
    added = {
        "embeddings": bulk["embeddings"],
        "generator": bulk["generator"],
        "residuals": int(br["value_bytes"]) + bulk["residuals"],
        "metadata": int(br["metadata_bytes"]) + bulk["metadata"],
        "state": bulk["state"],
    }
    added["total"] = sum(int(added[k]) for k in ee.BYTES_ADDED_FIELDS)
    return added


def extra_flops_per_output_element(k: int) -> float:
    """2*k*hidden MACs per layer, scored against the organ's output elements."""
    kk = int(k)
    if kk <= 0:
        return 0.0
    total = float(N_LAYERS) * 2.0 * float(kk) * float(HIDDEN)
    return total / float(ee.ORGAN_OUTPUT_ELEMENTS["mlp"])


def dispatch_delta_for_k(k: int, n_layers: int = N_LAYERS) -> float:
    """One gather-and-add launch per layer. Not fused away, not waved."""
    return float(n_layers) if int(k) > 0 else 0.0


def validate_billing(row: Mapping[str, Any]) -> None:
    """Load-bearing: k>0 with free indices or waved dispatch is a fabrication."""
    br = row.get("byte_breakdown") or {}
    added = row.get("bytes_added") or {}
    if not isinstance(br, Mapping) or not isinstance(added, Mapping):
        raise UnbilledResidualIndex("REFUSED: candidate is missing a byte ledger")
    k = int(row.get("k") if row.get("k") is not None else br.get("k") or 0)
    index_bytes = int(br.get("index_bytes") or 0)
    value_bytes = int(br.get("value_bytes") or 0)
    if k > 0 and index_bytes <= 0:
        raise UnbilledResidualIndex(
            "REFUSED: residual indices are not billed (index_bytes=0): fabrication"
        )
    expected_index = k * int(br.get("index_dtype_bytes") or INDEX_BYTES) * int(
        br.get("n_layers") or N_LAYERS
    )
    if k > 0 and index_bytes != expected_index:
        raise UnbilledResidualIndex(
            "REFUSED: residual indices are not billed: "
            f"index_bytes={index_bytes} != {expected_index} "
            f"(k={k} * {br.get('index_dtype_bytes') or INDEX_BYTES} * "
            f"{br.get('n_layers') or N_LAYERS})"
        )
    billed_res = int(added.get("residuals") or 0)
    if k > 0 and billed_res < value_bytes:
        raise UnbilledResidualIndex(
            "REFUSED: residual values are not billed in bytes_added.residuals "
            f"(residuals={billed_res}, value_bytes={value_bytes})"
        )
    bulk_added = row.get("bulk_bytes_added")
    expected = bytes_added_from_breakdown(
        br, bulk_added=bulk_added if isinstance(bulk_added, Mapping) else None
    )
    if k > 0 and int(added.get("metadata") or 0) != int(expected["metadata"]):
        raise UnbilledResidualIndex(
            "REFUSED: residual indices are not billed in bytes_added.metadata "
            f"(metadata={added.get('metadata')} != billed {expected['metadata']}; "
            f"index_bytes={index_bytes})"
        )
    if k > 0:
        for key in ee.BYTES_ADDED_FIELDS:
            if int(added.get(key) or 0) != int(expected[key]):
                raise UnbilledResidualIndex(
                    f"REFUSED: bytes_added[{key}]={added.get(key)} != billed {expected[key]}"
                )
    dispatch = float(row.get("dispatch_delta") if row.get("dispatch_delta") is not None else 0.0)
    if k > 0 and dispatch <= 0.0:
        raise UnbilledDispatch(
            "REFUSED: gather-and-add dispatch_delta is 0; the extra launch "
            "was waved, not billed"
        )


def residual_consumer_sketch(
    k: int,
    *,
    rematerialize_dense_W: bool = False,
    n_layers: int = N_LAYERS,
) -> dict[str, Any]:
    kk = int(k)
    if rematerialize_dense_W:
        return {
            "primitive": _require_primitive("FusedDecodeCompute"),
            "also": [],
            "algebra": "W = materialize(bulk) + scatter(W_k); y = W x",
            "consumes_directly": False,
            "rematerialize_dense_W": True,
            "runs_ordinary_gemv": True,
            "status": REJECTED_DENSE_REMAT,
            "dispatch_delta": 0.0,
            "why_dead": (
                "Rebuilding a dense W from bulk+residual then GEMV is "
                "REJECTED_DENSE_REMAT. The exception is a gathered k-row "
                "map plus scatter-add."
            ),
        }
    primitive = _require_primitive(GATHER_ADD_PRIMITIVE if kk > 0 else "TiledProjection")
    also = [_require_primitive(n) for n in GATHER_ADD_ALSO]
    ddelta = dispatch_delta_for_k(kk, n_layers=n_layers)
    return {
        "primitive": primitive,
        "also": also,
        "algebra": (
            "y = y_bulk"
            + (" + scatter(W_k x, indices)  # gather-and-add" if kk else "")
        ),
        "consumes_directly": True,
        "rematerialize_dense_W": False,
        "runs_ordinary_gemv": False,
        "status": DIRECT_CONSUME,
        "index_dtype_bytes": INDEX_BYTES,
        "index_billed": True,
        "dispatch_delta": ddelta,
        "k": kk,
        "why_not_gemv": (
            "The exception is a gathered k-row map plus scatter-add. "
            "Materializing into dense W is REJECTED_DENSE_REMAT."
        ),
        "dense_at_full_k": bool(kk >= HIDDEN),
    }


# ---------------------------------------------------------------------------
# Capability map → per-coordinate residual budget weights.
# ---------------------------------------------------------------------------


def load_capability_map() -> dict[str, Any]:
    path = REPO / MAP_REL
    if not path.is_file():
        raise CapabilityMapUnavailable(f"REFUSED: missing {MAP_REL}")
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise CapabilityMapUnavailable(f"REFUSED: {MAP_REL} is not an object")
    return doc


def _region_weight(region: Mapping[str, Any]) -> float:
    if region.get("supported"):
        return float(QUIET_WEIGHT)
    cos = region.get("layer_output_cosine")
    if cos is None:
        return 1.0
    return 1.0 + max(0.0, float(COSINE_BAR) - float(cos))


def capability_coord_weights(
    hidden: int = HIDDEN,
    *,
    layer: int = ROUND1_LAYER,
    n_blocks: int = N_BLOCKS,
    map_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-coordinate residual budget weights from the sensitivity map.

    F's output is the 5120-d down-projection. The map's only intra-MLP
    coordinate-scale measurements are the 4 gate.channel groups on L0 and
    L63. Those groups are used as an allocation PRIOR over 4 equal
    partitions of the 5120-d output — not as a claim that down-rows equal
    gate-rows. Uniform allocation ignores the prior. The map is never
    itself a byte win.
    """
    doc = dict(map_doc) if map_doc is not None else load_capability_map()
    regions = list((doc.get("allocation") or {}).get("regions") or [])
    h = int(hidden)
    nb = int(n_blocks)
    if h < nb:
        raise SparseResidualRefuse(f"hidden {h} < n_blocks {nb}")
    block_w = np.ones(nb, dtype=np.float64)
    channel_hits: list[dict[str, Any]] = []
    for rec in regions:
        rid = str(rec.get("id") or "")
        if ".mlp.gate.channel.rows_" not in rid:
            continue
        try:
            lo_s, hi_s = rid.split("rows_")[-1].split("_")
            lo_i, hi_i = int(lo_s), int(hi_s)
        except (TypeError, ValueError):
            continue
        # Map 17408-d gate rows onto nb equal output blocks.
        mid = 0.5 * (lo_i + hi_i)
        block = int(min(nb - 1, max(0, math.floor(mid / max(INTERMEDIATE, 1) * nb))))
        w = _region_weight(rec)
        channel_hits.append(
            {
                "id": rid,
                "layer": rec.get("layer"),
                "supported": bool(rec.get("supported")),
                "block": block,
                "weight": _r(w),
                "layer_output_cosine": rec.get("layer_output_cosine"),
            }
        )
    if channel_hits:
        acc = np.zeros(nb, dtype=np.float64)
        cnt = np.zeros(nb, dtype=np.float64)
        for hit in channel_hits:
            acc[int(hit["block"])] += float(hit["weight"])
            cnt[int(hit["block"])] += 1.0
        cnt = np.maximum(cnt, 1.0)
        block_w = acc / cnt
    sizes = np.full(nb, h // nb, dtype=np.int64)
    sizes[-1] = h - int(sizes[:-1].sum())
    coord_w = np.repeat(block_w, sizes[:nb])
    if coord_w.size != h:
        # last block already ate the remainder; clip/pad just in case
        if coord_w.size > h:
            coord_w = coord_w[:h]
        else:
            coord_w = np.concatenate(
                [coord_w, np.full(h - coord_w.size, float(block_w[-1]))]
            )
    licensed_mlp = int(
        ((doc.get("allocation") or {}).get("bytes_eliminated_by_organ") or {}).get(
            "mlp", 0
        )
        or 0
    )
    token_share = (
        (doc.get("answers") or {})
        .get("bytes_a_nonuniform_allocation_would_eliminate", {})
        .get("share_of_token")
    )
    return {
        "coord_weights": coord_w.astype(np.float64, copy=False),
        "block_weights": [_r(v) for v in block_w.tolist()],
        "block_sizes": [int(v) for v in sizes.tolist()],
        "n_blocks": nb,
        "hidden": h,
        "layer": int(layer),
        "channel_hits": channel_hits,
        "licensed_mlp_bytes": licensed_mlp,
        "licensed_token_share": None if token_share is None else _r(float(token_share), 9),
        "note": (
            "Block prior from mlp.gate.channel groups on sampled layers "
            "(L0, L63). Applied to 4 equal partitions of F's 5120-d output. "
            "Uniform allocation ignores this prior. The map licenses "
            f"{licensed_mlp} MLP bytes; that is an allocation prior, not a "
            "byte win."
        ),
    }


def _block_bounds(n: int, n_blocks: int) -> list[tuple[int, int]]:
    base = n // n_blocks
    bounds = []
    start = 0
    for b in range(n_blocks):
        end = n if b == n_blocks - 1 else start + base
        bounds.append((start, end))
        start = end
    return bounds


def _largest_remainder(shares: np.ndarray, k: int) -> np.ndarray:
    shares = np.asarray(shares, dtype=np.float64)
    total = float(shares.sum())
    if total <= 0.0:
        shares = np.ones_like(shares)
        total = float(shares.sum())
    raw = shares / total * int(k)
    base = np.floor(raw).astype(np.int64)
    need = int(k) - int(base.sum())
    order = np.argsort(-(raw - base))
    for i in range(max(0, need)):
        base[int(order[i])] += 1
    # If over-assigned by negative need (shouldn't), trim.
    while int(base.sum()) > int(k):
        j = int(np.argmax(base))
        if base[j] <= 0:
            break
        base[j] -= 1
    return base


def allocate_coords(
    energy: np.ndarray,
    k: int,
    *,
    policy: str,
    weights: np.ndarray | None = None,
    n_blocks: int = N_BLOCKS,
) -> np.ndarray:
    """Choose k output coordinates under an allocation policy.

    uniform: equal residual slots per equal-sized block, energy-greedy inside
    capability: slots proportional to sensitivity weights, energy-greedy inside
    energy_greedy: global top-k by residual energy (concentration ceiling)
    """
    e = np.asarray(energy, dtype=np.float64)
    n = int(e.size)
    kk = int(max(0, min(int(k), n)))
    if kk == 0:
        return np.zeros(0, dtype=np.int64)
    if policy == ENERGY_GREEDY:
        return np.argsort(e)[::-1][:kk].astype(np.int64, copy=False)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.size != n:
        raise SparseResidualRefuse(f"weights size {w.size} != energy size {n}")
    bounds = _block_bounds(n, n_blocks)
    if policy == UNIFORM:
        block_shares = np.array([float(t - s) for s, t in bounds], dtype=np.float64)
    elif policy == CAPABILITY:
        block_shares = np.array(
            [float(w[s:t].sum()) if t > s else 0.0 for s, t in bounds],
            dtype=np.float64,
        )
    else:
        raise SparseResidualRefuse(f"unknown allocation policy {policy!r}")
    kb = _largest_remainder(block_shares, kk)
    leftover = 0
    for b, (s, t) in enumerate(bounds):
        cap = t - s
        if int(kb[b]) > cap:
            leftover += int(kb[b]) - cap
            kb[b] = cap
    if leftover:
        order = np.argsort(-block_shares)
        for b in order:
            s, t = bounds[int(b)]
            room = (t - s) - int(kb[b])
            take = min(room, leftover)
            kb[b] += take
            leftover -= take
            if leftover <= 0:
                break
    parts = []
    for b, (s, t) in enumerate(bounds):
        take = int(kb[b])
        if take <= 0:
            continue
        parts.append(np.argsort(e[s:t])[::-1][:take] + s)
    if not parts:
        return np.argsort(e)[::-1][:kk].astype(np.int64, copy=False)
    idx = np.concatenate(parts).astype(np.int64, copy=False)
    _, first = np.unique(idx, return_index=True)
    idx = idx[np.sort(first)]
    if int(idx.size) < kk:
        taken = set(int(i) for i in idx.tolist())
        rest = [int(i) for i in np.argsort(e)[::-1] if int(i) not in taken]
        if rest:
            idx = np.concatenate(
                [idx, np.asarray(rest[: kk - int(idx.size)], dtype=np.int64)]
            )
    return idx[:kk]


# ---------------------------------------------------------------------------
# Concentration.
# ---------------------------------------------------------------------------


def gini(energy: np.ndarray) -> float:
    x = np.sort(np.maximum(np.asarray(energy, dtype=np.float64).reshape(-1), 0.0))
    s = float(x.sum())
    n = int(x.size)
    if n == 0 or s <= 0.0:
        return 0.0
    return float(2.0 * np.dot(np.arange(1, n + 1, dtype=np.float64), x) / (n * s) - (n + 1) / n)


def effective_n(energy: np.ndarray) -> float:
    e = np.maximum(np.asarray(energy, dtype=np.float64).reshape(-1), 0.0)
    s = float(e.sum())
    if s <= 0.0:
        return 0.0
    return float((s * s) / max(float(np.dot(e, e)), 1e-30))


def concentration_curve(
    energy: np.ndarray,
    fracs: Sequence[float] = CURVE_FRACS,
) -> list[dict[str, Any]]:
    e = np.maximum(np.asarray(energy, dtype=np.float64).reshape(-1), 0.0)
    n = int(e.size)
    total = float(e.sum())
    if n == 0:
        return []
    order = np.argsort(e)[::-1]
    c = np.cumsum(e[order])
    out = []
    for f in fracs:
        ff = float(f)
        if ff <= 0.0:
            k = 0
            captured = 0.0
        else:
            k = int(max(1, min(n, math.ceil(ff * n))))
            captured = float(c[k - 1]) / total if total > 0.0 else 0.0
        out.append(
            {
                "frac_kept": _r(ff),
                "n_kept": int(k),
                "frac_energy": _r(captured),
                "uniform_baseline": _r(float(k) / float(n)),
            }
        )
    return out


def _energy_at(curve: Sequence[Mapping[str, Any]], frac: float) -> float:
    want = float(frac)
    best = None
    for row in curve:
        if abs(float(row["frac_kept"]) - want) < 1e-12:
            return float(row["frac_energy"])
        if best is None or abs(float(row["frac_kept"]) - want) < abs(
            float(best["frac_kept"]) - want
        ):
            best = row
    return float(best["frac_energy"]) if best else 0.0


def _frac_kept_for_energy(energy: np.ndarray, captured: float) -> float:
    e = np.maximum(np.asarray(energy, dtype=np.float64).reshape(-1), 0.0)
    n = int(e.size)
    total = float(e.sum())
    if n == 0 or total <= 0.0:
        return 1.0
    c = np.cumsum(e[np.argsort(e)[::-1]])
    idx = int(np.searchsorted(c, float(captured) * total, side="left"))
    return float(min(n, idx + 1)) / float(n)


def is_dense(curve: Sequence[Mapping[str, Any]]) -> bool:
    e01 = _energy_at(curve, 0.01)
    e10 = _energy_at(curve, 0.10)
    return bool(e01 <= DENSE_ENERGY_AT_1PCT and e10 <= DENSE_ENERGY_AT_10PCT)


def summarize_axis(
    energy: np.ndarray,
    *,
    axis: str,
    ranked_on: str,
) -> dict[str, Any]:
    e = np.maximum(np.asarray(energy, dtype=np.float64).reshape(-1), 0.0)
    curve = concentration_curve(e)
    n = int(e.size)
    en = effective_n(e)
    return {
        "axis": axis,
        "ranked_on": ranked_on,
        "n": n,
        "gini": _r(gini(e)),
        "effective_n": _r(en, 3),
        "effective_n_over_n": _r(en / float(max(n, 1))),
        "frac_energy_at_1pct": _r(_energy_at(curve, 0.01)),
        "frac_energy_at_10pct": _r(_energy_at(curve, 0.10)),
        "frac_kept_for_50pct_energy": _r(_frac_kept_for_energy(e, 0.50)),
        "frac_kept_for_90pct_energy": _r(_frac_kept_for_energy(e, 0.90)),
        "dense": is_dense(curve),
        "curve": curve,
    }


def block_energy(coord_energy: np.ndarray, group: int = W_GROUP) -> np.ndarray:
    e = np.asarray(coord_energy, dtype=np.float64).reshape(-1)
    g = int(group)
    if g < 1:
        raise SparseResidualRefuse("W group size must be positive")
    n_blocks = int(e.size) // g
    if n_blocks < 1:
        return e.copy()
    return e[: n_blocks * g].reshape(n_blocks, g).sum(axis=1)


def input_direction_energy(x: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """||(X^T R)[d, :]||^2 — residual covariance energy of input dim d."""
    xt = np.ascontiguousarray(x, dtype=np.float64)
    rt = np.ascontiguousarray(residual, dtype=np.float64)
    prod = xt.T @ rt
    return np.sum(prod * prod, axis=1)


def grouped_energy(token_energy: np.ndarray, labels: Sequence[Any]) -> np.ndarray:
    if len(labels) != int(token_energy.size):
        raise SparseResidualRefuse("prompt labels do not match token residual rows")
    acc: dict[Any, float] = {}
    for e, lab in zip(token_energy.tolist(), labels):
        acc[lab] = acc.get(lab, 0.0) + float(e)
    return np.array(list(acc.values()), dtype=np.float64)


def residual_concentration(
    residual_ho: np.ndarray,
    *,
    x_ho: np.ndarray | None = None,
    residual_tr: np.ndarray | None = None,
    prompt_ids_ho: Sequence[Any] | None = None,
) -> dict[str, Any]:
    rho = np.ascontiguousarray(residual_ho, dtype=np.float64)
    out_e = np.sum(rho * rho, axis=0)
    tok_e = np.sum(rho * rho, axis=1)
    axes = {
        "output_coords": summarize_axis(out_e, axis="output_coords", ranked_on="hold"),
        "tokens": summarize_axis(tok_e, axis="tokens", ranked_on="hold"),
        "w_blocks_output": summarize_axis(
            block_energy(out_e, W_GROUP), axis="w_blocks_output", ranked_on="hold"
        ),
    }
    if x_ho is not None:
        in_e = input_direction_energy(x_ho, rho)
        axes["input_directions"] = summarize_axis(
            in_e, axis="input_directions", ranked_on="hold"
        )
        axes["w_blocks_input"] = summarize_axis(
            block_energy(in_e, W_GROUP), axis="w_blocks_input", ranked_on="hold"
        )
    if prompt_ids_ho is not None:
        axes["prompts"] = summarize_axis(
            grouped_energy(tok_e, prompt_ids_ho), axis="prompts", ranked_on="hold"
        )
    if residual_tr is not None:
        rtr = np.ascontiguousarray(residual_tr, dtype=np.float64)
        train_out = np.sum(rtr * rtr, axis=0)
        # Transferable support: rank on train energy, measure hold energy captured.
        n = int(out_e.size)
        total_ho = float(out_e.sum())
        order = np.argsort(train_out)[::-1]
        c = np.cumsum(out_e[order]) if n else np.zeros(0)
        transferred = []
        for f in CURVE_FRACS:
            k = 0 if float(f) <= 0 else int(max(1, min(n, math.ceil(float(f) * n))))
            captured = float(c[k - 1]) / total_ho if k and total_ho > 0 else 0.0
            transferred.append(
                {
                    "frac_kept": _r(float(f)),
                    "n_kept": int(k),
                    "frac_energy": _r(captured),
                    "uniform_baseline": _r(float(k) / float(max(n, 1))),
                }
            )
        axes["output_coords_train_support_on_hold"] = {
            "axis": "output_coords",
            "ranked_on": "train_applied_to_hold",
            "n": n,
            "frac_energy_at_1pct": _r(_energy_at(transferred, 0.01)),
            "frac_energy_at_10pct": _r(_energy_at(transferred, 0.10)),
            "dense": is_dense(transferred),
            "curve": transferred,
            "note": (
                "Coordinates ranked by train residual energy; energy measured "
                "on held-out residual. This is the transferable support."
            ),
        }
    dense_axes = [
        name
        for name, rec in axes.items()
        if isinstance(rec, dict) and rec.get("dense") is True
    ]
    return {
        "axes": axes,
        "output_coords_dense": bool(axes["output_coords"]["dense"]),
        "all_reported_axes_dense": bool(
            axes["output_coords"]["dense"]
            and axes["tokens"]["dense"]
            and axes.get("input_directions", {}).get("dense", True)
            and axes["w_blocks_output"]["dense"]
        ),
        "dense_axes": dense_axes,
    }


# ---------------------------------------------------------------------------
# Residual application (oracle ceiling + fitted linear gather-and-add).
# ---------------------------------------------------------------------------


def oracle_correct(pred: np.ndarray, target: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.array(pred, dtype=np.float32, copy=True)
    if int(idx.size) == 0:
        return out
    out[:, idx] = np.asarray(target, dtype=np.float32)[:, idx]
    return out


def factor_gram(x: np.ndarray, *, lam: float = RIDGE_LAM) -> dict[str, np.ndarray]:
    xt = np.ascontiguousarray(x, dtype=np.float64)
    g = xt.T @ xt
    n = int(g.shape[0])
    g.flat[:: n + 1] += float(lam)
    try:
        l = np.linalg.cholesky(g)
    except np.linalg.LinAlgError:
        g.flat[:: n + 1] += float(lam)
        l = np.linalg.cholesky(g)
    return {"L": l, "X": xt}


def solve_residual_map(factor: Mapping[str, np.ndarray], r_cols: np.ndarray) -> np.ndarray:
    """Cholesky back-substitution, not a general solve.

    factor["L"] comes from np.linalg.cholesky, so it is LOWER TRIANGULAR. Passing
    it to np.linalg.solve asks LAPACK for a general LU factorization with
    pivoting -- O(n^3) work to solve a system that back-substitution answers in
    O(n^2), twice per call. That was 31.0s of a 69.4s build across 144 solves.

    solve_triangular is the algorithm the factorization was computed FOR. It is
    not an approximation: same system, same solution, correct routine. The
    equivalence that matters was measured on the emitted receipt, not on the
    intermediate.
    """
    rhs = factor["X"].T @ np.ascontiguousarray(r_cols, dtype=np.float64)
    z = solve_triangular(factor["L"], rhs, lower=True, check_finite=False)
    return solve_triangular(factor["L"].T, z, lower=False, check_finite=False)


def apply_linear_residual(
    pred: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    out = np.array(pred, dtype=np.float32, copy=True)
    if int(idx.size) == 0:
        return out
    delta = np.ascontiguousarray(x, dtype=np.float64) @ np.ascontiguousarray(w, dtype=np.float64)
    out[:, idx] = (out[:, idx].astype(np.float64) + delta).astype(np.float32)
    return out


def apply_mean_residual(
    pred: np.ndarray,
    bias: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    out = np.array(pred, dtype=np.float32, copy=True)
    if int(idx.size) == 0:
        return out
    out[:, idx] = (out[:, idx].astype(np.float64) + np.asarray(bias, dtype=np.float64)).astype(
        np.float32
    )
    return out


def min_k_oracle_to_clear(
    pred: np.ndarray,
    target: np.ndarray,
    energy: np.ndarray,
    *,
    kill: float = HELD_OUT_KILL_REL,
) -> int:
    """Smallest energy-ranked k whose oracle correction falls below kill."""
    n = int(pred.shape[1])
    lo, hi = 0, n
    best = n
    while lo <= hi:
        mid = (lo + hi) // 2
        idx = allocate_coords(energy, mid, policy=ENERGY_GREEDY)
        rel = mean_l2_ratio(oracle_correct(pred, target, idx), target)
        if rel < float(kill):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(best)


# ---------------------------------------------------------------------------
# Economics wrapper. Every point goes through executable_economics.score.
# ---------------------------------------------------------------------------


def _economics(
    *,
    bytes_removed: int,
    bytes_added: Mapping[str, int],
    consuming_primitive: str,
    status: str,
    candidate_id: str,
    extra_flops_per_output_element: float,
    dispatch_delta: float,
) -> dict[str, Any]:
    scored = ee.score(
        bytes_removed=int(bytes_removed),
        bytes_added={k: int(bytes_added.get(k, 0)) for k in ee.BYTES_ADDED_FIELDS},
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        dispatch_delta=float(dispatch_delta),
        consuming_primitive=consuming_primitive,
        organ="mlp",
        stream_class="weight_codes",
        reusable_family=True,
        high_information_falsifier=True,
        status=status,
        candidate_id=candidate_id,
    )
    s20 = scored["s020_section_20"]
    assumptions = scored["assumptions"]
    return {
        "id": candidate_id,
        "status": scored["status"],
        "live": scored["live"],
        "verdict": scored["verdict"],
        "verdict_reasons": list(scored["verdict_reasons"]),
        "bytes_removed": scored["bytes_removed"],
        "bytes_added": {k: int(scored["bytes_added"].get(k, 0)) for k in ee.BYTES_ADDED_FIELDS},
        "bytes_added_total": int(scored["bytes_added"].get("total", 0)),
        "net_bytes": scored["net_bytes"],
        "consuming_primitive": scored["consuming_primitive"],
        "extra_flops_per_output_element": scored["extra_flops_per_output_element"],
        "dispatch_delta": scored["dispatch_delta"],
        "predicted_ms_delta": _r(scored["predicted_ms_delta"], 4),
        "predicted_ms_saved": _r(scored["predicted_ms_saved"], 4),
        "predicted_token_ms": _r(scored["predicted_token_ms"], 4),
        "predicted_tps": _r(scored["predicted_tps"], 3),
        "predicted_ms_delta_range": [
            _r(scored["predicted_ms_delta_range"][0], 4),
            _r(scored["predicted_ms_delta_range"][1], 4),
        ],
        "terms": {k: _r(v, 4) for k, v in scored["terms"].items()},
        "assumptions": {
            "bandwidth_regime": assumptions["bandwidth_regime"],
            "bandwidth_gb_s_nominal": _r(assumptions["bandwidth_gb_s_nominal"], 2),
            "bandwidth_gb_s_range": [
                _r(assumptions["bandwidth_gb_s_range"][0], 2),
                _r(assumptions["bandwidth_gb_s_range"][1], 2),
            ],
            "bandwidth_is_assumption": assumptions["bandwidth_is_assumption"],
            "bandwidth_note": assumptions["bandwidth_note"],
            "dispatch_class": assumptions["dispatch_class"],
            "dispatch_note": assumptions["dispatch_note"],
            "element_bytes": ELEMENT_BYTES,
            "index_dtype_bytes": INDEX_BYTES,
            "index_note": (
                "uint32 gather indices billed in bytes_added.metadata. "
                "A free index is a fabrication."
            ),
            "dispatch_delta_note": (
                "ASSUMPTION: one extra DirectRoutedAccumulate launch per "
                "layer (gather-and-add). Not fused away."
            ),
        },
        "s020_section_20": {
            "bar_ms": _r(s20["bar_ms"], 4),
            "plausible_ms_saved": _r(s20["plausible_ms_saved"], 4),
            "clears_time_bar": s20["clears_time_bar"],
            "reusable_family": s20["reusable_family"],
            "high_information_falsifier": s20["high_information_falsifier"],
        },
        "task_material_bar": {
            "ms_bar": MATERIAL_MS_BAR,
            "byte_frac_bar": MATERIAL_BYTE_FRAC,
            "note": (
                "MATERIAL means >= 1 ms of token time, or >= 5% of organ "
                "bytes, or a reusable family, or a decisive falsifier. "
                "Quoted from the lane contract over cited organ times; "
                "not a hardware measurement."
            ),
        },
    }


def emit_point(
    *,
    bulk_id: str,
    allocation: str,
    k: int,
    pred_ho: np.ndarray,
    y_ho: np.ndarray,
    pred_tr: np.ndarray,
    y_tr: np.ndarray,
    combined_ho: np.ndarray,
    oracle_ho: np.ndarray,
    mean_ho: np.ndarray,
    bulk_held_out: float,
    bulk_added: Mapping[str, int],
    n_layers: int,
    hidden: int,
    billed: bool,
    budget_name: str,
    mlp_frac_requested: float,
) -> dict[str, Any]:
    """The only constructor a residual sweep row may pass through."""
    br = residual_byte_breakdown(k=k, n_layers=n_layers, hidden=hidden)
    added = bytes_added_from_breakdown(br, bulk_added=bulk_added)
    ddelta = dispatch_delta_for_k(k, n_layers=n_layers)
    # Organ-scored extra FLOPs only when the geometry is the sealed MLP.
    flops = extra_flops_per_output_element(k) if int(hidden) == int(HIDDEN) else 0.0
    consumer = residual_consumer_sketch(k, n_layers=n_layers)
    ho = function_error(combined_ho, y_ho, split="hold", report_as="held_out")
    oracle = function_error(oracle_ho, y_ho, split="hold", report_as="held_out")
    mean_e = function_error(mean_ho, y_ho, split="hold", report_as="held_out")
    tr = function_error(pred_tr, y_tr, split="train", report_as="train")
    held = float(ho["held_out_relative_l2"])
    status = MEASURED_NEGATIVE if held >= HELD_OUT_KILL_REL else OPEN
    residual_bytes = int(br["value_bytes"] + br["index_bytes"])
    drop = float(bulk_held_out) - held
    oracle_drop = float(bulk_held_out) - float(oracle["held_out_relative_l2"])
    cid = f"{bulk_id}_{allocation}_k{k}"
    row: dict[str, Any] = {
        "id": cid,
        "bulk_id": bulk_id,
        "allocation": allocation,
        "budget_name": budget_name,
        "mlp_frac_requested": _r(float(mlp_frac_requested), 9),
        "mlp_frac_billed": _r(mlp_frac_of_k(k, n_layers=n_layers, hidden=hidden), 9),
        "k": int(k),
        "byte_breakdown": dict(br),
        "bytes_added": added,
        "dispatch_delta": ddelta,
        "bulk_bytes_added": {key: int(bulk_added.get(key, 0) or 0) for key in ee.BYTES_ADDED_FIELDS},
        "extra_flops_per_output_element": _r(flops, 6),
        "consumer": dict(consumer),
        "consumer_status": consumer["status"],
        "status": status,
        "error_authority": "held_out_relative_l2",
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "bulk_held_out_relative_l2": _r(bulk_held_out),
        "oracle_held_out_relative_l2": oracle["held_out_relative_l2"],
        "oracle_error_drop": _r(oracle_drop),
        "mean_residual_held_out_relative_l2": mean_e["held_out_relative_l2"],
        "error_drop": _r(drop),
        "residual_bytes": residual_bytes,
        "error_drop_per_residual_byte": (
            None if residual_bytes <= 0 else _r(drop / float(residual_bytes), 12)
        ),
        "oracle_error_drop_per_residual_byte": (
            None if residual_bytes <= 0 else _r(oracle_drop / float(residual_bytes), 12)
        ),
        "capability_restored_per_residual_byte_note": (
            "error_drop is held-out relative L2 of F, not downstream "
            "capability. The sensitivity map allocated the budget; it is "
            "not itself a byte win."
        ),
        "billed": bool(billed),
    }
    row.update(ho)
    row.update({k: tr[k] for k in tr if k not in row})
    validate_billing(row)
    validate_error_authority(row)
    if billed:
        row["economics"] = _economics(
            bytes_removed=ee.MLP_ACTIVE_BYTES,
            bytes_added=added,
            consuming_primitive=str(consumer["primitive"]),
            status=status,
            candidate_id=cid,
            extra_flops_per_output_element=flops,
            dispatch_delta=ddelta,
        )
        if status != OPEN:
            open_econ = _economics(
                bytes_removed=ee.MLP_ACTIVE_BYTES,
                bytes_added=added,
                consuming_primitive=str(consumer["primitive"]),
                status=OPEN,
                candidate_id=cid,
                extra_flops_per_output_element=flops,
                dispatch_delta=ddelta,
            )
            row["economics_if_function_held"] = {
                "verdict": open_econ["verdict"],
                "predicted_ms_saved": open_econ["predicted_ms_saved"],
                "clears_time_bar": open_econ["s020_section_20"]["clears_time_bar"],
                "net_bytes": open_econ["net_bytes"],
            }
    else:
        row["economics"] = None
        row["economics_note"] = (
            "oracle PCA peeks at Y; it is a control on the output manifold, "
            "not a consuming program. Not scored as a function replacement."
        )
    return _py(row)


# ---------------------------------------------------------------------------
# Bulks already on record.
# ---------------------------------------------------------------------------


def fit_recorded_bulk(
    bulk_id: str,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xho: np.ndarray,
    Yho: np.ndarray,
    *,
    rank: int = BULK_RANK,
) -> dict[str, Any]:
    rr = int(rank)
    if bulk_id.startswith("shared_input"):
        fit = msp.fit_shared_input(Xtr, Ytr, Xho, Yho, rank=rr, program="linear")
        billed = True
        shape = msp.SHARED_INPUT
    elif bulk_id.startswith("shared_output"):
        fit = msp.fit_shared_output(Xtr, Ytr, Xho, Yho, rank=rr)
        billed = True
        shape = msp.SHARED_OUTPUT
    elif bulk_id.startswith("shared_both"):
        fit = msp.fit_shared_both(Xtr, Ytr, Xho, Yho, rank=rr, residual_k=0)
        billed = True
        shape = msp.SHARED_BOTH
    elif bulk_id.startswith("oracle_pca"):
        b = msp.randomized_basis(Ytr, rr, seed=msp.RNG_SEED + 7)
        fit = {
            "shape": "ORACLE_PCA",
            "program": "pca_of_F",
            "rank_in": rr,
            "rank_out": rr,
            "residual_k": 0,
            "pred_tr": ((Ytr @ b) @ b.T).astype(np.float32, copy=False),
            "pred_ho": ((Yho @ b) @ b.T).astype(np.float32, copy=False),
            "algebra": "y ≈ (y B) B^T  # peeks at Y; not a program of x",
        }
        billed = False
        shape = "ORACLE_PCA"
    else:
        raise SparseResidualRefuse(f"unknown bulk {bulk_id!r}")
    ho = function_error(fit["pred_ho"], Yho, split="hold", report_as="held_out")
    tr = function_error(fit["pred_tr"], Ytr, split="train", report_as="train")
    validate_error_authority(ho)
    bulk_added = {k: 0 for k in ee.BYTES_ADDED_FIELDS}
    bulk_added["total"] = 0
    if billed:
        br = msp.byte_breakdown(
            shape=shape, rank_in=rr, rank_out=rr, residual_k=0
        )
        bulk_added = msp.bytes_added_from_breakdown(br)
    return {
        "id": bulk_id,
        "shape": shape,
        "billed": billed,
        "rank": rr,
        "program": fit.get("program"),
        "algebra": fit.get("algebra"),
        "pred_tr": fit["pred_tr"],
        "pred_ho": fit["pred_ho"],
        "held_out_relative_l2": ho["held_out_relative_l2"],
        "held_out_split": ho["held_out_split"],
        "train_relative_l2_diagnostic": tr["train_relative_l2_diagnostic"],
        "bytes_added": bulk_added,
        "error_authority": "held_out_relative_l2",
    }


# ---------------------------------------------------------------------------
# Pack + measure.
# ---------------------------------------------------------------------------


def load_pack(
    layer: int = ROUND1_LAYER,
    *,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    pack = msp.load_layer_split(layer, payload_dir=payload_dir)
    root = Path(pack["payload_dir"])
    rows_path = root / "rows.jsonl"
    train_prompts_row: list[str] = []
    hold_prompts_row: list[str] = []
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["layer"]) != int(layer):
                continue
            if row.get("synthetic"):
                raise CorpusUnavailable("REFUSED: SYNTHETIC_ROW in teacher payload (NNS-001)")
            split = str(row.get("split") or "")
            prompt = str(row["prompt_id"])
            if split == "train":
                train_prompts_row.append(prompt)
            elif split == "hold":
                hold_prompts_row.append(prompt)
    if len(train_prompts_row) != int(pack["n_train"]) or len(hold_prompts_row) != int(
        pack["n_hold"]
    ):
        raise CorpusUnavailable("REFUSED: prompt-per-row counts do not match the split")
    leaked = set(pack["train_prompt_ids"]) & set(pack["hold_prompt_ids"])
    if leaked:
        raise CorpusUnavailable(f"REFUSED: HELD_OUT_PROMPT_LEAK {sorted(leaked)[:8]}")
    pack["train_prompt_per_row"] = train_prompts_row
    pack["hold_prompt_per_row"] = hold_prompts_row
    pack["split_unit"] = "prompt_id"
    return pack


def named_k_list(
    *,
    hidden: int,
    n_layers: int,
    extra_k: Sequence[int] = (0, 1),
) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for name, frac in NAMED_BUDGETS:
        k = k_from_mlp_frac(frac, n_layers=n_layers, hidden=hidden)
        if k in seen and name != "bulk_only":
            # Still record the name against the existing k.
            for rec in out:
                if rec["k"] == k:
                    rec.setdefault("also_names", []).append(name)
                    rec.setdefault("also_fracs", []).append(_r(float(frac), 9))
                    break
            continue
        seen.add(k)
        out.append(
            {
                "name": name,
                "mlp_frac_requested": float(frac),
                "k": int(k),
            }
        )
    for k in extra_k:
        kk = int(max(0, min(int(hidden), int(k))))
        if kk in seen:
            continue
        seen.add(kk)
        out.append(
            {
                "name": f"k{kk}",
                "mlp_frac_requested": mlp_frac_of_k(kk, n_layers=n_layers, hidden=hidden),
                "k": kk,
            }
        )
    out.sort(key=lambda r: (int(r["k"]), str(r["name"])))
    return out


def measure_bulk(
    bulk: Mapping[str, Any],
    *,
    pack: Mapping[str, Any],
    factor: Mapping[str, np.ndarray] | None,
    cap: Mapping[str, Any],
    budgets: Sequence[Mapping[str, Any]],
    allocations: Sequence[str] = ALLOCATIONS,
    n_layers: int = N_LAYERS,
) -> dict[str, Any]:
    Xtr, Ytr = pack["Xtr"], pack["Ytr"]
    Xho, Yho = pack["Xho"], pack["Yho"]
    pred_tr = bulk["pred_tr"]
    pred_ho = bulk["pred_ho"]
    rtr = Ytr.astype(np.float32, copy=False) - pred_tr
    rho = Yho.astype(np.float32, copy=False) - pred_ho
    conc = residual_concentration(
        rho,
        x_ho=Xho,
        residual_tr=rtr,
        prompt_ids_ho=pack.get("hold_prompt_per_row"),
    )
    train_energy = np.sum(np.square(rtr.astype(np.float64)), axis=0)
    hold_energy = np.sum(np.square(rho.astype(np.float64)), axis=0)
    min_k_train = min_k_oracle_to_clear(pred_ho, Yho, train_energy)
    min_k_hold = min_k_oracle_to_clear(pred_ho, Yho, hold_energy)
    hidden = int(Yho.shape[1])
    weights = cap["coord_weights"]
    if int(weights.size) != hidden:
        raise SparseResidualRefuse("capability weights do not match hidden")
    bulk_held = float(bulk["held_out_relative_l2"])
    sweep: list[dict[str, Any]] = []
    for rec in budgets:
        k = int(rec["k"])
        for policy in allocations:
            idx = allocate_coords(
                train_energy, k, policy=policy, weights=weights, n_blocks=int(cap["n_blocks"])
            )
            oracle_ho = oracle_correct(pred_ho, Yho, idx)
            if k == 0:
                combined_ho = np.array(pred_ho, dtype=np.float32, copy=True)
                mean_ho = combined_ho
            else:
                bias = rtr[:, idx].astype(np.float64).mean(axis=0)
                mean_ho = apply_mean_residual(pred_ho, bias, idx)
                if factor is None:
                    w = msp.ridge_map(Xtr, rtr[:, idx], lam=RIDGE_LAM)
                else:
                    w = solve_residual_map(factor, rtr[:, idx])
                combined_ho = apply_linear_residual(pred_ho, Xho, w, idx)
            sweep.append(
                emit_point(
                    bulk_id=str(bulk["id"]),
                    allocation=policy,
                    k=k,
                    pred_ho=pred_ho,
                    y_ho=Yho,
                    pred_tr=pred_tr,
                    y_tr=Ytr,
                    combined_ho=combined_ho,
                    oracle_ho=oracle_ho,
                    mean_ho=mean_ho,
                    bulk_held_out=bulk_held,
                    bulk_added=bulk["bytes_added"],
                    n_layers=n_layers,
                    hidden=hidden,
                    billed=bool(bulk["billed"]),
                    budget_name=str(rec["name"]),
                    mlp_frac_requested=float(rec["mlp_frac_requested"]),
                )
            )
    # Allocation value of the map: capability vs uniform error_drop at each k.
    map_value = []
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in sweep:
        by_key[(int(row["k"]), str(row["allocation"]))] = row
    for rec in budgets:
        k = int(rec["k"])
        u = by_key.get((k, UNIFORM))
        c = by_key.get((k, CAPABILITY))
        if not u or not c:
            continue
        map_value.append(
            {
                "k": k,
                "budget_name": rec["name"],
                "uniform_held_out_relative_l2": u["held_out_relative_l2"],
                "capability_held_out_relative_l2": c["held_out_relative_l2"],
                "delta_held_out_relative_l2": _r(
                    float(u["held_out_relative_l2"]) - float(c["held_out_relative_l2"])
                ),
                "uniform_oracle_held_out_relative_l2": u["oracle_held_out_relative_l2"],
                "capability_oracle_held_out_relative_l2": c["oracle_held_out_relative_l2"],
                "delta_oracle_held_out_relative_l2": _r(
                    float(u["oracle_held_out_relative_l2"])
                    - float(c["oracle_held_out_relative_l2"])
                ),
                "note": (
                    "Positive delta means capability allocation restored more "
                    "held-out L2 than uniform at the same residual byte budget."
                ),
            }
        )
    k5 = k_from_mlp_frac(MATERIAL_BYTE_FRAC, n_layers=n_layers, hidden=hidden)
    idx5 = allocate_coords(train_energy, k5, policy=ENERGY_GREEDY)
    oracle5 = function_error(
        oracle_correct(pred_ho, Yho, idx5), Yho, split="hold", report_as="held_out"
    )
    return _py(
        {
            "id": bulk["id"],
            "shape": bulk["shape"],
            "billed": bulk["billed"],
            "rank": bulk["rank"],
            "program": bulk["program"],
            "algebra": bulk["algebra"],
            "held_out_relative_l2": bulk["held_out_relative_l2"],
            "held_out_split": "hold",
            "train_relative_l2_diagnostic": bulk["train_relative_l2_diagnostic"],
            "error_authority": "held_out_relative_l2",
            "bytes_added_bulk": bulk["bytes_added"],
            "concentration": conc,
            "min_k_oracle_train_support_to_clear_kill": int(min_k_train),
            "min_k_oracle_hold_support_to_clear_kill": int(min_k_hold),
            "min_mlp_frac_oracle_train_support": _r(
                mlp_frac_of_k(min_k_train, n_layers=n_layers, hidden=hidden), 9
            ),
            "min_frac_coords_oracle_train_support": _r(min_k_train / float(max(hidden, 1))),
            "oracle_at_5pct_mlp_bytes": {
                "k": int(k5),
                "mlp_frac": MATERIAL_BYTE_FRAC,
                "allocation": ENERGY_GREEDY,
                "held_out_relative_l2": oracle5["held_out_relative_l2"],
                "held_out_split": "hold",
                "clears_kill": bool(float(oracle5["held_out_relative_l2"]) < HELD_OUT_KILL_REL),
                "note": (
                    "Energy-greedy oracle correction at the 5% MLP-byte MATERIAL "
                    "bar. Ceiling on any sparse residual of that size. Support "
                    "ranked on train residual energy."
                ),
            },
            "map_value_vs_uniform": map_value,
            "budget_sweep": sweep,
        }
    )


def measure(
    *,
    pack: Mapping[str, Any],
    cap: Mapping[str, Any] | None = None,
    bulk_ids: Sequence[str] | None = None,
    n_layers: int = N_LAYERS,
    budgets: Sequence[Mapping[str, Any]] | None = None,
    fit_gram: bool = True,
    rank: int = BULK_RANK,
) -> dict[str, Any]:
    hidden = int(pack["Yho"].shape[1])
    cap_doc = cap if cap is not None else capability_coord_weights(hidden, layer=int(pack["layer"]))
    ids = list(bulk_ids) if bulk_ids is not None else [b[0] for b in BILLED_BULKS] + [ORACLE_PCA_ID]
    budget_rows = (
        list(budgets)
        if budgets is not None
        else named_k_list(hidden=hidden, n_layers=n_layers)
    )
    factor = factor_gram(pack["Xtr"]) if fit_gram else None
    bulks = []
    for bid in ids:
        fitted = fit_recorded_bulk(
            bid, pack["Xtr"], pack["Ytr"], pack["Xho"], pack["Yho"], rank=rank
        )
        bulks.append(
            measure_bulk(
                fitted,
                pack=pack,
                factor=factor,
                cap=cap_doc,
                budgets=budget_rows,
                n_layers=n_layers,
            )
        )
    billed = [b for b in bulks if b["billed"]]
    output_dense = all(b["concentration"]["output_coords_dense"] for b in billed) if billed else True
    all_axes_dense = (
        all(b["concentration"]["all_reported_axes_dense"] for b in billed) if billed else True
    )
    oracle5_dead = all(
        not b["oracle_at_5pct_mlp_bytes"]["clears_kill"] for b in billed
    ) if billed else True
    min_frac = max(
        (float(b["min_frac_coords_oracle_train_support"]) for b in billed),
        default=1.0,
    )
    if output_dense and oracle5_dead:
        school = RESIDUAL_DENSE_CLOSED
        why = (
            "Held-out residual energy is spread across output coordinates "
            "(concentration curve near uniform). Energy-greedy oracle "
            "correction at 5% of MLP bytes still sits above the 0.25 "
            "held-out kill. No sparse gather-and-add exception can rescue "
            "these bulk programs for this organ."
        )
    elif oracle5_dead:
        school = RESIDUAL_RESCUE_CLOSED
        why = (
            "Some concentration exists, but even an oracle residual at the "
            "5% MLP-byte MATERIAL bar cannot push held-out relative L2 below "
            "0.25. A residual large enough to carry F is no longer sparse."
        )
    else:
        school = OPEN
        why = (
            "Energy-greedy oracle correction at 5% of MLP bytes clears the "
            "held-out kill on at least one bulk. Whether a fitted gather-and-add "
            "program realises that ceiling is in the budget sweep."
        )
    return _py(
        {
            "layer": int(pack["layer"]),
            "n_train": int(pack["n_train"]),
            "n_hold": int(pack["n_hold"]),
            "n_train_prompts": len(pack["train_prompt_ids"]),
            "n_hold_prompts": len(pack["hold_prompt_ids"]),
            "split_unit": "prompt_id",
            "disjoint": True,
            "hidden": hidden,
            "n_layers_billed": int(n_layers),
            "payload_dir": pack.get("payload_dir"),
            "x_sha256": pack.get("x_sha256"),
            "y_sha256": pack.get("y_sha256"),
            "capability_allocation": {
                k: cap_doc[k]
                for k in (
                    "block_weights",
                    "block_sizes",
                    "n_blocks",
                    "licensed_mlp_bytes",
                    "licensed_token_share",
                    "note",
                    "channel_hits",
                    "layer",
                )
                if k in cap_doc
            },
            "budgets": budget_rows,
            "bulks": bulks,
            "school": {
                "status": school,
                "why": why,
                "output_coords_dense_on_every_billed_bulk": output_dense,
                "all_reported_axes_dense_on_every_billed_bulk": all_axes_dense,
                "oracle_at_5pct_mlp_bytes_clears_kill_any_billed": (not oracle5_dead),
                "max_min_frac_coords_oracle_train_support": _r(min_frac),
                "held_out_kill_rel": HELD_OUT_KILL_REL,
                "dense_energy_at_1pct_bar": DENSE_ENERGY_AT_1PCT,
                "dense_energy_at_10pct_bar": DENSE_ENERGY_AT_10PCT,
            },
        }
    )


# ---------------------------------------------------------------------------
# Fixtures (not a teacher-corpus stand-in; NNS-001).
# ---------------------------------------------------------------------------


def make_fixture(
    n_train: int = 40,
    n_hold: int = 12,
    hidden: int = 16,
    rank: int = 3,
    sparse_k: int = 2,
    seed: int = 0,
    dense: bool = False,
) -> dict[str, Any]:
    """Tiny bulk-plus-exception. Not a teacher-corpus stand-in (NNS-001)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((hidden, rank)).astype(np.float32)
    p = rng.standard_normal((hidden, rank)).astype(np.float32)
    x_tr = rng.standard_normal((n_train, hidden)).astype(np.float32)
    x_ho = rng.standard_normal((n_hold, hidden)).astype(np.float32)
    y_tr = (x_tr @ v) @ p.T
    y_ho = (x_ho @ v) @ p.T
    if dense:
        y_tr = y_tr + rng.standard_normal(y_tr.shape).astype(np.float32)
        y_ho = y_ho + rng.standard_normal(y_ho.shape).astype(np.float32)
    else:
        w_s = rng.standard_normal((hidden, sparse_k)).astype(np.float32)
        y_tr[:, :sparse_k] += x_tr @ w_s
        y_ho[:, :sparse_k] += x_ho @ w_s
    train_prompts = [f"tr:{i % max(2, n_train // 8)}" for i in range(n_train)]
    hold_prompts = [f"ho:{i % max(2, n_hold // 4)}" for i in range(n_hold)]
    return {
        "layer": 0,
        "Xtr": x_tr,
        "Ytr": y_tr,
        "Xho": x_ho,
        "Yho": y_ho,
        "n_train": n_train,
        "n_hold": n_hold,
        "train_prompt_ids": sorted(set(train_prompts)),
        "hold_prompt_ids": sorted(set(hold_prompts)),
        "train_prompt_per_row": train_prompts,
        "hold_prompt_per_row": hold_prompts,
        "sparse_k": sparse_k,
        "dense": dense,
        "payload_dir": None,
        "x_sha256": None,
        "y_sha256": None,
    }


def fixture_capability_weights(hidden: int, n_blocks: int = N_BLOCKS) -> dict[str, Any]:
    h = int(hidden)
    nb = int(n_blocks)
    block_w = np.ones(nb, dtype=np.float64)
    block_w[-1] = QUIET_WEIGHT
    sizes = np.full(nb, h // nb, dtype=np.int64)
    sizes[-1] = h - int(sizes[:-1].sum())
    coord_w = np.repeat(block_w, sizes)
    if coord_w.size > h:
        coord_w = coord_w[:h]
    elif coord_w.size < h:
        coord_w = np.concatenate([coord_w, np.full(h - coord_w.size, float(block_w[-1]))])
    return {
        "coord_weights": coord_w,
        "block_weights": [_r(v) for v in block_w.tolist()],
        "block_sizes": [int(v) for v in sizes.tolist()],
        "n_blocks": nb,
        "hidden": h,
        "layer": 0,
        "channel_hits": [],
        "licensed_mlp_bytes": 0,
        "licensed_token_share": 0.0,
        "note": "fixture prior: last block quiet",
    }


# ---------------------------------------------------------------------------
# Negative index + selftest + receipt.
# ---------------------------------------------------------------------------


def consult_index() -> dict[str, Any]:
    model = "qwen3.8-27b"
    organ = "mlp"
    families = (
        "function_replacement",
        "shared_input_transforms",
        "factorized_programs",
        "residual_codebook",
        "low_rank",
        "synthetic_activation",
    )
    queries = []
    refusals = []
    for family in families:
        hits = ni.query(model=model, organ=organ, hypothesis_family=family)
        queries.append(
            {
                "model": model,
                "organ": organ,
                "hypothesis_family": family,
                "n_hits": len(hits),
                "top": [
                    {
                        "scar_id": h.get("scar_id"),
                        "level": h.get("level"),
                        "hypothesis_family": h.get("hypothesis_family"),
                        "verdict": h.get("verdict"),
                    }
                    for h in hits[:3]
                ],
            }
        )
        refusal = ni.refuse_if_dead(
            {"model": model, "organ": organ, "hypothesis_family": family}
        )
        if refusal is not None:
            refusals.append(
                {
                    "hypothesis_family": family,
                    "scar_id": refusal.get("scar_id"),
                    "level": refusal.get("level"),
                    "reason": refusal.get("reason"),
                }
            )
    proposal_families = ("function_replacement",)
    proposal_refused = [r for r in refusals if r["hypothesis_family"] in proposal_families]
    return {
        "model": model,
        "organ": organ,
        "queries": queries,
        "refusals": refusals,
        "proposal_refused": proposal_refused,
        "proceed": len(proposal_refused) == 0,
        "cousins_not_this_object": [
            "residual_codebook / additive_residual_codebook_q2x2 is a W-space "
            "codebook, not a gather-and-add residual of F after a bulk program.",
            "COMPOSITE_MLP_SIMPLE_LINEAR_LOW_RANK_REFUTED is the bulk linear "
            "school this module does not rebuild; it measures the residual after it.",
        ],
        "note": (
            "This experiment is residual-rescue of already-refuted linear "
            "bulks, not a new linear basis. synthetic_activation is a method "
            "scar: this module refuses to fit on Gaussian X as the corpus."
        ),
    }


def selftest() -> dict[str, Any]:
    held_out_leak_refused = False
    try:
        y = np.ones((4, 3), dtype=np.float32)
        function_error(y, y, split="train", report_as="held_out")
    except TrainReportedAsHeldOut:
        held_out_leak_refused = True

    unbilled_index_refused = False
    br = residual_byte_breakdown(k=4, n_layers=2, hidden=16)
    added = bytes_added_from_breakdown(br)
    validate_billing(
        {"k": 4, "byte_breakdown": br, "bytes_added": added, "dispatch_delta": 2.0}
    )
    stolen = dict(added)
    stolen["metadata"] = int(br["metadata_base_bytes"])
    stolen["total"] = sum(int(stolen[k]) for k in ee.BYTES_ADDED_FIELDS)
    try:
        validate_billing(
            {
                "k": 4,
                "byte_breakdown": br,
                "bytes_added": stolen,
                "dispatch_delta": 2.0,
            }
        )
    except UnbilledResidualIndex:
        unbilled_index_refused = True

    zero_br = dict(br)
    zero_br["index_bytes"] = 0
    unbilled_zero_refused = False
    try:
        validate_billing(
            {
                "k": 4,
                "byte_breakdown": zero_br,
                "bytes_added": added,
                "dispatch_delta": 2.0,
            }
        )
    except UnbilledResidualIndex:
        unbilled_zero_refused = True

    dispatch_refused = False
    try:
        validate_billing(
            {
                "k": 4,
                "byte_breakdown": br,
                "bytes_added": added,
                "dispatch_delta": 0.0,
            }
        )
    except UnbilledDispatch:
        dispatch_refused = True

    fx = make_fixture(dense=False, hidden=16, sparse_k=2)
    # Zero bulk so the fixture's 2-coord exception is the whole residual.
    zero_pred = np.zeros_like(fx["Yho"])
    residual = fx["Yho"] - zero_pred
    conc = residual_concentration(residual, x_ho=fx["Xho"], residual_tr=fx["Ytr"] - np.zeros_like(fx["Ytr"]))
    sparse_ok = conc["output_coords_dense"] is False

    fx_d = make_fixture(dense=True, hidden=16, seed=1)
    conc_d = residual_concentration(
        fx_d["Yho"] - np.zeros_like(fx_d["Yho"]),
        x_ho=fx_d["Xho"],
        residual_tr=fx_d["Ytr"] - np.zeros_like(fx_d["Ytr"]),
    )
    # A 16-d isotropic residual is noisy; density is still well-defined.

    if not (
        held_out_leak_refused
        and unbilled_index_refused
        and unbilled_zero_refused
        and dispatch_refused
        and sparse_ok
    ):
        raise SystemExit(
            "selftest: guards did not fire "
            f"leak={held_out_leak_refused} unbilled_meta={unbilled_index_refused} "
            f"unbilled_zero={unbilled_zero_refused} dispatch={dispatch_refused} "
            f"sparse_ok={sparse_ok}"
        )
    return {
        "held_out_leak_refused": True,
        "unbilled_residual_index_refused": True,
        "unbilled_index_bytes_zero_refused": True,
        "unbilled_dispatch_refused": True,
        "sparse_fixture_not_dense": True,
        "dense_fixture_gini": conc_d["axes"]["output_coords"]["gini"],
        "held_out_leak_codes": ["TrainReportedAsHeldOut"],
        "unbilled_codes": ["UnbilledResidualIndex", "UnbilledDispatch"],
    }


def _answers(measured: Mapping[str, Any]) -> dict[str, str]:
    school = measured["school"]
    billed = [b for b in measured["bulks"] if b["billed"]]
    best = min(billed, key=lambda b: float(b["held_out_relative_l2"])) if billed else None
    oc = (best or {}).get("concentration", {}).get("axes", {}).get("output_coords", {})
    inn = (best or {}).get("concentration", {}).get("axes", {}).get("input_directions", {})
    tok = (best or {}).get("concentration", {}).get("axes", {}).get("tokens", {})
    wbo = (best or {}).get("concentration", {}).get("axes", {}).get("w_blocks_output", {})
    map_deltas = []
    for b in billed:
        for row in b.get("map_value_vs_uniform") or []:
            map_deltas.append(abs(float(row["delta_held_out_relative_l2"] or 0.0)))
    map_shift = max(map_deltas) if map_deltas else 0.0
    min_frac = school.get("max_min_frac_coords_oracle_train_support")
    return {
        "is_the_residual_after_the_best_bulk_sparse": (
            "NO on the axes a gather-and-add exception can spend. On the best "
            f"billed bulk ({best['id'] if best else '?'}), output-coord energy "
            f"at 1% kept is {oc.get('frac_energy_at_1pct')} (uniform 0.01), "
            f"50% of residual energy needs {oc.get('frac_kept_for_50pct_energy')} "
            f"of the 5120 coords, W-blocks of output are dense="
            f"{wbo.get('dense')}, tokens need {tok.get('frac_kept_for_50pct_energy')} "
            f"of rows for 50% energy. Input-direction covariance energy is "
            f"concentrated (1% of dims capture {inn.get('frac_energy_at_1pct')}; "
            "that is ||X^T R||^2, not a measured input-sparse program). "
            "Oracle correction still needs "
            f"{min_frac} of output coords to clear the 0.25 kill — not sparse."
        ),
        "can_a_sparse_exception_rescue_these_bulk_programs": (
            "NO. "
            + school["why"]
            if school["status"] != OPEN
            else "NOT CLOSED. " + school["why"]
        ),
        "does_the_sensitivity_map_change_the_residual_allocation_number": (
            f"The largest |capability − uniform| held-out L2 shift at matched "
            f"byte budget is {map_shift:.6f}. The map allocates the residual "
            "budget; it is not itself a byte win (licensed MLP bytes "
            f"{measured['capability_allocation'].get('licensed_mlp_bytes')}, "
            "token share "
            f"{measured['capability_allocation'].get('licensed_token_share')})."
        ),
        "best_billed_bulk_held_out_relative_l2": (
            f"{best['id']} at {best['held_out_relative_l2']}" if best else "none"
        ),
    }


def _negative_findings(measured: Mapping[str, Any]) -> list[str]:
    out = []
    school = measured["school"]
    if school["status"] in {RESIDUAL_DENSE_CLOSED, RESIDUAL_RESCUE_CLOSED}:
        out.append(
            "residual-rescue of the recorded linear bulks is closed: "
            + school["why"]
        )
    for b in measured["bulks"]:
        oc = b["concentration"]["axes"]["output_coords"]
        out.append(
            f"{b['id']}: output-coord gini={oc['gini']} "
            f"energy@1%={oc['frac_energy_at_1pct']} "
            f"energy@10%={oc['frac_energy_at_10pct']} "
            f"frac_kept_for_50%_energy={oc['frac_kept_for_50pct_energy']} "
            f"min_k_oracle_train={b['min_k_oracle_train_support_to_clear_kill']} "
            f"oracle@5%_mlp={b['oracle_at_5pct_mlp_bytes']['held_out_relative_l2']}"
        )
    return out


@lru_cache(maxsize=1)
def cached_measure() -> dict[str, Any]:
    pack = load_pack(ROUND1_LAYER)
    cap = capability_coord_weights(int(pack["Yho"].shape[1]), layer=ROUND1_LAYER)
    return measure(pack=pack, cap=cap, n_layers=N_LAYERS)


def build(*, consult: bool = True) -> Path:
    test = selftest()
    index = consult_index() if consult else {"proceed": True, "skipped": True}
    if consult and not index.get("proceed", False):
        raise SparseResidualRefuse(
            "REFUSED: negative_index refuse_if_dead fired on the proposal "
            f"families: {index.get('proposal_refused')}"
        )
    measured = cached_measure()
    n_points = sum(len(b["budget_sweep"]) for b in measured["bulks"])
    n_neg = sum(
        1
        for b in measured["bulks"]
        for p in b["budget_sweep"]
        if p.get("status") == MEASURED_NEGATIVE
    )
    n_open = sum(
        1
        for b in measured["bulks"]
        for p in b["budget_sweep"]
        if p.get("status") == OPEN
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Measure the residual F - F_bulk after the best shared-program "
            "bulks and the oracle PCA-of-F control. Report residual "
            "concentration, held-out bulk+residual error under uniform and "
            "capability-allocated budgets, and executable economics of a "
            "gather-and-add sparse exception whose indices and dispatch are billed."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "recorded_by": RECORDED_BY,
        "git_head": git("rev-parse", "HEAD") or None,
        "predecessors": [SHARED_REL, CORPUS_REL, MAP_REL, BUDGET_REL, "receipts/future/EXECUTABLE_ECONOMICS.json"],
        "corpus": {
            "receipt": CORPUS_REL,
            "payload_dir": measured.get("payload_dir"),
            "layer": measured["layer"],
            "layer_role": "typical",
            "why_layer": (
                "Layer 38 is the teacher-corpus typical H(q) representative "
                "used by MLP_SHARED_PROGRAM. Residuals are measured on the "
                "same split so they are residuals of those bulks, not of a "
                "different fit."
            ),
            "n_train": measured["n_train"],
            "n_hold": measured["n_hold"],
            "n_train_prompts": measured["n_train_prompts"],
            "n_hold_prompts": measured["n_hold_prompts"],
            "split_unit": "prompt_id",
            "disjoint": True,
            "x_sha256": measured.get("x_sha256"),
            "y_sha256": measured.get("y_sha256"),
        },
        "metric": {
            "authority": "held_out_relative_l2",
            "formula": "E_x ||F(x) - F_hat(x)|| / E_x ||F(x)||",
            "split": "prompt_id hold set of the teacher corpus",
            "kill_rel": HELD_OUT_KILL_REL,
            "oracle_correction": (
                "diagnostic ceiling: true F on the selected coordinates, "
                "support ranked on train residual energy. Not a program."
            ),
            "linear_residual": (
                "authority for bulk+residual: gather-and-add of a k-row "
                "ridge map fit on train residual, scored on hold."
            ),
            "weight_reconstruction": "not scored",
        },
        "consuming_primitive": {
            "name": GATHER_ADD_PRIMITIVE,
            "algebra": "y = y_bulk + scatter(W_k x, indices)",
            "also": list(GATHER_ADD_ALSO),
            "index_dtype_bytes": INDEX_BYTES,
            "dispatch_delta_per_layer": 1.0,
            "note": "gather-and-add. Indices and extra dispatch are billed.",
        },
        "element_bytes": ELEMENT_BYTES,
        "n_layers_billed": N_LAYERS,
        "mlp_bytes": ee.MLP_ACTIVE_BYTES,
        "mlp_ms_cited": ee.MLP_MS,
        "token_ms_cited": ee.CITED_TOKEN_MS,
        "index": index,
        "selftest": test,
        "anti_fabrication": {
            "detectors": [
                "UNBILLED_RESIDUAL_INDEX",
                "UNBILLED_DISPATCH",
                "TRAIN_REPORTED_AS_HELD_OUT",
                "SYNTHETIC_ROW",
                "HELD_OUT_PROMPT_LEAK",
            ],
            "loud_exceptions": [
                "UnbilledResidualIndex",
                "UnbilledDispatch",
                "TrainReportedAsHeldOut",
                "CorpusUnavailable",
            ],
            "rule": (
                "emit_point is the only constructor for a sweep row. k>0 with "
                "index_bytes=0 or metadata that does not cover the indices "
                "raises UnbilledResidualIndex. k>0 with dispatch_delta=0 "
                "raises UnbilledDispatch. A train-set figure labelled "
                "held-out raises TrainReportedAsHeldOut. A return-flag "
                "nobody checks is not a guard."
            ),
        },
        "dead_schools_not_rebuilt": [
            "COMPOSITE_MLP_SIMPLE_LINEAR_LOW_RANK_REFUTED — reused as bulk, not reswept",
            "MLP code-body entropy coding (1.87018 bits of 2; Markov-1 +0.00195)",
        ],
        "capability_allocation": measured["capability_allocation"],
        "budgets": measured["budgets"],
        "bulks": measured["bulks"],
        "school": measured["school"],
        "n_sweep_points": n_points,
        "candidate_counts": {
            "n": n_points,
            "measured_negative": n_neg,
            "open": n_open,
        },
        "answers": _answers(measured),
        "negative_findings": _negative_findings(measured),
        "gaps_closed": [
            "residual after the recorded rank-64 shared-program bulks and the oracle PCA control, held out by prompt",
            "concentration curves on output coords, input directions, tokens, W-blocks",
            "uniform and capability-allocated residual budgets both reported",
            "gather-and-add indices billed; unbilled index refused",
            "dispatch delta billed; waved dispatch refused",
            "every sweep point scored by executable_economics",
            "train-set figure cannot be reported as held-out (loud exception)",
        ],
        "what_this_does_not_prove": [
            "that a structured nonlinear bulk (Monarch/butterfly/distilled SwiGLU) cannot have a sparse residual — those bulks are untested",
            "capability at generate",
            "a protected TPS or complete-token number",
            "that the sensitivity map's gate.channel groups equal down-row identity; they are an allocation prior",
        ],
        "nomenclature": {
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "residual_dense_closed": RESIDUAL_DENSE_CLOSED,
            "residual_rescue_closed": RESIDUAL_RESCUE_CLOSED,
            "uniform": UNIFORM,
            "capability": CAPABILITY,
            "held_out_authority": "held_out_relative_l2",
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
        "next": (
            "If residual-rescue is closed, do not spend another linear bulk "
            "plus sparse exception on this organ. The live path remains a "
            "function_replacement that respects SwiGLU (a nonlinearity "
            "between linear maps), not an r-dimensional bottleneck plus "
            "exceptions."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv_list)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.build or not argv_list:
        path = build()
        print(path)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
