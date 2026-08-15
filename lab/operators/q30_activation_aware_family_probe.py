#!/usr/bin/env python3
"""Bounded Q30 activation-aware family ranking probe.

Hypothesis under test
---------------------
Fitting expert projections to REAL captured L0 router-input activations can
change family ranking and the achievable output fidelity at complete BPW <= 1.5,
relative to the incumbent raw-weight low-rank family that the live search uses.

This probe is deliberately component-bounded:
- reads existing current-HCLI L0 route+hidden capture (no server, no lease)
- positioned-reads a small set of layer-0 expert tensors from local source shards
- scores candidates by OUTPUT cosine on held-out real activations, plus weight
  cosine and the constant-mean null baseline
- never weakens gates and never packs a full model

Negative results are first-class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults bound to the measured assets on this host
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
    "qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
DEFAULT_CAPTURE_RUN = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/"
    "physical/qwen30/quality-candidates/gate-up-residual-v1/current-hcli-route-capture/runs/"
    "74c918d500b2a8fdc17c2a4a417bf0e967b6a17709e7cdba486466c7c39e862a_"
    "8bd3bfb36e16be850dc5e1909e3f07a3b0ddc4f49634455e258a0eb3d8660037"
)
DEFAULT_CAPTURE_RECEIPT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/"
    "physical/qwen30/quality-candidates/gate-up-residual-v1/current-hcli-route-capture/receipts/"
    "QWEN30_HQ30GR2_CURRENT_HCLI_L0_ROUTE_CAPTURE_"
    "74c918d500b2a8fdc17c2a4a417bf0e967b6a17709e7cdba486466c7c39e862a_"
    "8bd3bfb36e16be850dc5e1909e3f07a3b0ddc4f49634455e258a0eb3d8660037.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/"
    "quality-diagnostics/activation-aware-v1"
)

SCHEMA = "hawking.ascension.qwen30_activation_aware_family_probe.v1"
GROUP = 64
SEED = 0x5130A1
CEILING_BPW = 1.5
# Operational bar for "coherence-grade" on a single linear projection.
# Deployed Q30 mean weight cosine ~0.80 is already declared LOW_FIDELITY /
# not eligible for runtime promotion. Absolute output cosine is NOT enough:
# on this capture the constant-mean null is often ~0.90–0.95, so a raw 0.95
# bar is nearly free. Coherence-grade requires a real surplus over that null.
# Weight cosine is reported but not required — the GLM-5.2 lesson is that
# weight cosine is the wrong objective — but a result with high surplus and
# near-zero weight cosine is labelled distribution-local, not operator recovery.
COHERENCE_MIN_SURPLUS = 0.10
COHERENCE_MIN_OUTPUT_COS = 0.90  # only meaningful together with surplus
OPERATOR_RECOVERY_WEIGHT_COS = 0.50  # below this = distribution-local only

# Matched budgets the live search and the user's measured curve both explore.
# bpw is billed per-tensor for the component under test (honest component BPW).
BUDGET_POINTS: tuple[dict[str, Any], ...] = (
    {"label": "r64_b3", "rank": 64, "bits": 3, "target_band": "well_under_ceiling"},
    {"label": "r128_b3", "rank": 128, "bits": 3, "target_band": "search_default"},
    {"label": "r192_b4", "rank": 192, "bits": 4, "target_band": "best_under_1p5"},
    {"label": "r256_b3", "rank": 256, "bits": 3, "target_band": "near_ceiling"},
    # Over-ceiling anchors so the report can state the BPW where fidelity appears.
    {"label": "r384_b4", "rank": 384, "bits": 4, "target_band": "over_ceiling_anchor"},
    {"label": "r512_b4", "rank": 512, "bits": 4, "target_band": "over_ceiling_anchor"},
    {"label": "r640_b4", "rank": 640, "bits": 4, "target_band": "over_ceiling_anchor"},
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mean_row_cosine(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    yhn = y_hat / (np.linalg.norm(y_hat, axis=1, keepdims=True) + 1e-12)
    return float((yn * yhn).sum(axis=1).mean())


def constant_mean_null(y: np.ndarray) -> float:
    mean_row = np.repeat(y.mean(axis=0, keepdims=True), y.shape[0], axis=0)
    return mean_row_cosine(y, mean_row)


def weight_cosine(W: np.ndarray, W_hat: np.ndarray) -> float:
    a = np.asarray(W, dtype=np.float64).reshape(-1)
    b = np.asarray(W_hat, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-30) * (np.linalg.norm(b) + 1e-30)))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))


def matvec_rows(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """W is [out, in], X is [N, in] -> Y [N, out]."""
    return X @ W.T


# ---------------------------------------------------------------------------
# Uniform group quant (body billing matches dual_gravity factor codec)
# ---------------------------------------------------------------------------


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    out = bytearray()
    acc = 0
    acc_bits = 0
    for v in values.tolist():
        acc |= int(v) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def uniform_group_quantize(
    values: np.ndarray, *, bits: int, group_size: int = GROUP
) -> tuple[np.ndarray, int]:
    """Return reconstructed array and billed body bytes (scales f16 + packed codes)."""
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in 2..8")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size), constant_values=0.0).reshape(
        groups, group_size
    )
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(padded), axis=1) / max(bound, 1)).astype(np.float32)
    denom = np.where(scales > 0.0, scales, 1.0)
    signed = np.rint(padded / denom[:, None]).clip(-bound, bound).astype(np.int16)
    unsigned = (signed.reshape(-1) + bound).astype(np.uint16)
    code_bytes = pack_unsigned(unsigned, bits)
    rebuilt = (signed.astype(np.float32) * scales[:, None]).reshape(-1)[: flat.size]
    billed = int(groups * 2 + len(code_bytes))  # f16 scales + codes
    return rebuilt.reshape(values.shape).astype(np.float32), billed


def low_rank_factor_bpw(shape: tuple[int, int], rank: int, bits: int, group_size: int = GROUP) -> float:
    rows, cols = shape
    n_w = rows * cols
    n_factor = rank * (rows + cols)
    groups = math.ceil(n_factor / group_size)
    # approximate billed bytes: f16 scales + packed codes (no outer magic/header)
    code_bytes = math.ceil(n_factor * bits / 8)
    billed = groups * 2 + code_bytes
    return billed * 8.0 / max(n_w, 1)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def randomized_low_rank(W: np.ndarray, *, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.ascontiguousarray(W, dtype=np.float32)
    rows, columns = matrix.shape
    actual = min(max(1, rank), rows, columns)
    oversample = min(12, max(0, min(rows, columns) - actual))
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal((columns, actual + oversample), dtype=np.float32)
    basis, _ = np.linalg.qr(matrix @ probe, mode="reduced")
    small = basis.T @ matrix
    left, singular, right = np.linalg.svd(small, full_matrices=False)
    L = np.ascontiguousarray(basis @ (left[:, :actual] * singular[:actual]), dtype=np.float32)
    R = np.ascontiguousarray(right[:actual, :], dtype=np.float32)
    return L, R


def exact_svd_low_rank(W: np.ndarray, *, rank: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.ascontiguousarray(W, dtype=np.float32)
    actual = min(max(1, rank), matrix.shape[0], matrix.shape[1])
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    L = (u[:, :actual] * s[:actual]).astype(np.float32)
    R = vt[:actual, :].astype(np.float32)
    return L, R


def quantize_factors(L: np.ndarray, R: np.ndarray, *, bits: int) -> tuple[np.ndarray, np.ndarray, int]:
    Lq, b1 = uniform_group_quantize(L, bits=bits)
    Rq, b2 = uniform_group_quantize(R, bits=bits)
    return Lq, Rq, b1 + b2


def family_raw_weight_low_rank(
    W: np.ndarray, X_fit: np.ndarray, *, rank: int, bits: int, seed: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    del X_fit  # deliberately unused — raw-weight objective
    L, R = randomized_low_rank(W, rank=rank, seed=seed)
    Lq, Rq, billed = quantize_factors(L, R, bits=bits)
    return Lq @ Rq, billed, {"fit": "randomized_svd_on_weights", "rank": int(Lq.shape[1]), "bits": bits}


def family_activation_pca_low_rank(
    W: np.ndarray, X_fit: np.ndarray, *, rank: int, bits: int, seed: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Input-side activation basis (GLM-5.2 style): B from SVD of centered X, L = W @ B."""
    del seed
    X = np.asarray(X_fit, dtype=np.float32)
    mu = X.mean(axis=0)
    Xc = X - mu
    # economy SVD; right singular vectors of Xc are activation principal directions
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    actual = min(max(1, rank), vt.shape[0], W.shape[1], max(1, X.shape[0] - 1))
    B = vt[:actual].T.astype(np.float32)  # [in, r]
    L = (W @ B).astype(np.float32)  # [out, r]
    Lq, bL = uniform_group_quantize(L, bits=bits)
    Bq, bB = uniform_group_quantize(B, bits=bits)
    W_hat = Lq @ Bq.T
    var = (s.astype(np.float64) ** 2)
    cum = float(var[:actual].sum() / (var.sum() + 1e-30))
    return W_hat, bL + bB, {
        "fit": "activation_pca_basis_input_side",
        "rank": actual,
        "bits": bits,
        "activation_variance_captured": cum,
        "n_fit_tokens": int(X.shape[0]),
    }


def family_activation_weighted_binary(
    W: np.ndarray, X_fit: np.ndarray, *, rank: int, bits: int, seed: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Activation-weighted binary sign/scale (group) + optional top-column FP16 residual.

    Pure multi-bit group quant cannot land under 1.5 BPW (3-bit RTN ≈ 3.25 BPW).
    This family is the under-ceiling activation-weighted quant mechanism:
    - per-group binary sign + f16 scale after column rescaling by activation RMS
    - spend remaining budget (from the budget point's nominal low-rank bpw) on
      the highest-activation-energy columns stored as f16 residual

    `rank`/`bits` select residual budget via the matching low-rank bpw estimate.
    """
    del seed
    X = np.asarray(X_fit, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(X), axis=0) + 1e-12).astype(np.float32)
    scale = np.maximum(rms, float(np.median(rms)) * 0.01)
    Ws = W * scale[None, :]
    # binary group: 1 bit/weight + f16 scale per group
    flat = Ws.reshape(-1)
    group_size = GROUP
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size)).reshape(groups, group_size)
    scales = np.mean(np.abs(padded), axis=1).astype(np.float32)
    signs = np.where(padded >= 0.0, 1.0, -1.0)
    rebuilt = (signs * scales[:, None]).reshape(-1)[: flat.size].reshape(Ws.shape)
    code_bytes = math.ceil(flat.size / 8)
    billed = groups * 2 + code_bytes
    W_bin = rebuilt / scale[None, :]

    # residual budget from nominal low-rank point
    target_bpw = low_rank_factor_bpw(W.shape, rank=max(1, rank), bits=max(2, bits))
    target_bpw = min(target_bpw, CEILING_BPW)
    budget_bytes = int(target_bpw * W.size / 8)
    residual_bytes = max(0, budget_bytes - billed)
    # each residual column costs out * 2 bytes (f16)
    col_cost = W.shape[0] * 2
    n_cols = min(W.shape[1], residual_bytes // max(col_cost, 1))
    residual = (W - W_bin).astype(np.float32)
    # pick columns by activation energy * residual energy
    col_score = rms * np.linalg.norm(residual, axis=0)
    order = np.argsort(-col_score)
    keep = order[:n_cols]
    W_hat = W_bin.copy()
    if n_cols > 0:
        W_hat[:, keep] = W[:, keep]
        billed += int(n_cols * col_cost)
    return W_hat.astype(np.float32), billed, {
        "fit": "activation_rms_weighted_binary_plus_top_column_fp16_residual",
        "bits_base": 1,
        "group_size": GROUP,
        "n_residual_columns": int(n_cols),
        "target_bpw_nominal": float(target_bpw),
        "n_fit_tokens": int(X.shape[0]),
    }


def family_activation_weighted_svd_low_rank(
    W: np.ndarray, X_fit: np.ndarray, *, rank: int, bits: int, seed: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Low-rank in the activation-induced metric: SVD of W @ Chol(XᵀX / N).

    Minimizes ||(W - LR) X||_F (output Frobenius on the fit activations) rather
    than ||W - LR||_F. Materially different from raw-weight SVD and from pure
    activation-PCA projection of W onto span(X).
    """
    del seed
    X = np.asarray(X_fit, dtype=np.float32)
    n = max(1, X.shape[0])
    # G = XᵀX / n + ridge; use eigendecomposition for symmetric PSD sqrt
    G = (X.T @ X) / n
    G = G + (1e-5 * float(np.trace(G) / max(G.shape[0], 1)) + 1e-8) * np.eye(G.shape[0], dtype=np.float32)
    # eigh for stability
    evals, evecs = np.linalg.eigh(G.astype(np.float64))
    evals = np.clip(evals, 1e-12, None)
    sqrt_g = (evecs * np.sqrt(evals)) @ evecs.T
    inv_sqrt_g = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    # SVD of W weighted
    Ww = (W.astype(np.float64) @ sqrt_g).astype(np.float32)
    actual = min(max(1, rank), Ww.shape[0], Ww.shape[1])
    u, s, vt = np.linalg.svd(Ww, full_matrices=False)
    L = (u[:, :actual] * s[:actual]).astype(np.float32)
    R_w = vt[:actual, :].astype(np.float32)
    # map factors back: W ≈ L R_w inv_sqrt_g
    R = (R_w @ inv_sqrt_g.astype(np.float32)).astype(np.float32)
    Lq, Rq, billed = quantize_factors(L, R, bits=bits)
    return Lq @ Rq, billed, {
        "fit": "activation_weighted_svd_output_frobenius",
        "rank": actual,
        "bits": bits,
        "n_fit_tokens": int(X.shape[0]),
    }


FAMILIES: dict[str, Callable[..., tuple[np.ndarray, int, dict[str, Any]]]] = {
    "raw_weight_low_rank_q": family_raw_weight_low_rank,
    "activation_pca_low_rank_q": family_activation_pca_low_rank,
    "activation_weighted_binary_residual": family_activation_weighted_binary,
    "activation_weighted_svd_low_rank_q": family_activation_weighted_svd_low_rank,
}


# ---------------------------------------------------------------------------
# Capture / data loading
# ---------------------------------------------------------------------------


def load_capture(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "capture-result.json").read_text(encoding="utf-8"))


def collect_expert_activations(
    run_dir: Path, capture: Mapping[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Return expert_id -> X [N, hidden] of tokens that routed to that expert."""
    by_expert: dict[int, list[np.ndarray]] = defaultdict(list)
    provenance = {
        "capture_run": str(run_dir),
        "capture_result_sha256": sha256_file(run_dir / "capture-result.json"),
        "probes": [],
        "hidden_source": "device-produced L0 post-attention RMSNorm (router input)",
        "claim_boundary": capture.get("claim_boundary"),
        "status": capture.get("status"),
        "schema": capture.get("schema"),
    }
    total_steps = 0
    for probe in capture["probes"]:
        probe_id = probe["probe_id"]
        steps = probe["steps"]
        provenance["probes"].append(
            {
                "probe_id": probe_id,
                "n_steps": len(steps),
                "source_one_user_native_prompt_token_count": probe.get(
                    "source_one_user_native_prompt_token_count"
                ),
            }
        )
        for step in steps:
            total_steps += 1
            rel = step["router_input_hidden_f32le"]["relative_path"]
            path = run_dir / rel
            x = np.fromfile(path, dtype="<f4")
            if x.size != step["router_input_hidden_f32le"]["elements"]:
                raise RuntimeError(f"hidden size mismatch at {path}")
            for expert_id in step["selected_expert_ids"]:
                by_expert[int(expert_id)].append(x)
    stacked = {eid: np.stack(rows, axis=0) for eid, rows in by_expert.items()}
    hit_counts = {str(k): int(v.shape[0]) for k, v in sorted(stacked.items(), key=lambda kv: -kv[1].shape[0])}
    provenance["total_steps"] = total_steps
    provenance["token_expert_pairs"] = int(sum(v.shape[0] for v in stacked.values()))
    provenance["experts_with_hits"] = len(stacked)
    provenance["hit_counts_top"] = dict(list(hit_counts.items())[:20])
    return stacked, provenance


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def holdout_split(
    X: np.ndarray, *, seed: int, hold_frac: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 4:
        return X, X
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, min(n // 2, int(round(n * hold_frac))))
    hold_idx = perm[:n_hold]
    fit_idx = perm[n_hold:]
    if fit_idx.size == 0:
        fit_idx = perm
    return X[fit_idx], X[hold_idx]


@dataclass
class Score:
    family: str
    budget_label: str
    tensor: str
    expert: int
    component: str
    shape: list[int]
    n_weights: int
    rank: int | None
    bits: int
    billed_bytes: int
    component_bpw: float
    under_ceiling: bool
    weight_cosine: float
    weight_relative_l2: float
    output_cosine: float
    output_relative_l2: float
    null_baseline: float
    surplus_over_null: float
    beats_null: bool
    coherence_grade: bool
    distribution_local_only: bool
    n_fit_tokens: int
    n_hold_tokens: int
    fit_meta: dict[str, Any]


def score_candidate(
    *,
    family: str,
    budget: Mapping[str, Any],
    tensor_name: str,
    expert: int,
    component: str,
    W: np.ndarray,
    W_hat: np.ndarray,
    X_hold: np.ndarray,
    billed_bytes: int,
    n_fit: int,
    fit_meta: Mapping[str, Any],
) -> Score:
    y = matvec_rows(W, X_hold)
    y_hat = matvec_rows(W_hat, X_hold)
    out_cos = mean_row_cosine(y, y_hat)
    null = constant_mean_null(y)
    n_w = int(W.size)
    bpw = billed_bytes * 8.0 / max(n_w, 1)
    surplus = out_cos - null
    wt = weight_cosine(W, W_hat)
    under = bool(bpw <= CEILING_BPW + 1e-9)
    # Surplus-first coherence: absolute cosine alone is inadmissible when null is high.
    coh = bool(
        under
        and out_cos >= COHERENCE_MIN_OUTPUT_COS
        and surplus >= COHERENCE_MIN_SURPLUS
        and out_cos > null
    )
    rank_meta = fit_meta.get("rank")
    if rank_meta is None and family != "activation_weighted_binary_residual":
        rank_meta = int(budget["rank"])
    elif rank_meta is not None:
        rank_meta = int(rank_meta)
    return Score(
        family=family,
        budget_label=str(budget["label"]),
        tensor=tensor_name,
        expert=expert,
        component=component,
        shape=[int(x) for x in W.shape],
        n_weights=n_w,
        rank=rank_meta,
        bits=int(budget["bits"]),
        billed_bytes=int(billed_bytes),
        component_bpw=float(bpw),
        under_ceiling=under,
        weight_cosine=wt,
        weight_relative_l2=relative_l2(W, W_hat),
        output_cosine=out_cos,
        output_relative_l2=relative_l2(y, y_hat),
        null_baseline=null,
        surplus_over_null=surplus,
        beats_null=bool(out_cos > null),
        coherence_grade=coh,
        distribution_local_only=bool(coh and wt < OPERATOR_RECOVERY_WEIGHT_COS),
        n_fit_tokens=int(n_fit),
        n_hold_tokens=int(X_hold.shape[0]),
        fit_meta=dict(fit_meta),
    )


def select_experts(hit_counts: Mapping[int, int], *, requested: Sequence[int] | None) -> list[int]:
    if requested:
        return [int(e) for e in requested]
    # top-3 by hits + expert 127 (prior measurement continuity) if present
    ranked = sorted(hit_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [e for e, _ in ranked[:3]]
    if 127 in hit_counts and 127 not in chosen:
        chosen.append(127)
    return chosen


def run_probe(
    *,
    model_dir: Path,
    capture_run: Path,
    capture_receipt: Path | None,
    out_dir: Path,
    experts: Sequence[int] | None,
    components: Sequence[str],
    min_tokens: int,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = load_capture(capture_run)
    by_expert, act_prov = collect_expert_activations(capture_run, capture)
    hit_counts = {e: int(X.shape[0]) for e, X in by_expert.items()}
    chosen = select_experts(hit_counts, requested=experts)
    chosen = [e for e in chosen if hit_counts.get(e, 0) >= min_tokens]
    if not chosen:
        raise RuntimeError(
            f"no experts with >= {min_tokens} routed tokens among selection; "
            f"top hits={act_prov['hit_counts_top']}"
        )

    weight_map = load_weight_map(model_dir)
    scores: list[Score] = []
    tensor_notes: list[dict[str, Any]] = []

    for expert in chosen:
        X_all = by_expert[expert]
        for component in components:
            name = f"model.layers.0.mlp.experts.{expert}.{component}.weight"
            if name not in weight_map:
                raise RuntimeError(f"missing tensor {name}")
            W = load_tensor(model_dir, weight_map, name).astype(np.float32, copy=False)

            if component in ("gate_proj", "up_proj"):
                if W.shape[1] != X_all.shape[1]:
                    raise RuntimeError(f"{name} in-dim {W.shape[1]} != hidden {X_all.shape[1]}")
                X_use = X_all
            elif component == "down_proj":
                # Intermediate activations from real gate/up + silu, using true BF16 weights.
                # These are the honest expert inputs to down_proj for the routed tokens.
                Wg = load_tensor(
                    model_dir, weight_map, f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
                ).astype(np.float32, copy=False)
                Wu = load_tensor(
                    model_dir, weight_map, f"model.layers.0.mlp.experts.{expert}.up_proj.weight"
                ).astype(np.float32, copy=False)
                X_use = silu(matvec_rows(Wg, X_all)) * matvec_rows(Wu, X_all)
                if W.shape[1] != X_use.shape[1]:
                    raise RuntimeError(
                        f"{name} in-dim {W.shape[1]} != intermediate {X_use.shape[1]}"
                    )
            else:
                raise RuntimeError(f"unsupported component {component}")

            comp_seed = int(hashlib.sha256(component.encode()).hexdigest()[:8], 16)
            X_fit, X_hold = holdout_split(X_use, seed=SEED ^ (expert * 1009) ^ (comp_seed & 0xFFFF))
            tensor_notes.append(
                {
                    "tensor": name,
                    "expert": expert,
                    "component": component,
                    "shape": list(W.shape),
                    "n_routed_tokens": int(X_all.shape[0]),
                    "n_fit": int(X_fit.shape[0]),
                    "n_hold": int(X_hold.shape[0]),
                    "activation_kind": (
                        "router_input_hidden"
                        if component in ("gate_proj", "up_proj")
                        else "swiglu_intermediate_from_real_hidden_and_true_gate_up"
                    ),
                    "source_shard": weight_map[name],
                    "weight_sha256_prefix": sha256_bytes(W.tobytes())[:16],
                }
            )

            # Null-only diagnostic on true outputs (independent of family)
            y_true = matvec_rows(W, X_hold)
            null_only = constant_mean_null(y_true)

            for family_name, family_fn in FAMILIES.items():
                for budget in BUDGET_POINTS:
                    rank = int(budget["rank"])
                    bits = int(budget["bits"])
                    # For pure group quant, rank is meaningless; map budget to bits only.
                    # To produce comparable BPW, also offer multi-bit variants already
                    # encoded in BUDGET_POINTS via bits field.
                    W_hat, billed, meta = family_fn(
                        W, X_fit, rank=rank, bits=bits, seed=SEED ^ expert ^ rank
                    )
                    # Skip pathological over-rank for small matrices
                    if W_hat.shape != W.shape:
                        raise RuntimeError("reconstruction shape mismatch")
                    sc = score_candidate(
                        family=family_name,
                        budget=budget,
                        tensor_name=name,
                        expert=expert,
                        component=component,
                        W=W,
                        W_hat=W_hat,
                        X_hold=X_hold,
                        billed_bytes=billed,
                        n_fit=X_fit.shape[0],
                        fit_meta={**meta, "budget_band": budget["target_band"], "null_only_check": null_only},
                    )
                    scores.append(sc)

    rows = [asdict(s) for s in scores]
    under = [r for r in rows if r["under_ceiling"]]
    over = [r for r in rows if not r["under_ceiling"]]
    high_hit_experts = {e for e, n in hit_counts.items() if n >= 200}
    under_high = [r for r in under if r["expert"] in high_hit_experts]

    def best_of(group: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        if not group:
            return None
        return max(
            group,
            key=lambda r: (
                float(r[key]),
                float(r["surplus_over_null"]),
                -float(r["component_bpw"]),
            ),
        )

    def summarize_family(fam: str, group: list[dict[str, Any]]) -> dict[str, Any] | None:
        fam_rows = [r for r in group if r["family"] == fam]
        if not fam_rows:
            return {
                "family": fam,
                "n_rows": 0,
                "mean_output_cosine": None,
                "mean_weight_cosine": None,
                "mean_null_baseline": None,
                "mean_surplus_over_null": None,
                "frac_beats_null": None,
                "best_output_cosine": None,
                "best_surplus": None,
                "any_coherence": False,
                "any_coherence_with_operator_weight": False,
            }
        return {
            "family": fam,
            "n_rows": len(fam_rows),
            "mean_output_cosine": float(np.mean([r["output_cosine"] for r in fam_rows])),
            "mean_weight_cosine": float(np.mean([r["weight_cosine"] for r in fam_rows])),
            "mean_null_baseline": float(np.mean([r["null_baseline"] for r in fam_rows])),
            "mean_surplus_over_null": float(np.mean([r["surplus_over_null"] for r in fam_rows])),
            "frac_beats_null": float(np.mean([1.0 if r["beats_null"] else 0.0 for r in fam_rows])),
            "best_output_cosine": max(r["output_cosine"] for r in fam_rows),
            "best_surplus": max(r["surplus_over_null"] for r in fam_rows),
            "any_coherence": any(r["coherence_grade"] for r in fam_rows),
            "any_coherence_with_operator_weight": any(
                r["coherence_grade"] and not r["distribution_local_only"] for r in fam_rows
            ),
        }

    best_under_output = best_of(under, "output_cosine")
    best_under_surplus = best_of(under, "surplus_over_null")
    best_high_surplus = best_of(under_high, "surplus_over_null")
    any_coherence_under = any(r["coherence_grade"] for r in under)
    any_coherence_high = any(r["coherence_grade"] for r in under_high)
    any_operator_coherence = any(
        r["coherence_grade"] and not r["distribution_local_only"] for r in under
    )
    any_operator_coherence_high = any(
        r["coherence_grade"] and not r["distribution_local_only"] for r in under_high
    )
    # surplus-first coherence at any BPW (including over ceiling anchors)
    coherence_rows = [
        r
        for r in rows
        if r["output_cosine"] >= COHERENCE_MIN_OUTPUT_COS
        and r["surplus_over_null"] >= COHERENCE_MIN_SURPLUS
        and r["beats_null"]
    ]
    coherence_rows_sorted = sorted(
        coherence_rows,
        key=lambda r: (float(r["component_bpw"]), -float(r["surplus_over_null"])),
    )
    first_coherence = coherence_rows_sorted[0] if coherence_rows_sorted else None
    first_operator_coherence = next(
        (
            r
            for r in coherence_rows_sorted
            if r["weight_cosine"] >= OPERATOR_RECOVERY_WEIGHT_COS
            and r["expert"] in high_hit_experts
        ),
        None,
    )
    first_operator_coherence_any = next(
        (
            r
            for r in coherence_rows_sorted
            if r["weight_cosine"] >= OPERATOR_RECOVERY_WEIGHT_COS
        ),
        None,
    )

    family_summary = [
        s
        for s in (summarize_family(fam, under) for fam in FAMILIES)
        if s is not None
    ]
    family_summary.sort(
        key=lambda r: (
            -(r["mean_surplus_over_null"] if r["mean_surplus_over_null"] is not None else -1e9),
            -(r["mean_output_cosine"] if r["mean_output_cosine"] is not None else -1e9),
        )
    )
    family_summary_high = [
        s
        for s in (summarize_family(fam, under_high) for fam in FAMILIES)
        if s is not None
    ]
    family_summary_high.sort(
        key=lambda r: (
            -(r["mean_surplus_over_null"] if r["mean_surplus_over_null"] is not None else -1e9),
            -(r["mean_output_cosine"] if r["mean_output_cosine"] is not None else -1e9),
        )
    )

    raw = next((f for f in family_summary if f["family"] == "raw_weight_low_rank_q"), None)
    aa = next((f for f in family_summary if f["family"] == "activation_pca_low_rank_q"), None)
    aw_svd = next(
        (f for f in family_summary if f["family"] == "activation_weighted_svd_low_rank_q"), None
    )
    aw_bin = next(
        (f for f in family_summary if f["family"] == "activation_weighted_binary_residual"), None
    )
    raw_h = next((f for f in family_summary_high if f["family"] == "raw_weight_low_rank_q"), None)
    aa_h = next(
        (f for f in family_summary_high if f["family"] == "activation_pca_low_rank_q"), None
    )

    def beats(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
        if not a or not b:
            return None
        sa = a.get("mean_surplus_over_null")
        sb = b.get("mean_surplus_over_null")
        if sa is None or sb is None:
            return None
        return bool(sa > sb)

    nulls = [r["null_baseline"] for r in rows]
    mean_null = float(np.mean(nulls)) if nulls else None
    mean_null_high = (
        float(np.mean([r["null_baseline"] for r in under_high])) if under_high else None
    )

    verdict_parts: list[str] = []
    # Primary verdict uses high-hit experts to avoid N-token rank collapse.
    if any_coherence_high and any_operator_coherence_high:
        verdict_parts.append(
            "POSITIVE on high-hit experts: surplus-first coherence-grade cleared under "
            f"component BPW <= {CEILING_BPW} with weight cosine also above the "
            f"{OPERATOR_RECOVERY_WEIGHT_COS} distribution-local cutoff."
        )
    elif any_coherence_high:
        verdict_parts.append(
            "MIXED/NEGATIVE for promotion: high-hit experts clear a surplus-first local "
            f"output bar under BPW <= {CEILING_BPW}, but only as distribution-local matches "
            f"(weight cosine < {OPERATOR_RECOVERY_WEIGHT_COS}). Not evidence of a coherent "
            "full-model artifact."
        )
    else:
        verdict_parts.append(
            "NEGATIVE: no tested family reaches surplus-first coherence-grade fidelity on "
            f"high-hit experts at component BPW <= {CEILING_BPW} "
            f"(require surplus>={COHERENCE_MIN_SURPLUS}, output_cos>={COHERENCE_MIN_OUTPUT_COS}, beats null)."
        )

    if first_coherence is not None:
        verdict_parts.append(
            "First surplus-first coherence-grade point (any expert, any BPW) at component_bpw="
            f"{first_coherence['component_bpw']:.4f} "
            f"({first_coherence['family']}/{first_coherence['budget_label']}/"
            f"{first_coherence['component']} expert {first_coherence['expert']}, "
            f"out={first_coherence['output_cosine']:.4f}, null={first_coherence['null_baseline']:.4f}, "
            f"surplus={first_coherence['surplus_over_null']:+.4f}, "
            f"wt={first_coherence['weight_cosine']:.4f})."
        )
    if first_operator_coherence is not None:
        verdict_parts.append(
            "On high-hit experts, first surplus-first row that also clears the operator-recovery "
            f"weight-cosine cutoff is at component_bpw={first_operator_coherence['component_bpw']:.4f} "
            f"({first_operator_coherence['family']}/{first_operator_coherence['budget_label']}, "
            f"wt={first_operator_coherence['weight_cosine']:.4f}, "
            f"surplus={first_operator_coherence['surplus_over_null']:+.4f})."
        )
    else:
        wt_rows = sorted(
            [r for r in rows if r["expert"] in high_hit_experts],
            key=lambda r: (-float(r["weight_cosine"]), float(r["component_bpw"])),
        )
        if wt_rows:
            best_wt = wt_rows[0]
            verdict_parts.append(
                "On high-hit experts, no surplus-first coherence row also recovers the operator "
                f"(weight_cos>={OPERATOR_RECOVERY_WEIGHT_COS}). Best high-hit weight cosine is "
                f"{best_wt['weight_cosine']:.4f} at bpw={best_wt['component_bpw']:.4f} "
                f"({best_wt['family']}/{best_wt['budget_label']}, "
                f"surplus={best_wt['surplus_over_null']:+.4f})."
            )
    if first_operator_coherence_any is not None and first_operator_coherence is None:
        verdict_parts.append(
            "Low-hit footnote: operator-recovery+surplus first appears at "
            f"component_bpw={first_operator_coherence_any['component_bpw']:.4f} on expert "
            f"{first_operator_coherence_any['expert']} "
            f"({first_operator_coherence_any['family']}/"
            f"{first_operator_coherence_any['budget_label']})."
        )

    ranking_note = {
        "activation_pca_beats_raw_weight_on_surplus": beats(aa, raw),
        "activation_weighted_svd_beats_raw_weight_on_surplus": beats(aw_svd, raw),
        "activation_weighted_binary_beats_raw_weight_on_surplus": beats(aw_bin, raw),
        "activation_pca_beats_raw_weight_on_surplus_high_hit_experts": beats(aa_h, raw_h),
        "primary_ranking_metric": "mean_surplus_over_null_under_ceiling",
        "secondary_ranking_metric": "mean_output_cosine_under_ceiling",
        "high_hit_expert_threshold_tokens": 200,
        "high_hit_experts": sorted(high_hit_experts),
    }
    if ranking_note["activation_pca_beats_raw_weight_on_surplus_high_hit_experts"]:
        verdict_parts.append(
            "Family ranking on real activations DOES invert vs weight-space: activation-PCA "
            "low-rank beats raw-weight low-rank on surplus-over-null for high-hit experts. "
            "Raw-weight low-rank typically fails to beat the constant-mean null on this capture."
        )
    else:
        verdict_parts.append(
            "On high-hit experts, activation-aware families do not cleanly invert ranking "
            "enough to crown a coherent under-ceiling champion."
        )

    if mean_null_high is not None and mean_null_high >= 0.85:
        verdict_parts.append(
            f"CRITICAL NULL TRAP: mean constant-mean null on high-hit under-ceiling rows is "
            f"{mean_null_high:.4f}. Absolute output cosine without null subtraction is "
            "inadmissible here (prior campaign constant-mean null of ~0.90)."
        )
    if best_high_surplus is not None:
        verdict_parts.append(
            f"Best high-hit under-ceiling surplus={best_high_surplus['surplus_over_null']:+.4f} "
            f"(out={best_high_surplus['output_cosine']:.4f}, null={best_high_surplus['null_baseline']:.4f}, "
            f"wt={best_high_surplus['weight_cosine']:.4f}, bpw={best_high_surplus['component_bpw']:.4f}) "
            f"via {best_high_surplus['family']}/{best_high_surplus['budget_label']} "
            f"on {best_high_surplus['component']} expert {best_high_surplus['expert']}."
        )

    # Honest BPW-to-reachability on HIGH-HIT experts (primary), then any-expert footnote.
    def first_joint(group: list[dict[str, Any]]) -> dict[str, Any] | None:
        for r in sorted(group, key=lambda r: float(r["component_bpw"])):
            if (
                r["surplus_over_null"] >= COHERENCE_MIN_SURPLUS
                and r["output_cosine"] >= COHERENCE_MIN_OUTPUT_COS
                and r["beats_null"]
                and r["weight_cosine"] >= OPERATOR_RECOVERY_WEIGHT_COS
            ):
                return {
                    "component_bpw": r["component_bpw"],
                    "family": r["family"],
                    "budget_label": r["budget_label"],
                    "expert": r["expert"],
                    "component": r["component"],
                    "output_cosine": r["output_cosine"],
                    "null_baseline": r["null_baseline"],
                    "surplus_over_null": r["surplus_over_null"],
                    "weight_cosine": r["weight_cosine"],
                    "n_hold_tokens": r["n_hold_tokens"],
                }
        return None

    reachability_high = first_joint([r for r in rows if r["expert"] in high_hit_experts])
    reachability_any = first_joint(rows)
    if reachability_high is None:
        verdict_parts.append(
            "On high-hit experts, within the tested grid (up to rank-640 / ~4+ BPW component), "
            "no point jointly clears surplus-first coherence and the operator-recovery "
            "weight-cosine cutoff. Exact BPW for joint high-hit reachability is ABOVE the "
            "highest tested anchor for this three-prompt capture."
        )
    else:
        verdict_parts.append(
            "On high-hit experts, joint surplus+operator reachability first appears at "
            f"component_bpw={reachability_high['component_bpw']:.4f} "
            f"({reachability_high['family']}/{reachability_high['budget_label']})."
        )
    if reachability_any is not None and (
        reachability_high is None
        or reachability_any["component_bpw"] < reachability_high["component_bpw"] - 1e-9
    ):
        verdict_parts.append(
            "Footnote (low-hit experts only, not primary): joint surplus+operator first seen at "
            f"component_bpw={reachability_any['component_bpw']:.4f} on expert "
            f"{reachability_any['expert']} ({reachability_any['family']}/"
            f"{reachability_any['budget_label']}); small-N activations can inflate surplus."
        )

    status = (
        "EARNED_ACTIVATION_AWARE_FAMILY_PROBE_COMPLETE_OPERATOR_COHERENCE_UNDER_CEILING"
        if any_coherence_high and any_operator_coherence_high
        else (
            "EARNED_ACTIVATION_AWARE_FAMILY_PROBE_COMPLETE_LOCAL_ONLY_OR_NEGATIVE"
            if any_coherence_high or any_coherence_under
            else "EARNED_ACTIVATION_AWARE_FAMILY_PROBE_COMPLETE_NEGATIVE_UNDER_CEILING"
        )
    )

    receipt = {
        "schema": SCHEMA,
        "status": status,
        "recorded_at": utc_now(),
        "hypothesis": (
            "Fitting Q30 experts to REAL captured activations changes family ranking and "
            "achievable fidelity at complete BPW <= 1.5 relative to raw-weight low-rank."
        ),
        "coherence_grade_definition": {
            "output_cosine_min": COHERENCE_MIN_OUTPUT_COS,
            "surplus_over_null_min": COHERENCE_MIN_SURPLUS,
            "component_bpw_max": CEILING_BPW,
            "operator_recovery_weight_cosine_min": OPERATOR_RECOVERY_WEIGHT_COS,
            "primary_metric": "surplus_over_null on held-out real activations",
            "note": (
                "Absolute output cosine is inadmissible without the constant-mean null. "
                "Deployed Q30 mean weight cosine ~0.80 is already LOW_FIDELITY. "
                "A row may be 'coherence_grade' on surplus yet 'distribution_local_only' "
                "if weight cosine is below the operator-recovery cutoff."
            ),
        },
        "ceiling_bpw": CEILING_BPW,
        "claim_boundary": {
            "no_server_started": True,
            "no_lease_issued": True,
            "no_full_model_pack": True,
            "no_gate_weakened": True,
            "cpu_numpy_only": True,
            "component_bpw_not_complete_model_bpw": True,
            "activations": "real current-HCLI L0 router-input hiddens from sealed route capture",
            "down_proj_activations": "derived swiglu intermediate using true BF16 gate/up on those hiddens",
            "not_a_runtime_admission": True,
            "not_a_capability_claim": True,
            "capture_is_three_prompt_prefix_only": True,
        },
        "assets": {
            "model_dir": str(model_dir),
            "capture_run": str(capture_run),
            "capture_run_sha256_capture_result": act_prov["capture_result_sha256"],
            "capture_receipt": str(capture_receipt) if capture_receipt else None,
            "capture_receipt_sha256": (
                sha256_file(capture_receipt)
                if capture_receipt and capture_receipt.is_file()
                else None
            ),
            "activation_provenance": act_prov,
        },
        "selection": {
            "experts": chosen,
            "components": list(components),
            "min_tokens": min_tokens,
            "hit_counts": {str(e): hit_counts[e] for e in chosen},
            "high_hit_experts": sorted(high_hit_experts),
            "budget_points": list(BUDGET_POINTS),
            "families": list(FAMILIES.keys()),
            "seed": SEED,
            "holdout_frac": 0.25,
        },
        "tensors": tensor_notes,
        "family_summary_under_ceiling": family_summary,
        "family_summary_under_ceiling_high_hit_experts": family_summary_high,
        "ranking_on_real_activations": ranking_note,
        "best_under_ceiling_by_output_cosine": best_under_output,
        "best_under_ceiling_by_surplus": best_under_surplus,
        "best_high_hit_under_ceiling_by_surplus": best_high_surplus,
        "first_coherence_grade_point_any_bpw": first_coherence,
        "first_operator_recovery_coherence_point": first_operator_coherence,
        "joint_surplus_and_operator_reachability_high_hit_experts": reachability_high,
        "joint_surplus_and_operator_reachability_any_expert": reachability_any,
        "any_coherence_under_ceiling": any_coherence_under,
        "any_coherence_under_ceiling_high_hit_experts": any_coherence_high,
        "any_operator_recovery_coherence_under_ceiling": any_operator_coherence,
        "any_operator_recovery_coherence_under_ceiling_high_hit_experts": any_operator_coherence_high,
        "mean_null_baseline_all_rows": mean_null,
        "mean_null_baseline_high_hit_under_ceiling": mean_null_high,
        "verdict": " ".join(verdict_parts),
        "rows": rows,
        "seconds": round(time.time() - t0, 3),
        "code_sha256": sha256_file(Path(__file__)),
    }

    # Write JSON + compact markdown table
    json_path = out_dir / "Q30_ACTIVATION_AWARE_FAMILY_PROBE.json"
    md_path = out_dir / "Q30_ACTIVATION_AWARE_FAMILY_PROBE.md"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(receipt), encoding="utf-8")
    receipt["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    # rewrite with outputs
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def render_markdown(receipt: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Q30 activation-aware family probe")
    lines.append("")
    lines.append(f"- Recorded: `{receipt['recorded_at']}`")
    lines.append(f"- Status: `{receipt['status']}`")
    lines.append(f"- Ceiling: component BPW ≤ {receipt['ceiling_bpw']}")
    cg = receipt["coherence_grade_definition"]
    lines.append(
        f"- Coherence bar (surplus-first): output_cos ≥ {cg['output_cosine_min']}, "
        f"surplus ≥ {cg['surplus_over_null_min']}, "
        f"operator-recovery weight_cos ≥ {cg['operator_recovery_weight_cosine_min']}"
    )
    lines.append(
        f"- Mean null (high-hit under ceiling): "
        f"`{receipt.get('mean_null_baseline_high_hit_under_ceiling')}`"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(receipt["verdict"])
    lines.append("")
    lines.append("## Family summary — high-hit experts only (primary)")
    lines.append("")
    lines.append(
        "| family | n | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | local coh? | operator coh? |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for f in receipt["family_summary_under_ceiling_high_hit_experts"]:
        lines.append(
            f"| `{f['family']}` | {f['n_rows']} | "
            f"{_fmt(f['mean_output_cosine'])} | "
            f"{_fmt(f['mean_weight_cosine'])} | "
            f"{_fmt(f['mean_null_baseline'])} | "
            f"{_fmt(f['mean_surplus_over_null'])} | "
            f"{_fmt(f['frac_beats_null'])} | "
            f"{_fmt(f['best_surplus'])} | "
            f"{'yes' if f['any_coherence'] else 'no'} | "
            f"{'yes' if f['any_coherence_with_operator_weight'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Family summary — all selected experts (under ceiling)")
    lines.append("")
    lines.append(
        "| family | n | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | local coh? | operator coh? |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for f in receipt["family_summary_under_ceiling"]:
        lines.append(
            f"| `{f['family']}` | {f['n_rows']} | "
            f"{_fmt(f['mean_output_cosine'])} | "
            f"{_fmt(f['mean_weight_cosine'])} | "
            f"{_fmt(f['mean_null_baseline'])} | "
            f"{_fmt(f['mean_surplus_over_null'])} | "
            f"{_fmt(f['frac_beats_null'])} | "
            f"{_fmt(f['best_surplus'])} | "
            f"{'yes' if f['any_coherence'] else 'no'} | "
            f"{'yes' if f['any_coherence_with_operator_weight'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Full table (family × budget × tensor)")
    lines.append("")
    lines.append(
        "| family | budget | expert | component | bpw | under 1.5? | weight-cos | output-cos | null | surplus | beats null | coh | local-only |"
    )
    lines.append("|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---|---|")
    rows = sorted(
        receipt["rows"],
        key=lambda r: (
            0 if r["under_ceiling"] else 1,
            -float(r["surplus_over_null"]),
            -float(r["output_cosine"]),
            r["family"],
            r["budget_label"],
            r["expert"],
            r["component"],
        ),
    )
    for r in rows:
        lines.append(
            f"| `{r['family']}` | `{r['budget_label']}` | {r['expert']} | `{r['component']}` | "
            f"{r['component_bpw']:.4f} | {'yes' if r['under_ceiling'] else 'no'} | "
            f"{r['weight_cosine']:.4f} | {r['output_cosine']:.4f} | {r['null_baseline']:.4f} | "
            f"{r['surplus_over_null']:+.4f} | {'yes' if r['beats_null'] else 'no'} | "
            f"{'yes' if r['coherence_grade'] else 'no'} | "
            f"{'yes' if r['distribution_local_only'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    ap = receipt["assets"]["activation_provenance"]
    lines.append(f"- Capture run: `{receipt['assets']['capture_run']}`")
    lines.append(f"- Capture result sha256: `{ap['capture_result_sha256']}`")
    lines.append(f"- Hidden source: {ap['hidden_source']}")
    lines.append(f"- Probes: {', '.join(p['probe_id'] for p in ap['probes'])}")
    lines.append(f"- Token-expert pairs: {ap['token_expert_pairs']}")
    lines.append(f"- Experts: {receipt['selection']['experts']} hits={receipt['selection']['hit_counts']}")
    lines.append(f"- Model: `{receipt['assets']['model_dir']}`")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    for k, v in receipt["claim_boundary"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append(
        "Component BPW is the honest per-tensor billed rate for the compressed factors/codes. "
        "It is not a complete-model BPW ledger. No gate was weakened."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE_RUN)
    p.add_argument("--capture-receipt", type=Path, default=DEFAULT_CAPTURE_RECEIPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--experts", type=int, nargs="*", default=None)
    p.add_argument(
        "--components",
        nargs="*",
        default=["gate_proj", "up_proj", "down_proj"],
        choices=["gate_proj", "up_proj", "down_proj"],
    )
    p.add_argument("--min-tokens", type=int, default=32)
    args = p.parse_args(argv)

    receipt = run_probe(
        model_dir=args.model_dir,
        capture_run=args.capture_run,
        capture_receipt=args.capture_receipt,
        out_dir=args.out_dir,
        experts=args.experts,
        components=args.components,
        min_tokens=args.min_tokens,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "verdict": receipt["verdict"],
                "any_coherence_under_ceiling_high_hit_experts": receipt[
                    "any_coherence_under_ceiling_high_hit_experts"
                ],
                "any_operator_recovery_coherence_under_ceiling_high_hit_experts": receipt[
                    "any_operator_recovery_coherence_under_ceiling_high_hit_experts"
                ],
                "mean_null_baseline_high_hit_under_ceiling": receipt[
                    "mean_null_baseline_high_hit_under_ceiling"
                ],
                "joint_surplus_and_operator_reachability_high_hit_experts": receipt[
                    "joint_surplus_and_operator_reachability_high_hit_experts"
                ],
                "joint_surplus_and_operator_reachability_any_expert": receipt[
                    "joint_surplus_and_operator_reachability_any_expert"
                ],
                "family_summary_under_ceiling_high_hit_experts": receipt[
                    "family_summary_under_ceiling_high_hit_experts"
                ],
                "ranking_on_real_activations": receipt["ranking_on_real_activations"],
                "best_high_hit_under_ceiling_by_surplus": receipt[
                    "best_high_hit_under_ceiling_by_surplus"
                ],
                "outputs": receipt.get("outputs"),
                "seconds": receipt["seconds"],
                "n_rows": len(receipt["rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
