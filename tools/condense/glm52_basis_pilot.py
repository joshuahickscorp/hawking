#!/usr/bin/env python3.12
"""Bounded GLM-5.2 real-activation basis pilot (revision 1).

Compares three activation-basis constructions on identical fit/holdout rows,
real retained teacher capsules, and resident immutable source shards:

  1. centered residual SVD  (Generation B production builder)
  2. uncentered SVD
  3. explicit unit mean direction + centered residual SVD orthogonal to it

Revision 1 corrections over the first measured pass:

  * Rank-specific aggregates/floors include a point only when
    total_rank == requested_rank and rank_capped is false.
  * Low-traffic experts are diagnostic-only (not promotion min/median).
  * down_proj is measured twice:
      - production_output_side_down_negative_control (hidden-space output side)
      - activation_matched_input_side_down (Z=SwiGLU input space; promotion)
  * Separate panel aggregates; floors on the promotion-grade panel only.
  * Uncentered vs explicit-mean numerical equivalence tolerance.
  * full_traversal_authorized is false unless every preregistered promotion
    criterion truly passes.

This module is deliberately separate from ``glm52_activation_aware_pack.py`` so
production defaults cannot flip because a pilot arm wins.

    python3.12 tools/condense/glm52_basis_pilot.py selftest
    python3.12 tools/condense/glm52_basis_pilot.py run \\
        --out GLM52_BASIS_PILOT_RECEIPT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_activation_aware_pack as aap  # noqa: E402

SCHEMA = "hawking.glm52.basis_pilot.v1"
REVISION = 1
SEED = 0xB4515  # "BASIS"
HIDDEN = aap.HIDDEN  # 6144
INTERMEDIATE = 2048  # expert / shared MLP intermediate width
HELD_OUT_FRAC = 0.20
HOLDOUT_MIN = 32
DISK_FLOOR_BYTES = 75_000_000_000
DEFAULT_RANKS: tuple[int, ...] = (16, 64, 128, 256, 512)

# Median cosine difference below which uncentered and explicit_mean are tied.
NUMERICAL_EQUIVALENCE_TOLERANCE = 1e-4

PILOT_SOURCE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity/pilot_source"
CAPSULE_DIR = (
    Path.home()
    / "Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/capsules"
)
REHYDRATION_RECEIPT = REPO / "HAWKING_FINAL_ASCENT_SOURCE_REHYDRATION_RECEIPT.json"
REVISION_0_EVIDENCE = REPO / "GLM52_BASIS_PILOT_REVISION_0_EVIDENCE.json"

# Preregistered floors transferred as OBJECTIVES from the Llama-1B calibration.
# Numbers are diagnostic targets for this pilot, not whole-model promotion.
PREREGISTERED = {
    "source": "HAWKING_RESUME_CHECKPOINT.md quality floors (Llama-1B calibration); "
              "transfer the objective, re-measure numbers on GLM",
    "math_live_capability": {
        "min_cosine": 0.70,
        "median_cosine": 0.92,
        "rank_at_least": 256,
    },
    "strong_capability": {
        "min_cosine": 0.74,
        "median_cosine": 0.96,
        "rank_at_least": 512,
    },
    "note": "beats_null and reconstruction error are diagnostics only; "
            "a bounded tensor pilot cannot prove whole-model capability. "
            "Floors apply only to the promotion-grade high-traffic panel with "
            "uncapped calibrated ranks (total_rank == requested_rank).",
}

BASIS_MODES: tuple[str, ...] = ("centered", "uncentered", "explicit_mean")

# Panel membership for aggregates / floors.
PANEL_PROMOTION_GRADE = "promotion_grade_high_traffic_routed"
PANEL_SHARED_MLP = "shared_mlp"
PANEL_ATTENTION_ROUTER = "attention_router_controls"
PANEL_LOW_TRAFFIC = "low_traffic_diagnostics"
PANEL_OTHER = "other"

PROMOTION_GRADE_ORGANS = frozenset({
    "high_traffic_routed_gate",
    "high_traffic_routed_up",
    "high_traffic_routed_down",
})
SHARED_MLP_ORGANS = frozenset({
    "shared_mlp_gate",
    "shared_mlp_up",
    "shared_mlp_down",
})
ATTENTION_ROUTER_ORGANS = frozenset({
    "attention_q",
    "router_control",
})
LOW_TRAFFIC_ORGANS = frozenset({
    "low_traffic_routed_gate",
    "low_traffic_routed_up",
    "low_traffic_routed_down",
})

DOWN_NEG_CONTROL = "production_output_side_down_negative_control"
DOWN_PROMOTION = "activation_matched_input_side_down"

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


class PilotError(RuntimeError):
    """Hard pilot failure (missing real data, hash mismatch, disk floor)."""


# ---------------------------------------------------------------------------
# Hashing / resources
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path_text(path: Path) -> str:
    return sha256_file(path)


def free_disk_bytes(path: Path | None = None) -> int:
    return aap.free_bytes(path)


def estimate_peak_ram_bytes(
    n_rows: int = 4096,
    hidden: int = HIDDEN,
    max_rank: int = 512,
    n_weight_matrices: int = 3,
) -> int:
    """Conservative peak: activations + topk + up to 3 weight mats + 3 bases."""
    x = n_rows * hidden * 4
    topk = n_rows * 8 * 4
    w = n_weight_matrices * INTERMEDIATE * hidden * 4
    bases = 3 * hidden * max_rank * 4
    # Extra Z intermediates for activation-matched down (2048-wide)
    z = n_rows * INTERMEDIATE * 4 * 2
    scratch = n_rows * hidden * 4
    return int(x + topk + w + bases + z + scratch)


def panel_for_organ(organ_class: str) -> str:
    if organ_class in PROMOTION_GRADE_ORGANS:
        return PANEL_PROMOTION_GRADE
    if organ_class in SHARED_MLP_ORGANS:
        return PANEL_SHARED_MLP
    if organ_class in ATTENTION_ROUTER_ORGANS:
        return PANEL_ATTENTION_ROUTER
    if organ_class in LOW_TRAFFIC_ORGANS:
        return PANEL_LOW_TRAFFIC
    return PANEL_OTHER


def is_promotion_eligible_point(pt: dict[str, Any], requested_rank: int) -> bool:
    """A rank-specific aggregate/floor may include a point only when calibrated.

    Requires total_rank == requested_rank and rank_capped == false so a
    route-starved diagnostic cannot masquerade as a calibrated-rank measurement.
    """
    if "mean_row_cosine" not in pt:
        return False
    if int(pt.get("requested_rank", -1)) != int(requested_rank):
        return False
    if bool(pt.get("rank_capped", False)):
        return False
    if int(pt.get("total_rank", -1)) != int(requested_rank):
        return False
    return True


# ---------------------------------------------------------------------------
# Source verification
# ---------------------------------------------------------------------------
def load_rehydration_receipt(path: Path = REHYDRATION_RECEIPT) -> dict[str, Any]:
    if not path.is_file():
        raise PilotError(f"missing rehydration receipt: {path}")
    return json.loads(path.read_text())


def verify_resident_shards(
    pilot_source: Path = PILOT_SOURCE,
    receipt: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Verify each resident shard against the sealed rehydration receipt."""
    if receipt is None:
        receipt = load_rehydration_receipt()
    dest = Path(receipt.get("source", {}).get("destination", pilot_source))
    if not dest.is_dir():
        raise PilotError(f"pilot_source missing: {dest}")
    verified: list[dict[str, Any]] = []
    for shard in receipt["shards"]:
        name = shard["name"]
        path = dest / name
        if not path.is_file():
            raise PilotError(f"resident shard missing: {path}")
        size = path.stat().st_size
        if size != int(shard["bytes"]):
            raise PilotError(
                f"size mismatch for {name}: disk={size} receipt={shard['bytes']}"
            )
        digest = sha256_file(path)
        if digest != shard["sha256"]:
            raise PilotError(
                f"sha256 mismatch for {name}: got={digest} expected={shard['sha256']}"
            )
        verified.append({
            "name": name,
            "path": str(path),
            "bytes": size,
            "sha256": digest,
            "role": shard.get("role"),
            "verified": True,
        })
    return verified


def read_safetensors_header(shard: Path) -> dict[str, Any]:
    return aap.read_safetensors_header(shard)


def read_bf16_tensor(shard: Path, header: dict[str, Any], name: str) -> np.ndarray:
    return aap.read_bf16_tensor(shard, header, name)


def find_tensor_shard(
    tensor_name: str,
    shards: Sequence[Path],
) -> tuple[Path, dict[str, Any]]:
    for shard in shards:
        header = read_safetensors_header(shard)
        if tensor_name in header and isinstance(header[tensor_name], dict):
            if "data_offsets" in header[tensor_name]:
                return shard, header
    raise PilotError(
        f"tensor not resident in pilot_source shards: {tensor_name}. "
        f"Report the missing shard rather than fetching."
    )


# ---------------------------------------------------------------------------
# Capsule loading (real only)
# ---------------------------------------------------------------------------
def capsule_path_for_layer(layer: int, capsule_dir: Path = CAPSULE_DIR) -> Path:
    """Resolve the preferred NPZ that holds pre_router_hidden for ``layer``."""
    cmap = aap.discover_capsule_layers(capsule_dir)
    if layer not in cmap:
        raise PilotError(
            f"no real teacher capsule with pre_router_hidden for layer {layer} "
            f"under {capsule_dir}"
        )
    return cmap[layer][0]


def load_layer_arrays(
    layer: int,
    capsule_dir: Path = CAPSULE_DIR,
) -> dict[str, np.ndarray]:
    """Load only the arrays this pilot needs for one layer (mmap source)."""
    path = capsule_path_for_layer(layer, capsule_dir)
    prefix = f"layer_{layer:02d}/"
    need = {
        "pre_router_hidden": f"{prefix}pre_router_hidden",
        "topk_indices": f"{prefix}topk_indices",
        "attention_input": f"{prefix}attention_input",
    }
    alt = {
        "pre_router_hidden": f"layer_{layer}/pre_router_hidden",
        "topk_indices": f"layer_{layer}/topk_indices",
        "attention_input": f"layer_{layer}/attention_input",
    }
    out: dict[str, np.ndarray] = {"capsule_file": path}  # type: ignore[assignment]
    with np.load(path, mmap_mode="r") as z:
        files = set(z.files)
        for logical, key in need.items():
            use = key if key in files else alt[logical]
            if use not in files:
                if logical == "attention_input":
                    continue
                raise PilotError(f"{path.name} missing {key} (and {alt[logical]})")
            arr = np.asarray(z[use])
            if logical == "pre_router_hidden" or logical == "attention_input":
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.reshape(-1, arr.shape[-1])
                elif arr.ndim != 2:
                    raise PilotError(f"bad shape {arr.shape} for {use}")
            elif logical == "topk_indices":
                arr = np.asarray(arr)
                if arr.ndim == 3:
                    arr = arr.reshape(-1, arr.shape[-1])
                elif arr.ndim != 2:
                    raise PilotError(f"bad topk shape {arr.shape} for {use}")
            out[logical] = arr
        out["capsule_file"] = path  # type: ignore[assignment]
        out["capsule_sha256"] = sha256_file(path)  # type: ignore[assignment]
    return out  # type: ignore[return-value]


def route_row_indices(topk: np.ndarray, expert_id: int) -> np.ndarray:
    """Row indices where ``expert_id`` appears in the top-k route list."""
    if topk.ndim != 2:
        raise PilotError(f"topk must be [N,K], got {topk.shape}")
    mask = (topk == int(expert_id)).any(axis=1)
    return np.flatnonzero(mask).astype(np.int64)


def fit_holdout_indices(
    n: int,
    *,
    seed: int = SEED,
    salt: int = 0,
    held_out_frac: float = HELD_OUT_FRAC,
    holdout_min: int = HOLDOUT_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fit/holdout split. Identical for all basis arms."""
    if n <= 0:
        raise PilotError("need at least one row for fit/holdout")
    rng = np.random.default_rng(int(seed) ^ (int(salt) * 0x9E3779B1))
    perm = rng.permutation(n)
    if n == 1:
        return perm, perm.copy()
    n_hold = max(holdout_min, int(round(n * held_out_frac)))
    n_hold = min(n_hold, max(1, n // 2))
    n_hold = min(n_hold, n - 1)
    hold_idx = np.sort(perm[:n_hold])
    fit_idx = np.sort(perm[n_hold:])
    if fit_idx.size == 0:
        fit_idx = perm.copy()
        hold_idx = perm[: max(1, n // 5)].copy()
    return fit_idx.astype(np.int64), hold_idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Basis builders (the three arms)
# ---------------------------------------------------------------------------
@dataclass
class PilotBasis:
    """Orthonormal columns spanning the activation subspace used to project W."""

    mode: str
    basis: np.ndarray          # [H, r_total] float32, orthonormal columns
    mean_direction: np.ndarray | None  # [H] unit vector when explicit_mean
    singular_values: np.ndarray
    residual_rank: int
    total_rank: int
    X_fit_mean: np.ndarray

    def columns(self, total_rank: int | None = None) -> np.ndarray:
        r = self.total_rank if total_rank is None else min(int(total_rank), self.total_rank)
        if r <= 0:
            raise PilotError("rank must be positive")
        return self.basis[:, :r]


def _as_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise PilotError("zero mean direction; cannot build explicit_mean basis")
    return (v / n).astype(np.float32)


def build_pilot_basis(
    X_fit: np.ndarray,
    mode: str,
    total_rank: int,
) -> PilotBasis:
    """Build one of the three basis variants on the same X_fit rows.

    ``total_rank`` is the total number of stored directions (equal-byte budget).
    For ``explicit_mean`` that budget is 1 mean direction + (total_rank-1)
    residual columns — the mean is NOT free.
    """
    if mode not in BASIS_MODES:
        raise PilotError(f"unknown basis mode {mode!r}")
    X_fit = np.asarray(X_fit, dtype=np.float32)
    if X_fit.ndim != 2:
        raise PilotError(f"X_fit must be 2-D, got {X_fit.shape}")
    n, h = X_fit.shape
    if n < 1:
        raise PilotError("X_fit is empty")
    mu = X_fit.mean(axis=0).astype(np.float32)
    max_r = min(int(total_rank), h, n)
    if max_r < 1:
        raise PilotError("total_rank must be positive")

    if mode == "centered":
        Xc = X_fit - mu
        _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        r = min(max_r, vt.shape[0])
        B = vt[:r].T.astype(np.float32, copy=True)
        return PilotBasis(
            mode=mode,
            basis=B,
            mean_direction=None,
            singular_values=s[:r].astype(np.float32),
            residual_rank=r,
            total_rank=r,
            X_fit_mean=mu,
        )

    if mode == "uncentered":
        _u, s, vt = np.linalg.svd(X_fit, full_matrices=False)
        r = min(max_r, vt.shape[0])
        B = vt[:r].T.astype(np.float32, copy=True)
        return PilotBasis(
            mode=mode,
            basis=B,
            mean_direction=None,
            singular_values=s[:r].astype(np.float32),
            residual_rank=r,
            total_rank=r,
            X_fit_mean=mu,
        )

    # explicit_mean: unit mean + residual SVD in the orthogonal complement
    if max_r < 1:
        raise PilotError("explicit_mean needs total_rank >= 1")
    mdir = _as_unit(mu)
    residual_rank = max_r - 1
    if residual_rank <= 0:
        B = mdir.reshape(-1, 1)
        return PilotBasis(
            mode=mode,
            basis=B,
            mean_direction=mdir,
            singular_values=np.zeros(0, dtype=np.float32),
            residual_rank=0,
            total_rank=1,
            X_fit_mean=mu,
        )
    Xc = X_fit.astype(np.float64) - mu.astype(np.float64)
    proj = Xc @ mdir.astype(np.float64)
    Xc = Xc - proj[:, None] * mdir.astype(np.float64)
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    cols = []
    svals = []
    for i in range(vt.shape[0]):
        v = vt[i].astype(np.float64)
        v = v - float(np.dot(v, mdir.astype(np.float64))) * mdir.astype(np.float64)
        nn = float(np.linalg.norm(v))
        if nn < 1e-8:
            continue
        v = v / nn
        for prev in cols:
            v = v - float(np.dot(v, prev)) * prev
        nn = float(np.linalg.norm(v))
        if nn < 1e-8:
            continue
        v = v / nn
        cols.append(v)
        svals.append(float(s[i]))
        if len(cols) >= residual_rank:
            break
    if not cols:
        B = mdir.reshape(-1, 1)
        return PilotBasis(
            mode=mode,
            basis=B,
            mean_direction=mdir,
            singular_values=np.zeros(0, dtype=np.float32),
            residual_rank=0,
            total_rank=1,
            X_fit_mean=mu,
        )
    R = np.stack(cols, axis=1).astype(np.float32)
    B = np.concatenate([mdir.reshape(-1, 1), R], axis=1)
    Q, _ = np.linalg.qr(B.astype(np.float64), mode="reduced")
    B = Q.astype(np.float32)
    if float(np.dot(B[:, 0], mdir)) < 0:
        B[:, 0] *= -1
    return PilotBasis(
        mode=mode,
        basis=B,
        mean_direction=mdir,
        singular_values=np.asarray(svals, dtype=np.float32),
        residual_rank=int(R.shape[1]),
        total_rank=int(B.shape[1]),
        X_fit_mean=mu,
    )


def centered_matches_production(
    X_fit: np.ndarray,
    total_rank: int,
    atol: float = 1e-4,
) -> bool:
    """Regression: pilot centered arm matches packer SVD of centered X_fit."""
    mu = X_fit.mean(axis=0)
    Xc = X_fit - mu
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(int(total_rank), vt.shape[0], X_fit.shape[1])
    ref = vt[:r].T.astype(np.float32)
    pilot = build_pilot_basis(X_fit, "centered", r)
    G = np.abs(ref.T @ pilot.basis)
    return bool(np.all(np.diag(G) > 1.0 - atol) and pilot.total_rank == r)


# ---------------------------------------------------------------------------
# Byte accounting (equal total encoded bytes)
# ---------------------------------------------------------------------------
HEADER_BYTES = aap.HEADER_BYTES  # 64


def encoded_bytes(
    rows: int,
    cols: int,
    side: str,
    total_rank: int,
    *,
    bill_basis: bool = True,
) -> dict[str, int]:
    """Exact arithmetic float16 cost of coefficients + basis columns + header.

    Scope: per-tensor self-contained payload (header + coefficients + basis).
    This is an exact accounting estimate from the factor formula, not a
    physical on-disk file measurement of a serialized pack artifact.

    All three arms with the same ``total_rank`` bill identically: the explicit
    mean direction is one of the ``total_rank`` basis columns, not an extra.
    """
    parts = aap.factor_bytes(rows, cols, int(total_rank), side)
    total = parts["header"] + parts["coefficients"]
    if bill_basis:
        total += parts["basis"]
    return {
        "header": parts["header"],
        "coefficients": parts["coefficients"],
        "basis": parts["basis"] if bill_basis else 0,
        "total": int(total),
        "accounting_scope": (
            "exact_arithmetic_per_tensor_payload_float16_"
            "header_plus_coefficients_plus_basis"
        ),
        "is_physical_file_measurement": False,
    }


def assert_equal_byte_budget(rows: int, cols: int, side: str, total_rank: int) -> None:
    """Every arm at the same total_rank must price the same."""
    costs = [
        encoded_bytes(rows, cols, side, total_rank)["total"]
        for _mode in BASIS_MODES
    ]
    if len(set(costs)) != 1:
        raise PilotError(f"byte budget mismatch across arms: {costs}")


# ---------------------------------------------------------------------------
# Functional scoring on real activations
# ---------------------------------------------------------------------------
def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def mean_row_cosine(y: np.ndarray, y_hat: np.ndarray) -> float:
    return aap.mean_row_cosine(y, y_hat)


def constant_mean_null(y: np.ndarray) -> float:
    return aap.constant_mean_null(y)


def score_linear(
    W: np.ndarray,
    W_hat: np.ndarray,
    X: np.ndarray,
    *,
    side: str,
) -> dict[str, float]:
    """Score reconstruction on real activation rows. Never builds a Gaussian."""
    if side == "input":
        y = X @ W.T
        y_hat = X @ W_hat.T
    elif side == "output":
        if X.shape[1] != W.shape[1]:
            raise PilotError(
                f"output-side score needs X with width {W.shape[1]}, got {X.shape}"
            )
        y = X @ W.T
        y_hat = X @ W_hat.T
    else:
        raise PilotError(f"unknown side {side}")
    cos = mean_row_cosine(y, y_hat)
    null = constant_mean_null(y)
    recon = float(np.linalg.norm(W - W_hat) / (np.linalg.norm(W) + 1e-12))
    num = float(np.linalg.norm(y - y_hat))
    den = float(np.linalg.norm(y) + 1e-12)
    return {
        "mean_row_cosine": cos,
        "constant_mean_cosine_null": null,
        "beats_null": bool(cos > null),
        "surplus_over_null": cos - null,
        "relative_output_error": num / den,
        "reconstruction_relative_error_INADMISSIBLE": recon,
        "promotional": False,  # beats_null / recon error never promote
    }


def swiglu_intermediate(X: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Real routed SwiGLU intermediate: silu(X @ Wg.T) * (X @ Wu.T)."""
    g = silu(X @ W_gate.T)
    u = X @ W_up.T
    return g * u


def project_and_reconstruct(
    W: np.ndarray,
    basis: PilotBasis,
    total_rank: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    r = min(int(total_rank), basis.total_rank, min(W.shape))
    B = basis.columns(r)
    if side == "input" and B.shape[0] != W.shape[1]:
        raise PilotError(
            f"input-side basis width {B.shape[0]} != W.in {W.shape[1]}"
        )
    if side == "output" and B.shape[0] != W.shape[0]:
        raise PilotError(
            f"output-side basis width {B.shape[0]} != W.out {W.shape[0]}"
        )
    L = aap.project_factors(W, B, side)
    W_hat = aap.reconstruct(L, B, side)
    return W_hat, B


# ---------------------------------------------------------------------------
# Tensor catalog for the bounded pilot
# ---------------------------------------------------------------------------
@dataclass
class TensorSpec:
    name: str
    organ_class: str
    layer: int
    expert_id: int | None
    route_conditioned: bool
    activation_source: str  # pre_router_hidden | attention_input | swiglu_intermediate
    notes: str = ""


def pilot_tensor_catalog() -> list[TensorSpec]:
    """Representative critical classes reachable from the five resident shards."""
    return [
        # High-traffic early
        TensorSpec("model.layers.5.mlp.experts.11.gate_proj.weight",
                   "high_traffic_routed_gate", 5, 11, True, "pre_router_hidden"),
        TensorSpec("model.layers.5.mlp.experts.11.up_proj.weight",
                   "high_traffic_routed_up", 5, 11, True, "pre_router_hidden"),
        TensorSpec("model.layers.5.mlp.experts.11.down_proj.weight",
                   "high_traffic_routed_down", 5, 11, True, "swiglu_intermediate",
                   "SwiGLU intermediate from resident gate/up"),
        TensorSpec("model.layers.5.mlp.experts.165.gate_proj.weight",
                   "high_traffic_routed_gate", 5, 165, True, "pre_router_hidden"),
        TensorSpec("model.layers.5.mlp.experts.165.up_proj.weight",
                   "high_traffic_routed_up", 5, 165, True, "pre_router_hidden"),
        # High-traffic middle
        TensorSpec("model.layers.38.mlp.experts.73.gate_proj.weight",
                   "high_traffic_routed_gate", 38, 73, True, "pre_router_hidden"),
        TensorSpec("model.layers.38.mlp.experts.73.up_proj.weight",
                   "high_traffic_routed_up", 38, 73, True, "pre_router_hidden"),
        TensorSpec("model.layers.38.mlp.experts.73.down_proj.weight",
                   "high_traffic_routed_down", 38, 73, True, "swiglu_intermediate"),
        # High-traffic late
        TensorSpec("model.layers.74.mlp.experts.118.gate_proj.weight",
                   "high_traffic_routed_gate", 74, 118, True, "pre_router_hidden"),
        TensorSpec("model.layers.74.mlp.experts.118.up_proj.weight",
                   "high_traffic_routed_up", 74, 118, True, "pre_router_hidden"),
        TensorSpec("model.layers.74.mlp.experts.118.down_proj.weight",
                   "high_traffic_routed_down", 74, 118, True, "swiglu_intermediate"),
        # Low-traffic control (diagnostic only; not promotion panel)
        TensorSpec("model.layers.5.mlp.experts.100.gate_proj.weight",
                   "low_traffic_routed_gate", 5, 100, True, "pre_router_hidden",
                   "diagnostic only; 205 routes; not a promotion-panel min/median"),
        TensorSpec("model.layers.5.mlp.experts.100.up_proj.weight",
                   "low_traffic_routed_up", 5, 100, True, "pre_router_hidden",
                   "diagnostic only; not a promotion-panel min/median"),
        TensorSpec("model.layers.5.mlp.experts.100.down_proj.weight",
                   "low_traffic_routed_down", 5, 100, True, "swiglu_intermediate",
                   "diagnostic only; not a promotion-panel min/median"),
        # Shared / dense MLP
        TensorSpec("model.layers.38.mlp.shared_experts.gate_proj.weight",
                   "shared_mlp_gate", 38, None, False, "pre_router_hidden"),
        TensorSpec("model.layers.38.mlp.shared_experts.up_proj.weight",
                   "shared_mlp_up", 38, None, False, "pre_router_hidden"),
        TensorSpec("model.layers.38.mlp.shared_experts.down_proj.weight",
                   "shared_mlp_down", 38, None, False, "swiglu_intermediate",
                   "shared SwiGLU intermediate on all pre_router rows"),
        # Attention (real attention_input)
        TensorSpec("model.layers.38.self_attn.q_a_proj.weight",
                   "attention_q", 38, None, False, "attention_input"),
        # Router / control
        TensorSpec("model.layers.38.mlp.gate.weight",
                   "router_control", 38, None, False, "pre_router_hidden"),
    ]


MISSING_CLASSES = [
    {
        "class": "global_embed_tokens",
        "why": "embed_tokens is not among the five resident pilot shards",
    },
    {
        "class": "global_lm_head",
        "why": "lm_head is not among the five resident pilot shards",
    },
    {
        "class": "attention_o_proj_real_intermediate",
        "why": "o_proj is [6144,16384]; capsules retain attention_input/output at "
               "hidden=6144 but not the 16384-wide attention intermediate. "
               "Gaussian input is forbidden for promotion, so o_proj is omitted "
               "rather than scored under a proxy.",
    },
]


# ---------------------------------------------------------------------------
# Arm measurement helpers
# ---------------------------------------------------------------------------
def _measure_arms_on_basis_space(
    W: np.ndarray,
    *,
    basis_X_fit: np.ndarray,
    score_X_hold: np.ndarray,
    projection_side: str,
    score_side: str,
    ranks: Sequence[int],
    route_count: int,
    n_fit: int,
    n_hold: int,
    bill_basis: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Build bases on basis_X_fit, project W, score on score_X_hold."""
    m, n = int(W.shape[0]), int(W.shape[1])
    n_weights = int(np.prod(W.shape))
    max_needed = max(int(r) for r in ranks)
    max_feasible = min(
        max_needed,
        basis_X_fit.shape[0],
        basis_X_fit.shape[1],
        min(W.shape),
    )
    arms: dict[str, list[dict[str, Any]]] = {mode: [] for mode in BASIS_MODES}
    bases: dict[str, PilotBasis] = {}
    for mode in BASIS_MODES:
        bases[mode] = build_pilot_basis(basis_X_fit, mode, max_feasible)

    for rank in ranks:
        r = min(int(rank), max_feasible)
        if r < 1:
            continue
        capped = int(rank) > max_feasible or r != int(rank)
        for mode in BASIS_MODES:
            basis = bases[mode]
            if basis.total_rank < r:
                basis = build_pilot_basis(basis_X_fit, mode, r)
                bases[mode] = basis
            # Use actual total columns available after build
            r_eff = min(r, basis.total_rank)
            capped_eff = int(rank) != r_eff
            W_hat, B = project_and_reconstruct(W, basis, r_eff, projection_side)
            bytes_doc = encoded_bytes(m, n, projection_side, r_eff, bill_basis=bill_basis)
            score = score_linear(W, W_hat, score_X_hold, side=score_side)
            point = {
                "requested_rank": int(rank),
                "total_rank": int(r_eff),
                "rank_capped": bool(capped_eff),
                "max_feasible": int(max_feasible),
                "residual_rank": (
                    r_eff - 1 if mode == "explicit_mean" and r_eff >= 1 else r_eff
                ),
                "bytes": bytes_doc,
                "bpw": (bytes_doc["total"] * 8) / max(n_weights, 1),
                "byte_accounting_scope": bytes_doc["accounting_scope"],
                "route_count": route_count,
                "n_fit": n_fit,
                "n_hold": n_hold,
                "mean_included": mode == "explicit_mean",
                "projection_side": projection_side,
                "basis_width": int(basis_X_fit.shape[1]),
                "score_width": int(score_X_hold.shape[1]),
                **score,
            }
            if mode == "explicit_mean" and basis.mean_direction is not None and r_eff >= 2:
                Br = basis.columns(r_eff)
                dots = Br[:, 1:].T @ basis.mean_direction
                point["max_abs_mean_dot_residual"] = float(np.max(np.abs(dots)))
                gram = Br.T @ Br
                point["basis_orthogonality_err"] = float(
                    np.max(np.abs(gram - np.eye(r_eff, dtype=np.float32)))
                )
            arms[mode].append(point)
    return arms, int(max_feasible)


def choose_side(shape: tuple[int, ...], hidden: int = HIDDEN) -> str | None:
    return aap.choose_side(shape, hidden=hidden)


def evaluate_tensor(
    spec: TensorSpec,
    W: np.ndarray,
    *,
    X_all: np.ndarray,
    fit_idx: np.ndarray,
    hold_idx: np.ndarray,
    ranks: Sequence[int],
    W_gate: np.ndarray | None = None,
    W_up: np.ndarray | None = None,
    route_count: int,
    invalid_all_row_hold_idx: np.ndarray | None = None,
    invalid_all_row_X: np.ndarray | None = None,
    bill_basis: bool = True,
) -> dict[str, Any]:
    """Evaluate all three basis arms at each rank on identical rows.

    For down_proj (swiglu_intermediate), measures BOTH:
      * production_output_side_down_negative_control
      * activation_matched_input_side_down  (promotion metric)
    """
    panel = panel_for_organ(spec.organ_class)
    X_fit = X_all[fit_idx]
    X_hold = X_all[hold_idx]
    n_weights = int(np.prod(W.shape))
    m, n = int(W.shape[0]), int(W.shape[1])
    n_fit = int(X_fit.shape[0])
    n_hold = int(X_hold.shape[0])
    fit_hash = sha256_bytes(fit_idx.astype(np.int64).tobytes())
    hold_hash = sha256_bytes(hold_idx.astype(np.int64).tobytes())

    base_meta = {
        "name": spec.name,
        "organ_class": spec.organ_class,
        "panel": panel,
        "layer": spec.layer,
        "expert_id": spec.expert_id,
        "route_conditioned": spec.route_conditioned,
        "route_count": route_count,
        "activation_source": spec.activation_source,
        "shape": [m, n],
        "n_weights": n_weights,
        "n_fit": n_fit,
        "n_hold": n_hold,
        "fit_idx_sha256": fit_hash,
        "hold_idx_sha256": hold_hash,
        "notes": spec.notes,
        "promotional_panel_member": panel == PANEL_PROMOTION_GRADE,
    }

    # ------------------------------------------------------------------
    # down_proj: dual analysis
    # ------------------------------------------------------------------
    if spec.activation_source == "swiglu_intermediate":
        if W_gate is None or W_up is None:
            return {
                **base_meta,
                "status": "SKIPPED_MISSING_GATE_UP",
                "why": "down_proj requires resident gate/up for real SwiGLU input",
            }
        if "down_proj" not in spec.name:
            return {
                **base_meta,
                "status": "SKIPPED_BAD_SPEC",
                "why": "swiglu_intermediate only valid for down_proj",
            }

        # Derive real Z_fit / Z_hold from the same route-conditioned X rows.
        Z_fit = swiglu_intermediate(X_fit, W_gate, W_up)
        Z_hold = swiglu_intermediate(X_hold, W_gate, W_up)
        if Z_fit.shape[1] != INTERMEDIATE:
            # Still allow if intermediate width matches W.in
            if Z_fit.shape[1] != W.shape[1]:
                raise PilotError(
                    f"SwiGLU width {Z_fit.shape[1]} != W.in {W.shape[1]}"
                )

        # --- Negative control: production output-side representation ---
        # Basis from 6144-wide residual activations; project W on output side;
        # score with real Z_hold as linear input. Decisive negative evidence
        # for that production representation — preserved, not promotional.
        neg_arms, neg_max = _measure_arms_on_basis_space(
            W,
            basis_X_fit=X_fit,
            score_X_hold=Z_hold,
            projection_side="output",
            score_side="output",
            ranks=ranks,
            route_count=route_count,
            n_fit=n_fit,
            n_hold=n_hold,
            bill_basis=bill_basis,
        )

        # --- Promotion metric: activation-matched input-side ---
        # Bases in 2048-wide Z_fit input space; project down on input side;
        # score on Z_hold. Equal total direction count and exact byte formula.
        prom_arms, prom_max = _measure_arms_on_basis_space(
            W,
            basis_X_fit=Z_fit,
            score_X_hold=Z_hold,
            projection_side="input",
            score_side="input",
            ranks=ranks,
            route_count=route_count,
            n_fit=n_fit,
            n_hold=n_hold,
            bill_basis=bill_basis,
        )

        # Prove equal bytes across the two analyses at each equal total_rank.
        equal_bytes_ok = True
        for mode in BASIS_MODES:
            for pn, nn in zip(prom_arms[mode], neg_arms[mode]):
                if pn["total_rank"] == nn["total_rank"]:
                    if pn["bytes"]["total"] != nn["bytes"]["total"]:
                        equal_bytes_ok = False

        down_analyses = {
            DOWN_NEG_CONTROL: {
                "label": DOWN_NEG_CONTROL,
                "promotional": False,
                "basis_space": "pre_router_hidden",
                "basis_width": int(X_fit.shape[1]),
                "projection_side": "output",
                "score_space": "swiglu_intermediate",
                "score_width": int(Z_hold.shape[1]),
                "max_feasible_rank": neg_max,
                "arms": neg_arms,
                "note": (
                    "Production representation: basis from residual activations, "
                    "project down on its output side. Preserved as decisive "
                    "negative evidence; NOT the down promotion metric."
                ),
            },
            DOWN_PROMOTION: {
                "label": DOWN_PROMOTION,
                "promotional": panel == PANEL_PROMOTION_GRADE,
                "basis_space": "swiglu_intermediate",
                "basis_width": int(Z_fit.shape[1]),
                "projection_side": "input",
                "score_space": "swiglu_intermediate",
                "score_width": int(Z_hold.shape[1]),
                "max_feasible_rank": prom_max,
                "arms": prom_arms,
                "note": (
                    "Activation-matched: Z_fit/Z_hold from route-conditioned "
                    "X via resident gate/up; bases in intermediate input space; "
                    "project down on input side. This is the down promotion metric."
                ),
            },
        }

        return {
            **base_meta,
            "status": "MEASURED",
            "side": "input",  # promotion projection side
            "intermediate_note": "real_swiglu_from_resident_gate_up",
            "max_feasible_rank": prom_max,
            "down_analyses": down_analyses,
            "promotion_metric": DOWN_PROMOTION,
            "negative_control_metric": DOWN_NEG_CONTROL,
            "equal_bytes_across_down_analyses": equal_bytes_ok,
            # Promotion-facing arms for aggregation convenience
            "arms": prom_arms,
            "invalid_all_row_score_diagnostic": {
                "note": "INVALID for down_proj promotion; dual down analyses "
                        "already use route-conditioned real SwiGLU rows only.",
                "status": "NOT_APPLICABLE",
            },
            "gaussian_proxy_used": False,
        }

    # ------------------------------------------------------------------
    # gate / up / attention / router (input-side residual activations)
    # ------------------------------------------------------------------
    hidden_width = int(X_all.shape[1])
    side = choose_side(tuple(W.shape), hidden=hidden_width)
    if side is None:
        return {
            **base_meta,
            "status": "SKIPPED_NO_SIDE",
            "why": "tensor side does not match activation hidden width",
        }
    if side == "output":
        return {
            **base_meta,
            "status": "SKIPPED_NO_REAL_INPUT",
            "why": "output-side tensor lacks real input activations; Gaussian forbidden",
        }

    arms, max_feasible = _measure_arms_on_basis_space(
        W,
        basis_X_fit=X_fit,
        score_X_hold=X_hold,
        projection_side=side,
        score_side=side,
        ranks=ranks,
        route_count=route_count,
        n_fit=n_fit,
        n_hold=n_hold,
        bill_basis=bill_basis,
    )

    # Invalid all-row diagnostic (not promotion evidence)
    invalid_diag = None
    if (
        spec.route_conditioned
        and invalid_all_row_hold_idx is not None
        and invalid_all_row_X is not None
        and side == "input"
    ):
        mode = "centered"
        r = min(64, max_feasible)
        basis = build_pilot_basis(X_fit, mode, r)
        W_hat, _ = project_and_reconstruct(W, basis, r, side)
        X_bad = invalid_all_row_X[invalid_all_row_hold_idx]
        inv_score = score_linear(W, W_hat, X_bad, side=side)
        invalid_diag = {
            "note": "INVALID promotion evidence: holdout rows not restricted to "
                    "this expert's routes. Reported only as a defect diagnostic.",
            "rank": r,
            "promotional": False,
            **inv_score,
        }

    return {
        **base_meta,
        "status": "MEASURED",
        "side": side,
        "intermediate_note": None,
        "max_feasible_rank": max_feasible,
        "arms": arms,
        "invalid_all_row_score_diagnostic": invalid_diag,
        "gaussian_proxy_used": False,
    }


# ---------------------------------------------------------------------------
# Aggregation / verdict
# ---------------------------------------------------------------------------
def _iter_measured_rows(
    results: list[dict[str, Any]],
    *,
    panel: str | None = None,
) -> Iterable[dict[str, Any]]:
    for row in results:
        if row.get("status") != "MEASURED":
            continue
        if panel is not None and row.get("panel") != panel:
            continue
        yield row


def _collect_rank_points(
    results: list[dict[str, Any]],
    mode: str,
    rank: int,
    *,
    panel: str | None = None,
    require_uncapped: bool = True,
) -> tuple[list[float], int, int, list[dict[str, Any]]]:
    """Collect cosines for a rank.

    Returns (values, n_included, n_excluded, exclusion_details).
    When require_uncapped, only points with total_rank == requested_rank and
    rank_capped == false are included.
    """
    vals: list[float] = []
    n_excl = 0
    details: list[dict[str, Any]] = []
    for row in _iter_measured_rows(results, panel=panel):
        pts = row.get("arms", {}).get(mode, [])
        matched = [p for p in pts if int(p.get("requested_rank", -1)) == int(rank)]
        if not matched:
            continue
        pt = matched[0]
        if require_uncapped and not is_promotion_eligible_point(pt, rank):
            n_excl += 1
            details.append({
                "name": row["name"],
                "organ_class": row.get("organ_class"),
                "panel": row.get("panel"),
                "requested_rank": pt.get("requested_rank"),
                "total_rank": pt.get("total_rank"),
                "rank_capped": pt.get("rank_capped"),
                "reason": (
                    "rank_capped_or_total_rank_ne_requested"
                    if pt.get("rank_capped") or int(pt.get("total_rank", -1)) != int(rank)
                    else "missing_cosine"
                ),
                "mean_row_cosine_diagnostic_only": pt.get("mean_row_cosine"),
            })
            continue
        if "mean_row_cosine" not in pt:
            n_excl += 1
            continue
        vals.append(float(pt["mean_row_cosine"]))
    return vals, len(vals), n_excl, details


def _stats_from_values(
    vals: list[float],
    *,
    n_included: int,
    n_excluded: int,
    excluded_tensors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not vals:
        return {
            "n": 0,
            "n_included": 0,
            "n_excluded": n_excluded,
            "excluded_tensors": excluded_tensors or [],
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "n_included": n_included,
        "n_excluded": n_excluded,
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "worst_case": float(arr.min()),
        "excluded_tensors": excluded_tensors or [],
    }


def summarize_mode_panel(
    results: list[dict[str, Any]],
    mode: str,
    ranks: Sequence[int],
    panel: str | None,
    *,
    require_uncapped: bool = True,
) -> dict[str, Any]:
    by_rank: dict[str, Any] = {}
    for rank in ranks:
        vals, n_inc, n_exc, details = _collect_rank_points(
            results, mode, rank, panel=panel, require_uncapped=require_uncapped,
        )
        by_rank[str(rank)] = _stats_from_values(
            vals, n_included=n_inc, n_excluded=n_exc, excluded_tensors=details,
        )

    per_organ: dict[str, Any] = {}
    for row in _iter_measured_rows(results, panel=panel):
        oc = row["organ_class"]
        per_organ.setdefault(oc, [])
        pts = row["arms"][mode]
        # Prefer uncapped rank-256; else highest uncapped; else highest scored diagnostic
        uncapped = [
            p for p in pts
            if "mean_row_cosine" in p and is_promotion_eligible_point(p, int(p["requested_rank"]))
        ]
        preferred = [p for p in uncapped if int(p["requested_rank"]) == 256]
        if preferred:
            pick = preferred[0]
            pick_kind = "uncapped_rank_256"
        elif uncapped:
            pick = max(uncapped, key=lambda p: int(p["requested_rank"]))
            pick_kind = "highest_uncapped"
        else:
            scored = [p for p in pts if "mean_row_cosine" in p]
            if not scored:
                continue
            pick = max(scored, key=lambda p: int(p["total_rank"]))
            pick_kind = "diagnostic_capped_only"
        entry = {
            "name": row["name"],
            "layer": row["layer"],
            "requested_rank": pick["requested_rank"],
            "total_rank": pick["total_rank"],
            "rank_capped": pick["rank_capped"],
            "pick_kind": pick_kind,
            "cosine": pick["mean_row_cosine"],
            "null": pick["constant_mean_cosine_null"],
            "bpw": pick["bpw"],
            "bytes": pick["bytes"]["total"],
            "byte_accounting_scope": pick.get(
                "byte_accounting_scope",
                pick["bytes"].get("accounting_scope"),
            ),
            "route_count": row["route_count"],
            "panel": row.get("panel"),
        }
        # Surface negative-control down cosine when present
        if "down_analyses" in row:
            neg = row["down_analyses"][DOWN_NEG_CONTROL]["arms"][mode]
            neg_pts = [
                p for p in neg
                if int(p["requested_rank"]) == int(pick["requested_rank"])
            ]
            if neg_pts:
                entry["negative_control_cosine"] = neg_pts[0]["mean_row_cosine"]
                entry["promotion_down_analysis"] = DOWN_PROMOTION
                entry["negative_control_down_analysis"] = DOWN_NEG_CONTROL
        per_organ[oc].append(entry)
    return {"by_rank": by_rank, "per_organ": per_organ, "panel": panel}


def summarize_all_panels(
    results: list[dict[str, Any]],
    mode: str,
    ranks: Sequence[int],
) -> dict[str, Any]:
    panels = {
        PANEL_PROMOTION_GRADE: summarize_mode_panel(
            results, mode, ranks, PANEL_PROMOTION_GRADE, require_uncapped=True,
        ),
        PANEL_SHARED_MLP: summarize_mode_panel(
            results, mode, ranks, PANEL_SHARED_MLP, require_uncapped=True,
        ),
        PANEL_ATTENTION_ROUTER: summarize_mode_panel(
            results, mode, ranks, PANEL_ATTENTION_ROUTER, require_uncapped=True,
        ),
        PANEL_LOW_TRAFFIC: summarize_mode_panel(
            results, mode, ranks, PANEL_LOW_TRAFFIC, require_uncapped=True,
        ),
        # Legacy all-tensor diagnostic (still uncapped-only; excludes capped)
        "all_measured_uncapped": summarize_mode_panel(
            results, mode, ranks, None, require_uncapped=True,
        ),
    }
    return panels


def distinguish_verdict(
    results: list[dict[str, Any]],
    ranks: Sequence[int],
) -> dict[str, Any]:
    """Which arm wins on equal-byte held-out cosine, and by how much.

    Floors evaluate the promotion-grade high-traffic panel only. Low-traffic
    and capped-rank points remain visible but cannot clear a floor.
    """
    panel_summaries = {
        mode: summarize_all_panels(results, mode, ranks) for mode in BASIS_MODES
    }
    # Convenience: promotion-grade as the primary summary surface
    summaries = {
        mode: panel_summaries[mode][PANEL_PROMOTION_GRADE]
        for mode in BASIS_MODES
    }

    focus = [r for r in ranks if r in (64, 128, 256, 512)]
    if not focus:
        focus = list(ranks)
    pairwise: list[dict[str, Any]] = []
    for rank in focus:
        med = {
            mode: summaries[mode]["by_rank"].get(str(rank), {}).get("median")
            for mode in BASIS_MODES
        }
        mn = {
            mode: summaries[mode]["by_rank"].get(str(rank), {}).get("min")
            for mode in BASIS_MODES
        }
        n_inc = {
            mode: summaries[mode]["by_rank"].get(str(rank), {}).get("n_included", 0)
            for mode in BASIS_MODES
        }
        n_exc = {
            mode: summaries[mode]["by_rank"].get(str(rank), {}).get("n_excluded", 0)
            for mode in BASIS_MODES
        }
        if any(v is None for v in med.values()):
            continue
        # Do not declare a unique winner between uncentered and explicit_mean
        # when their medians are within numerical-equivalence tolerance.
        em = med["explicit_mean"]
        uc = med["uncentered"]
        ce = med["centered"]
        if abs(em - uc) < NUMERICAL_EQUIVALENCE_TOLERANCE:
            if max(em, uc) - ce >= 0.02:
                winner = "uncentered_or_explicit_mean_tied"
            else:
                winner = "no_material_gap"
        else:
            winner = max(med.keys(), key=lambda m: (med[m], mn[m]))
        pairwise.append({
            "rank": rank,
            "panel": PANEL_PROMOTION_GRADE,
            "median_cosine": med,
            "min_cosine": mn,
            "n_included": n_inc,
            "n_excluded": n_exc,
            "winner_by_median": winner,
            "explicit_mean_minus_centered_median": em - ce,
            "uncentered_minus_centered_median": uc - ce,
            "explicit_mean_minus_uncentered_median": em - uc,
            "numerical_equivalence_tolerance": NUMERICAL_EQUIVALENCE_TOLERANCE,
        })

    # Floors ONLY on promotion-grade panel with uncapped ranks
    floor_checks = []
    for label, floors in (
        ("math_live_capability", PREREGISTERED["math_live_capability"]),
        ("strong_capability", PREREGISTERED["strong_capability"]),
    ):
        r = int(floors["rank_at_least"])
        for mode in BASIS_MODES:
            st = summaries[mode]["by_rank"].get(str(r), {})
            if not st or st.get("n_included", 0) == 0:
                floor_checks.append({
                    "floor": label,
                    "mode": mode,
                    "rank": r,
                    "panel": PANEL_PROMOTION_GRADE,
                    "status": "NOT_MEASURED",
                    "n_included": st.get("n_included", 0) if st else 0,
                    "n_excluded": st.get("n_excluded", 0) if st else 0,
                    "note": "promotion-grade uncapped points only",
                })
                continue
            ok = (
                st["min"] >= float(floors["min_cosine"])
                and st["median"] >= float(floors["median_cosine"])
            )
            floor_checks.append({
                "floor": label,
                "mode": mode,
                "rank": r,
                "panel": PANEL_PROMOTION_GRADE,
                "status": "CLEARS" if ok else "FAILS",
                "min": st["min"],
                "median": st["median"],
                "required_min": floors["min_cosine"],
                "required_median": floors["median_cosine"],
                "n_included": st["n_included"],
                "n_excluded": st["n_excluded"],
                "excluded_tensors": st.get("excluded_tensors", []),
                "note": (
                    "promotion-grade high-traffic panel; "
                    "low-traffic and capped ranks excluded"
                ),
            })

    # Per-organ failures remain visible on promotion panel
    per_organ_failures: list[dict[str, Any]] = []
    for mode in BASIS_MODES:
        for oc, entries in summaries[mode].get("per_organ", {}).items():
            for e in entries:
                if e.get("cosine", 1.0) < 0.70:
                    per_organ_failures.append({
                        "mode": mode,
                        "organ_class": oc,
                        "name": e["name"],
                        "cosine": e["cosine"],
                        "total_rank": e["total_rank"],
                        "rank_capped": e["rank_capped"],
                        "panel": e.get("panel"),
                        "negative_control_cosine": e.get("negative_control_cosine"),
                    })

    # Dominant story with numerical-equivalence handling
    if pairwise:
        em_lifts = [p["explicit_mean_minus_centered_median"] for p in pairwise]
        uc_lifts = [p["uncentered_minus_centered_median"] for p in pairwise]
        em_uc = [p["explicit_mean_minus_uncentered_median"] for p in pairwise]
        mean_em = float(sum(em_lifts) / len(em_lifts))
        mean_uc = float(sum(uc_lifts) / len(uc_lifts))
        mean_em_uc = float(sum(em_uc) / len(em_uc))
    else:
        mean_em = mean_uc = mean_em_uc = float("nan")

    em_uc_tied = (
        pairwise
        and abs(mean_em_uc) < NUMERICAL_EQUIVALENCE_TOLERANCE
    )
    if pairwise and abs(mean_em) < 0.01 and abs(mean_uc) < 0.01:
        story = (
            "NO_MATERIAL_BASIS_GAP: on this bounded real-capsule promotion-grade "
            "panel, centered, uncentered, and explicit-mean arms are within ~0.01 "
            "median cosine. Centering alone does not explain Generation B's "
            "collapse here."
        )
        story_code = "NO_MATERIAL_BASIS_GAP"
    elif pairwise and em_uc_tied and max(mean_em, mean_uc) >= 0.02:
        story = (
            "RETAINING_MEAN_HELPS_CENTERED_RESIDUAL: uncentered and equal-byte "
            "explicit-mean are numerically tied "
            f"(|median lift difference| < {NUMERICAL_EQUIVALENCE_TOLERANCE}). "
            "Retaining the mean helps vs centered residual on held-out real "
            "activations. The implementation choice between uncentered and "
            "explicit-mean remains unresolved when B≈C. Production centering is "
            "a real defect; this is not whole-model capability proof."
        )
        story_code = "RETAINING_MEAN_HELPS_CENTERED_RESIDUAL"
    elif pairwise and mean_em >= 0.02 and mean_em > mean_uc + NUMERICAL_EQUIVALENCE_TOLERANCE:
        story = (
            "EXPLICIT_MEAN_HELPS: equal-byte explicit mean + residual beats "
            "centered residual and uncentered on held-out real activations "
            "beyond numerical-equivalence tolerance. Production centering is a "
            "real defect, but this is not whole-model capability proof."
        )
        story_code = "EXPLICIT_MEAN_HELPS"
    elif pairwise and mean_uc >= 0.02 and mean_uc > mean_em + NUMERICAL_EQUIVALENCE_TOLERANCE:
        story = (
            "UNCENTERED_HELPS: uncentered SVD beats centered residual and "
            "explicit-mean on held-out real activations beyond "
            "numerical-equivalence tolerance. Production centering is a real "
            "defect, but this is not whole-model capability proof."
        )
        story_code = "UNCENTERED_HELPS"
    else:
        story = (
            "MIXED_OR_SMALL: arm differences exist but are small or "
            "rank-dependent. See per-rank tables. Do not flip production "
            "defaults from this pilot alone."
        )
        story_code = "MIXED_OR_SMALL"

    # full_traversal_authorized only if every preregistered promotion criterion passes
    floor_clear = all(fc.get("status") == "CLEARS" for fc in floor_checks)
    full_traversal_authorized = bool(floor_clear and floor_checks)
    # Hard safety: never authorize full traversal from this bounded pilot unless
    # floors truly clear — and even then the safety block keeps the fence false
    # in the top-level safety dict. The verdict field records the scientific
    # authorization bit independently.
    if not floor_clear:
        full_traversal_authorized = False

    # Down negative-control snapshot (preserve first-pass finding)
    down_neg_snapshot: list[dict[str, Any]] = []
    down_prom_snapshot: list[dict[str, Any]] = []
    for row in _iter_measured_rows(results):
        if "down_analyses" not in row:
            continue
        for label, bucket in (
            (DOWN_NEG_CONTROL, down_neg_snapshot),
            (DOWN_PROMOTION, down_prom_snapshot),
        ):
            arms = row["down_analyses"][label]["arms"]
            for mode in BASIS_MODES:
                for p in arms[mode]:
                    if int(p["requested_rank"]) == 256:
                        bucket.append({
                            "name": row["name"],
                            "organ_class": row["organ_class"],
                            "panel": row.get("panel"),
                            "mode": mode,
                            "analysis": label,
                            "requested_rank": p["requested_rank"],
                            "total_rank": p["total_rank"],
                            "rank_capped": p["rank_capped"],
                            "cosine": p["mean_row_cosine"],
                            "basis_width": p.get("basis_width"),
                            "projection_side": p.get("projection_side"),
                        })

    return {
        "summaries": summaries,
        "panel_summaries": panel_summaries,
        "pairwise_at_focus_ranks": pairwise,
        "floor_checks": floor_checks,
        "per_organ_failures_visible": per_organ_failures,
        "mean_median_lift_explicit_mean_over_centered": mean_em,
        "mean_median_lift_uncentered_over_centered": mean_uc,
        "mean_median_lift_explicit_mean_over_uncentered": mean_em_uc,
        "numerical_equivalence_tolerance": NUMERICAL_EQUIVALENCE_TOLERANCE,
        "uncentered_explicit_mean_numerically_tied": bool(em_uc_tied),
        "distinguishing_story": story,
        "distinguishing_story_code": story_code,
        "full_traversal_authorized": False if not full_traversal_authorized else True,
        "down_negative_control_rank256": down_neg_snapshot,
        "down_promotion_metric_rank256": down_prom_snapshot,
        "non_promotional_diagnostics": [
            "beats_null",
            "reconstruction_relative_error_INADMISSIBLE",
            "invalid_all_row_score_diagnostic",
            DOWN_NEG_CONTROL,
            "low_traffic_diagnostics panel",
            "capped-rank points (total_rank != requested_rank)",
        ],
        "not_claimed": [
            "whole-model capability",
            "permission to change production defaults",
            "permission to start a 282-shard traversal",
            "unique superiority of explicit_mean over uncentered when B≈C",
        ],
    }


# ---------------------------------------------------------------------------
# Main pilot run
# ---------------------------------------------------------------------------
def _layer_cache_get(
    cache: dict[int, dict[str, Any]],
    layer: int,
    capsule_dir: Path,
) -> dict[str, Any]:
    if layer not in cache:
        cache[layer] = load_layer_arrays(layer, capsule_dir)
    return cache[layer]


def load_revision_0_evidence() -> dict[str, Any] | None:
    if REVISION_0_EVIDENCE.is_file():
        return json.loads(REVISION_0_EVIDENCE.read_text())
    # Fall back to hashing an existing receipt if present
    prior = REPO / "GLM52_BASIS_PILOT_RECEIPT.json"
    if prior.is_file():
        raw = prior.read_bytes()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "label": "revision_0",
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
                "note": "prior receipt present but not JSON-parseable",
            }
        if d.get("revision", 0) == 0 or "revision" not in d:
            return {
                "label": "revision_0",
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
                "at": d.get("at"),
                "schema": d.get("schema"),
                "distinguishing_story": d.get("verdict", {}).get("distinguishing_story"),
                "floor_checks": d.get("verdict", {}).get("floor_checks"),
                "note": "embedded from pre-revision receipt on disk",
            }
    return None


def run_pilot(
    *,
    pilot_source: Path = PILOT_SOURCE,
    capsule_dir: Path = CAPSULE_DIR,
    ranks: Sequence[int] = DEFAULT_RANKS,
    seed: int = SEED,
    out_json: Path | None = None,
    out_md: Path | None = None,
    skip_hash_verify: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    free = free_disk_bytes(pilot_source)
    peak_est = estimate_peak_ram_bytes()
    if free < DISK_FLOOR_BYTES:
        raise PilotError(
            f"disk floor: free={free} < required {DISK_FLOOR_BYTES}. "
            "Refuse to run; recover space without releasing pilot source shards."
        )

    rev0 = load_revision_0_evidence()

    receipt = load_rehydration_receipt()
    if skip_hash_verify:
        verified = [
            {
                "name": s["name"],
                "path": str(pilot_source / s["name"]),
                "bytes": s["bytes"],
                "sha256": s["sha256"],
                "role": s.get("role"),
                "verified": False,
                "note": "hash verify skipped by flag",
            }
            for s in receipt["shards"]
        ]
    else:
        verified = verify_resident_shards(pilot_source, receipt)

    shard_paths = [Path(v["path"]) for v in verified]
    catalog = pilot_tensor_catalog()
    layer_cache: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    capsule_hashes: dict[str, str] = {}

    needed_layers = sorted({s.layer for s in catalog})
    for layer in needed_layers:
        arrs = _layer_cache_get(layer_cache, layer, capsule_dir)
        cap_path: Path = arrs["capsule_file"]  # type: ignore[assignment]
        capsule_hashes[cap_path.name] = str(arrs["capsule_sha256"])

    header_cache: dict[Path, dict[str, Any]] = {}

    def load_W(name: str) -> np.ndarray:
        shard, header = find_tensor_shard(name, shard_paths)
        if shard not in header_cache:
            header_cache[shard] = header
        return read_bf16_tensor(shard, header_cache[shard], name)

    for spec in catalog:
        arrs = layer_cache[spec.layer]
        X_full = arrs["pre_router_hidden"]
        if spec.activation_source == "attention_input":
            if "attention_input" not in arrs:
                results.append({
                    "name": spec.name,
                    "organ_class": spec.organ_class,
                    "panel": panel_for_organ(spec.organ_class),
                    "status": "SKIPPED_NO_ATTENTION_INPUT",
                    "why": "capsule lacks attention_input for this layer",
                })
                continue
            X_full = arrs["attention_input"]

        if spec.route_conditioned:
            assert spec.expert_id is not None
            topk = arrs["topk_indices"]
            routes = route_row_indices(topk, spec.expert_id)
            route_count = int(routes.size)
            if route_count < 4:
                results.append({
                    "name": spec.name,
                    "organ_class": spec.organ_class,
                    "panel": panel_for_organ(spec.organ_class),
                    "status": "SKIPPED_TOO_FEW_ROUTES",
                    "route_count": route_count,
                    "why": "need enough route-conditioned rows for fit/holdout",
                })
                continue
            X_all = X_full[routes]
            salt = (spec.layer * 1009) ^ (spec.expert_id * 9176)
            fit_local, hold_local = fit_holdout_indices(
                X_all.shape[0], seed=seed, salt=salt
            )
            _fit_all, hold_all = fit_holdout_indices(
                X_full.shape[0], seed=seed, salt=spec.layer
            )
            invalid_hold = hold_all
        else:
            route_count = int(X_full.shape[0])
            X_all = X_full
            salt = spec.layer * 1009
            fit_local, hold_local = fit_holdout_indices(
                X_all.shape[0], seed=seed, salt=salt
            )
            invalid_hold = None

        try:
            W = load_W(spec.name)
        except PilotError as e:
            results.append({
                "name": spec.name,
                "organ_class": spec.organ_class,
                "panel": panel_for_organ(spec.organ_class),
                "status": "SKIPPED_NOT_RESIDENT",
                "why": str(e),
            })
            continue

        W_gate = W_up = None
        if spec.activation_source == "swiglu_intermediate":
            if "down_proj" not in spec.name:
                results.append({
                    "name": spec.name,
                    "status": "SKIPPED_BAD_SPEC",
                    "why": "swiglu_intermediate only valid for down_proj",
                })
                continue
            gate_name = spec.name.replace("down_proj", "gate_proj")
            up_name = spec.name.replace("down_proj", "up_proj")
            try:
                W_gate = load_W(gate_name)
                W_up = load_W(up_name)
            except PilotError as e:
                results.append({
                    "name": spec.name,
                    "organ_class": spec.organ_class,
                    "panel": panel_for_organ(spec.organ_class),
                    "status": "SKIPPED_MISSING_GATE_UP",
                    "why": str(e),
                })
                del W
                continue

        row = evaluate_tensor(
            spec,
            W,
            X_all=X_all,
            fit_idx=fit_local,
            hold_idx=hold_local,
            ranks=ranks,
            W_gate=W_gate,
            W_up=W_up,
            route_count=route_count,
            invalid_all_row_hold_idx=invalid_hold,
            invalid_all_row_X=X_full if spec.route_conditioned else None,
        )
        results.append(row)
        del W, W_gate, W_up

    verdict = distinguish_verdict(results, ranks)

    # Exact panel-total encoded bytes (arithmetic accounting, not physical file)
    panel_byte_totals = _panel_byte_totals(results, rank=256)

    code_hash = sha256_path_text(Path(__file__))
    pack_hash = sha256_path_text(HERE / "glm52_activation_aware_pack.py")

    # Safety fences: always false for this bounded pilot session.
    safety = {
        "full_parent_traversal_started": False,
        "teacher_capsules_modified": False,
        "prior_artifacts_modified": False,
        "MOP_touched": False,
        "production_defaults_changed": False,
        "ODYSSEY_LAUNCH_AUTHORIZED": False,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "HIDE_KERNEL_TURN": False,
        "gaussian_proxy_used_for_selection": False,
        "full_traversal_authorized": False,  # fence; see also verdict bit
    }

    next_action = _next_safe_action(verdict)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "revision": REVISION,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "Bounded real-activation basis comparison before any full GLM "
            "traversal (revision 1: rank eligibility, dual down analysis, "
            "promotion-panel floors)"
        ),
        "seed": seed,
        "ranks": list(map(int, ranks)),
        "basis_modes": list(BASIS_MODES),
        "preregistered_thresholds": PREREGISTERED,
        "numerical_equivalence_tolerance": NUMERICAL_EQUIVALENCE_TOLERANCE,
        "resources": {
            "free_disk_bytes_before": free,
            "disk_floor_bytes": DISK_FLOOR_BYTES,
            "disk_floor_enforced": True,
            "estimated_peak_ram_bytes": peak_est,
            "estimated_peak_ram_gib": peak_est / (1 << 30),
        },
        "revision_0_evidence": rev0,
        "inputs": {
            "pilot_source": str(pilot_source),
            "capsule_dir": str(capsule_dir),
            "rehydration_receipt": str(REHYDRATION_RECEIPT),
            "verified_shards": verified,
            "capsule_sha256": capsule_hashes,
            "code_sha256": {
                "glm52_basis_pilot.py": code_hash,
                "glm52_activation_aware_pack.py": pack_hash,
            },
            "note": "Five resident shards re-verified by hash; nothing fetched.",
        },
        "missing_classes": MISSING_CLASSES,
        "tensor_results": results,
        "panel_byte_totals_rank256": panel_byte_totals,
        "verdict": verdict,
        "safety": safety,
        "elapsed_seconds": round(time.time() - t0, 3),
        "remaining_uncertainty": [
            "Five shards / ~19 tensors cannot decide whole-model generation.",
            "Calibrated cosine floors came from Llama-1B; GLM absolute floors may differ.",
            "Shared layer bases transferred across experts were not the primary fit mode; "
            "route-conditioned per-expert bases were used for routed tensors.",
            "o_proj and global embed/lm_head remain unmeasured on real intermediates.",
            "Winning a basis arm does not authorize flipping production defaults.",
            "Uncentered and explicit-mean remain numerically tied; implementation "
            "choice unresolved when median difference is below tolerance.",
            "Activation-matched input-side down clears the bounded promotion "
            "panel floors, but transfer to unmeasured experts and layers remains "
            "unproven.",
            "Byte totals are exact arithmetic payload estimates, not physical "
            "serialized pack-file measurements.",
        ],
        "next_safe_action": next_action,
    }

    if out_json is not None:
        out_json = Path(out_json)
        out_json.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        doc["wrote_json"] = str(out_json)
    if out_md is not None:
        out_md = Path(out_md)
        out_md.write_text(render_markdown(doc))
        doc["wrote_md"] = str(out_md)
    return doc


def _panel_byte_totals(
    results: list[dict[str, Any]],
    *,
    rank: int = 256,
    mode: str = "centered",
) -> dict[str, Any]:
    """Sum exact arithmetic encoded bytes per panel at a calibrated rank.

    Only uncapped points (total_rank == requested_rank) contribute to panel
    totals used for promotion accounting. Scope is stated on each entry.
    """
    out: dict[str, Any] = {
        "rank": rank,
        "mode_for_byte_lookup": mode,
        "accounting_scope": (
            "exact_arithmetic_per_tensor_payload_float16_"
            "header_plus_coefficients_plus_basis; "
            "sum over uncapped promotion-eligible points only"
        ),
        "is_physical_file_measurement": False,
        "panels": {},
    }
    for panel in (
        PANEL_PROMOTION_GRADE,
        PANEL_SHARED_MLP,
        PANEL_ATTENTION_ROUTER,
        PANEL_LOW_TRAFFIC,
    ):
        total = 0
        n = 0
        per_tensor: list[dict[str, Any]] = []
        for row in _iter_measured_rows(results, panel=panel):
            pts = [
                p for p in row["arms"].get(mode, [])
                if is_promotion_eligible_point(p, rank)
            ]
            if not pts:
                # still report diagnostic capped if present
                capped = [
                    p for p in row["arms"].get(mode, [])
                    if int(p.get("requested_rank", -1)) == rank
                ]
                if capped:
                    per_tensor.append({
                        "name": row["name"],
                        "included": False,
                        "reason": "rank_capped",
                        "requested_rank": capped[0]["requested_rank"],
                        "total_rank": capped[0]["total_rank"],
                        "bytes_diagnostic": capped[0]["bytes"]["total"],
                        "bpw_diagnostic": capped[0]["bpw"],
                    })
                continue
            p = pts[0]
            total += int(p["bytes"]["total"])
            n += 1
            per_tensor.append({
                "name": row["name"],
                "included": True,
                "requested_rank": p["requested_rank"],
                "total_rank": p["total_rank"],
                "bytes": p["bytes"]["total"],
                "bpw": p["bpw"],
                "n_weights": row["n_weights"],
                "byte_accounting_scope": p.get(
                    "byte_accounting_scope",
                    p["bytes"].get("accounting_scope"),
                ),
            })
        out["panels"][panel] = {
            "n_tensors_included": n,
            "total_encoded_bytes": total,
            "mean_bpw_over_included": (
                None if n == 0 else
                sum(t["bpw"] for t in per_tensor if t.get("included")) / n
            ),
            "per_tensor": per_tensor,
        }
    # Grand total over promotion-grade only
    pg = out["panels"][PANEL_PROMOTION_GRADE]
    out["promotion_grade_total_encoded_bytes"] = pg["total_encoded_bytes"]
    return out


def _next_safe_action(verdict: dict[str, Any]) -> str:
    floors = verdict.get("floor_checks", [])
    any_clear = any(fc.get("status") == "CLEARS" for fc in floors)
    down_prom = verdict.get("down_promotion_metric_rank256", [])
    down_neg = verdict.get("down_negative_control_rank256", [])
    # Summarize down promotion cosines (centered mode)
    prom_c = [
        d["cosine"] for d in down_prom
        if d.get("mode") == "uncentered" or d.get("mode") == "explicit_mean"
    ]
    neg_c = [
        d["cosine"] for d in down_neg
        if d.get("mode") == "uncentered" or d.get("mode") == "explicit_mean"
    ]
    tied = verdict.get("uncentered_explicit_mean_numerically_tied", False)
    parts = []
    if tied:
        parts.append(
            "Treat uncentered and equal-byte explicit-mean as numerically tied; "
            "do not declare one uniquely superior. Prefer an opt-in pack basis "
            "flag that retains the mean (either implementation), not a production "
            "default flip."
        )
    else:
        parts.append(
            "If a larger route-conditioned panel replicates the lift, prefer an "
            "opt-in pack basis flag (not a production default flip)."
        )
    parts.append(
        "Do not start a 282-shard traversal: full_traversal_authorized is false "
        "unless every preregistered promotion-grade floor clears."
    )
    if not any_clear:
        parts.append(
            "Promotion-grade floors still fail. Inspect activation_matched "
            "input-side down cosines separately from the preserved production "
            "output-side negative control; escalate representation for down "
            "(or SwiGLU-joint codecs) if input-side remains far below floors."
        )
    parts.append("Never promote on beats_null, reconstruction error, or low-traffic diagnostics.")
    return " ".join(parts)


def render_markdown(doc: dict[str, Any]) -> str:
    v = doc["verdict"]
    lines = [
        "# GLM-5.2 bounded real-activation basis pilot (revision 1)",
        "",
        f"- schema: `{doc['schema']}`",
        f"- revision: `{doc.get('revision', 0)}`",
        f"- at: {doc['at']}",
        f"- seed: `{doc['seed']}`",
        f"- ranks: {doc['ranks']}",
        f"- elapsed_s: {doc['elapsed_seconds']}",
        f"- full_traversal_authorized: `{v.get('full_traversal_authorized')}`",
        "",
        "## Distinguishing story",
        "",
        v["distinguishing_story"],
        "",
        f"- story code: `{v.get('distinguishing_story_code')}`",
        f"- mean median lift explicit_mean − centered: "
        f"{v['mean_median_lift_explicit_mean_over_centered']!r}",
        f"- mean median lift uncentered − centered: "
        f"{v['mean_median_lift_uncentered_over_centered']!r}",
        f"- mean median lift explicit_mean − uncentered: "
        f"{v.get('mean_median_lift_explicit_mean_over_uncentered')!r}",
        f"- numerical equivalence tolerance: "
        f"{v.get('numerical_equivalence_tolerance')}",
        f"- uncentered/explicit_mean tied: "
        f"`{v.get('uncentered_explicit_mean_numerically_tied')}`",
        "",
        "## Revision 0 evidence (preserved)",
        "",
    ]
    rev0 = doc.get("revision_0_evidence") or {}
    if rev0:
        lines += [
            f"- label: `{rev0.get('label', 'revision_0')}`",
            f"- sha256: `{rev0.get('sha256')}`",
            f"- at: {rev0.get('at')}",
            f"- prior story: {rev0.get('distinguishing_story', '')[:200]}",
            f"- note: {rev0.get('note', '')}",
            "",
        ]
    else:
        lines += ["- (no revision_0 evidence block found)", ""]

    lines += [
        "## Promotion-grade panel (uncapped ranks only)",
        "",
        "Equal-byte median / min cosine. Points with `rank_capped` or "
        "`total_rank != requested_rank` are excluded. Included/excluded counts shown.",
        "",
        "| rank | centered med/min (n_in/n_ex) | uncentered | explicit_mean |",
        "|---:|---|---|---|",
    ]
    for rank in doc["ranks"]:
        cells = []
        for mode in BASIS_MODES:
            st = v["summaries"][mode]["by_rank"].get(str(rank), {})
            if not st or st.get("n_included", 0) == 0:
                cells.append(f"n/a (0/{st.get('n_excluded', 0)})")
            else:
                cells.append(
                    f"{st['median']:.4f} / {st['min']:.4f} "
                    f"({st['n_included']}/{st['n_excluded']})"
                )
        lines.append(f"| {rank} | " + " | ".join(cells) + " |")

    # Separate panel tables (brief)
    lines += [
        "",
        "## Separate panel aggregates @ rank 256 (uncapped)",
        "",
    ]
    for panel_name in (
        PANEL_PROMOTION_GRADE,
        PANEL_SHARED_MLP,
        PANEL_ATTENTION_ROUTER,
        PANEL_LOW_TRAFFIC,
    ):
        lines.append(f"### `{panel_name}`")
        lines.append("")
        for mode in BASIS_MODES:
            st = (
                v.get("panel_summaries", {})
                .get(mode, {})
                .get(panel_name, {})
                .get("by_rank", {})
                .get("256", {})
            )
            if not st or st.get("n_included", 0) == 0:
                lines.append(
                    f"- {mode}: n_included=0 n_excluded={st.get('n_excluded', 0)}"
                )
            else:
                lines.append(
                    f"- {mode}: med={st['median']:.4f} min={st['min']:.4f} "
                    f"n_included={st['n_included']} n_excluded={st['n_excluded']}"
                )
        lines.append("")

    lines += [
        "## Floor checks (promotion-grade panel only)",
        "",
    ]
    for fc in v["floor_checks"]:
        lines.append(
            f"- {fc.get('floor')} / {fc.get('mode')} @ rank {fc.get('rank')} "
            f"panel=`{fc.get('panel')}`: **{fc.get('status')}**"
            + (
                f" (min={fc.get('min'):.4f}, med={fc.get('median'):.4f}, "
                f"n_in={fc.get('n_included')}, n_ex={fc.get('n_excluded')})"
                if "min" in fc else
                f" (n_in={fc.get('n_included')}, n_ex={fc.get('n_excluded')})"
            )
        )

    lines += [
        "",
        "## Down analyses @ rank 256",
        "",
        "### production_output_side_down_negative_control (NOT promotional)",
        "",
    ]
    for d in v.get("down_negative_control_rank256", []):
        if d.get("mode") != "uncentered":
            continue
        lines.append(
            f"- `{d['name']}` panel={d.get('panel')} "
            f"total_rank={d['total_rank']} capped={d['rank_capped']} "
            f"cosine={d['cosine']:.4f} basis_width={d.get('basis_width')} "
            f"side={d.get('projection_side')}"
        )
    lines += [
        "",
        "### activation_matched_input_side_down (promotion metric)",
        "",
    ]
    for d in v.get("down_promotion_metric_rank256", []):
        if d.get("mode") != "uncentered":
            continue
        lines.append(
            f"- `{d['name']}` panel={d.get('panel')} "
            f"total_rank={d['total_rank']} capped={d['rank_capped']} "
            f"cosine={d['cosine']:.4f} basis_width={d.get('basis_width')} "
            f"side={d.get('projection_side')}"
        )

    # Byte totals
    pbt = doc.get("panel_byte_totals_rank256") or {}
    lines += [
        "",
        "## Panel-total encoded bytes @ rank 256 (exact arithmetic)",
        "",
        f"- accounting scope: {pbt.get('accounting_scope')}",
        f"- is_physical_file_measurement: `{pbt.get('is_physical_file_measurement')}`",
        f"- promotion-grade total bytes: `{pbt.get('promotion_grade_total_encoded_bytes')}`",
        "",
    ]
    for panel_name, pdata in (pbt.get("panels") or {}).items():
        lines.append(
            f"- `{panel_name}`: n={pdata.get('n_tensors_included')} "
            f"total_bytes={pdata.get('total_encoded_bytes')} "
            f"mean_bpw={pdata.get('mean_bpw_over_included')}"
        )

    lines += [
        "",
        "## Inputs (hashes)",
        "",
        f"- pilot code: `{doc['inputs']['code_sha256']['glm52_basis_pilot.py']}`",
        f"- pack module: `{doc['inputs']['code_sha256']['glm52_activation_aware_pack.py']}`",
        "",
        "### Verified shards",
        "",
    ]
    for s in doc["inputs"]["verified_shards"]:
        lines.append(
            f"- `{s['name']}` sha256=`{s['sha256'][:16]}…` verified={s['verified']}"
        )
    lines += [
        "",
        "### Capsules",
        "",
    ]
    for name, digest in sorted(doc["inputs"]["capsule_sha256"].items()):
        lines.append(f"- `{name}` sha256=`{digest[:16]}…`")
    lines += [
        "",
        "## Missing classes",
        "",
    ]
    for m in doc["missing_classes"]:
        lines.append(f"- **{m['class']}**: {m['why']}")
    lines += [
        "",
        "## Per-tensor status (promotion arms @ rank 256)",
        "",
    ]
    for row in doc["tensor_results"]:
        st = row.get("status")
        if st != "MEASURED":
            lines.append(f"- `{row.get('name')}`: {st} — {row.get('why', '')}")
            continue
        bits = []
        for mode in BASIS_MODES:
            pts = [
                p for p in row["arms"][mode]
                if p.get("requested_rank") == 256 and "mean_row_cosine" in p
            ]
            if pts:
                p = pts[0]
                tag = "capped" if p.get("rank_capped") else "ok"
                bits.append(f"{mode}={p['mean_row_cosine']:.4f}[{tag}]")
        neg = ""
        if "down_analyses" in row:
            neg_pts = [
                p for p in row["down_analyses"][DOWN_NEG_CONTROL]["arms"]["uncentered"]
                if p.get("requested_rank") == 256
            ]
            if neg_pts:
                neg = f" neg_ctrl={neg_pts[0]['mean_row_cosine']:.4f}"
        lines.append(
            f"- `{row['name']}` panel={row.get('panel')} routes={row['route_count']} "
            f"fit/hold={row['n_fit']}/{row['n_hold']} " + " ".join(bits) + neg
        )
    lines += [
        "",
        "## Safety",
        "",
    ]
    for k, val in doc["safety"].items():
        lines.append(f"- {k}: `{val}`")
    lines += [
        "",
        "## Remaining uncertainty",
        "",
    ]
    for u in doc["remaining_uncertainty"]:
        lines.append(f"- {u}")
    lines += [
        "",
        "## Next safe action",
        "",
        doc["next_safe_action"],
        "",
        "## Not claimed",
        "",
        "A bounded tensor pilot does **not** prove whole-model capability. "
        "`beats_null`, reconstruction error, the invalid all-row diagnostic, "
        "low-traffic diagnostics, and the production output-side down negative "
        "control are non-promotional. Do not change production defaults merely "
        "because an arm wins. Do not declare explicit-mean uniquely superior "
        "when it is numerically tied with uncentered.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test / unit-level checks (also covered by pytest)
# ---------------------------------------------------------------------------
def selftest() -> int:
    rng = np.random.default_rng(0)
    n, h = 400, 64
    mean = rng.standard_normal(h).astype(np.float32)
    mean /= np.linalg.norm(mean)
    X = (0.2 * rng.standard_normal((n, h)) + 3.0 * mean).astype(np.float32)
    fit, hold = fit_holdout_indices(n, seed=1, salt=2)
    assert fit_holdout_indices(n, seed=1, salt=2)[0].tolist() == fit.tolist()

    r = 8
    b_c = build_pilot_basis(X[fit], "centered", r)
    b_u = build_pilot_basis(X[fit], "uncentered", r)
    b_m = build_pilot_basis(X[fit], "explicit_mean", r)
    assert b_c.total_rank == r and b_u.total_rank == r
    assert b_m.total_rank == r
    assert b_m.residual_rank == r - 1
    assert b_m.mean_direction is not None
    assert abs(float(np.dot(b_m.columns(r)[:, 0], b_m.mean_direction))) > 0.99
    dots = b_m.columns(r)[:, 1:].T @ b_m.mean_direction
    assert float(np.max(np.abs(dots))) < 1e-4
    for side, rows, cols in (("input", 32, h), ("output", h, 16)):
        costs = [encoded_bytes(rows, cols, side, r)["total"] for _ in BASIS_MODES]
        assert len(set(costs)) == 1
    assert b_m.residual_rank == b_m.total_rank - 1
    assert centered_matches_production(X[fit], r)
    topk = np.array([[1, 2], [3, 1], [4, 5], [1, 9]], dtype=np.int32)
    assert route_row_indices(topk, 1).tolist() == [0, 1, 3]
    Wg = rng.standard_normal((16, h)).astype(np.float32)
    Wu = rng.standard_normal((16, h)).astype(np.float32)
    Wd = rng.standard_normal((h, 16)).astype(np.float32)
    inter = swiglu_intermediate(X[hold], Wg, Wu)
    assert inter.shape == (hold.size, 16)
    W_hat, _ = project_and_reconstruct(
        Wd, build_pilot_basis(X[fit], "centered", 4), 4, "output"
    )
    sc = score_linear(Wd, W_hat, inter, side="output")
    assert "mean_row_cosine" in sc

    # Rank eligibility: capped point cannot enter floor
    assert not is_promotion_eligible_point(
        {"requested_rank": 512, "total_rank": 164, "rank_capped": True,
         "mean_row_cosine": 0.19},
        512,
    )
    assert is_promotion_eligible_point(
        {"requested_rank": 256, "total_rank": 256, "rank_capped": False,
         "mean_row_cosine": 0.9},
        256,
    )

    # Dual down synthetic: equal bytes,  input-side basis width matches intermediate
    n2, h2, inter_w = 120, 48, 16
    X2 = rng.standard_normal((n2, h2)).astype(np.float32)
    Wg2 = rng.standard_normal((inter_w, h2)).astype(np.float32)
    Wu2 = rng.standard_normal((inter_w, h2)).astype(np.float32)
    Wd2 = rng.standard_normal((h2, inter_w)).astype(np.float32)
    fit2, hold2 = fit_holdout_indices(n2, seed=3, salt=1)
    spec = TensorSpec(
        "toy.down_proj.weight", "high_traffic_routed_down", 0, 0, True,
        "swiglu_intermediate",
    )
    row = evaluate_tensor(
        spec, Wd2, X_all=X2, fit_idx=fit2, hold_idx=hold2,
        ranks=(4, 8), W_gate=Wg2, W_up=Wu2, route_count=n2,
    )
    assert row["status"] == "MEASURED"
    assert DOWN_NEG_CONTROL in row["down_analyses"]
    assert DOWN_PROMOTION in row["down_analyses"]
    assert row["promotion_metric"] == DOWN_PROMOTION
    prom = row["down_analyses"][DOWN_PROMOTION]
    neg = row["down_analyses"][DOWN_NEG_CONTROL]
    assert prom["basis_width"] == inter_w
    assert neg["basis_width"] == h2
    assert prom["projection_side"] == "input"
    assert neg["projection_side"] == "output"
    assert row["equal_bytes_across_down_analyses"]
    assert row["gaussian_proxy_used"] is False

    print("selftest: ok")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    run_p = sub.add_parser("run")
    run_p.add_argument("--pilot-source", type=Path, default=PILOT_SOURCE)
    run_p.add_argument("--capsule-dir", type=Path, default=CAPSULE_DIR)
    run_p.add_argument("--ranks", type=str, default=",".join(map(str, DEFAULT_RANKS)))
    run_p.add_argument("--seed", type=int, default=SEED)
    run_p.add_argument("--out", type=Path, default=REPO / "GLM52_BASIS_PILOT_RECEIPT.json")
    run_p.add_argument("--out-md", type=Path, default=REPO / "GLM52_BASIS_PILOT_RECEIPT.md")
    run_p.add_argument("--skip-hash-verify", action="store_true",
                       help="dangerous; only for offline dry structure checks")
    args = p.parse_args(argv)
    if args.cmd == "selftest":
        return selftest()
    ranks = tuple(int(x) for x in args.ranks.split(",") if x.strip())
    doc = run_pilot(
        pilot_source=args.pilot_source,
        capsule_dir=args.capsule_dir,
        ranks=ranks,
        seed=args.seed,
        out_json=args.out,
        out_md=args.out_md,
        skip_hash_verify=args.skip_hash_verify,
    )
    print(json.dumps({
        "wrote_json": str(args.out),
        "wrote_md": str(args.out_md),
        "revision": doc.get("revision"),
        "story": doc["verdict"]["distinguishing_story"],
        "story_code": doc["verdict"].get("distinguishing_story_code"),
        "full_traversal_authorized": doc["verdict"].get("full_traversal_authorized"),
        "elapsed_seconds": doc["elapsed_seconds"],
        "n_measured": sum(1 for r in doc["tensor_results"] if r.get("status") == "MEASURED"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
