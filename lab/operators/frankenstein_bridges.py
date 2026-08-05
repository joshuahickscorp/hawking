#!/usr/bin/env python3.12
"""Reversible nonlinear bridges + small ablatable adapters for functional transfer.

Architecture (owner steer):
  norm → projection → gated MLP → low-rank residual

Adapters (small, hash-bound, independently ablatable, Gravity-accounted):
  GLM_METHOD_ADAPTER, GLM_DECOMPOSITION_ADAPTER, GLM_FORMALIZATION_ADAPTER,
  GLM_REPAIR_ADAPTER, GLM_VALUE_HEAD, optional math expert bank,
  method-conditioned route-bias residual.

Training is REQUIRES_TRAINING_LOOP — this module defines specs, apply/revert,
byte accounting, and a trainer *interface* that fails closed on fit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators.frankenstein_fusion_op import DEEPSEEK_V4_FLASH, GLM_5_2
from lab.operators.frankenstein_gates import (
    LINEAR_SUBSPACE_INITIALIZATION,
    REQUIRES_TRAINING_LOOP,
    fail_closed,
    gate_record,
)
from lab.receipts import seal


BRIDGE_MODULE_SCHEMA = "hawking.frankenstein.reversible_nonlinear_bridge.v1"
ADAPTER_BANK_SCHEMA = "hawking.frankenstein.adapter_bank.v1"
TRAINER_SCHEMA = "hawking.frankenstein.bridge_trainer_interface.v1"

ADAPTER_NAMES: tuple[str, ...] = (
    "GLM_METHOD_ADAPTER",
    "GLM_DECOMPOSITION_ADAPTER",
    "GLM_FORMALIZATION_ADAPTER",
    "GLM_REPAIR_ADAPTER",
    "GLM_VALUE_HEAD",
    "GLM_MATH_EXPERT_BANK",  # optional
    "METHOD_CONDITIONED_ROUTE_BIAS",
)

# Loss names optimized at train time (definitions only here).
LOSS_DEFINITIONS: dict[str, str] = {
    "functional_output": "held-out functional output match (span/action level)",
    "token_action_kl": "KL on aligned tokens/actions (never raw mismatched token ids)",
    "method_classification": "method-family classification CE",
    "route_behavior": "route histogram / load-balance match to distilled policy",
    "verifier_outcome": "verifier pass/fail CE on repair trajectories",
    # Explicitly NOT the sole objective:
    "latent_cosine_alone": "FORBIDDEN as sole objective",
}


class BridgeError(RuntimeError):
    """Bridge / adapter module error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _param_bytes(arrays: Mapping[str, np.ndarray]) -> int:
    return int(sum(np.asarray(v).nbytes for v in arrays.values()))


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + eps)
    return a / rms


def silu(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a / (1.0 + np.exp(-np.clip(a, -60, 60)))


@dataclass
class ReversibleBridge:
    """norm → proj → gated MLP → low-rank residual (reversible residual form).

    Forward (student space):
      h1 = RMSNorm(x)
      h2 = h1 @ W_in          # project (optionally from donor init width)
      gate = silu(h2 @ W_g)
      up = h2 @ W_u
      h3 = (gate * up) @ W_d
      # low-rank residual in student dim
      y = x + (h3 @ U) @ V + b

    Reverse: subtract the residual path recomputed from y≈x for small residuals;
    for exact reverse we store the residual delta at apply time when needed.
    Exact invertibility for the affine residual part: y = x + r(x); for small
    nonlinear r we use fixed-point reverse (iterate) and report residual error.
    """

    name: str
    d_model: int
    d_hidden: int
    rank: int
    eps: float = 1e-6
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        *,
        name: str = "GLM_NONLINEAR_BRIDGE",
        d_model: int = 4096,
        d_hidden: int = 256,
        rank: int = 16,
        seed: int = 0,
        linear_init: Mapping[str, np.ndarray] | None = None,
        scale: float = 0.02,
    ) -> "ReversibleBridge":
        rng = np.random.default_rng(seed)
        w_in = rng.standard_normal((d_model, d_hidden)) * scale
        w_g = rng.standard_normal((d_hidden, d_hidden)) * scale
        w_u = rng.standard_normal((d_hidden, d_hidden)) * scale
        w_d = rng.standard_normal((d_hidden, d_model)) * scale
        u = rng.standard_normal((d_model, rank)) * scale
        v = rng.standard_normal((rank, d_model)) * scale
        b = np.zeros(d_model, dtype=np.float64)
        # Optional: seed projection path from LINEAR_SUBSPACE_INITIALIZATION factors.
        if linear_init is not None:
            if "residual_A" in linear_init:
                a = np.asarray(linear_init["residual_A"], dtype=np.float64)
                # Take top-rank SVD of A as U,V init.
                uu, s, vt = np.linalg.svd(a, full_matrices=False)
                k = min(rank, uu.shape[1])
                u[:, :k] = uu[:, :k] * np.sqrt(s[:k])
                v[:k, :] = np.sqrt(s[:k])[:, None] * vt[:k, :]
            if "residual_bias" in linear_init:
                b = np.asarray(linear_init["residual_bias"], dtype=np.float64).reshape(-1)
                if b.shape[0] != d_model:
                    raise BridgeError("linear_init residual_bias width mismatch")
        arrays = {
            "W_in": w_in,
            "W_g": w_g,
            "W_u": w_u,
            "W_d": w_d,
            "U": u,
            "V": v,
            "b": b,
        }
        return cls(
            name=name,
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            arrays=arrays,
            meta={
                "initialized_from": (
                    LINEAR_SUBSPACE_INITIALIZATION if linear_init is not None else "random"
                ),
                "trained": False,
                "architecture": "norm_proj_gated_mlp_lowrank_residual",
            },
        )

    def content_hash(self) -> str:
        parts = []
        for k in sorted(self.arrays):
            arr = np.asarray(self.arrays[k], dtype=np.float64)
            parts.append(k.encode())
            parts.append(arr.tobytes())
        return _sha256(b"".join(parts))

    def parameter_count(self) -> int:
        return int(sum(np.asarray(v).size for v in self.arrays.values()))

    def parameter_bytes(self) -> int:
        return _param_bytes(self.arrays)

    def residual(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        if a.shape[-1] != self.d_model:
            raise BridgeError(f"expected last dim {self.d_model}, got {a.shape}")
        h1 = rms_norm(a, eps=self.eps)
        h2 = h1 @ self.arrays["W_in"]
        gate = silu(h2 @ self.arrays["W_g"])
        up = h2 @ self.arrays["W_u"]
        h3 = (gate * up) @ self.arrays["W_d"]
        low = (h3 @ self.arrays["U"]) @ self.arrays["V"]
        return low + self.arrays["b"]

    def apply(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        return a + self.residual(a)

    def revert(
        self,
        y: np.ndarray,
        *,
        n_iters: int = 8,
        atol: float = 1e-6,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Fixed-point reverse for y = x + r(x): x_{t+1} = y - r(x_t)."""

        y_arr = np.asarray(y, dtype=np.float64)
        x = y_arr.copy()
        last_err = None
        for _ in range(n_iters):
            x_next = y_arr - self.residual(x)
            last_err = float(np.max(np.abs(x_next - x)))
            x = x_next
            if last_err < atol:
                break
        # Measure reconstruction y vs apply(x)
        recon = float(np.max(np.abs(self.apply(x) - y_arr)))
        return x, {
            "iterations": n_iters,
            "last_step_delta": last_err,
            "recon_error": recon,
            "exact": recon < atol,
        }

    def gravity_accounting(self) -> dict[str, Any]:
        nbytes = self.parameter_bytes()
        # Rough FLOPs per token for residual path (matmul dominate).
        # W_in: d*h, W_g: h*h, W_u: h*h, W_d: h*d, U: d*r, V: r*d
        d, h, r = self.d_model, self.d_hidden, self.rank
        flops = 2 * (d * h + h * h + h * h + h * d + d * r + r * d)
        return {
            "parameter_count": self.parameter_count(),
            "parameter_bytes": nbytes,
            "parameter_bytes_mib": nbytes / (1024 * 1024),
            "approx_flops_per_token": flops,
            "tps_impact_note": (
                "TPS impact requires live forward measurement; formula is FLOP proxy only"
            ),
            "p99_note": "p99 latency unmeasured until runtime harness binds this module",
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "reversible": True,
        }

    def to_spec(self) -> dict[str, Any]:
        return seal(
            {
                "schema": BRIDGE_MODULE_SCHEMA,
                "name": self.name,
                "recorded_at": _utc_now(),
                "d_model": self.d_model,
                "d_hidden": self.d_hidden,
                "rank": self.rank,
                "architecture": "norm -> projection -> gated_mlp -> low_rank_residual",
                "parameter_count": self.parameter_count(),
                "content_hash": self.content_hash(),
                "meta": dict(self.meta),
                "gravity": self.gravity_accounting(),
                "trained": bool(self.meta.get("trained")),
                "loss_targets": dict(LOSS_DEFINITIONS),
                "linear_init_role": LINEAR_SUBSPACE_INITIALIZATION,
                "claim_boundary": {
                    "training_performed": bool(self.meta.get("trained")),
                    "capability_claim": False,
                    "proto_complete": False,
                },
            }
        )


@dataclass
class AdapterModule:
    """Small residual adapter: y = x + scale * (x @ A_lo @ A_hi) + b  (low-rank)."""

    name: str
    d_model: int
    rank: int
    scale: float = 1.0
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        name: str,
        *,
        d_model: int = 4096,
        rank: int = 8,
        seed: int = 0,
        scale: float = 0.05,
    ) -> "AdapterModule":
        if name not in ADAPTER_NAMES:
            raise BridgeError(f"unknown adapter {name!r}; permitted={ADAPTER_NAMES}")
        rng = np.random.default_rng(seed + hash(name) % 10007)
        # Special case: route bias is over experts, not hidden.
        if name == "METHOD_CONDITIONED_ROUTE_BIAS":
            n_experts = int(DEEPSEEK_V4_FLASH["n_routed_experts"])
            # bias table: method_id (small) × experts — store as single bias for init
            arrays = {
                "route_bias": np.zeros(n_experts, dtype=np.float64),
                "method_embed": rng.standard_normal((16, rank)) * 0.02,
                "method_to_bias": rng.standard_normal((rank, n_experts)) * 0.02,
            }
            return cls(
                name=name,
                d_model=n_experts,
                rank=rank,
                scale=scale,
                arrays=arrays,
                meta={"kind": "route_bias_residual", "trained": False, "copy_glm_router": False},
            )
        if name == "GLM_VALUE_HEAD":
            arrays = {
                "W": rng.standard_normal((d_model, 1)) * 0.02,
                "b": np.zeros(1, dtype=np.float64),
            }
            return cls(
                name=name,
                d_model=d_model,
                rank=1,
                scale=scale,
                arrays=arrays,
                meta={"kind": "value_head", "trained": False},
            )
        a_lo = rng.standard_normal((d_model, rank)) * 0.02
        a_hi = rng.standard_normal((rank, d_model)) * 0.02
        b = np.zeros(d_model, dtype=np.float64)
        return cls(
            name=name,
            d_model=d_model,
            rank=rank,
            scale=scale,
            arrays={"A_lo": a_lo, "A_hi": a_hi, "b": b},
            meta={"kind": "low_rank_residual", "trained": False},
        )

    def content_hash(self) -> str:
        parts = []
        for k in sorted(self.arrays):
            parts.append(k.encode())
            parts.append(np.asarray(self.arrays[k], dtype=np.float64).tobytes())
        return _sha256(b"".join(parts))

    def parameter_count(self) -> int:
        return int(sum(np.asarray(v).size for v in self.arrays.values()))

    def parameter_bytes(self) -> int:
        return _param_bytes(self.arrays)

    def apply(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        if self.name == "METHOD_CONDITIONED_ROUTE_BIAS":
            # x is router logits [..., n_experts]
            return a + self.scale * self.arrays["route_bias"]
        if self.name == "GLM_VALUE_HEAD":
            # returns values, not residual on x
            return a @ self.arrays["W"] + self.arrays["b"]
        if a.shape[-1] != self.d_model:
            raise BridgeError(f"{self.name}: expected dim {self.d_model}, got {a.shape}")
        delta = (a @ self.arrays["A_lo"]) @ self.arrays["A_hi"] + self.arrays["b"]
        return a + self.scale * delta

    def revert(self, y: np.ndarray) -> np.ndarray:
        """Exact reverse for affine low-rank residual when scale*A is small.

        y = x + s (x A_lo A_hi + b) = x (I + s A_lo A_hi) + s b
        Use Woodbury-ish fixed point for generality.
        """

        if self.name == "GLM_VALUE_HEAD":
            raise BridgeError("GLM_VALUE_HEAD is not a residual on hidden; cannot revert")
        y_arr = np.asarray(y, dtype=np.float64)
        if self.name == "METHOD_CONDITIONED_ROUTE_BIAS":
            return y_arr - self.scale * self.arrays["route_bias"]
        # Fixed point: x = y - s (x A_lo A_hi + b)
        x = y_arr.copy()
        for _ in range(12):
            delta = (x @ self.arrays["A_lo"]) @ self.arrays["A_hi"] + self.arrays["b"]
            x = y_arr - self.scale * delta
        return x

    def gravity_accounting(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count(),
            "parameter_bytes": self.parameter_bytes(),
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "reversible": self.name != "GLM_VALUE_HEAD",
            "copy_glm_router_weights": False,
        }

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "d_model": self.d_model,
            "rank": self.rank,
            "scale": self.scale,
            "meta": dict(self.meta),
            "gravity": self.gravity_accounting(),
            "content_hash": self.content_hash(),
            "parameter_count": self.parameter_count(),
        }


def build_adapter_bank(
    *,
    d_model: int = 4096,
    rank: int = 8,
    seed: int = 0,
    include_optional_expert_bank: bool = True,
) -> dict[str, Any]:
    """Initialize the full independently-ablatable adapter bank (untrained)."""

    names = list(ADAPTER_NAMES)
    if not include_optional_expert_bank:
        names = [n for n in names if n != "GLM_MATH_EXPERT_BANK"]
    adapters = [
        AdapterModule.initialize(n, d_model=d_model, rank=rank, seed=seed)
        for n in names
    ]
    total_bytes = sum(a.parameter_bytes() for a in adapters)
    document = {
        "schema": ADAPTER_BANK_SCHEMA,
        "recorded_at": _utc_now(),
        "adapters": [a.to_spec() for a in adapters],
        "total_parameter_bytes": total_bytes,
        "total_parameter_count": sum(a.parameter_count() for a in adapters),
        "trained": False,
        "native_dsv4f_routing_preserved": True,
        "glm_router_weights_copied": False,
        "independently_ablatable": True,
        "hash_bound": True,
        "gravity_accounted": True,
        "student": {
            "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
            "n_routed_experts": DEEPSEEK_V4_FLASH["n_routed_experts"],
        },
        "donor": {
            "hidden_size": GLM_5_2["hidden_size"],
            "repository": GLM_5_2["repository"],
        },
        "claim_boundary": {
            "training_performed": False,
            "capability_claim": False,
            "proto_complete": False,
        },
    }
    sealed = seal(document)
    # Attach live objects for in-process apply tests (not serialized).
    sealed["_modules"] = {a.name: a for a in adapters}
    return sealed


def apply_adapter_bank(
    hidden: np.ndarray,
    bank: Mapping[str, Any],
    *,
    enabled: Sequence[str] | None = None,
    skip: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply enabled residual adapters (not value head / route bias) in order."""

    modules: Mapping[str, AdapterModule] = bank.get("_modules") or {}
    if not modules:
        raise BridgeError("adapter bank missing live _modules; rebuild via build_adapter_bank")
    skip_set = set(skip or ())
    enable_set = set(enabled) if enabled is not None else set(modules)
    y = np.asarray(hidden, dtype=np.float64)
    applied: list[str] = []
    for name in ADAPTER_NAMES:
        if name not in enable_set or name in skip_set:
            continue
        if name not in modules:
            continue
        if name in {"GLM_VALUE_HEAD", "METHOD_CONDITIONED_ROUTE_BIAS", "GLM_MATH_EXPERT_BANK"}:
            # Applied at different hooks; skip in generic residual chain.
            continue
        y = modules[name].apply(y)
        applied.append(name)
    return y, {"applied": applied, "enabled": sorted(enable_set)}


def revert_adapter_bank(
    hidden_out: np.ndarray,
    bank: Mapping[str, Any],
    *,
    applied: Sequence[str],
) -> np.ndarray:
    modules: Mapping[str, AdapterModule] = bank.get("_modules") or {}
    x = np.asarray(hidden_out, dtype=np.float64)
    for name in reversed(list(applied)):
        x = modules[name].revert(x)
    return x


# ---------------------------------------------------------------------------
# Trainer interface (fail closed)
# ---------------------------------------------------------------------------


def trainer_interface_spec() -> dict[str, Any]:
    return seal(
        {
            "schema": TRAINER_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "INTERFACE_ONLY",
            "gate": gate_record(REQUIRES_TRAINING_LOOP, open_=False),
            "losses": dict(LOSS_DEFINITIONS),
            "forbidden_sole_objective": ["latent_cosine_alone"],
            "optimizer": None,
            "backward": False,
            "note": (
                "No DSV4F training loop exists (forward-only). "
                "fit_bridge / fit_adapters fail closed until REQUIRES_TRAINING_LOOP opens."
            ),
        }
    )


def fit_bridge(
    bridge: ReversibleBridge,
    *,
    train_batches: Any = None,
    optimizer: Any = None,
) -> dict[str, Any]:
    """Train a nonlinear bridge — REQUIRES_TRAINING_LOOP."""

    if optimizer is None or train_batches is None:
        closed = fail_closed(
            REQUIRES_TRAINING_LOOP,
            stage="5_nonlinear_bridges",
            operation="fit_bridge",
        )
        closed["bridge_name"] = bridge.name
        closed["bridge_hash"] = bridge.content_hash()
        closed["trained"] = False
        return closed
    raise BridgeError(
        "training loop callable path not implemented in this scaffold; "
        "provide a real loop outside fail-closed default"
    )


def fit_adapters(
    bank: Mapping[str, Any],
    *,
    train_batches: Any = None,
    optimizer: Any = None,
) -> dict[str, Any]:
    if optimizer is None or train_batches is None:
        closed = fail_closed(
            REQUIRES_TRAINING_LOOP,
            stage="6_distill_adapters",
            operation="fit_adapters",
        )
        closed["trained"] = False
        closed["bank_seal"] = bank.get("seal_sha256")
        return closed
    raise BridgeError("training loop not implemented in this scaffold")
