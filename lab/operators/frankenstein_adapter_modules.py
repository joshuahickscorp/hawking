#!/usr/bin/env python3.12
"""PyTorch modules for reversible bridges + small ablatable adapters.

Matches the functional-transfer scaffold architecture (frankenstein_bridges specs):

  Bridge:  RMSNorm → proj → gated MLP → low-rank residual
           y = x + r(x)

  Adapters (independently ablatable, hash-bound, byte-accounted):
    GLM_METHOD_ADAPTER, GLM_DECOMPOSITION_ADAPTER, GLM_FORMALIZATION_ADAPTER,
    GLM_REPAIR_ADAPTER, GLM_VALUE_HEAD, METHOD_CONDITIONED_ROUTE_BIAS
    (+ optional GLM_MATH_EXPERT_BANK stub)

These train with standard autograd on *paired activation data* — they do NOT
backprop through DSV4F Metal kernels.  Real paired captures come later from
the capture pipeline; this module only defines the trainable shapes.

New-file lane: does not edit frankenstein_bridges / frankenstein_transfer.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.operators.frankenstein_fusion_op import DEEPSEEK_V4_FLASH


ADAPTER_NAMES: tuple[str, ...] = (
    "GLM_METHOD_ADAPTER",
    "GLM_DECOMPOSITION_ADAPTER",
    "GLM_FORMALIZATION_ADAPTER",
    "GLM_REPAIR_ADAPTER",
    "GLM_VALUE_HEAD",
    "GLM_MATH_EXPERT_BANK",
    "METHOD_CONDITIONED_ROUTE_BIAS",
)

RESIDUAL_ADAPTER_NAMES: tuple[str, ...] = (
    "GLM_METHOD_ADAPTER",
    "GLM_DECOMPOSITION_ADAPTER",
    "GLM_FORMALIZATION_ADAPTER",
    "GLM_REPAIR_ADAPTER",
)

METHOD_FAMILIES: tuple[str, ...] = (
    "method_selection",
    "decomposition",
    "formalization",
    "repair",
    "other",
)

LOSS_DEFINITIONS: dict[str, str] = {
    "functional_output": "held-out functional output match (span/action level)",
    "token_action_kl": "KL on aligned tokens/actions (never raw mismatched token ids)",
    "method_classification": "method-family classification CE",
    "route_behavior": "route histogram / load-balance match to distilled policy",
    "verifier_outcome": "verifier pass/fail CE on repair trajectories",
    "latent_cosine_alone": "FORBIDDEN as sole objective",
}


class ModuleError(RuntimeError):
    """Adapter / bridge module error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tensor_content_hash(named: Mapping[str, torch.Tensor]) -> str:
    parts: list[bytes] = []
    for key in sorted(named):
        arr = named[key].detach().cpu().to(dtype=torch.float64).contiguous().numpy()
        parts.append(key.encode("utf-8"))
        parts.append(arr.tobytes())
    return _sha256(b"".join(parts))


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def param_bytes(module: nn.Module, *, bytes_per_param: int = 4) -> int:
    """Account parameter storage (default fp32 training weights)."""

    return int(param_count(module) * bytes_per_param)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class ReversibleBridge(nn.Module):
    """norm → proj → gated MLP → low-rank residual (reversible residual form).

    Forward (student space):
      h1 = RMSNorm(x)
      h2 = h1 @ W_in
      gate = silu(h2 @ W_g)
      up = h2 @ W_u
      h3 = (gate * up) @ W_d
      y = x + (h3 @ U) @ V + b
    """

    def __init__(
        self,
        *,
        name: str = "GLM_NONLINEAR_BRIDGE",
        d_model: int = 64,
        d_hidden: int = 32,
        rank: int = 4,
        eps: float = 1e-6,
        scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.name = name
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.rank = int(rank)
        self.eps = float(eps)
        self.norm = RMSNorm(d_model, eps=eps)
        self.w_in = nn.Linear(d_model, d_hidden, bias=False)
        self.w_g = nn.Linear(d_hidden, d_hidden, bias=False)
        self.w_u = nn.Linear(d_hidden, d_hidden, bias=False)
        self.w_d = nn.Linear(d_hidden, d_model, bias=False)
        self.u = nn.Linear(d_model, rank, bias=False)
        self.v = nn.Linear(rank, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        self._init_small(scale)
        self.meta: dict[str, Any] = {
            "architecture": "norm_proj_gated_mlp_lowrank_residual",
            "trained": False,
            "initialized_from": "random",
        }

    def _init_small(self, scale: float) -> None:
        for mod in (self.w_in, self.w_g, self.w_u, self.w_d, self.u, self.v):
            nn.init.normal_(mod.weight, mean=0.0, std=float(scale))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ModuleError(f"expected last dim {self.d_model}, got {tuple(x.shape)}")
        h1 = self.norm(x)
        h2 = self.w_in(h1)
        gate = F.silu(self.w_g(h2))
        up = self.w_u(h2)
        h3 = self.w_d(gate * up)
        low = self.v(self.u(h3))
        return low + self.b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)

    def apply_with_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply and return (y, r) so reverse is exact: x = y - r."""

        r = self.residual(x)
        return x + r, r

    @torch.no_grad()
    def revert_exact(self, y: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Exact reverse when residual was stored at apply time."""

        return y - residual

    @torch.no_grad()
    def revert(
        self,
        y: torch.Tensor,
        *,
        n_iters: int = 12,
        atol: float = 1e-5,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reverse y = x + r(x).

        Preferred: pass the residual stored at apply time (exact).
        Fallback: fixed-point x_{t+1} = y - r(x_t) for small/contractive r.
        """

        y_arr = y.detach()
        if residual is not None:
            x = self.revert_exact(y_arr, residual)
            recon = float(torch.max(torch.abs(self.forward(x) - y_arr)).item())
            # When residual is the one used at apply, recon may be nonzero after
            # weight changes; exact arithmetic reverse of the stored step is still exact.
            step_err = float(torch.max(torch.abs((x + residual) - y_arr)).item())
            return x, {
                "mode": "stored_residual",
                "iterations": 0,
                "last_step_delta": 0.0,
                "recon_error": recon,
                "stored_step_error": step_err,
                "exact": step_err < atol,
            }

        x = y_arr.clone()
        last_err = None
        for _ in range(int(n_iters)):
            x_next = y_arr - self.residual(x)
            last_err = float(torch.max(torch.abs(x_next - x)).item())
            x = x_next
            if last_err < atol:
                break
        recon = float(torch.max(torch.abs(self.forward(x) - y_arr)).item())
        return x, {
            "mode": "fixed_point",
            "iterations": int(n_iters),
            "last_step_delta": last_err,
            "recon_error": recon,
            "exact": recon < atol,
        }

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        d, h, r = self.d_model, self.d_hidden, self.rank
        # Matmul FLOP proxy per token (2 * mul-adds).
        flops = 2 * (d * h + h * h + h * h + h * d + d * r + r * d)
        nbytes = param_bytes(self, bytes_per_param=bytes_per_param)
        return {
            "name": self.name,
            "parameter_count": param_count(self),
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
            "bytes_per_param": int(bytes_per_param),
        }

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "rank": self.rank,
            "architecture": "norm -> projection -> gated_mlp -> low_rank_residual",
            "parameter_count": param_count(self),
            "content_hash": self.content_hash(),
            "meta": dict(self.meta),
            "gravity": self.gravity_accounting(),
            "trained": bool(self.meta.get("trained")),
            "loss_targets": dict(LOSS_DEFINITIONS),
        }


class LowRankAdapter(nn.Module):
    """y = x + scale * (x @ A_lo @ A_hi + b)."""

    def __init__(
        self,
        name: str,
        *,
        d_model: int = 64,
        rank: int = 4,
        scale: float = 0.05,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if name not in RESIDUAL_ADAPTER_NAMES and name != "GLM_MATH_EXPERT_BANK":
            raise ModuleError(f"not a low-rank residual adapter: {name!r}")
        self.name = name
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.scale = float(scale)
        self.a_lo = nn.Linear(d_model, rank, bias=False)
        self.a_hi = nn.Linear(rank, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.a_lo.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.a_hi.weight, mean=0.0, std=init_std)
        self.meta: dict[str, Any] = {"kind": "low_rank_residual", "trained": False}

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.a_hi(self.a_lo(x)) + self.b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ModuleError(f"{self.name}: expected dim {self.d_model}, got {tuple(x.shape)}")
        return x + self.scale * self.delta(x)

    @torch.no_grad()
    def revert(self, y: torch.Tensor, *, n_iters: int = 12) -> torch.Tensor:
        """Exact reverse via Woodbury for y = x + s (x L H + b).

        nn.Linear: a_lo(x)=x@W_lo.T, a_hi(z)=z@W_hi.T so L=W_lo.T, H=W_hi.T and
        y = x@(I + s L H) + s b.  Invert with
        (I + s L H)^{-1} = I - s L (I_r + s H L)^{-1} H.
        Falls back to fixed-point iteration if the r×r solve fails.
        """

        y_arr = y.detach()
        s = float(self.scale)
        w_lo = self.a_lo.weight  # [r, d]
        w_hi = self.a_hi.weight  # [d, r]
        try:
            eye_r = torch.eye(self.rank, device=y_arr.device, dtype=y_arr.dtype)
            # H L = W_hi.T @ W_lo.T  → [r, r]
            hl = w_hi.T @ w_lo.T
            mid = eye_r + s * hl
            # Z = (I_r + s H L)^{-1} H = mid^{-1} W_hi.T
            z = torch.linalg.solve(mid, w_hi.T)  # [r, d]
            u = y_arr - s * self.b
            # u @ L @ Z = a_lo(u) @ z
            corr = torch.matmul(self.a_lo(u), z)
            return u - s * corr
        except RuntimeError:
            x = y_arr.clone()
            for _ in range(int(n_iters)):
                x = y_arr - s * self.delta(x)
            return x

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "reversible": True,
            "copy_glm_router_weights": False,
        }


class ValueHead(nn.Module):
    """Scalar value head: v = x @ W + b (not residual on hidden)."""

    def __init__(self, *, d_model: int = 64, init_std: float = 0.02) -> None:
        super().__init__()
        self.name = "GLM_VALUE_HEAD"
        self.d_model = int(d_model)
        self.linear = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.linear.bias)
        self.meta: dict[str, Any] = {"kind": "value_head", "trained": False}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "reversible": False,
            "copy_glm_router_weights": False,
        }


class RouteBiasResidual(nn.Module):
    """Method-conditioned additive route bias (never copies GLM router weights).

    logits' = logits + scale * (base_bias + method_embed[m] @ method_to_bias)
    """

    def __init__(
        self,
        *,
        n_experts: int | None = None,
        n_methods: int = 16,
        rank: int = 4,
        scale: float = 0.05,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if n_experts is None:
            n_experts = int(DEEPSEEK_V4_FLASH["n_routed_experts"])
        self.name = "METHOD_CONDITIONED_ROUTE_BIAS"
        self.n_experts = int(n_experts)
        self.n_methods = int(n_methods)
        self.rank = int(rank)
        self.scale = float(scale)
        self.route_bias = nn.Parameter(torch.zeros(self.n_experts))
        self.method_embed = nn.Embedding(self.n_methods, self.rank)
        self.method_to_bias = nn.Linear(self.rank, self.n_experts, bias=False)
        nn.init.normal_(self.method_embed.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.method_to_bias.weight, mean=0.0, std=init_std)
        self.meta: dict[str, Any] = {
            "kind": "route_bias_residual",
            "trained": False,
            "copy_glm_router": False,
        }

    def bias_for_method(self, method_ids: torch.Tensor) -> torch.Tensor:
        """Return method-conditioned bias matching route logits layout.

        method_ids: [B] or [B, S]
        returns:     [B, E] or [B, S, E]
        """

        emb = self.method_embed(method_ids.long())
        return self.route_bias + self.method_to_bias(emb)

    def _broadcast_bias(
        self,
        bias: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Align bias to route_logits for trailing expert dim."""

        if bias.shape[-1] != route_logits.shape[-1]:
            raise ModuleError(
                f"route bias experts {bias.shape[-1]} != logits {route_logits.shape[-1]}"
            )
        # [E] → broadcast over leading dims
        if bias.dim() == 1:
            return bias
        # [B, E] + [B, S, E] → [B, 1, E]
        if bias.dim() + 1 == route_logits.dim():
            # insert sequence dim just before experts
            return bias.unsqueeze(-2)
        if bias.shape[:-1] == route_logits.shape[:-1]:
            return bias
        raise ModuleError(
            f"cannot broadcast route bias {tuple(bias.shape)} to logits {tuple(route_logits.shape)}"
        )

    def forward(
        self,
        route_logits: torch.Tensor,
        method_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if method_ids is None:
            bias = self._broadcast_bias(self.route_bias, route_logits)
            return route_logits + self.scale * bias
        bias = self._broadcast_bias(self.bias_for_method(method_ids), route_logits)
        return route_logits + self.scale * bias

    @torch.no_grad()
    def revert(
        self,
        route_logits: torch.Tensor,
        method_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if method_ids is None:
            bias = self._broadcast_bias(self.route_bias, route_logits)
            return route_logits - self.scale * bias
        bias = self._broadcast_bias(self.bias_for_method(method_ids), route_logits)
        return route_logits - self.scale * bias

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "reversible": True,
            "copy_glm_router_weights": False,
        }


class MethodClassifier(nn.Module):
    """Lightweight method-family head over adapted hidden (for method CE loss)."""

    def __init__(self, *, d_model: int, n_classes: int = len(METHOD_FAMILIES)) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean-pool if sequence dim present: [B,S,H] → [B,H]
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x)


class AdapterBank(nn.Module):
    """Independently ablatable bank of residual adapters + value head + route bias."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        rank: int = 4,
        n_experts: int | None = None,
        include_optional_expert_bank: bool = False,
        scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        residual: dict[str, LowRankAdapter] = {}
        for name in RESIDUAL_ADAPTER_NAMES:
            residual[name] = LowRankAdapter(name, d_model=d_model, rank=rank, scale=scale)
        if include_optional_expert_bank:
            residual["GLM_MATH_EXPERT_BANK"] = LowRankAdapter(
                "GLM_MATH_EXPERT_BANK", d_model=d_model, rank=rank, scale=scale
            )
        self.residual = nn.ModuleDict(residual)
        self.value_head = ValueHead(d_model=d_model)
        self.route_bias = RouteBiasResidual(
            n_experts=n_experts,
            n_methods=max(16, len(METHOD_FAMILIES)),
            rank=rank,
            scale=scale,
        )
        self.method_head = MethodClassifier(d_model=d_model)
        self.meta: dict[str, Any] = {
            "trained": False,
            "glm_router_weights_copied": False,
            "independently_ablatable": True,
        }

    def enabled_residual_names(
        self,
        *,
        enabled: Sequence[str] | None = None,
        skip: Sequence[str] | None = None,
    ) -> list[str]:
        skip_set = set(skip or ())
        if enabled is None:
            names = list(self.residual.keys())
        else:
            names = [n for n in enabled if n in self.residual]
        return [n for n in names if n not in skip_set]

    def apply_residual(
        self,
        x: torch.Tensor,
        *,
        enabled: Sequence[str] | None = None,
        skip: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        applied: list[str] = []
        y = x
        for name in self.enabled_residual_names(enabled=enabled, skip=skip):
            y = self.residual[name](y)
            applied.append(name)
        return y, applied

    @torch.no_grad()
    def revert_residual(
        self,
        y: torch.Tensor,
        applied: Sequence[str],
    ) -> torch.Tensor:
        x = y
        for name in reversed(list(applied)):
            x = self.residual[name].revert(x)
        return x

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        parts = {
            name: mod.gravity_accounting(bytes_per_param=bytes_per_param)
            for name, mod in self.residual.items()
        }
        parts[self.value_head.name] = self.value_head.gravity_accounting(
            bytes_per_param=bytes_per_param
        )
        parts[self.route_bias.name] = self.route_bias.gravity_accounting(
            bytes_per_param=bytes_per_param
        )
        total = sum(int(p["parameter_bytes"]) for p in parts.values())
        return {
            "adapters": parts,
            "total_parameter_bytes": total,
            "total_parameter_count": param_count(self),
            "hash_bound": self.content_hash(),
            "glm_router_weights_copied": False,
            "independently_ablatable": True,
        }

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": "hawking.frankenstein.adapter_bank.torch.v1",
            "d_model": self.d_model,
            "rank": self.rank,
            "adapters": list(self.residual.keys())
            + [self.value_head.name, self.route_bias.name],
            "gravity": self.gravity_accounting(),
            "trained": bool(self.meta.get("trained")),
            "glm_router_weights_copied": False,
            "meta": dict(self.meta),
        }


class FunctionalStack(nn.Module):
    """Composable stack: optional bridge + residual adapters + aux heads.

    Ablation masks select which pieces participate (A–G variants).
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        d_hidden: int = 32,
        rank: int = 4,
        n_experts: int | None = None,
        use_bridge: bool = True,
        residual_enabled: Sequence[str] | None = None,
        use_value_head: bool = True,
        use_route_bias: bool = True,
        use_method_head: bool = True,
        include_optional_expert_bank: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_bridge = bool(use_bridge)
        self.residual_enabled = (
            list(residual_enabled)
            if residual_enabled is not None
            else list(RESIDUAL_ADAPTER_NAMES)
        )
        self.use_value_head = bool(use_value_head)
        self.use_route_bias = bool(use_route_bias)
        self.use_method_head = bool(use_method_head)

        self.bridge = (
            ReversibleBridge(d_model=d_model, d_hidden=d_hidden, rank=rank)
            if use_bridge
            else None
        )
        self.bank = AdapterBank(
            d_model=d_model,
            rank=rank,
            n_experts=n_experts,
            include_optional_expert_bank=include_optional_expert_bank,
        )

    def forward_hidden(
        self,
        x: torch.Tensor,
        *,
        skip_adapters: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        applied: list[str] = []
        y = x
        if self.bridge is not None and self.use_bridge:
            y = self.bridge(y)
            applied.append(self.bridge.name)
        if self.residual_enabled:
            y, res_applied = self.bank.apply_residual(
                y, enabled=self.residual_enabled, skip=skip_adapters
            )
            applied.extend(res_applied)
        return y, {"applied": applied}

    def forward_bundle(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        skip_adapters: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        x = batch["student_hidden"]
        y, meta = self.forward_hidden(x, skip_adapters=skip_adapters)
        out: dict[str, torch.Tensor] = {"adapted_hidden": y}
        if self.use_value_head:
            out["value"] = self.bank.value_head(y)
        if self.use_method_head:
            out["method_logits"] = self.bank.method_head(y)
        if self.use_route_bias and "route_logits" in batch:
            method_ids = batch.get("method_id")
            out["route_logits"] = self.bank.route_bias(
                batch["route_logits"], method_ids=method_ids
            )
        elif "route_logits" in batch:
            out["route_logits"] = batch["route_logits"]
        if "action_logits" in batch:
            # Action logits are behavior targets; project adapted mean-pool to same dim.
            # Use a frozen random projection stored as buffer for stability when no head.
            act = batch["action_logits"]
            # Behavior distill: match teacher actions from adapted pooled hidden.
            pooled = y.mean(dim=1) if y.dim() == 3 else y
            # Linear map via value_head-less shared projection built on the fly:
            # register once as optional attribute.
            proj = getattr(self, "_action_proj", None)
            if proj is None or proj.in_features != pooled.shape[-1] or proj.out_features != act.shape[-1]:
                proj = nn.Linear(pooled.shape[-1], act.shape[-1], bias=True).to(pooled.device)
                # Keep as submodule so optimizer can train it when present.
                self.add_module("_action_proj", proj)
            out["action_logits"] = self._action_proj(pooled)
        out_meta = {k: v for k, v in meta.items()}
        # stash applied names for callers (non-tensor)
        self._last_applied = out_meta.get("applied", [])
        return out

    def trainable_parameter_groups(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def mark_trained(self) -> None:
        if self.bridge is not None:
            self.bridge.meta["trained"] = True
        self.bank.meta["trained"] = True
        for mod in self.bank.residual.values():
            mod.meta["trained"] = True
        self.bank.value_head.meta["trained"] = True
        self.bank.route_bias.meta["trained"] = True

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        parts: dict[str, Any] = {}
        total = 0
        if self.bridge is not None:
            g = self.bridge.gravity_accounting(bytes_per_param=bytes_per_param)
            parts["bridge"] = g
            total += int(g["parameter_bytes"])
        bank_g = self.bank.gravity_accounting(bytes_per_param=bytes_per_param)
        parts["bank"] = bank_g
        total += int(bank_g["total_parameter_bytes"])
        flops = 0
        if self.bridge is not None:
            flops += int(self.bridge.gravity_accounting()["approx_flops_per_token"])
        # residual low-rank approx: 2*(d*r + r*d) per residual adapter
        d, r = self.d_model, self.rank if hasattr(self, "rank") else self.bank.rank
        for _ in self.residual_enabled:
            flops += 2 * (d * r + r * d)
        return {
            "parts": parts,
            "total_parameter_bytes": total,
            "total_parameter_count": param_count(self),
            "approx_flops_per_token": flops,
            "hash_bound": tensor_content_hash(
                {k: v for k, v in self.state_dict().items()}
            ),
            "bytes_per_param": int(bytes_per_param),
        }


def build_stack_for_arm(
    arm: str,
    *,
    d_model: int = 64,
    d_hidden: int = 32,
    rank: int = 4,
    n_experts: int = 16,
) -> FunctionalStack:
    """Construct an ablatable stack for functional-transfer arm A–G.

    A: identity (no modules) — caller should not train
    B: linear-init stand-in (bridge only, tiny rank — infrastructure, not inheritance)
    C: behavior distill heads (value + method + action; no bridge residual adapters)
    D: nonlinear reversible bridge
    E: router/method adapters (residual method adapters + route bias)
    F: expert-iteration set (repair adapter + value + verifier path)
    G: complete stack
    """

    arm = str(arm).upper()
    if arm.startswith("A") or arm == "A_BASE_DSV4F":
        # Empty residual list, no bridge — forward is identity.
        return FunctionalStack(
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            n_experts=n_experts,
            use_bridge=False,
            residual_enabled=[],
            use_value_head=False,
            use_route_bias=False,
            use_method_head=False,
        )
    if arm.startswith("B") or "LINEAR" in arm:
        return FunctionalStack(
            d_model=d_model,
            d_hidden=d_hidden,
            rank=max(1, rank // 2),
            n_experts=n_experts,
            use_bridge=True,
            residual_enabled=[],
            use_value_head=False,
            use_route_bias=False,
            use_method_head=False,
        )
    if arm.startswith("C") or "BEHAVIOR" in arm:
        return FunctionalStack(
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            n_experts=n_experts,
            use_bridge=False,
            residual_enabled=[],
            use_value_head=True,
            use_route_bias=False,
            use_method_head=True,
        )
    if arm.startswith("D") or "BRIDGE" in arm or "NONLINEAR" in arm:
        return FunctionalStack(
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            n_experts=n_experts,
            use_bridge=True,
            residual_enabled=[],
            use_value_head=False,
            use_route_bias=False,
            use_method_head=False,
        )
    if arm.startswith("E") or "ROUTER" in arm or "METHOD_ADAPTER" in arm:
        return FunctionalStack(
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            n_experts=n_experts,
            use_bridge=False,
            residual_enabled=list(RESIDUAL_ADAPTER_NAMES),
            use_value_head=False,
            use_route_bias=True,
            use_method_head=True,
        )
    if arm.startswith("F") or "EXPERT" in arm:
        return FunctionalStack(
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            n_experts=n_experts,
            use_bridge=False,
            residual_enabled=["GLM_REPAIR_ADAPTER"],
            use_value_head=True,
            use_route_bias=False,
            use_method_head=True,
        )
    # G complete
    return FunctionalStack(
        d_model=d_model,
        d_hidden=d_hidden,
        rank=rank,
        n_experts=n_experts,
        use_bridge=True,
        residual_enabled=list(RESIDUAL_ADAPTER_NAMES),
        use_value_head=True,
        use_route_bias=True,
        use_method_head=True,
    )


def iter_named_modules(stack: FunctionalStack) -> Iterable[tuple[str, nn.Module]]:
    if stack.bridge is not None and stack.use_bridge:
        yield ("bridge", stack.bridge)
    for name in stack.residual_enabled:
        if name in stack.bank.residual:
            yield (name, stack.bank.residual[name])
    if stack.use_value_head:
        yield ("GLM_VALUE_HEAD", stack.bank.value_head)
    if stack.use_route_bias:
        yield ("METHOD_CONDITIONED_ROUTE_BIAS", stack.bank.route_bias)
    if stack.use_method_head:
        yield ("METHOD_CLASSIFIER", stack.bank.method_head)
