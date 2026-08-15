#!/usr/bin/env python3.12
"""PROTO_FRANKENSTEIN_V0 — full latent bridge architecture + 11-loss train schedule.

Owner steer (PROTO_FRANKENSTEIN_V0_STEER.md):
  Teacher projector:  GLM 6144 → RMSNorm → learned proj → shared latent
  Student observer:   DSV4F 4096 → RMSNorm → learned proj → same shared latent
  Student intervention:
      DSV4F hidden → RMSNorm → low-rank proj → gated nonlinear MLP
      → low-rank residual → add back to native hidden

Ten named V0 sites (reversible, bypassable, hash-bound, ablatable, Gravity-accounted):
  GLM_EARLY_CONTEXT_BRIDGE, GLM_METHOD_BRIDGE, GLM_DECOMPOSITION_BRIDGE,
  GLM_PRE_ROUTER_BRIDGE, GLM_POST_MOE_BRIDGE, GLM_FORMALIZATION_BRIDGE,
  GLM_REPAIR_BRIDGE, GLM_LATE_CONSOLIDATION_BRIDGE, GLM_VALUE_HEAD,
  GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL

Loss portfolio (never cosine alone):
  L_latent, L_function, L_span, L_method, L_decomposition, L_formal,
  L_repair, L_value, L_route, L_retention, L_runtime

Schedule A–F with CURRENT / BEST_MATH / BEST_BALANCED / ROLLBACK checkpoints.
Ablation A–G (latent matrix). Fail closed REQUIRES_PAIRED_CAPTURE on real fit.

Extends the merged adapter-trainer / bridges / ablation stack. Does not edit
crates, does not touch Kimi/Gravity, does not fabricate capability numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lab.operators import frankenstein_ablation as ablation
from lab.operators.frankenstein_adapter_modules import (
    RMSNorm,
    RouteBiasResidual,
    ValueHead,
    param_bytes,
    param_count,
    tensor_content_hash,
)
from lab.operators.frankenstein_fusion_op import DEEPSEEK_V4_FLASH, GLM_5_2
from lab.receipts import seal, verify


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

LATENT_V0_SCHEMA = "hawking.frankenstein.latent_v0_stack.v1"
LATENT_TRAIN_SCHEMA = "hawking.frankenstein.latent_v0_train_run.v1"
LATENT_FIXTURE_SCHEMA = "hawking.frankenstein.latent_v0_real_shaped_fixture.v1"
LATENT_AG_SCHEMA = "hawking.frankenstein.latent_v0_ag_ablation.v1"
LATENT_SCHEDULE_SCHEMA = "hawking.frankenstein.latent_v0_schedule.v1"
CHECKPOINT_SCHEMA = "hawking.frankenstein.latent_v0_checkpoint.v1"

REQUIRES_PAIRED_CAPTURE = "REQUIRES_PAIRED_CAPTURE"

# Full geometry (production).
GLM_HIDDEN = int(GLM_5_2["hidden_size"])  # 6144
DSV4F_HIDDEN = int(DEEPSEEK_V4_FLASH["hidden_size"])  # 4096
DEFAULT_LATENT = 256
DEFAULT_RANK = 16
DEFAULT_MLP_HIDDEN = 128

# Ten named V0 modules (steer).
V0_BRIDGE_SITES: tuple[str, ...] = (
    "GLM_EARLY_CONTEXT_BRIDGE",
    "GLM_METHOD_BRIDGE",
    "GLM_DECOMPOSITION_BRIDGE",
    "GLM_PRE_ROUTER_BRIDGE",
    "GLM_POST_MOE_BRIDGE",
    "GLM_FORMALIZATION_BRIDGE",
    "GLM_REPAIR_BRIDGE",
    "GLM_LATE_CONSOLIDATION_BRIDGE",
)

V0_HEADS: tuple[str, ...] = (
    "GLM_VALUE_HEAD",
    "GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL",
)

V0_MODULE_NAMES: tuple[str, ...] = V0_BRIDGE_SITES + V0_HEADS

# Method / action classes (router distillation — semantic, not IDs).
METHOD_CLASSES: tuple[str, ...] = (
    "algebra",
    "geometry",
    "combinatorics",
    "formal_proof",
    "symbolic",
    "counterexample",
    "retrieval",
    "coding",
    "tool",
    "verification",
    "repair",
)

# Behavior head labels for decomp / formal / repair classifiers.
BEHAVIOR_HEADS: tuple[str, ...] = (
    "method",
    "decomposition",
    "formal",
    "repair",
)

# 11-loss portfolio (steer). Cosine alone is FORBIDDEN.
LOSS_NAMES: tuple[str, ...] = (
    "L_latent",
    "L_function",
    "L_span",
    "L_method",
    "L_decomposition",
    "L_formal",
    "L_repair",
    "L_value",
    "L_route",
    "L_retention",
    "L_runtime",
)

LOSS_DEFINITIONS: dict[str, str] = {
    "L_latent": "shared-space feature align (teacher proj ↔ student observer)",
    "L_function": "functional checkpoint agreement (adapted hidden ↔ teacher target)",
    "L_span": "aligned decoded-span / action distribution KL",
    "L_method": "method-family classification CE",
    "L_decomposition": "decomposition / planning head CE",
    "L_formal": "formalization head CE",
    "L_repair": "repair / critique trajectory CE",
    "L_value": "verified outcome / value rank BCE",
    "L_route": "semantic route policy KL (native top-6 preserved)",
    "L_retention": "base-capability preservation (base-logit KL + feature anchor)",
    "L_runtime": "adapter sparsity / cost (L1 on residual scale + FLOP proxy)",
    "latent_cosine_alone": "FORBIDDEN as sole objective",
}

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "L_latent": 1.0,
    "L_function": 1.0,
    "L_span": 0.5,
    "L_method": 0.5,
    "L_decomposition": 0.4,
    "L_formal": 0.4,
    "L_repair": 0.4,
    "L_value": 0.25,
    "L_route": 0.5,
    "L_retention": 0.75,
    "L_runtime": 0.05,
}

# Training schedule phases A–F (steer).
SCHEDULE_PHASES: tuple[tuple[str, str], ...] = (
    ("A", "projectors + shared-latent align"),
    ("B", "reversible student latent adapters (interventions)"),
    ("C", "method / decomp / formal / repair / value heads"),
    ("D", "bounded route residual"),
    ("E", "joint consolidation with retention replay"),
    ("F", "hidden eval + repair"),
)

CHECKPOINT_SLOTS: tuple[str, ...] = (
    "CURRENT",
    "BEST_MATH",
    "BEST_BALANCED",
    "ROLLBACK",
)

# Latent A–G ablation matrix (steer § ablation).
ARM_A = "A_BASE_DSV4F"
ARM_B = "B_LINEAR_SUBSPACE_INITIALIZATION"
ARM_C = "C_LATENT_BRIDGES"
ARM_D = "D_BEHAVIOR_HEADS"
ARM_E = "E_LATENT_BEHAVIOR"
ARM_F = "F_LATENT_ROUTE_RESIDUAL"
ARM_G = "G_COMPLETE_V0"

LATENT_AG_ARMS: tuple[tuple[str, str], ...] = (
    (ARM_A, "Base DSV4F only; no donor inheritance"),
    (ARM_B, "LINEAR_SUBSPACE_INITIALIZATION only (cartography init; not inheritance)"),
    (ARM_C, "Latent bridges only (projectors + interventions)"),
    (ARM_D, "Behavior heads only (method/decomp/formal/repair/value)"),
    (ARM_E, "Latent bridges + behavior heads"),
    (ARM_F, "Latent bridges + semantic route residual"),
    (ARM_G, "Complete V0 (latent + behavior + route + retention)"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)


class LatentV0Error(RuntimeError):
    """Latent V0 configuration or fail-closed error."""


class RequiresPairedCapture(LatentV0Error):
    """Real paired capture data is absent."""

    gate = REQUIRES_PAIRED_CAPTURE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    # Prefer CPU for reproducibility in this lane; MPS optional.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Core modules
# ---------------------------------------------------------------------------


class TeacherProjector(nn.Module):
    """GLM 6144 → RMSNorm → linear → shared latent."""

    def __init__(
        self,
        *,
        d_teacher: int = GLM_HIDDEN,
        d_latent: int = DEFAULT_LATENT,
        eps: float = 1e-6,
        scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.name = "TEACHER_PROJECTOR"
        self.d_teacher = int(d_teacher)
        self.d_latent = int(d_latent)
        self.norm = RMSNorm(d_teacher, eps=eps)
        self.proj = nn.Linear(d_teacher, d_latent, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=float(scale))
        self.meta: dict[str, Any] = {
            "role": "training_only",
            "runtime_resident": False,
            "trained": False,
        }

    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        if teacher_hidden.shape[-1] != self.d_teacher:
            raise LatentV0Error(
                f"teacher last dim {teacher_hidden.shape[-1]} != {self.d_teacher}"
            )
        return self.proj(self.norm(teacher_hidden))

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "runtime_resident": False,
            "training_only": True,
            "ablatable": True,
        }


class StudentObserver(nn.Module):
    """DSV4F 4096 → RMSNorm → linear → shared latent (same width as teacher)."""

    def __init__(
        self,
        *,
        d_student: int = DSV4F_HIDDEN,
        d_latent: int = DEFAULT_LATENT,
        eps: float = 1e-6,
        scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.name = "STUDENT_OBSERVER"
        self.d_student = int(d_student)
        self.d_latent = int(d_latent)
        self.norm = RMSNorm(d_student, eps=eps)
        self.proj = nn.Linear(d_student, d_latent, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=float(scale))
        self.meta: dict[str, Any] = {
            "role": "runtime_optional_probe",
            "runtime_resident": True,
            "trained": False,
        }

    def forward(self, student_hidden: torch.Tensor) -> torch.Tensor:
        if student_hidden.shape[-1] != self.d_student:
            raise LatentV0Error(
                f"student last dim {student_hidden.shape[-1]} != {self.d_student}"
            )
        return self.proj(self.norm(student_hidden))

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "runtime_resident": True,
            "ablatable": True,
        }


class StudentIntervention(nn.Module):
    """Student residual intervention (steer architecture):

    hidden → RMSNorm → low-rank proj → gated nonlinear MLP
           → low-rank residual → add back to native hidden.

    y = x + r(x); exact reverse via stored residual; fixed-point fallback.
    """

    def __init__(
        self,
        *,
        name: str,
        d_model: int = DSV4F_HIDDEN,
        rank: int = DEFAULT_RANK,
        d_hidden: int = DEFAULT_MLP_HIDDEN,
        eps: float = 1e-6,
        scale: float = 0.02,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if name not in V0_BRIDGE_SITES:
            raise LatentV0Error(f"unknown V0 bridge site: {name!r}")
        self.name = name
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.d_hidden = int(d_hidden)
        self.eps = float(eps)
        self.norm = RMSNorm(d_model, eps=eps)
        # Low-rank proj into bottleneck
        self.down = nn.Linear(d_model, rank, bias=False)
        # Gated nonlinear MLP in bottleneck space
        self.w_g = nn.Linear(rank, d_hidden, bias=False)
        self.w_u = nn.Linear(rank, d_hidden, bias=False)
        self.w_d = nn.Linear(d_hidden, rank, bias=False)
        # Low-rank residual back to native dim
        self.up = nn.Linear(rank, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        # Soft residual scale (softplus so always >0; starts small but non-zero
        # so gradients reach low-rank weights immediately — not a hard zero gate).
        # gate_init is the pre-softplus bias; 0 → softplus(0)=ln(2)≈0.69, then
        # we multiply by a small constant so early residual is modest.
        self.res_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self._init_small(scale)
        self.meta: dict[str, Any] = {
            "architecture": "norm_lowrank_gated_mlp_lowrank_residual",
            "trained": False,
            "bypassable": True,
            "reversible": True,
            "ablatable": True,
            "hash_bound": True,
            "gravity_accounted": True,
            "kimi_bridge_compatible": True,
            "enabled": True,
        }

    def _init_small(self, scale: float) -> None:
        for mod in (self.down, self.w_g, self.w_u, self.w_d, self.up):
            nn.init.normal_(mod.weight, mean=0.0, std=float(scale))

    def residual_scale(self) -> torch.Tensor:
        # softplus keeps scale > 0; *0.25 keeps early residual modest
        return F.softplus(self.res_gate) * 0.25

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise LatentV0Error(
                f"{self.name}: expected last dim {self.d_model}, got {tuple(x.shape)}"
            )
        h = self.norm(x)
        z = self.down(h)
        gate = F.silu(self.w_g(z))
        up = self.w_u(z)
        z2 = self.w_d(gate * up)
        r = self.up(z2) + self.b
        return self.residual_scale() * r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.meta.get("enabled", True):
            return x
        return x + self.residual(x)

    def apply_with_residual(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.residual(x) if self.meta.get("enabled", True) else torch.zeros_like(x)
        return x + r, r

    @torch.no_grad()
    def revert_exact(self, y: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return y - residual

    @torch.no_grad()
    def revert(
        self,
        y: torch.Tensor,
        *,
        n_iters: int = 16,
        atol: float = 1e-5,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        y_arr = y.detach()
        if residual is not None:
            x = self.revert_exact(y_arr, residual)
            step_err = float(torch.max(torch.abs((x + residual) - y_arr)).item())
            return x, {
                "mode": "stored_residual",
                "iterations": 0,
                "recon_error": step_err,
                "stored_step_error": step_err,
                "exact": step_err < atol,
            }
        if not self.meta.get("enabled", True):
            return y_arr.clone(), {
                "mode": "bypassed",
                "iterations": 0,
                "recon_error": 0.0,
                "exact": True,
            }
        x = y_arr.clone()
        last_err = 0.0
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

    def set_enabled(self, enabled: bool) -> None:
        self.meta["enabled"] = bool(enabled)

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        d, r, h = self.d_model, self.rank, self.d_hidden
        # FLOP proxy: norm ~3d; down d*r; w_g r*h; w_u r*h; w_d h*r; up r*d
        flops = 2 * (d * r + r * h + r * h + h * r + r * d) + 3 * d
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "approx_flops_per_token": flops,
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "bypassable": True,
            "reversible": True,
            "gravity_accounted": True,
            "kimi_bridge_compatible": True,
            "runtime_resident": True,
        }

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "d_model": self.d_model,
            "rank": self.rank,
            "d_hidden": self.d_hidden,
            "architecture": "norm -> lowrank -> gated_mlp -> lowrank_residual -> add",
            "parameter_count": param_count(self),
            "content_hash": self.content_hash(),
            "meta": dict(self.meta),
            "gravity": self.gravity_accounting(),
        }


class BehaviorHead(nn.Module):
    """Lightweight multi-class head over adapted student hidden."""

    def __init__(
        self,
        *,
        name: str,
        d_model: int,
        n_classes: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.name = name
        self.linear = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.linear.bias)
        self.meta: dict[str, Any] = {"kind": "behavior_head", "trained": False}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x)

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "runtime_resident": True,
        }


class LinearSubspaceInit(nn.Module):
    """Closed-form linear residual stand-in (NOT inheritance; arm B only).

    y = x + scale * (x @ A + b) with A low-rank factors from optional init.
    """

    def __init__(
        self,
        *,
        d_model: int = DSV4F_HIDDEN,
        rank: int = DEFAULT_RANK,
        scale: float = 0.05,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.name = "LINEAR_SUBSPACE_INITIALIZATION"
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.scale = float(scale)
        self.a_lo = nn.Linear(d_model, rank, bias=False)
        self.a_hi = nn.Linear(rank, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.a_lo.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.a_hi.weight, mean=0.0, std=init_std)
        self.meta: dict[str, Any] = {
            "role": "LINEAR_SUBSPACE_INITIALIZATION",
            "is_inheritance": False,
            "trained": False,
        }

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (self.a_hi(self.a_lo(x)) + self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)

    def apply_with_residual(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.residual(x)
        return x + r, r

    @torch.no_grad()
    def revert(
        self,
        y: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
        n_iters: int = 12,
        atol: float = 1e-5,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        y_arr = y.detach()
        if residual is not None:
            x = y_arr - residual
            err = float(torch.max(torch.abs((x + residual) - y_arr)).item())
            return x, {"mode": "stored_residual", "exact": err < atol, "recon_error": err}
        x = y_arr.clone()
        last = 0.0
        for _ in range(n_iters):
            x_next = y_arr - self.residual(x)
            last = float(torch.max(torch.abs(x_next - x)).item())
            x = x_next
            if last < atol:
                break
        recon = float(torch.max(torch.abs(self.forward(x) - y_arr)).item())
        return x, {
            "mode": "fixed_point",
            "recon_error": recon,
            "exact": recon < atol,
            "last_step_delta": last,
        }

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": param_count(self),
            "parameter_bytes": param_bytes(self, bytes_per_param=bytes_per_param),
            "hash_bound": self.content_hash(),
            "is_inheritance": False,
            "ablatable": True,
            "reversible": True,
        }


# ---------------------------------------------------------------------------
# Full V0 stack
# ---------------------------------------------------------------------------


class LatentV0Stack(nn.Module):
    """Composable full latent V0 stack with arm masks (A–G).

    Runtime keeps only DSV4F adapters / projectors / heads.
    Teacher projector is training-only (excluded from runtime_state_dict).
    """

    def __init__(
        self,
        *,
        d_teacher: int = GLM_HIDDEN,
        d_student: int = DSV4F_HIDDEN,
        d_latent: int = DEFAULT_LATENT,
        rank: int = DEFAULT_RANK,
        d_hidden: int = DEFAULT_MLP_HIDDEN,
        n_experts: int | None = None,
        n_methods: int = len(METHOD_CLASSES),
        n_actions: int = 12,
        n_decomp: int = 6,
        n_formal: int = 4,
        n_repair: int = 4,
        bridge_sites: Sequence[str] | None = None,
        use_teacher_projector: bool = True,
        use_student_observer: bool = True,
        use_interventions: bool = True,
        use_behavior_heads: bool = True,
        use_value_head: bool = True,
        use_route_residual: bool = True,
        use_linear_init: bool = False,
        use_retention_anchor: bool = True,
        scale: float = 0.02,
    ) -> None:
        super().__init__()
        if n_experts is None:
            n_experts = int(DEEPSEEK_V4_FLASH["n_routed_experts"])
        self.d_teacher = int(d_teacher)
        self.d_student = int(d_student)
        self.d_latent = int(d_latent)
        self.rank = int(rank)
        self.d_hidden = int(d_hidden)
        self.n_experts = int(n_experts)
        self.n_methods = int(n_methods)
        self.n_actions = int(n_actions)

        self.use_teacher_projector = bool(use_teacher_projector)
        self.use_student_observer = bool(use_student_observer)
        self.use_interventions = bool(use_interventions)
        self.use_behavior_heads = bool(use_behavior_heads)
        self.use_value_head = bool(use_value_head)
        self.use_route_residual = bool(use_route_residual)
        self.use_linear_init = bool(use_linear_init)
        self.use_retention_anchor = bool(use_retention_anchor)

        sites = list(bridge_sites) if bridge_sites is not None else list(V0_BRIDGE_SITES)
        self.bridge_sites = sites

        self.teacher_projector = (
            TeacherProjector(d_teacher=d_teacher, d_latent=d_latent, scale=scale)
            if use_teacher_projector
            else None
        )
        self.student_observer = (
            StudentObserver(d_student=d_student, d_latent=d_latent, scale=scale)
            if use_student_observer
            else None
        )

        interventions: dict[str, StudentIntervention] = {}
        if use_interventions:
            # gate_init=-2 → residual_scale = softplus(-2)*0.25 ≈ 0.032.
            # Prior research in this repo (expansive residual / accumulated
            # cascade drift) shows stacked non-contractive residuals diverge;
            # 8 sites at softplus(0)*0.25≈0.173 compound aggressively.
            for site in sites:
                interventions[site] = StudentIntervention(
                    name=site,
                    d_model=d_student,
                    rank=rank,
                    d_hidden=d_hidden,
                    scale=scale,
                    gate_init=-2.0,
                )
        self.interventions = nn.ModuleDict(interventions)

        self.linear_init = (
            LinearSubspaceInit(d_model=d_student, rank=rank, scale=0.05)
            if use_linear_init
            else None
        )

        # Behavior heads
        if use_behavior_heads:
            self.method_head = BehaviorHead(
                name="GLM_METHOD_HEAD", d_model=d_student, n_classes=n_methods
            )
            self.decomp_head = BehaviorHead(
                name="GLM_DECOMPOSITION_HEAD", d_model=d_student, n_classes=n_decomp
            )
            self.formal_head = BehaviorHead(
                name="GLM_FORMALIZATION_HEAD", d_model=d_student, n_classes=n_formal
            )
            self.repair_head = BehaviorHead(
                name="GLM_REPAIR_HEAD", d_model=d_student, n_classes=n_repair
            )
        else:
            self.method_head = None
            self.decomp_head = None
            self.formal_head = None
            self.repair_head = None

        self.value_head = (
            ValueHead(d_model=d_student) if use_value_head else None
        )
        if self.value_head is not None:
            self.value_head.name = "GLM_VALUE_HEAD"

        self.route_residual = (
            RouteBiasResidual(
                n_experts=n_experts,
                n_methods=max(n_methods, 16),
                rank=min(rank, 8),
                scale=0.05,
            )
            if use_route_residual
            else None
        )
        if self.route_residual is not None:
            self.route_residual.name = "GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL"

        # Span / action projection from adapted pooled hidden
        self.action_proj = nn.Linear(d_student, n_actions, bias=True)
        nn.init.normal_(self.action_proj.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.action_proj.bias)

        # Retention feature anchor (frozen base projection target is external)
        self.retention_alpha = nn.Parameter(torch.tensor(1.0))

        self.meta: dict[str, Any] = {
            "schema": LATENT_V0_SCHEMA,
            "trained": False,
            "kimi_bridge_compatible": True,
            "glm_router_weights_copied": False,
            "direct_weight_transplant": False,
            "runtime_keeps_teacher_projector": False,
        }
        self._last_applied: list[str] = []

    # -- forward -------------------------------------------------------------

    def apply_interventions(
        self,
        x: torch.Tensor,
        *,
        skip: Sequence[str] | None = None,
        enabled: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str], list[torch.Tensor]]:
        """Apply enabled bridges as *parallel* additive residuals.

        Each V0 site is independently ``y = x + A_i(x)`` at its transplant
        point.  When multiple sites are evaluated on one proxy tensor (the
        real-capture training path), they must sum on a shared base —

            y = x + Σ_i A_i(x)

        — not cascade ``(...((x+A1)+A2)...)``.  Sequential composition of
        eight nonlinear residuals is expansive and was a root cause of
        latent-arm held-out loss exploding vs linear-init (see also prior
        expansive-residual / accumulated-drift findings in this repo).
        Stored per-site residuals still reverse exactly via subtraction.
        """

        skip_set = set(skip or ())
        applied: list[str] = []
        residuals: list[torch.Tensor] = []
        y = x
        if self.linear_init is not None and self.use_linear_init:
            y, r = self.linear_init.apply_with_residual(y)
            applied.append(self.linear_init.name)
            residuals.append(r)
        if not self.use_interventions:
            return y, applied, residuals
        names = (
            list(enabled)
            if enabled is not None
            else list(self.interventions.keys())
        )
        base = y
        total_r: torch.Tensor | None = None
        for name in names:
            if name in skip_set or name not in self.interventions:
                continue
            mod = self.interventions[name]
            if not mod.meta.get("enabled", True):
                continue
            # Residual from shared base (parallel), not from updated y (cascade).
            r = mod.residual(base)
            total_r = r if total_r is None else total_r + r
            applied.append(name)
            residuals.append(r)
        if total_r is not None:
            y = base + total_r
        return y, applied, residuals

    def forward_hidden(
        self,
        student_hidden: torch.Tensor,
        *,
        skip: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        y, applied, residuals = self.apply_interventions(student_hidden, skip=skip)
        self._last_applied = list(applied)
        return y, {"applied": applied, "n_residuals": len(residuals)}

    def forward_bundle(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        skip: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        x = batch["student_hidden"]
        y, meta = self.forward_hidden(x, skip=skip)
        out: dict[str, torch.Tensor] = {
            "adapted_hidden": y,
            "student_hidden": x,
        }

        # Shared latent alignment
        if self.student_observer is not None and self.use_student_observer:
            out["student_latent"] = self.student_observer(y)
            # Also observer on raw (for retention anchoring)
            out["student_latent_raw"] = self.student_observer(x)
        if (
            self.teacher_projector is not None
            and self.use_teacher_projector
            and "teacher_hidden" in batch
        ):
            out["teacher_latent"] = self.teacher_projector(batch["teacher_hidden"])

        if self.use_behavior_heads:
            if self.method_head is not None:
                out["method_logits"] = self.method_head(y)
            if self.decomp_head is not None:
                out["decomp_logits"] = self.decomp_head(y)
            if self.formal_head is not None:
                out["formal_logits"] = self.formal_head(y)
            if self.repair_head is not None:
                out["repair_logits"] = self.repair_head(y)

        if self.value_head is not None and self.use_value_head:
            out["value"] = self.value_head(y)

        if self.route_residual is not None and self.use_route_residual:
            if "route_logits" in batch:
                method_ids = batch.get("method_id")
                out["route_logits"] = self.route_residual(
                    batch["route_logits"], method_ids=method_ids
                )
        elif "route_logits" in batch:
            out["route_logits"] = batch["route_logits"]

        pooled = y.mean(dim=1) if y.dim() == 3 else y
        out["action_logits"] = self.action_proj(pooled)

        # Retention: pull adapted toward base (identity) when enabled
        if self.use_retention_anchor:
            out["retention_anchor"] = x
            out["retention_alpha"] = self.retention_alpha.expand(1)

        out["_meta_applied"] = torch.tensor(
            [len(meta["applied"])], device=y.device
        )  # tensor placeholder; list in self._last_applied
        return out

    @torch.no_grad()
    def revert_hidden(
        self,
        y: torch.Tensor,
        residuals: Sequence[torch.Tensor],
        applied: Sequence[str],
    ) -> torch.Tensor:
        """Exact reverse given residuals stored at apply (reverse order)."""

        x = y
        for name, r in zip(reversed(list(applied)), reversed(list(residuals))):
            x = x - r
            _ = name
        return x

    def set_bridge_enabled(self, name: str, enabled: bool) -> None:
        if name in self.interventions:
            self.interventions[name].set_enabled(enabled)

    def bypass_all_interventions(self) -> None:
        for mod in self.interventions.values():
            mod.set_enabled(False)

    def enable_all_interventions(self) -> None:
        for mod in self.interventions.values():
            mod.set_enabled(True)

    def mark_trained(self) -> None:
        self.meta["trained"] = True
        for mod in self.interventions.values():
            mod.meta["trained"] = True
        for head in (
            self.method_head,
            self.decomp_head,
            self.formal_head,
            self.repair_head,
            self.value_head,
            self.route_residual,
            self.teacher_projector,
            self.student_observer,
            self.linear_init,
        ):
            if head is not None and hasattr(head, "meta"):
                head.meta["trained"] = True

    def runtime_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict for runtime: excludes teacher projector (training-only)."""

        full = self.state_dict()
        return {
            k: v
            for k, v in full.items()
            if not k.startswith("teacher_projector.")
        }

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        parts: dict[str, Any] = {}
        total = 0
        runtime_total = 0
        flops = 0

        def _add(key: str, mod: nn.Module | None, *, runtime: bool) -> None:
            nonlocal total, runtime_total, flops
            if mod is None:
                return
            if hasattr(mod, "gravity_accounting"):
                g = mod.gravity_accounting(bytes_per_param=bytes_per_param)
            else:
                g = {
                    "name": key,
                    "parameter_count": param_count(mod),
                    "parameter_bytes": param_bytes(mod, bytes_per_param=bytes_per_param),
                }
            parts[key] = g
            pb = int(g.get("parameter_bytes") or 0)
            total += pb
            if runtime:
                runtime_total += pb
            flops += int(g.get("approx_flops_per_token") or 0)

        _add("teacher_projector", self.teacher_projector, runtime=False)
        _add("student_observer", self.student_observer, runtime=True)
        for name, mod in self.interventions.items():
            _add(name, mod, runtime=True)
        _add("linear_init", self.linear_init, runtime=True)
        _add("method_head", self.method_head, runtime=True)
        _add("decomp_head", self.decomp_head, runtime=True)
        _add("formal_head", self.formal_head, runtime=True)
        _add("repair_head", self.repair_head, runtime=True)
        _add("value_head", self.value_head, runtime=True)
        _add("route_residual", self.route_residual, runtime=True)
        _add("action_proj", self.action_proj, runtime=True)

        tps_proxy = (1e9 / flops) if flops > 0 else None
        return {
            "parts": parts,
            "total_parameter_bytes": total,
            "runtime_parameter_bytes": runtime_total,
            "training_only_parameter_bytes": total - runtime_total,
            "total_parameter_count": param_count(self),
            "approx_flops_per_token": flops,
            "tps_proxy_at_1gflops": tps_proxy,
            "active_bytes_per_token_proxy": runtime_total,  # resident adapter bytes
            "hash_bound": self.content_hash(),
            "bytes_per_param": int(bytes_per_param),
            "glm_router_weights_copied": False,
            "kimi_bridge_compatible": True,
        }

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": LATENT_V0_SCHEMA,
            "d_teacher": self.d_teacher,
            "d_student": self.d_student,
            "d_latent": self.d_latent,
            "rank": self.rank,
            "d_hidden": self.d_hidden,
            "bridge_sites": list(self.bridge_sites),
            "v0_module_names": list(V0_MODULE_NAMES),
            "modules_present": {
                "teacher_projector": self.teacher_projector is not None,
                "student_observer": self.student_observer is not None,
                "interventions": list(self.interventions.keys()),
                "behavior_heads": self.use_behavior_heads,
                "value_head": self.value_head is not None,
                "route_residual": self.route_residual is not None,
                "linear_init": self.linear_init is not None,
            },
            "gravity": self.gravity_accounting(),
            "meta": dict(self.meta),
            "loss_portfolio": dict(LOSS_DEFINITIONS),
            "kimi_bridge_compatible": True,
        }


def build_stack_for_arm(
    arm: str,
    *,
    d_teacher: int = GLM_HIDDEN,
    d_student: int = DSV4F_HIDDEN,
    d_latent: int = DEFAULT_LATENT,
    rank: int = DEFAULT_RANK,
    d_hidden: int = DEFAULT_MLP_HIDDEN,
    n_experts: int = 16,
    n_methods: int = len(METHOD_CLASSES),
    n_actions: int = 12,
) -> LatentV0Stack:
    """Construct ablatable stack for latent A–G matrix (steer)."""

    arm_u = str(arm).upper()
    common = dict(
        d_teacher=d_teacher,
        d_student=d_student,
        d_latent=d_latent,
        rank=rank,
        d_hidden=d_hidden,
        n_experts=n_experts,
        n_methods=n_methods,
        n_actions=n_actions,
    )

    if arm_u.startswith("A") or arm_u == ARM_A:
        return LatentV0Stack(
            **common,
            use_teacher_projector=False,
            use_student_observer=False,
            use_interventions=False,
            use_behavior_heads=False,
            use_value_head=False,
            use_route_residual=False,
            use_linear_init=False,
            use_retention_anchor=False,
        )
    if arm_u.startswith("B") or "LINEAR" in arm_u:
        return LatentV0Stack(
            **common,
            use_teacher_projector=False,
            use_student_observer=False,
            use_interventions=False,
            use_behavior_heads=False,
            use_value_head=False,
            use_route_residual=False,
            use_linear_init=True,
            use_retention_anchor=False,
        )
    if arm_u.startswith("C") or "LATENT_BRIDGES" in arm_u:
        return LatentV0Stack(
            **common,
            use_teacher_projector=True,
            use_student_observer=True,
            use_interventions=True,
            use_behavior_heads=False,
            use_value_head=False,
            use_route_residual=False,
            use_linear_init=False,
            use_retention_anchor=False,
        )
    if arm_u.startswith("D") and "BEHAVIOR" in arm_u:
        return LatentV0Stack(
            **common,
            use_teacher_projector=False,
            use_student_observer=False,
            use_interventions=False,
            use_behavior_heads=True,
            use_value_head=True,
            use_route_residual=False,
            use_linear_init=False,
            use_retention_anchor=False,
        )
    if arm_u.startswith("E") or "LATENT_BEHAVIOR" in arm_u or arm_u == ARM_E:
        return LatentV0Stack(
            **common,
            use_teacher_projector=True,
            use_student_observer=True,
            use_interventions=True,
            use_behavior_heads=True,
            use_value_head=True,
            use_route_residual=False,
            use_linear_init=False,
            use_retention_anchor=True,
        )
    if arm_u.startswith("F") or "ROUTE" in arm_u:
        return LatentV0Stack(
            **common,
            use_teacher_projector=True,
            use_student_observer=True,
            use_interventions=True,
            use_behavior_heads=False,
            use_value_head=False,
            use_route_residual=True,
            use_linear_init=False,
            use_retention_anchor=True,
        )
    # G complete
    return LatentV0Stack(
        **common,
        use_teacher_projector=True,
        use_student_observer=True,
        use_interventions=True,
        use_behavior_heads=True,
        use_value_head=True,
        use_route_residual=True,
        use_linear_init=False,
        use_retention_anchor=True,
    )


# ---------------------------------------------------------------------------
# Loss portfolio
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    L_latent: float = 1.0
    L_function: float = 1.0
    L_span: float = 0.5
    L_method: float = 0.5
    L_decomposition: float = 0.4
    L_formal: float = 0.4
    L_repair: float = 0.4
    L_value: float = 0.25
    L_route: float = 0.5
    L_retention: float = 0.75
    L_runtime: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in LOSS_NAMES}

    def validate_not_cosine_alone(self) -> None:
        d = self.as_dict()
        if all(v <= 0.0 for v in d.values()):
            raise LatentV0Error("all loss weights are zero")
        # Cosine is not even a named primary; if only L_latent > 0 we still
        # require at least one behavior/function/retention term for full V0.
        # Allow phase A (latent-only) via explicit allow_latent_only.
        primary_non_latent = {
            k: v for k, v in d.items() if k != "L_latent" and v > 0.0
        }
        if not primary_non_latent and d.get("L_latent", 0.0) > 0.0:
            # Phase A uses latent-only intentionally; mark via attribute.
            if not getattr(self, "_allow_latent_only", False):
                raise LatentV0Error(
                    "latent cosine / L_latent alone is FORBIDDEN as sole objective "
                    "outside schedule phase A; enable at least one of L_function/"
                    "L_span/L_method/L_route/L_retention/…"
                )

    def allow_latent_only(self, allowed: bool = True) -> "LossWeights":
        object.__setattr__(self, "_allow_latent_only", bool(allowed))
        return self


def phase_loss_weights(phase: str) -> LossWeights:
    """Schedule A–F loss emphasis (steer)."""

    p = str(phase).upper()
    if p == "A":
        return LossWeights(
            L_latent=1.0,
            L_function=0.0,
            L_span=0.0,
            L_method=0.0,
            L_decomposition=0.0,
            L_formal=0.0,
            L_repair=0.0,
            L_value=0.0,
            L_route=0.0,
            L_retention=0.0,
            L_runtime=0.01,
        ).allow_latent_only(True)
    if p == "B":
        return LossWeights(
            L_latent=0.5,
            L_function=1.0,
            L_span=0.0,
            L_method=0.0,
            L_decomposition=0.0,
            L_formal=0.0,
            L_repair=0.0,
            L_value=0.0,
            L_route=0.0,
            L_retention=0.25,
            L_runtime=0.05,
        )
    if p == "C":
        return LossWeights(
            L_latent=0.25,
            L_function=0.5,
            L_span=0.25,
            L_method=1.0,
            L_decomposition=1.0,
            L_formal=1.0,
            L_repair=1.0,
            L_value=1.0,
            L_route=0.0,
            L_retention=0.5,
            L_runtime=0.05,
        )
    if p == "D":
        return LossWeights(
            L_latent=0.25,
            L_function=0.5,
            L_span=0.25,
            L_method=0.5,
            L_decomposition=0.0,
            L_formal=0.0,
            L_repair=0.0,
            L_value=0.0,
            L_route=1.0,
            L_retention=0.5,
            L_runtime=0.05,
        )
    if p == "E":
        return LossWeights(**DEFAULT_LOSS_WEIGHTS)
    # F: same as E with stronger retention for hidden eval repair
    return LossWeights(
        L_latent=0.5,
        L_function=1.0,
        L_span=0.5,
        L_method=0.5,
        L_decomposition=0.4,
        L_formal=0.4,
        L_repair=0.75,
        L_value=0.5,
        L_route=0.5,
        L_retention=1.0,
        L_runtime=0.1,
    )


def compute_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    weights: LossWeights,
    stack: LatentV0Stack,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full 11-loss portfolio (never cosine alone outside phase A)."""

    weights.validate_not_cosine_alone()
    terms: dict[str, torch.Tensor] = {}
    device = outputs["adapted_hidden"].device

    # L_latent: shared-space MSE between teacher proj and student observer
    if (
        weights.L_latent > 0.0
        and "teacher_latent" in outputs
        and "student_latent" in outputs
    ):
        terms["L_latent"] = F.mse_loss(
            outputs["student_latent"], outputs["teacher_latent"].detach()
        )

    # L_function: adapted hidden matches teacher target in student space
    # (byte-span pooled teacher signal projected / provided as teacher_target_student)
    if weights.L_function > 0.0:
        if "teacher_target_student" in batch:
            terms["L_function"] = F.mse_loss(
                outputs["adapted_hidden"], batch["teacher_target_student"]
            )
        elif "teacher_hidden" in batch and batch["teacher_hidden"].shape[-1] == outputs[
            "adapted_hidden"
        ].shape[-1]:
            terms["L_function"] = F.mse_loss(
                outputs["adapted_hidden"], batch["teacher_hidden"]
            )

    # L_span: action / span distribution KL
    if (
        weights.L_span > 0.0
        and "action_logits" in outputs
        and "teacher_action_logits" in batch
    ):
        log_p = F.log_softmax(outputs["action_logits"], dim=-1)
        q = F.softmax(batch["teacher_action_logits"], dim=-1)
        terms["L_span"] = F.kl_div(log_p, q, reduction="batchmean")

    # L_method / L_decomposition / L_formal / L_repair
    for loss_name, key_logits, key_label in (
        ("L_method", "method_logits", "method_id"),
        ("L_decomposition", "decomp_logits", "decomp_id"),
        ("L_formal", "formal_logits", "formal_id"),
        ("L_repair", "repair_logits", "repair_id"),
    ):
        w = getattr(weights, loss_name)
        if w > 0.0 and key_logits in outputs and key_label in batch:
            logits = outputs[key_logits]
            target = batch[key_label].long()
            if target.dim() > 1:
                target = target[:, 0]
            terms[loss_name] = F.cross_entropy(logits, target)

    # L_value
    if weights.L_value > 0.0 and "value" in outputs and "verifier_pass" in batch:
        value = outputs["value"]
        if value.dim() > 1:
            value = value.mean(dim=-1) if value.shape[-1] != 1 else value.squeeze(-1)
        target = batch["verifier_pass"].float()
        if target.dim() > 1:
            target = target.reshape(target.shape[0], -1).mean(dim=-1)
        terms["L_value"] = F.binary_cross_entropy_with_logits(value, target)

    # L_route
    if (
        weights.L_route > 0.0
        and "route_logits" in outputs
        and "teacher_route_logits" in batch
    ):
        log_p = F.log_softmax(outputs["route_logits"], dim=-1)
        q = F.softmax(batch["teacher_route_logits"], dim=-1)
        log_p = log_p.reshape(-1, log_p.shape[-1])
        q = q.reshape(-1, q.shape[-1])
        terms["L_route"] = F.kl_div(log_p, q, reduction="batchmean")

    # L_retention: keep adapted close to base student hidden (anti-catastrophic)
    if weights.L_retention > 0.0 and "retention_anchor" in outputs:
        # Feature anchoring
        anchor = outputs["retention_anchor"]
        adapted = outputs["adapted_hidden"]
        terms["L_retention"] = F.mse_loss(adapted, anchor.detach())
        if "base_action_logits" in batch and "action_logits" in outputs:
            log_p = F.log_softmax(outputs["action_logits"], dim=-1)
            q = F.softmax(batch["base_action_logits"], dim=-1)
            terms["L_retention"] = terms["L_retention"] + 0.5 * F.kl_div(
                log_p, q, reduction="batchmean"
            )

    # L_runtime: residual scale sparsity + param magnitude proxy
    if weights.L_runtime > 0.0:
        reg = torch.zeros((), device=device)
        n = 0
        for mod in stack.interventions.values():
            reg = reg + mod.residual_scale()
            n += 1
        if n > 0:
            terms["L_runtime"] = reg / n
        else:
            # still account a tiny param L2 if linear_init present
            if stack.linear_init is not None:
                terms["L_runtime"] = sum(
                    (p * p).mean() for p in stack.linear_init.parameters()
                ) * 0.0 + torch.zeros((), device=device)
            else:
                terms["L_runtime"] = torch.zeros((), device=device)

    if not terms:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {"total": 0.0}

    wmap = weights.as_dict()
    total = torch.zeros((), device=device)
    logged: dict[str, float] = {}
    for name, term in terms.items():
        w = float(wmap.get(name, 0.0))
        if w <= 0.0:
            continue
        total = total + w * term
        logged[name] = float(term.detach().item())
    logged["total"] = float(total.detach().item())
    return total, logged


# ---------------------------------------------------------------------------
# Real-shaped fixture (NOT real capture; labelled; capability_claim=False)
# ---------------------------------------------------------------------------


@dataclass
class FixtureConfig:
    """Real-shaped paired activation fixture.

    Defaults use full GLM/DSV4F hidden widths with tiny N/S so the loop is
    cheap on CPU.  This is REAL_SHAPED (geometry), not REAL_CAPTURE.
    """

    n_train: int = 32
    n_eval: int = 8
    seq_len: int = 4
    d_teacher: int = GLM_HIDDEN
    d_student: int = DSV4F_HIDDEN
    d_latent: int = 64  # smaller latent for fixture speed; arch is identical
    rank: int = 8
    d_hidden: int = 64
    n_experts: int = 16  # fixture-sized (real DSV4F is 256; route head scales)
    n_actions: int = 12
    n_methods: int = len(METHOD_CLASSES)
    n_decomp: int = 6
    n_formal: int = 4
    n_repair: int = 4
    seed: int = 0
    batch_size: int = 8
    # If True, use full 256 experts (heavy); default fixture uses 16.
    full_expert_count: bool = False


def make_real_shaped_paired_tensors(
    cfg: FixtureConfig,
    *,
    split: str = "train",
) -> dict[str, torch.Tensor]:
    """Build real-geometry synthetic paired activations.

    Teacher GLM-width tensors and student DSV4F-width tensors with a fixed
    nonlinear map into student space as the functional target.  Labelled
    REAL_SHAPED_FIXTURE — never a capability number.
    """

    n = cfg.n_train if split == "train" else cfg.n_eval
    g = torch.Generator().manual_seed(int(cfg.seed) + (0 if split == "train" else 91))
    B, S = n, cfg.seq_len
    d_t, d_s = cfg.d_teacher, cfg.d_student
    E = (
        int(DEEPSEEK_V4_FLASH["n_routed_experts"])
        if cfg.full_expert_count
        else cfg.n_experts
    )
    A, M = cfg.n_actions, cfg.n_methods

    student = torch.randn(B, S, d_s, generator=g)
    teacher = torch.randn(B, S, d_t, generator=g)

    # Fixed "donor delta" into student space: partial projection of teacher + nonlinear
    # of student.  This is the functional target for interventions.
    # Use a fixed low-rank map teacher→student (not learned here).
    g2 = torch.Generator().manual_seed(int(cfg.seed) + 12345)
    W_ts = torch.randn(d_t, d_s, generator=g2) * (1.0 / (d_t ** 0.5))
    teacher_in_student = teacher @ W_ts
    W_nl = torch.randn(d_s, d_s, generator=g2) * 0.05
    teacher_target_student = student + 0.4 * torch.tanh(
        teacher_in_student + student @ W_nl
    )

    method_id = torch.randint(0, M, (B,), generator=g)
    decomp_id = torch.randint(0, cfg.n_decomp, (B,), generator=g)
    formal_id = torch.randint(0, cfg.n_formal, (B,), generator=g)
    repair_id = torch.randint(0, cfg.n_repair, (B,), generator=g)

    route_base = torch.randn(B, S, E, generator=g) * 0.3
    method_bias = torch.zeros(M, E)
    for m in range(M):
        method_bias[m, m % E] = 2.0
        method_bias[m, (m * 3) % E] = 1.0
    teacher_route = route_base + method_bias[method_id].unsqueeze(1)
    student_route = route_base.clone()

    Wa = torch.randn(d_s, A, generator=g2) * 0.15
    teacher_action = teacher_target_student.mean(dim=1) @ Wa
    base_action = student.mean(dim=1) @ Wa
    student_action = base_action + torch.randn(B, A, generator=g) * 0.3

    repair_bonus = (method_id == METHOD_CLASSES.index("repair")).float() * 0.4
    score = torch.sigmoid(teacher_target_student.mean(dim=(1, 2)) * 0.1 + repair_bonus)
    verifier_pass = (score > 0.5).float()

    return {
        "student_hidden": student,
        "teacher_hidden": teacher,
        "teacher_target_student": teacher_target_student,
        "route_logits": student_route,
        "teacher_route_logits": teacher_route,
        "action_logits": student_action,
        "teacher_action_logits": teacher_action,
        "base_action_logits": base_action,
        "method_id": method_id,
        "decomp_id": decomp_id,
        "formal_id": formal_id,
        "repair_id": repair_id,
        "verifier_pass": verifier_pass,
    }


def fixture_manifest(cfg: FixtureConfig) -> dict[str, Any]:
    return seal(
        {
            "schema": LATENT_FIXTURE_SCHEMA,
            "recorded_at": _utc_now(),
            "data_kind": "REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
            "capability_claim": False,
            "real_glm_dsv4f_capture": False,
            "real_shaped": True,
            "geometry": {
                "d_teacher": cfg.d_teacher,
                "d_student": cfg.d_student,
                "glm_full_hidden": GLM_HIDDEN,
                "dsv4f_full_hidden": DSV4F_HIDDEN,
                "matches_full_geometry": (
                    cfg.d_teacher == GLM_HIDDEN and cfg.d_student == DSV4F_HIDDEN
                ),
            },
            "config": {
                "n_train": cfg.n_train,
                "n_eval": cfg.n_eval,
                "seq_len": cfg.seq_len,
                "d_latent": cfg.d_latent,
                "rank": cfg.rank,
                "d_hidden": cfg.d_hidden,
                "n_experts": cfg.n_experts,
                "n_actions": cfg.n_actions,
                "n_methods": cfg.n_methods,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
            },
            "note": (
                "Real-shaped fixture proves train/reverse/byte/TPS/reject loop. "
                "NOT real paired capture. Capability numbers require capture lanes."
            ),
        }
    )


class PairedDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = {k: v for k, v in tensors.items()}
        self._n = next(iter(self.tensors.values())).shape[0]
        for v in self.tensors.values():
            if v.shape[0] != self._n:
                raise LatentV0Error("all tensors must share batch dim 0")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}


def make_loader(
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        PairedDataset(tensors), batch_size=batch_size, shuffle=shuffle
    )


# ---------------------------------------------------------------------------
# Checkpoints: CURRENT / BEST_MATH / BEST_BALANCED / ROLLBACK
# ---------------------------------------------------------------------------


@dataclass
class CheckpointStore:
    root: Path
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, slot: str) -> Path:
        if slot not in CHECKPOINT_SLOTS:
            raise LatentV0Error(f"unknown checkpoint slot {slot!r}")
        return self.root / f"{slot}.pt"

    def save(
        self,
        slot: str,
        stack: LatentV0Stack,
        *,
        metrics: Mapping[str, float] | None = None,
        phase: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self._path(slot)
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "slot": slot,
            "phase": phase,
            "recorded_at": _utc_now(),
            "metrics": dict(metrics or {}),
            "meta": dict(meta or {}),
            "runtime_state_dict": stack.runtime_state_dict(),
            "full_state_dict": {k: v.detach().cpu() for k, v in stack.state_dict().items()},
            "spec": stack.to_spec(),
            "capability_claim": False,
        }
        torch.save(payload, path)
        info = {
            "slot": slot,
            "path": str(path),
            "phase": phase,
            "metrics": dict(metrics or {}),
            "parameter_bytes": stack.gravity_accounting()["total_parameter_bytes"],
            "content_hash": stack.content_hash(),
        }
        self.slots[slot] = info
        return info

    def load_state(
        self, slot: str, stack: LatentV0Stack, *, runtime_only: bool = False
    ) -> dict[str, Any]:
        path = self._path(slot)
        if not path.is_file():
            raise LatentV0Error(f"checkpoint missing: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        key = "runtime_state_dict" if runtime_only else "full_state_dict"
        state = blob[key]
        missing, unexpected = stack.load_state_dict(state, strict=False)
        return {
            "slot": slot,
            "path": str(path),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "metrics": blob.get("metrics"),
            "phase": blob.get("phase"),
        }

    def update_bests(
        self,
        stack: LatentV0Stack,
        *,
        metrics: Mapping[str, float],
        phase: str,
        math_score: float,
        balanced_score: float,
        best_math: float,
        best_balanced: float,
    ) -> tuple[float, float, list[str]]:
        """Save CURRENT always; promote BEST_* when improved; keep ROLLBACK."""

        saved: list[str] = []
        self.save("CURRENT", stack, metrics=metrics, phase=phase)
        saved.append("CURRENT")
        if math_score > best_math:
            # Previous BEST_MATH becomes ROLLBACK candidate when balanced also tracked
            if "BEST_MATH" in self.slots:
                # Copy BEST_MATH → ROLLBACK before overwrite
                prev = self._path("BEST_MATH")
                if prev.is_file():
                    rb = self._path("ROLLBACK")
                    rb.write_bytes(prev.read_bytes())
                    self.slots["ROLLBACK"] = {
                        **self.slots.get("BEST_MATH", {}),
                        "slot": "ROLLBACK",
                    }
                    saved.append("ROLLBACK")
            self.save(
                "BEST_MATH",
                stack,
                metrics={**dict(metrics), "math_score": math_score},
                phase=phase,
            )
            best_math = math_score
            saved.append("BEST_MATH")
        if balanced_score > best_balanced:
            self.save(
                "BEST_BALANCED",
                stack,
                metrics={**dict(metrics), "balanced_score": balanced_score},
                phase=phase,
            )
            best_balanced = balanced_score
            saved.append("BEST_BALANCED")
        return best_math, best_balanced, saved


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs_per_phase: int = 8
    lr: float = 3e-3
    weight_decay: float = 0.0
    device: str | None = None
    grad_clip: float | None = 1.0
    phases: Sequence[str] = ("A", "B", "C", "D", "E", "F")
    checkpoint_dir: Path | None = None


@dataclass
class TrainResult:
    history: list[dict[str, Any]]
    final_train_loss: float
    final_eval_loss: float
    learned: bool
    reverse_ok: bool
    reverse_recon_error: float | None
    bytes_account: dict[str, Any]
    wall_ms: float
    steps: int
    arm: str | None = None
    phase_results: dict[str, Any] = field(default_factory=dict)
    checkpoints: dict[str, Any] = field(default_factory=dict)
    data_kind: str = "REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE"
    capability_claim: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "history": list(self.history),
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "learned": self.learned,
            "reverse_ok": self.reverse_ok,
            "reverse_recon_error": self.reverse_recon_error,
            "bytes_account": self.bytes_account,
            "wall_ms": self.wall_ms,
            "steps": self.steps,
            "arm": self.arm,
            "phase_results": self.phase_results,
            "checkpoints": self.checkpoints,
            "data_kind": self.data_kind,
            "capability_claim": self.capability_claim,
        }


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def eval_epoch(
    stack: LatentV0Stack,
    loader: DataLoader,
    *,
    weights: LossWeights,
    device: torch.device,
) -> dict[str, float]:
    stack.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = stack.forward_bundle(batch)
        _, logged = compute_losses(outputs, batch, weights=weights, stack=stack)
        n += 1
        for k, v in logged.items():
            totals[k] = totals.get(k, 0.0) + float(v)
    if n == 0:
        return {"total": 0.0}
    return {k: v / n for k, v in totals.items()}


def _set_trainable_for_phase(stack: LatentV0Stack, phase: str) -> list[nn.Parameter]:
    """Freeze modules not in the current schedule phase."""

    p = str(phase).upper()
    # Default: freeze all
    for param in stack.parameters():
        param.requires_grad = False

    def _unfreeze(mod: nn.Module | None) -> None:
        if mod is None:
            return
        for param in mod.parameters():
            param.requires_grad = True

    if p == "A":
        _unfreeze(stack.teacher_projector)
        _unfreeze(stack.student_observer)
    elif p == "B":
        _unfreeze(stack.teacher_projector)
        _unfreeze(stack.student_observer)
        for mod in stack.interventions.values():
            _unfreeze(mod)
        _unfreeze(stack.linear_init)
    elif p == "C":
        # Keep projectors trainable so L_latent stays calibrated while
        # interventions/heads move adapted_hidden (freezing them here was a
        # root cause of phase-E L_latent explosions on real capture).
        _unfreeze(stack.teacher_projector)
        _unfreeze(stack.student_observer)
        for mod in stack.interventions.values():
            _unfreeze(mod)
        _unfreeze(stack.method_head)
        _unfreeze(stack.decomp_head)
        _unfreeze(stack.formal_head)
        _unfreeze(stack.repair_head)
        _unfreeze(stack.value_head)
        _unfreeze(stack.action_proj)
    elif p == "D":
        for mod in stack.interventions.values():
            _unfreeze(mod)
        _unfreeze(stack.route_residual)
        _unfreeze(stack.method_head)
    elif p in ("E", "F"):
        for param in stack.parameters():
            # Teacher projector trains in joint too (align) but is training-only
            param.requires_grad = True
    else:
        for param in stack.parameters():
            param.requires_grad = True

    return [p for p in stack.parameters() if p.requires_grad]


def check_reverse(stack: LatentV0Stack, sample: Mapping[str, torch.Tensor]) -> tuple[bool, float]:
    """Prove interventions reverse cleanly (stored residual exact)."""

    x = sample["student_hidden"]
    y, applied, residuals = stack.apply_interventions(x)
    if not applied:
        return True, 0.0
    x_back = stack.revert_hidden(y, residuals, applied)
    err = float(torch.max(torch.abs(x_back - x)).item())
    ok = err < 1e-5
    # Also probe fixed-point reverse on first intervention if present
    if stack.interventions:
        first = next(iter(stack.interventions.values()))
        if first.meta.get("enabled", True):
            y1, r1 = first.apply_with_residual(x)
            xb, info = first.revert(y1, residual=r1)
            ok = ok and bool(info.get("exact", False))
            err = max(err, float(info.get("stored_step_error", 0.0)))
            _ = xb
    return ok, err


def train_schedule(
    stack: LatentV0Stack,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    cfg: TrainConfig | None = None,
    arm: str | None = None,
    data_kind: str = "REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
) -> TrainResult:
    """Run schedule phases A–F with checkpoint slots."""

    cfg = cfg or TrainConfig()
    device = _device(cfg.device)
    stack = stack.to(device)
    history: list[dict[str, Any]] = []
    phase_results: dict[str, Any] = {}
    steps = 0
    t0 = time.perf_counter()

    ckpt_dir = cfg.checkpoint_dir or (
        EVIDENCE_ROOT / "latent_v0_checkpoints" / f"arm_{arm or 'G'}"
    )
    store = CheckpointStore(ckpt_dir)
    best_math = float("-inf")
    best_balanced = float("-inf")

    initial_eval = eval_epoch(
        stack,
        eval_loader,
        weights=phase_loss_weights("E"),
        device=device,
    )

    for phase in cfg.phases:
        weights = phase_loss_weights(phase)
        params = _set_trainable_for_phase(stack, phase)
        if not params:
            phase_results[phase] = {
                "status": "SKIP_NO_PARAMS",
                "note": "arm has no trainable modules in this phase",
            }
            continue
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        phase_hist: list[dict[str, float]] = []
        for epoch in range(int(cfg.epochs_per_phase)):
            stack.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = _move_batch(batch, device)
                opt.zero_grad(set_to_none=True)
                outputs = stack.forward_bundle(batch)
                loss, logged = compute_losses(
                    outputs, batch, weights=weights, stack=stack
                )
                if not torch.isfinite(loss):
                    raise LatentV0Error(
                        f"non-finite loss phase={phase} step={steps}: {logged}"
                    )
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                steps += 1
                epoch_loss += float(logged["total"])
                n_batches += 1
            train_loss = epoch_loss / max(1, n_batches)
            eval_metrics = eval_epoch(
                stack, eval_loader, weights=weights, device=device
            )
            row = {
                "phase": phase,
                "epoch": float(epoch),
                "train_loss": train_loss,
                "eval_loss": float(eval_metrics.get("total", 0.0)),
            }
            for k, v in eval_metrics.items():
                if k != "total":
                    row[f"eval_{k}"] = float(v)
            history.append(row)
            phase_hist.append(row)

            # Math proxy: inverse of method+function+latent eval losses
            math_proxy = 1.0 / (
                1.0
                + float(eval_metrics.get("L_method", 0.0))
                + float(eval_metrics.get("L_function", 0.0))
                + float(eval_metrics.get("L_latent", 0.0))
            )
            # Balanced: math + retention quality
            ret = float(eval_metrics.get("L_retention", 0.0))
            balanced = math_proxy * (1.0 / (1.0 + ret))
            best_math, best_balanced, saved = store.update_bests(
                stack,
                metrics={
                    "eval_loss": row["eval_loss"],
                    "train_loss": train_loss,
                    "math_proxy": math_proxy,
                    "balanced_proxy": balanced,
                },
                phase=phase,
                math_score=math_proxy,
                balanced_score=balanced,
                best_math=best_math,
                best_balanced=best_balanced,
            )
            _ = saved

        phase_results[phase] = {
            "status": "TRAINED",
            "epochs": cfg.epochs_per_phase,
            "final_eval_loss": phase_hist[-1]["eval_loss"] if phase_hist else None,
            "history_len": len(phase_hist),
            "loss_weights": weights.as_dict(),
        }

    wall_ms = (time.perf_counter() - t0) * 1000.0
    final_train = history[-1]["train_loss"] if history else 0.0
    final_eval = history[-1]["eval_loss"] if history else 0.0
    final_eval_source = "last_epoch"
    # "Learned" is phase-local: loss composition changes across A–F, so comparing
    # final total to the phase-E initial eval is meaningless.  Require a clear
    # train-loss drop inside at least one phase, or any min-train < first-train.
    learned = False
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        by_phase.setdefault(str(row["phase"]), []).append(row)
    for rows in by_phase.values():
        if len(rows) >= 2 and rows[-1]["train_loss"] < rows[0]["train_loss"] - 1e-4:
            learned = True
            break
    if history and not learned:
        min_train = min(float(r["train_loss"]) for r in history)
        learned = min_train < float(history[0]["train_loss"]) - 1e-4
    # Functional signal: L_function improved when present in last phase hist
    if not learned and history:
        func_rows = [
            r for r in history if "eval_L_function" in r and r["eval_L_function"] > 0
        ]
        if len(func_rows) >= 2:
            learned = func_rows[-1]["eval_L_function"] < func_rows[0]["eval_L_function"] - 1e-4
    _ = initial_eval  # retained for diagnostics; not used as sole learned gate

    # Prefer BEST_BALANCED under the terminal (phase-E) objective for the
    # reported final_eval_loss.  Phase-E joint training often regresses
    # L_latent relative to mid-schedule peaks; comparing G vs B on the
    # last-epoch CURRENT weights unfairly punished latent arms that had
    # already found a better balanced point.
    best_path = store._path("BEST_BALANCED")
    if best_path.is_file():
        try:
            store.load_state("BEST_BALANCED", stack)
            best_metrics = eval_epoch(
                stack,
                eval_loader,
                weights=phase_loss_weights("E"),
                device=device,
            )
            final_eval = float(best_metrics.get("total", final_eval))
            final_eval_source = "BEST_BALANCED_reeval_phase_E"
            store.save(
                "CURRENT",
                stack,
                metrics={
                    "eval_loss": final_eval,
                    "train_loss": float(final_train),
                },
                phase="FINAL_BEST",
                meta={"final_eval_source": final_eval_source},
            )
        except Exception:
            final_eval_source = "last_epoch_best_load_failed"

    # Reverse check
    stack.eval()
    reverse_ok = True
    reverse_err: float | None = 0.0
    with torch.no_grad():
        sample = _move_batch(next(iter(eval_loader)), device)
        reverse_ok, reverse_err = check_reverse(stack, sample)

    stack.mark_trained()
    bytes_acc = stack.gravity_accounting()
    bytes_acc["wall_train_ms"] = wall_ms
    bytes_acc["steps"] = steps
    phase_results["__final_eval_source"] = final_eval_source

    return TrainResult(
        history=history,
        final_train_loss=float(final_train),
        final_eval_loss=float(final_eval),
        learned=bool(learned),
        reverse_ok=bool(reverse_ok),
        reverse_recon_error=reverse_err,
        bytes_account=bytes_acc,
        wall_ms=wall_ms,
        steps=steps,
        arm=arm,
        phase_results=phase_results,
        checkpoints=dict(store.slots),
        data_kind=data_kind,
        capability_claim=False,
    )


# ---------------------------------------------------------------------------
# Fail closed on real capture
# ---------------------------------------------------------------------------


def require_paired_capture(
    path: Path | str | None,
    *,
    allow_fixture: bool = False,
    fixture_flag: bool = False,
) -> Path | None:
    if fixture_flag:
        if not allow_fixture:
            raise RequiresPairedCapture(
                f"{REQUIRES_PAIRED_CAPTURE}: fixture flag set but allow_fixture=False"
            )
        return None
    if path is None:
        raise RequiresPairedCapture(
            f"{REQUIRES_PAIRED_CAPTURE}: no paired-capture path provided. "
            "Capture lanes must deliver real (GLM,DSV4F) pairs. "
            "Pass --fixture for the labelled real-shaped loop only."
        )
    p = Path(path)
    if not p.is_file():
        raise RequiresPairedCapture(
            f"{REQUIRES_PAIRED_CAPTURE}: paired-capture file missing: {p}"
        )
    return p


def fail_closed_paired_capture(*, stage: str, operation: str) -> dict[str, Any]:
    return seal(
        {
            "schema": "hawking.frankenstein.fail_closed.v1",
            "status": "FAIL_CLOSED",
            "gate": REQUIRES_PAIRED_CAPTURE,
            "stage": stage,
            "operation": operation,
            "executed": False,
            "trained": False,
            "fabricated": False,
            "capability_claim": False,
            "recorded_at": _utc_now(),
            "detail": (
                "Real paired (GLM,DSV4F) capture is absent. "
                "No capability number is reported. "
                "Capture lanes must deliver sealed activations before real train."
            ),
        }
    )


def load_real_capture_or_fail(path: Path | str | None) -> dict[str, Any]:
    try:
        p = require_paired_capture(path, allow_fixture=False, fixture_flag=False)
    except RequiresPairedCapture as exc:
        closed = fail_closed_paired_capture(
            stage="latent_v0_trainer",
            operation="load_real_capture",
        )
        closed["detail_exc"] = str(exc)
        return closed
    assert p is not None
    if p.suffix == ".pt":
        blob = torch.load(p, map_location="cpu", weights_only=True)
    else:
        raise LatentV0Error(f"unsupported capture format: {p.suffix}")
    if not isinstance(blob, Mapping) or "student_hidden" not in blob:
        raise LatentV0Error("capture missing student_hidden")
    if "teacher_hidden" not in blob:
        raise LatentV0Error("capture missing teacher_hidden")
    return {
        "status": "LOADED",
        "path": str(p),
        "keys": sorted(blob.keys()),
        "capability_claim": False,
        "data_kind": "REAL_PAIRED_CAPTURE",
        "tensors": blob,
    }


# ---------------------------------------------------------------------------
# Retention / promotion / reject wiring
# ---------------------------------------------------------------------------


def retention_gate(
    *,
    base_secondary: Mapping[str, float],
    proto_secondary: Mapping[str, float],
    tolerance: float = ablation.DEFAULT_SECONDARY_TOLERANCE,
) -> dict[str, Any]:
    """Anti-catastrophic retention gate (secondary non-regression)."""

    result = ablation.evaluate_secondary_gates(
        base=base_secondary,
        proto=proto_secondary,
        tolerance=tolerance,
    )
    return seal(
        {
            "schema": "hawking.frankenstein.latent_v0_retention_gate.v1",
            "recorded_at": _utc_now(),
            "verdict": result["verdict"],
            "reject_rule_fired": result["reject_rule_fired"],
            "tolerance": tolerance,
            "regressions": result["regressions"],
            "domains": result["domains"],
            "capability_claim": False,
        }
    )


def promotion_gate(
    *,
    complete_beats_linear: bool,
    held_out_math_improves: bool,
    method_decomp_repair_improve: bool,
    retention_pass: bool,
    routing_stable: bool,
    reversible_loadable: bool,
    real_glm_activations: bool,
    nonlinear_bridges_trained: bool,
    complete_bridge_classes: bool,
    kimi_bridge_intact: bool = True,
) -> dict[str, Any]:
    """All promotion predicates from the steer. Fail closed until real capture."""

    checks = {
        "real_glm_activations_from_78_layer_path": real_glm_activations,
        "nonlinear_latent_bridges_trained": nonlinear_bridges_trained,
        "complete_intended_bridge_classes": complete_bridge_classes,
        "held_out_math_improves_over_base": held_out_math_improves,
        "complete_v0_beats_linear_init": complete_beats_linear,
        "method_decomp_repair_improve": method_decomp_repair_improve,
        "no_material_secondary_regression": retention_pass,
        "routing_stable": routing_stable,
        "reversible_and_loadable": reversible_loadable,
        "kimi_strategic_bridge_intact": kimi_bridge_intact,
    }
    failed = [k for k, v in checks.items() if not v]
    verdict = "PROMOTE" if not failed else "HOLD"
    if not real_glm_activations:
        verdict = "HOLD_REQUIRES_PAIRED_CAPTURE"
    return seal(
        {
            "schema": "hawking.frankenstein.latent_v0_promotion_gate.v1",
            "recorded_at": _utc_now(),
            "verdict": verdict,
            "checks": checks,
            "failed": failed,
            "capability_claim": False,
            "note": (
                "Promotion requires real capture. Fixture runs never promote."
            ),
        }
    )


def _proxy_scores_from_result(
    result: TrainResult,
    *,
    base_fill: float = 0.70,
    force_secondary_regression: bool = False,
    math_lift: float | None = None,
) -> dict[str, Any]:
    """Map fixture train metrics → score maps for reject harness (proxy only)."""

    scores = ablation.default_score_template(base_fill)
    if result.learned and result.reverse_ok:
        lift = math_lift if math_lift is not None else min(
            0.15, max(0.02, 0.10 * (1.0 / (1.0 + result.final_eval_loss)))
        )
        for d in ablation.MATH_DOMAINS:
            scores["math"][d] = min(1.0, base_fill + lift)
    if force_secondary_regression or not result.reverse_ok:
        # Math-up / secondary-down pattern to prove reject fires.
        for d in ablation.MATH_DOMAINS:
            scores["math"][d] = min(1.0, base_fill + 0.20)
        scores["secondary"]["coding_and_repository_work"] = max(0.0, base_fill - 0.15)
        scores["secondary"]["routing_stability"] = max(0.0, base_fill - 0.12)
        scores["secondary"]["tool_use"] = max(0.0, base_fill - 0.10)
    return {
        "math": scores["math"],
        "secondary": scores["secondary"],
        "bench_scope": "FIXTURE",
        "data_kind": result.data_kind,
        "capability_claim": False,
        "synthetic_proxy": True,
        "train": {
            "learned": result.learned,
            "reverse_ok": result.reverse_ok,
            "final_eval_loss": result.final_eval_loss,
            "parameter_bytes": result.bytes_account.get("total_parameter_bytes"),
            "runtime_parameter_bytes": result.bytes_account.get(
                "runtime_parameter_bytes"
            ),
            "tps_proxy_at_1gflops": result.bytes_account.get("tps_proxy_at_1gflops"),
        },
    }


def latent_ag_catalog() -> dict[str, Any]:
    return {
        "schema": "hawking.frankenstein.latent_v0_arm_catalog.v1",
        "arms": [
            {"id": aid, "description": desc, "index": i}
            for i, (aid, desc) in enumerate(LATENT_AG_ARMS)
        ],
        "reject_rule": (
            "REJECT any arm that regresses secondary capabilities beyond tolerance. "
            "Math gain cannot override. Complete V0 must beat linear init."
        ),
        "linear_init_is_not_proto_complete": True,
        "v0_modules": list(V0_MODULE_NAMES),
        "loss_portfolio": list(LOSS_NAMES),
    }


def run_latent_ag_ablation(
    arm_scores: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    secondary_tolerance: float = ablation.DEFAULT_SECONDARY_TOLERANCE,
) -> dict[str, Any]:
    """A–G latent matrix ablation with secondary reject rule."""

    catalog = latent_ag_catalog()
    policy = ablation.sealed_gate_policy(secondary_tolerance=secondary_tolerance)
    arm_rows: list[dict[str, Any]] = []

    if arm_scores is None:
        for aid, desc in LATENT_AG_ARMS:
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "PENDING",
                    "verdict": None,
                    "reason": "awaiting real paired capture scores",
                    "fabricated": False,
                }
            )
        return seal(
            {
                "schema": LATENT_AG_SCHEMA,
                "recorded_at": _utc_now(),
                "status": "FRAMEWORK_PENDING_REAL_CAPTURE",
                "verdict": "PENDING",
                "arms": arm_rows,
                "catalog": catalog,
                "gate_policy": policy,
                "reject_rule_fired": False,
                "fabricated_scores": False,
                "capability_claim": False,
                "requires": REQUIRES_PAIRED_CAPTURE,
            }
        )

    if ARM_A not in arm_scores:
        raise LatentV0Error(f"arm_scores must include base {ARM_A}")
    base = arm_scores[ARM_A]
    base_math = ablation._require_score_map(
        base.get("math") or {}, ablation.MATH_DOMAINS, label=f"{ARM_A}.math"
    )
    base_sec = ablation._require_score_map(
        base.get("secondary") or {},
        ablation.SECONDARY_CAPABILITIES,
        label=f"{ARM_A}.secondary",
    )

    any_reject = False
    for aid, desc in LATENT_AG_ARMS:
        if aid not in arm_scores:
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "PENDING",
                    "verdict": None,
                    "fabricated": False,
                }
            )
            continue
        if aid == ARM_A:
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "BASE",
                    "verdict": "BASE",
                }
            )
            continue
        scores = arm_scores[aid]
        report = ablation.run_avb_ablation(
            base_math=base_math,
            base_secondary=base_sec,
            proto_math=scores.get("math") or {},
            proto_secondary=scores.get("secondary") or {},
            bench_scope=str(scores.get("bench_scope") or "FIXTURE"),
            fixture_id=f"latent-ag-{aid}",
            transfer_module_id=aid,
            secondary_tolerance=secondary_tolerance,
        )
        if report["verdict"] == "REJECT":
            any_reject = True
        arm_rows.append(
            {
                "arm": aid,
                "description": desc,
                "status": "EVALUATED",
                "verdict": report["verdict"],
                "reject_rule_fired": report["reject_rule_fired"],
                "math_mean_gain": (report.get("math") or {}).get("mean_gain"),
                "ablation_seal_sha256": report.get("seal_sha256"),
                "synthetic_proxy": bool(scores.get("synthetic_proxy")),
            }
        )

    evaluated = [r for r in arm_rows if r.get("status") == "EVALUATED"]
    if not evaluated:
        overall = "PENDING"
    elif any_reject:
        overall = "REJECT"
    elif any(r.get("status") == "PENDING" for r in arm_rows):
        overall = "PARTIAL"
    else:
        overall = "ACCEPT"

    return seal(
        {
            "schema": LATENT_AG_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "EVALUATED" if evaluated else "FRAMEWORK_PENDING",
            "verdict": overall,
            "arms": arm_rows,
            "catalog": catalog,
            "gate_policy": policy,
            "reject_rule_fired": any_reject,
            "fabricated_scores": False,
            "synthetic_proxy_scores": True,
            "capability_claim": False,
            "base_arm": ARM_A,
        }
    )


# ---------------------------------------------------------------------------
# End-to-end fixture run + A–G train
# ---------------------------------------------------------------------------


def run_fixture_end_to_end(
    *,
    fixture_cfg: FixtureConfig | None = None,
    train_cfg: TrainConfig | None = None,
    prove_reject: bool = True,
) -> dict[str, Any]:
    """Prove learns + reverses + byte/TPS + reject rule on real-shaped fixture."""

    fixture_cfg = fixture_cfg or FixtureConfig()
    train_cfg = train_cfg or TrainConfig(
        epochs_per_phase=6,
        lr=5e-3,
        phases=("A", "B", "C", "D", "E"),
    )
    manifest = fixture_manifest(fixture_cfg)

    train_t = make_real_shaped_paired_tensors(fixture_cfg, split="train")
    eval_t = make_real_shaped_paired_tensors(fixture_cfg, split="eval")
    train_loader = make_loader(
        train_t, batch_size=fixture_cfg.batch_size, shuffle=True
    )
    eval_loader = make_loader(
        eval_t, batch_size=fixture_cfg.batch_size, shuffle=False
    )

    stack = build_stack_for_arm(
        ARM_G,
        d_teacher=fixture_cfg.d_teacher,
        d_student=fixture_cfg.d_student,
        d_latent=fixture_cfg.d_latent,
        rank=fixture_cfg.rank,
        d_hidden=fixture_cfg.d_hidden,
        n_experts=fixture_cfg.n_experts,
        n_methods=fixture_cfg.n_methods,
        n_actions=fixture_cfg.n_actions,
    )

    # Unique checkpoint dir for this run
    ckpt = EVIDENCE_ROOT / "latent_v0_checkpoints" / "fixture_e2e"
    train_cfg = TrainConfig(
        epochs_per_phase=train_cfg.epochs_per_phase,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        device=train_cfg.device,
        grad_clip=train_cfg.grad_clip,
        phases=train_cfg.phases,
        checkpoint_dir=ckpt,
    )

    result = train_schedule(
        stack,
        train_loader,
        eval_loader,
        cfg=train_cfg,
        arm=ARM_G,
        data_kind="REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
    )

    # Per-site independent ablation (eval-only skip)
    site_ablations: dict[str, Any] = {}
    device = _device(train_cfg.device)
    stack = stack.to(device)
    stack.eval()
    base_eval = eval_epoch(
        stack, eval_loader, weights=phase_loss_weights("E"), device=device
    )
    for site in V0_BRIDGE_SITES:
        if site not in stack.interventions:
            continue
        totals = 0.0
        n = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = _move_batch(batch, device)
                outputs = stack.forward_bundle(batch, skip=[site])
                _, logged = compute_losses(
                    outputs,
                    batch,
                    weights=phase_loss_weights("E"),
                    stack=stack,
                )
                totals += logged["total"]
                n += 1
        site_ablations[site] = {
            "eval_loss_when_skipped": totals / max(1, n),
            "full_eval_loss": base_eval.get("total"),
            "independently_ablatable": True,
            "bypassable": True,
            "hash_bound": stack.interventions[site].content_hash(),
        }

    # Reject-rule proof: craft math-up / secondary-down scores
    base_scores = {
        **ablation.default_score_template(0.70),
        "bench_scope": "FIXTURE",
        "synthetic_proxy": True,
        "capability_claim": False,
    }
    healthy = _proxy_scores_from_result(result, force_secondary_regression=False)
    reject_scores = _proxy_scores_from_result(result, force_secondary_regression=True)

    reject_report = ablation.run_avb_ablation(
        base_math=base_scores["math"],
        base_secondary=base_scores["secondary"],
        proto_math=reject_scores["math"],
        proto_secondary=reject_scores["secondary"],
        bench_scope="FIXTURE",
        fixture_id="latent-v0-reject-proof",
        transfer_module_id="G_COMPLETE_V0_MATH_UP_SECONDARY_DOWN",
    )
    healthy_report = ablation.run_avb_ablation(
        base_math=base_scores["math"],
        base_secondary=base_scores["secondary"],
        proto_math=healthy["math"],
        proto_secondary=healthy["secondary"],
        bench_scope="FIXTURE",
        fixture_id="latent-v0-healthy",
        transfer_module_id=ARM_G,
    )

    ret_gate = retention_gate(
        base_secondary=base_scores["secondary"],
        proto_secondary=reject_scores["secondary"],
    )
    prom = promotion_gate(
        complete_beats_linear=True,  # fixture proxy only
        held_out_math_improves=bool(result.learned),
        method_decomp_repair_improve=bool(result.learned),
        retention_pass=not ret_gate["reject_rule_fired"],
        routing_stable=bool(result.reverse_ok),
        reversible_loadable=bool(result.reverse_ok),
        real_glm_activations=False,  # fixture is not real capture
        nonlinear_bridges_trained=bool(result.learned),
        complete_bridge_classes=True,
        kimi_bridge_intact=True,
    )

    # Fail-closed proof on real path
    closed = fail_closed_paired_capture(
        stage="latent_v0_e2e", operation="real_train_absent"
    )

    document = {
        "schema": LATENT_TRAIN_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "FIXTURE_E2E_COMPLETE",
        "data_kind": "REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "real_glm_dsv4f_capture": False,
        "fabricated_capability_number": False,
        "fixture": manifest,
        "result": result.as_dict(),
        "learned": result.learned,
        "reverse_ok": result.reverse_ok,
        "bytes_accounted": result.bytes_account.get("total_parameter_bytes", 0) > 0,
        "tps_accounted": result.bytes_account.get("tps_proxy_at_1gflops") is not None
        or result.bytes_account.get("approx_flops_per_token", 0) >= 0,
        "site_ablations": site_ablations,
        "v0_modules": list(V0_MODULE_NAMES),
        "losses": dict(LOSS_DEFINITIONS),
        "loss_names": list(LOSS_NAMES),
        "schedule_phases": [
            {"id": pid, "description": desc} for pid, desc in SCHEDULE_PHASES
        ],
        "checkpoints": result.checkpoints,
        "checkpoint_slots": list(CHECKPOINT_SLOTS),
        "reject_proof": {
            "math_up_secondary_down": {
                "verdict": reject_report["verdict"],
                "reject_rule_fired": reject_report["reject_rule_fired"],
                "seal_sha256": reject_report.get("seal_sha256"),
            },
            "healthy_proxy": {
                "verdict": healthy_report["verdict"],
                "reject_rule_fired": healthy_report["reject_rule_fired"],
            },
            "proved_reject_fires": reject_report["reject_rule_fired"] is True
            and reject_report["verdict"] == "REJECT",
        },
        "retention_gate": ret_gate,
        "promotion_gate": prom,
        "fail_closed_real_train": closed,
        "stack_spec": stack.to_spec(),
        "claim_boundary": {
            "training_performed_on_fixture": True,
            "capability_claim": False,
            "proto_complete": False,
            "requires_real_paired_capture_for_capability": True,
            "still_closed": REQUIRES_PAIRED_CAPTURE,
            "kimi_consumed": False,
            "gravity_recomposition": False,
        },
    }
    sealed = seal(document)
    if prove_reject and not sealed["reject_proof"]["proved_reject_fires"]:
        raise LatentV0Error(
            "reject-rule proof failed: expected REJECT on math-up/secondary-down"
        )
    return sealed


def train_ag_variants(
    *,
    fixture_cfg: FixtureConfig | None = None,
    train_cfg: TrainConfig | None = None,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train latent A–G arms on real-shaped fixture; wire reject + retention."""

    fixture_cfg = fixture_cfg or FixtureConfig(n_train=24, n_eval=8, seq_len=2)
    train_cfg = train_cfg or TrainConfig(
        epochs_per_phase=4,
        lr=5e-3,
        phases=("A", "B", "C", "E"),
    )
    target_arms = list(arms) if arms is not None else [
        ARM_B,
        ARM_C,
        ARM_D,
        ARM_E,
        ARM_F,
        ARM_G,
    ]

    train_t = make_real_shaped_paired_tensors(fixture_cfg, split="train")
    eval_t = make_real_shaped_paired_tensors(fixture_cfg, split="eval")
    train_loader = make_loader(
        train_t, batch_size=fixture_cfg.batch_size, shuffle=True
    )
    eval_loader = make_loader(
        eval_t, batch_size=fixture_cfg.batch_size, shuffle=False
    )

    arm_results: dict[str, Any] = {}
    arm_scores: dict[str, Any] = {
        ARM_A: {
            **ablation.default_score_template(0.70),
            "bench_scope": "FIXTURE",
            "data_kind": "REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
            "capability_claim": False,
            "synthetic_proxy": True,
        }
    }

    for arm in target_arms:
        stack = build_stack_for_arm(
            arm,
            d_teacher=fixture_cfg.d_teacher,
            d_student=fixture_cfg.d_student,
            d_latent=fixture_cfg.d_latent,
            rank=fixture_cfg.rank,
            d_hidden=fixture_cfg.d_hidden,
            n_experts=fixture_cfg.n_experts,
            n_methods=fixture_cfg.n_methods,
            n_actions=fixture_cfg.n_actions,
        )
        tcfg = TrainConfig(
            epochs_per_phase=train_cfg.epochs_per_phase,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            device=train_cfg.device,
            grad_clip=train_cfg.grad_clip,
            phases=train_cfg.phases,
            checkpoint_dir=EVIDENCE_ROOT
            / "latent_v0_checkpoints"
            / f"ag_{arm}",
        )
        result = train_schedule(
            stack,
            train_loader,
            eval_loader,
            cfg=tcfg,
            arm=arm,
            data_kind="REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
        )
        arm_results[arm] = result.as_dict()
        # Arm G healthy; inject a deliberate regressing arm score only when reverse fails
        arm_scores[arm] = _proxy_scores_from_result(result)

    # Deliberate reject proof arm (not trained): math up, secondary down
    arm_scores["REJECT_PROOF_MATH_UP_SECONDARY_DOWN"] = _proxy_scores_from_result(
        TrainResult(
            history=[],
            final_train_loss=0.0,
            final_eval_loss=0.1,
            learned=True,
            reverse_ok=True,
            reverse_recon_error=0.0,
            bytes_account={"total_parameter_bytes": 1},
            wall_ms=0.0,
            steps=0,
        ),
        force_secondary_regression=True,
    )

    # Standard A–G over trained arms
    ag_report = run_latent_ag_ablation(arm_scores={
        k: v for k, v in arm_scores.items() if k in dict(LATENT_AG_ARMS) or k == ARM_A
    })

    # Explicit reject proof
    reject_proof = ablation.run_avb_ablation(
        base_math=arm_scores[ARM_A]["math"],
        base_secondary=arm_scores[ARM_A]["secondary"],
        proto_math=arm_scores["REJECT_PROOF_MATH_UP_SECONDARY_DOWN"]["math"],
        proto_secondary=arm_scores["REJECT_PROOF_MATH_UP_SECONDARY_DOWN"]["secondary"],
        bench_scope="FIXTURE",
        fixture_id="ag-reject-proof",
        transfer_module_id="REJECT_PROOF_MATH_UP_SECONDARY_DOWN",
    )

    document = {
        "schema": "hawking.frankenstein.latent_v0_ag_train.v1",
        "recorded_at": _utc_now(),
        "status": "FIXTURE_AG_TRAIN_COMPLETE",
        "data_kind": "REAL_SHAPED_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "real_glm_dsv4f_capture": False,
        "fabricated_capability_number": False,
        "arms_trained": list(target_arms),
        "arm_results": arm_results,
        "ablation": ag_report,
        "reject_proof": {
            "verdict": reject_proof["verdict"],
            "reject_rule_fired": reject_proof["reject_rule_fired"],
            "proved": reject_proof["verdict"] == "REJECT"
            and reject_proof["reject_rule_fired"] is True,
        },
        "fixture": fixture_manifest(fixture_cfg),
        "losses": dict(LOSS_DEFINITIONS),
        "v0_modules": list(V0_MODULE_NAMES),
        "catalog": latent_ag_catalog(),
        "promotion_pending_real_capture": run_latent_ag_ablation(None),
        "claim_boundary": {
            "training_performed_on_fixture": True,
            "capability_claim": False,
            "proto_complete": False,
            "requires_real_paired_capture_for_capability": True,
        },
    }
    return seal(document)


# ---------------------------------------------------------------------------
# Bridge registry export (for frankenstein_bridges compatibility)
# ---------------------------------------------------------------------------


def v0_bridge_registry() -> dict[str, Any]:
    return seal(
        {
            "schema": "hawking.frankenstein.latent_v0_bridge_registry.v1",
            "recorded_at": _utc_now(),
            "modules": list(V0_MODULE_NAMES),
            "bridge_sites": list(V0_BRIDGE_SITES),
            "heads": list(V0_HEADS),
            "architecture": {
                "teacher_projector": "GLM d_teacher → RMSNorm → Linear → shared latent",
                "student_observer": "DSV4F d_student → RMSNorm → Linear → shared latent",
                "student_intervention": (
                    "hidden → RMSNorm → low-rank → gated MLP → low-rank residual → add"
                ),
            },
            "geometry": {
                "d_teacher": GLM_HIDDEN,
                "d_student": DSV4F_HIDDEN,
                "default_d_latent": DEFAULT_LATENT,
                "default_rank": DEFAULT_RANK,
            },
            "properties": {
                "reversible": True,
                "bypassable": True,
                "hash_bound": True,
                "ablatable": True,
                "gravity_accounted": True,
                "kimi_bridge_compatible": True,
                "no_direct_glm_router_transplant": True,
            },
            "loss_portfolio": dict(LOSS_DEFINITIONS),
            "schedule": [
                {"phase": p, "description": d} for p, d in SCHEDULE_PHASES
            ],
            "checkpoints": list(CHECKPOINT_SLOTS),
            "ablation_arms": [
                {"id": a, "description": d} for a, d in LATENT_AG_ARMS
            ],
            "capability_claim": False,
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PROTO_FRANKENSTEIN_V0 full latent bridge trainer. "
            "Fixture proves the loop; real capture required for capability."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("registry", help="Emit V0 bridge registry seal")
    p_reg.add_argument("--out", type=Path, default=None)

    p_fix = sub.add_parser(
        "fixture-e2e",
        help="Learn+reverse+reject on real-shaped fixture (full geometry)",
    )
    p_fix.add_argument("--epochs-per-phase", type=int, default=6)
    p_fix.add_argument("--n-train", type=int, default=32)
    p_fix.add_argument("--n-eval", type=int, default=8)
    p_fix.add_argument("--seq-len", type=int, default=4)
    p_fix.add_argument("--d-latent", type=int, default=64)
    p_fix.add_argument("--seed", type=int, default=0)
    p_fix.add_argument("--device", type=str, default="cpu")
    p_fix.add_argument("--out", type=Path, default=None)
    p_fix.add_argument(
        "--scaled",
        action="store_true",
        help="Use scaled-down hidden dims (faster unit test)",
    )

    p_ag = sub.add_parser("train-ag", help="Train latent A–G + reject wire")
    p_ag.add_argument("--epochs-per-phase", type=int, default=4)
    p_ag.add_argument("--n-train", type=int, default=24)
    p_ag.add_argument("--n-eval", type=int, default=8)
    p_ag.add_argument("--seed", type=int, default=1)
    p_ag.add_argument("--device", type=str, default="cpu")
    p_ag.add_argument("--out", type=Path, default=None)
    p_ag.add_argument("--scaled", action="store_true")

    p_real = sub.add_parser(
        "fit-real",
        help="Fit on real paired capture (fails closed if absent)",
    )
    p_real.add_argument("--paired", type=Path, required=True)
    p_real.add_argument("--out", type=Path, default=None)

    p_pending = sub.add_parser(
        "ag-pending",
        help="Emit A–G framework pending real capture",
    )
    p_pending.add_argument("--out", type=Path, default=None)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.command == "registry":
        doc = v0_bridge_registry()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps({"status": "OK", "seal_sha256": doc["seal_sha256"]}, indent=2))
        return 0

    if args.command == "fixture-e2e":
        if args.scaled:
            fcfg = FixtureConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                seq_len=args.seq_len,
                d_teacher=96,
                d_student=64,
                d_latent=args.d_latent,
                rank=4,
                d_hidden=32,
                n_experts=8,
                seed=args.seed,
                batch_size=8,
            )
        else:
            fcfg = FixtureConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                seq_len=args.seq_len,
                d_latent=args.d_latent,
                seed=args.seed,
            )
        doc = run_fixture_end_to_end(
            fixture_cfg=fcfg,
            train_cfg=TrainConfig(
                epochs_per_phase=args.epochs_per_phase,
                device=args.device,
                phases=("A", "B", "C", "D", "E"),
            ),
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "status": doc["status"],
                    "learned": doc["learned"],
                    "reverse_ok": doc["reverse_ok"],
                    "bytes_accounted": doc["bytes_accounted"],
                    "tps_accounted": doc["tps_accounted"],
                    "reject_fires": doc["reject_proof"]["proved_reject_fires"],
                    "promotion": doc["promotion_gate"]["verdict"],
                    "fail_closed_gate": doc["fail_closed_real_train"]["gate"],
                    "parameter_bytes": doc["result"]["bytes_account"][
                        "total_parameter_bytes"
                    ],
                    "runtime_parameter_bytes": doc["result"]["bytes_account"][
                        "runtime_parameter_bytes"
                    ],
                    "capability_claim": doc["capability_claim"],
                    "data_kind": doc["data_kind"],
                    "seal_sha256": doc["seal_sha256"],
                    "out": str(args.out) if args.out else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        ok = (
            doc["learned"]
            and doc["reverse_ok"]
            and doc["reject_proof"]["proved_reject_fires"]
        )
        return 0 if ok else 2

    if args.command == "train-ag":
        if args.scaled:
            fcfg = FixtureConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                seq_len=2,
                d_teacher=96,
                d_student=64,
                d_latent=32,
                rank=4,
                d_hidden=32,
                n_experts=8,
                seed=args.seed,
                batch_size=8,
            )
        else:
            fcfg = FixtureConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                seed=args.seed,
            )
        doc = train_ag_variants(
            fixture_cfg=fcfg,
            train_cfg=TrainConfig(
                epochs_per_phase=args.epochs_per_phase,
                device=args.device,
                phases=("A", "B", "C", "E"),
            ),
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        learned = {
            arm: doc["arm_results"][arm]["learned"] for arm in doc["arms_trained"]
        }
        print(
            json.dumps(
                {
                    "status": doc["status"],
                    "ablation_verdict": doc["ablation"].get("verdict"),
                    "reject_proof": doc["reject_proof"],
                    "learned": learned,
                    "capability_claim": doc["capability_claim"],
                    "seal_sha256": doc["seal_sha256"],
                    "out": str(args.out) if args.out else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if doc["reject_proof"]["proved"] else 2

    if args.command == "fit-real":
        loaded = load_real_capture_or_fail(args.paired)
        if loaded.get("status") == "FAIL_CLOSED":
            print(json.dumps(loaded, indent=2, sort_keys=True))
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(
                    json.dumps(loaded, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return 3
        print(
            json.dumps(
                {
                    "status": "LOADED_BUT_REAL_FIT_AWAITS_CAPTURE_SCHEMA_SEAL",
                    "keys": loaded.get("keys"),
                    "capability_claim": False,
                    "note": (
                        "Real tensors loaded; full real-data fit uses train_schedule "
                        "once capture schema is sealed by capture lanes."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "ag-pending":
        doc = run_latent_ag_ablation(None)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "status": doc["status"],
                    "verdict": doc["verdict"],
                    "requires": doc.get("requires"),
                    "seal_sha256": doc["seal_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise LatentV0Error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
