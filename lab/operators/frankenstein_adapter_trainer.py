#!/usr/bin/env python3.12
"""Training loop for reversible bridge/method adapters (paired activations).

Trains the SMALL PyTorch modules defined in frankenstein_adapter_modules on
paired (GLM, DSV4F) activation/behavior tensors with standard autograd.
Does NOT backprop through custom DSV4F Metal kernels.

Losses (owner steer — NOT latent cosine alone):
  * functional_output   — held-out functional target match (MSE on teacher hidden)
  * token_action_kl     — KL on aligned action logits
  * method_classification — method-family CE
  * route_behavior      — route distribution match (KL)
  * verifier_outcome    — verifier pass/fail BCE from value head

Fail-closed:
  * Real paired data absent → REQUIRES_PAIRED_DATA (no fabricated capability)
  * Synthetic fixture is explicitly labelled SYNTHETIC; capability_claim=False

A–G ablation:
  Trains arms B..G independently (A is baseline identity).  Synthetic-proxy
  scores feed the additive-not-subtractive reject rule via frankenstein_ablation.
  Never claims real math inheritance from synthetic runs.

New-file lane: does not edit frankenstein_transfer / frankenstein_bridges.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lab.operators import frankenstein_ablation as ablation
from lab.operators.frankenstein_adapter_modules import (
    ADAPTER_NAMES,
    LOSS_DEFINITIONS,
    METHOD_FAMILIES,
    RESIDUAL_ADAPTER_NAMES,
    FunctionalStack,
    ModuleError,
    ReversibleBridge,
    build_stack_for_arm,
    param_bytes,
    param_count,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
TRAINER_RUN_SCHEMA = "hawking.frankenstein.adapter_trainer_run.v1"
SYNTHETIC_FIXTURE_SCHEMA = "hawking.frankenstein.synthetic_paired_activation.v1"
AG_TRAIN_SCHEMA = "hawking.frankenstein.adapter_ag_train_ablation.v1"

REQUIRES_PAIRED_DATA = "REQUIRES_PAIRED_DATA"
REQUIRES_TRAINING_LOOP = "REQUIRES_TRAINING_LOOP"  # opened by this module for adapters

# Functional-transfer A–G arms (aligned with scaffold; local so this lane
# does not depend on an unmerged ablation extension).
ARM_FT_A = "A_BASE_DSV4F"
ARM_FT_B = "B_LINEAR_SUBSPACE_INIT"
ARM_FT_C = "C_BEHAVIOR_DISTILL"
ARM_FT_D = "D_NONLINEAR_BRIDGES"
ARM_FT_E = "E_ROUTER_METHOD_ADAPTERS"
ARM_FT_F = "F_EXPERT_ITERATION"
ARM_FT_G = "G_COMPLETE_PROTO_FRANKENSTEIN"

FUNCTIONAL_TRANSFER_ARMS: tuple[tuple[str, str], ...] = (
    (ARM_FT_A, "Base DSV4F only; no donor inheritance"),
    (ARM_FT_B, "Base + LINEAR_SUBSPACE_INITIALIZATION (not inheritance)"),
    (ARM_FT_C, "Behavior distillation objectives"),
    (ARM_FT_D, "Nonlinear reversible bridges"),
    (ARM_FT_E, "Router/method adapters (no GLM router copy)"),
    (ARM_FT_F, "Verified expert iteration (repair + value)"),
    (ARM_FT_G, "Complete Proto-Frankenstein stack"),
)

# Default loss weights — latent cosine is never the sole term.
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "functional_output": 1.0,
    "token_action_kl": 0.5,
    "method_classification": 0.5,
    "route_behavior": 0.25,
    "verifier_outcome": 0.25,
    # Optional auxiliary — must not be the only nonzero weight.
    "latent_cosine": 0.05,
}

TRAINABLE_ARMS: tuple[str, ...] = (
    ARM_FT_B,
    ARM_FT_C,
    ARM_FT_D,
    ARM_FT_E,
    ARM_FT_F,
    ARM_FT_G,
)


class TrainerError(RuntimeError):
    """Trainer fail-closed or configuration error."""


class RequiresPairedData(TrainerError):
    """Real paired activation data is absent."""

    gate = REQUIRES_PAIRED_DATA


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Fail-closed real-data gate
# ---------------------------------------------------------------------------


def require_paired_data(
    path: Path | str | None,
    *,
    allow_synthetic: bool = False,
    synthetic_flag: bool = False,
) -> Path | None:
    """Fail closed when real paired data is required but absent.

    Returns the validated path, or None when synthetic_flag is True and
    allow_synthetic is True.
    """

    if synthetic_flag:
        if not allow_synthetic:
            raise RequiresPairedData(
                f"{REQUIRES_PAIRED_DATA}: synthetic flag set but allow_synthetic=False"
            )
        return None
    if path is None:
        raise RequiresPairedData(
            f"{REQUIRES_PAIRED_DATA}: no paired-activation path provided "
            "(capture pipeline has not produced real (GLM,DSV4F) pairs). "
            "Pass --synthetic for the labelled fixture loop only."
        )
    p = Path(path)
    if not p.is_file():
        raise RequiresPairedData(
            f"{REQUIRES_PAIRED_DATA}: paired-activation file missing: {p}"
        )
    return p


def fail_closed_paired_data(*, stage: str, operation: str) -> dict[str, Any]:
    return seal(
        {
            "schema": "hawking.frankenstein.fail_closed.v1",
            "status": "FAIL_CLOSED",
            "gate": REQUIRES_PAIRED_DATA,
            "stage": stage,
            "operation": operation,
            "executed": False,
            "trained": False,
            "fabricated": False,
            "capability_claim": False,
            "recorded_at": _utc_now(),
            "detail": (
                "Real paired (GLM,DSV4F) activation/behavior data is absent. "
                "No capability number is reported."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    functional_output: float = 1.0
    token_action_kl: float = 0.5
    method_classification: float = 0.5
    route_behavior: float = 0.25
    verifier_outcome: float = 0.25
    latent_cosine: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "functional_output": float(self.functional_output),
            "token_action_kl": float(self.token_action_kl),
            "method_classification": float(self.method_classification),
            "route_behavior": float(self.route_behavior),
            "verifier_outcome": float(self.verifier_outcome),
            "latent_cosine": float(self.latent_cosine),
        }

    def validate_not_cosine_alone(self) -> None:
        d = self.as_dict()
        primary = {
            k: v
            for k, v in d.items()
            if k != "latent_cosine" and v > 0.0
        }
        if not primary and d.get("latent_cosine", 0.0) > 0.0:
            raise TrainerError(
                "latent_cosine alone is FORBIDDEN as the sole training objective; "
                "enable at least one of: functional_output, token_action_kl, "
                "method_classification, route_behavior, verifier_outcome"
            )
        if not primary and d.get("latent_cosine", 0.0) <= 0.0:
            raise TrainerError("all loss weights are zero")


def compute_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    weights: LossWeights,
    stack: FunctionalStack,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Steer losses on paired behavior/activation targets (not cosine alone)."""

    weights.validate_not_cosine_alone()
    terms: dict[str, torch.Tensor] = {}
    device = outputs["adapted_hidden"].device

    # 1) Held-out functional output: match teacher/target hidden after adapt.
    if weights.functional_output > 0.0 and "teacher_hidden" in batch:
        pred = outputs["adapted_hidden"]
        target = batch["teacher_hidden"]
        terms["functional_output"] = F.mse_loss(pred, target)

    # 2) Token/action-span KL (aligned action logits, never raw mismatched token ids).
    if weights.token_action_kl > 0.0 and "action_logits" in outputs and "teacher_action_logits" in batch:
        log_p = F.log_softmax(outputs["action_logits"], dim=-1)
        q = F.softmax(batch["teacher_action_logits"], dim=-1)
        terms["token_action_kl"] = F.kl_div(log_p, q, reduction="batchmean")

    # 3) Method classification CE.
    if (
        weights.method_classification > 0.0
        and "method_logits" in outputs
        and "method_id" in batch
    ):
        logits = outputs["method_logits"]
        target = batch["method_id"].long()
        if target.dim() > 1:
            target = target.reshape(-1)
            # If sequence-level labels, take first (synthetic uses per-example).
            if target.numel() != logits.shape[0]:
                target = batch["method_id"].long()
                if target.dim() > 1:
                    target = target[:, 0]
        terms["method_classification"] = F.cross_entropy(logits, target)

    # 4) Route behavior: KL between predicted route distrib and teacher.
    if (
        weights.route_behavior > 0.0
        and "route_logits" in outputs
        and "teacher_route_logits" in batch
    ):
        log_p = F.log_softmax(outputs["route_logits"], dim=-1)
        q = F.softmax(batch["teacher_route_logits"], dim=-1)
        # Flatten batch dims.
        log_p = log_p.reshape(-1, log_p.shape[-1])
        q = q.reshape(-1, q.shape[-1])
        terms["route_behavior"] = F.kl_div(log_p, q, reduction="batchmean")

    # 5) Verifier outcome BCE from value head.
    if (
        weights.verifier_outcome > 0.0
        and "value" in outputs
        and "verifier_pass" in batch
    ):
        value = outputs["value"]
        if value.dim() > 1:
            value = value.mean(dim=-1) if value.shape[-1] != 1 else value.squeeze(-1)
        target = batch["verifier_pass"].float()
        if target.dim() > 1:
            target = target.reshape(target.shape[0], -1).mean(dim=-1)
        terms["verifier_outcome"] = F.binary_cross_entropy_with_logits(value, target)

    # Optional auxiliary latent cosine (never sole).
    if weights.latent_cosine > 0.0 and "teacher_hidden" in batch:
        pred = outputs["adapted_hidden"]
        target = batch["teacher_hidden"]
        # mean cosine distance
        p = pred.reshape(-1, pred.shape[-1])
        t = target.reshape(-1, target.shape[-1])
        cos = F.cosine_similarity(p, t, dim=-1).mean()
        terms["latent_cosine"] = 1.0 - cos

    if not terms:
        # Identity / empty arm: zero loss on a parameter-free path.
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
# Synthetic paired-activation fixture
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    n_train: int = 128
    n_eval: int = 32
    seq_len: int = 8
    d_model: int = 64
    n_experts: int = 16
    n_actions: int = 12
    n_methods: int = len(METHOD_FAMILIES)
    seed: int = 0
    batch_size: int = 16


def make_synthetic_paired_tensors(
    cfg: SyntheticConfig,
    *,
    split: str = "train",
) -> dict[str, torch.Tensor]:
    """Build a SMALL labelled synthetic paired-activation fixture.

    Teacher targets are generated by a fixed random affine + nonlinearity of
    student activations (simulates a donor behavior delta).  This is NOT real
    GLM/DSV4F capture data and must never be reported as a capability number.
    """

    n = cfg.n_train if split == "train" else cfg.n_eval
    g = torch.Generator().manual_seed(int(cfg.seed) + (0 if split == "train" else 97))
    B, S, H = n, cfg.seq_len, cfg.d_model
    E, A, M = cfg.n_experts, cfg.n_actions, cfg.n_methods

    student = torch.randn(B, S, H, generator=g)
    # Fixed "teacher delta" (not learned here) — paired target.
    W = torch.randn(H, H, generator=g) * 0.15
    teacher = student + torch.tanh(student @ W) * 0.5

    method_id = torch.randint(0, M, (B,), generator=g)
    # Method-dependent route preference.
    route_base = torch.randn(B, S, E, generator=g) * 0.3
    method_bias = torch.zeros(M, E)
    for m in range(M):
        method_bias[m, m % E] = 2.0
        method_bias[m, (m * 3) % E] = 1.0
    teacher_route = route_base + method_bias[method_id].unsqueeze(1)
    student_route = route_base.clone()

    # Action logits from teacher mean hidden.
    Wa = torch.randn(H, A, generator=g) * 0.25
    teacher_action = teacher.mean(dim=1) @ Wa
    # Student actions: weaker / noisy
    student_action = student.mean(dim=1) @ Wa + torch.randn(B, A, generator=g) * 0.5

    # Verifier pass correlated with method "repair" (id 3) and teacher norm.
    repair_bonus = (method_id == 3).float() * 0.4
    score = torch.sigmoid(teacher.mean(dim=(1, 2)) * 0.5 + repair_bonus)
    verifier_pass = (score > 0.5).float()

    return {
        "student_hidden": student,
        "teacher_hidden": teacher,
        "route_logits": student_route,
        "teacher_route_logits": teacher_route,
        "action_logits": student_action,
        "teacher_action_logits": teacher_action,
        "method_id": method_id,
        "verifier_pass": verifier_pass,
    }


def synthetic_fixture_manifest(cfg: SyntheticConfig) -> dict[str, Any]:
    return seal(
        {
            "schema": SYNTHETIC_FIXTURE_SCHEMA,
            "recorded_at": _utc_now(),
            "data_kind": "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
            "capability_claim": False,
            "real_glm_dsv4f_capture": False,
            "config": {
                "n_train": cfg.n_train,
                "n_eval": cfg.n_eval,
                "seq_len": cfg.seq_len,
                "d_model": cfg.d_model,
                "n_experts": cfg.n_experts,
                "n_actions": cfg.n_actions,
                "n_methods": cfg.n_methods,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
            },
            "note": (
                "Synthetic only — proves train/reverse/byte-account loop. "
                "Real paired data comes from the capture pipeline."
            ),
        }
    )


class PairedActivationDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.keys = list(tensors.keys())
        self.tensors = {k: v for k, v in tensors.items()}
        n = next(iter(self.tensors.values())).shape[0]
        for v in self.tensors.values():
            if v.shape[0] != n:
                raise TrainerError("all tensors must share batch dimension 0")
        self._n = n

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
    ds = PairedActivationDataset(tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Fit / eval loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-3
    weight_decay: float = 0.0
    device: str | None = None
    loss_weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 10
    grad_clip: float | None = 1.0


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    final_train_loss: float
    final_eval_loss: float
    learned: bool
    reverse_ok: bool
    reverse_recon_error: float | None
    bytes_account: dict[str, Any]
    wall_ms: float
    steps: int
    arm: str | None = None
    data_kind: str = "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE"
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
            "data_kind": self.data_kind,
            "capability_claim": self.capability_claim,
        }


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def eval_epoch(
    stack: FunctionalStack,
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
        loss, logged = compute_losses(outputs, batch, weights=weights, stack=stack)
        n += 1
        for k, v in logged.items():
            totals[k] = totals.get(k, 0.0) + float(v)
    if n == 0:
        return {"total": 0.0}
    return {k: v / n for k, v in totals.items()}


def train_stack(
    stack: FunctionalStack,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    cfg: TrainConfig | None = None,
    arm: str | None = None,
    data_kind: str = "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
) -> TrainResult:
    """Adam fit/eval loop with reverse check + byte accounting."""

    cfg = cfg or TrainConfig()
    device = _device(cfg.device)
    stack = stack.to(device)
    params = [p for p in stack.parameters() if p.requires_grad]
    if not params:
        # Identity arm — no training.
        bytes_acc = stack.gravity_accounting()
        return TrainResult(
            history=[],
            final_train_loss=0.0,
            final_eval_loss=0.0,
            learned=False,
            reverse_ok=True,
            reverse_recon_error=0.0,
            bytes_account=bytes_acc,
            wall_ms=0.0,
            steps=0,
            arm=arm,
            data_kind=data_kind,
            capability_claim=False,
        )

    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []
    steps = 0
    t0 = time.perf_counter()
    initial_eval = eval_epoch(
        stack, eval_loader, weights=cfg.loss_weights, device=device
    )

    for epoch in range(int(cfg.epochs)):
        stack.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            outputs = stack.forward_bundle(batch)
            loss, logged = compute_losses(
                outputs, batch, weights=cfg.loss_weights, stack=stack
            )
            if not torch.isfinite(loss):
                raise TrainerError(f"non-finite loss at step {steps}: {logged}")
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            steps += 1
            epoch_loss += float(logged["total"])
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)
        eval_metrics = eval_epoch(
            stack, eval_loader, weights=cfg.loss_weights, device=device
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "eval_loss": float(eval_metrics.get("total", 0.0)),
        }
        for k, v in eval_metrics.items():
            if k != "total":
                row[f"eval_{k}"] = float(v)
        history.append(row)

    wall_ms = (time.perf_counter() - t0) * 1000.0
    final_train = history[-1]["train_loss"] if history else 0.0
    final_eval = history[-1]["eval_loss"] if history else 0.0
    learned = final_eval < initial_eval.get("total", float("inf")) - 1e-4 or (
        final_train < (history[0]["train_loss"] if history else 0.0) - 1e-4
    )

    # Reverse check: bridge via stored residual (exact); residuals via Woodbury.
    reverse_ok = True
    reverse_err: float | None = 0.0
    stack.eval()
    with torch.no_grad():
        sample = next(iter(eval_loader))
        sample = _move_batch(sample, device)
        x = sample["student_hidden"]
        if stack.bridge is not None and stack.use_bridge:
            y, r = stack.bridge.apply_with_residual(x)
            x_back, info = stack.bridge.revert(y, residual=r, atol=1e-5)
            reverse_err = float(info.get("stored_step_error", info["recon_error"]))
            reverse_ok = bool(info.get("exact", False)) and bool(
                torch.allclose(x_back, x, atol=1e-5, rtol=1e-5)
            )
            # Also record fixed-point recon quality (non-binding for reverse_ok).
            _xf, fp_info = stack.bridge.revert(y, n_iters=24, atol=1e-4)
            reverse_err = max(float(reverse_err or 0.0), float(fp_info["recon_error"]))
        # Residual bank reverse when residuals enabled.
        if stack.residual_enabled:
            y2, applied = stack.bank.apply_residual(x, enabled=stack.residual_enabled)
            x2 = stack.bank.revert_residual(y2, applied)
            err2 = float(torch.max(torch.abs(x2 - x)).item())
            reverse_err = max(float(reverse_err or 0.0), err2)
            reverse_ok = reverse_ok and err2 < 1e-4

    stack.mark_trained()
    bytes_acc = stack.gravity_accounting()
    # FLOP → rough TPS proxy at 1e9 FLOP/s (synthetic host proxy only).
    flops = float(bytes_acc.get("approx_flops_per_token") or 0.0)
    bytes_acc["tps_proxy_at_1gflops"] = (1e9 / flops) if flops > 0 else None
    bytes_acc["wall_train_ms"] = wall_ms
    bytes_acc["steps"] = steps

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
        data_kind=data_kind,
        capability_claim=False,
    )


# ---------------------------------------------------------------------------
# A–G ablation training (B..G)
# ---------------------------------------------------------------------------


def _proxy_scores_from_result(
    result: TrainResult,
    *,
    base_fill: float = 0.70,
) -> dict[str, Any]:
    """Map synthetic train metrics → score maps for the reject harness.

    These are SYNTHETIC PROXY scores only — not real capability measurements.
    They exercise the additive-not-subtractive wiring end-to-end.
    """

    scores = ablation.default_score_template(base_fill)
    # Small math lift if the arm learned + reversed cleanly.
    if result.learned and result.reverse_ok:
        lift = min(0.15, max(0.02, 0.10 * (1.0 / (1.0 + result.final_eval_loss))))
        for d in ablation.MATH_DOMAINS:
            scores["math"][d] = min(1.0, base_fill + lift)
    # Secondary: preserve base (no regression) for healthy synthetic runs.
    # If reverse failed, mark routing_stability regression to fire reject.
    if not result.reverse_ok:
        scores["secondary"]["routing_stability"] = max(0.0, base_fill - 0.10)
        scores["secondary"]["bridge_compatibility"] = max(0.0, base_fill - 0.10)
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
        },
    }


def train_ag_variants(
    *,
    synth_cfg: SyntheticConfig | None = None,
    train_cfg: TrainConfig | None = None,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train B..G stacks on synthetic paired fixture; wire reject rule."""

    synth_cfg = synth_cfg or SyntheticConfig()
    train_cfg = train_cfg or TrainConfig()
    target_arms = list(arms) if arms is not None else list(TRAINABLE_ARMS)

    train_t = make_synthetic_paired_tensors(synth_cfg, split="train")
    eval_t = make_synthetic_paired_tensors(synth_cfg, split="eval")
    train_loader = make_loader(train_t, batch_size=synth_cfg.batch_size, shuffle=True)
    eval_loader = make_loader(eval_t, batch_size=synth_cfg.batch_size, shuffle=False)

    arm_results: dict[str, Any] = {}
    arm_scores: dict[str, Any] = {
        ARM_FT_A: {
            **ablation.default_score_template(0.70),
            "bench_scope": "FIXTURE",
            "data_kind": "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
            "capability_claim": False,
            "synthetic_proxy": True,
        }
    }

    for arm in target_arms:
        stack = build_stack_for_arm(
            arm,
            d_model=synth_cfg.d_model,
            d_hidden=max(16, synth_cfg.d_model // 2),
            rank=4,
            n_experts=synth_cfg.n_experts,
        )
        # Arm-specific loss emphasis.
        lw = LossWeights()
        if arm == ARM_FT_B or arm == ARM_FT_D:
            lw = LossWeights(
                functional_output=1.0,
                token_action_kl=0.0,
                method_classification=0.0,
                route_behavior=0.0,
                verifier_outcome=0.0,
                latent_cosine=0.05,
            )
        elif arm == ARM_FT_C:
            lw = LossWeights(
                functional_output=0.5,
                token_action_kl=1.0,
                method_classification=0.5,
                route_behavior=0.0,
                verifier_outcome=0.25,
                latent_cosine=0.0,
            )
        elif arm == ARM_FT_E:
            lw = LossWeights(
                functional_output=0.5,
                token_action_kl=0.25,
                method_classification=1.0,
                route_behavior=1.0,
                verifier_outcome=0.0,
                latent_cosine=0.0,
            )
        elif arm == ARM_FT_F:
            lw = LossWeights(
                functional_output=0.5,
                token_action_kl=0.25,
                method_classification=0.5,
                route_behavior=0.0,
                verifier_outcome=1.0,
                latent_cosine=0.0,
            )
        # G uses full DEFAULT_LOSS_WEIGHTS
        tcfg = TrainConfig(
            epochs=train_cfg.epochs,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            device=train_cfg.device,
            loss_weights=lw,
            log_every=train_cfg.log_every,
            grad_clip=train_cfg.grad_clip,
        )
        result = train_stack(
            stack,
            train_loader,
            eval_loader,
            cfg=tcfg,
            arm=arm,
            data_kind="SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
        )
        arm_results[arm] = result.as_dict()
        arm_scores[arm] = _proxy_scores_from_result(result)

    # Wire additive-not-subtractive reject via existing A-vs-B harness per arm.
    # Prefer scaffold run_ag_ablation if present; else local pairwise eval.
    ag_report: dict[str, Any]
    if hasattr(ablation, "run_ag_ablation"):
        ag_report = ablation.run_ag_ablation(arm_scores=arm_scores)
    else:
        ag_report = _local_ag_ablation(arm_scores)

    document = {
        "schema": AG_TRAIN_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "SYNTHETIC_AG_TRAIN_COMPLETE",
        "data_kind": "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "real_glm_dsv4f_capture": False,
        "fabricated_capability_number": False,
        "arms_trained": list(target_arms),
        "arm_results": arm_results,
        "ablation": ag_report,
        "reject_rule": "additive_not_subtractive (secondary non-regression)",
        "fixture": synthetic_fixture_manifest(synth_cfg),
        "losses": dict(LOSS_DEFINITIONS),
        "claim_boundary": {
            "training_performed_on_synthetic": True,
            "capability_claim": False,
            "proto_complete": False,
            "requires_real_paired_data_for_capability": True,
        },
    }
    return seal(document)


def _local_ag_ablation(arm_scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pairwise A-vs-B reject against base for each trained arm (local fallback)."""

    base = arm_scores[ARM_FT_A]
    rows: list[dict[str, Any]] = []
    any_reject = False
    for aid, desc in FUNCTIONAL_TRANSFER_ARMS:
        if aid not in arm_scores:
            rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "PENDING",
                    "verdict": None,
                    "fabricated": False,
                }
            )
            continue
        if aid == ARM_FT_A:
            rows.append(
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
            base_math=base["math"],
            base_secondary=base["secondary"],
            proto_math=scores["math"],
            proto_secondary=scores["secondary"],
            bench_scope=str(scores.get("bench_scope") or "FIXTURE"),
            fixture_id=f"ag-train-{aid}",
            transfer_module_id=aid,
        )
        if report["verdict"] == "REJECT":
            any_reject = True
        rows.append(
            {
                "arm": aid,
                "description": desc,
                "status": "EVALUATED",
                "verdict": report["verdict"],
                "reject_rule_fired": report["reject_rule_fired"],
                "math_mean_gain": (report.get("math") or {}).get("mean_gain"),
                "ablation_seal_sha256": report.get("seal_sha256"),
                "synthetic_proxy": True,
            }
        )
    evaluated = [r for r in rows if r.get("status") == "EVALUATED"]
    if any_reject:
        overall = "REJECT"
    elif evaluated and all(r.get("verdict") == "ACCEPT" for r in evaluated):
        overall = "ACCEPT"
    elif evaluated:
        overall = "PARTIAL"
    else:
        overall = "PENDING"
    return seal(
        {
            "schema": "hawking.frankenstein.functional_transfer_ag_ablation.v1",
            "recorded_at": _utc_now(),
            "status": "EVALUATED",
            "verdict": overall,
            "arms": rows,
            "reject_rule_fired": any_reject,
            "fabricated_scores": False,
            "synthetic_proxy_scores": True,
            "capability_claim": False,
            "gate_policy": ablation.sealed_gate_policy(),
        }
    )


# ---------------------------------------------------------------------------
# Single-stack synthetic demo (learns + reverses)
# ---------------------------------------------------------------------------


def run_synthetic_end_to_end(
    *,
    synth_cfg: SyntheticConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    """Prove the trainer learns + reverses cleanly on a synthetic fixture."""

    synth_cfg = synth_cfg or SyntheticConfig()
    train_cfg = train_cfg or TrainConfig(epochs=50, lr=3e-3)
    fixture = synthetic_fixture_manifest(synth_cfg)

    train_t = make_synthetic_paired_tensors(synth_cfg, split="train")
    eval_t = make_synthetic_paired_tensors(synth_cfg, split="eval")
    train_loader = make_loader(train_t, batch_size=synth_cfg.batch_size, shuffle=True)
    eval_loader = make_loader(eval_t, batch_size=synth_cfg.batch_size, shuffle=False)

    stack = FunctionalStack(
        d_model=synth_cfg.d_model,
        d_hidden=max(16, synth_cfg.d_model // 2),
        rank=4,
        n_experts=synth_cfg.n_experts,
        use_bridge=True,
        residual_enabled=list(RESIDUAL_ADAPTER_NAMES),
        use_value_head=True,
        use_route_bias=True,
        use_method_head=True,
    )
    result = train_stack(
        stack,
        train_loader,
        eval_loader,
        cfg=train_cfg,
        arm=ARM_FT_G,
        data_kind="SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
    )

    # Independent ablations of each residual adapter (eval-only).
    ablations: dict[str, Any] = {}
    device = _device(train_cfg.device)
    stack = stack.to(device)
    stack.eval()
    base_eval = eval_epoch(
        stack, eval_loader, weights=train_cfg.loss_weights, device=device
    )
    for name in RESIDUAL_ADAPTER_NAMES:
        # Temporarily skip one adapter.
        totals = 0.0
        n = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = _move_batch(batch, device)
                outputs = stack.forward_bundle(batch, skip_adapters=[name])
                _, logged = compute_losses(
                    outputs, batch, weights=train_cfg.loss_weights, stack=stack
                )
                totals += logged["total"]
                n += 1
        ablations[name] = {
            "eval_loss_when_skipped": totals / max(1, n),
            "full_eval_loss": base_eval.get("total"),
            "independently_ablatable": True,
        }

    document = {
        "schema": TRAINER_RUN_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "SYNTHETIC_E2E_COMPLETE",
        "data_kind": "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "real_glm_dsv4f_capture": False,
        "fabricated_capability_number": False,
        "fixture": fixture,
        "result": result.as_dict(),
        "learned": result.learned,
        "reverse_ok": result.reverse_ok,
        "adapter_ablations": ablations,
        "losses": dict(LOSS_DEFINITIONS),
        "loss_weights": train_cfg.loss_weights.as_dict(),
        "modules": {
            "bridge": stack.bridge.to_spec() if stack.bridge is not None else None,
            "bank": stack.bank.to_spec(),
        },
        "claim_boundary": {
            "training_performed_on_synthetic": True,
            "capability_claim": False,
            "proto_complete": False,
            "requires_real_paired_data_for_capability": True,
            "opens_gate": REQUIRES_TRAINING_LOOP,
            "still_closed_for_capability": REQUIRES_PAIRED_DATA,
        },
    }
    return seal(document)


def load_real_paired_or_fail(path: Path | str | None) -> dict[str, Any]:
    """Attempt to load real paired data; fail closed if absent.

    Real capture format (future): torch .pt or .npz with keys matching
    the synthetic batch schema.  Not fabricated here.
    """

    try:
        p = require_paired_data(path, allow_synthetic=False, synthetic_flag=False)
    except RequiresPairedData as exc:
        closed = fail_closed_paired_data(
            stage="adapter_trainer",
            operation="load_real_paired",
        )
        closed["detail_exc"] = str(exc)
        return closed
    assert p is not None
    # Load without fabricating labels.
    if p.suffix == ".pt":
        blob = torch.load(p, map_location="cpu", weights_only=True)
    else:
        raise TrainerError(f"unsupported paired-data format: {p.suffix}")
    if not isinstance(blob, Mapping) or "student_hidden" not in blob:
        raise TrainerError("paired file missing student_hidden")
    return {
        "status": "LOADED",
        "path": str(p),
        "keys": sorted(blob.keys()),
        "capability_claim": False,
        "data_kind": "REAL_PAIRED_ACTIVATION",
        "tensors": blob,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train reversible bridge/method adapters on paired activations. "
            "Synthetic fixture proves the loop; real data required for capability."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser("synthetic-e2e", help="Learn+reverse on synthetic fixture")
    p_syn.add_argument("--epochs", type=int, default=50)
    p_syn.add_argument("--d-model", type=int, default=64)
    p_syn.add_argument("--n-train", type=int, default=128)
    p_syn.add_argument("--n-eval", type=int, default=32)
    p_syn.add_argument("--seed", type=int, default=0)
    p_syn.add_argument("--device", type=str, default=None)
    p_syn.add_argument("--out", type=Path, default=None)

    p_ag = sub.add_parser("train-ag", help="Train B..G variants + reject-rule wire")
    p_ag.add_argument("--epochs", type=int, default=30)
    p_ag.add_argument("--d-model", type=int, default=64)
    p_ag.add_argument("--n-train", type=int, default=96)
    p_ag.add_argument("--n-eval", type=int, default=24)
    p_ag.add_argument("--seed", type=int, default=1)
    p_ag.add_argument("--device", type=str, default=None)
    p_ag.add_argument("--out", type=Path, default=None)

    p_real = sub.add_parser(
        "fit-real",
        help="Fit on real paired path (fails closed if absent)",
    )
    p_real.add_argument("--paired", type=Path, required=True)
    p_real.add_argument("--out", type=Path, default=None)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "synthetic-e2e":
        doc = run_synthetic_end_to_end(
            synth_cfg=SyntheticConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                d_model=args.d_model,
                seed=args.seed,
            ),
            train_cfg=TrainConfig(epochs=args.epochs, device=args.device),
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
                    "final_eval_loss": doc["result"]["final_eval_loss"],
                    "final_train_loss": doc["result"]["final_train_loss"],
                    "parameter_bytes": doc["result"]["bytes_account"][
                        "total_parameter_bytes"
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
        return 0 if doc["learned"] and doc["reverse_ok"] else 2

    if args.command == "train-ag":
        doc = train_ag_variants(
            synth_cfg=SyntheticConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                d_model=args.d_model,
                seed=args.seed,
            ),
            train_cfg=TrainConfig(epochs=args.epochs, device=args.device),
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        learned = {
            arm: doc["arm_results"][arm]["learned"] for arm in doc["arms_trained"]
        }
        reverse = {
            arm: doc["arm_results"][arm]["reverse_ok"] for arm in doc["arms_trained"]
        }
        print(
            json.dumps(
                {
                    "status": doc["status"],
                    "ablation_verdict": doc["ablation"].get("verdict"),
                    "learned": learned,
                    "reverse_ok": reverse,
                    "capability_claim": doc["capability_claim"],
                    "data_kind": doc["data_kind"],
                    "seal_sha256": doc["seal_sha256"],
                    "out": str(args.out) if args.out else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "fit-real":
        loaded = load_real_paired_or_fail(args.paired)
        if loaded.get("status") == "FAIL_CLOSED":
            print(json.dumps(loaded, indent=2, sort_keys=True))
            return 3
        print(
            json.dumps(
                {
                    "status": "LOADED_BUT_REAL_FIT_DEFERRED",
                    "note": (
                        "Real tensors loaded; full real-data fit uses the same "
                        "train_stack path once capture schema is sealed."
                    ),
                    "keys": loaded.get("keys"),
                    "capability_claim": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise TrainerError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
