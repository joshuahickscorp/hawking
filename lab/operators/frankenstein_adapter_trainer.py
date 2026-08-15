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

CPU multi-worker parallelism (orchestration only — not module math):
  * Independent bridge SITES (GLM_METHOD_BRIDGE, …) train concurrently.
  * Independent A–G arms train concurrently the same way.
  * Thread budget is partitioned across workers to avoid BLAS oversubscription
    (N processes × 28 OpenMP threads would thrash, not saturate).

New-file lane: does not edit frankenstein_transfer / frankenstein_bridges.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from lab.operators.frankenstein_bridges import V0_BRIDGE_SITES
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
TRAINER_RUN_SCHEMA = "hawking.frankenstein.adapter_trainer_run.v1"
SYNTHETIC_FIXTURE_SCHEMA = "hawking.frankenstein.synthetic_paired_activation.v1"
AG_TRAIN_SCHEMA = "hawking.frankenstein.adapter_ag_train_ablation.v1"
SITE_TRAIN_SCHEMA = "hawking.frankenstein.site_bridge_train.v1"
PARALLEL_BENCH_SCHEMA = "hawking.frankenstein.bridge_train_parallel_bench.v1"

REQUIRES_PAIRED_DATA = "REQUIRES_PAIRED_DATA"
REQUIRES_PAIRED_CAPTURE = "REQUIRES_PAIRED_CAPTURE"  # latent V0 real-train gate
REQUIRES_TRAINING_LOOP = "REQUIRES_TRAINING_LOOP"  # opened by this module for adapters

# Default float tolerance when bit-exact match fails due to BLAS reduction order
# under mismatched thread counts. Correctness proofs pin threads_per_worker so
# parallel and serial share the same intra-op budget and match bit-exact.
PARALLEL_MATCH_ATOL = 1e-6
PARALLEL_MATCH_RTOL = 1e-5

# ThreadPool workers share process-global torch RNG; serialize seed+module-init
# only. Forward/backward for independent modules may still run concurrently.
_TORCH_INIT_LOCK = threading.Lock()

# Full latent V0 lives in frankenstein_latent_v0 (teacher proj / observer /
# interventions / 11-loss schedule A–F / latent A–G).  This module remains the
# functional-transfer adapter path and re-exports the latent entry points.
def _latent_v0():
    from lab.operators import frankenstein_latent_v0 as latent_v0

    return latent_v0

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
# CPU multi-worker parallelism (sites + A–G arms)
# ---------------------------------------------------------------------------


def resolve_cpu_parallel_budget(
    n_units: int,
    *,
    max_workers: int | None = None,
    cpu_count: int | None = None,
    threads_per_worker: int | None = None,
) -> dict[str, int]:
    """Partition CPU cores across independent train units.

    Avoids the thrash case: N processes each spinning full OMP/MKL/Accelerate
    thread pools (e.g. 8 × 28 = 224 threads on a 28-core machine).
    """

    n_units = max(0, int(n_units))
    cpus = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    cpus = max(1, cpus)
    if n_units <= 0:
        return {
            "n_units": 0,
            "n_workers": 0,
            "threads_per_worker": max(1, cpus),
            "cpu_count": cpus,
            "total_threads_budget": cpus,
        }
    if max_workers is None:
        n_workers = min(n_units, cpus)
    else:
        n_workers = max(1, min(int(max_workers), n_units, cpus))
    if threads_per_worker is not None:
        tpw = max(1, int(threads_per_worker))
    else:
        tpw = max(1, cpus // n_workers)
    return {
        "n_units": n_units,
        "n_workers": n_workers,
        "threads_per_worker": tpw,
        "cpu_count": cpus,
        "total_threads_budget": n_workers * tpw,
    }


def apply_cpu_thread_budget(n_threads: int) -> dict[str, int]:
    """Pin torch + BLAS thread pools for this process (call in every worker)."""

    n = max(1, int(n_threads))
    # Set env first so late-loaded BLAS libs pick it up; also re-set for spawn children.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # interop threads can only be set once per process
        pass
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "requested_threads": n,
    }


def unit_seed(base_seed: int, unit_key: str) -> int:
    """Deterministic per-unit seed so serial and parallel agree regardless of order."""

    digest = hashlib.sha256(f"{int(base_seed)}::{unit_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _mp_context() -> mp.context.BaseContext:
    # spawn: clean interpreter, env vars + thread budget apply correctly on macOS
    return mp.get_context("spawn")


def map_train_units(
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    payloads: Sequence[dict[str, Any]],
    *,
    n_workers: int,
    threads_per_worker: int,
    prefer_processes: bool = True,
    worker_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run independent train units serially or across workers.

    Preference order:
      1. ProcessPool (true isolation; each worker gets threads_per_worker).
         Preferred on unrestricted hosts — saturates cores without GIL.
      2. ThreadPool fallback when ProcessPool semaphores are blocked
         (sandboxed agents).  Forces torch/BLAS to 1 thread so concurrent
         Python workers do not thrash a shared OpenMP pool.  Init is
         serialized under ``_TORCH_INIT_LOCK`` for bit-exact determinism;
         forward/backward of independent modules still run concurrently.

    ``worker_name`` is accepted for API stability (subprocess backend was
    removed: after parent imports torch, child torch imports can hang on
    macOS OpenMP fork-safety).

    Results are returned in the same order as payloads.
    """

    _ = worker_name  # reserved / API-compat
    items = list(payloads)
    if not items:
        return []
    n_workers = max(1, int(n_workers))
    # Always pin in the parent too so serial path matches worker path.
    apply_cpu_thread_budget(threads_per_worker)
    if n_workers == 1 or len(items) == 1:
        rows = [worker(p) for p in items]
        for row in rows:
            if isinstance(row, dict):
                row.setdefault("_parallel_backend", "serial")
        return rows

    pool_size = min(n_workers, len(items))

    def _collect(pool: concurrent.futures.Executor) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(items)
        future_to_idx = {
            pool.submit(worker, payload): i for i, payload in enumerate(items)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
        out: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            if r is None:
                raise TrainerError(f"parallel train unit {i} returned no result")
            out.append(r)
        return out

    if prefer_processes:
        try:
            ctx = _mp_context()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=pool_size,
                mp_context=ctx,
                initializer=apply_cpu_thread_budget,
                initargs=(int(threads_per_worker),),
            ) as pool:
                rows = _collect(pool)
            for row in rows:
                if isinstance(row, dict):
                    row.setdefault("_parallel_backend", "process")
            return rows
        except (PermissionError, OSError):
            pass

    # ThreadPool fallback: pin intra-op to 1 so N concurrent units share the
    # process without N×threads BLAS oversubscription.
    # IMPORTANT: for bit-exact serial↔parallel match under ThreadPool, callers
    # must also pin threads_per_worker=1 on the serial path (compare helpers do).
    apply_cpu_thread_budget(1)
    thread_payloads = []
    for p in items:
        q = dict(p)
        q["threads_per_worker"] = 1
        q["_parallel_backend"] = "thread"
        thread_payloads.append(q)
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
        rows = _collect(pool)
    for row in rows:
        if isinstance(row, dict):
            row.setdefault("_parallel_backend", "thread")
    return rows


def loss_weights_for_arm(arm: str) -> LossWeights:
    """Arm-specific loss emphasis (shared by serial and parallel AG paths)."""

    if arm == ARM_FT_B or arm == ARM_FT_D:
        return LossWeights(
            functional_output=1.0,
            token_action_kl=0.0,
            method_classification=0.0,
            route_behavior=0.0,
            verifier_outcome=0.0,
            latent_cosine=0.05,
        )
    if arm == ARM_FT_C:
        return LossWeights(
            functional_output=0.5,
            token_action_kl=1.0,
            method_classification=0.5,
            route_behavior=0.0,
            verifier_outcome=0.25,
            latent_cosine=0.0,
        )
    if arm == ARM_FT_E:
        return LossWeights(
            functional_output=0.5,
            token_action_kl=0.25,
            method_classification=1.0,
            route_behavior=1.0,
            verifier_outcome=0.0,
            latent_cosine=0.0,
        )
    if arm == ARM_FT_F:
        return LossWeights(
            functional_output=0.5,
            token_action_kl=0.25,
            method_classification=0.5,
            route_behavior=0.0,
            verifier_outcome=1.0,
            latent_cosine=0.0,
        )
    # G and unknown use full defaults
    return LossWeights()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        # Prefer determinism where available; do not fail closed if an op is banned.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[call-arg]
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


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
    seed: int | None = None,
) -> DataLoader:
    """Build a single-process DataLoader (num_workers=0).

    When ``seed`` is set and shuffle=True, shuffle order is deterministic so
    serial and multi-process train units can match bit-exact given the same
    threads_per_worker budget.
    """

    ds = PairedActivationDataset(tensors)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,  # never nest DataLoader workers under site/arm process pool
    )


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


def _train_config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "epochs": int(cfg.epochs),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "device": cfg.device,
        "log_every": int(cfg.log_every),
        "grad_clip": cfg.grad_clip,
    }


def _synth_config_to_dict(cfg: SyntheticConfig) -> dict[str, Any]:
    return {
        "n_train": int(cfg.n_train),
        "n_eval": int(cfg.n_eval),
        "seq_len": int(cfg.seq_len),
        "d_model": int(cfg.d_model),
        "n_experts": int(cfg.n_experts),
        "n_actions": int(cfg.n_actions),
        "n_methods": int(cfg.n_methods),
        "seed": int(cfg.seed),
        "batch_size": int(cfg.batch_size),
    }


def _train_one_arm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: train a single independent A–G arm on synthetic fixture.

    Self-contained (rebuilds data + stack from seeds) so serial and parallel
    paths produce identical results when threads_per_worker is pinned equal.
    """

    apply_cpu_thread_budget(int(payload["threads_per_worker"]))
    arm = str(payload["arm"])
    seed = int(payload["unit_seed"])
    _seed_everything(seed)

    synth_cfg = SyntheticConfig(**payload["synth_cfg"])
    train_cfg_d = payload["train_cfg"]
    lw = loss_weights_for_arm(arm)
    tcfg = TrainConfig(
        epochs=int(train_cfg_d["epochs"]),
        lr=float(train_cfg_d["lr"]),
        weight_decay=float(train_cfg_d["weight_decay"]),
        device=train_cfg_d.get("device") or "cpu",
        loss_weights=lw,
        log_every=int(train_cfg_d.get("log_every") or 10),
        grad_clip=train_cfg_d.get("grad_clip"),
    )
    # Force CPU: standing convention for bridge train.
    if tcfg.device not in (None, "cpu"):
        tcfg.device = "cpu"

    train_t = make_synthetic_paired_tensors(synth_cfg, split="train")
    eval_t = make_synthetic_paired_tensors(synth_cfg, split="eval")
    train_loader = make_loader(
        train_t,
        batch_size=synth_cfg.batch_size,
        shuffle=True,
        seed=seed,
    )
    eval_loader = make_loader(
        eval_t,
        batch_size=synth_cfg.batch_size,
        shuffle=False,
        seed=None,
    )

    # Re-seed + construct + materialize lazy modules under lock so ThreadPool
    # workers cannot race process-global torch RNG during nn.init (including
    # FunctionalStack's lazy _action_proj created on first forward_bundle).
    with _TORCH_INIT_LOCK:
        _seed_everything(seed + 1)
        stack = build_stack_for_arm(
            arm,
            d_model=synth_cfg.d_model,
            d_hidden=max(16, synth_cfg.d_model // 2),
            rank=4,
            n_experts=synth_cfg.n_experts,
        )
        # Materialize lazy heads with a deterministic seed before concurrent train.
        _seed_everything(seed + 2)
        sample = next(iter(train_loader))
        dev = torch.device("cpu")
        stack = stack.to(dev)
        stack.eval()
        with torch.no_grad():
            stack.forward_bundle(_move_batch(sample, dev))
    result = train_stack(
        stack,
        train_loader,
        eval_loader,
        cfg=tcfg,
        arm=arm,
        data_kind="SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
    )
    state = {k: v.detach().cpu().clone() for k, v in stack.state_dict().items()}
    out = {
        "arm": arm,
        "unit_seed": seed,
        "result": result.as_dict(),
        "scores": _proxy_scores_from_result(result),
        "state_digest": _state_digest(state),
    }
    if payload.get("return_states"):
        out["state"] = state
    return out


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def train_ag_variants(
    *,
    synth_cfg: SyntheticConfig | None = None,
    train_cfg: TrainConfig | None = None,
    arms: Sequence[str] | None = None,
    max_workers: int | None = None,
    threads_per_worker: int | None = None,
    parallel: bool = True,
    return_states: bool = False,
) -> dict[str, Any]:
    """Train B..G stacks on synthetic paired fixture; wire reject rule.

    Arms are independent (own stack, own loss, no shared params) — safe to
    train concurrently across CPU workers when ``parallel=True``.
    """

    synth_cfg = synth_cfg or SyntheticConfig()
    train_cfg = train_cfg or TrainConfig()
    # Standing convention: bridge train stays on CPU.
    if train_cfg.device not in (None, "cpu"):
        train_cfg = TrainConfig(
            epochs=train_cfg.epochs,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            device="cpu",
            loss_weights=train_cfg.loss_weights,
            log_every=train_cfg.log_every,
            grad_clip=train_cfg.grad_clip,
        )
    target_arms = list(arms) if arms is not None else list(TRAINABLE_ARMS)

    budget = resolve_cpu_parallel_budget(
        len(target_arms),
        max_workers=(1 if not parallel else max_workers),
        threads_per_worker=threads_per_worker,
    )
    if not parallel:
        budget = resolve_cpu_parallel_budget(
            len(target_arms),
            max_workers=1,
            threads_per_worker=threads_per_worker
            if threads_per_worker is not None
            else budget["threads_per_worker"],
        )

    synth_d = _synth_config_to_dict(synth_cfg)
    train_d = _train_config_to_dict(train_cfg)
    payloads = [
        {
            "arm": arm,
            "unit_seed": unit_seed(synth_cfg.seed, f"arm:{arm}"),
            "synth_cfg": synth_d,
            "train_cfg": train_d,
            "threads_per_worker": budget["threads_per_worker"],
            "return_states": bool(return_states),
        }
        for arm in target_arms
    ]

    t0 = time.perf_counter()
    unit_rows = map_train_units(
        _train_one_arm_payload,
        payloads,
        n_workers=budget["n_workers"],
        threads_per_worker=budget["threads_per_worker"],
        worker_name="train_one_arm",
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    backend = (
        unit_rows[0].get("_parallel_backend") if unit_rows else "serial"
    )

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
    arm_states: dict[str, dict[str, torch.Tensor]] = {}
    arm_digests: dict[str, str] = {}
    for row in unit_rows:
        arm = row["arm"]
        arm_results[arm] = row["result"]
        arm_scores[arm] = row["scores"]
        arm_digests[arm] = row["state_digest"]
        if return_states and "state" in row:
            arm_states[arm] = row["state"]

    # Wire additive-not-subtractive reject via existing A-vs-B harness per arm.
    # Prefer scaffold run_ag_ablation if present; else local pairwise eval.
    ag_report: dict[str, Any]
    if hasattr(ablation, "run_ag_ablation"):
        ag_report = ablation.run_ag_ablation(arm_scores=arm_scores)
    else:
        ag_report = _local_ag_ablation(arm_scores)

    document: dict[str, Any] = {
        "schema": AG_TRAIN_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "SYNTHETIC_AG_TRAIN_COMPLETE",
        "data_kind": "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "real_glm_dsv4f_capture": False,
        "fabricated_capability_number": False,
        "arms_trained": list(target_arms),
        "arm_results": arm_results,
        "arm_state_digests": arm_digests,
        "ablation": ag_report,
        "reject_rule": "additive_not_subtractive (secondary non-regression)",
        "fixture": synthetic_fixture_manifest(synth_cfg),
        "losses": dict(LOSS_DEFINITIONS),
        "parallelism": {
            **budget,
            "parallel": bool(parallel and budget["n_workers"] > 1),
            "unit": "arm",
            "wall_ms": wall_ms,
            "backend": backend,
            "cpu_only": True,
            "gpu_used": False,
        },
        "claim_boundary": {
            "training_performed_on_synthetic": True,
            "capability_claim": False,
            "proto_complete": False,
            "requires_real_paired_data_for_capability": True,
        },
    }
    # States are non-JSON (torch tensors) — seal the public doc, re-attach after.
    sealed = seal(document)
    if return_states:
        sealed["_arm_states"] = arm_states
    return sealed


# ---------------------------------------------------------------------------
# Independent bridge SITE training (natural parallel unit)
# ---------------------------------------------------------------------------


def make_synthetic_site_tensors(
    cfg: SyntheticConfig,
    site: str,
    *,
    split: str = "train",
) -> dict[str, torch.Tensor]:
    """Per-site synthetic paired activations (independent teacher transform).

    Each site gets its own RNG stream from ``unit_seed(cfg.seed, site)`` so
    sites do not share state — matching the real transplant-point story where
    each bridge trains against its own layer's activations.
    """

    seed = unit_seed(cfg.seed, f"site-data:{site}:{split}")
    g = torch.Generator().manual_seed(seed)
    n = cfg.n_train if split == "train" else cfg.n_eval
    B, S, H = n, cfg.seq_len, cfg.d_model
    student = torch.randn(B, S, H, generator=g)
    # Site-specific affine + nonlinearity as teacher delta target.
    W = torch.randn(H, H, generator=g) * 0.15
    bias = torch.randn(H, generator=g) * 0.05
    # Mix in a site-name-stable offset so sites differ even if seeds collided.
    site_tag = (unit_seed(0, site) % 997) / 997.0
    teacher = student + torch.tanh(student @ W) * (0.4 + 0.2 * site_tag) + bias
    return {
        "student_hidden": student,
        "teacher_hidden": teacher,
    }


def train_one_site_bridge(
    site: str,
    train_t: Mapping[str, torch.Tensor],
    eval_t: Mapping[str, torch.Tensor],
    *,
    epochs: int = 40,
    lr: float = 3e-3,
    seed: int = 0,
    d_hidden: int | None = None,
    rank: int = 4,
    batch_size: int = 16,
    device: str = "cpu",
    threads_per_worker: int | None = None,
) -> dict[str, Any]:
    """Train a single independent ReversibleBridge at one V0 site (CPU)."""

    if threads_per_worker is not None:
        apply_cpu_thread_budget(threads_per_worker)
    if device != "cpu":
        device = "cpu"

    d_model = int(train_t["student_hidden"].shape[-1])
    d_h = int(d_hidden if d_hidden is not None else max(16, d_model // 2))
    # Seed + construct under lock: ThreadPool workers share global torch RNG.
    with _TORCH_INIT_LOCK:
        _seed_everything(seed)
        bridge = ReversibleBridge(
            name=site,
            d_model=d_model,
            d_hidden=d_h,
            rank=int(rank),
            scale=0.02,
        )
    dev = torch.device("cpu")
    bridge = bridge.to(dev)
    opt = torch.optim.Adam(bridge.parameters(), lr=float(lr))

    train_loader = make_loader(
        train_t, batch_size=batch_size, shuffle=True, seed=seed
    )
    eval_loader = make_loader(
        eval_t, batch_size=batch_size, shuffle=False, seed=None
    )

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    steps = 0
    for epoch in range(int(epochs)):
        bridge.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["student_hidden"].to(dev)
            y_tgt = batch["teacher_hidden"].to(dev)
            opt.zero_grad(set_to_none=True)
            y = bridge(x)
            loss = F.mse_loss(y, y_tgt)
            if not torch.isfinite(loss):
                raise TrainerError(f"non-finite site loss at step {steps}: {site}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            opt.step()
            steps += 1
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        bridge.eval()
        eval_loss = 0.0
        n_ev = 0
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["student_hidden"].to(dev)
                y_tgt = batch["teacher_hidden"].to(dev)
                eval_loss += float(F.mse_loss(bridge(x), y_tgt).item())
                n_ev += 1
        eval_loss = eval_loss / max(1, n_ev)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }
        )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    final_train = history[-1]["train_loss"] if history else 0.0
    final_eval = history[-1]["eval_loss"] if history else 0.0
    learned = bool(
        history
        and (
            final_eval < history[0]["eval_loss"] - 1e-4
            or final_train < history[0]["train_loss"] - 1e-4
        )
    )

    # Reverse exactness via stored residual
    reverse_ok = True
    reverse_err = 0.0
    bridge.eval()
    with torch.no_grad():
        sample = next(iter(eval_loader))
        x = sample["student_hidden"].to(dev)
        y, r = bridge.apply_with_residual(x)
        x_back, info = bridge.revert(y, residual=r, atol=1e-5)
        reverse_err = float(info.get("stored_step_error", info["recon_error"]))
        reverse_ok = bool(info.get("exact", False)) and bool(
            torch.allclose(x_back, x, atol=1e-5, rtol=1e-5)
        )

    bridge.meta["trained"] = True
    state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
    return {
        "site": site,
        "unit_seed": int(seed),
        "final_train_loss": float(final_train),
        "final_eval_loss": float(final_eval),
        "learned": learned,
        "reverse_ok": reverse_ok,
        "reverse_recon_error": reverse_err,
        "wall_ms": wall_ms,
        "steps": steps,
        "history": history,
        "state_digest": _state_digest(state),
        "state": state,
        "gravity": bridge.gravity_accounting(),
        "data_kind": "SYNTHETIC_SITE_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "device": "cpu",
    }


def _train_one_site_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker for one independent bridge site."""

    apply_cpu_thread_budget(int(payload["threads_per_worker"]))
    site = str(payload["site"])
    seed = int(payload["unit_seed"])
    synth_cfg = SyntheticConfig(**payload["synth_cfg"])
    train_t = make_synthetic_site_tensors(synth_cfg, site, split="train")
    eval_t = make_synthetic_site_tensors(synth_cfg, site, split="eval")
    # Drop bulky state from worker return unless requested.
    row = train_one_site_bridge(
        site,
        train_t,
        eval_t,
        epochs=int(payload["epochs"]),
        lr=float(payload["lr"]),
        seed=seed,
        d_hidden=payload.get("d_hidden"),
        rank=int(payload.get("rank") or 4),
        batch_size=int(synth_cfg.batch_size),
        device="cpu",
        threads_per_worker=int(payload["threads_per_worker"]),
    )
    if not payload.get("return_states"):
        row = {k: v for k, v in row.items() if k != "state"}
    return row


def train_sites(
    *,
    sites: Sequence[str] | None = None,
    synth_cfg: SyntheticConfig | None = None,
    epochs: int = 40,
    lr: float = 3e-3,
    rank: int = 4,
    max_workers: int | None = None,
    threads_per_worker: int | None = None,
    parallel: bool = True,
    return_states: bool = False,
) -> dict[str, Any]:
    """Train independent V0 bridge sites (CPU multi-worker when parallel=True).

    Correctness-safe: each site is a separate module with its own activations,
    optimizer, and seed — no shared state between sites.
    """

    synth_cfg = synth_cfg or SyntheticConfig()
    target_sites = list(sites) if sites is not None else list(V0_BRIDGE_SITES)
    budget = resolve_cpu_parallel_budget(
        len(target_sites),
        max_workers=(1 if not parallel else max_workers),
        threads_per_worker=threads_per_worker,
    )
    if not parallel:
        # Keep the same threads_per_worker as a parallel run would use when the
        # caller pins it; otherwise use full-CPU serial (single process).
        tpw = (
            threads_per_worker
            if threads_per_worker is not None
            else budget["threads_per_worker"]
        )
        budget = resolve_cpu_parallel_budget(
            len(target_sites),
            max_workers=1,
            threads_per_worker=tpw,
        )

    synth_d = _synth_config_to_dict(synth_cfg)
    payloads = [
        {
            "site": site,
            "unit_seed": unit_seed(synth_cfg.seed, f"site-train:{site}"),
            "synth_cfg": synth_d,
            "epochs": int(epochs),
            "lr": float(lr),
            "rank": int(rank),
            "d_hidden": max(16, synth_cfg.d_model // 2),
            "threads_per_worker": budget["threads_per_worker"],
            "return_states": bool(return_states),
        }
        for site in target_sites
    ]

    t0 = time.perf_counter()
    rows = map_train_units(
        _train_one_site_payload,
        payloads,
        n_workers=budget["n_workers"],
        threads_per_worker=budget["threads_per_worker"],
        worker_name="train_one_site",
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    backend = rows[0].get("_parallel_backend") if rows else "serial"

    site_results: dict[str, Any] = {}
    for row in rows:
        site = row["site"]
        # Strip state from sealed document unless caller needs it.
        public = {
            k: v
            for k, v in row.items()
            if k not in ("state", "_parallel_backend")
        }
        site_results[site] = public

    all_learned = all(r.get("learned") for r in site_results.values()) if site_results else False
    all_reverse = (
        all(r.get("reverse_ok") for r in site_results.values()) if site_results else False
    )

    document: dict[str, Any] = {
        "schema": SITE_TRAIN_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "SYNTHETIC_SITE_TRAIN_COMPLETE",
        "data_kind": "SYNTHETIC_SITE_PAIRED_ACTIVATION_FIXTURE",
        "capability_claim": False,
        "real_glm_dsv4f_capture": False,
        "fabricated_capability_number": False,
        "sites_trained": list(target_sites),
        "site_results": site_results,
        "all_learned": all_learned,
        "all_reverse_ok": all_reverse,
        "fixture": synthetic_fixture_manifest(synth_cfg),
        "parallelism": {
            **budget,
            "parallel": bool(parallel and budget["n_workers"] > 1),
            "unit": "site",
            "wall_ms": wall_ms,
            "backend": backend,
            "cpu_only": True,
            "gpu_used": False,
            "why_safe": (
                "Each V0 bridge site is an independent ReversibleBridge with its "
                "own activations, params, optimizer, and seed; no cross-site state."
            ),
        },
        "gate_still_closed": REQUIRES_PAIRED_DATA,
        "claim_boundary": {
            "training_performed_on_synthetic": True,
            "capability_claim": False,
            "requires_real_paired_data_for_capability": True,
        },
    }
    sealed = seal(document)
    if return_states:
        sealed["_site_states"] = {
            row["site"]: row["state"] for row in rows if "state" in row
        }
    return sealed


def compare_serial_parallel_sites(
    *,
    synth_cfg: SyntheticConfig | None = None,
    epochs: int = 25,
    sites: Sequence[str] | None = None,
    max_workers: int | None = None,
    atol: float = PARALLEL_MATCH_ATOL,
    rtol: float = PARALLEL_MATCH_RTOL,
) -> dict[str, Any]:
    """Prove parallel site training matches serial (same seed, pinned threads).

    Correctness pin: threads_per_worker=1 on BOTH serial and parallel so the
    ThreadPool fallback (forced to 1 BLAS thread) matches serial bit-exact.
    When ProcessPool is available the same pin still holds.
    """

    synth_cfg = synth_cfg or SyntheticConfig(n_train=48, n_eval=12, seq_len=4, d_model=32)
    target = list(sites) if sites is not None else list(V0_BRIDGE_SITES)
    tpw = 1

    serial = train_sites(
        sites=target,
        synth_cfg=synth_cfg,
        epochs=epochs,
        parallel=False,
        threads_per_worker=tpw,
        return_states=True,
        max_workers=1,
    )
    parallel = train_sites(
        sites=target,
        synth_cfg=synth_cfg,
        epochs=epochs,
        parallel=True,
        threads_per_worker=tpw,
        return_states=True,
        max_workers=max_workers,
    )

    mismatches: list[dict[str, Any]] = []
    bit_exact = True
    tol_match = True
    for site in target:
        s_dig = serial["site_results"][site]["state_digest"]
        p_dig = parallel["site_results"][site]["state_digest"]
        s_loss = float(serial["site_results"][site]["final_eval_loss"])
        p_loss = float(parallel["site_results"][site]["final_eval_loss"])
        s_state = serial["_site_states"][site]
        p_state = parallel["_site_states"][site]
        site_bit = s_dig == p_dig
        site_tol = True
        max_abs = 0.0
        for k in s_state:
            diff = (s_state[k] - p_state[k]).abs().max().item()
            max_abs = max(max_abs, float(diff))
            if not torch.allclose(s_state[k], p_state[k], atol=atol, rtol=rtol):
                site_tol = False
        bit_exact = bit_exact and site_bit
        tol_match = tol_match and site_tol
        if not site_bit or not site_tol:
            mismatches.append(
                {
                    "site": site,
                    "serial_digest": s_dig,
                    "parallel_digest": p_dig,
                    "serial_eval_loss": s_loss,
                    "parallel_eval_loss": p_loss,
                    "max_abs_param_diff": max_abs,
                    "bit_exact": site_bit,
                    "tol_match": site_tol,
                }
            )

    serial_ms = float(serial["parallelism"]["wall_ms"])
    parallel_ms = float(parallel["parallelism"]["wall_ms"])
    speedup = (serial_ms / parallel_ms) if parallel_ms > 0 else None

    return seal(
        {
            "schema": PARALLEL_BENCH_SCHEMA,
            "recorded_at": _utc_now(),
            "unit": "site",
            "sites": list(target),
            "threads_per_worker_pinned": tpw,
            "serial_wall_ms": serial_ms,
            "parallel_wall_ms": parallel_ms,
            "speedup": speedup,
            "parallel_budget": parallel["parallelism"],
            "serial_budget": serial["parallelism"],
            "bit_exact_match": bit_exact,
            "tolerance_match": tol_match,
            "atol": atol,
            "rtol": rtol,
            "mismatches": mismatches,
            "serial_digests": {
                s: serial["site_results"][s]["state_digest"] for s in target
            },
            "parallel_digests": {
                s: parallel["site_results"][s]["state_digest"] for s in target
            },
            "data_kind": "SYNTHETIC_SITE_PAIRED_ACTIVATION_FIXTURE",
            "capability_claim": False,
            "note": (
                "threads_per_worker pinned equal in serial and parallel so "
                "BLAS reduction order matches; bit-exact is the target, "
                "tolerance is documented fallback."
            ),
        }
    )


def compare_serial_parallel_arms(
    *,
    synth_cfg: SyntheticConfig | None = None,
    train_cfg: TrainConfig | None = None,
    arms: Sequence[str] | None = None,
    max_workers: int | None = None,
    atol: float = PARALLEL_MATCH_ATOL,
    rtol: float = PARALLEL_MATCH_RTOL,
) -> dict[str, Any]:
    """Prove parallel A–G arm training matches serial (same seed, pinned threads)."""

    synth_cfg = synth_cfg or SyntheticConfig(
        n_train=48, n_eval=12, seq_len=4, d_model=32, batch_size=8, seed=3
    )
    train_cfg = train_cfg or TrainConfig(epochs=15, lr=5e-3, device="cpu")
    target = list(arms) if arms is not None else list(TRAINABLE_ARMS)
    tpw = 1

    serial = train_ag_variants(
        synth_cfg=synth_cfg,
        train_cfg=train_cfg,
        arms=target,
        parallel=False,
        threads_per_worker=tpw,
        return_states=True,
        max_workers=1,
    )
    parallel = train_ag_variants(
        synth_cfg=synth_cfg,
        train_cfg=train_cfg,
        arms=target,
        parallel=True,
        threads_per_worker=tpw,
        return_states=True,
        max_workers=max_workers,
    )

    mismatches: list[dict[str, Any]] = []
    bit_exact = True
    tol_match = True
    for arm in target:
        s_dig = serial["arm_state_digests"][arm]
        p_dig = parallel["arm_state_digests"][arm]
        s_state = serial["_arm_states"][arm]
        p_state = parallel["_arm_states"][arm]
        site_bit = s_dig == p_dig
        site_tol = True
        max_abs = 0.0
        for k in s_state:
            diff = (s_state[k] - p_state[k]).abs().max().item()
            max_abs = max(max_abs, float(diff))
            if not torch.allclose(s_state[k], p_state[k], atol=atol, rtol=rtol):
                site_tol = False
        bit_exact = bit_exact and site_bit
        tol_match = tol_match and site_tol
        if not site_bit or not site_tol:
            mismatches.append(
                {
                    "arm": arm,
                    "serial_digest": s_dig,
                    "parallel_digest": p_dig,
                    "serial_eval_loss": serial["arm_results"][arm]["final_eval_loss"],
                    "parallel_eval_loss": parallel["arm_results"][arm]["final_eval_loss"],
                    "max_abs_param_diff": max_abs,
                    "bit_exact": site_bit,
                    "tol_match": site_tol,
                }
            )

    serial_ms = float(serial["parallelism"]["wall_ms"])
    parallel_ms = float(parallel["parallelism"]["wall_ms"])
    speedup = (serial_ms / parallel_ms) if parallel_ms > 0 else None

    return seal(
        {
            "schema": PARALLEL_BENCH_SCHEMA,
            "recorded_at": _utc_now(),
            "unit": "arm",
            "arms": list(target),
            "threads_per_worker_pinned": tpw,
            "serial_wall_ms": serial_ms,
            "parallel_wall_ms": parallel_ms,
            "speedup": speedup,
            "parallel_budget": parallel["parallelism"],
            "serial_budget": serial["parallelism"],
            "bit_exact_match": bit_exact,
            "tolerance_match": tol_match,
            "atol": atol,
            "rtol": rtol,
            "mismatches": mismatches,
            "data_kind": "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE",
            "capability_claim": False,
        }
    )


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
    p_syn.add_argument("--device", type=str, default="cpu")
    p_syn.add_argument("--out", type=Path, default=None)

    p_ag = sub.add_parser("train-ag", help="Train B..G variants + reject-rule wire")
    p_ag.add_argument("--epochs", type=int, default=30)
    p_ag.add_argument("--d-model", type=int, default=64)
    p_ag.add_argument("--n-train", type=int, default=96)
    p_ag.add_argument("--n-eval", type=int, default=24)
    p_ag.add_argument("--seed", type=int, default=1)
    p_ag.add_argument("--device", type=str, default="cpu")
    p_ag.add_argument("--out", type=Path, default=None)

    p_real = sub.add_parser(
        "fit-real",
        help="Fit on real paired path (fails closed if absent)",
    )
    p_real.add_argument("--paired", type=Path, required=True)
    p_real.add_argument("--out", type=Path, default=None)

    # Full latent V0 (PROTO_FRANKENSTEIN_V0) — delegates to frankenstein_latent_v0.
    p_lat = sub.add_parser(
        "latent-fixture-e2e",
        help="Full latent V0: learn+reverse+reject on real-shaped fixture",
    )
    p_lat.add_argument("--epochs-per-phase", type=int, default=6)
    p_lat.add_argument("--scaled", action="store_true")
    p_lat.add_argument("--device", type=str, default="cpu")
    p_lat.add_argument("--out", type=Path, default=None)

    p_lat_ag = sub.add_parser(
        "latent-train-ag",
        help="Full latent V0 A–G train + retention/reject wire",
    )
    p_lat_ag.add_argument("--epochs-per-phase", type=int, default=4)
    p_lat_ag.add_argument("--scaled", action="store_true")
    p_lat_ag.add_argument("--device", type=str, default="cpu")
    p_lat_ag.add_argument("--out", type=Path, default=None)

    p_ag.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="CPU process workers for independent A–G arms (default: min(arms, cpus))",
    )
    p_ag.add_argument(
        "--serial",
        action="store_true",
        help="Force serial arm training (still pins threads_per_worker)",
    )
    p_ag.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="BLAS/torch threads per worker (default: cpus // n_workers)",
    )

    p_sites = sub.add_parser(
        "train-sites",
        help="Train independent V0 bridge sites in parallel (synthetic fixture)",
    )
    p_sites.add_argument("--epochs", type=int, default=30)
    p_sites.add_argument("--d-model", type=int, default=64)
    p_sites.add_argument("--n-train", type=int, default=96)
    p_sites.add_argument("--n-eval", type=int, default=24)
    p_sites.add_argument("--seed", type=int, default=0)
    p_sites.add_argument("--max-workers", type=int, default=None)
    p_sites.add_argument("--serial", action="store_true")
    p_sites.add_argument("--threads-per-worker", type=int, default=None)
    p_sites.add_argument("--out", type=Path, default=None)

    p_bench = sub.add_parser(
        "bench-parallel",
        help="Serial vs parallel correctness + fixture-scale speedup (sites and/or arms)",
    )
    p_bench.add_argument(
        "--unit",
        choices=("sites", "arms", "both"),
        default="both",
    )
    p_bench.add_argument("--epochs", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=7)
    p_bench.add_argument("--max-workers", type=int, default=None)
    p_bench.add_argument("--out", type=Path, default=None)

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
            train_cfg=TrainConfig(epochs=args.epochs, device="cpu"),
            max_workers=args.max_workers,
            threads_per_worker=args.threads_per_worker,
            parallel=not args.serial,
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
                    "parallelism": doc.get("parallelism"),
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

    if args.command == "train-sites":
        doc = train_sites(
            synth_cfg=SyntheticConfig(
                n_train=args.n_train,
                n_eval=args.n_eval,
                d_model=args.d_model,
                seed=args.seed,
            ),
            epochs=args.epochs,
            max_workers=args.max_workers,
            threads_per_worker=args.threads_per_worker,
            parallel=not args.serial,
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
                    "sites_trained": doc["sites_trained"],
                    "all_learned": doc["all_learned"],
                    "all_reverse_ok": doc["all_reverse_ok"],
                    "parallelism": doc.get("parallelism"),
                    "gate_still_closed": doc.get("gate_still_closed"),
                    "capability_claim": doc["capability_claim"],
                    "seal_sha256": doc["seal_sha256"],
                    "out": str(args.out) if args.out else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if doc["all_learned"] and doc["all_reverse_ok"] else 2

    if args.command == "bench-parallel":
        reports: dict[str, Any] = {}
        if args.unit in ("sites", "both"):
            reports["sites"] = compare_serial_parallel_sites(
                synth_cfg=SyntheticConfig(
                    n_train=48,
                    n_eval=12,
                    seq_len=4,
                    d_model=32,
                    batch_size=8,
                    seed=args.seed,
                ),
                epochs=args.epochs,
                max_workers=args.max_workers,
            )
        if args.unit in ("arms", "both"):
            reports["arms"] = compare_serial_parallel_arms(
                synth_cfg=SyntheticConfig(
                    n_train=48,
                    n_eval=12,
                    seq_len=4,
                    d_model=32,
                    batch_size=8,
                    seed=args.seed,
                ),
                train_cfg=TrainConfig(epochs=args.epochs, lr=5e-3, device="cpu"),
                max_workers=args.max_workers,
            )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            # Strip non-JSON if any leaked
            args.out.write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        summary = {
            unit: {
                "bit_exact_match": r.get("bit_exact_match"),
                "tolerance_match": r.get("tolerance_match"),
                "speedup": r.get("speedup"),
                "serial_wall_ms": r.get("serial_wall_ms"),
                "parallel_wall_ms": r.get("parallel_wall_ms"),
                "threads_per_worker_pinned": r.get("threads_per_worker_pinned"),
                "n_workers": (r.get("parallel_budget") or {}).get("n_workers"),
                "mismatches": len(r.get("mismatches") or []),
            }
            for unit, r in reports.items()
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        ok = all(
            r.get("bit_exact_match") or r.get("tolerance_match")
            for r in reports.values()
        )
        return 0 if ok else 2

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

    if args.command == "latent-fixture-e2e":
        lv = _latent_v0()
        argv = ["fixture-e2e", "--device", args.device or "cpu"]
        argv += ["--epochs-per-phase", str(args.epochs_per_phase)]
        if args.scaled:
            argv.append("--scaled")
        if args.out:
            argv += ["--out", str(args.out)]
        return lv.main(argv)

    if args.command == "latent-train-ag":
        lv = _latent_v0()
        argv = ["train-ag", "--device", args.device or "cpu"]
        argv += ["--epochs-per-phase", str(args.epochs_per_phase)]
        if args.scaled:
            argv.append("--scaled")
        if args.out:
            argv += ["--out", str(args.out)]
        return lv.main(argv)

    raise TrainerError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
