"""Rung interface for doctor6.

Sibling lane `lane-absorb-doctor` owns the v5 registry wiring into lab/operators.
This module is the ONLY place doctor6 calls rungs. If absorb entry points are
present we use them; otherwise we run the campaign-validated local codecs
(dual gravity + organ-ladder methodology) behind this interface.

Do not import tools/condense/doctor_restored/* (absorb-owned).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from lab.operators.ascension_dual_gravity_worker import (
    GROUP_BINARY,
    GROUP_UNIFORM,
    _activation_weighted_svd_factors,
    _binary_codec,
    _factor_codec,
    _hadamard_lattice_codec,
    _mean_row_cosine,
    _residual_codec,
    _uniform_codec,
)
from lab.operators.residual_compact_codec import encode_residual_compact
from lab.operators.hgravs01_adapter import (
    HGRAVS01_RUNGS,
    encode_hgravs01_rung,
    is_hgravs01_rung,
)

import os as _os

# Rung budget clamp. Historically hardcoded 0.98, which silently rebudgeted any
# act-SVD rung above it down to rank 64 even when the prescription target was 1.5.
_RUNG_BUDGET_BPW = float(_os.environ.get("HAWKING_RUNG_BUDGET_BPW", "0.98"))

# ---------------------------------------------------------------------------
# Absorb-lane entry points (optional). Marked STUB when absent.
# ---------------------------------------------------------------------------

ABSORB_ENTRYPOINTS: dict[str, str] = {
    "registry": "lab.operators.doctor_registry",
    "ladder": "lab.operators.doctor_ladder",
    "mixed_prec": "lab.operators.mixed_precision_alloc",
    "expert_alloc": "lab.operators.expert_alloc",
    "lowbit_qat": "lab.operators.lowbit_qat",
    "activation_weighted": "lab.operators.hgravs01_adapter",
}


def resolve_absorb_entry(name: str) -> tuple[Any | None, str]:
    """Return (module_or_none, status). status is 'live' or 'STUB:...'."""
    mod_name = ABSORB_ENTRYPOINTS.get(name)
    if not mod_name:
        return None, f"STUB: unknown entry {name!r}"
    try:
        import importlib

        return importlib.import_module(mod_name), "live"
    except Exception as exc:  # noqa: BLE001 — intentional soft resolve
        return None, f"STUB: {mod_name} not importable ({exc.__class__.__name__}: {exc})"


@dataclass(frozen=True)
class RungResult:
    name: str
    W_hat: np.ndarray
    payload_bytes: int
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Local codecs (campaign methodology; not the retired v5 CLI surface)
# ---------------------------------------------------------------------------

def awq_scale(W: np.ndarray, X_fit: np.ndarray, *, alpha: float = 0.5) -> np.ndarray:
    x_abs = np.mean(np.abs(X_fit), axis=0).astype(np.float64)
    s = np.power(np.maximum(x_abs, 1e-12), alpha)
    s = s / (np.exp(np.mean(np.log(s))) + 1e-12)
    return s.astype(np.float32)


def random_orthogonal(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n), dtype=np.float64)
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(np.float32)


def quant_binary(W: np.ndarray, *, group_size: int = GROUP_BINARY) -> tuple[np.ndarray, int]:
    codec = _binary_codec(W, group_size=group_size)
    return codec.reconstruction.astype(np.float32).reshape(W.shape), len(codec.payload)


def quant_uniform(W: np.ndarray, *, bits: int, group_size: int = GROUP_UNIFORM) -> tuple[np.ndarray, int]:
    codec = _uniform_codec(W, bits=bits, group_size=group_size)
    return codec.reconstruction.astype(np.float32).reshape(W.shape), len(codec.payload)


def quant_act_svd(
    W: np.ndarray, X_fit: np.ndarray, *, rank: int, bits: int
) -> tuple[np.ndarray, int]:
    left, right, _ = _activation_weighted_svd_factors(W, X_fit, rank=rank)
    left_body, left_rebuilt, _ = _factor_codec(left, bits=bits)
    right_body, right_rebuilt, _ = _factor_codec(right, bits=bits)
    rec = (left_rebuilt @ right_rebuilt).astype(np.float32).reshape(W.shape)
    return rec, len(left_body) + len(right_body) + 64


def quant_residual(W: np.ndarray, *, outlier_ratio: float = 0.05) -> tuple[np.ndarray, int]:
    codec = _residual_codec(W, outlier_ratio=outlier_ratio, group_size=GROUP_BINARY)
    return codec.reconstruction.astype(np.float32).reshape(W.shape), len(codec.payload)


def quant_residual_compact(
    W: np.ndarray,
    *,
    outlier_ratio: float = 0.05,
    index_mode: str = "rice",
    value_bits: int = 1,
    value_scale: str | None = None,
    group_size: int = GROUP_BINARY,
) -> tuple[np.ndarray, int]:
    """Binary + sparse residual with a compact index/value encoding.

    ``quant_residual`` keeps the original 48-bit (uint32 + fp16) packing.
    This sibling only changes storage. Selection stays global top-k by
    |residual|. ``value_bits < 16`` is a reported value-quantization.

    Default is the measured Q80 operating point: Rice-coded indices plus a
    1-bit (sign * RMS) residual. Pass ``value_bits=16`` for the incumbent
    reconstruction at the cheaper index, or ``value_bits=4`` for a near-free
    12-bit/outlier uniform quantizer.
    """

    codec = encode_residual_compact(
        W,
        outlier_ratio=outlier_ratio,
        group_size=group_size,
        index_mode=index_mode,
        value_bits=value_bits,
        value_scale=value_scale,
    )
    return codec.reconstruction.astype(np.float32).reshape(W.shape), len(codec.payload)


def quant_hadamard(W: np.ndarray, *, bits: int = 2) -> tuple[np.ndarray, int]:
    codec = _hadamard_lattice_codec(W, bits=bits, group_size=GROUP_BINARY)
    return codec.reconstruction.astype(np.float32).reshape(W.shape), len(codec.payload)


def quant_outlier_channel(
    W: np.ndarray,
    *,
    outlier_pct: float = 5.0,
    base_bits: int = 2,
    outlier_bits: int = 8,
) -> tuple[np.ndarray, int]:
    out_c, in_c = W.shape
    col_score = np.mean(np.abs(W), axis=0)
    n_out_ch = max(1, int(math.ceil(in_c * (outlier_pct / 100.0))))
    top = np.argpartition(col_score, -n_out_ch)[-n_out_ch:]
    mask = np.zeros(in_c, dtype=bool)
    mask[top] = True
    W_base = W.copy()
    W_base[:, mask] = 0.0
    W_out = W.copy()
    W_out[:, ~mask] = 0.0
    rec_base, bytes_base = quant_uniform(W_base, bits=base_bits)
    rec_out, bytes_out = quant_uniform(W_out, bits=outlier_bits)
    n_base = out_c * (in_c - n_out_ch)
    n_out = out_c * n_out_ch
    base_share = n_base / max(W.size, 1)
    out_share = n_out / max(W.size, 1)
    billed = int(bytes_base * base_share + bytes_out * out_share + n_out_ch * 4)
    return (rec_base + rec_out).astype(np.float32), billed


def _scheduled_lr(
    step: int,
    *,
    base_lr: float,
    steps: int,
    warmup: int,
    schedule: str,
) -> float:
    if steps <= 0:
        return float(base_lr)
    if step < warmup:
        return float(base_lr) * float(step + 1) / float(max(warmup, 1))
    t = (step - warmup) / float(max(1, steps - warmup))
    t = min(1.0, max(0.0, t))
    if schedule == "linear":
        return float(base_lr) * (1.0 - t)
    # cosine (default)
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def block_qat_weights(
    W: np.ndarray,
    X_fit: np.ndarray,
    *,
    bits: int = 2,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    warmup_steps: int | None = None,
    schedule: str = "cosine",
) -> np.ndarray:
    """BRECQ-lite STE with warmup + cosine/linear LR.

    Stops when FIT cosine (quantized reconstruction vs teacher) plateaus.
    Measured: at 1-bit this REGRESSES vs post-hoc binary — caller must drop.
    """
    import torch
    import torch.nn.functional as F

    if steps <= 0:
        return np.asarray(W, dtype=np.float32)
    dev = torch.device(device)
    W0 = torch.tensor(W, dtype=torch.float32, device=dev)
    X = torch.tensor(X_fit, dtype=torch.float32, device=dev)
    with torch.no_grad():
        Y = X @ W0.T
        y_norm = F.normalize(Y, dim=-1)
    qmax = 2 ** (bits - 1) - 1
    warmup = int(warmup_steps) if warmup_steps is not None else max(1, steps // 10)
    warmup = min(max(warmup, 0), max(steps - 1, 0))
    patience = max(16, steps // 5)

    def fq(w: torch.Tensor) -> torch.Tensor:
        s = (w.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-8)
        q = torch.clamp((w / s).round(), -qmax - 1, qmax) * s
        return w + (q - w).detach()

    def fit_cosine(w: torch.Tensor) -> float:
        with torch.no_grad():
            y_hat = F.normalize(X @ fq(w).T, dim=-1)
            return float((y_hat * y_norm).sum(dim=-1).mean().clamp(-1.0, 1.0).item())

    W_var = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([W_var], lr=lr)
    best_cos = fit_cosine(W_var)
    best_state = W_var.detach().clone()
    stall = 0
    for step in range(int(steps)):
        lr_t = _scheduled_lr(
            step, base_lr=lr, steps=int(steps), warmup=warmup, schedule=schedule
        )
        for pg in opt.param_groups:
            pg["lr"] = lr_t
        opt.zero_grad()
        loss = F.mse_loss(X @ fq(W_var).T, Y)
        loss.backward()
        opt.step()
        cos = fit_cosine(W_var)
        if cos > best_cos + 1e-6:
            best_cos = cos
            best_state = W_var.detach().clone()
            stall = 0
        else:
            stall += 1
            if stall >= patience and step >= warmup + patience:
                break
    return best_state.detach().cpu().numpy().astype(np.float32)


def selfcheck_qat_fit_ge_calib(*, device: str = "cpu") -> dict[str, Any]:
    """ONE assert-based self-check: QAT fit ≥ calib fit on a well-conditioned organ."""
    rng = np.random.default_rng(0)
    # Low-rank, well-scaled plant + enough rows: STE has a real target.
    left = rng.standard_normal((32, 6), dtype=np.float32) * 0.15
    right = rng.standard_normal((6, 48), dtype=np.float32) * 0.15
    W = left @ right
    X = rng.standard_normal((256, 48), dtype=np.float32)
    rec_calib, _ = quant_act_svd(W, X, rank=6, bits=3)
    rec_qat = block_qat_weights(
        W, X, bits=3, steps=160, lr=1e-3, device=device, schedule="cosine"
    )
    import torch

    qmax = 2 ** (3 - 1) - 1
    w = torch.as_tensor(rec_qat, dtype=torch.float32)
    s = (w.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-8)
    rec_qat_q = (torch.clamp((w / s).round(), -qmax - 1, qmax) * s).numpy()
    y = X @ W.T
    cos_calib = float(_mean_row_cosine(y, X @ rec_calib.T))
    cos_qat = float(_mean_row_cosine(y, X @ rec_qat_q.T))
    assert cos_qat + 1e-4 >= cos_calib, (
        f"QAT fit {cos_qat:.6f} < calib fit {cos_calib:.6f} "
        "on synthetic well-conditioned organ"
    )
    return {
        "ok": True,
        "qat_fit": cos_qat,
        "calib_fit": cos_calib,
        "bits": 3,
        "steps": 160,
        "device": device,
    }


def score_vs_incumbent(
    *,
    W: np.ndarray,
    X_hold: np.ndarray,
    W_hat: np.ndarray,
    W_incumbent: np.ndarray,
) -> dict[str, float]:
    """Score candidate against the INCUMBENT representation, never a constant-mean null.

    Live packer surplus-over-null was measured anti-correlated with surplus-over-
    baseline (corr ≈ -0.076 on N=8454). v6 only uses incumbent surplus.
    """
    y = X_hold @ W.T
    y_hat = X_hold @ W_hat.T
    y_inc = X_hold @ W_incumbent.T
    cos = float(_mean_row_cosine(y, y_hat))
    cos_inc = float(_mean_row_cosine(y, y_inc))
    return {
        "output_cosine": cos,
        "incumbent_cosine": cos_inc,
        "surplus_over_incumbent": cos - cos_inc,
        "beats_incumbent": bool(cos > cos_inc),
    }


# ---------------------------------------------------------------------------
# Named rungs — each returns reconstruction in original weight domain
# ---------------------------------------------------------------------------

RUNG_ORDER: tuple[str, ...] = (
    "incumbent_binary",
    "l0_calib",  # calibration is the capture itself; no weight change
    "l1_awq",
    "l1_rotation",
    "l2_mixed_prec",
    "l3_outlier_residual",
    "l4_block_qat",
    *HGRAVS01_RUNGS,  # activation-weighted low-rank family; rank is searchable
)


def apply_rung(
    name: str,
    W: np.ndarray,
    X_fit: np.ndarray,
    *,
    organ_key: str,
    sensitivity: float,
    seed: int,
    device: str = "cpu",
    qat_steps: int = 200,
    qat_lr: float = 1e-3,
    qat_bits: int = 2,
    prefer_budget: bool = True,
) -> RungResult:
    """Apply one named rung. Local codecs always; absorb modules consulted when present."""

    if name == "incumbent_binary":
        rec, nbytes = quant_binary(W)
        return RungResult(name, rec, nbytes, {"codec": "binary_g128", "role": "incumbent"})

    if is_hgravs01_rung(name):
        # Honest physical HGRAVS01 container (factors + scales + header).
        # Rank is parsed from the rung name and clamped to n_fit_rows.
        encoded = encode_hgravs01_rung(name, W, X_fit)
        meta = {
            "codec": (
                f"hgravs01_activation_weighted_low_rank_"
                f"r{encoded['achieved_rank']}_b{encoded['bits']}"
            ),
            "family": encoded["family"],
            "schema": encoded["schema"],
            "representation": encoded["representation"],
            "activation_weighted": True,
            "low_rank": True,
            "hgravs": True,
            "requested_rank": encoded["requested_rank"],
            "rank": encoded["achieved_rank"],
            "rank_clamped_to_n_fit": encoded["rank_clamped_to_n_fit"],
            "n_fit_rows": encoded["n_fit_rows"],
            "bits": encoded["bits"],
            "ledger": encoded["ledger"],
        }
        return RungResult(name, encoded["W_hat"], int(encoded["payload_bytes"]), meta)

    if name == "l0_calib":
        # Domain calib is the capture rows; identity on weights. Serve via under-budget act-SVD.
        rank = min(64, X_fit.shape[0], W.shape[0], W.shape[1])
        if prefer_budget and rank >= 1:
            rec, nbytes = quant_act_svd(W, X_fit, rank=max(1, rank), bits=3)
            codec = f"calib_actsvd_r{rank}_b3"
        else:
            rec, nbytes = quant_binary(W)
            codec = "calib_binary"
        return RungResult(name, rec, nbytes, {"codec": codec, "calib": "capture_fit_rows"})

    if name == "l1_awq":
        s = awq_scale(W, X_fit, alpha=0.5)
        W_awq = (W * s[None, :]).astype(np.float32)
        rank = min(128 if sensitivity >= 0.33 else 64, X_fit.shape[0], W.shape[0], W.shape[1])
        rec_s, nbytes = quant_act_svd(W_awq, X_fit, rank=max(1, rank), bits=3)
        # Undo channel scale for matmul with original X: W_hat = rec / s
        W_hat = rec_s / np.maximum(s[None, :], 1e-12)
        return RungResult(
            name,
            W_hat.astype(np.float32),
            nbytes,
            {"codec": f"awq_actsvd_r{rank}_b3", "awq_alpha": 0.5},
        )

    if name == "l1_rotation":
        s = awq_scale(W, X_fit, alpha=0.5)
        W_awq = (W * s[None, :]).astype(np.float32)
        R = random_orthogonal(W.shape[1], seed=seed ^ (hash(organ_key) & 0xFFFFFFFF))
        W_rot = (W_awq @ R).astype(np.float32)
        X_rot = X_fit @ R
        rank = min(128 if sensitivity >= 0.33 else 64, X_rot.shape[0], W_rot.shape[0], W_rot.shape[1])
        rec_rot, nbytes = quant_act_svd(W_rot, X_rot, rank=max(1, rank), bits=3)
        W_hat = (rec_rot @ R.T) / np.maximum(s[None, :], 1e-12)
        return RungResult(
            name,
            W_hat.astype(np.float32),
            nbytes,
            {"codec": f"awq_rot_actsvd_r{rank}_b3", "rotation": "qr_orthogonal"},
        )

    if name == "l2_mixed_prec":
        # Absorb mixed_precision_alloc if live; else sensitivity water-fill.
        mp_mod, mp_status = resolve_absorb_entry("mixed_prec")
        s = awq_scale(W, X_fit, alpha=0.5)
        W_awq = (W * s[None, :]).astype(np.float32)
        R = random_orthogonal(W.shape[1], seed=seed ^ (hash(organ_key) & 0xFFFFFFFF))
        W_rot = (W_awq @ R).astype(np.float32)
        X_rot = X_fit @ R
        if sensitivity >= 0.66:
            rank, bits, tag = 192, 3, "hot"
        elif sensitivity >= 0.33:
            rank, bits, tag = 128, 3, "warm"
        else:
            rank, bits, tag = 64, 3, "cold"
        # down_proj discrimination: escalate rank for high-sensitivity organs.
        if "down_proj" in organ_key and sensitivity >= 0.4:
            rank, bits, tag = 256, 4, "down_hot"
        rank = min(rank, X_rot.shape[0], W_rot.shape[0], W_rot.shape[1])
        rec_rot, nbytes = quant_act_svd(W_rot, X_rot, rank=max(1, rank), bits=bits)
        local = 8.0 * nbytes / max(W.size, 1)
        if prefer_budget and local > _RUNG_BUDGET_BPW:
            rank2 = max(1, min(64, rank))
            rec_rot, nbytes = quant_act_svd(W_rot, X_rot, rank=rank2, bits=3)
            tag = f"{tag}_rebudget"
            rank, bits = rank2, 3
        W_hat = (rec_rot @ R.T) / np.maximum(s[None, :], 1e-12)
        return RungResult(
            name,
            W_hat.astype(np.float32),
            nbytes,
            {
                "codec": f"mixed_actsvd_r{rank}_b{bits}_{tag}",
                "sensitivity": float(sensitivity),
                "absorb_mixed_prec": mp_status,
            },
        )

    if name == "l3_outlier_residual":
        s = awq_scale(W, X_fit, alpha=0.5)
        W_awq = (W * s[None, :]).astype(np.float32)
        R = random_orthogonal(W.shape[1], seed=seed ^ (hash(organ_key) & 0xFFFFFFFF))
        W_rot = (W_awq @ R).astype(np.float32)
        X_rot = X_fit @ R
        rank = 192 if ("down_proj" in organ_key or sensitivity >= 0.5) else 128
        if sensitivity < 0.33:
            rank = 64
        rank = min(rank, X_rot.shape[0], W_rot.shape[0], W_rot.shape[1])
        base, base_bytes = quant_act_svd(W_rot, X_rot, rank=max(1, rank), bits=3)
        residual = W_rot - base
        flat = residual.reshape(-1)
        # denser residual for down_proj (discriminating organ)
        frac = 0.04 if "down_proj" in organ_key else 0.02
        count = max(1, int(math.ceil(flat.size * frac)))
        idx = np.argpartition(np.abs(flat), -count)[-count:]
        vals = flat[idx].astype(np.float16)
        rec = base.copy().reshape(-1)
        rec[idx] += vals.astype(np.float32)
        rec = rec.reshape(W_rot.shape)
        resid_bytes = int(idx.astype(np.uint32).nbytes + vals.nbytes)
        nbytes = base_bytes + resid_bytes
        codec = f"actsvd_r{rank}_b3_sparse_resid_{frac:.0%}"
        if prefer_budget and 8.0 * nbytes / max(W.size, 1) > _RUNG_BUDGET_BPW:
            rec, nbytes = base, base_bytes
            codec = f"actsvd_r{rank}_b3_resid_dropped_budget"
        W_hat = (rec @ R.T) / np.maximum(s[None, :], 1e-12)
        return RungResult(
            name,
            W_hat.astype(np.float32),
            nbytes,
            {"codec": codec, "residual_frac": frac},
        )

    if name == "l4_block_qat":
        # MUST be measured per organ — at 1-bit STE regresses vs post-hoc binary.
        W_qat = block_qat_weights(
            W, X_fit, bits=qat_bits, steps=qat_steps, lr=qat_lr, device=device
        )
        # Pack healed weights with L3-style under-budget serve.
        inner = apply_rung(
            "l3_outlier_residual",
            W_qat,
            X_fit,
            organ_key=organ_key,
            sensitivity=sensitivity,
            seed=seed,
            prefer_budget=prefer_budget,
        )
        meta = dict(inner.meta)
        meta.update(
            {
                "block_qat_bits": qat_bits,
                "block_qat_steps": qat_steps,
                "block_qat_lr": qat_lr,
                "block_qat_bpw_overhead": 0.0,
                "source_changed": True,
                "inner_rung": "l3_outlier_residual",
            }
        )
        return RungResult(name, inner.W_hat, inner.payload_bytes, meta)

    raise ValueError(f"unknown rung {name!r}")


def list_rung_status() -> list[dict[str, Any]]:
    rows = []
    for name, mod in ABSORB_ENTRYPOINTS.items():
        _, status = resolve_absorb_entry(name)
        rows.append({"entry": name, "module": mod, "status": status})
    rows.append(
        {
            "entry": "local_codecs",
            "module": "lab.operators.doctor6.rungs",
            "status": "live",
            "rungs": list(RUNG_ORDER),
        }
    )
    rows.append(
        {
            "entry": "hgravs01",
            "module": "lab.operators.hgravs01_adapter",
            "status": "live",
            "family": "hgravs01",
            "schema": "hawking.gravity.activation_weighted_svd_low_rank.v1",
            "representation": "activation_weighted_svd_low_rank",
            "rungs": list(HGRAVS01_RUNGS),
            "rank_searchable": True,
        }
    )
    return rows
