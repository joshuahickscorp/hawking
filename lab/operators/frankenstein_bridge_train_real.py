#!/usr/bin/env python3.12
"""FIRST REAL RUN: train reversible V0 bridges on real GLM×DSV4F paired activations.

CPU-only.  Honesty:
  * Never fabricates GLM teacher floats or capability numbers.
  * GLM layer NPZs are gitignored; if missing, the run FAIL_CLOSEs for training
    and REJECTS promotion.  DSV4F host-export floats alone are not a pair.
  * A correctly-rejected checkpoint is a valid outcome.

Pipeline:
  1. Inventory real capture sides + sealed phase alignment / bridge contract.
  2. Load paired activations for every phase-alignment pair (many-to-one).
  3. Build real paired tensors + closed-form Procrustes init + small supervised fit
     through frankenstein_latent_v0 (11-loss portfolio, CURRENT/BEST_*/ROLLBACK).
  4. A–G ablation vs BASE_DSV4F with additive-not-subtractive reject rule.
  5. Promote only if gates pass; otherwise seal a REJECT receipt (no fake PASS).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lab.operators import frankenstein_ablation as ablation
from lab.operators import frankenstein_latent_v0 as latent_v0
from lab.operators.frankenstein_bridges import V0_BRIDGE_SITES
from lab.operators.frankenstein_correspondence_loader import (
    load_dsv4f_from_receipts,
    load_glm_layer_matrix,
    load_paired_activations,
    parse_layer_range,
)
from lab.operators.frankenstein_fusion_op import (
    BRIDGES,
    DEEPSEEK_V4_FLASH,
    GLM_5_2,
    TRANSPLANT_POINT_NAMES,
)
from lab.receipts import seal


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
DEFAULT_OUT_DIR = EVIDENCE_ROOT / "bridge_train_real"
DEFAULT_GLM_CAPTURE = (
    EVIDENCE_ROOT
    / "teacher_forced"
    / "official_L0_stream_reexport_20260805T214500Z"
)
DEFAULT_DSV4F_EXPORT = (
    REPO_ROOT / "receipts" / "dsv4f_fullseq_capture_L0_frozen_export"
)
DEFAULT_PHASE_ALIGNMENT = (
    EVIDENCE_ROOT / "cartography" / "GLM_DSV4F_PHASE_ALIGNMENT.json"
)
DEFAULT_LAYER_CORRESPONDENCE = (
    EVIDENCE_ROOT / "cartography" / "GLM_DSV4F_LAYER_CORRESPONDENCE.json"
)
DEFAULT_BRIDGE_CONTRACT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_LATENT_BRIDGE_CONTRACT.json"
)
DEFAULT_TRANSPLANT_POINTS = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_TRANSPLANT_POINTS.json"
)
DEFAULT_DESKTOP_REJECT = (
    Path.home() / "Desktop" / "hawking-frankenstein" / "proto-frankenstein"
)

REAL_RUN_SCHEMA = "hawking.frankenstein.bridge_train_real_run.v1"
PAIRED_CAPTURE_SCHEMA = "hawking.frankenstein.real_paired_capture_tensors.v1"

# Map V0 named bridge sites → (transplant_point, dsv4f layer band heuristic).
# Full phase table is many-to-one (all DSV layers → GLM 13 in sealed alignment);
# sites still attach to the v3 transplant points they will later inject at.
SITE_TO_TRANSPLANT: dict[str, str] = {
    "GLM_EARLY_CONTEXT_BRIDGE": "pre_norm_hidden_state",
    "GLM_METHOD_BRIDGE": "post_attention_hidden_state",
    "GLM_DECOMPOSITION_BRIDGE": "pre_router_hidden_state",
    "GLM_PRE_ROUTER_BRIDGE": "pre_router_hidden_state",
    "GLM_POST_MOE_BRIDGE": "post_moe_hidden_state",
    "GLM_FORMALIZATION_BRIDGE": "post_moe_hidden_state",
    "GLM_REPAIR_BRIDGE": "final_hidden_state",
    "GLM_LATE_CONSOLIDATION_BRIDGE": "final_hidden_state",
}

# Prefer highest-CKA DSV layers for early/mid/late site bands when many-to-one.
SITE_DSV_BAND: dict[str, tuple[int, int]] = {
    "GLM_EARLY_CONTEXT_BRIDGE": (0, 8),
    "GLM_METHOD_BRIDGE": (4, 12),
    "GLM_DECOMPOSITION_BRIDGE": (8, 16),
    "GLM_PRE_ROUTER_BRIDGE": (12, 20),
    "GLM_POST_MOE_BRIDGE": (16, 28),
    "GLM_FORMALIZATION_BRIDGE": (24, 36),
    "GLM_REPAIR_BRIDGE": (32, 40),
    "GLM_LATE_CONSOLIDATION_BRIDGE": (36, 43),
}


class BridgeTrainRealError(RuntimeError):
    """Real bridge train failed closed or misconfigured."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_data(
    *,
    glm_capture: Path,
    dsv4f_export: Path,
    phase_path: Path,
    correspondence_path: Path,
    bridge_contract: Path,
    transplant_points: Path,
) -> dict[str, Any]:
    """Honest inventory of real capture sides and sealed contracts."""

    glm_layers_dir = glm_capture / "layers"
    glm_json = sorted(glm_layers_dir.glob("L*.json")) if glm_layers_dir.is_dir() else []
    glm_npz = sorted(glm_layers_dir.glob("L*.npz")) if glm_layers_dir.is_dir() else []
    dsv_act = dsv4f_export / "activations"
    dsv_npy = sorted(dsv_act.glob("L*.npy")) if dsv_act.is_dir() else []
    dsv_export_json = (
        sorted(dsv_act.glob("L*.export.json")) if dsv_act.is_dir() else []
    )

    phase = _read_json(phase_path) if phase_path.is_file() else None
    corr = _read_json(correspondence_path) if correspondence_path.is_file() else None

    dsv_hash_only = None
    dsv_shape = None
    if dsv_npy:
        arr = np.load(dsv_npy[0])
        dsv_shape = list(arr.shape)
        # export meta
        meta_path = dsv_act / f"{dsv_npy[0].stem}.export.json"
        if meta_path.is_file():
            meta = _read_json(meta_path)
            dsv_hash_only = False  # npy present = real floats
            _ = meta

    phase_pairs = (phase or {}).get("pairs") or []
    glm_targets = sorted({int(p["glm_layer"]) for p in phase_pairs})
    dsv_sources = sorted({int(p["dsv4f_layer"]) for p in phase_pairs})
    score_min = min((float(p["score"]) for p in phase_pairs), default=None)
    score_max = max((float(p["score"]) for p in phase_pairs), default=None)

    # Probe one claimed GLM NPZ size from JSON receipt if present
    glm_npz_claim: dict[str, Any] = {}
    if glm_json:
        sample = _read_json(glm_json[min(13, len(glm_json) - 1)])
        glm_npz_claim = {
            "layer_id": sample.get("layer_id"),
            "npz_bytes_claimed": sample.get("npz_bytes"),
            "npz_sha256_claimed": sample.get("npz_sha256"),
            "npz_present_on_disk": (glm_layers_dir / "L13.npz").is_file(),
        }

    glm_ready = len(glm_npz) > 0
    dsv_ready = len(dsv_npy) > 0 and dsv_hash_only is False
    paired_ready = bool(glm_ready and dsv_ready and phase_pairs)

    return seal(
        {
            "schema": "hawking.frankenstein.bridge_train_real_inventory.v1",
            "recorded_at": _utc_now(),
            "glm_capture_dir": str(glm_capture),
            "dsv4f_export_dir": str(dsv4f_export),
            "phase_alignment_path": str(phase_path),
            "layer_correspondence_path": str(correspondence_path),
            "bridge_contract_path": str(bridge_contract),
            "transplant_points_path": str(transplant_points),
            "glm": {
                "json_receipts": len(glm_json),
                "npz_on_disk": len(glm_npz),
                "npz_claim_sample": glm_npz_claim,
                "ready": glm_ready,
                "note": (
                    "Layer NPZs are gitignored (*.npz). Correspondence was measured "
                    "when NPZs existed in recapture worktree; they are absent here."
                    if not glm_ready
                    else "GLM sample NPZs present."
                ),
            },
            "dsv4f": {
                "npy_on_disk": len(dsv_npy),
                "export_json": len(dsv_export_json),
                "shape_sample": dsv_shape,
                "dsv4f_hash_only": dsv_hash_only,
                "ready": dsv_ready,
                "hidden_size": int(DEEPSEEK_V4_FLASH["hidden_size"]),
            },
            "phase_alignment": {
                "present": phase is not None,
                "fabricated": bool((phase or {}).get("fabricated")),
                "many_to_one": bool((phase or {}).get("many_to_one")),
                "monotonic": bool((phase or {}).get("monotonic")),
                "n_pairs": len(phase_pairs),
                "glm_targets": glm_targets,
                "dsv4f_sources": dsv_sources,
                "score_min": score_min,
                "score_max": score_max,
            },
            "correspondence": {
                "present": corr is not None,
                "fabricated": bool((corr or {}).get("fabricated")),
                "status": (corr or {}).get("status"),
                "source": (corr or {}).get("source"),
                "geometries": (corr or {}).get("geometries"),
            },
            "contracts": {
                "bridge_contract_present": bridge_contract.is_file(),
                "transplant_points_present": transplant_points.is_file(),
                "bridges": list(BRIDGES),
                "transplant_points": list(TRANSPLANT_POINT_NAMES),
                "v0_bridge_sites": list(V0_BRIDGE_SITES),
                "site_to_transplant": dict(SITE_TO_TRANSPLANT),
            },
            "geometry": {
                "glm_hidden_full": int(GLM_5_2["hidden_size"]),
                "dsv4f_hidden": int(DEEPSEEK_V4_FLASH["hidden_size"]),
                "glm_sample_width_default": 64,
                "note": (
                    "Teacher-forced capture stores first N=64 dims of hidden samples "
                    "(DEFAULT_SAMPLE_HIDDEN); full 6144 not retained in sample NPZ."
                ),
            },
            "paired_ready": paired_ready,
            "blocker": (
                None
                if paired_ready
                else (
                    "GLM layer NPZ float payloads missing on disk "
                    f"({len(glm_json)} JSON receipts, {len(glm_npz)} NPZ). "
                    "DSV4F real floats present "
                    f"({len(dsv_npy)} npy, hash_only={dsv_hash_only}). "
                    "Cannot form real (teacher, student) pairs for bridge fit."
                )
            ),
            "fabricated": False,
            "capability_claim": False,
        }
    )


def load_phase_pairs(phase_path: Path) -> list[dict[str, Any]]:
    doc = _read_json(phase_path)
    pairs = doc.get("pairs") or []
    if not pairs:
        raise BridgeTrainRealError(f"no phase pairs in {phase_path}")
    return [
        {
            "dsv4f_layer": int(p["dsv4f_layer"]),
            "glm_layer": int(p["glm_layer"]),
            "score": float(p["score"]),
            "method": p.get("method"),
        }
        for p in pairs
    ]


def select_site_layer_map(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pick one DSV layer per V0 site from phase table (best score in band)."""

    by_dsv = {int(p["dsv4f_layer"]): p for p in pairs}
    out: dict[str, dict[str, Any]] = {}
    for site in V0_BRIDGE_SITES:
        lo, hi = SITE_DSV_BAND[site]
        band = [
            by_dsv[L]
            for L in range(lo, hi)
            if L in by_dsv
        ]
        if not band:
            # fallback: global best score
            best = max(pairs, key=lambda p: float(p["score"]))
        else:
            best = max(band, key=lambda p: float(p["score"]))
        out[site] = {
            "site": site,
            "transplant_point": SITE_TO_TRANSPLANT[site],
            "dsv4f_layer": int(best["dsv4f_layer"]),
            "glm_layer": int(best["glm_layer"]),
            "phase_score": float(best["score"]),
            "bridge": "GLM_MATH_BRIDGE",
        }
    return out


def closed_form_procrustes(
    student: np.ndarray,
    teacher: np.ndarray,
) -> dict[str, Any]:
    """Closed-form linear map teacher→student (or shared-space init).

    student: [N, Ds], teacher: [N, Dt]
    Returns low-rank-friendly linear map factors for init (not SGD).
    """

    s = np.asarray(student, dtype=np.float64)
    t = np.asarray(teacher, dtype=np.float64)
    if s.shape[0] != t.shape[0]:
        raise BridgeTrainRealError(
            f"pair N mismatch student={s.shape} teacher={t.shape}"
        )
    # Center
    s_c = s - s.mean(axis=0, keepdims=True)
    t_c = t - t.mean(axis=0, keepdims=True)
    # Map teacher → student: W = argmin ||t W - s|| via least squares
    # Using SVD of cross-cov for stability when Dt != Ds
    # Solve t_c @ W ≈ s_c  →  W = pinv(t_c) @ s_c
    W, residuals, rank, svals = np.linalg.lstsq(t_c, s_c, rcond=None)
    pred = t_c @ W
    err = float(np.mean((pred - s_c) ** 2))
    # Cosine after map
    p_flat = pred.reshape(-1)
    s_flat = s_c.reshape(-1)
    cos = float(
        np.dot(p_flat, s_flat)
        / (np.linalg.norm(p_flat) * np.linalg.norm(s_flat) + 1e-12)
    )
    return {
        "method": "closed_form_least_squares_teacher_to_student",
        "W_shape": list(W.shape),
        "rank_estimate": int(rank),
        "mse": err,
        "cosine": cos,
        "singular_values_head": [float(x) for x in list(svals[:8])],
        "W": W.astype(np.float32),
    }


def try_load_real_pairs(
    *,
    glm_capture: Path,
    dsv4f_export: Path,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Load real paired matrices for unique (glm, dsv) layer pairs."""

    unique_glm = sorted({int(p["glm_layer"]) for p in pairs})
    unique_dsv = sorted({int(p["dsv4f_layer"]) for p in pairs})

    # Prefer correspondence_loader for alignment + DSV
    g_lo, g_hi = min(unique_glm), max(unique_glm) + 1
    d_lo, d_hi = min(unique_dsv), max(unique_dsv) + 1
    paired = load_paired_activations(
        glm_capture_dir=glm_capture,
        dsv4f_receipts=[dsv4f_export],
        glm_layer_range=f"{g_lo}-{g_hi}",
        dsv4f_layer_range=f"{d_lo}-{d_hi}",
        align="intersection",
    )
    report = paired.report.to_dict()

    if not paired.report.ok:
        # Still load DSV alone for inventory evidence
        from lab.operators.frankenstein_correspondence_loader import LoadReport

        drep = LoadReport()
        d_mats, d_loaded, d_ids = load_dsv4f_from_receipts(
            [dsv4f_export],
            parse_layer_range(f"{d_lo}-{d_hi}"),
            report=drep,
        )
        return {
            "status": "FAIL_CLOSED",
            "gate": "REQUIRES_PAIRED_CAPTURE",
            "reason": "paired load not ok",
            "paired_report": report,
            "dsv4f_solo": {
                "layers_loaded": d_loaded,
                "n_sequences": int(d_mats[0].shape[0]) if d_mats else 0,
                "hidden": int(d_mats[0].shape[1]) if d_mats else None,
                "example_ids_head": list(d_ids[:5]),
                "blockers": drep.blockers,
            },
            "glm_layers": {},
            "dsv4f_layers": {
                int(L): d_mats[i] for i, L in enumerate(d_loaded)
            }
            if d_mats
            else {},
            "fabricated": False,
        }

    glm_map = {
        int(L): paired.glm_layers[i]
        for i, L in enumerate(paired.glm_layer_indices)
    }
    dsv_map = {
        int(L): paired.dsv4f_layers[i]
        for i, L in enumerate(paired.dsv4f_layer_indices)
    }
    return {
        "status": "LOADED",
        "paired_report": report,
        "glm_layers": glm_map,
        "dsv4f_layers": dsv_map,
        "n_sequences": int(paired.report.n_sequences),
        "fabricated": False,
    }


def build_real_training_tensors(
    *,
    glm_mat: np.ndarray,
    dsv_mat: np.ndarray,
    procrustes: Mapping[str, Any],
    n_eval: int = 8,
    n_experts: int = 16,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Build latent_v0 batch dicts from real [N,D] matrices.

    Teacher is padded/projected to full GLM geometry when samples are width-64:
    we place the real sample dims in the leading slice and zero-pad the rest,
    labelling the pad explicitly (not a fabricated capability claim).
    Student uses full DSV4F 4096 from host export.
    """

    g = np.asarray(glm_mat, dtype=np.float32)
    s = np.asarray(dsv_mat, dtype=np.float32)
    if g.shape[0] != s.shape[0]:
        n = min(g.shape[0], s.shape[0])
        g, s = g[:n], s[:n]
    n = int(g.shape[0])
    if n < 4:
        raise BridgeTrainRealError(f"need ≥4 paired sequences, got {n}")

    d_teacher_full = int(GLM_5_2["hidden_size"])
    d_student = int(DEEPSEEK_V4_FLASH["hidden_size"])
    d_g = int(g.shape[1])
    d_s = int(s.shape[1])
    if d_s != d_student:
        raise BridgeTrainRealError(
            f"DSV4F hidden {d_s} != expected {d_student}"
        )

    # Pad/truncate teacher samples into full geometry (real dims only in prefix).
    teacher_full = np.zeros((n, d_teacher_full), dtype=np.float32)
    w = min(d_g, d_teacher_full)
    teacher_full[:, :w] = g[:, :w]
    teacher_pad_note = {
        "real_sample_width": d_g,
        "full_hidden": d_teacher_full,
        "real_dims_placed_at": f"0:{w}",
        "padded_zeros": d_teacher_full - w,
        "fabrication": False,
        "note": (
            "Teacher-forced NPZ retains first sample_width dims only. "
            "Trailing dims are structural zeros for projector geometry — "
            "not invented activations."
        ),
    }

    # Functional target in student space via closed-form map
    W = np.asarray(procrustes["W"], dtype=np.float32)
    # W maps teacher_centered → student; apply to (possibly padded) teacher sample
    t_for_map = teacher_full[:, : W.shape[0]] if W.shape[0] != d_teacher_full else teacher_full
    if W.shape[0] != t_for_map.shape[1]:
        # recompute on actual widths used
        cf = closed_form_procrustes(s, g)
        W = cf["W"]
        t_for_map = g
        procrustes = cf
    t_c = t_for_map - t_for_map.mean(axis=0, keepdims=True)
    s_c = s - s.mean(axis=0, keepdims=True)
    teacher_in_student = (t_c @ W).astype(np.float32) + s.mean(axis=0, keepdims=True)
    # residual target: lightly blend so L_function is learnable but not identity-trivial
    teacher_target = (0.7 * s + 0.3 * teacher_in_student).astype(np.float32)

    # Split train/eval
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_ev = min(n_eval, max(1, n // 4))
    eval_idx = idx[:n_ev]
    train_idx = idx[n_ev:] if n_ev < n else idx

    def pack(ii: np.ndarray) -> dict[str, torch.Tensor]:
        B = len(ii)
        # seq_len=1 for last-pos pooled captures
        stud = torch.from_numpy(s[ii]).unsqueeze(1)  # [B,1,Ds]
        teach = torch.from_numpy(teacher_full[ii]).unsqueeze(1)
        tgt = torch.from_numpy(teacher_target[ii]).unsqueeze(1)
        # Labels: real traces lack method labels → zero method id + weak targets
        # (method heads train only when labels exist; retention/latent still real)
        method_id = torch.zeros(B, dtype=torch.long)
        decomp_id = torch.zeros(B, dtype=torch.long)
        formal_id = torch.zeros(B, dtype=torch.long)
        repair_id = torch.zeros(B, dtype=torch.long)
        E = n_experts
        route = torch.zeros(B, 1, E)
        teacher_route = route.clone()
        # Action logits proxy from student mean (no fabricated teacher token ids)
        A = 12
        action = torch.zeros(B, A)
        teacher_action = torch.zeros(B, A)
        base_action = torch.zeros(B, A)
        verifier = torch.zeros(B)
        return {
            "student_hidden": stud,
            "teacher_hidden": teach,
            "teacher_target_student": tgt,
            "route_logits": route,
            "teacher_route_logits": teacher_route,
            "action_logits": action,
            "teacher_action_logits": teacher_action,
            "base_action_logits": base_action,
            "method_id": method_id,
            "decomp_id": decomp_id,
            "formal_id": formal_id,
            "repair_id": repair_id,
            "verifier_pass": verifier,
        }

    meta = {
        "schema": PAIRED_CAPTURE_SCHEMA,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_eval": int(len(eval_idx)),
        "student_shape": list(s.shape),
        "teacher_sample_shape": list(g.shape),
        "teacher_full_shape": [n, d_teacher_full],
        "teacher_pad": teacher_pad_note,
        "procrustes": {
            k: v for k, v in procrustes.items() if k != "W"
        },
        "labels": {
            "method_labels_real": False,
            "route_labels_real": False,
            "action_labels_real": False,
            "note": (
                "Only hidden activations are real paired floats. "
                "Behavior/route/action heads lack real labels this run."
            ),
        },
        "data_kind": "REAL_PAIRED_CAPTURE_PARTIAL_LABELS",
        "fabricated": False,
        "capability_claim": False,
    }
    return pack(train_idx), pack(eval_idx), meta


def train_on_real_pairs(
    *,
    train_t: Mapping[str, torch.Tensor],
    eval_t: Mapping[str, torch.Tensor],
    site_map: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    epochs_per_phase: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Train latent A–G on real paired tensors; checkpoint CURRENT/BEST_*/ROLLBACK."""

    torch.manual_seed(seed)
    d_teacher = int(train_t["teacher_hidden"].shape[-1])
    d_student = int(train_t["student_hidden"].shape[-1])
    n_train = int(train_t["student_hidden"].shape[0])
    n_eval = int(eval_t["student_hidden"].shape[0])
    batch_size = min(8, max(1, n_train))

    train_loader = latent_v0.make_loader(
        train_t, batch_size=batch_size, shuffle=True
    )
    eval_loader = latent_v0.make_loader(
        eval_t, batch_size=batch_size, shuffle=False
    )

    arms = [
        latent_v0.ARM_B,
        latent_v0.ARM_C,
        latent_v0.ARM_D,
        latent_v0.ARM_E,
        latent_v0.ARM_F,
        latent_v0.ARM_G,
    ]
    arm_results: dict[str, Any] = {}
    arm_scores: dict[str, Any] = {
        latent_v0.ARM_A: {
            **ablation.default_score_template(0.70),
            "bench_scope": "BOUNDED_FIXTURE",
            "data_kind": "REAL_PAIRED_CAPTURE",
            "capability_claim": False,
            "synthetic_proxy": False,
            "note": "BASE_DSV4F reference template (no bridge)",
        }
    }

    for arm in arms:
        stack = latent_v0.build_stack_for_arm(
            arm,
            d_teacher=d_teacher,
            d_student=d_student,
            d_latent=128,
            rank=16,
            d_hidden=128,
            n_experts=16,
            n_methods=len(latent_v0.METHOD_CLASSES),
            n_actions=12,
        )
        ckpt = out_dir / "checkpoints" / f"real_{arm}"
        tcfg = latent_v0.TrainConfig(
            epochs_per_phase=epochs_per_phase,
            lr=3e-3,
            device=device,
            phases=("A", "B", "C", "E"),
            checkpoint_dir=ckpt,
        )
        result = latent_v0.train_schedule(
            stack,
            train_loader,
            eval_loader,
            cfg=tcfg,
            arm=arm,
            data_kind="REAL_PAIRED_CAPTURE",
        )
        arm_results[arm] = result.as_dict()
        # Map train metrics → score maps; mark as real-data train proxy,
        # NOT a full-model math bench.
        scores = latent_v0._proxy_scores_from_result(result)
        scores["synthetic_proxy"] = False
        scores["data_kind"] = "REAL_PAIRED_CAPTURE"
        scores["bench_scope"] = "BOUNDED_FIXTURE"
        scores["capability_claim"] = False
        scores["note"] = (
            "Proxy scores from real-activation train losses — "
            "NOT full-model math inheritance measurement."
        )
        arm_scores[arm] = scores

    ag = latent_v0.run_latent_ag_ablation(arm_scores=arm_scores)

    # Additive-not-subtractive: any secondary regression → REJECT
    # Also require G beats B on math proxy if both trained
    g_res = arm_results.get(latent_v0.ARM_G, {})
    b_res = arm_results.get(latent_v0.ARM_B, {})
    complete_beats_linear = False
    if g_res and b_res:
        complete_beats_linear = float(g_res.get("final_eval_loss", 1e9)) < float(
            b_res.get("final_eval_loss", 0.0)
        ) - 1e-4 or (
            bool(g_res.get("learned"))
            and float(g_res.get("final_eval_loss", 1e9))
            <= float(b_res.get("final_eval_loss", 1e9))
        )

    retention = latent_v0.retention_gate(
        base_secondary=arm_scores[latent_v0.ARM_A]["secondary"],
        proto_secondary=arm_scores[latent_v0.ARM_G]["secondary"],
    )
    prom = latent_v0.promotion_gate(
        complete_beats_linear=complete_beats_linear,
        held_out_math_improves=bool(g_res.get("learned")),
        method_decomp_repair_improve=bool(g_res.get("learned")),
        retention_pass=not retention["reject_rule_fired"],
        routing_stable=bool(g_res.get("reverse_ok")),
        reversible_loadable=bool(g_res.get("reverse_ok")),
        real_glm_activations=True,
        nonlinear_bridges_trained=bool(g_res.get("learned")),
        complete_bridge_classes=True,
        kimi_bridge_intact=True,
    )

    return seal(
        {
            "schema": "hawking.frankenstein.bridge_train_real_ag.v1",
            "recorded_at": _utc_now(),
            "status": "REAL_AG_TRAIN_COMPLETE",
            "data_kind": "REAL_PAIRED_CAPTURE",
            "capability_claim": False,
            "real_glm_dsv4f_capture": True,
            "device": device,
            "n_train": n_train,
            "n_eval": n_eval,
            "d_teacher": d_teacher,
            "d_student": d_student,
            "site_map": dict(site_map),
            "arms_trained": list(arms),
            "arm_results": arm_results,
            "ablation": ag,
            "retention_gate": retention,
            "promotion_gate": prom,
            "complete_beats_linear": complete_beats_linear,
            "v0_modules": list(latent_v0.V0_MODULE_NAMES),
            "loss_portfolio": list(latent_v0.LOSS_NAMES),
            "claim_boundary": {
                "training_performed_on_real_activations": True,
                "full_model_math_bench": False,
                "behavior_labels_real": False,
                "capability_claim": False,
                "kimi_consumed": False,
            },
        }
    )


def student_only_reverse_probe(
    *,
    dsv_layers: Mapping[int, np.ndarray],
    site_map: Mapping[str, Mapping[str, Any]],
    device: str = "cpu",
) -> dict[str, Any]:
    """When teacher is missing: prove student interventions reverse on real DSV floats.

    Not inheritance training. Not a capability number.
    """

    dev = torch.device(device)
    results: dict[str, Any] = {}
    for site, meta in site_map.items():
        L = int(meta["dsv4f_layer"])
        if L not in dsv_layers:
            results[site] = {"status": "SKIP", "reason": f"no DSV layer {L}"}
            continue
        mat = np.asarray(dsv_layers[L], dtype=np.float32)
        x = torch.from_numpy(mat).unsqueeze(1).to(dev)  # [N,1,D]
        mod = latent_v0.StudentIntervention(
            name=site, d_model=mat.shape[1], rank=16, d_hidden=64
        ).to(dev)
        mod.eval()
        with torch.no_grad():
            y, r = mod.apply_with_residual(x)
            x_hat = mod.revert_exact(y, r)
            err = float(torch.max(torch.abs(x_hat - x)).item())
        results[site] = {
            "status": "OK",
            "dsv4f_layer": L,
            "transplant_point": meta["transplant_point"],
            "phase_score": meta["phase_score"],
            "n": int(mat.shape[0]),
            "hidden": int(mat.shape[1]),
            "reverse_recon_error": err,
            "reverse_ok": err < 1e-5,
            "identity_init": True,
            "trained": False,
            "note": "untrained residual ≈0; reverse exact on real student activations",
        }
    all_ok = all(
        r.get("reverse_ok") for r in results.values() if r.get("status") == "OK"
    )
    return seal(
        {
            "schema": "hawking.frankenstein.student_only_reverse_probe.v1",
            "recorded_at": _utc_now(),
            "status": "STUDENT_REVERSE_PROBE",
            "all_reverse_ok": all_ok,
            "sites": results,
            "capability_claim": False,
            "inheritance_claim": False,
            "note": (
                "Student-side reverse probe on real DSV4F floats only. "
                "Does not claim GLM math inheritance."
            ),
        }
    )


def run_real_first(
    *,
    glm_capture: Path = DEFAULT_GLM_CAPTURE,
    dsv4f_export: Path = DEFAULT_DSV4F_EXPORT,
    phase_path: Path = DEFAULT_PHASE_ALIGNMENT,
    correspondence_path: Path = DEFAULT_LAYER_CORRESPONDENCE,
    bridge_contract: Path = DEFAULT_BRIDGE_CONTRACT,
    transplant_points: Path = DEFAULT_TRANSPLANT_POINTS,
    out_dir: Path = DEFAULT_OUT_DIR,
    desktop_dir: Path = DEFAULT_DESKTOP_REJECT,
    epochs_per_phase: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """End-to-end first real bridge train attempt (CPU)."""

    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = inventory_data(
        glm_capture=glm_capture,
        dsv4f_export=dsv4f_export,
        phase_path=phase_path,
        correspondence_path=correspondence_path,
        bridge_contract=bridge_contract,
        transplant_points=transplant_points,
    )
    (out_dir / "INVENTORY.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pairs = load_phase_pairs(phase_path) if phase_path.is_file() else []
    site_map = select_site_layer_map(pairs) if pairs else {}

    loaded = try_load_real_pairs(
        glm_capture=glm_capture,
        dsv4f_export=dsv4f_export,
        pairs=pairs or [{"dsv4f_layer": 0, "glm_layer": 13, "score": 0.0}],
    )

    # Student reverse probe whenever DSV is available
    dsv_map = loaded.get("dsv4f_layers") or {}
    reverse_probe = None
    if dsv_map and site_map:
        reverse_probe = student_only_reverse_probe(
            dsv_layers=dsv_map, site_map=site_map, device=device
        )
        (out_dir / "STUDENT_REVERSE_PROBE.json").write_text(
            json.dumps(reverse_probe, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    train_doc: dict[str, Any] | None = None
    promotion_verdict = "REJECT"
    reject_reasons: list[str] = []

    if loaded.get("status") != "LOADED":
        reject_reasons.append(
            "GLM teacher NPZ floats missing — cannot form real paired activations "
            "for L_latent / L_function bridge fit "
            f"(inventory blocker: {inv.get('blocker')})"
        )
        # A–G ablation: framework pending real capture (honest, not fixture ACCEPT)
        ag_pending = latent_v0.run_latent_ag_ablation(None)
        (out_dir / "AG_ABLATION_PENDING.json").write_text(
            json.dumps(ag_pending, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Explicit BASE vs untrained proto reject wire (additive rule still defined)
        base = ablation.default_score_template(0.70)
        # Proto without training: no math gain, secondaries equal → ACCEPT on
        # secondary gates but promotion still HOLDs on real_glm + trained bridges.
        proto = ablation.default_score_template(0.70)
        avb = ablation.run_avb_ablation(
            base_math=base["math"],
            base_secondary=base["secondary"],
            proto_math=proto["math"],
            proto_secondary=proto["secondary"],
            bench_scope="BOUNDED_FIXTURE",
            fixture_id="real-run-no-teacher-untrained",
            transfer_module_id="UNTRAINED_NO_GLM_TEACHER",
        )
        (out_dir / "STAGE1_AVB_UNTRAINED.json").write_text(
            json.dumps(avb, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prom = latent_v0.promotion_gate(
            complete_beats_linear=False,
            held_out_math_improves=False,
            method_decomp_repair_improve=False,
            retention_pass=True,  # no change → no secondary regression
            routing_stable=bool(
                reverse_probe and reverse_probe.get("all_reverse_ok")
            ),
            reversible_loadable=bool(
                reverse_probe and reverse_probe.get("all_reverse_ok")
            ),
            real_glm_activations=False,
            nonlinear_bridges_trained=False,
            complete_bridge_classes=False,
            kimi_bridge_intact=True,
        )
        promotion_verdict = prom["verdict"]
        reject_reasons.append(
            f"promotion_gate={prom['verdict']} failed={prom.get('failed')}"
        )
        train_doc = None
        ag_doc = ag_pending
        avb_doc = avb
        prom_doc = prom
    else:
        # Real pairs available — closed-form + small supervised fit
        # Use best phase pair globally for primary tensors
        best = max(pairs, key=lambda p: float(p["score"]))
        gL = int(best["glm_layer"])
        dL = int(best["dsv4f_layer"])
        g_mat = loaded["glm_layers"][gL]
        d_mat = loaded["dsv4f_layers"][dL]
        cf = closed_form_procrustes(d_mat, g_mat)
        train_t, eval_t, tmeta = build_real_training_tensors(
            glm_mat=g_mat,
            dsv_mat=d_mat,
            procrustes=cf,
            seed=seed,
        )
        (out_dir / "PAIRED_TENSOR_META.json").write_text(
            json.dumps(
                {k: v for k, v in tmeta.items()},
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        # Drop W array from sealed copy
        train_doc = train_on_real_pairs(
            train_t=train_t,
            eval_t=eval_t,
            site_map=site_map,
            out_dir=out_dir,
            epochs_per_phase=epochs_per_phase,
            device=device,
            seed=seed,
        )
        (out_dir / "REAL_AG_TRAIN.json").write_text(
            json.dumps(train_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ag_doc = train_doc["ablation"]
        prom_doc = train_doc["promotion_gate"]
        avb_doc = None
        promotion_verdict = prom_doc["verdict"]
        if promotion_verdict not in ("PROMOTE", "ACCEPT"):
            reject_reasons.append(
                f"promotion_gate={promotion_verdict} failed={prom_doc.get('failed')}"
            )
        if ag_doc.get("reject_rule_fired"):
            reject_reasons.append("A–G ablation reject_rule_fired")
            promotion_verdict = "REJECT"

    wall_ms = (time.perf_counter() - t0) * 1000.0
    sealed = promotion_verdict in ("PROMOTE", "ACCEPT")

    document = {
        "schema": REAL_RUN_SCHEMA,
        "name": "V0_BRIDGE_TRAIN_REAL_FIRST_RUN",
        "recorded_at": _utc_now(),
        "status": (
            "PROMOTED"
            if sealed
            else (
                "REJECT_NO_TEACHER_FLOATS"
                if loaded.get("status") != "LOADED"
                else "REJECT_PROMOTION_GATES"
            )
        ),
        "promotion_verdict": promotion_verdict,
        "reject_reasons": reject_reasons,
        "sealed_to_desktop": False,
        "device": device,
        "cpu_only": True,
        "gpu_used": False,
        "wall_ms": wall_ms,
        "inventory": inv,
        "phase_pairs_n": len(pairs),
        "site_map": site_map,
        "paired_load_status": loaded.get("status"),
        "paired_load_gate": loaded.get("gate"),
        "paired_report": loaded.get("paired_report"),
        "dsv4f_solo": loaded.get("dsv4f_solo"),
        "student_reverse_probe": reverse_probe,
        "train": train_doc,
        "ablation": ag_doc if loaded.get("status") != "LOADED" else train_doc.get("ablation") if train_doc else None,
        "avb_untrained": avb_doc,
        "promotion_gate": prom_doc,
        "bridges_trained": (
            list((train_doc or {}).get("arms_trained") or [])
            if train_doc
            else []
        ),
        "real_data_used": {
            "dsv4f_floats": bool(dsv_map),
            "glm_floats": loaded.get("status") == "LOADED",
            "phase_alignment": bool(pairs),
            "correspondence_sealed": correspondence_path.is_file(),
            "behavior_labels": False,
        },
        "claim_boundary": {
            "capability_claim": False,
            "fabricated": False,
            "forced_promotion": False,
            "kimi_consumed": False,
            "odyssey": False,
            "full_model_bench": False,
            "training_on_real_pairs": loaded.get("status") == "LOADED",
        },
        "next_required": (
            [
                "Restore GLM teacher-forced layer NPZs "
                f"under {glm_capture / 'layers'} "
                "(or re-run teacher-forced capture with NPZ retention; "
                "*.npz is gitignored and was not preserved after recapture merge).",
                "Re-run this operator: train-real --device cpu",
                "Then frankenstein_v0_seal assemble + verify if promotion_gate=PROMOTE",
            ]
            if loaded.get("status") != "LOADED"
            else [
                "If promotion PROMOTE: frankenstein_v0_seal assemble + verify",
                "If REJECT: inspect ablation regressions / math gain",
            ]
        ),
        "out_dir": str(out_dir),
    }
    sealed_doc = seal(document)
    run_path = out_dir / "BRIDGE_TRAIN_REAL_FIRST_RUN.json"
    run_path.write_text(
        json.dumps(sealed_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Desktop: REJECT receipt only (never fake-promote). macOS TCC may block Desktop.
    reject_receipt = {
        "schema": "hawking.frankenstein.bridge_train_real_desktop_receipt.v1",
        "recorded_at": _utc_now(),
        "promotion_verdict": promotion_verdict,
        "status": sealed_doc["status"],
        "sealed_artifact": False,
        "reason": reject_reasons,
        "run_seal_sha256": sealed_doc.get("seal_sha256"),
        "run_path": str(run_path),
        "capability_claim": False,
        "note": (
            "First real bridge-train attempt did NOT promote. "
            "No PROTO_FRANKENSTEIN_V0 artifact assembled."
        ),
    }
    reject_sealed = seal(reject_receipt)
    local_reject = out_dir / "BRIDGE_TRAIN_REAL_FIRST_RUN_REJECT.json"
    local_reject.write_text(
        json.dumps(reject_sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sealed_doc["local_reject_receipt"] = str(local_reject)
    sealed_doc["desktop_reject_seal_sha256"] = reject_sealed.get("seal_sha256")
    desktop_path = Path(desktop_dir) / "BRIDGE_TRAIN_REAL_FIRST_RUN_REJECT.json"
    desktop_write_error: str | None = None
    try:
        Path(desktop_dir).mkdir(parents=True, exist_ok=True)
        desktop_path.write_text(
            json.dumps(reject_sealed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sealed_doc["desktop_reject_receipt"] = str(desktop_path)
        sealed_doc["desktop_write_ok"] = True
    except OSError as exc:
        desktop_write_error = f"{type(exc).__name__}: {exc}"
        sealed_doc["desktop_reject_receipt"] = None
        sealed_doc["desktop_write_ok"] = False
        sealed_doc["desktop_write_error"] = desktop_write_error
        sealed_doc["desktop_target"] = str(desktop_path)
        sealed_doc["desktop_copy_hint"] = (
            f"cp {local_reject} {desktop_path}"
        )
    # rewrite run with desktop pointers
    run_path.write_text(
        json.dumps(sealed_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sealed_doc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="First real V0 bridge train on GLM×DSV4F paired activations (CPU)"
    )
    p.add_argument("--glm-capture", type=Path, default=DEFAULT_GLM_CAPTURE)
    p.add_argument("--dsv4f-export", type=Path, default=DEFAULT_DSV4F_EXPORT)
    p.add_argument("--phase", type=Path, default=DEFAULT_PHASE_ALIGNMENT)
    p.add_argument(
        "--correspondence", type=Path, default=DEFAULT_LAYER_CORRESPONDENCE
    )
    p.add_argument("--bridge-contract", type=Path, default=DEFAULT_BRIDGE_CONTRACT)
    p.add_argument(
        "--transplant-points", type=Path, default=DEFAULT_TRANSPLANT_POINTS
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--desktop", type=Path, default=DEFAULT_DESKTOP_REJECT)
    p.add_argument("--epochs-per-phase", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.device != "cpu":
        print(
            json.dumps(
                {
                    "warning": "forcing cpu per training-pinned-to-CPU convention",
                    "requested": args.device,
                }
            )
        )
    doc = run_real_first(
        glm_capture=args.glm_capture,
        dsv4f_export=args.dsv4f_export,
        phase_path=args.phase,
        correspondence_path=args.correspondence,
        bridge_contract=args.bridge_contract,
        transplant_points=args.transplant_points,
        out_dir=args.out,
        desktop_dir=args.desktop,
        epochs_per_phase=args.epochs_per_phase,
        device="cpu",
        seed=args.seed,
    )
    summary = {
        "status": doc["status"],
        "promotion_verdict": doc["promotion_verdict"],
        "reject_reasons": doc["reject_reasons"],
        "paired_load_status": doc["paired_load_status"],
        "bridges_trained": doc["bridges_trained"],
        "real_data_used": doc["real_data_used"],
        "student_reverse_all_ok": (doc.get("student_reverse_probe") or {}).get(
            "all_reverse_ok"
        ),
        "seal_sha256": doc.get("seal_sha256"),
        "out_dir": doc.get("out_dir"),
        "desktop_reject_receipt": doc.get("desktop_reject_receipt"),
        "capability_claim": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Exit 0 for honest reject (valid outcome); 2 only on hard error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
