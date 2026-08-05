#!/usr/bin/env python3.12
"""Layer-correspondence cartography: CKA, CCA/Procrustes, functional-intervention.

Maps functional phases (not layer ratios).  Runnable on synthetic paired
matrices now; real GLM×DSV4F correspondence numbers are gated on live
activations (REQUIRES_GLM_RUNTIME).  Emitters for
GLM_DSV4F_LAYER_CORRESPONDENCE.json + GLM_DSV4F_PHASE_ALIGNMENT.json seal
PENDING honestly when activations are absent — never fabricate scores.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators.frankenstein_fusion_op import DEEPSEEK_V4_FLASH, GLM_5_2
from lab.operators.frankenstein_gates import (
    REQUIRES_GLM_RUNTIME,
    fail_closed,
    gate_record,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
CARTOGRAPHY_DIR = EVIDENCE_ROOT / "cartography"
DEFAULT_LAYER_CORRESPONDENCE_PATH = (
    CARTOGRAPHY_DIR / "GLM_DSV4F_LAYER_CORRESPONDENCE.json"
)
DEFAULT_PHASE_ALIGNMENT_PATH = CARTOGRAPHY_DIR / "GLM_DSV4F_PHASE_ALIGNMENT.json"

CARTOGRAPHY_SCHEMA = "hawking.frankenstein.layer_correspondence.v1"
LAYER_CORRESPONDENCE_SCHEMA = "hawking.frankenstein.glm_dsv4f_layer_correspondence.v1"
PHASE_ALIGNMENT_SCHEMA = "hawking.frankenstein.glm_dsv4f_phase_alignment.v1"

# Monotonic many-to-one functional phases (PROTO_FRANKENSTEIN_V0 steer).
FUNCTIONAL_PHASES: tuple[str, ...] = (
    "lexical_context",
    "early_reasoning",
    "method_selection",
    "planning_decomposition",
    "tool_formal_prep",
    "repair_critique",
    "answer_proof_consolidation",
)


class CartographyError(RuntimeError):
    """Cartography failed closed."""


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two [N, D] matrices.

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    with column-centered X, Y.
    """

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise CartographyError("CKA expects 2D matrices")
    if a.shape[0] != b.shape[0]:
        raise CartographyError(
            f"CKA sample count mismatch: {a.shape[0]} vs {b.shape[0]}"
        )
    if a.shape[0] < 2:
        raise CartographyError("CKA needs at least 2 samples")
    a = _center(a)
    b = _center(b)
    hsic_xy = float(np.linalg.norm(a.T @ b, ord="fro") ** 2)
    hsic_xx = float(np.linalg.norm(a.T @ a, ord="fro") ** 2)
    hsic_yy = float(np.linalg.norm(b.T @ b, ord="fro") ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom <= 0.0:
        return 0.0
    return hsic_xy / denom


def procrustes_similarity(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Orthogonal Procrustes: min ||X R - Y|| over orthogonal R.

    Returns residual relative energy and the rotation.
    When dims differ, projects both to min(d_x, d_y) via leading PCA.
    """

    a = _center(np.asarray(x, dtype=np.float64))
    b = _center(np.asarray(y, dtype=np.float64))
    if a.shape[0] != b.shape[0]:
        raise CartographyError("Procrustes sample count mismatch")
    # Match feature dims via truncated SVD bases if needed.
    if a.shape[1] != b.shape[1]:
        k = min(a.shape[1], b.shape[1], a.shape[0])
        ua, _, _ = np.linalg.svd(a, full_matrices=False)
        ub, _, _ = np.linalg.svd(b, full_matrices=False)
        a = ua[:, :k]
        b = ub[:, :k]
    m = a.T @ b
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    r = u @ vt
    aligned = a @ r
    resid = float(np.linalg.norm(aligned - b) ** 2)
    total = float(np.linalg.norm(b) ** 2)
    rel = resid / total if total > 0 else 0.0
    # Correlation of flattened aligned vs target
    av = aligned.ravel()
    bv = b.ravel()
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(av, bv)[0, 1])
    return {
        "relative_residual_energy": rel,
        "correlation": corr,
        "rotation_shape": list(r.shape),
        "method": "orthogonal_procrustes",
    }


def cca_similarity(x: np.ndarray, y: np.ndarray, *, n_components: int = 8) -> dict[str, Any]:
    """Simple CCA via whitened SVD of cross-covariance (top canonical corrs)."""

    a = _center(np.asarray(x, dtype=np.float64))
    b = _center(np.asarray(y, dtype=np.float64))
    if a.shape[0] != b.shape[0]:
        raise CartographyError("CCA sample count mismatch")
    n = a.shape[0]
    # Regularized covariances
    eps = 1e-6
    c_xx = (a.T @ a) / max(n - 1, 1) + eps * np.eye(a.shape[1])
    c_yy = (b.T @ b) / max(n - 1, 1) + eps * np.eye(b.shape[1])
    c_xy = (a.T @ b) / max(n - 1, 1)
    # Whiten
    # Use eigh for SPD
    evals_x, evecs_x = np.linalg.eigh(c_xx)
    evals_y, evecs_y = np.linalg.eigh(c_yy)
    evals_x = np.clip(evals_x, eps, None)
    evals_y = np.clip(evals_y, eps, None)
    wx = evecs_x @ np.diag(1.0 / np.sqrt(evals_x)) @ evecs_x.T
    wy = evecs_y @ np.diag(1.0 / np.sqrt(evals_y)) @ evecs_y.T
    t = wx @ c_xy @ wy
    _, s, _ = np.linalg.svd(t, full_matrices=False)
    k = min(n_components, s.shape[0])
    corrs = [float(v) for v in s[:k]]
    return {
        "canonical_correlations": corrs,
        "mean_top_k": float(np.mean(corrs)) if corrs else 0.0,
        "n_components": k,
        "method": "regularized_cca_svd",
    }


def correspondence_matrix(
    glm_layers: Sequence[np.ndarray],
    dsv4f_layers: Sequence[np.ndarray],
    *,
    metric: str = "cka",
) -> np.ndarray:
    """Compute GLM-layer × DSV4F-layer similarity matrix.

    Each entry is a [N, D_side] activation matrix for that layer.
    """

    n_g = len(glm_layers)
    n_d = len(dsv4f_layers)
    mat = np.zeros((n_g, n_d), dtype=np.float64)
    for i, gx in enumerate(glm_layers):
        for j, dx in enumerate(dsv4f_layers):
            if metric == "cka":
                mat[i, j] = linear_cka(gx, dx)
            elif metric == "procrustes":
                mat[i, j] = 1.0 - procrustes_similarity(gx, dx)["relative_residual_energy"]
            elif metric == "cca":
                mat[i, j] = cca_similarity(gx, dx, n_components=4)["mean_top_k"]
            else:
                raise CartographyError(f"unknown metric {metric!r}")
    return mat


def functional_phase_map(
    matrix: np.ndarray,
    *,
    glm_layer_count: int | None = None,
    dsv4f_layer_count: int | None = None,
) -> dict[str, Any]:
    """Map functional phases by argmax correspondence — not layer-ratio.

    For each DSV4F layer, pick the GLM layer with highest similarity.
    Also report early/mid/late phase blocks.
    """

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise CartographyError("phase map expects 2D correspondence matrix")
    n_g, n_d = m.shape
    glm_n = glm_layer_count or n_g
    dsv_n = dsv4f_layer_count or n_d
    pairs: list[dict[str, Any]] = []
    for j in range(n_d):
        i = int(np.argmax(m[:, j]))
        pairs.append(
            {
                "dsv4f_layer": j,
                "glm_layer": i,
                "score": float(m[i, j]),
                "method": "argmax_correspondence",
            }
        )

    def _phase(idx: int, n: int) -> str:
        if n <= 0:
            return "unknown"
        frac = idx / max(n - 1, 1)
        if frac < 1 / 3:
            return "early"
        if frac < 2 / 3:
            return "mid"
        return "late"

    phase_pairs = [
        {
            **p,
            "dsv4f_phase": _phase(p["dsv4f_layer"], dsv_n),
            "glm_phase": _phase(p["glm_layer"], glm_n),
        }
        for p in pairs
    ]
    return {
        "pairs": phase_pairs,
        "note": (
            "Functional phase map from similarity argmax; NOT a fixed layer-ratio "
            f"({glm_n}/{dsv_n})."
        ),
        "glm_layers": glm_n,
        "dsv4f_layers": dsv_n,
        "ratio_map_rejected": True,
    }


def causal_trace_scaffold(
    *,
    intervention_layer: int,
    target_metric: str = "output_kl",
) -> dict[str, Any]:
    """Interface for functional intervention / causal tracing (not executed here)."""

    return {
        "status": "SCAFFOLD_ONLY",
        "intervention_layer": intervention_layer,
        "target_metric": target_metric,
        "executed": False,
        "note": (
            "Causal tracing requires paired forward hooks + interventions. "
            "Real GLM side is REQUIRES_GLM_RUNTIME; student side needs forward hooks."
        ),
        "gate": gate_record(REQUIRES_GLM_RUNTIME, open_=False),
    }


def functional_intervention_sensitivity(
    source_layers: Sequence[np.ndarray],
    target_layers: Sequence[np.ndarray],
    *,
    noise_scale: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate how much corrupting source layer i moves target layer j (CKA drop).

    Unit-testable on synthetic paired matrices.  Does **not** claim causal
    tracing on real models; live intervention still needs forward hooks.
    """

    rng = np.random.default_rng(seed)
    n_s = len(source_layers)
    n_t = len(target_layers)
    # Baseline correspondence (CKA).
    baseline = correspondence_matrix(source_layers, target_layers, metric="cka")
    # For each source layer, replace with noise-corrupted version and remeasure.
    sensitivity = np.zeros((n_s, n_t), dtype=np.float64)
    for i in range(n_s):
        corrupted = list(source_layers)
        x = np.asarray(source_layers[i], dtype=np.float64)
        noise = rng.standard_normal(x.shape) * (noise_scale * (np.std(x) + 1e-8))
        corrupted[i] = x + noise
        mat = correspondence_matrix(corrupted, target_layers, metric="cka")
        # Sensitivity = drop in CKA (positive ⇒ layer i mattered for target j).
        sensitivity[i, :] = baseline[i, :] - mat[i, :]
    # Per-target: which source intervention hurts most.
    top_source = [int(np.argmax(sensitivity[:, j])) for j in range(n_t)]
    return {
        "method": "cka_drop_under_source_noise",
        "noise_scale": float(noise_scale),
        "baseline_cka_shape": list(baseline.shape),
        "sensitivity_shape": list(sensitivity.shape),
        "sensitivity": sensitivity.tolist(),
        "top_source_per_target": top_source,
        "mean_sensitivity": float(np.mean(sensitivity)),
        "note": (
            "Synthetic-matrix intervention proxy. Live causal tracing remains "
            "REQUIRES_GLM_RUNTIME + student forward hooks."
        ),
        "live_causal_trace": causal_trace_scaffold(intervention_layer=0),
    }


def monotonic_phase_alignment(
    matrix: np.ndarray,
    *,
    glm_layer_count: int | None = None,
    dsv4f_layer_count: int | None = None,
    phases: Sequence[str] = FUNCTIONAL_PHASES,
) -> dict[str, Any]:
    """Monotonic many-to-one phase map: DSV4F layers → GLM layers, non-decreasing.

    For each DSV4F layer j, pick the GLM layer with highest similarity among
    indices ≥ previous choice (isotonic argmax).  Then bucket DSV4F depth into
    named functional phases (not fixed layer ratios).
    """

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise CartographyError("phase alignment expects 2D correspondence matrix")
    n_g, n_d = m.shape
    glm_n = glm_layer_count or n_g
    dsv_n = dsv4f_layer_count or n_d

    pairs: list[dict[str, Any]] = []
    prev = 0
    for j in range(n_d):
        # Restrict to GLM indices >= prev for monotonicity.
        window = m[prev:, j]
        if window.size == 0:
            i = prev
            score = float(m[i, j]) if i < n_g else 0.0
        else:
            i = prev + int(np.argmax(window))
            score = float(m[i, j])
        pairs.append(
            {
                "dsv4f_layer": j,
                "glm_layer": i,
                "score": score,
                "method": "monotonic_argmax_correspondence",
            }
        )
        prev = i

    n_phases = len(phases)
    phase_blocks: list[dict[str, Any]] = []
    for p_idx, name in enumerate(phases):
        # Even depth buckets over DSV4F layers (phase is functional label on depth).
        start = int(round(p_idx * n_d / n_phases))
        end = int(round((p_idx + 1) * n_d / n_phases))
        block_pairs = [p for p in pairs if start <= p["dsv4f_layer"] < end]
        glm_lo = min((p["glm_layer"] for p in block_pairs), default=0)
        glm_hi = max((p["glm_layer"] for p in block_pairs), default=0)
        phase_blocks.append(
            {
                "phase": name,
                "phase_index": p_idx,
                "dsv4f_layer_range": [start, end],
                "glm_layer_range": [glm_lo, glm_hi + 1],
                "n_dsv4f_layers": max(0, end - start),
                "mean_score": (
                    float(np.mean([p["score"] for p in block_pairs]))
                    if block_pairs
                    else None
                ),
            }
        )

    # Verify monotonicity of the layer map.
    glm_seq = [p["glm_layer"] for p in pairs]
    monotonic = all(glm_seq[i] <= glm_seq[i + 1] for i in range(len(glm_seq) - 1))

    return {
        "pairs": pairs,
        "phase_blocks": phase_blocks,
        "phases": list(phases),
        "monotonic": monotonic,
        "many_to_one": True,
        "ratio_map_rejected": True,
        "note": (
            "Monotonic many-to-one phase alignment from similarity; NOT a fixed "
            f"layer-ratio ({glm_n}/{dsv_n})."
        ),
        "glm_layers": glm_n,
        "dsv4f_layers": dsv_n,
    }


def build_correspondence_report(
    glm_layers: Sequence[np.ndarray] | None,
    dsv4f_layers: Sequence[np.ndarray] | None,
    *,
    metric: str = "cka",
    source: str = "synthetic",
    glm_runtime_available: bool = False,
) -> dict[str, Any]:
    """Seal a cartography report.  Real GLM capture fails closed without runtime."""

    if source == "live_glm" and not glm_runtime_available:
        closed = fail_closed(
            REQUIRES_GLM_RUNTIME,
            stage="4_layer_cartography",
            operation="build_correspondence_report",
        )
        return seal(
            {
                "schema": CARTOGRAPHY_SCHEMA,
                **closed,
                "metric": metric,
                "source": source,
            }
        )

    if glm_layers is None or dsv4f_layers is None:
        raise CartographyError("glm_layers and dsv4f_layers required for synthetic/live run")

    matrix = correspondence_matrix(glm_layers, dsv4f_layers, metric=metric)
    phases = functional_phase_map(
        matrix,
        glm_layer_count=int(GLM_5_2["num_hidden_layers"]),
        dsv4f_layer_count=int(DEEPSEEK_V4_FLASH["num_hidden_layers"]),
    )
    mono = monotonic_phase_alignment(
        matrix,
        glm_layer_count=int(GLM_5_2["num_hidden_layers"]),
        dsv4f_layer_count=int(DEEPSEEK_V4_FLASH["num_hidden_layers"]),
    )
    intervention = functional_intervention_sensitivity(glm_layers, dsv4f_layers)
    # Also compute multi-metric summary on diagonal-ish pairs if square-ish.
    sample_i = 0
    sample_j = 0
    multi = {
        "cka": linear_cka(glm_layers[sample_i], dsv4f_layers[sample_j]),
        "procrustes": procrustes_similarity(glm_layers[sample_i], dsv4f_layers[sample_j]),
        "cca": cca_similarity(glm_layers[sample_i], dsv4f_layers[sample_j]),
    }
    document = {
        "schema": CARTOGRAPHY_SCHEMA,
        "status": "OK",
        "source": source,
        "metric": metric,
        "matrix_shape": list(matrix.shape),
        "matrix": matrix.tolist(),
        "functional_phases": phases,
        "monotonic_phase_alignment": mono,
        "functional_intervention_sensitivity": intervention,
        "sample_pair_metrics": multi,
        "causal_trace": causal_trace_scaffold(intervention_layer=0),
        "geometries": {
            "glm_hidden": GLM_5_2["hidden_size"],
            "glm_layers": GLM_5_2["num_hidden_layers"],
            "dsv4f_hidden": DEEPSEEK_V4_FLASH["hidden_size"],
            "dsv4f_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
        },
        "ratio_map_rejected": True,
        "fabricated": False,
        "claim_boundary": {
            "live_glm": bool(glm_runtime_available and source == "live_glm"),
            "synthetic_only": source == "synthetic",
            "functional_transfer_complete": False,
        },
    }
    return seal(document)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
            raise CartographyError(f"not a regular file: {path}")
        # Allow rewrite of PENDING scaffolding seals during development.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def emit_layer_correspondence(
    *,
    glm_layers: Sequence[np.ndarray] | None = None,
    dsv4f_layers: Sequence[np.ndarray] | None = None,
    source: str = "pending_activations",
    glm_runtime_available: bool = False,
    out_path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Seal GLM_DSV4F_LAYER_CORRESPONDENCE.json.

    With real paired activations → measured CKA/CCA/Procrustes/intervention.
    Without → PENDING, fabricated=False (never invent correspondence numbers).
    """

    path = Path(out_path) if out_path is not None else DEFAULT_LAYER_CORRESPONDENCE_PATH
    geometries = {
        "glm_hidden": GLM_5_2["hidden_size"],
        "glm_layers": GLM_5_2["num_hidden_layers"],
        "dsv4f_hidden": DEEPSEEK_V4_FLASH["hidden_size"],
        "dsv4f_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
    }

    if glm_layers is None or dsv4f_layers is None:
        closed = fail_closed(
            REQUIRES_GLM_RUNTIME,
            stage="4_layer_cartography",
            operation="emit_layer_correspondence",
        )
        doc = seal(
            {
                "schema": LAYER_CORRESPONDENCE_SCHEMA,
                "name": "GLM_DSV4F_LAYER_CORRESPONDENCE",
                "recorded_at": _utc_now(),
                "status": "PENDING_REAL_ACTIVATIONS",
                "gate": closed["gate"],
                "stage": closed["stage"],
                "operation": closed["operation"],
                "executed": False,
                "missing_infra": closed["missing_infra"],
                "note": closed["note"],
                "metrics": ["cka", "cca", "procrustes", "functional_intervention"],
                "matrix": None,
                "sample_pair_metrics": None,
                "functional_intervention_sensitivity": None,
                "geometries": geometries,
                "ratio_map_rejected": True,
                "source": source,
                "fabricated": False,
                "claim_boundary": {
                    "correspondence_numbers_measured": False,
                    "synthetic_only": False,
                    "awaiting": "paired GLM×DSV4F activations from capture lanes",
                },
            }
        )
    else:
        report = build_correspondence_report(
            glm_layers,
            dsv4f_layers,
            metric="cka",
            source=source,
            glm_runtime_available=glm_runtime_available,
        )
        # Also compute full multi-metric matrices when dims are small enough.
        multi_matrices = {
            "cka": report["matrix"],
            "procrustes": correspondence_matrix(
                glm_layers, dsv4f_layers, metric="procrustes"
            ).tolist(),
            "cca": correspondence_matrix(
                glm_layers, dsv4f_layers, metric="cca"
            ).tolist(),
        }
        intervention = report.get("functional_intervention_sensitivity")
        doc = seal(
            {
                "schema": LAYER_CORRESPONDENCE_SCHEMA,
                "name": "GLM_DSV4F_LAYER_CORRESPONDENCE",
                "recorded_at": _utc_now(),
                "status": "OK" if report.get("status") == "OK" else report.get("status"),
                "source": source,
                "metrics": ["cka", "cca", "procrustes", "functional_intervention"],
                "matrices": multi_matrices,
                "matrix": report.get("matrix"),
                "sample_pair_metrics": report.get("sample_pair_metrics"),
                "functional_intervention_sensitivity": intervention,
                "geometries": geometries,
                "ratio_map_rejected": True,
                "fabricated": False,
                "claim_boundary": {
                    "correspondence_numbers_measured": source != "synthetic",
                    "synthetic_only": source == "synthetic",
                    "live_glm": bool(glm_runtime_available and source == "live_glm"),
                },
                "cartography_seal_sha256": report.get("seal_sha256"),
            }
        )

    verify(doc, label="layer correspondence")
    if write:
        _atomic_write_json(path, doc)
        # Ephemeral path key is NOT part of the sealed body (verify without it).
        return {**doc, "_written_path": str(path)}
    return doc


def emit_phase_alignment(
    *,
    glm_layers: Sequence[np.ndarray] | None = None,
    dsv4f_layers: Sequence[np.ndarray] | None = None,
    source: str = "pending_activations",
    out_path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Seal GLM_DSV4F_PHASE_ALIGNMENT.json (monotonic many-to-one phases).

    PENDING without real (or synthetic test) activations — never fakes phase scores.
    """

    path = Path(out_path) if out_path is not None else DEFAULT_PHASE_ALIGNMENT_PATH
    phase_names = list(FUNCTIONAL_PHASES)

    if glm_layers is None or dsv4f_layers is None:
        closed = fail_closed(
            REQUIRES_GLM_RUNTIME,
            stage="4_layer_cartography",
            operation="emit_phase_alignment",
        )
        doc = seal(
            {
                "schema": PHASE_ALIGNMENT_SCHEMA,
                "name": "GLM_DSV4F_PHASE_ALIGNMENT",
                "recorded_at": _utc_now(),
                "status": "PENDING_REAL_ACTIVATIONS",
                "gate": closed["gate"],
                "stage": closed["stage"],
                "operation": closed["operation"],
                "executed": False,
                "missing_infra": closed["missing_infra"],
                "note": closed["note"],
                "phases": phase_names,
                "pairs": None,
                "phase_blocks": [
                    {
                        "phase": name,
                        "phase_index": i,
                        "dsv4f_layer_range": None,
                        "glm_layer_range": None,
                        "status": "PENDING",
                    }
                    for i, name in enumerate(phase_names)
                ],
                "monotonic": None,
                "many_to_one": True,
                "ratio_map_rejected": True,
                "source": source,
                "fabricated": False,
                "claim_boundary": {
                    "phase_scores_measured": False,
                    "awaiting": "paired activations + correspondence matrix",
                },
            }
        )
    else:
        matrix = correspondence_matrix(glm_layers, dsv4f_layers, metric="cka")
        mono = monotonic_phase_alignment(
            matrix,
            glm_layer_count=int(GLM_5_2["num_hidden_layers"]),
            dsv4f_layer_count=int(DEEPSEEK_V4_FLASH["num_hidden_layers"]),
        )
        doc = seal(
            {
                "schema": PHASE_ALIGNMENT_SCHEMA,
                "name": "GLM_DSV4F_PHASE_ALIGNMENT",
                "recorded_at": _utc_now(),
                "status": "OK",
                "source": source,
                "phases": phase_names,
                "pairs": mono["pairs"],
                "phase_blocks": mono["phase_blocks"],
                "monotonic": mono["monotonic"],
                "many_to_one": True,
                "ratio_map_rejected": True,
                "fabricated": False,
                "claim_boundary": {
                    "phase_scores_measured": source != "synthetic",
                    "synthetic_only": source == "synthetic",
                },
            }
        )

    verify(doc, label="phase alignment")
    if write:
        _atomic_write_json(path, doc)
        return {**doc, "_written_path": str(path)}
    return doc


def seal_cartography_emitters(
    *,
    glm_layers: Sequence[np.ndarray] | None = None,
    dsv4f_layers: Sequence[np.ndarray] | None = None,
    source: str = "pending_activations",
    out_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Emit both correspondence + phase alignment seals (PENDING or measured)."""

    base = Path(out_dir) if out_dir is not None else CARTOGRAPHY_DIR
    layer_path = base / "GLM_DSV4F_LAYER_CORRESPONDENCE.json"
    phase_path = base / "GLM_DSV4F_PHASE_ALIGNMENT.json"
    layer = emit_layer_correspondence(
        glm_layers=glm_layers,
        dsv4f_layers=dsv4f_layers,
        source=source,
        out_path=layer_path,
        write=write,
    )
    phase = emit_phase_alignment(
        glm_layers=glm_layers,
        dsv4f_layers=dsv4f_layers,
        source=source,
        out_path=phase_path,
        write=write,
    )
    return {
        "layer_correspondence": {
            "status": layer["status"],
            "seal_sha256": layer["seal_sha256"],
            "path": layer.get("_written_path"),
            "fabricated": layer.get("fabricated"),
        },
        "phase_alignment": {
            "status": phase["status"],
            "seal_sha256": phase["seal_sha256"],
            "path": phase.get("_written_path"),
            "fabricated": phase.get("fabricated"),
        },
    }


def synthetic_paired_layers(
    *,
    n_glm: int = 6,
    n_dsv: int = 4,
    n_samples: int = 64,
    d_glm: int = 32,
    d_dsv: int = 24,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Plant a known correspondence: dsv layer j ≈ glm layer map(j)."""

    rng = np.random.default_rng(seed)
    # Shared latent factors.
    latent = rng.standard_normal((n_samples, 8))
    glm: list[np.ndarray] = []
    for i in range(n_glm):
        w = rng.standard_normal((8, d_glm))
        noise = 0.05 * rng.standard_normal((n_samples, d_glm))
        glm.append(latent @ w + noise + i * 0.01)
    dsv: list[np.ndarray] = []
    # Map dsv j -> glm floor(j * n_glm / n_dsv)
    for j in range(n_dsv):
        src = min(int(j * n_glm / max(n_dsv, 1)), n_glm - 1)
        w = rng.standard_normal((glm[src].shape[1], d_dsv))
        # Project glm[src] into dsv dim with small noise → high CKA with src.
        base = glm[src] @ w
        dsv.append(base + 0.02 * rng.standard_normal(base.shape))
    return glm, dsv
