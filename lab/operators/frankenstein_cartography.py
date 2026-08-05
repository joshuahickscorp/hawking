#!/usr/bin/env python3.12
"""Layer-correspondence cartography: CKA, CCA/Procrustes, causal-trace scaffolding.

Maps functional phases (not layer ratios).  Runnable on synthetic paired
matrices now; real GLM side is REQUIRES_GLM_RUNTIME.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators.frankenstein_fusion_op import DEEPSEEK_V4_FLASH, GLM_5_2
from lab.operators.frankenstein_gates import (
    REQUIRES_GLM_RUNTIME,
    fail_closed,
    gate_record,
)
from lab.receipts import seal


CARTOGRAPHY_SCHEMA = "hawking.frankenstein.layer_correspondence.v1"


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
