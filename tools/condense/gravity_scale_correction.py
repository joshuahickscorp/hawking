#!/usr/bin/env python3.12
"""Per-expert OUTPUT-SCALE correction for sub-bit packed weights.

Measured failure mode (Qwen T5 diagnostics, one real routed expert): PQ packing preserves
DIRECTION (cosine 0.969) but destroys OUTPUT MAGNITUDE (relative_error 0.93, optimal scalar
gain 13.4 on gate_up). Direction is the expensive thing to keep; magnitude is one fp16 scalar
per expert-organ. This module fits that scalar in closed form.

Closed form, no iteration. Minimize over g:
    acts given:  || X Wo^T  -  g * X Wp^T ||_F   ->  g = <XWo^T, XWp^T> / <XWp^T, XWp^T>
    acts None:   || Wo - g Wp ||_F              ->  g = <Wo, Wp> / <Wp, Wp>
The weight-space form is the activation form under an isotropic input prior (E[XX^T] = I), so
the two agree when activations are white and diverge exactly as far as the input covariance is
anisotropic.

Rowwise is the same solve applied independently per output row: n_out scalars instead of one.
"""
from __future__ import annotations

import numpy as np

FP16_BITS = 16


def _outputs(w_orig: np.ndarray, w_packed: np.ndarray, acts: np.ndarray | None):
    """(A, B) matrices whose columns are per-output-row signals to align."""
    if w_orig.shape != w_packed.shape:
        raise ValueError(f"shape mismatch: {w_orig.shape} vs {w_packed.shape}")
    if acts is None:
        return w_orig.T.astype(np.float64), w_packed.T.astype(np.float64)
    if acts.ndim != 2 or acts.shape[1] != w_orig.shape[1]:
        raise ValueError(f"acts {acts.shape} incompatible with weight {w_orig.shape}")
    a = acts.astype(np.float64)
    return a @ w_orig.T.astype(np.float64), a @ w_packed.T.astype(np.float64)


def fit_scale(w_orig: np.ndarray, w_packed: np.ndarray, acts: np.ndarray | None = None) -> float:
    """Single scalar gain g minimizing ||Wo x - g (Wp x)|| over `acts` (or over W if acts is None)."""
    a, b = _outputs(w_orig, w_packed, acts)
    den = float(np.sum(b * b))
    return 1.0 if den <= 0.0 else float(np.sum(a * b) / den)


def fit_scale_rowwise(w_orig: np.ndarray, w_packed: np.ndarray,
                      acts: np.ndarray | None = None) -> np.ndarray:
    """One gain per OUTPUT row (len == w.shape[0]). Same least-squares solve, per column of A/B."""
    a, b = _outputs(w_orig, w_packed, acts)
    den = np.einsum("ij,ij->j", b, b)
    num = np.einsum("ij,ij->j", a, b)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 1.0).astype(np.float32)


def apply_scale(w_packed: np.ndarray, g) -> np.ndarray:
    """Scalar g, or a per-output-row vector g (broadcast down rows)."""
    g = np.asarray(g, dtype=np.float32)
    return (w_packed * (g if g.ndim == 0 else g[:, None])).astype(np.float32)


def scale_bits(n_experts: int, n_organs: int = 3) -> int:
    """One fp16 scalar per expert-organ."""
    return int(n_experts) * int(n_organs) * FP16_BITS


def scale_bits_rowwise(n_experts: int, rows_per_organ) -> int:
    """One fp16 scalar per output row per expert-organ. `rows_per_organ` is a per-organ row count
    sequence, e.g. (1536, 1536, 4096) for Qwen3 gate/up/down."""
    return int(n_experts) * int(sum(int(r) for r in rows_per_organ)) * FP16_BITS


def rel_error(w_orig: np.ndarray, w_hat: np.ndarray, acts: np.ndarray | None = None) -> float:
    """||Wo x - What x|| / ||Wo x||, over acts or over the weight matrix directly."""
    a, b = _outputs(w_orig, w_hat, acts)
    den = float(np.linalg.norm(a))
    return float("inf") if den == 0.0 else float(np.linalg.norm(a - b) / den)


def cosine(w_orig: np.ndarray, w_hat: np.ndarray, acts: np.ndarray | None = None) -> float:
    a, b = _outputs(w_orig, w_hat, acts)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den == 0.0 else float(np.sum(a * b) / den)


if __name__ == "__main__":  # tiny self-check
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    p = (w / 7.3 + 0.01 * rng.standard_normal(w.shape)).astype(np.float32)
    g = fit_scale(w, p)
    assert abs(g - 7.3) < 0.2, g
    assert rel_error(w, apply_scale(p, g)) < rel_error(w, p)
    print(f"ok g={g:.3f}")
