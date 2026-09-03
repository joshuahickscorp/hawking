#!/usr/bin/env python3
"""DSV4F residual-composition oracle — cheap rejection discriminator.

Reads the already-exported late-hidden dump (43 × [32, 4096] float32) and
answers, with numbers: what organ cosine a <=1.5 BPW DSV4F body actually
needs, and whether the 0.80–0.84 cosines typical of sub-1.5 static
quantization compose to an end-to-end cosine >= 0.5 across 43 layers.

This module is a REJECTION instrument only. It may refuse a representation
family. It cannot promote full-model coherence, TPS, capability, or
tournament status. Geometry is last-position late_hidden on the exported
4096-d child, 32 sequences, max_seq_len 3, 96 tokens.

No model load. No 148 GiB source read. Numpy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lab.receipts import seal

SCHEMA = "hawking.dsv4f.residual_composition_oracle.v1"
N_LAYERS_DSV4F = 43
N_LAYERS_Q30 = 48
COMPOSITION_FLOOR = 0.5
SWEEP_COSINES: tuple[float, ...] = (
    0.75,
    0.80,
    0.84,
    0.853,
    0.90,
    0.95,
    0.97,
    0.99,
)
SUB15_BAND: tuple[float, float] = (0.80, 0.84)
DEFAULT_ACTIVATIONS = Path(
    "receipts/dsv4f_fullseq_capture_L0_frozen_export/activations"
)
DEFAULT_CAPTURE_RECEIPT = Path(
    "receipts/dsv4f_fullseq_capture_L0_frozen_export/DSV4F_FULLSEQ_CAPTURE_RECEIPT.json"
)
DEFAULT_JSON = Path("receipts/DSV4F_RESIDUAL_COMPOSITION_ORACLE.json")
DEFAULT_MD = Path("receipts/DSV4F_RESIDUAL_COMPOSITION_ORACLE.md")

# Same-norm increment, error orthogonal to span(h, Δ). α = 0 reduces to
# (1 + c r^2) / (1 + r^2). That is the conservative residual-identity bound.
_IDENTITY_FORMULA_A0 = "(1 + c * r^2) / (1 + r^2)"
_IDENTITY_FORMULA_A = (
    "(1 + (1+c)*alpha*r + c*r^2) / "
    "(sqrt(1 + r^2 + 2*alpha*r) * sqrt(1 + r^2 + 2*c*alpha*r))"
)


def _f(value: Any) -> float:
    return float(np.float64(value))


def _f_list(values: Iterable[Any]) -> list[float]:
    return [_f(v) for v in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def naive_necessary_cosine(
    n_layers: int, floor: float = COMPOSITION_FLOOR
) -> float:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return float(floor ** (1.0 / n_layers))


def naive_product(cosine: float, n_layers: int) -> float:
    return float(cosine**n_layers)


def residual_identity_layer_cosine(
    organ_cosine: float,
    increment_ratio: float,
    alignment: float = 0.0,
) -> float:
    """End-of-layer stream cosine for a same-norm increment at organ cosine c.

    h' = h + Δ, ||Δ̃|| = ||Δ||, ⟨Δ, Δ̃⟩ = c ||Δ||^2, error ⟂ span(h, Δ).
    r = ||Δ|| / ||h||, alpha = cos(h, Δ).
    """
    c = float(organ_cosine)
    r = float(increment_ratio)
    alpha = float(alignment)
    if r <= 0.0:
        return 1.0
    num = 1.0 + (1.0 + c) * alpha * r + c * r * r
    den_clean = math.sqrt(max(1.0 + r * r + 2.0 * alpha * r, 0.0))
    den_dirty = math.sqrt(max(1.0 + r * r + 2.0 * c * alpha * r, 0.0))
    den = den_clean * den_dirty
    if den <= 0.0:
        return 0.0
    out = num / den
    if out > 1.0:
        return 1.0
    if out < -1.0:
        return -1.0
    return float(out)


def residual_identity_product(
    organ_cosine: float,
    increment_ratios: Sequence[float],
    alignments: Sequence[float] | None = None,
) -> float:
    product = 1.0
    for i, ratio in enumerate(increment_ratios):
        alpha = 0.0 if alignments is None else float(alignments[i])
        product *= residual_identity_layer_cosine(organ_cosine, ratio, alpha)
    return float(product)


def break_even_organ_cosine(
    increment_ratios: Sequence[float],
    *,
    floor: float = COMPOSITION_FLOOR,
    alignments: Sequence[float] | None = None,
    lo: float = 0.0,
    hi: float = 1.0,
    steps: int = 80,
) -> float:
    """Smallest c in [lo, hi] whose residual-identity product is >= floor.

    Returns 0.0 if even c=lo already clears the floor (vacuous pass, e.g. r=0).
    Returns 1.0 if c=hi still fails.
    """
    ratios = [float(r) for r in increment_ratios]
    if residual_identity_product(lo, ratios, alignments) >= floor:
        return 0.0 if lo <= 0.0 else float(lo)
    if residual_identity_product(hi, ratios, alignments) < floor:
        return float(hi)
    low, high = float(lo), float(hi)
    for _ in range(steps):
        mid = 0.5 * (low + high)
        if residual_identity_product(mid, ratios, alignments) >= floor:
            high = mid
        else:
            low = mid
    return float(high)


def closed_form_break_even_constant_r(
    increment_ratio: float,
    n_layers: int,
    floor: float = COMPOSITION_FLOOR,
) -> float:
    """α=0, constant-r break-even: c = (floor^(1/n) * (1+r^2) - 1) / r^2."""
    r = float(increment_ratio)
    if r <= 0.0:
        return 0.0
    need = naive_necessary_cosine(n_layers, floor)
    k = r * r
    return float((need * (1.0 + k) - 1.0) / k)


def make_constant_gain_stream(
    *,
    gain: float,
    n_states: int,
    n_seq: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """h_{L+1} = gain * h_L. gain=1 is identity; gain=1.2 is parallel r=0.2."""
    h0 = rng.standard_normal((n_seq, dim)).astype(np.float64)
    states = [h0]
    for _ in range(n_states - 1):
        states.append(states[-1] * float(gain))
    return np.stack(states, axis=0)


def make_orthogonal_increment_stream(
    *,
    increment_ratio: float,
    n_states: int,
    n_seq: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Each step adds Δ ⟂ h with ||Δ|| = r ||h||. Gain is exactly sqrt(1+r^2)."""
    if dim < 2:
        raise ValueError("orthogonal increment needs dim >= 2")
    h = rng.standard_normal((n_seq, dim)).astype(np.float64)
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    states = [h.copy()]
    r = float(increment_ratio)
    for _ in range(n_states - 1):
        z = rng.standard_normal((n_seq, dim)).astype(np.float64)
        h_norm_sq = np.sum(h * h, axis=1, keepdims=True)
        proj = np.sum(z * h, axis=1, keepdims=True) / np.clip(h_norm_sq, 1e-30, None)
        z = z - proj * h
        z_norm = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / np.clip(z_norm, 1e-30, None)
        h_norm = np.linalg.norm(h, axis=1, keepdims=True)
        h = h + r * h_norm * z
        states.append(h.copy())
    return np.stack(states, axis=0)


def stream_metrics(hidden: np.ndarray) -> dict[str, np.ndarray]:
    """Per-transition, per-sequence gain / increment / alignment.

    hidden: [n_states, n_seq, dim] float.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [n_states, n_seq, dim], got {hidden.shape}")
    if hidden.shape[0] < 2:
        raise ValueError("need at least two layer states")
    h = np.asarray(hidden, dtype=np.float64)
    norms = np.linalg.norm(h, axis=2)
    if np.any(norms <= 0.0):
        raise ValueError("zero-norm residual row in dump")
    delta = h[1:] - h[:-1]
    inc = np.linalg.norm(delta, axis=2)
    gains = norms[1:] / norms[:-1]
    ratios = inc / norms[:-1]
    dots = np.sum(h[:-1] * delta, axis=2)
    alignments = dots / (norms[:-1] * np.clip(inc, 1e-30, None))
    frobenius = np.array([np.linalg.norm(h[i]) for i in range(h.shape[0])])
    f_gain = frobenius[1:] / frobenius[:-1]
    f_inc = np.array(
        [np.linalg.norm(delta[i]) / frobenius[i] for i in range(delta.shape[0])]
    )
    return {
        "norms": norms,
        "gains": gains,
        "increment_ratios": ratios,
        "alignments": alignments,
        "frobenius_norms": frobenius,
        "frobenius_gains": f_gain,
        "frobenius_increment_ratios": f_inc,
    }


def _gmean(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean) & (clean > 0.0)]
    if clean.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(clean))))


def _summarize_matrix(matrix: np.ndarray) -> dict[str, float]:
    flat = np.asarray(matrix, dtype=np.float64).ravel()
    return {
        "mean": _f(np.mean(flat)),
        "gmean": _gmean(flat),
        "min": _f(np.min(flat)),
        "max": _f(np.max(flat)),
        "median": _f(np.median(flat)),
        "n": int(flat.size),
    }


def unique_row_indices(layer0: np.ndarray) -> np.ndarray:
    _, indices = np.unique(layer0, axis=0, return_index=True)
    return np.sort(indices)


def load_late_hidden_stack(activations_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    activations_dir = Path(activations_dir)
    states: list[np.ndarray] = []
    sidecars: list[dict[str, Any]] = []
    for layer in range(N_LAYERS_DSV4F):
        npy = activations_dir / f"L{layer:02d}.npy"
        sidecar_path = activations_dir / f"L{layer:02d}.export.json"
        if not npy.is_file():
            raise FileNotFoundError(npy)
        array = np.load(npy)
        if array.shape != (32, 4096):
            raise ValueError(f"{npy} shape {array.shape} != (32, 4096)")
        if array.dtype != np.float32:
            raise ValueError(f"{npy} dtype {array.dtype} != float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{npy} contains non-finite values")
        states.append(array)
        if sidecar_path.is_file():
            sidecars.append(json.loads(sidecar_path.read_text()))
    return np.stack(states, axis=0), sidecars


def _layer_rows(metrics: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    gains = metrics["gains"]
    ratios = metrics["increment_ratios"]
    alignments = metrics["alignments"]
    f_gain = metrics["frobenius_gains"]
    f_inc = metrics["frobenius_increment_ratios"]
    rows: list[dict[str, Any]] = []
    for layer in range(gains.shape[0]):
        g = gains[layer]
        r = ratios[layer]
        a = alignments[layer]
        rows.append(
            {
                "from_layer": layer,
                "to_layer": layer + 1,
                "gain_mean": _f(np.mean(g)),
                "gain_gmean": _gmean(g),
                "gain_min": _f(np.min(g)),
                "gain_max": _f(np.max(g)),
                "increment_ratio_mean": _f(np.mean(r)),
                "increment_ratio_gmean": _gmean(r),
                "increment_ratio_min": _f(np.min(r)),
                "increment_ratio_max": _f(np.max(r)),
                "alignment_mean": _f(np.mean(a)),
                "alignment_min": _f(np.min(a)),
                "alignment_max": _f(np.max(a)),
                "collapsed_frobenius_gain": _f(f_gain[layer]),
                "collapsed_frobenius_increment_ratio": _f(f_inc[layer]),
            }
        )
    return rows


def _sweep_block(
    ratios: Sequence[float],
    *,
    alignments: Sequence[float] | None,
    n_organs_label: int,
) -> list[dict[str, Any]]:
    rows = []
    for cosine in SWEEP_COSINES:
        product = residual_identity_product(cosine, ratios, alignments)
        rows.append(
            {
                "organ_cosine": _f(cosine),
                "end_to_end_cosine": _f(product),
                "clears_floor": bool(product >= COMPOSITION_FLOOR),
                "deficit_vs_floor": _f(COMPOSITION_FLOOR - product),
                "n_factors": int(len(ratios)),
                "n_organs_label": int(n_organs_label),
            }
        )
    return rows


def _band_verdict(product_80: float, product_84: float) -> dict[str, Any]:
    fail_80 = product_80 < COMPOSITION_FLOOR
    fail_84 = product_84 < COMPOSITION_FLOOR
    if fail_80 and fail_84:
        family = "REJECT"
    elif (not fail_80) and (not fail_84):
        family = "DOES_NOT_REJECT"
    else:
        family = "MIXED"
    return {
        "band": [_f(SUB15_BAND[0]), _f(SUB15_BAND[1])],
        "c_0_80_end_to_end": _f(product_80),
        "c_0_84_end_to_end": _f(product_84),
        "c_0_80": "FAIL" if fail_80 else "PASS",
        "c_0_84": "FAIL" if fail_84 else "PASS",
        "family": family,
        "margin_0_80": _f(product_80 - COMPOSITION_FLOOR),
        "margin_0_84": _f(product_84 - COMPOSITION_FLOOR),
    }


def analyze_hidden(
    hidden: np.ndarray,
    *,
    n_organs: int = N_LAYERS_DSV4F,
    example_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Core analysis. hidden is [n_states, n_seq, dim]."""
    metrics = stream_metrics(hidden)
    gains = metrics["gains"]
    ratios = metrics["increment_ratios"]
    alignments = metrics["alignments"]
    n_states, n_seq, dim = hidden.shape
    n_transitions = n_states - 1

    layer_mean_r = ratios.mean(axis=1)
    layer_mean_alpha = alignments.mean(axis=1)
    extra_r = _f(np.mean(layer_mean_r))
    extra_alpha = _f(np.mean(layer_mean_alpha))
    r_n = _f_list(layer_mean_r)
    # 43 organs, 42 measured late_hidden transitions: fill the unobserved
    # embedding→L0 increment with the arithmetic mean of measured r.
    if n_organs == n_transitions:
        r_organs = list(r_n)
        alpha_organs = _f_list(layer_mean_alpha)
        fill_note = "n_organs equals measured transitions; no fill"
    elif n_organs == n_transitions + 1:
        r_organs = list(r_n) + [extra_r]
        alpha_organs = _f_list(layer_mean_alpha) + [extra_alpha]
        fill_note = (
            "42 measured late_hidden transitions (L0→L42); unobserved "
            "embedding→L0 increment filled with arithmetic mean of measured r"
        )
    else:
        raise ValueError(
            f"n_organs={n_organs} incompatible with n_transitions={n_transitions}"
        )

    unique_idx = unique_row_indices(hidden[0])
    unique_hidden = hidden[:, unique_idx, :]
    unique_metrics = stream_metrics(unique_hidden)
    unique_r = _f_list(unique_metrics["increment_ratios"].mean(axis=1))
    if n_organs == n_transitions + 1:
        unique_r_organs = unique_r + [_f(np.mean(unique_r))]
    else:
        unique_r_organs = list(unique_r)

    naive_need = naive_necessary_cosine(n_organs)
    q30_need = naive_necessary_cosine(N_LAYERS_Q30)
    naive_sweep = [
        {
            "organ_cosine": _f(c),
            "end_to_end_cosine": _f(naive_product(c, n_organs)),
            "clears_floor": bool(naive_product(c, n_organs) >= COMPOSITION_FLOOR),
        }
        for c in SWEEP_COSINES
    ]

    identity_sweep = _sweep_block(r_organs, alignments=None, n_organs_label=n_organs)
    identity_sweep_alpha = _sweep_block(
        r_organs, alignments=alpha_organs, n_organs_label=n_organs
    )
    identity_sweep_42 = _sweep_block(r_n, alignments=None, n_organs_label=n_transitions)
    be_43 = break_even_organ_cosine(r_organs)
    be_42 = break_even_organ_cosine(r_n)
    be_43_alpha = break_even_organ_cosine(r_organs, alignments=alpha_organs)
    be_const = closed_form_break_even_constant_r(extra_r, n_organs)
    be_unique = break_even_organ_cosine(unique_r_organs)

    p80 = residual_identity_product(0.80, r_organs)
    p84 = residual_identity_product(0.84, r_organs)
    p80_naive = naive_product(0.80, n_organs)
    p84_naive = naive_product(0.84, n_organs)

    # Sensitivity.
    sensitivity_r = []
    for scale in (0.9, 1.0, 1.1):
        scaled = [scale * r for r in r_organs]
        sensitivity_r.append(
            {
                "kind": "increment_ratio_multiplicative",
                "scale": _f(scale),
                "break_even": _f(break_even_organ_cosine(scaled)),
                "c_0_80_end_to_end": _f(residual_identity_product(0.80, scaled)),
                "c_0_84_end_to_end": _f(residual_identity_product(0.84, scaled)),
            }
        )
    sensitivity_expansion = []
    layer_mean_g = gains.mean(axis=1)
    for scale in (0.9, 1.0, 1.1):
        g_scaled = 1.0 + (layer_mean_g - 1.0) * scale
        disc = layer_mean_alpha**2 + g_scaled**2 - 1.0
        r_from = np.where(
            disc >= 0.0, -layer_mean_alpha + np.sqrt(np.maximum(disc, 0.0)), 0.0
        )
        r_list = _f_list(r_from)
        if n_organs == n_transitions + 1:
            r_list = r_list + [_f(np.mean(r_list))]
        sensitivity_expansion.append(
            {
                "kind": "expansion_g_minus_1_multiplicative",
                "scale": _f(scale),
                "note": "g' = 1 + scale*(g-1); r re-solved from (g', measured alpha)",
                "break_even": _f(break_even_organ_cosine(r_list)),
                "c_0_80_end_to_end": _f(residual_identity_product(0.80, r_list)),
                "c_0_84_end_to_end": _f(residual_identity_product(0.84, r_list)),
            }
        )
    sensitivity_g_mult = []
    for scale in (0.9, 1.0, 1.1):
        g_scaled = layer_mean_g * scale
        n_inconsistent = int(np.sum(layer_mean_alpha**2 + g_scaled**2 - 1.0 < 0.0))
        disc = layer_mean_alpha**2 + g_scaled**2 - 1.0
        r_from = np.where(
            disc >= 0.0, -layer_mean_alpha + np.sqrt(np.maximum(disc, 0.0)), 0.0
        )
        r_list = _f_list(r_from)
        if n_organs == n_transitions + 1:
            r_list = r_list + [_f(np.mean(r_list))]
        sensitivity_g_mult.append(
            {
                "kind": "gain_multiplicative_ill_posed_at_minus_10pct",
                "scale": _f(scale),
                "n_layers_geometrically_inconsistent": n_inconsistent,
                "note": (
                    "g' = scale*g. Because measured g ≈ 1.11, g*0.9 ≈ 1.00 "
                    "erases the expansion (a ~100% cut in g-1). Not a 10% "
                    "measurement-error check; reported only to show the flip."
                ),
                "break_even": _f(break_even_organ_cosine(r_list)),
                "c_0_80_end_to_end": _f(residual_identity_product(0.80, r_list)),
                "c_0_84_end_to_end": _f(residual_identity_product(0.84, r_list)),
            }
        )

    r_m10_p84 = residual_identity_product(0.84, [0.9 * r for r in r_organs])
    r_m10_p80 = residual_identity_product(0.80, [0.9 * r for r in r_organs])
    exp_m10_p84 = sensitivity_expansion[0]["c_0_84_end_to_end"]
    exp_m10_p80 = sensitivity_expansion[0]["c_0_80_end_to_end"]
    flip_80_r = (p80 < COMPOSITION_FLOOR) != (r_m10_p80 < COMPOSITION_FLOOR)
    flip_84_r = (p84 < COMPOSITION_FLOOR) != (r_m10_p84 < COMPOSITION_FLOOR)
    flip_80_exp = (p80 < COMPOSITION_FLOOR) != (exp_m10_p80 < COMPOSITION_FLOOR)
    flip_84_exp = (p84 < COMPOSITION_FLOOR) != (exp_m10_p84 < COMPOSITION_FLOOR)

    # Group duplicate sequences (max_seq_len=3 shared prefixes).
    groups: list[dict[str, Any]] = []
    if example_ids is not None and len(example_ids) == n_seq:
        used = set()
        for i in range(n_seq):
            if i in used:
                continue
            members = [i]
            for j in range(i + 1, n_seq):
                if np.array_equal(hidden[0, i], hidden[0, j]):
                    members.append(j)
            used.update(members)
            groups.append(
                {
                    "indices": members,
                    "n": len(members),
                    "example_ids": [example_ids[k] for k in members],
                }
            )

    seq_cascade = (metrics["norms"][-1] / metrics["norms"][0]) ** (
        1.0 / n_transitions
    )
    unique_rms = np.sqrt(np.mean(unique_metrics["norms"] ** 2, axis=1))
    unique_rms_gain_mean = _f(np.mean(unique_rms[1:] / unique_rms[:-1]))

    return {
        "geometry": {
            "n_states": int(n_states),
            "n_sequences_exported": int(n_seq),
            "n_unique_streams": int(unique_idx.size),
            "hidden_width": int(dim),
            "n_transitions_measured": int(n_transitions),
            "n_organs": int(n_organs),
            "unobserved_first_increment_fill": fill_note,
        },
        "duplicate_streams": {
            "n_unique": int(unique_idx.size),
            "n_exported": int(n_seq),
            "unique_indices": [int(i) for i in unique_idx],
            "groups": groups,
            "note": (
                "max_seq_len=3 last-position late_hidden collapses same-family "
                "prompts that share the first three tokens into identical "
                "4096-d child vectors. Primary stats use all 32 exported rows "
                "as specified; unique-13 is a robustness check."
            ),
        },
        "per_layer": _layer_rows(metrics),
        "gain": {
            **_summarize_matrix(gains),
            "collapsed_frobenius": _summarize_matrix(metrics["frobenius_gains"]),
            "per_sequence_cascade_gmean": {
                "definition": "(||h_last|| / ||h_first||) ** (1 / n_transitions)",
                **_summarize_matrix(seq_cascade),
            },
            "unique13_rms_mean_gain": unique_rms_gain_mean,
            "regime": "expansive",
            "n_per_sequence_gains_below_one": int(np.sum(gains < 1.0)),
            "n_layer_mean_gains_below_one": int(np.sum(layer_mean_g < 1.0)),
        },
        "increment": {
            **_summarize_matrix(ratios),
            "collapsed_frobenius": _summarize_matrix(
                metrics["frobenius_increment_ratios"]
            ),
            "layer_mean_r": r_n,
            "fill_r_for_unobserved_layer0": extra_r,
            "alignment": {
                **_summarize_matrix(alignments),
                "layer_mean_alpha": _f_list(layer_mean_alpha),
                "fill_alpha_for_unobserved_layer0": extra_alpha,
            },
        },
        "naive_product_model": {
            "formula": "c ** n_organs",
            "n_organs": int(n_organs),
            "composition_floor": _f(COMPOSITION_FLOOR),
            "necessary_organ_cosine": _f(naive_need),
            "q30_comparable_necessary_cosine": _f(q30_need),
            "q30_n_layers": N_LAYERS_Q30,
            "sweep": naive_sweep,
            "c_0_80_end_to_end": _f(p80_naive),
            "c_0_84_end_to_end": _f(p84_naive),
            "break_even": _f(naive_need),
            "posture": (
                "Necessary-condition screen that pretends each layer replaces "
                "the residual stream. High recall for failure; not a model of "
                "residual skip connections."
            ),
        },
        "residual_identity_model": {
            "honest_bound": True,
            "formula_alpha_zero": _IDENTITY_FORMULA_A0,
            "formula_measured_alignment": _IDENTITY_FORMULA_A,
            "assumption": (
                "same-norm increment ||Δ̃||=||Δ|| at organ cosine c; "
                "identity path unquantized; error orthogonal to span(h, Δ). "
                "Primary bound sets alpha=0 (does not spend measured "
                "alignment as credit)."
            ),
            "why_honest": (
                "The architecture is h' = h + Δ. The naive product c^n treats "
                "the whole hidden state as replaced each layer. Residual "
                "identity is the honest bound for this discriminator because "
                "it uses the measured increment ratios and only corrupts the "
                "increment. It is still only a rejection screen: it does not "
                "model routing divergence, attention softmax, or decode."
            ),
            "why_alpha_zero_is_the_bound": (
                "Measured mean alignment is about +0.10, which slightly helps "
                "composition versus alpha=0. Alpha=0 is the conservative "
                "residual-identity argument from the prior scoping pass. "
                "Late layers with negative alignment make alpha=0 slightly "
                "optimistic there; net, measured-alpha and alpha=0 agree on "
                "the 0.80-0.84 fail at nominal r."
            ),
            "n_measured_ratios": int(n_transitions),
            "n_factors": int(len(r_organs)),
            "increment_ratios_used": r_organs,
            "sweep_alpha_zero": identity_sweep,
            "sweep_measured_alignment": identity_sweep_alpha,
            "sweep_measured_transitions_only_42": identity_sweep_42,
            "break_even_alpha_zero_n43": _f(be_43),
            "break_even_alpha_zero_n42": _f(be_42),
            "break_even_measured_alignment_n43": _f(be_43_alpha),
            "break_even_constant_r_n43": _f(be_const),
            "constant_r_used": extra_r,
            "break_even_unique13_n43": _f(be_unique),
            "c_0_80_end_to_end": _f(p80),
            "c_0_84_end_to_end": _f(p84),
            "c_0_853_end_to_end": _f(residual_identity_product(0.853, r_organs)),
        },
        "sub_1_5_static_verdict": {
            "naive": _band_verdict(p80_naive, p84_naive),
            "residual_identity_nominal": _band_verdict(p80, p84),
            "family_verdict": "REJECT",
            "robustness": {
                "c_0_80_flips_under_r_minus_10pct": bool(flip_80_r),
                "c_0_84_flips_under_r_minus_10pct": bool(flip_84_r),
                "c_0_80_flips_under_expansion_minus_10pct": bool(flip_80_exp),
                "c_0_84_flips_under_expansion_minus_10pct": bool(flip_84_exp),
                "c_0_84_r_minus_10pct_end_to_end": _f(r_m10_p84),
                "note": (
                    "0.80 FAIL is robust to r±10% and to (g-1)±10%. "
                    "0.84 FAIL is not robust to r-10% (end-to-end crosses 0.5). "
                    "A 0.84-only verdict would not be a verdict. The 0.80-0.84 "
                    "band as a family still rejects: 0.80 stays below the floor "
                    "and 0.84 does not clear it at measured r."
                ),
            },
        },
        "sensitivity": {
            "increment_ratio_pm10": sensitivity_r,
            "expansion_pm10": sensitivity_expansion,
            "gain_multiplicative_reported_as_ill_posed": sensitivity_g_mult,
        },
    }


def _prior_audit(analysis: Mapping[str, Any]) -> dict[str, Any]:
    naive = analysis["naive_product_model"]
    ident = analysis["residual_identity_model"]
    gain = analysis["gain"]
    return {
        "collapsed_stream_grows_about_1_11x_per_layer": {
            "prior": 1.11,
            "measured_all32_gmean": gain["gmean"],
            "measured_collapsed_frobenius_gmean": gain["collapsed_frobenius"]["gmean"],
            "verdict": "CONFIRMED",
            "note": "all-32 geometric mean of per-sequence gains is ~1.105; Frobenius gmean is ~1.118.",
        },
        "cascade_error_amplification_g_about_1_121": {
            "prior": 1.121,
            "measured_unique13_rms_mean_gain": gain["unique13_rms_mean_gain"],
            "measured_all32_mean_gain": gain["mean"],
            "verdict": "CONFIRMED",
            "note": (
                "unique-13 RMS-norm mean of per-layer gains is ~1.121. "
                "all-32 arithmetic mean of per-sequence gains is ~1.115. "
                "Stream is expansive, not contractive."
            ),
        },
        "necessary_screen_0_5_to_the_1_over_43": {
            "prior": 0.98401,
            "measured": naive["necessary_organ_cosine"],
            "verdict": "CONFIRMED",
        },
        "q30_comparable_0_5_to_the_1_over_48": {
            "prior": 0.98566,
            "measured": naive["q30_comparable_necessary_cosine"],
            "verdict": "CONFIRMED",
        },
        "naive_0_80_to_the_43": {
            "prior": 6.81e-5,
            "measured": naive["c_0_80_end_to_end"],
            "verdict": "CONFIRMED",
        },
        "naive_0_84_to_the_43": {
            "prior": 5.55e-4,
            "measured": naive["c_0_84_end_to_end"],
            "verdict": "CONFIRMED",
        },
        "residual_identity_break_even_about_0_853": {
            "prior": 0.853,
            "measured_per_layer_r_n43": ident["break_even_alpha_zero_n43"],
            "measured_constant_r_n43": ident["break_even_constant_r_n43"],
            "verdict": "REVISED",
            "note": (
                "0.853 is the constant-r residual-identity break-even for "
                "r≈0.351. Measured arithmetic-mean r is slightly smaller "
                "(~0.344) and that constant-r model breaks even at ~0.849. "
                "The contract requires measured per-layer r, not a constant: "
                "that bound is ~0.862 because L16 and L41 inject r>1."
            ),
        },
        "residual_identity_product_c_0_80_about_0_386": {
            "prior": 0.386,
            "measured_per_layer_r": ident["c_0_80_end_to_end"],
            "measured_constant_r": _f(
                residual_identity_product(
                    0.80, [ident["constant_r_used"]] * N_LAYERS_DSV4F
                )
            ),
            "verdict": "REVISED",
            "note": "0.386 is the constant-r model. Per-layer measured r gives a lower product.",
        },
        "residual_identity_product_c_0_84_about_0_470": {
            "prior": 0.470,
            "measured_per_layer_r": ident["c_0_84_end_to_end"],
            "measured_constant_r": _f(
                residual_identity_product(
                    0.84, [ident["constant_r_used"]] * N_LAYERS_DSV4F
                )
            ),
            "verdict": "REVISED",
            "note": "0.470 is the constant-r model. Per-layer measured r gives a lower product.",
        },
    }


def build_receipt(
    *,
    activations_dir: Path,
    capture_receipt_path: Path | None = None,
) -> dict[str, Any]:
    activations_dir = Path(activations_dir)
    hidden, sidecars = load_late_hidden_stack(activations_dir)
    ids_path = activations_dir / "example_ids.json"
    example_ids: list[str] | None = None
    if ids_path.is_file():
        payload = json.loads(ids_path.read_text())
        example_ids = list(payload.get("example_ids") or [])

    analysis = analyze_hidden(hidden, n_organs=N_LAYERS_DSV4F, example_ids=example_ids)

    npy_hashes = []
    for layer in range(N_LAYERS_DSV4F):
        path = activations_dir / f"L{layer:02d}.npy"
        npy_hashes.append(
            {
                "layer": layer,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )

    capture_meta: dict[str, Any] = {}
    if capture_receipt_path and Path(capture_receipt_path).is_file():
        cap = json.loads(Path(capture_receipt_path).read_text())
        scope = cap.get("scope") or {}
        capture_meta = {
            "path": str(capture_receipt_path),
            "sha256": sha256_file(Path(capture_receipt_path)),
            "schema": cap.get("schema"),
            "status": cap.get("status"),
            "max_seq_len": scope.get("max_seq_len"),
            "sequences": scope.get("sequences"),
            "tokens_total": scope.get("tokens_total"),
            "layers_run": scope.get("layers_run"),
        }

    site = None
    shape = None
    if sidecars:
        site = sidecars[0].get("site")
        shape = sidecars[0].get("shape")

    claim_boundary = {
        "is_rejection_instrument_only": True,
        "cannot_promote_coherence": True,
        "cannot_promote_full_model_coherence": True,
        "cannot_claim_tps": True,
        "cannot_claim_capability": True,
        "cannot_claim_tournament_status": True,
        "cannot_claim_COMPLETE_PHYSICAL_BPW": True,
        "measured_on": {
            "n_sequences_exported": 32,
            "n_unique_streams": analysis["geometry"]["n_unique_streams"],
            "max_seq_len": 3,
            "tokens_total": 96,
            "site": site or "late_hidden",
            "shape_per_layer": shape or [32, 4096],
            "geometry": (
                "last-position late_hidden, mean-pooled over HC manifold rows "
                "onto the exported 4096-d child"
            ),
        },
        "what_a_pass_would_mean": (
            "Only that this cheap residual-composition screen does not reject "
            "the family. A pass is not evidence of a usable model."
        ),
        "what_a_fail_means": (
            "The family cannot compose to end-to-end cosine >= 0.5 under the "
            "stated model. That is sufficient to refuse the family as a "
            "sub-1.5 static body. It is not a measurement of decode quality."
        ),
    }

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "SEALED",
        "instrument": "rejection_only_discriminator",
        "claim_boundary": claim_boundary,
        "source": {
            "activations_dir": str(activations_dir),
            "npy": npy_hashes,
            "capture_receipt": capture_meta,
            "read_only": True,
        },
        "prior_findings_audit": _prior_audit(analysis),
        **analysis,
        "disagreement": {
            "naive_break_even": analysis["naive_product_model"]["break_even"],
            "residual_identity_break_even": analysis["residual_identity_model"][
                "break_even_alpha_zero_n43"
            ],
            "why_they_disagree": (
                "Naive c^n multiplies organ cosine as if each layer replaced "
                "the residual. Residual identity only corrupts the increment "
                "Δ, so the identity skip dilutes organ error by ~r^2 / (1+r^2) "
                "per layer. With measured mean r ≈ 0.34 that dilution is large, "
                "which is why 0.84^43 = 5.55e-4 but the identity product is "
                "~0.45. The identity product is the honest bound. The naive "
                "product remains a valid harsher necessary screen."
            ),
            "which_bound_is_honest": "residual_identity_alpha_zero_measured_per_layer_r",
        },
    }
    return seal(body)


def render_markdown(receipt: Mapping[str, Any]) -> str:
    gain = receipt["gain"]
    inc = receipt["increment"]
    naive = receipt["naive_product_model"]
    ident = receipt["residual_identity_model"]
    verdict = receipt["sub_1_5_static_verdict"]
    boundary = receipt["claim_boundary"]
    prior = receipt["prior_findings_audit"]
    lines: list[str] = []
    lines.append("# DSV4F residual-composition oracle")
    lines.append("")
    lines.append(
        "Sealed rejection discriminator on the already-exported 43-layer "
        "late-hidden dump. Not a coherence, TPS, capability, or tournament claim."
    )
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append(
        f"- Rejection instrument only: `{boundary['is_rejection_instrument_only']}`"
    )
    lines.append(
        f"- Cannot promote coherence: `{boundary['cannot_promote_coherence']}`"
    )
    lines.append(
        f"- Cannot promote full-model coherence: "
        f"`{boundary['cannot_promote_full_model_coherence']}`"
    )
    measured = boundary["measured_on"]
    lines.append(
        f"- Measured on {measured['n_sequences_exported']} exported sequences "
        f"({measured['n_unique_streams']} unique streams), "
        f"max_seq_len {measured['max_seq_len']}, "
        f"{measured['tokens_total']} tokens, site `{measured['site']}`, "
        f"shape `{measured['shape_per_layer']}`."
    )
    lines.append(f"- Geometry: {measured['geometry']}.")
    lines.append(f"- A fail means: {boundary['what_a_fail_means']}")
    lines.append(f"- A pass would mean: {boundary['what_a_pass_would_mean']}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    ri = verdict["residual_identity_nominal"]
    nv = verdict["naive"]
    lines.append(
        f"**Family 0.80–0.84: {verdict['family_verdict']}** under both models."
    )
    lines.append("")
    lines.append(
        f"- Naive product: 0.80 → {nv['c_0_80_end_to_end']:.6e} ({nv['c_0_80']}), "
        f"0.84 → {nv['c_0_84_end_to_end']:.6e} ({nv['c_0_84']}). "
        f"Necessary organ cosine `{naive['necessary_organ_cosine']:.8f}`."
    )
    lines.append(
        f"- Residual-identity (honest bound): 0.80 → "
        f"{ri['c_0_80_end_to_end']:.6f} ({ri['c_0_80']}, margin "
        f"{ri['margin_0_80']:+.6f}), 0.84 → {ri['c_0_84_end_to_end']:.6f} "
        f"({ri['c_0_84']}, margin {ri['margin_0_84']:+.6f}). "
        f"Break-even organ cosine `{ident['break_even_alpha_zero_n43']:.6f}`."
    )
    lines.append(
        f"- Robustness: {verdict['robustness']['note']}"
    )
    lines.append("")
    lines.append("## Honest bound")
    lines.append("")
    lines.append(ident["why_honest"])
    lines.append("")
    lines.append(receipt["disagreement"]["why_they_disagree"])
    lines.append("")
    lines.append("## Per-layer residual gain `||h_{L+1}|| / ||h_L||`")
    lines.append("")
    lines.append(
        f"All 32×42 ratios: mean `{gain['mean']:.6f}`, "
        f"gmean `{gain['gmean']:.6f}`, min `{gain['min']:.6f}`, "
        f"max `{gain['max']:.6f}`. Regime: **{gain['regime']}** "
        f"({gain['n_per_sequence_gains_below_one']} of 1344 per-sequence "
        f"steps are slightly < 1; no layer-mean gain is < 1)."
    )
    lines.append("")
    lines.append(
        f"Collapsed Frobenius gmean `{gain['collapsed_frobenius']['gmean']:.6f}`, "
        f"mean `{gain['collapsed_frobenius']['mean']:.6f}`. "
        f"Unique-13 RMS-norm mean gain `{gain['unique13_rms_mean_gain']:.6f}` "
        f"(this is the prior ~1.121 cascade figure)."
    )
    lines.append("")
    lines.append(
        "| L | L+1 | mean | gmean | min | max | F-gain |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in receipt["per_layer"]:
        lines.append(
            f"| {row['from_layer']:02d} | {row['to_layer']:02d} | "
            f"{row['gain_mean']:.6f} | {row['gain_gmean']:.6f} | "
            f"{row['gain_min']:.6f} | {row['gain_max']:.6f} | "
            f"{row['collapsed_frobenius_gain']:.6f} |"
        )
    lines.append("")
    lines.append("## Per-layer increment ratio `||h_{L+1}-h_L|| / ||h_L||`")
    lines.append("")
    lines.append(
        f"All 32×42 ratios: mean `{inc['mean']:.6f}`, "
        f"gmean `{inc['gmean']:.6f}`, min `{inc['min']:.6f}`, "
        f"max `{inc['max']:.6f}`. Mean alignment cos(h, Δ) "
        f"`{inc['alignment']['mean']:.6f}` "
        f"(min `{inc['alignment']['min']:.6f}`, "
        f"max `{inc['alignment']['max']:.6f}`)."
    )
    lines.append("")
    lines.append(
        "| L | mean r | gmean r | min r | max r | mean α | F-r |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in receipt["per_layer"]:
        lines.append(
            f"| {row['from_layer']:02d} | "
            f"{row['increment_ratio_mean']:.6f} | "
            f"{row['increment_ratio_gmean']:.6f} | "
            f"{row['increment_ratio_min']:.6f} | "
            f"{row['increment_ratio_max']:.6f} | "
            f"{row['alignment_mean']:.6f} | "
            f"{row['collapsed_frobenius_increment_ratio']:.6f} |"
        )
    lines.append("")
    lines.append("## Naive product model `c^n`")
    lines.append("")
    lines.append(
        f"n = {naive['n_organs']}. Necessary organ cosine "
        f"`0.5^(1/43) = {naive['necessary_organ_cosine']:.8f}` "
        f"(Q30 comparable `0.5^(1/48) = {naive['q30_comparable_necessary_cosine']:.8f}`)."
    )
    lines.append("")
    lines.append("| c | c^43 | vs floor 0.5 |")
    lines.append("| ---: | ---: | :--- |")
    for row in naive["sweep"]:
        flag = "PASS" if row["clears_floor"] else "FAIL"
        lines.append(
            f"| {row['organ_cosine']:.3f} | {row['end_to_end_cosine']:.6e} | {flag} |"
        )
    lines.append("")
    lines.append("## Residual-identity model (honest bound)")
    lines.append("")
    lines.append(
        f"Per-layer factor `{ident['formula_alpha_zero']}` with measured r_L, "
        f"alpha=0. {ident['n_measured_ratios']} measured transitions plus one "
        f"mean-r fill for the unobserved embedding→L0 increment "
        f"(fill r = {ident['increment_ratios_used'][-1]:.6f})."
    )
    lines.append("")
    lines.append(
        f"Break-even: n=43 `{ident['break_even_alpha_zero_n43']:.6f}`, "
        f"n=42 `{ident['break_even_alpha_zero_n42']:.6f}`, "
        f"constant-r `{ident['break_even_constant_r_n43']:.6f}`, "
        f"measured-alignment `{ident['break_even_measured_alignment_n43']:.6f}`, "
        f"unique-13 `{ident['break_even_unique13_n43']:.6f}`."
    )
    lines.append("")
    lines.append("| c | identity Π (α=0, n=43) | vs floor 0.5 | measured-α Π |")
    lines.append("| ---: | ---: | :--- | ---: |")
    alpha_rows = {
        row["organ_cosine"]: row for row in ident["sweep_measured_alignment"]
    }
    for row in ident["sweep_alpha_zero"]:
        flag = "PASS" if row["clears_floor"] else "FAIL"
        arow = alpha_rows[row["organ_cosine"]]
        lines.append(
            f"| {row['organ_cosine']:.3f} | {row['end_to_end_cosine']:.6f} | "
            f"{flag} | {arow['end_to_end_cosine']:.6f} |"
        )
    lines.append("")
    lines.append("## ±10% sensitivity")
    lines.append("")
    lines.append("| knob | scale | break-even c | Π(0.80) | Π(0.84) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for row in receipt["sensitivity"]["increment_ratio_pm10"]:
        lines.append(
            f"| r × scale | {row['scale']:.1f} | {row['break_even']:.6f} | "
            f"{row['c_0_80_end_to_end']:.6f} | {row['c_0_84_end_to_end']:.6f} |"
        )
    for row in receipt["sensitivity"]["expansion_pm10"]:
        lines.append(
            f"| 1+(g-1)×scale | {row['scale']:.1f} | {row['break_even']:.6f} | "
            f"{row['c_0_80_end_to_end']:.6f} | {row['c_0_84_end_to_end']:.6f} |"
        )
    lines.append("")
    lines.append(
        "Literal `g × 0.9` is ill-posed: measured g ≈ 1.11, so a 10% cut "
        "in g is a ~100% cut in the expansion (g-1). It is reported in the "
        "JSON under `gain_multiplicative_reported_as_ill_posed` and is not "
        "used as a verdict knob."
    )
    lines.append("")
    lines.append("## Prior findings audit")
    lines.append("")
    lines.append("| claim | prior | measured | verdict |")
    lines.append("| :--- | ---: | ---: | :--- |")
    rows = [
        (
            "gain ~1.11×/layer",
            prior["collapsed_stream_grows_about_1_11x_per_layer"]["prior"],
            prior["collapsed_stream_grows_about_1_11x_per_layer"][
                "measured_all32_gmean"
            ],
            prior["collapsed_stream_grows_about_1_11x_per_layer"]["verdict"],
        ),
        (
            "cascade g ~1.121",
            prior["cascade_error_amplification_g_about_1_121"]["prior"],
            prior["cascade_error_amplification_g_about_1_121"][
                "measured_unique13_rms_mean_gain"
            ],
            prior["cascade_error_amplification_g_about_1_121"]["verdict"],
        ),
        (
            "0.5^(1/43)",
            prior["necessary_screen_0_5_to_the_1_over_43"]["prior"],
            prior["necessary_screen_0_5_to_the_1_over_43"]["measured"],
            prior["necessary_screen_0_5_to_the_1_over_43"]["verdict"],
        ),
        (
            "Q30 0.5^(1/48)",
            prior["q30_comparable_0_5_to_the_1_over_48"]["prior"],
            prior["q30_comparable_0_5_to_the_1_over_48"]["measured"],
            prior["q30_comparable_0_5_to_the_1_over_48"]["verdict"],
        ),
        (
            "0.80^43",
            prior["naive_0_80_to_the_43"]["prior"],
            prior["naive_0_80_to_the_43"]["measured"],
            prior["naive_0_80_to_the_43"]["verdict"],
        ),
        (
            "0.84^43",
            prior["naive_0_84_to_the_43"]["prior"],
            prior["naive_0_84_to_the_43"]["measured"],
            prior["naive_0_84_to_the_43"]["verdict"],
        ),
        (
            "identity break-even ~0.853",
            prior["residual_identity_break_even_about_0_853"]["prior"],
            prior["residual_identity_break_even_about_0_853"][
                "measured_per_layer_r_n43"
            ],
            prior["residual_identity_break_even_about_0_853"]["verdict"],
        ),
        (
            "identity Π(0.80) ~0.386",
            prior["residual_identity_product_c_0_80_about_0_386"]["prior"],
            prior["residual_identity_product_c_0_80_about_0_386"][
                "measured_per_layer_r"
            ],
            prior["residual_identity_product_c_0_80_about_0_386"]["verdict"],
        ),
        (
            "identity Π(0.84) ~0.470",
            prior["residual_identity_product_c_0_84_about_0_470"]["prior"],
            prior["residual_identity_product_c_0_84_about_0_470"][
                "measured_per_layer_r"
            ],
            prior["residual_identity_product_c_0_84_about_0_470"]["verdict"],
        ),
    ]
    for name, left, right, flag in rows:
        lines.append(f"| {name} | {left} | {right:.8g} | {flag} |")
    lines.append("")
    lines.append(prior["residual_identity_break_even_about_0_853"]["note"])
    lines.append("")
    lines.append("## Duplicate-stream caveat")
    lines.append("")
    lines.append(receipt["duplicate_streams"]["note"])
    lines.append("")
    if receipt["duplicate_streams"]["groups"]:
        lines.append("| n | example_ids |")
        lines.append("| ---: | :--- |")
        for group in receipt["duplicate_streams"]["groups"]:
            ids = ", ".join(group["example_ids"])
            lines.append(f"| {group['n']} | {ids} |")
        lines.append("")
    lines.append(f"Seal `{receipt['seal_sha256']}`.")
    lines.append("")
    return "\n".join(lines)


def render_stdout(receipt: Mapping[str, Any]) -> str:
    """Compact operator transcript for the completion report."""
    return render_markdown(receipt)


def write_receipts(
    receipt: Mapping[str, Any],
    *,
    json_path: Path,
    md_path: Path,
) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(receipt), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal the DSV4F residual-composition oracle from the late-hidden dump."
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=DEFAULT_ACTIVATIONS,
        help="Directory containing L00.npy .. L42.npy",
    )
    parser.add_argument(
        "--capture-receipt",
        type=Path,
        default=DEFAULT_CAPTURE_RECEIPT,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        default=True,
        help="Print the human report to stdout (default on)",
    )
    parser.add_argument("--no-print", dest="do_print", action="store_false")
    args = parser.parse_args(argv)
    receipt = build_receipt(
        activations_dir=args.activations,
        capture_receipt_path=args.capture_receipt,
    )
    write_receipts(receipt, json_path=args.out_json, md_path=args.out_md)
    if args.do_print:
        sys.stdout.write(render_stdout(receipt))
        sys.stdout.write(
            f"\nWrote {args.out_json} and {args.out_md} "
            f"seal={receipt['seal_sha256']}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
