"""ROUTER_SENSITIVE_ALLOCATION — which bits exist to preserve future control flow.

Flash already showed that a small hidden-state perturbation changes later routing.
This sidecar does not recompute that measurement. It consumes
`receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json` (and the sibling
Flash router receipts) and turns them into a per-surface precision ALLOCATION
rather than a scalar score.

    python3 tools/future/router_science.py --build
    python3 tools/future/router_science.py --selftest
    python3 -m pytest tools/future/test_router_science.py -q

Everything emitted is STATIC_ONLY / bench UNKNOWN. No GPU, no modellake reload,
no recompute of the L3-L4 SVD or membership map.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import hashlib
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.future._common import git, sha256_file


SCHEMA = "hawking.future.router_science.v1"
RECEIPT = "ROUTER_SENSITIVE_ALLOCATION.json"
K_DEFAULT = 10

# Cited from tools/flash_margin_residual_candidate.py conditional_policy, not re-derived.
TIGHT_MARGIN = 1e-5
ORDINARY_MARGIN = 1e-3
BASE_RESIDUAL_FRACTION = 0.005
TIGHT_RESIDUAL_FRACTION = 0.02
CITED_ORACLE_POLICY = (
    "0.5% residual normally; 2% when dense top10/top11 margin < 1e-5 "
    "(tools/flash_margin_residual_candidate.py conditional_policy)"
)

BIT_CLASSES = ("CRUSHED", "ORDINARY", "PREMIUM", "CONTROL_FLOW_PREMIUM")
BIT_CLASS_RANK = {
    "CRUSHED": 0,
    "ORDINARY": 1,
    "PREMIUM": 2,
    "CONTROL_FLOW_PREMIUM": 3,
}
# Storage recommendation, not a measured EBPW / not a hardware number.
BPW_FOR_CLASS = {
    "CRUSHED": 2.0,
    "ORDINARY": 4.25,
    "PREMIUM": 8.0,
    "CONTROL_FLOW_PREMIUM": 16.0,
}
DEFAULT_EPSILONS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
SYNTHETIC_SEED = 20260829

MAP_REL = "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json"
RESIDUAL_REL = "receipts/headless/FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4.json"
SELECTION_REL = "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json"
AB_REL = "receipts/headless/FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json"
STATE_F32_REL = "receipts/headless/FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4_CONDITIONAL_STATE.f32"

# Frozen scientific fields of FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json. Used only
# when the receipt is not reachable from this sparse worktree. Do not recompute.
RECOVERED_MAP: dict[str, Any] = {
    "schema": "hawking.flash.router_sensitivity_map.v1",
    "status": "MEASURED_SEAM_DIAGNOSTIC",
    "model": "Qwen3.8-Flash-Next",
    "seam": {
        "preceding_layer": 3,
        "router_layer": 4,
        "positions": 4,
        "hidden": 2560,
    },
    "router_source": {
        "tensor": "model.language_model.layers.4.mlp.gate.weight",
        "tensor_bytes": 2621440,
    },
    "routing": {
        "dense_top10": [
            [309, 290, 106, 410, 108, 287, 252, 306, 496, 367],
            [367, 131, 106, 381, 439, 506, 35, 252, 476, 301],
            [165, 106, 139, 131, 353, 476, 509, 59, 369, 162],
            [309, 252, 106, 488, 496, 59, 453, 68, 410, 306],
        ],
        "compact_top10": [
            [309, 290, 106, 410, 108, 287, 252, 306, 496, 367],
            [367, 131, 106, 381, 439, 506, 35, 252, 476, 144],
            [165, 106, 139, 131, 353, 476, 509, 59, 369, 162],
            [309, 106, 252, 488, 496, 59, 453, 68, 410, 306],
        ],
        "top10_membership_symmetric_difference": [0, 2, 0, 0],
        "rows_with_membership_change": 1,
        "dense_top10_top11_margin": [
            3.536790609359741e-05,
            7.858499884605408e-06,
            0.0007034391164779663,
            0.004267316311597824,
        ],
        "compact_top10_top11_margin": [
            2.9534101486206055e-05,
            7.46995210647583e-05,
            0.000664055347442627,
            0.007277142256498337,
        ],
        "dense_margin_min": 7.858499884605408e-06,
        "compact_margin_min": 2.9534101486206055e-05,
    },
    "delta": {
        "l2": 0.07877291738986969,
        "max_abs": 0.006731508299708366,
        "router_logit_l2": 0.03345860168337822,
        "router_logit_max_abs": 0.005146630574017763,
        "router_singular_values_head": [
            10.784625053405762, 4.076363563537598, 3.6030805110931396,
            3.209606170654297, 3.1556811332702637, 2.95176362991333,
            2.6126515865325925, 2.4526307582855225, 2.3108327388763428,
            2.115838050842285, 2.0432045459747314, 1.974076747894287,
            1.846590280532837, 1.8167051076889038, 1.7383095026016235,
            1.6708019971847534, 1.6559929847717285, 1.6140708923339844,
            1.551377534866333, 1.5336954593658447, 1.4949688911437988,
            1.4628504514694214, 1.4188809394836426, 1.3933573961257935,
            1.367780089378357, 1.3475364446640015, 1.334094762802124,
            1.2979071140289307, 1.2922284603118896, 1.2689149379730225,
            1.2339482307434082, 1.2085922956466675,
        ],
        "router_visible_subspace": [
            {"rank": 8, "delta_energy_fraction": 0.004872831050306559,
             "oracle_repaired_rows_with_membership_change": 1,
             "oracle_repaired_mean_topk_symmetric_difference": 0.5,
             "oracle_projected_repair_bytes_fp16": 64},
            {"rank": 16, "delta_energy_fraction": 0.013232707045972347,
             "oracle_repaired_rows_with_membership_change": 1,
             "oracle_repaired_mean_topk_symmetric_difference": 0.5,
             "oracle_projected_repair_bytes_fp16": 128},
            {"rank": 32, "delta_energy_fraction": 0.022886082530021667,
             "oracle_repaired_rows_with_membership_change": 1,
             "oracle_repaired_mean_topk_symmetric_difference": 0.5,
             "oracle_projected_repair_bytes_fp16": 256},
            {"rank": 64, "delta_energy_fraction": 0.0407201424241066,
             "oracle_repaired_rows_with_membership_change": 1,
             "oracle_repaired_mean_topk_symmetric_difference": 0.5,
             "oracle_projected_repair_bytes_fp16": 512},
            {"rank": 128, "delta_energy_fraction": 0.06615317612886429,
             "oracle_repaired_rows_with_membership_change": 1,
             "oracle_repaired_mean_topk_symmetric_difference": 0.5,
             "oracle_projected_repair_bytes_fp16": 1024},
            {"rank": 256, "delta_energy_fraction": 0.1200057864189148,
             "oracle_repaired_rows_with_membership_change": 1,
             "oracle_repaired_mean_topk_symmetric_difference": 0.5,
             "oracle_projected_repair_bytes_fp16": 2048},
            {"rank": 512, "delta_energy_fraction": 0.22175653278827667,
             "oracle_repaired_rows_with_membership_change": 0,
             "oracle_repaired_mean_topk_symmetric_difference": 0.0,
             "oracle_projected_repair_bytes_fp16": 4096},
        ],
    },
    "coordinate_salience": [
        {"fraction": 0.001, "dimensions": 3, "residual_bytes_f32": 48,
         "logit_delta_l2_fraction": 0.3462471067905426,
         "mean_topk_symmetric_difference": 0.5, "rows_with_membership_change": 1},
        {"fraction": 0.005, "dimensions": 13, "residual_bytes_f32": 208,
         "logit_delta_l2_fraction": 0.4253232479095459,
         "mean_topk_symmetric_difference": 0.5, "rows_with_membership_change": 1},
        {"fraction": 0.01, "dimensions": 26, "residual_bytes_f32": 416,
         "logit_delta_l2_fraction": 0.4671165347099304,
         "mean_topk_symmetric_difference": 0.5, "rows_with_membership_change": 1},
        {"fraction": 0.02, "dimensions": 51, "residual_bytes_f32": 816,
         "logit_delta_l2_fraction": 0.5412788987159729,
         "mean_topk_symmetric_difference": 0.5, "rows_with_membership_change": 1},
        {"fraction": 0.05, "dimensions": 128, "residual_bytes_f32": 2048,
         "logit_delta_l2_fraction": 0.657504677772522,
         "mean_topk_symmetric_difference": 0.5, "rows_with_membership_change": 1},
    ],
    "next_gate": (
        "fit a margin-aware router-sensitive residual NR and test exact "
        "layer-3/4 organ parity; retain dense router at high fidelity"
    ),
    "promotion_allowed": False,
}


class ControlFlowCrushError(ValueError):
    """Refuses to assign CRUSHED precision to bits that preserve routing."""


@dataclass(frozen=True)
class Surface:
    """A representation surface that may or may not carry future control flow.

    `n_params` and `position_index` are recorded for provenance. Classification
    does not read them — the negative-control tests exist to keep that true.
    """

    name: str
    kind: str  # router_weight | hidden_visible | hidden_inert | position_residual
    margin_min: float | None = None
    n_params: int = 0
    hidden: int = 0
    positions: int = 0
    position_index: int | None = None
    k: int = K_DEFAULT
    membership_flip: bool | None = None


# ---------------------------------------------------------------------------
# Read-only locators. Never write the Codex surface.
# ---------------------------------------------------------------------------

def _parent_checkout() -> Path | None:
    raw = git("rev-parse", "--git-common-dir")
    if not raw:
        return None
    common = Path(raw)
    if not common.is_absolute():
        common = (REPO / common).resolve()
    else:
        common = common.resolve()
    if common.name == ".git":
        return common.parent
    return None


def load_readonly_json(rel: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load JSON from this worktree, git HEAD, or the parent checkout.

    Untracked parent-disk files (the L3-L4 map) are readable this way. This
    function never writes.
    """
    prov: dict[str, Any] = {
        "rel": rel,
        "source": None,
        "sha256": None,
        "path": None,
        "in_git_HEAD": False,
        "on_worktree_disk": False,
        "on_parent_disk": False,
    }
    wt = REPO / rel
    if wt.is_file():
        prov.update(
            source="worktree",
            path=str(wt),
            sha256=sha256_file(wt),
            on_worktree_disk=True,
        )
        return load_json(wt), prov
    shown = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True
    )
    if shown.returncode == 0 and shown.stdout:
        prov.update(
            source="git_HEAD",
            sha256=hashlib.sha256(shown.stdout).hexdigest(),
            in_git_HEAD=True,
        )
        return json.loads(shown.stdout.decode()), prov
    parent = _parent_checkout()
    if parent is not None:
        pp = parent / rel
        if pp.is_file():
            prov.update(
                source="parent_worktree_disk",
                path=str(pp),
                sha256=sha256_file(pp),
                on_parent_disk=True,
            )
            return load_json(pp), prov
    return None, prov


def note_readonly_blob(rel: str) -> dict[str, Any]:
    """Record size/sha256 of a sibling artifact without interpreting it."""
    info: dict[str, Any] = {
        "rel": rel,
        "present": False,
        "size_bytes": None,
        "sha256": None,
        "loaded": False,
        "note": "noted, not loaded",
    }
    parent = _parent_checkout()
    candidates = [REPO / rel]
    if parent is not None:
        candidates.append(parent / rel)
    for p in candidates:
        if p.is_file():
            info.update(
                present=True,
                size_bytes=p.stat().st_size,
                sha256=sha256_file(p),
                path=str(p),
            )
            return info
    return info


def load_sensitivity_map() -> tuple[dict[str, Any], dict[str, Any]]:
    doc, prov = load_readonly_json(MAP_REL)
    if doc is None:
        blob = json.dumps(RECOVERED_MAP, sort_keys=True, separators=(",", ":")).encode()
        prov = {
            "rel": MAP_REL,
            "source": "recovered_snapshot",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "path": None,
            "in_git_HEAD": False,
            "on_worktree_disk": False,
            "on_parent_disk": False,
        }
        return json.loads(blob), prov
    return doc, prov


def n_experts_from_map(map_doc: dict[str, Any]) -> int:
    hidden = int(map_doc["seam"]["hidden"])
    nbytes = int((map_doc.get("router_source") or {}).get("tensor_bytes") or 0)
    if nbytes and hidden:
        # BF16 router: 2 bytes per element.
        return nbytes // (hidden * 2)
    return 512


# ---------------------------------------------------------------------------
# 1. Router Jacobian approximation
# ---------------------------------------------------------------------------

def finite_difference_logit_jacobian(
    W: np.ndarray, x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Column-wise FD of z(x) = x @ W.T. Exact (up to rounding) because z is affine."""
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z0 = x @ W.T
    _, hidden = W.shape
    columns = []
    for i in range(hidden):
        xp = x.copy()
        xp[i] += eps
        columns.append(((xp @ W.T) - z0) / eps)
    return np.stack(columns, axis=1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def softmax_jacobian_analytical(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.diag(p) - np.outer(p, p)


def softmax_prob_jacobian(logits: np.ndarray, eps: float = 1e-6) -> dict[str, Any]:
    z = np.asarray(logits, dtype=np.float64)
    p = softmax(z)
    analytical = softmax_jacobian_analytical(p)
    fd = np.empty_like(analytical)
    for i in range(z.size):
        zp = z.copy()
        zp[i] += eps
        fd[:, i] = (softmax(zp) - p) / eps
    err = float(np.max(np.abs(fd - analytical)))
    return {
        "analytical_form": "diag(p) - p p^T",
        "finite_difference_eps": eps,
        "finite_difference_max_abs_error": err,
        "matches_analytical": bool(err < 1e-5),
        "validity_range": (
            "Local linearization of softmax. Remainder is O(||Δz||^2). "
            "This is not a top-k Jacobian — top-k membership is piecewise constant."
        ),
    }


def jacobian_from_map(map_doc: dict[str, Any]) -> dict[str, Any]:
    """Name the Jacobian implied by the sensitivity-map shapes, without loading W."""
    seam = map_doc["seam"]
    hidden = int(seam["hidden"])
    experts = n_experts_from_map(map_doc)
    delta = map_doc["delta"]
    dx = float(delta["l2"])
    dz = float(delta["router_logit_l2"])
    dz_inf = float(delta["router_logit_max_abs"])
    sigmas = [float(s) for s in delta.get("router_singular_values_head") or []]
    sigma_max = sigmas[0] if sigmas else None
    lipschitz = (dz / dx) if dx else None
    bound_ok = (
        sigma_max is not None
        and lipschitz is not None
        and lipschitz <= sigma_max + 1e-9
    )
    margins = [float(m) for m in map_doc["routing"]["dense_top10_top11_margin"]]
    n_flip = int(map_doc["routing"]["rows_with_membership_change"])
    n_pos = int(seam["positions"])
    overpredicts = bool(
        dz_inf is not None and margins and dz_inf > max(margins) and n_flip < n_pos
    )
    return {
        "analytical_form": "J_logit = W = gate.weight, because z = x @ W.T",
        "jacobian_matrix": None,
        "jacobian_matrix_reason": (
            "W is not stored in the sensitivity map; this sidecar does not load "
            "the modellake shard. Shape and spectrum are known; entries are not."
        ),
        "shape": [experts, hidden],
        "n_experts": experts,
        "hidden": hidden,
        "k": K_DEFAULT,
        "observed_hidden_l2": dx,
        "observed_logit_l2": dz,
        "observed_logit_linf": dz_inf,
        "observed_operator_norm_proxy": lipschitz,
        "sigma_max": sigma_max,
        "below_operator_norm_bound": bound_ok,
        "validity_range": {
            "logit_jacobian": {
                "range": "exact for every perturbation in R^{hidden}; the map is affine",
                "finite_difference": (
                    "the compact-minus-dense step on this seam is an exact FD of "
                    "the logit map, not an infinitesimal eps"
                ),
            },
            "softmax_jacobian": {
                "form": "J_p = (diag(p) - p p^T) @ W",
                "range": "local; remainder O(||Δz||^2)",
                "observed_logit_linf": dz_inf,
                "numerical_p_available": False,
                "reason": "the map does not store the 512-vector of probabilities",
            },
            "topk_membership": {
                "form": "no Jacobian (piecewise constant)",
                "first_order_flip_distance": "K vs K+1 logit margin",
                "range": (
                    "a pair (i in top-K, j outside) flips when "
                    "(W_i - W_j) · δx exceeds that pair's gap"
                ),
                "full_delta_z_available": False,
                "global_linf_overpredicts_flips": overpredicts,
                "evidence": (
                    f"observed ||Δz||_∞={dz_inf} exceeds every stored margin, "
                    f"but only {n_flip}/{n_pos} rows flipped"
                ),
            },
        },
    }


def router_jacobian_approximation(
    *,
    W: np.ndarray | None = None,
    x: np.ndarray | None = None,
    eps: float = 1e-6,
    map_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analytical + FD logit Jacobian, plus the seam-level validity range.

    When `W` is supplied (synthetic tests) the FD residual is measured.
    When only the sensitivity map is supplied, W is not reconstructed.
    """
    out: dict[str, Any] = {
        "analytical_form": "J_logit[e, h] = W[e, h]  because z = x @ W.T",
        "softmax_form": "J_p = (diag(p) - p p^T) @ W  (local)",
        "topk_form": "no derivative; use route_margins",
    }
    if W is not None:
        W64 = np.asarray(W, dtype=np.float64)
        experts, hidden = W64.shape
        if x is None:
            x64 = np.zeros(hidden, dtype=np.float64)
        else:
            x64 = np.asarray(x, dtype=np.float64)
        fd = finite_difference_logit_jacobian(W64, x64, eps)
        err = float(np.max(np.abs(fd - W64)))
        _u, s, vt = np.linalg.svd(W64, full_matrices=True)
        inert = vt[experts:] if experts < hidden else np.zeros((0, hidden))
        inert_action = (
            float(np.max(np.abs(W64 @ inert.T))) if len(inert) else 0.0
        )
        top_action = float(np.linalg.norm(W64 @ vt[0]))
        out["synthetic"] = {
            "shape": [experts, hidden],
            "finite_difference_eps": eps,
            "finite_difference_max_abs_error": err,
            "matches_analytical": bool(err < 1e-6),
            "top_singular": float(s[0]),
            "top_singular_action": top_action,
            "inert_action_max_abs": inert_action,
            "routing_inert_dim": int(inert.shape[0]),
            "validity_range": (
                "Logit Jacobian is exact on all of R^H (affine). FD error is "
                "rounding only. Directions in the right nullspace of W are "
                "routing-inert: J v = 0."
            ),
        }
    if map_doc is not None:
        out["from_sensitivity_map"] = jacobian_from_map(map_doc)
    return out


# ---------------------------------------------------------------------------
# 2–3. Route-margin distribution and top-K boundary sensitivity
# ---------------------------------------------------------------------------

def route_margins(logits: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """Gap between the K-th and (K+1)-th expert logit, per row."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    if z.shape[1] <= k:
        raise ValueError(f"need at least k+1 experts, got {z.shape[1]} with k={k}")
    ordered = np.sort(z, axis=1)
    return ordered[:, -k] - ordered[:, -k - 1]


def margin_distribution(margins: list[float] | np.ndarray) -> dict[str, Any]:
    xs = [float(m) for m in margins]
    n = len(xs)
    ordered = sorted(xs)
    return {
        "n": n,
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "mean": (sum(ordered) / n) if n else None,
        "median": float(statistics.median(ordered)) if ordered else None,
        "sorted": ordered,
        "sample_is_tiny": n < 32,
        "definition": "z_{(K)} - z_{(K+1)} on the captured dense routes; K=10",
    }


def topk_boundary_sensitivity(
    margins: list[float] | np.ndarray,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
) -> dict[str, Any]:
    xs = [float(m) for m in margins]
    n = len(xs)
    rows = []
    for eps in epsilons:
        count = sum(1 for m in xs if m < eps)
        rows.append(
            {
                "epsilon": eps,
                "n_within": count,
                "fraction": (count / n) if n else None,
                "meaning": (
                    f"fraction of tokens whose K vs K+1 logit gap is < {eps}"
                ),
            }
        )
    return {"n": n, "k": K_DEFAULT, "by_epsilon": rows}


def flipped_row_is_min_margin(map_doc: dict[str, Any]) -> dict[str, Any]:
    diffs = [int(v) for v in map_doc["routing"]["top10_membership_symmetric_difference"]]
    margins = [float(m) for m in map_doc["routing"]["dense_top10_top11_margin"]]
    flip_idx = [i for i, d in enumerate(diffs) if d > 0]
    min_i = int(min(range(len(margins)), key=lambda i: margins[i])) if margins else None
    return {
        "flipped_positions": flip_idx,
        "min_margin_position": min_i,
        "min_margin": margins[min_i] if min_i is not None else None,
        "holds": bool(flip_idx) and min_i in flip_idx,
        "caveat": "n=4; coincidence is still the only seam evidence we have",
    }


# ---------------------------------------------------------------------------
# 4. Critical hidden directions vs routing-inert complement
# ---------------------------------------------------------------------------

def critical_hidden_directions(map_doc: dict[str, Any]) -> dict[str, Any]:
    hidden = int(map_doc["seam"]["hidden"])
    experts = n_experts_from_map(map_doc)
    subspace = list(map_doc["delta"]["router_visible_subspace"])
    repaired = [
        r for r in subspace if int(r["oracle_repaired_rows_with_membership_change"]) == 0
    ]
    first = min(repaired, key=lambda r: int(r["rank"])) if repaired else None
    coords = list(map_doc["coordinate_salience"])
    coords_failed = all(int(c["rows_with_membership_change"]) > 0 for c in coords)
    return {
        "router_visible_rank_upper_bound": experts,
        "routing_inert_dim_lower_bound": hidden - experts,
        "why_inert": (
            "W is (n_experts × hidden); its row-space in R^{hidden} has dimension "
            "at most n_experts. The orthogonal complement does not move logits."
        ),
        "spectrum_head": [float(s) for s in map_doc["delta"]["router_singular_values_head"]],
        "observed_delta_energy_by_visible_rank": subspace,
        "smallest_visible_rank_that_repaired_membership": (
            int(first["rank"]) if first else None
        ),
        "repair_evidence": first,
        "coordinate_salience_did_not_repair_at_any_tested_fraction": coords_failed,
        "coordinate_salience": coords,
        "denominator_warning": (
            "coordinate_salience fractions are of LOGIT-delta L2; SVD energy "
            "fractions are of STATE-delta energy. They are not the same pie."
        ),
        "implication": (
            "Control-flow-critical hidden directions are the row-space of "
            "gate.weight (right singular vectors). They are not the "
            "largest-|delta| coordinates of the compact error: a 5% coordinate "
            "slice carried ~66% of logit-delta L2 and still left a membership "
            "flip, while a rank-512 SVD projection of the same error repaired all rows."
        ),
    }


def per_dimension_sensitivity(W: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Map formula: sum_{n,e} |delta[n, j] * W[e, j]|. Used in synthetic tests."""
    delta = np.asarray(delta, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    return (np.abs(delta) * np.abs(W).sum(axis=0)[None, :]).sum(axis=0)


def per_dimension_sensitivity_from_map(map_doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "per_index_vector": None,
        "per_index_vector_reason": (
            "FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json stores aggregate "
            "coordinate_salience fractions, not the 2560-length order"
        ),
        "aggregates": list(map_doc["coordinate_salience"]),
        "recommendation": (
            "Do not treat the top coordinate-salience slice as the premium set. "
            "Allocate residual in the rank-512 router-visible subspace instead."
        ),
    }


# ---------------------------------------------------------------------------
# 5–6. Precision allocation and residual budget by route risk
# ---------------------------------------------------------------------------

def residual_budget_fraction(margin: float | None) -> float | None:
    """Cite the existing oracle fractions; this module does not invent new cuts."""
    if margin is None:
        return None
    return TIGHT_RESIDUAL_FRACTION if margin < TIGHT_MARGIN else BASE_RESIDUAL_FRACTION


def residual_budget_by_route_risk(
    margins: list[float],
    flips: list[int] | None = None,
) -> dict[str, Any]:
    rows = []
    for i, m in enumerate(margins):
        m = float(m)
        rows.append(
            {
                "position": i,
                "margin": m,
                "route_risk": 1.0 / max(m, 1e-30),
                "residual_hidden_fraction": residual_budget_fraction(m),
                "basis": "router_visible_right_singular_subspace",
                "cited_oracle_used_coordinate_salience": True,
                "this_module_redirects_to_svd_subspace": True,
                "membership_flip_on_compact_error": (
                    bool(flips[i] > 0) if flips is not None else None
                ),
            }
        )
    return {
        "policy": CITED_ORACLE_POLICY,
        "basis_change": (
            "Spend the cited fractions on the SVD-visible subspace, which is "
            "the only repair that restored membership in the same map. The "
            "oracle coordinate residual (FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4) "
            "still left 1 row changed at every tested fraction, including 2%."
        ),
        "positions": rows,
    }


def classify_surface(surface: Surface) -> str:
    """Margin- and kind-driven. Ignores n_params and position_index on purpose."""
    if surface.kind == "router_weight":
        return "CONTROL_FLOW_PREMIUM"
    if surface.kind == "hidden_inert":
        return "CRUSHED"
    if surface.kind == "hidden_visible":
        return "PREMIUM"
    if surface.kind == "position_residual":
        if surface.membership_flip or (
            surface.margin_min is not None and surface.margin_min < TIGHT_MARGIN
        ):
            return "PREMIUM"
        if surface.margin_min is not None and surface.margin_min < ORDINARY_MARGIN:
            return "ORDINARY"
        return "CRUSHED"
    raise ValueError(f"unknown surface kind {surface.kind!r}")


def _must_protect(surface: Surface, natural: str) -> bool:
    if natural in {"PREMIUM", "CONTROL_FLOW_PREMIUM"}:
        return True
    if surface.margin_min is not None and surface.margin_min < TIGHT_MARGIN:
        return True
    if surface.membership_flip:
        return True
    return False


def assign_bit_class(
    surface: Surface, requested: str | None = None
) -> dict[str, Any]:
    """Allocate precision to one surface.

    Passing requested='CRUSHED' on a control-flow-critical surface RAISES
    ControlFlowCrushError. That refusal is the guard.
    """
    natural = classify_surface(surface)
    chosen = requested or natural
    if chosen not in BIT_CLASS_RANK:
        raise ValueError(f"unknown bit class {chosen!r}")
    if chosen == "CRUSHED" and _must_protect(surface, natural):
        raise ControlFlowCrushError(
            f"refusing to crush control-flow bits on {surface.name}: "
            f"natural={natural} margin={surface.margin_min} "
            f"membership_flip={surface.membership_flip}"
        )
    residual = (
        residual_budget_fraction(surface.margin_min)
        if surface.kind == "position_residual"
        else None
    )
    reason = {
        "router_weight": (
            "gate.weight IS the control-flow function; Q4/G64 of the layer-0 "
            "router missed 2/10 experts and no low-bit candidate was exact. "
            "The L3-L4 map's next_gate is 'retain dense router at high fidelity'."
        ),
        "hidden_visible": (
            "row-space of gate.weight (rank ≤ n_experts). Rank-512 projection "
            "of the observed compact error restored top-10 membership; smaller ranks did not."
        ),
        "hidden_inert": (
            "orthogonal complement of the router row-space; logits are invariant "
            "to perturbations here (exact, not approximate)."
        ),
        "position_residual": (
            f"K vs K+1 margin={surface.margin_min}; "
            f"tight cut is {TIGHT_MARGIN} (cited oracle). "
            f"membership_flip={surface.membership_flip}"
        ),
    }[surface.kind]
    return {
        "name": surface.name,
        "kind": surface.kind,
        "bit_class": chosen,
        "natural_bit_class": natural,
        "recommended_storage_bpw": BPW_FOR_CLASS[chosen],
        "residual_budget_fraction": residual,
        "margin_min": surface.margin_min,
        "membership_flip": surface.membership_flip,
        "n_params": surface.n_params,
        "position_index": surface.position_index,
        "hidden": surface.hidden,
        "positions": surface.positions,
        "k": surface.k,
        "allocation_not_measurement": True,
        "reason": reason,
    }


def precision_rank(allocation: dict[str, Any]) -> tuple[int, float]:
    return (
        BIT_CLASS_RANK[allocation["bit_class"]],
        float(allocation.get("residual_budget_fraction") or 0.0),
    )


def allocate_precision(surfaces: list[Surface]) -> list[dict[str, Any]]:
    return [assign_bit_class(s) for s in sorted(surfaces, key=lambda s: s.name)]


def surfaces_from_map(map_doc: dict[str, Any]) -> list[Surface]:
    hidden = int(map_doc["seam"]["hidden"])
    positions = int(map_doc["seam"]["positions"])
    experts = n_experts_from_map(map_doc)
    margins = [float(m) for m in map_doc["routing"]["dense_top10_top11_margin"]]
    flips = [int(v) for v in map_doc["routing"]["top10_membership_symmetric_difference"]]
    surfaces = [
        Surface(
            name="flash.moe.gate.weight.family",
            kind="router_weight",
            n_params=experts * hidden,
            hidden=hidden,
            k=K_DEFAULT,
        ),
        Surface(
            name="flash.l3_state.router_visible_subspace",
            kind="hidden_visible",
            n_params=experts * positions,
            hidden=hidden,
            positions=positions,
            margin_min=min(margins) if margins else None,
            k=K_DEFAULT,
        ),
        Surface(
            name="flash.l3_state.router_inert_complement",
            kind="hidden_inert",
            n_params=(hidden - experts) * positions,
            hidden=hidden,
            positions=positions,
            k=K_DEFAULT,
        ),
    ]
    for i, (margin, diff) in enumerate(zip(margins, flips)):
        surfaces.append(
            Surface(
                name=f"flash.l3_l4.position.{i}",
                kind="position_residual",
                margin_min=float(margin),
                n_params=hidden,
                hidden=hidden,
                positions=1,
                position_index=i,
                k=K_DEFAULT,
                membership_flip=bool(diff > 0),
            )
        )
    return surfaces


# ---------------------------------------------------------------------------
# Headline: which bits exist primarily to preserve future control flow?
# ---------------------------------------------------------------------------

def which_bits_preserve_control_flow(
    map_doc: dict[str, Any],
    selection: dict[str, Any] | None,
    ab: dict[str, Any] | None,
) -> dict[str, Any]:
    hold = flipped_row_is_min_margin(map_doc)
    crit = critical_hidden_directions(map_doc)
    experts = n_experts_from_map(map_doc)
    hidden = int(map_doc["seam"]["hidden"])
    sel_parity = None
    if selection is not None:
        sel_parity = {
            "layer_tensor": (selection.get("source_block") or {}).get("tensor_name"),
            "status": (selection.get("source_selection_parity") or {}).get("status"),
            "top_k_overlap_count": (selection.get("source_selection_parity") or {}).get(
                "top_k_overlap_count"
            ),
            "top_k_overlap_fraction": (selection.get("source_selection_parity") or {}).get(
                "top_k_overlap_fraction"
            ),
            "logits_rmse": ((selection.get("source_selection_parity") or {}).get("logits") or {}).get(
                "rmse"
            ),
            "note": (
                "This is a layer-0 synthetic-vector Q4/G64 vs BF16 comparison, "
                "not the L3-L4 captured seam. Homology is a hypothesis."
            ),
        }
    ab_rec = (ab or {}).get("recommendation") if ab else None
    return {
        "question": "WHICH BITS EXIST PRIMARILY TO PRESERVE FUTURE CONTROL FLOW?",
        "answer": (
            "On the evidence actually in hand: (1) the router gate.weight bits "
            f"themselves (shape {experts}×{hidden} at the L3-L4 seam; the same "
            "contract applies to every MoE gate but is only Q4-tested at layer 0); "
            "(2) the component of the incoming hidden state that lies in that "
            f"gate's row-space (≤{experts} of {hidden} dimensions) — rank 512 is "
            "the smallest SVD repair that restored top-10 membership; (3) extra "
            "residual at tokens whose K vs K+1 logit margin is below 1e-5 — on "
            "this 4-token seam that is exactly the unique membership flip. "
            f"The remaining ≥{hidden - experts} hidden directions are routing-inert "
            "for this linear router (exact). Expert body bits are payload after "
            "the route is chosen; they do not preserve control flow. Shared-expert "
            "sigmoid is not router selection."
        ),
        "bits": [
            {
                "what": "router gate.weight",
                "class": "CONTROL_FLOW_PREMIUM",
                "why": (
                    "These bits are the routing function. The L3-L4 map's next_gate "
                    "is 'retain dense router at high fidelity'. Layer-0 Q4/G64 "
                    "selection overlap was 8/10; no low-bit candidate was exact."
                ),
            },
            {
                "what": "hidden state in the router-visible right-singular subspace",
                "class": "PREMIUM",
                "why": (
                    f"smallest repairing rank = {crit['smallest_visible_rank_that_repaired_membership']}; "
                    "rank 256 still left 1 membership change"
                ),
            },
            {
                "what": "sparse residual at low-margin token positions",
                "class": "PREMIUM",
                "why": (
                    f"flipped positions {hold['flipped_positions']} equal min-margin "
                    f"position {hold['min_margin_position']} (margin {hold['min_margin']})"
                ),
            },
            {
                "what": "routing-inert hidden complement",
                "class": "CRUSHED",
                "why": "J v = 0 for v in the right nullspace of gate.weight",
            },
        ],
        "flip_is_the_tightest_margin": hold,
        "layer0_q4_selection_parity": sel_parity,
        "layer0_representation_ab_recommendation": ab_rec,
        "could_not_answer": [
            "Named indices of the 2560 hidden dimensions (order not stored).",
            "Softmax-space Jacobian numbers on this seam (no 512-vector of p).",
            "Pairwise first-order flip predictor (no full Δz stored).",
            "Any layer other than the L3-L4 seam and the layer-0 Q4 study.",
            "Any sample larger than 4 captured positions.",
            "Whether 16-bit is the right premium (no protected measurement).",
            "Q4 of the layer-4 gate specifically (Q4 evidence is layer 0).",
            "Expert-body bits as control-flow bits — they are not; they are payload.",
        ],
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def _selection_excerpt(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    parity = doc.get("source_selection_parity") or {}
    cfg = ((doc.get("config") or {}).get("router") or {})
    return {
        "schema": doc.get("schema"),
        "tensor": (doc.get("source_block") or {}).get("tensor_name"),
        "num_experts": cfg.get("num_experts"),
        "num_experts_per_tok": cfg.get("num_experts_per_tok"),
        "router_logits": cfg.get("router_logits"),
        "selection_rule": cfg.get("selection"),
        "shared_expert_sigmoid_is_not_router_selection": cfg.get(
            "shared_expert_sigmoid_is_not_router_selection"
        ),
        "parity_status": parity.get("status"),
        "qualification": parity.get("qualification"),
        "top_k_overlap_count": parity.get("top_k_overlap_count"),
        "top_k_overlap_fraction": parity.get("top_k_overlap_fraction"),
        "expert_ids_exact_match": parity.get("expert_ids_exact_match"),
        "logits_cosine": (parity.get("logits") or {}).get("cosine"),
        "logits_rmse": (parity.get("logits") or {}).get("rmse"),
        "logits_max_abs_error": (parity.get("logits") or {}).get("max_abs_error"),
    }


def _ab_excerpt(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    cands = []
    for row in doc.get("candidates") or []:
        sp = row.get("source_selection_parity") or {}
        cands.append(
            {
                "id": row.get("id"),
                "family": row.get("family"),
                "effective_bits_per_value": row.get("effective_bits_per_value"),
                "group_size": row.get("group_size"),
                "top_k_overlap_count": sp.get("top_k_overlap_count"),
                "expert_ids_exact_match": sp.get("expert_ids_exact_match"),
            }
        )
    rec = doc.get("recommendation") or {}
    return {
        "schema": doc.get("schema"),
        "candidates": cands,
        "best_low_bit_overlap_candidate": rec.get("best_low_bit_overlap_candidate"),
        "best_low_bit_overlap_count": rec.get("best_low_bit_overlap_count"),
        "source_top_k_exact_for_any_low_bit_candidate": rec.get(
            "source_top_k_exact_for_any_low_bit_candidate"
        ),
        "decision": rec.get("decision"),
    }


def _residual_excerpt(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    return {
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "promotion_allowed": doc.get("promotion_allowed"),
        "conditional_policy": doc.get("conditional_policy"),
        "candidate_rows": [
            {
                "fraction": r.get("fraction"),
                "dimensions": r.get("dimensions"),
                "rows_with_top10_membership_change": r.get(
                    "rows_with_top10_membership_change"
                ),
                "route_stable_on_observed_seam": r.get("route_stable_on_observed_seam"),
            }
            for r in (doc.get("candidate_rows") or [])
        ],
        "note": (
            "Oracle coordinate residual never reached route_stable_on_observed_seam "
            "at any tested fraction. Consumed as negative evidence against "
            "coordinate-salience allocation, not re-run."
        ),
    }


def recovered_implementation(provs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "adequate_existing": [
            {
                "path": "tools/flash_router_sensitivity_map.py",
                "role": (
                    "Produced the L3-L4 membership / margin / SVD / coordinate map. "
                    "READ-ONLY. This sidecar does not recompute it."
                ),
                "in_git_HEAD": False,
                "on_parent_disk": True,
            },
            {
                "path": MAP_REL,
                "role": "the measurement this module allocates from",
                "provenance": provs.get("map"),
            },
            {
                "path": "tools/flash_margin_residual_candidate.py",
                "role": (
                    "Existing residual budget by route risk (0.5% / 2% at margin 1e-5). "
                    "Policy cited; coordinate basis rejected."
                ),
                "in_git_HEAD": False,
                "on_parent_disk": True,
            },
            {
                "path": RESIDUAL_REL,
                "role": "oracle residual candidate; still not route-stable",
                "provenance": provs.get("residual"),
            },
            {
                "path": STATE_F32_REL,
                "role": "conditional-repair state; noted, not loaded",
                "provenance": provs.get("state_f32"),
            },
            {
                "path": "tools/flash_route_stability.py",
                "role": "compact-bank reuse safety across tokens; different question",
                "in_git_HEAD": False,
            },
            {
                "path": "tools/flash_route_union_compare.py",
                "role": "dense vs route-union fingerprint parity; different question",
                "in_git_HEAD": False,
            },
            {
                "path": "hcli/agentos/flash_router_selection.py",
                "role": "executes FP32 softmax/top-k over a persisted body; does not allocate bits",
            },
            {
                "path": SELECTION_REL,
                "role": "layer-0 Q4/G64 vs source top-k mismatch (8/10)",
                "provenance": provs.get("selection"),
            },
            {
                "path": AB_REL,
                "role": "no low-bit router candidate is exact; NF4/G64 best at 9/10",
                "provenance": provs.get("ab"),
            },
            {
                "path": "tools/headless/sensitivity_allocation.py",
                "role": (
                    "Qwen27 organ VoI / EBPW assignment. Different species, different "
                    "question. Not forked."
                ),
            },
        ],
        "not_adequate_alone": (
            "The Flash map is a diagnostic of one seam. It is not a precision "
            "allocation. The Qwen27 sensitivity_allocation receipt is a scalar-per-organ "
            "VoI ranking of a dense model, not a router-control-flow allocation. "
            "Nothing on disk named the logit Jacobian, stated its validity range, "
            "or refused to crush control-flow bits."
        ),
        "in_git_HEAD_named_flash_tools": False,
        "named_flash_tools_untracked_on_parent_disk": True,
    }


def build() -> Path:
    map_doc, map_prov = load_sensitivity_map()
    selection, sel_prov = load_readonly_json(SELECTION_REL)
    ab, ab_prov = load_readonly_json(AB_REL)
    residual, res_prov = load_readonly_json(RESIDUAL_REL)
    state_info = note_readonly_blob(STATE_F32_REL)

    # If the on-disk map is present, it is authority; the snapshot is a fallback.
    snapshot_margin = RECOVERED_MAP["routing"]["dense_margin_min"]
    disk_margin = float(map_doc["routing"]["dense_margin_min"])
    snapshot_agrees = disk_margin == snapshot_margin or (
        abs(disk_margin - snapshot_margin) / max(abs(snapshot_margin), 1e-30) < 1e-12
    )

    margins = [float(m) for m in map_doc["routing"]["dense_top10_top11_margin"]]
    flips = [int(v) for v in map_doc["routing"]["top10_membership_symmetric_difference"]]
    surfaces = surfaces_from_map(map_doc)
    allocation = allocate_precision(surfaces)

    rng = np.random.default_rng(SYNTHETIC_SEED)
    W = rng.normal(size=(8, 16))
    x = rng.normal(size=(16,))
    jac = router_jacobian_approximation(W=W, x=x, eps=1e-6, map_doc=map_doc)
    sm = softmax_prob_jacobian(rng.normal(size=(8,)), eps=1e-6)

    headline = which_bits_preserve_control_flow(map_doc, selection, ab)
    crit = critical_hidden_directions(map_doc)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Precision allocation for bits that exist primarily to preserve "
            "future MoE control flow. Consumes the Flash L3-L4 sensitivity map; "
            "does not recompute it; does not claim a hardware number."
        ),
        "evidence_class": "STATIC_ONLY",
        "promotion_allowed": False,
        "model": map_doc.get("model"),
        "seam": {
            "preceding_layer": map_doc["seam"]["preceding_layer"],
            "router_layer": map_doc["seam"]["router_layer"],
            "positions": map_doc["seam"]["positions"],
            "hidden": map_doc["seam"]["hidden"],
            "n_experts": n_experts_from_map(map_doc),
            "k": K_DEFAULT,
            "tensor": (map_doc.get("router_source") or {}).get("tensor"),
        },
        "source_map": {
            "rel": MAP_REL,
            "provenance": map_prov,
            "schema": map_doc.get("schema"),
            "status": map_doc.get("status"),
            "snapshot_agrees_on_dense_margin_min": snapshot_agrees,
        },
        "router_jacobian_approximation": jac,
        "softmax_jacobian_synthetic": sm,
        "route_margin_distribution": margin_distribution(margins),
        "compact_margin_distribution": margin_distribution(
            map_doc["routing"]["compact_top10_top11_margin"]
        ),
        "topk_boundary_sensitivity": topk_boundary_sensitivity(margins),
        "critical_hidden_directions": crit,
        "per_dimension_sensitivity": per_dimension_sensitivity_from_map(map_doc),
        "residual_budget_by_route_risk": residual_budget_by_route_risk(margins, flips),
        "precision_allocation": allocation,
        "captured_routes": {
            "dense_top10": map_doc["routing"]["dense_top10"],
            "compact_top10": map_doc["routing"]["compact_top10"],
            "top10_membership_symmetric_difference": flips,
            "rows_with_membership_change": map_doc["routing"]["rows_with_membership_change"],
        },
        "headline_question": headline["question"],
        "headline_answer": headline,
        "consumed_router_selection": _selection_excerpt(selection),
        "consumed_representation_ab": _ab_excerpt(ab),
        "consumed_margin_residual_candidate": _residual_excerpt(residual),
        "noted_conditional_state_f32": state_info,
        "recovered_implementation": recovered_implementation(
            {
                "map": map_prov,
                "selection": sel_prov,
                "ab": ab_prov,
                "residual": res_prov,
                "state_f32": state_info,
            }
        ),
        "gaps_closed": [
            "Named the logit Jacobian J=W and stated its validity range over the map's shapes.",
            "Turned the four captured margins into a distribution and an epsilon-boundary table.",
            "Split hidden space into router-visible (rank ≤ 512) vs routing-inert (≥ 2048) directions.",
            "Emitted a per-surface precision allocation instead of a scalar sensitivity score.",
            "Redirected residual budget from coordinate salience to the SVD-visible subspace, because only the latter repaired membership.",
            "Added a crush-refusal guard so control-flow bits cannot be silently assigned CRUSHED.",
        ],
        "negative_findings": [
            "tools/flash_router_sensitivity_map.py and the L3-L4 map are untracked on the parent disk; they are not in git HEAD.",
            "This sparse worktree does not materialize receipts/headless; selection/AB were read via git show.",
            "The 2560-length coordinate-salience order is not in the map, so per-index allocation cannot be emitted.",
            "Full 512-vectors of logits/probabilities are not in the map, so the softmax Jacobian cannot be evaluated on the seam.",
            "Global ||Δz||_∞ over-predicts flips; pairwise (W_i-W_j)·δx is not recoverable from the receipt.",
            "n=4 tokens, one seam (L3→L4). Not a corpus distribution.",
            "Q4 evidence is layer-0 synthetic input, not the layer-4 captured seam.",
            "Oracle coordinate residual never restored membership; this module did not re-fit it.",
            "gate.weight itself was not loaded from modellake.",
            "FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4_CONDITIONAL_STATE.f32 was noted (40 KiB = 4×2560 f32) and not interpreted.",
            "tools/headless/sensitivity_allocation.py is Qwen27 organ VoI; it does not answer this Flash control-flow question.",
            "No PROTECTED_ABSOLUTE or DIAGNOSTIC_RELATIVE bench was taken; every number here is STATIC_ONLY.",
        ],
        "what_would_need_hardware": [
            "Whether premium router bits change complete-token ns, tps, joules, or bandwidth.",
            "Native organ parity of a compiled margin-aware residual on Metal.",
        ],
    }
    return write_receipt(RECEIPT, doc, "tools/future/router_science.py")


def selftest() -> Path:
    """Prove the math and the crush-refusal, then emit the receipt."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    W = rng.normal(size=(8, 16))
    x = rng.normal(size=(16,))
    approx = router_jacobian_approximation(W=W, x=x, eps=1e-6)
    err = approx["synthetic"]["finite_difference_max_abs_error"]
    if err > 1e-6:
        raise AssertionError(f"logit Jacobian FD residual {err} exceeds 1e-6")
    if approx["synthetic"]["inert_action_max_abs"] > 1e-10:
        raise AssertionError("routing-inert direction moved logits")
    logits = rng.normal(size=(5, 12))
    m = route_margins(logits, k=4)
    if m.shape != (5,):
        raise AssertionError("route_margins shape")
    tiny = Surface(
        name="tiny",
        kind="position_residual",
        margin_min=1e-8,
        n_params=10,
        hidden=8,
        positions=1,
        position_index=3,
        k=10,
    )
    huge = Surface(
        name="huge",
        kind="position_residual",
        margin_min=1.0,
        n_params=10,
        hidden=8,
        positions=1,
        position_index=3,
        k=10,
    )
    if precision_rank(assign_bit_class(tiny)) <= precision_rank(assign_bit_class(huge)):
        raise AssertionError("allocator is not margin-driven")
    fired = False
    try:
        assign_bit_class(tiny, requested="CRUSHED")
    except ControlFlowCrushError:
        fired = True
    if not fired:
        raise AssertionError("crush refusal did not fire")
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        out = selftest()
    else:
        out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
