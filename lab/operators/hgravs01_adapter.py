"""HGRAVS01 activation-weighted low-rank candidate family for doctor6.

Wires the existing activation-weighted SVD codec
(``ascension_dual_gravity_worker._activation_weighted_svd_low_rank_codec``)
and the existing rank grid (``BUDGET_POINTS``, which includes r128 and r192)
into doctor6 as a searchable candidate family.

This module does not change compose's keep rule, the complete-artifact
ceiling, or water-fill. It only *produces* honestly billed candidates.
Rank is clamped to ``min(budget_rank, n_fit_rows)`` and the reduced rank is
reported rather than presented as the requested point.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

import numpy as np

from lab.operators import ascension_dual_gravity_worker as dual
from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    BUDGET_POINTS,
)

HGRAVS01_SCHEMA = "hawking.gravity.activation_weighted_svd_low_rank.v1"
HGRAVS01_FAMILY = "hgravs01"
HGRAVS01_REPRESENTATION = "activation_weighted_svd_low_rank"
HGRAVS01_MAGIC = dual.MAGIC_ACT_SVD  # b"HGRAVS01"
_RUNG_RE = re.compile(r"^hgravs01_r(\d+)_b(\d+)$")

# Stable capture-identity envelope so prescribe and treat emit the same
# physical payload (header JSON is part of the billed bytes). The fit itself
# uses the organ's real X_fit; this is not a synthetic activation matrix.
DOCTOR6_CAPTURE_IDENTITY: dict[str, Any] = {
    "path": "doctor6.hgravs01",
    "capture_result_path": "doctor6.hgravs01",
    "sha256": hashlib.sha256(b"hawking.doctor6.hgravs01.v1").hexdigest(),
    "schema": "hawking.doctor6.hgravs01.capture_binding.v1",
    "status": "DOCTOR6_CANDIDATE",
    "fit_kind": "real_routed_activation_capture",
    "not_synthetic_unit_direction": True,
}


def rung_name(*, rank: int, bits: int) -> str:
    return f"hgravs01_r{int(rank)}_b{int(bits)}"


def hgravs01_budgets() -> tuple[dict[str, Any], ...]:
    """Search grid. Same points the existing repack operator already measured."""
    return tuple(dict(b) for b in BUDGET_POINTS)


HGRAVS01_RUNGS: tuple[str, ...] = tuple(
    rung_name(rank=int(b["rank"]), bits=int(b["bits"])) for b in BUDGET_POINTS
)


def parse_hgravs01_rung(name: str) -> tuple[int, int] | None:
    m = _RUNG_RE.fullmatch(str(name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_hgravs01_rung(name: str) -> bool:
    return parse_hgravs01_rung(name) is not None


def clamp_rank(budget_rank: int, n_fit_rows: int) -> int:
    """``rank = min(budget_rank, n_fit_rows)`` — same law as the repack operator."""
    return min(int(budget_rank), int(n_fit_rows))


def component_bpw(payload_bytes: int, n_params: int) -> float:
    return 8.0 * int(payload_bytes) / max(int(n_params), 1)


def encode_hgravs01(
    W: np.ndarray,
    X_fit: np.ndarray,
    *,
    rank: int,
    bits: int,
    capture_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode one HGRAVS01 point. Bills the full physical container.

    The payload is magic + header JSON + left factor body (f16 scales + codes)
    + right factor body (f16 scales + codes). That is the executable artifact
    the Rust HGRAVS01 reader consumes; component BPW uses ``len(payload)``.
    """
    requested_rank = int(rank)
    n_fit_rows = int(X_fit.shape[0])
    well_posed_rank = clamp_rank(requested_rank, n_fit_rows)
    if well_posed_rank < 1:
        raise ValueError(
            f"HGRAVS01 organ has n_fit_rows={n_fit_rows}; no well-posed rank exists"
        )
    identity = dict(capture_identity or DOCTOR6_CAPTURE_IDENTITY)
    codec = dual._activation_weighted_svd_low_rank_codec(
        W,
        rank=well_posed_rank,
        bits=int(bits),
        X_fit=X_fit,
        capture_identity=identity,
        X_hold=None,
    )
    header, body = dual._parse_container(codec.payload, expected_magic=HGRAVS01_MAGIC)
    achieved_rank = int(codec.metadata.get("rank") or header.get("rank") or well_posed_rank)
    left_body = int(header["left_body_bytes"])
    right_body = int(header["right_body_bytes"])
    header_bytes = len(codec.payload) - 12 - left_body - right_body
    left_meta = dict(header.get("left") or {})
    right_meta = dict(header.get("right") or {})
    scale_bytes = int(left_meta.get("scale_bytes") or 0) + int(
        right_meta.get("scale_bytes") or 0
    )
    code_bytes = int(left_meta.get("code_bytes") or 0) + int(
        right_meta.get("code_bytes") or 0
    )
    payload_bytes = int(len(codec.payload))
    return {
        "W_hat": np.asarray(codec.reconstruction, dtype=np.float32),
        "payload": codec.payload,
        "payload_bytes": payload_bytes,
        "component_bpw": component_bpw(payload_bytes, int(W.size)),
        "requested_rank": requested_rank,
        "achieved_rank": achieved_rank,
        "rank_clamped_to_n_fit": bool(achieved_rank != requested_rank),
        "n_fit_rows": n_fit_rows,
        "bits": int(bits),
        "schema": HGRAVS01_SCHEMA,
        "family": HGRAVS01_FAMILY,
        "representation": HGRAVS01_REPRESENTATION,
        "activation_weighted": True,
        "low_rank": True,
        "hgravs": True,
        "ledger": {
            "magic_bytes": 8,
            "header_len_field_bytes": 4,
            "header_bytes": int(header_bytes),
            "left_body_bytes": left_body,
            "right_body_bytes": right_body,
            "scale_bytes": scale_bytes,
            "code_bytes": code_bytes,
            "factor_body_bytes": left_body + right_body,
            "total_bytes": payload_bytes,
            "body_matches_ledger": bool(len(body) == left_body + right_body),
        },
        "rung": rung_name(rank=requested_rank, bits=int(bits)),
    }


def encode_hgravs01_rung(
    name: str,
    W: np.ndarray,
    X_fit: np.ndarray,
    *,
    capture_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_hgravs01_rung(name)
    if parsed is None:
        raise ValueError(f"not an HGRAVS01 rung: {name!r}")
    rank, bits = parsed
    return encode_hgravs01(
        W, X_fit, rank=rank, bits=bits, capture_identity=capture_identity
    )


def evaluate_hgravs01_candidates(
    *,
    W: np.ndarray,
    X_fit: np.ndarray,
    X_hold: np.ndarray,
    W_incumbent: np.ndarray,
    target_cos: float,
    max_legal_bpw: float,
    best_cos: float,
    already_clears_target: bool,
    capture_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every HGRAVS01 rank point with compose's keep rule.

    Legality is ``component_bpw <= max_legal_bpw`` (the prescription target,
    the same ceiling the incumbent is billed against). A point over that
    ceiling is recorded and refused, never silently admitted. Rank clamp is
    reported on every row. If the organ already clears the cosine target,
    points stay on the ballot but are not kept (cheapest-sufficient).
    """
    from lab.operators.doctor6.rungs import score_vs_incumbent

    measurements: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    cur_best = float(best_cos)
    eps = 1e-4
    seen_points: set[tuple[int, int]] = set()

    for budget in hgravs01_budgets():
        requested = int(budget["rank"])
        bits = int(budget["bits"])
        well_posed = clamp_rank(requested, int(X_fit.shape[0]))
        encoded = encode_hgravs01(
            W,
            X_fit,
            rank=requested,
            bits=bits,
            capture_identity=capture_identity,
        )
        point = (int(encoded["achieved_rank"]), bits)
        duplicate_clamp = point in seen_points and encoded["rank_clamped_to_n_fit"]
        seen_points.add(point)

        local_bpw = float(encoded["component_bpw"])
        score = score_vs_incumbent(
            W=W, X_hold=X_hold, W_hat=encoded["W_hat"], W_incumbent=W_incumbent
        )
        cos = float(score["output_cosine"])
        surplus = float(score["surplus_over_incumbent"])
        legal = bool(local_bpw <= float(max_legal_bpw) + 1e-12)
        rung = encoded["rung"]

        if not legal:
            keep = False
            reason = (
                f"over_budget local_bpw={local_bpw:.4f} > {float(max_legal_bpw)}"
            )
        elif already_clears_target and winner is None:
            keep = False
            reason = "organ_already_clears_target; hgravs01 on ballot only"
        elif already_clears_target and winner is not None:
            keep = False
            reason = "stopped_cheapest_sufficient"
        elif winner is not None and cur_best >= float(target_cos):
            keep = False
            reason = "stopped_cheapest_sufficient"
        elif cos > cur_best + eps:
            keep = True
            reason = "improves_best_legal"
        else:
            keep = False
            reason = (
                f"below_incumbent_or_unhelpful: cos={cos:.4f} "
                f"best={cur_best:.4f} surplus_over_incumbent={surplus:+.4f}"
            )

        meta = {
            "codec": (
                f"hgravs01_activation_weighted_low_rank_"
                f"r{encoded['achieved_rank']}_b{bits}"
            ),
            "family": HGRAVS01_FAMILY,
            "schema": HGRAVS01_SCHEMA,
            "representation": HGRAVS01_REPRESENTATION,
            "activation_weighted": True,
            "low_rank": True,
            "hgravs": True,
            "requested_rank": requested,
            "rank": int(encoded["achieved_rank"]),
            "rank_clamped_to_n_fit": bool(encoded["rank_clamped_to_n_fit"]),
            "n_fit_rows": int(encoded["n_fit_rows"]),
            "bits": bits,
            "budget_label": str(budget.get("label") or ""),
            "well_posed_rank": int(well_posed),
            "duplicate_clamped_point": bool(duplicate_clamp),
            "ledger": encoded["ledger"],
        }
        row = {
            "rung": rung,
            "kept": keep,
            "keep_reason": reason if keep else None,
            "output_cosine": cos,
            "surplus_over_incumbent": surplus,
            "beats_incumbent": bool(score["beats_incumbent"]),
            "payload_bytes": int(encoded["payload_bytes"]),
            "component_bpw": float(local_bpw),
            "legal_under_budget": legal,
            "family": HGRAVS01_FAMILY,
            "schema": HGRAVS01_SCHEMA,
            "activation_weighted": True,
            "low_rank": True,
            "hgravs": True,
            "requested_rank": requested,
            "rank": int(encoded["achieved_rank"]),
            "rank_clamped_to_n_fit": bool(encoded["rank_clamped_to_n_fit"]),
            "n_fit_rows": int(encoded["n_fit_rows"]),
            "bits": bits,
            "meta": meta,
        }
        measurements.append(row)
        if keep:
            winner = {
                "rung": rung,
                "W_hat": encoded["W_hat"],
                "payload_bytes": int(encoded["payload_bytes"]),
                "component_bpw": float(local_bpw),
                "output_cosine": cos,
                "meta": meta,
            }
            cur_best = cos
        else:
            dropped.append(
                {
                    "rung": rung,
                    "reason": reason,
                    "cosine_after": cos,
                    "surplus_over_incumbent": surplus,
                    "component_bpw": float(local_bpw),
                    "requested_rank": requested,
                    "rank": int(encoded["achieved_rank"]),
                    "rank_clamped_to_n_fit": bool(encoded["rank_clamped_to_n_fit"]),
                    "family": HGRAVS01_FAMILY,
                    "hgravs": True,
                    "activation_weighted": True,
                    "low_rank": True,
                }
            )

    return {
        "measurements": measurements,
        "dropped": dropped,
        "winner": winner,
    }


def apply_hgravs01_to_compose_result(
    result: dict[str, Any],
    *,
    W: np.ndarray,
    X_fit: np.ndarray,
    X_hold: np.ndarray,
    target_cos: float,
    max_legal_bpw: float,
    capture_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Widen a compose result's ballot with HGRAVS01; same keep rule."""
    W_inc = result.get("W_incumbent")
    if W_inc is None:
        raise ValueError("compose result lacks W_incumbent")
    already = bool(result.get("clears_target")) and not bool(
        result.get("fallback_to_incumbent")
    )
    ev = evaluate_hgravs01_candidates(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        W_incumbent=W_inc,
        target_cos=float(target_cos),
        max_legal_bpw=float(max_legal_bpw),
        best_cos=float(result["prescribed_cosine"]),
        already_clears_target=already,
        capture_identity=capture_identity,
    )
    measurements = list(result.get("measurements") or [])
    measurements.extend(ev["measurements"])
    dropped = list(result.get("dropped_rungs") or [])
    dropped.extend(ev["dropped"])
    result["measurements"] = measurements
    result["dropped_rungs"] = dropped
    result["hgravs01_on_ballot"] = True
    result["hgravs01_n_candidates"] = len(ev["measurements"])

    winner = ev["winner"]
    if winner is None:
        return result

    result["chain"] = [winner["rung"]]
    result["W_hat"] = winner["W_hat"]
    result["payload_bytes"] = int(winner["payload_bytes"])
    result["component_bpw"] = float(winner["component_bpw"])
    result["prescribed_cosine"] = float(winner["output_cosine"])
    result["surplus_over_incumbent"] = float(
        winner["output_cosine"] - float(result["incumbent_cosine"])
    )
    result["clears_target"] = bool(winner["output_cosine"] >= float(target_cos))
    result["deficit_vs_target"] = float(target_cos) - float(winner["output_cosine"])
    result["fallback_to_incumbent"] = False
    result["meta"] = dict(winner["meta"])
    return result
