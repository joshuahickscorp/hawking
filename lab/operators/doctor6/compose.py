"""Per-organ chain composition: cheapest-sufficient, drop unhelpful rungs.

Aggressive means adaptive, not more of everything:
  - per-organ chains rather than one global chain
  - escalate where deficit is largest (down_proj, band2)
  - stop the moment an organ clears the layer target
  - measure each rung vs the INCUMBENT; drop where it does not help
"""
from __future__ import annotations

from typing import Any

import numpy as np

from lab.operators.doctor6.coherence import LAYER_TARGET_COS
from lab.operators.doctor6.rungs import (
    RUNG_ORDER,
    apply_rung,
    quant_binary,
    score_vs_incumbent,
)

# Candidate rungs after the incumbent baseline, in leverage order.
ESCALATION: tuple[str, ...] = (
    "l0_calib",
    "l1_awq",
    "l1_rotation",
    "l2_mixed_prec",
    "l3_outlier_residual",
    "l4_block_qat",
)

# Component effort prior from measured post-L0-L4 medians:
#   gate 0.885, up 0.801, down 0.745  against target 0.9857
COMPONENT_PRIORITY: dict[str, int] = {
    "down_proj": 0,  # escalate first / keep more rungs
    "up_proj": 1,
    "gate_proj": 2,
}


def _min_rungs_for_component(component: str, deficit: float) -> list[str]:
    """Bias the starting chain by organ type and deficit size."""
    if component == "down_proj" or deficit >= 0.15:
        return ["l0_calib", "l1_awq", "l1_rotation", "l2_mixed_prec"]
    if component == "up_proj" or deficit >= 0.08:
        return ["l0_calib", "l1_awq", "l1_rotation"]
    return ["l0_calib", "l1_awq"]


def compose_organ_chain(
    *,
    W: np.ndarray,
    X_fit: np.ndarray,
    X_hold: np.ndarray,
    organ_key: str,
    component: str,
    sensitivity: float,
    seed: int,
    device: str = "cpu",
    qat_steps: int = 40,
    target_cos: float = LAYER_TARGET_COS,
    measure_all: bool = False,
    max_legal_bpw: float = 0.98,
) -> dict[str, Any]:
    """Compose a cheapest-sufficient under-budget chain for one organ.

    Scoring is always against the INCUMBENT binary representation on the same
    holdout rows (never a constant-mean null). The binary organ is a quality
    *reference* only: HQ30G1B1 projects above the 1.0 complete-BPW law, so the
    prescribed chain is chosen on the under-budget track. Among legal rungs we
    keep only those that improve holdout cosine over the current best legal
    reconstruction and drop the rest (adaptive, not "more of everything").
    """
    # Incumbent binary = live packed organ (quality reference; often over budget).
    W_inc, inc_bytes = quant_binary(W)
    inc_score = score_vs_incumbent(
        W=W, X_hold=X_hold, W_hat=W_inc, W_incumbent=W_inc
    )
    incumbent_cos = float(inc_score["output_cosine"])
    deficit0 = float(target_cos - incumbent_cos)
    inc_bpw = 8.0 * inc_bytes / max(W.size, 1)

    measurements: list[dict[str, Any]] = [
        {
            "rung": "incumbent_binary",
            "kept": False,
            "role": "quality_reference_over_budget"
            if inc_bpw > max_legal_bpw
            else "quality_reference",
            "output_cosine": incumbent_cos,
            "surplus_over_incumbent": 0.0,
            "payload_bytes": int(inc_bytes),
            "component_bpw": float(inc_bpw),
            "legal_under_budget": bool(inc_bpw <= max_legal_bpw + 1e-12),
        }
    ]

    # Under-budget track starts empty; first legal rung that runs becomes the base.
    best_W: np.ndarray | None = None
    best_cos = float("-inf")
    best_bytes = 0
    best_meta: dict[str, Any] = {}
    chain: list[str] = []
    dropped: list[dict[str, Any]] = []

    # Preferential order by organ type / deficit; down_proj escalates further.
    preferred = _min_rungs_for_component(component, deficit0)
    ordered: list[str] = []
    for r in preferred:
        if r not in ordered:
            ordered.append(r)
    for r in ESCALATION:
        if r not in ordered:
            ordered.append(r)

    for rung in ordered:
        if not measure_all and best_cos >= target_cos and best_W is not None:
            dropped.append(
                {
                    "rung": rung,
                    "reason": "organ_already_clears_target; skip further rungs",
                    "cosine_before": best_cos,
                }
            )
            continue

        qat_bits = 3 if component == "down_proj" else 2
        try:
            result = apply_rung(
                rung,
                W,
                X_fit,
                organ_key=organ_key,
                sensitivity=sensitivity,
                seed=seed,
                device=device,
                qat_steps=qat_steps,
                qat_bits=qat_bits,
                prefer_budget=True,
            )
        except Exception as exc:  # noqa: BLE001
            dropped.append(
                {
                    "rung": rung,
                    "reason": f"rung_raised: {exc.__class__.__name__}: {exc}",
                    "cosine_before": best_cos if best_W is not None else None,
                }
            )
            continue

        local_bpw = 8.0 * result.payload_bytes / max(W.size, 1)
        score = score_vs_incumbent(
            W=W, X_hold=X_hold, W_hat=result.W_hat, W_incumbent=W_inc
        )
        cos = float(score["output_cosine"])
        surplus = float(score["surplus_over_incumbent"])
        legal = bool(local_bpw <= max_legal_bpw + 1e-12)

        # Keep rule (under-budget track):
        #  1. must be legal under max_legal_bpw
        #  2. must improve on current best legal cosine (or be the first legal)
        #  3. L4 must not regress (measured 1-bit STE pathology generalizes)
        eps = 1e-4
        if not legal:
            keep = False
            reason = f"over_budget local_bpw={local_bpw:.4f} > {max_legal_bpw}"
        elif best_W is None:
            keep = True
            reason = "first_legal_base"
        elif cos > best_cos + eps:
            keep = True
            reason = "improves_best_legal"
        else:
            keep = False
            reason = (
                f"measured_unhelpful: cos={cos:.4f} best_legal={best_cos:.4f} "
                f"surplus_over_incumbent={surplus:+.4f}"
            )

        row = {
            "rung": rung,
            "kept": keep,
            "keep_reason": reason if keep else None,
            "output_cosine": cos,
            "surplus_over_incumbent": surplus,
            "beats_incumbent": bool(score["beats_incumbent"]),
            "payload_bytes": int(result.payload_bytes),
            "component_bpw": float(local_bpw),
            "legal_under_budget": legal,
            "meta": result.meta,
        }
        measurements.append(row)

        if keep:
            chain.append(rung)
            best_W = result.W_hat
            best_cos = cos
            best_bytes = int(result.payload_bytes)
            best_meta = dict(result.meta)
            # Cheapest sufficient: stop when target cleared.
            if best_cos >= target_cos and not measure_all:
                for rest in ordered[ordered.index(rung) + 1 :]:
                    dropped.append(
                        {
                            "rung": rest,
                            "reason": "stopped_cheapest_sufficient",
                            "cosine_before": best_cos,
                        }
                    )
                break
        else:
            dropped.append(
                {
                    "rung": rung,
                    "reason": reason,
                    "cosine_before": best_cos if best_W is not None else None,
                    "cosine_after": cos,
                    "surplus_over_incumbent": surplus,
                    "component_bpw": float(local_bpw),
                }
            )

        # After preferred prefix is fully measured: robust organs stop escalating;
        # down_proj continues into L2/L3 (discriminating organ).
        if (
            not measure_all
            and rung == preferred[-1]
            and component in ("gate_proj", "up_proj")
            and best_W is not None
            and best_cos < target_cos
        ):
            for rest in ordered[ordered.index(rung) + 1 :]:
                dropped.append(
                    {
                        "rung": rest,
                        "reason": (
                            f"{component}_preferred_prefix_complete; "
                            "further rungs reserved for down_proj deficit"
                        ),
                        "cosine_before": best_cos,
                    }
                )
            break

    if best_W is None:
        # Absolute fallback: legal act-SVD base even if scoring failed to keep.
        from lab.operators.doctor6.rungs import apply_rung as _ar

        fallback = _ar(
            "l0_calib",
            W,
            X_fit,
            organ_key=organ_key,
            sensitivity=sensitivity,
            seed=seed,
            prefer_budget=True,
        )
        best_W = fallback.W_hat
        best_bytes = int(fallback.payload_bytes)
        best_meta = dict(fallback.meta)
        score = score_vs_incumbent(
            W=W, X_hold=X_hold, W_hat=best_W, W_incumbent=W_inc
        )
        best_cos = float(score["output_cosine"])
        chain = ["l0_calib"]

    return {
        "organ_key": organ_key,
        "component": component,
        "chain": chain,
        "dropped_rungs": dropped,
        "measurements": measurements,
        "incumbent_cosine": incumbent_cos,
        "incumbent_component_bpw": float(inc_bpw),
        "prescribed_cosine": best_cos,
        "surplus_over_incumbent": best_cos - incumbent_cos,
        "payload_bytes": best_bytes,
        "component_bpw": 8.0 * best_bytes / max(W.size, 1),
        "clears_target": bool(best_cos >= target_cos),
        "deficit_vs_target": float(target_cos - best_cos),
        "sensitivity": float(sensitivity),
        "meta": best_meta,
        "W_hat": best_W,
        "W_incumbent": W_inc,
    }


def compose_status_summary(organ_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_comp: dict[str, list[str]] = {}
    for r in organ_results:
        by_comp.setdefault(r["component"], [])
        chain_s = "+".join(r["chain"])
        if chain_s not in by_comp[r["component"]]:
            by_comp[r["component"]].append(chain_s)
    return {
        "n_organs": len(organ_results),
        "n_clear_target": sum(1 for r in organ_results if r["clears_target"]),
        "mean_prescribed_cosine": float(
            np.mean([r["prescribed_cosine"] for r in organ_results])
        )
        if organ_results
        else float("nan"),
        "mean_incumbent_cosine": float(
            np.mean([r["incumbent_cosine"] for r in organ_results])
        )
        if organ_results
        else float("nan"),
        "distinct_chains_by_component": by_comp,
        "chains_differ_across_components": len({"+".join(r["chain"]) for r in organ_results})
        > 1,
    }
