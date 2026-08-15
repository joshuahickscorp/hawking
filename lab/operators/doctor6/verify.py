"""doctor6 verify — coherence screen + optional generation probe.

The coherence screen is the cheap necessary-condition gate. A real generation
probe is opt-in (no full-model repack in this lane).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from lab.operators.doctor6.coherence import (
    COMPOSITION_FLOOR,
    N_LAYERS,
    predict_composition,
    screen_organ_cosines,
)


def verify(
    *,
    artifact: Path | None = None,
    organ_cosines: Sequence[float] | None = None,
    organ_records: list[dict[str, Any]] | None = None,
    n_layers: int = N_LAYERS,
    composition_floor: float = COMPOSITION_FLOOR,
    run_generation: bool = False,
) -> dict[str, Any]:
    """Verify an artifact or an explicit organ-cosine vector.

    Ground-truth acceptance hooks:
      source-gated-v2 mean ≈ 0.685 → must REFUSE (incoherent)
      uniform-q4-g64 mean ≈ 0.987 → must PASS (coherent)
    """
    if organ_cosines is not None:
        verdict = screen_organ_cosines(
            organ_cosines, n_layers=n_layers, composition_floor=composition_floor
        )
        source = "explicit_organ_cosines"
    elif organ_records is not None:
        verdict = predict_composition(
            organ_records, n_layers=n_layers, composition_floor=composition_floor
        )
        source = "explicit_organ_records"
    elif artifact is not None:
        artifact = Path(artifact)
        data = json.loads(artifact.read_text())
        source = str(artifact)
        # Accept prescription / treatment / selection receipts.
        if "predicted_vs_actual" in data:
            rows = data["predicted_vs_actual"]
            cos_key = "actual_cosine" if "actual_cosine" in (rows[0] if rows else {}) else "predicted_cosine"
            records = [
                {
                    "layer": r.get("layer"),
                    "component": r.get("component"),
                    "output_cosine": r.get(cos_key, r.get("output_cosine")),
                }
                for r in rows
            ]
            verdict = predict_composition(
                records, n_layers=n_layers, composition_floor=composition_floor
            )
        elif "organs" in data and data["organs"] and isinstance(data["organs"][0], dict):
            rows = data["organs"]
            # source-gated selection receipt style
            if "output_cosine" in rows[0] or "prescribed_cosine" in rows[0]:
                records = [
                    {
                        "layer": r.get("layer"),
                        "component": r.get("component"),
                        "output_cosine": r.get(
                            "prescribed_cosine",
                            r.get("output_cosine", r.get("actual_cosine")),
                        ),
                    }
                    for r in rows
                ]
                verdict = predict_composition(
                    records, n_layers=n_layers, composition_floor=composition_floor
                )
            else:
                # Try selection receipt mean
                mean = (
                    data.get("selected_representation", {})
                    .get("summary", {})
                    .get("mean_output_cosine")
                )
                if mean is None:
                    mean = (
                        data.get("quality_summary", {})
                        .get("activation_aware", {})
                        .get("mean_output_cosine_selected")
                    )
                if mean is None:
                    raise ValueError(f"cannot extract organ cosines from {artifact}")
                verdict = screen_organ_cosines(
                    [float(mean)], n_layers=n_layers, composition_floor=composition_floor
                )
        else:
            mean = (
                data.get("selected_representation", {})
                .get("summary", {})
                .get("mean_output_cosine")
            )
            if mean is None:
                mean = (
                    data.get("quality_summary", {})
                    .get("activation_aware", {})
                    .get("mean_output_cosine_selected")
                )
            if mean is None:
                raise ValueError(f"cannot extract organ cosines from {artifact}")
            verdict = screen_organ_cosines(
                [float(mean)], n_layers=n_layers, composition_floor=composition_floor
            )
    else:
        raise ValueError("verify requires --artifact or organ cosines/records")

    generation: dict[str, Any] = {
        "ran": False,
        "note": (
            "generation probe skipped (no full-model repack in this lane); "
            "coherence screen is the cheap necessary-condition gate"
        ),
    }
    if run_generation:
        generation = {
            "ran": False,
            "status": "NOT_IMPLEMENTED_IN_LANE",
            "note": (
                "full generation requires a packed artifact + HCLI runtime; "
                "out of scope for doctor6 lane (no full-model repack)"
            ),
        }

    return {
        "schema": "hawking.doctor6.verify.v1",
        "source": source,
        "coherence_screen": verdict.as_dict(),
        "verdict": "COHERENT" if verdict.pass_gate else "INCOHERENT_REFUSE",
        "generation": generation,
    }


def verify_ground_truth_pair() -> dict[str, Any]:
    """Prove the screen against known campaign ground truth."""
    bad = screen_organ_cosines([0.685] * 48)
    good = screen_organ_cosines([0.987] * 48)
    return {
        "schema": "hawking.doctor6.coherence_ground_truth.v1",
        "source_gated_v2": {
            "mean_organ_cos": 0.685,
            "verdict": bad.as_dict(),
            "must_refuse": True,
            "refused": not bad.pass_gate,
            "pass": not bad.pass_gate,
        },
        "uniform_q4_g64": {
            "mean_organ_cos": 0.987,
            "verdict": good.as_dict(),
            "must_pass": True,
            "passed": good.pass_gate,
            "pass": good.pass_gate,
        },
        "all_pass": (not bad.pass_gate) and good.pass_gate,
    }
