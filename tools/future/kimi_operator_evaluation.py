"""Functional-simulation evaluation for the KIMI operator matrix.

This is the cheap candidate gate between registration and a clean-process live
model run. It exercises every OPA--OPG transformation on the same copied base,
checks that the immutable input remains byte-identical, and records Tabula
provenance. It intentionally does not call a model, score capability, or create
an artifact; those claims require separate evidence axes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.kimi_operator_candidates import (
    OperatorCandidate,
    apply_candidate,
    catalog,
)


SCHEMA = "hawking.future.kimi_operator_evaluation.v1"


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _fixture() -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(20260910)
    weights = {
        "model.layers.0.mlp.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.layers.0.mlp.shared_experts.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.layers.0.mlp.experts.2.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.layers.1.mlp.down_proj.weight": rng.normal(size=(8, 6)).astype(np.float32),
        "model.embed_tokens.weight": rng.normal(size=(10, 8)).astype(np.float32),
    }
    directions = {layer: rng.normal(size=8) for layer in (0, 1)}
    input_directions = {layer: rng.normal(size=6) for layer in (0, 1)}
    subspaces = {
        layer: np.column_stack((directions[layer], rng.normal(size=(8, 3))))
        for layer in (0, 1)
    }
    return weights, directions, input_directions, subspaces


def evaluate_candidates(
    candidates: Sequence[OperatorCandidate] | None = None,
    *,
    code_commit: str = "UNRECORDED",
) -> dict[str, Any]:
    """Evaluate all registered candidates without touching a model artifact."""
    weights, directions, input_directions, subspaces = _fixture()
    base_snapshot = {key: np.array(value, copy=True) for key, value in weights.items()}
    base_hash = _array_hash(
        np.concatenate([np.ascontiguousarray(base_snapshot[key]).reshape(-1)
                        for key in sorted(base_snapshot)])
    )
    rows: list[dict[str, Any]] = []
    candidates = tuple(candidates or catalog(
        n_layers=2,
        target_layers=(0, 1),
        selected_experts={0: (2,)},
        subspace_rank=4,
    ))
    for candidate in candidates:
        after, receipt = apply_candidate(
            weights,
            candidate,
            directions,
            input_directions=input_directions,
            subspaces=subspaces,
            code_commit=code_commit,
        )
        base_unchanged = all(
            np.array_equal(weights[key], base_snapshot[key]) for key in weights
        )
        output_changed = any(
            not np.array_equal(after[key], base_snapshot[key]) for key in after
        )
        touched_delta = sum(
            float(np.linalg.norm(
                after[row["key"]].astype(np.float64)
                - base_snapshot[row["key"]].astype(np.float64)
            ))
            for row in receipt["touched_tensors"]
        )
        rows.append({
            "candidate_id": candidate.candidate_id,
            "method": candidate.method,
            "functional_sim_passed": bool(
                base_unchanged
                and output_changed
                and receipt["weights_written"] is False
                and receipt["promotion"] == "NOT_PERFORMED"
                and receipt["source_model"] == "KIMI_BASE"
            ),
            "base_unchanged": base_unchanged,
            "output_changed": output_changed,
            "touched_tensors": len(receipt["touched_tensors"]),
            "touched_layers": receipt["touched_layers"],
            "touched_experts": receipt["touched_experts"],
            "copied_state_sha256": receipt["artifact_hash"],
            "touched_delta_l2": touched_delta,
            "reversible": candidate.reversible,
            "evidence_tier": "FUNCTIONAL_SIM",
            "behavioral_claim": "NOT_MEASURED",
            "capability_claim": "NOT_MEASURED",
            "physical_claim": "NOT_MEASURED",
            "weights_written": False,
            "promotion": "NOT_PERFORMED",
        })
    return {
        "schema": SCHEMA,
        "status": "FUNCTIONAL_SIM_COMPLETE" if all(
            row["functional_sim_passed"] for row in rows
        ) else "FUNCTIONAL_SIM_FAILED",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "base_fixture_sha256": base_hash,
        "candidate_count": len(rows),
        "candidates": rows,
        "candidate_matrix": ["OPA", "OPB", "OPC", "OPD", "OPE", "OPF", "OPG"],
        "ranking": None,
        "claim_boundary": (
            "Functional simulation only. This receipt proves copied-state and "
            "provenance behavior for the candidate matrix; it does not establish "
            "model behavior, capability, epistemic calibration, authorization, "
            "hardware parity, or promotion readiness."
        ),
    }


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "WORKTREE_UNCOMMITTED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_OPERATOR_EVALUATION_SIM_20260910.json",
    )
    args = parser.parse_args()
    result = evaluate_candidates(code_commit=_git_head())
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["recorded_by"] = "tools/future/kimi_operator_evaluation.py"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "candidate_count": result["candidate_count"],
    }, indent=2))
    return 0 if result["status"] == "FUNCTIONAL_SIM_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
