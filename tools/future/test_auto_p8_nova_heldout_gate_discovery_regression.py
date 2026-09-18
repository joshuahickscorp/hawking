"""Focused regression for the P8 NOVA held-out gate discovery tranche.

Exercises the newly added ``candidate_matrix_complete`` observable in
``tools.future.kimi_operator_evaluation.evaluate_candidates``. This is a
functional-simulation check only: it does not call a model, score capability,
write weights, or qualify hardware.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.kimi_operator_evaluation import evaluate_candidates


def test_ranking_is_deterministic_and_ordered_by_touched_delta() -> None:
    receipt = evaluate_candidates()
    rows = receipt["candidates"]
    assert receipt["ranking"] is None
    assert receipt["ranking_basis"] == "touched_delta_l2_desc"
    ranked = receipt["ranked_candidate_ids"]
    assert len(ranked) == len(rows)
    assert sorted(ranked) == sorted(row["candidate_id"] for row in rows)
    deltas = {row["candidate_id"]: row["touched_delta_l2"] for row in rows}
    ordered = [deltas[candidate_id] for candidate_id in ranked]
    assert ordered == sorted(ordered, reverse=True)
    assert receipt["ranked_candidate_ids"] == evaluate_candidates()["ranked_candidate_ids"]


EXPECTED_MATRIX = ["OPA", "OPB", "OPC", "OPD", "OPE", "OPF", "OPG"]


def test_candidate_matrix_complete_true_for_full_catalog() -> None:
    receipt = evaluate_candidates()
    assert receipt["candidate_matrix"] == EXPECTED_MATRIX
    assert receipt["candidate_matrix_complete"] is True
    assert receipt["status"] == "FUNCTIONAL_SIM_COMPLETE"
    assert receipt["candidate_count"] == len(EXPECTED_MATRIX)


def test_candidate_matrix_complete_false_for_partial_catalog() -> None:
    from tools.future.kimi_operator_candidates import catalog

    partial = tuple(catalog(
        n_layers=2,
        target_layers=(0, 1),
        selected_experts={0: (2,)},
        subspace_rank=4,
    ))[:3]
    receipt = evaluate_candidates(partial)
    assert receipt["candidate_matrix_complete"] is False
    assert receipt["candidate_count"] == 3


def test_candidate_matrix_complete_is_boolean_not_marker() -> None:
    receipt = evaluate_candidates()
    assert isinstance(receipt["candidate_matrix_complete"], bool)
    assert receipt["immutable_base"] is True
    assert receipt["ranking"] is None
    assert receipt["claim_boundary"].startswith("Functional simulation only.")