from __future__ import annotations

import numpy as np
import pytest

from tools.odyssey.stacked_expert_lowrank_sparse_repair import _bf16, screen


def test_bf16_rounding_is_deterministic() -> None:
    values = np.array([1.0, 2.0, 0.1], dtype=np.float32)
    np.testing.assert_array_equal(_bf16(values), _bf16(values))
    assert _bf16(np.array([1.0], dtype=np.float32))[0] == 1.0


def test_sparse_repair_bills_every_entry_and_improves_error() -> None:
    weight = np.diag(np.array([8.0, 4.0, 2.0, 1.0], dtype=np.float32))
    rows = screen(weight, [1, 2], [0.0, 0.25])
    assert len(rows) == 4
    by_rank = {row["rank"]: row for row in rows if row["residual_fraction"] == 0.0}
    assert by_rank[1]["complete_bytes"] > by_rank[1]["factor_bytes"]
    assert by_rank[1]["metadata_bytes"] == 128
    assert by_rank[1]["decoder_support_bytes"] == 65536
    for rank in (1, 2):
        base = next(row for row in rows if row["rank"] == rank and row["residual_fraction"] == 0.0)
        repaired = next(row for row in rows if row["rank"] == rank and row["residual_fraction"] == 0.25)
        assert repaired["relative_l2"] <= base["relative_l2"]
        assert repaired["complete_bytes"] >= base["complete_bytes"]


def test_screen_rejects_non_matrix() -> None:
    with pytest.raises(ValueError):
        screen(np.ones((2, 2, 2), dtype=np.float32), [1], [0.0])


def test_screen_reports_teacher_activation_output_without_changing_billing() -> None:
    weight = np.diag(np.array([8.0, 4.0, 2.0, 1.0], dtype=np.float32))
    activation = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
    rows = screen(
        weight,
        [1],
        [0.0, 0.25],
        source_activation=activation,
        repair_selection="activation_weighted",
    )
    base, repaired = rows
    assert base["complete_bytes"] < repaired["complete_bytes"]
    assert base["source_activation_output"]["count"] == 4
    assert base["source_activation_output"]["finite"] is True
    assert repaired["source_activation_output"]["relative_l2"] <= base["source_activation_output"]["relative_l2"]


def test_activation_weighted_selection_requires_matching_activation() -> None:
    with pytest.raises(ValueError, match="requires a source activation"):
        screen(
            np.eye(2, dtype=np.float32),
            [1],
            [0.0],
            repair_selection="activation_weighted",
        )
    with pytest.raises(ValueError, match="match the matrix columns"):
        screen(
            np.eye(2, dtype=np.float32),
            [1],
            [0.0],
            source_activation=np.ones(3, dtype=np.float32),
        )
