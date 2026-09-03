"""Focused tests for Q30's unrun source-BF16 oracle contract."""
from __future__ import annotations

import numpy as np
import pytest

from lab.operators import ascension_qwen30_quality_repack_source_oracle_three_way_contract as contract


def test_three_way_metric_requires_strict_candidate_improvement() -> None:
    source = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    control = np.asarray([1.8, -1.2, 3.8], dtype=np.float32)
    candidate = np.asarray([1.1, -1.9, 3.1], dtype=np.float32)
    result = contract.evaluate_three_way_vectors(source=source, control=control, candidate=candidate)
    assert result["candidate_strictly_improves_over_control"] is True
    assert result["source_to_candidate_relative_l2"] < result["source_to_control_relative_l2"]


def test_three_way_metric_rejects_non_finite_source() -> None:
    source = np.asarray([1.0, np.nan], dtype=np.float32)
    with pytest.raises(contract.SourceOracleContractError, match="non-finite"):
        contract.evaluate_three_way_vectors(source=source, control=source.copy(), candidate=source.copy())


def test_raw_payload_geometry_is_full_vocab_only() -> None:
    result = contract._raw_vector_requirements(vocab_rows=151_936)
    assert result["bytes_per_full_logit_vector"] == 607_744
    assert result["required_payload_count"] == 6
    assert result["required_total_payload_bytes"] == 3_646_464
    assert all(name.endswith(".f32le") for name in result["payload_filenames"])
