"""Pure CPU checks for the captured-current-trace L0/E0 chain discriminator."""
from __future__ import annotations

import numpy as np

from lab.operators.ascension_qwen30_quality_repack_current_hcli_l0_e0_chain import (
    INSUFFICIENT_STATUS,
    SUCCESS_STATUS,
    _assessment,
    _relative_under,
    _stage_metrics,
)


def _row(improvement: float) -> dict[str, object]:
    return {
        "probe_id": "literal_hawking",
        "position": 1,
        "chain_errors": {
            "down": {"candidate_relative_l2_improvement_vs_baseline": improvement},
        },
    }


def test_assessment_requires_improvement_on_every_actual_selected_position() -> None:
    earned = _assessment([_row(0.1), {**_row(0.2), "probe_id": "json_status"}, {**_row(0.3), "probe_id": "python_add"}])
    rejected = _assessment([_row(0.1), {**_row(0.0), "probe_id": "json_status"}, {**_row(0.3), "probe_id": "python_add"}])
    assert earned["status"] == SUCCESS_STATUS
    assert "PREPARE_ONLY" in earned["next_action"]
    assert rejected["status"] == INSUFFICIENT_STATUS
    assert "TARGET_BROADER" in rejected["next_action"]


def test_stage_metrics_report_gate_up_swiglu_and_down_boundaries() -> None:
    source = {
        "gate": np.array([2.0], dtype=np.float32),
        "up": np.array([3.0], dtype=np.float32),
        "swiglu": np.array([4.0], dtype=np.float32),
        "down": np.array([5.0, 1.0], dtype=np.float32),
    }
    baseline = {name: value * np.float32(0.5) for name, value in source.items()}
    candidate = {name: value * np.float32(0.75) for name, value in source.items()}
    metrics = _stage_metrics(source, baseline, candidate)
    assert tuple(metrics) == ("gate", "up", "swiglu", "down")
    assert metrics["down"]["candidate_relative_l2_improvement_vs_baseline"] > 0.0
    assert metrics["swiglu"]["source_to_candidate_metrics"]["finite"] is True


def test_hidden_relative_path_cannot_escape_capture_root(tmp_path) -> None:
    allowed = _relative_under(tmp_path, "hidden/probe/000001.f32le", label="test")
    assert allowed == (tmp_path / "hidden/probe/000001.f32le").resolve()
    try:
        _relative_under(tmp_path, "../escape.f32le", label="test")
    except Exception as exc:  # The public error class is intentionally not part of test setup.
        assert "escapes" in str(exc)
    else:
        raise AssertionError("escape path must fail closed")
