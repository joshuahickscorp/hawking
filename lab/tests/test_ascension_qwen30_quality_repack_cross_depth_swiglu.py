"""Pure CPU checks for the isolated cross-depth frozen-SwiGLU diagnostic."""
from __future__ import annotations

import numpy as np

from lab.operators.ascension_qwen30_quality_repack_cross_depth_swiglu import (
    _assess,
    compare_depth,
)


def test_cross_depth_panel_is_deterministic_and_control_identical() -> None:
    gate = np.array([[0.5, -0.25], [0.75, 0.125]], dtype=np.float32)
    up = np.array([[0.3, 0.2], [-0.1, 0.8]], dtype=np.float32)
    first = compare_depth(
        layer=24,
        source_gate=gate,
        source_up=up,
        baseline_gate=gate,
        baseline_up=up,
        candidate_gate=gate,
        candidate_up=up,
    )
    second = compare_depth(
        layer=24,
        source_gate=gate,
        source_up=up,
        baseline_gate=gate,
        baseline_up=up,
        candidate_gate=gate,
        candidate_up=up,
    )
    assert first["frozen_activation_panel"]["f32le_sha256"] == second["frozen_activation_panel"]["f32le_sha256"]
    assert first["baseline_to_candidate_metrics"]["max_abs"] == 0.0
    assert first["candidate_relative_l2_improvement_vs_baseline"] == 0.0


def test_assessment_never_promotes_local_improvement_to_coherence() -> None:
    rows = [
        {
            "layer": 0,
            "candidate_relative_l2_improvement_vs_baseline": 0.2,
            "baseline_to_candidate_metrics": {"max_abs": 1.0},
        },
        {
            "layer": 24,
            "candidate_relative_l2_improvement_vs_baseline": 0.0,
            "baseline_to_candidate_metrics": {"max_abs": 0.0},
        },
        {
            "layer": 47,
            "candidate_relative_l2_improvement_vs_baseline": 0.0,
            "baseline_to_candidate_metrics": {"max_abs": 0.0},
        },
    ]
    assessment = _assess(rows)
    assert assessment["global_coherence_causal_reach"] == "NOT_EARNED"
    assert assessment["middle_and_late_candidate_payload_effect_on_frozen_panel"] == "EXACTLY_ZERO_CONTROL_MATCH"
