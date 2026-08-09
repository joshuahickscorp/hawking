"""Focused pure-CPU regression for the HQ30GR2 causal-coverage discriminator."""
from __future__ import annotations

from lab.operators.ascension_qwen30_quality_repack_causal_coverage import (
    CoverageError,
    analyze_mutation_coverage,
)


def _tensor(name: str, sha: str, changed: bool) -> dict[str, object]:
    return {
        "tensor_name": name,
        "artifact_sha256": sha,
        "candidate_mutation": {"changed_from_admitted_control": changed},
    }


def test_coverage_proves_only_layer_zero_has_direct_hq30gr2_mutation() -> None:
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.24.mlp.experts.0.gate_proj.weight",
        "model.layers.24.mlp.experts.0.up_proj.weight",
        "model.layers.47.mlp.experts.0.gate_proj.weight",
        "model.layers.47.mlp.experts.0.up_proj.weight",
    ]
    baseline = {"tensors": [{"tensor_name": name, "artifact_sha256": f"b{index}"} for index, name in enumerate(names)]}
    candidate = {
        "tensors": [
            _tensor(name, f"c{index}" if index < 2 else f"b{index}", index < 2)
            for index, name in enumerate(names)
        ]
    }
    result = analyze_mutation_coverage(baseline, candidate, selected_organs=names[:2])
    assert result["changed_layer_indices"] == [0]
    assert result["depth_bands"]["early"]["directly_changed_organs"] == names[:2]
    assert result["depth_bands"]["middle"]["all_layer_payload_hashes_match_admitted_control"] is True
    assert result["depth_bands"]["late"]["all_layer_payload_hashes_match_admitted_control"] is True
    assert result["direct_mutation_covers_all_representative_depths"] is False


def test_coverage_refuses_an_unselected_payload_change() -> None:
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.24.mlp.experts.0.gate_proj.weight",
        "model.layers.24.mlp.experts.0.up_proj.weight",
        "model.layers.47.mlp.experts.0.gate_proj.weight",
        "model.layers.47.mlp.experts.0.up_proj.weight",
    ]
    baseline = {"tensors": [{"tensor_name": name, "artifact_sha256": f"b{index}"} for index, name in enumerate(names)]}
    candidate = {
        "tensors": [
            _tensor(name, f"c{index}" if index in {0, 1, 2} else f"b{index}", index < 2)
            for index, name in enumerate(names)
        ]
    }
    try:
        analyze_mutation_coverage(baseline, candidate, selected_organs=names[:2])
    except CoverageError as exc:
        assert "unselected control payload" in str(exc)
    else:
        raise AssertionError("unselected mutation must fail closed")
