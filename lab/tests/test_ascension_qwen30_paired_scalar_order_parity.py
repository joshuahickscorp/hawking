"""Regression coverage for the Qwen30 paired scalar-order handoff."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_paired_scalar_order_parity as parity


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _device_facts(forwards: int) -> dict[str, object]:
    return {
        "device_parity": {
            "enabled": True,
            "valid": True,
            "all_selected_route_major_activations_compared_on_device": True,
            "full_model_forwards_without_device_parity": 0,
            "full_model_forwards_compared": forwards,
            "layers_compared": forwards * 48,
            "max_abs_error": 0.0,
            "tolerance_max_abs": 0.004,
        }
    }


def _completion() -> dict[str, dict[str, object]]:
    return {
        "prompt_token_ids": {"control": [1, 2], "candidate": [1, 2]},
        "completion_token_ids": {"control": [3, 4], "candidate": [3, 4]},
        "full_model_forward_count": {"control": 4, "candidate": 4},
        "completion_feedback_full_forwards": {"control": 2, "candidate": 2},
    }


def test_production_no_parity_handoff_binds_current_parity_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serving option must be prepared from sealed, current A/B evidence only."""

    runtime_path = tmp_path / "runtime.json"
    refusal_path = tmp_path / "refusal.json"
    cpu_path = tmp_path / "cpu.json"
    parity_path = tmp_path / "template-parity.json"
    result_a = tmp_path / "prompt-a.json"
    result_b = tmp_path / "prompt-b.json"
    shader_path = tmp_path / "paired.metal"
    result_a.write_bytes(b"candidate-a")
    result_b.write_bytes(b"candidate-b")
    shader_path.write_bytes(b"exact-repaired-shader")

    runtime = {"seal_sha256": "runtime-seal"}
    scalar_binding = {
        "complete_manifest_seal_sha256": "manifest-seal",
        "runtime_executable_sha256": "scalar-executable",
    }
    cpu = {
        "seal_sha256": "cpu-seal",
        "observations": {
            "scalar_control_vs_paired_scalar_order_nonfused_difference_observed": False,
        },
    }
    receipt = {
        "schema": parity.SCHEMA,
        "status": "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY",
        "seal_sha256": "template-seal",
        "binding": {
            "attempt_id": parity.SUCCESSOR_ATTEMPT,
            "complete_manifest_seal_sha256": "manifest-seal",
            "scalar_runtime_receipt_path": str(runtime_path),
            "scalar_runtime_receipt_seal_sha256": "runtime-seal",
            "scalar_runtime_executable_sha256": "scalar-executable",
            "predecessor_compile_refusal": {
                "path": str(refusal_path),
                "status": "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY",
                "seal_sha256": "refusal-seal",
            },
            "candidate_source_sha256": {
                "paired_scalar_order_shader": _sha(shader_path.read_bytes()),
            },
        },
        "cpu_scalar_order_gate": {
            "path": str(cpu_path),
            "seal_sha256": "cpu-seal",
            "outcome": "PRECISION_CONTRACTION_DIFFERENCE_OBSERVED_PAIRED_SCALAR_ORDER_CPU_EXACT",
        },
        "candidate_results": {
            "prompt_a_sha256": _sha(result_a.read_bytes()),
            "prompt_b_sha256": _sha(result_b.read_bytes()),
        },
        "all_layer_device_parity_and_exact_completion_parity": {
            "prompt_a_native_device_parity": _device_facts(4),
            "prompt_b_native_device_parity": _device_facts(5),
            "prompt_a_exact_token_parity": _completion(),
            "prompt_b_exact_token_parity": _completion(),
        },
        "failures": [],
    }

    monkeypatch.setattr(parity, "CANONICAL_RUNTIME", runtime_path)
    monkeypatch.setattr(parity, "INITIAL_COMPILE_REFUSAL", refusal_path)
    monkeypatch.setattr(parity, "PRODUCTION_SUCCESSOR_CPU_PARITY", cpu_path)
    monkeypatch.setattr(parity, "PRODUCTION_SUCCESSOR_RESULT_A", result_a)
    monkeypatch.setattr(parity, "PRODUCTION_SUCCESSOR_RESULT_B", result_b)
    monkeypatch.setattr(parity, "PRODUCTION_SUCCESSOR_TEMPLATE_PARITY", parity_path)
    monkeypatch.setattr(parity, "_current_binding", lambda: (runtime, scalar_binding))
    monkeypatch.setattr(
        parity,
        "_sealed",
        lambda path: cpu if path == cpu_path else receipt,
    )

    original_sha = parity._sha256_file

    def fake_sha(path: Path) -> str:
        if path == shader_path:
            return _sha(shader_path.read_bytes())
        return original_sha(path)

    monkeypatch.setattr(parity, "_sha256_file", fake_sha)
    monkeypatch.setattr(
        parity,
        "REPO_ROOT",
        tmp_path,
    )
    (tmp_path / "crates/hawking-core/shaders").mkdir(parents=True)
    target_shader = tmp_path / "crates/hawking-core/shaders/qwen_direct_packed_gate_up_swiglu_paired_scalar_order.metal"
    target_shader.write_bytes(shader_path.read_bytes())

    handoff = parity.production_no_parity_requalification_binding()

    assert handoff["production_gate_up_swiglu_kernel"] == "paired-scalar-order-production-no-parity"
    assert handoff["candidate_template_parity_receipt_seal_sha256"] == "template-seal"
    assert handoff["candidate_cpu_parity_receipt_seal_sha256"] == "cpu-seal"
    assert handoff["scalar_control_runtime_receipt_seal_sha256"] == "runtime-seal"
    assert handoff["claim_boundary"] == {
        "preflight_handoff_only_not_a_runtime_transition": True,
        "does_not_select_or_serve_the_no_parity_kernel": True,
        "requires_fresh_preflight_full_token_template_profile_then_hcli": True,
    }

    result_b.write_bytes(b"drifted")
    with pytest.raises(parity.GateError, match="prompt b result drifted"):
        parity.production_no_parity_requalification_binding()
