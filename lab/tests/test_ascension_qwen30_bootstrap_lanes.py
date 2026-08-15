"""Focused regressions for the Qwen30 detached runtime watch state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.operators import ascension_qwen30_bootstrap_lanes as lanes
from lab.receipts import seal


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_paired_scalar_order_cpu_gate_requires_live_scalar_runtime_binding(
    tmp_path: Path, monkeypatch
) -> None:
    """A historical CPU diagnostic must never steer the detached watch state."""

    executable = tmp_path / "qwen30-scalar-runtime"
    executable.write_bytes(b"current-scalar-runtime")
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    runtime_path = tmp_path / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
    cpu_gate_path = tmp_path / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_CPU_PARITY_RECEIPT.json"
    binding = {
        "manifest_seal_sha256": "m" * 64,
        "source_audit_seal_sha256": "a" * 64,
        "source_revision": "r" * 40,
    }
    runtime = seal(
        {
            "schema": lanes.PHYSICAL_RUNTIME_SCHEMA,
            "status": lanes.PHYSICAL_RUNTIME_STATUS,
            "binding": {
                "model_id": "Qwen3-Coder-30B-A3B-Instruct",
                "runtime_executable_sha256": executable_sha256,
            },
        }
    )
    _write(runtime_path, runtime)
    cpu_gate = seal(
        {
            "schema": "hawking.ascension.qwen30_direct_packed_gate_up_precision_order_discriminator.v1",
            "status": "EARNED_CPU_DIRECT_PACKED_GATE_UP_ORDER_PRECISION_DISCRIMINATOR",
            "outcome": "PRECISION_CONTRACTION_DIFFERENCE_OBSERVED_PAIRED_SCALAR_ORDER_CPU_EXACT",
            "binding": {
                "manifest_seal_sha256": binding["manifest_seal_sha256"],
                "source_audit_seal_sha256": binding["source_audit_seal_sha256"],
                "source_revision": binding["source_revision"],
                "runtime": {
                    "path": str(runtime_path),
                    "schema": lanes.PHYSICAL_RUNTIME_SCHEMA,
                    "status": lanes.PHYSICAL_RUNTIME_STATUS,
                    "seal_sha256": runtime["seal_sha256"],
                    "runtime_executable_sha256": executable_sha256,
                },
            },
            "observations": {
                "scalar_control_vs_paired_scalar_order_nonfused_difference_observed": False,
            },
        }
    )
    _write(cpu_gate_path, cpu_gate)
    monkeypatch.setattr(lanes, "QWEN30_NATIVE_RUNTIME_EXECUTABLE", executable)
    monkeypatch.setattr(lanes, "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT", runtime_path)
    monkeypatch.setattr(lanes, "QWEN30_GATEUP_PAIRED_SCALAR_ORDER_CPU_PARITY", cpu_gate_path)

    observed = lanes._current_paired_scalar_order_cpu_gate(binding)
    assert observed is not None
    assert observed["seal_sha256"] == cpu_gate["seal_sha256"]

    stale = dict(cpu_gate)
    stale["observations"] = {
        "scalar_control_vs_paired_scalar_order_nonfused_difference_observed": True,
    }
    _write(cpu_gate_path, seal(stale))
    assert lanes._current_paired_scalar_order_cpu_gate(binding) is None


def test_compile_refusal_requires_no_candidate_device_or_template_result(
    tmp_path: Path, monkeypatch
) -> None:
    """Only the exact f594-style syntax refusal may hold HCLI closed."""

    executable = tmp_path / "qwen30-scalar-runtime"
    executable.write_bytes(b"current-scalar-runtime")
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    runtime_path = tmp_path / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
    refusal_path = tmp_path / "QWEN30_PAIRED_SCALAR_ORDER_REFUSAL.json"
    runtime = seal(
        {
            "schema": lanes.PHYSICAL_RUNTIME_SCHEMA,
            "status": lanes.PHYSICAL_RUNTIME_STATUS,
            "binding": {
                "model_id": "Qwen3-Coder-30B-A3B-Instruct",
                "runtime_executable_sha256": executable_sha256,
            },
        }
    )
    _write(runtime_path, runtime)
    binding = {
        "manifest_seal_sha256": "m" * 64,
        "source_audit_seal_sha256": "a" * 64,
        "source_revision": "r" * 40,
    }
    refusal = seal(
        {
            "schema": "hawking.ascension.qwen30_paired_scalar_order_gate_up_template_parity.v1",
            "status": "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY",
            "binding": {
                "complete_manifest_seal_sha256": binding["manifest_seal_sha256"],
                "scalar_runtime_receipt_path": str(runtime_path),
                "scalar_runtime_receipt_seal_sha256": runtime["seal_sha256"],
                "scalar_runtime_executable_sha256": executable_sha256,
            },
            "candidate_results": {"prompt_a_sha256": None, "prompt_b_sha256": None},
            "all_layer_device_parity_and_exact_completion_parity": {},
            "failures": ["candidate prompt A returned 2"],
        }
    )
    _write(refusal_path, refusal)
    monkeypatch.setattr(lanes, "QWEN30_NATIVE_RUNTIME_EXECUTABLE", executable)
    monkeypatch.setattr(lanes, "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT", runtime_path)
    monkeypatch.setattr(lanes, "QWEN30_GATEUP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY", refusal_path)

    observed = lanes._paired_scalar_order_compile_refusal(binding)
    assert observed is not None
    assert observed["seal_sha256"] == refusal["seal_sha256"]

    numerical_result = dict(refusal)
    numerical_result["candidate_results"] = {
        "prompt_a_sha256": "b" * 64,
        "prompt_b_sha256": None,
    }
    _write(refusal_path, seal(numerical_result))
    assert lanes._paired_scalar_order_compile_refusal(binding) is None


def test_production_http_command_has_explicit_no_parity_kernel_and_exact_binding(
    tmp_path: Path, monkeypatch
) -> None:
    """A server command may not silently fall back to the scalar default."""

    server = tmp_path / "ascension_qwen30_native_http_server"
    server.write_bytes(b"production-server")
    monkeypatch.setattr(lanes, "QWEN30_NATIVE_HTTP_SERVER", server)
    monkeypatch.setattr(
        lanes,
        "_effective_qwen30_gate_up_swiglu_cli",
        lambda: lanes.QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI,
    )
    binding = {
        "manifest_path": "/protected/QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
        "manifest_seal_sha256": "m" * 64,
        "source_audit_seal_sha256": "a" * 64,
        "source_revision": "r" * 40,
    }

    contract = lanes._native_http_adapter_kernel_contract()
    command = lanes._native_http_adapter_command(binding)

    assert contract["production_no_parity"] is True
    assert contract["kernel_id"] == "qwen30_paired_scalar_order_no_parity_v1"
    assert command[-2:] == [
        "--gate-up-swiglu-kernel",
        "paired-scalar-order-production-no-parity",
    ]
    exact_record = {"binding": dict(binding)}
    assert lanes._native_http_adapter_binding_matches(exact_record, binding)
    stale = {"binding": {**binding, "runtime_executable_sha256": "b" * 64}}
    assert not lanes._native_http_adapter_binding_matches(stale, binding)


def test_production_http_runtime_binding_carries_admission_and_canonical_authority(
    tmp_path: Path, monkeypatch
) -> None:
    """TPS must see the admitted artifact and the canonical runtime together."""

    executable = tmp_path / "qwen30-production-runtime"
    executable.write_bytes(b"production-runtime")
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    runtime = seal(
        {
            "schema": lanes.PHYSICAL_RUNTIME_SCHEMA,
            "status": lanes.PHYSICAL_RUNTIME_STATUS,
            "binding": {
                "complete_manifest_seal_sha256": "m" * 64,
                "runtime_executable_sha256": executable_sha256,
            },
            "runtime": {
                "gate_up_swiglu_kernel": lanes.QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL,
                "custom_kernel_used": True,
            },
        }
    )
    binding = {
        "admission_receipt_seal_sha256": "d" * 64,
        "manifest_seal_sha256": "m" * 64,
        "source_audit_seal_sha256": "a" * 64,
        "source_revision": "r" * 40,
    }
    deployment = {
        "binding": {
            "replacement_runtime_executable_sha256": executable_sha256,
            "production_gate_up_swiglu_kernel": lanes.QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI,
            "production_kernel_receipt_id": lanes.QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL,
        },
        "seal_sha256": "p" * 64,
    }
    monkeypatch.setattr(lanes, "QWEN30_NATIVE_RUNTIME_EXECUTABLE", executable)
    monkeypatch.setattr(lanes, "_paired_scalar_order_production_deployment", lambda: deployment)
    monkeypatch.setattr(
        lanes,
        "_effective_qwen30_gate_up_swiglu_cli",
        lambda: lanes.QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI,
    )

    observed = lanes._native_http_adapter_runtime_binding(binding, runtime)

    assert observed is not None
    assert observed["admission_receipt_seal_sha256"] == binding["admission_receipt_seal_sha256"]
    assert observed["canonical_runtime_receipt_seal_sha256"] == runtime["seal_sha256"]
    assert observed["runtime_executable_sha256"] == executable_sha256
