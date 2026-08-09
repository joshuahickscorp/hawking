"""Regression coverage for Qwen80 hybrid-runtime dependency accounting."""
from __future__ import annotations

import json
import hashlib

from lab.receipts import seal
from lab.operators import ascension_qwen80_bootstrap_lanes as lanes
from lab.operators.ascension_qwen80_bootstrap_lanes import _source_summary


def test_qwen80_source_summary_keeps_hybrid_and_moe_obligations_separate() -> None:
    summary = _source_summary(
        {
            "model.layers.0.linear_attn.in_proj_qkvz.weight": "unit.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "unit.safetensors",
            "model.layers.0.mlp.gate.weight": "unit.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "unit.safetensors",
            "model.layers.0.mlp.shared_expert.gate_proj.weight": "unit.safetensors",
        }
    )

    assert summary["layer_count"] == 1
    assert summary["gated_deltanet_tensor_count"] == 1
    assert summary["gated_attention_tensor_count"] == 1
    assert summary["router_tensor_count"] == 1
    assert summary["routed_expert_tensor_count"] == 1
    assert summary["shared_expert_tensor_count"] == 1


def test_qwen80_runtime_handoff_requires_current_manifest_selected_admission_receipt(
    tmp_path, monkeypatch
) -> None:
    complete_root = tmp_path / "complete-gravity"
    manifest = complete_root / "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    historical_receipt = complete_root / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
    pointer = complete_root / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json"
    monkeypatch.setattr(lanes, "COMPLETE_ROOT", complete_root)
    monkeypatch.setattr(lanes, "COMPLETE_MANIFEST", manifest)
    monkeypatch.setattr(lanes, "ADMISSION_RECEIPT", historical_receipt)
    monkeypatch.setattr(lanes, "ADMISSION_CURRENT_POINTER", pointer)
    candidate = {
        "manifest_seal_sha256": "a" * 64,
        "all_tensor_artifact_complete": True,
    }
    binding, state = lanes._admission_binding(candidate)
    assert binding is None
    assert state["status"] == "WAITING_FOR_CURRENT_TERMINAL_NATIVE_ADMISSION_POINTER"

    # A valid old fixed-path receipt is historical evidence only.  It must not
    # be treated as admission for the terminal manifest without a selector.
    historical_receipt.parent.mkdir(parents=True, exist_ok=True)
    historical_receipt.write_text(
        json.dumps(
            seal(
                {
                    "schema": lanes.ADMISSION_SCHEMA,
                    "status": lanes.ADMISSION_STATUS,
                    "model": {
                        "key": "qwen80",
                        "id": lanes.MODEL_ID,
                        "repository": "Qwen/Qwen3-Coder-Next",
                        "revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
                    },
                    "complete_manifest": {
                        "path": str(manifest),
                        "document_sha256": "c" * 64,
                        "seal_sha256": "d" * 64,
                    },
                    "current_source_revalidation": {
                        "revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
                        "source_audit_seal_sha256": "b" * 64,
                    },
                }
            )
        )
    )
    binding, state = lanes._admission_binding(candidate)
    assert binding is None
    assert state["status"] == "WAITING_FOR_CURRENT_TERMINAL_NATIVE_ADMISSION_POINTER"

    request_path = (
        complete_root
        / "complete-admission"
        / "requests"
        / f"QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_REQUEST_{'a' * 64}.json"
    )
    request_seal = "e" * 64
    receipt_path = (
        complete_root
        / "complete-admission"
        / "receipts"
        / f"QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT_{'a' * 64}.json"
    )
    current_receipt = seal(
        {
            "schema": lanes.ADMISSION_SCHEMA,
            "status": lanes.ADMISSION_STATUS,
            "model": {
                "key": "qwen80",
                "id": lanes.MODEL_ID,
                "repository": "Qwen/Qwen3-Coder-Next",
                "revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
            },
            "admission_request_path": str(request_path),
            "admission_request_seal_sha256": request_seal,
            "complete_manifest": {
                "path": str(manifest),
                "document_sha256": "c" * 64,
                "seal_sha256": "a" * 64,
            },
            "current_source_revalidation": {
                "revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
                "source_audit_seal_sha256": "b" * 64,
            },
        }
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(current_receipt), encoding="utf-8")
    receipt_document_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    pointer.write_text(
        json.dumps(
            seal(
                {
                    "schema": lanes.ADMISSION_CURRENT_POINTER_SCHEMA,
                    "status": lanes.ADMISSION_CURRENT_POINTER_STATUS,
                    "pointer_version": 1,
                    "model": current_receipt["model"],
                    "complete_manifest": current_receipt["complete_manifest"],
                    "admission_request_path": str(request_path),
                    "admission_request_seal_sha256": request_seal,
                    "admission_receipt": {
                        "path": str(receipt_path),
                        "document_sha256": receipt_document_sha,
                        "seal_sha256": current_receipt["seal_sha256"],
                        "selection_source": "VERSIONED_CURRENT_MANIFEST",
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    binding, state = lanes._admission_binding(candidate)
    assert state["status"] == "CURRENT_TERMINAL_ADMISSION_RECEIPT_BOUND"
    assert binding is not None
    assert binding["manifest_seal_sha256"] == "a" * 64
    assert binding["source_audit_seal_sha256"] == "b" * 64
    assert binding["receipt_path"] == str(receipt_path.resolve())


def test_direct_packed_stage_cache_uses_durable_underscore_result_key(
    tmp_path, monkeypatch
) -> None:
    """A passed direct stage must not be re-run because its CLI mode has hyphens."""

    binary = tmp_path / "qwen80-preflight"
    binary.write_bytes(b"sealed-native-preflight")
    monkeypatch.setattr(lanes, "RUNTIME_PREFLIGHT_BINARY", binary)
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    binding = {
        "receipt_seal_sha256": "r" * 64,
        "manifest_seal_sha256": "m" * 64,
    }
    prior = {
        "status": lanes.PREFLIGHT_DIRECT_PACKED_LINEAR_STAGE_STATUS,
        "mode": "direct-packed-linear-stage",
        "binary_sha256": binary_sha,
        "result": {"status": lanes.PREFLIGHT_DIRECT_PACKED_LINEAR_STAGE_STATUS},
    }
    existing = {
        "admission_receipt_seal_sha256": binding["receipt_seal_sha256"],
        "manifest_seal_sha256": binding["manifest_seal_sha256"],
        "direct_packed_linear_stage": prior,
    }

    def must_not_run(**_kwargs):
        raise AssertionError("a current passed direct stage must be reused")

    monkeypatch.setattr(lanes, "_run_native_runtime_preflight", must_not_run)
    reused = lanes._cached_or_run_native_runtime_preflight(
        mode="direct-packed-linear-stage",
        binding=binding,
        existing=existing,
    )
    assert reused == prior


def test_qwen80_watcher_holds_state_and_direct_preflights_for_current_qwen30_lease(
    tmp_path, monkeypatch
) -> None:
    """The latest matching HELD record pauses; a later release resumes."""

    runtime_root = tmp_path / "complete-runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(lanes, "RUNTIME_ROOT", runtime_root)
    binding = {
        "manifest_path": "/sealed/qwen80.json",
        "manifest_seal_sha256": "m" * 64,
        "source_audit_seal_sha256": "a" * 64,
        "receipt_seal_sha256": "r" * 64,
        "source_revision": "revision",
    }
    common = {
        "schema": lanes.GPU_COORDINATION_HOLD_SCHEMA,
        "source_binding": {
            "manifest_path": binding["manifest_path"],
            "manifest_seal_sha256": binding["manifest_seal_sha256"],
            "source_body_audit_seal_sha256": binding["source_audit_seal_sha256"],
            "admission_receipt_seal_sha256": binding["receipt_seal_sha256"],
            "source_revision": binding["source_revision"],
        },
        "coordination": {"qwen30_activity": "Qwen30 source-bound requalification owns lease"},
    }
    held = {
        **common,
        "recorded_at": "2026-08-08T22:00:00Z",
        "status": "HELD_QWEN80_TEST_BEFORE_UNGUARDED_METAL",
    }
    held_path = runtime_root / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220000Z.json"
    held_path.write_text(json.dumps(held), encoding="utf-8")
    active = lanes._active_qwen30_gpu_coordination_hold(binding)
    assert active is not None
    assert active["hold_status"] == held["status"]
    assert active["record_path"] == str(held_path)

    released = {
        **common,
        "recorded_at": "2026-08-08T22:01:00Z",
        "status": lanes.GPU_COORDINATION_RELEASED_STATUS,
    }
    (runtime_root / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220100Z.json").write_text(
        json.dumps(released), encoding="utf-8"
    )
    assert lanes._active_qwen30_gpu_coordination_hold(binding) is None


def test_qwen80_runtime_cycle_never_launches_state_or_direct_while_qwen30_hold_is_active(
    tmp_path, monkeypatch
) -> None:
    """A hold is a control-flow gate, not merely a status annotation."""

    runtime_root = tmp_path / "complete-runtime"
    runtime_root.mkdir()
    preflight = runtime_root / "QWEN80_COMPLETE_NATIVE_RUNTIME_PREFLIGHT.json"
    monkeypatch.setattr(lanes, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(lanes, "RUNTIME_PREFLIGHT_RESULT", preflight)
    runtime_source = tmp_path / "qwen80_complete_runtime.rs"
    runtime_source.write_text("// fixture\n", encoding="utf-8")
    monkeypatch.setattr(lanes, "QWEN80_NATIVE_RUNTIME", runtime_source)
    binding = {
        "manifest_path": "/sealed/qwen80.json",
        "manifest_seal_sha256": "m" * 64,
        "source_audit_seal_sha256": "a" * 64,
        "receipt_seal_sha256": "r" * 64,
        "source_revision": "revision",
    }
    (runtime_root / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220000Z.json").write_text(
        json.dumps(
            {
                "schema": lanes.GPU_COORDINATION_HOLD_SCHEMA,
                "recorded_at": "2026-08-08T22:00:00Z",
                "status": "HELD_QWEN80_TEST_BEFORE_UNGUARDED_METAL",
                "coordination": {"qwen30_activity": "Qwen30 owns the native requalification lease"},
                "source_binding": {
                    "manifest_path": binding["manifest_path"],
                    "manifest_seal_sha256": binding["manifest_seal_sha256"],
                    "source_body_audit_seal_sha256": binding["source_audit_seal_sha256"],
                    "admission_receipt_seal_sha256": binding["receipt_seal_sha256"],
                    "source_revision": binding["source_revision"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lanes, "_verified_source_audit", lambda: {"seal_sha256": "audit"})
    monkeypatch.setattr(lanes, "_weight_map", lambda: {})
    monkeypatch.setattr(lanes, "_source_summary", lambda _weights: {"fixture": True})
    monkeypatch.setattr(
        lanes,
        "_candidate_progress",
        lambda: {"all_tensor_artifact_complete": True, "manifest_seal_sha256": binding["manifest_seal_sha256"]},
    )
    monkeypatch.setattr(lanes, "_deltanet_component", lambda: {"status": "fixture"})
    monkeypatch.setattr(
        lanes,
        "_admission_binding",
        lambda _candidate: (binding, {"status": "CURRENT_TERMINAL_ADMISSION_RECEIPT_BOUND"}),
    )
    calls: list[str] = []

    def cached_only_catalog(*, mode, **_kwargs):
        calls.append(mode)
        assert mode == "catalog", "active Qwen30 hold must stop before state/direct launch"
        return {"status": lanes.PREFLIGHT_CATALOG_STATUS, "mode": mode}

    monkeypatch.setattr(lanes, "_cached_or_run_native_runtime_preflight", cached_only_catalog)
    lanes.run_runtime_cycle()

    assert calls == ["catalog"]
    status = json.loads((runtime_root / "QWEN80_COMPLETE_RUNTIME_STATUS.json").read_text())
    assert status["phase"] == "WAITING_FOR_COORDINATED_GPU_LEASE"
    handoff = json.loads(preflight.read_text())
    assert handoff["gpu_coordination_hold"]["status"] == "ACTIVE_QWEN30_GPU_COORDINATION_HOLD"
