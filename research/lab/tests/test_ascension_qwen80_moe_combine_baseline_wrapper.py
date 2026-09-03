"""Focused contract tests for the sealed Qwen80 MoE-combine CPU baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_moe_combine_baseline_wrapper as wrapper
from lab.receipts import seal, verify


def _write(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _patch_current_identity(monkeypatch: pytest.MonkeyPatch, manifest: Path) -> None:
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    monkeypatch.setattr(wrapper, "MANIFEST_SHA256", _sha256(manifest))
    monkeypatch.setattr(wrapper, "MANIFEST_SEAL", manifest_doc["seal_sha256"])
    monkeypatch.setattr(wrapper, "ADMISSION_RECEIPT_SEAL", "a" * 64)
    monkeypatch.setattr(wrapper, "SOURCE_BODY_AUDIT_SEAL", "b" * 64)
    monkeypatch.setattr(wrapper, "SOURCE_REVALIDATION_SEAL", "c" * 64)


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        seal(
            {
                "schema": wrapper.MANIFEST_SCHEMA,
                "status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
                "source": {"repository": wrapper.SOURCE_REPOSITORY, "tensor_count": 74_391},
                "source_body_audit_seal_sha256": "b" * 64,
                "source_revalidation_receipt_seal_sha256": "c" * 64,
            }
        ),
    )
    _patch_current_identity(monkeypatch, manifest)
    manifest_evidence = _evidence(manifest)
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))

    admission = tmp_path / "admission-current.json"
    _write(
        admission,
        seal(
            {
                "schema": wrapper.ADMISSION_SCHEMA,
                "status": "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED",
                "model": {
                    "id": wrapper.MODEL_ID,
                    "key": wrapper.MODEL_KEY,
                    "repository": wrapper.SOURCE_REPOSITORY,
                    "revision": wrapper.SOURCE_REVISION,
                },
                "complete_manifest": {
                    "path": manifest_evidence["path"],
                    "document_sha256": manifest_evidence["sha256"],
                    "seal_sha256": manifest_doc["seal_sha256"],
                },
                "admission_receipt": {"seal_sha256": "a" * 64},
            }
        ),
    )
    admission_evidence = _evidence(admission)

    router = tmp_path / "router-inner.json"
    _write(
        router,
        {
            "schema": wrapper.ROUTER_INNER_SCHEMA,
            "status": wrapper.ROUTER_INNER_STATUS,
            "mode": "metal",
            "component_only": True,
            "metal_device_or_dispatch_performed": True,
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_doc["seal_sha256"],
                "admission_current_path": admission_evidence["path"],
                "admission_receipt_seal_sha256": "a" * 64,
                "layer": 0,
                "hidden": 2_048,
                "experts_per_token": 10,
            },
            "source_stable_top10_router": {
                "ids": list(wrapper.SOURCE_TOP10_IDS),
                "device_ids": list(wrapper.SOURCE_TOP10_IDS),
                "device_ids_exact_match": True,
                "ids_unique_and_in_range": True,
                "renormalized_weights": [0.1] * 10,
            },
        },
    )
    router_evidence = _evidence(router)

    router_outer = tmp_path / "router-outer.json"
    _write(
        router_outer,
        seal(
            {
                "schema": wrapper.ROUTER_OUTER_SCHEMA,
                "status": wrapper.ROUTER_OUTER_STATUS,
                "source_binding": {
                    "manifest": manifest_evidence,
                    "admission_current": admission_evidence,
                },
                "inner_probe_capture": {
                    "path": router_evidence["path"],
                    "sha256": router_evidence["sha256"],
                    "schema": wrapper.ROUTER_INNER_SCHEMA,
                    "status": wrapper.ROUTER_INNER_STATUS,
                    "mode": "metal",
                    "metal_performed": True,
                },
            }
        ),
    )
    router_outer_evidence = _evidence(router_outer)
    router_outer_doc = json.loads(router_outer.read_text(encoding="utf-8"))

    capture = tmp_path / "cpu-capture"
    capture.mkdir()
    for name in ("invocation.json", "stdout.jsonl", "stderr.log"):
        (capture / name).write_text("{}\n", encoding="utf-8")
    cpu_inner = capture / "receipt.json"
    _write(
        cpu_inner,
        {
            "schema": wrapper.CPU_INNER_SCHEMA,
            "status": wrapper.CPU_INNER_STATUS,
            "mode": "cpu-oracle",
            "metal_device_or_dispatch_performed": False,
            "component_only": True,
            "routed_expert_aggregation_performed": True,
            "shared_expert_add_performed": True,
            "second_residual_performed": True,
            "complete_layer_or_token_performed": False,
            "materialized_source_route_shaped_fixture_only": True,
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_doc["seal_sha256"],
                "admission_current_path": admission_evidence["path"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
            "source_top10_binding": {
                "router_receipt_path": router_evidence["path"],
                "router_receipt_sha256": router_evidence["sha256"],
                "router_outer_receipt_path": router_outer_evidence["path"],
                "router_outer_receipt_sha256": router_outer_evidence["sha256"],
                "router_outer_receipt_seal_sha256": router_outer_doc["seal_sha256"],
                "ids": list(wrapper.SOURCE_TOP10_IDS),
            },
            "durable_capture": {
                "directory": str(capture.resolve()),
                "invocation_file": "invocation.json",
                "stdout_file": "stdout.jsonl",
                "stderr_file": "stderr.log",
                "receipt_file": "receipt.json",
                "receipt_written_last_is_completion_marker": True,
            },
            "cpu_oracle": {
                "routed_sum": {"elements": 2_048, "sha256": "d" * 64},
                "second_residual": {"elements": 2_048, "sha256": "e" * 64},
                "routed_sum_f32_vs_f64_max_abs": 0.0,
                "second_residual_f32_vs_f64_max_abs": 0.0,
            },
        },
    )
    return {
        "manifest": manifest,
        "admission": admission,
        "router": router,
        "router_outer": router_outer,
        "cpu_inner": cpu_inner,
    }


def test_seals_current_cpu_combine_baseline_without_promoting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    body = wrapper.build_wrapper(
        cpu_inner_receipt=inputs["cpu_inner"],
        manifest_path=inputs["manifest"],
        admission_current=inputs["admission"],
        router_receipt=inputs["router"],
        router_outer_receipt=inputs["router_outer"],
    )
    assert body["schema"] == wrapper.WRAPPER_SCHEMA
    assert body["status"] == wrapper.WRAPPER_STATUS
    assert body["source_binding"]["source_top10_binding"]["ids"] == list(wrapper.SOURCE_TOP10_IDS)
    assert body["claim_boundary"]["does_not_perform_metal_device_execution_or_issue_a_lease"] is True
    output = tmp_path / "baseline-wrapper.json"
    observed = wrapper._write_new_sealed(output, body)
    verify(observed)
    assert observed["cpu_inner_receipt"] == _evidence(inputs["cpu_inner"])


def test_rejects_reordered_router_identity_before_writing_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    inner = json.loads(inputs["cpu_inner"].read_text(encoding="utf-8"))
    inner["source_top10_binding"]["ids"][0], inner["source_top10_binding"]["ids"][1] = (
        inner["source_top10_binding"]["ids"][1],
        inner["source_top10_binding"]["ids"][0],
    )
    _write(inputs["cpu_inner"], inner)
    with pytest.raises(ValueError, match="source top-10"):
        wrapper.build_wrapper(
            cpu_inner_receipt=inputs["cpu_inner"],
            manifest_path=inputs["manifest"],
            admission_current=inputs["admission"],
            router_receipt=inputs["router"],
            router_outer_receipt=inputs["router_outer"],
        )


def test_refuses_to_overwrite_sealed_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    body = wrapper.build_wrapper(
        cpu_inner_receipt=inputs["cpu_inner"],
        manifest_path=inputs["manifest"],
        admission_current=inputs["admission"],
        router_receipt=inputs["router"],
        router_outer_receipt=inputs["router_outer"],
    )
    output = tmp_path / "baseline-wrapper.json"
    wrapper._write_new_sealed(output, body)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        wrapper._write_new_sealed(output, body)
