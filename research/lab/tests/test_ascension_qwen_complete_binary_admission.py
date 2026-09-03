"""Focused fail-closed coverage for complete-binary admission orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

from lab.operators import ascension_qwen_complete_binary_admission as admission
from lab.receipts import seal, verify


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _target(tmp_path: Path) -> admission.AdmissionTarget:
    return admission.AdmissionTarget(
        key="qwen30",
        prefix="TESTQWEN",
        model_id="test-model",
        repository="test/repository",
        revision="pinned-test-revision",
        manifest_schema="test.complete.binary.v1",
        complete_root=tmp_path / "complete-gravity",
        identity_path=tmp_path / "evolution" / "SOURCE_CONTENT_IDENTITY.json",
    )


def _fixture(target: admission.AdmissionTarget, *, terminal: bool = True) -> dict[str, str]:
    source_dir = target.complete_root.parent / "source"
    source_dir.mkdir(parents=True)
    index_path = source_dir / "model.safetensors.index.json"
    index_raw = b'{"weight_map":{"model.layers.0.weight":"model-00001-of-00001.safetensors"}}\n'
    index_path.write_bytes(index_raw)
    shard = source_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"source shard binding")
    source_audit = target.complete_root.parent / "source-audit.json"
    _write(source_audit, seal({"schema": "test.audit.v1", "status": "bound"}))
    historical_audit = "a" * 64
    identity = seal(
        {
            "schema": admission.IDENTITY_SCHEMA,
            "status": "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND",
            "content_identity_sha256": "c" * 64,
            "model": {
                "id": target.model_id,
                "repository": target.repository,
                "revision": target.revision,
                "source_dir": str(source_dir),
            },
            "source_content": {
                "repository": target.repository,
                "revision": target.revision,
                "control_files": [
                    {
                        "path": "model.safetensors.index.json",
                        "sha256": _sha(index_raw),
                        "bytes": len(index_raw),
                    }
                ],
            },
            "weight_body_audit_seal_sha256": historical_audit,
        }
    )
    _write(target.identity_path, identity)
    shard_stat = shard.stat()
    revalidation = seal(
        {
            "schema": admission.REVALIDATION_SCHEMA,
            "status": "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED",
            "source_repository": target.repository,
            "source_revision": target.revision,
            "source_model_dir": str(source_dir),
            "index_path": str(index_path),
            "index_sha256": _sha(index_raw),
            "source_audit_path": str(source_audit),
            "source_audit_document_sha256": _sha(source_audit.read_bytes()),
            "source_audit_seal_sha256": "b" * 64,
            "sealed_shard_count": 1,
            "sealed_shard_hashes_sha256": "d" * 64,
            "weight_map_sha256": "e" * 64,
            "shards": {
                shard.name: {
                    "expected_sha256": _sha(shard.read_bytes()),
                    "observed_sha256": _sha(shard.read_bytes()),
                    "expected_bytes": shard_stat.st_size,
                    "file_identity": {"bytes": shard_stat.st_size},
                }
            },
        }
    )
    _write(target.revalidation_path, revalidation)
    manifest = seal(
        {
            "schema": target.manifest_schema,
            "status": admission.MANIFEST_STATUS,
            "source_body_audit_seal_sha256": revalidation["source_audit_seal_sha256"],
            "source_revalidation_receipt_path": str(target.revalidation_path),
            "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
            "source": {"repository": target.repository, "model_dir": str(source_dir)},
        }
    )
    _write(target.manifest_path, manifest)
    _write(
        target.pack_status_path,
        {
            "schema": target.manifest_schema,
            "phase": admission.PACK_COMPLETE_PHASE if terminal else "PACKING_COMPLETE_BINARY_GRAVITY",
            "manifest_path": str(target.manifest_path),
            "progress": {"planned_tensors": 1, "completed_tensors": 1 if terminal else 0},
        },
    )
    return {
        "manifest_seal": manifest["seal_sha256"],
        "audit_seal": revalidation["source_audit_seal_sha256"],
        "index_path": str(index_path),
    }


def _fake_loader(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return path


def _success_runner(bindings: dict[str, str]):
    def run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        assert command[0].endswith("native-loader")
        assert timeout > 0
        result = {
            "schema": admission.NATIVE_RESULT_SCHEMA,
            "status": admission.NATIVE_RESULT_STATUS,
            "model": "qwen30",
            "manifest_path": command[4],
            "manifest_seal_sha256": bindings["manifest_seal"],
            "source_audit_path": "/unused/source-audit.json",
            "source_audit_seal_sha256": bindings["audit_seal"],
            "source_revision": "pinned-test-revision",
            "source_index_path": bindings["index_path"],
            "tensor_count": 1,
            "source_weight_elements": 128,
            "tensor_payload_bytes": 64,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    return run


def test_no_manifest_publishes_waiting_without_request_or_receipt(tmp_path: Path) -> None:
    target = _target(tmp_path)
    status = admission.run_once(target, native_loader=tmp_path / "missing-loader")
    assert status["phase"] == "WAITING_FOR_COMPLETE_BINARY_MANIFEST"
    assert not target.receipt_path.exists()
    assert not target.requests_root.exists()
    assert verify(json.loads(target.status_path.read_text(encoding="utf-8")))["phase"] == status["phase"]


def test_incomplete_pack_waits_even_when_a_manifest_file_exists(tmp_path: Path) -> None:
    target = _target(tmp_path)
    _fixture(target, terminal=False)
    status = admission.run_once(target, native_loader=tmp_path / "missing-loader")
    assert status["phase"] == "WAITING_FOR_COMPLETE_PACK"
    assert not target.receipt_path.exists()
    assert not target.requests_root.exists()


def test_success_requires_sealed_request_and_native_result_then_reuses_receipt(tmp_path: Path) -> None:
    target = _target(tmp_path)
    bindings = _fixture(target)
    loader = _fake_loader(tmp_path / "native-loader")
    status = admission.run_once(
        target,
        native_loader=loader,
        runner=_success_runner(bindings),
    )
    assert status["phase"] == admission.ADMISSION_RECEIPT_STATUS
    assert target.receipt_path.parent == target.complete_root
    assert target.legacy_receipt_path.parent == target.complete_root / "complete-admission"
    receipt = verify(json.loads(target.receipt_path.read_text(encoding="utf-8")))
    request_path = Path(receipt["admission_request_path"])
    request = verify(json.loads(request_path.read_text(encoding="utf-8")))
    assert request["schema"] == admission.REQUEST_SCHEMA
    assert request["complete_manifest"]["seal_sha256"] == bindings["manifest_seal"]
    assert request["current_source_revalidation"]["source_audit_seal_sha256"] == bindings["audit_seal"]
    assert request["immutable_source_identity"]["content_identity_sha256"] == "c" * 64
    assert receipt["native_loader"]["tensor_count"] == 1
    assert receipt["claim_boundary"]["admission_does_not_claim_capability_hcli_tps_tg_or_tournament_qualification"]
    pointer = verify(
        json.loads(target.current_receipt_pointer_path.read_text(encoding="utf-8"))
    )
    assert pointer["schema"] == admission.CURRENT_RECEIPT_POINTER_SCHEMA
    assert pointer["status"] == admission.CURRENT_RECEIPT_POINTER_STATUS
    assert pointer["complete_manifest"]["seal_sha256"] == bindings["manifest_seal"]
    assert pointer["admission_receipt"]["path"] == str(
        admission._versioned_receipt_path(target, request).resolve()
    )

    def must_not_run(_: list[str], __: float) -> subprocess.CompletedProcess[str]:
        raise AssertionError("an exact immutable receipt should be reused without rerunning native admission")

    reused = admission.run_once(target, native_loader=loader, runner=must_not_run)
    assert reused["phase"] == "EARNED_COMPLETE_BINARY_ADMISSION_RECEIPT_REUSED"


def test_stale_public_receipt_is_preserved_and_new_terminal_manifest_gets_versioned_current_receipt(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    bindings = _fixture(target)
    loader = _fake_loader(tmp_path / "native-loader")
    admitted = admission.run_once(
        target,
        native_loader=loader,
        runner=_success_runner(bindings),
    )
    assert admitted["phase"] == admission.ADMISSION_RECEIPT_STATUS
    historical_public_bytes = target.receipt_path.read_bytes()
    historical_receipt = verify(json.loads(historical_public_bytes))

    # A sealed manifest that has advanced after the old native pass is a new
    # immutable admission request.  The old public receipt remains byte-for-
    # byte intact; only the current pointer may move to a newly scanned,
    # versioned receipt.
    next_manifest = json.loads(target.manifest_path.read_text(encoding="utf-8"))
    next_manifest["terminal_manifest_generation"] = 2
    _write(target.manifest_path, seal(next_manifest))
    next_manifest = verify(json.loads(target.manifest_path.read_text(encoding="utf-8")))
    next_bindings = dict(bindings)
    next_bindings["manifest_seal"] = next_manifest["seal_sha256"]
    advanced = admission.run_once(
        target,
        native_loader=loader,
        runner=_success_runner(next_bindings),
    )
    assert advanced["phase"] == admission.ADMISSION_RECEIPT_STATUS
    assert target.receipt_path.read_bytes() == historical_public_bytes
    assert verify(json.loads(target.receipt_path.read_text(encoding="utf-8"))) == historical_receipt

    advanced_request = admission._build_request(target)
    advanced_receipt_path = admission._versioned_receipt_path(target, advanced_request)
    advanced_receipt = verify(json.loads(advanced_receipt_path.read_text(encoding="utf-8")))
    assert advanced_receipt["complete_manifest"]["seal_sha256"] == next_bindings["manifest_seal"]
    assert advanced_receipt["seal_sha256"] != historical_receipt["seal_sha256"]
    pointer = verify(
        json.loads(target.current_receipt_pointer_path.read_text(encoding="utf-8"))
    )
    assert pointer["complete_manifest"]["seal_sha256"] == next_bindings["manifest_seal"]
    assert pointer["admission_receipt"]["path"] == str(advanced_receipt_path.resolve())
    assert pointer["admission_receipt"]["seal_sha256"] == advanced_receipt["seal_sha256"]

    def must_not_run(_: list[str], __: float) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the exact new versioned receipt should be reused")

    reused = admission.run_once(target, native_loader=loader, runner=must_not_run)
    assert reused["phase"] == "EARNED_COMPLETE_BINARY_ADMISSION_RECEIPT_REUSED"
    assert reused["admission_receipt_path"] == str(advanced_receipt_path)


def test_mismatched_native_manifest_result_fails_closed_without_receipt(tmp_path: Path) -> None:
    target = _target(tmp_path)
    bindings = _fixture(target)
    loader = _fake_loader(tmp_path / "native-loader")

    def bad_runner(command: list[str], _: float) -> subprocess.CompletedProcess[str]:
        result = {
            "schema": admission.NATIVE_RESULT_SCHEMA,
            "status": admission.NATIVE_RESULT_STATUS,
            "model": "qwen30",
            "manifest_path": command[4],
            "manifest_seal_sha256": "f" * 64,
            "source_audit_seal_sha256": bindings["audit_seal"],
            "source_revision": target.revision,
            "source_index_path": bindings["index_path"],
            "tensor_count": 1,
            "source_weight_elements": 128,
            "tensor_payload_bytes": 64,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    status = admission.run_once(target, native_loader=loader, runner=bad_runner)
    assert status["phase"] == "BLOCKED_COMPLETE_BINARY_ADMISSION_FAIL_CLOSED"
    assert "manifest seal differs" in status["detail"]
    assert not target.receipt_path.exists()
