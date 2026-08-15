"""Focused integrity coverage for the isolated Qwen30 quality-repack lane."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import struct
from pathlib import Path

import numpy as np
import pytest

from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.operators.ascension_qwen30_quality_repack import (
    ARTIFACT_PREFIX,
    BASELINE_ADMISSION_NAME,
    BASELINE_MANIFEST_NAME,
    BASELINE_REVALIDATION_NAME,
    QualityRepackGravity,
    _pack_sparse_residual,
    _source_value_sha256,
    _unpack_sparse_residual,
)
from lab.operators.ascension_qwen30_quality_repack_admission import (
    NATIVE_RESULT_SCHEMA,
    NATIVE_RESULT_STATUS,
    QualityAdmissionTarget,
    run_once as run_quality_admission_once,
)
from lab.operators.ascension_qwen30_quality_repack_scalar_parity import (
    RESULT_SCHEMA as SCALAR_PARITY_RESULT_SCHEMA,
    RESULT_STATUS as SCALAR_PARITY_RESULT_STATUS,
    ScalarParityError,
    ScalarParityTarget,
    _invoke_native as invoke_scalar_parity_native,
)
from lab.receipts import seal, verify


GATE = "model.layers.0.mlp.experts.0.gate_proj.weight"
UP = "model.layers.0.mlp.experts.0.up_proj.weight"
CONTROL = "model.layers.0.self_attn.q_proj.weight"


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_source(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    generator = np.random.default_rng(7)
    gate = generator.normal(0.0, 0.08, size=(128, 128)).astype(np.float32)
    up = generator.normal(0.0, 0.08, size=(128, 128)).astype(np.float32)
    # A few deterministic outliers make the sparse-residual branch a real
    # measurable quality improvement rather than a fixture-only metadata path.
    gate.reshape(-1)[[3, 207, 4095, 9000]] = np.asarray([3.5, -4.0, 5.0, -3.0], dtype=np.float32)
    up.reshape(-1)[[8, 311, 5001, 12000]] = np.asarray([-3.25, 4.5, -5.5, 3.75], dtype=np.float32)
    tensors = {
        GATE: gate,
        UP: up,
        # This non-mutated control tensor amortizes the deliberately complete
        # manifest ledger, so the fixture exercises the real <=1.5 BPW gate
        # rather than a tiny-artifact header/receipt pathology.
        CONTROL: np.linspace(-0.5, 0.5, 512 * 1024, dtype=np.float32).reshape(512, 1024),
    }
    shard = model_dir / "model-00001-of-00001.safetensors"
    header: dict[str, object] = {}
    payload = bytearray()
    offset = 0
    for name, values in tensors.items():
        raw = values.astype("<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payload.extend(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    index = model_dir / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}}, sort_keys=True),
        encoding="utf-8",
    )
    audit = tmp_path / "source-audit.json"
    _write_json(
        audit,
        seal(
            {
                "schema": "test.source.audit.v1",
                "files": [
                    {
                        "path": index.name,
                        "bytes": index.stat().st_size,
                        "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    }
                ],
                "source": {
                    "repository": "test/repository",
                    "revision": "test-revision",
                    "shards": {
                        shard.name: {
                            "bytes": shard.stat().st_size,
                            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                        }
                    },
                },
            }
        ),
    )
    return model_dir, audit, tensors


def _fixture_worker(tmp_path: Path) -> tuple[QualityRepackGravity, Path, Path, Path]:
    model_dir, audit, tensors = _write_source(tmp_path)
    baseline = tmp_path / "baseline"
    baseline_worker = complete.CompleteBinaryGravity(
        model_dir=model_dir,
        source_audit=audit,
        root=baseline,
        repository="test/repository",
        model_id="test-model",
        artifact_prefix="QWEN30",
    )
    assert baseline_worker.run(max_tensors=16) == 0
    manifest_path = baseline / BASELINE_MANIFEST_NAME
    manifest = verify(json.loads(manifest_path.read_text(encoding="utf-8")), label=str(manifest_path))
    admission_path = baseline / BASELINE_ADMISSION_NAME
    _write_json(
        admission_path,
        seal(
            {
                "schema": "test.admission.v1",
                "status": "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED",
            }
        ),
    )
    admission = verify(json.loads(admission_path.read_text(encoding="utf-8")), label=str(admission_path))
    revalidation_path = baseline / BASELINE_REVALIDATION_NAME
    revalidation = verify(
        json.loads(revalidation_path.read_text(encoding="utf-8")), label=str(revalidation_path)
    )
    shard = next(iter(json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"].values()))
    source_shard_sha = hashlib.sha256((model_dir / shard).read_bytes()).hexdigest()
    quality_path = tmp_path / "quality.json"
    quality = seal(
        {
            "schema": "hawking.ascension.qwen30_direct_packed_gate_up_quality_diagnostic.v1",
            "status": "PASS_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_QUALITY_DIAGNOSTIC_NOT_MODEL_QUALITY",
            "source_binding": {
                "revalidation_receipt_path": str(revalidation_path),
                "revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
                "source_content_identity_sha256": "a" * 64,
                "tensors": [
                    {
                        "tensor_name": name,
                        "source_shard": shard,
                        "source_shard_sha256": source_shard_sha,
                        "source_value_sha256": _source_value_sha256(values),
                        "tensor_shape": list(values.shape),
                    }
                    for name, values in ((GATE, tensors[GATE]), (UP, tensors[UP]))
                ],
            },
        }
    )
    _write_json(quality_path, quality)
    proposal_path = tmp_path / "proposal.json"
    _write_json(
        proposal_path,
        seal(
            {
                "schema": "hawking.ascension.qwen30_gate_up_representation_repack_proposal.v1",
                "status": "PROPOSED_NOT_APPLIED_COMPLETE_ACCOUNTING_AND_CAPABILITY_RETEST_REQUIRED",
                "quality_receipt_path": str(quality_path),
                "quality_receipt_seal_sha256": quality["seal_sha256"],
                "baseline_control": {
                    "manifest_path": str(manifest_path),
                    "manifest_seal_sha256": manifest["seal_sha256"],
                    "admission_path": str(admission_path),
                    "admission_seal_sha256": admission["seal_sha256"],
                    "complete_physical_bpw": manifest["complete_physical_bpw_ledger"][
                        "complete_physical_bpw"
                    ],
                    "preserve_as_rollback_control": True,
                    "replacement_forbidden_until_all_acceptance_gates_pass": True,
                },
                "proposed_candidate_branch": {"initial_organs": [GATE, UP]},
            }
        ),
    )
    root = tmp_path / "quality-candidate"
    worker = QualityRepackGravity(
        model_dir=model_dir,
        source_audit=audit,
        root=root,
        proposal_path=proposal_path,
        quality_receipt_path=quality_path,
        baseline_root=baseline,
        repository="test/repository",
    )
    return worker, root, manifest_path, admission_path


def test_sparse_residual_payload_is_deterministic_and_reconstructs_better_than_binary() -> None:
    values = np.linspace(-3.0, 5.0, 256, dtype=np.float32).reshape(16, 16)
    binary_payload, _, binary = complete._pack_binary(values, values.shape)
    payload, metadata, reconstructed = _pack_sparse_residual(values, values.shape, fraction=0.125)
    repeated, repeated_metadata, repeated_reconstructed = _pack_sparse_residual(
        values, values.shape, fraction=0.125
    )
    unpacked, decoded = _unpack_sparse_residual(payload)

    assert payload == repeated
    assert metadata == repeated_metadata
    assert np.array_equal(reconstructed, repeated_reconstructed)
    assert len(payload) > len(binary_payload)
    assert _source_value_sha256(decoded) == _source_value_sha256(reconstructed)
    assert unpacked["selected_count"] == metadata["residual"]["selected_count"]
    assert np.linalg.norm(values - reconstructed) < np.linalg.norm(values - binary.reshape(values.shape))


def test_quality_candidate_preserves_control_and_seals_full_accounted_branch(tmp_path: Path) -> None:
    worker, root, baseline_manifest, baseline_admission = _fixture_worker(tmp_path)
    baseline_manifest_before = hashlib.sha256(baseline_manifest.read_bytes()).hexdigest()
    baseline_admission_before = hashlib.sha256(baseline_admission.read_bytes()).hexdigest()

    selection = worker.validate()
    assert selection["status"] == "EARNED_SOURCE_BOUND_QUALITY_REPACK_SELECTION_UNQUALIFIED"
    assert selection["selected_representation"]["residual_fraction"] in {0.0025, 0.005, 0.01}
    assert selection["selected_representation"]["pair_relative_l2_improvement_fraction"] >= 0.005

    assert worker.run(max_tensors=16) == 0
    manifest_path = root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    manifest = verify(json.loads(manifest_path.read_text(encoding="utf-8")), label=str(manifest_path))
    ledger = manifest["complete_physical_bpw_ledger"]
    assert ledger["passes_storage_threshold"] is True
    assert ledger["complete_physical_bpw"] <= 1.5
    assert manifest["quality_repack_branch"]["admission_state"] == "NOT_REQUESTED_REQUIRES_EXACT_SPARSE_RESIDUAL_NATIVE_READER"
    assert manifest["quality_repack_branch"]["changed_organs"] == [GATE, UP]
    changed = {
        row["tensor_name"]: row
        for row in manifest["tensors"]
        if row["candidate_mutation"]["changed_from_admitted_control"]
    }
    assert set(changed) == {GATE, UP}
    for row in changed.values():
        assert row["candidate_mutation"]["baseline_rollback"]["baseline_manifest_path"] == str(
            baseline_manifest
        )
        _, decoded = _unpack_sparse_residual(Path(row["artifact_path"]).read_bytes())
        assert decoded.shape == tuple(row["shape"])

    assert hashlib.sha256(baseline_manifest.read_bytes()).hexdigest() == baseline_manifest_before
    assert hashlib.sha256(baseline_admission.read_bytes()).hexdigest() == baseline_admission_before


def test_quality_terminal_handoff_never_requests_early_and_keeps_baseline_pointers_untouched(
    tmp_path: Path,
) -> None:
    """The separate watcher is inert until a sealed all-tensor terminal exists.

    The tiny fixture uses three source tensors, while production hard-codes the
    true 18,867 Qwen30 tensor count.  This keeps the same exact terminal/
    manifest/revalidation chain testable without pretending a partial journal
    is sufficient.
    """

    worker, root, baseline_manifest, baseline_admission = _fixture_worker(tmp_path)
    baseline_manifest_before = hashlib.sha256(baseline_manifest.read_bytes()).hexdigest()
    baseline_admission_before = hashlib.sha256(baseline_admission.read_bytes()).hexdigest()
    target = QualityAdmissionTarget(
        root=root,
        baseline_revalidation_path=worker.baseline_revalidation_path,
        repository="test/repository",
        revision="test-revision",
        expected_tensor_count=3,
    )
    fake_loader = tmp_path / "quality-native-admission"
    fake_loader.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(fake_loader, 0o700)

    # One bounded pack cycle has a source-bound journal, but no immutable
    # terminal manifest/ledger.  The watcher may heartbeat only; it cannot
    # create a request or invoke the native runner.
    assert worker.run(max_tensors=1) == 0
    calls: list[list[str]] = []

    def no_runner(command: object, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))  # pragma: no cover - this must never run here
        raise AssertionError("native quality admission started before terminal completion")

    waiting = run_quality_admission_once(target, native_loader=fake_loader, runner=no_runner)
    assert waiting["phase"] == "WAITING_FOR_IMMUTABLE_FULL_MANIFEST"
    assert calls == []
    assert not target.requests_root.exists()

    # Finish the separate physical candidate.  Its sealed terminal receipt is
    # now the only condition that permits a request/native scan.
    assert worker.run(max_tensors=16) == 0

    def native_success(command: object, timeout: float) -> subprocess.CompletedProcess[str]:
        values = list(command)
        calls.append(values)

        def argument(flag: str) -> str:
            return values[values.index(flag) + 1]

        payload = {
            "schema": NATIVE_RESULT_SCHEMA,
            "status": NATIVE_RESULT_STATUS,
            "model": "qwen30-quality-repack",
            "manifest_path": argument("--manifest"),
            "manifest_seal_sha256": argument("--expected-manifest-seal-sha256"),
            "source_audit_seal_sha256": argument("--expected-source-audit-seal-sha256"),
            "source_revision": "test-revision",
            "tensor_count": 3,
            "source_weight_elements": json.loads(target.manifest_path.read_text())["complete_physical_bpw_ledger"]["source_weight_elements"],
            "tensor_payload_bytes": json.loads(target.manifest_path.read_text())["complete_physical_bpw_ledger"]["tensor_payload_bytes"],
            "selected_residual_organs": [GATE, UP],
            "selected_residual_discriminators_verified": True,
            "payload_verification": {
                "mode": "bounded_parallel_source_shard_lanes_ordered_reconciliation_v1",
                "workers_used": 1,
                "workers_cap": 4,
                "manifest_rows": 3,
                "result_order": "manifest_ordinal_ascending_before_catalog_and_receipt",
                "candidate_only_read_path": True,
            },
        }
        return subprocess.CompletedProcess(values, 0, stdout=json.dumps(payload), stderr="")

    admitted = run_quality_admission_once(target, native_loader=fake_loader, runner=native_success)
    assert admitted["phase"] == NATIVE_RESULT_STATUS
    assert len(calls) == 1
    assert "--expected-terminal-seal-sha256" in calls[0]
    assert target.current_pointer_path.exists()
    assert len(list(target.requests_root.glob("*.json"))) == 1
    assert len(list(target.receipts_root.glob("*.json"))) == 1
    # A detached watcher revisits this terminal state.  It must reuse the
    # sealed candidate-local selector byte-for-byte rather than refresh its
    # timestamp and make downstream parity evidence drift.
    current_before = target.current_pointer_path.read_bytes()
    reused = run_quality_admission_once(target, native_loader=fake_loader, runner=no_runner)
    assert reused["phase"] == "EARNED_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_REUSED"
    assert target.current_pointer_path.read_bytes() == current_before
    assert len(calls) == 1
    assert hashlib.sha256(baseline_manifest.read_bytes()).hexdigest() == baseline_manifest_before
    assert hashlib.sha256(baseline_admission.read_bytes()).hexdigest() == baseline_admission_before


def test_parallel_scheduler_is_fixed_shard_disjoint_and_memory_bounded(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The coordinator may parallelize tensors, never a source shard/journal.

    This is intentionally a scheduling-level test: the production source plan
    has large contiguous shard ranges, so it proves that a four-worker wave is
    selected from distinct shards even when the fixed plan itself is grouped by
    shard.  The physical integration tests above cover actual atomic payload
    writes and terminal accounting.
    """

    worker, _, _, _ = _fixture_worker(tmp_path)
    planned = (
        ("a.safetensors", "a0"),
        ("a.safetensors", "a1"),
        ("b.safetensors", "b0"),
        ("b.safetensors", "b1"),
        ("c.safetensors", "c0"),
        ("d.safetensors", "d0"),
    )
    evidence = {shard: {"sha256": hashlib.sha256(shard.encode()).hexdigest()} for shard, _ in planned}

    def header(path: Path) -> dict[str, object]:
        prefix = path.name[0]
        return {
            f"{prefix}0": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            f"{prefix}1": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }

    monkeypatch.setattr(worker, "_header", header)
    work, evidence_row = worker._select_shard_disjoint_parallel_work(
        planned_order=planned,
        progress={},
        shard_evidence=evidence,
        worker_limit=3,
        memory_budget_bytes=128 * 1024 * 1024,
    )
    assert [item["tensor_name"] for item in work] == ["a0", "b0", "c0"]
    assert len({item["shard"] for item in work}) == len(work)
    assert evidence_row["selected_planned_ordinals"] == [0, 2, 4]
    assert evidence_row["fixed_plan_sha256"] == complete._canonical_sha256(list(planned))

    blocked, blocked_evidence = worker._select_shard_disjoint_parallel_work(
        planned_order=planned,
        progress={},
        shard_evidence=evidence,
        worker_limit=4,
        memory_budget_bytes=1,
    )
    assert blocked == []
    assert blocked_evidence["memory_deferred_shards"] == 4


def test_scalar_parity_native_result_refuses_any_direct_hq30gr2_fallback(tmp_path: Path) -> None:
    """The future scalar adapter cannot silently strip the residual wrapper."""

    probe = tmp_path / "scalar-parity-probe"
    probe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(probe, 0o700)
    pair_bindings = [
        {
            "organ": organ,
            "candidate": {"path": f"/candidate/{index}", "sha256": "a" * 64, "bytes": 10},
            "admitted_scalar_control": {"path": f"/control/{index}", "sha256": "b" * 64, "bytes": 8},
            "residual_count": 1,
        }
        for index, organ in enumerate((GATE, UP))
    ]
    evidence = {"pair_arguments": [], "pair_bindings": pair_bindings}

    def runner(command: object, timeout: float) -> subprocess.CompletedProcess[str]:
        pairs = []
        for binding in pair_bindings:
            pairs.append(
                {
                    "organ": binding["organ"],
                    "candidate_payload": binding["candidate"],
                    "admitted_scalar_control_payload": binding["admitted_scalar_control"],
                    "embedded_base_exactly_matches_admitted_control": True,
                    # The malicious/unsafe condition the wrapper must refuse.
                    "exact_fallback_refusal": {
                        "direct_decoder_refuses_hq30gr2": False,
                        "hq30gr2_decoder_refuses_direct_control": True,
                    },
                    "hq30gr2": {"magic": "HQ30GR2\\u0000", "residual_count": 1},
                    "scalar_identity": {
                        "candidate_equals_admitted_control_plus_exact_sparse_fp16_residual": True,
                        "changed_scalar_count": 1,
                    },
                    "projection_parity": {"max_abs_candidate_minus_control_minus_residual": 0.0},
                }
            )
        payload = {
            "schema": SCALAR_PARITY_RESULT_SCHEMA,
            "status": SCALAR_PARITY_RESULT_STATUS,
            "mode": "cpu_only_scalar_adapter_compatibility_parity_v1",
            "pairs": pairs,
            "claim_boundary": {
                "cpu_only": True,
                "metal_not_opened": True,
                "not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result": True,
                "not_a_capability_tg_agent_os_or_tournament_qualification": True,
            },
        }
        return subprocess.CompletedProcess(list(command), 0, stdout=json.dumps(payload), stderr="")

    target = ScalarParityTarget(root=tmp_path / "candidate", baseline_root=tmp_path / "baseline")
    with pytest.raises(ScalarParityError, match="exact residual semantics/fallback refusal"):
        invoke_scalar_parity_native(
            target=target,
            evidence=evidence,
            native_probe=probe,
            timeout_seconds=10.0,
            runner=runner,
        )
