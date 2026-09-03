"""CPU-only fake-child tests for the Qwen80 first-residual outer launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_first_residual_bridge_launcher as launcher


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(path: Path, document: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    sealed_document = seal(document)
    _write_json(path, sealed_document)
    return sealed_document, _evidence(path)


def _inputs(tmp_path: Path) -> dict[str, Path | str]:
    manifest = tmp_path / "manifest.json"
    manifest_document, manifest_evidence = _sealed(
        manifest,
        {"schema": launcher.MANIFEST_SCHEMA, "status": "FIXTURE_COMPLETE_ARTIFACT"},
    )
    admission_receipt = tmp_path / "admission-receipt.json"
    receipt_document, receipt_evidence = _sealed(
        admission_receipt,
        {
            "schema": launcher.ADMISSION_RECEIPT_SCHEMA,
            "status": launcher.ADMISSION_RECEIPT_STATUS,
            "complete_manifest": {
                "path": manifest_evidence["path"],
                "document_sha256": manifest_evidence["sha256"],
                "seal_sha256": manifest_document["seal_sha256"],
            },
            "current_source_revalidation": {
                "source_audit_seal_sha256": "c" * 64,
                "revision": "fixture-revision-a7fb",
            },
        },
    )
    admission = tmp_path / "admission-current.json"
    admission_document, admission_evidence = _sealed(
        admission,
        {
            "schema": launcher.ADMISSION_POINTER_SCHEMA,
            "status": launcher.ADMISSION_POINTER_STATUS,
            "complete_manifest": {
                "path": manifest_evidence["path"],
                "document_sha256": manifest_evidence["sha256"],
                "seal_sha256": manifest_document["seal_sha256"],
            },
            "admission_receipt": {
                "path": receipt_evidence["path"],
                "document_sha256": receipt_evidence["sha256"],
                "seal_sha256": receipt_document["seal_sha256"],
            },
        },
    )

    input_hidden = tmp_path / "input-hidden.f32le"
    input_hidden.write_bytes(bytes(range(256)) * 32)
    assert input_hidden.stat().st_size == launcher.FIRST_RESIDUAL_BYTES
    input_evidence = _evidence(input_hidden)
    first_residual = tmp_path / "first-residual.f32le"
    first_residual.write_bytes(bytes(reversed(range(256))) * 32)
    assert first_residual.stat().st_size == launcher.FIRST_RESIDUAL_BYTES
    residual_evidence = _evidence(first_residual)

    baseline = tmp_path / "cpu-baseline-receipt.json"
    _write_json(
        baseline,
        {
            "schema": launcher.CPU_BASELINE_SCHEMA,
            "status": launcher.CPU_BASELINE_STATUS,
            "mode": "cpu-oracle",
            "metal_device_or_dispatch_performed": False,
            "component_only": True,
            "complete_layer_or_token_performed": False,
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "source_audit_seal_sha256": "c" * 64,
                "source_revision": "fixture-revision-a7fb",
                "layer": 0,
                "linear_state_slot": 0,
                "admission_scan_performed_once_before_catalog_reuse": True,
                "direct_packed_payloads_only": True,
            },
            "same_input_provenance": {
                "kind": "source_direct_packed_embedding_with_zeroed_layer0_deltanet_state",
                "token_id": 42,
                "embedding_tensor": "model.embed_tokens.weight",
                "input_hidden": input_evidence,
                "input_hidden_f32le_sha256": input_evidence["sha256"],
                "initial_conv_state": {
                    "elements": launcher.CONV_STATE_ELEMENTS,
                    "f32le_sha256": "d" * 64,
                    "zero_initialized": True,
                },
                "initial_recurrent_state": {
                    "elements": launcher.RECURRENT_STATE_ELEMENTS,
                    "f32le_sha256": "e" * 64,
                    "zero_initialized": True,
                },
                "future_strict_metal_child_must_retain_exact_input_and_state_identity": True,
            },
            "first_residual_output": {
                "layer": 0,
                "linear_state_slot": 0,
                "elements": launcher.HIDDEN,
                "bytes": launcher.FIRST_RESIDUAL_BYTES,
                "sha256": residual_evidence["sha256"],
                "f32le_sha256": residual_evidence["sha256"],
                "file": residual_evidence,
                "same_command_graph_required_for_future_strict_metal_bridge": True,
            },
            "durable_capture": {
                "input_hidden_written_before_receipt": True,
                "first_residual_written_before_receipt": True,
                "receipt_written_last_is_completion_marker": True,
                "outer_reaped_strict_metal_capture_required_before_any_device_or_layer_promotion": True,
            },
        },
    )
    baseline_evidence = _evidence(baseline)
    lease = tmp_path / "quiet-lease.json"
    lease_document, lease_evidence = _sealed(
        lease,
        {
            "schema": launcher.LEASE_SCHEMA,
            "status": launcher.LEASE_STATUS,
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "automatic_retry_prohibited": True,
                "outer_reaped_capture_required": True,
            },
            "execution_policy": {
                "component": launcher.LEASE_COMPONENT,
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "artifact_binding": {
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_receipt_seal_sha256": receipt_document["seal_sha256"],
            },
            "cpu_baseline_binding": {
                "receipt_path": baseline_evidence["path"],
                "receipt_document_sha256": baseline_evidence["sha256"],
                "schema": launcher.CPU_BASELINE_SCHEMA,
                "status": launcher.CPU_BASELINE_STATUS,
            },
        },
    )
    return {
        "manifest": manifest,
        "admission": admission,
        "baseline": baseline,
        "lease": lease,
        "manifest_sha256": str(manifest_evidence["sha256"]),
        "manifest_seal": str(manifest_document["seal_sha256"]),
        "admission_pointer_seal": str(admission_document["seal_sha256"]),
        "admission_receipt_seal": str(receipt_document["seal_sha256"]),
        "baseline_sha256": str(baseline_evidence["sha256"]),
        "lease_sha256": str(lease_evidence["sha256"]),
        "lease_seal": str(lease_document["seal_sha256"]),
        "input_sha256": str(input_evidence["sha256"]),
        "residual_sha256": str(residual_evidence["sha256"]),
    }


def _probe(tmp_path: Path, body: str) -> tuple[Path, Path]:
    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{body}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _config(
    tmp_path: Path, probe: Path, inputs: dict[str, Path | str], *, include_lease: bool = True
) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        admission_current=inputs["admission"],  # type: ignore[arg-type]
        cpu_baseline_receipt=inputs["baseline"],  # type: ignore[arg-type]
        lease_receipt=inputs["lease"] if include_lease else None,  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        workers=1,
        timeout_seconds=10.0,
    )


def _inner_receipt(
    config: launcher.LaunchConfig, inputs: dict[str, Path | str]
) -> dict[str, object]:
    assert config.lease_receipt is not None
    baseline = json.loads(config.cpu_baseline_receipt.read_text(encoding="utf-8"))
    provenance = baseline["same_input_provenance"]
    return {
        "schema": launcher.DEVICE_INNER_SCHEMA,
        "status": launcher.DEVICE_INNER_STATUS,
        "mode": "metal",
        "metal_device_or_dispatch_performed": True,
        "component_only": True,
        "complete_layer_or_token_performed": False,
        "synthetic_input": False,
        "fixture_only": False,
        "durable_capture": {
            "receipt_written_last_is_completion_marker": True,
            "outer_reaped_capture_required": True,
            "replay_guarded": True,
        },
        "artifact_binding": {
            "manifest_document_sha256": inputs["manifest_sha256"],
            "manifest_seal_sha256": inputs["manifest_seal"],
            "admission_pointer_seal_sha256": inputs["admission_pointer_seal"],
            "admission_receipt_seal_sha256": inputs["admission_receipt_seal"],
            "source_audit_seal_sha256": "c" * 64,
            "source_revision": "fixture-revision-a7fb",
            "layer": 0,
            "linear_state_slot": 0,
        },
        "cpu_baseline_binding": {
            "receipt_path": str(config.cpu_baseline_receipt.resolve()),
            "receipt_document_sha256": inputs["baseline_sha256"],
            "schema": launcher.CPU_BASELINE_SCHEMA,
            "status": launcher.CPU_BASELINE_STATUS,
        },
        "same_input_provenance": {
            "kind": provenance["kind"],
            "token_id": provenance["token_id"],
            "embedding_tensor": provenance["embedding_tensor"],
            "input_hidden_f32le_sha256": provenance["input_hidden_f32le_sha256"],
            "initial_conv_state": provenance["initial_conv_state"],
            "initial_recurrent_state": provenance["initial_recurrent_state"],
        },
        "first_residual_output": {
            "layer": 0,
            "linear_state_slot": 0,
            "elements": launcher.HIDDEN,
            "bytes": launcher.FIRST_RESIDUAL_BYTES,
            "sha256": "f" * 64,
            "cpu_reference_sha256": inputs["residual_sha256"],
            "all_finite": True,
        },
        "same_command_graph": {
            "same_command_graph_required": True,
            "same_command_graph_retained": True,
            "command_buffer_identity": "fixture-command-buffer-1",
            "device_first_residual_buffer_bytes": launcher.FIRST_RESIDUAL_BYTES,
            "input_then_deltanet_then_first_residual_then_fence_order": True,
            "prefix_dispatches": 9,
            "suffix_dispatches": 0,
            "total_dispatches": 9,
            "prefix_only": True,
            "no_true_moe_suffix_encoded": True,
        },
        "cpu_device_parity": {
            "checked_elements": launcher.HIDDEN,
            "passed": True,
            "max_abs_error": 0.00001,
            "tolerance": 0.001,
        },
        "state_witness": {
            "linear_state_slot": 0,
            "conv_state_elements": launcher.CONV_STATE_ELEMENTS,
            "recurrent_state_elements": launcher.RECURRENT_STATE_ELEMENTS,
            "state_commit_after_parity_fence": True,
        },
        "metal_execution_policy": {
            "strict_math_required": True,
            "timing_or_benchmarking_allowed": False,
            "complete_layer_or_token_allowed": False,
            "tps_or_tg_claim_allowed": False,
            "lease_binding": {
                "receipt_path": str(config.lease_receipt.resolve()),
                "receipt_document_sha256": inputs["lease_sha256"],
                "seal_sha256": inputs["lease_seal"],
                "schema": launcher.LEASE_SCHEMA,
                "status": launcher.LEASE_STATUS,
            },
        },
    }


def _inner_body(receipt: dict[str, object]) -> str:
    rendered = shlex.quote(json.dumps(receipt, sort_keys=True))
    return (
        'capture=""; previous=""; '
        'for value in "$@"; do '
        'if [ "$previous" = "--capture-dir" ]; then capture="$value"; break; fi; '
        'previous="$value"; done; '
        'mkdir "$capture"; '
        f"printf '%s\\n' {rendered} > \"$capture/receipt.json\"; "
        'echo "child stdout"; echo "child stderr" >&2; exit 0'
    )


def test_requires_fresh_lease_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, include_lease=False)

    with pytest.raises(launcher.FirstResidualBridgeLauncherError, match="lease"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_rejects_historical_fixture_baseline_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    baseline = inputs["baseline"]
    assert isinstance(baseline, Path)
    document = json.loads(baseline.read_text(encoding="utf-8"))
    document["same_input_provenance"]["kind"] = "deterministic_fixture_input"
    _write_json(baseline, document)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)

    with pytest.raises(launcher.FirstResidualBridgeLauncherError, match="same-input"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_reaps_valid_strict_inner_and_replays_once(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(_inner_receipt(config, inputs))}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    first = launcher.run_attempt(config)
    second = launcher.run_attempt(config)

    assert first["status"] == launcher.CAPTURED_STATUS
    assert first["inner_probe_capture"]["binding_valid"] is True
    assert first["first_residual_output"] == {
        "layer": 0,
        "linear_state_slot": 0,
        "elements": launcher.HIDDEN,
        "same_command_graph_required": True,
        "sha256": "f" * 64,
    }
    assert second == first
    assert marker.read_text(encoding="utf-8") == "run"
    verify(first)


def test_zero_exit_without_same_command_graph_is_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    inner = _inner_receipt(config, inputs)
    inner["same_command_graph"] = {
        "same_command_graph_required": True,
        "same_command_graph_retained": False,
        "command_buffer_identity": "fixture-command-buffer-1",
        "device_first_residual_buffer_bytes": launcher.FIRST_RESIDUAL_BYTES,
        "input_then_deltanet_then_first_residual_then_fence_order": True,
    }
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(inner)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT")
    assert receipt["inner_probe_capture"]["binding_valid"] is False
    assert "same-graph" in receipt["inner_probe_capture"]["binding_error"]
    assert marker.read_text(encoding="utf-8") == "run"
    verify(receipt)
