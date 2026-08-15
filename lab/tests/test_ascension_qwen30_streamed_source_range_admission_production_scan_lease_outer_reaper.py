"""Focused fake-child-only tests for the Q30 production lease/outer reaper."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_lease_outer_reaper as lifecycle,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _write(path: Path, document: dict[str, object]) -> tuple[Path, dict[str, object]]:
    checked = seal(document)
    path.write_text(json.dumps(checked, sort_keys=True), encoding="utf-8")
    return path, checked


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(path: Path, document: dict[str, object]) -> dict[str, str]:
    return {
        "raw_document_sha256": _raw_sha(path),
        "seal_sha256": str(document["seal_sha256"]),
    }


def _bootstrap_preflight() -> dict[str, object]:
    return {
        "schema": outer.bootstrap.PREFLIGHT_SCHEMA,
        "status": outer.bootstrap.PREFLIGHT_STATUS,
        "prepared": True,
        "execution_authorized": False,
        "metadata_bindings": {
            "range_authority_document_sha256": SHA_A,
            "range_authority_content_sha256": SHA_B,
            "source_index_sha256": SHA_C,
            "maximum_declared_bf16_window_bytes": 524_288,
        },
        "future_bootstrap_lease": {
            "schema": outer.LEASE_SCHEMA,
            "status": outer.LEASE_STATUS,
            "one_shot": True,
            "separate_from_source_teacher_lease": True,
            "non_inference_only": True,
            "model_server_gpu_hcli_or_tps_allowed": False,
            "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
        },
        "future_outputs_required_before_source_teacher": {
            "flat_runtime_range_map": {
                "schema": "hawking.ascension.qwen30_source_bf16_range_map.v1",
                "shards": outer.SOURCE_SHARDS,
                "tensors": outer.SOURCE_TENSORS,
            },
            "runtime_admission": {
                "schema": "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1",
                "status": "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY",
            },
        },
        "execution_boundary": {
            "source_root_opened_or_statted": False,
            "source_tensor_payload_opened": False,
            "flat_runtime_range_map_emitted": False,
            "two_attestations_emitted": False,
            "runtime_admission_earned": False,
            "source_teacher_started": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
        },
    }


def _chain(tmp_path: Path) -> tuple[lifecycle.ExecuteConfig, dict[str, Path]]:
    preflight_path, preflight = _write(tmp_path / "bootstrap-preflight.json", _bootstrap_preflight())
    bootstrap_binary_path, bootstrap_binary = _write(
        tmp_path / "bootstrap-binary.json",
        {
            "schema": outer.bootstrap.BINARY_SCHEMA,
            "status": outer.bootstrap.BINARY_STATUS,
            "cpu_only": True,
            "scan_or_runtime_executed": False,
            "binary_sha256": SHA_D,
            "source_sha256": SHA_E,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
        },
    )
    bootstrap_resource_path, bootstrap_resource = _write(
        tmp_path / "bootstrap-resource.json",
        {
            "schema": outer.bootstrap.RESOURCE_SCHEMA,
            "status": outer.bootstrap.RESOURCE_STATUS,
            "fresh_observation": True,
            "exclusive_clean_window": True,
            "zero_swap": True,
            "zero_swapouts": True,
            "resource_admitted_for_one_future_child": True,
            "source_payload_opened": False,
            "source_model_loaded": False,
            "gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "child_started": False,
            "swap_used_bytes": 0,
            "swapouts_pages_delta": 0,
            "reclaimable_bytes": 10_000,
            "minimum_reclaimable_bytes_required": 1,
            "bootstrap_binary_sha256": SHA_D,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
            "resource_window_identity_sha256": SHA_C,
        },
    )
    executable = tmp_path / "production-scanner"
    executable.write_bytes(b"synthetic production scanner binding only\n")
    executable.chmod(0o755)
    source = tmp_path / "production-scanner.rs"
    source.write_text("// CPU-only synthetic binding\n", encoding="utf-8")
    binary_sha = _raw_sha(executable)
    source_sha = _raw_sha(source)
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_streamed_source_range_admission_production_scan_interface",
    ]
    command_sha = hashlib.sha256(
        json.dumps(command, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    production_binary_path, production_binary = _write(
        tmp_path / "production-binary.json",
        {
            "schema": outer.PRODUCTION_BINARY_SCHEMA,
            "status": outer.PRODUCTION_BINARY_STATUS,
            "cpu_only": True,
            "production_hash_scan_backend_compiled": True,
            "production_hash_scan_executed": False,
            "source_root_opened_or_statted": False,
            "source_payload_opened": False,
            "source_teacher_or_logits_executed": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "binary_sha256": binary_sha,
            "source_sha256": source_sha,
            "executable": {"path": str(executable), "bytes": executable.stat().st_size, "sha256": binary_sha},
            "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": source_sha},
            "compiled_command": command,
            "compiled_command_sha256": command_sha,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
            "bootstrap_binary": _pointer(bootstrap_binary_path, bootstrap_binary),
            "bootstrap_resource": _pointer(bootstrap_resource_path, bootstrap_resource),
        },
    )
    production_resource_path, production_resource = _write(
        tmp_path / "production-resource.json",
        {
            "schema": outer.PRODUCTION_RESOURCE_SCHEMA,
            "status": outer.PRODUCTION_RESOURCE_STATUS,
            "prepared": True,
            "fresh_observation": True,
            "observed_after_production_binary_binding": True,
            "exclusive_clean_window": True,
            "zero_swap": True,
            "zero_swapouts": True,
            "no_active_q30_or_q80_capture_child": True,
            "resource_admitted_for_one_future_child": True,
            "source_payload_opened": False,
            "source_model_loaded": False,
            "source_teacher_or_logits_executed": False,
            "native_phase_started": False,
            "gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "child_started": False,
            "swap_used_bytes": 0,
            "swapouts_pages_delta": 0,
            "reclaimable_bytes": 10_000,
            "minimum_reclaimable_bytes_required": 1,
            "production_binary_binding": _pointer(production_binary_path, production_binary),
            "bootstrap_resource_ancestry": _pointer(bootstrap_resource_path, bootstrap_resource),
            "production_resource_window_identity_sha256": SHA_E,
        },
    )
    interface_path, interface = _write(
        tmp_path / "production-interface.json",
        {
            "schema": outer.INTERFACE_SCHEMA,
            "status": outer.INTERFACE_STATUS,
            "prepared": True,
            "execution_authorized": False,
            "strict_non_fixture_boundary": {"before_source_root_access": True},
            "future_bounded_hash_scan": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
            },
        },
    )
    authority_path, authority = _write(
        tmp_path / "production-authority.json",
        {
            "schema": outer.PRODUCTION_AUTHORITY_SCHEMA,
            "status": outer.PRODUCTION_AUTHORITY_STATUS,
            "fresh_for_this_exact_scan": True,
            "one_shot": True,
            "non_inference_hash_scan_only": True,
            "source_root_open_only_after_all_authorities_validate": True,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "source_teacher_or_logits_authorized": False,
            "model_gpu_server_hcli_or_tps_authorized": False,
            "lease_consumed": False,
            "immutable_bindings": {
                "interface_authority": _pointer(interface_path, interface),
                "production_binary": _pointer(production_binary_path, production_binary),
                "production_resource_admission": _pointer(production_resource_path, production_resource),
            },
            "geometry": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "exact_scan_nonce_sha256": SHA_A,
        },
    )
    baseline = outer.build_outer_preflight(
        bootstrap_preflight_path=preflight_path,
        bootstrap_binary_path=bootstrap_binary_path,
        bootstrap_resource_path=bootstrap_resource_path,
        production_binary_path=production_binary_path,
        production_resource_path=production_resource_path,
        production_interface_path=interface_path,
        production_authority_path=authority_path,
    )
    assert baseline["blockers"] == ["fresh_production_hash_scan_bootstrap_lease_absent"]
    baseline_path, _ = _write(tmp_path / "baseline-outer.json", baseline)
    config = lifecycle.ExecuteConfig(
        outer_preflight_path=baseline_path,
        bootstrap_preflight_path=preflight_path,
        bootstrap_binary_path=bootstrap_binary_path,
        bootstrap_resource_path=bootstrap_resource_path,
        production_binary_path=production_binary_path,
        production_resource_path=production_resource_path,
        production_interface_path=interface_path,
        production_authority_path=authority_path,
        launch_dir=tmp_path / "new-launch",
        replay_dir=tmp_path / "new-replay",
        capture_dir=tmp_path / "new-capture",
    )
    return config, {"baseline": baseline_path, "authority": authority_path}


def test_preflight_consumes_no_lease_outer_and_creates_nothing(tmp_path: Path) -> None:
    config, _ = _chain(tmp_path)
    result = lifecycle.build_issuer_preflight(config)
    assert result["schema"] == lifecycle.SCHEMA
    assert result["status"] == lifecycle.PREFLIGHT_STATUS
    assert result["prepared"] is True
    assert result["lease_issued"] is False
    assert result["execution_boundary"]["source_root_opened_or_statted"] is False
    assert not config.launch_dir.exists()
    assert not config.replay_dir.exists()
    assert not config.capture_dir.exists()
    verify(result, label="lease issuer preflight")


def test_fake_child_is_reaped_once_and_can_never_earn_production_capture(tmp_path: Path) -> None:
    config, _ = _chain(tmp_path)
    result = lifecycle.run_fake_child_test(
        config,
        fake_child_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
    )
    lease = result["lease"]
    terminal = result["outer_terminal"]
    release = result["release"]
    assert lease["schema"] == outer.LEASE_SCHEMA
    authority_document = json.loads(
        config.production_authority_path.read_text(encoding="utf-8")
    )
    assert lease["production_scan_authority"]["canonical_document_sha256"] == hashlib.sha256(
        json.dumps(
            authority_document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert lease["production_resource_window_identity_sha256"] == SHA_E
    assert lease["fresh_production_resource_observation"]["zero_swap"] is True
    assert result["issued_outer_preflight"]["status"] == outer.PREPARED_STATUS
    assert terminal["status"] == lifecycle.TERMINAL_REFUSED_STATUS
    assert terminal["child_reaped"] is True
    assert terminal["child_capture"]["present"] is False
    assert "test-only fake child" in terminal["child_capture_validation_error"]
    assert release["release_after_outer_terminal"] is True
    assert release["one_shot_lease_finalized"] is True
    assert release["outer_terminal_seal_sha256"] == terminal["seal_sha256"]
    assert (config.replay_dir / lifecycle.REPLAY_FILENAME).is_file()
    assert result["outer_terminal_path"].is_file()
    assert result["release_path"].is_file()
    verify(lease, label="fake-test lease")
    verify(terminal, label="fake-test terminal")
    verify(release, label="fake-test release")

    with pytest.raises(lifecycle.ProductionLeaseOuterError, match="new absolute path"):
        lifecycle.run_fake_child_test(
            config,
            fake_child_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        )


def test_real_execute_is_refused_without_explicit_enablement_before_any_lease(tmp_path: Path) -> None:
    config, _ = _chain(tmp_path)
    with pytest.raises(lifecycle.ProductionLeaseOuterError, match="explicit real-source-scan"):
        lifecycle.run_execute(config)
    assert not config.launch_dir.exists()
    assert not config.replay_dir.exists()
    assert not config.capture_dir.exists()


def test_forbidden_fake_teacher_command_is_refused_before_any_lease_or_directory(
    tmp_path: Path,
) -> None:
    config, _ = _chain(tmp_path)
    with pytest.raises(lifecycle.ProductionLeaseOuterError, match="forbidden teacher/native/GPU/HCLI"):
        lifecycle.run_fake_child_test(config, fake_child_command=("synthetic-source-teacher",))
    assert not config.launch_dir.exists()
    assert not config.replay_dir.exists()
    assert not config.capture_dir.exists()


def test_fake_lease_regression_rejects_missing_or_substituted_authority_canonical_identity(
    tmp_path: Path,
) -> None:
    config, _ = _chain(tmp_path)
    context = lifecycle._load_context(config)
    expected = lifecycle._lease_document(context, lease_id=SHA_B)
    assert "canonical_document_sha256" in expected["production_scan_authority"]
    for label, replacement in (("missing", None), ("substituted", SHA_C)):
        tampered = dict(expected)
        pointer = dict(tampered["production_scan_authority"])
        if replacement is None:
            pointer.pop("canonical_document_sha256")
        else:
            pointer["canonical_document_sha256"] = replacement
        tampered["production_scan_authority"] = pointer
        path, _ = _write(tmp_path / f"{label}-lease.json", tampered)
        with pytest.raises(
            lifecycle.ProductionLeaseOuterError, match="canonical binding drifted"
        ):
            lifecycle._validate_issued_lease(path, context)
