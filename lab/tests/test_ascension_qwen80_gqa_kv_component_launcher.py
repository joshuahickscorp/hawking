"""CPU-only contract tests for the future Qwen80 GQA K/V outer launcher.

The executable used here is a shell fake that only writes a temporary JSON
receipt.  It never loads model data, opens a Metal context, dispatches work,
edits a registry, starts a watcher/server, or issues a lease.  The lease is a
read-only sealed fixture so the launcher can prove that it only consumes
future authority; these tests do not call or emulate any lease issuer.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_gqa_kv_component_launcher as launcher


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


def _binding(
    evidence: dict[str, object], *, schema: str, status: str, seal_sha256: str | None = None
) -> dict[str, object]:
    result = {**evidence, "schema": schema, "status": status}
    if seal_sha256 is not None:
        result["seal_sha256"] = seal_sha256
    return result


def _state_buffer(handle: str, salt: str, *, max_seq_len: int = 2) -> dict[str, object]:
    elements = max_seq_len * launcher.KV_ROW_ELEMENTS
    return {
        "handle": handle,
        "shape": [max_seq_len, launcher.GQA_KV_HEADS, launcher.GQA_HEAD_DIM],
        "elements": elements,
        "bytes": elements * 4,
        "f32le_sha256": salt * 64,
    }


def _inputs(tmp_path: Path) -> dict[str, object]:
    manifest = tmp_path / "manifest.json"
    manifest_document, manifest_evidence = _sealed(
        manifest,
        {"schema": launcher.MANIFEST_SCHEMA, "status": "FIXTURE_CURRENT_COMPLETE_ARTIFACT"},
    )

    admission_receipt = tmp_path / "admission-receipt.json"
    admission_receipt_document, admission_receipt_evidence = _sealed(
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
                "revision": launcher.SOURCE_REVISION,
                "source_audit_seal_sha256": "a" * 64,
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
                "path": admission_receipt_evidence["path"],
                "document_sha256": admission_receipt_evidence["sha256"],
                "seal_sha256": admission_receipt_document["seal_sha256"],
            },
        },
    )

    authority = tmp_path / "future-source-hidden-authority.json"
    authority_document, authority_evidence = _sealed(
        authority,
        {
            "schema": launcher.SOURCE_HIDDEN_AUTHORITY_SCHEMA,
            "status": launcher.SOURCE_HIDDEN_AUTHORITY_STATUS,
            "source_binding": {
                "model_id": launcher.MODEL_ID,
                "model_key": launcher.MODEL_KEY,
                "source_repository": launcher.SOURCE_REPOSITORY,
                "source_revision": launcher.SOURCE_REVISION,
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_pointer_seal_sha256": admission_document["seal_sha256"],
                "admission_receipt_seal_sha256": admission_receipt_document["seal_sha256"],
            },
            "source_hidden": {
                "layer": launcher.SELECTED_LAYER,
                "gqa_slot": launcher.SELECTED_SLOT,
                "elements": launcher.HIDDEN,
                "f32le_sha256": "b" * 64,
                "source_bound": True,
                "synthetic_or_fixture": False,
            },
            "caller_owned_active_and_rollback_state": {
                "session_id": "fixture-q80-gqa-session",
                "token_position": 0,
                "max_seq_len": 2,
                "selected_layer": launcher.SELECTED_LAYER,
                "selected_slot": launcher.SELECTED_SLOT,
                "caller_owned_by_upstream": True,
                "active_and_rollback_disjoint": True,
                "active_key": _state_buffer("upstream.active.key.layer3.slot0", "c"),
                "active_value": _state_buffer("upstream.active.value.layer3.slot0", "d"),
                "rollback_key": _state_buffer("upstream.rollback.key.layer3.slot0", "e"),
                "rollback_value": _state_buffer("upstream.rollback.value.layer3.slot0", "f"),
            },
            "upstream_evidence": {
                "source_hidden_parity_evidence_earned": True,
                "state_readback_authority_earned": True,
                "receipt_written_last_is_completion_marker": True,
                "complete_layer_or_token_performed": False,
                "decoder_or_generation_performed": False,
            },
        },
    )

    compact_abi = tmp_path / "layer3-slot0-compact-abi.json"
    _write_json(
        compact_abi,
        {
            "schema": launcher.COMPACT_ABI_SCHEMA,
            "status": launcher.COMPACT_ABI_STATUS,
            "source_binding": {
                "model_id": launcher.MODEL_ID,
                "model_key": launcher.MODEL_KEY,
                "source_repository": launcher.SOURCE_REPOSITORY,
                "source_revision": launcher.SOURCE_REVISION,
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_receipt_seal_sha256": admission_receipt_document["seal_sha256"],
            },
            "geometry": {
                "layer": launcher.SELECTED_LAYER,
                "slot": launcher.SELECTED_SLOT,
                "hidden": launcher.HIDDEN,
                "kv_heads": launcher.GQA_KV_HEADS,
                "head_dim": launcher.GQA_HEAD_DIM,
                "kv_row_elements": launcher.KV_ROW_ELEMENTS,
                "minimum_context": 2,
                "maximum_context": launcher.MAX_NATIVE_CONTEXT,
            },
            "direct_packed_projection_abi": list(launcher.EXPECTED_PROJECTION_ABI),
            "component_command_order": list(launcher.EXPECTED_COMMAND_ORDER),
            "claim_boundary": {
                "artifact_scan_or_payload_open_performed": False,
                "metal_context_or_dispatch_performed": False,
                "runtime_watcher_server_registry_or_hcli_changed": False,
                "complete_layer_or_token_performed": False,
                "decoder_or_generation_performed": False,
                "tps_or_tg_claim": False,
            },
        },
    )
    compact_abi_evidence = _evidence(compact_abi)

    # This is a static input fixture, not a lease issuance.  The launcher only
    # validates and hashes it; all produced receipt paths are under capture/.
    lease = tmp_path / "read-only-fixture-quiet-lease.json"
    lease_document, lease_evidence = _sealed(
        lease,
        {
            "schema": launcher.LEASE_SCHEMA,
            "status": launcher.LEASE_STATUS,
            "lease_id": "fixture-future-q80-gqa-component",
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "automatic_retry_prohibited": True,
                "outer_reaped_capture_required": True,
                "prior_terminal_receipt": None,
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
                "admission_receipt_seal_sha256": admission_receipt_document["seal_sha256"],
                "selected_layer": launcher.SELECTED_LAYER,
                "selected_slot": launcher.SELECTED_SLOT,
            },
            "source_hidden_authority_binding": _binding(
                authority_evidence,
                schema=launcher.SOURCE_HIDDEN_AUTHORITY_SCHEMA,
                status=launcher.SOURCE_HIDDEN_AUTHORITY_STATUS,
                seal_sha256=str(authority_document["seal_sha256"]),
            ),
            "compact_abi_contract_binding": _binding(
                compact_abi_evidence,
                schema=launcher.COMPACT_ABI_SCHEMA,
                status=launcher.COMPACT_ABI_STATUS,
            ),
        },
    )
    return {
        "manifest": manifest,
        "manifest_document": manifest_document,
        "manifest_evidence": manifest_evidence,
        "admission": admission,
        "admission_document": admission_document,
        "admission_evidence": admission_evidence,
        "admission_receipt_document": admission_receipt_document,
        "authority": authority,
        "authority_document": authority_document,
        "authority_evidence": authority_evidence,
        "compact_abi": compact_abi,
        "compact_abi_evidence": compact_abi_evidence,
        "lease": lease,
        "lease_document": lease_document,
        "lease_evidence": lease_evidence,
    }


def _probe(tmp_path: Path, body: str) -> tuple[Path, Path]:
    marker = tmp_path / "fake-child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{body}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _config(tmp_path: Path, probe: Path, inputs: dict[str, object]) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        admission_current=inputs["admission"],  # type: ignore[arg-type]
        source_hidden_authority=inputs["authority"],  # type: ignore[arg-type]
        compact_abi_contract=inputs["compact_abi"],  # type: ignore[arg-type]
        lease_receipt=inputs["lease"],  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        timeout_seconds=10.0,
    )


def _readback(elements: int, salt: str) -> dict[str, object]:
    return {"elements": elements, "exact": True, "max_abs_error": 0.0, "f32le_sha256": salt * 64}


def _inner_receipt(config: launcher.LaunchConfig, inputs: dict[str, object]) -> dict[str, object]:
    manifest_document = inputs["manifest_document"]
    admission_document = inputs["admission_document"]
    admission_receipt_document = inputs["admission_receipt_document"]
    authority_document = inputs["authority_document"]
    lease_document = inputs["lease_document"]
    assert isinstance(manifest_document, dict)
    assert isinstance(admission_document, dict)
    assert isinstance(admission_receipt_document, dict)
    assert isinstance(authority_document, dict)
    assert isinstance(lease_document, dict)
    authority_evidence = inputs["authority_evidence"]
    compact_abi_evidence = inputs["compact_abi_evidence"]
    lease_evidence = inputs["lease_evidence"]
    assert isinstance(authority_evidence, dict)
    assert isinstance(compact_abi_evidence, dict)
    assert isinstance(lease_evidence, dict)
    return seal(
        {
            "schema": launcher.EXPECTED_INNER_SCHEMA,
            "status": launcher.EXPECTED_INNER_STATUS,
            "component_only": True,
            "complete_layer_or_token_performed": False,
            "decoder_or_generation_performed": False,
            "metal_device_or_dispatch_performed": True,
            "metal_execution_policy": {
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "durable_capture": {
                "receipt_written_last_is_completion_marker": True,
                "source_hidden_and_state_readbacks_written_before_receipt": True,
                "outer_reaped_capture_required": True,
            },
            "artifact_binding": {
                "manifest_document_sha256": inputs["manifest_evidence"]["sha256"],  # type: ignore[index]
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_pointer_seal_sha256": admission_document["seal_sha256"],
                "admission_receipt_seal_sha256": admission_receipt_document["seal_sha256"],
            },
            "source_hidden_authority_binding": _binding(
                authority_evidence,
                schema=launcher.SOURCE_HIDDEN_AUTHORITY_SCHEMA,
                status=launcher.SOURCE_HIDDEN_AUTHORITY_STATUS,
                seal_sha256=str(authority_document["seal_sha256"]),
            ),
            "compact_abi_contract_binding": _binding(
                compact_abi_evidence,
                schema=launcher.COMPACT_ABI_SCHEMA,
                status=launcher.COMPACT_ABI_STATUS,
            ),
            "lease_binding": _binding(
                lease_evidence,
                schema=launcher.LEASE_SCHEMA,
                status=launcher.LEASE_STATUS,
                seal_sha256=str(lease_document["seal_sha256"]),
            ),
            "caller_owned_state_binding": {
                "session_id": "fixture-q80-gqa-session",
                "token_position": 0,
                "max_seq_len": 2,
                "selected_layer": launcher.SELECTED_LAYER,
                "selected_slot": launcher.SELECTED_SLOT,
                "source_hidden_f32le_sha256": "b" * 64,
                "active_key_f32le_sha256": "c" * 64,
                "active_value_f32le_sha256": "d" * 64,
                "rollback_key_f32le_sha256": "e" * 64,
                "rollback_value_f32le_sha256": "f" * 64,
            },
            "readback_parity": {
                "active_key_slot_row_after_append": _readback(launcher.KV_ROW_ELEMENTS, "1"),
                "active_value_slot_row_after_append": _readback(launcher.KV_ROW_ELEMENTS, "2"),
                "q_projection_rows": _readback(8_192, "3"),
                "o_projection_output": _readback(launcher.HIDDEN, "4"),
            },
            "rollback_readback": {
                "restored_active_key_exact": True,
                "restored_active_value_exact": True,
                "active_key_after_rollback_f32le_sha256": "c" * 64,
                "active_value_after_rollback_f32le_sha256": "d" * 64,
            },
        }
    )


def _inner_body(receipt: dict[str, object]) -> str:
    rendered = shlex.quote(json.dumps(receipt, sort_keys=True))
    return (
        'capture=""; previous=""; '
        'for value in "$@"; do '
        'if [ "$previous" = "--capture-dir" ]; then capture="$value"; break; fi; '
        'previous="$value"; done; '
        'mkdir "$capture"; '
        f"printf '%s\\n' {rendered} > \"$capture/receipt.json\"; "
        'echo "fake child stdout"; echo "fake child stderr" >&2; exit 0'
    )


def test_missing_future_source_hidden_authority_refuses_before_child_or_capture(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)
    config = launcher.LaunchConfig(
        **{**config.__dict__, "source_hidden_authority": tmp_path / "missing-upstream-authority.json"}
    )

    with pytest.raises(launcher.GqaKvComponentLauncherError, match="source-hidden-authority"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_reaps_one_valid_future_child_and_never_mutates_its_lease_fixture(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    child_receipt = _inner_receipt(config, inputs)
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(child_receipt)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    lease = inputs["lease"]
    assert isinstance(lease, Path)
    lease_before = lease.read_bytes()

    first = launcher.run_attempt(config)
    replay = launcher.run_attempt(config)

    assert first["status"] == "CAPTURED_QWEN80_GQA_KV_COMPONENT_OUTER_TERMINAL_COMPONENT_ONLY"
    assert first["child"]["terminal"]["reaped"] is True
    assert first["inner_probe_capture"]["binding_valid"] is True
    assert first["one_shot"]["terminal_receipt_written_last"] is True
    assert first["claim_boundary"]["outer_controller_is_cpu_only"] is True
    assert first["claim_boundary"]["outer_did_not_issue_or_mutate_a_lease"] is True
    assert "--source-hidden-authority" in first["child"]["command"]
    assert "--compact-abi-contract" in first["child"]["command"]
    assert marker.read_text(encoding="utf-8") == "run"
    assert replay == first
    assert lease.read_bytes() == lease_before
    assert list(tmp_path.glob("*quiet-lease*.json")) == [lease]
    verify(first)
    assert json.loads((config.capture_dir / launcher.TERMINAL_FILENAME).read_text(encoding="utf-8")) == first


def test_zero_exit_without_exact_rollback_readback_is_retained_as_refusal(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    child_receipt = _inner_receipt(config, inputs)
    rollback = dict(child_receipt["rollback_readback"])
    rollback["restored_active_value_exact"] = False
    child_receipt = seal({key: value for key, value in child_receipt.items() if key != "seal_sha256"} | {"rollback_readback": rollback})
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(child_receipt)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT")
    assert receipt["inner_probe_capture"]["binding_valid"] is False
    assert "rollback readback parity failed" in receipt["inner_probe_capture"]["binding_error"]
    assert marker.read_text(encoding="utf-8") == "run"
    verify(receipt)


def test_unsealed_compact_abi_byte_drift_refuses_before_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)
    compact_abi = inputs["compact_abi"]
    assert isinstance(compact_abi, Path)
    compact_abi.write_text(compact_abi.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(launcher.GqaKvComponentLauncherError, match="compact ABI.*drifted"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()
