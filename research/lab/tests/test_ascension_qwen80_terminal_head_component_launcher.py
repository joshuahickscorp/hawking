"""CPU-only fake-child tests for Qwen80's future terminal-head outer launcher.

All authority documents here are sealed local fixtures.  The fake child writes
only a JSON receipt; neither the tests nor the launcher touch Metal, a model
payload, a server, HCLI, or a lease issuer.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_terminal_head_component_launcher as launcher


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
    evidence: dict[str, object], *, schema: str, status: str, seal_sha256: str
) -> dict[str, object]:
    return {**evidence, "schema": schema, "status": status, "seal_sha256": seal_sha256}


def _source_binding(inputs: dict[str, object]) -> dict[str, object]:
    manifest_evidence = inputs["manifest_evidence"]
    admission_document = inputs["admission_document"]
    admission_receipt_document = inputs["admission_receipt_document"]
    assert isinstance(manifest_evidence, dict)
    assert isinstance(admission_document, dict)
    assert isinstance(admission_receipt_document, dict)
    return {
        "model_id": launcher.MODEL_ID,
        "model_key": launcher.MODEL_KEY,
        "source_repository": launcher.SOURCE_REPOSITORY,
        "source_revision": launcher.SOURCE_REVISION,
        "manifest_document_sha256": manifest_evidence["sha256"],
        "manifest_seal_sha256": inputs["manifest_document"]["seal_sha256"],  # type: ignore[index]
        "admission_pointer_seal_sha256": admission_document["seal_sha256"],
        "admission_receipt_seal_sha256": admission_receipt_document["seal_sha256"],
        "source_audit_seal_sha256": "a" * 64,
    }


def _packed_abi(tensor_name: str, shape: list[int]) -> dict[str, object]:
    return {
        "tensor_name": tensor_name,
        "shape": shape,
        "group_size": launcher.GROUP_SIZE,
        "packed_format": launcher.DIRECT_PACKED_FORMAT,
        "direct_packed_only": True,
        "bf16_shadow_allowed": False,
    }


def _terminal_abi() -> dict[str, object]:
    return {
        "ordered_stages": list(launcher.EXPECTED_STAGE_ORDER),
        "final_norm": _packed_abi("model.norm.weight", [launcher.HIDDEN])
        | {"rms_epsilon_bits": launcher.RMS_EPSILON_BITS},
        "lm_head": _packed_abi("lm_head.weight", [launcher.LM_HEAD_ROWS, launcher.HIDDEN])
        | {
            "all_rows_required": launcher.LM_HEAD_ROWS,
            "selected_row_shortcut_allowed": False,
        },
        "tail_mask": {
            "first_reserved_id": launcher.FIRST_RESERVED_ID,
            "last_reserved_id": launcher.LAST_RESERVED_ID,
            "reserved_tail_rows": launcher.RESERVED_TAIL_ROWS,
            "mask_value": "negative_infinity",
            "must_run_after_all_row_lm_head": True,
        },
        "deterministic_sample_feedback": {
            "policy": launcher.DETERMINISTIC_SAMPLER,
            "sample_must_follow_tail_mask": True,
            "sampled_token_must_be_tokenizer_addressable": True,
            "feedback_must_equal_sample": True,
            "feedback_must_be_validated_before_next_embedding_or_state_step": True,
        },
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
    admission_document, _ = _sealed(
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
    inputs: dict[str, object] = {
        "manifest": manifest,
        "manifest_document": manifest_document,
        "manifest_evidence": manifest_evidence,
        "admission": admission,
        "admission_document": admission_document,
        "admission_receipt_document": admission_receipt_document,
    }
    source_binding = _source_binding(inputs)
    post48 = tmp_path / "future-post48-hidden-authority.json"
    post48_document, post48_evidence = _sealed(
        post48,
        {
            "schema": launcher.POST48_HIDDEN_SCHEMA,
            "status": launcher.POST48_HIDDEN_STATUS,
            "source_binding": source_binding,
            "buffer_id_sha256": "b" * 64,
            "command_graph_capture_id_sha256": "c" * 64,
            "all_layer_hidden_sha256": "d" * 64,
            "device_parity_receipt_seal_sha256": "e" * 64,
            "source_token_or_feedback_provenance_sha256": "f" * 64,
            "shape": [launcher.HIDDEN],
            "byte_length": launcher.POST48_HIDDEN_BYTES,
            "produced_by_exact_48_layer_schedule": True,
            "all_48_layers_physically_completed": True,
            "source_bound": True,
            "artifact_bound": True,
            "synthetic_or_component_fixture": False,
            "fallback_used": False,
            "buffer_owned_by_logical_session": True,
            "retained_until_terminal_feedback_fence": True,
            "receipt_written_last_is_completion_marker": True,
            "token_or_generation_claim": False,
        },
    )
    baseline = tmp_path / "sealed-terminal-baseline.json"
    baseline_document, baseline_evidence = _sealed(
        baseline,
        {
            "schema": launcher.BASELINE_SCHEMA,
            "status": launcher.BASELINE_STATUS,
            "integrity_verified": True,
            "source_binding": source_binding,
            "terminal_head_cpu_receipt": {
                "schema": launcher.TERMINAL_CPU_RECEIPT_SCHEMA,
                "status": launcher.TERMINAL_CPU_RECEIPT_STATUS,
                "document_sha256": "1" * 64,
                "unsealed_preimage_sha256": "2" * 64,
                "full_row_cpu_oracle": True,
                "final_norm_abi": _packed_abi("model.norm.weight", [launcher.HIDDEN]),
                "lm_head_abi": _packed_abi("lm_head.weight", [launcher.LM_HEAD_ROWS, launcher.HIDDEN]),
                "rms_epsilon_bits": launcher.RMS_EPSILON_BITS,
                "all_lm_head_rows": launcher.LM_HEAD_ROWS,
                "tokenizer_addressable_rows": launcher.TOKENIZER_VOCAB,
                "first_reserved_id": launcher.FIRST_RESERVED_ID,
                "last_reserved_id": launcher.LAST_RESERVED_ID,
                "reserved_tail_rows": launcher.RESERVED_TAIL_ROWS,
            },
            "tokenizer_sampler_receipt": {
                "schema": launcher.TOKENIZER_RECEIPT_SCHEMA,
                "status": launcher.TOKENIZER_RECEIPT_STATUS,
                "document_sha256": "3" * 64,
                "unsealed_preimage_sha256": "4" * 64,
                "tail_mask_before_sampler": True,
                "tokenizer_feedback_validation_required": True,
            },
        },
    )
    contract = tmp_path / "sealed-terminal-contract.json"
    contract_document, contract_evidence = _sealed(
        contract,
        {
            "schema": launcher.TERMINAL_CONTRACT_SCHEMA,
            "status": launcher.TERMINAL_CONTRACT_STATUS,
            "component_contract_only": True,
            "source_binding": source_binding,
            "sealed_terminal_baseline_binding": _binding(
                baseline_evidence,
                schema=launcher.BASELINE_SCHEMA,
                status=launcher.BASELINE_STATUS,
                seal_sha256=str(baseline_document["seal_sha256"]),
            ),
            "terminal_head_abi": _terminal_abi(),
            "claim_boundary": {
                "artifact_scan_or_payload_open_performed": False,
                "metal_context_or_dispatch_performed": False,
                "model_runtime_or_server_started": False,
                "hcli_execution_performed": False,
                "tps_or_tg_measurement_performed": False,
                "token_or_generation_claim": False,
            },
        },
    )
    lease = tmp_path / "read-only-fixture-terminal-quiet-lease.json"
    lease_document, lease_evidence = _sealed(
        lease,
        {
            "schema": launcher.LEASE_SCHEMA,
            "status": launcher.LEASE_STATUS,
            "lease_id": "fixture-future-q80-terminal-head-component",
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "automatic_retry_prohibited": True,
                "outer_reaped_capture_required": True,
                "receipt_written_last_required": True,
                "prior_terminal_receipt": None,
            },
            "execution_policy": {
                "component": launcher.LEASE_COMPONENT,
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
                "hcli_or_server_allowed": False,
                "cpu_or_bf16_fallback_allowed": False,
                "selected_row_lm_head_allowed": False,
            },
            "artifact_binding": {
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_receipt_seal_sha256": admission_receipt_document["seal_sha256"],
            },
            "post48_hidden_authority_binding": _binding(
                post48_evidence,
                schema=launcher.POST48_HIDDEN_SCHEMA,
                status=launcher.POST48_HIDDEN_STATUS,
                seal_sha256=str(post48_document["seal_sha256"]),
            ),
            "sealed_terminal_baseline_binding": _binding(
                baseline_evidence,
                schema=launcher.BASELINE_SCHEMA,
                status=launcher.BASELINE_STATUS,
                seal_sha256=str(baseline_document["seal_sha256"]),
            ),
            "terminal_head_contract_binding": _binding(
                contract_evidence,
                schema=launcher.TERMINAL_CONTRACT_SCHEMA,
                status=launcher.TERMINAL_CONTRACT_STATUS,
                seal_sha256=str(contract_document["seal_sha256"]),
            ),
        },
    )
    inputs.update(
        {
            "post48": post48,
            "post48_document": post48_document,
            "post48_evidence": post48_evidence,
            "baseline": baseline,
            "baseline_document": baseline_document,
            "baseline_evidence": baseline_evidence,
            "contract": contract,
            "contract_document": contract_document,
            "contract_evidence": contract_evidence,
            "lease": lease,
            "lease_document": lease_document,
            "lease_evidence": lease_evidence,
        }
    )
    return inputs


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
        post48_hidden_authority=inputs["post48"],  # type: ignore[arg-type]
        sealed_terminal_baseline=inputs["baseline"],  # type: ignore[arg-type]
        terminal_head_contract=inputs["contract"],  # type: ignore[arg-type]
        lease_receipt=inputs["lease"],  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        timeout_seconds=10.0,
    )


def _inner_receipt(config: launcher.LaunchConfig, inputs: dict[str, object]) -> dict[str, object]:
    post48_evidence = inputs["post48_evidence"]
    baseline_evidence = inputs["baseline_evidence"]
    contract_evidence = inputs["contract_evidence"]
    lease_evidence = inputs["lease_evidence"]
    assert isinstance(post48_evidence, dict)
    assert isinstance(baseline_evidence, dict)
    assert isinstance(contract_evidence, dict)
    assert isinstance(lease_evidence, dict)
    post48_document = inputs["post48_document"]
    baseline_document = inputs["baseline_document"]
    contract_document = inputs["contract_document"]
    lease_document = inputs["lease_document"]
    assert isinstance(post48_document, dict)
    assert isinstance(baseline_document, dict)
    assert isinstance(contract_document, dict)
    assert isinstance(lease_document, dict)
    source_binding = _source_binding(inputs)
    return seal(
        {
            "schema": launcher.EXPECTED_INNER_SCHEMA,
            "status": launcher.EXPECTED_INNER_STATUS,
            "component_only": True,
            "complete_layer_or_token_performed": False,
            "decoder_or_generation_performed": False,
            "hcli_execution_performed": False,
            "tps_or_tg_measurement_performed": False,
            "metal_device_or_dispatch_performed": True,
            "durable_capture": {
                "receipt_written_last_is_completion_marker": True,
                "post48_input_and_all_terminal_readbacks_written_before_receipt": True,
                "outer_reaped_capture_required": True,
                "replay_guarded": True,
            },
            "source_binding": source_binding,
            "post48_hidden_authority_binding": _binding(
                post48_evidence,
                schema=launcher.POST48_HIDDEN_SCHEMA,
                status=launcher.POST48_HIDDEN_STATUS,
                seal_sha256=str(post48_document["seal_sha256"]),
            ),
            "sealed_terminal_baseline_binding": _binding(
                baseline_evidence,
                schema=launcher.BASELINE_SCHEMA,
                status=launcher.BASELINE_STATUS,
                seal_sha256=str(baseline_document["seal_sha256"]),
            ),
            "terminal_head_contract_binding": _binding(
                contract_evidence,
                schema=launcher.TERMINAL_CONTRACT_SCHEMA,
                status=launcher.TERMINAL_CONTRACT_STATUS,
                seal_sha256=str(contract_document["seal_sha256"]),
            ),
            "lease_binding": _binding(
                lease_evidence,
                schema=launcher.LEASE_SCHEMA,
                status=launcher.LEASE_STATUS,
                seal_sha256=str(lease_document["seal_sha256"]),
            ),
            "terminal_head_execution": {
                "ordered_stages": list(launcher.EXPECTED_STAGE_ORDER),
                "backend": "metal",
                "actual_device_execution": True,
                "device_dispatches": 5,
                "final_fence_before_capture_receipt": True,
                "fixture_only": False,
                "fallback_used": False,
                "selected_row_shortcut_used": False,
                "raw_logits_sha256": "8" * 64,
            },
            "readback_parity": {
                "post48_hidden": {
                    "elements": launcher.HIDDEN,
                    "source_device_parity_passed": True,
                    "all_finite": True,
                    "f32le_sha256": "d" * 64,
                },
                "final_norm": {
                    "elements": launcher.HIDDEN,
                    "source_device_parity_passed": True,
                    "all_finite": True,
                    "f32le_sha256": "9" * 64,
                },
                "lm_head_all_rows": {
                    "rows_evaluated": launcher.LM_HEAD_ROWS,
                    "all_rows_evaluated": launcher.LM_HEAD_ROWS,
                    "full_row_cpu_device_parity_passed": True,
                    "all_logits_finite_before_mask": True,
                    "selected_row_shortcut_used": False,
                    "raw_logits_sha256": "8" * 64,
                },
                "reserved_tail_mask": {
                    "first_reserved_id": launcher.FIRST_RESERVED_ID,
                    "last_reserved_id": launcher.LAST_RESERVED_ID,
                    "reserved_tail_rows": launcher.RESERVED_TAIL_ROWS,
                    "every_reserved_logit_negative_infinity": True,
                    "mask_applied_after_all_row_lm_head": True,
                },
                "deterministic_sample_feedback": {
                    "policy": launcher.DETERMINISTIC_SAMPLER,
                    "sampled_token_id": 42,
                    "feedback_token_id": 42,
                    "sampled_token_is_tokenizer_addressable": True,
                    "sample_after_tail_mask": True,
                    "feedback_matches_sample": True,
                    "feedback_validated_before_next_embedding_or_state_step": True,
                    "sample_feedback_is_component_proof_not_token_claim": True,
                },
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


def test_current_partial_evidence_is_hard_refused_before_child_or_capture(tmp_path: Path) -> None:
    refusal = launcher.assess_current_partial_evidence()
    assert refusal["status"] == launcher.REFUSED_CURRENT_PARTIAL_STATUS
    assert refusal["future_child_launch_eligible"] is False
    assert refusal["claim_boundary"]["no_metal_or_gpu_touched"] is True  # type: ignore[index]

    inputs = _inputs(tmp_path)
    partial = tmp_path / "current-partial-terminal-evidence.json"
    _write_json(
        partial,
        {
            "schema": "hawking.ascension.qwen80_direct_packed_terminal_head_cpu.v1",
            "status": "CURRENT_CPU_COMPONENT_ONLY_PARTIAL",
        },
    )
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)
    config = launcher.LaunchConfig(**{**config.__dict__, "post48_hidden_authority": partial})

    with pytest.raises(launcher.TerminalHeadComponentLauncherError, match="post48-hidden-authority"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_reaps_one_full_row_future_child_and_replays_once(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(_inner_receipt(config, inputs))}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    lease = inputs["lease"]
    assert isinstance(lease, Path)
    lease_before = lease.read_bytes()

    first = launcher.run_attempt(config)
    replay = launcher.run_attempt(config)

    assert first["status"] == launcher.CAPTURED_STATUS
    assert first["child"]["terminal"]["reaped"] is True  # type: ignore[index]
    assert first["inner_probe_capture"]["binding_valid"] is True  # type: ignore[index]
    assert first["one_shot"]["terminal_receipt_written_last"] is True  # type: ignore[index]
    assert first["claim_boundary"]["outer_controller_is_cpu_only"] is True  # type: ignore[index]
    assert first["claim_boundary"]["outer_did_not_issue_or_mutate_a_lease"] is True  # type: ignore[index]
    assert "--post48-hidden-authority" in first["child"]["command"]  # type: ignore[index]
    assert "--terminal-head-contract" in first["child"]["command"]  # type: ignore[index]
    assert marker.read_text(encoding="utf-8") == "run"
    assert replay == first
    assert lease.read_bytes() == lease_before
    verify(first)
    assert json.loads((config.capture_dir / launcher.TERMINAL_FILENAME).read_text(encoding="utf-8")) == first


def test_zero_exit_without_full_151936_row_parity_is_retained_as_refusal(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    inner = _inner_receipt(config, inputs)
    readback = dict(inner["readback_parity"])
    all_rows = dict(readback["lm_head_all_rows"])
    all_rows["rows_evaluated"] = launcher.LM_HEAD_ROWS - 1
    readback["lm_head_all_rows"] = all_rows
    inner = seal({key: value for key, value in inner.items() if key != "seal_sha256"} | {"readback_parity": readback})
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(inner)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("OUTER_ZERO_EXIT_WITHOUT_FULL_ROW_PROOF")
    assert receipt["inner_probe_capture"]["binding_valid"] is False  # type: ignore[index]
    assert "151936-row" in receipt["inner_probe_capture"]["binding_error"]  # type: ignore[index]
    assert marker.read_text(encoding="utf-8") == "run"
    verify(receipt)


def test_unsealed_terminal_contract_refuses_before_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)
    contract = inputs["contract"]
    assert isinstance(contract, Path)
    drifted = json.loads(contract.read_text(encoding="utf-8"))
    drifted["component_contract_only"] = False
    _write_json(contract, drifted)

    with pytest.raises(launcher.TerminalHeadComponentLauncherError, match="terminal-head-contract is not sealed"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()
