"""Focused non-GPU tests for the HQ30GR2 sparse gate/up outer launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
import textwrap
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_sparse_gate_up_device_parity_launcher as launcher
from lab.receipts import seal, verify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _ref(evidence: dict[str, object], seal_sha256: str | None = None) -> dict[str, object]:
    row = {"path": evidence["path"], "document_sha256": evidence["sha256"]}
    if seal_sha256 is not None:
        row["seal_sha256"] = seal_sha256
    return row


def _write_json(path: Path, document: dict[str, object]) -> dict[str, object]:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _write_sealed(path: Path, body: dict[str, object]) -> dict[str, object]:
    return _write_json(path, seal(body))


def _token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(b"".join(value.to_bytes(4, "little") for value in tokens)).hexdigest()


def _probe(tmp_path: Path, *, cpu_behavior: str, device_behavior: str) -> tuple[Path, Path]:
    """Create a fake child; it has no model, Metal, or GPU operations."""

    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    script = f'''#!/usr/bin/env python3
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

MARKER = Path({str(marker)!r})
CPU_BEHAVIOR = {cpu_behavior!r}
DEVICE_BEHAVIOR = {device_behavior!r}

def value(flag):
    args = sys.argv[1:]
    return args[args.index(flag) + 1]

mode = value("--mode")
MARKER.open("a", encoding="utf-8").write("c" if mode == "cpu-oracle" else "d")
behavior = CPU_BEHAVIOR if mode == "cpu-oracle" else DEVICE_BEHAVIOR
print("fake child stdout " + mode)
print("fake child stderr " + mode, file=sys.stderr)
if behavior == "nonzero":
    raise SystemExit(7)
if behavior == "signal":
    os.kill(os.getpid(), signal.SIGTERM)
    raise SystemExit(99)
if behavior != "success":
    raise SystemExit(98)

output = Path(value("--output-dir"))
output.mkdir(parents=True, exist_ok=True)
runtime_sha = hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest()
binding = {{
    "candidate_manifest_path": value("--manifest"),
    "candidate_manifest_seal_sha256": value("--expected-manifest-seal-sha256"),
    "source_audit_seal_sha256": value("--expected-source-audit-seal-sha256"),
    "source_revision": value("--expected-source-revision"),
    "revalidation": {{"path": value("--expected-revalidation-path"), "seal_sha256": value("--expected-revalidation-seal-sha256")}},
    "selection": {{"path": value("--expected-selection-path"), "seal_sha256": value("--expected-selection-seal-sha256")}},
    "source_snapshot": {{"path": value("--expected-source-snapshot-path"), "seal_sha256": value("--expected-source-snapshot-seal-sha256")}},
    "terminal": {{"path": value("--expected-terminal-path"), "seal_sha256": value("--expected-terminal-seal-sha256")}},
    "input_f32le": {{"path": value("--input-f32le"), "sha256": value("--expected-input-sha256"), "values": 2048}},
    "runtime_executable_sha256": runtime_sha,
    "fixed_topology": {{
        "selected_sparse_pair": "L0/E0 gate+up HQ30GR2",
        "kernel": "qwen30_quality_repack_sparse_gate_up_swiglu",
        "no_direct_fallback_for_sparse_pair": True,
        "no_bf16_or_dense_weight_path": True,
    }},
}}
catalog = {{
    "verified_payload_count": 18867,
    "direct_tensor_count": 18865,
    "sparse_residual_tensor_count": 2,
    "sparse_gate_up_dispatch": {{
        "kernel_name": "qwen30_quality_repack_sparse_gate_up_swiglu",
        "rows": 768,
        "cols": 2048,
        "group_size": 128,
        "gate_residual_count": 3933,
        "up_residual_count": 3933,
        "exact_non_fma_scalar_order_required": True,
        "direct_fallback_for_sparse_residual_forbidden": True,
    }},
}}
boundary = {{
    "not_a_complete_layer_or_full_token": True,
    "no_logits_sampler_generation_hcli_or_server": True,
    "not_coherence_tps_tg_capability_manager_or_tournament": True,
}}
cpu_bytes = b"\\0" * 6144
cpu_sha = hashlib.sha256(cpu_bytes).hexdigest()
result = {{
    "schema": "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity.v1",
    "mode": mode,
    "binding": binding,
    "typed_catalog": catalog,
    "cpu_oracle": {{
        "activation_f64le_sha256": cpu_sha,
        "activation_values": 768,
        "admission_snapshot_only": True,
        "raw_bf16_or_dense_weight_path": False,
    }},
    "claim_boundary": boundary,
}}
if mode == "cpu-oracle":
    cpu_path = output / "cpu-activation.f64le"
    cpu_path.write_bytes(cpu_bytes)
    result.update({{
        "status": "EARNED_HQ30GR2_CPU_FORMAT_ORACLE_NOT_DEVICE_OR_RUNTIME",
        "cpu_oracle_output": {{"path": str(cpu_path), "sha256": cpu_sha, "values": 768}},
    }})
else:
    device_bytes = b"\\0" * 3072
    device_path = output / "device-activation.f32le"
    device_path.write_bytes(device_bytes)
    result.update({{
        "status": "EARNED_HQ30GR2_SPARSE_GATE_UP_CPU_DEVICE_PARITY_NOT_LAYER_OR_RUNTIME",
        "protected_cpu_oracle": {{
            "path": value("--cpu-activation-f64le"),
            "sha256": value("--expected-cpu-activation-sha256"),
            "recomputed_current_sha256": cpu_sha,
        }},
        "device_parity": {{"passes": True, "values_compared": 768, "max_abs_error": 0.0, "max_rel_error": 0.0}},
        "device_output": {{"path": str(device_path), "sha256": hashlib.sha256(device_bytes).hexdigest(), "values": 768}},
        "device_execution": {{
            "metal_context_created": True,
            "kernel": "qwen30_quality_repack_sparse_gate_up_swiglu",
            "only_selected_l0_e0_gate_up_swiglu_pair_executed": True,
            "all_layers_executed": False,
            "full_token_executed": False,
        }},
    }})
(output / "result.json").write_text(json.dumps(result, sort_keys=True) + "\\n", encoding="utf-8")
'''
    probe.write_text(textwrap.dedent(script), encoding="utf-8")
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpu_behavior: str = "success",
    device_behavior: str = "success",
) -> dict[str, Path | str]:
    probe, marker = _probe(tmp_path, cpu_behavior=cpu_behavior, device_behavior=device_behavior)
    source_audit = "a" * 64
    source_revision = "fixture-source-revision"

    revalidation = tmp_path / "source-revalidation.json"
    revalidation_doc = _write_sealed(
        revalidation,
        {
            "schema": launcher.REVALIDATION_SCHEMA,
            "status": launcher.REVALIDATION_STATUS,
            "source_revision": source_revision,
            "source_audit_seal_sha256": source_audit,
        },
    )
    revalidation_evidence = _evidence(revalidation)

    terminal = tmp_path / "candidate-terminal.json"
    terminal_doc = _write_sealed(
        terminal,
        {"schema": launcher.CANDIDATE_TERMINAL_SCHEMA, "status": launcher.CANDIDATE_TERMINAL_STATUS},
    )
    terminal_evidence = _evidence(terminal)

    common_binding = {
        "selected_organs": list(launcher.EXPECTED_SELECTED_ORGANS),
        "immutable_source_revalidation": _ref(revalidation_evidence, str(revalidation_doc["seal_sha256"]))
        | {"source_revision": source_revision},
        "source_audit": {"seal_sha256": source_audit},
    }
    selection = tmp_path / "selection.json"
    selection_doc = _write_sealed(
        selection,
        {
            "schema": launcher.SELECTION_SCHEMA,
            "status": launcher.SELECTION_STATUS,
            "binding": common_binding,
        },
    )
    selection_evidence = _evidence(selection)
    snapshot = tmp_path / "source-snapshot.json"
    snapshot_doc = _write_sealed(
        snapshot,
        {
            "schema": launcher.SNAPSHOT_SCHEMA,
            "status": launcher.SNAPSHOT_STATUS,
            "binding": common_binding,
        },
    )
    snapshot_evidence = _evidence(snapshot)

    candidate = tmp_path / "candidate-manifest.json"
    candidate_doc = _write_sealed(
        candidate,
        {
            "schema": launcher.CANDIDATE_SCHEMA,
            "status": launcher.CANDIDATE_STATUS,
            "source_revalidation_receipt_path": str(revalidation.resolve()),
            "source_revalidation_receipt_seal_sha256": revalidation_doc["seal_sha256"],
            "source_body_audit_seal_sha256": source_audit,
            "representation": {"selected_organs": list(launcher.EXPECTED_SELECTED_ORGANS)},
            "quality_repack_branch": {
                "changed_organs": list(launcher.EXPECTED_SELECTED_ORGANS),
                "selection_receipt": _ref(selection_evidence, str(selection_doc["seal_sha256"])),
                "source_binding_snapshot": _ref(snapshot_evidence, str(snapshot_doc["seal_sha256"])),
            },
        },
    )
    candidate_evidence = _evidence(candidate)

    admission = tmp_path / "admission-receipt.json"
    admission_doc = _write_sealed(
        admission,
        {
            "schema": launcher.ADMISSION_SCHEMA,
            "status": launcher.ADMISSION_STATUS,
            "complete_manifest": _ref(candidate_evidence, str(candidate_doc["seal_sha256"])),
            "selection_receipt": _ref(selection_evidence, str(selection_doc["seal_sha256"])),
            "source_binding_snapshot": _ref(snapshot_evidence, str(snapshot_doc["seal_sha256"])),
            "immutable_source_revalidation": _ref(revalidation_evidence, str(revalidation_doc["seal_sha256"])),
            "terminal": _ref(terminal_evidence, str(terminal_doc["seal_sha256"])),
        },
    )
    admission_evidence = _evidence(admission)
    admission_current = tmp_path / "admission-current.json"
    admission_current_doc = _write_sealed(
        admission_current,
        {
            "schema": launcher.ADMISSION_CURRENT_SCHEMA,
            "status": launcher.ADMISSION_CURRENT_STATUS,
            "complete_manifest": _ref(candidate_evidence, str(candidate_doc["seal_sha256"]))
            | {"schema": launcher.CANDIDATE_SCHEMA, "status": launcher.CANDIDATE_STATUS},
            "admission_receipt": _ref(admission_evidence, str(admission_doc["seal_sha256"])),
        },
    )
    admission_current_evidence = _evidence(admission_current)

    run_root = tmp_path / "compiler-run"
    annotated = run_root / "probes/literal_hawking/compiler-pre-execution.annotated.json"
    annotated.parent.mkdir(parents=True)
    tokens = list(range(launcher.TARGET_TOKEN_COUNT))
    _write_json(
        annotated,
        {
            "schema": launcher.ANNOTATED_TRACE_SCHEMA,
            "status": launcher.ANNOTATED_TRACE_STATUS,
            "compiler_trace": {
                "status": launcher.ANNOTATED_TRACE_STATUS,
                "model_execution_started": False,
                "capture_timing": "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION",
            },
            "source_tokenizer_annotations": {
                "source_one_user_native_prompt": {
                    "token_ids": tokens,
                    "token_ids_u32le_sha256": _token_hash(tokens),
                    "add_special_tokens": True,
                }
            },
        },
    )
    annotated_evidence = _evidence(annotated)
    compiler_receipt = tmp_path / "compiler-receipt.json"
    compiler_doc = _write_sealed(
        compiler_receipt,
        {
            "schema": launcher.COMPILER_SCHEMA,
            "status": launcher.COMPILER_STATUS,
            "binding": {
                "candidate_manifest": _ref(candidate_evidence, str(candidate_doc["seal_sha256"])),
                "candidate_native_admission": {
                    "current_pointer_path": str(admission_current.resolve()),
                    "current_pointer_seal_sha256": admission_current_doc["seal_sha256"],
                },
                "selection_receipt": _ref(selection_evidence, str(selection_doc["seal_sha256"])),
                "source_snapshot": _ref(snapshot_evidence, str(snapshot_doc["seal_sha256"]))
                | {"immutable_source_revalidation": _ref(revalidation_evidence, str(revalidation_doc["seal_sha256"])) | {"source_revision": source_revision}},
                "run_root": str(run_root.resolve()),
            },
            "public_probe_compiler_traces": [
                {
                    "probe_id": launcher.TARGET_PROBE,
                    "annotated_trace_path": "probes/literal_hawking/compiler-pre-execution.annotated.json",
                    "annotated_trace_sha256": annotated_evidence["sha256"],
                }
            ],
        },
    )
    compiler_evidence = _evidence(compiler_receipt)
    compiler_current = tmp_path / "compiler-current.json"
    compiler_current_doc = _write_sealed(
        compiler_current,
        {
            "schema": launcher.COMPILER_CURRENT_SCHEMA,
            "status": launcher.COMPILER_CURRENT_STATUS,
            "compiler_trace_receipt": _ref(compiler_evidence, str(compiler_doc["seal_sha256"])),
        },
    )
    compiler_current_evidence = _evidence(compiler_current)

    output_root = tmp_path / "route-output"
    hidden = output_root / "hidden/literal_hawking/000337.f32le"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"\0" * launcher.INPUT_BYTES)
    hidden_evidence = _evidence(hidden)
    result = output_root / "capture-result.json"
    steps: list[dict[str, object]] = [{} for _ in range(launcher.TARGET_TOKEN_COUNT)]
    steps[launcher.TARGET_POSITION] = {
        "position": launcher.TARGET_POSITION,
        "selected_expert_ids": [4, 0, 7],
        "router_input_hidden_f32le": {
            "relative_path": "hidden/literal_hawking/000337.f32le",
            "sha256": hidden_evidence["sha256"],
            "bytes": launcher.INPUT_BYTES,
            "elements": launcher.INPUT_VALUES,
        },
    }
    _write_json(
        result,
        {
            "status": launcher.ROUTE_STATUS,
            "capture_protocol_revision": "l0-route-hidden-capture-output-parent-v2",
            "probes": [{"probe_id": launcher.TARGET_PROBE, "steps": steps}],
        },
    )
    result_evidence = _evidence(result)
    route_receipt = tmp_path / "route-receipt.json"
    route_doc = _write_sealed(
        route_receipt,
        {
            "schema": launcher.ROUTE_SCHEMA,
            "status": launcher.ROUTE_STATUS,
            "binding": {
                "compiler_trace": _ref(compiler_evidence, str(compiler_doc["seal_sha256"])),
                "candidate_selection": _ref(selection_evidence, str(selection_doc["seal_sha256"])),
                "capture_result_path": str(result.resolve()),
                "capture_result_sha256": result_evidence["sha256"],
                "capture_output_root": str(output_root.resolve()),
            },
            "probe_summary": [
                {
                    "probe_id": launcher.TARGET_PROBE,
                    "source_template_token_count": launcher.TARGET_TOKEN_COUNT,
                    "route_membership_and_hidden_steps": launcher.TARGET_TOKEN_COUNT,
                    "l0_expert0_selected_positions": [launcher.TARGET_POSITION],
                    "l0_expert0_selected_position_count": 1,
                    "hidden_payloads": [
                        {"relative_path": "hidden/literal_hawking/000337.f32le", "sha256": hidden_evidence["sha256"]}
                    ],
                }
            ],
        },
    )
    route_evidence = _evidence(route_receipt)
    route_current = tmp_path / "route-current.json"
    route_current_doc = _write_sealed(
        route_current,
        {
            "schema": launcher.ROUTE_CURRENT_SCHEMA,
            "status": launcher.ROUTE_CURRENT_STATUS,
            "route_capture_receipt": _ref(route_evidence, str(route_doc["seal_sha256"])),
        },
    )
    route_current_evidence = _evidence(route_current)

    probe_evidence = _evidence(probe)
    cpu_command = [
        str(probe.resolve()), "--mode", launcher.MODE_CPU,
        "--manifest", str(candidate.resolve()), "--expected-manifest-seal-sha256", str(candidate_doc["seal_sha256"]),
        "--expected-source-audit-seal-sha256", source_audit,
        "--expected-source-revision", source_revision,
        "--expected-revalidation-path", str(revalidation.resolve()), "--expected-revalidation-seal-sha256", str(revalidation_doc["seal_sha256"]),
        "--expected-selection-path", str(selection.resolve()), "--expected-selection-seal-sha256", str(selection_doc["seal_sha256"]),
        "--expected-source-snapshot-path", str(snapshot.resolve()), "--expected-source-snapshot-seal-sha256", str(snapshot_doc["seal_sha256"]),
        "--expected-terminal-path", str(terminal.resolve()), "--expected-terminal-seal-sha256", str(terminal_doc["seal_sha256"]),
        "--input-f32le", str(hidden.resolve()), "--expected-input-sha256", str(hidden_evidence["sha256"]),
        "--max-seq-len", "512",
    ]
    preparation = tmp_path / "preparation-receipt.json"
    preparation_doc = _write_sealed(
        preparation,
        {
            "schema": launcher.PREPARATION_SCHEMA,
            "status": launcher.PREPARATION_STATUS,
            "binding": {
                "probe_binary": probe_evidence,
                "candidate_manifest": candidate_evidence,
                "candidate_manifest_seal_sha256": candidate_doc["seal_sha256"],
                "candidate_admission_current": _ref(admission_current_evidence, str(admission_current_doc["seal_sha256"])),
                "candidate_admission_receipt": _ref(admission_evidence, str(admission_doc["seal_sha256"])),
                "route_capture_current": _ref(route_current_evidence, str(route_current_doc["seal_sha256"])),
                "route_capture_receipt": _ref(route_evidence, str(route_doc["seal_sha256"])),
                "component_input": {
                    "device_produced_router_input_f32le": {
                        "path": str(hidden.resolve()), "sha256": hidden_evidence["sha256"],
                        "bytes": launcher.INPUT_BYTES, "elements": launcher.INPUT_VALUES,
                    },
                    "probe_id": launcher.TARGET_PROBE,
                    "l0_e0_selected_position": launcher.TARGET_POSITION,
                    "source_template_token_count": launcher.TARGET_TOKEN_COUNT,
                },
            },
            "cpu_oracle_invocation": {
                "command_without_output_dir": cpu_command,
                "mode": launcher.MODE_CPU,
                "metal_context_or_dispatch_performed": False,
                "outer_controller_must_create_fresh_output_dir": True,
            },
            "claim_boundary": {
                "preparation_only": True,
                "no_metal_context_no_model_forward_no_server_or_watcher_change": True,
                "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": True,
            },
            "future_device_parity_contract": {
                "mode": launcher.MODE_DEVICE,
                "requires_cpu_oracle_activation_f64le_from_this_exact_binding": True,
                "requires_fresh_explicit_quiet_gpu_lease": True,
                "automatic_retry_forbidden": True,
                "one_device_process_one_lease_one_terminal_receipt_or_refusal": True,
            },
        },
    )
    preparation_evidence = _evidence(preparation)
    preparation_current = tmp_path / "preparation-current.json"
    preparation_current_doc = _write_sealed(
        preparation_current,
        {
            "schema": launcher.PREPARATION_CURRENT_SCHEMA,
            "status": launcher.PREPARATION_CURRENT_STATUS,
            "component_preparation_receipt": _ref(preparation_evidence, str(preparation_doc["seal_sha256"])),
        },
    )

    monkeypatch.setattr(launcher, "EXPECTED_PROBE_BINARY_SHA256", str(probe_evidence["sha256"]))
    monkeypatch.setattr(launcher, "PINNED_CANDIDATE_MANIFEST_SEAL", str(candidate_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_ADMISSION_RECEIPT_SEAL", str(admission_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_REVALIDATION_SEAL", str(revalidation_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_SELECTION_SEAL", str(selection_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_SOURCE_SNAPSHOT_SEAL", str(snapshot_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_COMPILER_TRACE_SEAL", str(compiler_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_ROUTE_CAPTURE_SEAL", str(route_doc["seal_sha256"]))
    monkeypatch.setattr(launcher, "PINNED_PREPARATION_RECEIPT_SEAL", str(preparation_doc["seal_sha256"]))
    return {
        "probe": probe,
        "marker": marker,
        "candidate": candidate,
        "admission_current": admission_current,
        "revalidation": revalidation,
        "selection": selection,
        "snapshot": snapshot,
        "compiler_current": compiler_current,
        "route_current": route_current,
        "preparation_current": preparation_current,
        "preparation_receipt": preparation,
        "preparation_current_seal": str(preparation_current_doc["seal_sha256"]),
    }


def _config(
    tmp_path: Path,
    inputs: dict[str, Path | str],
    *,
    mode: str,
    capture_name: str,
    cpu_outer: Path | None = None,
    lease: Path | None = None,
) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=inputs["probe"],  # type: ignore[arg-type]
        candidate_manifest=inputs["candidate"],  # type: ignore[arg-type]
        candidate_admission_current=inputs["admission_current"],  # type: ignore[arg-type]
        source_revalidation=inputs["revalidation"],  # type: ignore[arg-type]
        selection_receipt=inputs["selection"],  # type: ignore[arg-type]
        source_snapshot=inputs["snapshot"],  # type: ignore[arg-type]
        compiler_trace_current=inputs["compiler_current"],  # type: ignore[arg-type]
        route_capture_current=inputs["route_current"],  # type: ignore[arg-type]
        preparation_current=inputs["preparation_current"],  # type: ignore[arg-type]
        capture_dir=tmp_path / capture_name,
        mode=mode,
        timeout_seconds=10.0,
        cpu_oracle_outer_receipt=cpu_outer,
        lease_receipt=lease,
    )


def _lease(tmp_path: Path, cpu_receipt: Path) -> Path:
    outer = json.loads(cpu_receipt.read_text(encoding="utf-8"))
    source = outer["source_binding"]
    activation = outer["cpu_oracle_output"]
    lease = tmp_path / "quiet-lease.json"
    _write_sealed(
        lease,
        {
            "schema": launcher.QUIET_LEASE_SCHEMA,
            "status": launcher.QUIET_LEASE_STATUS,
            "one_shot_lifecycle": {
                "fresh_for_this_exact_launch": True,
                "prior_terminal_receipt": None,
                "automatic_retry_allowed": False,
            },
            "execution_policy": {
                "component": launcher.QUIET_LEASE_COMPONENT,
                "quiet_qwen_family_gpu_lease": True,
                "strict_math": True,
                "component_only": True,
                "one_child_process_group_only": True,
                "timing_or_benchmarking_allowed": False,
                "all_layer_or_full_token_allowed": False,
                "logit_or_generation_allowed": False,
                "hcli_or_server_allowed": False,
                "coherence_claim_allowed": False,
                "tps_or_tg_claim_allowed": False,
                "capability_claim_allowed": False,
                "tournament_claim_allowed": False,
            },
            "artifact_binding": {
                key: source[key]
                for key in (
                    "candidate_manifest", "candidate_admission_current", "candidate_admission_receipt",
                    "source_revalidation", "selection_receipt", "source_snapshot", "candidate_terminal",
                )
            },
            "upstream_binding": {
                key: source[key]
                for key in ("compiler_trace_receipt", "route_capture_receipt", "preparation_receipt", "literal_hawking_e0_input_f32le")
            },
            "cpu_oracle_binding": {
                "outer_terminal_receipt": _ref(_evidence(cpu_receipt), str(outer["seal_sha256"])),
                "cpu_activation_f64le": activation,
            },
        },
    )
    return lease


def _successful_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, device_behavior: str) -> tuple[dict[str, Path | str], launcher.LaunchConfig, dict[str, object]]:
    inputs = _inputs(tmp_path, monkeypatch, device_behavior=device_behavior)
    config = _config(tmp_path, inputs, mode=launcher.MODE_CPU, capture_name="cpu-capture")
    receipt = launcher.run_attempt(config)
    assert receipt["status"] == launcher._success_status(launcher.MODE_CPU)
    return inputs, config, receipt


def test_device_requires_completed_cpu_oracle_and_quiet_lease_before_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    config = _config(tmp_path, inputs, mode=launcher.MODE_DEVICE, capture_name="device-missing")

    with pytest.raises(launcher.SparseGateUpParityLauncherError, match="cpu-oracle"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not Path(inputs["marker"]).exists()  # type: ignore[arg-type]


def test_mismatched_lease_is_refused_before_device_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, cpu_config, _ = _successful_cpu(tmp_path, monkeypatch, device_behavior="success")
    cpu_receipt = cpu_config.capture_dir / launcher.TERMINAL_FILENAME
    lease = _lease(tmp_path, cpu_receipt)
    document = json.loads(lease.read_text(encoding="utf-8"))
    document.pop("seal_sha256")
    document["upstream_binding"]["literal_hawking_e0_input_f32le"]["sha256"] = "f" * 64
    _write_sealed(lease, document)
    config = _config(
        tmp_path,
        inputs,
        mode=launcher.MODE_DEVICE,
        capture_name="device-mismatch",
        cpu_outer=cpu_receipt,
        lease=lease,
    )

    with pytest.raises(launcher.SparseGateUpParityLauncherError, match="lease literal_hawking"):
        launcher.run_attempt(config)

    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "c"  # type: ignore[arg-type]
    assert not config.capture_dir.exists()


def test_cpu_oracle_nonzero_child_is_reaped_and_replayed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path, monkeypatch, cpu_behavior="nonzero")
    config = _config(tmp_path, inputs, mode=launcher.MODE_CPU, capture_name="cpu-nonzero")

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_NONZERO")
    assert receipt["child"]["terminal"] == {
        "reaped": True,
        "timed_out": False,
        "returncode": 7,
        "exit_code": 7,
        "signal": None,
    }
    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "c"  # type: ignore[arg-type]
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "fake child stdout cpu-oracle\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "fake child stderr cpu-oracle\n"
    verify(receipt)
    assert launcher.run_attempt(config) == receipt
    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "c"  # type: ignore[arg-type]


def test_device_nonzero_child_is_reaped_and_replayed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, cpu_config, _ = _successful_cpu(tmp_path, monkeypatch, device_behavior="nonzero")
    cpu_receipt = cpu_config.capture_dir / launcher.TERMINAL_FILENAME
    lease = _lease(tmp_path, cpu_receipt)
    config = _config(
        tmp_path,
        inputs,
        mode=launcher.MODE_DEVICE,
        capture_name="device-nonzero",
        cpu_outer=cpu_receipt,
        lease=lease,
    )

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_NONZERO")
    assert receipt["child"]["terminal"]["reaped"] is True
    assert receipt["child"]["terminal"]["exit_code"] == 7
    verify(receipt)
    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "cd"  # type: ignore[arg-type]
    assert launcher.run_attempt(config) == receipt
    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "cd"  # type: ignore[arg-type]


def test_device_signal_child_is_reaped_and_replayed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, cpu_config, _ = _successful_cpu(tmp_path, monkeypatch, device_behavior="signal")
    cpu_receipt = cpu_config.capture_dir / launcher.TERMINAL_FILENAME
    lease = _lease(tmp_path, cpu_receipt)
    config = _config(
        tmp_path,
        inputs,
        mode=launcher.MODE_DEVICE,
        capture_name="device-signal",
        cpu_outer=cpu_receipt,
        lease=lease,
    )

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_SIGNAL")
    assert receipt["child"]["terminal"] == {
        "reaped": True,
        "timed_out": False,
        "returncode": -15,
        "exit_code": None,
        "signal": 15,
    }
    verify(receipt)
    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "cd"  # type: ignore[arg-type]
    assert launcher.run_attempt(config) == receipt
    assert Path(inputs["marker"]).read_text(encoding="utf-8") == "cd"  # type: ignore[arg-type]


def test_current_production_binary_pin_is_explicit() -> None:
    assert launcher.EXPECTED_PROBE_BINARY_SHA256 == "42569eecd6d4abaf081ac83e8d98adaa02d62128323bb2ed8c5f1a0471ae82c3"
