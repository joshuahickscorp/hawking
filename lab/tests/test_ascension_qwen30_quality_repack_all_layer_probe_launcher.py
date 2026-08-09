"""Focused non-GPU tests for the prepared Q30 HQ30GR2 outer launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
import sys
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_quality_repack_all_layer_probe_launcher as launcher
from lab.receipts import seal, verify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(b"".join(token.to_bytes(4, "little") for token in tokens)).hexdigest()


def _evidence(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _write_sealed(path: Path, body: dict[str, object]) -> dict[str, object]:
    document = seal(body)
    _write_json(path, document)
    return document


def _rewrite_sealed(path: Path, mutate) -> dict[str, object]:
    current = json.loads(path.read_text(encoding="utf-8"))
    current.pop("seal_sha256")
    mutate(current)
    return _write_sealed(path, current)


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source_revision = "fixture-source-revision"
    source_revalidation = {"seal_sha256": "1" * 64, "source_revision": source_revision}

    candidate = tmp_path / "candidate-manifest.json"
    candidate_doc = _write_sealed(
        candidate,
        {
            "schema": launcher.CANDIDATE_SCHEMA,
            "status": launcher.CANDIDATE_STATUS,
        },
    )
    candidate_evidence = _evidence(candidate)

    admission_receipt = tmp_path / "candidate-admission-receipt.json"
    admission_receipt_doc = _write_sealed(
        admission_receipt,
        {
            "schema": launcher.ADMISSION_RECEIPT_SCHEMA,
            "status": launcher.ADMISSION_RECEIPT_STATUS,
            "complete_manifest": {
                "path": candidate_evidence["path"],
                "document_sha256": candidate_evidence["sha256"],
                "seal_sha256": candidate_doc["seal_sha256"],
            },
        },
    )
    admission_receipt_evidence = _evidence(admission_receipt)

    admission_current = tmp_path / "candidate-admission-current.json"
    admission_current_doc = _write_sealed(
        admission_current,
        {
            "schema": launcher.ADMISSION_CURRENT_SCHEMA,
            "status": launcher.ADMISSION_CURRENT_STATUS,
            "complete_manifest": {
                "path": candidate_evidence["path"],
                "document_sha256": candidate_evidence["sha256"],
                "seal_sha256": candidate_doc["seal_sha256"],
                "schema": launcher.CANDIDATE_SCHEMA,
                "status": launcher.CANDIDATE_STATUS,
            },
            "admission_receipt": {
                "path": admission_receipt_evidence["path"],
                "document_sha256": admission_receipt_evidence["sha256"],
                "seal_sha256": admission_receipt_doc["seal_sha256"],
            },
        },
    )
    admission_current_evidence = _evidence(admission_current)

    control_manifest = tmp_path / "control-manifest.json"
    control_manifest_doc = _write_sealed(
        control_manifest,
        {
            "schema": launcher.CONTROL_MANIFEST_SCHEMA,
            "status": launcher.CONTROL_MANIFEST_STATUS,
        },
    )
    control_manifest_evidence = _evidence(control_manifest)

    run_root = tmp_path / "compiler-run"
    annotated_path = run_root / "probes/literal_hawking/compiler-pre-execution.annotated.json"
    annotated_path.parent.mkdir(parents=True)
    token_ids = list(range(launcher.TARGET_TOKEN_COUNT))
    _write_json(
        annotated_path,
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
                    "token_ids": token_ids,
                    "token_ids_u32le_sha256": _token_hash(token_ids),
                    "add_special_tokens": True,
                },
                "selected_context_spans": [{"text": "fixture persisted compiler span"}],
            },
        },
    )
    annotated_evidence = _evidence(annotated_path)

    compiler_receipt = tmp_path / "compiler-trace-receipt.json"
    compiler_receipt_doc = _write_sealed(
        compiler_receipt,
        {
            "schema": launcher.COMPILER_SCHEMA,
            "status": launcher.COMPILER_STATUS,
            "binding": {
                "candidate_manifest": {
                    "path": candidate_evidence["path"],
                    "document_sha256": candidate_evidence["sha256"],
                    "seal_sha256": candidate_doc["seal_sha256"],
                },
                "candidate_native_admission": {
                    "current_pointer_path": admission_current_evidence["path"],
                    "current_pointer_seal_sha256": admission_current_doc["seal_sha256"],
                },
                "source_snapshot": {"immutable_source_revalidation": source_revalidation},
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
    compiler_receipt_evidence = _evidence(compiler_receipt)
    compiler_current = tmp_path / "compiler-trace-current.json"
    compiler_current_doc = _write_sealed(
        compiler_current,
        {
            "schema": launcher.COMPILER_CURRENT_SCHEMA,
            "status": launcher.COMPILER_CURRENT_STATUS,
            "compiler_trace_receipt": {
                "path": compiler_receipt_evidence["path"],
                "seal_sha256": compiler_receipt_doc["seal_sha256"],
            },
        },
    )

    selected_hidden = {
        "relative_path": "hidden/literal_hawking/000337.f32le",
        "sha256": "2" * 64,
    }
    route_receipt = tmp_path / "route-capture-receipt.json"
    route_receipt_doc = _write_sealed(
        route_receipt,
        {
            "schema": launcher.ROUTE_SCHEMA,
            "status": launcher.ROUTE_STATUS,
            "binding": {
                "compiler_trace": {
                    "path": compiler_receipt_evidence["path"],
                    "document_sha256": compiler_receipt_evidence["sha256"],
                    "seal_sha256": compiler_receipt_doc["seal_sha256"],
                },
                "baseline_direct_packed_control": {
                    "manifest_path": control_manifest_evidence["path"],
                    "manifest_seal_sha256": control_manifest_doc["seal_sha256"],
                    "source_audit_seal_sha256": "3" * 64,
                    "source_revision": source_revision,
                },
            },
            "probe_summary": [
                {
                    "probe_id": launcher.TARGET_PROBE,
                    "source_template_token_count": launcher.TARGET_TOKEN_COUNT,
                    "route_membership_and_hidden_steps": launcher.TARGET_TOKEN_COUNT,
                    "l0_expert0_selected_position_count": 1,
                    "l0_expert0_selected_positions": [launcher.TARGET_POSITION],
                    "hidden_payloads": [selected_hidden],
                }
            ],
        },
    )
    route_receipt_evidence = _evidence(route_receipt)
    route_current = tmp_path / "route-capture-current.json"
    route_current_doc = _write_sealed(
        route_current,
        {
            "schema": launcher.ROUTE_CURRENT_SCHEMA,
            "status": launcher.ROUTE_CURRENT_STATUS,
            "route_capture_receipt": {
                "path": route_receipt_evidence["path"],
                "seal_sha256": route_receipt_doc["seal_sha256"],
            },
        },
    )
    route_current_evidence = _evidence(route_current)

    preparation_receipt = tmp_path / "all-layer-preparation-receipt.json"
    preparation_receipt_doc = _write_sealed(
        preparation_receipt,
        {
            "schema": launcher.PREPARATION_SCHEMA,
            "status": launcher.PREPARATION_STATUS,
            "binding": {
                "candidate_manifest_seal_sha256": candidate_doc["seal_sha256"],
                "candidate_admission_current_pointer": {
                    "path": admission_current_evidence["path"],
                    "seal_sha256": admission_current_doc["seal_sha256"],
                },
                "route_capture_current_pointer": {
                    "path": route_current_evidence["path"],
                    "seal_sha256": route_current_doc["seal_sha256"],
                },
                "route_capture_receipt": {
                    "path": route_receipt_evidence["path"],
                    "document_sha256": route_receipt_evidence["sha256"],
                    "seal_sha256": route_receipt_doc["seal_sha256"],
                },
                "source_readiness": {
                    "live_qwen30_runtime_source": {"candidate_type_absent_from_live_runtime": True},
                    "typed_candidate_diagnostic_source": {
                        "typed_catalog_and_sparse_gate_up_host_dispatch_build_ready": True
                    },
                },
            },
            "planned_bounded_input": {
                "probe_id": launcher.TARGET_PROBE,
                "source_template_token_count": launcher.TARGET_TOKEN_COUNT,
                "l0_e0_selected_position": launcher.TARGET_POSITION,
                "l0_e0_router_input_hidden": selected_hidden,
            },
            "planned_execution_not_run": {
                "one_existing_trace_only": launcher.TARGET_PROBE,
                "prefill": "exact 369 source-template IDs through all 48 layers for baseline and candidate separately",
                "one_bounded_continuation": "derive baseline deterministic argmax after the exact prefix; force that same one token into both paths for one additional 48-layer forward",
                "explicitly_not_run_or_claimed": [
                    "HCLI endpoint",
                    "chat coherence",
                    "TPS, TG, capability, manager, or tournament",
                ],
            },
            "candidate_runtime_contract": {
                "new_runtime_type_required": "Qwen30QualityRepackNativeDiagnosticRuntime",
                "live_control_runtime_reuse_forbidden": "Qwen30CompleteNativeRuntime only accepts CompleteBinaryArtifact/direct headers",
            },
        },
    )
    preparation_receipt_evidence = _evidence(preparation_receipt)
    preparation_current = tmp_path / "all-layer-preparation-current.json"
    preparation_current_doc = _write_sealed(
        preparation_current,
        {
            "schema": launcher.PREPARATION_CURRENT_SCHEMA,
            "status": launcher.PREPARATION_CURRENT_STATUS,
            "preparation_receipt": {
                "path": preparation_receipt_evidence["path"],
                "seal_sha256": preparation_receipt_doc["seal_sha256"],
            },
        },
    )

    control_runtime = tmp_path / "control-runtime-receipt.json"
    control_runtime_doc = _write_sealed(
        control_runtime,
        {
            "schema": launcher.CONTROL_RUNTIME_SCHEMA,
            "status": launcher.CONTROL_RUNTIME_STATUS,
            "binding": {
                "complete_manifest_seal_sha256": control_manifest_doc["seal_sha256"],
                "source_revalidation_seal_sha256": source_revalidation["seal_sha256"],
            },
            "runtime": {
                "all_layers_executed": True,
                "all_weight_tensors_bound": True,
                "custom_kernel_used": True,
                "full_token_execution": True,
                "model_alone": True,
                "native_exact_decoder": True,
                "no_fallback": True,
                "prompt_template_bound": True,
                "tokenizer_bound": True,
            },
        },
    )
    control_runtime_evidence = _evidence(control_runtime)

    component_outer = tmp_path / "component-parity-outer-terminal.json"
    component_outer_doc = _write_sealed(
        component_outer,
        {
            "schema": launcher.COMPONENT_OUTER_SCHEMA,
            "status": launcher.COMPONENT_OUTER_STATUS,
            "source_binding": {
                "candidate_manifest": {**candidate_evidence, "seal_sha256": candidate_doc["seal_sha256"]},
                "candidate_admission_current": {
                    **admission_current_evidence,
                    "seal_sha256": admission_current_doc["seal_sha256"],
                },
                "candidate_admission_receipt": {
                    **admission_receipt_evidence,
                    "seal_sha256": admission_receipt_doc["seal_sha256"],
                },
                "compiler_trace_receipt": {
                    **compiler_receipt_evidence,
                    "seal_sha256": compiler_receipt_doc["seal_sha256"],
                },
                "route_capture_receipt": {
                    **route_receipt_evidence,
                    "seal_sha256": route_receipt_doc["seal_sha256"],
                },
            },
        },
    )
    component_outer_evidence = _evidence(component_outer)
    component_current = tmp_path / "component-parity-current.json"
    component_current_doc = _write_sealed(
        component_current,
        {
            "schema": launcher.COMPONENT_CURRENT_SCHEMA,
            "status": launcher.COMPONENT_CURRENT_STATUS,
            "candidate_manifest": {**candidate_evidence, "seal_sha256": candidate_doc["seal_sha256"]},
            "candidate_manifest_seal_sha256": candidate_doc["seal_sha256"],
            "component_parity_outer_terminal": {
                **component_outer_evidence,
                "seal_sha256": component_outer_doc["seal_sha256"],
            },
            "execution_scope": {
                "all_layers_executed": False,
                "full_token_executed": False,
                "literal_hawking_l0_e0_gate_up_swiglu_component_only": True,
            },
            "next_use_contract": {
                "only_typed_hq30gr2_all_layer_diagnostic_may_consume_this_component_receipt": True,
                "requires_separate_fresh_all_layer_gpu_lease": True,
            },
        },
    )
    component_current_evidence = _evidence(component_current)

    cpu_preflight = tmp_path / "all-layer-cpu-preflight.json"
    cpu_preflight_doc = _write_sealed(
        cpu_preflight,
        {
            "schema": launcher.CPU_PREFLIGHT_SCHEMA,
            "status": launcher.CPU_PREFLIGHT_STATUS,
            "candidate_manifest": {**candidate_evidence, "seal_sha256": candidate_doc["seal_sha256"]},
            "candidate_admission_current": {
                **admission_current_evidence,
                "seal_sha256": admission_current_doc["seal_sha256"],
            },
            "candidate_admission_receipt": {
                **admission_receipt_evidence,
                "seal_sha256": admission_receipt_doc["seal_sha256"],
            },
            "candidate_component_parity_current": {
                **component_current_evidence,
                "seal_sha256": component_current_doc["seal_sha256"],
            },
            "candidate_component_parity_terminal": {
                **component_outer_evidence,
                "seal_sha256": component_outer_doc["seal_sha256"],
            },
            "compiler_trace_current": {**_evidence(compiler_current), "seal_sha256": compiler_current_doc["seal_sha256"]},
            "compiler_trace_receipt": {**compiler_receipt_evidence, "seal_sha256": compiler_receipt_doc["seal_sha256"]},
            "route_capture_current": {**route_current_evidence, "seal_sha256": route_current_doc["seal_sha256"]},
            "route_capture_receipt": {**route_receipt_evidence, "seal_sha256": route_receipt_doc["seal_sha256"]},
            "preparation_current": {**_evidence(preparation_current), "seal_sha256": preparation_current_doc["seal_sha256"]},
            "preparation_receipt": {**preparation_receipt_evidence, "seal_sha256": preparation_receipt_doc["seal_sha256"]},
            "control_manifest": {**control_manifest_evidence, "seal_sha256": control_manifest_doc["seal_sha256"]},
            "control_runtime_receipt": {**control_runtime_evidence, "seal_sha256": control_runtime_doc["seal_sha256"]},
            "exact_source_template_input": {
                "annotated_trace": annotated_evidence,
                "new_diagnostic_not_historical": True,
                "probe_id": launcher.TARGET_PROBE,
                "token_count": launcher.TARGET_TOKEN_COUNT,
                "token_ids_u32le_sha256": _token_hash(token_ids),
            },
            "execution_boundary": {
                "all_layer_forward_performed": False,
                "endpoint_or_hcli_called": False,
                "future_device_executor_requires_a_new_quiet_lease_and_a_new_outer_capture": True,
                "host_fallback_for_future_candidate_execution": False,
                "metal_context_created": False,
                "metal_dispatch_performed": False,
                "raw_bf16_or_dense_weight_path": False,
                "server_watcher_or_adapter_modified": False,
                "token_loop_performed": False,
            },
            "typed_catalog_preflight": {
                "candidate_typed_catalog": {
                    "direct_tensor_count": 18865,
                    "immutable_verified_payloads": 18867,
                    "sparse_residual_tensor_count": 2,
                    "l0_e0_gate_up_layout": "HQ30GR2_SPARSE_RESIDUAL",
                    "sparse_gate_up_dispatch": {
                        "direct_fallback_for_sparse_residual_forbidden": True,
                        "exact_non_fma_scalar_order_required": True,
                    },
                },
                "control_direct_catalog": {
                    "immutable_verified_payloads": 18867,
                    "l0_e0_gate_layout": "HQ30G1B1_DIRECT",
                },
            },
            "claim_boundary": {
                "does_not_claim_capability_or_tournament": True,
                "does_not_claim_generation_or_coherence": True,
                "does_not_claim_hcli": True,
                "does_not_claim_native_runtime": True,
                "does_not_claim_tps_or_tg": True,
            },
        },
    )
    cpu_preflight_evidence = _evidence(cpu_preflight)

    lease = tmp_path / "fresh-quiet-diagnostic-lease.json"
    lease_doc = _write_sealed(
        lease,
        {
            "schema": launcher.QUIET_LEASE_SCHEMA,
            "status": launcher.QUIET_LEASE_STATUS,
            "lease_id": "fixture-fresh-q30-diagnostic-lease",
            "granted_at": "2026-08-08T00:00:00Z",
            "one_shot_lifecycle": {
                "fresh_for_this_exact_launch": True,
                "prior_terminal_receipt": None,
                "automatic_retry_allowed": False,
            },
            "execution_policy": {
                "component": launcher.QUIET_LEASE_COMPONENT,
                "quiet_qwen_family_gpu_lease": True,
                "strict_math": True,
                "diagnostic_only": True,
                "one_child_process_group_only": True,
                "timing_or_benchmarking_allowed": False,
                "hcli_or_server_allowed": False,
                "coherence_claim_allowed": False,
                "tps_or_tg_claim_allowed": False,
                "capability_claim_allowed": False,
                "tournament_claim_allowed": False,
            },
            "artifact_binding": {
                "candidate_manifest": candidate_evidence,
                "candidate_manifest_seal_sha256": candidate_doc["seal_sha256"],
                "candidate_admission_current_path": admission_current_evidence["path"],
                "candidate_admission_pointer_seal_sha256": admission_current_doc["seal_sha256"],
                "candidate_admission_receipt_seal_sha256": admission_receipt_doc["seal_sha256"],
                "control_manifest": control_manifest_evidence,
                "control_manifest_seal_sha256": control_manifest_doc["seal_sha256"],
                "control_runtime_receipt": control_runtime_evidence,
                "control_runtime_receipt_seal_sha256": control_runtime_doc["seal_sha256"],
            },
            "upstream_binding": {
                "compiler_trace_receipt": {
                    **compiler_receipt_evidence,
                    "document_sha256": compiler_receipt_evidence["sha256"],
                    "seal_sha256": compiler_receipt_doc["seal_sha256"],
                },
                "route_capture_receipt": {
                    **route_receipt_evidence,
                    "document_sha256": route_receipt_evidence["sha256"],
                    "seal_sha256": route_receipt_doc["seal_sha256"],
                },
                "preparation_receipt": {
                    **preparation_receipt_evidence,
                    "document_sha256": preparation_receipt_evidence["sha256"],
                    "seal_sha256": preparation_receipt_doc["seal_sha256"],
                },
            },
            "typed_candidate_readiness": {
                "cpu_preflight_receipt": {
                    **cpu_preflight_evidence,
                    "seal_sha256": cpu_preflight_doc["seal_sha256"],
                },
                "component_parity_current": {
                    **component_current_evidence,
                    "seal_sha256": component_current_doc["seal_sha256"],
                },
                "component_parity_terminal": {
                    **component_outer_evidence,
                    "seal_sha256": component_outer_doc["seal_sha256"],
                },
            },
            "trace_contract": {
                "probe_id": launcher.TARGET_PROBE,
                "source_template_token_count": launcher.TARGET_TOKEN_COUNT,
                "source_template_token_ids_u32le_sha256": _token_hash(token_ids),
                "forced_shared_continuation": True,
                "additional_forwards_per_path": 1,
                "complete_native_forwards_per_path": launcher.TOTAL_FULL_TOKEN_FORWARDS_PER_PATH,
                "complete_native_forwards_total": launcher.TOTAL_FULL_TOKEN_FORWARDS,
                "complete_native_layers_traversed_total": launcher.TOTAL_FULL_TOKEN_FORWARDS
                * launcher.TARGET_LAYER_COUNT,
            },
        },
    )
    lease_evidence = _evidence(lease)

    # Fixtures use harmless synthetic content, so pin the launcher to their
    # receipts while retaining the same exact-pinned behavior as production.
    monkeypatch.setattr(launcher, "PINNED_CANDIDATE_MANIFEST_SEAL", candidate_doc["seal_sha256"])
    monkeypatch.setattr(launcher, "PINNED_ADMISSION_RECEIPT_SEAL", admission_receipt_doc["seal_sha256"])
    monkeypatch.setattr(launcher, "PINNED_COMPILER_TRACE_SEAL", compiler_receipt_doc["seal_sha256"])
    monkeypatch.setattr(launcher, "PINNED_ROUTE_CAPTURE_SEAL", route_receipt_doc["seal_sha256"])
    monkeypatch.setattr(launcher, "PINNED_PREPARATION_SEAL", preparation_receipt_doc["seal_sha256"])
    monkeypatch.setattr(launcher, "PINNED_COMPONENT_CURRENT_SEAL", component_current_doc["seal_sha256"])
    monkeypatch.setattr(launcher, "PINNED_CPU_PREFLIGHT_SEAL", cpu_preflight_doc["seal_sha256"])

    return {
        "candidate": candidate,
        "candidate_evidence": candidate_evidence,
        "candidate_seal": candidate_doc["seal_sha256"],
        "admission_current": admission_current,
        "admission_current_evidence": admission_current_evidence,
        "admission_pointer_seal": admission_current_doc["seal_sha256"],
        "admission_receipt": admission_receipt,
        "admission_receipt_evidence": admission_receipt_evidence,
        "admission_receipt_seal": admission_receipt_doc["seal_sha256"],
        "compiler_current": compiler_current,
        "compiler_receipt": compiler_receipt,
        "compiler_receipt_evidence": compiler_receipt_evidence,
        "compiler_seal": compiler_receipt_doc["seal_sha256"],
        "route_current": route_current,
        "route_current_evidence": route_current_evidence,
        "route_receipt": route_receipt,
        "route_receipt_evidence": route_receipt_evidence,
        "route_seal": route_receipt_doc["seal_sha256"],
        "preparation_current": preparation_current,
        "preparation_receipt": preparation_receipt,
        "preparation_receipt_evidence": preparation_receipt_evidence,
        "preparation_seal": preparation_receipt_doc["seal_sha256"],
        "component_current": component_current,
        "component_current_evidence": component_current_evidence,
        "component_current_seal": component_current_doc["seal_sha256"],
        "component_outer": component_outer,
        "component_outer_evidence": component_outer_evidence,
        "component_outer_seal": component_outer_doc["seal_sha256"],
        "cpu_preflight": cpu_preflight,
        "cpu_preflight_evidence": cpu_preflight_evidence,
        "cpu_preflight_seal": cpu_preflight_doc["seal_sha256"],
        "control_manifest": control_manifest,
        "control_manifest_evidence": control_manifest_evidence,
        "control_manifest_seal": control_manifest_doc["seal_sha256"],
        "control_runtime": control_runtime,
        "control_runtime_evidence": control_runtime_evidence,
        "control_runtime_seal": control_runtime_doc["seal_sha256"],
        "lease": lease,
        "lease_evidence": lease_evidence,
        "lease_seal": lease_doc["seal_sha256"],
    }


def _probe(tmp_path: Path, body: str) -> tuple[Path, Path]:
    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    probe.write_text(
        "#!/bin/sh\n"
        f"printf run >> {shlex.quote(str(marker))}\n"
        f"{body}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _config(
    tmp_path: Path,
    probe: Path,
    inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_lease: bool = True,
) -> launcher.LaunchConfig:
    monkeypatch.setattr(launcher, "PINNED_EXECUTOR_SHA256", _sha256(probe))
    return launcher.LaunchConfig(
        probe_bin=probe,
        candidate_manifest=inputs["candidate"],  # type: ignore[arg-type]
        candidate_admission_current=inputs["admission_current"],  # type: ignore[arg-type]
        compiler_trace_current=inputs["compiler_current"],  # type: ignore[arg-type]
        route_capture_current=inputs["route_current"],  # type: ignore[arg-type]
        preparation_current=inputs["preparation_current"],  # type: ignore[arg-type]
        cpu_preflight_receipt=inputs["cpu_preflight"],  # type: ignore[arg-type]
        component_parity_current=inputs["component_current"],  # type: ignore[arg-type]
        control_manifest=inputs["control_manifest"],  # type: ignore[arg-type]
        control_runtime_receipt=inputs["control_runtime"],  # type: ignore[arg-type]
        lease_receipt=inputs["lease"] if include_lease else None,  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        workers=1,
        timeout_seconds=10.0,
    )


def _valid_inner_probe(tmp_path: Path, inputs: dict[str, object]) -> tuple[Path, Path]:
    """Create a harmless fake child that writes a sealed receipt last."""

    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    static = {
        "candidate_manifest": inputs["candidate_evidence"],
        "candidate_seal": inputs["candidate_seal"],
        "admission_current": inputs["admission_current_evidence"],
        "admission_pointer_seal": inputs["admission_pointer_seal"],
        "admission_receipt_seal": inputs["admission_receipt_seal"],
        "compiler_receipt": inputs["compiler_receipt_evidence"],
        "compiler_seal": inputs["compiler_seal"],
        "route_receipt": inputs["route_receipt_evidence"],
        "route_seal": inputs["route_seal"],
        "preparation_receipt": inputs["preparation_receipt_evidence"],
        "preparation_seal": inputs["preparation_seal"],
        "control_manifest": inputs["control_manifest_evidence"],
        "control_manifest_seal": inputs["control_manifest_seal"],
        "control_runtime": inputs["control_runtime_evidence"],
        "control_runtime_seal": inputs["control_runtime_seal"],
        "lease": inputs["lease_evidence"],
        "lease_seal": inputs["lease_seal"],
    }
    source = f'''#!{sys.executable}
import hashlib
import json
from pathlib import Path
import sys
from lab.receipts import seal

STATIC = json.loads({json.dumps(json.dumps(static))})
MARKER = Path({json.dumps(str(marker))})
args = sys.argv[1:]
def value(name):
    return Path(args[args.index(name) + 1])
def evidence(path):
    return {{"path": str(path.resolve()), "present": True, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
MARKER.write_text("run", encoding="utf-8")
contract_path = value("--input-contract")
contract = json.loads(contract_path.read_text(encoding="utf-8"))
inner_dir = value("--capture-dir")
inner_dir.mkdir()
contract_evidence = evidence(contract_path)
receipt = {{
    "schema": {json.dumps(launcher.EXPECTED_INNER_SCHEMA)},
    "status": {json.dumps(launcher.EXPECTED_INNER_STATUS)},
    "mode": {json.dumps(launcher.MODE)},
    "metal_device_or_dispatch_performed": True,
    "typed_hq30gr2_diagnostic_only": True,
    "durable_capture": {{"receipt_written_last_is_completion_marker": True}},
    "exact_trace_execution": {{
        "probe_id": "literal_hawking",
        "source_template_token_count": 369,
        "source_template_token_ids_u32le_sha256": contract["exact_trace"]["source_template_token_ids_u32le_sha256"],
        "baseline_exact_prefix_all_48_layers": True,
        "candidate_exact_prefix_all_48_layers": True,
        "unbounded_generation_or_sampling_loop_performed": False,
        "forced_continuation": {{
            "baseline_deterministic_argmax_after_exact_prefix": True,
            "forced_identical_token_into_baseline_and_candidate": True,
            "additional_forwards_per_path": 1,
            "baseline_additional_all_48_layers": True,
            "candidate_additional_all_48_layers": True,
            "forced_token_id": 7,
        }},
    }},
    "structural_witnesses": {{
        "control_scalar_path": {{
            "exact_prefix_token_forwards": 369,
            "all_layer_route_captures": 369 * 48,
            "layers_per_forward": 48,
            "route_trace_sha256": "a" * 64,
        }},
        "control_forced_continuation": {{
            "additional_forwards": 1,
            "step": {{
                "position": 369,
                "all_layers_route_captured": 48,
                "experts_per_layer": 8,
                "route_ids_u32le_sha256": "b" * 64,
                "command_buffers": 1,
                "metal_dispatches": 1,
            }},
        }},
        "candidate_typed_hq30gr2_path": {{
            "exact_prefix_token_forwards": 369,
            "all_layer_route_captures": 369 * 48,
            "layers_per_forward": 48,
            "route_trace_sha256": "c" * 64,
        }},
        "candidate_forced_continuation": {{
            "additional_forwards": 1,
            "step": {{
                "position": 369,
                "all_layers_route_captured": 48,
                "experts_per_layer": 8,
                "route_ids_u32le_sha256": "d" * 64,
                "command_buffers": 1,
                "metal_dispatches": 1,
            }},
        }},
        "typed_l0_e0_sparse_interception": {{
            "selected_residual_organs": [
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.0.up_proj.weight",
            ],
            "device_sparse_gate_up_encodes": 1,
            "matching_l0_e0_route_selections": 1,
            "direct_fallback_for_sparse_residual_forbidden": True,
            "scalar_control_topology_for_all_unchanged_organs": True,
        }},
        "model_bodies_concurrent": False,
        "timing_or_rate_values_recorded": False,
    }},
    "artifact_binding": {{
        "candidate_manifest": STATIC["candidate_manifest"],
        "candidate_manifest_seal_sha256": STATIC["candidate_seal"],
        "candidate_admission_current_path": STATIC["admission_current"]["path"],
        "candidate_admission_pointer_seal_sha256": STATIC["admission_pointer_seal"],
        "candidate_admission_receipt_seal_sha256": STATIC["admission_receipt_seal"],
        "control_manifest": STATIC["control_manifest"],
        "control_manifest_seal_sha256": STATIC["control_manifest_seal"],
        "control_runtime_receipt": STATIC["control_runtime"],
        "control_runtime_receipt_seal_sha256": STATIC["control_runtime_seal"],
    }},
    "upstream_diagnostic_binding": {{
        "compiler_trace_receipt": {{**STATIC["compiler_receipt"], "document_sha256": STATIC["compiler_receipt"]["sha256"], "seal_sha256": STATIC["compiler_seal"]}},
        "route_capture_receipt": {{**STATIC["route_receipt"], "document_sha256": STATIC["route_receipt"]["sha256"], "seal_sha256": STATIC["route_seal"]}},
        "preparation_receipt": {{**STATIC["preparation_receipt"], "document_sha256": STATIC["preparation_receipt"]["sha256"], "seal_sha256": STATIC["preparation_seal"]}},
    }},
    "input_contract": {{**contract_evidence, "document_sha256": contract_evidence["sha256"], "seal_sha256": contract["seal_sha256"], "schema": contract["schema"], "status": contract["status"]}},
    "metal_execution_policy": {{
        "strict_math_required": True,
        "diagnostic_only": True,
        "timing_or_benchmarking_allowed": False,
        "hcli_or_server_allowed": False,
        "tps_or_tg_claim_allowed": False,
        "coherence_claim_allowed": False,
        "capability_claim_allowed": False,
        "tournament_claim_allowed": False,
        "lease_binding": {{**STATIC["lease"], "document_sha256": STATIC["lease"]["sha256"], "seal_sha256": STATIC["lease_seal"], "schema": {json.dumps(launcher.QUIET_LEASE_SCHEMA)}, "status": {json.dumps(launcher.QUIET_LEASE_STATUS)}}},
    }},
    "claim_boundary": {{
        "does_not_claim_hcli": True,
        "does_not_claim_coherence": True,
        "does_not_claim_tps_or_tg": True,
        "does_not_claim_capability": True,
        "does_not_claim_tournament": True,
    }},
}}
(inner_dir / "receipt.json").write_text(json.dumps(seal(receipt), sort_keys=True) + "\\n", encoding="utf-8")
print("harmless child stdout")
print("harmless child stderr", file=sys.stderr)
'''
    probe.write_text(source, encoding="utf-8")
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def test_missing_current_compiler_evidence_refuses_before_a_child_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, monkeypatch)
    Path(inputs["compiler_current"]).unlink()  # type: ignore[arg-type]

    with pytest.raises(launcher.Qwen30AllLayerProbeLauncherError, match="compiler-trace-current"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_mismatched_sealed_route_trace_binding_refuses_before_a_child_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, monkeypatch)
    route_receipt = Path(inputs["route_receipt"])  # type: ignore[arg-type]
    replacement = _rewrite_sealed(
        route_receipt,
        lambda document: document["binding"]["compiler_trace"].__setitem__("seal_sha256", "f" * 64),
    )
    route_current = Path(inputs["route_current"])  # type: ignore[arg-type]
    _rewrite_sealed(
        route_current,
        lambda document: document["route_capture_receipt"].__setitem__("seal_sha256", replacement["seal_sha256"]),
    )
    monkeypatch.setattr(launcher, "PINNED_ROUTE_CAPTURE_SEAL", replacement["seal_sha256"])

    with pytest.raises(launcher.Qwen30AllLayerProbeLauncherError, match="compiler trace seal drifted"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_requires_fresh_quiet_lease_before_any_child_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, monkeypatch, include_lease=False)

    with pytest.raises(launcher.Qwen30AllLayerProbeLauncherError, match="lease"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_rejects_unpinned_executor_before_any_child_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, monkeypatch)
    monkeypatch.setattr(launcher, "PINNED_EXECUTOR_SHA256", "0" * 64)

    with pytest.raises(launcher.Qwen30AllLayerProbeLauncherError, match="probe-bin SHA-256"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_typed_cpu_preflight_contract_drift_refuses_before_any_child_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, monkeypatch)
    preflight = Path(inputs["cpu_preflight"])  # type: ignore[arg-type]
    replacement = _rewrite_sealed(
        preflight,
        lambda document: document["typed_catalog_preflight"]["candidate_typed_catalog"].__setitem__(
            "sparse_residual_tensor_count", 1
        ),
    )
    monkeypatch.setattr(launcher, "PINNED_CPU_PREFLIGHT_SEAL", replacement["seal_sha256"])

    with pytest.raises(launcher.Qwen30AllLayerProbeLauncherError, match="typed catalog contract drifted"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_launcher_reaps_nonzero_child_and_replays_without_a_second_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, 'echo "child stdout"; echo "child stderr" >&2; exit 7')
    config = _config(tmp_path, probe, inputs, monkeypatch)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_NONZERO")
    terminal = receipt["child"]["terminal"]
    assert terminal["reaped"] is True
    assert terminal["process_group_isolated"] is True
    assert terminal["timed_out"] is False
    assert terminal["returncode"] == 7
    assert terminal["exit_code"] == 7
    assert terminal["signal"] is None
    assert receipt["one_shot"]["terminal_receipt_written_last"] is True
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "child stdout\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "child stderr\n"

    assert launcher.run_attempt(config) == receipt
    assert marker.read_text(encoding="utf-8") == "run"


def test_outer_launcher_reaps_a_signal_terminated_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _probe(tmp_path, 'echo "before signal"; kill -TERM $$')
    config = _config(tmp_path, probe, inputs, monkeypatch)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_SIGNAL")
    terminal = receipt["child"]["terminal"]
    assert terminal["reaped"] is True
    assert terminal["process_group_isolated"] is True
    assert terminal["returncode"] == -15
    assert terminal["exit_code"] is None
    assert terminal["signal"] == 15
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"


def test_valid_harmless_child_is_bound_to_exact_trace_and_explicit_claim_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    probe, marker = _valid_inner_probe(tmp_path, inputs)
    config = _config(tmp_path, probe, inputs, monkeypatch)

    receipt = launcher.run_attempt(config)

    assert receipt["status"] == "CAPTURED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_DIAGNOSTIC_UNQUALIFIED"
    assert receipt["inner_probe_capture"]["binding_valid"] is True
    assert receipt["diagnostic_input_contract"]["source_template_token_count"] == 369
    assert receipt["diagnostic_input_contract"]["forced_shared_continuation"] is True
    assert receipt["one_shot"]["terminal_receipt_written_last"] is True
    for key in (
        "does_not_claim_hcli",
        "does_not_claim_coherence",
        "does_not_claim_tps_or_tg",
        "does_not_claim_capability",
        "does_not_claim_tournament",
    ):
        assert receipt["claim_boundary"][key] is True
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "harmless child stdout\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "harmless child stderr\n"
