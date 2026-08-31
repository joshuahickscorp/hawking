"""Write a compact, resumable overnight Hawking handoff receipt.

The handoff is a status snapshot, not a second control plane. It records the
authorities that already exist (Mission/DAG, provider profiles, ModelLake
supervision, and protected receipts), the exact pinned identities, and exact
continuation commands. Missing or unfinished work stays visible.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.agentos.modellake_receipts import (
    preferred_census_receipt,
    preferred_supervision_receipt,
)
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.overnight_handoff.v1"
DEFAULT_NAME = "OVERNIGHT_HAWKING_HANDOFF.json"
MODEL_LAKE_JOB = "job-2f77c1d6-e33b-44fe-bc12-549cf47805c7"
UNATTENDED_JOB = "job-a323d5ee-404c-4708-b7a8-aea7fcd15f3a"
FLASH_TRANSFORM_JOB = "job-db0bdba9-2515-4f68-9836-1849b435cd1d"


def _read_object(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> Optional[str]:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LC_ALL": "C",
                "LANG": "C",
            },
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def _receipt(repo: Path, name: str) -> Optional[Dict[str, Any]]:
    return _read_object(repo / "receipts" / "headless" / name)


def _compact_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "label",
            "state",
            "pid",
            "argv",
            "cwd",
            "resumable",
            "returncode",
            "started_at",
            "finished_at",
            "error",
            "log_path",
        )
        if key in job
    }


def _background(repo: Path) -> list[Dict[str, Any]]:
    try:
        from hcli.agentos.background import BackgroundJobStore

        store = BackgroundJobStore(repo, allowed_roots=(repo,))
        return [_compact_job(item) for item in store.list()]
    except (OSError, ValueError, RuntimeError):
        return []


def _window_summary(repo: Path) -> Dict[str, Any]:
    final_path = repo / "receipts" / "headless" / "HCLI_AGENTOS_UNATTENDED_WINDOW.json"
    final = _read_object(final_path)
    candidates = [
        repo / ".hcli" / "autonomy-window-verified" / "window-progress.json",
        repo / ".hcli" / "autonomy-window" / "window-progress.json",
        repo / ".hcli" / "unattended-qwen-long" / "window-progress.json",
    ]
    progress: list[Dict[str, Any]] = []
    for path in candidates:
        value = _read_object(path)
        if value is None:
            continue
        progress.append({
            "path": str(path),
            "window_kind": "qwen-accelerator-long" if path.parent.name == "unattended-qwen-long" else "autonomy",
            "status": value.get("status"),
            "started_at": value.get("started_at"),
            "deadline": value.get("deadline"),
            "cycles": len(value.get("cycles") or []),
            "metrics": value.get("metrics") or {},
            "checks": value.get("checks") or {},
        })
    return {
        "final_receipt": {
            "path": str(final_path),
            "present": final is not None,
            "status": final.get("status") if final else None,
            "metrics": final.get("metrics") if final else None,
            "checks": final.get("checks") if final else None,
            "elapsed_s": final.get("elapsed_s") if final else None,
        },
        "progress_workspaces": progress,
        "claim_boundary": "Only a completed receipt with fresh model-call evidence can qualify the requested unattended window.",
    }


def _model_lake_summary(repo: Path) -> Dict[str, Any]:
    census_path = preferred_census_receipt(repo)
    supervision_path = preferred_supervision_receipt(repo)
    census = _read_object(census_path) or {}
    supervision = _read_object(supervision_path) or {}
    job = supervision.get("job") if isinstance(supervision.get("job"), dict) else {}
    partial = supervision.get("partial") or {}
    final = supervision.get("final") or {}
    partial_inventory = census.get("partials") or []
    if isinstance(partial_inventory, dict):
        partial_inventory = partial_inventory.get("entries") or []
    partial_count = sum(
        1
        for entry in partial_inventory
        if isinstance(entry, dict) and entry.get("name") != ".DS_Store"
    )
    return {
        "root": "/Volumes/corpdrive/hawking-modellake",
        "census": {
            "status": census.get("status"),
            "capacity": census.get("capacity"),
            "verified_specimens": census.get("verified_specimens") or census.get("specimens"),
            "partial_count": partial_count,
            "target": census.get("target") or {},
            "receipt_path": str(census_path),
        },
        "supervision": {
            "status": supervision.get("status"),
            "qualification": supervision.get("qualification"),
            "job": _compact_job(job) if job else None,
            "partial": {"path": partial.get("path"), "bytes": partial.get("direct_bytes"), "files": partial.get("direct_files")},
            "final": {"present": final.get("present"), "path": final.get("path")},
            "checks": supervision.get("checks") or {},
            "receipt_path": str(supervision_path),
        },
        "no_delete_policy": True,
        "next_action": "Continue supervision; if interrupted, resume the same pinned argv only after re-census and headroom checks; publish only after full hash verification and atomic rename.",
    }


def _flash_summary(repo: Path) -> Dict[str, Any]:
    flash = _receipt(repo, "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json") or {}
    executable = _receipt(repo, "FLASH_NEXT_NOETIC_EXECUTABLE.json") or {}
    ebpw = _receipt(repo, "FLASH_EBPW_BUDGET.json") or {}
    token_ns = _receipt(repo, "FLASH_TOKEN_NS_BUDGET.json") or {}
    tensor_probe = _receipt(repo, "FLASH_FIRST_TENSOR_PROBE.json") or {}
    representation_experiment = _receipt(repo, "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json") or {}
    transform_parity = _receipt(repo, "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json") or {}
    loader_roundtrip = _receipt(repo, "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json") or {}
    loader_roundtrip_shared = _receipt(repo, "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP_SHARED.json") or {}
    body_kernel_path = repo / "receipts" / "headless" / "FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json"
    legacy_kernel_path = repo / "receipts" / "headless" / "FLASH_NOETIC_Q4_KERNEL_PARITY.json"
    kernel_path = body_kernel_path if body_kernel_path.is_file() else legacy_kernel_path
    kernel_parity = _read_object(kernel_path) or {}
    component_body = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_BODY.json") or {}
    component_campaign = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json") or {}
    graph_component = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json") or {}
    router_body = _receipt(repo, "FLASH_NOETIC_ROUTER_COMPONENT_BODY.json") or {}
    router_kernel = _receipt(repo, "FLASH_NOETIC_ROUTER_COMPONENT_KERNEL_PARITY.json") or {}
    router_graph = _receipt(repo, "FLASH_NOETIC_ROUTER_GRAPH.json") or {}
    router_selection = _receipt(repo, "FLASH_NOETIC_ROUTER_SELECTION.json") or {}
    router_selection_native = _receipt(repo, "FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json") or {}
    routed_expert_dispatch_native = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json") or {}
    gate_up_swiglu_native = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_GATE_UP_SWIGLU_NATIVE.json") or {}
    expert_composition_native = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_COMPOSITION_NATIVE.json") or {}
    shared_expert_composition_native = _receipt(repo, "FLASH_NOETIC_SHARED_EXPERT_COMPOSITION_NATIVE.json") or {}
    shared_residual_hyperconnection_native = _receipt(repo, "FLASH_NOETIC_SHARED_RESIDUAL_HYPERCONNECTION_NATIVE.json") or {}
    exact_hyperconnection_native = _receipt(repo, "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json") or {}
    gpu_work_ledger = _receipt(repo, "FLASH_GPU_WORK_LEDGER.json") or {}
    token_critical_path = _receipt(repo, "FLASH_TOKEN_CRITICAL_PATH.json") or {}
    cuda_capability_graph = _receipt(repo, "CUDA_CAPABILITY_GRAPH.json") or {}
    router_representation_ab = _receipt(repo, "FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json") or {}
    source = flash.get("source_identity") or flash.get("source") or {}
    promotion = flash.get("promotion_gate") or {}
    return {
        "repo": source.get("repo") or "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": source.get("pinned_revision") or source.get("resolved_revision"),
        "expected_complete_source_bytes": source.get("expected_complete_source_bytes"),
        "expected_file_count": source.get("expected_file_count"),
        "architecture_fingerprint": (flash.get("architecture_fingerprint") or {}).get("value"),
        "architecture": flash.get("architecture") or {},
        "gravity_targets": [
            "expert bank sharing/bases/residuals/factors/generators/router",
            "separate n-gram lookup/generator system",
            "DeltaNet state representation",
            "QSA sparse attention/indexer",
            "explicit MTP draft/verify/rollback accounting",
            "native kernels and command-boundary telemetry",
        ],
        "promotion": {
            "status": promotion.get("status"),
            "hard_gate": promotion.get("hard_gate") or {},
            "measured": promotion.get("measured") or {},
            "missing_or_refused": promotion.get("missing_or_refused") or [],
        },
        "noetic_executable": {
            "status": executable.get("status"),
            "qualification": executable.get("qualification"),
            "NOT_FOR_PROMOTION": executable.get("NOT_FOR_PROMOTION"),
            "promotion_allowed": executable.get("promotion_allowed"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NEXT_NOETIC_EXECUTABLE.json"),
            "native_loader": executable.get("native_loader"),
            "native_kernels": executable.get("native_kernels"),
            "graph_runtime": executable.get("graph_runtime"),
            "source_graph_component": executable.get("source_graph_component"),
            "complete_token_timing": executable.get("complete_token_timing"),
            "runtime_genome": executable.get("runtime_genome"),
        },
        "bounded_source_tensor_probe": {
            "status": tensor_probe.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_FIRST_TENSOR_PROBE.json"),
            "source_label": tensor_probe.get("source_label"),
            "candidate_label": tensor_probe.get("candidate_label"),
            "tensor_name": tensor_probe.get("tensor_name"),
            "organ": tensor_probe.get("organ"),
            "dense_vs_packed_low_bit": tensor_probe.get("dense_vs_packed_low_bit"),
            "body_mutated": tensor_probe.get("body_mutated"),
            "model_loaded": tensor_probe.get("model_loaded"),
            "claim_boundary": "bounded slice evidence only; whole-model capability and runtime remain untested",
        },
        "bounded_source_layout_experiment": {
            "status": representation_experiment.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json"),
            "source_label": representation_experiment.get("source_label"),
            "candidate_label": representation_experiment.get("candidate_label"),
            "tensor_name": representation_experiment.get("tensor_name"),
            "source_tensor": representation_experiment.get("source_tensor"),
            "candidates": representation_experiment.get("candidates"),
            "comparison": representation_experiment.get("comparison"),
            "replications": representation_experiment.get("replications"),
            "body_mutated": representation_experiment.get("body_mutated"),
            "model_loaded": representation_experiment.get("model_loaded"),
            "claim_boundary": "bounded source-layout/reference-vector evidence only; whole-model capability, native kernel, and runtime remain untested",
        },
        "full_tensor_transform_parity": {
            "status": transform_parity.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"),
            "source_tensor": transform_parity.get("source_tensor"),
            "candidates": transform_parity.get("candidates"),
            "comparison": transform_parity.get("comparison"),
            "transform_parity": transform_parity.get("transform_parity"),
            "next_experiment": transform_parity.get("next_experiment"),
            "body_mutated": transform_parity.get("body_mutated"),
            "model_loaded": transform_parity.get("model_loaded"),
            "claim_boundary": transform_parity.get("claim_boundary") or "full tensor transform only; whole-model loader, capability, native kernel, and runtime remain untested",
        },
        "noetic_loader_roundtrip": {
            "status": loader_roundtrip.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"),
            "candidate_id": loader_roundtrip.get("candidate_id"),
            "representation_descriptor": loader_roundtrip.get("representation_descriptor"),
            "serialized_descriptor_sha256": loader_roundtrip.get("serialized_descriptor_sha256"),
            "encoded_sample": loader_roundtrip.get("encoded_sample"),
            "loader_roundtrip": loader_roundtrip.get("loader_roundtrip"),
            "body_mutated": loader_roundtrip.get("body_mutated"),
            "model_loaded": loader_roundtrip.get("model_loaded"),
            "claim_boundary": loader_roundtrip.get("claim_boundary") or "bounded descriptor round-trip only; native loader and runtime remain untested",
            "quality_alternate_receipt": str(repo / "receipts" / "headless" / "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP_SHARED.json") if loader_roundtrip_shared else None,
            "quality_alternate_status": loader_roundtrip_shared.get("status"),
        },
        "bounded_native_kernel_parity": {
            "status": kernel_parity.get("status"),
            "receipt_path": str(kernel_path),
            "source_tensor": kernel_parity.get("source_tensor"),
            "noetic_descriptor": kernel_parity.get("noetic_descriptor"),
            "noetic_representation": kernel_parity.get("noetic_representation"),
            "native_loader": kernel_parity.get("native_loader"),
            "native_kernel": kernel_parity.get("native_kernel"),
            "candidate_body": kernel_parity.get("candidate_body"),
            "gpu_timing": kernel_parity.get("gpu_timing"),
            "parity": kernel_parity.get("parity"),
            "body_mutated": kernel_parity.get("body_mutated"),
            "model_loaded": kernel_parity.get("model_loaded"),
            "claim_boundary": kernel_parity.get("claim_boundary") or "bounded source-block Metal evidence only; complete Flash runtime remains untested",
            "next_action": kernel_parity.get("next_action"),
        },
        "source_independent_component_body": {
            "status": component_body.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_BODY.json"),
            "candidate_id": component_body.get("candidate_id"),
            "body": component_body.get("body"),
            "source_independent": component_body.get("source_independent"),
            "candidate_body_persisted": component_body.get("candidate_body_persisted"),
            "native_loader": component_body.get("native_loader"),
            "source_guard": component_body.get("source_guard"),
            "whole_model_capability": component_body.get("whole_model_capability"),
            "whole_model_runtime": component_body.get("whole_model_runtime"),
            "claim_boundary": component_body.get("claim_boundary"),
            "next_action": component_body.get("next_action"),
        },
        "source_independent_component_campaign": {
            "status": component_campaign.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json"),
            "component_status": component_campaign.get("component_status"),
            "component_count": component_campaign.get("component_count"),
            "component_windows": component_campaign.get("component_windows"),
            "source_independent_execution": component_campaign.get("source_independent_execution"),
            "candidate_body_persisted": component_campaign.get("candidate_body_persisted"),
            "physical_graph": component_campaign.get("physical_graph"),
            "noetic_ir": component_campaign.get("noetic_ir"),
            "whole_model_capability": component_campaign.get("whole_model_capability"),
            "complete_token_runtime": component_campaign.get("complete_token_runtime"),
            "promotion_allowed": component_campaign.get("promotion_allowed"),
            "claim_boundary": component_campaign.get("claim_boundary"),
            "next_action": component_campaign.get("next_action"),
        },
        "source_independent_router_component": {
            "status": router_body.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_COMPONENT_BODY.json"),
            "component_kind": router_body.get("component_kind"),
            "tensor_name": router_body.get("tensor_name"),
            "source_block": router_body.get("source_block"),
            "representation_descriptor": router_body.get("representation_descriptor"),
            "body": router_body.get("body"),
            "source_independent": router_body.get("source_independent"),
            "candidate_body_persisted": router_body.get("candidate_body_persisted"),
            "native_kernel": router_kernel.get("native_kernel"),
            "gpu_timing": router_kernel.get("gpu_timing"),
            "parity": router_kernel.get("parity"),
            "native_loader": router_kernel.get("native_loader"),
            "claim_boundary": router_body.get("claim_boundary"),
            "next_action": router_body.get("next_action"),
        },
        "bounded_noetic_router_graph": {
            "status": router_graph.get("status"),
            "component_status": router_graph.get("component_status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_GRAPH.json"),
            "candidate_id": router_graph.get("candidate_id"),
            "component_window": router_graph.get("component_window"),
            "source_independent_execution": router_graph.get("source_independent_execution"),
            "candidate_body_persisted": router_graph.get("candidate_body_persisted"),
            "physical_graph": router_graph.get("physical_graph"),
            "noetic_ir": router_graph.get("noetic_ir"),
            "whole_model_capability": router_graph.get("whole_model_capability"),
            "complete_token_runtime": router_graph.get("complete_token_runtime"),
            "promotion_allowed": router_graph.get("promotion_allowed"),
            "claim_boundary": router_graph.get("claim_boundary"),
            "next_action": router_graph.get("next_action"),
        },
        "bounded_noetic_router_selection": {
            "status": router_selection.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_SELECTION.json"),
            "semantic_type": router_selection.get("semantic_type"),
            "compiler_stage": router_selection.get("compiler_stage"),
            "source_identity": router_selection.get("source_identity"),
            "config": router_selection.get("config"),
            "selection": router_selection.get("selection"),
            "source_selection": router_selection.get("source_selection"),
            "source_selection_parity": router_selection.get("source_selection_parity"),
            "source_reference_execution": router_selection.get("source_reference_execution"),
            "execution": router_selection.get("execution"),
            "physical_graph": router_selection.get("physical_graph"),
            "noetic_ir": router_selection.get("noetic_ir"),
            "native_selection_execution_observed": router_selection.get("native_selection_execution_observed"),
            "whole_model_capability": router_selection.get("whole_model_capability"),
            "complete_token_runtime": router_selection.get("complete_token_runtime"),
            "promotion_allowed": router_selection.get("promotion_allowed"),
            "claim_boundary": router_selection.get("claim_boundary"),
            "next_action": router_selection.get("next_action"),
        },
        "bounded_noetic_router_selection_native": {
            "status": router_selection_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json"),
            "semantic_type": router_selection_native.get("semantic_type"),
            "compiler_stage": router_selection_native.get("compiler_stage"),
            "qualification": router_selection_native.get("qualification"),
            "candidate_body": router_selection_native.get("candidate_body"),
            "native_loader": router_selection_native.get("native_loader"),
            "native_kernel": router_selection_native.get("native_kernel"),
            "execution": router_selection_native.get("execution"),
            "selection": router_selection_native.get("selection"),
            "reference": router_selection_native.get("reference"),
            "source_selection_parity": router_selection_native.get("source_selection_parity"),
            "parity": router_selection_native.get("parity"),
            "gpu_timing": router_selection_native.get("gpu_timing"),
            "physical_graph": router_selection_native.get("physical_graph"),
            "noetic_ir": router_selection_native.get("noetic_ir"),
            "native_selection_execution_observed": router_selection_native.get("native_selection_execution_observed"),
            "whole_model_capability": router_selection_native.get("whole_model_capability"),
            "complete_token_runtime": router_selection_native.get("complete_token_runtime"),
            "promotion_allowed": router_selection_native.get("promotion_allowed"),
            "claim_boundary": router_selection_native.get("claim_boundary"),
            "next_action": router_selection_native.get("next_action"),
        },
        "bounded_noetic_routed_expert_dispatch_native": {
            "status": routed_expert_dispatch_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json"),
            "semantic_type": routed_expert_dispatch_native.get("semantic_type"),
            "compiler_stage": routed_expert_dispatch_native.get("compiler_stage"),
            "qualification": routed_expert_dispatch_native.get("qualification"),
            "router_receipt": routed_expert_dispatch_native.get("router_receipt"),
            "campaign_receipt": routed_expert_dispatch_native.get("campaign_receipt"),
            "selection": routed_expert_dispatch_native.get("selection"),
            "source_selection_parity": routed_expert_dispatch_native.get("source_selection_parity"),
            "components": routed_expert_dispatch_native.get("components"),
            "execution": routed_expert_dispatch_native.get("execution"),
            "gpu_timing": routed_expert_dispatch_native.get("gpu_timing"),
            "gather": routed_expert_dispatch_native.get("gather"),
            "physical_graph": routed_expert_dispatch_native.get("physical_graph"),
            "noetic_ir": routed_expert_dispatch_native.get("noetic_ir"),
            "native_routed_body_dispatch_observed": routed_expert_dispatch_native.get("native_routed_body_dispatch_observed"),
            "whole_model_capability": routed_expert_dispatch_native.get("whole_model_capability"),
            "complete_expert_runtime": routed_expert_dispatch_native.get("complete_expert_runtime"),
            "complete_token_runtime": routed_expert_dispatch_native.get("complete_token_runtime"),
            "promotion_allowed": routed_expert_dispatch_native.get("promotion_allowed"),
            "claim_boundary": routed_expert_dispatch_native.get("claim_boundary"),
            "next_action": routed_expert_dispatch_native.get("next_action"),
        },
        "bounded_noetic_gate_up_swiglu_native": {
            "status": gate_up_swiglu_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_GATE_UP_SWIGLU_NATIVE.json"),
            "semantic_type": gate_up_swiglu_native.get("semantic_type"),
            "compiler_stage": gate_up_swiglu_native.get("compiler_stage"),
            "qualification": gate_up_swiglu_native.get("qualification"),
            "router_receipt": gate_up_swiglu_native.get("router_receipt"),
            "component_receipt_policy": gate_up_swiglu_native.get("component_receipt_policy"),
            "selection": gate_up_swiglu_native.get("selection"),
            "source_selection_parity": gate_up_swiglu_native.get("source_selection_parity"),
            "components": gate_up_swiglu_native.get("components"),
            "execution": gate_up_swiglu_native.get("execution"),
            "gpu_timing": gate_up_swiglu_native.get("gpu_timing"),
            "gather": gate_up_swiglu_native.get("gather"),
            "physical_graph": gate_up_swiglu_native.get("physical_graph"),
            "noetic_ir": gate_up_swiglu_native.get("noetic_ir"),
            "native_gate_up_swiglu_observed": gate_up_swiglu_native.get("native_gate_up_swiglu_observed"),
            "native_expert_gate_up_activation_observed": gate_up_swiglu_native.get("native_expert_gate_up_activation_observed"),
            "whole_model_capability": gate_up_swiglu_native.get("whole_model_capability"),
            "complete_expert_runtime": gate_up_swiglu_native.get("complete_expert_runtime"),
            "complete_token_runtime": gate_up_swiglu_native.get("complete_token_runtime"),
            "promotion_allowed": gate_up_swiglu_native.get("promotion_allowed"),
            "claim_boundary": gate_up_swiglu_native.get("claim_boundary"),
            "next_action": gate_up_swiglu_native.get("next_action"),
        },
        "bounded_noetic_expert_composition_native": {
            "status": expert_composition_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_COMPOSITION_NATIVE.json"),
            "semantic_type": expert_composition_native.get("semantic_type"),
            "compiler_stage": expert_composition_native.get("compiler_stage"),
            "qualification": expert_composition_native.get("qualification"),
            "router_receipt": expert_composition_native.get("router_receipt"),
            "component_receipt_policy": expert_composition_native.get("component_receipt_policy"),
            "selection": expert_composition_native.get("selection"),
            "source_selection_parity": expert_composition_native.get("source_selection_parity"),
            "components": expert_composition_native.get("components"),
            "execution": expert_composition_native.get("execution"),
            "input": expert_composition_native.get("input"),
            "intermediate": expert_composition_native.get("intermediate"),
            "gpu_timing": expert_composition_native.get("gpu_timing"),
            "gather": expert_composition_native.get("gather"),
            "physical_graph": expert_composition_native.get("physical_graph"),
            "noetic_ir": expert_composition_native.get("noetic_ir"),
            "native_gate_up_swiglu_observed": expert_composition_native.get("native_gate_up_swiglu_observed"),
            "native_down_projection_observed": expert_composition_native.get("native_down_projection_observed"),
            "native_expert_composition_observed": expert_composition_native.get("native_expert_composition_observed"),
            "bounded_selected_expert_output_observed": expert_composition_native.get("bounded_selected_expert_output_observed"),
            "whole_model_capability": expert_composition_native.get("whole_model_capability"),
            "complete_expert_runtime": expert_composition_native.get("complete_expert_runtime"),
            "complete_token_runtime": expert_composition_native.get("complete_token_runtime"),
            "promotion_allowed": expert_composition_native.get("promotion_allowed"),
            "claim_boundary": expert_composition_native.get("claim_boundary"),
            "next_action": expert_composition_native.get("next_action"),
        },
        "bounded_noetic_shared_expert_composition_native": {
            "status": shared_expert_composition_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_SHARED_EXPERT_COMPOSITION_NATIVE.json"),
            "semantic_type": shared_expert_composition_native.get("semantic_type"),
            "compiler_stage": shared_expert_composition_native.get("compiler_stage"),
            "qualification": shared_expert_composition_native.get("qualification"),
            "layer": shared_expert_composition_native.get("layer"),
            "component_receipt_policy": shared_expert_composition_native.get("component_receipt_policy"),
            "components": shared_expert_composition_native.get("components"),
            "execution": shared_expert_composition_native.get("execution"),
            "input": shared_expert_composition_native.get("input"),
            "intermediates": shared_expert_composition_native.get("intermediates"),
            "parity": shared_expert_composition_native.get("parity"),
            "gpu_timing": shared_expert_composition_native.get("gpu_timing"),
            "physical_graph": shared_expert_composition_native.get("physical_graph"),
            "noetic_ir": shared_expert_composition_native.get("noetic_ir"),
            "native_shared_expert_gate_up_swiglu_observed": shared_expert_composition_native.get("native_shared_expert_gate_up_swiglu_observed"),
            "native_shared_expert_down_projection_observed": shared_expert_composition_native.get("native_shared_expert_down_projection_observed"),
            "native_shared_expert_scalar_gate_observed": shared_expert_composition_native.get("native_shared_expert_scalar_gate_observed"),
            "native_shared_expert_sigmoid_gate_observed": shared_expert_composition_native.get("native_shared_expert_sigmoid_gate_observed"),
            "native_shared_expert_composition_observed": shared_expert_composition_native.get("native_shared_expert_composition_observed"),
            "source_independent_execution": shared_expert_composition_native.get("source_independent_execution") or (shared_expert_composition_native.get("noetic_ir") or {}).get("source_independent"),
            "device_intermediate_no_host_roundtrip": shared_expert_composition_native.get("device_intermediate_no_host_roundtrip") or (shared_expert_composition_native.get("physical_graph") or {}).get("device_intermediate_no_host_roundtrip"),
            "whole_model_capability": shared_expert_composition_native.get("whole_model_capability"),
            "complete_expert_runtime": shared_expert_composition_native.get("complete_expert_runtime"),
            "complete_token_runtime": shared_expert_composition_native.get("complete_token_runtime"),
            "complete_system_ebpw": shared_expert_composition_native.get("complete_system_ebpw"),
            "flash_tps": shared_expert_composition_native.get("flash_tps"),
            "promotion_allowed": shared_expert_composition_native.get("promotion_allowed"),
            "claim_boundary": shared_expert_composition_native.get("claim_boundary"),
            "next_action": shared_expert_composition_native.get("next_action"),
        },
        "bounded_noetic_shared_residual_hyperconnection_native": {
            "status": shared_residual_hyperconnection_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_SHARED_RESIDUAL_HYPERCONNECTION_NATIVE.json"),
            "semantic_type": shared_residual_hyperconnection_native.get("semantic_type"),
            "compiler_stage": shared_residual_hyperconnection_native.get("compiler_stage"),
            "qualification": shared_residual_hyperconnection_native.get("qualification"),
            "layer": shared_residual_hyperconnection_native.get("layer"),
            "dependencies": shared_residual_hyperconnection_native.get("dependencies"),
            "component_receipt_policy": shared_residual_hyperconnection_native.get("component_receipt_policy"),
            "components": shared_residual_hyperconnection_native.get("components"),
            "execution": shared_residual_hyperconnection_native.get("execution"),
            "input": shared_residual_hyperconnection_native.get("input"),
            "intermediates": shared_residual_hyperconnection_native.get("intermediates"),
            "candidate_semantics": shared_residual_hyperconnection_native.get("candidate_semantics"),
            "parity": shared_residual_hyperconnection_native.get("parity"),
            "gpu_timing": shared_residual_hyperconnection_native.get("gpu_timing"),
            "physical_graph": shared_residual_hyperconnection_native.get("physical_graph"),
            "noetic_ir": shared_residual_hyperconnection_native.get("noetic_ir"),
            "native_shared_expert_gate_up_swiglu_observed": shared_residual_hyperconnection_native.get("native_shared_expert_gate_up_swiglu_observed"),
            "native_shared_expert_down_projection_observed": shared_residual_hyperconnection_native.get("native_shared_expert_down_projection_observed"),
            "native_shared_expert_sigmoid_gate_observed": shared_residual_hyperconnection_native.get("native_shared_expert_sigmoid_gate_observed"),
            "native_hyperconnection_stream_injection_observed": shared_residual_hyperconnection_native.get("native_hyperconnection_stream_injection_observed"),
            "native_hyperconnection_low_rank_down_observed": shared_residual_hyperconnection_native.get("native_hyperconnection_low_rank_down_observed"),
            "native_hyperconnection_low_rank_up_observed": shared_residual_hyperconnection_native.get("native_hyperconnection_low_rank_up_observed"),
            "native_hyperconnection_block_inject_observed": shared_residual_hyperconnection_native.get("native_hyperconnection_block_inject_observed"),
            "native_hyperconnection_residual_mix_observed": shared_residual_hyperconnection_native.get("native_hyperconnection_residual_mix_observed"),
            "native_shared_residual_composition_observed": shared_residual_hyperconnection_native.get("native_shared_residual_composition_observed"),
            "source_independent_execution": shared_residual_hyperconnection_native.get("source_independent_execution") or (shared_residual_hyperconnection_native.get("noetic_ir") or {}).get("source_independent"),
            "device_intermediate_no_host_roundtrip": shared_residual_hyperconnection_native.get("device_intermediate_no_host_roundtrip") or (shared_residual_hyperconnection_native.get("physical_graph") or {}).get("device_intermediate_no_host_roundtrip"),
            "whole_model_capability": shared_residual_hyperconnection_native.get("whole_model_capability"),
            "complete_expert_runtime": shared_residual_hyperconnection_native.get("complete_expert_runtime"),
            "complete_token_runtime": shared_residual_hyperconnection_native.get("complete_token_runtime"),
            "complete_system_ebpw": shared_residual_hyperconnection_native.get("complete_system_ebpw"),
            "flash_tps": shared_residual_hyperconnection_native.get("flash_tps"),
            "promotion_allowed": shared_residual_hyperconnection_native.get("promotion_allowed"),
            "claim_boundary": shared_residual_hyperconnection_native.get("claim_boundary"),
            "next_action": shared_residual_hyperconnection_native.get("next_action"),
        },
        "bounded_noetic_exact_hyperconnection_native": {
            "status": exact_hyperconnection_native.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json"),
            "schema": exact_hyperconnection_native.get("schema"),
            "semantic_type": exact_hyperconnection_native.get("semantic_type"),
            "compiler_stage": exact_hyperconnection_native.get("compiler_stage"),
            "qualification": exact_hyperconnection_native.get("qualification"),
            "layer": exact_hyperconnection_native.get("layer"),
            "dependencies": exact_hyperconnection_native.get("dependencies"),
            "source_reference": exact_hyperconnection_native.get("source_reference"),
            "semantics": exact_hyperconnection_native.get("semantics"),
            "execution": exact_hyperconnection_native.get("execution"),
            "input": exact_hyperconnection_native.get("input"),
            "parity": exact_hyperconnection_native.get("parity"),
            "gpu_timing": exact_hyperconnection_native.get("gpu_timing"),
            "physical_graph": exact_hyperconnection_native.get("physical_graph"),
            "noetic_ir": exact_hyperconnection_native.get("noetic_ir"),
            "source_selection_parity": exact_hyperconnection_native.get("source_selection_parity"),
            "routed_expert_count": exact_hyperconnection_native.get("routed_expert_count"),
            "routed_expert_ids": exact_hyperconnection_native.get("routed_expert_ids"),
            "selected_weight_sum": exact_hyperconnection_native.get("selected_weight_sum"),
            "native_hyperconnection_read_observed": exact_hyperconnection_native.get("native_hyperconnection_read_observed"),
            "native_hyperconnection_write_observed": exact_hyperconnection_native.get("native_hyperconnection_write_observed"),
            "exact_hyperconnection_semantics_observed": exact_hyperconnection_native.get("exact_hyperconnection_semantics_observed"),
            "native_routed_expert_gate_up_swiglu_observed": exact_hyperconnection_native.get("native_routed_expert_gate_up_swiglu_observed"),
            "native_routed_expert_down_projection_observed": exact_hyperconnection_native.get("native_routed_expert_down_projection_observed"),
            "native_moe_weighted_sum_observed": exact_hyperconnection_native.get("native_moe_weighted_sum_observed"),
            "native_moe_shared_add_observed": exact_hyperconnection_native.get("native_moe_shared_add_observed"),
            "device_intermediate_no_host_roundtrip": exact_hyperconnection_native.get("device_intermediate_no_host_roundtrip"),
            "source_independent_execution": exact_hyperconnection_native.get("source_independent_execution"),
            "source_hc_norm_payload_exact": exact_hyperconnection_native.get("source_hc_norm_payload_exact"),
            "complete_layer0_moe_candidate": exact_hyperconnection_native.get("complete_layer0_moe_candidate"),
            "complete_moe_combine": exact_hyperconnection_native.get("complete_moe_combine"),
            "whole_model_capability": exact_hyperconnection_native.get("whole_model_capability"),
            "complete_expert_runtime": exact_hyperconnection_native.get("complete_expert_runtime"),
            "complete_token_runtime": exact_hyperconnection_native.get("complete_token_runtime"),
            "complete_system_ebpw": exact_hyperconnection_native.get("complete_system_ebpw"),
            "flash_tps": exact_hyperconnection_native.get("flash_tps"),
            "promotion_allowed": exact_hyperconnection_native.get("promotion_allowed"),
            "claim_boundary": exact_hyperconnection_native.get("claim_boundary"),
            "next_action": exact_hyperconnection_native.get("next_action"),
        },
        "flash_gpu_work_ledger": {
            "status": gpu_work_ledger.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_GPU_WORK_LEDGER.json"),
            "schema": gpu_work_ledger.get("schema"),
            "qualification": gpu_work_ledger.get("qualification"),
            "source_receipt": gpu_work_ledger.get("source_receipt"),
            "device": gpu_work_ledger.get("device"),
            "physical_graph_fingerprint": gpu_work_ledger.get("physical_graph_fingerprint"),
            "measured_runs": gpu_work_ledger.get("measured_runs"),
            "dispatches_per_graph": gpu_work_ledger.get("dispatches_per_graph"),
            "graph_gpu_ns_median": gpu_work_ledger.get("graph_gpu_ns_median"),
            "graph_host_wall_ns_median": gpu_work_ledger.get("graph_host_wall_ns_median"),
            "graph_wall_minus_gpu_ns_median": gpu_work_ledger.get("graph_wall_minus_gpu_ns_median"),
            "stages": gpu_work_ledger.get("stages"),
            "device_intermediate_no_host_roundtrip": gpu_work_ledger.get("device_intermediate_no_host_roundtrip"),
            "complete_token_runtime": gpu_work_ledger.get("complete_token_runtime"),
            "flash_tps": gpu_work_ledger.get("flash_tps"),
            "complete_system_ebpw": gpu_work_ledger.get("complete_system_ebpw"),
            "promotion_allowed": gpu_work_ledger.get("promotion_allowed"),
            "claim_boundary": gpu_work_ledger.get("claim_boundary"),
        },
        "flash_token_critical_path": {
            "status": token_critical_path.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_TOKEN_CRITICAL_PATH.json"),
            "schema": token_critical_path.get("schema"),
            "source_receipt": token_critical_path.get("source_receipt"),
            "candidate_graph": token_critical_path.get("candidate_graph"),
            "complete_token_runtime": token_critical_path.get("complete_token_runtime"),
            "accepted_tokens": token_critical_path.get("accepted_tokens"),
            "complete_wall_ns_per_accepted_token": token_critical_path.get("complete_wall_ns_per_accepted_token"),
            "flash_tps": token_critical_path.get("flash_tps"),
            "complete_system_ebpw": token_critical_path.get("complete_system_ebpw"),
            "promotion_allowed": token_critical_path.get("promotion_allowed"),
            "blockers": token_critical_path.get("blockers"),
            "claim_boundary": token_critical_path.get("claim_boundary"),
        },
        "cuda_capability_graph": {
            "status": cuda_capability_graph.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "CUDA_CAPABILITY_GRAPH.json"),
            "schema": cuda_capability_graph.get("schema"),
            "execution_device": cuda_capability_graph.get("execution_device"),
            "native_backend_observed": cuda_capability_graph.get("native_backend_observed"),
            "cuda_execution_observed": cuda_capability_graph.get("cuda_execution_observed"),
            "qualification": cuda_capability_graph.get("qualification"),
            "nodes": cuda_capability_graph.get("nodes"),
            "edges": cuda_capability_graph.get("edges"),
            "promotion_allowed": cuda_capability_graph.get("promotion_allowed"),
            "claim_boundary": cuda_capability_graph.get("claim_boundary"),
        },
        "bounded_noetic_router_representation_ab": {
            "status": router_representation_ab.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json"),
            "semantic_type": router_representation_ab.get("semantic_type"),
            "compiler_stage": router_representation_ab.get("compiler_stage"),
            "source_identity": router_representation_ab.get("source_identity"),
            "config": router_representation_ab.get("config"),
            "source_block": router_representation_ab.get("source_block"),
            "source_selection": router_representation_ab.get("source_selection"),
            "candidates": router_representation_ab.get("candidates"),
            "recommendation": router_representation_ab.get("recommendation"),
            "physical_graph": router_representation_ab.get("physical_graph"),
            "noetic_ir": router_representation_ab.get("noetic_ir"),
            "validation": router_representation_ab.get("validation"),
            "candidate_bodies_persisted": router_representation_ab.get("candidate_bodies_persisted"),
            "whole_model_capability": router_representation_ab.get("whole_model_capability"),
            "complete_token_runtime": router_representation_ab.get("complete_token_runtime"),
            "promotion_allowed": router_representation_ab.get("promotion_allowed"),
            "claim_boundary": router_representation_ab.get("claim_boundary"),
            "next_action": router_representation_ab.get("next_action"),
        },
        "bounded_noetic_graph_component": {
            "status": graph_component.get("status"),
            "component_status": graph_component.get("component_status"),
            "receipt_path": str(repo / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json"),
            "candidate_id": graph_component.get("candidate_id"),
            "graph_fingerprint": graph_component.get("graph_fingerprint"),
            "source_backed": graph_component.get("source_backed"),
            "candidate_body_persisted": graph_component.get("candidate_body_persisted"),
            "whole_model_capability": graph_component.get("whole_model_capability"),
            "complete_token_runtime": graph_component.get("complete_token_runtime"),
            "promotion_allowed": graph_component.get("promotion_allowed"),
            "claim_boundary": graph_component.get("claim_boundary"),
        },
        "budgets": {
            "ebpw": {"status": ebpw.get("status"), "receipt_path": str(repo / "receipts" / "headless" / "FLASH_EBPW_BUDGET.json"), "measured": ebpw.get("measured"), "target_contract": ebpw.get("target_contract")},
            "token_ns": {"status": token_ns.get("status"), "receipt_path": str(repo / "receipts" / "headless" / "FLASH_TOKEN_NS_BUDGET.json"), "system_ledger": token_ns.get("system_ledger"), "target_contract": token_ns.get("target_contract")},
        },
        "next_action": "Use the PASSED bounded native routed-expert and layer-0 shared-expert compositions as Flash graph anchors, then compose the residual boundary and remaining Flash organs; keep complete-system EBPW and accepted Flash TPS unmeasured until native protected complete-token execution exists.",
    }


def _qwen27_summary(repo: Path) -> Dict[str, Any]:
    regression = _receipt(repo, "HCLI_ACCELERATOR_REGRESSION.json") or {}
    fusion_audit = _receipt(repo, "HCLI_QWEN38_FUSION_SOURCE_AUDIT.json") or {}
    identity = _receipt(repo, "QWEN27_HISTORICAL_RUNTIME_IDENTITY.json") or {}
    runtime_diff = _receipt(repo, "QWEN27_RUNTIME_DIFF.json") or {}
    mlp = _receipt(repo, "QWEN27_MLP_DIAGNOSTIC_AB.json") or {}
    protected = _receipt(repo, "QWEN_PROTECTED_BENCH_READY.json") or {}
    profile_path = repo / "hcli" / "hawking-native.sealed-3.14.json"
    profile = _read_object(profile_path) or {}
    current = profile.get("current_runtime") or {}
    return {
        "model_a": {
            "label": "Qwen3.8-27B sealed resident / NOETIC_PARENT_A",
            "profile": str(profile_path),
            "profile_sha256": _sha256(profile_path),
            "artifact_root": profile.get("artifact_root"),
            "binary": profile.get("resident_binary") or profile.get("binary"),
            "identity": profile.get("model_id") or profile.get("resident_identity"),
            "physical_ebpw": (profile.get("representation") or {}).get("physical_ebpw") or profile.get("physical_ebpw"),
        },
        "current_runtime_observation": {
            "complete_tps_current_measured": current.get("complete_tps_current_measured"),
            "complete_tps_historical_qualified": current.get("complete_tps_historical_qualified"),
            "fallbacks": current.get("fallbacks"),
            "bench_state": regression.get("bench_state"),
            "qualification": regression.get("qualification"),
            "benchmark_class": regression.get("benchmark_class"),
            "current_vs_historical": regression.get("current_vs_historical"),
            "dispatch_kernel_genome": regression.get("prior_dispatch_kernel_genome"),
        },
        "fusion_source_audit": {
            "status": fusion_audit.get("status"),
            "qualification": fusion_audit.get("qualification"),
            "receipt_path": str(repo / "receipts" / "headless" / "HCLI_QWEN38_FUSION_SOURCE_AUDIT.json"),
            "selected_graph": fusion_audit.get("selected_graph"),
            "source_contract": fusion_audit.get("source_contract"),
            "result": fusion_audit.get("result"),
        },
        "runtime_identity_archaeology": {
            "status": identity.get("status"),
            "receipt_path": str(repo / "receipts" / "headless" / "QWEN27_HISTORICAL_RUNTIME_IDENTITY.json"),
            "historical_selection": identity.get("historical_selection"),
            "diff_summary": runtime_diff.get("summary"),
            "diff_receipt_path": str(repo / "receipts" / "headless" / "QWEN27_RUNTIME_DIFF.json"),
        },
        "mlp_selector_diagnostic": {
            "status": mlp.get("status"),
            "benchmark_class": mlp.get("benchmark_class"),
            "qualification": mlp.get("qualification"),
            "NOT_FOR_PROMOTION": mlp.get("NOT_FOR_PROMOTION"),
            "experiment_verdict": mlp.get("experiment_verdict"),
            "selector_verdict": mlp.get("selector_verdict"),
            "receipt_path": str(repo / "receipts" / "headless" / "QWEN27_MLP_DIAGNOSTIC_AB.json"),
        },
        "protected_benchmark_watcher": {
            "status": protected.get("status"),
            "qualification": protected.get("qualification"),
            "NOT_FOR_PROMOTION": protected.get("NOT_FOR_PROMOTION"),
            "polls": len(protected.get("polls") or []),
            "runs": protected.get("runs") or [],
            "receipt_path": str(repo / "receipts" / "headless" / "QWEN_PROTECTED_BENCH_READY.json"),
        },
        "next_experiment": "Protected quiescent same-source A/B: record binary/artifact/tokenizer, representation, dispatches, complete wall/GPU timing, fallback count, capability, and cache/quiescence before accepting any optimization.",
    }


def _fpga_summary(repo: Path) -> Dict[str, Any]:
    preboard = _receipt(repo, "HCLI_FPGA_PREBOARD.json") or {}
    return {
        "preboard": {
            "status": preboard.get("status"),
            "fingerprint": preboard.get("fingerprint"),
            "physical_board": preboard.get("physical_board"),
            "fpga_backend": preboard.get("fpga_backend"),
            "checks": preboard.get("checks") or {},
            "simulation": preboard.get("simulation"),
        },
        "maps": {
            "qwen27": str(repo / "receipts" / "headless" / "QWEN27_FPGA_ORGAN_MAP.json"),
            "flash_next": str(repo / "receipts" / "headless" / "FLASH_NEXT_FPGA_ORGAN_MAP.json"),
        },
        "shared_primitives": preboard.get("shared_primitives") or [],
        "labels": {"verified": "[V]", "derived": "[D]", "simulated": "[S]"},
        "next_action": "Compile/verify HWIR and link sensitivity contracts; do not report board, bitstream, U50, or hardware timing until a physical receipt exists.",
    }


def _charge_summary(repo: Path) -> Dict[str, Any]:
    charge = _receipt(repo, "HAWKING_INITIAL_CHARGE.json") or {}
    return {
        "charge_id": charge.get("charge_id"),
        "status": charge.get("status"),
        "mission_id": charge.get("mission_id"),
        "workspace": charge.get("workspace"),
        "mission_state_path": charge.get("mission_state_path"),
        "provider_neutral": charge.get("provider_neutral"),
        "unit_count": len(charge.get("units") or []),
        "units": [
            {
                "id": item.get("id"),
                "priority": item.get("priority"),
                "role": item.get("role"),
                "dependencies": item.get("dependencies") or [],
                "resource_class": item.get("resource_class"),
                "retry_state": item.get("retry_state") or {},
                "stop_condition": item.get("stop_condition"),
            }
            for item in (charge.get("units") or [])
            if isinstance(item, dict)
        ],
        "next_action": charge.get("next_action"),
    }


def build_handoff(repo_root: Optional[str | os.PathLike[str]] = None, *, emit: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_NAME
    jobs = _background(repo)
    flash = _flash_summary(repo)
    router_body = flash.get("source_independent_router_component") or {}
    router_graph = flash.get("bounded_noetic_router_graph") or {}
    router_selection = flash.get("bounded_noetic_router_selection") or {}
    router_selection_native = flash.get("bounded_noetic_router_selection_native") or {}
    routed_expert_dispatch_native = flash.get("bounded_noetic_routed_expert_dispatch_native") or {}
    gate_up_swiglu_native = flash.get("bounded_noetic_gate_up_swiglu_native") or {}
    expert_composition_native = flash.get("bounded_noetic_expert_composition_native") or {}
    shared_expert_composition_native = flash.get("bounded_noetic_shared_expert_composition_native") or {}
    shared_residual_hyperconnection_native = flash.get("bounded_noetic_shared_residual_hyperconnection_native") or {}
    exact_hyperconnection_native = flash.get("bounded_noetic_exact_hyperconnection_native") or {}
    gpu_work_ledger = flash.get("flash_gpu_work_ledger") or {}
    token_critical_path = flash.get("flash_token_critical_path") or {}
    cuda_capability_graph = flash.get("cuda_capability_graph") or {}
    router_representation_ab = flash.get("bounded_noetic_router_representation_ab") or {}
    lake = _model_lake_summary(repo)
    window = _window_summary(repo)
    promotion_status = (flash.get("promotion") or {}).get("status")
    lake_status = (lake.get("supervision") or {}).get("status")
    window_status = (window.get("final_receipt") or {}).get("status")
    fusion_status = (_receipt(repo, "HCLI_QWEN38_FUSION_SOURCE_AUDIT.json") or {}).get("status")
    blockers = [
        "Flash-Next final promotion gate remains incomplete until every required byte/evidence field and both hard thresholds pass.",
        "Qwen27 current resident observation is a contaminated/contended regression audit, not a performance qualification.",
        "No physical FPGA board, bitstream, or hardware performance is claimed.",
    ]
    if lake_status not in {"PASSED", "READY", "COMPLETED"}:
        blockers.append("Flash-Next ModelLake acquisition is still partial or not yet atomically published.")
    if window_status != "PASSED":
        blockers.append("The corrected one-hour unattended window has not yet produced a final PASSED receipt.")
    if fusion_status != "PASSED":
        blockers.append("Qwen3.8 fusion source semantics are not yet resolved into a source-backed dispatch consequence.")
    qwen_mlp = _receipt(repo, "QWEN27_MLP_DIAGNOSTIC_AB.json") or {}
    if qwen_mlp.get("benchmark_class") != "QUALIFIED_PROTECTED":
        blockers.append("Qwen27 MLP selector result is diagnostic/contaminated or absent; it is not a protected performance qualification.")
    flash_executable = _receipt(repo, "FLASH_NEXT_NOETIC_EXECUTABLE.json") or {}
    if flash_executable.get("status") != "SCAFFOLD_ONLY" or flash_executable.get("promotion_allowed") is not False:
        blockers.append("Flash-Next noetic executable contract is missing an explicit scaffold/refusal boundary.")
    representation_experiment = _receipt(repo, "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json") or {}
    if representation_experiment.get("status") != "PASSED":
        blockers.append("Flash-Next source-layout representation experiment is absent or incomplete.")
    transform_parity = _receipt(repo, "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json") or {}
    if transform_parity.get("status") != "PASSED":
        blockers.append("Flash-Next full routed-expert transform parity is absent or incomplete.")
    loader_roundtrip = _receipt(repo, "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json") or {}
    if loader_roundtrip.get("status") != "PASSED":
        blockers.append("Flash-Next bounded noetic loader round-trip is absent or incomplete.")
    body_kernel_path = repo / "receipts" / "headless" / "FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json"
    legacy_kernel_path = repo / "receipts" / "headless" / "FLASH_NOETIC_Q4_KERNEL_PARITY.json"
    kernel_path = body_kernel_path if body_kernel_path.is_file() else legacy_kernel_path
    kernel_parity = _read_object(kernel_path) or {}
    if kernel_parity.get("status") != "PASSED":
        blockers.append("Flash-Next bounded native noetic kernel parity is absent or incomplete.")
    component_body = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_BODY.json") or {}
    if component_body.get("status") != "PASSED" or component_body.get("source_independent") is not True or component_body.get("candidate_body_persisted") is not True:
        blockers.append("Flash-Next source-independent routed-expert component body is absent or incomplete.")
    component_campaign = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json") or {}
    if component_campaign.get("status") != "PASSED" or component_campaign.get("source_independent_execution") is not True or component_campaign.get("candidate_body_persisted") is not True:
        blockers.append("Flash-Next bounded multi-component Noetic campaign is absent or incomplete.")
    if router_body.get("status") != "PASSED" or router_body.get("source_independent") is not True or router_body.get("candidate_body_persisted") is not True:
        blockers.append("Flash-Next bounded source-independent router matrix body is absent or incomplete.")
    if router_graph.get("status") != "PASSED" or router_graph.get("promotion_allowed") is not False:
        blockers.append("Flash-Next bounded Noetic router graph is absent or incomplete.")
    if router_selection.get("status") != "PASSED" or router_selection.get("promotion_allowed") is not False:
        blockers.append("Flash-Next bounded Noetic router selection edge is absent or incomplete.")
    if router_selection_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next bounded native Noetic router selection receipt is invalid or incomplete.")
    if router_selection_native.get("status") == "PASSED" and router_selection_native.get("promotion_allowed") is not False:
        blockers.append("Flash-Next bounded native Noetic router selection has not explicitly refused promotion.")
    if routed_expert_dispatch_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next bounded native routed-expert dispatch receipt is invalid or incomplete.")
    if routed_expert_dispatch_native.get("status") == "PASSED" and (
        routed_expert_dispatch_native.get("native_routed_body_dispatch_observed") is not True
        or routed_expert_dispatch_native.get("promotion_allowed") is not False
    ):
        blockers.append("Flash-Next bounded native routed-expert dispatch has not explicitly proven scoped physical execution with promotion refused.")
    if gate_up_swiglu_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next bounded native gate/up SwiGLU receipt is invalid or incomplete.")
    if gate_up_swiglu_native.get("status") == "PASSED" and (
        gate_up_swiglu_native.get("native_gate_up_swiglu_observed") is not True
        or gate_up_swiglu_native.get("native_expert_gate_up_activation_observed") is not True
        or gate_up_swiglu_native.get("promotion_allowed") is not False
    ):
        blockers.append("Flash-Next bounded native gate/up SwiGLU has not explicitly proven scoped physical activation with promotion refused.")
    if expert_composition_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next bounded native gate/up-to-down expert composition receipt is invalid or incomplete.")
    if expert_composition_native.get("status") == "PASSED" and (
        expert_composition_native.get("native_gate_up_swiglu_observed") is not True
        or expert_composition_native.get("native_down_projection_observed") is not True
        or expert_composition_native.get("native_expert_composition_observed") is not True
        or expert_composition_native.get("promotion_allowed") is not False
    ):
        blockers.append("Flash-Next bounded native expert composition has not explicitly proven device-resident scoped execution with promotion refused.")
    if shared_expert_composition_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next bounded native shared-expert composition receipt is invalid or incomplete.")
    if shared_expert_composition_native.get("status") == "PASSED" and (
        shared_expert_composition_native.get("native_shared_expert_gate_up_swiglu_observed") is not True
        or shared_expert_composition_native.get("native_shared_expert_down_projection_observed") is not True
        or shared_expert_composition_native.get("native_shared_expert_scalar_gate_observed") is not True
        or shared_expert_composition_native.get("native_shared_expert_sigmoid_gate_observed") is not True
        or shared_expert_composition_native.get("native_shared_expert_composition_observed") is not True
        or shared_expert_composition_native.get("promotion_allowed") is not False
    ):
        blockers.append("Flash-Next bounded native shared-expert composition has not explicitly proven device-resident scoped execution with promotion refused.")
    if shared_residual_hyperconnection_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next bounded native shared-expert residual/hyperconnection receipt is invalid or incomplete.")
    if shared_residual_hyperconnection_native.get("status") == "PASSED" and (
        shared_residual_hyperconnection_native.get("native_hyperconnection_stream_injection_observed") is not True
        or shared_residual_hyperconnection_native.get("native_hyperconnection_low_rank_down_observed") is not True
        or shared_residual_hyperconnection_native.get("native_hyperconnection_low_rank_up_observed") is not True
        or shared_residual_hyperconnection_native.get("native_hyperconnection_block_inject_observed") is not True
        or shared_residual_hyperconnection_native.get("native_hyperconnection_residual_mix_observed") is not True
        or shared_residual_hyperconnection_native.get("native_shared_residual_composition_observed") is not True
        or shared_residual_hyperconnection_native.get("source_independent_execution") is not True
        or shared_residual_hyperconnection_native.get("device_intermediate_no_host_roundtrip") is not True
        or shared_residual_hyperconnection_native.get("promotion_allowed") is not False
    ):
        blockers.append("Flash-Next bounded native shared-expert residual/hyperconnection composition has not explicitly proven the device-resident candidate graph with promotion refused.")
    if exact_hyperconnection_native.get("status") not in {None, "PASSED"}:
        blockers.append("Flash-Next exact HyperConnection routed-plus-shared MoE receipt is invalid or incomplete.")
    if exact_hyperconnection_native.get("status") == "PASSED" and (
        exact_hyperconnection_native.get("complete_layer0_moe_candidate") is not True
        or exact_hyperconnection_native.get("complete_moe_combine") is not True
        or exact_hyperconnection_native.get("native_hyperconnection_read_observed") is not True
        or exact_hyperconnection_native.get("native_hyperconnection_write_observed") is not True
        or exact_hyperconnection_native.get("native_moe_weighted_sum_observed") is not True
        or exact_hyperconnection_native.get("native_moe_shared_add_observed") is not True
        or exact_hyperconnection_native.get("device_intermediate_no_host_roundtrip") is not True
        or exact_hyperconnection_native.get("promotion_allowed") is not False
    ):
        blockers.append("Flash-Next exact HyperConnection routed-plus-shared MoE receipt has not explicitly proven its device-resident scoped graph with promotion refused.")
    graph_component = _receipt(repo, "FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json") or {}
    if graph_component.get("status") != "PASSED" or graph_component.get("promotion_allowed") is not False:
        blockers.append("Flash-Next bounded Noetic routed-expert graph component is absent or incomplete.")
    protected_watch = _receipt(repo, "QWEN_PROTECTED_BENCH_READY.json") or {}
    if protected_watch.get("status") not in {"COMPLETED", "WAITING_FOR_QUIESCENCE"}:
        blockers.append("The bounded protected Qwen benchmark watcher is not in a safe waiting/completed state.")
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "generated_at": time.time(),
        "status": "READY_FOR_OVERNIGHT_CONTINUATION",
        "claim_boundary": "This handoff is a resumable status snapshot. It does not certify model quality, accelerator performance, Flash promotion, FPGA hardware, or sovereignty.",
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "baseline": {
            "git_revision": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "working_tree_status": (_git(repo, "status", "--porcelain=v1") or "").splitlines(),
            "baseline_commits": ["5a0c84d4ebde2687e60891829f81f47e06fecd3d", "9d373ecc4863b244fc74761c99fc837f7705f3db"],
            "unrelated_preserved_edit": "tools/odyssey_ctl.py",
        },
        "hcli": {
            "provider_neutral_semantics": "Mission/DAG owns work identity, dependencies, resources, stop conditions, checkpoints, receipts, and retry state; provider/model identity is execution policy.",
            "current_default_profile": str(repo / "hcli" / "hawking-native.sealed-3.14.json"),
            "explicit_selection_rule": "--model/config/env/provider selection wins; the resident profile is only the local default when no explicit selection exists.",
            "initial_charge": _charge_summary(repo),
            "background_jobs": jobs,
            "unattended_window": window,
            "transfer_map": str(repo / "receipts" / "headless" / "QWEN38_ACCELERATOR_TRANSFER_MAP.json"),
            "precedent_map": str(repo / "receipts" / "headless" / "FLASH_NEXT_PRECEDENT_MAP.json"),
            "dense_vs_nf_scaffold": str(repo / "receipts" / "headless" / "HCLI_DENSE_VS_NF_AB_SCAFFOLD.json"),
        },
        "qwen27": _qwen27_summary(repo),
        "flash_next": flash,
        "modellake": lake,
        "fpga": _fpga_summary(repo),
        "verification": {
            "full_suite": {"command": "pytest -q", "last_observed_status": "PASSED", "last_observed": "747 passed, 2 skipped"},
            "provider_focus": {"last_observed_status": "PASSED", "last_observed": "30 passed, 2 warnings"},
            "receipt_gates": {
                "autonomy": (_receipt(repo, "HCLI_AGENTOS_AUTONOMY_GATE.json") or {}).get("status"),
                "accelerator_regression": (_receipt(repo, "HCLI_ACCELERATOR_REGRESSION.json") or {}).get("status"),
                "qwen38_fusion_audit": fusion_status,
                "preboard": (_receipt(repo, "HCLI_AGENTOS_PREBOARD.json") or {}).get("status"),
                "flash_pre_runtime": (_receipt(repo, "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json") or {}).get("status"),
                "flash_representation_experiment": representation_experiment.get("status"),
                "flash_transform_parity": transform_parity.get("status"),
                "flash_loader_roundtrip": loader_roundtrip.get("status"),
                "flash_kernel_parity": kernel_parity.get("status"),
                "flash_component_body": component_body.get("status"),
                "flash_component_campaign": component_campaign.get("status"),
                "flash_graph_component": graph_component.get("status"),
                "flash_router_component_body": router_body.get("status"),
                "flash_router_graph": router_graph.get("status"),
                "flash_router_selection": router_selection.get("status"),
                "flash_router_selection_native": router_selection_native.get("status"),
                "flash_routed_expert_dispatch_native": routed_expert_dispatch_native.get("status"),
                "flash_gate_up_swiglu_native": gate_up_swiglu_native.get("status"),
                "flash_expert_composition_native": expert_composition_native.get("status"),
                "flash_shared_expert_composition_native": shared_expert_composition_native.get("status"),
                "flash_shared_residual_hyperconnection_native": shared_residual_hyperconnection_native.get("status"),
                "flash_exact_hyperconnection": exact_hyperconnection_native.get("status"),
                "flash_gpu_work_ledger": gpu_work_ledger.get("status"),
                "flash_token_critical_path": token_critical_path.get("status"),
                "cuda_capability_graph": cuda_capability_graph.get("status"),
                "flash_router_representation_ab": router_representation_ab.get("status"),
                "modellake_supervision": lake_status,
                "unattended_window": window_status,
            },
        },
        "blockers": blockers,
        "continuation": {
            "inspect_status": f"python3 -m hcli agentos status --workspace {repo} --repo-root {repo}",
            "refresh_checkpoint": f"python3 -m hcli agentos checkpoint --repo-root {repo} --workspace {repo} --emit {repo / 'receipts/headless/HCLI_AGENTOS_CHECKPOINT.json'}",
            "supervise_modellake": f"python3 -m hcli agentos modellake-supervise --repo-root {repo} --job-id {MODEL_LAKE_JOB} --emit {repo / 'receipts/headless/HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json'}",
            "modellake_job_status": f"python3 -m hcli agentos background status --workspace {repo} --repo-root {repo} {MODEL_LAKE_JOB}",
            "unattended_job_status": f"python3 -m hcli agentos background status --workspace {repo} --repo-root {repo} {UNATTENDED_JOB}",
            "flash_transform_job_status": f"python3 -m hcli agentos background status --workspace {repo} --repo-root {repo} {FLASH_TRANSFORM_JOB}",
            "refresh_initial_charge": f"python3 -m hcli agentos initial-charge --repo-root {repo} --workspace {repo / '.hcli/initial-charge'} --emit {repo / 'receipts/headless/HAWKING_INITIAL_CHARGE.json'}",
            "refresh_maps": f"python3 -m hcli agentos science-maps --repo-root {repo} --transfer-emit {repo / 'receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json'} --precedent-emit {repo / 'receipts/headless/FLASH_NEXT_PRECEDENT_MAP.json'}",
            "refresh_ab": f"python3 -m hcli agentos ab-scaffold --repo-root {repo} --emit {repo / 'receipts/headless/HCLI_DENSE_VS_NF_AB_SCAFFOLD.json'}",
            "refresh_fpga_preboard": f"python3 -m hcli agentos fpga-preboard --repo-root {repo} --emit {repo / 'receipts/headless/HCLI_FPGA_PREBOARD.json'}",
            "run_full_tests": "pytest -q",
            "run_qwen38_fusion_audit": f"python3 -m hcli agentos qwen38-fusion-audit --repo-root {repo} --profile {repo / 'hcli/hawking-native.sealed-3.14.json'} --emit {repo / 'receipts/headless/HCLI_QWEN38_FUSION_SOURCE_AUDIT.json'}",
            "run_qwen27_runtime_archaeology": f"python3 -m hcli agentos qwen27-runtime-archaeology --repo-root {repo} --profile {repo / 'hcli/hawking-native.sealed-3.14.json'}",
            "run_qwen27_mlp_diagnostic": f"python3 -m hcli agentos qwen27-mlp-ab --repo-root {repo} --profile {repo / 'hcli/hawking-native.sealed-3.14.json'} --resident-binary {repo / '.hcli/instrumented/ascension_qwen38_resident'}",
            "watch_protected_qwen_window": f"python3 -m hcli agentos protected-bench-watch --repo-root {repo} --profile {repo / 'hcli/hawking-native.sealed-3.14.json'} --resident-binary {repo / '.hcli/instrumented/ascension_qwen38_resident'} --duration-s 21600 --interval-s 60",
            "build_flash_executable_scaffold": f"python3 -m hcli agentos flash-executable --repo-root {repo}",
            "run_flash_component_body": f"python3 -m hcli agentos flash-component-body --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --repo-root {repo} --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_BODY.json'}",
            "run_flash_component_campaign": f"python3 -m hcli agentos flash-component-campaign --repo-root {repo} --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json'}",
            "run_flash_graph_component": f"python3 -m hcli agentos flash-graph-component --repo-root {repo} --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json'}",
            "run_flash_router_component_body": f"python3 -m hcli agentos flash-matrix-body --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --repo-root {repo} --tensor-name model.language_model.layers.0.mlp.gate.weight --component-kind router --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_BODY.json'}",
            "run_flash_router_graph": f"python3 -m hcli agentos flash-router-graph --repo-root {repo} --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_GRAPH.json'}",
            "run_flash_router_selection": f"python3 -m hcli agentos flash-router-selection --repo-root {repo} --body-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_FULL_BODY.json'} --kernel-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_FULL_KERNEL_PARITY.json'} --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json'}",
            "build_flash_native_router_selection": "cargo build -p hawking-core --release --example flash_noetic_router_selection",
            "run_flash_native_router_selection": f"{repo / 'workspace/ops/build/rust/release/examples/flash_noetic_router_selection'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --body-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_FULL_BODY.json'} --kernel-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_FULL_KERNEL_PARITY.json'} --reference-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json'} --warmup 2 --reps 7 --out {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json'}",
            "build_flash_native_routed_expert_dispatch": "cargo build -p hawking-core --release --example flash_noetic_routed_expert_dispatch",
            "run_flash_native_routed_expert_dispatch": f"{repo / 'workspace/ops/build/rust/release/examples/flash_noetic_routed_expert_dispatch'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --router-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json'} --campaign-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json'} --warmup 2 --reps 7 --out {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json'}",
            "run_flash_native_gate_up_swiglu": f"{repo / 'workspace/ops/build/rust/release/examples/flash_noetic_routed_expert_dispatch'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --router-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json'} --gate-up-swiglu --warmup 2 --reps 7 --out {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_GATE_UP_SWIGLU_NATIVE.json'}",
            "run_flash_native_expert_composition": f"{repo / 'workspace/ops/build/rust/release/examples/flash_noetic_routed_expert_dispatch'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --router-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json'} --expert-composition --warmup 2 --reps 7 --out {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPOSITION_NATIVE.json'}",
            "run_flash_native_shared_expert_composition": f"{repo / 'workspace/ops/build/rust/release/examples/flash_noetic_routed_expert_dispatch'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --shared-expert-composition --warmup 2 --reps 7 --out {repo / 'receipts/headless/FLASH_NOETIC_SHARED_EXPERT_COMPOSITION_NATIVE.json'}",
            "run_flash_native_shared_residual_composition": f"{repo / 'workspace/ops/build/rust/release/examples/flash_noetic_routed_expert_dispatch'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --shared-residual-composition --warmup 2 --reps 7 --out {repo / 'receipts/headless/FLASH_NOETIC_SHARED_RESIDUAL_HYPERCONNECTION_NATIVE.json'}",
            "run_flash_native_exact_hyperconnection": f"tools/gpu_lane_lock.sh flash-next-exact-layer0-moe {repo / 'workspace/ops/build/rust/release/examples/flash_noetic_routed_expert_dispatch'} --exact-hyperconnection-composition --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --warmup 2 --reps 3 --out {repo / 'receipts/headless/FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json'}",
            "run_flash_router_representation_ab": f"python3 -m hcli agentos flash-router-representation-ab --repo-root {repo} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --emit {repo / 'receipts/headless/FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json'}",
            "run_flash_tensor_probe": f"python3 -m hcli agentos flash-tensor-probe --emit {repo / 'receipts/headless/FLASH_FIRST_TENSOR_PROBE.json'}",
            "run_flash_representation_experiment": f"python3 -m hcli agentos flash-representation-experiment --emit {repo / 'receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json'}",
            "run_flash_representation_replication": f"python3 -m hcli agentos flash-representation-experiment --expert-indices 32,33,34,35,36,37,38,39 --row-start 64 --row-count 16 --emit {repo / 'receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT_DISJOINT.json'}",
            "run_flash_transform_parity": f"python3 -m hcli agentos flash-transform-parity --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --chunk-rows 128 --emit {repo / 'receipts/headless/FLASH_FULL_TENSOR_TRANSFORM_PARITY.json'}",
            "run_flash_bounded_loader_roundtrip": f"python3 -m hcli agentos flash-loader-roundtrip --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --repo-root {repo} --candidate independent_q4_g64 --emit {repo / 'receipts/headless/FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json'}",
            "run_flash_bounded_loader_roundtrip_quality_alternate": f"python3 -m hcli agentos flash-loader-roundtrip --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --repo-root {repo} --candidate shared_bf16_basis_nf4_residual --emit {repo / 'receipts/headless/FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP_SHARED.json'}",
            "run_flash_native_kernel_parity": f"{repo / '.hcli/flash-kernel-build/release-fast/release/examples/flash_noetic_q4_kernel_parity'} --root /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc --descriptor {repo / 'receipts/headless/FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json'} --candidate-body {repo / '.hcli/flash-component/flash-routed-expert-independent-q4-g64-e0-r0-128.bin'} --body-receipt {repo / 'receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_BODY.json'} --reps 7 --warmup 2 --out {repo / 'receipts/headless/FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json'}",
            "resume_modellake_only_if_interrupted": f"python3 -m hcli agentos background resume --workspace {repo} --repo-root {repo} {MODEL_LAKE_JOB}",
            "do_not_start": "Do not launch or promote a new Odyssey; continue only through the existing HCLI/ModelLake authorities and governed windows.",
        },
    }
    payload["receipt_path"] = str(destination)
    atomic_write_json(destination, payload)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    print(json.dumps(build_handoff(args.repo_root, emit=args.emit), indent=2, sort_keys=True, default=str))
    return 0


__all__ = ["DEFAULT_NAME", "SCHEMA", "build_handoff", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
