"""Program-level AgentOS checkpoint and maturity census.

This is deliberately a census, not a success stamp.  It records which
control-plane surfaces and receipts exist, separates fixture proof from
production qualification, and leaves explicit blockers for work that has not
been measured.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hcli.persist import atomic_write_json
from hcli.providers import (
    CAPABILITY_SCHEMA,
    FAILURE_SCHEMA,
    GENERATION_REQUEST_SCHEMA,
    GENERATION_RESPONSE_SCHEMA,
    HEALTH_SCHEMA,
    PROFILE_SCHEMA,
    RECEIPT_SCHEMA,
    ROLE_SCHEMA,
)
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.agentos.modellake_receipts import (
    CENSUS_RECEIPT_NAMES,
    SUPERVISION_RECEIPT_NAMES,
)
from hcli.tool_registry import default_tool_registry


SCHEMA = "hcli.agentos.program_checkpoint.v1"
DEFAULT_NAME = "HCLI_AGENTOS_CHECKPOINT.json"


def _read_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _git_revision(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (proc.stdout or "").strip()
    return value if proc.returncode == 0 and value else None


def _receipt_inventory(roots: Iterable[Path], *, limit: int = 200) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if len(rows) >= limit:
                break
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                rows.append({
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "suffix": path.suffix.lower(),
                })
            except OSError:
                continue
    return rows


def _gate_summary(workspace: Path, repo_root: Path) -> Dict[str, Any]:
    recovery_candidates = [
        workspace / ".hcli" / "receipts" / "recovery-gate.json",
        repo_root / "receipts" / "headless" / "recovery-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RECOVERY_GATE.json",
    ]
    research_candidates = [
        workspace / ".hcli" / "receipts" / "research-gate.json",
        repo_root / "receipts" / "headless" / "research-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RESEARCH_GATE.json",
    ]
    vmcp_candidates = [
        workspace / ".hcli" / "receipts" / "vmcp-gate.json",
        repo_root / "receipts" / "headless" / "vmcp-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_VMCP_GATE.json",
    ]
    native_candidates = [
        workspace / ".hcli" / "receipts" / "native-gate.json",
        repo_root / "receipts" / "headless" / "native-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_NATIVE_GATE.json",
    ]
    resident_candidates = [
        workspace / ".hcli" / "receipts" / "resident-gate.json",
        repo_root / "receipts" / "headless" / "resident-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RESIDENT_GATE.json",
    ]
    mission_candidates = [
        workspace / ".hcli" / "receipts" / "native-mission-gate.json",
        repo_root / "receipts" / "headless" / "native-mission-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_NATIVE_MISSION_GATE.json",
    ]
    accelerator_candidates = [
        workspace / ".hcli" / "receipts" / "accelerator-native-smoke.json",
        repo_root / "receipts" / "headless" / "accelerator-native-smoke.json",
        repo_root / "receipts" / "headless" / "HCLI_ACCELERATOR_NATIVE_SMOKE.json",
    ]
    autonomy_candidates = [
        workspace / ".hcli" / "receipts" / "autonomy-gate.json",
        repo_root / "receipts" / "headless" / "autonomy-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_AUTONOMY_GATE.json",
    ]
    unattended_candidates = [
        workspace / ".hcli" / "receipts" / "unattended-window.json",
        repo_root / "receipts" / "headless" / "unattended-window.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_UNATTENDED_WINDOW.json",
    ]
    accelerator_regression_candidates = [
        workspace / ".hcli" / "receipts" / "accelerator-regression.json",
        repo_root / "receipts" / "headless" / "accelerator-regression.json",
        repo_root / "receipts" / "headless" / "HCLI_ACCELERATOR_REGRESSION.json",
    ]
    fusion_source_audit_candidates = [
        workspace / ".hcli" / "receipts" / "qwen38-fusion-audit.json",
        repo_root / "receipts" / "headless" / "qwen38-fusion-audit.json",
        repo_root / "receipts" / "headless" / "HCLI_QWEN38_FUSION_SOURCE_AUDIT.json",
    ]
    modellake_candidates = [
        workspace / ".hcli" / "receipts" / "modellake-census.json",
        repo_root / "receipts" / "headless" / CENSUS_RECEIPT_NAMES[0],
        repo_root / "receipts" / "headless" / "modellake-census.json",
        repo_root / "receipts" / "headless" / CENSUS_RECEIPT_NAMES[1],
    ]
    flash_science_candidates = [
        workspace / ".hcli" / "receipts" / "flash-science.json",
        repo_root / "receipts" / "headless" / "flash-science.json",
        repo_root / "receipts" / "headless" / "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json",
    ]
    preboard_candidates = [
        workspace / ".hcli" / "receipts" / "preboard.json",
        repo_root / "receipts" / "headless" / "preboard.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_PREBOARD.json",
    ]
    charge_candidates = [
        workspace / ".hcli" / "receipts" / "initial-charge.json",
        repo_root / "receipts" / "headless" / "initial-charge.json",
        repo_root / "receipts" / "headless" / "HAWKING_INITIAL_CHARGE.json",
    ]
    transfer_map_candidates = [repo_root / "receipts" / "headless" / "QWEN38_ACCELERATOR_TRANSFER_MAP.json"]
    precedent_map_candidates = [repo_root / "receipts" / "headless" / "FLASH_NEXT_PRECEDENT_MAP.json"]
    ab_candidates = [repo_root / "receipts" / "headless" / "HCLI_DENSE_VS_NF_AB_SCAFFOLD.json"]
    fpga_candidates = [repo_root / "receipts" / "headless" / "HCLI_FPGA_PREBOARD.json"]
    lake_supervision_candidates = [
        repo_root / "receipts" / "headless" / SUPERVISION_RECEIPT_NAMES[0],
        repo_root / "receipts" / "headless" / SUPERVISION_RECEIPT_NAMES[1],
    ]
    qwen27_identity_candidates = [repo_root / "receipts" / "headless" / "QWEN27_HISTORICAL_RUNTIME_IDENTITY.json"]
    qwen27_diff_candidates = [repo_root / "receipts" / "headless" / "QWEN27_RUNTIME_DIFF.json"]
    qwen27_mlp_candidates = [repo_root / "receipts" / "headless" / "QWEN27_MLP_DIAGNOSTIC_AB.json"]
    protected_watch_candidates = [repo_root / "receipts" / "headless" / "QWEN_PROTECTED_BENCH_READY.json"]
    flash_executable_candidates = [repo_root / "receipts" / "headless" / "FLASH_NEXT_NOETIC_EXECUTABLE.json"]
    flash_ebpw_candidates = [repo_root / "receipts" / "headless" / "FLASH_EBPW_BUDGET.json"]
    flash_token_ns_candidates = [repo_root / "receipts" / "headless" / "FLASH_TOKEN_NS_BUDGET.json"]
    flash_tensor_probe_candidates = [repo_root / "receipts" / "headless" / "FLASH_FIRST_TENSOR_PROBE.json"]
    flash_representation_candidates = [repo_root / "receipts" / "headless" / "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json"]
    flash_transform_candidates = [repo_root / "receipts" / "headless" / "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"]
    flash_loader_candidates = [repo_root / "receipts" / "headless" / "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"]
    flash_kernel_candidates = [
        repo_root / "receipts" / "headless" / "FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json",
        repo_root / "receipts" / "headless" / "FLASH_NOETIC_Q4_KERNEL_PARITY.json",
    ]
    flash_body_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_BODY.json"]
    flash_graph_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json"]
    flash_campaign_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json"]
    flash_router_body_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_COMPONENT_BODY.json"]
    flash_router_graph_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_GRAPH.json"]
    flash_router_selection_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_SELECTION.json"]
    flash_router_selection_native_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json"]
    flash_routed_expert_dispatch_native_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json"]
    flash_gate_up_swiglu_native_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_GATE_UP_SWIGLU_NATIVE.json"]
    flash_expert_composition_native_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTED_EXPERT_COMPOSITION_NATIVE.json"]
    flash_shared_expert_composition_native_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_SHARED_EXPERT_COMPOSITION_NATIVE.json"]
    flash_shared_residual_hyperconnection_native_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_SHARED_RESIDUAL_HYPERCONNECTION_NATIVE.json"]
    flash_exact_hyperconnection_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json"]
    flash_gpu_work_ledger_candidates = [repo_root / "receipts" / "headless" / "FLASH_GPU_WORK_LEDGER.json"]
    flash_token_critical_path_candidates = [repo_root / "receipts" / "headless" / "FLASH_TOKEN_CRITICAL_PATH.json"]
    cuda_capability_graph_candidates = [repo_root / "receipts" / "headless" / "CUDA_CAPABILITY_GRAPH.json"]
    flash_router_representation_candidates = [repo_root / "receipts" / "headless" / "FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json"]
    recovery = None
    for path in recovery_candidates:
        value = _read_object(path)
        if value is not None:
            recovery = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
            }
            break
    research = None
    for path in research_candidates:
        value = _read_object(path)
        if value is not None:
            research = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
            }
            break
    vmcp = None
    for path in vmcp_candidates:
        value = _read_object(path)
        if value is not None:
            vmcp = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
            }
            break
    native = None
    for path in native_candidates:
        value = _read_object(path)
        if value is not None:
            native = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "profile_path": value.get("profile_path"),
            }
            break
    resident = None
    for path in resident_candidates:
        value = _read_object(path)
        if value is not None:
            resident = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "profile_path": value.get("profile_path"),
            }
            break
    mission = None
    for path in mission_candidates:
        value = _read_object(path)
        if value is not None:
            mission = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "profile_path": value.get("profile_path"),
            }
            break
    accelerator = None
    for path in accelerator_candidates:
        value = _read_object(path)
        if value is not None:
            accelerator = {
                "status": "PASSED" if value.get("pass") is True else "FAILED",
                "qualification": "LIVE_ACCELERATOR_EXECUTION_NO_PERF_CLAIM",
                "receipt_path": str(path),
                "pass": value.get("pass"),
                "bench_state": (value.get("bench") or {}).get("state")
                if isinstance(value.get("bench"), dict) else None,
            }
            break
    autonomy = None
    for path in autonomy_candidates:
        value = _read_object(path)
        if value is not None:
            autonomy = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "stage_status": value.get("stage_status"),
            }
            break
    unattended = None
    for path in unattended_candidates:
        value = _read_object(path)
        if value is not None:
            unattended = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "metrics": value.get("metrics"),
                "duration_requested_s": value.get("duration_requested_s"),
                "elapsed_s": value.get("elapsed_s"),
            }
            break
    accelerator_regression = None
    for path in accelerator_regression_candidates:
        value = _read_object(path)
        if value is not None:
            accelerator_regression = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "bench_state": ((value.get("experiment") or {}).get("bench") or {}).get("state")
                if isinstance(value.get("experiment"), dict)
                else None,
                "perf_qualified": ((value.get("experiment") or {}).get("perf_qualified"))
                if isinstance(value.get("experiment"), dict)
                else None,
            }
            break
    fusion_source_audit = None
    for path in fusion_source_audit_candidates:
        value = _read_object(path)
        if value is not None:
            fusion_source_audit = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "selected_graph": value.get("selected_graph"),
                "result": value.get("result"),
            }
            break
    modellake = None
    for path in modellake_candidates:
        value = _read_object(path)
        if value is not None:
            modellake = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "download_performed": (value.get("acquisition_policy") or {}).get("download_performed")
                if isinstance(value.get("acquisition_policy"), dict)
                else None,
                "pinned_revision": (value.get("source") or {}).get("requested_revision")
                if isinstance(value.get("source"), dict)
                else None,
            }
            break
    flash_science = None
    for path in flash_science_candidates:
        value = _read_object(path)
        if value is not None:
            flash_science = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "promotion_gate": value.get("promotion_gate"),
                "architecture_fingerprint": value.get("architecture_fingerprint"),
            }
            break
    preboard = None
    for path in preboard_candidates:
        value = _read_object(path)
        if value is not None:
            preboard = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "claim_boundary": value.get("claim_boundary"),
            }
            break
    charge = None
    for path in charge_candidates:
        value = _read_object(path)
        if value is not None:
            charge = {
                "status": value.get("status"),
                "charge_id": value.get("charge_id"),
                "receipt_path": str(path),
                "mission_id": value.get("mission_id"),
                "workspace": value.get("workspace"),
                "unit_count": len(value.get("units") or []) if isinstance(value.get("units"), list) else None,
                "provider_neutral": value.get("provider_neutral"),
            }
            break
    transfer_map = None
    for path in transfer_map_candidates:
        value = _read_object(path)
        if value is not None:
            transfer_map = {
                "status": "PRESENT",
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "fingerprint": value.get("fingerprint"),
                "entries": len(value.get("transfer_matrix") or []) if isinstance(value.get("transfer_matrix"), list) else None,
            }
            break
    precedent_map = None
    for path in precedent_map_candidates:
        value = _read_object(path)
        if value is not None:
            precedent_map = {
                "status": "PRESENT",
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "fingerprint": value.get("fingerprint"),
                "entries": len(value.get("entries") or []) if isinstance(value.get("entries"), list) else None,
            }
            break
    ab_scaffold = None
    for path in ab_candidates:
        value = _read_object(path)
        if value is not None:
            ab_scaffold = {
                "status": value.get("status"),
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "evaluation": value.get("evaluation"),
            }
            break
    fpga = None
    for path in fpga_candidates:
        value = _read_object(path)
        if value is not None:
            fpga = {
                "status": value.get("status"),
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "fingerprint": value.get("fingerprint"),
                "checks": value.get("checks"),
                "physical_board": value.get("physical_board"),
            }
            break
    lake_supervision = None
    for path in lake_supervision_candidates:
        value = _read_object(path)
        if value is not None:
            lake_supervision = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "job": value.get("job"),
                "target": value.get("target"),
                "capacity": value.get("capacity"),
            }
            break
    def _science_receipt(candidates: list[Path], *, fields: tuple[str, ...] = ()) -> Optional[Dict[str, Any]]:
        for path in candidates:
            value = _read_object(path)
            if value is not None:
                return {
                    "status": value.get("status"),
                    "receipt_path": str(path),
                    "schema": value.get("schema"),
                    **{field: value.get(field) for field in fields},
                }
        return None

    qwen27_identity = _science_receipt(qwen27_identity_candidates, fields=("historical_selection", "checks"))
    qwen27_diff = _science_receipt(qwen27_diff_candidates, fields=("summary", "classification_policy"))
    qwen27_mlp = _science_receipt(qwen27_mlp_candidates, fields=("benchmark_class", "qualification", "NOT_FOR_PROMOTION", "experiment_verdict", "selector_verdict", "checks"))
    protected_watch = _science_receipt(protected_watch_candidates, fields=("qualification", "NOT_FOR_PROMOTION", "runs", "last_poll"))
    flash_executable = _science_receipt(flash_executable_candidates, fields=("status", "qualification", "NOT_FOR_PROMOTION", "promotion_allowed", "native_loader", "native_kernels", "complete_token_timing", "runtime_genome", "source_representation_experiment", "source_router_graph", "source_router_selection", "source_router_selection_native", "source_routed_expert_composition_native", "source_router_representation_ab"))
    flash_ebpw = _science_receipt(flash_ebpw_candidates, fields=("status", "measured", "target_contract", "promotion_allowed"))
    flash_token_ns = _science_receipt(flash_token_ns_candidates, fields=("status", "system_ledger", "target_contract", "promotion_allowed"))
    flash_tensor_probe = _science_receipt(flash_tensor_probe_candidates, fields=("source_label", "candidate_label", "source_tensor", "organ", "dense_vs_packed_low_bit", "next_experiment", "body_mutated", "model_loaded"))
    flash_representation = _science_receipt(flash_representation_candidates, fields=("source_label", "candidate_label", "source_tensor", "candidates", "comparison", "body_mutated", "model_loaded", "whole_model_capability", "whole_model_runtime", "next_experiment", "replications"))
    flash_transform = _science_receipt(flash_transform_candidates, fields=("source_label", "candidate_label", "source_tensor", "candidates", "comparison", "transform_parity", "body_mutated", "model_loaded", "whole_model_capability", "whole_model_runtime", "next_experiment", "claim_boundary"))
    flash_loader = _science_receipt(flash_loader_candidates, fields=("source_label", "candidate_label", "candidate_id", "transform_reference", "representation_descriptor", "serialized_descriptor_sha256", "encoded_sample", "loader_roundtrip", "body_mutated", "model_loaded", "whole_model_capability", "whole_model_runtime", "native_loader", "next_experiment", "claim_boundary"))
    flash_kernel = _science_receipt(flash_kernel_candidates, fields=("source_label", "derived_label", "model_lake_manifest", "source_tensor", "noetic_descriptor", "noetic_representation", "native_loader", "native_kernel", "gpu_timing", "parity", "body_mutated", "model_loaded", "complete_system_ebpw", "flash_tps", "promotion_allowed", "claim_boundary", "next_action"))
    flash_body = _science_receipt(flash_body_candidates, fields=("source_identity", "source_block", "representation_descriptor", "body", "native_loader", "source_guard", "source_independent", "candidate_body_persisted", "whole_model_capability", "whole_model_runtime", "complete_system_ebpw", "flash_tps", "promotion_allowed", "claim_boundary", "next_action"))
    flash_graph = _science_receipt(flash_graph_candidates, fields=("component_status", "semantic_type", "compiler_stage", "candidate_id", "source_identity", "source_backed", "candidate_body_persisted", "whole_model_capability", "complete_token_runtime", "physical_graph", "noetic_ir", "graph_fingerprint", "promotion_allowed", "claim_boundary", "next_action"))
    flash_campaign = _science_receipt(flash_campaign_candidates, fields=("component_status", "semantic_type", "compiler_stage", "candidate_id", "source_identity", "component_count", "component_windows", "components", "source_independent_execution", "candidate_body_persisted", "whole_model_capability", "complete_token_runtime", "physical_graph", "noetic_ir", "promotion_allowed", "claim_boundary", "next_action"))
    flash_router_body = _science_receipt(flash_router_body_candidates, fields=("component_kind", "source_identity", "source_block", "representation_descriptor", "body", "native_loader", "source_guard", "source_independent", "candidate_body_persisted", "whole_model_capability", "whole_model_runtime", "promotion_allowed", "claim_boundary", "next_action"))
    flash_router_graph = _science_receipt(flash_router_graph_candidates, fields=("component_status", "semantic_type", "compiler_stage", "candidate_id", "source_identity", "component_window", "source_independent_execution", "candidate_body_persisted", "whole_model_capability", "complete_token_runtime", "physical_graph", "noetic_ir", "promotion_allowed", "claim_boundary", "next_action"))
    flash_router_selection = _science_receipt(flash_router_selection_candidates, fields=("semantic_type", "compiler_stage", "source_identity", "config", "selection", "source_selection", "source_selection_parity", "source_reference_execution", "execution", "physical_graph", "noetic_ir", "native_selection_execution_observed", "whole_model_capability", "complete_token_runtime", "promotion_allowed", "claim_boundary", "next_action"))
    flash_router_selection_native = _science_receipt(flash_router_selection_native_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "body_receipt", "kernel_receipt", "source_block", "candidate_body", "noetic_representation", "native_loader", "native_kernel", "native_source_authority_kernel", "execution", "input", "selection_config", "selection", "source_native_selection", "reference", "source_selection_parity", "source_reference_parity", "parity", "source_native_parity", "gpu_timing", "source_gpu_timing", "noetic_ir", "physical_graph", "native_selection_execution_observed", "native_source_authority_execution_observed", "source_payload_exact", "source_guard_unchanged", "whole_model_capability", "complete_token_runtime", "promotion_allowed", "claim_boundary", "next_action"))
    flash_routed_expert_dispatch_native = _science_receipt(flash_routed_expert_dispatch_native_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "router_receipt", "campaign_receipt", "selection", "source_selection_parity", "components", "execution", "input", "gpu_timing", "gather", "noetic_ir", "physical_graph", "native_routed_body_dispatch_observed", "whole_model_capability", "complete_expert_runtime", "complete_token_runtime", "promotion_allowed", "claim_boundary", "next_action"))
    flash_gate_up_swiglu_native = _science_receipt(flash_gate_up_swiglu_native_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "router_receipt", "component_receipt_policy", "selection", "source_selection_parity", "components", "execution", "input", "gpu_timing", "gather", "noetic_ir", "physical_graph", "native_gate_up_swiglu_observed", "native_expert_gate_up_activation_observed", "whole_model_capability", "complete_expert_runtime", "complete_token_runtime", "promotion_allowed", "claim_boundary", "next_action"))
    flash_expert_composition_native = _science_receipt(flash_expert_composition_native_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "router_receipt", "component_receipt_policy", "selection", "source_selection_parity", "components", "execution", "input", "intermediate", "gpu_timing", "gather", "noetic_ir", "physical_graph", "native_gate_up_swiglu_observed", "native_down_projection_observed", "native_expert_composition_observed", "bounded_selected_expert_output_observed", "whole_model_capability", "complete_expert_runtime", "complete_token_runtime", "complete_system_ebpw", "flash_tps", "promotion_allowed", "claim_boundary", "next_action"))
    flash_shared_expert_composition_native = _science_receipt(flash_shared_expert_composition_native_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "layer", "component_receipt_policy", "components", "execution", "input", "intermediates", "parity", "gpu_timing", "noetic_ir", "physical_graph", "native_shared_expert_gate_up_swiglu_observed", "native_shared_expert_down_projection_observed", "native_shared_expert_scalar_gate_observed", "native_shared_expert_sigmoid_gate_observed", "native_shared_expert_composition_observed", "whole_model_capability", "complete_expert_runtime", "complete_token_runtime", "complete_system_ebpw", "flash_tps", "promotion_allowed", "claim_boundary", "next_action"))
    flash_shared_residual_hyperconnection_native = _science_receipt(flash_shared_residual_hyperconnection_native_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "layer", "dependencies", "component_receipt_policy", "components", "execution", "input", "intermediates", "candidate_semantics", "parity", "gpu_timing", "noetic_ir", "physical_graph", "native_shared_expert_gate_up_swiglu_observed", "native_shared_expert_down_projection_observed", "native_shared_expert_sigmoid_gate_observed", "native_hyperconnection_stream_injection_observed", "native_hyperconnection_low_rank_down_observed", "native_hyperconnection_low_rank_up_observed", "native_hyperconnection_block_inject_observed", "native_hyperconnection_residual_mix_observed", "native_shared_residual_composition_observed", "device_intermediate_no_host_roundtrip", "source_independent_execution", "whole_model_capability", "complete_expert_runtime", "complete_token_runtime", "complete_system_ebpw", "flash_tps", "promotion_allowed", "claim_boundary", "next_action"))
    flash_exact_hyperconnection = _science_receipt(flash_exact_hyperconnection_candidates, fields=("semantic_type", "compiler_stage", "qualification", "repo", "pinned_revision", "root", "layer", "dependencies", "source_reference", "semantics", "parity", "gpu_timing", "noetic_ir", "physical_graph", "source_selection_parity", "routed_expert_count", "routed_expert_ids", "selected_weight_sum", "native_hyperconnection_read_observed", "native_hyperconnection_write_observed", "exact_hyperconnection_semantics_observed", "native_routed_expert_gate_up_swiglu_observed", "native_routed_expert_down_projection_observed", "native_moe_weighted_sum_observed", "native_moe_shared_add_observed", "device_intermediate_no_host_roundtrip", "source_independent_execution", "source_hc_norm_payload_exact", "complete_layer0_moe_candidate", "complete_moe_combine", "whole_model_capability", "complete_expert_runtime", "complete_token_runtime", "complete_system_ebpw", "flash_tps", "promotion_allowed", "claim_boundary", "next_action"))
    flash_gpu_work_ledger = _science_receipt(flash_gpu_work_ledger_candidates, fields=("qualification", "source_receipt", "device", "scope", "physical_graph_fingerprint", "measured_runs", "dispatches_per_graph", "graph_gpu_ns_median", "graph_host_wall_ns_median", "graph_wall_minus_gpu_ns_median", "stages", "device_intermediate_no_host_roundtrip", "complete_token_runtime", "flash_tps", "complete_system_ebpw", "promotion_allowed", "claim_boundary"))
    flash_token_critical_path = _science_receipt(flash_token_critical_path_candidates, fields=("source_receipt", "candidate_graph", "complete_token_runtime", "accepted_tokens", "complete_wall_ns_per_accepted_token", "flash_tps", "complete_system_ebpw", "promotion_allowed", "blockers", "claim_boundary"))
    cuda_capability_graph = _science_receipt(cuda_capability_graph_candidates, fields=("execution_device", "native_backend_observed", "cuda_execution_observed", "qualification", "nodes", "edges", "promotion_allowed", "claim_boundary"))
    flash_router_representation = _science_receipt(flash_router_representation_candidates, fields=("semantic_type", "compiler_stage", "source_identity", "config", "source_block", "source_selection", "candidates", "recommendation", "physical_graph", "noetic_ir", "validation", "candidate_bodies_persisted", "whole_model_capability", "complete_token_runtime", "promotion_allowed", "claim_boundary", "next_action"))
    return {
        "recovery_gate": {
            **(recovery or {
                "status": "NOT_RUN",
                "qualification": "NONE",
                "receipt_path": None,
                "checks": {},
            }),
        },
        "research_gate": research or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
        },
        "vmcp_gate": vmcp or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
        },
        "native_gate": native or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "profile_path": None,
        },
        "resident_gate": resident or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "profile_path": None,
        },
        "native_mission_gate": mission or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "profile_path": None,
        },
        "accelerator_smoke": accelerator or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "pass": None,
            "bench_state": None,
        },
        "autonomy_gate": autonomy or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "stage_status": {},
        },
        "unattended": unattended or "NOT_PROVEN",
        "accelerator_regression": accelerator_regression or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "bench_state": None,
            "perf_qualified": None,
        },
        "qwen38_fusion_audit": fusion_source_audit or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "selected_graph": None,
            "result": None,
        },
        "modellake": modellake or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "download_performed": None,
            "pinned_revision": None,
        },
        "flash_science": flash_science or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "promotion_gate": None,
            "architecture_fingerprint": None,
        },
        "preboard": preboard or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "claim_boundary": None,
        },
        "initial_charge": charge or {
            "status": "NOT_RUN",
            "charge_id": None,
            "receipt_path": None,
            "mission_id": None,
            "workspace": None,
            "unit_count": None,
            "provider_neutral": None,
        },
        "transfer_map": transfer_map or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "fingerprint": None, "entries": None},
        "precedent_map": precedent_map or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "fingerprint": None, "entries": None},
        "ab_scaffold": ab_scaffold or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "evaluation": None},
        "fpga_preboard": fpga or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "fingerprint": None, "checks": {}, "physical_board": None},
        "modellake_supervision": lake_supervision or {"status": "NOT_RUN", "qualification": "NONE", "receipt_path": None, "checks": {}, "job": None, "target": None, "capacity": None},
        "qwen27_runtime_identity": qwen27_identity or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "historical_selection": None, "checks": {}},
        "qwen27_runtime_diff": qwen27_diff or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "summary": None, "classification_policy": None},
        "qwen27_mlp_diagnostic": qwen27_mlp or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "benchmark_class": None, "qualification": None, "NOT_FOR_PROMOTION": None, "experiment_verdict": None, "selector_verdict": None, "checks": {}},
        "protected_benchmark_watcher": protected_watch or {"status": "NOT_RUN", "receipt_path": None, "qualification": None, "NOT_FOR_PROMOTION": None, "runs": [], "last_poll": None},
        "flash_executable": flash_executable or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "qualification": None, "NOT_FOR_PROMOTION": None, "promotion_allowed": None, "native_loader": None, "native_kernels": None, "complete_token_timing": None, "runtime_genome": None},
        "flash_ebpw_budget": flash_ebpw or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "measured": None, "target_contract": None, "promotion_allowed": None},
        "flash_token_ns_budget": flash_token_ns or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "system_ledger": None, "target_contract": None, "promotion_allowed": None},
        "flash_tensor_probe": flash_tensor_probe or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_label": None, "candidate_label": None, "source_tensor": None, "organ": None, "dense_vs_packed_low_bit": None, "next_experiment": None, "body_mutated": None, "model_loaded": None},
        "flash_representation_experiment": flash_representation or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_label": None, "candidate_label": None, "source_tensor": None, "candidates": None, "comparison": None, "body_mutated": None, "model_loaded": None, "whole_model_capability": None, "whole_model_runtime": None, "replications": None},
        "flash_transform_parity": flash_transform or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_label": None, "candidate_label": None, "source_tensor": None, "candidates": None, "comparison": None, "transform_parity": None, "body_mutated": None, "model_loaded": None, "whole_model_capability": None, "whole_model_runtime": None, "next_experiment": None, "claim_boundary": None},
        "flash_loader_roundtrip": flash_loader or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_label": None, "candidate_label": None, "candidate_id": None, "transform_reference": None, "representation_descriptor": None, "serialized_descriptor_sha256": None, "encoded_sample": None, "loader_roundtrip": None, "body_mutated": None, "model_loaded": None, "whole_model_capability": None, "whole_model_runtime": None, "native_loader": None, "next_experiment": None, "claim_boundary": None},
        "flash_kernel_parity": flash_kernel or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_label": None, "derived_label": None, "model_lake_manifest": None, "source_tensor": None, "noetic_descriptor": None, "noetic_representation": None, "native_loader": None, "native_kernel": None, "gpu_timing": None, "parity": None, "body_mutated": None, "model_loaded": None, "complete_system_ebpw": None, "flash_tps": None, "promotion_allowed": None, "claim_boundary": None, "next_action": None},
        "flash_graph_component": flash_graph or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "component_status": "NOT_COMPILED", "semantic_type": None, "compiler_stage": None, "candidate_id": None, "source_identity": None, "source_backed": None, "candidate_body_persisted": None, "whole_model_capability": None, "complete_token_runtime": None, "physical_graph": None, "noetic_ir": None, "graph_fingerprint": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_component_body": flash_body or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_identity": None, "source_block": None, "representation_descriptor": None, "body": None, "native_loader": None, "source_guard": None, "source_independent": None, "candidate_body_persisted": None, "whole_model_capability": None, "whole_model_runtime": None, "complete_system_ebpw": None, "flash_tps": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_component_campaign": flash_campaign or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "component_status": "NOT_COMPILED", "semantic_type": None, "compiler_stage": None, "candidate_id": None, "source_identity": None, "component_count": 0, "component_windows": [], "components": [], "source_independent_execution": None, "candidate_body_persisted": None, "whole_model_capability": None, "complete_token_runtime": None, "physical_graph": None, "noetic_ir": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_router_component_body": flash_router_body or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "component_kind": None, "source_identity": None, "source_block": None, "representation_descriptor": None, "body": None, "native_loader": None, "source_guard": None, "source_independent": None, "candidate_body_persisted": None, "whole_model_capability": None, "whole_model_runtime": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_router_graph": flash_router_graph or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "component_status": "NOT_COMPILED", "semantic_type": None, "compiler_stage": None, "candidate_id": None, "source_identity": None, "component_window": None, "source_independent_execution": None, "candidate_body_persisted": None, "whole_model_capability": None, "complete_token_runtime": None, "physical_graph": None, "noetic_ir": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_router_selection": flash_router_selection or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "compiler_stage": None, "source_identity": None, "config": None, "selection": None, "execution": None, "physical_graph": None, "noetic_ir": None, "native_selection_execution_observed": None, "whole_model_capability": None, "complete_token_runtime": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_router_selection_native": flash_router_selection_native or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "compiler_stage": None, "qualification": None, "native_loader": None, "native_kernel": None, "native_source_authority_kernel": None, "execution": None, "selection": None, "source_native_selection": None, "reference": None, "source_selection_parity": None, "source_reference_parity": None, "parity": None, "source_native_parity": None, "gpu_timing": None, "source_gpu_timing": None, "physical_graph": None, "native_selection_execution_observed": None, "native_source_authority_execution_observed": None, "source_payload_exact": None, "source_guard_unchanged": None, "whole_model_capability": None, "complete_token_runtime": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_routed_expert_dispatch_native": flash_routed_expert_dispatch_native or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "qualification": None, "router_receipt": None, "campaign_receipt": None, "selection": None, "source_selection_parity": None, "components": [], "execution": None, "input": None, "gpu_timing": None, "gather": None, "noetic_ir": None, "physical_graph": None, "native_routed_body_dispatch_observed": None, "source_independent_execution": None, "whole_model_capability": None, "complete_expert_runtime": None, "complete_token_runtime": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_gate_up_swiglu_native": flash_gate_up_swiglu_native or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "qualification": None, "router_receipt": None, "component_receipt_policy": None, "selection": None, "source_selection_parity": None, "components": [], "execution": None, "input": None, "gpu_timing": None, "gather": None, "noetic_ir": None, "physical_graph": None, "native_gate_up_swiglu_observed": None, "native_expert_gate_up_activation_observed": None, "source_independent_execution": None, "whole_model_capability": None, "complete_expert_runtime": None, "complete_token_runtime": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_expert_composition_native": flash_expert_composition_native or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "qualification": None, "router_receipt": None, "component_receipt_policy": None, "selection": None, "source_selection_parity": None, "components": [], "execution": None, "input": None, "intermediate": None, "gpu_timing": None, "gather": None, "noetic_ir": None, "physical_graph": None, "native_gate_up_swiglu_observed": None, "native_down_projection_observed": None, "native_expert_composition_observed": None, "bounded_selected_expert_output_observed": None, "source_independent_execution": None, "whole_model_capability": None, "complete_expert_runtime": None, "complete_token_runtime": None, "complete_system_ebpw": None, "flash_tps": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_shared_expert_composition_native": flash_shared_expert_composition_native or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "qualification": None, "layer": None, "component_receipt_policy": None, "components": [], "execution": None, "input": None, "intermediates": None, "parity": None, "gpu_timing": None, "noetic_ir": None, "physical_graph": None, "native_shared_expert_gate_up_swiglu_observed": None, "native_shared_expert_down_projection_observed": None, "native_shared_expert_scalar_gate_observed": None, "native_shared_expert_sigmoid_gate_observed": None, "native_shared_expert_composition_observed": None, "source_independent_execution": None, "whole_model_capability": None, "complete_expert_runtime": None, "complete_token_runtime": None, "complete_system_ebpw": None, "flash_tps": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_shared_residual_hyperconnection_native": flash_shared_residual_hyperconnection_native or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "qualification": None, "layer": None, "dependencies": None, "component_receipt_policy": None, "components": [], "execution": None, "input": None, "intermediates": None, "candidate_semantics": None, "parity": None, "gpu_timing": None, "noetic_ir": None, "physical_graph": None, "native_shared_expert_gate_up_swiglu_observed": None, "native_shared_expert_down_projection_observed": None, "native_shared_expert_sigmoid_gate_observed": None, "native_hyperconnection_stream_injection_observed": None, "native_hyperconnection_low_rank_down_observed": None, "native_hyperconnection_low_rank_up_observed": None, "native_hyperconnection_block_inject_observed": None, "native_hyperconnection_residual_mix_observed": None, "native_shared_residual_composition_observed": None, "device_intermediate_no_host_roundtrip": None, "source_independent_execution": None, "whole_model_capability": None, "complete_expert_runtime": None, "complete_token_runtime": None, "complete_system_ebpw": None, "flash_tps": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_exact_hyperconnection": flash_exact_hyperconnection or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "qualification": None, "layer": None, "dependencies": None, "source_reference": None, "semantics": None, "parity": None, "gpu_timing": None, "noetic_ir": None, "physical_graph": None, "source_selection_parity": None, "routed_expert_count": None, "routed_expert_ids": None, "selected_weight_sum": None, "native_hyperconnection_read_observed": None, "native_hyperconnection_write_observed": None, "exact_hyperconnection_semantics_observed": None, "native_routed_expert_gate_up_swiglu_observed": None, "native_routed_expert_down_projection_observed": None, "native_moe_weighted_sum_observed": None, "native_moe_shared_add_observed": None, "device_intermediate_no_host_roundtrip": None, "source_independent_execution": None, "source_hc_norm_payload_exact": None, "complete_layer0_moe_candidate": None, "complete_moe_combine": None, "whole_model_capability": None, "complete_expert_runtime": None, "complete_token_runtime": None, "complete_system_ebpw": None, "flash_tps": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "flash_gpu_work_ledger": flash_gpu_work_ledger or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "qualification": None, "source_receipt": None, "device": None, "scope": None, "physical_graph_fingerprint": None, "measured_runs": None, "dispatches_per_graph": None, "graph_gpu_ns_median": None, "graph_host_wall_ns_median": None, "graph_wall_minus_gpu_ns_median": None, "stages": [], "device_intermediate_no_host_roundtrip": None, "complete_token_runtime": None, "flash_tps": None, "complete_system_ebpw": None, "promotion_allowed": False, "claim_boundary": None},
        "flash_token_critical_path": flash_token_critical_path or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "source_receipt": None, "candidate_graph": None, "complete_token_runtime": None, "accepted_tokens": None, "complete_wall_ns_per_accepted_token": None, "flash_tps": None, "complete_system_ebpw": None, "promotion_allowed": False, "blockers": [], "claim_boundary": None},
        "cuda_capability_graph": cuda_capability_graph or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "execution_device": None, "native_backend_observed": None, "cuda_execution_observed": None, "qualification": None, "nodes": [], "edges": [], "promotion_allowed": False, "claim_boundary": None},
        "flash_router_representation_ab": flash_router_representation or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "semantic_type": None, "compiler_stage": None, "source_identity": None, "config": None, "source_block": None, "source_selection": None, "candidates": [], "recommendation": None, "physical_graph": None, "noetic_ir": None, "validation": None, "candidate_bodies_persisted": None, "whole_model_capability": None, "complete_token_runtime": None, "promotion_allowed": False, "claim_boundary": None, "next_action": None},
        "production_provider_gate": "NOT_RUN",
    }


def build_program_checkpoint(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
    network: bool = False,
) -> Dict[str, Any]:
    repo = Path(repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    ws = Path(workspace or repo).expanduser().resolve()
    registry = default_tool_registry(ws, repo_root=repo)
    from hcli.connectivity import probe_connectivity

    connectivity = probe_connectivity(repo, workspace=ws, network=network)
    gates = _gate_summary(ws, repo)
    blockers = [
        "unattended production-provider continuation has not been proven",
        "accelerator performance qualification and unattended sovereign-resident operation remain unproven",
    ]
    if gates["recovery_gate"]["status"] == "PASSED":
        blockers = [item for item in blockers if "continuation" not in item]
    if gates["research_gate"]["status"] != "PASSED":
        blockers.append("public research operational gate has not passed")
    if gates["vmcp_gate"]["status"] != "PASSED":
        blockers.append("VMCP operational evidence-boundary gate has not passed")
    if gates["native_gate"]["status"] != "PASSED":
        blockers.append("live native HCLI A1-A6 ladder has not passed")
    if gates["resident_gate"]["status"] != "PASSED":
        blockers.append("20-request native resident proof has not passed")
    if gates["native_mission_gate"]["status"] != "PASSED":
        blockers.append("native tool/verifier mission gate has not passed")
    if gates["autonomy_gate"]["status"] != "PASSED":
        blockers.append("A1-A5 AgentOS autonomy and crash-recovery qualification has not passed")
    if gates["accelerator_smoke"]["status"] != "PASSED":
        blockers.append("live native accelerator smoke receipt has not passed")
    if gates["accelerator_regression"]["status"] != "PASSED":
        blockers.append("current-vs-historical accelerator regression audit has not passed")
    if gates["qwen38_fusion_audit"]["status"] != "PASSED":
        blockers.append("Qwen3.8 fusion source semantics and dispatch consequence audit has not passed")
    if gates["modellake"]["status"] != "PASSED":
        blockers.append("pinned Flash-Next ModelLake census has not passed")
    if gates["flash_science"]["status"] != "PASSED":
        blockers.append("Flash-Next pre-runtime architecture/organ science has not passed")
    if gates["preboard"]["status"] != "PASSED":
        blockers.append("negative-science/FPGA compiler preboard has not passed")
    if gates["initial_charge"]["status"] not in {"CREATED", "IDEMPOTENT_EXISTING_CHARGE"}:
        blockers.append("provider-neutral Hawking initial charge has not been persisted")
    if gates["transfer_map"]["status"] != "PRESENT":
        blockers.append("two-Qwen accelerator transfer map has not been persisted")
    if gates["precedent_map"]["status"] != "PRESENT":
        blockers.append("Flash-Next precedent map has not been persisted")
    if gates["ab_scaffold"]["status"] != "READY_SCAFFOLD":
        blockers.append("dense-vs-NF A/B scaffold has not been persisted")
    if gates["fpga_preboard"]["status"] != "PASSED":
        blockers.append("two-model FPGA preboard maps have not passed")
    if gates["modellake_supervision"]["status"] not in {"RUNNING_SAFE", "PASSED", "WAITING_OR_NOT_OBSERVED"}:
        blockers.append("pinned Flash-Next ModelLake acquisition is not in a safe observed state")
    flash_promotion = gates["flash_science"].get("promotion_gate")
    if isinstance(flash_promotion, dict) and flash_promotion.get("status") != "PROMOTABLE":
        blockers.append("Flash-Next final promotion gate is not PROMOTABLE (complete EBPW/TPS or required evidence is missing)")
    if gates["flash_executable"].get("status") not in {"PASSED", "SCAFFOLD_ONLY"}:
        blockers.append("Flash-Next noetic executable contract/scaffold has not been persisted")
    elif gates["flash_executable"].get("promotion_allowed") is not False:
        blockers.append("Flash-Next executable scaffold has not explicitly refused promotion")
    if gates["flash_tensor_probe"].get("status") != "PASSED":
        blockers.append("bounded Flash-Next source-tensor representation probe has not been persisted")
    if gates["flash_representation_experiment"].get("status") != "PASSED":
        blockers.append("bounded Flash-Next source-layout representation experiment has not been persisted")
    if gates["flash_transform_parity"].get("status") != "PASSED":
        blockers.append("full routed-expert Flash-Next transform parity has not been persisted")
    if gates["flash_loader_roundtrip"].get("status") != "PASSED":
        blockers.append("bounded Flash-Next noetic loader round-trip has not been persisted")
    if gates["flash_kernel_parity"].get("status") != "PASSED":
        blockers.append("bounded Flash-Next native noetic kernel parity has not been persisted")
    if gates["flash_component_body"].get("status") != "PASSED" or gates["flash_component_body"].get("source_independent") is not True or gates["flash_component_body"].get("candidate_body_persisted") is not True:
        blockers.append("bounded Flash-Next source-independent component body has not been persisted")
    if gates["flash_component_campaign"].get("status") != "PASSED" or gates["flash_component_campaign"].get("source_independent_execution") is not True or gates["flash_component_campaign"].get("candidate_body_persisted") is not True:
        blockers.append("bounded Flash-Next multi-component Noetic campaign has not been compiled")
    if gates["flash_router_component_body"].get("status") != "PASSED" or gates["flash_router_component_body"].get("source_independent") is not True or gates["flash_router_component_body"].get("candidate_body_persisted") is not True:
        blockers.append("bounded Flash-Next source-independent router matrix body has not been persisted")
    if gates["flash_router_graph"].get("status") != "PASSED" or gates["flash_router_graph"].get("promotion_allowed") is not False:
        blockers.append("bounded Flash-Next Noetic router graph has not been compiled")
    if gates["flash_router_selection"].get("status") != "PASSED" or gates["flash_router_selection"].get("promotion_allowed") is not False:
        blockers.append("bounded Flash-Next Noetic router selection edge has not been executed")
    if gates["flash_router_selection_native"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next native Noetic router selection receipt is invalid or incomplete")
    if gates["flash_router_selection_native"].get("status") == "PASSED" and gates["flash_router_selection_native"].get("promotion_allowed") is not False:
        blockers.append("bounded Flash-Next native Noetic router selection has not explicitly refused promotion")
    if gates["flash_routed_expert_dispatch_native"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next native routed-expert dispatch receipt is invalid or incomplete")
    if gates["flash_routed_expert_dispatch_native"].get("status") == "PASSED" and (
        gates["flash_routed_expert_dispatch_native"].get("native_routed_body_dispatch_observed") is not True
        or gates["flash_routed_expert_dispatch_native"].get("promotion_allowed") is not False
    ):
        blockers.append("bounded Flash-Next native routed-expert dispatch has not proven physical scoped execution with promotion refused")
    if gates["flash_gate_up_swiglu_native"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next native gate/up SwiGLU receipt is invalid or incomplete")
    if gates["flash_gate_up_swiglu_native"].get("status") == "PASSED" and (
        gates["flash_gate_up_swiglu_native"].get("native_gate_up_swiglu_observed") is not True
        or gates["flash_gate_up_swiglu_native"].get("native_expert_gate_up_activation_observed") is not True
        or gates["flash_gate_up_swiglu_native"].get("promotion_allowed") is not False
    ):
        blockers.append("bounded Flash-Next native gate/up SwiGLU has not proven physical scoped activation with promotion refused")
    if gates["flash_expert_composition_native"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next native gate/up-to-down expert composition receipt is invalid or incomplete")
    if gates["flash_expert_composition_native"].get("status") == "PASSED" and (
        gates["flash_expert_composition_native"].get("native_gate_up_swiglu_observed") is not True
        or gates["flash_expert_composition_native"].get("native_down_projection_observed") is not True
        or gates["flash_expert_composition_native"].get("native_expert_composition_observed") is not True
        or gates["flash_expert_composition_native"].get("promotion_allowed") is not False
    ):
        blockers.append("bounded Flash-Next native gate/up-to-down composition has not proven device-resident scoped execution with promotion refused")
    if gates["flash_shared_expert_composition_native"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next native shared-expert composition receipt is invalid or incomplete")
    if gates["flash_shared_expert_composition_native"].get("status") == "PASSED" and (
        gates["flash_shared_expert_composition_native"].get("native_shared_expert_gate_up_swiglu_observed") is not True
        or gates["flash_shared_expert_composition_native"].get("native_shared_expert_down_projection_observed") is not True
        or gates["flash_shared_expert_composition_native"].get("native_shared_expert_scalar_gate_observed") is not True
        or gates["flash_shared_expert_composition_native"].get("native_shared_expert_sigmoid_gate_observed") is not True
        or gates["flash_shared_expert_composition_native"].get("native_shared_expert_composition_observed") is not True
        or gates["flash_shared_expert_composition_native"].get("promotion_allowed") is not False
    ):
        blockers.append("bounded Flash-Next native shared-expert composition has not proven device-resident scoped execution with promotion refused")
    if gates["flash_shared_residual_hyperconnection_native"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next native shared-expert residual/hyperconnection receipt is invalid or incomplete")
    if gates["flash_shared_residual_hyperconnection_native"].get("status") == "PASSED" and (
        gates["flash_shared_residual_hyperconnection_native"].get("native_hyperconnection_stream_injection_observed") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("native_hyperconnection_low_rank_down_observed") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("native_hyperconnection_low_rank_up_observed") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("native_hyperconnection_block_inject_observed") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("native_hyperconnection_residual_mix_observed") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("native_shared_residual_composition_observed") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("device_intermediate_no_host_roundtrip") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("source_independent_execution") is not True
        or gates["flash_shared_residual_hyperconnection_native"].get("promotion_allowed") is not False
    ):
        blockers.append("bounded Flash-Next native shared-expert residual/hyperconnection composition has not proven the device-resident candidate graph with promotion refused")
    if gates["flash_exact_hyperconnection"].get("status") not in {"NOT_RUN", "PASSED"}:
        blockers.append("bounded Flash-Next exact HyperConnection routed-plus-shared MoE receipt is invalid or incomplete")
    if gates["flash_exact_hyperconnection"].get("status") == "PASSED" and (
        gates["flash_exact_hyperconnection"].get("complete_layer0_moe_candidate") is not True
        or gates["flash_exact_hyperconnection"].get("complete_moe_combine") is not True
        or gates["flash_exact_hyperconnection"].get("native_hyperconnection_read_observed") is not True
        or gates["flash_exact_hyperconnection"].get("native_hyperconnection_write_observed") is not True
        or gates["flash_exact_hyperconnection"].get("native_moe_weighted_sum_observed") is not True
        or gates["flash_exact_hyperconnection"].get("native_moe_shared_add_observed") is not True
        or gates["flash_exact_hyperconnection"].get("device_intermediate_no_host_roundtrip") is not True
        or gates["flash_exact_hyperconnection"].get("promotion_allowed") is not False
    ):
        blockers.append("bounded Flash-Next exact HyperConnection routed-plus-shared MoE receipt has not proven its protected scoped graph with promotion refused")
    if gates["qwen27_runtime_identity"].get("status") != "PASSED":
        blockers.append("Qwen27 current-versus-historical runtime identity archaeology has not been persisted")
    if gates["qwen27_mlp_diagnostic"].get("status") != "PASSED":
        blockers.append("Qwen27 MLP selector diagnostic has not produced a complete receipt")
    if connectivity.get("surfaces", {}).get("modellake", {}).get("status") != "AVAILABLE":
        blockers.append("ModelLake is not mounted in this environment")
    vmcp = connectivity.get("surfaces", {}).get("vmcp", {})
    if vmcp.get("status") not in {"AVAILABLE", "AUTHENTICATED"}:
        blockers.append("VMCP public surface is not fully importable/selected")
    tool_specs = registry.discover()
    return {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "generated_at": time.time(),
        "repo_root": str(repo),
        "workspace": str(ws),
        "git_revision": _git_revision(repo),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "provider_neutral_contracts": {
            "schemas": [
                PROFILE_SCHEMA,
                CAPABILITY_SCHEMA,
                GENERATION_REQUEST_SCHEMA,
                GENERATION_RESPONSE_SCHEMA,
                HEALTH_SCHEMA,
                FAILURE_SCHEMA,
                RECEIPT_SCHEMA,
                ROLE_SCHEMA,
            ],
            "role_policy_source": "hcli.providers.RoleRouter",
            "model_name_is_not_a_control_plane_type": True,
        },
        "tools": {
            "count": len(tool_specs),
            "names": [str(item.get("name")) for item in tool_specs],
            "specs": tool_specs,
        },
        "connectivity": connectivity,
        "gates": gates,
        "receipts": _receipt_inventory(
            (repo / "receipts", ws / ".hcli" / "receipts"),
        ),
        "maturity": {
            "control_plane": "FOUNDATION",
            "provider_generalization": "IMPLEMENTED_CONTRACTS",
            "durable_mission": "IMPLEMENTED",
            "fixture_recovery": gates["recovery_gate"]["status"],
            "research": gates["research_gate"]["qualification"],
            "vmcp": gates["vmcp_gate"]["qualification"],
            "native_hcli": gates["native_gate"]["qualification"],
            "native_resident": gates["resident_gate"]["qualification"],
            "native_mission": gates["native_mission_gate"]["qualification"],
            "autonomy": gates["autonomy_gate"]["qualification"],
            "accelerator_smoke": gates["accelerator_smoke"]["qualification"],
            "qwen38_fusion_audit": gates["qwen38_fusion_audit"]["qualification"],
            "unattended_sovereignty": (
                gates["unattended"].get("qualification")
                if isinstance(gates.get("unattended"), dict)
                else "NOT_CLAIMED"
            ),
            "flash_pre_runtime": gates["flash_science"]["qualification"],
            "negative_science_preboard": gates["preboard"]["qualification"],
            "initial_charge": gates["initial_charge"]["status"],
            "qwen38_transfer_map": gates["transfer_map"]["status"],
            "flash_precedent_map": gates["precedent_map"]["status"],
            "dense_vs_nf_ab": gates["ab_scaffold"]["status"],
            "fpga_preboard": gates["fpga_preboard"]["status"],
            "modellake_supervision": gates["modellake_supervision"]["status"],
            "flash_promotion": flash_promotion.get("status") if isinstance(flash_promotion, dict) else "NOT_PROVEN",
            "qwen27_runtime_identity": gates["qwen27_runtime_identity"]["status"],
            "qwen27_runtime_diff": gates["qwen27_runtime_diff"]["status"],
            "qwen27_mlp_diagnostic": gates["qwen27_mlp_diagnostic"]["status"],
            "protected_benchmark_watcher": gates["protected_benchmark_watcher"]["status"],
            "flash_executable": gates["flash_executable"]["status"],
            "flash_ebpw_budget": gates["flash_ebpw_budget"]["status"],
            "flash_token_ns_budget": gates["flash_token_ns_budget"]["status"],
            "flash_tensor_probe": gates["flash_tensor_probe"]["status"],
            "flash_representation_experiment": gates["flash_representation_experiment"]["status"],
            "flash_transform_parity": gates["flash_transform_parity"]["status"],
            "flash_loader_roundtrip": gates["flash_loader_roundtrip"]["status"],
            "flash_kernel_parity": gates["flash_kernel_parity"]["status"],
            "flash_component_body": gates["flash_component_body"]["status"],
            "flash_component_campaign": gates["flash_component_campaign"]["status"],
            "flash_graph_component": gates["flash_graph_component"]["status"],
            "flash_router_component_body": gates["flash_router_component_body"]["status"],
            "flash_router_graph": gates["flash_router_graph"]["status"],
            "flash_router_selection": gates["flash_router_selection"]["status"],
            "flash_router_selection_native": gates["flash_router_selection_native"]["status"],
            "flash_routed_expert_dispatch_native": gates["flash_routed_expert_dispatch_native"]["status"],
            "flash_gate_up_swiglu_native": gates["flash_gate_up_swiglu_native"]["status"],
            "flash_expert_composition_native": gates["flash_expert_composition_native"]["status"],
            "flash_shared_expert_composition_native": gates["flash_shared_expert_composition_native"]["status"],
            "flash_shared_residual_hyperconnection_native": gates["flash_shared_residual_hyperconnection_native"]["status"],
            "flash_exact_hyperconnection": gates["flash_exact_hyperconnection"]["status"],
            "flash_gpu_work_ledger": gates["flash_gpu_work_ledger"]["status"],
            "flash_token_critical_path": gates["flash_token_critical_path"]["status"],
            "cuda_capability_graph": gates["cuda_capability_graph"]["status"],
            "flash_router_representation_ab": gates["flash_router_representation_ab"]["status"],
        },
        "blockers": blockers,
        "next_actions": [
            "run recovery-gate against every configured production provider",
            "complete the one-hour unattended production-provider observation before making any sovereignty claim",
            "persist research provenance and protected benchmark receipts",
            "qualify additional providers only after their own deterministic verification closes",
            "continue the bounded protected Qwen watcher; do not treat contaminated A/B telemetry as promotion evidence",
            "use the bounded native routed-expert, layer-0 shared-expert, and exact routed-plus-shared HyperConnection candidate graphs as Flash anchors; close source router/top-k and source BF16 activation parity next, and fill actual EBPW/token-ns fields only from native protected complete-token receipts",
        ],
        "claim_boundary": "This checkpoint is an evidence census; it does not certify a model, runtime, hardware accelerator, or unattended sovereignty.",
    }


def write_program_checkpoint(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    network: bool = False,
) -> Dict[str, Any]:
    report = build_program_checkpoint(repo_root, workspace=workspace, network=network)
    repo = Path(repo_root or report["repo_root"]).expanduser().resolve()
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_NAME
    atomic_write_json(destination, report)
    report["checkpoint_path"] = str(destination)
    return report


__all__ = ["DEFAULT_NAME", "SCHEMA", "build_program_checkpoint", "write_program_checkpoint"]
