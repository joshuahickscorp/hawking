"""Build the non-fabricated Flash-Next executable work contract.

This module is intentionally a scaffold.  It turns the pinned header science
into the next executable interfaces—representation, loader, native kernels,
graph, capability, and complete-token timing—without pretending that a
weight body, a native Flash runtime, or a performance result exists.  It is
safe to run while ModelLake is acquiring: local inspection is bounded and
never reads or mutates the acquisition tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.flash_next import (
    ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
    COMPLETE_SYSTEM_BYTE_FIELDS,
    COMPLETE_SYSTEM_EBPW_MAX,
    EXPECTED_BYTES,
    PINNED_REVISION,
    REPO_ID,
    evaluate_flash_promotion,
)
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_next_noetic_executable.v1"
EBPW_SCHEMA = "hcli.agentos.flash_ebpw_budget.v1"
TOKEN_NS_SCHEMA = "hcli.agentos.flash_token_ns_budget.v1"
DERIVED = "[D]"
NOT_MEASURED = "NOT_MEASURED"
DEFAULT_SCIENCE = "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
DEFAULT_TENSOR_PROBE = "FLASH_FIRST_TENSOR_PROBE.json"
DEFAULT_REPRESENTATION_EXPERIMENT = "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json"
DEFAULT_REPRESENTATION_REPLICATION = "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT_DISJOINT.json"
DEFAULT_TRANSFORM_PARITY = "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"
DEFAULT_LOADER_ROUNDTRIP = "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"
DEFAULT_KERNEL_PARITY = "FLASH_NOETIC_Q4_KERNEL_PARITY.json"
DEFAULT_BODY_KERNEL_PARITY = "FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json"
DEFAULT_GRAPH_COMPONENT = "FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json"
DEFAULT_COMPONENT_CAMPAIGN = "FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json"
DEFAULT_ROUTER_GRAPH = "FLASH_NOETIC_ROUTER_GRAPH.json"
DEFAULT_ROUTER_SELECTION = "FLASH_NOETIC_ROUTER_SELECTION.json"
DEFAULT_ROUTER_REPRESENTATION_AB = "FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json"
DEFAULT_EXECUTABLE = "FLASH_NEXT_NOETIC_EXECUTABLE.json"
DEFAULT_EBPW = "FLASH_EBPW_BUDGET.json"
DEFAULT_TOKEN_NS = "FLASH_TOKEN_NS_BUDGET.json"
LAKE_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
LAKE_SLUG = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _value(row: Any, key: str = "value") -> Any:
    return row.get(key) if isinstance(row, Mapping) else None


def _direct_inventory(path: Path) -> Dict[str, Any]:
    """Inspect direct children only; never walk a potentially huge lake."""
    if not path.is_dir():
        return {"path": str(path), "present": False, "direct_files": 0, "direct_bytes": 0}
    files = 0
    total = 0
    names: list[str] = []
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= 512:
                    break
                try:
                    if entry.is_file(follow_symlinks=False):
                        files += 1
                        total += entry.stat(follow_symlinks=False).st_size
                        names.append(entry.name)
                except OSError:
                    continue
    except OSError as exc:
        return {"path": str(path), "present": True, "direct_files": files, "direct_bytes": total, "error": str(exc)[:400]}
    return {
        "path": str(path),
        "present": True,
        "direct_files": files,
        "direct_bytes": total,
        "entries": sorted(names),
    }


def _modellake_identity(repo: Path) -> Dict[str, Any]:
    census_path = repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json"
    supervision_path = repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json"
    census = _read_json(census_path) or {}
    supervision = _read_json(supervision_path) or {}
    final = LAKE_ROOT / "specimens" / LAKE_SLUG
    partial = LAKE_ROOT / "partial" / LAKE_SLUG
    manifest_path = LAKE_ROOT / "manifests" / f"{LAKE_SLUG}.json"
    manifest = _read_json(manifest_path)
    final_verified = bool(
        final.is_dir()
        and isinstance(manifest, Mapping)
        and manifest.get("resolved_sha") == PINNED_REVISION
    )
    return {
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "expected_bytes": EXPECTED_BYTES,
        "census_receipt": {"path": str(census_path), "sha256": _sha256(census_path), "present": census_path.is_file()},
        "supervision_receipt": {"path": str(supervision_path), "sha256": _sha256(supervision_path), "present": supervision_path.is_file()},
        "final": {"path": str(final), "present": final.is_dir(), "verified_manifest": final_verified},
        "partial": _direct_inventory(partial),
        "manifest": {
            "path": str(manifest_path),
            "present": manifest is not None,
            "resolved_sha": manifest.get("resolved_sha") if isinstance(manifest, Mapping) else None,
        },
        "observed_job_status": supervision.get("status"),
        "census_final_present": (census.get("flash_target_manifest") or {}).get("final_present") if isinstance(census.get("flash_target_manifest"), Mapping) else None,
        "body_read_by_this_scaffold": False,
        "mutation_by_this_scaffold": False,
        "status": "VERIFIED_FINAL_IDENTITY" if final_verified else ("PARTIAL_ACQUISITION" if partial.is_dir() else "NOT_STAGED"),
    }


def _tensor_probe_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read bounded probe evidence without treating it as a full model build."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_TENSOR_PROBE
    probe = _read_json(path)
    if probe is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "source_tensor": None,
            "dense_vs_packed_low_bit": None,
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        }
    return {
        "status": probe.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "source_label": probe.get("source_label"),
        "candidate_label": probe.get("candidate_label"),
        "root": probe.get("root"),
        "tensor_name": probe.get("tensor_name"),
        "source_tensor": probe.get("source_tensor"),
        "organ": probe.get("organ"),
        "dense_vs_packed_low_bit": probe.get("dense_vs_packed_low_bit"),
        "next_experiment": probe.get("next_experiment"),
        "body_mutated": probe.get("body_mutated"),
        "model_loaded": probe.get("model_loaded"),
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
    }


def _representation_experiment_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read bounded layout experiment evidence without promoting it."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_REPRESENTATION_EXPERIMENT
    experiment = _read_json(path)
    replication_path = repo / "receipts" / "headless" / DEFAULT_REPRESENTATION_REPLICATION
    replication = _read_json(replication_path)
    replications = []
    if replication is not None:
        replications.append({
            "status": replication.get("status"),
            "receipt_path": str(replication_path),
            "receipt_sha256": _sha256(replication_path),
            "source_tensor": replication.get("source_tensor"),
            "candidates": replication.get("candidates"),
            "comparison": replication.get("comparison"),
            "body_mutated": replication.get("body_mutated"),
            "model_loaded": replication.get("model_loaded"),
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        })
    if experiment is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "source_tensor": None,
            "candidates": None,
            "comparison": None,
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
            "replications": replications,
        }
    return {
        "status": experiment.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "source_label": experiment.get("source_label"),
        "candidate_label": experiment.get("candidate_label"),
        "root": experiment.get("root"),
        "tensor_name": experiment.get("tensor_name"),
        "source_tensor": experiment.get("source_tensor"),
        "candidates": experiment.get("candidates"),
        "comparison": experiment.get("comparison"),
        "next_experiment": experiment.get("next_experiment"),
        "body_mutated": experiment.get("body_mutated"),
        "model_loaded": experiment.get("model_loaded"),
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
        "replications": replications,
    }


def _transform_parity_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read full-tensor transform evidence without promoting it to runtime."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_TRANSFORM_PARITY
    transform = _read_json(path)
    if transform is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "source_tensor": None,
            "candidates": None,
            "comparison": None,
            "transform_parity": None,
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        }
    return {
        "status": transform.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "root": transform.get("root"),
        "tensor_name": transform.get("tensor_name"),
        "source_tensor": transform.get("source_tensor"),
        "candidates": transform.get("candidates"),
        "comparison": transform.get("comparison"),
        "transform_parity": transform.get("transform_parity"),
        "next_experiment": transform.get("next_experiment"),
        "body_mutated": transform.get("body_mutated"),
        "model_loaded": transform.get("model_loaded"),
        "whole_model_capability": transform.get("whole_model_capability"),
        "whole_model_runtime": transform.get("whole_model_runtime"),
        "native_loader": transform.get("native_loader"),
        "native_kernel": transform.get("native_kernel"),
        "runtime_performance": transform.get("runtime_performance"),
        "claim_boundary": transform.get("claim_boundary"),
    }


def _loader_roundtrip_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read bounded noetic loader evidence without calling it native runtime."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_LOADER_ROUNDTRIP
    loader = _read_json(path)
    if loader is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "candidate_id": None,
            "representation_descriptor": None,
            "encoded_sample": None,
            "loader_roundtrip": None,
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        }
    return {
        "status": loader.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "candidate_id": loader.get("candidate_id"),
        "tensor_name": loader.get("tensor_name"),
        "transform_reference": loader.get("transform_reference"),
        "representation_descriptor": loader.get("representation_descriptor"),
        "serialized_descriptor_sha256": loader.get("serialized_descriptor_sha256"),
        "encoded_sample": loader.get("encoded_sample"),
        "loader_roundtrip": loader.get("loader_roundtrip"),
        "source_sample": loader.get("source_sample"),
        "body_mutated": loader.get("body_mutated"),
        "model_loaded": loader.get("model_loaded"),
        "whole_model_capability": loader.get("whole_model_capability"),
        "whole_model_runtime": loader.get("whole_model_runtime"),
        "native_loader": loader.get("native_loader"),
        "native_kernel": loader.get("native_kernel"),
        "runtime_performance": loader.get("runtime_performance"),
        "claim_boundary": loader.get("claim_boundary"),
    }


def _kernel_parity_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read bounded native Metal evidence without treating it as Flash runtime."""
    if receipt:
        path = Path(receipt).expanduser().resolve()
    else:
        body_path = repo / "receipts" / "headless" / DEFAULT_BODY_KERNEL_PARITY
        path = body_path if body_path.is_file() else repo / "receipts" / "headless" / DEFAULT_KERNEL_PARITY
    kernel = _read_json(path)
    if kernel is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "source_tensor": None,
            "noetic_descriptor": None,
            "noetic_representation": None,
            "native_loader": None,
            "native_kernel": None,
            "gpu_timing": None,
            "parity": None,
            "candidate_body": None,
            "source_independent_execution": None,
            "candidate_body_persisted": None,
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        }
    native_kernel = kernel.get("native_kernel") if isinstance(kernel.get("native_kernel"), Mapping) else {}
    return {
        "status": kernel.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "repo": kernel.get("repo"),
        "pinned_revision": kernel.get("pinned_revision"),
        "root": kernel.get("root"),
        "model_lake_manifest": kernel.get("model_lake_manifest"),
        "source_tensor": kernel.get("source_tensor"),
        "noetic_descriptor": kernel.get("noetic_descriptor"),
        "noetic_representation": kernel.get("noetic_representation"),
        "native_loader": kernel.get("native_loader"),
        "native_kernel": kernel.get("native_kernel"),
        "gpu_timing": kernel.get("gpu_timing"),
        "parity": kernel.get("parity"),
        "input": kernel.get("input"),
        "candidate_body": kernel.get("candidate_body"),
        "body_mutated": kernel.get("body_mutated"),
        "model_loaded": kernel.get("model_loaded"),
        "source_independent_execution": (kernel.get("native_loader") or {}).get("source_independent_execution"),
        "candidate_body_persisted": (kernel.get("native_loader") or {}).get("candidate_body_persisted"),
        "whole_model_capability": native_kernel.get("whole_model_capability", "NOT_TESTED"),
        "whole_model_runtime": native_kernel.get("whole_model_runtime", "NOT_TESTED"),
        "complete_system_ebpw": kernel.get("complete_system_ebpw"),
        "flash_tps": kernel.get("flash_tps"),
        "promotion_allowed": kernel.get("promotion_allowed"),
        "claim_boundary": kernel.get("claim_boundary"),
        "next_action": kernel.get("next_action"),
    }


def _graph_component_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read the bounded compiled graph component without widening its scope."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_GRAPH_COMPONENT
    graph = _read_json(path)
    if graph is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "component_status": "NOT_COMPILED",
            "graph_fingerprint": None,
            "whole_model_capability": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "candidate_body_persisted": None,
            "promotion_allowed": False,
        }
    return {
        "status": graph.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "schema": graph.get("schema"),
        "nomenclature_version": graph.get("nomenclature_version"),
        "semantic_type": graph.get("semantic_type"),
        "compiler_stage": graph.get("compiler_stage"),
        "component_status": graph.get("component_status"),
        "graph_fingerprint": graph.get("graph_fingerprint"),
        "candidate_id": graph.get("candidate_id"),
        "source_identity": graph.get("source_identity"),
        "source_backed": graph.get("source_backed"),
        "source_independent_execution": graph.get("source_independent_execution"),
        "candidate_body_persisted": graph.get("candidate_body_persisted"),
        "physical_graph": graph.get("physical_graph"),
        "noetic_ir": graph.get("noetic_ir"),
        "validation": graph.get("validation"),
        "whole_model_capability": graph.get("whole_model_capability"),
        "complete_token_runtime": graph.get("complete_token_runtime"),
        "promotion_allowed": graph.get("promotion_allowed"),
        "claim_boundary": graph.get("claim_boundary"),
        "next_action": graph.get("next_action"),
    }


def _component_campaign_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read a bounded multi-component Noetic campaign without promoting it."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_COMPONENT_CAMPAIGN
    campaign = _read_json(path)
    if campaign is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "component_status": "NOT_COMPILED",
            "component_count": 0,
            "component_windows": [],
            "source_independent_execution": None,
            "candidate_body_persisted": None,
            "whole_model_capability": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "promotion_allowed": False,
        }
    return {
        "status": campaign.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "schema": campaign.get("schema"),
        "nomenclature_version": campaign.get("nomenclature_version"),
        "semantic_type": campaign.get("semantic_type"),
        "compiler_stage": campaign.get("compiler_stage"),
        "component_status": campaign.get("component_status"),
        "component_count": campaign.get("component_count"),
        "component_windows": campaign.get("component_windows"),
        "components": campaign.get("components"),
        "source_identity": campaign.get("source_identity"),
        "source_independent_execution": campaign.get("source_independent_execution"),
        "candidate_body_persisted": campaign.get("candidate_body_persisted"),
        "physical_graph": campaign.get("physical_graph"),
        "noetic_ir": campaign.get("noetic_ir"),
        "whole_model_capability": campaign.get("whole_model_capability"),
        "complete_token_runtime": campaign.get("complete_token_runtime"),
        "promotion_allowed": campaign.get("promotion_allowed"),
        "claim_boundary": campaign.get("claim_boundary"),
        "next_action": campaign.get("next_action"),
    }


def _router_graph_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read the bounded rank-2 router graph without promoting its semantics."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_ROUTER_GRAPH
    graph = _read_json(path)
    if graph is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "component_status": "NOT_COMPILED",
            "source_independent_execution": None,
            "candidate_body_persisted": None,
            "whole_model_capability": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "promotion_allowed": False,
        }
    return {
        "status": graph.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "schema": graph.get("schema"),
        "nomenclature_version": graph.get("nomenclature_version"),
        "semantic_type": graph.get("semantic_type"),
        "compiler_stage": graph.get("compiler_stage"),
        "component_status": graph.get("component_status"),
        "candidate_id": graph.get("candidate_id"),
        "component_window": graph.get("component_window"),
        "source_identity": graph.get("source_identity"),
        "source_independent_execution": graph.get("source_independent_execution"),
        "candidate_body_persisted": graph.get("candidate_body_persisted"),
        "physical_graph": graph.get("physical_graph"),
        "noetic_ir": graph.get("noetic_ir"),
        "validation": graph.get("validation"),
        "whole_model_capability": graph.get("whole_model_capability"),
        "complete_token_runtime": graph.get("complete_token_runtime"),
        "promotion_allowed": graph.get("promotion_allowed"),
        "claim_boundary": graph.get("claim_boundary"),
        "next_action": graph.get("next_action"),
    }


def _router_selection_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read the derived router selector without promoting it to native runtime."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_ROUTER_SELECTION
    selection = _read_json(path)
    if selection is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "selection_status": "NOT_EXECUTED",
            "source_independent_execution": None,
            "candidate_body_persisted": None,
            "native_selection_execution_observed": None,
            "whole_model_capability": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "promotion_allowed": False,
        }
    return {
        "status": selection.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "schema": selection.get("schema"),
        "nomenclature_version": selection.get("nomenclature_version"),
        "semantic_type": selection.get("semantic_type"),
        "compiler_stage": selection.get("compiler_stage"),
        "selection_status": "EXECUTED" if selection.get("selection") else "NOT_EXECUTED",
        "source_identity": selection.get("source_identity"),
        "config": selection.get("config"),
        "selection": selection.get("selection"),
        "source_selection": selection.get("source_selection"),
        "source_selection_parity": selection.get("source_selection_parity"),
        "execution": selection.get("execution"),
        "physical_graph": selection.get("physical_graph"),
        "noetic_ir": selection.get("noetic_ir"),
        "validation": selection.get("validation"),
        "source_independent_execution": selection.get("execution", {}).get("source_independent") if isinstance(selection.get("execution"), Mapping) else None,
        "candidate_body_persisted": selection.get("execution", {}).get("candidate_body_persisted") if isinstance(selection.get("execution"), Mapping) else None,
        "native_selection_execution_observed": selection.get("native_selection_execution_observed"),
        "whole_model_capability": selection.get("whole_model_capability"),
        "complete_token_runtime": selection.get("complete_token_runtime"),
        "promotion_allowed": selection.get("promotion_allowed"),
        "claim_boundary": selection.get("claim_boundary"),
        "next_action": selection.get("next_action"),
    }


def _router_representation_ab_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read the bounded router representation study without treating it as a body."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_ROUTER_REPRESENTATION_AB
    study = _read_json(path)
    if study is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "candidate_bodies_persisted": None,
            "whole_model_capability": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "promotion_allowed": False,
        }
    return {
        "status": study.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "schema": study.get("schema"),
        "nomenclature_version": study.get("nomenclature_version"),
        "semantic_type": study.get("semantic_type"),
        "compiler_stage": study.get("compiler_stage"),
        "source_identity": study.get("source_identity"),
        "config": study.get("config"),
        "source_selection": study.get("source_selection"),
        "candidates": study.get("candidates"),
        "recommendation": study.get("recommendation"),
        "physical_graph": study.get("physical_graph"),
        "noetic_ir": study.get("noetic_ir"),
        "validation": study.get("validation"),
        "candidate_bodies_persisted": study.get("candidate_bodies_persisted"),
        "whole_model_capability": study.get("whole_model_capability"),
        "complete_token_runtime": study.get("complete_token_runtime"),
        "promotion_allowed": study.get("promotion_allowed"),
        "claim_boundary": study.get("claim_boundary"),
        "next_action": study.get("next_action"),
    }


def _primary_organs(science: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = science.get("organ_graph")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping) and row.get("accounting_role") == "PRIMARY"]


def _source_summary(science: Mapping[str, Any]) -> Dict[str, Any]:
    audit = science.get("safetensors_header_audit") if isinstance(science.get("safetensors_header_audit"), Mapping) else {}
    architecture = science.get("architecture") if isinstance(science.get("architecture"), Mapping) else {}
    payload = _int(audit.get("payload_bytes"))
    dtype_bytes = 2
    # Header science currently reports BF16. Keep this derived from the
    # observed tensor layouts when possible, but never infer missing bodies.
    for organ in science.get("organ_graph") or []:
        if not isinstance(organ, Mapping):
            continue
        for layout in organ.get("tensor_layout") or []:
            if isinstance(layout, Mapping) and str(layout.get("dtype") or "").upper() in {"F16", "BF16"}:
                dtype_bytes = 2
                break
    parameters = payload // dtype_bytes if payload is not None else None
    return {
        "architecture_fingerprint": _safe(science.get("architecture_fingerprint")),
        "source_identity": _safe(science.get("source_identity")),
        "header_audit": {
            "complete": audit.get("complete"),
            "payload_bytes": payload,
            "header_tensor_count": audit.get("header_tensor_count"),
            "body_bytes_requested": audit.get("body_bytes_requested", 0),
            "body_bytes_loaded": audit.get("body_bytes_loaded", 0),
        },
        "declared_dtype_bytes": dtype_bytes,
        "source_parameter_count": parameters,
        "source_payload_bytes": payload,
        "index_total_size": _int(architecture.get("index_total_size")),
    }


def _organ_records(science: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    primary = _primary_organs(science)
    total_source = sum(
        max(0, _int(_value(row.get("stored_bytes"))) or 0)
        for row in primary
    )
    for row in primary:
        organ_id = str(row.get("id") or "unknown")
        source_bytes = _int(_value(row.get("stored_bytes")))
        source_active = _int(_value(row.get("active_bytes_per_token")))
        source_flops = _int(_value(row.get("flops_per_token")))
        fraction = (source_bytes / total_source) if source_bytes is not None and total_source else None
        rows.append({
            "organ": organ_id,
            "label": DERIVED,
            "source_bytes": source_bytes,
            "source_bytes_label": "[V] pinned safetensors header payload" if source_bytes is not None else NOT_MEASURED,
            "source_active_bytes_per_token": source_active,
            "source_flops_per_token": source_flops,
            "source_tensor_count": row.get("tensors"),
            "source_allocation_fraction": fraction,
            "representation_status": "CANDIDATE_NOT_BUILT",
            "chosen_representation": "shared-basis-plus-NF-residual for expert/routed paths; organ-native representation for state, sparse, n-gram, MTP, and vision paths",
            "actual_representation_bytes": None,
            "actual_bytes_label": NOT_MEASURED,
            "native_loader_status": "NOT_IMPLEMENTED",
            "native_kernel_status": "PLAN_ONLY",
            "capability_status": "NOT_RUN",
        })
    # These are explicit views/state requirements and must not disappear just
    # because they are not additive source tensors.
    rows.extend([
        {
            "organ": "recurrent_state",
            "label": DERIVED,
            "source_bytes": 0,
            "source_bytes_label": "[D] virtual runtime state; no source tensor payload",
            "source_active_bytes_per_token": _int(_value(next((r for r in science.get("organ_graph") or [] if isinstance(r, Mapping) and r.get("id") == "recurrent_state"), {}).get("active_bytes_per_token"))),
            "source_flops_per_token": _int(_value(next((r for r in science.get("organ_graph") or [] if isinstance(r, Mapping) and r.get("id") == "recurrent_state"), {}).get("flops_per_token"))),
            "source_allocation_fraction": 0.0,
            "representation_status": "REQUIRED_RESIDENT_STATE_NOT_BUILT",
            "chosen_representation": "resident sequence-isolated state with explicit read/modify/write accounting",
            "actual_representation_bytes": None,
            "actual_bytes_label": NOT_MEASURED,
            "native_loader_status": "NOT_IMPLEMENTED",
            "native_kernel_status": "PLAN_ONLY",
            "capability_status": "NOT_RUN",
        },
    ])
    return rows


def _ebpw_budget(
    science: Mapping[str, Any],
    source: Mapping[str, Any],
    tensor_probe: Mapping[str, Any],
    representation_experiment: Mapping[str, Any],
    transform_parity: Mapping[str, Any],
    loader_roundtrip: Mapping[str, Any],
    kernel_parity: Mapping[str, Any],
) -> Dict[str, Any]:
    parameters = source.get("source_parameter_count")
    system_ceiling = int(parameters * COMPLETE_SYSTEM_EBPW_MAX) if isinstance(parameters, int) else None
    organs = _organ_records(science)
    accounting = {
        field: {
            "budget_bytes": None,
            "actual_bytes": None,
            "label": DERIVED,
            "status": "WAITING_FOR_REPRESENTATION_AND_LOADER",
        }
        for field in COMPLETE_SYSTEM_BYTE_FIELDS
    }
    chosen_representation = {
        "id": "flash-routed-expert-dual-candidate-v1",
        "status": "HYPOTHESIS_NOT_BUILT",
        "label": DERIVED,
        "selection_policy": "minimize observed tensor-level effective bits per value while retaining a lower-error quality alternate; this does not select a whole-model runtime representation",
        "expert_bank": "shared basis plus per-expert NF residual",
        "router": "resident route metadata and selected-expert gather",
        "deltanet": "resident state-native representation",
        "ngram": "lookup/compositional representation, not generic dense quantization",
        "sparse_attention": "budget/index-native sparse representation",
        "mtp": "explicit draft/verify/rollback representation",
        "vision": "conditional multimodal path; text-resident omission requires an explicit separate contract",
    }
    if transform_parity.get("status") == "PASSED":
        comparison = transform_parity.get("comparison") or {}
        candidates = transform_parity.get("candidates") or {}
        source_tensor = transform_parity.get("source_tensor") or {}
        independent = candidates.get("independent_q4_g64") or {}
        shared = candidates.get("shared_bf16_basis_nf4_residual") or {}
        chosen_representation.update({
            "status": "FULL_TENSOR_TRANSFORM_OBSERVED_NOT_WHOLE_MODEL",
            "selected_candidate": "independent_q4_g64",
            "selected_candidate_reason": "lower observed tensor-level EBPW (4.25 versus 4.28125 bits/value); shared-basis/NF4 remains the lower-error quality alternate",
            "quality_alternate": "shared_bf16_basis_nf4_residual",
            "bounded_source_transform": {
                "receipt_path": transform_parity.get("receipt_path"),
                "tensor_name": transform_parity.get("tensor_name"),
                "source_layout": source_tensor.get("layout"),
                "shape": source_tensor.get("shape"),
                "source_payload_bytes": source_tensor.get("payload_bytes"),
                "full_payload_read": comparison.get("full_payload_read"),
                "independent_q4_candidate_bytes": independent.get("candidate_bytes"),
                "independent_q4_effective_bits_per_value": independent.get("effective_bits_per_value"),
                "shared_basis_nf4_residual_candidate_bytes": shared.get("candidate_bytes"),
                "shared_basis_nf4_residual_effective_bits_per_value": shared.get("effective_bits_per_value"),
                "independent_weight_reconstruction": independent.get("weight_reconstruction"),
                "shared_weight_reconstruction": shared.get("weight_reconstruction"),
                "independent_reference_vector": independent.get("reference_vector"),
                "shared_reference_vector": shared.get("reference_vector"),
                "pack_unpack_parity": (transform_parity.get("transform_parity") or {}).get("pack_unpack_parity"),
                "capability_parity": comparison.get("capability_parity", "NOT_TESTED"),
                "whole_model_runtime": comparison.get("whole_model_runtime", "NOT_TESTED"),
                "next_experiment": transform_parity.get("next_experiment"),
                "loader_roundtrip": {
                    "status": loader_roundtrip.get("status"),
                    "candidate_id": loader_roundtrip.get("candidate_id"),
                    "receipt_path": loader_roundtrip.get("receipt_path"),
                    "descriptor_sha256": loader_roundtrip.get("serialized_descriptor_sha256"),
                    "roundtrip": loader_roundtrip.get("loader_roundtrip"),
                    "label": DERIVED,
                },
                "native_kernel_parity": {
                    "status": kernel_parity.get("status"),
                    "receipt_path": kernel_parity.get("receipt_path"),
                    "descriptor": kernel_parity.get("noetic_descriptor"),
                    "native_loader": kernel_parity.get("native_loader"),
                    "kernel": (kernel_parity.get("native_kernel") or {}).get("kernel"),
                    "gpu_timing": kernel_parity.get("gpu_timing"),
                    "parity": kernel_parity.get("parity"),
                    "whole_model_capability": kernel_parity.get("whole_model_capability", "NOT_TESTED"),
                    "whole_model_runtime": kernel_parity.get("whole_model_runtime", "NOT_TESTED"),
                    "label": DERIVED,
                },
                "label": DERIVED,
            },
        })
    elif representation_experiment.get("status") == "PASSED":
        comparison = representation_experiment.get("comparison") or {}
        candidates = representation_experiment.get("candidates") or {}
        chosen_representation.update({
            "status": "BOUNDED_LAYOUT_EXPERIMENT_OBSERVED_NOT_WHOLE_MODEL",
            "bounded_source_experiment": {
                "receipt_path": representation_experiment.get("receipt_path"),
                "tensor_name": representation_experiment.get("tensor_name"),
                "source_layout": (representation_experiment.get("source_tensor") or {}).get("layout"),
                "experts": (representation_experiment.get("source_tensor") or {}).get("selected_experts"),
                "row_range": (representation_experiment.get("source_tensor") or {}).get("row_range"),
                "independent_q4_effective_bits_per_value": (candidates.get("independent_q4_g64") or {}).get("effective_bits_per_value"),
                "shared_basis_nf4_residual_effective_bits_per_value": (candidates.get("shared_bf16_basis_nf4_residual") or {}).get("effective_bits_per_value"),
                "shared_basis_nf4_residual_cosine": ((candidates.get("shared_bf16_basis_nf4_residual") or {}).get("reference_vector") or {}).get("cosine"),
                "same_source_rows": comparison.get("same_source_rows"),
                "same_reference_vector": comparison.get("same_reference_vector"),
                "capability_parity": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "disjoint_replications": [
                    {
                        "status": item.get("status"),
                        "receipt_path": item.get("receipt_path"),
                        "source_experts": (item.get("source_tensor") or {}).get("selected_experts"),
                        "source_row_range": (item.get("source_tensor") or {}).get("row_range"),
                        "label": DERIVED,
                    }
                    for item in representation_experiment.get("replications") or []
                ],
                "label": DERIVED,
            },
        })
    elif tensor_probe.get("status") == "PASSED":
        chosen_representation.update({
            "status": "BOUNDED_SLICE_OBSERVED_NOT_WHOLE_MODEL",
            "bounded_source_probe": {
                "receipt_path": tensor_probe.get("receipt_path"),
                "tensor_name": tensor_probe.get("tensor_name"),
                "organ": tensor_probe.get("organ"),
                "candidate_scheme": ((tensor_probe.get("dense_vs_packed_low_bit") or {}).get("candidate") or {}).get("scheme"),
                "candidate_effective_bits_per_value": ((tensor_probe.get("dense_vs_packed_low_bit") or {}).get("candidate") or {}).get("effective_bits_per_value"),
                "candidate_is_smaller_on_slice": ((tensor_probe.get("dense_vs_packed_low_bit") or {}).get("comparison") or {}).get("candidate_is_smaller"),
                "capability_parity": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": DERIVED,
            },
        })
    return {
        "schema": EBPW_SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "PLANNED_UNTIL_VERIFIED_BODY",
        "label": DERIVED,
        "source_identity": source,
        "target_contract": {
            "complete_system_ebpw_max": COMPLETE_SYSTEM_EBPW_MAX,
            "denominator": "source_parameter_count derived from complete pinned header payload / declared dtype bytes",
            "complete_system_byte_fields": list(COMPLETE_SYSTEM_BYTE_FIELDS),
            "target_ceiling_bytes": system_ceiling,
            "target_ceiling_is_not_an_actual_measurement": True,
        },
        "chosen_representation": chosen_representation,
        "bounded_source_probe": tensor_probe,
        "bounded_representation_experiment": representation_experiment,
        "bounded_transform_parity": transform_parity,
        "bounded_loader_roundtrip": loader_roundtrip,
        "bounded_kernel_parity": kernel_parity,
        "bounded_tensor_observation": {
            "status": "FULL_TENSOR_TRANSFORM_ONLY" if transform_parity.get("status") == "PASSED" else "NOT_MEASURED",
            "is_complete_system": False,
            "source_tensor": (transform_parity.get("source_tensor") if transform_parity.get("status") == "PASSED" else None),
            "comparison": (transform_parity.get("comparison") if transform_parity.get("status") == "PASSED" else None),
            "candidates": {
                name: {
                    "candidate_bytes": value.get("candidate_bytes"),
                    "effective_bits_per_value": value.get("effective_bits_per_value"),
                    "weight_reconstruction": value.get("weight_reconstruction"),
                    "reference_vector": value.get("reference_vector"),
                }
                for name, value in (transform_parity.get("candidates") or {}).items()
                if isinstance(value, Mapping)
            } if transform_parity.get("status") == "PASSED" else None,
            "label": DERIVED,
        },
        "organs": organs,
        "complete_system_accounting": accounting,
        "measured": {
            "complete_system_bytes": None,
            "complete_system_ebpw": None,
            "all_required_bytes_included": False,
            "fallback_count": None,
            "dense_parent_execution_fallback": None,
            "hidden_dense_rematerialization": None,
        },
        "budget_policy": {
            "allocation": "No guessed per-organ actual is accepted. A provisional ceiling may be allocated only after a representation body and loader manifest exist.",
            "source_payload_is_not_candidate_storage": True,
            "overhead_must_be_counted": True,
            "zero_bytes_are_not_inferred_from_missing_fields": True,
        },
        "promotion_allowed": False,
        "claim_boundary": "This is an EBPW budget and representation contract, not a compressed artifact measurement. Missing actual bytes remain missing.",
    }


def _token_ns_budget(science: Mapping[str, Any], source: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for organ in _organ_records(science):
        rows.append({
            "organ": organ["organ"],
            "label": DERIVED,
            "source_active_bytes_per_token": organ.get("source_active_bytes_per_token"),
            "source_flops_per_token": organ.get("source_flops_per_token"),
            "target_gpu_ns_per_token": None,
            "target_wall_ns_per_token": None,
            "target_dispatches_per_token": None,
            "target_state_read_write_bytes_per_token": None,
            "target_sync_ns": None,
            "target_copy_bytes": None,
            "actual_gpu_ns_per_token": None,
            "actual_complete_wall_ns_per_accepted_token": None,
            "actual_dispatches_per_token": None,
            "actual_state_read_write_bytes_per_token": None,
            "actual_sync_ns": None,
            "actual_copy_bytes": None,
            "status": "WAITING_FOR_NATIVE_EXECUTION",
        })
    return {
        "schema": TOKEN_NS_SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "PLANNED_UNTIL_NATIVE_EXECUTION",
        "label": DERIVED,
        "source_identity": source,
        "target_contract": {
            "accepted_capability_preserving_tps_min": ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
            "complete_wall_ns_per_accepted_token_max": int(1_000_000_000 / ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN),
            "timing_unit": "complete accepted generated token wall time, including all required graph/runtime/host ceremony",
            "kernel_only_or_raw_draft_timing_is_not_acceptable": True,
        },
        "organs": rows,
        "system_ledger": {
            "complete_generation_wall_ns": None,
            "accepted_tokens": None,
            "rejected_draft_tokens": None,
            "prefill_wall_ns": None,
            "decode_wall_ns": None,
            "host_wait_ns": None,
            "gpu_ns": None,
            "sync_ns": None,
            "copy_bytes": None,
            "dispatches": None,
            "fallback_count": None,
            "capability_parity": None,
            "protected_benchmark_class": None,
        },
        "measurement_protocol": {
            "same_source_input_and_output_contract": True,
            "dense_vs_nf_matched_controls": True,
            "complete_token_accounting": True,
            "protected_quiescent_before_and_after": True,
            "native_kernel_genome_and_dispatch_trace": True,
            "no_dense_parent_or_deep_rematerialization": True,
        },
        "promotion_allowed": False,
        "claim_boundary": "No Flash token rate, GPU time, dispatch count, or accepted-token result is claimed until a native executable produces a protected complete-token receipt.",
    }


def _executable_manifest(
    science: Mapping[str, Any],
    source: Mapping[str, Any],
    lake: Mapping[str, Any],
    ebpw: Mapping[str, Any],
    token_ns: Mapping[str, Any],
    tensor_probe: Mapping[str, Any],
    representation_experiment: Mapping[str, Any],
    transform_parity: Mapping[str, Any],
    loader_roundtrip: Mapping[str, Any],
    kernel_parity: Mapping[str, Any],
    graph_component: Mapping[str, Any],
    component_campaign: Mapping[str, Any],
    router_graph: Mapping[str, Any],
    router_selection: Mapping[str, Any],
    router_representation_ab: Mapping[str, Any],
) -> Dict[str, Any]:
    organs = [str(row.get("organ")) for row in ebpw.get("organs") or [] if isinstance(row, Mapping)]
    return {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "SCAFFOLD_ONLY",
        "qualification": False,
        "NOT_FOR_PROMOTION": True,
        "complete_system_ebpw": None,
        "accepted_capability_preserving_tps": None,
        "fallback_count": None,
        "dense_parent_execution_fallback": False,
        "hidden_dense_rematerialization": False,
        "declarations_are_requirements_not_runtime_evidence": True,
        "label": DERIVED,
        "source_identity": {
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "architecture_fingerprint": _safe(science.get("architecture_fingerprint")),
            "science_receipt": source.get("science_receipt"),
            "header_only_source": tensor_probe.get("status") != "PASSED",
            "bounded_source_slice_observed": tensor_probe.get("status") == "PASSED",
            "bounded_probe_receipt": tensor_probe.get("receipt_path"),
            "bounded_source_layout_experiment_observed": representation_experiment.get("status") == "PASSED",
            "bounded_representation_experiment_receipt": representation_experiment.get("receipt_path"),
            "bounded_full_tensor_transform_observed": transform_parity.get("status") == "PASSED",
            "bounded_transform_parity_receipt": transform_parity.get("receipt_path"),
            "bounded_loader_roundtrip_observed": loader_roundtrip.get("status") == "PASSED",
            "bounded_loader_roundtrip_receipt": loader_roundtrip.get("receipt_path"),
            "bounded_native_kernel_parity_observed": kernel_parity.get("status") == "PASSED",
            "bounded_native_kernel_parity_receipt": kernel_parity.get("receipt_path"),
            "bounded_noetic_graph_component_observed": graph_component.get("status") == "PASSED",
            "bounded_noetic_graph_component_receipt": graph_component.get("receipt_path"),
            "bounded_component_campaign_observed": component_campaign.get("status") == "PASSED",
            "bounded_component_campaign_receipt": component_campaign.get("receipt_path"),
            "bounded_router_graph_observed": router_graph.get("status") == "PASSED",
            "bounded_router_graph_receipt": router_graph.get("receipt_path"),
            "bounded_router_selection_observed": router_selection.get("status") == "PASSED",
            "bounded_router_selection_receipt": router_selection.get("receipt_path"),
            "bounded_router_representation_ab_observed": router_representation_ab.get("status") == "PASSED",
            "bounded_router_representation_ab_receipt": router_representation_ab.get("receipt_path"),
            "bounded_component_body_loaded": kernel_parity.get("source_independent_execution") is True,
            "bounded_component_body_receipt": (kernel_parity.get("candidate_body") or {}).get("receipt_path"),
            "weight_body_loaded": False,
        },
        "model_lake": lake,
        "source_tensor_probe": tensor_probe,
        "source_representation_experiment": representation_experiment,
        "source_transform_parity": transform_parity,
        "source_loader_roundtrip": loader_roundtrip,
        "source_kernel_parity": kernel_parity,
        "source_graph_component": graph_component,
        "source_component_campaign": component_campaign,
        "source_router_graph": router_graph,
        "source_router_selection": router_selection,
        "source_router_representation_ab": router_representation_ab,
        "chosen_representation": ebpw.get("chosen_representation"),
        "native_loader": {
            "status": "NOT_IMPLEMENTED",
            "bounded_descriptor_roundtrip_status": loader_roundtrip.get("status"),
            "bounded_descriptor_roundtrip_receipt": loader_roundtrip.get("receipt_path"),
            "bounded_native_descriptor_load_status": (kernel_parity.get("native_loader") or {}).get("status"),
            "bounded_native_descriptor_load_receipt": kernel_parity.get("receipt_path") if (kernel_parity.get("native_loader") or {}).get("status") else None,
            "bounded_source_independent_body_load_status": (kernel_parity.get("native_loader") or {}).get("source_independent_execution"),
            "bounded_source_independent_body_load_receipt": kernel_parity.get("receipt_path") if (kernel_parity.get("native_loader") or {}).get("source_independent_execution") else None,
            "bounded_component_body_persisted": kernel_parity.get("candidate_body_persisted"),
            "bounded_component_body": kernel_parity.get("candidate_body"),
            "bounded_native_kernel_parity_status": kernel_parity.get("status"),
            "bounded_native_kernel_parity_receipt": kernel_parity.get("receipt_path"),
            "required": ["verified body manifest", "zero-copy/streaming policy", "per-organ ownership", "resident lifetime", "loader hash"],
            "body_read_by_scaffold": False,
        },
        "native_kernels": {
            "status": "PLAN_ONLY",
            "coverage": [
                {"organ": "embeddings", "kernel": "partitioned_embedding_lookup", "status": "NOT_IMPLEMENTED"},
                {"organ": "routed_experts", "kernel": "native_nf_expert_gemv", "status": "NOT_IMPLEMENTED"},
                {"organ": "shared_expert", "kernel": "shared_expert_fused_gemv", "status": "NOT_IMPLEMENTED"},
                {"organ": "router", "kernel": "router_topk_gather", "status": "NOT_IMPLEMENTED"},
                {"organ": "deltanet", "kernel": "persistent_state_update", "status": "NOT_IMPLEMENTED"},
                {"organ": "recurrent_state", "kernel": "resident_state_read_modify_write", "status": "NOT_IMPLEMENTED"},
                {"organ": "sparse_attention", "kernel": "budgeted_sparse_gather_reduce", "status": "NOT_IMPLEMENTED"},
                {"organ": "ngram_engine", "kernel": "lookup_or_compositional_generator", "status": "NOT_IMPLEMENTED"},
                {"organ": "mtp", "kernel": "draft_verify_rollback", "status": "NOT_IMPLEMENTED"},
                {"organ": "norms", "kernel": "fused_norm_epilogue", "status": "NOT_IMPLEMENTED"},
                {"organ": "lm_head", "kernel": "vocabulary_projection_and_reduce", "status": "NOT_IMPLEMENTED"},
                {"organ": "vision_backbone", "kernel": "conditional_multimodal_vision_path", "status": "NOT_IMPLEMENTED"},
                {"organ": "residual_hyperconnections", "kernel": "low_rank_residual_mix", "status": "NOT_IMPLEMENTED"},
                {"organ": "support_misc", "kernel": "ownership_audit_required", "status": "UNRESOLVED"},
            ],
            "bounded_component_evidence": {
                "status": kernel_parity.get("status"),
                "kernel": (kernel_parity.get("native_kernel") or {}).get("kernel"),
                "receipt": kernel_parity.get("receipt_path"),
                "scope": "one real routed-expert source block only; not complete Flash execution",
                "source_independent_execution": (kernel_parity.get("native_loader") or {}).get("source_independent_execution"),
                "candidate_body": kernel_parity.get("candidate_body"),
                "label": DERIVED,
            },
            "bounded_campaign_evidence": {
                "status": component_campaign.get("status"),
                "receipt": component_campaign.get("receipt_path"),
                "component_count": component_campaign.get("component_count"),
                "component_windows": component_campaign.get("component_windows"),
                "physical_graph_fingerprint": (component_campaign.get("physical_graph") or {}).get("fingerprint"),
                "source_independent_execution": component_campaign.get("source_independent_execution"),
                "candidate_body_persisted": component_campaign.get("candidate_body_persisted"),
                "scope": "bounded routed-expert body campaign only; not complete Flash execution",
                "label": DERIVED,
            },
            "bounded_router_matrix_evidence": {
                "status": router_graph.get("status"),
                "receipt": router_graph.get("receipt_path"),
                "physical_graph_fingerprint": (router_graph.get("physical_graph") or {}).get("fingerprint"),
                "component_status": router_graph.get("component_status"),
                "source_independent_execution": router_graph.get("source_independent_execution"),
                "candidate_body_persisted": router_graph.get("candidate_body_persisted"),
                "scope": "bounded router matrix matvec only; sigmoid/top-k and complete Flash execution remain unimplemented",
                "label": DERIVED,
            },
            "bounded_router_selection_evidence": {
                "status": router_selection.get("status"),
                "receipt": router_selection.get("receipt_path"),
                "selection_status": router_selection.get("selection_status"),
                "source_selection_parity_status": (router_selection.get("source_selection_parity") or {}).get("status"),
                "source_selection_parity_qualified": (router_selection.get("source_selection_parity") or {}).get("expert_ids_exact_match"),
                "physical_graph_fingerprint": (router_selection.get("physical_graph") or {}).get("fingerprint"),
                "source_independent_execution": router_selection.get("source_independent_execution"),
                "candidate_body_persisted": router_selection.get("candidate_body_persisted"),
                "native_selection_execution_observed": router_selection.get("native_selection_execution_observed"),
                "scope": "derived CPU router softmax/top-k over a persisted full body; no native selection kernel or complete Flash execution",
                "label": DERIVED,
            },
            "bounded_router_representation_ab_evidence": {
                "status": router_representation_ab.get("status"),
                "receipt": router_representation_ab.get("receipt_path"),
                "physical_graph_fingerprint": (router_representation_ab.get("physical_graph") or {}).get("fingerprint"),
                "candidate_bodies_persisted": router_representation_ab.get("candidate_bodies_persisted"),
                "recommendation": router_representation_ab.get("recommendation"),
                "scope": "in-memory router representation comparison only; no persisted body, native kernel, or complete Flash execution",
                "label": DERIVED,
            },
            "dense_rematerialization": "FORBIDDEN_BY_FINAL_RUNTIME_POLICY",
        },
        "graph_runtime": {
            "status": (
                "BOUNDED_MULTI_COMPONENT_COMPILED"
                if component_campaign.get("status") == "PASSED"
                else "BOUNDED_COMPONENT_COMPILED"
                if graph_component.get("status") == "PASSED"
                else "PLAN_ONLY"
            ),
            "organ_order": organs,
            "graph_source": "pinned header organ graph; bounded component body-backed when the native body-load receipt is present, while remaining runtime edges still require implementation",
            "bounded_component": {
                "status": graph_component.get("component_status"),
                "receipt_path": graph_component.get("receipt_path"),
                "fingerprint": graph_component.get("graph_fingerprint"),
                "candidate_id": graph_component.get("candidate_id"),
                "source_backed": graph_component.get("source_backed"),
                "whole_model_capability": graph_component.get("whole_model_capability"),
                "complete_token_runtime": graph_component.get("complete_token_runtime"),
                "candidate_body_persisted": graph_component.get("candidate_body_persisted"),
            },
            "component_campaign": {
                "status": component_campaign.get("component_status"),
                "receipt_path": component_campaign.get("receipt_path"),
                "fingerprint": (component_campaign.get("physical_graph") or {}).get("fingerprint"),
                "component_count": component_campaign.get("component_count"),
                "component_windows": component_campaign.get("component_windows"),
                "source_independent_execution": component_campaign.get("source_independent_execution"),
                "candidate_body_persisted": component_campaign.get("candidate_body_persisted"),
                "whole_model_capability": component_campaign.get("whole_model_capability"),
                "complete_token_runtime": component_campaign.get("complete_token_runtime"),
            },
            "router_component": {
                "status": router_graph.get("component_status"),
                "receipt_path": router_graph.get("receipt_path"),
                "fingerprint": (router_graph.get("physical_graph") or {}).get("fingerprint"),
                "component_window": router_graph.get("component_window"),
                "source_independent_execution": router_graph.get("source_independent_execution"),
                "candidate_body_persisted": router_graph.get("candidate_body_persisted"),
                "whole_model_capability": router_graph.get("whole_model_capability"),
                "complete_token_runtime": router_graph.get("complete_token_runtime"),
            },
            "router_selection": {
                "status": router_selection.get("selection_status"),
                "receipt_path": router_selection.get("receipt_path"),
                "fingerprint": (router_selection.get("physical_graph") or {}).get("fingerprint"),
                "source_independent_execution": router_selection.get("source_independent_execution"),
                "candidate_body_persisted": router_selection.get("candidate_body_persisted"),
                "native_selection_execution_observed": router_selection.get("native_selection_execution_observed"),
                "source_selection_parity_qualified": (router_selection.get("source_selection_parity") or {}).get("expert_ids_exact_match"),
                "whole_model_capability": router_selection.get("whole_model_capability"),
                "complete_token_runtime": router_selection.get("complete_token_runtime"),
            },
            "router_representation_ab": {
                "status": router_representation_ab.get("status"),
                "receipt_path": router_representation_ab.get("receipt_path"),
                "fingerprint": (router_representation_ab.get("physical_graph") or {}).get("fingerprint"),
                "candidate_bodies_persisted": router_representation_ab.get("candidate_bodies_persisted"),
                "whole_model_capability": router_representation_ab.get("whole_model_capability"),
                "complete_token_runtime": router_representation_ab.get("complete_token_runtime"),
            },
            "text_only_vision_bypass": "CONDITIONAL_AND_UNPROVEN",
            "mtp_accept_reject": "EXPLICIT_REQUIRED_EDGE",
            "fallbacks": "No fallback may be silently counted as native Flash execution.",
        },
        "capability_contract": {
            "status": "NOT_RUN",
            "required": ["same-model output parity", "multimodal/text contract", "sequence isolation", "zero hidden dense rematerialization", "fallback disclosure", "accepted-token accounting"],
            "fallback_count": None,
            "dense_parent_execution_fallback": False,
            "hidden_dense_rematerialization": False,
            "declarations_are_requirements_not_runtime_evidence": True,
        },
        "complete_token_timing": {
            "status": "NOT_MEASURED",
            "budget_receipt": token_ns.get("system_ledger"),
            "complete_token_definition": token_ns.get("target_contract", {}).get("timing_unit"),
            "accepted_tps": None,
            "complete_wall_ns_per_accepted_token": None,
        },
        "runtime_genome": {
            "status": (
                "BOUNDED_MULTI_COMPONENT_ONLY"
                if component_campaign.get("status") == "PASSED"
                else "BOUNDED_COMPONENT_ONLY"
                if graph_component.get("status") == "PASSED"
                else "NOT_COMPILED"
            ),
            "executable_sha256": None,
            "loader_sha256": None,
            "kernel_source_hashes": [],
            "kernel_binary_hashes": [],
            "graph_fingerprint": graph_component.get("graph_fingerprint"),
            "component_campaign_fingerprint": (component_campaign.get("physical_graph") or {}).get("fingerprint"),
            "router_graph_fingerprint": (router_graph.get("physical_graph") or {}).get("fingerprint"),
            "router_selection_fingerprint": (router_selection.get("physical_graph") or {}).get("fingerprint"),
            "router_representation_ab_fingerprint": (router_representation_ab.get("physical_graph") or {}).get("fingerprint"),
            "device_identity": None,
            "compiler_identity": None,
            "representation_manifest_sha256": None,
            "receipt_schema": SCHEMA,
        },
        "ebpw_budget_receipt": DEFAULT_EBPW,
        "token_ns_budget_receipt": DEFAULT_TOKEN_NS,
        "promotion_gate": evaluate_flash_promotion({
            "artifact_contract": "FLASH_NEXT_COMPLETE_MULTIMODAL",
            "complete_system_ebpw": None,
            "accepted_capability_preserving_tps": None,
            "dense_parent_execution_fallback": False,
            "hidden_dense_rematerialization": False,
        }),
        "promotion_allowed": False,
        "claim_boundary": "FLASH_NEXT_NOETIC_EXECUTABLE remains a scaffold. It records full-tensor representation, one bounded source-independent component body, bounded descriptor-loader, and bounded native-kernel evidence, not a whole-model loader, capability, complete-system EBPW, or Flash TPS claim.",
    }


def run_flash_executable_scaffold(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    science_receipt: Optional[str | os.PathLike[str]] = None,
    tensor_probe_receipt: Optional[str | os.PathLike[str]] = None,
    representation_experiment_receipt: Optional[str | os.PathLike[str]] = None,
    transform_parity_receipt: Optional[str | os.PathLike[str]] = None,
    loader_roundtrip_receipt: Optional[str | os.PathLike[str]] = None,
    kernel_parity_receipt: Optional[str | os.PathLike[str]] = None,
    graph_component_receipt: Optional[str | os.PathLike[str]] = None,
    component_campaign_receipt: Optional[str | os.PathLike[str]] = None,
    router_graph_receipt: Optional[str | os.PathLike[str]] = None,
    router_selection_receipt: Optional[str | os.PathLike[str]] = None,
    router_representation_ab_receipt: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    ebpw_emit: Optional[str | os.PathLike[str]] = None,
    token_ns_emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    science_path = Path(science_receipt).expanduser().resolve() if science_receipt else repo / "receipts" / "headless" / DEFAULT_SCIENCE
    science = _read_json(science_path)
    started = time.time()
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_EXECUTABLE
    ebpw_path = Path(ebpw_emit).expanduser() if ebpw_emit else destination.parent / DEFAULT_EBPW
    token_path = Path(token_ns_emit).expanduser() if token_ns_emit else destination.parent / DEFAULT_TOKEN_NS
    if not ebpw_path.is_absolute():
        ebpw_path = ebpw_path.resolve()
    if not token_path.is_absolute():
        token_path = token_path.resolve()
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "RUNNING",
        "repo_root": str(repo),
        "science_receipt": str(science_path),
        "generated_at": started,
    }
    try:
        if science is None:
            raise FileNotFoundError(science_path)
        source = _source_summary(science)
        source["science_receipt"] = {"path": str(science_path), "sha256": _sha256(science_path), "status": science.get("status")}
        lake = _modellake_identity(repo)
        tensor_probe = _tensor_probe_summary(repo, tensor_probe_receipt)
        representation_experiment = _representation_experiment_summary(repo, representation_experiment_receipt)
        transform_parity = _transform_parity_summary(repo, transform_parity_receipt)
        loader_roundtrip = _loader_roundtrip_summary(repo, loader_roundtrip_receipt)
        kernel_parity = _kernel_parity_summary(repo, kernel_parity_receipt)
        graph_component = _graph_component_summary(repo, graph_component_receipt)
        component_campaign = _component_campaign_summary(repo, component_campaign_receipt)
        router_graph = _router_graph_summary(repo, router_graph_receipt)
        router_selection = _router_selection_summary(repo, router_selection_receipt)
        router_representation_ab = _router_representation_ab_summary(repo, router_representation_ab_receipt)
        ebpw = _ebpw_budget(science, source, tensor_probe, representation_experiment, transform_parity, loader_roundtrip, kernel_parity)
        token_ns = _token_ns_budget(science, source)
        manifest = _executable_manifest(science, source, lake, ebpw, token_ns, tensor_probe, representation_experiment, transform_parity, loader_roundtrip, kernel_parity, graph_component, component_campaign, router_graph, router_selection, router_representation_ab)
        atomic_write_json(ebpw_path, ebpw)
        atomic_write_json(token_path, token_ns)
        manifest["ebpw_budget_receipt"] = str(ebpw_path)
        manifest["token_ns_budget_receipt"] = str(token_path)
        atomic_write_json(destination, manifest)
        result.update({
            "status": "PASSED",
            "manifest": manifest,
            "ebpw_budget": ebpw,
            "token_ns_budget": token_ns,
            "checks": {
                "source_receipt_present": True,
                "source_revision_pinned": (science.get("source_identity") or {}).get("pinned_revision") == PINNED_REVISION if isinstance(science.get("source_identity"), Mapping) else False,
                "header_only_boundary_preserved": source.get("header_audit", {}).get("body_bytes_loaded") == 0 and source.get("header_audit", {}).get("body_bytes_requested") == 0,
                "model_lake_not_mutated": lake.get("mutation_by_this_scaffold") is False,
                "bounded_probe_is_explicit": tensor_probe.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_probe_does_not_claim_whole_model": tensor_probe.get("whole_model_capability") == "NOT_TESTED" and tensor_probe.get("whole_model_runtime") == "NOT_TESTED",
                "bounded_representation_experiment_is_explicit": representation_experiment.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_representation_experiment_does_not_claim_whole_model": representation_experiment.get("whole_model_capability") == "NOT_TESTED" and representation_experiment.get("whole_model_runtime") == "NOT_TESTED",
                "bounded_representation_replications_are_explicit": all(item.get("status") in {"PASSED", "NOT_RUN"} for item in representation_experiment.get("replications") or []),
                "bounded_transform_parity_is_explicit": transform_parity.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_transform_parity_does_not_claim_whole_model": transform_parity.get("whole_model_capability") == "NOT_TESTED" and transform_parity.get("whole_model_runtime") == "NOT_TESTED",
                "bounded_transform_parity_does_not_mutate_source": transform_parity.get("body_mutated") in {None, False},
                "bounded_loader_roundtrip_is_explicit": loader_roundtrip.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_loader_roundtrip_does_not_claim_whole_model": loader_roundtrip.get("whole_model_capability") == "NOT_TESTED" and loader_roundtrip.get("whole_model_runtime") == "NOT_TESTED",
                "bounded_loader_roundtrip_does_not_mutate_source": loader_roundtrip.get("body_mutated") in {None, False},
                "bounded_kernel_parity_is_explicit": kernel_parity.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_kernel_parity_does_not_claim_whole_model": kernel_parity.get("whole_model_capability") == "NOT_TESTED" and kernel_parity.get("whole_model_runtime") == "NOT_TESTED",
                "bounded_kernel_parity_does_not_mutate_source": kernel_parity.get("body_mutated") in {None, False},
                "bounded_native_descriptor_load_is_explicit": (kernel_parity.get("native_loader") or {}).get("status") in {None, "NOT_RUN", "BOUNDED_NOETIC_DESCRIPTOR_LOAD", "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD"},
                "bounded_graph_component_is_explicit": graph_component.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_graph_component_does_not_claim_whole_model": graph_component.get("whole_model_capability") in {None, "NOT_TESTED"} and graph_component.get("complete_token_runtime") in {None, "NOT_TESTED"},
                "bounded_graph_component_body_is_scoped": (
                    graph_component.get("candidate_body_persisted") in {None, False}
                    or (
                        graph_component.get("candidate_body_persisted") is True
                        and graph_component.get("source_independent_execution") is True
                        and graph_component.get("source_backed") is False
                    )
                ),
                "bounded_graph_component_refuses_promotion": graph_component.get("promotion_allowed") is False,
                "bounded_component_campaign_is_explicit": component_campaign.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_component_campaign_does_not_claim_whole_model": component_campaign.get("whole_model_capability") in {None, "NOT_TESTED"} and component_campaign.get("complete_token_runtime") in {None, "NOT_TESTED"},
                "bounded_component_campaign_body_is_scoped": (
                    component_campaign.get("status") == "NOT_RUN"
                    or (
                        component_campaign.get("status") == "PASSED"
                        and component_campaign.get("source_independent_execution") is True
                        and component_campaign.get("candidate_body_persisted") is True
                    )
                ),
                "bounded_component_campaign_refuses_promotion": component_campaign.get("promotion_allowed") is False,
                "bounded_router_graph_is_explicit": router_graph.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_router_graph_does_not_claim_whole_model": router_graph.get("whole_model_capability") in {None, "NOT_TESTED"} and router_graph.get("complete_token_runtime") in {None, "NOT_TESTED"},
                "bounded_router_graph_body_is_scoped": (
                    router_graph.get("status") == "NOT_RUN"
                    or (
                        router_graph.get("status") == "PASSED"
                        and router_graph.get("source_independent_execution") is True
                        and router_graph.get("candidate_body_persisted") is True
                    )
                ),
                "bounded_router_graph_refuses_promotion": router_graph.get("promotion_allowed") is False,
                "bounded_router_selection_is_explicit": router_selection.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_router_selection_does_not_claim_whole_model": router_selection.get("whole_model_capability") in {None, "NOT_TESTED"} and router_selection.get("complete_token_runtime") in {None, "NOT_TESTED"},
                "bounded_router_selection_body_is_scoped": (
                    router_selection.get("status") == "NOT_RUN"
                    or (
                        router_selection.get("status") == "PASSED"
                        and router_selection.get("source_independent_execution") is True
                        and router_selection.get("candidate_body_persisted") is True
                    )
                ),
                "bounded_router_selection_is_not_mislabeled_native": router_selection.get("native_selection_execution_observed") in {None, False},
                "bounded_router_selection_refuses_promotion": router_selection.get("promotion_allowed") is False,
                "bounded_router_representation_ab_is_explicit": router_representation_ab.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_router_representation_ab_does_not_claim_whole_model": router_representation_ab.get("whole_model_capability") in {None, "NOT_TESTED"} and router_representation_ab.get("complete_token_runtime") in {None, "NOT_TESTED"},
                "bounded_router_representation_ab_does_not_persist_bodies": router_representation_ab.get("candidate_bodies_persisted") in {None, False},
                "bounded_router_representation_ab_refuses_promotion": router_representation_ab.get("promotion_allowed") is False,
                "native_loader_status_explicit": manifest.get("native_loader", {}).get("status") == "NOT_IMPLEMENTED",
                "native_kernels_status_explicit": manifest.get("native_kernels", {}).get("status") == "PLAN_ONLY",
                "complete_token_timing_not_fabricated": manifest.get("complete_token_timing", {}).get("accepted_tps") is None,
                "promotion_refused": manifest.get("promotion_allowed") is False,
                "budgets_written": destination.is_file() and ebpw_path.is_file() and token_path.is_file(),
            },
        })
        result["status"] = "PASSED" if all(result["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - keep scaffold failures durable
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    result["finished_at"] = time.time()
    result["elapsed_s"] = round(result["finished_at"] - started, 3)
    result["receipt_path"] = str(destination)
    if result.get("status") == "FAILED":
        atomic_write_json(destination, result)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--science-receipt")
    parser.add_argument("--tensor-probe-receipt")
    parser.add_argument("--representation-experiment-receipt")
    parser.add_argument("--transform-parity-receipt")
    parser.add_argument("--loader-roundtrip-receipt")
    parser.add_argument("--kernel-parity-receipt")
    parser.add_argument("--graph-component-receipt")
    parser.add_argument("--component-campaign-receipt")
    parser.add_argument("--router-graph-receipt")
    parser.add_argument("--router-selection-receipt")
    parser.add_argument("--router-representation-ab-receipt")
    parser.add_argument("--emit")
    parser.add_argument("--ebpw-emit")
    parser.add_argument("--token-ns-emit")
    args = parser.parse_args(argv)
    report = run_flash_executable_scaffold(
        repo_root=args.repo_root,
        science_receipt=args.science_receipt,
        tensor_probe_receipt=args.tensor_probe_receipt,
        representation_experiment_receipt=args.representation_experiment_receipt,
        transform_parity_receipt=args.transform_parity_receipt,
        loader_roundtrip_receipt=args.loader_roundtrip_receipt,
        kernel_parity_receipt=args.kernel_parity_receipt,
        graph_component_receipt=args.graph_component_receipt,
        component_campaign_receipt=args.component_campaign_receipt,
        router_graph_receipt=args.router_graph_receipt,
        router_selection_receipt=args.router_selection_receipt,
        router_representation_ab_receipt=args.router_representation_ab_receipt,
        emit=args.emit,
        ebpw_emit=args.ebpw_emit,
        token_ns_emit=args.token_ns_emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_KERNEL_PARITY", "DEFAULT_LOADER_ROUNDTRIP", "DEFAULT_REPRESENTATION_EXPERIMENT", "DEFAULT_REPRESENTATION_REPLICATION", "DEFAULT_ROUTER_REPRESENTATION_AB", "DEFAULT_ROUTER_SELECTION", "DEFAULT_TENSOR_PROBE", "DEFAULT_TRANSFORM_PARITY", "EBPW_SCHEMA", "SCHEMA", "TOKEN_NS_SCHEMA", "main", "run_flash_executable_scaffold"]


if __name__ == "__main__":
    raise SystemExit(main())
