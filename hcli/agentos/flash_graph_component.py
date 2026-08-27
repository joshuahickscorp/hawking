"""Compose the validated Flash Noetic descriptor and native kernel as a graph.

This is the first executable graph boundary after descriptor and kernel
parity.  It compiles one routed-expert component from receipts and can consume
the persisted bounded component body; it never loads the whole model or claims
complete-token capability, EBPW, or Flash TPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json
from hcli.physical_graph import compile_physical_graph


SCHEMA = "hcli.agentos.flash_noetic_graph_component.v1"
PHYSICAL_GRAPH_SCHEMA = "hcli.physical_graph.v1"
DESCRIPTOR_SCHEMA = "hcli.noetic.representation_descriptor.v1"
DEFAULT_LOADER = "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"
DEFAULT_KERNEL = "FLASH_NOETIC_Q4_KERNEL_PARITY.json"
DEFAULT_BODY_KERNEL = "FLASH_NOETIC_Q4_BODY_KERNEL_PARITY.json"
DEFAULT_TRANSFORM = "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"
DEFAULT_EMIT = "FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json"
DERIVED = "[D]"
VERIFIED = "[V]"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _receipt_ref(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema": value.get("schema"),
        "status": value.get("status"),
        "label": VERIFIED,
    }


def _identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = value.get("model_lake_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    return {
        "repo": value.get("repo") or manifest.get("repo"),
        "revision": value.get("pinned_revision") or manifest.get("revision"),
        "resolved_sha": manifest.get("resolved_sha") or value.get("pinned_revision"),
        "root": value.get("root") or manifest.get("path"),
        "manifest_path": manifest.get("path"),
        "manifest_sha256": manifest.get("sha256") or manifest.get("manifest_sha256"),
    }


def _validate(
    loader_path: Path,
    loader: Mapping[str, Any],
    kernel_path: Path,
    kernel: Mapping[str, Any],
    transform_path: Path,
    transform: Mapping[str, Any],
) -> Dict[str, Any]:
    errors: list[str] = []
    descriptor = loader.get("representation_descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    source = descriptor.get("source_tensor")
    source = source if isinstance(source, Mapping) else {}
    kernel_source = kernel.get("source_tensor")
    kernel_source = kernel_source if isinstance(kernel_source, Mapping) else {}
    representation = kernel.get("noetic_representation")
    representation = representation if isinstance(representation, Mapping) else {}
    native_loader = kernel.get("native_loader")
    native_loader = native_loader if isinstance(native_loader, Mapping) else {}
    native_kernel = kernel.get("native_kernel")
    native_kernel = native_kernel if isinstance(native_kernel, Mapping) else {}
    parity = kernel.get("parity")
    parity = parity if isinstance(parity, Mapping) else {}
    transform_candidates = transform.get("candidates")
    transform_candidates = transform_candidates if isinstance(transform_candidates, Mapping) else {}
    candidate_id = str(loader.get("candidate_id") or descriptor.get("candidate_id") or "")

    if loader.get("status") != "PASSED":
        errors.append("loader receipt is not PASSED")
    if loader.get("schema") != "hcli.agentos.flash_loader_roundtrip.v1":
        errors.append("loader receipt schema is not the bounded loader schema")
    if descriptor.get("schema") != DESCRIPTOR_SCHEMA:
        errors.append("representation descriptor schema is not the Noetic descriptor schema")
    if not candidate_id:
        errors.append("descriptor has no candidate_id")
    if descriptor.get("candidate_id") != candidate_id:
        errors.append("loader candidate_id and descriptor candidate_id disagree")
    if transform.get("status") != "PASSED":
        errors.append("full-tensor transform receipt is not PASSED")
    if transform.get("schema") != "hcli.agentos.flash_transform_parity.v1":
        errors.append("transform receipt schema is not the transform parity schema")
    transform_candidate = transform_candidates.get(candidate_id)
    if not isinstance(transform_candidate, Mapping):
        errors.append(f"transform receipt has no candidate {candidate_id!r}")
    else:
        reference = descriptor.get("full_transform_reference")
        reference = reference if isinstance(reference, Mapping) else {}
        for key in ("candidate_bytes", "candidate_sha256"):
            if reference.get(key) != transform_candidate.get(key):
                errors.append(f"descriptor and transform disagree on {key}")
    if kernel.get("status") != "PASSED":
        errors.append("native kernel receipt is not PASSED")
    if kernel.get("schema") != "hawking.flash_noetic_q4_kernel_parity.v1":
        errors.append("kernel receipt schema is not the Flash Noetic kernel schema")
    if native_loader.get("status") not in {
        "BOUNDED_NOETIC_DESCRIPTOR_LOAD",
        "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD",
    }:
        errors.append("kernel did not load the bounded Noetic descriptor")
    if native_loader.get("candidate_id") != candidate_id:
        errors.append("kernel loader and descriptor candidate_id disagree")
    if representation.get("candidate_id") != candidate_id:
        errors.append("kernel representation and descriptor candidate_id disagree")
    kernel_descriptor = kernel.get("noetic_descriptor")
    kernel_descriptor = kernel_descriptor if isinstance(kernel_descriptor, Mapping) else {}
    if kernel_descriptor.get("schema") != DESCRIPTOR_SCHEMA:
        errors.append("kernel receipt does not identify the Noetic descriptor schema")
    if native_kernel.get("kernel_registered") is not True:
        errors.append("native kernel was not registered")
    if parity.get("within_tolerance") is not True:
        errors.append("native kernel parity is not within tolerance")
    if loader.get("body_mutated") is not False or kernel.get("body_mutated") is not False:
        errors.append("source mutation guard is not false in both receipts")
    if loader.get("model_loaded") is not False or kernel.get("model_loaded") is not False:
        errors.append("whole-model load guard is not false in both receipts")

    identities = {
        "loader": _identity(loader),
        "kernel": _identity(kernel),
        "transform": _identity(transform),
    }
    for role, identity in identities.items():
        if identity["repo"] != REPO_ID:
            errors.append(f"{role} receipt is not for {REPO_ID}")
        if identity["revision"] != PINNED_REVISION or identity["resolved_sha"] != PINNED_REVISION:
            errors.append(f"{role} receipt is not for the pinned revision")
    if len({identity["root"] for identity in identities.values() if identity["root"]}) > 1:
        errors.append("source roots disagree across receipts")
    if source.get("tensor_name") != kernel_source.get("tensor_name"):
        errors.append("descriptor and kernel source tensor names disagree")
    if source.get("shape") != kernel_source.get("shape"):
        errors.append("descriptor and kernel source tensor shapes disagree")
    if source.get("layout") != "row-major [expert, row, column]":
        errors.append("descriptor layout is not the validated routed-expert layout")
    if source.get("group_size") != 64:
        errors.append("descriptor group size is not 64")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_id": candidate_id,
        "descriptor": descriptor,
        "source_tensor": source,
        "kernel_source": kernel_source,
        "representation": representation,
        "native_loader": native_loader,
        "native_kernel": native_kernel,
        "parity": parity,
        "identities": identities,
        "transform_candidate": transform_candidate,
        "component_window": {
            "expert_index": int(kernel_source.get("selected_expert", 0) or 0),
            "row_start": int(kernel_source.get("selected_row_start", 0) or 0),
            "row_count": int(kernel_source.get("selected_row_count", 128) or 128),
        },
        "source_independent_execution": native_loader.get("source_independent_execution") is True,
        "candidate_body_persisted": native_loader.get("candidate_body_persisted") is True,
        "loader_ref": _receipt_ref(loader_path, loader),
        "kernel_ref": _receipt_ref(kernel_path, kernel),
        "transform_ref": _receipt_ref(transform_path, transform),
    }


def _component_graph(validation: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_id = str(validation["candidate_id"])
    descriptor = validation["descriptor"]
    source = validation["source_tensor"]
    representation = validation["representation"]
    kernel = validation["native_kernel"]
    window = validation["component_window"]
    expert_index = window["expert_index"]
    row_start = window["row_start"]
    row_count = window["row_count"]
    source_independent = bool(validation.get("source_independent_execution"))
    nodes = [
        {
            "id": "source_block_reference",
            "stage": "SourceSpecimen",
            "kind": "verified_source_reference",
            "tensor_name": source.get("tensor_name"),
            "expert": expert_index,
            "row_start": row_start,
            "row_count": row_count,
            "source_block_bytes": validation["kernel_source"].get("selected_block_bytes"),
            "source_backed": True,
            "execution_input": False,
        },
    ]
    if source_independent:
        nodes.append({
            "id": "noetic_component_body_load",
            "stage": "NoeticCompiler",
            "kind": "source_independent_component_body",
            "candidate_id": candidate_id,
            "expert_index": expert_index,
            "row_start": row_start,
            "row_count": row_count,
            "body_persisted": True,
            "execution_input": True,
        })
    nodes.extend([
        {
            "id": "noetic_descriptor_load",
            "stage": "NoeticCompiler",
            "kind": "serialized_representation_descriptor",
            "schema": descriptor.get("schema"),
            "candidate_id": candidate_id,
            "expert_index": expert_index,
            "row_start": row_start,
            "row_count": row_count,
            "descriptor_sha256": validation["native_loader"].get("descriptor_sha256"),
            "loader_status": validation["native_loader"].get("status"),
            "execution_input": True,
        },
        {
            "id": "routed_expert_q4_g64_matvec",
            "stage": "HawkingAccelerator",
            "kind": "native_kernel_dispatch",
            "kernel": kernel.get("kernel"),
            "expert_index": expert_index,
            "row_start": row_start,
            "row_count": row_count,
            "dispatches_per_sample": kernel.get("dispatches_per_sample"),
            "registered": kernel.get("kernel_registered"),
            "parity_within_tolerance": validation["parity"].get("within_tolerance"),
            "source_independent_execution": source_independent,
        },
        {
            "id": "bounded_component_output",
            "stage": "NoeticExecutableCandidate",
            "kind": "source_block_matvec_output",
            "expert_index": expert_index,
            "row_start": row_start,
            "row_count": row_count,
            "whole_model": False,
            "complete_token": False,
        },
    ])
    edges = [
        {"from": "source_block_reference", "to": "bounded_component_output", "kind": "parity_reference_only"},
        {"from": "noetic_descriptor_load", "to": "routed_expert_q4_g64_matvec", "kind": "descriptor_to_kernel"},
        {"from": "routed_expert_q4_g64_matvec", "to": "bounded_component_output", "kind": "kernel_to_output"},
    ]
    if source_independent:
        edges.insert(1, {"from": "noetic_component_body_load", "to": "noetic_descriptor_load", "kind": "body_to_descriptor"})
    physical = compile_physical_graph(
        {
            "model_id": REPO_ID,
            "architecture": {
                "component": "routed_expert",
                "candidate_id": candidate_id,
                "tensor_layout": source.get("layout"),
            },
            "organs": [
                {
                    "id": "routed_expert_component",
                    "present": True,
                    "tensor_count": 1,
                    "confidence": "[V]/[D]",
                }
            ],
            "evidence": [validation["loader_ref"], validation["kernel_ref"], validation["transform_ref"]],
        },
        provider={"provider": "apple-metal", "kernel": kernel.get("kernel")},
        devices=("apple_metal",),
    )
    physical["component_scope"] = f"one routed-expert source block expert={expert_index} rows={row_start}:{row_start + row_count}; not complete Flash execution"
    physical["computation"] = nodes
    physical["dependencies"] = edges
    physical["representation"] = {
        "descriptor_schema": descriptor.get("schema"),
        "candidate_id": candidate_id,
        "component_window": window,
        "effective_bits_per_value": representation.get("effective_bits_per_value"),
        "candidate_bytes_full_tensor": validation["transform_candidate"].get("candidate_bytes"),
        "candidate_body_persisted": validation["candidate_body_persisted"],
        "source_independent_execution": source_independent,
        "dense_rematerialization": "forbidden",
    }
    physical["qualification"] = "BOUNDED_COMPONENT_ONLY"
    physical["graph_execution_observed"] = False
    physical["native_kernel_execution_observed"] = True
    physical["fingerprint"] = _json_hash({key: value for key, value in physical.items() if key not in {"generated_at", "fingerprint"}})
    return physical


def _compile_report(
    loader_path: Path,
    loader: Mapping[str, Any],
    kernel_path: Path,
    kernel: Mapping[str, Any],
    transform_path: Path,
    transform: Mapping[str, Any],
) -> Dict[str, Any]:
    validation = _validate(loader_path, loader, kernel_path, kernel, transform_path, transform)
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "semantic_type": "NoeticExecutableCandidate",
        "compiler_stage": "PhysicalGraphCompiler",
        "status": "PASSED" if validation["valid"] else "FAILED",
        "component_status": "BOUNDED_COMPONENT_COMPILED" if validation["valid"] else "NOT_COMPILED",
        "qualification": False,
        "NOT_FOR_PROMOTION": True,
        "promotion_allowed": False,
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "candidate_id": validation["candidate_id"],
        "component_window": validation["component_window"],
        "source_identity": validation["identities"].get("kernel"),
        "source_backed": not validation["source_independent_execution"],
        "source_independent_execution": validation["source_independent_execution"],
        "candidate_body_persisted": validation["candidate_body_persisted"],
        "whole_model_capability": "NOT_TESTED",
        "complete_token_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
        "inputs": {
            "transform": validation["transform_ref"],
            "loader": validation["loader_ref"],
            "kernel": validation["kernel_ref"],
        },
        "validation": {
            "errors": validation["errors"],
            "descriptor_schema": validation["descriptor"].get("schema"),
            "source_tensor": validation["source_tensor"],
            "native_loader": validation["native_loader"],
            "native_kernel": validation["native_kernel"],
            "parity": validation["parity"],
            "full_transform_candidate": validation["transform_candidate"],
        },
        "noetic_ir": {
            "schema": "hcli.noetic.ir.v1",
            "semantic_type": "NoeticIR",
            "candidate_id": validation["candidate_id"],
            "operations": [
                "retain_verified_source_block_as_parity_reference",
                *(["load_persisted_source_independent_component_body"] if validation["source_independent_execution"] else []),
                "load_serialized_representation_descriptor",
                "dispatch_qwen_uniform_q4_group64_matvec",
            ],
            "source_independent": validation["source_independent_execution"],
            "complete_model": False,
        },
        "physical_graph": None,
        "claim_boundary": "Bounded routed-expert graph component compiled from validated Noetic descriptor, persisted component body, and native kernel receipts; no complete-model loader, capability, complete-token timing, EBPW, or Flash TPS claim.",
        "next_action": "compose the bounded source-independent component across the routed-expert organ, then extend independently-owned representation/loader contracts across the remaining Flash organs",
    }
    if validation["valid"]:
        report["physical_graph"] = _component_graph(validation)
        report["graph_fingerprint"] = report["physical_graph"].get("fingerprint")
    else:
        report["graph_fingerprint"] = None
    return report


def run_flash_graph_component(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    loader_receipt: Optional[str | os.PathLike[str]] = None,
    kernel_receipt: Optional[str | os.PathLike[str]] = None,
    transform_receipt: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    headless = repo / "receipts" / "headless"
    loader_path = Path(loader_receipt).expanduser().resolve() if loader_receipt else headless / DEFAULT_LOADER
    if kernel_receipt:
        kernel_path = Path(kernel_receipt).expanduser().resolve()
    else:
        body_kernel = headless / DEFAULT_BODY_KERNEL
        kernel_path = body_kernel if body_kernel.is_file() else headless / DEFAULT_KERNEL
    transform_path = Path(transform_receipt).expanduser().resolve() if transform_receipt else headless / DEFAULT_TRANSFORM
    destination = Path(emit).expanduser().resolve() if emit else headless / DEFAULT_EMIT
    started = time.time()
    loader = _read_json(loader_path)
    kernel = _read_json(kernel_path)
    transform = _read_json(transform_path)
    if not isinstance(loader, Mapping) or not isinstance(kernel, Mapping) or not isinstance(transform, Mapping):
        report: Dict[str, Any] = {
            "schema": SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "PhysicalGraphCompiler",
            "status": "FAILED",
            "component_status": "NOT_COMPILED",
            "qualification": False,
            "NOT_FOR_PROMOTION": True,
            "promotion_allowed": False,
            "error": "PASSED transform, loader, and kernel receipts are required",
        }
    else:
        report = _compile_report(loader_path, loader, kernel_path, kernel, transform_path, transform)
    report.update({
        "repo_root": str(repo),
        "generated_at": started,
        "finished_at": time.time(),
        "elapsed_s": round(time.time() - started, 3),
        "receipt_path": str(destination),
    })
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--loader-receipt")
    parser.add_argument("--kernel-receipt")
    parser.add_argument("--transform-receipt")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_graph_component(
        repo_root=args.repo_root,
        loader_receipt=args.loader_receipt,
        kernel_receipt=args.kernel_receipt,
        transform_receipt=args.transform_receipt,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_EMIT", "SCHEMA", "run_flash_graph_component", "main"]
